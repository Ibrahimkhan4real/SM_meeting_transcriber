"""Gemini analysis of a transcript.

Two design points worth knowing:

* Retryability is decided from the exception *type* and its numeric HTTP code,
  never by searching the message text. Substring matching used to misclassify
  permanent failures whose messages happened to contain a retryable number —
  "token count 1500 exceeds the maximum" contains "500".
* The response is schema-constrained JSON. Parsing Markdown headings out of
  free-form prose was the most fragile part of the pipeline; a heading the
  model reworded silently lost that section.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field

import httpx
from google.genai import errors as genai_errors

from .prompts import (MARKDOWN_PROMPT, SECTION_MAP, SECTIONS,
                      SYSTEM_INSTRUCTION, TRANSCRIPT_PROMPT, MeetingAnalysis)

log = logging.getLogger(__name__)

ABBREVIATIONS = ("e.g", "i.e", "etc", "vs", "cf", "Dr", "Prof", "Mr", "Ms",
                 "Mrs", "St", "approx")

#: HTTP codes worth retrying. 429 is rate limiting; 5xx are server-side.
RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})

#: Default per-request timeout (TRANSCRIBE_GEMINI_TIMEOUT overrides it). The
#: client default is none at all: after a 503, one stalled request hung a run
#: for 27 minutes without a word.
REQUEST_TIMEOUT_MS = 300_000

#: Longest wait honoured when a 429 says how long to back off (RetryInfo).
MAX_SERVER_DELAY = 60.0

#: Finish reasons that mean the model stopped without usable output.
BLOCKING_FINISH_REASONS = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT",
                           "BLOCKLIST", "SPII"}


class AnalysisError(RuntimeError):
    """The transcript could not be analysed."""


class TranscriptTooLong(AnalysisError):
    """The transcript exceeded the model's input or output budget."""


@dataclass
class Analysis:
    model: str
    raw: str
    executive_summary: str = ""
    meeting_notes: str = ""
    meeting_points: list[str] = field(default_factory=list)
    action_points: list[str] = field(default_factory=list)
    structured: bool = True

    def as_markdown(self) -> dict[str, str]:
        """Render to `{section key: markdown}` for writing to disk."""
        out: dict[str, str] = {}
        for key, heading, attr in SECTIONS:
            value = getattr(self, attr)
            if isinstance(value, list):
                if not value:
                    continue
                body = "\n".join(f"- {item}" for item in value)
            else:
                body = (value or "").strip()
                if not body:
                    continue
            out[key] = f"## {heading}\n\n{body}\n"
        return out


def is_retryable(exc: BaseException) -> bool:
    """Whether a failed API call is worth trying again.

    Decided from the exception type and its numeric code — never from the
    message text.
    """
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        try:
            return int(code) in RETRYABLE_CODES
        except (TypeError, ValueError):
            return isinstance(exc, genai_errors.ServerError)
    # Transport-level faults: the request may never have reached the server.
    # google-genai raises httpx's own types (ConnectError, ReadError, ...),
    # which are not the builtin ConnectionError, for a Wi-Fi blip or a reset.
    return isinstance(exc, (TimeoutError, ConnectionError, httpx.TimeoutException,
                            httpx.NetworkError, httpx.RemoteProtocolError))


def _server_delay(exc: BaseException) -> float | None:
    """Seconds Gemini asked us to wait (a 429's RetryInfo "37s"), if any."""
    try:
        for item in exc.details["error"]["details"]:
            if str(item.get("@type", "")).endswith("RetryInfo"):
                seconds = float(str(item["retryDelay"]).rstrip("s"))
                # "nans" or "-5s" would otherwise reach time.sleep() and end
                # the retry loop with a ValueError.
                return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return None


def parse_sections(text: str) -> dict[str, str]:
    """Split Markdown into named sections (fallback path, and older analyses)."""
    fenced = re.match(r"\s*```(?:markdown)?\s*\n(.*)\n```\s*\Z", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    parts = re.split(r"^#{2,3} (.+?)\s*$", text, flags=re.MULTILINE)
    sections: dict[str, str] = {}
    i = 1
    while i < len(parts) - 1:
        sections[parts[i].strip()] = parts[i + 1].strip()
        i += 2
    return sections


def first_sentence(text: str) -> str:
    """First sentence, tolerant of decimals ('0.03') and abbreviations ('e.g.')."""
    text = " ".join(text.split())
    if not text:
        return "[No summary]"
    for match in re.finditer(r"[.!?](?=\s|$)", text):
        candidate = text[:match.end()]
        tail = candidate[:-1].rsplit(" ", 1)[-1].rstrip(".")
        if tail in ABBREVIATIONS:
            continue
        if len(candidate) < 25:
            continue
        return candidate
    return (text[:200].rsplit(" ", 1)[0] + "…") if len(text) > 200 else text


def _finish_reason(response) -> str | None:
    try:
        reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None
    return getattr(reason, "name", str(reason)) if reason is not None else None


def _block_reason(response) -> str | None:
    """Why Gemini refused the prompt itself; it then returns no candidates at all."""
    reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    return getattr(reason, "name", str(reason)) if reason is not None else None


def _one_line(exc: Exception) -> str:
    """A schema mismatch otherwise prints a dozen lines of pydantic detail."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            fields = sorted({".".join(map(str, e["loc"])) for e in errors() if e.get("loc")})
            if fields:
                return f"schema mismatch in {', '.join(fields)}"
        except Exception:                          # noqa: BLE001
            pass
    return " ".join(str(exc).split())


class Analyser:
    """Calls Gemini, retrying and falling back through a chain of models."""

    def __init__(self, models, api_key: str | None = None, sleep=time.sleep,
                 max_attempts: int = 3, client=None,
                 timeout: float = REQUEST_TIMEOUT_MS / 1000):
        self.models = tuple(models)
        self.max_attempts = max_attempts
        self._sleep = sleep
        if client is not None:
            self.client = client
        else:
            from google import genai
            options = {"timeout": max(1, round(timeout * 1000))}     # milliseconds
            self.client = (genai.Client(api_key=api_key, http_options=options) if api_key
                           else genai.Client(http_options=options))

    # -- request ---------------------------------------------------------------

    def _call(self, model: str, transcript: str, structured: bool):
        if structured:
            config = {
                "temperature": 0.3,
                "system_instruction": SYSTEM_INSTRUCTION,
                "response_mime_type": "application/json",
                "response_schema": MeetingAnalysis,
            }
            contents = TRANSCRIPT_PROMPT.format(transcript=transcript)
        else:
            config = {"temperature": 0.3}
            contents = MARKDOWN_PROMPT.format(transcript=transcript)
        return self.client.models.generate_content(
            model=model, contents=contents, config=config)

    def _generate(self, transcript: str, structured: bool):
        """Try each model in turn, with backoff on retryable failures."""
        last: BaseException | None = None
        for model in self.models:
            delay = 4.0
            for attempt in range(self.max_attempts):
                try:
                    return self._call(model, transcript, structured), model
                except BaseException as exc:            # noqa: BLE001
                    last = exc
                    # A retired or renamed model says nothing about the rest of
                    # the chain; stopping here made every fallback useless.
                    if (isinstance(exc, genai_errors.ClientError)
                            and getattr(exc, "code", None) == 404):
                        log.warning("%s: %s; trying the next model.", model, _describe(exc))
                        break
                    if not is_retryable(exc):
                        raise
                    if attempt < self.max_attempts - 1:
                        asked = _server_delay(exc)
                        wait = delay if asked is None else min(max(asked, delay),
                                                               MAX_SERVER_DELAY)
                        log.warning("%s: %s; retrying in %.0fs (attempt %d/%d)",
                                    model, _describe(exc), wait,
                                    attempt + 2, self.max_attempts)
                        self._sleep(wait)
                        delay *= 2
            else:
                if model != self.models[-1]:
                    log.warning("%s exhausted; falling back to the next model.", model)
        raise AnalysisError(f"all models failed; last error: {_describe(last)}") from last

    # -- public ----------------------------------------------------------------

    def analyse(self, transcript: str) -> Analysis:
        try:
            return self._analyse(transcript, structured=True)
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code == 400 and "schema" in str(exc).lower():
                log.warning("Model rejected the response schema; "
                            "falling back to Markdown parsing.")
                return self._analyse(transcript, structured=False)
            raise

    def _analyse(self, transcript: str, structured: bool) -> Analysis:
        response, model = self._generate(transcript, structured)

        # A blocked prompt has no candidates, so it used to surface as
        # "malformed JSON: Expecting value", hiding the real cause.
        blocked = _block_reason(response)
        if blocked:
            raise AnalysisError(
                f"{model} refused the transcript (block reason: {blocked}). "
                "The transcript may have tripped a content filter.")
        reason = _finish_reason(response)
        if reason in BLOCKING_FINISH_REASONS:
            raise AnalysisError(
                f"{model} stopped without output (finish reason: {reason}). "
                "The transcript may have tripped a content filter.")
        if reason == "MAX_TOKENS":
            raise TranscriptTooLong(
                f"{model} hit its output limit on this transcript. "
                "Try a model with a larger output budget, or split the recording.")

        if structured:
            parsed = getattr(response, "parsed", None)
            if parsed is None:
                text = getattr(response, "text", "") or ""
                if not text.strip():
                    raise AnalysisError(f"{model} returned an empty response "
                                        f"(finish reason: {reason or 'none'})")
                try:
                    parsed = MeetingAnalysis.model_validate(json.loads(text))
                except Exception as exc:               # noqa: BLE001
                    raise AnalysisError(
                        f"{model} returned malformed JSON: {_one_line(exc)}") from exc
            result = Analysis(
                model=model,
                raw=parsed.model_dump_json(indent=2),
                executive_summary=parsed.executive_summary,
                meeting_notes=parsed.meeting_notes,
                meeting_points=list(parsed.meeting_points),
                action_points=list(parsed.action_points),
                structured=True,
            )
            # All four fields blank is no analysis at all. Saving it would mark
            # the transcript as analysed, so later runs would never retry it.
            if not result.as_markdown():
                raise AnalysisError(f"{model} returned an analysis with every "
                                    "section empty; nothing was saved")
            return result

        text = getattr(response, "text", "") or ""
        if not text.strip():
            raise AnalysisError(f"{model} returned an empty response")
        sections = parse_sections(text)
        unknown = [h for h in sections if h not in SECTION_MAP]
        if unknown:
            log.warning("Unrecognised section(s) %s kept only in the raw response.",
                        ", ".join(repr(u) for u in unknown))

        def bullets(value: str) -> list[str]:
            return [re.sub(r"^[-*]\s+", "", line).strip()
                    for line in value.splitlines() if line.strip().startswith(("-", "*"))]

        return Analysis(
            model=model,
            raw=text,
            executive_summary=sections.get("Executive Summary", ""),
            meeting_notes=sections.get("Meeting Notes", ""),
            meeting_points=bullets(sections.get("Meeting Points", "")),
            action_points=bullets(sections.get("My Action Points", "")),
            structured=False,
        )


def _describe(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown error"
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    if code is not None:
        return f"{type(exc).__name__} {code} {status or ''}".strip()
    return f"{type(exc).__name__}: {exc}"
