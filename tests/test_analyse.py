"""Error classification, response handling, and rendering."""

import json

import httpx

import pytest
from google.genai import errors as genai_errors

from transcribe.analyse import (Analyser, Analysis, AnalysisError,
                                TranscriptTooLong, first_sentence,
                                is_retryable, parse_sections)
from transcribe.prompts import MeetingAnalysis, SECTION_MAP


def api_error(code, message="boom"):
    """Build the exception subclass the library really raises for this code."""
    payload = {"error": {"code": code, "message": message}}
    cls = genai_errors.ClientError if 400 <= code < 500 else genai_errors.ServerError
    return cls(code, payload)


# --- retry classification (the Tier 3 fix) ------------------------------------

@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_retryable_codes(code):
    assert is_retryable(api_error(code))


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_codes_are_not_retried(code):
    assert not is_retryable(api_error(code))


@pytest.mark.parametrize("message", [
    "The input token count 1500 exceeds the maximum",
    "Request payload size 8504 bytes is too large",
    "Model gemini-3.5-flash-002 is not accessible",
])
def test_digits_in_message_do_not_trigger_a_retry(message):
    """The exact defect the old substring matcher had."""
    assert not is_retryable(api_error(400, message))


def test_transport_faults_are_retryable():
    assert is_retryable(TimeoutError("timed out"))
    assert is_retryable(ConnectionError("reset"))


def test_unrelated_exception_is_not_retryable():
    assert not is_retryable(ValueError("nope"))


# --- fakes --------------------------------------------------------------------

class FakeResponse:
    def __init__(self, parsed=None, text="", finish="STOP"):
        self.parsed = parsed
        self.text = text
        self.candidates = [type("C", (), {"finish_reason": type("F", (), {"name": finish})()})()]


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def analysis_obj():
    return MeetingAnalysis(executive_summary="Summary sentence here, long enough.",
                           meeting_notes="Notes.",
                           meeting_points=["p1", "p2"],
                           action_points=["a1"])


def make(script, models=("m1", "m2")):
    return Analyser(models, client=FakeClient(script), sleep=lambda _s: None)


# --- retry / fallback behaviour ------------------------------------------------

def test_retries_same_model_then_succeeds():
    a = make([api_error(503), FakeResponse(parsed=analysis_obj())])
    result = a.analyse("t")
    assert result.model == "m1"
    assert a.client.models.calls == ["m1", "m1"]


def test_falls_back_to_next_model_after_exhausting_the_first():
    a = make([api_error(503)] * 3 + [FakeResponse(parsed=analysis_obj())])
    result = a.analyse("t")
    assert result.model == "m2"
    assert a.client.models.calls == ["m1", "m1", "m1", "m2"]


def test_permanent_error_fails_immediately_without_retrying():
    a = make([api_error(403, "permission denied")])
    with pytest.raises(genai_errors.APIError):
        a.analyse("t")
    assert a.client.models.calls == ["m1"], "must not retry a 403"


def test_all_models_failing_raises_analysis_error():
    a = make([api_error(503)] * 6)
    with pytest.raises(AnalysisError):
        a.analyse("t")


# --- finish reasons ------------------------------------------------------------

def test_safety_block_is_reported_not_silently_empty():
    a = make([FakeResponse(parsed=None, text="", finish="SAFETY")])
    with pytest.raises(AnalysisError, match="content filter"):
        a.analyse("t")


def test_max_tokens_raises_a_specific_error():
    a = make([FakeResponse(parsed=None, text="", finish="MAX_TOKENS")])
    with pytest.raises(TranscriptTooLong):
        a.analyse("t")


def test_malformed_json_is_reported():
    a = make([FakeResponse(parsed=None, text="{not json", finish="STOP")])
    with pytest.raises(AnalysisError, match="malformed JSON"):
        a.analyse("t")


def test_json_text_without_parsed_is_recovered():
    payload = json.dumps(analysis_obj().model_dump())
    a = make([FakeResponse(parsed=None, text=payload, finish="STOP")])
    assert a.analyse("t").action_points == ["a1"]


# --- rendering -----------------------------------------------------------------

def test_as_markdown_produces_the_four_files():
    result = Analysis(model="m", raw="{}", executive_summary="S.", meeting_notes="N.",
                      meeting_points=["p1"], action_points=["a1"])
    files = result.as_markdown()
    assert set(files) == {"executive_summary", "meeting_notes",
                          "meeting_points", "action_points"}
    assert files["meeting_points"].startswith("## Meeting Points")
    assert "- p1" in files["meeting_points"]


def test_as_markdown_omits_empty_sections():
    result = Analysis(model="m", raw="{}", executive_summary="S.", meeting_notes="",
                      meeting_points=[], action_points=["a1"])
    assert set(result.as_markdown()) == {"executive_summary", "action_points"}


# --- markdown fallback ---------------------------------------------------------

GOOD_MD = ("## Executive Summary\n\nBody one.\n\n## Meeting Notes\n\nBody two.\n\n"
           "## Meeting Points\n\n- a\n- b\n\n## My Action Points\n\n- c\n")


def test_markdown_fallback_parses_bullets():
    a = make([FakeResponse(parsed=None, text=GOOD_MD, finish="STOP")])
    result = a._analyse("t", structured=False)
    assert result.meeting_points == ["a", "b"] and result.action_points == ["c"]
    assert result.structured is False


def test_schema_rejection_triggers_markdown_fallback():
    a = make([api_error(400, "Invalid JSON payload: unknown name response_schema"),
              FakeResponse(parsed=None, text=GOOD_MD, finish="STOP")])
    result = a.analyse("t")
    assert result.structured is False and result.meeting_points == ["a", "b"]


def test_parse_sections_handles_code_fence():
    assert set(parse_sections("```markdown\n" + GOOD_MD + "```")) == set(SECTION_MAP)


# --- summary extraction --------------------------------------------------------

@pytest.mark.parametrize("text,start", [
    ("The noise added 0.03 to the variance here. Then more.", "The noise added 0.03"),
    ("They suggested kernels, e.g. tri-cube, as options. Next.", "They suggested kernels, e.g."),
    ("Dr. Smith reviewed the instability analysis today. More.", "Dr. Smith reviewed"),
])
def test_first_sentence(text, start):
    assert first_sentence(text).startswith(start)


def test_first_sentence_empty():
    assert first_sentence("") == "[No summary]"


def test_stalled_request_times_out_and_is_retryable(monkeypatch):
    """httpx's default here is no timeout: one stalled call hung a run for 27 minutes."""
    import httpx
    from google import genai
    from transcribe import analyse
    seen = {}
    monkeypatch.setattr(genai, "Client", lambda **kwargs: seen.update(kwargs))
    Analyser(["m"], api_key="test-key")
    assert 0 < seen["http_options"]["timeout"] <= analyse.REQUEST_TIMEOUT_MS
    assert is_retryable(httpx.ReadTimeout("timed out"))


# --- blocked, empty and mismatched responses (real SDK response objects) -------

def sdk_response(**fields):
    from google.genai import types
    return types.GenerateContentResponse(**fields)


def test_blocked_prompt_is_reported_as_blocked_not_malformed_json():
    """A blocked prompt has no candidates; it used to read "malformed JSON"."""
    from google.genai import types
    a = make([sdk_response(prompt_feedback=types.GenerateContentResponsePromptFeedback(
        block_reason=types.BlockedReason.PROHIBITED_CONTENT))])
    with pytest.raises(AnalysisError, match="PROHIBITED_CONTENT") as info:
        a.analyse("t")
    assert "malformed JSON" not in str(info.value)


@pytest.mark.parametrize("finish", ["OTHER", "LANGUAGE"])
def test_empty_response_names_the_finish_reason(finish):
    from google.genai import types
    a = make([sdk_response(candidates=[types.Candidate(finish_reason=finish)])])
    with pytest.raises(AnalysisError, match=f"empty response.*{finish}"):
        a.analyse("t")


def test_schema_mismatch_is_reported_on_one_line():
    a = make([FakeResponse(parsed=None, text='{"executive_summary": "x"}', finish="STOP")])
    with pytest.raises(AnalysisError, match="malformed JSON") as info:
        a.analyse("t")
    assert "\n" not in str(info.value) and "action_points" in str(info.value)


def test_all_empty_structured_analysis_is_a_failure_not_a_saved_result():
    """Saving it marked the transcript analysed, so it was never retried."""
    empty = MeetingAnalysis(executive_summary="", meeting_notes=" ",
                            meeting_points=[], action_points=[])
    a = make([FakeResponse(parsed=empty)])
    with pytest.raises(AnalysisError, match="every section empty"):
        a.analyse("t")


def test_partly_empty_structured_analysis_is_still_saved():
    partial = MeetingAnalysis(executive_summary="Summary sentence here, long enough.",
                              meeting_notes="", meeting_points=[], action_points=[])
    assert make([FakeResponse(parsed=partial)]).analyse("t").executive_summary


# --- timeout ---------------------------------------------------------------------

def test_request_timeout_is_configurable(monkeypatch):
    from google import genai
    seen = {}
    monkeypatch.setattr(genai, "Client", lambda **kwargs: seen.update(kwargs))
    Analyser(["m"], api_key="test-key", timeout=45)
    assert seen["http_options"]["timeout"] == 45_000


def test_stalled_gemini_costs_three_attempts_per_model():
    """The worst case the README states: models x 3 attempts x the timeout."""
    import httpx
    a = Analyser(("m1", "m2", "m3"), client=FakeClient([httpx.ReadTimeout("t")] * 9),
                 sleep=lambda _s: None)
    with pytest.raises(AnalysisError):
        a.analyse("t")
    assert len(a.client.models.calls) == 9


# --- fallback and retry scope ----------------------------------------------------

def test_retired_model_404_falls_through_to_the_next_model():
    a = make([api_error(404, "models/m1 is not found"), FakeResponse(parsed=analysis_obj())])
    assert a.analyse("t").model == "m2"
    assert a.client.models.calls == ["m1", "m2"]


def test_every_model_404_is_an_analysis_error():
    a = make([api_error(404, "not found")] * 2)
    with pytest.raises(AnalysisError, match="404"):
        a.analyse("t")


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("connection refused"),
    httpx.ReadError("connection reset by peer"),
    httpx.RemoteProtocolError("server disconnected without sending a response"),
])
def test_network_blips_from_httpx_are_retried(exc):
    """google-genai raises httpx's types, not the builtin ConnectionError."""
    assert is_retryable(exc)
    a = make([exc, FakeResponse(parsed=analysis_obj())])
    assert a.analyse("t").model == "m1" and a.client.models.calls == ["m1", "m1"]


def rate_limited(delay):
    payload = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
                         "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                      "retryDelay": delay}]}}
    return genai_errors.ClientError(429, payload)


@pytest.mark.parametrize("delay,expected", [("37s", 37.0), ("600s", 60.0), ("1s", 4.0)])
def test_rate_limit_waits_as_long_as_gemini_asks_within_a_cap(delay, expected):
    slept = []
    a = Analyser(("m1",), client=FakeClient([rate_limited(delay),
                                             FakeResponse(parsed=analysis_obj())]),
                 sleep=slept.append)
    a.analyse("t")
    assert slept == [expected]


@pytest.mark.parametrize("delay", ["nans", "infs", "-5s"])
def test_nonsense_retry_delay_falls_back_to_the_backoff(delay):
    """V8: "nans" reached time.sleep() and ended the retry loop with a ValueError."""
    slept = []
    a = Analyser(("m1",), client=FakeClient([rate_limited(delay),
                                             FakeResponse(parsed=analysis_obj())]),
                 sleep=slept.append)
    a.analyse("t")
    assert slept == [4.0]
