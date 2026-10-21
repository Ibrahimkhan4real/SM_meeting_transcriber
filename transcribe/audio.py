"""Local speech-to-text and transcript formatting."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .storage import atomic_write

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".flac", ".webm")
SENTENCE_END = (".", "?", "!")
DATE_IN_NAME = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Word:
    """A timed word. Mirrors the shape faster-whisper yields."""

    text: str
    start: float
    end: float


def meeting_date(path: Path, override: str | None = None) -> str:
    """Folder date: explicit override, else YYYY-MM-DD in the name, else mtime."""
    if override:
        return override
    for match in DATE_IN_NAME.finditer(path.stem):
        try:                            # "call 2026-13-45" is not a date
            return date.fromisoformat(match.group(0)).isoformat()
        except ValueError:
            continue
    try:
        return date.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        return date.today().isoformat()


def hms(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def words_from_segments(segments) -> list[Word]:
    """Flatten segments into timed words.

    faster-whisper names the field `word`; our own Word uses `text`. Accepting
    either keeps this callable with plain test doubles instead of requiring a
    real model to exercise the paragraph logic.
    """
    out: list[Word] = []
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            text = getattr(w, "word", None)
            if text is None:
                text = getattr(w, "text", "")
            out.append(Word(text, w.start, w.end))
    return out


def build_paragraphs(segments, gap: float = 1.5, min_words: int = 25,
                     max_words: int = 90, lookback: int = 30
                     ) -> list[tuple[float, str]]:
    """Group a transcript into paragraphs at natural pauses.

    Whisper's own segments are contiguous — each starts exactly where the last
    ended — so segment boundaries carry no pause information. Word-level
    timestamps do, which is why they are requested.

    A pause marks where the speaker drew breath, but it usually lands
    mid-sentence, and breaking there reads worse than not breaking at all. So a
    pause triggers a break at the nearest sentence end behind it, and the words
    after that sentence end carry into the next paragraph.
    """
    words = words_from_segments(segments)

    if gap <= 0:                       # "one unbroken block", as documented
        if words:
            return [(words[0].start, "".join(w.text for w in words).strip())]
        text = " ".join(s.text.strip() for s in segments).strip()
        return [(segments[0].start, text)] if segments and text else []

    if not words:                      # word timestamps unavailable
        paragraphs, current, count, start = [], [], 0, None
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if start is None:
                start = seg.start
            if current and count >= max_words and current[-1].rstrip().endswith(SENTENCE_END):
                paragraphs.append((start, " ".join(current)))
                current, count, start = [], 0, seg.start
            current.append(text)
            count += len(text.split())
        if current:
            paragraphs.append((start, " ".join(current)))
        return paragraphs

    def sentence_cut(buf: list[Word]) -> int | None:
        """Index just past the last sentence end within `lookback` words."""
        floor = max(min_words, len(buf) - lookback)
        for i in range(len(buf) - 1, floor - 1, -1):
            if buf[i].text.rstrip().endswith(SENTENCE_END):
                return i + 1
        return None

    paragraphs: list[tuple[float, str]] = []
    current: list[Word] = []
    prev_end: float | None = None

    def flush(cut: int) -> None:
        nonlocal current
        head, current = current[:cut], current[cut:]
        if head:
            paragraphs.append((head[0].start, "".join(w.text for w in head).strip()))

    for word in words:
        if current and prev_end is not None:
            pause = word.start - prev_end
            cut = None
            if pause >= gap and len(current) >= min_words:
                cut = sentence_cut(current)
                if cut is None and pause >= gap * 3:
                    cut = len(current)          # very long pause: break anyway
            if cut is None and len(current) >= max_words:
                cut = sentence_cut(current)
            if cut is None and len(current) >= max_words * 3:
                cut = len(current)              # runaway with no full stop
            if cut:
                flush(cut)
        current.append(word)
        prev_end = word.end

    flush(len(current))
    return paragraphs


def render_transcript(paragraphs: list[tuple[float, str]], timestamps: bool) -> str:
    if timestamps:
        return "\n\n".join(f"[{hms(start)}] {text}" for start, text in paragraphs)
    return "\n\n".join(text for _, text in paragraphs)


def transcript_document(audio_name: str, body: str, *, model: str, language: str,
                        duration: float, paragraphs: int, device: str) -> str:
    header = (
        f"# Transcription: {audio_name}\n\n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
        f"**Model:** whisper-{model} ({device})  \n"
        f"**Language:** {language}  \n"
        f"**Duration:** {hms(duration)}  \n"
        f"**Paragraphs:** {paragraphs}\n\n"
        "---\n\n"
    )
    return header + body + "\n"


def write_transcript(path: Path, audio_name: str, body: str, **details) -> None:
    atomic_write(path, transcript_document(audio_name, body, **details))
