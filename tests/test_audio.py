"""Paragraph building — the logic that makes transcripts readable."""

from dataclasses import dataclass

import pytest

from transcribe.audio import Word, build_paragraphs, hms, meeting_date


@dataclass
class FakeSeg:
    start: float
    end: float
    text: str
    words: list


def seg(words):
    return FakeSeg(words[0].start, words[-1].end, "".join(w.text for w in words), words)


def speech(spec):
    """spec: list of (text, start, end) -> one segment holding those words."""
    return [seg([Word(t, s, e) for t, s, e in spec])]


def test_pause_breaks_at_sentence_end_not_mid_sentence():
    # A 3s pause lands mid-sentence; the break must fall back to the full stop.
    words = []
    t = 0.0
    for i in range(30):
        words.append((f" w{i}." if i == 25 else f" w{i}", t, t + 0.2))
        t += 0.25
    words.append((" after", t + 3.0, t + 3.2))       # long pause, mid-sentence
    for i in range(30):
        words.append((f" x{i}", t + 3.3 + i * 0.25, t + 3.4 + i * 0.25))

    paras = build_paragraphs(speech(words), gap=1.5)
    assert len(paras) >= 2
    # every paragraph except possibly the last ends on sentence punctuation
    for _, text in paras[:-1]:
        assert text.rstrip()[-1] in ".?!", f"broke mid-sentence: {text[-40:]!r}"


def test_gap_zero_gives_single_block():
    words = [(f" w{i}.", i * 2.0, i * 2.0 + 0.5) for i in range(50)]
    paras = build_paragraphs(speech(words), gap=0)
    assert len(paras) == 1


def test_no_words_falls_back_to_segments():
    segs = [FakeSeg(0, 5, "Hello there. " * 40, [])]
    paras = build_paragraphs(segs, gap=1.5)
    assert paras and paras[0][1].startswith("Hello there.")


def test_runaway_paragraph_is_eventually_broken():
    # 400 words, no sentence punctuation anywhere, no pauses
    words = [(f" w{i}", i * 0.2, i * 0.2 + 0.1) for i in range(400)]
    paras = build_paragraphs(speech(words), gap=1.5, max_words=90)
    assert len(paras) > 1, "a paragraph with no full stop must still be split"


def test_empty_input():
    assert build_paragraphs([], gap=1.5) == []


@pytest.mark.parametrize("seconds,expected", [(0, "00:00"), (90, "01:30"), (3725, "01:02:05")])
def test_hms(seconds, expected):
    assert hms(seconds) == expected


def test_meeting_date_prefers_name_then_override(tmp_path):
    f = tmp_path / "meeting 2026-04-16 notes.md"
    f.write_text("x")
    assert meeting_date(f) == "2026-04-16"
    assert meeting_date(f, override="2026-01-01") == "2026-01-01"


def test_meeting_date_falls_back_to_mtime(tmp_path):
    import os, time
    f = tmp_path / "no_date_here.md"
    f.write_text("x")
    os.utime(f, (time.time(), 1_700_000_000))     # 2023-11-14 UTC
    assert meeting_date(f).startswith("2023-11-1")


def test_impossible_date_in_file_name_is_ignored(tmp_path):
    f = tmp_path / "call 2026-13-45 then 2026-02-03.m4a"
    f.write_text("x")
    assert meeting_date(f) == "2026-02-03"
    g = tmp_path / "call 2026-13-45.m4a"
    g.write_text("x")
    assert meeting_date(g) != "2026-13-45"
