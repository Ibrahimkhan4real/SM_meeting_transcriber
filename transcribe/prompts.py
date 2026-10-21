"""Analysis schema and prompts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MeetingAnalysis(BaseModel):
    """The structure Gemini is asked to return.

    Constraining the response to this schema removes the need to parse Markdown
    headings out of free-form prose, which was the single most fragile part of
    the pipeline.
    """

    executive_summary: str = Field(
        description="2-3 paragraph overview: the overarching themes, the "
                    "supervisor's key feedback, and the direction of the research.")
    meeting_notes: str = Field(
        description="Chronological narrative of each topic as it arose and any "
                    "conclusions reached. Full sentences, past tense, as formal minutes.")
    meeting_points: list[str] = Field(
        description="Every distinct topic or discussion point raised by either "
                    "party. Specific and concrete, one entry per topic.")
    action_points: list[str] = Field(
        description="Every task, follow-up or commitment the PhD student must "
                    "complete before the next meeting, with any deadlines "
                    "mentioned. Note open-ended items as open-ended.")


SYSTEM_INSTRUCTION = """\
You are an expert academic assistant helping a PhD student review their supervision meeting.

You are given a verbatim transcription of a one-on-one supervision meeting between a PhD
student and their university supervisor. It is unedited and may contain informal speech,
filler words and occasional transcription errors — interpret charitably.

Produce the analysis from the perspective of the PhD student.
"""

TRANSCRIPT_PROMPT = "Transcript:\n{transcript}"

# Ordered: filename key -> (heading, schema field)
SECTIONS = (
    ("executive_summary", "Executive Summary", "executive_summary"),
    ("meeting_notes",     "Meeting Notes",     "meeting_notes"),
    ("meeting_points",    "Meeting Points",    "meeting_points"),
    ("action_points",     "My Action Points",  "action_points"),
)

# Retained for the Markdown fallback path, and for reading analyses written
# by earlier versions.
SECTION_MAP = {heading: key for key, heading, _ in SECTIONS}

MARKDOWN_PROMPT = """\
""" + SYSTEM_INSTRUCTION + """
Output exactly the four sections below, in order, with no additional commentary.

## Executive Summary
A concise 2–3 paragraph overview of the meeting.

## Meeting Notes
A chronological narrative covering each topic as it arose, as formal minutes.

## Meeting Points
A bullet list of every distinct topic or discussion point raised.

## My Action Points
A bullet list of every task, follow-up, or commitment the student must complete,
including any deadlines. If none were stated, note the item as open-ended.

---

Transcript:
{transcript}
"""
