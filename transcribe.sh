#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── Pre-flight checks ──────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[setup] Error: python3 is required but not found in PATH." >&2
    exit 1
fi

# ── Virtual environment setup ──────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -c "import faster_whisper" &>/dev/null; then
    echo "[setup] Installing faster-whisper..."
    pip install --quiet --upgrade pip
    pip install --quiet faster-whisper
fi

if ! python -c "import tqdm" &>/dev/null; then
    echo "[setup] Installing tqdm..."
    pip install --quiet tqdm
fi

# faster-whisper is used instead of openai-whisper — same model weights, same
# small/medium/large naming, but runs on CPU without hanging.

# ── Transcription ──────────────────────────────────────────────────────────────
# Accept files as arguments, or default to all audio files in the script's directory
if [ "$#" -gt 0 ]; then
    FILES=("$@")
else
    mapfile -t FILES < <(find "$SCRIPT_DIR/audio" -maxdepth 1 \
        -type f \( -iname "*.mp3" -o -iname "*.mp4" -o -iname "*.m4a" \
                   -o -iname "*.wav" -o -iname "*.ogg" -o -iname "*.flac" \
                   -o -iname "*.webm" \))
fi

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No audio files found. Pass file paths as arguments or place audio files in audio/."
    exit 1
fi

python - "$SCRIPT_DIR" "${FILES[@]}" <<'PYTHON'
import sys
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
from tqdm import tqdm

print("Loading 'small' model into memory...")
model = WhisperModel("small", device="cpu", compute_type="int8")

script_dir = Path(sys.argv[1])

for audio_path in sys.argv[2:]:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: The file '{audio_path}' was not found.")
        continue

    try:
        segments, info = model.transcribe(
            str(audio_path),
            language="en",
            condition_on_previous_text=False,
            beam_size=5,
        )

        if info.duration <= 0:
            print(f"Warning: '{audio_path.name}' has zero or unknown duration; skipping.")
            continue

        segment_list = []
        with tqdm(total=round(info.duration), unit="s", unit_scale=True,
                  desc=f"  Transcribing {audio_path.name}") as pbar:
            prev = 0.0
            for seg in segments:
                segment_list.append(seg)
                pbar.update(max(0.0, seg.end - prev))
                prev = seg.end
    except Exception as exc:
        print(f"Error transcribing '{audio_path.name}': {exc}")
        continue

    # Build plain text (mirrors wispher.py's result["text"])
    full_text = " ".join(seg.text.strip() for seg in segment_list)

    if not full_text.strip():
        print(f"Warning: No speech detected in '{audio_path.name}'.")

    print("\n--- Transcription Output ---")
    print(full_text)
    print("----------------------------\n")

    output_dir = script_dir / "transcriptions"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / (audio_path.stem + ".md")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcription: {audio_path.name}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
            f.write(f"**Model:** whisper-small  \n")
            f.write(f"**Language:** English\n\n")
            f.write("---\n\n")
            f.write(full_text + "\n")
        print(f"Saved → {output_path}")
    except OSError as exc:
        print(f"Error saving '{output_path}': {exc}")
        continue
PYTHON

# ── Gemini analysis ────────────────────────────────────────────────────────────
TRANSCRIPT_FILES=()
for f in "${FILES[@]}"; do
    stem=$(basename "${f%.*}")
    TRANSCRIPT_FILES+=("$SCRIPT_DIR/transcriptions/$stem.md")
done

export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

if ! python -c "from google import genai" &>/dev/null; then
    echo "[setup] Installing google-genai..."
    pip install --quiet google-genai
fi

python - "$SCRIPT_DIR" "${TRANSCRIPT_FILES[@]}" <<'PYTHON'
import sys
import os
from pathlib import Path
from google import genai

script_dir = Path(sys.argv[1])
transcript_paths = [Path(p) for p in sys.argv[2:]]

client = genai.Client()

PROMPT_TEMPLATE = """\
You are an expert academic assistant helping a PhD student review their supervision meeting.

The following is a verbatim transcription of a one-on-one supervision meeting between a
PhD student and their university supervisor. The transcription is unedited and may contain
informal speech, filler words, and occasional transcription errors — interpret charitably.

Produce a structured analysis from the perspective of the PhD student.
Output exactly the four sections below, in order, with no additional commentary.

## Executive Summary
A concise 2–3 paragraph overview of the meeting: the overarching themes, the supervisor's
key feedback, and the overall direction of the research.

## Meeting Notes
A clear, chronological narrative covering each topic as it arose, summarising what was
discussed and any conclusions reached. Write in full sentences, past tense, as formal minutes.

## Meeting Points
A bullet list of every distinct topic or discussion point raised, by either party.
Be specific and concrete — one bullet per topic.

## My Action Points
A bullet list of every task, follow-up, or commitment the PhD student must complete before
the next supervision meeting. Include any deadlines or timeframes mentioned.
If none were stated, note the item as open-ended.

---

Transcript:
{transcript}
"""

SECTION_MAP = {
    "Executive Summary": "executive_summary",
    "Meeting Notes":     "meeting_notes",
    "Meeting Points":    "meeting_points",
    "My Action Points":  "action_points",
}

def extract_date(stem):
    import re
    match = re.search(r'\d{4}-\d{2}-\d{2}', stem)
    return match.group(0) if match else "unknown_date"

def parse_sections(text):
    import re
    parts = re.split(r'^## (.+)$', text, flags=re.MULTILINE)
    sections = {}
    i = 1
    while i < len(parts) - 1:
        sections[parts[i].strip()] = parts[i + 1].strip()
        i += 2
    return sections

for transcript_path in transcript_paths:
    if not transcript_path.exists():
        print(f"[analyse] Transcript not found, skipping: {transcript_path}")
        continue

    print(f"[analyse] Analysing: {transcript_path.name}")
    try:
        transcript = transcript_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[analyse] Cannot read '{transcript_path}': {exc}")
        continue

    prompt = PROMPT_TEMPLATE.format(transcript=transcript)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0.3},
        )
    except Exception as exc:
        print(f"[analyse] Gemini API error for '{transcript_path.name}': {exc}")
        continue

    if not response.text:
        print(f"[analyse] Empty response from Gemini for '{transcript_path.name}'; skipping.")
        continue

    date = extract_date(transcript_path.stem)
    output_dir = transcript_path.parent / f"supervisor_meeting_{date}"
    output_dir.mkdir(exist_ok=True)

    sections = parse_sections(response.text)
    for header, content in sections.items():
        file_key = SECTION_MAP.get(header)
        if not file_key:
            print(f"[analyse] Unknown section '{header}', skipping.")
            continue
        out_path = output_dir / f"supervisor_meeting_{date}_{file_key}.md"
        out_path.write_text(f"## {header}\n\n{content}\n", encoding="utf-8")
        print(f"[analyse] Saved → {out_path}")

    # ── Meeting index ─────────────────────────────────────────────────────────
    index_path = transcript_path.parent / "meetings_index.md"
    raw_summary = sections.get("Executive Summary", "")
    first_sentence = raw_summary.split(".")[0].strip()
    summary_first_sentence = (first_sentence + ".") if first_sentence else "[No summary]"

    header_row = "| Date | Folder | Summary |\n|------|--------|--------|\n"
    row = f"| {date} | [{output_dir.name}]({output_dir.name}/) | {summary_first_sentence} |\n"

    if not index_path.exists():
        index_path.write_text(header_row + row, encoding="utf-8")
        print(f"[analyse] Created index → {index_path}")
    else:
        existing = index_path.read_text(encoding="utf-8")
        if date not in existing:
            index_path.write_text(existing + row, encoding="utf-8")
            print(f"[analyse] Updated index → {index_path}")
        else:
            print(f"[analyse] Index already contains entry for {date}, skipping.")
PYTHON

# ── Completion notification ───────────────────────────────────────────────────
MEETING_DATE=$(ls -d "$SCRIPT_DIR/transcriptions/supervisor_meeting_"* 2>/dev/null \
    | sort | tail -1 | xargs basename | sed 's/supervisor_meeting_//')

if [ -n "$MEETING_DATE" ] && command -v notify-send &>/dev/null; then
    notify-send "Transcription complete" \
        "Analysis saved to transcriptions/supervisor_meeting_${MEETING_DATE}" \
        --icon=document --urgency=normal
fi
echo "[done] All files processed."
