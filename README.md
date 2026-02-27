# Transcription

Transcribes audio files locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU, no GPU required), then sends the transcript to Google Gemini for structured analysis. Designed for PhD supervision meetings but works for any audio.

## What it produces

For each audio file processed, the script creates:

- `transcriptions/<stem>.md` — plain-text transcript
- `transcriptions/supervisor_meeting_<date>/` — four structured Markdown files:
  - `executive_summary.md`
  - `meeting_notes.md`
  - `meeting_points.md`
  - `action_points.md`
- `transcriptions/meetings_index.md` — running index of all meetings

## Prerequisites

- Python 3 (tested with 3.12)
- Bash

Dependencies (`faster-whisper`, `tqdm`, `google-genai`) are installed automatically into a local `.venv/` on first run.

## Setup

1. **Set your Gemini API key** in your shell before running:

   ```bash
   export GEMINI_API_KEY="your-key-here"
   ```

   > **Security note:** Never hardcode your API key directly in `transcribe.sh`.
   > If you have added a key to the script, remove it and use the environment variable above before pushing to a public repository.

2. **Place audio files** in the `audio/` directory (`.mp3`, `.mp4`, `.m4a`, `.wav`, `.ogg`, `.flac`, `.webm`), or pass file paths directly as arguments.

3. **Run the script:**

   ```bash
   bash transcribe.sh
   ```

   Or pass specific files:

   ```bash
   bash transcribe.sh path/to/recording.m4a
   ```

The virtual environment is created automatically in `.venv/` on first run.

## Files

| File | Purpose |
|------|---------|
| `transcribe.sh` | Main entry point — runs transcription and Gemini analysis |
| `wispher.py` | Legacy standalone transcription script (openai-whisper) |

## Notes

- Transcription uses the `small` Whisper model on CPU (`int8` quantisation). Larger models (`medium`, `large`) are more accurate but slower.
- Gemini model used: `gemini-2.5-flash`.
- The `audio/` and `transcriptions/` directories are excluded from version control (see `.gitignore`).
