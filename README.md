# Transcription

Transcribes audio files locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU, no GPU required), then sends the transcript to Google Gemini for structured analysis. Designed for PhD supervision meetings but works for any audio.

Tested on **Linux** (Ubuntu 22.04+) and **macOS** (Ventura 13+ on both Intel and Apple Silicon).

---

## What it produces

For each audio file processed, the script creates:

- `transcriptions/<stem>.md` — plain-text transcript
- `transcriptions/supervisor_meeting_<date>/` — four structured Markdown files:
  - `executive_summary.md`
  - `meeting_notes.md`
  - `meeting_points.md`
  - `action_points.md`
- `transcriptions/meetings_index.md` — running index of all meetings

---

## Quick start

### macOS

**1. Install Python 3**

If you don't have Python 3, the easiest ways are:

```bash
# Option A — Xcode Command Line Tools (no extra installs)
xcode-select --install

# Option B — Homebrew (recommended if you use Homebrew already)
brew install python
```

Verify: `python3 --version` should print `Python 3.x.x`.

**2. Get a Gemini API key**

Go to [Google AI Studio](https://aistudio.google.com/app/apikey), create a free key, then add it to your shell profile so it's always set:

```bash
# Add this line to ~/.zshrc (macOS default shell)
echo 'export GEMINI_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

**3. Clone the repo and place your audio files**

```bash
git clone https://github.com/Ibrahimkhan4real/SM_meeting_transcriber.git
cd SM_meeting_transcriber
mkdir -p audio
# Copy your .mp3 / .m4a / .wav etc. files into the audio/ folder
```

**4. Run**

```bash
bash transcribe.sh
```

Or point it directly at a file:

```bash
bash transcribe.sh ~/Downloads/supervision_2025-04-22.m4a
```

On first run the script creates a `.venv/` and installs `faster-whisper`, `tqdm`, and `google-genai` automatically. This takes a minute. Subsequent runs start immediately.

When the analysis is complete, a macOS notification appears from Notification Centre.

---

### Linux

**1. Install Python 3**

```bash
# Debian / Ubuntu
sudo apt install python3 python3-venv

# Fedora
sudo dnf install python3
```

**2. Get a Gemini API key**

Add to `~/.bashrc` (or `~/.zshrc` if using zsh):

```bash
echo 'export GEMINI_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**3. Clone the repo and place your audio files**

```bash
git clone https://github.com/Ibrahimkhan4real/SM_meeting_transcriber.git
cd SM_meeting_transcriber
mkdir -p audio
# Copy audio files into audio/
```

**4. Run**

```bash
bash transcribe.sh
```

A desktop notification appears via `notify-send` when the analysis is complete (requires `libnotify-bin` on Debian/Ubuntu: `sudo apt install libnotify-bin`).

---

## Supported audio formats

`.mp3`, `.mp4`, `.m4a`, `.wav`, `.ogg`, `.flac`, `.webm`

---

## Files

| File | Purpose |
|------|---------|
| `transcribe.sh` | Main entry point — transcription and Gemini analysis |
| `wispher.py` | Legacy standalone transcription script (openai-whisper) |

---

## Notes

- Transcription uses the `small` Whisper model on CPU (`int8` quantisation). Works on Intel and Apple Silicon. Larger models (`medium`, `large`) are more accurate but slower.
- Gemini model: `gemini-2.5-flash`.
- The `audio/` and `transcriptions/` directories are excluded from version control (`.gitignore`).
- If your file names contain a date in `YYYY-MM-DD` format, that date is used in the output folder name. Otherwise the folder is named `supervisor_meeting_unknown_date`.
