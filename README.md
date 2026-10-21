# Transcription

Transcribes audio files locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU, no GPU required), then optionally sends the transcript to Google Gemini for structured analysis. Designed for PhD supervision meetings but works for any audio.

Tested on **Linux** (Ubuntu 22.04+) and **macOS** (Ventura 13+ on both Intel and Apple Silicon).

---

## What it produces

For each audio file processed, the script always creates:

- `transcriptions/<stem>.md` — plain-text transcript; `_2`, `_3`, etc. prevent collisions with other recordings or existing unowned files

Completed transcripts are also saved in
`~/Documents/workspace/Notes/Meeting_Notes/` and opened in the desktop's default
Markdown application. The copy keeps the same transcription heading, metadata,
and paragraphs as the original. Identical copies are reused; if an existing note
has different content, a `_2`, `_3`, etc. copy is saved without replacing it.
This also works with `--no-analysis`. An unchanged batch does not reopen old notes;
explicitly passing a file opens its transcript even if it is already up to date.

Use `--notes-dir DIR` to change the notes destination, `--no-notes` to save only
the normal output, or `--no-open` for unattended runs. The equivalent config/env
settings are `TRANSCRIBE_NOTES_DIR`, `TRANSCRIBE_SAVE_NOTES=false`, and
`TRANSCRIBE_OPEN=false`.

When `GEMINI_API_KEY` is configured, it also creates:

- `transcriptions/supervisor_meeting_<date>/` — the structured analysis:
  - `analysis_raw.json` — the structured response as parsed against the schema
    (always written first); `analysis_raw.md` instead when the Markdown fallback
    was used, holding the model's unmodified text
  - `..._executive_summary.md`
  - `..._meeting_notes.md`
  - `..._meeting_points.md`
  - `..._action_points.md`
- `transcriptions/meetings_index.md` — running index of all meetings. The table is
  regenerated; a title or notes you write above or below it are kept.

If a second, different recording resolves to a date that already has a folder, the
new one becomes `supervisor_meeting_<date>_2`, `_3`, and so on. Re-running the same
recording reuses its own folder when its analysis is complete and matches the
transcript. A folder made from this recording's transcript is reused whatever date
its name carries (its index row gets the meeting's date); a folder the tool cannot
tie to the transcript is reused only if its date matches, otherwise a correctly
dated folder is created and the old one is left alone. A completion manifest checks
the transcript and generated output hashes; interrupted or stale analyses are
regenerated, and a section the new analysis no longer has is removed rather than
left behind from the previous run. Legacy and manually edited analyses are
preserved in their existing folder, with regeneration saved beside them.

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

**2. Optional: get a Gemini API key for structured meeting analysis**

Go to [Google AI Studio](https://aistudio.google.com/app/apikey), create a free key, then store it in the config file the script reads on every run:

```bash
mkdir -p -m 700 ~/.config/transcribe
printf 'export GEMINI_API_KEY="your-key-here"\n' > ~/.config/transcribe/gemini.env
chmod 600 ~/.config/transcribe/gemini.env
```

An exported `GEMINI_API_KEY` in your shell also works and takes effect the same way.

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
If creating the `.venv/` fails (on Debian/Ubuntu, usually a missing `python3-venv`),
the half-made folder is removed, so installing the package and running again is enough.
A `.venv/` left without pip by an older version of the script is rebuilt automatically.
An existing `.venv/` is never deleted: it is moved aside as `.venv.broken-<time>` (and
moved back if the rebuild fails), so you can remove it once the new one works.

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

**2. Optional: get a Gemini API key for structured meeting analysis**

Store it in the config file the script reads on every run (preferred — it keeps the
key out of every process your shell spawns):

```bash
mkdir -p -m 700 ~/.config/transcribe
printf 'export GEMINI_API_KEY="your-key-here"\n' > ~/.config/transcribe/gemini.env
chmod 600 ~/.config/transcribe/gemini.env
```

An exported `GEMINI_API_KEY` in your shell also works.

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

You can also pass a file directly. Quote paths that contain spaces:

```bash
transcribe "WhatsApp Audio 2026-08-25 at 8.51.09 AM.mp4"
```

If you forget the quotes, `transcribe` will reconstruct the path when the arguments together identify one existing file.

---

## Options

Run `transcribe --help` for the full list.

| Option | Purpose | Default |
|--------|---------|---------|
| `--model NAME` | Whisper model: `tiny`/`base`/`small`/`medium`/`large-v3`/`distil-large-v3`/`turbo`, or English-only `tiny.en`/`base.en`/`small.en`/`medium.en` | `small` |
| `--language CODE` | Language code, or `auto` to detect | `en` |
| `--device NAME` | `cpu`, `cuda`, or `auto` (detects, falls back to CPU) | `cpu` |
| `--compute-type NAME` | `int8`, `int8_float16`, `float16`, `float32` | `int8` |
| `--cpu-threads N` | CPU inference threads; `0` lets CTranslate2 choose | `0` |
| `--batch-size N` | Batched inference; `0`/`1` disables it | `0` |
| `--no-vad` | Disable voice-activity filtering | VAD on |
| `--paragraph-gap SECS` | Pause that starts a new paragraph; `0` for one block | `1.5` |
| `--timestamps` | Prefix each paragraph with `[MM:SS]` (`[HH:MM:SS]` after the first hour) | off |
| `--force` | Redo work even when output is current | off |
| `--gemini-model LIST` | Comma-separated models to try in order | see Notes |
| `--no-analysis` | Transcribe only, skip Gemini | off |
| `--analyse-only` | Inputs are transcripts; (re)generate their analysis (needs `GEMINI_API_KEY`) | off |
| `--gemini-timeout SECS` | Give up on one Gemini request after this long (`TRANSCRIBE_GEMINI_TIMEOUT`) | `300` |
| `--output-dir DIR` | Where output goes | `<repo>/transcriptions` |
| `--date YYYY-MM-DD` | Force the meeting date | inferred |
| `-v`, `--verbose` | Debug logging, including HTTP traffic | off |
| `-q`, `--quiet` | Warnings and errors only | off |
| `--log-file PATH` | Append a full debug log here | none |
| `--version` | Print the version | |

Any of these can also be set in `~/.config/transcribe/gemini.env`, e.g.
`export TRANSCRIBE_WHISPER_MODEL=medium`. Quote values that contain spaces. An empty
variable in the environment (`GEMINI_API_KEY=`) counts as unset, so the file's value
still applies.

### Readable transcripts

Transcripts are split into paragraphs. Whisper's own segments are contiguous, so
they carry no pause information; the script requests word-level timestamps and
breaks at the sentence end nearest a real pause. Voice-activity filtering is on
by default, which trims silence and reduces text hallucinated over quiet stretches.

### Re-analysing an existing transcript

```bash
transcribe --analyse-only "transcriptions/22-04-26_meeting.md"
```

Useful when a transcript exists but its analysis is missing, or to regenerate one
with a different model. Add `--force` to overwrite an analysis that already exists.
The analysis folder and `meetings_index.md` are written next to the transcript;
`--output-dir` is not used in this mode. Without a Gemini key it exits with an error.
A transcript with no speech (for example a silent recording) is never sent for analysis.

---

## Supported audio formats

`.mp3`, `.mp4`, `.m4a`, `.wav`, `.ogg`, `.flac`, `.webm`

---

## Files

| Path | Purpose |
|------|---------|
| `transcribe.sh` | Bootstrap: ensures the virtualenv, then runs the package |
| `transcribe/cli.py` | Argument parsing, logging, run orchestration |
| `transcribe/engine.py` | Whisper model loading and transcription |
| `transcribe/audio.py` | Paragraph building and transcript formatting |
| `transcribe/analyse.py` | Gemini calls, retry policy, response handling |
| `transcribe/storage.py` | Atomic writes, locking, folders, meetings index |
| `transcribe/hardware.py` | CPU/GPU detection and safe fallbacks |
| `transcribe/provenance.py` | Source ownership and freshness checks for safe re-runs |
| `transcribe/notes.py` | Copies transcripts into the meeting-notes folder |
| `transcribe/notify.py` | Desktop notifications and opening Markdown files |
| `transcribe/config.py` | Settings, config-file parsing, validation |
| `transcribe/prompts.py` | Response schema and prompts |
| `tests/` | pytest suite (`.venv/bin/pytest`) |
| `requirements.txt` | Pinned runtime dependencies |

---

## Architecture

`transcribe.sh` only bootstraps the virtualenv and hands off to `python -m transcribe`.
All logic lives in the package, which means it can be imported and tested; the shell
script is about 50 lines rather than 800.

A run is three stages:

1. **Transcribe** — `engine.py` loads Whisper once and processes each file.
   `audio.py` turns timed words into paragraphs.
2. **Analyse** — `analyse.py` sends the transcript to Gemini and gets back a
   schema-constrained JSON object.
3. **Store** — `storage.py` writes every file atomically and updates the index
   under an exclusive lock.

### Reliability notes

- **Every write is atomic.** Content goes to a temporary file in the destination
  directory and is renamed into place, so an interrupted run cannot truncate a
  file — including `meetings_index.md`, which is the catalogue of everything.
- **Index updates take a lock.** Without it, two concurrent runs lose roughly
  half their rows (there is a test that demonstrates exactly this).
- **The index only lists finished analyses.** Its row is written after the
  completion manifest; if a run stops in between, the next run adds the missing
  row without calling Gemini again. A row you delete yourself, or a deleted
  index, is not brought back, and a run with nothing to analyse never reads the
  index, so an index it cannot read does not fail it.
- **Retries are decided by HTTP status code**, not by searching the error text.
  Substring matching used to retry permanent failures whose messages happened to
  contain a retryable number, e.g. "token count 1500 exceeds the maximum".
  Network faults (connection refused or reset) are retried too, a 429 waits as
  long as Gemini asks (up to 60 s), and a model that no longer exists (404) moves
  straight on to the next model in the list.
- **The analysis is schema-constrained JSON**, so a reworded heading can no
  longer silently drop a section. A Markdown fallback remains for any model that
  rejects the schema.
- **Content filters and token limits raise clear errors** rather than producing
  an empty analysis. That includes a transcript Gemini refuses outright (block
  reason shown) and a reply whose sections are all empty: nothing is saved, so
  the next run tries again.
- **Gemini requests time out after 5 minutes** (`--gemini-timeout SECS` or
  `TRANSCRIBE_GEMINI_TIMEOUT`) and are retried like a 503. The client library's
  default is to wait forever; one stalled request after a 503 once hung a run for
  27 minutes with no output. Each model is tried 3 times, so when Gemini stalls
  the worst case is models x 3 x the timeout, plus about 12 s of backoff per
  model: about 45 minutes with the three default models. A 429 that names a
  delay can add up to 60 s per retry instead of the 4 s and 8 s.
- **`transcribe.sh` keeps the folder you run it from off Python's module
  path.** The tool itself starts without it (empty or relative `PYTHONPATH`
  entries are dropped too), and the setup steps (`venv`, `pip`) run isolated
  (`-I`) from `/`. A `logging.py` or `types.py` lying in that folder is
  therefore not imported in place of the real module.

## Development

```bash
.venv/bin/pytest                 # run the test suite
pre-commit install               # shellcheck + hygiene hooks (optional)
pip install -r requirements-dev.txt
```

## Performance

Transcription is CPU-bound. Measured on a 20-core i7-12700H, `whisper-small`/`int8`,
120 s of audio, medians of interleaved repeated runs:

| Setting | Result |
|---------|--------|
| CTranslate2's default thread count | 8.7 s |
| `--cpu-threads 4` | 8.9 s |
| `--cpu-threads 8` | 10.3 s |

**More threads is slower.** Oversubscribing the cores costs about 18%, so
`--cpu-threads` defaults to `0`, meaning "let CTranslate2 decide". Only override it
if you have measured your own machine.

End-to-end on the same 120 s clip, including the roughly 3.5 s model load:

| Config | Median |
|--------|--------|
| defaults | 12.2 s |
| `--batch-size 4` | 11.3 s |
| `--no-vad` | 13.4 s |
| `--paragraph-gap 0` | 12.8 s |

- `--batch-size 4` gains about 7% end-to-end on a short file (roughly 20% of the
  transcription itself; the fixed model load dilutes it) and costs about 1.5x the
  memory. Larger batches were slower again here. It stays opt-in.
- **Voice-activity filtering makes it faster, not slower** — about 10% — because
  silence it removes is silence not transcribed. It removed 12 s from a 90 s clip.
  On by default, and `--no-vad` costs you both speed and quality.
- **Word-level timestamps are effectively free.** `--paragraph-gap 0` skips them
  and measured no faster, so readable paragraphs cost nothing.
- **GPU**: `--device auto` uses CUDA when it is genuinely available and falls back
  to CPU with an explanation otherwise. A GPU that is present but whose driver is
  not running counts as unavailable — no crash, just a note. CUDA libraries such as
  cuBLAS only load on the first file, so if they are missing (`libcublas.so.12 is not
  found`) the run warns and finishes on CPU; `--device cpu` skips the attempt. Every
  number above is CPU-only: on this machine the T600 driver works but the CUDA 12
  runtime libraries are not installed (checked 2026-10-09).

---

## Notes

- Transcription uses the `small` Whisper model on CPU (`int8` quantisation) by default.
  Works on Intel and Apple Silicon. Larger models (`--model medium`, `--model large-v3`)
  are more accurate but slower.
- A Gemini API key is not required for local transcription. Without one, the script saves the transcript and skips structured analysis.
- Analysis requests a schema-constrained JSON response, so a reworded heading can
  no longer silently drop a section; the Markdown parser remains as a fallback.
- Gemini models are tried in order: `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-2.5-flash`.
  Flash-tier models intermittently return `503 high demand` — measured on 2026-09-05,
  `gemini-3.8-flash` failed 2 of 5 calls and `gemini-3.7-flash` 1 of 5, while 3.6 and below
  were perfect. The script retries with backoff, then falls back to the next model.
  Choose your own with `--gemini-model`. The run log names the model that produced each
  analysis; the Markdown fallback also records it at the top of `analysis_raw.md`.
- Dependencies are pinned in `requirements.txt`. faster-whisper does **not** need PyTorch —
  inference runs through CTranslate2.
- Re-running reuses transcripts only when source content, transcription settings,
  and the saved transcript match `.transcript-provenance.json`. Duplicate source
  arguments are processed once. Existing transcripts without provenance, and
  manually edited transcripts, are preserved; fresh output gets a numbered name.
  A different recording saved under a name used before (say a phone app's
  `Recording.m4a`) is a new meeting: it gets `Recording_2.md` and its own analysis,
  and the earlier meeting's transcript and analysis are kept.
  Verified complete analyses of unchanged transcripts are not re-sent to Gemini.
  `--force` regenerates owned output; user edits and legacy files remain preserved.
- The `audio/` and `transcriptions/` directories are excluded from version control (`.gitignore`).
- The meeting date comes from a real `YYYY-MM-DD` in the file name; failing that, from the
  audio file's modification time. It is never `unknown_date`. The date is stored in
  `.transcript-provenance.json` when the recording is transcribed, so later runs and
  `--analyse-only` keep it even after the recording is moved or its modification
  time changes. A transcript this tool did not make is dated by its own name or
  modification time.
- The script exits non-zero if any file failed to transcribe or analyse, so it can be
  used from cron or a wrapper script.
- `transcribe` works through a symlink (e.g. `~/.local/bin/transcribe`); output always
  goes to the repository's `transcriptions/`, not next to the symlink.
