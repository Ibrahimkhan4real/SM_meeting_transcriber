"""Command-line interface and run orchestration."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

from . import __version__
from .analyse import AnalysisError, first_sentence
from .audio import (AUDIO_SUFFIXES, build_paragraphs, meeting_date,
                    render_transcript, transcript_document)
from .config import (DEFAULT_GEMINI_MODELS, ConfigError, Settings,
                     config_path, default_notes_dir, load_config_file)
from .notes import save_meeting_note
from .provenance import (MANIFEST, allocate_transcript, analysis_current, analysis_owned,
                         complete_analysis, digest, index_pending, manifest_hash,
                         mark_indexed, owned_outputs, read_json, record_transcript,
                         recorded_date, recorded_source, source_signature,
                         transcript_current, transcript_hashes)
from .notify import notify, open_markdown
from .storage import (FOLDER_PREFIX, INDEX_NAME, analysis_dirs, atomic_write,
                      directory_lock, index_has_row, reserve_output_dir, write_index_row)

log = logging.getLogger("transcribe")

EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2
EXIT_INTERRUPTED = 130                  # the shell convention for Ctrl+C


def _bool_env(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show a default only where it means something.

    The stock formatter printed "--no-vad (default: True)", which reads as
    "VAD is off by default", and "(default: None)" on every optional value.
    """

    def _get_help_string(self, action):
        if action.nargs == 0 or action.default is None:
            return action.help
        return super()._get_help_string(action)


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio locally, then optionally analyse it with Gemini.",
        epilog=("Settings may also come from %s, e.g. "
                "TRANSCRIBE_WHISPER_MODEL=medium." % config_path()),
        formatter_class=_HelpFormatter,
    )
    p.add_argument("inputs", nargs="*", metavar="FILE",
                   help="audio files (or transcripts with --analyse-only); "
                        "defaults to everything in audio/")

    t = p.add_argument_group("transcription")
    t.add_argument("--model", dest="whisper_model", default=defaults.whisper_model,
                   help="Whisper model name")
    t.add_argument("--language", default=defaults.language,
                   help="language code, or 'auto' to detect")
    t.add_argument("--device", default=defaults.device, help="cpu, cuda, or auto")
    t.add_argument("--compute-type", default=defaults.compute_type,
                   help="quantisation, e.g. int8 or float16")
    t.add_argument("--batch-size", type=int, default=defaults.batch_size,
                   help="batched inference size; 0 or 1 disables batching")
    t.add_argument("--cpu-threads", type=int, default=defaults.cpu_threads,
                   help="CPU inference threads; 0 picks a sensible value")
    t.add_argument("--no-vad", dest="vad", action="store_false", default=defaults.vad,
                   help="disable voice-activity filtering")
    t.add_argument("--paragraph-gap", type=float, default=defaults.paragraph_gap,
                   metavar="SECS", help="pause that starts a paragraph; 0 for one block")
    t.add_argument("--timestamps", action="store_true", default=defaults.timestamps,
                   help="prefix each paragraph with [MM:SS], or [HH:MM:SS] past an hour")
    t.add_argument("--force", action="store_true", default=defaults.force,
                   help="redo work even when the output is up to date")

    a = p.add_argument_group("analysis")
    a.add_argument("--gemini-model", dest="gemini_models", default=None, metavar="LIST",
                   help="comma-separated models to try in order (default: %s)"
                        % ",".join(defaults.gemini_models))
    a.add_argument("--no-analysis", action="store_true", default=defaults.no_analysis,
                   help="transcribe only; skip Gemini")
    a.add_argument("--analyse-only", "--analyze-only", dest="analyse_only",
                   action="store_true", default=defaults.analyse_only,
                   help="inputs are existing transcripts; (re)generate their analysis")
    a.add_argument("--gemini-timeout", type=float, default=defaults.gemini_timeout,
                   metavar="SECS", help="give up on one Gemini request after this long; "
                                        "each model is tried up to 3 times")

    o = p.add_argument_group("output")
    o.add_argument("--output-dir", type=Path, default=defaults.output_dir,
                   help="where transcripts and analyses go")
    notes = o.add_mutually_exclusive_group()
    notes.add_argument("--notes-dir", type=Path, default=defaults.notes_dir,
                       help="also save transcripts in this meeting-notes folder")
    notes.add_argument("--no-notes", dest="notes_dir", action="store_const", const=None,
                       help="skip saving a copy in the meeting-notes folder")
    o.add_argument("--no-open", dest="open_transcript", action="store_false",
                   default=defaults.open_transcript,
                   help="do not open completed Markdown transcripts")
    o.add_argument("--date", dest="meeting_date", default=defaults.meeting_date,
                   metavar="YYYY-MM-DD", help="force the meeting date")

    d = p.add_argument_group("diagnostics")
    d.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    d.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    d.add_argument("--log-file", type=Path, default=None, help="also append logs here")
    d.add_argument("--version", action="version", version=f"transcribe {__version__}")
    return p


def settings_from_env(default_output_dir: Path) -> Settings:
    """Defaults layered from the config file and the environment."""
    file_values = load_config_file()
    # An empty variable counts as unset; "GEMINI_API_KEY=" in the environment
    # used to hide the key configured in the file.
    env = {**file_values, **{k: v for k, v in os.environ.items() if v != ""}}

    def get(name: str, fallback):
        return env.get(f"TRANSCRIBE_{name}") or fallback

    def number(name: str, kind, fallback):
        # A typo here used to crash every command, even --help, with a traceback.
        try:
            return kind(get(name, fallback))
        except ValueError:
            raise ConfigError(
                f"TRANSCRIBE_{name} must be a number (got {get(name, fallback)!r}); "
                f"check the environment and {config_path()}") from None

    models = get("GEMINI_MODEL", None)
    return Settings(
        whisper_model=get("WHISPER_MODEL", "small"),
        language=get("LANGUAGE", "en"),
        device=get("DEVICE", "cpu"),
        compute_type=get("COMPUTE_TYPE", "int8"),
        batch_size=number("BATCH_SIZE", int, 0),
        cpu_threads=number("CPU_THREADS", int, 0),
        vad=_bool_env(env.get("TRANSCRIBE_VAD"), True),
        paragraph_gap=number("PARAGRAPH_GAP", float, 1.5),
        timestamps=_bool_env(env.get("TRANSCRIBE_TIMESTAMPS"), False),
        force=_bool_env(env.get("TRANSCRIBE_FORCE"), False),
        gemini_models=tuple(m.strip() for m in models.split(",") if m.strip())
                      if models else DEFAULT_GEMINI_MODELS,
        gemini_timeout=number("GEMINI_TIMEOUT", float, 300.0),
        no_analysis=_bool_env(env.get("TRANSCRIBE_NO_ANALYSIS"), False),
        output_dir=Path(get("OUTPUT_DIR", default_output_dir)),
        notes_dir=Path(get("NOTES_DIR", default_notes_dir()))
                  if _bool_env(env.get("TRANSCRIBE_SAVE_NOTES"), True) else None,
        open_transcript=_bool_env(env.get("TRANSCRIBE_OPEN"), True),
        meeting_date=env.get("TRANSCRIBE_DATE") or None,
        api_key=env.get("GEMINI_API_KEY") or None,
    )


def configure_logging(settings: Settings) -> None:
    level = logging.DEBUG if settings.verbose else (
        logging.WARNING if settings.quiet else logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    if settings.log_file:
        try:
            settings.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot write --log-file {settings.log_file}: "
                              f"{exc.strerror or exc}") from None
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(file_handler)

    # Third-party chatter: useful at --verbose, noise otherwise.
    third_party_level = logging.DEBUG if settings.verbose else logging.WARNING
    # faster_whisper's INFO lines otherwise land in the middle of the progress bar.
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub",
                 "filelock", "google.auth", "faster_whisper"):
        logging.getLogger(name).setLevel(third_party_level)
    # google-genai emits an automatic-function-calling advisory at WARNING on
    # every single generate_content call; it is not actionable here.
    logging.getLogger("google_genai").setLevel(
        logging.DEBUG if settings.verbose else logging.ERROR)


def collect_inputs(settings: Settings, audio_dir: Path) -> list[Path]:
    if settings.inputs:
        joined = " ".join(settings.inputs)
        # A single unquoted path containing spaces, as typed at the shell.
        if len(settings.inputs) > 1 and Path(joined).is_file():
            return [Path(joined)]
        return [Path(i) for i in settings.inputs]
    if settings.analyse_only:
        raise ConfigError("--analyse-only needs one or more transcript .md files")
    audio_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in audio_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def run_transcription(settings: Settings, inputs: list[Path]) -> tuple[list[dict], int]:
    """Transcribe audio. Returns (records, failure count)."""
    records: list[dict] = []
    failures = 0
    pending: list[tuple[Path, Path, dict]] = []
    seen: set[Path] = set()

    try:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Cannot create output folder %s: %s", settings.output_dir,
                  exc.strerror or exc)
        return records, len(inputs)

    for audio_path in inputs:
        if audio_path.resolve() in seen:
            continue
        seen.add(audio_path.resolve())
        if audio_path.is_dir():
            log.error("%s is a folder; pass the audio files inside it instead.", audio_path)
            failures += 1
            continue
        if not audio_path.is_file():
            log.error("File not found: %s", audio_path)
            failures += 1
            continue
        try:
            signature = source_signature(audio_path, settings)
            transcript_path = allocate_transcript(audio_path, settings.output_dir,
                                                  signature["audio_sha256"])
            fresh = not settings.force and transcript_current(transcript_path, signature)
        except OSError as exc:
            log.error("Cannot prepare %s: %s", audio_path, exc)
            failures += 1
            continue
        # The date stored with this recording's transcript wins over the audio's
        # mtime, which a copy, sync or restore changes.
        date_str = (settings.meeting_date or recorded_date(transcript_path)
                    or meeting_date(audio_path))
        if fresh:
            log.info("[skip] %s is up to date (use --force to redo).", transcript_path.name)
            records.append({"transcript": transcript_path, "created": False,
                            "date": date_str})
            continue
        pending.append((audio_path, transcript_path, signature, date_str))

    if not pending:
        return records, failures

    from tqdm import tqdm

    from .engine import Transcriber
    try:
        engine = Transcriber(settings.whisper_model, settings.device,
                             settings.compute_type, settings.batch_size,
                             settings.cpu_threads)
    except Exception as exc:                         # noqa: BLE001
        # e.g. a model that is not downloaded yet while offline: one line, not 67.
        log.error("Cannot load Whisper model %r: %s", settings.whisper_model, exc)
        return records, failures + len(pending)

    def run_engine(audio_path: Path):
        with tqdm(unit="s", unit_scale=True, disable=settings.quiet,
                  desc=f"  {audio_path.name}") as bar:
            def set_total(seconds: float) -> None:
                bar.total = seconds                 # enables percentage and ETA
                bar.refresh()
            result = engine.transcribe(
                audio_path, language=settings.language, vad=settings.vad,
                # Word timing is only consulted when splitting paragraphs.
                word_timestamps=settings.paragraph_gap > 0,
                progress=bar.update, on_duration=set_total)
            if bar.total:
                bar.n = bar.total                   # trailing silence: end at 100%
            return result

    for audio_path, transcript_path, signature, date_str in pending:
        try:
            try:
                result = run_engine(audio_path)
            except RuntimeError as exc:
                if engine.device == "cpu":
                    raise
                # CUDA libraries (cuBLAS, cuDNN) load lazily on the first
                # inference, so a GPU that initialised fine can still fail here.
                log.warning("GPU transcription failed (%s); falling back to CPU. "
                            "Use --device cpu to skip the GPU attempt.", exc)
                engine = Transcriber(settings.whisper_model, "cpu", "int8",
                                     settings.batch_size, settings.cpu_threads)
                result = run_engine(audio_path)
        except Exception as exc:                     # noqa: BLE001
            log.error("Error transcribing %s: %s", audio_path.name, exc)
            failures += 1
            continue

        if result.duration <= 0:
            log.warning("%s has zero or unknown duration; skipping.", audio_path.name)
            failures += 1
            continue

        paragraphs = build_paragraphs(result.segments, gap=settings.paragraph_gap)
        body = render_transcript(paragraphs, settings.timestamps)
        if not body.strip():
            log.warning("No speech detected in %s.", audio_path.name)

        document = transcript_document(
            audio_path.name, body, model=settings.whisper_model, language=result.language,
            duration=result.duration, paragraphs=len(paragraphs), device=engine.device)
        try:
            # Ownership first: a crash after this leaves a file the next run
            # still recognises, not an orphan that forces a "_2" re-transcription.
            record_transcript(audio_path, transcript_path, signature,
                              hashlib.sha256(document.encode("utf-8")).hexdigest(),
                              meeting_date=date_str)
            atomic_write(transcript_path, document)
        except OSError as exc:
            log.error("Error saving %s: %s", transcript_path, exc)
            failures += 1
            continue

        log.info("Saved → %s  (%d paragraphs)", transcript_path, len(paragraphs))
        records.append({"transcript": transcript_path,
                        "created": True,
                        "date": date_str})

    return records, failures


def collect_transcripts(settings: Settings, inputs: list[Path]) -> tuple[list[dict], int]:
    """--analyse-only: the inputs are existing transcripts."""
    records, failures = [], 0
    for path in inputs:
        if not path.exists():
            log.error("Transcript not found: %s", path)
            failures += 1
            continue
        if path.suffix.lower() != ".md":
            log.error("--analyse-only expects .md transcripts, got %s", path.name)
            failures += 1
            continue
        log.info("Queued: %s", path.name)
        # Date the meeting as a normal run does: the date stored when it was
        # transcribed, else its recording if still there unchanged. The
        # transcript's own mtime is only when it happened to be transcribed.
        records.append({"transcript": path,
                        "date": (settings.meeting_date or recorded_date(path)
                                 or meeting_date(recorded_source(path) or path))})
    return records, failures


def run_analysis(settings: Settings, records: list[dict]) -> tuple[list[Path], int]:
    """Analyse each transcript. Returns (created folders, failure count)."""
    from .analyse import Analyser

    analyser = Analyser(settings.gemini_models, settings.api_key,
                        timeout=settings.gemini_timeout)
    created: list[Path] = []
    failures = 0

    for record in records:
        transcript_path: Path = record["transcript"]
        parent = transcript_path.parent

        if not settings.force:
            current = next((folder for folder in analysis_dirs(parent, transcript_path.stem)
                            if analysis_current(folder, transcript_path)), None)
            if current is not None:
                log.info("%s: analysis already exists (%s); use --force to regenerate.",
                         transcript_path.name, current.name)
                # A run stopped between the manifest and the index left no row.
                # Nothing else touches the index here, so a no-op run cannot
                # fail on it, and a row the user deleted is not brought back.
                if index_pending(current):
                    add_missing_row(parent, current, record["date"])
                continue

        try:
            transcript_bytes = transcript_path.read_bytes()
            transcript = transcript_bytes.decode("utf-8")
            transcript_digest = hashlib.sha256(transcript_bytes).hexdigest()
        except (OSError, UnicodeDecodeError) as exc:
            log.error("Cannot read %s: %s", transcript_path, exc)
            failures += 1
            continue

        # A silent recording yields only the header; analysing it spends a
        # Gemini call and files an invented meeting in the index.
        _, separator, body = transcript.partition("\n---\n")
        if not (body if separator else transcript).strip():
            log.warning("%s contains no speech; skipping analysis.", transcript_path.name)
            continue

        log.info("Analysing: %s", transcript_path.name)
        try:
            analysis = analyser.analyse(transcript)
        except AnalysisError as exc:
            log.error("Analysis failed for %s: %s", transcript_path.name, exc)
            failures += 1
            continue
        except Exception as exc:                     # noqa: BLE001
            log.error("Gemini error for %s: %s", transcript_path.name, exc)
            failures += 1
            continue

        log.info("Model used: %s (%s)", analysis.model,
                 "structured" if analysis.structured else "markdown fallback")

        try:
            with directory_lock(parent):
                folder = analysis_folder(parent, transcript_path, transcript_digest,
                                         record["date"])
                old_outputs = owned_outputs(read_json(folder / MANIFEST)) or {}
                files = analysis.as_markdown()
                target_names = ["analysis_raw.md", "analysis_raw.json"] + [
                    f"{folder.name}_{key}.md" for key in files]
                has_content = any(path for pattern in ("*.md", "*.json")
                                  for path in folder.glob(pattern))
                preserve = has_content and not analysis_owned(folder, target_names)
                if preserve:
                    # Keep legacy or manually edited analyses; allocate a fresh sibling.
                    folder = reserve_output_dir(parent, record["date"], transcript_path.stem,
                                                reuse=False)
                    old_outputs = {}
                outputs = []
                # Persist the raw response first: a change in Gemini's heading
                # format can then never destroy the analysis.
                raw_name = "analysis_raw.json" if analysis.structured else "analysis_raw.md"
                raw_path = folder / raw_name
                if analysis.structured:
                    atomic_write(raw_path, analysis.raw + "\n")
                else:
                    atomic_write(raw_path, f"<!-- model: {analysis.model} -->\n\n{analysis.raw}")
                outputs.append(raw_path)
                log.info("Saved → %s", raw_path)

                for key, content in files.items():
                    target = folder / f"{folder.name}_{key}.md"
                    atomic_write(target, content)
                    outputs.append(target)
                    log.info("Saved → %s", target)
                if len(files) < 4:
                    log.warning("Only %d/4 sections had content; see %s",
                                len(files), raw_path)

                # Remove our own, unedited files this analysis no longer produces:
                # last run's action points must not sit beside today's analysis,
                # and analysis_raw.md vs .json must say which one is current.
                current = {path.name for path in outputs}
                for name, expected in old_outputs.items():
                    stale = folder / name
                    if (name not in current and stale.is_file() and not stale.is_symlink()
                            and digest(stale) == expected):
                        stale.unlink()
                        log.debug("Removed superseded %s", name)

                # The manifest commits the folder. The index only lists committed
                # folders, so it is written after, and repaired by a later run.
                complete_analysis(folder, transcript_digest, outputs)
        except (OSError, ValueError) as exc:
            log.error("Cannot save analysis for %s: %s", transcript_path.name, exc)
            failures += 1
            continue

        if update_index(parent, folder, record["date"],
                        first_sentence(analysis.executive_summary)):
            try:
                mark_indexed(folder)
            except OSError as exc:              # the row is there; repaired as a no-op
                log.debug("Could not mark %s as indexed: %s", folder.name, exc)
        else:
            failures += 1
        created.append(folder)

    return created, failures


def folder_date(folder: Path) -> str:
    """The YYYY-MM-DD in a 'supervisor_meeting_<date>[_N]' folder name."""
    return folder.name[len(FOLDER_PREFIX):][:10]


def analysis_folder(parent: Path, transcript: Path, transcript_digest: str,
                    date_str: str) -> Path:
    """The folder this analysis goes to. Callers hold `directory_lock(parent)`.

    A folder whose manifest was made from this transcript (as it is now, or as
    the tool wrote it before its last re-transcription) is this meeting's,
    whatever date its name carries. A folder that cannot be tied to it is only
    reused when its date matches; otherwise a correctly dated sibling is made.
    """
    folders = analysis_dirs(parent, transcript.stem)
    hashes = transcript_hashes(transcript) | {transcript_digest}
    for folder in folders:
        if manifest_hash(folder) in hashes:
            return folder
    if folders and folder_date(folders[0]) == date_str:
        return folders[0]
    return reserve_output_dir(parent, date_str, transcript.stem, reuse=False)


def folder_date(folder: Path) -> str:
    """The YYYY-MM-DD in a 'supervisor_meeting_<date>[_N]' folder name."""
    return folder.name[len(FOLDER_PREFIX):][:10]


def update_index(parent: Path, folder: Path, date_str: str,
                 summary: str | None = None) -> bool:
    """Add or refresh the folder's row in meetings_index.md. False on failure."""
    index = parent / INDEX_NAME
    try:
        with directory_lock(parent):
            if summary is None:
                summary = saved_summary(folder)
            new = write_index_row(index, folder.name, date_str, summary)
    except (OSError, ValueError) as exc:        # ValueError: not valid UTF-8
        log.error("The analysis is saved, but %s could not be updated: %s", index, exc)
        return False
    if new:
        log.info("Updated index → %s", index)
    else:
        log.info("Index row for %s refreshed.", folder.name)
    return True


def add_missing_row(parent: Path, folder: Path, date_str: str) -> None:
    """Finish an index update an earlier run never made. Warns, never fails."""
    index = parent / INDEX_NAME
    try:
        if not index_has_row(index, folder.name):
            with directory_lock(parent):
                write_index_row(index, folder.name, date_str, saved_summary(folder))
            log.info("Added the missing row for %s to %s", folder.name, index)
        mark_indexed(folder)
    except (OSError, ValueError) as exc:        # ValueError: not valid UTF-8
        log.warning("Could not add the row for %s to %s: %s", folder.name, index, exc)


def saved_summary(folder: Path) -> str:
    """First sentence of a saved executive summary, for repairing the index."""
    try:
        text = (folder / f"{folder.name}_executive_summary.md").read_text(encoding="utf-8")
    except (OSError, ValueError):
        text = ""
    return first_sentence("\n".join(line for line in text.splitlines()
                                     if not line.startswith("## ")))


def publish_transcripts(settings: Settings, records: list[dict]) -> tuple[list[Path], int]:
    """Save notes even when analysis is disabled or fails; select files to open."""
    completed: list[Path] = []
    failures = 0
    for record in records:
        source = record["transcript"]
        target = source
        if settings.notes_dir is not None:
            try:
                target = save_meeting_note(source, settings.notes_dir)
                log.info("Meeting note → %s", target)
            except OSError as exc:
                log.error("Cannot save meeting note for %s: %s", source, exc)
                failures += 1
        # An unchanged batch stays a no-op on the desktop. Explicit inputs
        # still open their saved transcript when its transcription is reused.
        if (settings.inputs or record.get("created", True)) and target not in completed:
            completed.append(target)
    return completed, failures


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        # Every write is atomic, so finished files are intact and a re-run
        # skips them; a traceback here only alarms the user.
        log.error("\nInterrupted. Completed transcripts are saved; "
                  "re-run the same command to continue.")
        return EXIT_INTERRUPTED


def _main(argv: list[str] | None) -> int:
    package_root = Path(__file__).resolve().parent.parent
    try:
        defaults = settings_from_env(package_root / "transcriptions")
    except ConfigError as exc:
        print(f"transcribe: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    parser = build_parser(defaults)
    args = parser.parse_args(argv)

    settings = Settings(
        inputs=args.inputs,
        whisper_model=args.whisper_model, language=args.language,
        device=args.device, compute_type=args.compute_type,
        batch_size=args.batch_size, cpu_threads=args.cpu_threads, vad=args.vad,
        paragraph_gap=args.paragraph_gap, timestamps=args.timestamps,
        force=args.force,
        gemini_models=tuple(m.strip() for m in args.gemini_models.split(",") if m.strip())
                      if args.gemini_models else defaults.gemini_models,
        gemini_timeout=args.gemini_timeout,
        no_analysis=args.no_analysis, analyse_only=args.analyse_only,
        api_key=defaults.api_key,
        # Config values are data, not shell, so '~' arrives unexpanded and would
        # otherwise create a literal '~' directory in the current folder.
        output_dir=args.output_dir.expanduser(), meeting_date=args.meeting_date,
        notes_dir=args.notes_dir, open_transcript=args.open_transcript,
        verbose=args.verbose, quiet=args.quiet,
        log_file=args.log_file.expanduser() if args.log_file else None,
    )

    try:
        settings.validate()
        configure_logging(settings)
        inputs = collect_inputs(settings, package_root / "audio")
    except ConfigError as exc:
        parser.error(str(exc))
        return EXIT_USAGE

    if not inputs:
        log.error("No audio files found. Pass paths, or put files in %s.",
                  package_root / "audio")
        return EXIT_FAILURE

    if settings.analyse_only:
        records, failures = collect_transcripts(settings, inputs)
    else:
        records, failures = run_transcription(settings, inputs)

    if not records:
        log.error("No transcripts were produced; nothing to analyse.")
        return EXIT_FAILURE

    created: list[Path] = []
    completed, save_failures = publish_transcripts(settings, records)
    failures += save_failures
    if settings.no_analysis:
        log.info("--no-analysis given; skipping Gemini.")
    elif not settings.api_key and settings.analyse_only:
        # Analysis is the whole point of --analyse-only; don't report success.
        log.error("--analyse-only needs GEMINI_API_KEY; add it to %s.", config_path())
        failures += 1
    elif not settings.api_key:
        log.warning("GEMINI_API_KEY is not set; skipping analysis. "
                    "Add it to %s to enable it.", config_path())
    else:
        created, analyse_failures = run_analysis(settings, records)
        failures += analyse_failures

    if created:
        notify("Transcription complete",
               "Analysis saved to " + ", ".join(f.name for f in created))
    elif any(record.get("created") for record in records):
        # An unchanged batch (e.g. a cron re-run) stays silent on the desktop.
        notify("Transcription complete",
               f"Transcript(s) saved to {settings.output_dir.name}/")

    if settings.open_transcript:
        for transcript in completed:
            open_markdown(transcript)

    if failures:
        log.error("Finished with %d failure(s).", failures)
        return EXIT_FAILURE
    log.info("All files processed.")
    return EXIT_OK
