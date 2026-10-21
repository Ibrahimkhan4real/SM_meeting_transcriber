"""Command-line behaviour a person meets directly; synthetic data, no model or network."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from transcribe import cli
from transcribe.config import Settings


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """No real config file, no inherited TRANSCRIBE_* or Gemini settings."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in list(cli.os.environ):
        if name.startswith("TRANSCRIBE_") or name == "GEMINI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr(cli, "notify", lambda *args: None)
    monkeypatch.setattr(cli, "open_markdown", lambda path: None)
    return tmp_path


def write_config(root: Path, text: str) -> None:
    path = root / "config" / "transcribe" / "gemini.env"
    path.parent.mkdir(parents=True)
    path.write_text(text)


def test_bad_number_in_config_is_a_clean_usage_error(isolated_env, capsys):
    write_config(isolated_env, "export TRANSCRIBE_BATCH_SIZE=four\n")
    assert cli.main(["--help"]) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert "TRANSCRIBE_BATCH_SIZE" in err and "'four'" in err
    assert "Traceback" not in err


def test_tilde_in_configured_paths_is_expanded(isolated_env, monkeypatch):
    write_config(isolated_env, "export TRANSCRIBE_OUTPUT_DIR=~/Transcripts\n")
    cwd = isolated_env / "cwd"; cwd.mkdir()
    monkeypatch.chdir(cwd)
    seen = []
    monkeypatch.setattr(cli, "run_transcription",
                        lambda settings, inputs: (seen.append(settings), ([], 0))[1])
    cli.main(["--log-file", "~/logs/run.log", "meeting.wav"])
    home = isolated_env / "home"
    assert seen[0].output_dir == home / "Transcripts"
    assert (home / "logs" / "run.log").is_file()
    assert not (cwd / "~").exists()


def test_unwritable_log_file_is_a_usage_error(isolated_env):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--log-file", str(isolated_env), "meeting.wav"])
    assert exit_info.value.code == cli.EXIT_USAGE


def test_uncreatable_output_dir_fails_cleanly(tmp_path):
    blocker = tmp_path / "not-a-folder"; blocker.write_text("x")
    audio = tmp_path / "meeting.wav"; audio.write_text("synthetic audio")
    records, failures = cli.run_transcription(
        Settings(output_dir=blocker / "out", quiet=True), [audio])
    assert records == [] and failures == 1


def test_folder_argument_is_reported_as_a_folder(tmp_path, caplog):
    records, failures = cli.run_transcription(
        Settings(output_dir=tmp_path / "out", quiet=True), [tmp_path])
    assert records == [] and failures == 1
    assert "is a folder" in caplog.text


def test_analyse_only_without_key_is_a_failure(isolated_env, tmp_path):
    transcript = tmp_path / "meeting.md"; transcript.write_text("Synthetic transcript.")
    assert cli.main(["--analyse-only", "--no-notes", "--no-open", str(transcript)]) == 1


def test_ctrl_c_exits_130_without_traceback(isolated_env, monkeypatch, tmp_path):
    def interrupted(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "run_transcription", interrupted)
    assert cli.main(["--no-notes", str(tmp_path / "meeting.wav")]) == 130


@pytest.mark.parametrize("created,expected", [(False, 0), (True, 1)])
def test_unchanged_batch_sends_no_notification(isolated_env, monkeypatch, tmp_path,
                                               created, expected):
    sent = []
    monkeypatch.setattr(cli, "notify", lambda *args: sent.append(args))
    monkeypatch.setattr(cli, "run_transcription", lambda *args: (
        [{"transcript": tmp_path / "meeting.md", "date": "2026-10-01",
          "created": created}], 0))
    assert cli.main(["--no-notes", "--no-analysis", "meeting.wav"]) == 0
    assert len(sent) == expected


def test_gpu_failure_during_transcription_falls_back_to_cpu(tmp_path):
    """CUDA libraries load on first inference: the driver works, cuBLAS is missing."""
    audio = [tmp_path / name for name in ("a.wav", "b.wav")]
    for path in audio:
        path.write_text(f"synthetic audio {path.name}")
    result = SimpleNamespace(duration=12, language="en", segments=[
        SimpleNamespace(text="Synthetic speech.", start=0, words=[])])
    def missing_cublas(*args, **kwargs):
        raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
    gpu = SimpleNamespace(device="cuda", transcribe=missing_cublas)
    cpu = SimpleNamespace(device="cpu", transcribe=lambda *args, **kwargs: result)
    settings = Settings(output_dir=tmp_path / "out", device="cuda", quiet=True)
    with patch("transcribe.engine.Transcriber", side_effect=[gpu, cpu]) as factory:
        records, failures = cli.run_transcription(settings, audio)
    assert failures == 0 and len(records) == 2
    assert factory.call_count == 2 and factory.call_args.args[1:3] == ("cpu", "int8")
    assert "whisper-small (cpu)" in records[0]["transcript"].read_text()


def test_cpu_runtime_error_is_not_retried(tmp_path):
    audio = tmp_path / "a.wav"; audio.write_text("synthetic audio")
    with patch("transcribe.engine.Transcriber") as factory:
        factory.return_value.device = "cpu"
        factory.return_value.transcribe.side_effect = RuntimeError("decoder failure")
        records, failures = cli.run_transcription(
            Settings(output_dir=tmp_path / "out", quiet=True), [audio])
    assert records == [] and failures == 1 and factory.call_count == 1


def test_help_shows_only_meaningful_defaults():
    text = cli.build_parser(Settings()).format_help()
    assert "(default: small)" in text and "(default: 1.5)" in text
    assert "(default: True)" not in text and "(default: None)" not in text
    gemini = text[text.rindex("--gemini-model"):text.rindex("--no-analysis")]
    assert gemini.count("(default:") == 1


def test_unavailable_model_is_a_clean_error(tmp_path, caplog):
    audio = tmp_path / "a.wav"; audio.write_text("synthetic audio")
    with patch("transcribe.engine.Transcriber",
               side_effect=OSError("Cannot find an appropriate cached snapshot folder")):
        records, failures = cli.run_transcription(
            Settings(output_dir=tmp_path / "out", quiet=True), [audio])
    assert records == [] and failures == 1
    assert "Cannot load Whisper model 'small'" in caplog.text


def test_progress_bar_knows_the_total_duration():
    """With a total, tqdm can show a percentage and an ETA for a long file."""
    from transcribe.engine import Transcriber
    segments = [SimpleNamespace(end=40.0), SimpleNamespace(end=100.0)]
    engine = Transcriber.__new__(Transcriber)
    engine.pipeline = None
    engine.model = SimpleNamespace(transcribe=lambda *a, **k: (
        iter(segments), SimpleNamespace(duration=120.0, language="en")))
    totals, steps = [], []
    engine.transcribe(Path("a.wav"), progress=steps.append, on_duration=totals.append)
    assert totals == [120.0] and steps == [40.0, 60.0]


def test_gemini_timeout_comes_from_config_and_flag(isolated_env, monkeypatch):
    write_config(isolated_env, "export TRANSCRIBE_GEMINI_TIMEOUT=45\n")
    seen = []
    monkeypatch.setattr(cli, "run_transcription",
                        lambda settings, inputs: (seen.append(settings), ([], 0))[1])
    cli.main(["meeting.wav"])
    cli.main(["--gemini-timeout", "90", "meeting.wav"])
    assert [s.gemini_timeout for s in seen] == [45.0, 90.0]
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--gemini-timeout", "0", "meeting.wav"])
    assert exit_info.value.code == cli.EXIT_USAGE


def test_analysis_uses_the_configured_timeout(tmp_path):
    transcript = tmp_path / "meeting.md"; transcript.write_text("Synthetic transcript.")
    with patch("transcribe.analyse.Analyser") as factory:
        factory.return_value.analyse.side_effect = RuntimeError("stop here")
        cli.run_analysis(Settings(gemini_timeout=45, quiet=True),
                         [{"transcript": transcript, "date": "2026-10-01"}])
    assert factory.call_args.kwargs["timeout"] == 45


def test_analyse_only_dates_the_meeting_by_its_recording(tmp_path):
    """As a normal run does; the transcript's mtime is only when it was transcribed."""
    import os
    from transcribe.provenance import (allocate_transcript, record_transcript,
                                       source_signature)
    audio = tmp_path / "supervision.wav"; audio.write_text("synthetic audio")
    recorded = 1_791_460_800                 # 2026-10-08 12:00 UTC
    os.utime(audio, (recorded, recorded))
    transcript = allocate_transcript(audio, tmp_path / "out")
    transcript.write_text("Synthetic transcript.")
    record_transcript(audio, transcript, source_signature(audio, Settings()))
    records, failures = cli.collect_transcripts(Settings(quiet=True), [transcript])
    assert failures == 0 and records[0]["date"] == "2026-10-08"
    # Without a known recording it still falls back to the transcript itself.
    loose = tmp_path / "loose_2026-09-30.md"; loose.write_text("Synthetic transcript.")
    assert cli.collect_transcripts(Settings(quiet=True), [loose])[0][0]["date"] == "2026-09-30"


def test_empty_environment_variable_does_not_hide_the_configured_key(isolated_env,
                                                                     monkeypatch):
    write_config(isolated_env, "export GEMINI_API_KEY=configured-test-key\n")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert cli.settings_from_env(isolated_env / "out").api_key == "configured-test-key"


def test_analyse_only_ignores_a_recording_replaced_since(tmp_path):
    """Same name, different audio now: its date is not this transcript's date."""
    import os
    from transcribe.provenance import (allocate_transcript, record_transcript,
                                       source_signature)
    audio = tmp_path / "Recording.wav"; audio.write_text("week one audio")
    transcript = allocate_transcript(audio, tmp_path / "out")
    transcript.write_text("Synthetic transcript.")
    record_transcript(audio, transcript, source_signature(audio, Settings()))
    audio.write_text("week two audio")
    recorded = 1_791_460_800                 # 2026-10-08 12:00 UTC
    os.utime(audio, (recorded, recorded))
    assert cli.collect_transcripts(Settings(quiet=True), [transcript])[0][0]["date"] != "2026-10-08"
