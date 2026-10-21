"""Meeting-note publishing, preservation, and CLI completion behavior."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket

import pytest

from transcribe import cli, notify
from transcribe.audio import write_transcript
from transcribe.config import Settings
from transcribe.notes import save_meeting_note


@pytest.fixture
def transcript(tmp_path):
    path = tmp_path / "output" / "Meeting 2026-10-01.md"
    write_transcript(path, "Meeting 2026-10-01.mp4", "First paragraph.\n\nSecond paragraph.",
                     model="small", language="en", duration=90, paragraphs=2,
                     device="cpu")
    return path


def test_note_keeps_existing_transcript_format(transcript, tmp_path):
    destination = tmp_path / "Notes" / "Meeting_Notes"
    note = save_meeting_note(transcript, destination)
    assert note.read_bytes() == transcript.read_bytes()
    assert note.read_text().startswith("# Transcription: Meeting 2026-10-01.mp4\n")
    assert "---\n\nFirst paragraph.\n\nSecond paragraph." in note.read_text()
    assert list(destination.iterdir()) == [note]


def test_rerun_reuses_identical_note_without_rewriting(transcript, tmp_path):
    destination = tmp_path / "notes"
    first = save_meeting_note(transcript, destination)
    before = first.stat()
    assert save_meeting_note(transcript, destination) == first
    assert first.stat().st_ino == before.st_ino
    assert first.stat().st_mtime_ns == before.st_mtime_ns
    assert list(destination.iterdir()) == [first]


def test_preserves_edited_note_and_reuses_changed_transcript_copy(transcript, tmp_path):
    destination = tmp_path / "notes"
    original = save_meeting_note(transcript, destination)
    original.write_text("My edited meeting notes.\n")
    new = save_meeting_note(transcript, destination)
    assert new.name == "Meeting 2026-10-01_2.md"
    assert original.read_text() == "My edited meeting notes.\n"
    assert new.read_bytes() == transcript.read_bytes()
    assert save_meeting_note(transcript, destination) == new
    assert len(list(destination.iterdir())) == 2


def test_concurrent_exports_keep_different_meetings(tmp_path):
    sources = []
    for number in range(4):
        source = tmp_path / str(number) / "meeting.md"
        source.parent.mkdir()
        source.write_text(f"Meeting {number}\n")
        sources.append(source)
    destination = tmp_path / "notes"
    with ThreadPoolExecutor(max_workers=4) as pool:
        saved = list(pool.map(lambda p: save_meeting_note(p, destination), sources))
    assert len(set(saved)) == 4
    assert {p.read_text() for p in saved} == {p.read_text() for p in sources}
    assert len(list(destination.iterdir())) == 4


def test_failed_publish_leaves_no_partial_note(transcript, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("Disk full")
    monkeypatch.setattr("transcribe.notes.os.link", fail)
    destination = tmp_path / "notes"
    with pytest.raises(OSError, match="Disk full"):
        save_meeting_note(transcript, destination)
    assert list(destination.iterdir()) == []
    assert transcript.exists()


@pytest.fixture
def cli_run(transcript, tmp_path, monkeypatch):
    settings = Settings(output_dir=transcript.parent, notes_dir=tmp_path / "notes")
    monkeypatch.setattr(cli, "settings_from_env", lambda _: settings)
    monkeypatch.setattr(cli, "notify", lambda *args: None)
    opened = []
    monkeypatch.setattr(cli, "open_markdown", lambda path: opened.append(path))
    return settings, opened


@pytest.mark.parametrize("options,save_note,open_note", [
    ([], True, True),
    (["--no-open"], True, False),
    (["--no-notes"], False, True),
])
def test_cli_saves_and_opens_without_analysis(transcript, cli_run, options,
                                             save_note, open_note):
    settings, opened = cli_run
    assert cli.main(["--analyse-only", "--no-analysis", *options, str(transcript)]) == 0
    target = settings.notes_dir / transcript.name if save_note else transcript
    assert target.read_bytes() == transcript.read_bytes()
    assert opened == ([target.resolve()] if open_note else [])
    if not save_note:
        assert not settings.notes_dir.exists()


def test_analysis_failure_still_saves_and_opens_transcript(transcript, cli_run,
                                                          monkeypatch):
    settings, opened = cli_run
    settings.api_key = "synthetic-test-key"
    monkeypatch.setattr(cli, "run_analysis", lambda *args: ([], 1))
    assert cli.main(["--analyse-only", str(transcript)]) == 1
    target = settings.notes_dir / transcript.name
    assert target.read_bytes() == transcript.read_bytes()
    assert opened == [target.resolve()]


def test_save_failure_is_reported_and_local_transcript_opens(transcript, cli_run,
                                                            monkeypatch):
    _, opened = cli_run
    def fail(*args):
        raise OSError("Permission denied")
    monkeypatch.setattr(cli, "save_meeting_note", fail)
    assert cli.main(["--analyse-only", "--no-analysis", str(transcript)]) == 1
    assert opened == [transcript]


@pytest.mark.parametrize("created,should_open", [(False, False), (True, True)])
def test_batch_only_opens_new_transcripts(transcript, cli_run, monkeypatch,
                                         created, should_open):
    settings, opened = cli_run
    monkeypatch.setattr(cli, "collect_inputs", lambda *args: [Path("meeting.mp4")])
    monkeypatch.setattr(cli, "run_transcription", lambda *args: (
        [{"transcript": transcript, "date": "2026-10-01", "created": created}], 0))
    assert cli.main(["--no-analysis"]) == 0
    target = settings.notes_dir / transcript.name
    assert opened == ([target.resolve()] if should_open else [])


def test_open_uses_default_markdown_app_with_spaces(transcript, monkeypatch):
    launched = []
    monkeypatch.setattr(notify.sys, "platform", "linux")
    monkeypatch.setattr(notify.shutil, "which", lambda name: (
        "/usr/bin/xdg-open" if name == "xdg-open" else None))
    monkeypatch.setattr(notify.subprocess, "Popen",
                        lambda args, **kwargs: launched.append((args, kwargs)))
    assert notify.open_markdown(transcript)
    assert launched[0][0] == ["/usr/bin/xdg-open", str(transcript.resolve())]
    assert launched[0][1]["start_new_session"] is True


def test_missing_desktop_opener_does_not_fail_saved_transcript(transcript, monkeypatch):
    monkeypatch.setattr(notify.shutil, "which", lambda _: None)
    assert not notify.open_markdown(transcript)
    assert transcript.exists()


def test_gio_opens_using_the_existing_desktop_session(transcript, monkeypatch):
    launched = []
    monkeypatch.setattr(notify.sys, "platform", "linux")
    monkeypatch.setattr(notify.shutil, "which", lambda name: (
        "/usr/bin/gio" if name == "gio" else None))
    monkeypatch.setattr(notify.subprocess, "Popen",
                        lambda args, **kwargs: launched.append(args))
    assert notify.open_markdown(transcript)
    assert launched == [["/usr/bin/gio", "open", str(transcript.resolve())]]


def test_recovers_desktop_sockets_without_overriding_supplied_env(tmp_path, monkeypatch):
    monkeypatch.setattr(notify.sys, "platform", "linux")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    for name in ("DBUS_SESSION_BUS_ADDRESS", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    with socket.socket(socket.AF_UNIX) as bus, socket.socket(socket.AF_UNIX) as wayland:
        bus.bind(str(tmp_path / "bus"))
        wayland.bind(str(tmp_path / "wayland-0"))
        env = notify.desktop_environment()
        assert env["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={tmp_path / 'bus'}"
        assert env["WAYLAND_DISPLAY"] == "wayland-0"
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/supplied/bus")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-7")
        env = notify.desktop_environment()
        assert env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/supplied/bus"
        assert env["WAYLAND_DISPLAY"] == "wayland-7"


def test_notes_settings_config_environment_and_cli_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_config_file", lambda: {
        "TRANSCRIBE_NOTES_DIR": str(tmp_path / "configured"),
        "TRANSCRIBE_OPEN": "false",
    })
    monkeypatch.setenv("TRANSCRIBE_NOTES_DIR", str(tmp_path / "environment"))
    monkeypatch.delenv("TRANSCRIBE_OPEN", raising=False)
    monkeypatch.delenv("TRANSCRIBE_SAVE_NOTES", raising=False)
    defaults = cli.settings_from_env(tmp_path / "output")
    assert defaults.notes_dir == tmp_path / "environment"
    assert defaults.open_transcript is False
    args = cli.build_parser(defaults).parse_args(["--notes-dir", str(tmp_path / "explicit")])
    assert args.notes_dir == tmp_path / "explicit"
    monkeypatch.setenv("TRANSCRIBE_SAVE_NOTES", "false")
    assert cli.settings_from_env(tmp_path / "output").notes_dir is None


@pytest.fixture
def fails_instead_of_hanging():
    """The old code looped forever here; turn a hang into a test failure."""
    import signal
    def hung(*args):
        raise TimeoutError("save_meeting_note did not return within 5 s")
    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(5)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("link_target", ["/nonexistent/old-note.md", "SELF"])
def test_broken_symlink_in_notes_is_kept_and_skipped(transcript, tmp_path,
                                                     fails_instead_of_hanging, link_target):
    destination = tmp_path / "notes"; destination.mkdir()
    occupied = destination / transcript.name
    occupied.symlink_to(occupied.name if link_target == "SELF" else link_target)
    saved = save_meeting_note(transcript, destination)
    assert saved.name == "Meeting 2026-10-01_2.md"
    assert saved.read_bytes() == transcript.read_bytes()
    assert occupied.is_symlink()


def test_macos_notification_passes_text_as_arguments(monkeypatch):
    """Spliced into the script, "Bob's notes" broke it and a quote could inject code."""
    calls = []
    monkeypatch.setattr(notify.shutil, "which", lambda name: (
        "/usr/bin/osascript" if name == "osascript" else None))
    monkeypatch.setattr(notify.subprocess, "run", lambda args, **kw: calls.append(args))
    body = 'Saved to Bob\'s notes/ " & (do shell script "id") & "'
    notify.notify("Transcription complete", body)
    args = calls[0]
    assert args[-2:] == ["Transcription complete", body]
    assert not any(body in part or "Bob" in part for part in args[:-2])
