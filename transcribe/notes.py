"""Save transcripts in the meeting-notes repository without replacing notes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def save_meeting_note(transcript: Path, notes_dir: Path) -> Path:
    """Copy the original Markdown, reusing identical copies and keeping edits.

    Publishing a fully written temporary file with a hard link is atomic and
    cannot overwrite a competing writer's file. No lock or manifest is left
    behind in the notes repository.
    """
    content = transcript.read_bytes()
    notes_dir = notes_dir.expanduser().resolve()
    notes_dir.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=notes_dir, prefix=".transcribe-",
                                       suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        suffix = 1
        while True:
            filename = transcript.name if suffix == 1 else (
                f"{transcript.stem}_{suffix}{transcript.suffix}")
            target = notes_dir / filename
            try:
                os.link(temporary, target)
                return target
            except FileExistsError:
                try:
                    if target.read_bytes() == content:
                        return target
                except FileNotFoundError:
                    if not os.path.lexists(target):
                        continue            # removed meanwhile: retry this name
                    # A dangling symlink: retrying this name used to spin forever.
                except OSError:
                    pass                    # a folder, a symlink loop, unreadable
                suffix += 1
    finally:
        temporary.unlink(missing_ok=True)
