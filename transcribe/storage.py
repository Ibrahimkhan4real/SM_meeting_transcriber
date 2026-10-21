"""Durable, concurrency-safe output storage.

Every write goes to a temporary file in the destination directory and is then
renamed into place, so a crash or a full disk can never leave a half-written
file behind. Index updates additionally take an exclusive lock, because they
are read-modify-write and two concurrent runs would otherwise lose a row.
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

INDEX_NAME = "meetings_index.md"
INDEX_HEADER = "| Date | Folder | Summary |\n|------|--------|--------|\n"
SOURCE_MARKER = ".source"
FOLDER_PREFIX = "supervisor_meeting_"

_ROW_FOLDER = re.compile(r"\]\(([^)]+)/\)")


def atomic_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write `text` to `path` atomically.

    The temporary file is created in the same directory so the rename stays
    within one filesystem, which is what makes it atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def directory_lock(directory: Path, name: str = ".transcribe.lock"):
    """Hold an exclusive lock for the output directory.

    Guards the index read-modify-write and folder reservation against a second
    concurrent run. The lock file itself is never deleted: removing it would
    race with another process that already has it open.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / name
    handle = open(lock_path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def read_source_marker(folder: Path) -> str | None:
    try:
        return (folder / SOURCE_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def analysis_dirs(parent: Path, source_stem: str) -> list[Path]:
    """Analysis folders marked as this transcript's, newest first."""
    return [candidate for candidate in sorted(
                parent.glob(f"{FOLDER_PREFIX}*"),
                key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
            if candidate.is_dir() and read_source_marker(candidate) == source_stem]


def find_existing_dir(parent: Path, source_stem: str) -> Path | None:
    """The analysis folder already belonging to this transcript, if any."""
    folders = analysis_dirs(parent, source_stem)
    return folders[0] if folders else None


def reserve_output_dir(parent: Path, date_str: str, source_stem: str, *,
                       reuse: bool = True) -> Path:
    """Return this meeting's folder, never clobbering a different meeting.

    Re-running the same transcript reuses its folder; a second, different
    meeting on the same date becomes '_2', '_3', and so on. Callers hold
    `directory_lock` so the check-then-create cannot race.
    """
    existing = find_existing_dir(parent, source_stem)
    if reuse and existing is not None:
        return existing

    suffix = 0
    while True:
        name = f"{FOLDER_PREFIX}{date_str}" + (f"_{suffix + 1}" if suffix else "")
        candidate = parent / name
        if not candidate.exists():
            candidate.mkdir(parents=True)
            atomic_write(candidate / SOURCE_MARKER, source_stem + "\n")
            return candidate
        suffix += 1


def table_cell(text: str) -> str:
    """Make a string safe inside a Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def _is_data_row(line: str) -> bool:
    return (line.startswith("|")
            and not line.startswith("| Date")
            and not set(line) <= set("|- "))


def _split_index(text: str) -> tuple[list[str], list[str], list[str]]:
    """(lines before the table, table lines, lines after it).

    The table runs from the first line starting with '|' to the last such line
    before any other text; blank lines inside it are part of it, so rows split
    into groups still form one table (a refreshed row cannot be duplicated).
    Anything the user wrote around it (a title, a note) used to be dropped on
    every update.
    """
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith("|")), len(lines))
    end = i = start
    while i < len(lines) and (lines[i].strip().startswith("|") or not lines[i].strip()):
        if lines[i].strip():
            end = i + 1
        i += 1
    return lines[:start], lines[start:end], lines[end:]


def index_has_row(index_path: Path, folder_name: str) -> bool:
    try:
        text = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    for raw in _split_index(text)[1]:
        match = _ROW_FOLDER.search(raw)
        if _is_data_row(raw.strip()) and match and match.group(1) == folder_name:
            return True
    return False


def write_index_row(index_path: Path, folder_name: str, date_str: str, summary: str) -> bool:
    """Insert or replace one index row, keeping the file sorted by date.

    Rows are keyed on the folder rather than the date: two different meetings
    can legitimately fall on the same day. Returns True if the row is new.
    """
    row = f"| {date_str} | [{folder_name}]({folder_name}/) | {table_cell(summary)} |"
    before: list[str] = []
    table: list[str] = []
    after: list[str] = []
    rows: list[str] = []
    was_new = True

    if index_path.exists():
        before, table, after = _split_index(index_path.read_text(encoding="utf-8"))
        for raw in table:
            line = raw.strip()
            if not _is_data_row(line):
                continue
            match = _ROW_FOLDER.search(line)
            if match and match.group(1) == folder_name:
                was_new = False
                continue
            rows.append(line)

    rows.append(row)
    rows.sort(key=lambda r: r.split("|")[1].strip())
    if before and not table and before[-1].strip():
        before.append("")                   # a table needs a blank line above it
    text = "".join(line + "\n" for line in before)
    text += INDEX_HEADER + "\n".join(rows) + "\n"
    text += "".join(line + "\n" for line in after)
    atomic_write(index_path, text)
    return was_new
