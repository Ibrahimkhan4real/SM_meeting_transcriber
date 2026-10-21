"""Atomicity, locking, collision-safe folders, and the meetings index."""

import multiprocessing as mp
import os
from pathlib import Path

import pytest

from transcribe.storage import (INDEX_NAME, atomic_write, directory_lock,
                                find_existing_dir, reserve_output_dir,
                                table_cell, write_index_row)


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "out.md"
    atomic_write(target, "hello")
    assert target.read_text() == "hello"
    assert [p.name for p in tmp_path.iterdir()] == ["out.md"]


def test_atomic_write_preserves_old_content_on_failure(tmp_path):
    target = tmp_path / "index.md"
    atomic_write(target, "original")
    with pytest.raises(ValueError):
        # Simulate a crash midway through producing the new content.
        def boom():
            raise ValueError("disk full")
        atomic_write(target, "".join([boom() for _ in range(1)]))
    assert target.read_text() == "original", "old file must survive a failed write"
    assert list(tmp_path.iterdir()) == [target], "no .tmp litter left behind"


def test_reserve_reuses_folder_for_same_source(tmp_path):
    a = reserve_output_dir(tmp_path, "2026-09-05", "meeting_A")
    b = reserve_output_dir(tmp_path, "2026-09-05", "meeting_A")
    assert a == b


def test_reserve_never_clobbers_a_different_meeting(tmp_path):
    a = reserve_output_dir(tmp_path, "2026-09-05", "meeting_A")
    b = reserve_output_dir(tmp_path, "2026-09-05", "meeting_B")
    c = reserve_output_dir(tmp_path, "2026-09-05", "meeting_C")
    assert len({a, b, c}) == 3
    assert b.name.endswith("_2") and c.name.endswith("_3")


def test_find_existing_dir(tmp_path):
    made = reserve_output_dir(tmp_path, "2026-09-05", "meeting_A")
    assert find_existing_dir(tmp_path, "meeting_A") == made
    assert find_existing_dir(tmp_path, "nope") is None


def test_index_dedupes_by_folder_not_date(tmp_path):
    index = tmp_path / INDEX_NAME
    assert write_index_row(index, "supervisor_meeting_2026-08-25", "2026-08-25", "First.")
    # same date, different meeting -> must be a second row
    assert write_index_row(index, "supervisor_meeting_2026-08-25_2", "2026-08-25", "Second.")
    rows = [l for l in index.read_text().splitlines() if l.startswith("| 2026")]
    assert len(rows) == 2


def test_index_replaces_row_for_same_folder(tmp_path):
    index = tmp_path / INDEX_NAME
    write_index_row(index, "supervisor_meeting_2026-08-25", "2026-08-25", "First.")
    assert not write_index_row(index, "supervisor_meeting_2026-08-25", "2026-08-25", "Updated.")
    rows = [l for l in index.read_text().splitlines() if l.startswith("| 2026")]
    assert len(rows) == 1 and "Updated." in rows[0]


def test_index_stays_sorted(tmp_path):
    index = tmp_path / INDEX_NAME
    for d in ["2026-09-05", "2026-02-23", "2026-05-19"]:
        write_index_row(index, f"supervisor_meeting_{d}", d, "s.")
    dates = [l.split("|")[1].strip() for l in index.read_text().splitlines()
             if l.startswith("| 2026")]
    assert dates == sorted(dates)


def test_pipe_in_summary_cannot_break_the_table():
    assert "\\|" in table_cell("a | b")
    assert "\n" not in table_cell("a\nb")


def _append_rows(args):
    """Worker: add N rows under the directory lock."""
    directory, start = args
    directory = Path(directory)
    for i in range(start, start + 10):
        with directory_lock(directory):
            write_index_row(directory / INDEX_NAME,
                            f"supervisor_meeting_2026-01-{i:02d}", f"2026-01-{i:02d}", "s.")


def test_concurrent_writers_do_not_lose_rows(tmp_path):
    """Two processes interleaving read-modify-write on the index."""
    ctx = mp.get_context("spawn")
    args = [(str(tmp_path), 1), (str(tmp_path), 11)]
    with ctx.Pool(2) as pool:
        pool.map(_append_rows, args)
    rows = [l for l in (tmp_path / INDEX_NAME).read_text().splitlines()
            if l.startswith("| 2026")]
    assert len(rows) == 20, f"lost rows: got {len(rows)} of 20"


def test_index_keeps_lines_the_user_wrote(tmp_path):
    index = tmp_path / INDEX_NAME
    write_index_row(index, "supervisor_meeting_2026-08-25", "2026-08-25", "First.")
    index.write_text("# Supervision meetings\n\n" + index.read_text()
                     + "\nMeeting 3 was cancelled (illness).\n")
    write_index_row(index, "supervisor_meeting_2026-09-01", "2026-09-01", "Second.")
    text = index.read_text()
    assert text.startswith("# Supervision meetings\n\n| Date |")
    assert text.endswith("\nMeeting 3 was cancelled (illness).\n")
    assert "First." in text and "Second." in text


def test_index_without_a_table_gets_one_below_the_text(tmp_path):
    index = tmp_path / INDEX_NAME
    index.write_text("My meetings")
    write_index_row(index, "supervisor_meeting_2026-08-25", "2026-08-25", "First.")
    assert index.read_text().startswith("My meetings\n\n| Date |")


def test_generated_index_format_is_unchanged(tmp_path):
    index = tmp_path / INDEX_NAME
    write_index_row(index, "supervisor_meeting_2026-08-25", "2026-08-25", "First.")
    write_index_row(index, "supervisor_meeting_2026-08-25", "2026-08-25", "Again.")
    assert index.read_text() == ("| Date | Folder | Summary |\n|------|--------|--------|\n"
                                 "| 2026-08-25 | [supervisor_meeting_2026-08-25]"
                                 "(supervisor_meeting_2026-08-25/) | Again. |\n")


def test_index_rows_split_by_a_blank_line_stay_one_table(tmp_path):
    """V7: a refreshed row went into the first group; the old one stayed below."""
    from transcribe.storage import index_has_row
    index = tmp_path / INDEX_NAME
    index.write_text("| Date | Folder | Summary |\n|------|--------|--------|\n"
                     "| 2026-08-25 | [supervisor_meeting_2026-08-25](supervisor_meeting_2026-08-25/) | A. |\n"
                     "\n"
                     "| 2026-10-08 | [supervisor_meeting_2026-10-08](supervisor_meeting_2026-10-08/) | B. |\n"
                     "\nNotes below.\n")
    assert index_has_row(index, "supervisor_meeting_2026-10-08")
    write_index_row(index, "supervisor_meeting_2026-10-08", "2026-10-08", "B again.")
    text = index.read_text()
    assert text.count("[supervisor_meeting_2026-10-08]") == 1 and "B again." in text
    assert "A." in text and text.endswith("\nNotes below.\n")
