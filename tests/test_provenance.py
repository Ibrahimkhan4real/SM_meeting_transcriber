"""Output ownership and complete/current analysis regressions; synthetic data only."""
from pathlib import Path
import hashlib
import json

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from transcribe.analyse import Analysis
from transcribe.cli import run_analysis, run_transcription
from transcribe.config import Settings
from transcribe.provenance import (MANIFEST, allocate_transcript, analysis_current,
                                  record_transcript, source_signature,
                                  transcript_current)
from transcribe.storage import reserve_output_dir


def source(root, name, text='synthetic audio'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_stems_extensions_repeat_and_legacy(tmp_path):
    output = tmp_path / 'out'; output.mkdir()
    legacy = output / 'meeting.md'; legacy.write_text('user-owned legacy content')
    paths = [source(tmp_path, name) for name in ['a/meeting.wav', 'b/meeting.wav', 'a/meeting.mp3']]
    targets = [allocate_transcript(path, output) for path in paths]
    assert len(set(targets)) == 3
    assert legacy not in targets
    targets[0].write_text('new generated transcript')
    signature = source_signature(paths[0], Settings())
    record_transcript(paths[0], targets[0], signature)
    assert allocate_transcript(paths[0], output) == targets[0]
    assert legacy.read_text() == 'user-owned legacy content'


def test_content_settings_and_user_edit_freshness(tmp_path):
    audio = source(tmp_path, 'meeting.wav')
    target = allocate_transcript(audio, tmp_path / 'out')
    target.write_text('generated output')
    settings = Settings()
    signature = source_signature(audio, settings)
    record_transcript(audio, target, signature)
    assert transcript_current(target, signature)
    audio.write_text('changed audio of the same length')
    assert not transcript_current(target, source_signature(audio, settings))
    audio.write_text("synthetic audio")
    assert transcript_current(target, source_signature(audio, settings))
    settings.timestamps = True
    assert not transcript_current(target, source_signature(audio, settings))
    target.write_text('manually edited user transcript')
    assert allocate_transcript(audio, target.parent) != target
    assert target.read_text() == 'manually edited user transcript'


def test_mocked_transcription_collision_repeat_and_rerun(tmp_path):
    paths = [source(tmp_path, 'a/meeting.wav'), source(tmp_path, 'b/meeting.wav')]
    settings = Settings(output_dir=tmp_path / 'out', quiet=True)
    result = SimpleNamespace(duration=12, language='en', segments=[SimpleNamespace(text='Synthetic speech.', start=0, words=[])])
    with patch('transcribe.engine.Transcriber') as engine:
        engine.return_value.device = 'cpu'; engine.return_value.transcribe.return_value = result
        records, failures = run_transcription(settings, paths + [paths[0]])
        assert failures == 0 and len(records) == 2
        assert len({r['transcript'] for r in records}) == 2
        assert engine.return_value.transcribe.call_count == 2
        again, failures = run_transcription(settings, paths)
        assert failures == 0 and all(not r['created'] for r in again)
        assert engine.call_count == 1


def analysis_run(transcript, tmp_path, mock):
    return run_analysis(Settings(output_dir=tmp_path), [{'transcript': transcript, 'date': '2026-10-02', 'created': True}])


def response():
    return Analysis(model='mock', raw='{}', executive_summary='Synthetic summary.', meeting_notes='Synthetic notes.', meeting_points=['Point'], action_points=['Action'])


def test_analysis_interrupted_recovery_current_and_changed(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    reserve_output_dir(tmp_path, '2026-10-02', 'meeting')  # interrupted marker-only folder
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        created, failures = analysis_run(transcript, tmp_path, factory)
        assert failures == 0 and len(created) == 1
        assert analysis_current(created[0], transcript)
        assert len(analysis_run(transcript, tmp_path, factory)[0]) == 0
        assert factory.return_value.analyse.call_count == 1
        transcript.write_text('Changed synthetic transcript')
        regenerated, failures = analysis_run(transcript, tmp_path, factory)
        assert failures == 0 and len(regenerated) == 1
        assert analysis_current(regenerated[0], transcript)
        assert factory.return_value.analyse.call_count == 2


def test_legacy_and_edited_analyses_preserved(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    legacy = reserve_output_dir(tmp_path, '2026-10-02', 'meeting')
    old = legacy / 'analysis_raw.md'; old.write_text('User legacy analysis')
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        created, failures = analysis_run(transcript, tmp_path, factory)
        assert failures == 0 and created[0] != legacy
        assert old.read_text() == 'User legacy analysis'
        edited = created[0] / 'analysis_raw.json'; edited.write_text('User edited analysis')
        regenerated, failures = analysis_run(transcript, tmp_path, factory)
        assert failures == 0 and regenerated[0] != created[0]
        assert edited.read_text() == 'User edited analysis'
        assert len(analysis_run(transcript, tmp_path, factory)[0]) == 0


def test_failed_commit_never_marks_complete_and_recovers(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        with patch('transcribe.cli.complete_analysis', side_effect=OSError('mock disk full')):
            created, failures = analysis_run(transcript, tmp_path, factory)
        assert created == [] and failures == 1
        assert not list(tmp_path.glob('*/' + MANIFEST))
        created, failures = analysis_run(transcript, tmp_path, factory)
        assert failures == 0 and analysis_current(created[0], transcript)


def test_unmarked_analysis_folder_never_adopted(tmp_path):
    legacy = tmp_path / 'supervisor_meeting_2026-10-02'; legacy.mkdir()
    old = legacy / 'notes.md'; old.write_text('Legacy notes')
    made = reserve_output_dir(tmp_path, '2026-10-02', 'meeting')
    assert made != legacy and old.read_text() == 'Legacy notes'


def test_missing_expected_output_regenerates_without_false_success(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        created, failures = analysis_run(transcript, tmp_path, factory)
        missing = next(created[0].glob('*_action_points.md'))
        missing.unlink()
        assert not analysis_current(created[0], transcript)
        recovered, failures = analysis_run(transcript, tmp_path, factory)
        assert failures == 0 and analysis_current(recovered[0], transcript)
        assert factory.return_value.analyse.call_count == 2


@pytest.mark.parametrize('manifest', [
    {'unexpected': 1},
    {'transcript_sha256': 'a' * 64, 'outputs': {}},
    {'transcript_sha256': 'a' * 64, 'outputs': {'../analysis_raw.md': 'b' * 64}},
    {'transcript_sha256': 'a' * 64, 'outputs': {'analysis_raw.md': 'bad-hash'}},
])
def test_malformed_manifest_preserves_existing_analysis(tmp_path, manifest):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    old = reserve_output_dir(tmp_path, '2026-10-02', 'meeting')
    raw = old / 'analysis_raw.json'; raw.write_text('User-owned raw analysis')
    marker = old / MANIFEST; marker.write_text(json.dumps(manifest))
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        created, failures = analysis_run(transcript, tmp_path, factory)
    assert failures == 0 and created[0] != old
    assert raw.read_text() == 'User-owned raw analysis'
    assert json.loads(marker.read_text()) == manifest


def test_unowned_future_output_target_preserved(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    with patch('transcribe.analyse.Analyser') as factory:
        original = response(); original.action_points = []
        factory.return_value.analyse.return_value = original
        created, failures = analysis_run(transcript, tmp_path, factory)
        folder = created[0]
        manual = folder / f'{folder.name}_action_points.md'
        manual.write_text('User manually added action points')
        transcript.write_text('Changed synthetic transcript')
        factory.return_value.analyse.return_value = response()
        regenerated, failures = analysis_run(transcript, tmp_path, factory)
    assert failures == 0 and regenerated[0] != folder
    assert manual.read_text() == 'User manually added action points'


def test_manifest_digest_matches_exact_submitted_bytes(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Original synthetic transcript')
    original = transcript.read_bytes()
    with patch('transcribe.analyse.Analyser') as factory:
        def analyse(text):
            assert text == original.decode('utf-8')
            transcript.write_text('Different content arriving during analysis')
            return response()
        factory.return_value.analyse.side_effect = analyse
        created, failures = analysis_run(transcript, tmp_path, factory)
    assert failures == 0
    manifest = json.loads((created[0] / MANIFEST).read_text())
    assert manifest['transcript_sha256'] == hashlib.sha256(original).hexdigest()
    assert not analysis_current(created[0], transcript)


def test_invalid_transcript_encoding_counts_failure_without_provider_call(tmp_path):
    transcript = tmp_path / 'meeting.md'; transcript.write_bytes(b'\xff')
    with patch('transcribe.analyse.Analyser') as factory:
        created, failures = analysis_run(transcript, tmp_path, factory)
        factory.return_value.analyse.assert_not_called()
    assert created == [] and failures == 1


def test_invalid_extra_manifest_entry_not_ignored(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    old = reserve_output_dir(tmp_path, '2026-10-02', 'meeting')
    raw = old / 'analysis_raw.json'; raw.write_text('User raw content')
    manifest = {'transcript_sha256': hashlib.sha256(transcript.read_bytes()).hexdigest(),
                'outputs': {'analysis_raw.json': hashlib.sha256(raw.read_bytes()).hexdigest(),
                            '../outside.md': 'a' * 64}}
    (old / MANIFEST).write_text(json.dumps(manifest))
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        created, failures = analysis_run(transcript, tmp_path, factory)
    assert failures == 0 and created[0] != old
    assert raw.read_text() == 'User raw content'


def test_silent_transcript_is_not_sent_for_analysis(tmp_path):
    from transcribe.audio import write_transcript
    transcript = tmp_path / 'silence.md'
    write_transcript(transcript, 'silence.wav', '', model='tiny', language='en',
                     duration=5, paragraphs=0, device='cpu')
    with patch('transcribe.analyse.Analyser') as factory:
        created, failures = analysis_run(transcript, tmp_path, factory)
        factory.return_value.analyse.assert_not_called()
    assert created == [] and failures == 0
    assert not list(tmp_path.glob('supervisor_meeting_*'))


def fake_engine(engine, text):
    engine.return_value.device = 'cpu'
    engine.return_value.transcribe.return_value = SimpleNamespace(
        duration=12, language='en', segments=[SimpleNamespace(text=text, start=0, words=[])])


def test_new_recording_under_the_same_name_keeps_the_earlier_meeting(tmp_path):
    """A phone app's fixed file name must not overwrite last week's meeting."""
    audio = source(tmp_path, 'Recording.wav', 'first meeting audio')
    settings = Settings(output_dir=tmp_path / 'out', quiet=True)
    with patch('transcribe.engine.Transcriber') as engine:
        fake_engine(engine, 'First meeting.')
        first = run_transcription(settings, [audio])[0][0]['transcript']
        audio.write_text('second meeting audio')
        fake_engine(engine, 'Second meeting.')
        second = run_transcription(settings, [audio])[0][0]['transcript']
        audio.write_text('first meeting audio')
        again = run_transcription(settings, [audio])[0][0]
    assert second != first and 'First meeting.' in first.read_text()
    assert 'Second meeting.' in second.read_text()
    assert again['transcript'] == first and again['created'] is False


def test_analysis_folder_of_another_date_is_never_overwritten(tmp_path):
    transcript = source(tmp_path, 'Recording.md', 'Week one transcript')
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        week_one = run_analysis(Settings(), [{'transcript': transcript, 'date': '2026-10-05'}])[0][0]
        kept = {p.name: p.read_bytes() for p in week_one.iterdir()}
        transcript.write_text('Week two transcript')
        week_two = run_analysis(Settings(), [{'transcript': transcript, 'date': '2026-10-12'}])[0][0]
    assert week_two.name == 'supervisor_meeting_2026-10-12'
    assert {p.name: p.read_bytes() for p in week_one.iterdir()} == kept
    rows = [l for l in (tmp_path / 'meetings_index.md').read_text().splitlines() if l.startswith('| 2026')]
    assert len(rows) == 2 and all(l.split('|')[1].strip() in l.split('|')[2] for l in rows)


def test_reanalysis_with_fewer_sections_removes_the_stale_one(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        folder = analysis_run(transcript, tmp_path, factory)[0][0]
        stale = folder / f'{folder.name}_action_points.md'
        assert stale.is_file()
        fewer = response(); fewer.action_points = []
        factory.return_value.analyse.return_value = fewer
        again = run_analysis(Settings(force=True),
                             [{'transcript': transcript, 'date': '2026-10-02'}])[0][0]
    assert again == folder and not stale.exists()
    assert analysis_current(folder, transcript)


def test_index_row_follows_the_manifest_and_is_repaired_later(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        with patch('transcribe.cli.write_index_row', side_effect=OSError('mock disk full')):
            created, failures = analysis_run(transcript, tmp_path, factory)
        assert failures == 1 and analysis_current(created[0], transcript)
        assert not (tmp_path / 'meetings_index.md').exists()
        again, failures = analysis_run(transcript, tmp_path, factory)
        assert again == [] and failures == 0
        assert factory.return_value.analyse.call_count == 1      # no second Gemini call
    index = (tmp_path / 'meetings_index.md').read_text()
    assert f'[{created[0].name}]' in index and 'Synthetic summary.' in index


def test_failed_commit_leaves_no_index_row_for_the_unfinished_folder(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        with patch('transcribe.cli.complete_analysis', side_effect=OSError('mock disk full')):
            analysis_run(transcript, tmp_path, factory)
        assert not (tmp_path / 'meetings_index.md').exists()
        recovered = analysis_run(transcript, tmp_path, factory)[0][0]
    rows = [l for l in (tmp_path / 'meetings_index.md').read_text().splitlines() if l.startswith('| 2026')]
    assert len(rows) == 1 and f'[{recovered.name}]' in rows[0]


def test_index_that_is_not_utf8_is_reported_not_a_traceback(tmp_path):
    transcript = source(tmp_path, 'meeting.md', 'Synthetic transcript')
    index = tmp_path / 'meetings_index.md'
    index.write_bytes(b'Caf\xe9 meetings\n')
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        created, failures = analysis_run(transcript, tmp_path, factory)
    assert failures == 1 and analysis_current(created[0], transcript)
    assert index.read_bytes() == b'Caf\xe9 meetings\n'


def test_failed_registry_write_does_not_orphan_the_transcript(tmp_path):
    audio = source(tmp_path, 'meeting.wav')
    settings = Settings(output_dir=tmp_path / 'out', quiet=True)
    with patch('transcribe.engine.Transcriber') as engine:
        fake_engine(engine, 'Synthetic speech.')
        with patch('transcribe.cli.record_transcript', side_effect=OSError('mock disk full')):
            assert run_transcription(settings, [audio]) == ([], 1)
        records, failures = run_transcription(settings, [audio])
    assert failures == 0 and records[0]['transcript'].name == 'meeting.md'
    assert sorted(p.name for p in settings.output_dir.glob('*.md')) == ['meeting.md']


def test_failed_rewrite_keeps_the_old_transcript_recognisably_ours(tmp_path):
    audio = source(tmp_path, 'meeting.wav')
    with patch('transcribe.engine.Transcriber') as engine:
        fake_engine(engine, 'Synthetic speech.')
        run_transcription(Settings(output_dir=tmp_path / 'out', quiet=True), [audio])
        changed = Settings(output_dir=tmp_path / 'out', quiet=True, timestamps=True)
        with patch('transcribe.cli.atomic_write', side_effect=OSError('mock disk full')):
            assert run_transcription(changed, [audio])[1] == 1
        records, failures = run_transcription(changed, [audio])
    assert failures == 0 and records[0]['transcript'].name == 'meeting.md'
    assert '[00:00]' in records[0]['transcript'].read_text()


# --- follow-up: folder choice by provenance, stored dates, index ownership -------

def made_transcript(tmp_path, text='Synthetic transcript', recorded='2026-05-19'):
    """A transcript the tool made, as the registry records it."""
    audio = source(tmp_path, 'audio/chat.wav', 'synthetic audio')
    transcript = allocate_transcript(audio, tmp_path)
    transcript.write_text(text)
    record_transcript(audio, transcript, source_signature(audio, Settings()),
                      meeting_date=recorded)
    return audio, transcript


def rows(tmp_path):
    index = tmp_path / 'meetings_index.md'
    return [l for l in index.read_text().splitlines() if l.startswith('| 20')] if index.exists() else []


def test_same_recording_keeps_its_folder_whatever_date_it_carries(tmp_path):
    """V2: a folder misdated by old rules got a second folder and row on re-analysis."""
    audio, transcript = made_transcript(tmp_path)
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        first = run_analysis(Settings(), [{'transcript': transcript, 'date': '2026-10-18'}])[0][0]
        transcript.write_text('Re-transcribed synthetic transcript')       # e.g. --timestamps
        record_transcript(audio, transcript, source_signature(audio, Settings()))
        again = run_analysis(Settings(), [{'transcript': transcript, 'date': '2026-05-19'}])[0][0]
    assert again == first
    assert [p.name for p in tmp_path.glob('supervisor_meeting_*')] == [first.name]
    assert len(rows(tmp_path)) == 1 and rows(tmp_path)[0].startswith('| 2026-05-19 |')


def test_a_touched_stale_folder_does_not_win_over_the_current_one(tmp_path):
    """The newest folder by mtime is not necessarily this transcript's analysis."""
    import os
    _, transcript = made_transcript(tmp_path)
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        current = run_analysis(Settings(), [{'transcript': transcript, 'date': '2026-05-19'}])[0][0]
        stale = reserve_output_dir(tmp_path, '2026-10-18', transcript.stem, reuse=False)
        raw = stale / 'analysis_raw.json'; raw.write_text('{}')
        (stale / MANIFEST).write_text(json.dumps({
            'transcript_sha256': 'b' * 64,
            'outputs': {raw.name: hashlib.sha256(raw.read_bytes()).hexdigest()}}))
        os.utime(stale, (4_000_000_000, 4_000_000_000))                # newest by far
        created, failures = run_analysis(Settings(), [{'transcript': transcript,
                                                       'date': '2026-05-19'}])
        assert created == [] and failures == 0
        assert factory.return_value.analyse.call_count == 1
    assert analysis_current(current, transcript)


def test_meeting_date_is_stored_and_outlives_the_recording(tmp_path):
    """V3: once the audio left audio/, --analyse-only dated by the transcript."""
    import os
    from transcribe.cli import collect_transcripts
    audio = source(tmp_path, 'audio/supervision.wav')
    os.utime(audio, (1_791_460_800, 1_791_460_800))                 # 2026-10-08
    settings = Settings(output_dir=tmp_path / 'out', quiet=True)
    with patch('transcribe.engine.Transcriber') as engine:
        fake_engine(engine, 'Synthetic speech.')
        transcript = run_transcription(settings, [audio])[0][0]['transcript']
        os.utime(audio, None)                                       # copied, synced, restored
        assert run_transcription(settings, [audio])[0][0]['date'] == '2026-10-08'
    audio.unlink()                                                  # moved out of audio/
    assert collect_transcripts(Settings(quiet=True), [transcript])[0][0]['date'] == '2026-10-08'


def test_index_rows_the_user_removed_are_not_brought_back(tmp_path):
    """V4: the repair cannot tell a row never written from one deleted on purpose."""
    _, transcript = made_transcript(tmp_path)
    record = [{'transcript': transcript, 'date': '2026-05-19'}]
    index = tmp_path / 'meetings_index.md'
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        run_analysis(Settings(), record)
        index.write_text('# Supervision meetings\n')
        assert run_analysis(Settings(), record) == ([], 0)
        assert index.read_text() == '# Supervision meetings\n'
        index.unlink()
        assert run_analysis(Settings(), record) == ([], 0)
    assert not index.exists()


def test_unreadable_index_does_not_fail_a_run_with_nothing_to_do(tmp_path):
    """V5: every no-op cron run exited 1 once the index had a Latin-1 byte."""
    _, transcript = made_transcript(tmp_path)
    record = [{'transcript': transcript, 'date': '2026-05-19'}]
    index = tmp_path / 'meetings_index.md'
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        run_analysis(Settings(), record)
        index.write_bytes(b'# Caf\xe9 log\n' + index.read_bytes())
        assert run_analysis(Settings(), record) == ([], 0)


def test_pending_row_with_an_unreadable_index_only_warns(tmp_path, caplog):
    _, transcript = made_transcript(tmp_path)
    record = [{'transcript': transcript, 'date': '2026-05-19'}]
    index = tmp_path / 'meetings_index.md'
    index.write_bytes(b'# Caf\xe9 log\n')
    with patch('transcribe.analyse.Analyser') as factory:
        factory.return_value.analyse.return_value = response()
        assert run_analysis(Settings(), record)[1] == 1      # this run's own row failed
        assert run_analysis(Settings(), record) == ([], 0)   # a later no-op only warns
    assert 'Could not add the row' in caplog.text
    assert index.read_bytes() == b'# Caf\xe9 log\n'
