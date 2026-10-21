"""Config-file parsing (as data, never executed) and validation."""

import pytest

from transcribe.config import ConfigError, Settings, load_config_file


def test_parses_export_and_quotes(tmp_path):
    f = tmp_path / "gemini.env"
    f.write_text('export GEMINI_API_KEY="abc123"\nTRANSCRIBE_WHISPER_MODEL=medium\n')
    got = load_config_file(f)
    assert got["GEMINI_API_KEY"] == "abc123"
    assert got["TRANSCRIBE_WHISPER_MODEL"] == "medium"


def test_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / "c.env"
    f.write_text("# comment\n\n  \nKEY=value\n")
    assert load_config_file(f) == {"KEY": "value"}


def test_config_is_data_not_code(tmp_path):
    """A command substitution must be inert, not executed."""
    marker = tmp_path / "pwned"
    f = tmp_path / "c.env"
    f.write_text(f'KEY=$(touch {marker})\n')
    load_config_file(f)
    assert not marker.exists(), "config file content must never be executed"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_config_file(tmp_path / "nope.env") == {}


def test_malformed_line_does_not_crash(tmp_path):
    f = tmp_path / "c.env"
    f.write_text('BAD="unbalanced\nGOOD=ok\n')
    assert load_config_file(f).get("GOOD") == "ok"


@pytest.mark.parametrize("kwargs", [
    {"whisper_model": "enormous"},
    {"device": "tpu"},
    {"compute_type": "float64"},
    {"paragraph_gap": -1},
    {"meeting_date": "22-04-26"},
    {"meeting_date": "2026-13-45"},
    {"paragraph_gap": float("nan")},
    {"paragraph_gap": float("inf")},
    {"gemini_models": ()},
    {"quiet": True, "verbose": True},
])
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ConfigError):
        Settings(**kwargs).validate()


def test_valid_settings_pass():
    Settings().validate()
    Settings(whisper_model="large-v3", device="cuda", compute_type="float16",
             meeting_date="2026-04-22").validate()


@pytest.mark.parametrize("model", ["tiny.en", "base.en", "small.en", "medium.en"])
def test_english_only_models_are_accepted(model):
    """faster-whisper ships these; a cached medium.en used to be refused."""
    Settings(whisper_model=model).validate()


def test_skipped_config_line_is_reported_without_its_value(tmp_path, caplog):
    f = tmp_path / "gemini.env"
    f.write_text('# comment\nexport GEMINI_API_KEY="secret-part\nnot an assignment\nGOOD=ok\n')
    assert load_config_file(f) == {"GOOD": "ok"}
    assert "line 2" in caplog.text and "GEMINI_API_KEY" in caplog.text
    assert "line 3" in caplog.text and "line 1" not in caplog.text
    assert "secret-part" not in caplog.text


@pytest.mark.parametrize("seconds", [0, -5, float("nan"), float("inf")])
def test_invalid_gemini_timeout_is_rejected(seconds):
    with pytest.raises(ConfigError, match="gemini-timeout"):
        Settings(gemini_timeout=seconds).validate()


def test_one_bad_byte_does_not_discard_the_whole_config(tmp_path, caplog):
    f = tmp_path / "gemini.env"
    f.write_bytes(b"# r\xe9union notes\nexport GEMINI_API_KEY=abc123\n")
    assert load_config_file(f)["GEMINI_API_KEY"] == "abc123"
    assert "not valid UTF-8" in caplog.text


def test_unquoted_spaces_keep_the_whole_value_and_warn(tmp_path, caplog):
    f = tmp_path / "gemini.env"
    f.write_text("TRANSCRIBE_NOTES_DIR=/home/me/My Notes/Meetings  # where notes go\n")
    assert load_config_file(f)["TRANSCRIBE_NOTES_DIR"] == "/home/me/My Notes/Meetings"
    assert "unquoted spaces" in caplog.text and "My Notes" not in caplog.text


def test_api_key_is_not_in_the_settings_repr():
    assert "sk-synthetic-secret" not in repr(Settings(api_key="sk-synthetic-secret"))
