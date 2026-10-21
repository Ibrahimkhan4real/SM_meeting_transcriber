"""Configuration: defaults, config-file parsing, and validation.

Precedence, lowest to highest: built-in defaults, config file, environment,
command-line flags.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shlex
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DEFAULT_GEMINI_MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash")

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3",
                  "distil-large-v3", "turbo",
                  "tiny.en", "base.en", "small.en", "medium.en")
DEVICES = ("cpu", "cuda", "auto")
COMPUTE_TYPES = ("int8", "int8_float16", "int8_float32", "float16", "float32")

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")

log = logging.getLogger(__name__)


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "transcribe" / "gemini.env"


def load_config_file(path: Path | None = None) -> dict[str, str]:
    """Read KEY=VALUE pairs from the config file.

    Parsed as data, never executed. The previous implementation ran the file
    through `source`, which let anything that could write the file run
    arbitrary shell as the user.
    """
    path = path or config_path()
    values: dict[str, str] = {}
    try:
        data = path.read_bytes()
    except OSError:
        return values
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # One stray byte (say a Latin-1 'é' in a comment) used to discard the
        # whole file silently, key and all.
        log.warning("%s is not valid UTF-8; reading it anyway, check it.", path)
        text = data.decode("utf-8", errors="replace")

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.match(line)
        if not match:
            log.warning("%s line %d is not KEY=VALUE; ignored.", path, number)
            continue
        key, value = match.group(1), match.group(2).strip()
        if not value:
            values[key] = ""
            continue
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            # Skip, don't crash; but say so, or a broken key line reads as
            # "GEMINI_API_KEY is not set". Never echo the value: it may be a key.
            log.warning("%s line %d (%s): unbalanced quotes; ignored.", path, number, key)
            continue
        if len(parts) > 1:
            # A shell would cut "/home/me/My Notes" at the space; the whole
            # value is almost certainly what was meant. Never echo it.
            log.warning("%s line %d (%s): unquoted spaces; quote the value.",
                        path, number, key)
        if parts:
            values[key] = " ".join(parts)
    return values


class ConfigError(ValueError):
    """Raised when a setting is present but not usable."""


def default_notes_dir() -> Path:
    return Path.home() / "Documents" / "workspace" / "Notes" / "Meeting_Notes"


@dataclass
class Settings:
    """Everything one run needs, already validated."""

    inputs: list[str] = field(default_factory=list)

    whisper_model: str = "small"
    language: str = "en"
    device: str = "cpu"
    compute_type: str = "int8"
    vad: bool = True
    paragraph_gap: float = 1.5
    timestamps: bool = False
    force: bool = False
    batch_size: int = 0                   # 0 disables batched inference
    cpu_threads: int = 0                  # 0 = auto

    gemini_models: tuple[str, ...] = DEFAULT_GEMINI_MODELS
    gemini_timeout: float = 300.0         # seconds per Gemini request
    no_analysis: bool = False
    analyse_only: bool = False
    api_key: str | None = field(default=None, repr=False)     # never in a repr or log

    output_dir: Path = Path("transcriptions")
    notes_dir: Path | None = field(default_factory=default_notes_dir)
    open_transcript: bool = True
    meeting_date: str | None = None

    verbose: bool = False
    quiet: bool = False
    log_file: Path | None = None

    def validate(self) -> None:
        if self.whisper_model not in WHISPER_MODELS and "/" not in self.whisper_model:
            raise ConfigError(
                f"unknown --model {self.whisper_model!r}; expected one of "
                f"{', '.join(WHISPER_MODELS)} or a HuggingFace model path")
        if self.device not in DEVICES:
            raise ConfigError(f"--device must be one of {', '.join(DEVICES)}")
        if self.compute_type not in COMPUTE_TYPES:
            raise ConfigError(f"--compute-type must be one of {', '.join(COMPUTE_TYPES)}")
        if not math.isfinite(self.paragraph_gap):
            raise ConfigError("--paragraph-gap must be a finite number of seconds")
        if self.paragraph_gap < 0:
            raise ConfigError("--paragraph-gap cannot be negative")
        if self.batch_size < 0:
            raise ConfigError("--batch-size cannot be negative")
        if self.cpu_threads < 0:
            raise ConfigError("--cpu-threads cannot be negative")
        if self.meeting_date is not None and not DATE_RE.match(self.meeting_date):
            raise ConfigError(f"--date must be YYYY-MM-DD (got {self.meeting_date!r})")
        if self.meeting_date is not None:
            try:
                date.fromisoformat(self.meeting_date)
            except ValueError:
                raise ConfigError(
                    f"--date is not a real date (got {self.meeting_date!r})") from None
        if not self.gemini_models:
            raise ConfigError("--gemini-model needs at least one model name")
        if not (math.isfinite(self.gemini_timeout) and self.gemini_timeout > 0):
            raise ConfigError("--gemini-timeout must be a positive number of seconds")
        if self.quiet and self.verbose:
            raise ConfigError("--quiet and --verbose are mutually exclusive")
