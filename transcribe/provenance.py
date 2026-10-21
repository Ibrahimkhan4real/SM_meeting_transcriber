"""Source ownership and content fingerprints for safely reusable outputs."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from .storage import atomic_write, directory_lock

REGISTRY = ".transcript-provenance.json"
MANIFEST = ".analysis-complete.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def source_signature(source: Path, settings) -> dict:
    fields = ("whisper_model", "language", "device", "compute_type", "batch_size",
              "cpu_threads", "vad", "paragraph_gap", "timestamps")
    return {"audio_sha256": digest(source),
            "settings": {field: getattr(settings, field) for field in fields}}


def _audio_of(entry: dict) -> str | None:
    signature = entry.get("signature")
    audio = signature.get("audio_sha256") if isinstance(signature, dict) else None
    return audio if isinstance(audio, str) else None


def _owned_hashes(entry: dict) -> set[str]:
    return {value for value in (entry.get("transcript_sha256"), entry.get("previous_sha256"))
            if isinstance(value, str)}


def allocate_transcript(source: Path, output: Path, audio_sha256: str | None = None) -> Path:
    """Reserve one filename per recording; never claim existing legacy files.

    A different recording saved under the same path (a phone app's fixed
    "Recording.m4a") is a new meeting: it gets its own name, so the earlier
    meeting's transcript and analysis are kept instead of being overwritten.
    """
    identity = str(source.resolve())
    with directory_lock(output):
        registry = read_json(output / REGISTRY)
        for name, entry in registry.items():
            if (Path(name).name == name and name.endswith(".md")
                    and isinstance(entry, dict) and entry.get("source") == identity):
                recorded = _audio_of(entry)
                if audio_sha256 and recorded and recorded != audio_sha256:
                    continue
                target = output / name
                if not target.exists() or digest(target) in _owned_hashes(entry):
                    return target
        number = 1
        while True:
            name = source.stem + (f"_{number}" if number > 1 else "") + ".md"
            if name not in registry and not (output / name).exists():
                registry[name] = {"source": identity}
                atomic_write(output / REGISTRY, json.dumps(registry, indent=2) + "\n")
                return output / name
            number += 1


def transcript_current(target: Path, signature: dict) -> bool:
    entry = read_json(target.parent / REGISTRY).get(target.name, {})
    return (isinstance(entry, dict) and entry.get("signature") == signature
            and target.is_file() and entry.get("transcript_sha256") == digest(target))


def recorded_source(transcript: Path) -> Path | None:
    """The recording this transcript was made from, if it is still there unchanged."""
    entry = read_json(transcript.parent / REGISTRY).get(transcript.name)
    if not isinstance(entry, dict) or not isinstance(entry.get("source"), str):
        return None
    source = Path(entry["source"])
    expected = _audio_of(entry)
    try:
        if source.is_file() and (expected is None or digest(source) == expected):
            return source
    except OSError:
        pass
    return None


def recorded_date(transcript: Path) -> str | None:
    """The meeting date stored when this transcript was made, if any.

    It outlives the recording: once the audio is moved out of audio/, its
    mtime can no longer date the meeting.
    """
    entry = read_json(transcript.parent / REGISTRY).get(transcript.name)
    value = entry.get("date") if isinstance(entry, dict) else None
    try:
        return date.fromisoformat(value).isoformat() if isinstance(value, str) else None
    except ValueError:
        return None


def transcript_hashes(transcript: Path) -> set[str]:
    """Hashes this transcript had as the tool's own output (now and before
    its last re-transcription)."""
    entry = read_json(transcript.parent / REGISTRY).get(transcript.name)
    return _owned_hashes(entry) if isinstance(entry, dict) else set()


def record_transcript(source: Path, target: Path, signature: dict,
                      transcript_sha256: str | None = None,
                      meeting_date: str | None = None) -> None:
    """Record ownership of `target`.

    The CLI records the new text's hash before writing it: an interrupted
    write then leaves either the old file or the new one, and both stay
    recognisably ours instead of turning into an unowned legacy file.
    """
    with directory_lock(target.parent):
        registry = read_json(target.parent / REGISTRY)
        entry = {"source": str(source.resolve()), "signature": signature,
                 "transcript_sha256": transcript_sha256 or digest(target)}
        if meeting_date:
            entry["date"] = meeting_date
        old = registry.get(target.name)
        previous = old.get("transcript_sha256") if isinstance(old, dict) else None
        if isinstance(previous, str) and previous != entry["transcript_sha256"]:
            entry["previous_sha256"] = previous
        registry[target.name] = entry
        atomic_write(target.parent / REGISTRY, json.dumps(registry, indent=2) + "\n")


def owned_outputs(manifest: dict) -> dict | None:
    outputs = manifest.get("outputs")
    expected = manifest.get("transcript_sha256")
    valid_hash = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    if not valid_hash(expected) or not isinstance(outputs, dict) or not outputs:
        return None
    if not any(name in outputs for name in ("analysis_raw.md", "analysis_raw.json")):
        return None
    if not all(isinstance(name, str) and name not in ("", ".", "..", MANIFEST)
               and Path(name).name == name and valid_hash(value)
               for name, value in outputs.items()):
        return None
    return outputs


def analysis_owned(folder: Path, target_names: list[str]) -> bool:
    """Only overwrite targets whose valid manifest proves unchanged ownership."""
    outputs = owned_outputs(read_json(folder / MANIFEST))
    if outputs is None:
        return False
    try:
        if not all((folder / name).is_file() and not (folder / name).is_symlink()
                   and digest(folder / name) == expected
                   for name, expected in outputs.items()):
            return False
        return all(not (folder / name).exists() or name in outputs for name in target_names)
    except OSError:
        return False


def analysis_current(folder: Path, transcript: Path) -> bool:
    manifest = read_json(folder / MANIFEST)
    outputs = owned_outputs(manifest)
    try:
        return (outputs is not None and manifest.get("transcript_sha256") == digest(transcript)
                and analysis_owned(folder, []))
    except OSError:
        return False


def manifest_hash(folder: Path) -> str | None:
    """The transcript hash this folder's analysis was made from."""
    value = read_json(folder / MANIFEST).get("transcript_sha256")
    return value if isinstance(value, str) else None


def complete_analysis(folder: Path, transcript_digest: str, outputs: list[Path]) -> None:
    # "indexed" turns True once meetings_index.md has the row. Only a folder
    # still marked False gets its row added by a later run; a row the user
    # deleted (or a manifest from before this flag) is left alone.
    manifest = {"transcript_sha256": transcript_digest,
                "outputs": {path.name: digest(path) for path in outputs},
                "indexed": False}
    atomic_write(folder / MANIFEST, json.dumps(manifest, indent=2) + "\n")


def index_pending(folder: Path) -> bool:
    return read_json(folder / MANIFEST).get("indexed") is False


def mark_indexed(folder: Path) -> None:
    manifest = read_json(folder / MANIFEST)
    if manifest.get("indexed") is False:
        manifest["indexed"] = True
        atomic_write(folder / MANIFEST, json.dumps(manifest, indent=2) + "\n")
