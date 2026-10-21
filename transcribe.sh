#!/usr/bin/env bash
# Bootstrap only: ensure the virtualenv exists and is populated, then hand off
# to the Python package. All logic lives in transcribe/.
set -euo pipefail

# Resolve the real script location even when invoked through a symlink
# (e.g. ~/.local/bin/transcribe -> .../Transcription/transcribe.sh).
_source="${BASH_SOURCE[0]}"
while [ -L "$_source" ]; do
    _dir="$(cd -P "$(dirname "$_source")" && pwd)"
    _source="$(readlink "$_source")"
    [ "${_source:0:1}" = "/" ] || _source="$_dir/$_source"
done
SCRIPT_DIR="$(cd -P "$(dirname "$_source")" && pwd)"

VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
STAMP="$VENV_DIR/.requirements-stamp"

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required but not found in PATH." >&2
    exit 1
fi

# Setup commands run isolated (-I) from /, so nothing in the folder you run
# from (a types.py, a pip/ folder) and no PYTHON* variable can break them.
make_venv() {
    # An existing .venv is never cleared or deleted: it is moved aside, and
    # moved back if the new one cannot be built.
    local old=""
    if [ -e "$VENV_DIR" ] || [ -L "$VENV_DIR" ]; then
        old="$VENV_DIR.broken-$(date +%Y%m%d-%H%M%S)-$$"
        mv "$VENV_DIR" "$old"
    fi
    echo "[setup] Creating virtual environment..." >&2
    if ! (cd / && python3 -I -m venv "$VENV_DIR"); then
        rm -rf "$VENV_DIR"                      # only what this run just made
        [ -z "$old" ] || mv "$old" "$VENV_DIR"
        echo "Error: could not create $VENV_DIR." \
             "On Debian/Ubuntu: sudo apt install python3-venv" >&2
        exit 1
    fi
    [ -z "$old" ] || echo "[setup] The previous environment is kept as $old;" \
                          "delete it once this one works." >&2
}

PYTHON="$VENV_DIR/bin/python"
# Also when its Python is gone (e.g. a dangling link after a system upgrade).
[ -x "$PYTHON" ] || make_venv

# Reinstall only when requirements.txt actually changes, rather than importing
# each package on every run to find out.
if [ -r "$REQUIREMENTS" ]; then
    want="$(cksum <"$REQUIREMENTS")"
    have="$(cat "$STAMP" 2>/dev/null || true)"
    if [ "$want" != "$have" ]; then
        # No pip at all: left half-made by an older failed setup. Only its
        # presence is checked; pip's own config cannot make this fail.
        (cd / && "$PYTHON" -I -c 'import pip') &>/dev/null || make_venv
        echo "[setup] Installing dependencies..." >&2
        (cd / && "$PYTHON" -I -m pip install --quiet --upgrade pip)
        (cd / && "$PYTHON" -I -m pip install --quiet -r "$REQUIREMENTS")
        printf '%s' "$want" >"$STAMP"
    fi
elif ! (cd / && "$PYTHON" -I -c "import faster_whisper") &>/dev/null; then
    echo "[setup] Installing dependencies..." >&2
    (cd / && "$PYTHON" -I -m pip install --quiet faster-whisper tqdm google-genai)
fi

# Put the package on the path without changing directory, so relative file
# arguments given by the user still resolve against their own cwd. Only
# absolute PYTHONPATH entries are passed on: an empty or relative one (":/x",
# ".") would put the current folder back on the path.
_path="$SCRIPT_DIR"
_rest="${PYTHONPATH:-}"
while [ -n "$_rest" ]; do
    _entry="${_rest%%:*}"
    case "$_rest" in *:*) _rest="${_rest#*:}" ;; *) _rest="" ;; esac
    case "$_entry" in /*) _path="$_path:$_entry" ;; esac
done
export PYTHONPATH="$_path"
# Not "python -m": that puts the current directory first on sys.path, so a
# logging.py in whatever folder you run from would be imported instead of the
# real module. (-P / PYTHONSAFEPATH would need Python 3.11+.)
exec "$PYTHON" -c 'import sys; sys.path[:] = [p for p in sys.path if p != ""]
from transcribe.cli import main; sys.exit(main())' "$@"
