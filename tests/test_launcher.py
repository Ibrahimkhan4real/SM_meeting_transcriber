"""transcribe.sh bootstrap, run on a copy in tmp_path; never touches the real .venv."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

SCRIPT = Path(__file__).resolve().parent.parent / "transcribe.sh"

# Stands in for the package: proves which code ran and which `logging` it got.
STUB_CLI = ("import logging\n"
            "def main():\n"
            "    print('stub transcribe; logging from', logging.__file__)\n"
            "    return 0\n")

# A venv interpreter that has pip. Like a real one, a non-isolated "-m pip"
# fails when the current folder holds a types.py (it shadows the stdlib).
WORKING_PYTHON = f"""#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ -e "$PWD/types.py" ]; then
  echo "pip broken by ./types.py" >&2; exit 1
fi
[ "$1" = "-I" ] && shift
case "$1 $2" in "-m pip"|"-c import pip") exit 0 ;; esac
exec {sys.executable} "$@"
"""

BROKEN_PYTHON = '#!/bin/sh\necho "No module named pip" >&2\nexit 1\n'


def make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "transcribe").mkdir(parents=True)
    shutil.copy(SCRIPT, project / "transcribe.sh")
    (project / "requirements.txt").write_text("tqdm\n")
    (project / "transcribe" / "__init__.py").write_text("")
    (project / "transcribe" / "cli.py").write_text(STUB_CLI)
    (project / "transcribe" / "__main__.py").write_text(
        "import sys\nfrom .cli import main\nsys.exit(main())\n")
    return project


def make_venv(project: Path, python: str, stamp: str | None = None) -> Path:
    """An existing .venv with a file of the user's in it."""
    venv = project / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text(python)
    (venv / "bin" / "python").chmod(0o755)
    (venv / "installed-packages.marker").write_text("deps the user installed")
    if stamp is not None:
        (venv / ".requirements-stamp").write_text(stamp)
    return venv


def fake_python3(tmp_path: Path, venv_works: bool) -> dict:
    """`python3` on PATH: '[-I] -m venv [--clear] DIR' makes DIR/bin/python.

    venv_works=False mimics Debian without python3-venv: the folder appears,
    ensurepip fails, and the venv has no pip.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    body = WORKING_PYTHON if venv_works else BROKEN_PYTHON
    script = (
        "#!/bin/sh\n"
        '[ "$1" = "-I" ] && shift\n'
        'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then\n'
        '  for dir; do :; done\n'                      # the last argument
        '  case " $* " in *" --clear "*) rm -rf "$dir" ;; esac\n'
        '  mkdir -p "$dir/bin"\n'
        f"  cat > \"$dir/bin/python\" <<'PY'\n{body}PY\n"
        '  chmod +x "$dir/bin/python"\n'
        + ("  exit 0\n" if venv_works else
           '  echo "ensurepip is not available" >&2\n  exit 1\n')
        + "fi\nexit 1\n")
    (bin_dir / "python3").write_text(script)
    (bin_dir / "python3").chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}


def run(project: Path, env: dict, cwd: Path | None = None):
    return subprocess.run(["bash", str(project / "transcribe.sh"), "--version"], env=env,
                          cwd=cwd or project, capture_output=True, text=True, timeout=60)


def stamp_of(project: Path) -> str:
    return subprocess.run(["sh", "-c", 'cksum < "$1"', "sh",
                           str(project / "requirements.txt")],
                          capture_output=True, text=True, check=True).stdout.rstrip("\n")


def test_failed_venv_creation_leaves_nothing_behind(tmp_path):
    project = make_project(tmp_path)
    env = fake_python3(tmp_path, venv_works=False)
    for _ in range(2):
        result = run(project, env)
        assert result.returncode != 0
        # The half-made venv used to stay, so every later run (even after
        # installing python3-venv) died with "No module named pip".
        assert not (project / ".venv").exists()
        assert "python3-venv" in result.stderr
        assert "No module named pip" not in result.stderr


def test_venv_without_pip_is_rebuilt_and_the_old_one_kept(tmp_path):
    """A venv left broken by the old script heals; nothing of it is deleted."""
    project = make_project(tmp_path)
    make_venv(project, BROKEN_PYTHON)
    result = run(project, fake_python3(tmp_path, venv_works=True))
    assert result.returncode == 0, result.stderr
    assert "stub transcribe" in result.stdout
    assert (project / ".venv" / ".requirements-stamp").is_file()
    kept = list(project.glob(".venv.broken-*"))
    assert len(kept) == 1 and (kept[0] / "installed-packages.marker").is_file()


def test_failed_rebuild_puts_the_old_venv_back(tmp_path):
    project = make_project(tmp_path)
    make_venv(project, BROKEN_PYTHON)
    result = run(project, fake_python3(tmp_path, venv_works=False))
    assert result.returncode != 0 and "python3-venv" in result.stderr
    assert (project / ".venv" / "installed-packages.marker").is_file()
    assert not list(project.glob(".venv.*"))


@pytest.mark.parametrize("venv_works", [True, False])
def test_healthy_venv_survives_a_pip_check_broken_by_the_cwd(tmp_path, venv_works):
    """V1: after a requirements change, ./types.py made the pip probe fail,
    and the healthy venv was wiped (--clear) or deleted (rm -rf)."""
    project = make_project(tmp_path)
    venv = make_venv(project, WORKING_PYTHON, stamp="stamp of the old requirements")
    elsewhere = tmp_path / "Downloads"
    elsewhere.mkdir()
    (elsewhere / "types.py").write_text("raise SystemExit('SHADOWED by ./types.py')\n")
    result = run(project, fake_python3(tmp_path, venv_works), cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert (venv / "installed-packages.marker").is_file()
    assert (venv / ".requirements-stamp").read_text() == stamp_of(project)
    assert not list(project.glob(".venv.*"))


@pytest.mark.parametrize("pythonpath", [None, ":/nonexistent", "/nonexistent:", "."])
def test_current_directory_cannot_shadow_modules(tmp_path, pythonpath):
    """`python -m` put the cwd first on sys.path: its logging.py won. An empty
    or relative PYTHONPATH entry did the same."""
    project = make_project(tmp_path)
    make_venv(project, WORKING_PYTHON, stamp="placeholder")
    (project / ".venv" / ".requirements-stamp").write_text(stamp_of(project))
    elsewhere = tmp_path / "Downloads"
    elsewhere.mkdir()
    (elsewhere / "logging.py").write_text("raise SystemExit('SHADOWED by ./logging.py')\n")
    env = fake_python3(tmp_path, venv_works=True)
    env.pop("PYTHONPATH", None)
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    result = run(project, env, cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert "stub transcribe" in result.stdout and "SHADOWED" not in result.stderr
