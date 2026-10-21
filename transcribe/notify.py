"""Best-effort desktop notifications and Markdown opening.

Never raises: a missing DBUS session (cron, ssh) must not fail a run whose
transcripts are already safely on disk.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def desktop_environment() -> dict[str, str]:
    """Recover this user's existing Linux desktop when launched without its env."""
    env = os.environ.copy()
    if sys.platform.startswith("linux"):
        runtime = Path(env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
        if runtime.is_dir():
            env.setdefault("XDG_RUNTIME_DIR", str(runtime))
            if not env.get("DBUS_SESSION_BUS_ADDRESS") and (runtime / "bus").is_socket():
                env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime / 'bus'}"
            if not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
                sockets = [p for p in runtime.glob("wayland-*") if p.is_socket()]
                if len(sockets) == 1:
                    env["WAYLAND_DISPLAY"] = sockets[0].name
    return env


def open_markdown(path: Path) -> bool:
    """Launch the desktop Markdown application without waiting for it to close."""
    gio = shutil.which("gio") if sys.platform != "darwin" else None
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    executable = gio or shutil.which(opener)
    if not executable:
        log.warning("Cannot open %s automatically: %s is unavailable.", path, opener)
        return False
    try:
        command = [executable, "open"] if gio else [executable]
        subprocess.Popen([*command, str(path.resolve())],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True,
                         env=desktop_environment())
    except OSError as exc:
        log.warning("Cannot open %s automatically: %s", path, exc)
        return False
    log.info("Opening Markdown → %s", path)
    return True


def notify(title: str, body: str) -> None:
    try:
        if shutil.which("notify-send"):
            subprocess.run(["notify-send", title, body, "--icon=document",
                            "--urgency=normal"], check=False, timeout=10,
                           env=desktop_environment())
        elif shutil.which("osascript"):
            # Text goes in as arguments, never spliced into the script: a quote
            # in a folder name ("Bob's notes") broke it, and could inject code.
            subprocess.run(["osascript", "-e", "on run argv",
                            "-e", "display notification (item 2 of argv) "
                                  "with title (item 1 of argv)",
                            "-e", "end run", title, body], check=False, timeout=10)
    except Exception as exc:                      # noqa: BLE001 - never fatal
        log.debug("Desktop notification failed: %s", exc)
