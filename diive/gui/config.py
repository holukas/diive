"""
GUI.CONFIG: PERSISTED USER PREFERENCES
======================================

Loads/saves GUI preferences (appearance theme, window geometry, last-used
filetype) as JSON in the per-user config directory, so they survive restarts.
All failures are swallowed — preferences are best-effort, never fatal.

Saving is atomic: the blob goes to a temporary file next to the target and is
then swapped in with `os.replace`. This file also holds `last_project` and the
per-dataset `variable_metadata`, so a half-written one loses real user state —
and worse than a missing one, because a truncated file can still parse as
valid-but-empty and would then be loaded as if the user had no preferences.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QStandardPaths


def config_file() -> Path:
    """Path to the GUI settings JSON (created dir if needed)."""
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation)
    directory = Path(base) if base else (Path.home() / ".diive")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "gui_settings.json"


def load_config() -> dict:
    """Return the saved preferences, or {} if none/unreadable/not a mapping."""
    try:
        cfg = json.loads(config_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # ValueError covers both a malformed/truncated JSON document
        # (JSONDecodeError) and undecodable bytes (UnicodeDecodeError).
        return {}
    # A file holding valid JSON that is not an object (`null`, a list, a bare
    # number) would reach every `cfg.get(...)` call site as a non-dict and take
    # down startup. A corrupt config must not stop the app from starting.
    return cfg if isinstance(cfg, dict) else {}


def save_config(data: dict) -> None:
    """Write preferences to disk atomically (best-effort)."""
    tmp: Path | None = None
    try:
        # Serialize first: an unserializable value then fails before anything
        # touches the filesystem, so no temporary file is even created.
        payload = json.dumps(data, indent=2)
        target = config_file()
        # Same directory as the target, so it is on the same filesystem and
        # `os.replace` is a real atomic swap rather than a copy. The unique name
        # keeps two processes saving at once from writing one temporary file.
        fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                    dir=str(target.parent))
        tmp = Path(name)
        with open(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            # Without fsync the swap can win the race to disk and a crash then
            # leaves a zero-length file atomically replacing a good one.
            os.fsync(f.fileno())
        os.replace(tmp, target)  # atomic on Windows and POSIX (os.rename is not)
        tmp = None
    except (OSError, TypeError, ValueError):
        # OSError: unwritable path. TypeError/ValueError: `json.dumps` on a value
        # some producer put in the blob that isn't JSON-serializable (or is
        # circular). Preferences are best-effort, so none of these may take down
        # `closeEvent` — but only serialization and I/O are swallowed here.
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass  # Nothing more to do; the target is untouched either way.
