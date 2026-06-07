"""
GUI.CONFIG: PERSISTED USER PREFERENCES
======================================

Loads/saves GUI preferences (appearance theme, window geometry, last-used
filetype) as JSON in the per-user config directory, so they survive restarts.
All failures are swallowed — preferences are best-effort, never fatal.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import json
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
    """Return the saved preferences, or {} if none/unreadable."""
    try:
        return json.loads(config_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(data: dict) -> None:
    """Write preferences to disk (best-effort)."""
    try:
        config_file().write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
