"""
GUI.TABS.LOG: OUTPUT LOG TAB
============================

A tab wrapping `ConsolePanel`, which mirrors diive's Rich console output in
colour. Kept as its own tab so the live log doesn't take space from the
plotting area.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget

from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.console_panel import ConsolePanel


class LogTab(DiiveTab):
    """Live, coloured mirror of the diive library's console output."""

    title = "Log"

    def build(self) -> QWidget:
        self.console = ConsolePanel()
        return self.console

    def save_state(self) -> dict:
        """Persist the accumulated log text with the project."""
        return {"text": self.console.contents()}

    def restore_state(self, state: dict) -> None:
        """Restore log text saved with the project."""
        text = (state or {}).get("text") or ""
        if text:
            self.console.restore(text)
