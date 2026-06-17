"""
GUI.WIDGETS.COPY_BUTTON: "COPY PYTHON" BUTTON
=============================================

A small reusable button that copies generated Python code to the clipboard. The
code itself is produced by a library codegen function (separation rule: the GUI
never builds the script, it only asks the library for it and copies the result).

Give it a ``code_provider`` callable returning the snippet (or ``None`` to do
nothing); on click it copies and briefly flashes "Copied" as feedback.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QPushButton


class CopyPythonButton(QPushButton):
    """A button that copies a code snippet (from ``code_provider``) to the clipboard."""

    def __init__(self, code_provider: Callable[[], str | None],
                 text: str = "Copy Python", parent=None) -> None:
        super().__init__(text, parent)
        self._provider = code_provider
        self._label = text
        self.clicked.connect(self._copy)

    def _copy(self) -> None:
        code = self._provider()
        if not code:
            return
        QApplication.clipboard().setText(code)
        # Brief visual confirmation, then restore the label.
        self.setText("Copied ✓")
        self.setEnabled(False)
        QTimer.singleShot(1200, self._restore)

    def _restore(self) -> None:
        self.setText(self._label)
        self.setEnabled(True)
