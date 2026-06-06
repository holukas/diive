"""
GUI.WIDGETS.CONSOLE_PANEL: LIVE LIBRARY OUTPUT CONSOLE
======================================================

A dockable panel that mirrors diive's Rich console output into the GUI. It
registers a Rich mirror `Console` (via `add_console_sink`) that renders to an
in-memory stream as ANSI; a small SGR parser turns that into coloured text in a
read-only `QTextEdit`.

The stream emits a Qt signal per write, so output produced on a worker thread
is marshalled to the GUI thread safely.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import re

from rich.console import Console
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from diive.core.utils.console import add_console_sink, remove_console_sink

#: Width the mirror console renders at (affects rule length / wrapping).
_MIRROR_WIDTH = 100

#: Matches an SGR (color/style) escape, e.g. "\x1b[1;36m".
_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
#: Matches non-SGR CSI escapes (cursor moves etc.) -- stripped out.
_NON_SGR_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-ln-z]")

#: Standard ANSI foreground codes -> readable-on-white colors.
_FG = {
    30: "#212121", 31: "#D32F2F", 32: "#2E7D32", 33: "#B8860B",
    34: "#1565C0", 35: "#6A1B9A", 36: "#00838F", 37: "#616161",
    90: "#616161", 91: "#E53935", 92: "#43A047", 93: "#F9A825",
    94: "#1E88E5", 95: "#8E24AA", 96: "#0097A7", 97: "#9E9E9E",
}
_DEFAULT_FG = QColor("#212121")


class _SignalStream(QObject):
    """File-like object that emits its writes as a Qt signal."""

    text_written = Signal(str)

    def write(self, s: str) -> None:
        if s:
            self.text_written.emit(s)

    def flush(self) -> None:
        pass


class ConsolePanel(QWidget):
    """Read-only panel showing live, coloured diive console output."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        top.addStretch(1)
        save_btn = QPushButton("Save...")
        save_btn.clicked.connect(self._save)
        top.addWidget(save_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.text.clear())
        top.addWidget(clear_btn)
        layout.addLayout(top)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.text, stretch=1)

        self._fmt = self._base_format()

        # Mirror console renders library output as ANSI into the stream.
        self._stream = _SignalStream()
        self._stream.text_written.connect(self._append)
        self._mirror = Console(
            file=self._stream, force_terminal=True, color_system="standard",
            width=_MIRROR_WIDTH, highlight=False,
        )
        add_console_sink(self._mirror)

    @staticmethod
    def _base_format() -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(_DEFAULT_FG)
        fmt.setFontWeight(QFont.Weight.Normal)
        return fmt

    def _apply_sgr(self, params: str) -> None:
        codes = [int(c) for c in params.split(";") if c != ""] or [0]
        for code in codes:
            if code == 0:
                self._fmt = self._base_format()
            elif code == 1:
                self._fmt.setFontWeight(QFont.Weight.Bold)
            elif code == 22:
                self._fmt.setFontWeight(QFont.Weight.Normal)
            elif code == 2:  # dim
                self._fmt.setForeground(QColor("#9E9E9E"))
            elif code == 39:
                self._fmt.setForeground(_DEFAULT_FG)
            elif code in _FG:
                self._fmt.setForeground(QColor(_FG[code]))

    def _append(self, text: str) -> None:
        """Parse ANSI in `text` and append it to the view with formatting."""
        text = _NON_SGR_RE.sub("", text)
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        pos = 0
        for m in _SGR_RE.finditer(text):
            if m.start() > pos:
                cursor.insertText(text[pos:m.start()], self._fmt)
            self._apply_sgr(m.group(1))
            pos = m.end()
        if pos < len(text):
            cursor.insertText(text[pos:], self._fmt)
        self.text.setTextCursor(cursor)
        self.text.ensureCursorVisible()

    def _save(self) -> None:
        """Save the current log text to a file (plain text)."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save log", "diive_log.txt", "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text.toPlainText())
        except OSError as err:
            QMessageBox.critical(self, "Save failed", f"Could not save log:\n{err}")

    def closeEvent(self, event) -> None:
        remove_console_sink(self._mirror)
        super().closeEvent(event)
