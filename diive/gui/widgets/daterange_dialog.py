"""
GUI.WIDGETS.DATERANGE_DIALOG: DATE-RANGE SUBSELECTION PICKER
===========================================================

A small dialog to pick a *from* / *to* timestamp window for subselecting the
loaded dataset. Both pickers are seeded with the dataset's first/last
timestamps and constrained to that span, so the user can only narrow it. On
accept, `range()` returns the chosen `(start, end)` as pandas Timestamps.

This dialog only collects the two bounds; the actual slicing is done by the
library (`dv.times.keep_daterange`) -- see `MainWindow._select_daterange`.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QDateTime, Qt
from PySide6.QtWidgets import (
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


def _to_qdatetime(ts: pd.Timestamp) -> QDateTime:
    return QDateTime.fromString(ts.strftime("%Y-%m-%d %H:%M:%S"), "yyyy-MM-dd HH:mm:ss")


class DateRangeDialog(QDialog):
    """Pick a from/to window within ``[data_start, data_end]``.

    `start` and `end` seed the pickers (typically the dataset's full span, or
    the currently active subselection). Both edits are clamped to
    ``[data_start, data_end]`` so the result is always a sub-window.
    """

    def __init__(self, data_start: pd.Timestamp, data_end: pd.Timestamp,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select date range")
        self._data_start = pd.Timestamp(data_start)
        self._data_end = pd.Timestamp(data_end)
        start = pd.Timestamp(start) if start is not None else self._data_start
        end = pd.Timestamp(end) if end is not None else self._data_end

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Full range: {self._data_start:%Y-%m-%d %H:%M} "
            f"to {self._data_end:%Y-%m-%d %H:%M}"))

        form = QFormLayout()
        self.from_edit = self._make_edit(start)
        self.to_edit = self._make_edit(end)
        form.addRow("From", self.from_edit)
        form.addRow("To", self.to_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_edit(self, value: pd.Timestamp) -> QDateTimeEdit:
        edit = QDateTimeEdit()
        edit.setCalendarPopup(True)
        edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        edit.setMinimumDateTime(_to_qdatetime(self._data_start))
        edit.setMaximumDateTime(_to_qdatetime(self._data_end))
        edit.setDateTime(_to_qdatetime(value))
        return edit

    def _on_accept(self) -> None:
        # Keep the two bounds ordered so an inverted pick still yields a valid
        # range rather than an empty selection.
        if self.from_edit.dateTime() > self.to_edit.dateTime():
            self.from_edit, self.to_edit = self.to_edit, self.from_edit
        self.accept()

    def selected_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the chosen ``(start, end)`` as pandas Timestamps."""
        start = pd.Timestamp(self.from_edit.dateTime().toString("yyyy-MM-dd HH:mm:ss"))
        end = pd.Timestamp(self.to_edit.dateTime().toString("yyyy-MM-dd HH:mm:ss"))
        return start, end
