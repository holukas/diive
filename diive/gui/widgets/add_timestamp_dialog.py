"""
GUI.WIDGETS.ADD_TIMESTAMP_DIALOG: ADD A TIMESTAMP COLUMN
========================================================

A dialog to add a new timestamp **data column** derived from the dataset's
timestamp index. The index itself is never touched (it stays whatever convention
it has, normally TIMESTAMP_MIDDLE) — this only appends a column.

The user picks:

- a **convention** (START / MIDDLE / END of the averaging interval),
- an optional **strftime format** (e.g. ``%Y%m%d%H%M`` for the FLUXNET
  convention) — empty keeps a real ``datetime`` column,
- the **column name**.

A live preview shows the index mapped to the resulting timestamps. On accept the
caller reads :meth:`convention` / :meth:`fmt` / :meth:`column_name` and builds the
column via the library's :func:`diive.times.format_timestamp`. The dialog only
collects values; the timestamp maths lives in the library.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from diive.times import format_timestamp, validate_timestamp_column_name

# Convention label -> argument for the library function.
_CONVENTIONS = {"Start of interval": "start",
                "Middle of interval": "middle",
                "End of interval": "end"}

# A few common strftime patterns offered in the (editable) format combo.
_FORMAT_PRESETS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y%m%d%H%M",        # FLUXNET upload convention
    "%Y-%m-%d",
]

_PREVIEW_ROWS = 6


class AddTimestampColumnDialog(QDialog):
    """Collect the definition of a new timestamp data column.

    Args:
        data: The dataset; its index supplies the timestamps and its frequency is
            needed for the START/END half-period shift. A live preview is computed
            from the first rows.
    """

    def __init__(self, data: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add timestamp column")
        self.setMinimumWidth(460)
        self._data = data
        self._name_edited = False

        layout = QVBoxLayout(self)
        index_name = data.index.name or "index"
        layout.addWidget(QLabel(
            f"Add a timestamp column derived from the <b>{index_name}</b> index. "
            "The index itself is not changed."))

        form = QFormLayout()

        self.conv_combo = QComboBox()
        self.conv_combo.addItems(list(_CONVENTIONS))
        self.conv_combo.setCurrentText("End of interval")
        self.conv_combo.currentTextChanged.connect(self._on_convention_changed)
        form.addRow("Convention", self.conv_combo)

        self.datetime_check = QCheckBox("Keep as datetime object (no text formatting)")
        self.datetime_check.toggled.connect(self._on_datetime_toggled)
        form.addRow("", self.datetime_check)

        self.fmt_combo = QComboBox()
        self.fmt_combo.setEditable(True)
        self.fmt_combo.addItems(_FORMAT_PRESETS)
        self.fmt_combo.setCurrentText(_FORMAT_PRESETS[0])
        self.fmt_combo.setToolTip(
            "strftime format, e.g. %Y-%m-%d %H:%M:%S\n"
            "%Y year, %m month, %d day, %H hour, %M minute, %S second")
        self.fmt_combo.currentTextChanged.connect(self._update_preview)
        form.addRow("Format", self.fmt_combo)

        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._on_name_edited)
        self.name_edit.textChanged.connect(self._update_preview)
        form.addRow("Column name", self.name_edit)
        layout.addLayout(form)

        self.preview = QTableWidget(0, 2)
        self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview.verticalHeader().setVisible(False)
        self.preview.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        layout.addWidget(QLabel("Preview"))
        layout.addWidget(self.preview)

        self.error_label = QLabel()
        self.error_label.setStyleSheet("QLabel { color: #C62828; }")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._sync_default_name()
        self._update_preview()

    # --- reactions -----------------------------------------------------
    def _on_convention_changed(self, *_args) -> None:
        self._sync_default_name()
        self._update_preview()

    def _on_datetime_toggled(self, checked: bool) -> None:
        self.fmt_combo.setEnabled(not checked)
        self._update_preview()

    def _on_name_edited(self, _text: str) -> None:
        self._name_edited = True

    def _sync_default_name(self) -> None:
        """Default the column name to TIMESTAMP_<CONV> until the user edits it."""
        if self._name_edited:
            return
        self.name_edit.setText(f"TIMESTAMP_{self.convention().upper()}")

    def _update_preview(self, *_args) -> None:
        head = self._data.head(_PREVIEW_ROWS)
        name, convention = self.column_name(), self.convention()
        self.preview.setHorizontalHeaderLabels(
            [self._data.index.name or "index", name or "new column"])
        ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        try:
            series = format_timestamp(head, convention=convention, fmt=self.fmt())
        except Exception as err:
            self.error_label.setText(f"Cannot build preview: {err}")
            self.preview.setRowCount(0)
            ok_btn.setEnabled(False)
            return
        self.preview.setRowCount(len(series))
        for row, (idx, value) in enumerate(series.items()):
            self.preview.setItem(row, 0, QTableWidgetItem(str(idx)))
            self.preview.setItem(row, 1, QTableWidgetItem(str(value)))
        # A reserved timestamp name must match the chosen convention, and the name
        # can't be empty — block OK with a clear message otherwise.
        if not name:
            self.error_label.setText("Please enter a column name.")
            ok_btn.setEnabled(False)
            return
        try:
            validate_timestamp_column_name(name, convention)
        except ValueError as err:
            self.error_label.setText(str(err))
            ok_btn.setEnabled(False)
            return
        self.error_label.clear()
        ok_btn.setEnabled(True)

    # --- result --------------------------------------------------------
    def convention(self) -> str:
        """The chosen convention argument ('start' / 'middle' / 'end')."""
        return _CONVENTIONS[self.conv_combo.currentText()]

    def fmt(self) -> str | None:
        """The chosen strftime format, or None to keep a datetime column."""
        if self.datetime_check.isChecked():
            return None
        text = self.fmt_combo.currentText().strip()
        return text or None

    def column_name(self) -> str:
        """The name for the new data column."""
        return self.name_edit.text().strip()
