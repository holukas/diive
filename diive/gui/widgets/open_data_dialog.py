"""
GUI.WIDGETS.OPEN_DATA_DIALOG: FILE + FILETYPE PICKER WITH PREVIEW
================================================================

A dialog for opening a data file: pick the file, choose its filetype, and see a
live preview of the first parsed rows before committing. Reading is done by the
library (`dv.load_parquet` / `dv.ReadFileType`); this dialog only orchestrates
the choice and shows the result.

No default filetype is assumed (there are many) -- the user picks explicitly,
except for ``.parquet`` which is unambiguous and preselected. The preview reads
only a few rows (`data_nrows`) so switching filetypes stays fast.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from pathlib import Path

import diive as dv
from diive.configs.filetypes import get_filetypes
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

#: Combo entry for parquet files (read via dv.load_parquet, no filetype config).
_PARQUET_CHOICE = "Parquet"

#: Most-used filetype: pinned to the top of the list and starred.
_IMPORTANT_FILETYPE = "EDDYPRO-FLUXNET-CSV-30MIN"

#: Rows shown in the preview.
_PREVIEW_ROWS = 10

#: Last filetype the user loaded this session, pre-selected on the next open.
_last_choice: str | None = None


def _read(filepath: str, choice: str, nrows: int | None):
    """Read a file with the chosen filetype via the library.

    `nrows` limits rows for a fast preview (None = full file).
    """
    if choice == _PARQUET_CHOICE:
        df = dv.load_parquet(filepath=filepath)
        return df.head(nrows) if nrows else df
    df, _meta = dv.ReadFileType(
        filepath=filepath, filetype=choice, data_nrows=nrows,
    ).get_filedata()
    return df


class OpenDataDialog(QDialog):
    """Pick a file + filetype with a parsed preview; exposes the loaded df."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open data file")
        self.resize(760, 500)
        self.dataframe = None      # set on successful Load
        self.source_name = ""

        layout = QVBoxLayout(self)

        # File chooser row.
        file_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("No file selected")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        file_row.addWidget(self.path_edit, stretch=1)
        file_row.addWidget(browse)
        layout.addLayout(file_row)

        # Filetype row (no default -- starts unselected).
        ft_row = QHBoxLayout()
        ft_row.addWidget(QLabel("File type:"))
        # Display text may carry a star; the real filetype key is stored as
        # item data and used for reading. The most-used filetype is pinned to
        # the top with a separator below; everything else follows.
        self.ft_combo = QComboBox()
        self.ft_combo.addItem(f"★ {_IMPORTANT_FILETYPE}", _IMPORTANT_FILETYPE)
        self.ft_combo.insertSeparator(self.ft_combo.count())
        self.ft_combo.addItem(_PARQUET_CHOICE, _PARQUET_CHOICE)
        for key in get_filetypes():
            if key not in ("__init__", _IMPORTANT_FILETYPE):
                self.ft_combo.addItem(key, key)
        # Pre-select the last-used filetype so repeated loads of the same format
        # need no re-pick; otherwise start unselected (no default).
        if _last_choice is not None:
            self.ft_combo.setCurrentIndex(self.ft_combo.findData(_last_choice))
        else:
            self.ft_combo.setCurrentIndex(-1)
        self.ft_combo.setPlaceholderText("Choose filetype...")
        self.ft_combo.currentIndexChanged.connect(self._preview)
        ft_row.addWidget(self.ft_combo, stretch=1)
        layout.addLayout(ft_row)

        layout.addWidget(QLabel("Preview (first rows):"))
        self.preview = QTableWidget()
        self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.preview, stretch=1)

        self.status = QLabel("")
        layout.addWidget(self.status)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Load")
        self.buttons.accepted.connect(self._load)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._set_load_enabled(False)

    def _set_load_enabled(self, enabled: bool) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enabled)

    def _choice(self) -> str | None:
        return self.ft_combo.currentData() if self.ft_combo.currentIndex() >= 0 else None

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data file", "",
            "Data files (*.parquet *.csv *.csv.gz *.dat *.zip *.gz);;All files (*)",
        )
        if not path:
            return
        self.path_edit.setText(path)
        if Path(path).suffix.lower() == ".parquet":
            self.ft_combo.setCurrentIndex(self.ft_combo.findData(_PARQUET_CHOICE))
        self._preview()

    def _preview(self) -> None:
        """Read a few rows with the current choice and show them; enable Load."""
        self._set_load_enabled(False)
        path, choice = self.path_edit.text(), self._choice()
        if not path or not choice:
            return
        try:
            df = _read(path, choice, nrows=_PREVIEW_ROWS)
        except Exception as err:
            self.preview.clear()
            self.preview.setRowCount(0)
            self.preview.setColumnCount(0)
            self.status.setText(f"Preview failed: {err}")
            return
        self._fill_preview(df)
        self.status.setText(f"{df.shape[1]} columns detected. Ready to load.")
        self._set_load_enabled(True)

    def _fill_preview(self, df) -> None:
        cols = [str(c) for c in df.columns]
        nrows = min(len(df), _PREVIEW_ROWS)
        self.preview.setColumnCount(len(cols) + 1)
        self.preview.setHorizontalHeaderLabels(["TIMESTAMP", *cols])
        self.preview.setRowCount(nrows)
        for r in range(nrows):
            self.preview.setItem(r, 0, QTableWidgetItem(str(df.index[r])))
            for c, col in enumerate(df.columns):
                self.preview.setItem(r, c + 1, QTableWidgetItem(str(df.iloc[r, c])))
        self.preview.resizeColumnsToContents()

    def _load(self) -> None:
        """Read the full file with the chosen filetype, then accept."""
        path, choice = self.path_edit.text(), self._choice()
        if not path or not choice:
            return
        try:
            self.dataframe = _read(path, choice, nrows=None)
        except Exception as err:
            self.status.setText(f"Load failed: {err}")
            return
        global _last_choice
        _last_choice = choice
        self.source_name = Path(path).name
        self.accept()
