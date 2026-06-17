"""
GUI.WIDGETS.OPEN_DATA_DIALOG: FILE + FILETYPE PICKER WITH PREVIEW
================================================================

A dialog for opening data file(s): pick one or more files, choose their
filetype, and see a live preview of the first parsed rows before committing.
Reading is done by the library (`dv.load_parquet` / `dv.ReadFileType`, or
`MultiDataFileReader` when several files are selected); this dialog only
orchestrates the choice and shows the result.

Selecting multiple files merges them into one dataset (same filetype assumed).

No default filetype is assumed (there are many) -- the user picks explicitly,
except for ``.parquet`` which is unambiguous and preselected. The preview reads
only a few rows (`data_nrows`) so switching filetypes stays fast.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading
from pathlib import Path

import diive as dv
from diive.configs.filetypes import get_filetypes
from diive.core.io.filereader import MultiDataFileReader
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
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


def _read_many(filepaths: list[str], choice: str, progress_callback=None):
    """Read and merge one or more full files of the chosen filetype.

    A single file uses :func:`_read`. Multiple files are merged by the library:
    configured filetypes via `MultiDataFileReader`, parquet files via
    `dv.load_parquet_many` (both combine_first: existing values win, later files
    fill gaps).

    `progress_callback`, if given, is forwarded to the library reader and called
    per file as ``callback(phase, done, total, filepath)`` (``phase`` in
    ``{'reading', 'done'}``) so a caller can show per-file progress. Only fires
    for the multi-file case.
    """
    if len(filepaths) == 1:
        return _read(filepaths[0], choice, nrows=None)
    if choice == _PARQUET_CHOICE:
        return dv.load_parquet_many(filepaths=filepaths, progress_callback=progress_callback)
    return MultiDataFileReader(
        filepaths=filepaths, filetype=choice, progress_callback=progress_callback,
    ).data_df


class _FileProgressRow(QWidget):
    """One row in the multi-file list: a filename label + a progress bar.

    The bar is indeterminate (busy) while its file is being read and full once
    the file has been merged. Pending files show an empty bar.
    """

    def __init__(self, filename: str, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 2, 6, 2)
        label = QLabel(filename)
        label.setToolTip(filename)
        label.setMinimumWidth(180)
        row.addWidget(label, stretch=1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(14)
        row.addWidget(self.bar, stretch=1)

    def set_reading(self) -> None:
        self.bar.setRange(0, 0)  # indeterminate / busy

    def set_done(self) -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(1)


class OpenDataDialog(QDialog):
    """Pick file(s) + filetype with a parsed preview; exposes the loaded df.

    Multiple files of the same filetype are merged into one DataFrame.
    """

    #: Emitted from the load worker thread with the loaded DataFrame / error.
    _load_done = Signal(object)
    _load_failed = Signal(str)
    #: Per-file merge progress from the worker: (phase, done, total, filepath).
    _progress = Signal(str, int, int, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open data file")
        self.resize(760, 500)
        self.dataframe = None      # set on successful Load
        self.source_name = ""
        self._paths: list[str] = []
        #: filepath -> its progress row, while a multi-file load is running.
        self._file_rows: dict[str, _FileProgressRow] = {}

        layout = QVBoxLayout(self)

        # File chooser row.
        file_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("No file(s) selected")
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

        # Multi-file progress list: one row (filename + progress bar) per
        # selected file, shown only when several files are merged. Hidden for a
        # single file.
        self.files_label = QLabel("Files (merged in order):")
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.files_label.setVisible(False)
        self.files_list.setVisible(False)
        layout.addWidget(self.files_label)
        layout.addWidget(self.files_list, stretch=1)

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

        # Loading runs on a worker thread; results return via these signals.
        self._load_done.connect(self._on_load_done)
        self._load_failed.connect(self._on_load_failed)
        self._progress.connect(self._on_progress)

    def _set_load_enabled(self, enabled: bool) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enabled)

    def _choice(self) -> str | None:
        return self.ft_combo.currentData() if self.ft_combo.currentIndex() >= 0 else None

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select data file(s)", "",
            "Data files (*.parquet *.csv *.csv.gz *.dat *.zip *.gz);;All files (*)",
        )
        if not paths:
            return
        self._paths = [str(p) for p in paths]
        if len(self._paths) == 1:
            self.path_edit.setText(self._paths[0])
        else:
            self.path_edit.setText(f"{len(self._paths)} files selected")
        self._build_files_list()
        if all(Path(p).suffix.lower() == ".parquet" for p in self._paths):
            self.ft_combo.setCurrentIndex(self.ft_combo.findData(_PARQUET_CHOICE))
        self._preview()

    def _build_files_list(self) -> None:
        """Populate the per-file progress list; show it only for multiple files."""
        self.files_list.clear()
        self._file_rows.clear()
        multi = len(self._paths) > 1
        self.files_label.setVisible(multi)
        self.files_list.setVisible(multi)
        if not multi:
            return
        for fp in self._paths:
            row = _FileProgressRow(Path(fp).name)
            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            self.files_list.addItem(item)
            self.files_list.setItemWidget(item, row)
            self._file_rows[fp] = row

    def _preview(self) -> None:
        """Preview the first file's first rows with the current choice.

        Only the first file is previewed (to confirm the filetype parses);
        loading merges all selected files.
        """
        self._set_load_enabled(False)
        choice = self._choice()
        if not self._paths or not choice:
            return
        try:
            df = _read(self._paths[0], choice, nrows=_PREVIEW_ROWS)
        except Exception as err:
            self.preview.clear()
            self.preview.setRowCount(0)
            self.preview.setColumnCount(0)
            self.status.setText(f"Preview failed: {err}")
            return
        self._fill_preview(df)
        n = len(self._paths)
        suffix = f" (preview of first of {n} files; all will be merged)" if n > 1 else ""
        self.status.setText(f"{df.shape[1]} columns detected. Ready to load.{suffix}")
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
        """Read (and merge, if multiple) the selected files on a worker thread.

        Reading large / many files can take seconds; doing it off the GUI thread
        keeps the window responsive and lets the output console show progress
        live. The result returns via `_load_done` / `_load_failed`.
        """
        choice = self._choice()
        if not self._paths or not choice:
            return
        self._set_load_enabled(False)
        n = len(self._paths)
        self.status.setText(f"Loading {n} file(s)..." if n > 1 else "Loading...")
        # Reset all bars to pending before a (re-)load.
        for row in self._file_rows.values():
            row.bar.setRange(0, 1)
            row.bar.setValue(0)
        paths = list(self._paths)
        threading.Thread(
            target=self._load_worker, args=(paths, choice), daemon=True,
        ).start()

    def _load_worker(self, paths: list[str], choice: str) -> None:
        # The callback runs on this worker thread; marshal to the GUI thread via
        # the queued `_progress` signal.
        def on_progress(phase, done, total, filepath):
            self._progress.emit(phase, done, total, str(filepath))

        try:
            df = _read_many(paths, choice, progress_callback=on_progress)
        except Exception as err:
            self._load_failed.emit(str(err))
            return
        self._load_done.emit(df)

    def _on_progress(self, phase: str, done: int, total: int, filepath: str) -> None:
        row = self._file_rows.get(filepath)
        if row is not None:
            if phase == 'reading':
                row.set_reading()
            elif phase == 'done':
                row.set_done()
        self.status.setText(f"Merging files... {done}/{total}")

    def _on_load_done(self, df) -> None:
        self.dataframe = df
        global _last_choice
        _last_choice = self._choice()
        self.source_name = (
            Path(self._paths[0]).name if len(self._paths) == 1
            else f"{len(self._paths)} files (merged)"
        )
        self.accept()

    def _on_load_failed(self, msg: str) -> None:
        self.status.setText(f"Load failed: {msg}")
        self._set_load_enabled(True)
