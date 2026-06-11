"""
GUI.TABS.RENAME_VARIABLES: BULK PREFIX / SUFFIX RENAME
======================================================

Add a common prefix and/or suffix to *all* variable names at once. A live
old -> new preview table shows exactly what the rename would produce before it
is applied; clicking Apply renames the columns in the loaded dataset (and the
matching per-variable metadata, in the library store).

The rename itself is trivial string work; the column rename (``df.rename``) and
the metadata re-keying (``MetadataStore.rename``) are done by the main window /
library. This tab only collects the prefix/suffix and emits the mapping.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diive.gui import metadata_store, theme
from diive.gui.tabs.base import DiiveTab


class _RenameSignals(QObject):
    """Qt signals for the tab (DiiveTab is a plain ABC, not a QObject)."""

    variables_renamed = Signal(dict)


class RenameVariablesTab(DiiveTab):
    """Add a prefix/suffix to every variable name, with a preview before apply."""

    title = "Rename variables"

    def build(self) -> QWidget:
        self._names: list[str] = []
        self._sig = _RenameSignals()
        #: Exposed bound signal the main window connects to.
        self.variablesRenamed = self._sig.variables_renamed

        root = QWidget()
        outer = QVBoxLayout(root)

        intro = QLabel(
            "Add a common prefix and/or suffix to every variable name. The "
            "table previews the result; click Apply to rename the columns in "
            "the loaded dataset (the source file is untouched). Double-click a "
            "name in the table to rename that single variable.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6B7780;")
        outer.addWidget(intro)

        # --- prefix / suffix inputs ---
        controls = QGroupBox("Prefix / suffix")
        form = QFormLayout(controls)
        self.prefix = QLineEdit()
        self.prefix.setPlaceholderText("e.g. CH-DAV_")
        self.suffix = QLineEdit()
        self.suffix.setPlaceholderText("e.g. _2024")
        self.prefix.textChanged.connect(self._refresh)
        self.suffix.textChanged.connect(self._refresh)
        form.addRow("Prefix", self.prefix)
        form.addRow("Suffix", self.suffix)
        outer.addWidget(controls)

        # --- preview table: old name -> new name ---
        outer.addWidget(QLabel(theme.manager.label_text("Preview")))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Current name", "New name"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # Double-click a single variable to rename just that one (routes through
        # the app-wide rename flow: prompt + collision check in the main window).
        self.table.cellDoubleClicked.connect(self._rename_single)
        outer.addWidget(self.table, stretch=1)

        # --- footer: status + apply ---
        footer = QHBoxLayout()
        self.status = QLabel("")
        self.status.setStyleSheet("color: #6B7780;")
        footer.addWidget(self.status, stretch=1)
        self.apply_btn = QPushButton("Apply rename")
        self.apply_btn.clicked.connect(self._apply)
        theme.set_button_role(self.apply_btn, "confirm")
        footer.addWidget(self.apply_btn)
        outer.addLayout(footer)

        self._refresh()
        return root

    # --- mapping ---
    def _mapping(self) -> dict:
        """``{old: new}`` for names actually changed by the current prefix/suffix."""
        pre, suf = self.prefix.text(), self.suffix.text()
        if not pre and not suf:
            return {}
        return {n: f"{pre}{n}{suf}" for n in self._names if f"{pre}{n}{suf}" != n}

    # --- data ---
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._names = [str(c) for c in df.columns]
        self._refresh()

    def _refresh(self) -> None:
        mapping = self._mapping()
        self.table.setRowCount(len(self._names))
        for row, name in enumerate(self._names):
            new = mapping.get(name, name)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            new_item = QTableWidgetItem(new)
            if new != name:  # highlight the ones that actually change
                f = new_item.font()
                f.setBold(True)
                new_item.setFont(f)
            self.table.setItem(row, 1, new_item)
        n = len(mapping)
        if not self._names:
            self.status.setText("No data loaded.")
        elif n == 0:
            self.status.setText("Enter a prefix and/or suffix to rename.")
        else:
            self.status.setText(f"{n} variable{'' if n == 1 else 's'} will be renamed.")
        self.apply_btn.setEnabled(n > 0)

    def _rename_single(self, row: int, _col: int) -> None:
        """Rename one variable from its table row (current-name column)."""
        if 0 <= row < len(self._names):
            metadata_store.manager.request_rename(self._names[row])

    def _apply(self) -> None:
        mapping = self._mapping()
        if not mapping:
            return
        self.variablesRenamed.emit(mapping)
        self.status.setText(f"Renamed {len(mapping)} variables.")
        # Names now carry the prefix/suffix; clear the fields so re-applying
        # doesn't stack another copy (the re-pushed data refreshes the table).
        self.prefix.clear()
        self.suffix.clear()

    # --- project save / restore ---
    def save_state(self) -> dict:
        return {"prefix": self.prefix.text(), "suffix": self.suffix.text()}

    def restore_state(self, state: dict) -> None:
        self.prefix.setText(str(state.get("prefix", "")))
        self.suffix.setText(str(state.get("suffix", "")))
