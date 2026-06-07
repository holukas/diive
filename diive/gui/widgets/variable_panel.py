"""
GUI.WIDGETS.VARIABLE_PANEL: SHARED VARIABLE LIST
================================================

The single, reusable variable browser used by every tab so the left-hand
variable list looks and behaves identically everywhere: a filter field above a
`VariableList` painted by `VariableDelegate` (tag pills + selection highlight),
with separator-insensitive subsequence filtering. Tabs differ only in how they
*react* to selections, not in the list itself.

Usage:
    panel = VariablePanel()
    panel.selected.connect(on_selected)        # (name, ctrl_held)
    panel.set_variables(df.columns, created)    # populate
    panel.set_panels(["NEE_CUT_REF_f"])         # highlight (panel order)

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import re

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QLineEdit,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diive.gui import theme
from diive.gui.widgets.variable_delegate import (
    CREATED_ROLE,
    LOADING_ROLE,
    NAME_ROLE,
    PANEL_ROLE,
    VariableDelegate,
)
from diive.gui.widgets.variable_list import VariableList


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumerics for separator-insensitive search."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_subsequence(needle: str, hay: str) -> bool:
    """True if every char of `needle` appears in `hay` in order (gaps allowed)."""
    it = iter(hay)
    return all(c in it for c in needle)


class VariablePanel(QWidget):
    """Filter field + variable list with pills; shared across all tabs."""

    #: Emitted on item click as (variable_name, ctrl_held).
    selected = Signal(str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText("Filter variables...")
        self.search.textChanged.connect(self._apply_filter)

        self.list = VariableList()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._delegate = VariableDelegate(self.list)
        self.list.setItemDelegate(self._delegate)
        self.list.setMouseTracking(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setWordWrap(False)
        self.list.selected.connect(self.selected)  # re-emit upward

        layout.addWidget(self.search)
        layout.addWidget(self.list, stretch=1)

        # Shared, identical width across all tabs (editable in Appearance
        # settings). Live theme preview also repaints pills/highlights.
        self.setFixedWidth(theme.manager.list_width)
        theme.manager.changed.connect(self._on_theme_changed)

    def _on_theme_changed(self) -> None:
        self.setFixedWidth(theme.manager.list_width)
        self.list.viewport().update()

    # --- population ---
    def set_variables(self, names, created: set | None = None) -> None:
        """Replace the list contents; `created` names get the NEW pill."""
        created = created or set()
        self.list.clear()
        for name in names:
            self._add_item(str(name), str(name) in created)
        self._apply_filter(self.search.text())

    def add_name(self, name: str, created: bool = False) -> None:
        self._add_item(str(name), created)
        self._apply_filter(self.search.text())

    def remove_name(self, name: str) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(NAME_ROLE) == name:
                self.list.takeItem(i)
                return

    def names(self) -> list[str]:
        return [self.list.item(i).data(NAME_ROLE) for i in range(self.list.count())]

    def _add_item(self, name: str, created: bool) -> None:
        item = QListWidgetItem(name)
        item.setData(NAME_ROLE, name)
        item.setData(PANEL_ROLE, 0)
        item.setData(CREATED_ROLE, created)
        self.list.addItem(item)

    # --- highlight (panel order: 1 = primary, 2+ = additional) ---
    def set_panels(self, panels: list[str]) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            name = item.data(NAME_ROLE)
            item.setData(PANEL_ROLE, panels.index(name) + 1 if name in panels else 0)
        self.list.viewport().update()

    # --- loading indicator ---
    def set_loading(self, name: str) -> None:
        """Mark `name` as loading (busy wash + bar); clears it from others."""
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setData(LOADING_ROLE, item.data(NAME_ROLE) == name)
        self.list.viewport().update()

    def clear_loading(self) -> None:
        for i in range(self.list.count()):
            self.list.item(i).setData(LOADING_ROLE, False)
        self.list.viewport().update()

    def run_with_loading(self, name: str, fn) -> None:
        """Show the busy indicator on `name`, run `fn` (deferred), then clear it.

        matplotlib renders synchronously (blocking the event loop), so the
        indicator is painted *before* `fn` runs: `set_loading` + a forced
        repaint show it, a wait cursor signals the app is busy, and `fn` is
        deferred one tick so that paint lands first.
        """
        self.set_loading(name)
        self.list.viewport().repaint()  # paint the busy frame before fn blocks
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        def _go() -> None:
            try:
                fn()
            finally:
                self.clear_loading()
                QApplication.restoreOverrideCursor()

        QTimer.singleShot(0, _go)

    # --- filtering ---
    def _apply_filter(self, text: str) -> None:
        needle = _normalize(text)
        for i in range(self.list.count()):
            item = self.list.item(i)
            hay = _normalize(item.data(NAME_ROLE) or "")
            item.setHidden(not _is_subsequence(needle, hay))
