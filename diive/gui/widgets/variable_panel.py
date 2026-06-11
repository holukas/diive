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
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QLineEdit,
    QListWidgetItem,
    QMenu,
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

#: Original (dataset) insertion order of an item, so the list can be re-sorted
#: back to it when the filter is cleared. Panel-internal (not painted), so it
#: lives here rather than in the delegate's role set.
ORDER_ROLE = Qt.ItemDataRole.UserRole + 10


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumerics for separator-insensitive search."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _fuzzy_score(needle: str, hay: str):
    """Fuzzy-match `needle` against `hay` (both already normalized).

    Subsequence matching is the gate (every needle char appears in `hay` in
    order, gaps allowed); the return value ranks how *tight* the match is so the
    best candidates can float to the top. Higher is better. Contiguous runs, a
    match at the very start, an early first hit, and a haystack close in length
    to the needle all raise the score. Returns ``None`` when `needle` is not a
    subsequence of `hay`.
    """
    if not needle:
        return 0.0
    score = 0.0
    pos = 0       # search cursor into hay
    prev = -2     # index of the previously matched char
    first = None  # index of the first matched char
    run = 0       # current contiguous-run length
    for c in needle:
        idx = hay.find(c, pos)
        if idx == -1:
            return None
        if first is None:
            first = idx
        if idx == prev + 1:
            run += 1
            score += 2.0 + run  # contiguous matches are worth more, growing
        else:
            run = 0
            score += 1.0
        prev = idx
        pos = idx + 1
    if first == 0:
        score += 3.0  # matches anchored at the start rank highest
    score -= 0.05 * first              # earlier first hit is better
    score -= 0.1 * (len(hay) - len(needle))  # less padding around the match
    return score


class VariablePanel(QWidget):
    """Filter field + variable list with pills; shared across all tabs."""

    #: Emitted on item click as (variable_name, ctrl_held).
    selected = Signal(str, bool)
    #: Emitted when "Delete variable" is chosen from the right-click menu
    #: (only wired when the panel is built with `deletable=True`).
    deleteRequested = Signal(str)

    def __init__(self, parent=None, deletable: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("varpanel")  # white pane bg via QSS (search/list keep theirs)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._order_counter = 0  # next dataset-order index for added items

        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText("Fuzzy filter variables...")
        self.search.textChanged.connect(self._apply_filter)

        self.list = VariableList()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._delegate = VariableDelegate(self.list)
        self.list.setItemDelegate(self._delegate)
        self.list.setMouseTracking(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setWordWrap(False)
        self.list.selected.connect(self.selected)  # re-emit upward

        # Opt-in right-click "Delete variable" menu (tabs that own the dataset
        # wire `deleteRequested`; other tabs leave it off, so no dead menu).
        if deletable:
            self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.list.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.search)
        layout.addWidget(self.list, stretch=1)

        # Shared, identical width across all tabs (editable in Appearance
        # settings). Live theme preview also repaints pills/highlights.
        self.setFixedWidth(theme.manager.list_width)
        theme.manager.changed.connect(self._on_theme_changed)

    def _show_context_menu(self, pos) -> None:
        """Right-click menu for the item under the cursor: delete the variable."""
        item = self.list.itemAt(pos)
        if item is None:
            return
        name = item.data(NAME_ROLE)
        menu = QMenu(self.list)
        act = QAction(f"Delete '{name}'", menu)
        act.triggered.connect(lambda: self.deleteRequested.emit(name))
        menu.addAction(act)
        menu.exec(self.list.viewport().mapToGlobal(pos))

    def _on_theme_changed(self) -> None:
        self.setFixedWidth(theme.manager.list_width)
        self.list.viewport().update()

    # --- population ---
    def set_variables(self, names, created: set | None = None) -> None:
        """Replace the list contents; `created` names get the NEW pill."""
        created = created or set()
        self.list.clear()
        self._order_counter = 0
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
        item.setData(ORDER_ROLE, self._order_counter)
        self._order_counter += 1
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
        """Fuzzy-filter and rank the list.

        With an empty field, all items show in dataset order. While typing,
        non-matching items are hidden and matches are sorted best-first by their
        fuzzy score (ties keep dataset order), so the closest variable rises to
        the top.
        """
        needle = _normalize(text)
        items = [self.list.item(i) for i in range(self.list.count())]
        if not items:
            return
        if not needle:
            ordered = sorted(items, key=lambda it: it.data(ORDER_ROLE))
            self._reorder(ordered, hidden=set())
            return
        matched, unmatched = [], []
        for it in items:
            score = _fuzzy_score(needle, _normalize(it.data(NAME_ROLE) or ""))
            if score is None:
                unmatched.append(it)
            else:
                matched.append((score, it.data(ORDER_ROLE), it))
        matched.sort(key=lambda t: (-t[0], t[1]))  # best score first, then order
        ordered = [t[2] for t in matched]
        ordered += sorted(unmatched, key=lambda it: it.data(ORDER_ROLE))
        # QListWidgetItem is unhashable in PySide6, so track hidden ones by id().
        self._reorder(ordered, hidden={id(it) for it in unmatched})

    def _reorder(self, ordered: list, hidden: set) -> None:
        """Re-sequence the list to `ordered`, hiding items whose id() is in `hidden`.

        Items are detached and re-added in the new order; each keeps its data
        (name, pills, panel/loading state), so only position and the hidden flag
        change. `ordered` must contain every current item exactly once (it holds
        the references that keep the detached items alive).
        """
        self.list.setUpdatesEnabled(False)
        try:
            while self.list.count():
                self.list.takeItem(0)
            for it in ordered:
                self.list.addItem(it)
                it.setHidden(id(it) in hidden)
        finally:
            self.list.setUpdatesEnabled(True)
        self.list.viewport().update()
