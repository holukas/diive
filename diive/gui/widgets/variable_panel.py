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
    QInputDialog,
    QLineEdit,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import FAVORITE
from diive.gui import metadata_store, theme
from diive.gui.widgets.menu import studio_menu
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


def lock_panel_handle(splitter, handle_index: int = 1) -> None:
    """Disable the splitter handle next to the fixed-width ``VariablePanel``.

    The variable list has a fixed width (``theme.manager.list_width``), so the
    handle right after it can't resize anything — but it still shows a misleading
    horizontal-resize (↔) cursor. Disabling it and forcing the arrow cursor
    removes that. ``handle_index`` is the handle to the *right* of the panel (1
    when the panel is the splitter's first widget, as it always is here)."""
    splitter.setChildrenCollapsible(False)
    handle = splitter.handle(handle_index)
    if handle is not None:
        handle.setEnabled(False)
        handle.setCursor(Qt.CursorShape.ArrowCursor)


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
    #: Emitted when "Remove all tags & note" is chosen for a variable (only
    #: offered when the panel is built with `clearable=True`). The owning tab
    #: handles it so it can unbind any open editor before the store changes.
    clearRequested = Signal(str)

    def __init__(self, parent=None, clearable: bool = False,
                 draggable: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("varpanel")  # white pane bg via QSS (search/list keep theirs)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._order_counter = 0  # next dataset-order index for added items

        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText("Fuzzy filter variables...")
        self.search.textChanged.connect(self._apply_filter)

        self.list = VariableList(draggable=draggable)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._delegate = VariableDelegate(self.list)
        self.list.setItemDelegate(self._delegate)
        self.list.setMouseTracking(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setWordWrap(False)
        self.list.selected.connect(self.selected)  # re-emit upward

        # Right-click menu: rename/delete + metadata tags everywhere (routed
        # through the app-wide metadata manager, so no per-tab wiring is needed).
        self._clearable = clearable
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.search)
        layout.addWidget(self.list, stretch=1)

        # Shared, identical width across all tabs (editable in Appearance
        # settings). Live theme preview also repaints pills/highlights.
        self.setFixedWidth(theme.manager.list_width)
        theme.manager.changed.connect(self._on_theme_changed)
        # Re-sort + repaint when metadata changes anywhere: favorites float to
        # the top, so toggling one must re-order the list, not just repaint it.
        metadata_store.manager.changed.connect(
            lambda: self._apply_filter(self.search.text()))

    def _show_context_menu(self, pos) -> None:
        """Right-click menu: rename, delete, favorite + tag editing (all tabs)."""
        item = self.list.itemAt(pos)
        if item is None:
            return
        name = item.data(NAME_ROLE)
        md = metadata_store.manager.store.get(name)
        menu = studio_menu(self.list)

        # Jump to the Metadata explorer for this variable (offered everywhere
        # except inside the explorer's own list, which is already there).
        if not self._clearable:
            edit_act = QAction("Edit metadata…", menu)
            edit_act.triggered.connect(
                lambda: metadata_store.manager.request_edit(name))
            menu.addAction(edit_act)
            menu.addSeparator()

        # Rename / delete work in every tab's variable list (incl. the explorer's
        # own), routed through the app-wide metadata manager.
        rename_act = QAction("Rename…", menu)
        rename_act.triggered.connect(
            lambda: metadata_store.manager.request_rename(name))
        menu.addAction(rename_act)
        del_act = QAction("Delete…", menu)
        del_act.triggered.connect(
            lambda: metadata_store.manager.request_delete(name))
        menu.addAction(del_act)
        menu.addSeparator()

        fav = FAVORITE in md.tags
        fav_act = QAction("★ Unmark favorite" if fav else "★ Mark favorite", menu)
        fav_act.triggered.connect(
            lambda: metadata_store.manager.toggle_user_tag(name, FAVORITE))
        menu.addAction(fav_act)

        add_act = QAction("Add tag…", menu)
        add_act.triggered.connect(lambda: self._add_tag_dialog(name))
        menu.addAction(add_act)

        removable = [t for t in md.user_tags() if t != FAVORITE]
        if removable:
            sub = studio_menu(menu)
            sub.setTitle("Remove tag")
            menu.addMenu(sub)
            for tag in removable:
                a = QAction(tag, sub)
                a.triggered.connect(
                    lambda _checked=False, t=tag:
                    metadata_store.manager.remove_user_tag(name, t))
                sub.addAction(a)

        if self._clearable and (md.user_tags() or md.description):
            clear_act = QAction("Remove all tags & note", menu)
            clear_act.triggered.connect(lambda: self.clearRequested.emit(name))
            menu.addAction(clear_act)

        menu.exec(self.list.viewport().mapToGlobal(pos))

    @staticmethod
    def _is_fav(item) -> bool:
        return metadata_store.manager.is_favorite(item.data(NAME_ROLE))

    def _add_tag_dialog(self, name: str) -> None:
        text, ok = QInputDialog.getText(self, "Add tag", f"Tag for '{name}':")
        if ok and text.strip():
            metadata_store.manager.add_user_tag(name, text.strip())

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

    def clear_filter(self) -> None:
        """Clear the fuzzy-filter text (e.g. so a just-added variable that
        wouldn't match the current filter is not hidden)."""
        self.search.clear()

    def scroll_to(self, name: str) -> None:
        """Scroll the list so `name`'s row is visible (e.g. a just-added feature
        appended at the bottom of a long list)."""
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(NAME_ROLE) == name:
                self.list.scrollToItem(item)
                return

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
        # Favorites pin to the top in every view (sorts before the score/order).
        if not needle:
            ordered = sorted(
                items, key=lambda it: (not self._is_fav(it), it.data(ORDER_ROLE)))
            self._reorder(ordered, hidden=set())
            return
        matched, unmatched = [], []
        for it in items:
            score = _fuzzy_score(needle, _normalize(it.data(NAME_ROLE) or ""))
            if score is None:
                unmatched.append(it)
            else:
                matched.append((not self._is_fav(it), -score, it.data(ORDER_ROLE), it))
        matched.sort(key=lambda t: t[:3])  # favorites first, then score, then order
        ordered = [t[3] for t in matched]
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
