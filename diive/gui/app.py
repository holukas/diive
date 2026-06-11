"""
GUI.APP: APPLICATION BOOTSTRAP & MAIN WINDOW
============================================

`MainWindow` is a `QTabWidget`-hosting window that is agnostic to concrete
tabs: it iterates `registry.TAB_CLASSES`, instantiates each `DiiveTab`, and
adds it. `run()` boots the `QApplication` and enters the event loop.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import diive
from diive.gui import config, metadata_store, site, theme
from diive.gui.registry import (
    MENU_TAB_CLASSES,
    MENU_TABS,
    SINGLE_INSTANCE_TABS,
    TAB_CLASSES,
)
from diive.core.io.files import ALLOWED_TIMESTAMP_NAMES
from diive.gui.widgets.daterange_dialog import DateRangeDialog
from diive.gui.widgets.open_data_dialog import OpenDataDialog


def _namespace_metadata(raw: dict, key: str) -> dict:
    """Coerce a persisted ``variable_metadata`` blob to the per-dataset shape
    ``{dataset_key: {"tags": ..., "descriptions": ...}}``.

    Already-namespaced blobs pass through unchanged. Older non-namespaced ones —
    a single ``{"tags":..,"descriptions":..}`` or the legacy flat ``{name:[tags]}``
    — are migrated under ``key`` so a user's existing tags survive the upgrade
    (attached to whichever dataset loads first, since the old format didn't record
    which dataset they belonged to).
    """
    if not raw:
        return {}

    def _is_userdata(v) -> bool:
        return isinstance(v, dict) and (not v or "tags" in v or "descriptions" in v)

    if not ("tags" in raw or "descriptions" in raw) and all(
            _is_userdata(v) for v in raw.values()):
        return dict(raw)  # already namespaced by dataset
    if "tags" in raw or "descriptions" in raw:
        userdata = raw  # a single (non-namespaced) user_data dict
    else:
        userdata = {"tags": raw, "descriptions": {}}  # legacy flat name->tags
    return {key: userdata}


class _StudioRoot(QWidget):
    """Root container for the frameless Studio chrome.

    Paints a rounded, soft-gradient backdrop over the window's translucent
    background (giving the window its rounded corners + ambient frame).
    """

    _RADIUS = 14

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("studioroot")

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, self._RADIUS, self._RADIUS)
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QColor("#FCFBFA"))
        grad.setColorAt(0.55, QColor("#F6F2F4"))
        grad.setColorAt(1.0, QColor("#EFE8EF"))
        p.fillPath(path, grad)
        p.setPen(QPen(QColor(0, 0, 0, 22), 1))
        p.drawPath(path)
        p.end()


class MainWindow(QMainWindow):
    """Top-level window holding one tab per registered `DiiveTab`."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self._config = config or {}
        self._base_title = f"diive {diive.__version__}"
        self.setWindowTitle(self._base_title)
        from diive.gui.splash import app_icon
        self.setWindowIcon(app_icon())
        # Restore saved window geometry if available, else size to screen.
        geo = self._config.get("geometry")
        if geo:
            self.restoreGeometry(QByteArray.fromBase64(geo.encode("ascii")))
        else:
            self._size_to_screen()
        # The stylesheet is applied app-wide by theme.manager (see run()), so
        # live edits propagate everywhere; no per-window stylesheet here.

        # Retain the DiiveTab instances for the window's lifetime. Qt owns the
        # QWidgets, but the Python tab objects (which hold the signal slots,
        # e.g. PlottingTab._render) would otherwise be garbage-collected after
        # this loop, leaving their signal connections inert.
        # `_full_data` holds the complete loaded record (plus any engineered
        # features); `_data` is `_full_data` optionally narrowed to `_range`
        # (a (start, end) tuple) for a non-destructive date-range subselection.
        # Tabs always see `_data`. Reset restores the full range.
        self._full_data = None
        self._data = None
        self._range: tuple | None = None
        self._source = ""
        self._created: set = set()  # user-engineered feature column names
        # Per-variable user metadata (tags/notes) is persisted namespaced by
        # dataset, so the same column name in two datasets keeps separate tags.
        self._dataset_key: str | None = None
        self._saved_metadata: dict = dict(self._config.get("variable_metadata") or {})
        self._menu_tab_list: list = []  # open menu-activated tabs (multi-instance)

        self._tabs = []
        self._header = None  # set only in Studio chrome (None => native)
        self._studio_tabs = False  # Studio pill tabs carry favicon-style glyphs
        self._tabwidget = QTabWidget()
        for tab_cls in TAB_CLASSES:
            tab = tab_cls()
            self._tabs.append(tab)
            self._tabwidget.addTab(tab.widget(), tab.title)
            if hasattr(tab, "variableDeleted"):
                tab.variableDeleted.connect(self._delete_variable)
        # Tabs can be dragged to reorder and renamed by double-clicking. Menu
        # tabs get a (custom, clearly visible) close button on open; the
        # always-on Overview/Log are removed here so they stay open.
        self._tabwidget.setTabsClosable(True)
        self._tabwidget.setMovable(True)
        bar = self._tabwidget.tabBar()
        for i in range(self._tabwidget.count()):
            bar.setTabButton(i, QTabBar.ButtonPosition.RightSide, None)
        self._tabwidget.tabCloseRequested.connect(self._on_tab_close)
        self._tabwidget.tabBarDoubleClicked.connect(self._rename_tab)
        # Per-tab "pin" (freeze data): pinned tabs keep their current dataset and
        # are skipped by data pushes. Toggle via the tab's right-click menu.
        self._pinned: set = set()
        bar = self._tabwidget.tabBar()
        bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._tab_context_menu)

        # The GUI has a single look: the frameless Studio shell (custom header +
        # pill tabs).
        self._build_studio_chrome()

        # Auto-load the bundled example data so the app is usable on startup.
        self._load_example()

    def _build_studio_chrome(self) -> None:
        """Frameless rounded shell: custom header + tabs."""
        from diive.gui.icons import menu_icon
        from diive.gui.widgets.frameless import FramelessResizeHelper
        from diive.gui.widgets.header_bar import StudioHeaderBar

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._tabwidget.setObjectName("studiotabs")  # scopes the pill-tab QSS
        # Studio pills carry a small favicon-style glyph; give the always-on
        # tabs theirs now and flag later menu tabs to get one on open.
        self._studio_tabs = True
        # Favicon-style glyph on each pill, kept at its native 16px so it stays
        # crisp (the row height is grown via the tab font-size in the QSS, which
        # QTabWidget honours without top-aligning the inactive tabs).
        self._tabwidget.setIconSize(QSize(16, 16))
        for i in range(self._tabwidget.count()):
            self._tabwidget.setTabIcon(i, menu_icon(self._tabwidget.tabText(i)))

        root = _StudioRoot()
        rlay = QVBoxLayout(root)
        rlay.setContentsMargins(8, 8, 8, 8)
        rlay.setSpacing(6)

        self._header = StudioHeaderBar()
        # Each top-level menu (File, Data, Plot, …) becomes an inline dropdown
        # button in the header instead of a native menu bar; the same builder
        # populates both shells. Studio menus are translucent so the QSS rounds
        # their corners into a white card (no square popup-shadow artefact).
        def _add_menu(name):
            m = QMenu(self)
            m.setObjectName("studiomenu")
            # Frameless + translucent so the QSS-rounded white card has no
            # opaque (black) corners behind its rounded border, and no square
            # native popup shadow pokes out past the radius.
            m.setWindowFlags(m.windowFlags()
                             | Qt.WindowType.FramelessWindowHint
                             | Qt.WindowType.NoDropShadowWindowHint)
            m.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self._header.add_menu(name, m)
            return m
        self._build_menus(_add_menu)

        rlay.addWidget(self._header)
        rlay.addWidget(self._tabwidget, 1)
        self.setCentralWidget(root)

        # Native edge/corner resize (frameless windows lose the OS grips).
        self._resize_helper = FramelessResizeHelper(self, root)

    def _size_to_screen(self) -> None:
        """Size the window relative to the available screen and center it."""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1100, 720)
            return
        avail = screen.availableGeometry()
        self.resize(int(avail.width() * 0.88), int(avail.height() * 0.88))
        frame = self.frameGeometry()
        frame.moveCenter(avail.center())
        self.move(frame.topLeft())

    def _build_menus(self, add_menu) -> None:
        """Populate the menu tree, creating each top-level menu via `add_menu`.

        `add_menu(name) -> QMenu` decouples the menu contents from the shell:
        the Studio chrome passes a callback that adds each menu as an inline
        dropdown button in the header bar.
        """
        from diive.gui.icons import menu_icon

        def _act(text, slot, shortcut=None):
            """A QAction with a keyword-matched menu icon."""
            action = QAction(menu_icon(text), text, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            return action

        file_menu = add_menu("&File")
        file_menu.addAction(_act("&Open data file...", self._open_file, "Ctrl+O"))
        file_menu.addAction(_act("Load &example data", self._load_example))
        file_menu.addSeparator()
        file_menu.addAction(_act("&Save data as parquet...", self._save_file, "Ctrl+S"))
        file_menu.addSeparator()
        file_menu.addAction(_act("E&xit", self.close, "Ctrl+Q"))

        data_menu = add_menu("&Data")
        data_menu.addAction(_act("Select date &range...", self._select_daterange, "Ctrl+R"))
        self._reset_range_act = _act("Reset to &full range", self._reset_range)
        self._reset_range_act.setEnabled(False)
        data_menu.addAction(self._reset_range_act)
        # Menu-tab entries that belong under Data (e.g. Select variables) are
        # merged into this manually-built menu rather than getting their own.
        data_menu.addSeparator()
        for label in MENU_TABS.get("Data", {}):
            act = QAction(menu_icon(label), label, self)
            act.triggered.connect(lambda _checked, lab=label: self._open_menu_tab(lab))
            data_menu.addAction(act)

        for menu_name, group in MENU_TABS.items():
            if menu_name == "Data":
                continue  # already merged into the Data menu built above
            menu = add_menu(f"&{menu_name}")
            for label in group:
                act = QAction(menu_icon(label), label, self)
                act.triggered.connect(
                    lambda _checked, lab=label: self._open_menu_tab(lab))
                menu.addAction(act)

        help_menu = add_menu("&Help")
        help_menu.addAction(_act("&About", self._about))

    def _push_data(self) -> None:
        for tab in self._tabs:
            if tab in self._pinned:
                continue  # frozen: keep its own dataset
            tab.on_data_loaded(self._data, self._created)

    def _set_data(self, df, source: str) -> None:
        """Set a freshly loaded dataset, reset created features + range, push."""
        self._full_data = df
        self._source = source
        self._range = None  # a new dataset starts at its full range
        self._created = set()  # fresh dataset has no user-created features
        # Reset per-variable metadata to an "original" baseline for the new
        # columns, then re-apply user tags: persisted favorites first, then any
        # set earlier this session (so an in-session reload keeps them too).
        store = metadata_store.manager.store
        # Stash the outgoing dataset's user edits (tags/notes) before the store
        # is reset, so switching datasets — or reloading this one — keeps them.
        if self._dataset_key is not None:
            self._saved_metadata[self._dataset_key] = store.user_data()
        else:  # first load: migrate any pre-namespacing config onto this dataset
            self._saved_metadata = _namespace_metadata(self._saved_metadata, source)
        store.record_original(
            df.columns, operation=f"Imported from {source}",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
        store.load_user_data(self._saved_metadata.get(source) or {})
        self._dataset_key = source
        metadata_store.manager.notify()
        self._apply_range()
        self._tabwidget.setCurrentIndex(0)  # show the Overview tab on load

    def _apply_range(self) -> None:
        """Derive `_data` from `_full_data` (+ active `_range`) and push to tabs.

        The full record is kept in `_full_data`, so narrowing or resetting the
        range is non-destructive. The window title reflects the active window.
        """
        if self._full_data is None:
            return
        if self._range is None:
            self._data = self._full_data
            title = f"{self._base_title} — {self._source}"
        else:
            start, end = self._range
            self._data = diive.times.keep_daterange(self._full_data, start, end)
            title = (f"{self._base_title} — {self._source} "
                     f"[{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}]")
        self.setWindowTitle(title)
        if self._header is not None:
            self._header.set_title(title)
        self._reset_range_act.setEnabled(self._range is not None)
        self._push_data()

    def _select_daterange(self) -> None:
        """Pick a from/to window and narrow the dataset to it (non-destructive)."""
        if self._full_data is None or self._full_data.empty:
            QMessageBox.information(self, "Select date range", "No data loaded yet.")
            return
        full_start = self._full_data.index.min()
        full_end = self._full_data.index.max()
        cur_start, cur_end = self._range if self._range else (full_start, full_end)
        dlg = DateRangeDialog(full_start, full_end, cur_start, cur_end, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        start, end = dlg.selected_range()
        self._range = None if (start <= full_start and end >= full_end) else (start, end)
        self._apply_range()
        if self._data is not None:
            self.statusBar().showMessage(
                f"Date range: {len(self._data)} records selected", 5000)

    def _reset_range(self) -> None:
        """Revert to the full loaded date range (discard the subselection)."""
        if self._range is None:
            return
        self._range = None
        self._apply_range()
        self.statusBar().showMessage("Reverted to full date range", 5000)

    def _open_menu_tab(self, label: str) -> None:
        """Open a menu-activated tab.

        Most tabs open a new, numbered instance each time (Heatmap 1, 2, ...);
        singletons (see registry.SINGLE_INSTANCE_TABS) focus the existing one.
        """
        index = 0
        if label in SINGLE_INSTANCE_TABS:
            for tab in self._menu_tab_list:
                if tab._menu_label == label:
                    self._tabwidget.setCurrentWidget(tab.widget())
                    return
            title = label
        else:
            index = self._next_menu_index(label)
            title = f"{label} {index}"

        tab = MENU_TAB_CLASSES[label]()
        tab._menu_label = label
        tab._menu_index = index
        self._menu_tab_list.append(tab)
        self._tabs.append(tab)  # now receives data pushes
        # Build first (widget()/build sets featuresCreated) before connecting.
        idx = self._tabwidget.addTab(tab.widget(), title)
        self._tabwidget.tabBar().setTabButton(
            idx, QTabBar.ButtonPosition.RightSide,
            self._make_close_button(tab.widget()))
        if self._studio_tabs:  # favicon-style glyph on the pill (match the label)
            from diive.gui.icons import menu_icon
            self._tabwidget.setTabIcon(idx, menu_icon(label))
        if hasattr(tab, "featuresCreated"):
            tab.featuresCreated.connect(self._add_features)
        if hasattr(tab, "subsetSelected"):
            tab.subsetSelected.connect(self._show_overview_subset)
        if self._data is not None:
            tab.on_data_loaded(self._data, self._created)  # up to date on open
        self._tabwidget.setCurrentWidget(tab.widget())

    def _next_menu_index(self, label: str) -> int:
        """Smallest unused 1-based index among open tabs of this menu label."""
        used = {getattr(t, "_menu_index", 0)
                for t in self._menu_tab_list if t._menu_label == label}
        i = 1
        while i in used:
            i += 1
        return i

    def _on_tab_close(self, index: int) -> None:
        """Close a menu-opened tab and de-register it from data pushes.

        Only menu tabs are closable — the always-on Overview/Log have no close
        button and must also resist middle-click closing (QTabBar emits
        `tabCloseRequested` on middle-click regardless), so requests for them are
        ignored. After closing, focus falls back to the tab to the left, except
        never the Log tab (jump to Overview instead).
        """
        widget = self._tabwidget.widget(index)
        tab = next((t for t in self._menu_tab_list if t.widget() is widget), None)
        if tab is None:
            return  # always-on tab (Overview/Log): not closable
        self._menu_tab_list.remove(tab)
        if tab in self._tabs:
            self._tabs.remove(tab)
        self._pinned.discard(tab)
        self._tabwidget.removeTab(index)
        if self._tabwidget.count() == 0:
            return
        target = max(0, index - 1)
        if self._tabwidget.tabText(target) == "Log":
            target = 0  # Overview
        self._tabwidget.setCurrentIndex(target)

    def _make_close_button(self, widget) -> QToolButton:
        """A small, clearly visible "×" close button bound to `widget`'s tab."""
        from diive.gui.icons import close_icon
        btn = QToolButton()
        btn.setObjectName("tabclose")
        btn.setAutoRaise(True)
        btn.setIcon(close_icon())
        btn.setIconSize(QSize(12, 12))
        btn.setFixedSize(18, 18)
        btn.setToolTip("Close tab")
        btn.clicked.connect(lambda: self._close_widget(widget))
        return btn

    def _close_widget(self, widget) -> None:
        """Close the tab currently hosting `widget` (its index may have moved)."""
        idx = self._tabwidget.indexOf(widget)
        if idx >= 0:
            self._on_tab_close(idx)

    def _tab_context_menu(self, pos) -> None:
        """Right-click a tab to pin/unpin it (freeze/follow the dataset)."""
        bar = self._tabwidget.tabBar()
        index = bar.tabAt(pos)
        if index < 0:
            return
        widget = self._tabwidget.widget(index)
        # Only menu tabs are pinnable; the always-on Overview/Log stay live.
        tab = next((t for t in self._menu_tab_list if t.widget() is widget), None)
        if tab is None:
            return
        menu = QMenu(self)
        pinned = tab in self._pinned
        label = "Unpin tab (follow data)" if pinned else "Pin tab (freeze data)"
        menu.addAction(label).triggered.connect(lambda: self._toggle_pin(tab))
        menu.exec(bar.mapToGlobal(pos))

    def _toggle_pin(self, tab) -> None:
        """Freeze a tab on its current dataset, or unfreeze and re-sync it.

        Only menu tabs are pinnable; Overview/Log are always live.
        """
        if tab not in self._menu_tab_list:
            return
        if tab in self._pinned:
            self._pinned.discard(tab)
            if self._data is not None:  # catch up to the current dataset
                tab.on_data_loaded(self._data, self._created)
        else:
            self._pinned.add(tab)
        self._refresh_tab_icon(tab)

    def _refresh_tab_icon(self, tab) -> None:
        """Pinned tabs show a pin glyph; otherwise restore the favicon (Studio)."""
        index = self._tabwidget.indexOf(tab.widget())
        if index < 0:
            return
        if tab in self._pinned:
            from diive.gui.icons import pin_icon
            self._tabwidget.setTabIcon(index, pin_icon())
        elif self._studio_tabs:
            from diive.gui.icons import menu_icon
            label = getattr(tab, "_menu_label", None) or tab.title
            self._tabwidget.setTabIcon(index, menu_icon(label))
        else:
            self._tabwidget.setTabIcon(index, QIcon())

    def _rename_tab(self, index: int) -> None:
        """Rename a tab on double-click (changes the display label only)."""
        if index < 0:
            return
        new, ok = QInputDialog.getText(
            self, "Rename tab", "Tab name:", text=self._tabwidget.tabText(index))
        if ok and new.strip():
            self._tabwidget.setTabText(index, new.strip())

    def _add_features(self, new_df) -> None:
        """Merge engineered features into the dataset and re-push to tabs.

        Features merge into `_full_data` (the full record), so they survive a
        later range reset; rows outside a computed feature's range align to NaN.
        The active range is then re-derived so tabs see the new columns.
        """
        if new_df is None or new_df.empty or self._full_data is None:
            return
        for col in new_df.columns:
            self._full_data[col] = new_df[col]  # aligns on index
        self._created |= {str(c) for c in new_df.columns}
        # Record provenance the emitting tab attached to the frame (origin,
        # parent, operation, params, tags); stamp the time here (the library
        # model stays free of wall-clock calls). Frames without it are ignored.
        attrs = getattr(new_df, "attrs", {}).get(metadata_store.ATTRS_KEY) \
            if hasattr(new_df, "attrs") else None
        if attrs:
            metadata_store.manager.store.from_attrs(
                attrs, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
            metadata_store.manager.notify()
        self._apply_range()

    def _delete_variable(self, name: str) -> None:
        """Drop a column from the dataset (non-destructive to the source file).

        Removes it from the full record and re-derives the active range so every
        tab drops it. The on-disk file is untouched; re-loading restores it.
        """
        if self._full_data is None or name not in self._full_data.columns:
            return
        if QMessageBox.question(
                self, "Delete variable",
                f"Remove '{name}' from the loaded dataset?\n\n"
                "This affects the in-memory data only; the source file is "
                "untouched.") != QMessageBox.StandardButton.Yes:
            return
        self._full_data = self._full_data.drop(columns=[name])
        self._created.discard(name)
        metadata_store.manager.store.drop(name)
        metadata_store.manager.notify()
        self._apply_range()
        self.statusBar().showMessage(f"Deleted variable '{name}'", 5000)

    def _show_overview_subset(self, var_names: list) -> None:
        """Restrict the Overview tab's variable list to the selected subset
        (from the 'Select variables' tab). Overview-only; data is untouched."""
        for tab in self._tabs:
            if hasattr(tab, "show_variable_subset"):
                tab.show_variable_subset(var_names)
                break
        self.statusBar().showMessage(
            f"Overview showing {len(var_names)} selected variables", 5000)

    def _load_example(self) -> None:
        df = diive.load_exampledata_parquet()
        self._set_data(df, source="example data (CH-DAV)")

    def _open_file(self) -> None:
        dlg = OpenDataDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.dataframe is not None:
            self._set_data(dlg.dataframe, source=dlg.source_name)

    def _save_file(self) -> None:
        """Save the current dataset as a diive-format parquet file."""
        if self._data is None:
            QMessageBox.information(self, "Save data", "No data loaded yet.")
            return
        # Need a valid timestamp index name; ask if the current one isn't valid.
        ts_name = self._data.index.name
        if ts_name not in ALLOWED_TIMESTAMP_NAMES:
            ts_name, ok = QInputDialog.getItem(
                self, "Timestamp name",
                "Name the timestamp index (what the timestamp marks):",
                ALLOWED_TIMESTAMP_NAMES, 1, False)
            if not ok:
                return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save data as parquet", "data.parquet", "Parquet (*.parquet)")
        if not path:
            return
        p = Path(path)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            diive.save_parquet(
                filename=p.stem, data=self._data, outpath=str(p.parent),
                enforce_diive_format=True, timestamp_name=ts_name)
        except Exception as err:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Save failed", f"Could not save:\n{err}")
            return
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Saved {p.stem}.parquet", 5000)

    def _about(self) -> None:
        # Reuse the startup splash artwork as the About dialog.
        from diive.gui.splash import show_about
        show_about(self)

    def closeEvent(self, event) -> None:
        """Persist preferences (theme, window geometry, last filetype) on exit."""
        from diive.gui.widgets import open_data_dialog as odd
        # Fold the current dataset's user edits into the namespaced blob before
        # writing. Only user content (tags, notes) persists, keyed by dataset;
        # provenance/origin regenerate per session as operations run.
        if self._dataset_key is not None:
            self._saved_metadata[self._dataset_key] = \
                metadata_store.manager.store.user_data()
        config.save_config({
            "theme": theme.manager.as_dict(),
            "site": site.manager.as_dict(),
            "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
            "last_filetype": odd._last_choice,
            "variable_metadata": self._saved_metadata,
        })
        super().closeEvent(event)


def run() -> int:
    """Boot the QApplication, show the main window, run the event loop."""
    # Windows groups a Python process under python.exe in the taskbar (and uses
    # its icon) unless the app declares its own AppUserModelID first.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("diive.gui")
        except Exception:
            pass

    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("diive")
    app.setApplicationName("diive-gui")
    # Fusion style honours stylesheet item-selection colours consistently
    # (the native Windows style ignores them in combo-box popups).
    app.setStyle("Fusion")
    from diive.gui.splash import app_icon
    app.setWindowIcon(app_icon())  # taskbar / window icon (splash motif)

    # Restore saved preferences before building the window.
    cfg = config.load_config()
    theme.manager.load_dict(cfg.get("theme", {}))
    theme.manager.apply()
    site.manager.load_dict(cfg.get("site", {}))
    from diive.gui.widgets import open_data_dialog as odd
    odd._last_choice = cfg.get("last_filetype")

    # Splash while the window builds (it auto-loads the example dataset).
    from diive.gui.splash import create_splash, show_message
    splash = create_splash(app)
    splash.show()
    show_message(splash, "Loading…")
    app.processEvents()  # paint the splash before the (blocking) window build

    window = MainWindow(cfg)  # builds tabs + loads the example dataset
    window.show()
    # Keep the splash on top until the GUI is actually ready: the Overview
    # defers its first render by an event-loop tick (it renders synchronously
    # when that fires), so pump events to drain those deferred renders while the
    # splash is still up, then drop it. Without this, the splash would close the
    # instant the (still-empty) window is shown.
    splash.raise_()
    for _ in range(3):
        app.processEvents()
    splash.finish(window)
    return app.exec()
