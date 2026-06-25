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

import pandas as pd

from PySide6.QtCore import QByteArray, QEvent, QRectF, QSize, Qt, QTimer
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
    QMessageBox,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import diive
from diive.core.utils.console import info, success
from diive.gui import config, events, metadata_store, site, theme
from diive.gui.registry import (
    MENU_TAB_CLASSES,
    MENU_TABS,
    SINGLE_INSTANCE_TABS,
    TAB_CLASSES,
)
from diive.core.io.files import ALLOWED_TIMESTAMP_NAMES
from diive.gui.widgets.daterange_dialog import DateRangeDialog
from diive.gui.widgets.menu import studio_menu
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


class _StudioTabBar(QTabBar):
    """Studio pill tab bar that tints the always-on (system) tabs.

    Per-tab colours can't be expressed in the pill QSS (a stylesheet can't
    target an individual `::tab`), so the always-on Overview/Log tabs are
    distinguished by painting a faint slate wash over their pill after the base
    style has drawn it. `system_widgets` (the always-on tabs' page widgets) is
    assigned by `MainWindow`; identity-based so it survives tab reordering.
    """

    _WASH = QColor(58, 77, 92, 30)  # ACCENT (#3A4D5C) at low alpha

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.system_widgets: set = set()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self.system_widgets:
            return
        tw = self.parentWidget()  # the hosting QTabWidget
        radius = float(theme.manager.tokens.get("RADIUS", "12"))
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._WASH)
        for i in range(self.count()):
            page = tw.widget(i) if tw is not None else None
            if page is None or page not in self.system_widgets:
                continue
            # Inset to the visible pill (QSS margin: 6px vertical, 4px horizontal).
            pill = QRectF(self.tabRect(i)).adjusted(4, 6, -4, -6)
            p.drawRoundedRect(pill, radius, radius)
        p.end()


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

    def __init__(self, config: dict | None = None, autoload: bool = True) -> None:
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
        # Active app-wide variable subset (from the Select-variables tab), or None
        # for the full variable list. Like `_range`, it narrows `_data` (via
        # `dv.keep_vars`) non-destructively — the full record stays in `_full_data`
        # and the subset is resettable. Pinned tabs are skipped by `_push_data`,
        # so a frozen tab keeps its own (unsubset) dataset.
        self._var_subset: list | None = None
        self._source = ""
        self._created: set = set()  # user-engineered feature column names
        # EVENT_* columns this session created from the event list, so reconciling
        # never drops an EVENT_-named column that arrived as plain data.
        self._event_columns: set = set()
        # Per-variable user metadata (tags/notes) is persisted namespaced by
        # dataset, so the same column name in two datasets keeps separate tags.
        self._dataset_key: str | None = None
        self._saved_metadata: dict = dict(self._config.get("variable_metadata") or {})
        # Open diive project (folder), if any — set by Open/Save-as project.
        self._project_dir: Path | None = None
        self._project_name: str | None = None
        self._last_project_dir: str = self._config.get("last_project_dir") or ""
        self._last_project: str = self._config.get("last_project") or ""
        self._menu_tab_list: list = []  # open menu-activated tabs (multi-instance)

        self._tabs = []
        self._header = None  # set only in Studio chrome (None => native)
        self._studio_tabs = False  # Studio pill tabs carry favicon-style glyphs
        self._tabwidget = QTabWidget()
        # Custom tab bar so the always-on Overview/Log tabs read slightly
        # differently (a faint slate wash); set before adding tabs.
        self._tabwidget.setTabBar(_StudioTabBar())
        for tab_cls in TAB_CLASSES:
            tab = tab_cls()
            self._tabs.append(tab)
            self._tabwidget.addTab(tab.widget(), tab.title)
        # The always-on tabs' page widgets — tinted by the tab bar and protected
        # from rename. Identity-based, so it survives reordering.
        self._system_widgets = {t.widget() for t in self._tabs}
        self._tabwidget.tabBar().system_widgets = self._system_widgets
        # Tabs can be dragged to reorder and renamed by double-clicking. Menu
        # tabs get a (custom, clearly visible) close button on open; the
        # always-on Overview/Log are removed here so they stay open.
        self._tabwidget.setTabsClosable(True)
        self._tabwidget.setMovable(True)
        bar = self._tabwidget.tabBar()
        for i in range(self._tabwidget.count()):
            bar.setTabButton(i, QTabBar.ButtonPosition.RightSide, None)
        self._tabwidget.tabCloseRequested.connect(self._on_tab_close)
        # Rename only on a *left* double-click. tabBarDoubleClicked fires for any
        # button, so an event filter records the last double-click button and the
        # rename slot ignores middle/right ones.
        self._last_tab_dblclick_left = True
        self._tabwidget.tabBar().installEventFilter(self)
        self._tabwidget.tabBarDoubleClicked.connect(self._rename_tab)
        # Per-tab "pin" (freeze data): pinned tabs keep their current dataset and
        # are skipped by data pushes. Toggle via the tab's right-click menu.
        self._pinned: set = set()
        bar = self._tabwidget.tabBar()
        bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._tab_context_menu)

        # Variable-list "Edit metadata…" (any tab) opens the Metadata explorer.
        metadata_store.manager.editRequested.connect(self._edit_metadata)
        # Variable-list "Rename…" (any tab) prompts for a new name and applies it.
        metadata_store.manager.renameRequested.connect(self._rename_one_variable)
        # Variable-list "Delete…" (any tab) confirms and drops the column.
        metadata_store.manager.deleteRequested.connect(self._delete_variable)

        # Keep each event's 0/1 flag column in sync with the event list, and the
        # menu toggle in sync with the visibility flag (which the Events tab also
        # edits). The Overview reacts to the same signal to redraw its overlays.
        events.manager.changed.connect(self._on_events_changed)
        # A category-palette recolour is presentation-only: just redraw the
        # Overview overlays (no flag-column rebuild).
        events.manager.categories_changed.connect(self._on_event_categories_changed)
        # A card's "show on Overview" jumps to the Overview and zooms to the event.
        events.manager.focus_requested.connect(self._focus_event_on_overview)

        # The GUI has a single look: the frameless Studio shell (custom header +
        # pill tabs).
        self._build_studio_chrome()

        # Auto-load on startup (the last project, else the example). The real
        # launch passes autoload=False and defers this onto the event loop so the
        # splash spinner animates (see `run`); tests build with the default.
        if autoload:
            self._initial_load()

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
            # Frameless + translucent (via studio_menu) so the QSS-rounded white
            # card has no opaque (black) corners and no native popup shadow.
            m = studio_menu(self)
            self._header.add_menu(name, m)
            return m
        self._build_menus(_add_menu)

        rlay.addWidget(self._header)
        rlay.addWidget(self._tabwidget, 1)
        self.setCentralWidget(root)

        # No status bar: the frameless Studio shell has no use for the
        # QMainWindow status bar's empty grey strip below the rounded card.
        # Transient feedback goes to the Log tab instead (see _notify).
        self.setStatusBar(None)

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
        file_menu.addAction(_act("Open &project...", self._open_project, "Ctrl+Shift+O"))
        file_menu.addAction(_act("Load &example data", self._load_example))
        file_menu.addSeparator()
        file_menu.addAction(_act("&Save project", self._save_project, "Ctrl+S"))
        file_menu.addAction(_act("Save project &as...", self._save_project_as, "Ctrl+Shift+S"))
        file_menu.addAction(_act("Save data as par&quet...", self._save_file))
        file_menu.addSeparator()
        file_menu.addAction(_act("E&xit", self.close, "Ctrl+Q"))

        data_menu = add_menu("&Data")
        data_menu.addAction(_act("Select date &range...", self._select_daterange, "Ctrl+R"))
        self._reset_range_act = _act("Reset to &full range", self._reset_range)
        self._reset_range_act.setEnabled(False)
        data_menu.addAction(self._reset_range_act)
        self._reset_subset_act = _act("Reset to all &variables", self._reset_var_subset)
        self._reset_subset_act.setEnabled(False)
        data_menu.addAction(self._reset_subset_act)
        data_menu.addSeparator()
        data_menu.addAction(_act("Add timestamp co&lumn...", self._add_timestamp_column))
        # Menu-tab entries that belong under Data (e.g. Select variables) are
        # merged into this manually-built menu rather than getting their own.
        data_menu.addSeparator()
        for label in MENU_TABS.get("Data", {}):
            # Escape '&' so Qt doesn't read it as a mnemonic marker (would render
            # "Gaps & coverage" as an underlined "Gaps _coverage").
            act = QAction(menu_icon(label), label.replace("&", "&&"), self)
            act.triggered.connect(lambda _checked, lab=label: self._open_menu_tab(lab))
            data_menu.addAction(act)

        # Events: the Events tab (full list/edit UI) first, then a quick "add
        # one" + a master toggle for showing them on plots. Built manually so
        # the tab entry sits above the actions.
        events_menu = add_menu("&Events")
        for label in MENU_TABS.get("Events", {}):
            act = QAction(menu_icon(label), label.replace("&", "&&"), self)
            act.triggered.connect(lambda _checked, lab=label: self._open_menu_tab(lab))
            events_menu.addAction(act)
        events_menu.addSeparator()
        events_menu.addAction(_act("Add e&vent...", self._add_event))
        self._show_events_act = QAction(menu_icon("Show events"), "Show e&vents on plots", self)
        self._show_events_act.setCheckable(True)
        self._show_events_act.setChecked(events.manager.visible)
        self._show_events_act.toggled.connect(events.manager.set_visible)
        events_menu.addAction(self._show_events_act)

        for menu_name, group in MENU_TABS.items():
            if menu_name in ("Data", "Events"):
                continue  # built manually above
            menu = add_menu(f"&{menu_name}")
            for label in group:
                # Set the 3-D surface (GPU/VTK) apart from the 2-D plot methods.
                if menu_name == "Plot" and label == "3D surface":
                    menu.addSeparator()
                # Set the random-uncertainty tool apart from the partitioning tabs.
                if menu_name == "Flux" and label == "Random uncertainty (PAS20)":
                    menu.addSeparator()
                # Escape '&' so Qt doesn't read it as a mnemonic marker (would
                # render "Gaps & coverage" as an underlined "Gaps _coverage").
                act = QAction(menu_icon(label), label.replace("&", "&&"), self)
                act.triggered.connect(
                    lambda _checked, lab=label: self._open_menu_tab(lab))
                menu.addAction(act)
                # Set the stepwise screening (multi-method) apart from the
                # single-method outlier filters below it.
                if menu_name == "Outliers" and label == "Stepwise screening":
                    menu.addSeparator()
                # Set the full processing chain apart from the standalone tools.
                if menu_name == "Flux" and label == "Flux processing chain":
                    menu.addSeparator()

        help_menu = add_menu("&Help")
        help_menu.addAction(_act("&User manual", self._user_manual))
        help_menu.addAction(_act("&About", self._about))

    def _data_for(self, tab):
        """Dataset a tab should receive: the active (range/subset-narrowed)
        `_data`, except for tabs that opt into the full record via
        `wants_full_data` (the Select-variables tab, which must see every column
        to let the user pick a subset)."""
        if getattr(tab, "wants_full_data", False) and self._full_data is not None:
            return self._full_data
        return self._data

    def _push_data(self) -> None:
        for tab in self._tabs:
            if tab in self._pinned:
                continue  # frozen: keep its own dataset
            tab.on_data_loaded(self._data_for(tab), self._created)

    def _set_data(self, df, source: str, persist_metadata: bool = True) -> None:
        """Set a freshly loaded dataset, reset created features + range, push.

        `persist_metadata=False` loads a *clean* dataset: no saved tags/notes are
        applied and none are kept for it (used for the bundled example data, which
        should always open pristine).
        """
        self._full_data = df
        self._source = source
        self._range = None  # a new dataset starts at its full range
        self._var_subset = None  # and on its full variable list
        self._created = set()  # fresh dataset has no user-created features
        self._event_columns = set()  # rebuilt below from the live event list
        # A plain load leaves any open project; `_open_project` re-sets these
        # after calling here, so Ctrl+S can't overwrite a project with other data.
        self._project_dir = None
        self._project_name = None
        store = metadata_store.manager.store
        # Stash the outgoing dataset's user edits (tags/notes) before the store
        # is reset, so switching datasets — or reloading this one — keeps them.
        if self._dataset_key is not None:
            self._saved_metadata[self._dataset_key] = store.user_data()
        elif persist_metadata:  # first persisted load: migrate any old global config
            self._saved_metadata = _namespace_metadata(self._saved_metadata, source)
        # Reset per-variable metadata to an "original" baseline for the new columns.
        store.record_original(
            df.columns, operation=f"Imported from {source}",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
        if persist_metadata:
            store.load_user_data(self._saved_metadata.get(source) or {})
            self._dataset_key = source
        else:  # clean dataset: don't apply or keep any tags, and purge stale ones
            self._saved_metadata.pop(source, None)
            self._dataset_key = None
        metadata_store.manager.notify()
        # Recreate each event's 0/1 column on the new index (events persist across
        # loads; this also matches the EVENT_ columns a project's parquet carries).
        self._sync_event_columns()
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
        # Then narrow to the active variable subset. Names that no longer exist
        # (deleted/renamed away) are dropped here, so a stale subset can't raise.
        total = len(self._full_data.columns)
        if self._var_subset:
            keep = [v for v in self._var_subset if v in self._data.columns]
            if keep:
                self._data = diive.keep_vars(self._data, keep)
                title += f" [{len(keep)} of {total} variables]"
        self.setWindowTitle(title)
        if self._header is not None:
            self._header.set_title(title)
        self._reset_range_act.setEnabled(self._range is not None)
        self._reset_subset_act.setEnabled(bool(self._var_subset))
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
            info(f"Date range: {len(self._data)} records selected")

    def _reset_range(self) -> None:
        """Revert to the full loaded date range (discard the subselection)."""
        if self._range is None:
            return
        self._range = None
        self._apply_range()
        success("Reverted to full date range")

    def _edit_metadata(self, name: str) -> None:
        """Open (or focus) the Metadata explorer and select `name` there."""
        self._open_menu_tab("Metadata explorer")  # single-instance: focuses/opens
        for tab in self._menu_tab_list:
            if getattr(tab, "_menu_label", None) == "Metadata explorer":
                tab.select_variable(name)
                break

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
            tab.subsetSelected.connect(self._apply_var_subset)
        if hasattr(tab, "variablesRenamed"):
            tab.variablesRenamed.connect(self._rename_variables)
        if self._data is not None:
            tab.on_data_loaded(self._data_for(tab), self._created)  # up to date on open
        # Seed the variable picker with the subset currently in effect, so a
        # freshly opened picker reflects what's actually shown.
        if hasattr(tab, "set_active_subset") and self._var_subset:
            tab.set_active_subset(self._var_subset)
        self._tabwidget.setCurrentWidget(tab.widget())

    def _next_menu_index(self, label: str) -> int:
        """Smallest unused 1-based index among open tabs of this menu label."""
        used = {getattr(t, "_menu_index", 0)
                for t in self._menu_tab_list if t._menu_label == label}
        i = 1
        while i in used:
            i += 1
        return i

    # --- open-tab state (saved with a project) -------------------------
    def _open_tabs_state(self) -> list:
        """The open menu tabs (label/title/pinned) in tab-bar order, so a project
        can reopen the same workspace. Always-on Overview/Log are not recorded."""
        state = []
        for i in range(self._tabwidget.count()):
            widget = self._tabwidget.widget(i)
            tab = next((t for t in self._menu_tab_list if t.widget() is widget), None)
            if tab is None or getattr(tab, "_menu_label", None) not in MENU_TAB_CLASSES:
                continue
            entry = {
                "label": tab._menu_label,
                "title": self._tabwidget.tabText(i),
                "pinned": tab in self._pinned,
            }
            try:
                entry["state"] = tab.save_state()  # tab-specific inputs
            except Exception:
                entry["state"] = {}  # never let one tab's quirk break the save
            state.append(entry)
        return state

    def _close_all_menu_tabs(self) -> None:
        """Close every menu tab (e.g. before restoring a project's workspace)."""
        for tab in list(self._menu_tab_list):
            self._close_widget(tab.widget())

    def _restore_tabs(self, state) -> None:
        """Reopen the menu tabs recorded by `_open_tabs_state` (label/title/pin)."""
        for entry in state or []:
            label = entry.get("label")
            if label not in MENU_TAB_CLASSES:
                continue  # unknown/renamed feature — skip rather than crash
            self._open_menu_tab(label)
            tab = self._menu_tab_list[-1] if self._menu_tab_list else None
            if tab is None:
                continue
            idx = self._tabwidget.indexOf(tab.widget())
            if idx >= 0 and entry.get("title"):
                self._tabwidget.setTabText(idx, entry["title"])
            # Re-apply the tab's own inputs (the list is already populated, since
            # _open_menu_tab pushed the current data). Tolerant of stale state.
            try:
                tab.restore_state(entry.get("state") or {})
            except Exception:
                pass
            if entry.get("pinned"):
                self._toggle_pin(tab)
        self._tabwidget.setCurrentIndex(0)  # land on the Overview after restore

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
        menu = studio_menu(self)
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
                tab.on_data_loaded(self._data_for(tab), self._created)
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

    def eventFilter(self, obj, event):
        """Record the mouse button of a tab-bar double-click so `_rename_tab`
        can act on left double-clicks only (Qt's signal carries no button)."""
        if (obj is self._tabwidget.tabBar()
                and event.type() == QEvent.Type.MouseButtonDblClick):
            self._last_tab_dblclick_left = (
                event.button() == Qt.MouseButton.LeftButton)
        return super().eventFilter(obj, event)

    def _rename_tab(self, index: int) -> None:
        """Rename a tab on a left double-click (changes the display label only).

        The always-on Overview/Log tabs are not renamable.
        """
        if index < 0 or not self._last_tab_dblclick_left:
            return
        if self._tabwidget.widget(index) in self._system_widgets:
            return  # always-on Overview/Log: not renamable
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
        # While a variable subset is active, a freshly created column would be
        # filtered out (it isn't in the subset) and silently invisible — so add
        # it to the subset, mirroring the Overview's new-feature handling.
        if self._var_subset is not None:
            self._var_subset += [str(c) for c in new_df.columns
                                 if str(c) not in self._var_subset]
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

    def _add_timestamp_column(self) -> None:
        """Open the Add-timestamp-column dialog and append the derived column.

        Adds a START/MIDDLE/END timestamp (optionally strftime-formatted) as a new
        data column via the library's `format_timestamp`; the index is unchanged.
        The result is merged through `_add_features` so it gets provenance and
        survives a range reset like any engineered column.
        """
        from diive.core.metadata import DERIVED, ATTRS_KEY, provenance_attr
        from diive.gui.widgets.add_timestamp_dialog import AddTimestampColumnDialog
        from diive.times import format_timestamp, validate_timestamp_column_name

        if self._full_data is None or self._full_data.empty:
            QMessageBox.information(self, "Add timestamp column", "No data loaded yet.")
            return
        dlg = AddTimestampColumnDialog(self._full_data, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name, convention, fmt = dlg.column_name(), dlg.convention(), dlg.fmt()
        if not name:
            QMessageBox.warning(self, "Add timestamp column", "Please enter a column name.")
            return
        try:
            validate_timestamp_column_name(name, convention)
        except ValueError as err:
            QMessageBox.warning(self, "Add timestamp column", str(err))
            return
        if name in self._full_data.columns and QMessageBox.question(
                self, "Add timestamp column",
                f"Column '{name}' already exists. Overwrite it?") \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            series = format_timestamp(self._full_data, convention=convention, fmt=fmt)
        except Exception as err:
            QMessageBox.critical(self, "Add timestamp column",
                                 f"Could not build the column:\n{err}")
            return
        new_df = series.rename(name).to_frame()
        new_df.attrs[ATTRS_KEY] = {name: provenance_attr(
            origin=DERIVED, operation="Add timestamp column",
            params={"convention": convention, "format": fmt}, tags=["timestamp"])}
        self._add_features(new_df)

    def _add_event(self) -> None:
        """Open the Add-event dialog and register the event (creates its 0/1
        column + draws it on the plots via `events.manager.changed`)."""
        from diive.gui.widgets.add_event_dialog import AddEventDialog
        if self._full_data is None or self._full_data.empty:
            QMessageBox.information(self, "Add event", "No data loaded yet.")
            return
        dlg = AddEventDialog(self._full_data.index.min(),
                             self._full_data.index.max(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            events.manager.add(dlg.make_event())

    def _on_events_changed(self) -> None:
        """React to any event edit: keep the menu toggle in sync and reconcile
        each event's 0/1 flag column against the dataset."""
        if self._show_events_act.isChecked() != events.manager.visible:
            self._show_events_act.blockSignals(True)
            self._show_events_act.setChecked(events.manager.visible)
            self._show_events_act.blockSignals(False)
        # A column add/edit/delete re-renders the Overview through the data push;
        # a visibility-only toggle changes no column, so refresh it explicitly.
        if not self._sync_event_columns():
            overview = self._tabs[0]
            if hasattr(overview, "refresh_events"):
                overview.refresh_events()

    def _on_event_categories_changed(self) -> None:
        """Recolour the Overview's event overlays after a category-palette edit."""
        overview = self._tabs[0]
        if hasattr(overview, "refresh_events"):
            overview.refresh_events()

    def _focus_event_on_overview(self, start, end) -> None:
        """Switch to the Overview and zoom its time axis onto an event window."""
        self._tabwidget.setCurrentIndex(0)
        overview = self._tabs[0]
        if hasattr(overview, "focus_on"):
            overview.focus_on(start, end)

    def _sync_event_columns(self) -> bool:
        """Make the dataset's ``EVENT_*`` columns exactly match the event list.

        Each event becomes a 0/1 column (`dv.events.event_to_flag`); deleted
        events have their column dropped; edits rebuild the affected column. Only
        re-pushes when something actually changed, so a visibility-only toggle is
        cheap. The columns are real data (saved with the project, shown in every
        variable list, taggable), per the requested 'event = yes/no data column'.

        Returns True if any column was added/dropped/rebuilt (so the tabs were
        re-pushed), False otherwise (e.g. a visibility-only toggle).
        """
        if self._full_data is None:
            return False
        desired = {}
        for ev in events.manager.events:
            desired[ev.flag_name] = diive.events.event_to_flag(
                ev, self._full_data.index, name=ev.flag_name)
        changed = False
        added = []
        # Only drop EVENT_ columns we created (don't touch an EVENT_-named column
        # that arrived as plain data without a backing event).
        for col in list(self._event_columns):
            if col not in desired and col in self._full_data.columns:
                self._full_data = self._full_data.drop(columns=[col])
                self._created.discard(col)
                metadata_store.manager.store.drop(col)
                if self._var_subset is not None and col in self._var_subset:
                    self._var_subset.remove(col)
                changed = True
        for col, series in desired.items():
            if col not in self._full_data.columns \
                    or not self._full_data[col].equals(series):
                is_new = col not in self._full_data.columns
                self._full_data[col] = series
                self._created.add(col)
                if is_new:
                    added.append(col)
                    if self._var_subset is not None and col not in self._var_subset:
                        self._var_subset.append(col)
                changed = True
        self._event_columns = set(desired)
        if added:
            metadata_store.manager.store.record_original(
                added, operation="Event flag",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
            metadata_store.manager.notify()
        if changed:
            self._apply_range()
        return changed

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
        if self._var_subset is not None and name in self._var_subset:
            self._var_subset.remove(name)
        metadata_store.manager.store.drop(name)
        metadata_store.manager.notify()
        self._apply_range()
        success(f"Deleted variable '{name}'")

    def _rename_one_variable(self, name: str) -> None:
        """Prompt for a new name for one variable, then rename it (any tab)."""
        if self._full_data is None or name not in self._full_data.columns:
            return
        new, ok = QInputDialog.getText(
            self, "Rename variable", "New name:", text=name)
        new = new.strip()
        if not ok or not new or new == name:
            return
        if new in self._full_data.columns:
            QMessageBox.warning(
                self, "Rename variable",
                f"A variable named '{new}' already exists.")
            return
        self._rename_variables({name: new})

    def _rename_variables(self, mapping: dict) -> None:
        """Rename columns in the dataset (non-destructive to the source file).

        Renames in the full record, remaps the `created` set and the per-variable
        metadata store (origin/provenance/tags travel with the new name), then
        re-derives the active range so every tab sees the new names.
        """
        if self._full_data is None or not mapping:
            return
        mapping = {o: n for o, n in mapping.items() if o in self._full_data.columns}
        if not mapping:
            return
        self._full_data = self._full_data.rename(columns=mapping)
        self._created = {mapping.get(c, c) for c in self._created}
        if self._var_subset is not None:  # keep renamed vars in the active subset
            self._var_subset = [mapping.get(c, c) for c in self._var_subset]
        metadata_store.manager.store.rename(mapping)
        metadata_store.manager.notify()
        self._apply_range()
        success(f"Renamed {len(mapping)} variables")

    def _apply_var_subset(self, var_names: list) -> None:
        """Narrow the dataset to the selected variables, app-wide (from the
        'Select variables' tab). Non-destructive: the full record stays in
        `_full_data`, so it's resettable (Data > Reset to all variables). Every
        non-pinned tab sees only the subset; pinned (frozen) tabs are skipped by
        `_push_data` and keep their full dataset."""
        names = [n for n in var_names if self._full_data is not None
                 and n in self._full_data.columns]
        self._var_subset = names or None
        self._apply_range()
        n = len(self._var_subset or [])
        info(f"Showing {n} of {len(self._full_data.columns)} variables")

    def _reset_var_subset(self) -> None:
        """Revert to the full variable list (discard the subselection)."""
        if self._var_subset is None:
            return
        self._var_subset = None
        self._apply_range()
        success("Reverted to all variables")

    def _load_example(self) -> None:
        df = diive.load_exampledata_parquet()
        # The bundled example always opens clean — no persisted tags/notes.
        self._set_data(df, source="example data (CH-DAV)", persist_metadata=False)

    def _open_file(self) -> None:
        dlg = OpenDataDialog(self, can_add=self._full_data is not None)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.dataframe is None:
            return
        if dlg.load_mode == "add" and self._full_data is not None:
            self._add_dataset(dlg.dataframe, dlg.source_name)
        else:
            self._set_data(dlg.dataframe, source=dlg.source_name)

    def _add_dataset(self, new_df, source: str) -> None:
        """Merge a second loaded dataset onto the current one (shared index).

        Used for comparing datasets side by side (e.g. two processing runs of the
        same site). Columns align on the timestamp index via `_add_features`. A
        same-named column would overwrite the existing one, so collisions are
        flagged first — the user is expected to load with a label to avoid them.
        """
        collisions = [c for c in new_df.columns if c in self._full_data.columns]
        if collisions:
            preview = ", ".join(collisions[:5]) + ("..." if len(collisions) > 5 else "")
            resp = QMessageBox.question(
                self, "Variable name collision",
                f"{len(collisions)} variable(s) already exist and would be "
                f"overwritten:\n{preview}\n\nLoad again with a label (e.g. 'v2') "
                f"to keep them separate.\n\nOverwrite anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if resp != QMessageBox.StandardButton.Yes:
                return
        # Seed an "original" provenance baseline for the merged-in columns, as a
        # plain load does — `_add_features` only records provenance from attrs.
        metadata_store.manager.store.record_original(
            new_df.columns, operation=f"Imported from {source}",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
        self._add_features(new_df)
        metadata_store.manager.notify()
        success(f"Added {len(new_df.columns)} variable(s) from {source}")

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
        success(f"Saved {p.stem}.parquet")

    # --- projects ------------------------------------------------------
    def _save_project(self) -> None:
        """Ctrl+S: update the open project in place, or prompt if none is open."""
        if self._project_dir is not None:
            self._write_project(self._project_dir, self._project_name)
        else:
            self._save_project_as()

    def _save_project_as(self) -> None:
        """Pick a name + location and write a new diive project folder."""
        from diive.core.io.project import is_project, project_name_to_dirname
        from diive.gui.widgets.save_project_dialog import SaveProjectDialog
        if self._full_data is None:
            QMessageBox.information(self, "Save project", "No data loaded yet.")
            return
        default_name = self._project_name or (self._source or "project").split(".")[0]
        dlg = SaveProjectDialog(default_name, self._last_project_dir, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name, location = dlg.values()
        folder = Path(location) / project_name_to_dirname(name)
        if folder.exists() and not is_project(folder):
            QMessageBox.warning(
                self, "Save project",
                f"'{folder.name}' already exists and is not a diive project.")
            return
        if folder.exists() and QMessageBox.question(
                self, "Save project",
                f"Overwrite the existing project '{folder.name}'?") \
                != QMessageBox.StandardButton.Yes:
            return
        self._last_project_dir = location
        if self._write_project(folder, name):
            self._project_dir, self._project_name = folder, name
            self._last_project = str(folder)
            self._source = name
            self._apply_range()  # refresh title with the project name

    def _log_tab(self):
        """The always-on Log tab, located by type (order-independent)."""
        from diive.gui.tabs.log import LogTab
        return next((t for t in self._tabs if isinstance(t, LogTab)), None)

    def _write_project(self, folder, name: str) -> bool:
        """Write the current data + metadata + site/range into `folder`."""
        from diive.core.io.project import DiiveProject, save_project
        extras = {
            "site": site.manager.as_dict(),
            "events": events.manager.as_dict(),
            "range": ([self._range[0].isoformat(), self._range[1].isoformat()]
                      if self._range else None),
            "var_subset": list(self._var_subset) if self._var_subset else None,
            "created_columns": sorted(str(c) for c in self._created),
            "source": self._source,
            "saved": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "tabs": self._open_tabs_state(),  # reopen the same workspace on load
            "overview": self._tabs[0].save_state(),  # selected variable
            "active_tab": self._tabwidget.tabText(self._tabwidget.currentIndex()),
            "log": self._log_tab().save_state() if self._log_tab() else None,
        }
        project = DiiveProject(
            name=name, data=self._full_data,
            metadata=metadata_store.manager.store, extras=extras)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            save_project(folder, project)
        except Exception as err:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Save failed", f"Could not save project:\n{err}")
            return False
        QApplication.restoreOverrideCursor()
        success(f"Saved project '{name}'")
        return True

    def _initial_load(self) -> None:
        """Startup load: reopen the last project if it still exists, else the
        bundled example data."""
        from diive.core.io.project import is_project
        last = self._last_project
        if last and is_project(last) and self._load_project_folder(last, announce=False):
            return
        self._load_example()

    def _open_project(self) -> None:
        """Pick a diive project folder and open it."""
        from diive.core.io.project import is_project
        start = self._last_project_dir or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Open diive project", start)
        if not folder:
            return
        if not is_project(folder):
            QMessageBox.warning(
                self, "Open project",
                "That folder is not a diive project (no '__diive__' marker).")
            return
        self._load_project_folder(folder)

    def _load_project_folder(self, folder, announce: bool = True) -> bool:
        """Load a project from `folder`, restoring data + metadata + site + range.

        `announce=False` (startup) suppresses the error dialog and just reports
        failure so the caller can fall back to the example data.
        """
        from diive.core.io.project import load_project
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            project = load_project(folder)  # do the risky read before mutating state
        except Exception as err:
            QApplication.restoreOverrideCursor()
            if announce:
                QMessageBox.critical(self, "Open failed",
                                     f"Could not open project:\n{err}")
            return False
        QApplication.restoreOverrideCursor()

        # Restore site details first (so day/night-aware tabs see them on push).
        site_d = project.extras.get("site")
        if site_d:
            site.manager.load_dict(site_d)
            site.manager.changed.emit()
        # Clear the current workspace before loading, so the project's saved tabs
        # replace it (and stale tabs don't get the incoming data push).
        self._close_all_menu_tabs()
        # Load the data clean (no config tags), then overlay the project's full
        # metadata (origin/parent/provenance/tags/notes) authoritatively.
        self._set_data(project.data, source=project.name, persist_metadata=False)
        metadata_store.manager.store.load_dict(project.metadata.to_dict())
        self._created = {str(c) for c in project.extras.get("created_columns") or []}
        rng = project.extras.get("range")
        self._range = (pd.Timestamp(rng[0]), pd.Timestamp(rng[1])) if rng else None
        self._var_subset = list(project.extras.get("var_subset") or []) or None
        # Restore the project's events (rebuilds their 0/1 columns via changed →
        # _sync_event_columns; matches the EVENT_ columns the parquet carries).
        events.manager.load_dict(project.extras.get("events") or {})
        self._project_dir, self._project_name = Path(folder), project.name
        self._last_project_dir = str(Path(folder).parent)
        self._last_project = str(Path(folder))
        self._apply_range()  # re-push with restored created-columns + range
        metadata_store.manager.notify()
        # Overview's selected variable, then the saved workspace tabs.
        try:
            self._tabs[0].restore_state(project.extras.get("overview") or {})
        except Exception:
            pass
        self._restore_tabs(project.extras.get("tabs"))  # reopens tabs; lands on Overview
        log_tab = self._log_tab()
        if log_tab is not None:
            try:
                log_tab.restore_state(project.extras.get("log") or {})
            except Exception:
                pass
        self._restore_active_tab(project.extras.get("active_tab"))
        success(f"Opened project '{project.name}'")
        return True

    def _restore_active_tab(self, title) -> None:
        """Focus the tab the user had active when the project was saved."""
        if not title:
            return
        for i in range(self._tabwidget.count()):
            if self._tabwidget.tabText(i) == title:
                self._tabwidget.setCurrentIndex(i)
                return

    def _user_manual(self) -> None:
        """Open the styled HTML manual in the default browser.

        Falls back to the GitHub-hosted Markdown source if the bundled HTML is
        missing (e.g. a partial install)."""
        import sys

        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl

        candidates = [Path(__file__).with_name("MANUAL.html")]
        # Frozen (PyInstaller) builds: bundled data lives under _MEIPASS, where
        # the module's notional __file__ may not resolve.
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "diive" / "gui" / "MANUAL.html")
        for html in candidates:
            if html.is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(html)))
                return
        QDesktopServices.openUrl(
            QUrl("https://github.com/holukas/diive/blob/main/diive/gui/MANUAL.md"))

    def _about(self) -> None:
        # Reuse the startup splash artwork as the About dialog.
        from diive.gui.splash import show_about
        show_about(self)

    def closeEvent(self, event) -> None:
        """Persist preferences (theme, window geometry, last filetype) on exit."""
        from diive.gui.widgets import open_data_dialog as odd
        # Stop reacting to the app-wide event store once closed (the singleton
        # outlives the window; a stale window must not react to later edits).
        try:
            events.manager.changed.disconnect(self._on_events_changed)
        except (RuntimeError, TypeError):
            pass
        try:
            events.manager.categories_changed.disconnect(
                self._on_event_categories_changed)
        except (RuntimeError, TypeError):
            pass
        try:
            events.manager.focus_requested.disconnect(
                self._focus_event_on_overview)
        except (RuntimeError, TypeError):
            pass
        # Fold the current dataset's user edits into the namespaced blob before
        # writing. Only user content (tags, notes) persists, keyed by dataset;
        # provenance/origin regenerate per session as operations run.
        if self._dataset_key is not None:
            self._saved_metadata[self._dataset_key] = \
                metadata_store.manager.store.user_data()
        config.save_config({
            "theme": theme.manager.as_dict(),
            "site": site.manager.as_dict(),
            "events": events.manager.as_dict(),
            "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
            "last_filetype": odd._last_choice,
            "variable_metadata": self._saved_metadata,
            "last_project_dir": self._last_project_dir,
            "last_project": self._last_project,
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
    # Strip the native frame/shadow from every combo-box popup (black bars on the
    # frameless translucent window) — one app-wide filter covers all dropdowns.
    from diive.gui.widgets.combo import install_combo_popup_fix
    install_combo_popup_fix(app)
    from diive.gui.splash import app_icon
    icon = app_icon()  # taskbar / window icon (splash motif)
    app.setWindowIcon(icon)

    # Restore saved preferences before building the window.
    cfg = config.load_config()
    theme.manager.load_dict(cfg.get("theme", {}))
    theme.manager.apply()
    site.manager.load_dict(cfg.get("site", {}))
    events.manager.load_dict(cfg.get("events", {}))
    from diive.gui.widgets import open_data_dialog as odd
    odd._last_choice = cfg.get("last_filetype")

    # Splash while the window builds (it auto-loads the example dataset).
    from diive.gui.splash import create_splash, show_message
    splash = create_splash(app)
    splash.show()
    show_message(splash, "Loading…")
    app.processEvents()  # paint the splash before the (blocking) window build

    window = MainWindow(cfg, autoload=False)  # build the UI now; load data deferred
    window.showMaximized()  # always start maximized (restored geometry is the un-maximize size)
    splash.raise_()

    # Defer the data load (last project or example) onto the event loop so the
    # splash's spinner actually animates during it, instead of freezing behind a
    # blocking load in the constructor. The Overview also defers its first render
    # a tick, so pump a few times to drain those before dropping the splash.
    def _finish_startup() -> None:
        window._initial_load()
        for _ in range(3):
            app.processEvents()
        splash.finish(window)
        # Re-assert the window icon now that the native window exists. A
        # frameless + translucent window's HWND is created in a way that can
        # miss the icon set before show(), so Windows sometimes shows a blank /
        # default taskbar icon. Re-applying once the handle is live fixes it.
        window.setWindowIcon(icon)

    QTimer.singleShot(0, _finish_startup)
    return app.exec()
