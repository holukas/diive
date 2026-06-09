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
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import (
    QAction,
    QColor,
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
from diive.gui import config, theme
from diive.gui.registry import (
    MENU_TAB_CLASSES,
    MENU_TABS,
    SINGLE_INSTANCE_TABS,
    TAB_CLASSES,
)
from diive.core.io.files import ALLOWED_TIMESTAMP_NAMES
from diive.gui.widgets.daterange_dialog import DateRangeDialog
from diive.gui.widgets.open_data_dialog import OpenDataDialog


class _StudioRoot(QWidget):
    """Root container for the frameless Studio chrome.

    Paints a rounded, soft-gradient backdrop over the window's translucent
    background (giving the window its rounded corners + ambient frame) and
    keeps a floating overlay (the pill toolbar) anchored bottom-centre.
    """

    _RADIUS = 14

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("studioroot")
        self._overlay: QWidget | None = None

    def set_overlay(self, widget: QWidget) -> None:
        self._overlay = widget
        widget.raise_()
        self._place_overlay()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_overlay()

    def _place_overlay(self) -> None:
        if self._overlay is None:
            return
        self._overlay.adjustSize()
        x = (self.width() - self._overlay.width()) // 2
        y = self.height() - self._overlay.height() - 22
        self._overlay.move(max(0, x), max(0, y))

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
        self._menu_tab_list: list = []  # open menu-activated tabs (multi-instance)

        self._tabs = []
        self._header = None  # set only in Studio chrome (None => native)
        self._studio_tabs = False  # Studio pill tabs carry favicon-style glyphs
        self._tabwidget = QTabWidget()
        for tab_cls in TAB_CLASSES:
            tab = tab_cls()
            self._tabs.append(tab)
            self._tabwidget.addTab(tab.widget(), tab.title)
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

        # Two window shells, chosen by the active theme preset's `chrome` flag
        # (read once here; switching presets needs a relaunch to swap chrome).
        if theme.manager.chrome == "studio":
            self._build_studio_chrome()
        else:
            self._build_native_chrome()

        # Auto-load the bundled example data so the app is usable on startup.
        self._load_example()

    def _build_native_chrome(self) -> None:
        """Standard QMainWindow shell: native title bar + menu bar + tabs."""
        theme.manager.built_chrome = "native"
        self.setCentralWidget(self._tabwidget)
        self._build_menus(self.menuBar().addMenu)

    def _build_studio_chrome(self) -> None:
        """Frameless rounded shell: custom header + tabs + floating pill bar."""
        from diive.gui.icons import menu_icon
        from diive.gui.widgets.frameless import FramelessResizeHelper
        from diive.gui.widgets.header_bar import StudioHeaderBar
        from diive.gui.widgets.pill_toolbar import PillToolbar

        theme.manager.built_chrome = "studio"
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

        # Floating pill toolbar overlaying the content, anchored bottom-centre.
        pill = PillToolbar(root)
        pill.add_action(menu_icon("Open"), "Open data file…", self._open_file)
        pill.add_action(menu_icon("Save"), "Save data as parquet…", self._save_file)
        pill.add_action(menu_icon("date range"), "Select date range…", self._select_daterange)
        pill.add_action(menu_icon("reset"), "Reset to full range", self._reset_range)
        root.set_overlay(pill)

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
        native chrome passes `self.menuBar().addMenu`, Studio chrome passes a
        callback that adds submenus under the header's "≡" popup.
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

        for menu_name, group in MENU_TABS.items():
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
            tab.on_data_loaded(self._data, self._created)

    def _set_data(self, df, source: str) -> None:
        """Set a freshly loaded dataset, reset created features + range, push."""
        self._full_data = df
        self._source = source
        self._range = None  # a new dataset starts at its full range
        self._created = set()  # fresh dataset has no user-created features
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
        """Close any tab; de-register it so it stops receiving data pushes.

        After closing, fall back to the tab to the left of the one closed —
        except never land on the Log tab: if that's where we'd land, jump
        straight to the Overview tab instead.
        """
        widget = self._tabwidget.widget(index)
        for tab in list(self._menu_tab_list):
            if tab.widget() is widget:
                self._menu_tab_list.remove(tab)
                break
        for tab in list(self._tabs):
            if tab.widget() is widget:
                self._tabs.remove(tab)
                break
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
        self._apply_range()

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
        config.save_config({
            "theme": theme.manager.as_dict(),
            "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
            "last_filetype": odd._last_choice,
        })
        super().closeEvent(event)


def run() -> int:
    """Boot the QApplication, show the main window, run the event loop."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("diive")
    app.setApplicationName("diive-gui")
    # Fusion style honours stylesheet item-selection colours consistently
    # (the native Windows style ignores them in combo-box popups).
    app.setStyle("Fusion")

    # Restore saved preferences before building the window.
    cfg = config.load_config()
    theme.manager.load_dict(cfg.get("theme", {}))
    theme.manager.apply()
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
