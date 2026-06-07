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

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QTabBar,
    QTabWidget,
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
        self._tabwidget = QTabWidget()
        for tab_cls in TAB_CLASSES:
            tab = tab_cls()
            self._tabs.append(tab)
            self._tabwidget.addTab(tab.widget(), tab.title)
        # Always-on tabs are not closable; menu-opened tabs (added later) are.
        self._tabwidget.setTabsClosable(True)
        bar = self._tabwidget.tabBar()
        for i in range(self._tabwidget.count()):
            bar.setTabButton(i, QTabBar.ButtonPosition.RightSide, None)
        self._tabwidget.tabCloseRequested.connect(self._on_tab_close)
        self.setCentralWidget(self._tabwidget)

        self._build_menu()
        # Auto-load the bundled example data so the app is usable on startup.
        self._load_example()

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

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        open_act = QAction("&Open data file...", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_file)
        file_menu.addAction(open_act)

        example_act = QAction("Load &example data", self)
        example_act.triggered.connect(self._load_example)
        file_menu.addAction(example_act)

        file_menu.addSeparator()
        save_act = QAction("&Save data as parquet...", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_file)
        file_menu.addAction(save_act)

        file_menu.addSeparator()
        exit_act = QAction("E&xit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        data_menu = menubar.addMenu("&Data")
        range_act = QAction("Select date &range...", self)
        range_act.setShortcut("Ctrl+R")
        range_act.triggered.connect(self._select_daterange)
        data_menu.addAction(range_act)
        self._reset_range_act = QAction("Reset to &full range", self)
        self._reset_range_act.triggered.connect(self._reset_range)
        self._reset_range_act.setEnabled(False)
        data_menu.addAction(self._reset_range_act)

        for menu_name, group in MENU_TABS.items():
            menu = menubar.addMenu(f"&{menu_name}")
            for label in group:
                act = QAction(label, self)
                act.triggered.connect(
                    lambda _checked, lab=label: self._open_menu_tab(lab))
                menu.addAction(act)

        help_menu = menubar.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._about)
        help_menu.addAction(about_act)

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
        self._tabwidget.addTab(tab.widget(), title)
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
        """Close a menu-opened tab (always-on tabs have no close button)."""
        widget = self._tabwidget.widget(index)
        for tab in list(self._menu_tab_list):
            if tab.widget() is widget:
                self._tabwidget.removeTab(index)
                if tab in self._tabs:
                    self._tabs.remove(tab)
                self._menu_tab_list.remove(tab)
                return

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
        QMessageBox.about(
            self, "About diive",
            f"diive {diive.__version__}\n\nTime series processing for "
            f"ecosystem and flux data.\nhttps://github.com/holukas/diive",
        )

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

    window = MainWindow(cfg)
    window.show()
    return app.exec()
