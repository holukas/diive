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

from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QMessageBox,
    QTabBar,
    QTabWidget,
)

import diive
from diive.gui import theme
from diive.gui.registry import MENU_TAB_CLASSES, TAB_CLASSES
from diive.gui.widgets.open_data_dialog import OpenDataDialog

class MainWindow(QMainWindow):
    """Top-level window holding one tab per registered `DiiveTab`."""

    def __init__(self) -> None:
        super().__init__()
        self._base_title = f"diive {diive.__version__}"
        self.setWindowTitle(self._base_title)
        self.resize(1100, 720)
        # The stylesheet is applied app-wide by theme.manager (see run()), so
        # live edits propagate everywhere; no per-window stylesheet here.

        # Retain the DiiveTab instances for the window's lifetime. Qt owns the
        # QWidgets, but the Python tab objects (which hold the signal slots,
        # e.g. PlottingTab._render) would otherwise be garbage-collected after
        # this loop, leaving their signal connections inert.
        self._data = None
        self._created: set = set()  # user-engineered feature column names
        self._menu_tabs: dict = {}  # label -> lazily-created DiiveTab instance

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
        exit_act = QAction("E&xit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        self._build_plot_menu(menubar)

        tools_menu = menubar.addMenu("&Tools")
        for label in MENU_TAB_CLASSES:
            act = QAction(label, self)
            act.triggered.connect(lambda _checked, lab=label: self._open_menu_tab(lab))
            tools_menu.addAction(act)

        help_menu = menubar.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._about)
        help_menu.addAction(about_act)

    def _build_plot_menu(self, menubar) -> None:
        """Build a 'Plot' menu from any tab that declares plot types.

        Plot types are owned by the plotting tab; the menu is just an exclusive,
        checkable selector. The first type is the default — checked and applied
        on startup.
        """
        tab = next((t for t in self._tabs if hasattr(t, "plot_type_labels")), None)
        if tab is None:
            return
        plot_menu = menubar.addMenu("&Plot")
        group = QActionGroup(self)
        group.setExclusive(True)
        for i, label in enumerate(tab.plot_type_labels()):
            act = QAction(label, self, checkable=True)
            act.triggered.connect(lambda _checked, t=tab, lab=label: t.set_plot_type(lab))
            group.addAction(act)
            plot_menu.addAction(act)
            if i == 0:
                act.setChecked(True)
                tab.set_plot_type(label)  # default selected on startup

    def _push_data(self) -> None:
        for tab in self._tabs:
            tab.on_data_loaded(self._data, self._created)

    def _set_data(self, df, source: str) -> None:
        """Set a freshly loaded dataset, reset created features, push to tabs."""
        self.setWindowTitle(f"{self._base_title} — {source}")
        self._data = df
        self._created = set()  # fresh dataset has no user-created features
        self._push_data()

    def _open_menu_tab(self, label: str) -> None:
        """Open (or focus) a menu-activated tab; create it lazily on first use."""
        tab = self._menu_tabs.get(label)
        if tab is None:
            tab = MENU_TAB_CLASSES[label]()
            self._menu_tabs[label] = tab
            self._tabs.append(tab)  # now receives data pushes
            # Build first (widget()/build sets featuresCreated) before connecting.
            self._tabwidget.addTab(tab.widget(), tab.title)
            if hasattr(tab, "featuresCreated"):
                tab.featuresCreated.connect(self._add_features)
            if self._data is not None:
                tab.on_data_loaded(self._data, self._created)  # up to date on open
        self._tabwidget.setCurrentWidget(tab.widget())

    def _on_tab_close(self, index: int) -> None:
        """Close a menu-opened tab (always-on tabs have no close button)."""
        widget = self._tabwidget.widget(index)
        for label, tab in list(self._menu_tabs.items()):
            if tab.widget() is widget:
                self._tabwidget.removeTab(index)
                if tab in self._tabs:
                    self._tabs.remove(tab)
                del self._menu_tabs[label]
                return

    def _add_features(self, new_df) -> None:
        """Merge engineered features into the dataset and re-push to tabs."""
        if new_df is None or new_df.empty or self._data is None:
            return
        for col in new_df.columns:
            self._data[col] = new_df[col]  # aligns on index
        self._created |= {str(c) for c in new_df.columns}
        self._push_data()

    def _load_example(self) -> None:
        df = diive.load_exampledata_parquet()
        self._set_data(df, source="example data (CH-DAV)")

    def _open_file(self) -> None:
        dlg = OpenDataDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.dataframe is not None:
            self._set_data(dlg.dataframe, source=dlg.source_name)

    def _about(self) -> None:
        QMessageBox.about(
            self, "About diive",
            f"diive {diive.__version__}\n\nTime series processing for "
            f"ecosystem and flux data.\nhttps://github.com/holukas/diive",
        )


def run() -> int:
    """Boot the QApplication, show the main window, run the event loop."""
    app = QApplication.instance() or QApplication(sys.argv)
    # Fusion style honours stylesheet item-selection colours consistently
    # (the native Windows style ignores them in combo-box popups).
    app.setStyle("Fusion")
    # Apply the theme app-wide (covers dialogs and combo popups too); editing
    # settings re-applies via theme.manager for a live preview.
    theme.manager.apply()
    window = MainWindow()
    window.show()
    return app.exec()
