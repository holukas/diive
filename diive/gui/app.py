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
from diive.gui.registry import MENU_TAB_CLASSES, TAB_CLASSES
from diive.gui.widgets.open_data_dialog import OpenDataDialog

#: Light theme: white app/plot background, with the variable list given a
#: tinted blue-grey panel + border so the sidebar stands out against the white
#: matplotlib canvases.
_THEME_QSS = """
QWidget { background: #FFFFFF; color: #212121; }
QTabWidget::pane { border: 1px solid #E0E0E0; }
QTabBar::tab { background: #F5F5F5; padding: 6px 14px; border: 1px solid #E0E0E0; }
QTabBar::tab:selected { background: #FFFFFF; border-bottom: 2px solid #2196F3; }

QMenuBar { background: #FFFFFF; }
QMenuBar::item { padding: 4px 10px; background: transparent; border-radius: 4px; }
QMenuBar::item:selected { background: #FFCC80; }
QMenuBar::item:pressed { background: #FFB74D; }
QMenu { background: #FFFFFF; border: 1px solid #B0BEC5; padding: 4px; }
QMenu::item { padding: 5px 22px; border-radius: 4px; }
QMenu::item:selected { background: #FFCC80; }
QLineEdit {
    background: #FFFFFF;
    border: 1px solid #B0BEC5;
    border-radius: 6px;
    padding: 5px 8px;
    margin-bottom: 4px;
}
QLineEdit:focus { border: 1px solid #2196F3; }

QPushButton {
    background: #F5F5F5;
    border: 1px solid #B0BEC5;
    border-radius: 4px;
    padding: 5px 14px;
}
QPushButton:hover { background: #FFCC80; }
QPushButton:pressed { background: #FFB74D; }
QPushButton:disabled { color: #9E9E9E; background: #FAFAFA; }

QComboBox {
    background: #FFFFFF;
    border: 1px solid #B0BEC5;
    border-radius: 4px;
    padding: 3px 6px;
}
QComboBox:hover { border: 1px solid #FFB74D; }
QComboBox QAbstractItemView {
    background: #FFFFFF;
    border: 1px solid #B0BEC5;
    outline: 0;
    selection-background-color: #FFCC80;
    selection-color: #212121;
}
QComboBox QAbstractItemView::item { padding: 4px 8px; }
QComboBox QAbstractItemView::item:selected { background: #FFCC80; color: #212121; }
QListWidget {
    background: #ECEFF1;
    border: 1px solid #B0BEC5;
    border-radius: 6px;
    padding: 4px;
    outline: 0;
}

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #B0BEC5; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:horizontal { background: #B0BEC5; border-radius: 5px; min-width: 30px; }
QScrollBar::handle:hover { background: #78909C; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; background: none; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
"""


class MainWindow(QMainWindow):
    """Top-level window holding one tab per registered `DiiveTab`."""

    def __init__(self) -> None:
        super().__init__()
        self._base_title = f"diive {diive.__version__}"
        self.setWindowTitle(self._base_title)
        self.resize(1100, 720)
        self.setStyleSheet(_THEME_QSS)

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
    # Apply the theme app-wide so dialogs and combo-box popups (separate
    # top-level windows) are styled too, not just the main window.
    app.setStyleSheet(_THEME_QSS)
    window = MainWindow()
    window.show()
    return app.exec()
