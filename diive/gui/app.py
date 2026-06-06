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

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

import diive
from diive.gui.registry import TAB_CLASSES
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
        self._tabs = []
        tabs = QTabWidget()
        for tab_cls in TAB_CLASSES:
            tab = tab_cls()
            self._tabs.append(tab)
            tabs.addTab(tab.widget(), tab.title)
        self.setCentralWidget(tabs)

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

        help_menu = menubar.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._about)
        help_menu.addAction(about_act)

    def _set_data(self, df, source: str) -> None:
        """Push a freshly loaded dataset to every tab and update the title."""
        self.setWindowTitle(f"{self._base_title} — {source}")
        for tab in self._tabs:
            tab.on_data_loaded(df)

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
