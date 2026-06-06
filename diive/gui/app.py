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

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

import diive
from diive.gui.registry import TAB_CLASSES

#: Light theme: white app/plot background, with the variable list given a
#: tinted blue-grey panel + border so the sidebar stands out against the white
#: matplotlib canvases.
_THEME_QSS = """
QWidget { background: #FFFFFF; color: #212121; }
QTabWidget::pane { border: 1px solid #E0E0E0; }
QTabBar::tab { background: #F5F5F5; padding: 6px 14px; border: 1px solid #E0E0E0; }
QTabBar::tab:selected { background: #FFFFFF; border-bottom: 2px solid #2196F3; }
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
        self.setWindowTitle(f"diive {diive.__version__}")
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


def run() -> int:
    """Boot the QApplication, show the main window, run the event loop."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
