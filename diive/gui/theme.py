"""
GUI.THEME: CENTRAL APPEARANCE CONFIG (live-editable)
====================================================

Single source of truth for the diive GUI's look: colour tokens, pill tags, the
time-series palette, and the Qt stylesheet. The `ThemeManager` singleton
(`manager`) holds the current (editable) values and emits `changed` when they
are modified, so the Settings tab can offer a live preview — widgets re-read
colours from `manager` and repaint.

Note: which *variable name* maps to which pill kind (e.g. ``NEE_*`` -> NEE) is
LIBRARY domain knowledge (`dv.variables.classify_variable`); only the pill
**colours and labels** are configured here (GUI presentation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

# Fixed convenience colours (not user-editable).
WHITE = "#FFFFFF"
DARK = "#212121"

# --- editable defaults ------------------------------------------------------
DEFAULT_TOKENS: dict[str, str] = {
    "ACCENT": "#2196F3",              # focus / active-tab underline
    "BORDER": "#B0BEC5",              # control borders
    "LIST_BG": "#ECEFF1",            # variable-list sidebar tint
    "HOVER_BG": "#FFCC80",           # hover highlight (menus, list, combo)
    "HOVER_PRESSED": "#FFB74D",      # pressed/active hover
    "SCROLL_HANDLE": "#B0BEC5",
    "SCROLL_HANDLE_HOVER": "#78909C",
    "PRIMARY_BG": "#1565C0",         # var 1 / primary panel highlight
    "PRIMARY_FG": WHITE,
    "EXTRA_BG": "#BBDEFB",           # additional panels highlight
    "EXTRA_FG": "#0D47A1",
    "TEXT_FG": DARK,
}

#: kind -> [label, background, text]. Lists so values are editable in place.
DEFAULT_PILL_STYLE: dict[str, list[str]] = {
    "NEE": ["NEE", "#388E3C", WHITE],
    "FC": ["FC", "#388E3C", WHITE],
    "GPP": ["GPP", "#1976D2", WHITE],
    "Reco": ["RECO", "#D32F2F", WHITE],
    "LE": ["LE", "#8E24AA", WHITE],
    "ET": ["ET", "#8E24AA", WHITE],
    "Rg": ["Rg", "#FB8C00", DARK],
    "SW_IN": ["SW_IN", "#FB8C00", DARK],
    "PPFD": ["PPFD", "#FB8C00", DARK],
    "PAR": ["PAR", "#FB8C00", DARK],
    "LW": ["LW", "#FB8C00", DARK],
}

DEFAULT_NEW_PILL: list[str] = ["✦ NEW", "#D81B60", WHITE]

DEFAULT_TIMESERIES_COLORS: list[str] = [
    "#1976D2", "#E53935", "#43A047", "#FB8C00", "#8E24AA"]

#: Shared width (px) of the variable list panel, identical across all tabs.
DEFAULT_LIST_WIDTH = 240


def build_qss(t: dict[str, str]) -> str:
    """Build the application stylesheet from a token dict."""
    return f"""
QWidget {{ background: {WHITE}; color: {DARK}; }}
QTabWidget::pane {{ border: 1px solid #E0E0E0; }}
QTabBar::tab {{ background: #F5F5F5; padding: 6px 14px; border: 1px solid #E0E0E0; }}
QTabBar::tab:selected {{ background: {WHITE}; border-bottom: 2px solid {t['ACCENT']}; }}

QMenuBar {{ background: {WHITE}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {t['HOVER_BG']}; }}
QMenuBar::item:pressed {{ background: {t['HOVER_PRESSED']}; }}
QMenu {{ background: {WHITE}; border: 1px solid {t['BORDER']}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t['HOVER_BG']}; }}

QLineEdit {{
    background: {WHITE}; border: 1px solid {t['BORDER']}; border-radius: 6px;
    padding: 5px 8px; margin-bottom: 4px;
}}
QLineEdit:focus {{ border: 1px solid {t['ACCENT']}; }}

QPushButton {{
    background: #F5F5F5; border: 1px solid {t['BORDER']};
    border-radius: 4px; padding: 5px 14px;
}}
QPushButton:hover {{ background: {t['HOVER_BG']}; }}
QPushButton:pressed {{ background: {t['HOVER_PRESSED']}; }}
QPushButton:disabled {{ color: #9E9E9E; background: #FAFAFA; }}

QComboBox {{
    background: {WHITE}; border: 1px solid {t['BORDER']};
    border-radius: 4px; padding: 3px 6px;
}}
QComboBox:hover {{ border: 1px solid {t['HOVER_PRESSED']}; }}
QComboBox QAbstractItemView {{
    background: {WHITE}; border: 1px solid {t['BORDER']}; outline: 0;
    selection-background-color: {t['HOVER_BG']}; selection-color: {DARK};
}}
QComboBox QAbstractItemView::item {{ padding: 4px 8px; }}
QComboBox QAbstractItemView::item:selected {{ background: {t['HOVER_BG']}; color: {DARK}; }}

QListWidget {{
    background: {t['LIST_BG']}; border: 1px solid {t['BORDER']};
    border-radius: 6px; padding: 4px; outline: 0;
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['SCROLL_HANDLE']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:horizontal {{ background: {t['SCROLL_HANDLE']}; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:hover {{ background: {t['SCROLL_HANDLE_HOVER']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; background: none; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
"""


class ThemeManager(QObject):
    """Holds the live, editable theme; emits `changed` on every modification."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.reset(silent=True)

    def reset(self, silent: bool = False) -> None:
        self.tokens = dict(DEFAULT_TOKENS)
        self.pills = {k: list(v) for k, v in DEFAULT_PILL_STYLE.items()}
        self.new_pill = list(DEFAULT_NEW_PILL)
        self.ts_colors = list(DEFAULT_TIMESERIES_COLORS)
        self.list_width = DEFAULT_LIST_WIDTH
        if not silent:
            self.apply()

    def qss(self) -> str:
        return build_qss(self.tokens)

    def apply(self) -> None:
        """Re-apply the stylesheet app-wide and notify listeners (live preview)."""
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(self.qss())
        self.changed.emit()


#: Singleton used across the GUI.
manager = ThemeManager()
