"""
GUI.THEME: CENTRAL APPEARANCE CONFIG (live-editable)
====================================================

Single source of truth for the diive GUI's look: colour tokens, pill tags, the
time-series palette, and the Qt stylesheet. The `ThemeManager` singleton
(`manager`) holds the current (editable) values and emits `changed` when they
are modified, so the Settings tab can offer a live preview — widgets re-read
colours from `manager` and repaint.

The GUI has one look — **Studio**: a clean, minimal VIBECAD-style design with
near-white surfaces, soft borderless panels, monochrome line icons, uppercase
tracked nav/section labels, and a frameless window with a custom header. The
look is described by:

- ``tokens``     — colour tokens, fed to the stylesheet (live-applied).
- ``typography`` — ``{uppercase, tracking}`` for nav/section labels (live).
- ``icon_style`` — icon style flag read by ``gui.icons.menu_icon``.

Colours/typography apply instantly through the ``changed`` signal. The frameless
window shell is built once by ``MainWindow``.

Note: which *variable name* maps to which pill kind (e.g. ``NEE_*`` -> NEE) is
LIBRARY domain knowledge (`dv.variables.classify_variable`); only the pill
**colours and labels** are configured here (GUI presentation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

# Fixed convenience colours (not user-editable; kept for back-compat imports).
WHITE = "#FFFFFF"
DARK = "#212121"

# --- Studio (VIBECAD-like) token defaults -----------------------------------
# Monochrome neutrals + one restrained slate-ink accent. These are the look's
# starting colours; the Settings tab edits them live on top.
STUDIO_TOKENS: dict[str, str] = {
    "ACCENT": "#3A4D5C",
    "BORDER": "#E6E6E3",             # ~= canvas -> hairlines read as soft separators
    "INPUT_BG": "#F4F4F1",
    "LIST_BG": "#F7F7F4",
    "HOVER_BG": "#ECECE8",
    "HOVER_PRESSED": "#E0E0DB",
    "SCROLL_HANDLE": "#D8D8D3",
    "SCROLL_HANDLE_HOVER": "#BDBDB7",
    "PRIMARY_BG": "#3A4D5C",         # primary selection = the accent
    "PRIMARY_FG": WHITE,
    "EXTRA_BG": "#DCE3E8",           # muted accent tint for additional panels
    "EXTRA_FG": "#2A3942",
    "TEXT_FG": "#1E2226",            # near-black ink
    "CANVAS": WHITE,                 # white content background (incl. the area
                                     # around the variable list / splitter gaps)
    "INK": "#1E2226",
    "RADIUS": "12",                  # softer, larger corners
}

# Nav/section label typography. Tracking is QFont AbsoluteSpacing in px.
STUDIO_TYPOGRAPHY: dict = {"uppercase": True, "tracking": 1.5}

#: Icon style flag read by ``gui.icons.menu_icon`` (thin-line monochrome glyphs).
ICON_STYLE = "line"

#: kind -> [label, background, text]. Lists so values are editable in place.
#: Shared across presets (the colours encode domain meaning: NEE green, Reco red).
DEFAULT_PILL_STYLE: dict[str, list[str]] = {
    "NEE": ["NEE", "#388E3C", WHITE],
    "FC": ["FC", "#388E3C", WHITE],
    "GPP": ["GPP", "#1976D2", WHITE],
    "Reco": ["RECO", "#D32F2F", WHITE],
    "FCH4": ["FCH4", "#00897B", WHITE],   # teal — methane flux
    "FN2O": ["FN2O", "#AD1457", WHITE],   # pink — nitrous oxide flux
    "FH2O": ["FH2O", "#3949AB", WHITE],   # indigo — water vapour flux
    "LE": ["LE", "#8E24AA", WHITE],
    "ET": ["ET", "#8E24AA", WHITE],
    "Rg": ["Rg", "#FB8C00", DARK],
    "SW_IN": ["SW_IN", "#FB8C00", DARK],
    "PPFD": ["PPFD", "#FB8C00", DARK],
    "PAR": ["PAR", "#FB8C00", DARK],
    "LW": ["LW", "#FB8C00", DARK],
    "TA": ["TA", "#EF6C00", WHITE],     # deep orange — air temperature
    "VPD": ["VPD", "#0097A7", WHITE],   # cyan — vapour pressure deficit
    "SWC": ["SWC", "#6D4C41", WHITE],   # brown — soil water content
}

DEFAULT_NEW_PILL: list[str] = ["✦ NEW", "#D81B60", WHITE]

DEFAULT_TIMESERIES_COLORS: list[str] = [
    "#1976D2", "#E53935", "#43A047", "#FB8C00", "#8E24AA"]

#: Shared width (px) of the variable list panel, identical across all tabs.
DEFAULT_LIST_WIDTH = 240


def build_qss(t: dict[str, str]) -> str:
    """Build the application stylesheet from a token dict.

    Fully token-driven: the canvas/ink colours and the corner radius come from
    the tokens. Studio's hairline ``BORDER`` (~= canvas) makes the ``1px solid``
    rules read as soft separators. The trailing object-name selectors style the
    frameless Studio shell (header bar, pill tabs).
    """
    r = t["RADIUS"]
    return f"""
QWidget {{ background: {t['CANVAS']}; color: {t['INK']}; }}
QTabWidget::pane {{ border: 1px solid {t['BORDER']}; }}
QTabBar::tab {{ background: {t['LIST_BG']}; padding: 6px 14px; border: 1px solid {t['BORDER']}; }}
QTabBar::tab:selected {{ background: {t['CANVAS']}; border-bottom: 2px solid {t['ACCENT']}; }}

QMenuBar {{ background: {t['CANVAS']}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {t['HOVER_BG']}; }}
QMenuBar::item:pressed {{ background: {t['HOVER_PRESSED']}; }}
QMenu {{ background: {t['CANVAS']}; border: 1px solid {t['BORDER']}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t['HOVER_BG']}; }}

/* Editable inputs share a tinted background so it's obvious what can be edited. */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {t['INPUT_BG']}; border: 1px solid {t['BORDER']}; border-radius: {r}px;
    padding: 5px 8px;
}}
QLineEdit {{ margin-bottom: 4px; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {t['ACCENT']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {t['LIST_BG']}; color: #9E9E9E;
}}

QPushButton {{
    background: {t['LIST_BG']}; border: 1px solid {t['BORDER']};
    border-radius: {r}px; padding: 5px 14px;
}}
QPushButton:hover {{ background: {t['HOVER_BG']}; }}
QPushButton:pressed {{ background: {t['HOVER_PRESSED']}; }}
QPushButton:disabled {{ color: #9E9E9E; background: {t['CANVAS']}; }}

QComboBox {{
    background: {t['INPUT_BG']}; border: 1px solid {t['BORDER']};
    border-radius: {r}px; padding: 3px 6px;
}}
QComboBox:hover {{ border: 1px solid {t['HOVER_PRESSED']}; }}
QComboBox:focus {{ border: 1px solid {t['ACCENT']}; }}
QComboBox QAbstractItemView {{
    background: {t['CANVAS']}; border: 1px solid {t['BORDER']}; outline: 0;
    selection-background-color: {t['HOVER_BG']}; selection-color: {t['INK']};
}}
QComboBox QAbstractItemView::item {{ padding: 4px 8px; }}
QComboBox QAbstractItemView::item:selected {{ background: {t['HOVER_BG']}; color: {t['INK']}; }}

QListWidget {{
    background: {t['LIST_BG']}; border: 1px solid {t['BORDER']};
    border-radius: {r}px; padding: 4px; outline: 0;
}}

/* The variable pane's own surface (the area around the search field and list)
   is white; the search field and list keep their own tinted backgrounds. */
#varpanel {{ background: {WHITE}; }}

QGroupBox {{
    border: 1px solid {t['BORDER']}; border-radius: {r}px;
    margin-top: 10px; padding: 8px 6px 6px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 8px; padding: 0 4px; color: {t['TEXT_FG']}; font-weight: 600;
}}

QCheckBox {{ spacing: 7px; padding: 2px 0; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid {t['SCROLL_HANDLE_HOVER']}; background: {t['CANVAS']};
}}
QCheckBox::indicator:hover {{ border: 1px solid {t['ACCENT']}; }}
QCheckBox::indicator:checked {{
    background: {t['ACCENT']}; border: 1px solid {t['ACCENT']};
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['SCROLL_HANDLE']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:horizontal {{ background: {t['SCROLL_HANDLE']}; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:hover {{ background: {t['SCROLL_HANDLE_HOVER']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; background: none; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* Tab close button: flat with a soft rounded hover so the "×" stays visible. */
QToolButton#tabclose {{ border: none; border-radius: 5px; padding: 0; }}
QToolButton#tabclose:hover {{ background: {t['HOVER_PRESSED']}; }}

/* Tooltips: a clean rounded card with a soft accent-tinted border. */
QToolTip {{
    background: {WHITE}; color: {t['INK']};
    border: 1px solid {t['ACCENT']}; border-radius: 8px;
    padding: 6px 10px; font-size: 12px;
}}

/* Studio chrome: the frameless shell's header/root object-names. */
#studioroot {{ background: transparent; }}
StudioHeaderBar {{ background: transparent; }}

/* Inline header menu buttons: flat text + chevron, soft rounded hover. */
QToolButton#headermenu {{
    background: transparent; border: none; padding: 5px 10px; border-radius: 8px;
    color: {t['INK']};
}}
QToolButton#headermenu:hover {{ background: {t['HOVER_BG']}; }}
QToolButton#headermenu:pressed {{ background: {t['HOVER_PRESSED']}; }}
QToolButton#headermenu::menu-indicator {{ image: none; width: 0; }}

/* Dropdown popup: an elegant rounded white card (translucent window behind it). */
QMenu#studiomenu {{
    background: {WHITE}; border: 1px solid {t['BORDER']}; border-radius: 10px; padding: 6px;
}}
QMenu#studiomenu::item {{ padding: 6px 22px 6px 12px; border-radius: 6px; }}
QMenu#studiomenu::item:selected {{ background: {t['HOVER_BG']}; color: {t['INK']}; }}
/* A custom ::item rule disables Qt's native disabled greying, so restore it —
   otherwise a disabled entry (e.g. "Reset to full range") looks active but
   takes no hover highlight. */
QMenu#studiomenu::item:disabled {{ color: #9E9E9E; }}
QMenu#studiomenu::item:disabled:selected {{ background: transparent; }}
QMenu#studiomenu::separator {{ height: 1px; background: {t['BORDER']}; margin: 5px 8px; }}

/* Firefox-style pill tabs: rounded, spaced, filled active pill, no base line or
   per-tab borders. The #studiotabs id scopes this to the tab widget. */
/* Tabs sit on a soft grey strip; the content pane below them is white. */
#studiotabs {{ background: #EFEFEC; }}
#studiotabs::pane {{ border: none; top: -1px; background: {WHITE}; }}
#studiotabs QTabBar {{ qproperty-drawBase: 0; background: transparent; }}
/* Generous vertical padding makes a taller pill row (the only lever a
   QTabWidget honours); no min-width so each pill hugs its content with equal
   left/right padding, centring the icon+label horizontally. The style centres
   the content vertically within the pill. */
#studiotabs QTabBar::tab {{
    background: transparent; color: {t['INK']};
    border: none; border-radius: {r}px;
    /* font-size grows the row height (QTabWidget honours it, keeping every tab
       the same height so they stay vertically centred); vertical margin lifts
       the pill off the pane so all four corners round (a floating pill, not an
       attached tab); no min-width so the icon+label centre with equal padding. */
    font-size: 15px; padding: 10px 20px; margin: 6px 4px;
}}
#studiotabs QTabBar::tab:hover {{ background: {t['HOVER_BG']}; }}
#studiotabs QTabBar::tab:selected {{
    background: {t['EXTRA_BG']}; color: {t['EXTRA_FG']}; border: none;
}}
#studiotabs QTabBar::close-button {{ border-radius: 4px; }}
#studiotabs QTabBar::close-button:hover {{ background: {t['HOVER_PRESSED']}; }}
"""


class ThemeManager(QObject):
    """Holds the live, editable Studio theme; emits `changed` on modification.

    Token colours, typography, and the time-series palette update live (the
    Settings tab edits them); `icon_style` is fixed to the Studio line glyphs.
    """

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.reset(silent=True)

    def reset(self, silent: bool = False) -> None:
        """Restore the Studio defaults (tokens, typography, pills, palette)."""
        self.pills = {k: list(v) for k, v in DEFAULT_PILL_STYLE.items()}
        self.new_pill = list(DEFAULT_NEW_PILL)
        self.ts_colors = list(DEFAULT_TIMESERIES_COLORS)
        self.list_width = DEFAULT_LIST_WIDTH
        self.tokens = dict(STUDIO_TOKENS)
        self.typography = dict(STUDIO_TYPOGRAPHY)
        self.icon_style = ICON_STYLE
        if not silent:
            self.apply()

    def tracked_font(self, base: QFont | None = None, *, point_delta: float = 0.0) -> QFont:
        """A QFont with the Studio letter spacing applied."""
        font = QFont(base) if base is not None else QFont()
        if point_delta:
            font.setPointSizeF(max(1.0, font.pointSizeF() + point_delta))
        tracking = self.typography.get("tracking", 0.0)
        if tracking:
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, tracking)
        return font

    def label_text(self, text: str) -> str:
        """Uppercase a nav/section label (Studio uppercases them)."""
        return text.upper() if self.typography.get("uppercase") else text

    def qss(self) -> str:
        return build_qss(self.tokens)

    def apply(self) -> None:
        """Re-apply the stylesheet app-wide and notify listeners (live preview)."""
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(self.qss())
        self.changed.emit()

    def as_dict(self) -> dict:
        """Serialise the current theme (for persisting preferences)."""
        return {
            "tokens": dict(self.tokens),
            "pills": {k: list(v) for k, v in self.pills.items()},
            "new_pill": list(self.new_pill),
            "ts_colors": list(self.ts_colors),
            "list_width": self.list_width,
        }

    #: Tokens that define the look's structure (not user-editable in the UI);
    #: they always follow the Studio defaults, so default changes to them take
    #: effect on the next launch even if an old config persisted other values.
    STRUCTURAL_TOKENS = ("CANVAS", "INK", "RADIUS")

    def load_dict(self, data: dict) -> None:
        """Load a serialised theme (tolerant of missing/old keys); no emit.

        Layers saved token overrides on top of the Studio defaults — except the
        STRUCTURAL_TOKENS, which are re-pinned from the defaults so a stale
        persisted value (incl. one from the removed Classic look) can't shadow
        them.
        """
        if not data:
            return
        self.tokens.update(data.get("tokens", {}))
        for key in self.STRUCTURAL_TOKENS:
            self.tokens[key] = STUDIO_TOKENS[key]
        for kind, value in data.get("pills", {}).items():
            if kind in self.pills:
                self.pills[kind] = list(value)
        if "new_pill" in data:
            self.new_pill = list(data["new_pill"])
        if "ts_colors" in data:
            self.ts_colors = list(data["ts_colors"])
        if "list_width" in data:
            self.list_width = int(data["list_width"])


#: Singleton used across the GUI.
manager = ThemeManager()
