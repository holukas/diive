"""
GUI.WIDGETS.VARIABLE_DELEGATE: LIST ITEM RENDERER
=================================================

`VariableDelegate` paints variable-list rows itself so the panel highlight is
reliable: once a `QListWidget` is styled via a stylesheet, Qt ignores per-item
`setBackground`/`setForeground`, so role-based colouring silently fails. The
delegate also draws a colored "pill" tag for recognised variables. The
name->kind classification comes from `dv.variables.classify_variable` (library
domain knowledge); this module only maps a kind to its pill colour/label.

Each item carries its panel position in `PANEL_ROLE` (0 = not shown; 1 = the
primary panel; 2+ = additional panels), which the delegate maps to colours.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

from diive.variables import classify_variable

#: Variable name (shared with VariableList click handling).
NAME_ROLE = Qt.ItemDataRole.UserRole
#: Panel position: 0 = unselected, 1 = primary, 2+ = additional panels.
PANEL_ROLE = Qt.ItemDataRole.UserRole + 1
#: True for features created by the user (feature engineer) -> "NEW" pill.
CREATED_ROLE = Qt.ItemDataRole.UserRole + 2

_PRIMARY_BG = QColor("#1565C0")   # blue 800 -- solid, white text reads clearly
_PRIMARY_FG = QColor("#FFFFFF")
_EXTRA_BG = QColor("#BBDEFB")     # blue 100
_EXTRA_FG = QColor("#0D47A1")     # blue 900
_TEXT_FG = QColor("#212121")
_HOVER_BG = QColor("#FFCC80")     # orange 200 -- matches the menu hover
_WHITE = QColor("#FFFFFF")
_DARK = QColor("#212121")

#: GUI styling per variable kind (from `dv.variables.classify_variable`):
#: kind -> (label, background, text_color). The library owns the name->kind
#: classification; this map owns only the colours/labels.
# Five maximally distinct Material hues (green / blue / red / purple / orange),
# one per semantic group. Grouped variables share a colour by design
# (NEE+FC = carbon, LE+ET = water, all radiation = orange).
_GREEN = QColor("#388E3C")    # green 700
_BLUE = QColor("#1976D2")     # blue 700
_RED = QColor("#D32F2F")      # red 700
_PURPLE = QColor("#8E24AA")   # purple 600
_ORANGE = QColor("#FB8C00")   # orange 600 (dark text for contrast)
_PINK = QColor("#D81B60")     # pink 600 -- user-created features

#: Pill for user-created (feature-engineered) variables.
_NEW_PILL = ("✦ NEW", _PINK, QColor("#FFFFFF"))  # ✦ NEW

_PILL_STYLE = {
    "NEE": ("NEE", _GREEN, _WHITE),
    "FC": ("FC", _GREEN, _WHITE),       # CO2 flux (pre-NEE) -> carbon green
    "GPP": ("GPP", _BLUE, _WHITE),
    "Reco": ("RECO", _RED, _WHITE),
    # LE and ET are the same water flux in different units -> shared purple.
    "LE": ("LE", _PURPLE, _WHITE),
    "ET": ("ET", _PURPLE, _WHITE),
    # Radiation -> shared orange (dark text for contrast).
    "Rg": ("Rg", _ORANGE, _DARK),
    "SW_IN": ("SW_IN", _ORANGE, _DARK),
    "PPFD": ("PPFD", _ORANGE, _DARK),
    "PAR": ("PAR", _ORANGE, _DARK),
    "LW": ("LW", _ORANGE, _DARK),
}


def _pill_for(name):
    """Return ``(label, background, text_color)`` for a variable, or None.

    Classification (name -> kind) is the library's job; this only maps the
    kind to GUI colours.
    """
    vc = classify_variable(name)
    if vc is None:
        return None
    return _PILL_STYLE.get(vc.kind)


class VariableDelegate(QStyledItemDelegate):
    """Render a variable row: highlight + optional NEE pill."""

    def sizeHint(self, option, index) -> QSize:
        # Width 1 so long names never force a horizontal scrollbar (which would
        # push the right-aligned pill out of view). Rows still paint at the
        # full viewport width via option.rect; we elide the name ourselves.
        s = super().sizeHint(option, index)
        return QSize(1, max(s.height(), 26))

    def paint(self, painter, option, index) -> None:
        painter.save()
        painter.setClipRect(option.rect)  # never draw past the row
        painter.setRenderHint(painter.RenderHint.Antialiasing, True)

        rect: QRect = option.rect
        name = index.data(NAME_ROLE) or index.data(Qt.ItemDataRole.DisplayRole) or ""
        order = index.data(PANEL_ROLE) or 0

        # Row background by selection state (or hover).
        bg, fg, bold = None, _TEXT_FG, False
        if order == 1:
            bg, fg, bold = _PRIMARY_BG, _PRIMARY_FG, True
        elif order > 1:
            bg, fg, bold = _EXTRA_BG, _EXTRA_FG, True
        elif option.state & QStyle.StateFlag.State_MouseOver:
            bg = _HOVER_BG
        if bg is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(rect.adjusted(2, 1, -2, -1), 4, 4)

        # User-created features get the "NEW" pill; otherwise classify by name.
        pill = _NEW_PILL if index.data(CREATED_ROLE) else _pill_for(name)

        # Pill tag (right-aligned), drawn first so we know how much width to
        # reserve for the text.
        pill_w = 0
        if pill is not None:
            pill_label, pill_color, pill_fg = pill
            pill_font = QFont(option.font)
            pill_font.setBold(True)
            pill_font.setPointSizeF(max(7.0, option.font.pointSizeF() - 1.0))
            painter.setFont(pill_font)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(pill_label)
            ph = fm.height() + 2
            pw = tw + 14
            px = rect.right() - pw - 8
            py = rect.center().y() - ph // 2
            pill_rect = QRect(px, py, pw, ph)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(pill_color)
            painter.drawRoundedRect(pill_rect, ph / 2.0, ph / 2.0)
            painter.setPen(pill_fg)
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, pill_label)
            pill_w = pw + 12

        # Variable name (numbered with its panel position when shown).
        text_font = QFont(option.font)
        text_font.setBold(bold)
        painter.setFont(text_font)
        painter.setPen(fg)
        label = f"{order}  {name}" if order else str(name)
        text_rect = rect.adjusted(10, 0, -8 - pill_w, 0)
        elided = painter.fontMetrics().elidedText(
            label, Qt.TextElideMode.ElideRight, text_rect.width()
        )
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            elided,
        )

        painter.restore()
