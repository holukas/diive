"""
GUI.ICONS: SMALL DRAWN MENU ICONS
=================================

Tiny `QIcon`s drawn with `QPainter` (no image assets) for the menu bar, so every
menu entry gets a small cohesive glyph. `menu_icon(label)` matches a glyph to a
menu label by keyword (Plot methods get shape hints — colour grid, line, ridges;
file/data/tools/help entries get folder/disk/calendar/gear/… glyphs).

GUI-only presentation. Adding a menu entry only needs a matching keyword here
(unknown labels fall back to a generic chart glyph).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

_SIZE = 16
_INK = QColor("#455A64")        # blue-grey 700 — lines/outlines
_BLUE = QColor("#2196F3")       # accent
_AMBER = QColor("#FB8C00")
_GREEN = QColor("#43A047")
_RED = QColor("#E53935")
_WHITE = QColor("#FFFFFF")
# A small RdYlBu_r-ish ramp for the heatmap cells.
_RAMP = [QColor("#4575B4"), QColor("#91BFDB"), QColor("#E0F3F8"),
         QColor("#FEE090"), QColor("#FC8D59"), QColor("#D73027")]


def _canvas() -> tuple[QPixmap, QPainter]:
    pm = QPixmap(_SIZE, _SIZE)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    return pm, p


def _poly(points) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in points])


# --- plot-method glyphs ----------------------------------------------------
def _heatmap_icon() -> QIcon:
    pm, p = _canvas()
    p.setPen(Qt.PenStyle.NoPen)
    n = 4
    cell = (_SIZE - 2) / n
    for r in range(n):
        for c in range(n):
            p.setBrush(_RAMP[(r + c) % len(_RAMP)])
            p.drawRect(QRectF(1 + c * cell, 1 + r * cell, cell - 0.6, cell - 0.6))
    p.end()
    return QIcon(pm)


def _timeseries_icon() -> QIcon:
    pm, p = _canvas()
    p.setPen(QPen(_BLUE, 1.6))
    p.drawPolyline(_poly([(1, 11), (4, 6), (7, 9), (10, 3), (13, 7), (15, 4)]))
    p.end()
    return QIcon(pm)


def _ridgeline_icon() -> QIcon:
    pm, p = _canvas()
    for y, col in ((4, "#91BFDB"), (8, "#FEE090"), (12, "#FC8D59")):
        pts = [(0, y + 3)]
        for x in range(0, _SIZE + 1, 2):
            pts.append((x, y + 3 - 3.2 * math.exp(-((x - 8) ** 2) / 18.0)))
        pts.append((_SIZE, y + 3))
        poly = _poly(pts)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(col)))
        p.drawPolygon(poly)
        p.setPen(QPen(_INK, 0.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolyline(poly)
    p.end()
    return QIcon(pm)


def _generic_icon() -> QIcon:
    pm, p = _canvas()
    p.setPen(QPen(_INK, 1.4))
    p.drawLine(2, 14, 2, 2)
    p.drawLine(2, 14, 14, 14)
    p.setPen(QPen(_BLUE, 1.4))
    p.drawPolyline(_poly([(3, 11), (7, 7), (10, 9), (14, 4)]))
    p.end()
    return QIcon(pm)


# --- file / data / tools / help glyphs -------------------------------------
def _folder_icon() -> QIcon:  # Open data file
    pm, p = _canvas()
    p.setPen(QPen(_INK, 1))
    p.setBrush(QColor("#FFCC80"))
    p.drawPolygon(_poly([(1.5, 4), (6, 4), (7.5, 5.5), (1.5, 5.5)]))  # tab
    p.drawRoundedRect(QRectF(1.5, 5, 13, 8.5), 1.5, 1.5)              # body
    p.end()
    return QIcon(pm)


def _document_icon() -> QIcon:  # Load example data
    pm, p = _canvas()
    p.setPen(QPen(_INK, 1))
    p.setBrush(_WHITE)
    p.drawPolygon(_poly([(3, 1.5), (10, 1.5), (13, 4.5), (13, 14.5), (3, 14.5)]))
    p.setPen(QPen(QColor("#90A4AE"), 1))
    for y in (7, 9.5, 12):
        p.drawLine(5, y, 11, y)
    # sparkle (example highlight)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_AMBER)
    p.drawPolygon(_poly([(10.5, 3), (11.3, 5), (13.3, 5.8), (11.3, 6.6), (10.5, 8.6),
                         (9.7, 6.6), (7.7, 5.8), (9.7, 5)]))
    p.end()
    return QIcon(pm)


def _save_icon() -> QIcon:  # Save data as parquet
    pm, p = _canvas()
    p.setPen(QPen(_INK, 1))
    p.setBrush(QColor("#42A5F5"))
    p.drawRoundedRect(QRectF(2, 2, 12, 12), 1.5, 1.5)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_WHITE)
    p.drawRect(QRectF(5, 2.5, 6, 3.2))    # slider
    p.drawRect(QRectF(4.5, 8.5, 7, 5))    # label
    p.setBrush(QColor("#42A5F5"))
    p.drawRect(QRectF(9, 3, 1.4, 2.2))    # slider notch
    p.end()
    return QIcon(pm)


def _exit_icon() -> QIcon:  # Exit
    pm, p = _canvas()
    p.setPen(QPen(_INK, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(_poly([(8, 2.5), (3, 2.5), (3, 13.5), (8, 13.5)]))  # door frame
    p.setPen(QPen(_RED, 1.6))
    p.drawLine(7, 8, 14, 8)                                            # arrow shaft
    p.drawPolyline(_poly([(11, 5), (14, 8), (11, 11)]))               # arrowhead
    p.end()
    return QIcon(pm)


def _calendar_icon() -> QIcon:  # Select date range
    pm, p = _canvas()
    p.setPen(QPen(_INK, 1.1))
    p.drawLine(5, 1.5, 5, 4)
    p.drawLine(11, 1.5, 11, 4)
    p.setBrush(_WHITE)
    p.drawRoundedRect(QRectF(2, 3, 12, 11), 1.5, 1.5)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_BLUE)
    p.drawRect(QRectF(2.6, 3.6, 10.8, 2.8))     # header
    # a highlighted day range
    p.setBrush(QColor("#BBDEFB"))
    p.drawRect(QRectF(4, 8.5, 8, 2.5))
    p.setBrush(QColor("#90A4AE"))
    for cx in (5, 8, 11):
        p.drawEllipse(QRectF(cx - 0.6, 9.1, 1.2, 1.2))
    p.end()
    return QIcon(pm)


def _reset_icon() -> QIcon:  # Reset to full range
    pm, p = _canvas()
    p.setPen(QPen(_INK, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(3, 3, 10, 10), 70 * 16, 250 * 16)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_INK)
    p.drawPolygon(_poly([(11.2, 1.5), (12.2, 5), (8.6, 4.6)]))  # arrowhead
    p.end()
    return QIcon(pm)


def _gear_icon() -> QIcon:  # Feature engineering
    pm, p = _canvas()
    c, r_out, r_in = 8.0, 7.0, 4.6
    p.setPen(QPen(_INK, 1))
    p.setBrush(QColor("#90A4AE"))
    for k in range(8):
        a = k * math.pi / 4
        p.drawPolygon(_poly([
            (c + r_in * math.cos(a - 0.18), c + r_in * math.sin(a - 0.18)),
            (c + r_out * math.cos(a - 0.12), c + r_out * math.sin(a - 0.12)),
            (c + r_out * math.cos(a + 0.12), c + r_out * math.sin(a + 0.12)),
            (c + r_in * math.cos(a + 0.18), c + r_in * math.sin(a + 0.18))]))
    p.drawEllipse(QRectF(c - r_in, c - r_in, 2 * r_in, 2 * r_in))
    p.setBrush(_WHITE)
    p.drawEllipse(QRectF(c - 1.7, c - 1.7, 3.4, 3.4))
    p.end()
    return QIcon(pm)


def _palette_icon() -> QIcon:  # Appearance
    pm, p = _canvas()
    p.setPen(QPen(_INK, 1))
    p.setBrush(QColor("#ECEFF1"))
    p.drawEllipse(QRectF(1.5, 2, 13, 12))
    p.setBrush(_WHITE)
    p.drawEllipse(QRectF(8.8, 8.8, 3.6, 3.6))  # thumb hole
    p.setPen(Qt.PenStyle.NoPen)
    for cx, cy, col in ((5, 5, _RED), (8, 4, _AMBER), (10.8, 5.6, _GREEN), (4.6, 8.6, _BLUE)):
        p.setBrush(col)
        p.drawEllipse(QRectF(cx - 1, cy - 1, 2, 2))
    p.end()
    return QIcon(pm)


def _info_icon() -> QIcon:  # About
    pm, p = _canvas()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_BLUE)
    p.drawEllipse(QRectF(2, 2, 12, 12))
    p.setBrush(_WHITE)
    p.drawEllipse(QRectF(7.2, 4.3, 1.6, 1.6))   # dot
    p.drawRect(QRectF(7.2, 7, 1.6, 4.6))         # stem
    p.end()
    return QIcon(pm)


#: (keyword in lowercased label) -> icon factory. First match wins, in order.
_RULES = [
    ("heatmap", _heatmap_icon),
    ("ridge", _ridgeline_icon),
    ("time", _timeseries_icon),
    ("series", _timeseries_icon),
    ("open", _folder_icon),
    ("example", _document_icon),
    ("save", _save_icon),
    ("exit", _exit_icon),
    ("date range", _calendar_icon),
    ("reset", _reset_icon),
    ("feature", _gear_icon),
    ("appearance", _palette_icon),
    ("about", _info_icon),
]


def menu_icon(label: str) -> QIcon:
    """Pick a small drawn icon for a menu label by keyword.

    Strips the ``&`` mnemonic markers first so e.g. ``"E&xit"`` matches "exit".
    """
    key = label.lower().replace("&", "")
    for keyword, factory in _RULES:
        if keyword in key:
            return factory()
    return _generic_icon()


#: Backwards-compatible alias (Plot-menu callers / tests).
plot_menu_icon = menu_icon
