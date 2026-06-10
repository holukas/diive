"""
GUI.ICONS: SMALL DRAWN MENU ICONS
=================================

Tiny thin-line monochrome `QIcon`s drawn with `QPainter` (no image assets) for
the menu bar, so every menu entry gets a small cohesive glyph. `menu_icon(label)`
matches a glyph to a menu label by keyword (Plot methods get shape hints — grid,
line, ridges; file/data/tools/help entries get folder/disk/calendar/gear/…
glyphs). Glyph ink follows the active theme's INK token.

GUI-only presentation. Adding a menu entry only needs a matching keyword in
`_LINE_RULES` (unknown labels fall back to a generic chart glyph).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

_SIZE = 16


def _canvas() -> tuple[QPixmap, QPainter]:
    pm = QPixmap(_SIZE, _SIZE)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    return pm, p


def _poly(points) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in points])


# --- thin-line monochrome glyphs (Studio preset) ---------------------------
# Single ink colour, one stroke weight, round caps/joins, no fills — the
# minimal VIBECAD look. Icon colour follows the active theme's INK token.
def _line_ink() -> QColor:
    try:
        from diive.gui import theme
        return QColor(theme.manager.tokens.get("INK", "#1E2226"))
    except Exception:  # theme not importable in some isolated test contexts
        return QColor("#1E2226")


def _line_pen(width: float = 1.3) -> QPen:
    pen = QPen(_line_ink(), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _line_canvas() -> tuple[QPixmap, QPainter]:
    pm, p = _canvas()
    p.setPen(_line_pen())
    p.setBrush(Qt.BrushStyle.NoBrush)
    return pm, p


def _ln_chart() -> QIcon:  # series / generic / driver / cumulative / seasonal
    pm, p = _line_canvas()
    p.drawLine(2, 2, 2, 14)
    p.drawLine(2, 14, 14, 14)
    p.drawPolyline(_poly([(3, 11), (6, 6), (9, 9), (14, 3)]))
    p.end()
    return QIcon(pm)


def _ln_grid() -> QIcon:  # heatmap / spectrogram / gaps & coverage
    pm, p = _line_canvas()
    p.drawRoundedRect(QRectF(2, 2, 12, 12), 2, 2)
    for x in (6, 10):
        p.drawLine(x, 2, x, 14)
    for y in (6, 10):
        p.drawLine(2, y, 14, y)
    p.end()
    return QIcon(pm)


def _ln_bars() -> QIcon:  # histogram
    pm, p = _line_canvas()
    p.drawLine(2, 14, 14, 14)
    for x, top in ((3, 9), (6.5, 4), (10, 7)):
        p.drawRect(QRectF(x, top, 2.6, 14 - top))
    p.end()
    return QIcon(pm)


def _ln_wave() -> QIcon:  # ridgeline / diel cycle / flux chain
    pm, p = _line_canvas()
    for dy in (-3.2, 1.2):
        pts = [(x, 9 + dy - 2.6 * math.sin(2 * math.pi * x / 9)) for x in range(1, 16)]
        p.drawPolyline(_poly(pts))
    p.end()
    return QIcon(pm)


def _ln_scatter() -> QIcon:  # scatter XY
    pm, p = _line_canvas()
    for cx, cy in ((4, 11), (7, 7), (9, 10), (12, 4), (5, 5)):
        p.drawEllipse(QRectF(cx - 1.3, cy - 1.3, 2.6, 2.6))
    p.end()
    return QIcon(pm)


def _ln_folder() -> QIcon:  # open data file
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(2, 12.5), (2, 4.5), (6, 4.5), (7.5, 6), (14, 6),
                          (14, 12.5), (2, 12.5)]))
    p.end()
    return QIcon(pm)


def _ln_document() -> QIcon:  # load example data
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(9.5, 2), (4, 2), (4, 14), (12, 14), (12, 4.5), (9.5, 2),
                          (9.5, 4.5), (12, 4.5)]))
    for y in (8, 11):
        p.drawLine(6, y, 10, y)
    p.end()
    return QIcon(pm)


def _ln_save() -> QIcon:  # save data as parquet
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(3, 3), (11, 3), (13, 5), (13, 13), (3, 13), (3, 3)]))
    p.drawRect(QRectF(5, 3, 4, 2.6))      # slider
    p.drawRect(QRectF(5, 8.5, 6, 4.5))    # label
    p.end()
    return QIcon(pm)


def _ln_exit() -> QIcon:  # exit
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(8, 2.5), (3, 2.5), (3, 13.5), (8, 13.5)]))
    p.drawLine(7, 8, 14, 8)
    p.drawPolyline(_poly([(11, 5), (14, 8), (11, 11)]))
    p.end()
    return QIcon(pm)


def _ln_calendar() -> QIcon:  # select date range
    pm, p = _line_canvas()
    p.drawRoundedRect(QRectF(2, 3.5, 12, 10.5), 1.5, 1.5)
    p.drawLine(2, 7, 14, 7)
    p.drawLine(5, 2, 5, 4.5)
    p.drawLine(11, 2, 11, 4.5)
    p.drawLine(5, 10.5, 11, 10.5)   # a highlighted range
    p.end()
    return QIcon(pm)


def _ln_reset() -> QIcon:  # reset to full range
    pm, p = _line_canvas()
    p.drawArc(QRectF(3, 3, 10, 10), 70 * 16, 250 * 16)
    p.drawPolyline(_poly([(8.6, 1.6), (11.6, 2.4), (11.0, 5.4)]))  # arrowhead
    p.end()
    return QIcon(pm)


def _ln_gear() -> QIcon:  # feature engineering
    pm, p = _line_canvas()
    c, r_out, r_in = 8.0, 6.6, 4.4
    for k in range(8):
        a = k * math.pi / 4
        p.drawLine(c + r_in * math.cos(a), c + r_in * math.sin(a),
                   c + r_out * math.cos(a), c + r_out * math.sin(a))
    p.drawEllipse(QRectF(c - r_in, c - r_in, 2 * r_in, 2 * r_in))
    p.drawEllipse(QRectF(c - 1.7, c - 1.7, 3.4, 3.4))
    p.end()
    return QIcon(pm)


def _ln_palette() -> QIcon:  # appearance
    pm, p = _line_canvas()
    p.drawEllipse(QRectF(2, 2.5, 12, 11))
    p.drawEllipse(QRectF(8.8, 8.8, 3.4, 3.4))  # thumb hole
    for cx, cy in ((5, 5), (8, 4), (10.6, 5.6), (4.6, 8.6)):
        p.drawEllipse(QRectF(cx - 0.7, cy - 0.7, 1.4, 1.4))
    p.end()
    return QIcon(pm)


def _ln_info() -> QIcon:  # about
    pm, p = _line_canvas()
    p.drawEllipse(QRectF(2.5, 2.5, 11, 11))
    p.drawPoint(QPointF(8, 5.4))
    p.drawLine(8, 7.4, 8, 11.4)
    p.end()
    return QIcon(pm)


def _ln_generic() -> QIcon:
    return _ln_chart()


#: (keyword in lowercased label) -> icon factory. First match wins, in order.
#: Thin-line monochrome glyphs (the only icon set; ink colour from gui/theme.py).
_LINE_RULES = [
    ("heatmap", _ln_grid),
    ("ridge", _ln_wave),
    ("diel", _ln_wave),
    ("cumulative", _ln_chart),
    ("scatter", _ln_scatter),
    ("hampel", _ln_scatter),
    ("outlier", _ln_scatter),
    ("histogram", _ln_bars),
    ("time", _ln_chart),
    ("series", _ln_chart),
    ("open", _ln_folder),
    ("example", _ln_document),
    ("save", _ln_save),
    ("exit", _ln_exit),
    ("date range", _ln_calendar),
    ("reset", _ln_reset),
    ("gap", _ln_grid),
    ("coverage", _ln_grid),
    ("driver", _ln_chart),
    ("season", _ln_chart),
    ("anomal", _ln_chart),
    ("spectro", _ln_grid),
    ("feature", _ln_gear),
    ("chain", _ln_wave),
    ("flux", _ln_wave),
    ("appearance", _ln_palette),
    ("about", _ln_info),
    ("overview", _ln_grid),
    ("log", _ln_document),
]


def pin_icon() -> QIcon:
    """A small 'pinned' indicator (thumbtack) shown on frozen tabs."""
    pm, p = _canvas()
    ink = _line_ink()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(ink)
    p.drawEllipse(QRectF(4.5, 3.0, 7.0, 7.0))            # head
    pen = QPen(ink, 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(8.0, 10.0), QPointF(8.0, 13.5))   # needle
    p.end()
    return QIcon(pm)


def close_icon() -> QIcon:
    """A small, clearly visible "×" for tab close buttons (ink-coloured)."""
    pm, p = _canvas()
    pen = QPen(_line_ink(), 1.7)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(4.5, 4.5), QPointF(11.5, 11.5))
    p.drawLine(QPointF(11.5, 4.5), QPointF(4.5, 11.5))
    p.end()
    return QIcon(pm)


def menu_icon(label: str) -> QIcon:
    """Pick a small thin-line drawn icon for a menu label by keyword.

    Strips the ``&`` mnemonic markers first so e.g. ``"E&xit"`` matches "exit";
    unknown labels fall back to a generic chart glyph.
    """
    key = label.lower().replace("&", "")
    for keyword, factory in _LINE_RULES:
        if keyword in key:
            return factory()
    return _ln_generic()


#: Backwards-compatible alias (Plot-menu callers / tests).
plot_menu_icon = menu_icon
