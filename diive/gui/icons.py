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


# --- thin-line monochrome glyphs (Studio preset) ---------------------------
# Single ink colour, one stroke weight, round caps/joins, no fills — the
# minimal VIBECAD look. Icon colour follows the active theme's INK token.
def _line_ink() -> QColor:
    try:
        from diive.gui import theme
        return QColor(theme.manager.tokens.get("INK", "#3A4D5C"))
    except Exception:  # theme not importable in some isolated test contexts
        return QColor("#3A4D5C")


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


def _scatter_icon() -> QIcon:  # Scatter XY
    pm, p = _canvas()
    p.setPen(Qt.PenStyle.NoPen)
    dots = [(3, 12, _BLUE), (6, 8, _GREEN), (8, 11, _AMBER), (10, 5, _RED),
            (13, 7, _BLUE), (5, 4, _GREEN), (12, 12, _AMBER)]
    for x, y, col in dots:
        p.setBrush(col)
        p.drawEllipse(QRectF(x - 1.4, y - 1.4, 2.8, 2.8))
    p.end()
    return QIcon(pm)


def _cumulative_icon() -> QIcon:  # Cumulative year
    pm, p = _canvas()
    # Several rising curves fanning out (one cumulative per year).
    for col, scale in (("#43A047", 1.0), ("#2196F3", 0.72), ("#FB8C00", 0.46)):
        pts = [(x, 14 - scale * 12 * (x / _SIZE) ** 0.85) for x in range(0, _SIZE + 1, 2)]
        p.setPen(QPen(QColor(col), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolyline(_poly(pts))
    p.end()
    return QIcon(pm)


def _dielcycle_icon() -> QIcon:  # Diel cycle
    pm, p = _canvas()
    purple = QColor("#7E57C2")
    curve = [(x, 8 - 4 * math.sin(2 * math.pi * x / _SIZE)) for x in range(_SIZE + 1)]
    band = _poly([(x, y - 2.2) for x, y in curve]
                 + [(x, y + 2.2) for x, y in reversed(curve)])
    fill = QColor(purple)
    fill.setAlpha(60)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(fill)
    p.drawPolygon(band)
    p.setPen(QPen(purple, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(_poly(curve))
    p.end()
    return QIcon(pm)


def _chain_icon() -> QIcon:  # Flux processing chain
    pm, p = _canvas()
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(_BLUE, 1.7))
    p.drawRoundedRect(QRectF(1.5, 5, 7.5, 6), 3, 3)
    p.setPen(QPen(_GREEN, 1.7))
    p.drawRoundedRect(QRectF(7, 5, 7.5, 6), 3, 3)
    p.end()
    return QIcon(pm)


def _gap_icon() -> QIcon:  # Gaps & coverage
    pm, p = _canvas()
    p.setPen(Qt.PenStyle.NoPen)
    # Coverage bar: green segments split by a red gap in the middle.
    p.setBrush(_GREEN)
    p.drawRoundedRect(QRectF(1.5, 6, 5.2, 4), 1, 1)
    p.drawRoundedRect(QRectF(11, 6, 3.5, 4), 1, 1)
    p.setBrush(_RED)
    p.drawRect(QRectF(7.0, 6, 3.6, 4))            # the gap
    # Timeline baseline.
    p.setPen(QPen(_INK, 1))
    p.drawLine(1.5, 13, 14.5, 13)
    p.end()
    return QIcon(pm)


def _driver_icon() -> QIcon:  # Driver explorer
    pm, p = _canvas()
    # Axes, a rising trend line, and a few points around it (a relationship).
    p.setPen(QPen(_INK, 1.2))
    p.drawLine(2, 14, 2, 2)
    p.drawLine(2, 14, 14, 14)
    p.setPen(QPen(_BLUE, 1.6))
    p.drawLine(3, 12, 14, 4)                       # trend line
    p.setPen(Qt.PenStyle.NoPen)
    for x, y, col in ((4, 12, _GREEN), (7, 9, _AMBER), (9, 8, _RED), (12, 5, _GREEN)):
        p.setBrush(col)
        p.drawEllipse(QRectF(x - 1.3, y - 1.3, 2.6, 2.6))
    p.end()
    return QIcon(pm)


def _seasonal_icon() -> QIcon:  # Seasonal-trend & anomalies
    pm, p = _canvas()
    # A seasonal wave riding a rising trend line.
    p.setPen(QPen(_AMBER, 1.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    wave = [(x, 11 - 0.18 * x - 2.6 * math.sin(2 * math.pi * x / 8)) for x in range(1, 16)]
    p.drawPolyline(_poly(wave))
    p.setPen(QPen(_RED, 1.5))
    p.drawLine(1, 11, 15, 4)                        # trend
    p.end()
    return QIcon(pm)


def _spectrogram_icon() -> QIcon:  # Spectrogram
    pm, p = _canvas()
    # Time-frequency cells (columns of a ramp) with a bright horizontal band.
    p.setPen(Qt.PenStyle.NoPen)
    ncol, nrow = 5, 4
    cw = (_SIZE - 2) / ncol
    ch = (_SIZE - 2) / nrow
    for c in range(ncol):
        for r in range(nrow):
            shade = _RAMP[(r + (c % 3)) % len(_RAMP)]
            p.setBrush(shade)
            p.drawRect(QRectF(1 + c * cw, 1 + r * ch, cw - 0.5, ch - 0.5))
    # The "1 cycle/day" band.
    p.setPen(QPen(_WHITE, 1.2, Qt.PenStyle.DashLine))
    p.drawLine(1, int(1 + 2 * ch), 15, int(1 + 2 * ch))
    p.end()
    return QIcon(pm)


def _histogram_icon() -> QIcon:  # Histogram
    pm, p = _canvas()
    p.setPen(Qt.PenStyle.NoPen)
    # A bell-ish run of bars; the tallest (peak) highlighted in amber.
    heights = [3, 6, 10, 13, 9, 5]
    bw = (_SIZE - 2) / len(heights)
    peak = heights.index(max(heights))
    for i, h in enumerate(heights):
        p.setBrush(_AMBER if i == peak else QColor("#78909C"))
        p.drawRect(QRectF(1 + i * bw, 14 - h, bw - 0.8, h))
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
#: The "classic" colourful set and the "line" monochrome set share the same
#: keyword vocabulary, so the active preset's `icon_style` just swaps the table.
_CLASSIC_RULES = [
    ("heatmap", _heatmap_icon),
    ("ridge", _ridgeline_icon),
    ("diel", _dielcycle_icon),
    ("cumulative", _cumulative_icon),
    ("scatter", _scatter_icon),
    ("histogram", _histogram_icon),
    ("time", _timeseries_icon),
    ("series", _timeseries_icon),
    ("open", _folder_icon),
    ("example", _document_icon),
    ("save", _save_icon),
    ("exit", _exit_icon),
    ("date range", _calendar_icon),
    ("reset", _reset_icon),
    ("gap", _gap_icon),
    ("coverage", _gap_icon),
    ("driver", _driver_icon),
    ("season", _seasonal_icon),
    ("anomal", _seasonal_icon),
    ("spectro", _spectrogram_icon),
    ("feature", _gear_icon),
    ("chain", _chain_icon),
    ("flux", _chain_icon),
    ("appearance", _palette_icon),
    ("about", _info_icon),
]

_LINE_RULES = [
    ("heatmap", _ln_grid),
    ("ridge", _ln_wave),
    ("diel", _ln_wave),
    ("cumulative", _ln_chart),
    ("scatter", _ln_scatter),
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
]


def menu_icon(label: str) -> QIcon:
    """Pick a small drawn icon for a menu label by keyword.

    Strips the ``&`` mnemonic markers first so e.g. ``"E&xit"`` matches "exit".
    Honours the active theme preset's icon style ("classic" colourful glyphs or
    "line" monochrome glyphs); unknown labels fall back to a generic chart glyph.
    """
    try:
        from diive.gui import theme
        line = theme.manager.icon_style == "line"
    except Exception:
        line = False
    rules = _LINE_RULES if line else _CLASSIC_RULES
    key = label.lower().replace("&", "")
    for keyword, factory in rules:
        if keyword in key:
            return factory()
    return _ln_generic() if line else _generic_icon()


#: Backwards-compatible alias (Plot-menu callers / tests).
plot_menu_icon = menu_icon
