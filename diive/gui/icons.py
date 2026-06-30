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


def _ln_steps() -> QIcon:  # stepwise screening (descending staircase)
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(2, 3), (2, 13), (6, 13), (6, 9.5),
                          (10, 9.5), (10, 6), (14, 6)]))
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


def _ln_project() -> QIcon:  # open project (folder with a star marker)
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(2, 12.5), (2, 4.5), (6, 4.5), (7.5, 6), (14, 6),
                          (14, 12.5), (2, 12.5)]))
    c, r = QPointF(10.5, 9.5), 2.4
    star = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        star.append((c.x() + rr * math.cos(a), c.y() + rr * math.sin(a)))
    star.append(star[0])
    p.drawPolyline(_poly(star))
    p.end()
    return QIcon(pm)


def _ln_book() -> QIcon:  # user manual (open book)
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(8, 4), (4, 3), (2, 3.5), (2, 12.5), (4, 12),
                          (8, 13)]))                       # left page
    p.drawPolyline(_poly([(8, 4), (12, 3), (14, 3.5), (14, 12.5), (12, 12),
                          (8, 13)]))                       # right page
    p.drawLine(8, 4, 8, 13)                                # spine
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


def _ln_export() -> QIcon:  # export data as (tray + outgoing up-arrow)
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(3, 9), (3, 13), (13, 13), (13, 9)]))  # tray
    p.drawLine(8, 2, 8, 9.5)                                     # shaft
    p.drawPolyline(_poly([(5, 5), (8, 2), (11, 5)]))             # arrowhead
    p.end()
    return QIcon(pm)


def _ln_checklist() -> QIcon:  # select variables (pick items from a list)
    pm, p = _line_canvas()
    for i, y in enumerate((4, 8, 12)):
        p.drawRect(QRectF(2, y - 1.5, 3, 3))            # checkbox
        p.drawLine(7, y, 14, y)                          # item label
        if i < 2:
            p.drawPolyline(_poly([(2.6, y), (3.3, y + 1), (4.6, y - 1.4)]))  # tick
    p.end()
    return QIcon(pm)


def _ln_clock() -> QIcon:  # add timestamp column
    pm, p = _line_canvas()
    p.drawEllipse(QRectF(2.5, 2.5, 11, 11))
    p.drawLine(8, 8, 8, 4.6)     # hour hand
    p.drawLine(8, 8, 11, 9)      # minute hand
    p.end()
    return QIcon(pm)


def _ln_funnel() -> QIcon:  # select records by condition (filter rows)
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(2.5, 3), (13.5, 3), (9.5, 8), (9.5, 13.5),
                          (6.5, 11.5), (6.5, 8), (2.5, 3)]))
    p.end()
    return QIcon(pm)


def _ln_tag() -> QIcon:  # rename variables (label)
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(2.5, 8), (6, 4.5), (13, 4.5), (13, 11.5),
                          (6, 11.5), (2.5, 8)]))
    p.drawEllipse(QRectF(4.4, 7.4, 1.2, 1.2))   # hole
    p.end()
    return QIcon(pm)


def _ln_props() -> QIcon:  # metadata explorer (per-variable properties)
    pm, p = _line_canvas()
    p.drawRoundedRect(QRectF(2, 2.5, 12, 11), 1.5, 1.5)
    for y in (5.5, 8, 10.5):
        p.drawEllipse(QRectF(4, y - 0.6, 1.2, 1.2))   # bullet
        p.drawLine(6.5, y, 11.5, y)                    # value
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


def _ln_hexbin() -> QIcon:  # hexbin density plot
    pm, p = _line_canvas()
    cx, cy = 8.0, 8.0
    for r in (5.6, 2.8):
        pts = []
        for k in range(6):
            a = math.pi / 6 + k * math.pi / 3   # flat-top hexagon
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        pts.append(pts[0])
        p.drawPolyline(_poly(pts))
    p.end()
    return QIcon(pm)


def _ln_distribution() -> QIcon:  # shifted distribution (bell curve)
    pm, p = _line_canvas()
    p.drawLine(2, 13, 14, 13)                       # baseline
    pts = [(x, 13 - 9.0 * math.exp(-((x - 8) ** 2) / 11.0))
           for x in [2 + 0.5 * i for i in range(25)]]
    p.drawPolyline(_poly(pts))
    p.end()
    return QIcon(pm)


def _ln_windrose() -> QIcon:  # wind rose / compass star
    pm, p = _line_canvas()
    cx, cy = 8.0, 8.0
    for k in range(8):
        a = -math.pi / 2 + k * math.pi / 4
        r = 6.0 if k % 2 == 0 else 3.0              # alternating petal length
        p.drawLine(cx, cy, cx + r * math.cos(a), cy + r * math.sin(a))
    p.end()
    return QIcon(pm)


def _ln_treering() -> QIcon:  # tree rings (concentric circles)
    pm, p = _line_canvas()
    for r in (2.0, 4.2, 6.4):
        p.drawEllipse(QRectF(8 - r, 8 - r, 2 * r, 2 * r))
    p.end()
    return QIcon(pm)


def _ln_waterfall() -> QIcon:  # waterfall chart (stepped floating bars)
    pm, p = _line_canvas()
    bars = ((2.5, 9, 13), (5.6, 6, 9), (8.7, 6, 11), (11.8, 3, 11))
    for x, top, bottom in bars:
        p.drawRect(QRectF(x, top, 1.7, bottom - top))
    p.setPen(_line_pen(0.9))
    p.drawLine(4.2, 9, 5.6, 9)        # connectors bridging the steps
    p.drawLine(7.3, 6, 8.7, 6)
    p.drawLine(10.4, 11, 11.8, 11)
    p.end()
    return QIcon(pm)


def _ln_surface3d() -> QIcon:  # 3D surface mesh (isometric patch)
    pm, p = _line_canvas()
    # parallelogram corners (isometric tilt)
    tl, tr, br, bl = (3, 6), (10, 3), (13, 10), (6, 13)
    p.drawPolyline(_poly([tl, tr, br, bl, tl]))     # outline
    for t in (1 / 3, 2 / 3):                         # mesh lines both directions
        ax = (tl[0] + (bl[0] - tl[0]) * t, tl[1] + (bl[1] - tl[1]) * t)
        bx = (tr[0] + (br[0] - tr[0]) * t, tr[1] + (br[1] - tr[1]) * t)
        p.drawLine(ax[0], ax[1], bx[0], bx[1])
        ay = (tl[0] + (tr[0] - tl[0]) * t, tl[1] + (tr[1] - tl[1]) * t)
        by = (bl[0] + (br[0] - bl[0]) * t, bl[1] + (br[1] - bl[1]) * t)
        p.drawLine(ay[0], ay[1], by[0], by[1])
    p.end()
    return QIcon(pm)


def _ln_ustar() -> QIcon:  # u* threshold: saturating flux + breakpoint line
    pm, p = _line_canvas()
    p.drawLine(2, 2, 2, 14)                                    # y-axis
    p.drawLine(2, 14, 14, 14)                                  # x-axis
    p.drawPolyline(_poly([(3, 12.5), (5.5, 9), (8, 5.5),
                          (11, 5), (14, 5)]))                  # rise then plateau
    p.drawLine(8, 3, 8, 14)                                    # threshold at the knee
    p.end()
    return QIcon(pm)


def _ln_lag() -> QIcon:  # time lag: a signal and its right-shifted copy
    pm, p = _line_canvas()
    top = [(x, 5.2 - 2.0 * math.sin(2 * math.pi * (x - 2) / 8)) for x in range(2, 12)]
    bot = [(x + 3, 11.0 - 2.0 * math.sin(2 * math.pi * (x - 2) / 8)) for x in range(2, 12)]
    p.drawPolyline(_poly(top))
    p.drawPolyline(_poly(bot))
    p.drawLine(5, 8, 8, 8)                                     # shift indicator
    p.drawPolyline(_poly([(6, 7), (5, 8), (6, 9)]))           # left arrowhead
    p.drawPolyline(_poly([(7, 7), (8, 8), (7, 9)]))           # right arrowhead
    p.end()
    return QIcon(pm)


def _ln_partition() -> QIcon:  # NEE -> GPP + RECO split (Y-fork)
    pm, p = _line_canvas()
    p.drawLine(2, 8, 7.5, 8)                                   # incoming net flux
    p.drawLine(7.5, 8, 14, 4)                                  # upper branch
    p.drawLine(7.5, 8, 14, 12)                                 # lower branch
    p.drawEllipse(QRectF(6.8, 7.3, 1.4, 1.4))                  # fork node
    p.end()
    return QIcon(pm)


def _ln_uncertainty() -> QIcon:  # central estimate + error envelope band
    pm, p = _line_canvas()
    p.drawLine(2, 8, 14, 8)                                    # central estimate
    p.drawPolyline(_poly([(2, 5.5), (5, 4.2), (8, 4), (11, 4.2), (14, 5.5)]))   # upper envelope
    p.drawPolyline(_poly([(2, 10.5), (5, 11.8), (8, 12), (11, 11.8), (14, 10.5)]))  # lower envelope
    p.drawLine(5, 4.7, 5, 11.3)                                # whisker
    p.drawLine(11, 4.7, 11, 11.3)                              # whisker
    p.end()
    return QIcon(pm)


def _ln_outlier() -> QIcon:  # scatter with one flagged outlier
    pm, p = _line_canvas()
    for cx, cy in ((4, 11.5), (6.5, 10.5), (9, 11), (11.5, 10)):   # baseline cluster
        p.drawEllipse(QRectF(cx - 1.3, cy - 1.3, 2.6, 2.6))
    ox, oy = 8.5, 4.5                                              # flagged outlier
    p.drawEllipse(QRectF(ox - 1.3, oy - 1.3, 2.6, 2.6))
    p.drawLine(ox - 3.2, oy - 3.2, ox - 1.6, oy - 1.6)            # flag mark (X corners)
    p.drawLine(ox + 3.2, oy - 3.2, ox + 1.6, oy - 1.6)
    p.drawLine(ox - 3.2, oy + 3.2, ox - 1.6, oy + 1.6)
    p.drawLine(ox + 3.2, oy + 3.2, ox + 1.6, oy + 1.6)
    p.end()
    return QIcon(pm)


def _ln_correction() -> QIcon:  # two slider rows with offset handles
    pm, p = _line_canvas()
    for y, hx in ((5.5, 9.5), (10.5, 5.5)):     # row y, handle center x
        p.drawLine(2.5, y, 13.5, y)             # slider track
        p.drawRect(QRectF(hx - 1.6, y - 2.0, 3.2, 4.0))   # handle
    p.end()
    return QIcon(pm)


def _ln_event() -> QIcon:  # flag-on-pole event marker
    pm, p = _line_canvas()
    p.drawLine(4, 2, 4, 14)                                   # staff
    p.drawPolyline(_poly([(4, 3), (12.5, 5.5), (4, 8)]))      # pennant
    p.end()
    return QIcon(pm)


def _ln_database() -> QIcon:  # stacked-disk database cylinder
    pm, p = _line_canvas()
    p.drawEllipse(QRectF(3, 2.5, 10, 3.5))                    # top disk rim
    p.drawLine(3, 4.25, 3, 11.75)                             # left side
    p.drawLine(13, 4.25, 13, 11.75)                           # right side
    p.drawArc(QRectF(3, 10, 10, 3.5), 180 * 16, 180 * 16)     # curved bottom
    p.drawArc(QRectF(3, 6.25, 10, 3.5), 180 * 16, 180 * 16)   # mid stacked-disk arc
    p.end()
    return QIcon(pm)


def _ln_settings() -> QIcon:  # slider tracks for project settings
    pm, p = _line_canvas()
    for y, hx in ((4, 9.5), (8, 4.5), (12, 10.5)):
        p.drawLine(2.5, y, 13.5, y)                           # track
        p.drawRect(QRectF(hx - 1.3, y - 1.6, 2.6, 3.2))       # handle
    p.end()
    return QIcon(pm)


def _ln_profile() -> QIcon:  # tabular data-profile summary
    pm, p = _line_canvas()
    p.drawRoundedRect(QRectF(2, 2.5, 12, 11), 1.5, 1.5)
    p.drawLine(2, 6, 14, 6)                                   # header rule
    for y in (8.2, 10.2, 12.2):
        p.drawLine(4, y, 7, y)                                # row label
        p.drawLine(8.5, y, 12, y)                             # row value
    p.end()
    return QIcon(pm)


def _ln_extremes() -> QIcon:  # spikes breaching a threshold line
    pm, p = _line_canvas()
    p.drawLine(2, 6, 14, 6)                                   # threshold
    p.drawPolyline(_poly([(2, 11), (4, 11), (5.5, 3), (7, 11),
                          (9.5, 11), (11, 4), (12.5, 11), (14, 11)]))  # spikes
    p.end()
    return QIcon(pm)


def _ln_github() -> QIcon:  # git-branch fork for GitHub
    pm, p = _line_canvas()
    p.drawLine(4.5, 4.5, 4.5, 11.5)                           # trunk
    p.drawPolyline(_poly([(4.5, 5.5), (11.5, 7.5), (11.5, 9.5)]))  # branch up to node
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_line_ink())
    for cx, cy in ((4.5, 4.0), (4.5, 12.0), (11.5, 6.5)):     # nodes
        p.drawEllipse(QPointF(cx, cy), 1.5, 1.5)
    p.end()
    return QIcon(pm)


def _ln_issue() -> QIcon:  # circle-exclamation alert for issue report
    pm, p = _line_canvas()
    p.drawEllipse(QRectF(2.5, 2.5, 11, 11))
    p.drawLine(8, 4.6, 8, 8.6)                                # stroke (above)
    p.drawPoint(QPointF(8, 10.6))                             # dot (below)
    p.end()
    return QIcon(pm)


def _ln_gapfill() -> QIcon:  # series with a dot-filled gap
    pm, p = _line_canvas()
    p.drawPolyline(_poly([(2, 10), (4, 6), (6, 8)]))          # left solid segment
    p.drawPolyline(_poly([(10, 7), (12, 4), (14, 6)]))        # right solid segment
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_line_ink())
    for cx, cy in ((7, 8.2), (8, 7.7), (9, 7.2)):            # gap-fill dots
        p.drawEllipse(QPointF(cx, cy), 0.85, 0.85)
    p.end()
    return QIcon(pm)


def _ln_generic() -> QIcon:
    return _ln_chart()


#: (keyword in lowercased label) -> icon factory. First match wins, in order.
#: Thin-line monochrome glyphs (the only icon set; ink colour from gui/theme.py).
#: ORDER IS LOAD-BEARING: several labels contain a more-general keyword as a
#: substring, so the specific rule must precede the generic one. Notably
#: "partition"/"lag"/"offset" must beat "time" (day/nighttime, "time lag",
#: "nighttime zero offset"), "gap-filling" must beat "gap", "reset" must beat
#: "set to" ("Reset to full range"), "screening" must beat "database" ("Meteo
#: screening (database)"), and "removal" must beat "manual" ("Manual removal").
_LINE_RULES = [
    # Plot methods (specific shapes first).
    ("heatmap", _ln_grid),
    ("hexbin", _ln_hexbin),
    ("ridge", _ln_wave),
    ("diel", _ln_wave),
    ("scatter", _ln_scatter),
    ("histogram", _ln_bars),
    ("shifted distribution", _ln_distribution),
    ("distribution", _ln_distribution),
    ("wind rose", _ln_windrose),
    ("tree ring", _ln_treering),
    ("waterfall", _ln_waterfall),
    ("surface", _ln_surface3d),
    ("cumulative", _ln_chart),
    # Screening & outlier filters.
    ("stepwise", _ln_steps),
    ("screening", _ln_steps),
    ("hampel", _ln_outlier),
    ("removal", _ln_outlier),       # "Manual removal" -- before "manual"
    ("filter", _ln_outlier),
    ("outlier", _ln_outlier),
    # Flux tools (partition/ustar/uncertainty/lag before generic "time").
    ("partition", _ln_partition),
    ("ustar", _ln_ustar),
    ("uncertainty", _ln_uncertainty),
    ("time lag", _ln_lag),
    ("lag", _ln_lag),
    ("chain", _ln_wave),
    ("flux", _ln_wave),
    # Date-range reset (before corrections' "set to": "Reset to full range").
    ("date range", _ln_calendar),
    ("reset", _ln_reset),
    # Corrections (offset before generic "time": "nighttime zero offset").
    ("offset", _ln_correction),
    ("set to", _ln_correction),
    ("set exact", _ln_correction),
    # Gap-filling before "gap" (gaps & coverage).
    ("gap-filling", _ln_gapfill),
    ("gap", _ln_grid),
    ("coverage", _ln_grid),
    # Timestamps / generic time series.
    ("timestamp", _ln_clock),
    ("time", _ln_chart),
    ("series", _ln_chart),
    # Variable management.
    ("select variables", _ln_checklist),
    ("select records", _ln_funnel),
    ("metadata", _ln_props),
    ("feature", _ln_gear),
    ("combine", _ln_grid),
    ("rename", _ln_tag),
    ("profile", _ln_profile),
    # Analysis.
    ("driver", _ln_chart),
    ("season", _ln_chart),
    ("anomal", _ln_chart),
    ("spectro", _ln_grid),
    ("extreme", _ln_extremes),
    ("compound", _ln_extremes),
    # Events.
    ("event", _ln_event),
    # File / settings / help.
    ("open project", _ln_project),
    ("open", _ln_folder),
    ("user manual", _ln_book),
    ("changelog", _ln_document),
    ("example", _ln_document),
    ("export", _ln_export),
    ("save", _ln_save),
    ("exit", _ln_exit),
    ("appearance", _ln_palette),
    ("settings", _ln_settings),
    ("github", _ln_github),
    ("issue", _ln_issue),
    ("report", _ln_issue),
    ("about", _ln_info),
    ("database", _ln_database),
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


def trash_icon(color: str | None = None) -> QIcon:
    """A small thin-line trash can (per-card delete button).

    ``color`` overrides the ink (e.g. a red for a hover/pressed state); defaults
    to the theme ink."""
    pm, p = _canvas()
    ink = QColor(color) if color else _line_ink()
    pen = QPen(ink, 1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(3.0, 4.5), QPointF(13.0, 4.5))        # lid
    p.drawLine(QPointF(6.3, 4.5), QPointF(6.8, 2.8))         # handle left
    p.drawLine(QPointF(6.8, 2.8), QPointF(9.2, 2.8))         # handle top
    p.drawLine(QPointF(9.2, 2.8), QPointF(9.7, 4.5))         # handle right
    p.drawPolyline(_poly([(4.2, 4.5), (5.0, 13.4), (11.0, 13.4), (11.8, 4.5)]))
    for x in (6.6, 8.0, 9.4):                                # tines
        p.drawLine(QPointF(x, 6.3), QPointF(x, 11.6))
    p.end()
    return QIcon(pm)


def dots_icon(color: str | None = None) -> QIcon:
    """Three dots — an overflow/'more actions' (⋯) button glyph that always
    renders (the unicode ⋯ is missing from many fonts)."""
    pm, p = _canvas()
    ink = QColor(color) if color else _line_ink()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(ink)
    for cx in (4.0, 8.0, 12.0):
        p.drawEllipse(QPointF(cx, 8.0), 1.3, 1.3)
    p.end()
    return QIcon(pm)


def locate_icon(color: str | None = None) -> QIcon:
    """A small crosshair/target — 'show this on the plot' (locate) button."""
    pm, p = _canvas()
    ink = QColor(color) if color else _line_ink()
    pen = QPen(ink, 1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(4.5, 4.5, 7.0, 7.0))
    p.drawLine(QPointF(8.0, 1.5), QPointF(8.0, 3.8))
    p.drawLine(QPointF(8.0, 12.2), QPointF(8.0, 14.5))
    p.drawLine(QPointF(1.5, 8.0), QPointF(3.8, 8.0))
    p.drawLine(QPointF(12.2, 8.0), QPointF(14.5, 8.0))
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
