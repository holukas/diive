"""
GUI.WIDGETS.HOVER: VALUE-UNDER-CURSOR TOOLTIP FOR EMBEDDED MATPLOTLIB
====================================================================

`HoverAnnotator` attaches to an `MplCanvas` and shows a small floating box with
the value under the cursor as the mouse moves over a plot:

- **Line panels** (time series, cumulative, diel cycle, daily mean): snaps to
  the nearest sample along x (`argmin` over |x - cursor|, so it is correct even
  for non-monotonic lines like the per-month diel cycle) and shows its x (date or
  time-of-day) and value, with a marker.
- **Heatmaps** (`pcolormesh`): reads the cell under the cursor from the grid and
  shows its two axis values and the cell value.

It renders by **blitting** (cache the background once per draw, then redraw just
the annotation), so it never triggers a full repaint per mouse move. This is
pure presentation glue — no data/domain logic — so it lives in the GUI.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.dates import (
    AutoDateFormatter,
    ConciseDateConverter,
    ConciseDateFormatter,
    DateConverter,
    DateFormatter,
    DateLocator,
    num2date,
)

#: Tooltip / marker ink (blue-grey 800), matching the GUI's neutral accents.
_INK = "#37474F"

#: Max pixel distance from the cursor to a scatter point for it to register a
#: hover (beyond this the cursor is "in empty space" and other artists, e.g. a
#: binned trend line, may answer instead).
_SCATTER_PICK_RADIUS = 24.0


class HoverAnnotator:
    """Floating value tooltip for the line and heatmap plots on an `MplCanvas`."""

    def __init__(self, mpl_canvas) -> None:
        self.fig = mpl_canvas.fig
        self._canvas = mpl_canvas._canvas  # FigureCanvasQTAgg
        self._bg = None                    # cached background for blitting
        self._annotations: dict = {}       # ax -> annotation artist
        self._markers: dict = {}           # ax -> marker Line2D
        self._mesh_cache: dict = {}        # id(QuadMesh) -> (xb, yb, values)
        self._visible = False
        self._enabled = True

        self._canvas.mpl_connect("motion_notify_event", self._on_move)
        self._canvas.mpl_connect("draw_event", self._on_draw)
        self._canvas.mpl_connect("figure_leave_event", lambda _e: self._hide())
        self._canvas.mpl_connect("axes_leave_event", lambda _e: self._hide())

    # --- background / artist lifecycle ---
    def _on_draw(self, _event) -> None:
        # A full draw means the figure changed (new render, pan/zoom, resize):
        # recapture the clean background and drop artists bound to old axes.
        self._annotations.clear()
        self._markers.clear()
        self._mesh_cache.clear()
        self._bg = self._canvas.copy_from_bbox(self.fig.bbox)
        self._visible = False

    def _annotation_for(self, ax):
        ann = self._annotations.get(ax)
        if ann is None:
            ann = ax.annotate(
                "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4", fc=_INK, ec="none", alpha=0.92),
                color="white", fontsize=8, zorder=10000, annotation_clip=False)
            ann.set_animated(True)
            self._annotations[ax] = ann
        return ann

    def _marker_for(self, ax):
        marker = self._markers.get(ax)
        if marker is None:
            # scalex/scaley=False: creating the marker must NOT request an
            # autoscale of this axes. On a sharex-linked panel that still has
            # autoscale on (the toolbar only disables it on the directly-zoomed
            # axis), that autoscale would refit x to the full data range and
            # revert the user's zoom across all linked panels on the next draw.
            (marker,) = ax.plot(
                [], [], marker="o", ms=6, mfc="none", mec=_INK, mew=1.5,
                ls="none", zorder=9999, scalex=False, scaley=False)
            marker.set_animated(True)
            self._markers[ax] = marker
        return marker

    def set_enabled(self, enabled: bool) -> None:
        """Turn the hover tooltip on or off (hides any current box when off)."""
        self._enabled = enabled
        if not enabled:
            self._hide()

    # --- mouse handling ---
    def _on_move(self, event) -> None:
        if not self._enabled:
            return
        ax = event.inaxes
        if ax is None or self._bg is None or event.xdata is None:
            self._hide()
            return
        hit = self._value_at(ax, event)
        if hit is None:
            self._hide()
            return
        x, y, text, show_marker = hit
        ann = self._annotation_for(ax)
        ann.xy = (x, y)
        ann.set_text(text)
        ann.set_visible(True)
        marker = self._marker_for(ax)
        if show_marker:
            marker.set_data([x], [y])
            marker.set_visible(True)
        else:
            marker.set_visible(False)

        self._canvas.restore_region(self._bg)
        ax.draw_artist(marker)
        ax.draw_artist(ann)
        self._canvas.blit(self.fig.bbox)
        self._visible = True

    def _hide(self) -> None:
        for ann in self._annotations.values():
            ann.set_visible(False)
        for marker in self._markers.values():
            marker.set_visible(False)
        if self._visible and self._bg is not None:
            self._canvas.restore_region(self._bg)
            self._canvas.blit(self.fig.bbox)
        self._visible = False

    # --- value lookup ---
    def _value_at(self, ax, event):
        """Return ``(x, y, text, show_marker)`` for the artist under the cursor.

        Tries a heatmap (`QuadMesh`) first, then line artists; returns ``None``
        when there is nothing to report (e.g. cursor in a gap).
        """
        for coll in ax.collections:
            if isinstance(coll, QuadMesh):
                return self._heatmap_value(ax, coll, event)
        # Scatter points (ax.scatter -> PathCollection) take priority over a
        # binned trend line, but only when the cursor is near a point; otherwise
        # fall through so the line still answers.
        scatter = next((c for c in ax.collections
                        if isinstance(c, PathCollection) and c.get_visible()), None)
        if scatter is not None:
            hit = self._scatter_value(ax, scatter, event)
            if hit is not None:
                return hit
        lines = [ln for ln in ax.get_lines()
                 if ln.get_visible() and np.asarray(ln.get_xdata()).size > 2]
        if lines:
            return self._line_value(ax, lines, event)
        return None

    def _scatter_value(self, ax, coll, event):
        """Nearest scatter point to the cursor as ``(x, y, text, show_marker)``.

        Shows the point's x and y, plus its colour value (z) when the scatter is
        colour-coded. Returns ``None`` when no point is within the pick radius.
        """
        offsets = np.asarray(coll.get_offsets(), dtype=float)  # (N, 2) data coords
        if offsets.size == 0:
            return None
        pix = ax.transData.transform(offsets)  # -> pixels, matching event.x/y
        dx = pix[:, 0] - event.x
        dy = pix[:, 1] - event.y
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))
        if d2[idx] > _SCATTER_PICK_RADIUS ** 2:
            return None
        xd, yd = offsets[idx]
        text = (f"x: {self._fmt_axis(ax.xaxis, xd)}\n"
                f"y: {self._fmt_axis(ax.yaxis, yd)}")
        # A colour-coded scatter carries its z values in the mappable array.
        arr = coll.get_array()
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            if arr.size == offsets.shape[0] and np.isfinite(arr[idx]):
                text += f"\nz: {arr[idx]:.4g}"
        return xd, yd, text, True

    def _line_value(self, ax, lines, event):
        best = None  # (pixel_dist_sq, x, y)
        for line in lines:
            # orig=False returns the unit-converted floats matplotlib actually
            # plots (e.g. date ordinals), matching event.xdata and transData.
            x = np.asarray(line.get_xdata(orig=False), dtype=float)
            y = np.asarray(line.get_ydata(orig=False), dtype=float)
            if x.size == 0:
                continue
            # Nearest sample in x. argmin handles non-monotonic lines (e.g. the
            # per-month diel cycle) that searchsorted, which assumes ascending x,
            # would snap to the wrong point.
            with np.errstate(invalid="ignore"):
                dx = np.abs(x - event.xdata)
            if not np.any(np.isfinite(dx)):
                continue
            idx = int(np.nanargmin(dx))
            for i in (idx - 1, idx, idx + 1):
                if not (0 <= i < x.size) or not np.isfinite(y[i]):
                    continue
                px, py = ax.transData.transform((x[i], y[i]))
                dist = (px - event.x) ** 2 + (py - event.y) ** 2
                if best is None or dist < best[0]:
                    best = (dist, x[i], y[i])
        if best is None:
            return None
        _, bx, by = best
        return bx, by, f"{self._fmt_axis(ax.xaxis, bx)}\n{by:.4g}", True

    def _heatmap_value(self, ax, mesh, event):
        if event.ydata is None:
            return None
        xb, yb, values = self._mesh_grid(mesh)
        col = _cell_index(xb, event.xdata)
        row = _cell_index(yb, event.ydata)
        if col is None or row is None:
            return None
        val = values[row, col]
        xc = 0.5 * (xb[col] + xb[col + 1])
        yc = 0.5 * (yb[row] + yb[row + 1])
        vtxt = "no data" if val is None or not np.isfinite(val) else f"{val:.4g}"
        text = f"{self._fmt_axis(ax.xaxis, xc)} · {self._fmt_axis(ax.yaxis, yc)}\n{vtxt}"
        return xc, yc, text, False

    def _mesh_grid(self, mesh: QuadMesh):
        """Cell-boundary arrays + 2-D value grid for a QuadMesh (cached per draw)."""
        cached = self._mesh_cache.get(id(mesh))
        if cached is not None:
            return cached
        coords = np.asarray(mesh.get_coordinates())  # (R+1, C+1, 2)
        rows = coords.shape[0] - 1
        cols = coords.shape[1] - 1
        xb = np.asarray(coords[0, :, 0], dtype=float)
        yb = np.asarray(coords[:, 0, 1], dtype=float)
        values = np.ma.asarray(mesh.get_array())
        values = values.reshape(rows, cols).filled(np.nan).astype(float)
        result = (xb, yb, values)
        self._mesh_cache[id(mesh)] = result
        return result

    def _fmt_axis(self, axis, value: float) -> str:
        """Format a value on `axis` as a date, a time-of-day, or a plain number.

        The axis kind is read from matplotlib's own unit converter (and, for the
        diel-cycle/time-of-day axis, its 0-24 span) rather than guessed from the
        value's magnitude — so a radiation value of 1200 is not mistaken for a
        date and a temperature of 7.5 is not mistaken for 07:30.
        """
        if not np.isfinite(value):
            return "—"
        # A panel can tag an axis as integer-valued (e.g. the year/month
        # heatmap's Month/Year axes), labelling the value and skipping the
        # date / time-of-day heuristics below.
        intlabel = getattr(axis, "_diive_hover_intlabel", None)
        if intlabel is not None:
            return f"{intlabel} {int(np.floor(value))}"
        if self._is_date_axis(axis):
            # A panel can opt into a date-only label (e.g. daily aggregates,
            # where the time is always 00:00) by tagging its axes.
            date_only = getattr(getattr(axis, "axes", None), "_diive_hover_dateonly", False)
            fmt = "%Y-%m-%d" if date_only else "%Y-%m-%d %H:%M"
            return num2date(value).strftime(fmt)
        if self._is_hours_axis(axis) and 0 <= value <= 24:
            hours = int(value)
            minutes = int(round((value - hours) * 60))
            if minutes == 60:
                hours, minutes = hours + 1, 0
            return f"{hours:02d}:{minutes:02d}"
        return f"{value:.4g}"

    @staticmethod
    def _is_date_axis(axis) -> bool:
        """True when matplotlib treats this axis as dates (its floats are date
        ordinals) — detected from the unit converter, or from the major
        formatter/locator, not from the value's magnitude.

        The formatter/locator fallback matters because several plot paths render
        date ticks WITHOUT registering a matplotlib ``DateConverter``: a
        ``LineCollection`` of pre-converted ordinals, a bare ``xaxis.axis_date()``,
        or pandas' own date converter (which isn't a ``DateConverter`` subclass).
        Without it, those axes' values were shown as raw ordinals (e.g. 20590)."""
        try:
            conv = axis.get_converter()  # matplotlib >= 3.10
        except AttributeError:
            conv = getattr(axis, "converter", None)
        if isinstance(conv, (DateConverter, ConciseDateConverter)):
            return True
        if isinstance(axis.get_major_formatter(),
                      (DateFormatter, AutoDateFormatter, ConciseDateFormatter)):
            return True
        return isinstance(axis.get_major_locator(), DateLocator)

    @staticmethod
    def _is_hours_axis(axis) -> bool:
        """True for the diel-cycle / heatmap time-of-day axis: a non-date axis
        spanning roughly 0-24, so a value in [0, 24] reads as a clock time."""
        lo, hi = sorted(axis.get_view_interval())
        return lo >= -1.0 and hi <= 25.0 and (hi - lo) >= 6.0


def _cell_index(bounds: np.ndarray, value: float):
    """Index of the cell whose half-open boundary interval contains ``value``.

    Handles ascending or descending boundary arrays; returns ``None`` if the
    value is outside the grid.
    """
    if value is None or bounds.size < 2:
        return None
    if bounds[0] <= bounds[-1]:
        idx = int(np.searchsorted(bounds, value)) - 1
    else:  # descending boundaries
        idx = int(np.searchsorted(-bounds, -value)) - 1
    if 0 <= idx < bounds.size - 1:
        return idx
    return None
