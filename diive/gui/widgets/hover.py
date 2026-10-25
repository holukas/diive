"""
GUI.WIDGETS.HOVER: VALUE-UNDER-CURSOR TOOLTIP FOR EMBEDDED MATPLOTLIB
====================================================================

`HoverAnnotator` attaches to an `MplCanvas` and shows a small floating box with
the value under the cursor as the mouse moves over a plot:

- **Line panels** (time series, cumulative, diel cycle, daily mean): snaps to
  the nearest sample along x (`argmin` over |x - cursor|, so it is correct even
  for non-monotonic lines like the per-month diel cycle) and shows its x (date or
  time-of-day) and value, with a marker. A line drawn thinned (fewer points
  than the record) can carry its full samples as ``_diive_hover_xy = (x, y)``;
  the hover then snaps to those, so it still reports every record.
- **Heatmaps** (`pcolormesh` or `imshow`): reads the cell under the cursor from
  the grid and shows its two axis values and the cell value.

It renders by **blitting** (cache the background once per draw, then redraw just
the annotation), so it never triggers a full render per mouse move. A move
restores and repaints only the areas of the previous and the new tooltip, not
the whole canvas, and a move that lands on the same sample or cell as the last
one changes nothing. This is pure presentation glue (no data or domain logic), so
it lives in the GUI.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import math

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
from matplotlib.image import AxesImage
from matplotlib.transforms import Bbox

#: Tooltip / marker ink (blue-grey 800), matching the GUI's neutral accents.
_INK = "#37474F"

#: Max pixel distance from the cursor to a scatter point for it to register a
#: hover (beyond this the cursor is "in empty space" and other artists, e.g. a
#: binned trend line, may answer instead).
_SCATTER_PICK_RADIUS = 24.0

#: Pixels added around the tooltip and marker extents before restoring and
#: repainting them (antialiasing, the marker edge, and the integer rounding
#: in `FigureCanvasQTAgg.blit`).
_BLIT_PAD = 4.0


class HoverAnnotator:
    """Floating value tooltip for the line and heatmap plots on an `MplCanvas`."""

    def __init__(self, mpl_canvas) -> None:
        self.fig = mpl_canvas.fig
        self._canvas = mpl_canvas._canvas  # FigureCanvasQTAgg
        self._bg = None                    # cached background for blitting
        self._bg_size = None               # figure size the background was taken at
        self._annotations: dict = {}       # ax -> annotation artist
        self._markers: dict = {}           # ax -> marker Line2D
        self._mesh_cache: dict = {}        # id(QuadMesh/AxesImage) -> (xb, yb, values)
        self._scatter_cache: dict = {}     # id(PathCollection) -> _ScatterPixels
        self._line_cache: dict = {}        # id(Line2D) -> (line, src, x, y, ascending)
        self._drawn = None                 # display Bbox of the drawn tooltip + marker
        self._last_hit = None              # (ax, x, y, text, show_marker) last drawn
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
        self._scatter_cache.clear()
        self._line_cache.clear()
        self._bg = self._canvas.copy_from_bbox(self.fig.bbox)
        self._bg_size = tuple(self.fig.bbox.size)
        self._drawn = None
        self._last_hit = None
        self._visible = False

    def _bg_matches(self) -> bool:
        """True when the cached background has the figure's current size.

        While a resize settles the canvas defers its render (see
        `MplCanvas._on_resize`), so the figure is already larger or smaller
        than the last draw. Blitting then would ask for a renderer at the new
        size, which replaces the rendered buffer with a blank one."""
        return self._bg is not None and self._bg_size == tuple(self.fig.bbox.size)

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
        if ax is None or not self._bg_matches() or event.xdata is None:
            self._hide()
            return
        hit = self._value_at(ax, event)
        if hit is None:
            self._hide()
            return
        x, y, text, show_marker = hit
        key = (ax, x, y, text, show_marker)
        if self._visible and key == self._last_hit:
            return  # same sample or cell as the tooltip already shown
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

        old = self._drawn if self._visible else None
        self._restore(old)
        ax.draw_artist(marker)
        ax.draw_artist(ann)
        new = self._extent(ann, marker if show_marker else None)
        if old is None:
            self._blit(new)
        else:
            # One repaint while the boxes are close (the usual small mouse
            # step), two when a jump would make their union much larger.
            union = Bbox.union([old, new])
            if _area(union) <= _area(old) + _area(new):
                self._blit(union)
            else:
                self._blit(old)
                self._blit(new)
        self._drawn = new
        self._last_hit = key
        self._visible = True

    def _hide(self) -> None:
        for ann in self._annotations.values():
            ann.set_visible(False)
        for marker in self._markers.values():
            marker.set_visible(False)
        if self._visible and self._bg_matches():
            self._restore(self._drawn)
            self._blit(self._drawn)
        self._drawn = None
        self._last_hit = None
        self._visible = False

    # --- partial restore / repaint ---
    def _extent(self, ann, marker):
        """Display-space box covering the drawn tooltip and marker, padded."""
        renderer = self._canvas.get_renderer()
        # The rounded box reaches its pad (a fraction of the font size in
        # pixels) beyond the text. Padding the text extent by that is much
        # cheaper than the extent of the box's curved path.
        patch = ann.get_bbox_patch()
        box_pad = patch.get_mutation_scale() * getattr(patch.get_boxstyle(), "pad", 0.5)
        boxes = [ann.get_window_extent(renderer).padded(box_pad)]
        if marker is not None:
            boxes.append(marker.get_window_extent(renderer))
        return Bbox.union(boxes).padded(_BLIT_PAD)

    def _restore(self, bbox) -> None:
        """Put the cached background back inside `bbox` (display coordinates).

        Restoring only the tooltip's area instead of the whole figure keeps a
        move cheap on large canvases. The region's own extents are pixel
        coordinates with the origin at the top left, so the display box (origin
        at the bottom left) is flipped before it is passed on."""
        if bbox is None:
            return
        rx0, ry0, rx1, ry1 = self._bg.get_extents()  # whole figure: (0, 0, W, H)
        x0 = max(rx0, math.floor(bbox.x0))
        x1 = min(rx1, math.ceil(bbox.x1))
        top = max(ry0, math.floor(ry1 - bbox.y1))
        bottom = min(ry1, math.ceil(ry1 - bbox.y0))
        if x1 <= x0 or bottom <= top:
            return
        self._canvas.restore_region(self._bg, bbox=(x0, top, x1, bottom),
                                    xy=(rx0, ry0))

    def _blit(self, bbox) -> None:
        """Repaint only `bbox` (display coordinates) of the canvas widget."""
        if bbox is None:
            return
        clipped = Bbox.intersection(bbox, self.fig.bbox)
        if clipped is not None and clipped.width > 0 and clipped.height > 0:
            self._canvas.blit(clipped)

    # --- value lookup ---
    def _value_at(self, ax, event):
        """Return ``(x, y, text, show_marker)`` for the artist under the cursor.

        Tries a heatmap (`QuadMesh` or `AxesImage`) first, then line artists;
        returns ``None`` when there is nothing to report (e.g. cursor in a gap).
        """
        for coll in ax.collections:
            if isinstance(coll, QuadMesh):
                return self._heatmap_value(ax, coll, event)
        for image in ax.get_images():
            # A 2-D array is a colour-mapped grid; RGB(A) pictures carry no value.
            if (isinstance(image, AxesImage) and image.get_visible()
                    and np.ndim(image.get_array()) == 2):
                return self._heatmap_value(ax, image, event)
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
                 if ln.get_visible() and np.size(ln.get_xdata(orig=False)) > 2]
        if lines:
            return self._line_value(ax, lines, event)
        return None

    def _scatter_value(self, ax, coll, event):
        """Nearest scatter point to the cursor as ``(x, y, text, show_marker)``.

        Shows the point's x and y, plus its colour value (z) when the scatter is
        colour-coded. Returns ``None`` when no point is within the pick radius.
        """
        offsets, px, py, order = self._scatter_pixels(ax, coll)
        if px.size == 0:
            return None
        # Only points within the pick radius in x can be within it in 2-D, and
        # px is sorted, so two binary searches bound the candidates.
        lo = int(np.searchsorted(px, event.x - _SCATTER_PICK_RADIUS, side="left"))
        hi = int(np.searchsorted(px, event.x + _SCATTER_PICK_RADIUS, side="right"))
        if hi <= lo:
            return None
        dx = px[lo:hi] - event.x
        dy = py[lo:hi] - event.y
        d2 = dx * dx + dy * dy
        d2[~np.isfinite(d2)] = np.inf
        k = int(np.argmin(d2))
        if d2[k] > _SCATTER_PICK_RADIUS ** 2:
            return None
        idx = int(order[lo + k])
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

    def _scatter_pixels(self, ax, coll):
        """A scatter's offsets and their pixel positions, sorted by pixel x.

        Returns ``(offsets, px, py, order)``: ``offsets`` in data coordinates,
        ``px``/``py`` the pixel positions in ascending ``px``, and ``order``
        mapping each sorted position back to its row in ``offsets``.
        Transforming every point is the expensive part of a lookup (175,000
        points on the Driver explorer), and the pixel positions only change
        when the figure is redrawn, so the result is kept until the next draw.
        """
        cached = self._scatter_cache.get(id(coll))
        if cached is not None and cached[0] is coll:
            return cached[1]
        offsets = np.asarray(coll.get_offsets(), dtype=float)  # (N, 2) data coords
        if offsets.size == 0:
            empty = np.empty(0)
            result = (offsets, empty, empty, np.empty(0, dtype=int))
        else:
            pix = ax.transData.transform(offsets)  # -> pixels, matching event.x/y
            order = np.argsort(pix[:, 0], kind="stable")  # NaN x sorts last
            result = (offsets, pix[order, 0], pix[order, 1], order)
        # Keep the collection itself so a recycled id() can't serve stale data.
        self._scatter_cache[id(coll)] = (coll, result)
        return result

    def _line_arrays(self, line):
        """A line's x and y as floats, and whether x is finite and ascending.

        Kept until the next draw (or until the line gets new data), so the
        conversion and the ascending check run once, not per mouse move."""
        full = getattr(line, "_diive_hover_xy", None)
        if full is not None:
            src = full  # a thinned line's full record, already as floats
            x, y = full
        else:
            # orig=False returns the unit-converted floats matplotlib actually
            # plots (e.g. date ordinals), matching event.xdata and transData.
            x = line.get_xdata(orig=False)
            y = line.get_ydata(orig=False)
            src = (x, y)
        cached = self._line_cache.get(id(line))
        if (cached is not None and cached[0] is line
                and cached[1][0] is src[0] and cached[1][1] is src[1]):
            return cached[2:]
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        with np.errstate(invalid="ignore"):
            ascending = bool(x.size and np.isfinite(x).all()
                             and (x[1:] >= x[:-1]).all())
        # Keep the line and its source arrays so a recycled id() or new data
        # can't serve stale values.
        self._line_cache[id(line)] = (line, src, x, y, ascending)
        return x, y, ascending

    def _line_value(self, ax, lines, event):
        best = None  # (pixel_dist_sq, x, y)
        for line in lines:
            x, y, ascending = self._line_arrays(line)
            if x.size == 0:
                continue
            if ascending:
                idx = _nearest_ascending(x, event.xdata)
            else:
                # Nearest sample in x. argmin handles non-monotonic lines
                # (e.g. the per-month diel cycle) that searchsorted, which
                # assumes ascending x, would snap to the wrong point.
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

    def _heatmap_value(self, ax, grid_artist, event):
        if event.ydata is None:
            return None
        if isinstance(grid_artist, AxesImage):
            xb, yb, values = self._image_grid(grid_artist)
        else:
            xb, yb, values = self._mesh_grid(grid_artist)
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

    def _image_grid(self, image: AxesImage):
        """Cell-boundary arrays + 2-D value grid for an image (cached per draw).

        An image's cells are evenly spaced across its extent. With
        ``origin='upper'`` row 0 sits at the top, so the row boundaries run
        downwards (`_cell_index` handles descending boundaries)."""
        cached = self._mesh_cache.get(id(image))
        if cached is not None:
            return cached
        values = np.ma.asarray(image.get_array()).filled(np.nan).astype(float)
        rows, cols = values.shape[:2]
        left, right, bottom, top = image.get_extent()
        xb = np.linspace(left, right, cols + 1)
        if image.origin == "upper":
            yb = np.linspace(top, bottom, rows + 1)
        else:
            yb = np.linspace(bottom, top, rows + 1)
        result = (xb, yb, values)
        self._mesh_cache[id(image)] = result
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


def _area(bbox) -> float:
    return bbox.width * bbox.height


def _nearest_ascending(x: np.ndarray, value: float) -> int:
    """Index of the sample nearest to `value` in finite, ascending `x`.

    Gives the same index as ``nanargmin(abs(x - value))``, the first one on
    ties, with two binary searches instead of a pass over the whole array."""
    n = x.size
    pos = int(np.searchsorted(x, value, side="left"))
    if pos == 0:
        k = 0
    elif pos == n:
        k = n - 1
    else:
        # Here x[pos - 1] < value <= x[pos]. The earlier sample wins a tie.
        k = pos - 1 if value - x[pos - 1] <= x[pos] - value else pos
    # With repeated x values the first of them is the nearest by index.
    return int(np.searchsorted(x, x[k], side="left"))


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
