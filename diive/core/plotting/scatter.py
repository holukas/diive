"""
PLOTTING: SCATTER XY
====================

Scatter plot of x vs y with optional z colour-coding and bin aggregation.

Part of the diive library: https://github.com/holukas/diive
"""
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.core.dfun.stats import q25, q75
from diive.core.plotting.styles.format import FormatStyle

#: A single fully transparent face. The Agg, PDF and PS renderers skip a face
#: whose alpha is zero and SVG writes it as ``fill-opacity: 0``, so drawing
#: with it paints exactly what no face paints.
_NO_FACE = np.zeros((1, 4))


class _HollowMarkerCollection(PathCollection):
    """A `PathCollection` of unfilled markers that keeps matplotlib's fast path.

    ``Collection.draw`` renders a collection whose points share one path,
    size, face colour and edge colour through ``renderer.draw_markers``, which
    rasterizes the marker once and stamps it at every point. It only takes
    that path when the collection has exactly one face colour, and an unfilled
    marker (``c='none'``) has none, so a hollow scatter falls back to stroking
    every marker on its own. For 175,000 points that is seconds per draw.

    This class hands ``draw`` one fully transparent face colour for the
    duration of the call. The collection is otherwise an ordinary
    `PathCollection`: ``get_facecolor()`` still reports no face, and the
    legend handler, ``ax.collections`` and anything reading the offsets or
    sizes see what ``ax.scatter`` returned.

    The one visible change is the one ``draw_markers`` always makes: each
    marker is centred on the nearest pixel instead of at its exact sub-pixel
    position, so a marker can sit up to half a pixel from where the slow
    path put it.
    """

    def draw(self, renderer):
        # Settle the colours first: the first update after creation resets
        # the face to 'none' and would undo the swap below.
        self.update_scalarmappable()
        if len(self._facecolors) or not self.get_visible():
            super().draw(renderer)
            return
        facecolors = self._facecolors
        self._facecolors = _NO_FACE
        try:
            super().draw(renderer)
        finally:
            self._facecolors = facecolors


#: The path ``ax.scatter(marker='o')`` gives every point: a circle of radius
#: 0.5 in marker units, before the marker size scales it.
_CIRCLE = MarkerStyle('o').get_path().transformed(MarkerStyle('o').get_transform())

#: Fractional pixel positions are split into this many bins per axis, and each
#: bin gets its own pair of stencils. More bins find more hidden markers but
#: cost more stencil passes; 3 is where that stops paying off.
_SUBPIXEL_BINS = 3


def _cell_stencils(core_radius: float, reach: float, k: int, bins: int) -> dict:
    """Pixel stencils of a circle whose centre lies somewhere in a sub-pixel bin.

    For a centre in pixel (0, 0) at a fractional position inside bin
    ``(bx, by)``, the "core" stencil marks the pixels that lie entirely
    within ``core_radius`` of the centre wherever in the bin it is, and the
    "reach" stencil every pixel that comes within ``reach`` of it anywhere in
    the bin. Both are boolean arrays indexed ``[dy + k, dx + k]``.
    """
    d = np.arange(-k, k + 1, dtype=float)
    edges = np.arange(bins + 1) / bins
    far = []   # farthest distance from the bin to each pixel column/row
    near = []  # nearest distance from the bin to each pixel column/row
    for lo, hi in zip(edges[:-1], edges[1:]):
        far.append(np.max(np.abs([d - lo, d - hi, d + 1 - lo, d + 1 - hi]), axis=0))
        near.append(np.maximum(0.0, np.maximum(d - hi, lo - (d + 1))))
    return {(bx, by): (far[by][:, None] ** 2 + far[bx][None, :] ** 2 <= core_radius ** 2,
                       near[by][:, None] ** 2 + near[bx][None, :] ** 2 <= reach ** 2)
            for bx in range(bins) for by in range(bins)}


def _hidden_markers(xy: np.ndarray, opaque: np.ndarray, radius: float,
                    linewidth: float, clip: tuple) -> np.ndarray:
    """Flag the circle markers that leave no trace in an Agg raster.

    Agg draws a collection point by point, in order, and a pixel that lies
    entirely inside an opaque filled circle is overwritten with that circle's
    colour, whatever was underneath. So a marker changes nothing in the image
    when every pixel it can touch is either outside the clip area or fully
    inside the face of an opaque marker drawn after it. Leaving such markers
    out of the draw gives the same image, pixel for pixel.

    The test is conservative in both directions: a marker's "core" (the pixels
    it is sure to overwrite) is taken smaller than the circle, and the pixels
    it may touch are taken larger than the circle and its edge.

    Args:
        xy: Marker centres in display pixels, in drawing order.
        opaque: True where a marker's face colour has an alpha of exactly 1.
        radius: Circle radius in pixels.
        linewidth: Edge width in pixels.
        clip: ``(x0, y0, x1, y1)``, the display area the renderer paints.

    Returns:
        A boolean array, True for each marker that can be left out.
    """
    n = len(xy)
    # Agg flattens the circle's Bezier curves into chords that can dip up to
    # about 0.4 px inside the curve, and rounds coordinates to 1/256 px. It
    # paints only the pixels the outline touches, which lie within
    # radius + linewidth / 2 apart from the stroke's joins at the chord ends.
    core_radius = radius - 0.6 - 1e-3 * radius
    reach = radius + linewidth / 2 + 0.5 + 0.1 * linewidth
    k = int(np.ceil(reach)) + 1

    x0, y0, x1, y1 = clip
    hidden = np.ones(n, dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return hidden
    cx0, cy0 = int(np.floor(x0)), int(np.floor(y0))
    cx1, cy1 = int(np.ceil(x1)), int(np.ceil(y1))
    # A marker whose pixel lies more than k pixels outside the clip area
    # cannot paint inside it. The grid holds every pixel the others can reach.
    gx0, gy0 = cx0 - 2 * k, cy0 - 2 * k
    width, height = cx1 - cx0 + 4 * k, cy1 - cy0 + 4 * k

    with np.errstate(invalid='ignore'):
        px = np.floor(xy[:, 0])
        py = np.floor(xy[:, 1])
        near_clip = ((px >= cx0 - k) & (px < cx1 + k)
                     & (py >= cy0 - k) & (py < cy1 + k))
    idx = np.flatnonzero(near_clip)
    stencils = _cell_stencils(core_radius, reach, k, _SUBPIXEL_BINS)
    if (not len(idx) or not opaque[idx].any()
            or not any(core.any() for core, _ in stencils.values())
            or len(idx) * (2 * reach) ** 2 < (x1 - x0) * (y1 - y0)):
        # No marker can cover another (none opaque, or too small to have a
        # core), or too few to cover the area once, where the test would cost
        # more than it saves.
        return ~near_clip
    # Marker indices fit 32 bits for any plottable record; half the memory
    # traffic of int64 in the grid passes below.
    order_type = np.int32 if n < np.iinfo(np.int32).max else np.int64
    never = np.iinfo(order_type).max
    px = px[idx].astype(np.intp)
    py = py[idx].astype(np.intp)
    bins = _SUBPIXEL_BINS
    fx = np.minimum(((xy[idx, 0] - px) * bins).astype(np.intp), bins - 1)
    fy = np.minimum(((xy[idx, 1] - py) * bins).astype(np.intp), bins - 1)
    group = fx * bins + fy
    cell = (py - gy0) * width + (px - gx0)

    # For every pixel, the index of the last marker whose core covers it.
    last = np.full(height * width, -1, dtype=order_type)
    occluding = opaque[idx]
    order = idx.astype(order_type)
    for (bx, by), (core, _) in stencils.items():
        sel = occluding & (group == bx * bins + by)
        if not sel.any():
            continue
        dy, dx = np.nonzero(core)
        shift = (dy - k) * width + (dx - k)
        np.maximum.at(last, (cell[sel][:, None] + shift).ravel(),
                      np.repeat(order[sel], len(shift)))

    # A pixel outside the clip area is never painted, so it hides nothing and
    # needs no cover. One the clip edge cuts through is painted only in part,
    # so no marker is sure to overwrite it.
    last = last.reshape(height, width)
    left = gx0 + np.arange(width)
    bottom = gy0 + np.arange(height)
    in_x, out_x = (left >= x0) & (left + 1 <= x1), (left + 1 <= x0) | (left >= x1)
    in_y, out_y = (bottom >= y0) & (bottom + 1 <= y1), (bottom + 1 <= y0) | (bottom >= y1)
    last[~(in_y[:, None] & in_x[None, :])] = -1
    last[out_y[:, None] | out_x[None, :]] = never

    # Running minima over 1, 2, 4, ... pixels along each row, so the minimum
    # over any row segment of the reach stencil is two lookups.
    runs = [last]
    step = 1
    while 2 * step <= 2 * k + 1:
        prev = runs[-1]
        cur = prev.copy()
        np.minimum(prev[:, :-step], prev[:, step:], out=cur[:, :-step])
        runs.append(cur)
        step *= 2
    runs = [r.ravel() for r in runs]

    for (bx, by), (_, reach_cells) in stencils.items():
        sel = np.flatnonzero(group == bx * bins + by)
        if not len(sel):
            continue
        start = cell[sel]
        covered_until = np.full(len(sel), never, dtype=order_type)
        for row in range(2 * k + 1):
            cols = np.flatnonzero(reach_cells[row])
            if not len(cols):
                continue
            lo, hi = cols[0] - k, cols[-1] - k
            level = int(hi - lo + 1).bit_length() - 1
            row_start = start + (row - k) * width
            np.minimum(covered_until, runs[level][row_start + lo], out=covered_until)
            np.minimum(covered_until, runs[level][row_start + hi + 1 - 2 ** level],
                       out=covered_until)
        hidden[idx[sel]] = covered_until > order[sel]
    return hidden


class _OccludedMarkerCollection(PathCollection):
    """A colour-mapped `PathCollection` of circles that skips hidden markers.

    With one colour per point, ``Collection.draw`` cannot stamp one cached
    marker (see `_HollowMarkerCollection`) and rasterizes every marker on its
    own, which for 175,000 points takes seconds. In a dense scatter most of
    those markers end up fully covered by markers drawn after them. Under Agg
    this class works out which ones (see `_hidden_markers`) and draws only
    the rest, in their original order, through the same per-point path. The
    image is the same pixel for pixel, with no snapping.

    Outside ``draw`` the collection is exactly what ``ax.scatter`` returned:
    the array, offsets, colours, norm and colormap cover every point. Other
    renderers (PDF, SVG, PS), and any setting the test does not cover
    (per-point sizes, alpha or line widths, dashed edges, hatches, a clip
    path, other marker shapes), draw every point as before.
    """

    def draw(self, renderer):
        # _hidden settles the colours when it gets that far, so the colours
        # saved below are the full set.
        hidden = self._hidden(renderer) if self.get_visible() else None
        if hidden is None or not hidden.any():
            super().draw(renderer)
            return
        keep = ~hidden
        saved = (self._A, self._offsets, self._mapped_colors,
                 self._facecolors, self._edgecolors)
        self._A, self._offsets = self._A[keep], self._offsets[keep]
        try:
            super().draw(renderer)
        finally:
            (self._A, self._offsets, self._mapped_colors,
             self._facecolors, self._edgecolors) = saved

    def _hidden(self, renderer):
        """The markers this draw can leave out, or None to draw them all."""
        if not isinstance(renderer, RendererAgg):
            return None
        n = len(self._offsets)
        # A legend handle copies the full array but holds one or two offsets.
        if (self._A is None or np.ndim(self._A) != 1 or len(self._A) != n
                or np.iterable(self._alpha)
                or len(self._sizes) != 1 or len(self._linewidths) != 1
                or len(self._antialiaseds) != 1 or len(self._urls) != 1
                or any(dashes is not None for _, dashes in self._linestyles)
                or self._hatch or self.get_path_effects()
                or self.get_sketch_params() is not None or self.get_snap()
                or (self.get_clip_on() and self.get_clip_path() is not None)):
            return None
        self.update_scalarmappable()
        if (not self._face_is_mapped or len(self._facecolors) != n
                or (not self._edge_is_mapped and len(self._edgecolors) > 1)):
            return None
        # The marker size in pixels depends on the dpi of this draw.
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)
        transform, offset_trf, offsets, paths = self._prepare_points()
        if (len(paths) != 1 or not transform.is_affine
                or not np.array_equal(paths[0].vertices, _CIRCLE.vertices)
                or not np.array_equal(paths[0].codes, _CIRCLE.codes)):
            return None
        m = (transforms.Affine2D(self.get_transforms()[0]) + transform).get_matrix()
        if m[0, 1] or m[1, 0] or abs(m[0, 0]) != abs(m[1, 1]):
            return None
        clip = [0.0, 0.0, float(renderer.width), float(renderer.height)]
        if self.get_clip_on() and self.get_clip_box() is not None:
            bx0, by0, bx1, by1 = self.get_clip_box().extents
            clip = [max(clip[0], bx0), max(clip[1], by0),
                    min(clip[2], bx1), min(clip[3], by1)]
        return _hidden_markers(
            np.asarray(offset_trf.transform(offsets), dtype=float),
            self._facecolors[:, 3] == 1.0,
            radius=0.5 * abs(m[0, 0]),
            linewidth=renderer.points_to_pixels(self._linewidths[0]),
            clip=tuple(clip))


class ScatterXY:
    """Scatter plot of x vs y with optional z colour-coding and bin aggregation. See :meth:`__init__`."""

    def __init__(
            self,
            x: Series,
            y: Series,
            z: Series = None,
            nbins: int = 0,
            binagg: Literal['mean', 'median'] = 'median',
    ):
        """Scatter plot with optional third variable as color and bin aggregation.

        Visualize relationships between two variables (x, y) with optional color-coding
        by a third variable (z). Supports optional binning and aggregation for trend
        visualization with confidence intervals (median ± IQR or mean ± std).

        Args:
            x: Series for x-axis
            y: Series for y-axis
            z: Optional Series for color-coding scatter points
            nbins: Number of bins for x-axis aggregation (0 = no aggregation)
            binagg: Aggregation method for bins ('mean' or 'median', default: 'median')

        Features:
            - 2-variable scatter: Basic x vs y plot
            - 3-variable scatter: Color-code points by z variable with colorbar
            - Bin aggregation: Group data by x-axis bins and overlay trends
            - Confidence intervals: Show IQR (median) or std (mean) per bin

        Call `plot()` to render with styling options (labels, limits, title, colormap).

        Artists:
            The points are the first `PathCollection` in ``ax.collections``,
            as ``ax.scatter`` returns it. Without ``z`` the collection is a
            `PathCollection` subclass that draws all markers from one cached
            marker, which is much faster for large records; each marker then
            sits on the nearest pixel centre, up to half a pixel from its
            exact position. With ``z`` it is a `PathCollection` subclass that,
            in raster output, leaves out the markers that opaque markers drawn
            after them cover completely; the image is unchanged.

        See Also:
            examples/visualization/plot_scatter_xy_basic.py — Scatter plot variations with 2D and 3D coloring
        """
        self.xname = x.name
        self.yname = y.name
        self.zname = z.name if z is not None else None
        self.nbins = nbins
        self.binagg = binagg
        self.fig = None
        self.ax = None

        # Internal, guaranteed-unique column keys for data access. The display
        # names (xname/yname/zname) are kept only for labels: x, y and z may
        # share a name (e.g. colouring a variable by itself), which would make
        # pd.concat produce duplicate column labels and turn xy_df[name] into a
        # DataFrame instead of a Series.
        self._xc, self._yc, self._zc = "_x", "_y", "_z"

        # Prepare data
        df_list = [x.rename(self._xc), y.rename(self._yc)]
        if z is not None:
            df_list.append(z.rename(self._zc))
        self.xy_df = pd.concat(df_list, axis=1)
        self.xy_df = self.xy_df.dropna()

        self.binagg = None if self.nbins == 0 else self.binagg

        if self.nbins > 0:
            self._databinning()

    def _databinning(self):
        group, bins = pd.qcut(self.xy_df[self._xc], q=self.nbins, retbins=True, duplicates='drop')
        groupcol = f'GROUP_{self._xc}'
        self.xy_df[groupcol] = group
        self.xy_df_binned = self.xy_df.groupby(groupcol).agg({'mean', 'median', 'std', 'count', q25, q75})

    def plot(
            self,
            ax: plt.Axes = None,
            format_style: FormatStyle = None,
            xlim: list = None,
            ylim: list or Literal['auto'] = None,
            cmap: str = 'viridis',
            show_colorbar: bool = True,
            markersize: float = 40,
            alpha: float = 1.0,
            vmin: float = None,
            vmax: float = None,
    ):
        """Generate plot with optional styling and formatting.

        Renders scatter plot with all styling parameters. Can be called multiple
        times with different parameters/axes to explore different views of the same data.

        Chrome (title, labels, units, font sizes, colours, grid, legend, zero line)
        comes from a shared :class:`~diive.plotting.FormatStyle` so it looks and is
        configured the same way as every other diive plot. The data-rendering
        arguments (cmap/markersize/alpha/vmin/vmax/colorbar, axis limits) stay here.

        Args:
            ax: Matplotlib axes to plot on (default: creates new figure if None)
            format_style: A :class:`~diive.plotting.FormatStyle` describing the chrome
                (title/xlabel/ylabel/zlabel/xunits/yunits). When None the diive house
                style is used. The colorbar label reads from ``format_style.zlabel``.
            xlim: X-axis limits as [min, max] (default: data min/max)
            ylim: Y-axis limits as [min, max] or 'auto' (default: data range)
            cmap: Colormap name for z variable (default: 'viridis')
                  Examples: 'plasma', 'viridis', 'coolwarm', 'RdYlBu'
            show_colorbar: Display colorbar if z provided (default: True)
            markersize: Scatter point area in points^2 (default: 40)
            alpha: Scatter point opacity, 0-1 (default: 1.0)
            vmin: Lower bound of the z colour scale (default: data minimum)
            vmax: Upper bound of the z colour scale (default: data maximum)

        Notes:
            - xlim always uses full data range (no quantile trimming)
            - ylim='auto' with nbins: uses binned data limits
            - Colorbar shown only if z is provided and show_colorbar=True
        """
        style = format_style or FormatStyle()

        # Colorbar label comes only from the style; fall back to the z series name.
        zlabel = style.zlabel if style.zlabel is not None else self.zname

        if not ax:
            self.fig, self.ax = pf.create_ax(figsize=(8, 8))
            self._plot(self.ax, style, zlabel, xlim, ylim, cmap, show_colorbar,
                       markersize=markersize, alpha=alpha, vmin=vmin, vmax=vmax)
            # Skip tight_layout if colorbar is present (incompatible with new layout engine)
            if self.zname is None or not show_colorbar:
                plt.tight_layout()
            self.fig.show()
        else:
            self.ax = ax
            self._plot(self.ax, style, zlabel, xlim, ylim, cmap, show_colorbar,
                       markersize=markersize, alpha=alpha, vmin=vmin, vmax=vmax)

    def _plot(self, ax: plt.Axes, style: FormatStyle, zlabel: str = None,
              xlim: list = None, ylim: list or str = None, cmap: str = 'viridis',
              show_colorbar: bool = True, nbins: int = 10,
              markersize: float = 40, alpha: float = 1.0, vmin: float = None, vmax: float = None):
        """Generate plot on axis"""
        nbins += 1  # To include zero

        # Scatter plot with optional color
        if self.zname is not None:
            scatter = ax.scatter(x=self.xy_df[self._xc],
                                 y=self.xy_df[self._yc],
                                 c=self.xy_df[self._zc],
                                 s=markersize,
                                 alpha=alpha,
                                 marker='o',
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax,
                                 label=self.yname)
            # Same collection, drawn without its hidden markers (see the class).
            scatter.__class__ = _OccludedMarkerCollection
            if show_colorbar:
                cbar = ax.figure.colorbar(scatter, ax=ax)
                cbar.set_label(zlabel if zlabel else self.zname, fontsize=12)
        else:
            points = ax.scatter(x=self.xy_df[self._xc],
                                y=self.xy_df[self._yc],
                                c='none',
                                s=markersize,
                                alpha=alpha,
                                marker='o',
                                edgecolors='#607D8B',
                                label=self.yname)
            # Same collection, drawn through draw_markers (see the class).
            points.__class__ = _HollowMarkerCollection

        if self.nbins > 0:

            _min = self.xy_df_binned[self._yc]['count'].min()
            _max = self.xy_df_binned[self._yc]['count'].max()
            ax.plot(self.xy_df_binned[self._xc][self.binagg],
                    self.xy_df_binned[self._yc][self.binagg],
                    c='r', ms=10, marker='o', lw=2,
                    # c='none', ms=80, marker='o', edgecolors='r', lw=2,
                    label=f"binned data ({self.binagg}, {_min}-{_max} values per bin)")

            if self.binagg == 'median':
                ax.fill_between(self.xy_df_binned[self._xc][self.binagg],
                                self.xy_df_binned[self._yc]['q25'],
                                self.xy_df_binned[self._yc]['q75'],
                                alpha=.2, zorder=10, color='red',
                                label="interquartile range")

            if self.binagg == 'mean':
                ax.errorbar(x=self.xy_df_binned[self._xc][self.binagg],
                            y=self.xy_df_binned[self._yc][self.binagg],
                            xerr=self.xy_df_binned[self._xc]['std'],
                            yerr=self.xy_df_binned[self._yc]['std'],
                            elinewidth=3, ecolor='red', alpha=.6, lw=0,
                            label="standard deviation")

        self._apply_format(ax, style, xlim, ylim)
        ax.locator_params(axis='x', nbins=nbins)
        ax.locator_params(axis='y', nbins=nbins)

    def _apply_format(self, ax: plt.Axes, style: FormatStyle,
                      xlim: list = None, ylim: list or str = None):

        if xlim:
            xmin = xlim[0]
            xmax = xlim[1]
        else:
            xmin = self.xy_df[self._xc].min()
            xmax = self.xy_df[self._xc].max()
        ax.set_xlim(xmin, xmax)

        if ylim == 'auto':
            if self.binagg == 'median':
                ymin = self.xy_df_binned[self._yc]['q25'].min()
                ymax = self.xy_df_binned[self._yc]['q75'].max()
            elif self.binagg == 'mean':
                _lowery = self.xy_df_binned[self._yc]['mean'].sub(self.xy_df_binned[self._yc]['std'])
                _uppery = self.xy_df_binned[self._yc]['mean'].add(self.xy_df_binned[self._yc]['std'])
                ymin = _lowery.min()
                ymax = _uppery.max()
            else:
                ymin = self.xy_df[self._yc].quantile(0.01)
                ymax = self.xy_df[self._yc].quantile(0.99)
        elif isinstance(ylim, list):
            ymin = ylim[0]
            ymax = ylim[1]
        else:
            ymin = self.xy_df[self._yc].min()
            ymax = self.xy_df[self._yc].max()

        ax.set_ylim(ymin, ymax)

        # Shared formatting layer: title/labels/units/fonts/grid/legend/zeroline.
        style.apply(ax=ax, default_title=f"{self.yname} vs. {self.xname}",
                    default_xlabel=self.xname, default_ylabel=self.yname,
                    zeroline_data=self.xy_df[self._yc])


def scatter_to_code(xcol: str, ycol: str, zcol: str = None, *,
                    nbins: int = 0, binagg: str = 'median',
                    cmap: str = 'viridis', show_colorbar: bool = True,
                    markersize: float = 40, alpha: float = 1.0,
                    vmin: float = None, vmax: float = None,
                    format_kwargs: dict = None, df_name: str = 'df') -> str:
    """Return a runnable snippet that reproduces a :class:`ScatterXY` plot.

    Mirrors what the GUI's Scatter XY tab renders: the X/Y (and optional Z)
    columns of ``df_name``, the binning, and the data-render arguments. Only the
    non-default ``FormatStyle`` fields (from ``format_kwargs``) are emitted, so a
    plot left at the house style produces a clean call.
    """
    init = [f"    x={df_name}[{xcol!r}],", f"    y={df_name}[{ycol!r}],"]
    if zcol:
        init.append(f"    z={df_name}[{zcol!r}],")
    init += [f"    nbins={nbins!r},", f"    binagg={binagg!r},"]

    plot = ["    ax=ax,"]
    fmt = {k: v for k, v in (format_kwargs or {}).items() if v is not None}
    if fmt:
        args = ", ".join(f"{k}={v!r}" for k, v in fmt.items())
        plot.append(f"    format_style=dv.plotting.FormatStyle({args}),")
    plot += [
        f"    cmap={cmap!r},",
        f"    show_colorbar={show_colorbar!r},",
        f"    markersize={markersize!r},",
        f"    alpha={alpha!r},",
        f"    vmin={vmin!r},",
        f"    vmax={vmax!r},",
    ]
    return (
        "import matplotlib.pyplot as plt\n"
        "import diive as dv\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 8))\n"
        "dv.plotting.ScatterXY(\n"
        + "\n".join(init) + "\n"
        ").plot(\n"
        + "\n".join(plot) + "\n"
        ")\n"
        "plt.show()\n"
    )
