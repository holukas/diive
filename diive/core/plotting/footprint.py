"""
FOOTPRINT: MAP OF A FLUX FOOTPRINT OR FOOTPRINT CLIMATOLOGY
===========================================================

Plot of the results of ``ffp_footprint`` and ``ffp_climatology`` (Kljun et al. 2015).

Part of the diive library: https://github.com/holukas/diive

Please cite the paper of the footprint model when you show its results:

    Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015). A simple two-dimensional
    parameterisation for Flux Footprint Prediction (FFP). Geoscientific Model Development,
    8(11), 3695-3713. https://doi.org/10.5194/gmd-8-3695-2015

The map shows the footprint field on a logarithmic colour scale, the source-area
contour lines with their percentages, the tower, the footprint peak, range rings
around the tower and a north arrow. The subtitle gives the peak distance, the
distance of the 80 % source area and, for a climatology, the number of footprints.
For a single footprint, a side panel shows the crosswind-integrated footprint and its
cumulative contribution along the wind.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.styles._material_design_colors import (
    amber, bluegray, deeppurple, indigo, lightblue, teal, white)
from diive.core.plotting.styles.format import FormatStyle

_INK = bluegray(800)
_SOFT_INK = bluegray(400)
_HALO = [pe.withStroke(linewidth=3, foreground=white())]

# Light to dark: the rarely seen edge of the footprint fades into the paper, the core
# is dark. All stops are Material Design colours.
_FOOTPRINT_COLORS = ['#FFFFFF', lightblue(50), lightblue(200), teal(400), indigo(500),
                     deeppurple(900)]


def footprint_cmap() -> mcolors.Colormap:
    """Default colour map of the footprint field (white to deep purple)."""
    cmap = mcolors.LinearSegmentedColormap.from_list('diive_footprint', _FOOTPRINT_COLORS)
    cmap.set_under((1, 1, 1, 0))
    cmap.set_bad((1, 1, 1, 0))
    return cmap


class FootprintPlot:
    """Map of a flux footprint (``ffp_footprint``) or footprint climatology (``ffp_climatology``).

    Two-phase design: the result goes to ``__init__``, all styling to :meth:`plot`.

    Args:
        result: ``FFPFootprint`` or ``FFPClimatology`` from
            ``diive.flux.lowres.footprint``.
        source_area: Source area (fraction or percent) whose distance from the tower
            the subtitle reports (default 80 %). If it is not one of the contour
            lines of the result, the largest contour line inside the grid is used.

    Attributes:
        is_climatology: True for an ``FFPClimatology``.
        is_geographic: True if x points east and y north. False for a single
            footprint calculated without wind direction, which stays in the
            along-wind frame (x upwind, y crosswind) and gets no north arrow.
        n: Number of footprints (1 for a single footprint).
        peak_xy: Position of the footprint peak (m east, m north of the tower).
        peak_distance: Distance of the peak from the tower (m). For a single
            footprint this is ``x_ci_max``, the peak of the crosswind-integrated
            footprint. For a climatology it is the distance of the maximum of the
            averaged field.
        contours: List of ``(r, x, y)`` for each contour line inside the grid.
        source_area_r: Source area of ``source_area_distance`` (fraction), None if
            no contour line lies inside the grid.
        source_area_distance: Largest distance (m) of the ``source_area_r``
            contour line from the tower.
        cumulative_distance: For a single footprint, the distance (m) along the
            wind where the cumulative crosswind-integrated footprint reaches
            ``source_area``, None for a climatology.
        fig: Figure of the last :meth:`plot`.
        ax: Map axes of the last :meth:`plot`.
        colorbar: Colour bar of the last :meth:`plot`, None without one.
        ax_profile: Axes of the crosswind-integrated footprint (:meth:`plot_profile`).
        ax_cumulative: Axes of the cumulative footprint (:meth:`plot_profile`).

    Example:
        >>> import diive as dv
        >>> from diive.flux.lowres.footprint import ffp_footprint
        >>> res = ffp_footprint(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4,
        ...                     z0=0.15, wind_dir=220.)
        >>> ax = dv.plotting.FootprintPlot(res).plot()

    See Also:
        examples/visualization/plot_footprint.py

    References:
        Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015). A simple two-dimensional
        parameterisation for Flux Footprint Prediction (FFP). Geoscientific Model Development,
        8(11), 3695-3713. https://doi.org/10.5194/gmd-8-3695-2015
    """

    def __init__(self, result, source_area: float = 0.8):
        if hasattr(result, 'fclim_2d'):
            self.is_climatology = True
            self.f = np.asarray(result.fclim_2d, dtype=float)
            self.n = int(result.n)
        elif hasattr(result, 'f_2d'):
            self.is_climatology = False
            self.f = np.asarray(result.f_2d, dtype=float)
            self.n = 1
        else:
            raise TypeError("`result` must be an FFPFootprint or FFPClimatology.")
        self.result = result
        self.x = np.asarray(result.x_2d, dtype=float)
        self.y = np.asarray(result.y_2d, dtype=float)
        # A single footprint without wind direction stays in the along-wind frame:
        # x is constant across the wind and y along it
        self.is_geographic = self.is_climatology or not (
                np.array_equal(self.x, np.broadcast_to(self.x[:, :1], self.x.shape))
                and np.array_equal(self.y, np.broadcast_to(self.y[:1, :], self.y.shape)))

        self.contours = []
        if result.rs is not None and result.fr is not None:
            for r, fr, xr, yr in zip(result.rs, result.fr, result.xr, result.yr):
                if fr is not None and xr is not None:
                    self.contours.append((float(r), np.asarray(xr), np.asarray(yr)))
        self._levels = {float(r): fr for r, fr in zip(result.rs or [], result.fr or [])
                        if fr is not None}

        # Peak: the direction comes from the maximum of the 2-D field
        i, j = np.unravel_index(np.nanargmax(self.f), self.f.shape)
        px, py = self.x[i, j], self.y[i, j]
        if self.is_climatology:
            self.peak_xy = (float(px), float(py))
            self.peak_distance = float(np.hypot(px, py))
        else:
            # The 2-D maximum lies on the centre line, at the grid cell next to x_ci_max
            d = np.hypot(px, py)
            self.peak_distance = float(result.x_ci_max)
            scale = self.peak_distance / d if d > 0 else 0.
            self.peak_xy = (float(px * scale), float(py * scale))

        r_target = source_area / 100. if source_area >= 1 else float(source_area)
        self.source_area_r = None
        self.source_area_distance = None
        if self.contours:
            match = [c for c in self.contours if np.isclose(c[0], r_target)]
            r, xr, yr = match[0] if match else self.contours[-1]
            self.source_area_r = r
            self.source_area_distance = float(np.max(np.hypot(xr, yr)))

        self.cumulative_distance = None
        if not self.is_climatology:
            x_ci = np.asarray(result.x_ci, dtype=float)
            f_ci = np.asarray(result.f_ci, dtype=float)
            cum = np.concatenate(([0.], np.cumsum(0.5 * (f_ci[1:] + f_ci[:-1]) * np.diff(x_ci))))
            self._x_ci, self._f_ci, self._cum_ci = x_ci, f_ci, cum
            if cum[-1] >= r_target:
                self.cumulative_distance = float(np.interp(r_target, cum, x_ci))
        self._r_target = r_target

        self.fig = None
        self.ax = None
        self.ax_profile = None
        self.ax_cumulative = None
        self.colorbar = None

    # ------------------------------------------------------------------ geometry

    def _auto_extent(self, pad: float = 0.12):
        """Map limits around the largest contour line and the tower, padded."""
        if self.contours:
            xs = np.concatenate([self.contours[-1][1], [0.]])
            ys = np.concatenate([self.contours[-1][2], [0.]])
        else:
            # Without contour lines: the cells above 1 % of the maximum
            keep = self.f >= 0.01 * np.nanmax(self.f)
            xs = np.concatenate([self.x[keep], [0.]])
            ys = np.concatenate([self.y[keep], [0.]])
        # Square around the centre of the bounding box, so the map fills a square slot
        half = (1 + 2 * pad) * max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
        cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
        return cx - half, cx + half, cy - half, cy + half

    def _field_in_extent(self, extent, max_cells: int = 400):
        """Field cropped to the extent and thinned to at most max_cells per axis."""
        xmin, xmax, ymin, ymax = extent
        inside = (self.x >= xmin) & (self.x <= xmax) & (self.y >= ymin) & (self.y <= ymax)
        rows = np.where(inside.any(axis=1))[0]
        cols = np.where(inside.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return self.x, self.y, self.f
        r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, self.f.shape[0])
        c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, self.f.shape[1])
        sr = max(1, int(np.ceil((r1 - r0) / max_cells)))
        sc = max(1, int(np.ceil((c1 - c0) / max_cells)))
        sl = (slice(r0, r1, sr), slice(c0, c1, sc))
        return self.x[sl], self.y[sl], self.f[sl]

    @staticmethod
    def _nice_step(span: float, n: int = 4) -> float:
        """Round step (1, 2, 2.5 or 5 times a power of ten) for about n intervals."""
        raw = span / n
        mag = 10 ** np.floor(np.log10(raw))
        for m in (1, 2, 2.5, 5, 10):
            if m * mag >= raw:
                return m * mag
        return 10 * mag

    # ------------------------------------------------------------------ plotting

    def plot(self,
             ax=None,
             format_style: FormatStyle = None,
             figsize: tuple = None,
             figdpi: int = 100,
             cmap=None,
             vmin: float = None,
             vmax: float = None,
             extent: tuple = None,
             contour_color: str = _INK,
             contour_linewidth: float = 0.9,
             highlight_color: str = amber(700),
             show_contour_labels: bool = True,
             show_peak: bool = True,
             show_rings: bool = True,
             show_north_arrow: bool = True,
             show_colorbar: bool = True,
             show_profile: bool = True,
             show_info: bool = True,
             cb_label: str = 'Footprint (m$^{-2}$)',
             cb_labelsize: float = None,
             annotation_fontsize: float = None):
        """Draw the footprint map (Phase 2 of the two-phase design).

        Args:
            ax: Matplotlib axes for the map. If None, a new figure is created, and for
                a single footprint with ``show_profile`` also the side panel.
            format_style: Shared :class:`~diive.plotting.FormatStyle` for title, axis
                labels, ticks, grid and legend (``show_legend``, ``legend_loc``,
                ``legend_ncol``, ``legend_fontsize``). The title is placed on the left.
                The default style has no grid, the range rings take its place.
            figsize: Figure size in inches, only without ``ax``.
            figdpi: Figure DPI, only without ``ax``.
            cmap: Colour map of the footprint field (default: white to deep purple).
            vmin: Lowest footprint value with colour (m-2). Smaller values are
                transparent. Default: a third of the value of the outermost contour line.
            vmax: Highest footprint value of the colour scale (default: the maximum).
            extent: Map limits ``(xmin, xmax, ymin, ymax)`` in m. Default: the largest
                contour line and the tower, padded.
            contour_color: Colour of the contour lines.
            contour_linewidth: Line width of the contour lines.
            highlight_color: Colour of the source-area line reported in the subtitle
                (80 % by default) and of the peak marker.
            show_contour_labels: Label the contour lines with their percentages.
            show_peak: Mark the footprint peak.
            show_rings: Draw range rings around the tower.
            show_north_arrow: Draw a north arrow.
            show_colorbar: Draw the colour bar right of the map, with the contour
                levels marked on it.
            show_profile: Without ``ax``, add the side panel of a single footprint
                (crosswind-integrated footprint and its cumulative). See
                :meth:`plot_profile`.
            show_info: Write peak distance, source-area distance and number of
                footprints below the title.
            cb_label: Colour bar label.
            cb_labelsize: Colour bar font size (default: 0.8 times the tick font size).
            annotation_fontsize: Font size of the contour labels, ring labels and the
                info line (default 11).

        Returns:
            The map axes.
        """
        style = format_style or FormatStyle(show_grid=False)
        ticks_fs = style.ticks_fontsize if style.ticks_fontsize is not None else theme.FONTSIZE_TICKS
        cb_labelsize = cb_labelsize if cb_labelsize is not None else ticks_fs * 0.8
        ann_fs = annotation_fontsize if annotation_fontsize is not None else 11
        with_profile = ax is None and show_profile and not self.is_climatology
        own_figure = ax is None

        if ax is None:
            # Fixed rectangles: a square map slot (the map extent is square), the
            # colour bar to its right and, for a single footprint, the profile panels
            if figsize is None:
                figsize = (15, 7) if with_profile else (8.5, 7.6)
            fig = plt.figure(figsize=figsize, dpi=figdpi)
            fw, fh = figsize
            side = 0.78 * fh if with_profile else 0.7 * fw
            left = 0.065 if with_profile else 0.13
            ax = fig.add_axes((left, 0.1, side / fw, side / fh))
            if with_profile:
                pl, pw, ph, gap = 0.6, 0.37, (side / fh - 0.07) / 2, 0.07
                ax_cum = fig.add_axes((pl, 0.1, pw, ph))
                ax_prof = fig.add_axes((pl, 0.1 + ph + gap, pw, ph), sharex=ax_cum)
        self.fig = ax.figure
        self.ax = ax
        self.colorbar = None
        self._label_boxes = []  # contour labels: data position, pixel width and height

        if extent is None:
            extent = self._auto_extent()
        xmin, xmax, ymin, ymax = extent

        # Footprint field, log scale: the source areas are spread over orders of magnitude
        cmap = footprint_cmap() if cmap is None else mpl.colormaps.get_cmap(cmap)
        levels_sorted = sorted(self._levels.values())
        if vmax is None:
            vmax = float(np.nanmax(self.f))
        if vmin is None:
            vmin = levels_sorted[0] / 3. if levels_sorted else vmax * 1e-3
        xg, yg, fg = self._field_in_extent(extent)
        fg = np.where(fg > 0, fg, np.nan)
        mesh = ax.pcolormesh(xg, yg, fg, cmap=cmap,
                             norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                             shading='gouraud', rasterized=True, zorder=1)

        style.apply(ax, default_title='Footprint climatology' if self.is_climatology
                    else 'Flux footprint',
                    default_xlabel='Distance east of tower (m)' if self.is_geographic
                    else 'Distance upwind of tower (m)',
                    default_ylabel='Distance north of tower (m)' if self.is_geographic
                    else 'Crosswind distance (m)')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', adjustable='box')

        if show_rings:
            self._draw_rings(ax, extent, ann_fs)

        # Contour lines, the reported source area emphasised. A light halo keeps the
        # lines visible on the dark core of the field.
        first_plain = True
        for r, xr, yr in self.contours:
            is_hi = self._is_highlight(r)
            lw = contour_linewidth * (2.2 if is_hi else 1.)
            if is_hi:
                label = f"{r * 100:g} % source area"
            elif first_plain:
                plain = [c[0] * 100 for c in self.contours if not self._is_highlight(c[0])]
                label = (f"Source areas {plain[0]:g}–{plain[-1]:g} %" if len(plain) > 1
                         else f"{plain[0]:g} % source area")
                first_plain = False
            else:
                label = None
            ax.plot(xr, yr, color=highlight_color if is_hi else contour_color, lw=lw,
                    solid_joinstyle='round', zorder=4, label=label,
                    path_effects=[pe.Stroke(linewidth=lw + 1.8, foreground=white(), alpha=0.55),
                                  pe.Normal()])
        if show_contour_labels and self.contours:
            self._label_contours(ax, ann_fs, contour_color, highlight_color)

        # Tower
        ax.plot([0], [0], marker='o', ms=9, mfc=_INK, mec=white(), mew=2, ls='none', zorder=8,
                label='Tower')

        if show_peak:
            ax.plot([0, self.peak_xy[0]], [0, self.peak_xy[1]], color=_INK, lw=1, ls=(0, (2, 2)),
                    zorder=5)
            ax.plot([self.peak_xy[0]], [self.peak_xy[1]], marker='D', ms=7, mfc=highlight_color,
                    mec=white(), mew=1.5, ls='none', zorder=8,
                    label="Peak")

        if show_north_arrow and self.is_geographic:
            self._draw_north_arrow(ax, ann_fs)

        if show_colorbar:
            if own_figure:
                cb = ax.figure.colorbar(mesh, cax=ax.inset_axes([1.035, 0., 0.03, 0.5]),
                                        extend='min')
            else:
                # Space taken from the caller's axes, as matplotlib does by default
                cb = ax.figure.colorbar(mesh, ax=ax, extend='min', shrink=0.5, anchor=(0, 0),
                                        fraction=0.04, pad=0.03)
            self.colorbar = cb
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=cb_labelsize, length=2, color=_SOFT_INK,
                              labelcolor=_INK)
            cb.set_label(cb_label, fontsize=cb_labelsize, color=_INK)
            # Contour levels as marks on the colour bar
            for r, fr in self._levels.items():
                if vmin <= fr <= vmax:
                    cb.ax.axhline(fr, color=highlight_color if self._is_highlight(r) else white(),
                                  lw=1.6 if self._is_highlight(r) else 0.8)

        if style.show_legend:
            legend_fs = style.legend_fontsize if style.legend_fontsize is not None else ann_fs
            leg = ax.legend(loc=style.legend_loc, ncol=style.legend_ncol, fontsize=legend_fs,
                            frameon=True, framealpha=0.85, edgecolor='none', borderpad=0.6,
                            handlelength=1.6, labelcolor=_INK)
            leg.set_zorder(10)
            if style.legend_loc == 'best':
                self._place_legend(ax, leg, skip_upper_right=show_north_arrow and self.is_geographic)

        self._title_and_info(ax, ann_fs * 1.15 if show_info else None)

        if with_profile:
            self.plot_profile(ax_profile=ax_prof, ax_cumulative=ax_cum,
                              format_style=style, highlight_color=highlight_color,
                              annotation_fontsize=ann_fs)
        return ax

    def plot_profile(self,
                     ax_profile=None,
                     ax_cumulative=None,
                     format_style: FormatStyle = None,
                     figsize: tuple = (7, 7),
                     figdpi: int = 100,
                     color: str = indigo(500),
                     highlight_color: str = amber(700),
                     xmax: float = None,
                     annotation_fontsize: float = None):
        """Crosswind-integrated footprint and its cumulative along the wind.

        Only for a single footprint.

        Args:
            ax_profile: Axes of the crosswind-integrated footprint. If None (and
                ``ax_cumulative`` is None), a new figure with both panels is created.
            ax_cumulative: Axes of the cumulative footprint.
            format_style: Shared :class:`~diive.plotting.FormatStyle`.
            figsize: Figure size in inches, only for a new figure.
            figdpi: Figure DPI, only for a new figure.
            color: Colour of the curves.
            highlight_color: Colour of the peak and source-area markers.
            xmax: Largest distance shown (default: 1.6 times the source-area distance).
            annotation_fontsize: Font size of the annotations (default 11).

        Returns:
            Tuple of the two axes.
        """
        if self.is_climatology:
            raise ValueError("The profile exists only for a single footprint.")
        style = format_style or FormatStyle(show_grid=False)
        ann_fs = annotation_fontsize if annotation_fontsize is not None else 11
        if ax_profile is None and ax_cumulative is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi)
            gs = fig.add_gridspec(2, 1, left=0.16, right=0.95, bottom=0.1, top=0.95, hspace=0.12)
            ax_profile = fig.add_subplot(gs[0])
            ax_cumulative = fig.add_subplot(gs[1], sharex=ax_profile)
        self.ax_profile, self.ax_cumulative = ax_profile, ax_cumulative

        x, f, cum = self._x_ci, self._f_ci, self._cum_ci
        if xmax is None:
            ref = self.cumulative_distance or self.source_area_distance or x[-1]
            xmax = min(1.6 * ref, x[-1])
        keep = x <= xmax * 1.02
        xk, fk, ck = x[keep], f[keep], cum[keep]
        pct = f"{self._r_target * 100:g} %"

        if ax_profile is not None:
            ax = ax_profile
            ax.fill_between(xk, 0, fk * 1000, color=color, alpha=0.18, lw=0, zorder=2)
            ax.plot(xk, fk * 1000, color=color, lw=2, zorder=3)
            fpeak = np.interp(self.peak_distance, x, f) * 1000
            ax.plot([self.peak_distance] * 2, [0, fpeak], color=highlight_color, lw=1.5, zorder=4)
            ax.plot([self.peak_distance], [fpeak], marker='D', ms=7, mfc=highlight_color,
                    mec=white(), mew=1.5, zorder=5)
            ax.annotate(f"Peak {self.peak_distance:.0f} m", (self.peak_distance, fpeak),
                        xytext=(10, 2), textcoords='offset points', fontsize=ann_fs,
                        color=_INK, fontweight='bold', va='center')
            style.merged(title='').apply(ax, default_ylabel='Crosswind-integrated\n'
                                                            'footprint (km$^{-1}$)')
            ax.set_ylim(0, fk.max() * 1000 * 1.12)
            ax.tick_params(labelbottom=False)
            ax.set_xlabel('')
            ax.tick_params(top=False, right=False)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)

        if ax_cumulative is not None:
            ax = ax_cumulative
            ax.fill_between(xk, 0, ck * 100, color=color, alpha=0.08, lw=0, zorder=2)
            ax.plot(xk, ck * 100, color=color, lw=2, zorder=3)
            if self.cumulative_distance is not None:
                d = self.cumulative_distance
                r = self._r_target * 100
                ax.plot([0, d], [r, r], color=highlight_color, lw=1.2, ls=(0, (3, 2)), zorder=4)
                ax.plot([d, d], [0, r], color=highlight_color, lw=1.2, ls=(0, (3, 2)), zorder=4)
                ax.plot([d], [r], marker='o', ms=7, mfc=highlight_color, mec=white(), mew=1.5,
                        zorder=5)
                ax.annotate(f"{pct} within {d:.0f} m", (d, r), xytext=(-8, 8),
                            textcoords='offset points', ha='right', fontsize=ann_fs, color=_INK,
                            fontweight='bold')
            style.merged(title='').apply(ax, default_xlabel='Distance upwind of tower (m)',
                                         default_ylabel='Cumulative\nfootprint (%)')
            ax.set_ylim(0, 100)
            ax.set_xlim(0, xmax)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
            ax.tick_params(top=False, right=False)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
        return ax_profile, ax_cumulative

    # ------------------------------------------------------------------ decorations

    def _is_highlight(self, r) -> bool:
        return self.source_area_r is not None and bool(np.isclose(r, self.source_area_r))

    def _draw_rings(self, ax, extent, fs):
        """Dashed range rings around the tower with their distance."""
        xmin, xmax, ymin, ymax = extent
        corners = np.hypot([xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax])
        rmax = corners.max()
        step = self._nice_step(min(xmax - xmin, ymax - ymin) / 2, n=3)
        theta = np.linspace(0, 2 * np.pi, 361)
        # Label each ring as far from the footprint direction as the map allows
        peak_ang = np.arctan2(self.peak_xy[1], self.peak_xy[0])
        cand = np.deg2rad(np.arange(0, 360, 5))
        cand = cand[np.argsort(-np.abs(np.angle(np.exp(1j * (cand - peak_ang)))), kind='stable')]
        mx, my = 0.06 * (xmax - xmin), 0.04 * (ymax - ymin)
        for k in range(1, int(rmax / step) + 1):
            rr = k * step
            ax.plot(rr * np.cos(theta), rr * np.sin(theta), color=_SOFT_INK, lw=0.7,
                    ls=(0, (1, 3)), zorder=2)
            for a in cand:
                lx, ly = rr * np.cos(a), rr * np.sin(a)
                # Keep clear of the north arrow in the upper right corner
                in_arrow = (lx - xmin) / (xmax - xmin) > 0.82 and (ly - ymin) / (ymax - ymin) > 0.72
                if xmin + mx < lx < xmax - mx and ymin + my < ly < ymax - my and not in_arrow:
                    ax.text(lx, ly, f"{rr:g} m", fontsize=fs * 0.85, color=_SOFT_INK,
                            ha='center', va='center', path_effects=_HALO, zorder=3)
                    break

    def _label_contours(self, ax, fs, color, highlight_color):
        """Percent labels where each contour line crosses the ray from the tower through the peak.

        Labels that would overlap a larger one are left out, so the outer lines (the
        ones read most often) always keep theirs.
        """
        ang = np.arctan2(self.peak_xy[1], self.peak_xy[0])
        ux, uy = np.cos(ang), np.sin(ang)
        ax.apply_aspect()  # transforms with the final aspect
        to_px = ax.transData.transform
        placed = []
        pt = fs * ax.figure.dpi / 72  # font size in pixels
        for r, xr, yr in reversed(self.contours):
            # Vertex nearest the ray beyond the peak
            along = xr * ux + yr * uy
            across = np.abs(-xr * uy + yr * ux)
            cand = np.where(along > self.peak_distance)[0]
            if cand.size == 0:
                continue
            k = cand[np.argmin(across[cand] - 1e-9 * along[cand])]
            text = f"{r * 100:g} %"
            # Approximate label box (pixels), with the bbox padding
            p, w, h = to_px((xr[k], yr[k])), (0.6 * len(text) + 0.6) * pt, 1.6 * pt
            if any(abs(p[0] - q[0]) < (w + qw) / 2 and abs(p[1] - q[1]) < (h + qh) / 2
                   for q, qw, qh in placed):
                continue
            placed.append((p, w, h))
            self._label_boxes.append(((xr[k], yr[k]), w, h))
            is_hi = self._is_highlight(r)
            ax.text(xr[k], yr[k], text, fontsize=fs, ha='center', va='center',
                    color=highlight_color if is_hi else color,
                    fontweight='bold' if is_hi else 'normal',
                    bbox=dict(boxstyle='round,pad=0.18', fc=white(), ec='none', alpha=0.9),
                    zorder=6)

    def _place_legend(self, ax, leg, skip_upper_right):
        """Move the legend to the corner that hides the least of the footprint.

        Matplotlib's ``'best'`` counts every line vertex, so the range rings (long
        lines across the whole map) outweigh a hidden contour label. Here only the
        contour lines count, and covering a contour label, the tower or the peak
        rules a corner out.
        """
        ax.apply_aspect()  # final axes box, also after a colour bar took space
        to_px = ax.transData.transform
        verts = (np.concatenate([to_px(np.column_stack([xr, yr])) for _, xr, yr in self.contours])
                 if self.contours else np.empty((0, 2)))
        points = [to_px((0., 0.)), to_px(self.peak_xy)]
        boxes = [(to_px(xy), w, h) for xy, w, h in self._label_boxes]
        locs = ['upper left', 'lower left', 'lower right'] + ([] if skip_upper_right else ['upper right'])
        scores = []
        for loc in locs:
            leg.set_loc(loc)
            bb = leg.get_window_extent()
            covered = sum(bb.contains(*p) for p in points) + sum(
                abs(c[0] - (bb.x0 + bb.x1) / 2) < (w + bb.width) / 2
                and abs(c[1] - (bb.y0 + bb.y1) / 2) < (h + bb.height) / 2 for c, w, h in boxes)
            scores.append(1e6 * covered + bb.count_contains(verts))
        leg.set_loc(locs[int(np.argmin(scores))])

    @staticmethod
    def _draw_north_arrow(ax, fs):
        """North arrow in the upper right corner of the map."""
        x, y0, y1 = 0.93, 0.83, 0.94
        ax.annotate('', xy=(x, y1), xytext=(x, y0), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>,head_width=0.35,head_length=0.7',
                                    color=_INK, lw=1.6, shrinkA=0, shrinkB=0), zorder=9)
        ax.text(x, y0 - 0.015, 'N', transform=ax.transAxes, ha='center', va='top',
                fontsize=fs * 1.2, fontweight='bold', color=_INK, path_effects=_HALO, zorder=9)

    def info_text(self) -> str:
        """Key numbers of the footprint as one line (number, peak, source-area distance)."""
        parts = []
        if self.is_climatology:
            parts.append(f"n = {self.n:,} footprints")
        parts.append(f"Peak at {self.peak_distance:.0f} m")
        if self.source_area_distance is not None:
            parts.append(f"{self.source_area_r * 100:g} % contour reaches "
                         f"{self.source_area_distance:.0f} m")
        return '   ·   '.join(parts)

    def _title_and_info(self, ax, fs):
        """Left-aligned title with the key numbers as a subtitle below it.

        ``FormatStyle.apply`` sets a centred title. It is moved to the left slot with
        the same font, and the subtitle (if fs is not None) goes between title and map.
        """
        t = ax.title
        text = t.get_text()
        if fs is not None:
            # Offset in points, like the title pad: an offset in axes fraction grows
            # with the map height and pushed the subtitle into the title on tall maps
            ax.annotate(self.info_text(), xy=(0, 1), xycoords='axes fraction',
                        xytext=(0, fs * 0.4), textcoords='offset points', ha='left',
                        va='bottom', fontsize=fs, color=_SOFT_INK)
        if text:
            props = dict(color=t.get_color(), fontsize=t.get_fontsize(),
                         fontweight=t.get_fontweight())
            ax.set_title('')
            ax.set_title(text, loc='left', pad=fs * 2.2 if fs is not None else 8, **props)
