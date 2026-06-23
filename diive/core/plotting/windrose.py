"""
WINDROSE: VARIABLE AGGREGATED BY WIND DIRECTION
===============================================

Radial (polar) plot that aggregates any variable into wind-direction sectors.
Wind direction (degrees, 0-360, meteorological "from" convention) is binned into
``n_sectors`` compass sectors; within each sector the paired variable is reduced
by a chosen aggregation (mean / median / min / max / sum / std / count). Each
sector is drawn as a polar bar whose radial length is the aggregated value, with
North at the top and angles increasing clockwise — the standard wind-rose layout.

Unlike a classic wind rose (which shows wind-speed frequency by direction), this
plots the directional response of an arbitrary variable, e.g. mean CO2 flux by
wind sector, or total precipitation by sector.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pandas import Series
from rich.table import Table

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.styles.format import FormatStyle
from diive.core.utils.console import console, info

# 16-point compass labels, North-first, clockwise. 8- and 4-sector layouts are
# regular subsets of this list (every 2nd / every 4th label).
_COMPASS_16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
               'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

# Aggregation name -> results-table column. Drives both the plotted radius and
# the validation of the `agg` argument.
_AGG_COLUMNS = {
    'mean': 'MEAN',
    'median': 'MEDIAN',
    'min': 'MIN',
    'max': 'MAX',
    'sum': 'SUM',
    'std': 'STD',
    'count': 'N_VALS',
}


class WindRosePlot:
    """Aggregate a variable into wind-direction sectors and draw it as a polar bar plot.

    Bins wind direction into ``n_sectors`` equal compass sectors (North-centred,
    so the North sector straddles 0/360 degrees) and reduces the paired variable
    within each sector by ``agg``. The result is a tidy per-sector table
    (:attr:`results`) and a wind-rose-style polar bar plot where each bar's length
    is the sector's aggregated value.

    Two-phase design: data + aggregation parameters go to ``__init__``; all styling
    lives in :meth:`plot`, which can be called repeatedly with different looks.

    Args:
        series: Variable to aggregate (numeric Series).
        wind_dir: Wind direction in degrees (0-360), aligned to ``series`` by index.
        agg: Aggregation applied per sector — one of ``'mean'``, ``'median'``,
            ``'min'``, ``'max'``, ``'sum'`` (cumulative), ``'std'``, ``'count'``.
            Drives the plotted radius; the full table always holds every statistic.
        n_sectors: Number of equal wind-direction sectors (default 8). 4, 8 and 16
            get compass labels (N, NE, ...); other counts are labelled by degrees.
        z: Optional second variable used to **colour** the bars. When given, it is
            aggregated per sector (over its own valid records) and drives the bar
            colours + colorbar, while the bar *lengths* still come from ``series``.
            When omitted, bars are coloured by their own aggregated value.
        z_agg: Aggregation for ``z`` (same choices as ``agg``); defaults to ``agg``.
        verbose: Emit the Rich per-sector report on construction (default False).

    Attributes:
        results: Per-sector ``DataFrame`` indexed by sector label, with columns
            ``CENTER_DEG``, ``N_VALS``, ``MEAN``, ``MEDIAN``, ``MIN``, ``MAX``,
            ``STD``, ``SUM`` (plus ``Z`` when a colour variable is given).

    Example::

        import diive as dv
        df, meta = dv.load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN()
        rose = dv.plotting.WindRosePlot(series=df['co2_flux'], wind_dir=df['wind_dir'],
                                        agg='mean', n_sectors=16)
        rose.plot(cmap='RdBu_r', cb_label='Mean CO2 flux')
        print(rose.results)

    See Also:
        examples/visualization/plot_windrose_basic.py
    """

    def __init__(self,
                 series: Series,
                 wind_dir: Series,
                 agg: str = 'mean',
                 n_sectors: int = 8,
                 z: Series = None,
                 z_agg: str = None,
                 verbose: bool = False):
        if not isinstance(series, Series):
            raise TypeError("`series` must be a pandas Series.")
        if not isinstance(wind_dir, Series):
            raise TypeError("`wind_dir` must be a pandas Series.")
        if agg not in _AGG_COLUMNS:
            raise ValueError(f"`agg` must be one of {sorted(_AGG_COLUMNS)}, got '{agg}'.")
        if n_sectors < 2:
            raise ValueError(f"`n_sectors` must be >= 2, got {n_sectors}.")
        if z is not None and not isinstance(z, Series):
            raise TypeError("`z` must be a pandas Series.")
        # z defaults to the same aggregation as the bar variable.
        z_agg = agg if z_agg is None else z_agg
        if z is not None and z_agg not in _AGG_COLUMNS:
            raise ValueError(f"`z_agg` must be one of {sorted(_AGG_COLUMNS)}, got '{z_agg}'.")

        self.series = series
        self.wind_dir = wind_dir
        self.agg = agg
        self.n_sectors = int(n_sectors)
        self.z = z
        self.z_agg = z_agg
        self.verbose = verbose

        self.varname = series.name if series.name is not None else 'variable'
        self.z_varname = (z.name if z is not None and z.name is not None
                          else 'z') if z is not None else None
        self.results = None  # set by _aggregate()
        self.fig = None
        self.ax = None

        self._aggregate()

        if self.verbose:
            self.report()

    def _tick_layout(self, max_labels: int) -> tuple[np.ndarray, list[str]]:
        """Pick angular tick positions (degrees) + labels for the circumference.

        With few sectors each sector is labelled at its centre. Once there are
        more than ``max_labels`` sectors the per-sector labels collide, so we
        fall back to a fixed ring of compass bearings (16 or 8 cardinal points)
        — the standard wind-rose reference instead of an unreadable degree fan.
        """
        n = self.n_sectors
        if n <= max_labels:
            return self.results['CENTER_DEG'].to_numpy(dtype=float), list(self.results.index)
        n_show = 16 if max_labels >= 16 else 8
        labels = list(_COMPASS_16) if n_show == 16 else _COMPASS_16[::2]
        bearings = np.array([i * 360.0 / n_show for i in range(n_show)], dtype=float)
        return bearings, labels

    @staticmethod
    def _auto_decimals(ticks, max_decimals: int = 4) -> int:
        """Fewest decimals that render the colorbar ticks losslessly (0 = integers)."""
        vals = np.asarray([t for t in ticks if np.isfinite(t)], dtype=float)
        if vals.size == 0:
            return 0
        for d in range(max_decimals + 1):
            if np.allclose(vals, np.round(vals, d), atol=10 ** -(d + 4), rtol=0):
                return d
        return max_decimals

    def _sector_labels(self) -> list[str]:
        """Return compass labels for 4/8/16 sectors, else degree labels."""
        n = self.n_sectors
        if n == 16:
            return list(_COMPASS_16)
        if n == 8:
            return _COMPASS_16[::2]
        if n == 4:
            return _COMPASS_16[::4]
        step = 360.0 / n
        return [f"{i * step:.0f}°" for i in range(n)]

    def _sector_index(self, wd: Series) -> Series:
        """Assign each wind-direction reading to a North-centred sector index."""
        sector_width = 360.0 / self.n_sectors
        # Shift by half a sector so sector 0 is centred on North (covers
        # [-half, +half) around 0), matching the compass-labelled layout.
        half = sector_width / 2.0
        idx = (((wd + half) % 360.0) // sector_width).astype(int)
        return idx.clip(upper=self.n_sectors - 1)  # guard FP edge

    @staticmethod
    def _reduce(vals: Series, agg: str):
        """Apply a single aggregation to a sector's values (NaN when empty)."""
        if vals.count() == 0:
            return np.nan
        return getattr(vals, agg)()

    def _aggregate(self):
        """Bin wind direction into sectors and compute per-sector statistics."""
        # Pair the two series on their shared index, drop rows missing either value
        # or carrying an out-of-range / invalid direction.
        df = pd.DataFrame({'val': self.series, 'wd': self.wind_dir}).dropna()
        df = df[(df['wd'] >= 0) & (df['wd'] <= 360)].copy()
        # 360 degrees is the same bearing as 0 (North); fold it in so it does not
        # fall outside the sector range.
        df.loc[df['wd'] == 360, 'wd'] = 0.0

        self.n_used = len(df)
        sector_idx = self._sector_index(df['wd'])

        labels = self._sector_labels()
        sector_width = 360.0 / self.n_sectors
        centers = [i * sector_width for i in range(self.n_sectors)]

        rows = []
        for i in range(self.n_sectors):
            vals = df.loc[sector_idx == i, 'val']
            rows.append({
                'SECTOR': labels[i],
                'CENTER_DEG': centers[i],
                'N_VALS': int(vals.count()),
                'MEAN': self._reduce(vals, 'mean'),
                'MEDIAN': self._reduce(vals, 'median'),
                'MIN': self._reduce(vals, 'min'),
                'MAX': self._reduce(vals, 'max'),
                'STD': self._reduce(vals, 'std'),
                'SUM': self._reduce(vals, 'sum'),
            })

        self.results = pd.DataFrame(rows).set_index('SECTOR')

        # Optional colour variable: aggregate it per sector (over its own valid
        # records, independent of the bar variable's gaps) into a 'Z' column.
        if self.z is not None:
            self.results['Z'] = self._aggregate_z(labels)

    def _aggregate_z(self, labels: list[str]) -> list[float]:
        """Aggregate the optional colour variable ``z`` per wind sector."""
        zdf = pd.DataFrame({'z': self.z, 'wd': self.wind_dir}).dropna()
        zdf = zdf[(zdf['wd'] >= 0) & (zdf['wd'] <= 360)].copy()
        zdf.loc[zdf['wd'] == 360, 'wd'] = 0.0
        sector_idx = self._sector_index(zdf['wd'])
        return [self._reduce(zdf.loc[sector_idx == i, 'z'], self.z_agg)
                for i in range(self.n_sectors)]

    def report(self):
        """Print a Rich table of the per-sector aggregation plus a short summary."""
        col = _AGG_COLUMNS[self.agg]
        res = self.results

        table = Table(title=f"Wind rose: {self.varname} by wind direction "
                            f"({self.agg}, {self.n_sectors} sectors)",
                      title_style="bold blue", header_style="bold")
        table.add_column("Sector", justify="left", style="cyan")
        table.add_column("Center", justify="right")
        table.add_column("N", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Sum", justify="right")
        if self.z is not None:
            table.add_column(f"{self.z_agg} {self.z_varname}", justify="right", style="magenta")

        # Highlight the sector that holds the extreme of the *plotted* aggregate.
        plotted = res[col]
        hi = plotted.idxmax() if plotted.notna().any() else None

        def _f(x):
            return "-" if pd.isna(x) else f"{x:.3g}"

        for sector, r in res.iterrows():
            style = "bold green" if sector == hi else None
            cells = [
                str(sector), f"{r['CENTER_DEG']:.0f}°", f"{int(r['N_VALS'])}",
                _f(r['MEAN']), _f(r['MEDIAN']), _f(r['MIN']), _f(r['MAX']),
                _f(r['STD']), _f(r['SUM']),
            ]
            if self.z is not None:
                cells.append(_f(r['Z']))
            table.add_row(*cells, style=style)

        console.print(table)
        info(f"{self.n_used} records aggregated into {self.n_sectors} wind sectors "
             f"(plotted aggregate: '{self.agg}').", verbose=self.verbose)
        if hi is not None:
            info(f"Largest '{self.agg}' of {self.varname}: sector {hi} "
                 f"({plotted.loc[hi]:.3g}).", verbose=self.verbose)

    def plot(self,
             ax=None,
             format_style: FormatStyle = None,
             figsize: tuple = (9, 9),
             figdpi: int = 100,
             cmap: str = 'viridis',
             color: str = None,
             vmin: float = None,
             vmax: float = None,
             alpha: float = 0.9,
             edgecolor: str = 'white',
             edgewidth: float = 0.8,
             bar_width_frac: float = 0.9,
             show_colorbar: bool = True,
             show_sector_labels: bool = True,
             max_sector_labels: int = 16,
             show_zero_circle: bool = True,
             cb_label: str = None,
             cb_labelsize: float = None,
             cb_digits_after_comma: int = None,
             sector_label_fontsize: float = None,
             sector_label_pad: float = 14.0):
        """Render the wind rose on a polar axis (Phase 2 of the two-phase design).

        Args:
            ax: Polar matplotlib axes. If None, a new polar figure is created.
            format_style: Shared :class:`~diive.plotting.FormatStyle`. On this polar
                plot only the ``title`` / title-font fields apply (the cartesian
                spines/ticks/grid of ``FormatStyle.apply`` do not).
            figsize: Figure size in inches (default (9, 9)).
            figdpi: Figure DPI (default 100).
            cmap: Colormap used to colour bars by the colour source — the ``z``
                aggregate when one was given, else the bar value (default
                ``'viridis'``). Ignored when ``color`` is given.
            color: Single solid bar colour. When set, overrides ``cmap`` and the
                colorbar is suppressed.
            vmin: Low colour anchor (defaults to the colour source's minimum).
            vmax: High colour anchor (defaults to the colour source's maximum).
            alpha: Bar opacity (default 0.9).
            edgecolor: Bar edge colour (default 'white').
            edgewidth: Bar edge line width (default 0.8).
            bar_width_frac: Bar angular width as a fraction of the sector width
                (default 0.9; 1.0 = touching bars, lower = gaps between sectors).
            show_colorbar: Draw the value colorbar (default True; ignored with ``color``).
            show_sector_labels: Draw compass / degree labels around the rose (default True).
            max_sector_labels: Maximum number of per-sector labels before switching
                to a fixed ring of compass bearings (default 16). With more sectors
                than this, 16 (or 8) cardinal compass points are shown instead of one
                crowded label per sector.
            show_zero_circle: Draw a reference circle at value 0 when the aggregate
                spans negative values (default True).
            cb_label: Colorbar label (defaults to "{z_agg} {z_varname}" when a
                colour variable is given, else "{agg} {varname}").
            cb_labelsize: Colorbar tick/label font size (default theme value).
            cb_digits_after_comma: Decimal places on colorbar ticks. Default None
                auto-picks the fewest decimals the tick values need (integer labels
                when they suffice); pass an int to force a fixed count.
            sector_label_fontsize: Compass-label font size (default theme value).
            sector_label_pad: Radial gap (points) between the outer ring and the
                compass labels (default 14), so labels clear the plotting area.
        """
        style = format_style or FormatStyle()

        if cb_labelsize is None:
            cb_labelsize = theme.AX_LABELS_FONTSIZE
        if sector_label_fontsize is None:
            sector_label_fontsize = theme.AX_LABELS_FONTSIZE
        title_fs = style.title_fontsize if style.title_fontsize is not None else theme.FONTSIZE_TITLE
        title_color = style.text_color if style.text_color is not None else theme.COLOR_TEXT

        col = _AGG_COLUMNS[self.agg]
        values = self.results[col].to_numpy(dtype=float)
        centers_deg = self.results['CENTER_DEG'].to_numpy(dtype=float)

        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("No finite aggregated values to plot (all sectors empty).")
        data_min = float(np.nanmin(finite))
        data_max = float(np.nanmax(finite))

        # Colour source: the optional z aggregate when given, else the bar values.
        # Bar *length* always comes from `values`; only the colour changes.
        if self.z is not None:
            color_values = self.results['Z'].to_numpy(dtype=float)
        else:
            color_values = values
        finite_c = color_values[np.isfinite(color_values)]
        color_min = float(np.nanmin(finite_c)) if finite_c.size else 0.0
        color_max = float(np.nanmax(finite_c)) if finite_c.size else 1.0

        if vmin is None:
            vmin = color_min
        if vmax is None:
            vmax = color_max

        # Radial extent spans zero so the zero line is a real ring on the plot:
        # positive aggregates occupy [0, data_max], negative ones [data_min, 0].
        rmin = min(0.0, data_min)
        rmax = max(0.0, data_max)
        span = (rmax - rmin) or (abs(rmax) or 1.0)

        if ax is None:
            self.fig = plt.figure(figsize=figsize, dpi=figdpi)
            ax = self.fig.add_subplot(111, projection='polar')
        else:
            self.fig = ax.figure
        self.ax = ax

        # Meteorological layout: North at top, angles increase clockwise.
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        sector_width_rad = 2.0 * np.pi / self.n_sectors
        theta = np.deg2rad(centers_deg)
        bar_width = sector_width_rad * bar_width_frac

        # Anchor every bar at the zero line: positive aggregates grow outward from
        # zero, negative ones hang inward toward the centre. Bar *length* equals the
        # magnitude, so a sector only slightly above zero shows a short bar instead
        # of stretching all the way out from the inner hub.
        finite_mask = np.isfinite(values)
        bottoms = np.where(finite_mask, np.minimum(values, 0.0), 0.0)
        heights = np.where(finite_mask, np.abs(values), 0.0)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap(cmap)
        if color is not None:
            bar_colors = [color] * self.n_sectors
        else:
            bar_colors = [colormap(norm(v)) if np.isfinite(v) else (0, 0, 0, 0)
                          for v in color_values]

        ax.bar(theta, heights, width=bar_width, bottom=bottoms,
               color=bar_colors, edgecolor=edgecolor, linewidth=edgewidth,
               alpha=alpha, zorder=3)

        # Reference circle at value 0 when the aggregate reaches below zero.
        if show_zero_circle and rmin < 0:
            circle_theta = np.linspace(0, 2.0 * np.pi, 360)
            ax.plot(circle_theta, np.zeros_like(circle_theta),
                    color=theme.COLOR_CHROME, linewidth=1.0, linestyle='--',
                    alpha=0.7, zorder=2)

        ax.set_rorigin(rmin - span * 0.05)
        ax.set_rlim(rmin, rmax + span * 0.08)

        # Compass / degree tick labels around the circumference. Past
        # `max_sector_labels` sectors the per-sector labels collide, so a fixed
        # ring of compass bearings is shown instead.
        if show_sector_labels:
            tick_deg, tick_labels = self._tick_layout(max_sector_labels)
            ax.set_xticks(np.deg2rad(tick_deg))
            ax.set_xticklabels(tick_labels, fontsize=sector_label_fontsize)
            # Push the compass labels clear of the outer ring so they never
            # overlap the plotting area.
            ax.tick_params(axis='x', pad=sector_label_pad)
        else:
            ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=theme.TICKS_LABELS_FONTSIZE - 4)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        if style.title:
            self.fig.subplots_adjust(top=0.92)
            self.fig.suptitle(style.title, fontsize=title_fs, color=title_color,
                              fontweight=style.title_fontweight, y=0.97)

        if show_colorbar and color is None:
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cb_extend = 'neither'
            if vmin > color_min and vmax < color_max:
                cb_extend = 'both'
            elif vmin > color_min:
                cb_extend = 'min'
            elif vmax < color_max:
                cb_extend = 'max'
            cb = self.fig.colorbar(sm, ax=ax, orientation='vertical',
                                   shrink=0.6, pad=0.1, extend=cb_extend)
            # Integer tick labels when they suffice; add decimals only when the
            # actual tick values need them (or use the caller's explicit count).
            # Set the colorbar's *own* formatter + update_ticks(): a colorbar
            # resets its axis formatter on every draw, so a bare
            # yaxis.set_major_formatter() would be reverted.
            digits = (self._auto_decimals(cb.get_ticks())
                      if cb_digits_after_comma is None else cb_digits_after_comma)
            cb.formatter = mticker.FormatStrFormatter(f'%.{digits}f')
            cb.update_ticks()
            if cb_label is not None:
                label = cb_label
            elif self.z is not None:
                label = f"{self.z_agg} {self.z_varname}"
            else:
                label = f"{self.agg} {self.varname}"
            cb.set_label(label, fontsize=cb_labelsize, rotation=90, labelpad=10)
            cb.ax.tick_params(labelsize=cb_labelsize)

        return self.ax
