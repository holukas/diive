"""
PLOTTING: DIEL CYCLE
====================

Plot the mean diel (24-hour) cycle of a time series, optionally one curve per month.

Part of the diive library: https://github.com/holukas/diive
"""
import calendar
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import set_fig
from diive.core.plotting.styles.format import FormatStyle
from diive.core.times.resampling import diel_cycle


class DielCycle:
    """Plot the diel (24-hour) cycle of a time series. See :meth:`__init__` and :meth:`plot`."""

    def __init__(self, series: Series):
        """Plot diel cycles of time series.

        Args:
            series: Time series with datetime index.
                The index must contain date and time info.

        See Also:
            examples/visualization/dielcycle.py — Diurnal cycle analysis and visualization
        """
        self.series = series

        self.var = self.series.name

        self.fig = None
        self.ax = None
        self.showplot = False
        self._diel_cycles_df = None
        self.show_xticklabels = True
        self.show_xlabel = True

    def get_data(self) -> DataFrame:
        """Return the diel-cycle aggregates DataFrame (raises if :meth:`plot` not yet run)."""
        return self.diel_cycles_df

    @property
    def diel_cycles_df(self):
        """Return dataframe containing diel cycle aggregates."""
        if not isinstance(self._diel_cycles_df, DataFrame):
            raise Exception(f'No diel cycles dataframe available. Please run .plot() first.')
        return self._diel_cycles_df

    @staticmethod
    def _month_colors(n: int, cmap: str = None) -> list:
        """Colours for up to `n` per-month curves: sampled from `cmap` if given,
        else the diive 12-month palette."""
        if cmap:
            mpl_cmap = plt.get_cmap(cmap)
            return [mpl_cmap(i / max(1, n - 1)) for i in range(max(n, 1))]
        return theme.colors_12_months()

    def _band_bounds(self, month, agg_col: str, band: str, central: Series):
        """Return (lower, upper) Series for the uncertainty band, or (None, None).

        `sd`/`se` are symmetric around the central curve; `iqr`/`minmax` are the
        raw percentile / range columns (independent of the central aggregation).
        """
        row = self.diel_cycles_df.loc[month]
        if band == 'sd':
            sd = row['std']
            return central - sd, central + sd
        if band == 'se':
            se = row['std'] / np.sqrt(row['count'])
            return central - se, central + se
        if band == 'iqr':
            return row['q25'], row['q75']
        if band == 'minmax':
            return row['min'], row['max']
        return None, None

    #: Central aggregation -> (diel_cycle column, diel_cycle flag needed).
    _AGG_COLUMN = {
        'mean': ('mean', 'mean'), 'median': ('median', 'median'),
        'min': ('min', 'minmax'), 'max': ('max', 'minmax'),
        'p25': ('q25', 'quantiles'), 'p75': ('q75', 'quantiles'),
    }

    def plot(self,
             ax: plt.Axes = None,
             format_style: FormatStyle = None,
             color: str = None,
             agg: Literal['mean', 'median', 'min', 'max', 'p25', 'p75'] = 'mean',
             band: Literal['none', 'sd', 'se', 'iqr', 'minmax'] = 'sd',
             each_month: bool = False,
             cmap: str = None,
             marker: bool = False,
             markersize: float = 4,
             linewidth: float = 2,
             ylim: list = None,
             show_xticklabels: bool = True,
             show_xlabel: bool = True,
             mean: bool = None,
             std: bool = None,
             **kwargs):
        """Plot the diel (24-hour) cycle, optionally one curve per month.

        Chrome (title, labels, units, font sizes, colours, grid, legend, zero
        line) comes from a shared :class:`~diive.plotting.FormatStyle`.

        Args:
            agg: Central aggregation drawn as the curve: ``'mean'``, ``'median'``,
                ``'min'``, ``'max'``, ``'p25'`` or ``'p75'`` (25th/75th percentile).
            band: Uncertainty band drawn around the curve: ``'sd'`` (±1 standard
                deviation), ``'se'`` (±1 standard error), ``'iqr'`` (25th-75th
                percentile), ``'minmax'`` (per-time-of-day range), or ``'none'``.
            each_month: One curve per month (seasonal pattern) instead of a single
                curve over all data.
            cmap: Colormap name to draw the per-month colours from (``each_month``
                only). ``None`` uses the diive 12-month palette.
            marker: Draw a marker at each time-of-day point.
            markersize: Marker size (when ``marker`` is True).
            format_style: A :class:`~diive.plotting.FormatStyle` describing the
                chrome. When None the diive house style is used.
            mean: Deprecated. Kept for back-compat; ignored (use ``agg``).
            std: Deprecated. Kept for back-compat; ``False`` maps to ``band='none'``.
        """

        # Back-compat: the old boolean std= toggle maps onto the band selector.
        if std is not None and band == 'sd':
            band = 'sd' if std else 'none'

        agg = agg.lower()
        band = band.lower()
        agg_col, agg_flag = self._AGG_COLUMN.get(agg, ('mean', 'mean'))

        self.show_xticklabels = show_xticklabels
        self.show_xlabel = show_xlabel

        style = format_style or FormatStyle()

        # Resample, computing only the aggregates the chosen curve + band need.
        flags = {'mean': False, 'std': False, 'median': False,
                 'quantiles': False, 'minmax': False}
        flags[agg_flag] = True
        if band in ('sd', 'se'):
            flags['std'] = True
        elif band == 'iqr':
            flags['quantiles'] = True
        elif band == 'minmax':
            flags['minmax'] = True
        self._diel_cycles_df = diel_cycle(series=self.series, each_month=each_month,
                                          **flags)

        months = set(self.diel_cycles_df.index.get_level_values(0).tolist())

        self.fig, self.ax, self.showplot = set_fig(ax=ax)

        counter_plotted = -1
        n_months = len(months)
        alpha = 0.05 if n_months > 10 else 0.1
        auto_color = True if not color else False
        month_colors = self._month_colors(n_months, cmap) if auto_color else None
        for counter, month in enumerate(months):

            central = self.diel_cycles_df.loc[month][agg_col]
            if central.isnull().all():
                continue
            else:
                counter_plotted += 1

            lower, upper = self._band_bounds(month, agg_col, band, central)

            if auto_color:
                color = month_colors[counter_plotted % len(month_colors)]

            monthstr = calendar.month_abbr[month] if each_month else agg

            # Convert datetime.time index to decimal hours for a clean numeric x-axis.
            # Using datetime.time objects directly creates a categorical axis whose
            # registered categories may not include exact hour boundaries (e.g. when
            # timestamps are at :15/:45 offsets), causing set_xticks to fail.
            x_decimal = [t.hour + t.minute / 60 + t.second / 3600 for t in central.index]
            central_numeric = central.copy()
            central_numeric.index = x_decimal
            central_numeric.plot(ax=self.ax, label=f'{monthstr}', color=color, zorder=99,
                                 lw=linewidth,
                                 marker='o' if marker else None, markersize=markersize,
                                 **kwargs)

            if lower is not None:
                self.ax.fill_between(x_decimal,
                                     upper.to_numpy(),
                                     lower.to_numpy(),
                                     alpha=alpha, zorder=0, color=color, edgecolor='none',
                                     label=None)

        # Shared formatting layer: title/labels/units/fonts/grid/legend/zeroline.
        style.apply(ax=self.ax, default_xlabel="Time (hours of day)",
                    default_ylabel=self.series.name,
                    zeroline_data=self.diel_cycles_df[agg_col])

        # show_xlabel=False blanks the (already-set) x-label.
        if not self.show_xlabel:
            self.ax.set_xlabel("")

        # Fixed 0-24h diel x-axis with hourly ticks (domain-specific, not chrome).
        self.ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
        self.ax.set_xticklabels([3, 6, 9, 12, 15, 18, 21])
        if not self.show_xticklabels:
            self.ax.set_xticklabels([])
        self.ax.set_xlim([0, 24])
        self.ax.set_ylim(ylim)

        if self.showplot:
            self.fig.tight_layout()
            self.fig.show()
