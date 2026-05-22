"""
ANALYSIS: OPTIMUM RANGE
=======================

Find optimal ranges for bivariate relationships using statistical methods.
Identify acceptable operating conditions and valid data subsets.

Part of the diive library: https://github.com/holukas/diive
"""

from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple
from pandas import DataFrame

from diive.core.plotting.plotfuncs import save_fig


class FindOptimumRange:

    def __init__(self,
                 df: DataFrame,
                 xcol: str,
                 ycol: str,
                 n_bins: int = 100,
                 bins_agg: Literal['median', 'mean'] = 'median',
                 rwinsize: float = 0.1,
                 ragg: Literal['mean', 'median'] = 'mean',
                 define_optimum: Literal['min', 'max'] = 'max',
                 threshold: float = 0.95,
                 prominence_threshold: float = 1.0):
        """
        Find x range for optimum y.

        Bins x data and aggregates y values per bin, then applies rolling aggregation
        to identify the range where y is optimized (min or max). The optimum range is
        defined as the contiguous region of bins where the smoothed curve stays within
        `threshold` of the peak, measured relative to the full curve range. Useful for
        finding optimal conditions for ecosystem responses.

        Args:
            df: Input DataFrame
            xcol: Column name of driver variable (x)
            ycol: Column name of response variable (y)
            n_bins: Number of bins to divide x data into
            bins_agg: Aggregation method for y values within bins (default: 'median')
            rwinsize: Window size for rolling aggregation (as fraction of total bins)
            ragg: Rolling aggregation method (default: 'mean')
            define_optimum: Whether optimum is 'min' or 'max' of y
            threshold: Fraction of the curve range defining the optimum region. Bins are
                included while the smoothed value stays within (1 - threshold) of the
                curve range from the peak (default: 0.95). Must be in (0, 1).
            prominence_threshold: Minimum peak prominence (in units of curve std) required
                to consider the optimum meaningful. The peak prominence is
                |peak - curve_mean| / curve_std. Results below this value set
                `is_optimum_prominent=False` (default: 1.0).

        Properties:
            .results_optrange: Dictionary with optimum range results and data.
                Includes `is_optimum_prominent` (bool) and `optimum_prominence` (float)
                flagging whether the peak stands out from the curve mean by at least
                `prominence_threshold` std.

        Methods:
            .find_optimum(): Calculate optimum range
            .showfig(): Display analysis visualizations

        Example:
            See `examples/analysis/analysis_optimumrange.py` for complete examples.

        See Also:
            find_optimum_range : Convenience function.
        """
        self.df = df[[xcol, ycol]].copy()
        self.xcol = xcol
        self.ycol = ycol
        self.n_bins = n_bins
        self.bins_agg = bins_agg
        self.rwinsize = rwinsize
        self.ragg = ragg
        self.define_optimum = define_optimum
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold
        self.prominence_threshold = prominence_threshold

        self._results_optrange = {}

    @property
    def results_optrange(self) -> dict:
        """Return optimum range results"""
        if not self._results_optrange:
            raise RuntimeError('Results for optimum range are empty, call find_optimum() first')
        return self._results_optrange

    def find_optimum(self):
        bin_aggs_df = self._divide_xdata_into_bins()

        winsize = int(self.n_bins * self.rwinsize)
        winsize = winsize + 1 if (winsize % 2 == 0) else winsize  # Must be odd number
        rbin_aggs_df = self._rolling_agg(bin_aggs_df=bin_aggs_df,
                                         use_bin_agg=self.bins_agg,
                                         rolling_agg=self.ragg,
                                         winsize=winsize)

        roptimum_bin, roptimum_val = self._find_rolling_optimum(rolling_df=rbin_aggs_df,
                                                                use_rolling_agg=self.ragg)

        optimum_xstart, optimum_xend, optimum_ymean, \
            optimum_start_bin, optimum_end_bin = self._get_optimum_range(grouped_df=bin_aggs_df,
                                                                         rbin_aggs_df=rbin_aggs_df,
                                                                         roptimum_bin=roptimum_bin)

        validity = self._check_optimum_validity(rbin_aggs_df=rbin_aggs_df, roptimum_val=roptimum_val)

        vals_in_optimum_range_df = \
            self._values_in_optimum_range(optimum_xstart=optimum_xstart, optimum_xend=optimum_xend)

        self._results_optrange = dict(
            optimum_xstart=optimum_xstart,
            optimum_xend=optimum_xend,
            optimum_ymean=optimum_ymean,
            optimum_start_bin=optimum_start_bin,
            optimum_end_bin=optimum_end_bin,
            bin_aggs_df=bin_aggs_df,
            rbin_aggs_df=rbin_aggs_df,
            rwinsize=winsize,
            roptimum_bin=roptimum_bin,
            roptimum_val=roptimum_val,
            n_xbins=self.n_bins,
            xcol=self.xcol,
            ycol=self.ycol,
            vals_in_optimum_range_df=vals_in_optimum_range_df,
            **validity
        )

    def _values_in_optimum_range(self, optimum_xstart: float, optimum_xend: float) -> pd.DataFrame:
        labels = ['vals_belowoptimum', 'vals_inoptimum', 'vals_aboveoptimum']
        bins = [-np.inf, optimum_xstart, optimum_xend, np.inf]

        xcol_with_category = pd.cut(self.df[self.xcol], bins=bins, labels=labels)

        xcounts_df = (
            pd.DataFrame({'category': xcol_with_category, 'year': self.df.index.year})
            .groupby(['year', 'category'], observed=False)
            .size()
            .unstack(fill_value=0)
        )
        xcounts_df['vals_total'] = xcounts_df[labels].sum(axis=1)

        for col in labels:
            xcounts_df[f'{col}_perc'] = xcounts_df[col].div(xcounts_df['vals_total']).multiply(100)

        return xcounts_df.fillna(0)

    def _divide_xdata_into_bins(self) -> DataFrame:
        xbins = pd.qcut(self.df[self.xcol], self.n_bins, duplicates='drop')
        bins_df = self.df.assign(xbins=xbins)
        return bins_df.groupby('xbins').agg([self.bins_agg, 'count'])

    def _rolling_agg(self, bin_aggs_df, use_bin_agg, winsize, rolling_agg):
        rolling_df = bin_aggs_df[self.ycol][use_bin_agg].rolling(winsize, center=True)
        return rolling_df.agg([rolling_agg, 'std']).dropna()

    def _find_rolling_optimum(self, rolling_df: DataFrame, use_rolling_agg: str = 'mean'):
        """Find the bin where the smoothed curve reaches its min or max."""
        if self.define_optimum == 'min':
            roptimum_bin = rolling_df[use_rolling_agg].idxmin()
        elif self.define_optimum == 'max':
            roptimum_bin = rolling_df[use_rolling_agg].idxmax()
        else:
            raise ValueError(f"define_optimum must be 'min' or 'max', got {self.define_optimum!r}")
        roptimum_val = rolling_df[use_rolling_agg][roptimum_bin]
        print(f"Optimum {self.define_optimum} found in bin: {roptimum_bin}  /  value: {roptimum_val}")
        return roptimum_bin, roptimum_val

    def _get_optimum_range(self, grouped_df: DataFrame, rbin_aggs_df: DataFrame,
                           roptimum_bin: pd.Interval):
        """Get optimum range by walking outward from the peak until curve drops below threshold.

        The threshold is applied relative to the full curve range so it works for any sign
        of y values. For define_optimum='max': include bins while smoothed value >=
        curve_max - (1 - threshold) * curve_range. For 'min': the mirror image.
        """
        curve = rbin_aggs_df[self.ragg]
        curve_min = curve.min()
        curve_max = curve.max()
        curve_range = curve_max - curve_min

        if self.define_optimum == 'max':
            threshold_val = curve_max - (1 - self.threshold) * curve_range
            in_range = curve >= threshold_val
        else:
            threshold_val = curve_min + (1 - self.threshold) * curve_range
            in_range = curve <= threshold_val

        peak_ix = rbin_aggs_df.index.get_loc(roptimum_bin)

        left_ix = peak_ix
        while left_ix > 0 and in_range.iloc[left_ix - 1]:
            left_ix -= 1

        right_ix = peak_ix
        while right_ix < len(rbin_aggs_df) - 1 and in_range.iloc[right_ix + 1]:
            right_ix += 1

        optimum_start_bin = rbin_aggs_df.index[left_ix]
        optimum_end_bin = rbin_aggs_df.index[right_ix]

        optimum_range_xstart = optimum_start_bin.left
        optimum_range_xend = optimum_end_bin.right

        start_loc = grouped_df.index.get_loc(optimum_start_bin)
        end_loc = grouped_df.index.get_loc(optimum_end_bin)
        optimum_range_ymean = grouped_df[self.ycol][self.bins_agg].iloc[start_loc:end_loc + 1].mean()

        return optimum_range_xstart, optimum_range_xend, optimum_range_ymean, \
            optimum_start_bin, optimum_end_bin

    def _check_optimum_validity(self, rbin_aggs_df: DataFrame, roptimum_val: float) -> dict:
        """Flag whether the optimum peak stands out from the overall curve.

        Computes prominence as the distance of the peak from the curve mean in units of
        curve std. Values below 1.0 suggest the curve is too flat to identify a
        meaningful optimum.
        """
        curve = rbin_aggs_df[self.ragg]
        curve_std = curve.std()
        prominence = abs(roptimum_val - curve.mean()) / curve_std if curve_std > 0 else 0.0
        is_prominent = prominence >= self.prominence_threshold
        if not is_prominent:
            print(f"(!) Optimum may not be meaningful: curve appears flat "
                  f"(prominence={prominence:.2f} < {self.prominence_threshold})")
        return dict(optimum_prominence=round(prominence, 3), is_optimum_prominent=is_prominent)

    def showfig(self,
                saveplot: bool = False,
                title: str = None,
                path: Path | str = None,
                xlabel: str = None,
                ylabel: str = None,
                xunit: str = None,
                yunit: str = None):
        n_years = len(self.results_optrange['vals_in_optimum_range_df'])

        # Top panel scales with number of years; middle and timeseries panels are fixed
        top_h = max(1.5, n_years * 0.38)
        mid_h = 5.0
        ts_h = 2.2
        fig = plt.figure(figsize=(16, top_h + mid_h + ts_h + 1.5))
        gs = gridspec.GridSpec(3, 1, height_ratios=[top_h, mid_h, ts_h])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        self.plot_vals_in_optimum_range(ax=ax1)
        self.plot_bin_aggregates(ax=ax2, xlabel=xlabel, ylabel=ylabel, xunit=xunit, yunit=yunit)
        self.plot_optimum_range_timeseries(ax=ax3)
        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold')
        fig.tight_layout(pad=1.5, h_pad=2.5)
        plt.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)

    def plot_vals_in_optimum_range(self, ax):
        """Plot optimum range: values in, above and below optimum per year"""

        # kudos: https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py

        # Get data
        df = self.results_optrange['vals_in_optimum_range_df'].copy()
        plotcols = ['vals_inoptimum_perc', 'vals_aboveoptimum_perc', 'vals_belowoptimum_perc']
        df = df[plotcols]
        df = df.round(1)

        category_names = ['in optimum range (%)', 'above optimum (%)', 'below optimum (%)']
        # soft modern blue = in optimum, soft red = above, soft yellow = below
        category_colors = ['#64B5F6', '#E57373', '#FFD54F']

        # Format data for bar plot
        results = {}
        for ix in df.index:
            results[ix] = df.loc[ix].to_list()

        year_labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)

        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())
        ax.margins(y=0.02)  # remove default ~5% padding above/below bar group

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(year_labels, widths, left=starts, height=0.65,
                            label=colname, color=color)

            r, g, b, *_ = mcolors.to_rgba(color)
            text_color = 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'
            for rect, width in zip(rects, widths):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + rect.get_height() / 2,
                        f"{width:.1f}",
                        ha='center', va='center_baseline',
                        color=text_color)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                  loc='lower left', fontsize='small')

        return ax

    def plot_bin_aggregates(self, ax,
                            xlabel: str = None,
                            ylabel: str = None,
                            xunit: str = None,
                            yunit: str = None):
        """Plot bin aggregates with smoothed curve overlaid and optimum range shaded."""
        bin_aggs_df = self.results_optrange['bin_aggs_df'].copy()
        rbin_aggs_df = self.results_optrange['rbin_aggs_df'].copy()
        xcol = self.results_optrange['xcol']
        ycol = self.results_optrange['ycol']
        n_xbins = self.results_optrange['n_xbins']
        optimum_start_bin = self.results_optrange['optimum_start_bin']
        optimum_end_bin = self.results_optrange['optimum_end_bin']
        optimum_xstart = self.results_optrange['optimum_xstart']
        optimum_xend = self.results_optrange['optimum_xend']
        roptimum_bin = self.results_optrange['roptimum_bin']

        _xlabel = xlabel if xlabel else (f"{xcol} [{xunit}]" if xunit else xcol)
        _ylabel = ylabel if ylabel else (f"{ycol} [{yunit}]" if yunit else ycol)

        # Integer positions for each bin, used as x-coordinates throughout
        n = len(bin_aggs_df)
        bin_positions = np.arange(n)

        # x-tick labels: bin midpoints as actual x values, thinned to ~10 ticks
        midpoints = np.array([iv.mid for iv in bin_aggs_df.index])
        tick_step = max(1, n // 10)
        ax.set_xticks(bin_positions[::tick_step])
        ax.set_xticklabels([f"{v:.1f}" for v in midpoints[::tick_step]])

        # y-axis limits across both raw and smoothed curves
        ymax = max(bin_aggs_df[ycol][self.bins_agg].max(), rbin_aggs_df[self.ragg].max())
        ymin = min(bin_aggs_df[ycol][self.bins_agg].min(), rbin_aggs_df[self.ragg].min())
        ax.set_ylim(ymin, ymax)

        # Raw bin aggregates
        ax.plot(bin_positions, bin_aggs_df[ycol][self.bins_agg].values,
                color='steelblue', alpha=0.35, linewidth=1, zorder=2,
                label=f"bin {self.bins_agg}")

        # Smoothed curve
        smooth_ixs = [bin_aggs_df.index.get_loc(b) for b in rbin_aggs_df.index]
        ax.plot(smooth_ixs, rbin_aggs_df[self.ragg].values,
                color='steelblue', linewidth=2.2, zorder=3, label=f"rolling {self.ragg}")

        # Three shaded regions — below / in / above optimum
        # Use 500-level Material colours: distinct from the top panel's 300-level shades
        _shade_alpha = 0.12
        optimum_start_bin_ix = bin_aggs_df.index.get_loc(optimum_start_bin)
        optimum_end_bin_ix = bin_aggs_df.index.get_loc(optimum_end_bin)

        ax.fill_between([bin_positions[0], optimum_start_bin_ix],
                        ymin, ymax, color='#FFC107', alpha=_shade_alpha, zorder=1)  # amber: below
        ax.fill_between([optimum_start_bin_ix, optimum_end_bin_ix],
                        ymin, ymax, color='#2196F3', alpha=_shade_alpha, zorder=1,
                        label=f"optimum range [{optimum_xstart:.2f}, {optimum_xend:.2f}]")  # blue: in optimum
        ax.fill_between([optimum_end_bin_ix, bin_positions[-1]],
                        ymin, ymax, color='#F44336', alpha=_shade_alpha, zorder=1)  # red: above

        # Boundary lines
        ax.axvline(optimum_start_bin_ix, color='#455A64', linewidth=1.0, linestyle='--', alpha=0.5)
        ax.axvline(optimum_end_bin_ix, color='#455A64', linewidth=1.0, linestyle='--', alpha=0.5)

        # Peak marker
        peak_ix = bin_aggs_df.index.get_loc(roptimum_bin)
        ax.axvline(peak_ix, color='#37474F', linewidth=1.5, linestyle=':', zorder=4,
                   label=f"peak ({self.define_optimum})")

        ax.set_xlabel(_xlabel)
        ax.set_ylabel(_ylabel)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4, zorder=0)
        ax.set_xlim(bin_positions[0], bin_positions[-1])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best', fontsize='small', framealpha=0.8)

    def plot_optimum_range_timeseries(self, ax):
        """Plot percentage of data in each category as a time series over years."""
        df = self.results_optrange['vals_in_optimum_range_df']
        years = df.index.astype(int)

        series = [
            ('vals_inoptimum_perc',    'in optimum (%)',    '#64B5F6'),
            ('vals_aboveoptimum_perc', 'above optimum (%)', '#E57373'),
            ('vals_belowoptimum_perc', 'below optimum (%)', '#FFD54F'),
        ]

        for col, label, color in series:
            vals = df[col].values
            ax.plot(years, vals, color=color, marker='o', markersize=5,
                    linewidth=2, label=label, zorder=3)
            ax.fill_between(years, vals, alpha=0.15, color=color, zorder=1)

        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=0)
        ax.set_ylabel('(%)')
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4, zorder=0)
        ax.legend(loc='upper right', fontsize='small', framealpha=0.8, ncol=3)

    def plot_rolling_bin_aggregates(self, ax):
        """Plot rolling mean of y medians in bins of x"""

        # Get data
        rbin_aggs_df = self.results_optrange['rbin_aggs_df'].copy()
        xcol = self.results_optrange['xcol']
        ycol = self.results_optrange['ycol']
        n_xbins = self.results_optrange['n_xbins']
        optimum_start_bin = self.results_optrange['optimum_start_bin']
        optimum_end_bin = self.results_optrange['optimum_end_bin']
        optimum_xstart = self.results_optrange['optimum_xstart']
        optimum_xend = self.results_optrange['optimum_xend']

        # Find min/max across dataframe, used for scaling yaxis
        rbin_aggs_df['mean+std'] = rbin_aggs_df['mean'].add(rbin_aggs_df['std'])
        rbin_aggs_df['mean-std'] = rbin_aggs_df['mean'].sub(rbin_aggs_df['std'])
        dfmax = rbin_aggs_df[['mean+std', 'mean-std']].max().max()
        dfmin = rbin_aggs_df.min().min()
        ax.set_ylim(dfmin, dfmax)

        # Show rolling mean
        rbin_aggs_df.plot(ax=ax, y='mean', yerr='std', zorder=99,
                          title=f"Rolling mean of {ycol} medians in {n_xbins} bins of {xcol}")

        # Show optimum range
        optimum_start_bin_ix = rbin_aggs_df.index.get_loc(optimum_start_bin)
        optimum_end_bin_ix = rbin_aggs_df.index.get_loc(optimum_end_bin)
        ax.axvline(optimum_start_bin_ix)
        ax.axvline(optimum_end_bin_ix)
        area_opr = ax.fill_between([optimum_start_bin_ix,
                                    optimum_end_bin_ix],
                                   dfmin, dfmax,
                                   color='#FFC107', alpha=0.5, zorder=1,
                                   label=f"optimum range {self.define_optimum} between {optimum_xstart} and {optimum_xend}")

        ax.legend(
            [area_opr],
            [area_opr.get_label()],
            scatterpoints=1,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            ncol=2)
