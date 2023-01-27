"""
FLUX: OPTIMUM RANGE
===================
"""

from math import isclose
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from pandas import DataFrame

from diive.core.plotting.plotfuncs import save_fig


class FindOptimumRange:

    def __init__(self,
                 df: DataFrame,
                 xcoF: str,
                 ycoF: str,
                 n_vals_per_bin: int = 300,
                 bins_agg: Literal['median'] = 'median',
                 rwinsize: float = 0.1,
                 ragg: Literal['mean'] = 'mean',
                 define_optimum: Literal['min', 'max'] = 'max'):
        """
        Find x range for optimum y

        First, x data are aggregated in y bins. By default, the median
        value of x is calculated for each y bin (*bins_agg*). The number
        of bins that is used is defined by total length of data divided
        by *n_vals_per_bin*, i.e., each bin should contain e.g. 300 values.
        Then, the rolling mean (*ragg*) with window size *rwinsize* is
        calculated across all binned values. Here, *rwinsize* is given as
        the fraction of the total number of detected bins. The optimum is
        detected as the maximum (or other, *define_optimum*) of the values
        found in the rolling aggregation.

        Example: VPD (x) range where NEE (y) carbon uptake is highest (=smallest number)

        Args:
            df: Data
            xcoF: Column name of x in df
            ycoF: Column name of y in df
            n_vals_per_bin: Number of values per x bin
            bins_agg: How data in bins are aggregated
            rwinsize: Window size for rolling aggregation, expressed as fraction of
                the total number of bins. The total number of bins is calculated
                from the total length of the data and *n_vals_per_bin*. The resulting
                window size is then an integer value that is used in further calculations.
                If the integer window size results in an even number, +1 is added since
                the window size must be an odd number.
            ragg: Rolling aggregation that is used in the rolling window.
            define_optimum: Optimum can be based on 'min' or 'max'
        """
        self.df = df[[xcol, ycol]].copy()
        self.xcol = xcol
        self.ycol = ycol
        self.n_vals_per_bin = n_vals_per_bin
        self.bins_agg = bins_agg
        self.rwinsize = rwinsize
        self.ragg = ragg
        self.define_optimum = define_optimum

        self._results_optrange = {}

    @property
    def results_optrange(self) -> dict:
        """Return optimum range results"""
        if not self._results_optrange:
            raise Exception('Results for optimum range are empty')
        return self._results_optrange

    def find_optimum(self):
        # self._prepare_data() todo?

        bins_df, bin_aggs_df, n_xbins = self._divide_xdata_into_bins()

        winsize = int(n_xbins * self.rwinsize)
        winsize = winsize + 1 if (winsize % 2 == 0) else winsize  # Must be odd number
        rbin_aggs_df = self._rolling_agg(bin_aggs_df=bin_aggs_df,
                                         use_bin_agg=self.bins_agg,
                                         rolling_agg=self.ragg,
                                         winsize=winsize)

        roptimum_bin, roptimum_val = self._find_rolling_optimum(rolling_df=rbin_aggs_df,
                                                                use_rolling_agg=self.ragg)

        # rwinsize = int(num_xbins / 5)  # Window size for rolling aggs

        optimum_xstart, optimum_xend, optimum_ymean, \
        optimum_start_bin, optimum_end_bin = self._get_optimum_range(grouped_df=bin_aggs_df,
                                                                     roptimum_bin=roptimum_bin,
                                                                     winsize=winsize)
        self._validate(roptimum_val=roptimum_val, optimum_ymean=optimum_ymean)

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
            n_xbins=n_xbins,
            xcol=self.xcol,
            ycol=self.ycol,
            vals_in_optimum_range_df=vals_in_optimum_range_df
        )

    def _values_in_optimum_range(self, optimum_xstart: float, optimum_xend: float) -> pd.DataFrame:
        df = self.df[[self.xcol, self.ycol]].copy()

        # Full data range
        fullrange_df = df.groupby(df.index.year).agg({self.xcoF: ['count', 'mean']})

        xcounts_df = pd.DataFrame()
        # xcounts_df['vals_total'] = df.groupby(df.index.year).agg({'count'})
        xcounts_df['vals_total'] = \
            df.groupby(df.index.year).agg(vals_total=(self.xcol, 'count'))

        # Data in optimum
        _filter = (df[self.xcol] > optimum_xstart) & (df[self.xcol] <= optimum_xend)
        xcounts_df['vals_inoptimum'] = \
            df.loc[_filter].groupby(df.loc[_filter].index.year).agg(vals_inoptimum=(self.xcol, 'count'))

        # Above optimum
        _filter = (df[self.xcol] > optimum_xend)
        xcounts_df['vals_aboveoptimum'] = \
            df.loc[_filter].groupby(df.loc[_filter].index.year).agg(vals_aboveoptimum=(self.xcol, 'count'))

        # Below optimum
        _filter = (df[self.xcol] <= optimum_xstart)
        xcounts_df['vals_belowoptimum'] = \
            df.loc[_filter].groupby(df.loc[_filter].index.year).agg(vals_belowoptimum=(self.xcol, 'count'))

        # Percentages
        xcounts_df['vals_inoptimum_perc'] = xcounts_df['vals_inoptimum'].div(xcounts_df['vals_total']).multiply(100)
        xcounts_df['vals_aboveoptimum_perc'] = xcounts_df['vals_aboveoptimum'].div(xcounts_df['vals_total']).multiply(
            100)
        xcounts_df['vals_belowoptimum_perc'] = xcounts_df['vals_belowoptimum'].div(xcounts_df['vals_total']).multiply(
            100)

        # NaNs correspond to zero,
        # e.g. if no values above optimum are found
        xcounts_df = xcounts_df.fillna(0)

        return xcounts_df

    def _prepare_data(self):
        # Keep x values > 0
        self.df = self.df.loc[self.df[self.xcol] > 0, :]

    def _divide_xdata_into_bins(self) -> tuple[DataFrame, DataFrame, int]:
        """
        Divide x data into bins

        Column w/ bin membership is added to data

        Args:
            n_xbins: number of bins

        """
        bins_df = self.df.copy()

        # Detect number of x bins
        n_xbins = int(len(bins_df) / self.n_vals_per_bin)

        # Divide data into bins and add as column
        xbins = pd.qcut(bins_df[self.xcol], n_xbins, duplicates='drop')  # How awesome!
        bins_df = bins_df.assign(xbins=xbins)

        # Aggregate by bin membership
        bin_aggs_df = bins_df.groupby('xbins').agg({self.bins_agg, 'count'})

        return bins_df, bin_aggs_df, n_xbins

    def _rolling_agg(self, bin_aggs_df, use_bin_agg, winsize, rolling_agg):
        rolling_df = bin_aggs_df[self.ycol][use_bin_agg].rolling(winsize, center=True)
        return rolling_df.agg({rolling_agg, 'std'}).dropna()

    def _find_rolling_optimum(self, rolling_df: DataFrame, use_rolling_agg: str = 'mean'):
        """Find optimum bin in rolling data
        The rolling data is scanned for the bin with the highest or lowest value.
        """
        # Find bin with rolling mean min or max (e.g. max carbon uptake = minimum NEE value)
        roptimum_bin = None  # Index given as bin interval
        roptimum_val = None  # Value at bin interval
        if self.define_optimum == 'min':
            roptimum_bin = rolling_df[use_rolling_agg].idxmin()
            roptimum_val = rolling_df[use_rolling_agg][roptimum_bin]
        elif self.define_optimum == 'max':
            roptimum_bin = rolling_df[use_rolling_agg].idxmax()
            roptimum_val = rolling_df[use_rolling_agg].iloc[roptimum_bin]
        print(f"Optimum {self.define_optimum} found in class: {roptimum_bin}  /  value: {roptimum_val}")
        return roptimum_bin, roptimum_val

    def _get_optimum_range(self, grouped_df: DataFrame, roptimum_bin: pd.IntervalIndex, winsize: int):
        """Get data range (start and end) that was used to calculate rolling optimum"""

        # Find integer location of bin where rolling optimum value (y min or y max) was found
        int_loc = grouped_df.index.get_loc(roptimum_bin)
        print(f"Index integer location of found optimum: {int_loc}  /  {grouped_df.index[int_loc]}")

        # Get data range start and end
        roptimum_start_ix = int_loc - (int(winsize / 2))
        roptimum_end_ix = int_loc + (int(winsize / 2) + 1)  # was +1 b/c end of range not included in slicing

        # Optimum end index cannot be larger than available indices
        roptimum_end_ix = len(grouped_df) - 1 if roptimum_end_ix > len(grouped_df) - 1 else roptimum_end_ix

        # Optimum start index cannot be smaller than the first available index 0
        roptimum_start_ix = 0 if roptimum_start_ix < 0 else roptimum_start_ix

        # Get data range indices
        optimum_start_bin = grouped_df.iloc[roptimum_start_ix].name
        optimum_end_bin = grouped_df.iloc[roptimum_end_ix].name

        optimum_range_xstart = optimum_start_bin.left
        optimum_range_xend = optimum_end_bin.right
        optimum_range_ymean = grouped_df[self.ycol]['median'].iloc[roptimum_start_ix:roptimum_end_ix].mean()
        return optimum_range_xstart, optimum_range_xend, optimum_range_ymean, \
               optimum_start_bin, optimum_end_bin

    def _validate(self, roptimum_val, optimum_ymean):
        check = isclose(roptimum_val, optimum_ymean, abs_tol=10 ** -3)
        if check:
            print("Validation OK.")
        else:
            print("(!)Validation FAILED.")
            assert isclose(roptimum_val, optimum_ymean)

    def showfig(self,
                saveplot: bool = False,
                title: str = None,
                path: Path or str = None):
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(4, 1)  # rows, cols
        gs.update(wspace=.2, hspace=.5, left=.05, right=.95, top=.95, bottom=.05)
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax2 = fig.add_subplot(gs[2, 0])
        ax3 = fig.add_subplot(gs[3, 0])
        ax = self.plot_vals_in_optimum_range(ax=ax1)
        ax = self.plot_bin_aggregates(ax=ax2)
        ax = self.plot_rolling_bin_aggregates(ax=ax3)
        fig.show()
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
        # xcol = results_optrange['xcol']
        # ycol = results_optrange['ycol']

        # Names of categories, shown in legend above plot
        category_names = ['values in optimum range (%)', 'above optimum range (%)', 'below optimum range (%)']
        # category_names = ['vals_inoptimum_perc', 'vals_aboveoptimum_perc', 'vals_belowoptimum_perc']

        # Format data for bar plot
        results = {}
        for ix, row in df.iterrows():
            results[ix] = df.loc[ix].to_list()

        year_labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.colormaps['RdYlBu_r'](np.linspace(0.20, 0.80, data.shape[1]))

        # fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(year_labels, widths, left=starts, height=0.9,
                            label=colname, color=color)

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            ax.bar_label(rects, label_type='center', color=text_color)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                  loc='lower left', fontsize='small')

        # default_format(ax=ax, txt_xlabel="year", txt_ylabel=f'counts',
        #                              txt_ylabel_units='[#]')
        # default_grid(ax=ax)

        return ax

    def plot_bin_aggregates(self, ax):
        """Plot y median in bins of x"""

        # Get data
        bin_aggs_df = self.results_optrange['bin_aggs_df'].copy()
        xcol = self.results_optrange['xcol']
        ycol = self.results_optrange['ycol']
        n_xbins = self.results_optrange['n_xbins']
        optimum_start_bin = self.results_optrange['optimum_start_bin']
        optimum_end_bin = self.results_optrange['optimum_end_bin']
        optimum_xstart = self.results_optrange['optimum_xstart']
        optimum_xend = self.results_optrange['optimum_xend']

        # Find min/max of y, used for scaling yaxis
        ymax = bin_aggs_df[ycol]['median'].max()
        ymin = bin_aggs_df[ycol]['median'].min()
        ax.set_ylim(ymin, ymax)

        # Show rolling mean
        bin_aggs_df[ycol]['median'].plot(ax=ax, zorder=99,
                                         title=f"{ycol} medians in {n_xbins} bins of {xcol}")

        # Show optimum range
        optimum_start_bin_ix = bin_aggs_df.index.get_loc(optimum_start_bin)
        optimum_end_bin_ix = bin_aggs_df.index.get_loc(optimum_end_bin)
        ax.axvline(optimum_start_bin_ix)
        ax.axvline(optimum_end_bin_ix)
        area_opr = ax.fill_between([optimum_start_bin_ix,
                                    optimum_end_bin_ix],
                                   ymin, ymax,
                                   color='#FFC107', alpha=0.5, zorder=1,
                                   label=f"optimum range {self.define_optimum} between {optimum_xstart} and {optimum_xend}")

        l = ax.legend(
            [area_opr],
            [area_opr.get_label()],
            scatterpoints=1,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            ncol=2)

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

        l = ax.legend(
            [area_opr],
            [area_opr.get_label()],
            scatterpoints=1,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            ncol=2)


def example():
    pd.options.display.width = None
    pd.options.display.max_columns = None
    pd.set_option('display.max_rows', 3000)
    pd.set_option('display.max_columns', 3000)

    # Test data
    from diive.core.io.files import load_pickle
    df_orig = load_pickle(
        filepath=r"L:\Dropbox\luhk_work\20 - CODING\26 - NOTEBOOKS\GL-NOTEBOOKS\_data\ch-dav\CH-DAV_FP2022.1_1997-2022.08_ID20220826234456_30MIN.diive.csv.pickle")

    # # Check columns
    # import fnmatch
    # [print(col) for col in alldata_df.columns if any(fnmatch.fnmatch(col, ids) for ids in ['NEE_CUT_50*'])]

    # Select daytime data between May and September
    df = df_orig.copy()
    df = df.loc[(df.index.month >= 5) & (df.index.month <= 9)]
    df = df.loc[df['PotRad_CUT_REF'] > 20]

    # Optimum range
    optrange = FindOptimumRange(df=df, xcol='RH', ycol='NEE_CUT_REF_f', define_optimum="min", rwinsize=0.3)
    optrange.find_optimum()
    optrange.plot_results()


if __name__ == '__main__':
    example()
