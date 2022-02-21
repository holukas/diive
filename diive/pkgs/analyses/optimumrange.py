"""
FLUX: OPTIMUM RANGE
===================
"""

import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from pandas import DataFrame
from diive.core.plotting.plotfuncs import default_format, default_grid, default_legend
import numpy as np
import matplotlib.pyplot as plt

class FindOptimumRange:
    """
    Find x range for optimum y
    Example: VPD (x) range where NEE (y) carbon uptake is highest (=smallest number)
    """

    def __init__(self, df: DataFrame, xcol: str, ycol: str, define_optimum: str = 'max'):
        """
        :param df: data
        :param xcol: column name of x in df
        :param ycol: column name of y in df
        :param define_optimum: optimum can be based on 'min' or 'max'
        """
        self.df = df[[xcol, ycol]].copy()
        self.xcol = xcol
        self.ycol = ycol
        self.define_optimum = define_optimum

        self._results_optrange = {}

    def results_optrange(self):
        """Return optimum range results"""
        if not self._results_optrange:
            raise Exception('Results for optimum range are empty')
        return self._results_optrange

    def find_optimum(self):
        self._prepare_data()
        num_xbins = self._define_xbins()
        rwinsize = int(num_xbins / 5)  # Window size for rolling aggs
        bin_aggs_df = self._aggregate(bin_agg='median')
        rbin_aggs_df = self._rolling_agg(bin_aggs_df=bin_aggs_df,
                                         winsize=rwinsize,
                                         use_bin_agg='median',
                                         rolling_agg='mean')
        roptimum_bin, roptimum_val = self._find_rolling_optimum(rolling_df=rbin_aggs_df,
                                                                use_rolling_agg='mean')
        optimum_xstart, optimum_xend, optimum_ymean, \
        optimum_start_bin, optimum_end_bin = self._get_optimum_range(grouped_df=bin_aggs_df,
                                                                     roptimum_bin=roptimum_bin,
                                                                     winsize=rwinsize)
        self._validate(roptimum_val=roptimum_val, optimum_ymean=optimum_ymean)

        vals_in_optimum_range_df=\
            self._values_in_optimum_range(optimum_xstart=optimum_xstart, optimum_xend=optimum_xend)

        self._results_optrange = dict(
            optimum_xstart=optimum_xstart,
            optimum_xend=optimum_xend,
            optimum_ymean=optimum_ymean,
            optimum_start_bin=optimum_start_bin,
            optimum_end_bin=optimum_end_bin,
            bin_aggs_df=bin_aggs_df,
            rbin_aggs_df=rbin_aggs_df,
            rwinsize=rwinsize,
            roptimum_bin=roptimum_bin,
            roptimum_val=roptimum_val,
            num_xbins=num_xbins,
            xcol=self.xcol,
            ycol=self.ycol,
            vals_in_optimum_range_df=vals_in_optimum_range_df
        )

    def _values_in_optimum_range(self, optimum_xstart: float, optimum_xend: float) -> pd.DataFrame:
        df = self.df[[self.xcol, self.ycol]].copy()

        # Full data range
        fullrange_df = df.groupby(df.index.year).agg({self.xcol: ['count', 'mean']})

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

        return xcounts_df


    def _prepare_data(self):
        # Keep x values > 0
        self.df = self.df.loc[self.df[self.xcol] > 0, :]

    def _define_xbins(self):
        """ Divide x data into bins
        Number of bins is calculated from number of data rows
        Column w/ bin membership is added to data
        """
        num_xbins = int(len(self.df) / 100)  # Number of x bins
        xbins = pd.qcut(self.df[self.xcol], num_xbins)  # How awesome!
        self.df = self.df.assign(xbins=xbins)
        return num_xbins

    def _aggregate(self, bin_agg: str = 'median'):
        """Aggregate by bin membership"""
        return self.df.groupby('xbins').agg({bin_agg, 'count'})

    def _rolling_agg(self, bin_aggs_df, winsize, use_bin_agg, rolling_agg):
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
        print(f"Max uptake found in VPD class: {roptimum_bin}  /  value: {roptimum_val}")
        return roptimum_bin, roptimum_val

    def _get_optimum_range(self, grouped_df: DataFrame, roptimum_bin: pd.IntervalIndex, winsize: int):
        """Get data range (start and end) that was used to calculate rolling optimum"""

        # Find integer location of bin where rolling optimum value (y min or y max) was found
        int_loc = grouped_df.index.get_loc(roptimum_bin)
        print(f"Index integer location of rolling max uptake: {int_loc}  /  {grouped_df.index[int_loc]}")

        # Get data range start and end
        roptimum_start_ix = int_loc - (int(winsize / 2))
        roptimum_end_ix = int_loc + (int(winsize / 2) + 1)  # +1 b/c end of range not included in slicing

        # Get data range indices
        optimum_start_bin = grouped_df.iloc[roptimum_start_ix].name
        optimum_end_bin = grouped_df.iloc[roptimum_end_ix].name

        optimum_range_xstart = optimum_start_bin.left
        optimum_range_xend = optimum_end_bin.right
        optimum_range_ymean = grouped_df[self.ycol]['median'].iloc[roptimum_start_ix:roptimum_end_ix].mean()
        return optimum_range_xstart, optimum_range_xend, optimum_range_ymean, \
               optimum_start_bin, optimum_end_bin

    def _validate(self, roptimum_val, optimum_ymean):
        if roptimum_val == optimum_ymean:
            print("Validation OK.")

        # rolling_range_mean = grouped_df.iloc[roptimum_start_ix:roptimum_end_ix].mean()
        # print(f"\nStats of range of rolling max uptake:\n{rolling_range_mean}")

        # int_loc = rolling_df['mean'].index.get_loc(roptimum_ix)
        # print(f"Index integer location of rolling max uptake: {int_loc}  /  {rolling_df['mean'].index[int_loc]}")
        # max_class = mean.idxmin()
        # max_uptake = mean[mean.idxmin()]

        # print(f"\nRange of rolling max uptake: from {rolling_start_bin} to {rolling_end_bin}")

        # grouped_df[ycol]['median'].plot(figsize=(10, 5))
        # plt.axvline(int_loc)
        # plt.axvline(roptimum_start_ix)
        # plt.axvline(roptimum_end_ix)
        # plt.show()

        # optimum_df=optimum_df.set_index(xcol).sort_index()
        # optimum_df.rolling(window=100).mean().plot()
        # plt.show()

def plot_vals_in_optimum_range(ax, results_optrange: dict):
    """Plot optimum range: values in, above and below optimum per year"""

    # kudos: https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py

    # Get data
    df = results_optrange['vals_in_optimum_range_df'].copy()
    plotcols = ['vals_inoptimum_perc', 'vals_aboveoptimum_perc', 'vals_belowoptimum_perc']
    df = df[plotcols]
    xcol = results_optrange['xcol']
    ycol = results_optrange['ycol']

    # Names of categories, shown in legend above plot
    category_names = ['values in optimum range (%)', 'above optimum range (%)', 'below optimum range (%)']
    # category_names = ['vals_inoptimum_perc', 'vals_aboveoptimum_perc', 'vals_belowoptimum_perc']

    # Format data for bar plot
    results = {}
    for ix, row in df.iterrows():
        results[ix]=df.loc[ix].to_list()

    year_labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn_r'](np.linspace(0.15, 0.85, data.shape[1]))

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


def plot_bin_aggregates(ax, results_optrange: dict):
    """Plot y median in bins of x"""

    # Get data
    bin_aggs_df = results_optrange['bin_aggs_df'].copy()
    xcol = results_optrange['xcol']
    ycol = results_optrange['ycol']
    num_xbins = results_optrange['num_xbins']
    optimum_start_bin = results_optrange['optimum_start_bin']
    optimum_end_bin = results_optrange['optimum_end_bin']
    optimum_xstart = results_optrange['optimum_xstart']
    optimum_xend = results_optrange['optimum_xend']

    # Find min/max of y, used for scaling yaxis
    ymax = bin_aggs_df[ycol]['median'].max()
    ymin = bin_aggs_df[ycol]['median'].min()
    ax.set_ylim(ymin, ymax)

    # Show rolling mean
    bin_aggs_df[ycol]['median'].plot(ax=ax, zorder=99,
                                     title=f"{ycol} medians in {num_xbins} bins of {xcol}")

    # Show optimum range
    optimum_start_bin_ix = bin_aggs_df.index.get_loc(optimum_start_bin)
    optimum_end_bin_ix = bin_aggs_df.index.get_loc(optimum_end_bin)
    ax.axvline(optimum_start_bin_ix)
    ax.axvline(optimum_end_bin_ix)
    area_opr = ax.fill_between([optimum_start_bin_ix,
                                optimum_end_bin_ix],
                               ymin, ymax,
                               color='#FFC107', alpha=0.5, zorder=1,
                               label=f"optimum range between {optimum_xstart} and {optimum_xend}")

    l = ax.legend(
        [area_opr],
        [area_opr.get_label()],
        scatterpoints=1,
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        ncol=2)


def plot_rolling_bin_aggregates(ax, results_optrange: dict):
    """Plot rolling mean of y medians in bins of x"""

    # Get data
    rbin_aggs_df = results_optrange['rbin_aggs_df'].copy()
    xcol = results_optrange['xcol']
    ycol = results_optrange['ycol']
    num_xbins = results_optrange['num_xbins']
    optimum_start_bin = results_optrange['optimum_start_bin']
    optimum_end_bin = results_optrange['optimum_end_bin']
    optimum_xstart = results_optrange['optimum_xstart']
    optimum_xend = results_optrange['optimum_xend']

    # Find min/max across dataframe, used for scaling yaxis
    rbin_aggs_df['mean+std'] = rbin_aggs_df['mean'].add(rbin_aggs_df['std'])
    rbin_aggs_df['mean-std'] = rbin_aggs_df['mean'].sub(rbin_aggs_df['std'])
    dfmax = rbin_aggs_df[['mean+std', 'mean-std']].max().max()
    dfmin = rbin_aggs_df.min().min()
    ax.set_ylim(dfmin, dfmax)

    # Show rolling mean
    rbin_aggs_df.plot(ax=ax, y='mean', yerr='std', zorder=99,
                      title=f"Rolling mean of {ycol} medians in {num_xbins} bins of {xcol}")

    # Show optimum range
    optimum_start_bin_ix = rbin_aggs_df.index.get_loc(optimum_start_bin)
    optimum_end_bin_ix = rbin_aggs_df.index.get_loc(optimum_end_bin)
    ax.axvline(optimum_start_bin_ix)
    ax.axvline(optimum_end_bin_ix)
    area_opr = ax.fill_between([optimum_start_bin_ix,
                                optimum_end_bin_ix],
                               dfmin, dfmax,
                               color='#FFC107', alpha=0.5, zorder=1,
                               label=f"optimum range between {optimum_xstart} and {optimum_xend}")

    l = ax.legend(
        [area_opr],
        [area_opr.get_label()],
        scatterpoints=1,
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        ncol=2)

# def _testfig():
#     import matplotlib.gridspec as gridspec
#     import matplotlib.pyplot as plt
#     fig = plt.figure(figsize=(9, 9))
#     gs = gridspec.GridSpec(1, 1)  # rows, cols
#     # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
#     ax = fig.add_subplot(gs[0, 0])
#     fig.show()