import math

import matplotlib.transforms as transforms
import pandas as pd

import diive.core.plotting.plotfuncs as pf
from diive.core.funcs.funcs import zscore, val_from_zscore
from diive.core.plotting.plotfuncs import default_format

# pd.options.display.width = None
# pd.options.display.max_columns = None
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

from pandas import Series


class HistogramPlot:

    def __init__(self, s: Series, method, n_bins: int or list = None, ignore_fringe_bins: list = False,
                 highlight_peak: bool = True, xlabel: str = None, show_zscores: bool = True,
                 show_zscore_values: bool = True, show_info: bool = True, show_counts: bool = True,
                 show_title: bool = True, show_grid: bool = True):

        self.s = s
        self.method = method
        self.n_bins = n_bins
        self.ignore_fringe_bins = ignore_fringe_bins
        self.highlight_peak = highlight_peak
        self.xlabel = xlabel
        self.show_zscores = show_zscores
        self.show_zscore_values = show_zscore_values
        self.show_info = show_info
        self.show_counts = show_counts
        self.show_title = show_title
        self.show_grid = show_grid
        self.first_date = s.index[0]
        self.last_date = s.index[-1]

        self.fig = None
        self.ax = None
        self.counts = None
        self.edges = None

    def get_fig(self):
        return self.fig

    def get_ax(self):
        return self.ax

    def plot(self, ax=None):

        # Setup
        self.ax = ax
        self.fig, self.ax, showplot = pf.setup_figax(ax=self.ax, figsize=(16, 9))

        # Plot histogram
        self.counts, self.edges, bars = self.ax.hist(
            x=self.s,
            bins=self.n_bins,
            rwidth=0.95,
            color="#78909c"
        )
        self.ax.set_xticks(self.edges)

        if self.show_title:
            title = f"{self.s.name} (between {self.first_date} and {self.last_date})"
            ypos = 1
            ypos = 1.05 if self.show_zscores and not self.show_zscore_values else ypos
            ypos = 1.08 if self.show_zscores and self.show_zscore_values else ypos
            self.ax.set_title(title, fontsize=24, weight='bold', y=ypos)

        xlabel = self.xlabel if self.xlabel else ""

        ix_max = self.counts.argmax()

        # Show counts for each bar
        if self.show_counts:
            self.ax.bar_label(bars)

        # Peak: highlight bin with most counts
        if self.highlight_peak:
            bars[ix_max].set_fc('#FFA726')

        if self.show_info:
            info_txt = f"method: {self.method}"
            info_txt += f"\nn_bins: {self.n_bins}" if self.method == 'n_bins' else info_txt
            if self.highlight_peak:
                info_txt += f"\nPEAK between {self.edges[ix_max]:.02f} and {self.edges[ix_max + 1]:.02f}" if self.method == 'n_bins' else info_txt

            self.ax.text(0.05, 0.95, info_txt,
                         size=16, color="black", backgroundcolor='None', transform=self.ax.transAxes,
                         alpha=1, horizontalalignment='left', verticalalignment='top', zorder=999)

        # z-scores
        if self.show_zscores:
            zscores = zscore(series=self.s, absolute=False)
            for z in range(int(math.floor(zscores.min())), int(math.ceil(zscores.max()))):
                val = val_from_zscore(series=self.s, zscore=z)
                self.ax.axvline(val, ls=':', color='#AB47BC', alpha=.9)
                trans_ax = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)
                if self.show_zscore_values:
                    self.ax.text(val, 1.07, f"{z}\n{val:.02f}",
                                 size=16, color="#AB47BC", backgroundcolor='None', transform=trans_ax,
                                 alpha=1, horizontalalignment='center', verticalalignment='top', zorder=999)
                else:
                    self.ax.text(val, 1.04, f"{z}",
                                 size=16, color="#AB47BC", backgroundcolor='None', transform=trans_ax,
                                 alpha=1, horizontalalignment='center', verticalalignment='top', zorder=999)

        default_format(ax=self.ax, ax_xlabel_txt=xlabel, ax_ylabel_txt="counts",
                       ticks_width=2, ticks_length=6, ticks_direction='in',
                       spines_lw=1, showgrid=self.show_grid)

        self.ax.locator_params(axis='both', nbins=10)

        if showplot:
            self.fig.show()


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    data_df = load_exampledata_parquet()
    series = data_df['NEE_CUT_REF_f'].copy()

    hist = HistogramPlot(
        s=series,
        method='n_bins',
        n_bins=20,
        xlabel='flux',
        highlight_peak=True,
        show_zscores=True,
        show_zscore_values=True,
        show_info=True
        # ignore_fringe_bins=[1, 1]
    )
    hist.plot()
    # hist.results
    # hist.peakbins


def example_per_year():
    from diive.configs.exampledata import load_exampledata_parquet
    data_df = load_exampledata_parquet()
    years = data_df.index.year.unique()

    for y in years:
        series = data_df.loc[data_df.index.year == y, 'NEE_CUT_REF_f'].copy()
        hist = HistogramPlot(
            s=series,
            method='n_bins',
            n_bins=50,
            xlabel='flux',
            highlight_peak=True,
            show_zscores=True,
            show_info=True
            # ignore_fringe_bins=[1, 1]
        )
        hist.plot()
        # hist.results
        # hist.peakbins


if __name__ == '__main__':
    # example_per_year()
    example()
