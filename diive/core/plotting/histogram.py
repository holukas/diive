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

    def __init__(self, s: Series, method, n_bins, ignore_fringe_bins,
                 highlight_peak: bool = True, xlabel: str = None, show_zscores: bool = True,
                 show_info: bool = True, show_counts: bool = True):

        self.s = s
        self.method = method
        self.n_bins = n_bins
        self.ignore_fringe_bins = ignore_fringe_bins
        self.highlight_peak = highlight_peak
        self.xlabel = xlabel
        self.show_zscores = show_zscores
        self.show_info = show_info
        self.show_counts = show_counts
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
        title = f"{self.s.name} (between {self.first_date} and {self.last_date})"
        self.ax.set_title(title, fontsize=24, weight='bold')
        xlabel = self.xlabel if self.xlabel else ""

        ix_max = self.counts.argmax()

        # Show counts for each bar
        if self.show_counts:
            self.ax.bar_label(bars)

        # Peak: highlight bin with most counts
        if self.highlight_peak:
            bars[ix_max].set_fc('#EF5350')

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
                self.ax.axvline(val, ls='--', color='#0D47A1')
                trans_ax = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)
                self.ax.text(val, 0.8, f"  {z}\n{val:.02f}",
                             size=16, color="#0D47A1", backgroundcolor='None', transform=trans_ax,
                             alpha=1, horizontalalignment='left', verticalalignment='top', zorder=999)

        default_format(ax=self.ax, ax_xlabel_txt=xlabel, ax_ylabel_txt="counts",
                       ticks_width=2, ticks_length=6, ticks_direction='in',
                       spines_lw=1)

        self.ax.locator_params(axis='both', nbins=20)
        self.fig.show()


def example():
    from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
    data_df, metadata_df = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
    series = data_df['FC'].copy()

    hist = HistogramPlot(
        s=series,
        method='n_bins',
        n_bins=20,
        ignore_fringe_bins=None,
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
    example()
