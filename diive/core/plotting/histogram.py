import math
import warnings

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
    """Histogram plot with optional z-score overlay and peak highlighting.

    Visualize data distribution with optional z-score overlay and peak bin highlighting.

    Args:
        series: Series to plot
        method: Binning method (e.g., 'n_bins')
        n_bins: Number of bins for histogram (int or list)
        ignore_fringe_bins: Whether to ignore fringe bins

    Call `plot()` to render with styling options (title, labels, display options).

    See Also:
        examples/visualization/plot_histogram_basic.py — Histogram variations with z-score overlays
    """

    def __init__(self, series: Series = None, method=None, n_bins: int or list = None,
                 ignore_fringe_bins: list = False, s: Series = None):

        # `s` is the deprecated name for `series` (renamed for consistency with
        # the other plotting classes, which all take `series`).
        if s is not None:
            warnings.warn("HistogramPlot: the `s` argument is deprecated, use `series` instead.",
                          DeprecationWarning, stacklevel=2)
            series = s if series is None else series
        if series is None:
            raise ValueError("HistogramPlot requires `series`.")

        self.series = series
        self.s = series  # internal alias retained for backwards compatibility
        self.method = method
        self.n_bins = n_bins
        self.ignore_fringe_bins = ignore_fringe_bins
        self.first_date = series.index[0]
        self.last_date = series.index[-1]

        self.fig = None
        self.ax = None
        self.counts = None
        self.edges = None

    def get_fig(self):
        return self.fig

    def get_ax(self):
        return self.ax

    def plot(self, ax=None, xlabel: str = None, title: str = None, highlight_peak: bool = True,
             show_zscores: bool = True, show_zscore_values: bool = True, show_info: bool = True,
             show_counts: bool = True, show_title: bool = True, show_grid: bool = True):
        """Generate histogram plot with optional styling.

        Args:
            ax: Matplotlib axes (creates new if None)
            xlabel: X-axis label (default: empty)
            title: Plot title (default: "{series.name} (between {start_date} and {end_date})")
            highlight_peak: Highlight the bin with most counts (default: True)
            show_zscores: Show z-score overlay on top axis (default: True)
            show_zscore_values: Display z-score values and corresponding data values (default: True)
            show_info: Show method and peak information text (default: True)
            show_counts: Show count labels on each bar (default: True)
            show_title: Display title (default: True)
            show_grid: Display gridlines (default: True)
        """
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

        if show_title:
            plot_title = title if title else f"{self.s.name} (between {self.first_date} and {self.last_date})"
            self.ax.set_title(plot_title, fontsize=24, weight='bold')

        xlabel_text = xlabel if xlabel else ""

        ix_max = self.counts.argmax()

        # Show counts for each bar
        if show_counts:
            self.ax.bar_label(bars)

        # Peak: highlight bin with most counts
        if highlight_peak:
            bars[ix_max].set_fc('#FFA726')

        if show_info:
            info_txt = f"method: {self.method}"
            info_txt += f"\nn_bins: {self.n_bins}" if self.method == 'n_bins' else info_txt
            if highlight_peak:
                info_txt += f"\nPEAK between {self.edges[ix_max]:.02f} and {self.edges[ix_max + 1]:.02f}" if self.method == 'n_bins' else info_txt

            self.ax.text(0.05, 0.95, info_txt,
                         size=16, color="black", backgroundcolor='None', transform=self.ax.transAxes,
                         alpha=1, horizontalalignment='left', verticalalignment='top', zorder=999)

        # z-scores
        if show_zscores:
            zscores = zscore(series=self.s, absolute=False)
            self.axx = self.ax.twiny()
            self.axx.set_xlim(self.ax.get_xlim()[0], self.ax.get_xlim()[1])
            self.axx.grid(False)
            self.axx.xaxis.set_label_position('top')
            axx_zscores = []
            axx_ticks_pos = []
            for z in range(int(math.floor(zscores.min())), int(math.ceil(zscores.max()))):
                val = val_from_zscore(series=self.s, zscore=z)
                self.axx.axvline(val, ls=':', color='#AB47BC', alpha=.9)
                # self.ax.axvline(val, ls=':', color='#AB47BC', alpha=.9)
                # trans_ax = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)
                # if self.show_zscore_values:
                #     self.ax.text(val, 1.07, f"{z}\n{val:.02f}",
                #                  size=16, color="#AB47BC", backgroundcolor='None', transform=trans_ax,
                #                  alpha=1, horizontalalignment='center', verticalalignment='top', zorder=999)
                # else:
                #     self.ax.text(val, 1.04, f"{z}",
                #                  size=16, color="#AB47BC", backgroundcolor='None', transform=trans_ax,
                #                  alpha=1, horizontalalignment='center', verticalalignment='top', zorder=999)
                axx_zscores.append(z)
                axx_ticks_pos.append(val)
            self.axx.set_xticks(axx_ticks_pos)
            if show_zscore_values:
                axx_zscores = [f"{z}\n{v:.01f}" for z, v in zip(axx_zscores, axx_ticks_pos)]
                self.axx.set_xticklabels(axx_zscores)
            else:
                self.axx.set_xticklabels(axx_zscores)
            # self.axx.set_xlabel(color='#AB47BC', fontsize=20)
            self.axx.tick_params(axis='x', colors='#AB47BC', labelsize=16)
            self.axx.set_xlabel("z-score", color='#AB47BC', fontsize=16)

        default_format(ax=self.ax, ax_xlabel_txt=xlabel_text, ax_ylabel_txt="counts",
                       ticks_width=2, ticks_length=6, ticks_direction='in',
                       spines_lw=1, showgrid=show_grid)

        self.ax.locator_params(axis='both', nbins=10)

        if showplot:
            self.fig.show()
