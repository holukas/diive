"""
PLOTTING: HISTOGRAM
===================

Histogram plot with optional z-score overlay and peak-bin highlighting.

Part of the diive library: https://github.com/holukas/diive
"""
import math
import warnings

import numpy as np
import pandas as pd
from matplotlib import rcParams

import diive.core.plotting.plotfuncs as pf
from diive.core.funcs.funcs import zscore, val_from_zscore
from diive.core.plotting.styles.format import FormatStyle

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
        n_bins: Number of bins for histogram. An int (or None for the matplotlib
            default) lets the edges follow this series' own range; a sequence is
            used as the explicit bin edges. Passing the same sequence to two
            histograms is how a shared bin grid is pinned, so that bin *i* means
            the same interval in both (values outside the sequence are not binned).
        ignore_fringe_bins: List of two integers ``[i, j]``: the first *i* and the
            last *j* bins are dropped from the plot. The edges are derived from the
            full series first, so the bins that remain are the same intervals the
            untrimmed histogram would have shown, and records falling into a dropped
            bin are not plotted. Default False (keep all bins).

    Call `plot()` to render with styling options (title, labels, display options).

    See Also:
        examples/visualization/plot_histogram_basic.py — Histogram variations with z-score overlays
    """

    def __init__(self, series: Series = None, method=None, n_bins: int or list = None,
                 ignore_fringe_bins: list = False, s: Series = None):
        """Set up the histogram. See the class docstring for parameters (``s`` is a deprecated alias for ``series``)."""

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

    def _trimmed_edges(self):
        """Return the bin edges with the fringe bins removed.

        Same semantics as :meth:`diive.analysis.Histogram._ignore_fringe_bins`:
        the edges come from the *full* series, then the first `i` and last `j`
        bins are dropped, so the bins that survive are the ones the untrimmed
        histogram would have shown. Re-binning on the trimmed edge array (rather
        than slicing the counts afterwards) keeps one `ax.hist` call, which is
        what draws the bars and their labels.
        """
        n_first, n_last = self.ignore_fringe_bins[0], self.ignore_fringe_bins[1]
        # `bins=None` is matplotlib's "use the rcParam", which numpy does not know.
        bins = self.n_bins if self.n_bins is not None else rcParams['hist.bins']
        edges = np.histogram_bin_edges(self.s.dropna().to_numpy(), bins=bins)
        # len()-based stop so that n_last == 0 trims nothing from the end.
        kept = edges[n_first:len(edges) - n_last]
        if len(kept) < 2:
            raise ValueError(f"ignore_fringe_bins={self.ignore_fringe_bins} removes all "
                             f"{len(edges) - 1} bins of {self.s.name}, "
                             f"nothing would be left to plot.")
        return kept

    def get_fig(self):
        """Return the matplotlib Figure (available after :meth:`plot`)."""
        return self.fig

    def get_ax(self):
        """Return the matplotlib Axes (available after :meth:`plot`)."""
        return self.ax

    def plot(self, ax=None, format_style: FormatStyle = None,
             highlight_peak: bool = True, show_zscores: bool = True, show_zscore_values: bool = True,
             show_info: bool = True, show_counts: bool = True, show_title: bool = True,
             show_kde: bool = False, show_mean: bool = False, show_median: bool = False):
        """Generate histogram plot with optional styling.

        Chrome (title, x-label, grid, fonts, colours) comes from a shared
        :class:`~diive.plotting.FormatStyle` so it matches every other diive plot.
        The histogram-specific rendering (bar colour, peak highlight, z-score
        twiny axis, info/counts boxes and their toggles) stays here.

        Args:
            ax: Matplotlib axes (creates new if None)
            format_style: A :class:`~diive.plotting.FormatStyle` describing the chrome.
                When None the diive house style is used. Default title:
                "{series.name} (between {start_date} and {end_date})".
            highlight_peak: Highlight the bin with most counts (default: True)
            show_zscores: Show z-score overlay on top axis (default: True)
            show_zscore_values: Display z-score values and corresponding data values (default: True)
            show_info: Show method and peak information text (default: True)
            show_counts: Show count labels on each bar (default: True)
            show_title: Display title (default: True)
            show_kde: Overlay a Gaussian-KDE fit line, scaled to the counts (default: False)
            show_mean: Draw a dashed vertical line at the mean (default: False)
            show_median: Draw a dashed vertical line at the median (default: False)

        When ``show_kde``/``show_mean``/``show_median`` are set, their artists are
        labelled with the numeric values so the shared legend (built by
        :class:`~diive.plotting.FormatStyle`) doubles as a compact stats readout.
        """
        # show_title=False suppresses the title via an empty string on a copied style.
        style = format_style or FormatStyle()
        if not show_title:
            style = style.merged(title="")

        # Setup
        self.ax = ax
        self.fig, self.ax, showplot = pf.setup_figax(ax=self.ax, figsize=(16, 9))

        # A series without a single valid value has nothing to bin: `ax.hist`
        # autodetects the range and dies on `[nan, nan]`. Say so on the axes
        # instead -- this is reached from the outlier detectors' own diagnostic
        # plot, where a traceback would kill the detector run over one empty panel.
        if self.s.dropna().empty:
            self.ax.text(0.5, 0.5, f"{self.s.name}: no data",
                         size=16, color="black", transform=self.ax.transAxes,
                         horizontalalignment='center', verticalalignment='center')
            style.apply(ax=self.ax,
                        default_title=f"{self.s.name} (between {self.first_date} and {self.last_date})",
                        default_xlabel="", default_ylabel="Counts")
            if showplot:
                self.fig.show()
            return

        # Plot histogram
        bins = self.n_bins
        if self.ignore_fringe_bins:
            bins = self._trimmed_edges()
        self.counts, self.edges, bars = self.ax.hist(
            x=self.s,
            bins=bins,
            rwidth=0.95,
            color="#78909c"
        )
        self.ax.set_xticks(self.edges)

        ix_max = self.counts.argmax()

        # Show counts for each bar
        if show_counts:
            self.ax.bar_label(bars)

        # Peak: highlight bin with most counts
        if highlight_peak:
            bars[ix_max].set_fc('#FFA726')

        # Distribution overlays: a KDE fit line plus dashed mean/median markers.
        # Each carries its value in the label so the shared legend reads as a
        # small stats panel. The bars are counts, so the expected bar height at x
        # is N * density(x) * (width of the bin containing x) -- the width has to
        # be looked up per bin, because `bins` may be an explicit non-uniform
        # edge list and a single bin width would then misscale every other bin.
        if show_kde or show_mean or show_median:
            vals = self.s.dropna().to_numpy()
            bin_widths = np.diff(self.edges)
            if show_kde and vals.size > 1 and bin_widths.size and bin_widths.min() > 0:
                from scipy.stats import gaussian_kde
                xvals = np.linspace(self.edges[0], self.edges[-1], 200)
                # Index of the bin each sample point falls in (last bin is closed
                # on the right, matching np.histogram).
                ix_bin = np.clip(np.searchsorted(self.edges, xvals, side='right') - 1,
                                 0, bin_widths.size - 1)
                yvals = gaussian_kde(vals)(xvals) * self.counts.sum() * bin_widths[ix_bin]
                self.ax.plot(xvals, yvals, color="#5E35B1", linewidth=2,
                             zorder=500, label="KDE")
            if show_mean and vals.size:
                mean_val = float(np.mean(vals))
                self.ax.axvline(mean_val, color="#D81B60", linestyle="--",
                                linewidth=1.5, zorder=600, label=f"mean = {mean_val:.3g}")
            if show_median and vals.size:
                median_val = float(np.median(vals))
                self.ax.axvline(median_val, color="#1E88E5", linestyle="--",
                                linewidth=1.5, zorder=600, label=f"median = {median_val:.3g}")

        if show_info:
            info_txt = f"method: {self.method}"
            if self.method == 'n_bins':
                info_txt += f"\nn_bins: {self.n_bins}"
            # Otherwise the box claims a bin count the plot does not show.
            if self.ignore_fringe_bins:
                info_txt += f"\nignore_fringe_bins: {self.ignore_fringe_bins}"
            if highlight_peak and self.method == 'n_bins':
                info_txt += f"\nPEAK between {self.edges[ix_max]:.02f} and {self.edges[ix_max + 1]:.02f}"

            self.ax.text(0.05, 0.95, info_txt,
                         size=16, color="black", backgroundcolor='None', transform=self.ax.transAxes,
                         alpha=1, horizontalalignment='left', verticalalignment='top', zorder=999)

        # z-scores. A constant series has zero standard deviation, so every z-score
        # is NaN and there is no z-axis range to lay out. Only the overlay is
        # dropped -- the histogram of a constant series is still worth showing.
        zscores = zscore(series=self.s, absolute=False) if show_zscores else None
        if show_zscores and np.isfinite(zscores).any():
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
                axx_zscores = [f"{z}\n{v:.01f}" for z, v in zip(axx_zscores, axx_ticks_pos, strict=False)]
                self.axx.set_xticklabels(axx_zscores)
            else:
                self.axx.set_xticklabels(axx_zscores)
            # self.axx.set_xlabel(color='#AB47BC', fontsize=20)
            self.axx.tick_params(axis='x', colors='#AB47BC', labelsize=16)
            self.axx.set_xlabel("z-score", color='#AB47BC', fontsize=16)

        # Shared formatting layer: title/x-label/y-label/fonts/grid.
        style.apply(ax=self.ax, default_title=f"{self.s.name} (between {self.first_date} and {self.last_date})",
                    default_xlabel="", default_ylabel="Counts")

        self.ax.locator_params(axis='both', nbins=10)

        if showplot:
            self.fig.show()
