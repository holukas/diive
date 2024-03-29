"""
OUTLIER DETECTION: LOCAL STANDARD DEVIATION
===========================================

This module is part of the diive library:
https://github.com/holukas/diive

"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pandas import DatetimeIndex, Series

import diive.core.plotting.styles.LightTheme as theme
from diive.core.base.flagbase import FlagBase
from diive.core.plotting.plotfuncs import default_format
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
class LocalSD(FlagBase):
    flagid = 'OUTLIER_LOCALSD'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 n_sd: float = 7,
                 winsize: int = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Identify outliers based on the local standard deviation.

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier, added as suffix to output variable names.
            winsize: Window size. Is used to calculate the rolling median and
                rolling standard deviation in a time window of size *winsize* records.
            n_sd: Number of standard deviations. Records with sd outside this value
                are flagged as outliers.
            showplot: Show plot with removed data points.
            verbose: More text output to console if *True*.

        Returns:
            Flag series that combines flags from all iterations in one single flag.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.n_sd = n_sd
        self.winsize = winsize
        self.showplot = showplot
        self.verbose = verbose

        if self.showplot:
            self.fig, self.ax, self.ax2 = self._plot_init()

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """

        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:
            # Default plot for outlier tests, showing rejected values
            self.defaultplot(n_iterations=n_iterations)
            self._plot_finalize(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy()
        s = s.dropna()

        if not self.winsize:
            self.winsize = int(len(s) / 20)

        rmedian = s.rolling(window=self.winsize, center=True, min_periods=3).median()
        rsd = s.rolling(window=self.winsize, center=True, min_periods=3).std()
        upper_limit = rmedian + (rsd * self.n_sd)
        lower_limit = rmedian - (rsd * self.n_sd)

        ok = (s < upper_limit) & (s > lower_limit)
        ok = ok[ok].index
        rejected = (s > upper_limit) | (s < lower_limit)
        rejected = rejected[rejected].index

        n_outliers = len(rejected)

        if self.verbose:
            if self.verbose:
                print(f"ITERATION#{iteration}: Total found outliers: {len(rejected)} values")
        if self.showplot:
            self._plot_add_iteration(rmedian, upper_limit, lower_limit, iteration)

        return ok, rejected, n_outliers

    @staticmethod
    def _plot_init():
        """Initialize plot that collects iteration data."""
        fig = plt.figure(facecolor='white', figsize=(16, 12))
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        return fig, ax, ax2

    def _plot_add_iteration(self, rmedian, upper_limit, lower_limit, iteration):
        """Add iteration data to plot, but do not show plot yet."""
        if iteration == 1:
            self.ax.plot_date(self.series.index, self.series, label=f"{self.series.name}", color='none',
                              alpha=1, markersize=4, markeredgecolor='black', markeredgewidth=1, zorder=1)
        # self._filteredseries.loc[rejected] = np.nan
        self.ax.plot_date(rmedian.index, rmedian, label=f"rolling median", color="#FFA726",
                          alpha=1, markersize=0, markeredgecolor='none', ls='-', lw=2, zorder=3)
        self.ax.plot_date(upper_limit.index, upper_limit, label=f"upper limit", color="#7E57C2",
                          alpha=1, markersize=0, markeredgecolor='none', ls='--', lw=1, zorder=4)
        self.ax.plot_date(lower_limit.index, lower_limit, label=f"lower limit", color="#26C6DA",
                          alpha=1, markersize=0, markeredgecolor='none', ls='--', lw=1, zorder=4)

    def _plot_finalize(self, n_iterations):
        """Finalize and show plot."""
        rejected = self.overall_flag == 2
        n_outliers = rejected.sum()

        outliers_only = self.series.copy()
        outliers_only = outliers_only[rejected].copy()
        self.ax.plot_date(outliers_only.index, outliers_only,
                          label=f"filtered series", color='#F44336', zorder=2,
                          alpha=1, markersize=12, markeredgecolor='none', fmt='X')

        filtered = self.series.copy()
        filtered.loc[rejected] = np.nan
        self.ax2.plot_date(filtered.index, filtered,
                           label=f"filtered series", color='none',
                           alpha=1, markersize=4, markeredgecolor='black', markeredgewidth=1)
        default_format(ax=self.ax)
        default_format(ax=self.ax2)
        # default_legend(ax=self.ax, ncol=2, markerscale=5)
        plottitle = (
            f"Outlier detection based on the standard deviation in a rolling window for {self.series.name}\n"
            f"n_iterations = {n_iterations}, n_outliers = {n_outliers}")
        self.fig.suptitle(plottitle, fontsize=theme.FIGHEADER_FONTSIZE)
        self.fig.show()
