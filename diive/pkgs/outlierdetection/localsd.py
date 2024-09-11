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
                 constant_sd: bool = False,
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
            constant_sd: If *True*, the standard deviation across all data is used
                when calculating upper and lower limits that define outliers. By
                default, this is set to *False*, i.e., the rolling standard deviation
                within the defined window is used.
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
        self.constant_sd = constant_sd
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
        if self.constant_sd:
            sd = s.std()
        else:
            sd = s.rolling(window=self.winsize, center=True, min_periods=3).std()
        upper_limit = rmedian + (sd * self.n_sd)
        lower_limit = rmedian - (sd * self.n_sd)

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
        ax.xaxis.axis_date()
        ax2.xaxis.axis_date()
        return fig, ax, ax2

    def _plot_add_iteration(self, rmedian, upper_limit, lower_limit, iteration):
        """Add iteration data to plot, but do not show plot yet."""
        if iteration == 1:
            self.ax.plot(self.series.index, self.series, label=f"{self.series.name}", color='black',
                         marker='o', fillstyle='none', alpha=.9, markersize=4, linestyle='none',
                         markeredgecolor='black', markeredgewidth=1, zorder=1)
        # self._filteredseries.loc[rejected] = np.nan
        self.ax.plot(rmedian.index, rmedian, label=f"rolling median", color="#FFA726",
                     alpha=1, markersize=0, markeredgecolor='none', ls='-', lw=2, zorder=3)
        self.ax.plot(upper_limit.index, upper_limit, label=f"upper limit", color="#7E57C2",
                     alpha=1, markersize=0, markeredgecolor='none', ls='--', lw=1, zorder=4)
        self.ax.plot(lower_limit.index, lower_limit, label=f"lower limit", color="#26C6DA",
                     alpha=1, markersize=0, markeredgecolor='none', ls='--', lw=1, zorder=4)

    def _plot_finalize(self, n_iterations):
        """Finalize and show plot."""
        rejected = self.overall_flag == 2
        n_outliers = rejected.sum()

        outliers_only = self.series.copy()
        outliers_only = outliers_only[rejected].copy()
        self.ax.plot(outliers_only.index, outliers_only,
                     label=f"outlier", color='#F44336', zorder=2, ls='None',
                     alpha=1, markersize=12, markeredgecolor='none', marker='X')

        filtered = self.series.copy()
        filtered.loc[rejected] = np.nan
        self.ax2.plot(filtered.index, filtered,
                      label=f"filtered series", color='black', marker='o', fillstyle='none',
                      linestyle='none', alpha=.9, markersize=4, markeredgecolor='black', markeredgewidth=1)
        default_format(ax=self.ax)
        default_format(ax=self.ax2)
        # default_legend(ax=self.ax, ncol=2, markerscale=5)
        plottitle = (
            f"Outlier detection based on the standard deviation in a rolling window for {self.series.name}\n"
            f"n_iterations = {n_iterations}, n_outliers = {n_outliers}")
        self.fig.suptitle(plottitle, fontsize=theme.FIGHEADER_FONTSIZE)
        self.fig.show()


def example():
    import importlib.metadata
    import diive.configs.exampledata as ed
    from diive.pkgs.createvar.noise import add_impulse_noise
    import warnings
    warnings.filterwarnings('ignore')
    version_diive = importlib.metadata.version("diive")
    print(f"diive version: v{version_diive}")
    df = ed.load_exampledata_parquet()
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()
    s = s.loc[s.index.month == 7].copy()
    s_noise = add_impulse_noise(series=s,
                                factor_low=-10,
                                factor_high=3,
                                contamination=0.4)  # Add impulse noise (spikes)
    s_noise.name = f"{s.name}+noise"
    # TimeSeries(s_noise).plot()

    lsd = LocalSD(
        series=s_noise,
        n_sd=2.1,
        winsize=48 * 2,
        constant_sd=True,
        showplot=True,
        verbose=True
    )

    lsd.calc(repeat=True)


if __name__ == '__main__':
    example()
