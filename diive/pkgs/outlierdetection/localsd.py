"""
OUTLIER DETECTION: LOCAL STANDARD DEVIATION
===========================================

This module is part of the diive library:
https://github.com/holukas/diive

"""
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series

import diive.core.plotting.styles.LightTheme as theme
from diive.core.base.flagbase import FlagBase
from diive.core.plotting.plotfuncs import default_format
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.common import create_daytime_nighttime_flags


@ConsoleOutputDecorator()
class LocalSD(FlagBase):
    """
    Identifies outliers in a time series based on the local standard deviation
    within a rolling window.

    This method calculates a rolling median and a rolling standard deviation (SD)
    over a defined window size (`winsize`). Records that deviate from the
    rolling median by more than `n_sd` times the SD are flagged as outliers.
    The process can be repeated (`calc` method with `repeat=True`) to remove
    all detected outliers iteratively.
    """
    flagid = 'OUTLIER_LOCALSD'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 n_sd: float | list = 7,
                 winsize: int | list = None,
                 constant_sd: bool = False,
                 separate_daytime_nighttime: bool = False,
                 lat: float = None,
                 lon: float = None,
                 utc_offset: int = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Identify outliers based on the local standard deviation.

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier, added as suffix to output variable names.
            n_sd: **Number of standard deviations**. Records with data outside this value
                relative to the rolling median are flagged as outliers.
            winsize: **Window size** (number of records). Used to calculate the rolling median and
                rolling standard deviation in a time window of this size. If **None**, it is
                automatically set to $1/20^{th}$ of the series length in `_flagtests`.
            constant_sd: If **True**, the standard deviation across **all data** is used
                when calculating upper and lower limits that define outliers. If **False** (default),
                the rolling standard deviation within the defined window is used.
            showplot: Show plot with removed data points after calculation is complete.
            verbose: More text output to console if **True**.

        Returns:
            The initialized LocalSD object (inherits from FlagBase).

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.n_sd = n_sd
        self.constant_sd = constant_sd
        self.separate_daytime_nighttime = separate_daytime_nighttime
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset
        self.showplot = showplot
        self.verbose = verbose

        self.winsize = int(len(self.series) / 20) if not winsize else winsize

        # Validate and detect daytime and nighttime
        if self.separate_daytime_nighttime:
            self._validate_daytime_nighttime_setup()
            self.flag_daytime, self.flag_nighttime, self.is_daytime, self.is_nighttime = (
                create_daytime_nighttime_flags(timestamp_index=self.series.index, lat=lat, lon=lon,
                                               utc_offset=utc_offset))

        if self.showplot:
            self.fig_localsd, self.ax_localsd, self.ax2_localsd = self._plot_init()

    def _validate_daytime_nighttime_setup(self):
        if not isinstance(self.n_sd, list):
            raise ValueError(f"n_sd must be a list if separate_daytime_nighttime is True")
        if not isinstance(self.winsize, list):
            raise ValueError(f"winsize must be a list if separate_daytime_nighttime is True")
        if not self.lat:
            raise ValueError(f"lat must be set if separate_daytime_nighttime is True")
        if not self.lon:
            raise ValueError(f"lon must be set if separate_daytime_nighttime is True")
        if not self.utc_offset:
            raise ValueError(f"utc_offset must be set if separate_daytime_nighttime is True")

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        This method calls `self.repeat(self.run_flagtests, repeat=repeat)` to perform
        the outlier detection, potentially in a loop until no new outliers are found.

        Args:
            repeat: If **True**, the outlier detection is repeated until all
                outliers are removed.

        Returns:
            Flag series that combines flags from all iterations in one single flag.
            (This is handled by the `FlagBase` class but is the conceptual output).
        """
        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:

            if self.separate_daytime_nighttime:
                # Daytime/nighttime details for separated approach
                title = (f"Absolute limits filter daytime/nighttime: {self.series.name}, "
                         f"n_iterations = {n_iterations}, "
                         f"n_outliers = {self.series[self.overall_flag == 2].count()}")
                self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                    flag_quality=self.overall_flag, title=title)
            else:
                # Default plot only needed for non-separated outlier detection
                self.defaultplot(n_iterations=n_iterations)

            self._plot_finalize(n_iterations=n_iterations)

    def _identify_outliers(
            self,
            s: pd.Series,
            winsize: int,
            n_sd: float,
            iteration: int,
            infotxt: str = None
    ) -> tuple[Any, Any, int]:
        rmedian = s.rolling(window=winsize, center=True, min_periods=3).median()
        if self.constant_sd:
            sd = s.std()
        else:
            sd = s.rolling(window=winsize, center=True, min_periods=3).std()
        upper_limit = rmedian + (sd * n_sd)
        lower_limit = rmedian - (sd * n_sd)
        ok = (s < upper_limit) & (s > lower_limit)
        ok = ok[ok].index
        rejected = (s > upper_limit) | (s < lower_limit)
        rejected = rejected[rejected].index
        n_outliers = len(rejected)
        if self.verbose:
            if self.verbose:
                print(f"ITERATION#{iteration}{infotxt}: Total found outliers: {len(rejected)} values")
        if self.showplot:
            self._plot_add_iteration(rmedian, upper_limit, lower_limit, iteration)
        return ok, rejected, n_outliers

    def _flagtests(self, iteration: int) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """
        Performs the local standard deviation outlier detection test for one iteration.

        It calculates the rolling median and either the rolling or constant standard
        deviation, defines the upper and lower limits, and identifies records
        outside these limits as outliers.

        Args:
            iteration: The current iteration number of the flagging process.

        Returns:
            A tuple containing:
            - ok: DatetimeIndex of records that are **NOT** flagged as outliers.
            - rejected: DatetimeIndex of records that **ARE** flagged as outliers.
            - n_outliers: The total number of outliers found in this iteration.
        """

        ok = None
        rejected = None
        n_outliers = None

        # Working data
        s = self.filteredseries.copy().dropna()

        if not self.separate_daytime_nighttime:
            ok, rejected, n_outliers = self._identify_outliers(
                s=s, iteration=iteration, n_sd=self.n_sd, winsize=self.winsize, infotxt=" (daytime+nighttime)")

        if self.separate_daytime_nighttime:
            # Run for daytime (dt)
            s_dt = s[self.is_daytime].copy()  # Daytime data
            ok_dt, rejected_dt, n_outliers_dt = self._identify_outliers(
                s=s_dt, iteration=iteration, n_sd=self.n_sd[0], winsize=self.winsize[0], infotxt=" (daytime)")

            # Run for nighttime
            s_nt = s[self.is_nighttime].copy()  # Nighttime data
            ok_nt, rejected_nt, n_outliers_nt = self._identify_outliers(
                s=s_nt, iteration=iteration, n_sd=self.n_sd[1], winsize=self.winsize[1], infotxt=" (nighttime)")

            # Collect daytime and nighttime flags in one overall flag
            ok = ok_dt.union(ok_nt)
            rejected = rejected_dt.union(rejected_nt)
            n_outliers = n_outliers_dt + n_outliers_nt

        return ok, rejected, n_outliers

    @staticmethod
    def _plot_init():
        """
        Initializes the plot figure and two subplots (axes) for visualizing
        the outlier detection process and the resulting filtered series.

        The plot uses a GridSpec layout with two stacked subplots.

        Returns:
            A tuple containing:
            - fig: The matplotlib Figure object.
            - ax: The top Axes object (showing original data, limits, and outliers).
            - ax2: The bottom Axes object (showing the filtered series).
        """
        fig = plt.figure(facecolor='white', figsize=(16, 12))
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        ax.xaxis.axis_date()
        ax2.xaxis.axis_date()
        return fig, ax, ax2

    def _plot_add_iteration(self, rmedian, upper_limit, lower_limit, iteration):
        """
        Adds the rolling median and the calculated upper/lower limits for the
        current iteration to the top subplot (`self.ax_localsd`).

        The original series is plotted only during the first iteration.

        Args:
            rmedian: Series containing the rolling median.
            upper_limit: Series containing the upper outlier limit.
            lower_limit: Series containing the lower outlier limit.
            iteration: The current iteration number.
        """
        if iteration == 1:
            self.ax_localsd.plot(self.series.index, self.series, label=f"{self.series.name}", color='black',
                                 marker='o', fillstyle='none', alpha=.9, markersize=4, linestyle='none',
                                 markeredgecolor='black', markeredgewidth=1, zorder=1)
        # self._filteredseries.loc[rejected] = np.nan
        self.ax_localsd.plot(rmedian.index, rmedian, label=f"rolling median", color="#FFA726",
                             alpha=1, markersize=0, markeredgecolor='none', ls='-', lw=2, zorder=3)
        self.ax_localsd.plot(upper_limit.index, upper_limit, label=f"upper limit", color="#7E57C2",
                             alpha=1, markersize=0, markeredgecolor='none', ls='--', lw=1, zorder=4)
        self.ax_localsd.plot(lower_limit.index, lower_limit, label=f"lower limit", color="#26C6DA",
                             alpha=1, markersize=0, markeredgecolor='none', ls='--', lw=1, zorder=4)

    def _plot_finalize(self, n_iterations):
        """
        Finalizes the plot by adding the overall flagged outliers to the top
        subplot and the final filtered series to the bottom subplot.

        It also applies default formatting, sets the plot title, and displays the figure.

        Args:
            n_iterations: The total number of iterations performed during the
                outlier detection process.
        """
        rejected = self.overall_flag == 2
        n_outliers = rejected.sum()

        outliers_only = self.series.copy()
        outliers_only = outliers_only[rejected].copy()
        self.ax_localsd.plot(outliers_only.index, outliers_only,
                             label=f"outlier", color='#F44336', zorder=2, ls='None',
                             alpha=1, markersize=12, markeredgecolor='none', marker='X')

        filtered = self.series.copy()
        filtered.loc[rejected] = np.nan
        self.ax2_localsd.plot(filtered.index, filtered,
                              label=f"filtered series", color='black', marker='o', fillstyle='none',
                              linestyle='none', alpha=.9, markersize=4, markeredgecolor='black', markeredgewidth=1)
        default_format(ax=self.ax_localsd)
        default_format(ax=self.ax2_localsd)
        # default_legend(ax=self.ax, ncol=2, markerscale=5)
        plottitle = (
            f"Outlier detection based on the standard deviation in a rolling window for {self.series.name}\n"
            f"n_iterations = {n_iterations}, n_outliers = {n_outliers}")
        self.fig_localsd.suptitle(plottitle, fontsize=theme.FIGHEADER_FONTSIZE)
        self.fig_localsd.show()


def _example_localsd_daytime_nighttime():
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
                                contamination=0.4,
                                seed=42)  # Add impulse noise (spikes)
    s_noise.name = f"{s.name}+noise"
    # TimeSeries(s_noise).plot()

    lsd = LocalSD(
        series=s_noise,
        separate_daytime_nighttime=True,
        n_sd=[2, 3],
        winsize=[48 * 1, 48 * 2],
        constant_sd=True,
        lat=46.0,
        lon=11.0,
        utc_offset=1,
        showplot=True,
        verbose=True
    )

    lsd.calc(repeat=True)


def _example_localsd():
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
        n_sd=2,
        winsize=48 * 2,
        constant_sd=True,
        showplot=True,
        verbose=True
    )

    lsd.calc(repeat=True)


if __name__ == '__main__':
    _example_localsd_daytime_nighttime()
    # _example_localsd()
