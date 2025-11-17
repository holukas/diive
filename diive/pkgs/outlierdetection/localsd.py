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
from diive.core.plotting.plotfuncs import default_format, default_legend
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.common import create_daytime_nighttime_flags


@ConsoleOutputDecorator()
class LocalSD(FlagBase):
    """
    Identifies time series outliers using a local standard deviation approach.

    This class flags data points that deviate from a rolling median by a
    specified number of standard deviations (local or constant). It can
    iteratively remove outliers and recalculate until no new outliers are found.

    The class supports two main operation modes:
    1.  **Standard:** Analyzes the entire series with one set of parameters.
    2.  **Day/Night:** Analyzes daytime and nighttime data separately, using
        distinct parameters (window size, SD multiplier) for each period.

    Attributes:
        flagid (str): The identifier for this flagging method, 'OUTLIER_LOCALSD'.

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
        """
        Initializes the LocalSD outlier detection instance.

        Args:
            series (Series): The pandas time series data to analyze.
            idstr (str, optional): A unique identifier for this instance.
                Defaults to None.
            n_sd (float | list, optional): The number of standard deviations
                (multiplier) to define the outlier threshold. If
                `separate_daytime_nighttime` is True, this must be a list
                [day_sd, night_sd]. Defaults to 7.
            winsize (int | list, optional): The rolling window size for
                median/SD calculation. If `separate_daytime_nighttime` is True,
                this must be a list [day_win, night_win]. If None, defaults
                to 5% of the series length. Defaults to None.
            constant_sd (bool, optional): If True, uses the standard deviation
                of the *entire* series. If False (default), uses a *rolling*
                standard deviation based on `winsize`. Defaults to False.
            separate_daytime_nighttime (bool, optional): If True, runs the
                detection separately for day and night periods. Defaults to False.
            lat (float, optional): Latitude, required if
                `separate_daytime_nighttime` is True for sun-position
                calculations. Defaults to None.
            lon (float, optional): Longitude, required if
                `separate_daytime_nighttime` is True. Defaults to None.
            utc_offset (int, optional): UTC offset in hours, required if
                `separate_daytime_nighttime` is True. Defaults to None.
            showplot (bool, optional): If True, generates a plot of the
                detection process upon calling `.calc()`. Defaults to False.
            verbose (bool, optional): If True, prints iteration details to the
                console. Defaults to False.

        Attributes:
            series (Series): The input time series.
            n_sd (float | list): Standard deviation multiplier.
            winsize (int | list): Rolling window size.
            constant_sd (bool): Flag for constant vs. rolling SD.
            separate_daytime_nighttime (bool): Flag for day/night mode.
            flag_daytime (Series): Boolean flags for daytime (if enabled).
            flag_nighttime (Series): Boolean flags for nighttime (if enabled).
            is_daytime (Series): Boolean mask for daytime indices (if enabled).
            is_nighttime (Series): Boolean mask for nighttime indices (if enabled).
            fig_localsd (plt.Figure): Matplotlib figure object (if showplot).
            ax_localsd (plt.Axes): Top subplot for data and limits (if showplot).
            ax2_localsd (plt.Axes): Bottom subplot for filtered data (if showplot).
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
        """
        Runs the iterative outlier detection process.

        This method repeatedly calls the `_flagtests` method (via `self.repeat`
        from the parent `FlagBase`) to find and flag outliers.

        If `showplot` was set to True in the constructor, this method will
        also generate and display the final summary plot after calculation.

        Args:
            repeat (bool, optional): If True, the detection process is
                repeated until a stable state (no new outliers) is reached.
                If False, it runs only once. Defaults to True.

        Returns:
            pd.Series: A flag series (managed by the `FlagBase` parent class)
            where outliers are marked with flag ID 2.
        """
        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:

            if self.separate_daytime_nighttime:
                # Daytime/nighttime details for separated approach
                # This is a dedicated plot in FlagBase for daytime/nighttime methods
                title = (f"Absolute limits filter daytime/nighttime: {self.series.name}, "
                         f"n_iterations = {n_iterations}, "
                         f"n_outliers = {self.series[self.overall_flag == 2].count()}")
                self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                    flag_quality=self.overall_flag, title=title)
            else:
                # Default plot only needed for non-separated outlier detection
                # This is a general plot in FlagBase for outlier methods
                self.defaultplot(n_iterations=n_iterations)

            # Plot special to LocalSD
            self._plot_finalize(n_iterations=n_iterations)

    def _identify_outliers(
            self,
            s: pd.Series,
            winsize: int,
            n_sd: float,
            iteration: int,
            time_period: str = None
    ) -> tuple[Any, Any, int]:
        """
        Performs a single pass of outlier detection on a series.

        Calculates the rolling median, standard deviation (rolling or constant),
        and the resulting upper/lower thresholds. It then identifies all
        data points outside these thresholds.

        Args:
            s (pd.Series): The time series data to analyze (e.g., full series,
                or just daytime/nighttime subset).
            winsize (int): The rolling window size to use.
            n_sd (float): The standard deviation multiplier for the threshold.
            iteration (int): The current iteration number (for plotting and
                verbose output).
            time_period (str, optional): Additional text to display in verbose
                output (e.g., "(daytime)"). Defaults to None.

        Returns:
            tuple[pd.DatetimeIndex, pd.DatetimeIndex, int]:
                - ok: DatetimeIndex of valid data points.
                - rejected: DatetimeIndex of outlier data points.
                - n_outliers: The count of rejected outliers.
        """
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
            print(f"ITERATION#{iteration}{time_period}: Total found outliers: {len(rejected)} values")
        if self.showplot:
            self._plot_add_iteration(rmedian, upper_limit, lower_limit, iteration, time_period=time_period)
        return ok, rejected, n_outliers

    def _flagtests(self, iteration: int) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """
        Performs one full iteration of the local SD outlier test.

        This method handles the logic for a single pass. If day/night
        separation is *False*, it calls `_identify_outliers` once on the
        current filtered series.

        If day/night separation is *True*, it splits the data into daytime
        and nighttime subsets and calls `_identify_outliers` on each subset
        separately, using their respective `n_sd` and `winsize` parameters.
        The results are then merged.

        Args:
            iteration (int): The current iteration number, passed from the
                `repeat` loop (from `FlagBase`).

        Returns:
            tuple[DatetimeIndex, DatetimeIndex, int]:
                - ok: DatetimeIndex of all valid records for this iteration.
                - rejected: DatetimeIndex of all outliers found in this iteration.
                - n_outliers: The total number of outliers found.
        """
        ok = None
        rejected = None
        n_outliers = None

        # Working data
        s = self.filteredseries.copy().dropna()

        if not self.separate_daytime_nighttime:
            ok, rejected, n_outliers = self._identify_outliers(
                s=s, iteration=iteration, n_sd=self.n_sd, winsize=self.winsize, time_period=" (daytime+nighttime)")

        if self.separate_daytime_nighttime:
            # Run for daytime (dt)
            s_dt = s[self.is_daytime].copy()  # Daytime data
            ok_dt, rejected_dt, n_outliers_dt = self._identify_outliers(
                s=s_dt, iteration=iteration, n_sd=self.n_sd[0], winsize=self.winsize[0], time_period=" (daytime)")

            # Run for nighttime
            s_nt = s[self.is_nighttime].copy()  # Nighttime data
            ok_nt, rejected_nt, n_outliers_nt = self._identify_outliers(
                s=s_nt, iteration=iteration, n_sd=self.n_sd[1], winsize=self.winsize[1], time_period=" (nighttime)")

            # Collect daytime and nighttime flags in one overall flag
            ok = ok_dt.union(ok_nt)
            rejected = rejected_dt.union(rejected_nt)
            n_outliers = n_outliers_dt + n_outliers_nt

        return ok, rejected, n_outliers

    @staticmethod
    def _plot_init():
        fig = plt.figure(facecolor='white', figsize=(16, 12))
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        ax.xaxis.axis_date()
        ax2.xaxis.axis_date()
        return fig, ax, ax2

    def _plot_add_iteration(self, rmedian, upper_limit, lower_limit, iteration, time_period=""):
        """
        Adds the results of a single iteration to the main plot.

        This plots the rolling median, upper limit, and lower limit lines
        on the top subplot (`self.ax_localsd`). The original series is
        plotted only on the first iteration (iteration == 1).

        Args:
            rmedian (Series): Series of the rolling median for this iteration.
            upper_limit (Series): Series of the upper threshold for this iteration.
            lower_limit (Series): Series of the lower threshold for this iteration.
            iteration (int): The current iteration number.
        """
        if time_period == " (daytime+nighttime)":
            color = "#FFA726"
        else:
            if time_period == " (daytime)":
                color = "#FF5722"
            else:
                color = "#81D4FA"

        if iteration == 1:
            # label = f"{self.series.name}"
            self.ax_localsd.plot(self.series.index, self.series, label=None, color='black',
                                 marker='o', fillstyle='none', alpha=.9, markersize=4, linestyle='none',
                                 markeredgecolor='black', markeredgewidth=1, zorder=1)
        # self._filteredseries.loc[rejected] = np.nan
        self.ax_localsd.plot(rmedian.index, rmedian, label=None, color=color,
                             alpha=1, markersize=0, markeredgecolor='none', ls='-', lw=2, zorder=3)
        self.ax_localsd.plot(upper_limit.index, upper_limit, label=None, color=color,
                             alpha=1, markersize=0, markeredgecolor='none', ls='--', lw=1, zorder=4)
        self.ax_localsd.plot(lower_limit.index, lower_limit, label=None, color=color,
                             alpha=1, markersize=0, markeredgecolor='none', ls='--', lw=1, zorder=4)

    def _plot_finalize(self, n_iterations):
        """
        Finalizes and displays the outlier detection plot.

        This method adds the final set of all identified outliers (as 'X'
        markers) to the top plot and plots the complete filtered series
        (with NaNs for outliers) on the bottom plot. It then applies
        formatting and shows the figure.

        Args:
            n_iterations (int): The total number of iterations performed.
        """
        # Get overall time series
        rejected_total = self.overall_flag == 2
        n_outliers_total = rejected_total.sum()
        outliers_only = self.series.copy()
        outliers_only = outliers_only[rejected_total].copy()

        # Get daytime and nighttime time series
        if self.separate_daytime_nighttime:
            rejected_dt = (self.overall_flag == 2) & self.is_daytime
            rejected_nt = (self.overall_flag == 2) & self.is_nighttime
            n_outliers_dt = rejected_dt.sum()
            n_outliers_nt = rejected_nt.sum()
            outliers_only_dt = self.series.copy()
            outliers_only_dt = outliers_only_dt[rejected_dt].copy()
            outliers_only_nt = self.series.copy()
            outliers_only_nt = outliers_only_nt[rejected_nt].copy()
            self.ax_localsd.plot(outliers_only_dt.index, outliers_only_dt,
                                 label=f"daytime outlier ({n_outliers_dt})", color='#FF5722', zorder=2, ls='None',
                                 alpha=1, markersize=12, markeredgecolor='none', marker='X')
            self.ax_localsd.plot(outliers_only_nt.index, outliers_only_nt,
                                 label=f"nighttime outlier ({n_outliers_nt})", color='#81D4FA', zorder=2, ls='None',
                                 alpha=1, markersize=12, markeredgecolor='none', marker='X')
        else:
            self.ax_localsd.plot(outliers_only.index, outliers_only,
                                 label=f"outlier ({n_outliers_total})", color='#F44336', zorder=2, ls='None',
                                 alpha=1, markersize=12, markeredgecolor='none', marker='X')

        filtered = self.series.copy()
        filtered.loc[rejected_total] = np.nan
        self.ax2_localsd.plot(filtered.index, filtered,
                              label=f"filtered series", color='black', marker='o', fillstyle='none',
                              linestyle='none', alpha=.9, markersize=4, markeredgecolor='black', markeredgewidth=1)
        default_format(ax=self.ax_localsd)
        default_format(ax=self.ax2_localsd)
        default_legend(ax=self.ax_localsd, ncol=2, markerscale=1)
        plottitle = (
            f"Outlier detection based on the standard deviation in a rolling window for {self.series.name}\n"
            f"n_iterations = {n_iterations}, n_outliers = {n_outliers_total}")
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
        n_sd=[3, 2],
        winsize=[48 * 2, 48 * 1],
        constant_sd=False,
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
