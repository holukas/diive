"""
LOCAL SD: ROLLING WINDOW DEVIATION DETECTION
=============================================

Identify outliers as values that deviate from a rolling median by more than a
given number of standard deviations. Two modes are available:

- **Global Mode:** Single threshold applied to entire time series
  Fast, simple outlier detection using constant or rolling standard deviation.

- **Daytime/Nighttime Mode:** Separate thresholds for different times of day
  Useful when data characteristics vary significantly by time-of-day conditions.

Quality flags:
  - flag=0: Value within acceptable range (valid)
  - flag=2: Value detected as outlier (removed)
  - NaN: Record missing in the input, so no test could be performed

See examples/preprocessing/outlier_detection/outlier_localsd.py for working examples.

Part of the diive library: https://github.com/holukas/diive
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series

import diive.core.plotting.styles.chainplots as cp
from diive.core.base.flagbase import FlagBase
from diive.core.plotting.plotfuncs import nice_date_ticks
from diive.core.utils.console import VERBOSE_PROGRESS, detail
from diive.preprocessing.outlier_detection.common import (create_daytime_nighttime_flags, reject_legacy_params,
                                                          window_to_records)


def _reject_list_params(**params) -> None:
    """Raise if a caller still passes the old [daytime, nighttime] list form.

    A name-based check cannot catch this: n_sd and winsize kept their names and
    only stopped accepting a list, so the old call would otherwise index a float
    and fail somewhere far from the cause.
    """
    for name, value in params.items():
        if isinstance(value, list):
            raise TypeError(
                f"LocalSD: '{name}' no longer accepts a [daytime, nighttime] list. "
                f"Pass '{name}' to set both periods, and '{name}_daytime' / "
                f"'{name}_nighttime' to override one of them."
            )


class LocalSD(FlagBase):
    """Identifies outliers using rolling window standard deviation.

    The algorithm flags data points that deviate from a rolling median by a
    specified number of standard deviations. Supports two main modes:

    - **Global Mode:** Single set of parameters for entire series
      Uses either rolling or constant standard deviation calculation.

    - **Day/Night Mode:** Separate parameters for daytime/nighttime periods
      Useful when data characteristics vary by time-of-day conditions.

    The detection can be iterative: outliers are removed and the process repeats
    until no new outliers are found.

    Attributes:
        flagid (str): The identifier for this flagging method, 'OUTLIER_LOCALSD'.

    Example:
        ``winsize`` is a record count (48 is one day of 30-minute data) or a time span:

        >>> import diive as dv
        >>> s = dv.load_exampledata_parquet()['NEE_CUT_REF_orig'].loc['2022-07']
        >>> lsd = dv.outliers.LocalSD(series=s, n_sd=4, winsize=48).run()
        >>> cleaned = lsd.filteredseries
        >>> lsd = dv.outliers.LocalSD(series=s, n_sd=4, winsize='1D').run()

        Separate daytime and nighttime, stricter at night (needs the site location):

        >>> lsd = dv.outliers.LocalSD(series=s, n_sd=4, winsize=48, separate_day_night=True,
        ...                           n_sd_daytime=4, n_sd_nighttime=3,
        ...                           lat=46.815, lon=9.856, utc_offset=1).run()

        See `examples/preprocessing/outlier_detection/outlier_localsd.py` for complete examples.
    """
    flagid = 'OUTLIER_LOCALSD'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 n_sd: float = 7,
                 n_sd_daytime: float = None,
                 n_sd_nighttime: float = None,
                 winsize: int | str = None,
                 winsize_daytime: int | str = None,
                 winsize_nighttime: int | str = None,
                 constant_sd: bool = False,
                 separate_day_night: bool = False,
                 lat: float = None,
                 lon: float = None,
                 utc_offset: int = None,
                 showplot: bool = False,
                 verbose: bool = False,
                 **legacy):
        """
        Initializes the LocalSD outlier detection instance.

        Args:
            series (Series): The pandas time series data to analyze.
            idstr (str, optional): A unique identifier for this instance.
                Defaults to None.
            n_sd (float, optional): The number of standard deviations
                (multiplier) to define the outlier threshold. With
                `separate_day_night`, used for both periods unless overridden.
                Defaults to 7.
            n_sd_daytime, n_sd_nighttime (float, optional): Override `n_sd`
                for daytime or nighttime records. None uses `n_sd`.
            winsize (int or str, optional): The rolling window size for
                median/SD calculation, as a record count or as a time span such
                as ``'7D'``. A time span is converted to records at the series'
                frequency (``'1D'`` is 48 records of 30-min data, 144 of 10-min
                data) and must be a whole multiple of it. The time span is
                converted on the full series, but the window then rolls over
                the records present only: missing records (gaps, and outliers
                removed by earlier iterations) do not count, so the window
                stretches across them. With `separate_day_night`, `winsize` is
                used for both periods unless overridden, and the window counts
                records of that period only. ``'7D'`` (1008 records of 10-min
                data) then covers about 14 calendar days if day and night are
                equally long, and in winter about 16-20 days of daytime and
                10-13 days of nighttime data at mid latitudes. If None,
                defaults to 5% of the series length. Defaults to None.
            winsize_daytime, winsize_nighttime (int or str, optional): Override
                `winsize` for daytime or nighttime records, converted like
                `winsize` (record count of the full series, rolled over the
                records of that period). None uses `winsize`.
            constant_sd (bool, optional): If True, uses the standard deviation
                of the *entire* series. If False (default), uses a *rolling*
                standard deviation based on `winsize`. Defaults to False.
            separate_day_night (bool, optional): If True, runs the
                detection separately for day and night periods. Defaults to False.
            lat (float, optional): Latitude, required if
                `separate_day_night` is True for sun-position
                calculations. Defaults to None.
            lon (float, optional): Longitude, required if
                `separate_day_night` is True. Defaults to None.
            utc_offset (int, optional): UTC offset in hours, required if
                `separate_day_night` is True. Defaults to None.
            showplot (bool, optional): If True, generates a plot of the
                detection process upon calling `.calc()`. Defaults to False.
            verbose (bool, optional): If True, prints iteration details to the
                console. Defaults to False.

        Attributes:
            series (Series): The input time series.
            n_sd (float): Standard deviation multiplier.
            winsize (int): Rolling window size in records (time spans converted).
            constant_sd (bool): Flag for constant vs. rolling SD.
            separate_day_night (bool): Flag for day/night mode.
            flag_daytime (Series): Boolean flags for daytime (if enabled).
            flag_nighttime (Series): Boolean flags for nighttime (if enabled).
            is_daytime (Series): Boolean mask for daytime indices (if enabled).
            is_nighttime (Series): Boolean mask for nighttime indices (if enabled).
            fig_localsd (plt.Figure): Matplotlib figure object (if showplot).
            ax_localsd (plt.Axes): Top subplot for data and limits (if showplot).
            ax2_localsd (plt.Axes): Bottom subplot for filtered data (if showplot).
        """
        reject_legacy_params(legacy, 'LocalSD')
        _reject_list_params(n_sd=n_sd, winsize=winsize)
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.n_sd = n_sd
        self.constant_sd = constant_sd
        self.separate_day_night = separate_day_night
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset
        self.showplot = showplot
        self.verbose = verbose

        # Time spans are converted here, from the full series: a daytime or
        # nighttime subset has no regular frequency to convert with.
        winsize = window_to_records(winsize, self.series, name='winsize', verbose=self.verbose)
        winsize_daytime = window_to_records(winsize_daytime, self.series, name='winsize_daytime',
                                            verbose=self.verbose)
        winsize_nighttime = window_to_records(winsize_nighttime, self.series, name='winsize_nighttime',
                                              verbose=self.verbose)
        self.winsize = int(len(self.series) / 20) if winsize is None else winsize

        # Per-period overrides default to None and fall back to the global value,
        # so changing n_sd or winsize alone still affects both periods. winsize
        # falls back to the *resolved* global above, not the raw argument, so the
        # 5%-of-series default reaches both periods too.
        self.n_sd_daytime = n_sd_daytime if n_sd_daytime is not None else self.n_sd
        self.n_sd_nighttime = n_sd_nighttime if n_sd_nighttime is not None else self.n_sd
        self.winsize_daytime = winsize_daytime if winsize_daytime is not None else self.winsize
        self.winsize_nighttime = winsize_nighttime if winsize_nighttime is not None else self.winsize

        # Per-iteration detection band in data units (set by _flagtests); exposed
        # for visualisation. Series over the current cleaned series' index.
        self.last_upper_bound = None
        self.last_lower_bound = None

        # Initialize day/night attributes
        if self.separate_day_night:
            self._validate_daytime_nighttime_setup()
            self.flag_daytime, self.flag_nighttime, self.is_daytime, self.is_nighttime = (
                create_daytime_nighttime_flags(timestamp_index=self.series.index, lat=lat, lon=lon,
                                               utc_offset=utc_offset))
        else:
            self.flag_daytime = None
            self.flag_nighttime = None
            self.is_daytime = None
            self.is_nighttime = None

        if self.showplot:
            self.fig_localsd, self.ax_localsd, self.ax2_localsd = self._plot_init()

    def _validate_daytime_nighttime_setup(self):
        if self.lat is None:
            raise ValueError(f"lat must be set if separate_day_night is True")
        if self.lon is None:
            raise ValueError(f"lon must be set if separate_day_night is True")
        if self.utc_offset is None:
            raise ValueError(f"utc_offset must be set if separate_day_night is True")

    def calc(self, repeat: bool = True, progress_callback=None):
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
            progress_callback: Optional ``callable(iteration, n_outliers,
                filteredseries)`` invoked after each iteration (e.g. to drive a
                progress bar / live-update the cleaned series).

        Returns:
            None. The flag, with outliers marked 2, is in ``overall_flag``
            (managed by the `FlagBase` parent class).
        """
        self._overall_flag, n_iterations = self.repeat(
            self.run_flagtests, repeat=repeat, progress_callback=progress_callback)
        if self.showplot:

            if self.separate_day_night:
                # Daytime/nighttime details for separated approach
                # This is a dedicated plot in FlagBase for daytime/nighttime methods
                self.plot_outlier_daytime_nighttime(
                    series=self.series, flag_daytime=self.flag_daytime, flag_quality=self.overall_flag,
                    title=f"Local standard deviation test, day and night separately: {self.series.name}",
                    n_iterations=n_iterations)
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
    ) -> tuple[DatetimeIndex, DatetimeIndex, int, Series, Series]:
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
            tuple[pd.DatetimeIndex, pd.DatetimeIndex, int, pd.Series, pd.Series]:
                - ok: DatetimeIndex of valid data points.
                - rejected: DatetimeIndex of outlier data points.
                - n_outliers: The count of rejected outliers.
                - upper_limit: Upper threshold for each record of *s*.
                - lower_limit: Lower threshold for each record of *s*.
        """
        rmedian = s.rolling(window=winsize, center=True, min_periods=3).median()
        if self.constant_sd:
            sd = s.std()
        else:
            sd = s.rolling(window=winsize, center=True, min_periods=3).std()
        upper_limit = rmedian + (sd * n_sd)
        lower_limit = rmedian - (sd * n_sd)
        # Inclusive limits, so `ok` is the exact complement of `rejected` (which
        # rejects strictly outside). With strict limits on both sides a value
        # sitting exactly on a limit was in neither set, e.g. every record of a
        # constant series, where sd = 0 puts both limits on the data.
        ok = (s <= upper_limit) & (s >= lower_limit)
        ok = ok[ok].index
        rejected = (s > upper_limit) | (s < lower_limit)
        rejected = rejected[rejected].index
        n_outliers = len(rejected)
        if self.verbose:
            detail(f"ITERATION#{iteration}{time_period}: Total found outliers: {len(rejected)} values", verbose=self.verbose, min_level=VERBOSE_PROGRESS)
        if self.showplot:
            self._plot_add_iteration(rmedian, upper_limit, lower_limit, iteration, time_period=time_period)
        return ok, rejected, n_outliers, upper_limit, lower_limit

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
        # Working data
        s = self.filteredseries.copy().dropna()

        if not self.separate_day_night:
            ok, rejected, n_outliers, upper, lower = self._identify_outliers(
                s=s, iteration=iteration, n_sd=self.n_sd, winsize=self.winsize, time_period=" (global)")
            self.last_upper_bound = upper
            self.last_lower_bound = lower
        else:
            # Run for daytime (dt)
            s_dt = s[self.is_daytime]  # Daytime data
            ok_dt, rejected_dt, n_outliers_dt, up_dt, lo_dt = self._identify_outliers(
                s=s_dt, iteration=iteration, n_sd=self.n_sd_daytime, winsize=self.winsize_daytime,
                time_period=" (daytime)")

            # Run for nighttime
            s_nt = s[self.is_nighttime]  # Nighttime data
            ok_nt, rejected_nt, n_outliers_nt, up_nt, lo_nt = self._identify_outliers(
                s=s_nt, iteration=iteration, n_sd=self.n_sd_nighttime, winsize=self.winsize_nighttime,
                time_period=" (nighttime)")

            # Collect daytime and nighttime flags in one overall flag
            ok = ok_dt.union(ok_nt)
            rejected = rejected_dt.union(rejected_nt)
            n_outliers = n_outliers_dt + n_outliers_nt
            # Combined data-unit band over the union of day + night indices.
            self.last_upper_bound = pd.concat([up_dt, up_nt]).sort_index()
            self.last_lower_bound = pd.concat([lo_dt, lo_nt]).sort_index()

        return ok, rejected, n_outliers

    @staticmethod
    def _plot_init():
        fig = plt.figure(facecolor='white', figsize=(14, 8), layout='constrained')
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2])
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
        color = cp.DAY if time_period == " (daytime)" else cp.NIGHT

        if iteration == 1:
            s = self.series.dropna()
            self.ax_localsd.plot(s.index, s.to_numpy(), label=None, color=cp.KEPT, zorder=1,
                                 **cp.POINTS)

        median_label = f"median{time_period}" if iteration == 1 else None
        limit_label = f"threshold{time_period}" if iteration == 1 else None

        self.ax_localsd.plot(rmedian.index, rmedian, label=median_label, color=color,
                             alpha=1, ls='-', lw=1.6, zorder=3)
        self.ax_localsd.plot(upper_limit.index, upper_limit, label=limit_label, color=color,
                             alpha=.9, ls='--', lw=0.9, zorder=4)
        self.ax_localsd.plot(lower_limit.index, lower_limit, label=None, color=color,
                             alpha=.9, ls='--', lw=0.9, zorder=4)

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
        outliers_only = self.series[rejected_total]

        # Get daytime and nighttime time series
        if self.separate_day_night:
            rejected_dt = (self.overall_flag == 2) & self.is_daytime
            rejected_nt = (self.overall_flag == 2) & self.is_nighttime
            n_outliers_dt = rejected_dt.sum()
            n_outliers_nt = rejected_nt.sum()
            outliers_only_dt = self.series[rejected_dt]
            outliers_only_nt = self.series[rejected_nt]
            self.ax_localsd.plot(outliers_only_dt.index, outliers_only_dt,
                                 label=f"rejected, daytime ({n_outliers_dt:,})", color=cp.REJECTED,
                                 zorder=5, ls='None', marker='o', markersize=5, markeredgecolor=cp.DAY,
                                 markeredgewidth=1)
            self.ax_localsd.plot(outliers_only_nt.index, outliers_only_nt,
                                 label=f"rejected, nighttime ({n_outliers_nt:,})", color=cp.REJECTED,
                                 zorder=5, ls='None', marker='o', markersize=5, markeredgecolor=cp.NIGHT,
                                 markeredgewidth=1)
            subtitle = (f"{n_outliers_total:,} records rejected ({n_outliers_dt:,} daytime, "
                        f"{n_outliers_nt:,} nighttime)")
        else:
            self.ax_localsd.plot(outliers_only.index, outliers_only,
                                 label=f"rejected ({n_outliers_total:,})", color=cp.REJECTED, zorder=5,
                                 ls='None', marker='o', markersize=5, markeredgewidth=0)
            subtitle = f"{n_outliers_total:,} records rejected"

        filtered = self.series.copy()
        filtered.loc[rejected_total] = np.nan
        filtered = filtered.dropna()
        self.ax2_localsd.plot(filtered.index, filtered.to_numpy(), label=None, color=cp.CLEANED,
                              **cp.POINTS)
        for ax in (self.ax_localsd, self.ax2_localsd):
            cp.style_axes(ax)
            ax.set_ylabel(self.series.name)
        cp.panel_label(self.ax_localsd, "Rolling median and thresholds")
        cp.panel_label(self.ax2_localsd, "After the test")
        self.ax_localsd.legend(loc='lower right', bbox_to_anchor=(1, 1), frameon=False, ncol=3,
                               fontsize=9, markerscale=1.5, borderaxespad=0.2)
        self.ax_localsd.tick_params(axis='x', labelbottom=False)
        nice_date_ticks(ax=self.ax2_localsd)

        # Each iteration draws its own median and threshold lines.
        subtitle += f", {n_iterations} iteration{'s' if n_iterations != 1 else ''}"
        if n_iterations > 1:
            subtitle += " (one set of lines per iteration)"
        cp.figure_title(self.fig_localsd,
                        f"Local standard deviation test: {self.series.name}", subtitle)
        self.fig_localsd.show()
