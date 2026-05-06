"""
Outlier detection using Z-score methods.

This module provides three z-score-based outlier detection approaches:

- **Global:** Single z-score threshold for entire time series
- **Daytime/Nighttime:** Separate z-score thresholds for different times of day
- **Rolling:** Adaptive z-score using rolling mean and standard deviation

Quality flags:
  - flag=0: Value within acceptable range (valid)
  - flag=2: Value detected as outlier (removed)
  - NaN: Original missing data preserved

See examples/outlierdetection/zscore.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

import diive.core.funcs.funcs as funcs
from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.common import create_daytime_nighttime_flags


@ConsoleOutputDecorator()
class zScoreDaytimeNighttime(FlagBase):
    flagid = 'OUTLIER_ZSCOREDTNT'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 idstr: str = None,
                 thres_zscore: float = 4,
                 showplot: bool = False,
                 verbose: bool = False):
        """Detect outliers using z-score with separate day/night thresholds.

        Calculates z-scores separately for daytime and nighttime periods,
        useful when data characteristics vary significantly by time of day.

        Example:
            See `examples/outlierdetection/zscore.py` for complete examples.

        Args:
            series: Time series in which outliers are identified.
            lat: Latitude of location (e.g., 46.583056).
                Used to detect daytime/nighttime.
            lon: Longitude of location (e.g., 9.790639).
                Used to detect daytime/nighttime.
            utc_offset: UTC offset in hours (e.g., 1 for UTC+01:00).
                Used to detect daytime/nighttime.
            idstr: Identifier suffix for output variable names.
            thres_zscore: Z-score threshold for outlier detection (default 4).
                Values with |z-score| > threshold are flagged as outliers.
            showplot: If True, display results plot.
            verbose: If True, print iteration statistics.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)

        # Validate inputs
        if thres_zscore <= 0:
            raise ValueError('thres_zscore must be positive.')
        if lat is None or lon is None or utc_offset is None:
            raise ValueError('Location parameters (lat, lon, utc_offset) are required for day/night detection.')

        self.showplot = showplot
        self.verbose = verbose
        self.thres_zscore = thres_zscore

        # Detect daytime and nighttime
        self.flag_daytime, _, self.is_daytime, self.is_nighttime = (
            create_daytime_nighttime_flags(timestamp_index=self.series.index, lat=lat, lon=lon, utc_offset=utc_offset))

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """

        self._overall_flag, n_iterations = self.repeat(func=self.run_flagtests, repeat=repeat)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)
            title = (f"z-score filter daytime/nighttime: {self.series.name}, "
                     f"n_iterations = {n_iterations}, "
                     f"n_outliers = {self.series[self.overall_flag == 2].count()}")
            self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                flag_quality=self.overall_flag, title=title)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy().dropna()
        flag = pd.Series(index=s.index, data=np.nan)

        # Run for daytime (dt)
        _s_dt = s[self.is_daytime].copy()
        _zscore_dt = np.abs(funcs.zscore(series=_s_dt))
        _ok_dt = _zscore_dt <= self.thres_zscore
        _ok_dt = _ok_dt[_ok_dt].index
        _rejected_dt = _zscore_dt > self.thres_zscore
        _rejected_dt = _rejected_dt[_rejected_dt].index

        # Run for nighttime (nt)
        _s_nt = s[self.is_nighttime].copy()
        _zscore_nt = np.abs(funcs.zscore(series=_s_nt))
        _ok_nt = _zscore_nt <= self.thres_zscore
        _ok_nt = _ok_nt[_ok_nt].index
        _rejected_nt = _zscore_nt > self.thres_zscore
        _rejected_nt = _rejected_nt[_rejected_nt].index

        # Collect daytime and nighttime flags in one overall flag
        flag.loc[_ok_dt] = 0
        flag.loc[_rejected_dt] = 2
        flag.loc[_ok_nt] = 0
        flag.loc[_rejected_nt] = 2

        n_outliers = (flag == 2).sum()

        ok = (flag == 0)
        ok = ok[ok].index
        rejected = (flag == 2)
        rejected = rejected[rejected].index

        if self.verbose:
            print(f"ITERATION#{iteration}: Total found outliers: "
                  f"{n_outliers} (daytime+nighttime), "
                  f"{len(_rejected_dt)} (daytime), "
                  f"{len(_rejected_nt)} (nighttime)")

        return ok, rejected, n_outliers


@ConsoleOutputDecorator()
class zScore(FlagBase):
    flagid = 'OUTLIER_ZSCORE'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 thres_zscore: float = 4,
                 showplot: bool = False,
                 plottitle: str = None,
                 verbose: bool = False):
        """Detect outliers using z-score with global threshold.

        Single z-score threshold applied to entire time series.
        Simpler and faster than day/night separation when time-of-day
        variation is not critical.

        Example:
            See `examples/outlierdetection/zscore.py` for complete examples.

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier suffix for output variable names.
            thres_zscore: Z-score threshold for outlier detection (default 4).
                Values with |z-score| > threshold are flagged as outliers.
            showplot: If True, display results plot.
            plottitle: Optional title string for the plot.
            verbose: If True, print iteration statistics.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)

        # Validate inputs
        if thres_zscore <= 0:
            raise ValueError('thres_zscore must be positive.')

        self.showplot = showplot
        self.plottitle = plottitle
        self.verbose = verbose
        self.thres_zscore = thres_zscore

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """
        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy().dropna()

        # Run with threshold
        zscores = np.abs(funcs.zscore(series=s))
        ok = zscores <= self.thres_zscore
        ok = ok[ok].index
        rejected = zscores > self.thres_zscore
        rejected = rejected[rejected].index

        n_outliers = len(rejected)

        if self.verbose:
            print(f"ITERATION#{iteration}: Total found outliers: {len(rejected)} values")

        return ok, rejected, n_outliers


@ConsoleOutputDecorator()
class zScoreRolling(FlagBase):
    flagid = 'OUTLIER_ZSCOREROLLING'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 thres_zscore: float = 4,
                 winsize: int = None,
                 showplot: bool = False,
                 plottitle: str = None,
                 verbose: bool = False):
        """Detect outliers using rolling z-score (adaptive threshold).

        Calculates z-score from rolling mean and rolling std dev, centered
        on each value. Adapts threshold to local data characteristics,
        useful for non-stationary time series.

        Example:
            See `examples/outlierdetection/zscore.py` for complete examples.

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier suffix for output variable names.
            thres_zscore: Z-score threshold for outlier detection (default 4).
                Values with |z-score| > threshold are flagged as outliers.
            winsize: Window size in records for rolling statistics.
                If None, defaults to len(series) / 20 (5% of data).
            showplot: If True, display results plot.
            plottitle: Optional title string for the plot.
            verbose: If True, print iteration statistics.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)

        # Validate inputs
        if thres_zscore <= 0:
            raise ValueError('thres_zscore must be positive.')
        if winsize is not None and winsize < 3:
            raise ValueError('winsize must be at least 3 records.')

        self.showplot = showplot
        self.plottitle = plottitle
        self.verbose = verbose
        self.thres_zscore = thres_zscore
        self.winsize = winsize

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """
        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy().dropna()

        if not self.winsize:
            self.winsize = int(len(s) / 20)

        # Rolling mean and SD, centered on value
        rmean = s.rolling(window=self.winsize, center=True, min_periods=3).mean()
        rsd = s.rolling(window=self.winsize, center=True, min_periods=3).std()

        # Rolling z-score
        rzscore = np.abs((s - rmean) / rsd)

        # Run with threshold
        ok = rzscore <= self.thres_zscore
        ok = ok[ok].index
        rejected = rzscore > self.thres_zscore
        rejected = rejected[rejected].index

        n_outliers = len(rejected)

        if self.verbose:
            print(f"ITERATION#{iteration}: Total found outliers: {len(rejected)} values")

        return ok, rejected, n_outliers
