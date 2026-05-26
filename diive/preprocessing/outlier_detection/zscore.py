"""
Outlier detection using Z-score methods.

This module provides two z-score-based outlier detection approaches:

- **Global:** Single z-score threshold for entire time series
- **Daytime/Nighttime:** Separate z-score thresholds for different times of day
- **Rolling:** Adaptive z-score using rolling mean and standard deviation

Quality flags:
  - flag=0: Value within acceptable range (valid)
  - flag=2: Value detected as outlier (removed)
  - NaN: Original missing data preserved

See examples/preprocessing/outlier_detection/zscore.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

import diive.core.funcs.funcs as funcs
from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import detail
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.preprocessing.outlier_detection.common import create_daytime_nighttime_flags


@ConsoleOutputDecorator()
class zScore(FlagBase):
    """Detect outliers using z-score with flexible threshold modes.

    Supports two modes:

    1. **Global Mode (separate_daytime_nighttime=False):**
       Single z-score threshold applied to entire time series. Fast and simple.

    2. **Daytime/Nighttime Mode (separate_daytime_nighttime=True):**
       Separate z-score thresholds for daytime and nighttime. Useful when
       data characteristics vary significantly between day and night.

    **Algorithm:**
    - In global mode: Calculates z-score for entire series, flags values > threshold
    - In day/night mode: Calculates z-scores separately for daytime and nighttime periods,
      applies appropriate threshold to each period
    - Marks records with |z-score| > threshold as outliers (flag=2)

    **Quality Flags:**
    - 0: Value within acceptable range (valid)
    - 2: Value detected as outlier (removed)
    - NaN: Original missing data

    Example:
        See `examples/preprocessing/outlier_detection/zscore.py` for complete examples.
    """

    flagid = 'OUTLIER_ZSCORE'

    def __init__(self,
                 series: Series,
                 separate_daytime_nighttime: bool = False,
                 lat: float = None,
                 lon: float = None,
                 utc_offset: int = None,
                 idstr: str = None,
                 thres_zscore: float = 4,
                 showplot: bool = False,
                 plottitle: str = None,
                 verbose: bool = False):
        """Initialize z-score outlier detector.

        Args:
            series: Time series in which outliers are identified.
            separate_daytime_nighttime: If True, use separate thresholds for day/night;
                if False, use single global threshold. Default False.
            lat: Latitude of location as float (required if separate_daytime_nighttime=True).
                Example: 46.583056. Used to detect daytime/nighttime.
            lon: Longitude of location as float (required if separate_daytime_nighttime=True).
                Example: 9.790639. Used to detect daytime/nighttime.
            utc_offset: UTC offset of timestamp_index (required if separate_daytime_nighttime=True).
                Example: 1 for UTC+01:00. Used to detect daytime/nighttime.
            idstr: Identifier, added as suffix to output variable names.
            thres_zscore: Z-score threshold for outlier detection (default 4).
                Values with |z-score| > threshold are flagged as outliers.
            showplot: Show plot with removed data points.
            plottitle: Optional title for the plot.
            verbose: More text output to console if True.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)

        # Validate inputs
        if thres_zscore <= 0:
            raise ValueError('thres_zscore must be positive.')

        self.separate_daytime_nighttime = separate_daytime_nighttime
        self.showplot = showplot
        self.plottitle = plottitle
        self.verbose = verbose
        self.thres_zscore = thres_zscore

        if separate_daytime_nighttime:
            # Day/night mode
            if lat is None or lon is None or utc_offset is None:
                raise ValueError(
                    'lat, lon, and utc_offset are required when separate_daytime_nighttime=True'
                )

            # Detect daytime and nighttime
            self.flag_daytime, _, self.is_daytime, self.is_nighttime = (
                create_daytime_nighttime_flags(timestamp_index=self.series.index, lat=lat, lon=lon, utc_offset=utc_offset))

    def calc(self, repeat: bool = True):
        """Calculate overall flag based on z-score thresholds.

        Args:
            repeat: If True, outlier detection is repeated until convergence.
        """
        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)
            if self.separate_daytime_nighttime:
                title = (f"z-score filter daytime/nighttime: {self.series.name}, "
                         f"n_iterations = {n_iterations}, "
                         f"n_outliers = {self.series[self.overall_flag == 2].count()}")
                self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                    flag_quality=self.overall_flag, title=title)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""
        if self.separate_daytime_nighttime:
            return self._flagtests_daytime_nighttime(iteration)
        else:
            return self._flagtests_global(iteration)

    def _flagtests_global(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Global z-score test"""
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
            detail(f"ITERATION#{iteration}: Total found outliers: {len(rejected)} values", verbose=self.verbose)

        return ok, rejected, n_outliers

    def _flagtests_daytime_nighttime(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Separate daytime/nighttime z-score test"""
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
            detail(f"ITERATION#{iteration}: Total found outliers: "
                   f"{n_outliers} (daytime+nighttime), "
                   f"{len(_rejected_dt)} (daytime), "
                   f"{len(_rejected_nt)} (nighttime)", verbose=self.verbose)

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
            See `examples/preprocessing/outlier_detection/zscore.py` for complete examples.

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
            detail(f"ITERATION#{iteration}: Total found outliers: {len(rejected)} values", verbose=self.verbose)

        return ok, rejected, n_outliers
