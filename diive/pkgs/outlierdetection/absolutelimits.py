"""
Outlier detection using absolute value limits.

This module provides simple, interpretable outlier detection by comparing values
against fixed minimum and maximum thresholds. Two classes are available:

- **AbsoluteLimits:** Single threshold range applied to all data
  Fast, simple validation for any time series.

- **AbsoluteLimitsDaytimeNighttime:** Separate threshold ranges for daytime and nighttime
  Useful when data characteristics vary significantly by time of day.

Both classes use the quality flag system:
  - flag=0: Value within acceptable range (valid)
  - flag=2: Value outside acceptable range (outlier, removed)
  - NaN: Original missing data preserved

See examples/outlierdetection/absolutelimits.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.common import create_daytime_nighttime_flags


@ConsoleOutputDecorator()
class AbsoluteLimitsDaytimeNighttime(FlagBase):
    """Outlier detection using absolute value limits with separate day/night thresholds.

    Identifies values outside specified acceptable ranges, using different thresholds
    for daytime and nighttime periods. Useful when data characteristics vary significantly
    between day and night conditions.

    **Algorithm:**
    - Automatically detects daytime/nighttime from latitude, longitude, and UTC offset
    - Applies daytime threshold range to daytime records
    - Applies nighttime threshold range to nighttime records
    - Marks records outside their respective ranges as outliers (flag=2)
    - Supports iterative filtering: repeat detection until all outliers removed

    **Quality Flags:**
    - 0: Value within acceptable range (valid)
    - 2: Value outside acceptable range (outlier, removed)
    - NaN: Original missing data

    Example:
        See `examples/outlierdetection/absolutelimits.py` for complete examples.
    """

    flagid = 'OUTLIER_ABSLIM_DTNT'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 daytime_minmax: list[float, float],
                 nighttime_minmax: list[float, float],
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """

        Args:
            series: Time series in which outliers are identified.
            lat: Latitude of location as float, e.g. 46.583056
            lon: Longitude of location as float, e.g. 9.790639
            utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
                The datetime index of the resulting Series will be in this timezone.
            daytime_minmax: Allowed minimum and maximum values in *series* during daytime, e.g. [-50, 50].
            nighttime_minmax: Allowed minimum and maximum values in *series* during nighttime, e.g. [-5, 50].
            idstr: Identifier, added as suffix to output variable names.
            showplot: Show plot with removed data points.
            verbose: More text output to console if *True*.

        Returns:
            Results dataframe via the @repeater wrapper function, dataframe contains
            the filtered time series and flags from all iterations.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.daytime_minmax = daytime_minmax
        self.nighttime_minmax = nighttime_minmax
        self.showplot = showplot
        self.verbose = verbose

        # Detect daytime and nighttime
        self.flag_daytime, flag_nighttime, self.is_daytime, self.is_nighttime = (
            create_daytime_nighttime_flags(timestamp_index=self.series.index, lat=lat, lon=lon, utc_offset=utc_offset))

    def calc(self, repeat: bool = False):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """
        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)
            title = (f"Absolute limits filter daytime/nighttime: {self.series.name}, "
                     f"n_iterations = {n_iterations}, "
                     f"n_outliers = {self.series[self.overall_flag == 2].count()}")
            self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                flag_quality=self.overall_flag, title=title)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.series.copy().dropna()
        flag = pd.Series(index=self.series.index, data=np.nan)

        # Run for daytime (dt)
        _s_dt = s[self.is_daytime].copy()  # Daytime data
        # _zscore_dt = funcs.zscore(series=_s_dt)
        _ok_dt = (_s_dt >= self.daytime_minmax[0]) & (_s_dt <= self.daytime_minmax[1])
        _ok_dt = _ok_dt[_ok_dt].index
        _rejected_dt = (_s_dt < self.daytime_minmax[0]) | (_s_dt > self.daytime_minmax[1])
        _rejected_dt = _rejected_dt[_rejected_dt].index

        # Run for nighttime (nt)
        _s_nt = s[self.is_nighttime].copy()  # Nighttime data
        _ok_nt = (_s_nt >= self.nighttime_minmax[0]) & (_s_nt <= self.nighttime_minmax[1])
        _ok_nt = _ok_nt[_ok_nt].index
        _rejected_nt = (_s_nt < self.nighttime_minmax[0]) | (_s_nt > self.nighttime_minmax[1])
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
            print(f"Total found outliers: {len(_rejected_dt)} values (daytime)")
            print(f"Total found outliers: {len(_rejected_nt)} values (nighttime)")
            print(f"Total found outliers: {n_outliers} values (daytime+nighttime)")

        return ok, rejected, n_outliers


@ConsoleOutputDecorator()
class AbsoluteLimits(FlagBase):
    """Simple outlier detection using absolute minimum and maximum value limits.

    Identifies values outside specified acceptable range. This is the simplest
    outlier detection method: any value < minval or > maxval is flagged as outlier.
    Useful for basic value validation without time-of-day or other context considerations.

    **Algorithm:**
    - Checks if each value is within [minval, maxval] range
    - Marks values outside range as outliers (flag=2)
    - Single-pass detection (no iteration)

    **Quality Flags:**
    - 0: Value within acceptable range (valid)
    - 2: Value outside acceptable range (outlier, removed)
    - NaN: Original missing data

    Example:
        See `examples/outlierdetection/absolutelimits.py` for complete examples.
    """

    flagid = 'OUTLIER_ABSLIM'

    def __init__(self,
                 series: Series,
                 minval: float,
                 maxval: float,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """

        Args:
            series: Time series in which outliers are identified.
            minval: Allowed minimum values in *series*, e.g. -20.
            maxval: Allowed maximum values in *series*, e.g. 20.
            idstr: Identifier, added as suffix to output variable names.
            showplot: Show plot with removed data points.
            verbose: More text output to console if *True*.

        Returns:
            Results dataframe via the @repeater wrapper function, dataframe contains
            the filtered time series and flags from all iterations.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.minval = minval
        self.maxval = maxval
        self.showplot = showplot
        self.verbose = verbose

    def calc(self):
        """Calculate overall flag, based on individual flags from multiple iterations.

        """
        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=False)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""
        ok = (self.series >= self.minval) | (self.series <= self.maxval)
        ok = ok[ok].index
        rejected = (self.series < self.minval) | (self.series > self.maxval)
        rejected = rejected[rejected].index
        n_outliers = len(rejected)
        return ok, rejected, n_outliers
