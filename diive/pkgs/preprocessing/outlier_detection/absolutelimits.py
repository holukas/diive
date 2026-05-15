"""
Outlier detection using absolute value limits.

This module provides simple, interpretable outlier detection by comparing values
against fixed minimum and maximum thresholds.

- **Global Mode:** Single threshold range applied to all data
  Fast, simple validation for any time series.

- **Daytime/Nighttime Mode:** Separate threshold ranges for daytime and nighttime
  Useful when data characteristics vary significantly by time of day.

Both modes use the quality flag system:
  - flag=0: Value within acceptable range (valid)
  - flag=2: Value outside acceptable range (outlier, removed)
  - NaN: Original missing data preserved

See examples/preprocessing/outlier_detection/absolutelimits.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.preprocessing.outlier_detection.common import create_daytime_nighttime_flags


@ConsoleOutputDecorator()
class AbsoluteLimits(FlagBase):
    """Outlier detection using absolute value limits.

    Identifies values outside specified acceptable range(s). Can operate in two modes:

    1. **Global Mode (separate_daytime_nighttime=False):**
       Single threshold range applied to all data. Fast, simple validation.

    2. **Daytime/Nighttime Mode (separate_daytime_nighttime=True):**
       Separate threshold ranges for daytime and nighttime periods. Useful when
       data characteristics vary significantly between day and night conditions.

    **Algorithm:**
    - In global mode: Checks if each value is within [minval, maxval] range
    - In day/night mode: Automatically detects daytime/nighttime from location
      and applies appropriate threshold range to each period
    - Marks records outside their respective ranges as outliers (flag=2)

    **Quality Flags:**
    - 0: Value within acceptable range (valid)
    - 2: Value outside acceptable range (outlier, removed)
    - NaN: Original missing data

    Example:
        See `examples/preprocessing/outlier_detection/absolutelimits.py` for complete examples.
    """

    flagid = 'OUTLIER_ABSLIM'

    def __init__(self,
                 series: Series,
                 minval: float = None,
                 maxval: float = None,
                 separate_daytime_nighttime: bool = False,
                 daytime_minmax: list[float, float] = None,
                 nighttime_minmax: list[float, float] = None,
                 lat: float = None,
                 lon: float = None,
                 utc_offset: int = None,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """
        Initialize absolute limits outlier detector.

        Args:
            series: Time series in which outliers are identified.
            minval: Minimum acceptable value (global mode). Required if
                separate_daytime_nighttime=False.
            maxval: Maximum acceptable value (global mode). Required if
                separate_daytime_nighttime=False.
            separate_daytime_nighttime: If True, use separate day/night thresholds;
                if False, use global thresholds. Default False.
            daytime_minmax: [min, max] acceptable range during daytime (day/night mode).
                Required if separate_daytime_nighttime=True.
            nighttime_minmax: [min, max] acceptable range during nighttime (day/night mode).
                Required if separate_daytime_nighttime=True.
            lat: Latitude of location as float (required for day/night mode).
                Example: 46.583056
            lon: Longitude of location as float (required for day/night mode).
                Example: 9.790639
            utc_offset: UTC offset of timestamp_index (required for day/night mode).
                Example: 1 for UTC+01:00
            idstr: Identifier, added as suffix to output variable names.
            showplot: Show plot with removed data points.
            verbose: More text output to console if True.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)

        # Auto-detect separate_daytime_nighttime if day/night params are provided
        if daytime_minmax is not None or nighttime_minmax is not None:
            separate_daytime_nighttime = True

        self.separate_daytime_nighttime = separate_daytime_nighttime
        self.showplot = showplot
        self.verbose = verbose

        if separate_daytime_nighttime:
            # Day/night mode
            if daytime_minmax is None or nighttime_minmax is None:
                raise ValueError(
                    "daytime_minmax and nighttime_minmax are required when "
                    "separate_daytime_nighttime=True"
                )
            if lat is None or lon is None or utc_offset is None:
                raise ValueError(
                    "lat, lon, and utc_offset are required for daytime/nighttime "
                    "detection (separate_daytime_nighttime=True)"
                )

            self.daytime_minmax = daytime_minmax
            self.nighttime_minmax = nighttime_minmax

            # Detect daytime and nighttime
            self.flag_daytime, flag_nighttime, self.is_daytime, self.is_nighttime = (
                create_daytime_nighttime_flags(
                    timestamp_index=self.series.index,
                    lat=lat, lon=lon, utc_offset=utc_offset
                )
            )
        else:
            # Global mode
            if minval is None or maxval is None:
                raise ValueError(
                    "minval and maxval are required when separate_daytime_nighttime=False"
                )

            self.minval = minval
            self.maxval = maxval

    def calc(self, repeat: bool = False):
        """Calculate overall flag based on value limits.

        Args:
            repeat: If True, outlier detection is repeated until all outliers
                are removed (only applies to day/night mode).
        """
        if self.separate_daytime_nighttime:
            self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
            if self.showplot:
                self.defaultplot(n_iterations=n_iterations)
                title = (f"Absolute limits filter daytime/nighttime: {self.series.name}, "
                         f"n_iterations = {n_iterations}, "
                         f"n_outliers = {self.series[self.overall_flag == 2].count()}")
                self.plot_outlier_daytime_nighttime(
                    series=self.series, flag_daytime=self.flag_daytime,
                    flag_quality=self.overall_flag, title=title
                )
        else:
            # Global mode: no iteration needed
            self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=False)
            if self.showplot:
                self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        if self.separate_daytime_nighttime:
            return self._flagtests_daytime_nighttime(iteration)
        else:
            return self._flagtests_global(iteration)

    def _flagtests_global(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Global absolute limits test"""
        ok = (self.series >= self.minval) & (self.series <= self.maxval)
        ok = ok[ok].index
        rejected = (self.series < self.minval) | (self.series > self.maxval)
        rejected = rejected[rejected].index
        n_outliers = len(rejected)

        return ok, rejected, n_outliers

    def _flagtests_daytime_nighttime(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Separate daytime/nighttime absolute limits test"""

        # Working data
        s = self.series.copy().dropna()
        flag = pd.Series(index=self.series.index, data=np.nan)

        # Run for daytime (dt)
        _s_dt = s[self.is_daytime].copy()  # Daytime data
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
