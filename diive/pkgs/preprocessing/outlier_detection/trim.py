"""
TRIM LOW: SYMMETRIC OUTLIER REMOVAL
====================================

Detect outliers using trimmed mean approach: remove values below threshold, then remove equal number from high end.

Part of the diive library: https://github.com/holukas/diive

Trim filter details:

This module provides outlier detection by removing values below a threshold,
then removing an equal number of values from the high end. This is based on
the trimmed mean approach and is useful for symmetric outlier removal.

Quality flags:
  - flag=0: Value within acceptable range (valid)
  - flag=2: Value detected as outlier (removed)
  - NaN: Original missing data preserved

See examples/preprocessing/outlier_detection/trim.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series

from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import detail
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.preprocessing.outlier_detection.common import create_daytime_nighttime_flags


@ConsoleOutputDecorator()
class TrimLow(FlagBase):
    flagid = 'OUTLIER_TRIMLOW'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 trim_daytime: bool = False,
                 trim_nighttime: bool = False,
                 lower_limit: float = None,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Trim outliers using symmetric removal (trimmed mean approach).

        Removes values below a threshold, then removes an equal number of values
        from the high end. Supports separate processing for daytime/nighttime data.

        Example:
            See `examples/preprocessing/outlier_detection/trim.py` for complete examples.

        Args:
            series: Time series in which outliers are identified.
            trim_daytime: If True, apply filtering to daytime data.
            trim_nighttime: If True, apply filtering to nighttime data.
            lower_limit: Value below which values are considered outliers.
            lat: Latitude of location (e.g., 46.583056).
                Used to detect daytime/nighttime.
            lon: Longitude of location (e.g., 9.790639).
                Used to detect daytime/nighttime.
            utc_offset: UTC offset in hours (e.g., 1 for UTC+01:00).
                Used to detect daytime/nighttime.
            idstr: Identifier suffix for output variable names.
            showplot: If True, display results plot.
            verbose: If True, print iteration statistics.

        Returns:
            Flag series where 2=outlier and 0=valid.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)

        # Validate inputs
        if not trim_daytime and not trim_nighttime:
            raise ValueError('Either trim_daytime or trim_nighttime must be True.')
        if lower_limit is None:
            raise ValueError('lower_limit must be specified (not None).')
        if lat is None or lon is None or utc_offset is None:
            raise ValueError('Location parameters (lat, lon, utc_offset) are required for day/night detection.')

        self.showplot = showplot
        self.verbose = verbose
        self.trim_daytime = trim_daytime
        self.trim_nighttime = trim_nighttime
        self.lower_limit = lower_limit

        # Detect daytime and nighttime
        self.flag_daytime, _, self.is_daytime, self.is_nighttime = (
            create_daytime_nighttime_flags(timestamp_index=self.series.index, lat=lat, lon=lon, utc_offset=utc_offset))

    def calc(self):
        """Calculate overall flag based on trim filter threshold testing.

        Single-pass outlier detection (not iterative). Removes values below
        lower_limit, then removes equal number of values from high end.
        """
        self._overall_flag, n_iterations = self.repeat(func=self.run_flagtests, repeat=False)

        if self.showplot:
            # Default plot for outlier tests, showing rejected values
            self.defaultplot(n_iterations=n_iterations)

            # Determine filtering mode for plot title
            if self.trim_daytime and self.trim_nighttime:
                mode = "daytime/nighttime"
            elif self.trim_daytime:
                mode = "daytime only"
            else:
                mode = "nighttime only"

            title = (f"TrimLow filter {mode}: {self.series.name}, "
                     f"n_outliers = {self.series[self.overall_flag == 2].count()}")

            # Only plot day/night visualization if both day and night are enabled
            if self.trim_daytime and self.trim_nighttime:
                self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                    flag_quality=self.overall_flag, title=title)
            else:
                # Single-mode plot (no day/night separation visualization)
                self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy()
        s = s.dropna()

        flag = pd.Series(index=s.index, data=np.nan)

        if self.trim_daytime:
            _s = s[self.is_daytime].copy()
        elif self.trim_nighttime:
            _s = s[self.is_nighttime].copy()
        else:
            raise ValueError('Either trim_daytime or trim_nighttime must be True.')

        s_sorted = _s.sort_values(ascending=False)
        s_sorted_below = s_sorted.loc[s_sorted < self.lower_limit].copy()
        n_vals_below = s_sorted_below.count()

        # Handle case where no values fall below threshold
        if n_vals_below == 0:
            # No low outliers found, all values are valid
            flag.loc[_s.index] = 0
        else:
            # Trim symmetric: remove top N values equal to number of low outliers
            s_sorted_top = s_sorted.iloc[0:n_vals_below].copy()
            upper_lim = s_sorted_top.iloc[-1]

            # Classify: keep values in [lower_limit, upper_lim), reject others
            _ok = (_s >= self.lower_limit) & (_s < upper_lim)
            _ok = _ok[_ok].index
            _rejected = (_s <= self.lower_limit) | (_s >= upper_lim)
            _rejected = _rejected[_rejected].index

            flag.loc[_ok] = 0
            flag.loc[_rejected] = 2

        flag = flag.fillna(0)

        n_outliers = (flag == 2).sum()
        ok = (flag == 0)
        ok = ok[ok].index
        rejected = (flag == 2)
        rejected = rejected[rejected].index

        if self.verbose:
            detail(f"ITERATION#{iteration}: Total found outliers: "
                   f"{n_outliers}, "
                   f"minimum value: {_s.loc[flag == 0].min()}, "
                   f"maximum value: {_s.loc[flag == 0].max()}", verbose=self.verbose)

        return ok, rejected, n_outliers
