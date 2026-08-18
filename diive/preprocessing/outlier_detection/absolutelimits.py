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
  - NaN: Record missing in the input, so no test could be performed

See examples/preprocessing/outlier_detection/outlier_absolutelimits.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import VERBOSE_PROGRESS, detail
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.preprocessing.outlier_detection.common import create_daytime_nighttime_flags, reject_legacy_params


@ConsoleOutputDecorator()
class AbsoluteLimits(FlagBase):
    """Outlier detection using absolute value limits.

    Identifies values outside specified acceptable range(s). Can operate in two modes:

    1. **Global Mode (separate_day_night=False):**
       Single threshold range applied to all data. Fast, simple validation.

    2. **Daytime/Nighttime Mode (separate_day_night=True):**
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
    - NaN: Record missing in the input, so no test could be performed

    Example:
        See `examples/preprocessing/outlier_detection/outlier_absolutelimits.py` for complete examples.
    """

    flagid = 'OUTLIER_ABSLIM'

    def __init__(self,
                 series: Series,
                 minval: float = None,
                 maxval: float = None,
                 separate_day_night: bool = False,
                 minval_daytime: float = None,
                 maxval_daytime: float = None,
                 minval_nighttime: float = None,
                 maxval_nighttime: float = None,
                 lat: float = None,
                 lon: float = None,
                 utc_offset: int = None,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False,
                 **legacy):
        """
        Initialize absolute limits outlier detector.

        Args:
            series: Time series in which outliers are identified.
            minval: Minimum acceptable value (global mode). Required if
                separate_day_night=False.
            maxval: Maximum acceptable value (global mode). Required if
                separate_day_night=False.
            separate_day_night: If True, use separate day/night thresholds;
                if False, use global thresholds. Default False.
            minval_daytime: Override ``minval`` for daytime records only. If None,
                uses ``minval``. Setting it turns ``separate_day_night`` on.
            maxval_daytime: Override ``maxval`` for daytime records only. If None,
                uses ``maxval``. Setting it turns ``separate_day_night`` on.
            minval_nighttime: Override ``minval`` for nighttime records only. If
                None, uses ``minval``. Setting it turns ``separate_day_night`` on.
            maxval_nighttime: Override ``maxval`` for nighttime records only. If
                None, uses ``maxval``. Setting it turns ``separate_day_night`` on.
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
        reject_legacy_params(legacy, 'AbsoluteLimits')
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)

        # Setting any per-period limit turns the split on, so correct usage works
        # whether or not the caller also passes separate_day_night.
        _overrides = (minval_daytime, maxval_daytime, minval_nighttime, maxval_nighttime)
        if any(v is not None for v in _overrides):
            separate_day_night = True

        self.separate_day_night = separate_day_night
        self.showplot = showplot
        self.verbose = verbose

        # Per-period limits default to None and fall back to the global limits,
        # so minval/maxval alone apply to both periods.
        self.minval_daytime = minval_daytime if minval_daytime is not None else minval
        self.maxval_daytime = maxval_daytime if maxval_daytime is not None else maxval
        self.minval_nighttime = minval_nighttime if minval_nighttime is not None else minval
        self.maxval_nighttime = maxval_nighttime if maxval_nighttime is not None else maxval

        # Per-iteration detection band in data units (set by _flagtests); exposed
        # for visualisation. For absolute limits the band is the fixed min/max, so
        # it is constant across iterations.
        self.last_upper_bound = None
        self.last_lower_bound = None
        self.is_daytime = None  # global mode; day/night branch overrides below

        if separate_day_night:
            # Day/night mode. Each period needs a resolved pair, which is either
            # its own override or the global limit it fell back to above.
            _missing = [n for n, v in (('minval_daytime', self.minval_daytime),
                                       ('maxval_daytime', self.maxval_daytime),
                                       ('minval_nighttime', self.minval_nighttime),
                                       ('maxval_nighttime', self.maxval_nighttime))
                        if v is None]
            if _missing:
                raise ValueError(
                    f"no limit for {', '.join(_missing)} when separate_day_night=True. "
                    f"Set minval and maxval to cover both periods, or give the "
                    f"per-period limits explicitly."
                )
            if lat is None or lon is None or utc_offset is None:
                raise ValueError(
                    "lat, lon, and utc_offset are required for daytime/nighttime "
                    "detection (separate_day_night=True)"
                )

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
                    "minval and maxval are required when separate_day_night=False"
                )

            self.minval = minval
            self.maxval = maxval

    def calc(self, repeat: bool = False, progress_callback=None):
        """Calculate overall flag based on value limits.

        Args:
            repeat: If True, outlier detection is repeated until all outliers
                are removed (only applies to day/night mode).
            progress_callback: Optional ``callable(iteration, n_outliers,
                filteredseries)`` invoked after each iteration (e.g. to drive a
                progress bar / live-update the cleaned series).
        """
        if self.separate_day_night:
            self._overall_flag, n_iterations = self.repeat(
                self.run_flagtests, repeat=repeat, progress_callback=progress_callback)
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
            self._overall_flag, n_iterations = self.repeat(
                self.run_flagtests, repeat=False, progress_callback=progress_callback)
            if self.showplot:
                self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        if self.separate_day_night:
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

        # Detection band in DATA units (for visualisation): the fixed limits as
        # constant lines over the series index.
        self.last_lower_bound = pd.Series(data=self.minval, index=self.series.index)
        self.last_upper_bound = pd.Series(data=self.maxval, index=self.series.index)

        return ok, rejected, n_outliers

    def _flagtests_daytime_nighttime(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Separate daytime/nighttime absolute limits test"""

        # Working data
        s = self.series.copy().dropna()
        flag = pd.Series(index=self.series.index, data=np.nan)

        # Run for daytime (dt)
        _s_dt = s[self.is_daytime].copy()  # Daytime data
        _ok_dt = (_s_dt >= self.minval_daytime) & (_s_dt <= self.maxval_daytime)
        _ok_dt = _ok_dt[_ok_dt].index
        _rejected_dt = (_s_dt < self.minval_daytime) | (_s_dt > self.maxval_daytime)
        _rejected_dt = _rejected_dt[_rejected_dt].index

        # Run for nighttime (nt)
        _s_nt = s[self.is_nighttime].copy()  # Nighttime data
        _ok_nt = (_s_nt >= self.minval_nighttime) & (_s_nt <= self.maxval_nighttime)
        _ok_nt = _ok_nt[_ok_nt].index
        _rejected_nt = (_s_nt < self.minval_nighttime) | (_s_nt > self.maxval_nighttime)
        _rejected_nt = _rejected_nt[_rejected_nt].index

        # Collect daytime and nighttime flags in one overall flag
        flag.loc[_ok_dt] = 0
        flag.loc[_rejected_dt] = 2
        flag.loc[_ok_nt] = 0
        flag.loc[_rejected_nt] = 2

        n_outliers = (flag == 2).sum()

        # Per-record detection band in DATA units: daytime records carry the daytime
        # limits, nighttime records the nighttime limits (combined over the union
        # index for day/night-coloured visualisation).
        self.last_lower_bound = pd.concat([
            pd.Series(self.minval_daytime, index=_s_dt.index),
            pd.Series(self.minval_nighttime, index=_s_nt.index)]).sort_index()
        self.last_upper_bound = pd.concat([
            pd.Series(self.maxval_daytime, index=_s_dt.index),
            pd.Series(self.maxval_nighttime, index=_s_nt.index)]).sort_index()

        ok = (flag == 0)
        ok = ok[ok].index
        rejected = (flag == 2)
        rejected = rejected[rejected].index

        if self.verbose:
            detail(f"Total found outliers: {len(_rejected_dt)} values (daytime)", verbose=self.verbose, min_level=VERBOSE_PROGRESS)
            detail(f"Total found outliers: {len(_rejected_nt)} values (nighttime)", verbose=self.verbose, min_level=VERBOSE_PROGRESS)
            detail(f"Total found outliers: {n_outliers} values (daytime+nighttime)", verbose=self.verbose, min_level=VERBOSE_PROGRESS)

        return ok, rejected, n_outliers


def AbsoluteLimitsDaytimeNighttime(*args, separate_day_night: bool = True, **kwargs):
    """``AbsoluteLimits`` with daytime/nighttime separation on by default.

    This used to be a plain alias for ``AbsoluteLimits``, whose
    ``separate_day_night`` defaults to False. Picking this name for
    what it says therefore applied one set of limits to the whole series,
    with no error or warning.

    A wrapper function rather than a subclass because ``ConsoleOutputDecorator``
    replaces the decorated class with a function, which cannot be subclassed.
    Pass ``minval`` / ``maxval`` to cover both periods, and the
    ``*_daytime`` / ``*_nighttime`` overrides to differ, plus ``lat`` /
    ``lon`` / ``utc_offset``.
    """
    return AbsoluteLimits(*args, separate_day_night=separate_day_night, **kwargs)
