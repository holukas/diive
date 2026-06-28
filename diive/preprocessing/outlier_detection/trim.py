"""
TRIM LOW: SYMMETRIC OUTLIER REMOVAL
====================================

Detect outliers using trimmed mean approach: remove values below threshold, then remove equal number from high end.

Part of the diive library: https://github.com/holukas/diive

How it works:

By default the whole series is trimmed against one distribution: it counts how
many values fall below ``lower_limit``, rejects them, and then rejects the same
number of the highest values. Removing equal counts from both tails keeps the
distribution symmetric, mirroring the trimmed-mean statistic.

Day/night screening is optional. ``trim_daytime`` / ``trim_nighttime`` restrict
the trim to those periods, each screened against its own distribution (location
parameters are then required to split day from night). When both are enabled,
each period is trimmed independently and the flags are combined; when neither is
set, no coordinates are needed.

Use cases:

  - Symmetrically trimming both tails so a downstream mean/variance is not
    skewed by extreme values on either end (the trimmed-mean rationale).
  - Removing physically impossible low spikes (e.g. sensor dropouts below a
    known floor) while keeping the remaining data balanced by also discarding
    the matching count of suspicious high values.
  - Day/night-aware cleaning, where the plausible range differs between daytime
    and nighttime (e.g. radiation, fluxes, temperature) and each period should
    be screened against its own distribution.
  - Quick, deterministic pre-screening: the trim is single-pass and threshold-
    driven, so it is predictable and fast compared to iterative or model-based
    detectors, making it a useful first step before finer outlier methods.

When NOT to use:

  - When you only want to drop low (or only high) extremes — this method always
    removes an equal count from the opposite tail, so it discards otherwise-valid
    high values. Use a one-sided method (e.g. AbsoluteLimits) instead.
  - When more than half a subset lies below ``lower_limit``: the two tails
    overlap and the trim degenerates toward rejecting the whole subset.

Quality flags (overall_flag):
  - flag=0: Value within acceptable range (valid); also assigned to originally-missing records
  - flag=2: Value detected as outlier (removed)

The filtered series (.filteredseries) preserves the original NaNs and, in addition, sets
rejected values to NaN.

See examples/preprocessing/outlier_detection/outlier_trim.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series

from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import detail
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.preprocessing.outlier_detection.common import create_daytime_nighttime_flags


@ConsoleOutputDecorator()
class TrimLow(FlagBase):
    """Flag the lowest values (and an equal count of the highest) as outliers. See :meth:`__init__`."""

    flagid = 'OUTLIER_TRIMLOW'

    def __init__(self,
                 series: Series,
                 lower_limit: float = None,
                 trim_daytime: bool = False,
                 trim_nighttime: bool = False,
                 lat: float = None,
                 lon: float = None,
                 utc_offset: int = None,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Trim outliers using symmetric removal (trimmed mean approach).

        Removes values below ``lower_limit``, then removes an equal number of the
        highest values. By default the whole series is trimmed against one
        distribution; the optional ``trim_daytime`` / ``trim_nighttime`` flags
        restrict (and split) the trim to those periods instead, screening each
        against its own distribution.

        Example:
            See `examples/preprocessing/outlier_detection/outlier_trim.py` for complete examples.

        Args:
            series: Time series in which outliers are identified.
            lower_limit: Value below which values are considered outliers.
            trim_daytime: If True, trim daytime data (against its own distribution).
            trim_nighttime: If True, trim nighttime data (against its own distribution).
                When both ``trim_daytime`` and ``trim_nighttime`` are False (the
                default), the whole series is trimmed as one distribution and no
                location parameters are needed.
            lat: Latitude of location (e.g., 46.583056). Only required when
                ``trim_daytime`` or ``trim_nighttime`` is True (to detect day/night).
            lon: Longitude of location (e.g., 9.790639). Only required when
                ``trim_daytime`` or ``trim_nighttime`` is True.
            utc_offset: UTC offset in hours (e.g., 1 for UTC+01:00). Only required
                when ``trim_daytime`` or ``trim_nighttime`` is True.
            idstr: Identifier suffix for output variable names.
            showplot: If True, display results plot.
            verbose: If True, print iteration statistics.

        After running ``.calc()`` (or ``.run()``), the flag is available via
        ``.overall_flag`` (2=outlier, 0=valid) and the cleaned data via
        ``.filteredseries``.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)

        # Validate inputs
        if lower_limit is None:
            raise ValueError('lower_limit must be specified (not None).')
        # Day/night is opt-in. Location parameters are only needed when a split is
        # requested; without one, the whole series is trimmed (no coordinates).
        self._separate = trim_daytime or trim_nighttime
        if self._separate and (lat is None or lon is None or utc_offset is None):
            raise ValueError('Location parameters (lat, lon, utc_offset) are required '
                             'when trim_daytime or trim_nighttime is True.')

        self.showplot = showplot
        self.verbose = verbose
        self.trim_daytime = trim_daytime
        self.trim_nighttime = trim_nighttime
        self.lower_limit = lower_limit

        # Detector-interface contract (consumed by the GUI's outlier base): this
        # method has no single data-unit detection band — the symmetric trim
        # rejects the lowest values plus an equal count of the highest *by
        # position*, so the kept set is not a `[lower, upper]` envelope (like the
        # increments method, the bounds stay None and no band overlay is drawn).
        self.last_lower_bound = None
        self.last_upper_bound = None

        # Detect daytime and nighttime only when a split is requested; otherwise
        # the whole series is trimmed and no day/night classification is needed.
        if self._separate:
            self.flag_daytime, _, self.is_daytime, self.is_nighttime = (
                create_daytime_nighttime_flags(timestamp_index=self.series.index,
                                               lat=lat, lon=lon, utc_offset=utc_offset))
        else:
            self.flag_daytime = self.is_daytime = self.is_nighttime = None

    def calc(self, repeat: bool = False, progress_callback=None):
        """Calculate overall flag based on trim filter threshold testing.

        Single-pass outlier detection by default (not iterative). Removes values
        below lower_limit, then removes an equal number of values from the high end.

        Args:
            repeat: If True, repeat the trim until an iteration finds no more
                outliers. Defaults to False (single pass), which is the intended
                use for the symmetric trim.
            progress_callback: Optional ``callable(iteration, n_outliers,
                filteredseries)`` forwarded to ``repeat`` (e.g. to drive a GUI
                progress indicator).
        """
        self._overall_flag, n_iterations = self.repeat(
            func=self.run_flagtests, repeat=repeat, progress_callback=progress_callback)

        if self.showplot:
            # Determine filtering mode for plot title
            if self.trim_daytime and self.trim_nighttime:
                mode = "daytime/nighttime"
            elif self.trim_daytime:
                mode = "daytime only"
            elif self.trim_nighttime:
                mode = "nighttime only"
            else:
                mode = "all data"

            title = (f"TrimLow filter {mode}: {self.series.name}, "
                     f"n_outliers = {self.series[self.overall_flag == 2].count()}")

            # Only the both-modes case has a meaningful day/night separation to
            # visualize; the single-mode case gets one default before/after plot.
            if self.trim_daytime and self.trim_nighttime:
                self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                    flag_quality=self.overall_flag, title=title)
            else:
                self.defaultplot(n_iterations=n_iterations, title=title)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy()
        s = s.dropna()

        flag = pd.Series(index=s.index, data=np.nan)

        # Determine which subsets to trim. When both daytime and nighttime are
        # enabled, each is trimmed independently against its own distribution and
        # the flags are combined; with a single flag, only that subset is trimmed.
        # When neither is set, the whole series is trimmed as one distribution.
        # (A previous if/elif silently left nighttime unscreened when both modes
        # were True.)
        subsets = []
        if self.trim_daytime:
            subsets.append(s[self.is_daytime].copy())
        if self.trim_nighttime:
            subsets.append(s[self.is_nighttime].copy())
        if not subsets:
            subsets.append(s.copy())

        for _s in subsets:
            self._trim_subset(_s, flag)

        flag = flag.fillna(0)

        n_outliers = (flag == 2).sum()
        ok = (flag == 0)
        ok = ok[ok].index
        rejected = (flag == 2)
        rejected = rejected[rejected].index

        if self.verbose:
            kept = s.loc[flag == 0]
            detail(f"ITERATION#{iteration}: Total found outliers: "
                   f"{n_outliers}, "
                   f"minimum value: {kept.min()}, "
                   f"maximum value: {kept.max()}", verbose=self.verbose)

        return ok, rejected, n_outliers

    def _trim_subset(self, _s: Series, flag: Series) -> None:
        """Apply the symmetric trim to one subset, writing 0 (valid) / 2 (outlier)
        into ``flag`` at the subset's indices. Modifies ``flag`` in place."""
        s_sorted = _s.sort_values(ascending=False)
        s_sorted_below = s_sorted.loc[s_sorted < self.lower_limit].copy()
        n_vals_below = s_sorted_below.count()

        # Handle case where no values fall below threshold
        if n_vals_below == 0:
            # No low outliers found, all values are valid
            flag.loc[_s.index] = 0
        else:
            # Symmetric trim: reject the values below lower_limit, plus an equal
            # number (n_vals_below) of the highest values. Reject by POSITION
            # rather than by an upper-limit value threshold, so ties at the
            # boundary don't reject more (or fewer) than the intended count and
            # the kept/rejected sets stay strictly complementary.
            # Edge case: if more than half the subset is below lower_limit, the
            # high-end positions overlap the low set; .union dedupes, so fewer
            # than 2*n_vals_below unique records are rejected (degenerating to
            # "reject everything" when the whole subset is below the limit).
            low_idx = _s.index[_s < self.lower_limit]
            high_idx = s_sorted.iloc[0:n_vals_below].index  # s_sorted is descending
            rejected_idx = low_idx.union(high_idx)

            flag.loc[_s.index] = 0
            flag.loc[rejected_idx] = 2
