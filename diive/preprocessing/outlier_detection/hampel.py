"""
Outlier detection using the Hampel filter (Median Absolute Deviation).

This module provides robust, window-based outlier detection using the Hampel filter,
which relies on the Median Absolute Deviation (MAD). Two modes are available:

- **Global Mode:** Single threshold applied to entire time series
  Fast, simple outlier detection for any data.

- **Daytime/Nighttime Mode:** Separate thresholds for different times of day
  Useful when data characteristics vary significantly by time-of-day conditions.

Both modes support:
  - Double-differencing (Papale et al. 2006 method) to remove trends
  - Raw value detection (if trends are not a concern)
  - Iterative filtering until all outliers removed or single-pass detection

Quality flags:
  - flag=0: Value within acceptable range (valid)
  - flag=2: Value detected as outlier (removed)
  - NaN: Record missing in the input, so no test could be performed

See examples/preprocessing/outlier_detection/hampel.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series

from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import VERBOSE_PROGRESS, detail, warn
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.preprocessing.outlier_detection.common import create_daytime_nighttime_flags, reject_legacy_params


@ConsoleOutputDecorator()
class Hampel(FlagBase):
    """Robust outlier detection using the Hampel filter (Median Absolute Deviation).

    The Hampel filter identifies outliers as values that deviate significantly from
    the local median, making it robust to spikes and extreme values while preserving
    underlying patterns. Works in two modes:

    - **Global Mode:** Single threshold for entire time series (simple, fast)
    - **Daytime/Nighttime Mode:** Separate thresholds for different times of day
      (useful when data characteristics vary by time-of-day conditions)

    Optional double-differencing (Papale et al. 2006) removes trends before detection.

    Example:
        See `examples/preprocessing/outlier_detection/hampel.py` for complete examples.
    """

    flagid = 'OUTLIER_HAMPEL'

    def __init__(self,
                 series: Series,
                 lat: float = None,
                 lon: float = None,
                 utc_offset: int = None,
                 window_length: int = 48 * 13,
                 n_sigma: float = 5.5,
                 n_sigma_daytime: float = None,
                 n_sigma_nighttime: float = None,
                 k: float = 1.4826,
                 use_differencing: bool = True,
                 separate_day_night: bool = True,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False,
                 **legacy):
        """
        Initialize Hampel filter for outlier detection.

        The filter detects outliers using Median Absolute Deviation (MAD) with optional
        day/night separation. Two analysis modes:

        1. **Double-Differencing:** If ``use_differencing=True``, applies filter to
           double-differenced data ($d = (x_t - x_{t-1}) - (x_{t+1} - x_t)$).
           Removes trends and isolates short-term deviations/spikes.
        2. **Day/Night Separation:** If ``separate_day_night=True``, uses solar elevation
           to apply different thresholds for daytime and nighttime periods.

        The general formula for the detection interval is:
        $$Limit = \\text{Median} \\pm (n\\_sigma \\times k \\times MAD)$$

        Args:
            series (pd.Series): The time series to analyze.
            lat (float): Latitude of the site (Required if ``separate_day_night=True``).
            lon (float): Longitude of the site (Required if ``separate_day_night=True``).
            utc_offset (int): UTC offset in hours (Required if ``separate_day_night=True``).
            window_length (int): The size of the sliding window centered on the point,
                expressed as a record count (not a duration). Default is
                ``48 * 13 = 624`` records, which equals **13 days at the
                half-hourly (30-min) sampling rate** typical of eddy-covariance
                data — matching the Papale et al. 2006 spike-detection window.
                Scale for other sampling rates (e.g. ``24 * 13`` for hourly).
            n_sigma (float): The number of standard deviations for the threshold.
                Default is 5.5. Used for:
                * **Global mode (separate_day_night=False):** Applied to all records.
                * **Day/Night mode:** Default for both daytime and nighttime (can be overridden).
            n_sigma_daytime (float, optional): Override ``n_sigma`` for daytime records only.
                Only used if ``separate_day_night=True``. If not provided, uses ``n_sigma``.
            n_sigma_nighttime (float, optional): Override ``n_sigma`` for nighttime records only.
                Only used if ``separate_day_night=True``. If not provided, uses ``n_sigma``.
            k (float): Consistency constant to make MAD comparable to Standard Deviation.
                For a Gaussian distribution, $k \approx 1.4826$.
            use_differencing (bool): If ``True``, applies the filter to the double-differenced
                time series (Papale et al. 2006 method). If ``False``, applies to raw values.
            separate_day_night (bool): If ``True``, splits the dataset based on solar elevation
                and applies different thresholds for daytime and nighttime.
            idstr (str, optional): Identifier suffix added to output variable names.
            showplot (bool): If ``True``, displays a summary plot after calculation.
            verbose (bool): If ``True``, prints iteration statistics to the console.

        References:
            * Papale, D. et al. (2006). "Towards a standardized processing of Net Ecosystem Exchange
              measured with eddy covariance technique: algorithms and uncertainty estimation".
              Biogeosciences, 3(4), 571-583.
            * Hampel F. R. (1974). "The influence curve and its role in robust estimation".
              Journal of the American Statistical Association, 69, 382-393.

        Kudos:
            * https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.transformations.series.outlier_detection.HampelFilter.html
            * https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
            * https://medium.com/@miguel.otero.pedrido.1993/hampel-filter-with-python-17db1d265375

        """

        reject_legacy_params(legacy, 'Hampel')
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = showplot
        self.verbose = verbose
        self.window_length = window_length
        self.n_sigma = n_sigma
        # Per-period overrides default to None and fall back to the global value,
        # so changing n_sigma alone still affects both periods.
        self.n_sigma_daytime = n_sigma_daytime if n_sigma_daytime is not None else n_sigma
        self.n_sigma_nighttime = n_sigma_nighttime if n_sigma_nighttime is not None else n_sigma
        self.k = k
        self.use_differencing = use_differencing
        self.separate_day_night = separate_day_night
        # Per-iteration detection band in data units (set by _flagtests); exposed
        # for visualisation. Series over the current cleaned series' index.
        self.last_upper_bound = None
        self.last_lower_bound = None

        # Records left undecided because the local MAD was exactly zero, summed
        # over iterations. Reported after calc() so a degenerate window is visible
        # rather than silent.
        self._n_degenerate_scale = 0

        # Records whose double difference would reach across a gap in the INPUT
        # data. Computed once, from the original series, so that it stays fixed
        # across iterations: values removed by an earlier iteration must not make
        # their neighbours untestable, or a cluster of spikes would shelter itself
        # after the first pass.
        self._untestable = self._gap_flanking_records(self.series)

        # Detect daytime and nighttime
        if self.separate_day_night:
            if lat is None or lon is None or utc_offset is None:
                raise ValueError("If 'separate_day_night' is True, you must provide lat, lon, and utc_offset.")

            self.flag_daytime, _, self.is_daytime, self.is_nighttime = (
                create_daytime_nighttime_flags(timestamp_index=self.series.index,
                                               lat=lat, lon=lon, utc_offset=utc_offset))
        else:
            # Initialize empty/None to avoid attribute errors if accessed later
            self.is_daytime = None
            self.is_nighttime = None
            self.flag_daytime = None

    def calc(self, repeat: bool = True, progress_callback=None):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.
            progress_callback: Optional ``callable(iteration, n_outliers,
                filteredseries)`` invoked after each iteration (e.g. to drive a
                progress bar / live-update the cleaned series).

        """
        self._overall_flag, n_iterations = self.repeat(
            func=self.run_flagtests, repeat=repeat, progress_callback=progress_callback)

        if self._n_degenerate_scale:
            warn(f"Hampel: {self._n_degenerate_scale} record(s) could not be judged because the "
                 f"local MAD was exactly zero (more than half the window identical) and were left "
                 f"unflagged. Typical causes: coarser data upsampled onto a finer grid, quantized "
                 f"readings, or a stuck sensor. Consider a longer window_length, "
                 f"use_differencing=False, or screening the affected period separately.",
                 verbose=self.verbose)

        if self.showplot:
            # Default plot for outlier tests, showing rejected values
            self.defaultplot(n_iterations=n_iterations)
            mode = "daytime/nighttime" if self.separate_day_night else "global"
            title = (f"Hampel filter {mode}: {self.series.name}, "
                     f"n_iterations = {n_iterations}, "
                     f"n_outliers = {self.series[self.overall_flag == 2].count()}")

            if self.separate_day_night:
                self.plot_outlier_daytime_nighttime(series=self.series,
                                                    flag_daytime=self.flag_daytime,
                                                    flag_quality=self.overall_flag,
                                                    title=title)

    @staticmethod
    def _gap_flanking_records(series: Series) -> Series:
        """Records whose immediate neighbour is missing in the input data.

        The double difference at record *t* uses both of its neighbours, so *t*
        cannot be judged when either of them is absent: dropping missing records
        before differencing would silently pair *t* with a partner hours or days
        away and make every gap edge look like a spike.

        Returns a boolean Series over the input index (all False when the index
        is not a usable time axis, which leaves the previous behaviour in place).
        """
        index = series.index
        if not isinstance(index, DatetimeIndex) or len(index) < 3:
            return pd.Series(False, index=index)

        if index.freq is not None:
            step = pd.Timedelta(index.freq.nanos, unit='ns')
        else:
            step = pd.Series(index).diff().median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            return pd.Series(False, index=index)

        # A neighbour is missing either because the timestamp itself is absent
        # (irregular index) or because it carries no value (gap on a regular grid).
        steps = pd.Series(index).diff()
        far_before = (steps > step * 1.5).to_numpy()
        far_after = np.append(far_before[1:], False)
        empty = series.isna().to_numpy()
        empty_before = np.append(True, empty[:-1])
        empty_after = np.append(empty[1:], True)
        return pd.Series(far_before | far_after | empty_before | empty_after, index=index)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag using optimized Pandas operations."""

        # Prepare data
        s = self.filteredseries.copy().dropna()

        # 2. Transform data
        if self.use_differencing:
            # d = (x_t - x_{t-1}) - (x_{t+1} - x_t)
            s_to_test = s.diff() - s.diff().shift(-1)
            s_to_test = s_to_test.fillna(0)
            # Missing records were dropped above, so consecutive entries can be hours
            # or days apart. A difference taken across such a gap compares unrelated
            # records and makes the two records flanking every gap look like spikes.
            # Neutralize those (mask computed once from the input, see __init__).
            s_to_test = s_to_test.mask(self._untestable.reindex(s.index, fill_value=False))
        else:
            s_to_test = s

        # Calculate rolling stats (vectorized on the whole series)
        # This is much faster than splitting day/night first
        rolling_median = s_to_test.rolling(window=self.window_length, center=True, min_periods=1).median()

        # MAD (Median Absolute Deviation) calculation
        deviations = np.abs(s_to_test - rolling_median)
        rolling_mad = deviations.rolling(window=self.window_length, center=True, min_periods=1).median()

        # A window in which more than half the records are identical has a MAD of
        # exactly zero, and then the detection band has zero width: every value that
        # differs from the local median at all becomes an outlier, however small the
        # difference. That is not a strict filter, it is an undefined one - the data
        # carry no scale to judge against - and substituting a tiny epsilon turns it
        # into a silent mass rejection of the signal itself. Windows that arise from
        # upsampled coarse data, quantized readings or a stuck sensor hit this
        # routinely. Such records are therefore left unflagged (NaN limits compare
        # False), and the count is reported rather than hidden.
        degenerate = rolling_mad == 0
        if degenerate.any():
            self._n_degenerate_scale += int(degenerate.sum())
        rolling_mad = rolling_mad.where(~degenerate)

        # Define thresholds
        if self.separate_day_night:
            # Create a series of thresholds matching the data index
            # Default to nighttime threshold
            thresholds = pd.Series(data=self.n_sigma_nighttime, index=s_to_test.index)
            # Overwrite daytime indices with daytime threshold
            current_daytime = self.is_daytime.reindex(s_to_test.index, fill_value=False)
            thresholds.loc[current_daytime] = self.n_sigma_daytime
        else:
            # Global mode, single threshold for all everything
            thresholds = self.n_sigma

        # Detect outliers
        # Limit = k * MAD * n_sigma
        # k = 1.4826 (scaling factor for Gaussian consistency)
        limit = self.k * rolling_mad * thresholds
        upper_bound = rolling_median + limit
        lower_bound = rolling_median - limit

        # NaN limits (degenerate scale) and NaN test values (differences spanning a
        # gap) both compare False here, i.e. no decision is made for those records.
        is_outlier = ((s_to_test > upper_bound) | (s_to_test < lower_bound)).fillna(False)

        # Expose the per-iteration detection band in DATA units (for visualisation).
        # Raw mode: the bounds already are in data units. Double-differencing mode:
        # the test runs on d = 2*x_t - (x_{t-1} + x_{t+1}), so x_t is flagged iff it
        # leaves [lower, upper] mapped to data units as neighbour_avg + bound/2
        # (neighbour_avg = (x_{t-1} + x_{t+1}) / 2). This is the exact data-space
        # band the flag decision uses for the current iteration.
        if self.use_differencing:
            neighbour_avg = (s.shift(1) + s.shift(-1)) / 2.0
            self.last_upper_bound = neighbour_avg + upper_bound / 2.0
            self.last_lower_bound = neighbour_avg + lower_bound / 2.0
        else:
            self.last_upper_bound = upper_bound.copy()
            self.last_lower_bound = lower_bound.copy()

        # Formatting for return
        # Get indices of True/False
        ok = is_outlier[~is_outlier].index
        rejected = is_outlier[is_outlier].index
        n_outliers = len(rejected)

        # Note: FlagBase handles the actual '2' assignment
        # based on the returned 'rejected' index list, so we just return indices

        # Reporting
        if self.verbose:
            # Calculate total valid points in this iteration to get percentages
            n_total_valid = len(s_to_test)
            pct_total = (n_outliers / n_total_valid * 100) if n_total_valid > 0 else 0.0

            # Formatting helpers: ensures numbers align perfectly in the console
            iter_str = f"ITER #{iteration:02d}"  # e.g., "ITER #01"
            out_str = f"{n_outliers:>5}"  # Right-aligned count, e.g., "  123"
            pct_str = f"{pct_total:>6.2f}%"  # Fixed width percentage, e.g., " 12.34%"

            if self.separate_day_night:
                # 1. Align mask
                is_daytime_aligned = self.is_daytime.reindex(is_outlier.index, fill_value=False)

                # 2. Counts
                n_dt = (is_outlier & is_daytime_aligned).sum()
                n_nt = n_outliers - n_dt

                # 3. Print beautiful aligned output
                # Example: [Dt/Nt] ITER #01 | Outliers:   123 ( 0.45%) | Day:    50 | Night:    73
                detail(f"[Dt/Nt] {iter_str} | "
                       f"Outliers: {out_str} ({pct_str}) | "
                       f"Day: {n_dt:>5} | "
                       f"Night: {n_nt:>5}", verbose=self.verbose, min_level=VERBOSE_PROGRESS)
            else:
                # Global reporting
                # Example: [Global] ITER #01 | Outliers:   123 ( 0.45%)
                detail(f"[Global] {iter_str} | "
                       f"Outliers: {out_str} ({pct_str})", verbose=self.verbose, min_level=VERBOSE_PROGRESS)

        return ok, rejected, n_outliers


# Backward compatibility alias
HampelDaytimeNighttime = Hampel
