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
  - NaN: Original missing data preserved

See examples/preprocessing/outlier_detection/hampel.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.preprocessing.outlier_detection.common import create_daytime_nighttime_flags


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
                 window_length: int = 13,
                 n_sigma: float = 5.5,
                 n_sigma_daytime: float = None,
                 n_sigma_nighttime: float = None,
                 n_sigma_dt: float = None,
                 n_sigma_nt: float = None,
                 k: float = 1.4826,
                 use_differencing: bool = True,
                 separate_day_night: bool = True,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
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
            window_length (int): The size of the sliding window centered on the point.
                Default is 13 (approx. 6 hours for half-hourly data).
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

        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = showplot
        self.verbose = verbose
        self.window_length = window_length
        self.n_sigma = n_sigma
        # Short-form aliases (n_sigma_dt/n_sigma_nt) take precedence over long-form
        _n_sigma_daytime = n_sigma_dt if n_sigma_dt is not None else n_sigma_daytime
        _n_sigma_nighttime = n_sigma_nt if n_sigma_nt is not None else n_sigma_nighttime
        self.n_sigma_daytime = _n_sigma_daytime if _n_sigma_daytime is not None else n_sigma
        self.n_sigma_nighttime = _n_sigma_nighttime if _n_sigma_nighttime is not None else n_sigma
        self.k = k
        self.use_differencing = use_differencing
        self.separate_day_night = separate_day_night

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

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """
        self._overall_flag, n_iterations = self.repeat(func=self.run_flagtests, repeat=repeat)

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

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag using optimized Pandas operations."""

        # Prepare data
        s = self.filteredseries.copy().dropna()

        # 2. Transform data
        if self.use_differencing:
            # d = (x_t - x_{t-1}) - (x_{t+1} - x_t)
            s_to_test = s.diff() - s.diff().shift(-1)
            s_to_test = s_to_test.fillna(0)
        else:
            s_to_test = s

        # Calculate rolling stats (vectorized on the whole series)
        # This is much faster than splitting day/night first
        rolling_median = s_to_test.rolling(window=self.window_length, center=True, min_periods=1).median()

        # MAD (Median Absolute Deviation) calculation
        deviations = np.abs(s_to_test - rolling_median)
        rolling_mad = deviations.rolling(window=self.window_length, center=True, min_periods=1).median()

        # Add epsilon to avoid zero-division issues on flat signals
        rolling_mad = rolling_mad + 1e-6

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

        is_outlier = (s_to_test > upper_bound) | (s_to_test < lower_bound)

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
                print(f"[Dt/Nt] {iter_str} | "
                      f"Outliers: {out_str} ({pct_str}) | "
                      f"Day: {n_dt:>5} | "
                      f"Night: {n_nt:>5}")
            else:
                # Global reporting
                # Example: [Global] ITER #01 | Outliers:   123 ( 0.45%)
                print(f"[Global] {iter_str} | "
                      f"Outliers: {out_str} ({pct_str})")

        return ok, rejected, n_outliers


# Backward compatibility alias
HampelDaytimeNighttime = Hampel
