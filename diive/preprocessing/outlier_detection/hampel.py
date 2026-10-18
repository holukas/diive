"""
HAMPEL: ROLLING MEDIAN ABSOLUTE DEVIATION FILTER
================================================

Robust, window-based outlier detection with the Hampel filter, which relies on
the Median Absolute Deviation (MAD). Two modes are available:

- **Global Mode:** Single threshold applied to entire time series
  Fast, simple outlier detection for any data.

- **Daytime/Nighttime Mode:** daytime records are judged against the median and
  MAD of the daytime records in the window, nighttime records against those of
  the nighttime records, each with its own threshold (as ONEFlux qc_auto,
  Papale et al. 2006). Useful when the variability differs between day and
  night, e.g. radiation or fluxes.

Both modes support:
  - Double-differencing (Papale et al. 2006 method) to remove trends
  - Raw value detection (if trends are not a concern)
  - Iterative filtering until all outliers removed or single-pass detection

Quality flags:
  - flag=0: Value within acceptable range (valid)
  - flag=2: Value detected as outlier (removed)
  - NaN: Record missing in the input, so no test could be performed

See examples/preprocessing/outlier_detection/outlier_hampel.py for working examples.

Part of the diive library: https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series

from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import VERBOSE_PROGRESS, detail, warn
from diive.preprocessing.outlier_detection.common import (create_daytime_nighttime_flags, reject_legacy_params,
                                                          window_to_records)


class Hampel(FlagBase):
    """Robust outlier detection using the Hampel filter (Median Absolute Deviation).

    The Hampel filter identifies outliers as values that deviate significantly from
    the local median, making it robust to spikes and extreme values while preserving
    underlying patterns. Works in two modes:

    - **Global Mode:** Single threshold for entire time series (simple, fast)
    - **Daytime/Nighttime Mode:** separate median, MAD and threshold for daytime
      and nighttime records, as in ONEFlux (useful when the variability differs
      between day and night)

    Optional double-differencing (Papale et al. 2006) removes trends before detection.

    With double-differencing the filter finds single spikes and pairs of
    adjacent spikes, also at the series ends and next to gaps. A large spike
    usually costs its two neighbours too, because their double differences
    contain the spike. With ``repeat=True`` a block of three or four
    bad values is also removed, in the middle of the data, at either series end
    and on either side of a gap. In a block of five or more only the outer bad
    values and their good neighbours are removed (at a series end or a gap only
    the inner ones), because the inner values are consistent with each other.
    Runs of several bad values (logger glitch, stuck sensor) belong to absolute
    limits or manual removal. A level shift costs the two records at the step.
    When a series end or a gap follows within four records of the step, the one
    to three records in between go too, because they look like a bad block. A
    gap is a missing record in the input. In a stepwise screening this includes
    records removed by earlier tests.

    Example:
        Daytime and nighttime are separate by default and need the site location.
        ``window_length`` is a record count (48 * 7 is one week of 30-minute data)
        or a time span:

        >>> import diive as dv
        >>> s = dv.load_exampledata_parquet()['NEE_CUT_REF_orig'].loc['2022-07']
        >>> ham = dv.outliers.Hampel(series=s, window_length=48 * 7,
        ...                          n_sigma_daytime=5.5, n_sigma_nighttime=4,
        ...                          lat=46.815, lon=9.856, utc_offset=1).run()
        >>> cleaned = ham.filteredseries

        One threshold for all records:

        >>> ham = dv.outliers.Hampel(series=s, window_length=48 * 7, n_sigma=5.5,
        ...                          separate_day_night=False).run()
        >>> ham = dv.outliers.Hampel(series=s, window_length='7D', separate_day_night=False).run()

        See `examples/preprocessing/outlier_detection/outlier_hampel.py` for complete examples.
    """

    flagid = 'OUTLIER_HAMPEL'

    def __init__(self,
                 series: Series,
                 lat: float = None,
                 lon: float = None,
                 utc_offset: int = None,
                 window_length: int | str = 48 * 13,
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
            window_length (int or str): The size of the sliding window centered on
                the point, as a record count or as a time span such as ``'13D'``.
                A time span is converted to records at the series' frequency
                (``'13D'`` is 624 records of 30-min data, 1872 of 10-min data) and
                must be a whole multiple of it. Default is ``48 * 13 = 624``
                records, which equals **13 days at the half-hourly (30-min)
                sampling rate** typical of eddy-covariance data — matching the
                Papale et al. 2006 spike-detection window. For other sampling
                rates pass ``'13D'`` or scale the count (e.g. ``24 * 13`` for hourly).
                The window rolls over the records present, so missing records
                (gaps, and outliers removed by earlier iterations) do not count
                and the window stretches across them: with 20 % of the records
                missing, ``'13D'`` covers about 16 calendar days. With
                ``separate_day_night=True`` the median and MAD of a daytime
                record come from the daytime records in its window, those of a
                nighttime record from the nighttime records.
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
            separate_day_night (bool): If ``True``, daytime and nighttime (from potential
                radiation at the site) get their own rolling median and MAD, from the
                records of the same period in the window, and their own threshold, as in
                ONEFlux. Unlike ONEFlux, the records next to the change between day and
                night belong to one period only: against the near-zero night MAD of
                radiation every record on the light ramp would look like a spike. The
                night includes records with up to 20 W m-2 potential radiation, so in
                radiation a few records at sunrise and sunset can still be rejected.
                If ``False``, one median, MAD and threshold for all records.
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
        self.window_length = window_to_records(window_length, self.series, name='window_length',
                                               verbose=self.verbose)
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

        # Runs of records without a gap in the INPUT data. A gap in the input is
        # never bridged. The differences themselves use the runs of the current
        # series (see _double_differences), so a record removed by an earlier
        # iteration also ends a run.
        self._segment = self._input_segments(self.series)

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
    def _input_segments(series: Series) -> Series:
        """Number of the gap-free run each input record belongs to.

        A run ends where the next record is missing, either as an absent
        timestamp or as a missing value. Dropping missing records before
        differencing would otherwise pair a record with a partner hours or days
        away and make every gap edge look like a spike.

        Returns an integer Series over the input index (one run when the index
        is not a usable time axis).
        """
        index = series.index
        if not isinstance(index, DatetimeIndex) or len(index) < 3:
            return pd.Series(0, index=index)

        # A non-fixed offset (month/year start, business day, week) has no constant
        # duration and raises on `.nanos`; even `hasattr` propagates that error, so
        # the attempt itself is the test. The timestamps then supply the typical step.
        step = None
        if index.freq is not None:
            try:
                step = pd.Timedelta(index.freq.nanos, unit='ns')
            except ValueError:
                step = None
        if step is None:
            step = pd.Series(index).diff().median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            return pd.Series(0, index=index)

        # A neighbour is missing either because the timestamp itself is absent
        # (irregular index) or because it carries no value (gap on a regular grid).
        steps = pd.Series(index).diff()
        far_before = (steps > step * 1.5).to_numpy()
        empty = series.isna().to_numpy()
        empty_before = np.append(True, empty[:-1])
        return pd.Series(np.cumsum(far_before | empty_before), index=index)

    def _double_differences(self, s: Series) -> tuple[Series, Series, Series]:
        """Second differences of *s* that stay inside each gap-free run.

        A run ends at a gap in the input and at a record removed by an earlier
        iteration. Inside a run: ``d = 2 x_t - x_{t-1} - x_{t+1}``, the Papale
        et al. (2006) double difference. A record at the start or end of its run
        has a neighbour on one side only and is tested against the linear
        extrapolation from the next two records, ``d = x_t - 2 x_{t+1} + x_{t+2}``
        (or mirrored), which has the same noise and also cancels a trend.
        Without that test a spike at a run edge was never judged, and with
        ``repeat=True`` each iteration removed the next record paired with it,
        until the run was gone.

        Removed records end runs because differencing across them assumes equal
        spacing. On a diurnal ramp or across a level shift that mismatch looks
        like a spike, so each iteration removed the next record and
        ``repeat=True`` ate into good data (L226). The price: in a block of five
        or more bad values the inner records are consistent with each other and
        stay. Short runs left next to such a hole get a line test instead:

        - A run of one or two records with kept records of the same input run
          on both sides is compared with the straight line between the nearest
          kept record on either side.
        - A run of one or two records, or of three at a series end or input
          gap, with kept records of the same input run on one side only is
          compared with the line through the nearest two records on that side,
          extended across the hole. Those two records must belong to a run of
          three or more.

        A line never crosses an input gap. Other records in runs of fewer than
        three are NaN.

        Returns ``d`` plus, for mapping the detection band to data units, the
        value ``x_t`` is compared with and the factor on ``x_t`` in ``d``.
        """
        seg = self._input_segments(self.filteredseries).reindex(s.index)
        same = {k: (seg.shift(k) == seg) for k in (1, 2, -1, -2)}
        p1, p2, n1, n2 = s.shift(1), s.shift(2), s.shift(-1), s.shift(-2)
        central = same[1] & same[-1]
        forward = ~central & same[-2]
        backward = ~central & ~forward & same[2]
        reference = ((p1 + n1) / 2).where(central, (2 * n1 - n2).where(forward, (2 * p1 - p2).where(backward)))
        factor = pd.Series(np.where(central, 2.0, 1.0), index=s.index)

        # Short runs next to a hole made by an earlier iteration get the line
        # tests described in the docstring, so a block of three or four bad
        # values is removed completely wherever it sits. The factor gives both
        # tests the noise of the central difference. Longer runs are not
        # bridged: across wide holes the line misses the curvature of a diurnal
        # cycle.
        segv = seg.to_numpy()
        n = len(segv)
        first = np.searchsorted(segv, segv, side='left')
        last = np.searchsorted(segv, segv, side='right') - 1
        length = last - first + 1
        if (length <= 3).any():
            inseg = self._segment.reindex(s.index).to_numpy()

            def kept(i):
                # Record i exists and belongs to the same input run.
                return (i >= 0) & (i < n) & (inseg[np.clip(i, 0, n - 1)] == inseg)

            left, right = first - 1, last + 1
            # At an end or input gap only one side of a block is removed in the
            # first pass, so a block of four leaves three records there.
            short = (length <= 2) | ((length == 3) & ~(kept(left) & kept(right)))
            island = short & kept(left) & kept(right)
            # The line through two records comes only from a neighbouring run
            # of three or more. A shorter one may be the rest of a bad block.
            ahead = short & ~island & kept(right) & (length[np.clip(right, 0, n - 1)] >= 3)
            behind = short & ~island & ~ahead & kept(left) & (length[np.clip(left, 0, n - 1)] >= 3)
            # The line runs through records a and b. w is the position of the
            # tested record on it: 0 at a, 1 at b, negative beyond a.
            a = np.where(island | behind, left, right)
            b = np.where(island, right, np.where(behind, left - 1, right + 1))
            line = island | ahead | behind
            a, b = np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)
            pos = self.series.index.get_indexer(s.index)
            x = s.to_numpy(dtype=float)
            # Without a line test a and b can coincide, so w is not finite there.
            with np.errstate(divide='ignore', invalid='ignore'):
                w = (pos - pos[a]) / (pos[b] - pos[a])
                reference = reference.mask(line, (1 - w) * x[a] + w * x[b])
                factor = factor.mask(line, np.sqrt(6 / (1 + (1 - w) ** 2 + w ** 2)))
        return factor * (s - reference), reference, factor

    def _rolling_median_mad(self, x: Series) -> tuple[Series, Series]:
        """Centred rolling median and MAD of *x*. NaN values do not count.

        A window in which more than half the values are identical has a MAD of
        exactly zero, and then the detection band has zero width: every value that
        differs from the local median at all becomes an outlier, however small the
        difference. That is not a strict filter, it is an undefined one - the data
        carry no scale to judge against - and substituting a tiny epsilon turns it
        into a silent mass rejection of the signal itself. Quantized readings, a
        stuck sensor or coarse data upsampled before they reach this class hit this
        routinely. Such records get a NaN MAD, so they are left unflagged (NaN
        limits compare False), and the count is reported rather than hidden.
        """
        median = x.rolling(window=self.window_length, center=True, min_periods=1).median()
        mad = (x - median).abs().rolling(window=self.window_length, center=True, min_periods=1).median()
        degenerate = (mad == 0) & x.notna()
        if degenerate.any():
            self._n_degenerate_scale += int(degenerate.sum())
        return median, mad.where(mad != 0)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag using optimized Pandas operations."""

        # Prepare data
        s = self.filteredseries.copy().dropna()

        # 2. Transform data
        if self.use_differencing:
            # Missing records were dropped above, so consecutive entries can be hours
            # or days apart. The differences stay inside each gap-free run.
            s_to_test, reference, factor = self._double_differences(s)
        else:
            s_to_test = s

        if self.separate_day_night:
            # Median and MAD per period, as in ONEFlux qc_auto (Papale et al. 2006).
            # With shared statistics the quiet night (e.g. radiation near zero) set
            # the MAD and normal daytime variability looked like spikes (L239). The
            # window keeps its span, the records of the other period are blanked.
            # ONEFlux also counts the records next to the change between day and
            # night in both periods. Not here: for radiation, the night statistics
            # rejected nearly every record on the light ramp.
            in_day = self.is_daytime.reindex(s_to_test.index, fill_value=False).astype(bool)
            in_night = self.is_nighttime.reindex(s_to_test.index, fill_value=False).astype(bool)
            is_outlier = pd.Series(False, index=s_to_test.index)
            upper_bound = pd.Series(np.nan, index=s_to_test.index)
            lower_bound = pd.Series(np.nan, index=s_to_test.index)
            for in_period, n_sigma in ((in_day, self.n_sigma_daytime), (in_night, self.n_sigma_nighttime)):
                median, mad = self._rolling_median_mad(s_to_test.where(in_period))
                upper = median + self.k * mad * n_sigma
                lower = median - self.k * mad * n_sigma
                is_outlier |= (in_period & ((s_to_test > upper) | (s_to_test < lower))).fillna(False)
                upper_bound = upper_bound.mask(in_period, upper)
                lower_bound = lower_bound.mask(in_period, lower)
        else:
            median, mad = self._rolling_median_mad(s_to_test)
            upper_bound = median + self.k * mad * self.n_sigma
            lower_bound = median - self.k * mad * self.n_sigma
            # NaN limits (degenerate scale) and NaN test values (differences spanning a
            # gap) both compare False here, i.e. no decision is made for those records.
            is_outlier = ((s_to_test > upper_bound) | (s_to_test < lower_bound)).fillna(False)

        # Expose the per-iteration detection band in DATA units (for visualisation).
        # Raw mode: the bounds already are in data units. Double-differencing mode:
        # the test runs on d = factor * (x_t - reference), so x_t is flagged iff it
        # leaves reference + bound / factor. This is the exact data-space band the
        # flag decision uses for the current iteration.
        if self.use_differencing:
            self.last_upper_bound = reference + upper_bound / factor
            self.last_lower_bound = reference + lower_bound / factor
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
