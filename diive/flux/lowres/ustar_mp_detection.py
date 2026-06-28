"""
USTAR MOVING POINT DETECTION: FRICTION VELOCITY THRESHOLD
==========================================================

Detect friction velocity (u*) threshold using the ONEFlux moving point algorithm (Papale et al., 2006).

This is a faithful, vectorized port of the reference C implementation in ONEFlux
(`oneflux_steps/ustar_mp/src/ustar.c`). The defaults reproduce the ONEFlux default
configuration: forward-mode-2 only (``no_forward_mode_2 = 0``, all other modes off),
percentile check off, calendar-quarter seasons (``1,2,3;4,5,6;7,8,9;10,11,12``).

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List

from diive.core.utils.console import info, detail, warn


class UstarMovingPointDetection:
    """
    Detect USTAR threshold using ONEFlux moving point method (Papale et al., 2006).

    This class implements the complete USTAR threshold detection workflow for eddy covariance
    flux measurements. It identifies the friction velocity (u*) above which flux measurements
    are reliable by analyzing the stability of nighttime respiration across USTAR classes.

    The key insight is that during nighttime, without photosynthesis, NEE represents pure
    ecosystem respiration. At low u* (poor turbulence), measured respiration is systematically
    underestimated due to storage and vertical mixing limitations. At high u* (good turbulence),
    respiration stabilizes. The threshold is where respiration stops changing with increasing u*.

    Algorithm steps (mirrors ``ustar.c::ustar_threshold``):
    1. FILTER DATA: Select nighttime records only (SW_IN < 10 W/m2) for pure respiration signal
    2. STRATIFY BY SEASON: Divide year into 4 seasons (calendar quarters by default)
    3. STRATIFY BY TEMPERATURE: Within each season, divide into 7 temperature classes
    4. STRATIFY BY USTAR: Within each temperature class, divide into 20 USTAR classes
    5. COMPUTE STATISTICS: Calculate mean respiration per USTAR class
    6. VALIDATE: Check temperature-USTAR independence and first USTAR validity
    7. DETECT THRESHOLD: Use forward mode detection (ascending search) to find stability point
    8. AGGREGATE: Median across temperature classes gives season threshold;
       maximum across seasons gives the annual (conservative) threshold

    Class boundaries are tie-aware: equal TA / USTAR values are never split across adjacent
    classes (matches the C boundary-extension loop), so quantized USTAR data is binned exactly
    as ONEFlux does.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with datetime index and required columns: NEE, TA, USTAR, SW_IN
    nee_col : str, optional
        Net ecosystem exchange column name (auto-detected if None)
        Looks for columns containing 'NEE' and 'QCF' preferentially
    ta_col : str, optional
        Air temperature column name (auto-detected if None)
        Looks for columns containing 'TA_' or similar
    ustar_col : str, optional
        Friction velocity column name (auto-detected if None)
        Looks for columns containing 'USTAR' or 'U_STAR'
    swin_col : str, optional
        Shortwave radiation column name (auto-detected if None)
        Used to identify nighttime (SW_IN < 10 W/m2)
    ta_classes_count : int, default=7
        Number of temperature stratification classes per season (matches ONEFlux default)
    ustar_classes_count : int, default=20
        Number of USTAR stratification classes per temperature class (matches ONEFlux default)
    forward_mode_n : int, default=2
        Number of consecutive USTAR classes that must satisfy the plateau condition before a
        threshold is accepted (forward-mode order). n=2 is "Fw2", the default in both ONEFlux
        (only forward_mode_2 is enabled by default) and REddyProc (usEstUstarThresholdSingleFw2Binned).
        n=1 is "Fw1": looser, fires on a single bin, and never yields a higher threshold than Fw2.
        Higher n is stricter (requires a longer confirmed plateau), giving an equal-or-higher
        threshold that is more robust to single-bin noise.
    season_groups : List[List[int]], optional
        Custom season grouping by month numbers [1-12].
        Default: [[1,2,3], [4,5,6], [7,8,9], [10,11,12]] (calendar quarters), matching the
        ONEFlux default ``default_seasons_group = "1,2,3;4,5,6;7,8,9;10,11,12"``.
    bootstrapping_times : int, default=100
        Number of bootstrap resampling iterations for uncertainty estimation
    verbose : int, default=0
        Verbosity level: 0=silent, 1=progress, 2=detailed debug output

    Attributes
    ----------
    results_ : pd.DataFrame
        Detected thresholds with column: 'threshold'
        Index contains season labels (Season 1, 2, 3, 4)
    annual_thresholds_ : dict
        Annual threshold with key: 'threshold' (maximum across seasons)

    Examples
    --------
    >>> import diive as dv
    >>> df = dv.load_exampledata_parquet_lae()
    >>> detector = dv.flux.UstarMovingPointDetection(df)
    >>> thresholds = detector.detect()
    >>> print(detector.summary())
    >>> stats = detector.bootstrap()

    Notes
    -----
    The algorithm requires:
    - At least 3000 valid records for valid detection (MIN_VALUE_PERIOD)
    - At least 700 (= 100 * 7) records per season for per-season detection
      (TA_CLASS_MIN_SAMPLE * ta_classes_count); seasons below this are skipped
    - If every season is below 700, all night data is pooled into one season
      (ONEFlux "second method", one big season)

    Nighttime filtering is critical: SW_IN < 10 W/m2 selects purely dark conditions
    where respiration is the only flux component, free from photosynthetic variability.

    See Also
    --------
    UstarBootstrapThresholds : Multi-year sliding-window bootstrap wrapper.

    Example
    -------
    See `examples/flux/lowres/flux_ustar_mp_detection.py` for complete examples
    of USTAR threshold detection using the moving point method with bootstrap uncertainty estimation.
    """

    # Constants from ONEFlux (types.h) - DO NOT MODIFY without consulting Papale et al. (2006)
    NIGHT_THRESHOLD = 10.0  # W/m2 - threshold for identifying nighttime (SWIN_FOR_NIGHT in C code)
    MIN_SAMPLES_PERIOD = 3000  # Minimum total valid records required (MIN_VALUE_PERIOD)
    MIN_SAMPLES_SEASON = 160  # MIN_VALUE_SEASON: one-big-season eligibility threshold
    MIN_SAMPLES_TA_CLASS = 100  # Minimum records per temperature class (TA_CLASS_MIN_SAMPLE)
    CORRELATION_CHECK = 0.5  # Maximum |correlation(TA, USTAR)| allowed for valid TA class
    FIRST_USTAR_MEAN_CHECK = 0.2  # Maximum first USTAR class mean in m/s (validation threshold)
    THRESHOLD_CHECK = 1.0  # Plateau ratio (THRESHOLD_CHECK)
    WINDOW_SIZE_FORWARD_MODE = 10  # WINDOWS_SIZE_FOR_FORWARD_MODE
    THRESHOLD_NOT_FOUND = 10.0  # Marker: threshold could not be detected (USTAR_THRESHOLD_NOT_FOUND)

    def __init__(
        self,
        df: pd.DataFrame,
        nee_col: Optional[str] = None,
        ta_col: Optional[str] = None,
        ustar_col: Optional[str] = None,
        swin_col: Optional[str] = None,
        ta_classes_count: int = 7,
        ustar_classes_count: int = 20,
        forward_mode_n: int = 2,
        season_groups: Optional[List[List[int]]] = None,
        bootstrapping_times: int = 100,
        verbose: int = 0,
    ):
        # Validate input
        """Set up moving-point USTAR threshold detection. See the class docstring."""
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty")
        if forward_mode_n < 1:
            raise ValueError(f"forward_mode_n must be >= 1, got {forward_mode_n}")

        # Auto-detect columns if not specified
        if nee_col is None:
            candidates = [col for col in df.columns if 'NEE' in col and 'QCF' in col]
            if not candidates:
                candidates = [col for col in df.columns if 'NEE' in col]
            if candidates:
                nee_col = candidates[0]
            else:
                nee_col = self._find_column(df, ['NEE', 'NEE_CUT_REF', 'NEE_L3'])

        if ta_col is None:
            candidates = [col for col in df.columns if 'TA_' in col]
            if candidates:
                ta_col = candidates[0]
            else:
                ta_col = self._find_column(df, ['TA', 'TSYS', 'TAIR'])

        if ustar_col is None:
            candidates = [col for col in df.columns if 'USTAR' in col]
            if candidates:
                ustar_col = candidates[0]
            else:
                ustar_col = self._find_column(df, ['U_STAR'])

        if swin_col is None:
            candidates = [col for col in df.columns if 'SW_IN' in col]
            if candidates:
                swin_col = candidates[0]
            else:
                swin_col = self._find_column(df, ['SWIN', 'RAD'])

        if nee_col is None:
            raise ValueError("Could not find NEE column. Specify with nee_col parameter.")
        if ta_col is None:
            raise ValueError("Could not find TA (temperature) column. Specify with ta_col parameter.")
        if ustar_col is None:
            raise ValueError("Could not find USTAR column. Specify with ustar_col parameter.")
        if swin_col is None:
            raise ValueError("Could not find SW_IN (shortwave radiation) column. Specify with swin_col parameter.")

        # Keep a reference (not a copy) - this class never mutates df, and the per-iteration
        # copies in bootstrap workflows make a defensive copy here pure overhead.
        self.df = df
        self.nee_col = nee_col
        self.ta_col = ta_col
        self.ustar_col = ustar_col
        self.swin_col = swin_col

        self.ta_classes_count = ta_classes_count
        self.ustar_classes_count = ustar_classes_count
        self.forward_mode_n = forward_mode_n
        self.bootstrapping_times = bootstrapping_times
        self.verbose = verbose

        # Default season groups: calendar quarters (ONEFlux default)
        if season_groups is None:
            season_groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        self.season_groups = season_groups
        self.seasons_count = len(season_groups)

        # Ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'TIMESTAMP' in self.df.columns:
                self.df = self.df.set_index(pd.to_datetime(self.df['TIMESTAMP']))
            else:
                raise ValueError("DataFrame must have datetime index or TIMESTAMP column")

        self._month = self._month_per_group(self.df.index)

        # Results storage
        self.results_ = pd.DataFrame()
        self.annual_thresholds_: Dict[str, float] = {}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _month_per_group(index: pd.DatetimeIndex) -> np.ndarray:
        """
        Month (1-12) used for season grouping, with the ONEFlux end-of-period shift.

        FLUXNET timestamps mark the END of the averaging period, so a record stamped
        day-1 00:00 actually belongs to the previous month (matches ``dataset.c``: "day 1
        and 00:00 belongs to previous month"). Records at midnight on the 1st are shifted
        back one month; January wraps to December.
        """
        month = index.month.to_numpy().astype(np.int64).copy()
        boundary = (
            (index.day.to_numpy() == 1)
            & (index.hour.to_numpy() == 0)
            & (index.minute.to_numpy() == 0)
        )
        month[boundary] -= 1
        month[month == 0] = 12
        return month

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        """Pearson correlation, matching ustar.c::correlation (NaN -> not-a-number)."""
        xm = x.mean()
        ym = y.mean()
        dx = x - xm
        dy = y - ym
        denom = np.sqrt((dx * dx).sum()) * np.sqrt((dy * dy).sum())
        if denom == 0.0:
            return np.nan
        return float((dx * dy).sum() / denom)

    def _class_bounds(self, vals_sorted: np.ndarray, n_classes: int, n_per_class: int) -> np.ndarray:
        """
        Faithful ONEFlux class binning with tie extension.

        Returns an (n_classes, 2) int array of inclusive (start, end) indices into
        ``vals_sorted``; empty classes are (-1, -1). Mirrors the boundary-extension loop
        in ustar.c so that equal values are never split across adjacent classes.
        """
        N = len(vals_sorted)
        bounds = np.full((n_classes, 2), -1, dtype=np.int64)

        class_end = 0
        broke = False
        for i in range(n_classes - 1):
            class_start = class_end
            class_end = n_per_class * (i + 1) - 1
            if class_start >= N:
                broke = True
                break
            if class_end >= N:
                class_end = N - 1
            value = vals_sorted[class_end]
            # extend boundary forward across tied values
            class_end += 1
            while class_end < N and vals_sorted[class_end] == value:
                class_end += 1
            bounds[i, 0] = class_start
            bounds[i, 1] = class_end - 1

        # last class gets the remainder (C assigns window[i], i == n_classes-1 on full loop)
        if not broke and class_end < N:
            bounds[n_classes - 1, 0] = class_end
            bounds[n_classes - 1, 1] = N - 1

        return bounds

    @staticmethod
    def _class_means(arr_sorted: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """
        Mean of each class window via cumulative sums (empty classes -> 0.0, as in C).
        """
        cs = np.concatenate(([0.0], np.cumsum(arr_sorted, dtype=np.float64)))
        means = np.zeros(len(bounds), dtype=np.float64)
        for k in range(len(bounds)):
            s = bounds[k, 0]
            e = bounds[k, 1]
            if s < 0:
                continue  # leave 0.0 (matches C reset value for empty classes)
            means[k] = (cs[e + 1] - cs[s]) / (e - s + 1)
        return means

    @staticmethod
    def _meanws(arr: np.ndarray, index: int, element_count: int) -> float:
        """Mean-with-start, matching ustar.c::meanws (empty range -> NaN/invalid)."""
        n = len(arr)
        if index > n:
            return 0.0
        seg = arr[index:index + element_count]
        if seg.size == 0:
            return np.nan
        return float(seg.mean())

    def _forward_mode(self, ustar_mean: np.ndarray, fx_mean: np.ndarray, n: int) -> float:
        """
        Forward-mode threshold (ustar.c::forward_mode, percentile check disabled).

        For each USTAR class i (i <= n_classes - n), require that NEE in the next ``n``
        classes is >= the forward window mean: ``fx_mean[i+y] >= meanws(...)`` for all y.
        The first i that satisfies this returns ``ustar_mean[i]``.
        """
        n_classes = len(ustar_mean)
        if n < 1 or n_classes - n <= 0:
            return self.THRESHOLD_NOT_FOUND

        ws = self.WINDOW_SIZE_FORWARD_MODE
        tc = self.THRESHOLD_CHECK
        for i in range(n_classes - n + 1):  # inclusive upper bound, matches C
            means = np.empty(n, dtype=np.float64)
            invalid = False
            for y in range(n):
                m = self._meanws(fx_mean, i + 1 + y, ws)
                if not np.isfinite(m):
                    invalid = True
                    break
                means[y] = m
            if invalid:
                continue
            if np.all(fx_mean[i:i + n] >= means * tc):
                return float(ustar_mean[i])

        return self.THRESHOLD_NOT_FOUND

    # ------------------------------------------------------------------ core

    def _detect_ta_class(self, nee_c: np.ndarray, ustar_c: np.ndarray) -> float:
        """Detect USTAR threshold for one temperature class via USTAR stratification."""
        m = len(ustar_c)
        n_per_ustar = m // self.ustar_classes_count
        if n_per_ustar < 1:
            return self.THRESHOLD_NOT_FOUND

        order = np.argsort(ustar_c, kind='stable')
        ustar_s = ustar_c[order]
        nee_s = nee_c[order]

        bounds = self._class_bounds(ustar_s, self.ustar_classes_count, n_per_ustar)
        ustar_mean = self._class_means(ustar_s, bounds)
        fx_mean = self._class_means(nee_s, bounds)

        # First USTAR class must be in the low-turbulence regime
        if ustar_mean[0] > self.FIRST_USTAR_MEAN_CHECK:
            return self.THRESHOLD_NOT_FOUND

        return self._forward_mode(ustar_mean, fx_mean, n=self.forward_mode_n)

    def _detect_season(self, nee: np.ndarray, ta: np.ndarray, ustar: np.ndarray) -> float:
        """Detect USTAR threshold for one season (median across temperature classes)."""
        N = len(nee)
        n_per_ta = N // self.ta_classes_count
        if n_per_ta < self.MIN_SAMPLES_TA_CLASS:
            return self.THRESHOLD_NOT_FOUND

        order = np.argsort(ta, kind='stable')
        ta_s = ta[order]
        ustar_s = ustar[order]
        nee_s = nee[order]

        bounds = self._class_bounds(ta_s, self.ta_classes_count, n_per_ta)

        thresholds = []
        for k in range(self.ta_classes_count):
            s = bounds[k, 0]
            e = bounds[k, 1]
            if s < 0 or (e - s + 1) < self.MIN_SAMPLES_TA_CLASS:
                continue

            ta_c = ta_s[s:e + 1]
            ustar_cc = ustar_s[s:e + 1]
            nee_cc = nee_s[s:e + 1]

            corr = self._pearson(ta_c, ustar_cc)
            if not np.isfinite(corr) or abs(corr) > self.CORRELATION_CHECK:
                continue

            th = self._detect_ta_class(nee_cc, ustar_cc)
            if np.isfinite(th) and th != self.THRESHOLD_NOT_FOUND:
                thresholds.append(th)

        return float(np.median(thresholds)) if thresholds else self.THRESHOLD_NOT_FOUND

    def _compute_seasonal(
        self, nee: np.ndarray, ta: np.ndarray, ustar: np.ndarray, month: np.ndarray
    ) -> List[float]:
        """
        Per-season thresholds from night+valid arrays (NaN where season is too small).

        Implements the ONEFlux "one big season" fallback: when every season has fewer than
        TA_CLASS_MIN_SAMPLE * ta_classes_count samples, all data is pooled into a single
        season and the resulting threshold is broadcast to all season slots.
        """
        min_per_season = self.MIN_SAMPLES_TA_CLASS * self.ta_classes_count  # 700
        season_counts = [int(np.isin(month, months).sum()) for months in self.season_groups]

        if all(c < min_per_season for c in season_counts):
            # one big season (second method) - pool everything
            if len(nee) < self.MIN_SAMPLES_SEASON:
                return [np.nan] * self.seasons_count
            th = self._detect_season(nee, ta, ustar)
            th = th if (np.isfinite(th) and th != self.THRESHOLD_NOT_FOUND) else np.nan
            return [th] * self.seasons_count

        thresholds: List[float] = []
        for months, c in zip(self.season_groups, season_counts):
            if c < min_per_season:
                thresholds.append(np.nan)
                continue
            mask = np.isin(month, months)
            th = self._detect_season(nee[mask], ta[mask], ustar[mask])
            thresholds.append(th if (np.isfinite(th) and th != self.THRESHOLD_NOT_FOUND) else np.nan)
        return thresholds

    def _aggregate_annual(self, thresholds: List[float]) -> float:
        """Annual threshold = maximum across valid seasonal thresholds (conservative)."""
        valid = [t for t in thresholds if np.isfinite(t) and t != self.THRESHOLD_NOT_FOUND]
        return float(np.max(valid)) if valid else np.nan

    def _night_valid_arrays(self):
        """Extract night+valid (NEE, TA, USTAR, month) numpy arrays plus the valid count."""
        nee = self.df[self.nee_col].to_numpy(dtype='float64')
        ta = self.df[self.ta_col].to_numpy(dtype='float64')
        ustar = self.df[self.ustar_col].to_numpy(dtype='float64')
        swin = self.df[self.swin_col].to_numpy(dtype='float64')

        valid = np.isfinite(nee) & np.isfinite(ta) & np.isfinite(ustar) & np.isfinite(swin)
        night = valid & (swin < self.NIGHT_THRESHOLD)
        return nee, ta, ustar, self._month, valid, night

    # ------------------------------------------------------------------ public

    def detect(self) -> pd.DataFrame:
        """
        Detect USTAR thresholds for all seasons (Papale et al., 2006).

        Returns
        -------
        pd.DataFrame
            Seasonal thresholds with column 'threshold' (m/s), or NaN where detection failed.
            Rows: Season 1..N. Annual threshold (max across seasons) stored in
            ``annual_thresholds_`` and retrievable via ``get_annual_thresholds()``.
        """
        if self.verbose >= 1:
            info(f"Detecting USTAR thresholds (Papale et al., 2006) | "
                 f"{len(self.df)} records, {self.seasons_count} seasons, "
                 f"{self.ta_classes_count} TA classes, {self.ustar_classes_count} USTAR classes")

        if len(self.df) < self.MIN_SAMPLES_PERIOD:
            raise ValueError(
                f"Insufficient data: {len(self.df)} records, need at least {self.MIN_SAMPLES_PERIOD}"
            )

        nee, ta, ustar, month, valid, night = self._night_valid_arrays()

        n_valid = int(valid.sum())
        if n_valid < self.MIN_SAMPLES_PERIOD:
            raise ValueError(
                f"Insufficient valid data: {n_valid} records, need at least {self.MIN_SAMPLES_PERIOD}"
            )

        nee_n = nee[night]
        ta_n = ta[night]
        ustar_n = ustar[night]
        month_n = month[night]

        thresholds_list = self._compute_seasonal(nee_n, ta_n, ustar_n, month_n)

        if self.verbose >= 2:
            for i, t in enumerate(thresholds_list):
                if np.isfinite(t):
                    detail(f"  Season {i + 1}: {t:.4f} m/s")
                else:
                    detail(f"  Season {i + 1}: not found")

        self.results_ = pd.DataFrame(
            {'threshold': thresholds_list},
            index=[f'Season {i + 1}' for i in range(self.seasons_count)],
        )

        annual = self._aggregate_annual(thresholds_list)
        self.annual_thresholds_ = {
            'threshold': annual if np.isfinite(annual) else self.THRESHOLD_NOT_FOUND,
        }

        if self.verbose >= 1:
            if np.isfinite(annual):
                info(f"Annual threshold (max across seasons): {annual:.4f} m/s")
            else:
                info("Annual threshold: Not found")

        return self.results_

    def _iter_bootstrap_seasonal(self, n_iter: int, rng: np.random.Generator):
        """
        Yield the per-season threshold list for each bootstrap resample.

        Extracts the night+valid arrays once, then per iteration resamples the full record
        with replacement (matching ONEFlux: draw from all rows, keep the night+valid ones)
        and recomputes seasonal thresholds. No per-iteration object construction or
        DataFrame copy. Resamples producing no night data, or that fail, are skipped.
        """
        nee, ta, ustar, month, valid, night = self._night_valid_arrays()
        n_total = len(self.df)

        for boot_idx in range(n_iter):
            if self.verbose >= 2 and boot_idx % 10 == 0:
                detail(f"  Iteration {boot_idx + 1}/{n_iter}")

            idx = rng.integers(0, n_total, n_total)
            sel = night[idx]
            if not sel.any():
                continue
            b = idx[sel]

            try:
                yield self._compute_seasonal(nee[b], ta[b], ustar[b], month[b])
            except Exception as e:  # keep a single failed resample from aborting the run
                if self.verbose >= 2:
                    warn(f"Bootstrap {boot_idx} failed: {e}")
                continue

    def bootstrap_annual_samples(
        self, n_iter: Optional[int] = None, rng: Optional[np.random.Generator] = None
    ) -> List[float]:
        """
        Bootstrap distribution of the annual (max-across-seasons) threshold.

        Fast path consumed by ``UstarBootstrapThresholds``: build the detector once on the
        window, then call this to get the resampled annual thresholds directly from
        pre-extracted numpy arrays - avoiding a ``DataFrame.sample()`` copy and a fresh
        detector per iteration.

        Parameters
        ----------
        n_iter : int, optional
            Number of bootstrap iterations (uses self.bootstrapping_times if None).
        rng : numpy.random.Generator, optional
            Random generator (a fresh default generator is created if None).

        Returns
        -------
        list of float
            One annual threshold per successful resample (length <= n_iter).
        """
        if n_iter is None:
            n_iter = self.bootstrapping_times
        if rng is None:
            rng = np.random.default_rng()

        samples: List[float] = []
        for ths in self._iter_bootstrap_seasonal(n_iter, rng):
            ann = self._aggregate_annual(ths)
            if np.isfinite(ann):
                samples.append(ann)
        return samples

    def bootstrap(self, n_iter: Optional[int] = None) -> pd.DataFrame:
        """
        Bootstrap resampling for USTAR threshold uncertainty.

        Resamples the full record with replacement (matching ONEFlux: draw from all rows,
        keep night+valid ones), recomputes per-season and annual thresholds each iteration,
        and returns the resulting distributions. Operates entirely on pre-extracted numpy
        arrays - no per-iteration object construction or DataFrame copy.

        Parameters
        ----------
        n_iter : int, optional
            Number of bootstrap iterations (uses self.bootstrapping_times if None)

        Returns
        -------
        pd.DataFrame
            Statistics (mean, std, p05, p50, p95) per season plus an 'Annual' row giving the
            distribution of the annual (max-across-seasons) threshold - the quantity used for
            filtering.
        """
        if n_iter is None:
            n_iter = self.bootstrapping_times

        if self.verbose >= 1:
            info(f"Running {n_iter} bootstrap iterations...")

        rng = np.random.default_rng()
        per_season: List[List[float]] = [[] for _ in range(self.seasons_count)]
        annual_samples: List[float] = []

        for ths in self._iter_bootstrap_seasonal(n_iter, rng):
            for si, t in enumerate(ths):
                if np.isfinite(t):
                    per_season[si].append(t)
            ann = self._aggregate_annual(ths)
            if np.isfinite(ann):
                annual_samples.append(ann)

        rows = [self._stats(s) for s in per_season]
        index = [f'Season {i + 1}' for i in range(self.seasons_count)]
        rows.append(self._stats(annual_samples))
        index.append('Annual')

        return pd.DataFrame(rows, index=index)

    @staticmethod
    def _stats(vals: List[float]) -> Dict[str, float]:
        """Summary statistics for a list of bootstrap thresholds."""
        if not vals:
            return {'mean': np.nan, 'std': np.nan, 'p05': np.nan, 'p50': np.nan, 'p95': np.nan}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'p05': float(np.percentile(arr, 5)),
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
        }

    def summary(self) -> str:
        """Return formatted summary of detection results."""
        if self.results_.empty:
            return "No detection results. Run detect() first."

        lines = ["USTAR Threshold Detection Results (Papale et al., 2006)"]
        lines.append("=" * 60)
        lines.append(f"{'Season':<15} {'Threshold (m/s)':<20}")
        lines.append("-" * 60)

        for idx in self.results_.index:
            val = self.results_.loc[idx, 'threshold']
            threshold_str = f"{val:.4f}" if (np.isfinite(val) and val != self.THRESHOLD_NOT_FOUND) else "Not found"
            lines.append(f"{idx:<15} {threshold_str:<20}")

        return "\n".join(lines)

    def get_annual_thresholds(self) -> Dict[str, float]:
        """
        Get annual USTAR threshold (maximum across all seasons).

        Returns
        -------
        Dict[str, float]
            Annual threshold with key 'threshold' (m/s), or NaN if detection failed.

        Notes
        -----
        Conservative approach: maximum of seasonal thresholds (Papale et al., 2006:
        "the whole data set is filtered according to the highest threshold found").
        """
        if not self.annual_thresholds_:
            raise RuntimeError("Detection not yet performed. Call detect() first.")
        result = self.annual_thresholds_.copy()
        if result.get('threshold') == self.THRESHOLD_NOT_FOUND:
            result['threshold'] = np.nan
        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UstarMovingPointDetection("
            f"n_records={len(self.df)}, "
            f"ta_classes={self.ta_classes_count}, "
            f"ustar_classes={self.ustar_classes_count}, "
            f"seasons={self.seasons_count})"
        )
