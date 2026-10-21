"""
USTAR MOVING POINT DETECTION: FRICTION VELOCITY THRESHOLD
==========================================================

Detect the friction velocity (u*) threshold with the ONEFlux moving point method (Papale et al., 2006).

Part of the diive library: https://github.com/holukas/diive

This is a faithful port of ONEFlux ``ustar_mp`` 1.3.7 (``oneflux_steps/ustar_mp/src``).
For one calendar year it reproduces the compiled C bitwise: every season and
TA-class threshold, the annual threshold and, given the same drawn rows, every
bootstrap value. The defaults are the ONEFlux defaults: forward mode 2 only,
percentile check off, calendar-quarter seasons ``1,2,3;4,5,6;7,8,9;10,11,12``.
The loops run as numba kernels in ``_ustar_mp_kernels``.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from diive.core.utils.console import detail, info, warn

# Days per month (main.c:94), used for the bootstrap season boundaries
_MONTHS_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _kernels():
    """Import the numba kernels on first use, so importing this module stays cheap."""
    from diive.flux.lowres import _ustar_mp_kernels
    return _ustar_mp_kernels


class UstarMovingPointDetection:
    r"""
    Detect the USTAR threshold with the ONEFlux moving point method (Papale et al., 2006).

    At night NEE is ecosystem respiration. At low u* the measured flux
    underestimates it, because storage and poor mixing hide part of it. Above a
    threshold u* the nighttime flux no longer increases with u*. The method finds
    that threshold.

    Algorithm (``ustar.c::ustar_threshold``):

    1. Keep night records (SW_IN < 10 W m-2) with NEE, TA and USTAR present.
    2. Split them into seasons (calendar quarters by default).
    3. Sort each season by TA and split it into 7 TA classes.
    4. Skip a TA class with fewer than 100 records or with ``|r(TA, USTAR)| > 0.5``.
    5. Sort each TA class by USTAR and split it into 20 USTAR classes. Compute the
       mean USTAR and NEE of each class.
    6. Skip the TA class when the first USTAR class mean is above 0.2 m s-1.
    7. Forward mode: the threshold is the mean USTAR of the first class where
       NEE in this class and the next ``forward_mode_n - 1`` classes is at
       least the mean NEE of the following 10 classes.
    8. The season threshold is the median over the TA classes. The annual
       threshold is the highest season threshold.

    Class boundaries never split equal TA or USTAR values (the C extends a
    class over a run of ties).

    Parameters
    ----------
    df : pd.DataFrame
        Data with a DatetimeIndex and columns for NEE, TA, USTAR and SW_IN.
        For ONEFlux parity pass one calendar year on its full grid (17520 or
        17568 half-hourly records, or 8760 or 8784 hourly records), with NaN
        for missing values. See Notes for other inputs.
    nee_col : str, optional
        NEE column. Auto-detected if None: a column containing 'NEE' and 'QCF',
        else the first column containing 'NEE'.
    ta_col : str, optional
        Air temperature column. Auto-detected if None (first column containing 'TA\_').
    ustar_col : str, optional
        Friction velocity column. Auto-detected if None (first column containing 'USTAR').
    swin_col : str, optional
        Shortwave radiation column, used to identify night (SW_IN < 10 W m-2).
        Auto-detected if None (first column containing 'SW_IN').
    ta_classes_count : int, default=7
        Number of TA classes per season (ONEFlux default).
    ustar_classes_count : int, default=20
        Number of USTAR classes per TA class (ONEFlux default).
    forward_mode_n : int, default=2
        Number of consecutive USTAR classes that must pass the plateau test.
        2 is ONEFlux "forward mode 2", the only mode ONEFlux runs by default.
        1 and 3 are ONEFlux forward modes 1 and 3.
    season_groups : List[List[int]], optional
        Months (1-12) per season, each month in one season only (as the C
        parser requires by default). Default ``[[1, 2, 3], [4, 5, 6], [7, 8, 9],
        [10, 11, 12]]``, the ONEFlux ``default_seasons_group``.
    bootstrapping_times : int, default=100
        Number of bootstrap draws.
    random_state : int or None, default=42
        Seed for the bootstrap draws. None gives a fresh draw on each call.
        ``detect()`` does not use it.
    verbose : int, default=0
        0 silent, 1 progress, 2 details.

    Attributes
    ----------
    results_ : pd.DataFrame
        Season thresholds (column 'threshold', rows 'Season 1'...). NaN where
        no threshold was found. After a collapse to one big season (see Notes)
        every row holds the one-season value.
    annual_thresholds_ : dict
        ``{'threshold': value}``, the highest season threshold, NaN if none.
    ta_class_thresholds_ : pd.DataFrame
        Threshold per TA class (rows) and season (columns), NaN where the class
        was skipped or no threshold was found.
    one_big_season_ : bool
        True when the main run collapsed all seasons into one.
    bootstrap_failed_ : np.ndarray of bool
        After a bootstrap: True for each draw where ONEFlux ``ustar_threshold``
        fails (too few records in the draw).
    bootstrap_dropped_ : bool
        After a bootstrap: True when at least one draw failed. ONEFlux then
        stops bootstrapping and writes no bootstrap percentiles for that year
        (bootstrapping.c:194-201), so nee_proc has no MP thresholds for it.

    Examples
    --------
    >>> import diive as dv
    >>> df = dv.load_exampledata_parquet_lae()
    >>> detector = dv.flux.UstarMovingPointDetection(df)
    >>> thresholds = detector.detect()  # then detector.summary() / detector.bootstrap()

    Notes
    -----
    Record minimum and method, as in ONEFlux (``ustar.c:500-535``, ``main.c:991,1093``).
    "Night records" are night records with NEE, TA and USTAR present. "Day
    records" are all records that are not night, including records without SW_IN.

    - Fewer than ``ta_classes_count * ustar_classes_count`` (140) night records:
      no detection, ``ValueError``.
    - Night + day records >= 3000: one threshold per season.
    - Otherwise, more than 160 night records: all seasons pooled into one big season.
    - Otherwise: no detection, ``ValueError``.
    - When every season has fewer than ``100 * ta_classes_count`` (700) night
      records, the seasons are pooled into one big season too.
    - A season (or the big season) with fewer than 700 night records gets no
      threshold.

    ONEFlux quirks reproduced on purpose (this is a port, not a redesign):

    - Bootstrap seasons are cut by row position, not by date
      (``main.c:1014-1038``, ``bootstrapping.c:173-180``). In a leap year the
      February end adds 1 row instead of 48 (``main.c:1018-1019``). So in a leap
      year each bootstrap season boundary sits 47 rows (about one day) early,
      and the last 47 records of the year are counted to the first season.
    - Once the main run or one bootstrap draw collapses to one big season, all
      later bootstrap draws stay collapsed (``ustar.c:509-533`` overwrite the
      global ``seasons_group_count``).
    - A failed bootstrap draw makes ONEFlux drop the whole bootstrap. diive
      returns NaN for that draw, keeps the other draws and sets
      ``bootstrap_dropped_``.
    - ``IS_INVALID_VALUE`` truncates to int, so values in (-10000, -9999] count
      as missing.
    - After a tie run reaches the end of a season or TA class, the C assigns one
      more class that overlaps the previous one (``ustar.c:633-637,728-732``).
      The TA loop has no "start == end" branch, the USTAR loop has
      (``ustar.c:620-625,711-720``). An inverted class window gives a NaN mean.
    - Rows are sorted with the Microsoft UCRT ``qsort`` of the reference build.
      Its order of equal TA or USTAR values sets the order of the NEE sums. A
      glibc build of ONEFlux can differ in the last bits.

    Input that is not one calendar year: the main run follows the same rules
    on all records. The bootstrap then cuts seasons by month instead of by row
    position, since the C row rule only exists for one calendar year. The
    month of a record is the month of its timestamp, except that a record
    stamped on the 1st at 00:00 belongs to the previous month (``dataset.c:438-448``).
    This gives the C months for both TIMESTAMP_MIDDLE and TIMESTAMP_END stamps.

    The bootstrap draws row positions uniformly from all records, as ONEFlux
    does on Linux. The Windows build of ONEFlux can only draw the first 16384
    rows, because ``RAND_MAX`` is 32767 there (``common.c:738-741``).

    See Also
    --------
    UstarThresholdDetection : One entry point for all u* detection modes, with
        the ONEFlux pooling of the bootstrap thresholds into VUT and CUT.

    Example
    -------
    See `examples/flux/lowres/flux_ustar_detection.py` and
    `examples/flux/lowres/flux_ustar_method_comparison.py`.
    """

    # Constants from ONEFlux types.h. Do not change: the port reproduces the C.
    NIGHT_THRESHOLD = 10.0  # SWIN_FOR_NIGHT
    MIN_SAMPLES_PERIOD = 3000  # MIN_VALUE_PERIOD: night + day records for seasonal detection
    MIN_SAMPLES_SEASON = 160  # MIN_VALUE_SEASON: night records for one big season
    MIN_SAMPLES_TA_CLASS = 100  # TA_CLASS_MIN_SAMPLE
    CORRELATION_CHECK = 0.5  # maximum |r(TA, USTAR)| in a TA class
    FIRST_USTAR_MEAN_CHECK = 0.2  # maximum mean USTAR of the first USTAR class (m s-1)
    THRESHOLD_CHECK = 1.0  # plateau ratio
    WINDOW_SIZE_FORWARD_MODE = 10  # WINDOWS_SIZE_FOR_FORWARD_MODE(_2, _3)
    THRESHOLD_NOT_FOUND = 10.0  # USTAR_THRESHOLD_NOT_FOUND, never returned to users

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
        random_state: Optional[int] = 42,
        verbose: int = 0,
    ):
        """Set up moving-point USTAR threshold detection. See the class docstring."""
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty")
        if forward_mode_n < 1:
            raise ValueError(f"forward_mode_n must be >= 1, got {forward_mode_n}")
        if ta_classes_count < 1 or ustar_classes_count < 1:
            raise ValueError("ta_classes_count and ustar_classes_count must be >= 1")

        if nee_col is None:
            candidates = [col for col in df.columns if 'NEE' in col and 'QCF' in col]
            if not candidates:
                candidates = [col for col in df.columns if 'NEE' in col]
            nee_col = candidates[0] if candidates else self._find_column(df, ['NEE', 'NEE_CUT_REF', 'NEE_L3'])
        if ta_col is None:
            candidates = [col for col in df.columns if 'TA_' in col]
            ta_col = candidates[0] if candidates else self._find_column(df, ['TA', 'TSYS', 'TAIR'])
        if ustar_col is None:
            candidates = [col for col in df.columns if 'USTAR' in col]
            ustar_col = candidates[0] if candidates else self._find_column(df, ['U_STAR'])
        if swin_col is None:
            candidates = [col for col in df.columns if 'SW_IN' in col]
            swin_col = candidates[0] if candidates else self._find_column(df, ['SWIN', 'RAD'])

        if nee_col is None:
            raise ValueError("Could not find NEE column. Specify with nee_col parameter.")
        if ta_col is None:
            raise ValueError("Could not find TA (temperature) column. Specify with ta_col parameter.")
        if ustar_col is None:
            raise ValueError("Could not find USTAR column. Specify with ustar_col parameter.")
        if swin_col is None:
            raise ValueError("Could not find SW_IN (shortwave radiation) column. Specify with swin_col parameter.")

        if season_groups is None:
            season_groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        flat = [int(m) for g in season_groups for m in g]
        if any(m < 1 or m > 12 for m in flat) or len(flat) != len(set(flat)) or not all(season_groups):
            # parser.c:165-185 (duplicates are rejected unless -seasons_group_allow_duplicates)
            raise ValueError("season_groups must hold months 1-12, each month in one non-empty group only")

        # A reference, not a copy: this class never mutates df.
        self.df = df
        self.nee_col = nee_col
        self.ta_col = ta_col
        self.ustar_col = ustar_col
        self.swin_col = swin_col
        self.ta_classes_count = ta_classes_count
        self.ustar_classes_count = ustar_classes_count
        self.forward_mode_n = forward_mode_n
        self.bootstrapping_times = bootstrapping_times
        self.random_state = random_state
        self.verbose = verbose
        self.season_groups = season_groups
        self.seasons_count = len(season_groups)

        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'TIMESTAMP' in self.df.columns:
                self.df = self.df.set_index(pd.to_datetime(self.df['TIMESTAMP']))
            else:
                raise ValueError("DataFrame must have datetime index or TIMESTAMP column")

        self._month = self._month_per_group(self.df.index)
        self._prep: Optional[dict] = None

        self.results_ = pd.DataFrame()
        self.annual_thresholds_: Dict[str, float] = {}
        self.ta_class_thresholds_ = pd.DataFrame()
        self.one_big_season_ = False
        self.bootstrap_failed_ = np.zeros(0, dtype=bool)
        self.bootstrap_dropped_ = False

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

        ONEFlux reads TIMESTAMP_END, so a record stamped on the 1st at 00:00
        belongs to the previous month (``dataset.c:438-448``). January wraps to
        December. TIMESTAMP_MIDDLE stamps never sit at 00:00, so for them this
        is the month of the timestamp, which is the same month.
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

    def _rows_per_day(self) -> int:
        """Records per day from the median time step, 0 if the index has no regular step."""
        idx = self.df.index
        if len(idx) < 2:
            return 0
        # asi8 counts in the index unit (ns, us, ms or s), so convert the day to it
        diffs = np.diff(idx.asi8)
        step = int(np.median(diffs))
        if step <= 0 or not np.all(diffs == step):
            return 0
        day = int(pd.Timedelta(days=1) / pd.Timedelta(1, unit=getattr(idx, 'unit', 'ns')))
        return day // step if day % step == 0 else 0

    def _is_year_grid(self, rows_per_day: int) -> bool:
        """True for one calendar year on its full half-hourly or hourly grid.

        ONEFlux reads exactly such a year (``dataset.c:507-512``). The first
        record must belong to the first period of the year: stamped 1 January
        between 00:00 and one time step later (START, MIDDLE or END stamp).
        """
        n = len(self.df)
        if rows_per_day not in (24, 48) or n not in (365 * rows_per_day, 366 * rows_per_day):
            return False
        first = self.df.index[0]
        jan1 = pd.Timestamp(year=first.year, month=1, day=1, tz=first.tz)
        offset = (first - jan1).total_seconds()
        return 0 <= offset <= 86400 / rows_per_day

    def _prepare(self) -> dict:
        """Per-record arrays shared by the main run and the bootstrap (cached)."""
        if self._prep is not None:
            return self._prep
        # Writable copies: pandas can hand out read-only views, and numba compiles
        # (and caches) a separate kernel version for read-only arrays.
        nee = self.df[self.nee_col].to_numpy(dtype='float64', copy=True)
        ta = self.df[self.ta_col].to_numpy(dtype='float64', copy=True)
        ustar = self.df[self.ustar_col].to_numpy(dtype='float64', copy=True)
        swin = self.df[self.swin_col].to_numpy(dtype='float64', copy=True)
        n = len(nee)

        def present(v):
            # IS_INVALID_VALUE (common.h:40) plus NaN, which dataset.c:459 maps to -9999
            return ~np.isnan(v) & ~((v > -10000.0) & (v <= -9999.0))

        all_valid = present(nee) & present(ta) & present(ustar)  # ALL_VALID flag
        is_night = present(swin) & (swin < self.NIGHT_THRESHOLD)  # dataset.c:483-485
        is_day = ~is_night
        night_valid = all_valid & is_night

        # Main run order (main.c:1014-1036): season after season, month after
        # month in the order given, then record order.
        group_of_month = np.full(13, -1, dtype=np.int64)
        rank_of_month = np.zeros(13, dtype=np.int64)
        for g, months in enumerate(self.season_groups):
            for r, m in enumerate(months):
                group_of_month[int(m)] = g
                rank_of_month[int(m)] = r
        group = group_of_month[self._month]
        pos = np.arange(n)
        sel = np.flatnonzero(night_valid & (group >= 0))
        main_order = sel[np.lexsort((pos[sel], rank_of_month[self._month[sel]], group[sel]))]
        main_samples = np.bincount(group[main_order], minlength=self.seasons_count).astype(np.int64)

        rpd = self._rows_per_day()
        year_grid = self._is_year_grid(rpd)
        if year_grid:
            # Season end rows (main.c:1013-1038), with the leap-year February
            # adding 1 row instead of one day of rows (main.c:1018-1019).
            leap = n == 366 * rpd
            ends = []
            days = 0
            for months in self.season_groups:
                for m in months:
                    if m == 2 and leap:
                        days += 1
                    days += _MONTHS_DAYS[m - 1] * rpd
                ends.append(days)
            ends = np.asarray(ends, dtype=np.int64)
            # bootstrapping.c:173-180: first season whose end lies beyond the row, else season 0
            boot_season = np.searchsorted(ends, pos, side='right').astype(np.int64)
            boot_season[boot_season >= self.seasons_count] = 0
            key_of_row = pos.astype(np.int64)
        else:
            boot_season = group.astype(np.int64)
            rest = np.setdiff1d(pos, main_order, assume_unique=True)
            row_of_key_tmp = np.concatenate([main_order, rest])
            key_of_row = np.empty(n, dtype=np.int64)
            key_of_row[row_of_key_tmp] = np.arange(n)
        row_of_key = np.empty(n, dtype=np.int64)
        row_of_key[key_of_row] = np.arange(n)

        self._prep = dict(
            nee=nee, ta=ta, ustar=ustar, is_day=is_day, all_valid=all_valid,
            rows_count=int(night_valid.sum()), days=int(is_day.sum()),
            main_order=main_order, main_samples=main_samples,
            boot_season=boot_season, key_of_row=key_of_row, row_of_key=row_of_key,
            year_grid=year_grid,
        )
        return self._prep

    def _main_rows(self) -> np.ndarray:
        """NEE, TA, USTAR of the main-run records in the C order (the C ``rows`` array)."""
        p = self._prepare()
        o = p['main_order']
        return np.ascontiguousarray(np.column_stack((p['nee'][o], p['ta'][o], p['ustar'][o])))

    def _main_run(self) -> dict:
        """Run ``ustar_threshold`` once on all records (main.c:979-1141).

        Raises ``ValueError`` where ONEFlux skips the year without a result.
        """
        k = _kernels()
        p = self._prepare()
        tacl, ucl = self.ta_classes_count, self.ustar_classes_count
        rows_count, days = p['rows_count'], p['days']
        if rows_count == 0:
            raise ValueError("Insufficient data: no nighttime records with NEE, TA and USTAR.")
        if rows_count < tacl * ucl:
            # main.c:1093-1099
            raise ValueError(
                f"Insufficient data: {rows_count} nighttime records with NEE, TA and USTAR, "
                f"need at least {tacl * ucl} (ta_classes_count * ustar_classes_count).")
        samples = p['main_samples'].copy()
        # Slots of seasons merged into one big season are never written: keep them -9999
        container = np.full(self.seasons_count * tacl, -9999.0)
        ok, sgc = k.ustar_threshold(
            self._main_rows(), rows_count, days, samples, self.seasons_count, tacl, ucl,
            self.forward_mode_n, self.WINDOW_SIZE_FORWARD_MODE, self.THRESHOLD_CHECK, container)
        if not ok:
            raise ValueError(
                f"Insufficient data: {rows_count} nighttime records with NEE, TA and USTAR and "
                f"{days} daytime records. ONEFlux needs night + day records >= "
                f"{self.MIN_SAMPLES_PERIOD}, or more than {self.MIN_SAMPLES_SEASON} night records.")
        medians = np.array([k.median_ustar_threshold(container, tacl, s) for s in range(sgc)])
        return dict(container=container, sgc=int(sgc), medians=medians, samples=samples)

    def _check_indices(self, indices, n_iter):
        n = len(self.df)
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        if indices.ndim != 2:
            raise ValueError("indices must be a 2-D array (n_iter, n_draws)")
        if n_iter is not None and indices.shape[0] != n_iter:
            raise ValueError(f"indices has {indices.shape[0]} rows, n_iter is {n_iter}")
        if indices.size and (indices.min() < 0 or indices.max() >= n):
            raise ValueError(f"indices must be row positions in [0, {n})")
        return indices

    def _run_bootstrap(self, n_iter, rng, indices):
        """Season medians, annual values and failure flags for each bootstrap draw."""
        if indices is None:
            if n_iter is None:
                n_iter = self.bootstrapping_times
            if rng is None:
                rng = np.random.default_rng(self.random_state)
            n = len(self.df)
            indices = np.empty((n_iter, n), dtype=np.int64)
            for b in range(n_iter):
                # one call per draw keeps the stream of earlier diive versions
                indices[b] = rng.integers(0, n, n)
        else:
            indices = self._check_indices(indices, n_iter)

        # ONEFlux bootstraps only after a successful main run, which also
        # sets the season count the draws start from (sticky collapse).
        main = self._main_run()
        p = self._prepare()
        season_med, annual, failed, _ = _kernels().bootstrap(
            p['nee'], p['ta'], p['ustar'], p['is_day'], p['all_valid'],
            p['boot_season'], p['row_of_key'], p['key_of_row'], indices,
            main['sgc'], self.seasons_count, self.ta_classes_count,
            self.ustar_classes_count, self.forward_mode_n,
            self.WINDOW_SIZE_FORWARD_MODE, self.THRESHOLD_CHECK)
        self.bootstrap_failed_ = failed
        self.bootstrap_dropped_ = bool(failed.any())
        if self.bootstrap_dropped_ and self.verbose >= 1:
            warn(f"{int(failed.sum())} of {len(failed)} bootstrap draws had too few records. "
                 f"ONEFlux drops the whole bootstrap for such a year.", verbose=self.verbose)
        return season_med, annual

    # ------------------------------------------------------------------ public

    def detect(self) -> pd.DataFrame:
        """
        Detect USTAR thresholds for all seasons (Papale et al., 2006).

        Returns
        -------
        pd.DataFrame
            Season thresholds in column 'threshold' (m s-1), NaN where none was found.
            The annual threshold (highest season threshold) is in
            ``annual_thresholds_`` and ``get_annual_thresholds()``.
        """
        if self.verbose >= 1:
            info(f"Detecting USTAR thresholds (Papale et al., 2006) | "
                 f"{len(self.df)} records, {self.seasons_count} seasons, "
                 f"{self.ta_classes_count} TA classes, {self.ustar_classes_count} USTAR classes",
                 verbose=self.verbose)

        main = self._main_run()
        k = _kernels()
        tacl = self.ta_classes_count
        sgc = main['sgc']
        self.one_big_season_ = sgc == 1 and self.seasons_count > 1

        def clean(v):
            return np.nan if (k.is_invalid(v) or v == self.THRESHOLD_NOT_FOUND) else float(v)

        # After a collapse the C keeps one season. Every season shows its value.
        seasons = [clean(main['medians'][min(s, sgc - 1)]) for s in range(self.seasons_count)]
        table = np.array([[clean(main['container'][min(s, sgc - 1) * tacl + c])
                           for s in range(self.seasons_count)] for c in range(tacl)])
        season_labels = [f'Season {i + 1}' for i in range(self.seasons_count)]
        self.results_ = pd.DataFrame({'threshold': seasons}, index=season_labels)
        self.ta_class_thresholds_ = pd.DataFrame(
            table, index=[f'TA class {c + 1}' for c in range(tacl)], columns=season_labels)

        # Highest season value, as write_results selects it (main.c:305-309)
        valid = [v for v in seasons if np.isfinite(v)]
        annual = float(max(valid)) if valid else np.nan
        self.annual_thresholds_ = {'threshold': annual}

        if self.verbose >= 2:
            if self.one_big_season_:
                detail("  Too few records per season: one big season", verbose=self.verbose)
            for label, t in zip(season_labels, seasons, strict=True):
                detail(f"  {label}: {t:.4f} m/s" if np.isfinite(t) else f"  {label}: not found",
                       verbose=self.verbose)
        if self.verbose >= 1:
            info(f"Annual threshold (max across seasons): {annual:.4f} m/s" if np.isfinite(annual)
                 else "Annual threshold: Not found", verbose=self.verbose)
        return self.results_

    def bootstrap_annual_samples(
        self,
        n_iter: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Bootstrap distribution of the annual (highest season) threshold.

        Each draw takes ``len(df)`` record positions with replacement, as
        ONEFlux ``bootstrapping`` does, and reruns the detection on them.

        Parameters
        ----------
        n_iter : int, optional
            Number of draws. Default ``bootstrapping_times``, or the number of
            rows of ``indices``.
        rng : numpy.random.Generator, optional
            Generator for the draws. Default ``np.random.default_rng(random_state)``.
        indices : np.ndarray, optional
            Drawn record positions, shape (n_iter, n_draws). Replaces the
            random draws, for example to feed the same draws to ONEFlux.

        Returns
        -------
        np.ndarray
            One value per draw (length n_iter). NaN where the draw gave no
            threshold or failed. ``bootstrap_failed_`` marks the failed draws
            and ``bootstrap_dropped_`` tells whether ONEFlux would have dropped
            the whole bootstrap.

        Raises
        ------
        ValueError
            Where ONEFlux skips the year (see Notes of the class).
        """
        if self.verbose >= 1:
            info("Running bootstrap draws...", verbose=self.verbose)
        _, annual = self._run_bootstrap(n_iter, rng, indices)
        return annual

    def bootstrap(self, n_iter: Optional[int] = None) -> pd.DataFrame:
        """
        Bootstrap the season and annual thresholds.

        Draws are seeded with ``random_state``. Draws without a threshold, and
        failed draws (see ``bootstrap_failed_``), are left out of the statistics.

        Parameters
        ----------
        n_iter : int, optional
            Number of draws (default ``bootstrapping_times``).

        Returns
        -------
        pd.DataFrame
            mean, std, p05, p50 and p95 per season, plus an 'Annual' row for the
            annual (highest season) threshold used for filtering.
        """
        if n_iter is None:
            n_iter = self.bootstrapping_times
        if self.verbose >= 1:
            info(f"Running {n_iter} bootstrap iterations...", verbose=self.verbose)
        season_med, annual = self._run_bootstrap(n_iter, np.random.default_rng(self.random_state), None)
        rows = [self._stats(season_med[:, s]) for s in range(self.seasons_count)]
        rows.append(self._stats(annual))
        index = [f'Season {i + 1}' for i in range(self.seasons_count)] + ['Annual']
        return pd.DataFrame(rows, index=index)

    @staticmethod
    def _stats(vals) -> Dict[str, float]:
        """Summary statistics of the finite bootstrap thresholds."""
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {'mean': np.nan, 'std': np.nan, 'p05': np.nan, 'p50': np.nan, 'p95': np.nan}
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
            threshold_str = f"{val:.4f}" if np.isfinite(val) else "Not found"
            lines.append(f"{idx:<15} {threshold_str:<20}")
        return "\n".join(lines)

    def get_annual_thresholds(self) -> Dict[str, float]:
        """
        Get the annual USTAR threshold (highest season threshold).

        Returns
        -------
        Dict[str, float]
            ``{'threshold': value}`` in m s-1, NaN if detection found none.

        Notes
        -----
        Papale et al. (2006): "the whole data set is filtered according to the
        highest threshold found".
        """
        if not self.annual_thresholds_:
            raise RuntimeError("Detection not yet performed. Call detect() first.")
        return self.annual_thresholds_.copy()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UstarMovingPointDetection("
            f"n_records={len(self.df)}, "
            f"ta_classes={self.ta_classes_count}, "
            f"ustar_classes={self.ustar_classes_count}, "
            f"seasons={self.seasons_count})"
        )
