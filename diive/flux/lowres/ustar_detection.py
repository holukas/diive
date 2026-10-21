"""
USTAR DETECTION: ONE ENTRY POINT FOR THE U* THRESHOLD DETECTION PORTS
=====================================================================

Select the friction velocity (u*) threshold detection with ``mode``, the way
``FluxMDS(mode=...)`` selects the MDS gap-filling rules. ONEFlux, the approach
used by ICOS and FLUXNET2015, is the default.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diive.core.utils.console import console as _console, detail, info, warn

USTAR_MODES = ('oneflux', 'oneflux_mp', 'oneflux_cp', 'reddyproc')

# Methods each mode runs per calendar year. 'mp' = moving point (Papale et al.
# 2006, ONEFlux ustar_mp), 'cp' = change point (Barr et al. 2013, ONEFlux ustar_cp).
_MODE_METHODS = {
    'oneflux': ('mp', 'cp'),
    'oneflux_mp': ('mp',),
    'oneflux_cp': ('cp',),
}

# ONEFlux runs 100 bootstraps per method and year (ustar_mp types.h,
# ustar_cp launch.m:309). REddyProc draws 200 rows: the estimate on the
# original data plus 199 resamples (EddyUStarFilterDP.R:1280).
_DEFAULT_N_ITER = {'oneflux': 100, 'oneflux_mp': 100, 'oneflux_cp': 100, 'reddyproc': 200}
_DEFAULT_PERCENTILES = {'oneflux': (16, 50, 84), 'oneflux_mp': (16, 50, 84),
                        'oneflux_cp': (16, 50, 84), 'reddyproc': (5, 50, 95)}

# ONEFlux nee_proc reports CUT only for records of at least 3 years (dataset.c:5103).
ONEFLUX_CUT_MIN_YEARS = 3


def oneflux_percentile(values, percentile: float) -> float:
    """Percentile as ONEFlux nee_proc computes it (``common.c::get_percentile``).

    Nearest rank on the sorted valid values: index = ROUND(p / 100 * n) - 1, with
    ``p / 100 * n`` evaluated in single precision (``percentile`` is a C
    ``float``), clamped to the first and last value. NaN values are invalid and
    left out. Returns NaN when no value is valid (ONEFlux reports an error there).
    """
    v = np.asarray(values, dtype=np.float64)
    v = np.sort(v[np.isfinite(v)])
    n = len(v)
    if n == 0:
        return np.nan
    if n == 1:
        return float(v[0])
    # float / int and float * int stay float under FLT_EVAL_METHOD 0, the 0.5 of
    # ROUND is a double constant, so the sum is done in double.
    x = np.float32(np.float32(percentile) / np.float32(100)) * np.float32(n)
    x = float(np.float32(x))
    index = int(x + 0.5) if x >= 0 else int(x - 0.5)
    index -= 1
    if index < 0:
        index = 0
    if index >= n:
        return float(v[-1])
    return float(v[index])


def _oneflux_mp_text(values: np.ndarray) -> np.ndarray:
    """MP bootstrap values as nee_proc reads them back: written with ``%g``
    (6 significant digits, ustar_mp bootstrapping.c:269), parsed as double."""
    return np.array([float(f"{v:g}") if np.isfinite(v) else np.nan for v in values])


def _oneflux_cp_text(values: np.ndarray) -> np.ndarray:
    """CP bootstrap values as nee_proc reads them back: written by ``dlmwrite``
    with precision 8 (ustar_cp launch.m:326), parsed as double, 'NaN' invalid."""
    return np.array([float(f"{v:.8g}") if np.isfinite(v) else np.nan for v in values])


def _qc_auto_text(df: pd.DataFrame) -> pd.DataFrame:
    """Values as ustar_mp and ustar_cp read them: qc_auto writes its yearly input
    files with C ``%g``, 6 significant digits (qc_auto main.c:2255). Without this
    rounding a few records can fall into another class than in ONEFlux."""
    out = df.copy()
    for col in out.columns:
        v = out[col].to_numpy(dtype=float, copy=True)
        ok = np.isfinite(v)
        v[ok] = np.array([float(f"{x:g}") for x in v[ok]])
        out[col] = v
    return out


def _to_middle(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` on TIMESTAMP_MIDDLE. An index named TIMESTAMP_START or
    TIMESTAMP_END is shifted by half a period, any other index is taken as the
    middle of the period (diive's convention)."""
    if df.index.name in ('TIMESTAMP_START', 'TIMESTAMP_END'):
        from diive.core.times.times import convert_series_timestamp_to_middle
        return convert_series_timestamp_to_middle(df)
    return df


def _record_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Time step of the records: the index frequency, else the median spacing."""
    if index.freq is not None:
        return pd.Timedelta(index.freq)
    if len(index) < 2:
        raise ValueError("Need at least two records to know the time step.")
    # Median of the spacing as Timedeltas: asi8 is in the index's own unit
    # (ns, us, ...), which pandas no longer fixes to nanoseconds.
    return pd.Series(index).diff().median()


def _full_year(df: pd.DataFrame, year: int, step: pd.Timedelta) -> pd.DataFrame:
    """Records of one calendar year on the complete grid of that year (missing
    records NaN), as the yearly input files of ONEFlux hold them."""
    half = step / 2
    grid = pd.date_range(pd.Timestamp(year, 1, 1) + half,
                         pd.Timestamp(year + 1, 1, 1) - half, freq=step,
                         name='TIMESTAMP_MIDDLE')
    out = df.reindex(grid)
    out.index.freq = step
    return out


class UstarThresholdDetection:
    r"""Detect the friction velocity (u\*) threshold of a site with one of several methods.

    Below the threshold, nighttime turbulence is too weak and the measured flux
    underestimates the true exchange. ``mode`` selects the method:

    - ``'oneflux'`` (default): ONEFlux as used for FLUXNET2015 and ICOS
      (Pastorello et al. 2020). For each calendar year, the moving point method
      (MP, Papale et al. 2006, :class:`UstarMovingPointDetection`) and the change
      point method (CP, Barr et al. 2013, :class:`UstarChangePointDetection`) each
      give 100 bootstrap thresholds. Both sets are pooled: the VUT (variable u\*
      threshold) of year Y uses the thresholds of Y-1, Y and Y+1, the CUT
      (constant u\* threshold) the thresholds of all years.
    - ``'oneflux_mp'``: as ``'oneflux'`` with the MP thresholds only.
    - ``'oneflux_cp'``: as ``'oneflux'`` with the CP thresholds only.
    - ``'reddyproc'``: REddyProc 1.3.4 (Wutzler et al. 2018,
      :class:`UstarReddyProcDetection`), a moving point variant with seasons
      crossing the year boundary and a bootstrap within each season.

    Each mode reproduces its reference software, including the way it picks the
    percentiles (ONEFlux: nearest rank, REddyProc: R quantile type 7). The ONEFlux
    modes do not include the REF threshold, which ONEFlux selects afterwards by
    model efficiency over 40 gap-filled versions of NEE.

    The input is split into calendar years. In the ONEFlux modes each year is
    prepared as ONEFlux's yearly input files are: padded to the full grid of the
    year, so the bootstrap resamples the same rows, and rounded to 6 significant
    digits, as ONEFlux writes them.

    Args:
        df: Data with a DatetimeIndex (TIMESTAMP_MIDDLE, or an index named
            TIMESTAMP_START or TIMESTAMP_END), half-hourly or hourly.
        nee_col: NEE (umol m-2 s-1), after outlier removal and before u\* filtering.
        ta_col: Air temperature (deg C).
        ustar_col: Friction velocity (m s-1).
        swin_col: Incoming shortwave radiation (W m-2), defines the night.
        mode: ``'oneflux'`` (default), ``'oneflux_mp'``, ``'oneflux_cp'`` or
            ``'reddyproc'``.
        n_iter: Bootstrap draws per method and year. Default: 100 for the ONEFlux
            modes, 200 for REddyProc (its original estimate plus 199 resamples).
        percentiles: Percentiles of the bootstrap distribution to report. Default:
            (16, 50, 84) for the ONEFlux modes, (5, 50, 95) for REddyProc.
        random_state: Seed for the bootstrap, so a run can be repeated. None draws
            new samples on every run. The seed is derived per year and method, so
            years and methods are independent draws. ``'oneflux'`` (ONEFlux modes)
            draws the CP resamples from MATLAB's default random stream, continued
            over the years in order as in a ONEFlux run, so the CP thresholds equal
            those of ONEFlux. ONEFlux seeds the MP draws with the clock, so MP
            draws are new on every run then.
        n_jobs: Parallel processes over years and methods (1 = sequential, -1 = all CPUs).
        mp_kwargs: Extra arguments for the MP backend (ONEFlux modes).
        cp_kwargs: Extra arguments for the CP backend (ONEFlux modes).
        reddyproc_kwargs: Extra arguments for the REddyProc backend.
        verbose: Console verbosity: 0 or False = silent, 2 or True = progress.

    Attributes:
        samples\_: Per year and method, the bootstrap thresholds as used for pooling
            (NaN = no threshold in that draw).
        failures\_: Per year and method, why no thresholds were available.

    Example:
        >>> import diive as dv
        >>> df = dv.load_exampledata_parquet_lae()
        >>> det = dv.flux.UstarThresholdDetection(
        ...     df, nee_col='NEE_L3.1_L3.2_QCF_IRGA72', ta_col='TA_T1_47_1_gfXG_IRGA72',
        ...     ustar_col='USTAR_IRGA72', swin_col='SW_IN_T1_47_1_gfXG_IRGA72')
        >>> vut = det.run()  # then det.get_cut_threshold() / det.summary()

        See `examples/flux/lowres/flux_ustar_detection.py` and
        `examples/flux/lowres/flux_ustar_method_comparison.py`.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 nee_col: str,
                 ta_col: str,
                 ustar_col: str,
                 swin_col: str,
                 mode: str = 'oneflux',
                 n_iter: Optional[int] = None,
                 percentiles: Optional[Tuple[float, ...]] = None,
                 random_state: Optional[int | str] = 42,
                 n_jobs: int = 1,
                 mp_kwargs: Optional[dict] = None,
                 cp_kwargs: Optional[dict] = None,
                 reddyproc_kwargs: Optional[dict] = None,
                 verbose: int | bool = 0):
        """Set up the u* threshold detection. See the class docstring."""
        mode = str(mode).lower()
        if mode not in USTAR_MODES:
            raise ValueError(f"mode must be one of {USTAR_MODES}, got {mode!r}.")
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")
        cols = [nee_col, ta_col, ustar_col, swin_col]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in df: {missing}")

        self.df = _to_middle(df[cols].copy()).sort_index()
        self.nee_col = nee_col
        self.ta_col = ta_col
        self.ustar_col = ustar_col
        self.swin_col = swin_col
        self.mode = mode
        self.n_iter = int(n_iter) if n_iter is not None else _DEFAULT_N_ITER[mode]
        self.percentiles = tuple(sorted(percentiles if percentiles is not None
                                        else _DEFAULT_PERCENTILES[mode]))
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.mp_kwargs = dict(mp_kwargs or {})
        self.cp_kwargs = dict(cp_kwargs or {})
        self.reddyproc_kwargs = dict(reddyproc_kwargs or {})
        self.verbose = verbose

        self.years_: List[int] = sorted(self.df.index.year.unique().tolist())
        self.samples_: Dict[int, Dict[str, np.ndarray]] = {}
        self.failures_: Dict[int, Dict[str, str]] = {}
        self.vut_samples_: Dict[int, np.ndarray] = {}
        self.cut_samples_: Optional[np.ndarray] = None
        self.annual_stats_: Optional[pd.DataFrame] = None
        self.backend_: Optional[object] = None  # REddyProc backend after run()

    # ------------------------------------------------------------------ helpers

    def _seed(self, year: int, method: str):
        if self.random_state is None:
            return None
        if self.random_state == 'oneflux':
            # MP: ONEFlux seeds C rand() with the clock, so no draw can be repeated.
            # CP: one MATLAB stream for all years (see _run_oneflux).
            return None if method == 'mp' else self._matlab_stream
        return [int(self.random_state), int(year), 0 if method == 'mp' else 1]

    def _percentile(self, values: np.ndarray, p: float) -> float:
        if self.mode == 'reddyproc':
            # R type 7 over all draws, NaN when fewer than 40 % are finite (DP:1383-1387)
            from diive.flux.lowres.ustar_reddyproc_detection import UstarReddyProcDetection
            v = np.asarray(values, dtype=float)
            if not len(v):
                return np.nan
            return float(UstarReddyProcDetection.quantiles(v[:, None], [p / 100])[0, 0])
        return oneflux_percentile(values, p)

    # ------------------------------------------------------------------ ONEFlux

    def _run_oneflux(self) -> None:
        step = _record_step(self.df.index)
        tasks = []
        for year in self.years_:
            df_year = _qc_auto_text(_full_year(self.df, year, step))
            for method in _MODE_METHODS[self.mode]:
                tasks.append((year, method, df_year))

        n_workers = self.n_jobs
        if n_workers == -1:
            import os
            n_workers = os.cpu_count() or 1
        if self.random_state == 'oneflux':
            # ONEFlux runs ustar_cp once for all years, so MATLAB's default
            # stream (twister, seed 0) continues from year to year in file order.
            # The years must therefore run one after another.
            from diive.flux.lowres._ustar_cp_numerics import matlab_default_stream
            self._matlab_stream = matlab_default_stream()
            n_workers = 1
        if n_workers > 1 and len(tasks) > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_workers)(
                delayed(_oneflux_year_worker)(
                    year, method, df_year, self._backend_kwargs(method),
                    self.n_iter, self._seed(year, method))
                for year, method, df_year in tasks)
        else:
            results = [_oneflux_year_worker(year, method, df_year, self._backend_kwargs(method),
                                            self.n_iter, self._seed(year, method))
                       for year, method, df_year in tasks]

        for year, method, samples, failure in results:
            self.samples_.setdefault(year, {})
            self.failures_.setdefault(year, {})
            if failure:
                self.failures_[year][method] = failure
                detail(f"  {year} {method.upper()}: no thresholds ({failure})", verbose=self.verbose)
                continue
            self.samples_[year][method] = samples
            n_ok = int(np.isfinite(samples).sum())
            info(f"  {year} {method.upper()}: {n_ok}/{len(samples)} bootstrap thresholds",
                 verbose=self.verbose)

        # VUT: thresholds of the year and its neighbours (dataset.c::get_uts_y).
        for year in self.years_:
            pool = [self.samples_.get(y, {}).get(m)
                    for y in (year - 1, year, year + 1) if y in self.years_
                    for m in _MODE_METHODS[self.mode]]
            pool = [p for p in pool if p is not None]
            self.vut_samples_[year] = np.concatenate(pool) if pool else np.array([])
        # CUT: thresholds of all years (dataset.c::get_uts_c).
        pool = [s for y in self.years_ for s in self.samples_.get(y, {}).values()]
        self.cut_samples_ = np.concatenate(pool) if pool else np.array([])

    def _backend_kwargs(self, method: str) -> dict:
        kw = dict(nee_col=self.nee_col, ta_col=self.ta_col, ustar_col=self.ustar_col,
                  swin_col=self.swin_col)
        kw.update(self.mp_kwargs if method == 'mp' else self.cp_kwargs)
        return kw

    # ------------------------------------------------------------------ REddyProc

    def _run_reddyproc(self) -> None:
        if isinstance(self.random_state, str):
            raise ValueError("random_state='oneflux' applies to the ONEFlux modes only.")
        from diive.flux.lowres.ustar_reddyproc_detection import UstarReddyProcDetection
        backend = UstarReddyProcDetection(
            self.df, nee_col=self.nee_col, ta_col=self.ta_col, ustar_col=self.ustar_col,
            swin_col=self.swin_col, verbose=0, **self.reddyproc_kwargs)
        backend.run()
        # Row 0 is REddyProc's estimate on the original data, rows 1.. the
        # resamples. Its quantiles include row 0, as in REddyProc (DP:1374).
        backend.bootstrap(n_sample=self.n_iter, probs=[p / 100 for p in self.percentiles],
                          seed=self.random_state)
        self.backend_ = backend
        boot = backend.boot_samples_
        for year in self.years_:
            col = f'year_{year}'
            s = boot[col].to_numpy(dtype=float) if col in boot else np.array([])
            self.samples_[year] = {'reddyproc': s}
            self.failures_[year] = {} if np.isfinite(s).any() else {
                'reddyproc': 'no threshold in any bootstrap draw'}
            self.vut_samples_[year] = s
        # CUT is not part of REddyProc. diive pools the draws of all years.
        pool = [self.vut_samples_[y] for y in self.years_ if len(self.vut_samples_[y])]
        self.cut_samples_ = np.concatenate(pool) if pool else np.array([])

    # ------------------------------------------------------------------ public

    def run(self) -> pd.DataFrame:
        """Run the detection and return the VUT table (one row per year,
        columns ``p{percentile}``)."""
        info(f"u* threshold detection, mode={self.mode!r}: {self.n_iter} bootstraps "
             f"per year and method, {len(self.years_)} year(s)", verbose=self.verbose)
        self.samples_, self.failures_, self.vut_samples_ = {}, {}, {}
        if self.mode == 'reddyproc':
            self._run_reddyproc()
        else:
            self._run_oneflux()
            if len(self.years_) < ONEFLUX_CUT_MIN_YEARS:
                detail(f"  CUT from {len(self.years_)} year(s): ONEFlux itself reports CUT "
                       f"only for {ONEFLUX_CUT_MIN_YEARS} or more years.", verbose=self.verbose)

        rows = [{f'p{_pfmt(p)}': self._percentile(self.vut_samples_[y], p)
                 for p in self.percentiles} for y in self.years_]
        self.annual_stats_ = pd.DataFrame(rows, index=pd.Index(self.years_, name='year'))
        if self.annual_stats_.isna().all(axis=None):
            warn("u* threshold detection found no threshold in any year. Check that the "
                 "data have enough valid nighttime records.")
        return self.annual_stats_

    def get_vut_thresholds(self) -> pd.DataFrame:
        """VUT thresholds: one row per year, columns ``p{percentile}``."""
        if self.annual_stats_ is None:
            raise RuntimeError("Call run() before get_vut_thresholds().")
        return self.annual_stats_

    def get_cut_threshold(self) -> Dict[str, float]:
        """CUT thresholds: ``{'p{percentile}': value}`` from the thresholds of all years."""
        if self.cut_samples_ is None:
            raise RuntimeError("Call run() before get_cut_threshold().")
        return {f'p{_pfmt(p)}': self._percentile(self.cut_samples_, p) for p in self.percentiles}

    def summary(self) -> str:
        """Text summary of the VUT and CUT thresholds."""
        if self.annual_stats_ is None:
            return "No results. Call run() first."
        cols = list(self.annual_stats_.columns)
        w = 10
        lines = [f"u* thresholds (m s-1), mode={self.mode!r}", "=" * 60,
                 f"{'Year':<8}" + "".join(f"{c:>{w}}" for c in cols) + "   bootstrap values"]
        for y in self.years_:
            row = self.annual_stats_.loc[y]
            vals = "".join(f"{row[c]:>{w}.4f}" if np.isfinite(row[c]) else f"{'N/A':>{w}}"
                           for c in cols)
            used = {m: int(np.isfinite(s).sum()) for m, s in self.samples_.get(y, {}).items()}
            fails = self.failures_.get(y, {})
            note = ", ".join([f"{m.upper()} {n}" for m, n in used.items()]
                             + [f"{m.upper()} none" for m in fails])
            lines.append(f"{y:<8}{vals}   {note}")
        cut = self.get_cut_threshold()
        lines.append("-" * 60)
        lines.append(f"{'CUT':<8}" + "".join(f"{v:>{w}.4f}" if np.isfinite(v) else f"{'N/A':>{w}}"
                                             for v in cut.values()))
        return "\n".join(lines)

    def report(self) -> None:
        """Print :meth:`summary`."""
        _console.print(self.summary())

    def __repr__(self) -> str:
        return (f"UstarThresholdDetection(mode={self.mode!r}, years={self.years_}, "
                f"n_iter={self.n_iter}, percentiles={self.percentiles})")


def _pfmt(p: float) -> str:
    """Percentile label: 16 -> '16', 2.5 -> '2.5'."""
    return f"{p:g}"


def _oneflux_year_worker(year: int, method: str, df_year: pd.DataFrame, kwargs: dict,
                         n_iter: int, seed) -> tuple:
    """Bootstrap one method for one year, return the values as nee_proc reads them.

    Module level so joblib can pickle it. Returns ``(year, method, samples, failure)``.
    ``failure`` is set when ONEFlux would write no usable threshold file for the
    year: ustar_mp drops the whole bootstrap section when one draw fails
    (bootstrapping.c:194-201), ustar_cp writes no file when cpdAssign reports a
    failure (launch.m:324-341).
    """
    rng = seed if isinstance(seed, np.random.RandomState) else np.random.default_rng(seed)
    try:
        if method == 'mp':
            from diive.flux.lowres.ustar_mp_detection import UstarMovingPointDetection
            det = UstarMovingPointDetection(df_year, verbose=0, **kwargs)
            samples = np.asarray(det.bootstrap_annual_samples(n_iter, rng=rng), dtype=float)
            if det.bootstrap_dropped_:
                return year, method, None, "a bootstrap draw failed, ONEFlux drops all MP values"
            # A draw without a threshold stays in the file as invalid and is left
            # out of the percentiles on its own.
            return year, method, _oneflux_mp_text(samples), None
        from diive.flux.lowres.ustar_cp_detection import UstarChangePointDetection
        det = UstarChangePointDetection(df_year, verbose=0, **kwargs)
        samples = np.asarray(det.bootstrap_annual_samples(n_iter, rng=rng), dtype=float)
        failure = getattr(det, 'failure_', None)
        if failure:
            return year, method, None, str(failure)
        return year, method, _oneflux_cp_text(samples), None
    except ValueError as err:
        return year, method, None, f"{err}"
