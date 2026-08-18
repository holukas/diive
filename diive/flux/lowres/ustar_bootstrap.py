"""
USTAR BOOTSTRAP THRESHOLDS: MULTI-YEAR ANNUAL THRESHOLD ESTIMATION
===================================================================

Multi-year bootstrap wrapper for USTAR threshold detection.

Part of the diive library: https://github.com/holukas/diive
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Dict, List, Optional, Tuple

from diive.core.utils.console import info, detail, warn


def _bootstrap_window_worker(
    year: int,
    df_window: pd.DataFrame,
    detector_class: type,
    detector_kwargs: dict,
    n_iter: int,
    seed: Optional[int] = None,
) -> tuple:
    """
    Bootstrap all iterations for one window (module-level for picklability).

    Fast path: if the detector exposes ``bootstrap_annual_samples`` it is built once on the
    window and resamples internally on pre-extracted arrays (no per-iteration DataFrame copy
    or detector reconstruction). Otherwise falls back to the generic resample-and-detect loop.

    ``seed`` must already be derived per year by the caller. Both paths were previously
    unseeded, so VUT and CUT thresholds differed between runs. A single shared seed would
    be worse than none: every window would draw the same resample positions, correlating
    years that are supposed to be independent draws. None keeps the old random behaviour.
    """
    if hasattr(detector_class, 'bootstrap_annual_samples'):
        try:
            detector = detector_class(df_window, verbose=0, **detector_kwargs)
            samples = [float(s) for s in
                       detector.bootstrap_annual_samples(n_iter, rng=np.random.default_rng(seed))
                       if s is not None and not np.isnan(s)]
        except Exception as err:
            # Carry the reason out rather than swallowing it: an empty sample list
            # is indistinguishable between "too few records", "a mistyped column
            # name" and "the detection genuinely found nothing", and the caller
            # reports NaN thresholds either way.
            return year, [], f"{type(err).__name__}: {err}"
        return year, samples, None

    samples = []
    last_err = None
    for i in range(n_iter):
        # Offset by the iteration so the draws differ from each other, as the fast path's
        # single Generator does across its own iterations.
        df_boot = df_window.sample(n=len(df_window), replace=True,
                                   random_state=None if seed is None else seed + i)
        try:
            detector = detector_class(df_boot, verbose=0, **detector_kwargs)
            detector.detect()
            annual = detector.get_annual_thresholds()
            threshold = annual.get('threshold')
            if threshold is not None and not np.isnan(threshold):
                samples.append(float(threshold))
        except Exception as err:
            last_err = f"{type(err).__name__}: {err}"
            continue
    return year, samples, (last_err if not samples else None)


class UstarBootstrapThresholds:
    r"""
    Multi-year bootstrap USTAR threshold estimation — VUT and CUT.

    Wrapper around any USTAR detection class that implements detect() and
    get_annual_thresholds(). For each central year, bootstrap resampling is run
    on a 3-year window (central year plus its two neighbors); the distribution
    of annual thresholds is then summarized two ways, following the FLUXNET /
    ONEFlux convention:

    - **VUT (Variable U\* Threshold)** — one threshold *per year*. Returned by
      :meth:`run` / :meth:`get_vut_thresholds` (the ``annual_stats_`` table):
      each year's row holds the requested percentiles of that year's bootstrap
      distribution, so the threshold varies year to year.
    - **CUT (Constant U\* Threshold)** — one threshold for the *whole record*.
      Returned by :meth:`get_cut_threshold`: all bootstrap samples from every
      year are pooled into a single distribution and the percentiles taken once,
      giving one constant threshold applied uniformly to every year.

    .. note::
       **VUT here is smoothed across a 3-year window**, which differs from the
       strict ONEFlux VUT (each year estimated from that single year alone). The
       central year is pooled with its two neighbors (see *Window rules*) before
       bootstrapping, deliberately trading a little year-to-year resolution for a
       more stable, cleaner per-year threshold — sites often have too few good
       nighttime records in a single year for a robust single-year estimate.
       CUT (pooled across all years) matches the ONEFlux CUT meaning directly.

    Window rules
    ------------
    - Normal case: window = [year-1, year, year+1]
    - First year (no predecessor): window = [year, year+1, year+2]
    - Last year (no successor): window = [year-2, year-1, year]
    - 2 years total: both years use the full 2-year dataset
    - 3 years total: all years use the full 3-year dataset
    - 1 year total: that year is used on its own

    Parameters
    ----------
    df : pd.DataFrame
        Full time series data with DatetimeIndex. May span multiple years.
    detector_class : type
        USTAR detection class to use. Must implement detect() and
        get_annual_thresholds(). Supported: UstarMovingPointDetection,
        UstarVekuriThresholdDetection.
    detector_kwargs : dict, optional
        Keyword arguments forwarded to the detector constructor (excluding verbose,
        which is always set to 0 for inner detectors).
    n_iter : int, default=100
        Bootstrap iterations per central year.
    percentiles : tuple, default=(16, 50, 84)
        Percentiles to compute from the bootstrap distribution.
        p50 is the recommended threshold; p16/p84 bound the uncertainty.
    n_jobs : int, default=1
        Number of parallel worker processes. Each window (central year) runs
        its full N iterations in one worker process.
        1 = sequential; -1 = use all available CPUs; N = use N processes.
        Uses joblib/loky backend, which works correctly on Windows without
        requiring an if __name__ == '__main__' guard in scripts.
    random_state : int or None, default=42
        Seed for the bootstrap resampling, so the same input gives the same VUT and CUT
        thresholds. Pass None for a fresh draw on every run. The seed is derived per
        window year, so results do not depend on ``n_jobs`` or on the order in which
        windows finish, and separate years remain independent draws — a single shared
        seed would make every window resample the same positions.
    verbose : int, default=0
        Verbosity: 0=silent, 1=progress per year.

    Attributes
    ----------
    annual_stats\_ : pd.DataFrame
        Per-year (VUT) bootstrap percentile thresholds. Also returned by
        :meth:`get_vut_thresholds`.
        Index: year (int), columns: p16, p50, p84 (or custom percentiles).
    years\_ : list of int
        Calendar years present in the input data.

    Examples
    --------
    >>> import diive as dv
    >>> df = dv.load_exampledata_parquet_lae()
    >>> boot = dv.flux.UstarBootstrapThresholds(
    ...     df, detector_class=dv.flux.UstarMovingPointDetection,
    ...     n_iter=100, percentiles=(16, 50, 84))
    >>> annual = boot.run()  # then boot.get_cut_threshold() / boot.summary()

    See Also
    --------
    UstarMovingPointDetection : ONEFlux moving point detection (Papale et al. 2006).
    UstarVekuriThresholdDetection : Quantile-based detection (Vekuri approach).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        detector_class: type,
        detector_kwargs: Optional[dict] = None,
        n_iter: int = 100,
        percentiles: Tuple[int, ...] = (16, 50, 84),
        n_jobs: int = 1,
        random_state: Optional[int] = 42,
        verbose: int = 0,
    ):
        """Set up multi-year bootstrap USTAR threshold detection. See the class docstring.

        ``random_state`` seeds the resampling, so the same input gives the same VUT and
        CUT thresholds; pass None for the old per-run variation. The seed is derived per
        window year, so the result does not depend on ``n_jobs`` or on the order windows
        finish in, and separate years still draw independently.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        self.df = df.copy()
        self.detector_class = detector_class
        # Strip 'verbose' from kwargs — always forced to 0 for inner detectors
        self.detector_kwargs = {k: v for k, v in (detector_kwargs or {}).items() if k != 'verbose'}
        self.n_iter = n_iter
        self.percentiles = sorted(percentiles)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.years_: List[int] = sorted(self.df.index.year.unique().tolist())

        # Results storage (populated by run())
        self._raw_samples_: Dict[int, List[float]] = {}
        self._window_years_: Dict[int, List[int]] = {}
        self.annual_stats_: Optional[pd.DataFrame] = None
        self._all_samples_: Optional[List[float]] = None

    def _seed_for(self, year: int) -> Optional[int]:
        """Per-year bootstrap seed, or None when seeding is off.

        Derived from the year rather than shared, so each window is an independent draw
        and the result is identical whether the windows run serially or in parallel.
        """
        return None if self.random_state is None else int(self.random_state) + int(year)

    def _get_window_years(self, year_idx: int) -> List[int]:
        """
        Return the list of years to pool for bootstrap around a central year.

        Uses a 3-year sliding window with edge-case handling:
        - n <= 3: always return all years
        - first year: return years[0:3]
        - last year: return years[-3:]
        - otherwise: return [year-1, year, year+1]
        """
        n = len(self.years_)
        if n <= 3:
            return self.years_[:]
        if year_idx == 0:
            return self.years_[0:3]
        if year_idx == n - 1:
            return self.years_[n - 3:n]
        return self.years_[year_idx - 1:year_idx + 2]

    def run(self) -> pd.DataFrame:
        """
        Run the VUT bootstrap using a 3-year sliding window.

        For each central year, pools data from the window (central year plus its
        two neighbors, with edge-case handling), resamples N times with replacement,
        runs the detection method, collects the annual threshold, then computes
        percentiles from the resulting distribution. The result is the **VUT**
        (variable, per-year) threshold table; pool it across years with
        :meth:`get_cut_threshold` for the **CUT** (constant) threshold.

        Returns
        -------
        pd.DataFrame
            Per-year (VUT) bootstrap percentile thresholds.
            Index: year (int), columns: p{percentile} for each requested percentile.
            p50 is the recommended annual USTAR threshold for filtering.
        """
        n_workers = os.cpu_count() if self.n_jobs == -1 else self.n_jobs

        # With the per-detector fast path, each window is cheap; spawning a process pool for
        # only a handful of windows costs more (process startup + DataFrame pickling) than it
        # saves. Run sequentially below this point regardless of the requested n_jobs.
        if n_workers > 1 and len(self.years_) <= 3:
            # `verbose=` is required, not decoration: detail() defaults to
            # verbose=VERBOSE_PROGRESS (2) while its own min_level is
            # VERBOSE_DEBUG (3), so a bare detail() can never print at any
            # setting - the `verbose >= 1` guard around it was hiding that.
            detail(f"  {len(self.years_)} window(s): running sequentially "
                   f"(too few to amortize process-pool overhead)",
                   verbose=self.verbose)
            n_workers = 1

        if self.verbose >= 1:
            pct_labels = '/'.join(f'p{p}' for p in self.percentiles)
            mode = f"parallel ({n_workers} workers)" if n_workers > 1 else "sequential"
            info(f"Bootstrap USTAR thresholds: {self.detector_class.__name__}, "
                 f"{self.n_iter} iterations x {len(self.years_)} windows ({mode}), "
                 f"percentiles: [{pct_labels}]")

        self._raw_samples_ = {}
        self._window_years_ = {}

        # Pre-compute window years and window DataFrames for all central years
        windows: Dict[int, List[int]] = {}
        df_windows: Dict[int, pd.DataFrame] = {}
        for year_idx, year in enumerate(self.years_):
            w = self._get_window_years(year_idx)
            windows[year] = w
            self._window_years_[year] = w
            df_windows[year] = self.df[self.df.index.year.isin(w)]

        if n_workers <= 1:
            # Sequential execution (same worker as the parallel path, so the fast path applies)
            for year in self.years_:
                df_window = df_windows[year]

                if self.verbose >= 1:
                    win_str = '/'.join(str(y) for y in windows[year])
                    info(f"  {year} [window: {win_str}] ({len(df_window)} records)...")

                _, samples, err = _bootstrap_window_worker(
                    year, df_window, self.detector_class, self.detector_kwargs, self.n_iter,
                    self._seed_for(year)
                )
                self._raw_samples_[year] = samples

                if samples:
                    if self.verbose >= 1:
                        p50 = float(np.percentile(samples, 50))
                        info(f"    p50={p50:.4f} m/s ({len(samples)}/{self.n_iter} ok)")
                else:
                    # A window that yields nothing becomes a NaN threshold, so say
                    # so at ERROR level (always visible) with the reason attached.
                    warn(f"  {year}: no valid thresholds"
                         + (f" - {err}" if err else " (detection found none)"))

        else:
            # Parallel execution via joblib/loky (Windows-safe, no __main__ guard needed)
            if self.verbose >= 1:
                info(f"  Submitting {len(self.years_)} windows to {n_workers} workers...")

            results = Parallel(n_jobs=n_workers)(
                delayed(_bootstrap_window_worker)(
                    year, df_windows[year], self.detector_class, self.detector_kwargs, self.n_iter,
                    self._seed_for(year)
                )
                for year in self.years_
            )

            for year, samples, err in results:
                self._raw_samples_[year] = samples
                win_str = '/'.join(str(y) for y in windows[year])
                if not samples:
                    warn(f"  {year} [window: {win_str}]: no valid thresholds"
                         + (f" - {err}" if err else " (detection found none)"))
                elif self.verbose >= 1:
                    p50 = float(np.percentile(samples, 50))
                    info(f"  {year} [window: {win_str}]  p50={p50:.4f} m/s "
                         f"({len(samples)}/{self.n_iter} ok)")

        # Compute per-year percentiles
        rows = []
        for year in self.years_:
            s = self._raw_samples_[year]
            if s:
                rows.append({f'p{p}': float(np.percentile(s, p)) for p in self.percentiles})
            else:
                rows.append({f'p{p}': np.nan for p in self.percentiles})

        self.annual_stats_ = pd.DataFrame(rows, index=pd.Index(self.years_, name='year'))

        # Pool all samples across years for CUT
        self._all_samples_ = [v for s in self._raw_samples_.values() for v in s]

        if self.verbose >= 1:
            cut = self.get_cut_threshold()
            cut_str = '  '.join(f'{k}={v:.4f}' for k, v in cut.items())
            info(f"CUT (pooled): {cut_str}")

        return self.annual_stats_

    def get_vut_thresholds(self) -> pd.DataFrame:
        """
        Get VUT (variable, per-year) USTAR thresholds.

        Convenience accessor returning ``annual_stats_`` (the table produced by
        :meth:`run`): one row per year, columns ``p{percentile}``. Each year's
        values are the percentiles of that year's 3-year-window bootstrap
        distribution, so the threshold varies year to year.

        .. note::
           Each year is estimated from a **3-year window** (the year plus its two
           neighbors), not the single year alone, so this VUT is smoothed relative
           to the strict ONEFlux single-year VUT — a deliberate trade for a more
           stable per-year threshold. See the class docstring.

        Returns
        -------
        pd.DataFrame
            Per-year percentile thresholds (index: year; columns: p{percentile}).

        Raises
        ------
        RuntimeError
            If run() has not been called yet.
        """
        if self.annual_stats_ is None:
            raise RuntimeError("Call run() before get_vut_thresholds().")
        return self.annual_stats_

    def get_cut_threshold(self) -> Dict[str, float]:
        """
        Get CUT (constant) USTAR threshold pooled across all years.

        Pools all bootstrap samples from all years into a single distribution
        and extracts the requested percentiles. This gives a single conservative
        threshold that is stable across the full measurement record. This matches
        the ONEFlux CUT (constant U\\* threshold) meaning directly.

        Returns
        -------
        dict
            Keys: 'p{percentile}' (e.g., 'p16', 'p50', 'p84').
            Values: USTAR threshold in m/s. p50 is the recommended CUT threshold.

        Raises
        ------
        RuntimeError
            If run() has not been called yet.
        """
        if self._all_samples_ is None:
            raise RuntimeError("Call run() before get_cut_threshold().")

        if not self._all_samples_:
            return {f'p{p}': np.nan for p in self.percentiles}

        return {f'p{p}': float(np.percentile(self._all_samples_, p)) for p in self.percentiles}

    def summary(self) -> str:
        """Return formatted summary of VUT (per-year) and CUT (constant) thresholds."""
        if self.annual_stats_ is None:
            return "No results. Call run() first."

        p_cols = [f'p{p}' for p in self.percentiles]
        col_w = 12

        lines = [
            f"USTAR Bootstrap Threshold Results",
            "=" * 70,
            f"Method     : {self.detector_class.__name__}",
            f"Iterations : {self.n_iter} per window (3-yr sliding window)",
            f"Percentiles: {', '.join(p_cols)}",
            "",
            "VUT thresholds (variable, per-year; m/s)  [p50 = recommended; "
            "3-yr-window smoothed]",
            "-" * 70,
            f"{'Year':<8}{'Window':<16}" + "".join(f"{c:>{col_w}}" for c in p_cols),
        ]

        for year in self.years_:
            row = self.annual_stats_.loc[year]
            n = len(self._raw_samples_.get(year, []))
            window = self._window_years_.get(year, [year])
            win_str = '/'.join(str(y) for y in window)
            vals = "".join(
                f"{row[c]:>{col_w}.4f}" if not np.isnan(row[c]) else f"{'N/A':>{col_w}}"
                for c in p_cols
            )
            lines.append(f"{year:<8}{win_str:<16}{vals}  ({n}/{self.n_iter})")

        cut = self.get_cut_threshold()
        cut_vals = "".join(
            f"{v:>{col_w}.4f}" if not np.isnan(v) else f"{'N/A':>{col_w}}"
            for v in cut.values()
        )
        lines += [
            "",
            "CUT threshold (constant, pooled across all years)",
            "-" * 70,
            f"{'CUT':<8}{'all years':<16}{cut_vals}",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"UstarBootstrapThresholds("
            f"detector={self.detector_class.__name__}, "
            f"years={self.years_}, "
            f"n_iter={self.n_iter}, "
            f"percentiles={self.percentiles}, "
            f"n_jobs={self.n_jobs})"
        )
