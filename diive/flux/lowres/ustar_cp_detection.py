"""
USTAR CHANGE-POINT DETECTION: FRICTION VELOCITY THRESHOLD
=========================================================

Detect the friction velocity (u*) threshold with the ONEFlux change-point method (Barr et al., 2013).

This is a port of the MATLAB code that ONEFlux 1.3.7 runs as ``ustar_cp``
(``oneflux_steps/ustar_cp/``: ``launch.m``, ``cpdBootstrapUStarTh4Season20100901.m``,
``cpdEvaluateUStarTh4Season20100901.m``, ``cpdFindChangePoint20100901.m``,
``cpdFmax2pCp2.m``, ``fcBin.m``, ``cpdAssignUStarTh20100901.m``). It ports the
code, not the paper. The two differ: the code uses 4 seasons of equal record
counts (not moving windows), 4 to 8 temperature strata per season, 100 bootstrap
draws, a 10 % minimum success rate and only the 2-parameter model.

Part of the diive library: https://github.com/holukas/diive
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from diive.core.utils.console import detail, info, warn
from diive.flux.lowres._ustar_cp_core import (
    N_SEASONS,
    N_STRATA_MAX,
    STATS_FIELDS,
    assign_annual,
    december_flags,
    evaluate_draw,
    n_per_bin,
    n_per_day_from_t,
)
from diive.flux.lowres._ustar_cp_numerics import (
    matlab_default_stream,
    matlab_randi_positions,
)

RngLike = Union[None, str, int, np.random.Generator, np.random.RandomState]


class UstarChangePointDetection:
    """Detect the u* threshold with the ONEFlux change-point method (Barr et al., 2013).

    Port of ONEFlux 1.3.7 ``ustar_cp`` for one calendar year. The steps:

    1. Night records are those with SW_IN < 5 W m-2 (``launch.m:290``). Records
       with u* outside 0..4 m s-1 are rejected for the record check and outside
       0..3 m s-1 for the analysis.
    2. Each bootstrap draw resamples all records of the year with replacement and
       keeps them in time order (``sort(randi(nt, nt, 1))``).
    3. December is moved to the start of the year. The valid night records are
       split into 4 seasons of equal size, each season into 4 to 8 temperature
       strata (one per 250 records at 30 min) at the temperature percentiles.
    4. In each stratum u* is binned into bins of 5 records (3 for hourly data)
       and a 2-parameter change-point model (a sloped line that turns flat at the
       change point) is fitted at every bin. The best fit is tested against the
       Fmax table of Barr et al. (p <= 0.05).
    5. Over all draws, only significant change points of the dominant mode (D:
       NEE rises with u* below the change point, or E) are kept, outliers are
       removed, and the annual threshold of a draw is the mean of its kept
       change points.

    The year fails (no threshold) when less than 10 % of the attempted strata
    give a kept change point, or when fewer than 4 x n_iter remain after the
    outlier screen. ONEFlux then writes no output file. ``failure_`` holds the
    ONEFlux message.

    Parameters
    ----------
    df : pd.DataFrame
        One calendar year on its full time grid (17520 or 17568 half-hourly rows,
        8760 or 8784 hourly rows), missing values NaN (-9999 is read as missing
        too). The rows are those of the ONEFlux qc_auto input file. The index is
        not used for the analysis: like ``launch.m:284-287``, the day count t is
        built from the row number alone (t = 1 + k/nrPerDay for row k = 1, 2,
        ...), so any timestamp convention (diive's TIMESTAMP_MIDDLE, the
        TIMESTAMP_END of ONEFlux files) gives the same result as long as row k is
        the k-th averaging period of the year.
    nee_col, ta_col, ustar_col, swin_col : str
        Column names of NEE, air temperature, u* and incoming shortwave radiation.
    randi_mapping : str, default 'ceil'
        How the MATLAB random stream is turned into record positions, see
        ``matlab_randi_positions``.
    verbose : int, default 0
        0 silent, 1 progress, 2 detail.

    Attributes
    ----------
    results_ : pd.DataFrame
        ``detect()`` result: one row per evaluated (season, stratum) with the
        Stats2 fields and ``Cp2`` (the change point, NaN when not significant).
    annual_threshold_ : float
        ``detect()`` annual value, NaN on failure.
    detect_failure_ : str
        Failure message of ``detect()``, '' on success.
    bootstrap_cp2_ : np.ndarray
        (n_iter, 4, 8) change points of the last bootstrap.
    bootstrap_stats_ : dict
        Stats2 fields of the last bootstrap, each (n_iter, 4, 8).
    selected_ : np.ndarray
        (n_iter, 4, 8) bool, change points that entered the annual means.
    n_selected_ : np.ndarray
        Number of kept change points per draw (``nA`` in ONEFlux).
    failure_ : str
        cFailure of the last bootstrap, '' on success. ONEFlux writes the
        threshold file only when it is empty.
    mode_ : str
        Dominant change-point mode of the last bootstrap, 'D' or 'E'.
    frac_sig_ : float
        FracSig of the last bootstrap: share of attempted strata with a
        significant change point.
    frac_mode_d_ : float
        FracModeD: share of the significant change points in mode D.
    frac_select_ : float
        FracSelect: share of attempted strata with a kept change point.
    bootstrap_ran_ : bool
        False when the year had too few valid night records and ONEFlux skips
        the bootstrap (no random numbers are drawn then).

    Notes
    -----
    Numerics follow MATLAB R2018a (MATLAB Runtime v9.4, which ONEFlux uses):
    ``prctile``, the colon operator, ``round`` (halves away from zero) and
    ``interp1(..., 'pchip')`` are reproduced in ``_ustar_cp_numerics``. Bin
    means and annual means are summed in element order, as MATLAB does. The
    least squares fits use the centred closed form instead of MATLAB's QR, so
    results agree to rounding, not bitwise. Checked against the compiled
    ONEFlux ``ustar_cp`` (MATLAB Runtime R2018a): all 1000 annual values of 10
    half-hourly site-years agree to the 8 significant digits ONEFlux writes,
    and 6 further site-years fail with the same message (harness in
    ``diive-data/references/ustar_parity/cp/``). Hourly data match too: 9
    site-years, all 900 values (see ``december_flags``).

    The 3-parameter diagnostic model of ``cpdFindChangePoint20100901`` is not
    ported. ONEFlux computes it but discards it (``launch.m:316-323``). The
    seasonal sine fit and the per-window means of ``cpdAssignUStarTh20100901``
    are not ported either, as they do not reach the ONEFlux output.

    Example
    -------
    ::

        det = UstarChangePointDetection(df, nee_col='NEE', ta_col='TA',
                                        ustar_col='USTAR', swin_col='SW_IN')
        samples = det.bootstrap_annual_samples()  # 100 values, ONEFlux stream
        if not det.failure_:
            print(np.nanmedian(samples))
    """

    NIGHT_THRESHOLD = 5.0  # launch.m:290, fNight = Rg < 5
    USTAR_MAX_BOOTSTRAP = 4.0  # Bootstrap.m:75
    N_BOOT = 100  # launch.m:309

    def __init__(
        self,
        df: pd.DataFrame,
        nee_col: str,
        ta_col: str,
        ustar_col: str,
        swin_col: str,
        randi_mapping: str = 'ceil',
        verbose: int = 0,
    ):
        """Set up change-point detection for one calendar year. See the class docstring."""
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty")
        for c in (nee_col, ta_col, ustar_col, swin_col):
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found in DataFrame")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        nt = len(df)
        # launch.m:284-285
        nr_per_day = nt % 365
        if nr_per_day == 0:
            nr_per_day = nt % 364
        if nr_per_day not in (24, 48) or nt not in (365 * nr_per_day, 366 * nr_per_day):
            raise ValueError(
                f"Expected one full calendar year of half-hourly or hourly records "
                f"(17520/17568 or 8760/8784 rows, gaps as NaN), got {nt} rows")

        self.df = df
        self.nee_col = nee_col
        self.ta_col = ta_col
        self.ustar_col = ustar_col
        self.swin_col = swin_col
        self.randi_mapping = randi_mapping
        self.verbose = verbose
        self.nr_per_day = nr_per_day

        def col(name):
            a = df[name].to_numpy(dtype=np.float64, copy=True)
            a[a == -9999] = np.nan  # launch.m:249-253
            return a

        self._nee = col(nee_col)
        self._ta = col(ta_col)
        ustar = col(ustar_col)
        swin = col(swin_col)
        self._ustar_raw = ustar
        with np.errstate(invalid='ignore'):
            self._night = swin < self.NIGHT_THRESHOLD  # NaN radiation is not night
            self._ustar = np.where((ustar < 0) | (ustar > self.USTAR_MAX_BOOTSTRAP), np.nan, ustar)
        # launch.m:286-287: t by repeated addition (cumsum adds in the same order)
        step = 1.0 / nr_per_day
        self._t = np.cumsum(np.r_[1.0 + step, np.full(nt - 1, step)])
        self._dec = december_flags(self._t)

        self.results_ = pd.DataFrame()
        self.annual_threshold_ = np.nan
        self.detect_failure_ = ''
        self.bootstrap_cp2_ = None
        self.bootstrap_stats_ = None
        self.selected_ = None
        self.n_selected_ = None
        self.failure_ = ''
        self.mode_ = ''
        self.frac_sig_ = np.nan
        self.frac_mode_d_ = np.nan
        self.frac_select_ = np.nan
        self.bootstrap_ran_ = False

    # ------------------------------------------------------------------ checks

    def _empty_input_reason(self) -> str:
        """launch.m:257-264 skips a year whose NEE or u* is all missing."""
        if np.isnan(self._nee).all():
            return 'NEE is empty!'
        if np.isnan(self._ustar_raw).all():
            return 'uStar is empty!'
        return ''

    def _enough_night_records(self) -> bool:
        """Bootstrap.m:89-118: valid night records >= 4 seasons x 4 strata x 50 bins x nPerBin."""
        npb = n_per_bin(n_per_day_from_t(self._t))
        n_needed = N_SEASONS * 4 * 50 * npb
        valid = self._night & ~np.isnan(self._nee + self._ustar + self._ta)
        return int(valid.sum()) >= n_needed

    def _evaluate(self, pos: np.ndarray):
        """Evaluate the records at sorted positions ``pos`` (one draw)."""
        return evaluate_draw(self._t[pos], self._nee[pos], self._ustar[pos],
                             self._ta[pos], self._night[pos], self._dec[pos])

    # ------------------------------------------------------------------ public

    def detect(self) -> pd.DataFrame:
        """Evaluate the original (not resampled) year, for information.

        ONEFlux has no such run: it evaluates only the 100 bootstrap draws.
        This applies the same evaluation to the records as they are, then the
        selection of ``cpdAssignUStarTh20100901`` to this single draw (with the
        rules of its bootstrap branch, so at least 4 change points must
        remain). Use ``bootstrap_annual_samples()`` for the ONEFlux result.

        Returns
        -------
        pd.DataFrame
            One row per evaluated (season, stratum): the Stats2 fields, ``Cp2``
            and ``selected``. The annual value is in ``annual_threshold_``, the
            failure message in ``detect_failure_``.
        """
        pos = np.arange(len(self.df))
        cp2, stats = self._evaluate(pos)
        res = assign_annual(stats['Cp'][None], stats['b1'][None], stats['cib1'][None],
                            stats['p'][None], stats['mt'][None])
        rows = []
        for s in range(N_SEASONS):
            for k in range(N_STRATA_MAX):
                if np.isnan(stats['mt'][s, k]):
                    continue
                row = {'season': s + 1, 'stratum': k + 1, 'Cp2': cp2[s, k]}
                row.update({f: stats[f][s, k] for f in STATS_FIELDS})
                row['selected'] = bool(res['selected'][0, s, k])
                rows.append(row)
        cols = ['season', 'stratum', 'Cp2', *STATS_FIELDS, 'selected']
        self.results_ = pd.DataFrame(rows, columns=cols).set_index(['season', 'stratum'])
        self.annual_threshold_ = float(res['cp_annual'][0])
        self.detect_failure_ = res['failure']
        if self.verbose >= 1:
            if res['failure']:
                info(f"Change-point detection (no resampling): {res['failure']}", verbose=self.verbose)
            else:
                info(f"Change-point detection (no resampling): annual u* threshold "
                     f"{self.annual_threshold_:.4f} m/s, mode {res['mode']}", verbose=self.verbose)
        return self.results_

    def draw_positions(self, n_iter: int = N_BOOT, rng: RngLike = None) -> np.ndarray:
        """Sorted 0-based record positions of ``n_iter`` bootstrap draws, (n_iter, nt).

        ``rng``:

        - None or ``'matlab'``: a fresh MATLAB default stream (mt19937ar, seed
          5489), as at the start of a ONEFlux ``ustar_cp`` process.
        - ``np.random.RandomState``: used as a MATLAB stream that continues. One
          ONEFlux ``ustar_cp`` process handles all years of its input folder in
          one session, so the stream carries over from year to year. Pass the
          same RandomState to the years in file order to reproduce that.
        - int or ``np.random.Generator``: diive's own draws (numpy Generator).
        """
        nt = len(self.df)
        if rng is None or (isinstance(rng, str) and rng == 'matlab'):
            rng = matlab_default_stream()
        if isinstance(rng, np.random.RandomState):
            pos = np.empty((n_iter, nt), dtype=np.int64)
            for i in range(n_iter):
                pos[i] = matlab_randi_positions(rng, nt, self.randi_mapping)
        elif isinstance(rng, (int, np.integer, np.random.Generator)):
            gen = np.random.default_rng(rng)
            pos = gen.integers(0, nt, size=(n_iter, nt))
        else:
            raise ValueError(f"Unsupported rng: {rng!r}")
        pos.sort(axis=1)
        return pos

    def bootstrap_annual_samples(self, n_iter: int = N_BOOT, rng: RngLike = None,
                                 indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Annual change-point thresholds of ``n_iter`` bootstrap draws (the ONEFlux output).

        Parameters
        ----------
        n_iter : int, default 100
            Number of draws (``nBoot``). The minimum number of kept change
            points scales with it (4 x n_iter).
        rng : None, 'matlab', RandomState, int or Generator
            Random source, see ``draw_positions``. The default reproduces the
            ONEFlux stream of a process that handles this one year.
        indices : np.ndarray, optional
            (n_iter, nt) record positions (0-based) to use instead of random
            draws. Each row is sorted before use, as ONEFlux sorts its draws.

        Returns
        -------
        np.ndarray
            One annual threshold per draw, full double precision. NaN for draws
            without a kept change point, and all NaN when the year fails
            (``failure_`` is not empty). ONEFlux writes these values with 8
            significant digits (``dlmwrite(..., 'precision', 8)``).
        """
        nt = len(self.df)
        if indices is not None:
            indices = np.sort(np.asarray(indices, dtype=np.int64), axis=1)
            if indices.ndim != 2 or indices.shape[1] != nt:
                raise ValueError(f"indices must have shape (n_iter, {nt})")
            if indices.min() < 0 or indices.max() >= nt:
                raise ValueError("indices out of range")
            n_iter = indices.shape[0]

        cp2 = np.full((n_iter, N_SEASONS, N_STRATA_MAX), np.nan)
        stats = {f: np.full((n_iter, N_SEASONS, N_STRATA_MAX), np.nan) for f in STATS_FIELDS}

        empty = self._empty_input_reason()
        self.bootstrap_ran_ = not empty and self._enough_night_records()
        if self.bootstrap_ran_:
            # Draw only when ONEFlux would: a skipped year consumes no random numbers
            pos = indices if indices is not None else self.draw_positions(n_iter, rng)
            for i in range(n_iter):
                if self.verbose >= 2 and i % 10 == 0:
                    detail(f"  Bootstrap {i + 1}/{n_iter}", verbose=self.verbose)
                c, s = self._evaluate(pos[i])
                cp2[i] = c
                for f in STATS_FIELDS:
                    stats[f][i] = s[f]
        elif self.verbose >= 1 and not empty:
            warn("Too few valid night records for change-point detection, "
                 "bootstrap skipped as in ONEFlux", verbose=self.verbose)

        self.bootstrap_cp2_ = cp2
        self.bootstrap_stats_ = stats
        if empty:
            res = dict(cp_annual=np.full(n_iter, np.nan), n_selected=np.full(n_iter, np.nan),
                       selected=np.zeros(cp2.shape, dtype=bool), mode='', failure=empty,
                       frac_sig=np.nan, frac_mode_d=np.nan, frac_select=np.nan)
        else:
            res = assign_annual(stats['Cp'], stats['b1'], stats['cib1'], stats['p'], stats['mt'])
        self.selected_ = res['selected']
        self.n_selected_ = res['n_selected']
        self.failure_ = res['failure']
        self.mode_ = res['mode']
        self.frac_sig_ = res['frac_sig']
        self.frac_mode_d_ = res['frac_mode_d']
        self.frac_select_ = res['frac_select']

        if self.verbose >= 1:
            if self.failure_:
                info(f"Change-point bootstrap: {self.failure_}", verbose=self.verbose)
            else:
                info(f"Change-point bootstrap: median annual u* threshold "
                     f"{np.nanmedian(res['cp_annual']):.4f} m/s over {n_iter} draws, "
                     f"mode {self.mode_}", verbose=self.verbose)
        return res['cp_annual']

    def __repr__(self) -> str:
        return (f"UstarChangePointDetection(records={len(self.df)}, "
                f"nr_per_day={self.nr_per_day}, failure='{self.failure_}')")
