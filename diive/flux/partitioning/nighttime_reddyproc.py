"""
NIGHTTIME PARTITIONING REDDYPROC: NEE -> GPP + RECO (Reichstein et al. 2005)
============================================================================

Faithful, vectorized Python port of the REddyProc nighttime partitioning
(``sEddyProc_sMRFluxPartition`` in REddyProc's ``EddyPartitioning.R``), a second
reference implementation of the same Reichstein et al. (2005) nighttime ("MR")
method. It is intentionally a *separate* port from the ONEFlux variant
(:mod:`diive.flux.partitioning.nighttime_oneflux`): the two implementations of
the same paper differ in window geometry, day/night split, the E0 fitting
routine, and the working temperature units, so they do not produce identical
numbers.

The nighttime method estimates ecosystem respiration (RECO) from the
temperature response of *nighttime* NEE, then derives gross primary production
(GPP) as ``GPP = RECO - NEE``. At night there is no photosynthesis, so measured
NEE equals respiration; fitting the Lloyd & Taylor (1994) function to nighttime
fluxes and extrapolating to daytime temperatures recovers daytime respiration.

Algorithm (whole record, as in REddyProc - a single E0 for the entire series):

1. Flag nighttime records with a combined radiation threshold: incoming
   shortwave ``Rg <= 10`` W m-2 AND potential radiation ``<= 0`` (sun below the
   horizon). Potential radiation uses exact solar time (latitude, longitude,
   UTC offset; the ``solartime`` geometry REddyProc relies on).
2. Estimate one temperature sensitivity E0 from short overlapping windows
   (centered 15-day windows, 5-day steps): per window fit Lloyd-Taylor in
   Kelvin, trim the 5%/95% signed-residual tails, refit, and keep E0 only if
   its +/-1 standard-deviation interval lies inside [30, 350] K. Average the
   three lowest-SD estimates (``fRegrE0fromShortTerm``).
3. With E0 fixed, re-estimate the reference respiration Rref in centered 7-day
   windows (4-day steps) as the through-origin slope of nighttime NEE on the
   Lloyd-Taylor factor, then interpolate Rref to every record
   (``sRegrRref``).
4. RECO = LloydTaylor(Tair_f, Rref, E0); GPP = RECO - NEE_f.

If fewer than three short-term windows yield a well-constrained E0, REddyProc
aborts the whole partitioning (return code -111); this port then leaves every
record unpartitioned.

REddyProc has no outlier-robust RECO variant, so this port emits no ``*_ROB``
columns (unlike the ONEFlux variant).

Reference:
    Reichstein, M. et al. (2005). On the separation of net ecosystem exchange
    into assimilation and ecosystem respiration: review and improved algorithm.
    Global Change Biology, 11(9), 1424-1439.
    https://doi.org/10.1111/j.1365-2486.2005.001002.x

    Wutzler, T. et al. (2018). Basic and extensible post-processing of eddy
    covariance flux data with REddyProc. Biogeosciences, 15, 5015-5030.
    https://doi.org/10.5194/bg-15-5015-2018

    Lloyd, J. & Taylor, J. A. (1994). On the temperature dependence of soil
    respiration. Functional Ecology, 8(3), 315-323.
    https://doi.org/10.2307/2389824

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.optimize import leastsq

from diive.core.utils.console import info, warn, success

# Lloyd & Taylor reference/regression temperatures in KELVIN, as used by
# REddyProc (fLloydTaylor): TRef = 273.15 + 15 degC, T0 = 227.13 degK.
TREF_K = 273.15 + 15.0
T0_K = 227.13

# Windowing / fitting constants (REddyProc defaults).
E0_WINDOW_HALF = 7     # half-width of the E0 window (days) -> 15-day window
E0_STEP = 5            # E0 window step (days)
E0_MIN_ENTRIES = 6     # need > 6 nighttime records in a window to fit
E0_TEMP_RANGE = 5.0    # minimum temperature range (K) in a window
E0_TRIM_PERC = 5.0     # residual trim percentile per tail (%)
E0_NUM_BEST = 3        # number of lowest-SD E0 estimates to average
E0_MIN = 30.0          # E0 validity lower bound (K), Tair limits
E0_MAX = 350.0         # E0 validity upper bound (K), Tair limits

RREF_WINDOW_HALF = 3   # half-width of the Rref window (days) -> 7-day window
RREF_STEP = 4          # Rref window step (days)
RREF_MIN_ENTRIES = 2   # need > 2 nighttime records in a window to fit

DAY_MAX_SW_IN = 10.0   # Rg <= this (W m-2) is a necessary night condition
SOLAR_CONST = 1366.1   # total solar irradiance (W m-2), fCalcExtRadiation


def lloyd_taylor_kelvin(ta_k: np.ndarray, rref: float | np.ndarray,
                        e0: float | np.ndarray,
                        tref_k: float = TREF_K, t0_k: float = T0_K) -> np.ndarray:
    """Lloyd & Taylor (1994) respiration, REddyProc's Kelvin parameterization.

    Numerically identical to the degC form used by the ONEFlux variant
    (``(TRef - T0)`` and ``(Ta - T0)`` are the same in K and degC), but written
    in Kelvin to mirror REddyProc's ``fLloydTaylor`` exactly.

    Args:
        ta_k: Air (or soil) temperature in Kelvin.
        rref: Reference respiration at ``tref_k`` (umol m-2 s-1).
        e0: Temperature sensitivity in Kelvin.
        tref_k: Reference temperature in Kelvin.
        t0_k: Regression temperature in Kelvin (227.13).

    Returns:
        Respiration in the same units as ``rref``.
    """
    return rref * np.exp(e0 * ((1.0 / (tref_k - t0_k)) - (1.0 / (ta_k - t0_k))))


def potential_radiation(doy: np.ndarray, hour: np.ndarray, lat: float,
                        lon: float, utc_offset: float) -> np.ndarray:
    """Potential (top-of-canopy clear-sky) radiation in W m-2.

    Faithful port of REddyProc ``fCalcPotRadiation`` with ``useSolartime=TRUE``,
    which delegates the solar geometry to ``solartime::computeSunPositionDoyHour``
    (Cescatti). Used only for the day/night split, never as a flux.

    Args:
        doy: Day of year (1-366).
        hour: Decimal local-winter-time hour (e.g. 13.5 for 13:30).
        lat: Site latitude (decimal degrees).
        lon: Site longitude (decimal degrees).
        utc_offset: Time zone offset from UTC in hours (e.g. +1 for CET).

    Returns:
        Potential radiation (W m-2), zero where the sun is at/below the horizon.
    """
    frac_year = 2.0 * np.pi * (doy - 1.0) / 365.24

    # Equation of time + longitude correction -> local-to-solar time difference.
    eq_time = (0.0072 * np.cos(frac_year) - 0.0528 * np.cos(2 * frac_year)
               - 0.0012 * np.cos(3 * frac_year) - 0.1229 * np.sin(frac_year)
               - 0.1565 * np.sin(2 * frac_year) - 0.0041 * np.sin(3 * frac_year))
    loc_time = lon / 15.0 - utc_offset
    solar_time_hour = hour + loc_time + eq_time

    sol_time_rad = (solar_time_hour - 12.0) * np.pi / 12.0
    sol_time_rad = np.where(sol_time_rad < -np.pi, sol_time_rad + 2 * np.pi,
                            sol_time_rad)

    sol_decl = ((0.33281 - 22.984 * np.cos(frac_year) - 0.3499 * np.cos(2 * frac_year)
                 - 0.1398 * np.cos(3 * frac_year) + 3.7872 * np.sin(frac_year)
                 + 0.03205 * np.sin(2 * frac_year) + 0.07187 * np.sin(3 * frac_year))
                / 180.0 * np.pi)

    lat_rad = lat / 180.0 * np.pi
    sol_elev = np.arcsin(np.sin(sol_decl) * np.sin(lat_rad)
                         + np.cos(sol_decl) * np.cos(lat_rad) * np.cos(sol_time_rad))

    # Extraterrestrial radiation with the eccentricity correction (Lanini 2010).
    ext_rad = SOLAR_CONST * (1.00011 + 0.034221 * np.cos(frac_year)
                             + 0.00128 * np.sin(frac_year)
                             + 0.000719 * np.cos(2 * frac_year)
                             + 0.000077 * np.sin(2 * frac_year))

    return np.where(sol_elev <= 0.0, 0.0, ext_rad * np.sin(sol_elev))


def _leastsq_with_sd(resid_func, x0, entries: int):
    """``scipy.optimize.leastsq`` wrapper returning params and their std devs.

    Returns ``(params, std_devs, residuals)``; ``std_devs`` is NaN when the
    covariance is unavailable (mirrors how the standard error is read off an
    ``nls`` summary in REddyProc).
    """
    pars, cov_x, infodict, _msg, _ier = leastsq(
        resid_func, x0, full_output=True, maxfev=1000 * (entries + 1))
    npar = len(x0)
    residuals = infodict['fvec']
    if entries > npar and cov_x is not None:
        s_squared = (residuals ** 2).sum() / (entries - npar)
        std_devs = np.sqrt(np.abs(np.diag(cov_x * s_squared)))
    else:
        std_devs = np.full(npar, np.nan)
    return np.asarray(pars, dtype=float), std_devs, residuals


def _fit_e0_single(nee_night: np.ndarray, ta_k: np.ndarray, tref_k: float):
    """Port of ``fOptimSingleE0`` (default algorithm): fit, trim, refit.

    Fits Lloyd-Taylor (Rref, E0) to nighttime NEE vs. temperature (Kelvin),
    trims the 5%/95% *signed*-residual tails, and refits on the kept subset.

    Returns ``(e0_trim, e0_trim_sd)`` (the trimmed-fit E0 and its standard
    deviation) or ``None`` if either fit cannot be formed.
    """
    # Temperature-dependent exponent base, hoisted out of the optimizer loop:
    # the Lloyd-Taylor model is ``rref * exp(e0 * b)`` with ``b`` constant across
    # iterations, so computing it once (instead of inside lloyd_taylor_kelvin on
    # every residual call) is bit-identical but markedly faster.
    b = (1.0 / (tref_k - T0_K)) - (1.0 / (ta_k - T0_K))

    def residuals_full(par):
        rref, e0 = par
        return nee_night - rref * np.exp(e0 * b)

    try:
        _pars, _sd, res = _leastsq_with_sd(residuals_full, [2.0, 200.0],
                                           entries=nee_night.size)
    except Exception:
        return None

    # Trim points whose (data - model) residual is outside the [5%, 95%] range.
    lo, hi = np.quantile(res, [E0_TRIM_PERC / 100.0, 1.0 - E0_TRIM_PERC / 100.0])
    keep = (res >= lo) & (res <= hi)
    if keep.sum() < 3:
        return None

    nee_k = nee_night[keep]
    b_k = b[keep]

    def residuals_trim(par):
        rref, e0 = par
        return nee_k - rref * np.exp(e0 * b_k)

    try:
        # REddyProc's default-algorithm path restarts the trimmed fit at (2, 200).
        pars_t, sd_t, _res_t = _leastsq_with_sd(residuals_trim, [2.0, 200.0],
                                                entries=int(keep.sum()))
    except Exception:
        return None

    return float(pars_t[1]), float(sd_t[1])


def _window_slices(day_counter: np.ndarray, half: int, step: int):
    """``(lo, hi)`` array-slice bounds for each centered window.

    ``day_counter`` is monotonic non-decreasing (``(1:DIMS) %/% DTS``), so every
    window's records form a contiguous slice located by binary search. This is
    numerically identical to REddyProc rebuilding a full-length boolean mask each
    iteration, but avoids allocating one mask per window over the whole record.
    """
    last_day = int(day_counter.max())
    mids = np.arange(half + 1, last_day + 1, step)
    los = np.searchsorted(day_counter, mids - half, side='left')
    his = np.searchsorted(day_counter, mids + half, side='right')
    return los, his


def _regr_e0_from_short_term(nee_night: np.ndarray, ta: np.ndarray,
                             day_counter: np.ndarray, tref_k: float) -> float:
    """Port of ``fRegrE0fromShortTerm``: one representative E0 for the record.

    Slides a centered 15-day window in 5-day steps, fits E0 per window, keeps
    only well-constrained estimates, and averages the three with the smallest
    standard deviation. Returns NaN when fewer than three are valid (REddyProc
    aborts in that case).
    """
    e0_trim = []
    e0_trim_sd = []

    valid_all = ~np.isnan(nee_night) & ~np.isnan(ta)
    ta_k_all = ta + 273.15
    los, his = _window_slices(day_counter, E0_WINDOW_HALF, E0_STEP)
    for lo, hi in zip(los, his):
        m = valid_all[lo:hi]
        if int(m.sum()) <= E0_MIN_ENTRIES:
            continue
        ta_k = ta_k_all[lo:hi][m]
        if (np.max(ta_k) - np.min(ta_k)) < E0_TEMP_RANGE:
            continue
        fit = _fit_e0_single(nee_night[lo:hi][m], ta_k, tref_k)
        if fit is None:
            continue
        e0_trim.append(fit[0])
        e0_trim_sd.append(fit[1])

    if not e0_trim:
        return np.nan

    e0_trim = np.asarray(e0_trim)
    e0_trim_sd = np.asarray(e0_trim_sd)

    # Validity: the +/-1 SD interval must lie inside [E0_MIN, E0_MAX].
    with np.errstate(invalid='ignore'):
        valid = ((e0_trim - e0_trim_sd > E0_MIN)
                 & (e0_trim + e0_trim_sd < E0_MAX))
    if valid.sum() < E0_NUM_BEST:
        return np.nan

    # Sort valid estimates by SD ascending, average the best NUM_BEST.
    order = np.argsort(e0_trim_sd[valid])
    best = e0_trim[valid][order[:E0_NUM_BEST]]
    return round(float(np.mean(best)), 2)


def _regr_rref(nee_night: np.ndarray, ta: np.ndarray, day_counter: np.ndarray,
               e0: float, tref_k: float) -> np.ndarray:
    """Port of ``sRegrRref``: time-varying Rref with E0 held fixed.

    Slides a centered 7-day window in 4-day steps. Per window Rref is the
    through-origin OLS slope of nighttime NEE on the Lloyd-Taylor factor, placed
    at the mean record index of the window, then linearly interpolated to every
    record (constant beyond the first/last estimate). Negative slopes are
    dropped before interpolation.
    """
    n = nee_night.size
    rref_at = np.full(n, np.nan)
    record_idx = np.arange(1, n + 1)  # 1-based, like REddyProc's which(Subset.b)

    valid_all = ~np.isnan(nee_night) & ~np.isnan(ta)
    ta_k_all = ta + 273.15
    los, his = _window_slices(day_counter, RREF_WINDOW_HALF, RREF_STEP)
    for lo, hi in zip(los, his):
        m = valid_all[lo:hi]
        if int(m.sum()) <= RREF_MIN_ENTRIES:
            continue
        # Mean 1-based record index of the window, == round(mean(which(Subset.b))).
        mean_h = int(round(float((lo + np.nonzero(m)[0] + 1).mean())))
        factor = lloyd_taylor_kelvin(ta_k_all[lo:hi][m], 1.0, e0, tref_k)
        nee_sub = nee_night[lo:hi][m]
        denom = float((factor ** 2).sum())
        if denom <= 0:
            continue
        rref = float((factor * nee_sub).sum() / denom)
        if rref < 0:
            continue  # R_ref_ok: negative reference respiration -> dropped
        rref_at[mean_h - 1] = rref  # back to 0-based

    valid = ~np.isnan(rref_at)
    count = int(valid.sum())
    if count == 0:
        return rref_at
    if count == 1:
        return np.full(n, rref_at[valid][0])
    # numpy.interp clamps to the first/last value outside the sampled range,
    # i.e. constant ends, matching REddyProc's fInterpolateGaps.
    return np.interp(record_idx, record_idx[valid], rref_at[valid])


def _partition_record(nee: np.ndarray, ta: np.ndarray, sw_in: np.ndarray,
                      nee_f: np.ndarray, ta_f: np.ndarray, doy: np.ndarray,
                      hour: np.ndarray, lat: float, lon: float,
                      utc_offset: float, dts: int, verbose: int = 1) -> dict:
    """Run the REddyProc nighttime partitioning over the whole record."""
    n = nee.size
    out = {
        'NEE_NIGHT_RP': np.full(n, np.nan),
        'RECO_NT_RP': np.full(n, np.nan),
        'GPP_NT_RP': np.full(n, np.nan),
        'RREF_NT_RP': np.full(n, np.nan),
        'E0_NT_RP': np.full(n, np.nan),
    }

    # --- Day/night flag: Rg <= 10 AND potential radiation <= 0 ---
    potrad = potential_radiation(doy, hour, lat, lon, utc_offset)
    with np.errstate(invalid='ignore'):
        night_mask = (sw_in <= DAY_MAX_SW_IN) & (potrad <= 0.0)
    nee_night = np.where(night_mask & ~np.isnan(nee), nee, np.nan)
    out['NEE_NIGHT_RP'] = nee_night

    # Record-based day index, REddyProc's DayCounter = (1:DIMS) %/% DTS.
    day_counter = np.arange(1, n + 1) // dts

    # --- E0 from short-term windows (one value for the whole record) ---
    e0 = _regr_e0_from_short_term(nee_night, ta, day_counter, TREF_K)
    if not np.isfinite(e0):
        warn("Nighttime partitioning (ReddyProc): fewer than "
             f"{E0_NUM_BEST} well-constrained short-term E0 estimates; "
             "record left unpartitioned (REddyProc abort, code -111).",
             verbose=verbose)
        return out
    out['E0_NT_RP'][:] = e0

    # --- Rref with E0 fixed, then RECO and GPP ---
    rref = _regr_rref(nee_night, ta, day_counter, e0, TREF_K)
    out['RREF_NT_RP'] = rref

    reco = lloyd_taylor_kelvin(ta_f + 273.15, rref, e0, TREF_K)
    out['RECO_NT_RP'] = reco
    out['GPP_NT_RP'] = reco - nee_f
    return out


def _infer_dts(index: pd.DatetimeIndex) -> int:
    """Records per day from the dominant timestamp spacing."""
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("Cannot infer record frequency from a single timestamp.")
    step_seconds = float(diffs.dt.total_seconds().median())
    if step_seconds <= 0:
        raise ValueError("Non-positive timestamp spacing.")
    return int(round(86400.0 / step_seconds))


class NighttimePartitioningReddyProc:
    """Partition NEE into GPP and RECO with the nighttime method (REddyProc).

    Faithful, vectorized port of REddyProc's ``sMRFluxPartition`` (Reichstein
    et al. 2005). Unlike the ONEFlux variant, the whole record is partitioned at
    once with a single temperature sensitivity E0, matching REddyProc.

    Example: ``examples/flux/partitioning/partitioning_nighttime_reddyproc.py``

    Example:
        >>> part = NighttimePartitioningReddyProc(
        ...     nee=df['NEE_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
        ...     nee_f=df['NEE_f'], ta_f=df['Tair_f'],
        ...     lat=46.815, lon=9.855, utc_offset=1)
        >>> part.run()
        >>> results = part.results   # DataFrame with RECO_NT_RP, GPP_NT_RP, ...
    """

    def __init__(self,
                 nee: Series,
                 ta: Series,
                 sw_in: Series,
                 nee_f: Series,
                 ta_f: Series,
                 lat: float,
                 lon: float,
                 utc_offset: float,
                 verbose: int = 1):
        """
        Args:
            nee: Measured net ecosystem exchange (umol m-2 s-1). Gaps (NaN) are
                the records that were not measured / did not pass QC.
            ta: Measured air temperature (degC), gaps as NaN.
            sw_in: Incoming shortwave radiation (W m-2), used for the day/night
                split. Gaps as NaN.
            nee_f: Gap-filled NEE (umol m-2 s-1) - used for the GPP residual.
            ta_f: Gap-filled air temperature (degC) - used to compute RECO at
                every record.
            lat: Site latitude in decimal degrees.
            lon: Site longitude in decimal degrees (REddyProc needs longitude
                and the UTC offset for the solar-time day/night split).
            utc_offset: Time zone offset from UTC in hours (e.g. +1 for CET).
            verbose: Console verbosity level (0 silent, 1 progress).
        """
        self._inputs = self._validate(nee, ta, sw_in, nee_f, ta_f)
        self.lat = float(lat)
        self.lon = float(lon)
        self.utc_offset = float(utc_offset)
        self.verbose = verbose
        self._results: DataFrame | None = None

    @staticmethod
    def _validate(nee, ta, sw_in, nee_f, ta_f) -> DataFrame:
        series = {'nee': nee, 'ta': ta, 'sw_in': sw_in, 'nee_f': nee_f, 'ta_f': ta_f}
        for name, s in series.items():
            if not isinstance(s, Series):
                raise TypeError(f"'{name}' must be a pandas Series, got {type(s)}.")
            if not isinstance(s.index, pd.DatetimeIndex):
                raise TypeError(f"'{name}' must have a DatetimeIndex.")
        df = pd.DataFrame({k: v.astype(float) for k, v in series.items()})
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df

    def run(self) -> "NighttimePartitioningReddyProc":
        """Run the partitioning and populate :attr:`results`."""
        df = self._inputs
        index = df.index
        doy = index.dayofyear.to_numpy()
        hour = (index.hour + index.minute / 60.0).to_numpy()
        dts = _infer_dts(index)

        if self.verbose:
            info("Nighttime partitioning ReddyProc (Reichstein et al. 2005) "
                 f"starting for {len(index)} records ({dts} per day).",
                 verbose=self.verbose)

        out = _partition_record(
            nee=df['nee'].to_numpy(), ta=df['ta'].to_numpy(),
            sw_in=df['sw_in'].to_numpy(), nee_f=df['nee_f'].to_numpy(),
            ta_f=df['ta_f'].to_numpy(), doy=doy, hour=hour,
            lat=self.lat, lon=self.lon, utc_offset=self.utc_offset,
            dts=dts, verbose=self.verbose)

        cols = ['NEE_NIGHT_RP', 'RECO_NT_RP', 'GPP_NT_RP', 'RREF_NT_RP', 'E0_NT_RP']
        self._results = pd.DataFrame({c: out[c] for c in cols}, index=index)

        if self.verbose:
            e0 = out['E0_NT_RP']
            e0_val = e0[np.isfinite(e0)][0] if np.isfinite(e0).any() else np.nan
            reco = self._results['RECO_NT_RP']
            info(f"  E0={e0_val:.2f} K, RECO filled "
                 f"{int(reco.notna().sum())}/{len(index)} records.",
                 verbose=self.verbose)
            success("Nighttime partitioning (ReddyProc) finished.",
                    verbose=self.verbose)
        return self

    @property
    def results(self) -> DataFrame:
        """DataFrame of partitioning results (aligned to the input index).

        Columns: ``NEE_NIGHT_RP`` (nighttime NEE used), ``RECO_NT_RP``
        (ecosystem respiration), ``GPP_NT_RP`` (gross primary production),
        ``RREF_NT_RP`` (interpolated reference respiration), ``E0_NT_RP``
        (single temperature sensitivity for the whole record).
        """
        if self._results is None:
            raise RuntimeError("Call .run() before accessing .results.")
        return self._results

    @property
    def reco(self) -> Series:
        """Ecosystem respiration, umol m-2 s-1."""
        return self.results['RECO_NT_RP']

    @property
    def gpp(self) -> Series:
        """Gross primary production, umol m-2 s-1."""
        return self.results['GPP_NT_RP']


def partition_nee_nighttime_reddyproc(nee: Series, ta: Series, sw_in: Series,
                                      nee_f: Series, ta_f: Series, lat: float,
                                      lon: float, utc_offset: float,
                                      verbose: int = 1) -> DataFrame:
    """Functional wrapper around :class:`NighttimePartitioningReddyProc`.

    See :class:`NighttimePartitioningReddyProc` for argument semantics.

    Returns:
        Results DataFrame (RECO_NT_RP, GPP_NT_RP, ...).
    """
    return NighttimePartitioningReddyProc(
        nee=nee, ta=ta, sw_in=sw_in, nee_f=nee_f, ta_f=ta_f,
        lat=lat, lon=lon, utc_offset=utc_offset, verbose=verbose).run().results
