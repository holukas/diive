"""
NIGHTTIME PARTITIONING ONEFLUX: NEE -> GPP + RECO (Reichstein et al. 2005)
=========================================================================

Faithful, vectorized Python port of the ONEFlux nighttime partitioning
reference implementation (``oneflux.partition.nighttime``), which itself is a
clean-room reimplementation of the original PV-Wave code.

The nighttime method estimates ecosystem respiration (RECO) from the
temperature response of *nighttime* NEE, then derives gross primary production
(GPP) as ``GPP = RECO - NEE``. During the night there is no photosynthesis, so
measured NEE equals respiration; fitting a temperature-response function to
those nighttime fluxes and extrapolating to daytime temperatures yields the
daytime respiration that is otherwise masked by photosynthetic uptake.

Algorithm (per calendar year):

1. Flag nighttime records (sun below horizon AND SW_IN < 10 W m-2).
2. Fit the Lloyd & Taylor (1994) respiration model to nighttime NEE vs. air
   temperature over the full year (10% trimmed non-linear least squares) -
   a robust fallback estimate of the parameters Rref and E0.
3. Refit in short overlapping windows (5-day steps, 14-day windows) to get
   time-resolved Rref/E0 with standard errors.
4. Determine one representative temperature sensitivity E0 for the year from
   the (up to three) windows with the smallest E0 standard error.
5. With E0 held fixed, re-estimate the reference respiration Rref in short
   windows (4-day steps, 8-day windows), both ordinary and outlier-robust,
   then interpolate Rref to every record.
6. RECO = LloydTaylor(Tair_f, Rref, E0); GPP = RECO - NEE.

A year with no well-constrained short-term E0 (at least one window with small
absolute *and* relative E0 standard error) is left entirely unpartitioned, as
in ONEFlux.

Reference:
    Reichstein, M. et al. (2005). On the separation of net ecosystem exchange
    into assimilation and ecosystem respiration: review and improved algorithm.
    Global Change Biology, 11(9), 1424-1439.
    https://doi.org/10.1111/j.1365-2486.2005.001002.x

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

# Lloyd & Taylor reference/regression temperatures (degC), as used by ONEFlux.
TREF = 15.0
T0 = -46.02

# Windowing / fitting constants (match ONEFlux defaults).
STEP_SIZE = 5          # days the long-term window slides by
WINDOW_SIZE = 14       # length of the long-term window (days)
MIN_ENTRIES = 6        # minimum nighttime records in a window to attempt a fit
MIN_TRANGE = 5.0       # minimum temperature range (degC) in a window
DAY_MIN_SW_IN = 10.0   # SW_IN below this (W m-2) counts as night
TRIM_PERC = 10.0       # residual trimming percentage for the robust fit

REANALYSE_STEP = 4     # days the Rref re-analysis window slides by
REANALYSE_WINDOW = 8   # length of the Rref re-analysis window (days)

# Levenberg-Marquardt tuning (faithful to ONEFlux).
_STEP_BOUND_FACTOR = 0.25
_NO_CONVERGENCE_RETRY = 20


def lloyd_taylor(ta: np.ndarray, rref: float | np.ndarray, e0: float | np.ndarray,
                 tref: float = TREF, t0: float = T0) -> np.ndarray:
    """Lloyd & Taylor (1994) respiration as a function of temperature.

    Args:
        ta: Air (or soil) temperature in degC.
        rref: Reference respiration at ``tref`` (umol m-2 s-1).
        e0: Temperature sensitivity (degC).
        tref: Reference temperature (degC).
        t0: Regression temperature (degC).

    Returns:
        Respiration in the same units as ``rref``.
    """
    return rref * np.exp(e0 * ((1.0 / (tref - t0)) - (1.0 / (ta - t0))))


def sunrise_sunset(doy: np.ndarray, lat: float) -> tuple[np.ndarray, np.ndarray]:
    """True-solar-time sunrise/sunset hours for a day-of-year and latitude.

    Port of the ONEFlux ``sunrs`` routine (Linacre 1992 true solar time). Times
    are decimal hours relative to solar noon at 12:00, so they assume timestamps
    in local standard time.

    Args:
        doy: Day of year (1-366).
        lat: Site latitude in decimal degrees.

    Returns:
        Tuple ``(sunrise, sunset)`` of decimal hours.
    """
    # Days from Jan 1 to Mar 21 (spring equinox), as in the original code.
    march21_doy_diff = 80
    pi = 3.1415926
    rad_per_day = 2.0 * pi / 365.0
    decl_amp = 23.45 * pi / 180.0
    hours_per_hs = 24.0 / (2.0 * pi)

    lat_rad = lat * pi / 180.0
    decl = decl_amp * np.sin(rad_per_day * (doy - march21_doy_diff))
    hs = np.arccos(-np.tan(lat_rad) * np.tan(decl))
    sunrise = 12.0 - hs * hours_per_hs
    sunset = 12.0 + hs * hours_per_hs
    return sunrise, sunset


def _pct(array: np.ndarray, percent: float) -> float:
    """Rank-based percentile matching the ONEFlux ``pct`` helper.

    Not a standard percentile: it returns the value at the smallest integer
    rank strictly greater than ``n * percent / 100`` (averaging with the
    preceding rank when that critical rank is an integer).
    """
    nonnan = array[~np.isnan(array)]
    n = nonnan.size
    if n <= 1:
        raise ValueError("No non-NA value in percentile calculation")

    critical_rank = n * percent / 100.0
    # No rank exceeds the critical rank -> return the maximum.
    if n <= critical_rank:
        return float(np.max(nonnan))

    s = np.sort(nonnan)
    k = int(np.floor(critical_rank)) + 1  # smallest integer rank > critical_rank (1-based)
    val_k = s[k - 1]
    if float(critical_rank).is_integer() and (k - 1) >= 1:
        return float((val_k + s[k - 2]) / 2.0)
    return float(val_k)


def _leastsq_fit(func, x0, entries: int, maxfev: int, retry: bool = True):
    """scipy ``leastsq`` wrapper replicating ONEFlux step bound + retry + SE.

    Returns ``(params, std_devs, residuals)``.
    """
    pars, cov_x, infodict, _msg, ier = leastsq(
        func, x0, full_output=True, maxfev=maxfev, factor=_STEP_BOUND_FACTOR)

    if ier != 1 and infodict['nfev'] >= maxfev and retry:
        return _leastsq_fit(func, x0, entries, maxfev * _NO_CONVERGENCE_RETRY, retry=False)

    npar = len(x0)
    residuals = infodict['fvec']
    if entries > npar and cov_x is not None:
        s_squared = (residuals ** 2).sum() / (entries - npar)
        std_devs = np.sqrt(np.abs(np.diag(cov_x * s_squared)))
    else:
        std_devs = np.full(npar, np.nan)
    return np.asarray(pars, dtype=float), std_devs, residuals


def _fit_lloyd_taylor(nee_night: np.ndarray, tair: np.ndarray,
                      xguess=(2.0, 200.0), trim_perc: float = TRIM_PERC):
    """Trimmed non-linear least-squares fit of Lloyd-Taylor (Rref, E0).

    Port of ONEFlux ``nlinlts1``. Returns ``(rref, e0, rref_se, e0_se)`` or
    ``None`` when there are too few valid points.
    """
    npara = 2
    nonnan_indep = ~np.isnan(tair)
    if nonnan_indep.sum() < npara * 3:
        return None
    nonnan_dep = ~np.isnan(nee_night)
    if (nonnan_indep & nonnan_dep).sum() < npara * 3:
        return None

    # Clean dependent variable so NAs in the independent variable are not used.
    clean_dep = nee_night.copy()
    clean_dep[~nonnan_indep] = np.nan
    nan_dep_mask = np.isnan(clean_dep)
    temp = tair

    def trimmed_residuals(par):
        rref, e0 = par
        prediction = lloyd_taylor(temp, rref, e0)
        residuals = clean_dep - prediction
        residuals[nan_dep_mask] = 0.0
        if trim_perc == 0.0:
            return residuals
        absolute = np.abs(residuals)
        cutoff = _pct(absolute, 100.0 - trim_perc)
        residuals[absolute > cutoff] = 0.0
        return residuals

    pars, std_devs, _res = _leastsq_fit(
        trimmed_residuals, list(xguess), entries=len(clean_dep),
        maxfev=1000 * (len(clean_dep) + 1))
    return float(pars[0]), float(pars[1]), float(std_devs[0]), float(std_devs[1])


def _interp_missing(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaNs over coordinate ``x``, end values clamped.

    Equivalent to ONEFlux ``ipolmiss`` (linear, exact), implemented with
    ``numpy.interp`` (which clamps to the first/last valid value outside the
    sampled range).
    """
    mask = ~np.isnan(values)
    count = mask.sum()
    if count < 2 or count == values.size:
        return values
    return np.interp(x, x[mask], values[mask])


def _reanalyse_rref(nee_night: np.ndarray, tair: np.ndarray, tair_f: np.ndarray,
                    julday_dec: np.ndarray, e0: float,
                    step: int = REANALYSE_STEP, window: int = REANALYSE_WINDOW):
    """Re-estimate Rref with E0 held fixed, ordinary and outlier-robust.

    Port of ONEFlux ``reanalyse_rref``. Returns ``(reco, reco_rob, rref_ord)``
    arrays (RECO computed from gap-filled temperature ``tair_f``).
    """
    n = nee_night.size
    rref_ord = np.full(n, np.nan)
    rref_trim = np.full(n, np.nan)

    julday_int = (julday_dec + 0.5).astype(np.int64)
    last_day = int(julday_dec[-1])

    valid = (~np.isnan(tair)) & (~np.isnan(nee_night))

    for j in range(1, last_day, step):
        mask = (julday_int >= j) & (julday_int < (j + window)) & valid
        count = int(mask.sum())
        if count <= 2:
            continue

        idx = np.where(mask)[0]
        mid = int(round(idx.mean()))
        reco_average = nee_night[mask].mean()
        # E0 is fixed -> respiration is linear in Rref: nee = b * lloyd_fac.
        lloyd_fac = lloyd_taylor(tair[mask], rref=1.0, e0=e0)
        nee_sub = nee_night[mask]

        denom = (lloyd_fac ** 2).sum()
        b = (lloyd_fac * nee_sub).sum() / denom if denom > 0 else np.nan
        rref_ord[mid] = b if b > 1e-6 else 1e-6

        # Outlier-robust variant: drop the largest deviations from the mean.
        deviation = np.abs(nee_sub - reco_average)
        cutoff = _pct(deviation, 95.0)
        trim = deviation < cutoff
        if trim.sum() > 0:
            denom_t = (lloyd_fac[trim] ** 2).sum()
            b_t = (lloyd_fac[trim] * nee_sub[trim]).sum() / denom_t if denom_t > 0 else np.nan
            rref_trim[mid] = b_t if b_t > 1e-6 else 1e-6

    rref_ord = _interp_missing(rref_ord, julday_dec)
    rref_trim = _interp_missing(rref_trim, julday_dec)

    reco = lloyd_taylor(tair_f, rref=rref_ord, e0=e0)
    reco_rob = lloyd_taylor(tair_f, rref=rref_trim, e0=e0)
    return reco, reco_rob, rref_ord


def _partition_one_year(nee: np.ndarray, tair: np.ndarray, sw_in: np.ndarray,
                        nee_f: np.ndarray, tair_f: np.ndarray,
                        doy: np.ndarray, hr: np.ndarray, lat: float,
                        verbose: int = 1) -> dict:
    """Run the nighttime partitioning for a single year of arrays.

    All arrays are 1D and aligned. Returns a dict of result arrays.
    """
    n = nee.size
    out = {
        'NEE_NIGHT_OF': np.full(n, np.nan),
        'RECO_NT_OF': np.full(n, np.nan),
        'RECO_NT_OF_ROB': np.full(n, np.nan),
        'GPP_NT_OF': np.full(n, np.nan),
        'GPP_NT_OF_ROB': np.full(n, np.nan),
        'RREF_NT_OF': np.full(n, np.nan),
        'E0_NT_OF': np.full(n, np.nan),
    }

    # --- Day/night flag and nighttime NEE ---
    if lat is not None and np.isfinite(lat):
        sunrise, sunset = sunrise_sunset(doy, lat)
        daylight = (hr > sunrise) & (hr < sunset)
    else:
        daylight = np.zeros(n, dtype=bool)

    with np.errstate(invalid='ignore'):
        night_mask = (sw_in < DAY_MIN_SW_IN) & (~daylight)
    nee_night = np.where(night_mask, nee, np.nan)
    out['NEE_NIGHT_OF'] = nee_night

    # --- 1) Full-year (fallback) fit ---
    full = _fit_lloyd_taylor(nee_night, tair)
    if full is None:
        warn("Nighttime partitioning: full-year optimization failed (too few points).",
             verbose=verbose)
        rref_1, e0_1 = np.nan, np.nan
    else:
        rref_1, e0_1, _, _ = full
        e0_1 = max(0.0, min(450.0, e0_1))

    # --- 2) Windowed fits (5-day step, 14-day window) ---
    julmin, julmax = int(doy[0]), int(np.max(doy))
    win_rref, win_e0, win_e0_se, win_mid = [], [], [], []
    valid = (~np.isnan(nee_night)) & (~np.isnan(tair))

    for jday in range(julmin, julmax + 1, STEP_SIZE):
        w_mask = (doy >= jday) & (doy < jday + WINDOW_SIZE) & valid
        w_len = int(w_mask.sum())
        if w_len <= MIN_ENTRIES:
            continue
        w_where = np.where(w_mask)[0]
        temp_range = np.max(tair[w_mask]) - np.min(tair[w_mask])
        if temp_range < MIN_TRANGE:
            continue
        fit = _fit_lloyd_taylor(nee_night[w_mask], tair[w_mask])
        if fit is None:
            continue
        rref, e0, _rref_se, e0_se = fit
        win_rref.append(rref)
        win_e0.append(e0)
        win_e0_se.append(e0_se)
        win_mid.append(w_where[w_len // 2])

    win_rref = np.asarray(win_rref)
    win_e0 = np.asarray(win_e0)
    win_e0_se = np.asarray(win_e0_se)
    win_mid = np.asarray(win_mid, dtype=np.int64)

    # --- 3) Determine representative E0 from best (lowest-SE) windows ---
    best_e0 = np.nan
    if win_e0.size > 0:
        max_e0 = 350.0
        in_range = (win_e0 > 30.0) & (win_e0 < max_e0)
        if in_range.sum() > 1:
            idx_in = np.where(in_range)[0]
            order = np.argsort(win_e0_se[in_range])
            take = min(3, order.size)
            selected = idx_in[order[:take]]
            best_e0 = float(np.mean(win_e0[selected]))

    if not np.isfinite(best_e0):
        # Fall back to the full-year E0 estimate.
        if verbose:
            warn("Nighttime partitioning: no short-term E0; using full-year E0.",
                 verbose=verbose)
        best_e0 = e0_1

    if not np.isfinite(best_e0):
        return out  # nothing more can be done for this year

    out['E0_NT_OF'][:] = best_e0

    # --- ONEFlux gate: partition only if at least one short-term window
    # produced a well-constrained E0 (low absolute AND relative standard error).
    # ONEFlux leaves RECO/GPP empty for the whole year otherwise (the count>0
    # check in nighttime.flux_partition). Thresholds inlined as in the original.
    with np.errstate(invalid='ignore', divide='ignore'):
        well_constrained = ((win_e0_se < 100.0) & ((win_e0_se / win_e0) < 0.5)
                            & (win_e0 > 50.0) & (win_e0 < 450.0))
    if not well_constrained.any():
        warn("Nighttime partitioning: no well-constrained short-term E0; "
             "year left unpartitioned (ONEFlux parity).", verbose=verbose)
        return out

    # --- 4) Re-estimate Rref with E0 fixed, then RECO and GPP ---
    julday_dec = doy + (hr / 24.0)
    reco, reco_rob, rref_ord = _reanalyse_rref(
        nee_night=nee_night, tair=tair, tair_f=tair_f,
        julday_dec=julday_dec, e0=best_e0)

    out['RECO_NT_OF'] = reco
    out['RECO_NT_OF_ROB'] = reco_rob
    out['RREF_NT_OF'] = rref_ord

    # GPP = RECO - NEE, using the gap-filled NEE for continuity.
    out['GPP_NT_OF'] = reco - nee_f
    out['GPP_NT_OF_ROB'] = reco_rob - nee_f
    return out


class NighttimePartitioningOneFlux:
    """Partition NEE into GPP and RECO with the nighttime method (ONEFlux).

    Faithful, vectorized port of the ONEFlux nighttime partitioning
    (Reichstein et al. 2005). Each calendar year in the input is partitioned
    independently.

    Example: ``examples/flux/partitioning/partitioning_nighttime_oneflux.py``

    Example:
        >>> part = NighttimePartitioningOneFlux(
        ...     nee=df['NEE_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
        ...     nee_f=df['NEE_f'], ta_f=df['Tair_f'], lat=46.815)
        >>> part.run()
        >>> results = part.results   # DataFrame with RECO_NT_OF, GPP_NT_OF, ...
    """

    def __init__(self,
                 nee: Series,
                 ta: Series,
                 sw_in: Series,
                 nee_f: Series,
                 ta_f: Series,
                 lat: float,
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
            verbose: Console verbosity level (0 silent, 1 progress).
        """
        self._inputs = self._validate(nee, ta, sw_in, nee_f, ta_f)
        self.lat = float(lat)
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

    def run(self) -> "NighttimePartitioningOneFlux":
        """Run the partitioning and populate :attr:`results`."""
        df = self._inputs
        index = df.index
        doy_all = index.dayofyear.to_numpy()
        hr_all = (index.hour + index.minute / 60.0).to_numpy()
        years = index.year.to_numpy()

        cols = ['NEE_NIGHT_OF', 'RECO_NT_OF', 'RECO_NT_OF_ROB', 'GPP_NT_OF', 'GPP_NT_OF_ROB',
                'RREF_NT_OF', 'E0_NT_OF']
        result = pd.DataFrame(index=index, columns=cols, dtype=float)

        if self.verbose:
            info("Nighttime partitioning (Reichstein et al. 2005) starting "
                 f"for {len(np.unique(years))} year(s).", verbose=self.verbose)

        for year in np.unique(years):
            ymask = years == year
            year_out = _partition_one_year(
                nee=df['nee'].to_numpy()[ymask],
                tair=df['ta'].to_numpy()[ymask],
                sw_in=df['sw_in'].to_numpy()[ymask],
                nee_f=df['nee_f'].to_numpy()[ymask],
                tair_f=df['ta_f'].to_numpy()[ymask],
                doy=doy_all[ymask],
                hr=hr_all[ymask],
                lat=self.lat,
                verbose=self.verbose,
            )
            for col in cols:
                result.loc[ymask, col] = year_out[col]
            if self.verbose:
                e0 = year_out['E0_NT_OF']
                e0_val = e0[np.isfinite(e0)][0] if np.isfinite(e0).any() else np.nan
                reco = result.loc[ymask, 'RECO_NT_OF']
                info(f"  {int(year)}: E0={e0_val:.1f}, "
                     f"RECO filled {int(reco.notna().sum())}/{int(ymask.sum())} records.",
                     verbose=self.verbose)

        self._results = result
        if self.verbose:
            success("Nighttime partitioning finished.", verbose=self.verbose)
        return self

    @property
    def results(self) -> DataFrame:
        """DataFrame of partitioning results (aligned to the input index).

        Columns: ``NEE_NIGHT_OF`` (nighttime NEE used), ``RECO_NT_OF`` /
        ``RECO_NT_OF_ROB`` (ecosystem respiration, ordinary / outlier-robust),
        ``GPP_NT_OF`` / ``GPP_NT_OF_ROB`` (gross primary production), ``RREF_NT_OF``
        (interpolated reference respiration), ``E0_NT_OF`` (per-year temperature
        sensitivity).
        """
        if self._results is None:
            raise RuntimeError("Call .run() before accessing .results.")
        return self._results

    @property
    def reco(self) -> Series:
        """Ecosystem respiration (ordinary), umol m-2 s-1."""
        return self.results['RECO_NT_OF']

    @property
    def gpp(self) -> Series:
        """Gross primary production, umol m-2 s-1."""
        return self.results['GPP_NT_OF']


def partition_nee_nighttime_oneflux(nee: Series, ta: Series, sw_in: Series,
                            nee_f: Series, ta_f: Series, lat: float,
                            verbose: int = 1) -> DataFrame:
    """Functional wrapper around :class:`NighttimePartitioningOneFlux`.

    See :class:`NighttimePartitioningOneFlux` for argument semantics.

    Returns:
        Results DataFrame (RECO_NT_OF, GPP_NT_OF, ...).
    """
    return NighttimePartitioningOneFlux(
        nee=nee, ta=ta, sw_in=sw_in, nee_f=nee_f, ta_f=ta_f,
        lat=lat, verbose=verbose).run().results
