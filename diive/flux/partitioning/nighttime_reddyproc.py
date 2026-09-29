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
   Kelvin with R's ``nls`` (Gauss-Newton, start Rref = 2, E0 = 200), trim the
   5%/95% signed-residual tails, refit, and keep E0 only if its +/-1
   standard-deviation interval lies inside (30, 450) K. Average the three
   lowest-SD estimates and round to two decimals (``fRegrE0fromShortTerm``).
   A window where either ``nls`` call fails (no convergence in 50 iterations,
   singular gradient) is dropped, as in REddyProc.
3. With E0 fixed, re-estimate the reference respiration Rref in centered 7-day
   windows (4-day steps) as the through-origin slope of nighttime NEE on the
   Lloyd-Taylor factor (R's ``lm``), then interpolate Rref linearly to every
   record (``sRegrRref``).
4. RECO = LloydTaylor(Tair_f, Rref, E0); GPP = RECO - NEE_f.

The E0 bounds are (30, 450) K for any temperature. REddyProc's
``sRegrE0fromShortTerm`` has tighter PV-Wave bounds for columns named ``Tair``
(350) and ``Tsoil`` (550), but ``sMRFluxPartition`` always passes the
temperature under the internal name ``FP_Temp_NEW``, so neither branch runs.

If fewer than three short-term windows yield a well-constrained E0, REddyProc
aborts the whole partitioning (return code -111); this port then leaves every
record unpartitioned.

The fits reproduce R's arithmetic, not only its algorithm: ``nls`` and ``lm``
are ported down to LINPACK's QR (``dqrdc2``/``dqrsl``) and the reference-BLAS
sums, and ``quantile``, ``mean``, ``round`` and ``approx`` use R's formulas. A
generic least-squares solver reaches nearly the same optimum, but its last
digits differ, it keeps windows in which ``nls`` fails, and near-ties in the
rounded E0 or in the SD ranking can then change the result.

Measured agreement against native REddyProc 1.3.4 (R 4.5.3, Windows) on
identical half-hourly inputs, with and without gaps in measured TA, SW_IN and
nighttime NEE: CH-DAV 2016 and 2019 (8 runs) and CH-LAE 2017-2019 (12 runs,
measured NEE in only 17-20 % of the records). E0, the three averaged windows,
the set of windows ``nls`` fails in, and the annual RECO and GPP sums are the
same in every run; on the whole ten-year CH-DAV record (2013-2022) E0 is
282.89 on both sides. Per record, RECO and GPP differ by at most 3.6e-15
umol m-2 s-1 at CH-DAV and 7.1e-15 at CH-LAE (larger fluxes, same last bit),
and 95.6-99.8% of the records are bitwise identical. The rest is ``exp()``:
R on Windows computes it in 80-bit x87 arithmetic and rounds it to double,
numpy returns the nearly correctly rounded double, and the two differ by one
unit in the last place for about 0.5% of arguments. With an emulation
of R's ``exp`` in place of numpy's, RECO, GPP, Rref and E0 are bitwise
identical in every record of all 8 CH-DAV runs; with R's own ``exp`` values
looked up (the emulation misses the ~1 in 30 000 arguments where the x87
instruction itself decides), also in all 12 CH-LAE runs, every window fit
included. The individual window fits stay
sensitive to that last bit: in poorly constrained windows (E0 near zero or
negative, SD of hundreds of K) the difference grows to 2e-5 relative, in the
three averaged windows it stays below 1e-7. The same last-bit dependence
means R's own results can differ between platforms whose ``exp`` rounds
differently.

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

import math
from fractions import Fraction

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.core.utils.console import info, warn, success
from diive.flux.partitioning._report import partitioning_report

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
# E0 validity bounds. sEddyProc_sRegrE0fromShortTerm picks [30, 350] when the
# temperature column name contains 'Tair' and [30, 550] for 'Tsoil', but
# sMRFluxPartition always hands it the internal column 'FP_Temp_NEW', whatever
# temperature the user passed. Neither branch ever fires there, so the default
# [30, 450] applies to every nighttime partitioning (and parsE0Regression cannot
# override it: MinE0/MaxE0 are passed explicitly, a second value is an error).
E0_MIN = 30.0
E0_MAX = 450.0

# R stats::nls defaults (nls.control()); fOptimSingleE0 passes no control.
_NLS_MAXITER = 50
_NLS_TOL = 1e-5
_NLS_MINFAC = 1.0 / 1024.0
# numericDeriv forward step, .Machine$double.eps^(1/2).
_NUMDERIV_EPS = math.sqrt(np.finfo(float).eps)
# qr() default tolerance (LINPACK dqrdc2); also lm.fit's.
_QR_TOL = 1e-7
# Range in which reference-BLAS dnrm2 needs no rescaling (LAPACK >= 3.10).
_DNRM2_TSML = 2.0 ** -511
_DNRM2_TBIG = 2.0 ** 486

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
    # hour + (loc + eq), not (hour + loc) + eq: solartime adds the finished
    # local-to-solar difference to the hour; the other grouping rounds differently.
    solar_time_hour = hour + (loc_time + eq_time)

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


# --------------------------------------------------------------------------- #
# R numerics, ported operation by operation
# --------------------------------------------------------------------------- #
# REddyProc fits E0 with R's stats::nls (Gauss-Newton on a LINPACK QR of a
# forward-difference Jacobian) and Rref with lm (the same QR). A generic
# least-squares solver finds nearly the same optimum, but not the same bits, and
# it "succeeds" in windows where nls stops with an error (REddyProc then drops
# the window). The functions below reproduce R's arithmetic: the reference-BLAS
# dot product and norm, dqrdc2/dqrsl, nls_iter, summary.nls, quantile(type = 7),
# mean and round. Checked bitwise against R 4.5.3 (reference BLAS, LAPACK 3.12).


def _ddot(x: np.ndarray, y: np.ndarray) -> float:
    """Reference-BLAS ``ddot``: products summed strictly left to right.

    ``np.cumsum`` accumulates sequentially; ``np.sum``/``np.dot`` do not, and
    the different summation order changes the last bits of every QR.
    """
    if x.size == 0:
        return 0.0
    return float((x * y).cumsum()[-1])


def _dnrm2(x: np.ndarray) -> float:
    """Reference-BLAS ``dnrm2`` (LAPACK >= 3.10, bundled with R).

    In the normal range this is ``sqrt`` of the sequential sum of squares. The
    older scaled algorithm rounds differently and does not match R 4.5.
    """
    if x.size == 0:
        return 0.0
    ax = np.abs(x)
    if ax.max() > _DNRM2_TBIG or ax.min() < _DNRM2_TSML:
        nz = ax[ax > 0]
        if nz.size and (nz.max() > _DNRM2_TBIG or nz.min() < _DNRM2_TSML):
            # Far outside anything a Lloyd-Taylor gradient reaches; LAPACK
            # rescales here. Keep a correct (not bit-exact) norm.
            return float(np.hypot.reduce(nz))
    return math.sqrt(_ddot(ax, ax))


def _dqrdc2(x: np.ndarray, tol: float = _QR_TOL):
    """R's LINPACK ``dqrdc2`` (``qr(x)`` with ``LAPACK = FALSE``).

    Householder QR with R's limited pivoting: a column whose norm falls below
    ``tol`` times its original norm counts as negligible and lowers the rank.
    ``nls`` stops with "singular gradient" whenever the rank is below the
    number of parameters, so the rank test must be R's, not a determinant.

    Returns ``(qr, qraux, rank)`` in R's layout (``x`` is n x p).
    """
    x = np.array(x, dtype=float, copy=True)
    n, p = x.shape
    qraux = np.array([_dnrm2(x[:, j]) for j in range(p)])
    work1 = qraux.copy()
    work2 = qraux.copy()
    work2[work2 == 0.0] = 1.0
    k = p + 1
    for l in range(min(n, p)):
        # Move negligible columns to the end (LINPACK label 80).
        while not (l + 1 >= k or qraux[l] >= work2[l] * tol):
            order = list(range(l + 1, p)) + [l]
            x[:, l:] = x[:, order]
            for arr in (qraux, work1, work2):
                arr[l:] = arr[order]
            k -= 1
        if l == n - 1:
            continue
        nrmxl = _dnrm2(x[l:, l])
        if nrmxl == 0.0:
            continue
        if x[l, l] != 0.0:
            nrmxl = math.copysign(nrmxl, x[l, l])
        x[l:, l] = x[l:, l] * (1.0 / nrmxl)  # dscal by the reciprocal, as LINPACK
        x[l, l] = 1.0 + x[l, l]
        for j in range(l + 1, p):
            t = -_ddot(x[l:, l], x[l:, j]) / x[l, l]
            x[l:, j] = x[l:, j] + t * x[l:, l]
            if qraux[j] == 0.0:
                continue
            r = abs(x[l, j]) / qraux[j]
            tt = max(1.0 - r * r, 0.0)
            if abs(tt) < 1e-6:
                qraux[j] = _dnrm2(x[l + 1:, j])
                work1[j] = qraux[j]
            else:
                qraux[j] = qraux[j] * math.sqrt(tt)
        qraux[l] = x[l, l]
        x[l, l] = -nrmxl
    return x, qraux, min(k - 1, n)


def _qr_qty(qr: np.ndarray, qraux: np.ndarray, rank: int,
            y: np.ndarray) -> np.ndarray:
    """``qr.qty(QR, y)``: Q' y via LINPACK ``dqrsl``."""
    n = qr.shape[0]
    qty = np.array(y, dtype=float, copy=True)
    for j in range(min(rank, n - 1)):
        if qraux[j] == 0.0:
            continue
        h = qr[j:, j].copy()
        h[0] = qraux[j]
        t = -_ddot(h, qty[j:]) / h[0]
        qty[j:] = qty[j:] + t * h
    return qty


def _qr_coef(qr: np.ndarray, qraux: np.ndarray, rank: int,
             y: np.ndarray) -> np.ndarray | None:
    """``qr.coef(QR, y)`` for a full-rank QR; ``None`` where R stops with
    "exact singularity in 'qr.coef'"."""
    b = _qr_qty(qr, qraux, rank, y)[:rank].copy()
    for j in range(rank - 1, -1, -1):
        if qr[j, j] == 0.0:
            return None
        b[j] = b[j] / qr[j, j]
        if j > 0:
            b[:j] = b[:j] + (-b[j]) * qr[:j, j]
    return b


def _r_sum(x: np.ndarray) -> float:
    """R ``sum()`` of doubles.

    R accumulates in long double (80-bit on x86) and rounds once at the end;
    the correctly rounded ``math.fsum`` equals that except for rare
    double-rounding ties. A plain float64 sum does not.
    """
    return math.fsum(x)


def _r_mean(x: np.ndarray) -> float:
    """R ``mean()``: long-double sum plus a correction pass, i.e. the exact mean
    rounded once to double."""
    return float(sum(Fraction(float(v)) for v in x) / len(x))


def _r_round(x: float, digits: int) -> float:
    """R >= 4.0.0 ``round(x, digits)`` (nmath ``private_fround``).

    R picks whichever of the two neighbouring candidates with ``digits``
    decimals is closer in double arithmetic (ties to the even last digit).
    Python's ``round`` decides on the exact decimal value of ``x`` instead and
    disagrees near ties, e.g. for a mean E0 of ``280.675000000001``.
    """
    if not math.isfinite(x) or x == 0.0:
        return x
    if x < 0.0:
        return -_r_round(-x, digits)
    # logb(x) is frexp's exponent minus 1; DBL_DIG is 15.
    l10x = math.log10(2.0) * (0.5 + (math.frexp(x)[1] - 1))
    if l10x + digits > 15:
        return x
    pow10 = 10.0 ** digits
    x10 = pow10 * x
    i10 = math.floor(x10)
    xd = i10 / pow10
    xu = math.ceil(x10) / pow10
    du = xu - x
    dd = x - xd
    return xu if (du < dd or (du == dd and math.fmod(i10, 2.0) == 1.0)) else xd


def _r_quantile7(x: np.ndarray, prob: float) -> float:
    """R ``quantile(x, prob)`` (type 7), with R's interpolation formula.

    numpy's linear method interpolates as ``a + (b - a) * h`` (or from ``b``
    for ``h >= 0.5``); R uses ``(1 - h) * a + h * b``. The trim keeps records
    whose residual equals the quantile, so the threshold must carry R's bits.
    """
    xs = np.sort(x)
    index = 1.0 + max(xs.size - 1, 0) * prob
    lo = math.floor(index)
    hi = math.ceil(index)
    qs = float(xs[lo - 1])
    if index > lo and xs[hi - 1] != qs:
        h = index - lo
        qs = (1.0 - h) * qs + h * float(xs[hi - 1])
    return qs


def _lloyd_taylor_rhs_grad(par: np.ndarray, b: np.ndarray):
    """Model values and R ``numericDeriv`` Jacobian of ``rref * exp(e0 * b)``.

    Forward differences with step ``|p| * sqrt(eps)`` (``sqrt(eps)`` for
    ``p == 0``), divided by that step, as R's C ``numeric_deriv``. Returns
    ``None`` where R stops with "Missing value or an infinity produced when
    evaluating the model".
    """
    # A diverging Gauss-Newton step can overflow exp; that is the failure below.
    with np.errstate(over='ignore', invalid='ignore'):
        rhs = par[0] * np.exp(par[1] * b)
        if not np.isfinite(rhs).all():
            return None
        grad = np.empty((b.size, 2))
        for j in range(2):
            xx = abs(par[j])
            delta = _NUMDERIV_EPS if xx == 0 else xx * _NUMDERIV_EPS
            p2 = par.copy()
            p2[j] = par[j] + delta
            ans_del = p2[0] * np.exp(p2[1] * b)
            if not np.isfinite(ans_del).all():
                return None
            grad[:, j] = (ans_del - rhs) / delta
    return rhs, grad


def _nls_lloyd_taylor(y: np.ndarray, b: np.ndarray, start=(2.0, 200.0)):
    """R ``nls(REco ~ fLloydTaylor(RRef, E0, Temp), start = list(2, 200))``.

    Port of ``nlsModel`` + C ``nls_iter`` with the ``nls.control()`` defaults
    that ``fOptimSingleE0`` uses (``maxiter = 50``, ``tol = 1e-5``,
    ``minFactor = 1/1024``, ``scaleOffset = 0``): Gauss-Newton increments
    ``qr.coef(QR, resid)``, the relative-offset convergence criterion on
    ``qr.qty(QR, resid)``, step halving until the deviance does not increase.
    The standard errors are ``summary.nls``'s,
    ``sqrt(diag(chol2inv(R)) * deviance / (n - 2))``.

    Returns ``(par, se, resid)``, or ``None`` wherever R's ``nls`` stops with an
    error (singular gradient, step factor below ``minFactor``, iteration limit,
    non-finite model values); REddyProc's ``fOptimSingleE0`` turns that error
    into an all-NA window.
    """
    n = y.size
    par = np.array(start, dtype=float)
    state = _lloyd_taylor_rhs_grad(par, b)
    if state is None:
        return None
    _rhs, grad = state
    resid = y - _rhs
    dev = _r_sum(resid * resid)
    qr, qraux, rank = _dqrdc2(grad)
    if rank < 2:
        return None  # singular gradient matrix at initial parameter estimates

    fac = 1.0
    converged = False
    for _ in range(_NLS_MAXITER):
        rr = _qr_qty(qr, qraux, rank, resid)
        with np.errstate(divide='ignore', invalid='ignore'):
            conv = np.sqrt(np.float64(_r_sum(rr[:2] * rr[:2]))
                           / np.float64(0.0 + _r_sum(rr[2:] * rr[2:])))
        if conv <= _NLS_TOL:
            converged = True
            break
        incr = _qr_coef(qr, qraux, rank, resid)
        if incr is None:
            return None
        while fac >= _NLS_MINFAC:
            new_par = par + fac * incr
            state = _lloyd_taylor_rhs_grad(new_par, b)
            if state is None:
                return None
            new_rhs, new_grad = state
            new_resid = y - new_rhs
            new_dev = _r_sum(new_resid * new_resid)
            new_qr, new_qraux, new_rank = _dqrdc2(new_grad)
            # nls tests the rank at every trial point, also at one it rejects.
            if new_rank < 2:
                return None
            if new_dev <= dev:
                par, resid, dev = new_par, new_resid, new_dev
                qr, qraux, rank = new_qr, new_qraux, new_rank
                fac = min(2.0 * fac, 1.0)
                break
            fac /= 2.0
        if fac < _NLS_MINFAC:
            return None
    if not converged:
        return None

    # summary.nls: chol2inv(qr.R(QR)) goes through LAPACK dpotri (dtrti2 +
    # dlauu2); for the 2 x 2 triangle that is written out here so the rounding
    # is LAPACK's.
    ia = 1.0 / qr[0, 0]
    ic = 1.0 / qr[1, 1]
    u12 = (ia * qr[0, 1]) * (-ic)
    xtx_inv_diag = np.array([ia * ia + u12 * u12, ic * ic])
    resvar = dev / (n - 2) if n > 2 else np.nan
    se = np.sqrt(xtx_inv_diag * resvar)
    return par, se, resid


def _lm_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """``coef(lm(y ~ 0 + x))`` through R's ``dqrls`` (QR of the one column).

    Algebraically ``sum(x*y) / sum(x^2)``, but R gets there through a
    Householder reflection, which rounds differently.
    """
    qr, qraux, rank = _dqrdc2(x.reshape(-1, 1))
    if rank < 1:
        return np.nan  # aliased column: lm reports NA
    coef = _qr_coef(qr, qraux, rank, y)
    return np.nan if coef is None else float(coef[0])


def _fit_e0_single(nee_night: np.ndarray, ta_k: np.ndarray, tref_k: float):
    """Port of ``fOptimSingleE0`` (default algorithm): fit, trim, refit.

    Fits Lloyd-Taylor (Rref, E0) to nighttime NEE vs. temperature (Kelvin)
    with R's ``nls`` from (2, 200), keeps the records whose residual
    (data - model) lies within the 5 % and 95 % quantiles (inclusive), and
    refits those, again from (2, 200).

    Returns ``(R_ref, R_ref_SD, E_0, E_0_SD, E_0_trim, E_0_trim_SD)`` as in
    REddyProc's ``NLSRes.F``, or ``None`` if either ``nls`` call fails (the
    window is then all NA in REddyProc).
    """
    # Lloyd-Taylor exponent base, the same expression R evaluates inside
    # fLloydTaylor, computed once instead of on every model call.
    b = (1.0 / (tref_k - T0_K)) - (1.0 / (ta_k - T0_K))

    full = _nls_lloyd_taylor(nee_night, b)
    if full is None:
        return None
    par, se, resid = full

    lo = _r_quantile7(resid, E0_TRIM_PERC / 100.0)
    hi = _r_quantile7(resid, 1.0 - E0_TRIM_PERC / 100.0)
    keep = (resid >= lo) & (resid <= hi)

    trim = _nls_lloyd_taylor(nee_night[keep], b[keep])
    if trim is None:
        return None
    par_t, se_t, _resid_t = trim
    return (float(par[0]), float(se[0]), float(par[1]), float(se[1]),
            float(par_t[1]), float(se_t[1]))


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


_E0_WINDOW_COLS = ('Start', 'End', 'Num', 'TRange', 'R_ref', 'R_ref_SD',
                   'E_0', 'E_0_SD', 'E_0_trim', 'E_0_trim_SD')


def _e0_short_term_windows(nee_night: np.ndarray, ta: np.ndarray,
                           day_counter: np.ndarray, tref_k: float) -> dict:
    """Per-window E0 fits of ``fRegrE0fromShortTerm`` (REddyProc's ``NLSRes.F``).

    Slides a centered 15-day window in 5-day steps. A window gets a row when
    it has more than six nighttime records and a temperature range of at least
    5 K; the fit values of the row are NaN when ``nls`` failed.

    Returns a dict of equally long arrays keyed by :data:`_E0_WINDOW_COLS`.
    """
    rows = []
    valid_all = ~np.isnan(nee_night) & ~np.isnan(ta)
    ta_k_all = ta + 273.15
    los, his = _window_slices(day_counter, E0_WINDOW_HALF, E0_STEP)
    mids = np.arange(E0_WINDOW_HALF + 1, int(day_counter.max()) + 1, E0_STEP)
    for mid, lo, hi in zip(mids, los, his, strict=True):
        m = valid_all[lo:hi]
        num = int(m.sum())
        if num <= E0_MIN_ENTRIES:
            continue
        ta_k = ta_k_all[lo:hi][m]
        t_range = float(np.max(ta_k) - np.min(ta_k))
        if t_range < E0_TEMP_RANGE:
            continue
        fit = _fit_e0_single(nee_night[lo:hi][m], ta_k, tref_k)
        if fit is None:
            fit = (np.nan,) * 6
        rows.append((mid - E0_WINDOW_HALF, mid + E0_WINDOW_HALF, num, t_range)
                    + tuple(fit))
    table = np.array(rows, dtype=float).reshape(-1, len(_E0_WINDOW_COLS))
    return {c: table[:, i] for i, c in enumerate(_E0_WINDOW_COLS)}


def _e0_from_windows(windows: dict) -> float:
    """The E0 that ``fRegrE0fromShortTerm`` reports for a window table.

    Keeps estimates whose +/-1 SD interval lies inside
    (:data:`E0_MIN`, :data:`E0_MAX`), orders them by SD (stable, as R's
    ``order``), and returns ``round(mean(best three), 2)`` with R's ``mean`` and
    ``round``. NaN when fewer than three are valid (REddyProc aborts, -111).
    """
    e0_trim = windows['E_0_trim']
    e0_trim_sd = windows['E_0_trim_SD']
    with np.errstate(invalid='ignore'):
        valid = ((e0_trim - e0_trim_sd > E0_MIN)
                 & (e0_trim + e0_trim_sd < E0_MAX))
    if valid.sum() < E0_NUM_BEST:
        return np.nan
    order = np.argsort(e0_trim_sd[valid], kind='stable')
    best = e0_trim[valid][order[:E0_NUM_BEST]]
    return _r_round(_r_mean(best), 2)


def _regr_e0_from_short_term(nee_night: np.ndarray, ta: np.ndarray,
                             day_counter: np.ndarray, tref_k: float) -> float:
    """Port of ``fRegrE0fromShortTerm``: one representative E0 for the record.

    Fits E0 per short-term window (:func:`_e0_short_term_windows`) and averages
    the three well-constrained estimates with the smallest standard deviation
    (:func:`_e0_from_windows`). Returns NaN when fewer than three are valid
    (REddyProc aborts in that case).
    """
    return _e0_from_windows(
        _e0_short_term_windows(nee_night, ta, day_counter, tref_k))


_RREF_WINDOW_COLS = ('Start', 'End', 'Num', 'MeanH', 'R_ref')


def _rref_windows(nee_night: np.ndarray, ta: np.ndarray,
                  day_counter: np.ndarray, e0: float, tref_k: float) -> dict:
    """Per-window Rref regressions of ``sRegrRref`` (REddyProc's ``LMRes.F``).

    Slides a centered 7-day window in 4-day steps. Windows with more than two
    nighttime records get a row: ``MeanH`` is ``round(mean(which(Subset.b)))``
    (1-based record index, ties to even as R's ``round``), ``R_ref`` the slope
    of ``lm(NEE ~ 0 + fLloydTaylor(1, E0, T))``.

    Returns a dict of equally long arrays keyed by :data:`_RREF_WINDOW_COLS`.
    """
    rows = []
    valid_all = ~np.isnan(nee_night) & ~np.isnan(ta)
    ta_k_all = ta + 273.15
    los, his = _window_slices(day_counter, RREF_WINDOW_HALF, RREF_STEP)
    mids = np.arange(RREF_WINDOW_HALF + 1, int(day_counter.max()) + 1, RREF_STEP)
    for mid, lo, hi in zip(mids, los, his, strict=True):
        m = valid_all[lo:hi]
        num = int(m.sum())
        if num <= RREF_MIN_ENTRIES:
            continue
        mean_h = round(float((lo + np.nonzero(m)[0] + 1).mean()))
        factor = lloyd_taylor_kelvin(ta_k_all[lo:hi][m], 1.0, e0, tref_k)
        rref = _lm_through_origin(factor, nee_night[lo:hi][m])
        rows.append((mid - RREF_WINDOW_HALF, mid + RREF_WINDOW_HALF, num,
                     mean_h, rref))
    table = np.array(rows, dtype=float).reshape(-1, len(_RREF_WINDOW_COLS))
    return {c: table[:, i] for i, c in enumerate(_RREF_WINDOW_COLS)}


def _interpolate_gaps(data: np.ndarray) -> np.ndarray:
    """Port of REddyProc ``fInterpolateGaps`` (``approx``, constant ends).

    Uses R's ``approx`` formula ``y0 + (y1 - y0) * ((x - x0) / (x1 - x0))``;
    ``np.interp`` computes the slope first and rounds differently. All NaN
    stays all NaN (R's ``approx`` stops with an error there).
    """
    n = data.size
    known = np.flatnonzero(~np.isnan(data))
    if known.size == 0:
        return data.copy()
    data = data.copy()
    data[0] = data[known[0]]
    data[-1] = data[known[-1]]
    known = np.flatnonzero(~np.isnan(data))
    if known.size < 2:
        return data  # n == 1
    xk = known + 1.0
    yk = data[known]
    v = np.arange(1, n + 1, dtype=float)
    i = np.clip(np.searchsorted(xk, v, side='right') - 1, 0, xk.size - 2)
    j = i + 1
    out = yk[i] + (yk[j] - yk[i]) * ((v - xk[i]) / (xk[j] - xk[i]))
    out = np.where(v == xk[i], yk[i], out)
    return np.where(v == xk[j], yk[j], out)


def _regr_rref(nee_night: np.ndarray, ta: np.ndarray, day_counter: np.ndarray,
               e0: float, tref_k: float) -> np.ndarray:
    """Port of ``sRegrRref``: time-varying Rref with E0 held fixed.

    Per window (:func:`_rref_windows`) the through-origin slope is placed at
    the window's mean record index; negative slopes become NA (``R_ref_ok``).
    The estimates are written in window order, so a later window with the same
    index overwrites an earlier one, NA included, as R's
    ``Rref[LMRes.F$MeanH] <- LMRes.F$R_ref_ok``. Then linear interpolation to
    every record, constant beyond the first and last estimate.
    """
    windows = _rref_windows(nee_night, ta, day_counter, e0, tref_k)
    rref_at = np.full(nee_night.size, np.nan)
    for mean_h, rref in zip(windows['MeanH'], windows['R_ref'], strict=True):
        rref_at[int(mean_h) - 1] = rref if rref >= 0 else np.nan
    return _interpolate_gaps(rref_at)


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
        >>> import diive as dv
        >>> df = dv.load_exampledata_parquet()
        >>> part = dv.flux.NighttimePartitioningReddyProc(
        ...     nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
        ...     nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'],
        ...     lat=46.815, lon=9.855, utc_offset=1)
        >>> part.run()  # then part.results -> DataFrame with RECO_NT_RP, GPP_NT_RP, ...
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
                 verbose: int = 2):
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
            verbose: Console verbosity level (0 silent, 1 warnings, 2 progress
                + report, 3 debug). Default 2.
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

        self.report()
        if self.verbose:
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

    def report(self) -> None:
        """Print a Rich per-year summary of the partitioning result."""
        partitioning_report(
            title="Nighttime NEE Partitioning REddyProc (Reichstein et al. 2005)",
            reference="Wutzler et al. (2018), https://doi.org/10.5194/bg-15-5015-2018",
            results=self.results, reco_col='RECO_NT_RP', gpp_col='GPP_NT_RP',
            e0_col='E0_NT_RP', e0_unit='K', verbose=self.verbose)


def partition_nee_nighttime_reddyproc(nee: Series, ta: Series, sw_in: Series,
                                      nee_f: Series, ta_f: Series, lat: float,
                                      lon: float, utc_offset: float,
                                      verbose: int = 2) -> DataFrame:
    """Functional wrapper around :class:`NighttimePartitioningReddyProc`.

    See :class:`NighttimePartitioningReddyProc` for argument semantics.

    Returns:
        Results DataFrame (RECO_NT_RP, GPP_NT_RP, ...).
    """
    return NighttimePartitioningReddyProc(
        nee=nee, ta=ta, sw_in=sw_in, nee_f=nee_f, ta_f=ta_f,
        lat=lat, lon=lon, utc_offset=utc_offset, verbose=verbose).run().results
