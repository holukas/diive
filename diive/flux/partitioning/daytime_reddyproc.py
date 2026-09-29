"""
DAYTIME PARTITIONING REDDYPROC: NEE -> GPP + RECO (Lasslop et al. 2010)
======================================================================

Faithful, vectorized Python port of the REddyProc *daytime* partitioning
(``partitionNEEGL`` in REddyProc's ``PartitioningLasslop10.R``), the Lasslop et
al. (2010) light-response-curve (LRC) method. It is the daytime companion to the
nighttime REddyProc port (:mod:`diive.flux.partitioning.nighttime_reddyproc`)
and emits ``*_DT_RP`` columns so both can coexist in one dataframe.

The daytime method fits, for short overlapping windows, a rectangular-hyperbola
light-response curve to *daytime* NEE,

    NEP = GPP - RECO,
    GPP = (Amax * alpha * Rg) / (alpha * Rg + Amax),
    Amax = beta            (if VPD <= VPD0 or k == 0)
         = beta * exp(-k * (VPD - VPD0))   (if VPD > VPD0),
    RECO = RRef * exp(E0 * (1/(Tref - T0) - 1/(Tair - T0)))   (Lloyd & Taylor),

with five parameters per window: ``k`` (VPD sensitivity), ``beta`` (GPP
saturation), ``alpha`` (initial slope), ``RRef`` (basal respiration) and ``E0``
(temperature sensitivity). The temperature sensitivity ``E0`` is *not* fitted on
daytime data; it is estimated beforehand from nighttime NEE and held fixed while
the four light parameters are optimized.

Algorithm (REddyProc defaults, ``partGLControl()``):

1. Day/night split per record: ``Rg <= 4`` W m-2 AND potential radiation
   ``<= 0`` is night; ``Rg > 4`` AND potential radiation ``> 0`` is day
   (potential radiation uses the solar geometry latitude/longitude/UTC offset,
   identical to the nighttime port).
2. Temperature sensitivity ``E0`` from nighttime NEE: per centered 12-day window
   (4-day reference grid, 2-day step) fit Lloyd-Taylor in Kelvin with R's
   Gauss-Newton ``nls`` (reference temperature = window median, bounded
   ``E0 in [50, 400]``), extending the window to 24/48 days where a fit fails;
   then smooth the per-window ``E0`` across time with a Gaussian process
   (``mlegp``) and re-estimate ``RRef`` per window by linear regression.
3. Light-response curve per centered 4-day window (2-day step): fit ``k``,
   ``beta``, ``alpha``, ``RRef`` (``E0`` fixed) by penalized least squares
   (Lasslop priors, NEE-uncertainty weighting) with R's BFGS ``optim`` from
   three starting points, picking the lowest-cost fit, plus the Lasslop bounds
   refit cascade (fix VPD / fix alpha / reject out-of-range parameters). A
   window with fewer than 10 usable records that have VPD is fitted on all
   records without the VPD effect (``k = 0``).
4. Predict RECO and GPP for every record by interpolating the two neighboring
   windows' parameter sets with distance-based weights. Where VPD is missing
   and GPP therefore NA, all windows are refitted without the VPD effect and
   RECO and GPP of those records come from that fit
   (``isRefitMissingVPDWithNeglectVPDEffect``).

The port reproduces REddyProc's arithmetic, not only its algorithm, because
every stage is sensitive to the last bit. The nighttime ``nls`` fits use a
forward-difference Jacobian (step ``sqrt(eps) * |p|``), which turns a one-ulp
difference into ~1e-8, and each window starts from the previous window's E0,
so a difference travels along the year. The ``mlegp`` likelihood is flat at its
optimum: any other optimizer stops elsewhere with the same likelihood to 1e-8,
and the smoothed E0 then differs by 1e-4. The LRC ``optim`` stops at a relative
cost change of 1e-3, so in ill-conditioned windows (k near 0, alpha fixed or
not) a tiny input difference changes where it stops by up to 1e-3. Hence:
``nls`` runs on LINPACK's QR and R's sums and mean (shared with the nighttime
port), ``mlegp`` is ported run for run (its SFMT random starts, R 2.4.0
Nelder-Mead, liblbfgs), ``optim`` loop for loop, ``quantile`` with R's formula,
``lm`` through R's QR, and ``exp`` is rounded as R on Windows rounds it
(:func:`_exp_r`). BLAS is limited to one thread during the run, so
the result does not depend on the thread count.

Measured agreement against native REddyProc 1.3.4 / mlegp 3.1.9 (R 4.5.3,
Windows) on identical half-hourly inputs, with and without gaps in NEE,
drivers and nighttime data: CH-DAV 2016 and 2019 (8 runs) and CH-LAE
2017-2019 (12 runs, measured NEE in only 17-20 % of the records). Window
acceptance and convergence codes are identical in every run. Stage by stage,
each stage fed REddyProc's own input: the smoothed E0 agrees to 7e-15 and its
SD to 7e-14 (the mlegp hyperparameters beta are bitwise identical, mu and
sig2 differ in the last bit), RRef to 5e-15; the LRC parameters are bitwise
identical in 91-99 % of the windows and within 1e-8 in the rest (8e-9 in one
window, which becomes bitwise identical when R's 80-bit long-double sums are
emulated; ``math.fsum`` is used instead, the emulation is too slow). End to
end, RECO and GPP differ by at most 1e-4 umol m-2 s-1 in 4 of the 8 CH-DAV and
9 of the 12 CH-LAE runs, by 3e-4 to 1e-3 in five more, by 0.013 in CH-DAV 2016
with driver gaps and by 0.022 in CH-LAE 2019 with sparse nights; annual sums
agree to 0.001 %. These differences start in R's ``exp``: it runs in x87
extended precision, and :func:`_exp_r` reproduces its rounding except for
about 1 in 30 000 arguments, where the x87 instruction itself decides. With
R's own exp values in their place, the 12/24/48-day nighttime E0 fits are
bitwise identical to REddyProc's. Otherwise a few of the ~540 window fits
differ by 1e-10 to 4e-8. Each window starts from the previous window's E0,
which carries the difference along the year (E0 to 4e-7), the smoothed E0
follows (2e-7), and the LRC stop criterion (relative cost change 1e-3)
amplifies it in a few ill-conditioned windows; there even the last bit of the
smoothed E0 (the GP's linear algebra, OpenBLAS vs. R's reference LAPACK) moves
parameters by up to 2e-3 (CH-LAE 2019 with sparse nights, where the nighttime
fits are bitwise identical). R's own results differ between platforms at this
level.

Reference:
    Lasslop, G. et al. (2010). Separation of net ecosystem exchange into
    assimilation and respiration using a light response curve approach: critical
    issues and global evaluation. Global Change Biology, 16(1), 187-208.
    https://doi.org/10.1111/j.1365-2486.2009.02041.x

    Wutzler, T. et al. (2018). Basic and extensible post-processing of eddy
    covariance flux data with REddyProc. Biogeosciences, 15, 5015-5030.
    https://doi.org/10.5194/bg-15-5015-2018

Example: ``examples/flux/partitioning/partitioning_daytime_reddyproc.py``

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.linalg import lapack
from threadpoolctl import threadpool_limits

from diive.core.utils.console import info, warn, success
from diive.flux.partitioning._report import partitioning_report
# R's own arithmetic (LINPACK QR, R's sum/mean/quantile, lm), ported and checked
# bitwise against R 4.5.3 for the nighttime REddyProc port; the daytime method
# runs the same R functions.
from diive.flux.partitioning.nighttime_reddyproc import (
    potential_radiation, _infer_dts, T0_K, _dqrdc2, _qr_qty, _qr_coef,
    _r_sum, _r_mean, _r_quantile7, _lm_through_origin)

# Reference temperature for RRef/Lloyd-Taylor: 273.15 + 15 degC.
TREF_K = 273.15 + 15.0
VPD0 = 10.0  # hPa, Lasslop et al. 2010

# Window geometry (REddyProc applyWindows defaults).
WIN_REF_DAYS = 4       # reference window (and LRC window) width in days
STRIDE_DAYS = 2        # window step in days
WIN_NIGHT_DAYS = 12    # nighttime E0 window width in days
WIN_EXTEND = (24, 48)  # successively extended night windows for failed fits
MIN_NREC = 10          # minNRecInDayWindow
E0_MIN, E0_MAX = 50.0, 400.0
DAY_MAX_SW_IN = 4.0    # Rg <= this is a necessary night condition (Lasslop)

# Replace missing NEE uncertainty: max(minSd, perc*|NEE|).
SD_PERC, SD_MINSD = 0.2, 0.7

# Lasslop fixed prior standard deviations (k, beta, alpha, RRef, E0).
LASSLOP_SDPRIOR = np.array([50.0, 600.0, 10.0, 80.0, np.nan])

# R nls.control / optim constants.
_NLS_EPS = np.sqrt(np.finfo(float).eps)  # numericDeriv forward step (1.49e-8)
_NLS_TOL = 1e-5
_NLS_MAXITER = 20
_NLS_MINFAC = 1.0 / 1024
_OPTIM_NDEPS = 1e-3        # optim numeric-gradient step
_OPTIM_RELTOL = 1e-3       # LRCFitConvergenceTolerance
_OPTIM_MAXIT = 100
_VMMIN_STEPREDN = 0.2
_VMMIN_ACCTOL = 1e-4
_VMMIN_RELTEST = 10.0


# --------------------------------------------------------------------------- #
# Lloyd & Taylor respiration (Kelvin, custom reference temperature)
# --------------------------------------------------------------------------- #
def _lloyd_taylor(rref, e0, ta_k, tref_k=TREF_K):
    return rref * _exp_r(e0 * (1.0 / (tref_k - T0_K) - 1.0 / (ta_k - T0_K)))


# --------------------------------------------------------------------------- #
# R's nls default Gauss-Newton (for the nighttime E0 fit)
# --------------------------------------------------------------------------- #
# The per-window E0 fit is chaotic in the last bit: numericDeriv's forward
# difference (step sqrt(eps) * |p|) turns a one-ulp change of a parameter or a
# model value into a ~1e-8 relative change of the Jacobian, and the next
# Gauss-Newton step carries it into E0. numpy's QR, sums, mean and exp moved
# E0 by up to 4e-6 relative. So everything below is R's arithmetic: the
# start value is R's mean, the QR is LINPACK's dqrdc2/dqrsl, the sums are R's
# long-double sums, the SD is summary.nls's chol2inv, and the model's exp is
# rounded as R's (_exp_r).
def _numeric_deriv(predict, par, rhs):
    """Forward-difference Jacobian, R's numericDeriv (dir=1, eps=sqrt(macheps))."""
    grad = np.empty((rhs.size, par.size))
    for j in range(par.size):
        xx = abs(par[j])
        delta = _NLS_EPS if xx == 0 else xx * _NLS_EPS
        p2 = par.copy()
        p2[j] = par[j] + delta
        grad[:, j] = (predict(p2) - rhs) / delta
    return grad


def _nls_state(y, predict, par):
    """nlsModel's setPars: model, Jacobian, residuals, deviance and QR at ``par``.

    ``None`` where R's numericDeriv stops with "Missing value or an infinity
    produced when evaluating the model" (a diverging step overflows exp)."""
    with np.errstate(over='ignore', invalid='ignore'):
        rhs = predict(par)
        if not np.isfinite(rhs).all():
            return None
        grad = _numeric_deriv(predict, par, rhs)
        if not np.isfinite(grad).all():
            return None
        resid = y - rhs
        dev = _r_sum(resid * resid)  # inf for a wild step: rejected, as in R
    qr, qraux, rank = _dqrdc2(grad)
    return resid, dev, qr, qraux, rank


def _r_nls(y, predict, start):
    """Port of R's ``nls`` (default Gauss-Newton), ``nls.control(maxiter = 20)``.

    As REddyProc's ``partGLEstimateTempSensInBoundsE0Only`` calls it: C
    ``nls_iter`` with increments ``qr.coef(QR, resid)``, the relative-offset
    criterion on ``qr.qty(QR, resid)`` (tol 1e-5), step halving down to
    ``minFactor = 1/1024``, and nls's rank test (LINPACK ``dqrdc2``, tol 1e-7)
    at every trial point. For two parameters.

    Returns ``(par, cov)`` with ``cov = chol2inv(R) * deviance / (n - 2)``
    (``summary.nls``), or ``(None, None)`` wherever R's ``nls()`` stops with an
    error (singular gradient, step factor below minFactor, iteration limit,
    non-finite model values); REddyProc then treats the fit as NA.
    """
    npar = start.size
    par = start.astype(float).copy()
    state = _nls_state(y, predict, par)
    if state is None or state[4] < npar:
        return None, None
    resid, dev, qr, qraux, rank = state
    fac = 1.0
    converged = False
    for _ in range(_NLS_MAXITER):
        rr = _qr_qty(qr, qraux, rank, resid)
        with np.errstate(divide='ignore', invalid='ignore'):
            conv = np.sqrt(np.float64(_r_sum(rr[:npar] * rr[:npar]))
                           / np.float64(0.0 + _r_sum(rr[npar:] * rr[npar:])))
        if conv <= _NLS_TOL:
            converged = True
            break
        incr = _qr_coef(qr, qraux, rank, resid)
        if incr is None:
            return None, None
        while fac >= _NLS_MINFAC:
            new_par = par + fac * incr
            state = _nls_state(y, predict, new_par)
            # nls tests the rank at every trial point, also at one it rejects.
            if state is None or state[4] < npar:
                return None, None
            if state[1] <= dev:
                par = new_par
                resid, dev, qr, qraux, rank = state
                fac = min(2.0 * fac, 1.0)
                break
            fac /= 2.0
        if fac < _NLS_MINFAC:
            return None, None
    if not converged:
        return None, None
    # summary.nls: chol2inv(qr.R(QR)) through LAPACK dpotri (dtrti2 + dlauu2),
    # written out for the 2 x 2 triangle so the rounding is LAPACK's.
    ia = 1.0 / qr[0, 0]
    ic = 1.0 / qr[1, 1]
    u12 = (ia * qr[0, 1]) * (-ic)
    xtx_inv = np.array([[ia * ia + u12 * u12, ic * u12], [ic * u12, ic * ic]])
    return par, xtx_inv * (dev / (y.size - npar))


# --------------------------------------------------------------------------- #
# R's optim BFGS (vmmin) + numeric gradient/Hessian (for the LRC fit)
# --------------------------------------------------------------------------- #
def _fmingr(fn, p):
    """Central-difference gradient, R optim default (ndeps=1e-3, parscale=1)."""
    g = np.empty_like(p)
    for i in range(p.size):
        pp = p.copy()
        pp[i] = p[i] + _OPTIM_NDEPS
        v1 = fn(pp)
        pp[i] = p[i] - _OPTIM_NDEPS
        v2 = fn(pp)
        g[i] = (v1 - v2) / (2 * _OPTIM_NDEPS)
    return g


def _vmmin(b0, fn):
    """Port of R's vmmin BFGS (src/appl/optim.c), reltol=1e-3.

    Loop for loop as in C: B is kept as its lower triangle and every product
    and sum runs in C's order. Matrix products (``B @ g``, outer products)
    round differently, and the line search and the ``reltol`` stop turn such
    last-bit differences into different iterates in ill-conditioned windows.
    """
    n = b0.size
    b = b0.astype(float).copy()
    f = fn(b)
    Fmin = f
    g = _fmingr(fn, b)
    iter_ = 1
    gradcount = 1
    ilast = gradcount
    B = [[0.0] * n for _ in range(n)]
    t = [0.0] * n
    X = [0.0] * n
    c = [0.0] * n
    count = 0
    while True:
        if ilast == gradcount:
            for i in range(n):
                for j in range(i):
                    B[i][j] = 0.0
                B[i][i] = 1.0
        for i in range(n):
            X[i] = float(b[i])
            c[i] = float(g[i])
        gradproj = 0.0
        for i in range(n):
            s = 0.0
            for j in range(i + 1):
                s -= B[i][j] * g[j]
            for j in range(i + 1, n):
                s -= B[j][i] * g[j]
            t[i] = s
            gradproj += s * g[i]
        if gradproj < 0.0:
            steplength = 1.0
            accpoint = False
            while True:
                count = 0
                for i in range(n):
                    b[i] = X[i] + steplength * t[i]
                    if _VMMIN_RELTEST + X[i] == _VMMIN_RELTEST + b[i]:
                        count += 1
                if count < n:
                    f = fn(b)
                    accpoint = math.isfinite(f) and (
                        f <= Fmin + gradproj * steplength * _VMMIN_ACCTOL)
                    if not accpoint:
                        steplength *= _VMMIN_STEPREDN
                if count == n or accpoint:
                    break
            enough = (f > -np.inf) and abs(f - Fmin) > _OPTIM_RELTOL * (abs(Fmin) + _OPTIM_RELTOL)
            if not enough:
                count = n
                Fmin = f
            if count < n:
                Fmin = f
                g = _fmingr(fn, b)
                gradcount += 1
                iter_ += 1
                D1 = 0.0
                for i in range(n):
                    t[i] = steplength * t[i]
                    c[i] = g[i] - c[i]
                    D1 += t[i] * c[i]
                if D1 > 0:
                    D2 = 0.0
                    for i in range(n):
                        s = 0.0
                        for j in range(i + 1):
                            s += B[i][j] * c[j]
                        for j in range(i + 1, n):
                            s += B[j][i] * c[j]
                        X[i] = s
                        D2 += s * c[i]
                    D2 = 1.0 + D2 / D1
                    for i in range(n):
                        for j in range(i + 1):
                            B[i][j] += (D2 * t[i] * t[j] - X[i] * t[j] - t[i] * X[j]) / D1
                else:
                    ilast = gradcount
            else:
                if ilast < gradcount:
                    count = 0
                    ilast = gradcount
        else:
            count = 0
            if ilast == gradcount:
                count = n
            else:
                ilast = gradcount
        if iter_ >= _OPTIM_MAXIT:
            break
        if gradcount - ilast > 2 * n:
            ilast = gradcount
        if count == n and ilast == gradcount:
            break
    fail = 0 if iter_ < _OPTIM_MAXIT else 1
    return b, Fmin, fail


def _optim_hess(par, cost):
    """Port of R's optimHess: central diff of the (central-diff) gradient."""
    nn = par.size
    H = np.zeros((nn, nn))
    for i in range(nn):
        dp = par.copy()
        dp[i] += _OPTIM_NDEPS
        df1 = _fmingr(cost, dp)
        dp[i] -= 2 * _OPTIM_NDEPS
        df2 = _fmingr(cost, dp)
        H[:, i] = (df1 - df2) / (2 * _OPTIM_NDEPS)
    return 0.5 * (H + H.T)


# --------------------------------------------------------------------------- #
# Window geometry
# --------------------------------------------------------------------------- #
def _window_grid(n, dts):
    """Reference-window centers (applyWindows, winSizeRef=4, stride=2)."""
    n_day = int(np.ceil(n / dts))
    n_day_last = n_day - WIN_REF_DAYS / 2
    start_days = np.arange(1, n_day_last + 1e-9, STRIDE_DAYS).astype(int)
    i_central = 1 + ((start_days - 1) + WIN_REF_DAYS // 2) * dts
    return start_days, i_central


def _win_recs(i_central, win_days, dts, n):
    """1-based [iRecStart, iRecEnd] per window for a window size in days."""
    half = win_days / 2 * dts
    rec_start = np.maximum(1, (i_central - half).astype(int))
    rec_end = np.minimum(n, (i_central - 1 + half).astype(int))
    return rec_start, rec_end


# --------------------------------------------------------------------------- #
# Stage 2: nighttime temperature sensitivity (E0) + RRef
# --------------------------------------------------------------------------- #
def _is_valid_night(nee_w, temp_w, isnight_w):
    v = isnight_w & ~np.isnan(nee_w) & np.isfinite(temp_w)
    freezing = temp_w[v] <= -1
    if np.sum(~freezing) >= 12:
        vi = np.nonzero(v)[0]
        v[vi[freezing]] = False
    return v


def _fit_e0_window(reco, temp_k, prev_e0, tref_k):
    """Port of partGLEstimateTempSensInBoundsE0Only (R nls), bounded [50, 400]."""
    b = 1.0 / (tref_k - T0_K) - 1.0 / (temp_k - T0_K)
    start_e0 = prev_e0 if np.isfinite(prev_e0) else 100.0
    start_rref = _r_mean(reco)  # R's mean, not numpy's: the fit is chaotic in it

    def predict(p):
        return p[0] * _exp_r(p[1] * b)

    par, cov = _r_nls(reco, predict, np.array([start_rref, start_e0]))
    if par is None:
        return np.nan, np.nan, tref_k, np.nan
    rref, e0 = float(par[0]), float(par[1])
    sd_e0 = float(np.sqrt(abs(cov[1, 1])))
    if not np.isfinite(e0) or e0 < E0_MIN or e0 > E0_MAX:
        return np.nan, np.nan, tref_k, np.nan
    return e0, sd_e0, tref_k, rref


def _fit_nighttime_pass(nee, temp, is_night, i_central, win_days, dts, n):
    """One applyWindows pass: per-window E0 nls with sequential prevE0."""
    rec_start, rec_end = _win_recs(i_central, win_days, dts, n)
    nw = i_central.size
    e0 = np.full(nw, np.nan)
    sde0 = np.full(nw, np.nan)
    treffit = np.full(nw, np.nan)
    rreffit = np.full(nw, np.nan)
    prev_e0 = np.nan
    for w in range(nw):
        lo, hi = rec_start[w] - 1, rec_end[w]
        nee_w, temp_w, isn_w = nee[lo:hi], temp[lo:hi], is_night[lo:hi]
        v = _is_valid_night(nee_w, temp_w, isn_w)
        if v.sum() < MIN_NREC:
            prev_e0 = np.nan
            continue
        reco = nee_w[v]
        temp_k = temp_w[v] + 273.15
        tref_k = float(np.median(temp_w[v])) + 273.15
        e0w, sdw, trf, rrf = _fit_e0_window(reco, temp_k, prev_e0, tref_k)
        e0[w], sde0[w], treffit[w], rreffit[w] = e0w, sdw, trf, rrf
        prev_e0 = e0w
    return e0, sde0, treffit, rreffit


# --------------------------------------------------------------------------- #
# mlegp 3.1.9 (fitGP in src/fit_gp.h, createGP/predict.gp in R), as
# partGLSmoothTempSens calls it: mlegp(X = iCentralRec, Z = E0, nugget = sdE0^2)
# --------------------------------------------------------------------------- #
# mlegp maximises the likelihood over (log beta, log nugget scale) with 5
# Nelder-Mead runs from random starts, then L-BFGS from the best. The optimum
# is flat: other optimizers (or other starts) stop at a different point with
# the same likelihood to 1e-8, and the smoothed E0 then differs by ~1e-4. What
# mlegp returns is where *its* simplex stops, so the port reproduces the run
# itself: the SFMT random numbers of seed 0, the start values, R's 2.4.0
# nmmin and liblbfgs 1.x step by step. The simplex vertices are pure
# arithmetic on the start values; the likelihood only decides which vertex
# moves. So as long as those comparisons agree (the likelihood values here
# agree with mlegp's to ~1e-15), the result is mlegp's to the bit. In all 8
# CH-DAV parity runs the L-BFGS stage ended where it started: its gradient
# is a forward difference with h = 1e-10 on the natural scale, used as if it
# were the gradient in log space, so the line search fails and mlegp keeps the
# simplex result. It is ported anyway, but it is not reproducible to the bit if
# it ever moves: the h = 1e-10 difference amplifies last-bit differences of the
# likelihood by 1e10.
_MLEGP_LN2 = 0.693147180559945309417
# gp.h's fallback, used because math.h has no M_LNPI; it only shifts the
# likelihood, but it changes its rounding and so the comparisons above.
_MLEGP_LNPI = 1.1447298858494
_MLEGP_SIMPLEX_TRIES = 5
_MLEGP_SIMPLEX_MAXIT = 500
_MLEGP_SIMPLEX_RELTOL = 1e-8
_MLEGP_BFGS_MAXIT = 500
_MLEGP_BFGS_TOL = 0.01
_MLEGP_BFGS_H = 1e-10
_MLEGP_SEED = 0
_DBL_MAX = float(np.finfo(float).max)

# SFMT 1.3 with MEXP = 607 (mlegp's Makevars), the generator behind
# genrand_res53() in fitGP.
_SFMT_N = 5
_SFMT_POS1, _SFMT_SL1, _SFMT_SL2, _SFMT_SR1, _SFMT_SR2 = 2, 15, 3, 13, 3
_SFMT_MSK = (0xfdff37ff, 0xef7f3f7d, 0xff777b7d, 0x7ff7fb2f)
_SFMT_PARITY = (0x00000001, 0x00000000, 0x00000000, 0x5986f054)


def _sfmt607_res53(seed, count):
    """``count`` draws of SFMT-607 ``genrand_res53()`` after ``init_gen_rand(seed)``.

    mlegp seeds its own generator (not R's RNG) with ``seed = 0`` on every
    call, so the five start values are the same in every fit.
    """
    m32 = 0xffffffff
    m128 = (1 << 128) - 1
    st = [0] * (4 * _SFMT_N)
    st[0] = seed & m32
    for i in range(1, 4 * _SFMT_N):
        st[i] = (1812433253 * (st[i - 1] ^ (st[i - 1] >> 30)) + i) & m32
    # period_certification()
    inner = 0
    for i in range(4):
        inner ^= st[i] & _SFMT_PARITY[i]
    for sh in (16, 8, 4, 2, 1):
        inner ^= inner >> sh
    if not inner & 1:
        for i in range(4):
            bit = next((1 << k for k in range(32) if (1 << k) & _SFMT_PARITY[i]), 0)
            if bit:
                st[i] ^= bit
                break
    # 128-bit words, little endian as on x86 (u[0] is the low word).
    w = [st[4 * i] | st[4 * i + 1] << 32 | st[4 * i + 2] << 64 | st[4 * i + 3] << 96
         for i in range(_SFMT_N)]

    def recursion(a, b, c, d):
        x = (a << (_SFMT_SL2 * 8)) & m128
        y = c >> (_SFMT_SR2 * 8)
        r = 0
        for k in range(4):
            sh = 32 * k
            rk = (((a >> sh) & m32) ^ ((x >> sh) & m32)
                  ^ ((((b >> sh) & m32) >> _SFMT_SR1) & _SFMT_MSK[k])
                  ^ ((y >> sh) & m32) ^ ((((d >> sh) & m32) << _SFMT_SL1) & m32))
            r |= rk << sh
        return r

    out = []
    idx = 4 * _SFMT_N
    while len(out) < count:
        if idx >= 4 * _SFMT_N:  # gen_rand_all()
            r1, r2 = w[_SFMT_N - 2], w[_SFMT_N - 1]
            for i in range(_SFMT_N):
                w[i] = recursion(w[i], w[(i + _SFMT_POS1) % _SFMT_N], r1, r2)
                r1, r2 = r2, w[i]
            idx = 0
        v = (w[idx // 4] >> (32 * (idx % 4))) & ((1 << 64) - 1)  # gen_rand64()
        idx += 2
        # to_res53(): v * 2^-64 in long double, rounded once to double.
        out.append(float(v) * 2.0 ** -64)
    return out


def _exp_c(x):
    """Scalar ``exp``, correctly rounded, which is what R's x87 ``exp`` returns
    except in rare near-ties (see :func:`_exp_r`). UCRT's ``exp`` (numpy,
    ``math``) is off by one ulp for ~0.5 % of arguments, and these scalars are
    the GP parameters themselves."""
    if x > 709.8:
        return math.inf
    if x < -745.2:
        return 0.0
    with localcontext() as ctx:
        ctx.prec = 40
        return float(Decimal(float(x)).exp())


def _log_c(x):
    """Scalar ``log``, correctly rounded (see :func:`_exp_c`)."""
    with localcontext() as ctx:
        ctx.prec = 40
        return float(Decimal(float(x)).ln())


_EXP_TABLES = None


def _exp_tables():
    """Reduction constants and 2^(j/256) as double-double pairs (computed once)."""
    global _EXP_TABLES
    if _EXP_TABLES is None:
        with localcontext() as ctx:
            ctx.prec = 60
            ln2_256 = Decimal(2).ln() / 256
            # l1 has 32 significant bits, so k * l1 is exact for |k| < 2^21
            l1 = math.ldexp(math.floor(math.ldexp(float(ln2_256), 40)), -40)
            rest = ln2_256 - Decimal(l1)
            l2 = float(rest)
            l3 = float(rest - Decimal(l2))
            t_hi = np.empty(256)
            t_lo = np.empty(256)
            for j in range(256):
                t = Decimal(2) ** (Decimal(j) / 256)
                t_hi[j] = float(t)
                t_lo[j] = float(t - Decimal(t_hi[j]))
        _EXP_TABLES = (1.0 / float(ln2_256), l1, l2, l3, t_hi, t_lo)
    return _EXP_TABLES


def _exp_r(x):
    """Vectorized ``exp`` rounded as R computes it on Windows.

    R computes ``exp`` in x87 extended precision and rounds the 64-bit result
    to double; within 2^-12 ulp of a midpoint that double rounding goes to the
    even neighbour. numpy's ``exp`` (UCRT) differs from R's in the last bit
    for ~0.5 % of arguments, the correctly rounded value for ~0.04 %, this
    function for ~0.003 % (where the x87 instruction's own error decides). The
    E0 fits amplify one such bit to ~1e-8 relative (see :func:`_r_nls`), the
    LRC fits to ~1e-6. Double-double evaluation (table of 2^(j/256), degree-6
    polynomial, error < 1e-22); about 20 times slower than ``np.exp``.
    """
    inv, l1, l2, l3, t_hi, t_lo = _exp_tables()
    x = np.asarray(x, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        k = np.rint(x * inv)
        k = np.where(np.isfinite(k) & (np.abs(x) <= 708.0), k, 0.0)
        a = x - k * l1  # exact (Sterbenz)
        b = -(k * l2)
        r_hi = a + b
        bb = r_hi - a
        r_lo = (a - (r_hi - bb)) + (b - bb) - k * l3
        p = (r_hi * r_hi) * (0.5 + r_hi * (1.0 / 6 + r_hi * (
            1.0 / 24 + r_hi * (1.0 / 120 + r_hi * (1.0 / 720)))))
        s_lo = r_lo + p + r_lo * r_hi
        ki = k.astype(np.int64)
        j = ki & 255
        th, tl = t_hi[j], t_lo[j]
        # exact product th * r_hi (Dekker, no FMA)
        c = 134217729.0 * th
        th_h = c - (c - th)
        th_l = th - th_h
        c = 134217729.0 * r_hi
        r_h = c - (c - r_hi)
        r_l = r_hi - r_h
        p_hi = th * r_hi
        p_lo = ((th_h * r_h - p_hi) + th_h * r_l + th_l * r_h) + th_l * r_l
        s_hi = th + p_hi
        bb = s_hi - th
        s_err = (th - (s_hi - bb)) + (p_hi - bb)
        low = s_err + p_lo + th * s_lo + tl + tl * r_hi
        y = s_hi + low        # correctly rounded
        d = (s_hi - y) + low  # exact value minus y
        # x87 double rounding: R's exp rounds to 64 bits, then to 53. Within
        # 2^-12 ulp of a midpoint the 64-bit value is the midpoint itself,
        # and the second rounding goes to the even neighbour.
        nb = np.where(d > 0, np.nextafter(y, np.inf), np.nextafter(y, -np.inf))
        odd = (y.view(np.int64) & 1) == 1
        tie = np.abs(d) >= np.abs(nb - y) * (0.5 - 2.0 ** -12)
        y = np.where(tie & odd, nb, y)
        res = np.ldexp(y, (ki - j) >> 8)
        # outside the reduced range (and for inf/nan) numpy's exp is exact enough
        return np.where(np.abs(x) <= 708.0, res, np.exp(x))


def _nelder_mead_min(bvec, fminfn, abstol, intol, maxit,
                     alpha=1.0, bet=0.5, gamm=2.0):
    """mlegp's ``nelder_mead_min`` (R 2.4.0 ``nmmin``), statement by statement.

    Returns ``(x, fmin, fail)``.
    """
    n = len(bvec)
    bvec = [float(v) for v in bvec]
    big = 1e+140
    P = [[0.0] * (n + 2) for _ in range(n + 1)]
    f = fminfn(bvec)
    if not math.isfinite(f):
        return list(bvec), f, 1
    funcount = 1
    convtol = intol * (abs(f) + intol)
    n1 = n + 1
    C = n + 2
    P[n1 - 1][0] = f
    for i in range(n):
        P[i][0] = bvec[i]
    L = 1
    size = 0.0
    step = 0.0
    for i in range(n):
        if 0.1 * abs(bvec[i]) > step:
            step = 0.1 * abs(bvec[i])
    if step == 0.0:
        step = 0.1
    for j in range(2, n1 + 1):
        for i in range(n):
            P[i][j - 1] = bvec[i]
        trystep = step
        while P[j - 2][j - 1] == bvec[j - 2]:
            P[j - 2][j - 1] = bvec[j - 2] + trystep
            trystep *= 10
        size += trystep
    oldsize = size
    calcvert = True
    fail = 0
    while True:
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    for i in range(n):
                        bvec[i] = P[i][j]
                    f = fminfn(bvec)
                    if not math.isfinite(f):
                        f = big
                    funcount += 1
                    P[n1 - 1][j] = f
            calcvert = False
        VL = P[n1 - 1][L - 1]
        VH = VL
        H = L
        for j in range(1, n1 + 1):
            if j != L:
                f = P[n1 - 1][j - 1]
                if f < VL:
                    L = j
                    VL = f
                if f > VH:
                    H = j
                    VH = f
        if VH <= VL + convtol or VL <= abstol:
            break
        for i in range(n):
            temp = -P[i][H - 1]
            for j in range(n1):
                temp += P[i][j]
            P[i][C - 1] = temp / n
        for i in range(n):
            bvec[i] = (1.0 + alpha) * P[i][C - 1] - alpha * P[i][H - 1]
        f = fminfn(bvec)
        if not math.isfinite(f):
            f = big
        funcount += 1
        VR = f
        if VR < VL:
            P[n1 - 1][C - 1] = f
            for i in range(n):
                f = gamm * bvec[i] + (1 - gamm) * P[i][C - 1]
                P[i][C - 1] = bvec[i]
                bvec[i] = f
            f = fminfn(bvec)
            if not math.isfinite(f):
                f = big
            funcount += 1
            if f < VR:
                for i in range(n):
                    P[i][H - 1] = bvec[i]
                P[n1 - 1][H - 1] = f
            else:
                for i in range(n):
                    P[i][H - 1] = P[i][C - 1]
                P[n1 - 1][H - 1] = VR
        else:
            if VR < VH:
                for i in range(n):
                    P[i][H - 1] = bvec[i]
                P[n1 - 1][H - 1] = VR
            for i in range(n):
                bvec[i] = (1 - bet) * P[i][H - 1] + bet * P[i][C - 1]
            f = fminfn(bvec)
            if not math.isfinite(f):
                f = big
            funcount += 1
            if f < P[n1 - 1][H - 1]:
                for i in range(n):
                    P[i][H - 1] = bvec[i]
                P[n1 - 1][H - 1] = f
            elif VR >= VH:  # shrink towards the lowest vertex
                calcvert = True
                size = 0.0
                for j in range(n1):
                    if j + 1 != L:
                        for i in range(n):
                            P[i][j] = bet * (P[i][j] - P[i][L - 1]) + P[i][L - 1]
                            size += abs(P[i][j] - P[i][L - 1])
                if size < oldsize:
                    oldsize = size
                else:
                    fail = 10
                    break
        if funcount > maxit:
            break
    if funcount > maxit:
        fail = 1
    return [P[i][L - 1] for i in range(n)], P[n1 - 1][L - 1], fail


# liblbfgs 1.x as bundled with mlegp (lbfgs.c), defaults but epsilon and
# max_iterations. Error codes as in lbfgs.h.
_LBFGS_M, _LBFGS_MAX_LS = 6, 20
_LBFGS_MIN_STEP, _LBFGS_MAX_STEP = 1e-20, 1e20
_LBFGS_FTOL, _LBFGS_GTOL, _LBFGS_XTOL = 1e-4, 0.9, 1e-16
_LBFGSERR_OUTOFINTERVAL, _LBFGSERR_INCORRECT_TMINMAX = -1011, -1010
_LBFGSERR_ROUNDING_ERROR, _LBFGSERR_MINIMUMSTEP = -1009, -1008
_LBFGSERR_MAXIMUMSTEP, _LBFGSERR_MAXIMUMLINESEARCH = -1007, -1006
_LBFGSERR_MAXIMUMITERATION, _LBFGSERR_WIDTHTOOSMALL = -1005, -1004
_LBFGSERR_INCREASEGRADIENT = -1002


def _vecdot(x, y):
    s = 0.0
    for a, b in zip(x, y, strict=True):
        s += a * b
    return s


def _max2(a, b):
    return a if a >= b else b


def _min2(a, b):
    return a if a <= b else b


def _cubic_minimizer(u, fu, du, v, fv, dv):
    d = v - u
    theta = (fu - fv) * 3 / d + du + dv
    s = _max2(_max2(abs(theta), abs(du)), abs(dv))
    a = theta / s
    gamma = s * math.sqrt(a * a - (du / s) * (dv / s))
    if v < u:
        gamma = -gamma
    p = gamma - du + theta
    q = gamma - du + gamma + dv
    return u + (p / q) * d


def _cubic_minimizer2(u, fu, du, v, fv, dv, xmin, xmax):
    d = v - u
    theta = (fu - fv) * 3 / d + du + dv
    s = _max2(_max2(abs(theta), abs(du)), abs(dv))
    a = theta / s
    gamma = s * math.sqrt(_max2(0.0, a * a - (du / s) * (dv / s)))
    if u < v:
        gamma = -gamma
    p = gamma - dv + theta
    q = gamma - dv + gamma + du
    r = p / q
    if r < 0. and gamma != 0.:
        return v - r * d
    return xmax if a < 0 else xmin


def _update_trial_interval(st, t, ft, dt, tmin, tmax):
    """liblbfgs ``update_trial_interval`` (More-Thuente); updates ``st`` in place
    and returns ``(new_t, info)``."""
    x, fx, dx, y, fy, dy = st['x'], st['fx'], st['dx'], st['y'], st['fy'], st['dy']
    dsign = dt * (dx / abs(dx)) < 0.
    if st['brackt']:
        if t <= _min2(x, y) or _max2(x, y) <= t:
            return t, _LBFGSERR_OUTOFINTERVAL
        if 0. <= dx * (t - x):
            return t, _LBFGSERR_INCREASEGRADIENT
        if tmax < tmin:
            return t, _LBFGSERR_INCORRECT_TMINMAX
    if fx < ft:
        st['brackt'] = True
        bound = True
        mc = _cubic_minimizer(x, fx, dx, t, ft, dt)
        mq = x + dx / ((fx - ft) / (t - x) + dx) / 2 * (t - x)
        newt = mc if abs(mc - x) < abs(mq - x) else mc + 0.5 * (mq - mc)
    elif dsign:
        st['brackt'] = True
        bound = False
        mc = _cubic_minimizer(x, fx, dx, t, ft, dt)
        mq = t + dt / (dt - dx) * (x - t)
        newt = mc if abs(mc - t) > abs(mq - t) else mq
    elif abs(dt) < abs(dx):
        bound = True
        mc = _cubic_minimizer2(x, fx, dx, t, ft, dt, tmin, tmax)
        mq = t + dt / (dt - dx) * (x - t)
        if st['brackt']:
            newt = mc if abs(t - mc) < abs(t - mq) else mq
        else:
            newt = mc if abs(t - mc) > abs(t - mq) else mq
    else:
        bound = False
        if st['brackt']:
            newt = _cubic_minimizer(t, ft, dt, y, fy, dy)
        elif x < t:
            newt = tmax
        else:
            newt = tmin
    if fx < ft:
        st['y'], st['fy'], st['dy'] = t, ft, dt
    else:
        if dsign:
            st['y'], st['fy'], st['dy'] = x, fx, dx
        st['x'], st['fx'], st['dx'] = t, ft, dt
    if tmax < newt:
        newt = tmax
    if newt < tmin:
        newt = tmin
    if st['brackt'] and bound:
        mq = st['x'] + 0.66 * (st['y'] - st['x'])
        if st['x'] < st['y']:
            if mq < newt:
                newt = mq
        elif newt < mq:
            newt = mq
    return newt, 0


def _lbfgs_line_search(x, f, g, s, stp, evaluate):
    """liblbfgs ``line_search`` (More-Thuente). Returns ``(ls, x, f, g, stp)``;
    ``ls < 0`` is an error code, and then ``x`` is the last point tried."""
    count = 0
    uinfo = 0
    dginit = _vecdot(g, s)
    if 0 < dginit:
        return _LBFGSERR_INCREASEGRADIENT, x, f, g, stp
    stage1 = True
    finit = f
    dgtest = _LBFGS_FTOL * dginit
    width = _LBFGS_MAX_STEP - _LBFGS_MIN_STEP
    prev_width = 2.0 * width
    wa = list(x)
    st = dict(x=0., fx=finit, dx=dginit, y=0., fy=finit, dy=dginit, brackt=False)
    while True:
        if st['brackt']:
            stmin, stmax = _min2(st['x'], st['y']), _max2(st['x'], st['y'])
        else:
            stmin, stmax = st['x'], stp + 4.0 * (stp - st['x'])
        if stp < _LBFGS_MIN_STEP:
            stp = _LBFGS_MIN_STEP
        if _LBFGS_MAX_STEP < stp:
            stp = _LBFGS_MAX_STEP
        br = st['brackt']
        if (br and ((stp <= stmin or stmax <= stp) or _LBFGS_MAX_LS <= count + 1
                    or uinfo != 0)) or (br and (stmax - stmin <= _LBFGS_XTOL * stmax)):
            stp = st['x']
        x = [a + stp * b for a, b in zip(wa, s, strict=True)]
        f, g = evaluate(x)
        count += 1
        dg = _vecdot(g, s)
        ftest1 = finit + stp * dgtest
        if br and ((stp <= stmin or stmax <= stp) or uinfo != 0):
            return _LBFGSERR_ROUNDING_ERROR, x, f, g, stp
        if stp == _LBFGS_MAX_STEP and f <= ftest1 and dg <= dgtest:
            return _LBFGSERR_MAXIMUMSTEP, x, f, g, stp
        if stp == _LBFGS_MIN_STEP and (ftest1 < f or dgtest <= dg):
            return _LBFGSERR_MINIMUMSTEP, x, f, g, stp
        if br and (stmax - stmin) <= _LBFGS_XTOL * stmax:
            return _LBFGSERR_WIDTHTOOSMALL, x, f, g, stp
        if _LBFGS_MAX_LS <= count:
            return _LBFGSERR_MAXIMUMLINESEARCH, x, f, g, stp
        if f <= ftest1 and abs(dg) <= _LBFGS_GTOL * (-dginit):
            return count, x, f, g, stp
        if stage1 and f <= ftest1 and _min2(_LBFGS_FTOL, _LBFGS_GTOL) * dginit <= dg:
            stage1 = False
        if stage1 and ftest1 < f and f <= st['fx']:
            # modified function (psi) until sufficient decrease is reached
            sm = dict(x=st['x'], fx=st['fx'] - st['x'] * dgtest, dx=st['dx'] - dgtest,
                      y=st['y'], fy=st['fy'] - st['y'] * dgtest, dy=st['dy'] - dgtest,
                      brackt=st['brackt'])
            stp, uinfo = _update_trial_interval(sm, stp, f - stp * dgtest, dg - dgtest,
                                                stmin, stmax)
            st = dict(x=sm['x'], fx=sm['fx'] + sm['x'] * dgtest, dx=sm['dx'] + dgtest,
                      y=sm['y'], fy=sm['fy'] + sm['y'] * dgtest, dy=sm['dy'] + dgtest,
                      brackt=sm['brackt'])
        else:
            stp, uinfo = _update_trial_interval(st, stp, f, dg, stmin, stmax)
        if st['brackt']:
            if 0.66 * prev_width <= abs(st['y'] - st['x']):
                stp = st['x'] + 0.5 * (st['y'] - st['x'])
            prev_width = width
            width = abs(st['y'] - st['x'])


def _lbfgs(x, evaluate, epsilon, max_iterations):
    """liblbfgs ``lbfgs()`` (m = 6, More-Thuente). Returns ``(ret, x)``; on an
    error ``x`` is the last point the line search tried, as in C."""
    m = _LBFGS_M
    x = list(x)
    fx, g = evaluate(x)
    d = [-a for a in g]
    step = 1.0 / math.sqrt(_vecdot(d, d))
    k, end = 1, 0
    lm_s, lm_y = [None] * m, [None] * m
    lm_ys, lm_alpha = [0.0] * m, [0.0] * m
    while True:
        xp, gp = list(x), list(g)
        ls, x, fx, g, step = _lbfgs_line_search(x, fx, g, d, step, evaluate)
        if ls < 0:
            return ls, x
        gnorm = math.sqrt(_vecdot(g, g))
        xnorm = math.sqrt(_vecdot(x, x))
        if xnorm < 1.0:
            xnorm = 1.0
        if gnorm / xnorm <= epsilon:
            return 0, x
        if max_iterations != 0 and max_iterations < k + 1:
            return _LBFGSERR_MAXIMUMITERATION, x
        lm_s[end] = [a - b for a, b in zip(x, xp, strict=True)]
        lm_y[end] = [a - b for a, b in zip(g, gp, strict=True)]
        ys = _vecdot(lm_y[end], lm_s[end])
        yy = _vecdot(lm_y[end], lm_y[end])
        lm_ys[end] = ys
        bound = m if m <= k else k
        k += 1
        end = (end + 1) % m
        d = [-a for a in g]
        j = end
        for _ in range(bound):
            j = (j + m - 1) % m
            lm_alpha[j] = _vecdot(lm_s[j], d) / lm_ys[j]
            c = -lm_alpha[j]
            d = [a + c * b for a, b in zip(d, lm_y[j], strict=True)]
        c = ys / yy
        d = [a * c for a in d]
        for _ in range(bound):
            beta = _vecdot(lm_y[j], d) / lm_ys[j]
            c = lm_alpha[j] - beta
            d = [a + c * b for a, b in zip(d, lm_s[j], strict=True)]
            j = (j + 1) % m
        step = 1.0


class _MlegpLikelihood:
    """fitGP's ``f_min``: negative log likelihood of (log beta, log nugget
    scale), with the constant mean and the GP variance ``sig2`` profiled out.

    The correlation is ``exp(-beta * d^2)`` rounded as C does,
    ``exp((-beta * d) * d)``. The nugget is ``scale * sdE0^2`` on the diagonal.
    """

    def __init__(self, x, z, nug):
        self.x = np.asarray(x, float)
        self.z = np.asarray(z, float)
        self.nug = np.asarray(nug, float)
        self.n = self.z.size
        self.d = self.x[:, None] - self.x[None, :]
        self.diag = np.diag_indices(self.n)

    def corr(self, beta, nscale, exp=np.exp):
        # np.exp while optimizing (only comparisons of the likelihood matter),
        # R's rounding for the final estimates
        corr = exp(((-beta) * self.d) * self.d)
        corr[self.diag] += nscale * self.nug
        return corr

    def gls(self, corr):
        """Constant GLS mean ``bhat`` and ``sig2`` (calcBhat, calcMLESig2)."""
        ainv = lapack.dpotri(lapack.dpotrf(corr, lower=1)[0], lower=1)[0]
        ainv = np.tril(ainv) + np.tril(ainv, -1).T
        one_ainv = ainv.sum(axis=0)
        bhat = (1.0 / one_ainv.sum()) * float(one_ainv @ self.z)
        r = self.z - bhat
        return bhat, float(r @ ainv @ r) / self.n

    def __call__(self, v):
        beta = 0.0 if v[0] < -500 else _exp_c(v[0])
        nscale = 0.0 if v[1] < -500 else _exp_c(v[1])
        corr = self.corr(beta, nscale)
        chol, info = lapack.dpotrf(corr, lower=1)
        if info != 0:
            return _DBL_MAX
        bhat, sig2 = self.gls(corr)
        chol_v, info = lapack.dpotrf(corr * sig2, lower=1)
        if info != 0:
            return _DBL_MAX
        logdet = 2.0 * float(np.log(np.diag(chol_v)).sum())
        r = self.z - bhat
        vinv = lapack.dpotri(chol_v, lower=1)[0]
        vinv = np.tril(vinv) + np.tril(vinv, -1).T
        dd = float(r @ vinv @ r)
        return -(-(self.n / 2.0) * (_MLEGP_LN2 + _MLEGP_LNPI) - 0.5 * (logdet + dd))

    def fdf(self, v):
        """fdf_evaluate: value and forward-difference 'gradient' (step h on the
        natural scale, returned as if it were the gradient in log space)."""
        fv = self(v)
        vc = [0.0 if a < -500 else _exp_c(a) for a in v]
        g = []
        for i in range(len(v)):
            vp = list(vc)
            vp[i] = vc[i] + _MLEGP_BFGS_H
            fp = self([_log_c(a) for a in vp])
            if fv == _DBL_MAX:
                g.append(0.0)
            elif fp == _DBL_MAX:
                vp = list(vc)
                vp[i] = vc[i] - _MLEGP_BFGS_H
                fp = self([_log_c(a) for a in vp])
                g.append(0.0 if fp == _DBL_MAX else (fv - fp) / -_MLEGP_BFGS_H)
            else:
                g.append((fp - fv) / _MLEGP_BFGS_H)
        return fv, g


def _mlegp_fit(x, z, nug):
    """mlegp's fitGP for one input column, a constant mean and a nugget matrix
    ``nug`` (estimated scale). Returns ``(beta, mu, sig2, nugget_scale)``, the
    mlegp estimates; mlegp reports the nugget as ``nug * nugget_scale * sig2``.
    """
    lik = _MlegpLikelihood(x, z, nug)
    n = lik.n
    # vectorVariance(): C loops, mean then sum of squares, / (n - 1)
    mean = 0.0
    for a in lik.z:
        mean += a
    mean = mean / n
    sse = 0.0
    for a in lik.z:
        sse += (a - mean) * (a - mean)
    init_nugget = 1.0 / (sse / (n - 1))
    # getUnivariateCorRange(): starting beta between the values that give the
    # two closest design points a correlation of 0.65 and of 0.3
    d2 = (lik.x[:, None] - lik.x[None, :]) ** 2
    xmin = float(d2[d2 > 0].min())
    m1 = -_log_c(.65) / xmin
    m2 = -_log_c(.3) / xmin
    draws = _sfmt607_res53(_MLEGP_SEED, _MLEGP_SIMPLEX_TRIES)
    best_v, best_f = None, _DBL_MAX
    for t in range(_MLEGP_SIMPLEX_TRIES):
        v0 = [_log_c(m1 + (m2 - m1) * draws[t]), _log_c(init_nugget)]
        v, fval, _fail = _nelder_mead_min(v0, lik, -_DBL_MAX, _MLEGP_SIMPLEX_RELTOL,
                                          _MLEGP_SIMPLEX_MAXIT)
        if t == 0 or fval < best_f:
            best_v, best_f = v, fval
    _ret, v = _lbfgs(best_v, lik.fdf, _MLEGP_BFGS_TOL, _MLEGP_BFGS_MAXIT)
    fval = lik(v)
    if math.isnan(fval) or fval == _DBL_MAX:
        v = best_v  # L-BFGS failed: fitGP falls back to the simplex estimate
    beta = 0.0 if v[0] < -500 else _exp_c(v[0])
    nscale = 0.0 if v[1] < -500 else _exp_c(v[1])
    mu, sig2 = lik.gls(lik.corr(beta, nscale, _exp_r))
    return beta, mu, sig2, nscale


def _gp_smooth(x, z, nug):
    """mlegp GP fit + ``predict.gp(se.fit = TRUE)``. Returns ``(predict,
    nugget_vec)``, where ``nugget_vec`` is mlegp's ``gpFit$nugget``: the
    absolute nugget variance ``sdE0^2 * nugget_scale * sig2``."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    nug = np.asarray(nug, float)
    beta, mu, sig2, nscale = _mlegp_fit(x, z, nug)
    # mlegp2(): nugget matrix times the reported (scale * sig2); createGP()
    # then inverts sig2 * K + diag(nugget), K from calcVarMatrix, which rounds
    # the exponent as exp(-(beta * d^2)), unlike fitGP.
    nugget_vec = nug * (nscale * sig2)
    A = sig2 * _exp_r(-(beta * (x[:, None] - x[None, :]) ** 2))
    A[np.diag_indices(x.size)] += nugget_vec
    inv_var = np.linalg.inv(A)
    zc = z - mu

    def predict(xnew):
        xnew = np.atleast_1d(np.asarray(xnew, float))
        r = _exp_r(-(beta * (x[None, :] - xnew[:, None]) ** 2))
        r_inv = r @ inv_var
        fit = mu + sig2 * (r_inv @ zc)
        # calcPredictionError(): sig2 + 0 - sig2 * (r V^-1 r') * sig2, 0 if < 0
        v = sig2 - (sig2 * np.einsum('ij,ij->i', r_inv, r)) * sig2
        return fit, np.sqrt(np.where(v < 0, 0.0, v))

    return predict, nugget_vec


def _smooth_tempsens(e0fit, sde0fit, icentral, daystart):
    """Port of partGLSmoothTempSens (mlegp GP smoothing of E0 per year)."""
    e0 = e0fit.astype(float).copy()
    dup = np.concatenate([[False], np.diff(e0) == 0])
    e0[dup] = np.nan
    sde0 = sde0fit.astype(float).copy()
    year = np.ceil(daystart / 365).astype(int)
    out_e0 = np.full(e0.size, np.nan)
    out_sd = np.full(e0.size, np.nan)
    for yr in np.unique(year):
        ym = year == yr
        fin = ym & np.isfinite(e0)
        if fin.sum() == 0:
            continue
        ef, sf, xf = e0[fin], sde0[fin], icentral[fin].astype(float)
        if np.std(ef, ddof=1) / _r_mean(ef) < 0.01:
            out_e0[ym] = _r_mean(ef)
            out_sd[ym] = np.max(sf)
            continue
        predict, nugget = _gp_smooth(xf, ef, sf ** 2)
        fit, se = predict(icentral[ym].astype(float))
        nug_all = np.full(int(ym.sum()), _r_quantile7(nugget, 0.9))
        nug_all[np.isfinite(e0[ym])] = nugget
        out_e0[ym] = fit
        out_sd[ym] = se + np.sqrt(nug_all)
    nf = ~np.isfinite(out_e0)
    if nf.any() and (~nf).any():
        out_e0[nf] = _r_mean(out_e0[~nf])
        out_sd[nf] = _r_quantile7(out_sd[~nf], 0.9) * 1.5
    return out_e0, out_sd


def _fit_rref_windows(nee, temp, is_night, e0_smooth, i_central, dts, n, rref_fit):
    """Port of partGLFitNightRespRefOneWindow (lm) + fillNAForward.

    ``rref_fit`` is the RRef of the nighttime E0 fits per window (``RRefFit``,
    merged over the 12/24/48-day passes); only its first finite value is used,
    for a series that starts without an estimate.
    """
    rec_start, rec_end = _win_recs(i_central, WIN_NIGHT_DAYS, dts, n)
    nw = i_central.size
    rref = np.full(nw, np.nan)
    for w in range(nw):
        lo, hi = rec_start[w] - 1, rec_end[w]
        v = _is_valid_night(nee[lo:hi], temp[lo:hi], is_night[lo:hi])
        if v.sum() < MIN_NREC:
            continue
        reco = nee[lo:hi][v]
        if reco.size >= 3:
            tk = 273.15 + temp[lo:hi][v]
            tfac = _exp_r(e0_smooth[w] * (1.0 / (TREF_K - T0_K)
                                          - 1.0 / (tk - T0_K)))
            # coef(lm(REco ~ TFac - 1)): R's QR, not sum(x*y)/sum(x^2);
            # max(0, NA) stays NA in R
            coef = _lm_through_origin(tfac, reco)
            rref[w] = max(0.0, coef) if np.isfinite(coef) else np.nan
    # fillNAForward(RRef, firstValue = E0Smooth$RRef[which(is.finite(E0Smooth$RRef))[1]]).
    # E0Smooth has no RRef column yet, and R's `$` partially matches RRefFit, so
    # a series without an estimate in its first window starts with the first
    # finite RRefFit (the nighttime nls fit's RRef at the window's median
    # temperature), not with the first estimated RRef. Parity with REddyProc.
    if nw and not np.isfinite(rref[0]):
        fin_fit = rref_fit[np.isfinite(rref_fit)]
        rref[0] = fin_fit[0] if fin_fit.size else np.nan
    for w in range(1, nw):
        if not np.isfinite(rref[w]):
            rref[w] = rref[w - 1]
    return rref


# --------------------------------------------------------------------------- #
# Stage 3: light-response-curve fit per window
# --------------------------------------------------------------------------- #
def _make_cost(theta_full, iopt, flux, sdflux, prior, sdprior, rg, vpd, temp):
    """REddyProc's computeCost with predictLRC (rectangular hyperbola).

    The exponentials use :func:`_exp_r`: with numpy's ``exp`` the fits in a few
    ill-conditioned windows moved by up to 3e-6 relative (vmmin compares costs
    and differences them over ``ndeps = 1e-3``). They are cached, which keeps
    the bits: E0 is fixed during the fit, so the Lloyd-Taylor factor is a
    constant, and the VPD factor only changes with k.
    """
    iopt = np.asarray(iopt)
    # The optimizer probes large k/beta where exp() overflows to inf; that just
    # yields a non-finite cost the line search rejects, so silence the warnings.
    with np.errstate(over='ignore', invalid='ignore'):
        lloyd_taylor = _exp_r(theta_full[4] * (1.0 / (TREF_K - T0_K)
                                               - 1.0 / (temp + 273.15 - T0_K)))
    above = vpd > VPD0
    dvpd = vpd[above] - VPD0
    vpd_na = np.isnan(vpd)
    vpd_factor = {}

    def cost(theta_opt):
        theta = theta_full.copy()
        theta[iopt] = theta_opt
        k, beta, alpha, rref = theta[:4]
        with np.errstate(over='ignore', invalid='ignore'):
            amax = np.full(rg.shape, beta)
            if k != 0:  # fixVPD = (k == 0)
                f = vpd_factor.get(k)
                if f is None:
                    if len(vpd_factor) > 16:
                        vpd_factor.clear()
                    f = vpd_factor[k] = _exp_r(-k * dvpd)
                amax[above] = beta * f
                amax[vpd_na] = np.nan  # ifelse(NA > VPD0, ...) is NA
            gpp = (amax * alpha * rg) / (alpha * rg + amax)
            nep = gpp - rref * lloyd_taylor
            mfp = ((theta - prior) / sdprior) ** 2
            # R's sum() (long double), not numpy's pairwise sum
            return _r_sum(((nep - flux) / sdflux) ** 2) + _r_sum(mfp[~np.isnan(mfp)])

    return cost


def _get_iopt(fixed_vpd, fixed_alpha):
    if not fixed_vpd and not fixed_alpha:
        return [0, 1, 2, 3]
    if fixed_vpd and not fixed_alpha:
        return [1, 2, 3]
    if not fixed_vpd and fixed_alpha:
        return [0, 1, 3]
    return [1, 3]


def _optim_adjusted_prior(theta, iopt, day, prior):
    nee, sdnee, rg, vpd, temp = day
    fin = np.isfinite(nee) & np.isfinite(sdnee)
    nee, sdnee, rg, vpd, temp = nee[fin], sdnee[fin], rg[fin], vpd[fin], temp[fin]
    min_unc = _r_quantile7(sdnee, 0.3)
    fc_unc = np.maximum(sdnee, min_unc)  # isBoundLowerNEEUncertainty=TRUE
    sdprior = LASSLOP_SDPRIOR.copy()
    sdprior[[i for i in range(5) if i not in iopt]] = np.nan
    cost = _make_cost(theta, iopt, -nee, fc_unc, prior, sdprior, rg, vpd, temp)
    par, val, fail = _vmmin(theta[np.asarray(iopt)], cost)
    hess = _optim_hess(par, cost)
    theta_opt = theta.copy()
    theta_opt[np.asarray(iopt)] = par
    return dict(theta=theta_opt, iopt=list(iopt), value=val,
                convergence=fail, hessian=hess)


def _optim_lrc_bounds(theta0, prior, day, last_good, neglect_vpd=False):
    last_good = last_good.copy()
    if not np.isfinite(last_good[2]):
        last_good[2] = 0.22
    # isNeglectVPDEffect: k fixed at 0, so Amax = beta whatever VPD is
    is_fixed_vpd = neglect_vpd or (np.nansum(day[3] >= VPD0) == 0)
    is_fixed_alpha = False
    theta0_adj = theta0.copy()
    if neglect_vpd:
        theta0_adj[0] = 0
    res = _optim_adjusted_prior(theta0_adj, _get_iopt(is_fixed_vpd, False), day, prior)
    th = res['theta']
    if not np.isfinite(th[0]) or th[0] < 0:
        is_fixed_vpd = True
        theta0_adj[0] = 0
        res = _optim_adjusted_prior(theta0_adj, _get_iopt(True, False), day, prior)
        th = res['theta']
        if (not np.isfinite(th[2]) or th[2] > 0.22) and np.isfinite(last_good[2]):
            theta0_adj[2] = last_good[2]
            res = _optim_adjusted_prior(theta0_adj, _get_iopt(True, True), day, prior)
    else:
        if (not np.isfinite(th[2]) or th[2] > 0.22) and np.isfinite(last_good[2]):
            theta0_adj[2] = last_good[2]
            res = _optim_adjusted_prior(theta0_adj, _get_iopt(is_fixed_vpd, True), day, prior)
            th = res['theta']
            if not np.isfinite(th[0]) or th[0] < 0:
                theta0_adj[0] = 0
                res = _optim_adjusted_prior(theta0_adj, _get_iopt(True, True), day, prior)
    if res['convergence'] != 0:
        res['theta'] = np.full(5, np.nan)
    th = res['theta']
    if np.isfinite(th[0]) and (th[2] < 0 or th[3] < 0 or th[1] < 0 or th[1] >= 250):
        res['theta'] = np.full(5, np.nan)
        res['convergence'] = 1002
    return res


def _r_solve(a):
    """R's ``solve(a)``: ``None`` where R stops, i.e. for an exactly singular
    matrix and also when LAPACK's reciprocal condition number (1-norm,
    ``dgecon``) is below ``.Machine$double.eps``."""
    lu, piv, info = lapack.dgetrf(a)
    if info != 0:
        return None
    anorm = float(np.abs(a).sum(axis=0).max())
    rcond, info = lapack.dgecon(lu, anorm, norm='1')
    if not rcond >= np.finfo(float).eps:
        return None
    return lapack.dgetri(lu, piv)[0]


def _fit_lrc(day, e0, sde0, rref_night, last_good, neglect_vpd=False):
    nee = day[0]
    nee_fin = nee[np.isfinite(nee)]
    # R's quantile() interpolates as (1 - h) * a + h * b, numpy as a + (b - a) * h
    beta_prior = abs(_r_quantile7(nee_fin, 0.03) - _r_quantile7(nee_fin, 0.97))
    prior = np.array([0.05, beta_prior, 0.1, rref_night, e0])
    inits = np.tile(prior, (3, 1))
    inits[1, 1] = prior[1] * 1.3
    inits[2, 1] = prior[1] * 0.8
    results = [_optim_lrc_bounds(inits[r], prior, day, last_good, neglect_vpd)
               for r in range(3)]
    valid = [r for r in results if np.isfinite(r['theta'][0])]
    if not valid:
        return None
    best = min(valid, key=lambda r: r['value'])
    theta, iopt, hess = best['theta'], best['iopt'], best['hessian']
    if hess[0, 0] < 1e-8:
        cov_lrc = np.zeros_like(hess)
        inv = _r_solve(hess[1:, 1:])
        if inv is not None:
            cov_lrc[1:, 1:] = inv
    else:
        cov_lrc = inv = _r_solve(hess)
    if inv is None:
        return None  # 1006
    cov = np.zeros((5, 5))
    cov[4, 4] = sde0 ** 2
    ix = np.array(iopt)
    cov[np.ix_(ix, ix)] = cov_lrc
    if np.any(np.diag(cov) < 0):
        return None  # 1005
    sd_theta = np.full(5, np.nan)
    iopt_full = list(iopt) + [4]
    sd_theta[iopt_full] = np.sqrt(np.diag(cov)[iopt_full])
    if not np.isfinite(theta[1]):
        return None
    if theta[1] > 100 and sd_theta[1] >= theta[1]:
        return None  # 1002
    return best


# --------------------------------------------------------------------------- #
# Stage 4: interpolate fluxes between neighboring windows
# --------------------------------------------------------------------------- #
def _associate_special_rows(special, nrec):
    """Port of .partGPAssociateSpecialRows (1-based special record indices)."""
    nS = special.size
    i_before = np.zeros(nrec, int)
    i_after = np.zeros(nrec, int)
    w_before = np.zeros(nrec)
    w_after = np.zeros(nrec)
    for s in range(nS):
        r = special[s] - 1
        i_before[r] = i_after[r] = special[s]
        w_before[r] = w_after[r] = 0.5
    for s in range(nS):
        curr = special[s]
        prev = special[s] if s == 0 else special[s - 1]
        nxt = special[s] if s == nS - 1 else special[s + 1]
        dist_prev = curr - prev
        if dist_prev > 1:
            rows = np.arange(prev + 1, curr)
            i_after[rows - 1] = curr
            w_after[rows - 1] = np.arange(1, dist_prev) / dist_prev
        dist_next = nxt - curr
        if dist_next > 1:
            rows = np.arange(curr + 1, nxt)
            i_before[rows - 1] = curr
            w_before[rows - 1] = np.arange(dist_next - 1, 0, -1) / dist_next
    first, last = special[0], special[nS - 1]
    i_before[:first] = i_after[:first] = first
    w_before[:first] = w_after[:first] = 0.5
    i_before[last - 1:] = i_after[last - 1:] = last
    w_before[last - 1:] = w_after[last - 1:] = 0.5
    return i_before, i_after, w_before, w_after


def _interpolate_fluxes(i_mean, params, rg, vpd, temp, nrec):
    """Port of partGLInterpolateFluxes (isAssociateParmsToMeanOfValids=TRUE)."""
    # drop duplicate iMeanRec (keep first), like REddyProc
    seen = set()
    keep = []
    for i, m in enumerate(i_mean):
        if m not in seen:
            seen.add(m)
            keep.append(i)
    i_mean = i_mean[keep]
    params = params[keep]
    order = np.argsort(i_mean)
    i_mean = i_mean[order]
    params = params[order]
    mean_to_row = {m: i for i, m in enumerate(i_mean)}

    i_before, i_after, w_before, w_after = _associate_special_rows(i_mean, nrec)
    row_b = np.array([mean_to_row[m] for m in i_before])
    row_a = np.array([mean_to_row[m] for m in i_after])
    p_b, p_a = params[row_b], params[row_a]

    temp_pred = np.maximum(-40.0, temp)
    temp_k = temp_pred + 273.15

    def reco(p):
        return _lloyd_taylor(p[:, 3], p[:, 4], temp_k)

    def gpp(p):
        k, beta, alpha = p[:, 0], p[:, 1], p[:, 2]
        fix = (k == 0)
        with np.errstate(over='ignore', invalid='ignore'):
            # R's ifelse(VPD > VPD0, ...) is NA where VPD is NA (unless k == 0)
            amax = np.where(fix, beta,
                            np.where(vpd > VPD0, beta * _exp_r(-k * (vpd - VPD0)),
                                     np.where(np.isnan(vpd), np.nan, beta)))
            return (amax * alpha * rg) / (alpha * rg + amax)

    reco_out = w_before * reco(p_b) + w_after * reco(p_a)
    gpp_out = w_before * gpp(p_b) + w_after * gpp(p_a)
    return reco_out, gpp_out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _fit_lrc_windows(nee, sd_nee, ta, vpd, rg, is_day, i_central, e0_sm, sde0_sm,
                     rref_win, dts, n, neglect_vpd=False):
    """Port of partGLFitLRCWindows / partGLFitLRCOneWindow.

    Returns the lists ``(i_mean, params, i_central)`` of the accepted windows.
    ``neglect_vpd`` is ``controlGLPart$isNeglectVPDEffect``.
    """
    rec_start, rec_end = _win_recs(i_central, WIN_REF_DAYS, dts, n)
    i_mean_list, params_list, central_list = [], [], []
    last_good = np.full(5, np.nan)
    for w in range(i_central.size):
        if not np.isfinite(e0_sm[w]):
            continue
        lo, hi = rec_start[w] - 1, rec_end[w]
        sl = slice(lo, hi)
        valid_no_vpd = (is_day[sl] & np.isfinite(nee[sl]) & np.isfinite(ta[sl])
                        & np.isfinite(rg[sl]) & np.isfinite(sd_nee[sl]))
        neglect_vpd_w = neglect_vpd
        valid = valid_no_vpd if neglect_vpd else valid_no_vpd & np.isfinite(vpd[sl])
        if valid.sum() < MIN_NREC:
            # too few records with VPD: this window neglects the VPD effect
            neglect_vpd_w = True
            valid = valid_no_vpd
            if valid.sum() < MIN_NREC:
                continue
        i_mean_local = int(round(float(np.nonzero(valid)[0].mean()) + 1))  # 1-based
        i_mean_global = lo + i_mean_local  # iRecStart-1 + local
        day = (nee[sl][valid], sd_nee[sl][valid], rg[sl][valid],
               vpd[sl][valid], ta[sl][valid])
        res = _fit_lrc(day, e0_sm[w], sde0_sm[w], rref_win[w], last_good,
                       neglect_vpd_w)
        if res is None:
            continue
        last_good = res['theta']
        i_mean_list.append(i_mean_global)
        params_list.append(res['theta'])
        central_list.append(int(i_central[w]))
    return i_mean_list, params_list, central_list


def _partition_daytime(nee, sd_nee, ta, vpd, rg, doy, hour, lat, lon,
                       utc_offset, dts, verbose=1):
    n = nee.size
    out = {c: np.full(n, np.nan) for c in
           ('RECO_DT_RP', 'GPP_DT_RP', 'K_DT_RP', 'BETA_DT_RP',
            'ALPHA_DT_RP', 'RREF_DT_RP', 'E0_DT_RP')}

    potrad = potential_radiation(doy, hour, lat, lon, utc_offset)
    with np.errstate(invalid='ignore'):
        is_night = (rg <= DAY_MAX_SW_IN) & (potrad <= 0.0)
        is_day = (rg > DAY_MAX_SW_IN) & (potrad > 0.0)

    start_days, i_central = _window_grid(n, dts)
    nw = i_central.size

    # --- Stage 2: nighttime E0 (nls) + window extension ---
    e0, sde0, _, rref_fit = _fit_nighttime_pass(nee, ta, is_night, i_central,
                                                WIN_NIGHT_DAYS, dts, n)
    for win_days in WIN_EXTEND:
        miss = ~np.isfinite(e0)
        if not miss.any():
            break
        e0x, sdx, _, rfx = _fit_nighttime_pass(nee, ta, is_night, i_central,
                                               win_days, dts, n)
        # resNight[iNoSummary, ] <- resNightExtend[iNoSummary, ]: whole rows,
        # RRefFit included (the first finite one seeds the RRef series)
        e0[miss], sde0[miss], rref_fit[miss] = e0x[miss], sdx[miss], rfx[miss]

    n_finite = int(np.isfinite(e0).sum())
    if n_finite < 5 and n_finite < 0.1 * nw:
        warn("Daytime partitioning (ReddyProc): too few nighttime E0 estimates "
             f"({n_finite} windows); record left unpartitioned.", verbose=verbose)
        return out

    # GP smoothing of E0 across time, then RRef per window
    e0_sm, sde0_sm = _smooth_tempsens(e0, sde0, i_central, start_days)
    rref_win = _fit_rref_windows(nee, ta, is_night, e0_sm, i_central, dts, n, rref_fit)

    # --- Stage 3: LRC fit per window ---
    i_mean_list, params_list, central_list = _fit_lrc_windows(
        nee, sd_nee, ta, vpd, rg, is_day, i_central, e0_sm, sde0_sm, rref_win,
        dts, n)
    if not params_list:
        warn("Daytime partitioning (ReddyProc): no light-response curve could be "
             "fitted; record left unpartitioned.", verbose=verbose)
        return out

    # --- Stage 4: interpolate Reco/GPP to every record ---
    reco, gpp = _interpolate_fluxes(np.array(i_mean_list, int),
                                    np.array(params_list), rg, vpd, ta, n)
    # isRefitMissingVPDWithNeglectVPDEffect: where VPD is missing, GPP is NA
    # unless both neighbouring windows have k = 0. REddyProc then refits all
    # windows without the VPD effect and takes RECO and GPP from that fit there.
    na_vpd = np.isnan(vpd) & np.isnan(gpp)
    if na_vpd.any():
        i_mean_nv, params_nv, _ = _fit_lrc_windows(
            nee, sd_nee, ta, vpd, rg, is_day, i_central, e0_sm, sde0_sm,
            rref_win, dts, n, neglect_vpd=True)
        if params_nv:
            reco_nv, gpp_nv = _interpolate_fluxes(np.array(i_mean_nv, int),
                                                  np.array(params_nv), rg, vpd, ta, n)
            reco[na_vpd], gpp[na_vpd] = reco_nv[na_vpd], gpp_nv[na_vpd]
    out['RECO_DT_RP'] = reco
    out['GPP_DT_RP'] = gpp

    # report LRC parameters at the central record of each window (like REddyProc)
    for c, p in zip(central_list, params_list, strict=False):
        idx = c - 1
        if 0 <= idx < n:
            out['K_DT_RP'][idx] = p[0]
            out['BETA_DT_RP'][idx] = p[1]
            out['ALPHA_DT_RP'][idx] = p[2]
            out['RREF_DT_RP'][idx] = p[3]
            out['E0_DT_RP'][idx] = p[4]
    return out


def _replace_missing_sd(sd, nee):
    """REddyProc replaceMissingSdByPercentage: max(minSd, perc*|NEE|)."""
    sd = sd.astype(float).copy()
    fill = ~np.isfinite(sd)
    # pmax(..., na.rm = TRUE): minSd also where NEE is missing
    sd[fill] = np.fmax(SD_MINSD, np.abs(nee[fill] * SD_PERC))
    return sd


class DaytimePartitioningReddyProc:
    """Partition NEE into GPP and RECO with the daytime method (REddyProc).

    Faithful, vectorized port of REddyProc's ``partitionNEEGL`` (Lasslop et al.
    2010 light-response-curve method). Fits a rectangular-hyperbola LRC to
    daytime NEE in short windows with the temperature sensitivity ``E0`` fixed
    from nighttime data, then predicts GPP and RECO for every record. Emits
    ``*_DT_RP`` columns, mirroring the nighttime REddyProc port's ``*_NT_RP``.

    Example: ``examples/flux/partitioning/partitioning_daytime_reddyproc.py``

    Example:
        >>> import diive as dv
        >>> df = dv.load_exampledata_parquet()
        >>> part = dv.flux.DaytimePartitioningReddyProc(
        ...     nee=df['NEE_CUT_REF_orig'], ta=df['Tair_f'], vpd=df['VPD_f'],
        ...     sw_in=df['Rg_f'], lat=46.815, lon=9.855, utc_offset=1)
        >>> part.run()  # then part.results -> DataFrame with RECO_DT_RP, GPP_DT_RP, ...
    """

    def __init__(self,
                 nee: Series,
                 ta: Series,
                 vpd: Series,
                 sw_in: Series,
                 lat: float,
                 lon: float,
                 utc_offset: float,
                 nee_sd: Series | None = None,
                 vpd_in_kpa: bool = True,
                 verbose: int = 2):
        """
        Args:
            nee: Measured net ecosystem exchange (umol m-2 s-1). Gaps (NaN) are
                the records that were not measured / did not pass QC; the daytime
                LRC is fitted on the measured daytime values only.
            ta: Gap-filled air temperature (degC). REddyProc's daytime method
                uses the gap-filled meteo drivers throughout (both for fitting
                and for prediction), quality-filtering only NEE.
            vpd: Gap-filled vapour pressure deficit. By default in kPa (diive
                convention) and converted internally to hPa, the unit
                REddyProc's Lasslop LRC expects (VPD0 = 10 hPa). Pass
                ``vpd_in_kpa=False`` if ``vpd`` is already in hPa.
            sw_in: Gap-filled incoming shortwave radiation (W m-2). Used both for
                the day/night split and as the LRC light driver.
            lat: Site latitude in decimal degrees.
            lon: Site longitude in decimal degrees (needed for the solar-time
                day/night split).
            utc_offset: Time zone offset from UTC in hours (e.g. +1 for CET).
            nee_sd: Per-record NEE uncertainty (umol m-2 s-1) used to weight the
                LRC fit. If ``None``, REddyProc's default is reproduced: missing
                uncertainties are set to ``max(0.7, 0.2*|NEE|)``.
            vpd_in_kpa: If True (default), ``vpd`` is in kPa and multiplied by 10
                to hPa internally.
            verbose: Console verbosity level (0 silent, 1 warnings, 2 progress
                + report, 3 debug). Default 2.
        """
        self._inputs = self._validate(nee, ta, vpd, sw_in, nee_sd)
        self.lat = float(lat)
        self.lon = float(lon)
        self.utc_offset = float(utc_offset)
        self.vpd_in_kpa = bool(vpd_in_kpa)
        self.verbose = verbose
        self._results: DataFrame | None = None

    @staticmethod
    def _validate(nee, ta, vpd, sw_in, nee_sd) -> DataFrame:
        series = {'nee': nee, 'ta': ta, 'vpd': vpd, 'sw_in': sw_in}
        if nee_sd is not None:
            series['nee_sd'] = nee_sd
        for name, s in series.items():
            if not isinstance(s, Series):
                raise TypeError(f"'{name}' must be a pandas Series, got {type(s)}.")
            if not isinstance(s.index, pd.DatetimeIndex):
                raise TypeError(f"'{name}' must have a DatetimeIndex.")
        df = pd.DataFrame({k: v.astype(float) for k, v in series.items()})
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df

    def run(self) -> "DaytimePartitioningReddyProc":
        """Run the partitioning and populate :attr:`results`."""
        df = self._inputs
        index = df.index
        doy = index.dayofyear.to_numpy()
        hour = (index.hour + index.minute / 60.0).to_numpy()
        dts = _infer_dts(index)

        nee = df['nee'].to_numpy()
        vpd = df['vpd'].to_numpy() * (10.0 if self.vpd_in_kpa else 1.0)
        if 'nee_sd' in df:
            sd_nee = _replace_missing_sd(df['nee_sd'].to_numpy(), nee)
        else:
            sd_nee = _replace_missing_sd(np.full(nee.size, np.nan), nee)

        if self.verbose:
            info("Daytime partitioning ReddyProc (Lasslop et al. 2010) "
                 f"starting for {len(index)} records ({dts} per day).",
                 verbose=self.verbose)

        # Single-threaded BLAS: the GP smoother runs hundreds of small Cholesky
        # factorizations, which multi-threaded OpenBLAS makes slower (up to 10x
        # under load) and whose last bits depend on the thread count.
        with threadpool_limits(limits=1, user_api='blas'):
            out = _partition_daytime(
                nee=nee, sd_nee=sd_nee, ta=df['ta'].to_numpy(), vpd=vpd,
                rg=df['sw_in'].to_numpy(), doy=doy, hour=hour, lat=self.lat,
                lon=self.lon, utc_offset=self.utc_offset, dts=dts, verbose=self.verbose)

        cols = ['RECO_DT_RP', 'GPP_DT_RP', 'K_DT_RP', 'BETA_DT_RP',
                'ALPHA_DT_RP', 'RREF_DT_RP', 'E0_DT_RP']
        self._results = pd.DataFrame({c: out[c] for c in cols}, index=index)

        self.report()
        if self.verbose:
            success("Daytime partitioning (ReddyProc) finished.",
                    verbose=self.verbose)
        return self

    @property
    def results(self) -> DataFrame:
        """DataFrame of partitioning results (aligned to the input index).

        Columns: ``RECO_DT_RP`` (ecosystem respiration), ``GPP_DT_RP`` (gross
        primary production), and the fitted LRC parameters ``K_DT_RP``,
        ``BETA_DT_RP``, ``ALPHA_DT_RP``, ``RREF_DT_RP``, ``E0_DT_RP`` reported at
        the central record of each window (NaN elsewhere).
        """
        if self._results is None:
            raise RuntimeError("Call .run() before accessing .results.")
        return self._results

    @property
    def reco(self) -> Series:
        """Ecosystem respiration, umol m-2 s-1."""
        return self.results['RECO_DT_RP']

    @property
    def gpp(self) -> Series:
        """Gross primary production, umol m-2 s-1."""
        return self.results['GPP_DT_RP']

    def report(self) -> None:
        """Print a Rich per-year summary of the partitioning result."""
        partitioning_report(
            title="Daytime NEE Partitioning REddyProc (Lasslop et al. 2010)",
            reference="Wutzler et al. (2018), https://doi.org/10.5194/bg-15-5015-2018",
            results=self.results, reco_col='RECO_DT_RP', gpp_col='GPP_DT_RP',
            e0_col='E0_DT_RP', e0_unit='K', verbose=self.verbose)


def partition_nee_daytime_reddyproc(nee: Series, ta: Series, vpd: Series,
                                    sw_in: Series, lat: float, lon: float,
                                    utc_offset: float,
                                    nee_sd: Series | None = None,
                                    vpd_in_kpa: bool = True,
                                    verbose: int = 2) -> DataFrame:
    """Functional wrapper around :class:`DaytimePartitioningReddyProc`.

    See :class:`DaytimePartitioningReddyProc` for argument semantics.

    Returns:
        Results DataFrame (RECO_DT_RP, GPP_DT_RP, ...).
    """
    return DaytimePartitioningReddyProc(
        nee=nee, ta=ta, vpd=vpd, sw_in=sw_in, lat=lat, lon=lon,
        utc_offset=utc_offset, nee_sd=nee_sd, vpd_in_kpa=vpd_in_kpa,
        verbose=verbose).run().results
