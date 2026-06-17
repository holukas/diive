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
   refit cascade (fix VPD / fix alpha / reject out-of-range parameters).
4. Predict RECO and GPP for every record by interpolating the two neighboring
   windows' parameter sets with distance-based weights.

This port reproduces REddyProc's algorithm faithfully. Because the method is a
stack of three nested numerical optimizers (nighttime ``nls``, the ``mlegp``
Gaussian-process smoother, and the LRC ``optim``), exact bit-for-bit parity is
not attainable across languages, but each stage matches REddyProc closely:
the day/night split and the flux interpolation are exact (~1e-13), the LRC fit
matches per-window parameters to ~1e-6 for the large majority of windows, the
GP-smoothed ``E0`` matches to ~0.03 K (a flux-negligible <0.1% on RECO), and
window acceptance/rejection matches exactly.

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

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.optimize import minimize

from diive.core.utils.console import info, warn, success
from diive.flux.partitioning._report import partitioning_report
from diive.flux.partitioning.nighttime_reddyproc import (
    potential_radiation, _infer_dts, T0_K)

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
    return rref * np.exp(e0 * (1.0 / (tref_k - T0_K) - 1.0 / (ta_k - T0_K)))


# --------------------------------------------------------------------------- #
# R's nls default Gauss-Newton (for the nighttime E0 fit)
# --------------------------------------------------------------------------- #
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


def _r_nls(y, predict, start):
    """Faithful port of R's nls default Gauss-Newton (src/.../nls.c + nlsModel).

    Returns ``(par, cov)`` on convergence, or ``(None, None)`` on any
    non-convergence (singular gradient / step factor below minFactor / maxiter
    exceeded) - exactly when R's ``nls()`` throws and REddyProc treats the fit
    as NA.
    """
    npar = start.size
    par = start.astype(float).copy()
    rhs = predict(par)
    grad = _numeric_deriv(predict, par, rhs)
    resid = y - rhs
    dev = float((resid ** 2).sum())
    try:
        Q, R = np.linalg.qr(grad)
        if abs(np.linalg.det(R)) == 0.0:
            return None, None
    except np.linalg.LinAlgError:
        return None, None
    fac = 1.0
    converged = False
    for _ in range(_NLS_MAXITER):
        proj = Q.T @ resid
        ss_proj = float((proj ** 2).sum())
        denom = float((resid ** 2).sum()) - ss_proj
        conv = np.sqrt(ss_proj / denom) if denom > 0 else np.inf
        if conv <= _NLS_TOL:
            converged = True
            break
        incr = np.linalg.solve(R, proj)
        while fac >= _NLS_MINFAC:
            new_par = par + fac * incr
            new_rhs = predict(new_par)
            new_grad = _numeric_deriv(predict, new_par, new_rhs)
            try:
                new_Q, new_R = np.linalg.qr(new_grad)
            except np.linalg.LinAlgError:
                return None, None
            if not np.all(np.isfinite(new_R)) or abs(np.linalg.det(new_R)) == 0.0:
                return None, None
            new_resid = y - new_rhs
            new_dev = float((new_resid ** 2).sum())
            if new_dev <= dev:
                dev, par, rhs, grad = new_dev, new_par, new_rhs, new_grad
                Q, R, resid = new_Q, new_R, new_resid
                fac = min(2 * fac, 1.0)
                break
            fac /= 2.0
        if fac < _NLS_MINFAC:
            return None, None
    if not converged:
        return None, None
    n = y.size
    rinv = np.linalg.inv(R)
    cov = (dev / (n - npar)) * (rinv @ rinv.T)
    return par, cov


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
    """Faithful port of R's vmmin BFGS (src/appl/optim.c), reltol=1e-3."""
    n = b0.size
    b = b0.astype(float).copy()
    f = fn(b)
    Fmin = f
    g = _fmingr(fn, b)
    iter_ = 1
    gradcount = 1
    ilast = gradcount
    B = np.eye(n)
    count = 0
    while True:
        if ilast == gradcount:
            B = np.eye(n)
        X = b.copy()
        c = g.copy()
        t = -(B @ g)
        gradproj = float(t @ g)
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
                    accpoint = np.isfinite(f) and (
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
                t = steplength * t
                c = g - c
                D1 = float(t @ c)
                if D1 > 0:
                    X2 = B @ c
                    D2 = 1.0 + float(X2 @ c) / D1
                    B = B + (D2 * np.outer(t, t) - np.outer(X2, t)
                             - np.outer(t, X2)) / D1
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
    start_rref = float(np.nanmean(reco))

    def predict(p):
        return p[0] * np.exp(p[1] * b)

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


def _gp_smooth(x, z, nug):
    """mlegp GP MLE: Gaussian correlation, constant GLS mean + sig2 profiled out,
    free params (log beta, log nugget_scale). Returns (predict, nugget_vec)."""
    x = np.asarray(x, float)
    z = np.asarray(z, float).reshape(-1, 1)
    nug = np.asarray(nug, float)
    npts = x.size
    D2 = (x[:, None] - x[None, :]) ** 2
    one = np.ones((npts, 1))

    def neg_ll(v):
        beta, nscale = np.exp(v[0]), np.exp(v[1])
        A = np.exp(-beta * D2) + nscale * np.diag(nug)
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return 1e300
        mu = float(((one.T @ Ainv @ z) / (one.T @ Ainv @ one)).item())
        r = z - mu
        sig2 = float(((r.T @ Ainv @ r) / npts).item())
        if not np.isfinite(sig2) or sig2 <= 0:
            return 1e300
        _, logdet = np.linalg.slogdet(sig2 * A)
        return 0.5 * (npts * np.log(2 * np.pi) + logdet
                      + (r.T @ Ainv @ r)[0, 0] / sig2)

    xr = x.max() - x.min()
    best = None
    for b0 in (np.log(1.0 / xr ** 2 * f) for f in (0.1, 1.0, 10.0)):
        for s0 in (np.log(s) for s in (0.1, 1.0, 10.0)):
            res = minimize(neg_ll, [b0, s0], method='Nelder-Mead',
                           options=dict(xatol=1e-8, fatol=1e-8, maxiter=2000))
            if best is None or res.fun < best.fun:
                best = res
    beta, nscale = np.exp(best.x[0]), np.exp(best.x[1])
    K = np.exp(-beta * D2)
    nugget_vec = nscale * nug
    Ainv = np.linalg.inv(K + np.diag(nugget_vec))
    mu = float(((one.T @ Ainv @ z) / (one.T @ Ainv @ one)).item())
    sig2 = float((((z - mu).T @ Ainv @ (z - mu)) / npts).item())
    Vinv = np.linalg.inv(sig2 * K + np.diag(sig2 * nugget_vec))
    zc = z - mu

    def predict(xnew):
        xnew = np.atleast_1d(np.asarray(xnew, float))
        rr = np.exp(-beta * (xnew[:, None] - x[None, :]) ** 2)
        fit = mu + sig2 * (rr @ (Vinv @ zc)).ravel()
        var = sig2 - sig2 * np.einsum('ij,jk,ik->i', rr, Vinv, rr) * sig2
        return fit, np.sqrt(np.clip(var, 0, None))

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
        if np.std(ef, ddof=1) / np.mean(ef) < 0.01:
            out_e0[ym] = np.mean(ef)
            out_sd[ym] = np.max(sf)
            continue
        predict, nugget = _gp_smooth(xf, ef, sf ** 2)
        fit, se = predict(icentral[ym].astype(float))
        nug_all = np.full(int(ym.sum()), np.quantile(nugget, 0.9))
        nug_all[np.isfinite(e0[ym])] = nugget
        out_e0[ym] = fit
        out_sd[ym] = se + np.sqrt(nug_all)
    nf = ~np.isfinite(out_e0)
    if nf.any() and (~nf).any():
        out_e0[nf] = np.mean(out_e0[~nf])
        out_sd[nf] = np.quantile(out_sd[~nf], 0.9) * 1.5
    return out_e0, out_sd


def _fit_rref_windows(nee, temp, is_night, e0_smooth, i_central, dts, n):
    """Port of partGLFitNightRespRefOneWindow (lm) + fillNAForward."""
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
            tk = temp[lo:hi][v] + 273.15
            tfac = np.exp(e0_smooth[w] * (1.0 / (TREF_K - T0_K)
                                          - 1.0 / (tk - T0_K)))
            rref[w] = max(0.0, float((tfac * reco).sum() / (tfac * tfac).sum()))
    fin = np.isfinite(rref)
    if fin.any():
        cur = rref[fin][0]
        for w in range(nw):
            if np.isfinite(rref[w]):
                cur = rref[w]
            else:
                rref[w] = cur
    return rref


# --------------------------------------------------------------------------- #
# Stage 3: light-response-curve fit per window
# --------------------------------------------------------------------------- #
def _predict_nep(theta, rg, vpd, temp, fix_vpd):
    # The optimizer probes large k/beta where exp() overflows to inf; that just
    # yields a non-finite cost the line search rejects, so silence the warnings.
    k, beta, alpha, rref, e0 = theta
    with np.errstate(over='ignore', invalid='ignore'):
        if fix_vpd:
            amax = np.full(rg.shape, beta)
        else:
            amax = np.where(vpd > VPD0, beta * np.exp(-k * (vpd - VPD0)), beta)
        reco = rref * np.exp(e0 * (1.0 / (TREF_K - T0_K) - 1.0 / (temp + 273.15 - T0_K)))
        gpp = (amax * alpha * rg) / (alpha * rg + amax)
    return gpp - reco


def _make_cost(theta_full, iopt, flux, sdflux, prior, sdprior, rg, vpd, temp):
    iopt = np.asarray(iopt)

    def cost(theta_opt):
        theta = theta_full.copy()
        theta[iopt] = theta_opt
        fix_vpd = (theta[0] == 0)
        nep = _predict_nep(theta, rg, vpd, temp, fix_vpd)
        mfp = ((theta - prior) / sdprior) ** 2
        return float(np.sum(((nep - flux) / sdflux) ** 2)) + float(np.nansum(mfp))

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
    min_unc = np.quantile(sdnee, 0.3)
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


def _optim_lrc_bounds(theta0, prior, day, last_good):
    last_good = last_good.copy()
    if not np.isfinite(last_good[2]):
        last_good[2] = 0.22
    is_fixed_vpd = (np.nansum(day[3] >= VPD0) == 0)
    is_fixed_alpha = False
    theta0_adj = theta0.copy()
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


def _fit_lrc(day, e0, sde0, rref_night, last_good):
    nee = day[0]
    nee_fin = nee[np.isfinite(nee)]
    beta_prior = abs(np.quantile(nee_fin, 0.03) - np.quantile(nee_fin, 0.97))
    prior = np.array([0.05, beta_prior, 0.1, rref_night, e0])
    inits = np.tile(prior, (3, 1))
    inits[1, 1] = prior[1] * 1.3
    inits[2, 1] = prior[1] * 0.8
    results = [_optim_lrc_bounds(inits[r], prior, day, last_good) for r in range(3)]
    valid = [r for r in results if np.isfinite(r['theta'][0])]
    if not valid:
        return None
    best = min(valid, key=lambda r: r['value'])
    theta, iopt, hess = best['theta'], best['iopt'], best['hessian']
    try:
        if hess[0, 0] < 1e-8:
            cov_lrc = np.zeros_like(hess)
            cov_lrc[1:, 1:] = np.linalg.inv(hess[1:, 1:])
        else:
            cov_lrc = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
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
            amax = np.where(fix, beta,
                            np.where(vpd > VPD0, beta * np.exp(-k * (vpd - VPD0)), beta))
            return (amax * alpha * rg) / (alpha * rg + amax)

    reco_out = w_before * reco(p_b) + w_after * reco(p_a)
    gpp_out = w_before * gpp(p_b) + w_after * gpp(p_a)
    return reco_out, gpp_out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
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
    e0, sde0, _, _ = _fit_nighttime_pass(nee, ta, is_night, i_central,
                                         WIN_NIGHT_DAYS, dts, n)
    for win_days in WIN_EXTEND:
        miss = ~np.isfinite(e0)
        if not miss.any():
            break
        e0x, sdx, _, _ = _fit_nighttime_pass(nee, ta, is_night, i_central,
                                             win_days, dts, n)
        e0[miss], sde0[miss] = e0x[miss], sdx[miss]

    n_finite = int(np.isfinite(e0).sum())
    if n_finite < 5 and n_finite < 0.1 * nw:
        warn("Daytime partitioning (ReddyProc): too few nighttime E0 estimates "
             f"({n_finite} windows); record left unpartitioned.", verbose=verbose)
        return out

    # GP smoothing of E0 across time, then RRef per window
    e0_sm, sde0_sm = _smooth_tempsens(e0, sde0, i_central, start_days)
    rref_win = _fit_rref_windows(nee, ta, is_night, e0_sm, i_central, dts, n)

    # --- Stage 3: LRC fit per window ---
    rec_start, rec_end = _win_recs(i_central, WIN_REF_DAYS, dts, n)
    i_mean_list, params_list, central_list = [], [], []
    last_good = np.full(5, np.nan)
    for w in range(nw):
        if not np.isfinite(e0_sm[w]):
            continue
        lo, hi = rec_start[w] - 1, rec_end[w]
        sl = slice(lo, hi)
        valid = (is_day[sl] & np.isfinite(nee[sl]) & np.isfinite(ta[sl])
                 & np.isfinite(rg[sl]) & np.isfinite(sd_nee[sl]) & np.isfinite(vpd[sl]))
        if valid.sum() < MIN_NREC:
            valid = (is_day[sl] & np.isfinite(nee[sl]) & np.isfinite(ta[sl])
                     & np.isfinite(rg[sl]) & np.isfinite(sd_nee[sl]))
            if valid.sum() < MIN_NREC:
                continue
        i_mean_local = int(round(float(np.nonzero(valid)[0].mean()) + 1))  # 1-based
        i_mean_global = lo + i_mean_local  # iRecStart-1 + local
        day = (nee[sl][valid], sd_nee[sl][valid], rg[sl][valid],
               vpd[sl][valid], ta[sl][valid])
        res = _fit_lrc(day, e0_sm[w], sde0_sm[w], rref_win[w], last_good)
        if res is None:
            continue
        last_good = res['theta']
        i_mean_list.append(i_mean_global)
        params_list.append(res['theta'])
        central_list.append(int(i_central[w]))

    if not params_list:
        warn("Daytime partitioning (ReddyProc): no light-response curve could be "
             "fitted; record left unpartitioned.", verbose=verbose)
        return out

    params = np.array(params_list)
    i_mean = np.array(i_mean_list, int)

    # --- Stage 4: interpolate Reco/GPP to every record ---
    reco, gpp = _interpolate_fluxes(i_mean, params, rg, vpd, ta, n)
    out['RECO_DT_RP'] = reco
    out['GPP_DT_RP'] = gpp

    # report LRC parameters at the central record of each window (like REddyProc)
    for c, p in zip(central_list, params_list):
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
    sd[fill] = np.maximum(SD_MINSD, np.abs(nee[fill] * SD_PERC))
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
        >>> part = DaytimePartitioningReddyProc(
        ...     nee=df['NEE_orig'], ta=df['Tair_f'], vpd=df['VPD_f'],
        ...     sw_in=df['Rg_f'], lat=46.815, lon=9.855, utc_offset=1)
        >>> part.run()
        >>> results = part.results   # DataFrame with RECO_DT_RP, GPP_DT_RP, ...
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
