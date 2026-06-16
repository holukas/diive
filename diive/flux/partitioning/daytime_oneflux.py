"""
DAYTIME PARTITIONING ONEFLUX: NEE -> GPP + RECO (Lasslop et al. 2010)
====================================================================

Faithful, vectorized Python port of the ONEFlux *daytime* partitioning
(``flux_part_gl2010`` in ``oneflux.partition.daytime``), the Lasslop et al.
(2010) light-response-curve (LRC) method as implemented in the ONEFlux /
FLUXNET2015 processing pipeline. It is the daytime ONEFlux companion to the
nighttime ONEFlux port (:mod:`diive.flux.partitioning.nighttime_oneflux`) and
the daytime REddyProc port (:mod:`diive.flux.partitioning.daytime_reddyproc`),
and emits ``*_DT_OF`` columns so all variants coexist in one dataframe.

The daytime method fits, for short overlapping windows, a rectangular-hyperbola
light-response curve to *daytime* NEE,

    NEE = -GPP + RECO,
    GPP = alpha * beta * f(VPD) * Rg / (alpha * Rg + beta * f(VPD)),
    f(VPD) = min(exp(-k * (VPD - VPD0)), 1),
    RECO = RRef * exp(E0 * (1/(Tref - T0) - 1/(Tair - T0)))   (Lloyd & Taylor),

with parameters ``alpha`` (initial slope), ``beta`` (GPP saturation), ``k`` (VPD
sensitivity) and ``RRef`` (basal respiration). The temperature sensitivity
``E0`` is *not* fitted on daytime data; it is estimated beforehand from
nighttime NEE in a wider window and held fixed while the light parameters are
optimized.

Algorithm (per calendar year, ONEFlux defaults):

1. Per-record NEE uncertainty via a Reichstein-style marginal-distribution
   look-up gap-fill (``uncert_via_gapFill``) -> the per-record standard
   deviation used to weight the fits.
2. For each 4-day window (2-day step, indexed by day-of-year): fit the
   nighttime ``E0`` (Lloyd-Taylor) on the surrounding ~12-day nighttime data
   (``Rg <= 4``), then fit the daytime LRC (``Rg > 4``) with ``E0`` fixed, from
   three ``beta`` starting guesses, choosing the lowest-RMSE fit. A cascade of
   model variants handles degenerate parameters (drop the VPD term, fix
   ``alpha`` from the previous window, or fall back to respiration only).
3. Predict RECO and GPP for every record by interpolating the two neighboring
   windows' parameter sets with distance-based weights.
4. Propagate the GPP standard error from the fit covariance via the Jacobian.

The day/night split for *fitting* is ONEFlux's measured-radiation threshold
(``Rg <= 4`` night, ``Rg > 4`` day); the method does **not** use solar geometry
or latitude.

This port mirrors the ONEFlux reference function-for-function. The fits use the
same penalized (Bayesian-prior) least squares solved with the same SciPy
``leastsq`` (step-bound ``factor=0.25``), so per-window parameters reproduce a
native ONEFlux run on identical arrays to high precision.

Reference:
    Lasslop, G. et al. (2010). Separation of net ecosystem exchange into
    assimilation and respiration using a light response curve approach: critical
    issues and global evaluation. Global Change Biology, 16(1), 187-208.
    https://doi.org/10.1111/j.1365-2486.2009.02041.x

    Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux
    processing pipeline for eddy covariance data. Scientific Data, 7, 225.
    https://doi.org/10.1038/s41597-020-0534-3

Example: ``examples/flux/partitioning/partitioning_daytime_oneflux.py``

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats
from scipy.optimize import leastsq

from diive.core.utils.console import info, warn, success
from diive.flux.partitioning._report import partitioning_report

# Lloyd & Taylor reference/regression temperatures (degC), as in ONEFlux.
TREF = 15.0
T0 = -46.02
VPD0 = 10.0  # hPa, Lasslop et al. 2010

# ONEFlux missing-value sentinel and "is valid" test (value > NAN_TEST).
NAN = -9999.0
NAN_TEST = -9990.0

# Window geometry (ONEFlux estimate_parasets defaults).
WINSIZE = 4                 # window width in days
FGUESS0 = [0.01, 30.0, 0.0, 5.0, 100.0]  # [alpha, beta, k, rref, e0] start
BETAFAC = (0.5, 1.0, 2.0)   # three beta starting-guess multipliers
E0_MIN, E0_MAX = 50.0, 400.0
DAY_RG_THRESHOLD = 4.0      # Rg > 4 day, Rg <= 4 night (measured radiation)

# Prior standard deviations (sigm) per model, ONEFlux estimate_parasets.
SIGM_LLOYDTEMP = np.array([800.0, 1000.0])
SIGM_LLOYDVPD = np.array([10.0, 600.0, 50.0, 80.0])
SIGM_LLOYD = np.array([10.0, 600.0, 80.0])
SIGM_LLOYD_AFIX = np.array([600.0, 80.0])
SIGM_LLOYDVPD_AFIX = np.array([600.0, 50.0, 80.0])
SIGM_E0FIX = np.array([80.0])

# scipy leastsq tuning (faithful to ONEFlux library.least_squares).
_STEP_BOUND_FACTOR = 0.25
_NO_CONVERGENCE_RETRY = 20

# uncert_via_gapFill tolerances.
_RG_TOL = 50.0
_TA_TOL = 2.5
_VPD_TOL = 5.0
_TW_ORIG = 14   # base window (days)


def _notnan(a):
    return a > NAN_TEST


class _BrokenWindow(Exception):
    """Raised when a window's fit is singular (no covariance); the window is
    skipped, exactly as ONEFlux removes such windows via its error file."""


# --------------------------------------------------------------------------- #
# Eco/geo model functions (port of oneflux.partition.ecogeo)
# --------------------------------------------------------------------------- #
def _lloyd_taylor_dt(ta, rref, e0):
    return rref * np.exp(e0 * ((1.0 / (TREF - T0)) - (1.0 / (ta - T0))))


def _gpp_vpd(rg, vpd, par):
    alpha, beta, k = par[0], par[1], par[2]
    if beta == 0:
        return np.zeros(len(rg))
    with np.errstate(over='ignore', invalid='ignore'):
        min_arr = np.minimum(np.exp(-1.0 * k * (vpd - VPD0)), 1.0)
        return alpha * beta * min_arr * rg / (alpha * rg + beta * min_arr)


def _hlrc_lloyd(rg, ta, e0, par):
    alpha, beta, rd15 = par[0], par[1], par[2]
    resp = rd15 * np.exp(e0 * ((1.0 / (TREF - T0)) - (1.0 / (ta - T0))))
    return -1.0 * alpha * beta * rg / (alpha * rg + beta) + resp


def _hlrc_lloydvpd(rg, ta, e0, vpd, par):
    alpha, beta, k, rd15 = par[0], par[1], par[2], par[3]
    resp = rd15 * np.exp(e0 * ((1.0 / (TREF - T0)) - (1.0 / (ta - T0))))
    with np.errstate(over='ignore', invalid='ignore'):
        min_arr = np.minimum(np.exp(-1.0 * k * (vpd - VPD0)), 1.0)
        return -1.0 * alpha * beta * min_arr * rg / (alpha * rg + beta * min_arr) + resp


def _hlrc_lloyd_afix(rg, ta, e0, alpha, par):
    beta, rd15 = par[0], par[1]
    resp = rd15 * np.exp(e0 * ((1.0 / (TREF - T0)) - (1.0 / (ta - T0))))
    return -1.0 * alpha * beta * rg / (alpha * rg + beta) + resp


def _hlrc_lloydvpd_afix(rg, ta, e0, vpd, alpha, par):
    beta, k, rd15 = par[0], par[1], par[2]
    resp = rd15 * np.exp(e0 * ((1.0 / (TREF - T0)) - (1.0 / (ta - T0))))
    with np.errstate(over='ignore', invalid='ignore'):
        min_arr = np.minimum(np.exp(-1.0 * k * (vpd - VPD0)), 1.0)
        return -1.0 * alpha * beta * min_arr * rg / (alpha * rg + beta * min_arr) + resp


def _lloydt_e0fix(ta, e0, par):
    rd15 = par[0] if hasattr(par, '__len__') else par
    return rd15 * np.exp(e0 * ((1.0 / (TREF - T0)) - (1.0 / (ta - T0))))


def _build_predict(lts_func, ind):
    """Return a fast ``predict(par)`` closure for the model named ``lts_func``.

    Pieces that do not change across ``leastsq`` residual evaluations are computed
    once here: for the daytime models the Lloyd-Taylor respiration factor
    ``exp(E0*(...))`` (E0 is held fixed) and ``vpd - VPD0``. Reusing the
    deterministic ``exp`` is bit-for-bit identical to recomputing it each call,
    but removes the dominant per-evaluation cost (the non-VPD models become
    ``exp``-free in the residual loop).
    """
    if lts_func == "LloydTemp":
        # E0 is a fitted parameter here, so only the temperature offset is fixed.
        tdiff = (1.0 / (TREF - T0)) - (1.0 / (ind['ta'] - T0))

        def predict(par):
            return par[0] * np.exp(par[1] * tdiff)
        return predict

    ta, e0 = ind['ta'], ind['e0']
    rg = ind.get('rg')  # absent for LloydT_E0fix (respiration only)
    tfac = np.exp(e0 * ((1.0 / (TREF - T0)) - (1.0 / (ta - T0))))  # E0 fixed
    vpdm = ind['vpd'] - VPD0 if 'vpd' in ind else None
    alpha_fix = ind.get('alpha')

    if lts_func == "HLRC_Lloyd":
        def predict(par):
            return -1.0 * par[0] * par[1] * rg / (par[0] * rg + par[1]) + par[2] * tfac
    elif lts_func == "HLRC_LloydVPD":
        def predict(par):
            with np.errstate(over='ignore', invalid='ignore'):
                m = np.minimum(np.exp(-1.0 * par[2] * vpdm), 1.0)
                return (-1.0 * par[0] * par[1] * m * rg
                        / (par[0] * rg + par[1] * m) + par[3] * tfac)
    elif lts_func == "HLRC_Lloyd_afix":
        def predict(par):
            return (-1.0 * alpha_fix * par[0] * rg
                    / (alpha_fix * rg + par[0]) + par[1] * tfac)
    elif lts_func == "HLRC_LloydVPD_afix":
        def predict(par):
            with np.errstate(over='ignore', invalid='ignore'):
                m = np.minimum(np.exp(-1.0 * par[1] * vpdm), 1.0)
                return (-1.0 * alpha_fix * par[0] * m * rg
                        / (alpha_fix * rg + par[0] * m) + par[2] * tfac)
    elif lts_func == "LloydT_E0fix":
        def predict(par):
            return par[0] * tfac
    else:
        raise ValueError(lts_func)
    return predict


# --------------------------------------------------------------------------- #
# Least-squares core (port of oneflux.partition.library.least_squares / nlinlts2)
# --------------------------------------------------------------------------- #
def _cov2cor(cov):
    n = cov.shape[0]
    cor = np.zeros((n, n))
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(n):
            for j in range(n):
                cor[j, i] = cov[j, i] / np.sqrt(cov[i, i] * cov[j, j])
    return cor


def _least_squares(func, x0, entries, iterations, stop=False):
    """Port of library.least_squares: leastsq(factor=0.25) + retry + cov."""
    pars, cov_x, info_dict, _msg, ier = leastsq(
        func, x0, full_output=True, maxfev=iterations, factor=_STEP_BOUND_FACTOR)
    if ier != 1 and info_dict['nfev'] >= iterations and not stop:
        return _least_squares(func, x0, entries, iterations * _NO_CONVERGENCE_RETRY,
                              stop=True)
    residuals = info_dict['fvec']
    npar = len(x0)
    if entries > npar and cov_x is not None:
        s_squared = (residuals ** 2).sum() / (entries - npar)
        cov = cov_x * s_squared
        cor = _cov2cor(cov)
        std = np.sqrt(np.abs(np.array([cov[i, i] for i in range(npar)])))
    else:
        std = np.full(npar, np.nan)
        cov = None
        cor = None
    return np.asarray(pars, float), std, ier, residuals, cov, cor


def _rmse(nee, prediction):
    """Port of library.root_mean_sq_error (trim_perc=0)."""
    abs_err = np.abs(nee - prediction)
    return float(np.sqrt((abs_err * abs_err).sum() / len(nee)))


def _nlinlts2(lts_func, dep, indeps, npara, xguess, mprior, sigm, sigd):
    """Port of library.nlinlts2: penalized NLS via scipy leastsq.

    ``dep`` is the dependent array (nee_f), ``indeps`` the independent-variable
    dict, ``sigd`` the per-record data sigma. Returns a result dict with fitted
    parameters, standard errors, covariance/correlation, residuals and RMSE, or
    ``cov_matrix=None`` when the fit is singular / underdetermined.
    """
    # validity of independent vars (all finite within the subset)
    nonnan_indep = np.ones(dep.size, dtype=bool)
    for key in indeps:
        nonnan_indep &= _notnan(np.atleast_1d(indeps[key]) if np.ndim(indeps[key]) else
                                np.full(dep.size, indeps[key]))
    fail = dict(params=np.full(npara, NAN), std=np.full(npara, np.nan),
                cov_matrix=None, cor_matrix=None,
                residuals=np.full(int(nonnan_indep.sum()), np.nan), rmse=0.0)
    if int(nonnan_indep.sum()) < npara * 3:
        return fail
    nonnan_dep = _notnan(dep)
    if int((nonnan_indep & nonnan_dep).sum()) < npara * 3:
        return fail

    clean_dep = dep.copy()
    clean_dep[~nonnan_indep] = NAN
    nonnan_clean = _notnan(clean_dep)
    predict = _build_predict(lts_func, indeps)

    def resid(par):
        prediction = predict(par)
        r = (clean_dep - prediction) / sigd
        r[~nonnan_clean] = 0.0
        pres = (par - mprior) / sigm
        return np.append(r, pres)

    pars, std, ier, residuals, cov, cor = _least_squares(
        resid, list(xguess), entries=len(clean_dep),
        iterations=1000 * (len(clean_dep) + 1))
    prediction = predict(pars)
    rmse = _rmse(clean_dep, prediction)
    return dict(params=pars, std=std, cov_matrix=cov, cor_matrix=cor,
                residuals=residuals, rmse=rmse)


def _fit(lts_func, dep, indeps, npara, xguess, mprior, sigm, sigd):
    """nlinlts2 wrapper: raise _BrokenWindow on a singular fit (ONEFlux raises
    ONEFluxPartitionBrokenOptError there; we skip the window instead)."""
    res = _nlinlts2(lts_func, dep, indeps, npara, xguess, mprior, sigm, sigd)
    if res['cov_matrix'] is None or res['cor_matrix'] is None:
        raise _BrokenWindow(lts_func)
    return res


def _check_parameters(p):
    """Port of library.check_parameters. p = [alpha,beta,k,rref,e0,*se...]."""
    is_ok = 0
    if (p[0] >= 0) and (p[0] < 0.22) and (p[1] >= 0) and (p[1] < 250) \
            and (p[2] >= 0) and (p[3] > 0) and (p[4] >= 50) and (p[4] <= 400) \
            and (p[0] != FGUESS0[0]):
        is_ok = 1
    if (p[1] > 100) and (p[1] < p[6]):
        is_ok = 0
    return is_ok


def _percentiles_fn(values_arr, percs):
    """Port of daytime.percentiles_fn for a single column (no NA removal)."""
    n = values_arr.shape[0]
    if n <= 0:
        return -1
    order = np.argsort(values_arr)
    out = []
    for v in percs:
        if v <= 0.5:
            idx = int(v * n)
        else:
            idx = int(v * (n + 1))
        if idx >= n:
            idx = n - 1
        out.append(values_arr[order[idx]])
    return np.array(out)


# --------------------------------------------------------------------------- #
# Stage A: per-record NEE uncertainty (port of daytime.uncert_via_gapFill)
# --------------------------------------------------------------------------- #
def _uncert_via_gapfill(nee, rg, ta, vpd, hr, nperday, longest_marginal_gap=60):
    """Reichstein-style marginal-distribution look-up gap-fill of NEE.

    Returns ``nee_fs_unc`` (the per-record standard deviation of the look-up
    neighbours), the weight ``sigd`` used by the fits. All inputs use the
    ONEFlux ``-9999`` sentinel; ``rg``/``ta`` are measured (gappy), ``vpd``
    filled.
    """
    n = nee.size
    tofill_orig = nee.copy()
    filled_val = np.full(n, NAN)
    filled_s = np.full(n, NAN)
    tofill = np.full(n, NAN)

    large = np.zeros(n)
    ok = np.where(tofill_orig > NAN_TEST)[0]
    if ok.size == 0:
        return filled_s
    firstvalid, lastvalid = int(ok.min()), int(ok.max())
    if firstvalid > (48 * longest_marginal_gap):
        large[:(firstvalid + 1 - (48 * longest_marginal_gap))] = 1
    if lastvalid < (n - (48 * longest_marginal_gap)):
        large[(lastvalid + (48 * longest_marginal_gap)):] = 1

    # The window offset pattern depends only on t_window (constant within a
    # pass), so build it once per distinct t_window instead of per gap.
    _offset_cache = {}

    def window_idx(index, t_window):
        off = _offset_cache.get(t_window)
        if off is None:
            off = np.append(-np.arange(t_window / 2.0 * nperday),
                            np.arange(t_window / 2.0 * nperday - 1) + 1)
            _offset_cache[t_window] = off
        w = index + off
        np.clip(w, 0, n - 1, out=w)
        return w.astype(int)

    def fill_at(index, sel, method_window):
        vals = tofill_orig[sel]
        filled_val[index] = stats.tmean(vals)
        filled_s[index] = stats.tstd(vals)

    def gaps():
        return np.where((tofill < NAN_TEST) & (large == 0))[0]

    # Loop 1: meteo LUT (Rg, Ta, VPD), windows 14 & 28 days
    it = 0
    while True:
        ko = gaps()
        t_window = (it + 1) * _TW_ORIG
        if ko.size == 0:
            return filled_s
        for index in ko:
            w = window_idx(index, t_window)
            avg = np.where(tofill_orig[w] > NAN_TEST)[0]
            if avg.size > 9:
                w = w[avg]
                sel = np.where((np.abs(ta[w] - ta[index]) < _TA_TOL) &
                               (np.abs(rg[w] - rg[index]) < max(min(_RG_TOL, rg[index]), 20)) &
                               (np.abs(vpd[w] - vpd[index]) < _VPD_TOL) &
                               (rg[w] > NAN_TEST) & (vpd[w] > NAN_TEST) & (ta[w] > NAN_TEST))[0]
                if sel.size > 9:
                    fill_at(index, w[sel], t_window)
        tofill[:] = filled_val
        it += 1
        if it > 1:
            break

    # Loop 2: Rg-only LUT, window 14 days
    tofill[:] = filled_val
    it = 0
    while True:
        t_window = (it + 1) * _TW_ORIG
        ko = gaps()
        if ko.size == 0:
            return filled_s
        for index in ko:
            w = window_idx(index, t_window)
            sel = np.where((np.abs(rg[w] - rg[index]) < max(min(_RG_TOL, rg[index]), 20)) &
                           (tofill_orig[w] > NAN_TEST) & (rg[w] > NAN_TEST))[0]
            if sel.size > 9:
                fill_at(index, w[sel], t_window)
        tofill[:] = filled_val
        it += 1
        if it > 0:
            break

    # Loop 3: diurnal +-1h, windows 1, 3, 5 days
    tofill[:] = filled_val
    it = 0
    while True:
        ko = gaps()
        t_window = (2 * it + 1) * 1
        if ko.size == 0:
            return filled_s
        for index in ko:
            w = window_idx(index, t_window)
            sel = np.where((np.abs(hr[w] - hr[index]) < 1.1) & (tofill_orig[w] > NAN_TEST))[0]
            if sel.size > 9:
                fill_at(index, w[sel], t_window)
        tofill[:] = filled_val
        it += 1
        if it > 2:
            break

    # Loop 4: meteo LUT (large windows, Cat. B)
    it = 2
    while True:
        ko = gaps()
        t_window = (it + 1) * _TW_ORIG
        if ko.size == 0:
            return filled_s
        for index in ko:
            w = window_idx(index, t_window)
            avg = np.where(tofill_orig[w] > NAN_TEST)[0]
            if avg.size > 9:
                w = w[avg]
                sel = np.where((np.abs(ta[w] - ta[index]) < _TA_TOL) &
                               (np.abs(rg[w] - rg[index]) < max(min(_RG_TOL, rg[index]), 20)) &
                               (np.abs(vpd[w] - vpd[index]) < _VPD_TOL) &
                               (rg[w] > NAN_TEST) & (vpd[w] > NAN_TEST) & (ta[w] > NAN_TEST))[0]
                if sel.size > 9:
                    fill_at(index, w[sel], t_window)
        tofill[:] = filled_val
        it += 1
        if it > 10:
            break

    # Loop 5: Rg-only LUT (large windows, Cat. B)
    tofill[:] = filled_val
    it = 1
    while True:
        t_window = (it + 1) * _TW_ORIG
        ko = gaps()
        if ko.size == 0:
            return filled_s
        for index in ko:
            w = window_idx(index, t_window)
            sel = np.where((np.abs(rg[w] - rg[index]) < max(min(_RG_TOL, rg[index]), 20)) &
                           (tofill_orig[w] > NAN_TEST) & (rg[w] > NAN_TEST))[0]
            if sel.size > 9:
                fill_at(index, w[sel], t_window)
        tofill[:] = filled_val
        it += 1
        if it > 10:
            break

    # Loop 6: diurnal (large windows, Cat. C)
    tw_half = _TW_ORIG * 0.5
    tofill[:] = filled_val
    it = 0
    while True:
        ko = gaps()
        t_window = (it + 1) * tw_half
        if ko.size == 0:
            return filled_s
        for index in ko:
            w = window_idx(index, t_window)
            sel = np.where((np.abs(hr[w] - hr[index]) < 1.1) & (tofill_orig[w] > NAN_TEST))[0]
            if sel.size > 9:
                fill_at(index, w[sel], t_window)
        tofill[:] = filled_val
        it += 1
        if it > 60:
            break

    return filled_s


# --------------------------------------------------------------------------- #
# Stage B: per-window parameter estimation (port of daytime.estimate_parasets)
# --------------------------------------------------------------------------- #
def _estimate_parasets(D, nperday, verbose=1):
    """Per-window LRC parameter estimation.

    ``D`` holds the year's arrays: nee_f, nee_fqc, tair_f, rg_f, vpd_f, rg_meas
    (measured, -9999 sentinel), julday, nee_fs_unc. Returns lists of accepted
    windows: params (10,), whichmodel, cov (4x4), res_cor, and the 3 ind rows.
    """
    nee_f = D['nee_f']
    nee_fqc = D['nee_fqc']
    tair_f = D['tair_f']
    rg_f = D['rg_f']
    vpd_f = D['vpd_f']
    rg_meas = D['rg_meas']
    julday = D['julday']
    sigd_all = D['nee_fs_unc']

    n_parasets = (365 // WINSIZE) * 2
    fguess = list(FGUESS0)

    params_ok, ind_ok, whichmodel_ok, jtj_ok, rescor_ok = [], [], [], [], []
    i_ok = 0
    lloydtemp_e0 = None

    for i in range(n_parasets):
        day_begin = i * WINSIZE / 2.0
        day_end = day_begin + WINSIZE
        day_begin2 = (i - 2) * WINSIZE / 2.0 if i > 1 else 0
        day_end2 = (i + 2) * WINSIZE / 2.0 + WINSIZE if i < n_parasets - 2 else float(np.max(julday))

        central = int((day_begin + WINSIZE / 2.0) * 48.0)
        ind_rows = np.array([central, central, central], dtype=float)

        measured = (nee_fqc == 0)
        sub_m = (julday > day_begin) & (julday <= day_end) & measured
        subn_m = (julday > day_begin2) & (julday <= day_end2) & measured & (rg_meas <= DAY_RG_THRESHOLD)
        subd_m = (julday > day_begin) & (julday <= day_end) & measured & (rg_meas > DAY_RG_THRESHOLD)

        subn_sigd = sigd_all[subn_m].copy()
        subd_sigd = sigd_all[subd_m].copy()
        if subn_sigd.size and np.min(subn_sigd) < 0:
            subn_sigd[:] = 1
        if subd_sigd.size and np.min(subd_sigd) < 0:
            subd_sigd[:] = 1

        E0set = 0
        if subn_m.sum() <= 10 and i_ok > 0 and lloydtemp_e0 is not None:
            lloydtemp_e0 = params_ok[i_ok - 1][4]
            e0_se = params_ok[i_ok - 1][9]
            ind_rows[0] = ind_ok[i_ok - 1][0]
            E0set = 1

        if not ((subn_m.sum() > 10 or E0set == 1) and subd_m.sum() > 10):
            continue

        try:
            percs = _percentiles_fn(nee_f[sub_m], [0.03, 0.97])
            beta = abs(percs[0] - percs[1])
            rb = float(np.average(nee_f[subn_m]))
            fguess[3] = rb

            if E0set == 0:
                res = _fit("LloydTemp", nee_f[subn_m], {'ta': tair_f[subn_m]},
                           npara=2, xguess=fguess[3:5],
                           mprior=np.array(fguess[3:5], dtype='f4'),
                           sigm=SIGM_LLOYDTEMP, sigd=subn_sigd)
                rref, e0 = float(res['params'][0]), float(res['params'][1])
                e0_se = float(res['std'][1])
                lloydtemp_e0 = e0
                if e0 < E0_MIN or e0 > E0_MAX:
                    if i_ok > 0:
                        e0 = params_ok[i_ok - 1][4]
                        e0_se = params_ok[i_ok - 1][9]
                        ind_rows[0] = ind_ok[i_ok - 1][0]
                    elif e0 < E0_MIN:
                        e0, e0_se = E0_MIN, np.nan
                    elif e0 > E0_MAX:
                        e0, e0_se = E0_MAX, np.nan
            else:
                e0 = lloydtemp_e0

            # ONEFlux stores the fixed E0 driver column as float32.
            e0_arr_d = np.full(int(subd_m.sum()), e0, dtype=np.float32)

            pj = np.zeros((3, 10))
            indj = np.tile(ind_rows, (3, 1))
            rmse = np.zeros(3)
            wm = np.zeros(3, dtype=int)
            jtj = np.zeros((3, 4, 4))
            rescor = np.zeros(3)

            rg_d, ta_d, vpd_d = rg_f[subd_m], tair_f[subd_m], vpd_f[subd_m]
            nee_d = nee_f[subd_m]
            ndd = nee_d.size

            for j in range(3):
                fguess[1] = beta * BETAFAC[j]

                r = _fit("HLRC_LloydVPD", nee_d,
                         {'rg': rg_d, 'ta': ta_d, 'e0': e0_arr_d, 'vpd': vpd_d},
                         npara=4, xguess=fguess[0:4],
                         mprior=np.array(fguess[0:4], dtype='f4'),
                         sigm=SIGM_LLOYDVPD, sigd=subd_sigd)
                a, b, k, rd = r['params']
                wm[j] = 0
                rescor[j] = (r['residuals'] ** 2).sum() / (len(r['residuals']) - 4)
                pj[j] = [a, b, k, rd, e0, r['std'][0], r['std'][1], r['std'][2], r['std'][3], e0_se]
                rmse[j] = r['rmse']
                jtj[j] = r['cov_matrix']
                if pj[j, 2] == 0:
                    wm[j] = 1
                    jt = np.zeros((4, 4))
                    cov = r['cov_matrix']
                    jt[0, 0], jt[0, 1], jt[1, 0], jt[1, 1] = cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
                    jt[0, 2], jt[1, 2], jt[2, 2] = cov[0, 3], cov[1, 3], cov[3, 3]
                    jt[2, 0], jt[2, 1] = cov[3, 0], cov[3, 1]
                    jtj[j] = jt

                # k < 0 -> drop VPD effect (HLRC_Lloyd)
                if pj[j, 2] < 0:
                    r = _fit("HLRC_Lloyd", nee_d,
                             {'rg': rg_d, 'ta': ta_d, 'e0': e0_arr_d},
                             npara=3, xguess=[fguess[0], fguess[1], fguess[3]],
                             mprior=np.array([fguess[0], fguess[1], fguess[3]], dtype='f4'),
                             sigm=SIGM_LLOYD, sigd=subd_sigd)
                    a, b, rd = r['params']
                    wm[j] = 1
                    rescor[j] = (r['residuals'] ** 2).sum() / (len(r['residuals']) - 3)
                    pj[j] = [a, b, 0, rd, e0, r['std'][0], r['std'][1], 0, r['std'][2], e0_se]
                    rmse[j] = r['rmse']
                    jtj[j] = 0
                    jtj[j, 0:3, 0:3] = r['cov_matrix']

                    # alpha > 0.22 -> fix alpha from last window (HLRC_Lloyd_afix)
                    if (pj[j, 0] > 0.22) and i_ok > 0 and params_ok[i_ok - 1][0] > 0:
                        alpha = params_ok[i_ok - 1][0]
                        indj[j, 1] = ind_ok[i_ok - 1][1]
                        alpha_d = np.full(ndd, alpha, dtype=np.float32)
                        r = _fit("HLRC_Lloyd_afix", nee_d,
                                 {'rg': rg_d, 'ta': ta_d, 'e0': e0_arr_d, 'alpha': alpha_d},
                                 npara=2, xguess=[fguess[1], fguess[3]],
                                 mprior=np.array([fguess[1], fguess[3]], dtype='f4'),
                                 sigm=SIGM_LLOYD_AFIX, sigd=subd_sigd)
                        b, rd = r['params']
                        wm[j] = 2
                        rescor[j] = (r['residuals'] ** 2).sum() / (len(r['residuals']) - 2)
                        pj[j] = [alpha, b, 0, rd, e0, np.nan, r['std'][0], 0, r['std'][1], e0_se]
                        rmse[j] = r['rmse']
                        jtj[j] = 0
                        jtj[j, 0:2, 0:2] = r['cov_matrix']

                # alpha > 0.22 (VPD branch) -> fix alpha from last window
                elif (pj[j, 0] > 0.22) and i_ok > 0 and params_ok[i_ok - 1][0] > 0:
                    alpha = params_ok[i_ok - 1][0]
                    indj[j, 1] = ind_ok[i_ok - 1][1]
                    alpha_d = np.full(ndd, alpha, dtype=np.float32)
                    r = _fit("HLRC_LloydVPD_afix", nee_d,
                             {'rg': rg_d, 'ta': ta_d, 'e0': e0_arr_d, 'vpd': vpd_d, 'alpha': alpha_d},
                             npara=3, xguess=[fguess[1], fguess[2], fguess[3]],
                             mprior=np.array([fguess[1], fguess[2], fguess[3]], dtype='f4'),
                             sigm=SIGM_LLOYDVPD_AFIX, sigd=subd_sigd)
                    b, k, rd = r['params']
                    wm[j] = 3
                    rescor[j] = (r['residuals'] ** 2).sum() / (len(r['residuals']) - 3)
                    pj[j] = [alpha, b, k, rd, e0, 0, r['std'][0], r['std'][1], r['std'][2], e0_se]
                    rmse[j] = r['rmse']
                    jtj[j] = 0
                    jtj[j, 0:3, 0:3] = r['cov_matrix']
                    if pj[j, 2] == 0:
                        wm[j] = 2
                        jt = np.zeros((4, 4))
                        cov = r['cov_matrix']
                        jt[0, 0], jt[0, 1], jt[1, 0], jt[1, 1] = cov[0, 0], cov[2, 0], cov[0, 2], cov[2, 2]
                        jtj[j] = jt
                    if pj[j, 2] < 0:
                        r = _fit("HLRC_Lloyd_afix", nee_d,
                                 {'rg': rg_d, 'ta': ta_d, 'e0': e0_arr_d, 'alpha': alpha_d},
                                 npara=2, xguess=[fguess[1], fguess[3]],
                                 mprior=np.array([fguess[1], fguess[3]], dtype='f4'),
                                 sigm=SIGM_LLOYD_AFIX, sigd=subd_sigd)
                        b, rd = r['params']
                        wm[j] = 2
                        rescor[j] = (r['residuals'] ** 2).sum() / (len(r['residuals']) - 2)
                        pj[j] = [alpha, b, 0, rd, e0, 0, r['std'][0], 0, r['std'][1], e0_se]
                        rmse[j] = r['rmse']
                        jtj[j] = 0
                        jtj[j, 0:2, 0:2] = r['cov_matrix']

                # alpha or beta < 0 -> respiration only (LloydT_E0fix)
                if pj[j, 0] < 0 or pj[j, 1] < 0:
                    r = _fit("LloydT_E0fix", nee_d, {'ta': ta_d, 'e0': e0_arr_d},
                             npara=1, xguess=[fguess[3]],
                             mprior=np.array([fguess[3]], dtype='f4'),
                             sigm=SIGM_E0FIX, sigd=subd_sigd)
                    rd = r['params'][0]
                    wm[j] = 4
                    rescor[j] = (r['residuals'] ** 2).sum() / (len(r['residuals']) - 1)
                    pj[j] = [0, 0, 0, rd, e0, 0, 0, 0, r['std'][0], e0_se]
                    rmse[j] = r['rmse']
                    jtj[j] = 0
                    jtj[j, 0, 0] = np.asarray(r['cov_matrix']).flatten()[0]

                if _check_parameters(pj[j]) == 0:
                    rmse[j] = 9999.0
            # end for j

            jmin = int(np.where(rmse == np.min(np.abs(rmse)))[0][0])
            if _check_parameters(pj[jmin]) == 1:
                params_ok.append(pj[jmin].copy())
                ind_ok.append(indj[jmin].copy())
                whichmodel_ok.append(int(wm[jmin]))
                jtj_ok.append(jtj[jmin].copy())
                rescor_ok.append(float(rescor[jmin]))
                i_ok += 1
        except _BrokenWindow:
            continue
    # end for i

    return params_ok, ind_ok, whichmodel_ok, jtj_ok, rescor_ok


# --------------------------------------------------------------------------- #
# Stage C: interpolate Reco/GPP to every record (port of daytime.compute_flux)
# --------------------------------------------------------------------------- #
def _compute_flux(n, tair_f, rg_f, vpd_f, params_ok, central):
    nwin = len(params_ok)
    reco_mat = np.full((nwin, n), NAN)
    gpp_mat = np.full((nwin, n), NAN)
    ind = np.arange(n)
    for i in range(nwin):
        if i == 0:
            lo, hi = 0, central[i + 1]
            sub = (ind >= lo) & (ind < hi)
        elif i == nwin - 1:
            lo, hi = central[i - 1], np.max(ind)
            sub = (ind >= lo) & (ind <= hi)
        else:
            lo, hi = central[i - 1], central[i + 1]
            sub = (ind >= lo) & (ind < hi)
        p = params_ok[i]
        reco_mat[i, sub] = _lloyd_taylor_dt(tair_f[sub], p[3], p[4])
        gpp_mat[i, sub] = _gpp_vpd(rg_f[sub], vpd_f[sub], p[0:3])

    reco = np.zeros(n)
    gpp = np.zeros(n)
    pf1 = np.zeros(n)
    pf2 = np.zeros(n)
    for j in range(n):
        cov = np.where(reco_mat[:, j] > NAN)[0]
        count = cov.size
        if count > 1:
            c0, c1 = central[cov[0]], central[cov[1]]
            w1 = (c1 - j) / (c1 - c0)
            w2 = (j - c0) / (c1 - c0)
            reco[j] = reco_mat[cov[0], j] * w1 + reco_mat[cov[1], j] * w2
            gpp[j] = gpp_mat[cov[0], j] * w1 + gpp_mat[cov[1], j] * w2
            pf1[j] = abs(c1 - j)
            pf2[j] = abs(j - c0)
        elif count == 1:
            reco[j] = reco_mat[cov[0], j]
            gpp[j] = gpp_mat[cov[0], j]
            pf1[j] = abs(central[cov[0]] - j)
            pf2[j] = j if cov[0] == 0 else (n - 1 - j)
        else:
            reco[j] = NAN
            gpp[j] = NAN
    return reco, gpp, pf1, pf2


# --------------------------------------------------------------------------- #
# Stage D: GPP standard error (port of daytime.compute_var / varpred / jacobian)
# --------------------------------------------------------------------------- #
def _jacobian(func, rg, ta, e0, vpd, alpha, params):
    params = np.atleast_1d(np.asarray(params, float))
    nf = (_model_predict_var(func, rg, ta, e0, vpd, alpha, params)).size
    npp = len(params)
    jac = np.zeros((npp, nf))
    delta = 1.0e-3
    for p in range(npp):
        pp = params.copy()
        pp[p] = params[p] + delta * abs(params[p])
        pm = params.copy()
        pm[p] = params[p] - delta * abs(params[p])
        fp = _model_predict_var(func, rg, ta, e0, vpd, alpha, pp)
        fm = _model_predict_var(func, rg, ta, e0, vpd, alpha, pm)
        jac[p, :] = (fp - fm) / (pp[p] - pm[p])
    return jac


def _model_predict_var(func, rg, ta, e0, vpd, alpha, par):
    if func == "HLRC_Lloyd":
        return _hlrc_lloyd(rg, ta, e0, par)
    if func == "HLRC_LloydVPD":
        return _hlrc_lloydvpd(rg, ta, e0, vpd, par)
    if func == "HLRC_Lloyd_afix":
        return _hlrc_lloyd_afix(rg, ta, e0, alpha, par)
    if func == "HLRC_LloydVPD_afix":
        return _hlrc_lloydvpd_afix(rg, ta, e0, vpd, alpha, par)
    if func == "LloydT_E0fix":
        return _lloydt_e0fix(ta, e0, par)
    raise ValueError(func)


def _varpred(func, rg, ta, e0, vpd, alpha, jtj_inv, optpara, res):
    jac = _jacobian(func, rg, ta, e0, vpd, alpha, optpara)
    jtj_inv = np.asarray(jtj_inv)
    if jtj_inv.size == 1:
        return (jac * float(jtj_inv) * jac * res).ravel()
    x = np.dot(jac.T, jtj_inv)
    y = np.dot(x, jac)
    return np.diagonal(y * res)


def _compute_var(n, tair_f, rg_f, vpd_f, params_ok, central, whichmodel, jtj_ok, rescor):
    nwin = len(params_ok)
    var_mat = np.full((nwin, n), NAN)
    ind = np.arange(n)
    for i in range(nwin):
        if i == 0:
            lo, hi = 0, central[i + 1]
            sub = (ind >= lo) & (ind < hi)
        elif i == nwin - 1:
            lo, hi = central[i - 1], np.max(ind)
            sub = (ind >= lo) & (ind <= hi)
        else:
            lo, hi = central[i - 1], central[i + 1]
            sub = (ind >= lo) & (ind < hi)
        rg, ta, vpd = rg_f[sub], tair_f[sub], vpd_f[sub]
        p = params_ok[i]
        e0 = np.full(int(sub.sum()), p[4])
        a = np.full(int(sub.sum()), p[0])
        cov = jtj_ok[i]
        wm = whichmodel[i]
        if wm == 0:
            v = _varpred("HLRC_LloydVPD", rg, ta, e0, vpd, a, cov, [p[0], p[1], p[2], p[3]], rescor[i])
        elif wm == 1:
            v = _varpred("HLRC_Lloyd", rg, ta, e0, vpd, a, cov[0:3, 0:3], [p[0], p[1], p[3]], rescor[i])
        elif wm == 2:
            v = _varpred("HLRC_Lloyd_afix", rg, ta, e0, vpd, a, cov[0:2, 0:2], [p[1], p[3]], rescor[i])
        elif wm == 3:
            v = _varpred("HLRC_LloydVPD_afix", rg, ta, e0, vpd, a, cov[0:3, 0:3], [p[1], p[2], p[3]], rescor[i])
        else:
            v = _varpred("LloydT_E0fix", rg, ta, e0, vpd, a, cov[0, 0], p[3], rescor[i])
        var_mat[i, sub] = v

    var_gpp = np.zeros(n)
    for j in range(n):
        gi = np.where(var_mat[:, j] > NAN)[0]
        count = gi.size
        if count > 1:
            c0, c1 = central[gi[0]], central[gi[1]]
            w1 = (c1 - j) / (c1 - c0)
            w2 = (j - c0) / (c1 - c0)
            var_gpp[j] = var_mat[gi[0], j] * (w1 * w1) + var_mat[gi[1], j] * (w2 * w2)
        elif count == 1:
            var_gpp[j] = var_mat[gi[0], j]
        else:
            var_gpp[j] = NAN
    return var_gpp


# --------------------------------------------------------------------------- #
# Orchestrator (one calendar year)
# --------------------------------------------------------------------------- #
def _partition_one_year(nee, ta, sw_in, ta_f, sw_in_f, vpd, julday, hr, nperday,
                        verbose=1):
    """Run the ONEFlux daytime partitioning for one year of -9999-sentinel arrays."""
    n = nee.size
    out = {c: np.full(n, np.nan) for c in
           ('RECO_DT_OF', 'GPP_DT_OF', 'SE_GPP_DT_OF', 'ALPHA_DT_OF',
            'BETA_DT_OF', 'K_DT_OF', 'RREF_DT_OF', 'E0_DT_OF')}

    # Stage A: per-record NEE uncertainty (sigd). ONEFlux stores it as float32.
    nee_fs_unc = _uncert_via_gapfill(nee, sw_in, ta, vpd, hr, nperday).astype(np.float32)

    measured = _notnan(nee)
    D = dict(
        nee_f=np.where(measured, nee, NAN),
        nee_fqc=np.where(measured, 0.0, 1.0),
        tair_f=ta_f, rg_f=sw_in_f, vpd_f=vpd, rg_meas=sw_in,
        julday=julday, nee_fs_unc=nee_fs_unc,
    )

    # Stage B: per-window parameters
    params_ok, ind_ok, whichmodel, jtj_ok, rescor = _estimate_parasets(D, nperday, verbose)
    if not params_ok:
        warn("Daytime partitioning (ONEFlux): no light-response curve could be "
             "fitted; year left unpartitioned.", verbose=verbose)
        return out

    central = np.array([int(r[2]) for r in ind_ok], dtype=int)

    # Stage C: interpolate Reco/GPP
    reco, gpp, _pf1, _pf2 = _compute_flux(n, ta_f, sw_in_f, vpd, params_ok, central)
    # Stage D: GPP standard error
    var_gpp = _compute_var(n, ta_f, sw_in_f, vpd, params_ok, central, whichmodel,
                           jtj_ok, rescor)
    with np.errstate(invalid='ignore'):
        se_gpp = np.sqrt(var_gpp)

    out['RECO_DT_OF'] = np.where(reco > NAN, reco, np.nan)
    out['GPP_DT_OF'] = np.where(gpp > NAN, gpp, np.nan)
    out['SE_GPP_DT_OF'] = np.where(se_gpp > NAN, se_gpp, np.nan)

    # report fitted parameters at their source central records (like ONEFlux)
    for r, p in zip(ind_ok, params_ok):
        i2, i0, i1 = int(r[2]), int(r[0]), int(r[1])
        if 0 <= i2 < n:
            out['RREF_DT_OF'][i2] = p[3]
            out['BETA_DT_OF'][i2] = p[1]
            out['K_DT_OF'][i2] = p[2]
        if 0 <= i0 < n:
            out['E0_DT_OF'][i0] = p[4]
        if 0 <= i1 < n:
            out['ALPHA_DT_OF'][i1] = p[0]
    return out


def _to_sentinel(a):
    """diive NaN -> ONEFlux -9999 sentinel, stored as float32.

    ONEFlux stores all working arrays as float32 (FLOAT_PREC) and promotes to
    float64 only inside the model evaluations. Replicating the float32 storage is
    necessary for parity: borderline look-up tolerance comparisons and the
    ill-conditioned daytime fits otherwise diverge from a native ONEFlux run.
    """
    a = a.astype(float).copy()
    a[~np.isfinite(a)] = NAN
    return a.astype(np.float32)


class DaytimePartitioningOneFlux:
    """Partition NEE into GPP and RECO with the daytime method (ONEFlux).

    Faithful, vectorized port of the ONEFlux ``flux_part_gl2010`` (Lasslop et al.
    2010 light-response-curve method). Each calendar year is partitioned
    independently. Emits ``*_DT_OF`` columns, mirroring the nighttime ONEFlux
    port's ``*_NT_OF`` and the daytime REddyProc port's ``*_DT_RP``.

    The method uses ONEFlux's measured-radiation day/night threshold for fitting
    (``Rg > 4`` day, ``Rg <= 4`` night) and does not use latitude / solar
    geometry. It needs both the *measured* (gappy) and *gap-filled* drivers, as
    ONEFlux does: measured Rg/TA classify records and feed the internal NEE
    uncertainty look-up, while the gap-filled drivers feed the fits and the
    flux prediction.

    Example: ``examples/flux/partitioning/partitioning_daytime_oneflux.py``

    Example:
        >>> part = DaytimePartitioningOneFlux(
        ...     nee=df['NEE_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
        ...     ta_f=df['Tair_f'], sw_in_f=df['Rg_f'], vpd=df['VPD_f'])
        >>> part.run()
        >>> results = part.results   # DataFrame with RECO_DT_OF, GPP_DT_OF, ...
    """

    def __init__(self,
                 nee: Series,
                 ta: Series,
                 sw_in: Series,
                 ta_f: Series,
                 sw_in_f: Series,
                 vpd: Series,
                 vpd_in_kpa: bool = True,
                 verbose: int = 2):
        """
        Args:
            nee: Measured net ecosystem exchange (umol m-2 s-1). Gaps (NaN) are
                the records that were not measured / did not pass QC; the daytime
                LRC is fitted on the measured daytime values only.
            ta: Measured air temperature (degC), gaps as NaN. Used for the
                day/night classification and the internal NEE-uncertainty
                look-up.
            sw_in: Measured incoming shortwave radiation (W m-2), gaps as NaN.
                Used for the ``Rg``-threshold day/night split and the
                uncertainty look-up.
            ta_f: Gap-filled air temperature (degC) - used in the fits and to
                compute RECO at every record.
            sw_in_f: Gap-filled incoming shortwave radiation (W m-2) - the LRC
                light driver, used in the fits and the flux prediction.
            vpd: Gap-filled vapour pressure deficit. By default in kPa (diive
                convention) and converted internally to hPa, the unit ONEFlux's
                Lasslop LRC expects (VPD0 = 10 hPa). Pass ``vpd_in_kpa=False`` if
                ``vpd`` is already in hPa.
            vpd_in_kpa: If True (default), ``vpd`` is in kPa and multiplied by 10
                to hPa internally.
            verbose: Console verbosity level (0 silent, 1 warnings, 2 progress
                + report, 3 debug). Default 2.
        """
        self._inputs = self._validate(nee, ta, sw_in, ta_f, sw_in_f, vpd)
        self.vpd_in_kpa = bool(vpd_in_kpa)
        self.verbose = verbose
        self._results: DataFrame | None = None

    @staticmethod
    def _validate(nee, ta, sw_in, ta_f, sw_in_f, vpd) -> DataFrame:
        series = {'nee': nee, 'ta': ta, 'sw_in': sw_in, 'ta_f': ta_f,
                  'sw_in_f': sw_in_f, 'vpd': vpd}
        for name, s in series.items():
            if not isinstance(s, Series):
                raise TypeError(f"'{name}' must be a pandas Series, got {type(s)}.")
            if not isinstance(s.index, pd.DatetimeIndex):
                raise TypeError(f"'{name}' must have a DatetimeIndex.")
        df = pd.DataFrame({k: v.astype(float) for k, v in series.items()})
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df

    def run(self) -> "DaytimePartitioningOneFlux":
        """Run the partitioning and populate :attr:`results`."""
        df = self._inputs
        index = df.index
        years = index.year.to_numpy()

        cols = ['RECO_DT_OF', 'GPP_DT_OF', 'SE_GPP_DT_OF', 'ALPHA_DT_OF',
                'BETA_DT_OF', 'K_DT_OF', 'RREF_DT_OF', 'E0_DT_OF']
        result = pd.DataFrame(index=index, columns=cols, dtype=float)

        # records per day from the median timestamp spacing
        dt_min = np.median(np.diff(index.values).astype('timedelta64[m]').astype(float))
        nperday = int(round(24 * 60 / dt_min)) if dt_min > 0 else 48

        # END-convention hour (diive may use MIDDLE-stamped index); ONEFlux hr
        # is on the 0.0 / 0.5 grid -> derive from the period end.
        end = index + pd.Timedelta(minutes=int(dt_min / 2))
        hr_all = (end.hour + np.where(end.minute == 0, 0.0, 0.5)).to_numpy().astype(np.float32)
        julday_all = end.dayofyear.to_numpy().astype(np.float32)

        vpd_factor = 10.0 if self.vpd_in_kpa else 1.0

        if self.verbose:
            info("Daytime partitioning ONEFlux (Lasslop et al. 2010) starting "
                 f"for {len(np.unique(years))} year(s).", verbose=self.verbose)

        for year in np.unique(years):
            ym = years == year
            out = _partition_one_year(
                nee=_to_sentinel(df['nee'].to_numpy()[ym]),
                ta=_to_sentinel(df['ta'].to_numpy()[ym]),
                sw_in=_to_sentinel(df['sw_in'].to_numpy()[ym]),
                ta_f=_to_sentinel(df['ta_f'].to_numpy()[ym]),
                sw_in_f=_to_sentinel(df['sw_in_f'].to_numpy()[ym]),
                vpd=_to_sentinel(df['vpd'].to_numpy()[ym] * vpd_factor),
                julday=julday_all[ym], hr=hr_all[ym], nperday=nperday,
                verbose=self.verbose)
            for col in cols:
                result.loc[ym, col] = out[col]

        self._results = result
        self.report()
        if self.verbose:
            success("Daytime partitioning (ONEFlux) finished.", verbose=self.verbose)
        return self

    @property
    def results(self) -> DataFrame:
        """DataFrame of partitioning results (aligned to the input index).

        Columns: ``RECO_DT_OF`` (ecosystem respiration), ``GPP_DT_OF`` (gross
        primary production), ``SE_GPP_DT_OF`` (GPP standard error), and the
        fitted LRC parameters ``ALPHA_DT_OF``, ``BETA_DT_OF``, ``K_DT_OF``,
        ``RREF_DT_OF``, ``E0_DT_OF`` reported at the central record of each
        window (NaN elsewhere).
        """
        if self._results is None:
            raise RuntimeError("Call .run() before accessing .results.")
        return self._results

    @property
    def reco(self) -> Series:
        """Ecosystem respiration, umol m-2 s-1."""
        return self.results['RECO_DT_OF']

    @property
    def gpp(self) -> Series:
        """Gross primary production, umol m-2 s-1."""
        return self.results['GPP_DT_OF']

    def report(self) -> None:
        """Print a Rich per-year summary of the partitioning result."""
        partitioning_report(
            title="Daytime NEE Partitioning ONEFlux (Lasslop et al. 2010)",
            reference="Pastorello et al. (2020), https://doi.org/10.1038/s41597-020-0534-3",
            results=self.results, reco_col='RECO_DT_OF', gpp_col='GPP_DT_OF',
            e0_col='E0_DT_OF', e0_unit='degC', se_col='SE_GPP_DT_OF',
            verbose=self.verbose)


def partition_nee_daytime_oneflux(nee: Series, ta: Series, sw_in: Series,
                                  ta_f: Series, sw_in_f: Series, vpd: Series,
                                  vpd_in_kpa: bool = True,
                                  verbose: int = 2) -> DataFrame:
    """Functional wrapper around :class:`DaytimePartitioningOneFlux`.

    See :class:`DaytimePartitioningOneFlux` for argument semantics.

    Returns:
        Results DataFrame (RECO_DT_OF, GPP_DT_OF, ...).
    """
    return DaytimePartitioningOneFlux(
        nee=nee, ta=ta, sw_in=sw_in, ta_f=ta_f, sw_in_f=sw_in_f, vpd=vpd,
        vpd_in_kpa=vpd_in_kpa, verbose=verbose).run().results
