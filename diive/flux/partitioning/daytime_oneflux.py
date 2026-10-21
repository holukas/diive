"""
DAYTIME PARTITIONING ONEFLUX: NEE -> GPP + RECO (Lasslop et al. 2010)
=====================================================================

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
``leastsq`` (step-bound ``factor=0.25``). On identical half-hourly or hourly
arrays every fit reproduces a native ONEFlux run under NumPy 1 bit for bit. An
hourly year is first put onto a half-hourly grid, as ONEFlux does (see
:class:`DaytimePartitioningOneFlux`).

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
from scipy.optimize import leastsq

from diive.core.utils.console import info, warn, success
from diive.flux.partitioning._report import partitioning_report
from diive.gapfilling.similarity import mds_gapfill_cascade

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


def _notnan(a):
    return a > NAN_TEST


class _BrokenWindow(Exception):
    """Raised when a window's fit returns no covariance matrix (ONEFlux's
    ``ONEFluxPartitionBrokenOptError``). ``_estimate_parasets`` sets
    ``day_range`` to the window's night range ``(day_begin2, day_end2)``,
    ``_partition_one_year`` rounds it to whole days, and
    ``_partition_year_shared`` removes the measured NEE in that range from
    every NEE series of the year and partitions them again, as ONEFlux's
    pipeline does."""
    day_range = None


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

    The drivers are stored as float32, as in ONEFlux, but ONEFlux widens them to
    float64 before evaluating a model (``.astype(DOUBLE_PREC)`` in
    ``nlinlts2``). Doing the same matters: computed in float32, the residuals
    differ by about 1e-7, which is enough to flip the sign of a VPD sensitivity
    ``k`` that converges to zero, and the model cascade branches on that sign.
    """
    def f8(key):
        return np.asarray(ind[key], dtype=np.float64)

    if lts_func == "LloydTemp":
        # E0 is a fitted parameter here, so only the temperature offset is fixed.
        tdiff = (1.0 / (TREF - T0)) - (1.0 / (f8('ta') - T0))

        def predict(par):
            return par[0] * np.exp(par[1] * tdiff)
        return predict

    ta, e0 = f8('ta'), f8('e0')
    rg = f8('rg') if 'rg' in ind else None  # absent for LloydT_E0fix (respiration only)
    tfac = np.exp(e0 * ((1.0 / (TREF - T0)) - (1.0 / (ta - T0))))  # E0 fixed
    vpdm = f8('vpd') - VPD0 if 'vpd' in ind else None
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
    ``cov_matrix=None`` when the fit is singular.

    With fewer than ``3 * npara`` usable records no fit is run, and the result
    is ONEFlux's default ``result_dict``: parameters -9999, standard errors 0,
    a zero covariance matrix, a correlation matrix of -inf, residuals -9999 and
    RMSE 0. The covariance is not None there, so ONEFlux does not treat the
    window as broken: the -9999 parameters send its model cascade on to a
    model with fewer parameters. This decides every window with exactly 11
    daytime records (enough to enter, one short of the 12 HLRC_LloydVPD needs),
    which CH-LAE's sparse NEE coverage produces a few times a year.
    """
    # validity of independent vars (all finite within the subset)
    nonnan_indep = np.ones(dep.size, dtype=bool)
    for key in indeps:
        nonnan_indep &= _notnan(np.atleast_1d(indeps[key]) if np.ndim(indeps[key]) else
                                np.full(dep.size, indeps[key]))
    fail = dict(params=np.full(npara, NAN), std=np.zeros(npara),
                cov_matrix=np.zeros((npara, npara)),
                cor_matrix=np.full((npara, npara), -np.inf),
                residuals=np.full(int(nonnan_indep.sum()), NAN), rmse=0.0)
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
    """nlinlts2 wrapper: raise _BrokenWindow when the fit returns no covariance
    matrix, where ONEFlux raises ONEFluxPartitionBrokenOptError."""
    res = _nlinlts2(lts_func, dep, indeps, npara, xguess, mprior, sigm, sigd)
    if res['cov_matrix'] is None or res['cor_matrix'] is None:
        raise _BrokenWindow(lts_func)
    return res


def _check_parameters(p, reject_alpha_at_start=False):
    """Port of library.check_parameters. p = [alpha,beta,k,rref,e0,*se...]."""
    # Widen to float64 before comparing. ONEFlux passes a row of its float32
    # `params` table and compares it against the Python floats in `fguess`,
    # which under NumPy 1.x promotes to float64; NumPy 2 would instead demote
    # the Python float to float32 (NEP 50). The difference decides the last
    # condition below, `p[0] != FGUESS0[0]`: float32(0.01) != 0.01 in float64,
    # so a window whose alpha never left the starting guess still passes.
    # That is a bug in ONEFlux (its comment and the PV-Wave original reject
    # such a window); `reject_alpha_at_start` compares in float32 instead,
    # which is what the check was written to do.
    alpha_start = np.float32(FGUESS0[0]) if reject_alpha_at_start else FGUESS0[0]
    alpha = np.float32(p[0]) if reject_alpha_at_start else np.float64(p[0])
    p = np.asarray(p, dtype=np.float64)
    is_ok = 0
    if (p[0] >= 0) and (p[0] < 0.22) and (p[1] >= 0) and (p[1] < 250) \
            and (p[2] >= 0) and (p[3] > 0) and (p[4] >= 50) and (p[4] <= 400) \
            and (alpha != alpha_start):
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
def _tstd_oneflux(vals, ddof):
    """Sample SD of float32 values with the arithmetic of ONEFlux's
    ``scipy.stats.tstd`` under NumPy 1.

    scipy computes the mean and the mean of squared deviations in the input
    dtype (float32). The ``n / (n - ddof)`` correction and the square root then
    run in float64 under NumPy 1, because there ``n`` is a float32 scalar and
    ``n - ddof`` with a Python int promotes to float64. NumPy 2 (NEP 50) keeps
    them in float32, and ``np.std`` divides the sum by ``n - ddof`` in one
    float32 step. The three results differ by one float32 ulp in about a fifth
    of the records, and these SDs weight every daytime fit: that alone flipped
    the sign of a VPD sensitivity ``k`` near zero in four CH-DAV 2016 windows.
    With this arithmetic all fits match ONEFlux on NumPy 1 bit for bit.
    """
    d = vals - np.mean(vals, keepdims=True)
    var = np.mean(d * d)
    n = vals.size
    return np.float32(np.sqrt(np.float64(var) * (n / (n - ddof))))


def _uncert_via_gapfill(nee, rg, ta, vpd, hr, nperday, longest_marginal_gap=60):
    """Reichstein-style marginal-distribution look-up gap-fill of NEE.

    Returns ``nee_fs_unc`` (the per-record standard deviation of the look-up
    neighbours), the weight ``sigd`` used by the fits, in the ONEFlux ``-9999``
    sentinel convention. All inputs use the sentinel; ``rg``/``ta`` are measured
    (gappy), ``vpd`` filled.

    Thin wrapper over the shared MDS cascade
    (:func:`diive.gapfilling.similarity.mds_gapfill_cascade`): the uncertainty
    variant uses ``min_samples=10`` (ONEFlux ``>9``) and predicts at every record
    (``fill_all``). The cascade preserves the input float32 dtype for the
    tolerance comparisons, so this keeps ONEFlux FLOAT_PREC boundary behaviour.

    ``edge='clip'`` because this port's reference is ONEFlux's own Python
    ``daytime.uncert_via_gapFill``, which clips the look-up window onto record 0
    / n-1 rather than trimming it. The MDS gap-filler keeps the cascade default
    (``'trim'``): its reference is the C ``gf_mds`` tool, which narrows the
    window bounds instead. Clipping costs nothing away from the record ends; on
    CH-DAV 2016 it changed the uncertainty of 172 of 17568 records, all of them
    within 336 records of the trailing edge.

    The SD uses ONEFlux's own ``tstd`` arithmetic (:func:`_tstd_oneflux`), not
    ``np.std``.
    """
    def to_nan(x):
        x = np.asarray(x)
        out = np.where(x > NAN_TEST, x, np.nan)
        return out.astype(x.dtype) if np.issubdtype(x.dtype, np.floating) else out

    res = mds_gapfill_cascade(
        to_nan(nee), to_nan(rg), to_nan(ta), to_nan(vpd), np.asarray(hr), nperday,
        min_samples=10, swin_tol=(20.0, _RG_TOL), ta_tol=_TA_TOL, vpd_tol=_VPD_TOL,
        ddof=1, fill_all=True, longest_marginal_gap=longest_marginal_gap,
        edge='clip', sd_func=_tstd_oneflux,
    )
    sd = res['sd']
    return np.where(np.isfinite(sd), sd, NAN)


# --------------------------------------------------------------------------- #
# Stage B: per-window parameter estimation (port of daytime.estimate_parasets)
# --------------------------------------------------------------------------- #
def _estimate_parasets(D, nperday, verbose=1, reject_alpha_at_start=False):
    """Per-window LRC parameter estimation.

    ``D`` holds the year's arrays: nee_f, nee_fqc, tair_f, rg_f, vpd_f, rg_meas
    (measured, -9999 sentinel), julday, nee_fs_unc, and ``nee_f_first``, the
    gap-filled NEE of the year's first record (float32, -9999 sentinel) or
    None when no gap-filled NEE was given. Returns lists of accepted windows:
    params (10,), whichmodel, cov (4x4), res_cor, and the 3 ind rows, plus the
    ``(day_begin, day_end)`` of windows that ONEFlux would fit from record 0
    but that could not be fitted without ``nee_f_first``.
    Raises ``_BrokenWindow`` (with ``day_range`` set) at the first fit that
    returns no covariance matrix.
    """
    nee_f = D['nee_f']
    nee_fqc = D['nee_fqc']
    tair_f = D['tair_f']
    rg_f = D['rg_f']
    vpd_f = D['vpd_f']
    rg_meas = D['rg_meas']
    julday = D['julday']
    sigd_all = D['nee_fs_unc']
    nee_f_first = D.get('nee_f_first')

    n_parasets = (365 // WINSIZE) * 2
    fguess = list(FGUESS0)

    params_ok, ind_ok, whichmodel_ok, jtj_ok, rescor_ok = [], [], [], [], []
    no_night_windows = []
    i_ok = 0
    lloydtemp_e0 = None

    for i in range(n_parasets):
        day_begin = i * WINSIZE / 2.0
        day_end = day_begin + WINSIZE
        day_begin2 = (i - 2) * WINSIZE / 2.0 if i > 1 else 0
        day_end2 = (i + 2) * WINSIZE / 2.0 + WINSIZE if i < n_parasets - 2 else float(np.max(julday))

        # ONEFlux hardcodes 48 records per day here (daytime.py:1045). It can,
        # because an hourly year reaches it duplicated onto a half-hourly grid
        # (library.create_data_structures), and _run_shared does the same, so
        # nperday is 48 for both resolutions ONEFlux supports. Other
        # resolutions anchor on their own record count per day.
        central = int((day_begin + WINSIZE / 2.0) * float(nperday))
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

        # A window without any measured night record: ONEFlux selects with
        # library.newselif, which sets condition[0] = True when the selection
        # is empty. The window then has one night record, the year's first,
        # so it takes E0 from earlier (E0set below) and the gap-filled NEE of
        # that record, measured or not, becomes the Rref starting guess rb.
        # Only that rb is used: with E0set no LloydTemp fit runs. diive keeps
        # only measured NEE, so it needs the gap-filled value passed in
        # (nee_f_first). Without it the selection stays empty, rb is NaN and
        # the window is not fitted (L215).
        night_from_first = (nee_f_first is not None) and not subn_m.any()
        n_night = 1 if night_from_first else int(subn_m.sum())

        E0set = 0
        if n_night <= 10 and i_ok > 0 and lloydtemp_e0 is not None:
            # ONEFlux sets lloydtemp_e0 (and lloydtemp_e0_se) here but never
            # reads them again: the E0 column of the fits and the parameter
            # table get the local e0 and e0_se, which still hold the values of
            # the last window that ran LloydTemp, after its 50-400 check. They
            # differ from params_ok[i_ok - 1] when that window was rejected
            # (GF-Guy 2024: E0 282.6 instead of 57.0; L216). So e0 and e0_se
            # are left as they are.
            lloydtemp_e0 = params_ok[i_ok - 1][4]
            ind_rows[0] = ind_ok[i_ok - 1][0]
            E0set = 1

        if not ((n_night > 10 or E0set == 1) and subd_m.sum() > 10):
            continue
        if E0set == 1 and not subn_m.any() and not night_from_first:
            no_night_windows.append((day_begin, day_end))

        try:
            percs = _percentiles_fn(nee_f[sub_m], [0.03, 0.97])
            beta = abs(percs[0] - percs[1])
            if night_from_first:
                rb = float(nee_f_first)
            elif not subn_m.any():
                rb = np.nan  # np.average of nothing, without its RuntimeWarning
            else:
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
            # E0set == 1: e0 and e0_se keep the last LloydTemp window's values,
            # as in ONEFlux (see above).

            # ONEFlux stores the fixed E0 driver column as float32.
            e0_arr_d = np.full(int(subd_m.sum()), e0, dtype=np.float32)

            # ONEFlux keeps the per-window parameter table in FLOAT_PREC
            # (`estimate_parasets`: `params = numpy.zeros(..., dtype=FLOAT_PREC)`)
            # and every branch test, the accept test and the carry-over of alpha
            # and E0 to the next window read it back from there. Storing it in
            # float64 instead is not a rounding detail: `check_parameters` ends
            # in `params[0] != fguess[0]`, meant to reject a window whose alpha
            # never moved off the 0.01 starting guess, and float32(0.01) is not
            # equal to the Python float 0.01 - so in ONEFlux that guard never
            # fires and those windows are accepted.
            pj = np.zeros((3, 10), dtype=np.float32)
            indj = np.tile(ind_rows, (3, 1))
            # ONEFlux's RMSE table is FLOAT_PREC too, and the best of the three
            # beta starting guesses is picked from it. Two fits whose RMSEs
            # differ only beyond float32 precision tie there, and the first one
            # wins; in float64 the later one can win (CH-LAE 2019 with gaps in
            # measured TA/SW_IN: window of 7 June, alpha then carried to the next).
            rmse = np.zeros(3, dtype=np.float32)
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

                if _check_parameters(pj[j], reject_alpha_at_start) == 0:
                    rmse[j] = 9999.0
            # end for j

            jmin = int(np.where(rmse == np.min(np.abs(rmse)))[0][0])
            if _check_parameters(pj[jmin], reject_alpha_at_start) == 1:
                params_ok.append(pj[jmin].copy())
                ind_ok.append(indj[jmin].copy())
                whichmodel_ok.append(int(wm[jmin]))
                jtj_ok.append(jtj[jmin].copy())
                rescor_ok.append(float(rescor[jmin]))
                i_ok += 1
        except _BrokenWindow as e:
            # ONEFlux aborts the year here and reports the night range, not
            # the 4-day fitting window (daytime.py, ONEFluxPartitionBrokenOptError).
            e.day_range = (day_begin2, day_end2)
            raise
    # end for i

    return params_ok, ind_ok, whichmodel_ok, jtj_ok, rescor_ok, no_night_windows


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
def _remove_days(nee, julday, ranges):
    """Set NEE to missing for julday in (begin, end] of every range.

    Port of ONEFlux ``library.remove_errored_entries``, which selects with
    ``(julday > begin) == (julday <= end)``.
    """
    for begin, end in ranges:
        nee = np.where((julday > begin) == (julday <= end), np.float32(NAN), nee)
    return nee


def _partition_year_shared(nees, julday, verbose=1, nee_f_first=None, **kwargs):
    """Partition one year of several NEE arrays that share one list of removed days.

    A window whose fit returns no covariance aborts the year in ONEFlux
    (ONEFluxPartitionBrokenOptError). Its pipeline (PipelineNEEPartitionDT.run)
    appends the window's night range to the site's error file and runs
    ``partitioning_dt`` again; ``library.remove_errored_entries`` then marks
    the measured NEE in every range listed for that site, year and product as
    missing, which also changes the NEE uncertainty. The file is not keyed by
    u* percentile, so a range found in one percentile is removed in all of
    them. During this search the rerun keeps the outputs already written, so
    the search resumes at the percentile that broke. Once nothing breaks, the
    pipeline runs every percentile again with the complete file, and goes
    back to searching if that pass breaks. Years are partitioned
    independently, so per year this is: sweep the series in order; at a broken
    fit, add its range and retry the same series; after a sweep with a break,
    sweep again, until a sweep runs through without one. The order decides
    which ranges are found. It terminates: a range covers its own window's
    daytime records, so that window never fits again.

    A series whose last run already had every range removed is not run again
    in the next sweep: it would give the same result. So each series runs once
    when no fit breaks, and a single series runs exactly as often as it breaks
    plus one.

    Args:
        nees: ``{label: nee}``, -9999-sentinel float32 NEE arrays of one year,
            in partitioning order. ``label`` names the series in the warning;
            ``None`` for a single series.
        julday: Day of year per record.
        verbose: Console verbosity.
        nee_f_first: ``{label: value}``, the gap-filled NEE of each series'
            first record of the year (float32, -9999 sentinel), or None when
            no gap-filled NEE was given. See ``_estimate_parasets``.
        **kwargs: The other arguments of ``_partition_one_year`` (drivers,
            ``hr``, ``nperday``, ``reject_alpha_at_start``).

    Returns:
        ``(outs, ranges)``: ``{label: out}`` from ``_partition_one_year`` with
        all ``ranges`` removed, and the removed ``(begin, end)`` day ranges in
        the order they were found.
    """
    ranges = []
    outs = {}      # label -> result of its last clean run
    n_used = {}    # label -> number of ranges removed in that run
    while True:
        broke = False
        for label, nee in nees.items():
            while n_used.get(label) != len(ranges):
                try:
                    outs[label] = _partition_one_year(
                        nee=_remove_days(nee, julday, ranges), julday=julday,
                        verbose=verbose, nee_f_first=(nee_f_first or {}).get(label),
                        **kwargs)
                    n_used[label] = len(ranges)
                except _BrokenWindow as e:
                    begin, end = e.day_range
                    ranges.append((begin, end))
                    broke = True
                    if label is None:
                        warn(f"Daytime partitioning (ONEFlux): {e.args[0]} fit failed in "
                             f"the window of days {begin}-{end}; measured NEE there "
                             f"removed and the year partitioned again, as ONEFlux "
                             f"does.", verbose=verbose)
                    else:
                        warn(f"Daytime partitioning (ONEFlux): {e.args[0]} fit failed in "
                             f"the window of days {begin}-{end} of NEE series "
                             f"{label!r}; measured NEE there removed in all series and "
                             f"the year partitioned again, as ONEFlux does.",
                             verbose=verbose)
        if not broke:
            return outs, ranges


def _partition_one_year(nee, ta, sw_in, ta_f, sw_in_f, vpd, julday, hr, nperday,
                        verbose=1, reject_alpha_at_start=False, nee_f_first=None,
                        year=None):
    """Run the ONEFlux daytime partitioning for one year of -9999-sentinel arrays.

    ``nee_f_first`` is the gap-filled NEE of the year's first record (or None),
    ``year`` only names the year in warnings.

    Raises ``_BrokenWindow`` at the first fit without covariance, with
    ``day_range`` as whole days; ``_partition_year_shared`` handles the rerun.
    """
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
        julday=julday, nee_fs_unc=nee_fs_unc, nee_f_first=nee_f_first,
    )

    # Stage B: per-window parameters
    try:
        params_ok, ind_ok, whichmodel, jtj_ok, rescor, no_night = _estimate_parasets(
            D, nperday, verbose, reject_alpha_at_start)
    except _BrokenWindow as e:
        # The error file stores the range as whole days ('{:.0f}', read
        # back as int).
        e.day_range = tuple(int(format(d, '.0f')) for d in e.day_range)
        raise
    if no_night:
        days = ', '.join(f'{b:.0f}-{e:.0f}' for b, e in no_night)
        of_year = f' of {year}' if year is not None else ''
        warn(f"Daytime partitioning (ONEFlux): {len(no_night)} window(s){of_year} "
             f"without any measured nighttime NEE (days {days}) not fitted. ONEFlux "
             f"fits them, with the gap-filled NEE of the year's first record as the "
             f"Rref starting guess; pass nee_f (gap-filled NEE) to reproduce ONEFlux.",
             verbose=verbose)
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

    # A missing gap-filled driver arrives here as the -9999 sentinel, which the
    # models turn into a finite but meaningless flux (RECO at -9999 degC).
    # ONEFlux reads a missing driver as NaN, which propagates through the same
    # formulas, so blank the records whose drivers the flux depends on.
    ta_ok, rg_ok, vpd_ok = _notnan(ta_f), _notnan(sw_in_f), _notnan(vpd)
    out['RECO_DT_OF'] = np.where((reco > NAN) & ta_ok, reco, np.nan)
    out['GPP_DT_OF'] = np.where((gpp > NAN) & rg_ok & vpd_ok, gpp, np.nan)
    out['SE_GPP_DT_OF'] = np.where((se_gpp > NAN) & ta_ok & rg_ok & vpd_ok, se_gpp, np.nan)

    # report fitted parameters at their source central records (like ONEFlux)
    for r, p in zip(ind_ok, params_ok, strict=False):
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

    Input resolution: hourly data are partitioned the way ONEFlux partitions an
    hourly site. ONEFlux's daytime method works on a half-hourly grid. It
    repeats every hourly record, and the first copy stands for the half-hour
    that ends 30 minutes earlier. The windows are fitted on this grid, which
    holds every measured record twice, and ONEFlux's pipeline keeps the first
    copy of each record for its hourly output. diive does the same, so an
    hourly year gives ONEFlux's hourly result. The window parameters are
    reported at their central records on the hourly index. Half-hourly data
    are partitioned as they are. Other resolutions, which ONEFlux does not
    process, are partitioned at their own resolution.

    Measured agreement against a native ONEFlux 1.3.7 run (NumPy 1.26, SciPy
    1.17) on the same arrays: half-hourly CH-DAV 2016 and 2019 and CH-LAE
    2017-2019 (where only 17-20 % of NEE is measured), also with gaps in
    measured TA, SW_IN and nighttime NEE, and hourly US-Ha1 2023-2024, US-MMS
    2022-2023 and CH-DAV 2016 averaged to hourly records. The per-record NEE
    uncertainty is identical bit for bit, and so is every light-response fit
    and every window parameter (143 windows on CH-DAV 2016, 77-123 per CH-LAE
    run, 142-162 per hourly run). Per record, GPP and RECO differ by at most
    8e-6 umol m-2 s-1, float32 rounding, and the annual sums agree to
    0.0001%. On the FLUXNET v1.3 sites FI-Hyy, FI-Var, IL-Yat, AU-GWW and
    GF-Guy (2023, 2024) and DE-Hai (2020, 2024) every fit is bitwise
    identical as well.

    A fit that returns no covariance matrix (``leastsq`` did not converge)
    makes ONEFlux abort the year. Its pipeline then removes the measured NEE
    on the days that window's nighttime fit uses (about 12 days) and runs the
    year again, until no fit fails. diive does the same, so the NEE
    uncertainty and the windows around the failed one change too; the removed
    day ranges are in :attr:`removed_days`. In ONEFlux the removed days apply
    to all u* percentiles of that year and product. This class partitions one
    NEE series and removes only the days its own fits need. To share the
    removed days between u* scenarios, use
    :func:`partition_nee_daytime_oneflux_scenarios`, which Level 4.2 of the
    post-processing chain (``run_level42_daytime_oneflux``) does. That equals
    ONEFlux only when the scenarios are ONEFlux's percentile set in its order
    (``'1.25'``, ``'3.75'``, ... in ``PERC_TO_COMPARE``).

    ONEFlux's own output depends on the NumPy version it runs under. NumPy 2
    changed scalar type promotion (NEP 50), and that moves ONEFlux in two
    places: the SD in its uncertainty look-up rounds differently in a fifth of
    the records, and its alpha check starts to reject windows (see
    ``reject_alpha_at_start``). On CH-DAV 2016, ONEFlux under NumPy 2 fits 7
    fewer windows and its annual GPP rises by 5.8%. With the alpha check held
    at its NumPy-1 result, 17 windows still change, most of them windows where
    ``k`` converges to zero and the model cascade branches on its sign. diive
    reproduces ONEFlux under NumPy 1.

    Example: ``examples/flux/partitioning/partitioning_daytime_oneflux.py``

    Example:
        >>> import diive as dv
        >>> df = dv.load_exampledata_parquet()
        >>> part = dv.flux.DaytimePartitioningOneFlux(
        ...     nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
        ...     ta_f=df['Tair_f'], sw_in_f=df['Rg_f'], vpd=df['VPD_f'],
        ...     vpd_in_kpa=False)  # VPD_f is in hPa in the example data
        >>> part.run()  # then part.results -> DataFrame with RECO_DT_OF, GPP_DT_OF, ...
    """

    def __init__(self,
                 nee: Series,
                 ta: Series,
                 sw_in: Series,
                 ta_f: Series,
                 sw_in_f: Series,
                 vpd: Series,
                 vpd_in_kpa: bool = True,
                 reject_alpha_at_start: bool = False,
                 nee_f: Series | None = None,
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
            reject_alpha_at_start: ONEFlux's window check is meant to reject a
                window whose alpha never moved off its starting value of 0.01,
                but in ONEFlux 1.3.7 the check compares a float32 against a
                float64 and never fires. False (default) reproduces ONEFlux, so
                the results match its output. True applies the check as ONEFlux's
                own comment intends. On CH-DAV 2016 that rejects 7 more windows
                and raises annual GPP by 5.7%, in ONEFlux and here alike.
                ONEFlux run under NumPy 2 behaves like True, because there the
                comparison happens in float32.
            nee_f: Gap-filled NEE (umol m-2 s-1), optional. ONEFlux uses it in
                one place only: a window without any measured nighttime record
                takes the gap-filled NEE of the year's first record as its Rref
                starting guess (ONEFlux's selection falls back to that record
                when it would be empty) and is fitted. With ``nee_f`` diive
                does the same. Without it such windows are not fitted and a
                warning lists them. This happens at high latitudes in summer
                (FI-Hyy, FI-Var) and in long gaps of nighttime NEE.
            verbose: Console verbosity level (0 silent, 1 warnings, 2 progress
                + report, 3 debug). Default 2.
        """
        self._inputs = self._validate(nee, ta, sw_in, ta_f, sw_in_f, vpd, nee_f)
        self.vpd_in_kpa = bool(vpd_in_kpa)
        self.reject_alpha_at_start = bool(reject_alpha_at_start)
        self.verbose = verbose
        self._results: DataFrame | None = None
        self._removed_days: dict[int, list[tuple[int, int]]] | None = None

    @staticmethod
    def _validate(nee, ta, sw_in, ta_f, sw_in_f, vpd, nee_f=None) -> DataFrame:
        series = {'nee': nee, 'ta': ta, 'sw_in': sw_in, 'ta_f': ta_f,
                  'sw_in_f': sw_in_f, 'vpd': vpd}
        if nee_f is not None:
            series['nee_f'] = nee_f
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
        """Run the partitioning and populate :attr:`results` and :attr:`removed_days`."""
        _run_shared({None: self})
        return self

    @property
    def removed_days(self) -> dict[int, list[tuple[int, int]]]:
        """Day ranges whose measured NEE was removed after a failed fit.

        ``{year: [(begin, end), ...]}`` for every partitioned year, in the
        order the ranges were found (empty list when no fit failed). Each
        range covers the days of year ``begin < day <= end``, the night range
        of the window whose fit failed, as in ONEFlux's error file. After
        :func:`partition_nee_daytime_oneflux_scenarios`, all scenarios carry
        the same ranges.
        """
        if self._removed_days is None:
            raise RuntimeError("Call .run() before accessing .removed_days.")
        return self._removed_days

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


def _run_shared(parts: dict) -> None:
    """Run ``{label: DaytimePartitioningOneFlux}`` year by year with shared removed days.

    All instances must have the same index, drivers and settings; drivers and
    settings are read from the first one. Sets ``_results`` and
    ``_removed_days`` on every instance and prints each report.
    """
    first = next(iter(parts.values()))
    df = first._inputs
    index = df.index
    years = index.year.to_numpy()

    cols = ['RECO_DT_OF', 'GPP_DT_OF', 'SE_GPP_DT_OF', 'ALPHA_DT_OF',
            'BETA_DT_OF', 'K_DT_OF', 'RREF_DT_OF', 'E0_DT_OF']
    results = {label: pd.DataFrame(index=index, columns=cols, dtype=float)
               for label in parts}
    removed = {}

    # records per day from the median timestamp spacing
    dt_min = np.median(np.diff(index.values).astype('timedelta64[m]').astype(float))
    nperday = int(round(24 * 60 / dt_min)) if dt_min > 0 else 48

    # END-convention hour (diive may use MIDDLE-stamped index); ONEFlux hr
    # is on the 0.0 / 0.5 grid -> derive from the period end.
    end = index + pd.Timedelta(minutes=int(dt_min / 2))

    # Hourly data: ONEFlux partitions the daytime on a half-hourly grid.
    # library.create_data_structures (library.py:1264-1287) repeats every
    # record of a year with fewer than 17000 records, which in its pipeline
    # (full years only) means an hourly year. The first copy becomes the
    # half-hour that ends 30 minutes earlier: hour - 1, minute 30, hr - 0.5,
    # and for the record ending at 00:00 the date, julday and hr 23.5 of the
    # day before. The second copy keeps the record's own END. flux_part_gl2010
    # then runs on this grid with 48 records per day, and the pipeline keeps
    # every second record of its result, the first copies
    # (pipeline/wrappers.py:1953-1960, PipelinePrepareURE.check_cleanup_dt:
    # reco[::2], gpp[::2]). Building the grid from the END timestamps gives
    # the same julday and hr. Only the daytime duplicates (DT_STR at
    # library.py:1267): the nighttime partitioning runs on the hourly records.
    duplicate = nperday == 24
    if duplicate:
        end_first = (end - pd.Timedelta(minutes=30)).values
        end = pd.DatetimeIndex(np.column_stack([end_first, end.values]).ravel())
        nperday = 48
    years_grid = np.repeat(years, 2) if duplicate else years

    def on_grid(a):
        return np.repeat(a, 2) if duplicate else a

    hr_all = (end.hour + np.where(end.minute == 0, 0.0, 0.5)).to_numpy().astype(np.float32)
    julday_all = end.dayofyear.to_numpy().astype(np.float32)

    vpd_factor = 10.0 if first.vpd_in_kpa else 1.0
    verbose = first.verbose

    if verbose:
        series = f" and {len(parts)} NEE series" if len(parts) > 1 else ""
        info("Daytime partitioning ONEFlux (Lasslop et al. 2010) starting "
             f"for {len(np.unique(years))} year(s){series}.", verbose=verbose)

    for year in np.unique(years):
        ym = years == year
        yg = years_grid == year
        julday = julday_all[yg]
        # The END timestamp of a year's last record already belongs to the
        # next year, so its day-of-year wraps to 1 and the record would be
        # pooled into the January windows. ONEFlux repairs the same wrap in
        # library.create_data_structures (365 -> 366, 366 -> 367).
        if julday.size > 1 and julday[-1] == 1 and julday[-2] in (365.0, 366.0):
            julday = julday.copy()
            julday[-1] = julday[-2] + 1
        # The gap-filled NEE of each series' first record of the year, the
        # one value of nee_f that ONEFlux's daytime partitioning reads.
        nee_f_first = {label: (_to_sentinel(p._inputs['nee_f'].to_numpy()[ym][:1])[0]
                               if 'nee_f' in p._inputs else None)
                       for label, p in parts.items()}
        outs, ranges = _partition_year_shared(
            nees={label: _to_sentinel(on_grid(p._inputs['nee'].to_numpy()[ym]))
                  for label, p in parts.items()},
            nee_f_first=nee_f_first, year=int(year),
            ta=_to_sentinel(on_grid(df['ta'].to_numpy()[ym])),
            sw_in=_to_sentinel(on_grid(df['sw_in'].to_numpy()[ym])),
            ta_f=_to_sentinel(on_grid(df['ta_f'].to_numpy()[ym])),
            sw_in_f=_to_sentinel(on_grid(df['sw_in_f'].to_numpy()[ym])),
            vpd=_to_sentinel(on_grid(df['vpd'].to_numpy()[ym] * vpd_factor)),
            julday=julday, hr=hr_all[yg], nperday=nperday,
            verbose=verbose, reject_alpha_at_start=first.reject_alpha_at_start)
        removed[int(year)] = ranges
        for label, out in outs.items():
            for col in cols:
                # Hourly: the first copy of each record, as ONEFlux's pipeline
                # keeps it. Window parameters sit at even grid records
                # ((day_begin + 2) * 48 with day_begin a multiple of 2), so
                # none is lost.
                results[label].loc[ym, col] = out[col][::2] if duplicate else out[col]

    for label, p in parts.items():
        p._results = results[label]
        p._removed_days = {year: list(r) for year, r in removed.items()}
        p.report()
    if verbose:
        success("Daytime partitioning (ONEFlux) finished.", verbose=verbose)


def partition_nee_daytime_oneflux(nee: Series, ta: Series, sw_in: Series,
                                  ta_f: Series, sw_in_f: Series, vpd: Series,
                                  vpd_in_kpa: bool = True,
                                  reject_alpha_at_start: bool = False,
                                  nee_f: Series | None = None,
                                  verbose: int = 2) -> DataFrame:
    """Functional wrapper around :class:`DaytimePartitioningOneFlux`.

    See :class:`DaytimePartitioningOneFlux` for argument semantics.

    Returns:
        Results DataFrame (RECO_DT_OF, GPP_DT_OF, ...).
    """
    return DaytimePartitioningOneFlux(
        nee=nee, ta=ta, sw_in=sw_in, ta_f=ta_f, sw_in_f=sw_in_f, vpd=vpd,
        vpd_in_kpa=vpd_in_kpa, reject_alpha_at_start=reject_alpha_at_start,
        nee_f=nee_f, verbose=verbose).run().results


def partition_nee_daytime_oneflux_scenarios(
        nee: dict[str, Series], ta: Series, sw_in: Series, ta_f: Series,
        sw_in_f: Series, vpd: Series, vpd_in_kpa: bool = True,
        reject_alpha_at_start: bool = False,
        nee_f: dict[str, Series] | None = None,
        verbose: int = 2) -> dict[str, DaytimePartitioningOneFlux]:
    """Partition several NEE series (u* scenarios) that share drivers and removed days.

    Runs :class:`DaytimePartitioningOneFlux` on each series, but with the
    removed days shared, as ONEFlux's pipeline shares its error file between
    the u* percentiles of a year and product. Per year, the series are
    partitioned in the order of ``nee``. When a fit fails, the night range of
    its window is removed from the measured NEE of every series and the same
    series is partitioned again. After a pass in which a fit failed, the
    series whose result predates the last removed range are partitioned again,
    in order, until a pass runs through without a failed fit. When no fit
    fails, each series is partitioned once, as with the class.

    With one series the result equals the class. The result equals ONEFlux
    only when the series are ONEFlux's u* percentiles in its order (``'1.25'``,
    ``'3.75'``, ... in ``PERC_TO_COMPARE``) and of one product (CUT or VUT):
    the order decides which fits fail.

    Args:
        nee: ``{label: measured NEE}``, one Series per scenario, all with the
            same index. The order is the partitioning order.
        ta, sw_in, ta_f, sw_in_f, vpd, vpd_in_kpa, reject_alpha_at_start, verbose: As in
            :class:`DaytimePartitioningOneFlux`, shared by all series.
        nee_f: ``{label: gap-filled NEE}``, optional, the gap-filled NEE of
            each scenario (see :class:`DaytimePartitioningOneFlux`). A label
            missing from it partitions without gap-filled NEE.

    Returns:
        ``{label: DaytimePartitioningOneFlux}``, already run: ``results`` and
        ``removed_days`` are populated, and ``removed_days`` is the same for
        every series.
    """
    if not nee:
        raise ValueError("nee is empty; pass at least one NEE series.")
    nee_f = nee_f or {}
    unknown = [label for label in nee_f if label not in nee]
    if unknown:
        raise ValueError(f"nee_f has label(s) {unknown} that are not in nee.")
    parts = {label: DaytimePartitioningOneFlux(
        nee=s, ta=ta, sw_in=sw_in, ta_f=ta_f, sw_in_f=sw_in_f, vpd=vpd,
        vpd_in_kpa=vpd_in_kpa, reject_alpha_at_start=reject_alpha_at_start,
        nee_f=nee_f.get(label), verbose=verbose) for label, s in nee.items()}
    # The year loop reads the drivers from the first instance only. Each
    # instance aligns the same driver Series to its NEE index, so the driver
    # columns are identical only when the indexes are.
    first_index = next(iter(parts.values()))._inputs.index
    for label, p in parts.items():
        if not p._inputs.index.equals(first_index):
            raise ValueError(f"NEE series {label!r} has a different index than the "
                             f"first series; all series must share one index.")
    _run_shared(parts)
    return parts
