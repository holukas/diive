"""
USTAR CP CORE: ONEFLUX CHANGE-POINT ROUTINES
============================================

Array-level port of the ONEFlux ``ustar_cp`` MATLAB routines (Barr et al. 2013),
used by ``ustar_cp_detection.UstarChangePointDetection``.

File references are to ``oneflux_steps/ustar_cp/`` of ONEFlux 1.3.7. Only the
2-parameter operational model is ported: ``launch.m:316-323`` passes ``Stats2``
to ``cpdAssignUStarTh20100901`` and discards ``Cp3``/``Stats3``, so the
3-parameter diagnostic model never reaches an output.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
from scipy import special

from diive.flux.lowres._ustar_cp_numerics import (
    fmax2p_cp2,
    matlab_colon,
    sequential_sum,
    matlab_prctile,
    matlab_round,
)

N_SEASONS = 4  # cpdEvaluateUStarTh4Season20100901.m:217
N_STRATA_MIN = 4
N_STRATA_MAX = 8
N_BINS = 50
P_SIG = 0.05  # cpdFindChangePoint20100901.m:107, cpdAssignUStarTh20100901.m:95

#: Fields of the per-stratum Stats2 record that this port fills, in ONEFlux order.
STATS_FIELDS = ('n', 'Cp', 'Fmax', 'p', 'b0', 'b1', 'cib0', 'cib1',
                'mt', 'ti', 'tf', 'ruStarVsT', 'puStarVsT', 'mT', 'ciT')


def n_per_bin(n_per_day: int) -> int:
    """Records per u* bin (cpdEvaluateUStarTh4Season20100901.m:217-221)."""
    return {24: 3, 48: 5}.get(int(n_per_day), 5)


def n_per_day_from_t(t: np.ndarray) -> int:
    """``round(1/nanmedian(diff(t)))`` (Bootstrap.m:72, Evaluate4S.m:215)."""
    dt = np.diff(t)
    dt = dt[~np.isnan(dt)]
    return int(matlab_round(1.0 / np.median(dt)))


def _ols(x, y):
    """Least squares line y = b0 + b1*x and its residual sum of squares.

    MATLAB solves ``[ones(n,1) x] \\ y`` by QR. The centred closed form gives the
    same fit up to rounding. When all x are equal (rank-deficient design) the
    fit is the mean of y, as MATLAB's basic solution gives.
    """
    xm = x.mean()
    ym = y.mean()
    dx = x - xm
    sxx = np.dot(dx, dx)
    if sxx == 0.0:
        b1 = 0.0
    else:
        b1 = np.dot(dx, y - ym) / sxx
    b0 = ym - b1 * xm
    return b0, b1, sxx, xm


def find_change_point(xx: np.ndarray, yy: np.ndarray) -> tuple:
    """2-parameter change point of binned (u*, NEE) means (cpdFindChangePoint20100901.m).

    Returns ``(cp2, stats)``. ``cp2`` is the change point, NaN when p > 0.05
    (line 114). ``stats`` holds n, Cp, Fmax, p, b0, b1, cib0, cib1. Apart from
    n, these are filled only when the change point lies more than
    ``max(3, floor(0.05*n))`` points from either end (lines 128-137). So
    ``stats['Cp']`` is NaN near the ends even when ``cp2`` is significant.
    """
    stats = dict(n=np.nan, Cp=np.nan, Fmax=np.nan, p=np.nan,
                 b0=np.nan, b1=np.nan, cib0=np.nan, cib1=np.nan)
    cp2 = np.nan

    ok = ~np.isnan(xx + yy)
    x = np.asarray(xx, dtype=np.float64)[ok]
    y = np.asarray(yy, dtype=np.float64)[ok]
    n = x.size
    if n < 10:
        return cp2, stats

    # Exclude points more than 4 SD (N-1) from the mean residual of a linear fit (lines 57-59)
    b0, b1, _, _ = _ols(x, y)
    dy = y - (b0 + b1 * x)
    mdy = dy.mean()
    sdy = dy.std(ddof=1)
    keep = ~(np.abs(dy - mdy) > 4.0 * sdy)
    x = x[keep]
    y = y[keep]
    n = x.size
    if n < 10:
        return cp2, stats

    ym = y.mean()
    sse_red = np.sum((y - ym) ** 2)

    # Fc2 for i = 1..n-1 (lines 78-89): x1 = x with x(i+1:n) set to x(i), one row per i.
    # Rows with equal x(i) are identical, so their F values tie exactly and argmax
    # picks the first, as MATLAB max does.
    m = n - 1
    cols = np.arange(n)
    rows = np.arange(m)
    x1 = np.where(cols[None, :] > rows[:, None], x[:m, None], x[None, :])
    xm = x1.mean(axis=1)
    dx = x1 - xm[:, None]
    sxx = np.einsum('ij,ij->i', dx, dx)
    with np.errstate(divide='ignore', invalid='ignore'):
        b1v = np.where(sxx == 0.0, 0.0, (dx @ (y - ym)) / sxx)
        b0v = ym - b1v * xm
        res = y[None, :] - (b0v[:, None] + b1v[:, None] * x1)
        sse_full = np.einsum('ij,ij->i', res, res)
        fc2 = (sse_red - sse_full) / (sse_full / (n - 2))
    # A rank-deficient row reproduces the null model: F = 0 (MATLAB: 0 up to rounding)
    fc2[sxx == 0.0] = 0.0
    if np.all(np.isnan(fc2)):
        return cp2, stats
    i_cp = int(np.nanargmax(fc2))
    fmax = float(fc2[i_cp])
    x_cp = float(x[i_cp])

    # regress(y, [ones x1]) at the change point, 95 % intervals (lines 110-111)
    x1 = np.where(cols > i_cp, x_cp, x)
    b0, b1, sxx_cp, xm_cp = _ols(x1, y)
    res = y - (b0 + b1 * x1)
    mse = np.dot(res, res) / (n - 2)
    tval = special.stdtrit(n - 2, 0.975)
    with np.errstate(divide='ignore', invalid='ignore'):
        se_b1 = np.sqrt(mse / sxx_cp)
        se_b0 = np.sqrt(mse * (1.0 / n + xm_cp ** 2 / sxx_cp))
    p2 = fmax2p_cp2(fmax, n)
    cp2 = x_cp
    if p2 > P_SIG:
        cp2 = np.nan

    stats['n'] = n
    n_end = max(3, int(np.floor(0.05 * n)))
    i1 = i_cp + 1  # MATLAB index
    if n_end < i1 < n - n_end:
        stats.update(Cp=cp2, Fmax=fmax, p=p2, b0=b0, b1=b1,
                     cib0=tval * se_b0, cib1=tval * se_b1)
    return cp2, stats


def _corrcoef_p(x, y):
    """``[r, p] = corrcoef(x, y)``; NaN if any value is NaN (no 'rows' option)."""
    n = x.size
    if n < 2 or np.isnan(x).any() or np.isnan(y).any():
        return np.nan, np.nan
    dx = x - x.mean()
    dy = y - y.mean()
    den = np.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
    if den == 0.0:
        return np.nan, np.nan
    r = float(np.dot(dx, dy) / den)
    if n < 3:
        return r, np.nan
    with np.errstate(divide='ignore'):
        tstat = r * np.sqrt((n - 2) / (1.0 - r * r))
    p = 2.0 * special.stdtr(n - 2, -abs(tstat))
    return r, float(p)


def bin_stratum(u, nee, ta, npb):
    """Binned means of u*, NEE and TA for one stratum (fcBin.m, ``dx = []``).

    ``nBins = floor(n/nPerBin)`` bins with edges at the u* percentiles
    ``0:(100/nBins):100``. Bins are inclusive at both edges, so a value on an
    edge counts in both bins. Bins with fewer than nPerBin records are dropped
    and the kept means move to the front; the tail stays NaN. The TA means use
    the same bins (the second fcBin call, Evaluate4S.m:307).
    """
    n = u.size
    n_bins = n // npb
    mx = np.full(n_bins, np.nan)
    my = np.full(n_bins, np.nan)
    mt = np.full(n_bins, np.nan)
    if n_bins == 0:
        return mx, my, mt
    edges = matlab_prctile(u, matlab_colon(0.0, 100.0 / n_bins, 100.0))
    lo = edges[:-1]
    hi = edges[1:]
    member = (u[None, :] >= lo[:, None]) & (u[None, :] <= hi[:, None])
    cnt = member.sum(axis=1)
    keep = cnt >= npb
    if keep.any():
        # mean(x(ix)) adds the members in time order. Non-members add 0, which
        # leaves the running sum unchanged.
        vals = np.column_stack([u, nee, ta])
        sums = sequential_sum(member[keep][:, :, None] * vals[None, :, :], axis=1)
        means = sums / cnt[keep, None]
        k = int(keep.sum())
        mx[:k] = means[:, 0]
        my[:k] = means[:, 1]
        mt[:k] = means[:, 2]
    return mx, my, mt


def evaluate_draw(t, nee, ustar, ta, night, dec):
    """One run of cpdEvaluateUStarTh4Season20100901 on (resampled) records.

    Inputs are the records of one draw in time order. ``ustar`` must already be
    NaN outside 0..4 (Bootstrap.m:75). ``dec`` flags records MATLAB dates in
    December (see ``december_flags``). Returns ``(cp2, stats)``: ``cp2`` is a
    (4, 8) array and ``stats`` a dict of (4, 8) arrays over STATS_FIELDS. Strata
    that were not evaluated stay NaN.
    """
    cp2 = np.full((N_SEASONS, N_STRATA_MAX), np.nan)
    stats = {f: np.full((N_SEASONS, N_STRATA_MAX), np.nan) for f in STATS_FIELDS}

    npb = n_per_bin(n_per_day_from_t(t))
    n_needed = N_SEASONS * N_STRATA_MIN * N_BINS * npb

    # u* > 3 rejected here (line 225), on top of the 0..4 range of the bootstrap
    with np.errstate(invalid='ignore'):
        ustar = np.where((ustar < 0) | (ustar > 3), np.nan, ustar)
    valid = (night == 1) & ~np.isnan(nee + ustar + ta)
    nt_annual = int(valid.sum())
    if nt_annual < n_needed:
        return cp2, stats
    n_per_season = matlab_round(nt_annual / N_SEASONS)

    # Move December to the start, dated as the previous year (lines 256-260).
    # Without a December record MATLAB's reorder vector is empty, and nothing is evaluated.
    i_dec = np.flatnonzero(dec)
    if i_dec.size == 0:
        return cp2, stats
    first = int(i_dec[0])
    t = t.copy()
    t[i_dec] -= 366.0  # EndDOY: fcDoy(datenum(0,12,31.5)) = 366, MATLAB year 0 is a leap year
    order = np.r_[first:t.size, 0:first]
    t = t[order]
    ta = ta[order]
    ustar = ustar[order]
    nee = nee[order]
    valid = valid[order]

    it_annual = np.flatnonzero(valid)
    nt_annual = it_annual.size
    n_seasons = int(matlab_round(nt_annual / n_per_season))
    n_per_season = int(matlab_round(nt_annual / n_seasons))
    if n_seasons != N_SEASONS:  # cannot happen for nt_annual >= n_needed
        raise RuntimeError(f"Unexpected number of seasons: {n_seasons}")

    t_a = t[it_annual]
    ta_a = ta[it_annual]
    u_a = ustar[it_annual]
    nee_a = nee[it_annual]

    for i_season in range(n_seasons):
        j0 = i_season * n_per_season
        j1 = nt_annual if i_season == n_seasons - 1 else (i_season + 1) * n_per_season
        ta_s = ta_a[j0:j1]
        nt_season = ta_s.size
        n_strata = nt_season // (N_BINS * npb)
        n_strata = min(max(n_strata, N_STRATA_MIN), N_STRATA_MAX)
        t_th = matlab_prctile(ta_s, matlab_colon(0.0, 100.0 / n_strata, 100.0))
        u_s = u_a[j0:j1]
        nee_s = nee_a[j0:j1]
        t_s = t_a[j0:j1]
        for i_strata in range(n_strata):
            # Inclusive at both ends: a record on an edge is in two strata (line 298)
            sel = (ta_s >= t_th[i_strata]) & (ta_s <= t_th[i_strata + 1])
            u = u_s[sel]
            y = nee_s[sel]
            tt = ta_s[sel]
            mx, my, mtemp = bin_stratum(u, y, tt, npb)
            c, s = find_change_point(mx, my)
            r, p = _corrcoef_p(mx, mtemp)
            ts = t_s[sel]
            s['mt'] = ts.mean()
            s['ti'] = ts[0]
            s['tf'] = ts[-1]
            s['ruStarVsT'] = r
            s['puStarVsT'] = p
            s['mT'] = tt.mean()
            ci = matlab_prctile(tt, [2.5, 97.5])
            s['ciT'] = 0.5 * (ci[1] - ci[0])
            cp2[i_season, i_strata] = c
            for f in STATS_FIELDS:
                stats[f][i_season, i_strata] = s[f]
    return cp2, stats


def december_flags(t: np.ndarray) -> np.ndarray:
    """Records that ``fcDatevec`` (Barr's mydatevec) puts in December.

    ``t`` is the fractional day count of launch.m (1 + 1/nrPerDay for the first
    record). MATLAB reads it as a serial date number, so t = 1 is 1 January of
    year 0, a leap year: December is days 336-366 of year 0 whatever the real
    year. In a non-leap year the first real day of December therefore counts as
    November. mydatevec gives 00:00 as 24:00 of the previous day. ``datevec``
    rounds to the millisecond first. That only matters for hourly data, where the
    accumulated t of row 8040 (t = 336, the record ending on 1 December 00:00 in a
    leap year, 2 December 00:00 otherwise) is 1.3e-11 days above 336. With the
    rounding it stays in November. Checked against the compiled ONEFlux
    ``ustar_cp``: 9 hourly site-years agree in all 900 values with the rounding,
    only 374 agree without it (L371).
    """
    day = np.floor(t)
    ms = matlab_round((t - day) * 86400000.0)
    day = day + (ms >= 86400000.0)
    ms = np.where(ms >= 86400000.0, 0.0, ms)
    eff_day = np.where(ms == 0.0, day - 1.0, day)
    # Day numbers of year 0: 336..366 is December. 0 is 31 December of year -1.
    return ((eff_day >= 336.0) & (eff_day <= 366.0)) | (eff_day <= 0.0)


def _flatten_matlab(a: np.ndarray) -> np.ndarray:
    """(nBoot, 4, 8) array as MATLAB's column vector of a 4 x 8 x nBoot array."""
    return np.transpose(a, (0, 2, 1)).reshape(-1)


def _iqr_matlab(col: np.ndarray) -> float:
    """fcNaniqr.m (2-D case): prctile 75 - 25 of the non-NaN values, NaN if <= 3."""
    v = col[~np.isnan(col)]
    if v.size <= 3:
        return np.nan
    q = matlab_prctile(v, [25.0, 75.0])
    return float(q[1] - q[0])


def assign_annual(cp: np.ndarray, b1: np.ndarray, cib1: np.ndarray,
                  p: np.ndarray, mt: np.ndarray) -> dict:
    """Annual change points from Stats2 of all draws (cpdAssignUStarTh20100901.m).

    Inputs are (nBoot, 4, 8) arrays of the Stats2 fields. Returns a dict with
    ``cp_annual`` (nBoot,), ``n_selected`` (nBoot,), ``selected`` (nBoot, 4, 8),
    ``mode``, ``failure`` ('' on success, the ONEFlux cFailure text otherwise)
    and the fractions FracSig, FracModeD, FracSelect.

    The steps are those of the 3-D (bootstrap) branch: significant change points
    (p <= 0.05), the dominant mode (D: b1 >= 0, kept on ties, or E), the 10 %
    minimum success rate, the outlier screen and the 4*nBoot minimum. The
    outlier screen keeps a bug of the original (lines 135-136): every column of
    [Cp b1 cib1] is replaced by ``(Cp - median(column))/IQR(column)``, so Cp is
    compared with the medians and IQRs of b1 and cib1 too, and any of the three
    above 5 removes the point. On failure ``cp_annual`` is all NaN, because
    MATLAB returns before computing it.
    """
    n_boot = cp.shape[0]
    shape = cp.shape
    out = dict(cp_annual=np.full(n_boot, np.nan), n_selected=np.full(n_boot, np.nan),
               selected=np.zeros(shape, dtype=bool), mode='', failure='',
               frac_sig=np.nan, frac_mode_d=np.nan, frac_select=np.nan)
    cpv = _flatten_matlab(cp)
    b1v = _flatten_matlab(b1)
    cib1v = _flatten_matlab(cib1)
    pv = _flatten_matlab(p)
    mtv = _flatten_matlab(mt)
    c2v = 0.0 * b1v  # 2-parameter Stats: c2 is all NaN, so nPar = 2 and c2 = 0 (line 101)

    with np.errstate(invalid='ignore'):
        f_p = pv <= P_SIG
        ok = ~np.isnan(b1v + c2v + cpv)
        n_try = int(np.count_nonzero(~np.isnan(mtv)))
        n_sig = int(np.count_nonzero(f_p & ok))
        mode_e = f_p & (b1v < c2v)
        mode_d = f_p & (b1v >= c2v)
    n_mode_d = int(mode_d.sum())
    n_mode_e = int(mode_e.sum())
    if n_mode_d >= n_mode_e:
        f_select = mode_d.copy()
        mode = 'D'
    else:
        f_select = mode_e.copy()
        mode = 'E'
    out['mode'] = mode
    n_select = int(f_select.sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_select = n_select / n_try if n_try else np.nan
        out.update(frac_sig=n_sig / n_try if n_try else np.nan,
                   frac_mode_d=n_mode_d / n_sig if n_sig else np.nan,
                   frac_select=frac_select)
    # NaN < 0.10 is false in MATLAB, so a run with nothing tried passes this check
    if frac_select < 0.10:
        out['failure'] = 'Less than 10% successful detections. '
        return out

    x = np.column_stack([cpv, b1v, cib1v])
    with np.errstate(all='ignore'):
        mx = np.array([np.median(c[~np.isnan(c)]) if (~np.isnan(c)).any() else np.nan for c in x.T])
        sx = np.array([_iqr_matlab(c) for c in x.T])
        # The column bug: x(:,1) in every column
        x_norm = (x[:, [0]] - mx[None, :]) / sx[None, :]
        x_norm_max = np.fmax.reduce(np.abs(x_norm), axis=1)  # max ignores NaN
        f_out = x_norm_max > 5.0
    f_select &= ~f_out
    mode_d &= ~f_out
    mode_e &= ~f_out
    n_select = int(f_select.sum())
    n_sig = int((mode_d | mode_e).sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        out.update(frac_sig=n_sig / n_try if n_try else np.nan,
                   frac_mode_d=int(mode_d.sum()) / n_sig if n_sig else np.nan,
                   frac_select=n_select / n_try if n_try else np.nan)
    n_select_min = 4 * n_boot
    if n_select < n_select_min:
        out['failure'] = f'Too few selected change points: {n_select:g}/{n_select_min:g}'
        return out

    sel = f_select.reshape(n_boot, N_STRATA_MAX, N_SEASONS).transpose(0, 2, 1)
    out['selected'] = sel
    # nanmean (Statistics Toolbox): NaN -> 0, sum of the 32 values of a draw in
    # MATLAB order, divided by the count (lines 151-156)
    cp_sel = f_select.reshape(n_boot, -1) * np.nan_to_num(cpv.reshape(n_boot, -1))
    cnt = f_select.reshape(n_boot, -1).sum(axis=1)
    out['n_selected'] = cnt.astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        out['cp_annual'] = np.where(cnt > 0, sequential_sum(cp_sel, axis=1) / cnt, np.nan)
    return out
