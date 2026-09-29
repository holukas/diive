"""
SIMILARITY: METEOROLOGICAL SIMILARITY FOR MDS-STYLE METHODS
============================================================

Meteorological-similarity primitives for MDS gap-filling (Reichstein et al.
2005) and PAS20 random-uncertainty estimation (Pastorello et al. 2020 /
ONEFlux). Both pool measured fluxes that occur under "similar" meteorological
conditions (SWIN, TA, VPD) and reduce them to a statistic — the mean for
gap-filling, the standard deviation for uncertainty. The random-uncertainty
step shares only the tolerance constants and :func:`swin_tolerance`; it runs
its own window loop. The MDS cascade (:func:`mds_gapfill_cascade`) serves the
MDS gap-filler and the NEE uncertainty of the daytime ONEFlux partitioning.

The tolerance values mirror the ONEFlux ``GF_DRIVER_*`` defines in
``oneflux_steps/common/common.h``.

SWIN tolerance rule: ONEFlux (both ``common.c`` gap-filling and ``randunc.c``)
clamps the *target* record's own SWIN into ``[20, 50]`` — a continuous tolerance
that grows with radiation (see :func:`swin_tolerance`). Both the MDS gap-filler
and the random-uncertainty step use this rule.

Part of the diive library: https://github.com/holukas/diive
"""
import numpy as np

# ONEFlux meteorological-similarity tolerances (oneflux_steps/common/common.h).
SWIN_TOLERANCE_MIN = 20.0  # W m-2   (GF_DRIVER_1_TOLERANCE_MIN)
SWIN_TOLERANCE_MAX = 50.0  # W m-2   (GF_DRIVER_1_TOLERANCE_MAX)
TA_TOLERANCE = 2.5         # deg C   (GF_DRIVER_2A_TOLERANCE_MIN)
VPD_TOLERANCE = 5.0        # hPa     (GF_DRIVER_2B_TOLERANCE_MIN)


def swin_tolerance(swin, tol_min: float = SWIN_TOLERANCE_MIN, tol_max: float = SWIN_TOLERANCE_MAX):
    """SWIN similarity tolerance for a given radiation level.

    ONEFlux clamps the record's own SWIN value into ``[tol_min, tol_max]``
    (``common.c`` / ``randunc.c``): the tolerance equals the radiation up to
    ``tol_max`` and never drops below ``tol_min``. Accepts a scalar or a numpy
    array and returns the same shape.
    """
    return np.clip(swin, tol_min, tol_max)


def window_mean_sd_count(values, min_count: int, ddof: int = 0):
    """Reduce a window of (possibly NaN) flux values to ``(mean, sd, count)``.

    Drops NaNs, then returns the mean, standard deviation and the number of
    valid values — but mean and SD are NaN unless at least ``min_count`` valid
    values are present. ``count`` is always the true number of valid values.

    ``ddof`` selects the standard-deviation convention: ``0`` (population) or
    ``1`` (sample / N-1, the ONEFlux convention used by both the MDS gap-filler
    and the random-uncertainty step).
    """
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    count = int(valid.size)
    if count == 0:
        return np.nan, np.nan, 0
    if count >= min_count and count > ddof:
        return float(np.mean(valid)), float(np.std(valid, ddof=ddof)), count
    return np.nan, np.nan, count


# --------------------------------------------------------------------------- #
# Shared MDS cascade (ONEFlux marginal-distribution sampling)
# --------------------------------------------------------------------------- #
# The 6-loop window-expansion cascade below is the single implementation behind
# diive's MDS gap-filler AND the daytime-partitioning NEE-uncertainty step.
# ONEFlux has two versions of it, and ``edge`` picks the one a caller matches:
# ``'trim'`` the C ``gf_mds`` in ``common.c`` (the MDS gap-filler), ``'clip'``
# the Python ``uncert_via_gapFill`` in ``oneflux/partition/daytime.py`` (the
# daytime-partitioning uncertainty). The drivers-based loops 1, 2, 4 and 5 are
# the same in both. The diurnal-cycle loops 3 and 6, the window edges and the
# summation order differ (see ``mds_gapfill_cascade``). The quality collapse
# is byte-identical to both. Operates on integer record positions of a regular
# (gap-free index) half-hourly/hourly grid, using the np.nan "missing"
# convention throughout.

#: Base window width in days; the cascade expands in multiples of this.
_TW_ORIG = 14

#: MDS driver method codes (mirror the ONEFlux ``fillMethod`` values).
METHOD_ALL = 1      # all three drivers (SWIN + TA + VPD) similar
METHOD_SWIN = 2     # main driver only (SWIN) similar
METHOD_MDC = 3      # mean diurnal cycle: same time-of-day (+/- 1 h)


def meteo_similar_mask(w, index, *, swin=None, ta=None, vpd=None, hr=None,
                       swin_tol=(SWIN_TOLERANCE_MIN, SWIN_TOLERANCE_MAX),
                       ta_tol: float = TA_TOLERANCE, vpd_tol: float = VPD_TOLERANCE,
                       hr_tol: float = None):
    """Boolean "meteorologically similar to record ``index``" mask over ``w``.

    ``w`` is an array of candidate record positions; ``index`` the target
    record. Each supplied driver adds a strict ``< tolerance`` constraint on the
    absolute difference to the target (and requires the candidate driver to be
    finite). The drivers-based MDS cascade passes (methods 1 and 2) use it.
    The SWIN tolerance is clamped to the target's own radiation level
    (:func:`swin_tolerance`). Pass ``hr`` with ``hr_tol`` (e.g. 1.1) for a
    +/- 1 h time-of-day (diurnal) constraint.
    """
    mask = np.ones(w.shape, dtype=bool)
    if ta is not None:
        mask &= (np.abs(ta[w] - ta[index]) < ta_tol) & np.isfinite(ta[w])
    if swin is not None:
        tol = swin_tolerance(swin[index], swin_tol[0], swin_tol[1])
        mask &= (np.abs(swin[w] - swin[index]) < tol) & np.isfinite(swin[w])
    if vpd is not None:
        mask &= (np.abs(vpd[w] - vpd[index]) < vpd_tol) & np.isfinite(vpd[w])
    if hr is not None and hr_tol is not None:
        mask &= np.abs(hr[w] - hr[index]) < hr_tol
    return mask


def mds_quality_from(method, time_window):
    """Collapse ``(method, time_window)`` to the ONEFlux 1/2/3 quality flag.

    Byte-identical to the ``fillQC`` formula in ONEFlux ``uncert_via_gapFill``
    (and the C ``gf_mds`` quality block). Accepts scalars or numpy arrays;
    returns the same shape. ``method == 0`` (measured / unfilled) gives 0.
    """
    m = np.asarray(method)
    tw = np.asarray(time_window)
    q = (m > 0).astype(int)
    q = q + (((m == METHOD_ALL) & (tw > 14)) | ((m == METHOD_SWIN) & (tw > 14))
             | ((m == METHOD_MDC) & (tw > 1))).astype(int)
    q = q + (((m == METHOD_ALL) & (tw > 56)) | ((m == METHOD_SWIN) & (tw > 28))
             | ((m == METHOD_MDC) & (tw > 5))).astype(int)
    return q if q.ndim else int(q)


def mds_granular_flag(method, time_window):
    """Granular MDS gap-fill flag encoding ``(method, time_window)``.

    Richer than the collapsed 1/2/3 quality: ``method * 1000 + round(time_window)``
    so the driver method and exact window are both recoverable
    (``method = flag // 1000``, ``time_window = flag % 1000``). ``method == 0``
    (measured) maps to 0. E.g. ALL @ 14-day window -> ``1014``, SWIN @ 28 ->
    ``2028``, MDC @ 1 -> ``3001``. The window is capped at 999 in the flag:
    gf_mds's diurnal-cycle windows can grow past 1000 days on a multi-year
    record, which would otherwise spill into the method digit. 999 then means
    "999 days or more"; gf_mds's window sequence (7, 13, 19, ...) never hits
    999 exactly.
    """
    m = np.asarray(method)
    tw = np.asarray(time_window)
    flag = np.where(m > 0, m * 1000 + np.minimum(np.rint(tw).astype(int), 999), 0)
    return flag if flag.ndim else int(flag)


def mds_gapfill_cascade(tofill, swin, ta, vpd, hr, nperday, *,
                        min_samples: int = 2,
                        swin_tol=(SWIN_TOLERANCE_MIN, SWIN_TOLERANCE_MAX),
                        ta_tol: float = TA_TOLERANCE,
                        vpd_tol: float = VPD_TOLERANCE,
                        ddof: int = 1,
                        sym_mean: bool = False,
                        fill_all: bool = False,
                        longest_marginal_gap: int = 60,
                        edge: str = 'trim',
                        sd_func=None,
                        sample_ok=None,
                        progress_callback=None):
    """Faithful ONEFlux MDS marginal-distribution-sampling cascade.

    Six expanding-window passes (first success per record wins), a faithful port
    of ONEFlux C ``gf_mds`` (``edge='trim'``) or ONEFlux Python
    ``uncert_via_gapFill`` (``edge='clip'``), see module header:

      1. all drivers (SWIN+TA+VPD), windows 14 & 28 days  (method 1)
      2. SWIN only, 14 days                               (method 2)
      3. diurnal +/-1 h, windows 1, 3, 5 days             (method 3)
      4. all drivers, windows 42..154 days                (method 1)
      5. SWIN only, windows 28..154 days                  (method 2)
      6. diurnal +/-1 h, windows 7, 13, 19, ... days until the window covers
         the whole record (``'trim'``), or 7, 14, ..., 427 days (``'clip'``)
                                                          (method 3)

    Args:
        tofill: target values to gap-fill (np.nan = missing).
        swin/ta/vpd: similarity drivers (np.nan = missing). Units must match the
            tolerances (SWIN W m-2, TA deg C, VPD same unit as ``vpd_tol``).
        hr: time-of-day in hours (e.g. 0.0, 0.5, ..., 23.5) for the diurnal
            match. Used only with ``edge='clip'``.
        nperday: records per day (48 half-hourly, 24 hourly).
        min_samples: minimum similar samples to accept a fill. ONEFlux gap-fill
            uses 2 (``>1``); the uncertainty variant uses 10 (``>9``).
        swin_tol: ``(min, max)`` clamp bounds for the SWIN tolerance.
        ta_tol/vpd_tol: TA / VPD similarity tolerances.
        ddof: standard-deviation convention (1 = ONEFlux sample SD).
        sym_mean: use the Vekuri (2023) symmetric mean for the SWIN-driven
            methods (1 & 2) instead of the plain mean; off by default.
        fill_all: predict at every record (uncertainty mode, ``compute_hat``);
            otherwise only at missing ``tofill`` records (gap-filling mode).
        longest_marginal_gap: leading/trailing gaps longer than this many days
            are left unfilled (ONEFlux ``longestMarginalgap``).
        edge: which ONEFlux reference the cascade follows, named after how a
            look-up window that runs past the start or the end of the record
            is handled. ``'trim'`` (the C ``gf_mds`` tool) drops the
            out-of-range offsets, takes the diurnal-cycle samples by record
            position (same time of day +/- 1 h on whole days, across midnight
            too), grows the last diurnal window by 3 days a side, and sums the
            samples one after the other in time order, as the C code does.
            ``'clip'`` (the Python ``daytime.uncert_via_gapFill``) folds
            out-of-range offsets onto record 0 / n-1 so the edge record enters
            the mean, SD and count repeatedly, matches the time of day on
            ``hr`` (not across midnight), grows the last diurnal window by 3.5
            days a side and reduces with ``np.mean`` / ``np.std``.
        sample_ok: optional ``dict`` mapping ``'swin'``/``'ta'``/``'vpd'`` to a
            boolean array: False where that driver's value may not serve as a
            look-up sample (ONEFlux driver QC above threshold). The record
            being filled still compares against its own driver value, and the
            diurnal-cycle method ignores it. Mirrors C ``gf_mds``: method 1
            needs all three drivers eligible, method 2 only SWIN. Off
            (``None``) by default.
        sd_func: optional ``callable(values, ddof)`` that replaces
            ``np.std(values, ddof=ddof)`` for the SD of the similar samples, for
            a caller whose reference computes the SD with different arithmetic.
        progress_callback: optional ``callable(gaps_filled, total_gaps, quality)``
            invoked during/after each cascade pass (gap counts + the current
            1/2/3 quality), for a progress bar.

    Returns:
        dict of per-record arrays (np.nan / 0 where unfilled): ``filled`` (mean),
        ``sd``, ``count``, ``method`` (0/1/2/3), ``time_window`` (days),
        ``quality`` (0/1/2/3), ``flag`` (granular, see :func:`mds_granular_flag`).
    """
    # Preserve the input dtype for the tolerance comparisons: callers that store
    # float32 (ONEFlux FLOAT_PREC, e.g. the daytime-partitioning uncertainty)
    # get float32 boundary behaviour matching native ONEFlux, while the float64
    # MDS gap-filler keeps full precision. The mean/SD reduction runs in the
    # input dtype for the same reason (see ``fill_at``).
    tofill = np.asarray(tofill)
    swin = np.asarray(swin)
    ta = np.asarray(ta)
    vpd = np.asarray(vpd)
    hr = np.asarray(hr)
    n = tofill.size

    filled = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    count = np.zeros(n, dtype=int)
    method = np.zeros(n, dtype=int)
    time_window = np.zeros(n, dtype=float)

    target_valid = np.isfinite(tofill)
    ok = np.where(target_valid)[0]
    if ok.size == 0:
        return dict(filled=filled, sd=sd, count=count, method=method,
                    time_window=time_window, quality=method.copy(),
                    flag=method.copy())

    # Leading / trailing gaps longer than longest_marginal_gap days are excluded
    # (ONEFlux largemarginGap), reckoned in records (longest_marginal_gap * 48,
    # matching ONEFlux's hardcoded 48 regardless of nperday).
    large = np.zeros(n, dtype=bool)
    firstvalid, lastvalid = int(ok.min()), int(ok.max())
    if firstvalid > (48 * longest_marginal_gap):
        large[:(firstvalid + 1 - (48 * longest_marginal_gap))] = True
    if lastvalid < (n - (48 * longest_marginal_gap)):
        large[(lastvalid + (48 * longest_marginal_gap)):] = True

    # Records eligible to receive a fill: not in an excluded marginal gap, and
    # (gap-filling mode) originally missing — uncertainty mode predicts all.
    eligible = ~large if fill_all else (~large & ~target_valid)
    # Progress is reported over the actual gaps (originally-missing eligible
    # records), even in fill_all mode where measured records are also predicted.
    gap_eligible = ~large & ~target_valid

    # 'trim' follows the C gf_mds tool in full, 'clip' ONEFlux's Python
    # uncert_via_gapFill (see the `edge` argument).
    c_gf_mds = edge != 'clip'

    def gaps():
        return np.where(eligible & ~np.isfinite(filled))[0]

    _offset_cache = {}

    def window_idx(index, t_window):
        off = _offset_cache.get(t_window)
        if off is None:
            off = np.append(-np.arange(t_window / 2.0 * nperday),
                            np.arange(t_window / 2.0 * nperday - 1) + 1)
            if c_gf_mds:
                # gf_mds walks the window from its start to its end, so its
                # samples, and the order it sums them in, run forward in time.
                off = np.sort(off)
            _offset_cache[t_window] = off
        w = index + off
        if edge == 'clip':
            # Fold out-of-range offsets onto record 0 / n-1, so the edge record
            # enters the mean, SD and count once per folded offset. This is what
            # ONEFlux's own Python `daytime.uncert_via_gapFill` does
            # (`numpy.clip(w, 0, n - 1, out=w)` in each of its six loops), and
            # the daytime-partitioning uncertainty is measured against that.
            w = np.clip(w, 0, n - 1)
        else:
            # Trim to the record: ONEFlux's C `gf_mds` tool narrows the window
            # bounds instead (`common.c:2525-2533`: `if (window_start < 0)
            # window_start = 0;` / `if (window_end > end_window) window_end =
            # end_window;` and then loops `window_current < window_end`), and its
            # diurnal method skips out-of-range positions outright (`:2630`).
            # Either way a real record enters a fill at most once. This is the
            # MDS gap-filler's reference, so it stays the default.
            w = w[(w >= 0) & (w < n)]
        return w.astype(int)

    def fill_at(index, sel, m, tw):
        # Reduce in the input dtype: float32 callers (daytime uncertainty) match
        # ONEFlux's float32 tmean; the float64 MDS gap-filler stays f8. The SD
        # can be replaced through `sd_func` (the daytime uncertainty passes
        # ONEFlux's scipy tstd arithmetic).
        vals = tofill[sel]
        # Symmetric mean (Vekuri 2023): split the similar samples by whether the
        # candidate's SWIN is above/below the target's, average the two sub-means
        # (ONEFlux gf_get_similiar_mean_sym_mean; SWIN-driven methods 1 & 2 only,
        # candidates equal to the target count in both halves).
        if sym_mean and m in (METHOD_ALL, METHOD_SWIN):
            cs = swin[sel]
            parts = [vals[cs >= swin[index]].mean() if np.any(cs >= swin[index]) else np.nan,
                     vals[cs <= swin[index]].mean() if np.any(cs <= swin[index]) else np.nan]
            parts = [p for p in parts if np.isfinite(p)]
            filled[index] = np.mean(parts) if parts else np.nan
        elif c_gf_mds:
            # gf_mds adds the samples one after the other in time order
            # (common.c gf_get_similiar_mean). np.mean sums pairwise, which
            # rounds differently in the last bits; np.add.accumulate does not.
            filled[index] = np.add.accumulate(vals)[-1] / vals.size
        else:
            filled[index] = np.mean(vals)
        if vals.size <= ddof:
            sd[index] = np.nan
        elif sd_func is not None:
            sd[index] = sd_func(vals, ddof)
        elif c_gf_mds:
            # common.c gf_get_similiar_standard_deviation: sequential sum of
            # squared deviations from the (sequential, plain) mean
            dev = vals - np.add.accumulate(vals)[-1] / vals.size
            sd[index] = np.sqrt(np.add.accumulate(dev * dev)[-1] / (vals.size - ddof))
        else:
            sd[index] = np.std(vals, ddof=ddof)
        count[index] = vals.size
        method[index] = m
        time_window[index] = tw

    total_gaps = int(gap_eligible.sum())
    _PROGRESS_EVERY = 1024  # emit within a pass every ~1k gaps for a smooth bar

    def emit(quality):
        if progress_callback is not None:
            progress_callback(int(np.isfinite(filled[gap_eligible]).sum()),
                              total_gaps, int(quality))

    def meteo_pass(t_window, m, *, drivers):
        """One pass with the all-drivers / SWIN-only meteo predicate."""
        q = mds_quality_from(m, t_window)
        for k, index in enumerate(gaps()):
            if progress_callback is not None and (k % _PROGRESS_EVERY) == 0:
                emit(q)
            w = window_idx(index, t_window)
            nongap = w[np.isfinite(tofill[w])]
            if nongap.size < min_samples:
                continue
            similar = meteo_similar_mask(nongap, index, swin_tol=swin_tol,
                                         ta_tol=ta_tol, vpd_tol=vpd_tol, **drivers)
            if sample_ok is not None:
                # gf_mds clears a driver's valid bit when its QC is above the
                # threshold (common.c gf_mds mask loop), which removes the
                # record from the samples of every method using that driver
                for name in drivers:
                    if name in sample_ok:
                        similar &= sample_ok[name][nongap]
            sel = nongap[similar]
            if sel.size >= min_samples:
                fill_at(index, sel, m, t_window)
        emit(q)

    def mdc_pass(t_window):
        """One diurnal (+/- 1 h, same time-of-day) pass."""
        q = mds_quality_from(METHOD_MDC, t_window)
        for k, index in enumerate(gaps()):
            if progress_callback is not None and (k % _PROGRESS_EVERY) == 0:
                emit(q)
            w = window_idx(index, t_window)
            nongap = w[np.isfinite(tofill[w])]
            sel = nongap[np.abs(hr[nongap] - hr[index]) < 1.1]
            if sel.size >= min_samples:
                fill_at(index, sel, METHOD_MDC, t_window)
        emit(q)

    def mdc_pass_gf_mds(days):
        """One diurnal pass as C gf_mds does it (common.c gapfill, GF_TOFILL_METHOD).

        The samples are the records at the same position of each day within
        +/- `days` days, +/- 1 h: by record position, so the hour before
        midnight counts for a record just after it (the `hr` match of the
        Python reference drops it). The window is reported as 2 * days + 1.
        """
        per_h = nperday // 24
        off = (np.arange(-days, days + 1)[:, None] * nperday
               + np.arange(-per_h, per_h + 1)[None, :]).ravel()
        tw = 2 * days + 1
        q = mds_quality_from(METHOD_MDC, tw)
        for k, index in enumerate(gaps()):
            if progress_callback is not None and (k % _PROGRESS_EVERY) == 0:
                emit(q)
            w = index + off
            w = w[(w >= 0) & (w < n)]
            sel = w[np.isfinite(tofill[w])]
            if sel.size >= min_samples:
                fill_at(index, sel, METHOD_MDC, tw)
        emit(q)

    # Loop 1: all drivers, windows 14 & 28 days
    for it in range(2):
        if gaps().size == 0:
            break
        meteo_pass((it + 1) * _TW_ORIG, METHOD_ALL, drivers=dict(swin=swin, ta=ta, vpd=vpd))
    # Loop 2: SWIN only, window 14 days
    if gaps().size:
        meteo_pass(_TW_ORIG, METHOD_SWIN, drivers=dict(swin=swin))
    # Loop 3: diurnal, windows 1, 3, 5 days
    for it in range(3):
        if gaps().size == 0:
            break
        if c_gf_mds:
            mdc_pass_gf_mds(it)
        else:
            mdc_pass((2 * it + 1) * 1)
    # Loop 4: all drivers, windows 42..154 days
    for it in range(2, 11):
        if gaps().size == 0:
            break
        meteo_pass((it + 1) * _TW_ORIG, METHOD_ALL, drivers=dict(swin=swin, ta=ta, vpd=vpd))
    # Loop 5: SWIN only, windows 28..154 days
    for it in range(1, 11):
        if gaps().size == 0:
            break
        meteo_pass((it + 1) * _TW_ORIG, METHOD_SWIN, drivers=dict(swin=swin))
    # Loop 6: diurnal. gf_mds: +/- 3, 6, 9, ... days (windows 7, 13, 19, ...)
    # until the window covers the whole record; a larger window adds nothing.
    # The Python reference: windows 7, 14, ..., 427 days.
    if c_gf_mds:
        days = 3
        while gaps().size:
            mdc_pass_gf_mds(days)
            if days * nperday >= n:
                break
            days += 3
    else:
        for it in range(61):
            if gaps().size == 0:
                break
            mdc_pass((it + 1) * (_TW_ORIG * 0.5))

    quality = mds_quality_from(method, time_window)
    flag = mds_granular_flag(method, time_window)
    return dict(filled=filled, sd=sd, count=count, method=method,
                time_window=time_window, quality=quality, flag=flag)
