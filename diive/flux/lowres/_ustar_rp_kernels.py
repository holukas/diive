"""
USTAR RP KERNELS: NUMBA CORE OF THE REDDYPROC U* THRESHOLD PORT
===============================================================

The per-season estimate of REddyProc 1.3.4 (``.estimateUStarSeason`` and the
helpers it calls in ``EddyUStarFilterDP.R``) as numba kernels. Line numbers
below refer to that file. Orchestration (seasons, years, pooling, bootstrap)
lives in :mod:`diive.flux.lowres.ustar_reddyproc_detection`.

Part of the diive library: https://github.com/holukas/diive
"""
import numpy as np
from numba import njit

from diive.flux.lowres._ustar_rp_numerics import abs_cor_below, r_mean

# Status codes of estimate_subset
OK = 0
# .binWithEqualValuesBalanced found no value change at or after the start of
# a temperature class (DP:992-996): R stops with "NA/NaN argument".
ERR_TEMP_BREAK = 1


@njit(cache=True)
def first_last(season_idx, n_seasons):
    """Position of the first and the last record of every season."""
    first = np.full(n_seasons, -1, dtype=np.int64)
    last = np.full(n_seasons, -1, dtype=np.int64)
    for i in range(season_idx.shape[0]):
        s = season_idx[i]
        if first[s] < 0:
            first[s] = i
        last[s] = i
    return first, last


@njit(cache=True)
def which_ge(x, threshold, i_start):
    """``whichValueGreaterEqualC`` (DP:1055-1085), 1-based in and out: the
    first index ``i >= i_start`` with ``x[i] >= threshold``, 0 for R's NA."""
    n = x.shape[0]
    if i_start < 1:
        i_start = 1
    for i in range(i_start - 1, n):
        if x[i] >= threshold:
            return i + 1
    return 0


@njit(cache=True)
def bin_equal_values_balanced(x, n_bin, tol):
    """``.binWithEqualValuesBalanced`` (DP:968-1000) on sorted ``x``.

    Class sizes are ``floor(n / n_bin)``. A class starts at the first value
    change at or after its nominal start, so equal values share a class and
    the following class gets shorter. Returns 1-based class ids and a status
    (``ERR_TEMP_BREAK`` where R stops). ``n_bin == 1`` returns zeros, as R's
    ``integer(length(x))`` does (DP:977).
    """
    n = x.shape[0]
    bin_id = np.ones(n, dtype=np.int64)
    if n_bin == 1:
        return np.zeros(n, dtype=np.int64), OK
    bin_size = np.floor(n / n_bin)
    nb = 0
    breaks = np.empty(max(n - 1, 0), dtype=np.int64)
    for i in range(n - 1):
        if x[i + 1] - x[i] > tol:
            breaks[nb] = i + 2  # 1-based position of the new value
            nb += 1
    breaks = breaks[:nb]
    i_break = 1
    for i_class in range(2, n_bin + 1):
        # R: round((iClass - 1) * binSize) + 1, an integer-valued double
        start0 = (i_class - 1) * bin_size + 1
        i_break = which_ge(breaks, start0, i_break)
        if i_break == 0:
            return bin_id, ERR_TEMP_BREAK
        start1 = breaks[i_break - 1]
        bin_id[start1 - 1:] = i_class
    return bin_id, OK


@njit(cache=True)
def bin_equal_values_min_rec(x, n_bin, tol):
    """``.binWithEqualValuesMinRec`` (DP:1002-1053) on sorted ``x``.

    Sequential classes of at least ``floor(n / n_bin)`` records, each
    extended to the next value change (difference > ``tol``). If the rest
    after the last change is shorter than 90 % of a class it joins the
    previous class (DP:1029-1030). Ids above ``n_bin`` are set to ``n_bin``
    (DP:1050), so the last class can be large.
    """
    n = x.shape[0]
    bin_id = np.zeros(n, dtype=np.int64)
    bin_size = np.int64(np.floor(n / n_bin))
    nb = 0
    breaks = np.empty(max(n - 1, 0), dtype=np.int64)
    for i in range(n - 1):
        if x[i + 1] - x[i] > tol:
            breaks[nb] = i + 1  # 1-based: a break after position i + 1
            nb += 1
    breaks = breaks[:nb]
    i_break = 0
    i_end = 0
    i_bin = 0
    while i_end < n:
        i_bin += 1
        i_start = i_end + 1
        i_end = i_end + bin_size
        i_break = which_ge(breaks, i_end, i_break + 1)
        if i_break == 0:
            if (n + 1 - i_start) < bin_size * 0.9 and i_bin != 1:
                i_bin -= 1
            bin_id[i_start - 1:] = i_bin
            break
        i_end = breaks[i_break - 1]
        bin_id[i_start - 1:i_end] = i_bin
    for i in range(n):
        if bin_id[i] > n_bin:
            bin_id[i] = n_bin
    return bin_id


@njit(cache=True)
def fw2(nee_avg, ust_avg, n_bins, plateau_fwd, plateau_crit, min_n_plateau):
    """``usEstUstarThresholdSingleFw2Binned`` (DP:1129-1167).

    The first class ``u`` where both ``NEE[u]`` and ``NEE[u + 1]`` reach
    ``plateau_crit`` times the mean of the following ``plateau_fwd`` classes
    (windows cut at the last class, R's ``na.rm = TRUE`` on out-of-range
    indices). ``u`` runs up to ``n_bins - max(2, min_n_plateau)``. NaN if no
    class qualifies (``isOmitNoThresholdBins = TRUE``).
    """
    u_max = n_bins - max(2, min_n_plateau)
    u = 1
    while u <= u_max:
        e1 = min(u + plateau_fwd, n_bins)
        m1 = r_mean(nee_avg[u:e1])
        e2 = min(u + plateau_fwd + 1, n_bins)
        m2 = r_mean(nee_avg[u + 1:e2])
        if (nee_avg[u - 1] >= plateau_crit * m1) and (nee_avg[u] >= plateau_crit * m2):
            return ust_avg[u - 1]
        u += 1
    return np.nan


@njit(cache=True)
def estimate_subset(nee, ustar, ta, n_ta, n_us, min_rec_season, min_rec_temp,
                    corr_check, first_check, plateau_fwd, plateau_crit,
                    min_n_plateau):
    """``.estimateUStarSeason`` (DP:451-560) for the records of one season
    (or one pooled year), given in data order.

    Returns a tuple:

    - ``thresholds``: u* threshold per temperature class (NaN where none).
    - ``tbin``, ``ubin``: temperature and u* class per record, -1 for R's NA.
      ``ubin`` is 0 in classes skipped by the correlation check (DP:480, the
      default of ``uStarBinSortedT``).
    - ``bin_ust``, ``bin_nee``, ``bin_n``, ``n_bins``: u* class averages and
      record counts per temperature class (rows), as ``.binUstar`` returns
      them, and the number of u* classes per temperature class.
    - ``status``: ``OK`` or ``ERR_TEMP_BREAK``.
    """
    n = nee.shape[0]
    thresholds = np.full(n_ta, np.nan)
    tbin = np.full(n, -1, dtype=np.int64)
    ubin = np.full(n, -1, dtype=np.int64)
    bin_ust = np.full((n_ta, n_us), np.nan)
    bin_nee = np.full((n_ta, n_us), np.nan)
    bin_n = np.zeros((n_ta, n_us), dtype=np.int64)
    n_bins = np.zeros(n_ta, dtype=np.int64)
    # DP:466-478: too few records, the whole season is NA and unbinned
    if n < min_rec_season or n / n_ta < min_rec_temp:
        return thresholds, tbin, ubin, bin_ust, bin_nee, bin_n, n_bins, OK
    # DP:479-481: R's order() is stable, ties keep the data order
    order_t = np.argsort(ta, kind='mergesort')
    ta_s = ta[order_t]
    ust_s = ustar[order_t]
    nee_s = nee[order_t]
    tid, status = bin_equal_values_balanced(ta_s, n_ta, 1e-8)
    if status != OK:
        return thresholds, tbin, ubin, bin_ust, bin_nee, bin_n, n_bins, status
    ubin_sorted = np.zeros(n, dtype=np.int64)
    for k in range(1, n_ta + 1):
        idx = np.where(tid == k)[0]
        m = idx.shape[0]
        u_k = ust_s[idx]
        t_k = ta_s[idx]
        n_k = nee_s[idx]
        # DP:499-506: T and u* must be (nearly) uncorrelated, strictly < 0.5
        if not abs_cor_below(u_k, t_k, corr_check):
            continue
        # .binUstar (DP:924-966): stable sort by u*, then bin
        order_u = np.argsort(u_k, kind='mergesort')
        u_sorted = u_k[order_u]
        nee_sorted = n_k[order_u]
        u_class = bin_equal_values_min_rec(u_sorted, n_us, 1e-14)
        nb = 0
        for i in range(m):
            if u_class[i] > nb:
                nb = u_class[i]
        n_bins[k - 1] = nb
        # dplyr group_by + mean: records of a class in u*-sorted order
        start = 0
        for c in range(1, nb + 1):
            stop = start
            while stop < m and u_class[stop] == c:
                stop += 1
            bin_ust[k - 1, c - 1] = r_mean(u_sorted[start:stop])
            bin_nee[k - 1, c - 1] = r_mean(nee_sorted[start:stop])
            bin_n[k - 1, c - 1] = stop - start
            start = stop
        for i in range(m):
            ubin_sorted[idx[order_u[i]]] = u_class[i]
        # DP:532-538: first u* class already turbulent -> skip the class
        if bin_ust[k - 1, 0] > first_check:
            continue
        thresholds[k - 1] = fw2(bin_nee[k - 1, :nb], bin_ust[k - 1, :nb], nb,
                                plateau_fwd, plateau_crit, min_n_plateau)
    # DP:551-553: back to data order
    for i in range(n):
        tbin[order_t[i]] = tid[i]
        ubin[order_t[i]] = ubin_sorted[i]
    return thresholds, tbin, ubin, bin_ust, bin_nee, bin_n, n_bins, OK
