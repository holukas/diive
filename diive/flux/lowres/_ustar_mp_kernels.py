"""
USTAR MP KERNELS: COMPILED LOOPS FOR THE ONEFLUX MOVING POINT METHOD
====================================================================

Numba kernels behind ``UstarMovingPointDetection``, transcribed from ONEFlux
``ustar_mp`` (``ustar.c``, ``bootstrapping.c``) in the C evaluation order.

Part of the diive library: https://github.com/holukas/diive

Every sum runs sequentially in the order the C runs it, so the results are
bitwise identical to the compiled C. Two details matter for that:

- Rows are sorted with a port of the Microsoft UCRT ``qsort``, the C runtime of
  the reference build. ``qsort`` is not stable, so it decides the order of rows
  with equal TA or USTAR, and with it the order of the NEE sums. A glibc build
  of ONEFlux orders such ties differently and can differ in the last bits.
- ``error_model='numpy'``: an empty or inverted class window divides 0.0 by an
  integer count of 0 or less, which gives NaN or -0.0 in C. Numba's default
  error model would raise ZeroDivisionError instead.

The kernels are compiled with ``cache=True`` when numba finds a writable cache
location: the ``__pycache__`` next to this file, ``NUMBA_CACHE_DIR``, or the
user cache directory (the only option in a frozen PyInstaller app, keyed to the
executable). When none is writable, numba raises at decoration time. The kernels
then compile without a cache (about 8 s on first use in each process) instead of
failing the import.
"""

import numba
import numpy as np

# Constants from ONEFlux ustar_mp/src/types.h and common/common.h
INVALID_VALUE = -9999.0  # common.h:159
USTAR_THRESHOLD_NOT_FOUND = 10.0  # types.h:36
TA_CLASS_MIN_SAMPLE = 100  # types.h:31
MIN_VALUE_PERIOD = 3000  # types.h:29
MIN_VALUE_SEASON = 160  # types.h:30
CORRELATION_CHECK = 0.5  # types.h:32
FIRST_USTAR_MEAN_CHECK = 0.2  # types.h:34

# Column order of the row arrays (ROW.value[] in types.h, minus TIMESTAMP and SWIN)
_NEE = 0
_TA = 1
_USTAR = 2


def _jit(func):
    """Compile with a disk cache when numba finds a writable location, else without."""
    try:
        return numba.njit(cache=True, error_model='numpy')(func)
    except RuntimeError:
        # numba.core.caching raises "cannot cache function ...: no locator available"
        return numba.njit(cache=False, error_model='numpy')(func)


@_jit
def is_invalid(v):
    """``IS_INVALID_VALUE`` (common.h:40): ``-9999 == (int)v``.

    The cast truncates toward zero, so every value in (-10000, -9999] counts as
    missing, not only -9999 itself.
    """
    return -10000.0 < v <= -9999.0


# ---------------------------------------------------------------- qsort (UCRT)

@_jit
def _cmp(rows, a, b, col):
    """``compare_ta`` / ``compare_ustar`` (ustar.c:47-66)."""
    x = rows[a, col]
    y = rows[b, col]
    if x < y:
        return -1
    elif x > y:
        return 1
    return 0


@_jit
def _swap(rows, a, b):
    if a != b:
        for k in range(rows.shape[1]):
            t = rows[a, k]
            rows[a, k] = rows[b, k]
            rows[b, k] = t


@_jit
def _shortsort(rows, lo, hi, col):
    """UCRT ``shortsort``: selection sort that moves the last maximum to the end."""
    while hi > lo:
        mx = lo
        p = lo + 1
        while p <= hi:
            if _cmp(rows, p, mx, col) > 0:
                mx = p
            p += 1
        _swap(rows, mx, hi)
        hi -= 1


@_jit
def qsort_rows(rows, start, num, col):
    """Sort ``rows[start:start+num]`` by column ``col`` like the UCRT ``qsort``.

    Port of Microsoft's ``qsort.cpp`` (median-of-three quicksort, CUTOFF 8,
    selection sort below it, explicit stack, smaller part first). Pointer
    arithmetic is replaced by row positions.
    """
    if num < 2:
        return
    lostk = np.empty(64, np.int64)
    histk = np.empty(64, np.int64)
    stkptr = 0
    lo = start
    hi = start + num - 1
    while True:
        size = hi - lo + 1
        if size <= 8:
            _shortsort(rows, lo, hi, col)
        else:
            mid = lo + size // 2
            if _cmp(rows, lo, mid, col) > 0:
                _swap(rows, lo, mid)
            if _cmp(rows, lo, hi, col) > 0:
                _swap(rows, lo, hi)
            if _cmp(rows, mid, hi, col) > 0:
                _swap(rows, mid, hi)
            loguy = lo
            higuy = hi
            while True:
                if mid > loguy:
                    loguy += 1
                    while loguy < mid and _cmp(rows, loguy, mid, col) <= 0:
                        loguy += 1
                if mid <= loguy:
                    loguy += 1
                    while loguy <= hi and _cmp(rows, loguy, mid, col) <= 0:
                        loguy += 1
                higuy -= 1
                while higuy > mid and _cmp(rows, higuy, mid, col) > 0:
                    higuy -= 1
                if higuy < loguy:
                    break
                _swap(rows, loguy, higuy)
                if mid == higuy:
                    mid = loguy
            higuy += 1
            if mid < higuy:
                higuy -= 1
                while higuy > mid and _cmp(rows, higuy, mid, col) == 0:
                    higuy -= 1
            if mid >= higuy:
                higuy -= 1
                while higuy > lo and _cmp(rows, higuy, mid, col) == 0:
                    higuy -= 1
            if higuy - lo >= hi - loguy:
                if lo < higuy:
                    lostk[stkptr] = lo
                    histk[stkptr] = higuy
                    stkptr += 1
                if loguy < hi:
                    lo = loguy
                    continue
            else:
                if loguy < hi:
                    lostk[stkptr] = loguy
                    histk[stkptr] = hi
                    stkptr += 1
                if lo < higuy:
                    hi = higuy
                    continue
        stkptr -= 1
        if stkptr >= 0:
            lo = lostk[stkptr]
            hi = histk[stkptr]
        else:
            return


# ------------------------------------------------------------ ustar.c routines

@_jit
def _meanws(array, array_count, index, element_count):
    """``meanws`` (ustar.c:70-101). An empty window divides by 0 and gives NaN -> invalid."""
    if index > array_count:
        return 0.0
    mean = 0.0
    count = 0
    i = index
    while i < index + element_count:
        if i >= array_count:
            break
        if is_invalid(array[i]):
            return INVALID_VALUE
        mean += array[i]
        count += 1
        i += 1
    mean /= count
    if mean != mean:
        return INVALID_VALUE
    return mean


@_jit
def _correlation(rows, start, end):
    """``correlation`` (ustar.c:104-155): Pearson r of TA and USTAR, sequential sums."""
    xmean = 0.0
    ymean = 0.0
    for i in range(start, end + 1):
        if is_invalid(rows[i, _TA]):
            return INVALID_VALUE
        if is_invalid(rows[i, _USTAR]):
            return INVALID_VALUE
        xmean += rows[i, _TA]
        ymean += rows[i, _USTAR]
    xmean /= ((end - start) + 1)
    ymean /= ((end - start) + 1)
    s = 0.0
    xv = 0.0
    yv = 0.0
    for i in range(start, end + 1):
        t1 = rows[i, _TA] - xmean
        t2 = rows[i, _USTAR] - ymean
        xv = xv + t1 * t1
        yv = yv + t2 * t2
        s = s + t1 * t2
    corr = s / (np.sqrt(xv) * np.sqrt(yv))
    if corr != corr:
        corr = INVALID_VALUE
    return corr


@_jit
def _forward_mode(ustar_mean, fx_mean, ustar_classes_count, window_size, threshold_check, n):
    """``forward_mode`` (ustar.c:266-350) with the percentile check off (ONEFlux default)."""
    if n < 1 or ustar_classes_count - n <= 0:
        return INVALID_VALUE
    means = np.empty(n)
    for i in range(ustar_classes_count - n + 1):
        bad = False
        for y in range(n):
            means[y] = _meanws(fx_mean, ustar_classes_count, i + 1 + y, window_size)
            if is_invalid(means[y]):
                bad = True
                break
        if bad:
            continue
        z = 0
        for y in range(n):
            if fx_mean[i + y] >= (means[y] * threshold_check):
                z += 1
        if z == n:
            return ustar_mean[i]
    return USTAR_THRESHOLD_NOT_FOUND


@_jit
def median_ustar_threshold(container, ta_classes_count, season):
    """``median_ustar_threshold`` (ustar.c:214-263): median of the valid TA-class values."""
    vals = np.empty(ta_classes_count)
    count = 0
    for i in range(ta_classes_count):
        v = container[season * ta_classes_count + i]
        if not is_invalid(v) and v != USTAR_THRESHOLD_NOT_FOUND:
            vals[count] = v
            count += 1
    if count == 0:
        return INVALID_VALUE
    if count == 1:
        return vals[0]
    s = np.sort(vals[:count])
    if count & 1:
        return s[((count + 1) // 2) - 1]
    return (s[(count // 2) - 1] + s[count // 2]) / 2.0


@_jit
def ustar_threshold(rows, rows_count, days, samples_per_season, seasons_group_count,
                    ta_classes_count, ustar_classes_count, fw_n, window_size,
                    threshold_check, container):
    """``ustar_threshold`` (ustar.c:428-932), forward mode ``fw_n`` only.

    ``rows`` holds NEE, TA, USTAR of the valid night records, season after
    season, and is sorted in place exactly as the C sorts it. ``samples_per_season``
    is updated in place when the seasons collapse into one. Returns
    ``(ok, seasons_group_count)``: ``ok`` is 0 where the C returns 0 (not enough
    values), and the season count is 1 after a collapse, as the C global becomes.
    ``container`` receives the threshold per season and TA class (-9999 = no
    value, 10.0 = not found).
    """
    tacl = ta_classes_count
    ucl = ustar_classes_count
    sgc = seasons_group_count

    # ustar.c:500-516: method choice
    if rows_count < tacl * ucl:
        return 0, sgc
    elif rows_count + days >= MIN_VALUE_PERIOD:
        seasons_n = sgc
    elif rows_count > MIN_VALUE_SEASON:
        seasons_n = 1
        for i in range(1, sgc):
            samples_per_season[0] += samples_per_season[i]
        sgc = 1
    else:
        return 0, sgc

    # ustar.c:519-535: every season too small -> one big season
    if seasons_n > 1:
        small = 0
        for s in range(seasons_n):
            if samples_per_season[s] < TA_CLASS_MIN_SAMPLE * tacl:
                small += 1
        if small == seasons_n:
            seasons_n = 1
            for i in range(1, sgc):
                samples_per_season[0] += samples_per_season[i]
            sgc = 1

    for i in range(sgc):
        for y in range(tacl):
            container[i * tacl + y] = INVALID_VALUE

    ta_window = np.empty((tacl, 2), np.int64)
    ustar_window = np.empty((ucl, 2), np.int64)
    ustar_mean = np.empty(ucl)
    fx_mean = np.empty(ucl)

    for s in range(seasons_n):
        if samples_per_season[s] < TA_CLASS_MIN_SAMPLE * tacl:
            continue
        season_start_index = 0
        for i in range(s):
            season_start_index += samples_per_season[i]
        n_s = samples_per_season[s]
        limit = season_start_index + n_s

        qsort_rows(rows, season_start_index, n_s, _TA)

        ta_samples_count = n_s // tacl
        if ta_samples_count <= 0:
            return 0, sgc

        ta_class_end = season_start_index
        for i in range(tacl):
            ta_window[i, 0] = -1
            ta_window[i, 1] = -1

        # ustar.c:602-637. No "start == end" branch here, unlike the USTAR loop.
        # The C for-loop counter survives the loop: after a break, ta_window[i]
        # is the class at the break index, not the last class.
        i = 0
        while i < tacl - 1:
            ta_class_start = ta_class_end
            ta_class_end = season_start_index + (ta_samples_count * (i + 1) - 1)
            if ta_class_start >= limit:
                break
            if ta_class_end >= limit:
                ta_class_end = limit - 1
            value = rows[ta_class_end, _TA]
            ta_class_end += 1
            while ta_class_end < limit:
                if value != rows[ta_class_end, _TA]:
                    break
                ta_class_end += 1
            ta_window[i, 0] = ta_class_start
            ta_window[i, 1] = ta_class_end - 1
            i += 1
        if ta_class_end < limit:
            ta_window[i, 0] = ta_class_end
            ta_window[i, 1] = limit - 1

        for ta_class in range(tacl):
            st = ta_window[ta_class, 0]
            en = ta_window[ta_class, 1]
            if st == -1:
                continue
            if en - st + 1 < TA_CLASS_MIN_SAMPLE:
                continue
            corr = _correlation(rows, st, en)
            if abs(corr) > CORRELATION_CHECK:
                continue

            total = (en - st) + 1
            qsort_rows(rows, st, total, _USTAR)

            ustar_samples_count = total // ucl
            if ustar_samples_count <= 0:
                return 0, sgc

            ustar_class_end = 0
            for i in range(ucl):
                ustar_window[i, 0] = -1
                ustar_window[i, 1] = -1

            # ustar.c:693-732
            i = 0
            while i < ucl - 1:
                ustar_class_start = ustar_class_end
                ustar_class_end = ustar_samples_count * (i + 1) - 1
                if ustar_class_start >= total:
                    break
                if ustar_class_end >= total:
                    ustar_class_end = total - 1
                value = rows[ustar_class_end + st, _USTAR]
                ustar_class_end += 1
                while ustar_class_end < total:
                    if value != rows[ustar_class_end + st, _USTAR]:
                        if ustar_class_start == ustar_class_end:
                            # C `continue` re-runs `while ( ++ustar_class_end < ... )`
                            value = rows[ustar_class_end + st, _USTAR]
                            ustar_class_end += 1
                            continue
                        break
                    ustar_class_end += 1
                ustar_window[i, 0] = ustar_class_start + st
                ustar_window[i, 1] = ustar_class_end - 1 + st
                i += 1
            if ustar_class_end < total:
                ustar_window[i, 0] = ustar_class_end + st
                ustar_window[i, 1] = total - 1 + st

            # ustar.c:735-754. An inverted window divides 0.0 by a count <= 0.
            for k in range(ucl):
                ustar_mean[k] = 0.0
                fx_mean[k] = 0.0
                if ustar_window[k, 0] == -1:
                    continue
                for y in range(ustar_window[k, 0], ustar_window[k, 1] + 1):
                    ustar_mean[k] += rows[y, _USTAR]
                    fx_mean[k] += rows[y, _NEE]
                cnt = (ustar_window[k, 1] - ustar_window[k, 0]) + 1
                ustar_mean[k] /= cnt
                fx_mean[k] /= cnt

            if ustar_mean[0] > FIRST_USTAR_MEAN_CHECK:
                continue

            threshold = _forward_mode(ustar_mean, fx_mean, ucl, window_size,
                                      threshold_check, fw_n)
            container[s * tacl + ta_class] = threshold
            if seasons_n == 1:
                for i in range(sgc):
                    container[i * tacl + ta_class] = threshold
    return 1, sgc


# --------------------------------------------------------------- bootstrapping

@_jit
def bootstrap(nee, ta, ustar, is_day, all_valid, boot_season, row_of_key, key_of_row,
              indices, seasons_group_count, n_seasons, ta_classes_count,
              ustar_classes_count, fw_n, window_size, threshold_check):
    """Bootstrap loop of ``bootstrapping`` (bootstrapping.c:146-239).

    ``indices[b]`` are the drawn row positions of draw ``b``. A drawn row
    counts as a day when it is not night (missing SW_IN included), and is kept
    when it is night with NEE, TA and USTAR present. Kept rows are put in the
    order given by ``key_of_row`` (the row position for a calendar-year grid,
    as the C ``qsort`` by index does) and cut into seasons by the counts from
    ``boot_season``. ``seasons_group_count`` carries over between draws: once a
    draw collapses to one season, all later draws stay collapsed, as the C
    global does.

    Returns the season medians (n_iter x n_seasons, NaN = none), the annual
    value (maximum of the season medians, NaN = none), a flag per draw where the
    C ``ustar_threshold`` returned 0, and the final season count.
    """
    n_iter = indices.shape[0]
    n_draw = indices.shape[1]
    n_rows = nee.shape[0]
    tacl = ta_classes_count
    season_med = np.full((n_iter, n_seasons), np.nan)
    annual = np.full(n_iter, np.nan)
    failed = np.zeros(n_iter, np.bool_)
    samples = np.zeros(n_seasons, np.int64)
    hits = np.zeros(n_rows, np.int64)
    rows = np.empty((n_draw, 3))
    container = np.empty(n_seasons * tacl)
    sgc = seasons_group_count

    for b in range(n_iter):
        for i in range(sgc):
            samples[i] = 0
        hits[:] = 0
        days = 0
        cnt = 0
        for j in range(n_draw):
            r = indices[b, j]
            if is_day[r]:
                days += 1
            elif all_valid[r]:
                z = boot_season[r] if sgc > 1 else 0
                if z < 0:
                    continue
                hits[key_of_row[r]] += 1
                samples[z] += 1
                cnt += 1
        if cnt == 0:
            continue

        k = 0
        for key in range(n_rows):
            h = hits[key]
            if h:
                r = row_of_key[key]
                for _ in range(h):
                    rows[k, _NEE] = nee[r]
                    rows[k, _TA] = ta[r]
                    rows[k, _USTAR] = ustar[r]
                    k += 1

        ok, sgc = ustar_threshold(rows[:cnt], cnt, days, samples, sgc, tacl,
                                  ustar_classes_count, fw_n, window_size,
                                  threshold_check, container)
        if ok == 0:
            failed[b] = True
            continue

        taken = INVALID_VALUE
        for i in range(sgc):
            med = median_ustar_threshold(container, tacl, i)
            if not is_invalid(med):
                season_med[b, i] = med
            if i == 0:
                taken = med
            elif med > taken:
                taken = med
        if sgc == 1:
            for i in range(1, n_seasons):
                season_med[b, i] = season_med[b, 0]
        if not is_invalid(taken):
            annual[b] = taken
    return season_med, annual, failed, sgc
