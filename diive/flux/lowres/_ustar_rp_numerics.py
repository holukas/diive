"""
USTAR RP NUMERICS: R'S LONG-DOUBLE ARITHMETIC FOR THE REDDYPROC U* PORT
=======================================================================

numba kernels that reproduce the arithmetic of R's ``mean()``, ``median()``,
``cor()`` and ``quantile(type = 7)`` as R 4.x computes them on x86-64.

R accumulates ``mean()`` and the means inside ``cor()`` in ``long double``
(``LDOUBLE`` in R's ``summary.c`` and ``cov.c``). On x86-64 builds with gcc,
R on Windows included, that is the 80-bit x87 format with a 64-bit
significand. A plain float64 mean, and also the exactly rounded mean, differ
from R's result in the last bit for a few in a thousand bin averages. The
u* threshold is one of those averages, so the port needs R's bits.

numpy has no 80-bit type on Windows. The kernels hold an 80-bit value
exactly as an unevaluated sum ``hi + lo`` of two doubles (``hi`` is the value
rounded to double, ``lo`` the exact remainder) and round every operation
to 64 significant bits, ties to even, as the x87 unit does with precision
control set to extended (R's setting). Each operation first forms the exact
result as a short expansion of doubles with error-free transformations
(``two_sum``, ``two_prod``), then rounds that to the 64-bit grid.

Part of the diive library: https://github.com/holukas/diive
"""
import math

import numpy as np
from numba import njit

_SPLITTER = 134217729.0  # 2**27 + 1, Dekker's split for two_prod


@njit(cache=True, inline='always')
def _two_sum(a, b):
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err


@njit(cache=True, inline='always')
def _split(a):
    c = _SPLITTER * a
    abig = c - a
    ahi = c - abig
    return ahi, a - ahi


@njit(cache=True, inline='always')
def _two_prod(a, b):
    p = a * b
    ahi, alo = _split(a)
    bhi, blo = _split(b)
    err = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo
    return p, err


@njit(cache=True)
def _normalize4(a, b, c, d):
    """Exact value ``a + b + c + d`` as ``A + B + C + D`` with
    ``A + B == A``, ``B + C == B`` and ``C + D == C`` in double arithmetic,
    so ``|B| <= ulp(A)/2`` and so on (zero components last)."""
    for _ in range(64):
        A, x = _two_sum(a, b)
        B, y = _two_sum(x, c)
        C, D = _two_sum(y, d)
        if A + B == A and B + C == B and C + D == C:
            return A, B, C, D
        a, b, c, d = A, B, C, D
    raise RuntimeError('long-double emulation: expansion did not settle')


@njit(cache=True)
def _round64(a, b, c, d):
    """Round the exact sum ``a + b + c + d`` to a 64-bit significand, ties to
    even (one x87 operation). Returns ``(hi, lo)``."""
    A, B, C, _ = _normalize4(a, b, c, d)
    if A == 0.0:
        return 0.0, 0.0
    m, e = math.frexp(A)
    ex = e - 1  # floor(log2(|A|))
    if abs(m) == 0.5:
        # A is a power of two. A tail of the opposite sign puts the exact
        # value into the binade below, where the 64-bit grid is finer.
        tail = B if B != 0.0 else C
        if tail != 0.0 and ((tail > 0.0) != (A > 0.0)):
            ex -= 1
    q = math.ldexp(1.0, ex - 63)
    # A is a multiple of 2 * q, so only B (and C at an exact tie) decide.
    y = B / q
    f = math.floor(y)
    frac = y - f
    if frac > 0.5:
        k = f + 1.0
    elif frac < 0.5:
        k = f
    elif C > 0.0:
        k = f + 1.0
    elif C < 0.0:
        k = f
    elif f % 2.0 == 0.0:
        k = f
    else:
        k = f + 1.0
    return _two_sum(A, k * q)


@njit(cache=True)
def ld_mul(ah, al, bh, bl):
    """Product of two long doubles. The low parts carry at most 11 bits, so
    ``al * bl`` is exact in double and the other three products are split
    exactly with ``two_prod``."""
    t = np.empty(7)
    t[0], t[1] = _two_prod(ah, bh)
    t[2], t[3] = _two_prod(ah, bl)
    t[4], t[5] = _two_prod(al, bh)
    t[6] = al * bl
    for _ in range(64):
        done = True
        for i in range(6):
            s, e = _two_sum(t[i], t[i + 1])
            if s != t[i] or e != t[i + 1]:
                done = False
            t[i] = s
            t[i + 1] = e
        if done:
            return _round64(t[0], t[1], t[2], t[3])
    raise RuntimeError('long-double emulation: expansion did not settle')


@njit(cache=True, inline='always')
def ld_add_d(hi, lo, x):
    """``s + x`` with ``s = hi + lo`` long double and ``x`` double."""
    s, e = _two_sum(hi, x)
    return _round64(s, e, lo, 0.0)


@njit(cache=True, inline='always')
def ld_add_ld(h1, l1, h2, l2):
    """Sum of two long doubles."""
    s, e = _two_sum(h1, h2)
    t, f = _two_sum(l1, l2)
    return _round64(s, e, t, f)


@njit(cache=True)
def ld_div_n(hi, lo, n):
    """``s / n`` for long double ``s`` and a positive integer ``n``.

    The quotient is approximated to about 2**-104 relative in double-double.
    For ``n`` below 2**40 a quotient that is not exactly on a 64-bit rounding
    midpoint lies at least 2**-(64 + log2(n) + 1) relative away from one, and
    an exact midpoint is represented exactly, so the rounding is correct.
    """
    if hi == 0.0 and lo == 0.0:
        return 0.0, 0.0
    nd = float(n)
    q1 = hi / nd
    p, pe = _two_prod(q1, nd)
    r1, r2, _, _ = _normalize4(hi, -p, lo, -pe)
    q2 = (r1 + r2) / nd
    return _round64(q1, q2, 0.0, 0.0)


@njit(cache=True)
def ld_sqrt(hi, lo):
    """``sqrtl`` of a non-negative long double: the root to about 2**-104 in
    double-double (one Newton step), rounded to 64 bits."""
    if hi <= 0.0:
        return 0.0, 0.0
    r = math.sqrt(hi)
    p, pe = _two_prod(r, r)
    d1, d2, _, _ = _normalize4(hi, -p, lo, -pe)
    c = (d1 + d2) / (2.0 * r)
    return _round64(r, c, 0.0, 0.0)


@njit(cache=True)
def r_mean(x):
    """R ``mean()`` of a non-empty array of finite doubles.

    Fast path: the mean in double-double, rounded once to double. R's
    long-double result lies within about ``2**-62 * (|mean| + max|x - mean|)``
    of the exact mean, so where the exact mean is farther than that from a
    rounding midpoint between two doubles, both round to the same double.
    Otherwise (a few percent of calls) :func:`r_mean_ld` emulates R's
    arithmetic step by step.
    """
    n = x.shape[0]
    sh = 0.0
    sl = 0.0
    amax = 0.0
    for i in range(n):
        v = x[i]
        sh, e = _two_sum(sh, v)
        sl += e
        a = abs(v)
        if a > amax:
            amax = a
    nd = float(n)
    q1 = sh / nd
    p, pe = _two_prod(q1, nd)
    q2 = (((sh - p) - pe) + sl) / nd
    h, lo = _two_sum(q1, q2)
    if math.isfinite(h) and h != 0.0:
        m, e = math.frexp(h)
        if abs(m) != 0.5:  # binade edges: use the slow path
            half_ulp = math.ldexp(1.0, e - 54)
            margin = math.ldexp(abs(h) + 2.0 * amax, -58)
            if half_ulp - abs(lo) > margin:
                return h
    return r_mean_ld(x)


@njit(cache=True)
def r_mean_ld(x):
    """R ``mean()`` of a non-empty array of finite doubles, emulated step by
    step: long-double sum, division, a long-double correction pass, then one
    rounding to double (``summary.c``, ``do_summary`` case REALSXP)."""
    n = x.shape[0]
    hi = 0.0
    lo = 0.0
    for i in range(n):
        hi, lo = ld_add_d(hi, lo, x[i])
    hi, lo = ld_div_n(hi, lo, n)
    if not math.isfinite(hi):
        return hi
    th = 0.0
    tl = 0.0
    for i in range(n):
        dh, dl = _round64(x[i], -hi, -lo, 0.0)
        th, tl = ld_add_ld(th, tl, dh, dl)
    qh, ql = ld_div_n(th, tl, n)
    hi, lo = ld_add_ld(hi, lo, qh, ql)
    return hi


@njit(cache=True)
def r_mean2(a, b):
    """R ``mean(c(a, b))``, as ``median()`` takes it for an even count."""
    tmp = np.empty(2)
    tmp[0] = a
    tmp[1] = b
    return r_mean(tmp)


@njit(cache=True)
def r_median_finite(x):
    """R ``median(x, na.rm = TRUE)``; NaN entries are dropped, NaN if none
    is left. An even count averages the two middle values with R's
    ``mean()``."""
    v = x[~np.isnan(x)]
    n = v.shape[0]
    if n == 0:
        return np.nan
    v = np.sort(v)
    half = (n + 1) // 2
    if n % 2 == 1:
        return v[half - 1]
    return r_mean2(v[half - 1], v[half])


@njit(cache=True)
def abs_cor_below(x, y, limit):
    """R's ``is.finite(Cor1) && Cor1 < limit`` with ``Cor1 = abs(cor(x, y))``
    (DP:499-506).

    Only the decision matters. A float64 two-pass correlation is accurate to
    about 1e-13 here, so it decides unless it lies within 1e-9 of the limit.
    Then, and for constant input (R: NA), :func:`r_cor` decides.
    """
    n = x.shape[0]
    if n <= 1:
        return False
    xmin = x.min()
    ymin = y.min()
    if xmin == x.max() or ymin == y.max():
        return False
    xm = x.sum() / n
    ym = y.sum() / n
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    for k in range(n):
        dx = x[k] - xm
        dy = y[k] - ym
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    c = abs(sxy / math.sqrt(sxx * syy))
    if abs(c - limit) > 1e-9:
        return c < limit
    r = r_cor(x, y)
    return math.isfinite(r) and abs(r) < limit


@njit(cache=True)
def r_cor(x, y):
    """R ``cor(x, y)`` (Pearson, no missing values), ``cov.c``.

    The means are long-double two-pass means rounded to double (macro
    ``MEAN_``). Deviations from them, their products and the sums are
    long-double operations. The covariance is divided by ``n - 1`` and
    rounded to double. The standard deviations are ``sqrtl`` of the
    variances, rounded to double, and the correlation is clamped to
    [-1, 1]. Returns NaN for fewer than two records or a zero standard
    deviation, as R does (with a warning). Bitwise equal to R 4.5.3 on 2000
    random vector pairs.
    """
    n = x.shape[0]
    if n <= 1:
        return np.nan
    xm = r_mean(x)
    ym = r_mean(y)
    sh = 0.0
    sl = 0.0
    xh = 0.0
    xl = 0.0
    yh = 0.0
    yl = 0.0
    for k in range(n):
        # xxm and yym are long double in cov.c: differences and products
        # are long-double operations
        dxh, dxl = _round64(x[k], -xm, 0.0, 0.0)
        dyh, dyl = _round64(y[k], -ym, 0.0, 0.0)
        ph, pl = ld_mul(dxh, dxl, dyh, dyl)
        sh, sl = ld_add_ld(sh, sl, ph, pl)
        ph, pl = ld_mul(dxh, dxl, dxh, dxl)
        xh, xl = ld_add_ld(xh, xl, ph, pl)
        ph, pl = ld_mul(dyh, dyl, dyh, dyl)
        yh, yl = ld_add_ld(yh, yl, ph, pl)
    cov = ld_div_n(sh, sl, n - 1)[0]
    # xm[i] = (double) sqrtl(sum / n1): the root in long double, then rounded
    vh, vl = ld_div_n(xh, xl, n - 1)
    xsd = ld_sqrt(vh, vl)[0]
    vh, vl = ld_div_n(yh, yl, n - 1)
    ysd = ld_sqrt(vh, vl)[0]
    if xsd == 0.0 or ysd == 0.0:
        return np.nan
    r = cov / (xsd * ysd)
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    return r


def r_quantile7(x, probs):
    """R ``quantile(x, probs, na.rm = TRUE)`` (type 7, ``fuzz = 0``).

    R interpolates as ``(1 - h) * lo + h * hi`` and skips the
    interpolation where the two order statistics are equal. numpy's linear
    method uses another formula and can differ in the last bit.
    """
    v = np.sort(np.asarray(x, dtype=float))
    v = v[~np.isnan(v)]
    n = v.size
    out = np.full(len(probs), np.nan)
    if n == 0:
        return out
    for j, p in enumerate(probs):
        index = 1.0 + max(n - 1, 0) * p
        lo = math.floor(index)
        hi = math.ceil(index)
        qs = float(v[lo - 1])
        if index > lo and v[hi - 1] != qs:
            h = index - lo
            qs = (1.0 - h) * qs + h * float(v[hi - 1])
        out[j] = qs
    return out
