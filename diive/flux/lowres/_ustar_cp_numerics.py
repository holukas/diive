"""
USTAR CP NUMERICS: MATLAB HELPERS FOR THE CHANGE-POINT PORT
===========================================================

MATLAB numerics used by the ONEFlux ``ustar_cp`` port (``ustar_cp_detection.py``).

The compiled ONEFlux ``ustar_cp`` runs on MATLAB Runtime R2018a (MCR v9.4, see
``oneflux/metadata/bif_var_info.py``). The helpers reproduce the R2018a behaviour
of the functions the change-point code calls: the colon operator, ``round``,
``prctile``, ``interp1(..., 'pchip')`` and the default ``rand``/``randi`` stream.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
from scipy import special

_EPS = np.finfo(np.float64).eps


def matlab_round(x):
    """MATLAB ``round``: halves away from zero (numpy rounds halves to even)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def matlab_colon(a: float, d: float, b: float) -> np.ndarray:
    """MATLAB ``a:d:b`` for ``d > 0``.

    MATLAB builds the vector from both ends, so it is symmetric and its last
    element is exactly ``b`` when ``(b - a) / d`` is an integer within a few ulps
    (Moler, "The colon operator", ``colonop.m``). The first ``floor(n/2) + 1``
    elements are ``a + k*d``, the others ``b - (n - k)*d``. The percentile edges
    ``0:(100/n):100`` of ``prctile`` calls depend on these last bits.
    """
    if not np.isfinite(d):
        # 0:Inf:100 is the scalar 0 in MATLAB (fcBin with nBins = 0)
        return np.array([a], dtype=np.float64)
    tol = 2.0 * _EPS * max(abs(a), abs(b))
    q = (b - a) / d
    n = int(np.floor(q))
    if (n + 1) * d + a - b <= tol:
        n += 1
    if n < 0:
        return np.empty(0, dtype=np.float64)
    c = a + n * d
    if abs(c - b) <= tol:
        c = b
    v = np.empty(n + 1, dtype=np.float64)
    k = np.arange(n // 2 + 1)
    v[k] = a + k * d
    k = np.arange((n - 1) // 2 + 1)
    v[n - k] = c - k * d
    return v


def matlab_prctile(x, p) -> np.ndarray:
    """MATLAB R2018a ``prctile`` (Statistics Toolbox) of a vector.

    The sorted values sit at percents ``100*(i - 0.5)/n``. Below the first and
    above the last the end value is returned, in between ``prctile`` interpolates
    linearly as ``(0.5 + r)*x(k+1) + (0.5 - r)*x(k)``. Values hit exactly and
    equal neighbours are copied, not interpolated. This is numpy's 'hazen'
    definition, but numpy's formula differs in the last bits, and those bits
    decide bin membership when an edge falls on a value (the inclusive bins of
    ``fcBin`` and the temperature strata). NaNs are ignored.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(x[~np.isnan(x)])
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    n = x.size
    if n == 0:
        return np.full(p.shape, np.nan)
    r = (p / 100.0) * n
    k = np.floor(r + 0.5)
    kp1 = k + 1.0
    r = r - k
    k[(k < 1) | np.isnan(k)] = 1.0
    kp1 = np.minimum(kp1, n)
    ki = k.astype(np.int64) - 1
    kp1i = kp1.astype(np.int64) - 1
    y = (0.5 + r) * x[kp1i] + (0.5 - r) * x[ki]
    exact = r == -0.5
    y[exact] = x[ki[exact]]
    same = x[ki] == x[kp1i]
    y[same] = x[ki[same]]
    return y


def sequential_sum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Sum along ``axis`` strictly in element order, as MATLAB ``sum`` does for short vectors.

    numpy's ``sum`` adds in pairs (8 partial sums for 8 to 128 values), which
    changes the last bits. These bits decide the 8th digit that ONEFlux writes
    when the exact mean is a decimal tie: two of 1000 reference values (CH-LAE
    L1 2017 draw 6 and 2020 draw 51) agree only when both the u* bin means and
    the annual means are summed in order. ``cumsum`` adds sequentially.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[axis] == 0:
        return np.sum(x, axis=axis)
    return np.take(np.cumsum(x, axis=axis), -1, axis=axis)


def _pchip_end(h1, h2, del1, del2):
    """End slope of MATLAB ``pchip`` (``pchipend`` in ``pchip.m``)."""
    d = ((2.0 * h1 + h2) * del1 - h1 * del2) / (h1 + h2)
    if np.sign(d) != np.sign(del1):
        d = 0.0
    elif np.sign(del1) != np.sign(del2) and abs(d) > abs(3.0 * del1):
        d = 3.0 * del1
    return d


def matlab_pchip(x, y, xi: float) -> float:
    """MATLAB ``interp1(x, y, xi, 'pchip')`` for one point (``pchip.m`` + ``ppval``).

    Interior slopes are the Fritsch-Butland weighted harmonic mean in the scaled
    form of ``pchipslopes``. End slopes follow ``pchipend``. Outside ``x`` the end
    cubic is extrapolated (the default of ``interp1`` for 'pchip'). scipy's
    ``PchipInterpolator`` uses the same formulas in a different operation order.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    h = np.diff(x)
    dl = np.diff(y) / h
    d = np.zeros(n)
    k = np.nonzero(np.sign(dl[:-1]) * np.sign(dl[1:]) > 0)[0]
    hs = h[k] + h[k + 1]
    w1 = (h[k] + hs) / (3.0 * hs)
    w2 = (hs + h[k + 1]) / (3.0 * hs)
    dmax = np.maximum(np.abs(dl[k]), np.abs(dl[k + 1]))
    dmin = np.minimum(np.abs(dl[k]), np.abs(dl[k + 1]))
    d[k + 1] = dmin / (w1 * (dl[k] / dmax) + w2 * (dl[k + 1] / dmax))
    d[0] = _pchip_end(h[0], h[1], dl[0], dl[1])
    d[n - 1] = _pchip_end(h[n - 2], h[n - 3], dl[n - 2], dl[n - 3])
    # pwch.m coefficients, ppval.m nested evaluation
    j = int(np.searchsorted(x, xi, side='right')) - 1
    j = min(max(j, 0), n - 2)
    dzzdx = (dl[j] - d[j]) / h[j]
    dzdxdx = (d[j + 1] - dl[j]) / h[j]
    c3 = (dzdxdx - dzzdx) / h[j]
    c2 = 2.0 * dzzdx - dzdxdx
    s = xi - x[j]
    v = c3
    v = s * v + c2
    v = s * v + d[j]
    v = s * v + y[j]
    return float(v)


# cpdFmax2pCp2.m:21-37, the 2-parameter Fmax table (Barr, following Wang 2003)
_P_TABLE_2 = np.array([0.80, 0.90, 0.95, 0.99])
_N_TABLE_2 = np.array([10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000], dtype=np.float64)
_FMAX_TABLE_2 = np.array([
    [3.9293, 6.2992, 9.1471, 18.2659],
    [3.7734, 5.6988, 7.8770, 13.8100],
    [3.7516, 5.5172, 7.4426, 12.6481],
    [3.7538, 5.3224, 7.0306, 11.4461],
    [3.7941, 5.3030, 6.8758, 10.6635],
    [3.8548, 5.3480, 6.8883, 10.5026],
    [3.9798, 5.4465, 6.9184, 10.4527],
    [4.0732, 5.5235, 6.9811, 10.3859],
    [4.1467, 5.6136, 7.0624, 10.5596],
    [4.2770, 5.7391, 7.2005, 10.6871],
    [4.4169, 5.8733, 7.3421, 10.6751],
    [4.5556, 6.0591, 7.5627, 11.0072],
    [4.7356, 6.2738, 7.7834, 11.2319],
])
# 1 - pTable in double precision: 0.050000000000000044 at the 95 % column, so an
# Fmax exactly on the 95 % critical value gives p > 0.05 (not significant).
_ONE_MINUS_P_2 = 1.0 - _P_TABLE_2

_FMAX_CRIT_CACHE: dict = {}


def fmax_critical_cp2(n: int) -> np.ndarray:
    """Critical Fmax values for n points (cpdFmax2pCp2.m:39, pchip over nTable)."""
    crit = _FMAX_CRIT_CACHE.get(n)
    if crit is None:
        crit = np.array([matlab_pchip(_N_TABLE_2, _FMAX_TABLE_2[:, ip], float(n))
                         for ip in range(_P_TABLE_2.size)])
        _FMAX_CRIT_CACHE[n] = crit
    return crit


def fmax2p_cp2(fmax: float, n: int) -> float:
    """p-value of the 2-parameter change-point Fmax (cpdFmax2pCp2.m).

    Inside the table, p is interpolated (pchip) between the critical values.
    Below the 80 % value and above the 99 % value it is extrapolated with the
    ordinary F distribution (3, n) scaled to the table. The upper branch scales
    by the 95 % critical value, not the 99 % one (cpdFmax2pCp2.m:46), which is
    reproduced as is.
    """
    if np.isnan(fmax) or np.isnan(n) or n < 10:
        return np.nan
    crit = fmax_critical_cp2(int(n))
    if fmax < crit[0]:
        f_adj = special.fdtri(3, n, 0.90) * fmax / crit[0]
        p = 2.0 * (1.0 - special.fdtr(3, n, f_adj))
        return min(p, 1.0)
    if fmax > crit[-1]:
        f_adj = special.fdtri(3, n, 0.995) * fmax / crit[2]
        p = 2.0 * (1.0 - special.fdtr(3, n, f_adj))
        return max(p, 0.0)
    return matlab_pchip(crit, _ONE_MINUS_P_2, fmax)


# MATLAB's default random stream: Mersenne Twister mt19937ar, seed 0, which
# MATLAB maps to the reference seed 5489. numpy's legacy RandomState seeded with
# an integer uses the same init_genrand, and RandomState.random_sample uses the
# same 53-bit conversion (genrand_res53) as MATLAB ``rand``.
MATLAB_DEFAULT_SEED = 5489


def matlab_default_stream() -> np.random.RandomState:
    """A fresh MATLAB default stream, as at the start of a MATLAB (MCR) process."""
    return np.random.RandomState(MATLAB_DEFAULT_SEED)


def matlab_randi_positions(stream: np.random.RandomState, nt: int, mapping: str = 'ceil') -> np.ndarray:
    """0-based positions of ``randi(nt, nt, 1)`` drawn from ``stream``.

    Mappings from the stream to integers in 1..nt:

    - ``'ceil'``: ``ceil(rand*nt)``, one double per value. ``rng default;
      randi(10, 5)`` in MATLAB gives the documented 9 10 2 10 7 ..., which is
      ``ceil(10*rand)`` of the first doubles 0.8147 0.9058 0.1270 0.9134 0.6324.
    - ``'floor'``: ``floor(rand*nt) + 1``. Differs from 'ceil' only when
      ``rand*nt`` is an integer.
    - ``'int32'``: one 32-bit word per value, ``floor(w*nt/2^32) + 1``.
    """
    if mapping == 'ceil':
        u = stream.random_sample(nt)
        pos = np.ceil(u * nt).astype(np.int64) - 1
        return np.maximum(pos, 0)
    if mapping == 'floor':
        u = stream.random_sample(nt)
        return np.floor(u * nt).astype(np.int64)
    if mapping == 'int32':
        w = stream._bit_generator.random_raw(nt).astype(np.uint64)
        return ((w * np.uint64(nt)) >> np.uint64(32)).astype(np.int64)
    raise ValueError(f"Unknown randi mapping '{mapping}', use 'ceil', 'floor' or 'int32'")
