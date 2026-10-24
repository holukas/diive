"""
KDE: GAUSSIAN KERNEL DENSITY ON A GRID OF POINTS
================================================

Fast Gaussian kernel density of 1-D samples, evaluated at many points.

Part of the diive library: https://github.com/holukas/diive
"""

import math

import numpy as np
from numpy.polynomial.hermite_e import hermeval
from scipy.signal import fftconvolve

#: Farther than this many bandwidths, exp(-u**2 / 2) underflows to 0 and adds nothing.
_REACH = 40.0

#: Taylor terms of the binned sum. A sample sits at most a quarter bandwidth from its
#: node, so the first term left out is below 1e-18 of a kernel's peak (Cramer's bound
#: on Hermite polynomials).
_ORDER = 16

#: (point, sample) pairs per block of the direct sum, which bounds the memory it needs.
_PAIRS = 2_000_000

#: Largest binning grid. Beyond it the direct sum is used.
_NODES = 1 << 22


def gaussian_kde_on_grid(data: np.ndarray, x: np.ndarray, bw: float) -> np.ndarray:
    """Return the Gaussian kernel density of *data* with bandwidth *bw*, evaluated at *x*.

    The same density as ``KernelDensity(kernel='gaussian', bandwidth=bw)`` from
    scikit-learn (after ``exp`` of ``score_samples``), to about 1e-13 of the peak
    density, but in milliseconds instead of seconds for 10,000s of samples on a
    1000-point grid. scikit-learn sums every sample at every point. Here the sum
    takes one of two routes, whichever costs less:

    - direct: at each point only the samples within 40 bandwidths, where the
      kernel has not yet underflowed to zero. Cheap for a narrow bandwidth.
    - binned, for evenly spaced points (``np.linspace``): every
      sample goes to the nearest node of a grid that contains the points and is
      spaced at most half a bandwidth. Its kernel is written as a Taylor series
      around the node (Hermite polynomials), so the sums become a few
      convolutions of per-node moments. The cost grows with the grid size, not
      with the number of samples times points.

    Args:
        data: Samples, finite values only.
        x: Points at which the density is evaluated.
        bw: Kernel bandwidth (standard deviation of the Gaussian kernel).
    """
    data = np.sort(np.asarray(data, dtype=float))
    x = np.asarray(x, dtype=float)
    norm = len(data) * bw * math.sqrt(2 * math.pi)
    step = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 0.0
    if step == 0:  # all points equal: one sum is enough
        return np.full(len(x), _sum_direct(data, x[:1], bw)[0] / norm)
    if step < 0:  # decreasing points (e.g. a reversed axis)
        return gaussian_kde_on_grid(data, x[::-1], bw)[::-1]
    reach = _REACH * bw
    lo = np.searchsorted(data, x - reach, side="left")
    hi = np.searchsorted(data, x + reach, side="right")
    k = math.ceil(2 * step / bw)  # grid nodes per point step
    n_nodes = (len(x) - 1) * k + 2 * min(math.ceil(reach * k / step), (len(x) - 1) * k + 1)
    # The direct sum costs one exp per pair, the binned one some 17 convolutions over the grid
    if (hi - lo).sum() <= 10 * n_nodes or n_nodes > _NODES:
        total = _sum_direct(data, x, bw, lo, hi)
    else:
        total = _sum_binned(data[lo.min():hi.max()], x, bw, step, k)
    return total / norm


def _sum_direct(data_sorted, x, bw, lo=None, hi=None):
    """Sum exp(-u**2 / 2) at each point over the sorted samples ``data_sorted[lo:hi]``."""
    if lo is None:
        reach = _REACH * bw
        lo = np.searchsorted(data_sorted, x - reach, side="left")
        hi = np.searchsorted(data_sorted, x + reach, side="right")
    cnt = hi - lo
    out = np.zeros(len(x))
    if cnt.sum() == 0:
        return out
    csum = np.cumsum(cnt)
    edges = np.searchsorted(csum, np.arange(_PAIRS, csum[-1], _PAIRS), side="right")
    for j0, j1 in zip(np.r_[0, edges], np.r_[edges, len(x)], strict=True):
        c = cnt[j0:j1]
        if c.sum() == 0:
            continue
        point = np.repeat(np.arange(j1 - j0), c)
        sample = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c) + np.repeat(lo[j0:j1], c)
        u = (x[j0:j1][point] - data_sorted[sample]) / bw
        out[j0:j1] = np.bincount(point, weights=np.exp(-0.5 * u * u), minlength=j1 - j0)
    return out


def _sum_binned(data, x, bw, step, k):
    """Sum exp(-u**2 / 2) at increasing, evenly spaced points, binned with Taylor terms.

    A sample ``s`` bandwidths from its node contributes
    ``phi(t - s) = sum_p s**p / p! * He_p(t) * phi(t)`` at a point ``t``
    bandwidths from that node (``He_p``: probabilists' Hermite polynomials).
    Summed over samples, each term is the per-node moment ``sum s**p``
    convolved with ``He_p(t) * phi(t) / p!``.
    """
    delta = step / k
    node = np.rint((data - x[0]) / delta).astype(np.int64)
    s = (data - (x[0] + node * delta)) / bw  # |s| <= 0.25
    first = min(node.min(), 0)
    node -= first
    n_nodes = max(node.max() + 1, (len(x) - 1) * k - first + 1)
    half = min(math.ceil(_REACH * bw / delta), n_nodes)
    t = np.arange(-half, half + 1) * (delta / bw)
    phi = np.exp(-0.5 * t * t)
    total = np.zeros(n_nodes + 2 * half)
    s_pow = np.ones_like(s)
    for p in range(_ORDER + 1):
        coef = np.zeros(p + 1)
        coef[p] = 1.0 / math.factorial(p)
        moments = np.bincount(node, weights=s_pow, minlength=n_nodes)
        total += fftconvolve(moments, hermeval(t, coef) * phi)
        s_pow *= s
    # FFT rounding leaves values around 1e-17 of the peak, slightly negative in the tails.
    return np.maximum(total[half - first + np.arange(len(x)) * k], 0.0)
