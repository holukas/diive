"""
FFP_KERNEL: NUMERICS OF THE FLUX FOOTPRINT PREDICTION (FFP)
===========================================================

Model constants and the per-cell footprint of Kljun et al. (2015), used by
``diive.flux.lowres.footprint``.

Part of the diive library: https://github.com/holukas/diive

The module imports only numpy, so the worker processes of ``ffp_climatology`` start
without loading the rest of ``diive.flux``. The operations and their order follow the
reference code; changing either breaks the bitwise parity.
"""

import numpy as np

# Model parameters (Kljun et al. 2015, Eqs. 14, 17, 19)
A = 1.4524
B = -1.9914
C = 1.4622
D = 0.1359
AC = 2.17
BC = 1.66
CC = 20.0
XSTAR_END = 30
OLN = 5000  # Limit of |L| for neutral scaling
K = 0.4  # von Karman constant


def psi_f(zm, ol):
    """Stability correction of the wind profile (unstable/neutral or stable)."""
    if ol <= 0 or ol >= OLN:
        xx = (1 - 19.0 * zm / ol) ** 0.25
        return np.log((1 + xx ** 2) / 2.) + 2. * np.log((1 + xx) / 2.) - 2. * np.arctan(xx) + np.pi / 2
    return -5.3 * zm / ol


def scale_const(zm, ol):
    """Scaling constant of sig_y (convective or stable)."""
    if abs(ol) > OLN:
        ol = -1E6
    if ol <= 0:
        const = 1E-5 * abs(zm / ol) ** (-1) + 0.80
    else:
        const = 1E-5 * abs(zm / ol) ** (-1) + 0.55
    return min(const, 1.0)


def footprint_cells(rho, rotated_theta, ustar, sigmav, h, ol, zm, z0, umean):
    """Footprint of one record in the given cells, zero outside the footprint."""
    # Real-scale crosswind-integrated footprint
    if z0 is not None:
        psi = psi_f(zm, ol)
        xstar = (rho * np.cos(rotated_theta) / zm * (1. - (zm / h))
                 / (np.log(zm / z0) - psi))
        px = xstar > D
        xs = xstar[px]
        fstar = A * (xs - D) ** B * np.exp(-C / (xs - D))
        f_ci = fstar / zm * (1. - (zm / h)) / (np.log(zm / z0) - psi)
    else:
        xstar = (rho * np.cos(rotated_theta) / zm * (1. - (zm / h))
                 / (umean / ustar * K))
        px = xstar > D
        xs = xstar[px]
        fstar = A * (xs - D) ** B * np.exp(-C / (xs - D))
        f_ci = fstar / zm * (1. - (zm / h)) / (umean / ustar * K)

    # Scaled sig_y* and real-scale sig_y
    sigystar = AC * np.sqrt(BC * np.abs(xs) ** 2 / (1 + CC * np.abs(xs)))
    sigy = sigystar / scale_const(zm, ol) * zm * sigmav / ustar
    sigy[sigy < 0] = np.nan

    # Real-scale f(x, y). Cells outside px add zero, as in the reference.
    f = np.zeros(rho.size)
    f[px] = (f_ci / (np.sqrt(2 * np.pi) * sigy)
             * np.exp(-(rho[px] * np.sin(rotated_theta[px])) ** 2 / (2. * sigy ** 2)))
    return f


def climatology_block(block, records, use_z0):
    """Sum of the footprints of all records, in record order, in one block of cells.

    ``records`` holds one row per record: wind direction (radians), ustar, sigmav, h,
    ol, zm, z0, umean and the start and end of up to two slices of angle-sorted cells.
    Each cell gets the same operations as in the reference, so the sums are identical.
    """
    b0, theta, rho = block
    acc = np.zeros(theta.size)
    b1 = b0 + theta.size
    # Python floats, as the reference receives them from lists
    for wind_dir_rad, ustar, sigmav, h, ol, zm, z0, umean, s0, e0, s1, e1 in records.tolist():
        rec = dict(ustar=ustar, sigmav=sigmav, h=h, ol=ol, zm=zm,
                   z0=z0 if use_z0 else None, umean=None if use_z0 else umean)
        for start, end in ((s0, e0), (s1, e1)):
            lo, hi = int(max(start, b0)) - b0, int(min(end, b1)) - b0
            if lo < hi:
                acc[lo:hi] += footprint_cells(rho[lo:hi], theta[lo:hi] - wind_dir_rad, **rec)
    return b0, acc
