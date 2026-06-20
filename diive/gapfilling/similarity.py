"""
SIMILARITY: METEOROLOGICAL SIMILARITY FOR MDS-STYLE METHODS
============================================================

Shared meteorological-similarity primitives used by both MDS gap-filling
(Reichstein et al. 2005) and PAS20 random-uncertainty estimation
(Pastorello et al. 2020 / ONEFlux). Both pool measured fluxes that occur under
"similar" meteorological conditions (SWIN, TA, VPD) and reduce them to a
statistic — the mean for gap-filling, the standard deviation for uncertainty.
The tolerance constants and the per-window mean/SD/count reduction are the same
definition in both, so they live here once.

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

    ``ddof`` selects the standard-deviation convention: ``0`` (population, the
    MDS gap-filling default) or ``1`` (sample / N-1, the ONEFlux random-
    uncertainty convention).
    """
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    count = int(valid.size)
    if count == 0:
        return np.nan, np.nan, 0
    if count >= min_count and count > ddof:
        return float(np.mean(valid)), float(np.std(valid, ddof=ddof)), count
    return np.nan, np.nan, count
