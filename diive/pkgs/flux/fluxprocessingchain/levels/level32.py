"""
LEVEL 3.2: STEPWISE OUTLIER DETECTION
======================================

Composable callable that wraps a pre-configured ``StepwiseOutlierDetection``
instance and computes the overall QCF.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace

from diive.core.utils.console import rule
from diive.pkgs.flux.fluxprocessingchain.container import FluxLevelData
from diive.pkgs.flux.fluxprocessingchain.levels._qcf import finalize_level
from diive.pkgs.preprocessing.outlier_detection import StepwiseOutlierDetection


def make_level32_detector(data: FluxLevelData) -> StepwiseOutlierDetection:
    """
    Factory for a Level-3.2 ``StepwiseOutlierDetection`` wired to ``data``.

    Uses the correct ``dfin``, ``col``, site coordinates and ``idstr='L3.2'``
    so the user doesn't have to reproduce that wiring by hand.

    Usage::

        sod = make_level32_detector(data)

        # Each detector method + addflag() is one sequential step.
        # addflag() locks in the current flags so the *next* detector
        # operates only on the data that survived *this* step.
        # Omitting addflag() means the next detector sees unfiltered data.
        sod.flag_outliers_hampel_test(window_length=48 * 13, ...)
        sod.addflag()   # <- required after every detector call

        sod.flag_outliers_zscore_rolling(...)
        sod.addflag()   # <- required again

        data = run_level32(data, outlier_detector=sod)
    """
    if data.filteredseries is None:
        raise RuntimeError("run_level31() (or run_level2()) must be called first; "
                           "no filteredseries available to wire into the detector.")
    return StepwiseOutlierDetection(
        dfin=data.fpc_df,
        col=str(data.filteredseries.name),
        site_lat=data.meta.site_lat,
        site_lon=data.meta.site_lon,
        utc_offset=data.meta.utc_offset,
        idstr='L3.2',
    )


def run_level32(
        data: FluxLevelData,
        *,
        outlier_detector: StepwiseOutlierDetection,
) -> FluxLevelData:
    """
    Level-3.2: Apply a pre-configured StepwiseOutlierDetection and compute QCF.

    The caller is responsible for constructing and configuring
    ``outlier_detector``.  Use ``make_level32_detector(data)`` for the correct
    wiring, then chain detector calls and ``addflag()`` calls alternately before
    passing it in.  Each ``sod.addflag()`` call locks in the current flags so
    the *next* detector method sees only the surviving records — omitting it
    breaks the sequential filtering guarantee.

    Args:
        data: FluxLevelData after ``run_level31()``.
        outlier_detector: Fully configured ``StepwiseOutlierDetection`` instance.

    Returns:
        Updated FluxLevelData with ``levels.level32``, ``levels.level32_qcf``,
        and ``levels.filteredseries_level32_qcf`` populated.
    """
    if data.levels.flux_corrected_col is None:
        raise RuntimeError("run_level31() must be called before run_level32().")

    # Guard: if no FLAG_ columns exist the user forgot to call flag_outliers_* + addflag().
    # Without this check every record silently passes (QCF=0) because FlagQCF finds
    # nothing to combine — the data looks clean when no test actually ran.
    flag_cols = [c for c in outlier_detector.flags.columns if str(c).startswith("FLAG_")]
    if not flag_cols:
        raise RuntimeError(
            "outlier_detector has no FLAG_ columns — did you forget to call "
            "sod.flag_outliers_*() followed by sod.addflag() before run_level32()? "
            "Without at least one completed test every record would pass silently."
        )

    idstr = 'L3.2'
    rule("Level 3.2: Stepwise Outlier Detection")

    updated, qcf = finalize_level(
        data,
        run_qcf_on_col=data.levels.flux_corrected_col,
        idstr=idstr,
        level_df=outlier_detector.flags,
    )

    new_levels = replace(
        updated.levels,
        level32=outlier_detector,
        level32_qcf=qcf,
        filteredseries_level32_qcf=updated.filteredseries.copy(),
    )
    level_ids = list(updated.level_ids)
    if idstr not in level_ids:
        level_ids.append(idstr)

    return replace(updated, levels=new_levels, level_ids=level_ids)
