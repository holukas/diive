"""
LEVEL 3.2: STEPWISE OUTLIER DETECTION
======================================

Composable callable that wraps a pre-configured ``StepwiseOutlierDetection``
instance and computes the overall QCF.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace

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
        sod.flag_outliers_hampel_test(window_length=48 * 13, ...)
        sod.addflag()
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
    wiring, then call any number of ``sod.flag_outliers_*`` / ``sod.addflag()``
    methods before passing it in.

    Args:
        data: FluxLevelData after ``run_level31()``.
        outlier_detector: Fully configured ``StepwiseOutlierDetection`` instance.

    Returns:
        Updated FluxLevelData with ``levels.level32``, ``levels.level32_qcf``,
        and ``levels.filteredseries_level32_qcf`` populated.
    """
    if data.levels.flux_corrected_col is None:
        raise RuntimeError("run_level31() must be called before run_level32().")

    idstr = 'L3.2'
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
