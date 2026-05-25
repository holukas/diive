"""
LEVEL 3.3: USTAR FILTERING
============================

Composable callable that flags low-turbulence periods using one or more
constant USTAR thresholds and computes per-scenario QCFs.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace

from diive.pkgs.flux.fluxprocessingchain.container import FluxLevelData
from diive.pkgs.flux.fluxprocessingchain.levels._qcf import finalize_level
from diive.pkgs.flux.lowres.ustarthreshold import FlagMultipleConstantUstarThresholds


def run_level33_constant_ustar(
        data: FluxLevelData,
        *,
        thresholds: list[float],
        threshold_labels: list[str],
        showplot: bool = True,
        verbose: bool = True,
) -> FluxLevelData:
    """
    Level-3.3: Flag low-turbulence periods using one or more constant USTAR thresholds.

    Requires Level-3.1 (``data.levels.flux_corrected_col``) and Level-3.2
    (``data.levels.filteredseries_level32_qcf``) to have been run.

    Args:
        data: FluxLevelData after ``run_level32()``.
        thresholds: List of USTAR threshold values (one per scenario).
        threshold_labels: Label for each threshold (e.g. ``['CUT_16', 'CUT_50']``).
        showplot: Show diagnostic plots. Defaults to True.
        verbose: Print progress. Defaults to True.

    Returns:
        Updated FluxLevelData with ``levels.level33``, ``levels.level33_qcf``
        (dict keyed by scenario), and ``levels.filteredseries_level33_qcf``
        (dict keyed by scenario) populated.
    """
    if data.levels.flux_corrected_col is None:
        raise RuntimeError("run_level31() must be called before run_level33_constant_ustar().")

    idstr = 'L3.3'
    meta = data.meta
    flux_corrected_col = data.levels.flux_corrected_col

    level33 = FlagMultipleConstantUstarThresholds(
        series=data.fpc_df[flux_corrected_col],
        ustar=data.fpc_df[meta.ustarcol],
        thresholds=thresholds,
        threshold_labels=threshold_labels,
        idstr=idstr,
        showplot=showplot,
    )
    level33.calc()

    level33_qcf: dict = {}
    filteredseries_level33_qcf: dict = {}
    current = data

    for ustar_scen in threshold_labels:
        flagcols = [c for c in level33.results if ustar_scen in c]
        flagcol = flagcols[0] if len(flagcols) == 1 else None
        udf = level33.results[[flux_corrected_col, meta.ustarcol, flagcol]].copy()

        current, qcf = finalize_level(
            current,
            run_qcf_on_col=flux_corrected_col,
            idstr=f'L3.3_{ustar_scen}',
            level_df=udf,
            ustar_scenarios=threshold_labels,
        )
        level33_qcf[ustar_scen] = qcf
        filteredseries_level33_qcf[ustar_scen] = current.filteredseries.copy()
        if verbose:
            print(f"++ Calculated overall quality flag QCF for USTAR scenario {ustar_scen}.")

    new_levels = replace(
        current.levels,
        level33=level33,
        level33_qcf=level33_qcf,
        filteredseries_level33_qcf=filteredseries_level33_qcf,
    )
    level_ids = list(current.level_ids)
    if idstr not in level_ids:
        level_ids.append(idstr)

    return replace(current, levels=new_levels, level_ids=level_ids)
