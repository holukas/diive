"""
QCF HELPER: SHARED FINALIZATION ACROSS LEVELS
==============================================

Merges new flag columns into the working dataframe and computes the
overall quality flag (QCF) for the level.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from pandas import DataFrame

from diive.core.dfun.frames import detect_new_columns
from diive.core.utils.console import detail
from diive.pkgs.flux.fluxprocessingchain.container import FluxLevelData
from diive.pkgs.preprocessing.qaqc import FlagQCF


def finalize_level(
        data: FluxLevelData,
        *,
        run_qcf_on_col: str,
        idstr: str,
        level_df: DataFrame,
        ustar_scenarios: list[str] | None = None,
) -> tuple[FluxLevelData, FlagQCF]:
    """
    Merge new flag columns into ``fpc_df`` and compute an overall quality flag.

    Args:
        data: Current FluxLevelData.
        run_qcf_on_col: The flux column to run QCF on.
        idstr: Level identifier string (e.g. 'L2', 'L3.2').
        level_df: DataFrame produced by the level (containing new flag columns).
        ustar_scenarios: Optional list of USTAR scenario labels so FlagQCF
            picks the right ``FLAG_`` columns.

    Returns:
        (updated FluxLevelData, FlagQCF instance)
    """
    new_cols = detect_new_columns(df=level_df, other=data.fpc_df)
    fpc_df = pd.concat([data.fpc_df, level_df[new_cols]], axis=1)
    for col in new_cols:
        detail(f"Added column {col}.")

    qcf = FlagQCF(
        target_col=run_qcf_on_col,
        df=fpc_df,
        idstr=idstr,
        swinpot_col=data.meta.swinpot_col,
        nighttime_threshold=data.meta.nighttime_threshold,
        ustar_scenarios=ustar_scenarios,
    )
    qcf.calculate(
        daytime_accept_qcf_below=data.meta.daytime_accept_qcf_below,
        nighttime_accept_qcf_below=data.meta.nighttime_accept_qcf_below,
    )
    fpc_df = qcf.get()

    # Return a new FluxLevelData; never mutate the input.  ``levels`` is left
    # untouched here — the calling level function is responsible for setting
    # its own fields via ``dataclasses.replace(updated.levels, ...)``.
    updated = replace(
        data,
        fpc_df=fpc_df,
        filteredseries=qcf.filteredseries.copy(),
        level_ids=list(data.level_ids),
    )
    return updated, qcf
