"""
LEVEL 3.1: STORAGE CORRECTION
==============================

Composable callable that applies single-point storage correction to the
flux and re-applies the Level-2 QCF to the storage-corrected series.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from diive.core.dfun.frames import detect_new_columns
from diive.core.utils.console import detail, rule
from diive.pkgs.flux.fluxprocessingchain.container import FluxLevelData
from diive.pkgs.flux.lowres.storage_correction import (
    FluxStorageCorrectionSinglePointEddyPro,
)


def run_level31(
        data: FluxLevelData,
        *,
        gapfill_storage_term: bool = True,
        set_storage_to_zero: bool = False,
) -> FluxLevelData:
    """
    Level-3.1: Apply single-point storage correction to the flux.

    Requires Level-2 to have been run first (reads ``data.levels.level2_qcf``).

    Args:
        data: FluxLevelData after ``run_level2()``.
        gapfill_storage_term: Gap-fill missing storage values with a rolling
            median before adding the term to the flux.  Default True.
        set_storage_to_zero: Set the storage term to zero instead of using
            measured data.  Use this for fluxes where no storage profile
            exists (e.g. H, LE at low-canopy sites) or where the
            single-point approximation is considered unreliable.
            **Level-3.2 and Level-3.3 require Level-3.1 to have run.**
            Even for H and LE, call this function with
            ``set_storage_to_zero=True`` rather than skipping it entirely,
            so that the chain's ordering guards are satisfied.

    Returns:
        Updated FluxLevelData with ``levels.level31``,
        ``levels.flux_corrected_col``, and ``levels.filteredseries_level31_qcf``
        populated.
    """
    if data.levels.level2_qcf is None:
        raise RuntimeError("run_level2() must be called before run_level31().")

    idstr = 'L3.1'

    rule("Level 3.1: Storage Correction")
    meta = data.meta

    level31 = FluxStorageCorrectionSinglePointEddyPro(
        df=data.full_df,
        fluxcol=meta.fluxcol,
        basevar=meta.fluxbasevar,
        gapfill_storage_term=gapfill_storage_term,
        idstr=idstr,
        set_storage_to_zero=set_storage_to_zero,
    )
    level31.storage_correction()

    # Merge new columns from level31.results into fpc_df
    new_cols = detect_new_columns(df=level31.results, other=data.fpc_df)
    fpc_df = pd.concat([data.fpc_df, level31.results[new_cols]], axis=1)
    for col in new_cols:
        detail(f"Added column {col}.")

    # Apply Level-2 QCF to storage-corrected flux
    level2_qcf = data.levels.level2_qcf
    flux_corrected_col = level31.flux_corrected_col

    strg_qcf = level31.results[flux_corrected_col].copy()
    strg_qcf.loc[level2_qcf.filteredseries.isnull()] = np.nan
    strg_qcf.name = f"{flux_corrected_col}_QCF"

    strg_qcf0 = level31.results[flux_corrected_col].copy()
    strg_qcf0.loc[level2_qcf.filteredseries_hq.isnull()] = np.nan
    strg_qcf0.name = f"{flux_corrected_col}_QCF0"

    fpc_df = pd.concat(
        [fpc_df, pd.DataFrame({strg_qcf.name: strg_qcf, strg_qcf0.name: strg_qcf0})],
        axis=1,
    )
    for c in (strg_qcf.name, strg_qcf0.name):
        detail(f"Added column {c} (L3.1 with L2 quality flag applied).")

    new_levels = replace(
        data.levels,
        level31=level31,
        flux_corrected_col=flux_corrected_col,
        filteredseries_level31_qcf=strg_qcf.copy(),
        filteredseries_hq=strg_qcf0.copy(),
    )
    level_ids = list(data.level_ids)
    if idstr not in level_ids:
        level_ids.append(idstr)

    return replace(
        data,
        fpc_df=fpc_df,
        filteredseries=strg_qcf,
        levels=new_levels,
        level_ids=level_ids,
    )
