"""
LEVEL 3.1: STORAGE CORRECTION
==============================

Composable callable that applies single-point storage correction to the
flux and computes the L3.1 QCF by re-aggregating the L2-inherited flags
on the storage-corrected target.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace

from diive.core.utils.console import rule
from diive.flux.fluxprocessingchain.container import FluxLevelData
from diive.flux.fluxprocessingchain.levels._qcf import finalize_level
from diive.flux.fluxprocessingchain.levels._rerun import (
    cascade_reset,
    record_added_columns,
)
from diive.flux.lowres.storage_correction import (
    FluxStorageCorrectionSinglePointEddyPro,
)

#: Flux column -> the storage-term column L3.1 auto-detects for it (single-point
#: EddyPro-FLUXNET naming). Mirrors FluxStorageCorrectionSinglePointEddyPro's
#: internal map; lets a caller (e.g. the GUI) show / seed the storage column and
#: check availability, the L3.1 analogue of level2_test_inputs.
_STORAGE_COLS = {
    'FC': 'SC_SINGLE',
    'FH2O': 'SH2O_SINGLE',
    'LE': 'SLE_SINGLE',
    'ET': 'SET_SINGLE',
    'FN2O': 'SN2O_SINGLE',
    'FCH4': 'SCH4_SINGLE',
    'H': 'SH_SINGLE',
}


def level31_storage_col(fluxcol: str) -> str | None:
    """The storage-term column ``run_level31`` auto-detects for ``fluxcol``.

    Returns the EddyPro-FLUXNET single-point storage column name (e.g.
    ``'FC' -> 'SC_SINGLE'``), or ``None`` when the flux is not in the standard
    map (the caller should then ask the user for an explicit ``strgcol``).
    """
    return _STORAGE_COLS.get(fluxcol)


def run_level31(
        data: FluxLevelData,
        *,
        gapfill_storage_term: bool = True,
        set_storage_to_zero: bool = False,
        strgcol: str | None = None,
) -> FluxLevelData:
    """
    Level-3.1: Apply single-point storage correction and compute the L3.1 QCF.

    Requires Level-2 to have been run first (reads ``data.levels.level2_qcf``).

    Produces:

    - ``levels.level31`` — the fitted storage-correction object
      (``FluxStorageCorrectionSinglePointEddyPro``).
    - ``levels.level31_qcf`` — a ``FlagQCF`` re-aggregating the L2-inherited
      flags on the storage-corrected target column. Symmetric with the
      ``level2_qcf`` / ``level32_qcf`` / ``level33_qcf`` fields produced by
      the other levels.
    - ``levels.flux_corrected_col`` — name of the storage-corrected column
      (e.g. ``'NEE_L3.1'``).
    - ``levels.filteredseries_level31_qcf`` — the user-accepted filtered
      series. Same content as ``level31_qcf.filteredseries``.
    - ``levels.filteredseries_hq`` — the strictly-QCF=0 series (overwrites
      whatever L2 wrote, with the storage-correction applied).

    L3.1 introduces **no new quality test**. The ``FLAG_..._ISFILLED`` column
    that ``FluxStorageCorrectionSinglePointEddyPro`` emits when
    ``gapfill_storage_term=True`` is **informational only** (0 = storage was
    measured, 1 = storage was gap-filled with a rolling median); it does
    *not* end in ``_TEST`` and is therefore deliberately ignored by
    ``FlagQCF``. Whether a record's storage term was measured or filled is
    not a quality criterion — users who care about provenance read the
    ISFILLED column directly from ``data.fpc_df``.

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
        strgcol: Explicit name of the storage-term column in
            ``data.full_df``. Overrides the built-in auto-detection
            (``FC`` -> ``SC_SINGLE``, ``LE`` -> ``SLE_SINGLE``, ...). Pass
            this when your storage column does not follow the FLUXNET /
            Swiss FluxNet naming convention (e.g. EddyPro full-output
            ``co2_strg``, or a custom site-specific name). Default ``None``
            (auto-detect).

    Returns:
        Updated FluxLevelData; see the field list above for what is populated.
    """
    if data.levels.level2_qcf is None:
        raise RuntimeError("run_level2() must be called before run_level31().")

    idstr = 'L3.1'

    # Re-run cleanup: drop columns and downstream state from any prior L3.1
    # invocation before producing fresh ones. See levels/_rerun.py.
    if idstr in data.level_ids:
        data = cascade_reset(data, idstr)
    pre_columns = list(data.fpc_df.columns)

    rule("Level 3.1: Storage Correction")
    meta = data.meta

    level31 = FluxStorageCorrectionSinglePointEddyPro(
        df=data.full_df,
        fluxcol=meta.fluxcol,
        basevar=meta.fluxbasevar,
        gapfill_storage_term=gapfill_storage_term,
        idstr=idstr,
        set_storage_to_zero=set_storage_to_zero,
        strgcol=strgcol,
    )
    level31.storage_correction()

    flux_corrected_col = level31.flux_corrected_col

    # finalize_level handles: (a) the index-equality check, (b) adding the
    # storage-correction output columns (the corrected flux, the gap-filled
    # storage term, and the informational FLAG_..._ISFILLED provenance flag)
    # to fpc_df, (c) constructing a FlagQCF keyed by idstr=L3.1, (d) running
    # QCF aggregation across the L2-inherited flag columns relevant to the
    # corrected flux, and (e) writing the QCF / QCF0 filtered-series
    # columns. The ISFILLED column does not end in '_TEST' and is therefore
    # not picked up by FlagQCF — gap-filled storage is provenance, not a
    # quality test. Passing outname=meta.outname keeps the user-visible
    # column names compact (e.g. ``NEE_L3.1_QCF`` rather than
    # ``NEE_L3.1_L3.1_QCF``).
    updated, qcf = finalize_level(
        data,
        run_qcf_on_col=flux_corrected_col,
        idstr=idstr,
        level_df=level31.results,
        outname=meta.outname,
    )

    new_levels = replace(
        updated.levels,
        level31=level31,
        level31_qcf=qcf,
        flux_corrected_col=flux_corrected_col,
        filteredseries_level31_qcf=updated.filteredseries.copy(),
        filteredseries_hq=qcf.filteredseries_hq.copy(),
    )
    level_ids = list(updated.level_ids)
    if idstr not in level_ids:
        level_ids.append(idstr)

    final = replace(updated, levels=new_levels, level_ids=level_ids)
    return replace(final, added_columns=record_added_columns(final, idstr, pre_columns))
