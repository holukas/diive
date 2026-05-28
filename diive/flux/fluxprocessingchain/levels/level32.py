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
from diive.flux.fluxprocessingchain.container import FluxLevelData
from diive.flux.fluxprocessingchain.levels._qcf import finalize_level
from diive.flux.fluxprocessingchain.levels._rerun import (
    cascade_reset,
    record_added_columns,
)
from diive.preprocessing.outlier_detection import StepwiseOutlierDetection


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
    # Re-run support: if L3.3 has already run, filteredseries is None because
    # L3.3 deliberately clears it (multi-scenario). For a re-run of L3.2 to
    # work, we need to wire the detector against the L3.1 QCF series that the
    # cascade reset will restore inside run_level32(). We apply the cascade
    # pre-emptively here so the detector's input matches what run_level32()
    # will see — otherwise the staleness guard there would reject this
    # detector.
    if 'L3.3' in data.level_ids:
        from diive.flux.fluxprocessingchain.levels._rerun import cascade_reset
        data = cascade_reset(data, 'L3.2')

    if data.filteredseries is None:
        raise RuntimeError(
            "run_level31() (or run_level2()) must be called first; "
            "no filteredseries available to wire into the detector."
        )
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
        raise RuntimeError(
            "run_level31() must be called before run_level32(). For energy "
            "fluxes (H, LE) without a storage profile, do not skip Level-3.1 "
            "— call run_level31(data, set_storage_to_zero=True) to satisfy "
            "the chain's ordering requirement; L3.1 will still run, it just "
            "adds zero instead of a measured storage term."
        )

    idstr = 'L3.2'

    # Re-run cleanup: cascade-clear L3.2 and downstream state from any prior
    # invocation before running fresh. The detector staleness guards below
    # then operate against the post-cascade snapshot, so passing in a
    # pre-cascade detector correctly fails with a "rebuild" message.
    if idstr in data.level_ids:
        data = cascade_reset(data, idstr)
    pre_columns = list(data.fpc_df.columns)

    # Guard: the detector holds a *copy* of fpc_df made at construction time.
    # If the user rebuilt `data` (e.g. re-ran run_level31 with different
    # settings) without rebuilding the detector, the outlier flags would be
    # computed against stale data and then applied to the new fpc_df. Compare
    # the input series identity (label + values, NaN-aware) to catch this.
    expected_name = str(data.filteredseries.name) if data.filteredseries is not None else None
    if expected_name is None or outlier_detector.col != expected_name:
        raise RuntimeError(
            f"outlier_detector was wired to column {outlier_detector.col!r}, but "
            f"data.filteredseries is {expected_name!r}. Rebuild the detector via "
            f"make_level32_detector(data) after the most recent run_level31()."
        )
    if not outlier_detector.dfin[expected_name].equals(data.fpc_df[expected_name]):
        raise RuntimeError(
            "outlier_detector was constructed from a different FluxLevelData "
            "snapshot than the one passed to run_level32() (the input flux "
            "series differs). Rebuild the detector via "
            "make_level32_detector(data) after the most recent run_level31()."
        )

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

    # Guard: the most recent flag_outliers_*() call may not have been followed by
    # addflag(). In that case the last test's flag lives only in `_last_flag` and
    # is silently dropped when finalize_level reads `outlier_detector.flags`.
    # addflag() does not clear _last_flag, but it does copy it (possibly under a
    # renamed key for re-runs) into `.flags`. So if `_last_flag.name` matches no
    # column in `.flags` — neither directly nor under the `_N_TEST` re-run rename
    # — the user forgot to commit it.
    last = outlier_detector.last_flag
    if last is not None and not last.empty:
        last_name = str(last.name)
        committed_cols = set(map(str, outlier_detector.flags.columns))
        if last_name not in committed_cols:
            rerun_prefix = last_name.replace('_TEST', '_')
            rerun_match = any(c.startswith(rerun_prefix) and c.endswith('_TEST')
                              for c in committed_cols)
            if not rerun_match:
                raise RuntimeError(
                    f"outlier_detector has an uncommitted last flag {last_name!r} — "
                    f"the most recent sod.flag_outliers_*() call was not followed "
                    f"by sod.addflag(), so that test would be silently dropped from "
                    f"the QCF. Call sod.addflag() before run_level32()."
                )

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

    final = replace(updated, levels=new_levels, level_ids=level_ids)
    return replace(final, added_columns=record_added_columns(final, idstr, pre_columns))
