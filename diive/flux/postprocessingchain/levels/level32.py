"""
LEVEL 3.2: STEPWISE OUTLIER DETECTION
======================================

Composable callable that wraps a pre-configured ``StepwiseOutlierDetection``
instance and computes the overall QCF.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from diive.core.utils.console import rule
from diive.flux.postprocessingchain.container import FluxLevelData
from diive.flux.postprocessingchain.levels._qcf import finalize_level
from diive.flux.postprocessingchain.levels._rerun import (
    cascade_reset,
    record_added_columns,
)
from diive.preprocessing.outlier_detection import StepwiseOutlierDetection


def nighttime_uncertainty(
        data: FluxLevelData,
        outlier_detector: StepwiseOutlierDetection,
        randunc_col: str,
        ta: str,
        vpd: str,
        swin: str,
        vpd_in_kpa: bool = True,
        z_values: tuple = (2, 3),
        verbose: int | bool | None = None,
) -> pd.DataFrame:
    """Random uncertainty of each record, for the negative-nighttime test of Level 3.2.

    Three uncertainties per measured record, as columns of the returned frame:

    - ``FS``: the uncertainty EddyPro reports for each half-hour (``randunc_col``, e.g.
      ``FC_RANDUNC_HF``). It covers the sampling error of that half-hour's covariance only
      (Finkelstein and Sims 2001), so it is small.
    - ``PAS20``: the random uncertainty of FLUXNET2015 and ONEFlux (Pastorello et al. 2020,
      :class:`~diive.flux.lowres.uncertainty.RandomUncertaintyPAS20`): the spread of measured
      fluxes under similar conditions (air temperature, VPD, radiation) within days of each
      record. Besides the random error, that spread holds real flux differences between the
      records, so PAS20 is a generous estimate of the random error.
    - ``MAX``: the larger of the two for each record, or the one available where the other is
      missing. A value is then called clearly below zero only when neither uncertainty explains it.

    PAS20 is computed from the series the detector holds now, so call this after the other
    outlier tests have been accepted: outliers would otherwise widen it. Records without a value
    get no uncertainty, and a nighttime record without either uncertainty is not tested; the
    table reports how many there are. It compares, for each uncertainty and each ``z``, how many
    nighttime records lie below ``-z`` times it.

    Args:
        data: Chain container, for the drivers and the EddyPro uncertainty in ``full_df``.
        outlier_detector: The Level-3.2 detector, after the tests to run before this one.
        randunc_col: Column of ``data.full_df`` with EddyPro's random uncertainty of the flux.
        ta: Air temperature column of ``data.full_df`` (degC), complete.
        vpd: VPD column of ``data.full_df``, complete; kPa unless ``vpd_in_kpa`` is False.
        swin: Incoming shortwave radiation column of ``data.full_df`` (W m-2), complete.
        vpd_in_kpa: Whether ``vpd`` is in kPa.
        z_values: The ``z`` of the comparison table.
        verbose: Verbosity for the console helpers.

    Returns:
        DataFrame with the columns ``FS``, ``PAS20`` and ``MAX`` on the series' index.

    Example::

        data, sod = make_level32_detector(data)
        sod.flag_outliers_hampel_test(...); sod.addflag()
        sigma = nighttime_uncertainty(data, sod, randunc_col='FC_RANDUNC_HF',
                                      ta='TA_F', vpd='VPD_F', swin='SW_IN_F')
        sod.flag_outliers_negative_night_test(uncertainty=sigma['MAX'], z=2, showplot=True)
        sod.addflag()
    """
    from diive.flux.lowres.uncertainty import RandomUncertaintyPAS20
    from diive.flux.postprocessingchain.reports import _print_report, _table

    series = outlier_detector.series_hires_cleaned
    full = data.full_df
    for col in (randunc_col, ta, vpd, swin):
        if col not in full.columns:
            raise ValueError(f"{col!r} is not in data.full_df.")
    frame = pd.DataFrame({'flux': series,
                          # PAS20 takes a measured and a gap-filled flux; the test judges measured
                          # records only, so both are the measured series, under two names.
                          'flux_gf': series,
                          'ta': full[ta].reindex(series.index),
                          'vpd': full[vpd].reindex(series.index),
                          'swin': full[swin].reindex(series.index)})
    pas = RandomUncertaintyPAS20(df=frame, fluxcol='flux', fluxgapfilledcol='flux_gf', tacol='ta',
                                 vpdcol='vpd', swincol='swin', vpd_in_kpa=vpd_in_kpa)
    # PAS20 reports the time of each of its four methods; that is debug detail here.
    from diive.core.utils.console import VERBOSE_DEBUG, VERBOSE_ERROR, _vlevel, get_verbosity, set_verbosity
    previous = get_verbosity()
    if _vlevel(verbose) < VERBOSE_DEBUG:
        set_verbosity(VERBOSE_ERROR)
    try:
        pas.run()
    finally:
        set_verbosity(previous)
    has = series.notna()
    out = pd.DataFrame({'FS': full[randunc_col].reindex(series.index).where(has),
                        'PAS20': pas.randunc_series.reindex(series.index).where(has)})
    out['MAX'] = out[['FS', 'PAS20']].max(axis=1)

    night = data.fpc_df['NIGHTTIME'].reindex(series.index) == 1
    value = series[night & has]
    table = _table()
    table.add_column("uncertainty")
    table.add_column("median at night", justify="right")
    for z in z_values:
        table.add_column(f"below -{z:g} sigma", justify="right")
    for col in out.columns:
        sigma = out.loc[value.index, col]
        cells = [col, f"{sigma.median():.2f}"]
        for z in z_values:
            n = int((value < -z * sigma).sum())
            cells.append(f"{n:,} ({100 * n / len(value):.1f} %)" if len(value) else "-")
        table.add_row(*cells)
    notes = ["FS: EddyPro, sampling error of each half-hour. PAS20: spread of similar "
             "measurements (ONEFlux), a generous estimate. MAX: the larger of the two."]
    for col in out.columns:
        n_missing = int(out.loc[value.index, col].isna().sum())
        if n_missing:
            notes.append(f"{col} is missing for {n_missing:,} nighttime records"
                         f"{'; these are not tested' if col == 'MAX' else ''}.")
    _print_report(f"Random uncertainty of the {len(value):,} nighttime records", table,
                  notes, verbose=verbose)
    return out


def make_level32_detector(
        data: FluxLevelData,
) -> tuple[FluxLevelData, StepwiseOutlierDetection]:
    """
    Factory for a Level-3.2 ``StepwiseOutlierDetection`` wired to ``data``.

    Uses the correct ``dfin``, ``col``, site coordinates and ``idstr='L3.2'``
    so the user doesn't have to reproduce that wiring by hand.

    **Returns a tuple ``(data, sod)``.** The returned ``data`` is normally the
    same container the caller passed in, *except* in the re-run case below.

    **Re-run support:** if L3.3 has already run on the input container,
    ``data.filteredseries`` is ``None`` (cleared by L3.3 because there is no
    unambiguous filtered series across USTAR scenarios). To allow a re-run
    of L3.2, this factory pre-emptively performs the L3.2 cascade reset on
    the input and returns the cascaded container as the first element of
    the tuple. The caller **must** rebind ``data`` to the returned value to
    keep the rest of their script consistent — the detector's input column
    only exists in the cascaded snapshot.

    Usage::

        data, sod = make_level32_detector(data)

        # Each detector method + addflag() is one sequential step.
        # addflag() locks in the current flags so the *next* detector
        # operates only on the data that survived *this* step.
        # Omitting addflag() means the next detector sees unfiltered data.
        sod.flag_outliers_hampel_test(window_length=48 * 13, ...)
        sod.addflag()   # <- required after every detector call

        sod.flag_outliers_zscore_rolling_test(...)
        sod.addflag()   # <- required again

        data = run_level32(data, outlier_detector=sod)
    """
    # Re-run support: when L3.3 already ran, filteredseries was cleared. We
    # apply the cascade now and return the cascaded ``data`` to the caller
    # explicitly (tuple return) so they can update their reference — staleness
    # of the caller's data was a real footgun before this signature change.
    if 'L3.3' in data.level_ids:
        from diive.flux.postprocessingchain.levels._rerun import cascade_reset
        data = cascade_reset(data, 'L3.2')

    if data.filteredseries is None:
        raise RuntimeError(
            "run_level31() (or run_level2()) must be called first; "
            "no filteredseries available to wire into the detector."
        )
    sod = StepwiseOutlierDetection(
        dfin=data.fpc_df,
        col=str(data.filteredseries.name),
        site_lat=data.meta.site_lat,
        site_lon=data.meta.site_lon,
        utc_offset=data.meta.utc_offset,
        idstr='L3.2',
        nighttime_threshold=data.meta.nighttime_threshold,
    )
    return data, sod


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
    # Check column existence on both sides BEFORE indexing — a mutated
    # detector (.col reassigned out-of-sync with .dfin) or a fpc_df missing
    # the expected column would otherwise raise a confusing KeyError instead
    # of the intended "rebuild the detector" message.
    if expected_name not in outlier_detector.dfin.columns:
        raise RuntimeError(
            f"outlier_detector.col={outlier_detector.col!r} but that column "
            f"is not present in outlier_detector.dfin (columns: "
            f"{list(outlier_detector.dfin.columns)}). The detector appears "
            f"to have been mutated after construction; rebuild via "
            f"make_level32_detector(data)."
        )
    if expected_name not in data.fpc_df.columns:
        raise RuntimeError(
            f"data.fpc_df does not contain the expected flux column "
            f"{expected_name!r}; the chain state is inconsistent. Rebuild "
            f"the chain from init_flux_data()."
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
    # ``StepwiseOutlierDetection`` initialises ``_last_flag`` as an empty
    # ``pd.DataFrame()`` (in its ``__init__``). Each
    # ``flag_outliers_*()`` method then assigns a ``pd.Series`` to
    # ``_last_flag`` whose ``.name`` is the test's flag-column name. The
    # check below only makes sense for the Series case — narrow with
    # ``isinstance`` so the DataFrame initial-state path doesn't blow up
    # on ``str(last.name)`` (DataFrame has no ``.name``).
    last = outlier_detector.last_flag
    if isinstance(last, pd.Series) and not last.empty:
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
