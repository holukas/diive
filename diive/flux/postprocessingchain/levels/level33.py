"""
LEVEL 3.3: USTAR FILTERING
============================

Composable callables that flag low-turbulence periods using USTAR thresholds
and compute per-scenario QCFs.

Entry points:

* ``run_level33_constant_ustar`` — apply pre-determined constant threshold(s).
* ``run_level33_variable_ustar`` — apply time-varying (e.g. per-year, VUT)
  threshold Series, one per scenario.
* ``run_level33_ustar_detection`` — detect the threshold from the data
  (``detection_mode=``: ONEFlux, REddyProc, ...), then apply it as CUT
  (constant) or VUT (per-year), selected with ``mode=``.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from diive.core.utils.console import console as _console, info, rule
from diive.flux.postprocessingchain.container import FluxLevelData
from diive.flux.postprocessingchain.levels._qcf import finalize_level
from diive.flux.postprocessingchain.levels._rerun import (
    cascade_reset,
    record_added_columns,
)
from diive.flux.lowres.ustarthreshold import (
    FlagMultipleConstantUstarThresholds,
    FlagMultipleVariableUstarThresholds,
)

if TYPE_CHECKING:
    from diive.flux.lowres.ustar_detection import UstarThresholdDetection



def _swin_for_nighttime_only(data: FluxLevelData, nighttime_only: bool,
                             swin_col: str | None):
    """SW_IN series for nighttime-only u* filtering, or None when day and night are filtered."""
    if not nighttime_only:
        return None
    if not swin_col:
        raise ValueError("nighttime_only=True needs swin_col (incoming shortwave radiation).")
    if swin_col not in data.full_df.columns:
        raise ValueError(f"swin_col {swin_col!r} not found in data.full_df.")
    return data.full_df[swin_col].reindex(data.fpc_df.index)


def run_level33_constant_ustar(
        data: FluxLevelData,
        *,
        thresholds: list[float],
        threshold_labels: list[str] | None = None,
        nighttime_only: bool = False,
        swin_col: str | None = None,
        showplot: bool = True,
        verbose: bool = True,
) -> FluxLevelData:
    """
    Level-3.3: Flag low-turbulence periods using one or more constant USTAR thresholds.

    Requires Level-3.1 (``data.levels.flux_corrected_col``) and Level-3.2
    (``data.levels.filteredseries_level32_qcf``) to have been run.

    .. important::

        **Constant thresholds only.** This function accepts one or more fixed
        USTAR values.  The community standard since Papale et al. (2006) is to
        derive the threshold via a bootstrap analysis that accounts for
        inter-annual and seasonal variability.  Pass the percentiles of such
        an analysis (e.g. 16th, 50th, 84th, from REddyProc or a previous run)
        as ``thresholds`` here, or use :func:`run_level33_ustar_detection` to
        run the bootstrap in the chain.

        **USTAR filtering applies only to CO2, CH4, and N2O fluxes.**
        Do *not* apply it to energy fluxes (H, LE) — those are not subject
        to the USTAR-based turbulence screening described in FLUXNET/Swiss
        FluxNet protocols.

    Args:
        data: FluxLevelData after ``run_level32()``.
        thresholds: List of USTAR threshold values (m s-1), one per scenario.
            Typical values are the 16th, 50th, and 84th percentiles from a
            bootstrap USTAR threshold analysis (e.g. 0.10, 0.18, 0.25).
        threshold_labels: Short label for each threshold scenario.  These
            labels become dict keys in ``levels.level33_qcf`` and
            ``levels.filteredseries_level33_qcf``, and are embedded in all
            downstream column names.  Conventional names follow the pattern
            ``'CUT_16'``, ``'CUT_50'``, ``'CUT_84'`` (percentile of the
            bootstrap distribution), but any unique strings work.

            Optional — if omitted, labels default to ``['CUT_0', 'CUT_1', ...]``
            (positional index, **not** percentile). For percentile-based
            thresholds always pass explicit labels so the provenance is
            preserved in column names and result keys.
        nighttime_only: If True, only nighttime records (``swin_col`` below
            10 W m-2, or missing) are filtered, as REddyProc does by default.
            Defaults to False: day and night are filtered, as in ONEFlux.
        swin_col: Incoming shortwave radiation column (W m-2) in
            ``data.full_df``. Needed for ``nighttime_only``.
        showplot: Show diagnostic plots. Defaults to True.
        verbose: Print progress. Defaults to True.

    Returns:
        Updated FluxLevelData with ``levels.level33``, ``levels.level33_qcf``
        (dict keyed by scenario), and ``levels.filteredseries_level33_qcf``
        (dict keyed by scenario) populated.

        ``data.filteredseries`` is set to ``None`` after this call because
        there is no single unambiguous filtered series when multiple USTAR
        scenarios exist.  Always access per-scenario series explicitly::

            data.levels.filteredseries_level33_qcf['CUT_50']

    Note:
        **Level-3.2 is required** before calling this function — running
        USTAR filtering on outlier-contaminated data biases the threshold's
        effect: a spike at low USTAR can artificially suppress nighttime
        fluxes and inflate the apparent threshold. ``run_level33_*`` raises
        a ``RuntimeError`` if ``data.levels.level32_qcf`` is ``None``. If
        you have a deliberate reason to skip L3.2 (e.g. manual screening
        upstream), you would need to bypass the guard by constructing the
        downstream state yourself, which is intentionally awkward.

        **H and LE (energy fluxes):** USTAR filtering must not be applied
        to these variables.  To keep all records, pass
        ``thresholds=[0], threshold_labels=['CUT_NONE']``. A threshold of 0
        flags nothing, not even records without a USTAR value, and satisfies
        Level-4.1's ordering requirement.
    """
    if data.levels.flux_corrected_col is None:
        raise RuntimeError("run_level31() must be called before run_level33_constant_ustar().")
    if data.levels.level32_qcf is None:
        raise RuntimeError(
            "run_level32() must be called before run_level33_constant_ustar(). "
            "USTAR filtering must operate on outlier-screened data — otherwise "
            "an outlier at low USTAR can bias the threshold's effect and "
            "spuriously suppress nighttime flux. If you have screened "
            "outliers upstream and genuinely want to skip L3.2, call the "
            "composable per-level API and bypass this guard explicitly."
        )

    # Auto-generate positional labels when none supplied. We intentionally use
    # index labels (CUT_0, CUT_1, ...) rather than something that looks like a
    # percentile, so callers don't mistake an auto-label for a CUT_50-style
    # percentile annotation.
    if threshold_labels is None:
        threshold_labels = [f'CUT_{i}' for i in range(len(thresholds))]

    # USTAR filtering is defined only for scalar fluxes (CO2, CH4, N2O).
    # Applying a non-zero threshold to energy fluxes (H, LE, ET, FH2O) would
    # silently drop nighttime records based on a quantity that has no physical
    # interpretation for those fluxes. The documented escape hatch for keeping
    # the chain's level ordering is to pass thresholds=[0], which flags nothing
    # (not even missing USTAR). We therefore only reject *positive* thresholds.
    # Case-fold both sides because detect_fluxbasevar() returns uppercase for
    # FluxNet-output files and lowercase for full-output files.
    _energy_basevars_lower = {'h2o', 't_sonic', 'sonic_temperature'}
    if (str(data.meta.fluxbasevar).lower() in _energy_basevars_lower
            and any(t > 0 for t in thresholds)):
        raise ValueError(
            f"USTAR filtering with a non-zero threshold is not valid for energy "
            f"fluxes (got fluxcol={data.meta.fluxcol!r}, basevar="
            f"{data.meta.fluxbasevar!r}). USTAR filtering applies only to CO2, "
            f"CH4 and N2O fluxes. To satisfy the chain's L3.3 ordering "
            f"requirement for H/LE, call with thresholds=[0], "
            f"threshold_labels=['CUT_NONE']."
        )

    if len(thresholds) != len(threshold_labels):
        raise ValueError(
            f"thresholds and threshold_labels must have the same length; "
            f"got {len(thresholds)} thresholds and {len(threshold_labels)} labels."
        )
    _validate_ustar_labels(threshold_labels)
    swin = _swin_for_nighttime_only(data, nighttime_only, swin_col)

    idstr = 'L3.3'
    meta = data.meta
    flux_corrected_col = data.levels.flux_corrected_col

    # Re-run cleanup: drop columns from any previous L3.3 invocation (which
    # may have used different USTAR labels), and clear L4.1/L4.2 downstream state.
    if idstr in data.level_ids:
        data = cascade_reset(data, idstr)
    pre_columns = list(data.fpc_df.columns)

    rule("Level 3.3: USTAR Filtering")

    level33 = FlagMultipleConstantUstarThresholds(
        series=data.fpc_df[flux_corrected_col],
        ustar=data.fpc_df[meta.ustarcol],
        thresholds=thresholds,
        threshold_labels=threshold_labels,
        idstr=idstr,
        showplot=showplot,
        swin=swin,
        nighttime_only=nighttime_only,
    )
    level33.calc()

    return _finalize_ustar_level(data, level33, threshold_labels,
                                 flux_corrected_col=flux_corrected_col, meta=meta,
                                 idstr=idstr, pre_columns=pre_columns, verbose=verbose)


def _validate_ustar_labels(labels: list[str]) -> None:
    """Reject duplicate or substring-overlapping scenario labels (flag-lookup safety)."""
    if len(set(labels)) != len(labels):
        raise ValueError(f"USTAR scenario labels must be unique; got {labels}.")
    # Substring-overlap check: a label that contains another label as a substring
    # would silently match the wrong flag column at lookup time (e.g. 'CUT_5' is
    # contained in 'FLAG_..._CUT_50_..._TEST'). Reject up front so the user can
    # pick distinct labels (e.g. 'CUT_05' / 'CUT_50') before any work happens.
    for a in labels:
        for b in labels:
            if a != b and a in b:
                raise ValueError(
                    f"USTAR scenario labels overlap by substring: {a!r} is contained "
                    f"in {b!r}. This would cause flag-column lookup to match the "
                    f"wrong scenario. Use distinct labels (e.g. {a!r} -> "
                    f"{a + '_X'!r} or zero-pad like 'CUT_05' / 'CUT_50')."
                )


def _finalize_ustar_level(data, level33, scenario_labels, *, flux_corrected_col, meta,
                          idstr, pre_columns, verbose):
    """Per-scenario QCF + container update, shared by the constant and variable paths."""
    level33_qcf: dict = {}
    filteredseries_level33_qcf: dict = {}
    filteredseries_level33_hq: dict = {}
    current = data

    for ustar_scen in scenario_labels:
        # Match the label only at underscore-delimited boundaries so labels that
        # share a prefix (or appear as substrings of other columns) cannot be
        # confused. We also require the column to be a flag column
        # (`FLAG_..._TEST`) so a user-supplied non-flag column that happens to
        # include the scenario label as a substring (e.g. ``'CUT_50_aux'``)
        # never enters the candidate set.
        token = f"_{ustar_scen}_"
        flagcols = [c for c in level33.results
                    if (str(c).startswith('FLAG_') and str(c).endswith('_TEST'))
                    and (token in c
                         or c.endswith(f"_{ustar_scen}_TEST")
                         or c.startswith(f"FLAG_{ustar_scen}_"))]
        if len(flagcols) != 1:
            raise RuntimeError(
                f"Could not uniquely identify the USTAR flag column for scenario "
                f"{ustar_scen!r}: found {len(flagcols)} matching columns "
                f"({flagcols}). Scenario labels must be unique and must not "
                f"appear as substrings of each other or of unrelated column names."
            )
        flagcol = flagcols[0]
        udf = level33.results[[flux_corrected_col, meta.ustarcol, flagcol]].copy()

        # No outname override here (unlike L3.1): run_qcf_on_col is the
        # L3.1-corrected flux (e.g. 'NEE_L3.1'), and the storage-correction tag
        # is meaningful provenance that must be preserved. Appending idstr
        # 'L3.3_<scen>' yields 'NEE_L3.1_L3.3_CUT_50_QCF' — storage-corrected
        # (L3.1) then USTAR-filtered (L3.3) — matching L3.2's 'NEE_L3.1_L3.2_QCF'.
        current, qcf = finalize_level(
            current,
            run_qcf_on_col=flux_corrected_col,
            idstr=f'L3.3_{ustar_scen}',
            level_df=udf,
            ustar_scenarios=scenario_labels,
        )
        level33_qcf[ustar_scen] = qcf
        filteredseries_level33_qcf[ustar_scen] = current.filteredseries.copy()
        filteredseries_level33_hq[ustar_scen] = qcf.filteredseries_hq.copy()
        info(f"QCF computed for USTAR scenario {ustar_scen}", verbose=verbose)

    new_levels = replace(
        current.levels,
        level33=level33,
        level33_qcf=level33_qcf,
        filteredseries_level33_qcf=filteredseries_level33_qcf,
        filteredseries_level33_hq=filteredseries_level33_hq,
    )
    level_ids = list(current.level_ids)
    if idstr not in level_ids:
        level_ids.append(idstr)

    # Set filteredseries to None: with multiple USTAR scenarios there is no single
    # unambiguous "the" filtered series. Force users to access the scenario dict
    # explicitly via data.levels.filteredseries_level33_qcf['CUT_50'].
    final = replace(current, levels=new_levels, level_ids=level_ids, filteredseries=None)
    return replace(final, added_columns=record_added_columns(final, idstr, pre_columns))


def run_level33_variable_ustar(
        data: FluxLevelData,
        *,
        threshold_series: dict,
        nighttime_only: bool = False,
        swin_col: str | None = None,
        showplot: bool = True,
        verbose: bool = True,
) -> FluxLevelData:
    """
    Level-3.3: Flag low-turbulence periods using time-varying USTAR thresholds (e.g. VUT).

    Variable-threshold counterpart of :func:`run_level33_constant_ustar`: each scenario
    carries a full per-record threshold Series instead of one scalar, so a different
    threshold can apply to different periods. This is what the FLUXNET/ONEFlux **VUT**
    (Variable U\\* Threshold) approach needs — each year filtered by its own threshold.

    Requires Level-3.1 and Level-3.2 (same as :func:`run_level33_constant_ustar`).

    Args:
        data: FluxLevelData after ``run_level32()``.
        threshold_series: mapping ``{scenario_label: per_record_threshold_Series}``. Each
            Series must align to the flux index and contain no NaN. Labels must be unique
            and not substrings of one another (e.g. ``VUT_16`` / ``VUT_50`` / ``VUT_84``).
        nighttime_only, swin_col: as in :func:`run_level33_constant_ustar`.
        showplot: show diagnostic plots. Defaults to True.
        verbose: print progress. Defaults to True.

    Returns:
        Updated FluxLevelData with ``levels.level33`` / ``level33_qcf`` /
        ``filteredseries_level33_qcf`` / ``filteredseries_level33_hq`` populated, one
        entry per scenario label.
    """
    if data.levels.flux_corrected_col is None:
        raise RuntimeError("run_level31() must be called before run_level33_variable_ustar().")
    if data.levels.level32_qcf is None:
        raise RuntimeError(
            "run_level32() must be called before run_level33_variable_ustar(). "
            "USTAR filtering must operate on outlier-screened data."
        )
    if not threshold_series:
        raise ValueError("threshold_series is empty — provide at least one scenario.")

    scenario_labels = list(threshold_series)
    _validate_ustar_labels(scenario_labels)
    swin = _swin_for_nighttime_only(data, nighttime_only, swin_col)

    # Energy fluxes (H/LE/...) have no physical USTAR threshold; reject positive ones.
    _energy_basevars_lower = {'h2o', 't_sonic', 'sonic_temperature'}
    if str(data.meta.fluxbasevar).lower() in _energy_basevars_lower:
        if any((s.fillna(0) > 0).any() for s in threshold_series.values()):
            raise ValueError(
                f"Variable USTAR filtering with positive thresholds is not valid for "
                f"energy fluxes (fluxcol={data.meta.fluxcol!r}, basevar="
                f"{data.meta.fluxbasevar!r}). Use run_level33_constant_ustar with "
                f"thresholds=[0] for the chain's L3.3 ordering requirement."
            )

    idstr = 'L3.3'
    meta = data.meta
    flux_corrected_col = data.levels.flux_corrected_col

    if idstr in data.level_ids:
        data = cascade_reset(data, idstr)
    pre_columns = list(data.fpc_df.columns)

    rule("Level 3.3: USTAR Filtering (variable thresholds)")

    level33 = FlagMultipleVariableUstarThresholds(
        series=data.fpc_df[flux_corrected_col],
        ustar=data.fpc_df[meta.ustarcol],
        threshold_series=threshold_series,
        idstr=idstr,
        showplot=showplot,
        verbose=verbose,
        swin=swin,
        nighttime_only=nighttime_only,
    )
    level33.calc()

    return _finalize_ustar_level(data, level33, scenario_labels,
                                 flux_corrected_col=flux_corrected_col, meta=meta,
                                 idstr=idstr, pre_columns=pre_columns, verbose=verbose)


def run_level33_ustar_detection(
        data: FluxLevelData,
        *,
        ta_col: str,
        swin_col: str,
        detection_mode: str = 'oneflux',
        detection_kwargs: dict | None = None,
        n_iter: int | None = None,
        n_jobs: int = 1,
        percentiles: tuple[int, ...] = (16, 50, 84),
        mode: str = 'cut',
        nighttime_only: bool = False,
        detector_class=None,
        detector_kwargs: dict | None = None,
        showplot: bool = True,
        verbose: bool = True,
) -> FluxLevelData:
    """
    Level-3.3: Detect the USTAR threshold from the data, then apply it (CUT or VUT).

    Runs :class:`~diive.flux.lowres.ustar_detection.UstarThresholdDetection` and
    applies the result as USTAR filtering scenarios, one per requested percentile.
    ``detection_mode`` selects the method:

    * ``'oneflux'`` (default) — ONEFlux (FLUXNET2015, ICOS): moving point and
      change point thresholds per year, pooled.
    * ``'oneflux_mp'`` / ``'oneflux_cp'`` — ONEFlux with one of the two methods only.
    * ``'reddyproc'`` — REddyProc 1.3.4.

    Which thresholds are applied is set by ``mode``:

    * ``'cut'`` (default) — **CUT** (Constant U\\* Threshold): one constant threshold
      per percentile from the thresholds of all years, applied via
      ``run_level33_constant_ustar``. Scenarios labelled ``CUT_<p>``.
    * ``'vut'`` — **VUT** (Variable U\\* Threshold): a threshold per year and
      percentile, applied via ``run_level33_variable_ustar``. Scenarios labelled
      ``VUT_<p>``. A year with no threshold falls back to the CUT value.

    CUT and VUT are mutually exclusive filtering strategies — one is applied before
    gap-filling (the percentiles are the uncertainty scenarios within the chosen strategy).

    Requires Level-3.1 (``data.levels.flux_corrected_col``) and Level-3.2 to have
    been run: outliers at low turbulence would bias the bootstrap threshold.

    **When to use which function:**

    * ``run_level33_constant_ustar`` — you already have a threshold from an external
      analysis or a previous run. Fastest, no computation.
    * ``run_level33_ustar_detection`` — you want the threshold computed from the
      data in the same pipeline run. Slower (bootstrap) but reproducible.

    Args:
        data: FluxLevelData after ``run_level32()``.
        ta_col: Air temperature column name (deg C) in ``data.full_df``.
            Used to stratify nighttime records into temperature classes.
        swin_col: Incoming shortwave radiation column (W m-2) in ``data.full_df``.
            Defines the night for the detection (and for ``nighttime_only``).
        detection_mode: ``'oneflux'`` (default), ``'oneflux_mp'``, ``'oneflux_cp'``
            or ``'reddyproc'``, see above.
        detection_kwargs: Extra keyword arguments for ``UstarThresholdDetection``
            (e.g. ``random_state``, ``mp_kwargs``). The column names, ``mode``,
            ``n_iter``, ``n_jobs`` and ``percentiles`` are set from the arguments
            of this function and must not be included here.
        n_iter: Bootstrap draws per year (and method). Defaults to the reference
            value of the mode (100 for ONEFlux, 200 for REddyProc).
        n_jobs: Parallel workers (1 = sequential, -1 = all CPUs). Defaults to 1.
        percentiles: Bootstrap percentiles used as separate USTAR scenarios. Each
            value ``p`` produces one scenario labelled ``CUT_<p>`` (``VUT_<p>``
            with ``mode='vut'``), zero-padded to a uniform width so labels can
            never be substrings of each other (e.g. ``(5, 50, 95)`` ->
            ``CUT_05`` / ``CUT_50`` / ``CUT_95``). Defaults to ``(16, 50, 84)``.
        mode: ``'cut'`` (default) or ``'vut'``, see above.
        nighttime_only: If True, only nighttime records are filtered (as
            REddyProc). Defaults to False (day and night, as ONEFlux).
        detector_class: Deprecated. ``UstarMovingPointDetection`` maps to
            ``detection_mode='oneflux_mp'``.
        detector_kwargs: Deprecated. With ``detector_class``, the arguments of the
            moving point detection (passed on as ``mp_kwargs``).
        showplot: Show diagnostic plots from USTAR filtering. Defaults to True.
        verbose: Print progress and detection summary. Defaults to True.

    Returns:
        Updated FluxLevelData with ``levels.level33``, ``levels.level33_qcf``,
        ``levels.filteredseries_level33_qcf``, ``levels.filteredseries_level33_hq``,
        and ``levels.ustar_detection`` populated. ``ustar_detection`` holds the
        fitted ``UstarThresholdDetection``, so you can inspect the per-year
        thresholds and the bootstrap values afterwards::

            print(data.levels.ustar_detection.summary())
            vut = data.levels.ustar_detection.get_vut_thresholds()

    Raises:
        RuntimeError: If ``run_level31()`` or ``run_level32()`` has not been
            called, or if threshold detection produces NaN for any requested
            percentile (e.g. too little nighttime data).
        ValueError: If ``mode`` is not ``'cut'`` or ``'vut'``, or
            ``detection_kwargs`` holds a key set by this function.
    """
    import warnings

    from diive.flux.lowres.ustar_detection import UstarThresholdDetection

    if data.levels.flux_corrected_col is None:
        raise RuntimeError("run_level31() must be called before run_level33_ustar_detection().")
    if data.levels.level32_qcf is None:
        raise RuntimeError(
            "run_level32() must be called before run_level33_ustar_detection(). "
            "Outliers at low USTAR bias the bootstrap threshold estimate — "
            "screen them out at L3.2 first."
        )
    mode = str(mode).lower()
    if mode not in ('cut', 'vut'):
        raise ValueError(f"mode must be 'cut' or 'vut', got {mode!r}.")

    det_kwargs = dict(detection_kwargs or {})
    _reserved = ('df', 'nee_col', 'ta_col', 'ustar_col', 'swin_col', 'mode', 'n_iter',
                 'n_jobs', 'percentiles')
    conflicts = sorted(set(_reserved).intersection(det_kwargs))
    if conflicts:
        raise ValueError(
            f"detection_kwargs contains key(s) {conflicts} that are set from the "
            f"arguments of run_level33_ustar_detection. Remove them from detection_kwargs."
        )
    if detector_class is not None or detector_kwargs is not None:
        from diive.flux.lowres.ustar_mp_detection import UstarMovingPointDetection
        if detector_class not in (None, UstarMovingPointDetection):
            raise ValueError(
                f"detector_class={getattr(detector_class, '__name__', detector_class)} is no "
                f"longer supported. Use detection_mode='oneflux', 'oneflux_mp', "
                f"'oneflux_cp' or 'reddyproc'.")
        warnings.warn(
            "detector_class and detector_kwargs are deprecated and will be removed. Use "
            "detection_mode='oneflux_mp' and detection_kwargs={'mp_kwargs': {...}}.",
            FutureWarning, stacklevel=2)
        mp_kw = {k: v for k, v in (detector_kwargs or {}).items()
                 if k not in ('nee_col', 'ta_col', 'ustar_col', 'swin_col', 'verbose')}
        if len(mp_kw) != len({k: v for k, v in (detector_kwargs or {}).items() if k != 'verbose'}):
            raise ValueError(
                "detector_kwargs contains reserved column-name key(s). These are set "
                "automatically from ta_col, swin_col, data.meta.ustarcol and the "
                "storage-corrected flux column.")
        detection_mode = 'oneflux_mp'
        det_kwargs.setdefault('mp_kwargs', mp_kw)

    meta = data.meta
    flux_corrected_col = data.levels.flux_corrected_col

    # Detection input: storage-corrected flux after L3.2 + the met drivers
    det_df = data.full_df[[ta_col, swin_col, meta.ustarcol]].copy()
    det_df[flux_corrected_col] = data.fpc_df[flux_corrected_col]

    rule("Level 3.3: USTAR Threshold Detection", verbose=verbose)
    info(f"UstarThresholdDetection mode={detection_mode!r}  n_jobs={n_jobs}",
         verbose=verbose)

    det: UstarThresholdDetection = UstarThresholdDetection(
        df=det_df,
        nee_col=flux_corrected_col,
        ta_col=ta_col,
        ustar_col=meta.ustarcol,
        swin_col=swin_col,
        mode=detection_mode,
        n_iter=n_iter,
        n_jobs=n_jobs,
        percentiles=percentiles,
        verbose=verbose,
        **det_kwargs,
    )
    det.run()
    cut = det.get_cut_threshold()
    boot = det

    if verbose:
        _console.print(det.summary())

    # Extract the constant (CUT) threshold per percentile; raise if any are NaN.
    cut_thresholds = []
    for p in percentiles:
        thr = cut.get(f'p{p:g}', np.nan)
        if np.isnan(thr):
            raise RuntimeError(
                f"USTAR threshold detection produced NaN for percentile {p}. "
                "Check that the dataset has enough valid nighttime records "
                "(see data.levels.ustar_detection.failures_ for the reasons per year)."
            )
        cut_thresholds.append(float(thr))
    # Zero-pad to a uniform width so labels can never be substrings of one
    # another (two distinct equal-length strings can't contain each other).
    # Without this, e.g. percentiles=(5, 50, 95) would generate 'CUT_5' /
    # 'CUT_50' / 'CUT_95' and trip the substring-overlap guard ('CUT_5' in
    # 'CUT_50') — crashing *after* the expensive bootstrap had already run.
    _label_width = max(2, max(len(str(int(p))) for p in percentiles))

    if mode == 'cut':
        labels = [f'CUT_{int(p):0{_label_width}d}' for p in percentiles]
        updated = run_level33_constant_ustar(
            data, thresholds=cut_thresholds, threshold_labels=labels,
            nighttime_only=nighttime_only, swin_col=swin_col,
            showplot=showplot, verbose=verbose)
    else:  # vut
        # Build per-record (VUT) threshold Series per percentile from the per-year
        # table, mapping each timestamp's year to that year's threshold. Years with
        # no detected threshold (NaN) fall back to the pooled CUT value.
        flux_corrected_col = data.levels.flux_corrected_col
        index = data.fpc_df[flux_corrected_col].index
        year_of = index.year
        vut = boot.get_vut_thresholds()
        threshold_series: dict = {}
        for p, cut_val in zip(percentiles, cut_thresholds, strict=False):
            col = f'p{p:g}'
            per_year = {int(y): (vut.loc[y, col] if (y in vut.index
                        and np.isfinite(vut.loc[y, col])) else cut_val)
                        for y in vut.index}
            values = np.array([per_year.get(int(y), cut_val) for y in year_of], dtype=float)
            threshold_series[f'VUT_{int(p):0{_label_width}d}'] = pd.Series(values, index=index)
        updated = run_level33_variable_ustar(
            data, threshold_series=threshold_series,
            nighttime_only=nighttime_only, swin_col=swin_col,
            showplot=showplot, verbose=verbose)

    # Attach the detection instance to levels for post-hoc inspection
    new_levels = replace(updated.levels, ustar_detection=boot)
    return replace(updated, levels=new_levels)
