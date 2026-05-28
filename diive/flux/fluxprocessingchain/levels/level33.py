"""
LEVEL 3.3: USTAR FILTERING
============================

Composable callables that flag low-turbulence periods using one or more
constant USTAR thresholds and compute per-scenario QCFs.

Two entry points:

* ``run_level33_constant_ustar`` — apply pre-determined constant threshold(s).
* ``run_level33_ustar_detection`` — auto-detect threshold via bootstrap, then
  apply the detected CUT percentile thresholds as constant scenarios.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from diive.core.utils.console import console as _console, info, rule
from diive.flux.fluxprocessingchain.container import FluxLevelData
from diive.flux.fluxprocessingchain.levels._qcf import finalize_level
from diive.flux.fluxprocessingchain.levels._rerun import (
    cascade_reset,
    record_added_columns,
)
from diive.flux.lowres.ustarthreshold import FlagMultipleConstantUstarThresholds

if TYPE_CHECKING:
    from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds


def run_level33_constant_ustar(
        data: FluxLevelData,
        *,
        thresholds: list[float],
        threshold_labels: list[str] | None = None,
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
        inter-annual and seasonal variability (e.g. using the R package
        REddyProc).  Run that analysis externally, then pass its output
        percentiles (e.g. 16th, 50th, 84th) as ``thresholds`` here.

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
        **Level-3.2 is recommended but not required** before calling this
        function.  Running USTAR filtering on data that still contains
        outliers may bias the threshold estimate: a spike at low USTAR
        could artificially suppress nighttime fluxes, leading to an
        over-estimated threshold.  Run ``run_level32()`` first unless you
        have a specific reason to skip outlier removal.

        **H and LE (energy fluxes):** USTAR filtering must not be applied
        to these variables.  To keep all records, pass
        ``thresholds=[0], threshold_labels=['CUT_NONE']`` — this flags
        nothing (USTAR is always ≥ 0) and satisfies Level-4.1's ordering
        requirement.
    """
    if data.levels.flux_corrected_col is None:
        raise RuntimeError("run_level31() must be called before run_level33_constant_ustar().")

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
    # (USTAR is always >= 0); we therefore only reject *positive* thresholds.
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
    if len(set(threshold_labels)) != len(threshold_labels):
        raise ValueError(
            f"threshold_labels must be unique; got {threshold_labels}."
        )
    # Substring-overlap check: a label that contains another label as a substring
    # would silently match the wrong flag column at lookup time (e.g. 'CUT_5' is
    # contained in 'FLAG_..._CUT_50_..._TEST'). Reject up front so the user can
    # pick distinct labels (e.g. 'CUT_05' / 'CUT_50') before any work happens.
    for a in threshold_labels:
        for b in threshold_labels:
            if a != b and a in b:
                raise ValueError(
                    f"threshold_labels overlap by substring: {a!r} is contained "
                    f"in {b!r}. This would cause flag-column lookup to match the "
                    f"wrong scenario. Use distinct labels (e.g. {a!r} -> "
                    f"{a + '_X'!r} or zero-pad like 'CUT_05' / 'CUT_50')."
                )

    idstr = 'L3.3'
    meta = data.meta
    flux_corrected_col = data.levels.flux_corrected_col

    # Re-run cleanup: drop columns from any previous L3.3 invocation (which
    # may have used different USTAR labels), and clear L4.1 downstream state.
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
    )
    level33.calc()

    level33_qcf: dict = {}
    filteredseries_level33_qcf: dict = {}
    filteredseries_level33_hq: dict = {}
    current = data

    for ustar_scen in threshold_labels:
        # Match the label only at underscore-delimited boundaries so labels that
        # share a prefix (or appear as substrings of other columns) cannot be
        # confused. Belt-and-braces with the substring-overlap check above.
        token = f"_{ustar_scen}_"
        flagcols = [c for c in level33.results
                    if token in c
                    or c.endswith(f"_{ustar_scen}")
                    or c.startswith(f"{ustar_scen}_")
                    or c == ustar_scen]
        if len(flagcols) != 1:
            raise RuntimeError(
                f"Could not uniquely identify the USTAR flag column for scenario "
                f"{ustar_scen!r}: found {len(flagcols)} matching columns "
                f"({flagcols}). Threshold labels must be unique and must not "
                f"appear as substrings of each other or of unrelated column names."
            )
        flagcol = flagcols[0]
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


def run_level33_ustar_detection(
        data: FluxLevelData,
        *,
        ta_col: str,
        swin_col: str,
        detector_class=None,
        detector_kwargs: dict | None = None,
        n_iter: int = 100,
        n_jobs: int = 1,
        percentiles: tuple[int, ...] = (16, 50, 84),
        showplot: bool = True,
        verbose: bool = True,
) -> FluxLevelData:
    """
    Level-3.3: Auto-detect the USTAR threshold, then apply it as constant scenario(s).

    Runs a multi-year bootstrap USTAR threshold detection (using the ONEFlux moving
    point method by default) and converts the CUT (constant upper threshold) percentile
    results into one USTAR filtering scenario per requested percentile.  Each scenario
    is then passed to ``run_level33_constant_ustar()`` so that per-scenario QCFs and
    filtered series are produced in exactly the same way as when a pre-known threshold
    is supplied.

    Requires Level-3.1 (``data.levels.flux_corrected_col``) to have been run.
    Level-3.2 is strongly recommended before calling this function to avoid bias from
    outliers at low-turbulence conditions.

    **When to use which function:**

    * ``run_level33_constant_ustar`` — you already have a threshold from an external
      analysis (e.g. REddyProc) or a previous run.  Fastest; no computation overhead.
    * ``run_level33_ustar_detection`` — you want the threshold computed automatically
      from the data in the same pipeline run.  Slower (bootstrap) but fully reproducible.

    Args:
        data: FluxLevelData after ``run_level32()``.
        ta_col: Air temperature column name (deg C) in ``data.full_df``.
            Used to stratify nighttime records into temperature classes.
        swin_col: Incoming shortwave radiation column (W m-2) in ``data.full_df``.
            Used to identify nighttime periods (SW_IN < 10 W m-2).
        detector_class: USTAR detection class to use.  Must implement ``detect()``
            and ``get_annual_thresholds()``.  Defaults to
            ``UstarMovingPointDetection`` (ONEFlux algorithm, Papale et al. 2006).
        detector_kwargs: Extra keyword arguments forwarded to the detector constructor
            (e.g. ``ta_classes_count``, ``ustar_classes_count``).  The column-name
            arguments (``nee_col``, ``ta_col``, ``ustar_col``, ``swin_col``) are set
            automatically and must not be included here.
        n_iter: Bootstrap iterations per year window. Defaults to 100.
        n_jobs: Parallel workers for the bootstrap (1 = sequential, -1 = all CPUs).
            Defaults to 1.
        percentiles: Bootstrap percentiles to compute and use as separate USTAR
            scenarios.  Each value ``p`` produces one scenario labelled ``CUT_p``
            (e.g. ``(16, 50, 84)`` -> scenarios ``CUT_16``, ``CUT_50``, ``CUT_84``).
            Defaults to ``(16, 50, 84)``.
        showplot: Show diagnostic plots from USTAR filtering. Defaults to True.
        verbose: Print progress and detection summary. Defaults to True.

    Returns:
        Updated FluxLevelData with ``levels.level33``, ``levels.level33_qcf``,
        ``levels.filteredseries_level33_qcf``, ``levels.filteredseries_level33_hq``,
        and ``levels.ustar_detection`` populated.  The ``ustar_detection`` attribute
        holds the fitted ``UstarBootstrapThresholds`` instance so you can inspect
        annual per-year thresholds and the full bootstrap summary afterwards::

            print(data.levels.ustar_detection.summary())
            annual = data.levels.ustar_detection.annual_stats_

    Raises:
        RuntimeError: If ``run_level31()`` has not been called, or if threshold
            detection produces NaN for any requested percentile (e.g. too little
            nighttime data).
    """
    from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
    from diive.flux.lowres.ustar_mp_detection import UstarMovingPointDetection

    if data.levels.flux_corrected_col is None:
        raise RuntimeError("run_level31() must be called before run_level33_ustar_detection().")

    if detector_class is None:
        detector_class = UstarMovingPointDetection

    meta = data.meta
    flux_corrected_col = data.levels.flux_corrected_col

    # Assemble detection DataFrame: storage-corrected flux + required met drivers
    det_df = data.full_df[[ta_col, swin_col, meta.ustarcol]].copy()
    det_df[flux_corrected_col] = data.fpc_df[flux_corrected_col]

    # Column-name kwargs are set here from the explicit ta_col / swin_col /
    # data.meta.ustarcol arguments; allowing detector_kwargs to also supply
    # them would silently override one of two conflicting truths. Raise so the
    # user removes the duplicate rather than guessing which one wins.
    _reserved = ('nee_col', 'ta_col', 'ustar_col', 'swin_col')
    conflicts = sorted(set(_reserved).intersection(detector_kwargs or {}))
    if conflicts:
        raise ValueError(
            f"detector_kwargs contains reserved column-name key(s) {conflicts}. "
            f"These are set automatically from the explicit arguments "
            f"(ta_col, swin_col, data.meta.ustarcol) and the storage-corrected "
            f"flux column. Remove them from detector_kwargs."
        )
    kw = dict(detector_kwargs or {})
    kw['nee_col'] = flux_corrected_col
    kw['ta_col'] = ta_col
    kw['ustar_col'] = meta.ustarcol
    kw['swin_col'] = swin_col

    rule("Level 3.3: USTAR Threshold Detection", verbose=verbose)
    info(f"{detector_class.__name__}  {n_iter} iterations  n_jobs={n_jobs}",
         verbose=verbose)

    boot: UstarBootstrapThresholds = UstarBootstrapThresholds(
        df=det_df,
        detector_class=detector_class,
        detector_kwargs=kw,
        n_iter=n_iter,
        n_jobs=n_jobs,
        percentiles=percentiles,
        verbose=int(verbose),
    )
    boot.run()
    cut = boot.get_cut_threshold()

    if verbose:
        _console.print(boot.summary())

    # Extract threshold per percentile; raise if any are NaN
    thresholds = []
    for p in percentiles:
        thr = cut.get(f'p{p}', np.nan)
        if np.isnan(thr):
            raise RuntimeError(
                f"USTAR threshold detection produced NaN for percentile {p}. "
                "Check that the dataset has sufficient nighttime records "
                "(>= 3000 total, >= 160 per season)."
            )
        thresholds.append(float(thr))
    threshold_labels = [f'CUT_{p}' for p in percentiles]

    # Apply detected thresholds as constant USTAR scenarios
    updated = run_level33_constant_ustar(
        data,
        thresholds=thresholds,
        threshold_labels=threshold_labels,
        showplot=showplot,
        verbose=verbose,
    )

    # Attach the bootstrap instance to levels for post-hoc inspection
    new_levels = replace(updated.levels, ustar_detection=boot)
    return replace(updated, levels=new_levels)
