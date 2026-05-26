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

from diive.pkgs.flux.fluxprocessingchain.container import FluxLevelData
from diive.pkgs.flux.fluxprocessingchain.levels._qcf import finalize_level
from diive.pkgs.flux.lowres.ustarthreshold import FlagMultipleConstantUstarThresholds

if TYPE_CHECKING:
    from diive.pkgs.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds


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
    filteredseries_level33_hq: dict = {}
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
        filteredseries_level33_hq[ustar_scen] = qcf.filteredseries_hq.copy()
        if verbose:
            print(f"++ Calculated overall quality flag QCF for USTAR scenario {ustar_scen}.")

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
    return replace(current, levels=new_levels, level_ids=level_ids, filteredseries=None)


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
    from diive.pkgs.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
    from diive.pkgs.flux.lowres.ustar_mp_detection import UstarMovingPointDetection

    if data.levels.flux_corrected_col is None:
        raise RuntimeError("run_level31() must be called before run_level33_ustar_detection().")

    if detector_class is None:
        detector_class = UstarMovingPointDetection

    meta = data.meta
    flux_corrected_col = data.levels.flux_corrected_col

    # Assemble detection DataFrame: storage-corrected flux + required met drivers
    det_df = data.full_df[[ta_col, swin_col, meta.ustarcol]].copy()
    det_df[flux_corrected_col] = data.fpc_df[flux_corrected_col]

    # Build detector kwargs — column names are set here; user must not duplicate them
    kw = {k: v for k, v in (detector_kwargs or {}).items()
          if k not in ('nee_col', 'ta_col', 'ustar_col', 'swin_col')}
    kw['nee_col'] = flux_corrected_col
    kw['ta_col'] = ta_col
    kw['ustar_col'] = meta.ustarcol
    kw['swin_col'] = swin_col

    if verbose:
        print(f"\nL3.3 USTAR threshold detection  ({detector_class.__name__}, "
              f"{n_iter} iterations, n_jobs={n_jobs})")

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
        print(boot.summary())

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
