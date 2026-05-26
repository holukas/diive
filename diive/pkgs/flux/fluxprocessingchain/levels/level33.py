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
