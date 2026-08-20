"""
REPORTS: FLUX PROCESSING CHAIN REPORTING
========================================

Console reports and summary plots over a finished flux processing chain.

Each function takes the :class:`FluxLevelData` container as its first argument,
matching the composable per-level callables (``run_level2(data, ...)``). The
reports read only the container's public surface, so they stay outside
``container.py`` and its typed-container role.

Everything here reads Level-4.1 results. Call after at least one
``run_level41_*`` has completed.

Part of the diive library: https://github.com/holukas/diive
"""
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from pandas import DataFrame

from diive.core.utils.console import console as _console, info, rule

# Long names for the short method keys used throughout the chain.
_METHOD_LABELS = {'mds': 'MDS', 'rf': 'Random Forest', 'xgb': 'XGBoost'}


def _iter_level41(data) -> Iterator[tuple[str, str, Any]]:
    """Yield ``(method, ustar_scenario, model_instance)`` for every L4.1 result."""
    for method, scenarios in data.levels.level41_methods().items():
        for scenario, inst in scenarios.items():
            yield method, scenario, inst


def _optional(inst, attr: str):
    """Return ``inst.attr``, or None when the method does not provide it.

    The long-term gap-fillers expose their results as properties that *raise*
    when nothing was collected, so ``hasattr`` is not enough: it swallows
    ``AttributeError`` only, and lets those raise through.
    """
    try:
        return getattr(inst, attr)
    except Exception:
        return None


def _label(method: str) -> str:
    return _METHOD_LABELS.get(method, method)


def _scores_frame(scores: dict) -> DataFrame:
    """Build a frame from a per-year scores dict, whichever way round it nests."""
    try:
        return DataFrame.from_dict(scores, orient='columns')
    except ValueError:
        return DataFrame.from_dict(scores, orient='index')


def _print_frame(df: DataFrame, *, verbose: int | bool | None = None) -> None:
    """Print a result frame through the shared console."""
    _console.print(df.to_string())


def _write_csv(df: DataFrame, outpath: str | None, filename: str) -> None:
    if not outpath:
        return
    target = Path(outpath) / filename
    df.to_csv(target)
    info(f"Saved {target}")


def _report_scores(data, attr: str, label: str, outfile_prefix: str,
                   outpath: str | None, verbose: int | bool | None) -> None:
    """Shared body of the three score reports."""
    for method, scenario, inst in _iter_level41(data):
        scores = _optional(inst, attr)
        if scores is None:
            info(f"{_label(method)} ({scenario}): no {label} available.", verbose=verbose)
            continue
        rule(f"{label.upper()}  —  {_label(method)} ({scenario})", verbose=verbose)
        df = _scores_frame(scores)
        _print_frame(df)
        _write_csv(df, outpath, f"{outfile_prefix}_{scenario}_{method}.csv")


def report_gapfilling_variables(data, verbose: int | bool | None = None) -> None:
    """Report which target column each L4.1 method gap-filled, and into which column.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        verbose: Verbosity for the console helpers.

    Example::

        report_gapfilling_variables(data)
        # Random Forest (CUT_50): NEE_L3.1_L3.3_CUT_50_QCF -> ..._QCF_gfRF
    """
    rule("GAP-FILLED VARIABLES", verbose=verbose)
    for method, scenario, inst in _iter_level41(data):
        gapfilled = data.gapfilled_cols()[method][scenario]
        info(f"{_label(method)} ({scenario}): {inst.target_col} -> {gapfilled}", verbose=verbose)


def report_traintest_model_scores(data, outpath: str | None = None,
                                  verbose: int | bool | None = None) -> None:
    """Report held-out test scores per year, per method and USTAR scenario.

    These come from a random train/test split of the complete rows, which is the
    split that reproduces the gap-filling task. MDS has no such split and is
    reported as unavailable.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        verbose: Verbosity for the console helpers.
    """
    _report_scores(data, 'scores_traintest_', 'train/test model scores',
                   'traintest_model_scores', outpath, verbose)


def report_traintest_details(data, outpath: str | None = None,
                             verbose: int | bool | None = None) -> None:
    """Report train/test split details per year (record counts, split sizes).

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        verbose: Verbosity for the console helpers.
    """
    _report_scores(data, 'traintest_details_', 'train/test details',
                   'traintest_model_details', outpath, verbose)


def report_gapfilling_model_scores(data, outpath: str | None = None,
                                   verbose: int | bool | None = None) -> None:
    """Report in-sample model scores per year, per method and USTAR scenario.

    In-sample scores are optimistically biased; compare against
    :func:`report_traintest_model_scores` for the held-out counterpart.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        verbose: Verbosity for the console helpers.
    """
    _report_scores(data, 'scores_', 'model scores',
                   'gapfilling_model_scores', outpath, verbose)


def report_gapfilling_feature_importances(data, outpath: str | None = None,
                                          verbose: int | bool | None = None) -> None:
    """Report per-year SHAP feature importances for the ML gap-fillers.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        verbose: Verbosity for the console helpers.
    """
    for method, scenario, inst in _iter_level41(data):
        importances = _optional(inst, 'feature_importance_per_year')
        if importances is None:
            info(f"{_label(method)} ({scenario}): no feature importances available.",
                 verbose=verbose)
            continue
        rule(f"FEATURE IMPORTANCES  —  {_label(method)} ({scenario})", verbose=verbose)
        _print_frame(importances)
        _write_csv(importances, outpath,
                   f"gapfilling_model_feature_importances_{scenario}_{method}.csv")


def report_gapfilling_poolyears(data, verbose: int | bool | None = None) -> None:
    """Report which years of data each year's model was trained on.

    The long-term gap-fillers pool neighbouring years to train the model for a
    given year; this shows that pool. MDS does not pool and is skipped.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        verbose: Verbosity for the console helpers.
    """
    rule("DATA POOLS USED FOR MACHINE-LEARNING GAP-FILLING", verbose=verbose)
    for method, scenario, inst in _iter_level41(data):
        yearpools = _optional(inst, 'yearpools')
        if yearpools is None:
            info(f"{_label(method)} ({scenario}): no year pools used.", verbose=verbose)
            continue
        gapfilled = data.gapfilled_cols()[method][scenario]
        for year, pool in yearpools.items():
            info(f"{year}: {_label(method)} ({scenario}) used data from "
                 f"{pool['poolyears']} for gap-filling {inst.target_col} -> {gapfilled}",
                 verbose=verbose)


def gapfilled_variables(data) -> DataFrame:
    """Return a copy of the gap-filled columns alongside their pre-fill targets.

    Args:
        data: Chain container after at least one ``run_level41_*``.

    Returns:
        DataFrame holding one gap-filled column and one measured target column
        per method and USTAR scenario.
    """
    gapfilled = [c for scen in data.gapfilled_cols().values() for c in scen.values()]
    targets = [c for scen in data.nongapfilled_cols().values() for c in scen.values()]
    # A target can be shared by several methods (all fill the same column), so
    # de-duplicate while keeping the order.
    cols = list(dict.fromkeys(gapfilled + targets))
    return data.fpc_df[cols].copy()


def plot_gapfilled_cumulative(data, gain: float = 1, units: str = "",
                              per_year: bool = True,
                              start_year: int | None = None,
                              end_year: int | None = None,
                              show_reference: bool = True,
                              excl_years_from_reference: list | None = None,
                              showplot: bool = True) -> None:
    """Plot cumulative sums of the gap-filled fluxes.

    Two views. ``per_year=True`` draws one figure per gap-filled variable, with
    each year as its own line and an optional multi-year reference band —
    the year-over-year view. ``per_year=False`` draws a single figure
    accumulating over the whole record, one line per gap-filled variable.

    For the method-comparison view on one axes (RF vs XGBoost vs MDS for a
    single scenario), use ``data.plot_cumulative_comparison()`` instead.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        gain: Multiply every flux value by this factor before accumulating.
            For 30-min NEE in µmol CO2 m-2 s-1 -> gC m-2, use
            ``12.011 * 1e-6 * 1800``.
        units: Unit string shown on the y-axis.
        per_year: Year-over-year view (True) or one cumulative over the whole
            record (False).
        start_year: First year to include. Defaults to the first in the record.
        end_year: Last year to include. Defaults to the last in the record.
        show_reference: Draw the multi-year reference band (``per_year`` only).
        excl_years_from_reference: Years to leave out of that reference.
        showplot: Call ``plt.show()`` after rendering. Set False for headless use.
    """
    from diive.core.plotting.cumulative import Cumulative, CumulativeYear

    names = [c for scen in data.gapfilled_cols().values() for c in scen.values()]
    if not names:
        raise RuntimeError("No gap-filled columns found. "
                           "Run at least one run_level41_*() function first.")
    df = data.fpc_df[names].copy()

    if per_year:
        for name in names:
            CumulativeYear(
                series=df[name].multiply(gain),
                series_units=units,
                yearly_end_date=None,
                start_year=start_year,
                end_year=end_year,
                show_reference=show_reference,
                excl_years_from_reference=excl_years_from_reference,
                highlight_year=None,
            ).plot(showplot=showplot)
    else:
        Cumulative(df=df.multiply(gain), units=units,
                   start_year=start_year, end_year=end_year).plot(showplot=showplot)


def plot_feature_ranks_per_year(data) -> None:
    """Plot per-year feature ranks for each ML gap-filler.

    Args:
        data: Chain container after at least one ``run_level41_*``.
    """
    for method, scenario, inst in _iter_level41(data):
        results_yearly = _optional(inst, 'results_yearly_')
        if results_yearly is None:
            info(f"{_label(method)} ({scenario}): no feature ranks available.")
            continue
        gapfilled = data.gapfilled_cols()[method][scenario]
        first_key = next(iter(results_yearly))
        model_params = results_yearly[first_key].model_.get_params()
        inst.showplot_feature_ranks_per_year(
            title=f"{gapfilled} ({scenario})",
            subtitle=f"MODEL: {_label(method)} / PARAMS: {model_params}")


def plot_mds_gapfilling_qualities(data) -> None:
    """Plot the MDS fill-quality overview for every USTAR scenario.

    Args:
        data: Chain container after ``run_level41_mds``.
    """
    mds = data.levels.level41_mds
    if not mds:
        info("MDS was not run, no fill qualities to plot.")
        return
    for inst in mds.values():
        inst.showplot()
