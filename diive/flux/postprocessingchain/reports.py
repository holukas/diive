"""
REPORTS: FLUX POST-PROCESSING CHAIN REPORTING
=============================================

Console reports and summary plots over a finished flux post-processing chain.

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
from rich import box
from rich.console import Group
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from diive.core.utils.console import VERBOSE_PROGRESS, _vlevel, console as _console, info

# Long names for the short method keys used throughout the chain.
_METHOD_LABELS = {'mds': 'MDS', 'rf': 'Random Forest', 'xgb': 'XGBoost'}

# The score columns shown in the console tables, in order. MSE repeats RMSE, and MAPE is
# meaningless for a flux that crosses zero, so both stay in the CSV output only.
_SCORE_COLUMNS = [('r2', 'R²', '{:.3f}'), ('rmse', 'RMSE', '{:.2f}'), ('mae', 'MAE', '{:.2f}'),
                  ('medae', 'MedAE', '{:.2f}'), ('maxe', 'MaxE', '{:.1f}')]


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


def _table() -> Table:
    """Table in the style of the QCF reports. Rich falls back to ASCII lines on a cp1252 stdout."""
    return Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")


def _show(verbose: int | bool | None) -> bool:
    return _vlevel(verbose) >= VERBOSE_PROGRESS


def _print_report(title: str, table: Table | None = None, notes: list | None = None,
                  verbose: int | bool | None = None) -> None:
    """Print one report as a block: bold title, the table, then dim notes."""
    if not _show(verbose):
        return
    parts = [Text(title, style="bold")] if title else []
    if table is not None:
        parts.append(table)
    parts += [Text(n, style="dim") for n in (notes or [])]
    _console.print(Group(*parts, Text("")))


def _write_csv(df: DataFrame, outpath: str | None, filename: str) -> None:
    if not outpath:
        return
    target = Path(outpath) / filename
    df.to_csv(target)
    info(f"Saved {target}")


def _scores_by_year(scores: dict) -> dict:
    """Per-year score dicts, also for a method that reports one set for the whole record."""
    if scores and all(isinstance(v, dict) for v in scores.values()):
        return scores
    return {'all': scores}


def _year_span(years) -> str:
    """'2016-2018' for consecutive years, else the years listed."""
    years = sorted(int(y) for y in years)
    if len(years) > 1 and years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]}-{years[-1]}"
    return ", ".join(str(y) for y in years)


def _report_scores(data, attr: str, title: str, notes: list, outfile_prefix: str,
                   outpath: str | None, verbose: int | bool | None) -> None:
    """Shared body of the two score reports: one table over all methods and scenarios."""
    table = _table()
    for col in ("method", "USTAR", "year"):
        table.add_column(col)
    for _, label, _ in _SCORE_COLUMNS:
        table.add_column(label, justify="right")
    unavailable = []
    for method, scenario, inst in _iter_level41(data):
        scores = _optional(inst, attr)
        if scores is None:
            unavailable.append(f"{_label(method)} ({scenario})")
            continue
        by_year = _scores_by_year(scores)
        for year, s in by_year.items():
            table.add_row(_label(method), scenario, str(year),
                          *[fmt.format(s[k]) if isinstance(s.get(k), (int, float)) else "-"
                            for k, _, fmt in _SCORE_COLUMNS])
        if len(by_year) > 1:
            # The mean over years, the figure a multi-year record is usually judged by.
            mean = DataFrame(by_year).mean(axis=1, numeric_only=True)
            table.add_row(_label(method), scenario, "mean",
                          *[fmt.format(mean[k]) if k in mean else "-" for k, _, fmt in _SCORE_COLUMNS],
                          style="bold")
            table.add_section()
        _write_csv(_scores_frame(scores), outpath, f"{outfile_prefix}_{scenario}_{method}.csv")
    if unavailable:
        notes = notes + [f"Not available for {', '.join(unavailable)}."]
    _print_report(title, table if table.row_count else None, notes, verbose=verbose)


def report_gapfilling_variables(data, verbose: int | bool | None = None) -> None:
    """Report what each L4.1 run filled: record counts and the column names.

    One row per method and USTAR scenario: the records in the period, how many
    of them were measured (accepted after Level 3.3) and how many the method
    filled. The target and output column names follow below the table.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        verbose: Verbosity for the console helpers.

    Example::

        report_gapfilling_variables(data)
        # XGBoost  CUT_50  17,520  8,555  8,965  51.2
    """
    table = _table()
    for col in ("method", "USTAR"):
        table.add_column(col)
    for col in ("records", "measured", "filled", "filled %"):
        table.add_column(col, justify="right")
    names = []
    for method, scenario, inst in _iter_level41(data):
        gapfilled = data.gapfilled_cols()[method][scenario]
        target = data.nongapfilled_cols()[method][scenario]
        n = len(data.fpc_df)
        n_meas = int(data.fpc_df[target].notna().sum())
        n_filled = int(data.fpc_df[gapfilled].notna().sum()) - n_meas
        table.add_row(_label(method), scenario, f"{n:,}", f"{n_meas:,}", f"{n_filled:,}",
                      f"{100 * n_filled / n:.1f}" if n else "-")
        names.append(f"{_label(method)} ({scenario}): {inst.target_col} -> {gapfilled}")
    _print_report("Gap-filled variables", table, names, verbose=verbose)


def report_annual_sums(data, gain: float = 1.0, units: str = "",
                       verbose: int | bool | None = None) -> DataFrame:
    """Report the annual sum of every gap-filled flux, per method and USTAR scenario.

    This is the headline result of a gap-filled flux record: the annual balance,
    and how much of it rests on measured records. Each cell gives the sum of the
    gap-filled series and, in brackets, the share of the year's records that
    were measured. A year whose record count is below the full calendar year is
    marked as incomplete; its sum is not comparable with the others.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        gain: Multiply every flux value by this factor before summing. For
            30-min NEE in µmol CO2 m-2 s-1 -> gC m-2, use ``12.011 * 1e-6 * 1800``.
        units: Unit of the sums, shown in the title.
        verbose: Verbosity for the console helpers.

    Returns:
        DataFrame of the annual sums, one row per year and one column per
        gap-filled variable.

    Example::

        report_annual_sums(data, gain=12.011 * 1e-6 * 1800, units='gC m-2')
    """
    gf_cols = data.gapfilled_cols()
    ngf_cols = data.nongapfilled_cols()
    df = data.fpc_df
    years = df.index.year
    freq = df.index.to_series().diff().median()

    table = _table()
    table.add_column("year")
    sums = {}
    for method, scenario, _ in _iter_level41(data):
        col = gf_cols[method][scenario]
        table.add_column(f"{_label(method)}\n{scenario}", justify="right")
        sums[col] = df[col].multiply(gain).groupby(years).sum(min_count=1)
    sums = DataFrame(sums)

    incomplete = []
    for year in sums.index:
        in_year = years == year
        full = pd.Timestamp(f"{year + 1}-01-01") - pd.Timestamp(f"{year}-01-01")
        is_full = in_year.sum() * freq >= full
        if not is_full:
            incomplete.append(str(year))
        cells = []
        for method, scenario, _ in _iter_level41(data):
            meas = df.loc[in_year, ngf_cols[method][scenario]].notna().mean() * 100
            total = sums.loc[year, gf_cols[method][scenario]]
            cells.append(f"{total:,.0f} ({meas:.0f}%)" if pd.notna(total) else "-")
        table.add_row(f"{year}{'' if is_full else ' *'}", *cells)

    notes = ["In brackets: the share of the year's records that were measured."]
    if incomplete:
        notes.append(f"* Incomplete year ({', '.join(incomplete)}): the sum covers only part of it.")
    title = f"Annual sums{f' ({units})' if units else ''}"
    _print_report(title, table, notes, verbose=verbose)
    return sums


def report_traintest_model_scores(data, outpath: str | None = None,
                                  verbose: int | bool | None = None) -> None:
    """Report held-out test scores per year, per method and USTAR scenario.

    These come from a random train/test split of the complete rows, which is the
    split that reproduces the gap-filling task. MDS has no such split and is
    reported as unavailable. The table shows R², RMSE, MAE, MedAE and the
    maximum error; the CSV files hold every metric.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        verbose: Verbosity for the console helpers.
    """
    _report_scores(data, 'scores_traintest_', "Held-out test scores",
                   ["Test set: a random share of the complete records, not used in training.",
                    "Errors are in the units of the flux."],
                   'traintest_model_scores', outpath, verbose)


def report_traintest_details(data, outpath: str | None = None,
                             verbose: int | bool | None = None) -> None:
    """Report the train/test split of each year's model and the model settings.

    One row per method, USTAR scenario and year: the complete records the model
    could use, how many went into training and testing, and the number of
    features. The hyperparameters follow below the table, once per method when
    they are the same for all scenarios. MDS trains no model and is reported as
    unavailable.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        verbose: Verbosity for the console helpers.
    """
    table = _table()
    for col in ("method", "USTAR", "year"):
        table.add_column(col)
    for col in ("complete", "train", "test", "test %", "features"):
        table.add_column(col, justify="right")
    unavailable = []
    params = {}
    for method, scenario, inst in _iter_level41(data):
        details = _optional(inst, 'traintest_details_')
        if details is None:
            unavailable.append(f"{_label(method)} ({scenario})")
            continue
        rows = []
        for year, d in details.items():
            n_train, n_test = len(d['train_df']), len(d['test_df'])
            rows.append({'year': year, 'complete': n_train + n_test, 'train': n_train,
                         'test': n_test, 'test_pct': 100 * n_test / (n_train + n_test),
                         'features': len(d['X_names'])})
            table.add_row(_label(method), scenario, str(year), f"{n_train + n_test:,}",
                          f"{n_train:,}", f"{n_test:,}", f"{rows[-1]['test_pct']:.0f}",
                          str(len(d['X_names'])))
            model_params = d['model'].get_params()
            params.setdefault(method, {})[scenario] = {
                k: v for k, v in model_params.items() if v is not None}
        if len(details) > 1:
            table.add_section()
        _write_csv(DataFrame(rows).set_index('year'), outpath,
                   f"traintest_model_details_{scenario}_{method}.csv")

    notes = []
    for method, per_scenario in params.items():
        settings = list(per_scenario.values())
        if all(s == settings[0] for s in settings):
            per_scenario = {'all scenarios': settings[0]}
        for scenario, p in per_scenario.items():
            notes.append(f"{_label(method)} ({scenario}): "
                         + ", ".join(f"{k}={v}" for k, v in sorted(p.items())))
    if unavailable:
        notes.append(f"Not available for {', '.join(unavailable)} (no model trained).")
    _print_report("Training and test data", table if table.row_count else None, notes,
                  verbose=verbose)


def report_gapfilling_model_scores(data, outpath: str | None = None,
                                   verbose: int | bool | None = None) -> None:
    """Report in-sample model scores per year, per method and USTAR scenario.

    In-sample scores are optimistically biased; compare against
    :func:`report_traintest_model_scores` for the held-out counterpart.
    MDS in ``mode='diive'`` (the default) is the exception: it leaves each
    measured record out of its own prediction, so its scores are out-of-sample.
    The table shows R², RMSE, MAE, MedAE and the maximum error; the CSV files
    hold every metric.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        verbose: Verbosity for the console helpers.
    """
    _report_scores(data, 'scores_', "In-sample scores",
                   ["Model predictions against every measured record, training records "
                    "included, so optimistic for the ML methods.",
                    "Errors are in the units of the flux."],
                   'gapfilling_model_scores', outpath, verbose)


def report_gapfilling_feature_importances(data, outpath: str | None = None, top_n: int = 10,
                                          verbose: int | bool | None = None) -> None:
    """Report the strongest SHAP feature importances of the ML gap-fillers.

    One table per method and USTAR scenario, with the ``top_n`` features ranked
    by their mean importance across years and one column per year. The CSV files
    hold every feature.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        top_n: Number of features shown in each table.
        verbose: Verbosity for the console helpers.
    """
    unavailable = []
    for method, scenario, inst in _iter_level41(data):
        importances = _optional(inst, 'feature_importance_per_year')
        if importances is None:
            unavailable.append(f"{_label(method)} ({scenario})")
            continue
        _write_csv(importances, outpath,
                   f"gapfilling_model_feature_importances_{scenario}_{method}.csv")
        ranked = importances.assign(_mean=importances.mean(axis=1)).sort_values(
            '_mean', ascending=False)
        multi_year = importances.shape[1] > 1
        table = _table()
        table.add_column("#", justify="right", style="dim")
        table.add_column("feature")
        for year in importances.columns:
            table.add_column(str(year), justify="right")
        if multi_year:
            table.add_column("mean", justify="right", style="bold")
        for rank, (feature, row) in enumerate(ranked.head(top_n).iterrows(), start=1):
            cells = [f"{row[y]:.3f}" if pd.notna(row[y]) else "-" for y in importances.columns]
            if multi_year:
                cells.append(f"{row['_mean']:.3f}")
            table.add_row(str(rank), escape(str(feature)), *cells)
        shown = min(top_n, len(ranked))
        _print_report(f"Feature importances, {_label(method)} ({scenario}): "
                      f"top {shown} of {len(ranked)} features, mean |SHAP|", table,
                      verbose=verbose)
    notes = ["Names starting with '.' are features engineered from the drivers "
             "(lags, rolling statistics, timestamp terms)."]
    if unavailable:
        notes.append(f"Not available for {', '.join(unavailable)} (no model trained).")
    _print_report("", None, notes, verbose=verbose)


def report_gapfilling_poolyears(data, verbose: int | bool | None = None) -> None:
    """Report which years of data each year's model was trained on.

    The long-term gap-fillers pool neighbouring years to train the model for a
    given year; this shows that pool, one row per year. When every method and
    scenario used the same pools they share one column. MDS does not pool and
    is reported as such.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        verbose: Verbosity for the console helpers.
    """
    pools = {}
    unavailable = []
    for method, scenario, inst in _iter_level41(data):
        yearpools = _optional(inst, 'yearpools')
        if yearpools is None:
            unavailable.append(f"{_label(method)} ({scenario})")
            continue
        pools[f"{_label(method)} ({scenario})"] = {
            year: _year_span(pool['poolyears']) for year, pool in yearpools.items()}
    if pools and all(p == next(iter(pools.values())) for p in pools.values()):
        pools = {'all ML runs': next(iter(pools.values()))}

    table = _table()
    table.add_column("model year")
    for name in pools:
        table.add_column(f"trained on, {name}")
    for year in sorted({y for p in pools.values() for y in p}):
        table.add_row(str(year), *[p.get(year, "-") for p in pools.values()])
    notes = [f"Not available for {', '.join(unavailable)} (does not pool years)."] if unavailable else []
    _print_report("Years pooled for each year's model", table if pools else None, notes,
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


def _gapfilled_series(data) -> list:
    """``(column, label, colour)`` for every gap-filled column, in chain order.

    With one method the USTAR scenarios get the ordered colours (light CUT_16 to dark CUT_84),
    so their order reads off the colour; with several methods every series gets its own.
    """
    from diive.core.plotting.styles import chainplots as cp
    out = []
    methods = data.gapfilled_cols()
    for method, scenarios in methods.items():
        for i, (scenario, col) in enumerate(scenarios.items()):
            label = scenario if len(methods) == 1 else f"{_label(method)} {scenario}"
            out.append([col, label, None])
    for i, row in enumerate(out):
        row[2] = cp.SCENARIO_COLORS[i % len(cp.SCENARIO_COLORS)]
    return [tuple(r) for r in out]


def plot_gapfilled_cumulative(data, gain: float = 1, units: str = "",
                              per_year: bool = True,
                              start_year: int | None = None,
                              end_year: int | None = None,
                              show_reference: bool = True,
                              excl_years_from_reference: list | None = None,
                              showplot: bool = True) -> None:
    """Plot cumulative sums of the gap-filled fluxes.

    Two views. ``per_year=True`` draws one panel per gap-filled variable, side by side on a
    shared y-axis, with each year as its own line over the day of the year and an optional
    band for the mean and standard deviation across years. ``per_year=False`` draws a single
    panel accumulating over the whole record, one line per gap-filled variable, with its total
    at the end of each line. Infinite values count as gaps.

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
        show_reference: Draw the band of mean and standard deviation across years
            (``per_year`` only, with at least two years).
        excl_years_from_reference: Years to leave out of that reference.
        showplot: Call ``plt.show()`` after rendering. Set False for headless use.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from diive.core.plotting.plotfuncs import nice_date_ticks
    from diive.core.plotting.styles import chainplots as cp
    from diive.core.times.times import doy_cumulatives_per_year, doy_mean_cumulative

    series_info = _gapfilled_series(data)
    if not series_info:
        raise RuntimeError("No gap-filled columns found. "
                           "Run at least one run_level41_*() function first.")
    df = data.fpc_df[[c for c, _, _ in series_info]].copy()
    years = df.index.year
    first, last = start_year or years.min(), end_year or years.max()
    df = df.loc[(years >= first) & (years <= last)]
    df = df.mask(np.isinf(df.to_numpy(dtype=float, na_value=np.nan))) * gain
    flux = data._flux_label()
    unit_txt = f" ({units})" if units else ""

    if not per_year:
        fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
        totals = []
        for col, label, color in series_info:
            cum = df[col].cumsum()
            ax.plot(cum.index, cum.to_numpy(), color=color, linewidth=2.2, label=label, zorder=3)
            valid = cum.dropna()
            if valid.empty:
                continue
            totals.append(valid.iloc[-1])
            ax.plot(valid.index[-1], valid.iloc[-1], 'o', color=color, markersize=6, zorder=4)
            ax.annotate(f"{valid.iloc[-1]:,.0f}", (valid.index[-1], valid.iloc[-1]), xytext=(8, 0),
                        textcoords='offset points', va='center', fontsize=10, fontweight='bold',
                        color=color)
        ax.axhline(0, color=cp.INK, linewidth=0.8, alpha=0.5, zorder=1)
        ax.set_ylabel(f"Cumulative {flux}{unit_txt}")
        cp.style_axes(ax)
        ax.margins(x=0.06)
        ax.legend(loc='best', frameon=False, fontsize=9, title='USTAR scenario', title_fontsize=9)
        nice_date_ticks(ax=ax)
        subtitle = f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}"
        if len(totals) > 1:
            subtitle += (f". The totals differ by {max(totals) - min(totals):,.0f}"
                         f"{f' {units}' if units else ''} between the lowest and the highest.")
        cp.figure_title(fig, f"Cumulative {flux} of every gap-filled series", subtitle)
    else:
        n = len(series_info)
        fig, axes = plt.subplots(1, n, figsize=(min(18, 5.2 * n + 1), 6), sharey=True,
                                 layout='constrained', squeeze=False)
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                       'Nov', 'Dec']
        for ax, (col, label, color) in zip(axes[0], series_info):
            cum_df = doy_cumulatives_per_year(df[col].rename(col))
            cum_df = cum_df.loc[:, cum_df.notna().any()]
            year_list = list(cum_df.columns)
            # Years from light to dark, so the most recent year is the darkest line.
            shades = plt.get_cmap('Blues')(np.linspace(0.35, 1.0, len(year_list))
                                           if len(year_list) > 1 else [0.85])
            if show_reference and len(year_list) > 1:
                ref = doy_mean_cumulative(cumulatives_per_year_df=cum_df,
                                          excl_years_from_reference=excl_years_from_reference)
                ax.fill_between(ref.index.to_numpy(), ref['MEAN-SD'].to_numpy(),
                                ref['MEAN+SD'].to_numpy(), color=cp.KEPT, alpha=0.25,
                                linewidth=0, label='mean ± 1 SD', zorder=1)
                ax.plot(ref.index.to_numpy(), ref['MEAN_DOY_TIME'].to_numpy(), color=cp.INK,
                        linewidth=1.2, linestyle=(0, (4, 3)), label='mean', zorder=2)
            for shade, year in zip(shades, year_list):
                valid = cum_df[year].dropna()
                ax.plot(cum_df.index.to_numpy(), cum_df[year].to_numpy(), color=shade,
                        linewidth=1.6, zorder=3,
                        label=f"{year}: {valid.iloc[-1]:,.0f}" if not valid.empty else f"{year}")
            ax.axhline(0, color=cp.INK, linewidth=0.8, alpha=0.5, zorder=0)
            ax.set_xticks(month_starts, month_names)
            ax.set_xlim(1, 367)
            cp.style_axes(ax)
            cp.panel_label(ax, label)
            ax.legend(loc='best', frameon=False, fontsize=8, ncol=1 if len(year_list) < 8 else 2)
        axes[0][0].set_ylabel(f"Cumulative {flux}{unit_txt}")
        cp.figure_title(fig, f"Cumulative {flux} per year",
                        "One line per year, the total at the end in the legend. A year that "
                        "starts or ends inside the record has a shorter line and is not "
                        "comparable with full years.")
    if showplot:
        plt.show()


def _driver_groups(features, drivers: list) -> dict:
    """The driver each feature comes from: its own name, a driver it is built from, or 'time'.

    The feature engineer names a derived feature ``.{driver}_{statistic}`` (or ``.{driver}+1``
    for a lag), and timestamp terms ``.DOY_COS`` and alike, so the driver is a substring.
    """
    groups = {}
    for f in features:
        name = str(f).lstrip('.')
        match = [d for d in drivers if name == d or name.startswith(f"{d}_") or name.startswith(f"{d}+")
                 or name.startswith(f"{d}-")]
        groups[f] = max(match, key=len) if match else 'time'
    return groups


def plot_feature_ranks_per_year(data, top_n: int = 15) -> None:
    """Plot the most important features of each ML gap-filler, one panel per USTAR scenario.

    With one year of data each panel is a bar chart of the ``top_n`` features by mean
    |SHAP|. With several years it is a rank chart: one line per feature, its rank in each
    year, for the ``top_n`` features by mean importance across years. The colour shows the
    driver a feature comes from, so it is visible at a glance whether radiation, temperature,
    VPD or the time of the year carries the model.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        top_n: Number of features shown in each panel.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from diive.core.plotting.styles import chainplots as cp

    runs = []
    for method, scenario, inst in _iter_level41(data):
        imp = _optional(inst, 'feature_importance_per_year')
        if imp is None:
            info(f"{_label(method)} ({scenario}): no feature ranks available.")
            continue
        runs.append((method, scenario, imp))
    if not runs:
        return

    all_features = {f for _, _, imp in runs for f in imp.index}
    drivers = sorted({str(f) for f in all_features if not str(f).startswith('.')}, key=len,
                     reverse=True)
    palette = cp.DRIVER_COLORS
    driver_color = {d: palette[i % len(palette)] for i, d in enumerate(sorted(drivers))}
    driver_color['time'] = cp.KEPT

    n = len(runs)
    multi_year = any(imp.shape[1] > 1 for _, _, imp in runs)
    height = max(4.5, 0.34 * top_n + 2.2)
    fig, axes = plt.subplots(1, n, figsize=(min(20, 6.2 * n), height), layout='constrained',
                             squeeze=False)
    for ax, (method, scenario, imp) in zip(axes[0], runs):
        groups = _driver_groups(imp.index, drivers)
        ranked = imp.assign(_mean=imp.mean(axis=1)).sort_values('_mean', ascending=False)
        top = ranked.head(top_n)
        if not multi_year:
            values = top['_mean'][::-1]
            colors = [driver_color[groups[f]] for f in values.index]
            ax.barh(range(len(values)), values.to_numpy(), color=colors, height=0.72)
            ax.set_yticks(range(len(values)), [str(f) for f in values.index], fontsize=8)
            for y, v in enumerate(values.to_numpy()):
                ax.text(v, y, f" {v:.2f}", va='center', fontsize=8, color=cp.MUTED)
            ax.set_xlabel('mean |SHAP|')
            cp.style_axes(ax, grid_axis='x')
            ax.margins(x=0.12)
        else:
            years = [c for c in imp.columns]
            ranks = imp[years].rank(ascending=False, method='first')
            for f in top.index:
                color = driver_color[groups[f]]
                ax.plot(years, ranks.loc[f].to_numpy(), '-o', color=color, linewidth=1.8,
                        markersize=6, markeredgecolor='white', markeredgewidth=0.8)
                last = ranks.loc[f].dropna()
                if not last.empty:
                    ax.text(last.index[-1], last.iloc[-1], f"  {f}", va='center', fontsize=7.5,
                            color=color)
            ax.set_ylim(top_n + 0.5, 0.5)
            ax.set_yticks(range(1, top_n + 1))
            ax.set_ylabel('rank')
            ax.set_xticks(years, [str(y) for y in years], rotation=45 if len(years) > 6 else 0)
            cp.style_axes(ax, grid_axis='y')
            ax.margins(x=0.25)
        cp.panel_label(ax, f"{_label(method)}, {scenario}: top {len(top)} of {len(imp)} features")

    handles = [Line2D([0], [0], color=c, linewidth=6, label=('time terms' if d == 'time' else d))
               for d, c in driver_color.items()]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False,
               fontsize=9, title='feature comes from', title_fontsize=9)
    cp.figure_title(fig, "Most important features of the gap-filling models",
                    "mean |SHAP| over the test data" + (", ranked in each year" if multi_year else "")
                    + ". Names starting with '.' are features engineered from the drivers.")
    plt.show()


def _quality_classes(data, ustar_scenario: str | None) -> dict:
    """The flux split into quality classes for one USTAR scenario, plus its gap-filled result.

    ``best``: QCF 0 after Level 3.3, the highest quality. ``used``: the records the gap-filling
    started from (QCF below the accepted limit, 0 and 1 by default). ``rejected``: every record
    with a value that is not used, whatever test removed it. ``filled``: the gap-filled result of
    the first L4.1 method that ran for this scenario.
    """
    used_all = data.levels.filteredseries_level33_qcf
    if not used_all:
        raise RuntimeError("run_level33_*() must be called first.")
    scenario = ustar_scenario or next(iter(used_all))
    if scenario not in used_all:
        raise ValueError(f"USTAR scenario {scenario!r} not found. Available: {list(used_all)}")
    used = used_all[scenario]
    best = data.levels.filteredseries_level33_hq[scenario]
    corrected = data.fpc_df[data.levels.flux_corrected_col or data.meta.fluxcol]
    rejected = corrected.where(corrected.notna() & used.isna())
    gf_cols = data.gapfilled_cols()
    method = next((m for m, sc in gf_cols.items() if scenario in sc), None)
    filled = data.fpc_df[gf_cols[method][scenario]] if method else None
    return dict(scenario=scenario, best=best, used=used, qcf1=used.where(best.isna()),
                rejected=rejected, filled=filled, method=method,
                night=data.fpc_df['NIGHTTIME'] == 1, flux=data._flux_label())


def plot_quality_heatmaps(data, ustar_scenario: str | None = None, units: str = '',
                          showplot: bool = True) -> None:
    """Heatmaps of the flux by quality class, and of the gap-filled result.

    Four panels on one colour scale centred on zero: the highest-quality records (QCF 0), the
    records the gap-filling used (QCF 0 and 1 by default), the records rejected by any level,
    and the gap-filled result. Grey cells hold no value in that class.

    Args:
        data: Chain container after ``run_level33_*`` (the last panel needs ``run_level41_*``).
        ustar_scenario: USTAR scenario to show. Defaults to the first.
        units: Unit of the flux, for the colour bars.
        showplot: Call ``plt.show()`` after drawing.
    """
    import matplotlib.pyplot as plt
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    from diive.core.plotting.styles import chainplots as cp

    q = _quality_classes(data, ustar_scenario)
    panels = [(q['best'], "Highest quality (QCF 0)"),
              (q['used'], "Used for gap-filling"),
              (q['rejected'], "Rejected"),]
    if q['filled'] is not None:
        panels.append((q['filled'], f"Gap-filled, {_label(q['method'])}"))
    limit = cp.symmetric_limit(*[s for s, _ in panels])
    fig = plt.figure(figsize=(4.4 * len(panels), 6.2), layout='constrained')
    gs = fig.add_gridspec(1, len(panels))
    first = None
    for i, (series, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i], sharey=first)
        first = first or ax
        HeatmapDateTime(series=series, verbose=False).plot(
            ax=ax, fig=fig, vmin=-limit, vmax=limit, cmap='RdBu_r', cb_extend='both',
            cb_digits_after_comma='auto', cb_labelsize=8, zlabel=units or q['flux'],
            color_bad=cp.NAN_COLOR, as_image=True, format_style=cp.heatmap_style(title))
        if i:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.yaxis.get_label().set_visible(False)
    n = len(q['used'])
    counts = [f"QCF 0: {q['best'].count():,} ({100 * q['best'].count() / n:.1f} %)",
              f"used: {q['used'].count():,} ({100 * q['used'].count() / n:.1f} %)",
              f"rejected: {q['rejected'].count():,} ({100 * q['rejected'].count() / n:.1f} %)"]
    cp.figure_title(fig, f"{q['flux']} by quality class, USTAR scenario {q['scenario']}",
                    "   ".join(counts) + f"   of {n:,} half-hours. Grey: no value in that class.")
    if showplot:
        plt.show()


def plot_quality_timeseries(data, ustar_scenario: str | None = None, units: str = '',
                            showplot: bool = True) -> None:
    """Time series of the flux by quality class, by day and by night, and the gap-filled result.

    Three rows, each with its distribution beside it. Daytime and nighttime show the records of
    QCF 0, QCF 1 and the rejected ones in three colours; the last row shows the gap-filled result,
    measured and filled records apart. The y-axis spans the records the gap-filling used, with a
    margin; rejected records beyond it are counted in the subtitle, not drawn.

    Args:
        data: Chain container after ``run_level33_*`` (the last row needs ``run_level41_*``).
        ustar_scenario: USTAR scenario to show. Defaults to the first.
        units: Unit of the flux, for the y-axis label.
        showplot: Call ``plt.show()`` after drawing.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from diive.core.base.flagbase import _marginal_histogram
    from diive.core.plotting.plotfuncs import nice_date_ticks
    from diive.core.plotting.styles import chainplots as cp

    q = _quality_classes(data, ustar_scenario)
    night = q['night']
    lo, hi = q['used'].quantile([0.0005, 0.9995])
    pad = 0.15 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    c0, c1, c2 = cp.QCF_COLORS
    classes = [(q['rejected'], c2, 'rejected', 2), (q['qcf1'], c1, 'QCF 1', 3),
               (q['best'], c0, 'QCF 0', 4)]
    rows = [("Daytime", ~night), ("Nighttime", night)]
    has_filled = q['filled'] is not None
    n_rows = 3 if has_filled else 2

    fig = plt.figure(figsize=(15, 3.6 * n_rows + 0.8), layout='constrained')
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[5, 1])
    axes = []
    for r, (label, mask) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0], sharex=axes[0] if axes else None)
        ax_h = fig.add_subplot(gs[r, 1], sharey=ax)
        layers = []
        for series, color, name, z in classes:
            sel = series[mask].dropna()
            ax.plot(sel.index, sel.to_numpy(), color=color, zorder=z,
                    label=f"{name} ({len(sel):,})", **{**cp.POINTS, 'markersize': 3})
            layers.append((sel.clip(*ylim), color, name))
        _marginal_histogram(ax_h, layers)
        ax.axhline(0, color=cp.INK, linewidth=0.7, alpha=0.5, zorder=1)
        ax.set_ylim(ylim)
        cp.style_axes(ax)
        ax.set_ylabel(f"{q['flux']}{f' ({units})' if units else ''}")
        cp.panel_label(ax, label)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1), frameon=False, markerscale=3,
                  fontsize=9, ncol=3, borderaxespad=0.2)
        axes.append(ax)
    if has_filled:
        ax = fig.add_subplot(gs[2, 0], sharex=axes[0])
        ax_h = fig.add_subplot(gs[2, 1], sharey=ax)
        measured = q['filled'].where(q['used'].notna()).dropna()
        filled_only = q['filled'].where(q['used'].isna()).dropna()
        ax.plot(measured.index, measured.to_numpy(), color=cp.KEPT, zorder=2,
                label=f"measured ({len(measured):,})", **cp.POINTS)
        ax.plot(filled_only.index, filled_only.to_numpy(), color=cp.CLEANED, zorder=3,
                label=f"filled ({len(filled_only):,})", **cp.POINTS)
        _marginal_histogram(ax_h, [(measured.clip(*ylim), cp.KEPT, 'measured'),
                                   (filled_only.clip(*ylim), cp.CLEANED, 'filled')])
        ax.axhline(0, color=cp.INK, linewidth=0.7, alpha=0.5, zorder=1)
        ax.set_ylim(ylim)
        cp.style_axes(ax)
        ax.set_ylabel(f"{q['flux']}{f' ({units})' if units else ''}")
        cp.panel_label(ax, f"Gap-filled result, {_label(q['method'])}")
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1), frameon=False, markerscale=3,
                  fontsize=9, ncol=2, borderaxespad=0.2)
        axes.append(ax)
    for ax in axes[:-1]:
        ax.tick_params(axis='x', labelbottom=False)
    nice_date_ticks(ax=axes[-1])

    outside = int(((q['rejected'] < ylim[0]) | (q['rejected'] > ylim[1])).sum())
    cp.figure_title(fig, f"{q['flux']} by quality class, USTAR scenario {q['scenario']}",
                    f"The y-axis spans the records used for gap-filling; {outside:,} rejected "
                    f"records lie outside it and are not drawn (the histograms add them at the "
                    f"edges).")
    if showplot:
        plt.show()


def _nighttime_frame(data, ustar_scenario, randunc_col):
    q = _quality_classes(data, ustar_scenario)
    if randunc_col not in data.full_df.columns:
        raise ValueError(f"{randunc_col!r} is not in the input data. Pass the random uncertainty "
                         f"column of the flux as randunc_col.")
    nee = q['used'][q['night']].dropna()
    sigma = data.full_df[randunc_col].reindex(nee.index)
    keep = sigma.notna() & (sigma > 0)
    return q, nee[keep], sigma[keep]


def report_nighttime_negatives(data, randunc_col: str = 'FC_RANDUNC_HF',
                               z_values: tuple = (1, 1.96, 3), gain: float = 1.0, units: str = '',
                               verbose: int | bool | None = None) -> DataFrame:
    """Report how far the negative nighttime records lie below zero, in random uncertainties.

    At night there is no photosynthesis, so NEE is respiration and should not be negative. A
    negative value can still be measurement noise around a small positive respiration. Dividing
    each value by its random uncertainty tells the two apart: a value within ``z`` uncertainties
    of zero is consistent with noise, one further below is not. One row per USTAR scenario and
    ``z``, over the nighttime records the gap-filling used.

    Args:
        data: Chain container after ``run_level33_*``.
        randunc_col: Column of ``data.full_df`` with the random uncertainty of the flux, in the
            flux units (EddyPro: ``FC_RANDUNC_HF``).
        z_values: Numbers of uncertainties to count below.
        gain: Factor for the sums (``12.011 * 1e-6 * 1800`` for NEE in gC m-2).
        units: Unit of the sums.
        verbose: Verbosity for the console helpers.

    Returns:
        The table as a DataFrame.
    """
    rows = []
    table = _table()
    for col in ("USTAR", "night records", "negative", "below", "records", "%", "sum"):
        table.add_column(col, justify="right" if col not in ("USTAR", "below") else "left")
    for scenario in data.levels.filteredseries_level33_qcf:
        _, nee, sigma = _nighttime_frame(data, scenario, randunc_col)
        z = nee / sigma
        neg = nee < 0
        for k in z_values:
            sel = z < -k
            row = {'USTAR': scenario, 'night records': len(nee), 'negative': int(neg.sum()),
                   'below': f"-{k} sigma", 'records': int(sel.sum()),
                   'pct': 100 * sel.mean() if len(nee) else float('nan'),
                   'sum': nee[sel].sum() * gain}
            rows.append(row)
            table.add_row(scenario, f"{len(nee):,}", f"{int(neg.sum()):,} ({100 * neg.mean():.1f} %)",
                          row['below'], f"{row['records']:,}", f"{row['pct']:.1f}",
                          f"{row['sum']:+,.1f}")
        table.add_section()
    _print_report(f"Negative nighttime records against their random uncertainty ({randunc_col})",
                  table, [f"Records used for gap-filling only. Sum in {units or 'flux units'}: "
                          f"what those records add to the total."], verbose=verbose)
    return DataFrame(rows)


def plot_nighttime_negatives(data, ustar_scenario: str | None = None,
                             randunc_col: str = 'FC_RANDUNC_HF', z: float = 1.96,
                             units: str = '', showplot: bool = True) -> None:
    """Nighttime records against their random uncertainty: which negative values are plausible.

    Three panels over the nighttime records the gap-filling used. Left: each value against its
    random uncertainty, with the line ``-z`` uncertainties below zero; a negative value above the
    line is consistent with noise, one below it is not. Middle: the values in units of their
    uncertainty, with the same limit. Right: per month, the share of nighttime records that are
    negative and the share that lie below the limit.

    Args:
        data: Chain container after ``run_level33_*``.
        ustar_scenario: USTAR scenario to show. Defaults to the first.
        randunc_col: Column of ``data.full_df`` with the random uncertainty of the flux.
        z: Number of uncertainties below zero that the limit lies at.
        units: Unit of the flux, for the axis labels.
        showplot: Call ``plt.show()`` after drawing.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from diive.core.plotting.styles import chainplots as cp

    q, nee, sigma = _nighttime_frame(data, ustar_scenario, randunc_col)
    zs = nee / sigma
    below = zs < -z
    within = (nee < 0) & ~below
    positive = nee >= 0
    c0, c1, c2 = cp.QCF_COLORS
    unit_txt = f" ({units})" if units else ""

    fig = plt.figure(figsize=(15, 5.8), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1.2])
    ax_sc, ax_z, ax_m = (fig.add_subplot(gs[0, i]) for i in range(3))

    for mask, color, label, zo in ((positive, cp.KEPT, 'positive', 2),
                                   (within, c1, f'negative, within {z:g} sigma', 3),
                                   (below, c2, f'negative, below -{z:g} sigma', 4)):
        ax_sc.plot(sigma[mask], nee[mask], color=color, zorder=zo,
                   label=f"{label} ({int(mask.sum()):,})", **{**cp.POINTS, 'markersize': 3.5})
    xs = np.linspace(0, sigma.quantile(0.995), 50)
    ax_sc.plot(xs, -z * xs, color=c2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=5,
               label=f"-{z:g} x uncertainty")
    ax_sc.axhline(0, color=cp.INK, linewidth=0.7, alpha=0.5)
    ax_sc.set_xlim(0, sigma.quantile(0.995))
    ax_sc.set_ylim(nee.quantile(0.002) * 1.1 if nee.quantile(0.002) < 0 else -1,
                   nee.quantile(0.995))
    ax_sc.set_xlabel(f"random uncertainty{unit_txt}")
    ax_sc.set_ylabel(f"nighttime {q['flux']}{unit_txt}")
    cp.style_axes(ax_sc, grid_axis='both')
    cp.panel_label(ax_sc, "Value against its uncertainty")
    ax_sc.legend(loc='upper left', frameon=False, fontsize=8.5, markerscale=3)

    zclip = zs.clip(-15, 30)
    edges = np.arange(-15, 30.5, 0.5)
    ax_z.hist([zclip[positive], zclip[within], zclip[below]], bins=edges, stacked=True,
              color=[cp.KEPT, c1, c2], edgecolor='white', linewidth=0.3)
    ax_z.axvline(-z, color=c2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax_z.axvline(0, color=cp.INK, linewidth=0.7, alpha=0.5)
    ax_z.set_yscale('log')
    ax_z.set_xlabel("value / uncertainty")
    ax_z.set_ylabel("records")
    cp.style_axes(ax_z)
    cp.panel_label(ax_z, "Value in units of its uncertainty")

    months = nee.index.to_period('M')
    share_neg = (nee < 0).groupby(months).mean() * 100
    share_below = below.groupby(months).mean() * 100
    x = share_neg.index.to_timestamp() + pd.Timedelta(days=14)
    ax_m.bar(x, share_neg.to_numpy(), width=24, color=c1, label='negative')
    ax_m.bar(x, share_below.to_numpy(), width=24, color=c2, label=f'below -{z:g} sigma')
    ax_m.set_ylabel('% of nighttime records')
    cp.style_axes(ax_m)
    cp.panel_label(ax_m, "Per month")
    ax_m.legend(loc='upper right', frameon=False, fontsize=8.5)
    from diive.core.plotting.plotfuncs import nice_date_ticks
    nice_date_ticks(ax=ax_m)

    n = len(nee)
    cp.figure_title(
        fig, f"Negative nighttime {q['flux']}, USTAR scenario {q['scenario']}",
        f"{n:,} nighttime records used for gap-filling: {int((nee < 0).sum()):,} negative "
        f"({100 * (nee < 0).mean():.1f} %), {int(below.sum()):,} of them below -{z:g} times their "
        f"random uncertainty ({100 * below.mean():.1f} %). Median uncertainty "
        f"{sigma.median():.2f}, median value {nee.median():.2f}.")
    if showplot:
        plt.show()


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
