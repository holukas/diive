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
from rich.measure import Measurement
from rich.table import Table
from rich.text import Text

from diive.core.utils.console import VERBOSE_PROGRESS, _vlevel, console as _console, info, warn

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


def _mds_mode(data) -> str | None:
    """Mode of the L4.1 MDS run ('diive', 'oneflux' or 'reddyproc'), None when MDS did not run.

    Only ``mode='diive'`` leaves each measured record out of its own prediction. In the other
    modes the record is one of the records averaged for it, so its MDS prediction is in-sample.
    """
    for inst in data.levels.level41_mds.values():
        return getattr(inst, 'mode', 'diive')
    return None


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
    """Print one report as a block: bold title, the table, then dim notes.

    A table wider than the console is printed at its own width, so Rich does not cut its
    cells to fit (it would show '2.7…' instead of the number).
    """
    if not _show(verbose):
        return
    parts = [Text(title, style="bold")] if title else []
    width = _console.width
    if table is not None:
        parts.append(table)
        unbounded = _console.options.update_width(1000)
        width = max(width, Measurement.get(_console, unbounded, table).maximum)
    parts += [Text(n, style="dim") for n in (notes or [])]
    # print(width=...) cannot exceed the console width, so widen the console itself.
    default, _console.width = _console.width, width
    try:
        _console.print(Group(*parts, Text("")))
    finally:
        _console.width = default


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

    A gap-filler can leave records missing: MDS in ``mode='oneflux'`` removes
    long gaps by default, the REddyProc rules leave gaps over 60 days unfilled.
    The sum skips those records, so such a sum is marked as incomplete (``**``)
    and the notes give the unfilled records per year and column.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        gain: Multiply every flux value by this factor before summing. For
            30-min NEE in µmol CO2 m-2 s-1 -> gC m-2, use ``12.011 * 1e-6 * 1800``.
        units: Unit of the sums, shown in the title.
        verbose: Verbosity for the console helpers.

    Returns:
        DataFrame of the annual sums, one row per year and one column per
        gap-filled variable. ``attrs['unfilled']`` holds the number of records
        each column left missing, in the same layout.

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
    unfilled = df[list(sums.columns)].isna().groupby(years).sum()

    incomplete = []
    for year in sums.index:
        in_year = years == year
        full = pd.Timestamp(f"{year + 1}-01-01") - pd.Timestamp(f"{year}-01-01")
        is_full = in_year.sum() * freq >= full
        if not is_full:
            incomplete.append(str(year))
        cells = []
        for method, scenario, _ in _iter_level41(data):
            col = gf_cols[method][scenario]
            meas = df.loc[in_year, ngf_cols[method][scenario]].notna().mean() * 100
            total = sums.loc[year, col]
            mark = " **" if unfilled.loc[year, col] else ""
            cells.append(f"{total:,.0f} ({meas:.0f}%){mark}" if pd.notna(total) else "-")
        table.add_row(f"{year}{'' if is_full else ' *'}", *cells)

    notes = ["In brackets: the share of the year's records that were measured."]
    if incomplete:
        notes.append(f"* Incomplete year ({', '.join(incomplete)}): the sum covers only part of it.")
    if unfilled.to_numpy().any():
        notes.append("** Incomplete sum: the gap-filler left records unfilled, and the sum "
                     "skips them. Unfilled records per year (share of the year's records):")
        n_year = pd.Series(years).value_counts()
        for method, scenario, _ in _iter_level41(data):
            n = unfilled[gf_cols[method][scenario]]
            if n.any():
                notes.append(f"  {_label(method)} {scenario}: " + ", ".join(
                    f"{y} {int(v):,} ({100 * v / n_year[y]:.0f}%)" for y, v in n.items() if v))
    title = f"Annual sums{f' ({units})' if units else ''}"
    _print_report(title, table, notes, verbose=verbose)
    sums.attrs['unfilled'] = unfilled
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
    In ``mode='oneflux'`` and ``mode='reddyproc'`` MDS uses the record itself,
    so its scores are in-sample as well. The table shows R², RMSE, MAE, MedAE
    and the maximum error. The CSV files hold every metric.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        verbose: Verbosity for the console helpers.
    """
    mode = _mds_mode(data)
    if mode in (None, 'diive'):
        mds_note = ("MDS predicts each measured record without using it, so its scores are "
                    "out-of-sample and not comparable with those of the ML methods; "
                    "report_gapfilling_comparison scores both on the same held-out records.")
    else:
        mds_note = (f"MDS (mode='{mode}') uses each measured record in its own prediction, so "
                    f"its scores are in-sample as well. Only mode='diive' leaves the record out.")
    _report_scores(data, 'scores_', "In-sample scores",
                   ["Model predictions against every measured record, training records "
                    "included, so optimistic for the ML methods.",
                    mds_note,
                    "Errors are in the units of the flux."],
                   'gapfilling_model_scores', outpath, verbose)


def report_gapfilling_feature_importances(data, outpath: str | None = None, top_n: int = 10,
                                          ustar_scenario: str | None = None,
                                          verbose: int | bool | None = None) -> None:
    """Report the strongest SHAP feature importances of the ML gap-fillers.

    One table per method and USTAR scenario, with the ``top_n`` features ranked
    by their mean importance across years and one column per year. The CSV files
    hold every feature.

    Args:
        data: Chain container after at least one ``run_level41_*``.
        outpath: Directory to also write one CSV per method and scenario.
        top_n: Number of features shown in each table.
        ustar_scenario: Report only this USTAR scenario, e.g. to give each table its own
            heading in a notebook. Default: all scenarios.
        verbose: Verbosity for the console helpers.
    """
    unavailable = []
    for method, scenario, inst in _iter_level41(data):
        if ustar_scenario is not None and scenario != ustar_scenario:
            continue
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
            cells = [f"{row[y]:.2f}" if pd.notna(row[y]) else "-" for y in importances.columns]
            if multi_year:
                cells.append(f"{row['_mean']:.2f}")
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


def report_model_overview(data, verbose: int | bool | None = None) -> DataFrame:
    """Report how each gap-filling model was built and how well it did, one row per year.

    For each ML method, USTAR scenario and year: the years the model was trained on, the
    number of features, the training and test records, the number of trees (with early
    stopping: the number it chose, and the maximum), the records the final model was fitted
    on, the test scores, and how the target year's gaps were filled (by the model or, where a
    driver was missing, by the fallback model on timestamp features). The notes describe the
    procedure and list the hyperparameters. MDS trains no model and is left out.

    Args:
        data: Chain container after at least one ``run_level41_rf`` or ``run_level41_xgb``.
        verbose: Verbosity for the console helpers.

    Returns:
        DataFrame with one row per method, scenario and year.
    """
    rows = []
    params = {}
    for method, scenario, inst in _iter_level41(data):
        yearly = _optional(inst, 'results_yearly_')
        if yearly is None:
            continue
        pools = _optional(inst, 'yearpools') or {}
        for year, m in yearly.items():
            details = m.traintest_details_
            n_max = m.kwargs.get('n_estimators')
            flag = m.gapfilling_df_[m.target_gapfilled_flag_col]
            in_year = flag.index.year == int(year)
            rows.append({
                'method': _label(method), 'USTAR': scenario, 'year': year,
                'trained on': _year_span(pools[year]['poolyears']) if year in pools else '-',
                'features': len(details['X_names']),
                'train': len(details['train_df']), 'test': len(details['test_df']),
                'trees': m._n_trees if m._n_trees is not None else n_max, 'max trees': n_max,
                'final fit': len(m.model_df.dropna()),
                'test R2': m.scores_traintest_.get('r2'), 'test RMSE': m.scores_traintest_.get('rmse'),
                'measured': int((flag[in_year] == 0).sum()),
                'filled': int((flag[in_year] == 1).sum()),
                'fallback': int((flag[in_year] == 2).sum()),
            })
            params.setdefault(_label(method), {k: v for k, v in m.kwargs.items() if v is not None})
    out = DataFrame(rows)
    if out.empty:
        _print_report("Gap-filling models", None, ["No ML gap-filling model was trained."],
                      verbose=verbose)
        return out

    table = _table()
    show_method = out['method'].nunique() > 1
    cols = (['method'] if show_method else []) + ['USTAR', 'year', 'trained on']
    for col in cols:
        table.add_column(col.replace(' ', '\n'), no_wrap=True)
    for col in ('features', 'train', 'test', 'trees', 'final\nfit', 'test\nR²', 'test\nRMSE',
                'filled', 'fall-\nback'):
        table.add_column(col, justify="right", no_wrap=True)
    previous = None
    for _, r in out.iterrows():
        if previous is not None and r['USTAR'] != previous:
            table.add_section()
        previous = r['USTAR']
        trees = f"{r['trees']}" if r['trees'] == r['max trees'] else f"{r['trees']}/{r['max trees']}"
        table.add_row(*([r['method']] if show_method else []), r['USTAR'], str(r['year']),
                      r['trained on'], str(r['features']), f"{r['train']:,}", f"{r['test']:,}",
                      trees, f"{r['final fit']:,}", f"{r['test R2']:.3f}", f"{r['test RMSE']:.2f}",
                      f"{r['filled']:,}", f"{r['fallback']:,}")
    notes = [
        "Each year has its own model, trained on that year and its neighbours ('trained on').",
        "train / test: a random 75 / 25 % split of the complete records; the test scores come from "
        "the 25 %, which the model did not see.",
    ]
    early_stopping = any(p.get('early_stopping_rounds') for p in params.values())
    selection = _tree_selection(data)
    if early_stopping:
        notes.append("trees: with early stopping, the number it chose on 10 % of the training "
                     "records, out of the maximum; at the maximum, more trees might still help.")
    elif selection is not None:
        notes.append(f"trees: {selection.n_estimators} for every model, chosen by blocked "
                     f"cross-validation (report_tree_selection).")
    notes.append("final fit: the gaps are filled by a model refitted on all complete records with "
                 "that number of trees. filled / fallback: gaps of the model year filled by this "
                 "model, or by the fallback model on timestamp features where a driver was missing.")
    if early_stopping and out['trees'].eq(out['max trees']).any() and out['max trees'].notna().all():
        n_cap = int(out['trees'].eq(out['max trees']).sum())
        notes.append(f"{n_cap} model(s) used all trees: early stopping did not stop them.")
    for label, p in params.items():
        notes.append(f"{label}: " + ", ".join(f"{k}={v}" for k, v in sorted(p.items())))
    _print_report("Gap-filling models", table, notes, verbose=verbose)
    return out


def _tree_selection(data):
    """The blocked cross-validation result of the ML gap-fillers, or None."""
    for _, _, inst in _iter_level41(data):
        selection = getattr(inst, 'tree_selection_', None)
        if selection is not None:
            return selection
    return None


def _pool_labels(inst) -> dict:
    """``{year: '2016-2018'}``: the pool of each yearly model, labelled as the tree selection does."""
    labels = {}
    for year in inst.results_yearly_:
        years = inst.yearpools[year]['poolyears']
        labels[year] = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
    return labels


def report_tree_selection(data, verbose: int | bool | None = None) -> DataFrame:
    """Report the number of trees chosen by blocked cross-validation, and the blocked scores.

    One row per yearly pool of the scenario the selection ran on: the records, the RMSE, MAE
    and R² on held-out blocks of days at the chosen number of trees, and, for comparison, the
    scores on the random test records of the models of that pool. The blocked scores describe
    gaps of the block length; the random scores describe single missing half-hours. Needs
    ``run_level41_xgb(..., n_estimators_cv=True)``.

    Args:
        data: Chain container after ``run_level41_xgb`` with ``n_estimators_cv``.
        verbose: Verbosity for the console helpers.

    Returns:
        DataFrame with one row per pool.
    """
    selection = _tree_selection(data)
    if selection is None:
        _print_report("Number of trees", None,
                      ["No tree selection: run run_level41_xgb with n_estimators_cv=True."],
                      verbose=verbose)
        return DataFrame()
    inst = data.levels.level41_xgb[selection.scenario]
    random_scores = {}
    for year, label in _pool_labels(inst).items():
        random_scores.setdefault(label, []).append(inst.results_yearly_[year].scores_traintest_)
    out = selection.scores.copy()
    out['random RMSE'] = {k: sum(s['rmse'] for s in v) / len(v) for k, v in random_scores.items()}
    out['random R2'] = {k: sum(s['r2'] for s in v) / len(v) for k, v in random_scores.items()}
    out.index.name = 'pool'

    table = _table()
    table.add_column("pool", no_wrap=True)
    for col in ('records', 'blocked\nRMSE', 'blocked\nMAE', 'blocked\nR²', 'random\nRMSE',
                'random\nR²'):
        table.add_column(col, justify="right", no_wrap=True)
    for pool, r in out.iterrows():
        table.add_row(str(pool), f"{int(r['records']):,}", f"{r['rmse']:.2f}", f"{r['mae']:.2f}",
                      f"{r['r2']:.3f}", f"{r['random RMSE']:.2f}", f"{r['random R2']:.3f}")
    rule = (f"the smallest number at which the median error curve of the pools is within "
            f"{100 * selection.tolerance:g} % of its minimum" if selection.tolerance
            else "the minimum of the median error curve of the pools, rounded to ten")
    notes = [
        f"{selection.n_estimators} trees for every model: {rule} (up to {selection.max_trees} "
        f"tested). USTAR scenario {selection.scenario}.",
        f"blocked: {selection.n_folds}-fold cross-validation on the training records of each pool, "
        f"holding out whole {selection.block_days:g}-day blocks, at the chosen number of trees. "
        f"R² from the RMSE and the variance of the target.",
        "random: the test records of the models of that pool, random single half-hours that the "
        "models did not see. The difference shows how much harder gaps of days are than single "
        "missing records.",
    ]
    _print_report("Number of trees from blocked cross-validation", table, notes, verbose=verbose)
    return out


def plot_tree_selection(data, showplot: bool = True):
    """Plot the blocked cross-validation that chose the number of trees.

    Left: the error on the held-out blocks against the number of trees for each pool, relative
    to its minimum, with their median and the chosen number. Right: the blocked and the random
    test RMSE of each pool. Needs ``run_level41_xgb(..., n_estimators_cv=True)``.

    Args:
        data: Chain container after ``run_level41_xgb`` with ``n_estimators_cv``.
        showplot: Call ``plt.show()`` after drawing.

    Returns:
        The matplotlib figure, or None without a tree selection.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from diive.core.plotting.styles import chainplots as cp

    selection = _tree_selection(data)
    if selection is None:
        info("No tree selection: run run_level41_xgb with n_estimators_cv=True.")
        return None
    scores = report_tree_selection(data, verbose=0)
    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(15, 5.6), layout='constrained',
                                     gridspec_kw=dict(width_ratios=[1.4, 1]))
    relative = selection.curves / selection.curves.min()
    for col in relative.columns:
        ax_c.plot(relative.index, relative[col].to_numpy(), color=cp.KEPT, linewidth=1)
    ax_c.plot(selection.median_curve.index, selection.median_curve.to_numpy(), color=cp.INK,
              linewidth=2.2, label='median of the pools')
    if selection.tolerance:
        threshold = float(selection.median_curve.min()) * (1 + selection.tolerance)
        ax_c.axhline(threshold, color=cp.MUTED, linewidth=0.8, linestyle=(0, (4, 3)),
                     label=f'minimum + {100 * selection.tolerance:g} %')
    ax_c.axvline(selection.n_estimators, color=cp.REJECTED, linewidth=1.4,
                 label=f'chosen: {selection.n_estimators} trees')
    top = float(np.nanquantile(relative.to_numpy(), 0.98))
    ax_c.set_ylim(0.995, max(1.05, min(top, 1.3)))
    ax_c.set_xlabel('number of trees')
    ax_c.set_ylabel('RMSE on held-out blocks / minimum')
    cp.style_axes(ax_c)
    cp.panel_label(ax_c, "Error on held-out blocks, one line per pool")
    ax_c.legend(frameon=False, fontsize=8.5, loc='upper right')

    x = np.arange(len(scores))
    ax_s.bar(x - 0.2, scores['rmse'].to_numpy(), 0.4, color=cp.CLEANED,
             label=f'held-out {selection.block_days:g}-day blocks')
    ax_s.bar(x + 0.2, scores['random RMSE'].to_numpy(), 0.4, color=cp.KEPT,
             label='random test records')
    ax_s.set_xticks(x, [str(p) for p in scores.index], rotation=45, ha='right')
    ax_s.set_ylabel('RMSE')
    # Room above the bars for the legend.
    ax_s.set_ylim(0, 1.25 * float(np.nanmax(scores[['rmse', 'random RMSE']].to_numpy())))
    cp.style_axes(ax_s)
    cp.panel_label(ax_s, f"RMSE at {selection.n_estimators} trees")
    ax_s.legend(frameon=False, fontsize=8.5, loc='upper right', ncol=2)
    cp.figure_title(fig, "Number of trees from blocked cross-validation",
                    f"USTAR scenario {selection.scenario}, {selection.n_folds} folds of "
                    f"{selection.block_days:g}-day blocks. Every model uses {selection.n_estimators} "
                    f"trees, without early stopping.")
    if showplot:
        plt.show()
    return fig


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

    The USTAR scenarios get ordered shades (light CUT_16 to dark CUT_84), so their order reads off
    the colour. With several methods each method has its own hue.
    """
    from diive.core.plotting.styles import chainplots as cp
    out = []
    methods = data.gapfilled_cols()
    for method, scenarios in methods.items():
        shades = (cp.METHOD_COLORS.get(method, cp.SCENARIO_COLORS) if len(methods) > 1
                  else cp.SCENARIO_COLORS)
        for i, (scenario, col) in enumerate(scenarios.items()):
            label = scenario if len(methods) == 1 else f"{_label(method)} {scenario}"
            out.append((col, label, shades[i % len(shades)]))
    return out


def _series_method(data, col) -> str | None:
    """The L4.1 method that produced gap-filled column *col*."""
    return next((m for m, sc in data.gapfilled_cols().items() if col in sc.values()), None)


def _spread_labels(values: list, min_gap: float) -> list:
    """Positions for end labels: the values, pushed apart to at least *min_gap*."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = list(values)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if pos[j] - pos[i] < min_gap:
            pos[j] = pos[i] + min_gap
    return pos


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

    Records a gap-filler left missing (MDS in ``mode='oneflux'`` removes long gaps by default)
    are skipped, so a line with such records is incomplete: its legend entry gives the number
    of unfilled records, its total is marked with ``*`` and a console warning names the columns.

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
    flux = data._flux_varname()
    unit_txt = f" ({units})" if units else ""
    # Records the gap-filler left missing: the cumulative skips them, so say so.
    unfilled = df.isna()
    for col, n in unfilled.sum().items():
        if n:
            warn(f"{col}: {int(n):,} records left unfilled, the cumulative skips them and its "
                 f"total is incomplete.")
    incomplete_txt = ("* Incomplete: records left unfilled are skipped."
                      if unfilled.to_numpy().any() else "")

    if not per_year:
        fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
        ends = []
        for col, label, color in series_info:
            cum = df[col].cumsum()
            n_unfilled = int(unfilled[col].sum())
            if n_unfilled:
                label = f"{label} ({n_unfilled:,} unfilled)"
            ax.plot(cum.index, cum.to_numpy(), color=color, linewidth=2.2, label=label, zorder=3)
            valid = cum.dropna()
            if valid.empty:
                continue
            ends.append((valid.index[-1], valid.iloc[-1], color, col))
            ax.plot(valid.index[-1], valid.iloc[-1], 'o', color=color, markersize=6, zorder=4)
        if ends:
            lo, hi = ax.get_ylim()
            ypos = _spread_labels([e[1] for e in ends], min_gap=0.035 * (hi - lo))
            for (x, y, color, col), yl in zip(ends, ypos):
                mark = '*' if unfilled[col].any() else ''
                ax.annotate(f"{y:,.0f}{mark}", (x, yl), xytext=(8, 0), textcoords='offset points',
                            va='center', fontsize=10, fontweight='bold', color=color,
                            annotation_clip=False)
        ax.axhline(0, color=cp.INK, linewidth=0.8, alpha=0.5, zorder=1)
        ax.set_ylabel(f"Cumulative {flux}{unit_txt}")
        cp.style_axes(ax)
        ax.margins(x=0.08)
        n_methods = len(data.gapfilled_cols())
        ax.legend(loc='best', frameon=False, fontsize=9, title_fontsize=9,
                  title='method and USTAR scenario' if n_methods > 1 else 'USTAR scenario')
        nice_date_ticks(ax=ax)
        subtitle = f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}"
        u = f' {units}' if units else ''
        if n_methods > 1:
            by_method = {}
            for _, y, _, col in ends:
                by_method.setdefault(_series_method(data, col), []).append(y)
            subtitle += ". Range over the USTAR scenarios: " + ", ".join(
                f"{_label(m)} {max(v) - min(v):,.0f}{u}" for m, v in by_method.items())
            subtitle += ". See plot_gapfilling_comparison for the difference between the methods."
        elif len(ends) > 1:
            totals = [e[1] for e in ends]
            subtitle += (f". The totals differ by {max(totals) - min(totals):,.0f}{u} between "
                         f"the lowest and the highest.")
        if incomplete_txt:
            subtitle += f". {incomplete_txt}"
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
            # Years from light to dark, so the most recent year is the darkest line. Each method
            # keeps its hue.
            cmap = cp.METHOD_CMAPS.get(_series_method(data, col), 'Blues')
            shades = plt.get_cmap(cmap)(np.linspace(0.35, 1.0, len(year_list))
                                        if len(year_list) > 1 else [0.85])
            if show_reference and len(year_list) > 1:
                ref = doy_mean_cumulative(cumulatives_per_year_df=cum_df,
                                          excl_years_from_reference=excl_years_from_reference)
                ax.fill_between(ref.index.to_numpy(), ref['MEAN-SD'].to_numpy(),
                                ref['MEAN+SD'].to_numpy(), color=cp.KEPT, alpha=0.25,
                                linewidth=0, label='mean ± 1 SD', zorder=1)
                ax.plot(ref.index.to_numpy(), ref['MEAN_DOY_TIME'].to_numpy(), color=cp.INK,
                        linewidth=1.2, linestyle=(0, (4, 3)), label='mean', zorder=2)
            unfilled_year = unfilled[col].groupby(df.index.year).sum()
            for shade, year in zip(shades, year_list):
                valid = cum_df[year].dropna()
                year_label = f"{year}: {valid.iloc[-1]:,.0f}" if not valid.empty else f"{year}"
                n_unfilled = int(unfilled_year.get(int(year), 0))
                if n_unfilled:
                    year_label += f"* ({n_unfilled:,} unfilled)"
                ax.plot(cum_df.index.to_numpy(), cum_df[year].to_numpy(), color=shade,
                        linewidth=1.6, zorder=3, label=year_label)
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
                        "comparable with full years." + (f" {incomplete_txt}" if incomplete_txt else ""))
    if showplot:
        plt.show()


def _gap_lengths(measured: "pd.Series") -> "pd.Series":
    """Length in records of the gap each missing record belongs to (NaN where measured)."""
    gap = measured.isna()
    gap_id = (gap != gap.shift()).cumsum()
    length = gap.groupby(gap_id).transform('size')
    return length.where(gap)


def _comparison_frame(data, ustar_scenario: str, methods: tuple) -> DataFrame:
    """Filled values of two methods at the gaps of one USTAR scenario, with context columns.

    ``both`` marks the records where both methods have a value (measured or filled), ``gap``
    those of them that were not measured. A record that one method left unfilled is in neither.
    """
    a, b = methods
    gf, ngf = data.gapfilled_cols(), data.nongapfilled_cols()
    for m in methods:
        if ustar_scenario not in gf.get(m, {}):
            raise ValueError(f"Method {m!r} has no gap-filled series for {ustar_scenario!r}. "
                             f"Methods that ran: {list(gf)}")
    measured = data.fpc_df[ngf[a][ustar_scenario]]
    df = DataFrame({'a': data.fpc_df[gf[a][ustar_scenario]], 'b': data.fpc_df[gf[b][ustar_scenario]],
                    'measured': measured, 'night': data.fpc_df['NIGHTTIME'] == 1,
                    'gap_length': _gap_lengths(measured)})
    df['both'] = df['a'].notna() & df['b'].notna()
    df['gap'] = measured.isna() & df['both']
    return df


def _same_record_skill(data, ustar_scenario: str, methods: tuple) -> dict | None:
    """Scores of both methods on the held-out test records of the ML method.

    MDS in ``mode='diive'`` predicts each measured record without using it (leave one out), so
    its predictions at the ML test records are out-of-sample as well. In the other modes the
    record is part of its own MDS prediction, which would make the comparison favour MDS.
    Returns None if neither method is MDS, MDS did not run in ``mode='diive'`` or the ML method
    kept no test records.
    """
    import numpy as np
    if 'mds' not in methods:
        return None
    ml = next(m for m in methods if m != 'mds')
    mds = data.levels.level41_mds[ustar_scenario]
    if getattr(mds, 'mode', 'diive') != 'diive':
        return None
    ml_inst = getattr(data.levels, f'level41_{ml}')[ustar_scenario]
    yearly = _optional(ml_inst, 'results_yearly_')
    if not yearly:
        return None
    obs, p_ml, p_mds = [], [], []
    mds_pred = mds.gapfilling_df_['.PREDICTIONS']
    for year, m in yearly.items():
        d = m.traintest_details_
        test = d['test_df']
        if test.empty:
            continue
        in_year = test.index.year == int(year)
        test = test[in_year]
        pred = d['model'].predict(test.drop(columns=m.target_col).to_numpy())
        obs.append(test[m.target_col].to_numpy())
        p_ml.append(pred)
        p_mds.append(mds_pred.reindex(test.index).to_numpy())
    if not obs:
        return None
    obs, p_ml, p_mds = (np.concatenate(x) for x in (obs, p_ml, p_mds))
    ok = ~np.isnan(p_mds) & ~np.isnan(obs)

    def score(p):
        e = p[ok] - obs[ok]
        return {'r2': 1 - np.sum(e ** 2) / np.sum((obs[ok] - obs[ok].mean()) ** 2),
                'rmse': float(np.sqrt(np.mean(e ** 2))), 'bias': float(np.mean(e))}

    return {'n': int(ok.sum()), 'mds': score(p_mds), ml: score(p_ml)}


def report_gapfilling_comparison(data, methods: tuple = ('mds', 'xgb'), gain: float = 1.0,
                                 units: str = '', verbose: int | bool | None = None) -> DataFrame:
    """Compare the gap-filled results of two methods, for each USTAR scenario.

    Both methods are compared on the same records, those where both have a value (measured or
    filled), so their difference comes from the filled values alone and equals the difference of
    the two sums. The table gives, per scenario, the sum of each method and their difference
    (second minus first), the part of the difference from daytime and from nighttime gaps, the
    part from gaps of different lengths, and the scores of both methods on the same held-out
    records (the test records of the ML method, which MDS in ``mode='diive'`` also predicts
    without using them). Sums are times ``gain``.

    A record that one method left unfilled is left out of both sums: MDS in ``mode='oneflux'``
    removes long gaps by default, the REddyProc rules leave gaps over 60 days unfilled. The sums
    are then not totals of the record, and the table gives the unfilled records of each method.
    MDS in ``mode='oneflux'`` or ``mode='reddyproc'`` uses each measured record in its own
    prediction, so its scores on the held-out records would be in-sample. They are left out.

    Args:
        data: Chain container after both methods ran (e.g. ``run_level41_mds`` and
            ``run_level41_xgb``).
        methods: The two methods, e.g. ``('mds', 'xgb')``. The difference is second minus first.
        gain: Factor for the sums, e.g. ``12.011e-6 * 1800`` for gC m-2 from 30-min CO2 fluxes.
        units: Unit of the sums.
        verbose: Verbosity for the console helpers.

    Returns:
        DataFrame with one row per USTAR scenario.
    """
    a, b = methods
    gf = data.gapfilled_cols()
    scenarios = [sc for sc in gf.get(a, {}) if sc in gf.get(b, {})]
    classes = [(1, 1, '1'), (2, 12, '2-12'), (13, 48, '13-48'), (49, None, '>48')]
    rows = {}
    for sc in scenarios:
        d = _comparison_frame(data, sc, methods)
        diff = (d['b'] - d['a']).where(d['gap']) * gain
        # Both sums over the records both methods have, so the difference is sum b - sum a.
        row = {f'sum {_label(a)}': float(d['a'][d['both']].sum() * gain),
               f'sum {_label(b)}': float(d['b'][d['both']].sum() * gain),
               'difference': float(diff.sum()),
               'from day gaps': float(diff[~d['night']].sum()), 'from night gaps': float(diff[d['night']].sum()),
               f'unfilled {_label(a)}': int(d['a'].isna().sum()),
               f'unfilled {_label(b)}': int(d['b'].isna().sum())}
        for lo, hi, label in classes:
            sel = (d['gap_length'] >= lo) & ((d['gap_length'] <= hi) if hi else True)
            row[f'gaps {label}'] = float(diff[sel].sum())
        skill = _same_record_skill(data, sc, methods)
        if skill:
            row['test records'] = skill['n']
            for m in methods:
                row[f'R2 {_label(m)}'] = skill[m]['r2']
                row[f'RMSE {_label(m)}'] = skill[m]['rmse']
        rows[sc] = row
    out = DataFrame(rows).T
    out.index.name = 'USTAR'

    u = f" ({units})" if units else ""
    unfilled_cols = [f'unfilled {_label(m)}' for m in methods]
    any_unfilled = bool(len(out)) and bool(out[unfilled_cols].to_numpy().any())
    cols = [f'sum {_label(a)}', f'sum {_label(b)}', 'difference', 'from day gaps',
            'from night gaps', 'gaps 1', 'gaps 2-12', 'gaps 13-48', 'gaps >48']
    if any_unfilled:
        cols += unfilled_cols
    table = _table()
    table.add_column("USTAR")
    for col in cols:
        table.add_column(col.replace(' ', '\n', 1), justify="right", no_wrap=True)
    for sc, r in out.iterrows():
        table.add_row(sc, *[f"{r[c]:,.0f}" for c in cols])
    notes = [f"Sums{u}. difference = {_label(b)} - {_label(a)} over the filled gaps, split into "
             f"daytime and nighttime gaps and by gap length in records."]
    if any_unfilled:
        notes.append("Incomplete sums: a method left records unfilled (columns 'unfilled'). Both "
                     "sums cover only the records both methods filled or measured, so they are "
                     "not totals of the record.")
    mode = _mds_mode(data) if 'mds' in methods else None
    if mode not in (None, 'diive'):
        notes.append(f"No scores on the same held-out records: MDS (mode='{mode}') uses each "
                     f"measured record in its own prediction, so its scores there would be "
                     f"in-sample. Only mode='diive' leaves the record out.")
    if 'test records' in out.columns:
        for sc, r in out.iterrows():
            notes.append(f"{sc}, same {int(r['test records']):,} held-out records: "
                         + ", ".join(f"{_label(m)} R² {r[f'R2 {_label(m)}']:.2f} "
                                     f"RMSE {r[f'RMSE {_label(m)}']:.2f}" for m in methods))
        notes.append("Held-out records are random single half-hours, so these scores describe short "
                     "gaps; the sums depend more on the long ones.")
    _print_report(f"Gap-filling methods compared: {_label(a)} and {_label(b)}", table, notes,
                  verbose=verbose)
    return out


def plot_gapfilling_comparison(data, ustar_scenario: str | None = None, methods: tuple = ('mds', 'xgb'),
                               gain: float = 1.0, units: str = '', flux_units: str = '',
                               showplot: bool = True):
    """Show where two gap-filling methods disagree, in four panels.

    1. Filled values of both methods at the same gaps, by day and by night, with the 1:1 line:
       a cloud off the line is a bias, a wide cloud is scatter.
    2. Mean difference (second minus first) by month and hour of the day: where in the daily and
       seasonal cycle the methods differ.
    3. Contribution of each month to the difference of the sums, daytime and nighttime gaps apart.
    4. Cumulative difference over the record for every USTAR scenario, with the range of the
       scenarios of each method for comparison.

    All panels and sums use the records where both methods have a value. When a method left
    records unfilled (MDS in ``mode='oneflux'`` removes long gaps by default), the subtitle gives
    their number per method and a console warning says that the sums are incomplete.

    Args:
        data: Chain container after both methods ran.
        ustar_scenario: Scenario of panels 1 to 3. Defaults to the middle scenario.
        methods: The two methods; the difference is second minus first.
        gain: Factor for sums (panels 3 and 4), e.g. ``12.011e-6 * 1800`` for gC m-2.
        units: Unit of the sums.
        flux_units: Unit of the flux (panels 1 and 2).
        showplot: Call ``plt.show()`` after drawing.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from diive.core.plotting.plotfuncs import nice_date_ticks
    from diive.core.plotting.styles import chainplots as cp

    a, b = methods
    la, lb = _label(a), _label(b)
    gf = data.gapfilled_cols()
    scenarios = [sc for sc in gf.get(a, {}) if sc in gf.get(b, {})]
    if not scenarios:
        raise ValueError(f"Methods {methods} have no scenario in common. Methods that ran: {list(gf)}")
    scenario = ustar_scenario or scenarios[len(scenarios) // 2]
    d = _comparison_frame(data, scenario, methods)
    g = d[d['gap']]
    fu = f" ({flux_units})" if flux_units else ""
    u = f" {units}" if units else ""

    fig = plt.figure(facecolor='white', figsize=(16, 10.5), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    ax_sc, ax_mh, ax_mo, ax_cum = (fig.add_subplot(gs[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))

    for mask, color, label in ((~g['night'], cp.DAY, 'day'), (g['night'], cp.NIGHT, 'night')):
        x, y = g.loc[mask, 'a'], g.loc[mask, 'b']
        if len(x) > 2:
            r = np.corrcoef(x, y)[0, 1]
            label = f"{label}: {len(x):,} gaps, mean {lb} - {la} {np.mean(y - x):+.2f}, r {r:.2f}"
        ax_sc.plot(x, y, color=color, label=label, **{**cp.POINTS, 'markersize': 3, 'alpha': 0.5})
    lim = np.nanpercentile(np.abs(g[['a', 'b']].to_numpy()), 99.5) * 1.05 if len(g) else 1
    ax_sc.plot([-lim, lim], [-lim, lim], color=cp.INK, linewidth=0.8, linestyle=(0, (4, 3)))
    ax_sc.set_xlim(-lim, lim)
    ax_sc.set_ylim(-lim, lim)
    ax_sc.set_xlabel(f"filled by {la}{fu}")
    ax_sc.set_ylabel(f"filled by {lb}{fu}")
    cp.style_axes(ax_sc, grid_axis='both')
    cp.panel_label(ax_sc, f"Filled values at the same gaps, {scenario}")
    ax_sc.legend(loc='upper left', frameon=False, fontsize=8.5, markerscale=3)

    diff = (g['b'] - g['a'])
    hour = g.index.hour + g.index.minute / 60
    mh = diff.groupby([g.index.month, hour]).mean().unstack()
    mh = mh.reindex(index=range(1, 13))
    vmax = np.nanpercentile(np.abs(mh.to_numpy()), 98) if mh.notna().any().any() else 1
    im = ax_mh.pcolormesh(np.r_[mh.columns.to_numpy(), mh.columns.to_numpy()[-1] + 0.5] - 0.25,
                          np.arange(0.5, 13.5), mh.to_numpy(), cmap=cp.HEATMAP_CMAP,
                          vmin=-vmax, vmax=vmax, shading='flat')
    cb = fig.colorbar(im, ax=ax_mh, extend='both')
    cb.set_label(f"mean {lb} - {la}{fu}")
    ax_mh.set_yticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
                                    'Oct', 'Nov', 'Dec'])
    ax_mh.invert_yaxis()
    ax_mh.set_xticks(range(0, 25, 3))
    ax_mh.set_xlabel('hour of the day')
    cp.style_axes(ax_mh, grid_axis='none')
    cp.panel_label(ax_mh, "Mean difference at the gaps, by month and hour")

    contrib = (diff * gain).groupby([g.index.to_period('M'), g['night']]).sum().unstack(fill_value=0)
    contrib = contrib.reindex(columns=[False, True], fill_value=0)
    x = contrib.index.to_timestamp() + pd.Timedelta(days=14)
    ax_mo.bar(x, contrib[False].to_numpy(), width=12, color=cp.DAY, label='day gaps', align='edge')
    ax_mo.bar(x + pd.Timedelta(days=12), contrib[True].to_numpy(), width=12, color=cp.NIGHT,
              label='night gaps', align='edge')
    ax_mo.axhline(0, color=cp.INK, linewidth=0.7)
    ax_mo.set_ylabel(f"{lb} - {la}{f' ({units})' if units else ''}")
    cp.style_axes(ax_mo)
    cp.panel_label(ax_mo, f"Contribution of each month to the difference, {scenario}")
    ax_mo.legend(loc='best', frameon=False, fontsize=8.5)
    nice_date_ticks(ax=ax_mo)

    totals = {}
    shades = cp.METHOD_COLORS.get(b, cp.SCENARIO_COLORS)
    for i, sc in enumerate(scenarios):
        ds = _comparison_frame(data, sc, methods)
        cum = ((ds['b'] - ds['a']).where(ds['gap']).fillna(0) * gain).cumsum()
        ax_cum.plot(cum.index, cum.to_numpy(), color=cp.SCENARIO_COLORS[i % len(cp.SCENARIO_COLORS)],
                    linewidth=2, label=f"{sc}: {cum.iloc[-1]:+,.0f}{u}")
        totals[sc] = (ds['a'][ds['both']].sum() * gain, ds['b'][ds['both']].sum() * gain)
    ax_cum.axhline(0, color=cp.INK, linewidth=0.7)
    ax_cum.set_ylabel(f"cumulative {lb} - {la}{f' ({units})' if units else ''}")
    cp.style_axes(ax_cum)
    cp.panel_label(ax_cum, "Cumulative difference, every USTAR scenario")
    ax_cum.legend(loc='best', frameon=False, fontsize=8.5)
    nice_date_ticks(ax=ax_cum)

    spread_a = max(t[0] for t in totals.values()) - min(t[0] for t in totals.values())
    spread_b = max(t[1] for t in totals.values()) - min(t[1] for t in totals.values())
    d_sc = totals[scenario][1] - totals[scenario][0]
    n_gaps = int(d['gap'].sum())
    unfilled = {la: int(d['a'].isna().sum()), lb: int(d['b'].isna().sum())}
    if any(unfilled.values()):
        left = ", ".join(f"{k} {v:,}" for k, v in unfilled.items())
        gaps_txt = f"Compared on the {n_gaps:,} gaps both filled. Unfilled: {left}."
        warn(f"Gap-filling comparison, {scenario}: records left unfilled {left}. The sums cover "
             f"only the records both methods filled or measured.")
    else:
        gaps_txt = f"Both methods fill the same {n_gaps:,} gaps."
    cp.figure_title(fig, f"Gap-filling methods compared: {la} and {lb}",
                    f"{scenario}: {la} {totals[scenario][0]:,.0f}{u}, {lb} {totals[scenario][1]:,.0f}{u}, "
                    f"difference {d_sc:+,.0f}{u}. Range over the USTAR scenarios: {la} {spread_a:,.0f}{u}, "
                    f"{lb} {spread_b:,.0f}{u}. {gaps_txt}")
    if showplot:
        plt.show()
    return fig


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
    |SHAP|. With several years it is a rank table: one row per feature, for the ``top_n``
    features by mean importance across years, with its rank in each year written in the
    cell; ranks below ``top_n`` are shown in grey. The colour of the name shows the driver a
    feature comes from, so it is visible at a glance whether radiation, temperature, VPD or
    the time of the year carries the model. Methods without features (MDS) are skipped.

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
        if method == 'mds':  # no model, no features
            continue
        imp = _optional(inst, 'feature_importance_per_year')
        if imp is None:
            info(f"{_label(method)} ({scenario}): no feature ranks available.")
            continue
        runs.append((method, scenario, imp))
    if not runs:
        info("No feature ranks available: no ML gap-filling has run.")
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
    fig, axes = plt.subplots(1, n, figsize=(min(21, 7 * n), height), layout='constrained',
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
            years = list(imp.columns)
            ranks = imp[years].rank(ascending=False, method='first').loc[top.index]
            r = ranks.to_numpy(dtype=float)
            # Darker is more important; ranks below top_n get a grey cell and grey text.
            vmax = top_n * 1.6
            cmap = plt.get_cmap('Blues_r').with_extremes(over=cp.NAN_COLOR)
            ax.imshow(np.where(r > top_n, vmax + 1, r), cmap=cmap, vmin=1, vmax=vmax,
                      aspect='auto', interpolation='none')
            for (i, j), v in np.ndenumerate(r):
                if np.isnan(v):
                    continue
                inside = v <= top_n
                ax.text(j, i, f"{v:.0f}", ha='center', va='center', fontsize=7.5,
                        color=('white' if v <= top_n / 2 else cp.INK) if inside else cp.MUTED)
            ax.set_xticks(range(len(years)), [str(y) for y in years],
                          rotation=45 if len(years) > 6 else 0)
            ax.set_yticks(range(len(top)), [str(f) for f in top.index], fontsize=8)
            for label, f in zip(ax.get_yticklabels(), top.index):
                label.set_color(driver_color[groups[f]])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        cp.panel_label(ax, f"{_label(method)}, {scenario}: top {len(top)} of {len(imp)} features")

    handles = [Line2D([0], [0], color=c, linewidth=6, label=('time terms' if d == 'time' else d))
               for d, c in driver_color.items()]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False,
               fontsize=9, title='feature comes from', title_fontsize=9)
    cp.figure_title(fig, "Most important features of the gap-filling models",
                    "mean |SHAP| over the test data"
                    + (f", as the rank in each year (1 = most important; grey: below rank {top_n})"
                       if multi_year else "")
                    + ". Names starting with '.' are features engineered from the drivers.")
    plt.show()


def _quality_classes(data, ustar_scenario: str | None, method: str | None = None) -> dict:
    """The flux split into quality classes for one USTAR scenario, plus its gap-filled result.

    ``best``: QCF 0 after Level 3.3, the highest quality. ``used``: the records the gap-filling
    started from (QCF below the accepted limit, 0 and 1 by default). ``rejected``: every record
    with a value that is not used, whatever test removed it. ``filled``: the gap-filled result of
    *method* ('mds', 'rf', 'xgb'), or of the first L4.1 method that ran for this scenario.
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
    if method is not None and scenario not in gf_cols.get(method, {}):
        raise ValueError(f"No gap-filled series of method {method!r} for scenario {scenario!r}. "
                         f"Available methods: {[m for m, sc in gf_cols.items() if scenario in sc]}")
    method = method or next((m for m, sc in gf_cols.items() if scenario in sc), None)
    filled = data.fpc_df[gf_cols[method][scenario]] if method else None
    return dict(scenario=scenario, best=best, used=used, qcf1=used.where(best.isna()),
                rejected=rejected, filled=filled, method=method,
                night=data.fpc_df['NIGHTTIME'] == 1, flux=data._flux_varname(scenario))


def _gapfilling_input(data) -> dict:
    """Per USTAR scenario: the records the gap-filling uses and the highest-quality records."""
    used_all = data.levels.filteredseries_level33_qcf
    if not used_all:
        raise RuntimeError("run_level33_*() must be called first.")
    measured = data.fpc_df[data.levels.flux_corrected_col or data.meta.fluxcol].notna()
    passed = data.levels.filteredseries_level32_qcf
    passed = passed.notna() if passed is not None else measured
    return dict(used=used_all, best=data.levels.filteredseries_level33_hq, measured=measured,
                passed=passed, night=data.fpc_df['NIGHTTIME'] == 1)


def report_gapfilling_input(data, verbose: int | bool | None = None) -> DataFrame:
    """Report what the gap-filling starts from, for each USTAR scenario.

    After Level 3.3 every measured record that failed a test or the USTAR filter is removed. The
    table counts, per scenario, the measured records, the records removed by the quality tests and
    the outlier detection (Levels 2 to 3.2), the records removed by the USTAR filter, and the records
    left: those the gap-filling uses, split into the highest-quality records (QCF 0) and the records
    with QCF 1, and into day and night. Which QCF classes are used is set by
    ``daytime_accept_qcf_below`` and ``nighttime_accept_qcf_below`` of ``init_flux_data``. The
    highest-quality records are the measured flux for analyses that do not use gap-filled values.

    Args:
        data: Chain container after ``run_level33_*``.
        verbose: Verbosity for the console helpers.

    Returns:
        DataFrame with one row per USTAR scenario and the counts as columns.
    """
    g = _gapfilling_input(data)
    n = len(g['measured'])
    rows = {}
    for scenario, used in g['used'].items():
        used, best = used.notna(), g['best'][scenario].notna()
        rows[scenario] = {
            'measured': int(g['measured'].sum()),
            'removed by tests and outliers': int((g['measured'] & ~g['passed']).sum()),
            'removed by USTAR': int((g['passed'] & ~used).sum()),
            'used for gap-filling': int(used.sum()),
            'QCF 0': int(best.sum()),
            'QCF 1': int((used & ~best).sum()),
            'used by day': int((used & ~g['night']).sum()),
            'used by night': int((used & g['night']).sum()),
        }
    out = DataFrame(rows).T
    out.index.name = 'USTAR'

    table = _table()
    table.add_column("USTAR")
    headers = {'removed by tests and outliers': 'removed:\ntests', 'removed by USTAR': 'removed:\nUSTAR',
               'used for gap-filling': 'used', 'QCF 0': 'used:\nQCF 0', 'QCF 1': 'used:\nQCF 1',
               'used by day': 'used:\nday', 'used by night': 'used:\nnight'}
    for col in out.columns:
        table.add_column(headers.get(col, col), justify="right", no_wrap=True)
    for scenario, row in out.iterrows():
        table.add_row(scenario, *[f"{row[c]:,}\n{100 * row[c] / n:.0f}%" for c in out.columns])
    m = data.meta
    notes = [f"Records, and their share of all {n:,} records of the period.",
             "removed: tests = quality tests and outlier detection (Levels 2 to 3.2).",
             f"used = records for gap-filling: QCF below {m.daytime_accept_qcf_below} by day and below "
             f"{m.nighttime_accept_qcf_below} by night. QCF 0 is the highest quality, the measured flux "
             f"for analyses without gap-filled values."]
    _print_report("Records for gap-filling", table, notes, verbose=verbose)
    return out


def report_gaps(data, long_gap_days: float = 1, verbose: int | bool | None = None) -> DataFrame:
    """Report the gaps the gap-filling has to fill, for each USTAR scenario.

    Args:
        data: Chain container after ``run_level33_*``.
        long_gap_days: A gap of at least this many days counts as long.
        verbose: Verbosity for the console helpers.

    Returns:
        DataFrame with one row per USTAR scenario.
    """
    from diive.analysis import GapStats
    used_all = data.levels.filteredseries_level33_qcf
    if not used_all:
        raise RuntimeError("run_level33_*() must be called first.")
    per_day = pd.Timedelta('1D') / data.fpc_df.index.to_series().diff().median()
    rows = {}
    for scenario, series in used_all.items():
        gs = GapStats(series, long_gap_records=int(round(long_gap_days * per_day)))
        sm = gs.summary
        rows[scenario] = {
            'missing %': round(100 * series.isna().mean(), 1),
            'gaps': len(gs.results),
            'long gaps': sm['n_long_gaps'],
            'longest gap (days)': round(gs.results['GAP_LENGTH'].max() / per_day, 1)
            if len(gs.results) else 0.0,
            'worst month': pd.Timestamp(2000, sm['worst_month'], 1).strftime('%b'),
            'worst month missing %': round(sm['worst_month_missing_pct'], 1),
        }
    out = DataFrame(rows).T
    out.index.name = 'USTAR'
    table = _table()
    table.add_column("USTAR")
    for col in out.columns:
        table.add_column(col, justify="right")
    for scenario, row in out.iterrows():
        table.add_row(scenario, *[str(v) for v in row])
    _print_report("Gaps to fill", table,
                  [f"A long gap lasts at least {long_gap_days:g} day{'s' if long_gap_days != 1 else ''}."],
                  verbose=verbose)
    return out


def plot_gapfilling_input(data, units: str = '', showplot: bool = True) -> None:
    """Heatmaps of the records the gap-filling uses and of the highest-quality records.

    One column per USTAR scenario. The top row shows the records the gap-filling uses, the bottom
    row the highest-quality records (QCF 0) among them, on one colour scale centred on zero. Grey
    cells hold no value.

    Args:
        data: Chain container after ``run_level33_*``.
        units: Unit of the flux, for the colour bars.
        showplot: Call ``plt.show()`` after drawing.
    """
    import matplotlib.pyplot as plt
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    from diive.core.plotting.styles import chainplots as cp

    g = _gapfilling_input(data)
    scenarios = list(g['used'])
    n = len(g['measured'])
    rows = [(g['used'], "used for gap-filling"), (g['best'], "highest quality (QCF 0)")]
    limit = cp.symmetric_limit(*[r[0][s] for r in rows for s in scenarios])
    fig = plt.figure(figsize=(4.4 * len(scenarios), 12), layout='constrained')
    gs = fig.add_gridspec(2, len(scenarios))
    first = None
    for i, (series_by_scenario, label) in enumerate(rows):
        for j, scenario in enumerate(scenarios):
            series = series_by_scenario[scenario]
            ax = fig.add_subplot(gs[i, j], sharex=first, sharey=first)
            first = first or ax
            title = f"{scenario}: {label}\n{series.count():,} records ({100 * series.count() / n:.0f}%)"
            HeatmapDateTime(series=series, verbose=False).plot(
                ax=ax, fig=fig, vmin=-limit, vmax=limit, cmap=cp.HEATMAP_CMAP, cb_extend='both',
                cb_digits_after_comma='auto', cb_labelsize=8, zlabel=units or 'flux',
                color_bad=cp.NAN_COLOR, as_image=True, format_style=cp.heatmap_style(title))
            if j:
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.yaxis.get_label().set_visible(False)
    m = data.meta
    cp.figure_title(fig, "Records for gap-filling, per USTAR scenario",
                    f"Top: records the gap-filling uses (QCF below {m.daytime_accept_qcf_below} by day, "
                    f"below {m.nighttime_accept_qcf_below} by night). Bottom: highest quality (QCF 0). "
                    f"Shares of all {n:,} records. Grey: no value.")
    if showplot:
        plt.show()


def plot_quality_heatmaps(data, ustar_scenario: str | None = None, units: str = '',
                          method: str | None = None, showplot: bool = True) -> None:
    """Heatmaps of the flux by quality class, and of the gap-filled result.

    Four panels on one colour scale centred on zero: the highest-quality records (QCF 0), the
    records the gap-filling used (QCF 0 and 1 by default), the records rejected by any level,
    and the gap-filled result. Grey cells hold no value in that class.

    Args:
        data: Chain container after ``run_level33_*`` (the last panel needs ``run_level41_*``).
        ustar_scenario: USTAR scenario to show. Defaults to the first.
        units: Unit of the flux, for the colour bars.
        method: Gap-filling method of the last panel ('mds', 'rf', 'xgb'). Defaults to the
            first that ran.
        showplot: Call ``plt.show()`` after drawing.
    """
    import matplotlib.pyplot as plt
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    from diive.core.plotting.styles import chainplots as cp

    q = _quality_classes(data, ustar_scenario, method=method)
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
            ax=ax, fig=fig, vmin=-limit, vmax=limit, cmap=cp.HEATMAP_CMAP, cb_extend='both',
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
                            method: str | None = None, showplot: bool = True) -> None:
    """Time series of the flux by quality class, by day and by night, and the gap-filled result.

    Three rows, each with its distribution beside it. Daytime and nighttime show the records of
    QCF 0, QCF 1 and the rejected ones in three colours; the last row shows the gap-filled result,
    measured and filled records apart. The y-axis spans the records the gap-filling used, with a
    margin; rejected records beyond it are counted in the subtitle, not drawn.

    Args:
        data: Chain container after ``run_level33_*`` (the last row needs ``run_level41_*``).
        ustar_scenario: USTAR scenario to show. Defaults to the first.
        units: Unit of the flux, for the y-axis label.
        method: Gap-filling method of the last row ('mds', 'rf', 'xgb'). Defaults to the first
            that ran.
        showplot: Call ``plt.show()`` after drawing.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from diive.core.base.flagbase import _marginal_histogram
    from diive.core.plotting.plotfuncs import nice_date_ticks
    from diive.core.plotting.styles import chainplots as cp

    q = _quality_classes(data, ustar_scenario, method=method)
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
