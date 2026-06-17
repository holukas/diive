"""
PARTITIONING REPORT
===================

Shared Rich console report for the NEE partitioning methods. Renders a per-year
summary table (RECO / GPP fill counts and means, temperature sensitivity E0)
plus overall totals, in the same style as the other diive structured reports.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
from pandas import DataFrame

from diive.core.utils.console import console as _console, rule
from diive.core.utils.console import VERBOSE_PROGRESS, _vlevel


def _fill(series) -> str:
    n = int(series.notna().sum())
    pct = 100.0 * series.notna().mean() if len(series) else 0.0
    return f"{n:,d} ({pct:.0f}%)"


def _mean(series) -> str:
    s = series.dropna()
    return f"{s.mean():.2f}" if len(s) else "-"


def partitioning_report(*, title: str, reference: str, results: DataFrame,
                        reco_col: str, gpp_col: str, e0_col: str | None = None,
                        e0_unit: str = "", reco_rob_col: str | None = None,
                        gpp_rob_col: str | None = None, se_col: str | None = None,
                        verbose: int | bool = 1) -> None:
    """Print a Rich per-year + total summary of a partitioning result.

    Args:
        title: Report header (e.g. ``"Daytime NEE Partitioning ONEFlux"``).
        reference: Citation / URL printed below the header.
        results: The partitioning results DataFrame (DatetimeIndex).
        reco_col, gpp_col: RECO and GPP column names.
        e0_col: Optional temperature-sensitivity column (per-year mean shown).
        e0_unit: Unit label for E0 (e.g. ``"degC"`` or ``"K"``).
        reco_rob_col, gpp_rob_col: Optional outlier-robust RECO/GPP columns
            (shown as a footnote).
        se_col: Optional GPP standard-error column (mean shown as a footnote).
        verbose: Verbosity; the report prints at ``VERBOSE_PROGRESS`` (2)+.
    """
    if _vlevel(verbose) < VERBOSE_PROGRESS:
        return
    from rich.table import Table

    reco, gpp = results[reco_col], results[gpp_col]
    years = results.index.year
    unique_years = sorted(set(years))

    rule(f"{title} Report", verbose=verbose)
    _console.print(f"  Reference: {reference}")

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Year", style="dim", no_wrap=True)
    table.add_column("Records", justify="right")
    table.add_column("RECO filled", justify="right")
    table.add_column("GPP filled", justify="right")
    table.add_column("RECO mean", justify="right")
    table.add_column("GPP mean", justify="right")
    if e0_col is not None:
        table.add_column(f"E0 [{e0_unit}]" if e0_unit else "E0", justify="right")

    for yr in unique_years:
        ym = years == yr
        row = [str(yr), f"{int(ym.sum()):,d}", _fill(reco[ym]), _fill(gpp[ym]),
               _mean(reco[ym]), _mean(gpp[ym])]
        if e0_col is not None:
            row.append(_mean(results[e0_col][ym]))
        table.add_row(*row)

    if len(unique_years) > 1:
        total = ["all", f"{len(results):,d}", _fill(reco), _fill(gpp),
                 _mean(reco), _mean(gpp)]
        if e0_col is not None:
            total.append(_mean(results[e0_col]))
        table.add_section()
        table.add_row(*[f"[bold]{c}[/bold]" for c in total])

    _console.print(table)

    notes = []
    if reco_rob_col is not None and reco_rob_col in results:
        notes.append(f"outlier-robust: RECO mean {_mean(results[reco_rob_col])}, "
                     f"GPP mean {_mean(results[gpp_rob_col])}")
    if se_col is not None and se_col in results:
        notes.append(f"GPP standard error mean {_mean(results[se_col])}")
    if notes:
        _console.print("  " + "; ".join(notes))
    _console.print("  Fluxes in umol m-2 s-1.")
