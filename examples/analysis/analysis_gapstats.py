"""
===============================
Gap Statistics and Distribution
===============================

Characterize the spatial and temporal distribution of missing data in a
time series.

Extends basic gap detection (:class:`~diive.analysis.GapFinder`) with monthly
and annual breakdowns, explicit long-gap listing, and a three-panel figure.
Use this when you need to understand *when* gaps cluster (seasonal bias),
*how many* qualify as long gaps, and what the year-by-year coverage looks like.

Best for: Pre-gap-filling data quality assessment, flux processing chain audits
"""

# %%
# Load data
# ^^^^^^^^^

import diive as dv

data_df = dv.load_exampledata_parquet()

# Use the original (unfilled) NEE series — the series that will be gap-filled
series = data_df['NEE_CUT_REF_orig'].copy()

# %%
# Run gap statistics
# ^^^^^^^^^^^^^^^^^^
# long_gap_records=48 means "flag any gap >= 48 consecutive missing records as
# long" — on 30-min data that is exactly 1 day.  Adjust to match your sampling
# frequency and definition of a critical gap.

gs = dv.analysis.GapStats(
    series=series,
    long_gap_records=48,  # 48 records = 1 day at 30-min resolution
)

# %%
# Rich console report
# ^^^^^^^^^^^^^^^^^^^
# .report() prints a colour-coded summary: overview stats, long-gap table,
# monthly missing-data table, and annual coverage table.  Worst month and
# worst year are highlighted in yellow.

gs.report()

# %%
# Programmatic access — summary dict
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

s = gs.summary
print(f"Total missing: {s['total_missing_records']:,} records  ({s['missing_pct']:.1f}%)")
print(f"Long gaps (>= {s['long_gap_records_threshold']} records): {s['n_long_gaps']}")
print(f"Worst month: {s['worst_month']}  ({s['worst_month_missing_pct']:.1f}% missing)")
print(f"Annual coverage: {s['annual_coverage']}")

# %%
# Monthly statistics table
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Index is MONTH (1-12).  Columns: total_records, missing_records, missing_pct,
# n_gaps.  All years in the series are combined.

print(gs.monthly_stats)

# %%
# Annual coverage table
# ^^^^^^^^^^^^^^^^^^^^^
# Index is YEAR.  Columns: total_records, valid_records, missing_records,
# coverage_pct.

print(gs.annual_coverage)

# %%
# Long gaps table
# ^^^^^^^^^^^^^^^
# All gaps >= long_gap_records, sorted longest-first.
# Columns: GAP_START, GAP_END, GAP_LENGTH, GAP_DURATION, YEAR, MONTH.

print(f"Long gaps (top 10 of {len(gs.long_gaps)}):")
print(gs.long_gaps.head(10))

# %%
# Visualization
# ^^^^^^^^^^^^^
# Three-panel figure:
# - top:    daily data availability heatmap (day-of-year x year)
# - middle: monthly missing-data bar chart; bar annotation = number of gap periods
# - bottom: gap-length histogram on log scale with 1h / 1-day / 1-week reference lines

gs.showfig(
    title=f"Gap Statistics  --  {series.name}",
    saveplot=False,
    path=None,
)

# %%
# Embedding individual panels in a custom figure
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Every plot method accepts an Axes argument so panels can be mixed with
# other diive or matplotlib plots.  The polar chart requires a polar Axes
# created with ``projection='polar'``.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 8))
gsl = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

ax_hm = fig.add_subplot(gsl[0, :])
ax_po = fig.add_subplot(gsl[1, 0], projection='polar')
ax_tl = fig.add_subplot(gsl[1, 1])

gs.plot_availability_heatmap(ax=ax_hm)
gs.plot_monthly_polar(ax=ax_po)
gs.plot_gap_spike_timeline(ax=ax_tl)

fig.suptitle(f"Custom layout  --  {series.name}", fontsize=12, fontweight='bold')
fig.tight_layout()
plt.show()
