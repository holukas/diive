"""
=========================
Compound Extreme Detection
=========================

Classify time periods (months or days) into compound-extreme categories from the
standardized anomalies (z-scores) of two driver variables, and visualize them in a
quadrant scatter (after Wang et al., Fig. 2).

The canonical use case is compound dry-hot detection: a period is an **atmospheric**
dryness extreme when vapour pressure deficit (VPD) is anomalously high, a **soil**
dryness extreme when soil water content (SWC) is anomalously low, and a **compound**
extreme when both occur together. Periods that cross neither threshold are *none*.

This example demonstrates:

- **CompoundExtremes** — the classifier (``dv.analysis``)
- **CompoundExtremesPlot** — the quadrant scatter (``dv.plotting``)
- Both standardization modes (deseasonalized vs. whole-record)
- Monthly and daily resolution

Best for: identifying anomalous dry/hot periods and screening for compound events.
"""

# %%
# Overview
# ^^^^^^^^
# Each period is aggregated to the chosen resolution, both variables are converted
# to z-scores, and a period is flagged extreme for a variable when its z-score
# crosses the threshold in that variable's extreme direction (high VPD, low SWC).
# The two flags combine into four categories: none, var1-only, var2-only, compound.

import matplotlib.pyplot as plt

import diive as dv

data_df = dv.load_exampledata_parquet()
vpd = data_df['VPD_f'].copy()      # vapour pressure deficit
swc = data_df['SWC_FF0_0.15_1'].copy()  # soil water content

# %%
# Monthly compound extremes (deseasonalized z-scores)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``standardize_by='season'`` standardizes each month against the same calendar
# month across years, removing the seasonal cycle so genuinely anomalous months
# stand out. VPD's extreme is the high tail, SWC's is the low tail.

ce = dv.analysis.CompoundExtremes(
    var1=vpd,
    var2=swc,
    agg='monthly',
    var1_extreme='high',
    var2_extreme='low',
    threshold=2.0,
    standardize_by='season',
    var1_label='Air',
    var2_label='Soil',
)

print("Category counts (deseasonalized, monthly):")
print(ce.counts)
print("\nResults (first rows):")
print(ce.results.head())

# %%
# Visualize the quadrant scatter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``from_compound_extremes`` wires the z-score columns, categories, period labels,
# and threshold lines straight from the analysis instance.

ce_plot = dv.plotting.CompoundExtremesPlot.from_compound_extremes(ce)
fig, ax = plt.subplots(figsize=(9, 8))
ce_plot.plot(ax=ax)
# fig.show()  # disabled for the example gallery

# %%
# Whole-record standardization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``standardize_by='record'`` uses a single mean/std over the whole record. Simpler,
# but the seasonal cycle of VPD dominates the z-score (summer months tend to flag).

ce_record = dv.analysis.CompoundExtremes(
    var1=vpd, var2=swc, agg='monthly', threshold=2.0,
    standardize_by='record', var1_label='Air', var2_label='Soil',
)
print("Category counts (whole-record, monthly):")
print(ce_record.counts)

# %%
# Daily resolution
# ^^^^^^^^^^^^^^^^
# The same classification at daily resolution (deseasonalized by day-of-year).

ce_daily = dv.analysis.CompoundExtremes(
    var1=vpd, var2=swc, agg='daily', threshold=2.0,
    standardize_by='season', var1_label='Air', var2_label='Soil',
)
print("Category counts (deseasonalized, daily):")
print(ce_daily.counts)
