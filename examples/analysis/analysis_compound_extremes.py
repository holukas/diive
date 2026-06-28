"""
=========================
Compound Extreme Detection
=========================

Classify time periods (months or days) into compound-extreme categories from the
standardized anomalies (z-scores) of two driver variables, using
:class:`~diive.analysis.compoundextremes.CompoundExtremes` (after Wang et al., Fig. 2).

The canonical use case is compound dry-hot detection: a period is an **atmospheric**
dryness extreme when vapour pressure deficit (VPD) is anomalously high, a **soil**
dryness extreme when soil water content (SWC) is anomalously low, and a **compound**
extreme when both occur together. Periods that cross neither threshold are *none*.

This example covers:

- The four-category classification and its ``.results`` / ``.counts`` outputs
- Both standardization modes (deseasonalized vs. whole-record)
- Monthly and daily resolution
- Per-variable extreme direction and thresholds

The companion plot is shown in ``examples/visualization/plot_compound_extremes.py``.

Best for: identifying anomalous dry/hot periods and screening for compound events.
"""

# %%
# Overview
# ^^^^^^^^
# Each period is aggregated to the chosen resolution, both variables are converted
# to z-scores, and a period is flagged extreme for a variable when its z-score
# crosses the threshold in that variable's extreme direction (high VPD, low SWC).
# The two flags combine into four categories: none, var1-only, var2-only, compound.

import diive as dv

data_df = dv.load_exampledata_parquet()
vpd = data_df['VPD_f'].copy()           # vapour pressure deficit
swc = data_df['SWC_FF0_0.15_1'].copy()  # soil water content

# %%
# Monthly compound extremes (deseasonalized z-scores)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``standardize_by='season'`` standardizes each month against the same calendar
# month across years, removing the seasonal cycle so genuinely anomalous months
# stand out. VPD's extreme is the high tail (``var1_extreme='high'``), SWC's is the
# low tail (``var2_extreme='low'``). The shared ``threshold=2.0`` means a z-score of
# +2 (VPD) or -2 (SWC) marks an extreme.

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

# %%
# Inspect the per-period results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``.results`` is one row per analyzed period: the aggregated values, both z-scores,
# the per-variable extreme flags, the fixed CATEGORY code, and the human LABEL.
# ``.labels`` returns the period names ('2018-07', ...) for annotation.

print(ce.results[['VPD_f_Z', 'SWC_FF0_0.15_1_Z', 'CATEGORY', 'LABEL']].head())
print("\nExtreme months only:")
print(ce.results.loc[ce.results['CATEGORY'] != 'none', ['LABEL', 'PERIOD']])

# %%
# Whole-record standardization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``standardize_by='record'`` uses a single mean/std over the whole record. Simpler,
# but the seasonal cycle of VPD dominates the z-score (summer months tend to flag),
# so the flagged set differs from the deseasonalized run above.

ce_record = dv.analysis.CompoundExtremes(
    var1=vpd, var2=swc, agg='monthly', threshold=2.0,
    standardize_by='record', var1_label='Air', var2_label='Soil',
)
print("Category counts (whole-record, monthly):")
print(ce_record.counts)

# %%
# Daily resolution
# ^^^^^^^^^^^^^^^^
# The same classification at daily resolution. With ``standardize_by='season'`` the
# z-scores are deseasonalized by day-of-year.

ce_daily = dv.analysis.CompoundExtremes(
    var1=vpd, var2=swc, agg='daily', threshold=2.0,
    standardize_by='season', var1_label='Air', var2_label='Soil',
)
print("Category counts (deseasonalized, daily):")
print(ce_daily.counts)

# %%
# Per-variable thresholds and directions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The method is generic: each variable has its own extreme direction and (optionally)
# its own threshold magnitude. Here we require a stronger VPD anomaly (2.5 sigma) than
# the SWC anomaly (1.5 sigma).

ce_custom = dv.analysis.CompoundExtremes(
    var1=vpd, var2=swc, agg='monthly', standardize_by='season',
    var1_extreme='high', var1_threshold=2.5,
    var2_extreme='low', var2_threshold=1.5,
    var1_label='Air', var2_label='Soil',
)
print("Category counts (asymmetric thresholds, monthly):")
print(ce_custom.counts)
