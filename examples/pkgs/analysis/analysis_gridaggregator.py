"""
===================================
Grid Aggregation and Binning
===================================

Spatial aggregation of point measurements across two driver variables.

Demonstrates grid aggregation: bins ecosystem flux data across two driver
variables (VPD and soil water content) using custom bin edges, then
aggregates flux values in each bin. Reveals relationships and patterns
across environmental gradients.

Best for: Multidimensional binning and aggregation workflows
"""

# %%
# Load data
# ^^^^^^^^^

import diive as dv
from scipy.stats import zscore

df = dv.load_exampledata_parquet()

# Select variables for analysis
vpd_col = 'VPD_f'           # Vapor pressure deficit
swc_col = 'SWC_FF0_0.15_1'  # Soil water content
flux_col = 'NEE_CUT_REF_f'  # Net ecosystem productivity

subset = df[[flux_col, vpd_col, swc_col]].copy()

# %%
# Prepare data
# ^^^^^^^^^^^^

# Filter to growing season (May-October) and remove NaN
subset = subset.loc[(subset.index.month >= 5) & (subset.index.month <= 10)].copy()
subset = subset.dropna()

# Convert to z-scores for standardized comparison
subset = subset.apply(lambda x: zscore(x, nan_policy='omit'))

# %%
# Grid aggregation with custom bins
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Create grid aggregation with custom bins
ga = dv.gridaggregator(
    x=subset[vpd_col],
    y=subset[swc_col],
    z=subset[flux_col],
    binning_type='custom',
    custom_x_bins=list(range(-3, 5, 1)),  # VPD bins from -3 to 5
    custom_y_bins=list(range(-3, 5, 1)),  # SWC bins from -3 to 5
    min_n_vals_per_bin=5,
    aggfunc='mean'
)

# %%
# Results in wide format
# ^^^^^^^^^^^^^^^^^^^^^^

print("Grid aggregation results (wide format):")
print("Rows: SWC bins | Columns: VPD bins")
print(ga.df_agg_wide)

# %%
# Results in long format
# ^^^^^^^^^^^^^^^^^^^^^^

print("\n\nGrid aggregation results (long format):")
print(ga.df_agg_long)

# %%
# Data coverage
# ^^^^^^^^^^^^^

# Count non-NaN values in aggregated results
n_bins_with_data = ga.df_agg_long[flux_col].notna().sum()
print(f"\n\nBins with data: {n_bins_with_data} out of {len(ga.df_agg_long)} bins")
print(f"Coverage: {100 * n_bins_with_data / len(ga.df_agg_long):.1f}%")
