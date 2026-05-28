"""
=============================
2D Grid Aggregation
=============================

Aggregate time series data over a 2D grid using different binning strategies.

Grid aggregation is useful for:
- Understanding relationships between two variables by binning a third
- Detecting patterns in 2D parameter space (e.g., radiation vs. temperature effects)
- Creating heatmap visualizations of aggregated data
- Comparing different binning methods (quantiles, equal-width, custom)

Best for: Exploring multivariate relationships and creating summary statistics
over 2D bins of measurement space.
"""

# %%
# Overview: Why Grid Aggregation?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Grid aggregation creates bins across two dimensions (X and Y) and aggregates
# a third variable (Z) within each bin. This reveals patterns that might be hidden
# in raw time series data.
#
# **Use cases:**
# 1. Understand how vapor pressure deficit (VPD) varies with radiation and temperature
# 2. Analyze ecosystem responses across multiple environmental gradients
# 3. Create reference surfaces (e.g., mean stomatal conductance vs. light and VPD)
# 4. Compare data distribution across environmental conditions

# %%
# Load Data and Prepare Variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import diive as dv

# Load example data
df = dv.load_exampledata_parquet()

# Extract subset: May-September daytime data only
vpd_col = 'VPD_f'      # Vapor pressure deficit (Z-axis)
ta_col = 'Tair_f'      # Air temperature (Y-axis)
swin_col = 'Rg_f'      # Shortwave radiation (X-axis)

subset = df[[vpd_col, ta_col, swin_col]].copy()
subset = subset.loc[(subset.index.month >= 5) & (subset.index.month <= 9)].copy()  # May-Sept
subset = subset[subset[swin_col] > 0].copy()  # Daytime only (radiation > 0)
subset = subset.dropna()

print(f"Data Summary:")
print(f"  Period: {subset.index.min().date()} to {subset.index.max().date()}")
print(f"  Records: {len(subset)}")
print(f"  Radiation: {subset[swin_col].min():.1f} to {subset[swin_col].max():.1f} W/m2")
print(f"  Temperature: {subset[ta_col].min():.1f} to {subset[ta_col].max():.1f} C")
print(f"  VPD: {subset[vpd_col].min():.2f} to {subset[vpd_col].max():.2f} hPa")

# %%
# Example 1: Quantile Binning
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Divide X and Y into equal-frequency bins (quantile-based).
# Each bin contains approximately the same number of records.

print(f"\n{'='*60}")
print(f"EXAMPLE 1: Quantile Binning (10 bins each axis)")
print(f"{'='*60}")
print(f"This divides data into bins with equal record counts.")
print(f"Useful for: Understanding patterns across equal-populated regions.\n")

qa = dv.analysis.GridAggregator(
    x=subset[swin_col],
    y=subset[ta_col],
    z=subset[vpd_col],
    binning_type='quantiles',
    n_bins=10,
    min_n_vals_per_bin=5,
    aggfunc='mean'
)

print(f"Aggregated data (wide format):")
print(qa.df_agg_wide)

print(f"\nData shape: {qa.df_agg_wide.shape}")
print(f"Non-missing cells: {qa.df_agg_wide.notna().sum().sum()} / {qa.df_agg_wide.size}")

# %%
# Example 2: Equal-Width Binning
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Divide X and Y into equal-width bins (uniform spacing).
# Each bin spans the same range of values.

print(f"\n{'='*60}")
print(f"EXAMPLE 2: Equal-Width Binning (10 bins each axis)")
print(f"{'='*60}")
print(f"This divides data into bins with equal value ranges.")
print(f"Useful for: Physical interpretation with consistent bin sizes.\n")

qe = dv.analysis.GridAggregator(
    x=subset[swin_col],
    y=subset[ta_col],
    z=subset[vpd_col],
    binning_type='equal_width',
    n_bins=10,
    min_n_vals_per_bin=5,
    aggfunc='mean'
)

print(f"Aggregated data (wide format):")
print(qe.df_agg_wide)

print(f"\nData shape: {qe.df_agg_wide.shape}")
print(f"Non-missing cells: {qe.df_agg_wide.notna().sum().sum()} / {qe.df_agg_wide.size}")

# %%
# Example 3: Custom Binning
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Define custom bin edges based on domain knowledge or data characteristics.

print(f"\n{'='*60}")
print(f"EXAMPLE 3: Custom Binning")
print(f"{'='*60}")
print(f"This uses user-specified bin edges for fine control.")
print(f"Useful for: Aligning bins with ecologically meaningful thresholds.\n")

# Create custom bins based on data ranges
custom_x_bins = list(range(0, 1300, 100))  # 0 to 1200 W/m2 in 100 W/m2 steps
custom_y_bins = list(range(-7, 30, 1))     # -7 to 29 C in 1 C steps

qc = dv.analysis.GridAggregator(
    x=subset[swin_col],
    y=subset[ta_col],
    z=subset[vpd_col],
    binning_type='custom',
    custom_x_bins=custom_x_bins,
    custom_y_bins=custom_y_bins,
    min_n_vals_per_bin=5,
    aggfunc='mean'
)

print(f"Aggregated data (wide format):")
print(qc.df_agg_wide)

print(f"\nData shape: {qc.df_agg_wide.shape}")
print(f"Non-missing cells: {qc.df_agg_wide.notna().sum().sum()} / {qc.df_agg_wide.size}")

# %%
# Accessing Results
# ^^^^^^^^^^^^^^^^^
# The GridAggregator provides multiple output formats.

print(f"\n{'='*60}")
print(f"Output Formats")
print(f"{'='*60}")

# Wide format: rows=Y bins, columns=X bins, values=aggregated Z
print(f"\nWide format (rows=temperature, cols=radiation):")
print(f"  Shape: {qa.df_agg_wide.shape}")
print(f"  Use for: Matrix operations, heatmap visualization")

# Long format: each row is one bin combination
print(f"\nLong format (one row per bin combination):")
df_long = qa.df_agg_long
print(df_long.head(10))
print(f"  Shape: {df_long.shape}")
print(f"  Use for: Direct data export, statistical analysis")

# Non-aggregated long format: includes original data with bin assignments
print(f"\nNon-aggregated long format (all original records with bin assignments):")
df_data = qa.df_long
print(df_data.head(5))
print(f"  Shape: {df_data.shape}")
print(f"  Columns: {df_data.columns.tolist()}")
print(f"  Use for: Understanding bin composition, detailed analysis")

# %%
# Comparing Binning Methods
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Different binning strategies reveal different patterns.

print(f"\n{'='*60}")
print(f"Summary: Binning Method Comparison")
print(f"{'='*60}")

print(f"\n1. Quantile binning:")
print(f"   Grid shape: {qa.df_agg_wide.shape}")
print(f"   Bin interpretation: Equal number of records per bin")
print(f"   Best for: Detecting patterns in data-rich regions")

print(f"\n2. Equal-width binning:")
print(f"   Grid shape: {qe.df_agg_wide.shape}")
print(f"   Bin interpretation: Uniform value spacing")
print(f"   Best for: Physical interpretation and reproducibility")

print(f"\n3. Custom binning:")
print(f"   Grid shape: {qc.df_agg_wide.shape}")
print(f"   Bin interpretation: User-specified thresholds")
print(f"   Best for: Aligning with domain-specific boundaries")

print(f"\nKey Insights:")
print(f"• Quantile binning distributes records evenly across bins")
print(f"• Equal-width binning may have uneven record distribution")
print(f"• Custom binning allows domain-specific flexibility")
print(f"• Output includes both aggregated and original data")
