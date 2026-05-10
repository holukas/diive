"""
=============================
Distribution Analysis via Histograms
=============================

Histogram calculation and distribution analysis for time series data.

Demonstrates histogram calculation with different binning strategies
and fringe bin removal for cleaner distribution analysis of time series
flux measurements.

Best for: Understanding data distributions and identifying outliers
"""

# %%
# Load data
# ^^^^^^^^^

import diive as dv

data_df = dv.load_exampledata_parquet()

# Select a variable
series = data_df['NEE_CUT_REF_f'].copy()

# %%
# Create histogram with n_bins method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

hist = dv.Histogram(
    s=series,
    method='n_bins',
    n_bins=20,
    ignore_fringe_bins=[1, 1]  # Remove first and last bin
)

# %%
# Access results
# ^^^^^^^^^^^^^^

print("Histogram results:")
print(hist.results)

# Get peak bins (top 5 bins with most counts)
print("\nPeak bins (top 5 by count):")
print(hist.peakbins)

# %%
# Distribution statistics
# ^^^^^^^^^^^^^^^^^^^^^^^

print(f"\nDistribution statistics:")
print(f"  Mean: {series.mean():.3f}")
print(f"  Median: {series.median():.3f}")
print(f"  Std Dev: {series.std():.3f}")
print(f"  Min: {series.min():.3f}")
print(f"  Max: {series.max():.3f}")
print(f"  Valid values: {series.notna().sum()}")
print(f"  Missing values: {series.isna().sum()}")
