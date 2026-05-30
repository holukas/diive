"""
=============================
Histogram Distribution Analysis
=============================

Histogram calculation and distribution analysis for time series data
using different binning strategies and fringe bin removal.

This example demonstrates the Histogram class with four flexible approaches:

- **Fixed number of bins** — Coarse-grained overview with even bin widths
- **Fixed bins with fringe removal** — Remove edge bins to avoid boundary accumulation
- **Separate bin for each unique value** — Fine-grained analysis without binning assumptions
- **Unique values with fringe removal** — Exclude extreme bins for cleaner central distribution

Fringe bin removal is useful when data accumulates unnaturally at measurement
boundaries (min/max limits, detection thresholds), creating spurious peaks that
can mask the true distribution pattern.

Best for: Understanding data distributions, identifying genuine peaks, and analyzing
time series measurements with various histogram strategies.
"""

# %%
# Overview: Why Histograms Matter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Histograms bin time series data to reveal distribution patterns.
# The histogram can be calculated in 4 ways:
#
# 1. **Fixed number of bins** — Divides data range into N equal-width bins
# 2. **Fixed bins + fringe removal** — Same, but excludes edge bins
# 3. **Unique value bins** — One bin per unique value (no assumptions about ranges)
# 4. **Unique values + fringe removal** — Fine-grained analysis without edge artifacts
#
# **Why fringe bin removal?** Some measurements accumulate at boundaries
# (detection limits, sensor saturation, absolute threshold) creating false peaks
# that mask the true distribution. Removing these edge bins reveals the genuine signal.

# %%
# Load data and prepare time series
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We'll analyze NEE (CO2 net ecosystem exchange) flux distribution from the example dataset.

import diive as dv

data_df = dv.load_exampledata_parquet()

# Extract flux series for analysis
series = data_df['NEE_CUT_REF_f'].copy()
print(f"Time series shape: {series.shape}")
print(f"Series name: {series.name}")
print(f"Valid values: {series.notna().sum()}, Missing: {series.isna().sum()}")

# %%
# Method 1: Fixed number of bins
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Calculate histogram with specific number of bins (10 bins).
# Each bin covers equal width across the data range.

hist1 = dv.analysis.Histogram(
    series=series,
    method='n_bins',
    n_bins=10,
    ignore_fringe_bins=None
)

print("\n--- Method 1: 10 fixed-width bins ---")
print(hist1.results)
print(f"\nTop 5 peak bins: {hist1.peakbins}")

# %%
# Method 2: Fixed bins with fringe bin exclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Same as Method 1 but exclude the first bin and last 2 bins.
# This is useful when data accumulates at boundaries due to measurement limits.

hist2 = dv.analysis.Histogram(
    series=series,
    method='n_bins',
    n_bins=10,
    ignore_fringe_bins=[1, 2]
)

print("\n--- Method 2: 10 bins, excluding first and last 2 bins ---")
print(hist2.results)
print(f"\nTop 5 peak bins: {hist2.peakbins}")

# %%
# Method 3: Separate bin for each unique value
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Create one bin per unique value in the dataset.
# This provides fine-grained distribution without binning assumptions.

hist3 = dv.analysis.Histogram(
    series=series,
    method='uniques',
    ignore_fringe_bins=None
)

print("\n--- Method 3: One bin per unique value (all bins) ---")
print(f"Number of unique values: {len(hist3.results)}")
print("\nFirst 15 bins:")
print(hist3.results.head(15))
print(f"\nTop 5 peak bins: {hist3.peakbins}")

# %%
# Method 4: Unique values with fringe bin exclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Same as Method 3 but exclude first 8 and last 20 unique value bins.
# This removes extreme outliers from the distribution analysis.

hist4 = dv.analysis.Histogram(
    series=series,
    method='uniques',
    ignore_fringe_bins=[8, 20]
)

print("\n--- Method 4: Unique value bins, excluding first 8 and last 20 bins ---")
print(f"Number of bins after exclusion: {len(hist4.results)}")
print(hist4.results)
print(f"\nTop 5 peak bins: {hist4.peakbins}")

# %%
# Interpreting histogram results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Understanding what the bin ranges and peaks mean.

print("\n" + "="*60)
print("INTERPRETING HISTOGRAM RESULTS")
print("="*60)

# For Method 3 (all unique value bins)
peak_val = hist3.peakbins[0]
# Find the bin in results to determine width
peak_idx = hist3.results[hist3.results['BIN_START_INCL'] == peak_val].index[0]
if peak_idx + 1 < len(hist3.results):
    next_bin = hist3.results['BIN_START_INCL'].iloc[peak_idx + 1]
    bin_width = next_bin - peak_val
else:
    bin_width = (hist3.results['BIN_START_INCL'].iloc[1] -
                 hist3.results['BIN_START_INCL'].iloc[0])

peak_count = hist3.results[hist3.results['BIN_START_INCL'] == peak_val]['COUNTS'].values[0]

print(f"\nMethod 3 Analysis (Unique Values):")
print(f"  Peak distribution found at bin {peak_val:.4f}")
print(f"  - This bin contains {peak_count} counts")
print(f"  - Top 5 peak bins: {hist3.peakbins}")
print(f"  - Bin width: {bin_width:.4f}")
print(f"  - This means the peak bin covers values from {peak_val:.4f} (inclusive)")
print(f"    to {peak_val + bin_width:.4f} (exclusive)")

print(f"\nMethod 4 Analysis (Unique Values with Fringe Removal):")
print(f"  Removed first 8 and last 20 bins")
print(f"  - Bins remaining: {len(hist4.results)} (from {len(hist3.results)} original)")
print(f"  - New peak: {hist4.peakbins[0]:.4f}")
print(f"  - This excludes extreme values and provides a cleaner view")
print(f"    of the central distribution without boundary effects")

# %%
# Compare all methods
# ^^^^^^^^^^^^^^^^^^^
# Summary statistics across the different histogram methods.

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"Original series - Valid: {series.notna().sum()}, Missing: {series.isna().sum()}")
print(f"  Mean: {series.mean():.4f}, Median: {series.median():.4f}")
print(f"  Std Dev: {series.std():.4f}")
print(f"  Min: {series.min():.4f}, Max: {series.max():.4f}")

print(f"\nMethod 1 (10 fixed-width bins): Peak at {hist1.peakbins[0]:.4f}")
print(f"Method 2 (10 bins, fringe excluded): Peak at {hist2.peakbins[0]:.4f}")
print(f"Method 3 (unique values): {len(hist3.results)} bins, Peak at {hist3.peakbins[0]:.4f}")
print(f"Method 4 (unique values, fringe excluded): {len(hist4.results)} bins, Peak at {hist4.peakbins[0]:.4f}")

print(f"\nWhen to use each method:")
print(f"  Method 1-2: Quick overview with coarse binning, robust to outliers")
print(f"  Method 3-4: Fine-grained analysis, better for identifying true peaks")
