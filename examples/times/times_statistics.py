"""
TIME SERIES STATISTICS: QUICK STATISTICAL PROFILING
===================================================

Comprehensive summary of time series with percentiles, missing data assessment.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^
#
# Use meteorological data for demonstration.

import diive as dv

df = dv.load_exampledata_parquet()
series = df['Tair_f'].copy()

print("Example data loaded:")
print(f"  Records: {len(series)}")
print(f"  Period: {series.index.min()} to {series.index.max()}")
print(f"  Missing: {series.isna().sum()} ({100*series.isna().sum()/len(series):.1f}%)")

# %%
# Calculate comprehensive statistics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Get mean, median, standard deviation, variance, and percentiles.

stats = dv.sstats(series)

print("\nComprehensive statistics:")
print(stats)

# %%
# Understanding the output
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# `sstats()` returns a labeled table with:
#
# - **mean** — Average value
# - **median** — 50th percentile (central tendency, robust to outliers)
# - **sd** — Standard deviation (measure of spread)
# - **variance** — Variance (spread squared)
# - **p01, p05, p25, p75, p95, p99** — Percentiles (distribution shape)
# - **records** — Total data points
# - **valid** — Non-missing data points
# - **missing** — Missing value count
# - **start** — First timestamp
# - **end** — Last timestamp
# - **duration** — Time span

print("\nKey statistics:")
print(f"  Mean: {stats['mean']:.2f}°C")
print(f"  Median: {stats['median']:.2f}°C")
print(f"  Std Dev: {stats['sd']:.2f}°C")
print(f"  Range: {stats['p01']:.2f}°C to {stats['p99']:.2f}°C (1st-99th percentile)")

# %%
# Use case: Compare multiple variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate statistics for multiple columns simultaneously.

variables = ['Tair_f', 'RH', 'Rg_f']
print("\n" + "="*60)
print("Multi-variable statistics")
print("="*60)

for var in variables:
    if var in df.columns:
        s = df[var].copy()
        stats_var = dv.sstats(s)
        print(f"\n{var}:")
        print(f"  Mean: {stats_var['mean']:.2f}")
        print(f"  SD: {stats_var['sd']:.2f}")
        print(f"  Missing: {100*stats_var['missing']/stats_var['records']:.1f}%")

# %%
# Use case: Data quality assessment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Identify variables with excessive missing data.

print("\n" + "="*60)
print("Data quality assessment")
print("="*60)

missing_threshold = 5  # % missing data considered acceptable

for col in df.columns:
    s = df[col].copy()
    stats_col = dv.sstats(s)
    missing_pct = 100 * stats_col['missing'] / stats_col['records']

    status = "✓ Good" if missing_pct <= missing_threshold else "⚠ Poor"
    print(f"{col:<15} {missing_pct:5.1f}% missing  {status}")
