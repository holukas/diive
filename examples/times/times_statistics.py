"""
TIME SERIES STATISTICS: QUICK STATISTICAL PROFILING
===================================================

Calculate mean, median, SD, variance, and percentiles for time series data.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

import diive as dv

df = dv.load_exampledata_parquet()
series = df['Tair_f'].copy()

print("Example data loaded:")
print(f"  Records: {len(series)}")
print(f"  Period: {series.index.min()} to {series.index.max()}")
print(f"  Missing: {series.isna().sum()} rows")

# %%
# Calculate statistics
# ^^^^^^^^^^^^^^^^^^^^

stats = dv.sstats(series)

col = series.name

print("\n" + "=" * 60)
print("Statistics")
print("=" * 60)
print(f"Mean: {stats.loc['MEAN', col]:.2f}°C")
print(f"Median: {stats.loc['MEDIAN', col]:.2f}°C")
print(f"Std Dev: {stats.loc['SD', col]:.2f}°C")
print(f"Range (P01-P99): {stats.loc['P01', col]:.2f} to {stats.loc['P99', col]:.2f}°C")
print(f"Valid records: {int(stats.loc['NOV', col])} / {int(stats.loc['NOV', col] + stats.loc['MISSING', col])}")

# %%
# Output fields
# ^^^^^^^^^^^^^
#
# Returns 30 statistics including:
# - **MEAN, MEDIAN** — Central tendency
# - **SD, VARIANCE** — Spread
# - **P01, P05, P25, P75, P95, P99** — Percentiles
# - **NOV, MISSING** — Data counts
# - **STARTDATE, ENDDATE, PERIOD** — Time span

print("\nFull output:")
print(stats)

# %%
# Compare multiple variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

print("\n" + "=" * 60)
print("Multi-variable summary")
print("=" * 60)

for var in ['Tair_f', 'RH', 'Rg_f']:
    if var in df.columns:
        s = df[var].copy()
        st = dv.sstats(s)
        mean_val = st.loc['MEAN', var]
        sd_val = st.loc['SD', var]
        missing_pct = st.loc['MISSING_PERC', var]
        print(f"{var:<12} mean={mean_val:7.2f}  sd={sd_val:6.2f}  missing={missing_pct:5.1f}%")
