"""
DIURNAL CYCLE ANALYSIS: TIME-OF-DAY PATTERNS
============================================

Extract and analyze hourly patterns in time series data, showing seasonal variation.

See Also:
    examples/visualization/dielcycle.py — Plotting diel cycles with the DielCycle class

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

import diive as dv

df = dv.load_exampledata_parquet()
series = df['Tair_f'].copy()

print("Example data:")
print(f"  Records: {len(series)}")
print(f"  Period: {series.index[0]} to {series.index[-1]}")

# %%
# Annual diel cycle
# ^^^^^^^^^^^^^^^^^
#
# Calculate mean and standard deviation for each hour of day.

from diive.core.times.resampling import diel_cycle

diel_annual = diel_cycle(
    series=series,
    mean=True,
    std=True,
    each_month=False
)

print("\n" + "=" * 60)
print("Annual diel cycle (each_month=False)")
print("=" * 60)
print(f"Shape: {diel_annual.shape}")
print(f"\nOutput:")
print(diel_annual)

# %%
# Monthly diel cycles
# ^^^^^^^^^^^^^^^^^^^
#
# Calculate separate pattern for each month.

diel_monthly = diel_cycle(
    series=series,
    mean=True,
    std=True,
    each_month=True
)

print("\n" + "=" * 60)
print("Monthly diel cycles (each_month=True)")
print("=" * 60)
print(f"Shape: {diel_monthly.shape}")
print(f"Index levels: (month 1-12, time of day)")
print(f"\nJuly (month 7) data:")
print(diel_monthly.loc[7])
