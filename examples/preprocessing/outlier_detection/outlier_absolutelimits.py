"""
====================
Absolute Limits
====================

Enforce physical or measurement validity constraints by rejecting
values outside specified min/max ranges.
"""

# %%
# Create synthetic test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate random values between 0 and 100 with half-hourly frequency.

import numpy as np
import pandas as pd
import diive as dv

np.random.seed(100)
rows = 1000
data = np.random.rand(rows) * 100
tidx = pd.date_range('2019-01-01 00:30:00', periods=rows, freq='30min')
series = pd.Series(data, index=tidx, name='TESTDATA')

print("Test data created:")
print(f"  Records: {len(series)}")
print(f"  Range: {series.min():.2f} to {series.max():.2f}")
print(f"  Mean: {series.mean():.2f}")

# %%
# Basic absolute limits
# ^^^^^^^^^^^^^^^^^^^^^
#
# Single min/max range applied to all records.
# Rejects values < 6 or > 74.

al_basic = dv.outliers.AbsoluteLimits(
    series=series,
    minval=6,
    maxval=74,
    idstr='basic',
    showplot=False,
    verbose=1
)
al_basic.calc()

filtered_basic = al_basic.filteredseries
flag_basic = al_basic.flag

print("\nBasic absolute limits (6-74):")
print(f"  Outliers rejected: {(flag_basic == 2).sum()}")
print(f"  Valid records: {filtered_basic.notna().sum()}")
print(f"  Filtered range: {filtered_basic.min():.2f} to {filtered_basic.max():.2f}")

# %%
# Day/Night separated limits
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Different min/max ranges for daytime and nighttime.
# Useful when measurement characteristics differ across the day.

data_dn = np.random.rand(rows) * 100
tidx_dn = pd.date_range('2019-01-01 00:30:00', periods=rows, freq='30min')
series_dn = pd.Series(data_dn, index=tidx_dn, name='flux')

al_dtnt = dv.outliers.AbsoluteLimitsDaytimeNighttime(
    series=series_dn,
    daytime_minmax=[6.2, 74.9],   # Wider range for daytime
    nighttime_minmax=[29.5, 47.4],  # Narrower range for nighttime
    idstr='dtnt',
    lat=47.286417,
    lon=7.733750,
    utc_offset=1,
    showplot=False,
    verbose=1
)
al_dtnt.calc()

filtered_dtnt = al_dtnt.filteredseries
flag_dtnt = al_dtnt.flag

print("\nDay/Night absolute limits:")
print(f"  Daytime: 6.2 to 74.9")
print(f"  Nighttime: 29.5 to 47.4")
print(f"  Total outliers: {(flag_dtnt == 2).sum()}")
print(f"  Valid records: {filtered_dtnt.notna().sum()}")

# Breakdown by time of day
daytime_outliers = ((flag_dtnt == 2) & al_dtnt.is_daytime).sum()
nighttime_outliers = ((flag_dtnt == 2) & al_dtnt.is_nighttime).sum()
daytime_total = al_dtnt.is_daytime.sum()
nighttime_total = al_dtnt.is_nighttime.sum()

print(f"\nOutliers by time of day:")
print(f"  Daytime: {daytime_outliers}/{daytime_total} ({100*daytime_outliers/daytime_total:.1f}%)")
print(f"  Nighttime: {nighttime_outliers}/{nighttime_total} ({100*nighttime_outliers/nighttime_total:.1f}%)")

# %%
# Comparison
# ^^^^^^^^^^
#
# Day/night separation allows stricter filtering during stable nighttime periods
# while preserving more variable daytime data.

print("\nComparison:")
print(f"Original valid: {series.notna().sum()}")
print(f"Basic filtering: {filtered_basic.notna().sum()} retained ({100*filtered_basic.notna().sum()/series.notna().sum():.1f}%)")
print(f"Day/Night filtering: {filtered_dtnt.notna().sum()} retained ({100*filtered_dtnt.notna().sum()/series_dn.notna().sum():.1f}%)")
