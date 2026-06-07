"""
DAILY AGGREGATION: ONE VALUE PER CALENDAR DAY
=============================================

Resample a (sub-daily) time series to daily resolution with
`dv.times.resample_to_daily_agg` -- e.g. a daily-mean time series over the
record. Days with too few records can be dropped via `mincounts_perc`.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

import diive as dv

df = dv.load_exampledata_parquet()
series = df["Tair_f"].copy()

print("Half-hourly input:")
print(f"  Records: {len(series)}")
print(f"  Period:  {series.index.min()} to {series.index.max()}")

# %%
# Daily mean
# ^^^^^^^^^^
#
# The default aggregation is the mean: one value per calendar day.

daily_mean = dv.times.resample_to_daily_agg(series, agg="mean")

print("\nDaily mean:")
print(f"  Days:  {len(daily_mean)}")
print(f"  Range: {daily_mean.min():.2f} to {daily_mean.max():.2f}")
print(daily_mean.head())

# %%
# Other aggregations
# ^^^^^^^^^^^^^^^^^^^
#
# Any method pandas' Series.agg accepts works (sum, median, min, max, std, ...).

daily_max = dv.times.resample_to_daily_agg(series, agg="max")
daily_min = dv.times.resample_to_daily_agg(series, agg="min")

print("\nDaily min/max air temperature:")
print(f"  Coldest day: {daily_min.min():.2f}")
print(f"  Warmest day: {daily_max.max():.2f}")

# %%
# Require complete days
# ^^^^^^^^^^^^^^^^^^^^^
#
# `mincounts_perc` drops days that are less complete than the given fraction of
# the fullest day (1.0 = only fully sampled days are kept).

complete_only = dv.times.resample_to_daily_agg(series, agg="mean", mincounts_perc=1.0)

print("\nDaily mean, fully sampled days only:")
print(f"  Days kept: {len(complete_only)} of {len(daily_mean)}")
