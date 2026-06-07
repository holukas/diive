"""
DATE-RANGE SUBSELECTION: KEEP A TIME WINDOW
===========================================

Restrict a time series (or DataFrame) to a date range with
`dv.times.keep_daterange`. The selection is non-destructive: it returns a copy,
leaving the full record intact so you can keep it and revert. Either bound may
be left open.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

import diive as dv

df = dv.load_exampledata_parquet()

print("Full dataset:")
print(f"  Records: {len(df)}")
print(f"  Period:  {df.index.min()} to {df.index.max()}")

# %%
# Keep a closed window (both bounds, inclusive)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Bounds accept anything pandas can parse: a date, or a date with a time.

summer = dv.times.keep_daterange(df, start="2021-06-01", end="2021-08-31 23:30")

print("\nSummer 2021 subselection:")
print(f"  Records: {len(summer)}")
print(f"  Period:  {summer.index.min()} to {summer.index.max()}")

# The original is untouched -- keep it to revert at any time.
print(f"\nOriginal still has {len(df)} records (non-destructive).")

# %%
# Open-ended window (one bound only)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Omit a bound to leave that side open. Here: everything from 2022 onwards.

from_2022 = dv.times.keep_daterange(df, start="2022-01-01")

print("\nFrom 2022 onwards:")
print(f"  Records: {len(from_2022)}")
print(f"  Period:  {from_2022.index.min()} to {from_2022.index.max()}")

# %%
# Works on a single Series too
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

air_temp = dv.times.keep_daterange(df["Tair_f"], start="2021-12-01", end="2021-12-31 23:30")

print("\nAir temperature, December 2021:")
print(f"  Records: {len(air_temp)}")
print(f"  Mean:    {air_temp.mean():.2f}")
