"""
====================
Manual Removal
====================

Explicitly remove data points or date ranges flagged as problematic.
Useful for known equipment failures, maintenance periods, or measurement errors.
"""

# %%
# Load test data
# ^^^^^^^^^^^^^^
#
# Use temperature data from summer 2018.

import diive as dv

df = dv.load_exampledata_parquet()
s = df['Tair_f'].copy()
s = s.loc[s.index.year == 2018].copy()
s = s.loc[s.index.month == 7].copy()

print("Test data loaded:")
print(f"  Records: {len(s)}")
print(f"  Valid: {s.notna().sum()}")
print(f"  Period: {s.index.min()} to {s.index.max()}")

# %%
# Remove individual timestamps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Specify exact timestamps to remove for single record precision.
# Useful for individual measurement errors.

remove_dates_single = [
    '2018-07-05 10:30:00',
    '2018-07-12 14:15:00',
    '2018-07-20 09:00:00'
]

mr_single = dv.ManualRemoval(
    series=s,
    remove_dates=remove_dates_single,
    showplot=False,
    verbose=1
)

mr_single.calc()

flag_single = mr_single.get_flag()
filtered_single = s.copy()
filtered_single.loc[flag_single == 2] = None

print("\nSingle timestamp removal:")
print(f"  Timestamps removed: {len(remove_dates_single)}")
print(f"  Records removed: {(flag_single == 2).sum()}")
print(f"  Valid records remaining: {filtered_single.notna().sum()}")

# %%
# Remove date ranges
# ^^^^^^^^^^^^^^^^^^
#
# Specify start and end dates to remove entire periods.
# Effective for maintenance windows, power outages, or known drift periods.

remove_dates_ranges = [
    ['2018-07-02', '2018-07-04'],  # 3-day period
    ['2018-07-15 08:00:00', '2018-07-15 16:00:00'],  # Daytime window
    ['2018-07-25', '2018-07-27']  # Multi-day period
]

mr_ranges = dv.ManualRemoval(
    series=s,
    remove_dates=remove_dates_ranges,
    showplot=False,
    verbose=1
)

mr_ranges.calc()

flag_ranges = mr_ranges.get_flag()
filtered_ranges = s.copy()
filtered_ranges.loc[flag_ranges == 2] = None

print("\nDate range removal:")
print(f"  Removal periods: {len(remove_dates_ranges)}")
print(f"  Records removed: {(flag_ranges == 2).sum()}")
print(f"  Valid records remaining: {filtered_ranges.notna().sum()}")

# %%
# Comparison
# ^^^^^^^^^^
#
# Single-point removal is precise for isolated errors.
# Range removal is effective for extended problem periods.

print("\nComparison:")
print(f"Original valid records: {s.notna().sum()}")
print(f"After single removal: {filtered_single.notna().sum()} ({100*filtered_single.notna().sum()/s.notna().sum():.1f}%)")
print(f"After range removal: {filtered_ranges.notna().sum()} ({100*filtered_ranges.notna().sum()/s.notna().sum():.1f}%)")
