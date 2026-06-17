"""
====================
Manual Removal
====================

Explicitly flag known-bad records or date ranges as outliers, instead of
detecting them statistically. Selection is purely time-based, so it works even
when the bad values look plausible and a rule-based detector would miss them.

Typical use cases:

* Instrument malfunction, calibration, or maintenance windows recorded in a
  site logbook.
* Power outages, sensor swaps, or physical disturbances (mowing, grazing, snow
  on a sensor) whose timing is known.
* Records flagged as bad during visual inspection of plots.
* A final manual override step in a screening chain, removing records that
  survived the automatic detectors.
"""

# %%
# Load test data
# ^^^^^^^^^^^^^^
#
# Use air temperature data from summer 2018 (30-minute resolution).

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
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A full date(time) string removes exactly that one record. Useful for isolated
# measurement errors spotted during visual inspection.

# Timestamps must fall on the data's 30-min grid (:15 / :45) to match a record.
remove_dates_single = [
    '2018-07-05 10:45:00',
    '2018-07-12 14:15:00',
    '2018-07-20 09:15:00',
]

mr = dv.outliers.ManualRemoval(series=s, remove_dates=remove_dates_single, verbose=True).run()

# Results: the unified API exposes the cleaned series and the flag directly.
print("\nSingle timestamp removal:")
print(f"  Timestamps listed: {len(remove_dates_single)}")
print(f"  Records flagged (flag == 2): {(mr.flag == 2).sum()}")
print(f"  Valid records remaining: {mr.filteredseries.notna().sum()}")

# %%
# Remove a whole day with a bare date
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A date-only string covers the **entire day**, inclusive — handy for a
# maintenance day where the exact start/end times are not noted. Here the
# logbook says the sensor was serviced on 2018-07-10.

mr_day = dv.outliers.ManualRemoval(series=s, remove_dates=['2018-07-10'], verbose=True).run()

print("\nWhole-day removal ('2018-07-10'):")
print(f"  Records flagged: {(mr_day.flag == 2).sum()} (expected 48 for a full 30-min day)")

# %%
# Remove date ranges
# ^^^^^^^^^^^^^^^^^^
#
# A range is a **nested** ``[start, end]`` list (a flat list of two strings would
# be read as two single removals, not a range). The interval is closed on both
# ends, and bare dates again span whole days — so the period below covers all of
# 2018-07-02 through the end of 2018-07-04 (a full 3-day window).

remove_dates_ranges = [
    ['2018-07-02', '2018-07-04'],                    # 3 full days (power outage)
    ['2018-07-15 08:00:00', '2018-07-15 16:00:00'],  # a daytime window (calibration)
    ['2018-07-25', '2018-07-27'],                    # 3 full days (sensor swap)
]

mr_ranges = dv.outliers.ManualRemoval(series=s, remove_dates=remove_dates_ranges, verbose=True).run()

print("\nDate range removal:")
print(f"  Removal periods: {len(remove_dates_ranges)}")
print(f"  Records flagged: {(mr_ranges.flag == 2).sum()}")
print(f"  Valid records remaining: {mr_ranges.filteredseries.notna().sum()}")

# %%
# Mix single timestamps and ranges in one call
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A single ``remove_dates`` list may freely mix single-record strings and
# nested ``[start, end]`` ranges — the usual real-world case where a logbook
# lists both one-off spikes and extended outage periods.

remove_dates_mixed = [
    '2018-07-05 10:45:00',                           # one bad spike
    ['2018-07-02', '2018-07-04'],                    # an outage period
    '2018-07-20 09:15:00',                           # another bad spike
]

mr_mixed = dv.outliers.ManualRemoval(series=s, remove_dates=remove_dates_mixed, verbose=True).run()

print("\nMixed single + range removal:")
print(f"  Records flagged: {(mr_mixed.flag == 2).sum()}")
print(f"  Valid records remaining: {mr_mixed.filteredseries.notna().sum()}")

# %%
# Comparison
# ^^^^^^^^^^
#
# Single-record removal is precise for isolated errors; range removal handles
# extended problem periods. Both are reproducible — the exact dates live in the
# script, so the same input always yields the same cleaned series.

print("\nComparison (valid records):")
print(f"  Original:        {s.notna().sum()}")
print(f"  Single removal:  {mr.filteredseries.notna().sum()}")
print(f"  Whole-day:       {mr_day.filteredseries.notna().sum()}")
print(f"  Range removal:   {mr_ranges.filteredseries.notna().sum()}")
print(f"  Mixed removal:   {mr_mixed.filteredseries.notna().sum()}")
