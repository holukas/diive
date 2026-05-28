"""
==================
Z-Score Detection
==================

Statistical outlier detection using standard deviation thresholds.
Supports global, day/night, rolling window, and increment-based approaches.
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^
#
# Use temperature data for demonstration (single year for faster execution).

import diive as dv

df = dv.load_exampledata_parquet()
series = df['Tair_f'].copy()
series = series.loc[series.index.year == 2018].copy()  # Use single year

print("Example data loaded:")
print(f"  Records: {len(series)}")
print(f"  Valid: {series.notna().sum()}")
print(f"  Range: {series.min():.2f} to {series.max():.2f}°C")

# %%
# Global Z-Score
# ^^^^^^^^^^^^^^^
#
# Single threshold applied across entire time series.
# Simple and fast, but ignores time-of-day variation.

detector_global = dv.outliers.zScore(
    series=series,
    thres_zscore=2.0,
    showplot=True,
    verbose=1
)
detector_global.calc(repeat=False)

flag_global = detector_global.get_flag()
print(f"\nGlobal z-score (threshold=2.0):")
print(f"  Outliers: {(flag_global == 2).sum()}")
print(f"  Data retained: {(flag_global != 2).sum()}")

# %%
# Day/Night Separated Z-Score
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Calculate z-scores separately for daytime and nighttime records.
# Accounts for different signal characteristics across the day.

detector_dtnt = dv.outliers.zScore(
    series=series,
    separate_daytime_nighttime=True,
    lat=47.286417,
    lon=7.733750,
    utc_offset=1,
    thres_zscore=2.5,
    showplot=True,
    verbose=1
)
detector_dtnt.calc(repeat=True)

flag_dtnt = detector_dtnt.get_flag()
print(f"\nDay/night z-score (threshold=2.5):")
print(f"  Outliers: {(flag_dtnt == 2).sum()}")
print(f"  Data retained: {(flag_dtnt != 2).sum()}")

# %%
# Rolling Window Z-Score
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Adaptive threshold based on rolling mean and standard deviation.
# Better for non-stationary data where signal characteristics change over time.

detector_rolling = dv.outliers.zScoreRolling(
    series=series,
    thres_zscore=2.5,
    winsize=48,  # 24 hours at 30-min resolution
    showplot=True,
    verbose=1
)
detector_rolling.calc(repeat=False)

flag_rolling = detector_rolling.get_flag()
print(f"\nRolling z-score (threshold=2.5, window=48 records):")
print(f"  Outliers: {(flag_rolling == 2).sum()}")
print(f"  Data retained: {(flag_rolling != 2).sum()}")

# %%
# Increment-Based Z-Score
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Detects outliers as abrupt changes from one record to the next.
# Useful for identifying instrumental spikes or sudden shifts.

detector_incr = dv.outliers.zScoreIncrements(
    series=series,
    thres_zscore=2.5,
    showplot=True,
    verbose=1
)
detector_incr.calc(repeat=False)

flag_incr = detector_incr.get_flag()
print(f"\nIncrement z-score (threshold=2.5):")
print(f"  Outliers: {(flag_incr == 2).sum()}")
print(f"  Data retained: {(flag_incr != 2).sum()}")

# %%
# Method Comparison
# ^^^^^^^^^^^^^^^^^
#
# Summary of detection results from each approach.

print("\n" + "=" * 50)
print("Comparison of Z-Score Methods")
print("=" * 50)
print(f"Original valid records: {series.notna().sum()}")
print(f"Global: {(flag_global != 2).sum()} retained ({100 * (flag_global != 2).sum() / series.notna().sum():.1f}%)")
print(f"Day/Night: {(flag_dtnt != 2).sum()} retained ({100 * (flag_dtnt != 2).sum() / series.notna().sum():.1f}%)")
print(f"Rolling: {(flag_rolling != 2).sum()} retained ({100 * (flag_rolling != 2).sum() / series.notna().sum():.1f}%)")
print(f"Increments: {(flag_incr != 2).sum()} retained ({100 * (flag_incr != 2).sum() / series.notna().sum():.1f}%)")

# %%
# Parameter tuning: threshold sensitivity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Vary thres_zscore to understand sensitivity of global z-score approach.
# Lower thresholds detect more outliers (stricter), higher thresholds are more permissive.

print("\n" + "=" * 50)
print("Threshold Sensitivity Analysis")
print("=" * 50)

thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

for thresh in thresholds:
    detector = dv.outliers.zScore(
        series=series,
        thres_zscore=thresh,
        showplot=False,
        verbose=0
    )
    detector.calc(repeat=False)

    flag = detector.get_flag()
    outliers = (flag == 2).sum()
    retained = (flag != 2).sum()
    outlier_rate = 100 * outliers / series.notna().sum()

    print(f"\nThreshold {thresh}:")
    print(f"  Outliers detected: {outliers} ({outlier_rate:.2f}%)")
    print(f"  Data retained: {retained} ({100 * retained / series.notna().sum():.1f}%)")

# %%
# Choosing the right threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **Guidelines for z-score threshold selection:**
#
# - **2.0-2.5:** Strict QC, catch more anomalies (recommended for clean environments)
# - **3.0:** Balanced (typical for meteorological data)
# - **3.5-4.0:** Permissive, only catch extreme values
# - **4.0+:** Very permissive, only catch rare spikes
#
# Consider data distribution and domain requirements when choosing threshold.

print("\nThreshold Selection Guide:")
print("  Strict (2.0): Catch most anomalies")
print("  Moderate (2.5-3.0): Typical for meteorological data")
print("  Permissive (3.5-4.0): Only catch extreme values")
