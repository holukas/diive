"""
=====================================
Correct Relative Humidity Saturation
=====================================

Fix relative humidity measurements that exceed 100% (physically impossible).
The correction detects excess values, calculates daily mean offset,
and caps the maximum at 100%.
"""

# %%
# Load and scale RH data
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Real-world RH measurements sometimes exceed 100% due to sensor
# calibration drift or processing artifacts. We load example RH data
# and scale it to simulate this common issue.

import diive as dv
from diive.configs.exampledata import load_exampledata_parquet_meteo_rh_1MIN

df = load_exampledata_parquet_meteo_rh_1MIN()
series = df['RH_T1_47_1'].copy()

# Scale to simulate high RH values exceeding 100%
series = series.multiply(1.4)

print("Original RH data:")
print(f"  Range: {series.min():.2f}% to {series.max():.2f}%")
print(f"  Values exceeding 100%: {(series > 100).sum()}")

# %%
# Apply RH offset correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The correction identifies days with excess values, calculates
# the mean excess as the offset, and subtracts it from all data.
# Values are then capped at 100%.

series_corr = dv.corrections.remove_relativehumidity_offset(series=series, showplot=True)

print("\nCorrected RH data:")
print(f"  Range: {series_corr.min():.2f}% to {series_corr.max():.2f}%")
print(f"  Values exceeding 100% after correction: {(series_corr > 100).sum()}")

# %%
# Summary
# ^^^^^^^
#
# The correction successfully removed the saturation issue while
# preserving the underlying signal structure. Daily variability
# and long-term patterns remain intact.

print(f"\nCorrection applied:")
print(f"  Mean offset removed: {(series - series_corr).mean():.2f}%")
