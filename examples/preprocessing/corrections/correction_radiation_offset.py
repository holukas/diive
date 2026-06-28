"""
=========================================
Correct Radiation Zero Offset (Nighttime)
=========================================

Fix solar radiation measurements with nighttime offset. Uses solar geometry
to identify nighttime and corrects non-zero radiation readings during night.
Nighttime values are set to zero after offset correction.
"""

# %%
# Load radiation data and define site location
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Solar radiation should be zero at night. However, sensors sometimes
# show small non-zero values during nighttime due to calibration drift
# or electronic noise. We correct this using solar geometry.

import diive as dv
from diive.configs.exampledata import load_exampledata_parquet_meteo_swin_1MIN

df = load_exampledata_parquet_meteo_swin_1MIN()
series = df['SW_IN_T1_47_1'].copy()

# Site location (CH-LAE - Laegeren)
SITE_LAT = 47.478333
SITE_LON = 8.364389
UTC_OFFSET = 1

print("Original radiation data:")
print(f"  Range: {series.min():.2f} to {series.max():.2f} W/m²")

# %%
# Apply radiation zero offset correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The correction:
# 1. Calculates sunrise and sunset using solar geometry
# 2. Identifies nighttime periods
# 3. Calculates mean radiation during night as the offset
# 4. Subtracts this offset and sets nighttime to zero

series_corr = dv.corrections.remove_nighttime_zero_offset(
    series=series,
    lat=SITE_LAT,
    lon=SITE_LON,
    utc_offset=UTC_OFFSET,
    showplot=True
)

print("\nCorrected radiation data:")
print(f"  Range: {series_corr.min():.2f} to {series_corr.max():.2f} W/m²")

# %%
# Verify nighttime correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Check that radiation is properly zeroed at night after correction.

nighttime_mean_orig = series[series.index.hour < 6].mean()
nighttime_mean_corr = series_corr[series_corr.index.hour < 6].mean()

print(f"\nNighttime (00:00-06:00) radiation:")
print(f"  Original: {nighttime_mean_orig:.3f} W/m²")
print(f"  Corrected: {nighttime_mean_corr:.3f} W/m²")
print(f"  Offset removed: {(nighttime_mean_orig - nighttime_mean_corr):.3f} W/m²")
