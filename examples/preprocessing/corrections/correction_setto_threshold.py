"""
=====================================
Clip Values to Minimum/Maximum Limits
=====================================

Constrain data to physically realistic limits by clipping values
above or below specified thresholds.
"""

# %%
# Create test data with values outside acceptable range
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate a series with values that exceed expected bounds.
# In real applications, these might be measurement errors or sensor artifacts.

import pandas as pd
import diive as dv

series = pd.Series([0.5, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
series.name = "normalized_value"

print("Original data:")
print(f"  Values: {series.values}")
print(f"  Range: {series.min()} to {series.max()}")

# %%
# Apply maximum threshold
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Clip all values above 3.0 to exactly 3.0.
# Values below the threshold remain unchanged.

series_corr_max = dv.corrections.setto_threshold(
    series=series.copy(),
    threshold=3.0,
    type='max',
    showplot=False
)

print("\nAfter max threshold 3.0:")
print(f"  Values: {series_corr_max.values}")
print(f"  New range: {series_corr_max.min()} to {series_corr_max.max()}")
print(f"  Values clipped: {(series > 3.0).sum()}")

# %%
# Apply minimum threshold
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Clip all values below 1.0 to exactly 1.0.
# Values above the threshold remain unchanged.

series_corr_min = dv.corrections.setto_threshold(
    series=series.copy(),
    threshold=1.0,
    type='min',
    showplot=False
)

print("\nAfter min threshold 1.0:")
print(f"  Values: {series_corr_min.values}")
print(f"  New range: {series_corr_min.min()} to {series_corr_min.max()}")
print(f"  Values clipped: {(series < 1.0).sum()}")
