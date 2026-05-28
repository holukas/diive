"""
==================================
Set Exact Values to Missing (NaN)
==================================

Remove error codes and sentinel values by setting specific measurements to NaN.
Useful for removing known error codes or invalid flags from raw data.
"""

# %%
# Create test data with sentinel values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We create a simple series with some sentinel/error values that need to be removed.
# In real data, these might be error codes (e.g., -999, 0, 999) or other flags
# indicating measurement failures.

import pandas as pd
import diive as dv

series = pd.Series([1, 2, 0, 4, 5, 6, 7, 0, 9, 10])
series.name = "measurements"

print("Original data:")
print(f"  Values: {series.values}")
print(f"  Count: {len(series)}")

# %%
# Remove exact values
# ^^^^^^^^^^^^^^^^^^^
#
# Identify which values are sentinel/error codes and remove them by setting to NaN.
# Here we remove values 0, 1, and 10 which are known to be invalid.

series_corr = dv.corrections.set_exact_values_to_missing(
    series=series,
    values=[0, 1, 10],
    showplot=False,
    verbose=1
)

print("\nCorrected data:")
print(f"  Values: {series_corr.values}")
print(f"  Valid count: {series_corr.notna().sum()}")
print(f"  NaN count: {series_corr.isna().sum()}")
