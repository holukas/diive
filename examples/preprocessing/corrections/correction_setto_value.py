"""
=================================
Set Time Range Values to Constant
=================================

Replace measurements in specific time periods with a constant value.
Useful for removing instrumental errors or correcting known malfunction periods.
"""

# %%
# Create hourly time series
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate sample data spanning 48 hours with hourly frequency.
# Values range from 1-48, representing measurements over two days.

import pandas as pd
import diive as dv

dates = pd.date_range('2022-06-01', periods=48, freq='h')
values = range(1, 49)
series = pd.Series(values, index=dates, name='flux')

print("Original data:")
print(f"  Shape: {series.shape}")
print(f"  Range: {series.min()} to {series.max()}")
print(f"  First 12 hours:\n{series.head(12)}")

# %%
# Replace values in time ranges
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Set a single timestamp and a date range to a constant value (0).
# This is useful for removing periods of sensor malfunction or known errors.

series_corr = dv.setto_value(
    series=series,
    dates=[
        '2022-06-01 01:00:00',  # Single timestamp
        ['2022-06-01 03:00:00', '2022-06-01 15:00:00']  # Date range [start, end]
    ],
    value=0,
    verbose=1
)

print("\nCorrected data:")
print(f"  Changed values: {series.shape[0] - series_corr.notna().sum() + series_corr.isna().sum()}")
print(f"  New range: {series_corr[series_corr > 0].min()} to {series_corr.max()}")
print(f"  First 20 hours (showing replacements):\n{series_corr.head(20)}")
