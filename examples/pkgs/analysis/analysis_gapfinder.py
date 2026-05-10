"""
========================
Gap Detection and Analysis
========================

Find and analyze consecutive missing values in time series data.

Demonstrates gap detection: identifies consecutive missing values (NaN),
reports their locations, durations, and provides statistics on gap sizes
and distribution patterns.

Best for: Data quality assessment and gap characterization
"""

# %%
# Load data
# ^^^^^^^^^

import diive as dv

data_df = dv.load_exampledata_parquet()

# Get a single variable
series = data_df['NEE_CUT_REF_f'].copy()

# %%
# Intentionally create gaps for demonstration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Create artificial gaps of various sizes to demonstrate gap detection
series.iloc[100:105] = None      # 5-value gap
series.iloc[500:503] = None      # 3-value gap
series.iloc[1000:1001] = None    # 1-value gap

# %%
# Detect gaps
# ^^^^^^^^^^

# Find gaps with minimum length of 5 consecutive missing values
gf = dv.GapFinder(series=series, limit=5, sort_results=True)
results = gf.results

print("Gap detection results:")
print(f"Found {len(results)} gaps (limit=5 consecutive missing values)")
print(results)

# %%
# Gap statistics
# ^^^^^^^^^^^^^^

print(f"\nTotal records: {len(series)}")
print(f"Missing values: {series.isna().sum()}")
print(f"Valid values: {series.notna().sum()}")
print(f"Data coverage: {100 * series.notna().sum() / len(series):.1f}%")
