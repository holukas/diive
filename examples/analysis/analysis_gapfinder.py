"""
========================
Gap Detection and Analysis
========================

Identify and characterize missing data periods in time series.

Scans a time series for consecutive NaN values and reports where gaps occur,
how long they are, and which gaps are longest. Useful for spotting data loss
events and assessing overall data quality.

Best for: Understanding missing data patterns and data availability
"""

# %%
# Load data
# ^^^^^^^^^

import diive as dv

# Load the full 10-year dataset
data_df = dv.load_exampledata_parquet()

# Get the original NEE series (before any gap-filling)
series = data_df['NEE_CUT_REF_orig'].copy()

print(f"Dataset: {len(series)} half-hourly records from {series.index[0]} to {series.index[-1]}")
print(f"Missing values: {series.isna().sum()} ({100*series.isna().sum()/len(series):.1f}%)")

# %%
# Find all gaps
# ^^^^^^^^^^^^^

# Detect all consecutive missing periods (limit=None finds gaps of any size)
gf = dv.GapFinder(series=series, limit=None, sort_results=True)
gaps = gf.get_results()

print(f"\nFound {len(gaps)} gap periods (including single-record gaps)")

# %%
# Examine longest gaps
# ^^^^^^^^^^^^^^^^^^^^

# Display the top 5 longest gaps
print("\nTop 5 longest gaps:")
print(gaps.head(5))

# Examine the longest gap in detail
longest = gaps.iloc[0]
gap_length_days = longest['GAP_LENGTH'] * 0.5 / 24  # Convert 30min records to days
print(f"\nLongest gap: {longest['GAP_LENGTH']} missing records ({gap_length_days:.1f} days)")
print(f"  Start: {longest['GAP_START']}")
print(f"  End:   {longest['GAP_END']}")

# %%
# Gap distribution
# ^^^^^^^^^^^^^^^^

# Show statistics on gap sizes
print(f"\nGap size statistics:")
print(f"  Total gaps: {len(gaps)}")
print(f"  Median gap length: {gaps['GAP_LENGTH'].median():.0f} records")
print(f"  Mean gap length: {gaps['GAP_LENGTH'].mean():.0f} records")
print(f"  Longest gap: {gaps['GAP_LENGTH'].max():.0f} records")
print(f"  Total missing records: {gaps['GAP_LENGTH'].sum()}")

# Show how many gaps exceed certain thresholds
thresholds = [10, 100, 500]
for thresh in thresholds:
    count = (gaps['GAP_LENGTH'] >= thresh).sum()
    print(f"  Gaps >= {thresh} records: {count}")
