"""
===========================
Timestamp Sanitization
===========================

Comprehensive validation and cleaning of datetime indices through 10 configurable
steps. Demonstrates pipeline robustness with progressively severe data issues.

Best for: Understanding timestamp validation, fixing corrupted time series indices
"""

import pandas as pd
import numpy as np
import diive as dv

# %%
# Load example data (clean baseline)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
series = df['NEE_CUT_REF_f'].copy()

print(f"Loaded parquet file with {len(series)} records")
print(f"Date range: {series.index[0]} to {series.index[-1]}")

# %%
# Level 1: Clean data (minimal processing needed)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# No issues introduced - baseline sanitization.

clean_series = series.copy()

sanitizer = dv.times.TimestampSanitizer(
    data=clean_series,
    output_middle_timestamp=True,
    validate_naming=True,
    convert_to_datetime=True,
    remove_index_nat=True,
    sort_ascending=True,
    remove_duplicates=True,
    regularize=True,
    nominal_freq='30min',
    verbose=False
)

result = sanitizer.get()
status = sanitizer.get_status()

print("\nLevel 1 - Clean data:")
print(f"  Original: {status['original_shape'][0]} rows")
print(f"  Final: {status['final_shape'][0]} rows")
print(f"  Net change: {status['net_rows']:+d} rows")
print(f"  Frequency: {status['inferred_frequency']} (confidence: {status['frequency_confidence']:.0%})")

# %%
# Level 2: Minor issues (NaTs + duplicates)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Introduces 3 NaT values and 1 duplicate timestamp.

minor_series = series.copy()
idx_name = minor_series.index.name

# Add 3 NaT values
nat_positions = [50, 500, 1500]
for pos in sorted(nat_positions, reverse=True):
    minor_series = minor_series.drop(minor_series.index[pos])
minor_series.index.name = idx_name

# Add 1 duplicate
dup_idx = minor_series.index.tolist()
dup_idx[100] = dup_idx[99]
minor_series.index = dup_idx
minor_series.index.name = idx_name

sanitizer = dv.times.TimestampSanitizer(
    data=minor_series,
    output_middle_timestamp=True,
    validate_naming=True,
    convert_to_datetime=True,
    remove_index_nat=True,
    sort_ascending=True,
    remove_duplicates=True,
    regularize=True,
    nominal_freq='30min',
    verbose=False
)

result = sanitizer.get()
status = sanitizer.get_status()

print("\nLevel 2 - Minor issues (3 NaTs, 1 duplicate):")
print(f"  Removed {status['rows_removed']} rows (NaTs: {status['rows_removed_nat']}, "
      f"duplicates: {status['rows_removed_duplicates']})")
print(f"  Net change: {status['net_rows']:+d} rows")
print(f"  Frequency confidence: {status['frequency_confidence']:.0%}")

# %%
# Level 3: Moderate issues (scattered NaTs, multiple duplicates, unsorted)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Introduces 20 NaTs, 5 duplicates, shuffled chunks, and index name change.

moderate_series = series.copy()
moderate_series.index.name = 'TIMESTAMP_END'

# Add 20 NaT values scattered throughout
np.random.seed(42)
nat_positions = np.random.choice(len(moderate_series), 20, replace=False)
for pos in sorted(nat_positions, reverse=True):
    moderate_series = moderate_series.drop(moderate_series.index[pos])
moderate_series.index.name = 'TIMESTAMP_END'

# Add 5 duplicates
dup_idx = moderate_series.index.tolist()
dup_idx[100] = dup_idx[99]
dup_idx[500] = dup_idx[499]
dup_idx[1000] = dup_idx[999]
dup_idx[2000] = dup_idx[1999]
dup_idx[3000] = dup_idx[2999]
moderate_series.index = dup_idx

# Shuffle into 3 chunks
chunk_size = len(moderate_series) // 3
moderate_series = pd.concat([
    moderate_series.iloc[chunk_size*2:],
    moderate_series.iloc[:chunk_size],
    moderate_series.iloc[chunk_size:chunk_size*2]
])
moderate_series.index.name = 'TIMESTAMP_END'

sanitizer = dv.times.TimestampSanitizer(
    data=moderate_series,
    output_middle_timestamp=True,
    validate_naming=True,
    convert_to_datetime=True,
    remove_index_nat=True,
    sort_ascending=True,
    remove_duplicates=True,
    regularize=True,
    nominal_freq='30min',
    verbose=False
)

result = sanitizer.get()
status = sanitizer.get_status()

print("\nLevel 3 - Moderate issues (20 NaTs, 5 duplicates, unsorted):")
print(f"  Removed {status['rows_removed']} rows (NaTs: {status['rows_removed_nat']}, "
      f"duplicates: {status['rows_removed_duplicates']})")
print(f"  Net change: {status['net_rows']:+d} rows")
print(f"  Frequency confidence: {status['frequency_confidence']:.0%}")

# %%
# Level 4: Severe issues (string format, wrong name, many gaps)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Introduces 50 NaTs, 10 duplicates, string format, wrong index name,
# and completely shuffled sections.

severe_series = series.copy()

# Convert to string format (wrong!)
severe_series.index = severe_series.index.astype(str)
severe_series.index.name = 'WRONG_TIMESTAMP'

# Add 50 NaT values
np.random.seed(123)
nat_positions = np.random.choice(len(severe_series), 50, replace=False)
for pos in sorted(nat_positions, reverse=True):
    severe_series = severe_series.drop(severe_series.index[pos])
severe_series.index.name = 'WRONG_TIMESTAMP'

# Add 10 duplicates
dup_idx = severe_series.index.tolist()
for i in range(10):
    pos = (i + 1) * 300
    if pos < len(dup_idx) - 1:
        dup_idx[pos] = dup_idx[pos - 1]
severe_series.index = dup_idx

# Reverse chunks and shuffle
chunk_size = len(severe_series) // 4
severe_series = pd.concat([
    severe_series.iloc[chunk_size*3:],
    severe_series.iloc[chunk_size:chunk_size*2],
    severe_series.iloc[chunk_size*2:chunk_size*3],
    severe_series.iloc[:chunk_size]
])
severe_series.index.name = 'WRONG_TIMESTAMP'

try:
    sanitizer = dv.times.TimestampSanitizer(
        data=severe_series,
        output_middle_timestamp=True,
        validate_naming=True,
        convert_to_datetime=True,
        remove_index_nat=True,
        sort_ascending=True,
        remove_duplicates=True,
        regularize=True,
        nominal_freq='30min',
        verbose=False
    )
    result = sanitizer.get()
    status = sanitizer.get_status()

    print("\nLevel 4 - Severe issues (50 NaTs, 10 duplicates, wrong format/name, shuffled):")
    print(f"  Removed {status['rows_removed']} rows")
    print(f"  Net change: {status['net_rows']:+d} rows")
    print(f"  Frequency confidence: {status['frequency_confidence']:.0%}")
except ValueError as e:
    print(f"\nLevel 4 - Severe issues: Validation caught error (expected)")
    print(f"  Error: {str(e)[:60]}...")

# %%
# Level 5: Minimal processing (all steps disabled)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Shows frequency detection with all other cleaning steps disabled.

minimal_series = series.copy()
minimal_series.index.name = 'TIMESTAMP_START'

# Add a few issues
np.random.seed(99)
nat_pos = np.random.choice(len(minimal_series), 5, replace=False)
for pos in sorted(nat_pos, reverse=True):
    minimal_series = minimal_series.drop(minimal_series.index[pos])
minimal_series.index.name = 'TIMESTAMP_START'

# Add 3 duplicates
dup_idx = minimal_series.index.tolist()
dup_idx[200] = dup_idx[199]
dup_idx[400] = dup_idx[399]
dup_idx[600] = dup_idx[599]
minimal_series.index = dup_idx

# Reverse first chunk
chunk = minimal_series.iloc[:500].iloc[::-1]
minimal_series = pd.concat([chunk, minimal_series.iloc[500:]])
minimal_series.index.name = 'TIMESTAMP_START'

sanitizer = dv.times.TimestampSanitizer(
    data=minimal_series,
    output_middle_timestamp=False,
    validate_naming=False,
    convert_to_datetime=False,
    remove_index_nat=False,
    sort_ascending=False,
    remove_duplicates=False,
    regularize=False,
    nominal_freq=None,
    verbose=False
)

result = sanitizer.get()
status = sanitizer.get_status()

print("\nLevel 5 - Minimal processing (frequency detection only):")
print(f"  Data shape unchanged: {status['original_shape'][0]} -> {status['final_shape'][0]} rows")
print(f"  Inferred frequency: {status['inferred_frequency']}")
print(f"  Detection method: {status['frequency_detection_method']}")

# %%
# Processing summary
# ^^^^^^^^^^^^^^^^^^

print("\n" + "="*70)
print("SUMMARY: Timestamp Sanitization Pipeline Complexity")
print("="*70)
print("""
Level | Issues                              | Output
------|-------------------------------------|---------
  1   | None (baseline)                     | 4 lines
  2   | 3 NaTs + 1 duplicate                | 4 lines
  3   | 20 NaTs + 5 dups + unsorted         | 4 lines
  4   | 50 NaTs + string format + bad name  | 4 lines
  5   | Detection only (no cleaning)        | 4 lines

Key capabilities:
  - Validates timestamp naming conventions (TIMESTAMP_END, START, MIDDLE)
  - Converts to datetime from string format
  - Removes NaT values and duplicates
  - Detects frequency with confidence scoring
  - Regularizes gaps or preserves them
  - Converts to middle-of-period timestamps
  - Each step can be enabled/disabled independently
""")
