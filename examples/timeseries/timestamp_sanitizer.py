"""
Comprehensive example: TimestampSanitizer with progressively severe issues.

This example demonstrates the 9-step timestamp sanitization pipeline by creating
progressively more severe timestamp issues, from clean to completely broken data.
Shows all parameters and how the sanitizer handles escalating complexity.
"""

import pandas as pd
import numpy as np
import diive as dv


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)


def print_data_info(data, label="Data"):
    """Print diagnostic information about the data."""
    print(f"\n{label}:")
    print(f"  Shape: {data.shape}")
    print(f"  Index name: {data.index.name}")
    print(f"  Index type: {type(data.index).__name__}")
    freq = getattr(data.index, 'freq', None)
    print(f"  Index freq: {freq}")
    if len(data) > 0:
        print(f"  First timestamp: {data.index[0]}")
        print(f"  Last timestamp: {data.index[-1]}")


# Load example data
print_section("LOAD EXAMPLE DATA")
df = dv.load_exampledata_parquet()
series = df['NEE_CUT_REF_f'].copy()
print(f"Loaded parquet file with {len(series)} records")
print_data_info(series, "Original series (clean baseline)")

# ==============================================================================
# LEVEL 1: CLEAN DATA (baseline)
# ==============================================================================
print_section("LEVEL 1: CLEAN DATA (BASELINE)")
print("\nIssues introduced: NONE")
print("Expected processing time: Fast (minimal work)")

clean_series = series.copy()
print_data_info(clean_series, "BEFORE sanitization")

sanitizer = dv.TimestampSanitizer(
    data=clean_series,
    output_middle_timestamp=True,
    validate_naming=True,
    convert_to_datetime=True,
    remove_index_nat=True,
    sort_ascending=True,
    remove_duplicates=True,
    regularize=True,
    nominal_freq='30min',
    verbose=True
)

result = sanitizer.get()
print_data_info(result, "AFTER sanitization")

# Show status report
status = sanitizer.get_status()
print(f"\nStatus report:")
print(f"  Original rows: {status['original_shape'][0]}")
print(f"  Final rows: {status['final_shape'][0]}")
print(f"  Rows removed: {status['rows_removed']} (NaT: {status['rows_removed_nat']}, "
      f"duplicates: {status['rows_removed_duplicates']})")
print(f"  Rows added by regularization: {status['rows_added_by_regularization']}")
print(f"  Net change: {status['net_rows']:+d} rows")
print(f"  Frequency: {status['inferred_frequency']} (confidence: {status['frequency_confidence']:.0%})")
if status['frequency_percent_matching']:
    print(f"    - Detection method: {status['frequency_detection_method']}")
    print(f"    - Intervals matching: {status['frequency_percent_matching']:.1f}%")
if status['frequency_alternatives']:
    print(f"    - Alternatives detected: {', '.join(status['frequency_alternatives'])}")

# ==============================================================================
# LEVEL 2: MINOR ISSUES (few NaTs + 1 duplicate)
# ==============================================================================
print_section("LEVEL 2: MINOR ISSUES")
print("\nIssues introduced:")
print("  - 3 random NaT values")
print("  - 1 duplicate timestamp")
print("Expected processing time: Short (few removals)")

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

print_data_info(minor_series, "BEFORE sanitization")

sanitizer = dv.TimestampSanitizer(
    data=minor_series,
    output_middle_timestamp=True,
    validate_naming=True,
    convert_to_datetime=True,
    remove_index_nat=True,
    sort_ascending=True,
    remove_duplicates=True,
    regularize=True,
    nominal_freq='30min',
    verbose=True
)

result = sanitizer.get()
print_data_info(result, "AFTER sanitization")

status = sanitizer.get_status()
print(f"\nStatus: removed {status['rows_removed']} rows (NaT: {status['rows_removed_nat']}, "
      f"duplicates: {status['rows_removed_duplicates']}), "
      f"net change: {status['net_rows']:+d} rows, "
      f"frequency confidence: {status['frequency_confidence']:.0%}")

# ==============================================================================
# LEVEL 3: MODERATE ISSUES (many NaTs + multiple duplicates + partial sorting)
# ==============================================================================
print_section("LEVEL 3: MODERATE ISSUES")
print("\nIssues introduced:")
print("  - 20 scattered NaT values")
print("  - 5 duplicate timestamps")
print("  - Data chunks shuffled (unsorted)")
print("  - Index name changed to TIMESTAMP_END")
print("Expected processing time: Moderate (multiple issues)")

moderate_series = series.copy()
moderate_series.index.name = 'TIMESTAMP_END'

# Add 20 NaT values scattered throughout
np.random.seed(42)
nat_positions = np.random.choice(len(moderate_series), 20, replace=False)
for pos in sorted(nat_positions, reverse=True):
    moderate_series = moderate_series.drop(moderate_series.index[pos])
moderate_series.index.name = 'TIMESTAMP_END'  # Preserve after drop

# Add 5 duplicates at different positions
dup_idx = moderate_series.index.tolist()
dup_idx[100] = dup_idx[99]
dup_idx[500] = dup_idx[499]
dup_idx[1000] = dup_idx[999]
dup_idx[2000] = dup_idx[1999]
dup_idx[3000] = dup_idx[2999]
moderate_series.index = dup_idx

# Shuffle data into 3 chunks
chunk_size = len(moderate_series) // 3
moderate_series = pd.concat([
    moderate_series.iloc[chunk_size*2:],
    moderate_series.iloc[:chunk_size],
    moderate_series.iloc[chunk_size:chunk_size*2]
])
moderate_series.index.name = 'TIMESTAMP_END'  # Restore after concat

print_data_info(moderate_series, "BEFORE sanitization")

sanitizer = dv.TimestampSanitizer(
    data=moderate_series,
    output_middle_timestamp=True,
    validate_naming=True,
    convert_to_datetime=True,
    remove_index_nat=True,
    sort_ascending=True,
    remove_duplicates=True,
    regularize=True,
    nominal_freq='30min',
    verbose=True
)

result = sanitizer.get()
print_data_info(result, "AFTER sanitization")

status = sanitizer.get_status()
print(f"\nStatus: removed {status['rows_removed']} rows (NaT: {status['rows_removed_nat']}, "
      f"duplicates: {status['rows_removed_duplicates']}), "
      f"net change: {status['net_rows']:+d} rows, "
      f"frequency confidence: {status['frequency_confidence']:.0%}")

# ==============================================================================
# LEVEL 4: SEVERE ISSUES (almost everything wrong)
# ==============================================================================
print_section("LEVEL 4: SEVERE ISSUES")
print("\nIssues introduced:")
print("  - 50 NaT values (scattered)")
print("  - 10 duplicate timestamps")
print("  - Completely shuffled/reversed sections")
print("  - Wrong index name ('WRONG_TIMESTAMP')")
print("  - String format instead of datetime")
print("  - Gaps in time series (will be regularized)")
print("Expected processing time: Long (major cleanup needed)")

severe_series = series.copy()

# Convert to string format (wrong!)
severe_series.index = severe_series.index.astype(str)
severe_series.index.name = 'WRONG_TIMESTAMP'

# Add 50 NaT values
np.random.seed(123)
nat_positions = np.random.choice(len(severe_series), 50, replace=False)
for pos in sorted(nat_positions, reverse=True):
    severe_series = severe_series.drop(severe_series.index[pos])
severe_series.index.name = 'WRONG_TIMESTAMP'  # Preserve after drop

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
severe_series.index.name = 'WRONG_TIMESTAMP'  # Restore after concat

print_data_info(severe_series, "BEFORE sanitization")

try:
    sanitizer = dv.TimestampSanitizer(
        data=severe_series,
        output_middle_timestamp=True,
        validate_naming=True,  # Will catch wrong name
        convert_to_datetime=True,  # Will convert from string
        remove_index_nat=True,
        sort_ascending=True,
        remove_duplicates=True,
        regularize=True,
        nominal_freq='30min',
        verbose=True
    )
    result = sanitizer.get()
    print_data_info(result, "AFTER sanitization")
except ValueError as e:
    print(f"\n  [Expected error caught] Validation failed: {str(e)[:80]}...")

# ==============================================================================
# LEVEL 5: EXTREMELY SEVERE (all parameters disabled to see raw issues)
# ==============================================================================
print_section("LEVEL 5: MINIMAL PROCESSING (all steps disabled except detect freq)")
print("\nParameters set to False:")
print("  - validate_naming: False (skip validation)")
print("  - convert_to_datetime: False (keep string format)")
print("  - remove_index_nat: False (keep NaT values)")
print("  - sort_ascending: False (keep unsorted)")
print("  - remove_duplicates: False (keep duplicates)")
print("  - regularize: False (keep gaps)")
print("  - output_middle_timestamp: False (keep original)")
print("Expected: Data mostly unchanged, only frequency detected")

# Use a moderately broken dataset
minimal_series = series.copy()
minimal_series.index.name = 'TIMESTAMP_START'

# Add just a few issues to demonstrate
np.random.seed(99)
nat_pos = np.random.choice(len(minimal_series), 5, replace=False)
for pos in sorted(nat_pos, reverse=True):
    minimal_series = minimal_series.drop(minimal_series.index[pos])
minimal_series.index.name = 'TIMESTAMP_START'  # Preserve after drop

# Add 3 duplicates
dup_idx = minimal_series.index.tolist()
dup_idx[200] = dup_idx[199]
dup_idx[400] = dup_idx[399]
dup_idx[600] = dup_idx[599]
minimal_series.index = dup_idx

# Reverse first chunk
chunk = minimal_series.iloc[:500].iloc[::-1]
minimal_series = pd.concat([chunk, minimal_series.iloc[500:]])
minimal_series.index.name = 'TIMESTAMP_START'  # Restore after concat

print_data_info(minimal_series, "BEFORE sanitization")

sanitizer = dv.TimestampSanitizer(
    data=minimal_series,
    output_middle_timestamp=False,
    validate_naming=False,
    convert_to_datetime=False,
    remove_index_nat=False,
    sort_ascending=False,
    remove_duplicates=False,
    regularize=False,
    nominal_freq=None,
    verbose=True
)

result = sanitizer.get()
print_data_info(result, "AFTER sanitization (minimal processing)")
print(f"Inferred frequency: {sanitizer.inferred_freq}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print_section("SUMMARY: PROCESSING COMPLEXITY BY ISSUE LEVEL")
print("""
Level | Issues                                      | Processing | Output Lines
------|---------------------------------------------|------------|-------------
  1   | None                                        | Minimal    | 6 lines
  2   | Few NaTs + 1 duplicate                      | Light      | 6 lines
  3   | Many NaTs + duplicates + unsorted           | Moderate   | 8 lines
  4   | Severe: wrong format, wrong name, lots gaps | Heavy      | 10+ lines
  5   | Issues present but most steps disabled      | Detection  | 4 lines

Key features:
  - Validates timestamp naming (3 allowed conventions)
  - Converts to datetime, removes NaTs and duplicates
  - Detects frequency using 3 methods
  - Regularizes gaps or keeps them
  - Can convert to middle-of-period timestamps
  - Each step can be enabled/disabled independently

This example shows how the sanitizer handles data ranging from clean to badly
broken, and how selective processing lets you control what gets fixed.
""")

print_section("ALL DEMONSTRATIONS COMPLETED")
