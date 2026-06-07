# Timestamp & Time Series Handling Examples

Examples demonstrating timestamp sanitization, frequency detection, temporal aggregation, and statistical analysis.

**6 examples covering the complete time series handling pipeline.**

## Contents

**Core utilities:**
- **times_timestamp_sanitizer.py** — 10-step timestamp validation, cleaning, and regularization
- **times_keep_daterange.py** — non-destructive date-range subselection (keep a time window)

**Temporal analysis & aggregation:**
- **times_frequency_detection.py** — Auto-detect time resolution with confidence scoring
- **times_diel_cycles.py** — Calculate hourly aggregation patterns and diurnal cycles
- **times_temporal_matrices.py** — Convert time series to matrix format (years × months) for visualization
- **times_statistics.py** — Quick statistical profiling and data quality assessment

## Overview

The `TimestampSanitizer` is a comprehensive utility for:
- **Validating** datetime format and naming conventions
- **Cleaning** NaT rows, duplicates, and monotonicity violations
- **Regularizing** irregular time series by filling gaps
- **Detecting** frequency with confidence scoring
- **Tracking** all changes with detailed diagnostics

## Use Cases

**Clean raw flux data with timestamp issues:**
```python
from diive import TimestampSanitizer

# Handle common timestamp problems
sanitizer = TimestampSanitizer(
    data=df,
    output_middle_timestamp=True,          # Convert to mid-period timestamps
    validate_naming=True,                  # Check TIMESTAMP_END/START/MIDDLE convention
    convert_to_datetime=True,              # Ensure datetime format
    remove_index_nat=True,                 # Remove NaT rows
    sort_ascending=True,                   # Sort chronologically
    remove_duplicates=True,                # Remove duplicate timestamps
    regularize=True,                       # Fill gaps with NaN rows
    nominal_freq='30min',                  # Expected frequency
    verbose=True                           # Print diagnostics
)

# Get cleaned data
df_clean = sanitizer.get()

# Inspect what changed
status = sanitizer.get_status()
print(f"Removed {status['rows_removed']} problematic rows")
print(f"Added {status['rows_added_by_regularization']} rows to regularize gaps")
print(f"Detected frequency: {status['inferred_frequency']}")
print(f"Frequency confidence: {status['frequency_confidence']:.0%}")
```

**Validate timestamp consistency:**
```python
from diive import TimestampSanitizer

# Just validate without modifying
sanitizer = TimestampSanitizer(
    data=df,
    validate_naming=True,
    convert_to_datetime=True,
    remove_index_nat=False,           # Don't remove, just validate
    sort_ascending=False,             # Don't sort
    remove_duplicates=False,          # Don't remove
    regularize=False,                 # Don't fill gaps
    verbose=True
)

status = sanitizer.get_status()
if status['frequency_confidence'] < 0.9:
    print(f"WARNING: Low frequency confidence ({status['frequency_confidence']:.1%})")
    if status['frequency_alternatives']:
        print(f"  Alternatives: {status['frequency_alternatives']}")
```

**Detect gaps and frequency:**
```python
from diive import TimestampSanitizer

# Focus on frequency detection
sanitizer = TimestampSanitizer(
    data=df,
    validate_naming=False,
    convert_to_datetime=True,
    nominal_freq='30min'
)

status = sanitizer.get_status()
print(f"Detected frequency: {status['inferred_frequency']}")
print(f"  Method: {status['frequency_detection_method']}")
print(f"  Confidence: {status['frequency_confidence']:.0%}")
if status['frequency_percent_matching']:
    print(f"  Intervals matching: {status['frequency_percent_matching']:.1f}%")
```

## 10-Step Pipeline

The `TimestampSanitizer` applies 10 configurable steps in sequence:

1. **Validate naming** — Check TIMESTAMP_END/START/MIDDLE conventions
2. **Convert to datetime** — Ensure proper datetime format
3. **Remove NaT rows** — Delete rows with missing timestamps
4. **Sort ascending** — Sort by timestamp chronologically
5. **Remove duplicates** — Remove rows with identical timestamps
6. **Validate monotonicity** — Check strictly increasing timestamps
7. **Detect frequency** — Infer sampling frequency with confidence scoring
8. **Validate frequency** — Check against nominal frequency (if provided)
9. **Regularize gaps** — Fill missing periods with NaN rows
10. **Convert to mid-period** — Shift to middle-of-period timestamps

Each step is optional and can be disabled.

## Frequency Detection

The `TimestampSanitizer` detects frequency using 3 methods and reports confidence:

| Method | When Used | Strength |
|--------|-----------|----------|
| **all_methods_agree** | All 3 methods find same frequency | Highest confidence (100%) |
| **full_dataset** | Majority of intervals match most common frequency | High confidence (85-99%) |
| **start_end_chunks** | First/last chunk frequencies agree | Medium confidence (70-84%) |
| **timedelta** | Fallback if others fail | Low confidence (<70%) |

**Example output:**
```
Detected frequency: 30min (confidence: 95%)
  Detection method: full_dataset
  Intervals matching: 99.5%
  Alternatives: [15min (2%), 60min (<1%)]
```

## Status Tracking

`get_status()` returns comprehensive diagnostics:

```python
status = sanitizer.get_status()

# Row counts
status['rows_input']                          # Input rows
status['rows_removed']                        # Rows removed
status['rows_added_by_regularization']        # Rows added for gaps
status['rows_output']                         # Output rows

# Frequency detection
status['inferred_frequency']                  # Detected frequency (e.g., '30min')
status['frequency_confidence']                # Confidence score (0-1)
status['frequency_detection_method']          # Which method was used
status['frequency_percent_matching']          # % intervals matching frequency
status['frequency_alternatives']              # Other frequencies detected

# Data quality
status['n_nat']                               # NaT rows found
status['n_duplicates']                        # Duplicate timestamps
status['monotonicity_violations']             # Non-increasing timestamps
```

## Time Series Processing Pipeline

The examples follow a logical progression:

1. **Sanitize** (`times_timestamp_sanitizer.py`) — Clean and validate raw timestamps
2. **Detect** (`times_frequency_detection.py`) — Verify time resolution and regularity
3. **Analyze** (`times_statistics.py`) — Profile statistical characteristics
4. **Aggregate** (`times_diel_cycles.py`, `times_temporal_matrices.py`) — Group by time period (hour, month, year)

For ML feature engineering (temporal features with sin/cos encoding), see `examples/features/feature_engineer.py`.

## Example Descriptions

### 1. Timestamp Sanitization
Comprehensive 10-step validation and cleaning pipeline. Removes NaT values, duplicates, sorts chronologically, regularizes gaps, and detects frequency with confidence scoring.

**Use when:** Loading raw data with timestamp issues, needing to validate data quality.

### 2. Frequency Detection
Automatically detects time resolution (e.g., 30-minute, hourly) using 3 independent methods with confidence scoring. Shows alternatives when detection is ambiguous.

**Use when:** Need to verify data frequency, identify irregular sampling patterns, validate data consistency.

### 3. Diel Cycles
Extracts time-of-day patterns by grouping data by hour and calculating statistics for each. Shows seasonal variation (monthly diel cycles) and anomaly detection.

**Use when:** Understanding ecosystem processes with daily cycles (photosynthesis, evapotranspiration), identifying unusual daily patterns.

### 4. Temporal Matrices
Converts time series to year × month matrix format, ideal for heatmap visualization and long-term pattern analysis. Supports mean, sum, max, min aggregation and percentile ranking.

**Use when:** Visualizing multi-year patterns, detecting trends across years, identifying warmest/coldest periods.

### 5. Statistics
Quick statistical profiling: mean, median, SD, variance, percentiles (P01, P05, P25, P75, P95, P99). Useful for data quality assessment.

**Use when:** Need rapid summary statistics, comparing multiple variables, assessing missing data percentage.

## Running Examples

```bash
# Individual examples
uv run python examples/times/times_timestamp_sanitizer.py
uv run python examples/times/times_frequency_detection.py
uv run python examples/times/times_diel_cycles.py
uv run python examples/times/times_temporal_matrices.py
uv run python examples/times/times_statistics.py

# Run all examples
uv run python examples/run_all_examples.py
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Inconsistent TIMESTAMP columns** | Set `validate_naming=True`, fix column names |
| **NaT values in index** | Set `remove_index_nat=True` |
| **Unsorted timestamps** | Set `sort_ascending=True` |
| **Duplicate timestamps** | Set `remove_duplicates=True` |
| **Large gaps in data** | Set `regularize=True` to fill with NaN |
| **Timezone issues** | Ensure timezone-naive UTC timestamps before sanitizing |
| **Frequency detection failures** | Inspect `frequency_alternatives` in status report |

## Example Dataset Expectations

DIIVE examples use 30-minute timestamps (typical for flux tower data):
- Frequency: `30min`
- No gaps (fully regularized)
- No duplicates
- Strictly monotonic (increasing)
- UTC timezone (assumed)

Use `TimestampSanitizer` to prepare raw data to match these expectations.