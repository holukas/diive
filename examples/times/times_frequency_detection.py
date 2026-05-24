"""
TIME RESOLUTION DETECTION: AUTO-DETECT FREQUENCY
=================================================

Detect time series sampling frequency with confidence scores. Three approaches:
TimestampSanitizer for full validation, direct functions for fast checks, and
DetectFrequency class for details.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

import pandas as pd

import diive as dv

df = dv.load_exampledata_parquet()
series = df['Tair_f'].copy()

print("Example data loaded:")
print(f"  Records: {len(series)}")
print(f"  Period: {series.index.min()} to {series.index.max()}")

# %%
# Method 1: TimestampSanitizer (full validation)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Detects frequency as part of a 10-step validation pipeline. This gives
# a confidence score combining multiple detection methods.

sanitizer = dv.times.TimestampSanitizer(
    data=pd.DataFrame({'value': series}),
    regularize=False,  # Skip gap-filling
    verbose=False
)

status = sanitizer.get_status()

print("\n" + "=" * 60)
print("TimestampSanitizer Results")
print("=" * 60)
print(f"Detected frequency: {status['inferred_frequency']}")
print(f"Confidence: {status['frequency_confidence']:.1%}")
print(f"Detection method: {status['frequency_detection_method']}")
print(f"Intervals matching: {status['frequency_percent_matching']:.1f}%")

# Assess result quality
confidence = status['frequency_confidence']
if confidence >= 0.95:
    print(f"[OK] All methods agree")
elif confidence >= 0.90:
    print(f"[OK] Multiple methods agree")
elif confidence >= 0.70:
    print(f"[!] Some disagreement - check for gaps or irregular intervals")
else:
    print(f"[X] Methods disagree - data may be irregular")

# Show alternatives if any
if status['frequency_alternatives']:
    print(f"\nAlternatives detected: {status['frequency_alternatives']}")

# %%
# Method 2: Direct detection functions (lightweight)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Three independent functions for frequency analysis. Each method has different
# strengths - compare results to assess reliability.

freq_prog, info_prog = dv.times.timestamp_infer_freq_progressively(series.index)
freq_full, info_full = dv.times.timestamp_infer_freq_from_fullset(series.index)
freq_delta, info_delta = dv.times.timestamp_infer_freq_from_timedelta(series.index)

print("\n" + "=" * 60)
print("Direct Detection Functions")
print("=" * 60)

print(f"\n1. Progressive (fast):")
print(f"   Frequency: {freq_prog}")
print(f"   Info: {info_prog}")

print(f"\n2. Full dataset (requires regular data):")
print(f"   Frequency: {freq_full}")
print(f"   Info: {info_full}")

print(f"\n3. Timedelta (tolerates gaps):")
print(f"   Frequency: {freq_delta}")
print(f"   Info: {info_delta}")

# Check agreement
all_methods = [freq_prog, freq_full, freq_delta]
unique_results = set(f for f in all_methods if f is not None)

print("\nAgreement:")
if len(unique_results) == 1:
    print(f"[OK] All agree: {list(unique_results)[0]}")
elif len(unique_results) == 2:
    print(f"[!] Two methods agree on {list(unique_results)[0]}")
else:
    print(f"[X] All three disagree: {unique_results}")

# %%
# Method 3: DetectFrequency class (internal API)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Lower-level class used internally by TimestampSanitizer. Shows which
# detection method won and details about alternatives.

detector = dv.times.DetectFrequency(index=series.index, verbose=False)

print("\n" + "=" * 60)
print("DetectFrequency Class")
print("=" * 60)
print(f"Frequency: {detector.get()}")
print(f"Confidence: {detector.confidence:.1%}")
print(f"Method: {detector.detection_method}")
print(f"Percent matching: {detector.percent_matching}%")

if detector.alternatives:
    print(f"Alternatives: {detector.alternatives}")

# Detection method details
method_info = {
    'all_methods_agree': 'All three methods found same frequency (100%)',
    'full_dataset': 'Pandas inference from complete data (95%)',
    'timedelta': f'Most frequent interval ({detector.confidence:.0%})',
    'start_end_chunks': 'Start and end intervals match (70%)',
}

if detector.detection_method in method_info:
    print(f"\n{detector.detection_method}: {method_info[detector.detection_method]}")

# %%
# Validate detected frequency
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

expected = '30min'
detected = status['inferred_frequency']

print("\n" + "=" * 60)
print("Validation")
print("=" * 60)
print(f"Expected: {expected}")
print(f"Detected: {detected}")

if detected == expected:
    print(f"[OK] Match")
else:
    print(f"[X] Mismatch")
    print(f"Check for: gaps, duplicate timestamps, incorrect frequency specification")
