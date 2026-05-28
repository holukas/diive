"""
========================================
Correct Measurement Offset from Replicate
========================================

Detect and correct constant offset between a measurement and a reference
replicate. Uses brute-force search to find the offset that minimizes
absolute difference between measurement and replicate.
"""

# %%
# Create synthetic replicate and measurement data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We create a reference signal (replicate) and a measurement with a known
# constant offset. This simulates a sensor that reads systematically high
# or low compared to a reference instrument.

import numpy as np
import pandas as pd

import diive as dv

dates = pd.date_range('2024-01-01', periods=500, freq='30min')

# Reference signal with diurnal variation
replicate = pd.Series(
    50 + 20 * np.sin(np.arange(500) * 2 * np.pi / 48),
    index=dates,
    name='reference'
)

# Measurement with constant +4.2 offset
measurement = replicate.copy() + 4.2
measurement.name = 'measurement'

print("Initial data:")
print(f"  Replicate range: {replicate.min():.2f} to {replicate.max():.2f}")
print(f"  Measurement range: {measurement.min():.2f} to {measurement.max():.2f}")
print(f"  Expected offset: 4.2")

# %%
# Detect offset using brute-force search
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Search over a range of potential offsets to find the one that
# minimizes the mean absolute difference between measurement and reference.

offset_corrector = dv.corrections.MeasurementOffsetFromReplicate(
    measurement=measurement,
    replicate=replicate,
    offset_start=-10,
    offset_end=10,
    offset_stepsize=0.1
)

print("\nOffset detection:")
print(f"  Search range: -10 to +10")
print(f"  Step size: 0.1")

# %%
# Extract and evaluate correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Get the detected offset and corrected measurement values.

measurement_corrected = offset_corrector.get_corrected_measurement()
detected_offset = offset_corrector.get_offset()

print(f"\nResults:")
print(f"  Detected offset: {detected_offset:.2f}")
print(f"  Corrected measurement range: {measurement_corrected.min():.2f} to {measurement_corrected.max():.2f}")

# %%
# Evaluate correction quality
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Compare absolute differences before and after correction.

mae_before = (measurement - replicate).abs().mean()
mae_after = (measurement_corrected - replicate).abs().mean()

print(f"\nCorrection quality:")
print(f"  Mean absolute difference (before): {mae_before:.4f}")
print(f"  Mean absolute difference (after): {mae_after:.4f}")
print(f"  Improvement: {((mae_before - mae_after) / mae_before * 100):.1f}%")

# %%
# Apply known constant offset directly
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When the offset is already known (from calibration data, documentation,
# or previous analysis), apply it directly without detection.
# This is simpler and faster than the brute-force search approach.

known_offset = 4.2

# Simple offset correction: subtract the offset from measurement
measurement_simple_corrected = measurement - known_offset

mae_simple = (measurement_simple_corrected - replicate).abs().mean()

print(f"\nSimple constant offset correction:")
print(f"  Known offset: {known_offset}")
print(f"  Corrected range: {measurement_simple_corrected.min():.2f} to {measurement_simple_corrected.max():.2f}")
print(f"  Mean absolute difference: {mae_simple:.4f}")

# %%
# When to use each approach
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **Replicate-based detection (MeasurementOffsetFromReplicate):**
# - Offset value is unknown
# - Reference/replicate data is available
# - Want to find optimal offset automatically
#
# **Direct offset application:**
# - Offset is known (from calibration, documentation)
# - No reference data available
# - Need quick correction without computation
#
# Example comparison:
print(f"\nComparison of methods:")
print(f"  Detected offset: {detected_offset:.2f}")
print(f"  Known offset: {known_offset:.2f}")
print(f"  Detection error: {abs(detected_offset - known_offset):.4f}")
