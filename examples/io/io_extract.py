"""
==================
Binary Value Extraction
==================

Extract specific bits and encoded values from integer data.

Demonstrates extracting subranges of bits from integers and series, useful for
decoding diagnostic flags or instrument parameters stored as binary values.
"""

# %%
# Extract bits from a single integer value
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This example shows how to extract a specific range of bits from an integer,
# optionally applying a gain factor for scaling. Useful for decoding automatic
# gain control (AGC) information from instrument diagnostic values.

import pandas as pd
import diive as dv

# Extract bits 4-8 from integer 250 with gain 6.25
# Binary representation: 11111010
# Bits 4-8: 1010 (decimal 10)
# With gain: 10 * 6.25 = 62.5

value = dv.get_encoded_value_from_int(
    integer=250,
    bit_start=4,
    bit_end=8,
    gain=6.25,
    base=2,
    n_bits=8
)

print("Bit extraction from integer:")
print(f"  Integer: 250")
print(f"  Binary: {bin(250)}")
print(f"  Extract bits 4-8: {bin(250)[2:].zfill(8)[4:8]}")
print(f"  Decimal value: {int(bin(250)[2:].zfill(8)[4:8], 2)}")
print(f"  With gain 6.25: {value}")

# %%
# Extract bits from a series of integer values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Apply the same bit extraction operation to multiple integer values efficiently.
# This is useful for batch-decoding diagnostic flags from a DataFrame column.

# Create a series of diagnostic values
int_series = pd.Series([250, 250, 250, 128, 64], name='GA_DIAG_VALUE')

print("\nBit extraction from series:")
print(f"Original series:\n{int_series.values}")

# Extract bits 4-8 with gain
agc_series = dv.get_encoded_value_series(
    int_series=int_series.copy(),
    bit_start=4,
    bit_end=8,
    gain=6.25,
    base=2,
    n_bits=8
)

print(f"\nExtracted AGC values (bits 4-8, gain=6.25):\n{agc_series.values}")

# %%
# Create result dataframe
# ^^^^^^^^^^^^^^^^^^^^^^^

result_df = pd.DataFrame({
    'original': int_series,
    'agc_extracted': agc_series
})

print(f"\nResults dataframe:\n{result_df}")
