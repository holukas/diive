"""
Examples for binary value extraction using get_encoded_value_from_int and get_encoded_value_series.

Run this script to see binary extraction results:
    python examples/binary/extract.py
"""
import pandas as pd
import diive as dv


def example_extract_bits_from_int():
    """Extract a subrange of bits from an integer value.

    Demonstrates extracting specific bits from an integer that encodes
    multiple values. For example, extracting automatic gain control (AGC)
    information from the diagnostic value of an eddy covariance instrument.
    Shows conversion from binary representation to decimal with gain scaling.
    """
    # Extract bits 4-8 from integer 250 with gain 6.25
    # Binary: 11111010
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


def example_extract_bits_from_series():
    """Extract a subrange of bits from a series of integer values.

    Demonstrates extracting bits from multiple integer values efficiently.
    Useful for decoding diagnostic flags or instrument parameters stored
    as binary values in integer columns.
    """
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

    # Create result dataframe
    result_df = pd.DataFrame({
        'original': int_series,
        'agc_extracted': agc_series
    })

    print(f"\nResults dataframe:\n{result_df}")


if __name__ == '__main__':
    example_extract_bits_from_int()
    example_extract_bits_from_series()
