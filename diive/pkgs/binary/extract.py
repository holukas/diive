import numpy as np
from pandas import Series


def get_encoded_value_from_int(integer: int,
                               bit_start: int,
                               bit_end: int,
                               gain: float = 1,
                               base: int = 2,
                               n_bits: int = 8) -> float:
    """
     Extracts a subrange of bits from an integer, converts it to a float with an
     optional gain applied.

     Can be used to extract additional variables from a binary value that was
     stored as an integer.

     For example, using integer=18 first converts the integer to its binary
     representation ('10010', without the leading '0b'), then adds leading
     zeros to fulfill the number of bits (n_bits=8, yields '00010010'), then
     extracts the required bit subrange ('0010' for bit_start=4 and bit_end=8),
     then converts the subrange to integer representation (2) and then applies
     gain (in this example 6.25) to get the final value of 12.5.

     Some measurements are stored as binary values that were converted to integers
     for practical reasons. For example, in the binary value '10110001 00101110' (2 Bytes)
     each bit can represent a flag that is raised when certain conditions are met.
     Sometimes it is necessary to extract multiple specific bits from such binary
     values and convert them to a decimal representation for further processing.

     Args:
         integer: The integer value from which to extract bits.
         bit_start: The starting index of the bit subrange (inclusive).
         bit_end: The ending index of the bit subrange (exclusive).
         gain: A gain factor to apply to the converted value (default is 1).
         base: The base of the integer (default is 2, binary).
         n_bits: The total number of bits in the integer (default is 8).

     Returns:
         The converted and scaled value as a float.
     """
    bits = bin(integer).replace("0b", "")  # Convert to binary value
    bits = bits.zfill(n_bits)  # Add leading zeros, if needed
    bit_subrange = bits[bit_start:bit_end]  # Extract required subrange to calculate value
    value = int(str(bit_subrange), base)  # Convert to decimal value
    value *= gain
    return value


def get_encoded_value_series(
        int_series: Series,
        bit_start: int,
        bit_end: int,
        gain: float = 1,
        base: int = 2,
        n_bits: int = 8) -> Series:
    missing = int_series.isnull()
    available = ~missing
    int_series.loc[missing] = 0
    int_series = int_series.astype(int)

    if available.sum() > 0:
        int_series = int_series.map(lambda x: bin(x))
        int_series = int_series.str.replace("0b", "")
        int_series = int_series.map(lambda x: str(x).zfill(n_bits))
        int_series = int_series.map(lambda x: x[bit_start:bit_end])
        int_series = int_series.map(lambda x: int(str(x), base))
        int_series = int_series.multiply(gain)

    # Restore missing values
    int_series.loc[missing] = np.nan

    return int_series


def example_series():
    import pandas as pd
    from diive.core.io.filereader import ReadFileType, search_files
    # SOURCEFILE = r"F:\TMP\CH-FRU_ec_20230603-0830.dat"
    # SOURCEFILE = r"F:\TMP\CH-FRU_ec_20230702-1230.dat"
    # SOURCEFILE = r"F:\01-NEW\CH-FRU-FF202401\2023_rECord\raw_data_ascii_rECord\CH-FRU_ec_20230509-0730.dat"

    allseries = pd.Series()
    files = search_files(searchdirs=r"F:\TMP", pattern='*.dat')
    # files = search_files(searchdirs=r"F:\TMP\del", pattern='*.dat')
    # files = files[0:40]
    for ix, f in enumerate(files):
        rft = ReadFileType(filepath=f, filetype='RECORD_DAT_20HZ', output_middle_timestamp=True,
                           data_nrows=100)
        df, meta = rft.get_filedata()
        series = df['GA_DIAG_VALUE'].copy()
        new_series = get_encoded_value_series(int_series=series,
                                              bit_start=4,
                                              bit_end=8,
                                              gain=6.25,
                                              base=2,
                                              n_bits=8)

        if ix == 0:
            allseries = new_series.copy()
        else:
            allseries = pd.concat([allseries, new_series], ignore_index=True)

    import matplotlib.pyplot as plt
    allseries.plot()
    plt.show()


def example_int():
    value = get_encoded_value_from_int(integer=250,
                                       bit_start=4,
                                       bit_end=8,
                                       gain=6.25,
                                       base=2,
                                       n_bits=8)

    print(value)


if __name__ == '__main__':
    example_series()
    # example_int()
