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

     Background:
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
    """
    Convert a subrange of bits from an integer series to floats with an
    optional gain applied.

    Can be used to extract additional variables from binary values that were
    stored as integers.

    For example, using series [250, 250, 250] first converts the integers to their
    binary representation ('11111010', without the leading '0b'), then adds leading
    zeros to fulfill the number of bits (n_bits=8, yields '11111010', i.e., still
    the same because already 8 digits), then extracts the required bit subrange
    ('1010' for bit_start=4 and bit_end=8), then converts the subrange to integer
    representation (10) and then applies gain (in this example 6.25) to get the
    final values of 62.5. The final series is [62.5, 62.5, 62.5].

    Background:
    Some measurements are stored as binary values that were converted to integers
    for practical reasons. For example, in the binary value '10110001 00101110' (2 Bytes)
    each bit can represent a flag that is raised when certain conditions are met.
    Sometimes it is necessary to extract multiple specific bits from such binary
    values and convert them to a decimal representation for further processing.

    Args:
     int_series: Series of integer values from which to extract bits.
     bit_start: The starting index of the bit subrange (inclusive).
     bit_end: The ending index of the bit subrange (exclusive).
     gain: A gain factor to apply to the converted value (default is 1).
     base: The base of the integer (default is 2, binary).
     n_bits: The total number of bits in the integer (default is 8).

    Returns:
     Series of the converted and scaled values as a floats.

    """
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
    from pathlib import Path
    import pandas as pd
    from diive.core.io.filereader import ReadFileType, search_files

    OUTDIR = r"F:\01-NEW\CH-FRU-FF202401\2023_rECord\raw_data_ascii_rECord_filesWithAGC"

    batch = 4

    df_all = pd.Series()
    files = search_files(searchdirs=fr"F:\01-NEW\CH-FRU-FF202401\2023_rECord\raw_data_ascii_rECord_{batch}",
                         pattern='*.dat')
    # files = files[0:40]
    for ix, f in enumerate(files):
        rft = ReadFileType(filepath=f, filetype='RECORD_DAT_20HZ', output_middle_timestamp=True,
                           data_nrows=None)
        df, meta = rft.get_filedata()
        series = df['GA_DIAG_VALUE'].copy()
        df['AGC'] = get_encoded_value_series(int_series=series,
                                             bit_start=4,
                                             bit_end=8,
                                             gain=6.25,
                                             base=2,
                                             n_bits=8)

        filename = f.stem
        suffix = f.suffix
        outfilename = f"{filename}_withAGC{suffix}"
        outfilepath = Path(OUTDIR) / outfilename

        df.to_csv(outfilepath, index=False)
        print(f"Saved file {outfilepath}.")

        if ix == 0:
            df_all = df.copy()
        # elif ix == 10:
        #     break
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)

    # Plot all variables across all files
    for ix, v in enumerate(df_all.columns):
        plot = df_all[v].plot(title=f"{v} @20Hz across all files", figsize=(20, 9)).get_figure()
        outplotpath = Path(OUTDIR) / f"{ix}_{v}_timesseries_across_all_files_batch_{batch}.png"
        plot.savefig(outplotpath)
        plot.show()
        # plt.close(plot)
        print(f"Saved plot {outplotpath}.")


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
