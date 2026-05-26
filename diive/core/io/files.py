import os
import pickle
import time
import zipfile as zf
from pathlib import Path

from pandas import Series, DataFrame, read_parquet

from diive.core.times.times import TimestampSanitizer
from diive.core.utils.console import info, detail


def set_outpath(outpath: str or None, filename: str, fileextension: str):
    if outpath:
        outpath = Path(outpath)
        filepath = Path(outpath) / f"{filename}.{fileextension}"
    else:
        filepath = f"{filename}.{fileextension}"
    return filepath


def save_parquet(filename: str, data: DataFrame or Series, outpath: str or None = None) -> str:
    """
    Save pandas Series or DataFrame as Parquet file.

    Parquet is a columnar format designed for efficient storage and retrieval.
    Index and column names are preserved.

    Args:
        filename : str
            Name of output file (without extension; .parquet is added automatically).
        data : pandas.Series or pandas.DataFrame
            Data to save. Series is converted to single-column DataFrame.
            Index name is preserved (e.g., 'TIMESTAMP_END', 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE').
        outpath : str or Path, optional
            Directory path for output file. Default None: saves to current working directory.

    Returns:
        str
            Full filepath to saved Parquet file.

    Notes:
        Index name and frequency information are preserved in the Parquet file.
        When reloading with load_parquet(), use output_middle_timestamp to convert
        timestamp representation if needed.

    Example:
        See `examples/io/io_load_save_parquet.py`
    """
    filepath = set_outpath(outpath=outpath, filename=filename, fileextension='parquet')
    tic = time.time()
    if isinstance(data, Series):
        data = data.to_frame()
    data.to_parquet(filepath)
    toc = time.time() - tic
    info(f"Saved file {filepath} ({toc:.3f} seconds).")
    return str(filepath)


def load_parquet(filepath: str or Path, output_middle_timestamp: bool = True,
                 sanitize_timestamp: bool = True) -> DataFrame:
    """
    Load data from Parquet file to pandas DataFrame.

    Automatically detects time resolution and optionally converts timestamp representation.
    Index name and data types are preserved from the Parquet file.

    Args:
        filepath : str or Path
            Path to Parquet file.
        output_middle_timestamp : bool, optional (default True)
            Convert timestamp index to middle-of-period representation.
            Use when converting from 'TIMESTAMP_END' or 'TIMESTAMP_START' formats.
            **Requires: sanitize_timestamp=True AND index name must be one of**
            **'TIMESTAMP_END', 'TIMESTAMP_START', or 'TIMESTAMP_MIDDLE'.**
        sanitize_timestamp : bool, optional (default True)
            Validate and regularize timestamp index: check naming convention, auto-detect
            frequency, fill small gaps, remove duplicate timestamps. Calls TimestampSanitizer.
            Required for output_middle_timestamp to work.

    Returns:
        pandas.DataFrame
            Data with validated DatetimeIndex. Frequency is detected and stored.

    Raises:
        ValueError
            If output_middle_timestamp=True but sanitize_timestamp=False.
        ValueError
            If index name is not recognized (see sanitize_timestamp parameter).

    Notes:
        Parquet preserves index names, so ensure data was saved with proper naming
        (e.g., 'TIMESTAMP_END' for end-of-period data) before using output_middle_timestamp=True.

    Example:
        See `examples/io/io_load_save_parquet.py`
    """
    if output_middle_timestamp and not sanitize_timestamp:
        raise ValueError(
            "output_middle_timestamp=True requires sanitize_timestamp=True. "
            "Timestamp conversion (to middle-of-period) requires timestamp validation and "
            "frequency detection, which are performed during sanitization."
        )

    tic = time.time()
    df = read_parquet(filepath)
    toc = time.time() - tic
    info(f"Loaded .parquet file {filepath} ({toc:.3f} seconds).")

    if sanitize_timestamp:
        # Check timestamp, also detects frequency of time series, this info was lost when saving to the parquet file
        df = TimestampSanitizer(data=df, output_middle_timestamp=output_middle_timestamp).get()
        detail(f"Detected time resolution of {df.index.freq} / {df.index.freqstr}")
    return df


def save_as_pickle(outpath: str or None, filename: str, data) -> str:
    """Save data as pickle"""
    filepath = set_outpath(outpath=outpath, filename=filename, fileextension='pickle')
    tic = time.time()
    pickle_out = open(filepath, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    toc = time.time() - tic
    info(f"Saved pickle {filepath} ({toc:.3f} seconds).")
    return str(filepath)


def load_pickle(filepath: str):
    """Import data from pickle"""
    tic = time.time()
    pickle_in = open(filepath, "rb")
    data = pickle.load(pickle_in)
    toc = time.time() - tic
    info(f"Loaded pickle {filepath} ({toc:.3f} seconds).")
    return data


def unzip_file(filepath):
    """ Unzips zipped file in filepath

        Unzips the zipped file to a temporary directory, which is later, after
        data have been read in, deleted.

        Returns the filepath to the unzipped file and the directory to which
        the zipped file has been extracted to.
    """
    with zf.ZipFile(filepath, 'r') as zip_ref:
        # dir as string, w/ .amp.temp at the end of dir name
        dir_temp_unzipped = '{}{}'.format(filepath, ".temp")
        zip_ref.extractall(dir_temp_unzipped)

    ext = '.csv'
    dir_temp_unzipped = Path(dir_temp_unzipped)  # give dir as Path
    filename_unzipped = Path(dir_temp_unzipped).stem  # remove .temp for filename, only .amp, stem makes str
    filename_unzipped = Path(filename_unzipped).with_suffix(ext)  # replace .amp with .csv
    filepath = dir_temp_unzipped / filename_unzipped
    return filepath, dir_temp_unzipped


def loadfiles(sourcedir: str, fileext: str, filetype: str,
              idstr: str, limit_n_files: int = None) -> DataFrame:
    """Search and load data files of type *filetype*, merge data and store to one dataframe"""
    from diive.core.io.filereader import MultiDataFileReader

    info(f"Searching for {filetype} files with extension {fileext} and ID {idstr} in folder {sourcedir} ...")
    filepaths = [f for f in os.listdir(sourcedir) if f.endswith(fileext)]
    filepaths = [f for f in filepaths if idstr in f]
    filepaths = [sourcedir + "/" + f for f in filepaths]
    filepaths = [Path(f) for f in filepaths]
    info(f"Found {len(filepaths)} files:")
    for f in filepaths:
        detail(f"  {f}")
    if limit_n_files:
        filepaths = filepaths[0:limit_n_files]
    mergedfiledata = MultiDataFileReader(filetype=filetype, filepaths=filepaths)
    return mergedfiledata.data_df


def verify_dir(path: str) -> None:
    """ Create dir if it does not exist. """
    Path(path).mkdir(parents=True, exist_ok=True)
    return None


def _example_read_parquet():
    df = load_parquet(
        filepath=r"F:\Sync\luhk_work\40 - DATA\DATASETS\2025_FORESTS\2-parquet_merged\CH-Dav_ENF_ICOS+FXN_1997-2024.parquet",
        output_middle_timestamp=True,
        sanitize_timestamp=False
    )
    print(df)
    df.to_csv(r"F:\TMP\example.csv")


if __name__ == "__main__":
    _example_read_parquet()
