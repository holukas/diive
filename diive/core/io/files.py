"""
IO: FILES
=========

Save and load parquet and pickle files, unzip files, coerce data to the diive format.

Part of the diive library: https://github.com/holukas/diive
"""
import os
import pickle
import time
import zipfile as zf
from pathlib import Path
from typing import Callable, Optional

from pandas import Series, DataFrame, read_parquet

from diive.core.times.times import (TimestampSanitizer, validate_timestamp_naming, DetectFrequency,
                                    continuous_timestamp_freq)
from diive.core.utils.console import info, detail

#: Allowed names for a timestamp index in a diive parquet file.
ALLOWED_TIMESTAMP_NAMES = ['TIMESTAMP_END', 'TIMESTAMP_MIDDLE', 'TIMESTAMP_START']


def to_diive_format(data: DataFrame, timestamp_name: str = None) -> DataFrame:
    """Coerce a DataFrame to the diive parquet format.

    Ensures the data follows the diive convention:

    - **One header row**: MultiIndex columns are flattened to their top level so
      there is a single level of column names.
    - **Named timestamp index**: the index name must be one of
      ``'TIMESTAMP_END'``, ``'TIMESTAMP_MIDDLE'`` or ``'TIMESTAMP_START'``
      (indicating what the timestamp marks). Pass ``timestamp_name`` to set it;
      otherwise the existing index name is validated.

    Args:
        data: DataFrame to coerce.
        timestamp_name: Optional timestamp index name to set before validation.
            Must be one of :data:`ALLOWED_TIMESTAMP_NAMES`.

    Returns:
        A (possibly copied) DataFrame in diive format.

    Raises:
        ValueError: If no valid timestamp index name is available.
    """
    out = data
    if out.columns.nlevels > 1:
        out = out.copy()
        out.columns = out.columns.get_level_values(0)
    if timestamp_name is not None:
        if timestamp_name not in ALLOWED_TIMESTAMP_NAMES:
            raise ValueError(
                f"timestamp_name must be one of {ALLOWED_TIMESTAMP_NAMES}, "
                f"got '{timestamp_name}'.")
        if out.index.name != timestamp_name:
            out = out.copy()
            out.index.name = timestamp_name
    validate_timestamp_naming(out)  # raises if the index name is not allowed
    return out


def set_outpath(outpath: str or None, filename: str, fileextension: str):
    """Build the output file path from *outpath*, *filename* and *fileextension*."""
    if outpath:
        outpath = Path(outpath)
        filepath = Path(outpath) / f"{filename}.{fileextension}"
    else:
        filepath = f"{filename}.{fileextension}"
    return filepath


def save_parquet(filename: str, data: DataFrame or Series, outpath: str or None = None,
                 enforce_diive_format: bool = False, timestamp_name: str = None) -> str:
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
        enforce_diive_format : bool, optional (default False)
            If True, coerce the data to the diive parquet format before saving
            (see :func:`to_diive_format`): one header row (flattened columns) and
            a validly-named timestamp index. Raises ValueError if the index name
            is not valid and ``timestamp_name`` is not given.
        timestamp_name : str, optional
            Timestamp index name to set when ``enforce_diive_format`` is True.
            One of 'TIMESTAMP_END', 'TIMESTAMP_MIDDLE', 'TIMESTAMP_START'.

    Returns:
        str
            Full filepath to saved Parquet file.

    Notes:
        The index name is preserved in the Parquet file, the index frequency is not.
        load_parquet() detects the frequency again when it sanitizes the timestamp.
        When reloading with load_parquet(), use output_middle_timestamp to convert
        timestamp representation if needed.

    Example:
        See `examples/io/io_load_save_parquet.py`
    """
    filepath = set_outpath(outpath=outpath, filename=filename, fileextension='parquet')
    tic = time.time()
    if isinstance(data, Series):
        data = data.to_frame()
    if enforce_diive_format:
        data = to_diive_format(data, timestamp_name=timestamp_name)
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
            frequency, remove duplicate timestamps and insert missing timestamps as
            empty (NaN) rows. Calls TimestampSanitizer.
            Records that are not on the regular time grid (for example 1-min
            records in 10-min data, or a record 1 s late) are removed, with a
            warning that gives their number. Use sanitize_timestamp=False to
            load all records as saved.
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


def load_parquet_many(filepaths: list, output_middle_timestamp: bool = True,
                      sanitize_timestamp: bool = True,
                      progress_callback: Optional[Callable[[str, int, int, object], None]] = None) -> DataFrame:
    """Load several parquet files and merge them into one DataFrame.

    Each file is read with :func:`load_parquet` and merged with the running
    result via ``DataFrame.combine_first`` (existing values take precedence; the
    incoming file fills gaps). This is the parquet counterpart to
    :class:`diive.core.io.filereader.MultiDataFileReader` and shares its
    ``progress_callback`` contract.

    With ``sanitize_timestamp=True``, the merged data are put on a regular
    time grid when the files leave a time gap between them: missing
    timestamps are added as empty (NaN) records. Records that are not on the
    grid (for example from a file with a different time resolution) are
    removed, with a warning that gives their number.

    Args:
        filepaths: Parquet file paths to read and merge, in order.
        output_middle_timestamp: Forwarded to :func:`load_parquet`.
        sanitize_timestamp: Forwarded to :func:`load_parquet`.
        progress_callback: Optional callable invoked while merging so callers
            (e.g. a GUI) can report per-file progress. Called as
            ``callback(phase, done, total, filepath)`` with ``phase`` either
            ``'reading'`` (before a file is read) or ``'done'`` (after it has
            been merged); ``done`` is the count of finished files and ``total``
            the number of files.

    Returns:
        The merged DataFrame.
    """
    merged = None
    total = len(filepaths)
    for idx, filepath in enumerate(filepaths):
        if progress_callback:
            progress_callback('reading', idx, total, filepath)
        df = load_parquet(filepath=filepath, output_middle_timestamp=output_middle_timestamp,
                          sanitize_timestamp=sanitize_timestamp)
        merged = df if merged is None else merged.combine_first(df)
        if progress_callback:
            progress_callback('done', idx + 1, total, filepath)
    if sanitize_timestamp and merged is not None and merged.index.freq is None:
        # Files with a time gap between them leave missing timestamps in the
        # merged index
        freq = DetectFrequency(index=merged.index).get()
        merged = continuous_timestamp_freq(data=merged, freq=freq)
    return merged


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
