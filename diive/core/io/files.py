import os
import pickle
import time
import zipfile as zf
from pathlib import Path

from pandas import Series, DataFrame, read_parquet

from diive.core.io.filereader import MultiDataFileReader
from diive.core.times.times import TimestampSanitizer


def set_outpath(outpath: str or None, filename: str, fileextension: str):
    if outpath:
        outpath = Path(outpath)
        filepath = Path(outpath) / f"{filename}.{fileextension}"
    else:
        filepath = f"{filename}.{fileextension}"
    return filepath


def save_parquet(filename: str, data: DataFrame or Series, outpath: str or None = None) -> str:
    """
    Save pandas Series or DataFrame as parquet file

    Args:
        filename: str
            Name of the generated parquet file.
        data: pandas Series or DataFrame
        outpath: str or None
            If *None*, file is saved to system default folder. When used within
            a notebook, the file is saved in the same location as the notebook.

    Returns:
        str, filepath to parquet file
    """
    filepath = set_outpath(outpath=outpath, filename=filename, fileextension='parquet')
    tic = time.time()
    if isinstance(data, Series):
        data = data.to_frame()
    data.to_parquet(filepath)
    toc = time.time() - tic
    print(f"Saved file {filepath} ({toc:.3f} seconds).")
    return str(filepath)


def load_parquet(filepath: str or Path, output_middle_timestamp: bool = True) -> DataFrame:
    """
    Load data from Parquet file to pandas DataFrame

    Args:
        filepath: str
            filepath to parquet file
        output_middle_timestamp: Converts the timestamp to show the middle
            of the averaging interval.

    Returns:
        pandas DataFrame, data from Parquet file as pandas DataFrame
    """
    tic = time.time()
    df = read_parquet(filepath)
    toc = time.time() - tic
    # Check timestamp, also detects frequency of time series, this info was lost when saving to the parquet file
    df = TimestampSanitizer(data=df, output_middle_timestamp=output_middle_timestamp).get()
    print(f"Loaded .parquet file {filepath} ({toc:.3f} seconds). "
          f"Detected time resolution of {df.index.freq} / {df.index.freqstr} ")
    return df


def save_as_pickle(outpath: str or None, filename: str, data) -> str:
    """Save data as pickle"""
    filepath = set_outpath(outpath=outpath, filename=filename, fileextension='pickle')
    tic = time.time()
    pickle_out = open(filepath, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    toc = time.time() - tic
    print(f"Saved pickle {filepath} ({toc:.3f} seconds).")
    return str(filepath)


def load_pickle(filepath: str):
    """Import data from pickle"""
    tic = time.time()
    pickle_in = open(filepath, "rb")
    data = pickle.load(pickle_in)
    toc = time.time() - tic
    print(f"Loaded pickle {filepath} ({toc:.3f} seconds).")
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
    print(f"\nSearching for {filetype} files with extension {fileext} and"
          f"ID {idstr} in folder {sourcedir} ...")
    filepaths = [f for f in os.listdir(sourcedir) if f.endswith(fileext)]
    filepaths = [f for f in filepaths if idstr in f]
    filepaths = [sourcedir + "/" + f for f in filepaths]
    filepaths = [Path(f) for f in filepaths]
    print(f"    Found {len(filepaths)} files:")
    [print(f"   --> {f}") for f in filepaths]
    if limit_n_files:
        filepaths = filepaths[0:limit_n_files]
    mergedfiledata = MultiDataFileReader(filetype=filetype, filepaths=filepaths)
    return mergedfiledata.data_df


def verify_dir(path: str) -> None:
    """ Create dir if it does not exist. """
    Path(path).mkdir(parents=True, exist_ok=True)
    return None
