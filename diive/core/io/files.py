import pickle
import time
import zipfile as zf
from pathlib import Path


def save_as_pickle(outpath: str or None, filename: str, data) -> str:
    """Save data as pickle"""
    if outpath:
        outpath = Path(outpath)
        filepath = Path(outpath) / f"{filename}.pickle"
    else:
        filepath = f"{filename}.pickle"
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
