import os
from pathlib import Path

from diive.core.io.filereader import ReadFileType
from diive.core.io.files import load_pickle

DIR_PATH = os.path.dirname(os.path.realpath(__file__))  # Dir of this file


def load_exampledata_DIIVE_CSV_30MIN():
    filepath = Path(DIR_PATH) / 'exampledata_CH-DAV_FP2022.5_2022.07_ID20230206154316_30MIN.diive.csv'
    loaddatafile = ReadFileType(filetype='DIIVE_CSV_30MIN',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile._readfile()
    return data_df, metadata_df


def load_exampledata_pickle():
    filepath = Path(DIR_PATH) / 'exampledata_CH-DAV_FP2022.5_2022_ID20230206154316_30MIN.diive.csv.pickle'
    data_df = load_pickle(filepath=str(filepath))
    return data_df
