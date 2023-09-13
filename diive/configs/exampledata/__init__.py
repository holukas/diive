import os
from pathlib import Path

from pandas import DataFrame

from diive.core.io.filereader import ReadFileType
from diive.core.io.files import load_parquet
from diive.core.io.files import load_pickle

DIR_PATH = os.path.dirname(os.path.realpath(__file__))  # Dir of this file


def load_exampledata_parquet() -> DataFrame:
    filepath = Path(DIR_PATH) / 'exampledata_CH-DAV_FP2022.5_2013-2022_ID20230206154316_30MIN.parquet'
    data_df = load_parquet(filepath=filepath)
    return data_df


def load_exampledata_DIIVE_CSV_30MIN():
    filepath = Path(DIR_PATH) / 'exampledata_CH-DAV_FP2022.5_2022.07_ID20230206154316_30MIN.diive.csv'
    loaddatafile = ReadFileType(filetype='DIIVE_CSV_30MIN',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_eddypro_fluxnet_CSV_30MIN():
    filepath = Path(
        DIR_PATH) / 'exampledata_CH-AWS_2022.07_FR-20220127-164245_eddypro_fluxnet_2022-01-28T112538_adv.csv'
    loaddatafile = ReadFileType(filetype='EDDYPRO_FLUXNET_30MIN',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_pickle():
    """Load pickled dataframe"""
    filepath = Path(DIR_PATH) / 'exampledata_CH-DAV_FP2022.5_2022_ID20230206154316_30MIN.diive.csv.pickle'
    data_df = load_pickle(filepath=str(filepath))
    return data_df


def load_exampledata_winddir():
    """Load time series if wind direction in degrees"""
    filepath = Path(DIR_PATH) / 'exampledata_CH-FRU_2005-2022_winddirection_degrees.pickle'
    data_df = load_pickle(filepath=str(filepath))
    return data_df


def example():
    df = load_exampledata_parquet()
    print(df)


if __name__ == '__main__':
    example()
