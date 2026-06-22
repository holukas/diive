"""Loaders for the bundled example datasets."""
import os
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from diive.core.io.filereader import ReadFileType, MultiDataFileReader, search_files, DataFileReader
from diive.core.io.files import load_parquet, load_pickle

DIR_PATH = os.path.dirname(os.path.realpath(__file__))  # Dir of this file


def load_exampledata_parquet() -> DataFrame:
    """Load the bundled CH-DAV 2013-2022 30-min example dataset (parquet)."""
    filepath = Path(DIR_PATH) / 'exampledata_PARQUET_CH-DAV_FP2022.5_2013-2022_ID20230206154316_30MIN.parquet'
    data_df = load_parquet(filepath=filepath)
    return data_df


def load_exampledata_parquet_tlag_vars_level0() -> DataFrame:
    """Load the CH-HON 2025 Level-0 time-lag example variables (parquet)."""
    filepath = Path(DIR_PATH) / 'exampledata_PARQUET_CH-HON_2025_LEVEL-0_TLAG-VARS_IRGA75_SUBSET.parquet'
    data_df = load_parquet(filepath=filepath)
    return data_df


def load_exampledata_parquet_long() -> DataFrame:
    """Load the long CH-DAV 2013-2022 30-min example dataset (parquet)."""
    filepath = Path(DIR_PATH) / 'exampledata_PARQUET_CH-DAV_FP2022.5_2013-2022_ID20230206154316_30MIN.parquet'
    data_df = load_parquet(filepath=filepath)
    return data_df


def load_exampledata_parquet_cha() -> DataFrame:
    """Load the CH-CHA 30-min example subset (parquet)."""
    filepath = Path(DIR_PATH) / 'exampledata_PARQUET_CH-CHA_FP2025.3_subset_30MIN.parquet'
    data_df = load_parquet(filepath=filepath)
    return data_df


def load_exampledata_parquet_lae() -> DataFrame:
    """Load the CH-LAE 2016-2017 flux example dataset (parquet)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_PARQUET_CH-LAE_FP2025.X_22_IRGA75+IRGA72_FPC_NEE-QCF11-L3.2_2016-2017.parquet'
    data_df = load_parquet(filepath=filepath)
    return data_df


def load_exampledata_parquet_meteo_swin_1MIN() -> DataFrame:
    """Load the CH-LAE 1-min SW_IN meteo example series (parquet)."""
    filepath = Path(DIR_PATH) / 'exampledata_PARQUET_CH-LAE_METEO_SW_IN_1MIN.parquet'
    data_df = load_parquet(filepath=filepath)
    return data_df


def load_exampledata_parquet_meteo_rh_1MIN() -> DataFrame:
    """Load the CH-LAE 1-min relative-humidity meteo example series (parquet)."""
    filepath = Path(DIR_PATH) / 'exampledata_PARQUET_CH-LAE_METEO_RH_1MIN.parquet'
    data_df = load_parquet(filepath=filepath)
    return data_df


def load_exampledata_DIIVE_CSV_30MIN():
    """Load the CH-DAV 30-min DIIVE-CSV example file (returns data + metadata)."""
    filepath = Path(DIR_PATH) / 'exampledata_DIIVE-CSV-30MIN_CH-DAV_FP2022.5_2022.07_ID20230206154316_30MIN.diive.csv'
    loaddatafile = ReadFileType(filetype='DIIVE-CSV-30MIN',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_ETH_RECORD_TOA5_CSVGZ_20HZ():
    """Load the CH-FRU 20 Hz ETH-RECORD-TOA5 example file (returns data + metadata)."""
    filepath = Path(DIR_PATH) / 'exampledata_ETH-RECORD-TOA5-CSVGZ-20HZ_CH-FRU_ec_20240404-1300.csv.gz'
    loaddatafile = ReadFileType(filetype='ETH-RECORD-TOA5-CSVGZ-20HZ',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_ETH_SONICREAD_BICO_CSVGZ_20HZ():
    """Load the CH-FRU 20 Hz ETH-SONICREAD-BICO example file (returns data + metadata)."""
    filepath = Path(DIR_PATH) / 'exampledata_ETH-SONICREAD-BICO-CSVGZ-20HZ_CH-FRU_202307071300.csv.gz'
    loaddatafile = ReadFileType(filetype='ETH-SONICREAD-BICO-CSVGZ-20HZ',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN():
    """Load the CH-AWS 30-min EddyPro FLUXNET example file (returns data + metadata)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_EDDYPRO-FLUXNET-CSV-30MIN_CH-AWS_2022.07_FR-20220127-164245_eddypro_fluxnet_2022-01-28T112538_adv.csv'
    loaddatafile = ReadFileType(filetype='EDDYPRO-FLUXNET-CSV-30MIN',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_parquet_lae_level1_30MIN():
    """Load the CH-LAE 30-min Level-1 flux example dataset (parquet)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_PARQUET_CH-LAE_22.4_FLUXES_LEVEL-1_IRGA72+METEO7_2024_SUBSET.parquet'
    data_df = load_parquet(filepath=filepath)
    return data_df


def load_exampledata_multiple_EDDYPRO_FLUXNET_CSV_30MIN():
    """Load and merge several CH-FRU EddyPro FLUXNET 30-min example files."""
    filepaths = search_files(
        searchdirs=DIR_PATH,
        pattern='exampledata_EDDYPRO-FLUXNET-CSV-30MIN_*_eddypro_CH-FRU_FR-*_fluxnet_*_adv.csv')
    loaddatafile = MultiDataFileReader(filetype='EDDYPRO-FLUXNET-CSV-30MIN',
                                       filepaths=filepaths,
                                       output_middle_timestamp=True)
    data_df = loaddatafile.data_df
    metadata_df = loaddatafile.metadata_df
    return data_df, metadata_df


def load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN():
    """Load the CH-FRU 30-min EddyPro full-output example file (returns data + metadata)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_EDDYPRO-FULL-OUTPUT-CSV-30MIN_eddypro_CH-FRU_FR-20240408-101506_full_output_2024-04-08T101558_adv.csv'
    loaddatafile = ReadFileType(filetype='EDDYPRO-FULL-OUTPUT-CSV-30MIN',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_FLUXNET_FULLSET_HH_CSV_30MIN():
    """Load the CH-Cha FLUXNET2015 FULLSET half-hourly example file (returns data + metadata)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_FLUXNET-FULLSET-HH-CSV-30MIN_FLX_CH-Cha_FLUXNET2015_FULLSET_HH_2005-2020_beta-3.csv'
    loaddatafile = ReadFileType(filetype='FLUXNET-FULLSET-HH-CSV-30MIN',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_ICOS_H2R_CSVZIP_10S():
    """Load the CH-Dav 10-s ICOS-H2R example file (returns data + metadata)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_ICOS-H2R-CSVZIP-10S_CH-Dav_BM_20230328_L02_F03.zip'
    loaddatafile = ReadFileType(filetype='ICOS-H2R-CSVZIP-10S',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_TOA5_DAT_1MIN():
    """Load the CH-OE2 1-min TOA5 example file (returns data + metadata)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_TOA5-DAT-1MIN_CH-OE2_iDL_BOX1_0_1_TBL1_20220629-1714.dat'
    loaddatafile = ReadFileType(filetype='TOA5-DAT-1MIN',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_GENERIC_CSV_HEADER_1ROW_TS_MIDDLE_FULL_1MIN_long():
    """Load the CH-FRU 1-min generic-CSV example file (returns data + metadata)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_GENERIC-CSV-HEADER-1ROW-TS-MIDDLE-FULL-1MIN_CH-FRU_iDL_BOX1_0_1_TBL1_20240401-0000.dat.csv'
    loaddatafile = ReadFileType(filetype='GENERIC-CSV-HEADER-1ROW-TS-MIDDLE-FULL-1MIN',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_GENERIC_CSV_HEADER_1ROW_TS_MIDDLE_FULL_NS_20HZ():
    """Load the CH-DAS 20 Hz generic-CSV example file (returns data + metadata)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_GENERIC-CSV-HEADER-1ROW-TS-MIDDLE-FULL-NS-20HZ_CH-DAS_202305130830_30MIN-SPLIT_TR.csv'
    loaddatafile = ReadFileType(filetype='GENERIC-CSV-HEADER-1ROW-TS-MIDDLE-FULL-NS-20HZ',
                                filepath=filepath,
                                data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    return data_df, metadata_df


def load_exampledata_GENERIC_TXT_EDDY_COVARIANCE_10Hz():
    """Load the 10 Hz generic-TXT eddy-covariance example file."""
    filepath = Path(
        DIR_PATH) / 'exampledata_GENERIC-TXT-EDDY-COVARIANCE-10Hz-2023-06-24-03-30_LGRData.txt'
    data_df = pd.read_csv(filepath, sep=',')
    return data_df


def load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN_with_datafilereader_parameters():
    """Load the CH-AWS EddyPro FLUXNET example via an explicit DataFileReader config (returns data + metadata)."""
    filepath = Path(
        DIR_PATH) / 'exampledata_EDDYPRO-FLUXNET-CSV-30MIN_CH-AWS_2022.07_FR-20220127-164245_eddypro_fluxnet_2022-01-28T112538_adv.csv'
    dfr = DataFileReader(filepath=filepath,
                         data_header_section_rows=[0],  # Header section (before data) comprises 1 row
                         data_skip_rows=[],  # Skip no rows
                         data_header_rows=[0],
                         # Header with variable names and units, in this case only variable names in first row of header
                         data_varnames_row=0,  # Variable names are in first row of header
                         data_varunits_row=None,  # Header does not contain any variable units
                         data_na_vals=[-9999],
                         # List of values interpreted as missing values, EddyPro uses -9999 for missing values in ouput file
                         data_freq="30min",  # Time resolution of the data is 30-minutes
                         data_delimiter=",",  # This csv file uses the comma as delimiter
                         data_nrows=None,
                         # How many data rows to read from files, mainly used for testing, in this case None to read all rows in file
                         timestamp_idx_col=["TIMESTAMP_END"],  # Name of the column that is used for the timestamp index
                         timestamp_datetime_format="%Y%m%d%H%M",  # Timestamp in the files looks like this: 202107010300
                         timestamp_start_middle_end="end",
                         # Timestamp in the file defined in *timestamp_idx_col* refers to the END of the averaging interval
                         output_middle_timestamp=True,
                         # Timestamp in output dataframe (after reading the file) refers to the MIDDLE of the averaging interval
                         compression=None)  # File is not compressed (not zipped)
    data_df, metadata_df = dfr.get_data()
    return data_df, metadata_df


def load_exampledata_pickle():
    """Load pickled dataframe"""
    filepath = Path(DIR_PATH) / 'exampledata_PICKLE_CH-DAV_FP2022.5_2022_ID20230206154316_30MIN.diive.csv.pickle'
    data_df = load_pickle(filepath=str(filepath))
    return data_df


def load_exampledata_winddir():
    """Load time series if wind direction in degrees"""
    filepath = Path(DIR_PATH) / 'exampledata_PARQUET_winddirection_degrees_CH-FRU_2005-2022.parquet'
    data_df = load_parquet(filepath=str(filepath))
    return data_df
