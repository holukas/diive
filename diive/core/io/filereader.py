"""
FILEREADER
==========
This package is part of the diive library.

"""
import datetime
import fnmatch
import os
from pathlib import Path
from typing import Literal

import pandas as pd
import pandas.errors
import yaml
from pandas import DataFrame

from diive.configs.filetypes import get_filetypes
from diive.core.dfun.frames import compare_len_header_vs_data, parse_header, get_len_data, \
    sort_multiindex_columns_names, rename_cols, convert_data_to_numeric
from diive.core.times.times import continuous_timestamp_freq, TimestampSanitizer


def search_folders(searchdirs: str or list) -> list:
    """ Search files and store their filename and the path to the file in dictionary. """
    foundfolders = []
    if isinstance(searchdirs, str):
        searchdirs = [searchdirs]  # Use str as list
    for searchdir in searchdirs:
        for root, dirs, files in os.walk(searchdir):
            # print(root)
            foundfolders.append(root)
            # for idx, settings_file_name in enumerate(files):
            #     if fnmatch.fnmatch(settings_file_name, pattern):
            #         filepath = Path(root) / settings_file_name
            #         found_files_dict[settings_file_name] = filepath
    # foundfolders.sort()
    return foundfolders


def search_files(searchdirs: str or list, pattern: str) -> list:
    """ Search files and store their filename and the path to the file in dictionary. """
    # found_files_dict = {}
    foundfiles = []
    if isinstance(searchdirs, str):
        searchdirs = [searchdirs]  # Use str as list
    for searchdir in searchdirs:
        for root, dirs, files in os.walk(searchdir):
            for idx, settings_file_name in enumerate(files):
                if fnmatch.fnmatch(settings_file_name, pattern):
                    filepath = Path(root) / settings_file_name
                    # found_files_dict[settings_file_name] = filepath
                    foundfiles.append(filepath)
    foundfiles.sort()
    return foundfiles


class ConfigFileReader:
    """
    Load and validate configuration from YAML file and store in dict

    - kudos: https://stackoverflow.com/questions/57687058/yaml-safe-load-special-character-from-file

    """

    def __init__(self,
                 configfilepath: Path or str,
                 validation: Literal['filetype', 'meteopipe']):
        """
        Args:
            configfilepath: Path to YAML file with configuration
            validation:
                - If *filetype*, the file is checked whether it follows the
                structure of required filetype settings. These files give
                info about the structure of the respective filetype.
                - If *meteopipe*, currently no validation is done. These files
                define the QA/QC steps that are performed for each variable.
        """
        self.configfilepath = Path(configfilepath)
        self.validation = validation

    def read(self) -> dict:
        with open(self.configfilepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if self.validation == 'filetype':
            validate_filetype_config(config=config)
        return config


def validate_filetype_config(config: dict):
    """Convert to required types"""

    # GENERAL
    config['GENERAL']['NAME'] = str(config['GENERAL']['NAME'])
    config['GENERAL']['DESCRIPTION'] = str(config['GENERAL']['DESCRIPTION'])
    config['GENERAL']['TAGS'] = list(config['GENERAL']['TAGS'])

    # FILE
    config['FILE']['EXTENSION'] = str(config['FILE']['EXTENSION'])
    config['FILE']['COMPRESSION'] = str(config['FILE']['COMPRESSION'])
    config['FILE']['COMPRESSION'] = None if config['FILE']['COMPRESSION'] == 'None' else config['FILE']['COMPRESSION']

    # TIMESTAMP
    config['TIMESTAMP']['DESCRIPTION'] = str(config['TIMESTAMP']['DESCRIPTION'])

    if config['TIMESTAMP']['INDEX_COLUMN'] == '-not-available-':
        config['TIMESTAMP']['INDEX_COLUMN'] = None
    else:
        config['TIMESTAMP']['INDEX_COLUMN'] = list(config['TIMESTAMP']['INDEX_COLUMN'])
        config['TIMESTAMP']['INDEX_COLUMN'] = \
            _convert_timestamp_idx_col(var=config['TIMESTAMP']['INDEX_COLUMN'])

    if config['TIMESTAMP']['DATETIME_FORMAT'] == '-not-available-':
        config['TIMESTAMP']['DATETIME_FORMAT'] = None
    else:
        config['TIMESTAMP']['DATETIME_FORMAT'] = str(config['TIMESTAMP']['DATETIME_FORMAT'])
        config['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'] = str(
            config['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'])

    # DATA
    config['DATA']['HEADER_SECTION_ROWS'] = list(config['DATA']['HEADER_SECTION_ROWS'])
    config['DATA']['SKIP_ROWS'] = list(config['DATA']['SKIP_ROWS'])
    config['DATA']['HEADER_ROWS'] = list(config['DATA']['HEADER_ROWS'])
    config['DATA']['VARNAMES_ROW'] = int(config['DATA']['VARNAMES_ROW'])
    if config['DATA']['VARUNITS_ROW'] == '-not-available-':
        config['DATA']['VARUNITS_ROW'] = None
    else:
        config['DATA']['VARUNITS_ROW'] = int(config['DATA']['VARUNITS_ROW'])
    config['DATA']['NA_VALUES'] = list(config['DATA']['NA_VALUES'])
    config['DATA']['FREQUENCY'] = str(config['DATA']['FREQUENCY'])
    config['DATA']['DELIMITER'] = str(config['DATA']['DELIMITER'])

    return config


def _convert_timestamp_idx_col(var: int or list):
    """Convert to list of tuples if needed

    Since YAML is not good at processing list of tuples,
    they are given as list of lists,
        e.g. [ [ "date", "[yyyy-mm-dd]" ], [ "time", "[HH:MM]" ] ].
    In this case, convert to list of tuples,
        e.g.  [ ( "date", "[yyyy-mm-dd]" ), ( "time", "[HH:MM]" ) ].
    """
    new = var
    if isinstance(var[0], int):
        pass
    elif isinstance(var[0], list):
        for idx, c in enumerate(var):
            new[idx] = (c[0], c[1])
    return new


# def rename_duplicate_cols(df: DataFrame) -> DataFrame:
#     """
#     Rename columns with the same name to unique names
#
#     For example:
#         - two columns with the same name in *df*:
#             'co2_mean' and 'co2_mean'
#             will be renamed to
#             'co2_mean.1' and 'co2_mean.2'
#
#     Args:
#         df: DataFrame with multiple columns
#
#     Returns:
#         Complete df with renamed columns. If no duplicates
#         were found, the returned df is the same as the input df.
#
#     Kudos:
#     - https://stackoverflow.com/questions/24685012/pandas-dataframe-renaming-multiple-identically-named-columns
#
#     """
#     df.columns = ParserBase({'usecols': None})._maybe_dedup_names(df.columns)
#     return df


class ColumnNamesSanitizer:

    def __init__(self,
                 df: DataFrame):
        self.df = df.copy()

        # # For testing: create a duplicate column name
        # self.df['DOY_START2'] = self.df['DOY_START'].copy()
        # self.df['DOY_START3'] = self.df['DOY_START'].copy()
        # self.df['DOY_START4'] = self.df['DOY_START'].copy()
        # rename = {'DOY_START2': 'DOY_START', 'DOY_START3': 'DOY_START', 'DOY_START4': 'DOY_START'}
        # self.df.rename(columns=rename, inplace=True)

        self._run()

    def get(self) -> DataFrame:
        return self.df

    def _run(self):
        self._rename_duplicate_columns()

    def _rename_duplicate_columns(self):

        # Identify the duplicate columns
        duplicate_colnames_loc = self.df.columns.duplicated()

        # List of duplicate column names
        duplicate_names = self.df.loc[:, duplicate_colnames_loc].columns.to_list()
        uniq_duplicate_names = list(set(duplicate_names))

        if uniq_duplicate_names:
            print(f"(!)WARNING: Duplicate column names found")
            print(f"(!)WARNING: Duplicate columns will be renamed")

            for u in uniq_duplicate_names:
                occurrences = len(self.df[u].columns)
                print(f"(!)WARNING There are {occurrences} columns with name {u}")

            newcols = []
            for col in self.df.columns:
                if col not in newcols:
                    pass
                else:
                    # Add integer suffix
                    counter = 0
                    newcol = col
                    while newcol in newcols:
                        # Maybe the renamed column with suffix already exists, therefore
                        # increase the integer suffix until a free name is found
                        counter += 1
                        newcol = f'{col}.{counter}'
                    print(f"(!)WARNING Duplicate column names found: {col} was renamed to --> {newcol}")
                    col = newcol
                newcols.append(col)

            self.df.columns = newcols


class MultiDataFileReader:
    """Read and merge multiple datafiles of the same filetype"""

    def __init__(self, filepaths: list, filetype: str, output_middle_timestamp: bool = True):

        # Getting configs for filetype
        configfilepath = get_filetypes()[filetype]
        self.filetypeconfig = ConfigFileReader(configfilepath=configfilepath, validation='filetype').read()
        self.filepaths = filepaths
        self.output_middle_timestamp = output_middle_timestamp

        # Collect data from all files listed in filepaths
        self._data_df, self._metadata_df = self._get_incoming_data()

        self._data_df = continuous_timestamp_freq(data=self._data_df, freq=self.filetypeconfig['DATA']['FREQUENCY'])

    @property
    def data_df(self):
        """Get dataframe of merged files data"""
        if not isinstance(self._data_df, DataFrame):
            raise Exception('data is empty')
        return self._data_df

    @property
    def metadata_df(self):
        """Get dataframe of merged files metadata"""
        if not isinstance(self._data_df, DataFrame):
            raise Exception('metadata is empty')
        return self._metadata_df

    def _get_incoming_data(self) -> tuple[DataFrame, DataFrame]:
        """Merge data across all files"""
        data_df = None
        metadata_df = None
        for filepath in self.filepaths:
            # print(f"\n{'-' * 40}\nReading file {filepath.stem}\n{'-' * 40}")
            try:
                incoming_data_df, incoming_metadata_df = \
                    ReadFileType(filepath=filepath, filetypeconfig=self.filetypeconfig,
                                 output_middle_timestamp=self.output_middle_timestamp).get_filedata()
                data_df, metadata_df = \
                    self._merge_with_existing(incoming_data_df=incoming_data_df, data_df=data_df,
                                              incoming_metadata_df=incoming_metadata_df, metadata_df=metadata_df)
            except pandas.errors.EmptyDataError:
                continue
        data_df = sort_multiindex_columns_names(df=data_df, priority_vars=None)
        return data_df, metadata_df

    def _merge_with_existing(self,
                             incoming_data_df: DataFrame, data_df: DataFrame,
                             incoming_metadata_df: DataFrame, metadata_df: DataFrame
                             ) -> tuple[DataFrame, DataFrame]:
        if not isinstance(data_df, DataFrame):
            data_df = incoming_data_df.copy()
            metadata_df = incoming_metadata_df.copy()
        else:
            data_df = data_df.combine_first(incoming_data_df)
            metadata_df = metadata_df.combine_first(incoming_metadata_df)
        return data_df, metadata_df


class ReadFileType:
    """Read single data file using settings from dictionary for specified filetype"""

    def __init__(self,
                 filepath: str or Path,
                 filetypeconfig: dict = None,
                 filetype: str = None,
                 data_nrows: int = None,
                 output_middle_timestamp: bool = True):
        """

        Args:
            filepath:
            filetypeconfig:
            filetype:
        """
        self.filepath = Path(filepath)
        self.data_nrows = data_nrows
        self.output_middle_timestamp = output_middle_timestamp

        if filetype:
            # Read settins for specified filetype
            _available_filetypes = get_filetypes()
            self.filetypeconfig = ConfigFileReader(_available_filetypes[filetype], validation='filetype').read()
        else:
            # Use provided settings dict
            self.filetypeconfig = filetypeconfig

        self.data_df, self.metadata_df = self._readfile()

    def get_filedata(self) -> tuple[DataFrame, DataFrame]:
        return self.data_df, self.metadata_df

    def _readfile(self) -> tuple[DataFrame, DataFrame]:
        """Load data"""
        print(f"Reading file {self.filepath.name} ...")
        datafilereader = DataFileReader(
            filepath=self.filepath,
            data_skip_rows=self.filetypeconfig['DATA']['SKIP_ROWS'],
            data_header_rows=self.filetypeconfig['DATA']['HEADER_ROWS'],
            data_header_section_rows=self.filetypeconfig['DATA']['HEADER_SECTION_ROWS'],
            data_varnames_row=self.filetypeconfig['DATA']['VARNAMES_ROW'],
            data_varunits_row=self.filetypeconfig['DATA']['VARUNITS_ROW'],
            data_na_vals=self.filetypeconfig['DATA']['NA_VALUES'],
            data_delimiter=self.filetypeconfig['DATA']['DELIMITER'],
            data_freq=self.filetypeconfig['DATA']['FREQUENCY'],
            data_nrows=self.data_nrows,
            timestamp_idx_col=self.filetypeconfig['TIMESTAMP']['INDEX_COLUMN'],
            timestamp_datetime_format=self.filetypeconfig['TIMESTAMP']['DATETIME_FORMAT'],
            timestamp_start_middle_end=self.filetypeconfig['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'],
            output_middle_timestamp=self.output_middle_timestamp,
            compression=self.filetypeconfig['FILE']['COMPRESSION']
        )
        data_df, metadata_df = datafilereader.get_data()
        return data_df, metadata_df


class DataFileReader:
    """Read single data file with *provided settings*"""

    def __init__(
            self,
            filepath: Path or str,
            data_header_section_rows: list = None,
            data_skip_rows: list = None,
            data_header_rows: list = None,
            data_varnames_row: int = None,
            data_varunits_row: int = None,
            data_na_vals: list = None,
            data_delimiter: str = None,
            data_freq: str = None,
            data_nrows: int = None,
            timestamp_idx_col: list = None,
            timestamp_datetime_format: str = None,
            timestamp_start_middle_end: str = 'END',
            output_middle_timestamp: bool = True,
            compression: str = None
    ):

        self.filepath = filepath if isinstance(filepath, Path) else Path(filepath)

        self.data_header_section_rows = data_header_section_rows
        self.data_skip_rows = data_skip_rows
        self.data_header_rows = data_header_rows
        self.data_varnames_row = data_varnames_row
        self.data_varunits_row = data_varunits_row
        self.data_na_vals = data_na_vals
        self.data_freq = data_freq
        self.data_delimiter = data_delimiter
        self.data_nrows = data_nrows

        self.timestamp_datetime_format = timestamp_datetime_format
        self.timestamp_start_middle_end = timestamp_start_middle_end
        self.timestamp_idx_col = timestamp_idx_col
        self.output_middle_timestamp = output_middle_timestamp
        self.compression = compression

        self.data_df = pd.DataFrame()
        self.metadata_df = pd.DataFrame()
        self.generated_missing_header_cols_list = []

        self._read()

    def _read(self):

        # Parse variable names and units, get number of columns for header
        varnames_list, varunits_list, n_cols_header = parse_header(
            filepath=self.filepath,
            skiprows=self.data_skip_rows,
            headerrows=self.data_header_rows,
            varnames_row=self.data_varnames_row,
            varunits_row=self.data_varunits_row)

        # Get number of columns for data
        n_cols_data = get_len_data(
            filepath=self.filepath,
            skiprows=self.data_skip_rows,
            headerrows=self.data_header_rows)

        # Check for missing header columns (in case there are more data cols than header cols)
        varnames_list, varunits_list, self.generated_missing_header_cols_list = compare_len_header_vs_data(
            n_cols_data=n_cols_data,
            n_cols_header=n_cols_header,
            varnames_list=varnames_list,
            varunits_list=varunits_list
        )

        # Parse file
        self.data_df = self._parse_file(varnames_list=varnames_list)

        # Sanitize timestamp
        if self.timestamp_idx_col:
            self.data_df = TimestampSanitizer(data=self.data_df,
                                              output_middle_timestamp=self.output_middle_timestamp,
                                              nominal_freq=self.data_freq).get()

        # Make data numeric
        self.data_df = convert_data_to_numeric(df=self.data_df)

        # Get metadata, info about variables, units and more
        self.metadata_df = self._get_metadata(varnames_list=varnames_list, varunits_list=varunits_list)

        # self.data_df = flatten_multiindex_all_df_cols(df=self.data_df, keep_first_row_only=True)

        self.data_df = ColumnNamesSanitizer(df=self.data_df).get()

    def _get_metadata(self, varnames_list, varunits_list) -> DataFrame:
        """Create separate dataframe collecting metadata for each variable
        Metadata contains units, tags and time added.
        """
        # _flatcols = [col[0] for col in self.data_df.columns]
        index = ['UNITS', 'TAGS', 'ADDED']
        data_metadata_df = pd.DataFrame(columns=varnames_list, index=index)
        addtime = datetime.datetime.now()  # Datetime of when var was added
        for varix, varname in enumerate(self.data_df.columns):
            data_metadata_df.loc['UNITS', varname] = varunits_list[varix]
            data_metadata_df.loc['TAGS', varname] = ["#orig"]
            data_metadata_df.loc['ADDED', varname] = addtime
            data_metadata_df.loc['VARINDEX', varname] = varix
        data_metadata_df = data_metadata_df.transpose()
        return data_metadata_df

    def get_data(self) -> tuple[DataFrame, DataFrame]:
        return self.data_df, self.metadata_df

    def _convert_timestamp_idx_col(self, var):
        """Convert to list of tuples if needed

        Since YAML is not good at processing list of tuples,
        they are given as list of lists,
            e.g. [ [ "date", "[yyyy-mm-dd]" ], [ "time", "[HH:MM]" ] ].
        In this case, convert to list of tuples,
            e.g.  [ ( "date", "[yyyy-mm-dd]" ), ( "time", "[HH:MM]" ) ].
        """
        new = var
        if isinstance(var[0], int):
            pass
        elif isinstance(var[0], list):
            for idx, c in enumerate(var):
                new[idx] = (c[0], c[1])
        return new

    def _configure_timestamp_parsing(self):
        """Configure column settings for parsing dates / times correctly."""

        # Check filetype settings info about timestamp
        # Column name for timestamp index
        parsed_index_col = f"TIMESTAMP_{self.timestamp_start_middle_end.upper()}"

        # Create temporary column name for parsed index :: v0.41.0
        # This solves the issue that a column named *parsed_index_col* might already be
        # present in the dataset. In that case, parsing would not work and the code would
        # raise an exception (e.g., "ValueError: Date column TIMESTAMP_END already in dict").
        # The temporary column name will be changed to the correct name after data were
        # read (see below).
        _temp_parsed_index_col = f"_temp_{parsed_index_col}"

        parse_dates = self.timestamp_idx_col  # Columns used for creating the timestamp index
        parse_dates = {_temp_parsed_index_col: parse_dates}
        # date_parser = lambda x: dt.datetime.strptime(x, self.timestamp_datetime_format)
        # date_parser = lambda x: pd.to_datetime(x, format=self.timestamp_datetime_format, errors='coerce')
        return parse_dates, parsed_index_col, _temp_parsed_index_col

    def _parse_file(self, varnames_list):
        """Parse data file without parsing the header, columns names are given in *names* kwarg."""

        # if self.timestamp_idx_col:
        #     parse_dates, parsed_index_col, _temp_parsed_index_col = self._configure_timestamp_parsing()

        # TODO hier
        df = pd.read_csv(
            filepath_or_buffer=self.filepath,
            skiprows=self.data_header_section_rows,
            header=None,
            na_values=self.data_na_vals,  # warning
            encoding='utf-8',
            delimiter=self.data_delimiter,
            index_col=None,
            engine='python',  # todo 'python', 'c'
            nrows=self.data_nrows,
            compression=self.compression,
            on_bad_lines='warn',
            names=varnames_list,
            skip_blank_lines=True,
            dtype=None
            # date_format=self.timestamp_datetime_format,
            # parse_dates=self.timestamp_idx_col,
            # parse_dates=parse_dates,
            # keep_date_col=False,
            # usecols=None,
            # mangle_dupe_cols=True,  # deprecated since pandas 2.0
            # date_parser=date_parser,  # deprecated since pandas 2.0
        )

        if self.timestamp_idx_col:
            df = self._parse_timestamp_index(df=df)

        return df

    def _parse_timestamp_index(self, df: DataFrame) -> DataFrame:
        # Column name for timestamp index
        parsed_index_col = f"TIMESTAMP_{self.timestamp_start_middle_end.upper()}"

        # Create temporary column name for parsed index
        # This solves the issue that a column named *parsed_index_col* might already be
        # present in the dataset. In that case, parsing would not work and the code would
        # raise an exception (e.g., "ValueError: Date column TIMESTAMP_END already in dict").
        # The temporary column name will be changed to the correct name after data were
        # read (see below).
        _temp_parsed_index_col = f"_temp_{parsed_index_col}"

        # Construct timestamp as string
        df[_temp_parsed_index_col] = ''
        for col in self.timestamp_idx_col:

            # Column can be given as string or integer
            if isinstance(col, str):
                tscol = df[col].astype(str)
            elif isinstance(col, int):
                tscol = df.iloc[:, col].astype(str)
            else:
                raise TypeError(f"Error: unable to parse timestamp column with setting INDEX_COLUMN: {col}.")

            df[_temp_parsed_index_col] = df[_temp_parsed_index_col] + tscol
            df = df.drop(tscol.name, axis=1, inplace=False)  # Remove col from df

            if col != self.timestamp_idx_col[-1]:
                df[_temp_parsed_index_col] += ' '

        # Convert parsed timestamp string to datetime
        df[_temp_parsed_index_col] = pd.to_datetime(df[_temp_parsed_index_col],
                                                    format=self.timestamp_datetime_format, errors='coerce')

        # Check if column with the same name as *parsed_index_col* exists and drop
        if parsed_index_col in df.columns:
            df = df.drop(parsed_index_col, axis=1)

        # Rename temporary column name for parsed index to correct name
        df = rename_cols(df=df, renaming_dict={_temp_parsed_index_col: parsed_index_col})
        df = df.set_index(parsed_index_col, inplace=False)
        return df


def example_ep_fluxnet():
    from diive.core.times.times import insert_timestamp, format_timestamp_to_fluxnet_format

    FOLDER = r"F:\TMP\FRU\Level-1_FF-202303_FF-202401_2005-2023\Level-1_results_fluxnet"
    OUTDIR = r"L:\Sync\luhk_work\CURRENT\fru_prep"

    filepaths = search_files(FOLDER, "*.csv")
    # filepaths = [fp for fp in filepaths
    #              if fp.stem.startswith("eddypro_")
    #              and "_fluxnet_" in fp.stem
    #              and fp.stem.endswith("_adv")]
    print(filepaths)

    loaddatafile = MultiDataFileReader(filetype='EDDYPRO-FLUXNET-CSV-30MIN', filepaths=filepaths)
    df = loaddatafile.data_df

    # # Store original column order
    # orig_col_order = df.columns

    # Set all missing values to -9999 as required by FLUXNET
    df = df.fillna(-9999)

    # Add timestamp column TIMESTAMP_END
    df = insert_timestamp(data=df, convention='end', insert_as_first_col=True, verbose=True)
    # Add timestamp column TIMESTAMP_START
    df = insert_timestamp(data=df, convention='start', insert_as_first_col=True, verbose=True)

    print("\nAdjusting timestamp formats of TIMESTAMP_START and TIMESTAMP_END to %Y%m%d%H%M ...")
    df['TIMESTAMP_END'] = format_timestamp_to_fluxnet_format(df=df, timestamp_col='TIMESTAMP_END')
    df['TIMESTAMP_START'] = format_timestamp_to_fluxnet_format(df=df, timestamp_col='TIMESTAMP_START')

    # # Restore original column order
    # df = df[orig_col_order].copy()

    outpath = Path(OUTDIR) / 'merged.csv'
    df.to_csv(outpath, index=False)


def example_toa5():
    corrected = r"C:\Users\holukas\Downloads\corrected_files\c-CH-OE2_iDL_BOX1_0_1_TBL1_20220629-1714.dat"
    uncorrected = r"C:\Users\holukas\Downloads\corrected_files\CH-OE2_iDL_BOX1_0_1_TBL1_20220629-1714.dat"

    corr_df, corr_meta = ReadFileType(filepath=corrected, filetype='TOA5-DAT-1MIN',
                                      output_middle_timestamp=True).get_filedata()
    uncorr_df, uncorr_meta = ReadFileType(filepath=uncorrected, filetype='TOA5-DAT-1MIN',
                                          output_middle_timestamp=True).get_filedata()

    corr_descr = corr_df.describe()
    uncorr_descr = uncorr_df.describe()

    for c in corr_descr.columns:
        c_corr_descr = corr_descr[c]
        c_uncorr_descr = uncorr_descr[c]
        checkok = c_corr_descr.equals(c_uncorr_descr)
        if checkok:
            print(f"OK  -  Variable {c} is the same in both files")
        else:
            print(f"{'#' * 40}\n"
                  f"### FABIO EMERGENCY  -  Variable {c} is not the same in both files\n")
            print("Stats of CORRECTED:")
            print(c_corr_descr)
            print("Stats of uncorrected:")
            print(c_uncorr_descr)
            print(f"{'#' * 40}")


if __name__ == '__main__':
    example_ep_fluxnet()
    # example_toa5()
