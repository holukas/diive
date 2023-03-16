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
from pandas.io.parsers.base_parser import ParserBase

from diive import core
from diive.configs.filetypes import get_filetypes
# from diive.core.dfun.frames import rename_cols, sort_multiindex_columns_names, \
#     flatten_multiindex_all_df_cols, get_len_header, get_len_data
from diive.core import dfun
# from diive.core.dfun.frames import flatten_multiindex_all_df_cols
from diive.core.times.times import continuous_timestamp_freq, TimestampSanitizer


def search_files(searchdirs: str or list, pattern: str) -> list:
    """ Search files and store their filename and the path to the file in dictionary. """
    # found_files_dict = {}
    foundfiles = []
    if isinstance(searchdirs, str): searchdirs = [searchdirs]  # Use str as list
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

    config['TIMESTAMP']['INDEX_COLUMN'] = list(config['TIMESTAMP']['INDEX_COLUMN'])
    config['TIMESTAMP']['INDEX_COLUMN'] = \
        _convert_timestamp_idx_col(var=config['TIMESTAMP']['INDEX_COLUMN'])

    config['TIMESTAMP']['DATETIME_FORMAT'] = str(config['TIMESTAMP']['DATETIME_FORMAT'])
    config['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'] = str(
        config['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'])

    # DATA
    config['DATA']['HEADER_SECTION_ROWS'] = list(config['DATA']['HEADER_SECTION_ROWS'])
    config['DATA']['SKIP_ROWS'] = list(config['DATA']['SKIP_ROWS'])
    config['DATA']['HEADER_ROWS'] = list(config['DATA']['HEADER_ROWS'])
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


def rename_duplicate_cols(df: DataFrame) -> DataFrame:
    """
    Rename columns with the same name to unique names

    For example:
        - two columns with the same name in *df*:
            'co2_mean' and 'co2_mean'
            will be renamed to
            'co2_mean.1' and 'co2_mean.2'

    Args:
        df: DataFrame with multiple columns

    Returns:
        Complete df with renamed columns. If no duplicates
        were found, the returned df is the same as the input df.

    Kudos:
    - https://stackoverflow.com/questions/24685012/pandas-dataframe-renaming-multiple-identically-named-columns

    """
    df.columns = ParserBase({'usecols': None})._maybe_dedup_names(df.columns)
    return df


class ColumnNamesSanitizer:

    def __init__(self,
                 df: DataFrame):
        self.df = df.copy()
        self._run()

    def get(self) -> DataFrame:
        return self.df

    def _run(self):
        self.df = rename_duplicate_cols(df=self.df)


class MultiDataFileReader:
    """Read and merge multiple datafiles of the same filetype"""

    def __init__(self, filepaths: list, filetype: str, output_middle_timestamp:bool=True):

        # Getting configs for filetype
        configfilepath = get_filetypes()[filetype]
        self.filetypeconfig = ConfigFileReader(configfilepath=configfilepath, validation='filetype').read()
        self.filepaths = filepaths
        self.output_middle_timestamp=output_middle_timestamp

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
        data_df = dfun.frames.sort_multiindex_columns_names(df=data_df, priority_vars=None)
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
                 output_middle_timestamp:bool=True):
        """

        Args:
            filepath:
            filetypeconfig:
            filetype:
        """
        self.filepath = Path(filepath)
        self.data_nrows = data_nrows
        self.output_middle_timestamp=output_middle_timestamp

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
            data_skiprows=self.filetypeconfig['DATA']['SKIP_ROWS'],
            data_headerrows=self.filetypeconfig['DATA']['HEADER_ROWS'],
            data_headersection_rows=self.filetypeconfig['DATA']['HEADER_SECTION_ROWS'],
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
            filepath: Path,
            data_skiprows: list = None,
            data_headerrows: list = None,
            data_headersection_rows: list = None,
            data_na_vals: list = None,
            data_delimiter: str = None,
            data_freq: str = None,
            data_nrows: int = None,
            timestamp_idx_col: list = None,
            timestamp_datetime_format: str = None,
            timestamp_start_middle_end: str = 'END',
            output_middle_timestamp:bool=True,
            compression:str=None
    ):

        self.filepath = filepath
        self.data_skiprows = data_skiprows
        self.data_headerrows = data_headerrows
        self.data_headersection_rows = data_headersection_rows
        self.data_na_vals = data_na_vals
        self.data_delimiter = data_delimiter
        self.data_freq = data_freq
        self.data_nrows = data_nrows
        self.timestamp_datetime_format = timestamp_datetime_format
        self.timestamp_start_middle_end = timestamp_start_middle_end
        self.timestamp_idx_col = timestamp_idx_col
        self.output_middle_timestamp=output_middle_timestamp
        self.compression=compression

        self.data_df = pd.DataFrame()
        self.metadata_df = pd.DataFrame()
        self.generated_missing_header_cols_list = []

        self._read()

    def _read(self):
        headercols_list, self.generated_missing_header_cols_list = self._compare_len_header_vs_data()
        self.data_df = self._parse_file(headercols_list=headercols_list)
        self.data_df = TimestampSanitizer(data=self.data_df,
                                          output_middle_timestamp=self.output_middle_timestamp).get()
        self._clean_data()
        if len(self.data_headerrows) == 1:
            self.data_df = self._add_second_header_row(df=self.data_df)
        self.metadata_df = self._get_metadata()
        self.data_df = dfun.frames.flatten_multiindex_all_df_cols(df=self.data_df, keep_first_row_only=True)

        self.data_df = ColumnNamesSanitizer(df=self.data_df).get()

    def _get_metadata(self):
        """Create separate dataframe collecting metadata for each variable
        Metadata contains units, tags and time added.
        """
        _flatcols = [col[0] for col in self.data_df.columns]
        index = ['UNITS', 'TAGS', 'ADDED']
        data_metadata_df = pd.DataFrame(columns=_flatcols, index=index)
        addtime = datetime.datetime.now()  # Datetime of when var was added
        for ix, col in enumerate(self.data_df.columns):
            var = col[0]
            data_metadata_df.loc['UNITS', var] = col[1]
            data_metadata_df.loc['TAGS', var] = ["#orig"]
            data_metadata_df.loc['ADDED', var] = addtime
            data_metadata_df.loc['VARINDEX', var] = ix
        data_metadata_df = data_metadata_df.T
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

    def _add_second_header_row(self, df):
        """Check if there is only one header row, if yes, then add second row (standardization)"""
        lst_for_empty_units = []
        for e in range(len(df.columns)):  ## generate entry for all cols in df
            lst_for_empty_units.append('-no-units-')
        df.columns = [df.columns, lst_for_empty_units]  ## conv column index to multiindex
        return df

    def _compare_len_header_vs_data(self):
        """
        Check whether there are more data columns than given in the header

        If not checked, this would results in an error when reading the csv file
        with .read_csv, because the method expects an equal number of header and
        data columns. If this check is True, then the difference between the length
        of the first data row and the length of the header row(s) can be used to
        automatically generate names for the missing header columns.
        """
        num_headercols, headercols_list = dfun.frames.get_len_header(filepath=self.filepath,
                                                                     skiprows=self.data_skiprows,
                                                                     headerrows=self.data_headerrows)
        num_datacols = dfun.frames.get_len_data(filepath=self.filepath,
                                                skiprows=self.data_skiprows,
                                                headerrows=self.data_headerrows)

        # Check if there are more data columns than header columns
        more_data_cols_than_header_cols = False
        num_missing_header_cols = 0
        if num_datacols > num_headercols:
            more_data_cols_than_header_cols = True
            num_missing_header_cols = num_datacols - num_headercols

        # Generate missing header columns if necessary
        generated_missing_header_cols_list = []
        sfx = core.times.times.current_time_microseconds_str()
        if more_data_cols_than_header_cols:
            for m in list(range(1, num_missing_header_cols + 1)):
                missing_col = (f'unknown-{m}-{sfx}', '[-unknown-]')
                generated_missing_header_cols_list.append(missing_col)
                headercols_list.append(missing_col)

        return headercols_list, generated_missing_header_cols_list

    def _clean_data(self):
        """Sanitize time series"""
        # Sanitize time series, numeric data is needed
        # After this conversion, all columns are of float64 type, strings will be substituted
        # by NaN. This means columns that contain only strings, e.g. the columns 'date' or
        # 'filename' in the EddyPro full_output file, contain only NaNs after this step.
        # Not too problematic in case of 'date', b/c the index contains the datetime info.
        # todo For now, columns that contain only NaNs are still in the df.
        # todo at some point, the string columns should also be considered
        self.data_df = self.data_df.apply(pd.to_numeric, errors='coerce')

    def _parse_file(self, headercols_list):
        """Parse data file without header"""

        # Column settings for parsing dates / times correctly

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
        date_parser = lambda x: pd.to_datetime(x, format=self.timestamp_datetime_format, errors='coerce')

        data_df = pd.read_csv(
            self.filepath,
            skiprows=self.data_headersection_rows,
            header=None,
            names=headercols_list,
            na_values=self.data_na_vals,
            encoding='utf-8',
            delimiter=self.data_delimiter,
            mangle_dupe_cols=True,
            keep_date_col=False,
            parse_dates=parse_dates,
            date_parser=date_parser,
            index_col=None,
            dtype=None,
            skip_blank_lines=True,
            nrows=self.data_nrows,
            engine='python',  # todo 'python', 'c'
            compression=self.compression
        )

        # Rename temporary column name for parsed index to correct name (v0.41.0)
        data_df = dfun.frames.rename_cols(df=data_df, renaming_dict={_temp_parsed_index_col: parsed_index_col})
        # if parsed_index_col in headercols_list:
        #     headercols_list = [item.replace(parsed_index_col, f"{parsed_index_col}_ORIGINAL") for item in headercols_list]

        data_df.set_index(parsed_index_col, inplace=True)

        return data_df

    # def _standardize_timestamp_index(self):
    #     """Standardize timestamp index column"""
    #
    #     # Index name is now the same for all filetypes w/ timestamp in data
    #     self.data_df.set_index([('index', '[parsed]')], inplace=True)
    #     self.data_df.index.name = ('TIMESTAMP', '[yyyy-mm-dd HH:MM:SS]')
    #
    #     # Make sure the index is datetime
    #     self.data_df.index = pd.to_datetime(self.data_df.index)
    #
    #     # Make continuous timestamp at current frequency, from first to last datetime
    #     self.data_df = continuous_timestamp_freq(data=self.data_df, freq=self.data_freq)
    #
    #     # TODO? additional check: validate inferred freq from data like in `dataflow` script
    #
    #     # Timestamp convention
    #     # Shift timestamp by half-frequency, if needed
    #     if self.timestamp_start_middle_end == 'middle':
    #         pass
    #     else:
    #         timedelta = pd.to_timedelta(self.data_freq) / 2
    #         if self.timestamp_start_middle_end == 'end':
    #             self.data_df.index = self.data_df.index - pd.Timedelta(timedelta)
    #         elif self.timestamp_start_middle_end == 'start':
    #             self.data_df.index = self.data_df.index + pd.Timedelta(timedelta)


def example():
    import os
    from pathlib import Path
    FOLDER = r"Z:\CH-FRU_Fruebuel\20_ec_fluxes\2023\Level-0"
    filepaths = search_files(FOLDER, "*.csv")
    filepaths = [fp for fp in filepaths
                 if fp.stem.startswith("eddypro_")
                 and "_fluxnet_" in fp.stem
                 and fp.stem.endswith("_adv")]
    print(filepaths)

    loaddatafile = MultiDataFileReader(filetype='EDDYPRO_FLUXNET_30MIN', filepaths=filepaths)


if __name__ == '__main__':
    example()
