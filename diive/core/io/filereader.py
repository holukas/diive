"""
FILEREADER
==========
This package is part of the diive library.

"""
import datetime
import fnmatch
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.errors
import yaml
from pandas import DataFrame

from diive import core
from diive.configs.filetypes import get_filetypes
from diive.core import dfun
from diive.core.dfun.frames import flatten_multiindex_all_df_cols
from diive.core.times.times import continuous_timestamp_freq


def search_files(searchdir, pattern: str) -> list:
    """ Search files and store their filename and the path to the file in dictionary. """
    # found_files_dict = {}
    foundfiles = []
    for root, dirs, files in os.walk(searchdir):
        for idx, settings_file_name in enumerate(files):
            if fnmatch.fnmatch(settings_file_name, pattern):
                filepath = Path(root) / settings_file_name
                # found_files_dict[settings_file_name] = filepath
                foundfiles.append(filepath)
    # return found_files_dict
    foundfiles.sort()
    return foundfiles


class ConfigFileReader:

    def __init__(self, configfilepath: Path or str):
        self.configfilepath = Path(configfilepath)

    def read(self) -> dict:
        """
        Load configuration from YAML file and store in dict

        kudos: https://stackoverflow.com/questions/57687058/yaml-safe-load-special-character-from-file

        :param config_file: YAML file with configuration
        :return: dict
        """
        with open(self.configfilepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config


class MultiDataFileReader:
    """Read and merge multiple datafiles of the same filetype"""

    def __init__(self, filepaths: list, filetype: str):

        # Getting configs for filetype
        configfilepath = get_filetypes()[filetype]
        self.filetypeconfig = ConfigFileReader(configfilepath=configfilepath).read()
        self.filepaths = filepaths

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
            print(f"Reading file {filepath.stem}")
            try:
                incoming_data_df, incoming_metadata_df = \
                    ReadFileType(filepath=filepath, filetypeconfig=self.filetypeconfig).get_filedata()
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
                 data_nrows: int=None):
        """

        Args:
            filepath:
            filetypeconfig:
            filetype:
        """
        self.filepath = Path(filepath)
        self.data_nrows = data_nrows

        if filetype:
            # Read settins for specified filetype
            _available_filetypes = get_filetypes()
            self.filetypeconfig = ConfigFileReader(_available_filetypes[filetype]).read()
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
            timestamp_start_middle_end=self.filetypeconfig['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD']
        )
        data_df, metadata_df = datafilereader.get_data()
        return data_df, metadata_df


class DataFileReader:
    """Read single data file with provided settings"""

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
            timestamp_start_middle_end: str = 'END'
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

        self.data_df = pd.DataFrame()
        self.metadata_df = pd.DataFrame()
        self.generated_missing_header_cols_list = []

        self._read()

    def _read(self):
        headercols_list, self.generated_missing_header_cols_list = self._compare_len_header_vs_data()
        self.data_df = self._parse_file(headercols_list=headercols_list)
        self._standardize_timestamp_index()
        self._clean_data()
        if len(self.data_headerrows) == 1:
            self.data_df = self._add_second_header_row(df=self.data_df)
        self.metadata_df = self._get_metadata()
        self.data_df = flatten_multiindex_all_df_cols(df=self.data_df, keep_first_row_only=True)

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
            data_metadata_df.loc['TAGS', var] = "#orig"
            data_metadata_df.loc['ADDED', var] = addtime
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

        # There exist certain instances where the float64 data column can contain
        # non-numeric values that are interpreted as a float64 inf, which is basically
        # a NaN value. To harmonize missing values inf is also set to NaN.
        self.data_df.replace(float('inf'), np.nan, inplace=True)
        self.data_df.replace(float('-inf'), np.nan, inplace=True)

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
        parsed_index_col = ('index', '[parsed]')  # Column name for timestamp index
        parse_dates = self.timestamp_idx_col  # Columns used for creating the timestamp index
        parse_dates = {parsed_index_col: parse_dates}
        # date_parser = lambda x: dt.datetime.strptime(x, self.timestamp_datetime_format)
        date_parser = lambda x: pd.to_datetime(x, format=self.timestamp_datetime_format, errors='coerce')

        data_df = pd.read_csv(self.filepath,
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
                              engine='python')  # todo 'python', 'c'

        return data_df

    def _standardize_timestamp_index(self):
        """Standardize timestamp index column"""

        # Index name is now the same for all filetypes w/ timestamp in data
        self.data_df.set_index([('index', '[parsed]')], inplace=True)
        self.data_df.index.name = ('TIMESTAMP', '[yyyy-mm-dd HH:MM:SS]')

        # Make sure the index is datetime
        self.data_df.index = pd.to_datetime(self.data_df.index)

        # Make continuous timestamp at current frequency, from first to last datetime
        self.data_df = continuous_timestamp_freq(data=self.data_df, freq=self.data_freq)

        # TODO? additional check: validate inferred freq from data like in `dataflow` script

        # Timestamp convention
        # Shift timestamp by half-frequency, if needed
        if self.timestamp_start_middle_end == 'middle':
            pass
        else:
            timedelta = pd.to_timedelta(self.data_freq) / 2
            if self.timestamp_start_middle_end == 'end':
                self.data_df.index = self.data_df.index - pd.Timedelta(timedelta)
            elif self.timestamp_start_middle_end == 'start':
                self.data_df.index = self.data_df.index + pd.Timedelta(timedelta)


def example():
    # Load data
    filepath = Path(
        "M:\Downloads\Warm Winter 2020 ecosystem eddy covariance flux product for 73 stations in FLUXNET-Archive format—release 2022-1\Swiss_Sites\FLX_CH-Dav_FLUXNET2015_FULLSET_1997-2020_beta-3\FLX_CH-Dav_FLUXNET2015_FULLSET_HH_1997-2020_beta-3.csv")
    loaddatafile = ReadFileType(filetype='FLUXNET-FULLSET-HH-CSV-30MIN', filepath=filepath)
    data_df, metadata_df = loaddatafile._readfile()


if __name__ == '__main__':
    example()
