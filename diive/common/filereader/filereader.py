"""
FILEREADER
==========
This package is part of DIIVE.

    DataFileReader: Read data files to pandas DataFrame
    ConfigFileReader: Read YAML file to dictionary

"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from diive.common import dfun


class ConfigFileReader:

    def __init__(self, configfilepath: Path):
        self.configfilepath = configfilepath

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
    """Read and merge multiple datafiles with same config"""

    def __init__(self, filetype_config, filepaths: list):
        self.filepaths = filepaths

        self.data_skiprows = filetype_config['DATA']['SKIP_ROWS']
        self.data_headerrows = filetype_config['DATA']['HEADER_ROWS']
        self.data_headersection_rows = filetype_config['DATA']['HEADER_SECTION_ROWS']
        self.data_na_vals = filetype_config['DATA']['NA_VALUES']
        self.data_delimiter = filetype_config['DATA']['DELIMITER']
        self.data_freq = filetype_config['DATA']['FREQUENCY']
        self.timestamp_idx_col = filetype_config['TIMESTAMP']['INDEX_COLUMN']
        self.timestamp_datetime_format = filetype_config['TIMESTAMP']['DATETIME_FORMAT']
        self.timestamp_start_middle_end = filetype_config['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD']

        self.filesdata_df = pd.DataFrame()  # Data from all files in filepaths

    def get_incoming_data(self):
        for filepath in self.filepaths:
            incoming_filedata_df = self._readfile(filepath=filepath)
            self._merge_with_existing(incoming_filedata_df=incoming_filedata_df)
        self.filesdata_df = dfun.frames.sort_multiindex_columns_names_LEGACY(df=self.filesdata_df, priority_vars=None)
        return self.filesdata_df

    def _merge_with_existing(self, incoming_filedata_df):
        if self.filesdata_df.empty:
            self.filesdata_df = incoming_filedata_df
        else:
            self.filesdata_df = self.filesdata_df.combine_first(incoming_filedata_df)

    def _readfile(self, filepath):
        incoming_filedata_df = DataFileReader(
            filepath=filepath,
            data_skiprows=self.data_skiprows,
            data_headerrows=self.data_headerrows,
            data_headersection_rows=self.data_headersection_rows,
            data_na_vals=self.data_na_vals,
            data_delimiter=self.data_delimiter,
            data_freq=self.data_freq,
            timestamp_idx_col=self.timestamp_idx_col,
            timestamp_datetime_format=self.timestamp_datetime_format,
            timestamp_start_middle_end=self.timestamp_start_middle_end
        )._read()

        return incoming_filedata_df


class DataFileReader:
    """Read datafile"""

    def __init__(
            self,
            filepath: Path,
            data_skiprows: list = None,
            data_headerrows: list = None,
            data_headersection_rows: list = None,
            data_na_vals: list = None,
            data_delimiter: str = None,
            data_freq: str = None,
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
        self.timestamp_datetime_format = timestamp_datetime_format
        self.timestamp_start_middle_end = timestamp_start_middle_end
        self.timestamp_idx_col = timestamp_idx_col

        self.data_df = pd.DataFrame()
        self.generated_missing_header_cols_list = []

        self._read()

    def _read(self):
        headercols_list, self.generated_missing_header_cols_list = self._compare_len_header_vs_data()
        self.data_df = self._parse_file(headercols_list=headercols_list)
        self._standardize_index()
        self._clean_data()
        if len(self.data_headerrows) == 1:
            self.data_df = self._add_second_header_row(df=self.data_df)

    def get_data(self) -> pd.DataFrame:
        return self.data_df

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
        sfx = dfun.times.make_timestamp_microsec_suffix()
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

        # Set freq, at the same time this makes index CONTINUOUS w/o date gaps
        self.data_df.asfreq(self.data_freq)

        self.data_df.sort_index(inplace=True)

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
                              engine='python')

        return data_df

    def _standardize_index(self):
        """Standardize timestamp index column"""

        # Index name is now the same for all filetypes w/ timestamp in data
        self.data_df.set_index([('index', '[parsed]')], inplace=True)
        self.data_df.index.name = ('TIMESTAMP', '[yyyy-mm-dd HH:MM:SS]')

        # Make sure the index is datetime
        self.data_df.index = pd.to_datetime(self.data_df.index)

        # Shift timestamp by half-frequency, if needed
        if self.timestamp_start_middle_end == 'middle':
            pass
        else:
            timedelta = pd.to_timedelta(self.data_freq) / 2
            if self.timestamp_start_middle_end == 'end':
                self.data_df.index = self.data_df.index - pd.Timedelta(timedelta)
            elif self.timestamp_start_middle_end == 'start':
                self.data_df.index = self.data_df.index + pd.Timedelta(timedelta)


if __name__ == '__main__':
    pass
