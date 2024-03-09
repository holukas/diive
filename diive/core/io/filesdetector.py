import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)


class FilesDetector:

    def __init__(self,
                 filelist: list,
                 file_date_format: str,
                 file_generation_res: str,
                 data_res: float,
                 files_how_many: int = None):
        """Create overview dataframe of available and missing (expected) files.

        Args:
            filelist: List of found files
            file_pattern: Pattern to identify files
            file_date_format: Datetime information contained in file name
            file_generation_res: Regular interval at which files were created, e.g. '6h' for every 6 hours
            data_res: Interval in seconds at which data are logged, e.g. 0.05
            files_how_many:
        """

        # self.dir_input = indir
        self.filelist = filelist
        self.file_date_format = file_date_format
        self.file_generation_res = file_generation_res
        self.data_res = data_res
        self.files_how_many = files_how_many

        # Check if there are files listed in filelist
        if not self.filelist:
            print(f"\n(!)ERROR *filelist* must not be empty. Stopping script.")
            sys.exit()

        self._files_overview_df = DataFrame()

    @property
    def files_overview_df(self) -> DataFrame:
        """Return flag as Series"""
        if not isinstance(self._files_overview_df, DataFrame):
            raise Exception(f'No results available.')
        return self._files_overview_df

    def get_results(self) -> DataFrame:
        return self.files_overview_df

    def run(self):
        """Execute full processing stack."""
        self._files_overview_df = self.add_expected()
        self._files_overview_df = self.add_unexpected()
        self._files_overview_df = self.calc_expected_values()
        self._files_overview_df.loc[:, 'file_available'] = self.files_overview_df.loc[:, 'file_available'].fillna(0,
                                                                                                                  inplace=False)
        self._files_overview_df = self.restrict_numfiles()

    def restrict_numfiles(self):
        # Consider file limit
        _files_overview_df = self.files_overview_df.copy()

        if self.files_how_many:
            for idx, file in _files_overview_df.iterrows():
                _restricted_df = _files_overview_df.loc[_files_overview_df.index[0]:idx]
                num_available_files = _restricted_df['file_available'].sum()
                if num_available_files >= self.files_how_many:
                    _files_overview_df = _restricted_df.copy()
                    break

        return _files_overview_df

    def add_expected(self):
        """Create index of expected files (regular start time) and check
        which of these regular files are available.

        :return: DataFrame with info about regular (expected) files
        :rtype: pandas DataFrame
        """
        first_file_dt = dt.datetime.strptime(self.filelist[0].name, self.file_date_format)
        last_file_dt = dt.datetime.strptime(self.filelist[-1].name, self.file_date_format)
        expected_end_dt = last_file_dt + pd.Timedelta(self.file_generation_res)
        expected_index_dt = pd.date_range(first_file_dt, expected_end_dt, freq=self.file_generation_res)
        files_df = pd.DataFrame(index=expected_index_dt)

        for file_idx, filepath in enumerate(self.filelist):
            filename = filepath.name
            file_start_dt = dt.datetime.strptime(filename, self.file_date_format)

            if file_start_dt in files_df.index:
                files_df.loc[file_start_dt, 'file_available'] = 1
                files_df.loc[file_start_dt, 'filename'] = filename
                files_df.loc[file_start_dt, 'start'] = file_start_dt
                files_df.loc[file_start_dt, 'filepath'] = filepath
                files_df.loc[file_start_dt, 'filesize'] = Path(filepath).stat().st_size
                # files_df.loc[file_start_dt, 'expected_file'] = file_start_dt

        files_df.insert(0, 'expected_file', files_df.index)  # inplace
        return files_df

    def add_unexpected(self):
        """Add info about unexpected files (irregular start time).

        :return: DataFrame with added info about irregular files
        :rtype: pandas DataFrame
        """
        files_df = self.files_overview_df.copy()
        for file_idx, filepath in enumerate(self.filelist):
            filename = filepath.name
            file_start_dt = dt.datetime.strptime(filename, self.file_date_format)

            if file_start_dt not in files_df.index:
                files_df.loc[file_start_dt, 'file_available'] = 1
                files_df.loc[file_start_dt, 'filename'] = filename
                files_df.loc[file_start_dt, 'start'] = file_start_dt
                files_df.loc[file_start_dt, 'filepath'] = filepath
                files_df.loc[file_start_dt, 'filesize'] = Path(filepath).stat().st_size

        files_df = files_df.sort_index(inplace=False)
        return files_df

    def calc_expected_values(self):
        """Calculate expected end time, duration and records of files

        :return: DataFrame with added info about expected values
        """
        files_df = self.files_overview_df.copy()
        files_df['expected_end'] = files_df.index
        files_df['expected_end'] = files_df['expected_end'].shift(-1)
        files_df['expected_duration'] = (files_df['expected_end'] - files_df['start']).dt.total_seconds()
        files_df['expected_records'] = files_df['expected_duration'] / self.data_res
        # files_df['expected_end'] = files_df['start'] + pd.Timedelta(file_generation_res)
        # files_df.loc[files_df['file_available'] == 1, 'next_file'] = files_df['expected_file']
        # files_df['next_file'] = files_df['next_file'].shift(-1)
        return files_df


def read_segments_file(filepath):
    """
    Read file.

    Is used for reading segment covariances and lag search
    results for each segment. Can be used for all text files
    for which the .read_csv args are valid.

    Parameters
    ----------
    filepath: str

    Returns
    -------
    pandas DataFrame

    """
    # parse = lambda x: dt.datetime.strptime(x, '%Y%m%d%H%M%S')
    import time
    start_time = time.time()
    found_lags_df = pd.read_csv(filepath,
                                skiprows=None,
                                header=0,
                                # names=header_cols_list,
                                # na_values=-9999,
                                encoding='utf-8',
                                delimiter=',',
                                mangle_dupe_cols=True,
                                # keep_date_col=False,
                                parse_dates=False,
                                # date_parser=parse,
                                index_col=0,
                                dtype=None,
                                engine='c')
    # print(f"Read file {filepath} in {time.time() - start_time}s")
    return found_lags_df


def read_raw_data(filepath, nrows):
    header_rows_list = [0]
    skip_rows_list = []
    header_section_rows = [0]

    num_data_cols = \
        length_data_cols(filepath=filepath,
                         header_rows_list=header_rows_list,
                         skip_rows_list=skip_rows_list)

    num_header_cols, header_cols_df = \
        length_header_cols(filepath=filepath,
                           header_rows_list=header_rows_list,
                           skip_rows_list=skip_rows_list)

    more_data_cols_than_header_cols, num_missing_header_cols = \
        data_vs_header(num_data_cols=num_data_cols,
                       num_header_cols=num_header_cols)

    header_cols_list = \
        generate_missing_cols(header_cols_df=header_cols_df,
                              more_data_cols_than_header_cols=more_data_cols_than_header_cols,
                              num_missing_header_cols=num_missing_header_cols)

    start_time = time.time()
    data_df = pd.read_csv(filepath,
                          skiprows=header_section_rows,
                          header=None,
                          names=header_cols_list,
                          na_values=-9999,
                          encoding='utf-8',
                          delimiter=',',
                          mangle_dupe_cols=True,
                          keep_date_col=False,
                          parse_dates=False,
                          date_parser=None,
                          index_col=None,
                          dtype=None,
                          engine='c',
                          nrows=nrows)
    print(f"    Finished reading file '{filepath.name}' in {time.time() - start_time:.3f}s")

    return data_df


def insert_datetime_index(df, file_info_row, data_nominal_res):
    """Insert true timestamp based on number of records in the file and the
    file duration.

    Files measured at a given time resolution may still produce
    more or less than the expected number of records.

    For example, a six-hour file with data recorded at 20Hz is expected to have
    432 000 records, but may in reality produce slightly more or less than that
    due to small inaccuracies in the measurements instrument's internal clock.
    This in turn would mean that the defined time resolution of 20Hz is not
    completely accurate with the true frequency being slightly higher or lower.

    This causes a (minor) issue when merging mutliple data files due to overlapping
    record timestamps, i.e. the last timestamp in file #1 is the same as the first
    timestamp in file #2, resulting in duplicate entries in the timestamp index column
    during merging of files #1 and #2.

    In addition, sometimes more than one timestamp can overlap, resulting in more
    overlapping timestamps and therefore more data loss. Although this data loss is
    minor (e.g. 3 records per 432 000 records), missing records are not desirable when
    calculating covariances between times series. The time series must be as complete
    and without missing records as possible to avoid errors.
    """
    num_records = len(df)
    ratio = num_records / file_info_row['expected_records']
    if (ratio > 0.999) and (ratio < 1.001):
        file_complete = True
        true_resolution = np.float64(file_info_row['expected_duration'] / num_records)
    else:
        file_complete = False
        true_resolution = data_nominal_res

    df['sec'] = df.index * true_resolution
    df['file_start_dt'] = file_info_row['start']
    df['TIMESTAMP'] = pd.to_datetime(df['file_start_dt'], unit='us') \
                      + pd.to_timedelta(df['sec'], unit='s')
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64[us]')  # Reduce timestamp precision to microseconds
    df.drop(['sec', 'file_start_dt'], axis=1, inplace=True)
    df.set_index('TIMESTAMP', inplace=True)

    # pd.to_datetime(df.index, unit='us')

    return df, true_resolution


def add_data_stats(df, true_resolution, filename, files_overview_df, found_records):
    # Detect overall frequency
    data_duration = found_records * true_resolution
    data_freq = np.float64(found_records / data_duration)

    files_overview_df.loc[filename, 'first_record'] = df.index[0]
    files_overview_df.loc[filename, 'last_record'] = df.index[-1]
    files_overview_df.loc[filename, 'file_duration'] = (df.index[-1] - df.index[0]).total_seconds()
    files_overview_df.loc[filename, 'found_records'] = found_records
    files_overview_df.loc[filename, 'data_freq'] = data_freq

    return files_overview_df


def generate_missing_cols(header_cols_df, more_data_cols_than_header_cols, num_missing_header_cols):
    # Generate missing header columns if necessary
    header_cols_list = header_cols_df.columns.to_list()
    generated_missing_header_cols_list = []
    if more_data_cols_than_header_cols:
        for m in list(range(1, num_missing_header_cols + 1)):
            missing_col = (f'unknown_{m}')
            generated_missing_header_cols_list.append(missing_col)
            header_cols_list.append(missing_col)
    return header_cols_list


def detect_dates():
    pass


def length_data_cols(filepath, header_rows_list, skip_rows_list):
    # Check number of columns of the first data row after the header part
    skip_num_lines = len(header_rows_list) + len(skip_rows_list)
    first_data_row_df = pd.read_csv(filepath,
                                    skiprows=skip_num_lines,
                                    header=None,
                                    nrows=1)
    return first_data_row_df.columns.size


def length_header_cols(filepath, header_rows_list, skip_rows_list):
    # Check number of columns of the header part
    header_cols_df = pd.read_csv(filepath,
                                 skiprows=skip_rows_list,
                                 header=header_rows_list,
                                 nrows=0)
    return header_cols_df.columns.size, header_cols_df


def data_vs_header(num_data_cols, num_header_cols):
    # Check if there are more data columns than header columns
    if num_data_cols > num_header_cols:
        more_data_cols_than_header_cols = True
        num_missing_header_cols = num_data_cols - num_header_cols
    else:
        more_data_cols_than_header_cols = False
        num_missing_header_cols = 0
    return more_data_cols_than_header_cols, num_missing_header_cols


def setup_output_dirs(outdir='output', del_previous_results=False):
    """Make output directories."""
    new_dirs = ['stats', 'splits']
    outdirs = {}

    # Store keys and full paths in dict
    for nd in new_dirs:
        # outdirs[nd] = Path(outdir)
        outdirs[nd] = outdir / Path(nd)

    # Make dirs
    for key, path in outdirs.items():
        if not Path.is_dir(path):
            print(f"Creating folder {path} ...")
            os.makedirs(path)
        else:
            if del_previous_results:
                for filename in os.listdir(path):
                    filepath = os.path.join(path, filename)
                    try:
                        if os.path.isfile(filepath) or os.path.islink(filepath):
                            print(f"Deleting file {filepath} ...")
                            os.unlink(filepath)
                        # elif os.path.isdir(filepath):
                        #     shutil.rmtree(filepath)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (filepath, e))

    return outdirs


def example():
    SEARCHDIRS = [r'F:\Sync\luhk_work\20 - CODING\27 - VARIOUS\dyco\_testdata']
    PATTERN = 'CH-DAS_*.csv.gz'

    from diive.core.io.filereader import search_files
    foundfiles = search_files(searchdirs=SEARCHDIRS, pattern=PATTERN)

    fd = FilesDetector(filelist=foundfiles,
                       file_date_format='CH-DAS_%Y%m%d%H%M.csv.gz',
                       file_generation_res='6h',
                       data_res=0.05,
                       files_how_many=3)

    fd.run()

    files_overview_df = fd.get_results()

    print(files_overview_df)


if __name__ == "__main__":
    example()
