import datetime as dt
import fnmatch
import time
from typing import Literal

import pandas as pd
from pandas import DataFrame, Series
from pandas.tseries.frequencies import to_offset


class TimestampSanitizer():

    def __init__(self, data: Series or DataFrame):
        self.data = data.copy()
        self.inferred_freq = None
        self._run()

    def get(self) -> Series or DataFrame:
        return self.data

    def _run(self):
        print("Sanitizing timestamp ...")

        # Validate timestamp name
        _ = validate_timestamp_naming(data=self.data)

        # Convert timestamp to datetime
        self.data = convert_timestamp_to_datetime(self.data)

        # Sort timestamp index ascending
        self.data = sort_timestamp_ascending(self.data)

        # Remove index duplicates
        self.data = remove_index_duplicates(data=self.data, keep='last')

        # Detect time resolution from data
        self.inferred_freq = DetectFrequency(index=self.data.index).get()

        # Make timestamp continuous w/o date gaps
        self.data = continuous_timestamp_freq(data=self.data, freq=self.inferred_freq)

        # Convert timestamp to middle
        self.data = convert_series_timestamp_to_middle(data=self.data)


def sort_timestamp_ascending(data: Series or DataFrame) -> Series or DataFrame:
    print("Sorting timestamp ascending ...", end=" ")
    data.sort_index()
    print("OK")
    return data


def convert_timestamp_to_datetime(data: Series or DataFrame) -> Series or DataFrame:
    """
    Convert timestamp index to datetime format

    This acts as additional check to make sure that the timestamp
    index is in the required datetime format.

    Args:
        data: Data with timestamp index

    Returns:
        data with confirmed datetime index

    """
    print("Converting timestamp to datetime ...", end=" ")
    try:
        data.index = pd.to_datetime(data.index)
        print("OK")
    except:
        raise Exception("Conversion of timestamp to datetime format failed.")
    return data


def validate_timestamp_naming(data: Series or DataFrame) -> str:
    """
    Check if timestamp is correctly named

    This check is done to make sure that the timestamp gives specific
    information if it refers to the start, middle or end of the averaging
    period.

    Args:
        data: Data with timestamp index

    """
    print("Validating timestamp naming ...", end=" ")
    timestamp_name = data.index.name
    allowed_timestamp_names = ['TIMESTAMP_END', 'TIMESTAMP_START', 'TIMESTAMP_MID']
    if any(fnmatch.fnmatch(timestamp_name, allowed_name) for allowed_name in allowed_timestamp_names):
        print("OK")
        return timestamp_name
    else:
        raise Exception(f"Name of timestamp index must be one of the following: {allowed_timestamp_names} "
                        f"('{timestamp_name}' is not allowed)")


def current_unixtime() -> int:
    """
    Current time as integer number of nanoseconds since the epoch

    Notebook example available: https://gitlab.ethz.ch/diive/diive-notebooks
    """
    current_time_unix = time.time_ns()
    return current_time_unix


def current_datetime(str_format: str = '%Y-%m-%d %H:%M:%S') -> tuple[dt.datetime, str]:
    """
    Current datetime as datetime and string

    Notebook example available: https://gitlab.ethz.ch/diive/diive-notebooks
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime(str_format)
    return now_time_dt, now_time_str


def current_datetime_str_condensed() -> str:
    """
    Current datetime as string

    Notebook example available: https://gitlab.ethz.ch/diive/diive-notebooks
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%Y%m%d%H%M%S")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def current_time_microseconds_str() -> str:
    """
    Current time including microseconds as string

    Notebook example available: https://gitlab.ethz.ch/diive/diive-notebooks
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%H%M%S%f")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def make_run_id(prefix: str = False) -> str:
    """
    Create string identifier that includes current datetime

    Notebook example available: https://gitlab.ethz.ch/diive/diive-notebooks
    """
    now_time_dt, _ = current_datetime()
    now_time_str = now_time_dt.strftime("%Y%m%d-%H%M%S")
    prefix = prefix if prefix else "RUN"
    run_id = f"{prefix}-{now_time_str}"
    return run_id


def timedelta_to_string(timedelta):
    """
    Converts a pandas.Timedelta to a string representation
    compatible with pandas.Timedelta constructor format
    https://stackoverflow.com/questions/46429736/pandas-resampling-how-to-generate-offset-rules-string-from-timedelta
    https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    Notebook example available: https://gitlab.ethz.ch/diive/diive-notebooks
    """
    c = timedelta.components
    format = ''
    if c.days != 0:
        format += '%dD' % c.days
    if c.hours > 0:
        format += '%dH' % c.hours
    if c.minutes > 0:
        format += '%dT' % c.minutes
    if c.seconds > 0:
        format += '%dS' % c.seconds
    if c.milliseconds > 0:
        format += '%dL' % c.milliseconds
    if c.microseconds > 0:
        format += '%dU' % c.microseconds
    if c.nanoseconds > 0:
        format += '%dN' % c.nanoseconds

    return format


def generate_freq_timedelta_from_freq(to_duration, to_freq):
    """
    Generate timedelta with given duration and frequency

    Does not really work with M or Y frequency b/c of their different number of days,
    e.g. August 31 days but September has 30 days.

    The Timedelta can be directly used in operations, e.g. when one single timestamp
    entry is available and it is needed to calculate the previous timestamp. With the
    Timedelta, the previous timestamp can be calculated by simply subtracting the
    Timedelta from the available timestamp.

    Example:
        >> to_duration = 1
        >> to_freq = 'D'
        >> pd.to_timedelta(to_duration, unit=to_freq)
        Timedelta('1 days 00:00:00')

    :param to_duration: int
    :param to_freq: pandas frequency string
                    see here for options:
                    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    :return: Timedelta
    """
    timedelta = pd.to_timedelta(to_duration, unit=to_freq)
    return timedelta


def generate_freq_str(to_freq):
    """

    Time / date components in pandas:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-date-components

    :param to_freq:
    :return:
    """
    # Allowed expressions
    minutes = ['Minute(s)']
    hours = ['Hourly', 'Hour(s)']
    days = ['Daily', 'Day(s)']
    weeks = ['Weekly', 'Week(s)']
    months = ['Monthly', 'Month(s)']
    years = ['Yearly', 'Year(s)']

    if to_freq in minutes:
        freq_str = 'T'
    elif to_freq in hours:
        freq_str = 'H'
    elif to_freq in days:
        freq_str = 'D'
    elif to_freq in weeks:
        freq_str = 'W'  # Anchor to Sunday as last day of the week
    elif to_freq in months:
        freq_str = 'M'
    elif to_freq in years:
        freq_str = 'A'
    else:
        freq_str = 'Original'
    return freq_str


def build_timestamp_range(start_dt, df_len, freq):
    """ Builds timestamp column starting with start date and
        the given frequency.

    :param df_len: int (number of rows)
    :param freq: pandas freq string (e.g. '1S' for 1 second steps)
    :return:
    """

    add_timedelta = (df_len - 1) * pd.to_timedelta(freq)
    end_dt = start_dt + pd.Timedelta(add_timedelta)
    date_rng = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    return date_rng


def include_timestamp_as_cols(df,
                              year: bool = True,
                              doy: bool = True,
                              week: bool = True,
                              month: bool = True,
                              hour: bool = True,
                              info: bool = True):
    """Include timestamp info as data columns"""

    df = df.copy()
    newcols = []
    year_col = '.YEAR'
    doy_col = '.DOY'  # '[day_of_year]'
    week_col = '.WEEK'  # '[week_of_year]'
    month_col = '.MONTH'  # '[month]'
    hour_col = '.HOUR'  # '[hour]'

    if year:
        df[year_col] = df.index.year
        newcols.append(doy_col)
    if doy:
        df[doy_col] = df.index.dayofyear
        newcols.append(doy_col)
    if week:
        df[week_col] = df.index.isocalendar().week
        newcols.append(week_col)
    if month:
        df[month_col] = df.index.month
        newcols.append(month_col)
    if hour:
        df[hour_col] = df.index.hour
        newcols.append(hour_col)

    if info:
        print(f"Added timestamp as columns: {newcols}")
    return df


class DetectFrequency:
    """Detect data time resolution from time series index

    """

    def __init__(self, index: pd.DatetimeIndex):
        self.index = index
        # self.freq_expected = freq_expected
        self.num_datarows = self.index.__len__()
        self.freq = None
        self._run()

    def _run(self):
        print("Detecting time resolution from data ...", end=" ")

        freq_full = timestamp_infer_freq_from_fullset(timestamp_ix=self.index)
        freq_timedelta = timestamp_infer_freq_from_timedelta(timestamp_ix=self.index)
        freq_progressive = timestamp_infer_freq_progressively(timestamp_ix=self.index)

        if all(f for f in [freq_full, freq_timedelta, freq_progressive]):

            # List of {Set of detected freqs}
            freq_list = list({freq_timedelta, freq_full, freq_progressive})

            if len(freq_list) == 1:
                # Maximum certainty, one single freq found across all checks
                self.freq = freq_list[0]
                print(f"OK (detected {self.freq} time resolution with maximum confidence)")

        elif freq_full:
            # High certainty, freq found from full range of dataset
            self.freq = freq_full
            print(f"OK (detected {self.freq} time resolution {self.freq} with high confidence)")

        elif freq_progressive:
            # Medium certainty, freq found from start and end of dataset
            self.freq = freq_progressive
            print(f"OK (detected {self.freq} time resolution {self.freq} with medium confidence)")

        elif freq_timedelta:
            # High certainty, freq found from most frequent timestep that
            # occurred at least 99% of the time
            self.freq = freq_timedelta
            print(f"OK (detected {self.freq} time resolution {self.freq} high confidence)")

        else:
            raise Exception("Frequency detection failed.")

    def get(self) -> str:
        return self.freq


def timestamp_infer_freq_progressively(timestamp_ix: pd.DatetimeIndex) -> str or None:
    """Try to infer freq from first x and last x rows of data, if these
    match we can be relatively certain that the file has the same freq
    from start to finish.
    """
    # Try to infer freq, starting from first 1000 and last 1000 rows of data, must match
    n_datarows = timestamp_ix.__len__()
    inferred_freq = None
    checkrange = 1000
    if n_datarows > 0:
        for ndr in range(checkrange, 5, -1):  # ndr = number of data rows
            if n_datarows >= ndr * 2:  # Same amount of ndr needed for start and end of file
                _inferred_freq_start = pd.infer_freq(timestamp_ix[0:ndr])
                _inferred_freq_end = pd.infer_freq(timestamp_ix[-ndr:])
                inferred_freq = _inferred_freq_start if _inferred_freq_start == _inferred_freq_end else None
                if inferred_freq:
                    freqfrom = f'data {ndr}+{ndr}' if inferred_freq else '-'
                    return inferred_freq
            else:
                continue
    return inferred_freq


def timestamp_infer_freq_from_fullset(timestamp_ix: pd.DatetimeIndex) -> str or None:
    """
    Infer data frequency from all timestamps in time series index

    Minimum 10 values are required in timeseries index.

    Args:
        timestamp_ix: Timestamp index

    Returns:
        Frequency string, e.g. '10T' for 10-minute time resolution
    """
    n_datarows = timestamp_ix.__len__()
    if n_datarows < 10:
        return None
    inferred_freq = pd.infer_freq(timestamp_ix)
    if inferred_freq:
        return inferred_freq
    else:
        return None


def timestamp_infer_freq_from_timedelta(timestamp_ix: pd.DatetimeIndex) -> str or None:
    """Check DataFrame index for frequency by subtracting successive timestamps from each other
    and then checking the most frequent difference

    - https://stackoverflow.com/questions/16777570/calculate-time-difference-between-pandas-dataframe-indices
    - https://stackoverflow.com/questions/31469811/convert-pandas-freq-string-to-timedelta
    """
    df = pd.DataFrame(columns=['tvalue'])
    df['tvalue'] = timestamp_ix
    df['tvalue_shifted'] = df['tvalue'].shift()
    df['delta'] = (df['tvalue'] - df['tvalue_shifted'])
    n_rows = df['delta'].size  # Total length of data
    detected_deltas = df['delta'].value_counts()  # Found unique deltas
    most_frequent_delta = df['delta'].mode()[0]  # Delta with most occurrences
    most_frequent_delta_counts = detected_deltas[
        most_frequent_delta]  # Number of occurrences for most frequent delta
    most_frequent_delta_perc = most_frequent_delta_counts / n_rows  # Fraction
    # Check whether the most frequent delta appears in >99% of all data rows
    if most_frequent_delta_perc > 0.99:
        inferred_freq = timedelta_to_string(most_frequent_delta)
        # most_frequent_delta = pd.to_timedelta(most_frequent_delta)
        return inferred_freq
    else:
        return None


def remove_index_duplicates(data: Series or DataFrame,
                            keep: Literal["first", "last", False] = "last") -> Series or DataFrame:
    """Remove index duplicates"""
    print("Removing index duplicates ...", end=" ")
    n_duplicates = data.index.duplicated().sum()
    if n_duplicates > 0:
        # Duplicates found
        data = data[~data.index.duplicated(keep=keep)]
        print(f"OK (removed {n_duplicates} rows with duplicate timestamps)")
        return data
    else:
        # No duplicates found
        print(f"OK (no duplicates found in timestamp index)")
        return data


def continuous_timestamp_freq(data: Series or DataFrame, freq: str) -> Series or DataFrame:
    """Generate continuous timestamp of given frequency between first and last date of index

    This makes df continuous w/o date gaps but w/ data gaps at filled-in timestamps.
    """
    first_date = data.index[0]
    last_date = data.index[-1]

    print(f"Creating continuous {freq} timestamp index between {first_date} and {last_date} ...", end=" ")

    # Original timestamp name
    idx_name = data.index.name

    # Generate timestamp index b/w first and last date
    _index = pd.date_range(start=first_date, end=last_date, freq=freq)

    data = data.reindex(_index)
    data.index.name = idx_name

    # Set freq
    data.index = pd.to_datetime(data.index)
    data = data.asfreq(freq=freq)
    # df.sort_index(inplace=True)
    print("OK")
    return data


def convert_series_timestamp_to_middle(data: Series or DataFrame) -> Series or DataFrame:
    """Standardize timestamp index column"""

    timestamp_name_before = data.index.name
    timestamp_name_after = 'TIMESTAMP_MID'
    timestamp_freq = data.index.freq

    print(f"Converting timestamp index {timestamp_name_before} to show middle of averaging period ...")

    first_timestamp_before = data.index[0]
    last_timestamp_before = data.index[-1]

    if timestamp_name_before == 'TIMESTAMP_MID':
        pass
    else:
        to_offset('T')
        timedelta = pd.to_timedelta(timestamp_freq) / 2
        if timestamp_name_before == 'TIMESTAMP_END':
            data.index = data.index - pd.Timedelta(timedelta)
        elif timestamp_name_before == 'TIMESTAMP_START':
            data.index = data.index + pd.Timedelta(timedelta)
        else:
            raise Exception("Timestamp index of the Series must be "
                            "named 'TIMESTAMP_END', 'TIMESTAMP_START' or 'TIMESTAMP_MID'.")

    data.index.name = 'TIMESTAMP_MID'
    first_timestamp_after = data.index[0]
    last_timestamp_after = data.index[-1]

    print(f"    {timestamp_name_before} was converted to {timestamp_name_after}")
    print(f"    First and last dates:")
    print(f"        Before conversion: "
          f"{timestamp_name_before} from {first_timestamp_before} to {last_timestamp_before}")
    print(f"         After conversion: "
          f"{timestamp_name_after} from {first_timestamp_after} to {last_timestamp_after}")

    return data


def sanitize_timestamp_index(data: Series or DataFrame, freq: str) -> Series:
    """Sanitize timestamp index"""

    # Sort index
    data.sort_index(inplace=True)

    # Remove index duplicates
    data = remove_index_duplicates(data=data, keep='last')

    # Detect time resolution from data
    freq_validated = DetectFrequency(index=data.index, freq_expected=freq).get()

    # Make timestamp continuous w/o date gaps
    data = continuous_timestamp_freq(data=data, freq=freq_validated)

    return data

    # # Timestamp convention
    # # Shift timestamp by half-frequency, if needed
    # if self.timestamp_start_middle_end == 'middle':
    #     pass
    # else:
    #     timedelta = pd.to_timedelta(self.data_freq) / 2
    #     if self.timestamp_start_middle_end == 'end':
    #         self.data_df.index = self.data_df.index - pd.Timedelta(timedelta)
    #     elif self.timestamp_start_middle_end == 'start':
    #         self.data_df.index = self.data_df.index + pd.Timedelta(timedelta)


def add_timezone_info(timestamp_index, timezone_of_timestamp: str):
    """Add timezone info to timestamp index

    No data are changed, only the timezone info is added to the timestamp.

    :param: timezone_of_timestamp: If 'None', no timezone info is added. Otherwise
        can be `str` that describes the timezone in relation to UTC in the format:
        'UTC+01:00' (for CET), 'UTC+02:00' (for CEST), ...
        InfluxDB uses this info to upload data (always) in UTC/GMT.

    see: https://www.atmos.albany.edu/facstaff/ktyle/atm533/core/week5/04_Pandas_DateTime.html#note-that-the-timezone-is-missing-the-read-csv-method-does-not-provide-a-means-to-specify-the-timezone-we-can-take-care-of-that-though-with-the-tz-localize-method

    """
    return timestamp_index.tz_localize(timezone_of_timestamp)  # v0.3.1


if __name__ == '__main__':
    pass

    # # Test code
    # filepath = r'L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\tests\testdata\testfile_ch-dav_2020.diive.csv'
    # df = pd.read_csv(filepath, index_col=0, parse_dates=True, skiprows=[1])
    #
    # # Remove index duplicates
    # df = remove_index_duplicates(df=df, keep='last')
    #
    # # Detect time resolution from data
    # freq = DetectFrequency(index=df.index, freq_expected='30T').get()
    #
    # df = continuous_timestamp_freq(df=df, freq=freq)
    #
    # df = convert_timestamp_to_middle(df=df)
    #
    # from diive.core.times.resampling import resample_df_T_H
    # resample_df_T_H(df=df, to_freqstr='2H', agg='mean', mincounts_perc=.9)
    #
    #
    # print(freq)
