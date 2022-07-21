import datetime as dt
import time
from typing import Literal

import pandas as pd
from pandas import DataFrame, Series


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

    def __init__(self, index: pd.Index, freq_expected: str):
        self.index = index
        self.freq_expected = freq_expected
        self.num_datarows = self.index.__len__()
        self.freq = None
        self._run()

    def _run(self):
        freq_timedelta = self._infer_freq_timedelta()
        freq_full = self._infer_freq_full()
        freq_progressive = self._infer_freq_progressive()
        # List of {Set of detected and expected freqs}:
        freq_list = list({freq_timedelta, freq_full, freq_progressive, self.freq_expected})
        if len(freq_list) == 1:
            # Highest certainty, one single freq found
            self.freq = freq_list[0]
        elif freq_progressive == self.freq_expected:
            self.freq = freq_progressive
        else:
            raise Exception("Frequency detection failed.")

        print(f"Detected frequency from data: {self.freq}")

        # print(freq_timedelta)
        # print(freq_full)
        # print(freq_progressive)
        # self.freq = freq_list[0] if len(freq_list) == 1 else '-9999'

    def get(self) -> str:
        return self.freq

    def _infer_freq_timedelta(self):
        """Check DataFrame index for frequency by subtracting successive timestamps from each other
        and then checking the most frequent difference

        - https://stackoverflow.com/questions/16777570/calculate-time-difference-between-pandas-dataframe-indices
        - https://stackoverflow.com/questions/31469811/convert-pandas-freq-string-to-timedelta
        """
        df = pd.DataFrame(columns=['tvalue'])
        df['tvalue'] = self.index
        df['tvalue_shifted'] = df['tvalue'].shift()
        df['delta'] = (df['tvalue'] - df['tvalue_shifted'])
        most_frequent_delta = df['delta'].mode()[0]  # timedelta
        most_frequent_delta = timedelta_to_string(most_frequent_delta)
        # most_frequent_delta = pd.to_timedelta(most_frequent_delta)

        return most_frequent_delta

    def _infer_freq_full(self):
        """Infer data frequency from full time series index"""
        # Needs at least 3 values
        if self.num_datarows < 3:
            return None
        return pd.infer_freq(self.index)

    def _infer_freq_progressive(self):
        """Try to infer freq from first x and last x rows of data, if these
        match we can be relatively certain that the file has the same freq
        from start to finish.
        """
        # Try to infer freq from first x and last x rows of data, must match
        if self.num_datarows > 0:
            for ndr in range(50, 5, -1):  # ndr = number of data rows
                if self.num_datarows >= ndr * 2:  # Same amount of ndr needed for start and end of file
                    _inferred_freq_start = pd.infer_freq(self.index[0:ndr])
                    _inferred_freq_end = pd.infer_freq(self.index[-ndr:])
                    inferred_freq = _inferred_freq_start if _inferred_freq_start == _inferred_freq_end else None
                    if inferred_freq:
                        freqfrom = f'data {ndr}+{ndr}' if inferred_freq else '-'
                        return inferred_freq
                else:
                    continue


def remove_index_duplicates(series: Series, keep: Literal["first", "last", False] = "last") -> Series:
    """Remove index duplicates"""
    if series.index.duplicated().sum() > 0:
        # Duplicates found
        print("Removing index duplicates ...")
        return series[~series.index.duplicated(keep=keep)]
    else:
        # No duplicates found
        print("No index duplicates found.")
        return series


def continuous_timestamp_freq(data: Series or DataFrame, freq: str) -> Series:
    """Generate continuous timestamp of given frequency between first and last date of index

    This makes df continuous w/o date gaps but w/ data gaps at filled-in timestamps.
    """
    first_date = data.index[0]
    last_date = data.index[-1]

    print(f"Creating continuous {freq} timestamp index between {first_date} and {last_date} ...")

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
    return data

def convert_series_timestamp_to_middle(series: Series) -> Series:
    """Standardize timestamp index column"""

    timestamp_name = series.index.name
    timestamp_freq = series.index.freq

    if timestamp_name == 'TIMESTAMP_MID':
        pass
    else:
        from pandas.tseries.frequencies import to_offset
        to_offset('T')
        timedelta = pd.to_timedelta(timestamp_freq) / 2
        if timestamp_name == 'TIMESTAMP_END':
            series.index = series.index - pd.Timedelta(timedelta)
        elif timestamp_name == 'TIMESTAMP_START':
            series.index = series.index + pd.Timedelta(timedelta)
    series.index.name = 'TIMESTAMP_MID'
    return series

def convert_timestamp_to_middle(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize timestamp index column"""

    timestamp_name = df.index.name
    timestamp_freq = df.index.freqstr

    if timestamp_name == 'TIMESTAMP_MID':
        pass
    else:
        timedelta = pd.to_timedelta(timestamp_freq) / 2
        if timestamp_name == 'TIMESTAMP_END':
            df.index = df.index - pd.Timedelta(timedelta)
        elif timestamp_name == 'TIMESTAMP_START':
            df.index = df.index + pd.Timedelta(timedelta)
    df.index.name = 'TIMESTAMP_MID'
    return df

def sanitize_timestamp_index(data: Series or DataFrame, freq: str) -> Series:
    """Sanitize timestamp index"""

    # Sort index
    data.sort_index(inplace=True)

    # Remove index duplicates
    data = remove_index_duplicates(series=data, keep='last')

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
