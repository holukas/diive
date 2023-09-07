import datetime as dt
import fnmatch
import time
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, DatetimeIndex
from pandas.tseries.frequencies import to_offset


def detect_freq_groups(index: DatetimeIndex) -> Series:
    """
    Analyze timestamp for records where the time resolution is absolutely certain

    This function calculates the timedeltas (the time differences) between the current
    timestamp and the timestamp of the record before and after. For data records where
    the two differences are the same (in absolute terms) have an absolutely certain
    timestamp.

    The determined time resolution of each record is described in the newly created column
    'FREQ_AUTO_SEC' in terms of seconds. The column is added to *df* and can be used to
    access the different time resolution groups separately during later processing.

    Example:

                        TIMESTAMP_CURRENT   TIMESTAMP_PREV      TIMESTAMP_NEXT           DELTA_PREV  DELTA_NEXT  DELTA_DIFF
    TIMESTAMP_CURRENT
    2020-10-01 00:20:00 2020-10-01 00:20:00 2020-10-01 00:10:00 2020-10-01 00:30:00      -600.0       600.0         0.0
    2020-10-01 00:30:00 2020-10-01 00:30:00 2020-10-01 00:20:00 2020-10-01 00:40:00      -600.0       600.0         0.0
    2020-10-01 00:40:00 2020-10-01 00:40:00 2020-10-01 00:30:00 2020-10-01 00:50:00      -600.0       600.0         0.0
    ...                                 ...                 ...                 ...         ...         ...         ...
    2021-09-30 23:57:00 2021-09-30 23:57:00 2021-09-30 23:56:00 2021-09-30 23:58:00       -60.0        60.0         0.0
    2021-09-30 23:58:00 2021-09-30 23:58:00 2021-09-30 23:57:00 2021-09-30 23:59:00       -60.0        60.0         0.0
    2021-09-30 23:59:00 2021-09-30 23:59:00 2021-09-30 23:58:00 2021-10-01 00:00:00       -60.0        60.0         0.0

    The example shows a dataset that starts with 10MIN time resolution and ends with
    1MIN time resolution. For each record, the previous and next timestamp are detected
    and the delta between these two and the current timestamp are calculated. The sum
    of DELTA_PREV and DELTA_NEXT will yield DELTA_DIFF = 0 if the time differences
    were the same. DELTA_DIFF = 0 therefore describes locations where the time resolution
    is certain.

    For 10MIN time records, the 'FREQ_AUTO_SEC' will be set to '600', for 1MIN records
    to '60'. The first and last records of each frequency group are added during processing,
    e.g., the timestamp '2020-10-01 00:10:00' is the first timestamp for the '600' group
    although it is not part of the main index (main index starts one record later with
    '2020-10-01 00:20:00' because '2020-10-01 00:10:00' does not have a value for
    TIMESTAMP_PREV).

    Note:
        Sometimes there are transition periods between one time resolution and another.
        For example, when the time resolution changes from 10MIN to 1MIN, there might be
        several records in between that have neither 10MIN nor 1MIN, but e.g. 7S or 29MIN etc.
        For these transitional records, the time resolution is not clear and therefore they
        are discarded. This typically affects only a handful of records during the transition
        period(s).

    Args:
        index: Time series dataframe

    Returns:
        df: Time series dataframe with the new column 'FREQ_AUTO_SEC' added

    :: Added in v0.43.0
    """

    groups_ser = pd.Series(index=index, data=np.nan, name='FREQ_AUTO_SEC')
    # index['FREQ_AUTO_SEC'] = np.nan

    # Analyse data for different time resolutions
    timedeltas_df = pd.DataFrame()
    timedeltas_df['TIMESTAMP_CURRENT'] = index

    # Add previous and next timestamps
    timedeltas_df['TIMESTAMP_PREV'] = timedeltas_df['TIMESTAMP_CURRENT'].shift(1)
    timedeltas_df['TIMESTAMP_NEXT'] = timedeltas_df['TIMESTAMP_CURRENT'].shift(-1)

    # DELTA is the difference between the current and the previous/next timestamp,
    # expressed as total seconds
    timedeltas_df['DELTA_PREV'] = timedeltas_df['TIMESTAMP_PREV'].sub(timedeltas_df['TIMESTAMP_CURRENT'])
    timedeltas_df['DELTA_NEXT'] = timedeltas_df['TIMESTAMP_NEXT'].sub(timedeltas_df['TIMESTAMP_CURRENT'])
    timedeltas_df['DELTA_PREV'] = timedeltas_df['DELTA_PREV'].dt.total_seconds()
    timedeltas_df['DELTA_NEXT'] = timedeltas_df['DELTA_NEXT'].dt.total_seconds()

    # The sum of DELTA_PREV and DELTA_NEXT can identify data records where
    # the time resolution is unambiguous.
    # For example: DELTA_PREV = -60, DELTA_NEXT = +60, DELTA_DIFF = 0
    #   In this case the time differences of the current timestamp to
    #   the previous and next timestamps are the same (in absolute terms)
    #   and therefore yields the sum zero.
    timedeltas_df['DELTA_DIFF'] = timedeltas_df['DELTA_PREV'] + timedeltas_df['DELTA_NEXT']
    ix = timedeltas_df['DELTA_DIFF'] == 0
    timedelta_unambiguous_df = timedeltas_df.loc[ix].copy()
    timedelta_unambiguous_df.set_index(timedelta_unambiguous_df['TIMESTAMP_CURRENT'], inplace=True)

    # Count occurrences of respective DELTA
    delta_counts_df = timedelta_unambiguous_df['DELTA_NEXT'].groupby(
        timedelta_unambiguous_df['DELTA_NEXT']).count().sort_values(ascending=False)
    delta_counts_df = pd.DataFrame(delta_counts_df)
    delta_counts_df.rename(columns={"DELTA_NEXT": "COUNTS"}, inplace=True)

    # Calculate how much time is covered by each DELTA
    delta_counts_df['DELTA_NEXT'] = delta_counts_df.index
    delta_counts_df['DELTA_TOTAL_TIME'] = delta_counts_df['DELTA_NEXT'].multiply(delta_counts_df['COUNTS'])
    delta_counts_df['TOTAL_TIME'] = delta_counts_df['DELTA_TOTAL_TIME'].sum()
    delta_counts_df['%_DELTA_TOTAL_TIME'] = delta_counts_df['DELTA_TOTAL_TIME'] / delta_counts_df['TOTAL_TIME']
    delta_counts_df['%_DELTA_TOTAL_TIME'] = delta_counts_df['%_DELTA_TOTAL_TIME'] * 100

    # List of found time resolutions (unambiguous)
    deltas = delta_counts_df['DELTA_NEXT'].to_list()

    # Detect first and last date for each delta
    # First and last dates need to be included by using:
    #   - 'TIMESTAMP_PREV' for first date
    #   - 'TIMESTAMP_NEXT' for last date
    for d in deltas:
        this_delta = timedelta_unambiguous_df.loc[timedelta_unambiguous_df['DELTA_NEXT'] == d].copy()
        this_delta.set_index(this_delta['TIMESTAMP_CURRENT'], inplace=True)
        first_date = this_delta['TIMESTAMP_PREV'].min()
        last_date = this_delta['TIMESTAMP_NEXT'].max()

        # Add first and last date to df
        new_index = this_delta.index.union([first_date, last_date])
        this_delta = this_delta.reindex(new_index)

        groups_ser.loc[this_delta.index] = d

        # freq = f"{int(d)}S"
        # _index = pd.date_range(start=first_date, end=last_date, freq=freq)
        # this_delta.reindex(_index)
        delta_counts_df.loc[d, 'FIRST_DATE'] = first_date
        delta_counts_df.loc[d, 'LAST_DATE'] = last_date

    return groups_ser


class TimestampSanitizer:

    def __init__(self,
                 data: Series or DataFrame,
                 output_middle_timestamp: bool = True,
                 verbose: bool = False):
        """
        Validate and prepare timestamps for further processing

        Performs various checks on timestamps, in this order:
        - Validate timestamp naming
        - Convert timestamp to datetime
        - Sort timestamp ascending
        - Remove duplicates from timestamp index
        - Detect time resolution from timestamp
        - Make timestamp continuous without date gaps
        - Convert timestamp to show the middle of the averaging period (optional)

        The `TimestampSanitizer` class acts as a wrapper to combine various
        timestamp functions.

        For more info please refer to the docstring of the respective function.

        Args:
            data: Data with timestamp index
            output_middle_timestamp:
        """
        self.data = data.copy()
        self.output_middle_timestamp = output_middle_timestamp
        self.verbose = verbose

        self.inferred_freq = None if not data.index.freq else data.index.freq

        self._run()

    def get(self) -> Series or DataFrame:
        return self.data

    def _run(self):
        if self.verbose: print("Sanitizing timestamp ...")

        # Validate timestamp name
        _ = validate_timestamp_naming(data=self.data, verbose=self.verbose)

        # Convert timestamp to datetime
        self.data = convert_timestamp_to_datetime(self.data, verbose=self.verbose)

        # Sort timestamp index ascending
        self.data = sort_timestamp_ascending(self.data, verbose=self.verbose)

        # Remove index duplicates
        self.data = remove_index_duplicates(data=self.data, keep='last', verbose=self.verbose)

        # Detect time resolution from data
        if not self.inferred_freq:
            self.inferred_freq = DetectFrequency(index=self.data.index, verbose=self.verbose).get()

        # Make timestamp continuous w/o date gaps
        self.data = continuous_timestamp_freq(data=self.data, freq=self.inferred_freq, verbose=self.verbose)

        # Convert timestamp to middle
        if self.output_middle_timestamp:
            self.data = convert_series_timestamp_to_middle(data=self.data, verbose=self.verbose)


def sort_timestamp_ascending(data: Series or DataFrame, verbose: bool = False) -> Series or DataFrame:
    if verbose: print(f"Sorting timestamp {data.index.name} ascending ...", end=" ")
    data.sort_index()
    return data


def convert_timestamp_to_datetime(data: Series or DataFrame, verbose: bool = False) -> Series or DataFrame:
    """
    Convert timestamp index to datetime format

    This acts as additional check to make sure that the timestamp
    index is in the required datetime format.

    Args:
        data: Data with timestamp index

    Returns:
        data with confirmed datetime index

    """
    if verbose: print(f"Converting timestamp {data.index.name} to datetime ...", end=" ")
    try:
        data.index = pd.to_datetime(data.index)
        if verbose: print("OK")
    except:
        raise Exception("Conversion of timestamp to datetime format failed.")
    return data


def validate_timestamp_naming(data: Series or DataFrame, verbose: bool = False) -> str:
    """
    Check if timestamp is correctly named

    This check is done to make sure that the timestamp gives specific
    information if it refers to the start, middle or end of the averaging
    period.

    Args:
        data: Data with timestamp index

    """
    timestamp_name = data.index.name
    allowed_timestamp_names = ['TIMESTAMP_END', 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE']
    if verbose: print(f"Validating timestamp naming of timestamp column {timestamp_name} ...", end=" ")

    # First check if timestamp already has one of the required names
    if any(fnmatch.fnmatch(timestamp_name, allowed_name) for allowed_name in allowed_timestamp_names):
        if verbose: print("Timestamp name OK.")
        return timestamp_name

    else:
        raise Exception(f"Name of timestamp index must be one of the following: {allowed_timestamp_names} "
                        f"('{timestamp_name}' is not allowed)")


def current_unixtime() -> int:
    """
    Current time as integer number of nanoseconds since the epoch

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    current_time_unix = time.time_ns()
    return current_time_unix


def current_datetime(str_format: str = '%Y-%m-%d %H:%M:%S') -> tuple[dt.datetime, str]:
    """
    Current datetime as datetime and string

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime(str_format)
    return now_time_dt, now_time_str


def current_date_str_condensed() -> str:
    """
    Current date as string

    - Example notebook available in:
        -
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%Y%m%d")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def current_datetime_str_condensed() -> str:
    """
    Current datetime as string

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%Y%m%d%H%M%S")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def current_time_microseconds_str() -> str:
    """
    Current time including microseconds as string

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%H%M%S%f")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def make_run_id(prefix: str = False) -> str:
    """
    Create string identifier that includes current datetime

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
    """
    now_time_dt, _ = current_datetime()
    now_time_str = now_time_dt.strftime("%Y%m%d-%H%M%S")
    prefix = prefix if prefix else "RUN"
    run_id = f"{prefix}-{now_time_str}"
    return run_id


def timedelta_to_string(timedelta):
    """
    Converts a pandas.Timedelta to a frequency string representation
    compatible with pandas.Timedelta constructor format
    https://stackoverflow.com/questions/46429736/pandas-resampling-how-to-generate-offset-rules-string-from-timedelta
    https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    - Example notebook available in:
        notebooks/TimeFunctions/times.ipynb
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

    # Remove leading `1` to represent e.g. daily resolution
    # This is in line with how pandas handles frequency strings,
    # e.g., 1-minute time resolution is represented by `T` and
    # not by `1T`.
    if format == '1D':
        format = 'D'
    elif format == '1H':
        format = 'H'
    elif format == '1T':
        format = 'T'
    elif format == '1S':
        format = 'S'
    elif format == '1L':
        format = 'L'
    elif format == '1U':
        format = 'U'
    elif format == '1N':
        format = 'N'

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
                              season: bool = True,
                              month: bool = True,
                              week: bool = True,
                              doy: bool = True,
                              hour: bool = True,
                              txt: str = "",
                              verbose: int = 1) -> DataFrame:
    """
    Include timestamp info as data columns

    Kudos:
    - https://datascience.stackexchange.com/questions/60951/is-it-necessary-to-convert-labels-in-string-to-integer-for-scikit-learn-and-xgbo
    - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    """

    df = df.copy()
    newcols = []

    if year:
        year_col = '.YEAR'
        newcols.append(year_col)
        df[year_col] = df.index.year.astype(int)

    if season:
        season_col = '.SEASON'
        newcols.append(season_col)
        df[season_col] = insert_season(timestamp=df.index)

    if month:
        month_col = '.MONTH'
        newcols.append(month_col)
        df[month_col] = df.index.month.astype(int)

    if week:
        week_col = '.WEEK'
        newcols.append(week_col)
        df[week_col] = df.index.isocalendar().week.astype(int)

    if doy:
        doy_col = '.DOY'
        newcols.append(doy_col)
        df[doy_col] = df.index.dayofyear.astype(int)

    if hour:
        hour_col = '.HOUR'
        newcols.append(hour_col)
        df[hour_col] = df.index.hour.astype(int)


    # yeardoy_col = '.YEARDOY'
    # yearweek_col = '.YEARWEEK'
    # yearmonthweekdoy_col = '.YEARMONTHWEEKDOY'
    # yearmonth_col = '.YEARMONTH'
    # yearmonthweek_col = '.YEARMONTHWEEK'
    # weekhour_col = '.WEEKHOUR'

    # Combined variables
    # Year and month: YEAR2023+MONTH8 = 20238
    # df[yearmonth_col] = (df[year_col].astype(str) + df[month_col].astype(str)).astype(int)

    # Year and week: YEAR2023+WEEK15 = 202315
    # df[yearweek_col] = (df[year_col].astype(str) + df[week_col].astype(str)).astype(int)

    # Year and month and week: YEAR2023+MONTH8+WEEK15 = 2023815
    # df[yearmonthweek_col] = (df[year_col].astype(str) + df[month_col].astype(str) + df[week_col].astype(str)).astype(
    #     int)

    # Year and week and DOY: YEAR2023+MONTH8+WEEK15+DOY194 = 2023815194
    # df[yearmonthweekdoy_col] = (
    #         df[year_col].astype(str)
    #         + df[month_col].astype(str)
    #         + df[week_col].astype(str)
    #         + df[doy_col].astype(str)
    # ).astype(np.int64)  # todo check int64

    # Year and DOY: YEAR2023+DOY194 = 2023194
    # df[yeardoy_col] = (df[year_col].astype(str) + df[doy_col].astype(str)).astype(int)

    # Week and hour: WEEK42+HOUR22 = 4222
    # df[weekhour_col] = (df[week_col].astype(str) + df[hour_col].astype(str)).astype(int)

    if verbose > 0:
        print(f"Added timestamp as columns: {newcols} {txt}")

    return df


def insert_season(timestamp: DatetimeIndex) -> Series:
    """
    Insert meteorological season as integer

    spring = 1 (DJF)
    summer = 2 (MAM)
    autumn = 3 (JJA)
    winter = 4 (SON)

    Args:
        timestamp: timestamp of time series

    Returns:
        season series with timestamp
    """

    winter = [1, 2, 12]
    spring = [3, 4, 5]
    summer = [6, 7, 8]
    autumn = [9, 10, 11]
    season = pd.Series(data=timestamp.month, index=timestamp)

    is_winter = season.isin(winter)
    is_spring = season.isin(spring)
    is_summer = season.isin(summer)
    is_autumn = season.isin(autumn)

    season[is_spring] = 1
    season[is_summer] = 2
    season[is_autumn] = 3
    season[is_winter] = 4

    return season


class DetectFrequency:
    """Detect data time resolution from time series index


    - Example notebook available in:
        notebooks/TimeStamps/Detect_time_resolution.ipynb
    - Unittest:
        test_timestamps.TestTimestamps

    TODO detect freq like in dbc-influxdb

    """

    def __init__(self, index: pd.DatetimeIndex, verbose: bool = False):
        self.index = index
        self.verbose = verbose
        # self.freq_expected = freq_expected
        self.num_datarows = self.index.__len__()
        self.freq = None
        self._run()

    def _run(self):
        if self.verbose: print(f"Detecting time resolution from timestamp {self.index.name} ...", end=" ")

        freq_full, freqinfo_full = timestamp_infer_freq_from_fullset(timestamp_ix=self.index)
        freq_timedelta, freqinfo_timedelta = timestamp_infer_freq_from_timedelta(timestamp_ix=self.index)
        freq_progressive, freqinfo_progressive = timestamp_infer_freq_progressively(timestamp_ix=self.index)

        if all(f for f in [freq_full, freq_timedelta, freq_progressive]):

            # List of {Set of detected freqs}
            freq_list = list({freq_timedelta, freq_full, freq_progressive})

            if len(freq_list) == 1:
                # Maximum certainty, one single freq found across all checks
                self.freq = freq_list[0]
                if self.verbose:
                    print(f"OK\n"
                          f"   Detected {self.freq} time resolution with MAXIMUM confidence.\n"
                          f"   All approaches yielded the same result:\n"
                          f"       from full data = {freq_full} / {freqinfo_full} (OK)\n"
                          f"       from timedelta = {freq_timedelta} / {freqinfo_timedelta} (OK)\n"
                          f"       from progressive = {freq_progressive} / {freqinfo_progressive} (OK)\n")

        elif freq_full:
            # High certainty, freq found from full range of dataset
            self.freq = freq_full
            if self.verbose:
                print(f"OK\n"
                      f"   Detected {self.freq} time resolution with MAXIMUM confidence.\n"
                      f"   Full data has consistent timestamp:\n"
                      f"       from full data = {freq_full} / {freqinfo_full} (OK)\n"
                      f"       from timedelta = {freq_timedelta} / {freqinfo_timedelta} (not used)\n"
                      f"       from progressive = {freq_progressive} / {freqinfo_progressive} (not used)\n")

        elif freq_progressive:
            # Medium certainty, freq found from start and end of dataset
            self.freq = freq_progressive
            if self.verbose:
                print(f"OK (detected {self.freq} time resolution {self.freq} with MEDIUM confidence)")
            if self.verbose:
                print(f"OK\n"
                      f"   Detected {self.freq} time resolution with MEDIUM confidence.\n"
                      f"   Records at start and end of file have consistent timestamp:\n"
                      f"       from full data = {freq_full} / {freqinfo_full} (not used)\n"
                      f"       from timedelta = {freq_timedelta} / {freqinfo_timedelta} (not used)\n"
                      f"       from progressive = {freq_progressive} / {freqinfo_progressive} (OK)\n")

        elif freq_timedelta:
            # High certainty, freq found from most frequent timestep that
            # occurred at least 90% of the time
            self.freq = freq_timedelta
            if self.verbose:
                print(f"OK\n"
                      f"   Detected {self.freq} time resolution with HIGH confidence.\n"
                      f"   Resolution detected from most frequent timestep (timedelta):\n"
                      f"       from full data = {freq_full} / {freqinfo_full} (not used)\n"
                      f"       from timedelta = {freq_timedelta} / {freqinfo_timedelta} (OK)\n"
                      f"       from progressive = {freq_progressive} / {freqinfo_progressive} (not used)\n")

        else:
            raise Exception("Frequency detection failed.")

    def get(self) -> str:
        return self.freq


def timestamp_infer_freq_progressively(timestamp_ix: pd.DatetimeIndex) -> tuple:
    """Try to infer freq from first x and last x rows of data, if these
    match we can be relatively certain that the file has the same freq
    from start to finish.
    """
    # Try to infer freq, starting from first 1000 and last 1000 rows of data, must match
    n_datarows = timestamp_ix.__len__()
    inferred_freq = None
    freqinfo = None
    checkrange = 1000
    if n_datarows > 0:
        for ndr in range(checkrange, 3, -1):  # ndr = number of data rows
            if n_datarows >= ndr * 2:  # Same amount of ndr needed for start and end of file
                _inferred_freq_start = pd.infer_freq(timestamp_ix[0:ndr])
                _inferred_freq_end = pd.infer_freq(timestamp_ix[-ndr:])
                inferred_freq = _inferred_freq_start if _inferred_freq_start == _inferred_freq_end else None
                if inferred_freq:
                    freqinfo = f'data {ndr}+{ndr}' if inferred_freq else '-'
                    return inferred_freq, freqinfo
            else:
                continue
    return inferred_freq, freqinfo


def timestamp_infer_freq_from_fullset(timestamp_ix: pd.DatetimeIndex) -> tuple:
    """
    Infer data frequency from all timestamps in time series index

    Minimum 10 values are required in timeseries index.

    Args:
        timestamp_ix: Timestamp index

    Returns:
        Frequency string, e.g. '10T' for 10-minute time resolution
    """
    inferred_freq = None
    freqinfo = None
    n_datarows = timestamp_ix.__len__()
    if n_datarows < 10:
        freqinfo = '-not-enough-datarows-'
        return inferred_freq, freqinfo
    inferred_freq = pd.infer_freq(timestamp_ix)
    if inferred_freq:
        freqinfo = 'full data'
        return inferred_freq, freqinfo
    else:
        freqinfo = '-failed-'
        return inferred_freq, freqinfo


def timestamp_infer_freq_from_timedelta(timestamp_ix: pd.DatetimeIndex) -> tuple:
    """Check DataFrame index for frequency by subtracting successive timestamps from each other
    and then checking the most frequent difference

    - https://stackoverflow.com/questions/16777570/calculate-time-difference-between-pandas-dataframe-indices
    - https://stackoverflow.com/questions/31469811/convert-pandas-freq-string-to-timedelta
    """
    inferred_freq = None
    freqinfo = None
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
    if most_frequent_delta_perc > 0.90:
        inferred_freq = timedelta_to_string(most_frequent_delta)
        freqinfo = '>90% occurrence'
        # most_frequent_delta = pd.to_timedelta(most_frequent_delta)
        return inferred_freq, freqinfo
    else:
        freqinfo = '-failed-'
        return inferred_freq, freqinfo


def remove_index_duplicates(data: Series or DataFrame,
                            keep: Literal["first", "last", False] = "last",
                            verbose: bool = False) -> Series or DataFrame:
    """Remove index duplicates"""
    if verbose: print("Removing data records with duplicate indexes ...", end=" ")
    n_duplicates = data.index.duplicated().sum()
    if n_duplicates > 0:
        # Duplicates found
        data = data[~data.index.duplicated(keep=keep)]
        if verbose: print(f"OK (removed {n_duplicates} rows with duplicate timestamps)")
        return data
    else:
        # No duplicates found
        if verbose: print(f"OK (no duplicates found in timestamp index)")
        return data


def continuous_timestamp_freq(data: Series or DataFrame, freq: str, verbose: bool = False) -> Series or DataFrame:
    """Generate continuous timestamp of given frequency between first and last date of index

    This makes df continuous w/o date gaps but w/ data gaps at filled-in timestamps.
    """
    first_date = data.index[0]
    last_date = data.index[-1]

    if verbose: print(f"Creating continuous {freq} timestamp index for timestamp {data.index.name} "
                      f"between {first_date} and {last_date} ...", end=" ")

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


def insert_timestamp(
        data: DataFrame,
        convention: Literal['start', 'middle', 'end'],
        insert_as_first_col: bool = True,
        verbose: bool = False) -> DataFrame:
    """
    Insert timestamp column that shows the START, END or MIDDLE time of the averaging interval

    The new timestamp column is added as data column, the current
    *data* index remains unchanged.

    The current *data* index must be a properly named timestamp index.
    Allowed names are: 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE', 'TIMESTAMP_END'.

    Args:
        data: Dataset to which the new timestamp is added as new column
        convention: Timestamp convention of the new timestamp column
            - 'start': Timestamp denoting start of averaging interval
            - 'middle': Timestamp denoting middle of averaging interval
            - 'end': Timestamp denoting end of averaging interval
        insert_as_first_col: If *True*, the new timestamp column is
            added as the first column to *data*. If *False*, the new
            timestamp column is added as the last column to *data*.
        verbose: If *True*, gives additional text output

    Returns:
        *data* with newly added timestamp column

    Added in: v0.52.0
    """
    # Current index timestamp name
    timestamp_index_name = data.index.name

    # Check if current index timestamp properly named
    allowed_timestamp_names = ['TIMESTAMP_END', 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE']
    if timestamp_index_name not in allowed_timestamp_names:
        raise Exception("Timestamp index of the Series must be "
                        "named 'TIMESTAMP_END', 'TIMESTAMP_START' or 'TIMESTAMP_MIDDLE'.")

    # Name of new timestamp series
    new_timestamp_col = None
    if convention == 'start':
        new_timestamp_col = 'TIMESTAMP_START'
    elif convention == 'middle':
        new_timestamp_col = 'TIMESTAMP_MIDDLE'
    elif convention == 'end':
        new_timestamp_col = 'TIMESTAMP_END'

    # Get time resolution of data
    timestamp_freq = data.index.freq

    if verbose:
        print(f"\nAdding new timestamp column {new_timestamp_col} "
              f"to show {convention} of averaging period ...")

    # Interval of data records
    timedelta = pd.to_timedelta(timestamp_freq)
    timedelta_half = timedelta / 2

    # Data has MIDDLE timestamp
    if timestamp_index_name == 'TIMESTAMP_MIDDLE':
        if new_timestamp_col == 'TIMESTAMP_MIDDLE':
            data[new_timestamp_col] = data.index
        elif new_timestamp_col == 'TIMESTAMP_END':
            # '2023-03-05 18:15:00'  -->  '2023-03-05 18:30:00'
            data[new_timestamp_col] = data.index + pd.Timedelta(timedelta_half)
        elif new_timestamp_col == 'TIMESTAMP_START':
            # '2023-03-05 18:15:00'  -->  '2023-03-05 18:00:00'
            data[new_timestamp_col] = data.index - pd.Timedelta(timedelta_half)

    # Data has END timestamp
    elif timestamp_index_name == 'TIMESTAMP_END':
        if new_timestamp_col == 'TIMESTAMP_END':
            data[new_timestamp_col] = data.index
        elif new_timestamp_col == 'TIMESTAMP_MIDDLE':
            # '2023-03-05 18:30:00'  -->  '2023-03-05 18:15:00'
            data[new_timestamp_col] = data.index - pd.Timedelta(timedelta_half)
        elif new_timestamp_col == 'TIMESTAMP_START':
            # '2023-03-05 18:30:00'  -->  '2023-03-05 18:00:00'
            data[new_timestamp_col] = data.index - pd.Timedelta(timedelta)

    # Data has START timestamp
    elif timestamp_index_name == 'TIMESTAMP_START':
        if new_timestamp_col == 'TIMESTAMP_START':
            data[new_timestamp_col] = data.index
        elif new_timestamp_col == 'TIMESTAMP_MIDDLE':
            # '2023-03-05 18:00:00'  -->  '2023-03-05 18:15:00'
            data[new_timestamp_col] = data.index + pd.Timedelta(timedelta_half)
        elif new_timestamp_col == 'TIMESTAMP_END':
            # '2023-03-05 18:00:00'  -->  '2023-03-05 18:30:00'
            data[new_timestamp_col] = data.index + pd.Timedelta(timedelta)

    # Make new timestamp column the first column in data
    if insert_as_first_col:
        first_col = data.pop(new_timestamp_col)
        data.insert(0, new_timestamp_col, first_col)

    if verbose:
        print(f"    ++Added new timestamp column {new_timestamp_col}:\n"
              f"        first date: {data[new_timestamp_col].iloc[0]}\n"
              f"        last date:  {data[new_timestamp_col].iloc[-1]}")
        print(f"    The timestamp index was not changed:\n"
              f"        first date: {data.index[0]}\n"
              f"        last date:  {data.index[-1]}")

    return data


def convert_series_timestamp_to_middle(data: Series or DataFrame, verbose: bool = False) -> Series or DataFrame:
    """
    Convert the timestamp index to show middle of averaging period

    This conversion makes it easier to handle timeseries data. One of the
    issues it solves is that it becomes straight forward to aggregate data
    correctly.

    The timestamp of `data` must have one of the following names:
    - `TIMESTAMP_END` when the timestamp refers to the END of the averaging period.
      * Example: `2022-07-26 12:00` refers to the time period between `2022-07-26 11:30` and `2022-07-26 12:00`
    - `TIMESTAMP_START` when the timestamp refers to the START of the averaging period
      * Example: `2022-07-26 12:00` refers to the time period between `2022-07-26 12:00` and `2022-07-26 12:30`
    - `TIMESTAMP_MID` when the timestamp refers to the MIDDLE of the averaging period
        * Example: `2022-07-26 12:15` refers to the time period between `2022-07-26 12:00` and `2022-07-26 12:30`

    Note about timestamps:

        `TIMESTAMP_END` is widely used. However, aggregating data can
         easily lead to wrong aggregation windows. For example, half-hourly
         data for the day 26 July 2022 would have `2022-07-26 00:30` as the
         first valid timestamp for this day, the last timestamp would be
         `2022-07-27 00:00`. When these data are simply aggregated by *date*
         (`2022-07-26`), then `2022-07-26 00:00` would be taken as the first
         timestamp, and `2022-07-26 23:30` as the last timestamp, both of
         which is not correct. This leads to a wrongly attributed  first data
         record and a missing last record in this example.

        `TIMESTAMP_MID` solves this issue. Here, the first timestamp would be
        `2022-07-26 00:15`, and the last timestamp `2022-07-26 23:45`, both of
        which are correct when aggregating by *date*. The same is true for other
        aggregation windows, e.g., by month, year etc.

        The middle timestamp also helps in plotting the data correctly. Some plots
        set the ticks shown in the plot specifically at the start or end of the
        input timestamp, depending on the plot type. For example, plotting
        a heatmap might show the tick at `12:00` but then plots the respective
        data after the tick, which is not correct with `TIMESTAMP_END`. With the
        middle timestamp data are plotted correctly at `12:15`.

    Args:
        data: Data with timestamp index

    Returns:
        Data with timestamp index that shows the middle of the averaging period
    """
    timestamp_name_before = data.index.name
    timestamp_name_after = 'TIMESTAMP_MIDDLE'

    if timestamp_name_before == timestamp_name_after:
        return data

    timestamp_freq = data.index.freq

    if verbose: print(f"Converting timestamp index {timestamp_name_before} to show middle of averaging period ...")

    first_timestamp_before = data.index[0]
    last_timestamp_before = data.index[-1]

    if timestamp_name_before == 'TIMESTAMP_MIDDLE':
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
                            "named 'TIMESTAMP_END', 'TIMESTAMP_START' or 'TIMESTAMP_MIDDLE'.")

    data.index.name = 'TIMESTAMP_MIDDLE'
    first_timestamp_after = data.index[0]
    last_timestamp_after = data.index[-1]

    if verbose:
        print(f"    {timestamp_name_before} was converted to {timestamp_name_after}")
        print(f"    First and last dates:")
        print(f"        Before conversion: "
              f"{timestamp_name_before} from {first_timestamp_before} to {last_timestamp_before}")
        print(f"         After conversion: "
              f"{timestamp_name_after} from {first_timestamp_after} to {last_timestamp_after}")

    return data


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


def remove_after_date(data: Series or DataFrame, yearly_end_date: str) -> Series or DataFrame:
    """
    Remove data after specifified date

    Args:
        data: Data with timestamp index
        yearly_end_date: Month and day after which all data will be removed
            Example:
                "08-11" means that all data after 11 August will be removed,
                this is done for each year in the dataset. For a dataset that
                contains data of multiple years, e.g. 2016, 2017 and 2018, the
                returned dataset will only contain data from all years, but
                data for each year will end on 1 August (i.e., 11 August 2016,
                11 August 2017 and 11 August 2018).

    Returns:
        Data with all data after *yearly_end_date* removed
    """
    month = int(yearly_end_date[0:2])
    dayinmonth = int(yearly_end_date[3:])
    data.loc[(data.index.month > month)] = np.nan
    data.loc[(data.index.month == month) & (data.index.day > dayinmonth)] = np.nan
    data = data.dropna()
    return data


def keep_years(data: Series or DataFrame,
               start_year: int = None,
               end_year: int = None) -> Series or DataFrame:
    """
    Keep data between start and end year

    Args:
        data: Data with timeseries index
        start_year: First year of kept data
        end_year: Last year of kept data

    Returns:
        Data between start year and end year
    """
    if start_year:
        data = data.loc[data.index.year >= start_year]
    if end_year:
        data = data.loc[data.index.year <= end_year]
    return data


def calc_doy_timefraction(input_series: Series) -> DataFrame:
    df = pd.DataFrame(input_series)
    df['YEAR'] = df.index.year
    df['DOY'] = df.index.dayofyear
    df['TIMEFRACTION'] = (df.index.hour
                          + (df.index.minute / 60)
                          + (df.index.second / 3600)) / 24
    df['DOY_TIME'] = df['DOY'].add(df['TIMEFRACTION'])
    df[input_series.index.name] = df.index
    return df


def doy_cumulatives_per_year(series: Series) -> DataFrame:
    df = calc_doy_timefraction(input_series=series)
    return df.pivot(index='DOY_TIME', columns='YEAR', values=series.name).cumsum()


def doy_mean_cumulative(cumulatives_per_year_df: DataFrame,
                        excl_years_from_reference: list = None) -> DataFrame:
    reference_years_df = cumulatives_per_year_df.copy()
    if excl_years_from_reference:
        reference_years_df.drop(excl_years_from_reference, axis=1, inplace=True)
    df = pd.DataFrame()
    df['MEAN_DOY_TIME'] = reference_years_df.mean(axis=1)
    df['SD_DOY_TIME'] = reference_years_df.std(axis=1)
    df['MEAN+SD'] = df['MEAN_DOY_TIME'].add(df['SD_DOY_TIME'])
    df['MEAN-SD'] = df['MEAN_DOY_TIME'].sub(df['SD_DOY_TIME'])
    df['1.96_SD_DOY_TIME'] = df['SD_DOY_TIME'].multiply(1.96)
    df['MEAN+1.96_SD'] = df['MEAN_DOY_TIME'].add(df['1.96_SD_DOY_TIME'])
    df['MEAN-1.96_SD'] = df['MEAN_DOY_TIME'].sub(df['1.96_SD_DOY_TIME'])
    return df


if __name__ == '__main__':
    pass

    # # Test code
    # filepath = r'F:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\tests\testdata\testfile_ch-dav_2020.diive.csv'
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
