import datetime as dt
import logging
import time

import pandas as pd

from diive.logger import log


def get_current_unixtime():
    """
    Get current time as an integer number of nanoseconds since the epoch

    Returns
    -------
    int
    """
    current_time_unix = time.time_ns()
    return current_time_unix


def make_timestamp_suffix():
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%Y%m%d%H%M%S")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def make_timestamp_microsec_suffix():
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%H%M%S%f")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id


def get_current_time(str_format: str = '%Y-%m-%d %H:%M:%S'):
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime(str_format)
    logging.info(f'{now_time_dt} / {now_time_str}')
    log(name=get_current_time.__name__, dict={'current time': [now_time_dt, now_time_str]}, highlight=False)
    return now_time_dt, now_time_str


def make_run_id(prefix: str = False):
    now_time_dt, _ = get_current_time()
    now_time_str = now_time_dt.strftime("%Y%m%d-%H%M%S")
    prefix = prefix if prefix else "RUN"
    run_id = f"{prefix}-{now_time_str}"
    return run_id


def timedelta_to_string(timedelta):
    """
    Converts a pandas.Timedelta to a string rappresentation
    compatible with pandas.Timedelta constructor format
    https://stackoverflow.com/questions/46429736/pandas-resampling-how-to-generate-offset-rules-string-from-timedelta
    https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
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


def include_timestamp_as_cols(df, doy: bool = True, week: bool = True, month: bool = True, hour: bool = True,
                              info: bool = True):
    """Include timestamp info as data columns"""

    newcols = []
    doy_col = ('.DOY', '[day_of_year]')
    week_col = ('.WEEK', '[week_of_year]')
    month_col = ('.MONTH', '[month]')
    hour_col = ('.HOUR', '[hour]')

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
