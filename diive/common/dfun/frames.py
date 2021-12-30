"""
DATA FUNCTIONS: FRAMES
======================

# last update in: v0.23.0

This module is part of DIIVE:
https://gitlab.ethz.ch/holukas/diive

"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas._libs.tslibs import to_offset

from common.dfun.times import timedelta_to_string

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


def splitdata_daynight(
        df: pd.DataFrame,
        split_on_col: tuple,
        split_threshold: int = 20,
        split_flagtrue: str = 'Larger Than Threshold'
):
    """
    Split data into two separate datasets based on column

    Consecutive nighttime

    Done by generating a day/night flag that is then used to
    split the dataset.

    For example:
        Split dataset into separate daytime and nighttime datasets
        based on radiation.

    """

    date_col = ('.DATE', '[aux]')
    grp_daynight_col = ('.GRP_DAYNIGHT', '[aux]')

    # Add daytime flag to main data
    df, flag_daynight_col = \
        generate_flag_daynight(df=df,
                               flag_based_on_col=split_on_col,
                               flag_threshold=split_threshold,
                               flagtrue=split_flagtrue)



    # Add date as column
    df[date_col] = df.index.date
    df[date_col] = pd.to_datetime(df[date_col])

    # Find *consecutive* daytimes and nighttimes
    # While daytime is always consecutive (one date), the nighttime
    # spans midnight and therefore two different dates. This means that
    # one specfic night starts in the evening and ends on the next day
    # in the morning.
    df[grp_daynight_col] = (df[flag_daynight_col].diff(1) != 0).astype('int').cumsum()

    # Data where flag is 1
    daytime_ix = df[flag_daynight_col] == 1
    df_flagtrue = df[daytime_ix].copy()

    # Data where flag is 0
    daytime_ix = df[flag_daynight_col] == 0
    df_flagfalse = df[daytime_ix].copy()

    return df_flagtrue, df_flagfalse, grp_daynight_col, date_col, flag_daynight_col


def create_random_gaps(series: pd.Series, frac: float = 0.1):
    """Create random gaps in series"""
    ix_gaps = series.sample(frac=frac).index
    series_with_gaps = series.loc[ix_gaps] = np.nan
    return series_with_gaps


def get_len_data(filepath: Path,
                 skiprows: list = None,
                 headerrows: list = None):
    """Check number of columns of the first data row after the header part"""
    skip_num_lines = len(headerrows) + len(skiprows)
    first_data_row_df = pd.read_csv(filepath,
                                    skiprows=skip_num_lines,
                                    header=None,
                                    nrows=1)
    return first_data_row_df.columns.size


def get_len_header(filepath: Path,
                   skiprows: list = None,
                   headerrows: list = None):
    """Check number of columns of the header part"""
    headercols_df = pd.read_csv(filepath,
                                skiprows=skiprows,
                                header=headerrows,
                                nrows=0)
    num_headercols = headercols_df.columns.size
    header_cols_list = headercols_df.columns.to_list()
    return num_headercols, header_cols_list


def df_unique_values(df):
    """
    Return numpy array of unique values across all columns of a dataframe

    Parameters
    ----------
    df: pandas DataFrame

    Returns
    -------
    array
    """
    return pd.unique(df.values.ravel())


def count_unique_values(df):
    """
    Count number of occurrences of unique values across DataFrame

    Parameters
    ----------
    df: pandas DataFrame

    Returns
    -------
    pandas DataFrame
    """
    _unique_values = df_unique_values(df=df)
    counts_df = pd.DataFrame(index=_unique_values)
    for col in df.columns:
        counts_df[col] = df[col].value_counts(dropna=False)
    return counts_df.sort_index()


def flatten_multiindex_all_df_cols(df):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


def remove_duplicate_cols(df):
    return df.loc[:, ~df.columns.duplicated()]


def downsample_data(df, freq, max_freq):
    """ Downsample data if freq is faster than 1S b/c otherwise maaaany datapoints. """
    if to_offset(freq) < to_offset(max_freq):
        df = df.apply('1S').mean()
        new_freq = '1S'
    else:
        new_freq = freq
    return df, new_freq


def infer_freq(df_index):
    """
    Checks DataFrame index for frequency by subtracting successive timestamps from each other
    and then checking the most frequent difference.
    """
    # https://stackoverflow.com/questions/16777570/calculate-time-difference-between-pandas-dataframe-indices
    # https://stackoverflow.com/questions/31469811/convert-pandas-freq-string-to-timedelta
    df = pd.DataFrame(columns=['tvalue'])
    df['tvalue'] = df_index
    df['tvalue_shifted'] = df['tvalue'].shift()
    df['delta'] = (df['tvalue'] - df['tvalue_shifted'])
    most_frequent_delta = df['delta'].mode()[0]  # timedelta
    most_frequent_delta = timedelta_to_string(most_frequent_delta)
    # most_frequent_delta = pd.to_timedelta(most_frequent_delta)

    return most_frequent_delta


def insert_drop_timestamp_col(df, timestamp_colname):
    """
    Insert timestamp column index as normal column, then drop index.

    Row index is not exported as index, but as regular column. This way,
    the output file looks as expected with the index col coming first.

    :param df: DataFrame that is exported to file
    :return: DataFrame
    """
    _df = df.copy()
    try:
        _df.drop([timestamp_colname], axis=1, inplace=True)
    except:
        pass
    _df.insert(0, timestamp_colname, value=_df.index)  ## index col to normal data col
    return _df


def resample_df(df, freq_str, agg_method, min_vals, to_freq_duration, to_freq):
    """
    Resample data to selected frequency, using the selected aggregation method
    and while also considering the minimum required values in the aggregation
    time window.


    Note regarding .resample:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
        closed : {‘right’, ‘left’}, default None
            Which side of bin interval is closed. The default is ‘left’ for all frequency offsets
            except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’, and ‘W’ which all have a default of ‘right’.

        label : {‘right’, ‘left’}, default None
            Which bin edge label to label bucket with. The default is ‘left’ for all frequency offsets
            except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’, and ‘W’ which all have a default of ‘right’.

    By default, for weekly aggregation the first day of the week in pandas is Sunday, but Amp uses Monday.

    https://stackoverflow.com/questions/48340463/how-to-understand-closed-and-label-arguments-in-pandas-resample-method
        closed='right' =>  ( 3:00, 6:00 ]  or  3:00 <  x <= 6:00
        closed='left'  =>  [ 3:00, 6:00 )  or  3:00 <= x <  6:00

    """

    # RESAMPLING
    # ----------
    _df = df.copy()

    if to_freq in ['W', 'M', 'A']:
        label = 'right'
        closed = 'right'
        timestamp_shows_start = False
    elif to_freq in ['T', 'H', 'D']:
        label = 'left'
        closed = 'left'
        timestamp_shows_start = True
    else:
        label = closed = timestamp_shows_start = '-not-defined-'

    resampled_df = _df.resample(freq_str, label=label, closed=closed)  #

    # AGGREGATION
    # -----------
    agg_counts_df = resampled_df.count()  # Count aggregated values, always needed

    # Aggregated values
    if agg_method == 'Mean':
        agg_df = resampled_df.mean()
    elif agg_method == 'Median':
        agg_df = resampled_df.median_col()
    elif agg_method == 'SD':
        agg_df = resampled_df.std_col()
    elif agg_method == 'Minimum':
        agg_df = resampled_df.min()
    elif agg_method == 'Maximum':
        agg_df = resampled_df.max()
    elif agg_method == 'Count':
        agg_df = resampled_df.count()
    elif agg_method == 'Sum':
        agg_df = resampled_df.sum()
    elif agg_method == 'Variance':
        agg_df = resampled_df.var_col()
    else:
        agg_df = -9999

    filter_min = agg_counts_df >= min_vals
    agg_df = agg_df[filter_min]

    # TIMESTAMP CONVENTION
    # --------------------
    agg_df, timestamp_info_df = timestamp_convention(df=agg_df,
                                                     timestamp_shows_start=timestamp_shows_start,
                                                     out_timestamp_convention='Middle of Record')

    agg_df.index = pd.to_datetime(agg_df.index)

    return agg_df, timestamp_info_df


def timestamp_convention(df, timestamp_shows_start, out_timestamp_convention):
    """Set middle timestamp as main index"""

    original_cols = df.columns
    df.index = pd.to_datetime(df.index)
    orig_freq = df.index.freq

    # Original timestamp name
    ts_col = df.index.name
    ts_var = ts_col[0]
    ts_units = ts_col[1]

    # AUXILIARY COLUMNS
    # -----------------
    # Define columns names
    if timestamp_shows_start:
        start = (f"{ts_var}_START_INCL", ts_units)
        end = (f"{ts_var}_END_EXCL", ts_units)
    else:
        start = (f"{ts_var}_START_EXCL", ts_units)
        end = (f"{ts_var}_END_INCL", ts_units)

    middle = (f"{ts_var}_MIDDLE", ts_units)
    diff = (f"{ts_var}_DIFF", ts_units)
    halfdiff = (f"{ts_var}_HALFDIFF", ts_units)
    aux_cols = [start, middle, end, diff, halfdiff]
    # Add columns as empty
    for col in aux_cols:
        df[col] = np.nan

    # Calculate auxiliary, e.g. to calculate timestamp for middle of record
    if timestamp_shows_start:
        df[start] = df.index  # Original timestamp after resampling
        df[end] = df[start].shift(-1)  # Shifted for time difference between next and current
        fill_last_end = pd.date_range(start=df.index[-1], periods=2, freq=df.index.freq)
        df[end].iloc[-1] = fill_last_end[1]  # Otherwise empty b/c shift

    else:
        df[end] = df.index  # Original timestamp after resampling
        df[start] = df[end].shift(1)  # Shifted for time difference between next and current
        fill_first_start = pd.date_range(end=df.index[0], periods=2, freq=df.index.freq)
        df[start].iloc[0] = fill_first_start[0]  # Otherwise empty b/c shift

    df[diff] = df[end] - df[start]  # Time difference
    df[halfdiff] = df[diff] / 2  # Half-difference for middle of record
    df[middle] = df[end] - df[halfdiff]  # Middle of record

    # APPLY
    # -----
    # according to selected convention
    if out_timestamp_convention == 'Middle of Record':
        df.index = df[middle]
    elif out_timestamp_convention == 'End of Record':
        df.index = df[end]
    elif out_timestamp_convention == 'Start of Record':
        df.index = df[start]

    # Set frequency
    # The problem is that if the middle timestamp is set as the main timestamp, the
    # freq info is lost, although freq was not changed. Therefore, there is a check
    # here if the inferred freq from the new timestamp index is still the same as
    # the original timestamp index freq.
    _freq = pd.infer_freq(df.index)  # Inferred freq from new timestamp index
    if _freq == orig_freq:
        pass
    df = df.asfreq(orig_freq)

    # Format timestamp to full datetime
    # df.index
    # df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
    # df.index = pd.to_datetime(df.index)
    df.index.name = ts_col

    # Collect timestamp info in df
    timestamp_info_df = df[aux_cols]

    # Remove aux columns from data by only keeping original columns in df
    df = df[original_cols]

    return df, timestamp_info_df


def df_between_two_dates(df, start_date, end_date, dropna, dropna_col):
    """ Get data for the time window, greater than the start date and smaller than the end date.

    :param df: DataFrame
    :param start_date: datetime
    :param end_date: datetime
    :param dropna: bool
    :param dropna_col: Column name in df that is checked for NaNs.
    :return: Data between start_date and end_date.
    :rtype: DataFrame
    """
    mask = (df.index >= start_date) & (df.index <= end_date)
    snippet_df = df.loc[mask]

    if dropna:
        # Remove rows in df where dropna_col is NaN.
        filter_nan = snippet_df[dropna_col].isnull()  # True if NaN
        snippet_df = snippet_df.loc[~filter_nan]  # Keep False (=values, i.e. not NaN)
        # snippet_df = snippet_df.dropna()
    return snippet_df


def insert_datetimerange(df, win_days, win_hours):
    """Insert datetime range that describes start and end of chosen day window."""
    win_start_col = ('start_dt_{}d-{}h'.format(win_days, win_hours), '[datetime]')
    win_end_col = ('end_dt_{}d-{}h'.format(win_days, win_hours), '[datetime]')
    df[win_start_col] = df.index - pd.Timedelta(days=win_days) - pd.Timedelta(hours=win_hours)
    df[win_end_col] = df.index + pd.Timedelta(days=win_days) + pd.Timedelta(hours=win_hours)
    return df, win_start_col, win_end_col


def insert_timerange(df, win_hours):
    """Insert time range that describes start and end of chosen hour window."""
    win_start_time_col = ('start_time_{}h'.format(win_hours), '[time]')
    win_end_time_col = ('end_time_{}h'.format(win_hours), '[time]')
    df[win_start_time_col] = df.index - pd.Timedelta(hours=win_hours)
    df[win_end_time_col] = df.index + pd.Timedelta(hours=win_hours)

    df[win_start_time_col] = df[win_start_time_col].dt.time  # keep only time
    df[win_end_time_col] = df[win_end_time_col].dt.time
    return df, win_start_time_col, win_end_time_col


def find_nans_in_df_col(df, col):
    """
    Reduces df to rows where values for col are missing

    :param df:              pandas DataFrame; that is reduced to data rows where col is empty
    :param col:             tuple; column name in df that is used to reduce df
    :param prepost:         str; help string in output that indicates when the method was called

    :return: gaps_df:       pandas DataFrame; reduced df that only contains data rows where col is empty
    :return: gaps_exist:    bool; True = there are data rows where values for col are missing
    """
    temp = pd.isnull(df[col]).to_numpy().nonzero()
    gaps_df = df.iloc[temp]  # contains only NEE NaNs, but the index is important to locate missing NEE values
    gap_count = len(gaps_df[col])
    return gaps_df, gap_count


def sort_multiindex_columns_names(df, priority_vars):
    """ Sort column names, ascending, with priority vars at top

        Files w/ many columns are otherwise hard to navigate.

        This is trickier than anticipated (by me), b/c sorting by
              data_df.sort_index(axis=1, inplace=True)
        sorts the columns, but is case-sensitive, i.e. 'Var2' is placed before
        'var1', yielding the order 'Var2', 'var1'. However, we want it sorted like
        this: 'var1', 'Var2'. Therefore, the columns are converted to a list, the
        list is then sorted ignoring case, and the sorted list is then used to
        define the column order in the df.
    """
    cols_list = df.columns.to_list()  # list of tuples

    def custom_sort(col):
        return col[0].lower()  # sort by 1st tuple element (var name), strictly lowercase

    cols_list.sort(key=custom_sort)

    for ix, col in enumerate(cols_list):
        if col[0] in priority_vars:
            cols_list.insert(0, cols_list.pop(ix))  # removes from old location ix, puts to top of list

    # Custom vars are marked w/ a dot ('.') at beginning
    for ix, col in enumerate(cols_list):
        if col[0].startswith('.'):
            cols_list.insert(0, cols_list.pop(ix))

    df = df[cols_list]  # assign new (sorted) column order

    return df


def sort_multiindex_columns_names_LEGACY(df, priority_vars):
    """ Sort column names, ascending, with priority vars at top

        Files w/ many columns are otherwise hard to navigate.

        This is trickier than anticipated (by me), b/c sorting by
              data_df.sort_index(axis=1, inplace=True)
        sorts the columns, but is case-sensitive, i.e. 'Var2' is placed before
        'var1', yielding the order 'Var2', 'var1'. However, we want it sorted like
        this: 'var1', 'Var2'. Therefore, the columns are converted to a list, the
        list is then sorted ignoring case, and the sorted list is then used to
        define the column order in the df.
    """
    cols_list = df.columns.to_list()  # list of tuples

    def custom_sort(col):
        return col[0].lower()  # sort by 1st tuple element (var name), strictly lowercase

    cols_list.sort(key=custom_sort)

    # # Put priority vars first
    # top_of_list = []
    # for pv in priority_vars:
    #     cols_list
    #     top_of_list.append(pv)

    for ix, col in enumerate(cols_list):
        if col[0] in priority_vars:
            cols_list.insert(0, cols_list.pop(ix))  # removes from old location ix, puts to top of list

    for ix, col in enumerate(cols_list):
        # Custom vars are marked w/ ++ at beginning
        if (col[0].startswith('++')) and ('++_' not in str(col[0])):
            cols_list.insert(0, cols_list.pop(ix))

    # for ix, col in enumerate(cols_list):
    #     # Underscore marks auxiliary varibles, move to end of list
    #     if '++_' in str(col[0]):
    #         cols_list.insert(-1, cols_list.pop(ix))

    # elif any(fnmatch.fnmatch(col[0], gf_id) for gf_id in GAPFILLED_GENERAL_SCALARS):
    #     cols_list.insert(0, cols_list.pop(ix))

    df = df[cols_list]  # assign new (sorted) column order

    return df


def export_to_main(main_df, export_df, tab_data_df):
    """Rename cols so there is no overlap with other vars"""
    mapping_dict = {}

    # Tuples for lookup, does not work with MultiIndex columns
    export_df.columns = [tuple(x) for x in export_df.columns]

    # Add '++' to all variable names, or '+' if var was renamed previously.
    # This means that all newly created variables have a prefix of '++' or more.
    for col in export_df.columns:
        if str(col[0]).startswith('++'):
            renamed_col = ('+' + col[0], col[1])
        else:
            renamed_col = ('++' + col[0], col[1])
        while renamed_col in tab_data_df.columns:
            renamed_col = ('+' + renamed_col[0], renamed_col[1])
        mapping_dict[col] = renamed_col

    # Apply new names
    export_df.rename(columns=mapping_dict, inplace=True)

    # Make sure columns are MultiIndex
    export_df.columns = pd.MultiIndex.from_tuples(export_df.columns)

    # Init new cols in main
    for newcol in export_df.columns:
        main_df[newcol] = np.nan

    # Data to main
    main_df = main_df.combine_first(export_df)
    return main_df


def add_to_main(main_df, export_df):
    """Rename cols so there is no overlap with other vars"""
    # # Make sure columns are MultiIndex
    # export_df.columns = pd.MultiIndex.from_tuples(export_df.columns)

    # Init new cols in main
    for exported_col in export_df.columns:
        exported_series = export_df[exported_col]
        while exported_col in main_df.columns:
            exported_col = ('.' + exported_col[0], exported_col[1])
        exported_series.name = exported_col
        main_df[exported_series.name] = exported_series
    return main_df


def move_col_to_pos(df, colname, pos):
    # Changing columns order: https://engineering.hexacta.com/pandas-by-example-columns-547696ff78dd
    cols = df.columns.tolist()
    column_to_move = colname
    new_position = pos
    cols.insert(new_position,
                cols.pop(cols.index(column_to_move)))  # pandas pop: Return item and drop from frm_CategoryOptions
    df = df[cols]

    return df


def limit_data_range_percentiles(df, col, perc_limits):
    p_lower = df[col].quantile(perc_limits[0])
    p_upper = df[col].quantile(perc_limits[1])
    p_filter = (df[col] >= p_lower) & (df[col] <= p_upper)
    df = df[p_filter]
    return df


def add_second_header_row(df):
    lst_for_empty_units = []
    for e in range(len(df.columns)):  ## generate entry for all cols in df
        lst_for_empty_units.append('-no-units-')
    df.columns = [df.columns, lst_for_empty_units]  ## conv column index to multiindex
    return df


def create_lagged_variants(df: pd.DataFrame(), num_lags: int = 1, ignore_cols: list = None, info: bool = True):
    newdf = pd.DataFrame()
    lagged_cols = []
    ignored_cols = []
    for col in df.columns:
        if col in ignore_cols:
            newdf[col] = df[col].copy()
            ignored_cols.append(col)
            continue
        for lag in range(1, num_lags):
            newcol = f"({col[0]}+{lag}, {col[1]})"
            newdf[newcol] = df[col].shift(lag)
        lagged_cols.append(col)
    if info:
        print(f"Created lagged variants for: {lagged_cols}\n"
              f"No lagged variants for: {ignored_cols}")
    return newdf


def convert_to_arrays(df: pd.DataFrame, target_col: tuple, complete_rows: bool = True):
    """Convert data from df to numpy arrays and prepare targets, features and their timestamp"""

    # Keep complete rows only
    _df = df.dropna() if complete_rows else df.copy()

    # Targets (will be predicted)
    targets = np.array(_df[target_col])
    _df = _df.drop(target_col, axis=1)

    # Column names of features
    features_names = list(_df.columns)

    # Features (used to predict target)
    features = np.array(_df)

    # Original timestamp, will be merged with data later
    timestamp = np.array(_df.index)

    return targets, features, features_names, timestamp


def rolling_variants(df, records: int, aggtypes: list, exclude_cols: list = None) -> pd.DataFrame:
    """Create rolling variants of variables

    Calculates rolling aggregation over *records* of type *aggtypes*

    For example, with records=5 and aggtypes=['mean', 'max'] the mean
    and max over 5 records is produced.

    """
    for col in df.columns:
        if exclude_cols:
            if col in exclude_cols:
                continue

        for aggtype in aggtypes:
            aggname = (f".{col[0]}.r-{aggtype}{records}", col[1])
            min_periods = int(np.ceil(records / 2))
            df[aggname] = df[col].rolling(records, min_periods=min_periods).agg(aggtype)

    return df


def steplagged_variants(df: pd.DataFrame(),
                        stepsize: int = 1,
                        stepmax: int = 10,
                        exclude_cols: list = None,
                        info: bool = True):
    """Create step-lagged (no overlaps) variants of variables"""

    _included = []
    _excluded = []

    lagsteps = range(stepsize, stepmax + 1, stepsize)

    # # Create lagsteps for each period and collect
    # for s in stepsize:
    #     p_lagsteps = list(range(s, s * (num_lags + 1), s))
    #     # p_lagsteps = list(range(p, p * 4, p))
    #     [lagsteps.append(x) for x in p_lagsteps if x not in lagsteps]  # Avoid duplicates
    # lagsteps.sort(reverse=False)

    for col in df.columns:
        if exclude_cols:

            if col in exclude_cols:
                _excluded.append(col)
                continue

            for lagstep in lagsteps:
                stepname = (f".{col[0]}+{lagstep}", col[1])
                df[stepname] = df[col].shift(lagstep)
            _included.append(col)

        # if col[0].startswith('.'):
        #     for lagstep in lagsteps:
        #         stepname = (f".{col[0]}+{lagstep}", col[1])
        #         df[stepname] = df[col].shift(lagstep)
        #     _included.append(col)
        # else:
        #     _excluded.append(col)
        #     continue

    if info:
        print(f"Created step-lagged variants for: {_included}\n"
              f"No step-lagged variants for: {_excluded}")
    return df


def generate_flag(df: pd.DataFrame, target_col: tuple, tag: str,
                  upperlim_col: tuple, lowerlim_col: tuple, criterion_col=None):
    """Calculate flag where 1=True (outlier) and 0=False (no outlier)"""

    # Flag is based on criterion_col, which can be the target or some other col in df
    criterion_col = target_col if not criterion_col else criterion_col

    # Naming, based on target
    varname = f"{target_col[0]}"
    units = f"{target_col[1]}"
    tag = '' if tag in varname else tag
    prefix = '.'
    # prefix = '' if varname.startswith('.') else '.'

    # Get records within limits (bool True means within limits)
    flag_col = (f"{prefix}QCF_{varname}{tag}", "[1=outlier]")  # Outlier flag
    df[flag_col] = \
        (df[criterion_col] < df[upperlim_col]) & \
        (df[criterion_col] > df[lowerlim_col])

    # Convert bool flag to integers 0 and 1
    # The bool series needs to be inverted first, so that bool True
    # means outside limits. True is then translated to integer 1.

    # Set missing values to True (outside limits)
    # BUT: this also gives True for nan (when criterion is nan/missing)
    df[flag_col] = ~df[flag_col]

    # Convert True/False to 1/0 (1=outlier, 0=no outlier)
    df[flag_col] = df[flag_col].astype(int)

    # Set flag to nan if criterion is missing
    df.loc[df[criterion_col].isnull(), flag_col] = np.nan

    # Series that contains outlier values only
    target_outliervals_col = (f"{prefix}{varname}{tag}_outliers", units)
    is_outlier_ix = df[flag_col] == 1
    df.loc[is_outlier_ix, target_outliervals_col] = df[target_col]

    # Target series where outliers were removed
    target_nooutliers_col = (f"{prefix}{varname}{tag}", units)
    is_not_outlier_ix = df[flag_col] == 0
    df.loc[is_not_outlier_ix, target_nooutliers_col] = df[target_col]

    return df, flag_col, target_nooutliers_col, target_outliervals_col


def generate_flag_daynight(df: pd.DataFrame, flag_based_on_col: tuple, flagtrue: str, flag_threshold: float):
    """Add flag to indicate group membership: daytime or nighttime data"""
    flag_col = ('.FLAG_DAYTIME', '[1=daytime]')
    df[flag_col] = np.nan
    daytime_ix = None
    nighttime_ix = None
    if flagtrue == 'Larger Than Threshold':
        daytime_ix = df[flag_based_on_col] > flag_threshold
        nighttime_ix = df[flag_based_on_col] <= flag_threshold
    elif flagtrue == 'Smaller Than Threshold':
        daytime_ix = df[flag_based_on_col] < flag_threshold
        nighttime_ix = df[flag_based_on_col] >= flag_threshold
    df.loc[daytime_ix, [flag_col]] = 1
    df.loc[nighttime_ix, [flag_col]] = 0
    return df, flag_col


def init_class_df(df: pd.DataFrame, subset_cols: list, newcols: list):
    """Initialize df with columns required by class"""
    class_df = pd.DataFrame(df[subset_cols])  # Only required cols
    class_df.sort_index(axis=1, inplace=True)  # lexsort for better performance
    for newcol in newcols:
        class_df[newcol] = np.nan
    return class_df
