"""
DATA FUNCTIONS: FRAMES
======================

This module is part of the diive library:
https://github.com/holukas/diive

"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex
from pandas import Series

from diive.core.funcs.funcs import find_duplicates_in_list
from diive.core.times.times import current_time_microseconds_str
# from diive.core.times.times import timedelta_to_string
from diive.pkgs.gapfilling.interpolate import linear_interpolation

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 50)


def compare_len_header_vs_data(n_cols_data: int, n_cols_header: int, varnames_list: list, varunits_list: list):
    """
    Check whether there are more data columns than given in the header

    If not checked, this would results in an error when reading the csv file
    with .read_csv, because the method expects an equal number of header and
    data columns. If this check is True, then the difference between the length
    of the first data row and the length of the header row(s) can be used to
    automatically generate names for the missing header columns.
    """

    # Check if there are more data columns than header columns
    more_data_cols_than_header_cols = False
    num_missing_header_cols = 0
    if n_cols_data > n_cols_header:
        more_data_cols_than_header_cols = True
        num_missing_header_cols = n_cols_data - n_cols_header

    # Generate missing header columns if necessary
    generated_missing_header_cols_list = []
    sfx = current_time_microseconds_str()
    if more_data_cols_than_header_cols:
        for m in list(range(1, num_missing_header_cols + 1)):
            missing_col = f'unknown-{m}-{sfx}'
            generated_missing_header_cols_list.append(missing_col)
            varnames_list.append(missing_col)
            varunits_list.append('[-unknown-]')

    # Check column names for duplicates, add suffix if necessary
    duplicate_varnames = find_duplicates_in_list(varnames_list)
    if len(duplicate_varnames) > 0:
        new_varnames_list = []
        for i, v in enumerate(varnames_list):
            totalcount = varnames_list.count(v)
            count = varnames_list[:i].count(v)
            new_varnames_list.append(f"{v}_{str(count + 1)}" if totalcount > 1 else v)
        varnames_list = new_varnames_list.copy()

    return varnames_list, varunits_list, generated_missing_header_cols_list


def trim_frame(df: DataFrame, var: str) -> DataFrame:
    """Trim start and end of dataframe, based on the first and last
    record of a specific variable.

    Only start and end of the dataframe are trimmed, other missing values
    of *var* are ignored.

    Example:
        If *df* contains multiple variables with data between 05:00 and
        06:00, but *var* is only available between 05:20 and 05:50, then
        *df* is trimmed: all data before 05:20 and after 05:50 are removed
        and the returned dataframe is thus shorter.

    Args:
        df: Dataframe that contains *var*.
        var: Name of the variable in *df* that is used to trim *df*.

    Returns:
        Trimmed dataframe.

    """
    records = df[var].copy()
    records = records.dropna()
    if records.empty:
        df = pd.DataFrame()
    else:
        first_record = records.index[0]
        last_record = records.index[-1]
        keep = (df.index >= first_record) & (df.index <= last_record)
        df = df[keep].copy()
    return df


def detect_new_columns(df: DataFrame, other: DataFrame) -> list:
    """Detect columns in *df* that do not exist in *other*."""
    duplicatecols = [c for c in df.columns if c in other.columns]
    for col in df[duplicatecols]:
        if not df[col].equals(other[col]):
            raise Exception(
                f"Column {col} was identified as duplicate, but is not identical. "
                f"This error can occur e.g. when features used in machine learning models "
                f"have gaps.")

    newcols = [c for c in df.columns if c not in other.columns]

    return newcols


def aggregated_as_hires(aggregate_series: Series,
                        hires_timestamp,
                        to_freq: str = 'D',
                        to_agg: str = 'mean',
                        interpolate_missing_vals: bool = False,
                        interpolation_lim: int = False) -> Series:
    """
    Calculate aggregated values and apply high-res timestamp

    Example: half-hourly timestamp for daily maximum temperature
    """
    # Aggregate series
    lowres_df = pd.DataFrame(aggregate_series.resample(to_freq).agg(to_agg))
    # lowres_df = lowres_df.rolling(window=5, center=True).mean()  # Testing
    agghires_col = f".{aggregate_series.name}_{to_freq}_{to_agg}"
    lowres_df = rename_cols(df=lowres_df, renaming_dict={aggregate_series.name: agghires_col})

    # Fill missing values in agg
    if lowres_df[agghires_col].isnull().any() \
            and interpolate_missing_vals \
            and interpolation_lim:
        lowres_df[agghires_col] = linear_interpolation(series=lowres_df[agghires_col],
                                                       limit=interpolation_lim)

    # Testing:
    # lowres_df = lowres_df.backfill()
    # lowres_df = lowres_df.ffill()

    # lowres_df.interpolate(limit=1)
    # todo here linear interpolation with gap length limit

    # Input data
    hires_df = pd.DataFrame(index=hires_timestamp)
    hires_df[aggregate_series.name] = aggregate_series

    # Insert column to merge the two datasets on
    mergecol = None
    if to_freq == 'D':
        mergecol = 'date'
        lowres_df[mergecol] = lowres_df.index.date
        hires_df[mergecol] = hires_df.index.date
    elif to_freq == 'A':
        mergecol = 'year'
        lowres_df[mergecol] = lowres_df.index.year
        hires_df[mergecol] = hires_df.index.year
    elif to_freq == 'M':
        mergecol = 'year-month'
        lowres_df[mergecol] = lowres_df.index.year.astype(str).str.cat(lowres_df.index.month.astype(str).str.zfill(2),
                                                                       sep='-')
        hires_df[mergecol] = hires_df.index.year.astype(str).str.cat(hires_df.index.month.astype(str).str.zfill(2),
                                                                     sep='-')

    # Timestamp as column for index after merging (merging loses index)
    hires_df['_TIMESTAMP'] = hires_df.index
    hires_df = hires_df.merge(lowres_df, left_on=mergecol, right_on=mergecol, how='left')

    # Re-apply original index (merging lost index)
    hires_df = hires_df.set_index('_TIMESTAMP')
    hires_df.index.name = hires_timestamp.name
    return hires_df[agghires_col]


def rename_cols(df: DataFrame, renaming_dict: dict) -> DataFrame:
    """
    Rename columns in dataframe

    Args:
        df: Data containing *oldname* column
        renaming_dict: Dictionary that contains old column names as its keys,
            and new column names as values, e.g. {'oldname': 'newname', 'oldname2': 'newname2', ...}.

    Returns:
        DataFrame
    """
    multiindex = False
    if isinstance(df.columns, MultiIndex):  # Convert MultiIndex to flat index (tuples)
        df.columns = df.columns.to_flat_index()
        multiindex = True
    df = df.rename(columns=renaming_dict, inplace=False)
    if multiindex:
        df.columns = pd.MultiIndex.from_tuples(df.columns)  # Restore column MultiIndex
    return df


def rename_cols_to_multiindex(df: DataFrame, renaming_dict: dict) -> DataFrame:
    """Rename columns (one-row header) in dataframe to multiindex (two-row header).

    Args:
        df: Data containing *oldname* column
        renaming_dict: Dictionary that contains old column names as its keys,
            and new column names with units as tuples:
                e.g. {
                    'oldname1': ('newname1', 'units1'),
                    'oldname2': ('newname2', 'units2'),
                    'oldname3': ('newname3', 'units3'),
                    ...
                    }

    Returns:
        DataFrame
    """
    df = df.rename(columns=renaming_dict, inplace=False)
    df.columns = pd.MultiIndex.from_tuples(df.columns)  # Restore column MultiIndex
    return df


def convert_data_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize time series"""
    # Sanitize time series, numeric data is needed
    # After this conversion, all columns are of float64 type, strings will be substituted
    # by NaN. This means columns that contain only strings, e.g. the columns 'date' or
    # 'filename' in the EddyPro full_output file, contain only NaNs after this step.
    # Not too problematic in case of 'date', b/c the index contains the datetime info.
    # todo For now, columns that contain only NaNs are still in the df.
    # todo at some point, the string columns should also be considered
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


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


def parse_header(filepath: Path,
                 varnames_row: int,
                 varunits_row: int = None,
                 skiprows: list = None,
                 headerrows: list = None) -> tuple[list, list, int]:
    """Read variable names and units from a csv file.

    Units are optional since they are not included in all data files.

    Example:
        In a file, if data start in row 7 and variable names are in row 2 with
        variable units in row 3 the settings are:
            skiprows=[0, 1, 4, 5, 6]
            varnames_row=0
            varnames_units=1

    Args:
        filepath: Path to the csv file.
        varnames_row: Row index of variable names after considering *skiprows*.
        varunits_row: Row index of variable units after considering *skiprows*.
        skiprows: Ignored rows at start of file.
            For example: [0, 1, 2] would ignore the first 3 rows of the file.
        headerrows: (deprecated)

    Returns:
        Lists of variable names and units, and integer of number of rows.
    """
    df = pd.read_csv(filepath,
                     # skiprows=[0, 1],
                     skiprows=skiprows,
                     # header=headerrows,
                     # index_col=False,
                     header=None,
                     nrows=2)
    varnames_list = df.loc[varnames_row].copy().tolist()
    if varunits_row is not None:
        varunits_list = df.loc[varunits_row].copy().tolist()
    else:
        varunits_list = ['-no-units-'] * len(varnames_list)
    n_cols_header = len(varnames_list)
    # num_headercols = df.columns.size
    # header_cols_list = df.columns.to_list()
    return varnames_list, varunits_list, n_cols_header


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


def flatten_multiindex_all_df_cols(df: DataFrame, keep_first_row_only: bool = False) -> DataFrame:
    if keep_first_row_only:
        # Keep only the first row (variabels) of the MultiIndex column names
        vars = [col[0] for col in df.columns]
        # # Alternative approach:
        # df_orig_cols_as_tuples = df.columns.to_flat_index()
        # vars = map(lambda x: x[0], df_orig_cols_as_tuples)  # Using map for 0 index
        # vars = list(vars)  # Convert to list
        df.columns = vars
    else:
        #  Combine first and second row of the MultiIndex column names to one line
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


def df_between_two_dates(df: DataFrame or Series, start_date, end_date, dropna_col: str = None,
                         dropna: bool = False) -> DataFrame:
    """Get data for the time window, >= start date and <= end date

    Args:
        df:
        start_date:
        end_date:
        dropna: Remove NaNs
        dropna_col: Column name in df that is checked for NaNs

    Returns:
        Data between start_date and end_date
    """
    mask = (df.index >= start_date) & (df.index <= end_date)
    snippet_df = df.loc[mask]
    if dropna:
        # Remove rows in df where dropna_col is NaN.
        filter_nan = snippet_df[dropna_col].isnull()  # True if NaN
        snippet_df = snippet_df.loc[~filter_nan]  # Keep False (=values, i.e. not NaN)
        # snippet_df = snippet_df.dropna()
    return snippet_df


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

    if priority_vars:
        for ix, col in enumerate(cols_list):
            if col[0] in priority_vars:
                cols_list.insert(0, cols_list.pop(ix))  # removes from old location ix, puts to top of list

    # Custom vars are marked w/ a dot ('.') at beginning
    for ix, col in enumerate(cols_list):
        if col[0].startswith('.'):
            cols_list.insert(0, cols_list.pop(ix))

    df = df[cols_list]  # assign new (sorted) column order

    return df


def convert_to_arrays(df: pd.DataFrame, target_col: str, complete_rows: bool = True):
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


def add_continuous_record_number(df: DataFrame) -> DataFrame:
    """Add continuous record number as new column"""
    newcol = '.RECORDNUMBER'
    data = range(1, len(df) + 1)
    df[newcol] = data
    print(f"++ Added new column {newcol} with record numbers from {df[newcol].iloc[0]} to {df[newcol].iloc[-1]}.")
    return df


def generate_flag_daynight(df: pd.DataFrame,
                           flag_based_on: str = 'timestamp',
                           ts_daytime_start_hour: int = 7,
                           ts_daytime_end_hour: int = 19,
                           col_thres_flagtrue: str = 'Larger Than Threshold',
                           col_thres_flag_threshold: float = None):
    """Add flag to indicate group membership: daytime or nighttime data"""
    df = df.copy()
    flag_col = '.FLAG_DAYTIME'
    df.loc[:, flag_col] = np.nan
    daytime_ix = None
    nighttime_ix = None

    if flag_based_on == 'timestamp':
        daytime_ix = (df.index.hour >= ts_daytime_start_hour) & \
                     (df.index.hour <= ts_daytime_end_hour)
        nighttime_ix = (df.index.hour < ts_daytime_start_hour) | \
                       (df.index.hour > ts_daytime_end_hour)
    else:
        # If *flag_based_on* is name of column
        if col_thres_flagtrue == 'Larger Than Threshold':
            daytime_ix = df[flag_based_on] > col_thres_flag_threshold
            nighttime_ix = df[flag_based_on] <= col_thres_flag_threshold
        elif col_thres_flagtrue == 'Smaller Than Threshold':
            daytime_ix = df[flag_based_on] < col_thres_flag_threshold
            nighttime_ix = df[flag_based_on] >= col_thres_flag_threshold

    df.loc[daytime_ix, [flag_col]] = 1
    df.loc[nighttime_ix, [flag_col]] = 0
    return df, flag_col


def convert_matrix_to_longform(matrixdf: pd.DataFrame):
    # Convert to long-form

    # newdf = pd.DataFrame()
    # matrixdf_unstacked = matrixdf.unstack()
    # x = matrixdf_unstacked.index.get_level_values(level=1)
    # y = matrixdf_unstacked.index.get_level_values(level=0)
    # z = matrixdf_unstacked.values
    # newdf['x'] = x
    # newdf['y'] = y
    # newdf['z'] = z

    # TODO hier weiter
    long_form_df = pd.melt(matrixdf, var_name=matrixdf.columns.name, value_name='VALUE', ignore_index=False)
    long_form_df = long_form_df.reset_index(inplace=False)

    long_form_df = long_form_df.rename(columns={'YEAR': 'YEAR'})
    long_form_df['TIMESTAMP_START'] = \
        long_form_df['YEAR'].astype(str) + '-' + long_form_df['MONTH'].astype(str) + '-' + '01'
    long_form_df['TIMESTAMP_START'] = pd.to_datetime(long_form_df['TIMESTAMP_START'])
    long_form_df = long_form_df.drop(['YEAR', 'MONTH'], axis=1)
    long_form_df = long_form_df.set_index('TIMESTAMP_START')
    long_form_df = long_form_df.sort_index()
    series_monthly = long_form_df['RANK']
    series_monthly = series_monthly.astype(int)
    series_monthly.index.freq = 'MS'


def _example_convert_matrix_to_longform():
    import diive as dv
    from diive.configs.exampledata import load_exampledata_parquet_long
    df = load_exampledata_parquet_long()
    series = df['Tair_f'].copy()
    monthly = dv.resample_to_monthly_agg_matrix(series=series, agg='mean', ranks=True)
    convert_matrix_to_longform(matrixdf=monthly)


if __name__ == "__main__":
    _example_convert_matrix_to_longform()
