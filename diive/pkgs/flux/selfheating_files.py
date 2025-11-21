import datetime as dt

import pandas
import pandas as pd


def read(src, nrows):
    """Read source file to dataframe"""
    date_parser = lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')  # For DIIVE csv files
    df = pd.read_csv(src, header=[0, 1], delimiter=',', date_parser=date_parser, na_values=-9999, index_col=0,
                     parse_dates=[0], nrows=nrows)
    df.sort_index(axis=1, inplace=True)  # lexsorted
    summary(df)
    return df


def summary(df:pandas.DataFrame):
    """Generate descriptive stats"""
    for col_ix, col in enumerate(df.columns):
        print(f"column#{col_ix}: {col[0]}    {col[1]}")
    print(f"{'=' * 20}\n{len(df.columns)} columns in total")
    print(f"{len(df)} data rows in total")
    df.describe()
