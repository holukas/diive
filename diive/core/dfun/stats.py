# import diive.pkgs.dfun
# from stats.boxes import insert_statsboxes_txt
import pandas as pd
from pandas import Series, DataFrame

from diive.core.funcs.funcs import zscore


def q75(x):
    return x.quantile(0.75)


def q50(x):
    return x.quantile(0.5)


def q25(x):
    return x.quantile(0.25)


def q99(x):
    return x.quantile(0.99)


def q01(x):
    return x.quantile(0.01)


def q05(x):
    return x.quantile(0.05)


def q95(x):
    return x.quantile(0.95)


def series_start(series: pd.Series, dtformat: str = "%Y-%m-%d %H:%M"):
    """Return start datetime of series"""
    return series.index[0].strftime(dtformat)


def series_end(series: pd.Series, dtformat: str = "%Y-%m-%d %H:%M"):
    """Return end datetime of series"""
    return series.index[-1].strftime(dtformat)


def series_duration(series: pd.Series):
    """Return duration of series"""
    return series.index[-1] - series.index[0]


def series_numvals(series: pd.Series):
    """Return number of values in series"""
    return series.count()


def series_numvals_missing(series: pd.Series):
    """Return number of missing values in series"""
    return series.isnull().sum()


def series_perc_missing(series: pd.Series):
    """Return number of missing values in series as percentage"""
    return (series_numvals_missing(series) / len(series.index)) * 100


def series_sd_over_mean(series: pd.Series):
    """Return sd / mean"""
    return series.std() / series.mean()


def sstats(s: Series) -> DataFrame:
    """
    Calculate stats for time series and store results in dataframe

    - Example notebook available in:
        notebooks/Stats/TimeSeriesStats.ipynb

    """
    col = s.name
    df = pd.DataFrame(columns=[col])
    df.loc['STARTDATE', col] = series_start(s)
    df.loc['ENDDATE', col] = series_end(s)
    df.loc['PERIOD', col] = series_duration(s)
    df.loc['NOV', col] = series_numvals(s)
    df.loc['MISSING', col] = series_numvals_missing(s)
    df.loc['MISSING_PERC', col] = series_perc_missing(s)
    df.loc['MEAN', col] = s.mean()
    df.loc['MEDIAN', col] = s.quantile(q=0.50)
    df.loc['SD', col] = s.std()
    df.loc['VAR', col] = s.var()
    df.loc['SD/MEAN'] = series_sd_over_mean(s)
    # df.loc['MAD', col] = s.mad()  # deprecated in pandas
    # df.loc['CUMSUM_MIN', col] = s.cummin().iloc[-1]
    # df.loc['CUMSUM_MAX', col] = s.cummax().iloc[-1]
    df.loc['SUM', col] = s.sum()
    df.loc['MIN', col] = s.min()
    df.loc['MAX', col] = s.max()
    df.loc['P01', col] = s.quantile(q=0.01)
    df.loc['P05', col] = s.quantile(q=0.05)
    df.loc['P25', col] = s.quantile(q=0.25)
    df.loc['P75', col] = s.quantile(q=0.75)
    df.loc['P95', col] = s.quantile(q=0.95)
    df.loc['P99', col] = s.quantile(q=0.99)
    return df


def sstats_doublediff_abs(s: Series) -> DataFrame:
    """Calculate stats for absolute double difference of series."""
    doublediff_abs, diff_to_prev_abs, diff_to_next_abs = double_diff_absolute(s=s)
    df = sstats(s=doublediff_abs)
    return df


def sstats_zscore(s: Series) -> DataFrame:
    """Calculate stats for z-scores of series."""
    z = zscore(series=s)
    df = sstats(s=z)
    return df


def double_diff_absolute(s: Series) -> tuple[Series, Series, Series]:
    """Calculate the absolute sum of differences between a data point and
    the respective preceding and next value."""
    shifted_prev = s.shift(1)
    diff_to_prev = s - shifted_prev
    diff_to_prev_abs = diff_to_prev.abs()
    shifted_next = s.shift(-1)
    diff_to_next = s - shifted_next
    diff_to_next_abs = diff_to_next.abs()
    doublediff_abs = diff_to_prev_abs + diff_to_next_abs
    # dd_abs = dd_abs ** 2
    doublediff_abs.name = 'DOUBLE_DIFF_ABS'
    return doublediff_abs, diff_to_prev_abs, diff_to_next_abs


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    series = df['NEE_CUT_REF_f'].copy()
    stats = sstats_doublediff_abs(series)
    print(stats)


if __name__ == '__main__':
    example()
