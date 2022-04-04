# import diive.pkgs.dfun
# from stats.boxes import insert_statsboxes_txt

import pandas as pd


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


class CalcTimeSeriesStats():
    """Calc stats for time series and store results in stats_df"""

    def __init__(self, series):
        self.series = series

        self.stats_df = pd.DataFrame()

        self._calc()

    def get(self):
        return self.stats_df

    def _calc(self):
        self.stats_df.loc[0, 'startdate'] = series_start(self.series)
        self.stats_df.loc[0, 'enddate'] = series_end(self.series)
        self.stats_df.loc[0, 'period'] = series_duration(self.series)
        self.stats_df.loc[0, 'nov'] = series_numvals(self.series)
        self.stats_df.loc[0, 'dtype'] = self.series.dtypes
        self.stats_df.loc[0, 'missing'] = series_numvals_missing(self.series)
        self.stats_df.loc[0, 'missing_perc'] = series_perc_missing(self.series)
        self.stats_df.loc[0, 'mean'] = self.series.mean()
        self.stats_df.loc[0, 'sd'] = self.series.std()
        self.stats_df.loc[0, 'sd/mean'] = series_sd_over_mean(self.series)
        self.stats_df.loc[0, 'median'] = self.series.quantile(q=0.50)
        self.stats_df.loc[0, 'max'] = self.series.max()
        self.stats_df.loc[0, 'min'] = self.series.min()
        self.stats_df.loc[0, 'mad'] = self.series.mad()
        self.stats_df.loc[0, 'cumsum'] = self.series.cumsum().iloc[-1]
        self.stats_df.loc[0, 'p95'] = self.series.quantile(q=0.95)
        self.stats_df.loc[0, 'p75'] = self.series.quantile(q=0.75)
        self.stats_df.loc[0, 'p25'] = self.series.quantile(q=0.25)
        self.stats_df.loc[0, 'p05'] = self.series.quantile(q=0.05)
