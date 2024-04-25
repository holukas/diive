import pandas as pd


class TimeSince:
    upper_lim_col = "UPPER_LIMIT"
    lower_lim_col = "LOWER_LIMIT"
    flag_col = "FLAG_IS_OUTSIDE_RANGE"

    def __init__(self,
                 series: pd.Series,
                 upper_lim: float = None,
                 lower_lim: float = None,
                 include_lim: bool = True):
        self.series = series
        self.upper_lim = upper_lim
        self.lower_lim = lower_lim
        self.include_lim = include_lim

        self.timesince_col = f"TIMESINCE_{self.series.name}"
        self._timesince_df = self._setup()

    @property
    def timesince_df(self):
        """Get dataframe of merged files data"""
        if not isinstance(self._timesince_df, pd.DataFrame):
            raise Exception('data is empty')
        return self._timesince_df

    def get_timesince(self) -> pd.Series:
        return self._timesince_df[self.timesince_col].copy()

    def get_full_results(self) -> pd.DataFrame:
        return self.timesince_df.copy()

    def calc(self):
        """Detect all values that are within the specified limit range, use 0/1 to mark values."""

        # Get locations where series is within the specified limits
        if self.include_lim:
            filter_inrange = (
                    (self.timesince_df[self.series.name] <= self.timesince_df[self.upper_lim_col]) &
                    (self.timesince_df[self.series.name] >= self.timesince_df[self.lower_lim_col])
            )
        else:
            filter_inrange = (
                    (self.timesince_df[self.series.name] < self.timesince_df[self.upper_lim_col]) &
                    (self.timesince_df[self.series.name] > self.timesince_df[self.lower_lim_col])
            )

        self._timesince_df.loc[filter_inrange, self.flag_col] = 0  # Inside range
        self._timesince_df.loc[~filter_inrange, self.flag_col] = 1  # Outside range, note: this also counts NaNs as 1
        self._timesince_df[self.flag_col] = self._timesince_df[self.flag_col].astype(int)

        # print(self.timesince_df[self.timesince_df[self.flag_col] == 1].describe())

        # Set all NaN values to 1
        # OLD: Set all NaN values to 0, necessary for correct summations of values outside range
        # OLD: Otherwise, time periods with gaps would also be counted as "outside range", i.e. 1.
        self._timesince_df.loc[self._timesince_df[self.series.name].isnull(), self.flag_col] = 1

        # fantastic: https://stackoverflow.com/questions/27626542/counting-consecutive-positive-value-in-python-array
        y = self.timesince_df[self.flag_col].copy()
        yy = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

        self._timesince_df.loc[:, self.timesince_col] = yy.astype(int)

    def _setup(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df[self.series.name] = self.series.copy()

        # Upper limit
        if self.upper_lim:
            df[self.upper_lim_col] = self.upper_lim
        else:
            df[self.upper_lim_col] = self.series.max()

        # Lower limit
        if self.lower_lim:
            df[self.lower_lim_col] = self.lower_lim
        else:
            df[self.lower_lim_col] = self.series.min()

        df[self.flag_col] = pd.NA
        return df


def example_timesince():
    # Setup, user settings
    col = 'PREC_TOT_T1_25+20_1'

    # Example data
    from diive.configs.exampledata import load_exampledata_parquet
    df_orig = load_exampledata_parquet()

    # Subset
    # keep = df_orig.index.year >= 2021
    # df = df_orig[keep].copy()
    df = df_orig.copy()
    series = df[col].copy()

    # Time since
    ts = TimeSince(series, upper_lim=None, lower_lim=0, include_lim=False)
    ts.calc()

    ts_full_results = ts.get_full_results()

    from pathlib import Path
    outpath = Path(r"F:\TMP") / 'ts_full_results.csv'
    ts_full_results.to_csv(outpath, index=False)
    # ts_series = ts.get_timesince()

    # from diive.core.plotting.timeseries import TimeSeries  # For simple (interactive) time series plotting
    # TimeSeries(series=tsdata).plot()

    # # Plot
    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=observed).show()
    # HeatmapDateTime(series=gapfilled).show()
    #
    # # from diive.core.plotting.timeseries import TimeSeries  # For simple (interactive) time series plotting
    # # TimeSeries(series=df[TARGET_COL]).plot()
    #
    # print("Finished.")


if __name__ == '__main__':
    example_timesince()
