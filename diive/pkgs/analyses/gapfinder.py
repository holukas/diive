import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series


class GapFinder:
    """Find gaps in time series.

    Detects consecutive missing values (NaN) and reports their locations,
    duration, and statistics. Useful for understanding data quality and
    planning gap-filling strategies.

    Example:
        See `examples/analyses/gapfinder.py` for complete examples.
    """
    # Define column names
    gap_values = 'GAPVALUES'
    isgap_col = 'IS_GAP'  # flag that shows where the gaps are
    isnumeric_col = 'IS_NUMERIC'  # flag that shows where the numeric values are
    isnumeric_cumsum_col = 'IS_NUMERIC_CUMSUM'  # for detecting consecutive gaps
    gaplen_below_limit_col = 'GAPLEN_BELOW_LIM'

    def __init__(self,
                 series: Series,
                 limit: int = None,
                 sort_results: bool = True):
        """
        Args:
            series: Time series
            limit: Only consider gaps <= limit
            sort_results: Sort results by gap length, descending
        """
        self.limit = limit  # number of allowed consecutive gaps
        self.series_col = series.name
        self.sort_results = sort_results

        self.gapfinder_fullres_df = pd.DataFrame(series)

        self._make_required_cols()
        self.gapfinder_df = self._detect_gaps()

    def _make_required_cols(self):
        # Init new columns
        self.gapfinder_fullres_df[self.gapfinder_fullres_df.index.name] = \
            self.gapfinder_fullres_df.index  # needed for filling gap values
        self.gapfinder_fullres_df[self.gap_values] = np.nan
        self.gapfinder_fullres_df[self.isgap_col] = np.nan
        self.gapfinder_fullres_df[self.isnumeric_col] = np.nan
        self.gapfinder_fullres_df[self.gaplen_below_limit_col] = np.nan

        # The cumulative sum is central to the detection of consecutive gaps.
        # Since values are flagged with 1 and gaps with 0, the cumsum does not
        # change after the last value before the gap. Consecutive gaps will
        # therefore have the same cumsum. This can be used later by grouping
        # data on basis of cumsum.
        self.gapfinder_fullres_df[self.isnumeric_cumsum_col] = np.nan

    def _detect_gaps(self):
        """Detect consecutive gaps in measured"""
        # kudos: https://stackoverflow.com/questions/29007830/identifying-consecutive-nans-with-pandas

        # Fill flags into new columns
        # NaNs, 1 = True = NaN:
        self.gapfinder_fullres_df[self.isgap_col] = self.gapfinder_fullres_df[self.series_col].isnull().astype(int)
        # 1 = True = number:
        self.gapfinder_fullres_df[self.isnumeric_col] = self.gapfinder_fullres_df[self.series_col].notnull().astype(int)
        self.gapfinder_fullres_df[self.isnumeric_cumsum_col] = self.gapfinder_fullres_df[self.isnumeric_col].cumsum()

        # Narrow down the df to contain only the gap data
        filter_isgap = self.gapfinder_fullres_df[self.isgap_col] == 1
        self.gapfinder_fullres_df = self.gapfinder_fullres_df[filter_isgap]
        datetime_col = self.gapfinder_fullres_df.index.name
        self.gapfinder_fullres_df[datetime_col] = self.gapfinder_fullres_df.index  # needed for aggregate

        # Aggregate the df, gives stats about number of consecutive gaps
        # and their start and end datetime.
        gapfinder_df = \
            self.gapfinder_fullres_df.groupby(self.isnumeric_cumsum_col).aggregate({
                self.gapfinder_fullres_df.index.name: ['min', 'max'],
                self.isnumeric_cumsum_col: ['count']
            })
        # TODO Fehler wenn keine gaps mehr gibt

        # Remove the original column names (tuples w/ two elements)
        gapfinder_df.columns = gapfinder_df.columns.droplevel([0])

        if self.limit:
            gapfinder_df = self._apply_limit(df=gapfinder_df)

        gapfinder_df = self._rename_results(df=gapfinder_df)

        if self.sort_results:
            gapfinder_df = gapfinder_df.sort_values(by='GAP_LENGTH', ascending=False)

        return gapfinder_df

    def _rename_results(self, df: DataFrame) -> DataFrame:
        """Rename results"""
        renaming_dict = {'min': 'GAP_START',
                         'max': 'GAP_END',
                         'count': 'GAP_LENGTH'}
        df = df.rename(columns=renaming_dict)
        return df

    def _apply_limit(self, df: DataFrame) -> DataFrame:
        """Consider gaps smaller or equal to *limit*"""
        # Make flag that show if gap is above or below limit
        df[self.gaplen_below_limit_col] = np.nan
        filter_limit = df['count'] <= self.limit
        df.loc[filter_limit, self.gaplen_below_limit_col] = 1  # True, below or equal limit
        filter_limit = df['count'] > self.limit
        df.loc[filter_limit, self.gaplen_below_limit_col] = 0  # False, above limit

        # Get all gap positions below the limit
        filter_above_lim = df[self.gaplen_below_limit_col] == 1
        df = df[filter_above_lim]
        return df

    @property
    def results(self) -> DataFrame:
        """Gap detection results as DataFrame.

        Columns: GAP_START, GAP_END, GAP_LENGTH
        Sorted by gap length (descending) if sort_results=True.
        """
        return self.gapfinder_df

    def get_results(self) -> DataFrame:
        """Alias for .results property."""
        return self.results
