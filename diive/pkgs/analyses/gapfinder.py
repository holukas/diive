import numpy as np
import pandas as pd
from pandas import Series


class GapFinder:
    """Find gaps in Series

    Results are collected in a dataframe that info about gaps
    locations within the limit.
    """
    # Define column names
    gap_values = 'gap_values'
    isgap_col = 'isgap'  # flag that shows where the gaps are
    isval_col = 'isvalue'  # flag that shows where the numeric values are
    isval_cumsum_col = 'isvalue_cumsum'  # for detecting consecutive gaps
    gaplen_below_limit_col = 'gaplen_below_lim'

    def __init__(self, series: Series, limit: int = None):
        self.limit = limit  # number of allowed consecutive gaps
        self.series_col = series.name
        self.gapfinder_fullres_df = pd.DataFrame(series)
        self.make_required_cols()
        self.gapfinder_agg_df = self.check_gaps()

    def make_required_cols(self):
        # Init new columns
        self.gapfinder_fullres_df[self.gapfinder_fullres_df.index.name] = \
            self.gapfinder_fullres_df.index  # needed for filling gap values
        self.gapfinder_fullres_df[self.gap_values] = np.nan
        self.gapfinder_fullres_df[self.isgap_col] = np.nan
        self.gapfinder_fullres_df[self.isval_col] = np.nan
        self.gapfinder_fullres_df[self.gaplen_below_limit_col] = np.nan

        # The cumulative sum is central to the detection of consecutive gaps.
        # Since values are flagged with 1 and gaps with 0, the cumsum does not
        # change after the last value before the gap. Consecutive gaps will
        # therefore have the same cumsum. This can be used later by grouping
        # data on basis of cumsum.
        self.gapfinder_fullres_df[self.isval_cumsum_col] = np.nan

    def check_gaps(self):
        # Detect consecutive gaps in measured
        # kudos: https://stackoverflow.com/questions/29007830/identifying-consecutive-nans-with-pandas

        # Fill flags into new columns
        # NaNs, 1 = True = NaN:
        self.gapfinder_fullres_df[self.isgap_col] = self.gapfinder_fullres_df[self.series_col].isnull().astype(int)
        # 1 = True = number:
        self.gapfinder_fullres_df[self.isval_col] = self.gapfinder_fullres_df[self.series_col].notnull().astype(int)
        self.gapfinder_fullres_df[self.isval_cumsum_col] = self.gapfinder_fullres_df[self.isval_col].cumsum()

        # Narrow down the df to contain only the gap data
        filter_isgap = self.gapfinder_fullres_df[self.isgap_col] == 1
        self.gapfinder_fullres_df = self.gapfinder_fullres_df[filter_isgap]
        datetime_col = self.gapfinder_fullres_df.index.name
        self.gapfinder_fullres_df[datetime_col] = self.gapfinder_fullres_df.index  # needed for aggregate

        # Aggregate the df, gives stats about number of consecutive gaps
        # and their start and end datetime.
        gapfinder_agg_df = \
            self.gapfinder_fullres_df.groupby(self.isval_cumsum_col).aggregate({
                self.gapfinder_fullres_df.index.name: ['min', 'max'],
                self.isval_cumsum_col: ['count']
            })
        # TODO Fehler wenn keine gaps mehr gibt

        # Remove the original column names (tuples w/ two elements)
        gapfinder_agg_df.columns = gapfinder_agg_df.columns.droplevel([0])

        # # Put first_spotted column as index, needed for filtering
        # gap_groups_overview_df.index = gap_groups_overview_df[self.gap_datetime_col]

        if self.limit:
            # Make flag that show if gap is above or below limit
            gapfinder_agg_df[self.gaplen_below_limit_col] = np.nan
            filter_limit = gapfinder_agg_df['count'] <= self.limit
            gapfinder_agg_df.loc[filter_limit, self.gaplen_below_limit_col] = 1  # True, below or equal limit
            filter_limit = gapfinder_agg_df['count'] > self.limit
            gapfinder_agg_df.loc[filter_limit, self.gaplen_below_limit_col] = 0  # False, above limit

            # Get all gap positions below the limit
            filter_above_lim = gapfinder_agg_df[self.gaplen_below_limit_col] == 1
            gapfinder_agg_df = gapfinder_agg_df[filter_above_lim]

        return gapfinder_agg_df

    def get_results(self):
        return self.gapfinder_agg_df
