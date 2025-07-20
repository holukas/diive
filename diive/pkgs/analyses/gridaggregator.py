from typing import Literal

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


class GridAggregator:
    """
    Calculates aggregated values of a target variable (z) within a 2D grid
    defined by quantiles of two independent variables (x and y).

    This class bins the 'x' and 'y' Series into a specified number of quantile
    classes, forming a grid. For each cell in this grid, it computes an
    aggregate (e.g., mean, sum, count) of the 'z' Series.

    By default, 'x' and 'y' are divided into 10 quantile classes each,
    resulting in a 10x10 (100) grid. The default aggregation for 'z' is the mean.
    Bins with fewer than `min_n_vals_per_bin` observations are excluded from
    the results.

    The primary outputs are a pivoted DataFrame (matrix) representing the grid
    and its corresponding long-form (tidy) DataFrame.

    Attributes:
        x (pd.Series): The input Series for the first independent variable.
        y (pd.Series): The input Series for the second independent variable.
        z (pd.Series): The input Series for the dependent variable to aggregate.
        n_bins (int): The number of quantile bins to create for both x and y.
        min_n_vals_per_bin (int): The minimum number of data points required
                                  for a combined x-y bin to be included in the results.
                                  Bins with fewer than `min_n_vals_per_bin` observations
                                  will be excluded.
        binagg_z (Literal): The aggregation function to apply to 'z' within each bin.
                            Allowed values are 'mean', 'min', 'max', 'median', 'count'.

    Properties:
        pivotdf (pd.DataFrame): A pivoted DataFrame where rows are y-bins,
                                columns are x-bins, and values are the aggregated z.
                                Accessible after calling `run()`.
        longformdf (pd.DataFrame): A long-form DataFrame containing the original data
                                   with added x and y bin assignments, filtered by
                                   `min_n_vals_per_bin`. Accessible after calling `run()`.
    """

    def __init__(self,
                 x: Series,
                 y: Series,
                 z: Series,
                 quantiles: bool = False,
                 n_bins: int = 10,
                 min_n_vals_per_bin: int = 1,
                 binagg_z: Literal['mean', 'min', 'max', 'median', 'count', 'sum'] = 'mean'
                 ):
        """
        Initialize QuantileGridAggregator

            Args:
                x (pd.Series): The pandas Series representing the x-axis variable.
                               Its name will be used for labeling.
                y (pd.Series): The pandas Series representing the y-axis variable.
                               Its name will be used for labeling.
                z (pd.Series): The pandas Series representing the z-axis variable
                               (values to be aggregated). Its name will be used for labeling.
                n_bins (int, optional): The number of quantile bins to divide
                                             both the 'x' and 'y' data into.
                                             Defaults to 10.
                min_n_vals_per_bin (int, optional): The minimum number of observations
                                                    required in a combined (x, y) bin
                                                    for it to be considered valid and
                                                    included in the output. Bins with
                                                    fewer observations will be excluded.
                                                    Defaults to 1.
                binagg_z (Literal['mean', 'min', 'max', 'median', 'count'], optional):
                          The aggregation function to apply to the 'z' values within
                          each (x, y) quantile bin. Defaults to 'mean'.
                          'count' uses `np.count_nonzero`.
                """
        self.x = x
        self.y = y
        self.z = z
        self.xname = x.name if x.name is not None else 'x'
        self.yname = y.name if y.name is not None else 'y'
        self.zname = z.name if z.name is not None else 'z'
        self.quantiles = quantiles
        self.n_bins = n_bins
        self.min_n_vals_per_bin = min_n_vals_per_bin

        self.xbinname = f'BIN_{self.xname}'
        self.ybinname = f'BIN_{self.yname}'

        if binagg_z == 'sum':
            self.aggfunc = 'sum'
        elif binagg_z == 'mean':
            self.aggfunc = 'mean'
        elif binagg_z == 'min':
            self.aggfunc = 'min'
        elif binagg_z == 'max':
            self.aggfunc = 'max'
        elif binagg_z == 'median':
            self.aggfunc = 'median'
        elif binagg_z == 'count':
            self.aggfunc = np.count_nonzero

        self._df_wide = None
        self._df_long = None

    def run(self):
        # Create dataframe from input data
        plot_df = pd.concat([self.x, self.y, self.z], axis=1)

        # Remove duplicates, in case e.g. x=y
        plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]
        plot_df.index.name = 'DATE'

        self._df_wide, self._df_long = self._transform_data(df=plot_df)

    @property
    def df_wide(self) -> DataFrame:
        if not isinstance(self._df_wide, DataFrame):
            raise Exception(f'pivotdf is not available, use .run() first.')
        return self._df_wide

    @property
    def df_long(self) -> DataFrame:
        if not isinstance(self._df_long, DataFrame):
            raise Exception(f'longformdf is not available, use .run() first.')
        return self._df_long

    def _transform_data(self, df) -> tuple:
        """Transform data for plotting"""

        # Setup xy axes
        df_long = df.reset_index(drop=False).copy()

        # Bin x and y data
        df_long[self.xbinname] = self._assign_bins(series=df_long[self.xname])
        df_long[self.ybinname] = self._assign_bins(series=df_long[self.yname])
        # df_long = self._assign_xybins(df=df_long)

        # Add new column for unique bin combinations
        df_long['BINS_COMBINED_STR'] = df_long[self.xbinname].astype(str) + "+" + df_long[
            self.ybinname].astype(str)

        # Adding bins together makes only sense when x and y are quantiles
        if self.quantiles:
            df_long['BINS_COMBINED_INT'] = df_long[self.xbinname].add(df_long[self.ybinname])

        # Count number of available values per combined bin
        ok_bin = df_long.groupby('BINS_COMBINED_STR').count()['DATE'] >= self.min_n_vals_per_bin
        # ok_bin = longformdf.groupby('BIN_Tair_f_max').count()['DATE'] > self.min_n_vals_per_bin

        # Bins with enough values
        ok_bin = ok_bin[ok_bin]  # Filter (True/False) indicating if bin has enough values
        ok_binlist = list(ok_bin.index)  # List of bins with enough values

        # Keep data rows where the respective bin has enough values
        ok_row = df_long['BINS_COMBINED_STR'].isin(ok_binlist)
        df_long = df_long[ok_row]

        _counts = pd.pivot_table(df_long, index=self.ybinname, columns=self.xbinname, values=self.zname,
                                 aggfunc=np.count_nonzero)

        df_wide = pd.pivot_table(df_long, index=self.ybinname, columns=self.xbinname, values=self.zname,
                                 aggfunc=self.aggfunc)

        # # Reset the index to make 'BIN_VPD_f_mean' a column
        # df_reset = df_wide.reset_index()

        # Melt the DataFrame to long format:
        # - id_vars: The column(s) to remain as identifier variables. Here, it's our index after resetting.
        # - var_name: The name for the new column that will hold the original column names (e.g., '0', '25').
        # - value_name: The name for the new column that will hold the values from the melted columns.
        df_long = df_wide.reset_index().melt(id_vars=[self.ybinname], var_name=self.xbinname, value_name=self.zname)

        return df_wide, df_long

    def _assign_bins(self, series: pd.Series) -> pd.Series:
        # --- Dynamic Label Generation for X-axis ---
        # Step 1: Run qcut once WITHOUT labels to determine the actual number of bins
        if self.quantiles:
            temp_group_x, temp_bins_x = pd.qcut(
                series,
                q=self.n_bins,
                retbins=True,
                duplicates='drop')
        else:
            temp_group_x, temp_bins_x = pd.cut(
                series,
                bins=self.n_bins,
                retbins=True,
                duplicates='drop')

        # Number of bins is (num_edges - 1)
        actual_n_bins_x = len(temp_bins_x) - 1

        # Step 3: Run qcut again WITH the correctly sized labels
        if self.quantiles:

            # Step 2: Generate labels based on the actual number of bins
            # This ensures 'labels' count matches the actual bins created
            labels_stepsize_x = int(100 / actual_n_bins_x) if actual_n_bins_x > 0 else 0
            labels_x = list(range(0, actual_n_bins_x * labels_stepsize_x, labels_stepsize_x))
            if len(labels_x) != actual_n_bins_x:  # Adjust for potential rounding issues
                labels_x = list(range(0, 100, int(100 / actual_n_bins_x)))[:actual_n_bins_x]

            group_x, bins_x = pd.qcut(
                series,
                q=self.n_bins,
                labels=labels_x,
                retbins=True,
                duplicates='drop')
        else:
            labels_x = temp_bins_x[:actual_n_bins_x]
            group_x, bins_x = pd.cut(
                series,
                bins=self.n_bins,
                labels=labels_x,
                retbins=True,
                duplicates='drop')

        series_bins = group_x.astype(int)
        return series_bins


    # def _assign_xybins(self, df: DataFrame) -> DataFrame:
    #     """Create bins for x and y data"""
    #     df[self.xbinname] = self._assign_xbins(series=df[self.xname])
    #     df[self.ybinname] = self._assign_xbins(series=df[self.yname])
    #
    #     # # --- Dynamic Label Generation for X-axis ---
    #     # # Step 1: Run qcut once WITHOUT labels to determine the actual number of bins
    #     # if self.quantiles:
    #     #     temp_group_x, temp_bins_x = pd.qcut(
    #     #         df[self.xname],
    #     #         q=self.n_bins,
    #     #         retbins=True,
    #     #         duplicates='drop')
    #     # else:
    #     #     temp_group_x, temp_bins_x = pd.cut(
    #     #         df[self.xname],
    #     #         bins=self.n_bins,
    #     #         retbins=True,
    #     #         duplicates='drop')
    #     #
    #     # # Number of bins is (num_edges - 1)
    #     # actual_n_bins_x = len(temp_bins_x) - 1
    #     #
    #     #
    #     # # Step 3: Run qcut again WITH the correctly sized labels
    #     # if self.quantiles:
    #     #
    #     #     # Step 2: Generate labels based on the actual number of bins
    #     #     # This ensures 'labels' count matches the actual bins created
    #     #     labels_stepsize_x = int(100 / actual_n_bins_x) if actual_n_bins_x > 0 else 0
    #     #     labels_x = list(range(0, actual_n_bins_x * labels_stepsize_x, labels_stepsize_x))
    #     #     if len(labels_x) != actual_n_bins_x:  # Adjust for potential rounding issues
    #     #         labels_x = list(range(0, 100, int(100 / actual_n_bins_x)))[:actual_n_bins_x]
    #     #
    #     #     group_x, bins_x = pd.qcut(
    #     #         df[self.xname],
    #     #         q=self.n_bins,
    #     #         labels=labels_x,
    #     #         retbins=True,
    #     #         duplicates='drop')
    #     # else:
    #     #     labels_x = temp_bins_x[:actual_n_bins_x]
    #     #     group_x, bins_x = pd.cut(
    #     #         df[self.xname],
    #     #         bins=self.n_bins,
    #     #         labels=labels_x,
    #     #         retbins=True,
    #     #         duplicates='drop')
    #     #
    #     # df[self.xbinname] = group_x.astype(int)
    #
    #
    #
    #
    #     # # --- Dynamic Label Generation for Y-axis ---
    #     # # Step 1: Run qcut once WITHOUT labels to determine the actual number of bins
    #     # temp_group_y, temp_bins_y = pd.qcut(df[self.yname],
    #     #                                     q=self.n_bins,
    #     #                                     retbins=True,
    #     #                                     duplicates='drop'
    #     #                                     )
    #     # actual_n_bins_y = len(temp_bins_y) - 1
    #     #
    #     # # Step 2: Generate labels based on the actual number of bins
    #     # labels_stepsize_y = int(100 / actual_n_bins_y) if actual_n_bins_y > 0 else 0
    #     # labels_y = list(range(0, actual_n_bins_y * labels_stepsize_y, labels_stepsize_y))
    #     # if len(labels_y) != actual_n_bins_y:  # Adjust for potential rounding issues
    #     #     labels_y = list(range(0, 100, int(100 / actual_n_bins_y)))[:actual_n_bins_y]
    #     #
    #     # # Step 3: Run qcut again WITH the correctly sized labels
    #     # group_y, bins_y = pd.qcut(df[self.yname],
    #     #                           q=self.n_bins,
    #     #                           labels=labels_y,
    #     #                           retbins=True,
    #     #                           duplicates='drop'
    #     #                           )
    #     # df[self.ybinname] = group_y.astype(int)
    #
    #     return df


def _example():
    import diive as dv

    # Load example data
    df = dv.load_exampledata_parquet()

    # Make subset of three required columns
    vpd_col = 'VPD_f'  # Vapor pressure deficit
    ta_col = 'Tair_f'  # Air temperature
    swin_col = 'Rg_f'  # Shortwave incoming radiation
    subset = df[[vpd_col, ta_col, swin_col]].copy()
    subset = subset.loc[(subset.index.month >= 5) & (subset.index.month <= 9)].copy()  # Use data May and Sep
    daytime_locs = (subset[swin_col] > 0)  # Use daytime data
    subset = subset[daytime_locs].copy()
    subset = subset.dropna()

    q = dv.ga(
        x=subset[swin_col],
        y=subset[ta_col],
        z=subset[vpd_col],
        quantiles=False,
        n_bins=10,
        min_n_vals_per_bin=3,
        binagg_z='mean'
    )
    q.run()

    pivotdf = q.df_wide.copy()
    print(pivotdf)
    # print(q.df_long)

    # hm = dv.heatmapxyz(pivotdf=pivotdf)
    # hm.plot(cb_digits_after_comma=0,
    #         xlabel=r'Short-wave incoming radiation ($\mathrm{percentile}$)',
    #         ylabel=r'Air temperature ($\mathrm{percentile}$)',
    #         # ylabel=r'Percentile of TA ($\mathrm{°C}$)',
    #         zlabel=r'Vapor pressure deficit (counts)',
    #         # zlabel=r'Vapor pressure deficit ($\mathrm{gCO_{2}\ m^{-2}\ d^{-1}}$)',
    #         tickpos=[16, 25, 50, 75, 84],
    #         ticklabels=['16', '25', '50', '75', '84']
    #
    #         )

    # print(res)


if __name__ == '__main__':
    _example()
