"""
Grid Aggregation Module

This module provides the `GridAggregator` class, a tool for
transforming and aggregating tabular data into a 2D grid structure. It
supports flexible binning strategies and various aggregation methods,
facilitating data analysis and visualization in grid-based formats.

Author: Lukas Hörtnagl
License: GPL-3.0
"""

from typing import Literal, Callable, Union

import numpy as np
import pandas as pd


class GridAggregator:

    def __init__(self,
                 x: pd.Series,
                 y: pd.Series,
                 z: pd.Series,
                 binning_type: Literal['quantiles', 'equal_width'],
                 n_bins: int = 10,
                 min_n_vals_per_bin: int = 1,
                 aggfunc: Union[Literal['mean', 'min', 'max', 'median', 'sum'], Callable] = 'mean'
                 ):

        # Input validation
        if not all(isinstance(arg, pd.Series) for arg in [x, y, z]):
            raise TypeError("x, y, and z must be pandas Series.")
        if not all(len(arg) == len(x) for arg in [y, z]):
            raise ValueError("x, y, and z Series must have the same length.")
        if n_bins <= 0:
            raise ValueError("n_bins must be a positive integer.")
        if min_n_vals_per_bin <= 0:
            raise ValueError("min_n_vals_per_bin must be a positive integer.")
        if binning_type not in ['quantiles', 'equal_width']:
            raise ValueError("binning_type must be 'quantiles' or 'equal_width'.")

        # Initialize attributes
        self.x = x.copy()
        self.y = y.copy()
        self.z = z.copy()
        self.binning_type = binning_type
        self.n_bins = n_bins
        self.min_n_vals_per_bin = min_n_vals_per_bin

        self.xname = x.name if x.name is not None else 'x_data'
        self.yname = y.name if y.name is not None else 'y_data'
        self.zname = z.name if z.name is not None else 'z_data'

        self.xbinname = f'BIN_{self.xname}'
        self.ybinname = f'BIN_{self.yname}'

        # Map 'count' to np.count_nonzero, otherwise use the provided aggfunc
        self.aggfunc = np.count_nonzero if aggfunc == 'count' else aggfunc

        # Prepare dataframe
        # Ensure column names are unique if Series names might overlap
        self.df = pd.DataFrame({
            self.xname: self.x,
            self.yname: self.y,
            self.zname: self.z
        })
        self.df.index.name = 'INDEX'
        self.df = self.df.reset_index(drop=False).copy()

        self._df_agg_wide = None
        self._df_agg_long = None

        # Perform binning and aggregation on initialization
        self._bin_and_aggregate()

    @property
    def df_agg_wide(self) -> pd.DataFrame:
        if self._df_agg_wide is None:
            raise AttributeError("Aggregated wide DataFrame is not available. Ensure binning was successful.")
        return self._df_agg_wide

    @property
    def df_agg_long(self) -> pd.DataFrame:
        if self._df_agg_long is None:
            raise AttributeError("Aggregated long DataFrame is not available. Ensure binning was successful.")
        return self._df_agg_long

    def _bin_and_aggregate(self):
        if self.binning_type == 'quantiles':
            self._apply_quantile_binning()
        elif self.binning_type == 'equal_width':
            self._apply_equal_width_binning()

        # After binning, proceed to transform and aggregate
        self._transform_and_pivot(is_quantiles=(self.binning_type == 'quantiles'))

    def _apply_quantile_binning(self):
        self.df[self.xbinname] = self._assign_bins_quantiles(series=self.df[self.xname])
        self.df[self.ybinname] = self._assign_bins_quantiles(series=self.df[self.yname])

    def _apply_equal_width_binning(self):
        self.df[self.xbinname] = self._assign_bins_equal_widh(series=self.df[self.xname])
        self.df[self.ybinname] = self._assign_bins_equal_widh(series=self.df[self.yname])


    def _assign_bins_quantiles(self, series: pd.Series) -> pd.Series:
        """
        Assigns bins to a Series using quantile-based binning, with labels ranging from 0 to 100
        representing the lower bound of each quantile.
        """
        try:
            # Determine the bin edges first, allowing duplicates to be dropped if necessary
            # This gives us the actual number of bins created by pandas.qcut
            _, bins = pd.qcut(series, q=self.n_bins, retbins=True, duplicates='drop')

            actual_n_bins = len(bins) - 1

            if actual_n_bins == 0:
                # Handle cases where no bins can be formed (e.g., all series values are identical)
                print(
                    f"Warning: Could not form any quantile bins for series '{series.name}'. All values are identical or n_bins is too high relative to unique values.")
                return pd.Series(np.nan, index=series.index, name=f'BIN_{series.name}')

            # Calculate a step size for labels to distribute them evenly between 0 and 100
            # For example, if actual_n_bins is 4, labels would be [0, 25, 50, 75]
            labels = [int(i * (100 / actual_n_bins)) for i in range(actual_n_bins)]

            # Apply qcut with the generated percentile-based labels
            # We re-run qcut with the determined labels
            binned_series = pd.qcut(series, q=self.n_bins, labels=labels, retbins=False, duplicates='drop')

            # Convert the categorical result to integer labels
            return binned_series.astype(int)

        except ValueError as e:
            # Catch specific ValueError for qcut if quantiles are not unique
            print(
                f"Warning: Could not apply quantile binning to series '{series.name}'. Error: {e}. This might happen if there are too many duplicate values preventing the creation of distinct quantiles.")
            return pd.Series(np.nan, index=series.index, name=f'BIN_{series.name}')
        except Exception as e:
            # Catch any other unexpected errors during binning
            print(f"An unexpected error occurred during quantile binning for series '{series.name}': {e}")
            return pd.Series(np.nan, index=series.index, name=f'BIN_{series.name}')

    # def _assign_bins_quantiles(self, series: pd.Series):
    #     # Run qcut once WITHOUT labels to determine the actual number of bins
    #     _, temp_bins = pd.qcut(series, q=self.n_bins, retbins=True, duplicates='drop')
    #     actual_n_bins = len(temp_bins) - 1  # Number of bins is (num_edges - 1)
    #     # Generate labels based on the actual number of bins
    #     # This ensures 'labels' count matches the actual bins created
    #     labels_stepsize = int(100 / actual_n_bins) if actual_n_bins > 0 else 0
    #     labels = list(range(0, actual_n_bins * labels_stepsize, labels_stepsize))
    #     if len(labels) != actual_n_bins:  # Adjust for potential rounding issues
    #         labels = list(range(0, 100, int(100 / actual_n_bins)))[:actual_n_bins]
    #     group_x, bins_x = pd.qcut(series, q=self.n_bins, labels=labels, retbins=True, duplicates='drop')
    #     series_bins = group_x.astype(int)
    #     return series_bins

    def _assign_bins_equal_widh(self, series: pd.Series):
        # Run cut once WITHOUT labels to determine the actual number of bins
        _, temp_bins_x = pd.cut(series, bins=self.n_bins, retbins=True, duplicates='drop')
        actual_n_bins_x = len(temp_bins_x) - 1  # Number of bins is (num_edges - 1)
        labels_x = temp_bins_x[:actual_n_bins_x]
        group_x, bins_x = pd.cut(series, bins=self.n_bins, labels=labels_x, retbins=True, duplicates='drop')
        series_bins = group_x.astype(int)
        return series_bins

    def _transform_and_pivot(self, is_quantiles: bool) -> None:
        """Transform data for plotting"""
        # Drop rows where binning failed (NaNs in bin columns)
        initial_rows = len(self.df)
        # Using .loc for direct modification in place and better practice
        self.df = self.df.dropna(subset=[self.xbinname, self.ybinname]).copy()
        if len(self.df) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(self.df)} rows due to failed binning (NaNs in bin columns).")

        if self.df.empty:
            print("Warning: No data left after dropping NaNs in bin columns. Aggregated DataFrames will be empty.")
            self._df_agg_wide = pd.DataFrame()
            self._df_agg_long = pd.DataFrame()
            return

        # Add new column for unique bin combinations
        # Using .astype(str) for robustness if bin labels are not purely numeric
        self.df['BINS_COMBINED_STR'] = self.df[self.xbinname].astype(str) + "+" + self.df[self.ybinname].astype(str)

        # Add combined integer bin for quantile analysis if applicable
        if is_quantiles:
            # Check if bin columns are numeric AND contain no NaNs (after dropna above)
            if pd.api.types.is_numeric_dtype(self.df[self.xbinname]) and \
                    pd.api.types.is_numeric_dtype(self.df[self.ybinname]):
                self.df['BINS_COMBINED_INT'] = self.df[self.xbinname] + self.df[self.ybinname]
            else:
                # This warning should ideally not trigger if dropna was effective, but good for robustness
                print("Warning: Could not create 'BINS_COMBINED_INT' as bin columns are not numeric "
                      "or still contain non-numeric values after NaN drop.")

        # Filter out bins that don't meet the minimum value requirement
        # Perform groupby and count on the original index to get accurate counts per bin
        bin_counts = self.df.groupby('BINS_COMBINED_STR')['INDEX'].count()
        valid_bins = bin_counts[bin_counts >= self.min_n_vals_per_bin].index

        # Filter the DataFrame to keep only rows belonging to valid bins
        self.df = self.df[self.df['BINS_COMBINED_STR'].isin(valid_bins)].copy()

        if self.df.empty:
            print(
                "Warning: No data left after filtering for minimum values per bin. Aggregated DataFrames will be empty.")
            self._df_agg_wide = pd.DataFrame()
            self._df_agg_long = pd.DataFrame()
            return

        # # Count number of available values per combined bin
        # ok_bin = self.df.groupby('BINS_COMBINED_STR').count()['INDEX'] >= self.min_n_vals_per_bin
        #
        # # Bins with enough values
        # ok_bin = ok_bin[ok_bin]  # Filter (True/False) indicating if bin has enough values
        # ok_binlist = list(ok_bin.index)  # List of bins with enough values
        #
        # # Keep data rows where the respective bin has enough values
        # ok_row = self.df['BINS_COMBINED_STR'].isin(ok_binlist)
        # self.df = self.df[ok_row].copy()

        # Step 4: Perform aggregation using pivot_table
        self._df_agg_wide = pd.pivot_table(self.df,
                                           index=self.ybinname,  # Rows
                                           columns=self.xbinname,  # Columns
                                           values=self.zname,  # Values to aggregate
                                           aggfunc=self.aggfunc)  # Aggregation function

        # Step 5: Melt the DataFrame to long format for easier plotting/analysis
        df_agg_long_temp = self._df_agg_wide.reset_index()

        # Identify value columns to melt, excluding the ybinname (index in wide format)
        value_vars_for_melt = [col for col in df_agg_long_temp.columns if col != self.ybinname]

        self._df_agg_long = df_agg_long_temp.melt(id_vars=[self.ybinname],
                                                  value_vars=value_vars_for_melt,
                                                  var_name=self.xbinname,
                                                  value_name=self.zname)

        # Ensure bin columns in the long format DataFrame are integers
        self._df_agg_long[self.xbinname] = self._df_agg_long[self.xbinname].astype(int)
        self._df_agg_long[self.ybinname] = self._df_agg_long[self.ybinname].astype(int)

        # df_agg_wide = pd.pivot_table(self.df, index=self.ybinname, columns=self.xbinname,
        #                              values=self.zname, aggfunc=self.aggfunc)

        # # Melt the DataFrame to long format
        # df_agg_long = df_agg_wide.reset_index().melt(id_vars=[self.ybinname],
        #                                              var_name=self.xbinname,
        #                                              value_name=self.zname)

        # return df_agg_wide, df_agg_long


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
    daytime_locs = (subset[swin_col] > 50)  # Use daytime data
    subset = subset[daytime_locs].copy()
    subset = subset.dropna()

    q = dv.ga(
        x=subset[swin_col],
        y=subset[ta_col],
        z=subset[vpd_col],
        binning_type='equal_width',
        # binning_type='quantiles',
        n_bins=10,
        min_n_vals_per_bin=5,
        aggfunc='mean'
    )

    pivotdf = q.df_agg_wide.copy()
    print(pivotdf)
    df_long = q.df_agg_long.copy()
    print(df_long)

    hm = dv.heatmapxyz(
        x=df_long['BIN_Rg_f'],
        y=df_long['BIN_Tair_f'],
        z=df_long['VPD_f'],
        cb_digits_after_comma=0,
        xlabel=r'Short-wave incoming radiation ($\mathrm{percentile}$)',
        ylabel=r'Air temperature ($\mathrm{percentile}$)',
        zlabel=r'Vapor pressure deficit (mean)'
    )
    hm.show()

    # print(res)


if __name__ == '__main__':
    _example()
