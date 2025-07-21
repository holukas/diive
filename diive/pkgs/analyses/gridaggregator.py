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
                 binning_type: Literal['quantiles', 'equal_width', 'custom'],
                 n_bins: int = 10,
                 min_n_vals_per_bin: int = 1,
                 aggfunc: Union[Literal['mean', 'min', 'max', 'median', 'sum'], Callable] = 'mean',
                 custom_x_bins: Union[np.ndarray, list, None] = None,
                 custom_y_bins: Union[np.ndarray, list, None] = None
                 ):

        # Input validation
        if not all(isinstance(arg, pd.Series) for arg in [x, y, z]):
            raise TypeError("x, y, and z must be pandas Series.")
        if not all(len(arg) == len(x) for arg in [y, z]):
            raise ValueError("x, y, and z Series must have the same length.")
        if n_bins <= 0 and binning_type != 'custom':  # n_bins not strictly required for custom type
            raise ValueError("n_bins must be a positive integer for 'quantiles' or 'equal_width' binning.")
        if min_n_vals_per_bin <= 0:
            raise ValueError("min_n_vals_per_bin must be a positive integer.")
        if binning_type not in ['quantiles', 'equal_width', 'custom']:  # Updated validation
            raise ValueError("binning_type must be 'quantiles', 'equal_width', or 'custom'.")

        # Custom bin validation
        if binning_type == 'custom':
            if custom_x_bins is None or custom_y_bins is None:
                raise ValueError("For 'custom' binning, 'custom_x_bins' and 'custom_y_bins' must be provided.")
            if not isinstance(custom_x_bins, (np.ndarray, list)) or not isinstance(custom_y_bins, (np.ndarray, list)):
                raise TypeError("'custom_x_bins' and 'custom_y_bins' must be a numpy array or a list.")
            if len(custom_x_bins) < 2 or len(custom_y_bins) < 2:
                raise ValueError("'custom_x_bins' and 'custom_y_bins' must contain at least two bin edges.")
            if not all(custom_x_bins[i] < custom_x_bins[i + 1] for i in range(len(custom_x_bins) - 1)):
                raise ValueError("'custom_x_bins' must be monotonically increasing.")
            if not all(custom_y_bins[i] < custom_y_bins[i + 1] for i in range(len(custom_y_bins) - 1)):
                raise ValueError("'custom_y_bins' must be monotonically increasing.")

        # Initialize attributes
        self.x = x.copy()
        self.y = y.copy()
        self.z = z.copy()
        self.binning_type = binning_type
        self.n_bins = n_bins
        self.min_n_vals_per_bin = min_n_vals_per_bin
        self.custom_x_bins = np.array(custom_x_bins) if custom_x_bins is not None else None
        self.custom_y_bins = np.array(custom_y_bins) if custom_y_bins is not None else None

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
        elif self.binning_type == 'custom':
            self._apply_custom_binning()

        # After binning, proceed to transform and aggregate
        self._transform_and_pivot(is_quantiles=(self.binning_type == 'quantiles'))

    def _apply_quantile_binning(self):
        self.df[self.xbinname] = self._assign_bins_quantiles(series=self.df[self.xname])
        self.df[self.ybinname] = self._assign_bins_quantiles(series=self.df[self.yname])

    def _apply_equal_width_binning(self):
        self.df[self.xbinname] = self._assign_bins_equal_width(series=self.df[self.xname])
        self.df[self.ybinname] = self._assign_bins_equal_width(series=self.df[self.yname])

    def _apply_custom_binning(self):
        self.df[self.xbinname] = self._assign_bins_custom(series=self.df[self.xname], bins=self.custom_x_bins)
        self.df[self.ybinname] = self._assign_bins_custom(series=self.df[self.yname], bins=self.custom_y_bins)

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

    def _assign_bins_equal_width(self, series: pd.Series):
        try:
            # Run cut once WITHOUT labels to determine the actual number of bins
            _, bins = pd.cut(series, bins=self.n_bins, retbins=True, duplicates='drop')

            actual_n_bins = len(bins) - 1

            if actual_n_bins == 0:
                print(f"Warning: Could not form any equal-width bins for series '{series.name}'. "
                      f"All values might be identical or range is too small. Returning NaN for bin labels.")
                return pd.Series(np.nan, index=series.index, name=f'BIN_{series.name}')

            labels = bins[:actual_n_bins]

            binned_series = pd.cut(series, bins=self.n_bins, labels=labels, retbins=False, duplicates='drop',
                                   include_lowest=True)
            # binned_series = pd.cut(series, bins=self.n_bins, labels=labels, retbins=False, duplicates='drop')
            return binned_series.astype(float)
        except Exception as e:
            print(f"Warning: Could not apply equal-width binning to series '{series.name}'. Error: {e}. "
                  f"Returning NaN for bin labels.")
            return pd.Series(np.nan, index=series.index, name=f'BIN_{series.name}')

    def _assign_bins_custom(self, series: pd.Series, bins: np.ndarray) -> pd.Series:
        try:
            # Validate custom bins internally once more, though it's also done in __init__
            if not isinstance(bins, np.ndarray) or len(bins) < 2:
                raise ValueError("Custom bins must be a numpy array with at least two edges.")
            if not np.all(np.diff(bins) > 0):
                raise ValueError("Custom bins must be monotonically increasing.")

            # Create labels based on the lower bound of each bin
            # labels = [float(bins[i]) for i in range(len(bins))]
            labels = [float(bins[i]) for i in range(len(bins) - 1)]

            # Use pd.cut with custom bins
            binned_series = pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
            return binned_series.astype(float)
        except Exception as e:
            print(f"Warning: Could not apply custom binning to series '{series.name}'. Error: {e}. "
                  f"Returning NaN for bin labels.")
            return pd.Series(np.nan, index=series.index, name=f'BIN_{series.name}')

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

        # Perform aggregation using pivot_table
        self._df_agg_wide = pd.pivot_table(self.df,
                                           index=self.ybinname,  # Rows
                                           columns=self.xbinname,  # Columns
                                           values=self.zname,  # Values to aggregate
                                           aggfunc=self.aggfunc)  # Aggregation function

        # Melt the DataFrame to long format for easier plotting/analysis
        df_agg_long_temp = self._df_agg_wide.reset_index()

        # Identify value columns to melt, excluding the ybinname (index in wide format)
        value_vars_for_melt = [col for col in df_agg_long_temp.columns if col != self.ybinname]

        self._df_agg_long = df_agg_long_temp.melt(id_vars=[self.ybinname],
                                                  value_vars=value_vars_for_melt,
                                                  var_name=self.xbinname,
                                                  value_name=self.zname)

        # Ensure bin columns in the long format DataFrame are integers
        self._df_agg_long[self.xbinname] = self._df_agg_long[self.xbinname].astype(float)
        self._df_agg_long[self.ybinname] = self._df_agg_long[self.ybinname].astype(float)


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
    # daytime_locs = (subset[swin_col] > 50)  # Use daytime data
    # subset = subset[daytime_locs].copy()
    subset = subset.dropna()

    q = dv.ga(
        x=subset[swin_col],
        y=subset[ta_col],
        z=subset[vpd_col],
        binning_type='custom',
        custom_x_bins=list(range(0, 2000, 100)),
        custom_y_bins=list(range(0, 5, 1)),
        # binning_type='equal_width',
        # binning_type='quantiles',
        n_bins=10,
        min_n_vals_per_bin=5,
        aggfunc='mean'
    )

    pivotdf = q.df_agg_wide.copy()
    print(pivotdf)
    df_long = q.df_agg_long.copy()
    # print(df_long)

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
