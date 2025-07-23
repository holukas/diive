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
    """
    A class to perform 2D binning and aggregation on pandas Series data.

    It supports quantile, equal-width, and custom binning methods for both X and Y axes,
    and aggregates a Z series based on these bins. The aggregated data can be accessed
    in both wide (pivot table) and long (melted) formats.
    """

    def __init__(self,
                 x: pd.Series,
                 y: pd.Series,
                 z: pd.Series,
                 binning_type: Literal['quantiles', 'equal_width', 'custom'],
                 n_bins: int = 10,
                 min_n_vals_per_bin: int = 1,
                 aggfunc: Union[Literal['mean', 'min', 'max', 'median', 'sum', 'count'], Callable] = 'mean',
                 custom_x_bins: Union[np.ndarray, list, None] = None,
                 custom_y_bins: Union[np.ndarray, list, None] = None
                 ):
        """
        Initializes the GridAggregator with data and binning parameters.

        Args:
            x (pd.Series): The Series for the X-axis data.
            y (pd.Series): The Series for the Y-axis data.
            z (pd.Series): The Series for the Z-axis data (values to be aggregated).
            binning_type (Literal['quantiles', 'equal_width', 'custom']):
                The type of binning to apply.
            n_bins (int, optional): The number of bins for 'quantiles' or 'equal_width' binning.
                                    Defaults to 10. Ignored if binning_type is 'custom'.
            min_n_vals_per_bin (int, optional): The minimum number of values required in a bin
                                                for it to be included in the aggregation. Defaults to 1.
            aggfunc (Union[Literal['mean', 'min', 'max', 'median', 'sum', 'count'], Callable], optional):
                The aggregation function to apply. Can be a string ('mean', 'min', 'max', 'median', 'sum', 'count')
                or a callable function (e.g., np.mean). Defaults to 'mean'.
            custom_x_bins (Union[np.ndarray, list, None], optional): Custom bin edges for the X-axis.
                                                                    Required if binning_type is 'custom'.
            custom_y_bins (Union[np.ndarray, list, None], optional): Custom bin edges for the Y-axis.
                                                                    Required if binning_type is 'custom'.
        Raises:
            TypeError: If x, y, z are not pandas Series or custom bins are not numpy array/list.
            ValueError: If input series lengths differ, n_bins/min_n_vals_per_bin are invalid,
                        binning_type is unknown, or custom bins are invalid.
        """
        self._validate_init_inputs(x, y, z, binning_type, n_bins, min_n_vals_per_bin,
                                   custom_x_bins, custom_y_bins)

        # Initialize attributes
        self.x = x.copy()
        self.y = y.copy()
        self.z = z.copy()
        self.binning_type = binning_type
        self.n_bins = n_bins
        self.min_n_vals_per_bin = min_n_vals_per_bin
        # Ensure custom bins are numpy arrays if provided
        self.custom_x_bins = np.array(custom_x_bins) if custom_x_bins is not None else None
        self.custom_y_bins = np.array(custom_y_bins) if custom_y_bins is not None else None

        # Generate meaningful names for columns
        self.x_col_name = x.name if x.name is not None else 'x_data'
        self.y_col_name = y.name if y.name is not None else 'y_data'
        self.z_col_name = z.name if z.name is not None else 'z_data'

        self.x_bin_col_name = f'BIN_{self.x_col_name}'
        self.y_bin_col_name = f'BIN_{self.y_col_name}'

        # Map 'count' string to numpy.size (more appropriate for counts than nonzero)
        # For other aggregation strings, pandas handles them directly in pivot_table.
        self.aggfunc = np.size if aggfunc == 'count' else aggfunc

        # Prepare the internal DataFrame for long format data
        self._df_long = pd.DataFrame({
            self.x_col_name: self.x,
            self.y_col_name: self.y,
            self.z_col_name: self.z
        })
        self._df_long.index.name = 'INDEX'
        # Reset index to make the original index available as a column for counting
        self._df_long = self._df_long.reset_index(drop=False).copy()

        self._df_agg_wide = None
        self._df_agg_long = None

        # Perform binning and aggregation immediately upon initialization
        self._bin_and_aggregate()

    def _validate_init_inputs(self, x, y, z, binning_type, n_bins, min_n_vals_per_bin,
                              custom_x_bins, custom_y_bins):
        """Validates all inputs provided to the __init__ method."""
        if not all(isinstance(arg, pd.Series) for arg in [x, y, z]):
            raise TypeError("x, y, and z must be pandas Series.")
        if not all(len(arg) == len(x) for arg in [y, z]):
            raise ValueError("x, y, and z Series must have the same length.")
        if binning_type not in ['quantiles', 'equal_width', 'custom']:
            raise ValueError(f"Invalid binning_type: '{binning_type}'. "
                             "Must be 'quantiles', 'equal_width', or 'custom'.")

        if binning_type != 'custom':
            if not isinstance(n_bins, int) or n_bins <= 0:
                raise ValueError("n_bins must be a positive integer for 'quantiles' or 'equal_width' binning.")
            if not isinstance(min_n_vals_per_bin, int) or min_n_vals_per_bin <= 0:
                raise ValueError(
                    "min_n_vals_per_bin must be a positive integer for 'quantiles' or 'equal_width' binning.")
        else:  # binning_type == 'custom'
            if custom_x_bins is None or custom_y_bins is None:
                raise ValueError("For 'custom' binning, 'custom_x_bins' and 'custom_y_bins' must be provided.")
            if not isinstance(custom_x_bins, (np.ndarray, list)) or not isinstance(custom_y_bins, (np.ndarray, list)):
                raise TypeError("'custom_x_bins' and 'custom_y_bins' must be a numpy array or a list.")

            # Convert to numpy arrays for consistent validation and operations
            custom_x_bins = np.array(custom_x_bins)
            custom_y_bins = np.array(custom_y_bins)

            if len(custom_x_bins) < 2 or len(custom_y_bins) < 2:
                raise ValueError("'custom_x_bins' and 'custom_y_bins' must contain at least two bin edges.")
            if not np.all(np.diff(custom_x_bins) > 0):
                raise ValueError("'custom_x_bins' must be monotonically increasing.")
            if not np.all(np.diff(custom_y_bins) > 0):
                raise ValueError("'custom_y_bins' must be monotonically increasing.")

    @property
    def df_long(self) -> pd.DataFrame:
        if self._df_long.empty and not self._df_long.columns.empty:  # Check if empty but columns exist (e.g., after dropna leads to empty)
            raise AttributeError("Long (non-aggregated) DataFrame is not available.")
        return self._df_long.copy()  # Return a copy to prevent external modification

    @property
    def df_agg_wide(self) -> pd.DataFrame:
        if self._df_agg_wide is None or self._df_agg_wide.empty:
            raise AttributeError("Aggregated wide DataFrame is not available. Ensure binning was successful.")
        return self._df_agg_wide.copy()  # Return a copy

    @property
    def df_agg_long(self) -> pd.DataFrame:
        if self._df_agg_long is None or self._df_agg_long.empty:
            raise AttributeError("Aggregated long DataFrame is not available. Ensure binning was successful.")
        return self._df_agg_long.copy()  # Return a copy

    def _bin_and_aggregate(self):
        """
        Orchestrates the binning and aggregation process based on the specified binning type.
        """
        # Select the appropriate binning method
        binning_methods = {
            'quantiles': self._apply_quantile_binning,
            'equal_width': self._apply_equal_width_binning,
            'custom': self._apply_custom_binning
        }
        binning_method = binning_methods.get(self.binning_type)

        if binning_method:
            binning_method()
        else:
            # This case should ideally be caught by __init__ validation, but good for robustness
            raise ValueError(f"Unknown binning type: {self.binning_type}")

        # After binning, proceed to transform and aggregate
        self._transform_and_pivot(is_quantiles=(self.binning_type == 'quantiles'))

    def _apply_quantile_binning(self):
        self._df_long[self.x_bin_col_name] = self._assign_bins_quantiles(series=self._df_long[self.x_col_name])
        self._df_long[self.y_bin_col_name] = self._assign_bins_quantiles(series=self._df_long[self.y_col_name])

    def _apply_equal_width_binning(self):
        self._df_long[self.x_bin_col_name] = self._assign_bins_equal_width(series=self._df_long[self.x_col_name])
        self._df_long[self.y_bin_col_name] = self._assign_bins_equal_width(series=self._df_long[self.y_col_name])

    def _apply_custom_binning(self):
        self._df_long[self.x_bin_col_name] = self._assign_bins_custom(series=self._df_long[self.x_col_name],
                                                                      bins=self.custom_x_bins)
        self._df_long[self.y_bin_col_name] = self._assign_bins_custom(series=self._df_long[self.y_col_name],
                                                                      bins=self.custom_y_bins)

    def _assign_bins_quantiles(self, series: pd.Series) -> pd.Series:
        """
        Assigns bins to a Series using quantile-based binning.
        Labels are integers representing the lower bound percentile (0-100).
        """
        # Use a consistent name for the series in warnings
        series_name = series.name if series.name else 'Unnamed Series'

        try:
            # First, determine bins to get actual number of bins and drop duplicates if needed
            _, bins = pd.qcut(series, q=self.n_bins, retbins=True, duplicates='drop')

            actual_n_bins = len(bins) - 1

            if actual_n_bins == 0:
                print(f"Warning: No distinct quantile bins formed for '{series_name}'. "
                      "All values might be identical. Returning NaN for bin labels.")
                return pd.Series(np.nan, index=series.index, name=f'BIN_{series_name}')

            # Generate labels as percentiles (0, 10, 20... 90 for 10 bins)
            labels = [float(i * (100 / actual_n_bins)) for i in range(actual_n_bins)]

            # Apply qcut with the determined labels
            binned_series = pd.qcut(series, q=self.n_bins, labels=labels,
                                    duplicates='drop', retbins=False)  # retbins=False as we already have bins

            # Convert to float to handle potential NaNs more gracefully than int
            return binned_series.astype(float)

        except ValueError as e:
            print(f"Warning: Could not apply quantile binning to '{series_name}'. Error: {e}. "
                  "This might happen if there are too many duplicate values preventing distinct quantiles. "
                  "Returning NaN for bin labels.")
            return pd.Series(np.nan, index=series.index, name=f'BIN_{series_name}')
        except Exception as e:
            print(f"An unexpected error occurred during quantile binning for '{series_name}': {e}. "
                  "Returning NaN for bin labels.")
            return pd.Series(np.nan, index=series.index, name=f'BIN_{series_name}')

    def _assign_bins_equal_width(self, series: pd.Series) -> pd.Series:
        """
        Assigns bins to a Series using equal-width binning.
        Labels are the lower bound of each bin.
        """
        series_name = series.name if series.name else 'Unnamed Series'
        try:
            # Calculate bins and labels
            # pd.cut automatically handles cases where n_bins might be too high for the data range
            binned_series, bins = pd.cut(series, bins=self.n_bins, right=False,
                                         retbins=True, include_lowest=True, duplicates='drop')

            actual_n_bins = len(bins) - 1

            if actual_n_bins == 0:
                print(f"Warning: No distinct equal-width bins formed for '{series_name}'. "
                      "All values might be identical or range too small. Returning NaN for bin labels.")
                return pd.Series(np.nan, index=series.index, name=f'BIN_{series_name}')

            # Use the lower bound of each interval as the label
            binned_series_numeric_labels = binned_series.apply(lambda x: x.left if pd.notna(x) else np.nan)

            return binned_series_numeric_labels.astype(float)

        except Exception as e:
            print(f"Warning: Could not apply equal-width binning to '{series_name}'. Error: {e}. "
                  "Returning NaN for bin labels.")
            return pd.Series(np.nan, index=series.index, name=f'BIN_{series_name}')

    def _assign_bins_custom(self, series: pd.Series, bins: np.ndarray) -> pd.Series:
        """
        Assigns bins to a Series using custom bin edges.
        Labels are the lower bound of each custom bin.
        """
        series_name = series.name if series.name else 'Unnamed Series'
        try:
            # Custom bins are already validated in __init__
            # Create labels based on the lower bound of each bin
            labels = [float(bins[i]) for i in range(len(bins) - 1)]

            # Use pd.cut with custom bins
            # Ensure custom bins cover the data range to avoid NaNs, though NaNs are handled if they appear.
            binned_series = pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
            return binned_series.astype(float)
        except Exception as e:
            print(f"Warning: Could not apply custom binning to '{series_name}'. Error: {e}. "
                  f"Returning NaN for bin labels.")
            return pd.Series(np.nan, index=series.index, name=f'BIN_{series_name}')

    def _transform_and_pivot(self, is_quantiles: bool) -> None:
        """
        Transforms the binned data and performs pivot aggregation.
        Filters out rows with NaN bins and bins that don't meet min_n_vals_per_bin.
        """
        # Drop rows where binning failed (NaNs in bin columns)
        initial_rows = len(self._df_long)
        self._df_long.dropna(subset=[self.x_bin_col_name, self.y_bin_col_name], inplace=True)
        if len(self._df_long) < initial_rows:
            print(f"Info: Dropped {initial_rows - len(self._df_long)} rows due to NaN values "
                  f"in '{self.x_bin_col_name}' or '{self.y_bin_col_name}' after binning.")

        if self._df_long.empty:
            print("Warning: No data remaining after dropping rows with failed binning. "
                  "Aggregated DataFrames will be empty.")
            self._df_agg_wide = pd.DataFrame()
            self._df_agg_long = pd.DataFrame()
            return

        # Create a combined bin identifier string
        self._df_long['BIN_COMBINED_STR'] = (self._df_long[self.x_bin_col_name].astype(str) +
                                             "_" + self._df_long[self.y_bin_col_name].astype(str))

        # Filter out bins that don't meet the minimum value requirement
        # Use 'original_index' for accurate counts per bin from the initial dataset
        bin_counts = self._df_long.groupby('BIN_COMBINED_STR')['INDEX'].count()
        valid_bins = bin_counts[bin_counts >= self.min_n_vals_per_bin].index

        # Apply the filter to keep only rows belonging to valid bins
        self._df_long = self._df_long[self._df_long['BIN_COMBINED_STR'].isin(valid_bins)].copy()

        if self._df_long.empty:
            print("Warning: No data remaining after filtering for minimum values per bin. "
                  "Aggregated DataFrames will be empty.")
            self._df_agg_wide = pd.DataFrame()
            self._df_agg_long = pd.DataFrame()
            return

        # Perform aggregation using pivot_table
        # Pandas pivot_table can directly handle the string-based aggfunc for common functions.
        # For custom callables, it also works seamlessly.
        self._df_agg_wide = pd.pivot_table(self._df_long,
                                           index=self.y_bin_col_name,  # Rows
                                           columns=self.x_bin_col_name,  # Columns
                                           values=self.z_col_name,  # Values to aggregate
                                           aggfunc=self.aggfunc)  # Aggregation function

        # Ensure index and columns are sorted numerically for consistent output
        if pd.api.types.is_numeric_dtype(self._df_agg_wide.index):
            self._df_agg_wide = self._df_agg_wide.sort_index(axis=0)
        if pd.api.types.is_numeric_dtype(self._df_agg_wide.columns):
            self._df_agg_wide = self._df_agg_wide.sort_index(axis=1)

        # Melt the DataFrame to long format for easier plotting/analysis
        df_agg_long_temp = self._df_agg_wide.reset_index()

        # Identify value columns to melt, which are all columns except the y-bin column
        value_vars_for_melt = [col for col in df_agg_long_temp.columns if col != self.y_bin_col_name]

        self._df_agg_long = df_agg_long_temp.melt(id_vars=[self.y_bin_col_name],
                                                  value_vars=value_vars_for_melt,
                                                  var_name=self.x_bin_col_name,
                                                  value_name=self.z_col_name)

        # Ensure bin columns in the long format DataFrame are float
        # This is important for consistent data types after melting, especially if some bins might be NaN originally.
        self._df_agg_long[self.x_bin_col_name] = self._df_agg_long[self.x_bin_col_name].astype(float)
        self._df_agg_long[self.y_bin_col_name] = self._df_agg_long[self.y_bin_col_name].astype(float)

        # Drop rows where the aggregated value is NaN (e.g., bins with no data after filtering)
        self._df_agg_long.dropna(subset=[self.z_col_name], inplace=True)


def _example():
    import diive as dv

    # Load example data
    df = dv.load_exampledata_parquet()

    # Make subset of three required columns
    vpd_col = 'VPD_f'  # Vapor pressure deficit
    ta_col = 'Tair_f'  # Air temperature
    swin_col = 'Rg_f'  # Shortwave incoming radiation
    swc_col = 'SWC_FF0_0.15_1'
    flux_col = 'NEE_CUT_REF_f'
    subset = df[[flux_col, vpd_col, ta_col, swc_col, swin_col]].copy()

    # daytime_locs = (subset[swin_col] > 50)  # Use daytime data
    # subset = subset[daytime_locs].copy()
    # subset = subset.drop(swin_col, axis=1, inplace=False)  # Remove col from df

    # Convert to z-scores, ignoring NaNs
    from scipy.stats import zscore
    subset = subset.apply(lambda x: zscore(x, nan_policy='omit'))

    subset = subset.loc[(subset.index.month >= 5) & (subset.index.month <= 10)].copy()  # Use data May and Sep
    subset = subset.dropna()

    print(subset.describe())

    q = dv.ga(
        x=subset[vpd_col],
        y=subset[swc_col],
        z=subset[flux_col],
        # binning_type='custom',
        # custom_x_bins=[-99, -3, -2, -1, 0, 1, 2, 3],
        # custom_y_bins=[-99, -3, -2, -1, 0, 1, 2, 3],
        # custom_y_bins=list(range(0, 5, 1)),
        # binning_type='equal_width',
        binning_type='quantiles',
        n_bins=10,
        min_n_vals_per_bin=5,
        aggfunc='mean'
    )

    df_agg_wide = q.df_agg_wide.copy()
    print(df_agg_wide)
    df_agg_long = q.df_agg_long.copy()
    # print(q.df_long)

    hm = dv.heatmapxyz(
        x=df_agg_long[f'BIN_{vpd_col}'],
        y=df_agg_long[f'BIN_{swc_col}'],
        z=df_agg_long[f'{flux_col}'],
        cb_digits_after_comma=0,
        xlabel=r'x data',
        ylabel=r'y data',
        zlabel=r'z data'
    )
    hm.show()


if __name__ == '__main__':
    _example()
