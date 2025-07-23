"""
Grid Aggregation Module

This module provides the `GridAggregator` class, a tool for
transforming and aggregating tabular data into a 2D grid structure. It
supports flexible binning strategies and various aggregation methods,
facilitating data analysis and visualization in grid-based formats.

Author: Lukas Hörtnagl
License: GPL-3.0
"""

from typing import Literal, Union, Callable

import numpy as np
import pandas as pd


class GridAggregator:
    """
    Performs data aggregation over a 2D grid based on specified binning mechanisms.

    The GridAggregator class facilitates the aggregation of Z-axis data over a
    2D grid defined by X and Y axes. The binning can be configured through
    quantiles, equal-width bins, or custom bins. The class supports multiple aggregation
    functions, including built-in statistical measures and custom callable functions.

    Attributes:
        x (pd.Series): X-axis data to be used in binning.
        y (pd.Series): Y-axis data to be used in binning.
        z (pd.Series): Z-axis data to be aggregated based on X and Y bins.
        binning_type (Literal['quantiles', 'equal_width', 'custom']):
            Type of binning method used ('quantiles', 'equal_width', or 'custom').
        n_bins (int): Number of bins for 'quantiles' or 'equal_width' binning. Ignored if the binning type is 'custom'.
        min_n_vals_per_bin (int): Minimum required values in a bin to include in aggregation.
        custom_x_bins (Union[np.ndarray, list, None]):
            Custom bin edges for the X-axis when binning type is 'custom'.
        custom_y_bins (Union[np.ndarray, list, None]):
            Custom bin edges for the Y-axis when binning type is 'custom'.
        x_col_name (str): Name derived from the X data series or defaulted to 'x_data'.
        y_col_name (str): Name derived from the Y data series or defaulted to 'y_data'.
        z_col_name (str): Name derived from the Z data series or defaulted to 'z_data'.
        x_bin_col_name (str): Column name for X-axis binning results in the internal DataFrame.
        y_bin_col_name (str): Column name for Y-axis binning results in the internal DataFrame.
        aggfunc (Union[Literal['mean', 'min', 'max', 'median', 'sum', 'count'], Callable]):
            Aggregation function for binning operation. May be a string literal or custom callable.
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
        Initializes the class with input data, binning configurations, and aggregation setup.
        Handles preparation of internal DataFrame for processing, determines column names,
        validates inputs, and performs data binning and aggregation on initialization.

        Args:
            x (pd.Series): Input series for x-coordinates or independent variable.
            y (pd.Series): Input series for y-coordinates or independent variable.
            z (pd.Series): Input series for z-values or dependent variable to be aggregated.
            binning_type (Literal['quantiles', 'equal_width', 'custom']): Type of binning strategy to apply
                for x and y dimensions. Options include quantiles, equal-width, or custom boundaries.
            n_bins (int, optional): Number of bins to create for x and y when using quantiles or
                equal-width binning. Defaults to 10.
            min_n_vals_per_bin (int, optional): Minimum number of values required in each bin. If bins
                do not meet this threshold, outputs will reflect adjusted binning. Defaults to 1.
            aggfunc (Union[Literal['mean', 'min', 'max', 'median', 'sum', 'count'], Callable], optional):
                Aggregation function to apply to z-values within each bin. Can be a predefined function
                string ('mean', 'min', 'max', 'median', 'sum', 'count') or a callable. Defaults to 'mean'.
            custom_x_bins (Union[np.ndarray, list, None], optional): Custom bin edges for x-dimension.
                Must be a sorted array or list of values when provided. Defaults to None.
            custom_y_bins (Union[np.ndarray, list, None], optional): Custom bin edges for y-dimension.
                Must be a sorted array or list of values when provided. Defaults to None.

        Raises:
            ValueError: If input series lengths do not match, binning type is invalid,
                or custom bins are not formatted correctly.
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
        """
        Validates the input data for initialization parameters, ensuring consistency,
        appropriate types, and logical correctness based on the specified binning
        type. This function raises errors for invalid or inconsistent input parameters.

        Raises:
            TypeError: If x, y, or z is not a pandas Series, or if custom_x_bins or
                custom_y_bins is not a numpy array or a list.
            ValueError: If x, y, and z do not have the same length, or if binning_type
                is invalid, or if n_bins or min_n_vals_per_bin is not a positive integer for
                non-custom binning. Also raised if custom_x_bins or custom_y_bins is None,
                contains fewer than two edges, or is not monotonically increasing
                when binning_type is 'custom'.
        """
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
        """
        A property that retrieves a long-form (non-aggregated) pandas DataFrame.

        The property ensures that the DataFrame is not empty and that
        it contains columns. A copy of the DataFrame is returned to preserve data integrity and
        prevent external modifications.
        """
        if self._df_long.empty and not self._df_long.columns.empty:  # Check if empty but columns exist (e.g., after dropna leads to empty)
            raise AttributeError("Long (non-aggregated) DataFrame is not available.")
        return self._df_long.copy()  # Return a copy to prevent external modification

    @property
    def df_agg_wide(self) -> pd.DataFrame:
        """
        Provides a read-only property to access the aggregated wide DataFrame.
        """
        if self._df_agg_wide is None or self._df_agg_wide.empty:
            raise AttributeError("Aggregated wide DataFrame is not available. Ensure binning was successful.")
        return self._df_agg_wide.copy()  # Return a copy

    @property
    def df_agg_long(self) -> pd.DataFrame:
        """
        Property to access the aggregated long DataFrame. Ensures that the DataFrame is available
        and not empty before returning a copy.
        """
        if self._df_agg_long is None or self._df_agg_long.empty:
            raise AttributeError("Aggregated long DataFrame is not available. Ensure binning was successful.")
        return self._df_agg_long.copy()  # Return a copy

    def _bin_and_aggregate(self):
        """
        Applies appropriate binning and aggregation to the data based on the specified
        binning type. Supports multiple binning methods and transforms the data accordingly.
        Raises an error when the binning type is not recognized.
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
        self._transform_and_pivot()

    def _apply_quantile_binning(self):
        """
        Applies quantile binning to specified columns in the DataFrame.

        This method modifies the DataFrame by assigning quantile-based bin numbers to the
        values in the specified columns. The bin assignments are based on percentiles
        of the data, and they are stored in new columns within the DataFrame.
        """
        self._df_long[self.x_bin_col_name] = self._assign_bins_quantiles(series=self._df_long[self.x_col_name])
        self._df_long[self.y_bin_col_name] = self._assign_bins_quantiles(series=self._df_long[self.y_col_name])

    def _apply_equal_width_binning(self):
        """
        Applies equal-width binning to the specified columns in the DataFrame.

        The method modifies the DataFrame `self._df_long` by assigning bins with equal
        width to the columns specified by `self.x_col_name` and `self.y_col_name`, and
        stores the binned results in columns named `self.x_bin_col_name` and
        `self.y_bin_col_name`, respectively.
        """
        self._df_long[self.x_bin_col_name] = self._assign_bins_equal_width(series=self._df_long[self.x_col_name])
        self._df_long[self.y_bin_col_name] = self._assign_bins_equal_width(series=self._df_long[self.y_col_name])

    def _apply_custom_binning(self):
        """
        Applies custom binning to the dataframe columns specified by the configurations.

        Sets up custom binning for x and y columns based on the provided custom bin
        configuration and assigns the resulting bins to new columns.
        """
        self._df_long[self.x_bin_col_name] = self._assign_bins_custom(series=self._df_long[self.x_col_name],
                                                                      bins=self.custom_x_bins)
        self._df_long[self.y_bin_col_name] = self._assign_bins_custom(series=self._df_long[self.y_col_name],
                                                                      bins=self.custom_y_bins)

    def _assign_bins_quantiles(self, series: pd.Series) -> pd.Series:
        """
        Assigns quantile-based bins to a given pandas Series based on the specified number of bins.

        This method applies a quantile-based binning approach to divide the data into discrete bins
        as defined by the number of quantiles (`n_bins`). If distinct quantile bins cannot be formed
        due to repetitive or identical values, NaN is returned for all entries in the series.
        The bin labels are represented as floating-point values, which denote the corresponding
        quantile percentiles.

        Returns:
            pd.Series: A pandas Series where each value is replaced by its corresponding bin label.
                       If quantile bins cannot be formed, the output series contains NaN for all indices.
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
        Assigns equal-width bins to the values in the given pandas Series and labels them with the lower
        bound of the corresponding bin. If bin creation fails or no distinct bins are formed, returns NaN
        values for bin labels.

        Returns:
            pd.Series: A new pandas Series containing the numeric bin labels corresponding to the lower
            bounds of the bins. If an error occurs or no bins are created, returns a Series of NaN values.
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
        Assigns provided custom bins to a given pandas Series and returns a new Series with bin labels as
        float values, which represent the lower bounds of the bins. If an error occurs during bin assignment,
        it returns a Series with NaN values for the bin labels.

        Args:
            series (pd.Series): Input data series for which custom binning is applied.
            bins (np.ndarray): An array of numerical bin edges to be applied for binning. If desired,
                the bins must cover the range of the series to avoid generating NaN values for series elements.

        Returns:
            pd.Series: A pandas Series with assigned bin labels represented as float values, based on
            the lower bounds of the provided bins.
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

    def _transform_and_pivot(self) -> None:
        """
        Transforms the input data by filtering, pivoting, and reformatting it into
        wide and long aggregate DataFrames based on binning configuration. Handles
        both custom and default binning types. Provides warnings if data is mostly
        filtered out.
        """
        # Drop rows where binning failed (NaNs in bin columns)
        initial_rows = len(self._df_long)
        self._df_long = self._df_long.dropna(subset=[self.x_bin_col_name, self.y_bin_col_name], inplace=False)
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
                                             "+" + self._df_long[self.y_bin_col_name].astype(str))

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

        self._df_agg_wide = pd.pivot_table(self._df_long,
                                           index=self.y_bin_col_name,  # Rows
                                           columns=self.x_bin_col_name,  # Columns
                                           values=self.z_col_name,  # Values to aggregate
                                           aggfunc=self.aggfunc)

        # If binning_type is 'custom', reindex to ensure all custom bins are present
        if self.binning_type == 'custom':
            expected_x_labels = [float(self.custom_x_bins[i]) for i in range(len(self.custom_x_bins) - 1)]
            expected_y_labels = [float(self.custom_y_bins[i]) for i in range(len(self.custom_y_bins) - 1)]

            # Reindex the wide DataFrame, filling missing bins with NaN
            self._df_agg_wide = self._df_agg_wide.reindex(index=expected_y_labels, columns=expected_x_labels)

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

        # Only drop NaNs for non-custom binning in the long format to retain empty custom bins
        if self.binning_type != 'custom':
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
        binning_type='custom',
        custom_x_bins=list(range(-3, 5, 1)),
        custom_y_bins=list(range(-3, 5, 1)),
        # custom_y_bins=list(range(0, 5, 1)),
        # binning_type='equal_width',
        # binning_type='quantiles',
        n_bins=10,
        min_n_vals_per_bin=5,
        aggfunc='mean'
    )

    df_agg_wide = q.df_agg_wide.copy()
    print(df_agg_wide)
    df_agg_long = q.df_agg_long.copy()
    df_long = q.df_long.copy()
    print(df_long)

    # df_long.loc[df_long['SWC_FF0_0.15_1'] > 3].describe()
    # df_long.loc[df_long['SWC_FF0_0.15_1'] < -2.345]

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
