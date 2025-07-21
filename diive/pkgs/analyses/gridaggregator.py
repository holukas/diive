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

        # Input checks
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

    @property
    def df_agg_wide(self) -> pd.DataFrame:
        if not isinstance(self._df_agg_wide, pd.DataFrame):
            raise Exception(
                f'Aggregated wide DataFrame is not available. Please run `quantiles()` or `equal_width()` first.')
        return self._df_agg_wide

    @property
    def df_agg_long(self) -> pd.DataFrame:
        if not isinstance(self._df_agg_long, pd.DataFrame):
            raise Exception(
                f'Aggregated long DataFrame is not available. Please run `quantiles()` or `equal_width()` first.')
        return self._df_agg_long

    def bin(self):
        if self.binning_type == 'quantiles':
            self._quantiles()
        elif self.binning_type == 'equal_width':
            self._equal_width()

    def _quantiles(self):
        self.df[self.xbinname] = self._assign_bins_quantiles(series=self.df[self.xname])
        self.df[self.ybinname] = self._assign_bins_quantiles(series=self.df[self.yname])
        self._df_agg_wide, self._df_agg_long = self._transform_data(is_quantiles=True)

    def _equal_width(self):
        self.df[self.xbinname] = self._assign_bins_equal_widh(series=self.df[self.xname])
        self.df[self.ybinname] = self._assign_bins_equal_widh(series=self.df[self.yname])
        self._df_agg_wide, self._df_agg_long = self._transform_data(is_quantiles=False)

    def _assign_bins_quantiles(self, series: pd.Series):
        # Run qcut once WITHOUT labels to determine the actual number of bins
        temp_group, temp_bins = pd.qcut(series, q=self.n_bins, retbins=True, duplicates='drop')
        actual_n_bins = len(temp_bins) - 1  # Number of bins is (num_edges - 1)
        # Generate labels based on the actual number of bins
        # This ensures 'labels' count matches the actual bins created
        labels_stepsize = int(100 / actual_n_bins) if actual_n_bins > 0 else 0
        labels = list(range(0, actual_n_bins * labels_stepsize, labels_stepsize))
        if len(labels) != actual_n_bins:  # Adjust for potential rounding issues
            labels = list(range(0, 100, int(100 / actual_n_bins)))[:actual_n_bins]
        group_x, bins_x = pd.qcut(series, q=self.n_bins, labels=labels, retbins=True, duplicates='drop')
        series_bins = group_x.astype(int)
        return series_bins

    def _assign_bins_equal_widh(self, series: pd.Series):
        # Run cut once WITHOUT labels to determine the actual number of bins
        temp_group_x, temp_bins_x = pd.cut(series, bins=self.n_bins, retbins=True, duplicates='drop')
        actual_n_bins_x = len(temp_bins_x) - 1  # Number of bins is (num_edges - 1)
        labels_x = temp_bins_x[:actual_n_bins_x]
        group_x, bins_x = pd.cut(series, bins=self.n_bins, labels=labels_x, retbins=True, duplicates='drop')
        series_bins = group_x.astype(int)
        return series_bins

    def _transform_data(self, is_quantiles: bool) -> tuple:
        """Transform data for plotting"""

        # Add new column for unique bin combinations
        self.df['BINS_COMBINED_STR'] = self.df[self.xbinname].astype(str) + "+" + self.df[self.ybinname].astype(str)

        # Adding bins together makes only sense when x and y are quantiles
        if is_quantiles:
            self.df['BINS_COMBINED_INT'] = self.df[self.xbinname].add(self.df[self.ybinname])

        # Count number of available values per combined bin
        ok_bin = self.df.groupby('BINS_COMBINED_STR').count()['INDEX'] >= self.min_n_vals_per_bin

        # Bins with enough values
        ok_bin = ok_bin[ok_bin]  # Filter (True/False) indicating if bin has enough values
        ok_binlist = list(ok_bin.index)  # List of bins with enough values

        # Keep data rows where the respective bin has enough values
        ok_row = self.df['BINS_COMBINED_STR'].isin(ok_binlist)
        self.df = self.df[ok_row].copy()

        df_agg_wide = pd.pivot_table(self.df, index=self.ybinname, columns=self.xbinname,
                                     values=self.zname, aggfunc=self.aggfunc)

        # Melt the DataFrame to long format
        df_agg_long = df_agg_wide.reset_index().melt(id_vars=[self.ybinname],
                                                     var_name=self.xbinname,
                                                     value_name=self.zname)

        return df_agg_wide, df_agg_long


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
        # binning_type='equal_width',
        binning_type='quantiles',
        n_bins=10,
        min_n_vals_per_bin=5,
        aggfunc='mean'
    )
    q.bin()

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
