"""
Grid Aggregation Module

This module provides the `GridAggregator` class, a tool for
transforming and aggregating tabular data into a 2D grid structure. It
supports flexible binning strategies and various aggregation methods,
facilitating data analysis and visualization in grid-based formats.

Author: Lukas Hörtnagl
License: GPL-3.0
"""

from typing import Literal

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


class GridAggregator:
    """
        Aggregates three pandas Series (x, y, and z) into a 2D grid structure,
        allowing for various aggregation methods on the 'z' Series within each bin.

        This class facilitates the creation of both wide and long-form DataFrames
        from binned data, useful for visualization and further analysis of
        relationships between three variables.

        Attributes:
            x (pd.Series): The Series to be used for the x-axis, binned.
            y (pd.Series): The Series to be used for the y-axis, binned.
            z (pd.Series): The Series to be aggregated within the x-y bins.
            xname (str): The name of the x Series (or 'x' if not specified).
            yname (str): The name of the y Series (or 'y' if not specified).
            zname (str): The name of the z Series (or 'z' if not specified).
            quantiles (bool): If True, binning uses quantiles (qcut); otherwise,
                              it uses fixed-width bins (cut). Defaults to False.
            n_bins (int): The number of bins to create for both x and y Series.
                          Defaults to 10.
            min_n_vals_per_bin (int): The minimum number of values required in a
                                      combined x-y bin for it to be included in
                                      the output. Bins with fewer values are dropped.
                                      Defaults to 1.
            binagg_z (Literal['mean', 'min', 'max', 'median', 'count', 'sum']):
                      The aggregation function to apply to the 'z' values within
                      each bin. Defaults to 'mean'.
            xbinname (str): The name of the generated x-axis bin column (e.g., 'BIN_x').
            ybinname (str): The name of the generated y-axis bin column (e.g., 'BIN_y').
            aggfunc (str or callable): The internal aggregation function derived from
                                       `binagg_z`. Can be a string ('sum', 'mean', etc.)
                                       or `numpy.count_nonzero` for 'count'.
            _df_wide (pd.DataFrame or None): Stores the wide-format aggregated DataFrame
                                             after `run()` is called. Initially None.
            _df_long (pd.DataFrame or None): Stores the long-form aggregated DataFrame
                                             after `run()` is called. Initially None.
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

        _counts = pd.pivot_table(df_long, index=self.ybinname, columns=self.xbinname,
                                 values=self.zname, aggfunc=np.count_nonzero)

        df_wide_agg = pd.pivot_table(df_long, index=self.ybinname, columns=self.xbinname,
                                     values=self.zname, aggfunc=self.aggfunc)

        # Melt the DataFrame to long format
        df_long_agg = df_wide_agg.reset_index().melt(id_vars=[self.ybinname],
                                                     var_name=self.xbinname, value_name=self.zname)

        return df_wide_agg, df_long_agg

    def _assign_bins(self, series: pd.Series) -> pd.Series:

        # --- Dynamic Label Generation for X-axis ---
        # Run qcut/cut once WITHOUT labels to determine the actual number of bins
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

        # Run qcut/cut again WITH the correctly sized labels
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
        quantiles=False,
        n_bins=20,
        min_n_vals_per_bin=5,
        binagg_z='mean'
    )
    q.run()

    pivotdf = q.df_wide.copy()
    print(pivotdf)
    df_long = q.df_long.copy()

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
