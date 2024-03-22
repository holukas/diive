from typing import Literal

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


class QuantileXYAggZ:
    """
    Calculate z-aggregates in quantiles (classes) of x and y

    By default, x and y are binned into 10 classes (n_quantiles: int = 10) and
    the mean of z is shown in each of the resulting 100 classes (10*10).

    The result is a pivoted dataframe and its longform.
    """

    def __init__(self,
                 x: Series,
                 y: Series,
                 z: Series,
                 n_quantiles: int = 10,
                 min_n_vals_per_bin: int = 1,
                 binagg_z: Literal['mean', 'min', 'max', 'median', 'count'] = 'mean'
                 ):
        self.x = x
        self.y = y
        self.z = z
        self.xname = x.name
        self.yname = y.name
        self.zname = z.name
        self.n_quantiles = n_quantiles
        self.min_n_vals_per_bin = min_n_vals_per_bin

        self.xbinname = f'BIN_{self.xname}'
        self.ybinname = f'BIN_{self.yname}'

        if binagg_z == 'sum':
            self.aggfunc = np.sum
        elif binagg_z == 'mean':
            self.aggfunc = np.mean
        elif binagg_z == 'min':
            self.aggfunc = np.min
        elif binagg_z == 'max':
            self.aggfunc = np.max
        elif binagg_z == 'median':
            self.aggfunc = np.median
        elif binagg_z == 'count':
            self.aggfunc = np.count_nonzero

        self._pivotdf = None
        self._longformdf = None

    def run(self):
        # Create dataframe from input data
        plot_df = pd.concat([self.x, self.y, self.z], axis=1)

        # Remove duplicates, in case e.g. x=y
        plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]
        plot_df.index.name = 'DATE'

        self._pivotdf, self._longformdf = self._transform_data(df=plot_df)

    @property
    def pivotdf(self) -> DataFrame:
        if not isinstance(self._pivotdf, DataFrame):
            raise Exception(f'pivotdf is not available, use .run() first.')
        return self._pivotdf

    @property
    def longformdf(self) -> DataFrame:
        if not isinstance(self._longformdf, DataFrame):
            raise Exception(f'longformdf is not available, use .run() first.')
        return self._longformdf

    def _transform_data(self, df) -> tuple:
        """Transform data for plotting"""

        # Setup xy axes
        longformdf = df.reset_index(drop=False).copy()

        # Bin x and y data
        longformdf = self._assign_xybins(df=longformdf)

        # Add new column for unique bin combinations
        longformdf['BINS_COMBINED_STR'] = longformdf[self.xbinname].astype(str) + "+" + longformdf[
            self.ybinname].astype(str)
        longformdf['BINS_COMBINED_INT'] = longformdf[self.xbinname].add(longformdf[self.ybinname])

        # Count number of available values per combined bin
        ok_bin = longformdf.groupby('BINS_COMBINED_STR').count()['DATE'] >= self.min_n_vals_per_bin
        # ok_bin = longformdf.groupby('BIN_Tair_f_max').count()['DATE'] > self.min_n_vals_per_bin

        # Bins with enough values
        ok_bin = ok_bin[ok_bin]  # Filter (True/False) indicating if bin has enough values
        ok_binlist = list(ok_bin.index)  # List of bins with enough values

        # Keep data rows where the respective bin has enough values
        ok_row = longformdf['BINS_COMBINED_STR'].isin(ok_binlist)
        longformdf = longformdf[ok_row]

        _counts = pd.pivot_table(longformdf, index=self.ybinname, columns=self.xbinname, values=self.zname,
                                 aggfunc=np.count_nonzero)

        pivotdf = pd.pivot_table(longformdf, index=self.ybinname, columns=self.xbinname, values=self.zname,
                                 aggfunc=self.aggfunc)

        return pivotdf, longformdf

    def _assign_xybins(self, df: DataFrame) -> DataFrame:
        """Create bins for x and y data"""
        labels_stepsize = int(100 / self.n_quantiles)
        labels = range(0, 100, labels_stepsize)
        group, bins = pd.qcut(df[self.xname],
                              q=self.n_quantiles,
                              labels=labels,
                              retbins=True,
                              # duplicates='raise',
                              duplicates='drop'
                              )
        # group, bins = pd.cut(df[self.xname],
        #                      bins=self.n_quantiles,
        #                      labels=labels,
        #                      retbins=True,
        #                      duplicates='drop')
        df[self.xbinname] = group.astype(int)
        group, bins = pd.qcut(df[self.yname],
                              q=self.n_quantiles,
                              labels=labels,
                              retbins=True,
                              # duplicates='raise',
                              duplicates='drop'
                              )
        # group, bins = pd.cut(df[self.yname],
        #                      bins=self.n_xybins,
        #                      labels=range(1, self.n_xybins + 1),
        #                      retbins=True,
        #                      duplicates='drop')
        df[self.ybinname] = group.astype(int)
        return df


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    from diive.core.plotting.heatmap_xyz import HeatmapPivotXYZ

    vpd_col = 'VPD_f'
    ta_col = 'Tair_f'
    swin_col = 'Rg_f'

    df = load_exampledata_parquet()

    # Data between May and Sep
    df = df.loc[(df.index.month >= 5) & (df.index.month <= 9)].copy()

    # Subset
    df = df[[vpd_col, ta_col, swin_col]].copy()

    # Daytime
    daytime_locs = (df[swin_col] > 0)
    df = df[daytime_locs].copy()
    df = df.dropna()

    # # Aggregation to daily values
    # df = df.groupby(df.index.date).agg({ta_col: ['min', 'max'],
    #                                     vpd_col: ['min', 'max'],
    #                                     nee_col: 'mean'})
    #
    # df = flatten_multiindex_all_df_cols(df=df)
    # x = f"{ta_col}_max"
    # y = f"{vpd_col}_max"
    # z = f"{nee_col}_mean"

    q = QuantileXYAggZ(
        # x=df[x],
        # y=df[y],
        # z=df[z],
        x=df[swin_col],
        y=df[ta_col],
        z=df[vpd_col],
        n_quantiles=10,
        min_n_vals_per_bin=3,
        binagg_z='mean'
    )
    q.run()

    pivotdf = q.pivotdf.copy()
    print(pivotdf)
    # print(q.longformdf)

    hm = HeatmapPivotXYZ(pivotdf=pivotdf)
    hm.plot(cb_digits_after_comma=0,
            xlabel=r'Short-wave incoming radiation ($\mathrm{percentile}$)',
            ylabel=r'Air temperature ($\mathrm{percentile}$)',
            # ylabel=r'Percentile of TA ($\mathrm{°C}$)',
            zlabel=r'Vapor pressure deficit (counts)',
            # zlabel=r'Vapor pressure deficit ($\mathrm{gCO_{2}\ m^{-2}\ d^{-1}}$)',
            # tickpos=[16, 25, 50, 75, 84],
            # ticklabels=['16', '25', '50', '75', '84']
            )

    # print(res)


if __name__ == '__main__':
    example()
