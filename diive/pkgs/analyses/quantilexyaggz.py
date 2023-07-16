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

        # todo missing in nchds
        # 04.07.2006
        # 19.06.2007
        # 05.07.2017
        # 29.08.2017
        # 05.06.2022

        # todo checking
        _check_longformdf = longformdf.copy()
        _check_longformdf['DATE'] = pd.to_datetime(_check_longformdf['DATE'])
        _check_longformdf.set_index('DATE', inplace=True)
        _check_longformdf.loc['2006-07-04']
        _check_longformdf.loc['2007-06-19']
        _check_longformdf.loc['2017-07-05']
        _check_longformdf.loc['2017-08-29']
        _check_longformdf.loc['2022-06-05']
        # todo checking

        # Keep data rows where the respective bin has enough values
        ok_row = longformdf['BINS_COMBINED_STR'].isin(ok_binlist)
        longformdf = longformdf[ok_row]

        # todo checking
        _check_longformdf2 = longformdf.copy()
        _check_longformdf2['DATE'] = pd.to_datetime(_check_longformdf2['DATE'])
        _check_longformdf2.set_index('DATE', inplace=True)
        _check_longformdf2.loc['2017-07-05']
        # todo checking

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
                              duplicates='raise')
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
                              duplicates='raise')
        # group, bins = pd.cut(df[self.yname],
        #                      bins=self.n_xybins,
        #                      labels=range(1, self.n_xybins + 1),
        #                      retbins=True,
        #                      duplicates='drop')
        df[self.ybinname] = group.astype(int)
        return df


def example():
    from diive.core.io.files import load_pickle

    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    ta_col = 'Tair_f'
    gpp_col = 'GPP_DT_CUT_REF'
    reco_col = 'Reco_DT_CUT_REF'
    rh_col = 'RH'
    swin_col = 'Rg_f'

    # Load data, using pickle for fast loading
    source_file = r"F:\Sync\luhk_work\20 - CODING\21 - DIIVE\diive\__manuscripts\11.01_NEP-Penalty_CH-DAV_1997-2022 (2023)\data\CH-DAV_FP2022.5_1997-2022_ID20230206154316_30MIN.diive.csv.pickle"
    df_orig = load_pickle(filepath=source_file)
    # df_orig = df_orig.loc[df_orig.index.year >= 2019].copy()

    # Data between May and Sep
    # df_orig = df_orig.loc[(df_orig.index.month >= 5) & (df_orig.index.month <= 9)].copy()

    # Subset
    df = df_orig[[nee_col, vpd_col, ta_col, gpp_col, reco_col, swin_col]].copy()

    # # Daytime, warm data, HH
    # daytime_locs = (df[swin_col] > 50) & (df[ta_col] > 10)
    # df = df[daytime_locs].copy()
    # df = df.dropna()

    # Aggregation to daily values
    df = df.groupby(df.index.date).agg(
        {
            ta_col: ['min', 'max'],
            vpd_col: ['min', 'max'],
            nee_col: 'mean'
        }
    )

    # from diive.core.dfun.frames import flatten_multiindex_all_df_cols
    # df = flatten_multiindex_all_df_cols(df=df)
    # x = f"{ta_col}_max"
    # y = f"{vpd_col}_max"
    # z = f"{nee_col}_sum"

    res = QuantileXYAggZ(x=df[ta_col],
                         y=df[vpd_col],
                         z=df[nee_col],
                         n_quantiles=20,
                         min_n_vals_per_bin=10,
                         binagg_z='count').get()

    print(res)


if __name__ == '__main__':
    example()
