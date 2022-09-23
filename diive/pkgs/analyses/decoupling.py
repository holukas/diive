# todo incomplete



"""


    kudos:
    - https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap

"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.io.files import load_pickle
from diive.core.plotting.plotfuncs import default_legend, default_format


class SortingBinsMethod:

    def __init__(self, df, var1_col, var2_col, var3_col,
                 n_bins: int = 10,
                 n_subpins: int = 10):
        self._df = df.copy()
        self.var1_col = var1_col
        self.var2_col = var2_col
        self.var3_col = var3_col
        self.n_bins = n_bins
        self.n_subpins = n_subpins

        self._df = self._df.dropna()
        self._binmeans = {}
        self._assignbins()

    @property
    def df(self) -> DataFrame:
        if self._df is None:
            raise Exception('No data available.')
        return self._df

    @property
    def binmeans(self) -> dict:
        if not self._binmeans:
            raise Exception('No binned means available.')
        return self._binmeans

    def _assignbins(self):
        group, bins = pd.qcut(self.df[self.var1_col], q=self.n_bins, retbins=True, duplicates='drop')
        self.df['var1_group'] = group

    # def binagg(self):
    #     print(self.df.groupby('var1_group').mean())

    def calcbins(self):
        counter = 0
        grouped = self.df.groupby('var1_group')
        for g, g_df in grouped:
            group, bins = pd.qcut(g_df[self.var2_col], q=self.n_subpins, retbins=True, duplicates='drop')
            g_df['var2_group'] = group
            means = g_df.groupby(g_df['var2_group']).mean()
            means = means.sort_values(by=self.var2_col)
            means = means.reset_index()
            self._binmeans[str(counter)] = means
            counter += 1

    def plot_bins(self):
        fig = plt.figure(figsize=(9, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
        ax = fig.add_subplot(gs[0, 0])
        colors = plt.cm.YlOrRd(np.linspace(0.1, 1, self.n_bins))
        for ix, m in enumerate(self.binmeans.keys()):
            ax.plot(self.binmeans[m].index, self.binmeans[m][self.var3_col],
                    ls='-', lw=1, ms=10, marker='o', label=m, color=colors[ix],
                    mec='grey', alpha=.9)

        n_vals = int(self.df.groupby('var1_group').count().mean()[self.var1_col])
        ax.text(0.98, 0.98, f'{n_vals} values per {self.var1_col} class',
                size=theme.AXLABELS_FONTSIZE, color='k', backgroundcolor='none', transform=ax.transAxes,
                alpha=1, horizontalalignment='right', verticalalignment='top')

        default_format(ax=ax, txt_xlabel=f"{self.var2_col} class", txt_ylabel=self.var3_col)
        default_legend(ax=ax)
        fig.tight_layout()
        fig.show()

        # g_df[vpd_col].corr(g_df[nee_col])
        # print(g_df[self.var2_col].corr(g_df[self.var3_col]))


def example():
    # Variables
    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    ta_col = 'Tair_f'
    gpp_col = 'GPP_DT_CUT_REF'
    reco_col = 'Reco_DT_CUT_REF'

    # Load data, using pickle for fast loading
    source_file = r'F:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\manuscripts\co2penalty_dav\input_data\CH-DAV_FP2022.1_1997-2022.08_ID20220826234456_30MIN.diive.csv.pickle'
    df_orig = load_pickle(filepath=source_file)
    df_orig = df_orig.loc[df_orig.index.year >= 2020].copy()
    df_orig = df_orig.loc[(df_orig.index.month >= 5) & (df_orig.index.month <= 9)].copy()
    df = df_orig[[nee_col, vpd_col, ta_col, gpp_col, reco_col]].copy()

    sbm = SortingBinsMethod(df=df,
                            var1_col=ta_col,
                            var2_col=vpd_col,
                            var3_col=nee_col,
                            n_bins=30,
                            n_subpins=2)
    sbm.calcbins()
    sbm.plot_bins()


if __name__ == '__main__':
    example()
