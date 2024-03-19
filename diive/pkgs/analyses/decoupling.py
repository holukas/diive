"""
DECOUPLING: SORTING BINS METHOD
===============================

    Reference:
    - todo


    kudos:
    - https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap

"""
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import default_format
from diive.core.plotting.plotfuncs import default_legend
from diive.core.plotting.plotfuncs import save_fig


class SortingBinsMethod:

    def __init__(self,
                 df: DataFrame,
                 var1_col: str,
                 var2_col: str,
                 var3_col: str,
                 n_bins_var1: int = 48,
                 n_subbins_var2: int = 2,
                 convert_to_percentiles: bool = False):
        """Investigate binned aggregates of a variable z in binned classes of x and y.

        For example: show mean GPP (y) in 5 classes of VPD (x), separate for 10 classes
        of air temperature (z).

        Args:
            df:
            var1_col:
            var2_col:
            var3_col:
            n_bins_var1:
            n_subbins_var2:

        - Example notebook available in:
            notebooks/Analyses/DecouplingSortingBins.ipynb
        """
        self._df = df.copy().dropna()
        self.var1_col = var1_col
        self.var2_col = var2_col
        self.var3_col = var3_col
        self.n_bins_var1 = n_bins_var1
        self.n_subbins_var2 = n_subbins_var2
        self.convert_to_percentiles = convert_to_percentiles

        self.var1_group_col = f"group_{self.var1_col}"
        self.var2_group_col = f"group_{self.var2_col}"

        self._binmedians = {}
        if self.convert_to_percentiles:
            self._convert_to_percentiles()
        self._assignbins()

    @property
    def df(self) -> DataFrame:
        if self._df is None:
            raise Exception('No data available.')
        return self._df

    @property
    def binmedians(self) -> dict:
        if not self._binmedians:
            raise Exception('No binned means available, try to run .calcbins() first.')
        return self._binmedians

    def get_binmeans(self) -> dict:
        """Return dict of dataframes with variable 1 group as key"""
        return self.binmedians

    def _convert_to_percentiles(self):
        self._df = self.df.rank(pct=True).copy()

    def _assignbins(self):
        group, bins = pd.qcut(self.df[self.var1_col], q=self.n_bins_var1, retbins=True, duplicates='drop')

        self._df[self.var1_group_col] = group
        # perc_df = self.df[[self.var1_col, self.var2_col, self.var3_col]].copy()

    def calcbins(self):
        # counter = 0
        grouped = self.df.groupby(by=self.var1_group_col, observed=True, as_index=True, sort=True, group_keys=True)
        for g, g_df in grouped:
            g_df = g_df.set_index(self.var1_group_col)

            label_var1 = g_df[self.var1_col].mean()
            label_var1 = label_var1.round(decimals=2)
            label_var1 = f"{label_var1}"

            group, bins = pd.qcut(g_df[self.var2_col], q=self.n_subbins_var2, retbins=True, duplicates='drop')
            g_df[self.var2_group_col] = group

            medians = g_df.groupby(by=g_df[self.var2_group_col], observed=True, as_index=True, sort=False,
                                   group_keys=True).median()
            p25 = g_df.groupby(by=g_df[self.var2_group_col], observed=True, as_index=True, sort=False,
                               group_keys=True).quantile(.25)
            p75 = g_df.groupby(by=g_df[self.var2_group_col], observed=True, as_index=True, sort=False,
                               group_keys=True).quantile(.75)
            var2_p25_col = f"{self.var2_col}_P25"
            var3_p25_col = f"{self.var3_col}_P25"
            medians[var2_p25_col] = p25[self.var2_col]
            medians[var3_p25_col] = p25[self.var3_col]
            var2_p75_col = f"{self.var2_col}_P75"
            var3_p75_col = f"{self.var3_col}_P75"
            medians[var2_p75_col] = p75[self.var2_col]
            medians[var3_p75_col] = p75[self.var3_col]

            medians['xerror_neg'] = medians[self.var2_col] - medians[var2_p25_col]
            medians['xerror_pos'] = medians[self.var2_col] - medians[var2_p75_col]
            medians['yerror_neg'] = medians[self.var3_col] - medians[var3_p25_col]
            medians['yerror_pos'] = medians[self.var3_col] - medians[var3_p75_col]

            medians['xerror_neg'] = medians['xerror_neg'].abs()
            medians['xerror_pos'] = medians['xerror_pos'].abs()
            medians['yerror_neg'] = medians['yerror_neg'].abs()
            medians['yerror_pos'] = medians['yerror_pos'].abs()

            medians = medians.sort_values(by=self.var2_col)
            medians = medians.reset_index()
            self._binmedians[label_var1] = medians

            # means = g_df.groupby(by=g_df[self.var2_group_col], observed=True, as_index=True, sort=False,
            #                      group_keys=True).mean()
            # std = g_df.groupby(by=g_df[self.var2_group_col], observed=True, as_index=True, sort=False,
            #                    group_keys=True).std()
            # var2_sd_col = f"{self.var2_col}_SD"
            # var3_sd_col = f"{self.var3_col}_SD"
            # means[var2_sd_col] = std[self.var2_col]
            # means[var3_sd_col] = std[self.var3_col]
            # means = means.sort_values(by=self.var2_col)
            # means = means.reset_index()
            # self._binmeans[label_var1] = means

    def showplot_decoupling_sbm(self,
                                saveplot: bool = False,
                                title: str = None,
                                path: Path or str = None,
                                emphasize_lines: bool = True,
                                **kwargs):

        # Figure size and legen number of columns
        n_col = 1  # int(self.n_bins_var1 / 20)
        figsize = (12, 9)
        if (self.n_bins_var1 > 24) and (self.n_bins_var1 <= 48):
            n_col += 1
        elif (self.n_bins_var1 > 48) and (self.n_bins_var1 <= 72):
            n_col += 2
            figsize = (14, 9)
        elif (self.n_bins_var1 > 72) and (self.n_bins_var1 <= 96):
            n_col += 3
            figsize = (16, 9)
        elif (self.n_bins_var1 > 96) and (self.n_bins_var1 <= 120):
            n_col += 4
            figsize = (18, 9)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
        ax = fig.add_subplot(gs[0, 0])
        ax = self._plot_bins(ax=ax, n_col=n_col, emphasize_lines=emphasize_lines, **kwargs)
        fig.tight_layout()
        fig.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)

    def _plot_bins(self, ax, emphasize_lines, n_col, **kwargs):
        colors = plt.cm.coolwarm(np.linspace(0.1, 1, self.n_bins_var1))
        for ix, m in enumerate(self.binmedians.keys()):
            lw = 5 if emphasize_lines else 3
            ax.plot(self.binmedians[m][self.var2_col], self.binmedians[m][self.var3_col],
                    ls='-', lw=lw, ms=14, label=m, color=colors[ix],
                    mec='k', mew=1, alpha=1, zorder=99, **kwargs)
            ax.errorbar(x=self.binmedians[m][self.var2_col],
                        y=self.binmedians[m][self.var3_col],
                        xerr=[
                            self.binmedians[m]['xerror_neg'],
                            self.binmedians[m]['xerror_pos']
                        ],
                        yerr=[
                            self.binmedians[m]['yerror_neg'],
                            self.binmedians[m]['yerror_pos']
                        ],
                        elinewidth=8, ecolor=colors[ix], alpha=.3, lw=0)
            if emphasize_lines:
                ax.plot(self.binmedians[m][self.var2_col], self.binmedians[m][self.var3_col],
                        ls='-', lw=2, ms=0, label=None, color='black',
                        mec='k', mew=1, alpha=1, zorder=99)

        n_vals_var3 = self.df[self.var3_col].count()
        n_vals_datapoint = n_vals_var3 / self.n_bins_var1
        n_vals_datapoint = int(n_vals_datapoint / self.n_subbins_var2)
        txt_perc = " percentile " if self.convert_to_percentiles else " "

        txt = (f"showing medians with interquartile range\n"
               f"{n_vals_var3}{txt_perc}values of {self.var3_col}\n"
               f"in {self.n_subbins_var2}{txt_perc}classes of {self.var2_col},\n"
               f"separate for {self.n_bins_var1}{txt_perc}classes of {self.var1_col}\n"
               f"= {n_vals_datapoint} values per data point")
        # n_vals = self.df.groupby(self.var1_group_col).count().mean()[self.var1_col]
        # n_vals = int(n_vals / self.n_subbins_var2)
        ax.text(0.02, 0.98, txt,
                size=theme.AX_LABELS_FONTSIZE, color='k', backgroundcolor='none', transform=ax.transAxes,
                alpha=1, horizontalalignment='left', verticalalignment='top')
        default_format(ax=ax,
                       ax_xlabel_txt=f"{self.var2_col}{txt_perc}",
                       ax_ylabel_txt=f"{self.var3_col}{txt_perc}")

        textsize = theme.FONTSIZE_TXT_LEGEND
        default_legend(ax=ax, ncol=n_col,
                       title=f"{self.n_bins_var1}{txt_perc}classes of {self.var1_col} (median)",
                       loc='upper left',
                       textsize=textsize,
                       bbox_to_anchor=(1, 1.02))
        return ax


def example():
    # Variables
    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    ta_col = 'Tair_f'
    gpp_col = 'GPP_DT_CUT_REF'
    reco_col = 'Reco_DT_CUT_REF'
    # rh_col = 'RH'
    swin_col = 'Rg_f'

    from diive.configs.exampledata import load_exampledata_parquet  # Example data
    data_df = load_exampledata_parquet()
    data_df = data_df.loc[(data_df.index.month >= 6) & (data_df.index.month <= 9)].copy()
    df = data_df[[nee_col, gpp_col, reco_col, vpd_col, ta_col, swin_col]].copy()
    # df = data_df.copy()
    daytime_locs = (df[swin_col] > 50) & (df[ta_col] > 5)
    df = df[daytime_locs].copy()

    rename_dict = {
        ta_col: 'air_temperature',
        vpd_col: 'vapor_pressure_deficit',
        nee_col: 'net_ecosystem_productivity'
    }
    df = df.rename(columns=rename_dict, inplace=False)

    ta_col = 'air_temperature'
    vpd_col = 'vapor_pressure_deficit'
    nee_col = 'net_ecosystem_productivity'
    df = df[[ta_col, vpd_col, nee_col]].copy()
    df[nee_col] = df[nee_col].multiply(-1)

    sbm = SortingBinsMethod(df=df,
                            var1_col=ta_col,
                            var2_col=vpd_col,
                            var3_col=nee_col,
                            n_bins_var1=48,
                            n_subbins_var2=2,
                            convert_to_percentiles=False)
    sbm.calcbins()
    sbm.showplot_decoupling_sbm(marker='o', emphasize_lines=True)

    binmedians = sbm.get_binmeans()
    first = next(iter(binmedians))
    print(binmedians[first])


if __name__ == '__main__':
    example()
