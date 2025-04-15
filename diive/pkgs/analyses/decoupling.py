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
                 zvar: str,
                 xvar: str,
                 yvar: str,
                 n_bins_z: int = 48,
                 n_bins_x: int = 2,
                 convert_to_percentiles: bool = False,
                 agg: str = 'median',):
        """Investigate binned aggregates of a variable z in binned classes of x and y.

        For example: show median GPP (y) in 5 classes of VPD (x), separate for 10 classes
        of air temperature (z).

        Args:
            df: Dataframe with variables
            zvar: Name of the first binning variable, will be shown as colors (z) in the plot.
            xvar: Name of the second binning variable, will be shown on the x-axis (x) in the plot.
            yvar: Name of the variable of interest, will be shown on the y-axis (y) in the plot.
            n_bins_z: Number of bins for variable *var1_col*. Must be >0 and <= 120.
            n_bins_x: Number of bins for variable *var2_col*. Must be >= 2.
            agg: Aggregation method for binning, e.g. 'median', 'mean'.

        Returns:
            Dict with results stored as dataframes for each bin of *zvar*.

        - Example notebook available in:
            notebooks/Analyses/DecouplingSortingBins.ipynb
        """
        self._df = df.copy().dropna()
        self.zvar = zvar
        self.xvar = xvar
        self.yvar = yvar
        self.agg = agg

        if not (n_bins_z > 0) & (n_bins_z <= 120):
            raise ValueError("n_bins_var1 must be >0 and <= 120.")
        self.n_bins_z = n_bins_z

        if not (n_bins_x >= 2):
            raise ValueError("n_bins_var1 must be >= 2.")
        self.n_bins_x = n_bins_x

        self.convert_to_percentiles = convert_to_percentiles

        self.z_group_col = f"group_{self.zvar}"
        self.x_group_col = f"group_{self.xvar}"

        self._binaggs = {}
        if self.convert_to_percentiles:
            self._convert_to_percentiles()
        self._assignbins()

    @property
    def df(self) -> DataFrame:
        if self._df is None:
            raise Exception('No data available.')
        return self._df

    @property
    def binaggs(self) -> dict:
        if not self._binaggs:
            raise Exception('No binned means available, try to run .calcbins() first.')
        return self._binaggs

    def get_binaggs(self) -> dict:
        """Return dict of dataframes with variable 1 group as key"""
        return self.binaggs

    def _convert_to_percentiles(self):
        self._df = self.df.rank(pct=True).copy()

    def _assignbins(self):
        group, bins = pd.qcut(self.df[self.zvar], q=self.n_bins_z, retbins=True, precision=9, duplicates='drop')

        self._df[self.z_group_col] = group
        # perc_df = self.df[[self.var1_col, self.var2_col, self.var3_col]].copy()

    def calcbins(self):
        # counter = 0
        grouped = self.df.groupby(by=self.z_group_col, observed=True, as_index=True, sort=True, group_keys=True)
        for g, g_df in grouped:
            g_df = g_df.set_index(self.z_group_col)

            label_var1 = g_df[self.zvar].agg(self.agg)
            label_var1 = label_var1.round(decimals=2)
            label_var1 = f"{label_var1}"

            group, bins = pd.qcut(g_df[self.xvar], q=self.n_bins_x, retbins=True, precision=9, duplicates='drop')
            g_df[self.x_group_col] = group

            # Check if the required number of bins was generated
            if len(set(group.tolist())) != self.n_bins_x:
                print(f"(!)WARNING: Unable to produce requested number of bins for {label_var1}, skipped.")
                continue

            aggs = g_df.groupby(by=g_df[self.x_group_col], observed=True, as_index=True, sort=False,
                                   group_keys=True).agg(self.agg)

            # Counts
            counts = g_df.groupby(by=g_df[self.x_group_col], observed=True, as_index=True, sort=False,
                                  group_keys=True).count()
            var3_counts_col = f"{self.yvar}_COUNTS"
            aggs[var3_counts_col] = counts[self.yvar]


            # Percentile 25th
            p25 = g_df.groupby(by=g_df[self.x_group_col], observed=True, as_index=True, sort=False,
                               group_keys=True).quantile(.25)
            var2_p25_col = f"{self.xvar}_P25"
            var3_p25_col = f"{self.yvar}_P25"
            aggs[var2_p25_col] = p25[self.xvar]
            aggs[var3_p25_col] = p25[self.yvar]

            # Percentile 75th
            p75 = g_df.groupby(by=g_df[self.x_group_col], observed=True, as_index=True, sort=False,
                               group_keys=True).quantile(.75)
            var2_p75_col = f"{self.xvar}_P75"
            var3_p75_col = f"{self.yvar}_P75"
            aggs[var2_p75_col] = p75[self.xvar]
            aggs[var3_p75_col] = p75[self.yvar]

            aggs['xerror_neg'] = aggs[self.xvar] - aggs[var2_p25_col]
            aggs['xerror_pos'] = aggs[self.xvar] - aggs[var2_p75_col]
            aggs['yerror_neg'] = aggs[self.yvar] - aggs[var3_p25_col]
            aggs['yerror_pos'] = aggs[self.yvar] - aggs[var3_p75_col]

            aggs['xerror_neg'] = aggs['xerror_neg'].abs()
            aggs['xerror_pos'] = aggs['xerror_pos'].abs()
            aggs['yerror_neg'] = aggs['yerror_neg'].abs()
            aggs['yerror_pos'] = aggs['yerror_pos'].abs()

            aggs = aggs.sort_values(by=self.xvar)
            aggs = aggs.reset_index()
            self._binaggs[label_var1] = aggs

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

        # Figure size and legend number of columns
        n_col = 1  # int(self.n_bins_var1 / 20)
        figsize = (12, 9)
        if (self.n_bins_z > 24) and (self.n_bins_z <= 48):
            n_col += 1
        elif (self.n_bins_z > 48) and (self.n_bins_z <= 72):
            n_col += 2
            figsize = (14, 9)
        elif (self.n_bins_z > 72) and (self.n_bins_z <= 96):
            n_col += 3
            figsize = (16, 9)
        elif (self.n_bins_z > 96) and (self.n_bins_z <= 120):
            n_col += 4
            figsize = (18, 9)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
        ax = fig.add_subplot(gs[0, 0])
        ax = self._plot_bins(ax=ax, n_col=n_col, emphasize_lines=emphasize_lines, **kwargs)
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        fig.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)

    def _plot_bins(self, ax, emphasize_lines, n_col, **kwargs):
        colors = plt.cm.coolwarm(np.linspace(0.1, 1, self.n_bins_z))
        for ix, m in enumerate(self.binaggs.keys()):
            lw = 5 if emphasize_lines else 3
            ax.plot(self.binaggs[m][self.xvar], self.binaggs[m][self.yvar],
                    ls='-', lw=lw, ms=14, label=m, color=colors[ix],
                    mec='k', mew=1, alpha=1, zorder=99, **kwargs)
            ax.errorbar(x=self.binaggs[m][self.xvar],
                        y=self.binaggs[m][self.yvar],
                        xerr=[
                            self.binaggs[m]['xerror_neg'],
                            self.binaggs[m]['xerror_pos']
                        ],
                        yerr=[
                            self.binaggs[m]['yerror_neg'],
                            self.binaggs[m]['yerror_pos']
                        ],
                        elinewidth=8, ecolor=colors[ix], alpha=.3, lw=0)
            if emphasize_lines:
                ax.plot(self.binaggs[m][self.xvar], self.binaggs[m][self.yvar],
                        ls='-', lw=2, ms=0, label=None, color='black',
                        mec='k', mew=1, alpha=1, zorder=99)

        n_vals_var3 = self.df[self.yvar].count()
        n_vals_datapoint = n_vals_var3 / self.n_bins_z
        n_vals_datapoint = int(n_vals_datapoint / self.n_bins_x)
        txt_perc = " percentile " if self.convert_to_percentiles else " "

        # Check number of available bins
        n_bins_var1 = len(self.binaggs)
        if n_bins_var1 != self.n_bins_z:
            n_not_generated = self.n_bins_z - n_bins_var1
        else:
            n_not_generated = 0

        txt = (f"showing {self.agg} with interquartile range\n"
               f"{n_vals_var3}{txt_perc}values of {self.yvar}\n"
               f"in {self.n_bins_x}{txt_perc}classes of {self.xvar},\n"
               f"separate for {n_bins_var1}{txt_perc}classes of {self.zvar}\n"               
               f"= {n_vals_datapoint} values per data point")
        # n_vals = self.df.groupby(self.var1_group_col).count().mean()[self.var1_col]
        # n_vals = int(n_vals / self.n_subbins_var2)
        ax.text(0.98, 0.02, txt,
                size=theme.AX_LABELS_FONTSIZE, color='k', backgroundcolor='none', transform=ax.transAxes,
                alpha=1, horizontalalignment='right', verticalalignment='bottom')
        default_format(ax=ax,
                       ax_xlabel_txt=f"{self.xvar}{txt_perc}",
                       ax_ylabel_txt=f"{self.yvar}{txt_perc}")

        textsize = theme.FONTSIZE_TXT_LEGEND_SMALLER_14
        default_legend(ax=ax, ncol=n_col,
                       title=f"{n_bins_var1}{txt_perc}classes of {self.zvar} ({self.agg}) "
                             f"(not generated: {n_not_generated} classes)",
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
                            zvar=ta_col,
                            xvar=vpd_col,
                            yvar=nee_col,
                            n_bins_z=48,
                            n_bins_x=2,
                            convert_to_percentiles=False)
    sbm.calcbins()
    sbm.showplot_decoupling_sbm(marker='o', emphasize_lines=True)

    binaggs = sbm.get_binaggs()
    first = next(iter(binaggs))
    print(binaggs[first])


if __name__ == '__main__':
    example()
