import calendar

import matplotlib.pyplot as plt
from pandas import Series, DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import default_format, format_spines, default_legend, add_zeroline_y
from diive.core.plotting.plotfuncs import set_fig
from diive.core.times.resampling import diel_cycle


class DielCycle:

    def __init__(self, series: Series):
        """Plot diel cycles of time series.

        Args:
            series: Time series with datetime index.
                The index must contain date and time info.
        """
        self.series = series

        self.var = self.series.name

        self.fig = None
        self.ax = None
        self.showplot = False
        self.title = None
        self.ylabel = None
        self.txt_ylabel_units = None
        self._diel_cycles_df = None
        self.showgrid = True

    def get_data(self) -> DataFrame:
        return self.diel_cycles_df

    @property
    def diel_cycles_df(self):
        """Return dataframe containing diel cycle aggregates."""
        if not isinstance(self._diel_cycles_df, DataFrame):
            raise Exception(f'No diel cycles dataframe available. Please run .plot() first.')
        return self._diel_cycles_df

    def plot(self,
             ax: plt.Axes = None,
             title: str = None,
             color: str = None,
             txt_ylabel_units: str = None,
             mean: bool = True,
             std: bool = True,
             each_month: bool = False,
             legend_n_col: int = 1,
             ylim: list = None,
             ylabel: str = None,
             showgrid: bool = True,
             **kwargs):

        self.title = title
        self.txt_ylabel_units = txt_ylabel_units
        self.ylabel = ylabel
        self.showgrid = showgrid

        # Resample
        self._diel_cycles_df = diel_cycle(series=self.series,
                                          mean=mean,
                                          std=std,
                                          each_month=each_month)

        months = set(self.diel_cycles_df.index.get_level_values(0).tolist())

        self.fig, self.ax, self.showplot = set_fig(ax=ax)

        counter_plotted = -1
        n_months = len(months)
        alpha = 0.05 if n_months > 10 else 0.1
        auto_color = True if not color else False
        for counter, month in enumerate(months):

            means = self.diel_cycles_df.loc[month]['mean']
            if means.isnull().all():
                continue
            else:
                counter_plotted += 1

            means_add_sd = self.diel_cycles_df.loc[month]['mean+sd']
            means_sub_sd = self.diel_cycles_df.loc[month]['mean-sd']

            if auto_color:
                color = theme.colors_12_months()[counter_plotted]

            # monthstr = calendar.month_name[month] if each_month else 'Mittelwert'
            monthstr = calendar.month_abbr[month] if each_month else 'mean'
            means.plot(ax=self.ax, label=f'{monthstr}', color=color, zorder=99, lw=2, **kwargs)

            # label = "Mittelwert ± 1 Standardabweichung"
            label = None
            # label = "mean±1sd" if counter == 0 else ""
            self.ax.fill_between(means.index.values,
                                 means_add_sd.values,
                                 means_sub_sd.values,
                                 alpha=alpha, zorder=0, color=color, edgecolor='none',
                                 label=label)

        self._format(title=title, legend_n_col=legend_n_col, ylim=ylim)

        if self.showplot:
            self.fig.show()

    def _format(self, title, legend_n_col, ylim):
        # title = self.title if self.title else f"{self.series.name} in {self.series.index.freqstr} time resolution"

        if title:
            self.ax.set_title(title, color='black', fontsize=24)
        # ax_xlabel_txt = "Uhrzeit"
        ax_xlabel_txt = "Time (hours of day)"
        ylabel = self.ylabel if self.ylabel else self.series.name
        default_format(ax=self.ax, ax_xlabel_txt=ax_xlabel_txt, ax_ylabel_txt=ylabel,
                       txt_ylabel_units=self.txt_ylabel_units,
                       ticks_direction='in', ticks_length=8, ticks_width=2,
                       ax_labels_fontsize=20,
                       ticks_labels_fontsize=20,
                       showgrid=self.showgrid)
        format_spines(ax=self.ax, color='black', lw=1)
        default_legend(ax=self.ax, ncol=legend_n_col)
        add_zeroline_y(ax=self.ax, data=self.diel_cycles_df['mean'])
        self.ax.set_xticks(['3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
        # self.ax.set_xticklabels(['3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
        self.ax.set_xticklabels([3, 6, 9, 12, 15, 18, 21])
        self.ax.set_xlim(['0:00', '23:59:59'])
        self.ax.set_ylim(ylim)
        if self.showplot:
            self.fig.tight_layout()


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    series = df['NEE_CUT_REF_f'].copy()
    dc = DielCycle(series=series)
    title = r'$\mathrm{Mean\ CO_2\ flux\ (2013-2024)}$'
    units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
    dc.plot(ax=None, title=title, txt_ylabel_units=units,
            each_month=True, legend_n_col=2)

    # from diive.core.io.filereader import search_files, MultiDataFileReader
    # filepaths = search_files(searchdirs=fr"F:\CURRENT\HON\2-FLUXRUN\out", pattern='*.csv')
    # orig = MultiDataFileReader(filepaths=filepaths, filetype='EDDYPRO-FLUXNET-CSV-30MIN', output_middle_timestamp=True)
    # origdf = orig.data_df
    # origmeta = orig.metadata_df
    # origfreq = origdf.index.freq  # Original frequency
    # from diive.core.io.files import save_parquet
    # save_parquet(filename="merged", data=origdf, outpath=r"F:\CURRENT\HON\2-FLUXRUN\out")

    # import numpy as np
    # from diive.core.io.files import load_parquet
    # df = load_parquet(filepath=r"L:\Sync\luhk_work\CURRENT\HON\2-FLUXRUN\out\merged.parquet")
    #
    # # CO2
    # co2 = df['FC'].copy()
    # keep = (df['USTAR'] > 0.07) & (df['FC_SSITC_TEST'] <= 1)
    # co2[~keep] = np.nan
    # keep = (co2 < 50) & (co2 > -50)
    # co2[~keep] = np.nan
    # from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime
    # zs = zScoreDaytimeNighttime(series=co2,
    #                             lat=47.418861,
    #                             lon=8.491361,
    #                             utc_offset=1,
    #                             thres_zscore=3,
    #                             showplot=False)
    # zs.calc()
    # flag = zs.get_flag()
    # remove = flag == 2
    # co2[remove] = np.nan
    # keep = (co2.index.month >= 3) & (co2.index.month <= 5)
    # co2[~keep] = np.nan
    # title = r'$\mathrm{CO_2\ flux}$ (Mar-May 2024)'
    # units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
    #
    # h2o = df['FH2O'].copy()
    # keep = df['FH2O_SSITC_TEST'] <= 1
    # h2o[~keep] = np.nan
    # keep = (h2o < 50) & (h2o > -50)
    # h2o[~keep] = np.nan
    #
    # from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime
    # zs = zScoreDaytimeNighttime(series=h2o,
    #                             lat=47.418861,
    #                             lon=8.491361,
    #                             utc_offset=1,
    #                             thres_zscore=3,
    #                             showplot=False)
    # zs.calc()
    # flag = zs.get_flag()
    # remove = flag == 2
    # h2o[remove] = np.nan
    # keep = (h2o.index.month >= 3) & (h2o.index.month <= 5)
    # h2o[~keep] = np.nan
    #
    # import matplotlib.gridspec as gridspec
    # fig = plt.figure(facecolor='white', figsize=(14, 14), dpi=200)
    # gs = gridspec.GridSpec(2, 1)  # rows, cols
    # gs.update(wspace=0.3, hspace=0.4, left=0.08, right=0.93, top=0.93, bottom=0.07)
    # ax = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[1, 0])
    # dc_co2 = DielCycle(series=co2)
    # title = r'$\mathrm{CO_{2}\text{-}Austausch\ (März-Mai\ 2024)}$'
    # units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
    # dc_co2.plot(ax=ax, color='#388E3C', title=title, txt_ylabel_units=units)
    # dc_h2o = DielCycle(series=h2o)
    # title = r'$\mathrm{Wasserdampf\text{-}Austausch\ (März-Mai\ 2024)}$'
    # units = r'($\mathrm{mmol\ H_2O\ m^{-2}\ s^{-1}}$)'
    # dc_h2o.plot(ax=ax2, color='#1976D2', title=title, txt_ylabel_units=units)
    # fig.show()


if __name__ == '__main__':
    example()
