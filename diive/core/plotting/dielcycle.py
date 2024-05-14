import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

from diive.core.plotting.plotfuncs import set_fig
from diive.core.times.resampling import diel_cycle


class DielCycle:

    def __init__(self, series: Series, agg: str = 'mean'):
        self.series = series
        self.agg = agg

        self.var = self.series.name

        self.fig = None
        self.ax = None
        self.showplot = False
        self.title = None
        self.txt_ylabel_units = None

    def plot(self,
             ax: plt.Axes = None,
             title: str = None,
             color: str = None,
             txt_ylabel_units: str = None,
             mean: bool = True,
             std: bool = True,
             **kwargs):

        self.title = title
        self.txt_ylabel_units = txt_ylabel_units

        # Resample
        diel_cycles_df = diel_cycle(series=self.series,
                                    mean=mean,
                                    std=std)

        self.fig, self.ax, self.showplot = set_fig(ax=ax)

        label = 'mean'
        diel_cycles_df['mean'].plot(ax=self.ax, label=label, color=color, **kwargs)

        label = "mean±1sd"
        self.ax.fill_between(diel_cycles_df.index.values,
                             diel_cycles_df['mean+sd'].values,
                             diel_cycles_df['mean-sd'].values,
                             alpha=.1, zorder=0, color=color, edgecolor='none',
                             label=label)

        self._format(title=title, diel_cycles_df=diel_cycles_df)

        if self.showplot:
            self.fig.show()

    def _format(self, title, diel_cycles_df):
        # title = self.title if self.title else f"{self.series.name} in {self.series.index.freqstr} time resolution"
        from diive.core.plotting.plotfuncs import default_format, format_spines, default_legend, add_zeroline_y
        if title:
            self.ax.set_title(title, color='black', fontsize=24)
        ax_xlabel_txt = "Time"
        default_format(ax=self.ax, ax_xlabel_txt=ax_xlabel_txt, ax_ylabel_txt=self.series.name,
                       txt_ylabel_units=self.txt_ylabel_units,
                       ticks_direction='in', ticks_length=8, ticks_width=2,
                       ax_labels_fontsize=20,
                       ticks_labels_fontsize=20)
        format_spines(ax=self.ax, color='black', lw=1)
        default_legend(ax=self.ax)
        add_zeroline_y(ax=self.ax, data=diel_cycles_df['mean'])
        self.ax.set_xticks(['3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
        self.ax.set_xticklabels([3, 6, 9, 12, 15, 18, 21])
        if self.showplot:
            self.fig.tight_layout()


if __name__ == '__main__':
    # from diive.configs.exampledata import load_exampledata_parquet
    # df = load_exampledata_parquet()
    # series = df['VPD_f'].copy()

    # from diive.core.io.filereader import search_files, MultiDataFileReader
    # filepaths = search_files(searchdirs=fr"F:\CURRENT\HON\2-FLUXRUN\out", pattern='*.csv')
    # orig = MultiDataFileReader(filepaths=filepaths, filetype='EDDYPRO-FLUXNET-CSV-30MIN', output_middle_timestamp=True)
    # origdf = orig.data_df
    # origmeta = orig.metadata_df
    # origfreq = origdf.index.freq  # Original frequency
    # from diive.core.io.files import save_parquet
    # save_parquet(filename="merged", data=origdf, outpath=r"F:\CURRENT\HON\2-FLUXRUN\out")

    from diive.core.io.files import load_parquet

    df = load_parquet(filepath=r"L:\Sync\luhk_work\CURRENT\HON\2-FLUXRUN\out\merged.parquet")

    co2 = df['FC'].copy()
    keep = (df['USTAR'] > 0.07) & (df['FC_SSITC_TEST'] <= 1)
    co2[~keep] = np.nan
    keep = (co2 < 50) & (co2 > -50)
    co2[~keep] = np.nan

    from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime

    zs = zScoreDaytimeNighttime(series=co2,
                                lat=47.418861,
                                lon=8.491361,
                                utc_offset=1,
                                thres_zscore=3,
                                showplot=False)
    zs.calc()
    flag = zs.get_flag()
    remove = flag == 2
    co2[remove] = np.nan
    # series.plot()
    # plt.show()
    keep = (co2.index.month >= 3) & (co2.index.month <= 5)
    co2[~keep] = np.nan

    dc_co2 = DielCycle(series=co2)
    title = r'$\mathrm{CO_2\ flux}$ (Mar-May 2024)'
    units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
    dc_co2.plot(ax=None, color='#388E3C', title=title, txt_ylabel_units=units)

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

    # import matplotlib.gridspec as gridspec
    # fig = plt.figure(facecolor='white', figsize=(14, 14), dpi=200)
    # gs = gridspec.GridSpec(2, 1)  # rows, cols
    # gs.update(wspace=0.3, hspace=0.4, left=0.08, right=0.93, top=0.93, bottom=0.07)
    # ax = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[1, 0])
    # dc_co2 = DielCycle(series=co2)
    # title = r'$\mathrm{CO_2-Austausch}$ (März-Mai 2024)'
    # units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
    # dc_co2.plot(ax=ax, color='#388E3C', title=title, txt_ylabel_units=units)
    # dc_h2o = DielCycle(series=h2o)
    # title = r'$\mathrm{Wasserdampf-Austausch}$ (März-Mai 2024)'
    # units = r'($\mathrm{mmol\ H_2O\ m^{-2}\ s^{-1}}$)'
    # # [mmol+1s-1m-2]
    # dc_h2o.plot(ax=ax2, color='#1976D2', title=title, txt_ylabel_units=units)
    # fig.show()
