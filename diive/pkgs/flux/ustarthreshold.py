import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.createvar.daynightflag import daytime_nighttime_flag_from_swinpot


@ConsoleOutputDecorator()
class UstarThresholdConstantScenarios:
    """
    Check impact of different constant USTAR thresholds on available data

    Constant means that the threshold is the same for all data, e.g. the same for
    all years.
    ...

    Methods:
        calc(ustarthresholds:list=None, showplot: bool = False, verbose: bool = False):
            Creates timeseries of *series* in after application of different USTAR
            thresholds given in *ustarthresholds*.

    Properties:
        scenariosdf: DataFrame of the timeseries of *series* in different USTAR
            scenarios. Records of *series* where USTAR was below the respective
            threshold are set to missing.

    """

    def __init__(self, series: Series, ustar: Series, swinpot: Series):
        self.series = series
        self.ustar = ustar
        self.swinpot = swinpot
        self.showplot = False
        self.verbose = False

        self._scenariosdf = None

        # Detect daytime and nighttime from potential radiation
        self.daytime, self.nighttime = \
            daytime_nighttime_flag_from_swinpot(swinpot=swinpot, nighttime_threshold=50)

        # Convert 0/1 flags to False/True flag
        self.daytime = self.daytime == 1
        self.nighttime = self.nighttime == 1

    @property
    def scenariosdf(self):
        """Return timeseries of *series* of each USTAR threshold, values
        below the respective threshold were removed"""
        if not isinstance(self._scenariosdf, DataFrame):
            raise Exception(f'USTAR scenarios are empty. '
                            f'Solution: run .calc() to create USTAR scenarios for {self.series.name}.')
        return self._scenariosdf

    def calc(self, ustarthresholds: list = None, showplot: bool = False, verbose: bool = False):
        """Calculate flag"""
        if ustarthresholds is None:
            ustarthresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.showplot = showplot
        self.verbose = verbose
        # self.reset()

        self._scenariosdf = pd.DataFrame(self.series).copy()

        # Create timeseries for each USTAR threshold
        for u in ustarthresholds:
            suffix = f"CUT{u}"
            colname = f"{self.series.name}_{suffix}"
            series_cut = self.series.copy()
            series_cut.loc[self.ustar < u] = np.nan
            self._scenariosdf[colname] = series_cut

        # Get daytime and nighttime data separately
        _scenariosdf_daytime = self._scenariosdf.loc[self.daytime].copy()
        _scenariosdf_nighttime = self._scenariosdf.loc[self.nighttime].copy()

        # total_potential = len(self._scenariosdf)

        if self.showplot: self._plot(daytimedf=_scenariosdf_daytime, nighttimedf=_scenariosdf_nighttime)

    def _plot(self, daytimedf, nighttimedf):
        # Count available records for each USTAR threshold
        counts = self._scenariosdf.describe().loc['count']
        counts_daytime = daytimedf.describe().loc['count']
        counts_nighttime = nighttimedf.describe().loc['count']

        # Create new figure
        fig = plt.figure(facecolor='white', figsize=(12, 16))
        gs = gridspec.GridSpec(4, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.1, left=0.1, right=0.98, top=0.95, bottom=0.05)
        ax_dtnt = fig.add_subplot(gs[0, 0])
        ax_dt = fig.add_subplot(gs[1, 0], sharex=ax_dtnt)
        ax_nt = fig.add_subplot(gs[2, 0], sharex=ax_dtnt)
        ax_stacked = fig.add_subplot(gs[3, 0], sharex=ax_dtnt)

        # Generate bar plots
        bar_dtnt = ax_dtnt.bar(counts.index, counts, label='daytime + nighttime', width=.9, fc='#9CCC65')
        bar_dt = ax_dt.bar(counts_daytime.index, counts_daytime, label='daytime', width=.9, fc='#FFA726')
        bar_nt = ax_nt.bar(counts_nighttime.index, counts_nighttime, label='nighttime', width=.9, fc='#42A5F5')

        # Show text in bar plots
        axes_lst = [ax_dtnt, ax_dt, ax_nt]
        counts_lst = [counts, counts_daytime, counts_nighttime]
        bar_lst = [bar_dtnt, bar_dt, bar_nt]
        for ix, a in enumerate(axes_lst):
            self._bartxt(ax=axes_lst[ix], counts=counts_lst[ix], bar=bar_lst[ix])
            pf.default_legend(ax=a)
            plt.setp(a.get_xticklabels(), visible=False)

        # Stacked bar
        bar_stacked = ax_stacked.bar(counts_nighttime.index, counts_nighttime,
                                     width=.9, label='nighttime', fc='#42A5F5')
        bar_stacked = ax_stacked.bar(counts_daytime.index, counts_daytime,
                                     width=.9, bottom=counts_nighttime, label='daytime', fc='#FFA726')
        pf.default_legend(ax=ax_stacked)

        # Figure title
        title = "Available values after applying different constant USTAR thresholds"
        fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
        fig.text(0.5, 0.01, 'USTAR thresholds', ha='center', size=20)
        fig.text(0.02, 0.5, 'Available values', va='center', rotation='vertical', size=20)

        fig.show()

    def _bartxt(self, ax, counts, bar):
        counts_perc = counts.div(counts[0]).multiply(100)
        for ix, rect in enumerate(bar):
            height = rect.get_height()
            # Number of values
            ax.text(rect.get_x() + rect.get_width() / 2.0, height,
                    f'{height:.0f}', size=16, ha='center', va='bottom')
            # Percentage
            ax.text(rect.get_x() + rect.get_width() / 2.0, height / 2,
                    f'{counts_perc[ix]:.0f}%', size=20, ha='center', va='bottom')

        # _bottom = np.nan  # Needed for stacking multiple bars on top of each other
        #
        #         for flag, row in _plot_df.iterrows():
        #             _flag_counts = _plot_df.loc[flag].replace(np.nan, 0)  # Needs 0 for correct counts
        #             if flag == 0:
        #                 ax.bar(labels, _flag_counts, width=0.8, label=flag)
        #                 for bar_ix, bar in enumerate(ax.patches):
        #                     # kudos: https://www.pythoncharts.com/matplotlib/stacked-bar-charts-labels/
        #
        #                     # Show flag 0 (best) counts in plot
        #                     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
        #                             f"{round(bar.get_height())}", ha='center', color='w', weight='bold', size=6)
        #
        #                     # Show labels *inside* plot
        #                     ax.text(bar.get_x() + bar.get_width() / 10, bar.get_y() / 2,
        #                             f"{labels[bar_ix]}", ha='left', color='white', weight='bold', size=7, rotation=90)
        #
        #                 _bottom = _flag_counts
        #             else:
        #                 ax.bar(labels, _flag_counts, width=0.8, bottom=_bottom, label=flag)
        #                 _bottom = _flag_counts + _bottom

        # ok, rejected = self._flagtests(threshold=threshold)
        # self.setflag(ok=ok, rejected=rejected)
        # self.setfiltered(rejected=rejected)


def example():
    # Load data
    from diive.core.io.files import load_pickle
    df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data.pickle")
    ust = UstarThresholdConstantScenarios(series=df['FC'],
                                          swinpot=df['SW_IN_POT'],
                                          ustar=df['USTAR'])
    ust.calc(ustarthresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], showplot=True, verbose=True)


if __name__ == '__main__':
    example()
