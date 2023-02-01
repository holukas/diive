import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pandas import Series, DatetimeIndex, DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.base.flagbase import FlagBase
from diive.core.plotting.plotfuncs import default_format, default_legend
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.zscore import zScore


@ConsoleOutputDecorator()
class zScoreIncremental(FlagBase):
    """
    Identify outliers based on the z-score of increments
    ...

    Methods:
        calc(threshold: float = 4): Calculates flag

    After running calc(), results can be accessed with:
        flag: Series
            Flag series where accepted (ok) values are indicated
            with flag=0, rejected values are indicated with flag=2
        filteredseries: Series
            Data with rejected values set to missing

    """
    flagid = 'OUTLIER_INCRZ'

    def __init__(self, series: Series, levelid: str = None):
        super().__init__(series=series, flagid=self.flagid, levelid=levelid)
        self.showplot = False
        self.verbose = False

    def calc(self, threshold: float = 4, showplot: bool = False, verbose: bool = False):
        """Calculate flag"""
        self.showplot = showplot
        self.verbose = verbose
        self.reset()
        ok, rejected = self._flagtests(threshold=threshold)
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self, threshold: float = 4) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        # todo hier weiter
        s = self.series.copy()
        shifted = s.shift(1)
        diff = s - shifted
        _zsc = zScore(series=diff)
        _zsc.calc(threshold=threshold, plottitle=f"z-score of {self.series.name} increments",
                  showplot=True, verbose=True)
        ok = _zsc.flag == 0
        ok = ok[ok].index
        rejected = _zsc.flag == 2
        rejected = rejected[rejected].index
        total_outliers = len(rejected)

        if self.verbose:
            print(f"Total found outliers: {total_outliers} values (daytime+nighttime)")

        # if self.showplot: self._plot(df=df)

        return ok, rejected

    def _plot(self, df: DataFrame):
        fig = plt.figure(facecolor='white', figsize=(12, 16))
        gs = gridspec.GridSpec(3, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

        ax1.plot_date(x=df.index, y=df[self.series.name], marker='o', mec='none',
                      alpha=.5, color='black', label="series")

        ax2.plot_date(x=df.index, y=df['CLEANED'], marker='o', mec='none',
                      alpha=.5, label="cleaned series")

        ax3.plot_date(x=df.index, y=df['NOT_OUTLIER_'], marker='o', mec='none',
                      alpha=.5, label="OK daytime")
        ax3.plot_date(x=df.index, y=df['OUTLIER_'], marker='o', mec='none',
                      alpha=.5, color='red', label="outlier daytime")

        default_format(ax=ax1)
        default_format(ax=ax2)
        default_format(ax=ax3)

        default_legend(ax=ax1)
        default_legend(ax=ax2)
        default_legend(ax=ax3)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)

        title = f"Outlier detection - local outlier factor"
        fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
        fig.show()
