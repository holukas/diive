from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.zscore import zScore


@ConsoleOutputDecorator()
class zScoreIncrements(FlagBase):
    """
    Identify outliers based on the z-score of record increments
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

    def __init__(self, series: Series, idstr: str = None):
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
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

        s = self.series.copy()
        shifted = s.shift(1)
        increment = s - shifted
        increment.name = 'INCREMENT'
        _zsc = zScore(series=increment)
        _zsc.calc(threshold=threshold, plottitle=f"z-score of {self.series.name} increments",
                  showplot=True, verbose=True)
        ok = _zsc.flag == 0
        ok = ok[ok].index
        rejected = _zsc.flag == 2
        rejected = rejected[rejected].index
        total_outliers = len(rejected)

        if self.verbose:
            print(f"Total found outliers: {total_outliers} values (daytime+nighttime)")

        if self.showplot: self.plot(ok=ok, rejected=rejected,
                                    plottitle=f"Outlier detection based on z-score "
                                              f"from {self.series.name} increments")

        return ok, rejected

    # def _plot(self, ok:DatetimeIndex, rejected:DatetimeIndex, plottitle:str=""):
    #     fig = plt.figure(facecolor='white', figsize=(16, 7))
    #     gs = gridspec.GridSpec(2, 1)  # rows, cols
    #     gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
    #     ax_series = fig.add_subplot(gs[0, 0])
    #     ax_ok = fig.add_subplot(gs[1, 0], sharex=ax_series)
    #     ax_series.plot_date(self.series.index, self.series, label=f"{self.series.name}", color="#42A5F5",
    #                         alpha=.5, markersize=2, markeredgecolor='none')
    #     ax_series.plot_date(self.series[rejected].index, self.series[rejected],
    #                         label="outlier (rejected)", color="#F44336", marker="X", alpha=1,
    #                         markersize=8, markeredgecolor='none')
    #     ax_ok.plot_date(self.series[ok].index, self.series[ok], label=f"OK", color="#9CCC65", alpha=.5,
    #                     markersize=2, markeredgecolor='none')
    #     default_format(ax=ax_series)
    #     default_format(ax=ax_ok)
    #     default_legend(ax=ax_series)
    #     default_legend(ax=ax_ok)
    #     plt.setp(ax_series.get_xticklabels(), visible=False)
    #     fig.suptitle(plottitle, fontsize=theme.FIGHEADER_FONTSIZE)
    #     fig.show()
