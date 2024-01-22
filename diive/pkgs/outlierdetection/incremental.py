"""
OUTLIER DETECTION: INCREMENTAL
==============================

This module is part of the diive library:
https://github.com/holukas/diive

"""
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.repeater import repeater
from diive.pkgs.outlierdetection.zscore import zScore


@ConsoleOutputDecorator()
@repeater
class zScoreIncrements(FlagBase):
    """Identify outliers based on the z-score of record increments."""

    flagid = 'OUTLIER_INCRZ'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 thres_zscore: float = 4,
                 showplot: bool = False,
                 verbose: bool = False,
                 repeat: bool = True):
        """

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier, added as suffix to output variable names.
            thres_zscore: Threshold for z-score, scores above this value will
                be flagged as outlier. NOTE that in this case the z-scores are
                calculated from the increments between data records in *series*,
                whereby the increment at a point in time t is simply calculated as:
                increment(t) = value(t) - value(t-1).
            showplot: Show plot with results from the outlier detection.
            verbose: Print more text output.
            repeat: Repeat until no more outliers can be found.

        Returns:
            Results dataframe via the @repeater wrapper function, dataframe contains
            the filtered time series and flags from all iterations.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.thres_zscore = thres_zscore
        self.showplot = showplot
        self.verbose = verbose
        self.repeat = repeat

    def _calc(self):
        """Calculate flag"""
        self.reset()
        ok, rejected = self._flagtests()
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        s = self.series.copy()
        shifted = s.shift(1)
        increment = s - shifted
        increment.name = 'INCREMENT'

        # Run simple z-score
        # With repeat=False the results are a dataframe storing the filtered series and
        # one single outlier flag (because repeat=False). The filtered series can be removed,
        # the remaining dataframe then only contains the outlier flag.
        results_df = zScore(series=increment, thres_zscore=self.thres_zscore,
                            plottitle=f"z-score of {self.series.name} increments",
                            showplot=True, verbose=True, repeat=False)
        flag = results_df.drop(increment.name, axis=1)  # Remove filtered series, flag remains in dataframe
        flag = flag.squeeze()  # Convert dataframe with the single flag column to series

        ok = flag == 0
        ok = ok[ok].index
        rejected = flag == 2
        rejected = rejected[rejected].index
        total_outliers = len(rejected)

        if self.verbose:
            print(f"Total found outliers: {total_outliers} values (daytime+nighttime)")

        if self.showplot:
            self.plot(ok=ok, rejected=rejected,
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
