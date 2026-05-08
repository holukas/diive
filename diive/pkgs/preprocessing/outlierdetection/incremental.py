"""
Outlier detection using z-score of record increments.

This module provides outlier detection based on abrupt changes between consecutive values.
The method calculates z-scores for three types of increments (forward, backward, combined)
and flags values where all three exceed the threshold.

Quality flags:
  - flag=0: Value within acceptable range (valid)
  - flag=2: Value detected as outlier (removed)
  - NaN: Original missing data preserved

See examples/outlierdetection/incremental.py for working examples.

This module is part of the diive library:
https://github.com/holukas/diive
"""
from pandas import Series, DatetimeIndex

from diive.core.dfun.stats import double_diff_absolute
from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.preprocessing.outlierdetection.zscore import zScore


@ConsoleOutputDecorator()
class zScoreIncrements(FlagBase):
    """Identify outliers based on z-score of record increments.

    The algorithm detects outliers by analyzing abrupt changes between consecutive values.
    Three types of increments are calculated for each record:

    1. **Forward increment:** absolute difference from previous value
    2. **Backward increment:** absolute difference to next value
    3. **Combined increment:** sum of forward and backward increments

    Z-scores are calculated for each increment type, and values are flagged as outliers
    only when ALL THREE increments exceed the z-score threshold. This approach is robust
    to isolated spikes while allowing gradual changes.

    Example:
        See `examples/outlierdetection/incremental.py` for complete examples.
    """

    flagid = 'OUTLIER_INCRZ'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 thres_zscore: float = 4,
                 showplot: bool = False,
                 verbose: bool = False):
        """Initialize z-score increments outlier detection.

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier, added as suffix to output variable names.
            thres_zscore: Threshold for z-score. Scores above this value are flagged.
                Z-scores are calculated from increments between consecutive records.
                Default 4 is conservative; lower values detect more outliers.
            showplot: If True, displays visualization of detected outliers.
            verbose: If True, prints iteration statistics to console.

        Returns:
            Flag series that combines flags from all iterations in one single flag.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.verbose = False
        self.thres_zscore = thres_zscore
        self.showplot = showplot
        self.verbose = verbose

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """

        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        s = self.filteredseries.copy()

        doublediff_abs, diff_to_prev_abs, diff_to_next_abs = double_diff_absolute(s=s)

        # Run z-score test for all three diff series
        flag_collect = Series(index=doublediff_abs.index, data=doublediff_abs)
        for diff_ix, diff in enumerate([doublediff_abs, diff_to_prev_abs, diff_to_next_abs]):
            flagtest_zscore = zScore(series=diff, thres_zscore=self.thres_zscore,
                                     plottitle=f"z-score of {self.series.name} increments",
                                     showplot=False, verbose=False)
            flagtest_zscore.calc(repeat=False)
            flag_zscore = flagtest_zscore.get_flag()
            if diff_ix == 0:
                flag_collect = flag_zscore.copy()
            else:
                flag_collect = flag_collect.add(flag_zscore)


        # import matplotlib.pyplot as plt
        # flag_collect.plot()
        # plt.show()

        # increment.name = 'INCREMENT'
        # Run z-score test on increments and get resulting flag
        # flagtest_zscore = zScore(series=increment, thres_zscore=self.thres_zscore,
        #                          plottitle=f"z-score of {self.series.name} increments",
        #                          showplot=False, verbose=False)
        # flagtest_zscore.calc(repeat=False)
        # flag_zscore = flagtest_zscore.get_flag()

        # import pandas as pd
        # import matplotlib.pyplot as plt
        # df = pd.DataFrame(
        #     {
        #         'series': s,
        #         'doublediff_abs': doublediff_abs,
        #         'flag_zscore': flag_collect,
        #     }
        # )
        # df.plot(subplots=True)
        # plt.show()

        ok = flag_collect < 6
        ok = ok[ok].index
        rejected = flag_collect == 6  # z-score flags for all three diffs are 2 and 3*2=6
        rejected = rejected[rejected].index
        n_outliers = len(rejected)

        if self.verbose:
            print(
                f"ITERATION#{iteration}: Total found outliers: {n_outliers} values (daytime+nighttime)")

        return ok, rejected, n_outliers


