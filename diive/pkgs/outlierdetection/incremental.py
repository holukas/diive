"""
OUTLIER DETECTION: INCREMENTAL
==============================

This module is part of the diive library:
https://github.com/holukas/diive

"""
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.zscore import zScore


@ConsoleOutputDecorator()
class zScoreIncrements(FlagBase):
    flagid = 'OUTLIER_INCRZ'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 thres_zscore: float = 4,
                 showplot: bool = False,
                 verbose: bool = False):
        """Identify outliers based on the z-score of record increments.

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier, added as suffix to output variable names.
            thres_zscore: Threshold for z-score, scores above this value will
                be flagged as outlier. NOTE that in this case the z-scores are
                calculated from the increments between data records in *series*,
                whereby the increment at a point in time t is calculated as:
                (1) increment1(t) = absolute( value(t) - value(t-1) )
                (2) increment2(t) = absolute( value(t) - value(t+1) )
                (3) increment(t) = increment1(t) + increment2(t)
                (4) z-scores are calculated from increment(t)
            showplot: Show plot with results from the outlier detection.
            verbose: Print more text output.

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
        shifted_prev = s.shift(1)
        increment_to_prev = s - shifted_prev
        shifted_next = s.shift(-1)
        increment_to_next = s - shifted_next

        increment = increment_to_prev.abs() + increment_to_next.abs()

        increment.name = 'INCREMENT'

        # Run z-score test on increments and get resulting flag
        flagtest_zscore = zScore(series=increment, thres_zscore=self.thres_zscore,
                                 plottitle=f"z-score of {self.series.name} increments",
                                 showplot=False, verbose=False)
        flagtest_zscore.calc(repeat=False)
        flag_zscore = flagtest_zscore.get_flag()

        # import pandas as pd
        # import matplotlib.pyplot as plt
        # df = pd.DataFrame(
        #     {
        #         'series': s,
        #         # 'shifted_prev': shifted_prev,
        #         'increment_to_prev': increment_to_prev,
        #         # 'shifted_next': shifted_next,
        #         'increment_to_next': increment_to_next,
        #         'increment': increment,
        #         'flag_zscore': flag_zscore,
        #     }
        # )
        # df.plot(subplots=True)
        # plt.show()

        ok = flag_zscore == 0
        ok = ok[ok].index
        rejected = flag_zscore == 2
        rejected = rejected[rejected].index
        n_outliers = len(rejected)

        if self.verbose:
            print(
                f"ITERATION#{iteration}: Total found {increment.name} outliers: {n_outliers} values (daytime+nighttime)")

        return ok, rejected, n_outliers


def example():
    import importlib.metadata
    import diive.configs.exampledata as ed
    from diive.pkgs.createvar.noise import add_impulse_noise
    from diive.core.plotting.timeseries import TimeSeries
    import warnings
    warnings.filterwarnings('ignore')
    version_diive = importlib.metadata.version("diive")
    print(f"diive version: v{version_diive}")
    df = ed.load_exampledata_parquet()
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()
    s = s.loc[s.index.month == 7].copy()
    s_noise = add_impulse_noise(series=s,
                                factor_low=-10,
                                factor_high=3,
                                contamination=0.04)  # Add impulse noise (spikes)
    s_noise.name = f"{s.name}+noise"
    TimeSeries(s_noise).plot()

    zsi = zScoreIncrements(
        series=s_noise,
        thres_zscore=4.5,
        showplot=True,
        verbose=False
    )

    zsi.calc(repeat=True)


if __name__ == '__main__':
    example()
