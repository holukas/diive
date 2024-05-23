"""
OUTLIER DETECTION: INCREMENTAL
==============================

This module is part of the diive library:
https://github.com/holukas/diive

"""
from pandas import Series, DatetimeIndex

from diive.core.dfun.stats import double_diff_absolute
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

        First, several absolute increments are calcualted for each data record at time t:
            (1) increment1(t) = absolute( value(t) - value(t-1) )
            (2) increment2(t) = absolute( value(t) - value(t+1) )
            (3) increment1+2(t) = increment1(t) + increment2(t)

        Second, z-scores are calculated for each of these increments:
            (4) z-scores of increment1(t)
            (5) z-scores of increment2(t)
            (6) z-scores of increment1+2(t)

        Third, all data records where z-score > *thres_zscore* are flagged:
            (7) z-scores of increment1(t) > *thres_zscore* --> flag=2
            (8) z-scores of increment2(t) > *thres_zscore* --> flag=2
            (9) z-scores of increment1+2(t) > *thres_zscore* --> flag=2

        Fourth, all data records where all three increments were flagged are flagged as outlier.
            The sum of three flags in (7), (8) and (9) = 2 + 2 + 2 = 6 = outlier.

        Only data records where all three flags were raised are flagged as outlier.

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier, added as suffix to output variable names.
            thres_zscore: Threshold for z-score, scores above this value will
                be flagged as outlier. NOTE that in this case the z-scores are
                calculated from the increments between data records in *series*.
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


def example():
    import importlib.metadata
    import pandas as pd
    import matplotlib.pyplot as plt
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
                                contamination=0.04,
                                seed=42)  # Add impulse noise (spikes)
    s_noise.name = f"{s.name}+noise"
    TimeSeries(s_noise).plot()

    zsi = zScoreIncrements(
        series=s_noise,
        thres_zscore=5.5,
        showplot=True,
        verbose=False)

    zsi.calc(repeat=True)

    flag = zsi.get_flag()

    frame = {'s': s, 's_noise': s_noise, 'flag': flag}
    checkdf = pd.DataFrame.from_dict(frame)
    good_data = checkdf.loc[checkdf['flag'] == 0]['s_noise']
    rejected_data = checkdf.loc[checkdf['flag'] == 2]['s_noise']

    fig, ax = plt.subplots()
    ax.plot(good_data, color="#42A5F5", label="not an outlier", lw=0, ms=5, marker="o")
    ax.plot(rejected_data, color="red", label="outlier", lw=0, ms=7, marker="X")
    plt.title("Result")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example()
