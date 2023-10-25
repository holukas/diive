"""

- https://github.com/tblume1992/ThymeBoost
- https://towardsdatascience.com/time-series-outlier-detection-with-thymeboost-ec2046e17458
- https://towardsdatascience.com/thymeboost-a0529353bf34
"""
import pandas as pd
from ThymeBoost import ThymeBoost as tb
from pandas import Series, DatetimeIndex
from pandas.tseries.frequencies import to_offset

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.gapfilling.randomforest_ts import QuickFillRFTS


@ConsoleOutputDecorator()
class ThymeBoostOutlier(FlagBase):
    """
    Identify outliers based on thymeboost

    ...

    Methods:
        calc(): Calculates flag

    After running calc, results can be accessed with:
        flag: Series
            Flag series where accepted (ok) values are indicated
            with flag=0, rejected values are indicated with flag=2
        filteredseries: Series
            Data with rejected values set to missing

    kudos:
    - https://www.analyticsvidhya.com/blog/2022/08/outliers-pruning-using-python/
    - https://github.com/tblume1992/ThymeBoost

    """
    flagid = 'OUTLIER_THYME'

    def __init__(self, series: Series, levelid: str = None):
        super().__init__(series=series, flagid=self.flagid, levelid=levelid)
        self.showplot = False
        self.maxiter = 1

    def calc(self, maxiter: int = 1, showplot: bool = False):
        """Calculate flag"""
        self.maxiter = maxiter
        self.showplot = showplot
        self.reset()
        ok, rejected = self._flagtests()
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    @staticmethod
    def _plot(boosted_model, output):
        # Plots
        boosted_model.plot_results(output, figsize=(30, 9))
        boosted_model.plot_components(output, figsize=(16, 9))

    @staticmethod
    def _randomforest_quickfill(series: Series) -> Series:
        # Gapfilling random forest
        _df = pd.DataFrame(series)
        _df = pd.DataFrame(series)
        qf = QuickFillRFTS(df=_df, target_col=series.name)
        qf.fill()
        # print(qf.report())
        series = qf.get_gapfilled_target()
        return series

    def _flagtests(self) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        # Work series
        _series = self.series.copy()

        # Expected values per day for this freq
        num_vals_oneday = int(to_offset('1D') / to_offset(_series.index.freq))

        # Gap-filling w/ running mean, for outlier detection
        # Thyme Boost needs gapless data
        n_missing_vals = _series.isnull().sum()
        if n_missing_vals > 0:
            while n_missing_vals > 0:
                _series = self._randomforest_quickfill(series=_series)
                n_missing_vals = _series.isnull().sum()
        if n_missing_vals > 0:
            raise Exception("Thyme Boost outlier removal cannot handle gaps in series.")

        boosted_model = tb.ThymeBoost(normalize_seasonality=True,
                                      verbose=1,
                                      approximate_splits=True,
                                      cost_penalty=.001,
                                      n_rounds=None,
                                      regularization=1.2,
                                      smoothed_trend=True)

        # Collect flags for good and bad data from all iterations
        ok_coll = pd.Series(index=self.series.index, data=False)
        rejected_coll = pd.Series(index=self.series.index, data=False)

        outliers = True
        iteration = 0
        while outliers and iteration <= self.maxiter:
            iteration += 1
            print(f"========================"
                  f"Repetition #{iteration}"
                  f"========================")
            output = boosted_model.detect_outliers(_series,
                                                   trend_estimator='ses',
                                                   # trend_estimator='linear',
                                                   # trend_estimator=['linear', 'arima'],
                                                   # arima_order=(1, 1, 1),
                                                   # arima_order=[(1, 0, 0), (1, 0, 1), (1, 1, 1)],
                                                   seasonal_estimator='fourier',
                                                   seasonal_period=num_vals_oneday,
                                                   # global_cost='mse',
                                                   global_cost='maicc',
                                                   split_cost='mae',
                                                   fit_type='local',
                                                   window_size=int(num_vals_oneday / 10))

            if self.showplot:
                self._plot(boosted_model, output)

            # Outliers
            ok = output['outliers'] == False  # Non-outlier indices
            ok = ok[ok].index
            ok_coll.loc[ok] = True
            rejected = output['outliers'] == True  # Outlier indices (locations)
            rejected = rejected[rejected].index
            rejected_coll.loc[rejected] = True

            # Replace outlier values w/ predicted value (seasonality + trend)
            # In this step, outliers cannot be removed from the dataset b/c
            # thymeboost needs time series w/o NaN.
            _series.loc[rejected] = output.loc[rejected, 'yhat']

            # Stop outlier removal if no more outliers found
            num_outliers = len(_series.loc[rejected])
            if num_outliers == 0: outliers = False

        # Convert to index
        ok_coll = ok_coll[ok_coll].index
        rejected_coll = rejected_coll[rejected_coll].index

        print(f"Total found outliers: {len(rejected_coll)} values")

        if self.showplot:
            self.plot(ok_coll, rejected_coll,
                      plottitle=f"Outlier detection based on thymeboost results for {self.series.name}")

        return ok_coll, rejected_coll


if __name__ == '__main__':
    pass
