"""

- https://github.com/tblume1992/ThymeBoost
- https://towardsdatascience.com/time-series-outlier-detection-with-thymeboost-ec2046e17458
- https://towardsdatascience.com/thymeboost-a0529353bf34
"""

import matplotlib.pyplot as plt
import pandas as pd
from ThymeBoost import ThymeBoost as tb
from pandas import Series
from pandas.tseries.frequencies import to_offset

from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
def thymeboost(series: Series, flag_missing: Series) -> Series:
    flag_name = f"QCF_OUTLIER_THYME_{series.name}"
    _series = series.copy()

    # Expected values per day for this freq
    num_vals_oneday = int(to_offset('1D') / to_offset(_series.index.freq))

    # Gap-filling w/ running mean, for outlier detection
    num_missing_vals = flag_missing.sum()
    if num_missing_vals > 0:
        rolling_mean = _series.rolling(window=num_vals_oneday, min_periods=1, center=True).mean()
        _series = _series.fillna(rolling_mean)
        if _series.isnull().sum() > 0:
            raise Exception("Thyme Boost outlier removal cannot handle gaps in series.")

    boosted_model = tb.ThymeBoost(normalize_seasonality=True,
                                  verbose=1,
                                  approximate_splits=True,
                                  cost_penalty=.001,
                                  n_rounds=None,
                                  regularization=1.2,
                                  smoothed_trend=True)

    repeat = True
    repeat_run = -1
    flag_outliers = Series(index=_series.index, data=False)
    while repeat:
        repeat_run += 1
        print(f"========================"
              f"Repetition #{repeat_run}"
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

        # Plots
        boosted_model.plot_results(output, figsize=(30, 9))
        boosted_model.plot_components(output, figsize=(16, 9))

        # Outliers
        ix_outliers = output['outliers'] == True  # Outlier indices (locations)
        flag_outliers.loc[ix_outliers] = True
        # outliers = output.loc[ix_outliers, :]  # Get outlier rows
        # all_removed_outliers = all_removed_outliers.append(outliers)  # Collect outliers

        # Replace outlier values w/ predicted value (seasonality + trend)
        # In this step, outliers cannot be removed from the dataset b/c
        # thymeboost needs time series w/o NaN.
        _series.loc[ix_outliers] = output.loc[ix_outliers, 'yhat']

        # Stop outlier removal if no more outliers found
        num_outliers = len(_series.loc[ix_outliers])
        if num_outliers == 0: repeat = False

    num_vals_before = len(series.loc[flag_missing == False])  # From input series
    num_vals_after = len(_series.loc[(flag_missing == False) & (flag_outliers == False)])
    print("Outlier detection finished.")
    print(f"Number of values before outlier removal: {num_vals_before}")
    print(f"Number of values after outlier removal: {num_vals_after}")
    num_outliers_total = num_vals_before - num_vals_after
    outliers_perc = (num_outliers_total / num_vals_before) * 100
    print(f"Number of values identified as outliers: {outliers_perc:.3f}%")

    # Last run did not remove outliers, therefore 'repeat_run - 1'
    print(f"Total outliers removed ({repeat_run - 1} repetitions): {num_outliers_total}")
    print("Removed outliers:")
    print(series.loc[flag_outliers == True])
    flag_outliers.name = flag_name
    return flag_outliers


if __name__ == '__main__':
    pass

    # Testing code

    # Testing MeteoScreeningFromDatabase
    # Example file from dbget output
    testfile = r'L:\Dropbox\luhk_work\20 - CODING\26 - NOTEBOOKS\meteoscreening\test.csv'
    testdata = pd.read_csv(testfile)
    testdata.set_index('TIMESTAMP_END', inplace=True)
    testdata.index = pd.to_datetime(testdata.index)
    series = testdata['TA_M1B1_1.50_1'].copy()
    testdata.plot()
    plt.show()

    series_qc, removed_outliers = thymeboost(_series=series)
    # series_qc
