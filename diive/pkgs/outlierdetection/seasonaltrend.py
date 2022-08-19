"""
SEASONAL TREND DECOMPOSITION (LOESS)

    https://www.statsmodels.org/devel/examples/notebooks/generated/stl_decomposition.html
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
    https://github.com/jrmontag/STLDecompose
    https://github.com/ServiceNow/stl-decomp-4j
    https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
    https://towardsdatascience.com/stl-decomposition-how-to-do-it-from-scratch-b686711986ec
    https://github.com/hafen/stlplus
"""
import pandas as pd
from statsmodels.tsa.seasonal import STL


def calc():
    # https://www.statsmodels.org/devel/examples/notebooks/generated/stl_decomposition.html
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
    # https://github.com/ServiceNow/stl-decomp-4j
    # https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
    # https://towardsdatascience.com/stl-decomposition-how-to-do-it-from-scratch-b686711986ec
    # https://github.com/hafen/stlplus

    self.get_settings_from_fields()
    df = self.var_selected_df.copy()

    # Gapless series needed for STL
    num_gaps = self.var_selected_df[self.target_col].isnull().sum()
    if num_gaps > 0:
        target_series_gf = randomForest.quickfill(df=df.copy(),
                                                  target_col=self.target_col,
                                                  target_gf_col=self.target_gf_col)
        df[self.target_gf_col] = target_series_gf
    else:
        df[self.target_gf_col] = df[self.target_col].copy()

    trend, seasonal, resid = self.decompose(series_gf=df[self.target_gf_col],
                                            loops=0,
                                            period=self.period,
                                            trend=self.trend,
                                            seasonal=self.season,
                                            lowpass_filter=self.lowpass_filter,
                                            trend_deg=self.trend_deg,
                                            seasonal_deg=self.seasonal_deg,
                                            lowpass_deg=self.lowpass_deg)
    df[self.trend_col] = trend
    df[self.seasonal_col] = seasonal
    df[self.resid_col] = resid

    self.var_selected_df = df.copy()

    self.plot_stl_results()

    self.btn_add_as_new_var.setEnabled(True)


def decompose(series_gf, period, trend, lowpass_filter, loops=0, seasonal=7,
              trend_deg=0, seasonal_deg=0, lowpass_deg=0):
    # series_res_gf = series_gf.resample('H').mean()
    trend += 1 if (trend % 2) == 0 else 0  # Trend needs to be odd number
    res = STL(series_gf,
              period=period,
              trend=trend,
              seasonal=seasonal,
              low_pass=lowpass_filter,
              trend_deg=trend_deg,
              seasonal_deg=seasonal_deg,
              low_pass_deg=lowpass_deg).fit()
    # res.plot()
    # # res.seasonal[500:1050].plot()
    # plt.show()
    if loops > 0:
        for i in range(0, loops):
            series = res.observed
            series = series.sub(res.trend)
            series = series.sub(res.seasonal)
            res = STL(series, trend=trend, seasonal=seasonal, low_pass=lowpass_filter,
                      trend_deg=trend_deg, seasonal_deg=seasonal_deg, low_pass_deg=lowpass_deg).fit()
            res.plot()
            plt.show()
    return res.trend, res.seasonal, res.resid


def example_data_to_pickle():
    import pickle
    from pathlib import Path

    from dbc_influxdb import dbcInflux

    # Settings
    BUCKET = 'test'
    MEASUREMENTS = ['TA']
    FIELDS = ['TA_F']
    START = '2020-01-01 00:30:00'
    STOP = '2020-06-01 00:30:00'
    TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # We need returned timestamps in CET (winter time), which is UTC + 1 hour
    DATA_VERSION = 'FLUXNET-WW2020_RELEASE-2022-1'
    DIRCONF = r'L:\Dropbox\luhk_work\20 - CODING\22 - POET\configs'

    # Instantiate class
    dbc = dbcInflux(dirconf=DIRCONF)

    # Data download
    data_simple, data_detailed, assigned_measurements = \
        dbc.download(
            bucket=BUCKET,
            measurements=MEASUREMENTS,
            fields=FIELDS,
            start=START,
            stop=STOP,
            timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
            data_version=DATA_VERSION
        )

    # Store data as pickle for fast data loading during testing
    outfile = Path(r"M:\Downloads\_temp") / "temp.pickle"
    pickle_out = open(outfile, "wb")
    pickle.dump(data_simple, pickle_out)
    pickle_out.close()


def testing_from_pickle():
    import pickle
    from pathlib import Path
    import matplotlib.pyplot as plt

    outfile = Path(r"M:\Downloads\_temp") / "temp.pickle"
    pickle_in = open(outfile, "rb")
    data_simple = pickle.load(pickle_in)
    print(data_simple)

    df = pd.DataFrame(data_simple['TA_F'])
    df = df.asfreq('30T')

    trend, seasonal, resid = decompose(series_gf=df['TA_F'],
                                       loops=0,
                                       period=48,
                                       trend=48 * 21,
                                       trend_deg=1,
                                       seasonal=(48) + 1,
                                       seasonal_deg=1,
                                       lowpass_filter=49,
                                       lowpass_deg=1)
    df['trend'] = trend
    df['seasonal'] = seasonal
    df['residuals'] = resid
    df['reconstructed'] = df['trend'] + df['seasonal']
    df.plot(subplots=True)
    plt.show()

    df2 = pd.DataFrame()
    df2['TA_F'] = df['TA_F'].copy()
    df2['reconstructed_rmedian'] = df['reconstructed'].copy()
    df2['reconstructed_rmedian'] = df2['reconstructed_rmedian'].rolling(48, center=True).median()
    std = df2['reconstructed_rmedian'].rolling(48, center=True).std().multiply(10)
    df2['reconstructed+std'] = df2['reconstructed_rmedian'] + std
    df2['reconstructed-std'] = df2['reconstructed_rmedian'] - std
    df2.plot(subplots=False)
    plt.show()

    # Relative extrema
    # https://stackoverflow.com/questions/57069892/how-to-detect-anomaly-in-a-time-series-dataspecifically-with-trend-and-seasona
    import numpy as np
    from scipy.signal import argrelextrema
    a = argrelextrema(df['residuals'].values, np.greater, order=100)
    df3 = pd.DataFrame()
    df3['residuals'] = df['residuals'].copy()
    df3['residuals_extrema'] = df3['residuals'].iloc[a]

    plt.plot_date(df3['residuals'].index, df3['residuals'])
    plt.plot_date(df3['residuals_extrema'].index, df3['residuals_extrema'])
    plt.show()

    df['residuals'].iloc[a].plot()
    df['residuals'].plot()
    plt.show()


if __name__ == '__main__':
    # example_data_to_pickle()
    testing_from_pickle()
