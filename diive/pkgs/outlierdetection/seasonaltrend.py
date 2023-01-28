"""
SEASONAL TREND DECOMPOSITION (LOESS)


    - https://www.statsmodels.org/dev/examples/notebooks/generated/stl_decomposition.html
    - https://www.statsmodels.org/devel/examples/notebooks/generated/stl_decomposition.html
    - https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
    - https://github.com/jrmontag/STLDecompose
    - https://github.com/ServiceNow/stl-decomp-4j
    - https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
    - https://towardsdatascience.com/stl-decomposition-how-to-do-it-from-scratch-b686711986ec
    - https://github.com/hafen/stlplus
    - https://towardsdatascience.com/how-to-detect-seasonality-outliers-and-changepoints-in-your-time-series-5d0901498cff
    - https://github.com/facebookresearch/Kats
    - https://neptune.ai/blog/anomaly-detection-in-time-series
    - todo https://towardsdatascience.com/multi-seasonal-time-series-decomposition-using-mstl-in-python-136630e67530
    - todo from statsmodels.tsa.st import MSTL
    - todo MSTL is coming in statsmodels v0.14.0!
    - https://towardsdatascience.com/hands-on-unsupervised-outlier-detection-using-machine-learning-with-python-ec599fe5a6b5

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.seasonal import STL

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.createvar.nighttime_latlon import nighttime_flag_from_latlon
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS
from diive.pkgs.outlierdetection.zscore import zScoreIQR


@ConsoleOutputDecorator()
class OutlierSTLRIQRZ(FlagBase):
    """
    Identify outliers based on seasonal-trend decomposition and z-score calculations

    (S)easonal (T)rend decomposition using (L)OESS, based on (R)esidual analysis
    of the (I)nter(Q)uartile (R)ange using (Z)-scores

    ...

    Methods:
        calc(): Calculates flag

    After running calc, results can be accessed with:
        flag: Series
            Flag series where accepted (ok) values are indicated
            with flag=0, rejected values are indicated with flag=2
        filteredseries: Series
            Data with rejected values set to missing

    kudos: https://www.analyticsvidhya.com/blog/2022/08/outliers-pruning-using-python/

    """
    flagid = 'OUTLIER_STLZ'

    def __init__(self, series: Series, lat: float, lon: float, levelid: str = None):
        super().__init__(series=series, flagid=self.flagid, levelid=levelid)
        self.showplot = False
        self.repeat = True
        self.lat = lat
        self.lon = lon
        self.is_nighttime = self._detect_nighttime()

    def calc(self, zfactor: float = 4.5, decompose_downsampling_freq:str='1H',
             repeat: bool = False, showplot: bool = False):
        """Calculate flag"""
        self.showplot = showplot
        self.reset()
        ok, rejected = self._flagtests(zfactor=zfactor, repeat=repeat,
                                       decompose_downsampling_freq=decompose_downsampling_freq)
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _detect_nighttime(self) -> bool:
        """Create nighttime flag"""

        nighttimeflag = nighttime_flag_from_latlon(
            lat=self.lat, lon=self.lon, freq=self.series.index.freqstr,
            start=str(self.series.index[0]), stop=str(self.series.index[-1]),
            timezone_of_timestamp='UTC+01:00', threshold_daytime=0)

        # Reindex to hires timestamp
        nighttime_flag_in_hires = nighttimeflag.reindex(self.series.index, method='nearest')
        nighttime_ix = nighttime_flag_in_hires == 1
        return nighttime_ix

    def _flagtests(self, zfactor: float = 4.5, repeat: bool = True,
                   decompose_downsampling_freq: str = '1H') -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        # Work series
        _series = self.series.copy()
        n_available_prev = len(_series.dropna())

        # Collect flags for good and bad data from all iterations
        ok_coll = pd.Series(index=_series.index, data=False)  # Collects all good flags
        rejected_coll = pd.Series(index=_series.index, data=False)  # Collected all bad flags
        decompose_df = None

        # Repeat multiple times until all outliers removed (untile *outliers=False*)
        outliers = True
        while outliers:

            _series_gf, _flag_gf = self._randomforest_quickfill(series=_series)  # Adds suffix _gfRF to varname
            decompose_df = self._decompose(series=_series_gf, decompose_downsampling_freq=decompose_downsampling_freq)
            _flag_thisstep = self._detect_residual_outliers(series=decompose_df['RESIDUAL'], zfactor=zfactor)

            # Collect good and bad values
            _ok = _flag_thisstep.loc[_flag_thisstep == 0]
            _ok = _ok.loc[_ok.index]
            ok_coll.loc[_ok.index] = True
            _rejected = _flag_thisstep.loc[_flag_thisstep == 2]
            _rejected = _rejected.loc[_rejected.index]
            rejected_coll.loc[_rejected.index] = True

            _series.loc[rejected_coll] = np.nan  # Set rejected to missing

            n_available = len(_series.dropna())  # Number of missing values
            n_newoutliers = n_available_prev - n_available
            if repeat:
                outliers = True if n_newoutliers > 0 else False  # Continue while new outliers found
            else:
                outliers = False  # No repetition if *repeat=False*
            n_available_prev = len(_series.dropna())  # Update number
            print(f"New outliers: {n_newoutliers}")

        # Convert to index
        ok_coll = ok_coll[ok_coll].index
        rejected_coll = rejected_coll[rejected_coll].index

        return ok_coll, rejected_coll

    def _detect_residual_outliers(self, series: Series, zfactor: float = 4.5):
        """Detect residual outliers separately for daytime and nighttime data"""
        print("Detecting residual outliers ...")
        flag = pd.Series(index=series.index, data=np.nan)

        # Nighttime
        _series = series[self.is_nighttime].copy()
        _zscoreiqr = zScoreIQR(series=_series)
        _zscoreiqr.calc(factor=zfactor, showplot=False, verbose=False)
        _flag_nighttime = _zscoreiqr.flag

        # Daytime
        _series = series[~self.is_nighttime].copy()
        _zscoreiqr = zScoreIQR(series=_series)
        _zscoreiqr.calc(factor=zfactor, showplot=False, verbose=False)
        _flag_daytime = _zscoreiqr.flag
        # _series = series[self.is_daytime].copy()
        # _flag_daytime = zscoreiqr(series=_series, factor=2, level=3.2, showplot=False)

        rejected_daytime = _flag_daytime == 2
        rejected_daytime = rejected_daytime[rejected_daytime]
        flag.loc[rejected_daytime.index] = 2

        ok_daytime = _flag_daytime == 0
        ok_daytime = ok_daytime[ok_daytime]
        flag.loc[ok_daytime.index] = 0

        rejected_nighttime = _flag_nighttime == 2
        rejected_nighttime = rejected_nighttime[rejected_nighttime]
        flag.loc[rejected_nighttime.index] = 2

        ok_nighttime = _flag_nighttime == 0
        ok_nighttime = ok_nighttime[ok_nighttime]
        flag.loc[ok_nighttime.index] = 0
        return flag

    def _decompose(self, series: Series, decompose_downsampling_freq: str = '1H'):
        print("Decomposing timeseries ...")
        _series = series.resample(decompose_downsampling_freq).mean()

        # Expected values per day for this freq
        num_vals_oneday = int(to_offset('1D') / to_offset(_series.index.freq))

        period = num_vals_oneday
        seasonal = (num_vals_oneday * 28) + 1
        # trend = None
        trend = (num_vals_oneday * 28) + 1
        low_pass = period + 1
        seasonal_deg = 1
        trend_deg = 1
        low_pass_deg = 1
        robust = False
        seasonal_jump = 1
        trend_jump = 1
        low_pass_jump = 2
        res = STL(
            _series,
            period=period,
            seasonal=seasonal,
            trend=trend,
            low_pass=low_pass,
            seasonal_deg=seasonal_deg,
            trend_deg=trend_deg,
            low_pass_deg=low_pass_deg,
            robust=robust,
            seasonal_jump=seasonal_jump,
            trend_jump=trend_jump,
            low_pass_jump=low_pass_jump
        ).fit()

        _frame = {'TREND': res.trend,
                  'SEASONAL': res.seasonal,
                  'RESIDUAL': res.resid}
        _frame = pd.DataFrame(_frame)
        _frame = _frame.reindex(series.index, method='nearest')
        # _frame['TREND+SEASONAL'] = _frame['TREND'].add(_frame['SEASONAL'])
        _frame['SERIES'] = series.copy()
        # _frame['RESIDUAL'] = _frame['SERIES'] - _frame['TREND+SEASONAL']
        _frame.plot(subplots=True)
        plt.show()
        decompose_df = pd.DataFrame(_frame)

        # todo prophet?

        # frame = {'OBSERVED': res.observed, 'TREND': res.trend, 'SEASONAL': res.seasonal, 'RESIDUAL': res.resid}
        # decompose_df = pd.DataFrame(frame)
        return decompose_df

    def _randomforest_quickfill(self, series: Series) -> tuple[Series, Series]:
        # Gapfilling random forest
        rfts = RandomForestTS(df=pd.DataFrame(series),
                              target_col=series.name,
                              include_timestamp_as_features=True,
                              lagged_variants=None,
                              use_neighbor_years=True,
                              feature_reduction=False,
                              verbose=0)
        rfts.build_models(n_estimators=10,
                          random_state=42,
                          min_samples_split=20,
                          min_samples_leaf=10,
                          n_jobs=-1)
        rfts.gapfill_yrs()
        _gapfilled_df, _gf_results = rfts.get_gapfilled_dataset()
        gapfilled_name = f"{series.name}_gfRF"
        series = _gapfilled_df[gapfilled_name].copy()
        flag_gapfilled_name = f"QCF_{gapfilled_name}"
        flag_gapfilled = _gapfilled_df[flag_gapfilled_name].copy()
        # series.plot()
        # plt.show()
        return series, flag_gapfilled


# # Relative extrema
# # https://stackoverflow.com/questions/57069892/how-to-detect-anomaly-in-a-time-series-dataspecifically-with-trend-and-seasona
# import numpy as np
# from scipy.signal import argrelextrema
# a = argrelextrema(df['residuals'].values, np.greater, order=100)
# df3 = pd.DataFrame()
# df3['residuals'] = df['residuals'].copy()
# df3['residuals_extrema'] = df3['residuals'].iloc[a]
#
# plt.plot_date(df3['residuals'].index, df3['residuals'])
# plt.plot_date(df3['residuals_extrema'].index, df3['residuals_extrema'])
# plt.show()

def example():
    pass


if __name__ == '__main__':
    example()
