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
from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag
from diive.pkgs.gapfilling.randomforest_ts import QuickFillRFTS
from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime


@ConsoleOutputDecorator()
class OutlierSTLRZ(FlagBase):
    """
    Identify outliers based on seasonal-trend decomposition and z-score calculations

    (S)easonal (T)rend decomposition using (L)OESS,
    based on (R)esidual analysis using (Z)-scores

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
    flagid = 'OUTLIER_STLRZ'

    def __init__(self, series: Series, lat: float, lon: float,
                 utc_offset: int, idstr: str = None):
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.repeat = True
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset
        self.is_nighttime = self._detect_nighttime()

    def calc(self, zfactor: float = 4.5, decompose_downsampling_freq: str = '1H',
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

        # Detect nighttime
        dnf = DaytimeNighttimeFlag(
            timestamp_index=self.series.index,
            nighttime_threshold=50,
            lat=self.lat,
            lon=self.lon,
            utc_offset=self.utc_offset)
        nighttimeflag = dnf.get_nighttime_flag()
        # daytime = dnf.get_daytime_flag()

        # Reindex to hires timestamp
        # nighttime_flag_in_hires = nighttimeflag.reindex(self.series.index, method='nearest')
        nighttime_ix = nighttimeflag == 1
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

            # Quick gap-fill for series, needed for decompose
            _series_gf, _flag_gf = self._randomforest_quickfill(series=_series)  # Adds suffix _gfRF to varname

            # Decompose gap-filled series
            decompose_df = self._decompose(series=_series_gf, decompose_downsampling_freq=decompose_downsampling_freq)

            # Detect residual outliers in gap-filled series
            _flag_thisstep = self._detect_residual_outliers(series=decompose_df['RESIDUAL'], zfactor=zfactor)

            # Set flag for gap-filled locations to NaN
            _flag_thisstep[_flag_gf == 1] = np.nan

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
        zscores = zScoreDaytimeNighttime(series=series, lat=self.lat, lon=self.lon,
                                         utc_offset=self.utc_offset)
        zscores.calc(threshold=zfactor, showplot=True, verbose=True)
        return zscores.flag

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

    @staticmethod
    def _randomforest_quickfill(series: Series) -> tuple[Series, Series]:
        _df = pd.DataFrame(series)
        qf = QuickFillRFTS(df=_df, target_col=series.name)
        qf.fill()
        # print(qf.report())
        series = qf.get_gapfilled_target()
        flag_gapfilled = qf.get_flag()
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
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    series = df['Tair_f'].copy()

    stl = OutlierSTLRZ(
        series=series,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1)

    stl.calc(zfactor=1.5, decompose_downsampling_freq='6H')

    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    HeatmapDateTime(series=stl.filteredseries).show()
    HeatmapDateTime(series=stl.flag).show()


if __name__ == '__main__':
    example()
