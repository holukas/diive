"""
OUTLIER DETECTION: SEASONAL TREND DECOMPOSITION (LOESS)
=======================================================

This module is part of the diive library:
https://github.com/holukas/diive

Kudos:
    - kudos: https://www.analyticsvidhya.com/blog/2022/08/outliers-pruning-using-python/
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
from diive.pkgs.outlierdetection.repeater import repeater
from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime


@ConsoleOutputDecorator()
@repeater
class OutlierSTLRZ(FlagBase):
    """Identify outliers based on seasonal-trend decomposition and z-score calculations.

    (S)easonal (T)rend decomposition using (L)OESS,
    based on (R)esidual analysis using (Z)-scores.

    """
    flagid = 'OUTLIER_STLRZ'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 idstr: str = None,
                 thres_zscore: float = 4.5,
                 decompose_downsampling_freq: str = '1H',
                 repeat: bool = False,
                 showplot: bool = False):
        """

        Args:
            series: Time series in which outliers are identified.
            lat: Latitude of location as float, e.g. 46.583056
            lon: Longitude of location as float, e.g. 9.790639
            utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
                The datetime index of the resulting Series will be in this timezone.
            idstr: Identifier, added as suffix to output variable names.
            thres_zscore: Threshold for z-score, scores above this value will be flagged
                as outlier. NOTE that the z-scores are calculated based on the residuals
                of *series* values. Residuals are obtained via the seasonal-trend decomposition.
            decompose_downsampling_freq: Downsampling frequency to calculate the seasonal
                and trend decomposition. Used to speed up processing time, especially useful
                for very long (decades) time series.
            repeat: Repeat until no more outliers can be found.
            showplot: Show plot with results from the outlier detection.

        Returns:
            Results dataframe via the @repeater wrapper function, dataframe contains
            the filtered time series and flags from all iterations.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.repeat = True
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset
        self.is_nighttime = self._detect_nighttime()
        self.thres_zscore = thres_zscore
        self.decompose_downsampling_freq = decompose_downsampling_freq
        self.repeat = repeat
        self.showplot = showplot

    def _calc(self):
        """Calculate flag"""
        self.reset()
        ok, rejected = self._flagtests()
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

    def _flagtests(self) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        # Work series
        series = self.series.copy()
        n_available_prev = len(series.dropna())

        # Quick gap-fill for series, needed for decompose
        series_gf, flag_isfilled = self._randomforest_quickfill(series=series)  # Adds suffix _gfRF to varname

        # Decompose gap-filled series
        decompose_df = self._decompose(series=series_gf, decompose_downsampling_freq=self.decompose_downsampling_freq)

        # Detect residual outliers in gap-filled series
        flag = self._detect_residual_outliers(series=decompose_df[f'RESIDUAL_{self.decompose_downsampling_freq}'])

        # Set flag for gap-filled locations to NaN
        flag[flag_isfilled == 1] = np.nan

        # Collect good and bad values
        ok = flag == 0
        ok = ok[ok].index
        rejected = flag == 2
        rejected = rejected[rejected].index
        total_outliers = len(rejected)

        if self.showplot:
            decompose_df[series.name] = series.copy()
            decompose_df[flag.name] = flag.copy()
            decompose_df[flag_isfilled.name] = flag_isfilled.copy()
            decompose_df.plot(subplots=True)
            plt.show()

        return ok, rejected

    def _detect_residual_outliers(self, series: Series):
        """Detect residual outliers separately for daytime and nighttime data"""
        print("Detecting residual outliers ...")
        # Run z-score for daytime and nighttime
        # With repeat=False the results are a dataframe storing the filtered series and
        # one single outlier flag (because repeat=False). The filtered series can be removed,
        # the remaining dataframe then only contains the outlier flag.
        results_df = zScoreDaytimeNighttime(series=series, lat=self.lat, lon=self.lon,
                                            utc_offset=self.utc_offset, idstr=None, repeat=False,
                                            thres_zscore=self.thres_zscore, showplot=self.showplot,
                                            verbose=self.verbose)
        flag = results_df.drop(series.name, axis=1)  # Remove filtered series, flag remains in dataframe
        flag = flag.squeeze()  # Convert dataframe with the single flag column to series
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

        _frame = {f'TREND_{decompose_downsampling_freq}': res.trend,
                  f'SEASONAL_{decompose_downsampling_freq}': res.seasonal,
                  f'RESIDUAL_{decompose_downsampling_freq}': res.resid}
        _frame = pd.DataFrame(_frame)
        _frame = _frame.reindex(series.index, method='nearest')
        # _frame['TREND+SEASONAL'] = _frame['TREND'].add(_frame['SEASONAL'])
        _frame[f'SERIES_GAPFILLED_{decompose_downsampling_freq}'] = series.copy()
        # _frame['RESIDUAL'] = _frame['SERIES'] - _frame['TREND+SEASONAL']
        # _frame.plot(subplots=True)
        # plt.show()
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

    stl._calc(zfactor=1.5, decompose_downsampling_freq='6H')

    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    HeatmapDateTime(series=stl.filteredseries).show()
    HeatmapDateTime(series=stl.flag).show()


if __name__ == '__main__':
    example()
