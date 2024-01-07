import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

import diive.core.funcs.funcs as funcs
from diive.core.base.flagbase import FlagBase
from diive.core.times.times import DetectFrequency
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag
from diive.pkgs.outlierdetection.repeater import repeater


@ConsoleOutputDecorator()
@repeater
class zScoreDaytimeNighttime(FlagBase):
    """
    Identify outliers based on the z-score, separately for daytime and nighttime
    ...

    Methods:
        calc(factor: float = 4): Calculates flag

    After running calc(), results can be accessed with:
        flag: Series
            Flag series where accepted (ok) values are indicated
            with flag=0, rejected values are indicated with flag=2
        filteredseries: Series
            Data with rejected values set to missing

    """
    flagid = 'OUTLIER_ZSCOREDTNT'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 idstr: str = None,
                 thres_zscore: float = 4,
                 showplot: bool = False,
                 verbose: bool = False,
                 repeat: bool = True):
        """

        Args:
            series: Time series
            lat: Latitude of location as float, e.g. 46.583056
            lon: Longitude of location as float, e.g. 9.790639
            utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
                The datetime index of the resulting Series will be in this timezone.
            idstr: Identifier, added as suffix to output variable names
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.threshold = thres_zscore
        self.showplot = showplot
        self.verbose = verbose
        self.repeat = repeat

        # Make sure time series has frequency
        # Freq is needed for the detection of daytime/nighttime from lat/lon
        if not self.series.index.freq:
            freq = DetectFrequency(index=self.series.index, verbose=True).get()
            self.series = self.series.asfreq(freq)

        # Detect nighttime
        dnf = DaytimeNighttimeFlag(
            timestamp_index=self.series.index,
            nighttime_threshold=50,
            lat=lat,
            lon=lon,
            utc_offset=utc_offset)
        daytime = dnf.get_daytime_flag()
        nighttime = dnf.get_nighttime_flag()
        self.is_daytime = daytime == 1  # Convert 0/1 flag to False/True flag
        self.is_nighttime = nighttime == 1  # Convert 0/1 flag to False/True flag

    def calc(self):
        """Calculate flag"""
        self.reset()
        ok, rejected = self._flagtests()
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        # Working data
        s = self.series.copy().dropna()
        flag = pd.Series(index=self.series.index, data=np.nan)

        # Run for daytime (dt)
        _s_dt = s[self.is_daytime].copy()  # Daytime data
        _zscore_dt = funcs.zscore(series=_s_dt)
        _ok_dt = _zscore_dt <= self.threshold
        _ok_dt = _ok_dt[_ok_dt].index
        _rejected_dt = _zscore_dt > self.threshold
        _rejected_dt = _rejected_dt[_rejected_dt].index

        # Run for nighttime (nt)
        _s_nt = s[self.is_nighttime].copy()  # Daytime data
        _zscore_nt = funcs.zscore(series=_s_nt)
        _ok_nt = _zscore_nt <= self.threshold
        _ok_nt = _ok_nt[_ok_nt].index
        _rejected_nt = _zscore_nt > self.threshold
        _rejected_nt = _rejected_nt[_rejected_nt].index

        # Collect daytime and nighttime flags in one overall flag
        flag.loc[_ok_dt] = 0
        flag.loc[_rejected_dt] = 2
        flag.loc[_ok_nt] = 0
        flag.loc[_rejected_nt] = 2

        total_outliers = (flag == 2).sum()

        ok = (flag == 0)
        ok = ok[ok].index
        rejected = (flag == 2)
        rejected = rejected[rejected].index

        if self.verbose:
            print(f"Total found outliers: {len(_rejected_dt)} values (daytime)")
            print(f"Total found outliers: {len(_rejected_nt)} values (nighttime)")
            print(f"Total found outliers: {total_outliers} values (daytime+nighttime)")

        if self.showplot:
            self.plot(ok, rejected,
                      plottitle=f"Outlier detection based on "
                                f"daytime/nighttime z-scores of {self.series.name}")

        return ok, rejected


@ConsoleOutputDecorator()
@repeater
class zScore(FlagBase):
    """
    Identify outliers based on the z-score of records
    ...

    Methods:
        calc(factor: float = 4): Calculates flag

    After running calc(), results can be accessed with:
        flag: Series
            Flag series where accepted (ok) values are indicated
            with flag=0, rejected values are indicated with flag=2
        filteredseries: Series
            Data with rejected values set to missing

    kudos: https://www.analyticsvidhya.com/blog/2022/08/outliers-pruning-using-python/

    """
    flagid = 'OUTLIER_ZSCORE'

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 thres_zscore: float = 4,
                 showplot: bool = False,
                 plottitle: str = None,
                 verbose: bool = False,
                 repeat: bool = True):
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.plottitle = None
        self.verbose = False
        self.thres_zscore = thres_zscore
        self.showplot = showplot
        self.plottitle = plottitle
        self.verbose = verbose
        self.repeat = repeat

    def calc(self):
        """Calculate flag"""
        self.reset()
        ok, rejected = self._flagtests()
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)

    def _flagtests(self) -> tuple[DatetimeIndex, DatetimeIndex]:
        """Perform tests required for this flag"""

        # Working data
        s = self.series.copy().dropna()

        # Run with threshold
        zscores = funcs.zscore(series=s)
        ok = zscores <= self.thres_zscore
        ok = ok[ok].index
        rejected = zscores > self.thres_zscore
        rejected = rejected[rejected].index

        if self.verbose:
            print(f"Total found outliers: {len(rejected)} values")
        # print(f"z-score of {threshold} corresponds to a prob of {100 * 2 * norm.sf(threshold):0.2f}%")

        if self.showplot:
            self.plot(ok, rejected, plottitle=f"Outlier detection based on z-scores of {self.series.name}")

        return ok, rejected


# @ConsoleOutputDecorator()
# class zScoreIQR(FlagBase):
#     """
#     ::Removed in v0.59.0
#
#     Identify outliers based on max z-scores in the interquartile range data
#
#     Data are divided into 8 groups based on quantiles. Each of the 8 groups
#     have their own respective z-score threshold.
#
#     Then, the z-score threshold is calculated for each group by:
#
#     (1) Dividing each group into 8 subgroups, which corresponds to indexes
#         from 0-7. Subgroups with index 2,3,4 and 5 correspond to the IQR.
#     (2) The z-score is calculated for each data point in subgroups 2-5.
#     (3) The z-score group threshold is calculated from the maximum z-score
#         found in subgroups 2-5, multiplied by *factor*.
#     (4) Group data above the group threshold are marked as outliers.
#
#     ...
#
#     Methods:
#         calc(factor: float = 4): Calculates flag
#
#     After running calc, results can be accessed with:
#         flag: Series
#             Flag series where accepted (ok) values are indicated
#             with flag=0, rejected values are indicated with flag=2
#         filteredseries: Series
#             Data with rejected values set to missing
#
#     kudos: https://www.analyticsvidhya.com/blog/2022/08/outliers-pruning-using-python/
#
#     """
#     flagid = 'OUTLIER_ZSCOREIQR'
#
#     def __init__(self, series: Series, levelid: str = None):
#         super().__init__(series=series, flagid=self.flagid, levelid=levelid)
#         self.showplot = False
#         self.verbose = False
#         self.plottitle_add = None
#
#     def calc(self, factor: float = 4, showplot: bool = False, verbose: bool = False, plottitle_add: str = None):
#         """Calculate flag"""
#         self.showplot = showplot
#         self.verbose = verbose
#         self.plottitle_add = plottitle_add
#         self.reset()
#         ok, rejected = self._flagtests(factor=factor)
#         self.setflag(ok=ok, rejected=rejected)
#         self.setfiltered(rejected=rejected)
#
#     def _flagtests(self, factor: float = 4) -> tuple[DatetimeIndex, DatetimeIndex]:
#         """Perform tests required for this flag"""
#
#         # Collect flags for good and bad data
#         ok_coll = pd.Series(index=self.series.index, data=False)
#         rejected_coll = pd.Series(index=self.series.index, data=False)
#
#         # Working data
#         s = self.series.copy()
#         s = s.dropna()
#
#         # group, bins = pd.cut(s, bins=2, retbins=True, duplicates='drop')
#         group, bins = pd.qcut(s, q=8, retbins=True, duplicates='drop')
#         df = pd.DataFrame(s)
#         df['_GROUP'] = group
#         grouped = df.groupby('_GROUP')
#         for ix, group_df in grouped:
#             vardata = group_df[s.name]
#             mean = np.mean(vardata)
#             sd = np.std(vardata)
#             z_score = np.abs((vardata - mean) / sd)
#             # plt.scatter(z_score.index, z_score)
#             # plt.show()
#             threshold = self._detect_z_threshold_from_iqr(series=vardata, factor=factor, quantiles=8)
#             ok = z_score < threshold
#             ok = ok[ok]
#             ok_coll.loc[ok.index] = True
#             rejected = z_score > threshold
#             rejected = rejected[rejected]
#             rejected_coll.loc[rejected.index] = True
#
#         # Convert to index
#         ok_coll = ok_coll[ok_coll].index
#         rejected_coll = rejected_coll[rejected_coll].index
#
#         if self.verbose: print(f"Total found outliers: {len(rejected_coll)} values")
#         # print(f"z-score of {threshold} corresponds to a prob of {100 * 2 * norm.sf(threshold):0.2f}%")
#
#         if self.showplot:
#             plottitle = f"Outlier detection based on max z-scores " \
#                         f"in the interquartile range data of {self.series.name}"
#             if self.plottitle_add:
#                 plottitle += f" / {self.plottitle_add}"
#             self.plot(ok_coll, rejected_coll,
#                       plottitle=plottitle)
#
#         return ok_coll, rejected_coll
#
#     def _detect_z_threshold_from_iqr(self, series: Series, factor: float = 5, quantiles: int = 8) -> float:
#         # First detect the threshold for the z-value
#         # - Datapoints where z-value > threshold will be marked as outliers
#         # - The threshold is detected from z-values found for the interquartile range
#         #   of the data
#         # Divide data into 8 quantile groups
#         # - This means that we then have groups 0-7
#         # - Groups 2-5 correspond to the IQR
#         #         |- IQR-|
#         # - (0 1) 2 3 4 5 (6 7)
#         group, bins = pd.qcut(series, q=quantiles, retbins=True, duplicates='drop')
#         df = pd.DataFrame(series)
#         df['_GROUP'] = group
#         grouped = df.groupby('_GROUP')
#         _counter = -1
#         zvals_iqr = []
#         for ix, group_df in grouped:
#             _counter += 1
#             if (_counter >= 2) & (_counter <= 5):
#                 vardata = group_df[series.name]
#                 mean = np.mean(vardata)
#                 sd = np.std(vardata)
#                 z_score = np.abs((vardata - mean) / sd)
#                 zvals_iqr.append(z_score.max())
#         threshold = max(zvals_iqr) * factor
#         if self.verbose: print(f"Detected threshold for z-value from IQR data: {threshold} "
#                                f"(max z-value in IQR data multiplied by factor {factor})")
#         return threshold


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    series = df['Tair_f'].copy()

    zdn = zScoreDaytimeNighttime(
        series=series,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1,
        thres_zscore=2,
        showplot=True,
        verbose=True,
        repeat=True)

    # zdn.calc(threshold=1.5, showplot=False, verbose=True, repeat=True)

    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    HeatmapDateTime(series=zdn.filteredseries).show()
    HeatmapDateTime(series=zdn.flag).show()


if __name__ == '__main__':
    example()
