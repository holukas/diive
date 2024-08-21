"""
OUTLIER DETECTION: TRIM
=======================

This module is part of the diive library:
https://github.com/holukas/diive

"""
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series

from diive.core.base.flagbase import FlagBase
from diive.core.times.times import DetectFrequency
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag


@ConsoleOutputDecorator()
class TrimLow(FlagBase):
    flagid = 'OUTLIER_TRIMLOW'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 trim_daytime: bool = False,
                 trim_nighttime: bool = False,
                 lower_limit: float = None,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Flag values below a given absolute limit as outliers, then flag an
        equal number of datapoints at the high end as outliers.

        For example, if *lower_limit=-3* removes 10 data points from the low
        end of the data, then 10 data points are also removed from the high
        end of the data.

        Based on the trimmed mean approach.

        Args:
            series: Time series in which outliers are identified.
            trim_daytime: *True* if daytime data should be filtered.
            trim_nighttime: *True* if nighttime data should be filtered.
            lower_limit: Value below which values are considered outliers.
            lat: Latitude of location as float, e.g. 46.583056.
                Used to divide data into daytime and nighttime data.
            lon: Longitude of location as float, e.g. 9.790639
                Used to divide data into daytime and nighttime data.
            utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
                The datetime index of the resulting Series will be in this timezone.
                Used to divide data into daytime and nighttime data.
            idstr: Identifier, added as suffix to output variable names.
            showplot: Show plot with results from the outlier detection.
            verbose: Print more text output.

        Returns:
            Flag series where 2=outlier and 0=not outlier.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.showplot = showplot
        self.verbose = verbose
        self.trim_daytime = trim_daytime
        self.trim_nighttime = trim_nighttime
        self.lower_limit = lower_limit

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
        self.flag_daytime = dnf.get_daytime_flag()
        nighttime = dnf.get_nighttime_flag()
        self.is_daytime = self.flag_daytime == 1  # Convert 0/1 flag to False/True flag
        self.is_nighttime = nighttime == 1  # Convert 0/1 flag to False/True flag

    def calc(self):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """

        self._overall_flag, n_iterations = self.repeat(func=self.run_flagtests, repeat=False)
        if self.showplot:
            # Default plot for outlier tests, showing rejected values
            self.defaultplot(n_iterations=n_iterations)
            title = (f"Hampel filter daytime/nighttime: {self.series.name}, "
                     f"n_iterations = {n_iterations}, "
                     f"n_outliers = {self.series[self.overall_flag == 2].count()}")
            self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                flag_quality=self.overall_flag, title=title)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy()
        s = s.dropna()

        flag = pd.Series(index=s.index, data=np.nan)

        if self.trim_daytime:
            _s = s[self.is_daytime].copy()
        elif self.trim_nighttime:
            _s = s[self.is_nighttime].copy()
        else:
            raise ValueError('(!)Either trim_daytime or trim_daytime must be True.')

        s_sorted = _s.sort_values(ascending=False)
        s_sorted_below = s_sorted.loc[s_sorted < self.lower_limit].copy()
        n_vals_below = s_sorted_below.count()
        s_sorted_top = s_sorted.iloc[0:n_vals_below].copy()
        n_vals_top = s_sorted_top.count()
        upper_lim = s_sorted_top.iloc[-1]
        _ok = (_s >= self.lower_limit) & (_s < upper_lim)
        _ok = _ok[_ok].index
        _rejected = (_s <= self.lower_limit) | (_s >= upper_lim)
        _rejected = _rejected[_rejected].index
        # Collect nighttime flag in one overall flag
        flag.loc[_ok] = 0
        flag.loc[_rejected] = 2

        flag = flag.fillna(0)

        n_outliers = (flag == 2).sum()
        ok = (flag == 0)
        ok = ok[ok].index
        rejected = (flag == 2)
        rejected = rejected[rejected].index

        if self.verbose:
            print(f"ITERATION#{iteration}: Total found outliers: "
                  f"{n_outliers}, "
                  f"minimum value: {_s.loc[flag == 0].min()}",
                  f"maximum value: {_s.loc[flag == 0].max()}")

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

    # # Only nighttime data
    # keep = df['Rg_f'] < 50
    # df = df[keep].copy()

    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()
    # s = s.loc[s.index.month == 7].copy()

    s_noise = add_impulse_noise(series=s,
                                factor_low=-10,
                                factor_high=4,
                                contamination=0.04,
                                seed=42)  # Add impulse noise (spikes)
    s_noise.name = f"{s.name}+noise"
    TimeSeries(s_noise).plot()

    lsd = TrimLow(
        series=s_noise,
        trim_daytime=False,
        trim_nighttime=True,
        lower_limit=-75,
        showplot=True,
        verbose=True,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1
    )
    lsd.calc()


if __name__ == '__main__':
    example()
