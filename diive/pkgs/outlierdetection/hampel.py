"""
OUTLIER DETECTION: HAMPEL TEST
==============================

This module is part of the diive library:
https://github.com/holukas/diive

"""
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series
from sktime.transformations.series.outlier_detection import HampelFilter

from diive.core.base.flagbase import FlagBase
from diive.core.plotting.outlier_dtnt import plot_outlier_daytime_nighttime
from diive.core.times.times import DetectFrequency
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag


@ConsoleOutputDecorator()
class HampelDaytimeNighttime(FlagBase):
    flagid = 'OUTLIER_HAMPELDTNT'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 window_length: int = 10,
                 n_sigma: float = 5,
                 k: float = 1.4826,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Identify outliers in a sliding window based on the Hampel filter,
        separately for daytime and nighttime data.

        Args:
            series: Time series in which outliers are identified.
            window_length: Size of sliding window.
            n_sigma: Number of standard deviations. Records with sd outside this value
                are flagged as outliers.
            k: constant scale factor, for Gaussian it is approximately 1.4826.
            lat: Latitude of location as float, e.g. 46.583056
            lon: Longitude of location as float, e.g. 9.790639
            utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
                The datetime index of the resulting Series will be in this timezone.
            idstr: Identifier, added as suffix to output variable names.
            showplot: Show plot with results from the outlier detection.
            verbose: Print more text output.

        Returns:
            Flag series that combines flags from all iterations in one single flag.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.showplot = showplot
        self.verbose = verbose
        self.window_length = window_length
        self.n_sigma = n_sigma
        self.k = k

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

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """

        self._overall_flag, n_iterations = self.repeat(func=self.run_flagtests, repeat=repeat)
        if self.showplot:
            # Default plot for outlier tests, showing rejected values
            self.defaultplot(n_iterations=n_iterations)

            # Collect in dataframe for outlier daytime/nighttime plot
            frame = {
                'UNFILTERED': self.series,
                'CLEANED': self.series[self.overall_flag == 0],
                'OUTLIER': self.series[self.overall_flag == 2],
                'OUTLIER_DAYTIME': self.series[(self.overall_flag == 2) & (self.is_daytime == 1)],
                'OUTLIER_NIGHTTIME': self.series[(self.overall_flag == 2) & (self.is_nighttime == 1)],
                'NOT_OUTLIER_DAYTIME': self.series[(self.overall_flag == 0) & (self.is_daytime == 1)],
                'NOT_OUTLIER_NIGHTTIME': self.series[(self.overall_flag == 0) & (self.is_nighttime == 1)],
            }
            df = pd.DataFrame(frame)
            title = (f"Hampel filter daytime/nighttime: {self.series.name}, "
                     f"n_iterations = {n_iterations}, "
                     f"n_outliers = {self.series[self.overall_flag == 2].count()}")
            plot_outlier_daytime_nighttime(df=df, title=title)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy()
        s = s.dropna()

        flag = pd.Series(index=s.index, data=np.nan)

        # Run for daytime (dt)
        _s_dt = s[self.is_daytime].copy()
        transformer = HampelFilter(window_length=self.window_length,
                                   n_sigma=self.n_sigma,
                                   k=self.k,
                                   return_bool=True)
        is_outlier = transformer.fit_transform(_s_dt)
        _ok_dt = is_outlier == False
        _ok_dt = _ok_dt[_ok_dt].index
        _rejected_dt = is_outlier == True
        _rejected_dt = _rejected_dt[_rejected_dt].index
        # _zscore_dt = funcs.zscore(series=_s_dt)
        # _ok_dt = _zscore_dt <= self.thres_zscore
        # _ok_dt = _ok_dt[_ok_dt].index
        # _rejected_dt = _zscore_dt > self.thres_zscore
        # _rejected_dt = _rejected_dt[_rejected_dt].index

        # Run for nighttime (nt)
        _s_nt = s[self.is_nighttime].copy()
        transformer = HampelFilter(window_length=self.window_length,
                                   n_sigma=self.n_sigma,
                                   k=self.k,
                                   return_bool=True)
        is_outlier = transformer.fit_transform(_s_nt)
        _ok_nt = is_outlier == False
        _ok_nt = _ok_nt[_ok_nt].index
        _rejected_nt = is_outlier == True
        _rejected_nt = _rejected_nt[_rejected_nt].index
        # _s_nt = s[self.is_nighttime].copy()  # Daytime data
        # _zscore_nt = funcs.zscore(series=_s_nt)
        # _ok_nt = _zscore_nt <= self.thres_zscore
        # _ok_nt = _ok_nt[_ok_nt].index
        # _rejected_nt = _zscore_nt > self.thres_zscore
        # _rejected_nt = _rejected_nt[_rejected_nt].index

        # Collect daytime and nighttime flags in one overall flag
        flag.loc[_ok_dt] = 0
        flag.loc[_rejected_dt] = 2
        flag.loc[_ok_nt] = 0
        flag.loc[_rejected_nt] = 2

        n_outliers = (flag == 2).sum()

        ok = (flag == 0)
        ok = ok[ok].index
        rejected = (flag == 2)
        rejected = rejected[rejected].index

        if self.verbose:
            print(f"ITERATION#{iteration}: Total found outliers: "
                  f"{n_outliers} (daytime+nighttime), "
                  f"{len(_rejected_dt)} (daytime), "
                  f"{len(_rejected_nt)} (nighttime)")

        return ok, rejected, n_outliers


@ConsoleOutputDecorator()
class Hampel(FlagBase):
    flagid = 'OUTLIER_HAMPEL'

    def __init__(self,
                 series: Series,
                 window_length: int = 10,
                 n_sigma: float = 5,
                 k: float = 1.4826,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Identify outliers in a sliding window based on the Hampel filter.

        The Hampel filter employs a moving window and utilizes the Median Absolute Deviation (MAD)
        as a measure of data variability. MAD is calculated by taking the median of the absolute
        differences between each data point and the median of the moving window.

        kudos:
        - https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.transformations.series.outlier_detection.HampelFilter.html
        - https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
        - https://medium.com/@miguel.otero.pedrido.1993/hampel-filter-with-python-17db1d265375

        Reference:
            Hampel F. R., “The influence curve and its role in robust estimation”,
            Journal of the American Statistical Association, 69, 382-393, 1974

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier, added as suffix to output variable names.
            window_length: Size of sliding window.
            n_sigma: Number of standard deviations. Records with sd outside this value
                are flagged as outliers.
            k: constant scale factor, for Gaussian it is approximately 1.4826.
            showplot: Show plot with removed data points.
            verbose: More text output to console if *True*.

        Returns:
            Flag series that combines flags from all iterations in one single flag.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.n_sigma = n_sigma
        self.window_length = window_length
        self.showplot = showplot
        self.verbose = verbose
        self.k = k  # Scale factor for Gaussian distribution

        # if self.showplot:
        #     self.fig, self.ax, self.ax2 = self._plot_init()

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """

        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:
            # Default plot for outlier tests, showing rejected values
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy()
        s = s.dropna()

        transformer = HampelFilter(window_length=self.window_length,
                                   n_sigma=self.n_sigma,
                                   k=self.k,
                                   return_bool=True)

        is_outlier = transformer.fit_transform(s)

        ok = is_outlier == False
        ok = ok[ok].index
        rejected = is_outlier == True
        rejected = rejected[rejected].index

        n_outliers = len(rejected)

        if self.verbose:
            if self.verbose:
                print(f"ITERATION#{iteration}: Total found outliers: {len(rejected)} values")

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
    s = s.loc[s.index.month == 7].copy()

    s_noise = add_impulse_noise(series=s,
                                factor_low=-10,
                                factor_high=4,
                                contamination=0.04,
                                seed=42)  # Add impulse noise (spikes)
    s_noise.name = f"{s.name}+noise"
    TimeSeries(s_noise).plot()

    lsd = Hampel(
        series=s_noise,
        n_sigma=4,
        window_length=48 * 9,
        showplot=True,
        verbose=True
    )
    lsd.calc(repeat=True)


def example_dtnt():
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
                                factor_high=4,
                                contamination=0.04,
                                seed=42)  # Add impulse noise (spikes)
    s_noise.name = f"{s.name}+noise"
    TimeSeries(s_noise).plot()
    ham = HampelDaytimeNighttime(
        series=s_noise,
        n_sigma=4,
        window_length=48 * 9,
        showplot=True,
        verbose=True,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1
    )
    ham.calc(repeat=True)


if __name__ == '__main__':
    # example()
    example_dtnt()
