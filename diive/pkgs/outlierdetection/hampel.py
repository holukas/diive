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
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.outlierdetection.common import create_daytime_nighttime_flags


@ConsoleOutputDecorator()
class HampelDaytimeNighttime(FlagBase):
    flagid = 'OUTLIER_HAMPELDTNT'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 window_length: int = 10,
                 n_sigma_dt: float = 5.5,
                 n_sigma_nt: float = 5.5,
                 k: float = 1.4826,
                 use_differencing: bool = True,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Identify outliers in a sliding window based on the Hampel filter,
        separately for daytime and nighttime data.

        Args:
            series: Time series in which outliers are identified.
            window_length: Size of sliding window.
            n_sigma_dt: Number of standard deviations allowed for daytime data.
                Daytime records with sd outside this value are flagged as outliers.
            n_sigma_nt: Number of standard deviations allowed for nighttime data.
                Nighttime records with sd outside this value are flagged as outliers.
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
        self.showplot = showplot
        self.verbose = verbose
        self.window_length = window_length
        self.n_sigma_dt = n_sigma_dt
        self.n_sigma_nt = n_sigma_nt
        self.k = k
        self.use_differencing = use_differencing

        # Detect daytime and nighttime
        self.flag_daytime, flag_nighttime, self.is_daytime, self.is_nighttime = (
            create_daytime_nighttime_flags(timestamp_index=self.series.index, lat=lat, lon=lon, utc_offset=utc_offset))

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
            title = (f"Hampel filter daytime/nighttime: {self.series.name}, "
                     f"n_iterations = {n_iterations}, "
                     f"n_outliers = {self.series[self.overall_flag == 2].count()}")
            self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                flag_quality=self.overall_flag, title=title)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag using optimized Pandas operations."""

        # Prepare data
        s = self.filteredseries.copy().dropna()

        # 2. Transform data
        if self.use_differencing:
            # d = (x_t - x_{t-1}) - (x_{t+1} - x_t)
            s_to_test = s.diff() - s.diff().shift(-1)
            s_to_test = s_to_test.fillna(0)
        else:
            s_to_test = s

        # Calculate rolling stats (vectorized on the whole series)
        # This is much faster than splitting day/night first
        rolling_median = s_to_test.rolling(window=self.window_length, center=True, min_periods=1).median()

        # MAD (Median Absolute Deviation) calculation
        # deviations = | data - rolling_median |
        deviations = np.abs(s_to_test - rolling_median)
        # rolling_mad = median(deviations)
        rolling_mad = deviations.rolling(window=self.window_length, center=True, min_periods=1).median()

        # Define thresholds vectorized
        # Create a series of thresholds matching the data index
        # Default to nighttime threshold
        thresholds = pd.Series(data=self.n_sigma_nt, index=s_to_test.index)
        # Overwrite daytime indices with daytime threshold
        thresholds.loc[self.is_daytime] = self.n_sigma_dt

        # Detect outliers
        # Limit = k * MAD * n_sigma
        # k = 1.4826 (scaling factor for Gaussian consistency)
        limit = self.k * rolling_mad * thresholds

        upper_bound = rolling_median + limit
        lower_bound = rolling_median - limit

        is_outlier = (s_to_test > upper_bound) | (s_to_test < lower_bound)

        # Formatting for return
        # Get indices of True/False
        ok = is_outlier[~is_outlier].index
        rejected = is_outlier[is_outlier].index
        n_outliers = len(rejected)

        # Apply to flag (optional step for your specific return format logic)
        # Note: FlagBase handles the actual '2' assignment
        # based on the returned 'rejected' index list, so we just return indices

        if self.verbose:
            # 1. Align the global daytime mask to the current (possibly dropna'd) working series
            # fill_value=False ensures dropped rows don't count
            is_daytime_aligned = self.is_daytime.reindex(is_outlier.index, fill_value=False)

            # 2. Calculate counts using .sum() (counts Trues) instead of .count() (counts rows)
            n_dt = (is_outlier & is_daytime_aligned).sum()
            n_nt = n_outliers - n_dt

            print(f"ITERATION#{iteration}: Total found outliers: "
                  f"{n_outliers} (total), "
                  f"{n_dt} (daytime), "
                  f"{n_nt} (nighttime)")

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
    # s = s.loc[s.index.month == 7].copy()
    s_noise = add_impulse_noise(series=s,
                                factor_low=-10,
                                factor_high=4,
                                contamination=0.04,
                                seed=42)  # Add impulse noise (spikes)
    s_noise.name = f"{s.name}+noise"
    TimeSeries(s_noise).plot()
    ham = HampelDaytimeNighttime(
        series=s_noise,
        n_sigma_dt=4,
        n_sigma_nt=2,
        window_length=48 * 7,
        showplot=True,
        verbose=True,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1
    )
    ham.calc(repeat=False)


def example_cha():
    SOURCEDIR = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-cha_flux_product\dataset_ch-cha_flux_product\notebooks\30_MERGE_DATA"
    FILENAME = r"33.5_CH-CHA_IRGA+QCL+LGR+M10+MGMT_Level-1_eddypro_fluxnet_2005-2024.parquet"
    from pathlib import Path
    FILEPATH = Path(SOURCEDIR) / FILENAME
    print(f"Data will be loaded from the following file:\n{FILEPATH}")
    from diive.core.io.files import load_parquet
    maindf = load_parquet(filepath=FILEPATH)
    series = maindf['FC'].copy()
    # series = series[series.index.year == 2015].copy()
    # series = series[series.index.month == 6].copy()
    ham = HampelDaytimeNighttime(
        series=series,
        n_sigma_dt=5.5,
        n_sigma_nt=5.5,
        window_length=48,
        use_differencing=True,
        showplot=True,
        verbose=True,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1
    )
    ham.calc(repeat=True)


if __name__ == '__main__':
    # example()
    # example_dtnt()
    example_cha()
