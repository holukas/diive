import pandas as pd
from pandas import Series, DataFrame

from diive.core.plotting.timeseries import TimeSeries
from diive.core.times.times import TimestampSanitizer
from diive.pkgs.outlierdetection.absolutelimits import AbsoluteLimits, AbsoluteLimitsDaytimeNighttime
from diive.pkgs.outlierdetection.incremental import zScoreIncrements
from diive.pkgs.outlierdetection.local3sd import LocalSD
from diive.pkgs.outlierdetection.lof import LocalOutlierFactorDaytimeNighttime, LocalOutlierFactorAllData
from diive.pkgs.outlierdetection.manualremoval import ManualRemoval
from diive.pkgs.outlierdetection.missing import MissingValues
from diive.pkgs.outlierdetection.seasonaltrend import OutlierSTLRZ, OutlierSTLRIQRZ
from diive.pkgs.outlierdetection.thymeboost import ThymeBoostOutlier
from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime, zScoreIQR, zScore


class StepwiseOutlierDetection:
    """
    Step-wise outlier detection in time series data

    The class is optimized to work in Jupyter notebooks.

    Quality flags that can be directly created via this class:
    - `.flag_missingvals_test()`: Generate flag that indicates missing records in data
    - `.flag_outliers_abslim_test()`: Generate flag that indicates if values in data are outside the specified range
    - `.flag_outliers_abslim_dtnt_test()`: Generate flag that indicates if daytime and nighttime values in data are
        outside their respectively specified ranges
    - `.flag_outliers_increments_zcore_test()`: Identify outliers based on the z-score of increments
    - `.flag_outliers_localsd_test()`: Identify outliers based on the local standard deviation
    - `.flag_manualremoval_test()`: Remove data points for range, time or point-by-point
    - `.flag_outliers_stl_riqrz_test()`: Identify outliers based on seasonal-trend decomposition and z-score
        calculations, taking the inter-quartile range into account
    - `.flag_outliers_stl_rz_test()`: Identify outliers based on seasonal-trend decomposition and z-score calculations
    - `.flag_outliers_thymeboost_test()`: Identify outliers based on [thymeboost](https://github.com/tblume1992/ThymeBoost)
    - `.flag_outliers_zscore_dtnt_test()`: Identify outliers based on the z-score, separately for daytime and nighttime
    - `.flag_outliers_zscore_test()`:  Identify outliers based on the z-score
    - `.flag_outliers_zscoreiqr_test()`: Identify outliers based on max z-scores in the interquartile range data
    - `.flag_outliers_lof_dtnt_test()`: Identify outliers based on local outlier factor, daytime nighttime separately
    - `.flag_outliers_lof_test()`: Identify outliers based on local outlier factor, across all data

    The class is optimized to work in Jupyter notebooks. Various outlier detection
    methods can be called on-demand. Outlier results are displayed and the user can
    accept the results and proceed, or repeat the step with adjusted method parameters.
    An unlimited amount of tests can be chained together.

    At the end of the screening, an overall flag (`QCF`) can be calculated from ALL
    single flags using the `FlagQCF` class in `diive`. The overall flag can then be
    used to filter the time series.

    **Screening**
    The stepwise meteoscreening allows to perform **step-by-step** outlier removal tests
    on time series data. A preview plot after running a test is shown and the user can
    decide if results are satisfactory or if the same test with different parameters
    should be re-run. Once results are satisfactory, the respective test flag is added
    to the data with `.addflag()`.

    **Modular structure**
    Due to its modular (step-wise) approach, the stepwise screening can be easily adjusted
    to work with any type of time series data.

    **Current methods**
    A listing of currently implemented quality checks and corrections can be found at the top
    of this file.

    """

    def __init__(
            self,
            dataframe: DataFrame,
            col: str,
            site_lat: float,
            site_lon: float,
            timezone_of_timestamp: str
    ):
        self._dataframe_orig = dataframe.copy()
        self._series = self._dataframe_orig[col].copy()
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.timezone_of_timestamp = timezone_of_timestamp

        # Setup
        self._hires_flags, \
            self._series_hires_cleaned, \
            self._series_hires_orig = self._setup()

        # Returned variables
        self._last_results = None  # Results of most recent QC tests (objects)

    @property
    def last_results(self) -> dict:
        """Return high-resolution detailed data with tags as dict of objects"""
        if not isinstance(self._last_results, dict):
            raise Exception(f"No recent results available.")
        return self._last_results

    @property
    def series_hires_cleaned(self) -> dict:
        """Return cleaned time series of field(s) as dict of Series"""
        if not isinstance(self._series_hires_cleaned, dict):
            raise Exception(f"No hires quality-controlled data available.")
        return self._series_hires_cleaned

    @property
    def series_hires_orig(self) -> Series:
        """Return original time series of field(s) as dict of Series"""
        if not isinstance(self._series_hires_orig, Series):
            raise Exception(f"No hires original data available.")
        return self._series_hires_orig

    @property
    def hires_flags(self) -> DataFrame:
        """Return flag(s) as dict of Series"""
        if not isinstance(self._hires_flags, DataFrame):
            raise Exception(f"No hires flags available.")
        return self._hires_flags

    def showplot_orig(self, interactive: bool = False):
        """Show original high-resolution data used as input"""
        p = TimeSeries(series=self._series_hires_orig)
        p.plot() if not interactive else p.plot_interactive()

    def showplot_cleaned(self, interactive: bool = False):
        """Show *current* cleaned high-resolution data"""
        p = TimeSeries(series=self._series_hires_cleaned)
        p.plot() if not interactive else p.plot_interactive()

    def flag_missingvals_test(self):
        """Flag missing values"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = MissingValues(series=series_cleaned)
        results.calc()
        self._last_results = results

    def flag_manualremoval_test(self, remove_dates: list, showplot: bool = False, verbose: bool = False):
        """Flag specified records for removal"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = ManualRemoval(series=series_cleaned)
        results.calc(remove_dates=remove_dates, showplot=showplot, verbose=verbose)
        self._last_results = results

    def flag_outliers_zscore_dtnt_test(self, threshold: float = 4, showplot: bool = False, verbose: bool = False):
        """Flag outliers based on z-score, calculated separately for daytime and nighttime"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = zScoreDaytimeNighttime(series=series_cleaned, lat=self.site_lat, lon=self.site_lon,
                                         timezone_of_timestamp=self.timezone_of_timestamp)
        results.calc(threshold=threshold, showplot=showplot, verbose=verbose)
        self._last_results = results

    def flag_outliers_increments_zcore_test(self, threshold: int = 30, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on the z-score of record increments"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = zScoreIncrements(series=series_cleaned)
        results.calc(threshold=threshold, showplot=showplot, verbose=verbose)
        self._last_results = results

    def flag_outliers_zscoreiqr_test(self, factor: float = 4, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on the z-score of records in the IQR"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = zScoreIQR(series=series_cleaned)
        results.calc(factor=factor, showplot=showplot, verbose=verbose)
        self._last_results = results

    def flag_outliers_zscore_test(self, threshold: int = 4, showplot: bool = False, verbose: bool = False,
                                  plottitle: str = None):
        """Identify outliers based on the z-score of records"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = zScore(series=series_cleaned)
        results.calc(threshold=threshold, showplot=showplot, verbose=verbose, plottitle=plottitle)
        self._last_results = results

    def flag_outliers_thymeboost_test(self, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on thymeboost"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = ThymeBoostOutlier(series=series_cleaned)
        results.calc(showplot=showplot)
        self._last_results = results

    def flag_outliers_localsd_test(self, n_sd: float = 7, winsize: int = None, showplot: bool = False,
                                   verbose: bool = False):
        """Identify outliers based on standard deviation in a rolling window"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = LocalSD(series=series_cleaned)
        results.calc(n_sd=n_sd, winsize=winsize, showplot=showplot)
        self._last_results = results

    def flag_outliers_abslim_test(self, minval: float, maxval: float, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = AbsoluteLimits(series=series_cleaned)
        results.calc(min=minval, max=maxval, showplot=showplot)
        self._last_results = results

    def flag_outliers_abslim_dtnt_test(self,
                                       daytime_minmax: list[float, float],
                                       nighttime_minmax: list[float, float],
                                       showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits separately for daytime and nighttime"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = AbsoluteLimitsDaytimeNighttime(series=series_cleaned, lat=self.site_lat, lon=self.site_lon,
                                                 timezone_of_timestamp=self.timezone_of_timestamp)
        results.calc(daytime_minmax=daytime_minmax, nighttime_minmax=nighttime_minmax, showplot=showplot)
        self._last_results = results

    def flag_outliers_stl_rz_test(self, zfactor: float = 4.5, decompose_downsampling_freq: str = '1H',
                                  repeat: bool = False, showplot: bool = False):
        """Seasonsal trend decomposition with z-score on residuals"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = OutlierSTLRZ(series=series_cleaned, lat=self.site_lat, lon=self.site_lon,
                               timezone_of_timestamp=self.timezone_of_timestamp)
        results.calc(zfactor=zfactor, decompose_downsampling_freq=decompose_downsampling_freq,
                     repeat=repeat, showplot=showplot)
        self._last_results = results

    def flag_outliers_stl_riqrz_test(self, zfactor: float = 4.5, decompose_downsampling_freq: str = '1H',
                                     repeat: bool = False, showplot: bool = False):
        """Seasonsal trend decomposition with z-score on residuals"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = OutlierSTLRIQRZ(series=series_cleaned, lat=self.site_lat, lon=self.site_lon)
        results.calc(zfactor=zfactor, decompose_downsampling_freq=decompose_downsampling_freq,
                     repeat=repeat, showplot=showplot)
        self._last_results = results

    def flag_outliers_lof_dtnt_test(self, n_neighbors: int = None, contamination: float = 'auto',
                                    showplot: bool = False, verbose: bool = False):
        """Local outlier factor, separately for daytime and nighttime data"""
        series_cleaned = self._series_hires_cleaned.copy()
        # Number of neighbors is automatically calculated if not provided
        n_neighbors = int(len(series_cleaned.dropna()) / 100) if not n_neighbors else n_neighbors
        # Contamination is set automatically unless float is given
        contamination = contamination if isinstance(contamination, float) else 'auto'
        results = LocalOutlierFactorDaytimeNighttime(series=series_cleaned, site_lat=self.site_lat,
                                                     site_lon=self.site_lon)
        results.calc(n_neighbors=n_neighbors, contamination=contamination, showplot=showplot, verbose=verbose)
        self._last_results = results

    def flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float = 'auto',
                               showplot: bool = False, verbose: bool = False):
        """Local outlier factor, across all data"""
        series_cleaned = self._series_hires_cleaned.copy()
        # Number of neighbors is automatically calculated if not provided
        n_neighbors = int(len(series_cleaned.dropna()) / 100) if not n_neighbors else n_neighbors
        # Contamination is set automatically unless float is given
        contamination = contamination if isinstance(contamination, float) else 'auto'
        results = LocalOutlierFactorAllData(series=series_cleaned)
        results.calc(n_neighbors=n_neighbors, contamination=contamination, showplot=showplot, verbose=verbose)
        self._last_results = results

    def get(self) -> DataFrame:
        """Add outlier flags to full dataframe"""
        print("\nMerging flags with full dataframe:")
        # Flag data
        exportcols = self._hires_flags.columns
        # Remove potentially already existing flag data in full dataframe
        for col in exportcols:
            if col in self._dataframe_orig.columns:
                self._dataframe_orig = self._dataframe_orig.drop(columns=exportcols)
        # exportcols = [col for col in self._hires_flags if col not in self._dataframe_orig.columns]
        # Add flags to full dataframe
        df = pd.concat([self._dataframe_orig, self._hires_flags[exportcols]], axis=1)
        for col in self._hires_flags.columns:
            print(f"++Added flag column {col} as new column to full dataframe")
        return df

    def addflag(self):
        """Add flag of most recent test to data and update filtered series
        that will be used to continue with the next test"""
        flag = self._last_results.flag
        self._series_hires_cleaned = self._last_results.filteredseries
        if flag.name not in self._hires_flags.columns:
            self._hires_flags[flag.name] = flag
        else:
            pass  # todo check
        print(f"++Added flag column {flag.name} to flag data")

    def _setup(self) -> tuple[DataFrame, Series, Series]:
        """Setup data for outlier detection"""
        _series = self._series.copy()  # Data for this field

        # Sanitize timestamp
        _series = TimestampSanitizer(data=_series).get()

        # Initialize hires quality flags
        hires_flags = pd.DataFrame(index=_series.index)

        # Store original timeseries, will be cleaned
        series_hires_cleaned = _series.copy()  # Timeseries

        # Store original timeseries for this field dict, stays the same for later comparisons
        series_hires_orig = _series.copy()

        return hires_flags, series_hires_cleaned, series_hires_orig
