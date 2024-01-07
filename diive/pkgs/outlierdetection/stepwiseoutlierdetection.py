import pandas as pd
from pandas import Series, DataFrame

from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.timeseries import TimeSeries
from diive.core.times.times import TimestampSanitizer
from diive.pkgs.outlierdetection.absolutelimits import AbsoluteLimits, AbsoluteLimitsDaytimeNighttime
from diive.pkgs.outlierdetection.incremental import zScoreIncrements
from diive.pkgs.outlierdetection.local3sd import LocalSD
from diive.pkgs.outlierdetection.lof import LocalOutlierFactorDaytimeNighttime, LocalOutlierFactorAllData
from diive.pkgs.outlierdetection.manualremoval import ManualRemoval
from diive.pkgs.outlierdetection.seasonaltrend import OutlierSTLRZ
from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime, zScore


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
    - `.flag_outliers_localsd_test()`: Identify outliers based on the local standard deviation from a running median
    - `.flag_manualremoval_test()`: Remove data points for range, time or point-by-point
    - `.flag_outliers_stl_rz_test()`: Identify outliers based on seasonal-trend decomposition and z-score calculations
    - `.flag_outliers_zscore_dtnt_test()`: Identify outliers based on the z-score, separately for daytime and nighttime
    - `.flag_outliers_zscore_test()`:  Identify outliers based on the z-score
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

    """

    def __init__(
            self,
            dfin: DataFrame,
            col: str,
            site_lat: float,
            site_lon: float,
            utc_offset: int,
            idstr: str = None
    ):
        self.dfin = dfin.copy()
        self.col = col
        self._series = self.dfin[self.col].copy()
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.utc_offset = utc_offset
        self.idstr = validate_id_string(idstr=idstr)

        # Setup
        self._results, \
            self._series_hires_cleaned, \
            self._series_hires_orig = self._setup()

        # Returned variables
        self._last_results = pd.DataFrame()  # Results of most recent QC tests (objects)

    @property
    def last_results(self) -> DataFrame:
        """Return high-resolution detailed data with tags as dict of objects"""
        if not isinstance(self._last_results, object):
            raise Exception(f"No recent results available.")
        return self._last_results

    @property
    def series_hires_cleaned(self) -> Series:
        """Return cleaned time series of field(s) as dict of Series"""
        if not isinstance(self._series_hires_cleaned, Series):
            raise Exception(f"No hires quality-controlled data available.")
        return self._series_hires_cleaned

    @property
    def series_hires_orig(self) -> Series:
        """Return original time series of field(s) as dict of Series"""
        if not isinstance(self._series_hires_orig, Series):
            raise Exception(f"No hires original data available.")
        return self._series_hires_orig

    @property
    def results(self) -> DataFrame:
        """Return flag(s) as dataframe"""
        if not isinstance(self._results, DataFrame):
            raise Exception(f"No hires flags available.")
        return self._results

    def showplot_orig(self, interactive: bool = False):
        """Show original high-resolution data used as input"""
        p = TimeSeries(series=self._series_hires_orig)
        p.plot() if not interactive else p.plot_interactive()

    def showplot_cleaned(self, interactive: bool = False):
        """Show *current* cleaned high-resolution data"""
        p = TimeSeries(series=self._series_hires_cleaned)
        p.plot() if not interactive else p.plot_interactive()

    def flag_manualremoval_test(self, remove_dates: list, showplot: bool = False, verbose: bool = False):
        """Flag specified records for removal"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = ManualRemoval(series=series_cleaned, idstr=self.idstr, remove_dates=remove_dates,
                                showplot=showplot, verbose=verbose, repeat=False)
        # results.calc(remove_dates=remove_dates, showplot=showplot, verbose=verbose)
        self._last_results = results

    def flag_outliers_zscore_dtnt_test(self, thres_zscore: float = 4, showplot: bool = False, verbose: bool = False,
                                       repeat: bool = True):
        """Flag outliers based on z-score, calculated separately for daytime and nighttime"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = zScoreDaytimeNighttime(series=series_cleaned, lat=self.site_lat, lon=self.site_lon,
                                         utc_offset=self.utc_offset, idstr=self.idstr, repeat=repeat,
                                         thres_zscore=thres_zscore, showplot=showplot, verbose=verbose)
        # results.calc(threshold=threshold, showplot=showplot, verbose=verbose)
        self._last_results = results

    def flag_outliers_localsd_test(self, n_sd: float = 7, winsize: int = None, showplot: bool = False,
                                   verbose: bool = False, repeat: bool = True):
        """Identify outliers based on standard deviation in a rolling window"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = LocalSD(series=series_cleaned, idstr=self.idstr, n_sd=n_sd, winsize=winsize,
                          showplot=showplot, repeat=repeat, verbose=verbose)
        # results.calc(n_sd=n_sd, winsize=winsize, showplot=showplot)
        self._last_results = results

    def flag_outliers_abslim_dtnt_test(self,
                                       daytime_minmax: list[float, float],
                                       nighttime_minmax: list[float, float],
                                       showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits separately for daytime and nighttime"""
        series_cleaned = self._series_hires_cleaned.copy()
        # For setting absolute limits no iteration is necessary, therefore always repeat=False
        results = AbsoluteLimitsDaytimeNighttime(series=series_cleaned, lat=self.site_lat, lon=self.site_lon,
                                                 utc_offset=self.utc_offset, idstr=self.idstr,
                                                 daytime_minmax=daytime_minmax, nighttime_minmax=nighttime_minmax,
                                                 showplot=showplot, repeat=False)
        # results.calc(daytime_minmax=daytime_minmax, nighttime_minmax=nighttime_minmax, showplot=showplot)
        self._last_results = results

    def flag_outliers_abslim_test(self, minval: float, maxval: float, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits"""
        series_cleaned = self._series_hires_cleaned.copy()
        # For setting absolute limits no iteration is necessary, therefore always repeat=False
        results = AbsoluteLimits(series=series_cleaned, idstr=self.idstr,
                                 minval=minval, maxval=maxval, showplot=showplot,
                                 repeat=False)
        # results.calc(min=minval, max=maxval, showplot=showplot)
        self._last_results = results

    def flag_outliers_increments_zcore_test(self, thres_zscore: int = 30, showplot: bool = False, verbose: bool = False,
                                            repeat: bool = True):
        """Identify outliers based on the z-score of record increments"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = zScoreIncrements(series=series_cleaned, idstr=self.idstr, thres_zscore=thres_zscore,
                                   showplot=showplot, verbose=verbose, repeat=repeat)
        # results.calc(threshold=threshold, showplot=showplot, verbose=verbose)
        self._last_results = results

    def flag_outliers_zscore_test(self, thres_zscore: int = 4, showplot: bool = False, verbose: bool = False,
                                  plottitle: str = None, repeat: bool = True):
        """Identify outliers based on the z-score of records"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = zScore(series=series_cleaned, idstr=self.idstr, thres_zscore=thres_zscore, showplot=showplot,
                         verbose=verbose, plottitle=plottitle, repeat=repeat)
        # results.calc(threshold=threshold, showplot=showplot, verbose=verbose, plottitle=plottitle)
        self._last_results = results

    def flag_outliers_stl_rz_test(self, thres_zscore: float = 4.5, decompose_downsampling_freq: str = '1H',
                                  repeat: bool = False, showplot: bool = False):
        """Identify outliers based on seasonal-trend decomposition and z-score calculations"""
        series_cleaned = self._series_hires_cleaned.copy()
        results = OutlierSTLRZ(series=series_cleaned, lat=self.site_lat, lon=self.site_lon,
                               utc_offset=self.utc_offset, idstr=self.idstr,
                               thres_zscore=thres_zscore, decompose_downsampling_freq=decompose_downsampling_freq,
                               repeat=repeat, showplot=showplot)
        # results.calc(zfactor=zfactor, decompose_downsampling_freq=decompose_downsampling_freq,
        #              repeat=repeat, showplot=showplot)
        self._last_results = results

    def flag_outliers_lof_dtnt_test(self, n_neighbors: int = None, contamination: float = None,
                                    showplot: bool = False, verbose: bool = False, repeat: bool=True,
                                    n_jobs: int = 1):
        """Local outlier factor, separately for daytime and nighttime data"""
        series_cleaned = self._series_hires_cleaned.copy()
        # Number of neighbors is automatically calculated if not provided
        n_neighbors = int(len(series_cleaned.dropna()) / 100) if not n_neighbors else n_neighbors
        # Contamination is set automatically unless float is given
        contamination = contamination if isinstance(contamination, float) else 'auto'
        results = LocalOutlierFactorDaytimeNighttime(series=series_cleaned, lat=self.site_lat,
                                                     lon=self.site_lon, utc_offset=self.utc_offset, idstr=self.idstr,
                                                     n_neighbors=n_neighbors, contamination=contamination,
                                                     showplot=showplot, verbose=verbose, repeat=repeat, n_jobs=n_jobs)
        # results.calc(n_neighbors=n_neighbors, contamination=contamination, showplot=showplot, verbose=verbose)
        self._last_results = results

    def flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float = None,
                               showplot: bool = False, verbose: bool = False, repeat: bool=True, n_jobs: int=1):
        """Local outlier factor, across all data"""
        series_cleaned = self._series_hires_cleaned.copy()
        # Number of neighbors is automatically calculated if not provided
        n_neighbors = int(len(series_cleaned.dropna()) / 100) if not n_neighbors else n_neighbors
        # Contamination is set automatically unless float is given
        contamination = contamination if isinstance(contamination, float) else 'auto'
        results = LocalOutlierFactorAllData(series=series_cleaned, idstr=self.idstr,
                                            n_neighbors=n_neighbors, contamination=contamination,
                                            showplot=showplot, verbose=verbose, repeat=repeat,
                                            n_jobs=n_jobs)
        # results.calc(n_neighbors=n_neighbors, contamination=contamination, showplot=showplot, verbose=verbose)
        self._last_results = results

    def addflag(self):
        """Add flag(s) of most recent test to data and update filtered series
        that will be used to continue with the next test."""

        # Name of filtered series in last results is the same as the original name
        self._series_hires_cleaned = self.last_results[self.series_hires_orig.name]

        # Remove filtered series to keep only flag columns
        flags_df = self.last_results.drop(self.series_hires_orig.name, axis=1).copy()

        # Add flag columns to results data
        for flag in flags_df.columns:
            if flag not in self._results.columns:
                self._results[flag] = flags_df[flag].copy()
                print(f"++Added flag column {flag} to flag data")
            else:
                pass  # todo check(?)

        # flag = self._last_results.flag
        # # self._series_hires_cleaned = self._last_results.filteredseries
        # if flag.name not in self._results.columns:
        #     self._results[flag.name] = flag
        # else:
        #     pass
        # print(f"++Added flag column {flag.name} to flag data")

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
