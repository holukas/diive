"""
OUTLIER DETECTION: STEP-WISE OUTLIER DETECTION
==============================================

This module is part of the diive library:
https://github.com/holukas/diive

"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.timeseries import TimeSeries
from diive.core.times.times import TimestampSanitizer
from diive.pkgs.outlierdetection.absolutelimits import AbsoluteLimits, AbsoluteLimitsDaytimeNighttime
from diive.pkgs.outlierdetection.hampel import Hampel, HampelDaytimeNighttime
from diive.pkgs.outlierdetection.incremental import zScoreIncrements
from diive.pkgs.outlierdetection.localsd import LocalSD
from diive.pkgs.outlierdetection.lof import LocalOutlierFactorDaytimeNighttime, LocalOutlierFactorAllData
from diive.pkgs.outlierdetection.manualremoval import ManualRemoval
from diive.pkgs.outlierdetection.trim import TrimLow
from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime, zScore, zScoreRolling


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
    - `.flag_outliers_zscore_dtnt_test()`: Identify outliers based on the z-score, separately for daytime and nighttime
    - `.flag_outliers_zscore_rolling_test()`: Identify outliers based on the rolling z-score
    - `.flag_outliers_zscore_test()`:  Identify outliers based on the z-score
    - `.flag_outliers_lof_dtnt_test()`: Identify outliers based on local outlier factor, daytime nighttime separately
    - `.flag_outliers_lof_test()`: Identify outliers based on local outlier factor, across all data
    - `.flag_outliers_hampel_test()`: Identify outliers in a sliding window based on the Hampel filter
    - `.flag_outliers_hampel_dtnt_test()`: Identify based on the Hampel filter, daytime nighttime separately
    - `.flag_outliers_trim_low_test()`: Remove values below threshold and remove an equal amount of records from high end of data

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
        self._flags, \
            self._series_hires_cleaned, \
            self._series_hires_orig = self._setup()

        # Returned variables
        self._last_flag = pd.DataFrame()  # Flag of most recent QC test

    @property
    def last_flag(self) -> DataFrame:
        """Return flag of most recent QC test."""
        if not isinstance(self._last_flag, object):
            raise Exception(f"No recent results available.")
        return self._last_flag

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
    def flags(self) -> DataFrame:
        """Return flag(s) as dataframe"""
        if not isinstance(self._flags, DataFrame):
            raise Exception(f"No hires flags available.")
        return self._flags

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
        flagtest = ManualRemoval(series=series_cleaned, idstr=self.idstr, remove_dates=remove_dates,
                                 showplot=showplot, verbose=verbose)
        flagtest.calc()
        self._last_flag = flagtest.get_flag()

    def flag_outliers_zscore_dtnt_test(self, thres_zscore: float = 4, showplot: bool = False,
                                       verbose: bool = False, repeat: bool = True):
        """Flag outliers based on z-score, calculated separately for daytime and nighttime"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = zScoreDaytimeNighttime(series=series_cleaned, lat=self.site_lat, lon=self.site_lon,
                                          utc_offset=self.utc_offset, idstr=self.idstr,
                                          thres_zscore=thres_zscore, showplot=showplot, verbose=verbose)
        flagtest.calc(repeat=repeat)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_localsd_test(self, n_sd: float = 7, winsize: int = None, showplot: bool = False,
                                   verbose: bool = False, repeat: bool = True):
        """Identify outliers based on standard deviation in a rolling window"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = LocalSD(series=series_cleaned, idstr=self.idstr, n_sd=n_sd, winsize=winsize,
                           showplot=showplot, verbose=verbose)
        flagtest.calc(repeat=repeat)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_increments_zcore_test(self, thres_zscore: int = 30, showplot: bool = False,
                                            verbose: bool = False, repeat: bool = True):
        """Identify outliers based on the z-score of record increments"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = zScoreIncrements(series=series_cleaned, idstr=self.idstr, thres_zscore=thres_zscore,
                                    showplot=showplot, verbose=verbose)
        flagtest.calc(repeat=repeat)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_trim_low_test(self, trim_daytime: bool = False, trim_nighttime: bool = False,
                                    lower_limit: float = None, showplot: bool = False, verbose: bool = False):
        """Flag values below a given absolute limit as outliers, then flag an
        equal number of datapoints at the high end as outliers."""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = TrimLow(series=series_cleaned, idstr=self.idstr,
                           trim_daytime=trim_daytime, trim_nighttime=trim_nighttime,
                           lat=self.site_lat, lon=self.site_lon, utc_offset=self.utc_offset,
                           lower_limit=lower_limit, showplot=showplot, verbose=verbose)
        flagtest.calc()
        self._last_flag = flagtest.get_flag()

    def flag_outliers_hampel_test(self, window_length: int = 10, n_sigma: float = 5, k: float = 1.4826,
                                  showplot: bool = False, verbose: bool = False, repeat: bool = True):
        """Identify outliers in a sliding window based on the Hampel filter"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = Hampel(series=series_cleaned, idstr=self.idstr,
                          window_length=window_length, n_sigma=n_sigma, k=k,
                          showplot=showplot, verbose=verbose)
        flagtest.calc(repeat=repeat)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_hampel_dtnt_test(self, window_length: int = 10, n_sigma_dt: float = 5,
                                       n_sigma_nt: float = 5, k: float = 1.4826,
                                       showplot: bool = False, verbose: bool = False, repeat: bool = True):
        """Identify outliers in a sliding window based on the Hampel filter for daytime/nighttime"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = HampelDaytimeNighttime(series=series_cleaned, idstr=self.idstr,
                                          lat=self.site_lat, lon=self.site_lon, utc_offset=self.utc_offset,
                                          window_length=window_length, n_sigma_dt=n_sigma_dt, n_sigma_nt=n_sigma_nt,
                                          k=k, showplot=showplot, verbose=verbose)
        flagtest.calc(repeat=repeat)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_zscore_test(self, thres_zscore: int = 4, showplot: bool = False, verbose: bool = False,
                                  plottitle: str = None, repeat: bool = True):
        """Identify outliers based on the z-score of records"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = zScore(series=series_cleaned, idstr=self.idstr, thres_zscore=thres_zscore, showplot=showplot,
                          verbose=verbose, plottitle=plottitle)
        flagtest.calc(repeat=repeat)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_zscore_rolling_test(self, thres_zscore: int = 4, showplot: bool = False, verbose: bool = False,
                                          plottitle: str = None, repeat: bool = True, winsize: int = None):
        """Identify outliers based on the z-score of records"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = zScoreRolling(series=series_cleaned, idstr=self.idstr, thres_zscore=thres_zscore,
                                 showplot=showplot, verbose=verbose, plottitle=plottitle,
                                 winsize=winsize)
        flagtest.calc(repeat=repeat)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_lof_dtnt_test(self, n_neighbors: int = None, contamination: float = None,
                                    showplot: bool = False, verbose: bool = False, repeat: bool = True,
                                    n_jobs: int = 1):
        """Local outlier factor, separately for daytime and nighttime data"""
        series_cleaned = self._series_hires_cleaned.copy()
        # Number of neighbors is automatically calculated if not provided
        n_neighbors = int(len(series_cleaned.dropna()) / 200) if not n_neighbors else n_neighbors
        # Contamination is set automatically unless float is given
        contamination = contamination if isinstance(contamination, float) else 'auto'
        flagtest = LocalOutlierFactorDaytimeNighttime(series=series_cleaned, lat=self.site_lat,
                                                      lon=self.site_lon, utc_offset=self.utc_offset, idstr=self.idstr,
                                                      n_neighbors=n_neighbors, contamination=contamination,
                                                      showplot=showplot, verbose=verbose, n_jobs=n_jobs)
        flagtest.calc(repeat=repeat)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float = None,
                               showplot: bool = False, verbose: bool = False, repeat: bool = True, n_jobs: int = 1):
        """Local outlier factor, across all data"""
        series_cleaned = self._series_hires_cleaned.copy()
        # Number of neighbors is automatically calculated if not provided
        n_neighbors = int(len(series_cleaned.dropna()) / 200) if not n_neighbors else n_neighbors
        # Contamination is set automatically unless float is given
        contamination = contamination if isinstance(contamination, float) else 'auto'
        flagtest = LocalOutlierFactorAllData(series=series_cleaned, idstr=self.idstr,
                                             n_neighbors=n_neighbors, contamination=contamination,
                                             showplot=showplot, verbose=verbose, n_jobs=n_jobs)
        flagtest.calc(repeat=repeat)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_abslim_dtnt_test(self,
                                       daytime_minmax: list[float, float],
                                       nighttime_minmax: list[float, float],
                                       showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits separately for daytime and nighttime"""
        series_cleaned = self._series_hires_cleaned.copy()
        # For setting absolute limits no iteration is necessary, therefore always repeat=False
        flagtest = AbsoluteLimitsDaytimeNighttime(series=series_cleaned, lat=self.site_lat, lon=self.site_lon,
                                                  utc_offset=self.utc_offset, idstr=self.idstr,
                                                  daytime_minmax=daytime_minmax, nighttime_minmax=nighttime_minmax,
                                                  showplot=showplot)
        flagtest.calc(repeat=False)
        self._last_flag = flagtest.get_flag()

    def flag_outliers_abslim_test(self, minval: float, maxval: float, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits"""
        series_cleaned = self._series_hires_cleaned.copy()
        # For setting absolute limits no iteration is necessary, therefore always repeat=False
        flagtest = AbsoluteLimits(series=series_cleaned, idstr=self.idstr,
                                  minval=minval, maxval=maxval, showplot=showplot)
        flagtest.calc()
        self._last_flag = flagtest.get_flag()

    def addflag(self):
        """Add flag of most recent test to data and update filtered series
        that will be used to continue with the next test."""
        flag = self.last_flag.copy()

        # Filter original time series with quality flag from last test
        rejected = flag == 2
        # self._series_hires_cleaned = self.series_hires_orig.copy()
        self._series_hires_cleaned.loc[rejected] = np.nan

        # Add flag column to results data
        if flag.name not in self.flags.columns:
            self._flags[flag.name] = flag.copy()
            print(f"++Added flag column {flag.name} to flag data")
        else:
            # It is possible to re-run an outlier method, which produces a flag
            # with the same name as for the first (original) run. In this case
            # an integer is added to the flag name. For example, if the test
            # z-score daytime/nighttime is run the first time, it produces the flag with the name
            # FLAG_TA_T1_2_1_OUTLIER_ZSCOREDTNT_TEST. When the test is run again
            # (e.g. with different settings) then the name of the flag of this second
            # run is FLAG_TA_T1_2_1_OUTLIER_ZSCOREDTNT_2_TEST, etc ...
            new_flagname = flag.name
            rerun = 1
            while new_flagname in self.flags.columns:
                rerun += 1
                new_flagname = flag.name.replace('_TEST', f'_{rerun}_TEST')
            self._flags[new_flagname] = flag.copy()
            print(f"++Added flag column {new_flagname} to flag data")

        # # Name of filtered series in last results is the same as the original name
        # self._series_hires_cleaned = self.last_flag[self.series_hires_orig.name]
        # # Remove filtered series to keep only flag columns
        # flags_df = self.last_flag.drop(self.series_hires_orig.name, axis=1).copy()
        #
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
