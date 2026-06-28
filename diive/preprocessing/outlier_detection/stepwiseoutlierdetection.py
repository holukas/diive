"""
OUTLIER DETECTION: STEP-WISE OUTLIER DETECTION
==============================================

Chain multiple detection methods sequentially for progressive outlier filtering.
Includes Hampel, z-score, LOF, absolute limits, local SD, and manual removal.

Part of the diive library: https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.timeseries import TimeSeries
from diive.core.utils.console import info
from diive.core.times.times import TimestampSanitizer
from diive.preprocessing.outlier_detection.absolutelimits import AbsoluteLimits
from diive.preprocessing.outlier_detection.hampel import Hampel
from diive.preprocessing.outlier_detection.incremental import zScoreIncrements
from diive.preprocessing.outlier_detection.localsd import LocalSD
from diive.preprocessing.outlier_detection.lof import LocalOutlierFactor
from diive.preprocessing.outlier_detection.manualremoval import ManualRemoval
from diive.preprocessing.outlier_detection.trim import TrimLow
from diive.preprocessing.outlier_detection.zscore import zScore, zScoreRolling


class StepwiseOutlierDetection:
    """
    Step-wise outlier detection in time series data

    The class is optimized to work in Jupyter notebooks.

    Quality flags that can be directly created via this class:
    - `.flag_missingvals_test()`: Generate flag that indicates missing records in data
    - `.flag_outliers_abslim_test()`: Generate flag that indicates if values in data are outside the specified range (global or separate day/night)
    - `.flag_outliers_increments_zcore_test()`: Identify outliers based on the z-score of increments
    - `.flag_outliers_localsd_test()`: Identify outliers based on the local standard deviation from a running median (global or separate day/night)
    - `.flag_manualremoval_test()`: Remove data points for range, time or point-by-point
    - `.flag_outliers_zscore_test()`:  Identify outliers based on the z-score (global or separate day/night)
    - `.flag_outliers_zscore_rolling_test()`: Identify outliers based on the rolling z-score
    - `.flag_outliers_lof_test()`: Identify outliers based on local outlier factor (global or separate day/night)
    - `.flag_outliers_hampel_test()`: Identify outliers based on the Hampel filter (global or separate day/night)
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
            idstr: str = None,
            output_middle_timestamp: bool = True
    ):
        """Set up stepwise outlier detection. See the class docstring."""
        self.dfin = dfin.copy()
        self.col = col
        self._series = self.dfin[self.col].copy()
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.utc_offset = utc_offset
        self.idstr = validate_id_string(idstr=idstr)
        # When False, the sanitized index keeps the input timestamp convention
        # (e.g. TIMESTAMP_END) instead of being shifted to the middle of the
        # averaging period. Callers that must align the resulting flags back to
        # an existing dataframe (e.g. the GUI) set this False to avoid an index
        # mismatch on merge.
        self.output_middle_timestamp = output_middle_timestamp

        # Setup
        self._flags, \
            self._series_hires_cleaned, \
            self._series_hires_orig = self._setup()

        # Returned variables
        self._last_flag = pd.DataFrame()  # Flag of most recent QC test
        # Data-unit detection band of the most recent test, as (lower, upper)
        # Series (or (None, None) for tests with no single envelope). Lets a
        # caller (e.g. the GUI) overlay the band that produced a step's removals.
        self._last_bounds: tuple = (None, None)

    @property
    def last_flag(self) -> Series:
        """Return flag of most recent QC test."""
        if not isinstance(self._last_flag, Series) or self._last_flag.empty:
            raise Exception(f"No recent results available. Run an outlier test before accessing the last flag.")
        return self._last_flag

    @property
    def last_bounds(self) -> tuple:
        """``(lower, upper)`` data-unit Series for the most recent test's final
        detection band, or ``(None, None)`` for tests without a single envelope
        (z-score increments, LOF, manual removal, missing values)."""
        return self._last_bounds

    def _record_last(self, flagtest) -> None:
        """Capture the most recent test's flag and (when it exposes one) its
        final data-unit detection band, for caller-side limit-line overlays."""
        self._last_flag = flagtest.get_flag()
        self._last_bounds = (getattr(flagtest, "last_lower_bound", None),
                             getattr(flagtest, "last_upper_bound", None))

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

    def flag_missingvals_test(self, verbose: bool = False):
        """Flag missing records in the data (flag 2 where the value is missing)."""
        # Lazy import: a top-level import would create a circular import via the
        # qaqc package __init__ (qaqc -> meteoscreening -> outlier_detection -> here).
        from diive.preprocessing.qaqc.flags import MissingValues
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = MissingValues(series=series_cleaned, idstr=self.idstr, verbose=verbose)
        flagtest.calc(repeat=False)
        self._record_last(flagtest)

    def flag_manualremoval_test(self, remove_dates: list, showplot: bool = False, verbose: bool = False):
        """Flag specified records for removal"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = ManualRemoval(series=series_cleaned, idstr=self.idstr, remove_dates=remove_dates,
                                 showplot=showplot, verbose=verbose)
        flagtest.calc()
        self._record_last(flagtest)

    def flag_outliers_localsd_test(self, n_sd: float | list = 7, winsize: int | list = None, showplot: bool = False,
                                   constant_sd: bool = False, separate_daytime_nighttime: bool = False,
                                   verbose: bool = False, repeat: bool = True):
        """Identify outliers based on standard deviation in a rolling window"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = LocalSD(series=series_cleaned, idstr=self.idstr, n_sd=n_sd, winsize=winsize,
                           separate_daytime_nighttime=separate_daytime_nighttime, lat=self.site_lat, lon=self.site_lon,
                           utc_offset=self.utc_offset, constant_sd=constant_sd, showplot=showplot, verbose=verbose)
        flagtest.calc(repeat=repeat)
        self._record_last(flagtest)

    def flag_outliers_increments_zcore_test(self, thres_zscore: int = 30, showplot: bool = False,
                                            verbose: bool = False, repeat: bool = True):
        """Identify outliers based on the z-score of record increments"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = zScoreIncrements(series=series_cleaned, idstr=self.idstr, thres_zscore=thres_zscore,
                                    showplot=showplot, verbose=verbose)
        flagtest.calc(repeat=repeat)
        self._record_last(flagtest)

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
        self._record_last(flagtest)

    def flag_outliers_hampel_test(self, window_length: int = 48 * 13, n_sigma: float = 5.5,
                                  n_sigma_daytime: float = None, n_sigma_nighttime: float = None,
                                  k: float = 1.4826, use_differencing: bool = True,
                                  separate_daytime_nighttime: bool = True, showplot: bool = False,
                                  verbose: bool = False, repeat: bool = True):
        """Identify outliers in a sliding window based on the Hampel filter (global or separate day/night).

        Parameters
        ----------
        window_length : int, default 48*13 (=624)
            Size of the sliding window for median/MAD calculation, in record
            counts (not a duration). The default ``48 * 13 = 624`` corresponds
            to 13 days of half-hourly data — matching Papale et al. 2006.
            Scale for other sampling rates (e.g. ``24 * 13`` for hourly).
        n_sigma : float, default 5.5
            Threshold multiplier for global mode (number of MADs above median)
        n_sigma_daytime : float, optional
            Threshold for daytime data (when separate_daytime_nighttime=True)
        n_sigma_nighttime : float, optional
            Threshold for nighttime data (when separate_daytime_nighttime=True)
        k : float, default 1.4826
            Scaling factor for MAD (median absolute deviation)
        use_differencing : bool, default True
            If True, apply Hampel filter to differenced series (rate of change)
        separate_daytime_nighttime : bool, default False
            If False, apply single threshold globally across all records.
            If True, apply separate thresholds for daytime and nighttime data.
        showplot : bool, default False
            If True, display visualization of flagged outliers
        verbose : bool, default False
            If True, print flagging statistics
        repeat : bool, default True
            If True, iteratively repeat detection until convergence
        """
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = Hampel(
            series=series_cleaned, idstr=self.idstr,
            lat=self.site_lat if separate_daytime_nighttime else None,
            lon=self.site_lon if separate_daytime_nighttime else None,
            utc_offset=self.utc_offset if separate_daytime_nighttime else None,
            use_differencing=use_differencing, separate_day_night=separate_daytime_nighttime,
            window_length=window_length, n_sigma=n_sigma, n_sigma_daytime=n_sigma_daytime,
            n_sigma_nighttime=n_sigma_nighttime,
            k=k, showplot=showplot, verbose=verbose)
        flagtest.calc(repeat=repeat)
        self._record_last(flagtest)

    def flag_outliers_zscore_test(self,
                                  thres_zscore: float = 4,
                                  separate_daytime_nighttime: bool = False,
                                  lat: float = None,
                                  lon: float = None,
                                  utc_offset: int = None,
                                  showplot: bool = False,
                                  plottitle: str = None,
                                  verbose: bool = False,
                                  repeat: bool = True,
                                  idstr: str = None):
        """
        Flag outliers based on z-score threshold (global or separate day/night).

        Applies z-score detection to identify values deviating from the mean by more than
        a specified number of standard deviations. Can operate globally across all records
        or separately for daytime and nighttime periods.

        Parameters
        ----------
        thres_zscore : float, default 4
            Z-score threshold for flagging. Typical range: 2.5-5.
        separate_daytime_nighttime : bool, default False
            If False, apply single threshold across all records (global mode).
            If True, apply separate thresholds to daytime and nighttime records.
            Requires lat, lon, utc_offset when True.
        lat : float, default None
            Site latitude in decimal degrees. Required when separate_daytime_nighttime=True.
        lon : float, default None
            Site longitude in decimal degrees. Required when separate_daytime_nighttime=True.
        utc_offset : int, default None
            UTC offset in hours. Required when separate_daytime_nighttime=True.
        showplot : bool, default False
            If True, display outlier visualization.
        plottitle : str, default None
            Title for plot. If None, auto-generated.
        verbose : bool, default False
            If True, print detection statistics.
        repeat : bool, default True
            If True, iteratively repeat detection until convergence.
        idstr : str, default None
            Optional identifier string for labeling output.
        """
        # Use provided parameters or class defaults
        lat = lat if lat is not None else self.site_lat
        lon = lon if lon is not None else self.site_lon
        utc_offset = utc_offset if utc_offset is not None else self.utc_offset
        idstr = idstr if idstr is not None else self.idstr

        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = zScore(
            series=series_cleaned,
            thres_zscore=thres_zscore,
            separate_daytime_nighttime=separate_daytime_nighttime,
            lat=lat,
            lon=lon,
            utc_offset=utc_offset,
            idstr=idstr,
            showplot=showplot,
            plottitle=plottitle,
            verbose=verbose
        )
        flagtest.calc(repeat=repeat)
        self._record_last(flagtest)

    def flag_outliers_zscore_rolling_test(self, thres_zscore: float = 4, showplot: bool = False, verbose: bool = False,
                                          plottitle: str = None, repeat: bool = True, winsize: int = None):
        """Identify outliers based on the z-score of records"""
        series_cleaned = self._series_hires_cleaned.copy()
        flagtest = zScoreRolling(series=series_cleaned, idstr=self.idstr, thres_zscore=thres_zscore,
                                 showplot=showplot, verbose=verbose, plottitle=plottitle,
                                 winsize=winsize)
        flagtest.calc(repeat=repeat)
        self._record_last(flagtest)

    def flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float = None,
                               separate_daytime_nighttime: bool = False,
                               showplot: bool = False, verbose: bool = False, repeat: bool = True, n_jobs: int = 1):
        """Local outlier factor detection (global or separate day/night).

        Identifies density-based outliers using k-nearest neighbors. Can operate globally
        across all records or separately for daytime and nighttime periods.

        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors for LOF calculation; auto-calculated if None (1/200 of non-NaN records).
        contamination : float or 'auto', default None
            Expected fraction of outliers (float 0-1) or 'auto' for automatic detection.
        separate_daytime_nighttime : bool, default False
            If False, apply single LOF globally across all records.
            If True, apply separate LOF detection for daytime and nighttime data.
        showplot : bool, default False
            If True, display visualization of detected outliers.
        verbose : bool, default False
            If True, print detection statistics.
        repeat : bool, default True
            If True, iteratively repeat detection until convergence.
        n_jobs : int, default 1
            Number of parallel jobs (-1 uses all cores).
        """
        series_cleaned = self._series_hires_cleaned.copy()
        # Number of neighbors is automatically calculated if not provided
        n_neighbors = int(len(series_cleaned.dropna()) / 200) if not n_neighbors else n_neighbors
        # Contamination is set automatically unless float is given
        contamination = contamination if isinstance(contamination, float) else 'auto'

        lat = self.site_lat if separate_daytime_nighttime else None
        lon = self.site_lon if separate_daytime_nighttime else None
        utc_offset = self.utc_offset if separate_daytime_nighttime else None

        flagtest = LocalOutlierFactor(series=series_cleaned, lat=lat, lon=lon, utc_offset=utc_offset,
                                      idstr=self.idstr,
                                      n_neighbors=n_neighbors, contamination=contamination,
                                      separate_daytime_nighttime=separate_daytime_nighttime,
                                      showplot=showplot, verbose=verbose, n_jobs=n_jobs)
        flagtest.calc(repeat=repeat)
        self._record_last(flagtest)

    def flag_outliers_abslim_test(self, minval: float = None, maxval: float = None,
                                  separate_daytime_nighttime: bool = False,
                                  daytime_minmax: list[float, float] = None,
                                  nighttime_minmax: list[float, float] = None,
                                  showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits.

        Parameters
        ----------
        minval : float, optional
            Minimum acceptable value (global mode). Required if separate_daytime_nighttime=False.
        maxval : float, optional
            Maximum acceptable value (global mode). Required if separate_daytime_nighttime=False.
        separate_daytime_nighttime : bool, default False
            If True, use separate day/night thresholds; if False, use global thresholds.
        daytime_minmax : [min, max], optional
            Acceptable range during daytime (required if separate_daytime_nighttime=True).
        nighttime_minmax : [min, max], optional
            Acceptable range during nighttime (required if separate_daytime_nighttime=True).
        showplot : bool, default False
            If True, display visualization of flagged outliers
        verbose : bool, default False
            If True, print flagging statistics
        """
        series_cleaned = self._series_hires_cleaned.copy()
        # For setting absolute limits no iteration is necessary, therefore always repeat=False
        flagtest = AbsoluteLimits(
            series=series_cleaned,
            minval=minval,
            maxval=maxval,
            separate_daytime_nighttime=separate_daytime_nighttime,
            daytime_minmax=daytime_minmax,
            nighttime_minmax=nighttime_minmax,
            lat=self.site_lat if separate_daytime_nighttime else None,
            lon=self.site_lon if separate_daytime_nighttime else None,
            utc_offset=self.utc_offset if separate_daytime_nighttime else None,
            idstr=self.idstr,
            showplot=showplot,
            verbose=verbose
        )
        flagtest.calc(repeat=False)
        self._record_last(flagtest)

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
            info(f"++Added flag column {flag.name} to flag data")
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
            info(f"++Added flag column {new_flagname} to flag data")

    def _setup(self) -> tuple[DataFrame, Series, Series]:
        """Setup data for outlier detection"""
        _series = self._series.copy()  # Data for this field

        # Sanitize timestamp
        _series = TimestampSanitizer(data=_series, output_middle_timestamp=self.output_middle_timestamp).get()

        # Initialize hires quality flags
        hires_flags = pd.DataFrame(index=_series.index)

        # Store original timeseries, will be cleaned
        series_hires_cleaned = _series.copy()  # Timeseries

        # Store original timeseries for this field dict, stays the same for later comparisons
        series_hires_orig = _series.copy()

        return hires_flags, series_hires_cleaned, series_hires_orig
