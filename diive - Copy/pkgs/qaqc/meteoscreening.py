# todo compare radiation peaks for time shift
# todo check outliers before AND after first qc check

"""
METEOSCREENING
==============

This module is part of the 'diive' library.

"""
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas.tseries.frequencies import to_offset

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.core.plotting.plotfuncs import default_format, default_legend, nice_date_ticks
from diive.core.plotting.timeseries import TimeSeries
from diive.core.times.resampling import resample_series_to_30MIN
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import detect_freq_groups
from diive.pkgs.corrections.offsetcorrection import remove_radiation_zero_offset, remove_relativehumidity_offset
from diive.pkgs.corrections.setto_threshold import setto_threshold
from diive.pkgs.outlierdetection.absolutelimits import AbsoluteLimits
from diive.pkgs.outlierdetection.incremental import zScoreIncrements
from diive.pkgs.outlierdetection.local3sd import LocalSD
from diive.pkgs.outlierdetection.lof import LocalOutlierFactorDaytimeNighttime, LocalOutlierFactorAllData
from diive.pkgs.outlierdetection.manualremoval import ManualRemoval
from diive.pkgs.outlierdetection.missing import MissingValues
from diive.pkgs.outlierdetection.seasonaltrend import OutlierSTLRIQRZ
from diive.pkgs.outlierdetection.thymeboost import ThymeBoostOutlier
from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime, zScoreIQR, zScore
from diive.pkgs.qaqc.qcf import FlagQCF


class StepwiseMeteoScreeningDb:
    """
    Stepwise MeteoScreening from database: Screen multiple vars from single measurement

    The class is optimized to work in Jupyter notebooks.

    Corrections and QC flags that can be directly accessed via the class 'StepwiseMeteoScreeningDb':
    - `.correction_remove_radiation_zero_offset()`: Remove nighttime offset from all radiation data and set nighttime to zero
    - `.correction_remove_relativehumidity_offset()`: Remove relative humidity offset
    - `.correction_setto_max_threshold()`: Set values above a threshold value to threshold value
    - `.correction_setto_min_threshold()`: Set values below a threshold value to threshold value
    - `.flag_missingvals_test()`: Generate flag that indicates missing records in data
    - `.flag_outliers_abslim_test()`: Generate flag that indicates if values in data are outside the specified range
    - `.flag_outliers_increments_zcore_test()`: Identify outliers based on the z-score of increments
    - `.flag_outliers_localsd_test()`: Identify outliers based on the local standard deviation
    - `.flag_outliers_stl_riqrz_test()`: Identify outliers based on seasonal-trend decomposition and z-score calculations
    - `.flag_outliers_thymeboost_test()`: Identify outliers based on [thymeboost](https://github.com/tblume1992/ThymeBoost)
    - `.flag_outliers_zscore_dtnt_test()`: Identify outliers based on the z-score, separately for daytime and nighttime
    - `.flag_outliers_zscore_test()`:  Identify outliers based on the z-score
    - `.flag_outliers_zscoreiqr_test()`: Identify outliers based on max z-scores in the interquartile range data
    - `.flag_outliers_lof_dtnt_test()`: Identify outliers based on local outlier factor, daytime nighttime separately
    - `.flag_outliers_lof_test()`: Identify outliers based on local outlier factor, across all data

    The class is optimized to work in Jupyter notebooks. Various outlier detection
    methods can be called on-demand. Outlier results are displayed and the user can
    accept the results and proceed, or repeat the step with adjusted method parameters.
    An unlimited amount of tests can be chained together. At the end of the screening,
    an overall flag is calculated from ALL single flags. The overall flag is then used
    to filter the time series.

    **Screening**
    The stepwise meteoscreening allows to perform **step-by-step** quality tests on
    meteorological data. A preview plot after running a test is shown and the user can
    decide if results are satisfactory or if the same test with different parameters
    should be re-run. Once results are satisfactory, the respective test flag is added
    to the data with `.addflag()`. After running the desired tests, an overall flag
    `QCF` is calculated from all individual tests.

    **Corrections**
    In addition to the creation of quality flags, the stepwise screening allows to
    **correct data for common issues**. For example, short-wave radiation sensors
    often measure negative values during the night. These negative values are useful
    because they give info about the accuracy and precision of the sensor. In this
    case, values during the night should be zero. Instead of cutting off negative
    values, `diive` detects the nighttime offset for each day and then calculates
    a correction slope between individual days. This way, the daytime values are
    also corrected.

    **Resampling**
    After quality-screening and corrections, data are resampled to 30MIN time resolution.

    **Handling different time resolutions**
    One challenging aspect of the screening were the different time resolutions of the raw
    data. In some cases, the time resolution changed from e.g. 10MIN for older data to 1MIN
    for newer date. In cases of different time resolution, **the lower resolution is upsampled
    to the higher resolution**, the emerging gaps are *back-filled* with available data.
    Back-filling is used because the timestamp in the database always is TIMESTAMP_END, i.e.,
    it gives the *end* of the averaging interval. The advantage of upsampling is that all
    outlier detection routines can be applied to the whole dataset. Since data are resampled
    to 30MIN after screening and since the TIMESTAMP_END is respected, the upsampling itself
    has no impact on resulting aggregates.

    **Variables**
    The class allows the simultaneous quality-screening of multiple variables from one single
    measurement, e.g., multiple air temperature variables.

    **Database tags**
    Is optimized to work with the InfluxDB format of the ETH Grassland Sciences Group. The
    class can handle database tags and updates tags after data screening and resampling.

    **Modular structure**
    At the moment, the stepwise meteoscreening works for data downloaded from the `InfluxDB`
    database. The screening respects the database format (including tags) and prepares the
    screened, corrected and resampled data for direct database upload. Due to its modular
    approach, the stepwise screening can be easily adjusted to work with any type of data
    files. This adjustment will be done in one of the next updates.

    **Current methods**
    A listing of currently implemented quality checks and correction can be found at the top
    of this file.

    """

    def __init__(
            self,
            data_detailed: dict,
            measurement: str,
            fields: list or str,
            site: str,
            site_lat: float,
            site_lon: float
    ):
        self.site = site
        self._data_detailed = data_detailed.copy()
        self.measurement = measurement
        self.fields = fields if isinstance(fields, list) else list(fields)
        self.site_lat = site_lat
        self.site_lon = site_lon

        # Returned variables
        self._resampled_detailed = {}
        self._hires_flags = {}
        self._tags = {}
        self._series_hires_orig = {}
        self._series_hires_cleaned = {}
        self._results_qcf = {}
        self._last_results = {}  # Results of most recent QC tests (objects)

        self._setup()

    @property
    def resampled_detailed(self) -> dict:
        """Return flag(s) as dict of Series"""
        if not isinstance(self._resampled_detailed, dict):
            raise Exception(f"No resampled data available.")
        return self._resampled_detailed

    @property
    def results_qcf(self) -> dict:
        """Return results from overall flag QCF calculation as dict of objects"""
        if not isinstance(self._results_qcf, dict):
            raise Exception(f"No QCF results available.")
        return self._results_qcf

    @property
    def data_detailed(self) -> dict:
        """Return high-resolution detailed data with tags as dict of DataFrames"""
        if not isinstance(self._data_detailed, dict):
            raise Exception(f"No high-resolution detailed data with tags available.")
        return self._data_detailed

    @property
    def last_results(self) -> dict:
        """Return high-resolution detailed data with tags as dict of objects"""
        if not isinstance(self._last_results, dict):
            raise Exception(f"No recent results available.")
        return self._last_results

    @property
    def tags(self) -> dict:
        """Return tags as dict of Series"""
        if not isinstance(self._tags, dict):
            raise Exception(f"No tags available.")
        return self._tags

    @property
    def series_hires_cleaned(self) -> dict:
        """Return cleaned time series of field(s) as dict of Series"""
        if not isinstance(self._series_hires_cleaned, dict):
            raise Exception(f"No hires quality-controlled data available.")
        return self._series_hires_cleaned

    @property
    def series_hires_orig(self) -> dict:
        """Return original time series of field(s) as dict of Series"""
        if not isinstance(self._series_hires_orig, dict):
            raise Exception(f"No hires original data available.")
        return self._series_hires_orig

    @property
    def hires_flags(self) -> dict:
        """Return flag(s) as dict of Series"""
        if not isinstance(self._hires_flags, dict):
            raise Exception(f"No hires flags available.")
        return self._hires_flags

    def showplot_qcf_heatmaps(self, **kwargs):
        for field in self.fields:
            self.results_qcf[field].showplot_qcf_heatmaps(**kwargs)

    def showplot_qcf_timeseries(self, **kwargs):
        for field in self.fields:
            self.results_qcf[field].showplot_qcf_timeseries(**kwargs)

    def showplot_resampled(self):
        """Show resampled data after high-resolution screening and corrections"""

        for field in self.fields:
            series_orig = self.series_hires_orig[field]
            series_resampled = self.resampled_detailed[field][field]

            fig = plt.figure(facecolor='white', figsize=(18, 9))
            gs = gridspec.GridSpec(3, 5)  # rows, cols
            gs.update(wspace=0.4, hspace=0.1, left=0.03, right=0.96, top=0.91, bottom=0.06)

            # Axes
            ax_orig = fig.add_subplot(gs[0, 0:3])
            ax_resampled = fig.add_subplot(gs[1, 0:3], sharex=ax_orig)
            ax_both = fig.add_subplot(gs[2, 0:3], sharex=ax_orig)
            ax_heatmap_hires_before = fig.add_subplot(gs[0:3, 3])
            ax_heatmap_resampled_after = fig.add_subplot(gs[0:3, 4], sharey=ax_heatmap_hires_before)

            # Time series
            ax_orig.plot_date(series_orig.index, series_orig, label=f"{series_orig.name}", color="#78909C",
                              alpha=.5, markersize=2, markeredgecolor='none')
            ax_resampled.plot_date(series_resampled.index, series_resampled, label=f"resampled",
                                   color="#FFA726", alpha=1, markersize=3, markeredgecolor='none')
            ax_both.plot_date(series_orig.index, series_orig, label=f"{series_orig.name}", color="#78909C",
                              alpha=.5, markersize=2, markeredgecolor='none')
            ax_both.plot_date(series_resampled.index, series_resampled, label=f"resampled",
                              color="#FFA726", alpha=1, markersize=3, markeredgecolor='none')

            # Heatmaps
            kwargs_heatmap = dict(cb_labelsize=10, axlabels_fontsize=10, ticks_labelsize=10,
                                  minyticks=3, maxyticks=99)
            HeatmapDateTime(ax=ax_heatmap_hires_before, series=series_orig, **kwargs_heatmap).plot()
            HeatmapDateTime(ax=ax_heatmap_resampled_after, series=series_resampled, **kwargs_heatmap).plot()

            # Format time series
            default_format(ax=ax_orig, ticks_labelsize=10)
            default_format(ax=ax_resampled, ticks_labelsize=10)
            default_format(ax=ax_both, ticks_labelsize=10)
            nice_date_ticks(ax=ax_orig, minticks=3, maxticks=20, which='x', locator='auto')
            default_legend(ax=ax_orig, markerscale=3, textsize=10)
            default_legend(ax=ax_resampled, markerscale=3, textsize=10)
            default_legend(ax=ax_both, markerscale=3, textsize=10)
            plt.setp(ax_orig.get_xticklabels(), visible=False)
            plt.setp(ax_resampled.get_xticklabels(), visible=False)
            plt.setp(ax_heatmap_resampled_after.get_yticklabels(), visible=False)

            fig.suptitle(f"{self.series_hires_orig[field].name}: "
                         f"High-resolution before QC & corrections vs "
                         f"resampled after QC & corrections",
                         fontsize=theme.FIGHEADER_FONTSIZE)
            fig.show()

    def showplot_orig(self, interactive: bool = False):
        """Show original high-resolution data used as input"""
        for field in self.fields:
            p = TimeSeries(series=self._series_hires_cleaned[field])
            p.plot() if not interactive else p.plot_interactive()

    def showplot_cleaned(self, interactive: bool = False):
        """Show *current* cleaned high-resolution data"""
        for field in self.fields:
            p = TimeSeries(series=self._series_hires_cleaned[field])
            p.plot() if not interactive else p.plot_interactive()

    def report_qcf_evolution(self):
        for field in self.fields:
            self.results_qcf[field].report_qcf_evolution()

    def report_qcf_flags(self):
        for field in self.fields:
            self.results_qcf[field].report_qcf_flags()

    def report_qcf_series(self):
        for field in self.fields:
            self.results_qcf[field].report_qcf_series()

    def flag_missingvals_test(self):
        """Flag missing values"""
        for field in self.fields:
            # data_detailed = self._data_detailed[field]  # Detailed data with tags
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _miss = MissingValues(series=series_cleaned)
            _miss.calc()
            self._last_results[field] = _miss  # Store in dict
            # self._hires_flags[_miss.flag.name] = _miss.flag

    def flag_manualremoval_test(self, remove_dates: list, showplot: bool = False, verbose: bool = False):
        """Flag specified records for removal"""
        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _mr = ManualRemoval(series=series_cleaned)
            _mr.calc(remove_dates=remove_dates, showplot=showplot, verbose=verbose)
            self._last_results[field] = _mr  # Store in dict

    def flag_outliers_zscore_dtnt_test(self, threshold: float = 4, showplot: bool = False, verbose: bool = False):
        """z-score, calculated separately for daytime and nighttime"""
        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _zscoredtnt = zScoreDaytimeNighttime(series=series_cleaned, site_lat=self.site_lat, site_lon=self.site_lon)
            _zscoredtnt.calc(threshold=threshold, showplot=showplot, verbose=verbose)
            self._last_results[field] = _zscoredtnt

    def flag_outliers_increments_zcore_test(self, threshold: int = 30, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on the z-score of record increments"""
        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _izsc = zScoreIncrements(series=series_cleaned)
            _izsc.calc(threshold=threshold, showplot=showplot)
            self._last_results[field] = _izsc

    def flag_outliers_zscoreiqr_test(self, factor: float = 4, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on the z-score of records in the IQR"""
        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _zsciqr = zScoreIQR(series=series_cleaned)
            _zsciqr.calc(factor=factor, showplot=showplot, verbose=verbose)
            self._last_results[field] = _zsciqr

    def flag_outliers_zscore_test(self, threshold: int = 4, showplot: bool = False, verbose: bool = False,
                                  plottitle: str = None):
        """Identify outliers based on the z-score of records"""
        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _zsc = zScore(series=series_cleaned)
            _zsc.calc(threshold=threshold, showplot=showplot, verbose=verbose, plottitle=plottitle)
            self._last_results[field] = _zsc

    def flag_outliers_thymeboost_test(self, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on thymeboost"""
        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _thymeboost = ThymeBoostOutlier(series=series_cleaned)
            _thymeboost.calc(showplot=showplot)
            self._last_results[field] = _thymeboost

    def flag_outliers_localsd_test(self, n_sd: float, winsize: int = None, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on standard deviation in a rolling window"""
        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _locsd = LocalSD(series=series_cleaned)
            _locsd.calc(n_sd=n_sd, winsize=winsize, showplot=showplot)
            self._last_results[field] = _locsd

    def flag_outliers_abslim_test(self, min: float, max: float, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits"""
        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _abslim = AbsoluteLimits(series=series_cleaned)
            _abslim.calc(min=min, max=max, showplot=showplot)
            self._last_results[field] = _abslim

    def flag_outliers_stl_riqrz_test(self, zfactor: float = 4.5, decompose_downsampling_freq: str = '1H',
                                     repeat: bool = False, showplot: bool = False):
        """Seasonsal trend decomposition with z-score on residuals"""
        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries
            _stl = OutlierSTLRIQRZ(series=series_cleaned, lat=self.site_lat, lon=self.site_lon)
            _stl.calc(zfactor=zfactor, decompose_downsampling_freq=decompose_downsampling_freq,
                      repeat=repeat, showplot=showplot)
            self._last_results[field] = _stl

    def flag_outliers_lof_dtnt_test(self, n_neighbors: int = None, contamination: float = 'auto',
                                    showplot: bool = False, verbose: bool = False):
        """Local outlier factor, separately for daytime and nighttime data"""

        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries

            # Number of neighbors is automatically calculated if not provided
            n_neighbors = int(len(series_cleaned.dropna()) / 100) if not n_neighbors else n_neighbors

            # Contamination is set automatically unless float is given
            contamination = contamination if isinstance(contamination, float) else 'auto'

            _lof = LocalOutlierFactorDaytimeNighttime(series=series_cleaned, site_lat=self.site_lat,
                                                      site_lon=self.site_lon)
            _lof.calc(n_neighbors=n_neighbors, contamination=contamination, showplot=showplot, verbose=verbose)
            self._last_result = _lof

    def flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float = 'auto',
                               showplot: bool = False, verbose: bool = False):
        """Local outlier factor, across all data"""

        for field in self.fields:
            series_cleaned = self._series_hires_cleaned[field]  # Timeseries

            # Number of neighbors is automatically calculated if not provided
            n_neighbors = int(len(series_cleaned.dropna()) / 100) if not n_neighbors else n_neighbors

            # Contamination is set automatically unless float is given
            contamination = contamination if isinstance(contamination, float) else 'auto'

            _lof = LocalOutlierFactorAllData(series=series_cleaned)
            _lof.calc(n_neighbors=n_neighbors, contamination=contamination, showplot=showplot, verbose=verbose)
            self._last_result = _lof

    def correction_remove_radiation_zero_offset(self):
        """Remove nighttime offset from all radiation data and set nighttime to zero"""
        for field in self.fields:
            self._series_hires_cleaned[field] = \
                remove_radiation_zero_offset(series=self._series_hires_cleaned[field],
                                             lat=self.site_lat, lon=self.site_lon,
                                             timezone_of_timestamp='UTC+01:00', showplot=True)

    def correction_setto_max_threshold(self, threshold: float):
        """Set values above threshold to threshold"""
        for field in self.fields:
            self._series_hires_cleaned[field] = \
                setto_threshold(series=self._series_hires_cleaned[field],
                                threshold=threshold, type='max', showplot=True)

    def correction_setto_min_threshold(self, threshold: float):
        """Set values below threshold to threshold"""
        for field in self.fields:
            self._series_hires_cleaned[field] = \
                setto_threshold(series=self._series_hires_cleaned[field],
                                threshold=threshold, type='min', showplot=True)

    def correction_remove_relativehumidity_offset(self):
        """Remove nighttime offset from all radiation data and set nighttime to zero"""
        for field in self.fields:
            self._series_hires_cleaned[field] = \
                remove_relativehumidity_offset(series=self._series_hires_cleaned[field], showplot=True)

    def resample(self,
                 to_freqstr: Literal['30T'] = '30T',
                 agg: Literal['mean', 'sum'] = 'mean',
                 mincounts_perc: float = .25):

        for field in self.fields:

            # Resample to 30MIN
            series_resampled = resample_series_to_30MIN(series=self._series_hires_cleaned[field],
                                                        to_freqstr=to_freqstr,
                                                        agg=agg,
                                                        mincounts_perc=mincounts_perc)

            # Update tags with resampling info
            self._tags[field]['freq'] = '30T'
            self._tags[field]['data_version'] = 'meteoscreening'

            # Create df that includes the resampled series and its tags
            self._resampled_detailed[field] = pd.DataFrame()
            self._resampled_detailed[field][field] = series_resampled  # Store screened variable with original name
            self._resampled_detailed[field] = self._resampled_detailed[field].asfreq(series_resampled.index.freqstr)

            # Insert tags as columns
            for key, value in self._tags[field].items():
                self._resampled_detailed[field][key] = value

    def calc_qcf(self):
        """Calculate overall quality flag QCF and add QCF results to other flags"""
        for field in self.fields:
            series_orig = self._data_detailed[field][field].copy()
            qcf = FlagQCF(df=self._hires_flags[field], series=series_orig)
            qcf.calculate()
            self._hires_flags[field] = qcf.flags
            self._series_hires_cleaned[field] = qcf.filteredseries
            self._results_qcf[field] = qcf
            self._last_results[field] = qcf

    def addflag(self):
        """Add flag of most recent test to data and update filtered series
        that will be used to continue with the next test"""
        for field in self.fields:
            flag = self._last_results[field].flag
            self._series_hires_cleaned[field] = self._last_results[field].filteredseries
            if not flag.name in self._hires_flags[field].columns:
                self._hires_flags[field][flag.name] = flag
            else:
                pass  # todo check
            # if flag.name in self._fulldf.columns:
            #     self._fulldf.drop([flag.name], axis=1, inplace=True)
            # self._fulldf[flag.name] = flag
            print(f"++Added flag column {flag.name} to flag data")

    def _setup(self):
        """Setup variable (field) data for meteoscreening"""
        # Loop over fields in measurement
        for field in self.fields:
            data_detailed = self._data_detailed[field]  # Data for this field
            timestamp_name = data_detailed.index.name  # Get name of timestamp for later use
            self._check_units(data_detailed=data_detailed)
            self._check_fields(data_detailed=data_detailed)

            # Harmonize different time resolutions (upsampling to highest freq)
            groups = self._make_timeres_groups(data_detailed=data_detailed)
            group_counts = self._count_group_records(group_series=groups[field])
            targetfreq, used_freqs, rejected_freqs = self._validate_n_grouprecords(group_counts=group_counts)
            data_detailed = self._filter_data(data_detailed=data_detailed, used_freqs=used_freqs)
            data_detailed = self._harmonize_timeresolution(targetfreq=targetfreq, data_detailed=data_detailed,
                                                           timestamp_name=timestamp_name)
            data_detailed = self._sanitize_timestamp(targetfreq=targetfreq, data_detailed=data_detailed)

            # Store tags for this field in dict
            self._tags[field] = self._extract_tags(data_detailed=data_detailed, field=field)

            # Store data_detailed for this field in dict
            self._data_detailed[field] = data_detailed.copy()

            # Initialize quality flags for this field
            self._hires_flags[field] = self._init_flagsdf(data_detailed=data_detailed, field=field)

            # Store original timeseries for this field dict, will be cleaned
            self._series_hires_cleaned[field] = self._data_detailed[field][field].copy()  # Timeseries

            # Store original timeseries for this field dict, stays the same for later comparisons
            self._series_hires_orig[field] = self._data_detailed[field][field].copy()

    @staticmethod
    def _sanitize_timestamp(targetfreq, data_detailed):
        """
        Set frequency info and sanitize timestamp

        This also converts the timestamp to TIMESTAMP_MIDDLE.
        """
        offset = to_offset(pd.Timedelta(f'{targetfreq}S'))
        data_detailed = data_detailed.asfreq(offset.freqstr)
        data_detailed = TimestampSanitizer(data=data_detailed).get()
        return data_detailed

    @staticmethod
    def _harmonize_timeresolution(targetfreq, data_detailed, timestamp_name: str) -> DataFrame:
        """
        Create timestamp index of highest resolution and upsample
        lower resolution data

        Creates hires timestamp index between start and end date
        for data where the time resolution is not in target freq.
        For this purpose, the first date found in the data is not
        completely correct, because a TIMESTAMP_END of e.g.
        '2022-01-01 00:10' at 10MIN resolution is valid from
        '2022-01-01 00:01' until '2022-01-01 00:10' in a 1MIN
        timestamp index. The missing timestamp indexes are added
        here.
        """
        upsampleddf = pd.DataFrame()  # Collects upsampled data
        groups = data_detailed.groupby(data_detailed['FREQ_AUTO_SEC'])

        # Loop over different time resolutions
        for freq, groupdf in groups:

            # No upsampling for target freq, simply merge
            if freq == targetfreq:
                upsampleddf = pd.concat([upsampleddf, groupdf], axis=0)
                continue

            # Add missing timestamp indexes at start of data
            start = groupdf.index[0] - pd.Timedelta(seconds=freq)

            # Create hires timestamp index between start and end dates
            hires_ix = pd.date_range(start=start,
                                     end=groupdf.index[-1],
                                     freq=f'{targetfreq}S')

            # If target freq is e.g. 60S (1MIN) and current freq is 600S (10MIN)
            # then the 600S records are valid for ten 60S records, whereby
            # one original record is already available
            # limit = (600 / 60) - 1 = 9 records to fill
            limit = int((freq / targetfreq) - 1)

            # The timestamp is TIMESTAMP_END, therefore 'backfill'
            cur_upsampleddf = groupdf.reindex(hires_ix)
            cur_upsampleddf = cur_upsampleddf.fillna(method='backfill', limit=limit)

            # Delete first timestamp index, outside limit
            cur_upsampleddf = cur_upsampleddf.iloc[1:].copy()

            # Add to upsampled data
            # upsampleddf = pd.concat([upsampleddf, cur_upsampleddf], axis=0)
            # Better use .combine_first to avoid duplicates
            upsampleddf = upsampleddf.combine_first(cur_upsampleddf)

        # Sort timestamp index ascending
        upsampleddf = upsampleddf.sort_index(ascending=True)
        upsampleddf.index.name = timestamp_name

        # upsampleddf.index.duplicated().sum()

        # import matplotlib.pyplot as plt
        # upsampleddf['TA_NABEL_T1_35_1'].plot()
        # plt.show()
        return upsampleddf

    @staticmethod
    def _extract_tags(data_detailed, field) -> dict:
        """For each variable, extract tag columns from the respective DataFrame
         and store info in simplified dict"""
        tags_df = data_detailed.drop(columns=[field, 'FREQ_AUTO_SEC'])
        # tags_df.nunique()
        notags = tags_df.isnull().all(axis=1)
        tags_df = tags_df[~notags]  # Drop rows where all tags are missing; this is the case due to upsampling
        tags_dict = {}
        for tag in tags_df.columns:
            list_of_vals = list(tags_df[tag].unique())
            str_of_vals = ",".join([str(i) for i in list_of_vals])
            tags_dict[tag] = str_of_vals
        return tags_dict

    @staticmethod
    def _init_flagsdf(data_detailed, field) -> DataFrame:
        """Initialize dataframe that will contain all flags for each variable"""
        series = data_detailed[field]  # Timeseries of variable
        hires_flags = pd.DataFrame(index=series.index)
        return hires_flags

    @staticmethod
    def _check_units(data_detailed):
        """Check if units are the same for all records"""
        unique_units = list(set(data_detailed['units']))
        if len(unique_units) > 1:
            raise Exception(f"More than one type of units in column 'units', "
                            f"but only one allowed. All data records must be "
                            f"in same units.")

    @staticmethod
    def _check_fields(data_detailed):
        """Check if really only one field in data"""
        unique_fields = list(set(data_detailed['varname']))
        if len(unique_fields) > 1:
            raise Exception(f"More than one variable name in column 'varname', "
                            f"but only one allowed. All data records must be "
                            f"for same variable.")

    @staticmethod
    def _make_timeres_groups(data_detailed):
        """Group data by time resolution"""
        groups_ser = detect_freq_groups(index=data_detailed.index)
        data_detailed[groups_ser.name] = groups_ser
        groups = data_detailed.groupby(data_detailed['FREQ_AUTO_SEC'])
        return groups

    @staticmethod
    def _count_group_records(group_series):
        """Count records for each found time resolution"""
        group_counts = group_series.count().sort_values(ascending=False)
        return group_counts

    @staticmethod
    def _validate_n_grouprecords(group_counts) -> tuple[float, list, list]:
        """Detect which frequencies have enough records to be used"""
        n_vals = group_counts.sum()
        n_freqs = group_counts.index.unique()
        print(f"Found {len(n_freqs)} unique frequencies across {n_vals} records.")
        print("Found frequencies:")
        cumulative_counts = 0
        used_freqs = []
        rejected_freqs = []
        for freq in n_freqs:
            counts = group_counts[freq]
            cumulative_counts += counts
            counts_perc = (counts / n_vals) * 100
            print(f"    Found time resolution {freq} (seconds) with {counts} records "
                  f"({counts_perc:.2f}% of total records).", end=" ")
            if counts_perc > 0.01:
                used_freqs.append(freq)
                print("")
            else:
                rejected_freqs.append(rejected_freqs)
                print("  -->  Frequency will be ignored, too few records.")
        print(f"The following frequencies will be used: {used_freqs} (seconds)")
        targetfreq = min(used_freqs)
        if len(used_freqs) > 1:
            print(f"Note that there is more than one single time resolution and "
                  f"all data will be upsampled to match the highest found time "
                  f"resolution ({targetfreq}S).")
        return targetfreq, used_freqs, rejected_freqs

    def _filter_data(self, data_detailed, used_freqs):
        data_detailed = data_detailed.loc[data_detailed['FREQ_AUTO_SEC'].isin(used_freqs)]
        return data_detailed


# class MeteoScreeningFromDatabaseMultipleVars:
#
#     def __init__(self,
#                  data_detailed: dict,
#                  assigned_measurements: dict,
#                  site: str,
#                  **kwargs):
#         self.site = site
#         self.data_detailed = data_detailed
#         self.assigned_measurements = assigned_measurements
#         self.kwargs = kwargs
#
#         # Returned variables
#         self.resampled_detailed = {}
#         self.hires_qc = {}
#         self.hires_flags = {}
#         # self.run()
#
#     def get(self) -> tuple[dict, dict, dict]:
#         return self.resampled_detailed, self.hires_qc, self.hires_flags
#
#     def run(self):
#         for var in self.data_detailed.keys():
#             m = self.assigned_measurements[var]
#             mscr = MeteoScreeningFromDatabaseSingleVar(data_detailed=self.data_detailed[var].copy(),
#                                                        site=self.site,
#                                                        measurement=m,
#                                                        field=var,
#                                                        resampling_freq='30T',
#                                                        **self.kwargs)
#             mscr.run()
#
#             self.resampled_detailed[var], \
#                 self.hires_qc[var], \
#                 self.hires_flags[var] = mscr.get()


# class MeteoScreeningFromDatabaseSingleVar:
#     """Accepts the output df from the `dbc` library that includes
#      variable data (`field`) and tags
#      """
#
#     def __init__(
#             self,
#             data_detailed: DataFrame,
#             measurement: str,
#             field: str,
#             site: str,
#             site_lat: float = None,
#             site_lon: float = None,
#             resampling_freq: str = '30T',
#             timezone_of_timestamp: str = None
#     ):
#         self.data_detailed = data_detailed.copy()
#         self.measurement = measurement
#         self.field = field
#         self.site = site
#         self.site_lat = site_lat
#         self.site_lon = site_lon
#         self.resampling_freq = resampling_freq
#         self.timezone_of_timestamp = timezone_of_timestamp
#
#         self.pipe_config = None  # Variable settings
#
#         unique_units = list(set(self.data_detailed['units']))
#         if len(unique_units) > 1:
#             raise Exception(f"More than one type of units in column 'units', "
#                             f"but only one allowed. All data records must be "
#                             f"in same units.")
#
#         # Group data by time resolution
#         groups_ser = detect_freq_groups(index=self.data_detailed.index)
#         self.data_detailed[groups_ser.name] = groups_ser
#         self.grps = self.data_detailed.groupby(self.data_detailed['FREQ_AUTO_SEC'])
#
#         # Returned variables
#         # Collected data across (potentially) different time resolutions
#         self.coll_resampled_detailed = None
#         self.coll_hires_qc = None
#         self.coll_hires_flags = None
#
#     def run(self):
#         self._run_by_freq_group()
#         self.coll_resampled_detailed.drop(['FREQ_AUTO_SEC'], axis=1, inplace=True)  # Currently not used as tag in db
#         self.coll_resampled_detailed.sort_index(inplace=True)
#         self.coll_hires_qc.sort_index(inplace=True)
#         self.coll_hires_flags.sort_index(inplace=True)
#         # self._plots()
#
#     def _run_by_freq_group(self):
#         """Screen and resample by frequency
#
#         When downloading data from the database, it is possible that the same
#         variable was recorded at a different time resolution in the past. Each
#         time resolution has to be handled separately. After the meteoscreening,
#         all data are in 30MIN time resolution.
#         """
#         # Loop through frequency groups
#         # All group members have the same, confirmed frequency
#         grp_counter = 0
#         for grp_freq, grp_var_df in self.grps:
#             hires_raw = grp_var_df[self.field].copy()  # Group series
#             varname = hires_raw.name
#             tags_dict = self.extract_tags(var_df=grp_var_df, drop_field=self.field)
#             grp_counter += 1
#             print(f"({varname}) Working on data with time resolution {grp_freq}S")
#
#             # Since all group members have the same time resolution, the freq of
#             # the Series can already be set.
#             offset = to_offset(pd.Timedelta(f'{grp_freq}S'))
#             hires_raw = hires_raw.asfreq(offset.freqstr)  # Set to group freq
#
#             # Sanitize timestamp index
#             hires_raw = TimestampSanitizer(data=hires_raw).get()
#             # series = sanitize_timestamp_index(data=series, freq=grp_freq)
#
#             screening = ScreenMeteoVar(series=hires_raw,
#                                        measurement=self.measurement,
#                                        units=tags_dict['units'],
#                                        site=self.site,
#                                        site_lat=self.site_lat,
#                                        site_lon=self.site_lon,
#                                        timezone_of_timestamp=self.timezone_of_timestamp)
#             flags_hires = screening.flags
#             seriesqcf_hires = screening.seriesqcf
#             self.pipe_config = screening.pipe_config
#
#             # Resample to 30MIN
#             seriesqcf_resampled = resample_series_to_30MIN(series=seriesqcf_hires,
#                                                            to_freqstr=self.resampling_freq,
#                                                            agg=self.pipe_config['resampling_aggregation'],
#                                                            mincounts_perc=.25)
#
#             # Rename QC'd series to original name (without _QCF at the end)
#             # This is done to comply with the naming convention, which needs
#             # the positional indices at the end of the variable name (not _QCF).
#             seriesqcf_resampled.name = hires_raw.name
#
#             # Update tags after resampling
#             tags_dict['freq'] = '30T'
#             # tags_dict['freqfrom'] = 'resampling'
#             tags_dict['data_version'] = 'meteoscreening'
#
#             # Create df that includes the resampled series and its tags
#             resampled_detailed = pd.DataFrame(seriesqcf_resampled)
#             resampled_detailed = resampled_detailed.asfreq(seriesqcf_resampled.index.freqstr)
#
#             # Insert tags as columns
#             for key, value in tags_dict.items():
#                 resampled_detailed[key] = value
#
#             if grp_counter == 1:
#                 self.coll_resampled_detailed = resampled_detailed.copy()  # Collection
#                 self.coll_hires_qc = seriesqcf_hires.copy()
#                 self.coll_hires_flags = flags_hires.copy()
#             else:
#                 # self.coll_resampled_detailed = pd.concat([self.coll_resampled_detailed, resampled_detailed], axis=0)
#                 self.coll_resampled_detailed = self.coll_resampled_detailed.combine_first(other=resampled_detailed)
#                 self.coll_resampled_detailed.sort_index(ascending=True, inplace=True)
#                 self.coll_resampled_detailed = self.coll_resampled_detailed.asfreq(resampled_detailed.index.freqstr)
#
#                 # self.coll_hires_qc = pd.concat([self.coll_hires_qc, hires_qc], axis=0)
#                 self.coll_hires_qc = self.coll_hires_qc.combine_first(other=seriesqcf_hires)
#                 self.coll_hires_qc.sort_index(ascending=True, inplace=True)
#
#                 # self.coll_hires_flags = pd.concat([self.coll_hires_flags, hires_flags], axis=0)
#                 self.coll_hires_flags = self.coll_hires_flags.combine_first(other=flags_hires)
#                 self.coll_hires_flags.sort_index(ascending=True, inplace=True)
#
#     def get(self) -> tuple[DataFrame, Series, DataFrame]:
#         return self.coll_resampled_detailed, self.coll_hires_qc, self.coll_hires_flags
#
#     def extract_tags(self, var_df: DataFrame, drop_field: str) -> dict:
#         """Convert tag columns in DataFrame to simplified dict
#
#         Args:
#             var_df:
#             drop_field:
#
#         Returns:
#             dict of tags
#
#         """
#         _df = var_df.copy()
#         tags_df = _df.drop(columns=[drop_field])
#         tags_df.nunique()
#         tags_dict = {}
#         for col in tags_df.columns:
#             list_of_vals = list(tags_df[col].unique())
#             str_of_vals = ",".join([str(i) for i in list_of_vals])
#             tags_dict[col] = str_of_vals
#         return tags_dict
#
#     def show_groups(self):
#         print("Number of values per frequency group:")
#         print(self.grps.count())
#
#     def _plots(self):
#         varname = self.field
#
#         # Plot heatmap
#         HeatmapDateTime(series=self.coll_resampled_detailed[varname],
#                         title=f"{self.coll_resampled_detailed[varname].name} RESAMPLED DATA\n"
#                               f"AFTER HIGH-RES METEOSCREENING  |  "
#                               f"time resolution @{self.coll_resampled_detailed.index.freqstr}").show()
#
#         # Plot comparison
#         _hires_qc = self.coll_hires_qc.copy()
#         _hires_qc.name = f"{_hires_qc.name} (HIGH-RES, after quality checks)"
#         _resampled = self.coll_resampled_detailed[varname].copy()
#         _resampled.name = f"{_resampled.name} (RESAMPLED) " \
#                           f"{_resampled.index.freqstr} {self.pipe_config['resampling_aggregation']}"
#         quickplot([_hires_qc, _resampled], subplots=False, showplot=True,
#                   title=f"Comparison")

# class ScreenMeteoVar:
#     """Quality screening of one single meteo data time series
#
#     Input series is not altered. Instead, this class produces a DataFrame
#     that contains all quality flags.
#
#     Typical time series: radiation, air temperature, soil water content.
#
#     """
#
#     def __init__(
#             self,
#             series: Series,
#             measurement: str,
#             units: str,
#             site: str,
#             site_lat: float = None,
#             site_lon: float = None,
#             timezone_of_timestamp: str = None
#     ):
#         self.series = series
#         self.measurement = measurement
#         self.units = units
#         self.site = site
#         self.site_lat = site_lat
#         self.site_lon = site_lon
#         self.timezone_of_timestamp = timezone_of_timestamp
#
#         # Processing pipes
#         path = Path(__file__).parent.resolve()  # Search in this file's folder
#         self.pipes = filereader.ConfigFileReader(configfilepath=path / 'pipes_meteo.yaml',
#                                                  validation='meteopipe').read()
#         self._pipe_config = self._pipe_assign()
#
#         # Collect all flags
#         self._flags_df = None
#
#         self._run()
#
#     @property
#     def pipe_config(self) -> dict:
#         """Processing pipe configuration"""
#         if not isinstance(self._pipe_config, dict):
#             raise Exception('Pipe configuration not available')
#         return self._pipe_config
#
#     @property
#     def flags(self) -> DataFrame:
#         """Processing pipe configuration"""
#         if not isinstance(self._flags_df, DataFrame):
#             raise Exception('Flags are empty')
#         return self._flags_df
#
#     @property
#     def seriesqcf(self) -> Series:
#         """Series with rejected values set to missing"""
#         if not isinstance(self._seriesqcf, Series):
#             raise Exception('Flags are empty')
#         return self._seriesqcf
#
#     def _run(self):
#
#         # Call the various checks for this variable and generate flags
#         mqcf = self._call_pipe_steps()
#
#         # Print some info about QCF
#         mqcf.report_qcf_flags()
#         mqcf.report_qcf_series()
#         mqcf.report_qcf_evolution()
#
#         # Plot
#         mqcf.showplot_qcf_heatmaps()
#         mqcf.showplot_qcf_timeseries()
#
#         # # Find column that contains the QCF info
#         # flagqcfcol = findqcfcol(df=self._flags_df, varname=str(self.series.name))
#
#     def _call_pipe_steps(self):
#         # Initiate new flag collection
#         self._flags_df = pd.DataFrame(index=self.series.index)
#
#         # Processing steps for this variable
#         pipe_steps = self.pipe_config['pipe']
#
#         # Missing values flag, always generated (high-res)
#         _miss = MissingValues(series=self.series)
#         _miss.calc()
#         self._flags_df[_miss.flag.name] = _miss.flag
#
#         for step in pipe_steps:
#             # remove_highres_outliers_incremental_zscore,
#
#             if step == 'remove_highres_outliers_incremental_zscore':
#                 # Generates flag
#                 _mqcf = FlagQCF(df=self._flags_df, series=self.series)
#                 _mqcf.calculate()
#                 _izsc = zScoreIncrements(series=_mqcf.filteredseries)
#                 _izsc.calc(threshold=30, showplot=True)
#                 self._flags_df[_izsc.flag.name] = _izsc.flag
#
#             elif step == 'remove_highres_outliers_lof':
#                 # Generates flag, needs QC'd data
#                 _mqcf = FlagQCF(df=self._flags_df, series=self.series)
#                 _mqcf.calculate()
#                 _lof = LocalOutlierFactorAllData(series=_mqcf.filteredseries)
#                 _lof.calc(n_neighbors=100, contamination=0.01, showplot=True)
#                 self._flags_df[_lof.flag.name] = _lof.flag
#
#             elif step == 'remove_highres_outliers_stl':
#                 # Generates flag
#                 _stl = OutlierSTLRIQRZ(series=self.series, lat=self.site_lat, lon=self.site_lon)
#                 _stl.calc(zfactor=4.5, decompose_downsampling_freq='1H', showplot=True)
#                 self._flags_df[_stl.flag.name] = _stl.flag
#
#             elif step == 'remove_highres_outliers_thymeboost':
#                 # Generates flag, needs QC'd data
#                 _mqcf = FlagQCF(df=self._flags_df, series=self.series)
#                 _mqcf.calculate()
#                 _thymeboost = ThymeBoostOutlier(series=_mqcf.filteredseries)
#                 _thymeboost.calc(showplot=True)
#                 self._flags_df[_thymeboost.flag.name] = _thymeboost.flag
#
#             elif step == 'remove_highres_outliers_zscore':
#                 # Generates flag
#                 _mqcf = FlagQCF(df=self._flags_df, series=self.series)
#                 _mqcf.calculate()
#                 # import matplotlib.pyplot as plt
#                 # _mqcf.seriesqcf.plot()
#                 # plt.show()
#                 _zscoreiqr = zScoreIQR(series=_mqcf.filteredseries)
#                 _zscoreiqr.calc(factor=4.5, showplot=True, verbose=True)
#                 self._flags_df[_zscoreiqr.flag.name] = _zscoreiqr.flag
#
#             elif step == 'remove_highres_outliers_localsd':
#                 # Generates flag, needs QC'd data
#                 _mqcf = FlagQCF(df=self._flags_df, series=self.series)
#                 _mqcf.calculate()
#                 _localsd = LocalSD(series=_mqcf.filteredseries)
#                 _localsd.calc(n_sd=7, showplot=True, verbose=True)
#                 # _flag_outlier_local3sd = xxxlocalsd(series=_series, n_sd=7, showplot=True)
#                 self._flags_df[_localsd.flag.name] = _localsd.flag
#
#             elif step == 'remove_highres_outliers_absolute_limits':
#                 # Generates flag
#                 _abslim = AbsoluteLimits(series=self.series)
#                 _abslim.calc(min=self.pipe_config['absolute_limits'][0],
#                              max=self.pipe_config['absolute_limits'][1])
#                 self._flags_df[_abslim.flag.name] = _abslim.flag
#
#             elif step == 'remove_radiation_zero_offset':
#                 # Generates corrected series
#                 if (self.site_lat is None) \
#                         | (self.site_lon is None) \
#                         | (self.timezone_of_timestamp is None):
#                     raise Exception("Removing the radiation zero offset requires site latitude, "
#                                     "site longitude and site timezone (e.g., 'UTC+01:00' for CET).")
#                 self.series = remove_radiation_zero_offset(
#                     series=self.series,
#                     lat=self.site_lat,
#                     lon=self.site_lon,
#                     timezone_of_timestamp=self.timezone_of_timestamp,
#                     showplot=True
#                 )
#
#             elif step == 'setto_max_threshold':
#                 # Generates corrected series
#                 self.series = setto_threshold(
#                     series=self.series,
#                     threshold=self.pipe_config['absolute_limits'][1],
#                     type='max',
#                     showplot=True)
#
#             elif step == 'setto_min_threshold':
#                 # Generates corrected series
#                 self.series = setto_threshold(series=self.series,
#                                               threshold=self.pipe_config['absolute_limits'][0],
#                                               type='min',
#                                               showplot=True)
#
#             elif step == 'remove_relativehumidity_offset':
#                 # Generates corrected series
#                 self.series = remove_relativehumidity_offset(series=self.series,
#                                                              showplot=True)
#
#             else:
#                 raise Exception(f"No function defined for {step}.")
#
#         # Calculate overall quality flag QCF and Add QCF results to other flags
#         mqcf = self._calculate_qcf(missingflag=str(_miss.flag.name))
#         self._seriesqcf = mqcf.filteredseries.copy()
#         return mqcf
#
#     def _calculate_qcf(self, missingflag: str):
#         """Calculate overall quality flag QCF and Add QCF results to other flags"""
#         mqcf = FlagQCF(df=self._flags_df, series=self.series)
#         mqcf.calculate()
#         _flags_df = pd.concat([self._flags_df, mqcf.flags], axis=1)
#         _flags_df = _flags_df.loc[:, ~_flags_df.columns.duplicated(keep='last')]
#         self._flags_df = _flags_df
#         return mqcf
#
#     def _pipe_assign(self) -> dict:
#         """Assign appropriate processing pipe to this variable"""
#         if self.measurement not in self.pipes.keys():
#             raise KeyError(f"{self.measurement} is not defined in meteo_pipes.yaml")
#
#         pipe_config = var = None
#         for var in self.pipes[self.measurement].keys():
#             if var in self.series.name:
#                 if self.pipes[self.measurement][var]['units'] != self.units:
#                     continue
#                 pipe_config = self.pipes[self.measurement][var]
#                 break
#
#         print(f"Variable '{self.series.name}' has been assigned to processing pipe {pipe_config['pipe']}")
#
#         return pipe_config

def example():
    # todo For examples see notebooks/MeteoScreening

    # Settings
    SITE = 'ch-dav'  # Site name
    SITE_LAT = 46.815333
    SITE_LON = 9.855972
    MEASUREMENT = 'SW'
    FIELDS = ['SW_IN_NABEL_T1_35_1']  # Variable name; InfluxDB stores variable names as '_field'
    START = '2021-01-01 00:01:00'  # Download data starting with this date
    STOP = '2021-01-05 00:01:00'  # Download data before this date (the stop date itself is not included)

    # # Some info
    # from datetime import datetime
    # from pathlib import Path
    # import pkg_resources
    # # from diive.pkgs.qaqc.meteoscreening import MetScrDbMeasurementVars
    # dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"This page was last modified on: {dt_string}")
    # version_dbc_influxdb = pkg_resources.get_distribution("dbc_influxdb").version
    # print(f"dbc-influxdb version: v{version_dbc_influxdb}")

    # Auto-settings
    DIRCONF = r'L:\Sync\luhk_work\20 - CODING\22 - POET\configs'  # Folder with configurations
    TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # Timezone, e.g. "1" is translated to timezone "UTC+01:00" (CET, winter time)
    RESAMPLING_FREQ = '30T'  # During MeteoScreening the screened high-res data will be resampled to this frequency; '30T' = 30-minute time resolution
    RESAMPLING_AGG = 'mean'  # The resampling of the high-res data will be done using this aggregation methos; e.g., 'mean'
    from pathlib import Path
    basedir = Path(r"L:\Sync\luhk_work\_temp")
    BUCKET_RAW = f'{SITE}_raw'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_raw' contains all raw data for CH-LAE
    BUCKET_PROCESSING = f'{SITE}_processing'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_processing' contains all processed data for CH-LAE
    print(f"Bucket containing raw data (source bucket): {BUCKET_RAW}")
    print(f"Bucket containing processed data (destination bucket): {BUCKET_PROCESSING}")

    # Download data from database with "dbc-influxdb"
    from dbc_influxdb import dbcInflux
    dbc = dbcInflux(dirconf=DIRCONF)  # Instantiate class
    data_simple, data_detailed, assigned_measurements = \
        dbc.download(bucket=BUCKET_RAW,
                     measurements=[MEASUREMENT],
                     fields=FIELDS,
                     start=START,
                     stop=STOP,
                     timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
                     data_version='raw')
    # import matplotlib.pyplot as plt
    # data_simple.plot()
    # plt.show()

    # # Export data to pickle for fast testing
    # import pickle
    # pickle_out = open(basedir / "meteodata_simple.pickle", "wb")
    # pickle.dump(data_simple, pickle_out)
    # pickle_out.close()
    # pickle_out = open(basedir / "meteodata_detailed.pickle", "wb")
    # pickle.dump(data_detailed, pickle_out)
    # pickle_out.close()
    # pickle_out = open(basedir / "meteodata_assigned_measurements.pickle", "wb")
    # pickle.dump(assigned_measurements, pickle_out)
    # pickle_out.close()

    # # Import data from pickle for fast testing
    # # from diive.core.io.files import load_pickle
    # import pickle
    # pickle_in = open(basedir / "meteodata_simple.pickle", "rb")
    # data_simple = pickle.load(pickle_in)
    # pickle_in = open(basedir / "meteodata_detailed.pickle", "rb")
    # data_detailed = pickle.load(pickle_in)
    # pickle_in = open(basedir / "meteodata_assigned_measurements.pickle", "rb")
    # assigned_measurements = pickle.load(pickle_in)

    # # Restrict data for testing
    # from diive.core.dfun.frames import df_between_two_dates
    # for key in data_detailed.keys():
    #     data_detailed[key] = df_between_two_dates(df=data_detailed[key], start_date='2022-06-01', end_date='2022-06-30')

    # Start MeteoScreening session
    mscr = StepwiseMeteoScreeningDb(site=SITE,
                                    data_detailed=data_detailed,
                                    measurement=MEASUREMENT,
                                    fields=FIELDS,
                                    site_lat=SITE_LAT,
                                    site_lon=SITE_LON)

    # # Plot data
    # mscr.showplot_orig()
    # mscr.showplot_cleaned()

    # Missing values test
    mscr.flag_missingvals_test()
    mscr.addflag()

    # # Missing values test
    # mscr.flag_manualremoval_test(remove_dates=['2022-06-29 23:59:30',
    #                                            ['2022-06-01', '2022-06-16']
    #                                            ],
    #                              showplot=True, verbose=True)
    # mscr.addflag()

    # # Outlier detection: z-score over all data
    # mscr.flag_outliers_zscore_test(threshold=3, showplot=True, verbose=True)
    # mscr.addflag()

    # # Outlier detection: z-score over all data with IQR
    # mscr.flag_outliers_zscoreiqr_test(factor=4, showplot=True, verbose=True)
    # mscr.addflag()

    # # Outlier detection: z-score over all data, separate for daytime and nighttime
    # mscr.flag_outliers_zscore_dtnt_test(threshold=2, showplot=True, verbose=True)
    # mscr.addflag()

    # # Outlier detection: Seasonal trend decomposition (residuals, IQR, z-score)
    # mscr.flag_outliers_stl_riqrz_test(zfactor=4.5, decompose_downsampling_freq='2H', showplot=True, repeat=False)
    # mscr.addflag()

    # # Outlier detection: Increments z-score
    # mscr.flag_outliers_increments_zcore_test(threshold=5, showplot=True)
    # mscr.addflag()

    # # Outlier detection: Thymeboost
    # mscr.flag_outliers_thymeboost_test(showplot=True)
    # mscr.addflag()
    #
    # # Outlier detection: Absolute limits
    # mscr.flag_outliers_abslim_test(min=-50, max=50, showplot=True)
    # mscr.addflag()

    # # Outlier detection: Local SD
    # mscr.flag_outliers_localsd_test(n_sd=5, winsize=None, showplot=True)
    # mscr.addflag()

    # # Outlier detection: Local outlier factor, across all data
    # mscr.flag_outliers_lof_test(n_neighbors=None, showplot=True, verbose=True)
    # mscr.addflag()

    # # Outlier detection: Local outlier factor, daytime nighttime
    # mscr.flag_outliers_lof_dtnt_test(n_neighbors=None, showplot=True, verbose=True)
    # mscr.addflag()

    # After all QC flags generated, calculate overall flag QCF
    mscr.calc_qcf()

    # QCF reports
    mscr.report_qcf_evolution()
    # mscr.report_qcf_flags()
    # mscr.report_qcf_series()
    mscr.showplot_qcf_heatmaps()
    # mscr.showplot_qcf_timeseries()

    # # Apply corrections
    # mscr.correction_remove_radiation_zero_offset()
    # mscr.correction_setto_max_threshold(threshold=400)
    # mscr.correction_setto_min_threshold(threshold=100)
    # mscr.correction_remove_relativehumidity_offset()

    # End MeteoScreening session
    mscr.resample(to_freqstr='30T', agg='mean', mincounts_perc=.25)
    mscr.showplot_resampled()

    for v in mscr.resampled_detailed.keys():
        m = assigned_measurements[v]
        dbc.upload_singlevar(to_bucket=BUCKET_PROCESSING,
                             to_measurement=m,
                             var_df=mscr.resampled_detailed[v],
                             timezone_of_timestamp='UTC+01:00')


if __name__ == '__main__':
    example()
