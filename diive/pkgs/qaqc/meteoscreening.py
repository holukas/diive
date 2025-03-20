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

import diive.core.dfun.frames as frames
import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.core.plotting.plotfuncs import default_format, default_legend, nice_date_ticks
from diive.core.plotting.timeseries import TimeSeries
from diive.core.times.resampling import resample_series_to_30MIN
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import detect_freq_groups
from diive.pkgs.analyses.correlation import daily_correlation
from diive.pkgs.corrections.offsetcorrection import remove_radiation_zero_offset, remove_relativehumidity_offset
from diive.pkgs.corrections.setto_threshold import setto_threshold
from diive.pkgs.corrections.setto_value import setto_value
from diive.pkgs.createvar.potentialradiation import potrad
from diive.pkgs.outlierdetection.stepwiseoutlierdetection import StepwiseOutlierDetection
from diive.pkgs.qaqc.flags import MissingValues
from diive.pkgs.qaqc.qcf import FlagQCF


class StepwiseMeteoScreeningDb:
    """
    Stepwise MeteoScreening from database: Screen multiple vars from single measurement

    The class is optimized to work in Jupyter notebooks. Various outlier detection
    methods can be called on-demand. Outlier results are displayed and the user can
    accept the results and proceed, or repeat the step with adjusted method parameters.
    An unlimited amount of tests can be chained together. At the end of the screening,
    an overall flag is calculated from ALL single flags. The overall flag is then used
    to filter the time series.

    Implemented outlier tests:
    For a full list of outlier tests see: pkgs/outlierdetection/stepwiseoutlierdetection.py
    - `.flag_missingvals_test()`: Generate flag that indicates missing records in data
    - `.flag_outliers_abslim_test()`: Generate flag that indicates if values in data are outside the specified range
    - `.flag_outliers_increments_zcore_test()`: Identify outliers based on the z-score of increments
    - `.flag_outliers_localsd_test()`: Identify outliers based on the local standard deviation
    - `.flag_manualremoval_test()`: Remove data points for range, time or point-by-point
    - `.flag_outliers_zscore_dtnt_test()`: Identify outliers based on the z-score, separately for daytime and nighttime
    - `.flag_outliers_zscore_rolling_test()`: Identify outliers based on the rolling z-score
    - `.flag_outliers_zscore_test()`:  Identify outliers based on the z-score
    - `.flag_outliers_lof_dtnt_test()`: Identify outliers based on local outlier factor, daytime nighttime separately
    - `.flag_outliers_lof_test()`: Identify outliers based on local outlier factor, across all data

    Implemented corrections:
    - `.correction_remove_radiation_zero_offset()`: Remove nighttime offset from all radiation data and set nighttime to zero
    - `.correction_remove_relativehumidity_offset()`: Remove relative humidity offset
    - `.correction_setto_max_threshold()`: Set values above a threshold value to threshold value
    - `.correction_setto_min_threshold()`: Set values below a threshold value to threshold value
    - `.correction_setto_value()`: Set records in time range(s) to constant value

    Implemented analyses:
    - `.analysis_potential_radiation_correlation()`: Analyzes time series daily correlation with potential radiation

    **Outlier tests**
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

    """

    def __init__(
            self,
            data_detailed: dict,
            # measurement: str,
            fields: list or str,
            site: str,
            site_lat: float,
            site_lon: float,
            utc_offset: int
    ):
        self.site = site
        self._data_detailed = data_detailed.copy()
        # self.measurement = measurement
        self.fields = fields if isinstance(fields, list) else list(fields)
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.utc_offset = utc_offset

        # Initiate dictionaries
        # Results are stored for each variables, with the variable names (fields) as dictionary keys.
        self._tags = {}  # Contains tags required for the database
        self._series_hires_orig = {}  # The original, unfiltered time series without tags
        self._series_hires_cleaned = {}  # The cleaned time series without tags
        self._outlier_detection = {}  # Results (instances) from the outlier detection for each variable
        self._outlier_detection_qcf = {}  # Results (instances) from the overall quality flag QCF calculations
        self._resampled_detailed = {}  # Resampled time series with tags

        for field in self.fields:
            # Validate data_detailed
            print(f"Validating data for variable {field} ... ")
            self._data_detailed[field] = self._validate_data_detailed(
                data_detailed=self.data_detailed[field],
                field=field)

            # The original input series that is screened
            self._series_hires_orig[field] = self.data_detailed[field][field]

            # The cleaned input series, is the same as orig when screening is started
            self._series_hires_cleaned[field] = self.data_detailed[field][field]

            self._tags[field] = self._extract_tags(data_detailed=self.data_detailed[field], field=field)

    @property
    def data_detailed(self) -> dict:
        """Return high-resolution detailed data with tags as dict of DataFrames."""
        if not isinstance(self._data_detailed, dict):
            raise Exception("No high-resolution detailed data with tags available.")
        return self._data_detailed

    @property
    def outlier_detection(self) -> dict:
        """Return results from stepwise outlier detection as dict of instances."""
        if not isinstance(self._outlier_detection, dict):
            raise Exception("No results from stepwise outlier detection available.")
        return self._outlier_detection

    @property
    def outlier_detection_qcf(self) -> dict:
        """Return results from stepwise outlier detection overall quality flag as dict of instances."""
        if not isinstance(self._outlier_detection_qcf, dict):
            raise Exception("No results for overall quality flag QCF from stepwise outlier detection available.")
        return self._outlier_detection_qcf

    @property
    def series_hires_cleaned(self) -> dict:
        """Return cleaned time series of field(s) as dict of Series"""
        if not isinstance(self._series_hires_cleaned, dict):
            raise Exception("No hires quality-controlled data available.")
        return self._series_hires_cleaned

    @property
    def series_hires_orig(self) -> dict:
        """Return original time series of field(s) as dict of Series"""
        if not isinstance(self._series_hires_orig, dict):
            raise Exception("No hires original data available.")
        return self._series_hires_orig

    def start_outlier_detection(self):
        """Initiate step-wise outlier detection (sod) for each field.
        Each field gets its own sod instance."""
        for field in self.fields:
            print(f"Starting step-wise outlier detection for variable {field} ...")
            self._outlier_detection[field] = StepwiseOutlierDetection(
                dfin=self.data_detailed[field].copy(),
                col=field,
                site_lat=self.site_lat,
                site_lon=self.site_lon,
                utc_offset=self.utc_offset)

    @property
    def resampled_detailed(self) -> dict:
        """Return flag(s) as dict of Series"""
        if not isinstance(self._resampled_detailed, dict):
            raise Exception("No resampled data available.")
        return self._resampled_detailed

    @property
    def tags(self) -> dict:
        """Return tags as dict of Series"""
        if not isinstance(self._tags, dict):
            raise Exception("No tags available.")
        return self._tags

    def showplot_outlier_detection_cleaned(self, interactive: bool = False):
        """Show cleaned data from outlier detection."""
        for field in self.fields:
            self.outlier_detection[field].showplot_cleaned(interactive=interactive)

    def showplot_outlier_detection_qcf_heatmaps(self, **kwargs):
        for field in self.fields:
            self.outlier_detection_qcf[field].showplot_qcf_heatmaps(**kwargs)

    def showplot_outlier_detection_qcf_timeseries(self, **kwargs):
        for field in self.fields:
            self.outlier_detection_qcf[field].showplot_qcf_timeseries(**kwargs)

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

            ax_orig.xaxis.axis_date()
            ax_resampled.xaxis.axis_date()
            ax_both.xaxis.axis_date()

            ax_heatmap_hires_before = fig.add_subplot(gs[0:3, 3])
            ax_heatmap_resampled_after = fig.add_subplot(gs[0:3, 4], sharey=ax_heatmap_hires_before)

            # Time series
            ax_orig.plot(series_orig.index, series_orig, label=f"{series_orig.name}", color="#78909C",
                         alpha=.5, markersize=2, markeredgecolor='none')
            ax_resampled.plot(series_resampled.index, series_resampled, label="resampled",
                              color="#FFA726", alpha=1, markersize=3, markeredgecolor='none')
            ax_both.plot(series_orig.index, series_orig, label=f"{series_orig.name}", color="#78909C",
                         alpha=.5, markersize=2, markeredgecolor='none')
            ax_both.plot(series_resampled.index, series_resampled, label="resampled",
                         color="#FFA726", alpha=1, markersize=3, markeredgecolor='none')

            # Heatmaps
            kwargs_heatmap = dict(cb_labelsize=10, axlabels_fontsize=10, ticks_labelsize=10,
                                  minyticks=3, maxyticks=99)
            HeatmapDateTime(ax=ax_heatmap_hires_before, series=series_orig, **kwargs_heatmap).plot()
            HeatmapDateTime(ax=ax_heatmap_resampled_after, series=series_resampled, **kwargs_heatmap).plot()

            # Format time series
            default_format(ax=ax_orig, ticks_labels_fontsize=10)
            default_format(ax=ax_resampled, ticks_labels_fontsize=10)
            default_format(ax=ax_both, ticks_labels_fontsize=10)
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
            p = TimeSeries(series=self.series_hires_orig[field])
            p.plot() if not interactive else p.plot_interactive()

    def showplot_cleaned(self, interactive: bool = False):
        """Show *current* cleaned high-resolution data"""
        for field in self.fields:
            p = TimeSeries(series=self.series_hires_cleaned[field])
            p.plot() if not interactive else p.plot_interactive()

    def report_outlier_detection_qcf_evolution(self):
        for field in self.fields:
            self.outlier_detection_qcf[field].report_qcf_evolution()

    def report_outlier_detection_qcf_flags(self):
        for field in self.fields:
            self.outlier_detection_qcf[field].report_qcf_flags()

    def report_outlier_detection_qcf_series(self):
        for field in self.fields:
            self.outlier_detection_qcf[field].report_qcf_series()

    def flag_missingvals_test(self, verbose: bool = False):
        """Flag missing values and add flag to dataframe."""
        for field in self.fields:
            flagtest = MissingValues(series=self.data_detailed[field][field].copy(), verbose=verbose)
            flagtest.calc(repeat=False)
            flag = flagtest.get_flag()
            self._data_detailed[field][flag.name] = flag

    def flag_manualremoval_test(self, remove_dates: list, showplot: bool = False, verbose: bool = False):
        """Flag specified records for removal"""
        for field in self.fields:
            self.outlier_detection[field].flag_manualremoval_test(remove_dates=remove_dates,
                                                                  showplot=showplot,
                                                                  verbose=verbose)

    def flag_outliers_zscore_dtnt_test(self, thres_zscore: float = 4, showplot: bool = False, verbose: bool = False,
                                       repeat: bool = True):
        """z-score, calculated separately for daytime and nighttime"""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_zscore_dtnt_test(thres_zscore=thres_zscore,
                                                                         showplot=showplot,
                                                                         verbose=verbose,
                                                                         repeat=repeat)

    def flag_outliers_localsd_test(self, n_sd: float, winsize: int = None, showplot: bool = False,
                                   constant_sd: bool = False, verbose: bool = False, repeat: bool = True):
        """Identify outliers based on standard deviation in a rolling window"""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_localsd_test(n_sd=n_sd,
                                                                     winsize=winsize,
                                                                     constant_sd=constant_sd,
                                                                     showplot=showplot,
                                                                     verbose=verbose,
                                                                     repeat=repeat)

    def flag_outliers_increments_zcore_test(self, thres_zscore: int = 30, showplot: bool = False,
                                            verbose: bool = False, repeat: bool = True):
        """Identify outliers based on the z-score of record increments"""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_increments_zcore_test(thres_zscore=thres_zscore,
                                                                              showplot=showplot,
                                                                              verbose=verbose,
                                                                              repeat=repeat)

    def flag_outliers_zscore_test(self, thres_zscore: int = 4, showplot: bool = False, verbose: bool = False,
                                  plottitle: str = None, repeat: bool = True):
        """Identify outliers based on the z-score of records"""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_zscore_test(thres_zscore=thres_zscore,
                                                                    showplot=showplot,
                                                                    verbose=verbose,
                                                                    plottitle=plottitle,
                                                                    repeat=repeat)

    def flag_outliers_zscore_rolling_test(self, thres_zscore: float = 4, showplot: bool = False, verbose: bool = False,
                                          plottitle: str = None, repeat: bool = True, winsize: int = None):
        """Identify outliers based on the z-score of records"""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_zscore_rolling_test(thres_zscore=thres_zscore,
                                                                            showplot=showplot,
                                                                            verbose=verbose,
                                                                            plottitle=plottitle,
                                                                            repeat=repeat,
                                                                            winsize=winsize)

    def flag_outliers_hampel_test(self, window_length: int = 10, n_sigma: float = 5, k: float = 1.4826,
                                  showplot: bool = False, verbose: bool = False, repeat: bool = True):
        """Identify outliers in a sliding window based on the Hampel filter."""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_hampel_test(window_length=window_length,
                                                                    n_sigma=n_sigma,
                                                                    k=k,
                                                                    showplot=showplot,
                                                                    verbose=verbose,
                                                                    repeat=repeat)

    def flag_outliers_hampel_dtnt_test(self, window_length: int = 10, n_sigma_dt: float = 5,
                                       n_sigma_nt: float = 5, k: float = 1.4826,
                                       showplot: bool = False, verbose: bool = False, repeat: bool = True):
        """Identify outliers in a sliding window based on the Hampel filter,
        separately for daytime and nighttime data."""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_hampel_dtnt_test(window_length=window_length,
                                                                         n_sigma_dt=n_sigma_dt,
                                                                         n_sigma_nt=n_sigma_nt,
                                                                         k=k,
                                                                         showplot=showplot,
                                                                         verbose=verbose,
                                                                         repeat=repeat)

    def flag_outliers_trim_low_test(self, trim_daytime: bool = False, trim_nighttime: bool = False,
                                    lower_limit: float = None, showplot: bool = False, verbose: bool = False):
        """Flag values below a given absolute limit as outliers, then flag an
        equal number of datapoints at the high end as outliers."""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_trim_low_test(trim_daytime=trim_daytime,
                                                                      trim_nighttime=trim_nighttime,
                                                                      lower_limit=lower_limit,
                                                                      showplot=showplot,
                                                                      verbose=verbose)

    def flag_outliers_abslim_test(self, minval: float, maxval: float, showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits"""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_abslim_test(minval=minval,
                                                                    maxval=maxval,
                                                                    showplot=showplot,
                                                                    verbose=verbose)

    def flag_outliers_abslim_dtnt_test(self, daytime_minmax: list[float, float],
                                       nighttime_minmax: list[float, float], showplot: bool = False):
        """Identify outliers based on absolute limits"""
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_abslim_dtnt_test(daytime_minmax=daytime_minmax,
                                                                         nighttime_minmax=nighttime_minmax,
                                                                         showplot=showplot)

    def flag_outliers_lof_dtnt_test(self, n_neighbors: int = None, contamination: float = 'auto',
                                    showplot: bool = False, verbose: bool = False, n_jobs: int = 1,
                                    repeat: bool = True):
        """Local outlier factor, separately for daytime and nighttime data"""

        for field in self.fields:
            self.outlier_detection[field].flag_outliers_lof_dtnt_test(n_neighbors=n_neighbors,
                                                                      contamination=contamination,
                                                                      showplot=showplot,
                                                                      verbose=verbose,
                                                                      repeat=repeat,
                                                                      n_jobs=n_jobs)

    def flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float = 'auto',
                               showplot: bool = False, verbose: bool = False, repeat: bool = True,
                               n_jobs: int = 1):
        """Local outlier factor, across all data"""

        for field in self.fields:
            self.outlier_detection[field].flag_outliers_lof_test(n_neighbors=n_neighbors,
                                                                 contamination=contamination,
                                                                 showplot=showplot,
                                                                 verbose=verbose,
                                                                 repeat=repeat,
                                                                 n_jobs=n_jobs)

    def correction_remove_radiation_zero_offset(self):
        """Remove nighttime offset from all radiation data and set nighttime to zero"""
        for field in self.fields:
            self._series_hires_cleaned[field] = \
                remove_radiation_zero_offset(series=self._series_hires_cleaned[field],
                                             lat=self.site_lat, lon=self.site_lon,
                                             utc_offset=self.utc_offset, showplot=True)

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

    def correction_setto_value(self, dates: list, value: float, verbose: int = 1):
        """Set records within time range to value"""
        for field in self.fields:
            self._series_hires_cleaned[field] = \
                setto_value(series=self._series_hires_cleaned[field],
                            dates=dates, value=value, verbose=verbose)

    def correction_remove_relativehumidity_offset(self):
        """Remove nighttime offset from all radiation data and set nighttime to zero"""
        for field in self.fields:
            self._series_hires_cleaned[field] = \
                remove_relativehumidity_offset(series=self._series_hires_cleaned[field], showplot=True)

    def analysis_potential_radiation_correlation(self,
                                                 utc_offset: int,
                                                 mincorr: float = 0.7,
                                                 showplot: bool = True) -> dict:
        """Compare time series to potential radiation

        Args:
            utc_offset: UTC offset of *radiation* timestamp
                For example, for European winter time *utc_offset=1*.
            mincorr: minimum absolute correlation, only relevant when *showplot=True*,
                must be between -1 and 1 (inclusive)
                Example: with *0.8* all correlations between -0.8 and +0.8 are considered low,
                and all correlations smaller than -0.8 and higher than +0.8 are considered high.
            showplot: if *True*, show plot of results

        Returns:
            dict of series with correlations for each field and for each day

        """

        daily_correlations = {}
        for field in self.fields:
            series = self.series_hires_cleaned[field]
            # Calculate potential radiation SW_IN_POT
            swinpot = potrad(timestamp_index=series.index,
                             lat=self.site_lat,
                             lon=self.site_lon,
                             utc_offset=utc_offset)

            # Calculate daily correlation between potential and measured observation
            daycorrs = daily_correlation(
                s1=series,
                s2=swinpot,
                mincorr=mincorr,
                showplot=showplot
            )
            daily_correlations[field] = daycorrs

        return daily_correlations

    def resample(self,
                 to_freqstr: Literal['30T', '30min'] = '30min',
                 agg: Literal['mean', 'sum'] = 'mean',
                 mincounts_perc: float = .25):

        for field in self.fields:

            # Resample to 30MIN
            series_resampled = resample_series_to_30MIN(series=self._series_hires_cleaned[field],
                                                        to_freqstr=to_freqstr,
                                                        agg=agg,
                                                        mincounts_perc=mincounts_perc)

            # Update tags with resampling info
            self._tags[field]['freq'] = '30min'
            self._tags[field]['data_version'] = 'meteoscreening_diive'

            # Create df that includes the resampled series and its tags
            self._resampled_detailed[field] = pd.DataFrame()
            self._resampled_detailed[field][field] = series_resampled  # Store screened variable with original name
            self._resampled_detailed[field] = self._resampled_detailed[field].asfreq(series_resampled.index.freqstr)

            # Insert tags as columns
            for key, value in self._tags[field].items():
                self._resampled_detailed[field][key] = value

    def finalize_outlier_detection(self,
                                   daytime_accept_qcf_below: int = 2,
                                   nighttimetime_accept_qcf_below: int = 2) -> FlagQCF:

        for field in self.fields:
            # Detect new columns
            newcols = frames.detect_new_columns(df=self.outlier_detection[field].flags,
                                                other=self.data_detailed[field])
            self._data_detailed[field] = pd.concat(
                [self.data_detailed[field], self.outlier_detection[field].flags[newcols]], axis=1)
            [print(f"++Added new column {col}.") for col in newcols]

            # Calculate overall quality flag QCF
            qcf = FlagQCF(series=self.data_detailed[field][field],
                          df=self.data_detailed[field],
                          idstr='METSCR',
                          swinpot=None
                          # nighttime_threshold=nighttime_threshold
                          )
            qcf.calculate(daytime_accept_qcf_below=daytime_accept_qcf_below,
                          nighttimetime_accept_qcf_below=nighttimetime_accept_qcf_below)
            self._data_detailed[field] = qcf.get()
            self._outlier_detection_qcf[field] = qcf

            # Update filtered series in meteoscreening instance
            self._series_hires_cleaned[field] = self.outlier_detection_qcf[field].filteredseries

    def addflag(self):
        """Add flag of most recent outlier test to data."""
        for field in self.fields:
            self._outlier_detection[field].addflag()

    def _validate_data_detailed(self, data_detailed, field) -> dict:
        """Setup variable (field) data for meteoscreening"""

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

        return data_detailed

    @staticmethod
    def _sanitize_timestamp(targetfreq, data_detailed):
        """
        Set frequency info and sanitize timestamp

        This also converts the timestamp to TIMESTAMP_MIDDLE.
        """
        offset = to_offset(pd.Timedelta(f'{targetfreq}s'))
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
            raise Exception("More than one type of units in column 'units', "
                            "but only one allowed. All data records must be "
                            "in same units.")

    @staticmethod
    def _check_fields(data_detailed):
        """Check if really only one field in data"""
        unique_fields = list(set(data_detailed['varname']))
        if len(unique_fields) > 1:
            raise Exception("More than one variable name in column 'varname', "
                            "but only one allowed. All data records must be "
                            "for same variable.")

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


def example():
    from pathlib import Path
    from diive.core.times.times import DetectFrequency

    TESTDIR = Path(r"F:\TMP")

    SITE = 'ch-cha'
    SITE_LAT = 47.210227
    SITE_LON = 8.410645
    FIELDS = ['G_GF1_0.03_1']
    MEASUREMENT = 'G'
    START = '2024-06-01 00:00:01'
    STOP = '2024-07-01 00:00:01'
    RESAMPLING_AGG = 'mean'
    # RESAMPLING_AGG = 'sum'
    DATA_VERSION = 'raw'
    TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # Timezone, e.g. "1" is translated to timezone "UTC+01:00" (CET, winter time)
    RESAMPLING_FREQ = '30min'  # During MeteoScreening the screened high-res data will be resampled to this frequency; '30min' = 30-minute time resolution
    # DIRCONF = r'P:\Flux\RDS_calculations\_scripts\_configs\configs'  # Location of configuration files, needed e.g. for connection to database
    DIRCONF = r'L:\Sync\luhk_work\20 - CODING\22 - POET\configs'
    BUCKET_RAW = f'{SITE}_raw'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_raw' contains all raw data for CH-LAE
    BUCKET_PROCESSED = f'a'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_processed' contains all processed data for CH-LAE
    # BUCKET_PROCESSED = f'{SITE}_processed'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_processed' contains all processed data for CH-LAE

    from dbc_influxdb import dbcInflux
    dbc = dbcInflux(dirconf=DIRCONF)
    data_simple, data_detailed, assigned_measurements = dbc.download(
        bucket=BUCKET_RAW,
        measurements=[MEASUREMENT],
        fields=FIELDS,
        start=START,
        stop=STOP,
        timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
        data_version=DATA_VERSION
    )
    print(data_simple)
    print(data_detailed)
    print(assigned_measurements)
    import matplotlib.pyplot as plt
    data_simple.plot()
    plt.show()

    # print(f"Data available for: {data_detailed.keys()}\n")
    # vars_not_available = [v for v in FIELDS if v not in data_detailed.keys()]
    # print(f"No data available for the following variables: {vars_not_available}")
    # for rem in vars_not_available:
    #     print(rem)
    #     FIELDS.remove(rem)
    #     print(f"Removed variables {rem} from FIELDS because it is not available during this time period.")

    # for varname, frame in data_detailed.items():
    #     TimeSeries(series=frame[varname]).plot()

    # # Export data to pickle and parquet for fast testing
    # from diive.core.io.files import save_parquet, save_as_pickle
    # save_parquet(filename="meteodata_simple", data=data_simple, outpath=TESTDIR)
    # save_as_pickle(filename="meteodata_detailed", data=data_detailed, outpath=TESTDIR)
    # save_as_pickle(filename="meteodata_assigned_measurements", data=assigned_measurements, outpath=TESTDIR)

    # Import data from pickle for fast testing
    from diive.core.io.files import load_parquet, load_pickle
    data_simple = load_parquet(filepath=TESTDIR / "meteodata_simple.parquet")
    _f = str(TESTDIR / "meteodata_detailed.pickle")
    data_detailed = load_pickle(_f)
    _f = str(TESTDIR / "meteodata_assigned_measurements.pickle")
    assigned_measurements = load_pickle(_f)

    # # Restrict data for testing
    # from diive.core.dfun.frames import df_between_two_dates
    # for key in data_detailed.keys():
    #     data_detailed[key] = df_between_two_dates(df=data_detailed[key], start_date='2022-06-01', end_date='2022-06-30')

    mscr = StepwiseMeteoScreeningDb(site=SITE,
                                    data_detailed=data_detailed,
                                    # measurement=MEASUREMENT,
                                    fields=FIELDS,
                                    site_lat=SITE_LAT,
                                    site_lon=SITE_LON,
                                    utc_offset=TIMEZONE_OFFSET_TO_UTC_HOURS)
    # mscr.showplot_orig()
    mscr.showplot_cleaned()

    # -----------------
    # OUTLIER DETECTION
    # -----------------
    mscr.start_outlier_detection()  # If needed
    # REMOVE_DATES = [
    #     ['2010-12-24 07:15:00', '2011-01-01 00:00:15'],  # Remove time range
    #     # '2022-08-23 11:45:00',  # Remove data point
    #     # ['2022-08-15', '2022-08-16']  # Remove time range
    # ]
    # mscr.flag_manualremoval_test(remove_dates=REMOVE_DATES, showplot=True, verbose=True)
    # mscr.flag_outliers_zscore_dtnt_test(thres_zscore=4, showplot=True, verbose=True, repeat=True)
    mscr.flag_outliers_hampel_test(window_length=48 * 7, n_sigma=5.5, showplot=True, verbose=True, repeat=True)
    mscr.flag_outliers_hampel_dtnt_test(window_length=48 * 7, n_sigma_dt=5.5, n_sigma_nt=4.5, showplot=True,
                                        verbose=True, repeat=True)
    mscr.addflag()
    mscr.flag_outliers_trim_low_test(trim_daytime=False, trim_nighttime=True, lower_limit=10,
                                     showplot=True, verbose=True)
    mscr.addflag()
    # mscr.flag_outliers_zscore_rolling_test(thres_zscore=3, winsize=48, showplot=True, verbose=True, repeat=True)
    # mscr.flag_outliers_localsd_test(n_sd=2.5, winsize=24, showplot=True, verbose=True, repeat=False)
    mscr.flag_outliers_localsd_test(n_sd=2.5, winsize=24, showplot=True, verbose=True, repeat=False)
    # mscr.flag_outliers_increments_zcore_test(thres_zscore=9, showplot=True, verbose=True, repeat=True)
    # mscr.flag_outliers_zscore_test(thres_zscore=15, showplot=True, verbose=True, repeat=True)
    # mscr.flag_outliers_lof_dtnt_test(n_neighbors=3, contamination=0.00001, showplot=True,
    #                                  verbose=True, repeat=False, n_jobs=-1)
    # mscr.flag_outliers_lof_test(n_neighbors=30, contamination=0.01, showplot=True, verbose=True,
    #                             repeat=False, n_jobs=-1)
    # mscr.flag_outliers_abslim_test(minval=-20, maxval=50, showplot=True)
    # mscr.flag_outliers_abslim_dtnt_test(daytime_minmax=[-20.0, 40.0], nighttime_minmax=[-9.0, 20.0], showplot=True)
    # mscr.addflag()
    # mscr.flag_missingvals_test(verbose=True)  # No .addflag() needed

    # # -----------
    # # QCF
    # # -----------
    # # QCF: After all QC flags generated, calculate overall flag QCF
    # mscr.finalize_outlier_detection()
    # mscr.report_outlier_detection_qcf_evolution()
    # mscr.report_outlier_detection_qcf_flags()
    # mscr.report_outlier_detection_qcf_series()
    # mscr.showplot_outlier_detection_qcf_heatmaps()
    # # mscr.showplot_outlier_detection_qcf_timeseries()

    # # -----------
    # # CORRECTIONS
    # # -----------
    # mscr.showplot_orig()
    # mscr.showplot_cleaned()
    mscr.correction_remove_radiation_zero_offset()  # Remove radiation zero offset
    # mscr.correction_remove_relativehumidity_offset()  # Remove relative humidity offset
    # mscr.correction_setto_max_threshold(threshold=30)  # Set to max threshold
    # mscr.correction_setto_min_threshold(threshold=-5)  # Set to min threshold
    # DATES = [['2022-04-01', '2022-04-05'], ['2022-09-05', '2022-09-07']]
    # mscr.correction_setto_value(dates=DATES, value=3.7, verbose=1)  # Set to value

    # # -----------
    # # ANALYSES
    # # -----------
    # # # Potential radiation correlation
    # _ = mscr.analysis_potential_radiation_correlation(utc_offset=1, mincorr=0.7, showplot=True)

    # -----------
    # RESAMPLING
    # -----------
    mscr.resample(to_freqstr=RESAMPLING_FREQ, agg=RESAMPLING_AGG, mincounts_perc=.25)
    # mscr.showplot_resampled()
    for v in mscr.resampled_detailed.keys():
        print(f"{'-' * 20}\n{v}")
        _checkfreq = DetectFrequency(index=mscr.resampled_detailed[v].index, verbose=True).get()
        if _checkfreq == RESAMPLING_FREQ:
            print(f"TEST PASSED - The resampled variable {v} has a time resolution of {_checkfreq}.")
        else:
            print(
                f"{'#' * 20}(!)TEST FAILED - The resampled variable {v} does not have the expected time resolution of {_checkfreq}.{'#' * 20}")

    # from diive.core.plotting.dielcycle import DielCycle
    # for v in mscr.resampled_detailed.keys():
    #     series = mscr.resampled_detailed[v]['TA_T1_2_1']
    #     dc = DielCycle(series=series)
    #     dc.plot(ax=None, each_month=True, legend_n_col=2)

    # -----------
    # UPLOAD
    # -----------
    from dbc_influxdb import dbcInflux
    dbc = dbcInflux(dirconf=DIRCONF)
    for v in mscr.resampled_detailed.keys():
        # mscr.resampled_detailed[v][v] = mscr.resampled_detailed[v][v].multiply(999)
        # mscr.resampled_detailed[v]['hpos'] = '999'
        m = assigned_measurements[v]
        dbc.upload_singlevar(to_bucket=BUCKET_PROCESSED,
                             to_measurement=m,
                             var_df=mscr.resampled_detailed[v],
                             timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
                             delete_from_db_before_upload=True)

    # -----------
    # DOWNLOAD
    # -----------

    # MEASUREMENT = ['TA']
    # data_simple, data_detailed, assigned_measurements \
    #     = dbc.download(bucket=BUCKET_PROCESSED,
    #                    measurements=MEASUREMENT,
    #                    fields=FIELDS,
    #                    start=START,
    #                    stop=STOP,
    #                    timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
    #                    data_version='meteoscreening_diive')
    #
    # print("END.")


if __name__ == '__main__':
    example()
