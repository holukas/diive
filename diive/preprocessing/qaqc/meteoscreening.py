"""
METEOSCREENING: MULTI-STAGE METEOROLOGICAL SCREENING
====================================================

Multi-stage quality control and outlier detection for meteorological data.
Includes: outlier detection, data corrections, resampling, and quality flag generation.

Part of the diive library: https://github.com/holukas/diive
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
from diive.core.utils.console import info, detail, warn
from diive.core.plotting.plotfuncs import default_format, default_legend, nice_date_ticks
from diive.core.plotting.styles.format import FormatStyle
from diive.core.plotting.timeseries import TimeSeries
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import detect_freq_groups
from diive.analysis import daily_correlation
from diive.variables.radiation import potrad
from diive.preprocessing.corrections import remove_nighttime_zero_offset, remove_relativehumidity_offset
from diive.preprocessing.corrections import set_exact_values_to_missing, setto_threshold, setto_value
from diive.preprocessing.outlier_detection import StepwiseOutlierDetection
from diive.preprocessing.qaqc.flags import MissingValues
from diive.preprocessing.qaqc.qcf import FlagQCF

_SWINPOT_COL = '_SW_IN_POT_METSCR'  # Temporary day/night input for the QCF


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
    For a full list of outlier tests see: pkgs/preprocessing/outlier_detection/stepwiseoutlierdetection.py
    - `.flag_missingvals_test()`: Generate flag that indicates missing records in data
    - `.flag_outliers_abslim_test()`: Generate flag that indicates if values in data are outside the specified range (global or separate day/night)
    - `.flag_outliers_increments_zcore_test()`: Identify outliers based on the z-score of increments
    - `.flag_outliers_localsd_test()`: Identify outliers based on the local standard deviation (global or separate day/night)
    - `.flag_manualremoval_test()`: Remove data points for range, time or point-by-point
    - `.flag_outliers_zscore_test()`:  Identify outliers based on the z-score (global or separate day/night)
    - `.flag_outliers_zscore_rolling_test()`: Identify outliers based on the rolling z-score
    - `.flag_outliers_lof_test()`: Identify outliers based on local outlier factor (global or separate day/night)
    - `.flag_outliers_hampel_test()`: Identify outliers based on the Hampel filter (global or separate day/night)
    - `.flag_outliers_trim_low_test()`: Remove values below threshold and remove an equal amount of records from high end of data

    Implemented corrections:
    - `.correction_remove_nighttime_zero_offset()`: Remove nighttime offset from a variable that should be zero at night (e.g. radiation) and set nighttime to zero
    - `.correction_remove_relativehumidity_offset()`: Remove relative humidity offset
    - `.correction_setto_max_threshold()`: Set values above a threshold value to threshold value
    - `.correction_setto_min_threshold()`: Set values below a threshold value to threshold value
    - `.correction_setto_value()`: Set records in time range(s) to constant value
    - `.correction_set_exact_value_to_missing()`: Set records with exact value to missing values (NaN)

    Implemented analysis:
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
    The time resolution of the raw data can change, e.g. from 10MIN for older data to 1MIN
    for newer data. All records are then placed on one grid at the finest resolution, each
    **only at its own END timestamp** (the database timestamp is TIMESTAMP_END). A coarse
    record is not copied onto the finer slots it covers: those slots stay empty through the
    whole screening, and corrections do not fill them. Outlier tests run on this sparse
    series; a window given as a record count counts grid slots, a window given as a time
    span ('7D') is unaffected. `.flag_manualremoval_test()` and `.correction_setto_value()`
    therefore act on whole records. `.resample()` weights each record by the time it
    covers: a 10MIN record counts ten times as much as a 1MIN record in a mean, and once
    in a sum.

    **Timestamps**
    Screening runs on TIMESTAMP_MIDDLE (converted from the database's TIMESTAMP_END).
    `.flag_manualremoval_test()` and `.correction_setto_value()` take dates in the
    database's TIMESTAMP_END convention.

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

    Example:
        ``data_detailed`` maps each variable to a DataFrame as downloaded from the
        database: a TIMESTAMP_END index, the variable column and the tag columns.
        Here four days of synthetic 10-minute air temperature with one spike:

        >>> import numpy as np, pandas as pd
        >>> import diive as dv
        >>> from diive.core.io.db.influx.common import TAGS
        >>> idx = pd.date_range('2024-07-01 00:10', periods=4 * 144, freq='10min',
        ...                     name='TIMESTAMP_END')
        >>> hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
        >>> ta = 15 + 5 * np.sin(2 * np.pi * (hour - 9) / 24)
        >>> ta += np.random.default_rng(42).normal(0, 0.2, len(idx))
        >>> ta[200] = 60  # spike at END timestamp 2024-07-02 09:30
        >>> df = pd.DataFrame({'TA_T1_2_1': ta}, index=idx)
        >>> df[TAGS] = '-'
        >>> df[['site', 'varname', 'units', 'freq']] = ['CH-XYZ', 'TA_T1_2_1', 'degC', '10min']

        Screen, keep the flag, remove flagged records and resample:

        >>> mscr = dv.qaqc.StepwiseMeteoScreeningDb(
        ...     data_detailed={'TA_T1_2_1': df}, fields='TA_T1_2_1', site='ch-xyz',
        ...     site_lat=47.29, site_lon=7.73, utc_offset=1)
        >>> mscr.start_outlier_detection()
        >>> mscr.flag_outliers_abslim_test(minval=-30, maxval=50)
        >>> mscr.addflag()
        >>> mscr.finalize_outlier_detection()
        >>> int(mscr.series_hires_cleaned['TA_T1_2_1'].isna().sum())
        1
        >>> mscr.resample(to_freqstr='30min', agg='mean')
        >>> len(mscr.resampled_detailed['TA_T1_2_1'])
        192

        The method examples on this page continue from this ``mscr``.
    """

    def __init__(
            self,
            data_detailed: dict,
            # measurement: str,
            fields: list | str,
            site: str,
            site_lat: float,
            site_lon: float,
            utc_offset: int
    ):
        """Set up stepwise meteo screening. See the class docstring."""
        self.site = site
        # Copy the frames too: validation adds columns and must not touch the caller's data.
        self._data_detailed = {key: df.copy() for key, df in data_detailed.items()}
        # self.measurement = measurement
        self.fields = [fields] if isinstance(fields, str) else list(fields)
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
            info(f"Validating data for variable {field} ...")
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
        Each field gets its own sod instance. Tests run on the current series, so
        corrections applied before this call are what the tests see."""
        for field in self.fields:
            info(f"Starting step-wise outlier detection for variable {field} ...")
            dfin = self.data_detailed[field].copy()
            dfin[field] = self._series_hires_cleaned[field]
            self._outlier_detection[field] = StepwiseOutlierDetection(
                dfin=dfin,
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
        """Show the QCF outlier-detection heatmaps."""
        for field in self.fields:
            self.outlier_detection_qcf[field].showplot_qcf_heatmaps(**kwargs)

    def showplot_outlier_detection_qcf_timeseries(self, **kwargs):
        """Show the QCF outlier-detection time series."""
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
            records_orig = self._plot_records(field, series_orig)
            ax_orig.plot(records_orig.index, records_orig, label=f"{series_orig.name}", color="#78909C",
                         alpha=.5, markersize=2, markeredgecolor='none')
            ax_resampled.plot(series_resampled.index, series_resampled, label="resampled",
                              color="#FFA726", alpha=1, markersize=3, markeredgecolor='none')
            ax_both.plot(records_orig.index, records_orig, label=f"{series_orig.name}", color="#78909C",
                         alpha=.5, markersize=2, markeredgecolor='none')
            ax_both.plot(series_resampled.index, series_resampled, label="resampled",
                         color="#FFA726", alpha=1, markersize=3, markeredgecolor='none')

            # Heatmaps
            kwargs_heatmap = dict(cb_labelsize=10, minticks=3, maxticks=99)
            format_style_heatmap = FormatStyle(axlabel_fontsize=10, ticks_fontsize=10)
            HeatmapDateTime(series=series_orig).plot(ax=ax_heatmap_hires_before,
                                                     format_style=format_style_heatmap, **kwargs_heatmap)
            HeatmapDateTime(series=series_resampled).plot(ax=ax_heatmap_resampled_after,
                                                          format_style=format_style_heatmap, **kwargs_heatmap)

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
            p = TimeSeries(series=self._plot_records(field, self.series_hires_orig[field]))
            p.plot() if not interactive else p.plot_interactive()

    def showplot_cleaned(self, interactive: bool = False):
        """Show *current* cleaned high-resolution data"""
        for field in self.fields:
            p = TimeSeries(series=self._plot_records(field, self.series_hires_cleaned[field]))
            p.plot() if not interactive else p.plot_interactive()

    def _plot_records(self, field: str, series: pd.Series) -> pd.Series:
        """Return *series* ready for a line plot.

        With more than one time resolution, coarse records sit between empty
        grid slots and a line would not connect them, so only records are kept.
        Single-resolution data keep their gaps as breaks in the line.
        """
        if self.data_detailed[field]['FREQ_AUTO_SEC'].nunique() > 1:
            return series.dropna()
        return series

    def report_outlier_detection_qcf_evolution(self):
        """Print the QCF flag-evolution report."""
        for field in self.fields:
            self.outlier_detection_qcf[field].report_qcf_evolution()

    def report_outlier_detection_qcf_flags(self):
        """Print the QCF flags report."""
        for field in self.fields:
            self.outlier_detection_qcf[field].report_qcf_flags()

    def report_outlier_detection_qcf_series(self):
        """Print the QCF series report."""
        for field in self.fields:
            self.outlier_detection_qcf[field].report_qcf_series()

    def flag_missingvals_test(self, verbose: bool = False):
        """Flag missing values and add flag to dataframe.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            The flag goes straight into the data, no ``addflag()`` needed:

            >>> mscr.flag_missingvals_test()
        """
        for field in self.fields:
            flagtest = MissingValues(series=self.data_detailed[field][field].copy(), verbose=verbose)
            flagtest.calc(repeat=False)
            flag = flagtest.get_flag()
            # Empty slots inside a coarse record's period are not missing data,
            # only a real gap is; leave them unflagged so reports don't count them.
            flag = flag.mask(self._inside_coarse_record(field))
            self._data_detailed[field][flag.name] = flag

    def flag_manualremoval_test(self, remove_dates: list, showplot: bool = False, verbose: bool = False):
        """Flag specified records for removal.

        Dates are in the database's TIMESTAMP_END convention, i.e. as the records
        appear in the database, not as the TIMESTAMP_MIDDLE used during screening.
        Each entry is a single date(time) string or a ``[start, end]`` list; a
        bare date such as '2024-07-14' covers all records whose END timestamp is
        on that day. See ``ManualRemoval`` for the format.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example
            (10-minute data, END timestamps 2024-07-01 00:10 to 2024-07-05 00:00).
            Each call replaces the previous result until ``addflag()`` keeps it.

            A timestamp with time removes that one record. A bare date removes all
            records whose END timestamp falls on that day (144 records here):

            >>> mscr.flag_manualremoval_test(remove_dates=['2024-07-03 10:00'])
            >>> mscr.flag_manualremoval_test(remove_dates=['2024-07-03'])

            A range is a nested ``[start, end]`` list and includes both ends. With
            times it covers 06:00 to 09:30 (22 records); with bare dates it covers
            whole days, here all of 3 and 4 July (288 records):

            >>> mscr.flag_manualremoval_test(remove_dates=[['2024-07-03 06:00', '2024-07-03 09:30']])
            >>> mscr.flag_manualremoval_test(remove_dates=[['2024-07-03', '2024-07-04']])

            A flat list of two strings means two single records, not a range:

            >>> mscr.flag_manualremoval_test(remove_dates=['2024-07-03 06:00', '2024-07-03 09:30'])
            >>> flag = mscr.outlier_detection['TA_T1_2_1'].last_flag
            >>> int((flag == 2).sum())
            2
            >>> mscr.flag_manualremoval_test(remove_dates=[['2024-07-03 06:00', '2024-07-03 09:30']])
            >>> flag = mscr.outlier_detection['TA_T1_2_1'].last_flag
            >>> int((flag == 2).sum())
            22

            One list can mix all forms. Keep the result with ``addflag()``:

            >>> mscr.flag_manualremoval_test(remove_dates=[
            ...     '2024-07-03 10:00',                        # one record
            ...     '2024-07-04',                              # whole day
            ...     ['2024-07-03 06:00', '2024-07-03 09:30'],  # range with times
            ...     ['2024-07-01', '2024-07-02'],              # range of whole days
            ... ])
            >>> mscr.addflag()
        """
        for field in self.fields:
            sod = self.outlier_detection[field]
            sod.flag_manualremoval_test(
                remove_dates=self._end_dates_to_middle(index=sod.series_hires_cleaned.index,
                                                       remove_dates=remove_dates),
                showplot=showplot,
                verbose=verbose)

    @staticmethod
    def _end_dates_to_middle(index: pd.DatetimeIndex, remove_dates: list) -> list:
        """Translate TIMESTAMP_END date specs to ranges on the TIMESTAMP_MIDDLE *index*."""
        # Match on END timestamps so a bare date or a range selects exactly the
        # records the user sees in the database, then pass the matched records on
        # as explicit MIDDLE ranges.
        middle_by_end = pd.Series(index, index=index + pd.Timedelta(index.freq) / 2)
        converted = []
        for spec in remove_dates:
            if isinstance(spec, str):
                start = end = spec
            elif isinstance(spec, (list, tuple)) and len(spec) == 2:
                start, end = spec
            else:
                converted.append(spec)  # ManualRemoval reports the invalid entry
                continue
            matched = middle_by_end.loc[start:end]
            if not matched.empty:
                converted.append([str(matched.iloc[0]), str(matched.iloc[-1])])
        return converted

    def flag_outliers_localsd_test(self, n_sd: float = 7, winsize: int | str = None,
                                   showplot: bool = False, constant_sd: bool = False,
                                   separate_day_night: bool = False,
                                   verbose: bool = False, repeat: bool = True,
                                   n_sd_daytime: float = None, n_sd_nighttime: float = None,
                                   winsize_daytime: int | str = None, winsize_nighttime: int | str = None):
        """Identify outliers based on standard deviation in a rolling window.

        With ``separate_day_night=True``, ``n_sd_daytime``/``n_sd_nighttime`` and
        ``winsize_daytime``/``winsize_nighttime`` override ``n_sd`` and ``winsize``
        for one period; ``None`` uses the global value. See ``LocalSD``.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            ``winsize`` is a record count (144 is one day of 10-minute data) or a time span:

            >>> mscr.flag_outliers_localsd_test(n_sd=7, winsize=144)
            >>> mscr.flag_outliers_localsd_test(n_sd=7, winsize='1D')

            Separate daytime and nighttime, stricter at night:

            >>> mscr.flag_outliers_localsd_test(n_sd=7, winsize=144, separate_day_night=True,
            ...                                 n_sd_daytime=7, n_sd_nighttime=5)
            >>> mscr.addflag()
        """
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_localsd_test(
                n_sd=n_sd, winsize=winsize, separate_day_night=separate_day_night,
                n_sd_daytime=n_sd_daytime, n_sd_nighttime=n_sd_nighttime,
                winsize_daytime=winsize_daytime, winsize_nighttime=winsize_nighttime,
                constant_sd=constant_sd, showplot=showplot,
                verbose=verbose, repeat=repeat)

    def flag_outliers_increments_zcore_test(self, thres_zscore: int = 30, showplot: bool = False,
                                            verbose: bool = False, repeat: bool = True):
        """Identify outliers based on the z-score of record increments

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

            >>> mscr.flag_outliers_increments_zcore_test(thres_zscore=30)
            >>> mscr.addflag()
        """
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_increments_zcore_test(thres_zscore=thres_zscore,
                                                                              showplot=showplot,
                                                                              verbose=verbose,
                                                                              repeat=repeat)

    def flag_outliers_zscore_test(self,
                                  thres_zscore: float = 4,
                                  separate_day_night: bool = False,
                                  thres_zscore_daytime: float = None,
                                  thres_zscore_nighttime: float = None,
                                  showplot: bool = False,
                                  plottitle: str = None,
                                  verbose: bool = False,
                                  repeat: bool = True,
                                  idstr: str = None):
        r"""
        Flag outliers using z-score threshold (global or separate day/night).

        Applies z-score detection to identify values deviating from the mean by more than
        a specified number of standard deviations. Can operate globally across all records
        or separately for daytime and nighttime periods.

        Global mode: Computes mean and standard deviation across entire time series, then
        flags any value where \|z\| > threshold. Simple and fast, but ignores time-of-day
        variation in signal characteristics.

        Day/Night mode: Computes separate mean/std for daytime and nighttime records.
        Accounts for different signal distributions between periods (e.g., temperature,
        radiation, ecosystem fluxes often vary between day/night). Requires site location
        to determine day/night boundaries.

        Parameters
        ----------
        thres_zscore : float, default 4
            Z-score threshold for flagging. Typical range: 2.5-5. Values where \|z\| > threshold
            are flagged as outliers. Lower values (2.5-3) more aggressive; higher values (4-5)
            more conservative.
        separate_day_night : bool, default False
            If False, apply single threshold across all records (global mode).
            If True, apply separate thresholds to daytime and nighttime records.
            Day/night boundaries are derived from the site location (``site_lat``,
            ``site_lon``, ``utc_offset``) supplied when the class was initialized.
        thres_zscore_daytime : float, default None
            Override ``thres_zscore`` for daytime records (separate_day_night=True).
            If None, uses ``thres_zscore``.
        thres_zscore_nighttime : float, default None
            Override ``thres_zscore`` for nighttime records (separate_day_night=True).
            If None, uses ``thres_zscore``.
        showplot : bool, default False
            If True, display outlier visualization.
        plottitle : str, default None
            Title for plot. If None, auto-generated.
        verbose : bool, default False
            If True, print detection statistics (count of outliers, retention percentage).
            For day/night mode, prints separate statistics for each period.
        repeat : bool, default True
            If True, iteratively repeat detection until convergence (no new outliers detected).
            Useful for removing cascading outliers, but may over-filter.
        idstr : str, default None
            Optional identifier string for labeling output in verbose mode.

        Examples
        --------
        ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

        >>> mscr.flag_outliers_zscore_test(thres_zscore=4)

        Separate daytime and nighttime, stricter at night:

        >>> mscr.flag_outliers_zscore_test(separate_day_night=True,
        ...                                thres_zscore_daytime=4, thres_zscore_nighttime=3)
        >>> mscr.addflag()
        """
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_zscore_test(
                thres_zscore=thres_zscore,
                separate_day_night=separate_day_night,
                thres_zscore_daytime=thres_zscore_daytime,
                thres_zscore_nighttime=thres_zscore_nighttime,
                showplot=showplot,
                plottitle=plottitle,
                verbose=verbose,
                repeat=repeat,
                idstr=idstr
            )

    def flag_outliers_zscore_rolling_test(self, thres_zscore: float = 4, showplot: bool = False, verbose: bool = False,
                                          plottitle: str = None, repeat: bool = True, winsize: int | str = None):
        """Identify outliers based on the rolling z-score of records

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            ``winsize`` is a record count (144 is one day of 10-minute data) or a time span:

            >>> mscr.flag_outliers_zscore_rolling_test(thres_zscore=4, winsize=144)
            >>> mscr.flag_outliers_zscore_rolling_test(thres_zscore=4, winsize='1D')
            >>> mscr.addflag()
        """
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_zscore_rolling_test(thres_zscore=thres_zscore,
                                                                            showplot=showplot,
                                                                            verbose=verbose,
                                                                            plottitle=plottitle,
                                                                            repeat=repeat,
                                                                            winsize=winsize)

    def flag_outliers_hampel_test(self, window_length: int | str = 13, n_sigma: float = 5.5,
                                  n_sigma_daytime: float = None, n_sigma_nighttime: float = None,
                                  k: float = 1.4826, use_differencing: bool = True,
                                  separate_day_night: bool = True, showplot: bool = False,
                                  verbose: bool = False, repeat: bool = True):
        """Identify outliers in a sliding window based on the Hampel filter (global or separate day/night).

        Parameters
        ----------
        window_length : int or str, default 13
            Size of the sliding window for median/MAD calculation, as a record
            count or a time span such as ``'7D'`` (converted to records at the
            data frequency, of which it must be a whole multiple)
        n_sigma : float, default 5.5
            Threshold multiplier for global mode (number of MADs above median)
        n_sigma_daytime : float, optional
            Threshold for daytime data (when separate_day_night=True)
        n_sigma_nighttime : float, optional
            Threshold for nighttime data (when separate_day_night=True)
        k : float, default 1.4826
            Scaling factor for MAD (median absolute deviation)
        use_differencing : bool, default True
            If True, apply Hampel filter to differenced series (rate of change)
        separate_day_night : bool, default True
            If False, apply single threshold globally across all records.
            If True, apply separate thresholds for daytime and nighttime data.
        showplot : bool, default False
            If True, display visualization of flagged outliers
        verbose : bool, default False
            If True, print flagging statistics
        repeat : bool, default True
            If True, iteratively repeat detection until convergence

        Examples
        --------
        ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
        Daytime and nighttime are separate by default; ``window_length`` is a
        record count (144 is one day of 10-minute data) or a time span:

        >>> mscr.flag_outliers_hampel_test(window_length=144, n_sigma=5.5)
        >>> mscr.flag_outliers_hampel_test(window_length='1D', n_sigma=5.5)

        Stricter at night, or one threshold for all records:

        >>> mscr.flag_outliers_hampel_test(window_length=144,
        ...                                n_sigma_daytime=5.5, n_sigma_nighttime=4)
        >>> mscr.flag_outliers_hampel_test(window_length=144, n_sigma=5.5,
        ...                                separate_day_night=False)
        >>> mscr.addflag()
        """
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_hampel_test(
                window_length=window_length, n_sigma=n_sigma, n_sigma_daytime=n_sigma_daytime,
                n_sigma_nighttime=n_sigma_nighttime,
                k=k, use_differencing=use_differencing, separate_day_night=separate_day_night,
                showplot=showplot, verbose=verbose, repeat=repeat)

    def flag_outliers_trim_low_test(self, trim_daytime: bool = False, trim_nighttime: bool = False,
                                    lower_limit: float = None, showplot: bool = False, verbose: bool = False):
        """Flag values below a given absolute limit as outliers, then flag an
        equal number of datapoints at the high end as outliers.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            Trim nighttime values below 10 and as many of the highest nighttime values:

            >>> mscr.flag_outliers_trim_low_test(trim_nighttime=True, lower_limit=10)
            >>> mscr.addflag()
        """
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_trim_low_test(trim_daytime=trim_daytime,
                                                                      trim_nighttime=trim_nighttime,
                                                                      lower_limit=lower_limit,
                                                                      showplot=showplot,
                                                                      verbose=verbose)

    def flag_outliers_abslim_test(self, minval: float = None, maxval: float = None,
                                  separate_day_night: bool = False,
                                  minval_daytime: float = None, maxval_daytime: float = None,
                                  minval_nighttime: float = None, maxval_nighttime: float = None,
                                  showplot: bool = False, verbose: bool = False):
        """Identify outliers based on absolute limits (global or separate day/night).

        Parameters
        ----------
        minval : float, optional
            Minimum acceptable value. Applies to both periods unless overridden.
        maxval : float, optional
            Maximum acceptable value. Applies to both periods unless overridden.
        separate_day_night : bool, default False
            If False, apply single threshold globally across all records.
            If True, apply separate thresholds for daytime and nighttime data.
        minval_daytime, maxval_daytime : float, optional
            Override minval/maxval for daytime records. Setting either turns
            separate_day_night on.
        minval_nighttime, maxval_nighttime : float, optional
            Override minval/maxval for nighttime records. Setting either turns
            separate_day_night on.
        showplot : bool, default False
            If True, display visualization of flagged outliers
        verbose : bool, default False
            If True, print flagging statistics

        Examples
        --------
        ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

        >>> mscr.flag_outliers_abslim_test(minval=-30, maxval=50)

        Separate limits for daytime and nighttime:

        >>> mscr.flag_outliers_abslim_test(separate_day_night=True,
        ...                                minval_daytime=-20, maxval_daytime=50,
        ...                                minval_nighttime=-30, maxval_nighttime=35)
        >>> mscr.addflag()
        """
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_abslim_test(minval=minval,
                                                                    maxval=maxval,
                                                                    separate_day_night=separate_day_night,
                                                                    minval_daytime=minval_daytime,
                                                                    maxval_daytime=maxval_daytime,
                                                                    minval_nighttime=minval_nighttime,
                                                                    maxval_nighttime=maxval_nighttime,
                                                                    showplot=showplot,
                                                                    verbose=verbose)

    def flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float = 'auto',
                               separate_day_night: bool = False,
                               showplot: bool = False, verbose: bool = False, repeat: bool = False,
                               n_jobs: int = 1):
        """Local outlier factor detection (global or separate day/night).

        Identifies density-based outliers using k-nearest neighbors. Can operate globally
        across all records or separately for daytime and nighttime periods.

        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors for LOF calculation; auto-calculated if None
        contamination : float or 'auto', default 'auto'
            Expected fraction of outliers (float 0-1) or 'auto' for automatic detection
        separate_day_night : bool, default False
            If False, apply single LOF globally across all records.
            If True, apply separate LOF detection for daytime and nighttime data.
        showplot : bool, default False
            If True, display visualization of detected outliers
        verbose : bool, default False
            If True, print detection statistics
        repeat : bool, default False
            If True, repeat detection until no new outliers are found.
            Each pass flags the ``contamination`` fraction of the remaining records
            again, so repeating keeps removing valid data.
        n_jobs : int, default 1
            Number of parallel jobs (-1 uses all cores)

        Examples
        --------
        ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
        With a float ``contamination`` every pass flags that fraction again, so
        run a single pass with ``repeat=False``:

        >>> mscr.flag_outliers_lof_test(n_neighbors=20, contamination=0.01, repeat=False)
        >>> mscr.flag_outliers_lof_test(n_neighbors=20, contamination=0.01, repeat=False,
        ...                             separate_day_night=True)
        >>> mscr.addflag()
        """
        for field in self.fields:
            self.outlier_detection[field].flag_outliers_lof_test(
                n_neighbors=n_neighbors,
                contamination=contamination,
                separate_day_night=separate_day_night,
                showplot=showplot,
                verbose=verbose,
                repeat=repeat,
                n_jobs=n_jobs
            )

    def _set_corrected(self, field: str, series: pd.Series):
        """Store a corrected series; outlier tests run after this see the corrected values."""
        # Some corrections write every slot (the nighttime zero offset sets all of
        # the night to 0), but a slot without an original record must stay empty.
        series = series.where(self._has_record(field))
        self._series_hires_cleaned[field] = series
        if field in self._outlier_detection:
            self._outlier_detection[field].rebase_series(series)

    def _has_record(self, field: str) -> pd.Series:
        """True for grid slots that hold an original record, False for empty slots."""
        return self.data_detailed[field]['FREQ_AUTO_SEC'].notna()

    def _inside_coarse_record(self, field: str) -> pd.Series:
        """True for empty grid slots that lie within the period of a coarse record."""
        freq_sec = self.data_detailed[field]['FREQ_AUTO_SEC']
        index = freq_sec.index
        grid = pd.Timedelta(index.freq)
        # A record at MIDDLE m with resolution f covers the slots whose MIDDLE is
        # in (m + grid/2 - f, m]. Each empty slot looks at the next record.
        start = pd.Series(index + grid / 2 - pd.to_timedelta(freq_sec, unit='s'), index=index)
        next_start = start.where(freq_sec.notna()).bfill()
        return freq_sec.isna() & (next_start < index)

    def correction_remove_nighttime_zero_offset(self, showplot: bool = True):
        """Remove nighttime offset from variables that should be zero at night (e.g. radiation)

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example;
            in practice it would hold a radiation variable:

            >>> mscr.correction_remove_nighttime_zero_offset(showplot=False)
        """
        for field in self.fields:
            self._set_corrected(field, remove_nighttime_zero_offset(
                series=self._series_hires_cleaned[field],
                lat=self.site_lat, lon=self.site_lon,
                utc_offset=self.utc_offset, showplot=showplot))

    def correction_setto_max_threshold(self, threshold: float, showplot: bool = True):
        """Set values above threshold to threshold

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

            >>> mscr.correction_setto_max_threshold(threshold=35, showplot=False)
        """
        for field in self.fields:
            self._set_corrected(field, setto_threshold(
                series=self._series_hires_cleaned[field],
                threshold=threshold, type='max', showplot=showplot))

    def correction_set_exact_value_to_missing(self, values: list, verbose: int = 0, showplot: bool = True):
        """Set exact values to missing values

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            Set records that are exactly -9999 or 0 to missing:

            >>> mscr.correction_set_exact_value_to_missing(values=[-9999, 0], showplot=False)
        """
        for field in self.fields:
            self._set_corrected(field, set_exact_values_to_missing(
                series=self._series_hires_cleaned[field],
                values=values, showplot=showplot, verbose=verbose))

    def correction_setto_min_threshold(self, threshold: float, showplot: bool = True):
        """Set values below threshold to threshold

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

            >>> mscr.correction_setto_min_threshold(threshold=-5, showplot=False)
        """
        for field in self.fields:
            self._set_corrected(field, setto_threshold(
                series=self._series_hires_cleaned[field],
                threshold=threshold, type='min', showplot=showplot))

    def correction_setto_value(self, dates: list, value: float, verbose: int = 1):
        """Set records within time range to value.

        Dates are in the database's TIMESTAMP_END convention, as for
        ``flag_manualremoval_test``.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            Dates are END timestamps and take the same forms as in
            ``flag_manualremoval_test``: a timestamp is one record, a bare date is a
            whole day, a nested ``[start, end]`` list is a range with both ends
            included:

            >>> mscr.correction_setto_value(dates=['2024-07-03 10:00'], value=0)
            >>> mscr.correction_setto_value(dates=['2024-07-03'], value=0)
            >>> mscr.correction_setto_value(dates=[['2024-07-03 06:00', '2024-07-03 09:30']], value=0)
            >>> mscr.correction_setto_value(dates=[['2024-07-03', '2024-07-04']], value=0)

            As there, a flat list of two strings is two single records, not a range.
        """
        for field in self.fields:
            series = self._series_hires_cleaned[field]
            self._set_corrected(field, setto_value(
                series=series,
                dates=self._end_dates_to_middle(index=series.index, remove_dates=dates),
                value=value, verbose=verbose))

    def correction_remove_relativehumidity_offset(self, showplot: bool = True):
        """Remove the offset of relative humidity values above 100% and cap them at 100

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example;
            in practice it would hold relative humidity:

            >>> mscr.correction_remove_relativehumidity_offset(showplot=False)
        """
        for field in self.fields:
            self._set_corrected(field, remove_relativehumidity_offset(
                series=self._series_hires_cleaned[field], showplot=showplot))

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

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

            >>> corr = mscr.analysis_potential_radiation_correlation(utc_offset=1, mincorr=0.7,
            ...                                                      showplot=False)
            >>> daily = corr['TA_T1_2_1']  # one correlation per day
        """

        daily_correlations = {}
        for field in self.fields:
            series = self.series_hires_cleaned[field]
            # Calculate potential radiation SW_IN_POT
            swinpot = potrad(timestamp_index=series.index,
                             lat=self.site_lat,
                             lon=self.site_lon,
                             utc_offset=utc_offset)

            # Calculate daily correlation between potential and measured observation.
            # daily_correlation is the class DailyCorrelation, which does not plot from
            # its constructor, so showplot is honoured by calling plot() afterwards. The
            # dict holds the correlation Series, not the object, as documented above.
            daycorrs = daily_correlation(
                s1=series,
                s2=swinpot,
                mincorr=mincorr
            )
            if showplot:
                daycorrs.plot()
            daily_correlations[field] = daycorrs.correlations

        return daily_correlations

    def resample(self,
                 to_freqstr: str = '30min',
                 agg: Literal['mean', 'sum'] = 'mean',
                 mincounts_perc: float = .25):

        """Resample the screened series to the target frequency (default 30min,
        but any lower resolution, e.g. '10min', '1h').

        A target period ending at T collects the kept records whose END timestamp
        is in (T - period, T]. Each record weighs the time it covers, i.e. its
        original resolution: ``agg='mean'`` is the time-weighted mean,
        ``agg='sum'`` adds each record once. ``mincounts_perc`` is the minimum
        fraction of the period the kept records must cover, counted in slots of
        the high-resolution grid and rounded down; if that minimum is below three
        slots, one covered slot is enough (e.g. 10MIN data resampled to 30min).
        Periods below the minimum are NaN, so the result spans the whole screened
        range.

        A coarse record is assigned whole to the period that contains its END
        timestamp. This is exact when its resolution divides the target period.
        When it does not, e.g. 20MIN records resampled to 30min, a record that
        overlaps two periods is not split between them.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            The result, with updated tags, is in ``resampled_detailed``:

            >>> mscr.resample(to_freqstr='30min', agg='mean')
            >>> mscr.resampled_detailed['TA_T1_2_1'].index.freqstr
            '30min'

            Hourly sums (e.g. precipitation), each hour at least half covered:

            >>> mscr.resample(to_freqstr='1h', agg='sum', mincounts_perc=0.5)
        """
        if agg not in ('mean', 'sum'):
            raise ValueError(f"agg must be 'mean' or 'sum', not {agg!r}.")
        # pandas 3 no longer reads the old minute alias ('30T').
        if to_freqstr.endswith('T'):
            to_freqstr = f"{to_freqstr[:-1]}min"
        for field in self.fields:
            series = self._series_hires_cleaned[field]
            grid = series.index.freq
            if to_offset(to_freqstr) < grid:
                raise NotImplementedError(f"Upsampling not allowed: target frequency {to_freqstr} must be "
                                          f"a lower time resolution than the data ({grid.freqstr}).")
            info(f"Resampling {field} from {grid.freqstr} to {to_freqstr} ({agg}) ...")

            # A kept record weighs the grid slots its original resolution covers
            # (10 for a 10MIN record on a 1MIN grid); empty slots and rejected records weigh 0.
            grid_sec = pd.Timedelta(grid).total_seconds()
            weight = (self.data_detailed[field]['FREQ_AUTO_SEC'] / grid_sec).where(series.notna(), 0)

            # The index is TIMESTAMP_MIDDLE, so label='right' puts each record into
            # the period (T - to_freqstr, T] that contains its END timestamp.
            periods = dict(rule=to_freqstr, label='right')
            covered = weight.resample(**periods).sum()
            if agg == 'mean':
                series_resampled = (series * weight).resample(**periods).sum() / covered
            else:
                series_resampled = series.resample(**periods).sum()

            # Same minimum rule as resample_series_to_freq (truncated, at least one
            # slot below three), so periods are kept exactly as before this layout.
            maxcounts = pd.Series(1, index=series.index).resample(**periods).count().max()
            mincounts = int(maxcounts * mincounts_perc)
            mincounts = 1 if mincounts < 3 else mincounts
            series_resampled = series_resampled.where(covered >= mincounts)

            series_resampled = series_resampled.asfreq(to_freqstr)
            series_resampled.index.name = 'TIMESTAMP_END'
            series_resampled = TimestampSanitizer(data=series_resampled, output_middle_timestamp=False).get()

            # Update tags with resampling info
            self._tags[field]['freq'] = to_freqstr
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
                                   nighttime_accept_qcf_below: int = 2) -> None:

        """Finalize outlier detection and aggregate the QCF flag.

        Records with QCF=2 are removed from the current series. Corrections
        applied before this call are kept; corrections applied afterwards work
        on the filtered series. Calling it again after more tests replaces the
        previous QCF results.

        Args:
            daytime_accept_qcf_below: Accept daytime records where QCF is below
                this value. Default 2 keeps QCF=0 and QCF=1; 1 also rejects
                QCF=1 (records with soft flags). Daytime and nighttime come from
                potential radiation at the site (daytime above 20 W m-2).
            nighttime_accept_qcf_below: Same for nighttime records.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            Add one or more flags, then aggregate them:

            >>> mscr.flag_outliers_zscore_test(thres_zscore=4)
            >>> mscr.addflag()
            >>> mscr.finalize_outlier_detection()

            Stricter at night, also rejecting nighttime records with QCF=1:

            >>> mscr.finalize_outlier_detection(daytime_accept_qcf_below=2,
            ...                                 nighttime_accept_qcf_below=1)
        """
        for field in self.fields:
            # A previous run's QCF columns would otherwise stay in data_detailed,
            # since FlagQCF.get() only appends columns that are missing.
            previous = self._outlier_detection_qcf.get(field)
            if previous is not None:
                self._data_detailed[field] = self.data_detailed[field].drop(
                    columns=[previous.flagqcfcol, previous.sumflagscol, previous.sumhardflagscol,
                             previous.sumsoftflagscol, previous.filteredseriescol,
                             previous.filteredseriescol_hq],
                    errors='ignore')

            # Detect new columns
            newcols = frames.detect_new_columns(df=self.outlier_detection[field].flags,
                                                other=self.data_detailed[field])
            self._data_detailed[field] = pd.concat(
                [self.data_detailed[field], self.outlier_detection[field].flags[newcols]], axis=1)
            for col in newcols:
                detail(f"++Added new column {col}.")

            # Calculate overall quality flag QCF. FlagQCF auto-detects all
            # FLAG_*_TEST columns; idstr only names the output columns. Potential
            # radiation is only passed so the day/night accept thresholds take
            # effect; it is not kept in data_detailed.
            swinpot = potrad(timestamp_index=self.data_detailed[field].index,
                             lat=self.site_lat, lon=self.site_lon, utc_offset=self.utc_offset)
            qcf = FlagQCF(df=self.data_detailed[field].assign(**{_SWINPOT_COL: swinpot}),
                          target_col=field,
                          idstr='METSCR',
                          swinpot_col=_SWINPOT_COL)
            qcf.calculate(daytime_accept_qcf_below=daytime_accept_qcf_below,
                          nighttime_accept_qcf_below=nighttime_accept_qcf_below)
            self._data_detailed[field] = qcf.get().drop(columns=_SWINPOT_COL)
            self._outlier_detection_qcf[field] = qcf

            # Mask the current series instead of taking qcf.filteredseries, which
            # holds the raw values and would drop corrections made before this.
            self._series_hires_cleaned[field] = \
                self._series_hires_cleaned[field].mask(qcf.flagqcf == 2).where(
                    self._has_record(field)).rename(qcf.filteredseries.name)

    def addflag(self):
        """Add flag of most recent outlier test to data."""
        for field in self.fields:
            self._outlier_detection[field].addflag()

    def _validate_data_detailed(self, data_detailed, field) -> dict:
        """Setup variable (field) data for meteoscreening"""

        timestamp_name = data_detailed.index.name  # Get name of timestamp for later use
        self._check_units(data_detailed=data_detailed)
        self._check_fields(data_detailed=data_detailed)

        # Harmonize different time resolutions (one grid at the highest freq)
        groups = self._make_timeres_groups(data_detailed=data_detailed)
        group_counts = self._count_group_records(group_series=groups[field])
        targetfreq, used_freqs, rejected_freqs = self._validate_n_grouprecords(group_counts=group_counts)
        data_detailed = self._filter_data(data_detailed=data_detailed, used_freqs=used_freqs)
        data_detailed = self._harmonize_timeresolution(data_detailed=data_detailed,
                                                       timestamp_name=timestamp_name)
        data_detailed = self._sanitize_timestamp(targetfreq=targetfreq, data_detailed=data_detailed)

        return data_detailed

    @staticmethod
    def _sanitize_timestamp(targetfreq, data_detailed):
        """
        Set frequency info and sanitize timestamp

        This also converts the timestamp to TIMESTAMP_MIDDLE.
        """
        offset = to_offset(pd.Timedelta(seconds=targetfreq))
        data_detailed = data_detailed.asfreq(offset.freqstr)
        data_detailed = TimestampSanitizer(data=data_detailed).get()
        return data_detailed

    @staticmethod
    def _harmonize_timeresolution(data_detailed, timestamp_name: str) -> DataFrame:
        """
        Keep each record only at its own TIMESTAMP_END

        The grid at the highest resolution is created afterwards in
        `_sanitize_timestamp`. A record of a coarser resolution stays a single row
        at its END timestamp; the grid slots before it that its averaging interval
        covers stay empty (NaN in all columns, FREQ_AUTO_SEC included). Copying the
        value onto those slots would make a record removable only in part and
        would give difference-based outlier tests runs of zero increments.
        `resample` weights each record by its FREQ_AUTO_SEC instead.
        """
        data_detailed = data_detailed.sort_index(ascending=True)
        data_detailed.index.name = timestamp_name
        return data_detailed

    @staticmethod
    def _extract_tags(data_detailed, field) -> dict:
        """For each variable, extract tag columns from the respective DataFrame
         and store info in simplified dict"""
        tags_df = data_detailed.drop(columns=[field, 'FREQ_AUTO_SEC'])
        # tags_df.nunique()
        notags = tags_df.isnull().all(axis=1)
        tags_df = tags_df[~notags]  # Drop empty grid slots, which hold no record and no tags
        tags_dict = {}
        for tag in tags_df.columns:
            # dropna: data merged from tables with different tag sets leaves a tag
            # empty for some records, which must not end up as a literal 'nan'.
            list_of_vals = list(tags_df[tag].dropna().unique())
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
        info(f"Found {len(n_freqs)} unique frequencies across {n_vals} records.")
        info("Found frequencies:")
        cumulative_counts = 0
        used_freqs = []
        rejected_freqs = []
        for freq in n_freqs:
            counts = group_counts[freq]
            cumulative_counts += counts
            counts_perc = (counts / n_vals) * 100
            if counts_perc > 0.2:  # At least 0.2% of the data must have this resolution to be considered
                used_freqs.append(freq)
                info(f"  Found time resolution {freq} (seconds) with {counts} records "
                     f"({counts_perc:.2f}% of total records).")
            else:
                rejected_freqs.append(freq)
                # info, not detail: this is the only report of a resolution group being discarded,
                # and the records it names are dropped from the screening.
                info(f"  Found time resolution {freq} (seconds) with {counts} records "
                     f"({counts_perc:.2f}% of total records). --> Frequency will be ignored, too few records.")
        info(f"The following frequencies will be used: {used_freqs} (seconds)")
        targetfreq = min(used_freqs)
        if len(used_freqs) > 1:
            info(f"Note that there is more than one single time resolution. All records "
                 f"are placed on a grid at the highest found time resolution "
                 f"({targetfreq}S), each at its own END timestamp.")
        return targetfreq, used_freqs, rejected_freqs

    def _filter_data(self, data_detailed, used_freqs):
        keep = data_detailed['FREQ_AUTO_SEC'].isin(used_freqs)
        n_dropped = int((~keep).sum())
        if n_dropped:
            # Off-grid records and very small resolution groups are removed here;
            # without this message the loss would be invisible.
            warn(f"{n_dropped} records were removed because they do not belong to "
                 f"a used time resolution {used_freqs} (seconds).")
        return data_detailed.loc[keep]
