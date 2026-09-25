"""
METEOSCREENING: MULTI-STAGE METEOROLOGICAL SCREENING
====================================================

Multi-stage quality control and outlier detection for meteorological data.
Includes: outlier detection, data corrections, resampling, and quality flag generation.

Part of the diive library: https://github.com/holukas/diive
"""
from functools import partial
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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
from diive.preprocessing.outlier_detection.common import window_to_records
from diive.preprocessing.qaqc.flags import MissingValues
from diive.preprocessing.qaqc.measurements import (
    CORR_RADIATION_ZERO_OFFSET, CORR_RELATIVEHUMIDITY_OFFSET, CORR_SET_EXACT_TO_MISSING,
    CORR_SETTO_MAX, CORR_SETTO_MIN, CORR_SETTO_VALUE)
from diive.preprocessing.qaqc.qcf import FlagQCF

_SWINPOT_COL = '_SW_IN_POT_METSCR'  # Temporary day/night input for the QCF
# Records off the phase of the others may shrink the grid only this far. Legitimate
# phase shifts need about 2-10 slots per record (10MIN data moving from :00 to :05
# or :01); timestamps a second late need 60 (1MIN data) to 600 (10MIN data).
_MAX_SLOTS_PER_RECORD = 20


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
    For a full list of outlier tests see: diive/preprocessing/outlier_detection/stepwiseoutlierdetection.py
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
    - `.set_corrections()`: Replace all corrections with an ordered list of corrections (as used by the GUI)

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
    After quality-screening and corrections, `.resample()` resamples the data to a coarser
    time resolution (default 30min, e.g. also '1h' or '1D').

    **Handling different time resolutions**
    The time resolution of the raw data can change, e.g. from 10MIN for older data to 1MIN
    for newer data. All records are then placed on one grid, each **only at its own END
    timestamp** (the database timestamp is TIMESTAMP_END). The grid is the finest resolution,
    or finer where needed to hold every record (10MIN, then 15MIN data give a 5MIN grid). A
    coarse record is not copied onto the finer slots it covers: those slots stay empty through
    the whole screening, and corrections do not fill them. Records off the phase of the
    others, e.g. END timestamps a second late, would need a grid of seconds or less: if
    that grid would be finer than 1S or hold more than 20 slots per record, a ValueError
    asks to clean or resample the timestamps first.

    The rolling-window and difference tests (Hampel, local SD, rolling z-score, z-score of
    increments) run on each **resolution period** separately, i.e. on each run of records
    with the same resolution, at that resolution: a window given as a record count counts
    that period's records, a time span ('7D') covers that span, and day/night follows from
    the middle of each record's own period. A period shorter than the window is tested with
    the records it has, with a warning; a period with fewer than three records is left
    untested (flag NaN), with a warning. A time-span window must be a whole multiple of
    each period's resolution. The other tests run on all records at once.

    `.flag_manualremoval_test()` and `.correction_setto_value()` act on whole records and
    warn about dates that match no record's END timestamp. The missing-values flag counts
    a missing coarse record once, at its END. `.resample()` weights each record by the time
    it covers: a 10MIN record counts ten times as much as a 1MIN record in a mean, and once
    in a sum. Where a record's period overlaps the previous record's, the overlap counts once.

    **Timestamps**
    Screening runs on TIMESTAMP_MIDDLE (converted from the database's TIMESTAMP_END).
    With more than one time resolution the index is the middle of each grid slot,
    END - grid/2, which for a coarse record is not the middle of its own period. The
    tests that run on all records at once (e.g. z-score, absolute limits, LOF) classify
    a coarse record as day or night at this slot time. The rolling-window and difference
    tests and the day/night QCF thresholds use each record's own middle.
    `.flag_manualremoval_test()` and `.correction_setto_value()` take dates in the
    database's TIMESTAMP_END convention.

    **Site coordinates**
    `site_lat`, `site_lon` and `utc_offset` (the offset of the data timestamps to UTC)
    are attributes that are read when they are needed: by `.start_outlier_detection()`
    for the tests, by the corrections and by `.finalize_outlier_detection()`. New
    values assigned before `.start_outlier_detection()` therefore apply to the whole
    screening, without building the class again.

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
    approach, the stepwise screening could be adjusted to work with other data sources.

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
        # The corrected series before the QCF removes records. Every finalize starts
        # from it, so a looser re-run brings back what a stricter one removed.
        self._series_hires_corrected = {}
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
        """Return the resampled data as dict ``{field: DataFrame}``.

        Each DataFrame holds the resampled series and the tag columns, on a
        TIMESTAMP_END index. Filled by ``resample()``.
        """
        if not isinstance(self._resampled_detailed, dict):
            raise Exception("No resampled data available.")
        return self._resampled_detailed

    @property
    def tags(self) -> dict:
        """Return the database tags of each field as dict of dicts ``{field: {tag: value}}``.

        Each value is a string. A tag with more than one value among the records,
        e.g. ``freq`` of a download whose time resolution changes, holds all its
        values joined by commas in order of first appearance (``'10min,1min'``);
        missing values are skipped. ``resample()`` sets ``freq`` to the target
        frequency and ``data_version`` to ``'meteoscreening_diive'``.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example,
            which ran ``resample()``:

            >>> mscr.tags['TA_T1_2_1']['units']
            'degC'
            >>> mscr.tags['TA_T1_2_1']['freq']
            '30min'
        """
        if not isinstance(self._tags, dict):
            raise Exception("No tags available.")
        return self._tags

    def records_count(self, field: str = None) -> int:
        """Number of original records of *field*, with or without a value.

        These are the rows of the download that passed the time-resolution check,
        not the slots of the screening grid: with more than one time resolution
        the grid also holds empty slots inside coarse records.

        Args:
            field: Variable name; may be left out when there is only one field.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example
            (four days of 10-minute records):

            >>> mscr.records_count()
            576
        """
        return int(self._has_record(self._one_field(field)).sum())

    def resolutions(self, field: str = None) -> list[str]:
        """Time resolutions of the records of *field* as offset strings.

        In order of first appearance, e.g. ``['10min', '1min']`` for a download
        that switches from 10-minute to 1-minute records.

        Args:
            field: Variable name; may be left out when there is only one field.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

            >>> mscr.resolutions()
            ['10min']
        """
        freq_sec = self.data_detailed[self._one_field(field)]['FREQ_AUTO_SEC'].dropna()
        offsets = (to_offset(pd.Timedelta(int(round(sec * 1e9)), unit='ns')) for sec in pd.unique(freq_sec))
        # With the count, e.g. '1min' where freqstr gives 'min'.
        return [f"{o.n}{o.name}" for o in offsets]

    def _one_field(self, field: str | None) -> str:
        """Return *field*, or the only field when *field* is None."""
        if field is not None:
            return field
        if len(self.fields) > 1:
            raise ValueError(f"More than one field ({', '.join(self.fields)}), pass field=.")
        return self.fields[0]

    def showplot_outlier_detection_cleaned(self, interactive: bool = False):
        """Show cleaned data from outlier detection."""
        for field in self.fields:
            series = self._display_lines(field, self.outlier_detection[field].series_hires_cleaned)
            p = TimeSeries(series=series)
            p.plot() if not interactive else p.plot_interactive()

    def showplot_outlier_detection_qcf_heatmaps(self, **kwargs):
        """Show the QCF outlier-detection heatmaps."""
        for field in self.fields:
            qcf = self.outlier_detection_qcf[field]
            qcf.showplot_qcf_heatmaps(flags=self._display_heatmap(field, qcf.flags), **kwargs)

    def showplot_outlier_detection_qcf_timeseries(self, **kwargs):
        """Show the QCF outlier-detection time series."""
        for field in self.fields:
            qcf = self.outlier_detection_qcf[field]
            qcf.showplot_qcf_timeseries(flags=self._display_lines(field, qcf.flags), **kwargs)

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
            records_orig = self._display_lines(field, series_orig)
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
            hires_before = self._display_heatmap(field, series_orig)
            HeatmapDateTime(series=hires_before).plot(ax=ax_heatmap_hires_before,
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
            p = TimeSeries(series=self._display_lines(field, self.series_hires_orig[field]))
            p.plot() if not interactive else p.plot_interactive()

    def showplot_cleaned(self, interactive: bool = False):
        """Show *current* cleaned high-resolution data"""
        for field in self.fields:
            p = TimeSeries(series=self._display_lines(field, self.series_hires_cleaned[field]))
            p.plot() if not interactive else p.plot_interactive()

    def display_lines(self, data: pd.Series | DataFrame, field: str = None) -> pd.Series | DataFrame:
        """Return a copy of *data* ready for a line plot. For display only.

        *data* is on the screening grid of *field*, e.g. ``series_hires_cleaned[field]``
        or the flags of the QCF. With more than one time resolution the empty slots
        inside coarse records are left out, so a line joins neighbouring records,
        while a missing or removed record stays NaN and breaks the line. Data with
        one record per slot come back unchanged. Do not screen or resample the
        result.

        Args:
            data: Series or DataFrame on the grid of *field*.
            field: Variable name; may be left out when there is only one field.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            With one resolution the copy equals the data:

            >>> series = mscr.series_hires_cleaned['TA_T1_2_1']
            >>> mscr.display_lines(series).equals(series)
            True
        """
        return self._display_lines(self._one_field(field), data).copy()

    def display_heatmap(self, data: pd.Series | DataFrame, field: str = None) -> pd.Series | DataFrame:
        """Return a copy of *data* ready for a heatmap. For display only.

        *data* is on the screening grid of *field*. With more than one time
        resolution each empty slot inside a coarse record takes the record's value,
        so the record fills its whole period. Data with one record per slot come
        back unchanged. Do not screen or resample the result: it holds copies of
        records.

        Args:
            data: Series or DataFrame on the grid of *field*.
            field: Variable name; may be left out when there is only one field.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

            >>> series = mscr.series_hires_cleaned['TA_T1_2_1']
            >>> mscr.display_heatmap(series).equals(series)
            True
        """
        return self._display_heatmap(self._one_field(field), data).copy()

    def _display_lines(self, field: str, data: pd.Series | DataFrame) -> pd.Series | DataFrame:
        """Return *data* (on the grid of *field*) ready for a line plot. For display only.

        The empty slots inside the period of a record are left out, so a line
        joins neighbouring records. The slot of a missing or removed record stays
        (NaN) and still breaks the line. Data with one record per slot are
        returned as they are.
        """
        not_a_record = self._not_a_record(field).to_numpy()
        if not not_a_record.any():
            return data
        return data.loc[~not_a_record]

    def _display_heatmap(self, field: str, data: pd.Series | DataFrame) -> pd.Series | DataFrame:
        """Return a copy of *data* (on the grid of *field*) ready for a heatmap. For display only.

        Each empty slot inside the period of a record takes that record's value,
        so a coarse record fills its whole period instead of a single slot. The
        record is the next slot that is not such an empty slot; the slot of a
        missing record passes its NaN on in the same way. Data with one record
        per slot are returned as they are.
        """
        not_a_record = self._not_a_record(field).to_numpy()
        if not not_a_record.any():
            return data
        position = np.arange(len(not_a_record), dtype=float)
        owner = pd.Series(np.where(not_a_record, np.nan, position)).bfill().to_numpy()
        has_owner = ~np.isnan(owner)
        shown = data.iloc[np.where(has_owner, owner, 0).astype(int)].copy()
        shown.index = data.index
        if not has_owner.all():
            shown.loc[~has_owner] = np.nan
        return shown

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
            flag = self._missingvals_flag(field, verbose=verbose)
            self._data_detailed[field][flag.name] = flag

    def _missingvals_flag(self, field: str, verbose: bool = False) -> pd.Series:
        """Missing-values flag of *field*: 2 per missing record, NaN at slots that are no record."""
        flagtest = MissingValues(series=self.data_detailed[field][field].copy(), verbose=verbose)
        flagtest.calc(repeat=False)
        # A missing coarse record is one missing record, not one per grid slot it
        # covers, and the empty slots inside a present record are not missing at
        # all. Leave those slots unflagged so the reports count records.
        return flagtest.get_flag().mask(self._not_a_record(field))

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
            self.outlier_detection[field].flag_manualremoval_test(
                remove_dates=self._end_dates_to_middle(field=field, remove_dates=remove_dates),
                showplot=showplot,
                verbose=verbose)

    def _end_dates_to_middle(self, field: str, remove_dates: list) -> list:
        """Translate TIMESTAMP_END date specs to ranges of grid slots (TIMESTAMP_MIDDLE).

        Warns about each entry that matches no record, also when the caller is not
        verbose: the entry is a mistake in the input, not a result to report.
        """
        # Match on the END timestamps of records only, so a bare date or a range
        # selects exactly the records the user sees in the database, then pass the
        # matched records on as explicit MIDDLE ranges.
        index = self.data_detailed[field].index
        middle_by_end = pd.Series(index, index=index + pd.Timedelta(index.freq) / 2)
        middle_by_end = middle_by_end[self._has_record(field).to_numpy()]
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
            if matched.empty:
                # A typo, or a time inside a coarse record instead of its END.
                warn(f"{field}: no record has its END timestamp in {spec!r}, "
                     f"this entry changes nothing.")
            else:
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
        self._run_per_period(
            'flag_outliers_localsd_test', windows=(winsize, winsize_daytime, winsize_nighttime),
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
        self._run_per_period('flag_outliers_increments_zcore_test', windows=(),
                             thres_zscore=thres_zscore, showplot=showplot,
                             verbose=verbose, repeat=repeat)

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
        self._run_per_period('flag_outliers_zscore_rolling_test', windows=(winsize,),
                             thres_zscore=thres_zscore, showplot=showplot, verbose=verbose,
                             plottitle=plottitle, repeat=repeat, winsize=winsize)

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
        self._run_per_period(
            'flag_outliers_hampel_test', windows=(window_length,),
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

    def flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float | str = 'auto',
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
            # After finalize, tests run on the corrected series before the QCF
            # removed records, as they do without a correction in between: a later
            # finalize decides from all flags, so a record a looser run brings back
            # must have been tested. Records rejected by added flags stay removed.
            self._outlier_detection[field].rebase_series(self._series_hires_corrected.get(field, series))

    def _has_record(self, field: str) -> pd.Series:
        """True for grid slots that hold an original record, False for empty slots."""
        return self.data_detailed[field]['FREQ_AUTO_SEC'].notna()

    def _not_a_record(self, field: str) -> pd.Series:
        """True for empty grid slots that stand for no record, present or missing.

        A record covers the slots of its period that end at or before its END. An
        empty slot stands for a missing record where a record of the next
        record's resolution would end, counted back from the next record. Inside
        one resolution period these are the ENDs of that period's missing records;
        a gap between two periods is counted at the resolution of the later one.
        All other empty slots lie inside the period of a record and are no record.
        With one record per slot every empty slot is a missing record.
        """
        freq_sec = self.data_detailed[field]['FREQ_AUTO_SEC']
        slot = self.data_detailed[field].index.to_series()
        to_next_record = (slot.where(freq_sec.notna()).bfill() - slot).dt.total_seconds()
        return freq_sec.isna() & (to_next_record % freq_sec.bfill() != 0)

    def _resolution_periods(self, field: str) -> list:
        """Split the records of *field* into resolution periods.

        A resolution period is a run of consecutive records with the same time
        resolution; empty slots do not interrupt it. A record whose END is not a
        whole number of steps after the previous record's also starts a new
        period, so each period fits a regular grid at its own resolution.

        Returns:
            One ``(slots, middles, resolution)`` tuple per period: the grid slots
            of its records, their true TIMESTAMP_MIDDLE (END - resolution / 2)
            and the resolution as a Timedelta.
        """
        freq_sec = self.data_detailed[field]['FREQ_AUTO_SEC'].dropna()
        slots = freq_sec.index
        res_ns = np.round(freq_sec.to_numpy() * 1e9).astype(np.int64)
        step_ns = np.diff(slots.as_unit('ns').asi8)
        new = np.r_[True, (res_ns[1:] != res_ns[:-1]) | (step_ns % res_ns[1:] != 0)]
        # Slots are END - grid/2, so the true middle is the slot + grid/2 - resolution/2.
        grid = pd.Timedelta(self.data_detailed[field].index.freq)
        middles = slots + grid / 2 - pd.to_timedelta(res_ns, unit='ns') / 2
        periods = []
        for period in np.split(np.arange(len(slots)), np.flatnonzero(new)[1:]):
            periods.append((slots[period], middles[period], pd.Timedelta(res_ns[period[0]], unit='ns')))
        return periods

    def _run_per_period(self, method: str, windows: tuple, **kwargs):
        """Run the rolling-window or difference test *method* of StepwiseOutlierDetection.

        With one resolution period the test runs as usual. With more, it runs on
        each period on its own, on a regular grid at the period's resolution with
        the records at their true middle: a window given as a record count counts
        that period's records, a time span such as '7D' covers that span,
        differences are taken between neighbouring records, and day/night follows
        from the true middle. The flags are mapped back to the records' slots and
        become the test result for `.addflag()`.

        A period with fewer than three records is left untested (flag NaN). The
        warnings about short and untested periods are shown at any ``verbose``.

        Args:
            method: Name of the ``flag_*`` method of StepwiseOutlierDetection.
            windows: The window arguments among *kwargs* (None ignored), to warn
                when a period is shorter than a window.
            **kwargs: Passed unchanged to *method*.

        Raises:
            ValueError: If a window given as a time span is not a whole multiple
                of the resolution of a period; the message names the period.
        """
        verbose = kwargs.get('verbose', False)

        def listed(labels: list) -> str:
            return '; '.join(labels[:5]) + ('; ...' if len(labels) > 5 else '')

        for field in self.fields:
            sod = self.outlier_detection[field]
            periods = self._resolution_periods(field)
            if len(periods) == 1:
                getattr(sod, method)(**kwargs)
                continue

            series = sod.series_hires_cleaned
            flag = pd.Series(np.nan, index=series.index)
            flagname = None
            short, untested = [], []
            for n, (slots, middles, res) in enumerate(periods, start=1):
                label = f"{middles[0] + res / 2} to {middles[-1] + res / 2} (END, {to_offset(res).freqstr})"
                if len(slots) < 3:
                    # One record has no time axis, and with two every difference-based
                    # or robust statistic is degenerate: flag 0 would claim a test passed.
                    untested.append(f"{label}, {len(slots)} record(s)")
                    continue
                grid = pd.date_range(middles[0], middles[-1], freq=res, name='TIMESTAMP_MIDDLE')
                values = pd.Series(series.loc[slots].to_numpy(), index=middles, name=field).reindex(grid)
                try:
                    too_short = max([window_to_records(w, values) or 0 for w in windows], default=0) > len(grid)
                except ValueError as err:
                    raise ValueError(
                        f"{field}: resolution period {n} of {len(periods)}, {label}: {err} With more "
                        f"than one time resolution a time-span window is converted to the records of "
                        f"each resolution period, so it must be a whole multiple of every period's "
                        f"resolution.") from err
                if too_short:
                    short.append(f"{label}, {len(grid)} records")
                info(f"{field}: resolution period {n} of {len(periods)}: {label}", verbose=verbose)
                part = StepwiseOutlierDetection(dfin=values.to_frame(), col=field, site_lat=self.site_lat,
                                                site_lon=self.site_lon, utc_offset=self.utc_offset)
                getattr(part, method)(**kwargs)
                flag.loc[slots] = part.last_flag.reindex(middles).to_numpy()
                flagname = part.last_flag.name
            # No verbose= on these warnings: they report records that were tested
            # differently or not at all, which the caller must know at any verbosity.
            if short:
                warn(f"{field}: {len(short)} resolution period(s) shorter than the window, tested "
                     f"with the records they have: {listed(short)}")
            if flagname is None:
                # addflag() still needs the test's flag name. Three empty slots of the
                # grid give it without testing anything.
                empty = pd.Series(np.nan, index=series.index[:3], name=field)
                dummy = StepwiseOutlierDetection(dfin=empty.to_frame(), col=field, site_lat=self.site_lat,
                                                 site_lon=self.site_lon, utc_offset=self.utc_offset)
                getattr(dummy, method)(**{**kwargs, 'showplot': False, 'verbose': False})
                flagname = dummy.last_flag.name
                warn(f"{field}: all {len(periods)} resolution periods have fewer than 3 records, "
                     f"no record was tested (flag {flagname} is empty): {listed(untested)}")
            elif untested:
                warn(f"{field}: {len(untested)} resolution period(s) with fewer than 3 records, "
                     f"left untested: {listed(untested)}")
            sod.set_pending_flag(flag.rename(flagname))

    def _correct(self, field: str, correction, **kwargs):
        """Apply *correction*, a function taking ``series=``, to the current series of *field*."""
        corrected = correction(series=self._series_hires_cleaned[field], **kwargs)
        if field in self._series_hires_corrected:
            # After finalize the current series lacks the records the QCF removed.
            # Correct those too, quietly, so a later finalize with looser
            # thresholds brings them back corrected. The records kept keep the
            # values corrected above.
            quiet = {key: (0 if key == 'verbose' else False if key == 'showplot' else value)
                     for key, value in kwargs.items()}
            removed = self._outlier_detection_qcf[field].flagqcf == 2
            unmasked = correction(series=self._series_hires_corrected[field], **quiet)
            self._series_hires_corrected[field] = corrected.mask(removed, unmasked).where(self._has_record(field))
        self._set_corrected(field, corrected)

    def _covered_seconds(self, field: str) -> pd.Series:
        """Seconds each record covers, NaN at empty slots.

        A record covers its resolution, except where its period overlaps the
        previous record's (e.g. 1MIN records up to END 00:05, then a 10MIN record
        END 00:10): the overlap is counted once, with the previous record.
        """
        freq_sec = self.data_detailed[field]['FREQ_AUTO_SEC']
        records = freq_sec.dropna()
        since_previous = records.index.to_series().diff().dt.total_seconds()
        covered = records.where(~(since_previous < records), since_previous)
        return covered.reindex(freq_sec.index)

    def correction_remove_nighttime_zero_offset(self, showplot: bool = True, clamp_negatives: bool = True):
        """Remove nighttime offset from variables that should be zero at night (e.g. radiation)

        With ``clamp_negatives=True`` negative values left after the correction are
        set to zero. See ``remove_nighttime_zero_offset``.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example;
            in practice it would hold a radiation variable:

            >>> mscr.correction_remove_nighttime_zero_offset(showplot=False)
        """
        for field in self.fields:
            self._correct(field, remove_nighttime_zero_offset,
                          lat=self.site_lat, lon=self.site_lon,
                          utc_offset=self.utc_offset, showplot=showplot,
                          clamp_negatives=clamp_negatives)

    def correction_setto_max_threshold(self, threshold: float, showplot: bool = True):
        """Set values above threshold to threshold

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

            >>> mscr.correction_setto_max_threshold(threshold=35, showplot=False)
        """
        for field in self.fields:
            self._correct(field, setto_threshold, threshold=threshold, type='max', showplot=showplot)

    def correction_set_exact_value_to_missing(self, values: list, verbose: int = 0, showplot: bool = True):
        """Set exact values to missing values

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example.
            Set records that are exactly -9999 or 0 to missing:

            >>> mscr.correction_set_exact_value_to_missing(values=[-9999, 0], showplot=False)
        """
        for field in self.fields:
            self._correct(field, set_exact_values_to_missing,
                          values=values, showplot=showplot, verbose=verbose)

    def correction_setto_min_threshold(self, threshold: float, showplot: bool = True):
        """Set values below threshold to threshold

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

            >>> mscr.correction_setto_min_threshold(threshold=-5, showplot=False)
        """
        for field in self.fields:
            self._correct(field, setto_threshold, threshold=threshold, type='min', showplot=showplot)

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
            self._correct(field, setto_value,
                          dates=self._end_dates_to_middle(field=field, remove_dates=dates),
                          value=value, verbose=verbose)

    def correction_remove_relativehumidity_offset(self, showplot: bool = True):
        """Remove the offset of relative humidity values above 100% and cap them at 100

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example;
            in practice it would hold relative humidity:

            >>> mscr.correction_remove_relativehumidity_offset(showplot=False)
        """
        for field in self.fields:
            self._correct(field, remove_relativehumidity_offset, showplot=showplot)

    def set_corrections(self, corrections: list[dict], showplot: bool = False):
        """Replace all corrections with *corrections*, applied in order to the uncorrected data.

        Afterwards the applied corrections are exactly this list: earlier calls of
        ``set_corrections()`` or of the ``correction_*`` methods are undone first,
        so calling it twice with the same list gives the same result as once, and
        an empty list removes all corrections. Each entry runs through its
        ``correction_*`` method, so empty grid slots stay empty and the dates of
        ``'setto_value'`` are END timestamps.

        It works at any stage, as the ``correction_*`` methods do: before
        ``start_outlier_detection()``, between tests (records rejected by added
        flags stay removed) and after ``finalize_outlier_detection()`` (records
        the QCF removed stay removed, and a later finalize starts from the newly
        corrected data). After finalize the corrections work on the records the
        QCF kept, as a ``correction_*`` call there would.

        Args:
            corrections: Ordered ``{"key": str, "kwargs": dict}`` dicts, the format
                of ``apply_corrections``. ``key`` is a ``CORR_*`` constant of
                ``diive.preprocessing.qaqc.measurements``; ``kwargs`` are
                ``{"clamp_negatives": bool}`` (optional) for
                ``'radiation_zero_offset'``, none for ``'relativehumidity_offset'``,
                ``{"threshold": float}`` for ``'setto_max'`` and ``'setto_min'``,
                ``{"dates": list, "value": float}`` (value default 0) for
                ``'setto_value'`` and ``{"values": list}`` for ``'set_exact_to_missing'``.
            showplot: Show a plot for each correction that has one.

        Raises:
            ValueError: If a key is unknown. Nothing is changed then.

        Example:
            ``mscr`` is a ``StepwiseMeteoScreeningDb`` built as in the class example:

            >>> from diive.preprocessing.qaqc.measurements import CORR_SETTO_MAX, CORR_SETTO_MIN
            >>> mscr.set_corrections([{'key': CORR_SETTO_MAX, 'kwargs': {'threshold': 18}},
            ...                       {'key': CORR_SETTO_MIN, 'kwargs': {'threshold': 12}}])
            >>> float(mscr.series_hires_cleaned['TA_T1_2_1'].max())
            18.0

            A shorter list keeps only its own corrections:

            >>> mscr.set_corrections([{'key': CORR_SETTO_MIN, 'kwargs': {'threshold': 12}}])
            >>> float(mscr.series_hires_cleaned['TA_T1_2_1'].max()) > 18
            True
            >>> mscr.set_corrections([])  # no corrections
        """
        calls = []
        # All calls are built before anything is undone, so an unknown key or a
        # missing argument leaves the current corrections in place.
        for correction in corrections:
            key = correction['key']
            kwargs = correction.get('kwargs', {})
            if key == CORR_RADIATION_ZERO_OFFSET:
                call = partial(self.correction_remove_nighttime_zero_offset, showplot=showplot,
                               clamp_negatives=kwargs.get('clamp_negatives', True))
            elif key == CORR_RELATIVEHUMIDITY_OFFSET:
                call = partial(self.correction_remove_relativehumidity_offset, showplot=showplot)
            elif key == CORR_SETTO_MAX:
                call = partial(self.correction_setto_max_threshold, threshold=kwargs['threshold'],
                               showplot=showplot)
            elif key == CORR_SETTO_MIN:
                call = partial(self.correction_setto_min_threshold, threshold=kwargs['threshold'],
                               showplot=showplot)
            elif key == CORR_SETTO_VALUE:
                call = partial(self.correction_setto_value, dates=kwargs['dates'],
                               value=kwargs.get('value', 0))
            elif key == CORR_SET_EXACT_TO_MISSING:
                call = partial(self.correction_set_exact_value_to_missing, values=kwargs['values'],
                               showplot=showplot)
            else:
                raise ValueError(f"Unknown correction key: {key!r}")
            calls.append(call)

        for field in self.fields:
            uncorrected = self._series_hires_orig[field]
            if field in self._series_hires_corrected:
                # After finalize: the next finalize starts from the uncorrected
                # records, and the current series keeps the QCF removals.
                self._series_hires_corrected[field] = uncorrected
                removed = self._outlier_detection_qcf[field].flagqcf == 2
                current = uncorrected.mask(removed).rename(self._series_hires_cleaned[field].name)
            else:
                current = uncorrected
            self._set_corrected(field, current)
        for call in calls:
            call()

    def analysis_potential_radiation_correlation(self,
                                                 utc_offset: int,
                                                 mincorr: float = 0.7,
                                                 showplot: bool = True) -> dict:
        """Compare time series to potential radiation

        Args:
            utc_offset: UTC offset of the series timestamps, used for potential
                radiation. For example, for European winter time *utc_offset=1*.
                Taken from this argument, not from ``self.utc_offset``.
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
                if self._not_a_record(field).any():
                    # Plot from display copies so the line panels join the records;
                    # the correlations are the same, as the rows left out have no value.
                    daily_correlation(s1=self._display_lines(field, series),
                                      s2=self._display_lines(field, swinpot),
                                      mincorr=mincorr).plot()
                else:
                    daycorrs.plot()
            daily_correlations[field] = daycorrs.correlations

        return daily_correlations

    def resample(self,
                 to_freqstr: str = '30min',
                 agg: Literal['mean', 'sum'] = 'mean',
                 mincounts_perc: float = .25):

        """Resample the screened series to the target frequency (default 30min,
        but any lower resolution, e.g. '10min', '1h', '1D').

        A target period ending at T collects the kept records whose END timestamp
        is in (T - period, T]. Each record weighs the time it covers, i.e. its
        original resolution, or less where its period overlaps the previous
        record's (the overlap counts once): ``agg='mean'`` is the time-weighted mean,
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
            # As durations: pandas 3 cannot compare a Day offset with a Minute one.
            if pd.Timedelta(to_offset(to_freqstr).nanos, unit='ns') < pd.Timedelta(grid):
                raise NotImplementedError(f"Upsampling not allowed: target frequency {to_freqstr} must be "
                                          f"a lower time resolution than the data ({grid.freqstr}).")
            info(f"Resampling {field} from {grid.freqstr} to {to_freqstr} ({agg}) ...")

            # A kept record weighs the grid slots its original resolution covers
            # (10 for a 10MIN record on a 1MIN grid, less where it overlaps the
            # previous record); empty slots and rejected records weigh 0.
            grid_sec = pd.Timedelta(grid).total_seconds()
            weight = (self._covered_seconds(field) / grid_sec).where(series.notna(), 0)

            # The index is TIMESTAMP_MIDDLE, so label='right' puts each record into
            # the period (T - to_freqstr, T] that contains its END timestamp.
            periods = dict(rule=to_freqstr, label='right')
            covered = weight.resample(**periods).sum()
            if agg == 'mean':
                series_resampled = (series * weight).resample(**periods).sum() / covered
            else:
                series_resampled = series.resample(**periods).sum()

            # Same minimum rule as resample_series_to_freq (truncated, at least one
            # slot below three), so periods are kept as resample_series_to_freq
            # keeps them for single-resolution data.
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
        on the filtered series. Calling it again after more tests, or with other
        thresholds, replaces the previous QCF results: each run starts from the
        corrected series before any record was removed, so looser thresholds
        bring back records a stricter run removed. Tests run after this call,
        with or without corrections in between, also test the records the QCF
        removed (not those rejected by an added flag), so every record a later
        run brings back has been through all tests.

        With more than one time resolution the missing-values flag is needed to
        tell records from empty grid slots, so it is added here if
        ``flag_missingvals_test()`` was not run.

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

            # FlagQCF counts a row as a record only where the missing-values flag
            # is not NaN. Without that flag, the empty slots between coarse records
            # would count as records that passed every test.
            if self._not_a_record(field).any():
                missing = self._missingvals_flag(field)
                if missing.name not in self.data_detailed[field].columns:
                    self._data_detailed[field][missing.name] = missing
                    info(f"Added the missing-values flag {missing.name}, which tells records "
                         f"from empty grid slots in the QCF.")

            # Detect new columns
            newcols = frames.detect_new_columns(df=self.outlier_detection[field].flags,
                                                other=self.data_detailed[field])
            self._data_detailed[field] = pd.concat(
                [self.data_detailed[field], self.outlier_detection[field].flags[newcols]], axis=1)
            for col in newcols:
                detail(f"++Added new column {col}.")

            # Calculate overall quality flag QCF. FlagQCF uses every FLAG_*_TEST
            # column of the field; idstr only names the output columns. Potential
            # radiation is only passed so the day/night accept thresholds take
            # effect; it is not kept in data_detailed.
            swinpot = self._potential_radiation(field)
            qcf = FlagQCF(df=self.data_detailed[field].assign(**{_SWINPOT_COL: swinpot}),
                          target_col=field,
                          idstr='METSCR',
                          swinpot_col=_SWINPOT_COL)
            qcf.calculate(daytime_accept_qcf_below=daytime_accept_qcf_below,
                          nighttime_accept_qcf_below=nighttime_accept_qcf_below)
            self._data_detailed[field] = qcf.get().drop(columns=_SWINPOT_COL)
            self._outlier_detection_qcf[field] = qcf

            # Mask the corrected series instead of taking qcf.filteredseries, which
            # holds the raw values and would drop corrections made before this.
            # Start from the series before any previous finalize removed records.
            corrected = self._series_hires_corrected.setdefault(field, self._series_hires_cleaned[field])
            self._series_hires_cleaned[field] = corrected.mask(qcf.flagqcf == 2).where(
                self._has_record(field)).rename(qcf.filteredseries.name)

    def _potential_radiation(self, field: str) -> pd.Series:
        """Potential radiation at the grid slots of *field*, for the day/night QCF thresholds.

        A record gets the mean over its own period. With more than one resolution
        period the slot of a coarse record is not the middle of its period, so
        each period is evaluated on its own grid of true middles.
        """
        coords = dict(lat=self.site_lat, lon=self.site_lon, utc_offset=self.utc_offset)
        swinpot = potrad(timestamp_index=self.data_detailed[field].index, **coords)
        periods = self._resolution_periods(field)
        if len(periods) > 1:
            for slots, middles, res in periods:
                # potrad needs two timestamps to know the averaging period; a
                # single-record period keeps the value at its slot.
                if len(slots) > 1:
                    grid = pd.date_range(middles[0], middles[-1], freq=res)
                    swinpot.loc[slots] = potrad(timestamp_index=grid, **coords).reindex(middles).to_numpy()
        return swinpot

    def addflag(self):
        """Add flag of most recent outlier test to data."""
        for field in self.fields:
            self._outlier_detection[field].addflag()

    def _validate_data_detailed(self, data_detailed, field) -> DataFrame:
        """Setup variable (field) data for meteoscreening"""

        timestamp_name = data_detailed.index.name  # Get name of timestamp for later use
        self._check_units(data_detailed=data_detailed)
        self._check_fields(data_detailed=data_detailed)

        # Harmonize different time resolutions (one grid that holds every record)
        groups = self._make_timeres_groups(data_detailed=data_detailed)
        group_counts = self._count_group_records(group_series=groups[field])
        used_freqs, rejected_freqs = self._validate_n_grouprecords(group_counts=group_counts)
        data_detailed = self._filter_data(data_detailed=data_detailed, used_freqs=used_freqs)
        if len(data_detailed) < 2:
            raise ValueError(f"{field}: {len(data_detailed)} record(s) left after detecting the time "
                             f"resolution; too few regular records to screen.")
        data_detailed = self._harmonize_timeresolution(data_detailed=data_detailed,
                                                       timestamp_name=timestamp_name)
        targetfreq = self._grid_seconds(data_detailed=data_detailed, field=field)
        if len(used_freqs) > 1 or targetfreq != used_freqs[0]:
            info(f"Records with more than one time resolution or phase are placed on one "
                 f"grid of {targetfreq}S that holds each record at its own END timestamp.")
        data_detailed = self._sanitize_timestamp(targetfreq=targetfreq, data_detailed=data_detailed)

        return data_detailed

    @staticmethod
    def _grid_seconds(data_detailed, field: str) -> float:
        """Step in seconds of the grid that holds every record at its own END timestamp.

        The greatest common divisor of all resolutions and of the offsets between
        END timestamps, so records at a resolution that the finest one does not
        divide (10MIN, then 15MIN: 5MIN grid) or at another phase keep their slot.
        Regular single-resolution data get their own resolution.

        Raises:
            ValueError: If records off the phase of the others (e.g. END timestamps
                a few seconds late) shrink the grid below the common divisor of
                the resolutions, to a step below one second or to more than
                ``_MAX_SLOTS_PER_RECORD`` slots per record.
        """
        ends = data_detailed.index.as_unit('ns').asi8
        resolutions = np.unique(np.round(data_detailed['FREQ_AUTO_SEC'].to_numpy() * 1e9).astype(np.int64))
        step = np.gcd.reduce(np.concatenate([resolutions, np.unique(ends - ends[0])]))
        common = np.gcd.reduce(resolutions)
        if step < common:
            # Only a phase shift shrinks the grid below the resolutions' own divisor.
            # A few records off by a second would otherwise build millions of rows.
            slots_per_record = ((ends[-1] - ends[0]) // step + 1) / len(ends)
            if step < 10 ** 9 or slots_per_record > _MAX_SLOTS_PER_RECORD:
                phase = ends % common
                phases, counts = np.unique(phase, return_counts=True)
                off = data_detailed.index[phase != phases[np.argmax(counts)]]
                # With the count, e.g. '1min' where freqstr gives 'min'.
                common_str, step_str = (f"{o.n}{o.name}" for o in
                                        (to_offset(pd.Timedelta(int(ns), unit='ns')) for ns in (common, step)))
                examples = ', '.join(str(t) for t in off[:5]) + (', ...' if len(off) > 5 else '')
                raise ValueError(
                    f"{field}: {len(off)} of {len(ends)} records are off the {common_str} phase "
                    f"of the other records, e.g. END {examples}. Holding every record at its own "
                    f"END timestamp would need a grid of {step_str} with {slots_per_record:.0f} "
                    f"slots per record. Clean the timestamps first (e.g. round them to "
                    f"{common_str}) or resample the data.")
        return step / 1e9

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

        The grid (see `_grid_seconds`) is created afterwards in
        `_sanitize_timestamp`. A record of a coarser resolution stays a single row
        at its END timestamp; the grid slots before it that its averaging interval
        covers stay empty (NaN in all columns, FREQ_AUTO_SEC included). Copying the
        value onto those slots would make a record removable only in part and
        would give difference-based outlier tests runs of zero increments.
        `resample` weights each record by the seconds it covers instead
        (`_covered_seconds`: its FREQ_AUTO_SEC, less where it overlaps the
        previous record).
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
    def _validate_n_grouprecords(group_counts) -> tuple[list, list]:
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
        return used_freqs, rejected_freqs

    def _filter_data(self, data_detailed, used_freqs):
        keep = data_detailed['FREQ_AUTO_SEC'].isin(used_freqs)
        n_dropped = int((~keep).sum())
        if n_dropped:
            # Off-grid records and very small resolution groups are removed here;
            # without this message the loss would be invisible.
            warn(f"{n_dropped} records were removed because they do not belong to "
                 f"a used time resolution {used_freqs} (seconds).")
        return data_detailed.loc[keep]
