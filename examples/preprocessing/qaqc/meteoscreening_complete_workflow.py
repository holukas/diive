"""
===================================================
Complete Meteorological Screening Workflow (QC/QA)
===================================================

This example demonstrates the complete workflow for quality control and outlier detection
of high-resolution meteorological data downloaded from an InfluxDB database.

The workflow includes:
1. Downloading raw data from the database
2. Quality screening with multiple outlier detection methods
3. Data corrections for known measurement issues
4. Resampling to 30-minute time resolution
5. Uploading processed data back to the database

The workflow supports iterative inspection and parameter adjustment at each stage.
The StepwiseMeteoScreeningDb class is optimized for Jupyter notebooks but works
equally well in Python scripts.

**About this example:**

In this notebook, the raw data are downloaded from the database, quality-screened,
resampled and then uploaded to the database using the StepwiseMeteoScreeningDb class
in diive.

Here is an overview of what is done:
- (1) **USER SETTINGS**: Specify general settings for the site and variable
- (2) **AUTO-SETTINGS**: Automatic configuration of derived settings
- (3) **DOWNLOAD DATA FROM DATABASE**: Original raw data are downloaded from the database
  using dbc-influxdb
- (4) **METEOSCREENING**: The downloaded data is quality-screened using diive. The
  screening is done on high-resolution data, then resampled to 30MIN time resolution
- (5) **UPLOAD DATA TO DATABASE**: The screened and resampled data are uploaded back
  to the database
"""

import importlib.metadata
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from dbc_influxdb import dbcInflux

from diive.core.plotting.timeseries import TimeSeries
from diive.core.times.times import DetectFrequency
from diive.pkgs.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Example started: {dt_string}")
version_dbc = importlib.metadata.version("dbc_influxdb")
print(f"dbc-influxdb version: v{version_dbc}")
version_diive = importlib.metadata.version("diive")
print(f"diive version: v{version_diive}")

# %%
# User Settings (please adjust)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Specify the site, variables, and time period you want to screen. Adjust these
# settings to match your data.
#
# **Site Information:**
# Provide geographic coordinates and site identifier for the location being screened.

TESTDIR = Path(r"F:\TMP")

SITE = 'ch-hon'
SITE_LAT = 47.41887  # CH-HON
SITE_LON = 8.491318  # CH-HON

# %%
# Variables to Screen
# ^^^^^^^^^^^^^^^^^^^
#
# Specify variables as shown in the database.
#
# - **FIELDS**: Variables are called FIELDS in the database. InfluxDB stores variable
#   names as '_field'. You can specify multiple fields, given as a list
#   e.g. `['TA_NABEL_T1_35_1', 'TA_T1_20_1']`
# - **MEASUREMENT**: Only **one** measurement allowed. Measurement name that is used
#   to group similar variables together. Examples: `TA` contains all air temperature
#   variables, `SW` are all short-wave radiation measurements, `SWC` all soil water
#   measurements.

FIELDS = ['TA_T1_4_2']
MEASUREMENT = 'TA'

# %%
# Time Range to Screen
# ^^^^^^^^^^^^^^^^^^^^
#
# - **START**: Screen data starting with this date (the start date itself **IS** included)
# - **STOP**: Screen data before this date (the stop date itself **IS NOT** included)

START = '2025-07-01 00:00:01'
STOP = '2025-08-01 00:00:01'

# %%
# Resampling Aggregation
# ^^^^^^^^^^^^^^^^^^^^^^
#
# The resampling of the high-res data to 30MIN time resolution will be done using
# this aggregation method; two options: `mean` or `sum`
#
# For **precipitation** make sure to use `sum` because we need the 30MIN sums.

RESAMPLING_AGG = 'mean'
# RESAMPLING_AGG = 'sum'  # Use for precipitation

# %%
# Auto-settings (do not adjust)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# These settings are automatically configured based on the user settings above.

DATA_VERSION = 'raw'
TIMEZONE_OFFSET_TO_UTC_HOURS = 1
RESAMPLING_FREQ = '30min'
DIRCONF = r'F:\Sync\luhk_work\20 - CODING\22 - POET\configs'

BUCKET_RAW = f'{SITE}_raw'
BUCKET_PROCESSED = f'{SITE}_processed'

print(f"Bucket containing raw data (source bucket): {BUCKET_RAW}")
print(f"Bucket containing processed data (destination bucket): {BUCKET_PROCESSED}")

# %%
# Connect to Database
# ^^^^^^^^^^^^^^^^^^^

dbc = dbcInflux(dirconf=DIRCONF)

# %%
# Download Data from Database with dbc-influxdb
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Download the raw meteorological data. The dbc library downloads three variables:
#
# - **data_simple**: Simplest format with only the time series
# - **data_detailed**: Most important - contains time series AND database tags.
#   The tags are required when uploading data back to the database
# - **assigned_measurements**: Mapping of variables to measurements

print("Downloading data from database...")
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

# %%
# Check Available Variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Since we are now connected to the database, we can check if the variable(s) we want
# to screen are really in the database.
#
# **Important note:** If the desired variable(s) are indeed listed here, it does not
# necessarily mean that they are also available during the selected **time period**.
# This can be the case if the variable(s) for that time period were not uploaded to
# the database.

print(f"Data available for: {data_detailed.keys()}\n")
vars_not_available = [v for v in FIELDS if v not in data_detailed.keys()]
print(f"No data available for the following variables: {vars_not_available}")
for rem in vars_not_available:
    print(rem)
    FIELDS.remove(rem)
    print(f"Removed variables {rem} from FIELDS because it is not available during this time period.")

# %%
# Downloaded Data Summary
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The dbc library downloads three different variables:
#
# - **data_simple**: This is the simplest data format. It contains only the
#   high-resolution variable(s) from the specified measurement. Each variable is in
#   a separate column. Since different variables can have different time resolutions,
#   the highest time resolution across the variables is used as the index. This means
#   that lower resolution variables will show gaps in the higher resolution timestamp.
#
# - **data_detailed**: This is the most important variable for the MeteoScreening from
#   the database, because it contains not only the high-resolution time series of the
#   variable(s), but also their tags. The tags are important when uploading data to
#   the database. This is a very special format, because data for each variable are
#   stored in a dictionary. A dictionary is a data structure that stores key-value
#   pairs. The key is the variable name (e.g., `TA_NABEL_T1_35_1`) and the value is
#   a complete dataframe that contains the time series of the respective variable and
#   all tags.
#
# - **assigned_measurements**: An auxiliary variable that is useful to check whether
#   the measurement of the variable(s) is correct. In case we are screening air
#   temperatures, the measurement must be `TA`.

# %%
# Plot downloaded high-res data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualize the raw data at full resolution before quality screening.

for varname, frame in data_detailed.items():
    TimeSeries(series=frame[varname]).plot()

# %%
# Start Meteoscreening with diive
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Initialize the StepwiseMeteoScreeningDb class which handles:
# - Different time resolutions (upsamples to highest resolution)
# - Database tags preservation
# - Multiple variables from one measurement
# - Geographic information (lat/lon/UTC offset)

mscr = StepwiseMeteoScreeningDb(
    site=SITE,
    data_detailed=data_detailed,
    fields=FIELDS,
    site_lat=SITE_LAT,
    site_lon=SITE_LON,
    utc_offset=TIMEZONE_OFFSET_TO_UTC_HOURS
)

# mscr.showplot_orig(interactive=True)
mscr.showplot_orig()

# %%
# Outlier Detection
# ^^^^^^^^^^^^^^^^^
#
# Generate quality flags on full-resolution data using various outlier detection methods.
# Each test generates a preview plot before results are committed.
#
# The quality flag is NOT added until mscr.addflag() is executed. This allows
# preview inspection and parameter adjustment before accepting results.
#
# **Important:** Each subsequent test operates on data already filtered by
# previous tests. Order matters!

print("\nStarting outlier detection...")
mscr.start_outlier_detection()

# %%
# Plot current outlier-cleaned data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# During outlier detection, you can jump back here to plot the current version
# of cleaned data after all tests applied so far.

for key, val in mscr.outlier_detection.items():
    val.showplot_cleaned(interactive=False)

# %%
# Manual Removal Test
# ^^^^^^^^^^^^^^^^^^^
#
# Exclude data from specific time ranges or timestamps. Useful for known equipment
# failures, maintenance windows, or other documented bad data periods. Supports both
# individual timestamps and date ranges.
#
# Parameters:
# - **remove_dates**: List of timestamps or [start, stop] ranges to exclude
# - **showplot**: If True, display visualization of removed data
# - **verbose**: If True, print removal statistics

REMOVE_DATES = [
    ['2025-07-14 07:15:00', '2025-07-19 00:00:15'],  # Remove time range
    # '2022-08-23 11:45:00',  # Remove individual data point
    # ['2022-08-15', '2022-08-16']  # Remove date range
]
mscr.flag_manualremoval_test(
    remove_dates=REMOVE_DATES,  # List of timestamps or [start, stop] ranges to exclude
    showplot=True,  # Display visualization
    verbose=True  # Print removal statistics
)
# %%
mscr.addflag()

# %%
# Hampel Filter (Daytime/Nighttime)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Detects spikes within a sliding window using median absolute deviation (MAD). Less
# sensitive to extreme outliers than standard deviation. Daytime/nighttime separation
# accounts for different signal characteristics between day and night.
#
# When `separate_day_night=True`, apply `n_sigma_daytime` and `n_sigma_nighttime` for
# day/night thresholds. The `n_sigma` parameter is ignored when separation is active.
#
# Parameters:
# - **window_length**: Sliding window size in records (e.g., 48*7 = 1 week of 48-record days)
# - **n_sigma_daytime**: MAD-based sigma threshold for daytime
# - **n_sigma_nighttime**: MAD-based sigma threshold for nighttime
# - **use_differencing**: If True, detect spikes in double-differenced data
# - **separate_day_night**: If True, apply separate thresholds for day vs night

WINDOW_LENGTH = 60 * 24 * 7  # 7 days (1min resolution)
N_SIGMA_DAYTIME = 16.5  # Daytime threshold
N_SIGMA_NIGHTTIME = 16.5  # Nighttime threshold
USE_DIFFERENCING = True
SEPARATE_DAY_NIGHT = True
REPEAT = False

mscr.flag_outliers_hampel_test(
    window_length=WINDOW_LENGTH,  # Sliding window size in records
    n_sigma=5.5,  # Default sigma (ignored when separate_day_night=True)
    n_sigma_daytime=N_SIGMA_DAYTIME,  # MAD-based sigma for daytime
    n_sigma_nighttime=N_SIGMA_NIGHTTIME,  # MAD-based sigma for nighttime
    k=1.4826,  # MAD scaling constant
    use_differencing=USE_DIFFERENCING,  # Apply to double-differenced data
    separate_day_night=SEPARATE_DAY_NIGHT,  # Separate thresholds for day/night
    showplot=True,  # Display visualization
    verbose=True,  # Print statistics
    repeat=REPEAT  # Iteratively repeat until convergence
)
# %%
mscr.addflag()

# %%
# Z-score (Daytime/Nighttime)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Flags readings far from daytime or nighttime averages using separate thresholds.
# Accounts for different signal strengths and ranges between day and night (e.g.,
# daytime flux is typically higher than nighttime).
#
# Parameters:
# - **thres_zscore**: Z-score threshold for flagging (typically 3-5)
# - **showplot**: If True, display outlier visualization
# - **verbose**: If True, print detection statistics
# - **repeat**: If True, iteratively repeat detection until convergence
#
# **Note on radiation:** Flags negative nighttime values, but these are corrected
# in the correction step. Do not remove if radiation correction is planned.
#
# **Note on relative humidity:** Flags RH > 100%, but these are corrected later.
# Do not remove if RH offset correction is planned.

THRESHOLD = 4.5
mscr.flag_outliers_zscore_test(
    thres_zscore=THRESHOLD,  # Z-score threshold for flagging
    separate_daytime_nighttime=True,  # Separate thresholds for day/night
    lat=SITE_LAT,  # Site latitude
    lon=SITE_LON,  # Site longitude
    utc_offset=TIMEZONE_OFFSET_TO_UTC_HOURS,  # UTC offset for day/night calculation
    showplot=True,  # Display visualization
    verbose=True,  # Print statistics
    repeat=True  # Iteratively repeat until convergence
)
# %%
mscr.addflag()

# %%
# Z-score (Rolling, All Data)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Flags readings far from a moving window average. Threshold adapts to local variability
# rather than global statistics. Useful for detecting outliers in data with slow drift
# (instrument calibration shifts, seasonal changes) without removing valid gradual changes.
#
# Parameters:
# - **thres_zscore**: Z-score threshold for flagging (typically 3-5)
# - **winsize**: Window size in records for rolling calculation
# - **plottitle**: Optional title for plots
# - **showplot**: If True, display outlier visualization
# - **verbose**: If True, print detection statistics
# - **repeat**: If True, iteratively repeat detection until convergence

THRESHOLD = 4.5
WINSIZE = 1440 * 7  # 7 days at 1-min resolution

mscr.flag_outliers_zscore_rolling_test(
    thres_zscore=THRESHOLD,  # Z-score threshold for flagging
    winsize=WINSIZE,  # Window size in records
    plottitle=None,  # Optional title for plots
    showplot=True,  # Display visualization
    verbose=True,  # Print statistics
    repeat=True  # Iteratively repeat until convergence
)
# %%
mscr.addflag()

# %%
# Local Standard Deviation (Daytime/Nighttime)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Flags readings that deviate from the local rolling standard deviation. Daytime/nighttime
# separation adapts to different noise characteristics. Window-based SD better captures
# local patterns than global statistics.
#
# Parameters:
# - **n_sd**: Standard deviation multiplier; list [daytime, nighttime] for separate thresholds
# - **winsize**: Window size in records; list [daytime, nighttime] for separate windows
# - **separate_daytime_nighttime**: If True, apply different thresholds for day/night
# - **constant_sd**: If False, calculate SD from each window; if True, use global SD
# - **showplot**: If True, display outlier visualization
# - **verbose**: If True, print detection statistics
# - **repeat**: If True, iteratively repeat detection until convergence

N_SD = [4.5, 1.1]
WINSIZE = [300, 200]

mscr.flag_outliers_localsd_test(
    n_sd=N_SD,  # Standard deviation multiplier [daytime, nighttime]
    winsize=WINSIZE,  # Window size in records [daytime, nighttime]
    separate_daytime_nighttime=True,  # If True, separate thresholds for day/night
    constant_sd=False,  # If False, calculate SD from window; if True, use global
    showplot=True,  # Display visualization
    verbose=True,  # Print statistics
    repeat=False  # Iteratively repeat until convergence
)
# %%
mscr.addflag()

# %%
# Increments Z-score (All Data)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Flags sudden jumps between consecutive readings (first differences). Targets spikes
# and discontinuities rather than absolute values. Sensitive to sensor glitches and
# data transmission errors that cause abrupt shifts.
#
# Parameters:
# - **thres_zscore**: Z-score threshold for increment anomalies (typically 20-40)
# - **showplot**: If True, display outlier visualization
# - **verbose**: If True, print detection statistics
# - **repeat**: If True, iteratively repeat detection until convergence

THRESHOLD = 40

mscr.flag_outliers_increments_zcore_test(
    thres_zscore=THRESHOLD,  # Z-score threshold for increment anomalies
    showplot=True,  # Display visualization
    verbose=True,  # Print statistics
    repeat=True  # Iteratively repeat until convergence
)
# %%
mscr.addflag()

# %%
# Z-score (All Data)
# ^^^^^^^^^^^^^^^^^^
#
# Flags readings far from the global mean. Simplest approach with no day/night
# separation or moving windows. Fast but less nuanced for variables with legitimate
# day/night differences in magnitude or variability.
#
# Parameters:
# - **thres_zscore**: Z-score threshold for flagging (typically 3-5)
# - **plottitle**: Optional title for plots
# - **showplot**: If True, display outlier visualization
# - **verbose**: If True, print detection statistics
# - **repeat**: If True, iteratively repeat detection until convergence

THRESHOLD = 3.5

mscr.flag_outliers_zscore_test(
    thres_zscore=THRESHOLD,  # Z-score threshold for flagging
    plottitle=None,  # Optional title for plots
    showplot=True,  # Display visualization
    verbose=True,  # Print statistics
    repeat=True  # Iteratively repeat until convergence
)
# %%
mscr.addflag()

# %%
# Local Outlier Factor
# ^^^^^^^^^^^^^^^^^^^^
#
# Density-based detection. Flags readings that are isolated relative to their neighbors
# in the data space. Captures outliers that don't deviate much in individual variables
# but form unusual combinations.
#
# Parameters:
# - **n_neighbors**: Number of neighbors for LOF; auto-calculated if None
# - **contamination**: Expected outlier fraction ('auto' or 0-1)
# - **separate_daytime_nighttime**: If True, separate LOF for day/night; if False, global LOF
# - **showplot**: If True, display outlier visualization
# - **verbose**: If True, print detection statistics
# - **repeat**: If True, iteratively repeat detection until convergence
# - **n_jobs**: Number of parallel jobs (-1 uses all cores)
#
# **Note:** Slow on high-resolution data (1-minute or finer). Suitable for 30-minute
# or coarser resampled data.

mscr.flag_outliers_lof_test(
    n_neighbors=30,  # Number of neighbors for LOF calculation
    contamination=0.01,  # Expected outlier fraction ('auto' or 0-1)
    separate_daytime_nighttime=False,  # If True, separate LOF for day/night
    showplot=True,  # Display visualization
    verbose=True,  # Print statistics
    repeat=False,  # Iteratively repeat until convergence
    n_jobs=-1  # Parallel jobs (-1 = use all cores)
)
# %%
mscr.addflag()

# %%
# Absolute Limits
# ^^^^^^^^^^^^^^^
#
# Hard threshold based on physical validity. Removes values outside specified ranges.
# Fast, transparent, and based on measurement specifications. Can apply separate
# limits for daytime and nighttime.
#
# Parameters:
# - **minval**: Minimum valid value (global mode)
# - **maxval**: Maximum valid value (global mode)
# - **separate_daytime_nighttime**: If True, separate limits for day/night; if False, global
# - **daytime_minmax**: [min, max] valid range for daytime (day/night mode)
# - **nighttime_minmax**: [min, max] valid range for nighttime (day/night mode)
# - **showplot**: If True, display flagged data visualization
# - **verbose**: If True, print flagging statistics

MIN = -18
MAX = 50

mscr.flag_outliers_abslim_test(
    minval=MIN,  # Minimum valid value
    maxval=MAX,  # Maximum valid value
    separate_daytime_nighttime=False,  # If True, use separate day/night thresholds
    daytime_minmax=None,  # [min, max] range for daytime (day/night mode only)
    nighttime_minmax=None,  # [min, max] range for nighttime (day/night mode only)
    showplot=True,  # Display visualization
    verbose=False  # Print statistics
)
# %%
mscr.addflag()

# %%
# Trim Low Test
# ^^^^^^^^^^^^^
#
# Systematically removes lowest and highest percentages of readings. Flags values
# below a threshold, then flags an equal number from the high end. Symmetric removal
# useful for addressing known systematic bias without relying on statistical tests.
#
# Parameters:
# - **trim_daytime**: If True, apply trimming to daytime data
# - **trim_nighttime**: If True, apply trimming to nighttime data
# - **lower_limit**: Threshold; values below are flagged, then equal count from high end
# - **showplot**: If True, display flagged data visualization
# - **verbose**: If True, print trimming statistics

TRIM_DAYTIME = False
TRIM_NIGHTTIME = True
LOWER_LIMIT = 10

mscr.flag_outliers_trim_low_test(
    trim_daytime=TRIM_DAYTIME,  # If True, apply trimming to daytime data
    trim_nighttime=TRIM_NIGHTTIME,  # If True, apply trimming to nighttime data
    lower_limit=LOWER_LIMIT,  # Threshold; values below are flagged
    showplot=True,  # Display visualization
    verbose=True  # Print trimming statistics
)
# %%
mscr.addflag()

# %%
# Missing Values Test
# ^^^^^^^^^^^^^^^^^^^
#
# Records original missing data (NaN) in quality flags. Distinguishes original gaps
# from data removed by quality tests. Part of overall quality tracking, not outlier
# detection per se.
#
# Parameters:
# - **verbose**: If True, print missing data statistics

mscr.flag_missingvals_test(
    verbose=True  # Print missing data statistics
)

# %%
# Quality Control Flag (QCF)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Calculate overall quality-control flag QCF for full-resolution data. This
# combines all individual test flags into one comprehensive quality indicator:
#
# - **QCF=0**: Good quality (all tests pass)
# - **QCF=1**: Marginal quality (1-3 soft warnings, no hard failures)
# - **QCF=2**: Poor quality (>3 soft warnings or >=2 hard failures)

print("\nFinalizing outlier detection and calculating QCF...")
mscr.finalize_outlier_detection()

# %%
# QCF Reports
# ^^^^^^^^^^^

mscr.report_outlier_detection_qcf_evolution()
mscr.report_outlier_detection_qcf_flags()
mscr.report_outlier_detection_qcf_series()

# %%
# QCF Plots
# ^^^^^^^^^

mscr.showplot_outlier_detection_qcf_heatmaps()
# mscr.showplot_outlier_detection_qcf_timeseries()

# %%
# Corrections
# ^^^^^^^^^^^
#
# Apply corrections for common issues. Corrections are done on high-resolution data
# before resampling.

print("\nApplying data corrections...")

# Show cleaned data after QCF
mscr.showplot_cleaned()

# %%
# Correction: Remove Radiation Zero Offset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Remove nighttime offset from radiation data and set nighttime to zero.
# Can be used for: SW_IN, SW_OUT, PPFD_IN, PPFD_OUT

mscr.correction_remove_radiation_zero_offset()

# %%
# Correction: Remove Relative Humidity Offset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Remove relative humidity offset (correct values > 100%).
# Can be used for: RH

mscr.correction_remove_relativehumidity_offset()

# %%
# Correction: Set to Max Threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Set values above threshold to threshold value.

mscr.correction_setto_max_threshold(threshold=30)

# %%
# Correction: Set to Min Threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Set values below threshold to threshold value.

mscr.correction_setto_min_threshold(threshold=-5)

# %%
# Correction: Set to Value
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Set records within time range(s) to a constant value. Useful for setting
# precipitation to zero during sensor testing or known bad data periods.

DATES = [
    ['2022-04-01', '2022-04-05'],
    ['2022-09-05', '2022-09-07']
]
# mscr.correction_setto_value(dates=DATES, value=3.7, verbose=1)

# %%
# Correction: Set Exact Values to Missing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Set records with exact values to missing (NaN). Useful for removing test values
# or known bad values from the dataset.

# First, inspect value counts
for ff in mscr.fields:
    _valcounts = mscr._series_hires_cleaned[ff].value_counts()
    _total_n_records = mscr._series_hires_cleaned[ff].count()
    print("--------")
    print(f"Top 20 unique values and occurrences for {ff}:")
    print(_valcounts.head(20))
    print(f"\nTotal number of records: {_total_n_records}")

# mscr.correction_set_exact_value_to_missing(values=[0])

# %%
# Show cleaned data after corrections
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

mscr.showplot_cleaned()

# %%
# Analysis: Potential Radiation Correlation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Compare time series to potential radiation to detect potential timestamp shifts.
# Can be used for: SW_IN, SW_OUT, PPFD_IN, PPFD_OUT
#
# This generates plots showing:
# - Daily correlations with potential radiation
# - Low correlation days
# - Days with lowest correlation
# - Days with highest correlation

_ = mscr.analysis_potential_radiation_correlation(utc_offset=1, mincorr=0.7, showplot=True)

# %%
# Resampling
# ^^^^^^^^^^
#
# Resample high-resolution quality-controlled data to 30MIN time resolution.

print(f"\nResampling to {RESAMPLING_FREQ} using {RESAMPLING_AGG} aggregation...")
mscr.resample(
    to_freqstr=RESAMPLING_FREQ,
    agg=RESAMPLING_AGG,
    mincounts_perc=0.25
)

# %%
# Plot original and resampled data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

mscr.showplot_resampled()

# %%
# Check time resolution of resampled data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Verify that the resampled data has the expected 30-minute time resolution.

for v in mscr.resampled_detailed.keys():
    print(f"{'-' * 20}\n{v}")
    _checkfreq = DetectFrequency(
        index=mscr.resampled_detailed[v].index,
        verbose=True
    ).get()
    if _checkfreq == RESAMPLING_FREQ:
        print(f"TEST PASSED - The resampled variable {v} has a time resolution of {_checkfreq}.")
    else:
        print(
            f"{'#' * 20}(!)TEST FAILED - The resampled variable {v} does not have the expected time resolution of {_checkfreq}.{'#' * 20}")

# %%
# Upload Data to Database with dbc-influxdb
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Upload screened and resampled data back to the database. The upload preserves
# all database tags and marks data with 'meteoscreening_diive' version.

print(f"\nData will be uploaded to bucket {BUCKET_PROCESSED}")

for v in mscr.resampled_detailed.keys():
    m = assigned_measurements[v]
    dbc.upload_singlevar(
        to_bucket=BUCKET_PROCESSED,
        to_measurement=m,
        var_df=mscr.resampled_detailed[v],
        timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
        delete_from_db_before_upload=True
    )

# %%
# Verify Upload: Download from Database
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Download the data back from the database to verify successful upload.

dbc_verify = dbcInflux(dirconf=DIRCONF)

MEASUREMENT = ['TA']
data_simple, data_detailed, assigned_measurements = dbc_verify.download(
    bucket=BUCKET_PROCESSED,
    measurements=MEASUREMENT,
    fields=FIELDS,
    start=START,
    stop=STOP,
    timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
    data_version='meteoscreening_diive'
)

print("Downloaded data from processed bucket:")
print(data_simple)

# %%
# Check time resolution of downloaded data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Verify that the downloaded data has the expected time resolution.

for v in data_detailed.keys():
    print(f"{'-' * 20}\n{v}")
    _checkfreq = DetectFrequency(
        index=data_detailed[v].index,
        verbose=True
    ).get()
    if _checkfreq == RESAMPLING_FREQ:
        print(f"TEST PASSED - The downloaded variable {v} has a time resolution of {_checkfreq}.")
    else:
        print(
            f"{'#' * 20}(!)TEST FAILED - The downloaded variable {v} does not have the expected time resolution of {_checkfreq}.{'#' * 20}")

# %%
# End of Notebook
# ^^^^^^^^^^^^^^^

dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Finished. {dt_string}")
print("\nMeteoScreening workflow completed successfully!")
