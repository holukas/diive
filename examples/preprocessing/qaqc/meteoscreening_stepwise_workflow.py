"""
==================================
Stepwise Meteorological Screening
==================================

Complete workflow for quality control and outlier detection of high-resolution
meteorological data from database. Demonstrates the StepwiseMeteoScreeningDb class
for multi-stage quality screening with various outlier detection methods,
data corrections, and resampling.
"""

from pathlib import Path
from diive.core.times.times import DetectFrequency

# %%
# Configuration
# ^^^^^^^^^^^^^

TESTDIR = Path(r"F:\TMP")

SITE = 'ch-lae'
SITE_LAT = 46.815333  # CH-DAV
SITE_LON = 9.855972  # CH-DAV
FIELDS = ['TA_NABEL_T1_35_1']
MEASUREMENT = 'TA'
START = '1997-01-01 00:00:01'
STOP = '1998-01-01 00:00:01'
RESAMPLING_AGG = 'mean'
# RESAMPLING_AGG = 'sum'
DATA_VERSION = 'raw'
TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # Timezone, e.g. "1" is translated to timezone "UTC+01:00" (CET, winter time)
RESAMPLING_FREQ = '30min'  # During MeteoScreening the screened high-res data will be resampled to this frequency; '30min' = 30-minute time resolution
# DIRCONF = r'P:\Flux\RDS_calculations\_scripts\_configs\configs'  # Location of configuration files, needed e.g. for connection to database
DIRCONF = r'F:\Sync\luhk_work\20 - CODING\22 - POET\configs'
BUCKET_RAW = f'{SITE}_raw'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_raw' contains all raw data for CH-LAE
BUCKET_PROCESSED = f'{SITE}_processed'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_processed' contains all processed data for CH-LAE
# BUCKET_PROCESSED = f'{SITE}_processed'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_processed' contains all processed data for CH-LAE

# %%
# Download data from database
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

from diive.core.io.db.influx import InfluxIO
dbc = InfluxIO(dirconf=DIRCONF)
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
# import matplotlib.pyplot as plt
# data_simple.plot()
# plt.show()

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

# # Import data from pickle for fast testing
# from diive.core.io.files import load_parquet, load_pickle
# data_simple = load_parquet(filepath=TESTDIR / "meteodata_simple.parquet")
# _f = str(TESTDIR / "meteodata_detailed.pickle")
# data_detailed = load_pickle(_f)
# _f = str(TESTDIR / "meteodata_assigned_measurements.pickle")
# assigned_measurements = load_pickle(_f)

# # Restrict data for testing
# from diive.core.dfun.frames import df_between_two_dates
# for key in data_detailed.keys():
#     data_detailed[key] = df_between_two_dates(df=data_detailed[key], start_date='2022-06-01', end_date='2022-06-30')

# %%
# Initialize MeteoScreening
# ^^^^^^^^^^^^^^^^^^^^^^^^

from diive.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb

mscr = StepwiseMeteoScreeningDb(site=SITE,
                                data_detailed=data_detailed,
                                # measurement=MEASUREMENT,
                                fields=FIELDS,
                                site_lat=SITE_LAT,
                                site_lon=SITE_LON,
                                utc_offset=TIMEZONE_OFFSET_TO_UTC_HOURS)
# mscr.showplot_orig()
mscr.showplot_cleaned()

# %%
# Outlier Detection
# ^^^^^^^^^^^^^^^^

mscr.start_outlier_detection()  # If needed
# REMOVE_DATES = [
#     ['2010-12-24 07:15:00', '2011-01-01 00:00:15'],  # Remove time range
#     # '2022-08-23 11:45:00',  # Remove data point
#     # ['2022-08-15', '2022-08-16']  # Remove time range
# ]
# mscr.flag_manualremoval_test(remove_dates=REMOVE_DATES, showplot=True, verbose=True)
# mscr.flag_outliers_zscore_test(thres_zscore=4, separate_daytime_nighttime=True,
#                                lat=SITE_LAT, lon=SITE_LON, utc_offset=TIMEZONE_OFFSET_TO_UTC_HOURS,
#                                showplot=True, verbose=True, repeat=True)
mscr.flag_outliers_hampel_test(window_length=48 * 7, n_sigma_daytime=5.5, n_sigma_nighttime=5.5,
                               use_differencing=True, separate_daytime_nighttime=True,
                               showplot=True, verbose=True, repeat=True)
# mscr.addflag()
# mscr.flag_outliers_trim_low_test(trim_daytime=False, trim_nighttime=True, lower_limit=10,
#                                  showplot=True, verbose=True)
# mscr.addflag()
# mscr.flag_outliers_zscore_rolling_test(thres_zscore=3, winsize=48, showplot=True, verbose=True, repeat=True)
# mscr.flag_outliers_localsd_test(n_sd=2.5, winsize=24, showplot=True, verbose=True, repeat=False)
# mscr.flag_outliers_localsd_test(n_sd=2.5, winsize=24, showplot=True, verbose=True, repeat=False)
mscr.flag_outliers_localsd_test(separate_daytime_nighttime=True, n_sd=[2.5, 2.5], winsize=[24, 24],
                                showplot=True, verbose=True, repeat=False, constant_sd=False)
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

# %%
# Quality Control Flag (QCF)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# # QCF: After all QC flags generated, calculate overall flag QCF
# mscr.finalize_outlier_detection()
# mscr.report_outlier_detection_qcf_evolution()
# mscr.report_outlier_detection_qcf_flags()
# mscr.report_outlier_detection_qcf_series()
# mscr.showplot_outlier_detection_qcf_heatmaps()
# # mscr.showplot_outlier_detection_qcf_timeseries()

# %%
# Data Corrections
# ^^^^^^^^^^^^^^^^

# mscr.showplot_orig()
# mscr.showplot_cleaned()
# mscr.correction_remove_nighttime_zero_offset()  # Remove radiation zero offset
# mscr.correction_remove_relativehumidity_offset()  # Remove relative humidity offset
# mscr.correction_setto_max_threshold(threshold=30)  # Set to max threshold
# mscr.correction_setto_min_threshold(threshold=-5)  # Set to min threshold
# DATES = [['2022-04-01', '2022-04-05'], ['2022-09-05', '2022-09-07']]
# mscr.correction_setto_value(dates=DATES, value=3.7, verbose=1)  # Set to value
# mscr.correction_set_exact_value_to_missing(values=[0])

# %%
# Analysis
# ^^^^^^^^

# # Potential radiation correlation
# _ = mscr.analysis_potential_radiation_correlation(utc_offset=1, mincorr=0.7, showplot=True)

# %%
# Resampling
# ^^^^^^^^^^

mscr.resample(to_freqstr=RESAMPLING_FREQ, agg=RESAMPLING_AGG, mincounts_perc=.25)
mscr.showplot_resampled()
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

# %%
# Database Upload (Optional)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# # from diive.core.io.db.influx import InfluxIO
# # dbc = InfluxIO(dirconf=DIRCONF)
# # for v in mscr.resampled_detailed.keys():
# #     # mscr.resampled_detailed[v][v] = mscr.resampled_detailed[v][v].multiply(999)
# #     # mscr.resampled_detailed[v]['hpos'] = '999'
# #     m = assigned_measurements[v]
# #     dbc.upload_singlevar(to_bucket=BUCKET_PROCESSED,
# #                          to_measurement=m,
# #                          var_df=mscr.resampled_detailed[v],
# #                          timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
# #                          delete_from_db_before_upload=True)

# %%
# Download from Processed Bucket (Optional)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
