"""
METEOSCREENING
==============

This module is part of the 'diive' library.

"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandas.tseries.frequencies import to_offset

import diive.core.io.filereader as filereader
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.core.plotting.plotfuncs import quickplot
from diive.core.times.resampling import resample_series_to_30MIN
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import detect_freq_groups
from diive.pkgs.corrections.offsetcorrection import remove_radiation_zero_offset, remove_relativehumidity_offset
from diive.pkgs.corrections.setto_threshold import setto_threshold
from diive.pkgs.outlierdetection.absolutelimits import AbsoluteLimits
from diive.pkgs.outlierdetection.local3sd import LocalSD
from diive.pkgs.outlierdetection.missing import MissingValues
from diive.pkgs.outlierdetection.seasonaltrend import OutlierSTLIQR
from diive.pkgs.outlierdetection.thymeboost import ThymeBoostOutlier
from diive.pkgs.outlierdetection.zscore import zScoreIQR
from diive.pkgs.qaqc.qcf import MeteoQCF


class ScreenMeteoVar:
    """Quality screening of one single meteo data time series

    Input series is not altered. Instead, this class produces a DataFrame
    that contains all quality flags.

    Typical time series: radiation, air temperature, soil water content.

    """

    def __init__(
            self,
            series: Series,
            measurement: str,
            units: str,
            site: str,
            site_lat: float = None,
            site_lon: float = None,
            timezone_of_timestamp: str = None
    ):
        self.series = series
        self.measurement = measurement
        self.units = units
        self.site = site
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.timezone_of_timestamp = timezone_of_timestamp

        # Processing pipes
        path = Path(__file__).parent.resolve()  # Search in this file's folder
        self.pipes = filereader.ConfigFileReader(configfilepath=path / 'pipes_meteo.yaml',
                                                 validation='meteopipe').read()
        self._pipe_config = self._pipe_assign()

        # Collect all flags
        self._flags_df = None

        self._run()

    @property
    def pipe_config(self) -> dict:
        """Processing pipe configuration"""
        if not isinstance(self._pipe_config, dict):
            raise Exception('Pipe configuration not available')
        return self._pipe_config

    @property
    def flags(self) -> DataFrame:
        """Processing pipe configuration"""
        if not isinstance(self._flags_df, DataFrame):
            raise Exception('Flags are empty')
        return self._flags_df

    @property
    def seriesqcf(self) -> Series:
        """Series with rejected values set to missing"""
        if not isinstance(self._seriesqcf, Series):
            raise Exception('Flags are empty')
        return self._seriesqcf

    def _run(self):

        # Call the various checks for this variable and generate flags
        mqcf = self._call_pipe_steps()

        # Print some info about QCF
        # # mqcf.report_flags()
        # # mqcf.report_series()
        mqcf.report_qcf_evolution()

        # Plot
        mqcf.showplot_heatmaps()
        mqcf.showplot_timeseries()

        # # Find column that contains the QCF info
        # flagqcfcol = findqcfcol(df=self._flags_df, varname=str(self.series.name))

    def _call_pipe_steps(self):
        # Initiate new flag collection
        self._flags_df = pd.DataFrame(index=self.series.index)

        # Processing steps for this variable
        pipe_steps = self.pipe_config['pipe']

        # Missing values flag, always generated (high-res)
        _miss = MissingValues(series=self.series)
        _miss.calc()
        self._flags_df[_miss.flag.name] = _miss.flag

        for step in pipe_steps:

            if step == 'remove_highres_outliers_stl':
                _stl = OutlierSTLIQR(series=self.series, lat=self.site_lat, lon=self.site_lon)
                _stl.calc(showplot=True)

            elif step == 'remove_highres_outliers_thymeboost':
                # Generates flag, needs QC'd data
                _mqcf = MeteoQCF(flags_df=self._flags_df, series=self.series, missingflag=str(_miss.flag.name))
                _mqcf.calculate()
                _thymeboost = ThymeBoostOutlier(series=_mqcf.seriesqcf)
                _thymeboost.calc(showplot=True)
                self._flags_df[_thymeboost.flag.name] = _thymeboost.flag

            elif step == 'remove_highres_outliers_zscore':
                # Generates flag
                _zscoreiqr = zScoreIQR(series=self.series)
                _zscoreiqr.calc(factor=4.5, showplot=True, verbose=True)
                self._flags_df[_zscoreiqr.flag.name] = _zscoreiqr.flag

            elif step == 'remove_highres_outliers_localsd':
                # Generates flag, needs QC'd data
                _mqcf = MeteoQCF(flags_df=self._flags_df, series=self.series, missingflag=str(_miss.flag.name))
                _mqcf.calculate()
                _localsd = LocalSD(series=_mqcf.seriesqcf)
                _localsd.calc(n_sd=7, showplot=True, verbose=True)
                # _flag_outlier_local3sd = xxxlocalsd(series=_series, n_sd=7, showplot=True)
                self._flags_df[_localsd.flag.name] = _localsd.flag

            elif step == 'remove_highres_outliers_absolute_limits':
                # Generates flag
                _abslim = AbsoluteLimits(series=self.series)
                _abslim.calc(min=self.pipe_config['absolute_limits'][0],
                             max=self.pipe_config['absolute_limits'][1])
                self._flags_df[_abslim.flag.name] = _abslim.flag

            elif step == 'remove_radiation_zero_offset':
                # Generates corrected series
                if (self.site_lat is None) \
                        | (self.site_lon is None) \
                        | (self.timezone_of_timestamp is None):
                    raise Exception("Removing the radiation zero offset requires site latitude, "
                                    "site longitude and site timezone (e.g., 'UTC+01:00' for CET).")
                self.series = remove_radiation_zero_offset(
                    series=self.series,
                    lat=self.site_lat,
                    lon=self.site_lon,
                    timezone_of_timestamp=self.timezone_of_timestamp,
                    showplot=True
                )

            elif step == 'setto_max_threshold':
                # Generates corrected series
                self.series = setto_threshold(
                    series=self.series,
                    threshold=self.pipe_config['absolute_limits'][1],
                    type='max',
                    showplot=True)

            elif step == 'setto_min_threshold':
                # Generates corrected series
                self.series = setto_threshold(series=self.series,
                                              threshold=self.pipe_config['absolute_limits'][0],
                                              type='min',
                                              showplot=True)

            elif step == 'remove_relativehumidity_offset':
                # Generates corrected series
                self.series = remove_relativehumidity_offset(series=self.series,
                                                             showplot=True)

            else:
                raise Exception(f"No function defined for {step}.")

        # Calculate overall quality flag QCF and Add QCF results to other flags
        mqcf = self._calculate_qcf(missingflag=str(_miss.flag.name))
        self._seriesqcf = mqcf.seriesqcf.copy()
        return mqcf

    def _calculate_qcf(self, missingflag: str):
        """Calculate overall quality flag QCF and Add QCF results to other flags"""
        mqcf = MeteoQCF(flags_df=self._flags_df, series=self.series, missingflag=missingflag)
        mqcf.calculate()
        _flags_df = pd.concat([self._flags_df, mqcf.flags], axis=1)
        _flags_df = _flags_df.loc[:, ~_flags_df.columns.duplicated(keep='last')]
        self._flags_df = _flags_df
        return mqcf

    def _pipe_assign(self) -> dict:
        """Assign appropriate processing pipe to this variable"""
        if self.measurement not in self.pipes.keys():
            raise KeyError(f"{self.measurement} is not defined in meteo_pipes.yaml")

        pipe_config = var = None
        for var in self.pipes[self.measurement].keys():
            if var in self.series.name:
                if self.pipes[self.measurement][var]['units'] != self.units:
                    continue
                pipe_config = self.pipes[self.measurement][var]
                break

        print(f"Variable '{self.series.name}' has been assigned to processing pipe {pipe_config['pipe']}")

        return pipe_config


class MeteoScreeningFromDatabaseMultipleVars:

    def __init__(self,
                 data_detailed: dict,
                 assigned_measurements: dict,
                 site: str,
                 **kwargs):
        self.site = site
        self.data_detailed = data_detailed
        self.assigned_measurements = assigned_measurements
        self.kwargs = kwargs

        # Returned variables
        self.resampled_detailed = {}
        self.hires_qc = {}
        self.hires_flags = {}
        # self.run()

    def get(self) -> tuple[dict, dict, dict]:
        return self.resampled_detailed, self.hires_qc, self.hires_flags

    def run(self):
        for var in self.data_detailed.keys():
            m = self.assigned_measurements[var]
            mscr = MeteoScreeningFromDatabaseSingleVar(data_detailed=self.data_detailed[var].copy(),
                                                       site=self.site,
                                                       measurement=m,
                                                       field=var,
                                                       resampling_freq='30T',
                                                       **self.kwargs)
            mscr.run()

            self.resampled_detailed[var], \
                self.hires_qc[var], \
                self.hires_flags[var] = mscr.get()


class MeteoScreeningFromDatabaseSingleVar:
    """Accepts the output df from the `dbc` library that includes
     variable data (`field`) and tags
     """

    def __init__(
            self,
            data_detailed: DataFrame,
            measurement: str,
            field: str,
            site: str,
            site_lat: float = None,
            site_lon: float = None,
            resampling_freq: str = '30T',
            timezone_of_timestamp: str = None
    ):
        self.data_detailed = data_detailed.copy()
        self.measurement = measurement
        self.field = field
        self.site = site
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.resampling_freq = resampling_freq
        self.timezone_of_timestamp = timezone_of_timestamp

        self.pipe_config = None  # Variable settings

        unique_units = list(set(self.data_detailed['units']))
        if len(unique_units) > 1:
            raise Exception(f"More than one type of units in column 'units', "
                            f"but only one allowed. All data records must be "
                            f"in same units.")

        # Group data by time resolution
        groups_ser = detect_freq_groups(index=self.data_detailed.index)
        self.data_detailed[groups_ser.name] = groups_ser
        self.grps = self.data_detailed.groupby(self.data_detailed['FREQ_AUTO_SEC'])

        # Returned variables
        # Collected data across (potentially) different time resolutions
        self.coll_resampled_detailed = None
        self.coll_hires_qc = None
        self.coll_hires_flags = None

    def run(self):
        self._run_by_freq_group()
        self.coll_resampled_detailed.drop(['FREQ_AUTO_SEC'], axis=1, inplace=True)  # Currently not used as tag in db
        self.coll_resampled_detailed.sort_index(inplace=True)
        self.coll_hires_qc.sort_index(inplace=True)
        self.coll_hires_flags.sort_index(inplace=True)
        # self._plots()

    def _run_by_freq_group(self):
        """Screen and resample by frequency

        When downloading data from the database, it is possible that the same
        variable was recorded at a different time resolution in the past. Each
        time resolution has to be handled separately. After the meteoscreening,
        all data are in 30MIN time resolution.
        """
        # Loop through frequency groups
        # All group members have the same, confirmed frequency
        grp_counter = 0
        for grp_freq, grp_var_df in self.grps:
            hires_raw = grp_var_df[self.field].copy()  # Group series
            varname = hires_raw.name
            tags_dict = self.extract_tags(var_df=grp_var_df, drop_field=self.field)
            grp_counter += 1
            print(f"({varname}) Working on data with time resolution {grp_freq}S")

            # Since all group members have the same time resolution, the freq of
            # the Series can already be set.
            offset = to_offset(pd.Timedelta(f'{grp_freq}S'))
            hires_raw = hires_raw.asfreq(offset.freqstr)  # Set to group freq

            # Sanitize timestamp index
            hires_raw = TimestampSanitizer(data=hires_raw).get()
            # series = sanitize_timestamp_index(data=series, freq=grp_freq)

            # Quality checks directly on high-res data
            # hires_screened, \
            #     hires_flags, \
            #     self.pipe_config = \

            screening = ScreenMeteoVar(series=hires_raw,
                                       measurement=self.measurement,
                                       units=tags_dict['units'],
                                       site=self.site,
                                       site_lat=self.site_lat,
                                       site_lon=self.site_lon,
                                       timezone_of_timestamp=self.timezone_of_timestamp)
            flags_hires = screening.flags
            seriesqcf_hires = screening.seriesqcf
            self.pipe_config = screening.pipe_config

            # Resample to 30MIN
            seriesqcf_resampled = resample_series_to_30MIN(series=seriesqcf_hires,
                                                           to_freqstr=self.resampling_freq,
                                                           agg=self.pipe_config['resampling_aggregation'],
                                                           mincounts_perc=.25)

            # Rename QC'd series to original name (without _QCF at the end)
            # This is done to comply with the naming convention, which needs
            # the positional indices at the end of the variable name (not _QCF).
            seriesqcf_resampled.name = hires_raw.name

            # Update tags after resampling
            tags_dict['freq'] = '30T'
            # tags_dict['freqfrom'] = 'resampling'
            tags_dict['data_version'] = 'meteoscreening'

            # Create df that includes the resampled series and its tags
            resampled_detailed = pd.DataFrame(seriesqcf_resampled)
            resampled_detailed = resampled_detailed.asfreq(seriesqcf_resampled.index.freqstr)

            # Insert tags as columns
            for key, value in tags_dict.items():
                resampled_detailed[key] = value

            if grp_counter == 1:
                self.coll_resampled_detailed = resampled_detailed.copy()  # Collection
                self.coll_hires_qc = seriesqcf_hires.copy()
                self.coll_hires_flags = flags_hires.copy()
            else:
                # self.coll_resampled_detailed = pd.concat([self.coll_resampled_detailed, resampled_detailed], axis=0)
                self.coll_resampled_detailed = self.coll_resampled_detailed.combine_first(other=resampled_detailed)
                self.coll_resampled_detailed.sort_index(ascending=True, inplace=True)
                self.coll_resampled_detailed = self.coll_resampled_detailed.asfreq(resampled_detailed.index.freqstr)

                # self.coll_hires_qc = pd.concat([self.coll_hires_qc, hires_qc], axis=0)
                self.coll_hires_qc = self.coll_hires_qc.combine_first(other=seriesqcf_hires)
                self.coll_hires_qc.sort_index(ascending=True, inplace=True)

                # self.coll_hires_flags = pd.concat([self.coll_hires_flags, hires_flags], axis=0)
                self.coll_hires_flags = self.coll_hires_flags.combine_first(other=flags_hires)
                self.coll_hires_flags.sort_index(ascending=True, inplace=True)

    def get(self) -> tuple[DataFrame, Series, DataFrame]:
        return self.coll_resampled_detailed, self.coll_hires_qc, self.coll_hires_flags

    def extract_tags(self, var_df: DataFrame, drop_field: str) -> dict:
        """Convert tag columns in DataFrame to simplified dict

        Args:
            var_df:
            drop_field:

        Returns:
            dict of tags

        """
        _df = var_df.copy()
        tags_df = _df.drop(columns=[drop_field])
        tags_df.nunique()
        tags_dict = {}
        for col in tags_df.columns:
            list_of_vals = list(tags_df[col].unique())
            str_of_vals = ",".join([str(i) for i in list_of_vals])
            tags_dict[col] = str_of_vals
        return tags_dict

    def show_groups(self):
        print("Number of values per frequency group:")
        print(self.grps.count())

    def _plots(self):
        varname = self.field

        # Plot heatmap
        HeatmapDateTime(series=self.coll_resampled_detailed[varname],
                        title=f"{self.coll_resampled_detailed[varname].name} RESAMPLED DATA\n"
                              f"AFTER HIGH-RES METEOSCREENING  |  "
                              f"time resolution @{self.coll_resampled_detailed.index.freqstr}").show()

        # Plot comparison
        _hires_qc = self.coll_hires_qc.copy()
        _hires_qc.name = f"{_hires_qc.name} (HIGH-RES, after quality checks)"
        _resampled = self.coll_resampled_detailed[varname].copy()
        _resampled.name = f"{_resampled.name} (RESAMPLED) " \
                          f"{_resampled.index.freqstr} {self.pipe_config['resampling_aggregation']}"
        quickplot([_hires_qc, _resampled], subplots=False, showplot=True,
                  title=f"Comparison")


def example():
    # Testing code
    import pickle

    # from datetime import datetime
    # import pkg_resources
    # dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"This page was last modified on: {dt_string}")
    # version_diive = pkg_resources.get_distribution("diive").version
    # print(f"diive version: v{version_diive}")
    # version_dbc_influxdb = pkg_resources.get_distribution("dbc_influxdb").version
    # print(f"dbc-influxdb version: v{version_dbc_influxdb}")

    # Testing MeteoScreeningFromDatabase

    # =======================================
    # SCREENING DATA DOWNLOADED FROM DATABASE
    # =======================================

    # Settings
    DIRCONF = r'L:\Sync\luhk_work\20 - CODING\22 - POET\configs'  # Folder with configurations
    SITE = 'ch-dav'  # Site name
    MEASUREMENTS = ['SWC']
    # MEASUREMENTS = ['TA']
    FIELDS = ['SWC_FF1_0.05_3']  # Variable name; InfluxDB stores variable names as '_field'
    # FIELDS = ['TA_NABEL_T1_35_1']  # Variable name; InfluxDB stores variable names as '_field'
    DATA_VERSION = 'raw'
    START = '2022-12-01 00:01:00'  # Download data starting with this date
    STOP = '2023-01-01 00:01:00'  # Download data before this date (the stop date itself is not included)
    TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # Timezone, e.g. "1" is translated to timezone "UTC+01:00" (CET, winter time)
    RESAMPLING_FREQ = '30T'  # During MeteoScreening the screened high-res data will be resampled to this frequency; '30T' = 30-minute time resolution
    RESAMPLING_AGG = 'mean'  # The resampling of the high-res data will be done using this aggregation methos; e.g., 'mean'

    basedir = Path(r"L:\Sync\luhk_work\_temp")

    BUCKET_RAW = f'{SITE}_raw'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_raw' contains all raw data for CH-LAE
    BUCKET_PROCESSING = f'{SITE}_processing'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_processing' contains all processed data for CH-LAE
    print(f"Bucket containing raw data (source bucket): {BUCKET_RAW}")
    print(f"Bucket containing processed data (destination bucket): {BUCKET_PROCESSING}")

    # # Instantiate class
    # from dbc_influxdb import dbcInflux
    # dbc = dbcInflux(dirconf=DIRCONF)
    # data_simple, data_detailed, assigned_measurements = dbc.download(
    #     bucket=BUCKET_RAW,
    #     measurements=MEASUREMENTS,
    #     fields=FIELDS,
    #     start=START,
    #     stop=STOP,
    #     timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
    #     data_version=DATA_VERSION)
    # import matplotlib.pyplot as plt
    # data_simple.plot()
    # plt.show()

    # # Export data to pickle for fast testing
    # pickle_out = open(basedir / "meteodata_simple.pickle", "wb")
    # pickle.dump(data_simple, pickle_out)
    # pickle_out.close()
    # pickle_out = open(basedir / "meteodata_detailed.pickle", "wb")
    # pickle.dump(data_detailed, pickle_out)
    # pickle_out.close()
    # pickle_out = open(basedir / "meteodata_assigned_measurements.pickle", "wb")
    # pickle.dump(assigned_measurements, pickle_out)
    # pickle_out.close()

    # Import data from pickle for fast testing
    pickle_in = open(basedir / "meteodata_simple.pickle", "rb")
    data_simple = pickle.load(pickle_in)
    pickle_in = open(basedir / "meteodata_detailed.pickle", "rb")
    data_detailed = pickle.load(pickle_in)
    pickle_in = open(basedir / "meteodata_assigned_measurements.pickle", "rb")
    assigned_measurements = pickle.load(pickle_in)

    # print(data_simple)
    # print(data_detailed)
    # print(assigned_measurements)

    # DAV
    site_lat = 46.815333
    site_lon = 9.855972

    # # LAE
    # site_lat = 47.478333
    # site_lon = 8.364389
    # data_detailed['TA_NABEL_T1_35_1'] = data_detailed['TA_NABEL_T1_35_1'].iloc[0:10000]

    # data_detailed['TA_NABEL_T1_35_1']['TA_NABEL_T1_35_1'].iloc[1000] = 99

    mscr = MeteoScreeningFromDatabaseMultipleVars(site=SITE,
                                                  data_detailed=data_detailed,
                                                  assigned_measurements=assigned_measurements,
                                                  site_lat=site_lat,
                                                  site_lon=site_lon,
                                                  timezone_of_timestamp='UTC+01:00')
    mscr.run()
    resampled_detailed, hires_qc, hires_flags = mscr.get()

    # todo compare radiation peaks for time shift
    # todo check outliers before AND after first qc check


if __name__ == '__main__':
    example()
