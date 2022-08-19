"""
METEOSCREENING
==============

This module is part of the 'diive' library.

"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import diive.core.io.filereader as filereader
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.core.plotting.plotfuncs import quickplot
from diive.core.times.resampling import resample_series_to_30MIN
from diive.core.times.times import TimestampSanitizer
from diive.pkgs.corrections.offsetcorrection import remove_radiation_zero_offset, remove_relativehumidity_offset
from diive.pkgs.corrections.setto_threshold import setto_threshold
from diive.pkgs.outlierdetection.absolutelimits import absolute_limits
from diive.pkgs.outlierdetection.missing import missing_values
from diive.pkgs.outlierdetection.thymeboost import thymeboost


class ScreenVar:
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
        self.pipes = filereader.ConfigFileReader(configfilepath=path / 'pipes_meteo.yaml').read()
        self.pipe_config = self._pipe_assign()

        # Collect all flags
        self.flags_df = pd.DataFrame(index=self.series.index)

        self._plot_data(type='RAW', order='BEFORE')
        self._call_pipe_steps()
        self._plot_data(type='QUALITY-CHECKED RAW', order='AFTER')

    def _plot_data(self, type: str, order: str):
        HeatmapDateTime(series=self.series,
                        title=f"{self.series.name} {type} DATA\n"
                              f"{order} HIGH-RES METEOSCREENING  |  "
                              f"time resolution @{self.series.index.freqstr}").show()

    def get(self) -> tuple[Series, DataFrame]:
        """Return all flags in DataFrame"""
        return self.series, self.flags_df

    def _call_pipe_steps(self):
        pipe_steps = self.pipe_config['pipe']

        # Missing values flag, always generated (high-res)
        _flag_missing = missing_values(series=self.series)
        self.flags_df[_flag_missing.name] = _flag_missing

        for step in pipe_steps:

            if step == 'remove_highres_outliers_thymeboost':
                # Generates flag
                _flag_outlier_thyme = thymeboost(series=self.series,
                                                 flag_missing=self.flags_df[_flag_missing.name])
                self.flags_df[_flag_outlier_thyme.name] = _flag_outlier_thyme


            elif step == 'remove_highres_outliers_absolute_limits':
                # Generates flag
                _flag_outlier_abslim = absolute_limits(series=self.series,
                                                       min=self.pipe_config['absolute_limits'][0],
                                                       max=self.pipe_config['absolute_limits'][1])
                self.flags_df[_flag_outlier_abslim.name] = _flag_outlier_abslim

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
                self.series = setto_threshold(series=self.series,
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

            # elif step == 'remove_relativehumidity_offset':
            #     series_qc = self._remove_relativehumidity_offset(series=series_qc)

        # Overall quality flag
        self.flags_df.loc[:, 'QCF'] = self.flags_df.sum(axis=1)


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


# class MeteoScreeningFromFiles:
#     """Quality screening of selected variables in a dataframe
#
#     """
#
#     def __init__(
#             self,
#             df: DataFrame,
#             cols: dict,
#             site: str,
#             site_lat: float = None,
#             site_lon: float = None,
#             outdir: str or Path = None,
#     ):
#         self.df = df
#         self.cols = cols
#         self.site = site
#         self.site_lat = site_lat
#         self.site_lon = site_lon
#         self.outdir = Path(outdir)
#
#         self._qc_df_resampled_gf = None
#
#     @property
#     def qc_df(self) -> DataFrame:
#         """Get dataframe of quality checked data"""
#         if not isinstance(self._qc_df_resampled_gf, DataFrame):
#             raise Exception('data is empty')
#         return self._qc_df_resampled_gf
#
#     def run(self):
#         if self.outdir:
#             verify_dir(self.outdir)
#         subset_df = self._subset()
#         qc_df = self._screening_loop(subset_df=subset_df)
#         qc_df_resampled = self._resampling(qc_df=qc_df)
#         self._qc_df_resampled_gf = self._fill_missing(qc_df_resampled=qc_df_resampled)
#         if self.outdir:
#             self._export_to_file()
#
#     def _export_to_file(self):
#         self._qc_df_resampled_gf.to_csv(self.outdir / 'out.csv')
#         quickplot_df(self._qc_df_resampled_gf.replace(-9999, np.nan),
#                      title="** COLUMNS AFTER QUALITY CONTROL **",
#                      saveplot=self.outdir)
#
#     def _fill_missing(self, qc_df_resampled: DataFrame) -> DataFrame:
#         """Fill missing values with -9999"""
#         qc_df_resampled.fillna(-9999, inplace=True)
#         return qc_df_resampled
#
#     def _resampling(self, qc_df: DataFrame) -> DataFrame:
#         """Resample data to output freq"""
#         qc_df_resampled, _ = frames.resample_df(df=qc_df, freq_str='30T', agg='mean',
#                                                 mincounts_perc=.9, to_freq='T')
#         return qc_df_resampled
#
#     def _screening_loop(self, subset_df: DataFrame) -> DataFrame:
#         """Loop variables and perform quality checks"""
#         qc_df = pd.DataFrame()
#         for col in self.cols.keys():
#             series = subset_df[col].copy()
#             measurement = self.cols[col]['measurement']
#             units = self.cols[col]['units']
#             col_qc = ScreenVar(series=series, measurement=measurement, units=units,
#                                site=self.site, site_lat=self.site_lat, site_lon=self.site_lon,
#                                saveplot=self.outdir).get()
#             qc_df[col_qc.name] = col_qc
#         return qc_df
#
#     def _subset(self) -> DataFrame:
#         subset_cols = []
#         for col in self.cols.keys():
#             subset_cols.append(col)
#         subset_df = self.df[subset_cols].copy()
#
#         if self.outdir:
#             quickplot_df(subset_df.replace(-9999, np.nan),
#                          title="COLUMNS *BEFORE* QUALITY CONTROL",
#                          saveplot=self.outdir)
#
#         return subset_df


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
                                                       resampling_agg='mean',
                                                       resampling_freq='30T',
                                                       **self.kwargs)
            mscr.run()

            self.resampled_detailed[var], \
            self.hires_qc[var], \
            self.hires_flags[var] = mscr.get()


#                         self.coll_resampled_detailed, \
#             self.coll_hires_qc, \
#             self.coll_hires_flags = mscr.get()


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
            resampling_agg: str = 'mean',
            timezone_of_timestamp: str = None
    ):
        self.data_detailed = data_detailed
        self.measurement = measurement
        self.field = field
        self.site = site
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.resampling_freq = resampling_freq
        self.resampling_agg = resampling_agg
        self.timezone_of_timestamp = timezone_of_timestamp

        unique_units = list(set(self.data_detailed['units']))
        if len(unique_units) > 1:
            raise Exception(f"More than one type of units in column 'units', "
                            f"but only one allowed. All data records must be "
                            f"in same units.")

        self.grps = self.data_detailed.groupby(self.data_detailed['freq'])  # Groups by freq

        # Returned variables
        # Collected data across (potentially) different time resolutions
        self.coll_resampled_detailed = None
        self.coll_hires_qc = None
        self.coll_hires_flags = None

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

    def run(self):
        self._run_by_freq_group()
        self.coll_resampled_detailed.sort_index(inplace=True)
        self.coll_hires_qc.sort_index(inplace=True)
        self.coll_hires_flags.sort_index(inplace=True)
        self._plots()

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
        _resampled.name = f"{_resampled.name} (RESAMPLED)"
        quickplot([_hires_qc, _resampled], subplots=False, showplot=True,
                  title=f"Comparison")

    def _run_by_freq_group(self):
        """Screen and resample by frequency

        When downloading data from the database, it is possible that the same
        variable was recorded at a different time resolution in the past. Each
        time resolution has to be handled separately. After the meteoscreening,
        all data are in 30MIN time resolution.
        """
        grp_counter = 0
        for grp_freq, grp_var_df in self.grps:
            hires_raw = grp_var_df[self.field].copy()  # Group series
            varname = hires_raw.name
            tags_dict = self.extract_tags(var_df=grp_var_df, drop_field=self.field)
            grp_counter += 1
            print(f"({varname}) Frequency group: {grp_freq}")

            # Sanitize timestamp index
            hires_raw = TimestampSanitizer(data=hires_raw).get()
            # series = sanitize_timestamp_index(data=series, freq=grp_freq)

            # Quality checks directly on high-res data
            hires_screened, \
            hires_flags = ScreenVar(series=hires_raw,
                                    measurement=self.measurement,
                                    units=tags_dict['units'],
                                    site=self.site,
                                    site_lat=self.site_lat,
                                    site_lon=self.site_lon,
                                    timezone_of_timestamp=self.timezone_of_timestamp).get()

            # Set flagged hi-res data to missing
            hires_qc = hires_screened.copy()
            hires_qc.loc[hires_flags['QCF'] > 0] = np.nan

            # Resample to 30MIN
            resampled = resample_series_to_30MIN(series=hires_qc,
                                                 to_freqstr=self.resampling_freq,
                                                 agg=self.resampling_agg,
                                                 mincounts_perc=.9)

            # Update tags after resampling
            tags_dict['freq'] = '30T'
            # tags_dict['freqfrom'] = 'resampling'
            tags_dict['data_version'] = 'meteoscreening'

            # Create df that includes the resampled series and its tags
            resampled_detailed = pd.DataFrame(resampled)

            # Insert tags as columns
            for key, value in tags_dict.items():
                resampled_detailed[key] = value

            if grp_counter == 1:
                self.coll_resampled_detailed = resampled_detailed.copy()  # Collection
                self.coll_hires_qc = hires_qc.copy()
                self.coll_hires_flags = hires_flags.copy()
            else:
                self.coll_resampled_detailed.combine_first(other=resampled_detailed)
                self.coll_hires_qc.combine_first(other=hires_qc)
                self.coll_hires_flags.combine_first(other=hires_flags)
                # .append(grp_vardata_df_resampled)?


def example():
    # Testing code
    import pickle

    # Testing MeteoScreeningFromDatabase

    # =======================================
    # SCREENING DATA DOWNLOADED FROM DATABASE
    # =======================================

    from dbc_influxdb import dbcInflux

    # Settings
    SITE = 'ch-dav'
    BUCKET = 'ch-dav_raw'
    MEASUREMENTS = ['RH']
    FIELDS = ['RH_NABEL_T1_35_1']
    START = '2021-04-01 00:01:00'
    STOP = '2021-04-30 00:01:00'
    TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # We need returned timestamps in CET (winter time), which is UTC + 1 hour
    DATA_VERSION = 'raw'
    DIRCONF = r'L:\Dropbox\luhk_work\20 - CODING\22 - POET\configs'

    # # Instantiate class
    # dbc = dbcInflux(dirconf=DIRCONF)
    #
    # # Data download
    # data_simple, data_detailed, assigned_measurements = \
    #     dbc.download(
    #         bucket=BUCKET,
    #         measurements=MEASUREMENTS,
    #         fields=FIELDS,
    #         start=START,
    #         stop=STOP,
    #         timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
    #         data_version=DATA_VERSION
    #     )
    #
    basedir = Path(r"M:\_temp")
    #
    # # Export data to pickle for fast testing
    # pickle_out = open(basedir / "data_simple.pickle", "wb")
    # pickle.dump(data_simple, pickle_out)
    # pickle_out.close()
    # pickle_out = open(basedir / "data_detailed.pickle", "wb")
    # pickle.dump(data_detailed, pickle_out)
    # pickle_out.close()
    # pickle_out = open(basedir / "assigned_measurements.pickle", "wb")
    # pickle.dump(assigned_measurements, pickle_out)
    # pickle_out.close()

    # Import data from pickle for fast testing
    pickle_in = open(basedir / "data_simple.pickle", "rb")
    data_simple = pickle.load(pickle_in)
    pickle_in = open(basedir / "data_detailed.pickle", "rb")
    data_detailed = pickle.load(pickle_in)
    pickle_in = open(basedir / "assigned_measurements.pickle", "rb")
    assigned_measurements = pickle.load(pickle_in)

    print(data_simple)
    print(data_detailed)
    print(assigned_measurements)

    mscr = MeteoScreeningFromDatabaseMultipleVars(site=SITE,
                                                  data_detailed=data_detailed,
                                                  assigned_measurements=assigned_measurements,
                                                  site_lat=46.815333,
                                                  site_lon=9.855972,
                                                  timezone_of_timestamp='UTC+01:00')
    mscr.run()
    resampled_detailed, hires_qc, hires_flags = mscr.get()

    # todo compare radiation peaks for time shift
    # todo check outliers before AND after first qc check



if __name__ == '__main__':
    example()
