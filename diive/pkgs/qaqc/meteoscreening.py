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
from diive.core.times.resampling import resample_series_to_30MIN
from diive.core.times.times import sanitize_timestamp_index
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
    ):
        self.series = series
        self.measurement = measurement
        self.units = units
        self.site = site
        self.site_lat = site_lat
        self.site_lon = site_lon

        # Processing pipes
        path = Path(__file__).parent.resolve()  # Search in this file's folder
        self.pipes = filereader.ConfigFileReader(configfilepath=path / 'pipes_meteo.yaml').read()
        self.pipe_config = self._pipe_assign()

        # Collect all flags
        self.flags_df = pd.DataFrame(index=self.series.index)

        self._call_pipe_steps()

    def get(self) -> DataFrame:
        """Return all flags in DataFrame"""
        return self.flags_df

    def _call_pipe_steps(self):
        pipe_steps = self.pipe_config['pipe']

        # Missing values flag, always generated (high-res)
        _flag_missing = missing_values(series=self.series)
        self.flags_df[_flag_missing.name] = _flag_missing

        for step in pipe_steps:

            if step == 'remove_highres_outliers_thymeboost':
                _flag_outlier_thyme = thymeboost(series=self.series,
                                                 flag_missing=self.flags_df[_flag_missing.name])
                self.flags_df[_flag_outlier_thyme.name] = _flag_outlier_thyme


            elif step == 'remove_highres_outliers_absolute_limits':
                _flag_outlier_abslim = absolute_limits(series=self.series,
                                                       min=self.pipe_config['absolute_limits'][0],
                                                       max=self.pipe_config['absolute_limits'][1])
                self.flags_df[_flag_outlier_abslim.name] = _flag_outlier_abslim

            else:
                raise Exception(f"No function defined for {step}.")

            # elif step == 'remove_radiation_offset':
            #     series_qc = self._remove_radiation_offset(series=series_qc)

            # elif step == 'remove_relativehumidity_offset':
            #     series_qc = self._remove_relativehumidity_offset(series=series_qc)

            # elif step == 'setto_max_threshold':
            #     series_qc, _flag = self._setto_max_threshold(series=series_qc)

            # elif step == 'setto_min_threshold':
            #     series_qc, _flag = self._setto_min_threshold(series=series_qc)

        # Overall quality flag
        self.flags_df.loc[:, 'QCF'] = self.flags_df.sum(axis=1)

    def _remove_relativehumidity_offset(self, series: Series) -> Series:
        return remove_relativehumidity_offset(series=series,
                                              show=True, saveplot=self.saveplot)

    def _setto_max_threshold(self, series: Series) -> tuple[Series, Series]:
        series_qc, flag = setto_threshold(series=series, threshold=self.pipe_config['range'][1],
                                          type='max', show=True, saveplot=self.saveplot)
        return series_qc, flag

    def _setto_min_threshold(self, series: Series) -> tuple[Series, Series]:
        series_qc, flag = setto_threshold(series=series, threshold=self.pipe_config['range'][0],
                                          type='min', show=True, saveplot=self.saveplot)
        return series_qc, flag

    def _remove_radiation_offset(self, series: Series) -> Series:
        if (self.site_lat is None) | (self.site_lon is None):
            raise Exception("Radiation offset requires site latitude and longitude.")
        return remove_radiation_zero_offset(_series=series,
                                            lat=self.site_lat,
                                            lon=self.site_lon,
                                            show=True,
                                            saveplot=self.saveplot)

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

    def __init__(self, data_detailed: dict, assigned_measurements: dict, site: str):
        self.site = site
        self.data_detailed = data_detailed
        self.assigned_measurements = assigned_measurements
        self.vars_qc_resampled = {}
        self.qcflags_hires = {}
        self.run()

    def get(self):
        return self.vars_qc_resampled, self.qcflags_hires

    def run(self):
        for var in self.data_detailed.keys():
            m = self.assigned_measurements[var]
            mscr = MeteoScreeningFromDatabaseSingleVar(var_df=self.data_detailed[var].copy(),
                                                       site=self.site,
                                                       measurement=m,
                                                       field=var,
                                                       resampling_agg='mean',
                                                       resampling_freq='30T')
            mscr.run()

            self.vars_qc_resampled[var], \
            self.qcflags_hires[var] = \
                mscr.get()


class MeteoScreeningFromDatabaseSingleVar:
    """Accepts the output df from the `dbc` library that includes
     variable data (`field`) and tags
     """

    def __init__(self,
                 var_df: DataFrame,
                 measurement: str,
                 field: str,
                 site: str,
                 site_lat: float = None,
                 site_lon: float = None,
                 resampling_freq: str = '30T',
                 resampling_agg: str = 'mean'):
        self.var_df = var_df
        self.measurement = measurement
        self.field = field
        self.site = site
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.resampling_freq = resampling_freq
        self.resampling_agg = resampling_agg

        unique_units = list(set(self.var_df['units']))
        if len(unique_units) > 1:
            raise Exception(f"More than one type of units in column 'units', "
                            f"but only one allowed. All data records must be "
                            f"in same units.")

        self.grps = self.var_df.groupby(self.var_df['freq'])  # Groups by freq

        # Returned variables
        self.var_qc_resampled_df = None
        self.flags_hires_df = None

    def get(self) -> tuple[DataFrame, DataFrame]:
        return self.var_qc_resampled_df, self.flags_hires_df

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
        self.var_qc_resampled_df.sort_index(inplace=True)
        self.flags_hires_df.sort_index(inplace=True)

    def _run_by_freq_group(self):
        """Screen and resample by frequency

        When downloading data from the database, it is possible that the same
        variable was recorded at a different time resolution in the past. Each
        time resolution has to be handled separately. After the meteoscreening,
        all data are in 30MIN time resolution.
        """
        grp_counter = 0
        for grp_freq, grp_var_df in self.grps:
            series = grp_var_df[self.field].copy()  # Group series
            tags_dict = self.extract_tags(var_df=grp_var_df, drop_field=self.field)
            grp_counter += 1
            print(f"Frequency group: {grp_freq}")

            # Sanitize timestamp index
            series = sanitize_timestamp_index(data=series, freq=grp_freq)

            # Quality checks directly on high-res data
            grp_flags_hires_df = ScreenVar(series=series,
                                           measurement=self.measurement,
                                           units=tags_dict['units'],
                                           site=self.site,
                                           site_lat=self.site_lat,
                                           site_lon=self.site_lon).get()

            # Set flagged data to missing
            series_qc = series.copy()
            series_qc.loc[grp_flags_hires_df['QCF'] > 0] = np.nan

            # Resample to 30MIN
            series_qc_resampled = resample_series_to_30MIN(series=series_qc,
                                                           to_freqstr=self.resampling_freq,
                                                           agg=self.resampling_agg,
                                                           mincounts_perc=.9)

            # Update tags after resampling
            tags_dict['freq'] = '30T'
            # tags_dict['freqfrom'] = 'resampling'
            tags_dict['data_version'] = 'meteoscreening'

            # Create df that includes the resampled series and its tags
            grp_var_qc_resampled_df = pd.DataFrame(series_qc_resampled)

            # Insert tags as columns
            for key, value in tags_dict.items():
                grp_var_qc_resampled_df[key] = value

            if grp_counter == 1:
                self.var_qc_resampled_df = grp_var_qc_resampled_df.copy()
                self.flags_hires_df = grp_flags_hires_df.copy()
            else:
                self.var_qc_resampled_df.combine_first(other=grp_var_qc_resampled_df)
                self.flags_hires_df.combine_first(other=grp_flags_hires_df)
                # .append(grp_vardata_df_resampled)?


def test():
    # Testing code

    # Testing MeteoScreeningFromDatabase

    # Example file from dbc.download output
    testfile = r'L:\Dropbox\luhk_work\20 - CODING\26 - NOTEBOOKS\GL-NOTEBOOKS\General Notebooks\MeteoScreening\TA_NABEL_T1_35_1.csv'
    testdata = pd.read_csv(testfile, nrows=10000)
    testdata.set_index('TIMESTAMP_END', inplace=True)
    testdata.index = pd.to_datetime(testdata.index)
    testdata['TA_NABEL_T1_35_1'].plot()
    import matplotlib.pyplot as plt
    plt.show()

    # from dbc_influxdb import dbcInflux
    #
    # DIRCONF = r'L:\Dropbox\luhk_work\20 - CODING\22 - POET\configs'  # Folder with configurations
    # # Site name
    # SITE = 'ch-dav'
    # # Measurement name, used to group similar variable together, e.g., 'TA' contains all air temperature variables
    # MEASUREMENTS = ['TA']
    # # Variable name; InfluxDB stores variable names as '_field'
    # FIELDS = ['TA_NABEL_T1_35_1']
    # # Download data starting with this date
    # START = '2021-06-01 00:01:00'
    # # Download data before this date (the stop date itself is not included)
    # STOP = '2021-06-02 00:01:00'
    # # During MeteoScreening the screened high-res data will be resampled to this frequency;
    # # '30T' = 30-minute time resolution
    # resampling_freq = '30T'
    # # The resampling of the high-res data will be done using this aggregation methos; e.g., 'mean'
    # resampling_agg = 'mean'
    #
    # BUCKET_RAW = f'{SITE}_raw'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_raw' contains all raw data for CH-LAE
    # BUCKET_PROCESSING = f'{SITE}_processing'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_processing' contains all processed data for CH-LAE
    # print(f"Bucket containing raw data (source bucket): {BUCKET_RAW}")
    # print(f"Bucket containing processed data (destination bucket): {BUCKET_PROCESSING}")
    #
    # dbc = dbcInflux(dirconf=DIRCONF)
    #
    # data_simple, \
    # data_detailed, \
    # assigned_measurements = \
    #     dbc.download(bucket=BUCKET_RAW, measurements=MEASUREMENTS, fields=FIELDS,
    #                  start=START, stop=STOP, timezone_offset_to_utc_hours=1, data_version='raw')

    mscr = \
        MeteoScreeningFromDatabaseSingleVar(var_df=testdata.copy(),
                                            site='CH-DAV',
                                            measurement='TA',
                                            field='TA_NABEL_T1_35_1',
                                            resampling_agg='mean',
                                            resampling_freq='30T')
    mscr.run()
    vars_qc_resampled, qcflags_hires = mscr.get()

    # var_qc_resampled_df.to_csv(r'L:\Dropbox\luhk_work\20 - CODING\26 - NOTEBOOKS\meteoscreening\test_qc.csv')

    # # Testing MeteoScreeningFromFiles:
    # # todo compare radiation peaks for time shift
    # # todo check outliers before AND after first qc check
    # indir = r'F:\CH-AWS\snowheight\1-in'
    # mergeddir = r'F:\CH-AWS\snowheight\2-merged'
    # outdir = r'F:\CH-AWS\snowheight\3-out'
    #
    # # Search & merge high-res data files
    # searchdir = indir
    # pattern = '*.csv'
    # filetype = 'diive_CSV_30MIN'
    # filepaths = filereader.search_files(searchdir=searchdir, pattern=pattern)
    # mdfr = filereader.MultiDataFileReader(filepaths=filepaths, filetype=filetype)
    # data_df = mdfr.data_df
    # data_df.fillna(-9999, inplace=True)
    # data_df.to_csv(Path(mergeddir) / "merged.diive.csv")
    # metadata_df = mdfr.metadata_df
    # metadata_df.to_csv(Path(mergeddir) / "merged.diive.metadata.csv")
    #
    # # Search merged high-res file
    # searchdir = mergeddir
    # pattern = 'merged.diive.csv'
    # filetype = 'diive_CSV_1MIN'
    # filepaths = filereader.search_files(searchdir=searchdir, pattern=pattern)
    # mdfr = filereader.MultiDataFileReader(filepaths=filepaths, filetype=filetype)
    # data_df = mdfr.data_df
    # # metadata_df = mdfr.metadata_df  # todo automatic reading of metadata for diive formats
    #
    # # cols = {
    # #     'D_SNOW_1_1_1': {'measurement': 'D_SNOW', 'units': 'm'},
    # #     'TA_M1_1.8_1': {'measurement': 'TA', 'units': 'degC'}
    # #     # 'SHFM3_05_Avg': {'measurement': 'G', 'units': 'W m-2'}
    # #     # 'TA_T2_2x1_1_Avg': {'measurement': 'TA', 'units': 'degC'},
    # #     # 'LW_IN_T2_2x1_1_Avg': {'measurement': 'LW', 'units': 'W m-2'},
    # #     # 'SW_IN_T2_2x1_1_Avg': {'measurement': 'SW', 'units': 'W m-2'},
    # #     # 'RH_T2_2x1_1_Avg': {'measurement': 'RH', 'units': '%'},
    # #     # 'PPFD_IN_T2_2x1_1_Avg': {'measurement': 'PPFD', 'units': 'umol m-2 s-1'},
    # # }
    #
    # cols = {
    #     'D_SNOW_GF0_0_1': {'measurement': 'D_SNOW', 'units': 'm'},
    #     'TA_M3_2.5_1': {'measurement': 'TA', 'units': 'degC'}
    # }
    #
    # kwargs = dict(df=data_df.copy(),
    #               site='ch-das',
    #               site_lat=47.478333,  # CH-LAE
    #               site_lon=8.364389,  # CH-LAE
    #               # site_lat=46.815333,  # CH-DAS
    #               # site_lon=9.855972,  # CH-DAS
    #               outdir=outdir)
    # mscr = MeteoScreeningFromFiles(cols=cols, **kwargs)
    #
    # mscr.run()
    # qc_df = mscr.qc_df


if __name__ == '__main__':
    test()
