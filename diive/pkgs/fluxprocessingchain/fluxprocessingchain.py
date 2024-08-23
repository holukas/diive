#  TODO NEEDS FLOW CHECK
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.core.dfun.frames import detect_new_columns
from diive.core.funcs.funcs import filter_strings_by_elements
from diive.core.io.filereader import MultiDataFileReader, search_files
from diive.pkgs.createvar.daynightflag import daytime_nighttime_flag_from_swinpot
from diive.pkgs.createvar.potentialradiation import potrad
from diive.pkgs.flux.common import detect_fluxbasevar
from diive.pkgs.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsEddyPro
from diive.pkgs.fluxprocessingchain.level31_storagecorrection import FluxStorageCorrectionSinglePointEddyPro
from diive.pkgs.outlierdetection.stepwiseoutlierdetection import StepwiseOutlierDetection
from diive.pkgs.qaqc.qcf import FlagQCF


class FluxProcessingChain:

    def __init__(
            self,
            maindf: DataFrame,
            fluxcol: str,
            site_lat: float,
            site_lon: float,
            utc_offset: int,
            nighttime_threshold: float = 50
    ):

        self.maindf = maindf
        self.fluxcol = fluxcol
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.utc_offset = utc_offset
        self.nighttime_threshold = nighttime_threshold

        # Detect base variable that was used to produce this flux
        self.fluxbasevar = detect_fluxbasevar(fluxcol=fluxcol)

        # Collect all relevant variables for this flux in dataframe
        self._fpc_df = self.maindf[[fluxcol]].copy()

        # Add potential radiation and daytime and nighttime flags
        self._fpc_df, self.swinpot_col = self._add_swinpot_dt_nt_flag(df=self._fpc_df)

        # Get the name of the base flux, used to assemble meaningful names for output variables
        if self.fluxcol == 'FC':
            # CO2 flux changes to NEE during processing (in Level-3.1)
            self.outname = 'NEE'
        else:
            self.outname = self.fluxcol

        # Init new variables
        self._levelidstr = []  # ID strings used to tag the different flux levels
        self._metadata = None
        self._filteredseries = None
        self._filteredseries_level2_qcf = None
        self._filteredseries_level31_qcf = None
        self._filteredseries_level32_qcf = None
        self._maindf = None
        self._level2 = None
        self._level31 = None
        self._level32 = None
        self._level2_qcf = None
        self._level32_qcf = None

    @property
    def filteredseries(self) -> Series:
        """Return time series of flux, filtered by all available QCF checks."""
        if not isinstance(self._filteredseries, Series):
            raise Exception(f'No filtered time series for {self.fluxcol} available, '
                            f'please run .level2_quality_flag_expansion() first.')
        return self._filteredseries

    @property
    def filteredseries_level2_qcf(self) -> Series:
        """Return time series of quality-filtered flux after Level-2."""
        if not isinstance(self._filteredseries_level2_qcf, Series):
            raise Exception(f'No filtered time series for {self.fluxcol} available, '
                            f'please run .level2_quality_flag_expansion() first.')
        return self._filteredseries_level2_qcf

    @property
    def filteredseries_level31_qcf(self) -> Series:
        """Return time series of quality-filtered flux after Level-3.1."""
        if not isinstance(self._filteredseries_level31_qcf, Series):
            raise Exception(f'No filtered time series for {self.fluxcol} available, '
                            f'please run .level31_storage_correction() first.')
        return self._filteredseries_level31_qcf

    @property
    def filteredseries_level32_qcf(self) -> Series:
        """Return time series of quality-filtered flux after Level-3.2."""
        if not isinstance(self._filteredseries_level32_qcf, Series):
            raise Exception(f'No filtered time series for {self.fluxcol} available, '
                            f'please run .level32_stepwise_outlier_detection() first.')
        return self._filteredseries_level32_qcf

    @property
    def fpc_df(self) -> DataFrame:
        """Return fluxes and flags from each level."""
        if not isinstance(self._fpc_df, DataFrame):
            raise Exception('No flux data available.')
        return self._fpc_df

    @property
    def level2_qcf(self) -> FlagQCF:
        """Return instance of Level-2 QCF creation."""
        if not isinstance(self._level2_qcf, FlagQCF):
            raise Exception('No Level-2 data available, please run .level2_quality_flag_expansion() first.')
        return self._level2_qcf

    @property
    def level32_qcf(self) -> FlagQCF:
        """Return instance of Level-3.2 QCF creation."""
        if not isinstance(self._level32_qcf, FlagQCF):
            raise Exception('No Level-3.2 data available, please run .level32_stepwise_outlier_detection() first.')
        return self._level32_qcf

    @property
    def level2(self) -> FluxQualityFlagsEddyPro:
        """Return instance of Level-2 flag creation."""
        if not isinstance(self._level2, FluxQualityFlagsEddyPro):
            raise Exception('No Level-2 data available, please run .level2_quality_flag_expansion() first.')
        return self._level2

    @property
    def level31(self) -> FluxStorageCorrectionSinglePointEddyPro:
        """Return instance of Level-3.1 storage correction."""
        if not isinstance(self._level31, FluxStorageCorrectionSinglePointEddyPro):
            raise Exception('No Level-3.1 data available, please run .level31_storage_correction() first.')
        return self._level31

    @property
    def level32(self) -> StepwiseOutlierDetection:
        """Return instance of Level-3.2 outlier detection."""
        if not isinstance(self._level32, StepwiseOutlierDetection):
            raise Exception('No Level-3.2 data available, please run .level32_stepwise_outlier_detection() first.')
        return self._level32

    @property
    def levelidstr(self) -> list:
        """Return strings that were used to tag the different flux levels."""
        if not isinstance(self._levelidstr, list):
            raise Exception('Level IDs not available, please run .level2_quality_flag_expansion() first.')
        return self._levelidstr

    def _add_swinpot_dt_nt_flag(self, df: DataFrame) -> tuple[DataFrame, str]:
        # Add potential radiation, used for detecting daytime/nighttime
        swinpot = potrad(timestamp_index=df.index,
                         lat=self.site_lat, lon=self.site_lon, utc_offset=self.utc_offset)
        swinpot_col = str(swinpot.name)
        print(f"Calculated potential radiation from latitude and longitude ({swinpot_col}) ... ")

        # Add flags for daytime and nighttime data records
        daytime_flag, nighttime_flag = daytime_nighttime_flag_from_swinpot(
            swinpot=swinpot,
            nighttime_threshold=self.nighttime_threshold,
            daytime_col='DAYTIME',
            nighttime_col='NIGHTTIME')
        daytime_flag_col = str(daytime_flag.name)
        nighttime_flag_col = str(nighttime_flag.name)
        print(f"Calculated daytime flag {daytime_flag_col} and "
              f"nighttime flag {nighttime_flag_col} from {swinpot_col} ...")
        df[swinpot_col] = swinpot
        df[daytime_flag_col] = daytime_flag.copy()
        df[nighttime_flag_col] = nighttime_flag.copy()

        return df, swinpot_col

    def level2_quality_flag_expansion(
            self,
            signal_strength: dict or False = False,
            raw_data_screening_vm97: dict or False = False,
            ssitc: bool = True,
            gas_completeness: bool = False,
            spectral_correction_factor: bool = True,
            angle_of_attack: dict or False = False,
            steadiness_of_horizontal_wind: bool = False
    ):
        """Expand flux quality flag based on EddyPro output"""
        idstr = 'L2'
        self._levelidstr.append(idstr)
        self._level2 = FluxQualityFlagsEddyPro(fluxcol=self.fluxcol,
                                               dfin=self.maindf,
                                               idstr=idstr,
                                               fluxbasevar=self.fluxbasevar)
        self._level2.missing_vals_test()

        if ssitc:
            self._level2.ssitc_test()

        if gas_completeness:
            self._level2.gas_completeness_test()

        if spectral_correction_factor:
            self._level2.spectral_correction_factor_test()

        if signal_strength['test_signal_strength']:
            self._level2.signal_strength_test(signal_strength_col=signal_strength['signal_strength_col'],
                                              method=signal_strength['method'],
                                              threshold=signal_strength['threshold'])

        if raw_data_screening_vm97['raw_data_screening_vm97']:
            self._level2.raw_data_screening_vm97_tests(spikes=raw_data_screening_vm97['spikes'],
                                                       amplitude=raw_data_screening_vm97['amplitude'],
                                                       dropout=raw_data_screening_vm97['dropout'],
                                                       abslim=raw_data_screening_vm97['abslim'],
                                                       skewkurt_hf=raw_data_screening_vm97['skewkurt_hf'],
                                                       skewkurt_sf=raw_data_screening_vm97['skewkurt_sf'],
                                                       discont_hf=raw_data_screening_vm97['discont_hf'],
                                                       discont_sf=raw_data_screening_vm97['discont_sf'])
        if angle_of_attack['test_rawdata_angle_of_attack']:
            self._level2.angle_of_attack_test(
                application_dates=angle_of_attack['test_rawdata_angle_of_attack_application_dates']
            )

        if steadiness_of_horizontal_wind:
            self._level2.steadiness_of_horizontal_wind()

    def _finalize_level(self,
                        run_qcf_on_col: str,
                        idstr: str,
                        level_df: DataFrame,
                        nighttime_threshold: int = 50,
                        daytime_accept_qcf_below: int = 2,
                        nighttimetime_accept_qcf_below: int = 2) -> FlagQCF:

        # Detect new columns
        newcols = detect_new_columns(df=level_df, other=self.fpc_df)
        self._fpc_df = pd.concat([self.fpc_df, level_df[newcols]], axis=1)
        [print(f"++Added new column {col}.") for col in newcols]

        # Calculate overall quality flag QCF
        qcf = FlagQCF(series=self.fpc_df[run_qcf_on_col],
                      df=self.fpc_df,
                      idstr=idstr,
                      swinpot=self.fpc_df['SW_IN_POT'],
                      nighttime_threshold=nighttime_threshold)
        qcf.calculate(daytime_accept_qcf_below=daytime_accept_qcf_below,
                      nighttimetime_accept_qcf_below=nighttimetime_accept_qcf_below)
        self._fpc_df = qcf.get()

        self._filteredseries = qcf.filteredseries.copy()

        return qcf

    def finalize_level2(self,
                        nighttime_threshold: int = 50,
                        daytime_accept_qcf_below: int = 2,
                        nighttimetime_accept_qcf_below: int = 2):
        """Calculate overall quality flag QCF after Level-2"""
        self._level2_qcf = self._finalize_level(
            run_qcf_on_col=self.fluxcol,
            idstr='L2',
            level_df=self.level2.results,
            nighttime_threshold=nighttime_threshold,
            daytime_accept_qcf_below=daytime_accept_qcf_below,
            nighttimetime_accept_qcf_below=nighttimetime_accept_qcf_below
        )
        self._filteredseries_level2_qcf = self.filteredseries.copy()  # Store filtered series as variable

    def finalize_level32(self,
                         nighttime_threshold: int = 50,
                         daytime_accept_qcf_below: int = 2,
                         nighttimetime_accept_qcf_below: int = 2):
        """Calculate overall quality flag QCF after Level-3.2"""
        self._level32_qcf = self._finalize_level(
            run_qcf_on_col=self.level31.flux_corrected_col,
            idstr='L3.2',
            level_df=self.level32.flags,
            nighttime_threshold=nighttime_threshold,
            daytime_accept_qcf_below=daytime_accept_qcf_below,
            nighttimetime_accept_qcf_below=nighttimetime_accept_qcf_below
        )
        self._filteredseries_level32_qcf = self.filteredseries.copy()  # Store filtered series as variable

    def level31_storage_correction(self, gapfill_storage_term: bool = False):
        """Correct flux with storage term from single point measurement."""
        idstr = 'L3.1'
        self._levelidstr.append(idstr)
        self._level31 = FluxStorageCorrectionSinglePointEddyPro(df=self.maindf,
                                                                fluxcol=self.fluxcol,
                                                                basevar=self.fluxbasevar,
                                                                gapfill_storage_term=gapfill_storage_term,
                                                                idstr=idstr)
        self._level31.storage_correction()

    def finalize_level31(self):

        newcols = detect_new_columns(df=self.level31.results, other=self.fpc_df)
        self._fpc_df = pd.concat([self.fpc_df, self.level31.results[newcols]], axis=1)
        [print(f"++Added new column {col}.") for col in newcols]

        # Apply QCF also to Level-3.1 flux output
        self._apply_level2_qcf_to_level31_flux()

    def _apply_level2_qcf_to_level31_flux(self):
        """Apply the overall quality flag QCF that was calculated in Level-2 to Level-3.1 fluxes."""
        # Apply QCF
        strg_corrected_flux_qcf = self.level31.results[self.level31.flux_corrected_col].copy()
        reject = self.level2_qcf.filteredseries.isnull()
        strg_corrected_flux_qcf.loc[reject] = np.nan
        strg_corrected_flux_qcf.name = f"{strg_corrected_flux_qcf.name}_QCF"

        # Apply QCF, highest quality fluxes (QCF0)
        strg_corrected_flux_qcf0 = self.level31.results[self.level31.flux_corrected_col].copy()
        reject = self.level2_qcf.filteredseries_hq.isnull()
        strg_corrected_flux_qcf0.loc[reject] = np.nan
        strg_corrected_flux_qcf0.name = f"{strg_corrected_flux_qcf0.name}_QCF0"

        frame = {strg_corrected_flux_qcf.name: strg_corrected_flux_qcf,
                 strg_corrected_flux_qcf0.name: strg_corrected_flux_qcf0}
        newcols = pd.DataFrame.from_dict(frame)
        self._fpc_df = pd.concat([self._fpc_df, newcols], axis=1)
        [print(f"++Added new column {c} (Level-3.1 with applied quality flag from Level-2).") for c in frame.keys()]
        self._filteredseries = strg_corrected_flux_qcf.copy()

        self._filteredseries_level31_qcf = self._filteredseries.copy()  # Store filtered series as variable

    def level32_stepwise_outlier_detection(self):
        idstr = 'L3.2'
        self._levelidstr.append(idstr)
        self._level32 = StepwiseOutlierDetection(dfin=self.fpc_df,
                                                 col=str(self.filteredseries.name),
                                                 site_lat=self.site_lat,
                                                 site_lon=self.site_lon,
                                                 utc_offset=self.utc_offset,
                                                 idstr=idstr)

    def level32_flag_outliers_abslim_dtnt_test(self,
                                               daytime_minmax: list[float, float],
                                               nighttime_minmax: list[float, float],
                                               showplot: bool = False, verbose: bool = False):
        self._level32.flag_outliers_abslim_dtnt_test(daytime_minmax=daytime_minmax,
                                                     nighttime_minmax=nighttime_minmax,
                                                     showplot=showplot, verbose=verbose)

    def level32_flag_outliers_abslim_test(self, minval: float, maxval: float,
                                          showplot: bool = False, verbose: bool = False):
        self._level32.flag_outliers_abslim_test(minval=minval,
                                                maxval=maxval,
                                                showplot=showplot, verbose=verbose)

    def level32_flag_outliers_zscore_dtnt_test(self, thres_zscore: float = 4, showplot: bool = False,
                                               verbose: bool = False, repeat: bool = True):
        self._level32.flag_outliers_zscore_dtnt_test(thres_zscore=thres_zscore, showplot=showplot, verbose=verbose,
                                                     repeat=repeat)

    def level32_flag_outliers_localsd_test(self, n_sd: float = 7, winsize: int = None, showplot: bool = False,
                                           verbose: bool = False, repeat: bool = True):
        self._level32.flag_outliers_localsd_test(n_sd=n_sd, winsize=winsize, showplot=showplot, verbose=verbose,
                                                 repeat=repeat)

    def level32_flag_manualremoval_test(self, remove_dates: list, showplot: bool = False, verbose: bool = False):
        self._level32.flag_manualremoval_test(remove_dates=remove_dates, showplot=showplot, verbose=verbose)

    def level32_flag_outliers_increments_zcore_test(self, thres_zscore: int = 30, showplot: bool = False,
                                                    verbose: bool = False, repeat: bool = True):
        self._level32.flag_outliers_increments_zcore_test(thres_zscore=thres_zscore, showplot=showplot, verbose=verbose,
                                                          repeat=repeat)

    def level32_flag_outliers_trim_low_test(self, trim_daytime: bool = False, trim_nighttime: bool = False,
                                            lower_limit: float = None, showplot: bool = False, verbose: bool = False):
        self._level32.flag_outliers_trim_low_test(trim_daytime=trim_daytime, trim_nighttime=trim_nighttime,
                                                  lower_limit=lower_limit,
                                                  showplot=showplot, verbose=verbose)

    def level32_flag_outliers_hampel_test(self, window_length: int = 10, n_sigma: float = 5, k: float = 1.4826,
                                          showplot: bool = False, verbose: bool = False, repeat: bool = True):
        self._level32.flag_outliers_hampel_test(window_length=window_length, n_sigma=n_sigma, k=k,
                                                showplot=showplot, verbose=verbose, repeat=repeat)

    def level32_flag_outliers_hampel_dtnt_test(self, window_length: int = 10, n_sigma_dt: float = 5,
                                               n_sigma_nt: float = 5, k: float = 1.4826,
                                               showplot: bool = False, verbose: bool = False, repeat: bool = True):
        self._level32.flag_outliers_hampel_dtnt_test(window_length=window_length, n_sigma_dt=n_sigma_dt,
                                                     n_sigma_nt=n_sigma_nt, k=k,
                                                     showplot=showplot, verbose=verbose, repeat=repeat)

    def level32_flag_outliers_zscore_rolling_test(self, thres_zscore: int = 4, showplot: bool = False,
                                                  verbose: bool = False, plottitle: str = None,
                                                  repeat: bool = True, winsize: int = None):
        self._level32.flag_outliers_zscore_rolling_test(thres_zscore=thres_zscore, showplot=showplot, verbose=verbose,
                                                        plottitle=plottitle, winsize=winsize, repeat=repeat)

    def level32_flag_outliers_zscore_test(self, thres_zscore: int = 4, showplot: bool = False, verbose: bool = False,
                                          plottitle: str = None, repeat: bool = True):
        self._level32.flag_outliers_zscore_test(thres_zscore=thres_zscore, showplot=showplot,
                                                verbose=verbose, plottitle=plottitle, repeat=repeat)

    def level32_flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float = None,
                                       showplot: bool = False, verbose: bool = False, repeat: bool = True,
                                       n_jobs: int = 1):
        self._level32.flag_outliers_lof_test(n_neighbors=n_neighbors, contamination=contamination, showplot=showplot,
                                             verbose=verbose, repeat=repeat, n_jobs=n_jobs)

    def level32_flag_outliers_lof_dtnt_test(self, n_neighbors: int = None, contamination: float = None,
                                            showplot: bool = False, verbose: bool = False, repeat: bool = True,
                                            n_jobs: int = 1):
        self._level32.flag_outliers_lof_dtnt_test(n_neighbors=n_neighbors, contamination=contamination,
                                                  showplot=showplot, verbose=verbose, repeat=repeat, n_jobs=n_jobs)

    def level32_addflag(self):
        """Add current Level-3.2 flag to results."""
        self._level32.addflag()


class LoadEddyProOutputFiles:

    def __init__(
            self,
            sourcedir: str or list,
            filetype: Literal['EDDYPRO-FLUXNET-CSV-30MIN', 'EDDYPRO-FULL-OUTPUT-CSV-30MIN']
    ):
        self.sourcedir = sourcedir
        self.filetype = filetype

        self._maindf = None
        self._filepaths = None
        self._metadata = None

    @property
    def maindf(self) -> DataFrame:
        """Return input dataframe."""
        if not isinstance(self._maindf, DataFrame):
            raise Exception('No Level-1 data available, please run .loadfiles() first.')
        return self._maindf

    @property
    def filepaths(self) -> list:
        """Return filepaths to found Level-1 files."""
        if not isinstance(self._filepaths, list):
            raise Exception('Filepaths not available, please run .searchfiles() first.')
        return self._filepaths

    @property
    def metadata(self) -> DataFrame:
        """Return input dataframe."""
        if not isinstance(self._metadata, DataFrame):
            raise Exception('No Level-1 metadata available, please run .loadfiles() first. '
                            'Note that units are only available in _full_output_ files.')
        return self._metadata

    def searchfiles(self, extension: str = '*.csv'):
        """Search CSV files in source folder and keep selected filetypes."""
        fileids = self._init_filetype()
        self._filepaths = search_files(self.sourcedir, extension)
        self._filepaths = filter_strings_by_elements(list1=self.filepaths, list2=fileids)
        print(f"Found {len(self.filepaths)} files with extension {extension} and file IDs {fileids}:")
        [print(f" Found file #{ix + 1}: {f}") for ix, f in enumerate(self.filepaths)]

    def loadfiles(self):
        """Load data files"""
        loaddatafile = MultiDataFileReader(filetype=self.filetype, filepaths=self.filepaths)
        self._maindf = loaddatafile.data_df
        self._metadata = loaddatafile.metadata_df

    def _init_filetype(self):
        if self.filetype == 'EDDYPRO-FLUXNET-CSV-30MIN':
            fileids = ['eddypro_', '_fluxnet_']
        elif self.filetype == 'EDDYPRO-FULL-OUTPUT-CSV-30MIN':
            fileids = ['eddypro_', '_full_output_']
        else:
            raise Exception("Filetype is unknown.")
        return fileids


class QuickFluxProcessingChain:

    def __init__(self,
                 fluxvars: list,
                 sourcedirs: list,
                 site_lat: float,
                 site_lon: float,
                 utc_offset: int,
                 nighttime_threshold: int = 50,
                 daytime_accept_qcf_below: int = 2,
                 nighttimetime_accept_qcf_below: int = 2):
        self.fluxvars = fluxvars
        self.sourcedirs = sourcedirs
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.utc_offset = utc_offset
        self.nighttime_threshold = nighttime_threshold
        self.daytime_accept_qcf_below = daytime_accept_qcf_below
        self.nighttimetime_accept_qcf_below = nighttimetime_accept_qcf_below

        self.fpc = None

        self._run()

    def _run(self):
        self.maindf, self.metadata = self._load_data()

        for fluxcol in self.fluxvars:
            self.fpc = self._start_fpc(fluxcol=fluxcol)
            self._run_level2()
            self._run_level31()
            self._run_level32()

    def _run_level32(self):
        self.fpc.level32_stepwise_outlier_detection()
        self.fpc.level32_flag_outliers_zscore_dtnt_test(thres_zscore=4, showplot=True, verbose=True, repeat=True)
        self.fpc.level32_addflag()
        self.fpc.finalize_level32(nighttime_threshold=self.nighttime_threshold,
                                  daytime_accept_qcf_below=self.daytime_accept_qcf_below,
                                  nighttimetime_accept_qcf_below=self.nighttimetime_accept_qcf_below)

        # self.fpc.filteredseries
        # self.fpc.level32.results
        self.fpc.level32_qcf.showplot_qcf_heatmaps()
        # self.fpc.level32_qcf.showplot_qcf_timeseries()
        # self.fpc.level32_qcf.report_qcf_flags()
        self.fpc.level32_qcf.report_qcf_evolution()
        self.fpc.level32_qcf.report_qcf_series()

    def _run_level31(self):
        self.fpc.level31_storage_correction(gapfill_storage_term=False)
        self.fpc.finalize_level31()
        # fpc.level31.showplot(maxflux=50)
        self.fpc.level31.report()

    def _run_level2(self):
        LEVEL2_SETTINGS = {
            'signal_strength': False,
            'raw_data_screening_vm97': {'spikes': True, 'amplitude': False,
                                        'dropout': False, 'abslim': False,
                                        'skewkurt_hf': False, 'skewkurt_sf': False,
                                        'discont_hf': False,
                                        'discont_sf': False},
            'ssitc': True,
            'gas_completeness': False,
            'spectral_correction_factor': True,
            'angle_of_attack': False,
            'steadiness_of_horizontal_wind': False
        }
        self.fpc.level2_quality_flag_expansion(**LEVEL2_SETTINGS)
        self.fpc.finalize_level2(nighttime_threshold=self.nighttime_threshold,
                                 daytime_accept_qcf_below=self.daytime_accept_qcf_below,
                                 nighttimetime_accept_qcf_below=self.nighttimetime_accept_qcf_below)

    def _start_fpc(self, fluxcol: str):
        fpc = FluxProcessingChain(
            maindf=self.maindf,
            fluxcol=fluxcol,
            site_lat=self.site_lat,
            site_lon=self.site_lon,
            utc_offset=self.utc_offset
        )
        return fpc

    def _load_data(self):
        ep = LoadEddyProOutputFiles(sourcedir=self.sourcedirs, filetype='EDDYPRO-FLUXNET-CSV-30MIN')
        ep.searchfiles()
        ep.loadfiles()
        return ep.maindf, ep.metadata


def example_quick():
    QuickFluxProcessingChain(
        fluxvars=['FC', 'LE', 'H'],
        sourcedirs=[r'L:\Sync\luhk_work\CURRENT\fru\Level-1_results_fluxnet_2022'],
        site_lat=47.115833,
        site_lon=8.537778,
        utc_offset=1,
        nighttime_threshold=50,
        daytime_accept_qcf_below=2,
        nighttimetime_accept_qcf_below=2
    )


def example():
    # Source data
    from pathlib import Path
    from diive.core.io.files import load_parquet
    SOURCEDIR = r"L:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_cha_fp2024_2005-2023\0_data\RESULTS-IRGA-Level-1_fluxnet_2005-2023"
    FILENAME = r"CH-CHA_IRGA_Level-1_eddypro_fluxnet_2005-2023_availableVars.parquet"
    FILEPATH = Path(SOURCEDIR) / FILENAME
    maindf = load_parquet(filepath=FILEPATH)
    maindf = maindf.loc[maindf.index.year == 2023, :].copy()
    metadata = None
    print(maindf)

    # Flux processing chain settings
    FLUXVAR = "FC"
    SITE_LAT = 47.210227
    SITE_LON = 8.410645
    UTC_OFFSET = 1
    NIGHTTIME_THRESHOLD = 50  # Threshold for potential radiation in W m-2, conditions below threshold are nighttime
    DAYTIME_ACCEPT_QCF_BELOW = 2
    NIGHTTIMETIME_ACCEPT_QCF_BELOW = 2

    from diive.core.dfun.stats import sstats  # Time series stats
    sstats(maindf[FLUXVAR])
    # TimeSeries(series=level1_df[FLUXVAR]).plot()

    fpc = FluxProcessingChain(
        maindf=maindf,
        fluxcol=FLUXVAR,
        site_lat=SITE_LAT,
        site_lon=SITE_LON,
        utc_offset=UTC_OFFSET
    )

    # --------------------
    # Level-2
    # --------------------
    TEST_SSITC = False  # Default True
    TEST_GAS_COMPLETENESS = False  # Default True
    TEST_SPECTRAL_CORRECTION_FACTOR = False  # Default True

    # Signal strength
    TEST_SIGNAL_STRENGTH = False
    TEST_SIGNAL_STRENGTH_COL = 'CUSTOM_AGC_MEAN'
    TEST_SIGNAL_STRENGTH_METHOD = 'discard above'
    TEST_SIGNAL_STRENGTH_THRESHOLD = 90
    # TimeSeries(series=maindf[TEST_SIGNAL_STRENGTH_COL]).plot()

    TEST_RAWDATA = True  # Default True
    TEST_RAWDATA_SPIKES = True  # Default True
    TEST_RAWDATA_AMPLITUDE = True  # Default True
    TEST_RAWDATA_DROPOUT = True  # Default True
    TEST_RAWDATA_ABSLIM = True  # Default False
    TEST_RAWDATA_SKEWKURT_HF = True  # Default False
    TEST_RAWDATA_SKEWKURT_SF = True  # Default False
    TEST_RAWDATA_DISCONT_HF = True  # Default False
    TEST_RAWDATA_DISCONT_SF = True  # Default False

    TEST_RAWDATA_ANGLE_OF_ATTACK = False  # Default False
    # TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = [['2023-07-01', '2023-09-01']]  # Default False
    TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = False  # Default False

    TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND = False  # Default False

    LEVEL2_SETTINGS = {
        'signal_strength': {
            'test_signal_strength': TEST_SIGNAL_STRENGTH,
            'signal_strength_col': TEST_SIGNAL_STRENGTH_COL,
            'method': TEST_SIGNAL_STRENGTH_METHOD,
            'threshold': TEST_SIGNAL_STRENGTH_THRESHOLD},
        'raw_data_screening_vm97': {
            'raw_data_screening_vm97': TEST_RAWDATA,
            'spikes': TEST_RAWDATA_SPIKES,
            'amplitude': TEST_RAWDATA_AMPLITUDE,
            'dropout': TEST_RAWDATA_DROPOUT,
            'abslim': TEST_RAWDATA_ABSLIM,
            'skewkurt_hf': TEST_RAWDATA_SKEWKURT_HF,
            'skewkurt_sf': TEST_RAWDATA_SKEWKURT_SF,
            'discont_hf': TEST_RAWDATA_DISCONT_HF,
            'discont_sf': TEST_RAWDATA_DISCONT_SF},
        'ssitc': TEST_SSITC,
        'gas_completeness': TEST_GAS_COMPLETENESS,
        'spectral_correction_factor': TEST_SPECTRAL_CORRECTION_FACTOR,
        'angle_of_attack': {
            'test_rawdata_angle_of_attack': TEST_RAWDATA_ANGLE_OF_ATTACK,
            'test_rawdata_angle_of_attack_application_dates': TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES},
        'steadiness_of_horizontal_wind': TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND
    }
    fpc.level2_quality_flag_expansion(**LEVEL2_SETTINGS)
    fpc.finalize_level2(nighttime_threshold=NIGHTTIME_THRESHOLD, daytime_accept_qcf_below=DAYTIME_ACCEPT_QCF_BELOW,
                        nighttimetime_accept_qcf_below=NIGHTTIMETIME_ACCEPT_QCF_BELOW)
    fpc.level2_qcf.showplot_qcf_heatmaps()
    # fpc.level2_qcf.report_qcf_evolution()
    # fpc.level2_qcf.report_qcf_flags()
    # fpc.level2.results
    # fpc.fpc_df
    # fpc.filteredseries
    # [x for x in fpc.fpc_df.columns if 'L2' in x]

    # # --------------------
    # # Level-3.1
    # # --------------------
    # fpc.level31_storage_correction(gapfill_storage_term=False)
    # fpc.finalize_level31()
    # # fpc.level31.showplot(maxflux=50)
    # fpc.level31.report()
    # # fpc.fpc_df
    # # fpc.filteredseries
    # # fpc.level31.results
    # # [x for x in fpc.fpc_df.columns if 'L3.1' in x]

    # --------------------
    # Level-3.2
    # --------------------
    # fpc.level32_stepwise_outlier_detection()

    # fpc.level32_flag_manualremoval_test(
    #     remove_dates=[
    #         ['2022-05-05 19:45:00', '2022-06-05 19:45:00'],
    #         '2022-12-12 12:45:00',
    #         '2022-01-12 13:15:00',
    #         ['2022-08-15', '2022-08-31']
    #     ],
    #     showplot=True, verbose=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_hampel_test(window_length=48 * 9, n_sigma=5, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_hampel_dtnt_test(window_length=48 * 9, n_sigma_dt=7, n_sigma_nt=5,
    #                                            showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_zscore_rolling_test(winsize=48 * 9, thres_zscore=5, showplot=True, verbose=True,
    #                                               repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_zscore_dtnt_test(thres_zscore=4, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_localsd_test(n_sd=3, winsize=480, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_increments_zcore_test(thres_zscore=4, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()
    # fpc.level32.showplot_cleaned()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_lof_dtnt_test(n_neighbors=20, contamination=None, showplot=True,
    #                                         verbose=True, repeat=False, n_jobs=-1)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_lof_test(n_neighbors=20, contamination=None, showplot=True, verbose=True,
    #                                    repeat=False, n_jobs=-1)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_zscore_test(thres_zscore=3, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()
    # fpc.level32.results

    # fpc.level32_flag_outliers_abslim_test(minval=-20, maxval=10, showplot=True, verbose=True)
    # fpc.level32_addflag()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_abslim_dtnt_test(daytime_minmax=[-50, 50], nighttime_minmax=[-10, 50], showplot=True,
    #                                            verbose=True)
    # fpc.level32_addflag()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_trim_low_test(trim_nighttime=True, lower_limit=-20, showplot=True, verbose=True)
    # fpc.level32_addflag()

    # fpc.finalize_level32(nighttime_threshold=50, daytime_accept_qcf_below=2, nighttimetime_accept_qcf_below=2)

    # # fpc.filteredseries
    # # fpc.level32.flags
    # fpc.level32_qcf.showplot_qcf_heatmaps()
    # # fpc.level32_qcf.showplot_qcf_timeseries()
    # # fpc.level32_qcf.report_qcf_flags()
    # fpc.level32_qcf.report_qcf_evolution()
    # # fpc.level32_qcf.report_qcf_series()
    # # fpc.levelidstr

    # fpc.filteredseries_level2_qcf
    # fpc.filteredseries_level31_qcf
    # fpc.filteredseries_level32_qcf

    # # https://fitter.readthedocs.io/en/latest/tuto.html
    # # https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
    # from fitter import Fitter, get_common_distributions
    # f = Fitter(_test.dropna().to_numpy(), distributions=get_common_distributions())
    # f.fit()
    # f.summary()
    #
    # #          sumsquare_error          aic            bic    kl_div  ks_statistic      ks_pvalue
    # # norm            0.006988  1403.489479 -377396.228381  0.057412      0.061269   4.660184e-82
    # # lognorm         0.007025  1428.315952 -377253.532484  0.057257      0.063151   3.743164e-87
    # # gamma           0.008857  1449.221743 -371458.509961  0.073552      0.077689  9.752981e-132
    # # cauchy          0.009007  1025.350595 -371047.546496  0.064971      0.080458  2.817290e-141
    # # chi2            0.010701  1420.444096 -366728.335918  0.092422      0.090688  1.948207e-179
    #
    # f.get_best(method='sumsquare_error')
    # # {'norm': {'loc': -4.008143649847604, 'scale': 6.44354010359457}}
    #
    # f.fitted_param["norm"]
    # # (-4.008143649847604, 6.44354010359457)
    #
    # pdf = f.fitted_pdf['norm']
    #
    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=_test, vmin=-20, vmax=20).show()
    #
    # import math
    # import numpy as np
    # from scipy.stats import shapiro
    # from scipy.stats import lognorm
    #
    # # make this example reproducible
    # np.random.seed(1)
    #
    # # generate dataset that contains 1000 log-normal distributed values
    # lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)
    #
    # # perform Shapiro-Wilk test for normality
    # shapiro(lognorm_dataset)
    #
    # shapiro(_test.dropna().to_numpy())
    #
    # import math
    # import numpy as np
    # from scipy.stats import kstest
    # from scipy.stats import lognorm
    #
    # # make this example reproducible
    # np.random.seed(1)
    #
    # # generate dataset that contains 1000 log-normal distributed values
    # lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)
    #
    # # perform Kolmogorov-Smirnov test for normality
    # kstest(lognorm_dataset, 'norm')
    #
    # kstest(_test.dropna().to_numpy(), 'norm')
    #
    # import math
    # import numpy as np
    # from scipy.stats import lognorm
    # import statsmodels.api as sm
    # import matplotlib.pyplot as plt
    #
    # # make this example reproducible
    # np.random.seed(1)
    #
    # # generate dataset that contains 1000 log-normal distributed values
    # lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)
    #
    # # create Q-Q plot with 45-degree line added to plot
    # fig = sm.qqplot(lognorm_dataset, line='45')
    #
    # fig = sm.qqplot(_test.dropna().to_numpy(), line='45')
    #
    # plt.show()
    #
    # import math
    # import numpy as np
    # from scipy.stats import lognorm
    # import matplotlib.pyplot as plt
    #
    # # make this example reproducible
    # np.random.seed(1)
    #
    # # generate dataset that contains 1000 log-normal distributed values
    # lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)
    #
    # # create histogram to visualize values in dataset
    # plt.hist(lognorm_dataset, edgecolor='black', bins=20)
    # plt.show()
    #
    # plt.hist(_test.dropna().to_numpy(), edgecolor='black', bins=20)
    # plt.show()
    #
    # dt = _df['SW_IN_POT'] > 50
    # plt.hist(_test.loc[dt].dropna().to_numpy(), edgecolor='black', bins=20)
    # plt.show()
    # fig = sm.qqplot(_test.loc[dt].dropna().to_numpy(), line='45')
    # plt.show()
    #
    # nt = _df['SW_IN_POT'] < 50
    # plt.hist(_test.loc[nt].dropna().to_numpy(), edgecolor='black', bins=20)
    # plt.show()
    # fig = sm.qqplot(_test.loc[nt].dropna().to_numpy(), line='45')
    # plt.show()
    #
    # # https://www.statology.org/normality-test-python/
    # # https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
    #
    # print("X")


if __name__ == '__main__':
    # example_quick()
    example()
