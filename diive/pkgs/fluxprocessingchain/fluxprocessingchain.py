#  TODO NEEDS FLOW CHECK
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.core.funcs.funcs import filter_strings_by_elements
from diive.core.io.filereader import MultiDataFileReader, search_files
from diive.pkgs.createvar.potentialradiation import potrad
from diive.pkgs.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsEddyPro
from diive.pkgs.fluxprocessingchain.level31_storagecorrection import FluxStorageCorrectionSinglePointEddyPro
from diive.pkgs.outlierdetection.stepwiseoutlierdetection import StepwiseOutlierDetection
from diive.pkgs.qaqc.qcf import FlagQCF
from diive.pkgs.flux.common import detect_basevar


class LoadEddyProOutputFiles:

    def __init__(
            self,
            sourcedir: str or list,
            filetype: Literal['EDDYPRO_FLUXNET_30MIN', 'EDDYPRO_FULL_OUTPUT_30MIN']
    ):
        self.sourcedir = sourcedir
        self.filetype = filetype

        self._level1_df = None
        self._filepaths = None
        self._metadata = None

    @property
    def level1_df(self) -> DataFrame:
        """Return input dataframe."""
        if not isinstance(self._level1_df, DataFrame):
            raise Exception('No Level-1 data available, please run .loadfiles() first.')
        return self._level1_df

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
        self._level1_df = loaddatafile.data_df
        self._metadata = loaddatafile.metadata_df

    def _init_filetype(self):
        if self.filetype == 'EDDYPRO_FLUXNET_30MIN':
            fileids = ['eddypro_', '_fluxnet_']
        elif self.filetype == 'EDDYPRO_FULL_OUTPUT_30MIN':
            fileids = ['eddypro_', '_full_output_']
        else:
            raise Exception("Filetype is unknown.")
        return fileids


class FluxProcessingChain:

    def __init__(
            self,
            level1_df: DataFrame,
            filetype: Literal['EDDYPRO_FLUXNET_30MIN', 'EDDYPRO_FULL_OUTPUT_30MIN'],
            fluxcol: str,
            site_lat: float,
            site_lon: float,
            utc_offset: int,
            level1_metadata: DataFrame = None
    ):

        self.level1_df = level1_df
        self.fluxcol = fluxcol
        self.filetype = filetype
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.utc_offset = utc_offset

        # Get units from metadata, later only needed for _full_output_ files (for VM97 quality flags)
        self.level1_units = level1_metadata['UNITS'].copy()
        self.level1_units = self.level1_units.to_dict()

        # Detect base variable that was used to produce this flux
        self.basevar = detect_basevar(fluxcol=fluxcol, filetype=self.filetype)

        # Collect all relevant variables for this flux in dataframe
        self._fpc_df = self.level1_df[[fluxcol]].copy()

        # Add potential radiation, used for detecting daytime/nighttime
        swinpot = potrad(timestamp_index=self._fpc_df.index,
                         lat=site_lat, lon=site_lon, utc_offset=utc_offset)
        self.swinpot_col = swinpot.name
        self._fpc_df[self.swinpot_col] = swinpot

        # Get the name of the base flux, used to assemble meaningful names for output variables
        if self.fluxcol == 'co2_flux' or self.fluxcol == 'FC':
            # CO2 flux changes to NEE during processing (in Level-3.1)
            self.outname = 'NEE'
        else:
            self.outname = self.fluxcol

        # Init new variables
        self._levelidstr = []  # ID strings used to tag the different flux levels
        self._metadata = None
        self._filteredseries = None
        self._level1_df = None
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
    def fpc_df(self) -> DataFrame:
        """Return fluxes and flags from each levels."""
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

    def level2_quality_flag_expansion(
            self,
            signal_strength: dict or False = False,
            raw_data_screening_vm97: dict or False = False,
            ssitc: bool = True,
            gas_completeness: bool = False,
            spectral_correction_factor: bool = True,
            angle_of_attack: bool = False,
            steadiness_of_horizontal_wind: bool = False
    ):
        """Expand flux quality flag based on EddyPro output"""
        idstr = 'L2'
        self._levelidstr.append(idstr)
        self._level2 = FluxQualityFlagsEddyPro(fluxcol=self.fluxcol,
                                               dfin=self.level1_df,
                                               units=self.level1_units,
                                               idstr=idstr,
                                               basevar=self.basevar,
                                               filetype=self.filetype)
        self._level2.missing_vals_test()

        if ssitc:
            self._level2.ssitc_test()

        if gas_completeness:
            self._level2.gas_completeness_test()

        if spectral_correction_factor:
            self._level2.spectral_correction_factor_test()

        if signal_strength:
            self._level2.signal_strength_test(signal_strength_col=signal_strength['signal_strength_col'],
                                              method=signal_strength['method'],
                                              threshold=signal_strength['threshold'])

        if raw_data_screening_vm97:
            self._level2.raw_data_screening_vm97_tests(spikes=raw_data_screening_vm97['spikes'],
                                                       amplitude=raw_data_screening_vm97['amplitude'],
                                                       dropout=raw_data_screening_vm97['dropout'],
                                                       abslim=raw_data_screening_vm97['abslim'],
                                                       skewkurt_hf=raw_data_screening_vm97['skewkurt_hf'],
                                                       skewkurt_sf=raw_data_screening_vm97['skewkurt_sf'],
                                                       discont_hf=raw_data_screening_vm97['discont_hf'],
                                                       discont_sf=raw_data_screening_vm97['discont_sf'])
        if angle_of_attack:
            self._level2.angle_of_attack_test()

        if steadiness_of_horizontal_wind:
            self._level2.steadiness_of_horizontal_wind()

    def _detect_new_columns(self, level_df: DataFrame) -> list:
        # Before export, check for already existing columns
        existcols = [c for c in level_df.columns if c in self.fpc_df.columns]
        for c in level_df[existcols]:
            if not level_df[c].equals(self.fpc_df[c]):
                raise Exception(f"Column {c} was identified as duplicate, but is not identical.")

        # Add new columns to processing chain dataframe
        newcols = [c for c in level_df.columns if c not in self.fpc_df.columns]
        [print(f"++Added new column {col}.") for col in newcols]
        return newcols

    def _finalize_level(self,
                        run_qcf_on_col: str,
                        idstr: str,
                        level_df: DataFrame,
                        nighttime_threshold: int = 50,
                        daytime_accept_qcf_below: int = 2,
                        nighttimetime_accept_qcf_below: int = 2) -> FlagQCF:

        # Detect new columns
        newcols = self._detect_new_columns(level_df=level_df)
        self._fpc_df = pd.concat([self.fpc_df, level_df[newcols]], axis=1)

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

    def finalize_level32(self,
                         nighttime_threshold: int = 50,
                         daytime_accept_qcf_below: int = 2,
                         nighttimetime_accept_qcf_below: int = 2):
        """Calculate overall quality flag QCF after Level-3.2"""
        self._level32_qcf = self._finalize_level(
            run_qcf_on_col=self.level31.flux_corrected_col,
            idstr='L3.2',
            level_df=self.level32.results,
            nighttime_threshold=nighttime_threshold,
            daytime_accept_qcf_below=daytime_accept_qcf_below,
            nighttimetime_accept_qcf_below=nighttimetime_accept_qcf_below
        )

    def level31_storage_correction(self, gapfill_storage_term: bool = False):
        """Correct flux with storage term from single point measurement."""
        idstr = 'L3.1'
        self._levelidstr.append(idstr)
        self._level31 = FluxStorageCorrectionSinglePointEddyPro(df=self.level1_df,
                                                                fluxcol=self.fluxcol,
                                                                basevar=self.basevar,
                                                                gapfill_storage_term=gapfill_storage_term,
                                                                filetype=self.filetype,
                                                                idstr=idstr)
        self._level31.storage_correction()

    def finalize_level31(self):

        newcols = self._detect_new_columns(level_df=self.level31.results)
        self._fpc_df = pd.concat([self.fpc_df, self.level31.results[newcols]], axis=1)

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

    def level32_flag_outliers_zscore_dtnt_test(self, threshold: float = 4, showplot: bool = False,
                                               verbose: bool = False):
        self._level32.flag_outliers_zscore_dtnt_test(threshold=threshold, showplot=showplot, verbose=verbose)

    def level32_flag_outliers_localsd_test(self, n_sd: float = 7, winsize: int = None, showplot: bool = False,
                                           verbose: bool = False):
        self._level32.flag_outliers_localsd_test(n_sd=n_sd, winsize=winsize, showplot=showplot, verbose=verbose)

    def level32_flag_manualremoval_test(self, remove_dates: list, showplot: bool = False, verbose: bool = False):
        self._level32.flag_manualremoval_test(remove_dates=remove_dates, showplot=showplot, verbose=verbose)

    def level32_flag_outliers_increments_zcore_test(self, threshold: int = 30, showplot: bool = False,
                                                    verbose: bool = False):
        self._level32.flag_outliers_increments_zcore_test(threshold=threshold, showplot=showplot, verbose=verbose)

    def level32_flag_outliers_zscore_test(self, threshold: int = 4, showplot: bool = False, verbose: bool = False,
                                          plottitle: str = None):
        self._level32.flag_outliers_zscore_test(threshold=threshold, showplot=showplot,
                                                verbose=verbose, plottitle=plottitle)

    def level32_flag_outliers_stl_rz_test(self, zfactor: float = 4.5, decompose_downsampling_freq: str = '1H',
                                          repeat: bool = False, showplot: bool = False):
        self._level32.flag_outliers_stl_rz_test(zfactor=zfactor,
                                                decompose_downsampling_freq=decompose_downsampling_freq,
                                                repeat=repeat, showplot=showplot)

    def level32_flag_outliers_thymeboost_test(self, showplot: bool = False, verbose: bool = False):
        self._level32.flag_outliers_thymeboost_test(showplot=showplot, verbose=verbose)

    def level32_flag_outliers_lof_test(self, n_neighbors: int = None, contamination: float = 'auto',
                                       showplot: bool = False, verbose: bool = False):
        self._level32.flag_outliers_lof_test(n_neighbors=n_neighbors, contamination=contamination, showplot=showplot,
                                             verbose=verbose)

    def level32_flag_outliers_lof_dtnt_test(self, n_neighbors: int = None, contamination: float = 'auto',
                                            showplot: bool = False, verbose: bool = False):
        self._level32.flag_outliers_lof_dtnt_test(n_neighbors=n_neighbors, contamination=contamination,
                                                  showplot=showplot, verbose=verbose)

    def level32_addflag(self):
        """Add current Level-3.2 flag to results."""
        self._level32.addflag()


def example():
    ep = LoadEddyProOutputFiles(sourcedir=r'L:\Sync\luhk_work\TMP\fru',
                                filetype='EDDYPRO_FULL_OUTPUT_30MIN')
    ep.searchfiles()
    ep.loadfiles()
    level1_df = ep.level1_df
    level1_metadata = ep.metadata

    fpc = FluxProcessingChain(
        level1_df=level1_df,
        # filetype='EDDYPRO_FLUXNET_30MIN',
        filetype='EDDYPRO_FULL_OUTPUT_30MIN',
        # fluxcol='FN2O',
        fluxcol='co2_flux',
        # fluxcol='n2o_flux',
        site_lat=47.115833,
        site_lon=8.537778,
        utc_offset=1,
        level1_metadata=level1_metadata
    )

    fpc.level2_quality_flag_expansion(
        # signal_strength=dict(signal_strength_col='CUSTOM_AGC_MEAN',
        #                      method='discard above',
        #                      threshold=90),
        signal_strength=dict(signal_strength_col='agc_mean',
                             method='discard above',
                             threshold=90),
        raw_data_screening_vm97=dict(spikes=True,
                                     amplitude=True,
                                     dropout=True,
                                     abslim=False,
                                     skewkurt_hf=False,
                                     skewkurt_sf=False,
                                     discont_hf=False,
                                     discont_sf=False),
        ssitc=True,
        gas_completeness=True,
        spectral_correction_factor=True,
        angle_of_attack=True,
        steadiness_of_horizontal_wind=True
    )

    fpc.finalize_level2(nighttime_threshold=50,
                        daytime_accept_qcf_below=2,
                        nighttimetime_accept_qcf_below=2)

    fpc.filteredseries
    fpc.level2.results
    fpc.level2_qcf.showplot_qcf_heatmaps()
    fpc.level2_qcf.showplot_qcf_timeseries()
    fpc.level2_qcf.report_qcf_evolution()
    fpc.level2_qcf.report_qcf_series()
    fpc.level2_qcf.report_qcf_flags()

    fpc.level31_storage_correction(gapfill_storage_term=False)
    fpc.finalize_level31()

    fpc.filteredseries
    fpc.level31.results

    fpc.level32_stepwise_outlier_detection()

    fpc.level32_flag_outliers_abslim_dtnt_test(daytime_minmax=[-50, 50], nighttime_minmax=[-10, 50],
                                               showplot=True, verbose=True)
    fpc.level32_addflag()

    fpc.level32_flag_outliers_abslim_test(minval=-50, maxval=50, showplot=True, verbose=True)
    fpc.level32_addflag()

    # fpc.level32_flag_outliers_zscore_dtnt_test(threshold=4, showplot=True, verbose=True)
    # fpc.level32_addflag()
    #
    # fpc.level32_flag_outliers_localsd_test(n_sd=4, winsize=480, showplot=True, verbose=True)
    # fpc.level32_addflag()
    #
    # fpc.level32_flag_manualremoval_test(remove_dates=[['2023-03-05 19:45:00', '2023-04-05 19:45:00']],
    #                                     showplot=True, verbose=True)
    # fpc.level32_addflag()
    #
    # fpc.level32_flag_outliers_increments_zcore_test(threshold=8, showplot=True, verbose=True)
    # fpc.level32_addflag()
    #
    # fpc.level32_flag_outliers_zscore_test(threshold=5, showplot=True, verbose=True)
    # fpc.level32_addflag()
    #
    # fpc.level32_flag_outliers_stl_rz_test(zfactor=3, decompose_downsampling_freq='6H', repeat=False, showplot=True)
    # fpc.level32_addflag()
    #
    # fpc.level32_flag_outliers_thymeboost_test(showplot=True, verbose=True)
    # fpc.level32_addflag()
    #
    # fpc.level32_flag_outliers_lof_test(n_neighbors=None, contamination=0.005, showplot=True, verbose=True)
    # fpc.level32_addflag()
    #
    # fpc.level32_flag_outliers_lof_dtnt_test(n_neighbors=None, contamination=0.0005, showplot=True, verbose=True)
    # fpc.level32_addflag()

    fpc.finalize_level32(nighttime_threshold=50, daytime_accept_qcf_below=2, nighttimetime_accept_qcf_below=2)

    fpc.filteredseries
    fpc.level32.results
    fpc.level32_qcf.showplot_qcf_heatmaps()
    fpc.level32_qcf.showplot_qcf_timeseries()
    fpc.level32_qcf.report_qcf_flags()
    fpc.level32_qcf.report_qcf_evolution()
    fpc.level32_qcf.report_qcf_series()
    fpc.levelidstr

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
    example()
