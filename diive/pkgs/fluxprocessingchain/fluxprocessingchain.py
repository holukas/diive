#  TODO NEEDS FLOW CHECK
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.core.dfun.frames import detect_new_columns
from diive.core.funcs.funcs import filter_strings_by_elements
from diive.core.io.filereader import MultiDataFileReader, search_files
from diive.core.plotting.cumulative import Cumulative, CumulativeYear
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.pkgs.createvar.daynightflag import daytime_nighttime_flag_from_swinpot
from diive.pkgs.createvar.potentialradiation import potrad
from diive.pkgs.flux.common import detect_fluxbasevar
from diive.pkgs.flux.hqflux import analyze_highest_quality_flux
from diive.pkgs.flux.ustarthreshold import FlagMultipleConstantUstarThresholds
from diive.pkgs.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsEddyPro
from diive.pkgs.fluxprocessingchain.level31_storagecorrection import FluxStorageCorrectionSinglePointEddyPro
from diive.pkgs.gapfilling.longterm import LongTermGapFillingRandomForestTS
from diive.pkgs.outlierdetection.stepwiseoutlierdetection import StepwiseOutlierDetection
from diive.pkgs.qaqc.qcf import FlagQCF


class FluxProcessingChain:

    def __init__(
            self,
            df: DataFrame,
            fluxcol: str,
            site_lat: float,
            site_lon: float,
            utc_offset: int,
            nighttime_threshold: float = 20,
            daytime_accept_qcf_below: int = 1,
            nighttimetime_accept_qcf_below: int = 1
    ):

        self._df = df.copy()
        self.fluxcol = fluxcol
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.utc_offset = utc_offset
        self.nighttime_threshold = nighttime_threshold
        self.daytime_accept_qcf_below = daytime_accept_qcf_below
        self.nighttimetime_accept_qcf_below = nighttimetime_accept_qcf_below

        # Detect base variable that was used to produce this flux
        self.fluxbasevar = detect_fluxbasevar(fluxcol=fluxcol)

        # Collect all relevant variables for this flux in dataframe
        self.ustarcol = 'USTAR'
        requiredcols = [self.fluxcol, self.ustarcol]
        self._fpc_df = self.df[requiredcols].copy()

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
        self._filteredseries_level33_qcf = dict()  # dict because there can be multiple USTAR scenarios
        self._maindf = None
        self._level2 = None
        self._level31 = None
        self._level32 = None
        self._level33 = None
        self._level41 = dict()  # dict because there can be multiple USTAR scenarios
        self._level2_qcf = None
        self._level32_qcf = None
        self._level33_qcf = dict()  # dict because there can be multiple USTAR scenarios

    def get_data(self, verbose: int = 1):
        """Return dataframe containing all input data and results."""
        # newcols = detect_new_columns(df=self.fpc_df, other=self.df)
        newcols = [c for c in self.fpc_df.columns if c not in self.df.columns]
        if verbose:
            print("NEW VARIABLES FROM FLUX PROCESSING CHAIN:")
            [print(f"++ {c}") for c in newcols]
            print("No variables in input data were overwritten, only new variables added.")
        full_data_df = pd.concat([self.df, self.fpc_df[newcols]], axis=1)
        # [print(f"++Added column {col}.") for col in self.fpc_df.columns]
        return full_data_df

    def get_gapfilled_names(self) -> list:
        """Return names of gap-filled variables."""
        gapfilled_vars = []
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                gapfilled_vars.append(self.level41[gfmethod][ustar_scenario].gapfilled_.name)
        return gapfilled_vars

    def get_nongapfilled_names(self) -> list:
        """Return names of target variables that were gap-filled."""
        nongapfilled_vars = []
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                nongapfilled_vars.append(self.level41[gfmethod][ustar_scenario].target_col)
        return nongapfilled_vars

    def report_gapfilling_variables(self):
        """Show names of variables before and after gap-filling."""
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                gapfilled_name = self.level41[gfmethod][ustar_scenario].gapfilled_.name
                nongapfilled_name = self.level41[gfmethod][ustar_scenario].target_col
                print(f"{gfmethod} ({ustar_scenario}): {nongapfilled_name} -> {gapfilled_name}")

    def get_gapfilled_variables(self) -> DataFrame:
        """Return data of gap-filled and non-gap-filled variables."""
        gapfilled_vars = self.get_gapfilled_names()
        nongapfilled_vars = self.get_nongapfilled_names()
        gfvars = gapfilled_vars + nongapfilled_vars
        return self.fpc_df[gfvars].copy()

    def report_gapfilling_model_scores(self):
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                print(f"\nMODEL SCORES ({gfmethod}): {ustar_scenario}")
                modelscores = pd.DataFrame.from_dict(self.level41[gfmethod][ustar_scenario].scores_, orient='columns')
                print(modelscores)

    def report_gapfilling_poolyears(self):
        print("DATA POOLS USED FOR GAP_FILLING:")
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                for yr, pool in self.level41[gfmethod][ustar_scenario].yearpools.items():
                    print(f"{yr}: {gfmethod} used data from {pool['poolyears']} "
                          f"for gap-filling {self.level41[gfmethod][ustar_scenario].target_col} and "
                          f"producing --> {self.level41[gfmethod][ustar_scenario].gapfilled_.name}")
                print("\n")

    def showplot_feature_ranks_per_year(self):
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                results = self.level41[gfmethod][ustar_scenario]
                title = f"{results.gapfilled_.name} ({ustar_scenario})"
                first_key = next(iter(self.level41[gfmethod][ustar_scenario].results_yearly_))
                model_params = self.level41[gfmethod][ustar_scenario].results_yearly_[
                    first_key].model_.get_params()
                txt = f"MODEL: {gfmethod} / PARAMS: {model_params}"
                results.showplot_feature_ranks_per_year(title=f"{title}", subtitle=f"{txt}")

    def showplot_gapfilled_heatmap(self):
        """Show heatmap plot for gap-filled and non-gap-filled data in each USTAR scenario."""
        gapfilled_vars = self.get_gapfilled_names()
        nongapfilled_vars = self.get_nongapfilled_names()
        gfvars = self.get_gapfilled_variables()
        for ix, g in enumerate(gapfilled_vars):
            fig = plt.figure(figsize=(9, 9), dpi=100)
            gs = gridspec.GridSpec(1, 2)  # rows, cols
            # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
            ax_ngf = fig.add_subplot(gs[0, 0])
            ax_gf = fig.add_subplot(gs[0, 1])
            hm = HeatmapDateTime(ax=ax_ngf, series=gfvars[nongapfilled_vars[ix]]).plot()
            hm = HeatmapDateTime(ax=ax_gf, series=gfvars[g]).plot()
            fig.tight_layout()
            fig.show()

    def showplot_gapfilled_cumulative(self, gain: float = 1, units: str = "", per_year: bool = True,
                                      start_year: int = None, end_year: int = None):
        """Show cumulative plot for gap-filled data in each USTAR scenario."""
        gapfilled_vars = self.get_gapfilled_names()
        gfvars = self.get_gapfilled_variables()[gapfilled_vars]
        if per_year:
            for g in gfvars:
                CumulativeYear(
                    series=gfvars[g].multiply(gain),
                    series_units=units,
                    yearly_end_date=None,
                    # yearly_end_date='08-11',
                    start_year=start_year,
                    end_year=end_year,
                    show_reference=True,
                    # excl_years_from_reference=[2005, 2008, 2009, 2010, 2021, 2022, 2023],
                    excl_years_from_reference=None,
                    highlight_year=None,
                    highlight_year_color='#F44336').plot()
        else:
            df = gfvars[gapfilled_vars].copy()
            df = df.multiply(gain)
            Cumulative(
                df=df,
                units=units,
                start_year=start_year,
                end_year=end_year).plot()

    @property
    def df(self) -> DataFrame:
        """Dataframe containing all input data and results."""
        if not isinstance(self._df, DataFrame):
            raise Exception(f'No dataframe available.')
        return self._df

    @property
    def filteredseries(self) -> Series:
        """Return time series of flux, filtered by all available QCF checks."""
        if not isinstance(self._filteredseries, Series):
            raise Exception(f'No filtered time series for {self.fluxcol} available, '
                            f'please run .level2_quality_flag_expansion() first.')
        return self._filteredseries

    @property
    def filteredseries_hq(self) -> Series:
        """Return time series of highest-quality flux, filtered by all available QCF checks."""
        if not isinstance(self._filteredseries_hq, Series):
            raise Exception(f'No filtered time series for {self.fluxcol} available, '
                            f'please run .level2_quality_flag_expansion() first.')
        return self._filteredseries_hq

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
    def filteredseries_level33_qcf(self) -> dict:
        """Return time series of quality-filtered flux after Level-3.3."""
        if not isinstance(self._filteredseries_level33_qcf, dict):
            raise Exception(f'No filtered time series for {self.fluxcol} available, '
                            f'please run .level33...() first.')
        return self._filteredseries_level33_qcf

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
    def level33_qcf(self) -> dict:
        """Return instance of Level-3.3 QCF creation."""
        if not isinstance(self._level33_qcf, dict):
            raise Exception('No Level-3.3 data available.')
        return self._level33_qcf

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
    def level33(self) -> FlagMultipleConstantUstarThresholds:
        """Return instance of Level-3.3 ustar."""
        if not isinstance(self._level33, FlagMultipleConstantUstarThresholds):
            raise Exception('No Level-3.3 data available, please run .level32_stepwise_outlier_detection() first.')
        return self._level33

    @property
    def level41(self) -> dict:
        """Return instance of Level-4.1 gap-filling."""
        if not isinstance(self._level41, dict):
            raise Exception('No Level-4.1 data available, please run .level33...() first.')
        return self._level41

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
            ssitc: dict or False = False,
            gas_completeness: dict or False = False,
            spectral_correction_factor: dict or False = False,
            angle_of_attack: dict or False = False,
            steadiness_of_horizontal_wind: dict or False = False
    ):
        """Expand flux quality flag based on EddyPro output"""
        idstr = 'L2'
        self._levelidstr.append(idstr)
        self._level2 = FluxQualityFlagsEddyPro(fluxcol=self.fluxcol,
                                               dfin=self.df,
                                               idstr=idstr,
                                               fluxbasevar=self.fluxbasevar)
        self._level2.missing_vals_test()

        if ssitc['apply']:
            self._level2.ssitc_test()

        if gas_completeness['apply']:
            self._level2.gas_completeness_test()

        if spectral_correction_factor['apply']:
            self._level2.spectral_correction_factor_test()

        if signal_strength['apply']:
            self._level2.signal_strength_test(signal_strength_col=signal_strength['signal_strength_col'],
                                              method=signal_strength['method'],
                                              threshold=signal_strength['threshold'])

        if raw_data_screening_vm97['apply']:
            self._level2.raw_data_screening_vm97_tests(spikes=raw_data_screening_vm97['spikes'],
                                                       amplitude=raw_data_screening_vm97['amplitude'],
                                                       dropout=raw_data_screening_vm97['dropout'],
                                                       abslim=raw_data_screening_vm97['abslim'],
                                                       skewkurt_hf=raw_data_screening_vm97['skewkurt_hf'],
                                                       skewkurt_sf=raw_data_screening_vm97['skewkurt_sf'],
                                                       discont_hf=raw_data_screening_vm97['discont_hf'],
                                                       discont_sf=raw_data_screening_vm97['discont_sf'])
        if angle_of_attack['apply']:
            self._level2.angle_of_attack_test(application_dates=angle_of_attack['application_dates'])

        if steadiness_of_horizontal_wind['apply']:
            self._level2.steadiness_of_horizontal_wind()

    def _finalize_level(self,
                        run_qcf_on_col: str,
                        idstr: str,
                        level_df: DataFrame,
                        daytime_accept_qcf_below: int = 2,
                        nighttimetime_accept_qcf_below: int = 2,
                        ustar_scenarios: list = None) -> FlagQCF:

        # Detect new columns
        newcols = detect_new_columns(df=level_df, other=self.fpc_df)
        self._fpc_df = pd.concat([self.fpc_df, level_df[newcols]], axis=1)
        [print(f"++Added new column {col}.") for col in newcols]

        # Calculate overall quality flag QCF
        qcf = FlagQCF(series=self.fpc_df[run_qcf_on_col],
                      df=self.fpc_df,
                      idstr=idstr,
                      swinpot=self.fpc_df['SW_IN_POT'],  # Calculated during init
                      nighttime_threshold=self.nighttime_threshold,
                      ustar_scenarios=ustar_scenarios  # Required to get correct USTAR FLAG_ columns for each scenario
                      )
        qcf.calculate(daytime_accept_qcf_below=daytime_accept_qcf_below,
                      nighttimetime_accept_qcf_below=nighttimetime_accept_qcf_below)
        self._fpc_df = qcf.get()

        self._filteredseries = qcf.filteredseries.copy()

        return qcf

    def finalize_level2(self):
        """Calculate overall quality flag QCF after Level-2"""
        self._level2_qcf = self._finalize_level(
            run_qcf_on_col=self.fluxcol,
            idstr='L2',
            level_df=self.level2.results,
            daytime_accept_qcf_below=self.daytime_accept_qcf_below,
            nighttimetime_accept_qcf_below=self.nighttimetime_accept_qcf_below
        )
        self._filteredseries_level2_qcf = self.filteredseries.copy()  # Store filtered series as variable

    def finalize_level32(self):
        """Calculate overall quality flag QCF after Level-3.2"""
        self._level32_qcf = self._finalize_level(
            run_qcf_on_col=self.level31.flux_corrected_col,
            idstr='L3.2',
            level_df=self.level32.flags,
            daytime_accept_qcf_below=self.daytime_accept_qcf_below,
            nighttimetime_accept_qcf_below=self.nighttimetime_accept_qcf_below
        )
        self._filteredseries_level32_qcf = self.filteredseries.copy()  # Store filtered series as variable

    def level33_constant_ustar(self, thresholds: list, threshold_labels: list,
                               showplot: bool = True, verbose: bool = True):
        """Create flag to indicate time periods of low turbulence using
        one or more known constant USTAR thresholds."""
        idstr = 'L3.3'
        self._levelidstr.append(idstr)
        self._level33 = FlagMultipleConstantUstarThresholds(series=self.fpc_df[self.level31.flux_corrected_col],
                                                            ustar=self.fpc_df[self.ustarcol],
                                                            thresholds=thresholds,
                                                            threshold_labels=threshold_labels,
                                                            idstr=idstr,
                                                            showplot=showplot)
        self._level33.calc()

    def finalize_level33(self):
        """Calculate overall quality flag QCF after Level-3.3"""

        ustar_scenarios = self.level33.threshold_labels

        for u in ustar_scenarios:
            print(f"Calculating overall quality flag QCF for USTAR scenario {u}...")

            flagcol = [c for c in self.level33.results if u in c]
            flagcol = flagcol[0] if len(flagcol) == 1 else None
            udf = self.level33.results[[self.level31.flux_corrected_col, self.ustarcol, flagcol]].copy()

            # Calculate QCF for all USTAR scenarios and store in dict
            self._level33_qcf[u] = self._finalize_level(
                run_qcf_on_col=self.level31.flux_corrected_col,
                idstr=f'L3.3_{u}',
                level_df=udf,
                daytime_accept_qcf_below=self.daytime_accept_qcf_below,
                nighttimetime_accept_qcf_below=self.nighttimetime_accept_qcf_below,
                ustar_scenarios=ustar_scenarios  # Required to get the correct USTAR FLAG_ columns for each scenario
            )
            self._filteredseries_level33_qcf[u] = self.filteredseries.copy()  # Store filtered series as variable

    def level41_gapfilling_longterm(
            self,
            features: list,
            run_random_forest: bool = True,
            run_mds: bool = False,  # todo
            features_lag: list or None = None,
            reduce_features: bool = False,
            include_timestamp_as_features: bool = False,
            add_continuous_record_number: bool = False,
            sanitize_timestamp: bool = True,
            perm_n_repeats: int = 10,
            rf_kwargs: dict = None,
            verbose: int = 0
    ):
        idstr = 'L4.1'
        self._levelidstr.append(idstr)

        # todo
        if run_random_forest:
            self._run_random_forest(features, rf_kwargs, include_timestamp_as_features, add_continuous_record_number,
                                    features_lag, verbose, sanitize_timestamp, perm_n_repeats, reduce_features)

    def _run_random_forest(self, features, rf_kwargs, include_timestamp_as_features, add_continuous_record_number,
                           features_lag, verbose, sanitize_timestamp, perm_n_repeats, reduce_features):

        # Store results in separate dict within Level-4.1
        self._level41['random_forest'] = dict()

        # Default parameters for random forest models
        if not isinstance(rf_kwargs, dict):
            rf_kwargs = {'n_estimators': 200,
                         'random_state': 42,
                         'min_samples_split': 2,
                         'min_samples_leaf': 1,
                         'n_jobs': -1}

        # Collect data and run model for each USTAR scenario for gapfilling
        # todo what is if there is no scenario eg h2o fluxes?
        for ustar_scen, ustar_flux in self.filteredseries_level33_qcf.items():

            # Get features from input data
            this_ust_scen_df = self.df[features].copy()

            # Add USTAR-filtered flux from recent results (Level-3.3)
            this_ust_scen_df = pd.concat([this_ust_scen_df, self.fpc_df[ustar_flux.name]], axis=1)

            # self.level33.flux_corrected_col
            general_kwargs = dict(
                input_df=this_ust_scen_df,
                target_col=ustar_flux.name,
                include_timestamp_as_features=include_timestamp_as_features,
                add_continuous_record_number=add_continuous_record_number,
                features_lag=features_lag,
                verbose=verbose,
                sanitize_timestamp=sanitize_timestamp,
                perm_n_repeats=perm_n_repeats
            )

            # Initialize random forest for this scenario
            instance = LongTermGapFillingRandomForestTS(**general_kwargs, **rf_kwargs)

            # Assign model data
            instance.create_yearpools()

            # Init models
            instance.initialize_yearly_models()

            # Feature reduction *across all* years (not *per* year)
            if reduce_features:
                instance.reduce_features_across_years()

            # Train model and fill gaps
            instance.fillgaps()

            # Add gap-filled flux and gap-filling flag to flux processing chain dataframe
            fluxdata = instance.gapfilled_.copy()
            flagname = [c for c in instance.gapfilling_df_
                        if str(c).startswith("FLAG_") and str(c).endswith("_ISFILLED")]
            flagname = flagname[0] if len(flagname) == 1 else None
            flagdata = instance.gapfilling_df_[flagname].copy()
            self._fpc_df = pd.concat([self.fpc_df, fluxdata, flagdata], axis=1)

            # Save instance to Level-4.1
            self._level41['random_forest'][ustar_scen] = instance

        # # How to access data for each USTAR scenario:
        # for ustar_scen, ustar_flux in self.filteredseries_level33_qcf.items():
        #     fluxname = self.level41['random_forest'][ustar_scen].gapfilled_.name
        #     gapfilled_ = self.level41['random_forest'][ustar_scen].gapfilled_
        #     results_yearly_ = self.level41['random_forest'][ustar_scen].results_yearly_
        #     scores_ = self.level41['random_forest'][ustar_scen].scores_
        #     fi = self.level41['random_forest'][ustar_scen].feature_importances_yearly_
        #     feature_ranks_per_year = self.level41['random_forest'][ustar_scen].feature_ranks_per_year
        #     feature_importance_per_year = self.level41['random_forest'][ustar_scen].feature_importance_per_year
        #     features_reduced_across_years = self.level41['random_forest'][ustar_scen].features_reduced_across_years
        #     self.level41['random_forest'][ustar_scen].showplot_feature_ranks_per_year(title=f"{ustar_scen}")
        #     print(fluxname)
        #     print(gapfilled_)
        #     print(results_yearly_)
        #     print(scores_)
        #     print(fi)
        #     print(feature_ranks_per_year)
        #     print(feature_importance_per_year)
        #     print(features_reduced_across_years)

    def level31_storage_correction(self, gapfill_storage_term: bool = True):
        """Correct flux with storage term from single point measurement."""
        idstr = 'L3.1'
        self._levelidstr.append(idstr)
        self._level31 = FluxStorageCorrectionSinglePointEddyPro(df=self.df,
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
        self._filteredseries_hq = strg_corrected_flux_qcf0.copy()

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
                                           constant_sd: bool = False, verbose: bool = False, repeat: bool = True):
        self._level32.flag_outliers_localsd_test(n_sd=n_sd, winsize=winsize, showplot=showplot, verbose=verbose,
                                                 constant_sd=constant_sd, repeat=repeat)

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

    def level32_flag_outliers_zscore_rolling_test(self, thres_zscore: float = 4, showplot: bool = False,
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

    def level33_flag_constant_ustar_test(self, n_neighbors: int = None, contamination: float = None,
                                         showplot: bool = False, verbose: bool = False, repeat: bool = True,
                                         n_jobs: int = 1):
        self._level32.flag_outliers_lof_dtnt_test(n_neighbors=n_neighbors, contamination=contamination,
                                                  showplot=showplot, verbose=verbose, repeat=repeat, n_jobs=n_jobs)

    def level32_addflag(self):
        """Add current Level-3.2 flag to results."""
        self._level32.addflag()

    def analyze_highest_quality_flux(self, showplot: bool = True):
        analyze_highest_quality_flux(flux=self.fpc_df[self.filteredseries_hq.name],
                                     nighttime_flag=self.fpc_df['NIGHTTIME'],
                                     showplot=showplot)


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
        self.fpc.finalize_level32()

        # self.fpc.filteredseries
        # self.fpc.level32.results
        self.fpc.level32_qcf.showplot_qcf_heatmaps()
        # self.fpc.level32_qcf.showplot_qcf_timeseries()
        # self.fpc.level32_qcf.report_qcf_flags()
        self.fpc.level32_qcf.report_qcf_evolution()
        self.fpc.level32_qcf.report_qcf_series()

    def _run_level31(self):
        self.fpc.level31_storage_correction(gapfill_storage_term=True)
        self.fpc.finalize_level31()
        # fpc.level31.showplot(maxflux=50)
        self.fpc.level31.report()

    def _run_level2(self):
        TEST_SSITC = True  # Default True
        TEST_GAS_COMPLETENESS = True  # Default True
        TEST_SPECTRAL_CORRECTION_FACTOR = True  # Default True
        TEST_SIGNAL_STRENGTH = True
        TEST_SIGNAL_STRENGTH_COL = 'CUSTOM_AGC_MEAN'
        TEST_SIGNAL_STRENGTH_METHOD = 'discard above'
        TEST_SIGNAL_STRENGTH_THRESHOLD = 90
        # TimeSeries(series=maindf[TEST_SIGNAL_STRENGTH_COL]).plot()
        TEST_RAWDATA = True  # Default True
        TEST_RAWDATA_SPIKES = True  # Default True
        TEST_RAWDATA_AMPLITUDE = False  # Default True
        TEST_RAWDATA_DROPOUT = True  # Default True
        TEST_RAWDATA_ABSLIM = False  # Default False
        TEST_RAWDATA_SKEWKURT_HF = False  # Default False
        TEST_RAWDATA_SKEWKURT_SF = False  # Default False
        TEST_RAWDATA_DISCONT_HF = False  # Default False
        TEST_RAWDATA_DISCONT_SF = False  # Default False
        TEST_RAWDATA_ANGLE_OF_ATTACK = False  # Default False
        # TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = [['2008-01-01', '2010-01-01'],
        #                                                   ['2016-03-01', '2016-05-01']]  # Default False
        TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = False  # Default False
        TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND = False  # Default False
        LEVEL2_SETTINGS = {
            'signal_strength': {
                'apply': TEST_SIGNAL_STRENGTH,
                'signal_strength_col': TEST_SIGNAL_STRENGTH_COL,
                'method': TEST_SIGNAL_STRENGTH_METHOD,
                'threshold': TEST_SIGNAL_STRENGTH_THRESHOLD},
            'raw_data_screening_vm97': {
                'apply': TEST_RAWDATA,
                'spikes': TEST_RAWDATA_SPIKES,
                'amplitude': TEST_RAWDATA_AMPLITUDE,
                'dropout': TEST_RAWDATA_DROPOUT,
                'abslim': TEST_RAWDATA_ABSLIM,
                'skewkurt_hf': TEST_RAWDATA_SKEWKURT_HF,
                'skewkurt_sf': TEST_RAWDATA_SKEWKURT_SF,
                'discont_hf': TEST_RAWDATA_DISCONT_HF,
                'discont_sf': TEST_RAWDATA_DISCONT_SF},
            'ssitc': {
                'apply': TEST_SSITC},
            'gas_completeness': {
                'apply': TEST_GAS_COMPLETENESS},
            'spectral_correction_factor': {
                'apply': TEST_SPECTRAL_CORRECTION_FACTOR},
            'angle_of_attack': {
                'apply': TEST_RAWDATA_ANGLE_OF_ATTACK,
                'application_dates': TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES},
            'steadiness_of_horizontal_wind': {
                'apply': TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND}
        }
        self.fpc.level2_quality_flag_expansion(**LEVEL2_SETTINGS)
        self.fpc.finalize_level2()

    def _start_fpc(self, fluxcol: str):
        fpc = FluxProcessingChain(
            df=self.maindf,
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
        sourcedirs=[r'L:\Sync\luhk_work\TMP'],
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
    SOURCEDIR = r"L:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_cha_fp2024_2005-2023\notebooks\31_FLUXES_L1_mergeData_IRGA+M7+MGMT"
    FILENAME = r"31.6_CH-CHA_IRGA-L1+M7+MGMT_2005-2023.parquet"
    FILEPATH = Path(SOURCEDIR) / FILENAME
    maindf = load_parquet(filepath=str(FILEPATH))
    # SOURCEDIRS = [r"F:\TMP\x"]
    # ep = LoadEddyProOutputFiles(sourcedir=SOURCEDIRS, filetype='EDDYPRO-FLUXNET-CSV-30MIN')
    # ep.searchfiles()
    # ep.loadfiles()
    # maindf = ep.maindf
    # metadata = ep.metadata

    # locs = (maindf.index.year >= 2019) & (maindf.index.year <= 2023)
    locs = (maindf.index.year >= 2023) & (maindf.index.year <= 2023)
    maindf = maindf.loc[locs, :].copy()
    # metadata = None
    # print(maindf)

    # import matplotlib.pyplot as plt
    # locs = (maindf.index.year >= 2008) & (maindf.index.year <= 2009)
    # maindf.loc[locs, 'FC'].plot(x_compat=True)
    # plt.show()
    # plt.show()

    # RANDOM UNCERTAINTY
    # [print(c) for c in maindf.columns if "SSITC" in c]
    # import matplotlib.pyplot as plt
    # maindf['FC_RANDUNC_HF'].plot(x_compat=True)
    # locs = (
    #         (maindf['FC'] < 50)
    #         & (maindf['FC'] > -50)
    #         & (maindf['NIGHT'] == 1)
    #         & (maindf['FC_RANDUNC_HF'] < maindf['FC'].abs())
    #         & (maindf['FC_SSITC_TEST'] == 0)
    # )
    # plt.scatter(maindf.loc[locs, 'FC'], maindf.loc[locs, 'FC_RANDUNC_HF'])
    # plt.show()

    # Flux processing chain settings
    # FLUXVAR = "LE"
    # FLUXVAR = "H"
    FLUXVAR = "FC"
    SITE_LAT = 47.210227
    SITE_LON = 8.410645
    UTC_OFFSET = 1
    NIGHTTIME_THRESHOLD = 20  # Threshold for potential radiation in W m-2, conditions below threshold are nighttime
    DAYTIME_ACCEPT_QCF_BELOW = 2
    NIGHTTIMETIME_ACCEPT_QCF_BELOW = 1

    # from diive.core.dfun.stats import sstats  # Time series stats
    # sstats(maindf[FLUXVAR])
    # TimeSeries(series=level1_df[FLUXVAR]).plot()

    fpc = FluxProcessingChain(
        df=maindf,
        fluxcol=FLUXVAR,
        site_lat=SITE_LAT,
        site_lon=SITE_LON,
        utc_offset=UTC_OFFSET,
        nighttime_threshold=NIGHTTIME_THRESHOLD,
        daytime_accept_qcf_below=DAYTIME_ACCEPT_QCF_BELOW,
        nighttimetime_accept_qcf_below=NIGHTTIMETIME_ACCEPT_QCF_BELOW
    )

    # --------------------
    # Level-2
    # --------------------
    TEST_SSITC = True  # Default True
    TEST_GAS_COMPLETENESS = True  # Default True
    TEST_SPECTRAL_CORRECTION_FACTOR = True  # Default True
    TEST_SIGNAL_STRENGTH = True
    TEST_SIGNAL_STRENGTH_COL = 'CUSTOM_AGC_MEAN'
    TEST_SIGNAL_STRENGTH_METHOD = 'discard above'
    TEST_SIGNAL_STRENGTH_THRESHOLD = 90
    # TimeSeries(series=maindf[TEST_SIGNAL_STRENGTH_COL]).plot()
    TEST_RAWDATA = True  # Default True
    TEST_RAWDATA_SPIKES = True  # Default True
    TEST_RAWDATA_AMPLITUDE = False  # Default True
    TEST_RAWDATA_DROPOUT = True  # Default True
    TEST_RAWDATA_ABSLIM = False  # Default False
    TEST_RAWDATA_SKEWKURT_HF = False  # Default False
    TEST_RAWDATA_SKEWKURT_SF = False  # Default False
    TEST_RAWDATA_DISCONT_HF = False  # Default False
    TEST_RAWDATA_DISCONT_SF = False  # Default False
    TEST_RAWDATA_ANGLE_OF_ATTACK = True  # Default False
    TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = [['2008-01-01', '2010-01-01'],
                                                      ['2016-03-01', '2016-05-01']]  # Default False
    # TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = False  # Default False
    TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND = False  # Default False

    LEVEL2_SETTINGS = {
        'signal_strength': {
            'apply': TEST_SIGNAL_STRENGTH,
            'signal_strength_col': TEST_SIGNAL_STRENGTH_COL,
            'method': TEST_SIGNAL_STRENGTH_METHOD,
            'threshold': TEST_SIGNAL_STRENGTH_THRESHOLD},
        'raw_data_screening_vm97': {
            'apply': TEST_RAWDATA,
            'spikes': TEST_RAWDATA_SPIKES,
            'amplitude': TEST_RAWDATA_AMPLITUDE,
            'dropout': TEST_RAWDATA_DROPOUT,
            'abslim': TEST_RAWDATA_ABSLIM,
            'skewkurt_hf': TEST_RAWDATA_SKEWKURT_HF,
            'skewkurt_sf': TEST_RAWDATA_SKEWKURT_SF,
            'discont_hf': TEST_RAWDATA_DISCONT_HF,
            'discont_sf': TEST_RAWDATA_DISCONT_SF},
        'ssitc': {
            'apply': TEST_SSITC},
        'gas_completeness': {
            'apply': TEST_GAS_COMPLETENESS},
        'spectral_correction_factor': {
            'apply': TEST_SPECTRAL_CORRECTION_FACTOR},
        'angle_of_attack': {
            'apply': TEST_RAWDATA_ANGLE_OF_ATTACK,
            'application_dates': TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES},
        'steadiness_of_horizontal_wind': {
            'apply': TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND}
    }
    fpc.level2_quality_flag_expansion(**LEVEL2_SETTINGS)
    fpc.finalize_level2()
    # fpc.level2_qcf.showplot_qcf_heatmaps()
    # fpc.level2_qcf.report_qcf_evolution()
    # fpc.level2_qcf.analyze_highest_quality_flux()
    # fpc.level2_qcf.report_qcf_flags()
    # fpc.level2.results
    # fpc.fpc_df
    # fpc.filteredseries
    # [x for x in fpc.fpc_df.columns if 'L2' in x]

    # --------------------
    # Level-3.1
    # --------------------
    fpc.level31_storage_correction(gapfill_storage_term=True)
    fpc.finalize_level31()
    # fpc.level31.report()
    # fpc.level31.showplot()
    # fpc.fpc_df
    # fpc.filteredseries
    # fpc.level31.results
    # [x for x in fpc.fpc_df.columns if 'L3.1' in x]
    # -------------------------------------------------------------------------

    # --------------------
    # (OPTIONAL) ANALYZE
    # --------------------
    # fpc.analyze_highest_quality_flux(showplot=True)
    # -------------------------------------------------------------------------

    # --------------------
    # Level-3.2
    # --------------------
    fpc.level32_stepwise_outlier_detection()

    # fpc.level32_flag_manualremoval_test(
    #     remove_dates=[
    #         ['2016-03-18 12:15:00', '2016-05-03 06:45:00'],
    #         # '2022-12-12 12:45:00',
    #     ],
    #     showplot=True, verbose=True)
    # fpc.level32_addflag()

    DAYTIME_MINMAX = [-50, 50]
    NIGHTTIME_MINMAX = [-50, 50]
    fpc.level32_flag_outliers_abslim_dtnt_test(daytime_minmax=DAYTIME_MINMAX, nighttime_minmax=NIGHTTIME_MINMAX,
                                               showplot=False, verbose=False)
    # fpc.level32_flag_outliers_abslim_dtnt_test(daytime_minmax=DAYTIME_MINMAX, nighttime_minmax=NIGHTTIME_MINMAX, showplot=True, verbose=True)
    fpc.level32_addflag()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_zscore_dtnt_test(thres_zscore=4, showplot=True, verbose=False, repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_hampel_test(window_length=48 * 7, n_sigma=5, showplot=True, verbose=True, repeat=False)
    # fpc.level32_addflag()

    # # fpc.level32_flag_outliers_hampel_dtnt_test(window_length=48 * 3, n_sigma_dt=3.5, n_sigma_nt=3.5, showplot=False, verbose=False, repeat=False)
    # fpc.level32_flag_outliers_hampel_dtnt_test(window_length=48 * 3, n_sigma_dt=3.5, n_sigma_nt=3.5, showplot=True,
    #                                            verbose=True, repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_zscore_rolling_test(winsize=48 * 7, thres_zscore=4, showplot=False, verbose=True,
    #                                               repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_localsd_test(n_sd=5, winsize=48 * 13, constant_sd=False, showplot=False, verbose=True,
    #                                        repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_localsd_test(n_sd=3, winsize=48 * 3, constant_sd=True, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_increments_zcore_test(thres_zscore=4, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()
    # fpc.level32.showplot_cleaned()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_lof_dtnt_test(n_neighbors=48 * 5, contamination=None, showplot=True, verbose=True, repeat=True, n_jobs=-1)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_lof_test(n_neighbors=20, contamination=None, showplot=True, verbose=True,
    #                                    repeat=False, n_jobs=-1)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_zscore_test(thres_zscore=3, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()
    # fpc.level32.results

    # fpc.level32_flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False, verbose=True)
    # fpc.level32_addflag()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_trim_low_test(trim_nighttime=True, lower_limit=-10, showplot=True, verbose=True)
    # fpc.level32_addflag()

    fpc.finalize_level32()

    # # fpc.filteredseries
    # # fpc.level32.flags
    # fpc.level32_qcf.showplot_qcf_heatmaps()
    # # fpc.level32_qcf.showplot_qcf_timeseries()
    # # fpc.level32_qcf.report_qcf_flags()
    # fpc.level32_qcf.report_qcf_evolution()
    # # fpc.level32_qcf.report_qcf_series()
    # -------------------------------------------------------------------------

    # --------------------
    # Level-3.3
    # --------------------
    # 0.0532449, 0.0709217, 0.0949867
    # ustar_scenarios = ['NO_USTAR']
    # ustar_thresholds = [-9999]
    ustar_scenarios = ['CUT_50']
    ustar_thresholds = [0.0709217]
    # ustar_scenarios = ['CUT_50', 'CUT_84']
    # ustar_thresholds = [0.0709217, 0.0949867]
    # TODO check flag issue during finalizing
    # ustar_scenarios = ['CUT_16', 'CUT_50', 'CUT_84']
    # ustar_thresholds = [0.0532449, 0.0709217, 0.0949867]
    fpc.level33_constant_ustar(thresholds=ustar_thresholds,
                               threshold_labels=ustar_scenarios,
                               showplot=False)
    # Finalize: stores results for each USTAR scenario in a dict
    fpc.finalize_level33()

    # # Save current instance to pickle for faster testing
    # from diive.core.io.files import save_as_pickle
    # save_as_pickle(outpath=r"F:\TMP", filename="test", data=fpc)
    # fpc = load_pickle(filepath=r"F:\TMP\test.pickle")

    # for ustar_scenario in ustar_scenarios:
    #     fpc.level33_qcf[ustar_scenario].showplot_qcf_heatmaps()
    #     fpc.level33_qcf[ustar_scenario].report_qcf_evolution()
    #     # fpc.filteredseries
    #     # fpc.level33
    #     # fpc.level33_qcf.showplot_qcf_timeseries()
    #     # fpc.level33_qcf.report_qcf_flags()
    #     # fpc.level33_qcf.report_qcf_series()
    #     # fpc.levelidstr
    #     # fpc.filteredseries_level2_qcf
    #     # fpc.filteredseries_level31_qcf
    #     # fpc.filteredseries_level32_qcf
    #     # fpc.filteredseries_level33_qcf

    # --------------------
    # Level-4.1
    # --------------------

    fpc.level41_gapfilling_longterm(
        run_random_forest=True,
        run_mds=False,
        features=["TA_T1_2_1", "SW_IN_T1_2_1", "VPD_T1_2_1"],
        # features_lag=[-1, -1],
        reduce_features=False,
        include_timestamp_as_features=False,
        add_continuous_record_number=False,
        verbose=True,
        perm_n_repeats=2,
        rf_kwargs={
            'n_estimators': 3,
            'random_state': 42,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'n_jobs': -1
        }
    )

    results = fpc.get_data()
    gapfilled_names = fpc.get_gapfilled_names()
    nongapfilled_names = fpc.get_nongapfilled_names()
    gapfilled_vars = fpc.get_gapfilled_variables()
    fpc.report_gapfilling_variables()
    fpc.report_gapfilling_model_scores()
    fpc.report_gapfilling_poolyears()

    # todo get full data

    fpc.showplot_gapfilled_heatmap()
    fpc.showplot_gapfilled_cumulative(gain=0.02161926, units=r'($\mathrm{µmol\ CO_2\ m^{-2}}$)', per_year=True)
    fpc.showplot_gapfilled_cumulative(gain=0.02161926, units=r'($\mathrm{µmol\ CO_2\ m^{-2}}$)', per_year=False)
    fpc.showplot_feature_ranks_per_year()

    # TODO heatmap of used model data pools

    print("XXX")
    print("XXX")
    print("XXX")
    print("XXX")
    print("XXX")

    # newcols = [c for c in fpc.fpc_df.columns if c not in maindf.columns]
    # maindf2 = pd.concat([maindf, fpc.fpc_df[newcols]], axis=1)
    # # FLUXVAR33QCF = fpc.filteredseries_level33_qcf.name
    # # maindf2 = maindf2[[FLUXVAR33QCF, "TA_T1_2_1", "SW_IN_T1_2_1", "VPD_T1_2_1"]].copy()

    # # Collect data for each scenario for gapfilling
    # ust_scen = {}
    # for key, value in fpc.filteredseries_level33_qcf.items():
    #     ust_scen[value.name] = maindf2[[value.name, "TA_T1_2_1", "SW_IN_T1_2_1", "VPD_T1_2_1"]].copy()

    # import matplotlib.pyplot as plt
    # maindf2.plot(subplots=True, x_compat=True, title="After USTAR Threshold", figsize=(12, 4.5))
    # plt.show()

    # df2 = df2.dropna()
    # y = df2[FLUXVAR32QCF].copy()
    #
    # X = df2[["TA_T1_2_1", "SW_IN_T1_2_1", "VPD_T1_2_1"]].copy()
    #
    #
    # # https://www.youtube.com/watch?v=aLOQD66Sj0g
    # # https://github.com/liannewriting/YouTube-videos-public/blob/main/xgboost-python-tutorial-example/xgboost_python.ipynb
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train = X_train.to_numpy()
    # X_test = X_test.to_numpy()
    # y_train = y_train.to_numpy()
    # y_test = y_test.to_numpy()
    #
    # from sklearn.pipeline import Pipeline
    # from category_encoders.target_encoder import TargetEncoder
    # from xgboost import XGBRegressor
    # estimators = [
    #     ('encoder', TargetEncoder()),
    #     ('clf', XGBRegressor(random_state=42))  # can customize objective function with the objective parameter
    # ]
    # pipe = Pipeline(steps=estimators)
    # print(pipe)
    #
    # from skopt import BayesSearchCV
    # from skopt.space import Real, Categorical, Integer
    #
    # search_space = {
    #     'clf__max_depth': Integer(2, 8),
    #     'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    #     'clf__subsample': Real(0.5, 1.0),
    #     'clf__colsample_bytree': Real(0.5, 1.0),
    #     'clf__colsample_bylevel': Real(0.5, 1.0),
    #     'clf__colsample_bynode': Real(0.5, 1.0),
    #     'clf__reg_alpha': Real(0.0, 10.0),
    #     'clf__reg_lambda': Real(0.0, 10.0),
    #     'clf__gamma': Real(0.0, 10.0),
    #     'clf__n_estimators': Integer(2, 99)
    # }
    #
    # # opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=20, scoring='neg_mean_absolute_error', random_state=42)
    # opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=20, scoring='r2', random_state=42)
    # # opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=42)
    #
    # opt.fit(X_train, y_train)
    #
    # print(opt.best_estimator_)
    # print(opt.best_score_)
    # # print(opt.score(X_test, y_test))
    # # print(opt.predict(X_test))
    # # print(opt.predict_proba(X_test))

    # -----
    # GAP-FILLING
    # use_gapfilling = 1
    # N_ESTIMATORS = 99
    # gf = None
    # results = pd.DataFrame()
    #
    # for key, value in ust_scen.items():
    #
    #     if use_gapfilling == 1:
    #         # Random forest
    #         from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS
    #         MAX_DEPTH = None
    #         MIN_SAMPLES_SPLIT = 2
    #         MIN_SAMPLES_LEAF = 1
    #         CRITERION = 'squared_error'  # “squared_error”, “absolute_error”, “friedman_mse”, “poisson”
    #         gf = RandomForestTS(
    #             input_df=ust_scen[key],
    #             target_col=key,
    #             verbose=True,
    #             # features_lag=None,
    #             features_lag=[-1, -1],
    #             # features_lag_exclude_cols=['test', 'test2'],
    #             # include_timestamp_as_features=False,
    #             include_timestamp_as_features=True,
    #             # add_continuous_record_number=False,
    #             add_continuous_record_number=True,
    #             sanitize_timestamp=True,
    #             perm_n_repeats=3,
    #             n_estimators=N_ESTIMATORS,
    #             random_state=42,
    #             # random_state=None,
    #             max_depth=MAX_DEPTH,
    #             min_samples_split=MIN_SAMPLES_SPLIT,
    #             min_samples_leaf=MIN_SAMPLES_LEAF,
    #             criterion=CRITERION,
    #             test_size=0.2,
    #             n_jobs=-1
    #         )
    #
    #     elif use_gapfilling == 2:
    #         # XGBoost
    #         from diive.pkgs.gapfilling.xgboost_ts import XGBoostTS
    #         gf = XGBoostTS(
    #             input_df=maindf2,
    #             target_col=FLUXVAR33QCF,
    #             verbose=1,
    #             # features_lag=None,
    #             features_lag=[-1, -1],
    #             # features_lag_exclude_cols=['TIMESINCE_PREC_TOT_T1_25+20_1'],
    #             # features_lag_exclude_cols=['Rg_f', 'TA>0', 'TA>20', 'DAYTIME', 'NIGHTTIME'],
    #             # include_timestamp_as_features=False,
    #             include_timestamp_as_features=True,
    #             # add_continuous_record_number=False,
    #             add_continuous_record_number=True,
    #             sanitize_timestamp=True,
    #             perm_n_repeats=3,
    #             n_estimators=N_ESTIMATORS,
    #             random_state=42,
    #             # booster='gbtree',  # gbtree (default), gblinear, dart
    #             # device='cpu',
    #             # validate_parameters=True,
    #             # disable_default_eval_metric=False,
    #             early_stopping_rounds=10,
    #             # learning_rate=0.1,
    #             max_depth=6,
    #             # max_delta_step=0,
    #             # subsample=1,
    #             # min_split_loss=0,
    #             # min_child_weight=1,
    #             # colsample_bytree=1,
    #             # colsample_bylevel=1,
    #             # colsample_bynode=1,
    #             # reg_lambda=1,
    #             # reg_alpha=0,
    #             # tree_method='auto',  # auto, hist, approx, exact
    #             # scale_pos_weight=1,
    #             # grow_policy='depthwise',  # depthwise, lossguide
    #             # max_leaves=0,
    #             # max_bin=256,
    #             # num_parallel_tree=1,
    #             n_jobs=-1
    #         )
    #
    #     # c = gf.gapfilling_df_['NEE_L3.1_L3.3_QCF_gfXG'].copy()
    #
    #     # rfts.reduce_features()
    #     # rfts.report_feature_reduction()
    #     gf.trainmodel(showplot_scores=False, showplot_importance=False)
    #     # rfts.report_traintest()
    #     gf.fillgaps(showplot_scores=False, showplot_importance=False)
    #     gf.report_gapfilling()
    #
    #     # c = gf.gapfilling_df_['LE_L3.1_L3.3_QCF_gfRF'].copy()
    #     c = gf.gapfilling_df_[f"{key}_gfRF"].copy()
    #     results[c.name] = c.multiply(0.02161926).copy()  # Save NEE in g C m-2 30min-1
    #
    # print(f"Sum in gC:\n{results.sum()}")

    # import matplotlib.pyplot as plt
    # rfts.gapfilling_df_['.GAPFILLED_CUMULATIVE'].plot(x_compat=True)
    # rfts.gapfilling_df_['NEE_L3.1_L3.2_QCF_gfRF'].plot(x_compat=True)
    # plt.show()

    # c = c.multiply(0.02161926)
    # from diive.core.io.files import save_parquet, load_parquet
    # save_parquet(data=c.to_frame(), filename="test", outpath=r"F:\TMP")
    # c = load_parquet(filepath=r"F:\TMP\test.parquet")
    # c = c.loc[c.index.year < 2024].copy()

    # import matplotlib.pyplot as plt
    # for g in gapfilled_vars:
    #     gapfilled_vars[g].cumsum().plot(x_compat=True)
    # plt.legend()
    # plt.show()

    # from diive.core.plotting.cumulative import CumulativeYear
    # for g in gapfilled_vars:
    #     CumulativeYear(
    #         series=gapfilled_vars[g],
    #         series_units="X",
    #         yearly_end_date=None,
    #         # yearly_end_date='08-11',
    #         start_year=2006,
    #         end_year=2023,
    #         show_reference=True,
    #         # excl_years_from_reference=[2005, 2008, 2009, 2010, 2021, 2022, 2023],
    #         excl_years_from_reference=None,
    #         highlight_year=2023,
    #         highlight_year_color='#F44336').plot()
    # ---

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


def example_cumu():
    from diive.core.io.files import load_parquet
    c = load_parquet(filepath=r"F:\TMP\test.parquet")

    # from diive.core.io.files import save_parquet, load_parquet
    # save_parquet(data=c['NEE_L3.1_L3.2_QCF_gfRF'], filename="test", outpath=r"F:\TMP")

    from diive.core.plotting.cumulative import CumulativeYear
    CumulativeYear(
        series=c['NEE_L3.1_L3.2_QCF_gfRF'],
        series_units="X",
        yearly_end_date=None,
        # yearly_end_date='08-11',
        start_year=2018,
        end_year=2020,
        show_reference=True,
        excl_years_from_reference=None,
        # excl_years_from_reference=[2022],
        # highlight_year=2022,
        highlight_year_color='#F44336').plot()


if __name__ == '__main__':
    # example_quick()
    example()
    # example_cumu()
