"""
Implements the multi-step Swiss FluxNet-style processing chain for eddy covariance flux data.

This chain typically involves:
1. Level-1: Load flux data, including meteo variables (handled by LoadEddyProOutputFiles).
2. Level-2: Quality flag expansion (VM97, signal strength, etc.).
3. Level-3.1: Storage correction using a single-point profile.
4. Level-3.2: Stepwise Outlier detection.
5. Level-3.3: USTAR threshold filtering for low-turbulence periods.
6. Level-4.1: Gap-filling using methods like Random Forest or MDS.

The `FluxProcessingChain` class manages the data flow and state across these levels.
"""

from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.core.dfun.frames import detect_new_columns
from diive.core.funcs.funcs import filter_strings_by_elements
from diive.core.io.filereader import MultiDataFileReader, search_files
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.core.plotting.cumulative import Cumulative, CumulativeYear
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.pkgs.createvar.daynightflag import daytime_nighttime_flag_from_swinpot
from diive.pkgs.createvar.potentialradiation import potrad
from diive.pkgs.flux.common import detect_fluxbasevar
from diive.pkgs.flux.hqflux import analyze_highest_quality_flux
from diive.pkgs.flux.ustarthreshold import FlagMultipleConstantUstarThresholds
from diive.pkgs.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsEddyPro
from diive.pkgs.fluxprocessingchain.level31_storagecorrection import FluxStorageCorrectionSinglePointEddyPro
from diive.pkgs.gapfilling.longterm import LongTermGapFillingRandomForestTS, LongTermGapFillingXGBoostTS
from diive.pkgs.gapfilling.mds import FluxMDS
from diive.pkgs.outlierdetection.stepwiseoutlierdetection import StepwiseOutlierDetection
from diive.pkgs.qaqc.qcf import FlagQCF


class FluxProcessingChain:
    """
    Manages the multi-level processing chain for eddy covariance flux data.

    This class orchestrates the application of quality control (QC) tests,
    corrections (storage), filtering (USTAR), and final gap-filling, tracking
    the state and results at each level.

    Attributes:
        fluxcol (str): The name of the primary flux column (e.g., 'FC').
        site_lat (float): Latitude of the measurement site.
        site_lon (float): Longitude of the measurement site.
        utc_offset (int): UTC offset of the site (in hours).
        nighttime_threshold (float): Potential incoming shortwave radiation threshold (W/m²)
                                     to define nighttime periods.
        daytime_accept_qcf_below (int): QCF value (0, 1, or 2) below which daytime data are retained.
        nighttimetime_accept_qcf_below (int): QCF value (0, 1, or 2) below which nighttime data are retained.
        fluxbasevar (str): The base gas/variable name used for the flux (e.g., 'CO2').
        ustarcol (str): The column name for friction velocity (USTAR).
        outname (str): The output name for the final flux (e.g., 'NEE' for 'FC').
        """

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
        """
        Initializes the flux processing chain instance.

        Sets up metadata, detects the base flux variable, and adds necessary
        pre-calculated variables (potential radiation, day/night flags) to the
        internal dataframes.

        Args:
            df (DataFrame): Input DataFrame containing flux and meteorological data.
            fluxcol (str): Name of the raw flux column to process (e.g., 'FC', 'LE', 'H').
            site_lat (float): Site latitude.
            site_lon (float): Site longitude.
            utc_offset (int): UTC offset (in hours).
            nighttime_threshold (float, optional): Potential radiation threshold (W/m²)
                                                  for defining nighttime. Defaults to 20.
            daytime_accept_qcf_below (int, optional): QCF value (0, 1, or 2) below which daytime data are retained.
            nighttimetime_accept_qcf_below (int, optional): QCF value (0, 1, or 2) below which nighttime data are retained.
        """

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
        self._fpc_df, self.swinpot_col, daytime_flag_col, nighttime_flag_col = \
            self._add_swinpot_dt_nt_flag(df=self._fpc_df)

        # Add also to main data, so it can be accessed as feature for ML models
        newcols = [self.swinpot_col, daytime_flag_col, nighttime_flag_col]

        for c in newcols:
            existingcol = True if c in self.df.columns else False
            self._df[c] = self.fpc_df[c].copy()
            if existingcol:
                print(f"++ Added new column {c} to input data.  (!) Existing {c} in input data is overwritten.")
            else:
                print(f"++ Added new column {c} to input data.")
        # self._df = pd.concat([self._df, self.fpc_df[newcols]], axis=1)

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
        """
        Returns the full input DataFrame combined with all new results/flags from the processing chain.

        Args:
            verbose (int, optional): If > 0, prints a list of the new variables added
                                     by the processing chain. Defaults to 1.

        Returns:
            DataFrame: The combined DataFrame.
        """
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
        """
        Retrieves a list of gap-filled variable names.

        This method collects names of all gap-filled variables from the nested dictionary
        `level41`. It iterates over gap-filling methods and associated u* (ustar) scenarios to
        extract the gap-filled variable names and compiles them into a list.

        Returns:
            list: A list containing the names of all gap-filled variables.
        """
        gapfilled_vars = []
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                gapfilled_vars.append(self.level41[gfmethod][ustar_scenario].gapfilled_.name)
        return gapfilled_vars

    def get_nongapfilled_names(self) -> list:
        """
        Retrieves a list of nongapfilled variable names from the level41 attribute.

        This method iterates over the nested dictionary structure under the level41
        attribute and extracts the `target_col` attribute from each contained
        element. The method collects and returns these values as a list. The resulting
        list represents variable names that were not gap-filled, based on the provided
        structure of the level41 dictionary.

        Returns:
            list: A list of nongapfilled variable names extracted from the level41 attribute.
        """
        nongapfilled_vars = []
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                nongapfilled_vars.append(self.level41[gfmethod][ustar_scenario].target_col)
        # nongapfilled_vars = list(set(nongapfilled_vars))
        return nongapfilled_vars

    def report_gapfilling_variables(self):
        """
        Reports gap-filling variables and their mappings.

        This method iterates through the hierarchical structure of gap-filling methods and
        associated u* (ustar) scenarios within `level41`. For each combination of gap-filling
        method and u* scenario, it retrieves and displays the original (non-gap-filled) variable name
        and its corresponding gap-filled variable name in a formatted string.

        Args:
            None

        Returns:
            None
        """
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                gapfilled_name = self.level41[gfmethod][ustar_scenario].gapfilled_.name
                nongapfilled_name = self.level41[gfmethod][ustar_scenario].target_col
                print(f"{gfmethod} ({ustar_scenario}): {nongapfilled_name} -> {gapfilled_name}")

    def get_gapfilled_variables(self) -> DataFrame:
        """
        Fetches and returns the DataFrame containing both gap-filled and non-gap-filled variables.

        This method aggregates variable names marked as gap-filled and non-gap-filled,
        then filters the internal DataFrame to include only these variables while maintaining
        their original data.

        Returns:
            DataFrame: A new DataFrame containing columns corresponding to both gap-filled
                and non-gap-filled variable names.
        """
        gapfilled_vars = self.get_gapfilled_names()
        nongapfilled_vars = self.get_nongapfilled_names()
        gfvars = gapfilled_vars + nongapfilled_vars
        return self.fpc_df[gfvars].copy()

    def report_traintest_model_scores(self, outpath: str = None):
        """
        Reports and optionally saves train/test model scores for each u*-scenario associated
        with each gap-filling method, if available.

        This method iterates through all the u*-scenarios and checks if the required
        instance property ('scores_traintest_') exists for a given gap-filling method
        and u*-scenario combination. If the scores are available, it prints them in
        a structured tabular format using pandas. Optionally, the scores can also
        be saved to a file in CSV format within a specified output directory.

        Args:
            outpath (str, optional): Directory path where train/test model scores will be
                saved as CSV files if provided. If not specified, no files will be saved.

        Raises:
            ValueError: Raised internally when attempting to convert invalid score
                dictionaries into pandas DataFrames, specifically if the orientation
                needed for the conversion does not align appropriately.
        """
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                # Check if needed instance property exists
                instance = self.level41[gfmethod][ustar_scenario]
                if hasattr(instance, 'scores_traintest_'):
                    print(f"\nMODEL SCORES from TRAINING/TESTING ({gfmethod}): "
                          f"\n trained on training data, tested on unseen test data "
                          f"{ustar_scenario}")
                    try:
                        modelscores = pd.DataFrame.from_dict(self.level41[gfmethod][ustar_scenario].scores_traintest_,
                                                             orient='columns')
                    except ValueError:
                        modelscores = pd.DataFrame.from_dict(self.level41[gfmethod][ustar_scenario].scores_traintest_,
                                                             orient='index')
                    print(modelscores)
                    if outpath:
                        outfile = f"traintest_model_scores_{ustar_scenario}_{gfmethod}.csv"
                        outfilepath = Path(outpath) / outfile
                        modelscores.to_csv(outfilepath)
                else:
                    print(f"{gfmethod} {ustar_scenario} does not have train/test model scores.")

    def report_traintest_details(self, outpath: str = None):
        """
        Reports the training and testing details for a specific instance in the `level41` dictionary. The
        method prints the details to the console and optionally saves them to a CSV file if an output path
        is provided.

        Args:
            outpath (str, optional): The file path where the training and testing details will be saved in a
                CSV format. Defaults to None.
        """
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                # Check if needed instance property exists
                instance = self.level41[gfmethod][ustar_scenario]
                if hasattr(instance, 'traintest_details_'):
                    print(f"\nDETAILS from TRAINING/TESTING ({gfmethod} {ustar_scenario}): "
                          f"\n trained on training data, tested on unseen test data")
                    try:
                        modelscores = pd.DataFrame.from_dict(self.level41[gfmethod][ustar_scenario].traintest_details_,
                                                             orient='columns')
                    except ValueError:
                        modelscores = pd.DataFrame.from_dict(self.level41[gfmethod][ustar_scenario].traintest_details_,
                                                             orient='index')
                    print(modelscores)
                    if outpath:
                        outfile = f"traintest_model_details_{ustar_scenario}_{gfmethod}.csv"
                        outfilepath = Path(outpath) / outfile
                        modelscores.to_csv(outfilepath)
                else:
                    print(f"{gfmethod} {ustar_scenario} does not have train/test details.")

    #     def showplot_feature_ranks_per_year(self):
    #         for gfmethod, ustar_scenarios in self.level41.items():
    #             for ustar_scenario in ustar_scenarios:
    #                 try:
    #                     results = self.level41[gfmethod][ustar_scenario]
    #                     title = f"{results.gapfilled_.name} ({ustar_scenario})"
    #                     first_key = next(iter(self.level41[gfmethod][ustar_scenario].results_yearly_))
    #                     model_params = self.level41[gfmethod][ustar_scenario].results_yearly_[
    #                         first_key].model_.get_params()
    #                     txt = f"MODEL: {gfmethod} / PARAMS: {model_params}"
    #                     results.showplot_feature_ranks_per_year(title=f"{title}", subtitle=f"{txt}")
    #                 except AttributeError:
    #                     print(f"{gfmethod} {ustar_scenario} does not have feature ranks.")

    def report_gapfilling_model_scores(self, outpath: str = None):
        """
        Generates a report containing model scores for gap-filling methods and ustar scenarios.
        Iterates over gap-filling methods and their associated ustar scenarios,
        extracting and formatting model score data, then printing or saving the
        results as a CSV file if an output path is provided.

        Args:
            outpath (str, optional): Directory path where CSV files with model scores
                will be saved. If None, the scores are only printed to the console.
        """
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                print(f"\nMODEL SCORES ({gfmethod}): {ustar_scenario}")
                try:
                    modelscores = pd.DataFrame.from_dict(self.level41[gfmethod][ustar_scenario].scores_,
                                                         orient='columns')
                except ValueError:
                    modelscores = pd.DataFrame.from_dict(self.level41[gfmethod][ustar_scenario].scores_,
                                                         orient='index')
                print(modelscores)
                if outpath:
                    outfile = f"gapfilling_model_scores_{ustar_scenario}_{gfmethod}.csv"
                    outfilepath = Path(outpath) / outfile
                    modelscores.to_csv(outfilepath)

    def report_gapfilling_feature_importances(self, outpath: str = None):
        """
        Reports the feature importances for gapfilling models and saves them to the specified
        output path if provided.

        This method iterates over the gapfilling methods and U* scenarios stored in the
        `level41` attribute. For each combination, it checks for the presence of the
        `feature_importance_per_year` attribute within the corresponding instance. If the
        attribute exists, it prints the feature importances and optionally saves them as a
        CSV file to the provided output path.

        Args:
            outpath (str, optional): The path where the feature importance CSV files will
                be saved. If `None`, the files are not saved but the results are printed
                to the console.
        """
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                # Check if needed instance property exists
                instance = self.level41[gfmethod][ustar_scenario]
                if hasattr(instance, 'feature_importance_per_year'):
                    print(f"\nFEATURE IMPORTANCES ({gfmethod}): {ustar_scenario}")
                    _df = self.level41[gfmethod][ustar_scenario].feature_importance_per_year
                    print(_df)
                    if outpath:
                        outfile = f"gapfilling_model_feature_importances_{ustar_scenario}_{gfmethod}.csv"
                        outfilepath = Path(outpath) / outfile
                        _df.to_csv(outfilepath)
                else:
                    print(f"{gfmethod} {ustar_scenario} does not have feature importances.")

    def report_gapfilling_poolyears(self):
        """
        Reports the data pools used for machine-learning-based gap-filling for each year and their associated
        scenarios and methods. Iterates through the provided levels and prints metadata regarding the years
        of data used, the target variable being gap-filled, and the resulting gap-filled data column.

        Raises:
            AttributeError: If the required attributes are missing in the data structure during iteration.
        """
        print("DATA POOLS USED FOR MACHINE-LEARNING GAP-FILLING:")
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                try:
                    for yr, pool in self.level41[gfmethod][ustar_scenario].yearpools.items():
                        print(f"{yr}: {gfmethod} used data from {pool['poolyears']} "
                              f"for gap-filling {self.level41[gfmethod][ustar_scenario].target_col} and "
                              f"producing --> {self.level41[gfmethod][ustar_scenario].gapfilled_.name}")
                except AttributeError:
                    print(f"{gfmethod} {ustar_scenario} did not use poolyears.")

    def showplot_feature_ranks_per_year(self):
        """
        Displays feature ranks per year for each scenario and method in the given dataset.

        This method iterates over the gap-filling methods and u* scenarios from a nested data
        structure (`level41`). For each combination, it retrieves results, constructs a title
        and subtitle containing model information, and calls a plotting function to visualize
        feature ranks per year. If feature ranks are unavailable for a particular case,
        it catches the exception and prints an informative message instead of failing.

        Raises:
            AttributeError: Raised when feature rank data or a required intermediate attribute
                does not exist for a specific combination of method and scenario.
        """
        for gfmethod, ustar_scenarios in self.level41.items():
            for ustar_scenario in ustar_scenarios:
                try:
                    results = self.level41[gfmethod][ustar_scenario]
                    title = f"{results.gapfilled_.name} ({ustar_scenario})"
                    first_key = next(iter(self.level41[gfmethod][ustar_scenario].results_yearly_))
                    model_params = self.level41[gfmethod][ustar_scenario].results_yearly_[
                        first_key].model_.get_params()
                    txt = f"MODEL: {gfmethod} / PARAMS: {model_params}"
                    results.showplot_feature_ranks_per_year(title=f"{title}", subtitle=f"{txt}")
                except AttributeError:
                    print(f"{gfmethod} {ustar_scenario} does not have feature ranks.")

    def showplot_mds_gapfilling_qualities(self):
        """
        Displays the plot of gap filling qualities using the MDS method for available
        ustar scenarios.

        This function iterates through the `level41` attribute, filtering for the
        'MDS' (Marginal Data Substitution) gap filling method. It subsequently
        invokes the `showplot` method for each ustar scenario within this method,
        displaying the associated plots.

        Raises:
            AttributeError: If the `results` object for a specific ustar scenario
            does not have a `showplot` method defined.
        """
        for gfmethod, ustar_scenarios in self.level41.items():
            if gfmethod != 'mds':
                continue
            for ustar_scenario in ustar_scenarios:
                # try:
                results = self.level41[gfmethod][ustar_scenario]
                # txt = f"MODEL: {gfmethod}"
                results.showplot()
                # except AttributeError:
                #     print(f"{gfmethod} {ustar_scenario} does not have feature ranks.")

    def showplot_gapfilled_heatmap(self, vmin: float = None, vmax: float = None):
        """
        Displays a heatmap visualization for gap-filled and non-gap-filled variables.

        This method generates and displays heatmaps for gap-filled and corresponding
        non-gap-filled data. For each pair of gap-filled and non-gap-filled variables
        identified, the method creates a figure containing two heatmaps side-by-side:
        one for the non-gap-filled data and one for the gap-filled data. If the
        non-gap-filled data is stored as a DataFrame, it is converted to a Series
        before plotting.

        Args:
            vmin (float, optional): Minimum value for the heatmap color scale. If not
                provided, the minimum value is inferred from the data.
            vmax (float, optional): Maximum value for the heatmap color scale. If not
                provided, the maximum value is inferred from the data.
        """
        gapfilled_vars = self.get_gapfilled_names()
        nongapfilled_vars = self.get_nongapfilled_names()
        gfvars = self.get_gapfilled_variables()
        for ix, g in enumerate(gapfilled_vars):
            fig = plt.figure(figsize=(12, 12), dpi=100)
            gs = gridspec.GridSpec(1, 2)  # rows, cols
            # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
            ax_ngf = fig.add_subplot(gs[0, 0])
            ax_gf = fig.add_subplot(gs[0, 1])

            # Non-gapfilled data must be a Series, but if the gap-filled time series
            # have the same data basis then it is a DataFrame. In that case, the dataframe
            # is converted to a series.
            series_ngf = gfvars[nongapfilled_vars[ix]].copy()
            if isinstance(series_ngf, pd.DataFrame):
                series_ngf = series_ngf.loc[:, ~series_ngf.columns.duplicated()].copy()
                series_ngf = series_ngf.squeeze()

            hm = HeatmapDateTime(ax=ax_ngf, series=series_ngf, vmin=vmin, vmax=vmax).plot()
            hm = HeatmapDateTime(ax=ax_gf, series=gfvars[g], vmin=vmin, vmax=vmax).plot()
            fig.tight_layout()
            fig.show()

    def showplot_gapfilled_cumulative(self, gain: float = 1, units: str = "", per_year: bool = True,
                                      start_year: int = None, end_year: int = None,
                                      show_reference: bool = True, excl_years_from_reference: list = None,):
        """
        Plots cumulative data for gap-filled variables, either as a yearly cumulative plot or
        overall cumulative plot. This method processes gap-filled variables, scales them by the
        `gain` factor, and plots them individually per year or as a whole, based on the `per_year`
        indicator.

        Args:
            gain (float): Scaling factor for the gap-filled variables.
            units (str): Units of the scaled values to be displayed in the plot.
            per_year (bool): Indicator for yearly cumulative plot. If True, plots cumulatively
                for each year; if False, plots overall cumulative data.
            start_year (int): Optional starting year for the cumulative plot. Defaults to None.
            end_year (int): Optional ending year for the cumulative plot. Defaults to None.
        """
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
                    show_reference=show_reference,
                    excl_years_from_reference=excl_years_from_reference,
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

    def _add_swinpot_dt_nt_flag(self, df: DataFrame) -> tuple[DataFrame, str, str, str]:
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

        return df, swinpot_col, daytime_flag_col, nighttime_flag_col

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
            self._level2.ssitc_test(setflag_timeperiod=ssitc['setflag_timeperiod'])

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

    def level41_mds(
            self,
            swin: str = None,
            ta: str = None,
            vpd: str = None,
            swin_tol: list = None,  # Default defined below: [20, 50]
            ta_tol: float = 2.5,
            vpd_tol: float = 0.5,
            avg_min_n_vals: int = 5,
    ):

        idstr = 'L4.1'
        if idstr not in self._levelidstr:
            self._levelidstr.append(idstr)

        # Store results in separate dict within Level-4.1
        self._level41['mds'] = dict()

        # Collect data and run model for each USTAR scenario for gapfilling
        # todo what is if there is no scenario eg h2o fluxes?
        for ustar_scen, ustar_flux in self.filteredseries_level33_qcf.items():
            # Get features from input data
            mdscols = [swin, ta, vpd]
            this_ust_scen_df = self.df[mdscols].copy()

            # Add USTAR-filtered flux from recent results (Level-3.3)
            this_ust_scen_df = pd.concat([this_ust_scen_df, self.fpc_df[ustar_flux.name]], axis=1)

            instance = FluxMDS(
                df=this_ust_scen_df,
                flux=ustar_flux.name,
                swin=swin,
                ta=ta,
                vpd=vpd,
                swin_tol=swin_tol,
                ta_tol=ta_tol,
                vpd_tol=vpd_tol,
                avg_min_n_vals=avg_min_n_vals
            )
            instance.run()

            # Add gap-filled flux and gap-filling flag to flux processing chain dataframe
            fluxdata = instance.get_gapfilled_target()
            # flagname = [c for c in instance.gapfilling_df_
            #             if str(c).startswith("FLAG_") and str(c).endswith("_ISFILLED")]
            # flagname = flagname[0] if len(flagname) == 1 else None
            flagdata = instance.get_flag()
            self._fpc_df = pd.concat([self.fpc_df, fluxdata, flagdata], axis=1)

            # Save instance to Level-4.1
            self._level41['mds'][ustar_scen] = instance

            # instance.report()
            # instance.showplot()

    def level41_longterm_random_forest(
            self,
            features: list = None,
            reduce_features: bool = False,
            verbose: int = 0,
            features_lag: list = None,
            features_lag_stepsize: int = 1,
            features_lag_exclude_cols: list = None,
            features_rolling: list = None,
            features_rolling_exclude_cols: list = None,
            features_rolling_stats: list = None,
            features_diff: list = None,
            features_diff_exclude_cols: list = None,
            features_ema: list = None,
            features_ema_exclude_cols: list = None,
            features_poly_degree: int = None,
            features_poly_exclude_cols: list = None,
            features_stl: bool = False,
            features_stl_method: str = 'stl',
            features_stl_seasonal_period: int = None,
            features_stl_exclude_cols: list = None,
            features_stl_components: list = None,
            vectorize_timestamps: bool = False,
            add_continuous_record_number: bool = False,
            sanitize_timestamp: bool = False,
            **rf_kwargs
    ):
        """
        Level-4.1 Gap-filling using long-term Random Forest with feature engineering (v0.91.0).

        This method implements multi-year gap-filling using Random Forest with standalone
        FeatureEngineer for composable feature engineering. Features are engineered once
        and reused across all USTAR scenarios.

        Workflow:
            1. Create FeatureEngineer with feature engineering parameters
            2. For each USTAR scenario:
               a. Engineer features from input data
               b. Create LongTermGapFillingRandomForestTS with pre-engineered data
               c. Train yearly pooled models
               d. Fill gaps and store results

        Args:
            features (list): Column names to use as feature inputs (gap-filling inputs).
            reduce_features (bool): Whether to apply SHAP-based feature selection across all years.
            verbose (int): Verbosity level (0=silent, 1+=progress).

            **Feature Engineering Parameters (passed to FeatureEngineer):**
            features_lag (list): [min_lag, max_lag] range for lag features.
            features_lag_stepsize (int): Step size for lag generation (default 1).
            features_lag_exclude_cols (list): Columns to exclude from lagging.
            features_rolling (list): Window sizes for rolling statistics (e.g., [12, 24]).
            features_rolling_exclude_cols (list): Columns to exclude from rolling.
            features_rolling_stats (list): Advanced rolling stats (['median', 'min', 'max', 'std', 'q25', 'q75']).
            features_diff (list): Difference orders [1, 2, ...] for temporal differencing.
            features_diff_exclude_cols (list): Columns to exclude from differencing.
            features_ema (list): EMA spans [6, 24, ...] for exponential moving averages.
            features_ema_exclude_cols (list): Columns to exclude from EMA.
            features_poly_degree (int): Polynomial degree (2, 3, ...) for non-linear terms.
            features_poly_exclude_cols (list): Columns to exclude from polynomial.
            features_stl (bool): Enable STL (Seasonal-Trend Loess) decomposition.
            features_stl_method (str): STL method ('stl', 'classical', 'harmonic').
            features_stl_seasonal_period (int): Seasonal period for STL decomposition.
            features_stl_exclude_cols (list): Columns to exclude from STL.
            features_stl_components (list): Components to extract (['trend', 'seasonal', 'residual']).
            vectorize_timestamps (bool): Add timestamp features (year, month, hour, etc.).
            add_continuous_record_number (bool): Add sequential record numbering for trend.
            sanitize_timestamp (bool): Validate and prepare timestamps.

            **Random Forest Hyperparameters (passed to sklearn RandomForestRegressor):**
            **rf_kwargs: n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state, n_jobs, etc.
                Default: n_estimators=200, random_state=42, min_samples_split=2, min_samples_leaf=1, n_jobs=-1

        Returns:
            None. Results stored in self._level41['long_term_random_forest'] dict keyed by USTAR scenario.

        Example:
            fpc.level41_longterm_random_forest(
                features=['TA', 'SW_IN', 'VPD'],
                features_lag=[-1, 1],
                features_rolling=[12, 24],
                features_rolling_stats=['median', 'min', 'max'],
                features_diff=[1],
                features_ema=[6, 24],
                features_poly_degree=2,
                vectorize_timestamps=True,
                add_continuous_record_number=True,
                reduce_features=False,
                verbose=1,
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        """

        idstr = 'L4.1'
        if idstr not in self._levelidstr:
            self._levelidstr.append(idstr)

        # Store results in separate dict within Level-4.1
        self._level41['long_term_random_forest'] = dict()

        # Default parameters for random forest models
        if not isinstance(rf_kwargs, dict):
            rf_kwargs = {'n_estimators': 200,
                         'random_state': 42,
                         'min_samples_split': 2,
                         'min_samples_leaf': 1,
                         'n_jobs': -1}

        # Step 1: Engineer features from input data
        engineer = FeatureEngineer(
            target_col='_temp_target_placeholder_',  # Temporary, will be replaced per scenario
            features_lag=features_lag,
            features_lag_stepsize=features_lag_stepsize,
            features_lag_exclude_cols=features_lag_exclude_cols,
            features_rolling=features_rolling,
            features_rolling_exclude_cols=features_rolling_exclude_cols,
            features_rolling_stats=features_rolling_stats,
            features_diff=features_diff,
            features_diff_exclude_cols=features_diff_exclude_cols,
            features_ema=features_ema,
            features_ema_exclude_cols=features_ema_exclude_cols,
            features_poly_degree=features_poly_degree,
            features_poly_exclude_cols=features_poly_exclude_cols,
            features_stl=features_stl,
            features_stl_method=features_stl_method,
            features_stl_seasonal_period=features_stl_seasonal_period,
            features_stl_exclude_cols=features_stl_exclude_cols,
            features_stl_components=features_stl_components,
            vectorize_timestamps=vectorize_timestamps,
            add_continuous_record_number=add_continuous_record_number,
            sanitize_timestamp=sanitize_timestamp
        )

        # Step 2: Collect data and run model for each USTAR scenario for gapfilling
        for ustar_scen, ustar_flux in self.filteredseries_level33_qcf.items():

            # Get features from input data and engineer them
            this_ust_scen_df = self.df[features].copy()

            # Engineer features on feature columns
            feature_engineer_input = engineer.fit_transform(this_ust_scen_df)

            # Add USTAR-filtered flux from recent results (Level-3.3)
            this_ust_scen_df = pd.concat([feature_engineer_input, self.fpc_df[ustar_flux.name]], axis=1)
            this_ust_scen_df = this_ust_scen_df.copy()

            general_kwargs = dict(
                input_df=this_ust_scen_df,
                target_col=ustar_flux.name,
                verbose=verbose
            )

            # Step 3: Initialize random forest for this scenario with pre-engineered data
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
            self._level41['long_term_random_forest'][ustar_scen] = instance

    def level41_longterm_xgboost(
            self,
            features: list = None,
            reduce_features: bool = False,
            verbose: int = 0,
            features_lag: list = None,
            features_lag_stepsize: int = 1,
            features_lag_exclude_cols: list = None,
            features_rolling: list = None,
            features_rolling_exclude_cols: list = None,
            features_rolling_stats: list = None,
            features_diff: list = None,
            features_diff_exclude_cols: list = None,
            features_ema: list = None,
            features_ema_exclude_cols: list = None,
            features_poly_degree: int = None,
            features_poly_exclude_cols: list = None,
            features_stl: bool = False,
            features_stl_method: str = 'stl',
            features_stl_seasonal_period: int = None,
            features_stl_exclude_cols: list = None,
            features_stl_components: list = None,
            vectorize_timestamps: bool = False,
            add_continuous_record_number: bool = False,
            sanitize_timestamp: bool = False,
            **xgb_kwargs
    ):
        """
        Level-4.1 Gap-filling using long-term XGBoost with feature engineering (v0.91.0).

        This method implements multi-year gap-filling using XGBoost gradient boosting with
        standalone FeatureEngineer for composable feature engineering. Features are engineered
        once and reused across all USTAR scenarios. XGBoost often outperforms Random Forest
        on non-linear patterns and is useful for comparison.

        Workflow:
            1. Create FeatureEngineer with feature engineering parameters
            2. For each USTAR scenario:
               a. Engineer features from input data
               b. Create LongTermGapFillingXGBoostTS with pre-engineered data
               c. Train yearly pooled models with early stopping
               d. Fill gaps and store results

        Args:
            features (list): Column names to use as feature inputs (gap-filling inputs).
            reduce_features (bool): Whether to apply SHAP-based feature selection across all years.
            verbose (int): Verbosity level (0=silent, 1+=progress).

            **Feature Engineering Parameters (passed to FeatureEngineer):**
            features_lag (list): [min_lag, max_lag] range for lag features.
            features_lag_stepsize (int): Step size for lag generation (default 1).
            features_lag_exclude_cols (list): Columns to exclude from lagging.
            features_rolling (list): Window sizes for rolling statistics (e.g., [12, 24]).
            features_rolling_exclude_cols (list): Columns to exclude from rolling.
            features_rolling_stats (list): Advanced rolling stats (['median', 'min', 'max', 'std', 'q25', 'q75']).
            features_diff (list): Difference orders [1, 2, ...] for temporal differencing.
            features_diff_exclude_cols (list): Columns to exclude from differencing.
            features_ema (list): EMA spans [6, 24, ...] for exponential moving averages.
            features_ema_exclude_cols (list): Columns to exclude from EMA.
            features_poly_degree (int): Polynomial degree (2, 3, ...) for non-linear terms.
            features_poly_exclude_cols (list): Columns to exclude from polynomial.
            features_stl (bool): Enable STL (Seasonal-Trend Loess) decomposition.
            features_stl_method (str): STL method ('stl', 'classical', 'harmonic').
            features_stl_seasonal_period (int): Seasonal period for STL decomposition.
            features_stl_exclude_cols (list): Columns to exclude from STL.
            features_stl_components (list): Components to extract (['trend', 'seasonal', 'residual']).
            vectorize_timestamps (bool): Add timestamp features (year, month, hour, etc.).
            add_continuous_record_number (bool): Add sequential record numbering for trend.
            sanitize_timestamp (bool): Validate and prepare timestamps.

            **XGBoost Hyperparameters (passed to XGBRegressor):**
            **xgb_kwargs: n_estimators, max_depth, learning_rate, early_stopping_rounds, subsample,
                colsample_bytree, random_state, n_jobs, etc.
                Default: n_estimators=200, random_state=42, max_depth=6, learning_rate=0.3,
                early_stopping_rounds=10, n_jobs=-1

        Returns:
            None. Results stored in self._level41['long_term_xgboost'] dict keyed by USTAR scenario.

        Example:
            fpc.level41_longterm_xgboost(
                features=['TA', 'SW_IN', 'VPD'],
                features_lag=[-1, 1],
                features_rolling=[12, 24],
                features_rolling_stats=['median', 'min', 'max'],
                features_diff=[1],
                features_ema=[6, 24],
                features_poly_degree=2,
                vectorize_timestamps=True,
                add_continuous_record_number=True,
                reduce_features=False,
                verbose=1,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.3,
                early_stopping_rounds=10,
                random_state=42,
                n_jobs=-1
            )
        """

        idstr = 'L4.1'
        if idstr not in self._levelidstr:
            self._levelidstr.append(idstr)

        # Store results in separate dict within Level-4.1
        self._level41['long_term_xgboost'] = dict()

        # Default parameters for XGBoost models
        if not isinstance(xgb_kwargs, dict):
            xgb_kwargs = {'n_estimators': 200,
                          'random_state': 42,
                          'early_stopping_rounds': 10,
                          'max_depth': 6,
                          'learning_rate': 0.3,
                          'n_jobs': -1}

        # Step 1: Engineer features from input data
        engineer = FeatureEngineer(
            target_col='_temp_target_placeholder_',  # Temporary, will be replaced per scenario
            features_lag=features_lag,
            features_lag_stepsize=features_lag_stepsize,
            features_lag_exclude_cols=features_lag_exclude_cols,
            features_rolling=features_rolling,
            features_rolling_exclude_cols=features_rolling_exclude_cols,
            features_rolling_stats=features_rolling_stats,
            features_diff=features_diff,
            features_diff_exclude_cols=features_diff_exclude_cols,
            features_ema=features_ema,
            features_ema_exclude_cols=features_ema_exclude_cols,
            features_poly_degree=features_poly_degree,
            features_poly_exclude_cols=features_poly_exclude_cols,
            features_stl=features_stl,
            features_stl_method=features_stl_method,
            features_stl_seasonal_period=features_stl_seasonal_period,
            features_stl_exclude_cols=features_stl_exclude_cols,
            features_stl_components=features_stl_components,
            vectorize_timestamps=vectorize_timestamps,
            add_continuous_record_number=add_continuous_record_number,
            sanitize_timestamp=sanitize_timestamp
        )

        # Step 2: Collect data and run model for each USTAR scenario for gapfilling
        for ustar_scen, ustar_flux in self.filteredseries_level33_qcf.items():

            # Get features from input data and engineer them
            this_ust_scen_df = self.df[features].copy()

            # Engineer features on feature columns
            feature_engineer_input = engineer.fit_transform(this_ust_scen_df)

            # Add USTAR-filtered flux from recent results (Level-3.3)
            this_ust_scen_df = pd.concat([feature_engineer_input, self.fpc_df[ustar_flux.name]], axis=1)
            this_ust_scen_df = this_ust_scen_df.copy()

            general_kwargs = dict(
                input_df=this_ust_scen_df,
                target_col=ustar_flux.name,
                verbose=verbose
            )

            # Step 3: Initialize XGBoost for this scenario with pre-engineered data
            instance = LongTermGapFillingXGBoostTS(**general_kwargs, **xgb_kwargs)

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
            self._level41['long_term_xgboost'][ustar_scen] = instance

    def level31_storage_correction(self, gapfill_storage_term: bool = True, set_storage_to_zero: bool = False):
        """Correct flux with storage term from single point measurement."""
        idstr = 'L3.1'
        self._levelidstr.append(idstr)
        self._level31 = FluxStorageCorrectionSinglePointEddyPro(df=self.df,
                                                                fluxcol=self.fluxcol,
                                                                basevar=self.fluxbasevar,
                                                                gapfill_storage_term=gapfill_storage_term,
                                                                idstr=idstr,
                                                                set_storage_to_zero=set_storage_to_zero)
        self._level31.storage_correction()

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

    def finalize_level31(self):
        newcols = detect_new_columns(df=self.level31.results, other=self.fpc_df)
        self._fpc_df = pd.concat([self.fpc_df, self.level31.results[newcols]], axis=1)
        [print(f"++Added new column {col}.") for col in newcols]

        # Apply QCF also to Level-3.1 flux output
        self._apply_level2_qcf_to_level31_flux()

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

    def level32_flag_outliers_localsd_test(self, n_sd: float | list = 7, winsize: int | list = None,
                                           showplot: bool = False, constant_sd: bool = False,
                                           separate_daytime_nighttime: bool = False,
                                           verbose: bool = False, repeat: bool = True):
        self._level32.flag_outliers_localsd_test(n_sd=n_sd, winsize=winsize,
                                                 separate_daytime_nighttime=separate_daytime_nighttime,
                                                 constant_sd=constant_sd, showplot=showplot,
                                                 verbose=verbose, repeat=repeat)

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

    def level32_flag_outliers_hampel_dtnt_test(self, window_length: int = 13, n_sigma_dt: float = 5.5,
                                               n_sigma_nt: float = 5.5,
                                               k: float = 1.4826, use_differencing: bool = True,
                                               separate_day_night: bool = True, showplot: bool = False,
                                               verbose: bool = False, repeat: bool = True):
        self._level32.flag_outliers_hampel_dtnt_test(window_length=window_length, n_sigma_dt=n_sigma_dt,
                                                     n_sigma_nt=n_sigma_nt, k=k, use_differencing=use_differencing,
                                                     separate_day_night=separate_day_night,
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

    def __init__(
            self,
            fluxvars: list,
            sourcedirs: list,
            site_lat: float,
            site_lon: float,
            utc_offset: int,
            nighttime_threshold: int = 20,
            daytime_accept_qcf_below: int = 2,
            nighttimetime_accept_qcf_below: int = 2,
            test_signal_strength=False,
            test_signal_strength_col='',
            test_signal_strength_method='discard above',
            test_signal_strength_threshold=999
    ):
        self.fluxvars = fluxvars
        self.sourcedirs = sourcedirs
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.utc_offset = utc_offset
        self.nighttime_threshold = nighttime_threshold
        self.daytime_accept_qcf_below = daytime_accept_qcf_below
        self.nighttimetime_accept_qcf_below = nighttimetime_accept_qcf_below

        # Tests that are not always available
        self.test_signal_strength = test_signal_strength
        self.test_signal_strength_col = test_signal_strength_col
        self.test_signal_strength_method = test_signal_strength_method
        self.test_signal_strength_threshold = test_signal_strength_threshold

        self.fpc = None

        self._run()

    def _run(self):
        self.maindf, self.metadata = self._load_data()

        for fluxcol in self.fluxvars:
            self.fpc = self._start_fpc(fluxcol=fluxcol)
            self._run_level2()
            self._run_level31()
            self._run_level32()
            self._run_level33(fluxcol=fluxcol)

    def _run_level33(self, fluxcol):
        if fluxcol in ['H', 'LE']:
            thresholds = [0]
            ustar_scenarios = ['CUT_NONE']
        else:
            thresholds = [0.08]
            ustar_scenarios = ['CUT_PRELIM']
        self.fpc.level33_constant_ustar(thresholds=thresholds,
                                        threshold_labels=ustar_scenarios,
                                        showplot=False)
        self.fpc.finalize_level33()

        for ustar_scenario in ustar_scenarios:
            self.fpc.level33_qcf[ustar_scenario].showplot_qcf_heatmaps()
            self.fpc.level33_qcf[ustar_scenario].report_qcf_evolution()
            # fpc.filteredseries
            # fpc.level33
            # self.fpc.level33_qcf[ustar_scenario].showplot_qcf_timeseries()
            # self.fpc.level33_qcf[ustar_scenario].report_qcf_flags()
            # self.fpc.level33_qcf[ustar_scenario].report_qcf_series()
            # fpc.levelidstr
            # fpc.filteredseries_level2_qcf
            # fpc.filteredseries_level31_qcf
            # fpc.filteredseries_level32_qcf
            # fpc.filteredseries_level33_qcf

    def _run_level32(self):
        self.fpc.level32_stepwise_outlier_detection()
        self.fpc.level32_flag_outliers_zscore_dtnt_test(thres_zscore=4, showplot=False, verbose=True, repeat=True)
        self.fpc.level32_addflag()
        self.fpc.finalize_level32()

        # self.fpc.filteredseries
        # self.fpc.level32.results
        # self.fpc.level32_qcf.showplot_qcf_heatmaps()
        # self.fpc.level32_qcf.showplot_qcf_timeseries()
        # self.fpc.level32_qcf.report_qcf_flags()
        # self.fpc.level32_qcf.report_qcf_evolution()
        # self.fpc.level32_qcf.report_qcf_series()

    def _run_level31(self):
        self.fpc.level31_storage_correction(gapfill_storage_term=True)
        self.fpc.finalize_level31()
        # fpc.level31.showplot(maxflux=50)
        # self.fpc.level31.report()

    def _run_level2(self):
        TEST_SSITC = True  # Default True
        TEST_SSITC_SETFLAG_TIMEPERIOD = None
        TEST_GAS_COMPLETENESS = True  # Default True
        TEST_SPECTRAL_CORRECTION_FACTOR = True  # Default True
        # TEST_SIGNAL_STRENGTH = True
        # TEST_SIGNAL_STRENGTH_COL = 'CUSTOM_AGC_MEAN'
        # TEST_SIGNAL_STRENGTH_METHOD = 'discard above'
        # TEST_SIGNAL_STRENGTH_THRESHOLD = 90
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
                'apply': self.test_signal_strength,
                'signal_strength_col': self.test_signal_strength_col,
                'method': self.test_signal_strength_method,
                'threshold': self.test_signal_strength_threshold},
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
                'apply': TEST_SSITC,
                'setflag_timeperiod': TEST_SSITC_SETFLAG_TIMEPERIOD},
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
            utc_offset=self.utc_offset,
            daytime_accept_qcf_below=self.daytime_accept_qcf_below,
            nighttimetime_accept_qcf_below=self.nighttimetime_accept_qcf_below
        )
        return fpc

    def _load_data(self):
        ep = LoadEddyProOutputFiles(sourcedir=self.sourcedirs, filetype='EDDYPRO-FLUXNET-CSV-30MIN')
        ep.searchfiles()
        ep.loadfiles()
        return ep.maindf, ep.metadata


def _example_quick():
    QuickFluxProcessingChain(
        # fluxvars=['FC'],
        fluxvars=['FC', 'LE', 'H'],
        sourcedirs=[r'F:\TMP\HON'],
        site_lat=47.115833,
        site_lon=8.537778,
        utc_offset=1,
        nighttime_threshold=20,
        daytime_accept_qcf_below=2,
        nighttimetime_accept_qcf_below=2,
        test_signal_strength=True,
        test_signal_strength_col='CUSTOM_AGC_MEAN',
        test_signal_strength_method='discard above',
        test_signal_strength_threshold=90,
    )


def _example():
    # Source data
    from pathlib import Path

    SOURCEDIR = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-hon_flux_product\dataset_ch-hon_flux_product\notebooks\0_data\OPENLAG-IRGA-Level-0_fluxnet_2024-2026.03"

    # # Search files and store filepaths in list
    # from diive.core.io.filereader import search_files, MultiDataFileReader
    # sourcefiles = search_files(searchdirs=SOURCEDIR, pattern='*_fluxnet_*.csv')
    # d = MultiDataFileReader(filepaths=sourcefiles,
    #                         filetype='EDDYPRO-FLUXNET-CSV-30MIN',
    #                         output_middle_timestamp=True)
    # df = d.data_df
    # from diive.core.io.files import save_parquet
    # filepath = save_parquet(filename="FLUXES_L0_ALL", data=df, outpath=SOURCEDIR)

    # Load from parquet
    from diive.core.io.files import load_parquet
    FILENAME = r"FLUXES_L0_ALL.parquet"
    FILEPATH = Path(SOURCEDIR) / FILENAME
    maindf = load_parquet(filepath=str(FILEPATH))

    # # Or load EddyPro _fluxnet_ output files
    # SOURCEDIRS = [r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-hon_flux_product\dataset_ch-hon_flux_product\notebooks\0_data\OPENLAG-IRGA-Level-0_fluxnet_2024-2026.03"]
    # ep = LoadEddyProOutputFiles(sourcedir=SOURCEDIRS, filetype='EDDYPRO-FLUXNET-CSV-30MIN')
    # ep.searchfiles()
    # ep.loadfiles()
    # maindf = ep.maindf
    # metadata = ep.metadata

    # # Restrict time range
    # locs = ((maindf.index.year >= 2023) & (maindf.index.year <= 2023)
    #         & (maindf.index.month >= 6) & (maindf.index.month <= 7))
    # maindf = maindf.loc[locs, :].copy()
    # # metadata = None
    # # print(maindf)

    # Restrict by wind direction (CH-HON)
    import numpy as np
    locs = (maindf['WD'] > 180) & (maindf['WD'] < 350)
    maindf.loc[locs, :] = np.nan

    # Flux processing chain settings
    FLUXVAR = "FC"
    SITE_LAT = 47.41887  # CH-HON
    SITE_LON = 8.491318  # CH-HON
    UTC_OFFSET = 1
    NIGHTTIME_THRESHOLD = 20  # Threshold for potential radiation in W m-2, conditions below threshold are nighttime
    DAYTIME_ACCEPT_QCF_BELOW = 1
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
    TEST_SSITC_SETFLAG_TIMEPERIOD = None
    # TEST_SSITC_SETFLAG_TIMEPERIOD = {2: [[1, '2022-05-01', '2023-09-30']]}  # Set flag 1 to 2 (bad) during time period
    TEST_GAS_COMPLETENESS = False  # Default True
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
            'apply': TEST_SSITC,
            'setflag_timeperiod': TEST_SSITC_SETFLAG_TIMEPERIOD},
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
    fpc.level2_qcf.report_qcf_evolution()
    # fpc.level2_qcf.analyze_highest_quality_flux()
    # fpc.level2_qcf.report_qcf_flags()
    # fpc.level2.results
    # fpc.fpc_df
    # fpc.filteredseries
    # [x for x in fpc.fpc_df.columns if 'L2' in x]

    # --------------------
    # Level-3.1
    # --------------------
    fpc.level31_storage_correction(gapfill_storage_term=False, set_storage_to_zero=False)
    # fpc.level31_storage_correction(gapfill_storage_term=False)
    fpc.finalize_level31()
    # fpc.level31.report()
    # fpc.level31.showplot()
    # fpc.fpc_df
    # fpc.filteredseries
    # fpc.level31.results
    # [x for x in fpc.fpc_df.columns if 'L3.1' in x]
    # -------------------------------------------------------------------------

    # # --------------------
    # # (OPTIONAL) ANALYZE
    # # --------------------
    # fpc.analyze_highest_quality_flux(showplot=True)
    # # -------------------------------------------------------------------------

    # --------------------
    # Level-3.2
    # --------------------
    fpc.level32_stepwise_outlier_detection()

    # fpc.level32_flag_manualremoval_test(
    #     remove_dates=[
    #         ['2016-03-18 12:15:00', '2016-05-03 06:45:00'],
    #         # '2022-12-12 12:45:00',
    #     ],
    #     showplot=False, verbose=True)
    # fpc.level32_addflag()

    # DAYTIME_MINMAX = [-50, 50]
    # NIGHTTIME_MINMAX = [-50, 50]
    # fpc.level32_flag_outliers_abslim_dtnt_test(daytime_minmax=DAYTIME_MINMAX, nighttime_minmax=NIGHTTIME_MINMAX,
    #                                            showplot=False, verbose=False)
    # # fpc.level32_flag_outliers_abslim_dtnt_test(daytime_minmax=DAYTIME_MINMAX, nighttime_minmax=NIGHTTIME_MINMAX, showplot=True, verbose=True)
    # fpc.level32_addflag()
    # # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_zscore_dtnt_test(thres_zscore=4, showplot=True, verbose=False, repeat=True)
    # fpc.level32_addflag()

    # # fpc.level32_flag_outliers_hampel_dtnt_test(window_length=48 * 3, n_sigma_dt=3.5, n_sigma_nt=3.5, showplot=False, verbose=False, repeat=False)
    fpc.level32_flag_outliers_hampel_dtnt_test(window_length=48 * 13, n_sigma_dt=5.5, n_sigma_nt=5.5, showplot=False,
                                               verbose=True, use_differencing=True, separate_day_night=True,
                                               repeat=True)
    fpc.level32_addflag()

    # fpc.level32_flag_outliers_zscore_rolling_test(winsize=48 * 7, thres_zscore=4, showplot=False, verbose=True,
    #                                               repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_localsd_test(n_sd=5, winsize=48 * 13, constant_sd=False, showplot=False, verbose=True,
    #                                        repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_localsd_test(n_sd=4, winsize=48 * 13, constant_sd=True, showplot=False, verbose=True, repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_localsd_test(n_sd=[1.2, 99], winsize=[48 * 13, 48 * 2], constant_sd=False,
    #                                        separate_daytime_nighttime=True, lat=SITE_LAT, lon=SITE_LON, utc_offset=1,
    #                                        showplot=True, verbose=True, repeat=False)
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
    # print("STOP")
    # -------------------------------------------------------------------------

    # --------------------
    # Level-3.3
    # --------------------
    # 0.052945, 0.069898, 0.092841
    # ustar_scenarios = ['NO_USTAR']
    # ustar_thresholds = [-9999]
    # ustar_scenarios = ['CUT_50']
    # ustar_thresholds = [0.069898]
    ustar_scenarios = ['CUT_50']
    ustar_thresholds = [0.09]  # CH-LAE
    # ustar_scenarios = ['CUT_16', 'CUT_50', 'CUT_84']
    # ustar_thresholds = [0.271619922, 0.303628125, 0.339684084]  # CH-LAE
    # ustar_scenarios = ['CUT_16', 'CUT_50', 'CUT_84']
    # ustar_thresholds = [0.052945, 0.069898, 0.092841]
    fpc.level33_constant_ustar(thresholds=ustar_thresholds,
                               threshold_labels=ustar_scenarios,
                               showplot=False)
    # Finalize: stores results for each USTAR scenario in a dict
    fpc.finalize_level33()

    # # Save current instance to pickle for faster testing
    # from diive.core.io.files import save_as_pickle
    # save_as_pickle(outpath=r"F:\TMP", filename="test", data=fpc)
    # fpc = load_pickle(filepath=r"F:\TMP\test.pickle")

    for ustar_scenario in ustar_scenarios:
        # fpc.level33_qcf[ustar_scenario].showplot_qcf_heatmaps()
        fpc.level33_qcf[ustar_scenario].report_qcf_evolution()
        # fpc.filteredseries
        # fpc.level33
        # fpc.level33_qcf.showplot_qcf_timeseries()
        # fpc.level33_qcf.report_qcf_flags()
        # fpc.level33_qcf.report_qcf_series()
        # fpc.levelidstr
        # fpc.filteredseries_level2_qcf
        # fpc.filteredseries_level31_qcf
        # fpc.filteredseries_level32_qcf
        # fpc.filteredseries_level33_qcf

    # --------------------
    # Level-4.1
    # --------------------
    # FEATURES = ["TA_T1_47_1_gfXG", "SW_IN_T1_47_1_gfXG", "VPD_T1_47_1_gfXG"]
    FEATURES = ["TA_EP", "SW_IN_POT", "VPD_EP"]


    # # ---------------------------
    # # RANDOM FOREST WITH COMPREHENSIVE CO2 FLUX CONFIGURATION
    # # ---------------------------
    # # Level-4.1 Random Forest Gap-Filling optimized for CO2 half-hourly flux (NEE) data
    # # Random Forest is interpretable, robust to outliers, and excellent for exploring
    # # feature importance. It uses the SAME feature engineering as XGBoost for direct
    # # comparison. RF often slightly underperforms XGBoost on non-linear patterns but
    # # provides better uncertainty estimates and is less prone to overfitting.
    # #
    # # KEY TUNING FOR CO2 FLUX (30-min resolution):
    # # - Lag features: Short windows (1-2 steps = 30-60 min) for fast response
    # # - Rolling windows: 3-24 hours (6-48 steps) for diurnal pattern context
    # # - Differencing: Order-1 for rate-of-change, helps detect flux transitions
    # # - EMA: Multi-timescale (3-24 hours) captures memory effects from stomata/photosystem
    # # - STL: CRITICAL for CO2 - strong daily cycle + seasonal dormancy/growth
    # # - Timestamps: ESSENTIAL - diurnal photosynthesis is time-of-day dependent
    # # - Polynomial: Captures non-linear light saturation (Michaelis-Menten kinetics)
    # #
    # # Feature count: ~45-50 engineered features (same as XGBoost for fair comparison)
    # # Typical training time: 3-8 min per year per USTAR scenario (slower than XGBoost)
    # # Expected model R²: 0.60-0.80 (slightly lower than XGBoost, but more robust)
    # # Memory usage: ~2-3× higher than XGBoost for same feature count
    # #
    # # RANDOM FOREST VS XGBOOST:
    # # - RF: More interpretable, robust to outliers, good for exploration
    # # - XGB: Better on non-linear patterns, faster, smaller models
    # # - Compare both on same data to identify which fits your site better
    #
    # fpc.level41_longterm_random_forest(
    #     features=FEATURES,
    #
    #     # ===== FEATURE ENGINEERING PARAMETERS (Identical to XGBoost) =====
    #
    #     # Stage 1: LAG FEATURES (Immediate past context)
    #     # For CO2: Short lags (1-2 steps = 30-60 min) as flux responds quickly to environment
    #     features_lag=[-2, -1],          # Past 30-60 min only (no future, no current)
    #     features_lag_stepsize=1,        # Include every lag (dense temporal context)
    #     features_lag_exclude_cols=None, # Lag all input features
    #
    #     # Stage 2: ROLLING STATISTICS (Diurnal pattern capture)
    #     # For CO2: 30-min, 1-hr, 2-hr, 6-hr, 12-hr, 24-hr windows to capture diurnal patterns
    #     # Window sizes for 30-min data: 2=1hr, 4=2hr, 12=6hr, 24=12hr, 48=24hr
    #     features_rolling=[2, 4, 12, 24, 48],  # 1hr, 2hr, 6hr, 12hr, 24hr windows
    #     features_rolling_exclude_cols=None,   # Apply to all input features
    #     # Advanced rolling stats: Median robust to outliers, min/max/quantiles capture asymmetry
    #     features_rolling_stats=['median', 'min', 'max', 'std', 'q25', 'q75'],
    #
    #     # Stage 3: TEMPORAL DIFFERENCING (Rate of change, flux transitions)
    #     # For CO2: Order-1 (rate) captures sunrise/sunset transitions and weather events
    #     # Order-2 (acceleration) helps detect rapid state changes
    #     features_diff=[1, 2],           # First and second-order differencing
    #     features_diff_exclude_cols=None,
    #
    #     # Stage 4: EXPONENTIAL MOVING AVERAGE (Multi-timescale memory)
    #     # For CO2: Captures stomatal/photosynthetic adjustment at multiple timescales
    #     # Spans for 30-min data: 6=3hr, 12=6hr, 24=12hr, 48=24hr
    #     features_ema=[6, 12, 24, 48],   # 3hr, 6hr, 12hr, 24hr exponential moving averages
    #     features_ema_exclude_cols=None,
    #
    #     # Stage 5: POLYNOMIAL EXPANSION (Non-linear relationships)
    #     # For CO2: Degree-2 essential for light saturation (Michaelis-Menten curve)
    #     # Captures photosynthetic saturation and respiratory asymmetry
    #     features_poly_degree=2,         # Quadratic terms (e.g., Tair², Rg² for saturation)
    #     features_poly_exclude_cols=None,
    #
    #     # Stage 6: STL DECOMPOSITION (Trend/Seasonal separation)
    #     # For CO2: CRITICAL - separates respiration trend from photosynthetic pattern
    #     # Daily cycle: photosynthesis (daytime negative NEE), respiration (nighttime positive)
    #     # Seasonal cycle: dormancy (winter respiration), growth (summer photosynthesis)
    #     features_stl=True,                      # Enable STL decomposition
    #     features_stl_method='stl',              # Robust LOESS method (handles gaps)
    #     features_stl_seasonal_period=48,        # 30-min × 48 = 24 hours (daily cycle)
    #     features_stl_exclude_cols=None,         # Apply to all input features
    #     features_stl_components=['trend', 'seasonal', 'residual'],  # Extract all
    #
    #     # Stage 7: TIMESTAMP FEATURES (Diurnal/Seasonal cycles)
    #     # For CO2: ESSENTIAL - photosynthesis depends on time-of-day (solar elevation)
    #     # and season (leaf phenology, dormancy)
    #     vectorize_timestamps=True,      # Creates ~19 features: year, season, DOY, hour, etc.
    #
    #     # Stage 8: SEQUENTIAL RECORD NUMBER (Long-term drift)
    #     # For CO2: Useful if site shows long-term drift (instrument aging, vegetation change)
    #     add_continuous_record_number=True,  # 1, 2, 3, ... for drift capture
    #
    #     # Data quality preprocessing
    #     sanitize_timestamp=True,        # Validate timestamps (catch gaps/duplicates)
    #
    #     # ===== GAP-FILLING PARAMETERS =====
    #     reduce_features=True,          # ENABLED: Apply SHAP-based feature selection
    #                                     # Selects only important features across all years
    #                                     # Reduces feature count from ~45-50 to ~10-20 features
    #                                     # Benefits: Faster training, better generalization, smaller models
    #                                     # Drawback: Removes potentially useful features
    #     verbose=True,                   # Print progress and model scores
    #
    #     # ===== RANDOM FOREST HYPERPARAMETERS =====
    #     # Tuned for flux data (non-linear, heteroscedastic, with clear diurnal cycle)
    #     # RF typically needs more estimators than XGBoost to reach same performance
    #
    #     n_estimators=350,               # 350 trees (more than XGBoost due to bagging)
    #                                     # Random Forest needs ~50% more trees than XGBoost
    #                                     # Increase (400-500) if underfitting (R² too low)
    #                                     # Decrease (200-300) if overfitting or memory-limited
    #                                     # Training time ~linear with n_estimators
    #
    #     max_depth=15,                   # Tree depth (deeper than XGBoost)
    #                                     # RF can use deeper trees without overfitting
    #                                     # Default 15 balances complexity and stability
    #                                     # Increase (20-25) for complex patterns
    #                                     # Decrease (8-10) if overfitting
    #
    #     min_samples_split=5,            # Minimum samples required to split a node
    #                                     # Higher values prevent overfitting to individual data points
    #                                     # 5 = good balance for 30-min flux data (~hours of training data)
    #                                     # Increase (10-15) if overfitting to noise
    #                                     # Decrease (2-3) if underfitting
    #
    #     min_samples_leaf=2,             # Minimum samples required at leaf node
    #                                     # Higher = smoother predictions, less overfitting
    #                                     # 2 = permissive, allows feature detection
    #                                     # Increase (5-10) if overfitting
    #                                     # Decrease (1) if underfitting
    #
    #     n_jobs=-1,                      # Use all CPU cores (parallel tree building)
    #     random_state=42,                # Reproducibility (same results every run)
    # )
    # # ===== ACCESS RESULTS =====
    # # model = fpc.level41['long_term_random_forest']['CUT_50']
    # # gapfilled_co2 = model.gapfilled_
    # # scores = model.scores_  # R², MAE, RMSE on test data
    # # feature_importance = model.feature_importances_  # SHAP importance per feature
    # # yearly_models = model.results_yearly_  # Per-year model results (dict keyed by year)
    # #
    # # ===== COMPARE RF VS XGBOOST =====
    # # rf_r2 = fpc.level41['long_term_random_forest']['CUT_50'].scores_['r2']
    # # xgb_r2 = fpc.level41['long_term_xgboost']['CUT_50'].scores_['r2']
    # # print(f"Random Forest R²: {rf_r2:.3f}")
    # # print(f"XGBoost R²: {xgb_r2:.3f}")
    # # print(f"Winner: {'XGBoost' if xgb_r2 > rf_r2 else 'Random Forest'}")

    # ---------------------------
    # XGBOOST WITH COMPREHENSIVE CO2 FLUX CONFIGURATION
    # ---------------------------
    # Level-4.1 XGBoost Gap-Filling optimized for CO2 half-hourly flux (NEE) data
    # This configuration balances capture of diurnal photosynthetic patterns with
    # computational efficiency. XGBoost often outperforms Random Forest on non-linear
    # flux responses (light saturation, stomatal conductance).
    #
    # KEY TUNING FOR CO2 FLUX (30-min resolution):
    # - Lag features: Short windows (1-2 steps = 30-60 min) for fast response
    # - Rolling windows: 3-24 hours (6-48 steps) for diurnal pattern context
    # - Differencing: Order-1 for rate-of-change, helps detect flux transitions
    # - EMA: Multi-timescale (3-24 hours) captures memory effects from stomata/photosystem
    # - STL: CRITICAL for CO2 - strong daily cycle + seasonal dormancy/growth
    # - Timestamps: ESSENTIAL - diurnal photosynthesis is time-of-day dependent
    # - Polynomial: Captures non-linear light saturation (Michaelis-Menten kinetics)
    #
    # Feature count: ~45-50 engineered features
    # Typical training time: 2-5 min per year per USTAR scenario
    # Expected model R²: 0.65-0.85 depending on site complexity

    fpc.level41_longterm_xgboost(
        features=FEATURES,

        # ===== FEATURE ENGINEERING PARAMETERS =====

        # Stage 1: LAG FEATURES (Immediate past context)
        # For CO2: Short lags (1-2 steps = 30-60 min) as flux responds quickly to environment
        features_lag=[-2, -1],          # Past 30-60 min only (no future, no current)
        features_lag_stepsize=1,        # Include every lag (dense temporal context)
        features_lag_exclude_cols=None, # Lag all input features

        # Stage 2: ROLLING STATISTICS (Diurnal pattern capture)
        # For CO2: 30-min, 1-hr, 2-hr, 6-hr, 12-hr, 24-hr windows to capture diurnal patterns
        # Window sizes for 30-min data: 2=1hr, 4=2hr, 12=6hr, 24=12hr, 48=24hr
        features_rolling=[2, 4, 12, 24, 48],  # 1hr, 2hr, 6hr, 12hr, 24hr windows
        features_rolling_exclude_cols=None,   # Apply to all input features
        # Advanced rolling stats: Median robust to outliers, min/max/quantiles capture asymmetry
        features_rolling_stats=['median', 'min', 'max', 'std', 'q25', 'q75'],

        # Stage 3: TEMPORAL DIFFERENCING (Rate of change, flux transitions)
        # For CO2: Order-1 (rate) captures sunrise/sunset transitions and weather events
        # Order-2 (acceleration) helps detect rapid state changes
        features_diff=[1, 2],           # First and second-order differencing
        features_diff_exclude_cols=None,

        # Stage 4: EXPONENTIAL MOVING AVERAGE (Multi-timescale memory)
        # For CO2: Captures stomatal/photosynthetic adjustment at multiple timescales
        # Spans for 30-min data: 6=3hr, 12=6hr, 24=12hr, 48=24hr
        features_ema=[6, 12, 24, 48],   # 3hr, 6hr, 12hr, 24hr exponential moving averages
        features_ema_exclude_cols=None,

        # Stage 5: POLYNOMIAL EXPANSION (Non-linear relationships)
        # For CO2: Degree-2 essential for light saturation (Michaelis-Menten curve)
        # Captures photosynthetic saturation and respiratory asymmetry
        features_poly_degree=2,         # Quadratic terms (e.g., Tair², Rg² for saturation)
        features_poly_exclude_cols=None,

        # Stage 6: STL DECOMPOSITION (Trend/Seasonal separation)
        # For CO2: CRITICAL - separates respiration trend from photosynthetic pattern
        # Daily cycle: photosynthesis (daytime negative NEE), respiration (nighttime positive)
        # Seasonal cycle: dormancy (winter respiration), growth (summer photosynthesis)
        features_stl=True,                      # Enable STL decomposition
        features_stl_method='stl',              # Robust LOESS method (handles gaps)
        features_stl_seasonal_period=48,        # 30-min × 48 = 24 hours (daily cycle)
        features_stl_exclude_cols=None,         # Apply to all input features
        features_stl_components=['trend', 'seasonal', 'residual'],  # Extract all

        # Stage 7: TIMESTAMP FEATURES (Diurnal/Seasonal cycles)
        # For CO2: ESSENTIAL - photosynthesis depends on time-of-day (solar elevation)
        # and season (leaf phenology, dormancy)
        vectorize_timestamps=True,      # Creates ~19 features: year, season, DOY, hour, etc.

        # Stage 8: SEQUENTIAL RECORD NUMBER (Long-term drift)
        # For CO2: Useful if site shows long-term drift (instrument aging, vegetation change)
        add_continuous_record_number=True,  # 1, 2, 3, ... for drift capture

        # Data quality preprocessing
        sanitize_timestamp=True,        # Validate timestamps (catch gaps/duplicates)

        # ===== GAP-FILLING PARAMETERS =====
        reduce_features=True,          # ENABLED: Apply SHAP-based feature selection
                                        # Selects only important features across all years
                                        # Reduces feature count from ~45-50 to ~10-20 features
                                        # Benefits: Faster training, better generalization, smaller models
                                        # Drawback: Removes potentially useful features
        verbose=True,                   # Print progress and model scores

        # ===== XGBOOST HYPERPARAMETERS =====
        # Tuned for flux data (non-linear, heteroscedastic, with clear diurnal cycle)

        n_estimators=500,               # 250 boosting rounds (less than Random Forest)
                                        # XGBoost needs fewer estimators than RF
                                        # Increase if underfitting (R² too low)
                                        # Decrease if overfitting (test R² << train R²)

        max_depth=6,                    # Tree depth (shallow, prevents overfitting)
                                        # Default 6 is good. Increase (7-8) for complex patterns
                                        # Decrease (4-5) if overfitting

        learning_rate=0.05,              # Shrinkage parameter (how fast to learn)
                                        # 0.1 = standard, 0.05 = slower/safer, 0.3 = aggressive
                                        # Smaller = better generalization, slower training

        early_stopping_rounds=30,       # Stop if validation doesn't improve for 20 rounds
                                        # Prevents overfitting, reduces training time
                                        # Increase (30-50) for more aggressive training

        n_jobs=-1,                      # Use all CPU cores (parallel training)
        random_state=42,                # Reproducibility (same results every run)
        min_child_weight=5,
    )
    # ===== ACCESS RESULTS =====
    # model = fpc.level41['long_term_xgboost']['CUT_50']
    # gapfilled_co2 = model.gapfilled_
    # scores = model.scores_  # R², MAE, RMSE on test data
    # feature_importance = model.feature_importances_  # SHAP importance per feature
    # yearly_models = model.results_yearly_  # Per-year model results (dict keyed by year)

    # fpc.level41_mds(
    #     swin="SW_IN_POT",
    #     ta="TA_EP",
    #     vpd="VPD_EP",
    #     swin_tol=[20, 50],
    #     ta_tol=2.5,
    #     vpd_tol=0.5,
    #     avg_min_n_vals=5
    # )

    results = fpc.get_data()
    # gapfilled_names = fpc.get_gapfilled_names()
    # nongapfilled_names = fpc.get_nongapfilled_names()
    # gapfilled_vars = fpc.get_gapfilled_variables()
    fpc.report_gapfilling_variables()
    fpc.report_gapfilling_model_scores()
    fpc.report_traintest_model_scores()
    fpc.report_traintest_details()
    # fpc.report_gapfilling_feature_importances()

    # # Only ML models:
    # fpc.report_gapfilling_poolyears()

    # todo get full data

    fpc.showplot_gapfilled_heatmap(vmin=-30, vmax=30)
    fpc.showplot_gapfilled_cumulative(gain=0.02161926, units=r'($\mathrm{µmol\ CO_2\ m^{-2}}$)', per_year=True)
    fpc.showplot_gapfilled_cumulative(gain=0.02161926, units=r'($\mathrm{g\ C\ m^{-2}}$)', per_year=False)


    from diive.core.plotting.dielcycle import DielCycle
    series = results['NEE_L3.1_L3.3_CUT_50_QCF_gfXG'].copy()
    # series = results['NEE_L3.1_L3.3_CUT_50_QCF_gfMDS'].copy()
    dc = DielCycle(series=series)
    title = r'$\mathrm{Mean\ CO_2\ flux\ (Feb 2024 - Mar 2026)}$'
    units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
    dc.plot(ax=None, title=title, txt_ylabel_units=units,
            each_month=True, legend_n_col=2)

    # # # Only ML models:
    # fpc.showplot_feature_ranks_per_year()

    # # # Only MDS:
    # fpc.showplot_mds_gapfilling_qualities()



    print("END")


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
    _example()
    # example_cumu()
