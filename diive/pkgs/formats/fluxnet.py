import re
from pathlib import Path

import numpy as np
from pandas import DataFrame

from diive.core.io.files import loadfiles
from diive.core.times.times import current_date_str_condensed
from diive.core.times.times import format_timestamp_to_fluxnet_format
from diive.core.times.times import insert_timestamp
from diive.pkgs.outlierdetection.manualremoval import ManualRemoval
from diive.pkgs.qaqc.eddyproflags import flag_signal_strength_eddypro_test

# Names of variables in the EddyPro _fluxnet_ output file
VARS_CO2 = ['FC', 'FC_SSITC_TEST', 'SC_SINGLE', 'CO2']
VARS_H2O = ['LE', 'LE_SSITC_TEST', 'SLE_SINGLE', 'H2O']
VARS_H = ['H', 'H_SSITC_TEST', 'SH_SINGLE']
VARS_WIND = ['USTAR', 'WD', 'WS', 'FETCH_70', 'FETCH_90', 'FETCH_MAX']
VARS_METEO = ['SW_IN_1_1_1', 'TA_1_1_1', 'RH_1_1_1', 'PA_1_1_1', 'LW_IN_1_1_1', 'PPFD_IN_1_1_1',
              'G_1_1_1', 'NETRAD_1_1_1', 'TS_1_1_1', 'P_1_1_1', 'SWC_1_1_1']
VARIABLES = VARS_CO2 + VARS_H2O + VARS_H + VARS_WIND + VARS_METEO

# Some variables need to be renamed to comply with FLUXNET variable codes
renaming_dict = {
    'SC_SINGLE': 'SC',
    'SLE_SINGLE': 'SLE',
    'SH_SINGLE': 'SH'
}


class FormatEddyProFluxnetFileForUpload:
    """
    Helper class to convert EddyPro _fluxnet_ output files to the file
    format required for data upload (data sharing) to FLUXNET

    The class does the following:
    - Search source folder for _fluxnet_ files
    - Load data from all found files and store in dataframe
    - Make subset that contains required variables
    - Set missing values to FLUXNET format (-9999)
    - Rename variables by adding the FLUXNET suffix (_1_1_1)
    - Insert two timestamp columns denoting START and END of averaging interval
    - Format the two timestamp columns to FLUXNET format YYYYMMDDhhmm
    - Save data from dataframe to yearly files

    - Example notebook available in:
        notebooks/Formats/ConvertEddyProFluxnetFileForUpload.ipynb

    Variables shared with FLUXNET:

        CO2:
        - FC (µmolCO2 m-2 s-1): Carbon Dioxide (CO2) turbulent flux (without storage component)
        - FC_SSITC_TEST (adimensional): Quality check - CO2 flux - * see the note
        - SC (µmolCO2 m-2 s-1): Carbon Dioxide (CO2) storage flux measured with a vertical profile
          system, optional if tower shorter than 3 m
        - CO2 (µmolCO2 mol-1): Carbon Dioxide (CO2) mole fraction in moist air

        LE:
        - LE (W m-2): latent heat turbulent flux, without storage correction
        - LE_SSITC_TEST (adimensional): Quality check - latent heat flux - * see the note
        - SLE (W m-2): Latent heat storage below flux measurement level
        - H2O (ppt: mmolH2O mol-1): Water (H2O) vapor mole fraction

        H:
        - H (W m-2): sensible heat turbulent flux, without storage correction
        - H_SSITC_TEST (adimensional): Quality check - sensible heat flux - * see the note
        - SH (W m-2): Heat storage in air below flux measurement level

        Wind & Footprint:
        - USTAR (m s-1): friction velocity
        - WD (°): wind direction
        - WS (m s-1): horizontal wind speed
        - FETCH_70 (m): Fetch at which footprint cumulated probability is 70%
        - FETCH_90 (m): Fetch at which footprint cumulated probability is 90%
        - FETCH_MAX (m): Fetch at which footprint cumulated probability is maximum

        Mandatory meteo:
        - TA (°C): air temperature
        - RH (%): relative humidity (range 0–100%)
        - PA (kPa): atmospheric pressure
        - SW_IN (W m-2): incoming shortwave radiation

        Non-mandatory meteo:
        - G (W m-2): ground heat flux, not mandatory, but needed for the energy balance closure calculations
        - NETRAD (W m-2): net radiation, not mandatory, but needed for the energy balance closure calculations
        - TS (°C): soil temperature
        - PPFD_IN (µmolPhotons m-2 s-1): incoming photosynthetic photon flux density
        - P (mm): precipitation total of each 30 or 60 minute period
        - LW_IN (W m-2): incoming (down-welling) longwave radiation
        - SWC (%): soil water content (volumetric), range 0–100%

    FLUXNET variable codes:
        - http://www.europe-fluxdata.eu/home/guidelines/how-to-submit-data/variables-codes

    """

    def __init__(self,
                 site: str,
                 sourcedir: str,
                 outdir: str,
                 add_runid: bool = True):
        self.site = site
        self.sourcedir = sourcedir
        self.outdir = outdir
        self.add_runid = add_runid

        self._merged_df = None
        self._subset_fluxnet = None

        print(f"\nInitiated formatting for datafiles with the following settings:")
        print(f"    site: {self.site}")
        print(f"    source folder: {self.sourcedir}")
        print(f"    output folder: {self.outdir}")
        print(f"    add run ID: {self.add_runid}")

    @property
    def merged_df(self) -> DataFrame:
        """Return merged dataframe of all data in all files"""
        if not isinstance(self._merged_df, DataFrame):
            raise Exception(f"No merged data available.")
        return self._merged_df

    @property
    def subset_fluxnet(self) -> DataFrame:
        """Return dataframe with variable subset for FLUXNET"""
        if not isinstance(self._subset_fluxnet, DataFrame):
            raise Exception(f"No merged data available.")
        return self._subset_fluxnet

    def mergefiles(self, limit_n_files: int = None):
        self._merged_df = loadfiles(filetype='EDDYPRO-FLUXNET-CSV-30MIN',
                                    sourcedir=self.sourcedir,
                                    limit_n_files=limit_n_files,
                                    fileext='.csv',
                                    idstr='_fluxnet_')

    def remove_erroneous_data(self, var: str, remove_dates: list, showplot: bool):
        """
        Remove erroneous time periods from data

        Args:
            var: str, name of variable from which data points are removed
            remove_dates: list, see docstring of pkgs.outlierdetection.manualremoval.ManualRemoval.calc
                for examples
            showplot: show plot with removed data points

        Returns:
            updated *var* in merged dataframe, with data points removed
        """
        print(f"\nRemoving {var} data points for the following date(s) and/or time range(s):")
        for d in remove_dates:
            if isinstance(d, str):
                print(f"    REMOVING data for {var} single data point {d}")
            elif isinstance(d, list):
                print(f"    REMOVING data for {var} time range between {d} (dates are inclusive)")
        series = self.merged_df[var].copy()
        n_vals_before = series.dropna().count()
        flagtest = ManualRemoval(series=series, remove_dates=remove_dates,
                                 showplot=showplot, verbose=True)
        flagtest.calc(repeat=False)
        flag = flagtest.get_flag()

        # Locations where flag is > 0
        reject = flag > 0

        # Remove rejected series values from series (i.e., set to missing values)
        series.loc[reject] = np.nan

        # Insert filtered series in dataset
        self._merged_df[var] = series.copy()

        # Info number of rejected values
        n_vals_after = self.merged_df[var].dropna().count()
        n_rejected = reject.sum()
        print(f"Manual removal rejected {n_rejected} values of {var}, all rejected "
              f"value were removed from the dataset.")
        print(f"\nAvailable values of {var} before removing fluxes: {n_vals_before}")
        print(f"Available values of {var} after removing fluxes: {n_vals_after}")

    def apply_fluxnet_format(self):
        self._subset_fluxnet = self._make_subset(df=self.merged_df)
        self._subset_fluxnet = self._missing_values(df=self._subset_fluxnet)
        self._subset_fluxnet = self._rename_to_variable_codes(df=self._subset_fluxnet)
        self._subset_fluxnet = self._rename_add_suffix(df=self._subset_fluxnet)
        self._subset_fluxnet = self._insert_timestamp_columns(df=self._subset_fluxnet)
        self._subset_fluxnet['TIMESTAMP_END'] = \
            format_timestamp_to_fluxnet_format(df=self._subset_fluxnet, timestamp_col='TIMESTAMP_END')
        self._subset_fluxnet['TIMESTAMP_START'] = \
            format_timestamp_to_fluxnet_format(df=self._subset_fluxnet, timestamp_col='TIMESTAMP_START')

    def export_yearly_files(self):
        """Create one file per year"""
        self._save_one_file_per_year(df=self._subset_fluxnet)

    def get_data(self):
        return self._subset_fluxnet

    def _save_one_file_per_year(self, df: DataFrame):
        """Save data to yearly files"""
        print(f"\nSaving yearly CSV files ...")
        uniq_years = list(df.index.year.unique())
        runid = f"_{current_date_str_condensed()}" if self.add_runid else ""
        for year in uniq_years:
            outname = f"{self.site}_{year}_fluxes_meteo{runid}.csv"
            outpath = Path(self.outdir) / outname
            yearlocs = df.index.year == year
            yeardata = df[yearlocs].copy()
            yeardata.to_csv(outpath, index=False)
            print(f"    --> Saved file {outpath}.")

    @staticmethod
    def _missing_values(df: DataFrame):
        """Set all missing values to -9999 as required by FLUXNET"""
        print("\nSetting all missing values to -9999 ...")
        return df.fillna(-9999)

    @staticmethod
    def _insert_timestamp_columns(df: DataFrame):
        """Insert timestamp columns denoting start and end of averaging interval"""
        # Add timestamp column TIMESTAMP_END
        df = insert_timestamp(data=df, convention='end', insert_as_first_col=True, verbose=True)
        # Add timestamp column TIMESTAMP_START
        df = insert_timestamp(data=df, convention='start', insert_as_first_col=True, verbose=True)
        return df

    @staticmethod
    def _rename_add_suffix(df: DataFrame) -> DataFrame:
        """Rename variables to FLUXNET format by adding suffix _1_1_1"""
        renaming_dict = {}
        notrenamed = []
        for var in df.columns:
            # Check whether the variable name ends with a FLUXNET suffix
            has_suffix = re.match('.*_[0-9]_[0-9]_[0-9]$', var)
            if not has_suffix:
                renaming_dict[var] = f"{var}_1_1_1"
            else:
                notrenamed.append(var)
        df = df.rename(columns=renaming_dict)
        print(f"\nThe following variables have been renamed:")
        for ix, val in renaming_dict.items():
            print(f"    RENAMED --> {ix} was renamed to {val}")
        print(f"\nThe following variables have not been renamed:")
        for var in notrenamed:
            print(f"    NOT RENAMED --> {var} was not renamed")
        return df

    @staticmethod
    def _rename_to_variable_codes(df: DataFrame) -> DataFrame:
        """
        Rename variables to comply with FLUXNET variable codes

        see: http://www.europe-fluxdata.eu/home/guidelines/how-to-submit-data/variables-codes
        """
        print("\nThe following variables are renamed to comply with FLUXNET variable codes:")
        for key, val in renaming_dict.items():
            print(f"    RENAMED --> {key} was renamed to {val}")
        df = df.rename(columns=renaming_dict)
        return df

    @staticmethod
    def _make_subset(df: DataFrame) -> DataFrame:
        """Make subset that contains variables available for sharing"""
        print("\nAssembling subset of variables ...")
        available_vars = df.columns
        subsetcols = []
        notavailablecols = []
        for var in VARIABLES:
            subsetcols.append(var) if var in available_vars else notavailablecols.append(var)
        print(f"    Found: {subsetcols}")
        print(f"    Not found: {notavailablecols}")
        subset = df[subsetcols].copy()
        return subset

    def remove_low_signal_data(self,
                               fluxcol: str,
                               signal_strength_col: str,
                               method: str,
                               threshold: int) -> None:
        """
        Remove flux values where signal strength / AGC is not sufficient (too high or too low)

        Args:
            fluxcol: str, name of flux variable
            signal_strength_col: str, name of signal strength or AGC variable
            method: str, 'discard above' or 'discard below' *threshold*
            threshold: int, threshold to remove data points

        Returns:
            None
        """
        print(f"\n\n{'=' * 80}\nRemoving {fluxcol} flux values where signal strength / AGC is not sufficient:")
        levelid = 'L2'  # ID to identify newly created columns
        df = self.merged_df.copy()
        keepcols = df.columns.copy()  # Original columns in df, used to keep only original variable names
        n_vals_before = df[fluxcol].dropna().count()

        print(f"\nPerforming signal strength / AGC quality check ...\n")
        flag = flag_signal_strength_eddypro_test(df=df,
                                                 signal_strength_col=signal_strength_col,
                                                 var_col=fluxcol,
                                                 method=method,
                                                 threshold=threshold,
                                                 idstr=levelid)
        # Locations where flag is > 0
        reject = flag > 0

        # Remove rejected fluxcol values from dataset (i.e., set to missing values)
        df.loc[reject, fluxcol] = np.nan

        # Info number of rejected values
        n_vals_after = df[fluxcol].dropna().count()
        n_rejected = reject.sum()
        print(f"{signal_strength_col} rejected {n_rejected} values of {fluxcol}, all rejected "
              f"value were removed from the dataset.")
        print(f"\nAvailable values of {fluxcol} before removing low signal fluxes: {n_vals_before}")
        print(f"Available values of {fluxcol} after removing low signal fluxes: {n_vals_after}")

        # Keep original columns only, do not keep the newly generated L2 columns
        print("\nRemoving all newly generated columns relating to quality check "
              "(not needed for FLUXNET), restoring original set of variables ...")
        self._merged_df = df[keepcols].copy()


def example():
    # from diive.configs.exampledata import load_exampledata_eddypro_fluxnet_CSV_30MIN
    # data_df, metadata_df = load_exampledata_eddypro_fluxnet_CSV_30MIN()

    # Setup
    SOURCE = r"F:\Sync\luhk_work\CURRENT\fru\Level-1_results_fluxnet_2023\0-eddypro_fluxnet_files"  # This is the folder where datafiles are searched
    OUTDIR = r"F:\Sync\luhk_work\CURRENT\fru\Level-1_results_fluxnet_2023\1-formatted_for_upload"  # Output yearly CSV to this folder

    # Imports
    import importlib.metadata
    from datetime import datetime
    from diive.pkgs.formats.fluxnet import FormatEddyProFluxnetFileForUpload

    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"This page was last modified on: {dt_string}")
    version_diive = importlib.metadata.version("diive")
    print(f"diive version: v{version_diive}")

    # Show docstring
    print(FormatEddyProFluxnetFileForUpload.__name__)
    print(FormatEddyProFluxnetFileForUpload.__doc__)

    # Initialize
    fxn = FormatEddyProFluxnetFileForUpload(
        site='CH-FRU',
        sourcedir=SOURCE,
        outdir=OUTDIR,
        add_runid=True)

    # Search and merge _fluxnet_ datafiles
    fxn.mergefiles(limit_n_files=1)

    # Merged dataframe
    print(fxn.merged_df)

    # Test for signal strength / AGC
    fxn.remove_low_signal_data(fluxcol='FC',
                               signal_strength_col='CUSTOM_AGC_MEAN',
                               method='discard above',
                               threshold=90)

    # Remove problematic time periods
    fxn.remove_erroneous_data(var='FC',
                              remove_dates=[
                                  '2023-11-01 23:58:15',
                                  ['2023-11-05 00:00:15', '2023-12-07 14:15:00'],
                                  ['2023-06-01', '2023-08-15']
                              ],
                              showplot=True)

    # Format data for FLUXNET
    fxn.apply_fluxnet_format()

    # Save yearly CSV files
    fxn.export_yearly_files()


if __name__ == '__main__':
    example()
