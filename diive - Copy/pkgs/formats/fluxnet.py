"""

Variables shared with FLUXNET:

CO2:
- FC (µmolCO2 m-2 s-1): Carbon Dioxide (CO2) turbulent flux (without storage component)
- FC_SSITC_TEST (adimensional): Quality check - CO2 flux - * see the note
- SC (µmolCO2 m-2 s-1): Carbon Dioxide (CO2) storage flux measured with a vertical profile system, optional if tower shorter than 3 m
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

"""
import os
import re
from pathlib import Path

from diive.core.io.filereader import MultiDataFileReader
from diive.core.times.times import insert_timestamp

VARS_CO2 = ['FC', 'FC_SSITC_TEST', 'SC_SINGLE', 'CO2']
VARS_H2O = ['LE', 'LE_SSITC_TEST', 'SLE_SINGLE', 'H2O']
VARS_H = ['H', 'H_SSITC_TEST', 'SH_SINGLE']
VARS_WIND = ['USTAR', 'WD', 'WS', 'FETCH_70', 'FETCH_90', 'FETCH_MAX']
VARS_METEO = ['SW_IN_1_1_1', 'TA_1_1_1', 'RH_1_1_1', 'PA_1_1_1', 'LW_IN_1_1_1', 'PPFD_IN_1_1_1',
              'G_1_1_1', 'NETRAD_1_1_1', 'TS_1_1_1', 'P_1_1_1', 'SWC_1_1_1']
VARIABLES = VARS_CO2 + VARS_H2O + VARS_H + VARS_WIND + VARS_METEO


class ConvertEddyProFluxnetFileForUpload:
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

    Added in v0.52.0
    """

    def __init__(self,
                 sourcedir: str,
                 outdir: str):
        self.sourcedir = sourcedir
        self.outdir = outdir

        self.data_df = None
        self.subset = None

    def run(self):
        """Convert files to FLUXNET format"""
        self._load_sourcefiles()
        self._make_subset()
        self._missing_values()
        self._rename()
        self._insert_timestamp_columns()
        self._adjust_timestamp_formats()
        self._save_one_file_per_year()

    def get_data(self):
        return self.subset

    def _load_sourcefiles(self):
        """Search for data files, merge data and store to one dataframe"""
        print(f"Searching for EddyPro _fluxnet_ files in folder {self.sourcedir} ...")
        filepaths = [f for f in os.listdir(self.sourcedir) if f.endswith(".csv")]
        # filepaths = [f for f in filepaths if '_eddypro_' in f]
        filepaths = [f for f in filepaths if '_fluxnet_' in f]
        filepaths = [self.sourcedir + "/" + f for f in filepaths]
        filepaths = [Path(f) for f in filepaths]
        print(f"    Found {len(filepaths)} files:")
        [print(f"       --> {f}") for f in filepaths]
        loaddatafile = MultiDataFileReader(filetype='EDDYPRO_FLUXNET_30MIN', filepaths=filepaths)
        self.data_df = loaddatafile.data_df

    def _save_one_file_per_year(self):
        """Save data to yearly files"""
        uniq_years = list(self.subset.index.year.unique())
        for year in uniq_years:
            outname = f"data_for_fluxnet_upload_{year}.csv"
            outpath = Path(self.outdir) / outname
            yearlocs = self.subset.index.year == year
            yeardata = self.subset[yearlocs].copy()
            yeardata.to_csv(outpath, index=False)
            print(f"\n--> Saved file {outpath}.")

    def _missing_values(self):
        """Set all missing values to -9999 as required by FLUXNET"""
        self.subset = self.subset.fillna(-9999)

    def _adjust_timestamp_formats(self):
        """Apply FLUXNET timestamp format (YYYYMMDDhhmm) to timestamp columns (not index)"""
        self.subset['TIMESTAMP_END'] = self.subset['TIMESTAMP_END'].dt.strftime('%Y%m%d%H%M')
        self.subset['TIMESTAMP_START'] = self.subset['TIMESTAMP_START'].dt.strftime('%Y%m%d%H%M')

    def _insert_timestamp_columns(self):
        """Insert timestamp columns denoting start and end of averaging interval"""
        # Add timestamp column TIMESTAMP_END
        self.subset = insert_timestamp(data=self.subset, convention='end', insert_as_first_col=True, verbose=True)
        # Add timestamp column TIMESTAMP_START
        self.subset = insert_timestamp(data=self.subset, convention='start', insert_as_first_col=True, verbose=True)

    def _rename(self):
        """Rename variables to FLUXNET format by adding suffix _1_1_1"""
        renaming_dict = {}
        notrenamed = []
        for var in self.subset.columns:
            # Check whether the variable name ends with a FLUXNET suffix
            has_suffix = re.match('.*_[0-9]_[0-9]_[0-9]$', var)
            if not has_suffix:
                renaming_dict[var] = f"{var}_1_1_1"
            else:
                notrenamed.append(var)
        self.subset = self.subset.rename(columns=renaming_dict)
        print(f"\nThe following variables have been renamed:")
        for ix, val in renaming_dict.items():
            print(f"    RENAMED --> {ix} was renamed to {val}")
        print(f"\nThe following variables have not been renamed:")
        for var in notrenamed:
            print(f"    NOT RENAMED --> {var} was not renamed")

    def _make_subset(self):
        """Make subset that contains variables available for sharing"""
        available_vars = self.data_df.columns
        subsetcols = []
        notavailablecols = []
        for var in VARIABLES:
            subsetcols.append(var) if var in available_vars else notavailablecols.append(var)
        print(f"Found: {subsetcols}")
        print(f"Not found: {notavailablecols}")
        self.subset = self.data_df[subsetcols].copy()


def example():
    # from diive.configs.exampledata import load_exampledata_eddypro_fluxnet_CSV_30MIN
    # data_df, metadata_df = load_exampledata_eddypro_fluxnet_CSV_30MIN()
    SOURCE = r"F:\01-NEW\FF202303\FRU\2022\2-FLUXRUN\Level-1_FR-20230402-163038\2-0_eddypro_flux_calculations\results"
    con = ConvertEddyProFluxnetFileForUpload(
        sourcedir=SOURCE,
        outdir='F:\_temp')
    con.run()
    # data_fluxnet = con.get_data()


if __name__ == '__main__':
    example()
