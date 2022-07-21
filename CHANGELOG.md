# Changelog

![DIIVE](images/logo_diive1_256px.png)

## v0.32.0 | XX XXX 2022

### MeteoScreening Air Temperature
#### General logic
MeteoScreening uses a general settings file `pipes_meteo.yaml` that contains info how
specific `measurements` should be screened. Such `measurements` can be air temperature `TA`
#### `pkgs.qaqc.meteoscreening.ScreenVar`
- Performs quality screening of air temperature `TA`. 
- As first check, I implemented outlier detection via the newly added package `ThymeBoost`,
along with checks for absolute limits.
- Screening applies the checks defined in the file `pipes_meteo.yaml` for the respective
`measurement`, e.g. `TA` for air temperature.
- The screening outputs a separate dataframe that contains `QCF` flags for each check.
- The checks do not change the original time series. Instead, only the flags are generated.
- Screening routines for more variables will be added over the next updates. 
#### `pkgs.qaqc.meteoscreening.MeteoScreeningFromDatabaseSingleVar`
- Performs quality screening *and* resampling to 30MIN of variables downloaded from the database.
- It uses the `detailed` data when downloading data from the database using `dbc-influxdb`.
- The `detailed` data contains the measurement of the variable, along with multiple tags that
describe the data. The tags are needed for storage in the database.
- After quality screening of the original high-resolution data, flagged values are removed and
then data are resampled.
- It also handles the issue that data downloaded for a specific variable can have different time
resolution over the years, although I still need to test this.
- After screening and resampling, data are in a format that can be directly uploaded to the 
database using `dbc-influxdb`.
#### `pkgs.qaqc.meteoscreening.MeteoScreeningFromDatabaseMultipleVars`
- Wrapper where multiple variables can be screened in one run.
- This should also work in combination of different `measurements`. For example, screening
radiation and temperature data in one run.
### Outlier Detection
- Added `pkgs.outlierdetection.thymeboost.thymeboost`
- Added `pkgs.outlierdetection.absolutelimits.absolute_limits`


[//]: # (- optimum range)
[//]: # (- `diive.core.times` `DetectFrequency` )
[//]: # (- `diive.core.times`: `resampling` module )
[//]: # (- New package in env: `ThymeBoost` [GitHub]&#40;https://github.com/tblume1992/ThymeBoost/tree/main/ThymeBoost&#41; )


## v0.31.0 | 4 Apr 2022

### Carbon cost

#### **GENERAL**
- This version introduces the code for calculating carbon cost and critical heat days.

#### **NEW PACKAGES**
- Added new package for flux-specific calculations: `diive.pkgs.flux`

#### **NEW MODULES**
- Added new module for calculating carbon cost: `diive.pkgs.flux.carboncost`
- Added new module for calculating critical heat days: `diive.pkgs.flux.criticalheatdays`

#### **CHANGES & ADDITIONS**
- None

#### **BUGFIXES**
- None


## v0.30.0 | 15 Feb 2022

### Starting diive library

#### **GENERAL**
The `diive` library contains packages and modules that aim to facilitate working
with time series data, in particular ecosystem data.

Previous versions of `diive` included a GUI. The GUI component will from now on
be developed separately as `diive-gui`, which makes use of the `diive` library.

Previous versions of `diive` (up to v0.22.0) can be found in the separate repo 
[diive-legacy](https://gitlab.ethz.ch/diive/diive-legacy).

This initial version of the `diive` library contains several first versions of 
packages that will be extended with the next versions. 

Notable introduction in this version is the package `echires` for working with
high-resolution eddy covariance data. This package contains the module `fluxdetectionlimit`,
which allows the calculation of the flux detection limit following Langford et al. (2015).

#### **NEW PACKAGES**
- Added `common`: Common functionality, e.g. reading data files
- Added `pkgs > analyses`: General analyses
- Added `pkgs > corrections`: Calculate corrections for existing variables
- Added `pkgs > createflag`: Create flag variables, e.g. for quality checks
- Added `pkgs > createvar`: Calculate new variables, e.g. potential radiation
- Added `pkgs > echires`: Calculations for eddy covariance high-resolution data, e.g. 20Hz data
- Added `pkgs > gapfilling`: Gap-filling routines
- Added `pkgs > outlierdetection`: Outlier detection
- Added `pkgs > qaqc`: Quality screening for timeseries variables

#### **NEW MODULES**
- Added `optimumrange` in `pkgs > analyses`
- Added `gapfinder` in `pkgs > analyses`
- Added `offsetcorrection` in `pkgs > corrections`
- Added `setto_threshold` in `pkgs > corrections`
- Added `outsiderange` in `pkgs > createflag`
- Added `potentialradiation` in `pkgs > createvar`
- Added `fluxdetectionlimit` in `pkgs > echires`
- Added `interpolate` in `pkgs > gapfilling`
- Added `hampel` in `pkgs > outlierdetection`
- Added `meteoscreening` in `pkgs > qaqc`

#### **CHANGES & ADDITIONS**
- None

#### **BUGFIXES**
- None

#### **REFERENCES**
Langford, B., Acton, W., Ammann, C., Valach, A., & Nemitz, E. (2015). Eddy-covariance data with low signal-to-noise ratio: Time-lag determination, uncertainties and limit of detection. Atmospheric Measurement Techniques, 8(10), 4197–4213. https://doi.org/10.5194/amt-8-4197-2015
