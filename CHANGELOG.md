# Changelog

![DIIVE](images/logo_diive1_256px.png)

## v0.38.0 | 3 Sep 2022

### Optimum Range Detection

- New analysis: `pkgs.analyses.optimumrange.FindOptimumRange` finds the optimum for a
  variable in binned other variable. This is useful for e.g. detecting the air temperature
  range where CO2 uptake was highest (=most negative).  
  ![DIIVE](images/analysesOptimumRange_diive_v0.38.0.png)

## v0.37.0 | 2 Sep 2022

### Cumulative and Anomaly Plots

- New plot: `core.plotting.cumulative.CumulativeYear` plots cumulative sums per year  
  ![DIIVE](images/plotCumulativeYear_diive_v0.37.0.png)
- New plot: `core.plotting.bar.LongtermAnomaliesYear` plots yearly anomalies in relation to a reference period  
  ![DIIVE](images/plotBarLongtermAnomaliesYear_diive_v0.37.0.png)
- Refactored various code bits for plotting

## v0.36.0 | 27 Aug 2022

### Random Forest Update

- Refactored code for `pkgs/gapfilling/randomforest_ts.py`
    - Implemented lagged variants of variables
    - Implemented long-term gap-filling, where the model to gap-fill a specific year is built from the
      respective year and its neighboring years
    - Implemented feature reduction using sklearn's RFECV
    - Implemented TimeSeriesSplit used as the cross-validation splitting strategy during feature reduction
- Implemented `TimestampSanitizer` also when reading from file with `core.io.filereader.DataFileReader`
- Removed old code in `.core.dfun.files` and moved files logistics to `.core.io.files` instead
- Implemented saving and loading Python `pickles` in `.core.io.files`

## v0.35.0 | 19 Aug 2022

### Meteoscreening PA, RH

- Added function `pkgs.corrections.offsetcorrection.remove_relativehumidity_offset` to correct
  humidity measurements for values > 100%

### Other

- Added first code for outlier detection via seasonal trends in `pkgs/outlierdetection/seasonaltrend.py`
- Prepared `pkgs/analyses/optimumrange.py` for future updates

## v0.34.0 | 29 Jul 2022

### MeteoScreening Radiation

#### MeteoScreening

- Implemented corrections and quality screening for radiation data in `pkgs.qaqc.meteoscreening`

#### Corrections

Additions to `pkgs.corrections`:

- Added function `.offsetcorrection.remove_radiation_zero_offset` to correct radiation
  data for nighttime offsets
- Added function `.setto_threshold.setto_threshold` to set values above or below a
  specfied threshold value to the threshold.

#### Plotting

- Added function `core.plotting.plotfuncs.quickplot` for quickly plotting pandas
  Series and DataFrame data

#### Resampling

- Implemented `TimeSanitizer` in `core.times.resampling.resample_series_to_30MIN`

#### Other

- Added decorator class `core.utils.prints.ConsoleOutputDecorator`, a wrapper to
  execute functions with additional info that is output to the console.

## v0.33.0 | 26 Jul 2022

### MeteoScreening Preparations

- Added new class `core.times.times.TimestampSanitizer`
    - Class that handles timestamp checks and fixes, such as the creation of a continuous
      timestamp without date gaps.
- Added `pkgs.createvar.nighttime_latlon.nighttime_flag_from_latlon`
    - Function for the calculation of a nighttime flag (1=nighttime) from latitude and
      longitude coordinates of a specific location.
- Added `core.plotting.heatmap_datetime.HeatmapDateTime`
    - Class to generate a heatmap plot from timeseries data.

## v0.32.0 | 22 Jul 2022

### MeteoScreening Air Temperature

MeteoScreening uses a general settings file `pipes_meteo.yaml` that contains info how
specific `measurements` should be screened. Such `measurements` group similar variables
together, e.g. different air temperatures are measurement `TA`.   
Additions to module `pkgs.qaqc.meteoscreening`:

- Added class `ScreenVar`
    - Performs quality screening of air temperature `TA`.
    - As first check, I implemented outlier detection via the newly added package `ThymeBoost`,
      along with checks for absolute limits.
    - Screening applies the checks defined in the file `pipes_meteo.yaml` for the respective
      `measurement`, e.g. `TA` for air temperature.
    - The screening outputs a separate dataframe that contains `QCF` flags for each check.
    - The checks do not change the original time series. Instead, only the flags are generated.
    - Screening routines for more variables will be added over the next updates.
- Added class `MeteoScreeningFromDatabaseSingleVar`
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
- Added class `MeteoScreeningFromDatabaseMultipleVars`
    - Wrapper where multiple variables can be screened in one run.
    - This should also work in combination of different `measurements`. For example, screening
      radiation and temperature data in one run.

### Outlier Detection

Additions to `pkgs.outlierdetection`:

- Added module `thymeboost`
- Added module `absolute_limits`

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

Langford, B., Acton, W., Ammann, C., Valach, A., & Nemitz, E. (2015). Eddy-covariance data with low signal-to-noise
ratio: Time-lag determination, uncertainties and limit of detection. Atmospheric Measurement Techniques, 8(10),
4197–4213. https://doi.org/10.5194/amt-8-4197-2015
