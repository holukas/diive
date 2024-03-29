![](images/logo_diive1_256px.png)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyPI - Version](https://img.shields.io/pypi/v/diive?style=for-the-badge&color=%23EF6C00&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdiive%2F)
![GitHub License](https://img.shields.io/github/license/holukas/diive?style=for-the-badge&color=%237CB342)

# Time series data processing

`diive` is a Python library for time series processing, in particular ecosystem data. Originally developed
for [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/) by the [ETH Grassland Sciences group](https://gl.ethz.ch/).

Recent updates: [CHANGELOG](https://github.com/holukas/diive/blob/main/CHANGELOG.md)   
Recent releases: [Releases](https://github.com/holukas/diive/releases)

First example notebooks can be found in the folder `notebooks`.

More notebooks are added constantly.

## Current Features

### Analyses

- Calculate z-aggregates in quantiles (classes) of x and
  y ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/CalculateZaggregatesInQuantileClassesOfXY.ipynb))
- Daily
  correlation ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/DailyCorrelation.ipynb))
- Decoupling: Sorting bins
  method ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/DecouplingSortingBins.ipynb))
- Find data gaps ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/GapFinder.ipynb))
- Histogram ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/Histogram.ipynb))
- Optimum range
- Percentiles ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/Percentiles.ipynb))

### Corrections

- Offset correction
- Set to threshold
- Wind direction offset detection and
  correction ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Corrections/WindDirectionOffset.ipynb))

### Create variable

- Calculate daytime flag, nighttime flag and potential radiation from latitude and
  longitude ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/CalculateVariable/Daytime_and_nighttime_flag.ipynb))
- Day/night flag from sun angle
- VPD from air temperature and
  RH ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/CalculateVariable/Calculate_VPD_from_TA_and_RH.ipynb))

### Eddy covariance high-resolution

- Flux detection limit from high-resolution data
- Find maximum covariance between turbulent wind and scalar
- Wind rotation to calculate turbulent departures of wind components and scalar (e.g. CO2)

### Files

- Detect expected and unexpected (irregular) files in a list of files
- Split multiple files into smaller parts and export them as (compressed) CSV files

### Fits

- Bin fitter

### Flux

- Critical heat days for NEP, based on air temperature and VPD
- CO2 penalty
- USTAR threshold scenarios

### Flux processing chain

For info about the Swiss FluxNet flux levels,
see [here](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/).

- Flux processing
  chain ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/FluxProcessingChain/FluxProcessingChain.ipynb))
    - The notebook example shows the application of:
        - Level-2 quality flags
        - Level-3.1 storage correction
        - Level-3.2 outlier removal
- Quick flux processing
  chain ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/FluxProcessingChain/QuickFluxProcessingChain.ipynb))

### Formats

Format data to specific formats

- Convert EddyPro fluxnet output files for upload to FLUXNET
  database ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Formats/FormatEddyProFluxnetFileForUpload.ipynb))
- Load and save parquet
  files ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Formats/LoadSaveParquetFile.ipynb))

### Gap-filling

Fill gaps in time series with various methods

- Linear
  interpolation ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/GapFilling/LinearInterpolation.ipynb))
-
RandomForestTS ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/GapFilling/RandomForestGapFilling.ipynb))
- Quick random forest
  gap-filling ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/GapFilling/QuickRandomForestGapFilling.ipynb))

### Outlier Detection

- Absolute limits
- Absolute limits, separately defined for daytime and nighttime data
- Incremental z-score: Identify outliers based on the z-score of increments
- Local standard deviation: Identify outliers based on the local standard deviation from a running median
- Local outlier factor: Identify outliers based on local outlier factor, across all data
- Local outlier factor: Identify outliers based on local outlier factor, daytime nighttime separately
- Manual removal: Remove time periods (from-to) or single records from time series
- Missing values: Simply creates a flag that indicated available and missing data in a time series
- z-score: Identify outliers based on the z-score across all time series data
- z-score: Identify outliers based on the z-score, separately for daytime and nighttime
- z-score: Identify outliers based on max z-scores in the interquartile range data

### Plotting

- Heatmap showing values (z) of time series as date (y) vs time (
  x) ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/HeatmapDateTime.ipynb))
- Heatmap showing values (z) of time series as year (y) vs month (
  x) ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/HeatmapYearMonth.ipynb))
- Long-term anomalies per
  year ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/LongTermAnomalies.ipynb))
- Simple (interactive) time series
  plot ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/TimeSeries.ipynb))
- ScatterXY plot ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/ScatterXY.ipynb))
- Various classes to generate heatmaps, bar plots, time series plots and scatter plots, among others

### Quality control

- Stepwise MeteoScreening from
  database ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/MeteoScreening/StepwiseMeteoScreeningFromDatabase.ipynb))

### Stats

- Time series
  stats ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Stats/TimeSeriesStats.ipynb))

### Timestamps

- Create continuous timestamp based on number of records in the file and the file duration
- Detect time resolution from
  data ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/TimeStamps/Detect_time_resolution.ipynb))
- Insert additional timestamps in various formats

## Installation

`diive` can be installed from source code, e.g. using [`poetry`](https://python-poetry.org/) for dependencies.

`diive` is currently developed under Python 3.9.7, but newer (and many older) versions should also work.

One way to install and use `diive` with a specific Python version on a local machine:

- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Start `miniconda` prompt
- Create a environment named `diive-env` that contains Python 3.9.7:
  `conda create --name diive-env python=3.9.7`
- Activate the new environment: `conda activate diive-env`
- Install `diive` version directly from source code:
  `pip install https://github.com/holukas/diive/archive/refs/tags/v0.63.1.tar.gz` (select .tar.gz file of the desired
  version)
- If you want to use `diive` in Jupyter notebooks, you can install Jupyterlab.
  In this example Jupyterlab is installed from the `conda` distribution channel `conda-forge`:
  `conda install -c conda-forge jupyterlab`
- If used in Jupyter notebooks, `diive` can generate dynamic plots. This requires the installation of:
  `conda install -c bokeh jupyter_bokeh`