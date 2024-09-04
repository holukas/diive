![](images/logo_diive1_256px.png)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyPI - Version](https://img.shields.io/pypi/v/diive?style=for-the-badge&color=%23EF6C00&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdiive%2F)
![GitHub License](https://img.shields.io/github/license/holukas/diive?style=for-the-badge&color=%237CB342)

[![DOI](https://zenodo.org/badge/708559210.svg)](https://zenodo.org/doi/10.5281/zenodo.10884017)

*`diive` is currently under active developement with frequent updates.*

# Time series data processing

`diive` is a Python library for time series processing, in particular ecosystem data. Originally developed
by the [ETH Grassland Sciences group](https://gl.ethz.ch/) for [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/).

Recent updates: [CHANGELOG](https://github.com/holukas/diive/blob/main/CHANGELOG.md)   
Recent releases: [Releases](https://github.com/holukas/diive/releases)

## Overview of example notebooks

- For many examples see notebooks here: [Notebook overview](https://github.com/holukas/diive/blob/main/notebooks/OVERVIEW.ipynb)
- More notebooks are added constantly.

## Current Features

### Analyses

- **Daily correlation**: calculate daily correlation between two time series ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/DailyCorrelation.ipynb))
- **Decoupling**: Investigate binned aggregates (median) of a variable z in binned classes of x and y ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/DecouplingSortingBins.ipynb))
- **Quantile aggregation**: calculate z-aggregates in quantiles (classes) of x and y ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/CalculateZaggregatesInQuantileClassesOfXY.ipynb))
- **Data gaps identification**: ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/GapFinder.ipynb))
- **Histogram calculation**: calculate histogram from Series ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/Histogram.ipynb))
- **Optimum range**: find x range for optimum y
- **Percentiles**: Calculate percentiles 0-100 for series ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/Percentiles.ipynb))

### Corrections

- **Offset correction radiation**: correct nighttime offset of radiation data and set nighttime to zero
- **Offset correction relative humidity**: correct RH values > 100%
- **Offset correction wind direction**: correct wind directions by offset, calculated based on reference time period ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Corrections/WindDirectionOffset.ipynb))
- **Set to threshold**: set values above or below a threshold value to threshold value

### Create variable

_Functions to create various variables._

- **Time since**: calculate time since last occurrence, e.g. since last precipitation ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/CalculateVariable/TimeSince.ipynb))
- **Daytime/nighttime flag**: calculate daytime flag, nighttime flag and potential radiation from latitude and longitude ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/CalculateVariable/Daytime_and_nighttime_flag.ipynb))
- **Vapor pressure deficit**: calculate VPD from air temperature and RH ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/CalculateVariable/Calculate_VPD_from_TA_and_RH.ipynb))

### Eddy covariance high-resolution

- **Flux detection limit**: calculate flux detection limit from high-resolution data (20 Hz)
- **Maximum covariance**: find maximum covariance between turbulent wind and scalar
- **Turbulence**: wind rotation to calculate turbulent departures of wind components and scalar (e.g. CO2)

### Files

_Input/output functions._

- **Detect files**: detect expected and unexpected (irregular) files in a list of files
- **Split files**: split multiple files into smaller parts and export them as (compressed) CSV files
- **Read single data files**: read file using parameters ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/ReadFiles/Read_single_EddyPro_fluxnet_output_file_with_DataFileReader.ipynb))
- **Read single data files**: read file using pre-defined filetypes ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/ReadFiles/Read_single_EddyPro_fluxnet_output_file_with_ReadFileType.ipynb))
- **Read multiple data files**: read files using pre-defined filetype ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/ReadFiles/Read_multiple_EddyPro_fluxnet_output_files_with_MultiDataFileReader.ipynb))

### Fits

- **Bin fitter**

### Flux

_Specific analyses of eddy covariance flux data._

- **USTAR threshold scenarios**: display data availability under different USTAR threshold scenarios

### Flux processing chain

_Post-processing of eddy covariance flux data._
_For info about the Swiss FluxNet flux levels, see [here](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/)._

- Flux processing chain ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/FluxProcessingChain/FluxProcessingChain.ipynb))
    - The notebook example shows the application of:
        - Level-2 quality flags
        - Level-3.1 storage correction
        - Level-3.2 outlier removal
- Quick flux processing chain ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/FluxProcessingChain/QuickFluxProcessingChain.ipynb))

### Formats

_Format data to specific formats._

- **Format**: convert EddyPro fluxnet output files for upload to FLUXNET database ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Formats/FormatEddyProFluxnetFileForUpload.ipynb))
- **Parquet files**: load and save parquet files ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Formats/LoadSaveParquetFile.ipynb))

### Gap-filling

_Fill gaps in time series with various methods._

- **XGBoostTS** ([notebook example (minimal)](https://github.com/holukas/diive/blob/main/notebooks/GapFilling/XGBoostGapFillingMinimal.ipynb), [notebook example (more extensive)](https://github.com/holukas/diive/blob/main/notebooks/GapFilling/XGBoostGapFillingExtensive.ipynb))
- **RandomForestTS** ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/GapFilling/RandomForestGapFilling.ipynb))
- **Linear interpolation** ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/GapFilling/LinearInterpolation.ipynb))
- **Quick random forest gap-filling** ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/GapFilling/QuickRandomForestGapFilling.ipynb))

### Outlier Detection

#### Multiple tests combined

- **Step-wise outlier detection**: combine multiple outlier flags to one single overall flag

#### Single tests

_Create single outlier flags where `0=OK` and `2=outlier`._

- **Absolute limits**: define absolute limits ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/AbsoluteLimits.ipynb))
- **Absolute limits daytime/nighttime**: define absolute limits separately for daytime and nighttime data ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/AbsoluteLimitsDaytimeNighttime.ipynb))
- **Hampel filter**: based on Median Absolute Deviation (MAD) in a moving window ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/Hampel.ipynb))
- **Hampel filter daytime/nighttime**, separately for daytime and nighttime data ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/HampelDaytimeNighttime.ipynb))
- **Local standard deviation**: Identify outliers based on the local standard deviation from a running median ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/LocalSD.ipynb))
- **Local outlier factor**: Identify outliers based on local outlier factor, across all data ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/LocalSD.ipynb))
- **Local outlier factor daytime/nighttime**: Identify outliers based on local outlier factor, daytime nighttime separately ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/LocalOutlierFactorDaytimeNighttime.ipynb))
- **Manual removal**: Remove time periods (from-to) or single records from time series ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/ManualRemoval.ipynb))
- **Missing values**: Simply creates a flag that indicated available and missing data in a time series ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/MissingValues.ipynb))
- **Trimming**: Remove values below threshold and remove an equal amount of records from high end of data ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/TrimLow.ipynb))
- **z-score**: Identify outliers based on the z-score across all time series data ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/zScore.ipynb))
- **z-score increments daytime/nighttime**: Identify outliers based on the z-score of double increments ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/zScoreIncremental.ipynb))
- **z-score daytime/nighttime**: Identify outliers based on the z-score, separately for daytime and nighttime ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/OutlierDetection/zScoreDaytimeNighttime.ipynb))
- **z-score rolling**: Identify outliers based on the rolling z-score

### Plotting

- **Diel cycle per month** ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/DielCycle.ipynb))
- **Heatmap date/time**: showing values (z) of time series as date (y) vs time (x) ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/HeatmapDateTime.ipynb))
- **Heatmap year/month**: showing values (z) of time series as year (y) vs month (x) ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/HeatmapYearMonth.ipynb))
- **Histogram**: includes options to show z-score limits and to highlight the peak distribution bin ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/Histogram.ipynb))
- **Long-term anomalies**: calculate and plot long-term anomaly for a variable, per year, compared to a reference period. ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/LongTermAnomalies.ipynb))
- **Time series plot**: Simple (interactive) time series plot ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/TimeSeries.ipynb))
- **ScatterXY plot** ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Plotting/ScatterXY.ipynb))
- Various classes to generate heatmaps, bar plots, time series plots and scatter plots, among others

### Quality control

- **Stepwise MeteoScreening from database** ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/MeteoScreening/StepwiseMeteoScreeningFromDatabase.ipynb))

### Resampling

- **Diel cycle**: calculate diel cycle per month ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Resampling/ResamplingDielCycle.ipynb))

### Stats

- **Time series stats** ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Stats/TimeSeriesStats.ipynb))

### Timestamps

- **Continuous timestamp**: create continuous timestamp based on number of records in the file and the file duration
- **Time resolution**: detect time resolution from data ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/TimeStamps/Detect_time_resolution.ipynb))
- **Timestamps**: create and insert additional timestamps in various formats

## Installation

`diive` is currently under active developement using Python 3.9.7, but newer (and many older) versions should also work.

### Using pip

`pip install diive`

### Using poetry

`poetry add diive`

### Using conda

`conda intall -c conda-forge diive`

### From source

Directly use .tar.gz file of the desired version.

`pip install https://github.com/holukas/diive/archive/refs/tags/v0.76.2.tar.gz`

### Create and use a conda environment for diive

One way to install and use `diive` with a specific Python version on a local machine:

- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Start `miniconda` prompt
- Create a environment named `diive-env` that contains Python 3.9.7: `conda create --name diive-env python=3.9.7`
- Activate the new environment: `conda activate diive-env`
- Install `diive` using pip: `pip install diive`
- If you want to use `diive` in Jupyter notebooks, you can install Jupyterlab.
  In this example Jupyterlab is installed from the `conda` distribution channel `conda-forge`:
  `conda install -c conda-forge jupyterlab`
- If used in Jupyter notebooks, `diive` can generate dynamic plots. This requires the installation of:
  `conda install -c bokeh jupyter_bokeh`
