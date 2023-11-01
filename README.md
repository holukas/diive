![](images/logo_diive1_512px.png)

# Time series data processing

`diive` is a Python library for time series processing.

Recent updates: [CHANGELOG](CHANGELOG.md)   
Recent releases: [Releases](https://github.com/holukas/diive/releases)

First example notebooks can be found in the folder `notebooks`.

More notebooks are added constantly.

## Current Features

### Analyses

- Decoupling
- Detect time resolution from data ([notebook example](notebooks/TimeStamps/Detect_time_resolution.ipynb))
- Find data gaps ([notebook example](notebooks/Analyses/GapFinder.ipynb))
- Histogram
- Optimum range
- Quantiles

### Corrections

- Offset correction
- Set to threshold
- Wind direction offset detection and correction ([notebook example](notebooks/Corrections/WindDirectionOffset.ipynb))

### Create variable

- Calculate daytime flag, nighttime flag and potential radiation from latitude and longitude ([notebook example](notebooks/CalculateVariable/Daytime_and_nighttime_flag.ipynb))
- Day/night flag from sun angle
- VPD from air temperature and RH ([notebook example](notebooks/CalculateVariable/Calculate_VPD_from_TA_and_RH.ipynb))

### Eddy covariance high-resolution

- Flux detection limit from high-resolution data

### Formats

- Convert EddyPro fluxnet output files for upload to FLUXNET
  database ([notebook example](notebooks/Formats/FormatEddyProFluxnetFileForUpload.ipynb))
- Load and save parquet files ([notebook example](notebooks/Formats/LoadSaveParquetFile.ipynb))

### Fits

- Bin fitter

### Flux

- Critical heat days for NEP, based on air temperature and VPD
- CO2 penalty
- USTAR threshold scenarios

### Flux processing chain

For info about the Swiss FluxNet flux levels,
see [here](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/).

- Flux processing chain ([notebook example](notebooks/FluxProcessingChain/FluxProcessingChain.ipynb))
    - The notebook example shows the application of:
        - Level-2 quality flags
        - Level-3.1 storage correction
        - Level-3.2 outlier removal

### Formats

Format data to specific formats

- Format EddyPro _fluxnet_ output file for upload to FLUXNET
  database ([notebook example](notebooks/Formats/FormatEddyProFluxnetFileForUpload.ipynb))

### Gap-filling

Fill gaps in time series with various methods

- Interpolate
- RandomForestTS ([notebook example](notebooks/GapFilling/RandomForestGapFilling.ipynb))
- Quick random forest gap-filling ([notebook example](notebooks/GapFilling/QuickRandomForestGapFilling.ipynb))

### Outlier Detection

- Absolute limits
- Absolute limits, separately defined for daytime and nighttime data
- Incremental z-score: Identify outliers based on the z-score of increments
- Local standard deviation: Identify outliers based on the local standard deviation from a running median
- Local outlier factor: Identify outliers based on local outlier factor, across all data
- Local outlier factor: Identify outliers based on local outlier factor, daytime nighttime separately
- Manual removal: Remove time periods (from-to) or single records from time series
- Missing values: Simply creates a flag that indicated available and missing data in a time series
- Seasonal trend decomposition using LOESS, identify outliers based on seasonal-trend decomposition and
  z-score calculations
- Thymeboost: Identify outliers based on [thymeboost](https://github.com/tblume1992/ThymeBoost)
- z-score: Identify outliers based on the z-score across all time series data
- z-score: Identify outliers based on the z-score, separately for daytime and nighttime
- z-score: Identify outliers based on max z-scores in the interquartile range data

### Plotting

- Simple (interactive) time series plot ([notebook example](notebooks/Plotting/TimeSeries.ipynb))
- Various classes to generate heatmaps, bar plots, time series plots and scatter plots, among others

### Quality control

- Stepwise MeteoScreening from
  database ([notebook example](notebooks/MeteoScreening/StepwiseMeteoScreeningFromDatabase.ipynb))

### Stats

- Time series stats ([notebook example](notebooks/Stats/TimeSeriesStats.ipynb))

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
  `pip install https://github.com/holukas/diive/archive/refs/tags/v0.63.1.tar.gz`
- If you want to use `diive` in Jupyter notebooks, you can install Jupyterlab.
  In this example Jupyterlab is installed from the `conda` distribution channel `conda-forge`:
  `conda install -c conda-forge jupyterlab`
- If used in Jupyter notebooks, `diive` can generate dynamic plots. This requires the installation of:
  `conda install -c bokeh jupyter_bokeh`