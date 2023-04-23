![](images/logo_diive1_512px.png)

# Time series data processing

`diive` is a Python library for time series processing.

Recent updates: [CHANGELOG](CHANGELOG.md)   
Recent releases: [Releases](https://gitlab.ethz.ch/holukas/diive/-/releases)

First example notebooks can be found in the folder `notebooks`.

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

- Day/night flag from sun angle
- VPD from air temperature and RH

### Eddy covariance high-resolution

- Flux detection limit

### Fits

- Bin fitter

### Flux

- Critical days
- NEP penalty
- USTAR threshold scenarios

### Flux processing chain

For info about the Swiss FluxNet flux levels,
see [here](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/).

- Level-2 quality flags
- Level-3.1 storage correction
- Level-3.2 outlier removal

### Gap-filling

- Interpolate
- RandomForestTS

### Outlier Detection

- Absolute limits
- Incremental z-score
- Local standard deviation
- Local outlier factor
- Missing values
- Seasonal trend decomposition using LOESS, z-score on residuals of IQR
- Thymeboost
- Various z-score approaches

### Quality control

- Stepwise MeteoScreening

### Plotting

- Various classes to generate heatmaps, bar plots, time series plots and scatter plots, among others

## Installation

`diive` can be installed from source code, e.g. using [`poetry`](https://python-poetry.org/) for dependencies.