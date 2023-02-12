![](images/logo_diive1_512px.png)

# Time series data processing

**D**ata - **I**mport - **I**nvestigate - **V**isualize - **E**xport

`diive` is a Python library for time series processing.

Recent updates: [CHANGELOG](CHANGELOG.md)   
Recent releases: [Releases](https://gitlab.ethz.ch/holukas/diive/-/releases)

## Current Features

### Analyses

- Decoupling
- Gapfinder
- Histogram
- Optimum range
- Quantiles

### Corrections

- Offset correction
- Set to threshold

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