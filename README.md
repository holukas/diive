![](images/logo_diive1_512px.png)

# Post-processing for time series data

**D**ata - **I**mport - **I**nvestigate - **V**isualize - **E**xport

`DIIVE` is a GUI-based Python application that aims to facilitate
working with time series data. 

The app is still in alpha phase, which means that new features
are added (and sometimes removed) frequently (as needed) and bugs can be expected.

Recent updates: [CHANGELOG](CHANGELOG.md)   
Recent releases: [Releases](https://gitlab.ethz.ch/holukas/diive/-/releases)

## Current Features

### Plots
- Correlation Matrix
- Cumulative
- Diel Cycles
- Heatmap / Fingerprint
- Hexbins
- Histogram
- Multipanel
- Quantiles
- Scatter
- Wind Sectors

### Outlier Detection
- Absolute Limits
- Double-differenced Time Series
- Hampel  
- Interquartile Range
- Running
- Trim

### Analyses
- Aggregator
- Class Finder
- Feature Selection
- Gap Finder

### Gap-filling
- Look-up Table (Marginal Distribution Sampling)
- Random Forest
- Simple Running

### Modifications
- Limit Dataset Range
- Remove Time Range
- Rename Variables
- Span Correction (experimental)
- Subset

### Create Variable
- Add New Event
- Apply Gain
- Binning
- Combine Columns
- Define Seasons
- Lag Features  
- Time Since

### Eddy Covariance
- Flux Quality Control

### Export
- Export Dataset

## Installation

If you want to try out DIIVE, you can download the source code of one of the releases and
start it in a conda environment. Miniconda is the minmal installer for conda. Conda is an open-source
package management system for Python (and others).

1. Download and install Miniconda from here: [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Download DIIVE source code from here: [DIIVE Releases](https://gitlab.ethz.ch/holukas/diive/-/releases)
3. In the downloaded DIIVE folder, use the file ```environment.yml``` to create the conda environment for DIIVE.  
4. In conda, activate the conda environment for DIIVE and run
```python <diive-folder>\src\main\python start.py```



There is also a compiled version for Windows available (please contact me directly). 

