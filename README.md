![](images/logo_diive1_256px.png)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/diive?style=for-the-badge&color=%23EF6C00&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdiive%2F)](https://pypi.org/project/diive/)
[![GitHub License](https://img.shields.io/github/license/holukas/diive?style=for-the-badge&color=%237CB342)](https://github.com/holukas/diive/blob/indev/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/diive)](https://pepy.tech/projects/diive)
[![DOI](https://zenodo.org/badge/708559210.svg)](https://zenodo.org/doi/10.5281/zenodo.10884017)

*`diive` is currently under active developement with frequent updates.*

# Time series data processing

`diive` is a Python library for time series processing, in particular ecosystem data. Originally developed
by the [ETH Grassland Sciences group](https://gl.ethz.ch/) for [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/).

Recent updates: [CHANGELOG](https://github.com/holukas/diive/blob/main/CHANGELOG.md)
Recent releases: [Releases](https://github.com/holukas/diive/releases)

---

## Quick Access to Common Classes

All frequently-used classes are available directly from the `diive` namespace for convenient access in notebooks and scripts:

### Interactive use (notebooks)
```python
import diive as dv

# Plotting
plot = dv.timeseries(series=data)
plot = dv.cumulative(series=data)
plot = dv.dielcycle(series=data)

# Gap-filling
rf_model = dv.randomforest_ts(input_df=df, target_col='NEE')
xgb_model = dv.xgboost_ts(input_df=df, target_col='NEE')
quick_fill = dv.quickfillrfts(input_df=df, target_col='NEE')

# Analysis
grid = dv.gridaggregator(x=x_series, y=y_series, z=z_series)
decomp = dv.seasonaltrend(series=data)
```

### Explicit use (production code)
```python
from diive import TimeSeries, RandomForestTS, GridAggregator

plot = TimeSeries(series=data)
model = RandomForestTS(input_df=df, target_col='NEE')
grid = GridAggregator(x=x_series, y=y_series, z=z_series)
```

### Available exports

**Plotting:** `timeseries`, `TimeSeries`, `cumulative`, `Cumulative`, `dielcycle`, `DielCycle`, `heatmapdatetime`, `HeatmapDateTime`, and more

**Gap-filling:** `randomforest_ts`, `RandomForestTS`, `xgboost_ts`, `XGBoostTS`, `quickfillrfts`, `QuickFillRFTS`, `fluxmds`, `FluxMDS`

**Analysis:** `gridaggregator`, `GridAggregator`, `seasonaltrend`, `SeasonalTrendDecomposition`

**I/O:** `load_parquet`, `save_parquet`, `load_exampledata_parquet`, `search_files`

For the complete list of available aliases, see `diive.__all__`.

---

## Examples

Executable example scripts demonstrating common workflows are available in the `examples/` folder:

```bash
python examples/visualization/heatmap_datetime.py    # Heatmap visualization examples
python examples/gap_filling/randomforest_ts.py       # Gap-filling workflows (Phase 2)
```

See [examples/README.md](examples/README.md) for a complete index of all available examples organized by topic (visualization, gap-filling, feature engineering).

Additional examples available in **Jupyter notebooks** at [notebooks/](notebooks/) with comprehensive workflows and tutorials.

---

## Package Structure

```
diive/
├── core/               # Foundational utilities shared across the library
│   ├── base/           # FlagBase — base class for quality and outlier flags
│   ├── dfun/           # DataFrame helpers: stats, regression, bin fitting
│   ├── funcs/          # Miscellaneous utility functions
│   ├── io/             # File detection, reading (CSV, EddyPro, TOA5), parquet I/O
│   ├── ml/             # MlRegressorGapFillingBase — base class for RF/XGBoost gap-filling
│   ├── plotting/       # Heatmaps, time series, scatter, histograms, ridge lines, cumulatives
│   ├── times/          # Timestamp sanitization, frequency detection, vectorization, resampling
│   └── utils/          # Helper utilities
│
└── pkgs/               # Domain-specific algorithms
    ├── analyses/        # Correlation, GridAggregator, GapFinder, decoupling, quantiles
    ├── binary/          # Binary-encoded value extraction
    ├── corrections/     # Offset, radiation, RH, wind direction corrections
    ├── createvar/       # DaytimeNighttimeFlag, VPD, ET, TimeSince, potential radiation
    ├── echires/         # High-resolution eddy covariance: FluxDetectionLimit, WindRotation2D
    ├── fits/            # BinFitterCP
    ├── flux/            # USTAR thresholds, self-heating correction, flux uncertainty
    ├── fluxprocessingchain/  # Orchestrated Level-2 through Level-4 flux workflows
    ├── formats/         # FLUXNET and EddyPro file format conversions
    ├── gapfilling/      # XGBoostTS, RandomForestTS, long-term multi-year gap-filling, FluxMDS, linear interpolation
    ├── outlierdetection/# Hampel, z-score, LOF, absolute limits, stepwise detection
    └── qaqc/            # FlagQCF, EddyPro flags, StepwiseMeteoScreeningDb
```

| Package                          | Key classes / functions                                                                                                                       | Description                                                                                                                              |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `diive.core.base`                | `FlagBase`                                                                                                                                    | Base class for building quality and outlier flags; provides flag encoding, filtering, and visualization                                  |
| `diive.core.ml`                  | `FeatureEngineer`, `MlRegressorGapFillingBase`                                                                                               | Standalone feature engineering (8-stage pipeline) and base class for ML gap-filling (RF, XGBoost); separate feature engineering from model training for better reusability |
| `diive.core.io`                  | `DataFileReader`, `MultiDataFileReader`, `ReadFileType`, `FileSplitter`                                                                       | Read single or multiple instrument files (CSV, EddyPro, TOA5); detect file structure; split large files; load/save Parquet               |
| `diive.core.plotting`            | `HeatmapDateTime`, `HeatmapXYZ`, `HexbinPlot`, `TimeSeries`, `ScatterXY`, `HistogramPlot`, `DielCycle`, `RidgeLinePlot`, `CumulativeYear`   | Comprehensive visualization suite covering heatmaps, time series, scatter, histograms, diurnal cycles, ridge lines, hexbin plots, and cumulative plots |
| `diive.core.times`               | `TimestampSanitizer`, `DetectFrequency`, `vectorize_timestamps()`, `continuous_timestamp_freq()`                                              | Sanitize and validate timestamps, detect/infer data frequency, vectorize time attributes, resample diel cycles                           |
| `diive.core.dfun`                | `sstats()`, `fit_to_bins_linreg()`, `fit_to_bins_polyreg()`                                                                                   | DataFrame statistics, linear/polynomial bin fitting, regression utilities                                                                |
| `diive.pkgs.gapfilling`          | `XGBoostTS`, `RandomForestTS`, `QuickFillRFTS`, `LongTermGapFillingRandomForestTS`, `LongTermGapFillingXGBoostTS`, `FluxMDS`               | Fill time series gaps with XGBoost, Random Forest (standard and long-term multi-year), MDS, or linear interpolation                      |
| `diive.pkgs.outlierdetection`    | `HampelDaytimeNighttime`, `zScore`, `zScoreDaytimeNighttime`, `LocalOutlierFactorAllData`, `AbsoluteLimits`, `AbsoluteLimitsDaytimeNighttime` | Detect and flag outliers using Hampel filter, z-score, LOF, absolute limits, local SD, manual removal, or stepwise combinations          |
| `diive.pkgs.flux`                | `FluxProcessingChain`                                                                                                                         | Post-process eddy covariance fluxes: Level-2 quality flags, storage correction, USTAR filtering, gap-filling (RF/XGBoost/MDS), self-heating correction |
| `diive.pkgs.fluxprocessingchain` | `FluxProcessingChain`                                                                                                                         | Orchestrate a complete Level-2 → Level-4 flux processing workflow in a single pipeline                                                   |
| `diive.pkgs.analyses`            | `GapFinder`, `GridAggregator`, `daily_correlation()`, `SeasonalTrendDecomposition`                                                         | Locate data gaps, aggregate variables into 2-D grids, compute daily correlations, decoupling analysis, quantiles, seasonal-trend decomposition |
| `diive.pkgs.corrections`         | `OffsetCorrection`, `WindDirectionOffset`, `SetToThreshold`, `SetToMissing`                                                                   | Apply measurement offsets, correct wind directions, clamp values to thresholds, set periods to missing                                   |
| `diive.pkgs.createvar`           | `DaytimeNighttimeFlag`, `TimeSince`, `calc_vpd_from_ta_rh()`, `et_from_le()`, `potrad()`                                                      | Derive new variables: daytime/nighttime flags, VPD, ET, time-since-event, potential radiation                                            |
| `diive.pkgs.qaqc`                | `FlagQCF`, `StepwiseMeteoScreeningDb`                                                                                                         | Manage FLUXNET quality control flags; apply stepwise meteorological screening                                                            |
| `diive.pkgs.echires`             | `FluxDetectionLimit`, `WindRotation2D`, `MaxCovariance`                                                                                       | Process 20 Hz eddy covariance data: detection limits, 2-D wind rotation, maximum covariance lag                                          |
| `diive.pkgs.formats`             | `FormatEddyProFluxnetFileForUpload`, `FormatMeteoForEddyProFluxProcessing`                                                                    | Convert EddyPro output to FLUXNET submission format; prepare meteorological data for EddyPro                                             |
| `diive.pkgs.fits`                | `BinFitterCP`                                                                                                                                 | Fit data to bins using cumulative-probability approach                                                                                   |

---

## Overview of example notebooks

- For many examples see notebooks
  here: [Notebook overview](https://github.com/holukas/diive/blob/main/notebooks/OVERVIEW.ipynb)
- More notebooks are added constantly.

## Current Features

### Analyses

- **Daily correlation**: calculate daily correlation between two time
  series · func:
  `daily_correlation()` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/analyses/DailyCorrelation.ipynb))
- **Decoupling**: Investigate binned aggregates (median) of a variable z in binned classes of x and
  y ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/analyses/DecouplingSortingBins.ipynb))
- **Data gaps identification** · class:
  `GapFinder` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/analyses/GapFinder.ipynb))
- **Grid aggregator**: calculate z-aggregates in bins (classes) of x and
  y · class:
  `GridAggregator` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/analyses/GridAggregator.ipynb))
- **Histogram calculation**: calculate histogram from
  Series ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/analyses/Histogram.ipynb))
- **Optimum range**: find x range for optimum y
- **Percentiles**: Calculate percentiles 0-100 for
  series ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/analyses/Percentiles.ipynb))
- **Seasonal-Trend Decomposition**: Separate time series into trend, seasonal, and residual components using STL (Seasonal-Trend Loess), classical, or harmonic methods · class:
  `SeasonalTrendDecomposition` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/Analyses/SeasonalTrendDecomposition.ipynb))
  
### Corrections

- **Offset correction for measurement**: correct measurement by offset in comparison to
  replicate · class:
  `OffsetCorrection` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/MeasurementOffset.ipynb))
- **Offset correction radiation**: correct nighttime offset of radiation data and set nighttime to zero
- **Offset correction relative humidity**: correct RH values > 100%
- **Offset correction wind direction**: correct wind directions by offset, calculated based on reference time
  period · class:
  `WindDirectionOffset` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/WindDirectionOffset.ipynb))
- **Set to threshold**: set values above or below a threshold value to threshold value · class: `SetToThreshold`
- **Set exact values to missing**: set exact values to missing
  records · class:
  `SetToMissing` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/SetExactValuesToMissing.ipynb))

### Create variable

_Functions to create various variables._

- **Time since**: calculate time since last occurrence, e.g. since last
  precipitation · class:
  `TimeSince` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/variables/TimeSince.ipynb))
- **Daytime/nighttime flag**: calculate daytime flag, nighttime flag and potential radiation from latitude and
  longitude · class:
  `DaytimeNighttimeFlag` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/variables/Daytime_and_nighttime_flag.ipynb))
- **Vapor pressure deficit**: calculate VPD from air temperature and
  RH · func:
  `calc_vpd_from_ta_rh()` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/variables/Calculate_VPD_from_TA_and_RH.ipynb))
- **Calculate ET from LE**: calculate evapotranspiration from latent heat
  flux · func:
  `et_from_le()` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/variables/Calculate_ET_from_LE.ipynb))
- **Calculate air temperature from sonic anemometer temperature** · func:
  `air_temp_from_sonic_temp()` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/variables/Calculate_air_temp_from_sonic_temp.ipynb))

### Eddy covariance high-resolution

- **Flux detection limit**: calculate flux detection limit from high-resolution data (20 Hz) · class:
  `FluxDetectionLimit`
- **Maximum covariance**: find maximum covariance between turbulent wind and scalar · class: `MaxCovariance`
- **Turbulence**: wind rotation to calculate turbulent departures of wind components and scalar (e.g. CO2) · class:
  `WindRotation2D`

### Files

_Input/output functions._

- **Detect files**: detect expected and unexpected (irregular) files in a list of files · class: `FileDetector`
- **Split files**: split multiple files into smaller parts and export them as (compressed) CSV files · class:
  `FileSplitter`
- **Read single data files**: read file using
  parameters · class:
  `DataFileReader` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/io/Read_single_EddyPro_fluxnet_output_file_with_DataFileReader.ipynb))
- **Read single data files**: read file using pre-defined
  filetypes · class:
  `ReadFileType` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/io/Read_single_EddyPro_fluxnet_output_file_with_ReadFileType.ipynb))
- **Read multiple data files**: read files using pre-defined
  filetype · class:
  `MultiDataFileReader` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/io/Read_multiple_EddyPro_fluxnet_output_files_with_MultiDataFileReader.ipynb))

### Fits

- **Bin fitter** · class:
  `BinFitterCP` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/analyses/BinFitterCP.ipynb))

### Flux

_Function specifically for eddy covariance flux data._

- **Flux processing chain** · class:
  `FluxProcessingChain` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/flux/FluxProcessingChain.ipynb))
    - The notebook example shows the application of:
        - _Post-processing of eddy covariance flux data._
        - Level-2 quality flags
        - Level-3.1 storage correction
        - Level-3.2 outlier removal
        - Level-3.3: USTAR filtering using constant thresholds
        - Level-4.1: gap-filling using long-term random forest, XGBoost, and/or MDS
        - _For info about the Swiss FluxNet flux levels,
          see [here](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/)._
- **Quick flux processing chain
  ** ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/flux/QuickFluxProcessingChain.ipynb))
- **Flux detection limit**: calculate flux detection limit from high-resolution eddy covariance
  data · class:
  `FluxDetectionLimit` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/flux/FluxDetectionLimit/FluxDetectionLimit.ipynb))
- **Self-heating correction for open-path IRGA NEE fluxes**:
    - create scaling factors table and apply to correct open-path NEE fluxes during a time period of parallel
      measurements ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/flux/self-heating_correction/SelfHeatingCorrectionNEE_1_CreateScalingFactorsTable.ipynb))
    - apply previously created scaling factors table to long-term open-path NEE flux data, outside the time period of
      parallel
      measurements ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/flux/self-heating_correction/SelfHeatingCorrectionNEE_2_ApplyScalingFactorsTable.ipynb))
- **USTAR threshold scenarios**: display data availability under different USTAR threshold scenarios

### Formats

_Format data to specific formats._

- **Format**: convert EddyPro fluxnet output files for upload to FLUXNET
  database · class:
  `FormatEddyProFluxnetFileForUpload` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/io/FormatEddyProFluxnetFileForUpload.ipynb))
- **Parquet files**: load and save parquet
  files · funcs: `load_parquet()`,
  `save_parquet()` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/io/LoadSaveParquetFile.ipynb))

### Gap-filling

_Fill gaps in time series with various methods._

**Feature Engineering (v0.91.0)** · class: `FeatureEngineer`
- Standalone 8-stage feature engineering pipeline (composable, reusable across models)
  - Stage 1: Lagged features from past and future values
  - Stage 2: Rolling statistics (mean, std, median, min, max, quartiles)
  - Stage 3: Temporal differencing (1st and 2nd order momentum)
  - Stage 4: Exponential Moving Average (EMA) with recent-value emphasis
  - Stage 5: Polynomial expansion (squared, cubed terms)
  - Stage 6: STL decomposition (trend, seasonal, residual components)
  - Stage 7: Timestamp vectorization (season, month, hour, etc.)
  - Stage 8: Continuous record numbering for trend detection
- Pre-engineer features once, reuse across multiple models (RF + XGB simultaneously)
- Independent testing and debugging of feature engineering

- **XGBoostTS** · class:
  `XGBoostTS` ([notebook example (minimal)](https://github.com/holukas/diive/blob/main/notebooks/gapfilling/XGBoostGapFillingMinimal.ipynb), [notebook example (more extensive)](https://github.com/holukas/diive/blob/main/notebooks/gapfilling/XGBoostGapFillingExtensive.ipynb))
  - Use `FeatureEngineer` to create features, pass pre-engineered data to XGBoostTS
- **RandomForestTS** · class:
  `RandomForestTS` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/gapfilling/RandomForestGapFilling.ipynb))
  - Use `FeatureEngineer` to create features, pass pre-engineered data to RandomForestTS
- **Long-term gap-filling using RandomForestTS** · class:
  `LongTermGapFillingRandomForestTS` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/gapfilling/LongTermRandomForestGapFilling.ipynb))
- **Long-term gap-filling using XGBoostTS** · class:
  `LongTermGapFillingXGBoostTS` (for multi-year data with USTAR scenario support)
- **Linear interpolation** · func:
  `linear_interpolation()` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/gapfilling/LinearInterpolation.ipynb))
- **Quick random forest gap-filling** · class:
  `QuickFillRFTS` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/gapfilling/QuickRandomForestGapFilling.ipynb))
- **MDS gap-filling of ecosystem fluxes** · class:
  `FluxMDS` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/gapfilling/FluxMDSGapFilling.ipynb)),
  approach by [Reichstein et al., 2005](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2486.2005.001002.x)

#### Comprehensive Examples for CO2 Flux Data

- **FluxProcessingChain examples** for CO2 half-hourly flux (NEE) gap-filling:
  - Both **Random Forest** and **XGBoost** examples are fully activated and comprehensively documented
  - Optimized feature engineering for diurnal photosynthetic patterns (lag, rolling, EMA, STL decomposition)
  - Feature reduction enabled by default (SHAP-based selection reduces ~45-50 features to ~10-20)
  - Hyperparameters tuned for ecosystem flux data with detailed tuning guidance
  - Model comparison code to select best algorithm for your site
  - See `diive/pkgs/fluxprocessingchain/fluxprocessingchain.py` for detailed examples (~100 lines each)

### Outlier Detection

#### Multiple tests combined

- **Step-wise outlier detection**: combine multiple outlier flags to one single overall flag

#### Single tests

_Create single outlier flags where `0=OK` and `2=outlier`._

- **Absolute limits**: define absolute
  limits · class:
  `AbsoluteLimits` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/AbsoluteLimits.ipynb))
- **Absolute limits daytime/nighttime**: define absolute limits separately for daytime and nighttime
  data · class:
  `AbsoluteLimitsDaytimeNighttime` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/AbsoluteLimitsDaytimeNighttime.ipynb))
- **Hampel filter daytime/nighttime**, separately for daytime and nighttime
  data · class:
  `HampelDaytimeNighttime` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/HampelDaytimeNighttime.ipynb))
- **Local standard deviation**: Identify outliers based on the local standard deviation from a running
  median ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/LocalSD.ipynb))
- **Local outlier factor**: Identify outliers based on local outlier factor, across all
  data · class:
  `LocalOutlierFactorAllData` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/LocalOutlierFactorAllData.ipynb))
- **Local outlier factor daytime/nighttime**: Identify outliers based on local outlier factor, daytime nighttime
  separately ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/LocalOutlierFactorDaytimeNighttime.ipynb))
- **Manual removal**: Remove time periods (from-to) or single records from time
  series ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/ManualRemoval.ipynb))
- **Missing values**: Simply creates a flag that indicated available and missing data in a time
  series · class:
  `MissingValues` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/MissingValues.ipynb))
- **Trimming**: Remove values below threshold and remove an equal amount of records from high end of
  data ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/TrimLow.ipynb))
- **z-score**: Identify outliers based on the z-score across all time series
  data · class:
  `zScore` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/zScore.ipynb))
- **z-score increments daytime/nighttime**: Identify outliers based on the z-score of double
  increments ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/zScoreIncremental.ipynb))
- **z-score daytime/nighttime**: Identify outliers based on the z-score, separately for daytime and
  nighttime · class:
  `zScoreDaytimeNighttime` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/zScoreDaytimeNighttime.ipynb))
- **z-score rolling**: Identify outliers based on the rolling
  z-score ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/zScoreRolling.ipynb))

### Plotting

- **Cumulatives across all years for multiple variables** · class:
  `Cumulative` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/Cumulative.ipynb))
- **Cumulatives per year** · class:
  `CumulativeYear` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/CumulativesPerYear.ipynb))
- **Diel cycle per month** · class:
  `DielCycle` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/DielCycle.ipynb))
- **Heatmap date/time**: showing values (z) of time series as date (y) vs time (
  x) · class:
  `HeatmapDateTime` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/HeatmapDateTime.ipynb))
- **Heatmap year/month**: plot monthly ranks across
  years · class:
  `HeatmapYearMonth` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/HeatmapYearMonth.ipynb))
- **Heatmap XYZ**: show z-values in bins of x and y — pairs naturally with `GridAggregator` · class:
  `HeatmapXYZ` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/HeatmapXYZ.ipynb))
- **Hexbin plot**: aggregate flux values into 2D hexagonal bins of driver variables; supports percentile normalization and configurable aggregation functions · class:
  `HexbinPlot` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/HexbinPlot.ipynb))
- **Histogram**: includes options to show z-score limits and to highlight the peak distribution
  bin · class:
  `HistogramPlot` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/Histogram.ipynb))
- **Long-term anomalies**: calculate and plot long-term anomaly for a variable, per year, compared to a reference
  period · class:
  `LongtermAnomaliesYear` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/LongTermAnomalies.ipynb))
- **Ridgeline plot**: looks a bit like a
  landscape · class:
  `RidgeLinePlot` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/RidgeLine.ipynb))
- **Time series plot**: Simple (interactive) time series
  plot · class:
  `TimeSeries` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/TimeSeries.ipynb))
- **ScatterXY plot** · class:
  `ScatterXY` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/plotting/ScatterXY.ipynb))
- Various classes to generate heatmaps, bar plots, time series plots and scatter plots, among others

### Quality control

- **Stepwise MeteoScreening from database** · class:
  `StepwiseMeteoScreeningDb` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/qc/StepwiseMeteoScreeningFromDatabase.ipynb))

### Resampling

- **Diel cycle**: calculate diel cycle per
  month · func:
  `diel_cycle()` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/timeseries/ResamplingDielCycle.ipynb))

### Stats

- **Time series stats** · func:
  `sstats()` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/timeseries/TimeSeriesStats.ipynb))

### Timestamps

- **Continuous timestamp**: create continuous timestamp based on number of records in the file and the file duration ·
  func: `continuous_timestamp_freq()`
- **Time resolution**: detect time resolution from
  data · class:
  `DetectFrequency` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/timeseries/Detect_time_resolution.ipynb))
- **Timestamps**: create and insert additional timestamps in various formats · class: `TimestampSanitizer`
- **Vectorize timestamps**: add date attributes as columns to dataframe, including sine/cosine variants fpr cyclical
  variables (e.g., day of
  year) · func:
  `vectorize_timestamps()` ([notebook example](https://github.com/holukas/diive/blob/main/notebooks/timeseries/VectorizeTimestamps.ipynb))

## Installation

`diive` is currently under active developement using **Python v3.11**.

### Using pip

`pip install diive`

### Using poetry

`poetry add diive`

### From source

Directly use .tar.gz file of the desired version.

`pip install https://github.com/holukas/diive/archive/refs/tags/v0.76.2.tar.gz`

### Create and use a conda environment for diive

One way to install and use `diive` with a specific Python version on a local machine:

- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Start `miniconda` prompt
- Create a environment named `diive-env` that contains Python 3.11: `conda create --name diive-env python=3.11`
- Activate the new environment: `conda activate diive-env`
- Install `diive` using pip: `pip install diive`
- To start JupyterLab type `jupyter lab` in the prompt
