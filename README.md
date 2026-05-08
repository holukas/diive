![](images/logo_diive1_256px.png)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/diive?style=for-the-badge&color=%23EF6C00&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdiive%2F)](https://pypi.org/project/diive/)
[![GitHub License](https://img.shields.io/github/license/holukas/diive?style=for-the-badge&color=%237CB342)](https://github.com/holukas/diive/blob/indev/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/diive)](https://pepy.tech/projects/diive)
[![DOI](https://zenodo.org/badge/708559210.svg)](https://zenodo.org/doi/10.5281/zenodo.10884017)

*`diive` is currently under active developement with frequent updates.*

**Update 8 May 2026**  
*`diive` is currently prepared for the first main release. There is still
quite a bit missing, e.g., code examples for all public classes and functions.
After that, the documentation will be created and hosted on ReadTheDocs.
Most of the notebooks will be removed once the documentation is ready.
New features are still added regularly, and existing functionality is
continuously improved.*

# Time series data processing

`diive` is a Python library for time series processing, in particular ecosystem data. Originally developed
by the [ETH Grassland Sciences group](https://gl.ethz.ch/) for [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/).

Recent updates: [CHANGELOG](https://github.com/holukas/diive/blob/main/CHANGELOG.md)
Recent releases: [Releases](https://github.com/holukas/diive/releases)

---

## Quick Access to Common Classes

Classes are available directly from the `diive` namespace with both PascalCase and snake_case names:

```python
# PascalCase (class name)
from diive.core.plotting import TimeSeries

plot = TimeSeries(series=data)

# snake_case (alias)
import diive as dv

plot = dv.plot_time_series(series=data)
```

### Available exports

**Plotting:** `time_series`, `TimeSeries`, `cumulative`, `Cumulative`, `diel_cycle`, `DielCycle`, `heatmap_datetime`,
`HeatmapDateTime`, and more

**Gap-filling:** `randomforest_ts`, `RandomForestTS`, `xgboost_ts`, `XGBoostTS`, `quick_fill_rfts`, `QuickFillRFTS`,
`flux_mds`, `FluxMDS`, `optimize_params_ts`, `OptimizeParamsTS`, `optimize_params_rfts`, `OptimizeParamsRFTS`

**Analysis:** `gridaggregator`, `GridAggregator`, `seasonaltrend`, `SeasonalTrendDecomposition`

**Eddy Covariance:** `FluxDetectionLimit`, `fdl`, `MaxCovariance`, `max_covariance`, `WindRotation2D`,
`wind_rotation_2d`

**I/O:** `load_parquet`, `save_parquet`, `load_exampledata_parquet`, `search_files`

For the complete list of available aliases, see `diive.__all__`.

---

## Examples

**58 executable examples** organized to mirror the codebase structure in the `examples/` folder.

**Quick links:**
- 📖 **[CATALOG.md](examples/CATALOG.md)** — Find examples by use case (workflows, methods, difficulty levels)
- 📊 **[EXAMPLE_DATASET.md](examples/EXAMPLE_DATASET.md)** — Dataset documentation (37 variables, gaps, quality)
- 🚀 **[examples/README.md](examples/README.md)** — Quick start and structure overview

**Run all examples (parallelized, 8 workers):**

```bash
uv run python examples/run_all_examples.py
```

**Run individual examples:**

```bash
# Quick visualization
uv run python examples/core/visualization/heatmap_datetime.py

# Time series analysis
uv run python examples/pkgs/analysis/seasonaltrend.py

# Gap-filling
uv run python examples/pkgs/gapfilling/randomforest_ts.py
uv run python examples/pkgs/gapfilling/comparison.py      # Compare methods

# Quality control
uv run python examples/pkgs/preprocessing/outlierdetection/hampel.py
uv run python examples/pkgs/preprocessing/qaqc/qcf.py

# Flux processing
uv run python examples/pkgs/flux/hires/lag.py             # Time lag detection
uv run python examples/pkgs/flux/ustarthreshold/ustarthreshold.py
```

**Example categories:**
- **Visualization** (22): heatmap_datetime, hexbin, timeseries, cumulative, dielcycle, histogram, ridgeline, scatter
- **Analyses** (8): correlation, decoupling, gapfinder, gridaggregator, histogram, optimumrange, quantiles,
  seasonaltrend
- **Data Processing** (50): binary extraction, corrections (setto, offsetcorrection), outlierdetection (absolutelimits,
  hampel, incremental, localsd, lof, manualremoval, stepwise, trim), qaqc (FlagQCF, 6 EddyProFlags examples),
  createvar (air, conversions, daynightflag, laggedvariants, noise, potentialradiation, timesince, vpd)
- **Gap-Filling** (10): linear_interpolation, mds, mds_comparison, randomforest_ts (3 examples: full, quick, optimize),
  xgboost_ts (2 examples: full, optimize), comparison (MDS vs RF vs XGB)
- **Eddy Covariance & Flux** (9): fluxdetectionlimit, lag, windrotation, hqflux, selfheating, uncertainty,
  ustarthreshold (3 examples)
- **Fits** (1): fitter

See [examples/README.md](examples/README.md) for a complete index of all examples with descriptions and quick start
guides.

Additional examples available in **Jupyter notebooks** at [notebooks/](notebooks/) with comprehensive workflows and
tutorials.

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
- **Seasonal-Trend Decomposition**: Separate time series into trend, seasonal, and residual components using STL (
  Seasonal-Trend Loess), classical, or harmonic methods · class:
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
    - See `examples/gap_filling/` folder for standalone runnable examples (Phase 2, coming soon)
    - Or see `diive/pkgs/fluxprocessingchain/fluxprocessingchain.py` for detailed inline examples

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
- **Hexbin plot**: aggregate flux values into 2D hexagonal bins of driver variables; supports percentile normalization
  and configurable aggregation functions · class:
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

- **Overall Quality Control Flag (QCF)** · class:
  `FlagQCF` ([example](examples/qaqc/qcf.py))
    - Combines multiple individual test flags into a single overall quality indicator
    - Supports daytime/nighttime separation and USTAR filtering scenarios
    - Generates comprehensive reports: QCF distribution, test statistics, sequential impact analysis
    - Visualizations: 4-panel heatmap (original, QC, flag sums, QCF flag)
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

`diive` requires **Python 3.12-3.13**.

### Using pip (Recommended)

```bash
pip install diive
```

### Using uv (Modern, Fast)

[uv](https://docs.astral.sh/uv/) is a modern Python package installer and resolver (5-10x faster than pip):

```bash
uv pip install diive
```

Or create a project with uv:

```bash
uv venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
uv pip install diive
```

### Using poetry

```bash
poetry add diive
```

### From source (Development)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/holukas/diive.git
cd diive
uv sync                       # Install dependencies with uv
uv run pytest tests/          # Run tests
```

### Legacy: Using conda

If you prefer conda, create a new environment:

```bash
conda create -n diive python=3.12
conda activate diive
pip install diive
```

For development with conda:

```bash
conda env create -f environment.yml
conda activate diive
pip install -e .
```
