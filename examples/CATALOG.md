# Examples Catalog

**Find examples by what you want to do.** All 58 examples organized by use case and functionality.

## Quick Navigation

- [📊 Visualization & Plotting](#visualization--plotting)
- [🔬 Data Analysis](#data-analysis)
- [🛠️ Data Processing & Corrections](#data-processing--corrections)
- [🎯 Quality Control](#quality-control)
- [🌿 Feature Engineering](#feature-engineering)
- [💨 Flux Processing](#flux-processing)
- [🔌 Gap-Filling](#gap-filling)
- [📈 Curve Fitting](#curve-fitting)

---

## Visualization & Plotting

**Learn to create publication-quality plots and visualizations.**

| Example | What You'll Learn | Related |
|---------|------------------|---------|
| [**heatmap_datetime.py**](core/visualization/heatmap_datetime.py) | Heatmaps with datetime indices (daily/monthly aggregation) | HeatmapDateTime, HeatmapYearMonth |
| [**scatter_xy.py**](core/visualization/scatter_xy.py) | Customizable 2D scatter plots with annotations | ScatterXY, correlation plots |
| [**timeseries.py**](core/visualization/timeseries.py) | Interactive time series line plots | TimeSeries, trend visualization |
| [**timeseries_and_cumulative.py**](core/visualization/timeseries_and_cumulative.py) | Combine time series with cumulative sum plots | Cumulative, CumulativeYear |
| [**dielcycle.py**](core/visualization/dielcycle.py) | Diurnal (diel) cycle analysis by month/season | DielCycle, daytime/nighttime patterns |
| [**hexbin.py**](core/visualization/hexbin.py) | 2D hexagonal binning density plots | HexbinPlot, 2D distributions |
| [**histogram.py**](core/visualization/histogram.py) | Distribution histograms and statistics | HistogramPlot, quantile analysis |
| [**ridgeline.py**](core/visualization/ridgeline.py) | Ridge line plots comparing distributions | RidgeLinePlot, kernel density |
| [**other_plots.py**](core/visualization/other_plots.py) | Specialized plot types (anomalies, trends) | LongtermAnomaliesYear |

**See also:** [core/visualization/README.md](core/visualization/README.md)

---

## Data Analysis

**Analyze time series patterns, trends, correlations, and gaps.**

| Example | What You'll Learn | Use Case |
|---------|------------------|----------|
| [**correlation.py**](pkgs/analysis/correlation.py) | Cross-correlation, autocorrelation, anomaly detection | Identify lagged relationships |
| [**seasonaltrend.py**](pkgs/analysis/seasonaltrend.py) | Seasonal decomposition (STL method) | Isolate trend from seasonality |
| [**decoupling.py**](pkgs/analysis/decoupling.py) | Stratified analysis, canopy-scale processes | Analyze by vegetation layers |
| [**gapfinder.py**](pkgs/analysis/gapfinder.py) | Detect and visualize data gaps | Assess data completeness |
| [**gridaggregator.py**](pkgs/analysis/gridaggregator.py) | 2D spatial gridding and aggregation | Aggregate point measurements to grids |
| [**histogram_distribution.py**](pkgs/analysis/histogram_distribution.py) | Distribution analysis, percentiles | Quantile-based analysis |
| [**optimumrange.py**](pkgs/analysis/optimumrange.py) | Find optimal value ranges in data | Condition-based filtering |
| [**quantiles.py**](pkgs/analysis/quantiles.py) | Percentile and quantile calculations | Non-parametric statistics |

**See also:** [pkgs/analysis/README.md](pkgs/analysis/README.md)

---

## Data Processing & Corrections

**Clean, correct, and transform data for quality and usability.**

| Example | What You'll Learn | When to Use |
|---------|------------------|------------|
| [**offsetcorrection.py**](pkgs/preprocessing/corrections/offsetcorrection.py) | Correct systematic offsets in sensors | Temperature, humidity, radiation drift |
| [**setto.py**](pkgs/preprocessing/corrections/setto.py) | Replace values, set to missing, apply thresholds | Known measurement errors, maintenance periods |
| [**timestamp_sanitizer.py**](core/times/timestamp_sanitizer.py) | Clean, validate, regularize datetime indices | Ensure monotonic, regular time series |

**See also:** 
- [pkgs/preprocessing/corrections/README.md](pkgs/preprocessing/corrections/README.md)
- [core/times/](core/times/)

---

## Quality Control

**Detect outliers, flag bad data, and assess data quality.**

### **Outlier Detection Methods**

| Example | Method | Best For | Difficulty |
|---------|--------|----------|------------|
| [**hampel.py**](pkgs/preprocessing/outlierdetection/hampel.py) | Hampel filter (MAD-based) | Robust spike detection | Beginner ⭐ |
| [**zscore.py**](pkgs/preprocessing/outlierdetection/zscore.py) | Z-score (global, day/night, rolling) | Statistical thresholding | Beginner ⭐ |
| [**localsd.py**](pkgs/preprocessing/outlierdetection/localsd.py) | Local standard deviation | Adaptive thresholds | Intermediate ⭐⭐ |
| [**absolutelimits.py**](pkgs/preprocessing/outlierdetection/absolutelimits.py) | Min/max thresholds | Known physical limits | Beginner ⭐ |
| [**incremental.py**](pkgs/preprocessing/outlierdetection/incremental.py) | Increment-based detection | Abrupt changes | Intermediate ⭐⭐ |
| [**lof.py**](pkgs/preprocessing/outlierdetection/lof.py) | Local Outlier Factor | Density-based anomalies | Advanced ⭐⭐⭐ |
| [**manualremoval.py**](pkgs/preprocessing/outlierdetection/manualremoval.py) | Manual point/range removal | Known issues | Simple ⭐ |
| [**trim.py**](pkgs/preprocessing/outlierdetection/trim.py) | Trimmed mean approach | Symmetric removal | Beginner ⭐ |
| [**stepwise.py**](pkgs/preprocessing/outlierdetection/stepwise.py) | Chain multiple methods | Multi-stage QA/QC | Advanced ⭐⭐⭐ |

### **Overall Quality Flags**

| Example | What You'll Learn |
|---------|------------------|
| [**qcf.py**](pkgs/preprocessing/qaqc/qcf.py) | Combine multiple test flags into overall QCF |
| [**eddyproflags.py**](pkgs/preprocessing/qaqc/eddyproflags.py) | EddyPro quality flags (VM97, signal strength, SSITC) |

**See also:** [pkgs/preprocessing/outlierdetection/README.md](pkgs/preprocessing/outlierdetection/README.md)

---

## Feature Engineering

**Create and engineer variables for modeling and analysis.**

| Example | What You'll Learn | Output Variables |
|---------|------------------|------------------|
| [**daynightflag.py**](pkgs/features/daynightflag.py) | Daytime/nighttime classification from solar geometry | Day/night boolean flag |
| [**vpd.py**](pkgs/features/vpd.py) | Vapor Pressure Deficit calculation | VPD (hPa) |
| [**air.py**](pkgs/features/air.py) | Air properties (density, resistance, heat capacity) | Aerodynamic properties |
| [**conversions.py**](pkgs/features/conversions.py) | Unit conversions (temperature, energy, water) | Derived units |
| [**potentialradiation.py**](pkgs/features/potentialradiation.py) | Clear-sky radiation calculation | Potential radiation (W/m²) |
| [**laggedvariants.py**](pkgs/features/laggedvariants.py) | Create lagged and shifted variables | Lag features for models |
| [**timesince.py**](pkgs/features/timesince.py) | Time-since-event features | Relative time metrics |
| [**noise.py**](pkgs/features/noise.py) | Synthetic noise generation | Test noise robustness |

**See also:** [pkgs/features/README.md](pkgs/features/README.md)

---

## Flux Processing

**Process eddy covariance flux data through the complete pipeline.**

### **High-Resolution Analysis**

| Example | What You'll Learn |
|---------|------------------|
| [**lag.py**](pkgs/flux/hires/lag.py) | Time lag detection via maximum covariance |
| [**windrotation.py**](pkgs/flux/hires/windrotation.py) | Coordinate rotation for wind measurements |
| [**fluxdetectionlimit.py**](pkgs/flux/hires/fluxdetectionlimit.py) | Detection limit and signal-to-noise analysis |

### **Quality & Uncertainty**

| Example | What You'll Learn |
|---------|------------------|
| [**hqflux.py**](pkgs/flux/hqflux/hqflux.py) | Extract highest-quality flux using Hampel filter |
| [**uncertainty.py**](pkgs/flux/uncertainty/uncertainty.py) | Random uncertainty estimation (PAS20 method) |

### **USTAR Filtering (Low-Turbulence)**

| Example | What You'll Learn |
|---------|------------------|
| [**ustarthreshold.py**](pkgs/flux/ustarthreshold/ustarthreshold.py) | USTAR threshold detection and filtering |
| [**ustar_mp_detection.py**](pkgs/flux/ustar_mp_detection/ustar_mp_detection.py) | Moving Point (MP) USTAR method |

### **Corrections**

| Example | What You'll Learn |
|---------|------------------|
| [**selfheating.py**](pkgs/flux/selfheating/selfheating.py) | Oxygen sensor self-heating correction |

### **Common Utilities**

| Example | What You'll Learn |
|---------|------------------|
| [**common.py**](pkgs/flux/common.py) | Helper functions and utilities for flux processing |

**See also:** [pkgs/flux/README.md](pkgs/flux/README.md)

---

## Gap-Filling

**Fill missing data using various methods, from simple to advanced ML.**

| Example | Method | Training | Speed | Accuracy | Best For |
|---------|--------|----------|-------|----------|----------|
| [**interpolate.py**](pkgs/gapfilling/interpolate.py) | Linear interpolation | No | ⚡ Very fast | ⭐ Simple | Small gaps (<1 day) |
| [**randomforest_ts.py**](pkgs/gapfilling/randomforest_ts.py) | Random Forest | Yes | 🔄 Medium | ⭐⭐⭐⭐ Good | General purpose |
| [**xgboost_ts.py**](pkgs/gapfilling/xgboost_ts.py) | XGBoost | Yes | 🔄 Medium | ⭐⭐⭐⭐⭐ Excellent | High accuracy |
| [**mds.py**](pkgs/gapfilling/mds.py) | Meteorological Data Similarity | No | ⚡ Fast | ⭐⭐ Moderate | No training data |
| [**mds_comparison.py**](pkgs/gapfilling/mds_comparison.py) | MDS variants | No | ⚡ Fast | Variable | Method comparison |
| [**comparison.py**](pkgs/gapfilling/comparison.py) | Benchmark multiple methods | Mixed | - | - | Performance evaluation |

**See also:** [pkgs/gapfilling/README.md](pkgs/gapfilling/README.md)

---

## Curve Fitting

**Fit curves and models to data.**

| Example | What You'll Learn |
|---------|------------------|
| [**fitter.py**](pkgs/fits/fitter.py) | Polynomial and custom function fitting |

**See also:** [pkgs/fits/README.md](pkgs/fits/README.md)

---

## Workflows by Use Case

### **"I need to process raw flux data"**
1. Load data: `dv.load_exampledata_parquet()`
2. Clean timestamps: [timestamp_sanitizer.py](core/times/timestamp_sanitizer.py)
3. Correct offsets: [offsetcorrection.py](pkgs/preprocessing/corrections/offsetcorrection.py)
4. Detect outliers: [hampel.py](pkgs/preprocessing/outlierdetection/hampel.py) or [stepwise.py](pkgs/preprocessing/outlierdetection/stepwise.py)
5. Flag quality: [qcf.py](pkgs/preprocessing/qaqc/qcf.py)
6. Fill gaps: [randomforest_ts.py](pkgs/gapfilling/randomforest_ts.py) or [xgboost_ts.py](pkgs/gapfilling/xgboost_ts.py)
7. Visualize: [heatmap_datetime.py](core/visualization/heatmap_datetime.py) or [timeseries.py](core/visualization/timeseries.py)

### **"I want to analyze seasonal patterns"**
1. [seasonaltrend.py](pkgs/analysis/seasonaltrend.py) — Decompose trend + seasonal
2. [dielcycle.py](core/visualization/dielcycle.py) — Diurnal cycles
3. [heatmap_datetime.py](core/visualization/heatmap_datetime.py) — Monthly/seasonal heatmaps
4. [correlation.py](pkgs/analysis/correlation.py) — Cross-seasonal correlations

### **"I need to understand measurement quality"**
1. [gapfinder.py](pkgs/analysis/gapfinder.py) — Where are the gaps?
2. [eddyproflags.py](pkgs/preprocessing/qaqc/eddyproflags.py) — What does EddyPro say?
3. [qcf.py](pkgs/preprocessing/qaqc/qcf.py) — Overall quality assessment
4. [hqflux.py](pkgs/flux/hqflux/hqflux.py) — Extract only the best data

### **"I want to compare gap-filling methods"**
1. [comparison.py](pkgs/gapfilling/comparison.py) — Benchmark all methods
2. [randomforest_ts.py](pkgs/gapfilling/randomforest_ts.py) — Random Forest details
3. [xgboost_ts.py](pkgs/gapfilling/xgboost_ts.py) — XGBoost details
4. [mds_comparison.py](pkgs/gapfilling/mds_comparison.py) — MDS variants

### **"I'm new to DIIVE"**
**Start with these beginner-friendly examples:**
1. [timestamp_sanitizer.py](core/times/timestamp_sanitizer.py) — Data cleaning basics
2. [heatmap_datetime.py](core/visualization/heatmap_datetime.py) — First visualization
3. [vpd.py](pkgs/features/vpd.py) — Simple variable creation
4. [hampel.py](pkgs/preprocessing/outlierdetection/hampel.py) — Basic outlier detection
5. [randomforest_ts.py](pkgs/gapfilling/randomforest_ts.py) — ML gap-filling

---

## See Also

- [**EXAMPLE_DATASET.md**](EXAMPLE_DATASET.md) — Detailed documentation of the example data
- [**README.md**](README.md) — Examples overview and quick start
- [**run_all_examples.py**](run_all_examples.py) — Run all 58 examples in parallel
