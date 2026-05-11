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
| [**heatmap_datetime_basic.py**](core/visualization/heatmap_datetime_basic.py) | Datetime heatmaps with vertical/horizontal orientations | HeatmapDateTime, diurnal patterns |
| [**heatmap_advanced.py**](core/visualization/heatmap_advanced.py) | Year-month aggregation and multi-variable comparison | HeatmapYearMonth, seasonal analysis |
| [**scatter_xy_basic.py**](core/visualization/scatter_xy_basic.py) | Basic 2D scatter plots for exploring variable relationships | ScatterXY, correlation analysis |
| [**scatter_xy_colored.py**](core/visualization/scatter_xy_colored.py) | 3D scatter with color coding and bin aggregation | ScatterXY, trend visualization |
| [**timeseries.py**](core/visualization/timeseries.py) | Time series line plots with matplotlib | TimeSeries, trend visualization |
| [**timeseries_interactive.py**](core/visualization/timeseries_interactive.py) | Interactive Bokeh plots with zoom, pan, export | TimeSeries, data exploration |
| [**cumulative_basic.py**](core/visualization/cumulative_basic.py) | Cumulative flux across all time with scenario comparison | Cumulative, running totals |
| [**cumulative_year.py**](core/visualization/cumulative_year.py) | Yearly cumulative sums with reference band and highlighting | CumulativeYear, annual budgets |
| [**dielcycle.py**](core/visualization/dielcycle.py) | Diurnal (diel) cycle analysis by month/season | DielCycle, daytime/nighttime patterns |
| [**hexbin_basic.py**](core/visualization/hexbin_basic.py) | 2D hexagonal binning with percentile normalization | HexbinPlot, standardized comparison |
| [**hexbin_advanced.py**](core/visualization/hexbin_advanced.py) | Advanced hexbin with absolute values and overlays | HexbinPlot, detailed visualization |
| [**histogram_basic.py**](core/visualization/histogram_basic.py) | Distribution histograms with z-score overlay | HistogramPlot, outlier detection |
| [**histogram_yearly.py**](core/visualization/histogram_yearly.py) | Yearly comparison histograms for temporal patterns | HistogramPlot, seasonal analysis |
| [**ridgeline_basic.py**](core/visualization/ridgeline_basic.py) | Ridge line plots with weekly grouping | RidgeLinePlot, kernel density |
| [**ridgeline_advanced.py**](core/visualization/ridgeline_advanced.py) | Ridge line plots with monthly grouping and styling options | RidgeLinePlot, temporal patterns |
| [**other_plots.py**](core/visualization/other_plots.py) | Specialized plot types (anomalies, trends) | LongtermAnomaliesYear |

**See also:** [core/visualization/README.md](core/visualization/README.md)

---

## Data Analysis

**Analyze time series patterns, trends, correlations, and gaps.**

| Example | What You'll Learn | Use Case |
|---------|------------------|----------|
| [**analysis_correlation.py**](pkgs/analysis/analysis_correlation.py) | Cross-correlation, autocorrelation, anomaly detection | Identify lagged relationships |
| [**analysis_seasonaltrend.py**](pkgs/analysis/analysis_seasonaltrend.py) | Seasonal decomposition (STL method) | Isolate trend from seasonality |
| [**analysis_decoupling.py**](pkgs/analysis/analysis_decoupling.py) | Stratified analysis, canopy-scale processes | Analyze by vegetation layers |
| [**analysis_gapfinder.py**](pkgs/analysis/analysis_gapfinder.py) | Detect and visualize data gaps | Assess data completeness |
| [**analysis_gridaggregator.py**](pkgs/analysis/analysis_gridaggregator.py) | 2D spatial gridding and aggregation | Aggregate point measurements to grids |
| [**analysis_histogram_distribution.py**](pkgs/analysis/analysis_histogram_distribution.py) | Distribution analysis, percentiles | Quantile-based analysis |
| [**analysis_optimumrange.py**](pkgs/analysis/analysis_optimumrange.py) | Find optimal value ranges in data | Condition-based filtering |
| [**analysis_quantiles.py**](pkgs/analysis/analysis_quantiles.py) | Percentile and quantile calculations | Non-parametric statistics |
| [**analysis_harmonic.py**](pkgs/analysis/analysis_harmonic.py) | Spectral analysis and Fourier decomposition | Frequency content analysis |

**See also:** [pkgs/analysis/README.md](pkgs/analysis/README.md)

---

## Data Processing & Corrections

**Clean, correct, and transform data for quality and usability.**

### **Offset Corrections**

| Example | What You'll Learn | When to Use |
|---------|------------------|------------|
| [**correction_relativehumidity_offset.py**](pkgs/preprocessing/corrections/correction_relativehumidity_offset.py) | Fix RH measurements exceeding 100% | RH saturation issues from sensor drift |
| [**correction_radiation_offset.py**](pkgs/preprocessing/corrections/correction_radiation_offset.py) | Correct radiation nighttime offset | Non-zero radiation readings at night |
| [**correction_measurement_offset_replicate.py**](pkgs/preprocessing/corrections/correction_measurement_offset_replicate.py) | Detect offset vs. reference replicate | Compare two instruments, constant bias |
| [**correction_winddir_offset.py**](pkgs/preprocessing/corrections/correction_winddir_offset.py) | Correct wind direction drift | Anemometer misalignment over time |

### **Value Replacement & Validation**

| Example | What You'll Learn | When to Use |
|---------|------------------|------------|
| [**correction_set_exact_values_to_missing.py**](pkgs/preprocessing/corrections/correction_set_exact_values_to_missing.py) | Set exact values to NaN | Remove error codes or sentinel values |
| [**correction_setto_value.py**](pkgs/preprocessing/corrections/correction_setto_value.py) | Replace values in time periods with constant | Known malfunction periods or instrument errors |
| [**correction_setto_threshold.py**](pkgs/preprocessing/corrections/correction_setto_threshold.py) | Clip values to min/max thresholds | Enforce physically realistic limits |
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
| [**outlier_hampel.py**](pkgs/preprocessing/outlierdetection/outlier_hampel.py) | Hampel filter (MAD-based) | Robust spike detection | Beginner ⭐ |
| [**outlier_zscore.py**](pkgs/preprocessing/outlierdetection/outlier_zscore.py) | Z-score (global, day/night, rolling) | Statistical thresholding | Beginner ⭐ |
| [**outlier_localsd.py**](pkgs/preprocessing/outlierdetection/outlier_localsd.py) | Local standard deviation | Adaptive thresholds | Intermediate ⭐⭐ |
| [**outlier_absolutelimits.py**](pkgs/preprocessing/outlierdetection/outlier_absolutelimits.py) | Min/max thresholds | Known physical limits | Beginner ⭐ |
| [**outlier_incremental.py**](pkgs/preprocessing/outlierdetection/outlier_incremental.py) | Increment-based detection | Abrupt changes | Intermediate ⭐⭐ |
| [**outlier_lof.py**](pkgs/preprocessing/outlierdetection/outlier_lof.py) | Local Outlier Factor | Density-based anomalies | Advanced ⭐⭐⭐ |
| [**outlier_manualremoval.py**](pkgs/preprocessing/outlierdetection/outlier_manualremoval.py) | Manual point/range removal | Known issues | Simple ⭐ |
| [**outlier_trim.py**](pkgs/preprocessing/outlierdetection/outlier_trim.py) | Trimmed mean approach | Symmetric removal | Beginner ⭐ |
| [**outlier_stepwise.py**](pkgs/preprocessing/outlierdetection/outlier_stepwise.py) | Chain multiple methods | Multi-stage QA/QC | Advanced ⭐⭐⭐ |

### **Overall Quality Flags**

| Example | What You'll Learn |
|---------|------------------|
| [**qc_overall_flag.py**](pkgs/preprocessing/qaqc/qc_overall_flag.py) | Combine multiple test flags into overall QCF |
| [**qc_eddypro_flags.py**](pkgs/preprocessing/qaqc/qc_eddypro_flags.py) | EddyPro quality flags (VM97, signal strength, completeness) |

**See also:** [pkgs/preprocessing/outlierdetection/README.md](pkgs/preprocessing/outlierdetection/README.md)

---

## Feature Engineering

**Create and engineer variables for modeling and analysis.**

| Example | What You'll Learn | Output Variables |
|---------|------------------|------------------|
| [**feature_daynightflag.py**](pkgs/features/feature_daynightflag.py) | Daytime/nighttime classification from solar geometry | Day/night boolean flag |
| [**feature_vpd.py**](pkgs/features/feature_vpd.py) | Vapor Pressure Deficit calculation | VPD (hPa) |
| [**feature_air.py**](pkgs/features/feature_air.py) | Air properties (density, resistance, heat capacity) | Aerodynamic properties |
| [**feature_conversions.py**](pkgs/features/feature_conversions.py) | Unit conversions (temperature, energy, water) | Derived units |
| [**feature_potentialradiation.py**](pkgs/features/feature_potentialradiation.py) | Clear-sky radiation calculation | Potential radiation (W/m²) |
| [**feature_laggedvariants.py**](pkgs/features/feature_laggedvariants.py) | Create lagged and shifted variables | Lag features for models |
| [**feature_timesince.py**](pkgs/features/feature_timesince.py) | Time-since-event features | Relative time metrics |
| [**feature_noise.py**](pkgs/features/feature_noise.py) | Synthetic noise generation | Test noise robustness |

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
| [**hqflux.py**](pkgs/flux/lowres/hqflux/hqflux.py) | Extract highest-quality flux using Hampel filter |
| [**uncertainty.py**](pkgs/flux/lowres/uncertainty/uncertainty.py) | Random uncertainty estimation (PAS20 method) |

### **USTAR Filtering (Low-Turbulence)**

| Example | What You'll Learn |
|---------|------------------|
| [**ustarthreshold.py**](pkgs/flux/lowres/ustarthreshold/ustarthreshold.py) | USTAR threshold detection and filtering |
| [**ustar_mp_detection.py**](pkgs/flux/lowres/ustar_mp_detection/ustar_mp_detection.py) | Moving Point (MP) USTAR method |

### **Corrections**

| Example | What You'll Learn |
|---------|------------------|
| [**selfheating.py**](pkgs/flux/lowres/selfheating/selfheating.py) | Oxygen sensor self-heating correction |

### **Common Utilities**

| Example | What You'll Learn |
|---------|------------------|
| [**common.py**](pkgs/flux/lowres/common.py) | Helper functions and utilities for flux processing |

**See also:** [pkgs/flux/README.md](pkgs/flux/README.md)

---

## Gap-Filling

**Fill missing data using various methods, from simple to advanced ML.**

| Example | Method | Training | Speed | Accuracy | Best For |
|---------|--------|----------|-------|----------|----------|
| [**gapfill_interpolate.py**](pkgs/gapfilling/gapfill_interpolate.py) | Linear interpolation | No | ⚡ Very fast | ⭐ Simple | Small gaps (<1 day) |
| [**gapfill_randomforest_ts.py**](pkgs/gapfilling/gapfill_randomforest_ts.py) | Random Forest | Yes | 🔄 Medium | ⭐⭐⭐⭐ Good | General purpose |
| [**gapfill_xgboost_ts.py**](pkgs/gapfilling/gapfill_xgboost_ts.py) | XGBoost | Yes | 🔄 Medium | ⭐⭐⭐⭐⭐ Excellent | High accuracy |
| [**gapfill_mds.py**](pkgs/gapfilling/gapfill_mds.py) | Meteorological Data Similarity | No | ⚡ Fast | ⭐⭐ Moderate | No training data |
| [**gapfill_mds_comparison.py**](pkgs/gapfilling/gapfill_mds_comparison.py) | MDS variants | No | ⚡ Fast | Variable | Method comparison |
| [**gapfill_comparison.py**](pkgs/gapfilling/gapfill_comparison.py) | Benchmark multiple methods | Mixed | - | - | Performance evaluation |

**See also:** [pkgs/gapfilling/README.md](pkgs/gapfilling/README.md)

---

## Curve Fitting

**Fit curves and models to data.**

| Example | What You'll Learn |
|---------|------------------|
| [**fit_fitter.py**](pkgs/fits/fit_fitter.py) | Polynomial and custom function fitting |

**See also:** [pkgs/fits/README.md](pkgs/fits/README.md)

---

## Workflows by Use Case

### **"I need to process raw flux data"**
1. Load data: `dv.load_exampledata_parquet()`
2. Clean timestamps: [timestamp_sanitizer.py](core/times/timestamp_sanitizer.py)
3. Correct offsets: [correction_*.py](pkgs/preprocessing/corrections/) — RH, radiation, wind direction, or measurement bias
4. Detect outliers: [hampel.py](pkgs/preprocessing/outlierdetection/hampel.py) or [stepwise.py](pkgs/preprocessing/outlierdetection/stepwise.py)
5. Flag quality: [qcf.py](pkgs/preprocessing/qaqc/qcf.py)
6. Fill gaps: [randomforest_ts.py](pkgs/gapfilling/randomforest_ts.py) or [xgboost_ts.py](pkgs/gapfilling/xgboost_ts.py)
7. Visualize: [heatmap_datetime.py](core/visualization/heatmap_datetime.py) or [timeseries.py](core/visualization/timeseries.py)

### **"I want to analyze seasonal patterns"**
1. [analysis_seasonaltrend.py](pkgs/analysis/analysis_seasonaltrend.py) — Decompose trend + seasonal
2. [dielcycle.py](core/visualization/dielcycle.py) — Diurnal cycles
3. [heatmap_datetime.py](core/visualization/heatmap_datetime.py) — Monthly/seasonal heatmaps
4. [analysis_correlation.py](pkgs/analysis/analysis_correlation.py) — Cross-seasonal correlations

### **"I need to understand measurement quality"**
1. [analysis_gapfinder.py](pkgs/analysis/analysis_gapfinder.py) — Where are the gaps?
2. [qc_eddypro_flags.py](pkgs/preprocessing/qaqc/qc_eddypro_flags.py) — What does EddyPro say?
3. [qc_overall_flag.py](pkgs/preprocessing/qaqc/qc_overall_flag.py) — Overall quality assessment
4. [flux_hqflux.py](pkgs/flux/lowres/flux_hqflux.py) — Extract only the best data

### **"I want to compare gap-filling methods"**
1. [gapfill_comparison.py](pkgs/gapfilling/gapfill_comparison.py) — Benchmark all methods
2. [gapfill_randomforest.py](pkgs/gapfilling/gapfill_randomforest.py) — Random Forest details
3. [gapfill_xgboost.py](pkgs/gapfilling/gapfill_xgboost.py) — XGBoost details
4. [gapfill_mds_comparison.py](pkgs/gapfilling/gapfill_mds_comparison.py) — MDS variants

### **"I'm new to DIIVE"**
**Start with these beginner-friendly examples:**
1. [timestamp_sanitizer.py](core/times/timestamp_sanitizer.py) — Data cleaning basics
2. [heatmap_datetime.py](core/visualization/heatmap_datetime.py) — First visualization
3. [feature_vpd.py](pkgs/features/feature_vpd.py) — Simple variable creation
4. [outlier_hampel.py](pkgs/preprocessing/outlierdetection/outlier_hampel.py) — Basic outlier detection
5. [gapfill_randomforest.py](pkgs/gapfilling/gapfill_randomforest.py) — ML gap-filling

---

## See Also

- [**EXAMPLE_DATASET.md**](EXAMPLE_DATASET.md) — Detailed documentation of the example data
- [**README.md**](README.md) — Examples overview and quick start
- [**run_all_examples.py**](run_all_examples.py) — Run all examples in parallel (user can execute for full validation)
