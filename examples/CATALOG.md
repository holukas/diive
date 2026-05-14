# Examples Catalog

Find examples by topic and use case.

## Quick Navigation

- [Visualization & Plotting](#visualization--plotting)
- [Data Analysis](#data-analysis)
- [Data Processing & Corrections](#data-processing--corrections)
- [Quality Control](#quality-control)
- [Feature Engineering](#feature-engineering)
- [Flux Processing](#flux-processing)
- [Gap-Filling](#gap-filling)
- [Curve Fitting](#curve-fitting)

---

## Visualization & Plotting

| Example | Description |
|---------|-------------|
| [**plot_heatmap_datetime_basic.py**](visualization/plot_heatmap_datetime_basic.py) | Datetime heatmaps with vertical/horizontal orientations, diurnal patterns |
| [**plot_heatmap_advanced.py**](visualization/plot_heatmap_advanced.py) | Year-month aggregation heatmaps, multi-variable comparison, seasonal analysis |
| [**plot_heatmap_xyz_basic.py**](visualization/plot_heatmap_xyz_basic.py) | Pre-aggregated 2D heatmaps from GridAggregator, flux binned by two variables |
| [**plot_scatter_xy_basic.py**](visualization/plot_scatter_xy_basic.py) | 2D scatter plots for exploring variable relationships |
| [**plot_scatter_xy_colored.py**](visualization/plot_scatter_xy_colored.py) | 3D scatter with color coding, bin aggregation, trend visualization |
| [**plot_timeseries.py**](visualization/plot_timeseries.py) | Time series line plots with matplotlib |
| [**plot_timeseries_interactive.py**](visualization/plot_timeseries_interactive.py) | Interactive Bokeh plots with zoom, pan, export |
| [**plot_cumulative_basic.py**](visualization/plot_cumulative_basic.py) | Cumulative flux across all time, scenario comparison |
| [**plot_cumulative_year.py**](visualization/plot_cumulative_year.py) | Yearly cumulative sums with reference bands, annual budgets |
| [**plot_dielcycle.py**](visualization/plot_dielcycle.py) | Diurnal cycle analysis by month/season |
| [**plot_hexbin_basic.py**](visualization/plot_hexbin_basic.py) | 2D hexagonal binning with percentile normalization |
| [**plot_hexbin_advanced.py**](visualization/plot_hexbin_advanced.py) | Advanced hexbin with absolute values and overlays |
| [**plot_histogram_basic.py**](visualization/plot_histogram_basic.py) | Distribution histograms with z-score overlay |
| [**plot_histogram_yearly.py**](visualization/plot_histogram_yearly.py) | Yearly comparison histograms for temporal patterns |
| [**plot_ridgeline_basic.py**](visualization/plot_ridgeline_basic.py) | Ridge line plots with weekly grouping |
| [**plot_ridgeline_advanced.py**](visualization/plot_ridgeline_advanced.py) | Ridge line plots with monthly grouping |
| [**plot_other_plots.py**](visualization/plot_other_plots.py) | Specialized plot types (long-term anomalies, trends) |

See: [visualization/README.md](visualization/README.md)

---

## Data Analysis

| Example | Description |
|---------|-------------|
| [**analysis_correlation.py**](analysis/analysis_correlation.py) | Cross-correlation, autocorrelation, lag detection, anomaly detection |
| [**analysis_daily_correlation.py**](analysis/analysis_daily_correlation.py) | Daily correlation coefficients between time series, quality checks, relationship analysis |
| [**analysis_granger.py**](analysis/analysis_granger.py) | Granger causality testing for predictive relationships between time series |
| [**analysis_seasonaltrend.py**](analysis/analysis_seasonaltrend.py) | STL decomposition, trend isolation, seasonality extraction |
| [**analysis_decoupling.py**](analysis/analysis_decoupling.py) | Stratified binning to reveal how ecosystem responses change across temperature ranges |
| [**analysis_gapfinder.py**](analysis/analysis_gapfinder.py) | Identify and characterize consecutive missing data periods in time series |
| [**analysis_gridaggregator.py**](analysis/analysis_gridaggregator.py) | 2D grid aggregation with quantile, equal-width, and custom binning methods |
| [**analysis_histogram_distribution.py**](analysis/analysis_histogram_distribution.py) | Histogram binning methods: fixed bins, unique values, fringe bin removal |
| [**analysis_optimumrange.py**](analysis/analysis_optimumrange.py) | Find optimal value ranges, condition-based filtering |
| [**analysis_quantiles.py**](analysis/analysis_quantiles.py) | Percentile and quantile calculations, non-parametric statistics |
| [**analysis_harmonic.py**](analysis/analysis_harmonic.py) | Spectral analysis, Fourier decomposition, frequency content |

See: [analysis/README.md](analysis/README.md)

---

## Data Processing & Corrections

### Offset Corrections

| Example | Description |
|---------|-------------|
| [**correction_relativehumidity_offset.py**](preprocessing/corrections/correction_relativehumidity_offset.py) | Fix RH measurements exceeding 100% due to sensor saturation |
| [**correction_radiation_offset.py**](preprocessing/corrections/correction_radiation_offset.py) | Remove radiation nighttime offset, non-zero readings at night |
| [**correction_measurement_offset_replicate.py**](preprocessing/corrections/correction_measurement_offset_replicate.py) | Detect constant bias between two instruments |
| [**correction_winddir_offset.py**](preprocessing/corrections/correction_winddir_offset.py) | Correct wind direction calibration drift |

### Value Replacement & Clipping

| Example | Description |
|---------|-------------|
| [**correction_set_exact_values_to_missing.py**](preprocessing/corrections/correction_set_exact_values_to_missing.py) | Replace exact values with NaN (error codes, sentinel values) |
| [**correction_setto_value.py**](preprocessing/corrections/correction_setto_value.py) | Replace values in time periods with constant (malfunction times) |
| [**correction_setto_threshold.py**](preprocessing/corrections/correction_setto_threshold.py) | Clip values to physically realistic min/max bounds |

### Timestamp Handling

| Example | Description |
|---------|-------------|
| [**times_timestamp_sanitizer.py**](times/times_timestamp_sanitizer.py) | Clean, validate, regularize datetime indices, gap filling, frequency detection |

See: [pkgs/preprocessing/corrections/README.md](preprocessing/corrections/README.md)

---

## Quality Control

### Outlier Detection

| Example | Method | Use |
|---------|--------|-----|
| [**outlier_hampel.py**](preprocessing/outlierdetection/outlier_hampel.py) | Hampel filter (MAD-based) | Robust spike detection |
| [**outlier_zscore.py**](preprocessing/outlierdetection/outlier_zscore.py) | Z-score (global, day/night, rolling) | Statistical thresholding |
| [**outlier_localsd.py**](preprocessing/outlierdetection/outlier_localsd.py) | Local standard deviation | Adaptive thresholds |
| [**outlier_absolutelimits.py**](preprocessing/outlierdetection/outlier_absolutelimits.py) | Min/max thresholds | Known physical limits |
| [**outlier_incremental.py**](preprocessing/outlierdetection/outlier_incremental.py) | Increment-based detection | Abrupt changes |
| [**outlier_lof.py**](preprocessing/outlierdetection/outlier_lof.py) | Local Outlier Factor | Density-based anomalies |
| [**outlier_manualremoval.py**](preprocessing/outlierdetection/outlier_manualremoval.py) | Manual point/range removal | Known issues |
| [**outlier_trim.py**](preprocessing/outlierdetection/outlier_trim.py) | Trimmed mean approach | Symmetric removal |
| [**outlier_stepwise.py**](preprocessing/outlierdetection/outlier_stepwise.py) | Chain multiple methods | Multi-stage QA/QC |

### Overall Quality Flags

| Example | Description |
|---------|-------------|
| [**qc_overall_flag.py**](preprocessing/qaqc/qc_overall_flag.py) | Combine multiple test flags into overall QCF (0=good, 1=marginal, 2=poor) |
| [**qc_eddypro_flags.py**](preprocessing/qaqc/qc_eddypro_flags.py) | Extract EddyPro quality flags (VM97 tests, signal strength, completeness) |

See: [preprocessing/outlierdetection/README.md](preprocessing/outlierdetection/README.md) and [preprocessing/qaqc/README.md](preprocessing/qaqc/README.md)

---

## Feature Engineering

| Example | Description |
|---------|-------------|
| [**feature_daynightflag.py**](features/feature_daynightflag.py) | Daytime/nighttime classification from solar geometry |
| [**feature_vpd.py**](features/feature_vpd.py) | Vapor Pressure Deficit calculation |
| [**feature_air.py**](features/feature_air.py) | Air properties (density, resistance, heat capacity) |
| [**feature_conversions.py**](features/feature_conversions.py) | Unit conversions (temperature, energy, water) |
| [**feature_potentialradiation.py**](features/feature_potentialradiation.py) | Clear-sky radiation calculation |
| [**feature_laggedvariants.py**](features/feature_laggedvariants.py) | Lagged and shifted variable creation for modeling |
| [**feature_timesince.py**](features/feature_timesince.py) | Time-since-event features |
| [**feature_noise.py**](features/feature_noise.py) | Synthetic noise generation |

See: [features/README.md](features/README.md)

---

## Flux Processing

### Multi-Level Processing Pipeline

| Example | Description |
|---------|-------------|
| [**fluxprocessingchain.py**](flux/fluxprocessingchain/fluxprocessingchain.py) | Complete Swiss FluxNet workflow (L2-L4.1): quality flags, storage correction, outlier removal, USTAR filtering, gap-filling |

### Low-Resolution (30-min) Processing

| Example | Description |
|---------|-------------|
| [**flux_timelag_analysis.py**](flux/lowres/flux_timelag_analysis.py) | Time lag detection and visualization for gas concentrations |
| [**flux_common.py**](flux/lowres/flux_common.py) | Flux variable base detection and nomenclature |
| [**flux_hqflux.py**](flux/lowres/flux_hqflux.py) | Extract highest-quality flux using Hampel filter |
| [**flux_selfheating.py**](flux/lowres/flux_selfheating.py) | SCOP self-heating correction for open-path IRGA |
| [**flux_uncertainty.py**](flux/lowres/flux_uncertainty.py) | Random uncertainty estimation (PAS20 method) |
| [**flux_ustar_mp_detection.py**](flux/lowres/flux_ustar_mp_detection.py) | Moving Point (MP) USTAR detection |

### High-Resolution (10 Hz) Analysis

| Example | Description |
|---------|-------------|
| [**flux_lag.py**](flux/hires/flux_lag.py) | Time lag detection via covariance analysis |
| [**flux_windrotation.py**](flux/hires/flux_windrotation.py) | Wind rotation and coordinate transformation |
| [**flux_fluxdetectionlimit.py**](flux/hires/flux_fluxdetectionlimit.py) | Flux detection limit and measurement sensitivity |

See: [flux/README.md](flux/README.md)

---

## Gap-Filling

| Example | Method | Training |
|---------|--------|----------|
| [**gapfill_interpolate_conservative.py**](gapfilling/gapfill_interpolate_conservative.py) | Linear interpolation | No |
| [**gapfill_interpolate_generous.py**](gapfilling/gapfill_interpolate_generous.py) | Linear interpolation | No |
| [**gapfill_randomforest.py**](gapfilling/gapfill_randomforest.py) | Random Forest | Yes |
| [**gapfill_quickfill.py**](gapfilling/gapfill_quickfill.py) | Quick Random Forest | Yes |
| [**gapfill_optimize_randomforest.py**](gapfilling/gapfill_optimize_randomforest.py) | Random Forest with hyperparameter tuning | Yes |
| [**gapfill_xgboost.py**](gapfilling/gapfill_xgboost.py) | XGBoost | Yes |
| [**gapfill_optimize_xgboost.py**](gapfilling/gapfill_optimize_xgboost.py) | XGBoost with hyperparameter tuning | Yes |
| [**gapfill_mds.py**](gapfilling/gapfill_mds.py) | Meteorological Data Similarity | No |
| [**gapfill_mds_comparison.py**](gapfilling/gapfill_mds_comparison.py) | MDS variants comparison | No |
| [**gapfill_comparison.py**](gapfilling/gapfill_comparison.py) | Compare all methods side-by-side | Mixed |

See: [gapfilling/README.md](gapfilling/README.md)

---

## Curve Fitting

| Example | Description |
|---------|-------------|
| [**fit_binfittercp.py**](fits/fit_binfittercp.py) | Binned curve fitting with confidence/prediction intervals, result exploration |
| [**fit_fitter.py**](fits/fit_fitter.py) | Ecosystem driver-response fitting, NEE-VPD relationship, uncertainty quantification |

See: [fits/README.md](fits/README.md)

---

## File I/O

| Example | Description |
|---------|-------------|
| [**io_extract.py**](io/io_extract.py) | Binary value extraction, bit-level data manipulation |

See: [io/README.md](io/README.md)

---

## Example Data

The examples use a single standardized example dataset for consistency and reproducibility. However, this represents just one example of time series data structure and quality. Real-world datasets may have:

- Different time frequencies (hourly, daily, sub-30min, etc.)
- Different gap patterns and sizes
- Different quality characteristics and noise levels
- Domain-specific variables and units
- Various timestamp formats and timezone conventions

All examples demonstrate principles applicable to different datasets. Adapt the code to your specific data format and requirements.
