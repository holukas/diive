# DIIVE Examples

Executable examples demonstrating how to use DIIVE for time series processing, gap-filling, and visualization.

## Structure

```
examples/
├── visualization/          # Plotting and visualization examples
│   ├── [heatmap_datetime.py](visualization/heatmap_datetime.py)            # HeatmapDateTime and HeatmapYearMonth (6 examples)
│   ├── [hexbin.py](visualization/hexbin.py)                      # HexbinPlot 2D hexagonal binning (3 examples)
│   ├── [timeseries_and_cumulative.py](visualization/timeseries_and_cumulative.py)   # Cumulative and CumulativeYear (3 examples)
│   ├── [other_plots.py](visualization/other_plots.py)                 # LongtermAnomaliesYear (1 example)
│   ├── [timeseries.py](visualization/timeseries.py)                  # TimeSeries interactive plots (1 example)
│   ├── [dielcycle.py](visualization/dielcycle.py)                   # DielCycle diurnal analysis (1 example)
│   ├── [histogram.py](visualization/histogram.py)                   # HistogramPlot distribution analysis (2 examples)
│   ├── [ridgeline.py](visualization/ridgeline.py)                   # RidgeLinePlot kernel density plots (2 examples)
│   ├── [scatter_xy.py](visualization/scatter_xy.py)                  # ScatterXY scatter plots (3 examples)
│   └── heatmap_xyz.py                 # HeatmapXYZ 3D scatter heatmaps (TODO)
├── analyses/               # Time series analysis examples
│   ├── [correlation.py](analyses/correlation.py)                 # DailyCorrelation analysis with statistics and anomaly detection (1 example)
│   ├── [decoupling.py](analyses/decoupling.py)                  # StratifiedAnalysis for hierarchical binning analysis (1 example)
│   ├── [gapfinder.py](analyses/gapfinder.py)                   # GapFinder for gap detection and analysis (1 example)
│   ├── [gridaggregator.py](analyses/gridaggregator.py)              # GridAggregator for 2D grid-based aggregation (1 example)
│   ├── [histogram.py](analyses/histogram.py)                   # Histogram for distribution analysis (1 example)
│   ├── [optimumrange.py](analyses/optimumrange.py)                # FindOptimumRange for optimal condition analysis (1 example)
│   ├── [quantiles.py](analyses/quantiles.py)                   # percentiles101 for distribution analysis (1 example)
│   └── [seasonaltrend.py](analyses/seasonaltrend.py)               # SeasonalTrendDecomposition for time series decomposition (1 example)
├── binary/                # Binary data processing examples
│   └── [extract.py](binary/extract.py)                   # Binary bit extraction from integers (2 examples)
├── corrections/           # Data correction examples
│   ├── [setto.py](corrections/setto.py)                  # Set values to missing, specific values, or thresholds (3 examples)
│   └── [offsetcorrection.py](corrections/offsetcorrection.py)  # Correct RH, radiation, measurement, and wind direction offsets (4 examples)
├── qaqc/                  # Quality assurance / Quality control examples
│   ├── [qcf.py](qaqc/qcf.py)                             # Overall Quality Control Flag (QCF) combining multiple test flags (1 example)
│   └── [eddyproflags.py](qaqc/eddyproflags.py)          # EddyPro quality flags from raw data tests (Vickers & Mahrt, 1997) (6 examples)
├── outlierdetection/      # Outlier detection and quality control examples
│   ├── [absolutelimits.py](outlierdetection/absolutelimits.py)  # Absolute value limits with separate day/night thresholds (2 examples)
│   ├── [hampel.py](outlierdetection/hampel.py)                  # Hampel filter (Median Absolute Deviation) outlier detection (2 examples)
│   ├── [incremental.py](outlierdetection/incremental.py)        # Z-score increments outlier detection (1 example)
│   ├── [zscore.py](outlierdetection/zscore.py)                  # Z-score outlier detection (3 examples: global, day/night, rolling)
│   ├── [localsd.py](outlierdetection/localsd.py)                # Local standard deviation rolling window outlier detection (2 examples)
│   ├── [lof.py](outlierdetection/lof.py)                        # Local Outlier Factor density-based detection (2 examples)
│   ├── [manualremoval.py](outlierdetection/manualremoval.py)    # Manual data point/range removal for known issues (2 examples)
│   ├── [stepwise.py](outlierdetection/stepwise.py)              # Step-wise orchestration - chain multiple detection methods (1 example)
│   └── [trim.py](outlierdetection/trim.py)                      # Trim filter - symmetric removal of low and high outliers (2 examples)
├── createvar/             # Derived variable creation examples
│   ├── [air.py](createvar/air.py)                             # Air properties: aerodynamic resistance and dry air density (2 examples)
│   ├── [conversions.py](createvar/conversions.py)              # Unit conversions: air temperature, latent heat, evapotranspiration (3 examples)
│   ├── [daynightflag.py](createvar/daynightflag.py)           # Daytime/nighttime flags from solar geometry (1 example)
│   ├── [laggedvariants.py](createvar/laggedvariants.py)       # Create lagged variants for temporal analysis (3 examples)
│   ├── [noise.py](createvar/noise.py)                         # Generate synthetic data and add noise (4 examples)
│   ├── [potentialradiation.py](createvar/potentialradiation.py) # Calculate potential solar radiation (4 examples)
│   ├── [timesince.py](createvar/timesince.py)                 # Count records since condition met (3 examples)
│   └── [vpd.py](createvar/vpd.py)                             # Vapor pressure deficit calculation and gap-filling (3 examples)
├── echires/               # Eddy covariance high-resolution analysis examples
│   ├── [fluxdetectionlimit.py](echires/fluxdetectionlimit.py) # FluxDetectionLimit for minimum detectable flux (2 examples)
│   ├── [lag.py](echires/lag.py)                               # MaxCovariance for time lag detection (1 example)
│   └── [windrotation.py](echires/windrotation.py)             # WindRotation2D for coordinate rotation and tilt correction (1 example)
├── timeseries/            # Time series data preparation and analysis examples
│   ├── [timestamp_sanitizer.py](timeseries/timestamp_sanitizer.py)  # Comprehensive timestamp validation and cleaning (5 levels, clean to severely corrupted data)
│   └── [harmonic.py](timeseries/harmonic.py)                  # Spectrogram analysis - daily and annual CO2 patterns (2 examples)
├── flux/                  # Flux quality and analysis examples
│   ├── [common.py](flux/common.py)                            # Flux variable base detection (1 example)
│   ├── [hqflux.py](flux/hqflux.py)                            # High-quality CO2 flux analysis with Hampel filter (1 example)
│   ├── [selfheating.py](flux/selfheating.py)                  # Self-heating correction with SCOP methodology (1 example)
│   ├── [uncertainty.py](flux/uncertainty.py)                  # Random uncertainty quantification (PAS20 method) (1 example)
│   └── [ustarthreshold.py](flux/ustarthreshold.py)            # USTAR threshold detection and multiple filtering scenarios (3 examples)
└── gap_filling/           # Gap-filling workflow examples
    ├── [interpolate.py](gap_filling/interpolate.py)            # Linear interpolation gap-filling (2 examples: conservative & generous limits)
    ├── [mds.py](gap_filling/mds.py)                            # Marginal Distribution Sampling (1 example)
    ├── [mds_comparison.py](gap_filling/mds_comparison.py)      # Performance comparison: Original vs Optimized MDS (1 example)
    ├── [randomforest_ts.py](gap_filling/randomforest_ts.py)    # Random Forest gap-filling with harmonized feature engineering (3 examples)
    ├── [xgboost_ts.py](gap_filling/xgboost_ts.py)              # XGBoost gap-filling with hyperparameter optimization (2 examples)
    ├── [comparison.py](gap_filling/comparison.py)              # Three-way comparison: MDS vs Random Forest vs XGBoost with heatmaps and cumulative curves (1 example)
    └── longterm_models.py         # Long-term multi-year models (TODO - Phase 2)
```

## Running Examples

**Run all examples at once:**
```bash
python examples/run_all_examples.py
```

Executes all 112 examples across 53 files (22 visualization + 8 analysis + 2 binary + 7 corrections + 7 qaqc + 17 outlierdetection + 23 createvar + 4 echires + 7 flux + 10 gap_filling + 7 timeseries + 1 fits) in parallel (4 concurrent workers) with execution time tracking.
- Shows individual timing for each example
- Detailed error messages if any fail
- ~2.7x faster than sequential execution

**Run individual examples:**
Each example file is standalone and executable:

```bash
python examples/visualization/heatmap_datetime.py
python examples/gap_filling/randomforest_ts.py
```

## Quick Start

**Visualization:**
```bash
python examples/visualization/heatmap_datetime.py
```

Displays heatmaps in vertical/horizontal orientations, year-month aggregations with ranks, and colormap previews.

**Gap-Filling:**
```bash
python examples/gap_filling/randomforest_ts.py
python examples/gap_filling/xgboost_ts.py
python examples/gap_filling/mds.py
```

Gap-filling methods with comprehensive examples:
- **Harmonized ML comparison:** Random Forest vs XGBoost with identical feature engineering on same data
  - Both use 2020 data with harmonized 8-stage feature engineering (lag, rolling, STL, timestamps)
  - Side-by-side heatmap visualizations (observed vs gap-filled)
  - Cumulative flux plots for ecosystem carbon balance assessment
- Linear interpolation for conservative gap-filling
- Marginal Distribution Sampling (MDS) with meteorological similarity
- MDS performance comparison (4.0x speedup with bit-identical results)

**Flux Quality & Analysis (`examples/flux/`):**
```bash
python examples/flux/hqflux.py
```

Demonstrates robust outlier detection for CO2 net ecosystem exchange (NEE) flux using Hampel filter with automatic day/night separation. Includes summary statistics, quality metrics, and visualization of rolling percentile bands.

## Finding Help

- **API documentation:** See class docstrings (e.g., `help(HeatmapDateTime)`)
- **Examples:** Browse this folder for your use case
- **Architecture:** See `CLAUDE.md` for design decisions and workflows

## Contributing Examples

When adding a new example:
1. Create a function with a descriptive name: `example_<feature>_<variant>()`
2. Add docstring explaining what it demonstrates
3. Keep it runnable standalone: `python examples/<category>/<file>.py`
4. Use `dv.load_exampledata_parquet()` for consistent test data
5. Update this README with the new example

## Phases

- **Phase 1 (Complete):** Core examples across visualization, analysis, and data processing (94 examples)
  - **Visualization:** HeatmapDateTime/YearMonth (6), HexbinPlot (3), TimeSeries (1), Cumulative (3), Other (1), DielCycle (1), Histogram (2), RidgeLine (2), ScatterXY (3) = 22 examples
  - **Analysis:** DailyCorrelation, StratifiedAnalysis, GapFinder, GridAggregator, Histogram, FindOptimumRange, Quantiles, SeasonalTrendDecomposition = 8 examples
  - **Data Processing:** Binary (2), Corrections (7), QAQC (7 - 1 QCF + 6 EddyPro), Outlierdetection (17 - absolutelimits 2 + hampel 2 + incremental 1 + localsd 2 + lof 2 + manualremoval 2 + stepwise 1 + trim 2 + zscore 3), Variable creation (23) = 52 examples
  - **Eddy Covariance:** FluxDetectionLimit (2), MaxCovariance (1), WindRotation2D (1), Flux (7 - common + hqflux + selfheating + uncertainty + 3 ustarthreshold) = 11 examples
  - **Time Series:** Harmonic/Spectrogram (2 - daily pattern, annual phenology) = 2 examples
  - **Fits:** BinFitterCP (1) = 1 example
- **Phase 2 (Complete):** Gap-filling workflow examples (9 examples)
  - **Gap-filling:** Linear interpolation (2), MDS (1), MDS comparison (1), Random Forest (3), XGBoost (2), MDS vs RF vs XGB comparison (1) = 9 examples
  - **TOTAL Phase 1 + Phase 2:** 107 examples across 52 files
  - TODO: Long-term multi-year models, HeatmapXYZ
- **Phase 3+ (Future):** Advanced feature engineering examples
