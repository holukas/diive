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
├── flux/                  # Flux quality and analysis examples
│   ├── [common.py](flux/common.py)                            # Flux variable base detection (1 example)
│   ├── [hqflux.py](flux/hqflux.py)                            # High-quality CO2 flux analysis with Hampel filter (1 example)
│   ├── [selfheating.py](flux/selfheating.py)                  # Self-heating correction with SCOP methodology (1 example)
│   ├── [uncertainty.py](flux/uncertainty.py)                  # Random uncertainty quantification (PAS20 method) (1 example)
│   └── [ustarthreshold.py](flux/ustarthreshold.py)            # USTAR threshold detection and multiple filtering scenarios (3 examples)
└── gap_filling/           # Gap-filling workflow examples (TODO)
    ├── quick_start.py             # Simple interpolation + quickfill (TODO)
    ├── randomforest_ts.py         # RandomForestTS examples (TODO)
    ├── xgboost_ts.py              # XGBoostTS examples (TODO)
    ├── longterm_models.py         # Long-term multi-year models (TODO)
    └── mds_filling.py             # MDS meteorological similarity (TODO)
```

## Running Examples

**Run all examples at once:**
```bash
python examples/run_all_examples.py
```

Executes all 74 examples (22 visualization + 8 analysis + 2 binary + 7 corrections + 23 createvar + 4 echires + 7 flux + 1 fits) in parallel (4 concurrent workers) with execution time tracking.
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

**Gap-Filling (TODO - Phase 2):**
```bash
python examples/gap_filling/quick_start.py
```

Demonstrates simple interpolation and quick Random Forest gap-filling.

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

- **Phase 1 (Complete):** Core examples across visualization, analysis, and data processing
  - **Visualization:** HeatmapDateTime/YearMonth (6), HexbinPlot (3), TimeSeries (1), Cumulative (3), Other (1), DielCycle (1), Histogram (2), RidgeLine (2), ScatterXY (3) = 22 examples
  - **Analysis:** DailyCorrelation, StratifiedAnalysis, GapFinder, GridAggregator, Histogram, FindOptimumRange, Quantiles, SeasonalTrendDecomposition = 8 examples
  - **Data Processing:** Binary (2), Corrections (7), Variable creation (23) = 32 examples
  - **Eddy Covariance:** FluxDetectionLimit (2), MaxCovariance (1), WindRotation2D (1), Flux (7 - common + hqflux + selfheating + uncertainty + 3 ustarthreshold) = 11 examples
  - **Fits:** BinFitterCP (1) = 1 example
  - **Total:** 74 examples
- **Phase 2 (Planned):** Gap-filling workflow examples and HeatmapXYZ
  - Quick start interpolation + quickfill
  - RandomForestTS, XGBoostTS, long-term models, MDS gap-filling
- **Phase 3+ (Future):** Advanced feature engineering examples
