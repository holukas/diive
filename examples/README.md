# DIIVE Examples

Executable examples demonstrating how to use DIIVE for time series processing, gap-filling, quality control, and visualization.

**82 examples across 9 organized folders by functional domain.**

## Structure

Examples are organized by **functional domain**, not source code structure:

```
examples/
├── visualization/         # Plotting and visualization (17 examples)
│   ├── plot_heatmap_datetime_basic.py
│   ├── plot_heatmap_advanced.py
│   ├── plot_heatmap_xyz_basic.py
│   ├── plot_hexbin_basic.py
│   ├── plot_hexbin_advanced.py
│   ├── plot_histogram_basic.py
│   ├── plot_histogram_yearly.py
│   ├── plot_scatter_xy_basic.py
│   ├── plot_scatter_xy_colored.py
│   ├── plot_timeseries.py
│   ├── plot_cumulative_basic.py
│   ├── plot_cumulative_year.py
│   ├── plot_dielcycle.py
│   ├── plot_ridgeline_basic.py
│   ├── plot_ridgeline_advanced.py
│   ├── plot_other_plots.py
│   └── plot_timeseries_interactive.py
├── times/                 # Timestamp handling (5 examples)
│   ├── times_timestamp_sanitizer.py
│   ├── times_frequency_detection.py
│   ├── times_diel_cycles.py
│   ├── times_temporal_matrices.py
│   └── times_statistics.py
├── analysis/              # Time series analysis (9 examples)
│   ├── analysis_correlation.py
│   ├── analysis_seasonaltrend.py
│   ├── analysis_gapfinder.py
│   └── ...
├── features/              # Variable creation & engineering (11 examples)
│   ├── feature_engineer.py
│   ├── feature_sonic_temp_conversion.py
│   ├── feature_latent_heat.py
│   ├── feature_evapotranspiration.py
│   ├── feature_vpd.py
│   ├── feature_daynightflag.py
│   └── ...
├── fits/                  # Data fitting (2 examples)
│   ├── fit_fitter.py
│   └── fit_binfittercp.py
├── flux/                  # Eddy covariance flux processing (10 examples)
│   ├── fluxprocessingchain/
│   │   └── fluxprocessingchain.py
│   ├── lowres/            # Low-resolution processing (7 examples)
│   │   ├── flux_timelag_analysis.py
│   │   ├── flux_common.py
│   │   ├── flux_hqflux.py
│   │   ├── flux_selfheating.py
│   │   ├── flux_selfheating_production.py
│   │   ├── flux_uncertainty.py
│   │   └── flux_ustar_mp_detection.py
│   └── hires/             # High-resolution analysis (3 examples)
│       ├── flux_fluxdetectionlimit.py
│       ├── flux_lag.py
│       └── flux_windrotation.py
├── gapfilling/            # Gap-filling methods (10 examples)
│   ├── gapfill_interpolate_generous.py
│   ├── gapfill_interpolate_conservative.py
│   ├── gapfill_randomforest.py
│   ├── gapfill_quickfill.py
│   ├── gapfill_optimize_randomforest.py
│   ├── gapfill_xgboost.py
│   ├── gapfill_optimize_xgboost.py
│   ├── gapfill_mds.py
│   ├── gapfill_mds_comparison.py
│   └── gapfill_comparison.py
├── io/                    # File I/O (1 example)
│   └── io_extract.py
└── preprocessing/         # Data quality & corrections (18 examples)
    ├── corrections/       # Offset & bias corrections (7 examples)
    │   ├── correction_relativehumidity_offset.py
    │   ├── correction_radiation_offset.py
    │   ├── correction_measurement_offset_replicate.py
    │   ├── correction_winddir_offset.py
    │   ├── correction_set_exact_values_to_missing.py
    │   ├── correction_setto_value.py
    │   └── correction_setto_threshold.py
    ├── outlierdetection/  # 9 detection methods (9 examples)
    │   ├── outlier_hampel.py
    │   ├── outlier_zscore.py
    │   ├── outlier_localsd.py
    │   ├── outlier_lof.py
    │   └── ...
    └── qaqc/              # Quality flags & EddyPro QC (2 examples)
        ├── qc_overall_flag.py
        └── qc_eddypro_flags.py
```

## Quick Start

**Run a single example:**

```bash
uv run python examples/visualization/plot_heatmap_datetime_basic.py
uv run python examples/gapfilling/gapfill_randomforest.py
```

**Run all examples in parallel:**

```bash
uv run python examples/run_all_examples.py
```

This runs all 82 examples in parallel with 8 workers, reporting execution time and any errors.

## Finding Examples

Each category folder has a **README.md** with:
- Brief description of examples in that folder
- List of files with what each demonstrates
- Links to relevant documentation
- Usage examples

Browse by topic:

- **visualization/README.md** — 16 plot types (heatmaps, scatter, timeseries, etc.)
- **times/README.md** — Timestamp validation and regularization
- **analysis/README.md** — Correlation, decomposition, gap detection, spectral analysis
- **features/README.md** — Variable creation, unit conversions, derived properties
- **fits/README.md** — Polynomial and custom curve fitting
- **flux/README.md** — Multi-level flux processing (L2-L4.1), quality filtering, high-res analysis
- **gapfilling/README.md** — Linear, Random Forest, XGBoost, MDS methods
- **io/README.md** — Binary value extraction and encoding
- **preprocessing/corrections/README.md** — Offset corrections, value clipping
- **preprocessing/outlierdetection/README.md** — 9 outlier detection methods
- **preprocessing/qaqc/README.md** — Quality flags, EddyPro integration

## Example Coverage

| Domain | Files | Topics |
|--------|-------|--------|
| **Visualization** | 17 | Heatmaps, scatter, timeseries, histograms, ridgelines, cumulative, diurnal cycles |
| **Times** | 5 | Timestamp validation, frequency detection, diel cycles, temporal matrices, statistics |
| **Analysis** | 9 | Correlation, decomposition, gap detection, spatial aggregation, harmonic analysis |
| **Features** | 11 | Feature engineering pipeline, air properties, unit conversions, day/night flags, VPD, lagged variants |
| **Fits** | 2 | Binned fitting, ecosystem response fitting |
| **Flux** | 11 | Time lag analysis, processing chain, HQ filtering, USTAR detection, self-heating (2), uncertainty, high-res analysis |
| **Gapfilling** | 10 | Linear interpolation, Random Forest (3 variants), XGBoost (3 variants), MDS (2), comparison |
| **IO** | 1 | Binary value extraction |
| **Preprocessing** | 18 | Corrections (7), outlier detection (9), QA/QC (2) |
| **TOTAL** | **82** | **~100+ individual functions demonstrated** |

## Running Options

```bash
# Run one specific example
uv run python examples/analysis/analysis_correlation.py

# Run all examples in parallel
uv run python examples/run_all_examples.py

# Run examples by category
uv run python examples/visualization/plot_heatmap_datetime_basic.py
uv run python examples/flux/hires/flux_lag.py
```

## Documentation & Guides

- **[CATALOG.md](CATALOG.md)** — Find examples by use case (workflows, analysis types, methods)
- **[EXAMPLE_DATASET.md](EXAMPLE_DATASET.md)** — Complete documentation of the example dataset (columns, availability, quality)

## Contributing

When adding new examples:
1. Place in the appropriate functional domain folder (`visualization/`, `analysis/`, `gapfilling/`, etc.)
2. Add file reference to the category's **README.md**
3. Add file path to `run_all_examples.py` in the correct section
4. Update **CATALOG.md** if introducing a new use case
5. Ensure example runs in <60 seconds
