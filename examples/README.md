# DIIVE Examples

Executable examples demonstrating how to use DIIVE for time series processing, gap-filling, quality control, and visualization.

**105 examples across 9 organized folders by functional domain.**

## Structure

Examples are organized by **functional domain**, not source code structure:

```
examples/
├── visualization/         # Plotting and visualization (19 examples)
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
│   ├── plot_timeseries_interactive.py
│   ├── plot_treering_temperature.py
│   └── plot_treering_line_temperature.py
├── times/                 # Timestamp handling (5 examples)
│   ├── times_timestamp_sanitizer.py
│   ├── times_frequency_detection.py
│   ├── times_time_features.py
│   ├── times_diel_cycles.py
│   └── times_temporal_matrices.py
├── analysis/              # Time series analysis (12 examples)
│   ├── analysis_seasonaltrend.py
│   ├── analysis_gapfinder.py
│   ├── analysis_gapstats.py
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
├── flux/                  # Eddy covariance flux processing (20 examples)
│   ├── fluxprocessingchain/
│   │   ├── fluxprocessingchain_runchain.py
│   │   └── fluxprocessingchain_composable.py
│   ├── lowres/            # Low-resolution processing (9 examples)
│   │   ├── flux_timelag_analysis.py
│   │   ├── flux_common.py
│   │   ├── flux_hqflux.py
│   │   ├── flux_selfheating.py
│   │   ├── flux_selfheating_production.py
│   │   ├── flux_uncertainty.py
│   │   ├── flux_ustar_mp_detection.py
│   │   ├── flux_ustar_vekuri_detection.py
│   │   └── flux_ustar_method_comparison.py
│   └── hires/             # High-resolution analysis (7 examples)
│       ├── flux_fluxdetectionlimit.py
│       ├── flux_lag.py
│       ├── flux_lag_pwb.py
│       ├── flux_lag_pwbopt.py
│       ├── flux_lag_pwb_batch.py
│       ├── flux_lag_pwb_batch_cli.py
│       └── flux_windrotation.py
├── gapfilling/            # Gap-filling methods (12 examples)
│   ├── gapfill_interpolate_generous.py
│   ├── gapfill_interpolate_conservative.py
│   ├── gapfill_randomforest.py
│   ├── gapfill_randomforest_longterm.py
│   ├── gapfill_quickfill.py
│   ├── gapfill_optimize_randomforest.py
│   ├── gapfill_xgboost.py
│   ├── gapfill_optimize_xgboost.py
│   ├── gapfill_mds.py
│   ├── gapfill_mds_comparison.py
│   ├── gapfill_swin.py
│   └── gapfill_comparison.py
├── io/                    # File I/O (5 examples)
│   ├── io_load_save_parquet.py
│   ├── io_read_single_file_with_datafilereader.py
│   ├── io_read_multiple_files_with_multidatafilereader.py
│   ├── io_read_single_file_with_readfiletype.py
│   └── io_extract.py
└── preprocessing/         # Data quality & corrections (20 examples)
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
    └── qaqc/              # Quality flags & EddyPro QC (3 examples)
        ├── qc_overall_flag.py
        ├── qc_eddypro_flags.py
        └── qaqc_detect_timestamp_shifts.py
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

This runs all 105 examples in parallel with 8 workers, reporting execution time and any errors.

## Finding Examples

Each category folder has a **README.md** with:
- Brief description of examples in that folder
- List of files with what each demonstrates
- Links to relevant documentation
- Usage examples

Browse by topic:

- **visualization/README.md** — 19 plot types (heatmaps, scatter, timeseries, tree-ring, etc.)
- **times/README.md** — Timestamp validation and regularization
- **analysis/README.md** — Correlation, decomposition, gap detection, spectral analysis
- **features/README.md** — Variable creation, unit conversions, derived properties
- **fits/README.md** — Polynomial and custom curve fitting
- **flux/README.md** — Multi-level flux processing (L2-L4.1), quality filtering, high-res analysis
- **gapfilling/README.md** — Linear, Random Forest, XGBoost, MDS methods
- **io/README.md** — Binary value extraction and encoding
- **preprocessing/corrections/README.md** — Offset corrections, value clipping
- **preprocessing/outlier_detection/README.md** — 9 outlier detection methods
- **preprocessing/qaqc/README.md** — Quality flags, EddyPro integration

## Example Coverage

| Domain | Files | Topics |
|--------|-------|--------|
| **Visualization** | 19 | Heatmaps, scatter, timeseries, histograms, ridgelines, cumulative, diurnal cycles, tree-ring spiral and radial line |
| **Times** | 5 | Timestamp validation, frequency detection, diel cycles, temporal matrices |
| **Analysis** | 12 | Correlation, daily correlation, decomposition, gap detection, gap statistics, spatial aggregation, harmonic analysis |
| **Features** | 11 | Feature engineering pipeline, air properties, unit conversions, day/night flags, VPD, lagged variants |
| **Fits** | 2 | Binned fitting, ecosystem response fitting |
| **Flux** | 18 | Time lag analysis, processing chain, HQ filtering, USTAR detection (3), self-heating (2), uncertainty, PWB batch detection (CLI + API), high-res analysis |
| **Gapfilling** | 12 | Linear interpolation, Random Forest (4 variants), XGBoost (3 variants), MDS (2), SW_IN physics+XGBoost, comparison |
| **IO** | 5 | Parquet file I/O, EddyPro CSV reading, binary value extraction |
| **Preprocessing** | 21 | Corrections (7), outlier detection (9), QA/QC (3), other (2) |
| **TOTAL** | **105** | |

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

- **[COOKBOOK.md](COOKBOOK.md)** — Start here: 6 minimal working workflows (load data, clean timestamps, remove outliers, gap-fill, flux chain, visualize)
- **[CATALOG.md](CATALOG.md)** — Find examples by use case (workflows, analysis types, methods)
- **[EXAMPLE_DATASET.md](EXAMPLE_DATASET.md)** — Complete documentation of the example dataset (columns, availability, quality)

## Contributing

When adding new examples:
1. Place in the appropriate functional domain folder (`visualization/`, `analysis/`, `gapfilling/`, etc.)
2. Add file reference to the category's **README.md**
3. Add file path to `run_all_examples.py` in the correct section
4. Update **CATALOG.md** if introducing a new use case
5. Ensure example runs in <60 seconds
