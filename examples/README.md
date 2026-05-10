# DIIVE Examples

Executable examples demonstrating how to use DIIVE for time series processing, gap-filling, quality control, and visualization.

**62 examples across 18 organized folders, mirroring the `diive` package structure.**

## Structure

The examples are organized into two main categories:

### **Core** — System-level utilities

```
examples/core/
├── visualization/         # Plotting and visualization (9 examples)
│   ├── heatmap_datetime.py
│   ├── scatter_xy.py
│   ├── timeseries.py
│   └── ... (see README)
└── times/                 # Timestamp handling (1 example)
    └── timestamp_sanitizer.py
```

### **Pkgs** — Domain-specific packages

```
examples/pkgs/
├── analysis/              # Time series analysis (9 examples)
│   ├── correlation.py
│   ├── seasonaltrend.py
│   ├── gapfinder.py
│   └── ... (see README)
├── features/              # Variable creation & engineering (8 examples)
│   ├── vpd.py
│   ├── daynightflag.py
│   ├── conversions.py
│   └── ... (see README)
├── fits/                  # Data fitting (1 example)
│   └── fitter.py
├── flux/                  # Eddy covariance flux processing (11 examples)
│   ├── common.py
│   ├── hires/             # High-resolution analysis (3 examples)
│   │   ├── lag.py
│   │   ├── windrotation.py
│   │   └── fluxdetectionlimit.py
│   ├── hqflux/            # Highest-quality flux filtering (1 example)
│   ├── selfheating/       # Sensor correction (1 example)
│   ├── uncertainty/       # Random uncertainty (1 example)
│   ├── ustarthreshold/    # USTAR filtering (1 example)
│   └── ustar_mp_detection/ # Moving Point USTAR (1 example)
├── gapfilling/            # Gap-filling methods (6 examples)
│   ├── randomforest_ts.py
│   ├── xgboost_ts.py
│   ├── mds.py
│   ├── comparison.py
│   └── ... (see README)
├── io/                    # File I/O (1 example)
│   └── extract.py
└── preprocessing/         # Data quality & corrections (20 examples)
    ├── corrections/       # Offset & bias corrections (8 examples)
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
    │   └── ... (see README)
    └── qaqc/              # Quality flags & EddyPro QC (2 examples)
        ├── qc_overall_flag.py
        └── qc_eddypro_flags.py
```

## Quick Start

**Run a single example:**

```bash
uv run python examples/core/visualization/heatmap_datetime.py
uv run python examples/pkgs/gapfilling/randomforest_ts.py
```

**Run all examples in parallel:**

```bash
uv run python examples/run_all_examples.py
```

This runs all 58 examples in parallel with 8 workers, reporting execution time and any errors.

## Finding Examples

Each category folder has a **README.md** with:
- Brief description of examples in that folder
- List of files with what each demonstrates
- Links to relevant documentation
- Usage examples

Browse by topic:

- **[core/visualization/README.md](core/visualization/README.md)** — Plotting types and visualization
- **[pkgs/analysis/README.md](pkgs/analysis/README.md)** — Time series analysis methods
- **[pkgs/features/README.md](pkgs/features/README.md)** — Variable creation & conversions
- **[pkgs/flux/README.md](pkgs/flux/README.md)** — Flux processing pipeline
- **[pkgs/gapfilling/README.md](pkgs/gapfilling/README.md)** — Gap-filling algorithms
- **[pkgs/preprocessing/outlierdetection/README.md](pkgs/preprocessing/outlierdetection/README.md)** — Outlier detection methods
- **[pkgs/preprocessing/qaqc/README.md](pkgs/preprocessing/qaqc/README.md)** — Quality control flags

## Example Coverage

| Category | Files | Topics |
|----------|-------|--------|
| **Core** | 10 | Visualization (9), timestamp handling (1) |
| **Analysis** | 8 | Correlation, decomposition, gap detection, spatial aggregation |
| **Features** | 8 | Air properties, unit conversions, day/night flags, VPD |
| **Fits** | 1 | Polynomial fitting |
| **Flux** | 10 | Processing chain, HQ filtering, USTAR detection, self-heating, uncertainty, high-res analysis |
| **Gapfilling** | 6 | RandomForest, XGBoost, MDS, linear interpolation, comparisons |
| **IO** | 1 | Binary file operations |
| **Preprocessing** | 19 | Corrections (8), outlier detection (9), QA/QC (2) |
| **TOTAL** | **62** | **~100+ individual functions demonstrated** |

## Running Options

```bash
# Run one specific example
uv run python examples/pkgs/analysis/correlation.py

# Run all examples in a folder
uv run python examples/run_all_examples.py

# Run examples by category
uv run python examples/core/visualization/heatmap_datetime.py
uv run python examples/pkgs/flux/hires/lag.py
```

## Documentation & Guides

- **[CATALOG.md](CATALOG.md)** — Find examples by use case (workflows, analysis types, methods)
- **[EXAMPLE_DATASET.md](EXAMPLE_DATASET.md)** — Complete documentation of the example dataset (columns, availability, quality)

## Contributing

When adding new examples:
1. Place in the appropriate folder under `core/` or `pkgs/`
2. Add file reference to the category's **README.md**
3. Add file path to `run_all_examples.py` in the correct section
4. Update **CATALOG.md** if introducing a new use case
5. Ensure example runs in <60 seconds
