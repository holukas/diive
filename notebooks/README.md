![](../images/logo_diive1_128px.png)

# DIIVE Examples & Notebooks

---

## Python Examples (Primary Reference)

Most functionality is now available as runnable Python scripts in the `examples/` folder — these are maintained, tested,
and the primary reference for learning diive.

**113 example scripts** organized by topic (visualization, gap-filling, flux processing, quality control, feature
engineering, and more)

**See the complete catalog:** [examples/CATALOG.md](../examples/CATALOG.md)

You can run examples directly:

```bash
uv run python examples/gapfilling/gapfill_randomforest.py
uv run python examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py
python examples/run_all_examples.py  # Run all in parallel
```

See [examples/README.md](../examples/README.md) for detailed instructions.

---

## Active Notebooks

The following notebooks provide specialized functionality for data processing and flux analysis. Most examples have been
migrated to Python scripts — **use the Python examples above for learning and standard workflows**.

**Note:** the two `FormatMeteo*` notebooks and the four InfluxDB notebooks read from an InfluxDB database, so they
need a database connection and a config directory to run. `FormatEddyProFluxnetFileForUpload` reads EddyPro output
files from disk instead.

### Flux Workflows

- [FluxProcessingChain.ipynb](FluxProcessingChain.ipynb) — Post-processing of Level-1 fluxes:
  quality flag extension (L2), storage correction (L3.1), outlier removal (L3.2), USTAR threshold (L3.3), gap-filling
  (L4.1) with random forest, XGBoost and MDS, then model reporting, result plots and export. Runs on the bundled
  example data. Rewritten for the composable per-level API in v0.91.0

### Database (InfluxDB)

- [DatabaseInfluxStepwiseMeteoScreening.ipynb](DatabaseInfluxStepwiseMeteoScreening.ipynb) — Quality screening of
  meteorological data with direct database connection: download, screen/correct on high-res data, resample, upload
  (`StepwiseMeteoScreeningDb`, `InfluxIO`)
- [DatabaseInfluxDownloadSpecificVars.ipynb](DatabaseInfluxDownloadSpecificVars.ipynb) — Download specific variables from
  the InfluxDB database into a DataFrame, inspect, and optionally save to file (read-only, `InfluxIO`)
- [DatabaseInfluxDownloadAllVarsOfMeasurements.ipynb](DatabaseInfluxDownloadAllVarsOfMeasurements.ipynb) — Download every
  variable of one or more measurements at once (`FIELDS = None`), across one or several data versions (read-only,
  `InfluxIO`)
- [DatabaseInfluxDeleteData.ipynb](DatabaseInfluxDeleteData.ipynb) — Permanently delete variables or an entire data version
  from the InfluxDB database (destructive; delete calls are commented out by default, `InfluxIO`)

### Data Formatting & File I/O

- [FormatEddyProFluxnetFileForUpload.ipynb](FormatEddyProFluxnetFileForUpload.ipynb) — Prepare
  EddyPro output for FLUXNET database upload
- [FormatMeteoForEddyProFluxProcessing.ipynb](FormatMeteoForEddyProFluxProcessing.ipynb) — Format
  meteorological data for EddyPro flux processing
- [FormatMeteoForFluxnetUpload.ipynb](FormatMeteoForFluxnetUpload.ipynb) — Format meteorological
  data for FLUXNET database submission

### Workbench (Testing & Experimental)

Additional notebooks for testing and site-specific analysis are available in the `Workbench/` subdirectory. These are
development and scratch notebooks, not part of the standard workflow.

---

## Quick Links

- [examples/CATALOG.md](../examples/CATALOG.md) — Complete index of 113 Python examples
- [examples/README.md](../examples/README.md) — Examples folder overview
- [CLAUDE.md](../CLAUDE.md) — Development guide and project overview
- [CHANGELOG.md](../CHANGELOG.md) — Version history and recent updates
