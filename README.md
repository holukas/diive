![](images/logo_diive1_256px.png)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/diive?style=for-the-badge&color=%23EF6C00&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdiive%2F)](https://pypi.org/project/diive/)
[![GitHub License](https://img.shields.io/github/license/holukas/diive?style=for-the-badge&color=%237CB342)](https://github.com/holukas/diive/blob/indev/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/diive)](https://pepy.tech/projects/diive)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10884017.svg)](https://doi.org/10.5281/zenodo.10884017)

_**`diive` is currently being prepared for the v1.0 release.**_

# Time series data processing

`diive` is a Python library for time series processing, focused on ecosystem data. It was originally developed by the [ETH Grassland Sciences group](https://gl.ethz.ch/) for [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/).

[CHANGELOG](CHANGELOG.md) | [Releases](https://github.com/holukas/diive/releases) | [Project overview](OVERVIEW.md)

---

## Citation

Cite `diive` using DOI [10.5281/zenodo.10884017](https://doi.org/10.5281/zenodo.10884017). This concept DOI resolves to the latest release, so include the version number in your citation.

**BibTeX format:**

```bibtex
@software{diive2026,
  author = {Hörtnagl, Lukas},
  title = {diive: Python library for time series processing},
  version = {0.91.0},
  year = {2026},
  doi = {10.5281/zenodo.10884017}
}
```

Replace `version` and `year` with the values for your target release.

---

## Installation

Requires **Python 3.12+**

```bash
pip install diive
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install diive
```

**Optional desktop GUI** (PySide6, not installed by default):

```bash
pip install 'diive[gui]'   # or: uv sync --extra gui
diive-gui                  # launch
```

### Quick start

```python
import diive as dv

# Load example data (a 37-variable ecosystem dataset)
df = dv.load_exampledata_parquet()

# Plot a time series — two-phase: construct, then .plot()
dv.plotting.TimeSeries(series=df['NEE_CUT_REF_orig']).plot()

# Gap-fill with Random Forest
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.gapfilling.randomforest_ts import RandomForestTS

engineer = FeatureEngineer(target_col='NEE_CUT_REF_orig', features_lag=[-1, 1])
df_engineered = engineer.fit_transform(df)

model = RandomForestTS(input_df=df_engineered, target_col='NEE_CUT_REF_orig', n_estimators=100)
model.run()                       # trains the model, then fills gaps
gapfilled = model.results.gapfilled
```

---

## API

`import diive as dv` exposes ten domain namespaces. Classes live under the namespace for their area:

```python
import diive as dv

plot = dv.plotting.TimeSeries(series=data)
model = dv.gapfilling.RandomForestTS(input_df=df, target_col='NEE')
```

| Namespace | Common exports |
|---|---|
| `dv.plotting` | `TimeSeries`, `Cumulative`, `DielCycle`, `HeatmapDateTime` |
| `dv.gapfilling` | `RandomForestTS`, `XGBoostTS`, `FluxMDS` |
| `dv.analysis` | `GridAggregator`, `SeasonalTrendDecomposition`, `BinFitterCP` |
| `dv.flux` | `run_chain`, `FluxConfig`, `FluxDetectionLimit`, `WindDoubleRotation`, NEE partitioning (`partition_nee_*`) |
| `dv.outliers` / `dv.corrections` / `dv.qaqc` | outlier methods, offset corrections, `FlagQCF` |
| `dv.times` / `dv.variables` | timestamp sanitization, derived variables (VPD, potential radiation, ...) |
| `dv.events` | `Event`, `event_to_flag`, `overlay_events` |

A few I/O helpers are top-level: `dv.load_parquet`, `dv.save_parquet`, `dv.load_exampledata_parquet`.

For the full list, see `diive.__all__` and each namespace's `__all__`.

---

## Examples

113 runnable examples are organized by topic in [examples/](examples/README.md). They follow Sphinx Gallery format (`# %%` sections), so they run as plain scripts and convert to HTML docs automatically. Browse by use case in [CATALOG.md](examples/CATALOG.md), or check [EXAMPLE_DATASET.md](examples/EXAMPLE_DATASET.md) for documentation of the 37-variable dataset used throughout.

```bash
uv run python examples/visualization/plot_heatmap_datetime_basic.py
uv run python examples/analysis/analysis_daily_correlation.py
uv run python examples/gapfilling/gapfill_randomforest.py
uv run python examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py
```

---

## Features

### Gap-filling

`FeatureEngineer` runs an 8-stage feature pipeline (lag features, rolling stats, differencing, EMA, polynomial terms, STL decomposition, timestamps, record numbering). You build the features once and reuse them across models.

| Method | How it works |
|---|---|
| `XGBoostTS` | Gradient boosting |
| `RandomForestTS` | Ensemble learning with SHAP importance |
| `FluxMDS` | Meteorological similarity, no training needed |
| Linear interpolation | Short gaps only |

Long-term variants support multi-year data with USTAR scenario options. See [examples/gapfilling/](examples/gapfilling/).

### Flux processing chain

Post-processing from quality flags through gap-filling, covering Levels 2 to 4.1 following Swiss FluxNet standards. Two entry points:

- **`run_chain(data, config)`** — single call drives the full pipeline (L2 → L3.1 → L3.2 → L3.3 → L4.1) from one `FluxConfig`. Intentionally simple: fixed defaults for per-detector / per-model knobs (Hampel sub-options, MDS tolerances, ML hyperparameters). Use this for the standard FLUXNET-style workflow.
- **Composable per-level callables** (`run_level2`, `run_level31`, `make_level32_detector` + `run_level32`, `run_level33_constant_ustar` / `run_level33_variable_ustar` / `run_level33_ustar_detection` (CUT or VUT), `run_level41_mds` / `_rf` / `_xgb`) — full control. Every detector class, model hyperparameter, MDS tolerance, and diagnostic flag is reachable here and only here.

Need a computed driver (e.g. VPD in kPa) for L4.1? Use `add_driver(data, series)` to put it where L4.1 actually reads from. Call `data.gap_stats()` at any level for a monthly/annual breakdown with long-gap listing. `data.plot_gapfilled_heatmaps()` puts all gap-filling methods side by side; `data.plot_cumulative_comparison()` overlays their cumulative sums on one axes.

Reference: [Swiss FluxNet flux processing](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/) | Examples: [examples/flux/fluxprocessingchain/](examples/flux/fluxprocessingchain/)

### NEE partitioning

Split net ecosystem exchange (NEE) into gross primary production (GPP) and ecosystem respiration (RECO). diive ships four faithful Python ports of the standard reference routines — two **nighttime** methods (Reichstein et al. 2005, fitting the temperature response of nighttime respiration) and two **daytime** methods (Lasslop et al. 2010, fitting a light-response curve to daytime NEE). Each is validated against its reference implementation; output columns carry a `_NT` / `_DT` (nighttime / daytime) plus `_OF` / `_RP` (ONEFlux / REddyProc) token, so all four can coexist in one dataframe.

| Method (class / function) | Columns | Approach |
|---|---|---|
| `NighttimePartitioningOneFlux` / `partition_nee_nighttime_oneflux` | `*_NT_OF` | Reichstein 2005, ONEFlux port; per calendar year, incl. an outlier-robust variant |
| `NighttimePartitioningReddyProc` / `partition_nee_nighttime_reddyproc` | `*_NT_RP` | Reichstein 2005, REddyProc `sMRFluxPartition` port; whole record with a single E0 |
| `DaytimePartitioningReddyProc` / `partition_nee_daytime_reddyproc` | `*_DT_RP` | Lasslop 2010, REddyProc `partitionNEEGL` port (light-response curve) |
| `DaytimePartitioningOneFlux` / `partition_nee_daytime_oneflux` | `*_DT_OF` | Lasslop 2010, ONEFlux `flux_part_gl2010` port; incl. GPP standard error |

Inputs are in physical units — air temperature in °C, VPD in kPa (`vpd_in_kpa=True`). For the day/night split the REddyProc ports take `lat` / `lon` / `utc_offset` (solar geometry) and the ONEFlux nighttime port takes `lat`; the ONEFlux daytime port needs no coordinates (it uses a measured-radiation threshold). Each class prints a Rich per-year summary report on `.run()` (or call `.report()`). See [examples/flux/partitioning/](examples/flux/partitioning/) — including `partitioning_comparison.py`, which runs all four side by side.

### Quality control and outlier detection

`FlagQCF` merges multiple test flags into a single quality indicator with daytime/nighttime separation and USTAR scenario support.

Nine outlier detection methods are available: Hampel filter, Z-score (global, rolling, or split by day/night), local SD, Local Outlier Factor, absolute limits, incremental detection, manual removal, trimmed mean, and stepwise chaining across multiple methods. See [examples/preprocessing/outlier_detection/](examples/preprocessing/outlier_detection/).

### Corrections and preprocessing

Tools cover offset correction for measurements, radiation, humidity, and wind direction; threshold and missing value handling; and timestamp sanitization (validation, regularization, frequency detection). See [examples/preprocessing/corrections/](examples/preprocessing/corrections/) and [examples/times/](examples/times/).

### Analysis

Seasonal-trend decomposition (STL, classical, or harmonic), lagged correlation and binned analysis, 2D grid aggregation, gap detection with monthly/annual breakdown, and percentiles/histograms. See [examples/analysis/](examples/analysis/).

### Derived variables

VPD from temperature and humidity, day/night flags from solar geometry, air density, aerodynamic resistance, unit conversions, lagged features, and clear-sky potential radiation. See [examples/features/](examples/features/).

### Eddy covariance

Flux detection limit from 20 Hz data, maximum covariance lag, pre-whitening bootstrap (PWB) for trace gases (CH4, N2O) with single-period and multi-file parallel variants, an end-to-end per-chunk PWB time-lag detect+remove pipeline (`diive-tlag-pwb-detect-remove`, plus a Textual TUI `diive-tlag-pwb-detect-remove-tui`) that splits long raw files into 30-min chunks and writes lag-corrected raw files (line endings preserved), wind double rotation, self-heating correction for open-path IRGAs, USTAR filtering, and random error propagation. See [examples/flux/](examples/flux/).

### Visualization

19+ plot types including time series, cumulative, diel cycle, heatmaps (datetime and year-month), hexbin, histogram, ridgeline, scatter, tree-ring, and anomaly plots. Both Matplotlib and Plotly are supported. See [examples/visualization/](examples/visualization/).

### I/O

Load and save parquet files, read single or batch EddyPro output, detect and split irregular files, and format data for FLUXNET submission. See [examples/io/](examples/io/).

### Desktop GUI (optional)

A PySide6 desktop app (`diive-gui`, install with the `gui` extra) for interactive exploration: an Overview tab (per-variable stats + multi-panel figure); per-method plot tabs (heatmaps, time series, diel cycle, cumulative, ridgeline, scatter, hexbin, histogram — opened from the Plot menu, multiple at once); Analyze tabs (gaps & coverage, driver explorer, seasonal-trend & anomalies, spectrogram); outlier-detection tabs (Hampel, Local SD, three z-score variants) with live previews; a guided flux processing chain; and Data tools to select, rename, and tag variables and engineer features. Every variable carries editable metadata (tags + full provenance history); the whole working state — data, metadata, project settings, sticky notes, and open tabs — saves to a portable `.diive` **project** folder. See the [GUI user manual](diive/gui/MANUAL.md).

---

## Contributing

See [CLAUDE.md](CLAUDE.md) for development setup, coding standards, and testing.

---

## License

`diive` is released under the [GNU General Public License v3.0](LICENSE).
