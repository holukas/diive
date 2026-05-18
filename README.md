![](images/logo_diive1_256px.png)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/diive?style=for-the-badge&color=%23EF6C00&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdiive%2F)](https://pypi.org/project/diive/)
[![GitHub License](https://img.shields.io/github/license/holukas/diive?style=for-the-badge&color=%237CB342)](https://github.com/holukas/diive/blob/indev/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/diive)](https://pepy.tech/projects/diive)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10884017.svg)](https://doi.org/10.5281/zenodo.10884017)

_**`diive` is currently prepared for the v1.0 release.**_

# Time series data processing

`diive` is a Python library for time series processing, in particular ecosystem data. Originally developed
by the [ETH Grassland Sciences group](https://gl.ethz.ch/) for [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/).

**Recent updates:** [CHANGELOG](CHANGELOG.md) • **Development:** [CLAUDE.md](CLAUDE.md) • **Releases:
** [GitHub](https://github.com/holukas/diive/releases)

---

## Getting Started

### Installation

Requires **Python 3.12+**

```bash
pip install diive
```

Or use [uv](https://docs.astral.sh/uv/):

```bash
uv pip install diive
```

### Minimal Example

```python
import diive as dv

# Load example data
df = dv.load_exampledata_parquet()

# Plot time series
dv.plot_time_series(series=df['NEE']).plot()

# Gap-fill with Random Forest
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS

engineer = FeatureEngineer(target_col='NEE', features_lag=[-1, 1], features_rolling=[12, 24])
df_engineered = engineer.fit_transform(df)

model = RandomForestTS(input_df=df_engineered, target_col='NEE', n_estimators=100)
model.trainmodel()
model.fillgaps()
```

### Next Steps

- **[98 Runnable Examples](examples/README.md)** — organized by topic (visualization, gap-filling, flux, etc.)
    - Find examples by use case: [CATALOG.md](examples/CATALOG.md)
    - Dataset documentation: [EXAMPLE_DATASET.md](examples/EXAMPLE_DATASET.md)
- **[Development Setup](CLAUDE.md)** — source setup, coding standards, testing

---

## Quick API Access

Classes are available directly from the `diive` namespace with both PascalCase and snake_case names:

```python
# PascalCase (class name)
from diive.core.plotting import TimeSeries

plot = TimeSeries(series=data)

# snake_case (alias)
import diive as dv

plot = dv.plot_time_series(series=data)
```

**Common exports:**

- **Plotting:** `time_series`, `TimeSeries`, `cumulative`, `Cumulative`, `diel_cycle`, `DielCycle`, `heatmap_datetime`,
  `HeatmapDateTime`
- **Gap-filling:** `randomforest_ts`, `RandomForestTS`, `xgboost_ts`, `XGBoostTS`, `flux_mds`, `FluxMDS`
- **Analysis:** `gridaggregator`, `GridAggregator`, `seasonaltrend`, `SeasonalTrendDecomposition`
- **Eddy Covariance:** `flux_processing_chain`, `FluxProcessingChain`, `flux_detection_limit`, `FluxDetectionLimit`
- **I/O:** `load_parquet`, `save_parquet`, `load_exampledata_parquet`, `search_files`

For the complete list, see `diive.__all__`.

---

## 98 Runnable Examples

Organized by functional domain. All examples follow Sphinx Gallery format (`# %%` sections) — runnable as plain scripts
and auto-converted to HTML docs.

**Quick start:**

```bash
# Run individual examples
uv run python examples/visualization/plot_heatmap_datetime_basic.py
uv run python examples/analysis/analysis_daily_correlation.py
uv run python examples/gapfilling/gapfill_randomforest.py
uv run python examples/flux/fluxprocessingchain/fluxprocessingchain.py
```

**Find your way:**

- **[CATALOG.md](examples/CATALOG.md)** — organized by use case (visualization, analysis, gap-filling, etc.)
- **[EXAMPLE_DATASET.md](examples/EXAMPLE_DATASET.md)** — dataset documentation (37 variables, availability, quality)
- **[examples/README.md](examples/README.md)** — quick start and folder structure

**Example categories:**

- **Visualization** (17 examples) — heatmaps, time series, diel cycles, cumulative plots, histograms, scatter, ridgelines
- **Times** (5 examples) — timestamp validation, frequency detection, diel cycles, temporal matrices
- **Analysis** (11 examples) — correlation, daily correlation, seasonal decomposition, gap detection, gridding, spectral analysis
- **Data Processing** (20 examples) — corrections (7), outlier detection (9), quality flags (2), other (2)
- **Features** (11 examples) — VPD, unit conversions, day/night flags, lagged features, potential radiation
- **Gap-Filling** (11 examples) — linear interpolation, Random Forest, XGBoost, MDS, comparisons, optimization, long-term
- **Flux Processing** (16 examples) — flux chain, low-res analysis, high-res analysis, time lag, wind rotation, USTAR filtering, self-heating
- **Curve Fitting** (2 examples) — polynomial and binned fitting
- **I/O** (5 examples) — file reading, extraction, and data loading

Browse [examples/README.md](examples/README.md) for the full index with descriptions.

---

## Feature Overview

### Gap-Filling

**Feature Engineering Pipeline (v0.91.0)** · `FeatureEngineer`

- 8-stage pipeline: lag features, rolling stats, differencing, EMA, polynomial, STL decomposition,
  timestamps, record numbering
- Pre-engineer once, reuse across multiple gap-filling models
- Full examples: [examples/gapfilling/](examples/gapfilling/)

**Methods:**

- **XGBoostTS** — gradient boosting
- **RandomForestTS** — ensemble learning with SHAP feature importance
- **FluxMDS** — meteorological similarity, no training required
- **Linear interpolation** — for simple gaps only
- **Long-term variants** support multi-year data with USTAR scenario options

**Flux Processing Chain** · `FluxProcessingChain`

- Post-processing pipeline covering quality flags, storage correction, outlier detection, USTAR filtering, and
  gap-filling
- Implements Levels 2–4.1 following Swiss FluxNet standards
- Example: [examples/flux/fluxprocessingchain/](examples/flux/fluxprocessingchain/)

Reference: [Swiss FluxNet flux processing](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/)

### Quality Control & Outlier Detection

**Overall Quality Flag (QCF)** · `FlagQCF`

- Merges multiple test flags into a single quality indicator
- Daytime/nighttime separation and USTAR scenario support
- Example: [examples/preprocessing/qaqc/qc_overall_flag.py](examples/preprocessing/qaqc/qc_overall_flag.py)

**10 Outlier Detection Methods:**

- **Hampel filter** — robust spike detection using MAD (median absolute deviation)
- **Z-score** — global, rolling, or day/night variants
- **Local SD** — adaptive local thresholds
- **Local Outlier Factor (LOF)** — density-based anomaly detection
- **Absolute limits** — physical bounds on values
- **Incremental detection** — find abrupt changes between records
- **Manual removal** — explicit period or point flagging
- **Trimmed mean** — symmetric removal of high and low outliers
- **Stepwise orchestration** — chain multiple methods together
- Examples: [examples/preprocessing/outlier_detection/](examples/preprocessing/outlier_detection/)

### Data Processing & Corrections

- **Offset correction** — adjust measurement, radiation, humidity, and wind direction biases
- **Set to threshold/missing** — apply thresholds or manual value replacements
- **Timestamp sanitization** — validate, regularize, and detect frequency
- Examples: [examples/preprocessing/corrections/](examples/preprocessing/corrections/),
  [examples/times/](examples/times/)

### Analysis

- **Seasonal-Trend Decomposition** — STL, classical, or harmonic methods
- **Correlation & decoupling** — lagged relationships and binned analysis
- **Grid aggregation** — 2D binning and statistics
- **Gap finder** — identify missing data patterns
- **Percentiles & histogram** — distribution analysis
- Examples: [examples/analysis/](examples/analysis/)

### Feature Engineering

- **Vapor Pressure Deficit (VPD)** — calculate from temperature and humidity
- **Day/night flags** — solar geometry classification
- **Air properties** — density, resistance, heat capacity
- **Unit conversions** — temperature, energy, and water
- **Lagged features** — time-shifted variables
- **Potential radiation** — clear-sky calculation
- Examples: [examples/features/](examples/features/)

### Eddy Covariance & Flux

- **Flux detection limit** — signal-to-noise analysis from high-frequency (20 Hz) data
- **Maximum covariance** — find optimal time lag
- **Wind rotation** — coordinate transformation, turbulent departures
- **Self-heating correction** — open-path IRGA oxygen flux adjustment
- **USTAR filtering** — threshold detection and filtering
- **Uncertainty estimation** — random error propagation
- Examples: [examples/flux/](examples/flux/)

### Visualization

- **14+ plot types** — time series, cumulative, diel cycle, heatmap (datetime/year-month), hexbin, histogram,
  ridge line, scatter, anomalies
- **Interactive plots** — Matplotlib and Plotly support
- Examples: [examples/visualization/](examples/visualization/)

### I/O & File Handling

- **Load/save parquet** — efficient columnar format for time series
- **Read EddyPro files** — single or batch file reading
- **Detect/split files** — identify irregular files, split large datasets
- **Format for FLUXNET** — prepare data for upload

---

## Contributing

See [CLAUDE.md](CLAUDE.md) for development setup, coding standards, and testing.

---

## Citation

If you use `diive` in your research, please cite it:

```bibtex
@software{diive2024,
  title={diive},
  author={Hörtnagl, Lukas},
  orcid={https://orcid.org/0000-0002-5569-0761},
  url={https://github.com/holukas/diive},
  doi={10.5281/zenodo.10884017},
  year={2024}
}
```

---

## License

`diive` is licensed under the [GNU General Public License v3.0](LICENSE).
