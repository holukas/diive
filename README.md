![](images/logo_diive1_256px.png)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/diive?style=for-the-badge&color=%23EF6C00&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdiive%2F)](https://pypi.org/project/diive/)
[![GitHub License](https://img.shields.io/github/license/holukas/diive?style=for-the-badge&color=%237CB342)](https://github.com/holukas/diive/blob/indev/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/diive)](https://pepy.tech/projects/diive)
[![DOI](https://zenodo.org/badge/708559210.svg)](https://zenodo.org/doi/10.5281/zenodo.10884017)

*`diive` is under active development with frequent updates and improvements.*

**Update 11 May 2026**  
diive is currently prepared for the first main release. Code examples have been expanded to 62+ examples across 18 organized
folders with comprehensive documentation (CATALOG.md, EXAMPLE_DATASET.md). Documentation will be created and hosted on ReadTheDocs after the initial release. New features are still added regularly, and existing functionality is continuously improved.

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

- **📖 [62+ Executable Examples](examples/README.md)** — Learn by doing
    - Find examples by use case: [CATALOG.md](examples/CATALOG.md)
    - Dataset documentation: [EXAMPLE_DATASET.md](examples/EXAMPLE_DATASET.md)
- **💻 [Development Setup](CLAUDE.md)** — Contribute or work with the source

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

## 62+ Executable Examples

Organized to mirror the codebase structure. All examples follow Sphinx Gallery format (executable Python scripts with # %% sections) for version control friendliness and auto-documentation generation.

**Quick start:**

```bash
# Run individual examples
uv run python examples/core/visualization/heatmap_datetime.py
uv run python examples/pkgs/analysis/analysis_correlation.py
uv run python examples/pkgs/gapfilling/gapfill_randomforest.py
uv run python examples/pkgs/flux/fluxprocessingchain/fluxprocessingchain.py
```

**Find your way:**

- **[CATALOG.md](examples/CATALOG.md)** — Organized by use case with workflows (visualization, analysis, gap-filling, etc.)
- **[EXAMPLE_DATASET.md](examples/EXAMPLE_DATASET.md)** — Complete dataset documentation (37 variables, availability, quality)
- **[examples/README.md](examples/README.md)** — Quick start and folder structure

**Example categories:**

- **Visualization** (9 examples) — Heatmaps, time series, diel cycles, cumulative plots, histograms
- **Analysis** (9 examples) — Correlation, seasonal decomposition, gap detection, gridding, spectral analysis
- **Data Processing** (19 examples) — Corrections (8), outlier detection (9), quality flags (2)
- **Feature Engineering** (8 examples) — VPD, unit conversions, day/night flags, lagged features, potential radiation
- **Gap-Filling** (10 examples) — Linear interpolation, Random Forest, XGBoost, MDS, comparisons, optimization
- **Flux Processing** (10 examples) — Time lag, wind rotation, USTAR filtering, uncertainty, self-heating, fluxchain
- **Curve Fitting** (1 example) — Polynomial fitting
- **I/O & Utilities** (1 example) — Binary value extraction

Browse [examples/README.md](examples/README.md) for the full index with descriptions.

---

## Feature Overview

### Gap-Filling

**Feature Engineering Pipeline (v0.91.0)** · `FeatureEngineer`

- 8-stage composable pipeline: lag features, rolling stats, differencing, EMA, polynomial, STL decomposition,
  timestamps, record numbering
- Pre-engineer once, reuse across multiple gap-filling models
- Full examples: [examples/pkgs/gapfilling/](examples/gapfilling/)

**Methods:**

- **XGBoostTS**: Gradient boosting for speed and high accuracy
- **RandomForestTS** — Ensemble learning; interpretable and robust
- **FluxMDS** (meteorological similarity, no training required)
- **Linear interpolation**: For simple gaps only
- **Long-term variants** support multi-year data with USTAR scenario options

**Flux Processing Chain** · `FluxProcessingChain`

- Post-processing pipeline covering quality flags, storage correction, outlier detection, USTAR filtering, and
  gap-filling
- Implements Levels 2–4.1 following Swiss FluxNet standards
- Example: [examples/pkgs/flux/fluxprocessingchain/](examples/flux/fluxprocessingchain/)
-
Reference: [Swiss FluxNet flux processing](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/)

### Quality Control & Outlier Detection

**Overall Quality Flag (QCF)** · `FlagQCF`

- Merges multiple test flags into a single quality indicator
- Daytime/nighttime separation and USTAR scenario support
- Example: [examples/pkgs/preprocessing/qaqc/qcf.py](examples/preprocessing/qaqc/qcf.py)

**10 Outlier Detection Methods:**

- **Hampel filter**: Robust spike detection using MAD (median absolute deviation)
- **Z-score** — Global, rolling, or day/night variants
- **Local SD** (adaptive local thresholds)
- **Local Outlier Factor (LOF)**: Density-based anomaly detection
- **Absolute limits** — Physical bounds on values
- **Incremental detection**: Find abrupt changes between records
- **Manual removal** — Explicit period or point flagging
- **Trimmed mean**: Symmetric removal of high and low outliers
- **Stepwise orchestration** — Chain multiple methods together
- Examples: [examples/pkgs/preprocessing/outlierdetection/](examples/preprocessing/outlierdetection/)

### Data Processing & Corrections

- **Offset correction**: Adjust measurement, radiation, humidity, and wind direction biases
- **Set to threshold/missing** — Apply thresholds or manual value replacements
- **Timestamp sanitization**: Validate, regularize, and detect frequency
- **Examples:
  ** [examples/pkgs/preprocessing/corrections/](examples/preprocessing/corrections/), [examples/core/times/](examples/times/)

### Analysis

- **Seasonal-Trend Decomposition**: STL, classical, or harmonic methods
- **Correlation & decoupling** — Lagged relationships and binned analysis
- **Grid aggregation** (2D binning and statistics)
- **Gap finder**: Identify missing data patterns
- **Percentiles & histogram** — Distribution analysis
- **Examples:** [examples/pkgs/analysis/](examples/analysis/)

### Feature Engineering

- **Vapor Pressure Deficit (VPD)**: Calculate from temperature and humidity
- **Day/night flags** — Solar geometry classification
- **Air properties** (density, resistance, heat capacity)
- **Unit conversions**: Temperature, energy, and water
- **Lagged features** — Time-shifted variables
- **Potential radiation**: Clear-sky calculation
- **Examples:** [examples/pkgs/features/](examples/features/)

### Eddy Covariance & Flux

- **Flux detection limit**: Signal-to-noise analysis from high-frequency (20 Hz) data
- **Maximum covariance** — Find optimal time lag
- **Wind rotation** (coordinate transformation, turbulent departures)
- **Self-heating correction**: Open-path IRGA oxygen flux adjustment
- **USTAR filtering** — Threshold detection and filtering
- **Uncertainty estimation**: Random error propagation
- **Examples:** [examples/pkgs/flux/](examples/flux/)

### Visualization

- **14+ plot types**: Time series, cumulative, diel cycle, heatmap (datetime/year-month), hexbin, histogram, ridge line,
  scatter, anomalies
- **Interactive plots** — Matplotlib and Plotly support
- **Examples:** [examples/core/visualization/](examples/visualization/)

### I/O & File Handling

- **Load/save parquet**: Efficient columnar format for time series
- **Read EddyPro files** — Single or batch file reading
- **Detect/split files** (identify irregular files, split large datasets)
- **Format for FLUXNET**: Prepare data for upload

---

## Installation

### Using pip (Recommended)

```bash
pip install diive
```

### Using uv

[uv](https://docs.astral.sh/uv/) is a modern Python package installer:

```bash
uv pip install diive
```

### Using poetry

```bash
poetry add diive
```

### From source (Development)

```bash
git clone https://github.com/holukas/diive.git
cd diive
uv sync                       # Install dependencies
uv run pytest tests/          # Run tests
```

See [CLAUDE.md](CLAUDE.md) for detailed development setup.

### Legacy: Using conda

```bash
conda create -n diive python=3.12
conda activate diive
pip install diive
```

---

## Advanced: Comprehensive Jupyter Notebooks

Detailed Jupyter notebooks with full example workflows are available at [notebooks/](notebooks/):

- **[Notebook Overview](https://github.com/holukas/diive/blob/main/notebooks/OVERVIEW.ipynb)** — Complete listing of all
  notebooks
- **Gap-Filling:** XGBoost, Random Forest, MDS, linear interpolation, quick methods
- **Flux Processing:** Complete chain, quick chain, self-heating correction, detection limit
- **Quality Control:** Outlier detection, z-score variants, Hampel, LOF, quality flags
- **Analysis:** Correlation, seasonality, decomposition, binned analysis
- **I/O:** Reading EddyPro files, parquet, formatting for FLUXNET
- **Visualization:** All plot types and customization options

Use notebooks as detailed references; see [examples/](examples/) for quick, runnable demonstrations.

---

## Contributing

Contributions are welcome! See [CLAUDE.md](CLAUDE.md) for development guidelines, testing, and coding standards.

Key points:

- **Setup:** `uv sync` to install dependencies
- **Tests:** `uv run pytest tests/ -v`
- **Examples:** Always include runnable examples, update [CATALOG.md](examples/CATALOG.md) and category READMEs when
  adding new ones
- **Changelog:** Document changes in [CHANGELOG.md](CHANGELOG.md)

---

## Citation

If you use `diive` in your research, please cite it:

```bibtex
@software{diive2024,
  title={DIIVE: Data Integration and Interactive Visualization Engine},
  author={Holukas, Dominik},
  url={https://github.com/holukas/diive},
  doi={10.5281/zenodo.10884017},
  year={2024}
}
```

---

## License

`diive` is licensed under the [MIT License](LICENSE).
