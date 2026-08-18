![](images/logo_diive1_256px.png)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/diive?style=for-the-badge&color=%23EF6C00&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdiive%2F)](https://pypi.org/project/diive/)
[![GitHub License](https://img.shields.io/github/license/holukas/diive?style=for-the-badge&color=%237CB342)](https://github.com/holukas/diive/blob/indev/LICENSE)
[![Documentation](https://img.shields.io/readthedocs/diive?style=for-the-badge&color=%230066cc)](https://diive.readthedocs.io/)
[![PyPI Downloads](https://static.pepy.tech/badge/diive)](https://pepy.tech/projects/diive)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10884017.svg)](https://doi.org/10.5281/zenodo.10884017)

_**`diive` is currently being prepared for the v1.0 release.**_

# Time series data processing

`diive` is a Python library for time series processing, focused on ecosystem data. It was originally developed by the
[ETH Grassland Sciences group](https://gl.ethz.ch/) for [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/): eddy
covariance flux processing, gap-filling, quality control, and the plots that go with them.

There are two ways to use it, and you can pick whichever fits:

- a **library** — `import diive as dv`, ten domain namespaces
- a **desktop GUI** — `diive-gui`, for interactive work without writing code

`diive` works on averaged (e.g. 30-minute) data. For raw high-frequency (10/20 Hz) eddy covariance data — wind
rotation, flux detection limit, time-lag detection and removal — see [dyco](https://github.com/holukas/dyco).

[Documentation](https://diive.readthedocs.io/) | [Project overview](OVERVIEW.md) | [Examples](examples/README.md) |
[GUI manual](diive/gui/MANUAL.md) | [CHANGELOG](CHANGELOG.md) | [Releases](https://github.com/holukas/diive/releases)

---

## Install

Requires **Python 3.12 or 3.13**.

```bash
pip install diive                 # core library
pip install 'diive[gui]'          # + desktop GUI, then launch with: diive-gui
pip install 'diive[gui,gui3d]'    # + 3-D surface views (PyVista/VTK)
pip install 'diive[db]'           # + InfluxDB read/write
```

### From a clone, with `uv`

```bash
uv sync                              # core library + the 'dev' group (synced by default)
uv sync --all-extras --all-groups    # everything: all extras AND all groups
```

**`uv sync --all-extras` alone is not everything.** The optional pieces are split across two uv
mechanisms, and `--all-extras` reaches only the first:

| Kind | Name | Pulls in | Install |
|---|---|---|---|
| extra | `gui` | PySide6 desktop GUI (`diive-gui`) | `uv sync --extra gui` |
| extra | `gui3d` | PyVista/VTK 3-D surface tabs, `trimesh` glTF export | `uv sync --extra gui3d` |
| extra + group | `db` | `influxdb-client`, the InfluxDB read/write engine | `uv sync --group db` |
| group | `dev` | test, lint and notebook tooling | synced by default |
| group | `build` | PyInstaller, for the standalone Windows app | `uv sync --group build` |

`db` is deliberately both. Working *on* diive, use the group — `uv sync --group db`. Depending on
diive from another project, ask for the extra — `diive[db]` — because a dependency group is local to
the project that declares it and never reaches the published metadata.

Combine as needed, e.g. `uv sync --extra gui --extra gui3d --group db`. Then run anything through
`uv run`:

```bash
uv run pytest tests/ -v
uv run diive-gui
```

[CONTRIBUTING.md](CONTRIBUTING.md) has the rest of the development setup.

## Quick start

```python
import diive as dv

df = dv.load_exampledata_parquet()   # bundled multi-year 30-min eddy covariance record

dv.plotting.TimeSeries(series=df['NEE_CUT_REF_f']).plot()
dv.plotting.HeatmapDateTime(series=df['NEE_CUT_REF_f']).plot()
```

Plots follow a two-phase pattern throughout: the constructor takes the data, `.plot()` takes the styling.

From here, the [cookbook](examples/COOKBOOK.md) walks through six minimal workflows — load data, clean timestamps, remove
outliers, gap-fill, run the flux chain, visualize.

## What's in it

`import diive as dv` exposes ten domain namespaces. Each row links to runnable examples for that area:

| Namespace | Covers | Examples |
|---|---|---|
| `dv.plotting` | 18 plot types: time series, heatmaps, diel cycle, cumulative, ridgeline, scatter, hexbin, wind rose, tree ring, 3-D surface, ... | [visualization/](examples/visualization/README.md) |
| `dv.gapfilling` | `RandomForestTS`, `XGBoostTS`, `SWINGapFillerXGBoost`, `FluxMDS`, linear interpolation, long-term variants, `FeatureEngineer` | [gapfilling/](examples/gapfilling/README.md) |
| `dv.flux` | Flux processing chain (L2–L4.2), NEE partitioning, USTAR filtering, uncertainty | [flux/](examples/flux/README.md) |
| `dv.outliers` | Nine detection methods (Hampel, z-score variants, local SD, LOF, absolute limits, ...) | [outlier_detection/](examples/preprocessing/outlier_detection/README.md) |
| `dv.corrections` | Offset corrections (measurement, radiation, humidity, wind direction), thresholds, missing values | [corrections/](examples/preprocessing/corrections/README.md) |
| `dv.qaqc` | `FlagQCF` quality flags, EddyPro flag handling, meteo screening | [qaqc/](examples/preprocessing/qaqc/README.md) |
| `dv.analysis` | Seasonal-trend decomposition, lagged correlation, grid aggregation, gap statistics, spectral analysis | [analysis/](examples/analysis/README.md) |
| `dv.times` | Timestamp sanitization, frequency detection, resampling, date-range handling | [times/](examples/times/README.md) |
| `dv.variables` | Derived variables (VPD, potential radiation, day/night flags, air properties), feature engineering | [features/](examples/features/README.md) |
| `dv.events` | Time-stamped event markers, 0/1 flag columns, plot overlays | [events/](examples/events/README.md) |

I/O helpers are top-level (`dv.load_parquet`, `dv.save_parquet`, `dv.ReadFileType`) — see [io/](examples/io/README.md).
For the authoritative symbol list, check `diive.__all__` and each namespace's `__all__`.

## Highlights

**Flux processing chain** — post-processing from quality flags through gap-filling and NEE partitioning (Levels 2 to
4.2), following [Swiss FluxNet standards](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/).
Either `run_chain(data, config)` for the standard workflow, or composable per-level callables when you need every
detector, hyperparameter and diagnostic flag. → [examples/flux/fluxprocessingchain/](examples/flux/fluxprocessingchain/)

**NEE partitioning** — four faithful ports of the reference routines, each validated against its original
implementation: nighttime and daytime (Reichstein 2005, Lasslop 2010) × ONEFlux and REddyProc. Output columns are tagged
so all four coexist in one dataframe. → [examples/flux/partitioning/](examples/flux/partitioning/)

**Gap-filling** — Random Forest and XGBoost with SHAP-based feature reduction, plus a faithful MDS port that needs no
training. An 8-stage feature engineer feeds them all. → [examples/gapfilling/](examples/gapfilling/README.md)

**Desktop GUI** — the same library code behind an interactive app: plotting, cleaning, gap-filling and flux tabs, a
guided processing chain, per-variable metadata with full provenance, and portable `.diive` project folders.
→ [GUI manual](diive/gui/MANUAL.md)

## Documentation

| Where | What |
|---|---|
| [diive.readthedocs.io](https://diive.readthedocs.io/) | Hosted docs: API reference and example gallery |
| [OVERVIEW.md](OVERVIEW.md) | How the pieces fit together: library, GUI, docs, packaging |
| [examples/COOKBOOK.md](examples/COOKBOOK.md) | Six minimal workflows — the place to start |
| [examples/CATALOG.md](examples/CATALOG.md) | All 113 examples, indexed by use case |
| [examples/EXAMPLE_DATASET.md](examples/EXAMPLE_DATASET.md) | The bundled 37-variable dataset |
| [diive/gui/MANUAL.md](diive/gui/MANUAL.md) | Desktop GUI user manual |
| [diive/gui/README.md](diive/gui/README.md) | GUI architecture, for developers |
| [notebooks/README.md](notebooks/README.md) | Jupyter workflows, including the InfluxDB database |
| [packaging/README.md](packaging/README.md) | Building the standalone Windows app |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, coding standards, testing |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

Examples run as plain scripts:

```bash
uv run python examples/visualization/plot_heatmap_datetime_basic.py
uv run python examples/gapfilling/gapfill_randomforest.py
uv run python examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py
```

---

## Citation

Cite `diive` using DOI [10.5281/zenodo.10884017](https://doi.org/10.5281/zenodo.10884017). This concept DOI resolves to
the latest release, so include the version number in your citation.

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

## License

`diive` is released under the [GNU General Public License v3.0](LICENSE).
</content>
</invoke>
