![](images/logo_diive1_256px.png)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/diive?style=for-the-badge&color=%23EF6C00&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdiive%2F)](https://pypi.org/project/diive/)
[![GitHub License](https://img.shields.io/github/license/holukas/diive?style=for-the-badge&color=%237CB342)](https://github.com/holukas/diive/blob/main/LICENSE)
[![Documentation](https://img.shields.io/readthedocs/diive?style=for-the-badge&color=%230066cc)](https://diive.readthedocs.io/)
[![PyPI Downloads](https://static.pepy.tech/badge/diive)](https://pepy.tech/projects/diive)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10884017.svg)](https://doi.org/10.5281/zenodo.10884017)

_**`diive` is being prepared for the v1.0 release.**_

# Time series processing for ecosystem data

`diive` is a Python library for processing time series of ecosystem measurements. The
[ETH Grassland Sciences group](https://gl.ethz.ch/) developed it for [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/),
mainly to post-process eddy covariance fluxes: quality control, gap-filling, flux partitioning and plots.

`diive` can be used in two ways:

- **As a library** in your own Python code: `import diive as dv`.
- **As a desktop app** (`diive-gui`) for interactive work without writing code.

`diive` works with averaged data, for example 30-minute values. For raw high-frequency (10 or 20 Hz) eddy covariance
data, such as wind rotation, flux detection limits and time lags, use [dyco](https://github.com/holukas/dyco).

[Documentation](https://diive.readthedocs.io/) | [Project overview](OVERVIEW.md) | [Examples](examples/README.md) |
[GUI manual](diive/gui/MANUAL.md) | [CHANGELOG](CHANGELOG.md) | [Releases](https://github.com/holukas/diive/releases)

---

## Installation

`diive` needs **Python 3.12 or 3.13**.

### Required tools

Each installation option uses only some of these tools, as listed in the last column.

| Tool                                    | What it does                                                                                     | Needed for                                     |
|-----------------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------------|
| [conda](https://conda-forge.org/download/) | Creates separate Python environments, each with its own Python version and packages.         | Option A (recommended for new Python users) |
| [pip](https://pip.pypa.io/)             | Installs Python packages, such as `diive`, from [PyPI](https://pypi.org/project/diive/). It comes with Python. | Options A, B and C                              |
| [git](https://git-scm.com/downloads)    | Downloads the source code from GitHub and keeps it up to date.                                   | Options C and D                                 |
| [uv](https://docs.astral.sh/uv/)        | Installs Python, creates the environment and installs the packages, all in one tool.             | Option D (development)          |

An **environment** is a folder with one Python installation and the packages you add to it. Installing `diive` into its
own environment keeps it from clashing with other Python projects on your computer.

### Basic installation and optional components

`pip install diive` installs the library and everything it needs for processing and plotting: pandas, NumPy,
scikit-learn, XGBoost, SHAP, statsmodels, matplotlib, seaborn, Bokeh, PyArrow and a few smaller packages. This is sufficient
for using `diive` in scripts.

It does **not** install the desktop app, the 3-D views, the database connection or Jupyter. These are optional components:

| Purpose                                             | Install                              | Adds                         |
|-----------------------------------------------------|--------------------------------------|------------------------------|
| Use `diive` in Python scripts                       | `diive`                              | The library (basic install)  |
| Use the desktop app                                 | `"diive[gui]"`                       | PySide6 (Qt)                 |
| See 3-D surface plots in the app                    | `"diive[gui,gui3d]"`                 | PyVista and VTK (large)      |
| Connect to an InfluxDB v2 database (see note below)  | `"diive[db]"`, with the app `"diive[gui,db]"` | `influxdb-client`            |
| Everything above                                    | `"diive[gui,gui3d,db]"`              |                              |
| Run Jupyter notebooks, e.g. those in `notebooks/`   | `jupyterlab` (a separate package)    | JupyterLab                   |

Put package names with brackets in double quotes, as shown, so that the command works in every shell (Command
Prompt, PowerShell, bash, zsh).

The InfluxDB connection is a specialized feature for groups that keep their measurements in an InfluxDB v2
database. Most users do not need it. It is used by the library (`InfluxIO`), the app's **File ▸ Database** tabs,
the database source of the app's meteo screening tab and the database notebooks in `notebooks/`. Without it, these
parts of the app show a short notice instead.

### Option A: conda (recommended for new Python users)

1. **Install conda.** Download and install [Miniforge](https://conda-forge.org/download/), a free conda distribution.
   On Windows, keep the default options. On macOS and Linux, answer **yes** when the installer asks whether to
   initialize conda.
2. **Open a terminal with conda.** On Windows, open **Miniforge Prompt** from the Start menu. On macOS and Linux, open
   a new terminal window.
3. **Create an environment** named `diive` with Python 3.12, and switch to it:

   ```bash
   conda create -n diive python=3.12
   conda activate diive
   ```

   The prompt now starts with `(diive)`. Every following command runs inside this environment.
4. **Install `diive` with pip.** `diive` is published on PyPI, so it is installed with pip, not with conda. For the
   library and the desktop app:

   ```bash
   python -m pip install "diive[gui]"
   ```

   For the library only, use `python -m pip install diive`. Writing `python -m pip` makes sure the package goes into
   the active environment.
5. **Start the desktop app:**

   ```bash
   diive-gui
   ```

   The first time, the app opens with the bundled example dataset. Later it reopens the project you used last.

**Starting the app later.** Open the terminal (step 2) and run:

```bash
conda activate diive
diive-gui
```

**Using the library.** The library is available in the same environment, in scripts, in Python or in Jupyter:

```bash
python -m pip install jupyterlab   # once, only if you want notebooks
jupyter lab
```

**Updating to the latest release.**

```bash
conda activate diive
python -m pip install --upgrade "diive[gui]"
```

**Removing the environment.** This deletes the environment and all its packages:

```bash
conda deactivate
conda env remove -n diive
```

In this environment, install further packages with pip as well. Installing the same packages with conda afterwards
can replace the versions `diive` needs.

### Option B: pip in an existing environment

If you already have Python 3.12 or 3.13 and an environment you want to use (for example one made with `venv`), activate
it and run:

```bash
python -m pip install "diive[gui]"   # or just: diive
diive-gui                            # starts the desktop app
```

### Option C: development version from GitHub

PyPI has the latest release. New work reaches GitHub earlier: it lands in the `indev` branch first and moves to `main`
with each merge. To install from GitHub, you need [git](https://git-scm.com/downloads).

To install directly from GitHub, run this inside your environment (for example the conda environment from Option A):

```bash
python -m pip install "diive[gui] @ git+https://github.com/holukas/diive.git"         # main branch
python -m pip install "diive[gui] @ git+https://github.com/holukas/diive.git@indev"   # indev branch
```

Alternatively, clone the repository with git and install it from the local folder. With `-e` (editable), changes to the code, or new code
fetched with `git pull`, take effect without installing again:

```bash
git clone https://github.com/holukas/diive.git
cd diive
python -m pip install -e ".[gui]"
```

To get the newest code later, run `git pull` in the `diive` folder. Switch to the development branch with
`git checkout indev`.

### Option D: development setup with uv

uv installs a matching Python version itself and keeps the environment in the folder `.venv`. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and [git](https://git-scm.com/downloads), then:

```bash
git clone https://github.com/holukas/diive.git
cd diive
uv sync --extra gui    # library, desktop app and the development tools
uv run diive-gui       # starts the desktop app
```

`uv run` runs a command inside the project environment, for example `uv run pytest tests/` or
`uv run python examples/gapfilling/gapfill_randomforest.py`. Plain `uv sync` installs the library and the development
tools (tests, linting, docs, Jupyter). Optional components:

| Part    | Adds                                               | Command                 |
|---------|----------------------------------------------------|-------------------------|
| `gui`   | The desktop app (PySide6)                          | `uv sync --extra gui`   |
| `gui3d` | 3-D surface plots in the app (PyVista, VTK)        | `uv sync --extra gui3d` |
| `db`    | InfluxDB v2 access (`influxdb-client`)             | `uv sync --group db`    |
| `dev`   | Tests, linting, docs and Jupyter                   | included in `uv sync`   |
| `build` | PyInstaller, to build the standalone Windows app   | `uv sync --group build` |

Combine them as needed, e.g. `uv sync --extra gui --extra gui3d --group db`. `uv sync --all-extras --all-groups`
installs everything. `--all-extras` alone leaves out `db`, `dev` and `build`, because uv counts those as groups.

`db` is both an extra and a group on purpose. When you work on `diive` itself, use the group (`uv sync --group db`).
A project that depends on `diive` asks for the extra (`diive[db]`), because a group never reaches the published
package. [CONTRIBUTING.md](CONTRIBUTING.md) has the rest of the development setup.

### Standalone Windows app

The desktop app can also be built as a Windows program that runs without Python. In PowerShell, after
`uv sync --extra gui --group build`, run `.\packaging\build_gui.ps1 -Clean`. Details:
[packaging/README.md](packaging/README.md).

### Verifying the installation

```bash
python -c "import diive; print(diive.__version__)"
```

This prints the installed version. If it fails, or `diive-gui` is not found:

- **The environment is not active.** Run `conda activate diive` (Option A) or activate your own environment. With uv,
  put `uv run` in front of the command.
- **`No module named 'PySide6'` when starting the app.** The desktop app was not installed. Install `"diive[gui]"`.
- **`diive-gui` is still not found.** Start the app with `python -m diive.gui._cli` instead.

## Quick start

```python
import diive as dv

df = dv.load_exampledata_parquet()  # bundled multi-year record of 30-minute eddy covariance data

dv.plotting.TimeSeries(series=df['NEE_CUT_REF_f']).plot()
dv.plotting.HeatmapDateTime(series=df['NEE_CUT_REF_f']).plot()
```

Every plot works in two steps: the class takes the data, and `.plot()` takes the styling.

The [cookbook](examples/COOKBOOK.md) continues with six short workflows: load data, clean timestamps, remove outliers,
gap-fill, run the flux post-processing chain and make plots.

## Library overview

`import diive as dv` gives access to ten namespaces, one per topic. Each row links to runnable examples:

| Namespace        | Contents                                                                                                                                | Examples                                                                  |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `dv.plotting`    | 18 plot types: time series, heatmaps, diel cycle, cumulative, ridgeline, scatter, hexbin, wind rose, tree ring, 3-D surface and more    | [visualization/](examples/visualization/README.rst)                       |
| `dv.gapfilling`  | `RandomForestTS`, `XGBoostTS`, `SWINGapFillerXGBoost`, `FluxMDS`, linear interpolation, long-term variants, `FeatureEngineer`           | [gapfilling/](examples/gapfilling/README.rst)                             |
| `dv.flux`        | Flux post-processing chain (Levels 2 to 4.2), NEE partitioning, u* filtering, uncertainty                                               | [flux/](examples/flux/README.rst)                                         |
| `dv.outliers`    | Nine detection methods, including Hampel, z-score variants, local SD, local outlier factor and absolute limits                          | [outlier_detection/](examples/preprocessing/outlier_detection/README.rst) |
| `dv.corrections` | Offset corrections (measurement, radiation, relative humidity, wind direction), thresholds, missing values                             | [corrections/](examples/preprocessing/corrections/README.rst)             |
| `dv.qaqc`        | `FlagQCF` overall quality flag, EddyPro flags, meteo screening                                                                          | [qaqc/](examples/preprocessing/qaqc/README.rst)                           |
| `dv.analysis`    | Seasonal-trend decomposition, lagged correlation, grid aggregation, gap statistics, spectral analysis                                   | [analysis/](examples/analysis/README.rst)                                 |
| `dv.times`       | Timestamp checks and fixes, time resolution detection, resampling, date ranges                                                          | [times/](examples/times/README.rst)                                       |
| `dv.variables`   | Derived variables (VPD, potential radiation, day/night flags, air properties), feature engineering                                      | [features/](examples/features/README.rst)                                 |
| `dv.events`      | Event markers with dates, 0/1 flag columns, plot overlays                                                                               | [events/](examples/events/README.rst)                                     |

Reading and writing files is available directly on `dv` (`dv.load_parquet`, `dv.save_parquet`, `dv.ReadFileType`), see
[io/](examples/io/README.rst). The full list of names is in `diive.__all__` and in the `__all__` of each namespace.

## Main features

**Flux post-processing chain.** Takes fluxes from quality flags through gap-filling to NEE partitioning (Levels 2 to
4.2), following the [Swiss FluxNet processing chain](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/).
`run_chain(data, config)` runs the standard workflow. The levels can also be called one by one, with full control over
each test, model setting and flag. Examples: [examples/flux/postprocessingchain/](examples/flux/postprocessingchain/)

**NEE partitioning.** Four methods, each ported from its reference code and checked against it: nighttime
(Reichstein et al. 2005) and daytime (Lasslop et al. 2010), each as in ONEFlux and as in REddyProc. `PartitioningNEE`
picks one with `method='nighttime'|'daytime'` and `mode='oneflux'|'reddyproc'`. The default is ONEFlux, the method used
by ICOS. Each method writes columns with its own suffix, so all four can sit in one table. Examples:
[examples/flux/partitioning/](examples/flux/partitioning/)

**Gap-filling.** Random Forest and XGBoost models that use SHAP values to drop unimportant features, and an MDS
gap-filler that needs no training. By default, MDS follows the ONEFlux rules and adds the symmetric mean of Vekuri et
al. (2023). Two more modes reproduce the MDS of ONEFlux and of REddyProc. A feature engineer with 8 stages builds the inputs
for the machine-learning models. Examples: [examples/gapfilling/](examples/gapfilling/README.rst)

**Desktop app.** Provides the library functions in an interactive interface: plotting, data cleaning, gap-filling and flux tabs,
a guided post-processing chain, metadata for each variable with its processing history, and project folders
(`.diive`) that can be moved to another computer. Manual: [GUI manual](diive/gui/MANUAL.md)

![The diive desktop app showing the Overview tab: variable list, summary statistics, and linked time series, heatmap, cumulative, diel cycle and distribution panels](images/diive-gui_v0.91.0_20260820.png)

*The Overview tab in diive-gui v0.91.0 with ten years of half-hourly data. Selecting a variable on the left redraws all
panels. Zooming into one of the date panels updates the statistics and panels for the visible period.*

## Documentation

| Where                                                      | What                                                         |
|------------------------------------------------------------|--------------------------------------------------------------|
| [diive.readthedocs.io](https://diive.readthedocs.io/)      | Online documentation: API reference and example gallery      |
| [OVERVIEW.md](OVERVIEW.md)                                 | How library, desktop app, docs and packaging fit together    |
| [examples/COOKBOOK.md](examples/COOKBOOK.md)               | Six short workflows, the recommended starting point         |
| [examples/CATALOG.md](examples/CATALOG.md)                 | All 113 examples, sorted by task                             |
| [examples/EXAMPLE_DATASET.md](examples/EXAMPLE_DATASET.md) | The bundled example dataset (37 variables)                   |
| [diive/gui/MANUAL.md](diive/gui/MANUAL.md)                 | User manual of the desktop app                               |
| [diive/gui/README.md](diive/gui/README.md)                 | How the desktop app is built, for developers                 |
| [notebooks/README.md](notebooks/README.md)                 | Jupyter notebooks, including the InfluxDB database workflows |
| [packaging/README.md](packaging/README.md)                 | Building the standalone Windows app                          |
| [CONTRIBUTING.md](CONTRIBUTING.md)                         | Development setup, coding standards, testing                 |
| [CHANGELOG.md](CHANGELOG.md)                               | Version history                                              |

The examples are plain scripts in the repository, so you need a copy of the code (option C or D). Run one with
`python` in your environment, or with `uv run python` in a uv setup:

```bash
python examples/visualization/plot_heatmap_datetime_basic.py
uv run python examples/gapfilling/gapfill_randomforest.py
```

---

## Citation

Cite `diive` with the DOI [10.5281/zenodo.10884017](https://doi.org/10.5281/zenodo.10884017). This DOI always points to
the latest release, so add the version number you used:

```bibtex
@software{diive2026,
  author = {Hörtnagl, Lukas},
  title = {diive: Python library for time series processing},
  version = {0.92.0},
  year = {2026},
  doi = {10.5281/zenodo.10884017}
}
```

Replace `version` and `year` with the release you used.

## License

`diive` is released under the [GNU General Public License v3.0](LICENSE).
