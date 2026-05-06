# CLAUDE.md - DIIVE Development Guide

This file documents the DIIVE project architecture, development conventions, and key decisions to help Claude Code work effectively with the codebase.

## Getting Started

### First-Time Setup

**Clone and Install:**
```bash
# Clone repository
git clone <repository-url>
cd diive

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate diive

# Verify installation
python -m pytest tests/test_gapfilling.py -v
```

**Verify Python Version:**
```bash
python --version  # Should be 3.11.x
```

**Key Files to Review:**
- `CLAUDE.md` (this file) — Development guide and architecture
- `setup.py` or `pyproject.toml` — Package configuration
- `examples/README.md` — Example usage guide
- `.claude/settings.json` — Claude Code harness configuration

### Running Tests Locally

```bash
# Test gap-filling (fastest)
python -m pytest tests/test_gapfilling.py -v

# Test flux processing chain
python -m pytest tests/test_fluxprocessingchain.py -v

# Run all tests
python -m pytest tests/ -v
```

**Expected test times:**
- `test_gapfilling_randomforest` — 2-3 seconds
- `test_gapfilling_xgboost` — 2-3 seconds
- `test_gapfilling_longterm_randomforest` — 25-30 seconds
- `test_fluxprocessingchain` — 20-25 seconds

## Project Overview

**DIIVE** (Data Integration and Interactive Visualization Engine) is a Python library for time series processing, particularly ecosystem data. Originally developed by ETH Grassland Sciences for Swiss FluxNet, it provides tools for:
- Time series processing and feature engineering
- Machine learning-based gap-filling (Random Forest, XGBoost)
- Flux processing chain automation (Level-2 through Level-4.1)
- Quality control and outlier detection
- Data visualization (heatmaps, time series, scatter plots)

## Development Environment

**Conda Environment:**
```bash
# Windows
C:\Users\nopan\miniconda3\envs\diive

# macOS/Linux
~/miniconda3/envs/diive
```

**Python Version:** 3.11.x (exact, not 3.10 or 3.12)

**Install Environment:**
```bash
conda env create -f environment.yml
conda activate diive
python -m pip install -e .  # Install diive in editable mode
```

**Core Dependencies (Pinned Versions):**
| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | ≥1.3.0, <1.5.0 | Random Forest gap-filling |
| xgboost | ≥2.0.0, <2.2.0 | XGBoost gap-filling |
| shap | ≥0.43.0, <0.44.0 | SHAP feature importance |
| pandas | ≥2.0.0, <2.2.0 | DataFrames and data processing |
| numpy | ≥1.24.0, <1.26.0 | Numerical computing |
| matplotlib | ≥3.8.0, <3.10.0 | Plotting and visualization |
| scipy | ≥1.11.0, <1.13.0 | Scientific computing, STL decomposition |
| statsmodels | ≥0.14.0, <0.15.0 | STL decomposition (seasonal-trend) |
| pyarrow | ≥13.0.0, <14.0.0 | Parquet file I/O |

**Development Dependencies:**
- pytest ≥7.4.0 — Unit testing
- pytest-cov ≥4.1.0 — Code coverage
- black ≥23.0.0 — Code formatting (optional)
- flake8 ≥6.0.0 — Linting (optional)

**Always Activate Before Work:**
```bash
conda activate diive
```

**Verify Setup:**
```bash
python -c "import diive; print(diive.__version__)"
python -m pytest tests/test_gapfilling.py::TestGapFilling::test_gapfilling_randomforest -v
```

## .claude Configuration (Claude Code Harness)

**Purpose:** Configure Claude Code behavior, permissions, and dev servers for consistent cross-machine development.

### Shared Configuration (Version-Controlled)

**File:** `.claude/settings.json` — Check into git for team consistency

```json
{
  "version": "0.0.1",
  "permissions": {
    "default": "deny",
    "allow": [
      "bash:conda",
      "bash:pytest",
      "read:**",
      "edit:diive/**",
      "edit:tests/**",
      "edit:examples/**"
    ]
  },
  "env": {
    "CONDA_ENV": "diive",
    "PYTHONPATH": "${workspaceRoot}"
  }
}
```

### Dev Server Configuration (Version-Controlled)

**File:** `.claude/launch.json` — Check into git for consistent dev servers

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "pytest-gapfilling",
      "runtimeExecutable": "python",
      "runtimeArgs": ["-m", "pytest", "tests/test_gapfilling.py", "-v"]
    },
    {
      "name": "pytest-fluxprocessingchain",
      "runtimeExecutable": "python",
      "runtimeArgs": ["-m", "pytest", "tests/test_fluxprocessingchain.py", "-v"]
    }
  ]
}
```

**Usage:** `preview_start pytest-gapfilling` to run tests via Claude Code dev server.

### Machine-Specific Overrides (Don't Commit)

**File:** `.claude/settings.local.json` — Per-machine overrides

```json
{
  "model": "opus",
  "permissions": {
    "allow": [
      "bash:conda activate diive && python",
      "bash:cd /path/to/diive && pytest"
    ]
  }
}
```

**Add to .gitignore:**
```
.claude/settings.local.json
.claude/keybindings.json
.claude/memory/
.claude/worktrees/
.claude/scheduled-tasks/
```

### Key Harness Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| `permissions.default` | `"deny"` | Restrictive by default, allow specific operations |
| `env.CONDA_ENV` | `"diive"` | Conda environment name for subprocesses |
| `env.PYTHONPATH` | `"${workspaceRoot}"` | Allow imports from repo root |

## Project Structure

```
diive/
├── core/
│   ├── ml/
│   │   ├── common.py              # MlRegressorGapFillingBase (base class for all ML gap-filling)
│   │   └── feature_engineer.py    # FeatureEngineer (standalone feature engineering, v0.91.0)
│   ├── plotting/                  # Visualization (HeatmapDateTime, TimeSeries, HeatmapXYZ, HexbinPlot, etc.)
│   │   └── seasonaltrend.py       # SeasonalTrendDecomposition visualization (NEW v0.91.0)
│   ├── times/
│   │   └── decomposition_utils.py # STL, classical, harmonic decomposition (NEW v0.91.0)
│   └── dfun/                      # DataFrame utilities
│
├── pkgs/
│   ├── gapfilling/
│   │   ├── xgboost_ts.py         # XGBoostTS class
│   │   ├── randomforest_ts.py    # RandomForestTS and QuickFillRFTS classes
│   │   ├── longterm.py           # LongTermGapFillingBase and subclasses
│   │   └── mds.py                # MDS gap-filling
│   │
│   ├── fluxprocessingchain/
│   │   └── fluxprocessingchain.py # FluxProcessingChain (orchestrates Levels 2-4.1)
│   │
│   ├── analyses/
│   │   └── seasonaltrend.py       # SeasonalTrendDecomposition analysis module (NEW v0.91.0)
│   │
│   ├── timeseries/
│   │   └── harmonic.py            # Fourier analysis utilities (NEW v0.91.0)
│   │
│   └── [other packages...]
│
├── tests/
│   ├── test_gapfilling.py        # Gap-filling tests (optimized for speed)
│   └── test_fluxprocessingchain.py
│
├── examples/                      # Executable example scripts (v0.91.0+)
│   ├── README.md                  # Examples index and quick start
│   ├── visualization/
│   │   ├── heatmap_datetime.py    # HeatmapDateTime and HeatmapYearMonth examples
│   │   └── hexbin.py              # HexbinPlot 2D hexagonal binning examples
│   └── gap_filling/               # (Phase 2) Gap-filling workflow examples
│
└── notebooks/
    ├── Analyses/
    │   └── SeasonalTrendDecomposition.ipynb # 5 real-world examples with tutorial
    └── [other notebooks...]
```

## Module Architecture & API Overview

### Core Modules (`diive/core/`)

**`core/ml/`** — Machine Learning Infrastructure (v0.91.0)
- `feature_engineer.py` — Standalone 8-stage composable feature engineering
- `common.py` — Base class for all ML gap-filling models (SHAP-based feature importance)

**`core/plotting/`** — Visualization & Analysis (14 plot types)
- `heatmap_datetime.py` — Time series heatmaps (daily/monthly/yearly)
- `heatmap_xyz.py` — 3D scatter heatmaps
- `timeseries.py` — Interactive time series plots
- `hexbin_plot.py`, `ridgeline.py`, `histogram.py` — Alternative plot types
- `seasonaltrend.py` — Seasonal/trend decomposition visualization (NEW v0.91.0)
- `cumulative.py` — Cumulative flux analysis
- `dielcycle.py` — Diurnal cycle analysis

**`core/dfun/`** — DataFrame Utilities
- `frames.py` — Pivot/transform operations (yearmonth matrix ↔ long-form)
- `stats.py` — Statistical calculations (percentiles, aggregations)
- `fits.py` — Curve fitting utilities
- `regression.py` — Linear regression helpers

**`core/times/`** — Time Series Processing
- `resampling.py` — Temporal aggregation (daily→monthly, monthly→yearly)
- `decomposition_utils.py` — STL, classical, harmonic decomposition (NEW v0.91.0)

**`core/io/`** — Input/Output Operations
- `filereader.py` — Load various file formats (CSV, NetCDF, etc.)
- `files.py` — Parquet I/O (fast for large datasets)
- `filedetector.py` — Auto-detect file types and formats

**`core/base/`** — Base Classes
- `flagbase.py` — Quality flag definitions and utilities
- `identify.py` — Data identification and validation

### Package Modules (`diive/pkgs/`)

**`pkgs/gapfilling/`** — Gap-Filling Algorithms (v0.91.0 Composition Pattern)
- `randomforest_ts.py` — Random Forest gap-filling + QuickFillRFTS
- `xgboost_ts.py` — XGBoost gradient boosting gap-filling
- `longterm.py` — Multi-year yearly pooled models
- `mds.py` — Meteorological data similarity method (no ML training)
- `interpolate.py` — Linear/spline interpolation fallbacks

**`pkgs/fluxprocessingchain/`** — Complete Flux Processing (Levels 2-4.1)
- `fluxprocessingchain.py` — Main orchestrator for multi-level processing
- `level2_qualityflags.py` — Quality control flag generation
- `level31_storagecorrection.py` — Storage term correction

**`pkgs/analyses/`** — Time Series Analysis
- `seasonaltrend.py` — Decomposition analysis (separate from feature engineering)
- `correlation.py` — Daily correlation analysis: `DailyCorrelation` class with summary statistics, anomaly detection, and visualization
- `decoupling.py` — Stratified binning analysis: `StratifiedAnalysis` class for hierarchical binning (e.g., photosynthetic decoupling)
- `gapfinder.py` — Gap detection: `GapFinder` class to find and analyze consecutive missing values
- `gridaggregator.py` — Grid aggregation: `GridAggregator` class for 2D binning and aggregation across driver variables
- `histogram.py` — Distribution analysis: `Histogram` class for calculating and analyzing value distributions
- `optimumrange.py` — Optimum range analysis: `FindOptimumRange` class to identify optimal conditions for ecosystem responses
- `quantiles.py` — Distribution analysis: `percentiles101()` function to calculate percentiles 0-100
- `gapfinder.py` — Gap detection and classification
- `gridaggregator.py` — Spatial grid aggregation
- `quantiles.py` — Quantile-based analysis

**`pkgs/createvar/`** — Derived Variable Creation
- `air.py` — Aerodynamic resistance, dry air density
- `conversions.py` — Unit conversions (air temp, latent heat, evapotranspiration)
- `daynightflag.py` — Day/night classification from solar geometry
- `laggedvariants.py` — Time-lagged variable creation (past/future values)
- `noise.py` — Synthetic noise generation and impulse injection
- `potentialradiation.py` — Solar radiation estimates (Stull, Equation of Time)
- `timesince.py` — Count records since condition (dry periods, frost detection)
- `vpd.py` — Vapor pressure deficit calculations

**`pkgs/corrections/`** — Data Quality Corrections
- `setto.py` — Data flagging (set to missing, constant values, thresholds)
- `offsetcorrection.py` — Systematic bias correction (RH, radiation, measurement, wind direction)

**`pkgs/echires/`** — Eddy Covariance High Resolution Analysis
- `windrotation.py` — Coordinate rotation for wind vectors
- `lag.py` — Flux time lag detection
- `fluxdetectionlimit.py` — Detection limit calculation

**`pkgs/flux/`** — Flux Calculations & QA/QC
- `hqflux.py` — High-quality flux extraction
- `ustarthreshold.py` — U* threshold determination
- `ustar_mp_detection.py` — USTAR moving point detection (Papale et al., 2006) with bootstrap uncertainty (NEW v0.91.0)
- `uncertainty.py` — Uncertainty propagation
- `selfheating.py` — Sensor self-heating correction

**`pkgs/outlierdetection/`** — Outlier & Anomaly Detection
- Multiple methods for identifying erratic measurements

**`pkgs/qaqc/`** — Quality Assurance/Control
- Data validation and quality checks

**`pkgs/timeseries/`** — Additional Time Series Tools
- `harmonic.py` — Fourier analysis and spectral methods (NEW v0.91.0)

**`pkgs/formats/`** — Data Format Standardization
- `fluxnet.py` — FluxNet CSV format handling
- `meteo.py` — Meteorological data formats

**`pkgs/fits/`**, **`pkgs/binary/`**, **`pkgs/echires/`** — Specialized analysis

### Data Flow Architecture

```
Input Data (CSV/NetCDF/Parquet)
    ↓
[core/io] filereader, files
    ↓
[pkgs/createvar] Derived variables (VPD, ET, etc.)
    ↓
[pkgs/corrections] Data corrections and flagging
    ↓
[pkgs/qaqc] Quality assurance checks
    ↓
[pkgs/echires] Flux post-processing (rotation, lag, u*, limits)
    ↓
[pkgs/fluxprocessingchain] Multi-level orchestration
    ├─ Level 2: Quality flags
    ├─ Level 3.1: Storage correction
    ├─ Level 3.2: USTAR filtering
    ├─ Level 3.3: USTAR scenarios
    └─ Level 4.1: Gap-filling
         ├─ [core/ml/feature_engineer] Feature engineering
         ├─ [pkgs/gapfilling] RF/XGB/MDS gap-filling
         └─ Results per USTAR scenario
    ↓
[core/plotting] Visualization & analysis
    └─ [pkgs/analyses] Time series decomposition, trends, etc.
    ↓
Output (Parquet + Plots)
```

### Key Dependencies & Imports

```python
# Core ecosystem
import pandas as pd          # DataFrames
import numpy as np          # Numerical computing
import matplotlib.pyplot as plt  # Plotting base

# Machine learning
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap                 # Feature importance (TreeExplainer)

# Time series
from statsmodels.tsa.seasonal import STL
import scipy.fft           # Spectral analysis

# Data I/O
import pyarrow.parquet as parquet
```

## Core Architecture: Gap-Filling Pipeline

### Class Hierarchy

```
FeatureEngineer (core/ml/feature_engineer.py) [v0.91.0]
│   └── Standalone feature engineering for all 8-stage pipeline
│
MlRegressorGapFillingBase (core/ml/common.py)
├── RandomForestTS
│   └── (used in FluxProcessingChain.level41_longterm_random_forest)
├── XGBoostTS
│   └── (used in FluxProcessingChain.level41_longterm_xgboost)
└── LongTermGapFillingBase
    ├── LongTermGapFillingRandomForestTS
    └── LongTermGapFillingXGBoostTS
```

### Feature Engineering Pipeline

**v0.91.0 Architecture:** Feature engineering is now a standalone `FeatureEngineer` class. All ML models use the same composable 8-stage feature engineering pipeline:

**Composition-Based Workflow:**
```python
# Step 1: Create and apply feature engineer
engineer = FeatureEngineer(
    target_col='NEE',
    features_lag=[-1, 1],
    features_rolling=[12, 24],
    # ... configure all feature stages
)
df_engineered = engineer.fit_transform(df)

# Step 2: Pass pre-engineered features to gap-filling model
model = RandomForestTS(input_df=df_engineered, target_col='NEE', n_estimators=100)
```

**Benefits of Separation:**
- Pre-compute features once, reuse across multiple models (RF + XGB simultaneously)
- Test feature engineering independently from models
- Reuse features with non-gap-filling models
- Cleaner separation of concerns

1. **Lag Features** (`features_lag`, `features_lag_stepsize`, `features_lag_exclude_cols`)
   - Creates temporal context from past/future values
   - Example: `features_lag=[-2, 2]` creates lags [-2, -1, +1, +2]
   - Naming: `{col}{sign}{lag}` (e.g., `Tair_f-1`, `Tair_f+1`)

2. **Rolling Statistics** (`features_rolling`, `features_rolling_stats`, `features_rolling_exclude_cols`)
   - Default: rolling mean and std for each window
   - Advanced: median, min, max, std, q25, q75
   - Naming: `.{col}_mean{w}`, `.{col}_ROLLMEDIAN{w}`, etc.

3. **Temporal Differencing** (`features_diff`, `features_diff_exclude_cols`)
   - 1st order: rate of change
   - 2nd order: acceleration
   - Naming: `.{col}_DIFF{order}`

4. **Exponential Moving Average** (`features_ema`, `features_ema_exclude_cols`)
   - Weighted historical averages with exponential decay
   - Multiple span values for different time scales
   - Naming: `.{col}_EMA{span}`

5. **Polynomial Expansion** (`features_poly_degree`, `features_poly_exclude_cols`)
   - Non-linear relationship modeling
   - Degree 2: squared terms
   - Naming: `.{col}_POL{degree}`

6. **STL Decomposition** (`features_stl`, `features_stl_method`, `features_stl_seasonal_period`, `features_stl_exclude_cols`, `features_stl_components`)
   - Separates time series into trend, seasonal, residual components
   - Methods: 'stl', 'classical', 'harmonic'
   - Applied only to complete columns (no gaps to avoid circular dependency)
   - Naming: `.{col}_STL_TREND`, `.{col}_STL_SEASONAL`, `.{col}_STL_RESIDUAL`

7. **Timestamp Features** (optional, `vectorize_timestamps`)
   - Year, season, month, week, DOY, hour
   - Captures annual and diurnal cycles

8. **Sequential Record Number** (optional, `add_continuous_record_number`)
   - Simple 1, 2, 3, ... numbering for temporal ordering

### Key Implementation Details

**FeatureEngineer (`diive/core/ml/feature_engineer.py`) [v0.91.0]:**
- Standalone feature engineering class
- Implements all 8-stage composable pipeline (lag → rolling → diff → ema → poly → stl → timestamp → record_number)
- Methods:
  - `fit_transform(df)`: Create and apply all engineered features
  - `transform(df)`: Apply engineering (equivalent to fit_transform for stateless operations)
  - Private methods for each stage: `_lag_features()`, `_rolling_features()`, `_differencing_features()`, etc.
- **Naming conventions:** `.{col}_TYPE{detail}` (e.g., `.Tair_f_MEAN12`, `.Tair_f_STL_TREND`)
- **Target column:** Preserved in output and excluded from engineered features

**MlRegressorGapFillingBase (`diive/core/ml/common.py`):**
- Base class for all ML gap-filling (v0.91.0: accepts pre-engineered features only)
- Expects **pre-engineered features** from FeatureEngineer (no longer performs feature engineering)
- Methods:
  - `reduce_features()`: SHAP-based feature selection
  - `trainmodel()`: fit on training data, evaluate on test data
  - `fillgaps()`: predict all gaps using trained model
- **Default:** `shap_threshold_factor=0.5` (0.5-sigma confidence for feature acceptance)

**RandomForestTS and XGBoostTS:**
- Simple wrappers around MlRegressorGapFillingBase
- Accept pre-engineered features from FeatureEngineer
- **RandomForestTS:** Interpretable, robust to outliers
- **XGBoostTS:** Gradient boosting, excellent for non-linear patterns

**LongTermGapFillingBase & Subclasses:**
- Multi-year gap-filling with yearly model pooling
- USTAR scenario support for flux processing chain
- Methods:
  - `create_yearpools()`: partition data by year
  - `initialize_yearly_models()`: create model per year
  - `reduce_features_across_years()`: feature selection across all years
  - `fillgaps()`: train and predict for all years

**FluxProcessingChain (v0.91.0 - Full Feature Engineering Integration):**
- Orchestrates complete flux processing workflow (Levels 2-4.1)
- All gap-filling methods (RF, XGBoost, MDS) integrated with FeatureEngineer
- Methods:
  - `level41_longterm_random_forest()`: RF gap-filling with full feature engineering (v0.91.0)
  - `level41_longterm_xgboost()`: XGBoost gap-filling with full feature engineering (v0.91.0)
  - `level41_mds()`: MDS gap-filling (no feature engineering needed)
- Feature Engineering: All 24 FeatureEngineer parameters accepted in both RF and XGBoost methods
- Results stored in `level41['long_term_random_forest']`, `level41['long_term_xgboost']`, `level41['mds']`
- Each contains dict keyed by USTAR scenario
- Same engineered features used for fair RF vs XGBoost comparison

## Development Conventions

### Code Style

- **Python 3.11+** (f-strings, type hints preferred)
- **PEP 8** (4-space indentation)
- **Type hints** for function signatures (encouraged but not mandatory)
- **Docstrings:** NumPy/Google style with Args, Returns, Examples sections

### Commits

**CRITICAL:** Claude Code should **NEVER commit changes**. All commits are handled exclusively by the user.

After completing code changes:
1. Run tests to verify everything works
2. Stage files: `git add <files>`
3. **STOP** - Do NOT run `git commit`
4. Return to user with summary of what was changed
5. User will review and commit with appropriate message

**If user requests commit:** Only proceed if explicitly asked. Never assume.

**Format for user commits:**
```
One-line summary (under 70 chars, imperative mood)

Detailed explanation of what changed and why.
Multiple paragraphs OK.

Related files listed with @ notation:
@path/to/file.py brief description of changes
```

**Important:** Do NOT add Claude as co-author. User commits are standalone:
```bash
git commit -m "message"
```

### Parameter Documentation

**v0.91.0 - Composition-Based API:**

All feature engineering parameters are now passed to `FeatureEngineer`, not to gap-filling models:

```python
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS

# Step 1: Engineer features (configure 8-stage feature engineering pipeline)
engineer = FeatureEngineer(
    target_col='NEE',
    features_lag=[-1, -1],
    features_lag_stepsize=1,
    features_lag_exclude_cols=None,
    features_rolling=[12, 24],
    features_rolling_exclude_cols=None,
    features_rolling_stats=['median', 'min', 'max'],
    features_diff=[1],
    features_diff_exclude_cols=None,
    features_ema=[6, 24],
    features_ema_exclude_cols=None,
    features_poly_degree=2,
    features_poly_exclude_cols=None,
    features_stl=False,
    features_stl_method='stl',
    features_stl_seasonal_period=None,
    features_stl_exclude_cols=None,
    features_stl_components=None,
    vectorize_timestamps=True,
    add_continuous_record_number=False,
    sanitize_timestamp=False
)

# Step 2: Apply feature engineer to data
df_engineered = engineer.fit_transform(df)

# Step 3: Create gap-filling model with engineered features
rfts = RandomForestTS(
    input_df=df_engineered,
    target_col='NEE',
    verbose=1,
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Step 4: Train and fill gaps
rfts.trainmodel(showplot_scores=True, showplot_importance=True)
rfts.fillgaps()
gapfilled = rfts.get_gapfilled_target()
```

**Key Changes in v0.91.0:**
- Feature engineering parameters go to `FeatureEngineer`, NOT to gap-filling models
- Gap-filling models accept only `input_df` (pre-engineered), `target_col`, and model hyperparameters
- Set unused feature params to `None` explicitly to show full parameter space

## Coding Best Practices & Error Detection

### Input Validation

**Always validate at system boundaries** (user input, external data, API responses):

```python
def process_data(df, target_col, n_estimators=100):
    # Validate input data
    if df is None or df.empty:
        raise ValueError("Input DataFrame cannot be None or empty")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    if not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer")
    
    # Safe to proceed
    return result
```

**Don't validate internal/trusted data** — trust framework guarantees and internal contracts.

### Error Detection Strategy

1. **Use assertions for logic invariants** (fail fast during development):
   ```python
   assert len(features) > 0, "Features list cannot be empty"
   assert model_ is not None, "Model must be trained before prediction"
   ```

2. **Use exceptions at API boundaries** (catch external/user errors):
   ```python
   if not isinstance(series, pd.Series):
       raise TypeError(f"Expected pd.Series, got {type(series)}")
   ```

3. **Use logging for debugging** (trace execution without raising):
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.debug(f"Processing {len(df)} records with {len(features)} features")
   logger.warning(f"Feature '{col}' has {nan_count} NaN values")
   ```

4. **Return meaningful error messages**:
   ```python
   # Good
   raise ValueError("NEE_CUT_REF_orig column not found. Available columns: {list(df.columns)}")
   
   # Bad
   raise ValueError("Invalid column")
   ```

### Exception Handling

**Handle exceptions only where you can recover:**

```python
# Good - handle specific, recoverable errors
try:
    model = load_model(path)
except FileNotFoundError:
    logger.info(f"Model not found at {path}, training new model")
    model = train_model()

# Bad - catching all exceptions hides bugs
try:
    result = process_data()
except Exception:  # Too broad
    pass
```

**Let exceptions propagate when you can't handle them:**
```python
# Bad - hides the real error
try:
    gapfilled = model.fillgaps()
except:
    gapfilled = None  # Downstream code breaks without explanation

# Good - let it fail with context
gapfilled = model.fillgaps()  # Will raise if model is invalid
```

### Debugging Strategy

**When a test fails or code breaks:**

1. **Read the error message carefully**
   - Exception type tells you what went wrong
   - Traceback shows the call chain
   - Error message provides context

2. **Add logging before rerunning**
   ```python
   logger.debug(f"DataFrame shape: {df.shape}, dtypes: {df.dtypes}")
   logger.debug(f"Features selected: {features}")
   ```

3. **Check assumptions in order**
   - Does input data have expected structure?
   - Are function parameters the right type and values?
   - Do intermediate results match expectations?

4. **Use `print()` sparingly** — use logging instead so it can be disabled/filtered

### Code Review Checklist

Before committing, verify all of:

**Correctness:**
- [ ] **Tests pass locally** — `pytest tests/ -v` exits with code 0
- [ ] **No silent failures** — errors raise exceptions, don't return None
- [ ] **Inputs validated** — user/external data checked at boundaries
- [ ] **Error messages clear** — exceptions provide actionable context
- [ ] **SHAP/permutation variability handled** — tests use flexible ranges, not exact values
- [ ] **Feature naming consistent** — use `.{col}_TYPE{detail}` convention for ML features

**Code Quality:**
- [ ] **Type hints present** — function signatures have types (Args and return type)
- [ ] **Docstrings complete** — Args, Returns, Raises, Examples sections
- [ ] **No commented-out code** — remove or document why it's kept
- [ ] **No hardcoded paths** — use configs or parameters
- [ ] **Dead imports removed** — `import X` but X never used
- [ ] **No duplicate code** — extract to function if 3+ similar lines
- [ ] **Logging instead of print()** — use logger for debug output

**Documentation:**
- [ ] **CLAUDE.md updated** — if architecture/workflow changed
- [ ] **CHANGELOG.md updated** — if this is a feature/fix
- [ ] **Example updated/added** — if public API changed
- [ ] **README.md updated** — if setup instructions changed

**Performance:**
- [ ] **Tests run in <5 min** — gap-filling test ~30s, chain ~25s
- [ ] **No N² algorithms** — nested loops flagged for optimization
- [ ] **Feature reduction working** — should drop 50%+ of engineered features

**Machine Independence:**
- [ ] **No absolute Windows paths** — use env vars or relative paths
- [ ] **Works on Unix-like systems** — forward slashes, no `\n` assumptions
- [ ] **No hardcoded user paths** — avoid `C:\Users\username\...`
- [ ] **.claude config checked in** — shared settings.json/launch.json versioned

### Common Bug Patterns

**Pattern 1: Silent NaN Propagation**
```python
# Bad - NaN silently spreads
result = df[col].mean()  # Returns NaN if col has NaN, no warning

# Good - explicit handling
if df[col].isnull().any():
    logger.warning(f"Column {col} has {df[col].isnull().sum()} NaN values")
result = df[col].mean()  # Now aware of potential NaN
```

**Pattern 2: Mutable Default Arguments**
```python
# Bad - shared state across calls
def fit_features(features_list=[]):
    features_list.append(new_feature)  # Modifies default, causes bugs
    
# Good - use None and create fresh
def fit_features(features_list=None):
    if features_list is None:
        features_list = []
    features_list.append(new_feature)
```

**Pattern 3: Off-by-One in Ranges**
```python
# Bad - inclusive range creates unexpected behavior
features_lag=[-1, 1]  # Was this meant to be [-1, 0, 1] or just [-1, 1]?

# Good - use step parameter for clarity
features_lag=[-2, 2]  # With stepsize=1 creates [-2, -1, 0, 1, 2]
features_lag_stepsize=1
```

**Pattern 4: Missing Post-Condition Checks**
```python
# Bad - assumes operation succeeded
df.fillna(0)  # Returns new DataFrame, original unchanged
# Code assumes df is modified (it's not!)

# Good - verify result
df = df.fillna(0)  # Now df is actually modified
assert df.isnull().sum().sum() == 0  # Verify no NaN remain
```

## Testing

### Test Execution

**Always use conda environment:**
```bash
C:\Users\nopan\miniconda3\envs\diive\python -m pytest tests/test_gapfilling.py -v
```

### Key Test Files

- `tests/test_gapfilling.py`: RandomForest and XGBoost gap-filling tests
  - `test_gapfilling_randomforest()`: ~2.8 seconds
  - `test_gapfilling_xgboost()`: ~2.6 seconds
  - `test_gapfilling_longterm_randomforest()`: ~27 seconds

- `tests/test_fluxprocessingchain.py`: End-to-end flux processing chain
  - Tests Levels 2-4.1 including both RF and XGB gap-filling
  - ~22 seconds

### Test Optimization

**Minimal settings for speed:**
```python
# Gap-filling tests use minimal parameters
n_estimators=3              # Very fast
min_samples_split=5-10      # Larger minimums
min_samples_leaf=2-5
vectorize_timestamps=False  # Skip expensive features
add_continuous_record_number=False
```

**Assertion Strategy:**
- Use flexible ranges instead of hardcoded values
- `assertGreater()` / `assertLess()` instead of `assertAlmostEqual()`
- Accommodates SHAP calculation variability (±5-10%)

### Test Failure Workflow

**When a test fails:**
1. **Check the origin class/function** being tested first
2. **Fix the root cause** in the actual implementation if it's broken
3. **Then fix the test** to match the corrected implementation

Never modify tests to pass without fixing the underlying code — always fix root causes in the source code first, then adjust tests accordingly.

## Examples (v0.91.0+)

**Purpose:** Consolidated, executable examples demonstrating library usage. Located in `examples/` folder organized by topic.

**Structure:**
```
examples/
├── README.md                              # Index and quick start guide
├── visualization/                         # Plotting examples (10 examples)
│   ├── heatmap_datetime.py                # HeatmapDateTime with single/multi-panel layouts (6 examples)
│   ├── hexbin.py, scatter_xy.py, ...      # Other visualization classes
├── analyses/                              # Time series analysis examples (8 examples)
│   ├── correlation.py, decoupling.py, ...
│   └── seasonaltrend.py                   # SeasonalTrendDecomposition analysis
├── binary/                                # Binary data processing (2 examples)
│   └── extract.py                         # Bit extraction and manipulation
├── corrections/                           # Data corrections (7 examples)
│   ├── setto.py                           # Set values to missing/constant/threshold (3 examples)
│   └── offsetcorrection.py                # RH, radiation, measurement, wind direction offsets (4 examples)
├── createvar/                             # Derived variable creation (23 examples)
│   ├── air.py                             # Aerodynamic resistance, dry air density (2 examples)
│   ├── conversions.py                     # Air temp, latent heat, ET conversion (3 examples)
│   ├── daynightflag.py                    # Daytime/nighttime flag with heatmaps (1 example)
│   ├── laggedvariants.py                  # Lagged variable creation (3 examples)
│   ├── noise.py                           # Synthetic noise and impulse generation (4 examples)
│   ├── potentialradiation.py              # Solar radiation calculations (4 examples)
│   ├── timesince.py                       # Count records since condition (3 examples)
│   └── vpd.py                             # Vapor pressure deficit (3 examples)
├── echires/                               # Eddy covariance high-resolution analysis (4 examples)
│   ├── fluxdetectionlimit.py              # Flux detection limit (2 examples)
│   ├── lag.py                             # MaxCovariance time lag detection (1 example)
│   └── windrotation.py                    # WindRotation2D coordinate rotation (1 example)
├── flux/                                  # Flux processing & quality (7 examples)
│   ├── common.py                          # Detect flux variables (1 example)
│   ├── hqflux.py                          # High-quality flux extraction (1 example)
│   ├── selfheating.py                     # Self-heating correction (1 example)
│   ├── uncertainty.py                     # Random uncertainty quantification (1 example)
│   └── ustarthreshold.py                  # USTAR threshold detection (3 examples)
├── fits/                                  # Curve fitting (1 example)
│   └── fitter.py                          # BinFitterCP curve fitting (1 example)
├── timeseries/                            # Spectral & harmonic analysis (2 examples)
│   └── harmonic.py                        # Spectrogram & frequency interpretation (2 examples)
└── gap_filling/                           # (Phase 2) Gap-filling workflows
```

**Current count:** 76 runnable examples across 38 files (8 categories)

**Why separate examples from source files?**
- Keeps source code clean (no `_example_*()` functions)
- Executable by users: `python examples/visualization/heatmap_datetime.py`
- Easier to discover and test
- Single source of truth (not scattered across docstrings/notebooks)

**Running examples:**
```bash
# Run all examples at once (parallelized, ~2.7x faster)
python examples/run_all_examples.py

# Run individual examples
python examples/visualization/heatmap_datetime.py
python examples/analyses/correlation.py
python examples/gap_filling/randomforest_ts.py
```

**Example runner script (`examples/run_all_examples.py`):**
- Executes all examples in parallel (4 concurrent workers)
- Shows per-example execution time
- Reports failures with error messages
- Exit code 0 (success) or 1 (failure) for CI/CD integration
- Total time tracking: ~30s for 10 examples (9 visualization + 1 analysis) vs 80s sequential (~2.7x speedup)

**Adding new examples:**
1. Create function: `example_<feature>_<variant>()`
2. Add docstring describing what it demonstrates
3. Keep it runnable standalone
4. Use `dv.load_exampledata_parquet()` for consistent test data
5. Update `examples/README.md` with reference

**Consolidating examples from source files:**

When moving embedded examples from source code to `examples/` folder:
1. **Preserve original data sources** — If the embedded example used real data (e.g., `load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()`), replicate that in the new example. This ensures validation against known good results.
2. **Remove embedded example from source** — Delete `_example_*()` functions and `if __name__ == '__main__'` blocks from source file
3. **Update docstring** — Change reference from "Example notebook available in: notebooks/..." to "See `examples/createvar/conversions.py` for complete examples"
4. **Export new functions/classes** — Add exports to `diive/__init__.py` with both PascalCase and snake_case aliases
5. **Update tracking** — Increment example count in `examples/README.md`, `examples/run_all_examples.py`, and `CHANGELOG.md`

**Multi-panel visualization pattern:**

For displaying multiple heatmaps or plots side-by-side with proper spacing:
```python
fig, axes = plt.subplots(1, 3, figsize=(20, 9),
                         gridspec_kw={'wspace': 0.2},
                         constrained_layout=True)

dv.plot_heatmap_datetime(ax=axes[0], series=var1).plot()
dv.plot_heatmap_datetime(ax=axes[1], series=var2).plot()
dv.plot_heatmap_datetime(ax=axes[2], series=var3).plot()

axes[0].set_title("Variable 1")
axes[1].set_title("Variable 2")
axes[2].set_title("Variable 3")

fig.show()
```
Key parameters:
- `figsize=(20, 9)` — wider figure for side-by-side layout
- `gridspec_kw={'wspace': 0.2}` — 20% spacing between columns
- `constrained_layout=True` — automatic spacing management

**Docstring references:**
Minimal examples in class docstrings should reference the examples folder:
```python
Example:
    See `examples/visualization/heatmap_datetime.py` for complete examples.
```

## Architecture: v0.91.0 - Separated Feature Engineering from Gap-Filling

**Key Change - v0.91.0:** This major architectural update improves modularity and composability. Feature engineering is now a standalone responsibility via the new `FeatureEngineer` class.

**What Changed:**
1. **New `FeatureEngineer` class** (`diive/core/ml/feature_engineer.py`)
   - Standalone feature engineering for all 8-stage pipeline
   - Separates feature engineering from gap-filling
   - Enables pre-computing features once and reusing across multiple models

2. **Simplified Gap-Filling Classes**
   - `MlRegressorGapFillingBase` no longer accepts feature engineering parameters
   - `RandomForestTS` and `XGBoostTS` accept only `input_df` (pre-engineered), `target_col`, and model hyperparameters
   - `LongTermGapFillingBase` and subclasses similarly simplified

3. **FluxProcessingChain Updates**
   - `level41_longterm_random_forest()` and `level41_longterm_xgboost()` method signatures simplified
   - Feature engineering parameters removed from user-facing API

4. **Benefits of Separation**
   - **Composition-based design:** Engineer features → pass to any gap-filling model
   - **Reusability:** Same features can be used with RF, XGB, and other models simultaneously
   - **Testability:** Feature engineering can be tested independently
   - **Flexibility:** Features can be reused with non-gap-filling models

**Migration Pattern (v0.91.0):**

```python
# Feature engineering parameters go to FeatureEngineer
engineer = FeatureEngineer(
    target_col='NEE',
    features_lag=[-1, 1],
    features_rolling=[12, 24],
    # ... configure features ...
)
df_engineered = engineer.fit_transform(df)

# Gap-filling models accept pre-engineered data only
rfts = RandomForestTS(
    input_df=df_engineered,
    target_col='NEE',
    n_estimators=100  # model hyperparameters only
)
```

**Files Modified in v0.91.0 Architecture:**
- `diive/core/ml/feature_engineer.py` - NEW 550 LOC standalone class
- `diive/core/ml/common.py` - Simplified, accepts pre-engineered data
- `diive/pkgs/gapfilling/randomforest_ts.py` - Updated signatures and examples
- `diive/pkgs/gapfilling/xgboost_ts.py` - Updated signatures and examples
- `diive/pkgs/gapfilling/longterm.py` - Updated LongTermGapFilling classes
- `diive/pkgs/fluxprocessingchain/fluxprocessingchain.py` - Simplified method signatures
- `tests/test_gapfilling.py` - All tests updated to use new API

---

## Recent Implementations (v0.91.0)

### 1. Standalone Feature Engineering Class (Phase NEW - v0.91.0)
- Added `FeatureEngineer` class to `diive/core/ml/feature_engineer.py`
- Implements all 8-stage composable feature engineering pipeline
- Replaces embedded feature engineering in gap-filling classes
- Enables feature reuse across multiple models (RF, XGB simultaneously)
- All feature engineering parameters now in FeatureEngineer, not gap-filling models

### 2. Polynomial Features (Phase 2 of Feature Engineering)
- Added `features_poly_degree` and `features_poly_exclude_cols` parameters
- Implemented `_polynomial_features()` method in MlRegressorGapFillingBase
- Creates `.{col}_POL{degree}` columns (e.g., `.Tair_f_POL2` for squared)
- Integrated across all gap-filling classes and long-term variants
- Updated examples and tests to use polynomial features

### 2. Long-Term XGBoost Gap-Filling with FluxProcessingChain Integration
- Implemented `LongTermGapFillingXGBoostTS` class
- `FluxProcessingChain.level41_longterm_xgboost()` fully integrated with FeatureEngineer (v0.91.0)
- Accepts all 24 feature engineering parameters (same as Random Forest)
- Enables fair comparison: RF and XGB models trained on identical features
- Default hyperparameters: n_estimators=200, max_depth=6, learning_rate=0.3, early_stopping_rounds=10
- Results accessible via: `fpc.level41['long_term_xgboost'][ustar_scenario]`

### 3. SHAP-Based Feature Importance
- Replaced permutation importance with SHAP TreeExplainer
- Added `shap_threshold_factor` parameter (default 0.5 for lenient selection)
- Threshold calculation: `random_importance + k * random_sd`
- More robust to SHAP variability

### 4. QuickFillRFTS - Fixed for v0.91.0 Composition Pattern
- **FIXED (v0.91.0):** Now uses FeatureEngineer pattern internally
- Creates FeatureEngineer with minimal parameters (lag only, no rolling/diff/ema)
- Engineers features before passing to RandomForestTS
- Fixed invalid `features_lag=[-1]` parameter (must be range like `[-1, -1]`)
- Uses minimal hyperparameters for fast exploration: n_estimators=3, min_samples_split=10, min_samples_leaf=5
- Enhanced docstring emphasizing exploratory/testing use case

### 5. Test Suite Optimization
- 60-70% speedup in RandomForest tests
- Flexible assertion ranges instead of hardcoded values
- Reduced from ~6 seconds to ~2.8 seconds for basic test

### 6. STL (Seasonal-Trend Loess) Decomposition (Phase 6)
- Added 5 new STL parameters to feature engineering pipeline:
  - `features_stl`: bool (enable/disable STL)
  - `features_stl_method`: str ('stl', 'classical', 'harmonic')
  - `features_stl_seasonal_period`: int (period for seasonal component)
  - `features_stl_exclude_cols`: list (columns to exclude)
  - `features_stl_components`: list (['trend', 'seasonal', 'residual'])
- Implemented `_stl_features()` method in MlRegressorGapFillingBase
- Creates `.{col}_STL_TREND`, `.{col}_STL_SEASONAL`, `.{col}_STL_RESIDUAL` columns
- Applied only to complete columns (no gaps) to avoid circular dependency with gap-filling
- **Compatibility:** Uses `features_stl_method='harmonic'` if statsmodels STL version incompatible
- Integrated across all gap-filling classes and long-term variants
- Added test cases for STL features with different methods and components

### 7. Time Series Analysis: SeasonalTrendDecomposition (NEW v0.91.0)
- **Standalone analysis module** for decomposing time series (separate from feature engineering)
- **Class:** `SeasonalTrendDecomposition` in `diive.pkgs.analyses.seasonaltrend`
- **Three decomposition methods:**
  - **STL (default):** Seasonal-Trend Loess; robust to gaps and non-stationary data
  - **Classical:** Moving-average method; assumes stationarity
  - **Harmonic:** FFT-based Fourier analysis; no series length constraints
- **Key features:**
  - Properties: `.trend`, `.seasonal`, `.residual`, `.seasonality_strength`
  - Methods: `.detrend()`, `.deseasonalize()`, `.reconstruct()`, `.summary()`
  - Lazy evaluation: components cached after first access
  - Auto-detection: detects seasonal period via periodogram if not specified
  - Quality-weighted fitting: uses quality flags during decomposition
- **Utilities:**
  - `diive.core.times.decomposition_utils`: Core functions (STL, classical, harmonic, quality-weighted)
  - `diive.pkgs.timeseries.harmonic`: Fourier analysis (harmonics, periodogram, FFT)
  - `diive.core.plotting.seasonaltrend`: Visualization (4-panel plots, spectral analysis)
- **Notebook:** `notebooks/Analyses/SeasonalTrendDecomposition.ipynb` with tutorial and 5 examples
  - Example 1: Detrending for ML Gap-Filling
  - Example 2: Anomaly Detection & Quality Control
  - Example 3: Method Comparison (Harmonic vs Classical)
  - Example 4: Climate Change Impact Analysis
  - Example 5: Ecosystem Recovery Trends
- **Use cases:** Ecosystem recovery analysis, climate change detection, anomaly detection, detrending for ML

**Key Distinction:** Unlike STL feature engineering (extracts trend/seasonal for gap-filling models), the `SeasonalTrendDecomposition` class is for **independent time series analysis** — analyzing trends in isolation, anomaly detection, understanding seasonal patterns.

### 8. Enhanced FeatureEngineer Documentation (v0.91.0)
- **Improved docstring** in `diive/core/ml/feature_engineer.py` with scenario-based guidance
- Each parameter now includes:
  - **Typical values** for different data types (especially 30-min flux data)
  - **Default behavior** — what happens when set to None/False
  - **Effect on time series** — how it impacts temporal patterns
  - **Typical use cases** — when to enable/disable
- **Three complete scenarios** with parameter configurations:
  - Quick/Minimal (testing, ~5 features)
  - Fast/Standard (production balanced, ~15-20 features)
  - Comprehensive (research detailed, ~50-100 features)
- **Time series context** with ecological explanations (photosynthesis saturation, diurnal cycles, senescence)
- Users can now easily understand which parameters apply to their data

### 9. Comprehensive CO2 Flux Examples in FluxProcessingChain (v0.91.0)
- **Both RF and XGBoost examples now ACTIVE** (previously commented out)
- **Optimized for CO2 half-hourly flux (NEE) data** with detailed documentation
- **Identical feature engineering** for fair model comparison:
  - Lag: [-2, -1] (30-60 min past context)
  - Rolling: [2, 4, 12, 24, 48] (1hr to 24hr windows)
  - Differencing: [1, 2] (rate of change + acceleration)
  - EMA: [6, 12, 24, 48] (3hr to 24hr exponential moving averages)
  - Polynomial: degree 2 (light saturation curves)
  - STL: period=48 (daily cycle), all components
  - Timestamps: Enabled (essential for diurnal photosynthesis)
- **Feature reduction ENABLED** (reduce ~45-50 features to ~10-20)
- **Hyperparameters tuned for flux data**:
  - Random Forest: n_estimators=350, max_depth=15, min_samples_split=5
  - XGBoost: n_estimators=250, max_depth=6, learning_rate=0.1
- **Performance guidance**:
  - RF: 3-8 min per year per USTAR scenario, R² 0.60-0.80, more interpretable
  - XGB: 2-5 min per year per USTAR scenario, R² 0.65-0.85, faster and smaller models
- **Model comparison code** included for selecting best algorithm per site

### 10. Enhanced XGBoostTS Hyperparameter Documentation (v0.91.0)
- **Added comprehensive min_child_weight documentation** to `diive/pkgs/gapfilling/xgboost_ts.py`
  - Default: 1 (permissive, fine-grained splits)
  - Range: 1-10 typical
  - For flux data: 3-5 (moderate regularization), 10+ (heavy regularization)
  - Effect: Higher values prevent overfitting to noise, create shallower trees
- **Enhanced other hyperparameter documentation**:
  - n_estimators, max_depth, learning_rate with tuning guidance
  - subsample, colsample_bytree for additional regularization
  - early_stopping_rounds impact on convergence
- **Time series specific** — guidance tailored to noisy flux measurements
- Users can now confidently tune XGBoost for ecosystem flux data

### 11. TimeSince Class - Event-Based Time Tracking (v0.91.0)
- **New class** `TimeSince` in `diive/pkgs/createvar/timesince.py`
- **Purpose:** Count consecutive records since last occurrence of a condition
- **Use cases:**
  - Dry period detection (time since last precipitation > 0 mm)
  - Frost period detection (time since freezing temperature <= 0°C)
  - Warm spell analysis
  - Event-based time tracking for ecosystem analysis
- **Key features:**
  - Flexible limit ranges (upper_lim, lower_lim, include_lim boolean)
  - Flag column indicating inside/outside range (0=in range, 1=outside/NaN)
  - Time-since column counting consecutive out-of-range records
- **Usage:**
  ```python
  ts = dv.TimeSince(series, lower_lim=0, include_lim=False)  # Count > 0
  ts.calc()
  results = ts.get_full_results()  # DataFrame with results
  ```
- **Comprehensive documentation:**
  - Full class docstring with Parameters, Attributes, Methods sections
  - Examples in docstring (dry periods, frost detection)
  - See Also references examples/createvar/timesince.py
  - Notes on NaN handling and reset behavior
- **Exported:** `diive.TimeSince`, `diive.timesince`

### 12. Examples Consolidation Phase 1 Complete (74 examples)
- **Consolidated from source files to dedicated `examples/` folder:**
  - Visualization: 22 examples (heatmap 6, hexbin 3, timeseries 1, cumulative 3, other 1, dielcycle 1, histogram 2, ridgeline 2, scatter 3)
  - Analysis: 8 examples (correlation 1, decoupling 1, gapfinder 1, gridaggregator 1, histogram 1, optimumrange 1, quantiles 1, seasonaltrend 1)
  - Variable creation: 23 examples (air 2, conversions 3, daynightflag 1, laggedvariants 3, noise 4, potentialradiation 4, timesince 3, vpd 3)
  - Corrections: 7 examples (setto 3, offsetcorrection 4)
  - Binary: 2 examples (extract 2)
  - Eddy covariance: 11 examples (fluxdetectionlimit 2, lag 1, windrotation 1, flux/common 1, flux/hqflux 1, flux/selfheating 1, flux/uncertainty 1, flux/ustarthreshold 3)
  - Fits: 1 example (fitter 1)
- **Structure:** `examples/{visualization,analyses,corrections,createvar,binary,echires,fits,flux}/*.py` with 1-4 examples per file
- **Key improvements:**
  - Parallel runner script: `examples/run_all_examples.py` (4 concurrent workers, ~2.7x speedup)
  - Examples runnable standalone: `python examples/createvar/timesince.py`
  - Proper docstrings with Args, Returns, Examples sections
  - Multi-panel visualizations for complex examples
  - Real data usage patterns maintained from originals
- **Documentation:**
  - `examples/README.md` with structure, quick start, finding help (updated with echires examples)
  - Updated CLAUDE.md examples section with all 66 examples listed
  - CHANGELOG.md updated with example consolidation details

### 13. Eddy Covariance, Flux & Curve Fitting Examples (NEW v0.91.0)
- **Added 6 new examples to `examples/` folder:**
  - `echires/fluxdetectionlimit.py`: Flux detection limit calculation (2 examples)
  - `echires/lag.py`: MaxCovariance time lag detection with synthetic data
  - `echires/windrotation.py`: WindRotation2D coordinate rotation and tilt correction
  - `flux/common.py`: Detect base variables from flux variable names
  - `flux/hqflux.py`: High-quality CO2 flux analysis with Hampel filter (1 example)
  - `fits/fitter.py`: BinFitterCP curve fitting with confidence intervals and prediction bands
- **Exports added to `diive/__init__.py`:**
  - `MaxCovariance` (also as `max_covariance`)
  - `WindRotation2D` (also as `wind_rotation_2d`)
  - `BinFitterCP` (also as `bin_fitter_cp`)
  - `FluxDetectionLimit` already exported as `fdl`, `flux_detection_limit`
- **Total example count updated:** 62 → 70 examples (with uncertainty consolidation: 70 → 71, with ustarthreshold consolidation: 71 → 74)
- **Updated `examples/run_all_examples.py`:** Added 6 new example files to parallel runner

### 14. Parquet Subsetting with Flexible Column Filtering (v0.91.0)
- **File:** `dev_scripts/parquet_time_subset.py`
- **Three-tier cascading column filtering:**
  - **Tier 1:** Optional dot-column removal (`REMOVE_DOT_COLUMNS=True`)
    - Removes columns starting with '.' (intermediate processing columns)
  - **Tier 2:** Pattern-based exclusion (`EXCLUDE_PATTERNS=None`)
    - Exclude columns matching patterns (e.g., "FLAG", "BADM", or ["FLAG", "SUM"])
  - **Tier 3:** Pattern-based inclusion (`COLUMN_PATTERN=None`)
    - Keep only columns matching patterns (e.g., "FC", or ["FC", "LE"])
- **Configuration examples:**
  - Default: `COLUMN_PATTERN=None, EXCLUDE_PATTERNS=None, REMOVE_DOT_COLUMNS=True` — removes dot columns only
  - Exclude flags: `EXCLUDE_PATTERNS=["FLAG", "BADM"]` — excludes columns containing FLAG or BADM
  - CO2 flux only: `COLUMN_PATTERN="FC"` — keeps only columns containing FC
  - Combined: `COLUMN_PATTERN="FC", EXCLUDE_PATTERNS=["BADM"]` — keeps FC columns, removes BADM
- **Output:** Statistics show column counts at each stage (original → dots removed → exclusions applied → inclusions filtered)
- **All columns printed alphabetically** for easy verification and documentation
- **Use case:** Prepare lightweight subset parquet files for examples or analysis focused on specific variables

### 15. Enhanced High-Quality Flux Analysis Function (v0.91.0)
- **File:** `diive/pkgs/flux/hqflux.py`
- **Function:** `analyze_highest_quality_flux()`
- **New features:**
  - **Comprehensive input validation (8 checks):**
    - Flux must be pd.Series with datetime index
    - Latitude: -90 to 90, Longitude: -180 to 180
    - Window length positive, sigma thresholds positive
    - Clear error messages for each validation failure
  - **Robust error handling:**
    - Try-except wrapper around Hampel filter calculation
    - Graceful handling of empty data before statistical computations
  - **Summary statistics dict** (optional):
    - Returns `(dataframe, summary)` tuple when `return_summary=True`
    - Summary includes: total/valid records, outliers found, percentages, method parameters
    - Backward compatible (default `return_summary=False` returns dataframe only)
  - **Flexible plot configuration:**
    - `figsize` parameter for custom plot dimensions
    - `show_percentiles` parameter to conditionally display percentile lines
  - **Enhanced documentation:**
    - Detailed Raises section documenting all validation errors
    - Parameter descriptions with units and valid ranges
    - Note about rolling window requirements
- **Use case:** Robust flux quality control with configurable validation and optional analysis summaries

### 16. Self-Heating Correction Examples & XGBoostTS Integration (v0.91.0)
- **File:** `examples/flux/selfheating.py`
- **Moved from:** `diive/pkgs/flux/selfheating.py` (removed `_example()` and `_example_lae()` functions)
- **Single example:** `example_selfheating_ch_lae()`
  - Full SCOP workflow: physics → optimizer → applicator
  - Parallel IRGA measurements (open-path IRGA75 vs closed-path IRGA72)
  - Uses `load_exampledata_parquet_lae()` for example data
- **Refactored `_gapfill()` method in source:**
  - **XGBoostTS replacement:** Faster training, smaller models than Random Forest
  - **FeatureEngineer integration:** 4-stage standalone feature engineering (lag, rolling, timestamps)
  - **Removed MDV fallback:** Not needed for derived physics variable
  - **Performance metrics:** RMSE 10.7→1.4 (train), 2.6 (test) over 150 boosting rounds
  - **v0.91.0 composition pattern:** Feature engineering decoupled from gap-filling
- **Updated counts:**
  - Examples: 70 → 71 (added 1 selfheating + 1 uncertainty)
  - Flux examples: 2 → 4 (hqflux, selfheating, uncertainty)
  - Example files in runner: 36 files, 71 functions
- **Use case:** Self-heating correction for open-path IRGA sensors with XGBoost-based gap-filling

### 17. Random Uncertainty Example Consolidation & Visualization Improvements (v0.91.0)
- **Files:** `examples/flux/uncertainty.py`, `diive/pkgs/flux/uncertainty.py`

**Example Consolidation:**
- Moved example from source to examples folder (removed `example()` function)
- Single example: `example_random_uncertainty_pas20()` demonstrates 4-method hierarchical uncertainty quantification
- Demonstrates 4-method hierarchy:
  - Method 1: Sliding window (±7 days, ±1 hr) with meteorological similarity
  - Method 2: Median of similar-flux uncertainties (expanding window fallback)
  - Method 3: Similar flux range without time window restrictions
  - Method 4: 5 nearest fluxes by magnitude (final fallback)
- Cumulative uncertainty propagation with upper/lower bounds using uncertainties package

**Error Propagation Refactoring:**
- Explicit DataFrame creation (only sum needed columns)
- Separated cumsum operations for clarity and efficiency
- Added None handling in uncertainty extraction
- Enhanced documentation on error propagation assumptions (independent random errors)

**New Reporting Methods:**
- `report_method_summary()` — Formatted pandas DataFrame showing:
  - Method distribution (records per method, percentage coverage)
  - Uncertainty statistics (mean, std, min, max, range)
  - ASCII-compatible units for Windows console compatibility
- `report_cumulative_uncertainty_propagation()` — Enhanced table format:
  - Cumulative flux and uncertainty bounds
  - Upper/lower limits and range
  - Uncertainties package notation

**Plot Improvements:**
- `showplot_random_uncertainty()`:
  - Uses constrained_layout with explicit margins (left=0.05, right=0.98, top=0.94, bottom=0.08)
  - Tighter subplot spacing: hspace=0.25, wspace=0.25 (17-29% reduction in whitespace)
  - Descriptive titles with method explanations
  - Grid lines for readability
  - Better axis labels with units
- `showplot_cumulative_uncertainty_propagation()`:
  - Redesigned with fill_between for uncertainty band visualization
  - Bold title and legend styling
  - Dashed lines for bounds, thicker line for flux
  - Improved visual hierarchy
- Histogram in example:
  - Mean/median indicator lines
  - Better spacing and margins

**Spacing Optimizations:**
- Reduced figure heights (18×10 → 18×9, 14×6 → 14×5.5, 11×6 → 11×5.5)
- Explicit margins instead of constrained_layout for better control
- ~85-90% of available figure area now utilized
- All titles, labels, and legends fully visible without cutoff

**Source Cleanup:**
- Removed `example()` function and `if __name__ == '__main__':` block
- Kept RandomUncertaintyPAS20 class unchanged
- Clean separation between implementation and demonstrations

**Exports added to `diive/__init__.py`:**
- `RandomUncertaintyPAS20` (also as `random_uncertainty_pas20`)

**Updated counts:**
- Examples: 70 → 71 (added 1 uncertainty example)
- Flux examples: 3 → 4
- Example files in runner: 36 files, 71 functions

**Use case:** Quantify random measurement uncertainty for eddy covariance flux data (PAS20/FLUXNET standard) with comprehensive reporting and improved visualizations

### 18. USTAR Threshold Examples Consolidation (v0.91.0)
- **File:** `examples/flux/ustarthreshold.py`
- **Moved from:** `diive/pkgs/flux/ustarthreshold.py` (removed 3 example functions and if __name__ block)
- **Three consolidated examples:**
  1. `example_ustar_detection_mpt()` — Papale et al. (2006) multiple temperature class threshold detection
     - Demonstrates UstarDetectionMPT with 6 temperature classes, 20 USTAR subclasses, 10 bootstrap runs
     - Shows timing breakdown: initialization, detection, total execution
     - Reports seasonal USTAR thresholds with uncertainty bounds
  2. `example_ustar_threshold_constant_scenarios()` — Create flux scenarios with different USTAR thresholds
     - Demonstrates UstarThresholdConstantScenarios class
     - Creates 7 flux datasets with thresholds: 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 m/s
     - Shows scenario generation and filtering workflow
  3. `example_flag_multiple_constant_ustar_thresholds()` — Apply multiple USTAR thresholds for QA/QC
     - Demonstrates FlagMultipleConstantUstarThresholds class
     - Applies CUT_16 (0.053), CUT_50 (0.071), CUT_84 (0.095) thresholds (percentiles from detection)
     - Creates separate quality flags for each threshold (used in standard flux processing chains)
     - Enables joint uncertainty analysis combining random uncertainty + USTAR threshold scenarios
- **Use cases:**
  - Multi-threshold flux filtering for low-turbulence (u*) conditions
  - Creating USTAR scenario datasets for uncertainty propagation
  - Standard flux processing chain (FLUXNET, Swiss FluxNet Level 3.3)
- **Cleaned source:** Removed example(), example_scenarios(), example_flag_constant_ustar_threshold() functions
- **Updated counts:**
  - Examples: 71 → 74 (added 3 ustarthreshold examples)
  - Flux examples: 4 → 7
  - Example files in runner: 36 → 37 files, 74 total functions
- **Documentation updates:**
  - Updated `examples/README.md` (flux section + example counts)
  - Updated `CHANGELOG.md` with consolidation details
  - Updated `examples/run_all_examples.py` to include ustarthreshold examples

### 19. Harmonic Analysis & Spectral Examples (v0.91.0)
- **File:** `examples/timeseries/harmonic.py`
- **Two spectral analysis examples** demonstrating time-frequency analysis of CO₂ flux (NEE_CUT_REF_f):
  1. `example1_spectrogram_daily_pattern()` — 10-day spectrogram revealing persistent 24-hour photosynthesis cycle
  2. `example2_annual_spectrogram_phenology()` — Full-year spectrogram showing seasonal phenology changes
- **Visualization technique:** Short-time Fourier Transform (scipy.signal.spectrogram) with power spectral density evolution
- **Frequency interpretation guide:** Identifies and explains key periodicities for ecosystem CO₂ flux:
  - **0.02 (24h):** Primary photosynthesis/respiration cycle (predictable, use in gap-filling)
  - **0.04 (12h):** Semi-diurnal secondary oscillation driven by temperature and respiration patterns
  - **0.06 (8h):** Tertiary atmospheric circulation or measurement artifacts
  - **0.1 (5h):** High-frequency weather/turbulence noise (ignore in gap-filling models)
- **Harmonic bands:** Shows how secondary harmonics (0.04, 0.06) follow the intensity of primary 24h cycle
  - Weak harmonics during winter dormancy (Jan-Mar)
  - Bright harmonic bands during active growing season (Apr-Sep)
  - Fading bands during senescence/leaf fall (Oct-Dec)
- **Gap-filling context:** Validates feature engineering strategy—focus on predictable 24h photosynthesis cycle, naturally smooth out unpredictable high-frequency noise through rolling statistics and model averaging
- **Updated example count:** 78 examples across 39 files (Phase 2 consolidation: _example_rfts() moved to examples/gap_filling/randomforest_ts.py)
- **Updated CHANGELOG.md:** Added "Harmonic Analysis & Spectral Analysis" section with frequency interpretation details

### 20. Linear Interpolation Refactoring & Enhanced Documentation (v0.91.0)
- **File:** `diive/pkgs/gapfilling/interpolate.py`
- **Improvements:**
  - ✅ **Input validation** (5 checks): type, emptiness, limit ≥1, datetime index required
  - ✅ **Extracted helper function** `_calculate_gap_sizes()`: clearer, reusable gap size calculation
  - ✅ **Efficient statistics** (single filter): ~3x faster for large gap counts
  - ✅ **Edge case handling** (3 early returns): no missing data, no gaps, no fillable gaps
  - ✅ **Dead code cleanup**: removed commented-out code, dead variables
  - ✅ **Better documentation**: comprehensive docstring with workflow, advantages, edge cases
- **Performance gains:**
  - 10-15% faster typical datasets (fewer dataframe filters)
  - ~3x faster for high gap counts (optimized statistics)
  - Early returns save computation for edge cases
- **Output maintained:**
  - ✓ Same gap detection algorithm (GapFinder)
  - ✓ Same interpolation method (pandas.interpolate)
  - ✓ Same verbose table formatting
  - ✓ Identical return values and statistics
- **Examples (2 scenarios):**
  - `example_linear_interpolation_limit5()` — Generous gap-filling (fills gaps ≤5 records)
  - `example_linear_interpolation_limit1()` — Conservative gap-filling (fills only single-value gaps)
  - Both show detailed summary statistics with parameters, gap analysis, recovery rates
- **Enhanced verbose output:**
  - Shows method parameters (limit, time series length)
  - Gap analysis with fillable vs. skipped counts
  - Recovery rate with actual filled values (e.g., "47.27% of 1,843")
  - Gap size distribution (min/median/max/mean)
  - Professional columnar layout for readability
- **Exported:** `diive.linear_interpolation` (available as `dv.linear_interpolation()`)
- **Use case:** Simple, fast gap-filling for small gaps; preserves larger gaps for ML methods

### 21. FluxMDS Memory-Efficient Gap-Filling Optimization (v0.91.0)
- **File:** `diive/pkgs/gapfilling/mds.py`
- **Achievement:** **4.0x speedup** (improved from initial 2.3-2.4x Phase 1 optimization)
- **Key optimizations:**
  - Eliminated DataFrame copies at each quality level (18+ copies → 1-2 total)
  - Replaced `_prepare_dataframes()` with boolean masking on `self._gapfilling_df`
  - Introduced `_missing_mask` to track remaining gaps without creating copies
  - Refactored gap-filling methods to use in-place updates instead of returning new dataframes
  - Modified vectorized prediction methods to accept gap indices instead of workdf copies
- **Implementation details:**
  - `_fill_gap_predictions()` applies predictions directly using `.loc[]` with index labels
  - `_missing_mask` updated in-place using `.iloc[]` to avoid position/label confusion
  - Memory usage reduced by ~70% during quality level iterations
- **Validation:**
  - ✓ Final gap-filled values: bit-identical (max difference 1.78e-15 = machine epsilon)
  - ✓ Quality flags: exact match with original implementation
  - ✓ Predictions/SD differences: pass within floating-point tolerance
  - ✓ Results validated against original _FluxMDS reference implementation
- **Example:** `examples/gap_filling/mds_comparison.py` demonstrates 4.0x speedup with validation
- **Reference implementations:**
  - `FluxMDS`: Optimized class with vectorization + memory efficiency
  - `_FluxMDS`: Original unoptimized reference for validation/comparison
- **Note:** Minor dtype difference in counts column comparison (doesn't affect gap-filling quality)

### 22. Harmonized Gap-Filling Examples: RandomForest vs XGBoost (v0.91.0)
- **Files:** `examples/gap_filling/randomforest_ts.py`, `examples/gap_filling/xgboost_ts.py`
- **Purpose:** Direct side-by-side comparison of ML gap-filling methods on identical data/features
- **Harmonization details:**
  - **Data:** Both use 2020 (one year) for consistent scope and timing
  - **Features:** Identical 8-stage feature engineering pipeline
    - Lag: [-2, -1], Rolling: [2, 4, 12, 24, 48] with ['mean', 'median', 'min', 'max']
    - Diff: [1, 2], EMA: [6, 12, 24, 48], Poly: degree 2, STL: enabled, Timestamps: enabled
  - **Workflow:** Identical structure (load → engineer → model → reduce → train → fill)
  - **Visualizations:** Both include heatmap (observed vs gap-filled) + cumulative flux
  - **Hyperparameters (method-specific):**
    - RF: n_estimators=3, max_depth=None, min_samples_split=5, min_samples_leaf=2
    - XGB: n_estimators=50, max_depth=6, learning_rate=0.1, early_stopping_rounds=10
- **Performance (2020 data):** RF R²=0.8185, XGB R²=0.8141 (comparable, different architectures)
- **Key advantage:** Fair comparison enables users to select appropriate method for their data

### 23. Simplified Gap-Filling Docstrings (v0.91.0)
- **Files updated:** `diive/pkgs/gapfilling/mds.py`, `diive/pkgs/gapfilling/randomforest_ts.py`, `diive/pkgs/gapfilling/xgboost_ts.py`, `diive/pkgs/gapfilling/interpolate.py`, `diive/pkgs/gapfilling/scores.py`
- **Changes:**
  - Removed implementation details (workflow steps, optimization claims, speedup comparisons)
  - Focused on what each function/class does (functionality, use cases)
  - Added example file references (e.g., "See examples/gap_filling/randomforest_ts.py")
  - Simplified from verbose to concise while preserving essential information
- **Benefits:**
  - Clearer docstrings for API documentation
  - Better discoverability of example files
  - Consistent style across gap-filling module

### 24. Three-Way Gap-Filling Comparison: MDS vs Random Forest vs XGBoost (v0.91.0)
- **File:** `examples/gap_filling/comparison.py`
- **Update:** Extended comparison.py to include XGBoost as third gap-filling method
- **Visualizations:**
  - **Heatmap:** Four-panel side-by-side showing Observed (with gaps), MDS, Random Forest, XGBoost gap-filled results
  - **Cumulative flux:** All four series plotted with unit conversion and legend
  - **Performance table:** Execution time, training requirement, features used, algorithm approach
- **Data consistency:**
  - All three methods use identical May 2022 data (720 records, ~3-4% gaps)
  - Random Forest and XGBoost use identical engineered features (lag, rolling, diff, EMA, polynomial, STL, timestamps)
  - Fair comparison enabling method selection based on accuracy vs speed tradeoffs
- **Key features:**
  - MDS: No training required, meteorological similarity (SWIN, TA, VPD)
  - Random Forest: Bagging approach, interpretable, robust to outliers
  - XGBoost: Boosting approach, often higher accuracy, smaller models
- **Updated documentation:**
  - `examples/README.md` updated to describe three-way comparison
  - Example count: 82 examples across 44 files maintained

### 25. Generalized Hyperparameter Optimization for Any ML Model (v0.91.0)
- **File:** NEW `diive/core/ml/optimization.py` (dedicated to ML hyperparameter optimization)
- **Change:** Refactored `OptimizeParamsRFTS` → `OptimizeParamsTS` for model-agnostic design
- **Key improvements:**
  - Accepts any sklearn-compatible regressor via `regressor_class` parameter
  - Works with RandomForestRegressor, XGBRegressor, and custom models
  - Dynamic recommendation generation (auto-detects wrapper class: RandomForestTS, XGBoostTS, etc.)
  - Uses GridSearchCV with TimeSeriesSplit to avoid time series data leakage
- **Architecture:**
  - Moved from `diive/pkgs/gapfilling/randomforest_ts.py` to `diive/core/ml/optimization.py`
  - Now lives in core ML infrastructure (alongside FeatureEngineer, MlRegressorGapFillingBase)
  - Separated from gap-filling methods (which stay in `pkgs/gapfilling/`)
- **API & Backward Compatibility:**
  - Primary name: `OptimizeParamsTS` (imported from core.ml.optimization)
  - Backward compatibility: `OptimizeParamsRFTS = OptimizeParamsTS` alias
  - Snake-case aliases: `optimize_params_ts`, `optimize_params_rfts`
  - All public via `diive` namespace
- **New example:**
  - Added `example_xgboost_hyperparameter_optimization()` in `examples/gap_filling/xgboost_ts.py`
  - Demonstrates XGBoost parameter tuning with GridSearchCV
  - Mirrors RandomForest optimization example for consistency
- **Updated documentation:**
  - `examples/README.md`: xgboost_ts.py now shows "2 examples" (was 1)
  - Example count: 82 → 83 examples across 44 files
  - Gap-filling examples: 9 → 10

### 26. USTAR Moving Point Detection (ONEFlux Implementation) (v0.91.0)
- **Files:** NEW `diive/pkgs/flux/ustar_mp_detection.py` (main implementation), NEW `examples/flux/ustar_mp_detection.py` (example)
- **Class:** `UstarMovingPointDetection` — Faithful Python port of ONEFlux USTAR threshold detection
- **Algorithm:** Papale et al. (2006) moving point method for low-turbulence flux filtering
- **Key features:**
  - Nighttime respiration analysis (SW_IN < 10 W/m²) for pure signal without photosynthesis confounding
  - Seasonal stratification (4 seasons: DJF, MAM, JJA, SON) due to atmospheric stability variation
  - Temperature class stratification (7 classes per season) for atmospheric stability categorization
  - USTAR class stratification (20 classes per TA class) for friction velocity partitioning
  - Forward mode detection (ascending USTAR search, window_size=10) for stabilization point
  - Back mode detection (descending USTAR search, window_size=6) for validation
  - Median aggregation across TA classes (robust aggregation matching ONEFlux)
  - Annual threshold calculation (conservative: MAX of seasonal values) for data filtering
  - Bootstrap uncertainty estimation (100 iterations default, configurable) with 5%-95% CI
- **Matches ONEFlux constants (types.h):**
  - TA_CLASSES_COUNT = 7, USTAR_CLASSES_COUNT = 20
  - CORRELATION_CHECK = 0.5, PERCENTILE_CHECK = 90
  - THRESHOLD_CHECK = 1.0 (100% flux stabilization)
  - NIGHT_THRESHOLD = 10.0 W/m²
- **Usage example:**
  ```python
  import diive as dv
  df = dv.load_exampledata_parquet_lae()
  detector = dv.UstarMovingPointDetection(df, bootstrapping_times=30, verbose=1)
  thresholds = detector.detect()  # Seasonal thresholds
  annual = detector.get_annual_thresholds()  # Annual threshold (conservative)
  stats = detector.bootstrap()  # Uncertainty estimation
  ```
- **Annual threshold calculation (ONEFlux conservative approach):**
  - Computes median threshold across 7 TA classes for each of 4 seasons
  - Selects MAXIMUM seasonal threshold across all 4 seasons
  - Ensures most restrictive (highest) USTAR requirement applied to entire year
  - Prevents underestimation of respiration by using highest detected threshold
- **Example demonstrates:**
  - Detection workflow: load → detect → visualize
  - Bootstrap uncertainty with mean ± std and confidence intervals
  - Per-season visualization: 7 temperature classes stratified by color
  - Threshold marking on respiration vs USTAR scatter plots
- **Exports added to `diive/__init__.py`:**
  - `UstarMovingPointDetection` (PascalCase) and `ustar_mp_detection` (snake_case)
  - Also added `load_exampledata_parquet_lae` for LAE (Laegeren) example site
- **Comprehensive documentation:**
  - Module-level docstring explaining algorithm with Papale et al. reference
  - Class docstring with 9-step algorithm explanation
  - Method docstrings with step-by-step comments matching ONEFlux C code
  - Cross-checked against ONEFlux `ustar.c` and `types.h`
  - New method `get_annual_thresholds()` for conservative threshold retrieval

### 27. Outlier Detection Examples Consolidation (v0.91.0+)
- **File:** NEW `examples/outlierdetection/absolutelimits.py` (examples), modified `diive/pkgs/outlierdetection/absolutelimits.py` (source)
- **Module:** `diive.pkgs.outlierdetection.absolutelimits`
- **Two examples moved from source to examples folder:**
  1. `example_absolute_limits_basic()` — simple min/max threshold filtering
     - Demonstrates `AbsoluteLimits` class with fixed value range constraints
     - Shows statistics comparison (original vs filtered)
     - Use case: Simple value validation for any time series
  2. `example_absolute_limits_daytime_nighttime()` — separate day/night thresholds
     - Demonstrates `AbsoluteLimitsDaytimeNighttime` class with latitude/longitude-aware time detection
     - Shows statistics and breakdown by time of day
     - Use case: Flux data filtering (daytime variability vs nighttime stability)
- **Key improvements:**
  - Enabled plots (`showplot=True`) so users see visualizations when running examples
  - Clarified output: shows "Valid values after filtering" to distinguish series length from actual data points
  - Added percentage breakdown for day/night outlier rejection
  - Class docstrings updated to reference `examples/outlierdetection/absolutelimits.py`
- **Exports added to `diive/__init__.py`:**
  - `AbsoluteLimits` / `absolute_limits`
  - `AbsoluteLimitsDaytimeNighttime` / `absolute_limits_daytime_nighttime`
- **Cleaned source:**
  - Removed `example()` and `example_daytime_nighttime()` functions from source file
  - Removed `if __name__ == '__main__':` block
- **Updated documentation:**
  - `examples/README.md`: Added outlierdetection section with 2 examples
  - Example count: 83 → 85 examples across 44 → 45 files
  - Parallel runner updated to include outlierdetection examples
  - CLAUDE.md: This entry

### 28. Hampel Filter Examples Consolidation (v0.91.0+)
- **File:** NEW `examples/outlierdetection/hampel.py` (examples), modified `diive/pkgs/outlierdetection/hampel.py` (source)
- **Module:** `diive.pkgs.outlierdetection.hampel`
- **Two examples moved from source to examples folder:**
  1. `example_hampel_with_impulse_noise()` — day/night thresholds with synthetic spikes
     - Demonstrates `HampelDaytimeNighttime` class with separate day/night Median Absolute Deviation (MAD) thresholds
     - Shows iterative outlier removal until convergence
     - Use case: Robust spike/outlier detection in time series with time-of-day variation
  2. `example_hampel_global_threshold()` — single global threshold, iterative filtering
     - Demonstrates global mode (no day/night separation)
     - Shows single-pass vs iterative detection
     - Use case: Simple, consistent outlier detection across entire time series
- **Key improvements:**
  - Enabled plots (`showplot=True`) so users see visualizations when running examples
  - Adapted hardcoded example data paths to use `load_exampledata_parquet()`
  - Simplified output with clear statistics (original vs filtered)
  - Added detailed docstrings explaining Hampel algorithm and use cases
- **Exports added to `diive/__init__.py`:**
  - `HampelDaytimeNighttime` / `hampel_daytime_nighttime`
- **Cleaned source:**
  - Removed `_example_dtnt()` and `_example_cha()` functions from source file
  - Removed `if __name__ == '__main__':` block
  - Updated module docstring to universally describe algorithm (no domain-specific references)
  - Updated class docstring to reference `examples/outlierdetection/hampel.py`
- **Updated documentation:**
  - `examples/README.md`: Added hampel entry with 2 examples
  - `README.md`: Updated example count 85 → 87, added to examples list
  - Example count: 85 → 87 examples across 45 → 46 files
  - Parallel runner updated to include hampel examples
  - CLAUDE.md: This entry

### 29. Hampel Class API Refactoring (v0.91.0+)
- **File:** Modified `diive/pkgs/outlierdetection/hampel.py` (class rename and parameter refactoring)
- **Class:** Renamed `HampelDaytimeNighttime` → `Hampel` with clearer API
- **Problem Addressed:**
  - Old API was confusing: parameters `n_sigma_dt` and `n_sigma_nt` existed, but when `separate_day_night=False`, only `n_sigma_dt` was used and `n_sigma_nt` was ignored
  - Unclear which parameter to use when (global mode vs day/night mode)
- **Solution (Option 2):**
  - Renamed class from `HampelDaytimeNighttime` to `Hampel` (simpler, clearer intent)
  - Changed API to use `n_sigma` as primary parameter (used in all modes)
  - Added optional `n_sigma_daytime` and `n_sigma_nighttime` parameters for day/night overrides
  - When `separate_day_night=False`: uses `n_sigma` for global threshold
  - When `separate_day_night=True`: uses `n_sigma_daytime` and `n_sigma_nighttime` if provided, otherwise falls back to `n_sigma`
- **Backward Compatibility:**
  - `HampelDaytimeNighttime = Hampel` alias added to source file
  - Old class name still works for existing code
  - Old parameter names (`n_sigma_dt`, `n_sigma_nt`) will cause AttributeError (users can migrate to new names)
- **Exports updated in `diive/__init__.py`:**
  - New: `Hampel` / `hampel` (primary name)
  - Kept: `HampelDaytimeNighttime` / `hampel_daytime_nighttime` (backward compatibility)
- **Examples updated (`examples/outlierdetection/hampel.py`):**
  - Updated both examples to use `Hampel` class name
  - Example 1: `n_sigma=5.5` (single parameter for day/night mode)
  - Example 2: `n_sigma=5.5` (single parameter for global mode)
  - Removed redundant `n_sigma_dt` and `n_sigma_nt` parameters
- **Flagid Updated:**
  - Changed from `OUTLIER_HAMPELDTNT` to `OUTLIER_HAMPEL` (matches class name)
- **Key Improvements:**
  - **Clearer intent:** Users see `n_sigma` and immediately understand it's the main threshold
  - **Optional overrides:** Users can optionally specify different day/night thresholds without needing two parameters
  - **Backward compatible:** Old code using `HampelDaytimeNighttime` still works
  - **Consistent naming:** `Hampel` class name matches the algorithm name (Hampel filter)

### Migration Guide (Hampel API Change)

**Old Code (v0.90.0):**
```python
from diive.pkgs.outlierdetection.hampel import HampelDaytimeNighttime

# Global mode (confusing: n_sigma_dt used, n_sigma_nt ignored)
ham = HampelDaytimeNighttime(series=s, n_sigma_dt=5.5, separate_day_night=False)

# Day/night mode (confusing: both parameters must be specified)
ham = HampelDaytimeNighttime(series=s, n_sigma_dt=5.5, n_sigma_nt=5.5, separate_day_night=True)
```

**New Code (v0.91.0+):**
```python
import diive as dv

# Global mode (clear: use n_sigma for all records)
ham = dv.Hampel(series=s, n_sigma=5.5, separate_day_night=False)

# Day/night mode with same threshold for both
ham = dv.Hampel(series=s, n_sigma=5.5, separate_day_night=True)

# Day/night mode with different thresholds (optional)
ham = dv.Hampel(series=s, n_sigma=5.5, n_sigma_daytime=4.5, n_sigma_nighttime=6.5, separate_day_night=True)

# Still works (backward compatibility)
ham = dv.HampelDaytimeNighttime(series=s, n_sigma=5.5, separate_day_night=True)
```

### 30. Z-Score Increments Examples Consolidation (v0.91.0+)
- **File:** NEW `examples/outlierdetection/incremental.py` (examples), modified `diive/pkgs/outlierdetection/incremental.py` (source)
- **Module:** `diive.pkgs.outlierdetection.incremental`
- **One example moved from source to examples folder:**
  - `example_incremental_zscore()` — Detect outliers based on abrupt changes between consecutive values
    - Demonstrates `zScoreIncrements` class with synthetic impulse noise
    - Shows z-score calculation on forward, backward, and combined increments
    - Use case: Robust detection of spikes in any time series data
- **Key improvements:**
  - Enabled plot (`showplot=True`) so users see visualization when running example
  - Simplified output with clear statistics (original vs filtered)
  - Added detailed docstring explaining algorithm (three increment types)
  - Adapted hardcoded example data path to use `load_exampledata_parquet()`
- **Exports added to `diive/__init__.py`:**
  - `zScoreIncrements` / `zscore_increments`
- **Cleaned source:**
  - Removed `example()` function from source file
  - Removed `if __name__ == '__main__':` block
  - Updated module docstring with general description and quality flags
  - Updated class docstring with algorithm explanation and reference to examples
  - Removed flux-specific language (ecosystem data references)
- **Updated documentation:**
  - `examples/README.md`: Added incremental entry with 1 example
  - `examples/run_all_examples.py`: Added incremental.py to EXAMPLE_FILES
  - `README.md`: Updated example count 87 → 88, added to examples list and categories
  - Example count: 87 → 88 examples across 46 → 47 files
  - Outlierdetection examples: 4 → 5

## Troubleshooting Guide

### Setup Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: No module named 'diive'` | Conda env not activated or diive not installed | `conda activate diive` then `python -m pip install -e .` |
| `ModuleNotFoundError: No module named 'pytest'` | pytest not in conda env | `conda install pytest pytest-cov` or reinstall from environment.yml |
| `python: command not found` | Conda env not in PATH | `conda activate diive` and verify with `which python` |
| Wrong Python version (3.10/3.12 instead of 3.11) | Conda env created with wrong Python | `conda env remove -n diive && conda env create -f environment.yml` |
| `ImportError: cannot import name 'RandomForestTS'` | diive not installed or old version | `python -m pip install -e . --upgrade` |

**Verify fix:**
```bash
python --version  # Should show 3.11.x
python -c "from diive.pkgs.gapfilling import RandomForestTS; print('OK')"
```

### Test Execution Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `pytest: command not found` | pytest not installed | `conda install pytest` |
| `test_gapfilling_longterm_randomforest` hangs (>60s) | Slow machine or many features | Run smaller test: `pytest tests/test_gapfilling.py::TestGapFilling::test_gapfilling_randomforest -v` |
| `FAILED ... assert R² > 0.60` | Unstable SHAP importance, feature reduction dropped good features | Normal variability (±5%), rerun test; if persistent, check data quality |
| `Windows encoding error: codec can't decode...` | Windows path encoding with non-ASCII | Ensure locale is UTF-8; see "Windows Encoding Fix" below |
| `MemoryError: Unable to allocate...` | Too many features engineered or large dataset | Reduce `features_rolling` window count or disable `vectorize_timestamps` |

**Debug a failing test:**
```bash
# Run with verbose output
pytest tests/test_gapfilling.py::TestGapFilling::test_gapfilling_randomforest -vv -s

# Run with Python debugger
python -m pytest tests/test_gapfilling.py -vv --pdb
```

### Data Loading Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `FileNotFoundError: example data not found` | Example data not in repo or wrong path | Run from repo root; data in `diive/data/` or use `load_exampledata_parquet()` |
| `ParquetError: couldn't open file` | Corrupted or missing parquet file | Verify file exists: `ls -la data/` and check permissions |
| `ValueError: Index has duplicate entries` | Duplicate timestamps in data | Use `df = df[~df.index.duplicated(keep='first')]` to remove duplicates |
| `KeyError: Column not found` | Typo in column name or column doesn't exist | Print available columns: `print(df.columns)` |

### Model Training Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `ValueError: could not convert to float` | NaN in numeric column or wrong dtype | Check: `df.dtypes` and `df.isnull().sum()` |
| `sklearn.exceptions.NotFittedError: Model not trained` | Calling predict before `trainmodel()` | Always call `model.trainmodel()` before `model.fillgaps()` |
| `SHAP: TreeExplainer requires sklearn >= 1.0` | Old scikit-learn version | `pip install --upgrade scikit-learn` (must be ≥1.3.0) |
| `XGBoostError: Invalid predictor encountered` | XGBoost model save/load version mismatch | Retrain model; don't load pickled XGBoost models across versions |
| Gap-filling produces only NaN | No training data available after feature engineering | Check if features have gaps; engineered features often have NaN at edges |

**Debug model training:**
```python
# Check data quality before training
print(f"Shape: {df.shape}")
print(f"NaN counts:\n{df.isnull().sum()}")
print(f"Dtypes:\n{df.dtypes}")

# Check engineered features
engineer = FeatureEngineer(...)
df_eng = engineer.fit_transform(df)
print(f"Engineered shape: {df_eng.shape}")
print(f"Engineered NaN:\n{df_eng.isnull().sum().sum()}")
```

### Windows-Specific Issues

#### Windows Encoding Fix

**Problem:** UnicodeDecodeError on file read/write

**Solution - Add to Python script:**
```python
import sys
import io

# Force UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Or for files:
df = pd.read_csv('file.csv', encoding='utf-8')
df.to_csv('output.csv', encoding='utf-8', index=False)
```

#### Path Issues

**Problem:** Backslash paths cause escaping errors

**Solution:** Use forward slashes or raw strings:
```python
# Bad
path = "C:\Users\name\diive\data"  # \U is invalid escape

# Good
path = "C:/Users/name/diive/data"
path = r"C:\Users\name\diive\data"
from pathlib import Path
path = Path.home() / "diive" / "data"
```

### Claude Code Integration Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Permission denied: `bash:pytest` | Permission not in `.claude/settings.json` | Add to `allow` list in `.claude/settings.json` |
| `preview_start: server not found` | Dev server not in `.claude/launch.json` | Check `launch.json` has correct `name` and `runtimeExecutable` |
| Tests run with wrong Python version | `.claude/launch.json` uses system python | Specify full conda env path: `"/path/to/miniconda3/envs/diive/python"` |
| Memory not persisting across sessions | Memory folder not versioned | `.claude/memory/` is auto-generated, not git-tracked (this is correct) |

**Verify Claude Code setup:**
```bash
# Check settings validity
cat .claude/settings.json  # Should be valid JSON

# Test a permission
python --version
```

### Performance Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Gap-filling takes >10 min per year | Too many features or large n_estimators | Use `n_estimators=3` for quick test; `reduce_features=True` cuts features 50% |
| Memory usage spikes during SHAP | SHAP computes feature coalitions (expensive) | Use `reduce_features=False` to skip SHAP, or reduce `max_depth` |
| XGBoost slower than expected | early_stopping not converging | Lower `learning_rate` from 0.3 to 0.1, increase `early_stopping_rounds` |

**Profile code execution:**
```python
import time

t0 = time.time()
# Your code here
print(f"Elapsed: {time.time() - t0:.2f}s")
```

### Getting Help

**If the issue persists:**
1. Check git log for recent changes that might have broken something
2. Add logging: `logger.debug(f"variable = {variable}")` to trace execution
3. Check CLAUDE.md "Known Issues" section for common problems
4. Run example scripts to verify they work independently
5. Reduce problem to minimal reproducible example

## Known Issues & Workarounds

### 1. XGBoost Scientific Notation (FIXED)
**Issue:** XGBoost's `base_score` returned in bracket-enclosed scientific notation format: `[-4.12E0]`
**Workaround:** Monkey-patch in `MlRegressorGapFillingBase.__init__()` strips brackets before float conversion
**Status:** Fixed, no further action needed

### 2. SHAP Importance Variability
**Issue:** SHAP importances fluctuate (±5-10%) across runs
**Why:** TreeExplainer samples feature coalitions; variability is inherent to SHAP
**Solution:** Use flexible assertion ranges in tests (e.g., `assertGreater/assertLess` instead of `assertAlmostEqual`)

### 3. Feature Reduction Edge Case
**Issue:** Features with SHAP exactly equal to threshold were silently excluded
**Fix:** Changed condition to `~accepted_locs` (proper negation)
**Status:** Fixed in feature reduction logic

### 4. Rejected Features Count
**Issue:** Display showed "0 rejected" even when features were removed
**Root Cause:** Logic counted only rejected engineered features, not original inputs
**Fix:** Include all engineered features in rejection report

## Key Design Decisions

### 1. Why SHAP instead of Permutation Importance?
- SHAP provides principled game-theoretic feature contribution estimates
- More stable and interpretable than permutation importance
- TreeExplainer efficient for tree-based models
- Directly integrated with reduction threshold calculation

### 2. Why `shap_threshold_factor=0.5` (lenient)?
- Original default of 1.0 (1-sigma) was too strict
- 0.5 (0.5-sigma) better accommodates SHAP variability
- Allows more nuanced feature selection
- Can be tuned: k=1.0 (standard), k=2.0 (conservative)

### 3. Why Polynomial Features in Base Class?
- Ensures feature parity across RandomForest and XGBoost
- Consistent column naming and integration
- Simplifies long-term gap-filling implementation
- Allows composability with other feature engineering stages

### 4. Why Separate Long-Term Classes?
- Multi-year modeling requires yearly pooling logic
- USTAR scenario support for flux processing
- Feature reduction across years (not per-year)
- Clear separation of concerns from single-year models

### 5. Why FluxProcessingChain Level-4.1 Methods?
- Abstracts gap-filling strategy selection
- Supports multiple methods simultaneously (RF, XGB, MDS)
- Integrates with USTAR scenarios from Level-3.3
- Propagates results through complete processing chain

## Working with the Codebase (v0.91.0)

### Adding a New Feature Engineering Parameter (v0.91.0+)

**v0.91.0:** Feature engineering parameters are now added to `FeatureEngineer`, NOT to gap-filling classes.

1. **Update FeatureEngineer.__init__()** (`diive/core/ml/feature_engineer.py`)
   - Add parameter to signature with default None
   - Store as instance variable
   - Add docstring documentation

2. **Implement pipeline method** (e.g., `_new_features()`)
   - Add method to FeatureEngineer class
   - Call from `_create_features()` orchestrator (after appropriate stage)
   - Return modified DataFrame
   - Use `.{col}_NEWNAME{detail}` naming convention

3. **Update examples**
   - Show FULL parameter specification in FeatureEngineer (all params, unused as None)
   - Update RandomForestTS and XGBoostTS example docstrings to show FeatureEngineer usage
   - Update `_example_rfts()` and `_example_xgbts()` functions

4. **Update tests**
   - Add test case using FeatureEngineer with new parameter
   - Create gap-filling model with engineered features
   - Use flexible assertion ranges

5. **Update CHANGELOG and README**
   - Document in v0.91.0+ section
   - Add to README feature list if appropriate

**Important:** Gap-filling classes NO LONGER accept feature engineering parameters.
- RandomForestTS, XGBoostTS, LongTermGapFillingBase, etc.
- They accept only `input_df` (pre-engineered), `target_col`, and model hyperparameters
- This simplification makes the codebase more maintainable

### Adding a New Gap-Filling Method to FluxProcessingChain

**Note:** Both `level41_longterm_random_forest()` and `level41_longterm_xgboost()` are fully implemented 
with FeatureEngineer integration (v0.91.0). To add additional methods:

1. **Create method** (e.g., `level41_newmethod()`)
   - Add all 24 feature engineering parameters to method signature
   - Create FeatureEngineer instance with these parameters
   - Engineer features for each USTAR scenario
   - Pass pre-engineered data to gap-filling model
   - Process each USTAR scenario independently
   - Store results in `self._level41['new_method']` dict

2. **Example Pattern:**
   ```python
   def level41_longterm_newmethod(self, features, reduce_features=False, verbose=0,
                                  features_lag=None, features_lag_stepsize=1,
                                  # ... all 24 feature parameters ...
                                  **model_kwargs):
       # Create FeatureEngineer
       engineer = FeatureEngineer(...)
       
       # For each USTAR scenario:
       for ustar_scen, ustar_flux in self.filteredseries_level33_qcf.items():
           # Engineer features
           engineered_data = engineer.fit_transform(self.df[features])
           # Add target flux
           engineered_data = pd.concat([engineered_data, self.fpc_df[ustar_flux.name]], axis=1)
           # Create and train model
           model = NewGapFillingMethod(input_df=engineered_data, ...)
   ```

3. **Update test_fluxprocessingchain.py**
   - Call new method with test data
   - Add assertions for all USTAR scenarios

4. **Update example code** in docstring and notebooks

5. **Update CHANGELOG and README**

### Debugging SHAP Importance Issues

1. **Check random baseline:** Is `.RANDOM` feature included?
2. **Verify threshold:** `random_mean + k * random_sd` calculation
3. **Check feature counts:** Before vs. after reduction
4. **Use verbose=2:** Get detailed feature engineering output
5. **Inspect SHAP values:** Access via `model_.feature_importances_traintest_`

## Performance Tips

- **QuickFillRFTS** for rapid prototyping (3 seconds)
- **Minimal RandomForest** for quick testing (n_estimators=3, no timestamp features)
- **Long-term gap-filling** for multi-year data (use yearly pooling, not single model)
- **Feature reduction** essential for high-dimensional feature spaces
- **Vectorize timestamps** only if annual/diurnal patterns expected
- **Use n_jobs=-1** for parallel processing (if not in tests)

## Where ML Gap-Filling Is Used

ML gap-filling is integrated into the **FluxProcessingChain** workflow at **Level-4.1** (post-processing gap-filling):

1. **FluxProcessingChain.level41_longterm_random_forest()** — Multi-year Random Forest gap-filling
   - Uses `LongTermGapFillingRandomForestTS` class
   - Yearly pooled models across multiple years of data
   - Processes each USTAR scenario independently
   - Stores results in `fpc.level41['long_term_random_forest'][ustar_scenario]`

2. **FluxProcessingChain.level41_longterm_xgboost()** — Multi-year XGBoost gap-filling  
   - Uses `LongTermGapFillingXGBoostTS` class
   - Same architecture as Random Forest for fair comparison
   - Useful for non-linear patterns where boosting outperforms bagging
   - Stores results in `fpc.level41['long_term_xgboost'][ustar_scenario]`

3. **FluxProcessingChain.level41_mds()** — MDS gap-filling (alternative, no feature engineering)
   - Uses meteorological relationships (temperature, solar radiation, VPD)
   - No ML model training required
   - Stores results in `fpc.level41['mds'][ustar_scenario]`

### Workflow

For each **USTAR scenario** (e.g., CUT_50, CUT_90):
1. **FeatureEngineer** creates 8-stage engineered features from input data
2. **LongTermGapFillingBase** subclass trains yearly pooled models
3. Gap-filling results added to `fpc.fpc_df` with flux values and gap-filled flags
4. Results accessible via `fpc.level41[method][ustar_scenario]` for detailed analysis

### Output Structure

Results stored per method and USTAR scenario:
```python
# Access model instance for detailed analysis
model = fpc.level41['long_term_random_forest']['CUT_50']

# Key attributes:
model.gapfilled_           # Gapfilled flux values (pd.Series)
model.results_yearly_      # Dict of yearly model results
model.scores_              # Model performance scores
model.feature_importances_yearly_  # Feature importance per year
model.features_reduced_across_years  # Selected features after SHAP reduction
```

## FluxProcessingChain Gap-Filling Integration (v0.91.0)

### Both Random Forest and XGBoost Fully Implemented

**Method 1: Random Forest Gap-Filling**
```python
fpc.level41_longterm_random_forest(
    features=['TA', 'SW_IN', 'VPD'],  # Feature columns from self.df
    # Feature Engineering (all 24 parameters)
    features_lag=[-1, 1],
    features_rolling=[12, 24],
    features_rolling_stats=['median', 'min', 'max'],
    features_diff=[1],
    features_ema=[6, 24],  # 6hr and 24hr exponential moving averages
    features_poly_degree=2,
    features_stl=False,
    vectorize_timestamps=True,
    add_continuous_record_number=True,
    sanitize_timestamp=True,
    # Gap-Filling Parameters
    reduce_features=False,
    verbose=True,
    # Random Forest Hyperparameters
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
# Access results: fpc.level41['long_term_random_forest']['CUT_50']
```

**Method 2: XGBoost Gap-Filling (for comparison)**
```python
fpc.level41_longterm_xgboost(
    features=['TA', 'SW_IN', 'VPD'],
    # Same feature engineering as Random Forest for fair comparison
    features_lag=[-1, 1],
    features_rolling=[12, 24],
    features_rolling_stats=['median', 'min', 'max'],
    features_diff=[1],
    features_ema=[6, 24],
    features_poly_degree=2,
    features_stl=False,
    vectorize_timestamps=True,
    add_continuous_record_number=True,
    sanitize_timestamp=True,
    # Gap-Filling Parameters
    reduce_features=False,
    verbose=True,
    # XGBoost Hyperparameters
    n_estimators=200,
    max_depth=6,
    learning_rate=0.3,
    early_stopping_rounds=10,
    random_state=42,
    n_jobs=-1,
)
# Access results: fpc.level41['long_term_xgboost']['CUT_50']
```

**Key Points:**
- Both methods use **identical feature engineering** for fair comparison
- FeatureEngineer automatically created and applied inside each method
- Features engineered once per USTAR scenario, not per year
- Results stored in separate dicts within `fpc.level41`
- Same hyperparameters can be used for both, or tuned separately

### Notebook Example
See `notebooks/flux/FluxProcessingChain.ipynb` for complete working example:
- Cell #307: Random Forest with full feature engineering (ready to run)
- Cell #308: XGBoost with full feature engineering (commented out, uncomment to compare)
- Cell #311: MDS gap-filling

## Future Phases (Not Yet Implemented)

- **Phase 7+:** Interaction features (e.g., Tair × Rg)
- **Phase 8+:** Autocorrelation features
- (Note: Phases 1-6 now complete: lag, rolling, diff, EMA, polynomial, STL)

## Useful References

- **XGBoost Parameters:** https://xgboost.readthedocs.io/en/stable/parameter.html
- **Random Forest:** https://scikit-learn.org/stable/modules/ensemble.html#random-forests
- **SHAP:** https://shap.readthedocs.io/
- **Swiss FluxNet Processing Chain:** https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/

## Common Development Workflows

### Workflow 1: Fix a Bug in Gap-Filling

**Scenario:** Test fails for RandomForestTS gap-filling

**Approach:**
1. Run the failing test: `pytest tests/test_gapfilling.py::TestGapFilling::test_gapfilling_randomforest -v`
2. Read error message → identify which class is involved (e.g., MlRegressorGapFillingBase, FeatureEngineer)
3. Add logging to that class to understand state
4. Check the root cause in the class implementation
5. Fix the class first (not the test)
6. Run test again to verify
7. If test still fails, update assertions to match corrected behavior
8. Run full test suite: `pytest tests/test_gapfilling.py -v`

**Key Files:** `diive/core/ml/common.py`, `diive/pkgs/gapfilling/randomforest_ts.py`

### Workflow 2: Add a New Feature Engineering Stage

**Scenario:** Add exponential moving average (EMA) features

**Approach:**
1. Add parameter to `FeatureEngineer.__init__()` (e.g., `features_ema`)
2. Implement `_ema_features()` method following existing stage pattern
3. Call from `_create_features()` orchestrator at correct position
4. Use consistent naming: `.{col}_EMA{span}`
5. Update docstring with parameter docs
6. Update examples in RandomForestTS/XGBoostTS docstrings
7. Add test case: create engineer with new param, verify column names/values
8. Test with both RandomForest and XGBoost for feature parity
9. Update CLAUDE.md parameter documentation

**Key Files:** `diive/core/ml/feature_engineer.py`, tests

### Workflow 3: Debug Quality Issues in Results

**Scenario:** Gap-filled values look suspicious (wrong scale/distribution)

**Approach:**
1. Check input data validity: NaN counts, ranges, units
2. Check feature engineering: are engineered features correct?
   - Lag features: verify correct past/future offsets
   - Rolling stats: verify window sizes and calculations
   - STL decomposition: verify seasonal period is appropriate
3. Check model training: is there sufficient training data?
   - Minimum ~100 records with all features available
   - Check train/test split is balanced
4. Check SHAP importance: which features actually used?
   - Some features may be rejected during feature reduction
5. Check prediction logic: is model predicting reasonable ranges?
   - Use `model.scores_` to assess fit quality (R², MAE, RMSE)
6. If issue persists, add intermediate assertions to catch divergence early

**Debugging Commands:**
```python
# Check feature engineering output
engineer = FeatureEngineer(...)
df_eng = engineer.fit_transform(df)
print(df_eng.columns)  # Verify all stages applied
print(df_eng.isnull().sum())  # Check for unexpected NaN

# Check model state
rfts = RandomForestTS(input_df=df_eng, ...)
print(rfts.scores_)  # Check train/test quality
print(rfts.feature_importances_)  # Which features used?
```

### Workflow 4: Profile Code for Performance Bottlenecks

**Scenario:** Gap-filling takes too long

**Approach:**
1. Identify slow section: Which stage takes longest?
   - Feature engineering? (use minimal features for quick test)
   - Model training? (reduce n_estimators, tree depth)
   - SHAP computation? (expensive, use feature_importance_ for quick debug)
2. Add timing checkpoints:
   ```python
   import time
   t0 = time.time()
   result = operation()
   print(f"Operation took {time.time() - t0:.2f}s")
   ```
3. Check test settings — production uses `n_estimators=100+`, tests use `n_estimators=3`
4. For long-term gap-filling, yearly pooling is intentional (not an optimization opportunity)

**Quick Settings for Testing:**
- `n_estimators=3` (very fast)
- `vectorize_timestamps=False` (skip expensive feature generation)
- `features_rolling=None, features_diff=None` (minimal feature stages)
- `random_state=42` (reproducible, helps with debugging)

### Workflow 5: Add a New Gap-Filling Method to FluxProcessingChain

**Scenario:** Implement new gap-filling algorithm (e.g., Kalman filter)

**Approach:**
1. Create new gap-filling class inheriting from `MlRegressorGapFillingBase` or standalone
2. Implement required methods: `trainmodel()`, `fillgaps()`, `get_gapfilled_target()`
3. Create method in FluxProcessingChain: `level41_newmethod()`
4. Accept all 24 feature engineering parameters + method hyperparameters
5. Create FeatureEngineer, apply pre-engineering, pass to gap-filling model
6. Store results in `self._level41['new_method']` dict keyed by USTAR scenario
7. Update test_fluxprocessingchain.py with end-to-end test
8. Update docstrings and examples
9. Update CLAUDE.md with integration details

**Template:**
```python
def level41_newmethod(self, features, reduce_features=False, verbose=0,
                      features_lag=None, ..., **method_kwargs):
    """Method docstring with Args, Returns, Example"""
    engineer = FeatureEngineer(...)
    self._level41['new_method'] = dict()
    
    for ustar_scen, ustar_flux in self.filteredseries_level33_qcf.items():
        # Engineer features
        df_eng = engineer.fit_transform(self.df[features])
        # Create and train model
        model = NewGapFillingMethod(input_df=df_eng, ...)
        # Fill gaps
        model.fillgaps()
        # Store results
        self._level41['new_method'][ustar_scen] = model
```

## Quick Command Reference

```bash
# Activate environment
conda activate diive

# Run specific test
python -m pytest tests/test_gapfilling.py::TestGapFilling::test_gapfilling_randomforest -v

# View class docstring
python -c "from diive.core.ml.common import MlRegressorGapFillingBase; help(MlRegressorGapFillingBase)"

# Check conda env and Python version
python --version

# Run all gapfilling tests
python -m pytest tests/test_gapfilling.py -v

# Run flux processing chain test
python -m pytest tests/test_fluxprocessingchain.py -v

# Run all tests with coverage
python -m pytest tests/ --cov=diive --cov-report=html

# Check code style (optional)
black --check diive/
flake8 diive/ --max-line-length=120
```

---

## Maintaining CLAUDE.md

This document is the **single source of truth** for DIIVE development. Keep it current as the codebase evolves.

### When to Update CLAUDE.md

**Always update CLAUDE.md when:**
1. **Architecture changes** — new modules, classes, or design patterns
2. **New major features** — gap-filling method, analysis type, visualization
3. **Development conventions change** — code style, commit format, testing approach
4. **API surface changes** — new public functions/classes, parameter changes
5. **Setup/environment changes** — dependencies added/removed, Python version updated
6. **Known issues discovered** — add to "Known Issues" section with workaround

**Optionally update CLAUDE.md for:**
- Bug fixes (unless they reveal design issues)
- Internal refactors that don't change API
- Performance optimizations
- Documentation improvements

### How to Update CLAUDE.md

**Structure:**
- Keep existing sections; don't reorganize unless really needed
- Add new "Recent Implementations" entry at the end of that section (before "Known Issues")
- Format: numbered heading (### 27. ...) with implementation details
- Use consistent bullet points and code blocks

**Format for New Implementation:**
```markdown
### N. Feature Title (v0.XX.0)
- **Files:** Path/to/file.py (main), Path/to/test.py (test)
- **Class/Function:** ClassName or function_name
- **Key changes:**
  - Bullet point explaining what changed
  - Why it matters
  - Dependencies or side effects
- **Example usage:**
  ```python
  # Code snippet showing how to use
  ```
- **Updated documentation:**
  - What sections of CLAUDE.md were updated
  - What examples were added
  - Test coverage status
```

**When committing CLAUDE.md changes:**
```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md: add section on [feature]"
```

**Note:** Claude Code (you) should NEVER commit. User commits all changes, including CLAUDE.md updates.

### Keeping Sections Accurate

| Section | Update Frequency | Who Updates | Purpose |
|---------|------------------|-------------|---------|
| Getting Started | When setup changes | Developer | First-time setup guide |
| Development Environment | When dependencies change | Developer | Exact versions, installation |
| Project Structure | When file structure changes | Developer | Module organization overview |
| Known Issues | As issues are discovered | Developer | Workarounds and solutions |
| Recent Implementations | With each feature | Developer | Version history and decisions |
| Code Review Checklist | When standards change | Team consensus | Quality gates |
| Troubleshooting Guide | As new problems arise | Developer | Common issues and fixes |

### Deprecation Notice Format

When removing or deprecating features, add to "Known Issues" section:

```markdown
### Deprecated: Feature Name (v0.XX.0)
**Status:** Removed in v0.XX.0

**What changed:** Old way no longer works
**Why:** Better approach available
**Migration:**
- Old: `old_function()`
- New: `new_function()`
- Example: `result = new_function(data, param=True)`

**Timeline:** Supported until v0.XX.0, removed in v0.XX.0+1
```

### Cross-Linking

- Always link to file paths with `:line_number` when referencing implementation
- Link to example files: `examples/gap_filling/randomforest_ts.py`
- Reference tests: `tests/test_gapfilling.py::TestGapFilling::test_name`
- Use relative paths from repo root

### Version Numbering

- **v0.90.0** — Major API change (gap-filling refactor)
- **v0.91.0** — Feature engineering separation, multiple gap-filling methods
- **vX.YY.Z** — X=major, YY=minor, Z=patch

Increment in CLAUDE.md header when releasing new version.

---

## v0.91.0 Completion Status

**FULLY IMPLEMENTED:**
✅ FeatureEngineer - Standalone 8-stage composable feature engineering
✅ MlRegressorGapFillingBase - Accepts pre-engineered features only
✅ RandomForestTS - Full feature engineering support (via FeatureEngineer)
✅ XGBoostTS - Full feature engineering support (via FeatureEngineer)
✅ LongTermGapFillingRandomForestTS - Full feature engineering support
✅ LongTermGapFillingXGBoostTS - Full feature engineering support
✅ FluxProcessingChain.level41_longterm_random_forest() - FeatureEngineer integrated
✅ FluxProcessingChain.level41_longterm_xgboost() - FeatureEngineer integrated
✅ QuickFillRFTS - Fixed to use FeatureEngineer pattern
✅ All tests updated to use new composition pattern
✅ Notebooks updated with working examples

**READY FOR PRODUCTION USE**

---

**Last Updated:** 2026-05-06
**Version:** v0.91.0+ (with Getting Started, .claude Configuration, Troubleshooting Guide, cross-machine setup, outlier detection examples consolidation + API refactoring)
