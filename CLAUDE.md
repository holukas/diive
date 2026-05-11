# CLAUDE.md - DIIVE Development Guide

Quick reference for DIIVE development. For detailed version history, see `CHANGELOG.md`.

## Quick Start

**Setup (first time):**

```bash
uv sync                                           # Install dependencies
uv run pytest tests/test_gapfilling.py -v        # Run tests
```

**Or using conda (legacy):**

```bash
conda env create -f environment.yml
conda activate diive
python -m pytest tests/test_gapfilling.py -v
```

**Key files:**

- `.claude/settings.json` — Claude Code configuration
- `examples/README.md` — Example usage guide
- `CHANGELOG.md` — Version history and recent implementations

## Project Overview

**DIIVE** — Data Integration and Interactive Visualization Engine for time series processing (ecosystem flux data).

**Core capabilities:**

- **Timestamp sanitization** (10-step validation, monotonicity check, frequency detection)
- Feature engineering (8-stage composable pipeline)
- ML gap-filling (Random Forest, XGBoost)
- Flux processing chain (Levels 2-4.1)
- Quality control and outlier detection
- Data visualization

## Development Environment

**Python:** 3.12-3.13 (via `pyproject.toml`)

**Package Manager:** **`uv`** (modern, fast, reliable)

- Install uv: https://docs.astral.sh/uv/getting-started/
- Faster dependency resolution than pip/conda
- Deterministic builds via `uv.lock`

**Quick commands:**

- `uv sync` — Install all dependencies
- `uv run python script.py` — Run scripts with project env
- `uv run pytest tests/` — Run tests
- `uv add package_name` — Add dependencies
- `uv pip list` — List installed packages

**Legacy conda support (optional):**

```bash
conda env create -f environment.yml
conda activate diive
python -m pip install -e .
```

## Project Structure

```
diive/
├── core/ml/              # Feature engineering, ML base classes
├── pkgs/gapfilling/      # Gap-filling methods (RF, XGB, MDS)
├── pkgs/fluxprocessingchain/  # Multi-level flux processing
├── pkgs/analyses/        # Time series analysis (decomposition, correlation, etc.)
├── pkgs/outlierdetection/     # 6 outlier detection methods
├── pkgs/corrections/     # Data corrections (offsets, flagging)
└── core/plotting/        # 14+ visualization types

examples/                 # Runnable examples by category
tests/                    # Unit tests
```

## Architecture: v0.91.0 (Composed Feature Engineering)

**Key change:** Feature engineering is now standalone via `FeatureEngineer` class.

```python
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS

# Step 1: Create engineered features (8-stage pipeline)
engineer = FeatureEngineer(
    target_col='NEE',
    features_lag=[-1, 1],
    features_rolling=[12, 24],
    features_diff=[1],
    features_ema=[6, 24],
    features_poly_degree=2,
    features_stl=False,
    vectorize_timestamps=True,
)

# Step 2: Apply to data
df_engineered = engineer.fit_transform(df)

# Step 3: Pass pre-engineered data to gap-filling model
model = RandomForestTS(
    input_df=df_engineered,
    target_col='NEE',
    n_estimators=100,
)

# Step 4: Train and fill gaps
model.trainmodel()
model.fillgaps()
gapfilled = model.get_gapfilled_target()
```

**Benefits:**

- Features computed once, reused across multiple models
- Cleaner separation of concerns
- Pre-engineered data can be used with other algorithms
- All feature engineering parameters go to `FeatureEngineer`, not gap-filling classes

## Timestamp Sanitization (TimestampSanitizer)

Comprehensive validation and cleaning of datetime indices through 10 configurable steps.

```python
from diive import TimestampSanitizer

# Clean timestamps with validation
sanitizer = TimestampSanitizer(
    data=df,
    output_middle_timestamp=True,  # Convert to middle-of-period
    validate_naming=True,  # Check TIMESTAMP_END/START/MIDDLE
    convert_to_datetime=True,  # Ensure datetime format
    remove_index_nat=True,  # Remove NaT rows
    sort_ascending=True,  # Sort chronologically
    remove_duplicates=True,  # Remove duplicate timestamps
    regularize=True,  # Fill gaps with NaN rows
    nominal_freq='30min',  # Expected frequency (optional)
    verbose=True
)

clean_df = sanitizer.get()
status = sanitizer.get_status()  # Track what changed
print(f"Removed {status['rows_removed']} rows, added {status['rows_added_by_regularization']}")
print(f"Frequency: {status['inferred_frequency']} (confidence: {status['frequency_confidence']:.0%})")
print(f"  Detection method: {status['frequency_detection_method']}")
if status['frequency_percent_matching']:
    print(f"  Intervals matching: {status['frequency_percent_matching']:.1f}%")
if status['frequency_alternatives']:
    print(f"  Alternatives: {', '.join(status['frequency_alternatives'])}")
```

**Pipeline (10 steps, all optional except monotonicity):**

1. Validate naming convention (TIMESTAMP_END/START/MIDDLE)
2. Convert to datetime format
3. Remove NaT rows
4. Sort ascending
5. Remove duplicates
6. Validate monotonicity (if sorting enabled)
7. Detect frequency with confidence scoring (3 detection methods)
8. Validate frequency against expected (if provided)
9. Regularize gaps to create continuous time series
10. Convert to middle-of-period timestamp

**Status tracking via `get_status()`:**
Returns comprehensive diagnostics including:

- Rows removed/added statistics
- Detected frequency and confidence score (0-1)
- Detection method used (all_methods_agree, full_dataset, timedelta, start_end_chunks)
- % of intervals matching detected frequency
- Alternative frequencies detected by other methods

**Example:** `examples/timeseries/timestamp_sanitizer.py` demonstrates 5 progressively severe data issues (clean →
corrupted), showing robustness and diagnostic output at each level.

## Gap-Filling Methods

| Method                   | Training | Features                  | Accuracy      | Notes                              |
|--------------------------|----------|---------------------------|---------------|-------------------------------------|
| **Random Forest**        | Yes      | 8-stage engineered        | R² 0.60-0.80  | Interpretable, robust to outliers  |
| **XGBoost**              | Yes      | 8-stage engineered        | R² 0.65-0.85  | Better non-linear, smaller models  |
| **MDS**                  | No       | Meteorological similarity | R² 0.40-0.70  | No training, meteorological match  |
| **Linear Interpolation** | No       | None                      | Simple linear | Small gaps only                    |

## Feature Engineering Pipeline (8 stages)

1. **Lag features** (`features_lag`) — Past/future values
2. **Rolling stats** (`features_rolling`) — Mean, median, min, max over windows
3. **Differencing** (`features_diff`) — Rate of change (1st, 2nd order)
4. **EMA** (`features_ema`) — Exponential moving averages
5. **Polynomial** (`features_poly_degree`) — Squared/cubic terms
6. **STL** (`features_stl`) — Trend, seasonal, residual decomposition
7. **Timestamps** (`vectorize_timestamps`) — Year, season, month, hour
8. **Record number** (`add_continuous_record_number`) — Temporal ordering

## Testing

**Run tests:**

```bash
pytest tests/test_gapfilling.py -v              # Gap-filling
pytest tests/test_fluxprocessingchain.py -v     # End-to-end
pytest tests/ -v                                 # All tests
```

**Guidelines:**

- Use flexible assertion ranges (`assertGreater/assertLess`) for SHAP variability
- Validate at API boundaries (user input, external data), not internal contracts
- Don't mock databases in integration tests

## Coding Standards

**Input validation:** Check at system boundaries only

```python
def process_data(df, target_col):
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be empty")
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found")
```

**Error handling:** Let exceptions propagate unless you can recover

```python
# Good: specific, recoverable
try:
    result = operation()
except FileNotFoundError:
    logger.info("Using default fallback")
```

**Comments:** Only for non-obvious WHY, not WHAT

```python
# Good: explains hidden constraint
# Exclude dot columns to avoid circular dependency with gap-filling

# Bad: explains what code does
# Add 1 to result
result = result + 1
```

**No file I/O in examples:** Show API, not export/load

```python
# Good: shows how to use
result = model.fillgaps()

# Bad: includes I/O
result.to_csv('output.csv')  # ← Remove this from examples
```

**Explicit parameters in function calls:** Always list all parameters, even defaults

When calling functions in examples, make parameters explicit with inline comments showing defaults. This:
- Documents what each parameter does
- Shows all available options to users
- Makes examples self-documenting
- Easier to adapt examples to different use cases

```python
# Good: All parameters explicit with comments
scatter = dv.plot_scatter_xy(
    x=df['A'],
    y=df['B'],
    z=None,               # Optional color variable
    nbins=0,              # No binning (0 = off)
    binagg='median'       # Aggregation method (ignored if nbins=0)
)
scatter.plot(
    ax=None,              # Create new figure
    xlabel='X label',
    ylabel='Y label',
    title='Title',
    cmap='viridis',       # Colormap name
    show_colorbar=True    # Display color bar if z provided
)

# Bad: Sparse, doesn't show all options
scatter = dv.plot_scatter_xy(x=df['A'], y=df['B'])
scatter.plot(xlabel='X', ylabel='Y', title='Title')
```

**Rationale:**
- New users see what parameters exist and their defaults
- Examples become quick reference documentation
- Makes it easy to modify examples for other use cases
- Inline comments (not separate docstrings) keep examples readable

## Plotting Class Design Standard

All plotting classes follow a **two-phase design**:

**Phase 1: `__init__()` — Data + Computation ONLY**
- Accept data (Series, DataFrame, arrays)
- Accept computation parameters (nbins, aggregation method, binning strategy, etc.)
- Store only what's needed for data preparation
- Do NOT include: plot destination (`ax`), styling, titles, labels, limits, or colors

**Phase 2: `plot()` — All Styling + Rendering**
- Accept `ax` (plot destination — where to render the figure)
- Accept ALL styling/presentation parameters (title, labels, limits, colors, etc.)
- Can be called multiple times with different styles/axes on same data
- All parameters have sensible defaults (creates new figure if `ax=None`)

**Rationale:**
- Separates concerns (data ≠ presentation)
- Follows matplotlib conventions (create figure → style → display)
- Enables replotting same data with different styles or on different axes
- Cleaner API — simpler object creation, all styling in one place
- Flexible rendering — decide where to plot at plot-time, not object-creation-time

### The `ax` Parameter

**`ax` is a PRESENTATION parameter, NOT a data parameter. Always put it in `plot()` method.**

| Aspect | Details |
|--------|---------|
| **Where** | `plot()` method, as first parameter |
| **Default** | `None` — creates new figure if not provided |
| **Purpose** | Specify which matplotlib axes to plot on |
| **Use case** | Render same data on different axes, subplots, existing figures |

**Why NOT in `__init__()`:**
- ❌ Couples data preparation to plot destination
- ❌ Makes object reuse inflexible (can't plot on different axes)
- ❌ Violates separation of concerns

**Pattern — Plot same data on multiple axes:**

```python
# Create once
scatter = dv.plot_scatter_xy(x=df['A'], y=df['B'])

# Render on different axes with different styling
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
scatter.plot(ax=axes[0], title='Linear scale')
scatter.plot(ax=axes[1], title='Log scale', ylim='auto')
plt.show()
```

**Pattern — Subplot integration:**

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
scatter1 = dv.plot_scatter_xy(x=df['T'], y=df['VPD'])
scatter2 = dv.plot_scatter_xy(x=df['SW'], y=df['T'])
hist = dv.plot_histogram(s=df['NEE'], method='n_bins', n_bins=20)

scatter1.plot(ax=axes[0, 0], title='VPD vs Temp')
scatter2.plot(ax=axes[0, 1], title='Temp vs Radiation')
hist.plot(ax=axes[1, 0], title='NEE Distribution')
# axes[1, 1] left empty or used for legend, text, etc.

plt.tight_layout()
plt.show()
```

**Example: HistogramPlot**

```python
# Phase 1: Data + Computation
hist = dv.plot_histogram(
    s=series,                  # Data
    method='n_bins',           # Computation
    n_bins=20                  # Computation
)

# Phase 2: All styling in plot()
hist.plot(
    xlabel='NEE flux',         # Styling
    title='My Histogram',      # Styling
    highlight_peak=True,       # Styling
    show_zscores=True,         # Styling
    show_info=True             # Styling
)
```

**Example: ScatterXY**

```python
# Phase 1: Data + Computation
scatter = dv.plot_scatter_xy(
    x=df['Tair_f'],            # Data
    y=df['VPD_f'],             # Data
    z=df['Rg_f'],              # Data (optional)
    nbins=10,                  # Computation
    binagg='median'            # Computation
)

# Phase 2: All styling in plot() — including ax (plot destination)
scatter.plot(
    ax=my_axes,                # Where to render (optional)
    xlabel='Temperature (°C)',
    ylabel='VPD (hPa)',
    zlabel='Radiation (W/m²)',
    title='Temperature vs VPD',
    cmap='plasma',
    show_colorbar=True,
    xlim=[0, 30],
    ylim=[0, 5]
)
```

**Checklist for plotting classes:**

- ✓ `__init__()` contains ONLY: data + computation parameters
- ✓ NO rendering or styling parameters in `__init__()` (explicitly exclude: `ax`, `title`, `xlabel`, `ylabel`, `colors`, `limits`, `cmap`, `show_*`, etc.)
- ✓ `ax` parameter in `plot()` method as first parameter, default `None`
- ✓ `ax=None` creates new figure automatically via `pf.create_ax()` or similar
- ✓ All styling parameters moved to `plot()` method with sensible defaults
- ✓ `plot()` has comprehensive docstring listing all options with descriptions
- ✓ `plot()` can be called multiple times with different `ax`/styling
- ✓ Examples updated to pass `ax` and styling to `plot()`, not `__init__()`

**Correct Implementation Pattern:**

```python
class MyPlotter:
    def __init__(self, data1, data2, nbins=10):
        """Data + computation only."""
        self.data1 = data1
        self.data2 = data2
        self.nbins = nbins
        # DO NOT include: ax, title, xlabel, ylabel, colors, etc.

    def plot(self, ax=None, title=None, xlabel=None, ylabel=None, **styling):
        """All styling parameters here, including ax."""
        if not ax:
            fig, ax = pf.create_ax()
        
        # Render with styling
        ax.plot(self.data1, self.data2, ...)
        if title:
            ax.set_title(title)
        # ... more styling
        
        if not existing_ax:
            fig.show()
```

**Usage:**

```python
# Create object once
plotter = MyPlotter(data1, data2)

# Render multiple times on different axes
plotter.plot(ax=axes[0], title='View 1')
plotter.plot(ax=axes[1], title='View 2', xlabel='Custom X')
plotter.plot()  # Creates new figure
```

## Development Workflow

**[CRITICAL] NEVER COMMIT CHANGES.** User commits and stages changes themselves.

**[CRITICAL] NEVER ADD CLAUDE AS CO-AUTHOR IN COMMIT MESSAGES.** When writing commit messages, do not include:
```
Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
```
Only the user is the author.

**Do NOT run `examples/run_all_examples.py` proactively.** User will run this themselves to validate the full suite.
Only run individual example files for validation during development (e.g., `uv run python examples/pkgs/analysis/analysis_correlation.py`).

**Do NOT perform git operations** (git add/commit/push). User commits and stages changes themselves.

**Do NOT run pre-commit hooks** (skip with --no-verify) unless explicitly requested.

## FluxProcessingChain (Swiss FluxNet Multi-Level Workflow)

Complete eddy covariance flux post-processing
following [FLUXNET conventions](https://fluxnet.org/data/fluxnet2015-dataset/fullset-data-product/) and Swiss FluxNet
standards. Data flows sequentially through 5 levels, each building on the previous.

**Typical data retention per level (example: NEE):**

- L2 → L3.1: ~100% (only missing flux values)
- L3.1 → L3.2: ~70-90% (outlier removal)
- L3.2 → L3.3 daytime: ~98%+ (strong turbulence)
- L3.2 → L3.3 nighttime: ~50-70% (weak turbulence filtered)

### Level-2: Quality Flag Expansion

**Class:** `FluxQualityFlagsEddyPro` | Applies 7 configurable quality tests:

1. **SSITC** — Steady-State & Integrated Turbulent Characteristics (flow stationarity)
2. **Gas Completeness** — Base variable availability (e.g., CO2 for FC flux)
3. **Spectral Correction Factor** — Spectral distortion detection
4. **Signal Strength** — IRGA diagnostic quality (AGC or signal_strength)
    - Threshold method: `'discard above'` (closed-path, high=bad) or `'discard below'` (open-path, low=bad)
5. **VM97 Raw Data Tests** — 8 individual EddyPro tests: spikes, amplitude, dropout, absolute limits, skew/kurtosis,
   discontinuities
6. **Angle of Attack** — Wind angle relative to anemometer (disabled by default, ICOS standard)
7. **Steadiness of Horizontal Wind** — Wind stationarity (disabled by default, ICOS standard)

**Key parameter:** `daytime_accept_qcf_below=2` accepts good + medium quality daytime data.

### Level-3.1: Storage Correction

**Class:** `FluxStorageCorrectionSinglePointEddyPro` | Adds single-point profile gas storage term to flux.

- Applies: `{flux}_L3.1 = {flux} + {storage_term}`
- Missing storage terms gap-filled using expanding window rolling mean
- **Critical:** Without storage correction, lose ~2-3% systematic error
- Terminology change: `FC` (CO2 flux) → `NEE` (Net Ecosystem Exchange)

### Level-3.2: Outlier Detection (Sequential Chain)

**Class:** `StepwiseOutlierDetection` | Chains multiple detection methods sequentially.

**Workflow pattern:**

```python
fpc.level32_stepwise_outlier_detection()  # Initialize

fpc.level32_flag_outliers_hampel_dtnt_test(n_sigma=5.5, ...)
fpc.level32_addflag()  # Accept results before next test

fpc.level32_flag_outliers_zscore_test(thres_zscore=4, ...)
fpc.level32_addflag()  # Each test operates on previous output
```

**Key design:** Each subsequent test operates on data already filtered by previous test. Order matters; always inspect
preview plots before `addflag()`.

Available methods: Hampel, z-score (global/rolling/increments), absolute limits, local SD, Local Outlier Factor,
TrimLow, manual removal.

### Level-3.3: USTAR Filtering

**Class:** `FlagMultipleConstantUstarThresholds` | Removes low-turbulence nighttime data.

**Critical FLUXNET standard:** USTAR filtering applied **ONLY to CO2/CH4/N2O**. NOT applied to H (sensible heat) or LE (
latent heat) because advective fluxes don't proportionally affect energy fluxes at night.

**Multiple percentile scenarios for uncertainty:**

```python
fpc.level33_constant_ustar(
    thresholds=[0.0529, 0.0699, 0.0929],
    threshold_labels=['CUT_16', 'CUT_50', 'CUT_84']  # 16th, 50th, 84th percentiles
)
```

**Quality threshold strategy:**

```python
nighttime_accept_qcf_below = 1  # Strict: only best quality (QCF=0)
# Nighttime signal is weak; stricter prevents error cascade
```

### Level-4.1: Gap-Filling

**Methods:** `level41_longterm_random_forest()`, `level41_longterm_xgboost()`, `level41_mds()`

**Common 8-stage feature engineering pipeline (identical for RF and XGBoost):**

1. Lag features (e.g., `[-2, -1]` = prior 30-60 min for 30-min data)
2. Rolling stats (e.g., `[2, 4, 12, 24, 48]` = 1-24 hr windows, stats: median/min/max/std/q25/q75)
3. Differencing (1st and 2nd order for rate/acceleration of change)
4. EMA (e.g., `[6, 12, 24, 48]` = 3-24 hr exponential moving averages)
5. Polynomial (e.g., `degree=2` for light saturation curves)
6. STL (trend/seasonal/residual, usually disabled; expensive)
7. **Timestamps** (`vectorize_timestamps=True`) — ESSENTIAL: creates ~19 features (year, season, hour, etc.) for
   diurnal/seasonal cycles
8. Record number (`add_continuous_record_number=True`) — long-term drift detection

**Method comparison:**

| Method        | R²        | Training | Notes                                      |
|---------------|-----------|----------|---------------------------------------------|
| Random Forest | 0.60-0.80 | Yes      | Interpretable, robust to outliers          |
| XGBoost       | 0.65-0.85 | Yes      | Better non-linear patterns, smaller models |
| MDS           | 0.40-0.70 | No       | Meteorological similarity, no overfitting  |

**MDS critical parameters (exact units required):**

```python
fpc.level41_mds(
    swin="SW_IN_POT",  # MUST be W/m² (not µmol)
    ta="TA_EP",  # MUST be °C (not K)
    vpd="VPD_EP",  # MUST be hPa (not Pa)
    swin_tol=[20, 50],  # 2-level radiation tolerance
    ta_tol=2.5,  # Temperature tolerance
    vpd_tol=0.5,  # VPD tolerance
    avg_min_n_vals=5  # Min matching records to average
)
```

Wrong units silently produce garbage results.

**Complete workflow example:**

```python
from diive.pkgs.flux.fluxprocessingchain import FluxProcessingChain

fpc = FluxProcessingChain(
    df=maindf,
    fluxcol='FC',
    site_lat=47.478,
    site_lon=8.364,
    utc_offset=1,
    nighttime_threshold=20,
    daytime_accept_qcf_below=2,
    nighttimetime_accept_qcf_below=1
)

# L2: Quality flags
fpc.level2_quality_flag_expansion(
    signal_strength={'apply': True, 'signal_strength_col': 'CUSTOM_AGC_MEAN',
                     'method': 'discard above', 'threshold': 0.85},
    raw_data_screening_vm97={'apply': True, 'spikes': True, 'amplitude': False, ...}
)
fpc.finalize_level2()

# L3.1: Storage correction
fpc.level31_storage_correction(gapfill_storage_term=False)
fpc.finalize_level31()

# L3.2: Outlier removal (sequential chain)
fpc.level32_stepwise_outlier_detection()
fpc.level32_flag_outliers_hampel_dtnt_test(window_length=48 * 13, n_sigma_dt=5.5, n_sigma_nt=5.5)
fpc.level32_addflag()
fpc.finalize_level32()

# L3.3: USTAR filtering
fpc.level33_constant_ustar(thresholds=[0.09], threshold_labels=['CUT_50'])
fpc.finalize_level33()

# L4.1: Gap-fill with XGBoost
fpc.level41_longterm_xgboost(
    features=['TA_EP', 'SW_IN_POT', 'VPD_EP'],
    features_lag=[-2, -1],
    features_rolling=[2, 4, 12, 24, 48],
    features_rolling_stats=['median', 'min', 'max', 'std', 'q25', 'q75'],
    features_diff=[1, 2],
    features_ema=[6, 12, 24, 48],
    features_poly_degree=2,
    vectorize_timestamps=True,
    add_continuous_record_number=True,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    early_stopping_rounds=30,
    random_state=42
)

# Extract results
results = fpc.get_data()
model = fpc.level41['long_term_xgboost']['CUT_50']
gapfilled_flux = model.gapfilled_
scores = model.scores_  # R², MAE, RMSE on test data
```

**Critical pitfalls to avoid:**

1. **Wrong USTAR threshold** — Filters too much/too little nighttime data. Must use site-specific calibrated values from
   FLUXNET or REddyProc.
2. **Wrong MDS units** — Radiation not W/m², temperature not °C, VPD not hPa → garbage results (silent failure).
3. **Missing pre-gapfilled features** — If TA/SW_IN/VPD all missing entirely, RF features have no signal.
4. **Sequential outlier test order** — Early aggressive test removes data needed by later tests. Always preview each
   test before `addflag()`.
5. **Ignoring nighttime quality** — Even filtered nighttime data can be unreliable due to weak signals; consider
   separate day/night analysis for annual budgets.
6. **Neglecting method divergence** — Compare RF and MDS results; high divergence signals model uncertainty in that
   period.

## Quality Control (QCF - Overall Quality Flag)

**FlagQCF** · class: `FlagQCF` ([example](examples/qaqc/qcf.py))

Combines multiple individual quality test flags into a single overall quality indicator (QCF). Each record receives:

- **QCF=0:** Good quality (all tests pass)
- **QCF=1:** Marginal quality (1-3 soft warnings, no hard fails)
- **QCF=2:** Poor quality (>3 soft warnings or >=2 hard fails)

Features:

- Automatic identification of `FLAG_*_TEST` columns
- Hard flags (value 2) vs soft flags (value 1) weighted differently
- Optional daytime/nighttime separation
- USTAR filtering scenario support
- Comprehensive reports: flag statistics, sequential impact analysis
- 4-panel heatmap visualization (original, QC series, flag sums, QCF flag)

```python
qcf = FlagQCF(
    df=data,
    target_col='NEE',
    swinpot_col='SW_IN_POT',  # Optional: enables day/night separation
    idstr='_L41'
)
qcf.calculate(daytime_accept_qcf_below=2)
quality_series = qcf.filteredseries  # NaN for QCF=2
highest_quality = qcf.filteredseries_hq  # NaN for QCF>0
qcf.report_qcf_series()  # Comprehensive summary
qcf.report_qcf_flags()  # Individual test statistics
qcf.report_qcf_evolution()  # Sequential impact analysis
qcf.showplot_qcf_heatmaps()  # Visualization
```

## EddyPro Quality Flag Functions

**Module:** `diive.pkgs.preprocessing.qaqc.eddyproflags`

Extracts and converts quality flags from EddyPro output files. EddyPro uses different flag formats than DIIVE's
standard (0=good, 1=warning, 2=bad):

- Some tests output binary flags (1=fail) requiring conversion
- Multi-digit codes encode multiple tests (e.g., VM97 8-digit integer)
- Signal strength is continuous and requires threshold-based classification

**7 Functions:**

1. **`flag_signal_strength_eddypro_test()`** — Signal strength threshold classification
    - Continuous values (0-100 or 0-1 range) thresholded to DIIVE flags
    - Good for open-path IRGA and sonic anemometer data

2. **`flag_steadiness_horizontal_wind_eddypro_test()`** — Wind steadiness test
    - Converts EddyPro 1=bad to DIIVE 2=bad
    - Detects non-stationary wind conditions

3. **`flag_angle_of_attack_eddypro_test()`** — Angle of attack assessment
    - Wind vector relative to anemometer orientation
    - Converts 1=bad to 2=bad

4. **`flags_vm97_eddypro_fluxnetfile_tests()`** — Vickers & Mahrt (1997) raw data tests
    - Extracts 8 individual tests from single 8-digit integer code
    - Tests: spikes, amplitude, dropout, absolute limits, skewness/kurtosis (hard + soft), discontinuities (hard + soft)
    - Each test returns 0=pass, 1=warning (soft), 2=fail (hard)

5. **`flag_fluxbasevar_completeness_eddypro_test()`** — Base variable data completeness
    - Completeness percentage → DIIVE flags
    - Example: CO2 completeness for FC flux calculation

6. **`flag_ssitc_eddypro_test()`** — Steady State and Integral Turbulence Characteristics
    - Evaluates measurement stationarity conditions
    - Converts 1=bad to 2=bad

**Helper Consolidation Pattern:**

Multi-digit flag extraction consolidated into `_extract_and_convert_flag_from_multidigit()`:

```python
def _extract_and_convert_flag_from_multidigit(df, column_name, position, is_hard_flag=True):
    """Extract single flag from multi-digit integer and convert to DIIVE format."""
    flag = df[column_name].copy()
    flag = flag.apply(pd.to_numeric, errors='coerce').astype(float)
    flag = flag.fillna(899999999)  # 9 = missing flag
    flag = flag.astype(str)
    flag = flag.str[int(position)]  # Extract digit at position
    flag = flag.apply(pd.to_numeric, errors='coerce')  # Handle non-numeric chars like '.'
    flag = flag.replace(9, np.nan)  # Restore missing flags
    if is_hard_flag:
        flag = flag.replace(1, 2)  # Convert hard flag 1 to DIIVE format 2
    return flag
```

Used by: `flag_steadiness_horizontal_wind_eddypro_test()`, `flag_angle_of_attack_eddypro_test()`,
`flags_vm97_eddypro_fluxnetfile_tests()`

**Examples:** 6 examples in `examples/qaqc/eddyproflags.py`

## Outlier Detection Methods

10 built-in methods available (11 classes with z-score variants):

1. **AbsoluteLimits** — Min/max threshold
2. **Hampel** — Robust spike detection (MAD-based)
3. **LocalSD** — Local standard deviation (adaptive)
4. **zScore** — Global z-score threshold
5. **zScoreDaytimeNighttime** — Z-score with separate day/night thresholds
6. **zScoreRolling** — Rolling z-score (adaptive threshold)
7. **zScoreIncrements** — Abrupt change detection
8. **LocalOutlierFactor** — Density-based anomalies
9. **ManualRemoval** — Explicit data removal
10. **TrimLow** — Symmetric removal (trimmed mean approach)

**Step-wise Orchestration:** Chain multiple detection methods sequentially via `StepwiseOutlierDetection`. Each method
operates on the previous result, progressively filtering outliers. Useful for multi-stage QA/QC workflows.

```python
from diive.pkgs.preprocessing.outlierdetection import StepwiseOutlierDetection

detector = StepwiseOutlierDetection(dfin=df, col='NEE', site_lat=46.8, site_lon=8.6, utc_offset=1)

# Chain: aggressive first, then refine
detector.flag_outliers_hampel_dtnt_test(n_sigma=5.5)
detector.addflag()

detector.flag_outliers_localsd_test(n_sd=[3.5, 3.5], winsize=[24, 24], separate_daytime_nighttime=True)
detector.addflag()

detector.flag_outliers_zscore_test(thres_zscore=4)
detector.addflag()

# View results
print(detector.series_hires_cleaned)  # Cleaned series
print(detector.flags)  # All flags from all steps
```

See `examples/outlierdetection/stepwise.py` for complete example (6 methods with full parameters).

## Examples

Located in `examples/` organized to mirror `diive/pkgs` and `diive/core` structure.

**Structure:**

```
examples/
├── core/               # System-level utilities
│   ├── visualization/  # Plotting (9 examples)
│   └── times/          # Timestamps (1 example)
└── pkgs/               # Domain packages
    ├── analysis/       # Time series analysis (9 examples)
    ├── features/       # Variable creation (8 examples)
    ├── flux/           # Flux processing (11 examples)
    ├── gapfilling/     # Gap-filling methods (6 examples)
    └── preprocessing/  # QC & corrections (16 examples)
```

**Documentation:**

- **`examples/CATALOG.md`** — Find examples by use case (workflows, methods, difficulty levels)
- **`examples/EXAMPLE_DATASET.md`** — Complete dataset documentation (37 variables, gaps, availability)
- **`examples/README.md`** — Quick start and structure overview
- **Category READMEs** — One per folder with descriptions and usage

**Running Examples:**

- Run all: `uv run python examples/run_all_examples.py` (parallel, 8 workers)
- Run category: `uv run python examples/pkgs/gapfilling/gapfill_randomforest_ts.py`
- Coverage: 62 examples across 18 organized folders

**Maintaining Examples (IMPORTANT):**
When adding/modifying examples:

1. Update **category README.md** with file description
2. Update **examples/CATALOG.md** with use case/workflow info
3. Update **examples/run_all_examples.py** with file path
4. Update **CHANGELOG.md** to note changes
5. Update **source code docstrings** (see checklist below)
6. Keep **examples/EXAMPLE_DATASET.md** in sync if data changes

The examples catalog and dataset docs are user-facing—keep them accurate and in sync.

### Sphinx Gallery Format (v0.91.0+)

Examples are now structured as **executable Python scripts** designed for Sphinx Gallery documentation generation:

**File naming:**
- `correction_*.py` — Data correction methods (e.g., `correction_relativehumidity_offset.py`)
- `outlier_*.py` — Outlier detection methods (e.g., `outlier_hampel.py`)
- `qc_*.py` — Quality control methods (e.g., `qc_overall_flag.py`)
- `gapfill_*.py` — Gap-filling methods (e.g., `gapfill_randomforest_ts.py`)
- `io_*.py` — File I/O operations (e.g., `io_extract.py`)
- `fluxprocessingchain.py` — Complete multi-level workflow

**Structure within files:**
- Top docstring with title and overview
- `# %%` separators between logical sections (becomes separate cells when rendered)
- Explanatory text blocks before code sections (non-executed comments)
- Code that executes top-to-bottom
- Inline prints and outputs visible to readers

**Benefits:**
- ✓ Pure Python — version control friendly (no notebook JSON)
- ✓ Readable diffs — changes to code are clear
- ✓ Auto-execution — Sphinx Gallery runs scripts, captures plots/output
- ✓ Professional docs — generates beautiful HTML with code + output side-by-side
- ✓ Reproducible — no hidden state, no kernel issues

**Example structure:**
```python
"""
=================
Example Title
=================

One-line summary. Detailed explanation of what this example demonstrates.
"""

# %%
# Section 1: Load data
# ^^^^^^^^^^^^^^^^^^^^
# Explanatory text before code

import diive as dv
data = dv.load_example_data()

# %%
# Section 2: Process
# ^^^^^^^^^^^^^^^^^^
# More explanation

results = dv.process(data)
print(results)
```

### Recent Example Updates (v0.91.0+)

**Flux Processing Chain (`pkgs/flux/fluxprocessingchain/`)**
- Restructured as Sphinx Gallery example (`fluxprocessingchain.py`)
- Demonstrates all 5 processing levels (L2-L4.1) with real output
- Shows both Random Forest and XGBoost gap-filling methods
- Shared `GAPFILLING_PARAMS` dict reduces code duplication
- Uses 1 month of data for faster demonstration

**Data Offset Corrections (`pkgs/preprocessing/corrections/`)**
- Split from monolithic `offsetcorrection.py` into 4 focused examples:
  - `correction_relativehumidity_offset.py` — RH saturation fix
  - `correction_radiation_offset.py` — Radiation nighttime offset
  - `correction_measurement_offset_replicate.py` — Instrument offset detection
  - `correction_winddir_offset.py` — Wind direction calibration
- Each as standalone Sphinx Gallery example
- Domain-specific `correction_` prefix (not `plot_`) reflects actual content

**Outlier Detection & QA/QC (`pkgs/preprocessing/outlierdetection/` and `qaqc/`)**
- 9 outlier detection methods converted to Sphinx Gallery format with consistent `outlier_*` naming
- 2 QA/QC examples (`qc_overall_flag.py`, `qc_eddypro_flags.py`)
- Organized by complexity level (beginner, intermediate, advanced)

**Gap-Filling Methods (`pkgs/gapfilling/`)**
- All 6 gap-filling examples converted to Sphinx Gallery format with consistent `gapfill_*` naming:
  - `gapfill_randomforest_ts.py` — Random Forest with 3 sub-workflows
  - `gapfill_xgboost_ts.py` — XGBoost with hyperparameter optimization
  - `gapfill_mds.py` — Meteorological Data Similarity
  - `gapfill_mds_comparison.py` — Original vs. optimized MDS performance
  - `gapfill_interpolate.py` — Linear interpolation (conservative & generous)
  - `gapfill_comparison.py` — Benchmark all three methods side-by-side

**File I/O (`pkgs/io/`)**
- Binary value extraction example renamed to `io_extract.py` with Sphinx Gallery format

### Example File Structure & Documentation Checklist

**Every example file must follow this structure:**

#### 1. File Naming Convention
- Prefix by domain: `correction_`, `outlier_`, `qc_`, `gapfill_`, `io_`, `feature_`, `analysis_`, etc.
- Suffix with function/method name: `correction_relativehumidity_offset.py`, `gapfill_randomforest.py`
- **ONE function/workflow per file** — no combined functions in single files
- Lowercase with underscores (snake_case)

#### 2. File Content Structure

**Top-level docstring (reStructuredText format):**
```python
"""
=================
Descriptive Title
=================

One-line summary. Clear explanation of what this example demonstrates and when to use it.

Best for: [specific use case or condition]
"""
```

**Sections within file:**
- Use `# %%` separators between logical sections (becomes cells in Sphinx Gallery)
- Each section starts with a header comment block:
  ```python
  # %%
  # Section Title
  # ^^^^^^^^^^^^^^
  #
  # Explanatory narrative text (non-executed comments describing what follows)
  ```
- Code follows immediately after narrative
- Include `print()` statements to show outputs to readers
- **No file I/O** (no `.to_csv()`, `.to_parquet()`, etc.) — show API only

**Example structure:**
```python
"""
============================
Gap-Filling with Random Forest
============================

Demonstrates training and applying Random Forest for time series gap-filling.

Best for: Medium-sized gaps with diurnal/seasonal patterns.
"""

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

import diive as dv
df = dv.load_exampledata_parquet()

# %%
# Train model
# ^^^^^^^^^^^
# Create feature engineering pipeline...

from diive.core.ml.feature_engineer import FeatureEngineer
engineer = FeatureEngineer(...)
```

#### 3. Source Code Docstring Updates

Update the function/class docstring in the source code to reference the example:

**Add to docstring sections (in order):**
1. **Description** — What the function does
2. **Parameters** — Argument details (Args:)
3. **Returns** — Return value details
4. **See Also** — Related functions
   ```python
   See Also:
       other_function : Brief description of related function.
   ```
5. **Example** — Reference the example file
   ```python
   Example:
       See `examples/pkgs/gapfilling/gapfill_randomforest.py` for complete examples.
   ```

#### 4. Documentation Files to Update (7-point checklist)

For each new/modified example, update:

**A. Category README** (`examples/pkgs/{category}/README.md`)
   - Add file under appropriate subsection (e.g., "Machine Learning Methods")
   - Format: `- **filename.py** — One-line description`
   - Example:
     ```markdown
     ### Machine Learning Methods
     - **gapfill_randomforest.py** — Random Forest gap-filling with feature engineering
     - **gapfill_xgboost.py** — XGBoost gradient boosting gap-filling
     ```

**B. Running Examples section** (in same README)
   - Add bash command for single example:
     ```bash
     uv run python examples/pkgs/gapfilling/gapfill_randomforest.py
     ```

**C. examples/run_all_examples.py**
   - Add file path to `EXAMPLE_FILES` list in correct section:
     ```python
     # PKGS: Gap-filling
     'pkgs/gapfilling/gapfill_randomforest.py',
     'pkgs/gapfilling/gapfill_xgboost.py',
     ```

**D. examples/README.md**
   - Update total example count in header
   - Update example folder structure if new category
   - Update coverage table with file count per category

**E. examples/CATALOG.md**
   - Add file reference under appropriate workflow/use case
   - Link to category README for more details

**F. CHANGELOG.md**
   - Note new/updated examples in version entry
   - Example: `- New gap-filling examples: gapfill_randomforest.py, gapfill_xgboost.py`

**G. Source Code Docstrings** (in `diive/pkgs/{module}/{file}.py`)
   - Update "Example" section with new example path
   - Add "See Also" section linking to related functions
   - Format: `` `examples/pkgs/gapfilling/gapfill_randomforest.py` ``

#### 5. Common Mistakes to Avoid

- ❌ **Multiple functions in one file** — Create separate files for each function
- ❌ **Forgetting to update source code docstrings** — Always update Example and See Also sections
- ❌ **Including file I/O** — Don't add `.to_csv()` or `.to_parquet()` to examples
- ❌ **Updating only examples/, forgetting source docstrings** — Both must be synchronized
- ❌ **Inconsistent file naming** — Must use correct domain prefix (gapfill_, io_, etc.)
- ❌ **Updating examples/run_all_examples.py but not README** — Update all 7 documentation files

#### 6. Verification Checklist

Before considering an example complete:

```
□ File created with correct naming: {prefix}_{function}.py
□ File follows Sphinx Gallery structure (docstring, # %%, narrative, code)
□ No file I/O operations included
□ Examples run without errors: uv run python examples/pkgs/{category}/{file}.py
□ Category README.md updated with file description and running example
□ examples/run_all_examples.py updated with file path
□ examples/README.md example count and structure updated
□ examples/CATALOG.md updated with workflow references
□ CHANGELOG.md updated with new example entry
□ Source code docstring "Example" section updated with path
□ Source code docstring "See Also" section added if related functions exist
□ All documentation cross-references are consistent (no broken links)
```

### Subdirectory Organization Strategy

**Key Principle:** Example subdirectories must mirror the source code organization in `diive/pkgs/`.

#### Why Mirror Source Structure?

- **Consistency:** Users familiar with the package structure can immediately find related examples
- **Discoverability:** Grouping examples by function (hires, lowres, corrections) makes sense both in source and docs
- **Maintainability:** When source code is refactored, the example organization stays aligned
- **Scalability:** New examples naturally fit into existing categorical folders

#### Example: Flux Processing (`examples/pkgs/flux/`)

Source code structure (`diive/pkgs/flux/`):
```
flux/
├── fluxprocessingchain/    # Multi-level processing workflow
├── hires/                  # High-resolution analysis (10Hz)
├── lowres/                 # Low-resolution processing (30min)
└── io/                     # File I/O (if applicable)
```

Examples structure MUST match:
```
examples/pkgs/flux/
├── fluxprocessingchain/
│   └── fluxprocessingchain.py        (no prefix needed, already in subfolder)
├── hires/
│   ├── flux_lag.py
│   ├── flux_windrotation.py
│   └── flux_fluxdetectionlimit.py
└── lowres/
    ├── flux_common.py
    ├── flux_hqflux.py
    ├── flux_selfheating.py
    ├── flux_uncertainty.py
    └── flux_ustar_mp_detection.py
```

#### Naming Convention with Subdirectories

- **No prefix needed for top-level files** in single-purpose folders (e.g., `correction_*.py` at `examples/pkgs/preprocessing/corrections/`)
- **With subdirectories:** Use prefix to clarify domain (e.g., `flux_lag.py` in `hires/` subdirectory)
- **Exception:** Root-level module examples use prefix (e.g., `flux_common.py` instead of just `common.py`)

**Formula:**
```
examples/pkgs/{category}/{optional_subfolder}/{domain_prefix}_{function}.py
                                                ↑ Add prefix when in subfolder with other domains
```

#### Updating Documentation When Adding Subdirectory Examples

When examples are organized in subdirectories, update these files:

**1. Category README.md**
```markdown
### Subfolder Name
- **subdirectory/flux_lag.py** — Time lag detection description
- **subdirectory/flux_windrotation.py** — Wind rotation description
```

**2. examples/run_all_examples.py**
```python
# PKGS: Category - Subfolder
'pkgs/category/subfolder/flux_lag.py',
'pkgs/category/subfolder/flux_windrotation.py',
```

**3. examples/README.md structure**
```
├── category/              # Category (N examples)
│   ├── subfolder1/        # Subfolder A (3 examples)
│   │   ├── example1.py
│   │   ├── example2.py
│   │   └── example3.py
│   └── subfolder2/        # Subfolder B (2 examples)
│       ├── example4.py
│       └── example5.py
```

#### Complete Subdirectory Example Workflow

**Task:** Convert 10 flux examples organized into 3 source subdirectories

**Process:**

1. **Create subdirectory structure** in examples/ matching source
   ```bash
   mkdir -p examples/pkgs/flux/{fluxprocessingchain,hires,lowres}
   ```

2. **Create Sphinx Gallery files** with domain prefix (flux_) for clarity
   - `flux_lag.py` in `hires/`
   - `flux_common.py` in `lowres/`

3. **Update run_all_examples.py** with full paths including subdirectories
   ```python
   'pkgs/flux/hires/flux_lag.py',
   'pkgs/flux/lowres/flux_common.py',
   ```

4. **Update category README.md** with subsections showing subdirectories
   ```markdown
   ### High-Resolution Analysis
   - **hires/flux_lag.py** — Time lag detection
   ```

5. **Verify examples work:**
   ```bash
   uv run python examples/pkgs/flux/hires/flux_lag.py
   ```

6. **Validate syntax** for all new files
   ```bash
   cd examples/pkgs/flux && find . -name "*.py" -exec python3 -m py_compile {} \;
   ```

## Common Workflows

### Adding New Feature Engineering Stage

1. Add parameter to `FeatureEngineer.__init__()` (default None)
2. Implement `_stagename_features()` method
3. Call from `_create_features()` orchestrator
4. Use naming: `.{col}_TYPE{detail}` (e.g., `.Tair_f_POL2`)
5. Update docstring and example

### Adding Gap-Filling Method to FluxProcessingChain

1. Create `level41_newmethod()` with all 24 feature parameters
2. Create `FeatureEngineer`, apply to data
3. Create and train gap-filling model
4. Store results in `self._level41['new_method'][ustar_scenario]`
5. Update tests and examples

### Debugging SHAP Importance Issues

1. Check `.RANDOM` baseline feature included
2. Verify threshold: `random_mean + k * random_sd`
3. Check feature counts before/after reduction
4. Inspect `model_.feature_importances_traintest_`

## Known Issues & Workarounds

| Issue                                     | Workaround                                    |
|-------------------------------------------|-----------------------------------------------|
| SHAP importance fluctuates (±5-10%)       | Use flexible assertion ranges in tests        |
| XGBoost base_score in scientific notation | Monkey-patched in `MlRegressorGapFillingBase` |
| Feature reduction too strict              | Reduce `shap_threshold_factor` (default 0.5)  |

## Claude Code Config

**.claude/settings.json:**

```json
{
  "permissions": {
    "default": "deny",
    "allow": [
      "bash:uv",
      "bash:pytest",
      "read:**",
      "edit:diive/**"
    ]
  },
  "env": {
    "PYTHONPATH": "${workspaceRoot}"
  }
}
```

## Quick Reference

**Install dependencies:** `uv sync`

**Run tests:** `uv run pytest tests/ -v`

**Run example:** `uv run python examples/qaqc/qcf.py`

**View class:**
`uv run python -c "from diive.core.ml.common import MlRegressorGapFillingBase; help(MlRegressorGapFillingBase)"`

**Check version:** `uv run python -c "import diive; print(diive.__version__)"`

**Add dependency:** `uv add package_name`

---

**Last Updated:** 2026-05-09  
**Version:** v0.91.0+  
**Package Manager:** `uv` (uv sync, uv run python ...)
**Reference:** See `notebooks/flux/FluxProcessingChain.ipynb` for detailed workflow explanations and visualizations

For detailed implementation history, see `CHANGELOG.md`.
