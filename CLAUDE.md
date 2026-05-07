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

## Gap-Filling Methods

| Method | Training | Features | Speed | Accuracy | Notes |
|--------|----------|----------|-------|----------|-------|
| **Random Forest** | Yes | 8-stage engineered | ~3-8 min/year | R² 0.60-0.80 | Interpretable |
| **XGBoost** | Yes | 8-stage engineered | ~2-5 min/year | R² 0.65-0.85 | Gradient boosting |
| **MDS** | No | Meteorological similarity | Fast | R² 0.40-0.70 | No training required |
| **Linear Interpolation** | No | None | Very fast | Simple linear | Small gaps only |

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
pytest tests/test_gapfilling.py -v              # Gap-filling (fast)
pytest tests/test_fluxprocessingchain.py -v     # End-to-end
pytest tests/ -v                                 # All tests
```

**Expected times:**
- Gap-filling tests: ~3-5 sec
- Flux processing chain: ~20-25 sec

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

## FluxProcessingChain (Multi-Level Workflow)

Complete flux processing with quality control, storage correction, USTAR filtering, and gap-filling.

```python
from diive.pkgs.fluxprocessingchain import FluxProcessingChain

fpc = FluxProcessingChain(df, site_lat=47.286, site_lon=7.734)

# Level 2: Quality flags
fpc.level2_qualityflags(cols=['FC', 'LE', 'H'])

# Level 3.1: Storage correction
fpc.level31_storagecorrection(...)

# Level 3.2-3.3: USTAR filtering
fpc.level33_ustarfiltering(...)

# Level 4.1: Gap-filling
fpc.level41_longterm_random_forest(
    features=['TA', 'SW_IN', 'VPD'],
    features_lag=[-1, 1],
    features_rolling=[12, 24],
    # ... other feature parameters
    n_estimators=100,
)

# Access results
filled_flux = fpc.level41['long_term_random_forest']['CUT_50']
```

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
quality_series = qcf.filteredseries          # NaN for QCF=2
highest_quality = qcf.filteredseries_hq      # NaN for QCF>0
qcf.report_qcf_series()                      # Comprehensive summary
qcf.report_qcf_flags()                       # Individual test statistics
qcf.report_qcf_evolution()                   # Sequential impact analysis
qcf.showplot_qcf_heatmaps()                  # Visualization
```

## EddyPro Quality Flag Functions

**Module:** `diive.pkgs.qaqc.eddyproflags`

Extracts and converts quality flags from EddyPro output files. EddyPro uses different flag formats than DIIVE's standard (0=good, 1=warning, 2=bad):
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

Used by: `flag_steadiness_horizontal_wind_eddypro_test()`, `flag_angle_of_attack_eddypro_test()`, `flags_vm97_eddypro_fluxnetfile_tests()`

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

**Step-wise Orchestration:** Chain multiple detection methods sequentially via `StepwiseOutlierDetection`. Each method operates on the previous result, progressively filtering outliers. Useful for multi-stage QA/QC workflows.

```python
from diive.pkgs.outlierdetection.stepwiseoutlierdetection import StepwiseOutlierDetection

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

Located in `examples/` organized by topic. Each is a self-contained, runnable Python file.

**Structure:** `examples/{category}/{module}.py` with 1-4 examples per file

**Run all:** `uv run python examples/run_all_examples.py` (parallel, ~2.7x faster)

**Run individual:** `uv run python examples/visualization/heatmap_datetime.py`

**Coverage:** 107 examples across 52 files
- Visualization: 22, Analysis: 8, Data Processing: 56 (Binary 2, Corrections 7, QAQC 7, Outlierdetection 17, Variable creation 23)
- Eddy Covariance: 11, Time Series: 2, Fits: 1, Gap-filling: 9

See `examples/README.md` for full catalog.

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

| Issue | Workaround |
|-------|-----------|
| SHAP importance fluctuates (±5-10%) | Use flexible assertion ranges in tests |
| XGBoost base_score in scientific notation | Monkey-patched in `MlRegressorGapFillingBase` |
| Feature reduction too strict | Reduce `shap_threshold_factor` (default 0.5) |

## Claude Code Config

**.claude/settings.json:**
```json
{
  "permissions": {
    "default": "deny",
    "allow": ["bash:uv", "bash:pytest", "read:**", "edit:diive/**"]
  },
  "env": {"PYTHONPATH": "${workspaceRoot}"}
}
```

## Quick Reference

**Install dependencies:** `uv sync`

**Run tests:** `uv run pytest tests/ -v`

**Run example:** `uv run python examples/qaqc/qcf.py`

**View class:** `uv run python -c "from diive.core.ml.common import MlRegressorGapFillingBase; help(MlRegressorGapFillingBase)"`

**Check version:** `uv run python -c "import diive; print(diive.__version__)"`

**Add dependency:** `uv add package_name`

---

**Last Updated:** 2026-05-07  
**Version:** v0.91.0+  
**Package Manager:** `uv` (uv sync, uv run python ...)

For detailed implementation history, see `CHANGELOG.md`.
