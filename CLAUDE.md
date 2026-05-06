# CLAUDE.md - DIIVE Development Guide

Quick reference for DIIVE development. For detailed version history, see `CHANGELOG.md`.

## Quick Start

**Setup (first time):**
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

**Python:** 3.11.x (exact)

**Conda:** `C:\Users\nopan\miniconda3\envs\diive` (Windows)

**Install:** `conda activate diive && python -m pip install -e .`

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

## Examples

Located in `examples/` organized by topic. Each is a self-contained, runnable Python file.

**Structure:** `examples/{category}/{module}.py` with 1-4 examples per file

**Run all:** `python examples/run_all_examples.py` (parallel, ~2.7x faster)

**Run individual:** `python examples/visualization/heatmap_datetime.py`

See `examples/README.md` for full catalog (94+ examples across 50 files).

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
    "allow": ["bash:conda", "bash:pytest", "read:**", "edit:diive/**"]
  },
  "env": {"CONDA_ENV": "diive", "PYTHONPATH": "${workspaceRoot}"}
}
```

## Quick Reference

**Activate environment:** `conda activate diive`

**Run tests:** `pytest tests/ -v`

**View class:** `python -c "from diive.core.ml.common import MlRegressorGapFillingBase; help(MlRegressorGapFillingBase)"`

**Format code:** `black diive/` (optional)

**Check imports:** `python -c "import diive; print(diive.__version__)"`

---

**Last Updated:** 2026-05-06  
**Version:** v0.91.0+

For detailed implementation history, see `CHANGELOG.md`.
