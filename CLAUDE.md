# CLAUDE.md - DIIVE Development Guide

Quick reference for DIIVE development. See `CHANGELOG.md` for version history and recent implementations.

## Quick Start

**Setup (first time):**

```bash
uv sync
uv run pytest tests/test_gapfilling.py -v
```

**Common commands:**

```bash
uv run python script.py              # Run script
uv run pytest tests/ -v              # Run tests
uv add package_name                  # Add dependency
uv pip list                          # List packages
```

**Legacy conda (optional):**

```bash
conda env create -f environment.yml && conda activate diive
python -m pytest tests/test_gapfilling.py -v
```

## Project Overview

**DIIVE** — Python library for time series processing, particularly ecosystem flux data.

**Core capabilities:**

- Timestamp sanitization (10-step validation, frequency detection)
- Feature engineering (8-stage pipeline)
- ML gap-filling (Random Forest, XGBoost, MDS)
- Flux processing chain (5-level workflow)
- Outlier detection & quality control
- Data visualization (14+ plot types)

## Development Environment

**Python:** 3.12-3.13 (via `pyproject.toml`)  
**Package Manager:** `uv` (modern, fast, deterministic via `uv.lock`)  
[Install uv](https://docs.astral.sh/uv/getting-started/)

**Key dependencies (v0.91.0+):**

- pandas 3.0.3, numpy 2.4+, scikit-learn 1.8+, xgboost 3.2+
- matplotlib 3.10+, statsmodels 0.14+, pyarrow 19.0+

All pinned in `pyproject.toml` for reproducibility.

## Project Structure

```
diive/
├── core/ml/                  # Feature engineering, ML base classes
├── core/plotting/            # 14+ visualization types
├── core/times/               # Timestamp handling
├── core/io/                  # File I/O
├── pkgs/gapfilling/          # RF, XGBoost, MDS gap-filling
├── pkgs/flux/                # Flux processing (lowres, hires, chain)
├── pkgs/preprocessing/       # Outlier detection, corrections, QA/QC
├── pkgs/analysis/            # Time series analysis, decomposition
└── pkgs/features/            # Variable calculations
examples/                      # Runnable examples (86)
tests/                        # Unit tests
```

## Core Concepts

### Feature Engineering (8-stage pipeline)

1. **Lag features** — Past/future values
2. **Rolling stats** — Mean, median, min, max, std, quantiles
3. **Differencing** — 1st/2nd order rate of change
4. **EMA** — Exponential moving averages
5. **Polynomial** — Squared/cubic terms
6. **STL** — Trend, seasonal, residual decomposition
7. **Timestamps** — Year, season, month, hour (vectorized)
8. **Record number** — Temporal ordering

Used by: `FeatureEngineer` class, fed into gap-filling models.

### Gap-Filling Methods

| Method         | Training | Features                  | Accuracy     | Use case              |
|----------------|----------|---------------------------|--------------|-----------------------|
| Random Forest  | Yes      | 8-stage engineered        | R² 0.60-0.80 | Interpretable, robust |
| XGBoost        | Yes      | 8-stage engineered        | R² 0.65-0.85 | Non-linear, efficient |
| MDS            | No       | Meteorological similarity | R² 0.40-0.70 | No overfitting risk   |
| Linear Interp. | No       | None                      | Simple       | Small gaps only       |

### Timestamp Sanitization

10-step validation pipeline (configurable, monotonicity required):

```python
from diive import TimestampSanitizer

sanitizer = TimestampSanitizer(
    data=df,
    output_middle_timestamp=True,  # Convert to mid-period
    nominal_freq='30min',  # Expected frequency
    verbose=True
)
clean_df = sanitizer.get()
status = sanitizer.get_status()  # Diagnostics: rows removed/added, frequency confidence, detection method
```

**Example:** `examples/times/times_timestamp_sanitizer.py` demonstrates 5 severity levels (clean → corrupted).

## Flux Processing Chain (Swiss FluxNet Workflow)

5-level eddy covariance flux post-processing following FLUXNET/Swiss standards.

**Levels:**

- **L2** — Quality flag expansion (7 tests)
- **L3.1** — Storage correction
- **L3.2** — Outlier removal (sequential chain)
- **L3.3** — USTAR turbulence filtering (nighttime only)
- **L4.1** — Gap-filling (RF, XGBoost, MDS)

**Critical pitfalls:**

- Wrong USTAR threshold filters too much/little nighttime
- MDS requires exact units: W/m² (radiation), °C (temp), hPa (VPD)
- USTAR filtering applies ONLY to CO2/CH4/N2O, not H/LE (energy fluxes)

**Example:** `examples/flux/fluxprocessingchain/fluxprocessingchain.py` (all 5 levels, both gap-filling methods).

## Outlier Detection Methods

10 built-in methods:

1. **AbsoluteLimits** — Min/max threshold
2. **Hampel** — Robust spike detection (MAD-based)
3. **LocalSD** — Adaptive local standard deviation
4. **zScore** (4 variants) — Global, rolling, increments, day/night
5. **LocalOutlierFactor** — Density-based anomalies
6. **TrimLow** — Trimmed mean approach
7. **ManualRemoval** — Explicit removal

**Chain multiple methods sequentially:** `StepwiseOutlierDetection` class orchestrates each method operating on data
filtered by previous tests.

**Examples:** `examples/preprocessing/outlier_detection/` (9 files, one per method + stepwise).

## Quality Control (QCF)

**FlagQCF** combines multiple test flags into single quality indicator:

- **QCF=0** — Good (all tests pass)
- **QCF=1** — Marginal (1-3 soft warnings)
- **QCF=2** — Poor (>3 soft warnings or ≥2 hard fails)

Features: Auto-detect test flags, day/night separation, USTAR scenario support, impact analysis.

**Example:** `examples/preprocessing/qaqc/qc_overall_flag.py`

## Coding Standards

### Input validation

Only at system boundaries (user input, external data). Trust internal code.

```python
def process_data(df, target_col):
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be empty")
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found")
```

### Error handling

Let exceptions propagate unless you can recover.

```python
try:
    result = operation()
except FileNotFoundError:
    logger.info("Using fallback")
```

### Comments

Only WHY, not WHAT. Hidden constraints, workarounds, non-obvious logic.

```python
# Good: explains constraint
# Exclude dot columns to avoid circular dependency with gap-filling

# Bad: explains what code does
# Add 1 to result
result = result + 1
```

### Examples (Sphinx Gallery format)

- No file I/O (show API only, not `.to_csv()`)
- Use `# %%` cell markers (becomes sections in Sphinx)
- Single year of data for speed
- Disable `showplot=True` (matplotlib blocks rendering)
- Explicit parameters with inline comments

```python
"""
===================
Example Title
===================

Brief description of what this teaches.
"""

# %%
# Section Title
# ^^^^^^^^^^^^^^
# Explanatory text about this section

import diive as dv

data = dv.load_exampledata_parquet()
```

**Documentation checklist (7-point):**

1. Category README.md — Add file description
2. examples/run_all_examples.py — Register file path
3. examples/CATALOG.md — Add to workflow table
4. examples/README.md — Update file count
5. Source code docstring "Example" section — Reference file
6. CHANGELOG.md — Note new/updated example
7. Run example to verify no errors

## Plotting Class Design (Two-Phase Pattern)

**Phase 1: `__init__()`** — Data + computation ONLY

- Accept data (Series, DataFrame, arrays)
- Accept computation parameters (nbins, aggregation, etc.)
- DO NOT include: ax, title, labels, colors, limits

**Phase 2: `plot()`** — All styling + rendering

- Accept `ax` (plot destination, default `None` = new figure)
- Accept styling (title, labels, colors, limits, etc.)
- Can be called multiple times with different styles/axes
- All parameters have sensible defaults

**Rationale:** Separates concerns, enables replotting same data with different styles, follows matplotlib convention.

**Example:**

```python
# Create once
scatter = dv.plot_scatter_xy(x=df['A'], y=df['B'])

# Render multiple times with different styles
fig, axes = plt.subplots(1, 2)
scatter.plot(ax=axes[0], title='Linear')
scatter.plot(ax=axes[1], title='Log', ylim='auto')
```

**Checklist:**

- `__init__()` contains ONLY data + computation parameters
- `ax` parameter in `plot()` method (first parameter)
- All styling parameters moved to `plot()`
- `ax=None` creates new figure via `pf.create_ax()`
- Examples call `plot()` for styling, not `__init__()`

**Refactoring status (May 2026):**

- ✅ HeatmapBase, HeatmapDateTime, HeatmapXYZ, HexbinPlot
- ✅ All 17 visualization examples

See `CHANGELOG.md` for detailed refactoring notes.

## Development Workflow

**[CRITICAL] NEVER COMMIT CHANGES.** User stages and commits exclusively.

**Commit message style:**

- One-line title (< 50 chars)
- Bullet points for details
- Example: `Refine hyperparameter optimization\n- Filter single-value parameters\n- Remove redundant legend`

**[CRITICAL] NEVER RUN EXAMPLE SUITE.** Only test individual examples during development.

```bash
uv run python examples/gapfilling/gapfill_randomforest.py
```

User runs `examples/run_all_examples.py` for full validation.

**Do NOT:**

- Run `uv` commands without explicit approval
- Skip pre-commit hooks (`--no-verify`)
- Force-push to main/master
- Include Claude as co-author in commit messages

## Testing

**Run tests:**

```bash
pytest tests/test_gapfilling.py -v              # Gap-filling
pytest tests/test_fluxprocessingchain.py -v     # Flux chain
pytest tests/ -v                                 # All
```

**Guidelines:**

- Use flexible assertion ranges (`assertGreater/assertLess`) for SHAP variability
- Validate at API boundaries (user input, external data), not internal contracts
- Don't mock databases in integration tests

## Module Docstring Format

**Standard format (reStructuredText):**

```python
"""
MODULE_NAME: DESCRIPTIVE_TITLE
================================

Brief one-line description of scope and purpose.

Part of the diive library: https://github.com/holukas/diive
"""
```

## Text Writing Standards

Use `/llm-detox` skill for all written content (documentation, comments, commit messages, examples).

## Known Issues & Workarounds

| Issue                                     | Workaround                                    |
|-------------------------------------------|-----------------------------------------------|
| SHAP importance fluctuates ±5-10%         | Use flexible assertion ranges in tests        |
| XGBoost base_score in scientific notation | Monkey-patched in `MlRegressorGapFillingBase` |
| Feature reduction too strict              | Reduce `shap_threshold_factor` (default 0.5)  |
| Unicode encoding on Windows (→ char)      | Use ASCII equivalents (>, >) in examples      |

## Examples (86 runnable scripts)

Organized by functional domain. Each category has a README with file descriptions and usage.

**Structure:**

- `visualization/` — 17 plotting examples
- `times/` — 6 timestamp handling
- `analysis/` — 10 time series analysis
- `features/` — 11 variable engineering
- `fits/` — 2 data fitting
- `io/` — 1 file I/O
- `preprocessing/` — 18 (corrections, outlier detection, QA/QC)
- `flux/` — 11 (processing chain, low-res, high-res)
- `gapfilling/` — 10 (RF, XGBoost, MDS, interpolation, comparison)

**Running examples:**

```bash
uv run python examples/gapfilling/gapfill_randomforest.py
python examples/run_all_examples.py  # All in parallel (8 workers)
```

See `examples/CATALOG.md` for complete listing and `examples/README.md` for details.

## Common Workflows

### Adding new feature engineering stage

1. Add parameter to `FeatureEngineer.__init__()` (default None)
2. Implement `_stagename_features()` method
3. Call from `_create_features()` orchestrator
4. Use naming: `.{col}_TYPE{detail}` (e.g., `.Tair_f_POL2`)

### Adding gap-filling method to FluxProcessingChain

1. Create `level41_newmethod()` with all 24 feature parameters
2. Create `FeatureEngineer`, apply to data
3. Create and train gap-filling model
4. Store results in `self._level41['new_method'][ustar_scenario]`

### Debugging SHAP importance issues

1. Check `.RANDOM` baseline feature included
2. Verify threshold: `random_mean + k * random_sd`
3. Check feature counts before/after reduction
4. Inspect `model_.feature_importances_traintest_`

## Quick Reference

| Task             | Command                                                     |
|------------------|-------------------------------------------------------------|
| Install          | `uv sync`                                                   |
| Run test         | `uv run pytest tests/ -v`                                   |
| Run example      | `uv run python examples/gapfilling/gapfill_randomforest.py` |
| Add package      | `uv add package_name`                                       |
| List packages    | `uv pip list`                                               |
| Check version    | `uv run python -c "import diive; print(diive.__version__)"` |
| Run all examples | `python examples/run_all_examples.py`                       |

---

**Last Updated:** 2026-05-15  
**Version:** v0.91.0+  
**Package Manager:** `uv`
