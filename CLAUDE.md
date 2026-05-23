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
- Data visualization (18+ plot types)

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

## High-Resolution EC Analysis (hires)

Tools for 10/20 Hz raw sonic anemometer data. All live in `diive/pkgs/flux/hires/`.

**Available tools:**

| Class / Function | Purpose | Example |
|---|---|---|
| `WindDoubleRotation` | Double rotation tilt correction (Wilczak et al. 2001) | `flux_windrotation.py` |
| `reynolds_decomposition` | Turbulent fluctuation x' = x - mean(x) | `flux_windrotation.py` |
| `MaxCovariance` | Time lag detection via cross-covariance maximisation | `flux_lag.py` |
| `FluxDetectionLimit` | Flux detection limit and signal-to-noise ratio (Langford et al. 2015) | `flux_fluxdetectionlimit.py` |
| `PreWhiteningBootstrap` | Robust lag detection for low-magnitude fluxes (Vitale et al. 2024) | `flux_lag_pwb.py` |
| `PwbBatchDetection` | Parallel batch PWB across many averaging-period files | `flux_lag_pwb_batch.py` |

**Typical per-averaging-period workflow:**

```
raw 20 Hz file
  -> WindDoubleRotation        # align coordinate system
  -> reynolds_decomposition    # extract w', c'
  -> MaxCovariance             # find time lag
  -> flux = mean(w' * c')      # eddy covariance flux
```

### Two-step wind rotation workflow

Wind rotation and Reynolds decomposition are **two separate steps** — do not combine them:

```python
import diive as dv

# Step 1: double rotation (Wilczak et al. 2001)
# Aligns coordinate system with mean wind; mean(v2) ~ 0, mean(w2) ~ 0
wr = dv.WindDoubleRotation(u=df['u'], v=df['v'], w=df['w'])

# Step 2: Reynolds decomposition — apply to rotated wind AND any scalar
w_prime = dv.reynolds_decomposition(wr.w2)   # x' = x - mean(x)
c_prime = dv.reynolds_decomposition(df['CO2'])

# Flux
flux = (w_prime * c_prime).mean()
```

**Design rationale:**

- `WindDoubleRotation` takes only `u, v, w` — the scalar has no role in rotation
- `reynolds_decomposition(x)` is a standalone function (`x - x.mean()`), not a class
- Keeping them separate makes each step explicit and reusable

**`WindDoubleRotation` attributes after construction:**

| Attribute | Description |
|-----------|-------------|
| `theta`   | First rotation angle, radians (yaw: sets mean v to zero) |
| `phi`     | Second rotation angle, radians (pitch: sets mean w to zero) |
| `u2`, `v2`, `w2` | Rotated wind components (high-res Series) |

**Critical pitfall:** always apply `reynolds_decomposition` to `wr.w2` (rotated), not the raw `w`.

**Example:** `examples/flux/hires/flux_windrotation.py`

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

## Timestamp Shift Detection

**DetectTimestampShifts** detects clock/timestamp errors by comparing measured shortwave radiation
against theoretical potential radiation. Three methods with a shared sign convention
(positive = measured peaks earlier / leading clock, negative = later / lagging clock):

- `fft_phase_shift()` — k=1 Fourier phase-angle comparison; fast, no upsampling needed
- `crosscorr()` — upsample to 1-min, scipy cross-correlation; 1-minute precision
- `noon_shift()` — vectorised daily peak-time delta; quick heuristic

Five plot methods cover time series, histograms, polar plots, monthly boxplots, diel cycles,
and radiation fingerprint heatmaps.

**Critical pitfall:** all three methods require clear or mostly-clear days; heavily overcast days
are filtered via a clearness index threshold before any phase analysis is performed.

**Example:** `examples/preprocessing/qaqc/qaqc_detect_timestamp_shifts.py`

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
- ✅ All 18 visualization examples

See `CHANGELOG.md` for detailed refactoring notes.

## Plotting Conventions

### Color palette

Use **Material Design** colours consistently across multi-panel figures:

| Zone | 300-level (top/bar panels) | 500-level (shaded background) |
|------|---------------------------|-------------------------------|
| in optimum / primary | `#64B5F6` blue | `#2196F3` blue |
| above optimum | `#E57373` red | `#F44336` red |
| below optimum | `#FFD54F` yellow | `#FFC107` amber |
| boundary lines | — | `#455A64` blue-grey |
| peak marker | — | `#37474F` dark blue-grey |

300-level for filled bars and line plots; 500-level for shaded background regions. Using two levels within the same figure avoids the top and bottom panels visually clashing.

### Bar label centering

Use `va='center_baseline'` (not `va='center'`) to visually center digit-only strings inside horizontal bars. `va='center'` aligns the bounding box, which includes descender whitespace — labels drift above center for strings without descenders (numbers).

```python
ax.text(rect.get_x() + rect.get_width() / 2,
        rect.get_y() + rect.get_height() / 2,
        f"{val:.1f}",
        ha='center', va='center_baseline',
        color=text_color)
```

### Automatic label contrast

Compute luminance with the WCAG formula to choose white or black label text automatically:

```python
import matplotlib.colors as mcolors
r, g, b, *_ = mcolors.to_rgba(color)
text_color = 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'
```

### Dynamic panel height scaling

When a panel contains one row per year, scale its height dynamically so it stays compact regardless of dataset length:

```python
n_years = len(yearly_df)
top_h = max(1.5, n_years * 0.38)   # ~0.38 in per year, minimum 1.5 in
```

Pair with `ax.margins(y=0.02)` to remove the default 5% vertical padding that creates excess whitespace when bars are few.

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
| Unicode encoding on Windows (arrow chars)  | Use ASCII equivalents (>, ->) in examples     |

## Examples (~100 runnable scripts)

Organized by functional domain. Each category has a README with file descriptions and usage.

**Structure:**

- `visualization/` — 18 plotting examples
- `times/` — 5 timestamp handling
- `analysis/` — 10 time series analysis
- `features/` — 11 variable engineering
- `fits/` — 2 data fitting
- `io/` — 5 file I/O
- `preprocessing/` — 21 (corrections, outlier detection, QA/QC)
- `flux/` — 18 (processing chain, low-res, high-res)
- `gapfilling/` — 11 (RF, XGBoost, MDS, interpolation, comparison)

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

**Last Updated:** 2026-05-21  
**Version:** v0.91.0+  
**Package Manager:** `uv`
