# CLAUDE.md - DIIVE Development Guide

Quick reference for DIIVE development. See `CHANGELOG.md` for version history.

**Quick navigation:** [Behavioral Guidelines](#behavioral-guidelines) | [Quick Start](#quick-start) | [Coding Standards](#coding-standards) | [Flux Processing Chain](#flux-processing-chain-swiss-fluxnet-workflow) | [Common Workflows](#common-workflows) | [Quick Reference](#quick-reference)

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

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
├── core/plotting/            # Visualization types
├── core/times/               # Timestamp handling
├── core/io/                  # File I/O
├── pkgs/gapfilling/          # Gap-filling (RF, XGBoost, MDS)
├── pkgs/flux/                # Flux processing (lowres, hires, chain)
├── pkgs/preprocessing/       # Outlier detection, corrections, QA/QC
├── pkgs/analysis/            # Time series analysis
└── pkgs/features/            # Variable calculations
examples/                      # ~100 runnable examples
tests/                        # Unit tests
```

## Public API Overview

`import diive as dv` exposes 9 domain namespaces:

| Namespace | Contents |
|---|---|
| `dv.outliers` | `AbsoluteLimits`, `Hampel`, `LocalSD`, `LocalOutlierFactor`, `zScore`, `zScoreRolling`, `zScoreIncrements`, `TrimLow`, `ManualRemoval`, + daytime/nighttime variants |
| `dv.gapfilling` | `RandomForestTS`, `XGBoostTS`, `FluxMDS`, `QuickFillRFTS`, `OptimizeParamsRFTS`, `OptimizeParamsTS`, `FeatureEngineer`, `linear_interpolation` |
| `dv.flux` | `FluxProcessingChain`, `WindDoubleRotation`, `reynolds_decomposition`, `MaxCovariance`, `PreWhiteningBootstrap`, `PwbBatchDetection`, `FluxDetectionLimit`, ustar classes |
| `dv.analysis` | `DailyCorrelation`, `GrangerCausality`, `StratifiedAnalysis`, `GapFinder`, `GridAggregator`, `Histogram`, `FindOptimumRange`, `SeasonalTrendDecomposition`, `BinFitterCP`, `percentiles101` |
| `dv.plotting` | `HeatmapDateTime`, `HeatmapXYZ`, `HeatmapYearMonth`, `HexbinPlot`, `ScatterXY`, `TimeSeries`, `DielCycle`, `RidgeLinePlot`, `HistogramPlot`, `Cumulative`, `CumulativeYear`, `LongtermAnomaliesYear`, `TreeRingPlot` |
| `dv.times` | `TimestampSanitizer`, `DetectFrequency`, `resample_to_monthly_agg_matrix`, `timestamp_infer_freq_*` |
| `dv.features` | `DaytimeNighttimeFlag`, `TimeSince`, `potrad`, `potrad_eot`, `calc_vpd_from_ta_rh`, `aerodynamic_resistance`, `dry_air_density`, `et_from_le`, `latent_heat_of_vaporization`, `air_temp_from_sonic_temp`, `lagged_variants`, noise helpers |
| `dv.corrections` | `MeasurementOffsetFromReplicate`, `WindDirOffset`, `remove_radiation_zero_offset`, `remove_relativehumidity_offset`, `set_exact_values_to_missing`, `setto_threshold`, `setto_value` |
| `dv.qaqc` | `FlagQCF`, `StepwiseMeteoScreeningDb` |

Top-level (no namespace): `load_exampledata_parquet`, `load_parquet`, `save_parquet`, `ReadFileType`, `search_files`, `sstats`, `transform_yearmonth_matrix_to_longform`, `get_encoded_value_from_int`, `get_encoded_value_series`

## Core Concepts

### Feature Engineering (8-stage)

1. Lag features (past/future values)
2. Rolling stats (mean, std, quantiles, etc.)
3. Differencing (1st/2nd order rate of change)
4. EMA (exponential moving averages)
5. Polynomial (squared/cubic terms)
6. STL decomposition (trend, seasonal, residual)
7. Timestamps (year, season, month, hour)
8. Record number (temporal ordering)

Used by `FeatureEngineer` class, fed into gap-filling models.

### Gap-Filling Methods

| Method         | Training | Features                  | Use case              |
|----------------|----------|---------------------------|-----------------------|
| Random Forest  | Yes      | 8-stage engineered        | Interpretable, robust |
| XGBoost        | Yes      | 8-stage engineered        | Non-linear, efficient |
| MDS            | No       | Meteorological similarity | No overfitting risk   |
| Linear Interp. | No       | None                      | Small gaps only       |

**Results:** All gap-filling classes expose `.results` property (after `.run()`) returning `GapFillingResult` with:
- `gapfilled` — Series
- `flag` — 0=observed, 1=gap-filled, 2=fallback
- `scores['r2']` — gap-filling R²
- `feature_importances` — SHAP DataFrame (ML methods only)
- `model` — trained regressor (ML methods only)

MDS validates required columns at init and provides domain-aware errors.
Legacy `.result` property (raw DataFrame) still available.

### Timestamp Sanitization

10-step validation pipeline with monotonicity enforcement:

```python
sanitizer = dv.times.TimestampSanitizer(df, nominal_freq='30min', verbose=True)
clean_df = sanitizer.get()
status = sanitizer.get_status()  # Diagnostics: rows removed/added, detection method
```

See `examples/times/times_timestamp_sanitizer.py`.

## Flux Processing Chain (Swiss FluxNet Workflow)

5-level eddy covariance flux post-processing following FLUXNET/Swiss standards.

**Levels:**

- **L2** — Quality flag expansion (7 tests)
- **L3.1** — Storage correction
- **L3.2** — Outlier removal (sequential chain)
- **L3.3** — USTAR turbulence filtering (nighttime only)
- **L4.1** — Gap-filling (RF, XGBoost, MDS)

**Three API styles** (since v0.91.0+):

1. **Multi-flux loop** — `FluxConfig` + `run_flux_chain`. Write site parameters once; iterate over all fluxes. Each flux gets its own `FluxConfig` with tailored USTAR thresholds, outlier sigma, and L2 tests. Use this for typical site processing (FC/NEE, H, LE, N2O, CH4).

   ```python
   from diive.pkgs.flux.fluxprocessingchain import FluxConfig, run_flux_chain

   SITE = dict(site_lat=47.42, site_lon=9.84, utc_offset=1,
               nighttime_threshold=20, daytime_accept_qcf_below=2,
               nighttime_accept_qcf_below=2)

   fc_cfg = FluxConfig(
       fluxcol='FC', ustar_thresholds=[0.30], ustar_labels=['CUT_50'],
       outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,  # required, no defaults
       gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_1_1_1'],
       level2_tests={'ssitc': {'apply': True, 'setflag_timeperiod': None}, ...},
       mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa_1_1_1',
   )
   h_cfg = FluxConfig(
       fluxcol='H', ustar_thresholds=[0.0], ustar_labels=['CUT_NONE'],  # no USTAR for energy fluxes
       outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,
       set_storage_to_zero=True,  # no heat storage profile available
       ...
   )

   results: dict[str, FluxLevelData] = {}
   for cfg in [fc_cfg, h_cfg]:
       results[cfg.fluxcol] = run_flux_chain(df, cfg, **SITE, engineer=engineer)
   ```

2. **Composable functions** — one pure callable per level, each taking and returning a `FluxLevelData` container. Pure functions — never mutate input. Use this when you want a partial pipeline (e.g. L2 + L3.1 only), a custom L3.2, or branching at L4.1.

   ```python
   from diive.pkgs.flux.fluxprocessingchain import (
       init_flux_data, run_level2, run_level31,
       make_level32_detector, run_level32,
       run_level33_constant_ustar,
       run_level41_mds, run_level41_rf, run_level41_xgb,
   )

   data = init_flux_data(df, fluxcol='FC', site_lat=46.6, site_lon=9.8, utc_offset=1)
   data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None}, ...)
   data = run_level31(data, gapfill_storage_term=True)
   # stop here if you only need L2 + L3.1
   final_df = data.fpc_df
   ```

3. **`FluxProcessingChain` class** — convenience orchestrator that wraps the callables for the common "run all 5 levels" path. All existing methods/properties (`fpc.fpc_df`, `fpc.level2`, `fpc.level32_qcf`, etc.) keep working unchanged.

   ```python
   fpc = dv.flux.FluxProcessingChain(df=df, fluxcol='FC', ...)
   fpc.level2_quality_flag_expansion(**LEVEL2_SETTINGS)
   fpc.level31_storage_correction()
   # ... rest of the chain
   ```

**Container types:**

| Field | Type | Description |
|---|---|---|
| `data.fpc_df` | `DataFrame` | Working dataframe; grows as levels append flag/QCF columns. Use this for results/export. |
| `data.full_df` | `DataFrame` | Full input dataframe (with day/night flags added). Used read-only by L2, L3.1, L4.1 for driver columns. |
| `data.filteredseries` | `Series \| None` | QCF-filtered flux from the most recent level |
| `data.meta` | `FluxMeta` (frozen) | Site coordinates, fluxcol, swinpot_col, QCF thresholds |
| `data.levels` | `LevelResults` | Typed bag of per-level outputs (see below) |
| `data.level_ids` | `list[str]` | Identifiers of levels run, in order |
| `data.summary()` | `str` | Per-level data availability with daytime/nighttime breakdown |
| `data.gapfilled_cols()` | `dict[str, dict[str, str]]` | Gap-filled column names per L4.1 method and USTAR scenario |

`LevelResults` exposes every per-level instance behind a named field — no magic-string dict lookups:

```python
data.levels.level2                      # FluxQualityFlagsEddyPro
data.levels.level2_qcf                  # FlagQCF
data.levels.filteredseries_level2_qcf   # Series
data.levels.filteredseries_hq           # Series (QCF=0 only)
data.levels.level31                     # FluxStorageCorrectionSinglePointEddyPro
data.levels.flux_corrected_col          # str
data.levels.filteredseries_level31_qcf  # Series
data.levels.level32                     # StepwiseOutlierDetection
data.levels.level32_qcf                 # FlagQCF
data.levels.filteredseries_level32_qcf  # Series
data.levels.level33                     # FlagMultipleConstantUstarThresholds
data.levels.level33_qcf                 # dict[ustar_scenario, FlagQCF]
data.levels.filteredseries_level33_qcf  # dict[ustar_scenario, Series]
data.levels.filteredseries_level33_hq   # dict[ustar_scenario, Series] (QCF=0 only)
data.levels.level41_mds                 # dict[ustar_scenario, FluxMDS]
data.levels.level41_rf                  # dict[ustar_scenario, LongTermGapFillingRandomForestTS]
data.levels.level41_xgb                 # dict[ustar_scenario, LongTermGapFillingXGBoostTS]
```

**Architecture notes:**

- Level callables live in `diive/pkgs/flux/fluxprocessingchain/levels/` (one module per level).
- `finalize_level2()`, `finalize_level31()`, `finalize_level33()` are now no-ops emitting `DeprecationWarning` — the matching `levelXX_*` method runs everything in one go.
- Level-3.2 still uses the multi-call pattern because `StepwiseOutlierDetection` is inherently stateful: `level32_stepwise_outlier_detection()`, multiple `level32_flag_*` calls, `level32_addflag()`, then `finalize_level32()`.
- For the composable API, use `make_level32_detector(data)` to get a properly-wired `StepwiseOutlierDetection`, then pass it to `run_level32(data, outlier_detector=sod)`.
- `run_level41_rf` / `run_level41_xgb` take a pre-built `FeatureEngineer` instance (the class wrapper still accepts the legacy `features_*` keyword set for backward compatibility, and constructs the engineer internally).

**Critical pitfalls:**

- Wrong USTAR threshold filters too much/little nighttime
- MDS requires exact units: W/m² (radiation), °C (temp), **kPa (VPD)** — EddyPro outputs VPD in hPa; divide by 10 before passing to `run_level41_mds()`
- USTAR filtering applies ONLY to CO2/CH4/N2O, not H/LE (energy fluxes); for H/LE use `thresholds=[0], threshold_labels=['CUT_NONE']`
- L3.2 and L3.3 require L3.1 to have run; for H/LE call `run_level31(data, set_storage_to_zero=True)` instead of skipping
- L4.1 `features` and MDS driver columns must exist in `data.full_df`, not `data.fpc_df`; run `data.gapfilled_cols()` to find gap-filled column names after L4.1
- `nighttime_accept_qcf_below` (was `nighttimetime_accept_qcf_below` before v0.91.1 — typo fixed)
- Default `daytime_accept_qcf_below=1` is stricter than the FLUXNET/Swiss FluxNet convention of `2` (keep QCF=0 and QCF=1); QCF=0 means all tests pass, QCF=1 is soft warnings, QCF=2 is hard failure
- `run_level33_constant_ustar` only supports constant thresholds; use REddyProc externally for bootstrap threshold estimation, then pass the percentile values here
- `FeatureEngineer(target_col='_target_', ...)` — `target_col` is a required placeholder for L4.1; any string not in your feature list works

**Examples:**

- `examples/flux/fluxprocessingchain/fluxprocessingchain.py` — all 5 levels via the orchestrator class (RF + XGBoost).
- `examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py` — full L2→L4.1 pipeline using composable functions, including RF, XGBoost, and MDS gap-filling; heatmaps and cumulative plots.
- `examples/flux/fluxprocessingchain/fluxprocessingchain_multiflux.py` — multi-flux loop with `FluxConfig` + `run_flux_chain` for FC, H, and N2O; combined export and gap-filling fraction.
- `examples/flux/fluxprocessingchain/fluxprocessingchain_quick.py` — `QuickFluxProcessingChain` wrapper for rapid exploratory checks.

## High-Resolution EC Analysis (hires)

Tools for 10/20 Hz sonic anemometer data. Typical workflow:
```
raw 20 Hz  →  WindDoubleRotation  →  reynolds_decomposition  →  flux
```

**Key classes:** `WindDoubleRotation` (Wilczak et al. 2001 rotation), `reynolds_decomposition`, `MaxCovariance` (lag detection), `FluxDetectionLimit`, `PreWhiteningBootstrap` (PWB robust lag), `PwbBatchDetection` (parallel batch PWB).

**Wind rotation workflow — two separate steps:**

```python
wr = dv.flux.WindDoubleRotation(u=df['u'], v=df['v'], w=df['w'])  # Rotate
w_prime = dv.flux.reynolds_decomposition(wr.w2)   # Extract fluctuations
c_prime = dv.flux.reynolds_decomposition(df['CO2'])
flux = (w_prime * c_prime).mean()
```

**Critical:** Apply `reynolds_decomposition` to rotated `wr.w2`, not raw `w`. See `examples/flux/hires/flux_windrotation.py`.

## Outlier Detection & QC

**10 outlier methods:** AbsoluteLimits, Hampel, LocalSD, zScore (4 variants), LocalOutlierFactor, TrimLow, ManualRemoval.
Chain via `StepwiseOutlierDetection`. See `examples/preprocessing/outlier_detection/`.

**Quality Control (QCF):** Combines test flags into 0 (good) / 1 (marginal) / 2 (poor). See `examples/preprocessing/qaqc/qc_overall_flag.py`.

**Timestamp Shift Detection:** Three methods compare measured vs. theoretical radiation (positive = measured peaks early, negative = late). Requires mostly-clear days. See `examples/preprocessing/qaqc/qaqc_detect_timestamp_shifts.py`.

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

### Console Output & Verbosity

**All production output uses Rich console helpers** from the shared singleton at `diive/core/utils/console.py`.

**NO `print()` calls in production code.** Print is allowed only in:
- Example files (`examples/*/`)
- Docstring code snippets (non-executable documentation)
- `if __name__ == '__main__'` blocks
- CLI helper functions like `_cli_main()`

**Import pattern:**

```python
from diive.core.utils.console import console as _console, info, detail, warn, success, rule
```

**Helper functions & verbosity levels:**

| Function | Level | Use case | Example |
|----------|-------|----------|---------|
| `rule(title)` | PROGRESS (2) | Section headers, major milestones | `rule("Gap-Filling Report")` |
| `info(msg)` | PROGRESS (2) | Key progress messages, results | `info(f"Train R²: {r2:.3f}")` |
| `success(msg)` | PROGRESS (2) | Operation completion | `success("Gap-filling complete")` |
| `warn(msg)` | ERROR (1) | Warnings, always visible | `warn("No features passed SHAP threshold")` |
| `error(msg)` | ERROR (1) | Errors, always visible | `error("Target column not found")` |
| `detail(msg)` | DEBUG (3) | Inner-loop details, per-iteration | `detail(f"Iteration {i}: loss={loss}")` |
| `_console.print(msg)` | None | User-facing reports, formatted text | Multi-line report tables, formatted output |

**Verbosity levels:**

- `VERBOSE_SILENT = 0` — No output
- `VERBOSE_ERROR = 1` — Errors and warnings only
- `VERBOSE_PROGRESS = 2` — Section headers and results (default)
- `VERBOSE_DEBUG = 3` — All detail lines

All helpers accept `verbose=` argument (int or bool):

```python
info("This step took 2.5s", verbose=self.verbose)
detail(f"Iteration {i}", verbose=3)  # Only shows when verbose >= 3
```

**Pattern for integer-verbose files** (e.g., `if self.verbose >= 1`):

When a class uses integer verbose guards, call helpers WITHOUT a `verbose=` arg inside the block. The guard handles visibility:

```python
class MyProcessor:
    def process(self):
        if self.verbose >= 1:
            info("Starting processing")  # No verbose= arg needed
        if self.verbose >= 2:
            detail(f"Step details")
```

**Report methods** use `_console.print()` directly for formatted, user-facing output (no verbosity gating):

```python
def report(self):
    rule("Gap-Filling Results")
    _console.print(f"  Target: {self.target_col}")
    _console.print(f"  R² (train): {self.r2_train:.3f}")
```

**Do NOT:**

- Use `print()` in production code
- Create separate `Console` instances (use the shared `_console`)
- Use `logging` module for general output (it's for persistent log files)
- Mix print() and Rich helpers in the same file

### Examples (Sphinx Gallery format)

Use `# %%` cell markers. No file I/O (API only). Single year of data. Disable `showplot=True`.

**Checklist for new examples:**
1. Register in `examples/run_all_examples.py` and `examples/CATALOG.md`
2. Add category README description
3. Reference in source docstring "Example" section
4. Update `examples/README.md` file count
5. Note in CHANGELOG.md
6. Verify it runs without errors

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
scatter = dv.plotting.ScatterXY(x=df['A'], y=df['B'])

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

**Color palette:** Material Design colors. Use 300-level (bars, lines) and 500-level (backgrounds): blue `#2196F3`, red `#F44336`, amber `#FFC107`, grey `#455A64`.

**Bar label centering:** Use `va='center_baseline'` (not `va='center'`) for digit-only strings inside bars.

**Label contrast:** Choose white/black text based on WCAG luminance: `text_color = 'white' if 0.299*r + 0.587*g + 0.114*b < 0.5 else 'black'`.

**Dynamic height:** Scale multi-year panels: `height = max(1.5, n_years * 0.38)` inches.

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

| Issue | Workaround |
|-------|-----------|
| SHAP importance fluctuates ±5-10% | Use flexible ranges in tests (`assertGreater/Less`) |
| XGBoost base_score in scientific notation | Monkey-patched in `MlRegressorGapFillingBase` |
| Feature reduction too strict | Reduce `shap_threshold_factor` (default 0.5) |
| Unicode on Windows (arrow chars) | Use ASCII (>, ->) in examples |

## Examples (~100 runnable scripts)

Organized by functional domain: visualization (18), times (5), analysis (10), features (11), preprocessing (21), flux (20), gapfilling (11), io (5), fits (2).

**Run individual example:**
```bash
uv run python examples/gapfilling/gapfill_randomforest.py
```

**Run all examples:**
```bash
python examples/run_all_examples.py  # All in parallel
```

See `examples/CATALOG.md` for complete listing.

## Common Workflows

**Add feature engineering stage:**
1. Add parameter to `FeatureEngineer.__init__()`
2. Implement `_stagename_features()` method
3. Call from `_create_features()` orchestrator
4. Use naming: `.{col}_TYPE{detail}` (e.g., `.Tair_f_POL2`)

**Add gap-filling method:**
1. Create `level41_newmethod()` with 24 feature parameters
2. Build `FeatureEngineer`, create and train model
3. Store results in `self._level41['new_method'][ustar_scenario]`

**Debug SHAP importance:**
1. Check `.RANDOM` baseline included
2. Verify threshold calculation
3. Inspect feature counts before/after reduction

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

**Last Updated:** 2026-05-26 (Rich console migration completed)  
**Version:** v0.91.0+  
**Package Manager:** `uv`
