# CLAUDE.md - DIIVE Development Guide

See `CHANGELOG.md` for version history.

## Behavioral Guidelines

**Bias toward caution over speed. For trivial tasks, use judgment.**

**Think before coding:** State assumptions explicitly. If multiple interpretations exist, present them. If something is unclear, ask. Push back when a simpler approach exists.

**Simplicity first:** Minimum code to solve the problem. No speculative features, abstractions for single-use code, or error handling for impossible scenarios.

**Surgical changes:** Touch only what you must. Don't "improve" adjacent code. Match existing style. Mention unrelated dead code — don't delete it. Remove only imports/variables that YOUR changes made unused.

**Goal-driven execution:** For multi-step tasks, state a plan with verifiable success criteria before coding.

## Quick Start

```bash
uv sync
uv run pytest tests/test_gapfilling.py -v
uv run python script.py
uv run pytest tests/ -v
uv add package_name
```

## Development Environment

**Python:** 3.12-3.13 | **Package Manager:** `uv`

**Key dependencies (v0.91.1+):** pandas 3.0.3, numpy 2.4+, scikit-learn 1.8+, xgboost 3.2+, matplotlib 3.10+, statsmodels 0.14+, pyarrow 19.0+

## Project Structure

```
diive/
├── core/ml/                  # Feature engineering, ML base classes
├── core/plotting/            # Visualization types
├── core/times/               # Timestamp handling
├── core/io/                  # File I/O
├── gapfilling/               # Gap-filling (RF, XGBoost, MDS)
├── flux/                     # Flux processing (lowres, hires, chain)
├── preprocessing/            # Wrapper for domain-based preprocessing modules
├── corrections/              # Offset/gain removal, value corrections
├── outliers/                 # 10+ outlier detection methods
├── qaqc/                     # Quality control flags and screening
├── analysis/                 # Time series analysis
└── variables/                # Feature engineering and calculations
examples/                      # ~100 runnable examples
tests/                        # Unit tests
```

## Public API Overview

`import diive as dv` exposes 9 domain namespaces:

| Namespace | Contents |
|---|---|
| `dv.outliers` | `AbsoluteLimits`, `Hampel`, `LocalSD`, `LocalOutlierFactor`, `zScore`, `zScoreRolling`, `zScoreIncrements`, `TrimLow`, `ManualRemoval`, + daytime/nighttime variants |
| `dv.gapfilling` | `RandomForestTS`, `XGBoostTS`, `SWINGapFillerXGBoost`, `FluxMDS`, `QuickFillRFTS`, `OptimizeParamsRFTS`, `OptimizeParamsTS`, `FeatureEngineer`, `linear_interpolation` |
| `dv.flux` | `FluxProcessingChain`, `WindDoubleRotation`, `reynolds_decomposition`, `MaxCovariance`, `PreWhiteningBootstrap`, `PwbBatchDetection`, `FluxDetectionLimit`, ustar classes |
| `dv.analysis` | `DailyCorrelation`, `GrangerCausality`, `StratifiedAnalysis`, `GapFinder`, `GapStats`, `GridAggregator`, `Histogram`, `FindOptimumRange`, `SeasonalTrendDecomposition`, `BinFitterCP`, `percentiles101` |
| `dv.plotting` | `HeatmapDateTime`, `HeatmapXYZ`, `HeatmapYearMonth`, `HexbinPlot`, `ScatterXY`, `TimeSeries`, `DielCycle`, `RidgeLinePlot`, `HistogramPlot`, `Cumulative`, `CumulativeYear`, `LongtermAnomaliesYear`, `TreeRingPlot` |
| `dv.times` | `TimestampSanitizer`, `DetectFrequency`, `resample_to_monthly_agg_matrix`, `timestamp_infer_freq_*` |
| `dv.variables` | `DaytimeNighttimeFlag`, `TimeSince`, `potrad`, `potrad_eot`, `calc_vpd_from_ta_rh`, `aerodynamic_resistance`, `dry_air_density`, `et_from_le`, `latent_heat_of_vaporization`, `air_temp_from_sonic_temp`, `lagged_variants`, noise helpers |
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

| Method                | Training | Features                          | Use case                          |
|-----------------------|----------|-----------------------------------|-----------------------------------|
| Random Forest         | Yes      | 8-stage engineered                | Interpretable, robust             |
| XGBoost               | Yes      | 8-stage engineered                | Non-linear, efficient             |
| SWINGapFillerXGBoost  | Yes      | SW_IN_POT + timestamps (+ opt. TA/VPD) | SW_IN with physical nighttime constraint; `nighttime_threshold=0.001` matches `remove_radiation_zero_offset` |
| MDS                   | No       | Meteorological similarity         | No overfitting risk               |
| Linear Interp.        | No       | None                              | Small gaps only                   |

**Results:** All gap-filling classes expose `.results` (after `.run()`) returning `GapFillingResult`:
- `gapfilled` — Series; `flag` — 0=observed, 1=gap-filled, 2=fallback
- `scores['r2']`, `feature_importances` (SHAP, ML only), `model` (ML only)

Legacy `.result` property (raw DataFrame) still available.

### Timestamp Sanitization

```python
sanitizer = dv.times.TimestampSanitizer(df, nominal_freq='30min', verbose=True)
clean_df = sanitizer.get()
status = sanitizer.get_status()  # rows removed/added, detection method
```

## Flux Processing Chain (Swiss FluxNet Workflow)

5-level eddy covariance post-processing: L2 (quality flags) → L3.1 (storage correction) → L3.2 (outlier removal) → L3.3 (USTAR filtering) → L4.1 (gap-filling).

Each level is a pure function — never mutate input.

```python
from diive.flux.fluxprocessingchain import (
    init_flux_data, run_level2, run_level31, run_level33_constant_ustar, run_level41_mds
)

data = init_flux_data(df, fluxcol='FC', site_lat=46.6, site_lon=9.8, utc_offset=1)
data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None}, ...)
data = run_level31(data, gapfill_storage_term=True)
data = run_level33_constant_ustar(data, ustar_thresholds=[0.30])
data = run_level41_mds(data, mds_swin='SW_IN', mds_ta='TA', mds_vpd='VPD')
final_df = data.fpc_df
```

**Container fields:**

| Field | Type | Description |
|---|---|---|
| `data.fpc_df` | `DataFrame` | Working dataframe; grows as levels append columns. Use for results/export. |
| `data.full_df` | `DataFrame` | Full input (with day/night flags). Read-only source for L2, L3.1, L4.1 driver columns. |
| `data.filteredseries` | `Series\|None` | QCF-filtered flux from most recent level |
| `data.meta` | `FluxMeta` (frozen) | Site coordinates, fluxcol, swinpot_col, QCF thresholds |
| `data.levels` | `LevelResults` | Typed bag of per-level outputs (see code for full field list) |
| `data.summary()` | `str` | Per-level data availability with daytime/nighttime breakdown |
| `data.gapfilled_cols()` | `dict` | Gap-filled column names per L4.1 method and USTAR scenario |
| `data.gap_stats(level='L33')` | `dict[str, GapStats]` | On-demand gap analysis; keyed by USTAR scenario (single-entry for L2/L31/L32) |
| `data.plot_cumulative_comparison(...)` | `None` | Overlay cumulative sums of all gap-filling methods on one axes |
| `data.plot_gapfilled_heatmaps(...)` | `None` | Side-by-side heatmaps: measured + one panel per gap-filling method; one figure per USTAR scenario |

Key `data.levels` fields: `level2`, `level2_qcf`, `level31`, `level32`, `level32_qcf`, `level33`, `level33_qcf`, `level41_mds`, `level41_rf`, `level41_xgb` (dicts keyed by ustar_scenario for L3.3+).

**Architecture notes:**

- L3.2 uses the multi-call pattern (stateful): `make_level32_detector(data)` → multiple `level32_flag_*` calls → `run_level32(data, outlier_detector=sod)`.
- `run_level41_rf` / `run_level41_xgb` take a pre-built `FeatureEngineer` instance.
- `finalize_level2/31/33()` are no-ops with `DeprecationWarning`.

**Critical pitfalls:**

- MDS requires exact units: W/m² (radiation), °C (temp), **kPa (VPD)** — EddyPro outputs VPD in hPa; divide by 10.
- USTAR filtering applies ONLY to CO2/CH4/N2O; for H/LE use `thresholds=[0], threshold_labels=['CUT_NONE']`.
- L3.2 and L3.3 require L3.1; for H/LE call `run_level31(data, set_storage_to_zero=True)`.
- L4.1 features and MDS driver columns must exist in `data.full_df`, not `data.fpc_df`.
- `nighttime_accept_qcf_below` (was `nighttimetime_accept_qcf_below` before v0.91.1 — typo fixed).
- Default `daytime_accept_qcf_below=1` is stricter than FLUXNET convention of `2`; QCF=0 all pass, QCF=1 soft warning, QCF=2 hard failure.
- `run_level33_constant_ustar` only supports constant thresholds; use REddyProc externally for bootstrap, then pass percentile values here.
- `FeatureEngineer(target_col='_target_', ...)` — `target_col` is a required placeholder; any string not in the feature list works.

**Example:** `examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py`

## High-Resolution EC Analysis (hires)

Tools for 10/20 Hz data. Workflow: `raw 20 Hz → WindDoubleRotation → reynolds_decomposition → flux`

```python
wr = dv.flux.WindDoubleRotation(u=df['u'], v=df['v'], w=df['w'])
w_prime = dv.flux.reynolds_decomposition(wr.w2)   # use rotated w2, not raw w
c_prime = dv.flux.reynolds_decomposition(df['CO2'])
flux = (w_prime * c_prime).mean()
```

**Classes:** `WindDoubleRotation`, `reynolds_decomposition`, `MaxCovariance`, `FluxDetectionLimit`, `PreWhiteningBootstrap`, `PwbBatchDetection`.

## Outlier Detection & QC

- **Single method:** `dv.outliers.Hampel(series).run()`
- **Chained:** `dv.outliers.StepwiseOutlierDetection()` (orchestrates multiple methods)
- **Corrections:** `dv.corrections.MeasurementOffsetFromReplicate()`, `remove_radiation_zero_offset()`, `setto_*()`
- **QCF aggregation:** `dv.qaqc.FlagQCF()` → 0 (good) / 1 (marginal) / 2 (poor)
- **Full pipeline:** `dv.qaqc.StepwiseMeteoScreeningDb()` — corrections → outlier detection → quality flags
- **Timestamp shift:** three methods comparing measured vs. theoretical radiation (requires clear days)

## Coding Standards

### Input validation

Only at system boundaries (user input, external data). Trust internal code.

### Error handling

Let exceptions propagate unless you can recover.

### Comments

Only WHY, not WHAT. Hidden constraints, workarounds, non-obvious logic.

### Console Output & Verbosity

**All production output uses Rich console helpers** from `diive/core/utils/console.py`. **NO `print()` in production code** (allowed in `examples/*/`, docstrings, `__main__` blocks, `_cli_main()`).

```python
from diive.core.utils.console import console as _console, info, detail, warn, success, rule
```

| Function | Level | Use case |
|----------|-------|----------|
| `rule(title)` | PROGRESS (2) | Section headers |
| `info(msg)` | PROGRESS (2) | Key progress, results |
| `success(msg)` | PROGRESS (2) | Operation completion |
| `warn(msg)` | ERROR (1) | Warnings (always visible) |
| `error(msg)` | ERROR (1) | Errors (always visible) |
| `detail(msg)` | DEBUG (3) | Inner-loop details |
| `_console.print(msg)` | None | User-facing formatted reports |

Levels: `VERBOSE_SILENT=0`, `VERBOSE_ERROR=1`, `VERBOSE_PROGRESS=2` (default), `VERBOSE_DEBUG=3`.

All helpers accept `verbose=` arg. When using integer `if self.verbose >= N:` guards, call helpers WITHOUT `verbose=` inside the block.

**Do NOT:** use `print()`, create separate `Console` instances, use `logging` for general output, mix `print()` and Rich in the same file.

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

**Phase 1: `__init__()`** — data + computation parameters ONLY (no `ax`, title, labels, colors, limits).

**Phase 2: `plot(ax=None, ...)`** — all styling + rendering. `ax=None` creates a new figure. Can be called multiple times with different styles/axes.

```python
scatter = dv.plotting.ScatterXY(x=df['A'], y=df['B'])
scatter.plot(ax=axes[0], title='Linear')
scatter.plot(ax=axes[1], title='Log', ylim='auto')
```

## Plotting Conventions

**Colors:** Material Design 300-level (bars/lines) and 500-level (backgrounds): blue `#2196F3`, red `#F44336`, amber `#FFC107`, grey `#455A64`.

**Bar labels:** `va='center_baseline'` (not `va='center'`) for digit-only strings inside bars.

**Label contrast:** `text_color = 'white' if 0.299*r + 0.587*g + 0.114*b < 0.5 else 'black'`

**Dynamic height:** `height = max(1.5, n_years * 0.38)` inches for multi-year panels.

## Development Workflow

**[CRITICAL] NEVER COMMIT CHANGES.** User stages and commits exclusively.

**[CRITICAL] NEVER RUN EXAMPLE SUITE.** Only test individual examples during development:
```bash
uv run python examples/gapfilling/gapfill_randomforest.py
```

**Commit message style:** One-line title (< 50 chars) + bullet points for details.

**Do NOT:** run `uv` commands without explicit approval, skip pre-commit hooks (`--no-verify`), force-push to main/master, include Claude as co-author in commit messages.

## Testing

```bash
pytest tests/test_gapfilling.py -v              # Gap-filling
pytest tests/test_fluxprocessingchain.py -v     # Flux chain
pytest tests/ -v                                 # All
```

- Use flexible assertion ranges (`assertGreater/assertLess`) for SHAP variability
- Don't mock databases in integration tests

## Module Docstring Format

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

## Common Workflows

**Add feature engineering stage:**
1. Add parameter to `FeatureEngineer.__init__()`
2. Implement `_stagename_features()` method
3. Call from `_create_features()` orchestrator
4. Use naming: `.{col}_TYPE{detail}` (e.g., `.Tair_f_POL2`)

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

**Last Updated:** 2026-05-27 | **Version:** v0.91.0+ | **Package Manager:** `uv`
