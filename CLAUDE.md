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

**Key dependencies (minimum pins):** pandas 3.0+, numpy 2.2+, scikit-learn 1.6+, xgboost 3.0+, matplotlib 3.10+, statsmodels 0.14+, pyarrow 19.0+

## Project Structure

```
diive/
├── core/ml/                  # Feature engineering, ML base classes
├── core/plotting/            # Visualization types
├── core/times/               # Timestamp handling
├── core/io/                  # File I/O
├── core/metadata/            # Per-variable tag + provenance model (GUI-backing)
├── gapfilling/               # Gap-filling (RF, XGBoost, MDS)
├── flux/                     # Flux processing (lowres, hires, chain)
├── preprocessing/            # Wrapper for domain-based preprocessing modules
├── corrections/              # Offset/gain removal, value corrections
├── outliers/                 # 10+ outlier detection methods
├── qaqc/                     # Quality control flags and screening
├── analysis/                 # Time series analysis
├── variables/                # Feature engineering and calculations
└── gui/                      # PySide6 desktop GUI (optional 'gui' extra)
examples/                      # ~100 runnable examples
tests/                        # Unit tests
```

## Public API Overview

`import diive as dv` exposes 10 domain namespaces:

| Namespace | Contents |
|---|---|
| `dv.outliers` | `AbsoluteLimits`, `Hampel`, `LocalSD`, `LocalOutlierFactor`, `zScore`, `zScoreRolling`, `zScoreIncrements`, `TrimLow`, `ManualRemoval`, + daytime/nighttime variants |
| `dv.events` | `Event` (instant or period time-stamped marker; `resolved_color(i, colors=)` accepts an optional `{category: hex}` override map), `event_to_flag` (→ 0/1 yes/no column aligned to an index), `overlay_events` (draw lines/spans on a datetime axes — `axis='x'` value-vs-time, `axis='y'` heatmap; `colors=` category-override map), `make_event_flag_name`, `CATEGORY_COLORS` |
| `dv.gapfilling` | `RandomForestTS`, `XGBoostTS`, `SWINGapFillerXGBoost`, `FluxMDS`, `QuickFillRFTS`, `OptimizeParamsRFTS`, `OptimizeParamsTS`, `LongTermGapFillingRandomForestTS`, `LongTermGapFillingXGBoostTS`, `FeatureEngineer`, `GapFillingResult`, `prediction_scores`, `linear_interpolation` |
| `dv.flux` | `FluxConfig`, `FluxLevelData`, `run_chain`, `init_flux_data`, `add_driver`, `WindDoubleRotation`, `reynolds_decomposition`, `MaxCovariance`, `PreWhiteningBootstrap`, `PwbBatchDetection`, `TlagApplier`, `PerFilePipeline`, `process_one_file`, `FluxDetectionLimit`, ustar classes, `NighttimePartitioning` / `partition_nee_nighttime` (NEE→GPP+RECO, Reichstein 2005 nighttime method; `lloyd_taylor`, `sunrise_sunset` helpers in `diive.flux.partitioning`). Per-level `run_level*`, `make_level32_detector`, and `chain_to_code` / `level2_to_code` … `level41_to_code` (render chain choices as a reproducible script) live in `diive.flux.fluxprocessingchain`. |
| `dv.analysis` | `DailyCorrelation`, `GrangerCausality`, `StratifiedAnalysis`, `GapFinder`, `GapStats`, `GridAggregator`, `Histogram`, `FindOptimumRange`, `SeasonalTrendDecomposition`, `BinFitterCP`, `harmonic_analysis`, `spectrogram`, `percentiles101`, `rank_drivers`, `profile_dataframe` (per-variable profiling table — one row per column: dtype/count/missing/gaps/unique/zeros/constant + numeric summaries), `dataframe_overview` (dataset-level facts: rows/cols/duplicate timestamps/missing %/freq/span/memory), `count_gaps` |
| `dv.analysis.experimental` | **(provisional, API may change)** `DriverAnalysis`, `DriverAnalysisResult`, `AleCurve`, `Ale2DResult`, `accumulated_local_effects`, `accumulated_local_effects_2d`, `ExperimentalWarning` — evidence-triangulation driver attribution; emits a one-time `ExperimentalWarning` on use |
| `dv.plotting` | `HeatmapDateTime`, `HeatmapXYZ`, `HeatmapYearMonth`, `HexbinPlot`, `ScatterXY`, `TimeSeries`, `DielCycle`, `RidgeLinePlot`, `HistogramPlot`, `ShiftedDistributionPlot`, `Cumulative`, `CumulativeYear`, `LongtermAnomaliesYear`, `TreeRingPlot` |
| `dv.times` | `TimestampSanitizer`, `DetectFrequency`, `keep_daterange` (non-destructive date-range subselection; inclusive `start`/`end`, either bound optional), `insert_timestamp` (add a START/MIDDLE/END timestamp **column** without changing the index), `format_timestamp` (return an aligned START/MIDDLE/END timestamp Series for a chosen convention; optional `fmt=` strftime → formatted strings, e.g. FLUXNET `%Y%m%d%H%M`; else `datetime64`), `validate_timestamp_column_name` (reject a reserved name `TIMESTAMP_START/MIDDLE/END` used for a column that follows a different convention), `resample_to_daily_agg` (sub-daily → one value per calendar day; `agg=`, `mincounts_perc=`), `resample_to_monthly_agg_matrix`, `timestamp_infer_freq_*` |
| `dv.variables` | `DaytimeNighttimeFlag`, `daytime_nighttime_flag_from_swinpot`, `TimeSince`, `potrad`, `potrad_eot`, `calc_vpd_from_ta_rh`, `aerodynamic_resistance`, `dry_air_density`, `et_from_le`, `latent_heat_of_vaporization`, `air_temp_from_sonic_temp`, `lagged_variants`, `classify_variable`, noise helpers |
| `dv.corrections` | `MeasurementOffsetFromReplicate`, `WindDirOffset`, `remove_radiation_zero_offset`, `remove_relativehumidity_offset`, `set_exact_values_to_missing`, `setto_threshold`, `setto_value`, `apply_corrections` (run an ordered list of `{"key","kwargs"}` corrections in one call — the single key→function mapping the GUI + `corrections_to_code` share) |
| `dv.qaqc` | `FlagQCF`, `StepwiseMeteoScreeningDb`, plus meteo-measurement helpers: `MEASUREMENTS`/`Measurement` (measurement codes + descriptions, e.g. `TA - air temperature`), `CORRECTIONS`/`CorrectionSpec`, `corrections_for_measurement(code)` (which corrections are physically meaningful for a measurement — radiation zero-offset for SW/PPFD, RH offset for RH, plus generics), `detect_measurement(varname)` (guess the measurement from a column name), `measurement_label`, `correction_spec` |

Top-level (no namespace): `load_exampledata_parquet`, `load_exampledata_parquet_lae`, `load_parquet`, `load_parquet_many` (read + `combine_first`-merge several parquet files; optional `progress_callback(phase, done, total, filepath)` — parquet counterpart to `MultiDataFileReader`), `save_parquet` (with `enforce_diive_format=True` → single header row + valid `TIMESTAMP_*` index name), `to_diive_format`, `ReadFileType`, `search_files`, `sstats`, `keep_vars` (non-destructive column subselection — column analogue of `times.keep_daterange`), `transform_yearmonth_matrix_to_longform`, `get_encoded_value_from_int`, `get_encoded_value_series`

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

Each level is a pure function — never mutate input. **Two entry points:**

**Single-call driver** — `run_chain(data, FluxConfig)` for the standard FLUXNET-style pipeline:

```python
from diive.flux.fluxprocessingchain import FluxConfig, init_flux_data, run_chain

cfg = FluxConfig(
    fluxcol='FC', ustar_thresholds=[0.18], ustar_labels=['CUT_50'],
    outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,
    gapfilling_features=['TA', 'SW_IN', 'VPD_kPa'],
    level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
    mds_swin='SW_IN', mds_ta='TA', mds_vpd='VPD_kPa',
)
data = init_flux_data(df, fluxcol='FC', site_lat=46.6, site_lon=9.8, utc_offset=1)
data = run_chain(data, cfg)
```

**Composable per-level callables** — for custom L3.2 pipelines or custom feature engineering:

```python
from diive.flux.fluxprocessingchain import (
    init_flux_data, run_level2, run_level31, run_level33_constant_ustar, run_level41_mds,
)

data = init_flux_data(df, fluxcol='FC', site_lat=46.6, site_lon=9.8, utc_offset=1)
data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None}, ...)
data = run_level31(data, gapfill_storage_term=True)
data = run_level33_constant_ustar(data, thresholds=[0.30])  # labels auto-generated as CUT_0
data = run_level41_mds(data, swin='SW_IN', ta='TA', vpd='VPD_kPa')
final_df = data.fpc_df
```

**Per-level signatures intentionally differ** (per-test dicts at L2, booleans at L3.1, pre-built object at L3.2, parallel lists at L3.3, built engineer + kwargs at L4.1). Shape matches what each level controls; see `diive/flux/fluxprocessingchain/__init__.py` docstring for the rule of thumb. `FluxConfig` is consumed only by `run_chain`, never by `run_level*`.

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
| `data.gap_stats(level='L33')` | `dict[str, GapStats]` | On-demand gap analysis; `{label: GapStats}` — label = level name for L2/L31/L32, USTAR scenario label for L33 |
| `data.plot_cumulative_comparison(..., showplot=True)` | `None` | Overlay cumulative sums of all gap-filling methods on one axes; pass `showplot=False` for headless |
| `data.plot_gapfilled_heatmaps(..., showplot=True)` | `None` | Side-by-side heatmaps: measured + one panel per gap-filling method; one figure per USTAR scenario |
| `data.levels.level41_methods()` | `dict[str, dict]` | Short keys: `'mds'`, `'rf'`, `'xgb'` (matches `gapfilled_cols()`) |

Key `data.levels` fields: `level2`, `level2_qcf`, `level31`, `level31_qcf`, `level32`, `level32_qcf`, `level33`, `level33_qcf`, `level41_mds`, `level41_rf`, `level41_xgb` (dicts keyed by ustar_scenario for L3.3+). **Flag column naming convention**: `FLAG_..._TEST` (individual quality tests, 0/1/2) and `FLAG_..._QCF` (level-overall aggregated flag) are both consumed by `FlagQCF` when aggregating; `FLAG_..._ISFILLED` (e.g. storage-correction provenance) is **informational only** and explicitly NOT consumed by QCF. L3.1 introduces no new quality test — its QCF re-aggregates L2-inherited flags on the storage-corrected target.

**Architecture notes:**

- L3.2 uses the multi-call pattern (stateful): `make_level32_detector(data)` → multiple `flag_outliers_*` + `addflag()` pairs → `run_level32(data, outlier_detector=sod)`. `run_level32` validates the detector is wired to the *current* `data` snapshot (raises if you rebuilt `data` without rebuilding the detector) and rejects detectors with no committed flags or with an uncommitted last test.
- `run_level41_rf` / `run_level41_xgb` take a pre-built `FeatureEngineer` instance.
- `finalize_level2/31/33()` are no-ops with `DeprecationWarning`.
- `LevelResults` is not `frozen=True` but treat as immutable — every level rebuilds it via `dataclasses.replace`. Don't mutate fields or `level41_*` dict keys in place.
- `add_driver(data, series, name=None)` puts a Series into `data.full_df` (where L4.1 reads from) instead of `data.fpc_df`; validates index, name, and absence of column collision.
- **Re-runs cascade.** Re-running L2/L3.1/L3.2/L3.3 on a `data` that already passed through that level drops the previous run's `fpc_df` columns and downstream `LevelResults` fields before producing fresh output (see `levels/_rerun.py`). Cascade: re-running level N invalidates N and every later level, because those levels' state was computed against the now-stale upstream. Columns are tracked in `data.added_columns: dict[idstr -> list[col]]`. L4.1 is per-method (`'L4.1_mds'`/`'L4.1_rf'`/`'L4.1_xgb'`) and additive across methods — each `run_level41_*` drops its own previous columns but leaves the other methods' results alone.

**Critical pitfalls:**

- MDS requires exact units: W/m² (radiation), °C (temp), **kPa (VPD)** — EddyPro outputs VPD in hPa; divide by 10. `run_level41_mds` warns when VPD median > 10 (likely hPa), TA median > 100 (likely Kelvin), TA median > 50, or SW_IN median > 2000.
- USTAR filtering applies ONLY to CO2/CH4/N2O; for H/LE use `thresholds=[0], threshold_labels=['CUT_NONE']`. `run_level33_constant_ustar` raises if a non-zero threshold is passed for an energy-flux basevar (`H2O`, `T_SONIC`, lowercase variants).
- L3.2 and L3.3 require L3.1; L3.3 also requires L3.2 (USTAR filtering must operate on outlier-screened data — `run_level33_*` raises if `level32_qcf` is None). For H/LE call `run_level31(data, set_storage_to_zero=True)`. `run_chain` runs L3.2 unconditionally; users who must skip it use the composable API.
- L4.1 features and MDS driver columns must exist in `data.full_df`, not `data.fpc_df`. Use `add_driver()` to add computed drivers to the right place.
- `init_flux_data` raises if `df` already contains `SW_IN_POT` / `DAYTIME` / `NIGHTTIME` (reserved names — would silently overwrite user data).
- `nighttime_accept_qcf_below` (was `nighttimetime_accept_qcf_below` before v0.91.0 — typo fixed).
- Default `daytime_accept_qcf_below=1` is stricter than FLUXNET convention of `2`; QCF=0 all pass, QCF=1 soft warning, QCF=2 hard failure.
- L3.3 USTAR filtering has three composable entry points: `run_level33_constant_ustar` (pre-known scalar threshold(s)), `run_level33_variable_ustar` (time-varying per-record threshold Series, e.g. per-year VUT — a constant threshold is just a constant Series, so the two share one flag path via `FlagMultipleVariableUstarThresholds`), and `run_level33_ustar_detection` (in-pipeline bootstrap detection). The detector's `mode=` selects the strategy: `'cut'` (constant pooled threshold → `CUT_16/50/84`) or `'vut'` (per-year threshold → `VUT_16/50/84`); CUT and VUT are **mutually exclusive** (the percentiles are the uncertainty scenarios within the chosen strategy). diive's VUT is smoothed over a 3-year window (`UstarBootstrapThresholds`); a year with no detected threshold falls back to its CUT value. `run_chain` only does CUT detection (`FluxConfig(ustar_detection_mode='bootstrap', ustar_bootstrap_ta_col=..., ustar_bootstrap_swin_col=...)`); VUT is composable-only. `threshold_labels` (constant path) is optional — auto-generates `CUT_0`, `CUT_1`, ... (positional index, **not** percentile); pass explicit labels like `['CUT_16', 'CUT_50', 'CUT_84']` for percentile-based thresholds. Length and uniqueness are validated; substring overlap (e.g. `CUT_5` inside `CUT_50`) is also rejected.
- `run_level33_ustar_detection` raises if `detector_kwargs` contains `nee_col` / `ta_col` / `ustar_col` / `swin_col` (these are set internally).
- `run_level41_*` emits a `UserWarning` when a re-run would overwrite previously stored scenarios in `levels.level41_*`.
- `FeatureEngineer(target_col='_target_', ...)` — `target_col` is a required placeholder; any string not in the feature list works.

**Example:** `examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py`

## Desktop GUI (`diive.gui`)

**[CRITICAL] Strict GUI ↔ library separation.** `diive/gui/` contains ONLY GUI code — Qt widgets, layout, rendering
glue, event handling, and presentation choices (colors, labels, styling). ALL algorithms and domain logic
(gap-filling, flux processing, variable classification, data computations, plotting math) live in the main library;
the GUI *calls* them, never reimplements them. The dependency arrow points one way: `gui` → library, never the reverse
(no other diive module may import from `diive.gui`). When building a GUI feature, if any piece of functionality would be
reusable, is domain knowledge, or is an algorithm, it belongs in the library — **notify the user and propose the move**
rather than putting it in `gui/`. (Examples already moved out: `cb_digits_after_comma='auto'` on `HeatmapDateTime`;
`dv.variables.classify_variable`.)

PySide6 desktop app. **Optional dependency** (`gui` extra, lazy-imported like `causal`) — never pulled into a headless
install. Launch: `uv sync --extra gui` then `diive-gui` (console script → `diive.gui._cli:_gui_main`). See
`diive/gui/README.md` for the file map. To ship it as a **standalone Windows app** (no Python/uv for end users), use the
PyInstaller one-folder build in `packaging/` (`build_gui.ps1`); see `packaging/README.md`.

- **Central appearance config: `diive/gui/theme.py`.** The `ThemeManager` singleton (`theme.manager`) holds the live,
  editable colours (`tokens`, `pills`, `new_pill`, `ts_colors`) and builds the stylesheet via `build_qss(tokens)`. It
  emits `changed` on edit; `apply()` re-applies the stylesheet app-wide. The **Appearance settings** tab (Tools menu)
  edits these with a live preview — the pill delegate and time-series plot read from `theme.manager` and repaint on
  `changed`; the Settings tab shows a sample variable list as a pill/highlight preview. (Which *variable name* maps to
  which pill kind stays in the library: `dv.variables.classify_variable` — kinds include the carbon fluxes `NEE`/`FC`/
  `GPP`/`Reco`/`FCH4`, the water fluxes `LE`/`ET`/`FH2O`, the nitrogen flux `FN2O`, plus radiation/meteo/soil kinds. Add
  a kind by adding a rule there *and* a colour entry in `theme.DEFAULT_PILL_STYLE`. Only colours/labels are GUI.)
  - **Single Studio look.** The GUI has one design — **Studio** (clean, minimal, VIBECAD-style: near-white surfaces,
    soft borderless panels, `icons.py` thin-line monochrome glyphs, uppercase tracked nav/section labels via
    `tracked_font`/`label_text`, and a frameless rounded window with `widgets/header_bar.py`'s inline-dropdown header).
    `theme.manager` holds `STUDIO_TOKENS` (editable live) and `STUDIO_TYPOGRAPHY`; `MainWindow`
    always builds the Studio chrome (`_build_studio_chrome`). Structural tokens (`CANVAS`/`INK`/`RADIUS`) are re-pinned
    from `STUDIO_TOKENS` in `load_dict`, so an old persisted config (incl. one from the removed Classic look — its
    `"preset"` key is ignored) can't shadow them while other colour overrides survive.

- **Shared variable list: `widgets/variable_panel.py` (`VariablePanel`).** Every tab's left-hand variable list MUST be
  this one component (filter + `VariableList` + `VariableDelegate` pills + fuzzy filtering) so styling, pills, and
  filtering are identical everywhere. Tabs differ only in how they react to `selected(name, ctrl_held)` and what they
  pass to `set_panels(...)`/`set_variables(...)`. Don't build ad-hoc variable lists. The filter is an **fzf-style fuzzy
  search**: separator-insensitive subsequence matching is the gate, but matches are *scored* (`_fuzzy_score`: contiguous
  runs, start-anchored, early hit, length-closeness) and the list **reorders best-match-first** while typing; clearing
  the field restores dataset order (tracked via `ORDER_ROLE`). Its width is a shared appearance
  setting (`theme.manager.list_width`, editable in Appearance settings) applied as a fixed width, so it's identical in
  every tab — don't set per-tab widths on it. `run_with_loading(name, fn)` shows a busy indicator (`LOADING_ROLE`:
  translucent wash + bottom bar) on the clicked variable plus a wait cursor, force-repaints it, then runs `fn` (the
  render) one tick later. matplotlib renders synchronously (blocks the event loop), so the indicator is a *static*
  busy cue painted before the freeze — it cannot smoothly animate. True animation would require off-thread Agg
  rendering (losing the interactive toolbar).
- **Splash screen.** `gui/splash.py` draws a `QSplashScreen` (subclass `_SplashScreen`) entirely with `QPainter` (no
  image assets, like the menu icons): a blue-teal gradient, the diive wordmark + `diive.__version__` + tagline, layered
  sine **waves**, a credits line, and a **rotating loading spinner** (a `QTimer` advances `_angle` and `repaint()`s; a
  12-spoke fading "comet" drawn in `drawContents`). For the spinner to actually animate, `run()` builds `MainWindow(cfg,
  autoload=False)` (UI only) and **defers the data load onto the event loop** via `QTimer.singleShot(0, …)` → `window.
  _initial_load()`; otherwise a blocking load in the constructor would freeze the spinner. After the deferred load it
  pumps `processEvents()` (drains the Overview's deferred first render) then `splash.finish(window)`. The same artwork
  backs **Help ▸ About** (`splash.show_about` → frameless modal `_AboutDialog`). `AUTHOR`/`SUPPORTERS` constants. Tests
  build `MainWindow()` with the default `autoload=True`, so they load synchronously in the constructor as before.
- **Startup load.** `MainWindow._initial_load()` reopens the **last project** (`config.last_project`, if the folder is
  still a diive project) via `_load_project_folder(announce=False)`, else loads the bundled example
  (`_load_example`, clean). `_open_project` is just the folder-picker wrapper around `_load_project_folder`.
- **Window sizing.** `MainWindow._size_to_screen()` sizes the window to ~88% of the available screen and centres it
  (adapts to resolution); Qt handles high-DPI scaling. Restored from saved geometry if present.
- **Persisted preferences.** `gui/config.py` saves/loads JSON (`QStandardPaths` config dir) on close/launch: theme
  (`ThemeManager.as_dict`/`load_dict`), site details (`site.manager.as_dict`/`load_dict`), window geometry, last-used
  filetype. Best-effort (failures swallowed).
- **Project settings store.** `gui/site.py`'s `SiteManager` singleton (`site.manager`) holds the project settings: the
  `author` name, a free-text `description`, and the measurement site's metadata (name, latitude, longitude, elevation,
  UTC offset) plus a `configured` flag; `update(...)` sets them and emits `changed`. The **Project settings** tab
  (`tabs/site.py`, class `ProjectSettingsTab` — `SiteDetailsTab` kept as a back-compat alias; `Settings ▸ Project
  settings`, single-instance) is a form that reads the values on build and writes them back on **Save**. Values only (no
  domain logic) — the GUI collects them here and passes them to library functions that take `lat`/`lon`/`utc_offset`
  (daytime/nighttime split, flux chain, ...). Persisted both with the GUI prefs (`config.py`) and inside a project
  (`app.py` `extras["site"]`), so author/description/site travel with a saved `.diive` folder. The tab's otherwise-empty
  right side holds a **notes wall** (`widgets/notes_wall.py`, `NotesWall`): a pinboard of draggable/resizable/recolourable
  sticky-note cards (bold header + body). The wall mirrors its `state()` into `site.manager.notes` on every edit (plain
  attribute set — no `changed` — to avoid a rebuild loop) and rebuilds only when the store's notes genuinely differ
  (project open), so notes ride along in `extras["site"]`/prefs with no `app.py` change. GUI-only (cards/colours/
  positions; card text colour is the WCAG-contrast pick) — no domain logic.
- **Registry-driven tabs.** `MainWindow` iterates `registry.TAB_CLASSES` (always-on tabs: Overview, Log) — it knows
  nothing about concrete tabs. Add a feature area = write a `DiiveTab` (`title` + `build()`) and append it. This is how
  the flux processing chain will plug in later.
- **Menu tabs are multi-instance.** Tabs opened from menus (`registry.MENU_TABS`, factories) open a NEW numbered
  instance each time (Heatmap 1, 2, 3 ...), all closable; tracked in `MainWindow._menu_tab_list`. Labels in
  `registry.SINGLE_INSTANCE_TABS` (e.g. Appearance) instead focus the existing one. Always-on tabs have their close
  button removed; `_next_menu_index` reuses the smallest free number. On close (`_on_tab_close`) focus falls back to the
  tab to the *left* of the closed one, but never the Log tab — if that's where it would land, it jumps to Overview.
- **Tab UX.** Tabs are movable (drag to reorder), renamable (**left** double-click → `_rename_tab`), and menu tabs carry a
  custom visible "×" (`icons.close_icon`); the always-on Overview/Log are not closable (a `tabCloseRequested`, incl.
  middle-click, for them is ignored). `tabBarDoubleClicked` fires for any button, so an `eventFilter` on the tab bar
  records the double-click button and `_rename_tab` ignores middle/right ones. In Studio chrome the tabs are Firefox-style pills with a favicon glyph. **Per-tab
  pin/freeze**: right-click a *menu* tab → Pin; pinned tabs (`MainWindow._pinned`) are skipped by `_push_data`, so they
  keep their dataset (cheap — references + pandas Copy-on-Write), and show a pin glyph (`icons.pin_icon`); unpin
  re-syncs. Overview/Log are never pinnable.
- **Events** (`gui/events.py` `EventManager` singleton `events.manager` — mirrors `site.manager`: live `list[Event]` +
  a `visible` toggle + a user **`categories` palette** (`{name: hex}`), `changed`/`categories_changed` signals,
  `as_dict`/`load_dict`). Model is the **library's** `dv.events` (`Event`,
  `event_to_flag`, `overlay_events`) — no domain logic in the GUI. **Data ▸ Add event…** (`MainWindow._add_event`) opens
  `widgets/add_event_dialog.py` (name/category — combo lists `events.manager.known_categories()` — three timing modes —
  single date/time, from/to, start + duration — calendar pickers, colour); its result method is `make_event()` (**not**
  `event()`, which would override `QDialog.event`). The full UI is the single-instance **Data ▸ Events** tab
  (`tabs/events.py`, `EventsTab`): a **reflowing board of modern event cards** (`widgets/flow_layout.py::FlowLayout`) on a
  soft-grey board, one per event — soft shadow, coloured left accent + category pill (stable per-category glyph), title +
  date settings, relative-time hint, a position-in-record mini bar (`_SpanBar`) and a description preview, **accented by
  category colour and sorted by start date** — double-click to edit, a **trashcan** delete (`icons.trash_icon`), a **⋯
  menu** (show on Overview / edit / duplicate / shift ±1 day), plus a **filter** field, **Group** (None/Category/Year) and
  **Density** (Comfortable/Compact) combos, a dashed `＋ Add event` ghost card, **Add event…**, **Manage categories…**
  (`widgets/category_dialog.py::CategoryDialog` — add / rename / recolour / remove categories in
  `events.manager.categories`, a palette seeded with generic `category1/2/3` where the **last one is undeletable**;
  `categories_changed` repaints cards + overlays via `MainWindow._on_event_categories_changed`, no flag-column rebuild).
  "Show on Overview" → `events.manager.request_focus(start,end)` → `MainWindow._focus_event_on_overview` →
  `OverviewTab.focus_on` zooms the linked datetime panels onto the event. `EventsTab.save_state`/`restore_state` persist
  the group + density. There's also a master **Show events on plots** checkbox
  (synced with the Data-menu checkable action, both driving `events.manager.visible`). A category colour overrides the
  library default on the cards **and** the plot overlays via `Event.resolved_color(i, colors=)` /
  `overlay_events(..., colors=events.manager.categories)`. `MainWindow._sync_event_columns` (connected to `events.manager.changed`)
  reconciles the dataset's `EVENT_<name>` **0/1 columns** to the event list — each event = one yes/no column (1 = took
  place), tracked in `_event_columns` so a reconcile never drops an `EVENT_`-named column that arrived as plain data;
  added columns join `_created` + get an "Event flag" origin. The Overview redraws on `changed` (`_overlay_events`):
  `overlay_events(axis='x')` on the time-series/cumulative/daily panels (labels on the time series only) and
  `axis='y'` on the heatmap (date on its y-axis). Events persist in GUI prefs (`config` `"events"`) and projects
  (`extras["events"]`); `_set_data` rebuilds them on the new index so they survive loads.
- **Select variables tab** (`tabs/variable_selector.py`, `Data ▸ Select variables`, single-instance) — a dual-list
  picker (two `VariablePanel`s: available ↔ selected) emitting a `subsetSelected` signal (same `QObject`-helper pattern
  as `featuresCreated`); `MainWindow._apply_var_subset` applies it **app-wide** (non-destructive). Like the date range,
  the chosen subset narrows `_data` via `dv.keep_vars` (full record stays in `_full_data`), so **every non-pinned tab**
  sees only the chosen variables — pinned (frozen) tabs are skipped by `_push_data` and keep their full dataset. **Data ▸
  Reset to all variables** (`_reset_var_subset`) clears it. The subset is maintained through edits (new features append,
  renames remap, deletes drop) and persisted in projects (`extras["var_subset"]`). The picker itself opts into the full
  record (`wants_full_data` → `MainWindow._data_for`) so it can always re-add dropped variables, and reflects the active
  subset on open (`set_active_subset`).
- **Rename variables tab** (`tabs/rename_variables.py`, `Data ▸ Rename variables`, single-instance) — adds a prefix
  and/or suffix to **all** variables with a live old→new preview table; **Apply** emits `variablesRenamed(mapping)` →
  `MainWindow._rename_variables` (renames columns in `_full_data`, remaps the `created` set, calls the library
  `MetadataStore.rename(mapping)`, re-derives the range). Double-clicking a table row routes one variable through the
  same flow via `metadata_store.manager.request_rename`. The prefix/suffix mapping is trivial string work (GUI); the
  DataFrame `rename` + metadata re-keying are the actual operations.
- **Rename / delete from any variable list.** The `VariablePanel` right-click menu offers **Rename…** and **Delete…**
  in every tab, routed through `metadata_store.manager` (`request_rename`/`request_delete` → `renameRequested`/
  `deleteRequested`, connected once in `MainWindow` — same singleton-relay pattern as `editRequested`, so no per-tab
  wiring). `_rename_one_variable` prompts + collision-checks; `_delete_variable` confirms + drops. Both non-destructive
  to the source file. (The old per-panel `deletable=True` gating was removed.) `VariablePanel` also exposes
  `scroll_to(name)` and `clear_filter()`.
- **Two-phase plot classes are GUI-ready.** The plotting tab renders diive plots straight into an embedded canvas via
  `Plot(series).plot(ax=canvas.ax, fig=canvas.fig)`; no GUI-specific plot variants needed.
- **`Plot` menu = one closable tab per method.** There is no single "Plotting" tab. Each plot method (Heatmap date/time,
  Heatmap year/month, Time series, Diel cycle, Cumulative year, Ridgeline, Scatter XY, Hexbin, Histogram, ...) is a menu-activated,
  closable `PlottingTab(plot_type, title)` instance, registered as a factory in `registry.MENU_TABS["Plot"]`. **All** menu
  entries (every menu, not just Plot) get a small `QPainter`-drawn glyph via `gui/icons.py::menu_icon(label)`
  (keyword-matched; `&` mnemonics stripped). Add a method = add a factory there + a branch in `plotting._draw_one` +
  matching controls in `plot_settings`. Most types pick **comparison panels** (Ctrl+click); Hexbin and Scatter XY instead
  pick variables by **X/Y/Z role** (`_XYZ_TYPES`, role readout via `settings.set_xyz`) — Scatter needs X+Y (Z optional for
  colour), Hexbin needs all three. The two heatmap kinds share `_HEATMAP_TYPES` for the side-by-side layout; the per-var
  line plots stack vertically. The **ridgeline** and the **histogram** are single-variable (Ctrl+click replaces, never
  stacks): the histogram is information-dense (bar counts + a z-score twiny axis + peak/info box), the ridgeline is the
  exception to the per-`ax` model — `RidgeLinePlot` builds its own stacked-density gridspec on the whole figure, gets
  `canvas.fig` via the class's `fig=` param, and the tab sets `canvas.auto_layout=False` so the constrained-layout
  freeze/resize machinery leaves its manual gridspec alone (see `_render_ridgeline`).
- **GUI-only presentation passes (no library param).** A few settings have no `plot()` kwarg and are applied *after* the
  diive plot renders (like the Overview's uniform-font pass): (1) the **Axes** group (`plot_settings._build_axes_group`)
  — X/Y limits, log X/Y, invert Y, additive grid — on the line/scatter types (`TIMESERIES`, `CUMULATIVE_YEAR`, `SCATTER`,
  Y-only for `DIELCYCLE`); `values()` carries them under the `_axes` key and `plotting._apply_axes(axes)` applies them.
  It is plot-type-aware: heatmaps/ridgeline have no `_axes` (no-op), the diel cycle's fixed 0–24 h x-axis is untouched
  (Y-only group). (2) The **Reverse colormap** checkbox (heatmap, year/month, hexbin, scatter) flips the `_r` suffix in
  `values()` (`_reverse_cmap`). Export **DPI** is a `MplCanvas` bottom-bar spinbox (default 150); `_SaveDpiToolbar`
  overrides the toolbar Save to write `savefig` at that DPI.
- **Params apply on a button, not live.** Editing a plot-settings control no longer re-renders; the tab's **Update
  plot** button (`PlottingTab.update_btn`, below the settings) reads `values()` and calls `_render()` on click — one
  consistent trigger for every control type (avoids the `QLineEdit.editingFinished` vs `QSpinBox.valueChanged`
  inconsistency, where typed fields only committed on Enter/focus-loss). The panel still emits `changed`, but the tab
  doesn't connect it to a render. **Variable selection stays live** (`_on_selected` → `run_with_loading`). In tests,
  set the controls then `tab.update_btn.click()` to apply (not `settings.changed.emit()`). For **project save/restore**
  the panel exposes `state()`/`apply_state()` — a *raw* positional snapshot of every control (field widgets of each
  `QFormLayout` row, so internal spin/combo editors are excluded; round-trips, unlike the transformed `values()`);
  `PlottingTab.save_state`/`restore_state` use it so a reopened plot has its exact colormap/limits/labels.
- **Data flow.** The `File` menu loads data via `OpenDataDialog` (parquet → `dv.load_parquet`, else `dv.ReadFileType`;
  multiple files → `MultiDataFileReader` / `dv.load_parquet_many` for parquet; reading + merge is library work, the
  dialog only calls it). When several files are selected the dialog shows a **per-file progress list** (one row +
  progress bar each), driven from the load worker thread via the library reader's `progress_callback`.
  `MainWindow` holds the current DataFrame and pushes it to every tab via the `DiiveTab.on_data_loaded(df, created)`
  hook; data-presenting tabs override it. Example data auto-loads on startup. **File ▸ Save data as parquet…** writes a
  diive-format parquet via `dv.save_parquet`; `app.to_diive_parquet_frame` enforces single-level columns (one header
  row) + a valid `TIMESTAMP_END/MIDDLE/START` index name (prompts if unset).
- **Date-range subselection (`Data` menu).** Non-destructive: `MainWindow` keeps the full record in `_full_data`; `_data`
  (what every tab sees) is `_full_data` optionally narrowed to `_range=(start,end)` via `dv.times.keep_daterange`.
  **Data ▸ Select date range…** opens `DateRangeDialog` (from/to pickers seeded + clamped to the data span) and sets
  `_range`; **Data ▸ Reset to full range** clears it. `_apply_range()` re-derives `_data`, updates the title with the
  active window, toggles the reset action, and pushes to tabs. Engineered features merge into `_full_data` (so they
  survive a range reset; out-of-range rows align to NaN). The slicing math is the library's `keep_daterange`, not the GUI.
- **Overview tab.** First tab, focused on every load (`setCurrentIndex(0)`). Top: variable list + a GridSpec figure
  (2×4: time series, `Cumulative` (`fill=True` shaded to zero), `DielCycle` (`each_month=True`, per-month colours +
  zero line), daily-mean time series via `dv.times.resample_to_daily_agg` (with a zero line), date/time heatmap;
  extensible via `_PANELS`). The variable name is a blue badge inside the time-series panel (not a figure title).
  Bottom: a compact, borderless **metrics ribbon** of `dv.sstats` values (`_StatItem`, hairline-separated) with
  per-stat hover descriptions from `SSTATS_DESCRIPTIONS`. `_StatCard` (boxed) is still used by the Gaps/Drivers/Seasonal
  tabs. **Linked zoom**: the three datetime panels (time series, cumulative, daily mean) share an x-axis (`sharex`) so
  zooming/panning one zooms all to the same window; an `xlim_changed` handler then **recomputes the diel cycle on the
  zoomed date range** (`dv.times.keep_daterange` → re-plot) and **clips the heatmap to that date range** (its date is on
  the y-axis, same date-number units, so a `set_ylim`). Repaint uses `MplCanvas.draw_idle()` (not `draw()`) so it never
  re-freezes the layout mid-resize; `_on_resize` runs **two** constrained-layout solve passes so the multi-panel zoomed
  layout converges instead of collapsing. **New-feature handling in `on_data_loaded`**: it diffs the incoming
  `created` set against the previous to find newly added columns and, for them, clears the list's fuzzy filter (a leftover
  filter would hide a non-matching new name — the cause of "added var doesn't show"), `scroll_to`s the row, and
  auto-selects/plots the new variable (skipping its `FLAG_…_TEST`). The Overview holds **no subset state of its own** —
  variable subsetting is app-wide in `MainWindow` (`_apply_var_subset`/`_var_subset`, narrowing `_data`), so the Overview
  just shows whatever columns it's pushed. The Overview's splitter handle is disabled (the var panel is fixed-width) so
  it shows no misleading ↔ resize cursor.
- **Hover tooltip.** `MplCanvas` attaches a `HoverAnnotator` (`widgets/hover.py`) in its constructor, so every embedded
  figure (Overview, plotting tabs) shows the value under the cursor with no per-tab wiring. Lines snap to the nearest
  sample (`np.searchsorted` on `get_xdata(orig=False)` — the unit-converted floats, **not** raw datetimes); `pcolormesh`
  heatmaps read the cell from `get_coordinates()`/`get_array()`. Renders by blitting (background re-captured on
  `draw_event`); GUI-only presentation, no domain logic.
- **Flux processing chain tab** (`tabs/fluxchain.py`, `Flux ▸ Flux processing chain`, single-instance). Guided
  Swiss-FluxNet chain, **Input → L2 → L3.1 → L3.2 → L3.3 → L4.1**, all on the **composable per-level** path
  (`init_flux_data` → `run_level2` → `run_level31` → `make_level32_detector`/`run_level32` →
  `run_level33_constant_ustar` / `run_level33_ustar_detection` → `run_level41_*`) — deliberately **not**
  `run_chain`/`FluxConfig`, so L3.2 stays a real
  inspected stepwise chain with its own QCF surface. **Layout is a horizontal pipeline rail**
  (`widgets/flux_pipeline_rail.py`: `PipelineRail` + `StageCard`, GUI-only): the six stages (Input › L2 › L3.1 › L3.2 ›
  L3.3 › L4.1) are selectable cards joined by chevrons, each with a level badge + a live **status pill** (`4 tests`,
  `2 scenarios`, `rf · mds`, `off`/`none`, or a red `warn` when required-but-empty); clicking a card swaps that stage's
  controls into a single inspector (`QStackedWidget`), so only the edited stage takes space and the canvas gets the rest.
  Status pills come from `_stage_statuses()` (refreshed via `_update_run_label`/`_wire_status_refresh`); a run lights
  every reached card with a green ✓ (`rail.set_reached_through(idx)`). **Levels run incrementally**: each inspector page
  has its own run button (`Initialize chain`, `Run Level 2`, … `Run Level 4.1`, via `_level_run_button(idx)`) that runs
  *only that level* on the evolving container `self._data` (`_run_level`/`_level_plan`/`_level_worker`), so each level's
  output feeds the next; a button unlocks only once its predecessor has run (`_update_level_buttons` gates on
  `self._reached`), and re-running an earlier level cascades the deeper state away (library re-run rule) and resets the
  reach. The action-bar **Run through Level N** button still runs the whole chain from scratch through the deepest
  configured level on a worker thread (ML gap-filling is slow); `_finalize(data, idx)` adopts the result and
  `_render_stage` shows that level's diagnostic. **Copy Python** is the standardized `CopyPythonButton(self._code)` (no
  code dump in the summary box — clipboard only, with the "Copied ✓" flash) emitting the exact reproducible composable
  script. Layout is **inspector (fixed-width, ~545px so the widest stage page fits — no horizontal overflow / left-clip)
  | heatmap canvas | a right text column** (run status over a monospace **QCF-report panel**): after a level runs,
  `_update_report` fills it with `qcf.screening_report()` (per-test retained/rejected, day/night split) for that level's `FlagQCF`
  (`_level_qcf` maps stage → `level2_qcf`/`level31_qcf`/`level32_qcf`/first `level33_qcf` scenario), with a **Copy
  report** button. **L2 detail:** each test has a **column picker** (`l2_cols`, one combo per input — two for the
  completeness test) seeded to the standard EddyPro-FLUXNET name (`_populate_l2_cols`) with a present/✗ availability
  marker (`_refresh_l2_availability`); a test with no valid column is disabled and dropped from the run. The chosen
  column flows through a **library** per-test override: `run_level2`'s test config dicts take an optional `'col'` key
  (`'expect_nr_col'`/`'basevar_nr_col'` for completeness), threaded down to the `flag_*_eddypro_test` functions (default
  `None` → the standard templated name); `_level2_settings` adds `col` only when the pick differs from the default
  (keeps run/codegen minimal). The eight **VM97 sub-tests** are individual toggles, and signal-strength exposes column +
  direction + threshold. The default column names + the VM97 sub-test list are library domain knowledge —
  `dv.flux.level2_test_inputs(fluxcol, fluxbasevar)` (templated names mirroring the `flag_*_eddypro_test` reads) and
  `VM97_SUBTESTS`; the GUI derives `fluxbasevar` via `detect_fluxbasevar` and only displays/gates. **The other levels
  mirror this column-info pattern** (`_refresh_levels_info`): Input has a **USTAR column** picker (`init_flux_data`'s
  `ustarcol`, so non-`USTAR`-named data can initialize) + availability marker; L3.1 shows the auto-detected **storage
  column** (`dv.flux.level31_storage_col(fluxcol)` — the `FC→SC_SINGLE` map, a library helper) + present/missing marker;
  L3.3 names the USTAR column it filters on. **L3.3 has two modes** (a Mode selector swaps the inspector page):
  **constant thresholds** (add value/label scenarios) or **moving-point detection** (`run_level33_ustar_detection`) with
  TA/SW_IN pickers, bootstrap iterations/workers/percentiles + detector params and an **Apply** selector **CUT**
  (constant pooled → `CUT_16/50/84`) vs **VUT** (per-year → `VUT_16/50/84`); the detected thresholds appear in the stage
  summary and Copy Python renders the chosen `mode`. L4.1 shows a SW_IN/TA/VPD driver-availability marker. **L4.1** ticks rf / xgb / mds (additive across methods — each
  replaces only its own previous result), picks rf/xgb predictor features + MDS SW_IN/TA/VPD driver columns (MDS units:
  VPD kPa, TA °C — the library warns on suspicious medians; driver auto-pick skips `FLAG_*` columns), fans out one
  gap-fill per USTAR scenario. **Reproducibility:** a shared **Random seed** field (default 42) is always passed to
  rf/xgb (`cfg["rf_kwargs"]/["xgb_kwargs"]["random_state"]`) — without it sklearn/xgboost reseed every run and the
  output drifts; it's also emitted in `level41_to_code`. The page also exposes **Reduce features (SHAP)** and per-method
  hyperparameters (RF `n_estimators`/`max_depth`, XGB `n_estimators`/`max_depth`/`learning_rate`, MDS `ta_tol`/`vpd_tol`);
  `_level41_cfg` omits any whose value still equals the library default (so the run is unchanged and the script stays
  minimal — only `random_state` is unconditional). **L4.1 layout:** each method's settings live in its own `QGroupBox`
  shown only when that method is ticked (`_sync_l41_visibility`) — a shared RF/XGB box (features + seed + reduce),
  separate RF / XGB hyperparameter boxes, and an MDS drivers+tolerances box — so the page is uncluttered (only enabled
  methods' params show). Features use `widgets/feature_picker.py::FeaturePicker` (a GUI-only **searchable checkable
  list** with a selected-count and All/None acting on the filtered subset — replaces the old ctrl-click multi-select).
  It renders the method comparison via `data.plot_cumulative_comparison(ax=...)` /
  `plot_gapfilled_heatmaps(fig=...)` (two-phase `ax=`/`fig=` embedding, like `RidgeLinePlot.plot(fig=...)`; the heatmaps
  view sets `canvas.auto_layout=False`). Script-gen lives in the library (`flux/fluxprocessingchain/codegen.py`:
  `chain_to_code` for the `run_chain`/`FluxConfig` path, `level2_to_code` … `level41_to_code` for the composable
  per-level path; all omit default-valued kwargs) — the GUI only calls it. Needs real EddyPro-FLUXNET input
  (`load_exampledata_parquet_lae_level1_30MIN`), not the default CH-DAV.
- **USTAR detection tab** (`tabs/ustar_detection.py`, `Flux ▸ USTAR detection`, single-instance). Standalone
  moving-point u\* threshold detection (Papale 2006), independent of the chain. Pick NEE/TA/USTAR/SW_IN + stratification
  params (TA/USTAR classes, forward-mode-n) and run a **single seasonal detection** (`UstarMovingPointDetection` →
  per-season + annual table) or tick **Multi-year bootstrap** (`UstarBootstrapThresholds`) for **VUT** (variable,
  per-year p16/p50/p84) + **CUT** (constant, pooled) thresholds, shown as a table + a diagnostic plot. Runs on a worker
  thread; all detection is library work (the small result plot is the only presentation in the tab — a candidate to
  move to a library plot helper if reused).
- **Feature engineering tab.** Menu-activated (`Data ▸ Feature engineering`, from `registry.MENU_TAB_CLASSES`) — not in
  the tab bar until selected, and closable (always-on tabs get their close button removed). Runs `FeatureEngineer`
  (library) on selected variables, emits new columns via a `featuresCreated` signal; `MainWindow` merges them, tracks
  them in a `created` set, re-pushes. Created columns get a pink **✦ NEW** pill (delegate `CREATED_ROLE`). Tab signals
  live on a `QObject` helper because `DiiveTab` is a plain `ABC`, not a `QObject` — class-level `Signal`s on a `DiiveTab`
  won't bind. When lazily creating a menu tab, call `tab.widget()` (builds it) **before** connecting `featuresCreated`,
  which `build()` sets.
- **Gap & coverage dashboard tab** (`tabs/gaps.py`, `Analyze ▸ Gaps & coverage`, single-instance). Diagnostics for
  missing data: stat cards + a two-panel **gap map** (availability heatmap + gap-spike timeline) + a long-gap table.
  All gap logic is the library's `dv.analysis.GapStats` — `.summary`, `.long_gaps`, the per-`ax` `plot_availability_heatmap`/
  `plot_gap_spike_timeline`, and `gap_at(timestamp)` (new: maps a click to the containing/nearest gap). The tab only
  arranges widgets and wires the **clickable** interactions both ways (table row → highlight overlay on the timeline;
  timeline click → `gap_at` → highlight + select the matching row), with a `_syncing` guard against selection echo.
  Defaults to the gappiest variable (`df.isna().sum().idxmax()`) so it's useful on open. The long-gap threshold spinbox
  re-runs `GapStats` (cheap). **Note:** the library's `plot_*` panel methods attach their colorbar via `ax.figure.colorbar`
  (not `plt.colorbar`, which targets pyplot's current figure) so they embed correctly in the GUI canvas.
- **Driver explorer tab** (`tabs/drivers.py`, `Analyze ▸ Driver explorer`, single-instance). "What relates to this
  variable, and at what lag?" Pick a target; a ranked table lists every other variable by correlation strength, click a
  driver to see the target-vs-driver scatter. The ranking + lag scan is the library's new `dv.analysis.rank_drivers(df,
  target, method=, max_lag=)` → DataFrame `[DRIVER, CORR, ABS_CORR, BEST_LAG, N]` (positive `BEST_LAG` = driver leads
  the target); the scatter is `dv.plotting.ScatterXY`. The tab only collects target/method/max-lag, fills widgets, and
  renders the selected driver's scatter (shifted by its `BEST_LAG`) — no stats of its own. Target selection is live;
  method/max-lag apply on a **Rank drivers** button (the lag scan can be heavier). Table sorts numerically via a small
  `_NumItem` (sorts on the stored value, not display text). Defaults to `NEE_CUT_REF_f` (a continuous flux makes the
  ranking informative), else the first numeric column.
- **Seasonal-trend & anomaly explorer tab** (`tabs/seasonaltrend.py`, `Analyze ▸ Seasonal-trend & anomalies`,
  single-instance). "Is this variable changing over the years?" Decomposes a variable's **daily-mean** series into
  trend/seasonal/residual (4 stacked panels) and, in a second **view**, shows each year's anomaly vs a reference period.
  Maths is the library's: `dv.times.resample_to_daily_agg` → `dv.analysis.SeasonalTrendDecomposition` (STL/classical/
  harmonic) → `dv.plotting.LongtermAnomaliesYear`. Decomposes at the annual period (365); STL runs `robust=False` +
  `seasonal_jump=trend_jump≈12` for sub-second speed (a "Robust" checkbox opts into the slower outlier-resistant fit).
  Variable selection + view + reference-year changes re-render live; method/robust apply on **Update** (STL is the
  expensive recompute). Degrades gracefully on <2 years of data (annual STL needs two cycles) — shows a message and
  keeps the anomaly view working. **Library bug fixed along the way:** the STL wrapper (`core/times/decomposition_utils.
  py::stl_decompose`) never passed `period` to statsmodels and called `STL.fit(weights=...)` (unsupported) — so STL
  always raised on real data. Now passes `period` + a small odd seasonal smoother and `fit()` without weights; classical/
  harmonic were already fine.
- **Spectrogram tab** (`tabs/spectrogram.py`, `Analyze ▸ Spectrogram`, single-instance). Time-frequency view: how the
  strength of a variable's cycles (esp. the 1/day diel rhythm) changes over the record. Wraps `dv.analysis.spectrogram`;
  the tab maps the result onto **calendar-time × cycles-per-day** axes (records-per-day inferred from the index spacing;
  segment centres mapped to real timestamps via the non-NaN sample index, so the x-axis is correct even across gaps) and
  shows a plain-language **explanation label** of what a spectrogram is (the user asked for it). Window length / overlap /
  window-function apply on **Update** (recompute); max-cycles-per-day (y-limit) and colormap are live re-renders. No
  signal processing in the GUI.
- **Outlier tabs** (`Outliers` menu) — a family of `BaseOutlierTab` (`tabs/_outlier_base.py`) subclasses, one per
  library detector: **Absolute limits** (`tabs/outliers_absolutelimits.py`), **Hampel** (`tabs/outliers.py`),
  **Local SD** (`tabs/outliers_localsd.py`), the three z-score methods **Z-score** / **Z-score (rolling)** /
  **Z-score (increments)** (`tabs/outliers_zscore.py` / `outliers_zscorerolling.py` / `outliers_zscoreincrements.py`),
  **Local Outlier Factor** (`tabs/outliers_lof.py`), **Trim-low** (`tabs/outliers_trim.py`), and **Manual removal**
  (`tabs/outliers_manualremoval.py`). Each runs its detector on a worker thread and keeps the
  **original** (untouched), the **cleaned** series (`{var}_{METHOD_SUFFIX}`, outliers→NaN), and the **flag**
  (`FLAG_{var}_OUTLIER_{flagid}_TEST`, 0/2 — note the flag id comes from the *library* class, so it can differ from the
  cleaned-column suffix, e.g. `ZSCOREINCREMENTS` cleaned column but `FLAG_..._OUTLIER_INCRZ_TEST`). Two stacked preview
  panels: top = original + red outlier markers, bottom = cleaned. They share the time x-axis (`sharex`) but **not** y —
  the cleaned panel autoscales to the outlier-free range. **Add cleaned + flag to dataset** emits the columns via a
  `featuresCreated` signal (the feature-engineering merge mechanism). All detection is library work; a subclass supplies
  only param widgets, kwargs, the detector, and codegen.
  - **Detector interface contract.** `BaseOutlierTab._worker` requires the library detector to expose `.run(repeat,
    progress_callback)` (→ `.calc(...)`), `.filteredseries`, `.overall_flag`, `.last_lower_bound`/`.last_upper_bound`
    (per-iteration detection band in **data units**, for the optional limit-line overlay), and `.is_daytime` (day/night
    mode). When adding a GUI tab for another `dv.outliers` detector, verify/extend the library class to this contract —
    do **not** reimplement detection in the GUI. (zScore/zScoreRolling/zScoreIncrements/AbsoluteLimits/LOF were extended
    to it. `ManualRemoval` (`tabs/outliers_manualremoval.py`) satisfies the `.run(repeat,
    progress_callback)` part — `repeat` is accepted-but-ignored since manual removal is index-based and would never
    converge — but has **no detection band** (`last_lower_bound`/`last_upper_bound` exposed as `None`, like the increments
    method — selection is a set of timestamps, not a `[lower, upper]` envelope) and **no day/night mode** (`is_daytime`);
    its tab sets `supports_daynight = False` + `supports_repeat = False` and skips the band overlay. `supports_repeat`
    (new) hides the "Repeat until no more outliers" checkbox; `run_label` overrides the run button text. `TrimLow`
    (`tabs/outliers_trim.py`) satisfies `.run(repeat, progress_callback)` (`calc` forwards both to `repeat`; default
    single-pass) and exposes `.filteredseries`/`.overall_flag`/`.is_daytime` plus `last_lower_bound`/`last_upper_bound`
    `= None` (the symmetric positional trim rejects the lowest values plus an equal count of the highest, so the kept set
    is not a single `[lower, upper]` envelope — like the increments method, no band is drawn). Its tab sets
    `supports_daynight = False` because TrimLow's day/night split is method-specific **and opt-in**: by default
    (`trim_daytime=trim_nighttime=False`) the *whole series* is trimmed against one distribution and **no coordinates are
    needed** (`lat`/`lon`/`utc_offset` default `None`; they're validated only when a split is requested). The optional
    **Trim daytime only**/**Trim nighttime only** checkboxes pick which period to screen, each against its own
    distribution — so the coords + checkboxes live in the method rows (prefixed `trim_…` so they don't collide with the
    base's `self.lat/lon/utc`, which it nulls when `supports_daynight = False`); the coordinate widgets enable only when a
    side is ticked, and `lower_limit` is seeded from the variable's minimum on selection. (Constructor arg order is
    `series, lower_limit, trim_daytime, trim_nighttime, lat, lon, utc_offset, …` — `lower_limit` is the primary param;
    callers use keywords.))
  - **Two class flags gate optional UI.** `supports_daynight = False` omits the whole day/night box (rolling & increment
    z-score have no day/night mode). `band_center_label = "<text>"` draws the detection band's **centre** (its midpoint —
    the rolling mean/median the band is symmetric around) as a solid line beside the dashed/dotted limits, labelled with
    that text (set on the rolling z-score tab; the band is `rolling_mean ± t·rolling_std`, so the midpoint *is* the
    rolling mean — derived GUI-side from the bounds, no extra library output).
  - **No limit band for the increments method (by design).** `zScoreIncrements` leaves `last_lower_bound`/`upper_bound`
    `None`: it flags a point only when the z-scores of all three increments (forward/backward/combined) exceed the
    threshold, so the accepted region is a *union* of intervals, not a single `[lower, upper]` envelope — there is no
    faithful data-unit band to draw, and the base skips the overlay when the bounds are `None`.
- **Per-variable metadata (tags + provenance).** Model is the **library's** `diive.core.metadata` (`VariableMetadata`,
  `ProvenanceEntry`, `MetadataStore`, `provenance_attr`, `ATTRS_KEY`, `truncate_words`, `MAX_DESCRIPTION_WORDS=50`) —
  headless, no Qt, no wall-clock (timestamps are passed in). The GUI holds it app-wide in `gui/metadata_store.py`'s
  `manager` singleton (mirrors `theme.manager`/`site.manager`; emits `changed`). Each variable carries an **origin**
  (`original`/`modified`/`derived`), parent links, an ordered **provenance** list, free-text **description** (≤50 words),
  and **tags** that track their source (only `USER` tags persist). **Operations emit provenance with no signal change**
  by attaching `df.attrs[ATTRS_KEY]` (built via `provenance_attr`) to the frame they pass to `featuresCreated`;
  `MainWindow._add_features` consumes it (`store.from_attrs`, stamping the time GUI-side). The outlier base
  (`tabs/_outlier_base.py`) and feature tab tag their outputs this way; `record_original` seeds each loaded column with
  an "Imported from <source>" history entry. **Cumulative history**: `record_derived` makes a new column **inherit its
  parent's full provenance** on first creation (a *copied* snapshot — not shared, so a later rename can't alias-mutate
  it), so `FC → FC_LOCALSD → FC_LOCALSD_HAMPEL` shows all three steps, not just the last. `MetadataStore.rename(mapping)`
  re-keys records and rewrites parent + provenance links so lineage survives a rename. **Tolerant deserialization** (for
  older `.diive` projects): `VariableMetadata.from_dict` accepts an aliased/missing name and either the `{tag: source}`
  dict or an older bare tag list (`_coerce_tag_sources`), and `load_dict` skips malformed entries instead of failing the
  whole load. **Display:** the delegate paints a ★ (favorite) + `●N` (extra-tag count)
  and `VariablePanel` sorts favorites to the top; `VariableList` shows a rich hover tooltip; right-click renames/deletes
  the variable and edits tags. The
  **Data ▸ Metadata explorer** tab (`tabs/metadata_explorer.py`, single-instance) does full editing (origin badge,
  auto-coloured tag chips via `theme.tag_color`, the 50-word note, provenance timeline, a confirmed **Clear all tags
  & notes** footer button → `MetadataStore.clear_user_data()`, and a per-variable right-click **Remove all tags & note**
  → `clear_variable_user_data()`; both drop user tags + descriptions but keep origin/provenance/function tags. The
  per-variable clear is offered both as a detail-panel button and a list right-click — gated by
  `VariablePanel(clearable=True)` and routed through its `clearRequested` signal so the explorer can unbind the open
  note editor before the store changes; other tabs' panels don't offer it). Every *other* tab's variable-list
  right-click has an **Edit metadata…** entry → `manager.request_edit(name)` → `manager.editRequested` →
  `MainWindow._edit_metadata` opens/focuses the explorer and calls its `select_variable(name)`. **Persistence is
  namespaced by
  dataset:** `config.variable_metadata` is `{dataset_key(source): {"tags":…, "descriptions":…}}` — so the same column
  name in two datasets keeps separate tags. `MainWindow._set_data` stashes the outgoing dataset's `user_data()` before
  reset and loads the incoming one; `_namespace_metadata()` migrates older non-namespaced configs onto the first dataset.
  `_set_data(persist_metadata=False)` loads a **clean** dataset — no saved tags/notes applied or kept, and any stale
  entry for that source is purged; the bundled example auto-load uses it so the example always opens pristine.
- **Projects.** A *diive project* is a self-contained `<name>.diive` folder: `__diive__` marker, `project.json`
  manifest, `data.parquet`. Format is the **library's** `diive.core.io.project` (`save_project`, `load_project`,
  `is_project`, `DiiveProject`, `project_name_to_dirname`) — it serializes the **full** `MetadataStore`
  (`to_dict`/`from_dict`: origin, parents, provenance, tags+sources, notes — richer than the lightweight per-session
  config) plus an opaque `extras` dict (so the library never imports GUI types). The GUI (`File ▸ Save project` Ctrl+S /
  `Save project as…` / `Open project…`, dialog `widgets/save_project_dialog.py`) puts site (`site.manager.as_dict`),
  active date range, `created` columns, and the **open menu tabs** (`_open_tabs_state` → label/title/pinned **+ each tab's
  `save_state()`**; restored by `_close_all_menu_tabs` + `_restore_tabs` → `tab.restore_state()` after the data push).
  Every `DiiveTab` has `save_state()`/`restore_state()` (default no-op) capturing its *inputs only* — selected
  variable(s) + control values (via `widgets/state_utils.save_controls`/`restore_controls`); heavy results re-compute on
  restore. Both wrapped in try/except so one tab's quirk can't break save/load. `extras` also carries `overview` (the
  always-on Overview's selected variable + active subset, via its own `save_state`/`restore_state`) and `active_tab`
  (the focused tab's title, refocused by `_restore_active_tab` after the workspace is rebuilt).
  `MainWindow._open_project` loads data clean
  (`_set_data(persist_metadata=False)`) then overlays the project's metadata via `store.load_dict`; a plain load clears
  `_project_dir` so Ctrl+S can't overwrite a project with unrelated data. `_save_project` updates the open project in
  place, else prompts.
- **Var list sync.** All tabs refresh via `MainWindow._push_data()` → `on_data_loaded(df, created)` on every data
  change; menu tabs get current data on open and are dropped from the push list on close.
- **Output console.** The `Log` tab (`LogTab` → `ConsolePanel`) mirrors diive's Rich output in colour. The library tees
  output to any sink registered via `add_console_sink` (`diive.core.utils.console`); the panel renders the ANSI stream.
  The redirect hook is library-owned; the panel only renders (separation rule).

**PySide6 gotchas (already handled in code — don't reintroduce):**

- **Retain tab instances** (`MainWindow._tabs`). Qt owns the QWidgets, but the Python `DiiveTab` objects hold the
  signal slots; if GC'd, their signals silently go inert (symptom: clicks stop working after startup).
- **A stylesheet touching `QListWidget::item` disables per-item `setBackground`/`setForeground`.** Row colouring goes
  through a `QStyledItemDelegate` (`VariableDelegate`), not item roles. The delegate also draws the NEE/GPP/Reco pills.
- **Matplotlib's Qt toolbar recolours icons from the widget palette.** `MplCanvas` sets a light palette *before*
  building the toolbar (else icons render white-on-white on dark system themes).
- **Use synchronous `canvas.draw()`, not `draw_idle()`,** after a user action so the canvas repaints immediately.
- **Share axes for comparison panels** via `subplots(..., sharex=True, sharey=True)` so pan/zoom is synchronised.
- **A widget with its own stylesheet detaches its tooltips from the app-wide `QToolTip` rule** — they fall back to the
  dark, low-contrast system default (and bare `color`/`background` leak in, making text barely visible). Append
  `theme.manager.tooltip_qss()` to that widget's `setStyleSheet(...)` to re-attach the light card look. **The appended
  string must be all-selector QSS** — `setStyleSheet("color: x; background: y;" + tooltip_qss())` silently drops the
  `QToolTip` block because Qt can't parse bare declarations mixed with a rule block; wrap the bare props in a selector
  first: `setStyleSheet("QLabel { color: x; background: y; }" + tooltip_qss())`. Reference fixes: the Overview stat
  items (`tabs/overview.py`) and the event cards (`tabs/events.py`).

## High-Resolution EC Analysis (hires)

Tools for 10/20 Hz data. Workflow: `raw 20 Hz → WindDoubleRotation → reynolds_decomposition → flux`

```python
wr = dv.flux.WindDoubleRotation(u=df['u'], v=df['v'], w=df['w'])
w_prime = dv.flux.reynolds_decomposition(wr.w2)   # use rotated w2, not raw w
c_prime = dv.flux.reynolds_decomposition(df['CO2'])
flux = (w_prime * c_prime).mean()
```

**Classes:** `WindDoubleRotation`, `reynolds_decomposition`, `MaxCovariance`, `FluxDetectionLimit`, `PreWhiteningBootstrap`, `PwbBatchDetection`, `TlagApplier`, `PerFilePipeline`, `process_one_file`.

### Time-lag detection & removal (PWB)

Three CLIs (console scripts in `pyproject.toml`), all requiring **wind-rotation-corrected** W for detection (or done in-memory):

| CLI | Class | Does |
|---|---|---|
| `diive-tlag-pwb-batch` | `PwbBatchDetection` | Detect lags across many averaging-period files → `tlag_results.csv` (+ PWBOPT S1/S2/S3 columns) |
| `diive-tlag-apply-batch` | `TlagApplier` | Apply lags from a `tlag_results.csv` to raw files (shift scalars by `round(tlag_s·hz)`) |
| `diive-tlag-pwb-detect-remove` | `PerFilePipeline` | **Two-phase per-chunk** detect+remove in one run |
| `diive-tlag-pwb-detect-remove-tui` | `DetectRemoveTUI` | Textual TUI wrapping `PerFilePipeline`; `--demo` previews it without data |

`diive-tlag-pwb-detect-remove` ([detect_and_remove_tlag.py](diive/flux/hires/detect_and_remove_tlag.py)) splits each long raw file into fixed-length chunks (`--chunk-seconds`, default 30 min): **phase 1** rotates each chunk in memory + runs PWB per scalar (no write); **PWBOPT** picks the best lag per chunk across the full sequence; **phase 2** shifts each scalar by that lag (`--lag-column-template`, default `{prefix}_tlag_final_pf_s` — the same column `TlagApplier` removes, NOT raw `tlag_s`) and writes one file per chunk. Parallel unit is one chunk; chunk count is measured **per file**. Everything is parameterized in seconds × `--hz`, so 10 Hz / 60-min chunks just need `--hz 10` / `--chunk-seconds 3600` (one uniform format per run). Output is numbered by phase: `1_lag_detection/` (summary CSV, checkpoints, plots/, plots_summary/) and `2_lag_removed/` (corrected chunk files — clean input for the next flux step); root holds only those two folders + `log.txt`. **Downstream flux processing must run with EC time-lag maximization disabled.**

- Per-chunk output filenames come from `--chunk-name-template` ({stem}/{suffix}/{index}/{starttime}); `{starttime}` needs `--start-time-regex` + `--start-time-format` and names each chunk by its own start time (e.g. `CH-CHA_{starttime}{suffix}` → `..._202107271330.csv`). The output line terminator defaults to `--lineterm auto` (reproduces the input file's CRLF/LF; header lines normalised to match — never mixed).
- `PerFilePipeline.run(cancel_event=threading.Event())` is cooperative-cancellable: pending chunks cancelled, in-flight ones finish, remove phase skipped if cancelled during detect; `pipeline.cancelled` reports it. `run()` writes the summary CSV + overview plots itself (so TUI/CLI/Python callers all produce them).
- **TUI** (`DetectRemoveTUI`, `--demo` for no-data preview): full CLI-option coverage with per-field tooltips + focus help; **Check** preflight (file count, header columns listed + verified, chunk plan); **Stop** (cancel); **Open output folder**; **▾ column picker** (scan first file's header, pick the exact bracketed name); auto-preflight on Run; overwrite guard; live field validation; per-worker animated spinner rows; log lines show `parent › chunk`; the lag-removal phase is shown as **"align"** (paper's "temporal alignment"), not "remove".

## Outlier Detection & QC

- **Single method:** `dv.outliers.Hampel(series).run()`
- **Chained:** `dv.outliers.StepwiseOutlierDetection()` (orchestrates multiple methods)
- **Corrections:** `dv.corrections.MeasurementOffsetFromReplicate()`, `remove_radiation_zero_offset()`, `setto_*()`
- **QCF aggregation:** `dv.qaqc.FlagQCF()` → 0 (good) / 1 (marginal) / 2 (poor)
- **Full pipeline:** `dv.qaqc.StepwiseMeteoScreeningDb()` — corrections → outlier detection → quality flags
- **Timestamp shift:** three methods comparing measured vs. theoretical radiation (requires clear days)

**[CONVENTION] Day/night threshold parameters.** Outlier methods that support `separate_day_night` follow one rule so
the GUI and API behave predictably — **all new day/night-capable methods MUST match it:**

1. **A single global knob (`n_sigma`, `threshold`, …) is the source of truth.** Per-period overrides
   (`n_sigma_daytime` / `n_sigma_nighttime`, etc.) **default to `None`, never to a literal**, and fall back to the
   global value when unset: `self.x_daytime = x_daytime if x_daytime is not None else x`. Defaulting a per-period
   parameter to a literal (the old `Hampel` bug: `n_sigma_daytime=5.5`) silently shadows the global knob, so the global
   parameter appears to "do nothing" in day/night mode. Verify that changing the global value alone changes the result.
2. **`separate_day_night` only changes results when the day and night thresholds differ.** With equal thresholds it is
   mathematically identical to no separation. So a GUI exposing the feature should expose *per-period* thresholds (seeded
   from the global value, then independently editable), not just the toggle. The Hampel tab (`gui/tabs/outliers.py`) is
   the reference pattern: separate daytime/nighttime sigma fields + red/blue day/night outlier markers + a day/night
   count in the status line.

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
pytest tests/test_gui.py -v                      # Desktop GUI (offscreen, needs 'gui' extra)
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
| Textual `App` already has internal `_running`/`_workers` attrs | Don't name your own App attributes `_running` (use e.g. `_busy`); Textual sets `_running=True` on mount, silently breaking your guards |
| Textual `@work` method not starting when called from a non-handler context | Dispatch background work with `threading.Thread(target=…, daemon=True)`; `call_from_thread` delivers UI updates from any thread |

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

**Last Updated:** 2026-06-12 | **Version:** v0.91.0 | **Package Manager:** `uv`
