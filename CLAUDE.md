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

`import diive as dv` exposes 9 domain namespaces:

| Namespace | Contents |
|---|---|
| `dv.outliers` | `AbsoluteLimits`, `Hampel`, `LocalSD`, `LocalOutlierFactor`, `zScore`, `zScoreRolling`, `zScoreIncrements`, `TrimLow`, `ManualRemoval`, + daytime/nighttime variants |
| `dv.gapfilling` | `RandomForestTS`, `XGBoostTS`, `SWINGapFillerXGBoost`, `FluxMDS`, `QuickFillRFTS`, `OptimizeParamsRFTS`, `OptimizeParamsTS`, `LongTermGapFillingRandomForestTS`, `LongTermGapFillingXGBoostTS`, `FeatureEngineer`, `GapFillingResult`, `prediction_scores`, `linear_interpolation` |
| `dv.flux` | `FluxConfig`, `FluxLevelData`, `run_chain`, `init_flux_data`, `add_driver`, `WindDoubleRotation`, `reynolds_decomposition`, `MaxCovariance`, `PreWhiteningBootstrap`, `PwbBatchDetection`, `TlagApplier`, `PerFilePipeline`, `process_one_file`, `FluxDetectionLimit`, ustar classes. Per-level `run_level*` and `make_level32_detector` live in `diive.flux.fluxprocessingchain`. |
| `dv.analysis` | `DailyCorrelation`, `GrangerCausality`, `StratifiedAnalysis`, `GapFinder`, `GapStats`, `GridAggregator`, `Histogram`, `FindOptimumRange`, `SeasonalTrendDecomposition`, `BinFitterCP`, `harmonic_analysis`, `percentiles101` |
| `dv.analysis.experimental` | **(provisional, API may change)** `DriverAnalysis`, `DriverAnalysisResult`, `AleCurve`, `Ale2DResult`, `accumulated_local_effects`, `accumulated_local_effects_2d`, `ExperimentalWarning` — evidence-triangulation driver attribution; emits a one-time `ExperimentalWarning` on use |
| `dv.plotting` | `HeatmapDateTime`, `HeatmapXYZ`, `HeatmapYearMonth`, `HexbinPlot`, `ScatterXY`, `TimeSeries`, `DielCycle`, `RidgeLinePlot`, `HistogramPlot`, `ShiftedDistributionPlot`, `Cumulative`, `CumulativeYear`, `LongtermAnomaliesYear`, `TreeRingPlot` |
| `dv.times` | `TimestampSanitizer`, `DetectFrequency`, `resample_to_monthly_agg_matrix`, `timestamp_infer_freq_*` |
| `dv.variables` | `DaytimeNighttimeFlag`, `daytime_nighttime_flag_from_swinpot`, `TimeSince`, `potrad`, `potrad_eot`, `calc_vpd_from_ta_rh`, `aerodynamic_resistance`, `dry_air_density`, `et_from_le`, `latent_heat_of_vaporization`, `air_temp_from_sonic_temp`, `lagged_variants`, `classify_variable`, noise helpers |
| `dv.corrections` | `MeasurementOffsetFromReplicate`, `WindDirOffset`, `remove_radiation_zero_offset`, `remove_relativehumidity_offset`, `set_exact_values_to_missing`, `setto_threshold`, `setto_value` |
| `dv.qaqc` | `FlagQCF`, `StepwiseMeteoScreeningDb` |

Top-level (no namespace): `load_exampledata_parquet`, `load_exampledata_parquet_lae`, `load_parquet`, `save_parquet`, `ReadFileType`, `search_files`, `sstats`, `transform_yearmonth_matrix_to_longform`, `get_encoded_value_from_int`, `get_encoded_value_series`

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
- `run_level33_constant_ustar` only supports constant thresholds; for in-pipeline bootstrap detection use `run_level33_ustar_detection` (composable) or `FluxConfig(ustar_detection_mode='bootstrap', ustar_bootstrap_ta_col=..., ustar_bootstrap_swin_col=...)` (via `run_chain`). `threshold_labels` is optional — auto-generates `CUT_0`, `CUT_1`, ... (positional index, **not** percentile); pass explicit labels like `['CUT_16', 'CUT_50', 'CUT_84']` for percentile-based thresholds. Length and uniqueness are validated; substring overlap (e.g. `CUT_5` inside `CUT_50`) is also rejected.
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
`diive/gui/README.md` for the file map.

- **Central appearance config: `diive/gui/theme.py`.** The `ThemeManager` singleton (`theme.manager`) holds the live,
  editable colours (`tokens`, `pills`, `new_pill`, `ts_colors`) and builds the stylesheet via `build_qss(tokens)`. It
  emits `changed` on edit; `apply()` re-applies the stylesheet app-wide. The **Appearance settings** tab (Tools menu)
  edits these with a live preview — the pill delegate and time-series plot read from `theme.manager` and repaint on
  `changed`; the Settings tab shows a sample variable list as a pill/highlight preview. (Which *variable name* maps to
  which pill kind stays in the library: `dv.variables.classify_variable`. Only colours/labels are GUI.)

- **Shared variable list: `widgets/variable_panel.py` (`VariablePanel`).** Every tab's left-hand variable list MUST be
  this one component (filter + `VariableList` + `VariableDelegate` pills + subsequence filtering) so styling, pills, and
  filtering are identical everywhere. Tabs differ only in how they react to `selected(name, ctrl_held)` and what they
  pass to `set_panels(...)`/`set_variables(...)`. Don't build ad-hoc variable lists. Its width is a shared appearance
  setting (`theme.manager.list_width`, editable in Appearance settings) applied as a fixed width, so it's identical in
  every tab — don't set per-tab widths on it. `run_with_loading(name, fn)` shows a busy indicator (`LOADING_ROLE`:
  translucent wash + bottom bar) on the clicked variable plus a wait cursor, force-repaints it, then runs `fn` (the
  render) one tick later. matplotlib renders synchronously (blocks the event loop), so the indicator is a *static*
  busy cue painted before the freeze — it cannot smoothly animate. True animation would require off-thread Agg
  rendering (losing the interactive toolbar).
- **Window sizing.** `MainWindow._size_to_screen()` sizes the window to ~88% of the available screen and centres it
  (adapts to resolution); Qt handles high-DPI scaling.
- **Registry-driven tabs.** `MainWindow` iterates `registry.TAB_CLASSES` (always-on tabs: Overview, Log) — it knows
  nothing about concrete tabs. Add a feature area = write a `DiiveTab` (`title` + `build()`) and append it. This is how
  the flux processing chain will plug in later.
- **Menu tabs are multi-instance.** Tabs opened from menus (`registry.MENU_TABS`, factories) open a NEW numbered
  instance each time (Heatmap 1, 2, 3 ...), all closable; tracked in `MainWindow._menu_tab_list`. Labels in
  `registry.SINGLE_INSTANCE_TABS` (e.g. Appearance) instead focus the existing one. Always-on tabs have their close
  button removed; `_next_menu_index` reuses the smallest free number.
- **Two-phase plot classes are GUI-ready.** The plotting tab renders diive plots straight into an embedded canvas via
  `Plot(series).plot(ax=canvas.ax, fig=canvas.fig)`; no GUI-specific plot variants needed.
- **`Plot` menu = one closable tab per method.** There is no single "Plotting" tab. Each plot method (Heatmap, Time
  series, ...) is a menu-activated, closable `PlottingTab(plot_type, title)` instance, registered as a factory in
  `registry.MENU_TABS["Plot"]`. Add a method = add a factory there + a branch in `plotting._draw_one` (dispatch on the
  `HEATMAP`/`TIMESERIES`/... constants).
- **Data flow.** The `File` menu loads data via `OpenDataDialog` (parquet → `dv.load_parquet`, else `dv.ReadFileType`;
  multiple files → `MultiDataFileReader` / parquet `combine_first`; reading is library work, the dialog only calls it).
  `MainWindow` holds the current DataFrame and pushes it to every tab via the `DiiveTab.on_data_loaded(df, created)`
  hook; data-presenting tabs override it. Example data auto-loads on startup.
- **Overview tab.** First tab, focused on every load (`setCurrentIndex(0)`). Top: variable list + a GridSpec figure
  (time series, `Cumulative`, `DielCycle` mean diel cycle, date/time heatmap; extensible via `_PANELS`). Bottom: a
  full-width strip of KPI-style stat cards (`_StatCard`) from `dv.sstats`.
- **Feature engineering tab.** Menu-activated (`Tools ▸ Feature engineering`, from `registry.MENU_TAB_CLASSES`) — not in
  the tab bar until selected, and closable (always-on tabs get their close button removed). Runs `FeatureEngineer`
  (library) on selected variables, emits new columns via a `featuresCreated` signal; `MainWindow` merges them, tracks
  them in a `created` set, re-pushes. Created columns get a pink **✦ NEW** pill (delegate `CREATED_ROLE`). Tab signals
  live on a `QObject` helper because `DiiveTab` is a plain `ABC`, not a `QObject` — class-level `Signal`s on a `DiiveTab`
  won't bind. When lazily creating a menu tab, call `tab.widget()` (builds it) **before** connecting `featuresCreated`,
  which `build()` sets.
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

**Last Updated:** 2026-06-06 | **Version:** v0.91.0 | **Package Manager:** `uv`
