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
uv sync                              # core + dev
uv sync --all-extras --group db      # everything: GUI, 3D + InfluxDB (influxdb-client)
uv run pytest tests/test_gapfilling.py -v
uv run python script.py
uv run pytest tests/ -v
uv add package_name
```

## Development Environment

**Python:** 3.12-3.13 | **Package Manager:** `uv`

**Key dependencies (minimum pins):** pandas 3.0+, numpy 2.2+, scikit-learn 1.6+, xgboost 3.0+, matplotlib 3.10+, statsmodels 0.14+, pyarrow 19.0+

**Optional dependencies** split across two uv mechanisms (so `--all-extras` alone is *not* "everything"):

| Kind | Name | Pulls in | Install |
|---|---|---|---|
| extra | `gui` | PySide6 desktop GUI | `uv sync --extra gui` |
| extra | `gui3d` | PyVista/VTK 3D surface tab | `uv sync --extra gui3d` |
| group | `db` | `influxdb-client` (backs diive's InfluxDB engine, `diive/core/io/db/influx`) | `uv sync --group db` |
| group | `dev` | test/lint/notebook tooling (synced by default) | — |

`db` is a **dependency group**, not an extra (personal/local InfluxDB workflow), so it needs `--group db`. Install all of the above with `uv sync --all-extras --group db`. The InfluxDB download/upload/delete engine lives in `diive/core/io/db/influx` (`InfluxIO`); `influxdb-client` is imported lazily, so the default `uv sync` never pulls it in.

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

`import diive as dv` exposes 10 domain namespaces. Authoritative per-symbol detail lives in the code docstrings — this table is a discovery index, not a spec.

| Namespace | Contents |
|---|---|
| `dv.outliers` | `AbsoluteLimits`, `Hampel`, `LocalSD`, `LocalOutlierFactor`, `zScore`, `zScoreRolling`, `zScoreIncrements`, `TrimLow`, `ManualRemoval`, + daytime/nighttime variants |
| `dv.events` | `Event` (instant or period marker; `resolved_color(i, colors=)`), `event_to_flag` (→ 0/1 column on an index), `overlay_events` (`axis='x'` value-vs-time / `axis='y'` heatmap; `colors=` override map), `make_event_flag_name`, `CATEGORY_COLORS` |
| `dv.gapfilling` | `RandomForestTS`, `XGBoostTS`, `SWINGapFillerXGBoost`, `FluxMDS`, `QuickFillRFTS`, `OptimizeParamsRFTS`, `OptimizeParamsTS`, `LongTermGapFillingRandomForestTS`, `LongTermGapFillingXGBoostTS`, `FeatureEngineer`, `GapFillingResult`, `prediction_scores`, `linear_interpolation` |
| `dv.flux` | `FluxConfig`, `FluxLevelData`, `run_chain`, `init_flux_data`, `add_driver`, `WindDoubleRotation`, `reynolds_decomposition`, `MaxCovariance`, `PreWhiteningBootstrap`, `PwbBatchDetection`, `TlagApplier`, `PerFilePipeline`, `process_one_file`, `FluxDetectionLimit`, ustar classes, plus **NEE→GPP+RECO partitioning** and **uncertainty** — see the two sub-sections below |
| `dv.analysis` | `DailyCorrelation`, `GrangerCausality`, `StratifiedAnalysis`, `GapFinder`, `GapStats`, `GridAggregator`, `Histogram`, `FindOptimumRange`, `SeasonalTrendDecomposition`, `BinFitterCP`, `CompoundExtremes`, `harmonic_analysis`, `spectrogram`, `percentiles101`, `rank_drivers`, `profile_dataframe`, `dataframe_overview`, `count_gaps` |
| `dv.analysis.experimental` | **(provisional)** `DriverAnalysis`, `DriverAnalysisResult`, `AleCurve`, `Ale2DResult`, `accumulated_local_effects`, `accumulated_local_effects_2d`, `ExperimentalWarning` |
| `dv.plotting` | `HeatmapDateTime`, `HeatmapXYZ`, `HeatmapYearMonth`, `HexbinPlot`, `ScatterXY`, `TimeSeries`, `DielCycle`, `RidgeLinePlot`, `HistogramPlot`, `ShiftedDistributionPlot`, `Cumulative`, `CumulativeYear`, `LongtermAnomaliesYear`, `TreeRingPlot`, `DateTimeSurface` (+ `datetime_surface_grid`), `WaterfallPlot`, `CompoundExtremesPlot`, `WindRosePlot`, `FormatStyle` — see **Plotting** below |
| `dv.times` | `TimestampSanitizer`, `DetectFrequency`, `keep_daterange`, `insert_timestamp`, `format_timestamp`, `validate_timestamp_column_name`, `resample_to_daily_agg`, `resample_to_monthly_agg_matrix`, `timestamp_infer_freq_*` |
| `dv.variables` | `DaytimeNighttimeFlag`, `daytime_nighttime_flag_from_swinpot`, `TimeSince`, `potrad`, `potrad_eot`, `calc_vpd_from_ta_rh` (+ codegen `calc_vpd_from_ta_rh_to_code`), `aerodynamic_resistance`, `dry_air_density`, `et_from_le`, `latent_heat_of_vaporization`, `air_temp_from_sonic_temp`, `lagged_variants`, `classify_variable`, `combine_variables` (+ codegen `combine_variables_to_code`), noise helpers |
| `dv.corrections` | `MeasurementOffsetFromReplicate`, `WindDirOffset`, `remove_nighttime_zero_offset` (corr key stays `'radiation_zero_offset'`; `clamp_negatives=True` default), `nighttime_zero_offset_diagnostics`, `NighttimeZeroOffsetResult`, `remove_relativehumidity_offset`, `set_exact_values_to_missing`, `setto_threshold`, `setto_value`, `apply_corrections` |
| `dv.qaqc` | `FlagQCF`, `StepwiseMeteoScreeningDb`, `MEASUREMENTS`/`Measurement`, `CORRECTIONS`/`CorrectionSpec`, `corrections_for_measurement(code)`, `detect_measurement(varname)`, `measurement_label`, `correction_spec` |

Top-level (no namespace): `load_exampledata_parquet`, `load_exampledata_parquet_lae`, `load_parquet`, `load_parquet_many` (with `progress_callback`), `save_parquet` (`enforce_diive_format=True`), `to_diive_format`, `ReadFileType`, `search_files`, `sstats`, `keep_vars`, `keep_records_where` (+ codegen `select_records_to_code`; both in `diive/core/dfun/frames.py`), `transform_yearmonth_matrix_to_longform`, `get_encoded_value_from_int`, `get_encoded_value_series`

### NEE→GPP+RECO partitioning (`diive.flux.partitioning`)

Four **faithful ports**, distinguished by a token after the `_NT` (nighttime) / `_DT` (daytime) suffix — `_OF` ONEFlux / `_RP` REddyProc — so all coexist in one dataframe:

| Class / function | Suffix | Notes |
|---|---|---|
| `NighttimePartitioningOneFlux` / `partition_nee_nighttime_oneflux` | `*_NT_OF` | ONEFlux Reichstein 2005, per-calendar-year, incl. robust `*_NT_OF_ROB`; helpers `lloyd_taylor`, `sunrise_sunset` |
| `NighttimePartitioningReddyProc` / `partition_nee_nighttime_reddyproc` | `*_NT_RP` | REddyProc `sMRFluxPartition`, **whole-record single E0**, signature adds `lon`/`utc_offset`, potential-radiation day/night split, Kelvin Lloyd-Taylor, no robust variant; helpers `lloyd_taylor_kelvin`, `potential_radiation`; reproduces ReddyProc columns 1:1, RECO r≈0.997 on CH-DAV |
| `DaytimePartitioningReddyProc` / `partition_nee_daytime_reddyproc` | `*_DT_RP` | REddyProc `partitionNEEGL` / Lasslop 2010 LRC. `nee` measured + `ta`/`vpd`/`sw_in` gap-filled + `lat`/`lon`/`utc_offset`, optional `nee_sd`, `vpd_in_kpa=True`; emits fitted `K`/`BETA`/`ALPHA`/`RREF`/`E0_DT_RP`. RECO r≈0.999, GPP r≈0.9999 vs a fresh ReddyProc run; bootstrap `*_SD`/CUT_16/84 not yet emitted |
| `DaytimePartitioningOneFlux` / `partition_nee_daytime_oneflux` | `*_DT_OF` | ONEFlux `flux_part_gl2010` / Lasslop 2010 FLUXNET2015. **Day/night split is measured-`Rg`≤4/>4, NO solar geometry.** `nee`/`ta`/`sw_in` measured + `ta_f`/`sw_in_f`/`vpd` filled, `vpd_in_kpa=True`; emits `SE_GPP_DT_OF` + central `ALPHA`/`BETA`/`K`/`RREF`/`E0_DT_OF`. float32 working arrays (FLOAT_PREC); RECO r≈0.999, GPP r≈0.9999 vs native ONEFlux; ~22s/year |

All four wire into the chain as **Level 4.2** (`run_level42_nighttime_oneflux` / `_nighttime_reddyproc` / `_daytime_reddyproc` / `_daytime_oneflux`): one partitioning per USTAR scenario, columns merged into `fpc_df` with the scenario label appended (`RECO_NT_OF_CUT_50`); meteo drivers from `data.full_df`, coords from `data.meta`; nighttime variants read gap-filled NEE from a selectable L4.1 method (`gapfill_method='mds'` default), daytime use measured NEE only. Exposed on `run_chain` via `FluxConfig.partition_*`, with `partitioned_cols()` lookup and `levels.level42_*`.

### Uncertainty (`diive.flux.lowres`)

- **`RandomUncertaintyPAS20`** (`uncertainty.py`) — Pastorello 2020 / faithful ONEFlux `randunc` port. 4-method hierarchical cascade (first to succeed wins) emitting `{flux}_RANDUNC` + `WINDOW_N_VALS_METHOD1..4`: (1) std of >5 meteo-similar measured fluxes in ±7d/±1h (MDS similarity via shared `gapfilling.similarity` kernel — the only *direct* measurement); (2, ONEFlux) median of method-1 uncertainties of ±20%-similar fluxes (floor 2 µmol) in ±14d; (3)&(4) diive extensions (whole-record / nearest-magnitude) so every record gets a value. `.run(progress_callback=)`; `.randunc_series`/`.randunc_results`/`.randunc_results_cumulatives`. Hot loops are vectorised numpy (~35x faster, bit-identical). Cumulative = quadrature `sqrt((randunc**2).cumsum())`, NaN-safe (NOT an object-dtype `ufloat` cumsum — one NaN must not poison the tail). Codegen `randunc_to_code` (`codegen.py`). Takes `vpd_in_kpa=True`.
- **`JointUncertaintyPAS20`** / `joint_uncertainty_pas20` — faithful ONEFlux `compute_join`. Combines random uncertainty with scenario-ensemble percentile spread in quadrature per record: `JOINTUNC = √(RANDUNC² + ((upper−lower)/divisor)²)`. `divisor` = `JOINT_DIVISOR_1SIGMA=2.0` for NEE 16th/84th USTAR scenarios, `JOINT_DIVISOR_IQR=1.349` for LE/H 25th/75th. NaN in any input → NaN. Default name strips trailing `_RANDUNC`→`_JOINTUNC`. Cumulative: random independent → quadrature; scenario choice fully correlated → running spread `(cumsum(upper)−cumsum(lower))/divisor`, combined in quadrature. Codegen `jointunc_to_code`.

## Core Concepts

### Feature Engineering (8-stage)

1. Lag features  2. Rolling stats  3. Differencing (1st/2nd order)  4. EMA  5. Polynomial  6. STL decomposition  7. Timestamps  8. Record number

`FeatureEngineer` class, fed into gap-filling models. Naming `.{col}_TYPE{detail}` (e.g. `.Tair_f_POL2`). `FeatureEngineer(target_col='_target_', ...)` — `target_col` is a required placeholder; any string not in the feature list works.

### Gap-Filling Methods

| Method | Training | Features | Use case |
|---|---|---|---|
| Random Forest | Yes | 8-stage engineered | Interpretable, robust |
| XGBoost | Yes | 8-stage engineered | Non-linear, efficient |
| SWINGapFillerXGBoost | Yes | SW_IN_POT + timestamps (+ opt. TA/VPD) | SW_IN w/ nighttime constraint; `nighttime_threshold=0.001` matches `remove_nighttime_zero_offset` |
| MDS | No | Meteorological similarity | Faithful ONEFlux port; no overfitting |
| Linear Interp. | No | None | Small gaps only |

**MDS faithful ONEFlux port** (`FluxMDS`, `diive/gapfilling/mds.py`): the cascade lives once in `diive/gapfilling/similarity.py::mds_gapfill_cascade` (6-stage expanding-window, ported from `uncert_via_gapFill`, fills r≈0.9997 vs native ONEFlux), plus `meteo_similar_mask`, `mds_quality_from`, `mds_granular_flag`. The **same cascade** backs daytime-partitioning NEE uncertainty (`daytime_oneflux._uncert_via_gapfill`), so it is **dtype-preserving**: float32 caller gets f4 boundary behaviour, float64 (MDS) full precision. `avg_min_n_vals` default 2 (uncertainty path uses 10); SD is N-1; `sym_mean` is the optional Vekuri (2023) variant. Public flag `FLAG_*_gfMDS_ISFILLED` is **granular** `method*1000 + time_window` (0=measured; method 1/2/3 = ALL/SWIN/MDC); faithful 1/2/3 quality kept in `.PREDICTIONS_QUALITY`. `FluxMDS` and `RandomUncertaintyPAS20` both take **`vpd_in_kpa=True`** (converted to hPa internally for the 5-hPa/0.5-kPa tolerance — pass `False` for hPa). For these ports, required units are **stated in docstrings, not validated** — don't add unit-guessing warnings or EddyPro-specific hints to the library (caller owns units).

**Results:** all gap-filling classes expose `.results` (after `.run()`) → `GapFillingResult`: `gapfilled` (Series), `flag` (0=observed, 1=gap-filled, 2=fallback), `scores['r2']`, `feature_importances` (SHAP, ML only), `model` (ML only). Legacy `.result` (raw DataFrame) still available.

**[VALIDATION — expect this question often]** The ML gap-fillers' held-out scores (`scores_traintest_`) come from a **random** train/test split (`test_size`, default 25%) of the *complete* rows, **not** a temporal/block split. This is **correct and intentional for gap-filling**: the model predicts each gap from driver values at that timestamp, gaps are interspersed with observed data, so a random hold-out reproduces exactly the gap-filling task. A temporal/block split answers a *different* question (transferability to an unseen period) and conflates fill skill with regime change. (Do **not** frame this as "long gaps belong to MDS" — wrong: MDS degrades on long gaps, driver-based ML often handles them better.) Documented at the split in `common.py.__init__`, on `scores_traintest_`, in GUI `test_size`/HELD-OUT TEST tooltips, and `MANUAL.md`. `scores_` is the in-sample (optimistically biased) counterpart.

**Console report:** `MlRegressorGapFillingBase` emits a Rich coloured report at `verbose>=2` (all ML gap-fillers inherit it): phase banners, a Configuration table (regressor + every hyperparameter), data/split summary, held-out + in-sample score tables, feature-importance tables, a feature-reduction table (SHAP vs random benchmark with accept/reject), and a gap-fill summary. Default `verbose=0` silent; step-chatter at DEBUG(3). GUI runs at `verbose=2` → streams into the Log tab. **Keep console strings cp1252-safe** (Windows stdout): ASCII `->`, not `→`. **Tee:** `_TeeConsole` overrides only `print`/`log`, not `rule` (Rich's `rule()` renders via `self.print`, so a `rule` override double-forwards to mirrors).

ML classes (RF/XGB) expose `plot_feature_importances(ax=None, traintest=False, max_features=None, …)` (two-phase SHAP bar plot, on `MlRegressorGapFillingBase`). The structured `feature_importances_` DataFrame (`SHAP_IMPORTANCE`/`SHAP_SD`) is the data behind it.

### Timestamp Sanitization

```python
sanitizer = dv.times.TimestampSanitizer(df, nominal_freq='30min', verbose=True)
clean_df = sanitizer.get()
status = sanitizer.get_status()  # rows removed/added, detection method
```

## Flux Processing Chain (Swiss FluxNet Workflow)

6-level EC post-processing: L2 (quality flags) → L3.1 (storage correction) → L3.2 (outlier removal) → L3.3 (USTAR filtering) → L4.1 (gap-filling) → L4.2 (NEE→GPP+RECO partitioning, optional). Each level is a pure function — never mutate input.

**Single-call driver** — `run_chain(data, FluxConfig)` for the standard FLUXNET pipeline:

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

**Composable per-level callables** — for custom L3.2 pipelines or feature engineering:

```python
from diive.flux.fluxprocessingchain import (
    init_flux_data, run_level2, run_level31, run_level33_constant_ustar, run_level41_mds)
data = init_flux_data(df, fluxcol='FC', site_lat=46.6, site_lon=9.8, utc_offset=1)
data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None}, ...)
data = run_level31(data, gapfill_storage_term=True)
data = run_level33_constant_ustar(data, thresholds=[0.30])  # labels auto-gen as CUT_0
data = run_level41_mds(data, swin='SW_IN', ta='TA', vpd='VPD_kPa')
final_df = data.fpc_df
```

**Per-level signatures intentionally differ** (per-test dicts at L2, booleans at L3.1, pre-built object at L3.2, parallel lists at L3.3, built engineer + kwargs at L4.1). `FluxConfig` is consumed only by `run_chain`, never by `run_level*`. Per-level `run_level*`, `make_level32_detector`, and codegen `chain_to_code`/`level2_to_code`…`level41_to_code`/`level42_to_code` live in `diive.flux.fluxprocessingchain`.

**Container fields:**

| Field | Type | Description |
|---|---|---|
| `data.fpc_df` | `DataFrame` | Working dataframe; grows as levels append columns. Use for results/export. |
| `data.full_df` | `DataFrame` | Full input (+ day/night flags). Read-only source for L2, L3.1, L4.1 drivers. |
| `data.filteredseries` | `Series\|None` | QCF-filtered flux from most recent level |
| `data.meta` | `FluxMeta` (frozen) | Site coords, fluxcol, swinpot_col, QCF thresholds |
| `data.levels` | `LevelResults` | Typed bag of per-level outputs |
| `data.summary()` | `str` | Per-level data availability (day/night breakdown) |
| `data.gapfilled_cols()` / `data.partitioned_cols()` | `dict` | Gap-filled / L4.2 output columns per method & scenario |
| `data.gap_stats(level='L33')` | `dict[str, GapStats]` | On-demand gap analysis |
| `data.plot_cumulative_comparison(...)` / `data.plot_gapfilled_heatmaps(...)` | `None` | Method-comparison plots (`showplot=False` headless) |
| `data.levels.level41_methods()` | `dict` | Short keys `'mds'`/`'rf'`/`'xgb'` |

Key `data.levels` fields: `level2`, `level2_qcf`, `level31`, `level31_qcf`, `level32`, `level32_qcf`, `level33`, `level33_qcf`, `level41_mds/rf/xgb` (dicts keyed by ustar_scenario for L3.3+). **Flag naming:** `FLAG_..._TEST` (individual tests, 0/1/2) and `FLAG_..._QCF` (level-overall) are both consumed by `FlagQCF`; `FLAG_..._ISFILLED` is **informational only**, NOT consumed by QCF. L3.1 introduces no new test — its QCF re-aggregates L2 flags on the storage-corrected target.

**Architecture notes:**

- L3.2 is multi-call/stateful: `make_level32_detector(data)` → multiple `flag_outliers_*` + `addflag()` pairs → `run_level32(data, outlier_detector=sod)`. `run_level32` validates the detector is wired to the *current* `data` snapshot; rejects detectors with no committed flags or an uncommitted last test.
- `run_level41_rf`/`run_level41_xgb` take a pre-built `FeatureEngineer`.
- `finalize_level2/31/33()` are no-ops w/ `DeprecationWarning`.
- `LevelResults` isn't `frozen=True` but treat as immutable — every level rebuilds it via `dataclasses.replace`. Don't mutate fields or `level41_*` keys in place.
- `add_driver(data, series, name=None)` puts a Series into `data.full_df` (where L4.1 reads), not `fpc_df`; validates index/name/collision.
- **Re-runs cascade.** Re-running L2/L3.1/L3.2/L3.3 drops the previous run's `fpc_df` columns + downstream `LevelResults` before fresh output (`levels/_rerun.py`); re-running level N invalidates N and every later level. Columns tracked in `data.added_columns`. L4.1 is per-method (`'L4.1_mds'`/`'_rf'`/`'_xgb'`) and additive — each drops only its own previous columns.

**Critical pitfalls:**

- MDS requires exact units: W/m² (radiation), °C (temp), **kPa (VPD)** — stated in docstrings, not validated (caller's responsibility).
- USTAR filtering applies ONLY to CO2/CH4/N2O; for H/LE use `thresholds=[0], threshold_labels=['CUT_NONE']`. `run_level33_constant_ustar` raises on a non-zero threshold for an energy-flux basevar (`H2O`, `T_SONIC`, lowercase variants).
- L3.2 and L3.3 require L3.1; L3.3 also requires L3.2 (`run_level33_*` raises if `level32_qcf` is None). For H/LE call `run_level31(data, set_storage_to_zero=True)`. `run_chain` runs L3.2 unconditionally; to skip it use the composable API.
- L4.1 features and MDS driver columns must exist in `data.full_df`, not `fpc_df`. Use `add_driver()`.
- `init_flux_data` raises if `df` already contains `SW_IN_POT`/`DAYTIME`/`NIGHTTIME` (reserved).
- `nighttime_accept_qcf_below` (was `nighttimetime_...` before v0.91.0 — typo fixed).
- Default `daytime_accept_qcf_below=1` is stricter than FLUXNET's `2`; QCF=0 pass, =1 soft warn, =2 hard fail.
- L3.3 has three composable entry points: `run_level33_constant_ustar` (scalar threshold(s)), `run_level33_variable_ustar` (time-varying Series; a constant is just a constant Series, so both share `FlagMultipleVariableUstarThresholds`), `run_level33_ustar_detection` (in-pipeline bootstrap). Detector `mode=`: `'cut'` (constant pooled → `CUT_16/50/84`) or `'vut'` (per-year → `VUT_16/50/84`); CUT and VUT mutually exclusive. diive's VUT is smoothed over a 3-year window (`UstarBootstrapThresholds`); a year with no threshold falls back to its CUT value. `run_chain` only does CUT detection (`FluxConfig(ustar_detection_mode='bootstrap', ...)`); VUT is composable-only. `threshold_labels` (constant path) optional — auto-generates `CUT_0`, `CUT_1`, … (positional, NOT percentile); pass explicit `['CUT_16','CUT_50','CUT_84']` for percentile thresholds. Length/uniqueness validated; substring overlap (`CUT_5` in `CUT_50`) rejected.
- `run_level33_ustar_detection` raises if `detector_kwargs` contains `nee_col`/`ta_col`/`ustar_col`/`swin_col` (set internally).
- `run_level41_*` warns (`UserWarning`) when a re-run would overwrite stored scenarios.

**Example:** `examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py`

## Desktop GUI (`diive.gui`)

PySide6 desktop app, optional dependency (`gui` extra, lazy-imported). Launch: `uv sync --extra gui` then `diive-gui`. File map: `diive/gui/README.md`. Standalone Windows build: `packaging/build_gui.ps1` (see `packaging/README.md`).

**[CRITICAL] Strict GUI ↔ library separation.** `diive/gui/` contains ONLY GUI code — Qt widgets, layout, rendering glue, event handling, presentation (colors/labels/styling). ALL algorithms and domain logic live in the main library; the GUI *calls* them, never reimplements them. Dependency arrow points one way: `gui` → library, never reverse (no other module imports `diive.gui`). If a GUI piece is reusable, is domain knowledge, or is an algorithm, it belongs in the library — **notify the user and propose the move** rather than putting it in `gui/`.

### Architecture principles (these are the load-bearing patterns)

- **Theme: `gui/theme.py`.** `ThemeManager` singleton (`theme.manager`) holds live editable colours (`tokens`/`pills`/`new_pill`/`ts_colors`), builds the stylesheet via `build_qss(tokens)`, emits `changed`, `apply()` re-applies app-wide. Edited via the **Appearance** tab. Single design — **Studio** (clean/minimal, `icons.py` thin-line glyphs, frameless rounded window via `widgets/header_bar.py`). `STUDIO_TOKENS`/`STUDIO_TYPOGRAPHY`; structural tokens (`CANVAS`/`INK`/`RADIUS`) re-pinned from `STUDIO_TOKENS` in `load_dict` so a stale persisted config (incl. removed "Classic") can't shadow them. Which *variable name* maps to which pill kind stays in the library: `dv.variables.classify_variable` (add a kind = rule there + colour in `theme.DEFAULT_PILL_STYLE`).
- **Shared variable list: `widgets/variable_panel.py` (`VariablePanel`).** Every tab's left-hand list MUST be this component (filter + `VariableList` + `VariableDelegate` pills + fzf-style fuzzy filtering). Filter is subsequence-gated, *scored* (`_fuzzy_score`) and reorders best-match-first; clearing restores dataset order (`ORDER_ROLE`). Width is a shared setting (`theme.manager.list_width`) — don't set per-tab widths. `run_with_loading(name, fn)` paints a static busy cue before matplotlib's synchronous render freezes the loop. Also exposes `scroll_to(name)`, `clear_filter()`.
- **Registry-driven tabs.** `MainWindow` iterates `registry.TAB_CLASSES` (always-on: Overview, Log) and knows nothing about concrete tabs. Add a feature area = write a `DiiveTab` (`title` + `build()`) and register it.
- **Menu tabs are multi-instance.** `registry.MENU_TABS` factories open a NEW numbered instance each time (tracked in `_menu_tab_list`, `_next_menu_index` reuses smallest free number). `registry.SINGLE_INSTANCE_TABS` instead focus the existing one — reserved for the **three app-wide singleton editors** (`Appearance`, `Project settings`, `Metadata explorer`), each editing one shared singleton. Always-on tabs have no close button; on close, focus falls to the tab left of the closed one but never Log (jumps to Overview).
- **Tab UX.** Movable, renamable (left double-click → `_rename_tab`), custom "×" (`icons.close_icon`); `tabBarDoubleClicked` fires for any button so an `eventFilter` records the double-click button (ignore middle/right). Studio tabs are Firefox-style pills. **Per-tab pin/freeze:** right-click a menu tab → Pin; pinned tabs (`_pinned`) skipped by `_push_data` (keep their dataset, cheap CoW), show a pin glyph. Overview/Log never pinnable.

### Shared templates (reuse these — don't hand-roll layout/flow)

- **`widgets/worker.py` (`WorkerRunner`)** — `QObject` with `done(object)`/`failed(str)` + `is_running` guard; owns daemon-thread spawn + exception→`failed`. Tab keeps `self._runner`, connects signals, calls `self._runner.run(self._compute_payload, *args)` where `_compute_payload` is a **pure** function (no Qt/threading). The outlier/correction/ustar tabs keep a thin synchronous `_worker(*args)` shim because their tests drive it directly.
- **`widgets/dual_variable_picker.py` (`DualVariablePicker`)** — click-to-add/remove available↔selected (two `VariablePanel`s, pick order kept). `set_variables`/`set_selected`/`selected_names`/`select`/`deselect`/`select_all`/`clear` + `changed`. Used by Select variables and XGBoost feature picker.
- **`widgets/sub_tabs.py` (`SubTabs`)** — segmented pill buttons over a `QStackedWidget` (only active page takes space). `add_page`/`set_page`/`current_index`/`set_label` (count badges)/`changed`; `add_corner_widget`/`add_corner_separator`. Used by the ML gap-filling template.
- **`tabs/_explorer_base.py` (`SingleVariableExplorerTab`)** — base for "pick one variable left, compute view right" tabs (Driver explorer, Gaps & coverage, Seasonal trend, Spectrogram, 3D surface). Owns the split skeleton, `select → run_with_loading → _compute`, default-variable picking (`default_var`/`default_numeric_only`, override `_default_variable`), `_recompute()`, opt-in stats strip, opt-in variable-list header (set `list_title`/`list_hint`; default None = no header). Concrete tab overrides only `_build_right()` + `_compute()`; selected var is always `self._target`.
- **`widgets/tab_chrome.py`** — `build_titlebar(title, *trailing)` and `list_header(title, hint)`. Presentation only.
- **`tabs/_ml_gapfilling_base.py` (`MlGapFillingTab`)** — full XGBoost-style layout/flow as a reusable template for every ML gap-filler. Concrete tab overrides a small hook surface: class attrs `title`/`method_name`/`method_chip_*`, methods `_model_class()`/`_build_model_box()`/`_method_kwargs()`/`_method_controls()`/`_codegen(...)`. Base supplies Model/Results `SubTabs`, title bar (Copy Python far-right, Run/Add as corner widgets), three-list target/feature picker, shared SHAP feature-reduction box, performance hero, heatmaps + SHAP table, Results dashboard, worker/emit flow. Adding a method ≈ ~140-line subclass + a codegen wrapper + registry entry. Subclasses: `tabs/gapfilling.py` (XGBoost), `tabs/gapfilling_randomforest.py` (Random Forest).
- **`tabs/_outlier_base.py` (`BaseOutlierTab`)** — one subclass per `dv.outliers` detector. See **Outlier tabs** below for the detector contract.
- **`tabs/_correction_base.py` (`BaseCorrectionTab`)** — one tab per correction; routes through library `apply_corrections`/`corrections_to_code`. Subclass sets `corr_key`/`method_suffix`/`method_chip_*`/`needs_coords`/`suited_for` and implements `_add_method_rows`/`_current_kwargs`/`_validate`/`_method_controls`. Output `{var}_{method_suffix}` with MODIFIED provenance.
- **`tabs/_partitioning_base.py` (`BasePartitioningTab`)** — one tab per partitioning port. Declarative: subclass lists `inputs` (auto-seeded combos w/ ✓/✗ markers; `needle` may be a list of alternatives; `optional` inputs stay `(none)` unless matched), coords needed (`needs_lat`/`needs_lon`/`needs_utc`), `has_vpd_unit`, `reco_col`/`gpp_col`/`method_suffix`, and implements `_build_partitioner(series_map, coords, vpd_in_kpa)`. Base runs `.run()` on a worker, previews daily-mean GPP/RECO/NEE + cumulative, emits `*_NT_OF`/`*_NT_RP`/`*_DT_RP`/`*_DT_OF` via `featuresCreated` (DERIVED).

### Tab inventory (file → purpose; non-obvious gotchas only)

Each tab is a `DiiveTab`; full widget detail lives in the code. Tab signals live on a `QObject` helper (`DiiveTab` is a plain ABC). When lazily creating a menu tab, call `tab.widget()` *before* connecting `featuresCreated` (`build()` sets it).

- **Overview** (`tabs/overview.py`) — first tab, focused on every load. 2×4 GridSpec + `sstats` ribbon. **Linked zoom:** three datetime panels share x; `xlim_changed` recomputes the diel cycle on the zoomed range and clips the heatmap. Repaint via `draw_idle()` (the deliberate exception to "use `draw()`"); `_on_resize` runs two constrained-layout passes. `on_data_loaded` diffs `created`, clears the fuzzy filter, `scroll_to`s + auto-selects new columns. Holds no subset state.
- **Hover tooltip** — `MplCanvas` attaches `HoverAnnotator` (`widgets/hover.py`); lines snap via `argmin` on `get_xdata(orig=False)` (unit-converted floats, NOT raw datetimes); `pcolormesh` (heatmap) reads `get_coordinates()`/`get_array()`; scatter `PathCollection` snaps to the nearest point within `_SCATTER_PICK_RADIUS` and shows x/y(/z). Per-axis tag `_diive_hover_intlabel` formats an integer axis (year/month heatmap) as `Month 5`/`Year 2018` instead of misreading it as a clock time. Blits.
- **Plot menu = one closable `PlottingTab(plot_type, title)` per method** (no single Plotting tab), `registry.MENU_TABS["Plot"]`. Menu glyphs via `gui/icons.py::menu_icon(label)`. Add a method = factory + branch in `plotting._draw_one` + controls in `plot_settings`. Comparison types pick panels (Ctrl+click, incl. time series **and diel cycle**); **every X/Y/Z role-picked type — Scatter, Wind rose, Hexbin, Heatmap x/y/z — assigns roles via dropdowns** (drag a variable onto a field or pick from the complete list) (`_ROLE_DROPDOWN_TYPES`, shared `_build_role_combos(labels, none_ok=)`/`set_xyz`/`xyz_values`; list draggable → `_DropComboBox` drop targets; list-click is a no-op). Scatter/Wind-rose make the colour role optional (`none_ok=(F,F,T)`); Hexbin/Heatmap-xyz require all three (`none_ok=(F,F,F)`). **`Update plot` sits below the header, left-aligned, dirty-gated** (enabled by `settings.changed`/`xyz_changed`, disabled at each render); the plot updates ONLY on click — no live re-render even for variable/role selection. **Every** plot tab has a title-bar **Copy Python** button (`_python_code` dispatches per type to library codegen: `scatter_to_code` in `scatter.py`, all others in `core/plotting/codegen.py`; no-op while role picks incomplete; multi-panel tabs emit the active panel). Preserve-pan/zoom (`preserve_view`) only re-applies a limit that still **overlaps** the new data (`_ranges_overlap`), so flipping a heatmap's orientation doesn't scroll the view off the data. Ridgeline/wind rose/tree ring/shifted distribution single-figure (own gridspec/polar axes, `canvas.auto_layout=False`); the wind rose's per-sector table sits in a **horizontal** splitter to the right of the canvas. GUI-only passes: Axes group (`_axes`→`_apply_axes`; grid lives only in Format's `show_grid`), Reverse colormap, export DPI (`_SaveDpiToolbar`, default 150). Diel-cycle stacked panels share **one** auto-column legend (drawn on panel 0).
- **Data flow** — `File` menu via `OpenDataDialog` (parquet → `dv.load_parquet`, else `dv.ReadFileType`; multi → `MultiDataFileReader`/`load_parquet_many` w/ per-file progress via `progress_callback`). `MainWindow` pushes via `on_data_loaded(df, created)`. **Save as parquet** → `dv.save_parquet`; `app.to_diive_parquet_frame` enforces single-level columns + valid `TIMESTAMP_*` index name. **Load mode "Add to current"** (`_add_dataset`, offered only when data is loaded) *extends* the record, non-destructively: unions the timestamp index (new periods appended) then `combine_first` so existing (non-null) values always win — new data only fills new rows/gaps. Same-named columns are extended in place (documented via `record_derived` "Extended with data from …", **not** flagged NEW); genuinely new columns get an `original` baseline + the NEW flag (`_created`). A **Label** prefixes all incoming columns → all-new names (side-by-side run comparison). This is distinct from `_add_features` (feature-engineering merge: same index, assign-in-place, all columns flagged NEW).
- **Date-range subselection** (`Data` menu) — non-destructive: `_full_data` full, `_data` narrowed to `_range` via `dv.times.keep_daterange`. Engineered features merge into `_full_data` (survive a reset).
- **Select variables** (`tabs/variable_selector.py`, single-instance) — `DualVariablePicker` → `_apply_var_subset` app-wide via `dv.keep_vars` (pinned tabs skipped). Persisted in `extras["var_subset"]`; picker opts into full record (`wants_full_data`).
- **Select records by condition** (`tabs/select_records.py`, multi-instance) — filter a target by a condition var's range; operations stack (Undo/Reset). All range logic is `dv.keep_records_where` (remove → `invert=True`); working series = `target.where(keep_mask)`, synchronous. **Add** → `{target}_SEL` (DERIVED); **Copy Python** → `select_records_to_code`.
- **Combine variables** (`tabs/combine_variables.py`, multi-instance) — drag a var onto heatmap 1 + another onto heatmap 2 (`_HeatmapSlot` drop targets; heatmap 3 = read-only result), pick a method (`multiply`/`add`/`subtract`/`divide`/`fillgaps`) + "keep overlapping only" (disabled for `fillgaps`). All maths is `dv.variables.combine_variables` (NaN where either missing unless keep-overlap off, which uses the op's identity; `fillgaps` = `series1.combine_first(series2)`). **Add** → `{name}` (DERIVED, parents v1/v2); **Copy Python** → `combine_variables_to_code`. Heatmaps use compact fonts + `cb_digits_after_comma="auto"`.
- **Derived-variable tabs** (`tabs/_derived_variable_base.py` `BaseDerivedVariableTab`) — one tab per single-formula derived variable (the thermodynamic/radiation family). Three-column layout (draggable `VariablePanel` | Settings | stacked preview heatmaps) like the correction/partitioning tabs; input fields are drop-target `ColumnPicker` combos (`DropComboBox` — drag a var from the list onto a field or pick it). **Synchronous** (calcs are instant, so no worker/coords, unlike `BasePartitioningTab`) but computed **only on the Calculate button** (input heatmaps refresh on field change; the result heatmap + Add wait for Calculate). Subclass sets `inputs` (ColumnPicker specs; optional `short` titles its preview heatmap), `default_name`/`out_unit`/`method_tags`, and implements `_compute(df, picks)` (library fn) + `_code(picks, name)` (codegen). Base previews one date/time heatmap per input plus the result, emits `{name}` (DERIVED, parent = first input). Only subclass so far: **VPD (TA + RH)** (`tabs/derived_vpd.py`, Variables ▸ **Calculate** section, multi-instance) — VPD kPa from TA+RH via `dv.variables.calc_vpd_from_ta_rh`; previews TA/RH/VPD, emits `VPD_kPa`, **Copy Python** → `calc_vpd_from_ta_rh_to_code`.
- **Rename variables** (`tabs/rename_variables.py`, single-instance) — prefix/suffix to all; **Apply** → `MainWindow._rename_variables` (renames `_full_data`, remaps `created`, `MetadataStore.rename`). Any variable list's right-click also offers Rename…/Delete… via `metadata_store.manager` relays.
- **Flux processing chain** (`tabs/fluxchain.py`, single-instance) — guided Input→L2→…→L4.1 on the **composable per-level** path (deliberately not `run_chain`/`FluxConfig`, so L3.2 stays a real stepwise chain w/ its own QCF surface). Horizontal `PipelineRail` (`widgets/flux_pipeline_rail.py`, GUI-only) of `StageCard`s; clicking swaps controls into a `QStackedWidget`. **Levels run incrementally** — each page's run button gates on its predecessor (`_update_level_buttons`/`self._reached`); re-running an earlier level cascades deeper state away. Right column: `qcf.screening_report()` panel. **L2** per-test column pickers seeded to EddyPro-FLUXNET names, threaded through library per-test `'col'` override; default names + VM97 sub-test list are library domain knowledge (`dv.flux.level2_test_inputs(...)`, `VM97_SUBTESTS`); GUI derives `fluxbasevar` via `detect_fluxbasevar`. Other levels mirror via `_refresh_levels_info` (Input USTAR picker, `dv.flux.level31_storage_col`). **L3.3 two modes** (constant / detection w/ CUT vs VUT). **L4.1** ticks rf/xgb/mds (additive). **Shared Random seed** (default 42) always passed to rf/xgb (`random_state`) + emitted — without it output drifts; `_level41_cfg` omits kwargs equal to library defaults. Needs EddyPro-FLUXNET input (`load_exampledata_parquet_lae_level1_30MIN`). Codegen in `flux/fluxprocessingchain/codegen.py`.
- **USTAR detection** (`tabs/ustar_detection.py`, single-instance) — standalone moving-point (Papale 2006); single seasonal (`UstarMovingPointDetection`) or multi-year bootstrap (`UstarBootstrapThresholds`) for VUT+CUT. Worker thread.
- **NEE partitioning tabs** (`Flux` menu, single-instance each) — one per port via `BasePartitioningTab`: Nighttime ONEFlux/REddyProc, Daytime REddyProc/ONEFlux (`*_NT_OF`/`*_NT_RP`/`*_DT_RP`/`*_DT_OF`).
- **Random uncertainty (PAS20)** (`tabs/uncertainty_randunc.py`, single-instance) — standalone `DiiveTab` (single output). 3-panel preview; the measured-vs-uncertainty scatter is **method-1 records only** (`WINDOW_N_VALS_METHOD1>=6`) so fallbacks don't streak. Progress bar via `RandomUncertaintyPAS20.run(progress_callback=)`. **Add** → `{flux}_RANDUNC`; **Copy Python** → `randunc_to_code`.
- **Joint uncertainty (PAS20)** (`tabs/uncertainty_jointunc.py`) — combines a `{flux}_RANDUNC` (run random first) with scenario spread via `JointUncertaintyPAS20`. Scenario combo → divisor + auto-pick needles biased toward the randunc column's flux base (`_scenario_prefer()`). **Add** → `{base}_JOINTUNC`; **Copy Python** → `jointunc_to_code`.
- **Feature engineering** (`Data ▸ Feature engineering`, closable) — runs `FeatureEngineer`, emits via `featuresCreated`; new columns get a pink ✦ NEW pill.
- **MDS gap-filling** (`tabs/gapfilling_mds.py`, single-instance) — NOT an `MlGapFillingTab` subclass (no SHAP/held-out split/reduction); shares only chrome. Fixed three-driver picker (SWIN/TA/VPD). Flag granular `method*1000 + time_window`. Codegen `mds_gapfill_to_code`. Results = `MdsResultsPanel`; `FluxMDS.quality_breakdown()` + `mds_quality_description()` are library domain knowledge. Progress bar via `FluxMDS.run(progress_callback=)`.
- **XGBoost / Random Forest gap-filling** (`Gap-filling` menu, single-instance) — `MlGapFillingTab` subclasses (XGB `*_gfXG` lilac, RF `*_gfRF` green). XGB `random_state=-1`=none (reseeds); RF `max_depth=0`=unlimited, `n_jobs=-1`. **No feature-engineering settings** (use that tab first). Model sub-tab = QHBoxLayout (**NOT a splitter** — a splitter over-allocates). Results sub-tab = `GapFillResultsPanel`; reduction `k` passed via `update(..., shap_threshold_factor=)`. Emits `*_gf*` + `FLAG_*_ISFILLED` (DERIVED).
- **Gaps & coverage** (`tabs/gaps.py`, single-instance) — `GapStats` (`.summary`/`.long_gaps`/`plot_*`/`gap_at(timestamp)`). Clickable both ways, `_syncing` guard. **Note:** library `plot_*` use `ax.figure.colorbar` (NOT `plt.colorbar`) so they embed.
- **Driver explorer** (`tabs/drivers.py`, single-instance) — `dv.analysis.rank_drivers(...)` → `[DRIVER, CORR, ABS_CORR, BEST_LAG, N]`; `ScatterXY` shifted by `BEST_LAG`.
- **Compound extremes** (`tabs/compound_extremes.py`) — synchronous. `dv.analysis.CompoundExtremes` → `CompoundExtremesPlot.from_compound_extremes(ce)`. **Copy Python** → `compound_extremes_to_code`. No columns emitted.
- **Seasonal-trend & anomalies** (`tabs/seasonaltrend.py`, single-instance) — `resample_to_daily_agg` → `SeasonalTrendDecomposition` (period 365) → `LongtermAnomaliesYear`. **Library bug fixed:** STL wrapper (`core/times/decomposition_utils.py::stl_decompose`) never passed `period` and called `STL.fit(weights=...)` (unsupported) — now passes `period` + small odd smoother, `fit()` no weights.
- **Spectrogram** (`tabs/spectrogram.py`, single-instance) — `dv.analysis.spectrogram` mapped onto calendar-time × cycles-per-day (segment centres → real timestamps via non-NaN index, correct across gaps).
- **3D surface** (`tabs/surface3d.py`, single-instance, optional **`gui3d`** extra) — date×time-of-day grid (`dv.plotting.datetime_surface_grid` → `DateTimeSurface`) as a PyVista relief; vertical controls column + list header. Two **Style**s: extruded heatmap (default; flat bar per cell via a doubled-coordinate "staircase" `StructuredGrid` from `_extruded_grid`, gap cells dropped with `threshold` so it renders opaque without `nan_opacity`) or smooth interpolated surface (opt. `subdivide`+`smooth_taubin`). Y stretch widens the date axis; Y cell binning + aggregator (`_bin_rows`, NaN-aware) widens bars / tames spikes; plus exaggeration, opacity, colormap. Lighting flat by default; optional **Shadows** = overhead spotlight + `enable_shadows` (adjustable length). `Pyvista3DCanvas.frame_default` = orthographic lower-left 45° view, re-framed only on variable change (settings tweaks keep the view via `_framed_target` + `add_mesh(reset_camera=False)`); `apply_shadows` swaps flat headlight ↔ spotlight. Copy Python → `datetime_surface_to_code` (matplotlib, runs without the extra). Shadow mapping is GPU-driver-dependent (best-effort, wrapped).
- **Outlier tabs** (`Outliers` menu) — `BaseOutlierTab` subclasses, one per detector. Each keeps original + cleaned (`{var}_{METHOD_SUFFIX}`) + flag (`FLAG_{var}_OUTLIER_{flagid}_TEST` 0/2 — flag id from the *library* class, can differ from suffix, e.g. `ZSCOREINCREMENTS` col but `..._INCRZ_TEST`).
  - **Detector contract:** `_worker` requires `.run(repeat, progress_callback)` (→ `.calc(...)`), `.filteredseries`, `.overall_flag`, `.last_lower_bound`/`.last_upper_bound` (band in data units; `None` when there's no single envelope), `.is_daytime`. Adding a tab = verify/extend the library class to this contract, don't reimplement detection. Class flags `supports_daynight`/`supports_repeat`/`band_center_label` gate optional UI. `ManualRemoval`/`TrimLow`/`zScoreIncrements` have no band; `TrimLow`'s day/night split is opt-in (`trim_daytime`/`trim_nighttime`, method rows prefixed `trim_…`; coords validated only when split requested; constructor `series, lower_limit, trim_daytime, trim_nighttime, lat, lon, utc_offset, …`).
- **Correction tabs** (`Corrections` menu) — `BaseCorrectionTab` subclasses, one per correction. All independently available for any variable (`suited_for` is a hint, not a lock). Hooks: `_apply(series, kwargs, coords) -> (corrected, extra)`, `_hero_metrics`, `_render_result`, `_status_text`. The **nighttime-offset** tab is the rich one: coords + **Clamp negatives** checkbox (default on) + 4-panel diagnostic + below-zero hero; same `clamp_negatives` mirrored on the stepwise screening panel.
- **Screening tabs** (`tabs/_screening_base.py` `ScreeningTabBase`) — the full stepwise-screening machinery (variable list + segmented Outliers/Corrections/Report inspector + method-card chain via `widgets/stepwise_cards.py` + `StepwiseOutlierDetection`→`FlagQCF` worker + `apply_corrections` + preview + save/restore) shared by two thin subclasses. Variants override three seams: data source (it's `self._df`/`self._var`-centric, so feed a synthetic frame), `_inspector_pages` (extra page), `_emit_frame` (emitted columns). **Stepwise screening** (`tabs/stepwise.py`, Outliers menu) is the base unchanged. **Meteo screening (database)** (`tabs/meteo_screening.py`, Database menu, single-instance) feeds a synthetic frame from the staged DB `data_detailed`, adds a **Resample** page (target defaults to the working dataset's `DetectFrequency` resolution; no-op when source==target via `resample_series_to_freq`), and `_emit_frame` resamples the screened (+corrected) series + `convert_series_timestamp_to_middle` (END→MIDDLE alignment), with collision rename, overlap guard, DB-origin + all-tags provenance, and a **download-vs-project timezone** mismatch warning. It reuses `StepwiseOutlierDetection`, **not** `StepwiseMeteoScreeningDb` (which stays a library/notebook API). `FlagQCF` day/night needs SW_IN_POT (off here); the shared preview shows the hi-res screening, not the resampled result.
- **Database tabs** (`Database` menu) — over `diive.core.io.db` (`InfluxDBBackend` adapter → `InfluxIO` engine; optional `db` group, `influxdb_available()` gates). **Database connection** + **Database explorer** single-instance; **Meteo screening (database)** (see above). `gui/db.py` `DbConnectionManager` (`db.manager`) holds the live backend + config-dir path (path persisted, never the token), `changed` (header pill + explorer refresh), `screeningRequested` (explorer→screening hand-off via `MainWindow._send_to_meteo_screening`). Explorer: drill bucket→data_version→measurement→field→**field overview** (all tags + first/last record); **download a field** for a date range on worker threads — **UTC offset defaults to the project timezone** (DB stores UTC; a mismatch silently shifts the merge — visible note + screening-tab warning), **Match dataset time range** (shifts END↔MIDDLE by half a period), **chunked download with a live-updating plot** + shared `widgets/progress_bar.py` `ProgressBar`, request caching (plot + screening reuse one download). The whole download/screen/resample/merge pipeline works in **TIMESTAMP_END** (DB native) then converts to **TIMESTAMP_MIDDLE** for the diive working frame.

### Persistence, metadata, projects, events

- **Per-variable metadata** — library's `diive.core.metadata` (`VariableMetadata`, `ProvenanceEntry`, `MetadataStore`, `provenance_attr`, `ATTRS_KEY`, `MAX_DESCRIPTION_WORDS=50`), headless. GUI holds it app-wide in `gui/metadata_store.py` (`manager` singleton). Each var: origin (`original`/`modified`/`derived`), parent links, ordered provenance, ≤50-word description, tags w/ source (only `USER` tags persist). Operations emit provenance via `df.attrs[ATTRS_KEY]`; `MainWindow._add_features` consumes it (`store.from_attrs`). `record_derived` makes a new column **inherit its parent's full provenance** (a *copied* snapshot, not shared). `MetadataStore.rename(mapping)` re-keys + rewrites links. **Tolerant deserialization** (`from_dict` accepts aliased/missing name + dict-or-list tags; `load_dict` skips malformed). **Metadata explorer** tab (single-instance) does full editing; other tabs' right-click **Edit metadata…** → `manager.request_edit` → `MainWindow._edit_metadata`. **Persistence is namespaced by dataset:** `config.variable_metadata` is `{dataset_key(source): {...}}`. `_set_data(persist_metadata=False)` loads clean (example auto-load uses it).
- **Persisted prefs** — `gui/config.py` saves JSON (`QStandardPaths`) on close/launch: theme, site, geometry, filetype, events, metadata. Best-effort.
- **Project settings** — `gui/site.py` `SiteManager` singleton (`site.manager`): author/description/site metadata + `configured`. Tab `tabs/site.py` (`ProjectSettingsTab`; `SiteDetailsTab` alias, single-instance). Values only, passed to library functions taking `lat`/`lon`/`utc_offset`. Right side = notes wall (`widgets/notes_wall.py`; mirrors `state()` into `site.manager.notes` via plain attribute set — no `changed` — to avoid a rebuild loop).
- **Events** — `gui/events.py` `EventManager` (`events.manager`); model is the library's `dv.events`. **Add event…** dialog result method is `make_event()`, NOT `event()` (which overrides `QDialog.event`). Events tab `tabs/events.py` (reflowing cards; Manage categories… — last category undeletable). `_sync_event_columns` (on `changed`) reconciles `EVENT_<name>` 0/1 columns. Overview overlays via `overlay_events(axis='x'/'y')`. Category colour overrides via `Event.resolved_color(i, colors=)` / `overlay_events(..., colors=events.manager.categories)`.
- **Projects** — a `<name>.diive` folder (`__diive__` marker, `project.json`, `data.parquet`). Format is library's `diive.core.io.project` — serializes the **full** `MetadataStore` + an opaque `extras` dict (library never imports GUI types). GUI (`File ▸ Save project` Ctrl+S / Save as… / Open…) puts site, range, `created`, open tabs (label/title/pinned + each `save_state()`). Every `DiiveTab` has `save_state()`/`restore_state()` (default no-op) capturing *inputs only* (via `widgets/state_utils`); heavy results re-compute. Wrapped in try/except. `_open_project` loads clean then overlays `store.load_dict`; a plain load clears `_project_dir` so Ctrl+S can't overwrite a project with unrelated data.
- **Output console** — `Log` tab mirrors diive's Rich output; library tees to any sink via `add_console_sink` (`diive.core.utils.console`); panel only renders.
- **Splash / startup** — `gui/splash.py` `QPainter`-drawn spinner. `run()` builds `MainWindow(cfg, autoload=False)` then defers the load via `QTimer.singleShot(0, …)` → `window._initial_load()` (else a blocking constructor load freezes the spinner). Tests build `MainWindow()` (`autoload=True`, synchronous). `_initial_load()` reopens `config.last_project` (if still a diive project) else the bundled example.

### PySide6 gotchas (already handled — don't reintroduce)

- **Retain tab instances** (`MainWindow._tabs`). Qt owns the QWidgets, but the Python `DiiveTab` objects hold the signal slots; if GC'd their signals go inert (symptom: clicks stop working after startup).
- **A stylesheet touching `QListWidget::item` disables per-item `setBackground`/`setForeground`.** Row colouring goes through `QStyledItemDelegate` (`VariableDelegate`), not item roles.
- **Matplotlib's Qt toolbar recolours icons from the widget palette.** `MplCanvas` sets a light palette *before* building the toolbar (else white-on-white on dark themes).
- **Use synchronous `canvas.draw()`, not `draw_idle()`,** after a user action so the canvas repaints immediately. (Overview zoom is the deliberate exception — see its note.)
- **Share axes for comparison panels** via `subplots(..., sharex=True, sharey=True)`.
- **A widget with its own stylesheet detaches its tooltips from the app-wide `QToolTip` rule** — append `theme.manager.tooltip_qss()`. The appended string must be **all-selector QSS** — `setStyleSheet("color:x;background:y;" + tooltip_qss())` silently drops the `QToolTip` block; wrap bare props in a selector first: `setStyleSheet("QLabel { color:x; background:y; }" + tooltip_qss())`. (Reference: Overview stat items, event cards.) For tables, put no `color:` on `::item` so per-item `setForeground` paints.
- **Window must fit the screen work area.** Startup uses `show_filling_workarea()` (sized to `availableGeometry`), NOT `showMaximized()` — a frameless maximize covers the taskbar and clips the active tab's bottom. It also lowers the window minimum size: the Overview's height-for-width `_HeroBand` reports a tall single-column *minimum* (its min-width height) that otherwise forces the window taller than the screen. Restored geometry is also pulled back on-screen via `_clamp_to_screen`.

## High-Resolution EC Analysis (hires)

Tools for 10/20 Hz data. Workflow: `raw 20 Hz → WindDoubleRotation → reynolds_decomposition → flux`

```python
wr = dv.flux.WindDoubleRotation(u=df['u'], v=df['v'], w=df['w'])
w_prime = dv.flux.reynolds_decomposition(wr.w2)   # rotated w2, not raw w
c_prime = dv.flux.reynolds_decomposition(df['CO2'])
flux = (w_prime * c_prime).mean()
```

**Classes:** `WindDoubleRotation`, `reynolds_decomposition`, `MaxCovariance`, `FluxDetectionLimit`, `PreWhiteningBootstrap`, `PwbBatchDetection`, `TlagApplier`, `PerFilePipeline`, `process_one_file`.

### Time-lag detection & removal (PWB)

Three CLIs (console scripts), all requiring **wind-rotation-corrected** W for detection:

| CLI | Class | Does |
|---|---|---|
| `diive-tlag-pwb-batch` | `PwbBatchDetection` | Detect lags across many averaging-period files → `tlag_results.csv` (+ PWBOPT S1/S2/S3) |
| `diive-tlag-apply-batch` | `TlagApplier` | Apply lags from a `tlag_results.csv` (shift scalars by `round(tlag_s·hz)`) |
| `diive-tlag-pwb-detect-remove` | `PerFilePipeline` | Two-phase per-chunk detect+remove in one run |
| `diive-tlag-pwb-detect-remove-tui` | `DetectRemoveTUI` | Textual TUI wrapping `PerFilePipeline`; `--demo` previews without data |

`diive-tlag-pwb-detect-remove` ([detect_and_remove_tlag.py](diive/flux/hires/detect_and_remove_tlag.py)) splits each long raw file into fixed-length chunks (`--chunk-seconds`, default 30 min): phase 1 rotates each chunk in memory + runs PWB per scalar (no write); PWBOPT picks the best lag per chunk; phase 2 shifts each scalar by it (`--lag-column-template`, default `{prefix}_tlag_final_pf_s` — what `TlagApplier` removes, NOT raw `tlag_s`) and writes one file per chunk. Parameterized in seconds × `--hz`. Output: `1_lag_detection/` + `2_lag_removed/` + `log.txt`. **Downstream flux processing must run with EC time-lag maximization disabled.**

- Per-chunk filenames from `--chunk-name-template` ({stem}/{suffix}/{index}/{starttime}); `{starttime}` needs `--start-time-regex` + `--start-time-format`. Terminator `--lineterm auto` (reproduces input CRLF/LF; headers normalised — never mixed).
- `PerFilePipeline.run(cancel_event=threading.Event())` is cooperative-cancellable; `pipeline.cancelled` reports it. `run()` writes summary CSV + overview plots (TUI/CLI/Python all get them).
- **TUI** (`DetectRemoveTUI`, `--demo`): full option coverage + tooltips; Check preflight; Stop; Open output folder; ▾ column picker; overwrite guard; live validation; spinner rows; lag-removal phase shown as **"align"** (paper's term), not "remove".
- **Per-gas time-lag search windows.** `PreWhiteningBootstrap` takes optional `lws`/`uws` (seconds): the bootstrap peak search (and diagnostic `tlag_pw`) is restricted to `[lws, uws]` while the CCF is still computed symmetrically over `±lag_max`. `None`/`None` = full-window (byte-identical). Positive-only `[0,5]` keeps physical tube-delay lags; a long-inlet gas (H2O) can run a wider window than dry gases in one run (EddyPro applies one uniform lag downstream). Threading: `PerFilePipeline`/`process_one_file`/`detect_one_chunk` take global `lws`/`uws` + `per_gas_lag` (`{label: {lag_max_s, block_length_s, lws, uws}}`), resolved per gas by `_resolve_gas_lag`; CLI `--scalar "H2O:h2o@lag=30;uws=25"` (`parse_scalar_spec`) + `--lws`/`--uws`. Library helper **`window_to_lag_params(lws, uws, min_block_s=20.0)`**: per gas `lag_max_s = max(|lws|,|uws|)`, `block_length_s = max(20s, 2·half)` (paper's 20s floor, growing for wide windows). The block-length warning fires only when a block is **shorter** than `2·lag_max`, not for the intentional floor. **TUI:** Scalars field = gases only (`LABEL:column`); a Win s field carries `LABEL:[lower,upper]` (`parse_win_ranges`/`format_win_ranges`, `Input.Changed` on `#scalars` → `_sync_win_field`), seeded `[-LagMax,+LagMax]`, ⟳ re-seed. **Lag max s is only the seed** — once a gas has a window its half-width is `lag_max`. Keep the expected lag mid-window (boundary detection = unreliable, paper's discard rule). Tests: `tests/test_echires.py::TestPwbPerGasWindow`.

## Outlier Detection & QC

- **Single:** `dv.outliers.Hampel(series).run()`
- **Chained:** `dv.outliers.StepwiseOutlierDetection()`
- **Corrections:** `dv.corrections.MeasurementOffsetFromReplicate()`, `remove_nighttime_zero_offset()`, `setto_*()`
- **QCF aggregation:** `dv.qaqc.FlagQCF()` → 0 (good) / 1 (marginal) / 2 (poor)
- **Full pipeline:** `dv.qaqc.StepwiseMeteoScreeningDb()` — corrections → outlier detection → quality flags
- **Timestamp shift:** three methods comparing measured vs theoretical radiation (requires clear days)

**[CONVENTION] Day/night threshold parameters.** Outlier methods supporting `separate_day_night` follow one rule — **all new day/night-capable methods MUST match it:**

1. **A single global knob (`n_sigma`, `threshold`, …) is the source of truth.** Per-period overrides (`n_sigma_daytime`/`n_sigma_nighttime`) **default to `None`, never a literal**, and fall back to the global value: `self.x_daytime = x_daytime if x_daytime is not None else x`. Defaulting to a literal (the old `Hampel` bug: `n_sigma_daytime=5.5`) silently shadows the global knob. Verify that changing the global value alone changes the result.
2. **`separate_day_night` only changes results when day and night thresholds differ.** With equal thresholds it's mathematically identical to no separation. So a GUI exposing the feature should expose *per-period* thresholds (seeded from the global value, then independently editable), not just the toggle. The Hampel tab (`gui/tabs/outliers.py`) is the reference: separate sigma fields + red/blue day/night markers + a day/night count in the status line.

## Coding Standards

- **Input validation** — only at system boundaries (user input, external data). Trust internal code.
- **Error handling** — let exceptions propagate unless you can recover.
- **Comments** — only WHY, not WHAT. Hidden constraints, workarounds, non-obvious logic.

### Console Output & Verbosity

**All production output uses Rich console helpers** from `diive/core/utils/console.py`. **NO `print()` in production code** (allowed in `examples/*/`, docstrings, `__main__`, `_cli_main()`).

```python
from diive.core.utils.console import console as _console, info, detail, warn, success, rule
```

| Function | Level | Use case |
|---|---|---|
| `rule(title)` | PROGRESS (2) | Section headers |
| `info(msg)` | PROGRESS (2) | Key progress, results |
| `success(msg)` | PROGRESS (2) | Operation completion |
| `warn(msg)` | ERROR (1) | Warnings (always visible) |
| `error(msg)` | ERROR (1) | Errors (always visible) |
| `detail(msg)` | DEBUG (3) | Inner-loop details |
| `_console.print(msg)` | None | User-facing formatted reports |

Levels: `VERBOSE_SILENT=0`, `VERBOSE_ERROR=1`, `VERBOSE_PROGRESS=2` (default), `VERBOSE_DEBUG=3`. All helpers accept `verbose=`. When using `if self.verbose >= N:` guards, call helpers WITHOUT `verbose=` inside the block. **Do NOT:** use `print()`, create separate `Console` instances, use `logging` for general output, mix `print()` and Rich in one file.

### Examples (Sphinx Gallery format)

Use `# %%` cell markers. No file I/O (API only). Single year of data. Disable `showplot=True`. **Checklist for new examples:** 1. Register in `examples/run_all_examples.py` + `examples/CATALOG.md`. 2. Add category README description. 3. Reference in source docstring "Example" section. 4. Update `examples/README.md` file count. 5. Note in CHANGELOG.md. 6. Verify it runs.

## Plotting

**Two-phase pattern.** Phase 1 `__init__()` — data + computation params ONLY (no `ax`, title, labels, colors, limits). Phase 2 `plot(ax=None, ...)` — all styling + rendering; `ax=None` creates a new figure; can be called multiple times w/ different styles/axes.

```python
scatter = dv.plotting.ScatterXY(x=df['A'], y=df['B'])
scatter.plot(ax=axes[0], title='Linear')
scatter.plot(ax=axes[1], title='Log', ylim='auto')
```

**Conventions:**
- **Colors:** Material Design 300-level (bars/lines) + 500-level (backgrounds): blue `#2196F3`, red `#F44336`, amber `#FFC107`, grey `#455A64`.
- **Bar labels:** `va='center_baseline'` (not `va='center'`) for digit-only strings inside bars.
- **Label contrast:** `text_color = 'white' if 0.299*r + 0.587*g + 0.114*b < 0.5 else 'black'`
- **Dynamic height:** `height = max(1.5, n_years * 0.38)` inches for multi-year panels.

**`FormatStyle`** (`diive/core/plotting/styles/format.py`) is the **only** way to set plot **chrome** (title/labels/units/fontsizes/weights/colors/grid/legend/zeroline) — pass `plot(format_style=FormatStyle(...))`. `None` fields resolve to the `LightTheme` standard block, so a bare `FormatStyle()` **is** the house style; editing those constants restyles every plot. Old flat chrome kwargs (`title`/`xlabel`/`ylabel`/`series_units`/...) were **removed** from every `plot()` (v0.91.0 breaking change). Use `.merged(**overrides)` to vary one field. Data-render args (`color`/`cmap`/`marker`/`vmin`/`vmax`) and colorbar args (`cb_*`, `zlabel`) stay direct `plot()` kwargs. The matplotlib work is `FormatStyle.apply(ax, default_title=, default_xlabel=, default_ylabel=, zeroline_data=)`. Chrome-only conversion for the multi-axes `RidgeLinePlot` and polar `TreeRingPlot`. GUI plot-settings panel has one shared **Format** section read via `_format_values()` → `values()["_format"]` → `FormatStyle(**...)`.

**Per-plot data-render notes:**
- **`DielCycle.plot`** takes `agg` (`mean`/`median`/`min`/`max`/`p25`/`p75`), `band` (`none`/`sd`/`se`/`iqr`/`minmax`), `each_month`, `cmap` (per-month palette), `marker`/`markersize`. Aggregates are computed on demand by `diel_cycle(..., mean=/std=/median=/quantiles=/minmax=)`. Deprecated `mean=`/`std=` kept (std=False→band='none'). `scatter_to_code` is the scatter codegen.
- **`ScatterXY`** uses internal unique column keys (`_x`/`_y`/`_z`) so the same variable can fill two roles (e.g. colour-by-X) without `pd.concat` collapsing duplicate-named columns into a frame.
- **`RidgeLinePlot`** sets the gridspec `hspace` (overlap) at creation, *before* adding subplots — a later `gs.update(hspace=)` is a silent no-op for an embedded GUI figure (not in pyplot's `Gcf`).

## Development Workflow

**[CRITICAL] NEVER COMMIT CHANGES.** User stages and commits exclusively.

**[CRITICAL] NEVER RUN EXAMPLE SUITE.** Only test individual examples:
```bash
uv run python examples/gapfilling/gapfill_randomforest.py
```

**Commit message style:** one-line title (< 50 chars) + bullet points.

**Do NOT:** run `uv` commands without explicit approval, skip pre-commit hooks (`--no-verify`), force-push to main/master, include Claude as co-author.

## Testing

```bash
pytest tests/test_gapfilling.py -v              # Gap-filling
pytest tests/test_fluxprocessingchain.py -v     # Flux chain
pytest tests/test_gui.py -v                      # Desktop GUI (offscreen, needs 'gui' extra)
pytest tests/ -v                                 # All
```

- Use flexible assertion ranges (`assertGreater/assertLess`) for SHAP variability.
- Don't mock databases in integration tests.

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
|---|---|
| SHAP importance fluctuates ±5-10% | Flexible ranges in tests (`assertGreater/Less`) |
| XGBoost base_score in scientific notation | Monkey-patched in `MlRegressorGapFillingBase` |
| Feature reduction too strict | Reduce `shap_threshold_factor` (default 0.5) |
| Unicode on Windows (arrow chars) | Use ASCII (>, ->) in examples |
| Textual `App` has internal `_running`/`_workers` | Don't name your App attr `_running` (use `_busy`); Textual sets `_running=True` on mount |
| Textual `@work` not starting from a non-handler context | Dispatch via `threading.Thread(target=…, daemon=True)`; `call_from_thread` delivers UI updates |

## Common Workflows

**Add feature engineering stage:** 1. Add param to `FeatureEngineer.__init__()`. 2. Implement `_stagename_features()`. 3. Call from `_create_features()`. 4. Naming `.{col}_TYPE{detail}` (e.g. `.Tair_f_POL2`).

**Debug SHAP importance:** 1. Check `.RANDOM` baseline included. 2. Verify threshold calculation. 3. Inspect feature counts before/after reduction.

## Quick Reference

| Task | Command |
|---|---|
| Install | `uv sync` |
| Run test | `uv run pytest tests/ -v` |
| Run example | `uv run python examples/gapfilling/gapfill_randomforest.py` |
| Add package | `uv add package_name` |
| List packages | `uv pip list` |
| Check version | `uv run python -c "import diive; print(diive.__version__)"` |
| Run all examples | `python examples/run_all_examples.py` |

---

**Last Updated:** 2026-06-29 | **Version:** v0.91.0 | **Package Manager:** `uv`
