# Changelog

![DIIVE](images/logo_diive1_256px.png)

## v0.91.0 | XX May 2026

**Major release: desktop GUI, composable flux chain, 10 domain namespaces, four NEE partitioning ports.**
652 commits since v0.90.0. Method detail and validation live in the docstrings and `examples/`; this is what changed.

### Breaking Changes

Two of these change results silently, with no error and no warning:

- **`ScopPhysics`'s gap-filled results column is renamed `FCT_UNSC_gfRF` -> `FCT_UNSC_gfXG`.** The suffix named Random
  Forest while the fill has been XGBoost, and it disagreed with the `.fct_unsc_gf` attribute holding the very same
  series, which was already named `FCT_UNSC_gfXG`. Code doing `physics.get_results()["FCT_UNSC_gfRF"]` now raises
  `KeyError`; there is no alias, since a duplicated column in a results frame invites publishing numbers from the one
  that names the wrong method. `ScopApplicator` accepts either name (it normalises its input), so only direct indexing
  of the physics results frame is affected. The name now lives in one place and doubles as the lookup key into the
  gap-filler's output, so a future regressor change fails loudly instead of mislabelling.

- **`potrad` now runs a different algorithm**, a faithful port of ONEFlux's `get_rpot` (the routine behind FLUXNET's
  `SW_IN_POT`). The signature is unchanged, so existing calls keep running and return different numbers: RMSE 20.0 W/m2
  against the old version, annual sum +1.1%, day/night classification changed on 2.27% of half-hours. Re-check any
  stored `SW_IN_POT`, day/night flag, or downstream result. `potrad_eot` is removed with no replacement.
- **`SW_IN_POT` and day/night classification shift everywhere as a result**: the flux chain, USTAR detection,
  gap-filling, NEE partitioning, meteo screening, `SWINGapFillerXGBoost`, `DetectTimestampShifts`, and every
  day/night-capable outlier method (`LocalSD`, `Hampel`, `AbsoluteLimits`, `LocalOutlierFactor`, the zScore variants,
  `TrimLow`).

The rest raise or fail at import:

- **Raw high-frequency tooling moved to [dyco](https://github.com/holukas/dyco).** `WindDoubleRotation`,
  `reynolds_decomposition`, `MaxCovariance`, `FluxDetectionLimit` and the PWB time-lag classes
  (`PreWhiteningBootstrap`, `PwbBatchDetection`, `TlagApplier`, `PerFilePipeline`, `DetectRemoveTUI`) are gone from
  `dv.flux`. The four `diive-tlag-*` console scripts are now `dyco-pwb-batch`, `dyco-apply-batch`,
  `dyco-detect-remove` and `dyco-detect-remove-tui`, leaving `diive-gui` as the only script diive installs. diive
  starts at averaged (e.g. 30-minute) data now. `TimeLagAnalysis` stays: it reads EddyPro's `*_TLAG_ACTUAL` columns,
  not raw data. `textual` and `polars` went with the move.
- **10 domain namespaces** replace the flat namespace: `dv.outliers`, `dv.gapfilling`, `dv.flux`, `dv.analysis`,
  `dv.plotting`, `dv.times`, `dv.variables`, `dv.corrections`, `dv.qaqc`, `dv.events`. Update all imports.
- **Plot chrome is `FormatStyle`-only.** Every flat chrome keyword (`title`, `xlabel`, `ylabel`, `series_units`,
  `axlabels_fontsize`, `legend_loc`, `show_grid`, ...) is removed from `plot()`. Pass
  `format_style=dv.plotting.FormatStyle(...)`, and `.merged(**overrides)` to vary one field. Data and colorbar args
  (`color`, `cmap`, `vmin`/`vmax`, `cb_*`) are unchanged.
- **Gap-filling `features_*` parameters moved to `FeatureEngineer`.** Pass a pre-built instance to `RandomForestTS` /
  `XGBoostTS`.
- **MDS gap-fill flag is now `method * 1000 + time_window`** (0 = measured), replacing the old 1-60 levels. The faithful
  ONEFlux 1/2/3 quality moved to `.PREDICTIONS_QUALITY`. `avg_min_n_vals` default 5 -> 2.
- **`remove_radiation_zero_offset` renamed to `remove_nighttime_zero_offset`** (and the matching
  `StepwiseMeteoScreeningDb` method), since it suits any variable that reads zero at night. New optional
  `clamp_negatives=True`. Saved projects still load.
- **`zScore` unified**: use `zScore(separate_day_night=True)`. `zScoreDaytimeNighttime` removed.
- **One way to set day/night thresholds across every outlier detector.** The switch is `separate_day_night`
  everywhere (`separate_daytime_nighttime` is gone), and per-period values are always a global value plus optional
  `*_daytime` / `*_nighttime` overrides that fall back to it:
    - `LocalSD`: `n_sd` and `winsize` no longer accept a `[day, night]` list. Use `n_sd_daytime`, `n_sd_nighttime`,
      `winsize_daytime`, `winsize_nighttime`.
    - `AbsoluteLimits`: `daytime_minmax` / `nighttime_minmax` are replaced by `minval_daytime`, `maxval_daytime`,
      `minval_nighttime`, `maxval_nighttime`. `minval` / `maxval` alone now cover both periods instead of raising.
    - `Hampel`: the duplicate `n_sigma_dt` / `n_sigma_nt` pair is removed; use `n_sigma_daytime` /
      `n_sigma_nighttime`.
  Defaults are unchanged, so results do not move. Every removed name raises a message naming its replacement, so a
  stale call fails immediately rather than running with a silently ignored argument.
- **`AbsoluteLimitsDaytimeNighttime` and `LocalOutlierFactorDaytimeNighttime` now separate day from night.** Both were
  aliases for their base class and inherited its off-by-default switch, so they ran whole-series detection under a name
  promising otherwise; `LocalOutlierFactorDaytimeNighttime` was the same object as `LocalOutlierFactorAllData`. Code
  that relied on the old behaviour will see different flags. `LocalOutlierFactorAllData` and `HampelDaytimeNighttime`
  are unaffected: both names already matched what their base class did.
- **`DailyCorrelation` is now a class**; the function API is removed.
- **`HeatmapXYZ` requires pre-aggregated input.**
- **Plotting aliases use the `plot_` prefix**; old unprefixed names removed.
- **`make_level32_detector` returns `(data, sod)`.**
- **Flux chain renames**: `level41_methods()` keys are now `'rf'` / `'xgb'`; `nighttimetime_accept_qcf_below` is
  `nighttime_accept_qcf_below`; `gapfill_storage_term` defaults to True; the energy-flux set adds `G`, `SH`, `SLE`,
  `FH2O`.
- **`FluxProcessingChain` is superseded** by `run_chain` plus the composable per-level callables. It still works, but
  `finalize_level2/31/33()` no-op with a `DeprecationWarning`.
- **The code-review round removes more public API** (`flux_type`, `keep_overlap_only`, the STL quality-weighting
  surface, four dead harmonic functions, `UstarDetectionMPT`) **and makes outlier flags NaN at missing records.** It
  also changes results in more than twenty other places with no error and no warning. All of it is under *Fixed (code
  review)* below.

### Added

- **Desktop GUI** (`pip install 'diive[gui]'`, then `diive-gui`): 67 tabs driving the library, each with a **Copy
  Python** button that emits a runnable script. Covers plotting (17 types), outlier detection (9 methods), corrections
  (5), gap-filling (XGBoost / Random Forest / MDS, with a long-term per-year mode), a guided flux processing chain,
  NEE partitioning, uncertainty, analysis tabs, InfluxDB browsing, per-variable metadata with provenance, events, and
  portable `.diive` project folders. Optional `gui3d` extra adds two GPU 3-D surface tabs. Manual:
  `diive/gui/MANUAL.md`. Standalone Windows build: `packaging/`.
- **NEE partitioning**: four faithful ports, each validated against its reference implementation and tagged so all four
  coexist in one dataframe. `NighttimePartitioningOneFlux` (`*_NT_OF`), `NighttimePartitioningReddyProc` (`*_NT_RP`),
  `DaytimePartitioningReddyProc` (`*_DT_RP`, RECO r = 0.9992 / GPP r = 0.9999 vs a fresh REddyProc run),
  `DaytimePartitioningOneFlux` (`*_DT_OF`, RECO r = 0.999 / GPP r = 0.9999 vs native ONEFlux). Also available as
  `partition_nee_*` functions and as chain Level 4.2 (`run_level42_*`). Bootstrap uncertainty is not yet emitted.
- **Composable flux chain**: `run_chain(data, FluxConfig)` for the standard L2 to L4.2 pipeline, or one pure callable
  per level (`run_level2`, `run_level31`, `make_level32_detector` + `run_level32`, `run_level33_constant_ustar` /
  `_variable_ustar` / `_ustar_detection`, `run_level41_mds` / `_rf` / `_xgb`, `run_level42_*`) for full control, over
  typed containers (`FluxLevelData`, `FluxMeta`, `LevelResults`). Adds in-chain USTAR detection (CUT or VUT), Level 4.2
  partitioning, a proper L3.1 QCF, cascading re-runs, and `add_driver`.
- **InfluxDB engine `InfluxIO`** (`diive.core.io.db`, `pip install 'diive[db]'` or `uv sync --group db`): in-house
  download / upload / delete / schema browsing, replacing the external `dbc-influxdb` dependency. `influxdb-client` is
  imported lazily.
- **USTAR detection**: `UstarMovingPointDetection` (ONEFlux moving-point, Papale 2006),
  `UstarVekuriThresholdDetection` (quantile-based), `UstarBootstrapThresholds` (per-year p16/p50/p84).
- **Uncertainty**: `JointUncertaintyPAS20` / `joint_uncertainty_pas20`, the ONEFlux `compute_join` port combining random
  uncertainty with scenario spread in quadrature.
- **Gap-filling**: `SWINGapFillerXGBoost`, physics-aware for shortwave radiation (nighttime gaps set to zero, daytime
  gaps modelled by XGBoost on SW_IN_POT + timestamp features; needs only lat/lon/UTC offset). With no context drivers
  every feature is a deterministic function of the timestamp, so the model reproduces a climatology and cannot recover
  a gap's sky state; passing a second radiation measurement (a pyranometer or PPFD sensor) through `context_df` is what
  breaks that ceiling. **The class is designed to need nothing but `series` + `lat`/`lon`/`utc_offset`**, so the
  remaining settings adapt themselves. `interpolate_short_gaps` defaults to `'auto'`: clearness-index interpolation is
  enabled at a 2-record limit when there is no `context_df` and disabled when there is, which is the better branch in
  each direction (CH-DAV, 1 year, 15% scattered gaps: no-context 100.6 -> 75.2 W m-2 with it on, but PPFD-context
  12.6 -> 66.0 if forced on). Defaults tuned late in the dev
  cycle, which shift results for anyone tracking `indev`: **nighttime offset correction is on by default**
  (`correct_nighttime_offset=True`), near a no-op on a quality-controlled series but removing the thermal-offset bias
  on a raw pyranometer record; **lag features are off by default** (`features_lag=[]`) because a lag of a gappy
  context driver is NaN whenever a neighbour is missing, which demotes otherwise-fillable records to the timestamp-only
  fallback; and **SW_IN_POT is excluded from the rolling and EMA stages**, since rolling/EMA variants of a
  deterministic timestamp curve measure identically to leaving them out. A continuous record number is added by
  default (`add_continuous_record_number=True`) as cheap insurance against sensor drift on long raw records (neutral
  on clean data). Feature-engineering settings are configured the same way as the XGBoost ones: a **`feature_kwargs`
  dict** overriding the SW_IN defaults in `_FE_DEFAULTS`, replacing the individual `features_lag` /
  `features_rolling` / `features_rolling_stats` / `features_ema` / `add_record_number` parameters. This reaches
  *every* `FeatureEngineer` argument, including the diff, polynomial and STL stages that the old signature could not
  express. Passing a `FeatureEngineer` argument as a top-level keyword now raises `TypeError` pointing at
  `feature_kwargs` — previously it would have been swallowed into `**kwargs` and silently ignored by XGBRegressor.
  XGBoost defaults are now set for SW_IN rather than inherited from XGBoost: **`n_estimators=3000`, `max_depth=6`,
  `early_stopping_rounds=20`**, all overridable through `**kwargs`. The large tree budget with early stopping cuts
  daytime-gap RMSE by 17% with no context driver (CH-DAV, 10 years, 20% gaps: 132 -> 109 W m-2) and 5% with a PPFD
  context sensor (23.2 -> 21.9), stopping at ~600-1200 trees. Early stopping is not optional at this budget: building
  all 3000 trees barely improves RMSE but makes the SHAP pass several times slower, since TreeSHAP cost is linear in
  tree count. To keep that affordable, SHAP importances are now computed on a capped 10k-row subsample
  (new `shap_max_rows` on `MlRegressorGapFillingBase` / `XGBoostTS`, `None` = every row and the default everywhere
  else). Mean |SHAP| converges well before the full record — on a 10-year half-hourly record the subsample reproduced
  the feature ranking exactly (Kendall tau 1.000) with importances within 2%. Predictions and scores are unaffected;
  only the reported importances (and, if enabled, `reduce_features` selection) read the subsample. Also
  `FeatureEngineer` as a standalone 8-stage pipeline, a new
  `GapFillingResult.feature_importances_reduction` carrying the SHAP table as it stood *before* feature reduction
  dropped anything — the only view that includes the `.RANDOM` benchmark column the keep threshold is derived from —
  `GapFillingResult` +
  a `.results` property on every gap-filler, `plot_feature_importances()` on the ML base class, and a Rich console
  report at `verbose>=2`. **Fixed:** `quickplot` keyed a list of series by name, so same-named series silently
  collapsed to one panel and the survivor was labelled with a dropped series' name — visible in
  `remove_nighttime_zero_offset(showplot=True)`, which passes the raw and the corrected series (both carrying the
  variable's name) and therefore lost the measured panel while labelling the corrected one as the measurement.
  Duplicates are now suffixed, and that caller names each stage explicitly.
  `SWINGapFillerXGBoost` also exposes the model that did the filling as **`daytime_model_`**
  (`None` when there were no daytime gaps): its `traintest_details_` carries the held-out test set, so the fill can be
  validated against data the model never saw — which the results object alone does not allow, since `gapfilled` equals
  the measurement wherever the flag is 0.
- **Analysis**: `CompoundExtremes` (+ `CompoundExtremesPlot`), `GapStats`, `GrangerCausality`,
  `SeasonalTrendDecomposition`, `DetectTimestampShifts`, `spectrogram`, `harmonic_analysis`, `rank_drivers`,
  `profile_dataframe`, `keep_records_where`, `keep_vars`. `DriverAnalysis` ships **provisionally** in
  `dv.analysis.experimental` and emits an `ExperimentalWarning`.
- **Plots**: `WindRosePlot`, `WaterfallPlot`, `TreeRingPlot`, `ShiftedDistributionPlot`, `DateTimeSurface`,
  `TimeSeries.plot_rangetool()` (interactive Bokeh).
- **Events** (`dv.events`): `Event`, `event_to_flag`, `overlay_events` for time-stamped markers as 0/1 columns and plot
  overlays.
- **Per-variable metadata** (`diive.core.metadata`) and the `.diive` **project format** (`diive.core.io.project`):
  headless tag + provenance model, and save/load of a full working state.
- **Codegen**: `*_to_code` renderers across plotting, gap-filling, outliers, corrections, partitioning, uncertainty and
  the flux chain, so any call can be emitted as a runnable script. This is what the GUI's Copy Python uses.
- **`resample_series_to_freq`**, which generalises `resample_series_to_30MIN` to any resolution.

### Improved

- **`import diive` dropped from 2.35 s to 0.96 s.** The ten domain namespaces and `diive.io`'s submodules now load on
  first attribute access (PEP 562), so a script that only reads a parquet file no longer pays for sklearn, xgboost,
  shap and statsmodels. `dv.gapfilling`, `from diive import flux` and every documented import path behave as before;
  IDEs and type checkers still see the real modules. Most of the saving came from `diive.io`: `formats.fluxnet`
  imports `ManualRemoval`, which pulled in the whole preprocessing tree down to bokeh.
- **`core.ml` no longer depends on the `gapfilling` package.** `prediction_scores` moved to `diive.core.ml.scores`;
  `diive.gapfilling.scores` re-exports it, and `dv.gapfilling.prediction_scores` is unchanged. Importing
  `diive.core.ml.common` on its own previously raised `ImportError` from a circular import that only stayed hidden
  because `diive/__init__` happened to import `gapfilling` first.
- **Ruff is configured and enforced**, replacing a config that set only a cache path. `line-length` is now explicit
  (the 88 default would have had `ruff format` rewrite 290 of 303 files), bugbear rules are on, and `gui/`'s
  one-line-per-widget Qt style is exempted rather than fought. Findings went 411 -> 93, and every remaining one is
  style or judgment: deliberate lazy imports, leftover locals, and the Qt setup lines.
- **Every `zip()` states its length handling.** Most are `strict=False`, which is accurate: in the plotting loops the
  artist collection is built from the sequence being zipped, so a mismatch cannot occur. Seven are `strict=True` where
  both sides are provably equal and a future edit should fail loudly. Three stay `strict=False` because truncation is
  intended (seeding X/Y/Z from fewer than three numeric columns, restoring saved GUI state that predates a widget, and
  a `.get(key, [])` default).
- **Line endings are pinned** in `.gitattributes` (`* text=auto eol=lf` plus the binary formats), so working trees stop
  drifting from the index. No stored content changed: no blob had CRLF.
- **MDS is now a faithful ONEFlux port**: the 6-stage expanding-window cascade, `>=2`-sample acceptance, N-1 standard
  deviation, and the ONEFlux SWIN tolerance. Fill values r ~ 0.9997 to 0.99997 against native ONEFlux. Shared with
  random uncertainty via `diive.gapfilling.similarity`, so there is one similarity scan. Also 4x faster,
  bit-identical to before.
- **Random uncertainty `RandomUncertaintyPAS20` aligned to the ONEFlux C reference** (methods 1 and 2), ~35x faster with
  bit-identical output, and takes `vpd_in_kpa=True` so VPD units are consistent across the ONEFlux ports.
- **`UstarMovingPointDetection` rewritten** for ONEFlux parity (calendar-quarter seasons, tie-aware class boundaries,
  the one-big-season fallback, an `Annual` bootstrap row) and ~8x faster.
- **Nighttime threshold standardised to 20 W/m2** (was 50) throughout.
- **`shap` floor raised to `>=0.50.0`** and the XGBoost `base_score` monkey-patch removed; it was fatal on shap
  `>=0.52`. Environments pinned to shap 0.48/0.49 must upgrade.
- **`verbose=0` is now actually silent** for gap-filling (XGBoost's own eval history is gated to DEBUG).
- **Rich console migration** across production modules, and the console now renders correctly in Jupyter (wider tables,
  legible `rule()` colour).
- **`.run()` / `.result` unified** across the outlier, gap-filling and analysis class families, enabling
  `dv.outliers.Hampel(s, n_sigma=5).run().result`.
- **Two-phase design** extended to the heatmaps and `HexbinPlot` (data in `__init__`, styling in `plot()`), and
  `FormatStyle` gives every plot one shared chrome definition resolving to the `LightTheme` house style.
- `OptimizeParamsRFTS` generalised to `OptimizeParamsTS` (any sklearn-compatible regressor). SHAP replaces permutation
  importance for feature reduction. `GapFinder` single-pass vectorized (`limit` -> `max_length`, new `min_length`).
  `Hampel`, `LocalOutlierFactor` and `zScore` consolidated from their day/night variants.
- `FlagMultipleVariableUstarThresholds` is now public, and `LevelResults.level33` is annotated with both flagger types.

### Fixed

- **`InfluxIO.delete(measurements=True)` deleted nothing and reported success.** Expanding `True` went through
  `schema.measurements()` without a `start` argument, so InfluxDB applied its 30-day default and returned no
  measurements for any bucket whose newest record is older than that. The delete loop then never ran, while the summary
  line — built from the *inputs*, not from what was deleted — still announced that all measurements had been wiped. The
  lookup now covers the full history and is scoped to `data_version`, an empty result raises instead of passing
  silently, and the summary names the measurements actually targeted. `schema.fieldKeys()` in `fields_in_bucket` had the
  same missing `start`, so `show_fields_in_bucket` under-reported for older buckets (the GUI Database explorer's
  measurement list too).
- **`Hampel` ignored `n_sigma` in day/night mode**: the per-period defaults were the literal `5.5` instead of `None`, so
  the fall-back was dead. **Results change** for callers who passed `n_sigma` alone with `separate_day_night=True`.
- **`make_patch_spines_invisible` raised `AttributeError` on every call**, so the heatmap black-and-white render and
  `make_secondary_yaxis`'s twin axis both failed. `ax.spines.values()` had been rewritten to `ax.spines.to_numpy()()` by
  the global `.values` -> `.to_numpy()` replace during the pandas 3.0 upgrade. The rest of the codebase was swept for the
  same damage; this was the only instance.
- **Every ridgeline plot failed with `unhashable type: 'numpy.ndarray'`.** `adjust_color_lightness` looked a colour up in
  `matplotlib.colors.cnames`, which raises `TypeError` (not `KeyError`) for an RGBA tuple or numpy array, and
  `RidgeLinePlot` passes colours straight from a colormap. It now accepts named, hex, tuple and array colours alike.
- **Warnings blamed the library instead of the caller.** Three `warnings.warn` calls had no `stacklevel`, so a complaint
  about the caller's series being too short was reported at `decomposition_utils.py` rather than at the line that passed
  the data.
- **Re-raised exceptions discarded the original traceback.** Eight handlers interpolated the caught exception into a new
  message and dropped it; an STL or Granger failure inside statsmodels surfaced as a wrapped string with nothing pointing
  at where it broke. All now chain with `raise ... from e`.
- **Two bare `except:` handlers** also swallowed `KeyboardInterrupt` and `SystemExit`; both are narrowed.
- **`FluxLevelData.gap_stats` had an undefined annotation** (`dict[str, 'GapStats']` with `GapStats` never imported), so
  `typing.get_type_hints()` and Sphinx autodoc failed on it.
- **`lagged_variants` silently produced no lags for a single-column dataframe**, returning it unchanged with no warning.
  The same column lags correctly when it is one of two, so the guard was treating "only one column" as "nothing to lag".
  It now raises only when that column is also in `exclude_cols`. It also no longer adds its columns to the caller's
  dataframe as a side effect, and neither does `add_continuous_record_number`.
- **`sstats` raised on a series with no valid values**: `ZeroDivisionError` for an all-NaN series, `IndexError` for an
  empty one, although a variable with no data in the selected period is ordinary. `series_start` / `series_end` /
  `series_duration` return `NaT` on an empty index and `outlier_percentage` returns `NaN` when nothing is valid. An
  all-NaN series keeps its real start and end, since only the values are missing.
- **`DetectFrequency`'s failure message recommended three fixes, none of which worked.** `regularize` and `nominal_freq`
  belong to `TimestampSanitizer`, already default to the suggested values, and cannot skip detection: it runs before
  regularization, and `nominal_freq` only gates a later validation step.
- **Dead code removed**: `diive/logger.py` (unreferenced, and broken since its PyQt5 import was commented out) and
  `plotfuncs.remove_prev_lines` (unreferenced, and assigning to `ax.collections` has raised since matplotlib 3.7).
- **`RandomUncertaintyPAS20` cumulative uncertainty was poisoned by a single NaN**, turning every later value NaN.
- **`ManualRemoval` date matching was not day-inclusive**: a date-only spec matched only the `00:00:00` record.
  **Results change**: it now removes the whole day. Malformed `remove_dates` entries raise instead of being ignored.
- **`TrimLow` required lat/lon/UTC offset** even with no day/night split, and drew its plot twice. Coordinates are now
  validated only when a split is requested.
- **`SeasonalTrendDecomposition(method='stl')` always raised**: the wrapper never passed `period` and called an
  unsupported `STL.fit(weights=...)`.
- **`Hampel` rejected the signal instead of the outliers wherever the local MAD was zero.** A window in which more than
  half the records are identical has a MAD of exactly `0`; the code substituted `1e-6` to dodge the division, which
  collapses the detection band to zero width and flags every value that differs from the local median at all. On a
  soil-moisture record whose 10MIN era had been upsampled to 1MIN (runs of ten identical values) this rejected **19.4%
  of that era — 97 085 records — at any `n_sigma`**: raising it from 8 to 100 changed the count by 1%. Such records are
  now left unflagged, with the count reported, since a window with no scale cannot judge anything. **Results change**
  for any series with flat or quantized stretches; a real spike surrounded by normal variability is unaffected.
- **`Hampel(use_differencing=True)` differenced across gaps.** Missing records are dropped before the double
  difference is taken, so the two records flanking every gap were compared with a partner hours or days away and looked
  like spikes. Differences that span more than 1.5x the nominal record spacing are now excluded from the test.
  **Results change** on gappy series.
- **`verbose=True` printed no statistics** in `Hampel`, `zScore`, `LocalSD`, `LocalOutlierFactor` and `AbsoluteLimits`.
  Their per-iteration "Total found outliers" lines go through `detail()`, which only prints from `VERBOSE_DEBUG` (3),
  while the documented `verbose=True` maps to `VERBOSE_PROGRESS` (2) - so the one number the caller asked for was the
  one number never shown, visible only in the preview figure's title. Those lines are now pinned to
  `VERBOSE_PROGRESS`. Output only.
- **`default_format` wrote the string "False" into axis labels.** Its "no label" default is `False`, which was passed
  straight to `ax.set_xlabel()` / `ax.set_ylabel()` and rendered literally on every plot that relied on the defaults,
  including the outlier detection preview plots.
- **`StepwiseMeteoScreeningDb` raised on input with more than one time resolution**, i.e. on any variable whose logger
  changed sampling rate partway through the record. `_harmonize_timeresolution` built the upsampling frequency as
  `f'{targetfreq}S'` from the float seconds returned by `detect_freq_groups`, giving pandas the invalid alias
  `'60.0S'`, and then back-filled with the `fillna(method=...)` removed in pandas 3. Frequencies are now passed as
  `Timedelta`. Single-resolution input never reached either line, which is why this survived the pandas 3 migration
  untested; `tests/test_meteoscreening.py` now covers both paths.
- **`HeatmapYearMonth` raised `AttributeError`** from a wrong import path.
- **`import diive` forced the matplotlib `Agg` backend**, disabling interactive windows.
- **`_TeeConsole` double-printed every `rule()`** to mirror consoles such as the GUI Log tab.
- Flux chain L3.3 gained VUT filtering (`mode='cut'|'vut'`), and `UstarBootstrapThresholds` gained
  `get_vut_thresholds()` alongside `get_cut_threshold()`. diive's VUT is smoothed over a 3-year window, a deliberate
  departure from strict ONEFlux VUT.
- XGBoost/SHAP on Python 3.13; pandas 3.0.3 compatibility; `LocalOutlierFactor` `contamination='auto'`;
  `linear_interpolation` on the current `GapFinder` API; `data.filteredseries` reset after
  `run_level33_constant_ustar()`; `calc_vpd_from_ta_rh` export; `SortingBinsMethod` alias; 7 misrouted `__init__`
  exports.

### Documentation

- **113 examples** across 10 domain folders (Sphinx Gallery format), with new coverage for the flux chain (L2 standalone,
  composable, multi-flux, partitioning), all four partitioning ports plus a comparison, USTAR methods, SW_IN
  gap-filling, compound extremes, gap stats, events, and I/O. `examples/CATALOG.md` now lists every one.
- **New InfluxDB notebooks** (`notebooks/DatabaseInflux*`) for download, meteo screening, and delete. 21 older notebooks
  archived, their content migrated to examples.
- Switched from poetry to `uv` for dependency management.

### Fixed (timestamp-shift detection)

- **`DetectTimestampShifts.crosscorr()` now recovers a clock offset.** It returned 0 for a planted 60-minute shift and
  -54 for a 120-minute one, on noise-free synthetic radiation — while a brute-force Pearson scan over the same day
  found the true lag at r = 1.0000, so the signal was there and the method was losing it. Two causes: the daytime mask
  (`sun_up`, derived from *potential* radiation) clipped both series before correlating and truncated the shifted
  measured curve, and the FFT cross-correlation was not normalised per lag by the overlap count, so lags with less
  overlap scored lower and `argmax` was pulled toward zero. The search window is now padded by `max_shift_min` on each
  side and the lag scan is a direct Pearson correlation — which is what the docstring always promised. **Results
  change**: on CH-DAV 2022 the method now reports -5.0 min against the FFT method's -5.9 min (they used to disagree by
  ~6 min, with crosscorr stuck near 0), and a perfect match now scores r = 1.000 instead of 0.913 — the latter mattered
  because `plot_crosscorr_results` filters at `min_corr=0.97` by default and was hiding good days. One year takes
  0.78 s (was 0.31 s for the wrong answer). `max_shift_min` and the reported shift are now converted through the
  upsampled step, so a non-default `upsample_freq` no longer silently reinterprets minutes as samples.

### Fixed (naming)

- **A falsy `idstr` no longer leaks the text "None" into column names.** `validate_id_string` returned `None`
  unchanged for a falsy input, and callers interpolate the result straight into names, so omitting the optional
  argument produced `FLAGNone_FC_QCF` / `FCNone_QCF` from `FlagQCF` — and the same latent bug sat in
  `eddyproflags` (`FLAGNone_FC_SIGNAL_STRENGTH_TEST`), `storage_correction` and `quality_flags`. It now normalises to
  `''`, giving `FLAG_FC_QCF` / `FC_QCF` / `FC_QCF0`. Callers that branch on `if idstr:` (`FlagBase`, the USTAR
  flaggers) are unaffected, since an empty string is falsy too. **No shipped code path changes**: every production
  call site passes an explicit idstr (`L2`, `L3.1`, `STEPWISE`, `METSCR`), so only direct library calls that omit the
  documented-optional argument see different names.

### Fixed (console output)

- **Console reports no longer crash on a redirected Windows stdout.** `FlagQCF.report_qcf_flags()` printed U+2550
  box-drawing rules, so `python ... | head` or `> log.txt` raised `UnicodeEncodeError: 'charmap' codec can't encode
  character` — Python falls back to cp1252 whenever stdout is a pipe, while a terminal and pytest are both UTF-8, which
  is why it went unnoticed. CLAUDE.md already required console strings to be cp1252-safe. Fixed in the two places that
  actually print: `qcf.py` (box-drawing -> `= - |`) and `gapfilling/interpolate.py` (`≥` -> `>=`, including the
  `ValueError` message, which reaches stderr the same way).
- **New `tests/test_console.py::TestConsoleStringsAreCp1252Safe`** walks the library's printed string literals — those
  passed to the console helpers, `_console.print/log/rule`, builtin `print`, and `raise` — and fails on any character
  cp1252 cannot encode. Its blind spots are documented in the test: strings assembled into a variable before printing
  are not seen, and `diive/gui/` (Qt) is excluded because it never writes to plain stdout.
  Docstrings and comments are deliberately ignored.

### Fixed (corrections)

- **Three corrections no longer rename the caller's Series.** `setto_threshold`, `set_exact_values_to_missing` and
  `remove_relativehumidity_offset` did `series.name = "input_data"` on the parameter itself, so the object the caller
  still held came back named `"input_data"`. The returned series was always named correctly, and `apply_corrections`
  copies before it dispatches, so the GUI and the meteo-screening path were unaffected — a direct library call was
  not. Each now binds a renamed copy (`work = series.rename("input_data")`), the pattern `_nighttime_zero_offset`
  already used. `setto_threshold` validates its `type` argument *after* the old rename, so a rejected call left the
  caller's series renamed too; that is fixed with it. Values, index and output names are unchanged. One visible side
  effect: the console line `Variable: input_data` now names the real variable, as do the `showplot` titles.

### Fixed (code review)

A read-only review of the library and the GUI, in three rounds, recorded 105 findings ranked so that a silently wrong
number sits above a crash: a traceback blocks you, a plausible wrong number gets published. 87 are fixed and 2 are
closed as by design. The fixes landed as separate commits, each naming the findings it closes, and every closed entry
in `CODE_REVIEW_FINDINGS.md` carries what was changed, why that direction was chosen (code or docstring), and how it
was verified. **Removed API comes first below, then the fixes that change numbers with no error and no warning**, since
those are the ones to check existing results against.

#### Removed or renamed API

- **`flux_type` is gone from `ScopPhysics`, `ScopOptimizer` and `ScopApplicator`, and the self-heating correction is no
  longer offered for LE at all.** Whether a Burba-type correction applies to the latent heat flux is unresolved in eddy
  covariance, so diive must not offer one. The removed branch never worked either: it computed its µmol to W conversion
  and then discarded it, so the fitted scaling factor absorbed Lv and `ScopApplicator` multiplied by Lv a second time. A
  planted factor of 1.500 came back as 0.0666 through the H2O path, and comes back exactly on the CO2 path. Drop the
  argument from existing calls; it had one legal value left, and all 9 example call sites passed it.
- **Outlier flags are NaN where the input record is missing, not 0.** A flag records a test result, and no test can run
  where there is no value, but `FlagBase.repeat` summed an all-NaN row to `0`, so every never-measured record read as
  "tested, passed". Any code reading `overall_flag` directly, or counting `(flag == 0)`, changes. `MissingValues` keeps
  0/2, since missing records are that detector's subject, via the new `nan_flag_at_missing` class attribute. `FlagQCF`
  output is bit-identical (NaN and 0 both contribute nothing to the flag sums), so the flux processing chain is
  unaffected. Seven docstring claims across five modules had promised this behaviour all along and were rewritten to
  describe the flag as it is; preserving missing data is `.filteredseries`'s job, not the flag's.
- **`keep_overlap_only` is gone from `combine_variables` and `combine_variables_to_code`.** The arithmetic methods are
  now always overlap-only, NaN wherever either input is missing. The option substituted the operation's identity for a
  missing operand, which returned `-B` for `subtract` and `1/B` for `divide` when the left operand was the missing one,
  so `NEE - RECO` silently yielded `-RECO` wherever NEE was missing. Even for `add` and `multiply` a one-sided result is
  just the other variable wearing a sum's label. Substituting a value for a gap is a gap-filling decision, which
  `method='fillgaps'` already states plainly, so nothing is lost. The GUI tab reports how many records the overlap rule
  costs, split by which variable was missing, since a large one-sided count usually means the two variables cover
  different periods.
- **`quality_weighted_decompose` is removed, `weights` is gone from `stl_decompose`, and `quality` /
  `quality_weighted` are gone from `SeasonalTrendDecomposition`** along with the "Quality-weighted: True" summary line.
  None of it ever weighted anything: statsmodels' STL accepts no observation weights, the computed weights were dropped
  before the fit, and weighted and unweighted output were byte-identical. Nothing in the library, GUI, tests or examples
  used it. `robust=` is the real outlier knob and stays.
- **Four dead harmonic functions removed** (182 lines): `reconstruct_harmonics`, `periodogram`, `fft_decompose`,
  `multi_scale_harmonics`. None had a caller anywhere and none was in `dv.analysis.__all__`. `harmonic_analysis` and
  `spectrogram` are unaffected and stay exported.
- **`UstarDetectionMPT` removed** (653 lines) with its `dv.flux` export. `UstarMovingPointDetection` is the faithful
  ONEFlux port of the same algorithm and is what the chain, the bootstrap wrapper and the GUI use. The removed class
  read 11 attributes that were never assigned anywhere in it, and `run()` never called its own collectors, so its
  threshold was only ever printed, never stored.
- **`TimeLagAnalysis`'s `histogram_startbin` / `histogram_endbin` are renamed to `histogram_start_seconds` /
  `histogram_end_seconds`.** Both are compared against each histogram bin's inclusive start, so the values are lag
  values in seconds and never were bin indices; the names invited a caller to pass a count, which lands somewhere
  arbitrary on the lag axis. Passing an old name raises a `TypeError` naming its replacement, the way the outlier
  detectors answer their pre-unification parameter names, so existing code stops instead of analysing an unintended
  range. The GUI's two fields moved with them and now accept fractional seconds.
- Two smaller removals: the `'iterations'` key is gone from `stl_decompose`'s result dict (it returned
  `DecomposeResult.nobs`, a shape tuple, where a count was documented, and statsmodels exposes no iteration count to
  report honestly), and the unread `UstarVekuriThresholdDetection.bootstrap_results_` attribute is gone.

#### Results change, with no error and no warning

- **The MDS similarity window is trimmed at the ends of a record, not clipped.** ONEFlux narrows the window bounds and
  its diurnal method skips out-of-range positions, so a real record enters a fill at most once; diive folded every
  out-of-range offset onto record 0 or n-1 and counted the edge value hundreds of times in the mean, the SD and the
  count. The cascade's largest window reaches 427 days on each side, so the bias reached more than a year into each end
  of a record. Interior fills are bit-identical, which is why the ~5e-7 agreement with a native ONEFlux run never
  caught it. On a 40-day synthetic record a gap at index 2 moved from count 453 / fill -3.3071 / SD 0.5381 to
  count 120 / fill -4.0003 / SD 0.6638:
  duplicates carry no spread, so the reported uncertainty near the edges was understated by nearly half. Affects
  `FluxMDS`, `RandomUncertaintyPAS20` and the daytime-partitioning NEE uncertainty, which share the kernel.
- **u\* filtering now rejects a record whose u\* is missing.** The rejected set was `ustar < threshold`, and neither
  comparison matches NaN, so those records landed in neither set and their flag summed to 0: accepted, with turbulence
  unknown. ONEFlux rejects exactly these records (a missing u\* is `INVALID_VALUE` = -9999, below every threshold), so
  diive accepting them was a deviation rather than a choice. **u\*-filtered fluxes therefore lose those records, and how
  many depends entirely on how gappy the site's u\* record is.** A per-record threshold Series that does not cover every
  record now raises, which is what its docstring already required, instead of quietly rejecting the uncovered records;
  the in-chain VUT path builds a full-coverage series, so only a hand-built one can trip it.
- **Windowed FFT amplitudes are corrected for the window's coherent gain, and `harmonic_analysis` reads the bin it
  means to.** Both functions applied a window (hamming by default, mean ~0.54) and reported `2*|rfft(x)|/n` without
  dividing by the window mean, so every amplitude was about 54% of the truth and a reconstruction built from them was
  short by the rest, with the shortfall landing in `residual`. `harmonic_analysis` additionally indexed DC-stripped
  arrays with the full-rfft bin number, so a tone sitting exactly on its bin came back with amplitude 0, and its
  harmonics list disagreed with the spectrum in the same result dict. A 3.0 cosine now returns 3.0 under boxcar,
  hamming, hann and blackman.
- **`harmonic_decompose` selects spectral peaks instead of the strongest bins**, so a windowed component's leakage
  shoulder can no longer be returned as a second component. Two components (period 50 at amplitude 3, period 25 at
  amplitude 1) previously came back as period 50 twice, with the residual absorbing the component that was missed.
  Knock-on: the reconstruction is built from N distinct components rather than N bins, so it carries slightly less of
  the signal's energy, and **anything built on `features_stl` with the harmonic method shifts**. Used as a gap-filling
  feature it is marginally weaker: `test_gapfilling_stl_features_xgboost` moved mae 2.79 to 3.01 and r2 0.53 to 0.47,
  and its bound was widened. Fewer than `n_harmonics` components come back when the spectrum holds fewer peaks.
- **STL survives a gap.** statsmodels' STL has no NaN handling and propagates rather than raising, so one missing value
  in 1460 returned three all-NaN components, `seasonality_strength` 0.0 and a `summary()` of "nan ± nan", on the default
  path for real EC data, and against four docstring claims of gap tolerance. It now fits on a linearly interpolated copy
  (`limit_direction='both'`, since one leftover edge NaN is enough to poison the fit) and masks the gaps back out of all
  three components, so a gap in is a gap out and no interpolated value is ever returned. The count is reported as
  `n_interpolated`; an all-NaN series raises. **A gappy series that used to decompose to nothing now returns
  components**, which reaches `SeasonalTrendDecomposition` and, through the next item, gap-filling features.
- **`FeatureEngineer(features_stl=True)` gives gappy drivers their STL features.** The blanket skip of any column
  holding a single NaN existed only because STL could not fit around a gap, and it announced itself through `detail()`
  at DEBUG level, so asking for STL features at default verbosity produced none of them and no visible sign. Measured on
  a driver with 300 gaps, the components are NaN exactly where the source column is, so the feature costs no model rows.
  `'classical'` genuinely cannot decompose gaps; that column alone is skipped, now with a warning, plus a summary
  warning when no column could be decomposed at all.
- **`DriverAnalysis` (experimental): every number it reports moves.** `deseasonalize=True` interpolated every gap so
  statsmodels' STL could run and then returned the interpolated values as part of the decomposition, so fabricated
  target records entered every model matrix, including the chronological hold-out the class scores itself on: a 4-day
  gap came back with 0 NaN. The interpolation stays, since STL cannot fit without it, but is masked back out in the
  helper, which covers `_apply_deseasonalize`, `scale_resolved` and `granger` at once. Separately, per-scale and
  per-regime relevance was judged against the headline model's injected `.RANDOM` floor instead of each submodel's own,
  although each submodel is fitted on a different subset with its own noise scale. On the CH-DAV example month the
  per-scale floors span 0.013 to 0.392 against a global 0.246, so individual labels move, and since `regime_dependence`
  is the only trigger for `verdict='context_dependent'`, the headline verdict can flip with them.
- **`StratifiedAnalysis` analyses the records it used to drop and keeps the z bins it used to lose**, so both the curves
  and the bin count in an existing figure change. `dropna()` ran over every column of the input frame, so passing a
  working dataframe, which is the obvious call, cut 20 000 rows to 100 through one unrelated gappy column, and the
  on-plot count reported the survivors as if they were the data; `zvar` / `xvar` / `yvar` are extracted first now, and
  `n_records_input` / `n_records_used` say what was dropped. Per-bin results were also keyed by the bin aggregate
  rounded to 2 decimals, so adjacent quantile bins that rounded alike overwrote each other: 60 bins over a z range of
  0.1 stored 11 of them, with the legend blaming classes that were "not generated". Labels are now widened until they
  are distinct, so the legend still reads in real units.
- **Cells are no longer drawn over regions that hold no data.** `HeatmapYearMonth` handed the months that occurred to
  `pcolormesh` as cell boundaries while keeping regular 1..12 ticks, so a November to February record drew February
  across the region labelled March to September. `GridAggregator` reindexed onto the full bin lattice for
  `binning_type='custom'` only, although `min_n_vals_per_bin` can empty a bin of any type, so a bimodal variable over 30
  equal-width bins collapsed to 5 columns and its empty middle rendered as solid data (which also defeated the X/Y/Z
  surface's `_drop_gap_risers`). Fixed in `GridAggregator`, so `HeatmapXYZ` and the 3-D X/Y/Z surface both benefit, and
  its wide output now carries every bin the binning defined.
- **`WindDirOffset` honours `hist_n_bins`, and its bins span the full circle.** The per-year histogram was hardcoded to
  360 bins while the reference histogram used `hist_n_bins`, so any other setting left the two `COUNTS` series with
  different lengths and `Series.corr()` aligned them on the RangeIndex: the best offset was picked by correlating a
  truncated, mismatched bin set. The edges are now pinned to `linspace(0, 360, hist_n_bins + 1)` rather than derived
  from each subset's own range, since correlating bin by bin only means something once bin *i* covers the same
  directions in both. The wrap is `% 360`, which handles any offset magnitude and maps an exact 360 to 0.
- **BUR08 applies the water-vapour dilution factor** `(1 + 1.6077 rho_v/rho_d)` that BUR06 and JAR09 already applied,
  settled from Burba et al. 2008 and the LI-7500 poster, which give the already-WPL-corrected form diive implements with
  the factor present. `correction_method_base` had been changing two things at once: the surface-temperature model, and
  whether the factor applied at all. On CH-LAE 2016-2017 the BUR08 correction term rises 1.16% in the mean and 1.31% in
  the median, with one record fewer, since a record without humidity cannot carry the factor.
- **The self-heating gap-fill actually fills gaps, and stops deleting measurements.** `_gapfill()` dropped rows with a
  NaN in any column including the target, which is exactly the rows to be filled, so XGBoost reported "Filling 0 missing
  records" and the console blamed insufficient drivers (CH-LAE June to August 2016: 497 gaps in, 497 out, now 0). A
  record with a measured flux but no correction term was deleted by `flux + NaN`; it is carried through uncorrected
  instead, with a new informational `FLAG_NEE_OP_CORR_ISCORRECTED` (1 corrected, 0 carried through, NaN no measured
  flux) and a warning giving the count. An unrecognised `correction_method_base` raises instead of leaving the
  correction empty and completing without a word, and classes with fewer than `MIN_ROWS_PER_CLASS` complete records are
  reported rather than silently taking a neighbouring class's scaling factor.
- **The 3000-record minimum applies to every u\* detection entry point.** `detect()` enforced it while `bootstrap()` and
  `bootstrap_annual_samples()` reached the internals directly, and those are the paths `UstarBootstrapThresholds` and
  the chain's L3.3 detection actually call: at 1000 records `detect()` raised and `bootstrap()` returned 0.4197. The
  check moved into the one point every public entry passes through. A window that finds nothing now reports why at
  `warn` level, where "too few records", "mistyped column name" and "detection genuinely found nothing" had been
  indistinguishable from each other and from bare NaN thresholds. `annual_thresholds_` holds NaN after a failed
  detection instead of the `THRESHOLD_NOT_FOUND` sentinel 10.0, a plausible-looking threshold that would filter out
  every record. Class binning now follows the C tie-extension branch diive was missing and treats an inverted class as
  empty, the way the C accumulation loop does; on CH-LAE no final threshold moves, so that one corrects an intermediate
  value.
- **Random uncertainty method 4 averages 10 neighbours, not 9.** The slice took 5 below the target and only 4 above,
  skewed toward lower fluxes, against a documented 10. It affects only records that fall through to method 4: about
  0.01% on CH-DAV, 2 of 21 000 on the shipped example. Methods 1 and 2, the ONEFlux port, are untouched.
- **A joint-uncertainty cumulative record with incomplete scenario inputs is NaN** rather than a wrong number. The
  random term was masked to available flux and the scenario term was not, so a record carrying only one percentile moved
  one cumsum and not the other, and with a narrow band the running spread could come out negative. Both documented
  cumulative conventions are preserved.
- **`detect_seasonality` raises instead of fabricating a result.** With no candidate period in `[2, max_period]` it
  returned `primary_period=365`, `secondary_periods=[7, 30]` and `strength=0.0`, so a failed detection was
  indistinguishable from a real answer, and `SeasonalTrendDecomposition._get_seasonal_period` calls it whenever
  `seasonal_period` is not supplied: a 5-point series was decomposed at period 365. The message names the range, the
  number of valid values and the way out. The "no peaks, use max power" branch stays; that one is a real answer.
- **`CompoundExtremes` raises when nothing can be classified.** With the documented defaults (`agg='monthly'`,
  `standardize_by='season'`) a single year gives every calendar-month group one member, so the ddof=1 standard deviation
  is NaN, every period drops out, and the empty result was the same output a record genuinely holding no extremes would
  give. The message names the cause, how many years the record spans, and `standardize_by='record'`. A partial loss
  warns with the count instead of passing unmentioned: on CH-DAV a 13-month slice keeps 2 periods and warns about the 11
  it dropped.
- **`analyze_highest_quality_flux` counts only measured records.** It reported 1519 of 3000 records as "Valid records:
  2986 (99.5%)", in the console and in the public `summary` dict alike. The flag change above fixes the count; the rates
  were wrong independently, since they used the potential-record count as the denominator although a record with no
  value was never a candidate for being an outlier. Both rates are per measured record now, with `measured_records`
  reported alongside.
- **Frequency detection counts intervals, not rows.** The modal-delta count was divided by `len(df)`, so
  `DetectFrequency.percent_matching` and `TimestampSanitizer.get_status()['frequency_percent_matching']` came out one
  interval short (99.0 for a perfectly regular 100-record index) and a clean 2-record series was rejected as too
  irregular. Both values were additionally recovered by parsing them back out of a `'{:.0f}% occurrence'` string, so a
  genuine 99.9% read as an indistinguishable 100.0; the exact fraction is carried internally now, and the public tuple
  and its display string are unchanged. The detected frequency itself does not move for normal-length data: identical on
  all four bundled datasets, and a 3000-trial randomised sweep found one arithmetically constrained change, even *n*
  where the top delta occurs exactly *n*/2 times, which is a genuine majority of intervals that the old denominator
  rejected outright. So the change only adds detections. A 1-record index now reaches the informative `RuntimeError`
  instead of a bare `KeyError`.
- **An unset bound in `keep_records_where` is open for every `inclusive` value.** It substituted the condition's
  observed `min()` / `max()`, so `inclusive='neither'`, `'left'` or `'right'` excluded the extreme record although the
  docstring promised that side was open; `-inf` / `+inf` are used now. Reachable straight from the GUI's Select records
  by condition tab, whose shaded preview and result now agree, and `select_records_to_code` inherits the fix because it
  omits an unset bound.
- **The 3-D date/time surface sits on the heatmap's time axis.** `datetime_surface_grid` sanitized the index but never
  converted it to `TIMESTAMP_START` as `HeatmapBase._setup_timestamp` does, so under diive's MIDDLE working convention
  the surface's time-of-day axis sat half a period from the heatmap's for the same data, against the function's own
  claim to use "the same preparation the 2-D heatmap uses". Two assertions in
  `test_datetime_surface_grid_shape_and_axes` encoded the offset and moved with the fix.
- **A variable named `DATE` or `TIME` is no longer destroyed by the pivot helpers.** In `datetime_surface_grid` this
  raised a `TypeError`; in `HeatmapDateTime` it did not, because without `dtype=float` the array stays object dtype, so
  the heatmap silently painted the timestamps and produced a plausible-looking wrong figure. Both pivot through an
  internal value key now.
- **The glTF `.glb` export no longer mirrors its texture along the date axis.** glTF and trimesh sample from a
  lower-left texture origin, so vertex row *i* sampled texel row *d-1-i*: the geometry was right while an exported
  annual NEE surface painted the winter ridge with the summer colours. Both export paths, smooth and extruded, were
  affected.
- **`transform_yearmonth_matrix_to_longform` accepts a matrix with months missing**, which its own producer emits for a
  seasonal record and its own docstring example shows. Note that it now always returns a full-year span, so a partial
  year that happened to be contiguous is padded to 12 months with NaN.
- **`crosscorr` writes a NaN row for three early-outs that used to omit the date entirely**, so a caller aligning the
  result to a full date index no longer finds holes where it expects NaN, and an all-omitted result still carries the
  `max_corr` column its plotter looks for.

#### Crashes on documented input

- `Hampel` on any non-fixed frequency (monthly `MS`, yearly `YS`, business-day, weekly). `index.freq.nanos` raises for
  those offsets, and the median-of-diffs branch beside it already handled them. pandas 3 offers no non-raising fixedness
  test, so the attempt is the test. The fixed-frequency path is unchanged code, so no existing number moves.
- `run_level31(set_storage_to_zero=True)` without a storage column, which is the exact case the flag documents for H and
  LE: `KeyError: "['SLE_SINGLE'] not in index"`. The correction, the report and the plot all cope with no storage term
  now; the real storage paths are untouched.
- `ScopApplicator` with any legal input series name. It read incoming series through hardcoded canonical names, so what
  `ScopPhysics.run(gapfill=False)` produces (`FCT_UNSC`), `physics.fct_unsc_gf` (`FCT_UNSC_gfXG`) and a day/night flag
  named anything but `DAYTIME` all raised `KeyError`. Both names are normalised at the boundary, which is where the
  class's own `ColumnConfig` convention was meant to apply.
- `TimeLagAnalysis` on a gas with few distinct lag values. The default fringe trimming (`[5, 10]`) removed every bin of
  a `n_unique - 1` bin histogram and `peakbins[0]` raised a bare `IndexError` that escaped `analyze_all_gases` and
  `plot_all_gases` despite their documented warn-and-continue. Trimming is skipped for that gas with a warning, since
  trimming anyway would return a peak from a partly-trimmed histogram, turning a crash into a quietly wrong number. A
  `histogram_start_seconds` / `histogram_end_seconds` range that excludes every lag emptied `results` the same way
  through different parameters.
- `UstarThresholdConstantScenarios.calc(showplot=True)` on pandas 3, at two sites rather than the one the finding named:
  `counts.div(counts[0])` indexes a label-keyed Series positionally. Display only, so no threshold moves.
- `UstarVekuriThresholdDetection.summary()` before `detect()`, where `results_` was initialised as a dict, so the guard
  that exists to print "run detect() first" raised `AttributeError` on `.empty`. It now behaves like the sibling ONEFlux
  port rather than raising. The documented `bootstrap_stats_` attribute was never assigned; `bootstrap()` assigns it.
- `RidgeLinePlot` on any series containing a gap, since NaN went straight into `KernelDensity.fit`. Dropped once in
  `__init__`, so a group left with no valid values simply gets no ridge, and an all-NaN series raises with a message
  saying why. The GUI path worked only because its codegen and tab dropped NaN first, so the public API was strictly
  worse than the GUI's.
- `Cumulative.plot` and `CumulativeYear.plot` on an all-NaN column, normal for a scenario column that was never filled:
  the legend label indexed `dropna().iloc[-1]`. Such a column is labelled "no data" and its end marker skipped, so the
  legend keeps one entry per column.
- `transform_yearmonth_matrix_to_longform` on any matrix not produced by `resample_to_monthly_agg_matrix`: unnamed axes
  gave `KeyError: None`, any other naming `KeyError: "['YEAR', 'MONTH'] not found in axis"`. The axis names are pinned
  on a copy, leaving the caller's frame alone.
- `MultiDataFileReader` when every input file is empty or the file list is empty, which reached
  `sort_multiindex_columns_names(df=None)` and an `AttributeError`. It raises with two messages, naming either "no files
  given" or "all files empty" and quoting the first path so a bad glob is visible. Returning an empty frame was rejected
  as the fix: it has no timestamp index and would arrive in the GUI as an inexplicably blank dataset.
- `harmonic_decompose` returned `frequencies` one element longer than the arrays it is documented to pair with, so
  `plot(frequencies, amplitudes)` raised.

#### Reporting, contracts and dead code

- **`detail()` output was unreachable at every verbosity setting.** It defaulted to `verbose=VERBOSE_PROGRESS` (2) while
  its own `min_level` is `VERBOSE_DEBUG` (3), so a bare `detail(msg)` printed at no setting at all, and 25 debug lines
  across the library were lines their authors believed they had written. Passing `verbose=` at each site was not
  available, since 24 of the 25 have no `self.verbose` and no `verbose` parameter, so the helpers now default to
  `verbose=None`, meaning "use the module default", and that default is settable: **new `dv.set_verbosity(level)` and
  `dv.get_verbosity()`, exported at top level.** Behaviour at the default is unchanged (`detail` quiet, `info` printing)
  and an explicit `verbose=` still wins. The CLAUDE.md convention that produced the defect, calling helpers without
  `verbose=` inside a `if self.verbose >= N:` guard, is corrected.
- **Three quality-flag contracts corrected.** `flag_ssitc_eddypro_test` documented a conversion it does not perform, but
  the code is right: EddyPro SSITC is already 0/1/2, so the conversion is the identity and promoting 1 to 2 would reject
  records FLUXNET treats as usable, which is what `setflag_timeperiod` exists for. The docstring was fixed instead.
  `FlagQCF`'s documented "NaN if no flag available" branch is genuinely unreachable and should stay so, since a NaN QCF
  would pass both of `_calculate_series_qcf`'s filters; it is documented as never-NaN with a test to keep it that way.
  A scalar `0` flag code no longer becomes NaN through a `'0.0'[1]` round-trip, which is a reporting fix, not a
  screening change, since flag sums count only 1 and 2.
- `ScopPhysics` documented "a hybrid approach using Random Forest and Mean Diurnal Variation" and printed "-> Imputed
  (RF + MDV)"; the gap-fill is XGBoost with no MDV stage, and MDV lives in `ScopApplicator`, which the docstring now
  points at. The results column is renamed from its legacy `FCT_UNSC_gfRF` to
  `FCT_UNSC_gfXG`, so the attribute and the frame agree; see *Breaking Changes*.
- `TimeLagAnalysis`'s class docstring stated three parameter facts the code contradicts: `ignore_fringe_bins` defaults
  to `[5, 10]` and counts leading and trailing bins rather than naming indices, `zoom_margin` defaults to `[0.5, 1.5]`,
  and `histogram_startbin` / `histogram_endbin` are lag values in seconds, hence floats. Those two names were
  subsequently renamed to state the unit, see *Removed or renamed API* above.
- `reconstruct_from_components` masked every reconstruction with the trend's NaN regardless of `components_to_use`, so a
  seasonal-only reconstruction from a classical decomposition lost the trend's edge records (30 of 400 at period 31) and
  disagreed with `detrend()` on the same request. The mask was redundant where the trend was included, since gaps
  propagate through the arithmetic anyway. `seasonality_strength`'s docstring described a formula the code does not
  compute; it computes the Wang/Hyndman variant.
- `show_less_xticklabels` is applied by `HeatmapDateTime`, which accepted, documented and forwarded it while only
  `HeatmapYearMonth` read it. The GUI exposes it from a shared checkbox and codegen emits it into copied snippets, so
  removing it was not an option.
- `GridAggregator` keys its working frame by internal `_x` / `_y` / `_z` names, the pattern `ScatterXY` already uses, so
  two roles sharing a Series name no longer collide: a z sharing the x name replaced x and the plot carried z's range on
  the x axis, and x and y sharing a name raised "Grouper not 1-dimensional". Public naming is unchanged.
- The `FeatureEngineer` rolling stages no longer re-engineer their own output. They were the only per-column stages
  without a dot-prefix filter, so feeding the engineer a frame that already contains engineered columns, which the GUI's
  Feature engineering tab makes easy, produced names like `..TA_POL2_MEAN4`. No library path changes: the rolling stages
  receive `df[original_input_features]` and no caller passes an engineered column, so no model's features, SHAP values or
  `reduce_features` selection move.
- `MetadataStore.rename` raises on a collision instead of collapsing two entries and losing the loser's metadata,
  provenance and tags. The collision is judged on the resulting names, so a swap `{A: B, B: A}` and a whole-set prefix
  rename still work, and it is checked before any mutation, so a rejected rename leaves the store untouched.
  `MainWindow._rename_variables` validates its mapping the same way rather than handing the frame duplicate column
  labels.
- Smaller ones: `LocalSD` no longer leaves a value sitting exactly on the limit in neither `ok` nor `rejected` (no flag
  count moves, since `rejected` is untouched); `vectorize_timestamps` returns `.SEASON` as a plain integer instead of a
  nullable `Int64`, which had been forcing an object-dtype array into every ML fit (RF and XGBoost predictions are
  bit-identical, only the hidden conversion is gone); `lagged_variants`' closing message no longer claims the edge fill
  is unconditional, since it is correctly skipped for a gappy source, where an edge fill cannot be told apart from
  inventing driver values; `sort_multiindex_columns_names` no longer reverses the columns it moves;
  `gapfilling/interpolate.py`'s verbose header no longer prints a literal `{limit}`, and its dead `_calculate_gap_sizes`
  is gone; `convert_ts_to_timezone` accepts the `DatetimeIndex` its docstring promises; `MultiDataFileReader.metadata_df`
  tests its own frame rather than the data frame; and the USTAR docstring examples import from `dv.flux` and run.

#### Desktop GUI

- **The NEE partitioning tabs refuse to run without site coordinates.** The lat/lon/UTC spin boxes default to 0/0/0 and
  `_seed_site` left them there when the project site is unset, so a run partitioned at (0, 0) on UTC and returned
  plausible-looking GPP and RECO. The guard covers `_python_code` too, which would otherwise emit a runnable snippet
  carrying `lat=0.0, lon=0.0`. It applies to three of the four tabs: Daytime ONEFlux splits on measured Rg, not solar
  geometry, so it reads no coordinate.
- **An outlier tab discards a result whose dataset changed mid-run.** Detection runs on a background thread, so creating
  a feature, narrowing the date range or deleting a variable during a run left `_on_done` indexing the new frame with
  the old variable: `KeyError` when the column was gone, and no exception at all when the range had merely changed, in
  which case the mismatched result was adopted silently. The run's index travels in the payload, so the worker still
  writes no tab state. `_rerender_last` is tightened the same way; its existing guard covered only the missing-column
  half.
- **A saved control that could not be restored is reported.** `restore_controls` kept the widget's current value when a
  saved combo entry was gone, so reopening a project whose columns or preset labels were renamed left a tab pointing at
  a different input with no indication. The worst case is numeric: the joint-uncertainty divisor falls back from
  `JOINT_DIVISOR_IQR` (1.349) to `JOINT_DIVISOR_1SIGMA` (2.0). The unrestored keys are surfaced in the tabs where the
  fallback changes a number (random and joint uncertainty, and the four partitioning tabs). The return value is purely
  additive, so callers that ignore it behave as before.
- **Pinned tabs are frozen.** `_add_features` and `_sync_event_columns` wrote columns in place, so a pinned tab holding
  the same frame object saw them anyway, while column drops rebind and did not leak, making the freeze inconsistent in
  both directions.
- **`WorkerRunner` clears `is_running` on the GUI thread**, just before emitting `done` / `failed`, so a caller using it
  as a re-entry guard can no longer start a second run whose result interleaves with the first. `_screening_base` hand
  rolls its own thread and needed its own guard: it had been starting an unbounded number of CPU-heavy chains on rapid
  chain edits. `_outlier_base._compute_payload` no longer writes tab state from the worker thread; the daytime mask
  travels with the progress signal.
- **The 3-D export buttons no longer write the previous variable's relief** under the current variable's filename after
  a render that produced nothing, either an all-NaN variable or a Z role that is not a real column. The rolling cell
  aggregator uses an *n*-row window for even *n*, as its docstring and its tooltip both say.
- `save_config` catches `TypeError` and `ValueError` as well as `OSError`, which is what `json.dumps` raises, so one
  unserializable value no longer costs the whole file. Project load no longer syncs the outgoing session's `EVENT_*`
  columns onto the incoming frame before replacing them.
- **A test fixture now fails a test when Qt swallows a slot exception.** PySide6 routes an exception raised inside a slot
  it invokes to `sys.excepthook` and returns normally, so a test that drives behaviour through a signal and then only
  checks that nothing broke passes even when the slot crashed. The fixture immediately exposed a real defect:
  `VariablePanel` connected a lambda to the process-wide `metadata_store.manager.changed` signal, which Qt cannot sever
  when the receiver dies, so every closed tab left a dead slot firing. 44 tests were passing over 168 swallowed
  `RuntimeError`s from that one line, and in the running GUI it raised and swallowed one per closed tab on every
  metadata edit. Seven tests whose only evidence was a stale figure standing after a failed render now assert concrete
  post-conditions.

#### Bundled example data

- **Six of the eight example parquet files had never been committed.** A `*.parquet` ignore rule added in January 2025
  meant `git add` silently skipped them, so only a working-directory copy existed: the documented
  `load_exampledata_parquet_lae*` loaders raised `FileNotFoundError` for anyone installing diive, 8 examples did not
  run, and once that local copy went, 23 tests failed or errored (the whole GUI flux chain tab, all of
  `test_ustar_mp.py`, the L2 custom columns). Recovered from the v0.91.0 GUI build, which had frozen a copy at packaging
  time. Every file now carries an explicit exception to the ignore rule so it cannot hide them again. 33.6 MB total, of
  which 15.4 MB was already tracked.

#### One deviation from ONEFlux, kept on purpose

- **u\* filtering is `ustar >= threshold` and nothing more.** ONEFlux, and Pastorello et al. 2020, additionally discard
  the first half-hour above the threshold following a period below it, to avoid the false emission pulse from CO2 that
  accumulated under the canopy and flushes past in one burst. diive keeps that record, favouring data availability, and
  leaves the trade-off to the user, who can drop those records. It is documented in
  `FlagMultipleConstantUstarThresholds` with the reason and the reference, and pointed at from the single and variable
  flaggers, so it cannot be mistaken for an oversight. Everything else in the comparison follows ONEFlux, including the
  missing-u\* rejection above.

#### Still open

16 findings are open, each with a repro or a source citation in `CODE_REVIEW_FINDINGS.md`. The three that can move a
number: `UstarVekuriThresholdDetection.bootstrap()` calls `df.sample` with no `random_state`, so its percentiles differ
from run to run (one season's p50 moved 0.1634 to 0.1488 across two runs) and those thresholds feed u\* filtering;
`classical_decompose` passes `extrapolate=` where the parameter is `extrapolate_trend`, so it always raises internally,
always falls into the no-extrapolation fallback, and its trend edges are unconditionally NaN; and
`ScreeningTabBase._select` does not bump `_run_id`, so switching the selected variable mid-run can adopt the previous
variable's chain result. Two need an external source: whether InfluxDB v2's delete range excludes `stop`, in which case
the pre-upload delete leaves the last record and duplicates survive, and BUR06/JAR09's aerodynamic resistance, where
diive passes the canopy momentum resistance `u/u*^2` in place of Burba 2006's per-element `7.4*sqrt(d/U)` (about 6x
apart on CH-LAE) and has no equivalent of `fr`, the fraction of instrument heat retained in the optical path. Neither
resistance issue is fatal in the pipeline, because `ScopOptimizer` fits a scaling factor against a closed-path
reference and absorbs both, but it makes those two methods semi-empirical rather than the published formulation, which
matters for anyone running `ScopPhysics` without the optimizer.

### Unittests

- **New `tests/test_codegen.py` covers the `*_to_code` script generators.** 47 of the 55 had no test, and the 8 that
  did were either the six flux-chain ones in `tests/test_flux_codegen.py` or reached only incidentally by
  `tests/test_gui.py` — `core/plotting/codegen.py` read 94% covered while nothing asserted anything about it. Each
  generator is now checked three ways: the snippet compiles, it contains the call it claims to reproduce, and every
  keyword it passes to a `dv.*` callable is accepted by that callable's signature. The third check is the one that
  matters — a renamed library parameter leaves a snippet that still compiles but raises `TypeError` when a user runs
  it, which compiling alone cannot see. `**legacy` is deliberately not treated as a wildcard: `reject_legacy_params`
  raises on unknown names, so the named parameters are the real accepted set, and without that carve-out the check
  skipped every outlier-detector constructor. A completeness test fails if a new `*_to_code` function lands without a
  test. 67 tests, ~3 s, no GUI or data required.
- **`TestPlotClasses` in `tests/test_plots.py` covers the plot classes that no non-GUI test reached.** The file tested
  five classes; `HeatmapDateTime` (used by 16 of the 113 examples), `HeatmapYearMonth`, `Cumulative`,
  `CumulativeYear`, `RidgeLinePlot`, `TreeRingPlot`, `ShiftedDistributionPlot`, `WaterfallPlot`,
  `LongtermAnomaliesYear` and `datetime_surface_grid` were executed only incidentally by `tests/test_gui.py`. The
  fixture is a deterministic synthetic three-year hourly series, so aggregates are exact expected values: the
  `Cumulative` curve must end at the series sum, the waterfall budget must close on the total, and mean/max/ranks
  aggregation must give genuinely different meshes. `RidgeLinePlot`'s test pins the documented gotcha that `hspace`
  has to reach the gridspec at creation. Without any GUI-test contribution these modules now run 80-100% covered
  (`surface_grid.py` 100%, `heatmap_datetime.py` 91%, `ridgeline.py` 91%, `cumulative.py` 87%, `treering.py` 80%).
- **`tests/test_corrections.py` grew from 2 tests to 26**, covering all ten public symbols of `dv.corrections`. The
  `diive/corrections/__init__.py` namespace module was at 0% — no test imported it, so its `__all__` was unverified
  and a dropped re-export would have kept the suite green; it is now 100%, as is the `apply_corrections` dispatch
  table (previously reached only by `tests/test_gui.py`), with `offsetcorrection.py` at 98% and `setto.py` at 94%.
  Each dispatch key is compared against calling the underlying function directly rather than merely running, every
  `CorrectionSpec` key in `dv.qaqc.CORRECTIONS` is asserted dispatchable, and the nighttime offset is injected at a
  known constant so the detected daily offset is an exact expected value.
- **The `dv.qaqc` measurement registry and `dv.variables` classification are tested.** Both read as well-covered
  (98% and 97%) while being verified by nothing — every covered line came from `tests/test_gui.py`, which drove them
  without asserting on the answers. `tests/test_qaqc.py` went from 3 tests to 14, `tests/test_createvar.py` from 21 to
  41; `measurements.py` and `classification.py` are now at 100% from real tests. The assertions cover the two
  documented traps — `SWC` must beat `SW` in `detect_measurement`, and `FC` must not swallow `FCH4` in
  `classify_variable` — plus cross-checks that every code `detect_measurement` returns exists in `MEASUREMENTS` and
  every key `corrections_for_measurement` returns exists in `CORRECTIONS`. `combine_variables`, `auto_pick_column`
  and `daytime_nighttime_flag_from_swinpot` are covered too.
- **`tests/test_imports.py` now verifies the whole public namespace surface** — 515 subtests covering every symbol of
  all ten namespaces. Nothing checked these lists before, so a symbol dropped from a re-export would have vanished
  from the public API with the suite still green. The test is driven off `_LAZY_SUBMODULES` rather than a hard-coded
  list, and additionally enforces the four-place namespace registration CLAUDE.md documents: `_LAZY_SUBMODULES`, the
  `TYPE_CHECKING` block, `diive.__all__`, and `packaging/diive_gui.spec`'s `hiddenimports`. The last one otherwise
  fails **only in the frozen GUI build**, since PyInstaller cannot follow a PEP 562 `__getattr__` — an unlisted
  namespace passes every test and every dev run, then is missing from the packaged app.
- **The non-GUI test suite runs in half the time** (402 s -> 199 s). One `setUpClass` in
  `tests/test_driveranalysis.py` cost 220 s — 55% of the whole non-GUI suite — by running `DriverAnalysis` at static
  + temporal levels over a 4-month fixture. Row count is what costs: every temporal stage ends in a TreeSHAP pass
  over the full matrix and more rows also deepen the forest, so 2x the data costs 7.5x the time, while the lag span
  is nearly free. The fixture is now 2 months (matching the static class) with the lag range unchanged: setup 220 s
  -> 27.6 s. Ten convergence/verdict branches that `months=4` happened to reach are no longer covered; no test
  asserted on them, and they are noted in `COVERAGE_GAPS.md` as needing deliberate tests.
- **`GapFillingResult` and `prediction_scores` are tested.** `GapFillingResult` is the documented return type of
  `.results` on every gap-filler; `core/ml/results.py` read 100% covered while every one of its lines came from
  `tests/test_gui.py`, and `core/ml/scores.py` had no test at all. Both are now at 100% from real tests. The ML
  contract is pinned (no NaN in `gapfilled`, `flag` in {0,1,2}, observed records never overwritten, all seven metrics
  in `scores` and `scores_traintest`, `model`/`feature_importances` populated), as is the reduction-field behaviour
  and the MDS variant where the regressor-only fields stay `None`. `prediction_scores` is checked for argument order:
  `r2` and `mape` are asymmetric in (true, predicted), so a swapped internal call changes them while the other five
  metrics do not — verified by mutation.

- **The flux-chain re-run cascade is tested.** Re-running a level invalidates that level and every later one; without
  it a second `run_level2` would concat duplicate column labels into `fpc_df`, leaving ambiguous lookups and stale
  flags for `FlagQCF`. The behaviour was documented but untested, its coverage arriving only from the GUI driving
  levels repeatedly — `levels/_rerun.py` goes from 87% incidental to 98% from real tests. Covered: the cascade and
  its column-drop, container purity across a re-run, the `filteredseries` fallback to the newest surviving level,
  clearing of the additive L4.1/L4.2 state, and `drop_columns_for_key` keeping gap-filling methods independent.
  Mutation-tested by neutering `cascade_reset`.
- **`stl_decompose` has regression tests for the two bugs fixed in it** (`core/times/decomposition_utils.py`, now 84%
  on that function). The wrapper never passed `seasonal` through as statsmodels' `period`, so the caller's cycle
  length was ignored, and it called `STL.fit(weights=...)`, which statsmodels does not accept. Both are pinned by
  restoring the original bug in the source and confirming the test fails: a known 24-step cycle must come back with
  lag-24 autocorrelation above 0.99 (correct period gives 0.9999, a wrong one 0.005, with a control test proving the
  assertion is not vacuous), and a `weights=` call must not raise. Additive reconstruction, trend-window
  normalisation, argument validation and the short-series warning are covered too.
- **`FlagQCF` goes from 53% to 95%** (`preprocessing/qaqc/qcf.py`). The existing tests covered aggregation, the
  filtered series and the OVERALL screening report; the whole `swinpot_col` day/night path, all three console reports
  and both plot methods had none. Added: the `> 3 soft flags` boundary the old tests jumped over (exactly 3 must
  still be QCF 1), the day/night acceptance thresholds (a stricter daytime threshold promotes marginal daytime
  records while nighttime survives), the DAYTIME/NIGHTTIME split in the screening report, the three reports via a
  console sink, both plot methods, and the `KeyError` validation. Mutation-tested by breaking three QCF rules in
  turn. Two bugs found while writing these are recorded in `COVERAGE_GAPS.md`: the reports crash on a cp1252 stdout,
  and omitting the optional `idstr` yields column names containing a literal "None".
- **New `tests/test_timestamp_shifts.py` takes `DetectTimestampShifts` from 0% to 92%** — the largest library file no
  test executed at all (281 statements), despite having a worked example. The tests plant a known clock offset in
  noise-free synthetic radiation and check each of the three methods recovers it, which validates the algorithms
  rather than their plumbing — and immediately exposed that `crosscorr` cannot (see Known issues). Construction
  (auto-computed vs supplied potential radiation, coordinate validation), the clearness filters, all five plot
  methods and the timedelta formatter are covered too.
- **`pytest-cov` added to the `dev` group**, giving the first line-coverage baseline: 57% library, 68% GUI, 62%
  combined. Gaps are catalogued in `COVERAGE_GAPS.md`.
- **`tests/test_echires.py` left with the code it tested** and runs in dyco as `tests/test_pwb.py`. The suite goes from
  727 to 653 tests, which is exactly that file's 74. Nothing was lost.

## v0.90.0 | 13 Jan 2026

**Feature Highlights and Logic Changes**

### Time Series and Date Handling

* **Vectorize timestamps**:
    * Renamed function `include_timestamp_as_cols` to `vectorize_timestamps`.
    * The function now supports generating sine/cosine variants for cyclical timestamp attributes (e.g., DOY) (18).
    * Added new notebook `VectorizeTimestamps` (17).
    * Added new unit test `test_vectorize_timestamps` (30).
    * The same function is used as parameter `vectorize_timestamps` in machine learning approaches to include timestamp
      attributes in feature vectors, i.e., timestamp info is converted to columns. Notebooks and unit tests that use
      XGBoost or random forest were updated (if necessary) and re-run accordingly (31)(32)(33)(34)(35)(36).
* **Insert season**: Refactored `insert_season` for better performance (11).

### Outlier Detection

* `LocalSD`:
    * Added an option to `LocalSD` to run the outlier filter separately for daytime and nighttime
      periods (13).
    * Updated `FluxProcessingChain` (15) and `StepwiseMeteoScreeningFromDatabase` (16) notebooks, as well as relevant
      unit tests (14), to support the new `LocalSD` functionality.
* `HampelDaytimeNighttime`:
    * Refactored `HampelDaytimeNighttime` outlier removal method, it now runs 100x faster.
    * Also added parameter `use_differencing` to calculate outliers from the double-differenced time series instead of
      the original data (23).
    * Also added parameter `separate_day_night` to run the filter without the separation into daytime/nighttime data.
    * The original Hampel class is therefore now implemented here and was removed (24).
    * Updated notebook accordingly (37).
    * Also added unit test for double-difference option (25).
    * Also added unit test for basic Hampel filtering (26).
    * Removed old Hampel test case (27).
    * Removed old notebook for Hampel filtering.
    * The filter was also implemented in step-wise outlier detection (28), flux processing chain (Level-3.2) (29) and in
      meteoscreening from database (38).
* All outlier detection classes: Harmonized the creation of daytime/nighttime flags across all outlier detection
  methods (12).
* `StepwiseOutlierDetection`:
    * Fixed `flag_outliers_hampel_test`: the `n_sigma_daytime`/`n_sigma_nighttime` parameters defaulted to the literal
      `5.5` instead of `None`, which shadowed the global `n_sigma` so changing it alone had no effect in
      daytime/nighttime mode. They now default to `None` and fall back to `n_sigma`, matching the `Hampel` class.
    * Added the missing `flag_missingvals_test()` method (it was documented in the class but not implemented).
    * Added the `output_middle_timestamp` parameter (default `True`, unchanged behaviour). Set it to `False` to keep the
      input timestamp convention (e.g. `TIMESTAMP_END`) instead of shifting to the middle of the averaging period, so the
      resulting flags align to an existing dataframe on merge.
    * Fixed the `last_flag` property guard, which never raised when no test had been run yet.
    * Extended the example to show how to calculate the overall quality flag (`QCF`) from the accumulated test flags.

### Eddy Covariance and Flux Processing

* **Flux detection limit**:
    * Refactored `FluxDetectionLimit`.
    * **Important**: The logic for time lags has been inverted to be more intuitive; a positive time lag now means the
      lagged variable (e.g., a gas) lags *behind* the reference variable (turbulent vertical wind) (9).
    * **Performance Boost**: Maximum covariance for high-res data is now calculated using the [polars](https://pola.rs/)
      library. ~3x speed improvements on half-hourly 10Hz data, depending on CPU (4).
    * Added new notebook `FluxDetectionLimit` (8)
    * Added unit test (10)
* **Self-heating correction for fluxes from open-path IRGAs (SCOP)**:
    * Added classes in `selfheating.py` for the correction of the self-heating effect in open-path infrared gas
      analyzers, based on parallel measurements from an (en)closed-path IRGA (40).
    * Generally, the SCOP code implements a physics-based correction to remove spurious flux biases caused by instrument
      surface heating observed for eddy covariance fluxes from open-path infrared gas analyzers. It calculates an
      unscaled correction term based on environmental drivers like air temperature and wind speed, optimizes it using
      parallel enclosed-path reference data through bootstrapping, and applies the final scaled correction to produce
      corrected gas flux measurements.
    * There are several classes: `ScopPhysics` implements the physical modeling of instrument self-heating for open-path
      IRGAs,`ScopOptimizer` optimizes scaling factors for the self-heating correction using parallel enclosed-path
      reference data and statistical minimization, and `ScopApplicator` applies the optimized scaling factors to
      open-path flux data. These three classes are designed to be used in a pipeline, see notebook examples.
    * Added 2 notebooks:
        * `SelfHeatingCorrectionNEE_1_CreateScalingFactorsTable` shows how the table of scaling factors is created and
          applied during a time period of parallel measurements (41).
        * `SelfHeatingCorrectionNEE_2_ApplyScalingFactors` shows how a previously created scaling factors table is
          applied to open-path flux data outside the time period of parallel measurements (42).
* Update notebook `FluxProcessingChain` (15)
* Added option to set storage to zero when applying storage correction in Level-3.1. If *True*, sets the storage term to
  zero, in which case the storage data in the dataframe is ignored. Normally not needed, but can be useful during
  testing or when developing a correction method for FC (the CO2 flux not corrected for storage) but still needing
  outlier-removed values from the FluxProcessingChain. (22)

### Physics and Variable Conversions

* **New Function**: Added functionality to calculate air temperature derived from sonic temperature (5). Added
  `Calculate_air_temp_from_sonic_temp` notebook (7) and associated unit tests (6).
* Added new function `potrad_eot` for an alternative to `potrad` to calculate potential radiation. Takes into account
  the equation of time. (19)
* Added new function to calculate `aerodynamic_resistance` (21)
* Added new function to calculate `dry_air_density` (20)

### System and Visualization Improvements

#### Visualization

* **Heatmaps**: Added `cb_extend` parameter to allow colorbar extensions in heatmap plots (3).
* **Bugfix**: Fixed an issue with incorrect parameter naming for min/max ticks (1).

#### Quality Control and System

* **Thresholds**: Increased the percentage threshold required for a time resolution to be considered valid to 0.2% (2).
* **Environment**: Updated all packages to the newest possible versions.
* **Testing**: Currently, 71/71 unit tests are passing successfully.

### Experimental

* I am testing a method to detect potential time shifts in time series data using FFT. The `execute_phase_shift_fft`
  function detects time-series drift by using a targeted Discrete Fourier Transform to compare the phase angle of the
  24-hour diurnal cycle in measured radiation against theoretical potential radiation. It quantifies these timing errors
  and generates a visualization to analyze the distribution and seasonal patterns of the detected shifts.Looks promising
  so far, but not fully tested.(39)

### References

* (1) `diive.pkgs.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb.showplot_resampled`
* (2) `diive.pkgs.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb._validate_n_grouprecords`
* (3) `diive.core.plotting.heatmap_base.py`
* (4) `diive.pkgs.flux.hires.lag.MaxCovariance._find_max_cov_peak`
* (5) `diive.pkgs.createvar.conversions.air_temp_from_sonic_temp`
* (6) `tests.test_createvar.TestCreateVar.test_air_temp_from_sonic_temp`
* (7) `notebooks/CalculateVariable/Calculate_air_temp_from_sonic_temp.ipynb`
* (8) `notebooks/CalculateVariable/FluxDetectionLimit/FluxDetectionLimit.ipynb`
* (9) `diive.pkgs.flux.hires.fluxdetectionlimit.FluxDetectionLimit`
* (10) `tests.test_echires.TestEcHires`
* (11) `diive.core.times.times.insert_season`
* (12) `diive.pkgs.preprocessing.outlier_detection.common.create_daytime_nighttime_flags`
* (13) `diive.pkgs.preprocessing.outlier_detection.localsd.LocalSD`
* (14) `tests.test_outlierdetection.TestOutlierDetection.test_localsd_daytime_nighttime`
* (15) `notebooks/FluxProcessingChain/FluxProcessingChain.ipynb`
* (16) `notebooks/MeteoScreening/StepwiseMeteoScreeningFromDatabase.ipynb`
* (17) `notebooks/TimeFunctions/VectorizeTimestamps.ipynb`
* (18) `diive.core.times.times.vectorize_timestamps`
* (19) `diive.pkgs.createvar.potentialradiation.potrad_eot`
* (20) `diive.pkgs.createvar.air.dry_air_density`
* (21) `diive.pkgs.createvar.air.aerodynamic_resistance`
* (22) `diive.pkgs.fluxprocessingchain.level31_storagecorrection.FluxStorageCorrectionSinglePointEddyPro`
* (23) `diive.pkgs.preprocessing.outlier_detection.hampel.HampelDaytimeNighttime`
* (24) `diive.pkgs.preprocessing.outlier_detection.hampel.Hampel`
* (25) `tests.test_outlierdetection.TestOutlierDetection.test_hampel_filter_daytime_nighttime_doublediff`
* (26) `tests.test_outlierdetection.TestOutlierDetection.test_hampel_filter_basic`
* (27) `tests.test_outlierdetection.TestOutlierDetection.test_hampel_filter`
* (28)
  `diive.pkgs.preprocessing.outlier_detection.stepwiseoutlierdetection.StepwiseOutlierDetection.flag_outliers_hampel_dtnt_test`
* (29) `diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain.level32_flag_outliers_hampel_dtnt_test`
* (30) `tests.test_time.TestTime.test_vectorize_with_default_parameters`
* (31) `notebooks/GapFilling/LongTermRandomForestGapFilling.ipynb`
* (32) `notebooks/GapFilling/QuickRandomForestGapFilling.ipynb`
* (33) `notebooks/GapFilling/RandomForestGapFilling.ipynb`
* (34) `notebooks/GapFilling/RandomForestParamOptimization.ipynb`
* (35) `notebooks/GapFilling/XGBoostGapFillingExtensive.ipynb`
* (36) `notebooks/GapFilling/XGBoostGapFillingMinimal.ipynb`
* (37) `notebooks/OutlierDetection/HampelDaytimeNighttime.ipynb`
* (38) `diive.pkgs.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb.flag_outliers_abslim_dtnt_test`
* (39) `diive.pkgs.preprocessing.qaqc.experimental_indev.detect_timestamp_shifts.execute_phase_shift_fft`
* (40) `diive/pkgs/flux/selfheating.py`
* (41)
  `notebooks/FluxProcessingChain/self-heating_correction/SelfHeatingCorrectionNEE_1_CreateScalingFactorsTable.ipynb`
* (42) `notebooks/FluxProcessingChain/self-heating_correction/SelfHeatingCorrectionNEE_2_ApplyScalingFactorsTable.ipynb`

## v0.89.0 | 23 Jul 2025

Version 0.89.0 introduces a new `GridAggregator` class for 2D data aggregation with support for quantile,
equal-width, and custom binning methods, along with comprehensive documentation improvements and major dependency
updates including shapiq integration for enhanced analysis capabilities.

See the [notebook](https://github.com/holukas/diive/blob/main/notebooks/Analyses/GridAggregator.ipynb) for example
usage.

### Added

- New `GridAggregator` class for 2D grid data aggregation (`diive/pkgs/analyses/gridaggregator.py`)
    - Supports quantile, equal-width, and custom binning methods
    - Flexible aggregation functions
    - Comprehensive input validation and error handling
    - Added unit tests covering core functionality
    - Added example notebook:
        - `notebooks/Examples/GridAggregator.ipynb` - demonstrates 2D data aggregation and binning

### Enhanced

- Improved documentation across modules
    - Added detailed docstrings for methods and classes
    - Updated example notebooks for better clarity
    - Streamlined notebook structure in Overview

### Dependencies

- Updated multiple Python dependencies to their latest versions
- Added new dependencies:
    - shapiq (>=1.3.1,<2.0.0)
    - galois
    - networkx
    - sparse-transform

### Unittests

- Added unittests for `dv.heatmap_xyz`
- 66/66 unittests ran successfully

## v0.88.0 | 18 Jul 2025

![plotHeatmapYearMonthMaxTA_diive_v0.88.0](images/plotHeatmapYearMonthMaxTA_diive_v0.88.0.png)
*Heatmaps can now be plotted in horizontal orientation by setting the parameter `ax_orientation='horizontal'`. This
example plot shows the monthly maximum air temperature.*

### Changes

#### Heatmap updates

- There are several improvements for heatmap visualizations:
    - More consistent heatmap creation: The `.heatmapdatetime()`, `.heatmapyearmonth()` and `.heatmapxyz()` functions
      now offer a more unified experience for generating heatmaps.
    - Flexible orientation: heatmaps can now be displayed vertically or horizontally using the new parameter
      `ax_orientation`.
    - The rank plot introduced in the previous version can now be created using the parameter `ranks=True` when using
      `.heatmapyearmonth()`.

Fyi, `.heatmapdatetime()` is an alias for the `diive.core.plotting.heatmap_datetime.HeatmapDateTime` class,
`.heatmapyearmonth()` is an alias for `diive.core.plotting.heatmap_datetime.HeatmapYearMonth`, `.heatmapxyz()` is an
alias for `diive.core.plotting.heatmap_xyz.HeatmapXYZ`. All of these classes use
`diive.core.plotting.heatmap_base.HeatmapBase` or `diive.core.plotting.heatmap_base.HeatmapBaseXYZ` as base class for
their core functionality.

### Notebooks

- Updated notebook for `QuantileGridAggregator` (formerly `CalculateZaggregatesInQuantileClassesOfXY`)
- Updated notebook for `HeatmapDateTime`
- Updated notebook for `HeatmapYearMonth`

### Unittests

- Updated test case for `tests.test_analyses.TestAnalyses.test_quantilegridaggregator`
- 56/56 unittests ran successfully

## v0.87.1 | 12 Jun 2025

### New features

- Added new function `.set_exact_values_to_missing()` to set specific values in a time series to missing values (
  `diive.pkgs.preprocessing.corrections.setto_missing.set_exact_values_to_missing`)

### Additions

- Added parameters when plotting diel cycles:
    - Added parameter `show_xticklabels` for showing grid
    - Added parameter `show_xlabel` for showing x-ticklabels
    - Added parameter `show_legend` for showing legend
    - (`diive.core.plotting.dielcycle.DielCycle.plot`)
- Similarly, added more params for plotting cumulatives (`diive.core.plotting.cumulative.Cumulative`)

### Changes

- In `.quickplot()`, other rows now use the same scaling for x-axis as the plot in the first row (
  `diive.core.plotting.plotfuncs.quickplot`)
- Scaling of the y-axis is now slightly extended (by 5%) when plotting cumulatives (
  `diive.core.plotting.cumulative.Cumulative`)

### Notebooks

- Updated `StepwiseMeteoScreeningFromDatabase.ipynb`, added new correction `.set_exact_values_to_missing()`

### Unittests

- Added test case for `.set_exact_values_to_missing()` (`tests.test_corrections.TestCorrections.test_settomissing`)
- 56/56 unittests ran successfully

## v0.87.0 | 17 May 2025

### Heatmap rank plot

`diive` can now create heatmap rank plots.

![plotHeatmapYearMonthRank_diive_v0.87.0.png](images/plotHeatmapYearMonthRank_diive_v0.87.0.png)

*Example heatmap rank plot for air temperatures. This heatmap displays the rank of average monthly air temperatures
compared across different years. For instance, May 2022 had the highest average temperature among all Mays on record (
rank 1), as did October 2022 for Octobers. Conversely, January 2019 recorded the lowest average temperature for January
within the 26-year period shown.*

Heatmap rank plots display the relative ranking of monthly aggregated values across multiple years. Essentially, it
shows how each month's overall value compares to the same month in other years. By default, the plot ranks the monthly
mean (average) of the selected variable.

Other aggregation methods commonly used in the `pandas` library are possible, such as `median`, `min`, `max` and `std`,
among others.

Basic example:

```
import diive as dv
hm = dv.heatmapyearmonth(ranks=True, ...)  # Use parameter  
hm.plot()  # Generate basic plot
```

See the notebook here for more examples:
`notebooks/Plotting/HeatmapYearMonth.ipynb`

### New features

- Now deprecated: ~~Added new class `.heatmapyearmonth_ranks()` to plot monthly ranks of an aggregated value across
  years (`diive.core.plotting.heatmap_datetime.HeatmapYearMonthRanks`)~~ Use `.heatmapyearmonth(ranks=True, ...)`
  instead
- Added new function `.resample_to_monthly_agg_matrix()` to calculate a matrix of monthly aggregates across years (
  `diive.core.times.resampling.resample_to_monthly_agg_matrix`)
- Added new function `.transform_yearmonth_matrix_to_longform()` to convert monthly aggregation matrix to long-form time
  series (`diive.core.dfun.frames.transform_yearmonth_matrix_to_longform`)
- Added new function to calculate ET (evapotranspiration in mm h-1) from LE (latent heat flux in W m-2). (
  `diive.pkgs.createvar.conversions.et_from_le`)
- Added new function to calculate latent heat of vaporization. Originally needed for calculating ET from LE. (
  `diive.pkgs.createvar.conversions.latent_heat_of_vaporization`)

### Additions

- Heatmap plotting:
    - Heatmaps can now show the z-value for each rectangle in the plot, using the parameters `show_values` and
      `show_values_n_dec_places`. This makes more sense for data that are plotted month vs. year than for e.g.
      half-hourly data.
    - Simplified API to call heatmap plots: after `import diive as dv`, the heatmaps can now be called via
      `dv.heatmapyearmonth()` and `dv.heatmapdatetime()`.
- `SortingBinsMethod`:
    - The counts per bin are now also part of the bin stats
    - Sometimes the required number of bins cannot be generated, in this case the stats for the respective bin are now
      skipped and the bin is missing from the output (`.calcbins`)
    - All parameters were renamed to better reflect what is going on
    - (`diive.pkgs.analysis.decoupling.SortingBinsMethod`)
    - Added `agg` parameter to define aggregation method used in binning the data
    - Renamed and reworked `conversion` paramater, now allows conversion to z-scores in addition to percentiles
- Added new filetype `FLUXNET-FULLSET-HR-CSV-60MIN` for reading FLUXNET files with 60MIN time resolution

### Notebooks

- Added new notebook for calculating a monthly aggregation matrix (`notebooks/Resampling/ResamplingMonthlyMatrix.ipynb`)
- Updated notebook `HeatmapDateTime`
- Updated notebook `HeatmapYearMonth`
- Changed name of notebook `ridgeline` to camel-case `RidgeLine`

### Unittests

- Added test case for `.et_from_le()` (`tests.test_createvar.TestCreateVar.test_conversion_et_from_le`)
- Added test case for `.resample_to_monthly_agg_matrix()`, this test also includes the transformation to long-form time
  series using `.transform_yearmonth_matrix_to_longform()` (
  `tests.test_resampling.TestResampling.test_resample_to_monthly_agg_matrix`)
- 55/55 unittests ran successfully

### Environment

- `diive` is now using Python version `3.11` upwards
- Updated environment, poetry `pyproject.toml` file now has the currently used structure

## v0.86.0 | 20 Mar 2025

### New features

### Ridgeline plot

`diive` can now create ridgeline plots.

![plotRidgeLinePlot_diive_v0.86.0.png](images/plotRidgeLinePlot_diive_v0.86.0.png)

The ridgeline plot visualizes the distribution of a quantitative variable by stacking overlapping density plots,
creating a "ridged" landscape. I think this is quite pleasing to look at. With the implementation in `diive`, it
facilitates the comparison of distributional shapes and changes of time series data across weeks, months and years.
Ridgeline plots are quite space-efficient and hopefully visually intuitive for revealing patterns and trends in data.

This is also the first function that uses a simplified API. After importing `diive`, the plot can simply be accessed via
`.ridgeline()`. This is a shortcut to access the class `RidgeLinePlot` that is otherwise deeply buried in the code
here: `diive.core.plotting.ridgeline.RidgeLinePlot`. In the future, other classes and functions will also be
accessible via similar shortforms.

Basic example:

```
import diive as dv
rp = dv.ridgeline(series=series)  # Initialize instance, series is a pandas Series
rp.plot()  # Generate basic plot
```

See the notebook here for more examples:
`notebooks/Plotting/RidgeLine.ipynb`

### Additions

- Additions to the flux processing chain:
    - Added two methods to get details about training and testing when using machine-learning models in the flux
      processing chain: `.report_traintest_model_scores()` and `.report_traintest_details()`
    - Added parameter `setflag_timeperiod` to set the flag for the SSITC to another value during certain time periods,
      for example when a time period needs stricter filtering (e.g. due to issues with the sonic anemometer). In this
      case the parameter can be used to set all values where flag=1 (medium quality data) to flag=2 (bad data).
        - Example from docstring:```
      Set flag 1 to value 2 between '2022-05-01' and '2023-09-30', and between 
      '2024-04-02' and '2024-04-19' (dates inclusive): 
      setflag_timeperiod={2: [ [1, '2022-05-01', '2023-09-30'], [1, '2024-04-02', '2024-04-19'] ]}
      ``` (`diive.pkgs.preprocessing.qaqc.eddyproflags.flag_ssitc_eddypro_test`)
    - Added params to export some gap-filling results (e.g. model scores) to csv files (e.g.,
      `.report_gapfilling_model_scores(outpath=...)`)
    - (`diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain`)
- Added check if time series has a name when plotting heatmaps. If time series does not have a name, it is automatically
  assigned the name `data`. Implemented in class `HeatmapBase` that is used by all heatmap plotters. (
  `diive.core.plotting.heatmap_base.HeatmapBase`)
- Added new filetype for 60MIN EddyPro output (`diive/configs/filetypes/EDDYPRO-FLUXNET-CSV-60MIN.yml`)

### Notebooks

- Added notebook for ridgeline plot (`notebooks/Plotting/ridgeline.ipynb`)

### Bugfixes

- Fixed bug where the flux processing chain would crash when a variable with the same name as one of the automatically
  generated variables was already present in the input data. For example, the potential radiation `SW_IN_POT` is
  generated when the flux processing chain starts and then it is added also to the input data. If the input data already
  has a variable with the same name, the processing chain would crash. Now, the automatically generated `SW_IN_POT` is
  given priority, which means the variable in the input data is overwritten. (
  `diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain`)

### Environment

- Updated packages

### Unittests

- 53/53 unittests ran successfully

## v0.85.7 | 26 Feb 2025

### New features

- Added class for formatting meteo data for upload to FLUXNET (`diive.pkgs.formats.meteo.FormatMeteoForFluxnetUpload`)

### Notebooks

- Added new notebook `notebooks/Formats/FormatMeteoForFluxnetUpload.ipynb`

## v0.85.6 | 25 Feb 2025

### New features

- Added class to format meteo data as input file for EddyPro flux calcs (
  `diive.pkgs.formats.meteo.FormatMeteoForEddyProFluxProcessing`)

### Changes

- Updated formatting for FLUXNET upload (`diive.pkgs.formats.fluxnet.FormatEddyProFluxnetFileForUpload`)
- `HeatmapYearMonth` plot now shows every year on y-axis (`diive.core.plotting.heatmap_datetime.HeatmapYearMonth`)
- Improved check for excluded columns when creating lagged variants (
  `diive.pkgs.createvar.laggedvariants.lagged_variants`)
- More text output when reducting features (`diive.core.ml.common.MlRegressorGapFillingBase.reduce_features`)
- Fixed colorwheel running out of colors when plotting feature ranks (
  `diive.pkgs.gapfilling.longterm.LongTermGapFillingBase.showplot_feature_ranks_per_year`)
- Less text output when filling storage term (
  `diive.pkgs.fluxprocessingchain.level31_storagecorrection.FluxStorageCorrectionSinglePointEddyPro._gapfill_storage_term`)
- Smaller fixes

### Notebooks

- Added new notebook `notebooks/Formats/FormatMeteoForEddyProFluxProcessing.ipynb`
- Updated notebook `notebooks/Formats/notebooks/Formats/FormatEddyProFluxnetFileForUpload.ipynb`

## v0.85.5 | 3 Feb 2025

### Updates to MDS gap-filling

The community-standard MDS gap-filling method for eddy covariance ecosystem fluxes (e.g., CO2 flux) is now integrated
into the `FluxProcessingChain`. MDS is used during gap-filling in flux Level-4.1.

- **Example notebook** using MDS as part of the flux processing chain where it is used together with random
  forest: [Flux Processing Chain](/notebooks/FluxProcessingChain/FluxProcessingChain.ipynb)
- **Example notebook** using MDS as stand alone class
  `FluxMDS`: [MDS gap-filling of ecosystem fluxes](/notebooks/GapFilling/FluxMDSGapFilling.ipynb)

The `diive` implementation of the MDS gap-filling method adheres to the descriptions in Reichstein et al. (2005) and
Vekuri et al. (2023), similar to the standard gap-filling procedures used by FLUXNET, ICOS, ReddyProc, and other similar
platforms. This method fills gaps by substituting missing flux values with average flux values observed under comparable
meteorological conditions.

![DIIVE](images/plotMDS_diive_v0.85.5.png)

#### Background: different flux levels

- The class `FluxProcessingChain` in `diive` follows the flux processing steps as shown in
  the [Flux Processing Chain](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/)
  outlined by [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/).
-
- The flux processing chain uses different levels for different steps in the chain:
    - Level-0: preliminary flux calculations, e.g. during the year,
      using [EddyPro](https://www.licor.com/products/eddy-covariance/eddypro)
    - Level-1: final flux calculations, e.g. for complete year,
      using [EddyPro](https://www.licor.com/products/eddy-covariance/eddypro)
    - Level-2: quality flag expansion (flagging)
    - Level-3.1: storage correction (using one point measurement only, from profile not included by default)
    - Level-3.2: outlier removal (flagging)
    - Level-3.3: USTAR filtering (constant threshold, must be known, detection process not included by default)  (
      flagging)
    - Following Level 3.3, a comprehensive quality flag (`QCF`) is generated by combining individual quality flags.
      Prior to subsequent processing steps, low-quality data (flag=2) is removed. Medium-quality data (flag=1) can be
      retained if necessary, while the highest quality data (flag=0) is always kept.
    - Level-4.1: gap-filling (MDS, long-term random forest)

### Changes

- Changes in `FluxMDS`:
    - Added parameter `avg_min_n_vals` in MDS gap-filling
    - Renamed tolerance parameters for MDS gap-filling to `*_tol`
    - (`diive.pkgs.gapfilling.mds.FluxMDS`)
- When reading a parquet file, sanitizing the timestamp is now optional (`diive.core.io.files.load_parquet`)
- The function for creating lagged variants is now found in `diive.pkgs.createvar.laggedvariants.lagged_variants`

### Additions

- Added more text output for fill quality during gap-filling with MDS (`diive.pkgs.gapfilling.mds.FluxMDS`)
- Added MDS gap-filling to flux processing chain (
  `diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain`)
- Allow fitting to unbinned data (`diive.pkgs.fits.fitter.BinFitterCP`)
- Added parameter to edit y-label (`diive.core.plotting.dielcycle.DielCycle`)
- Added preliminary USTAR filtering for NEE to quick flux processing chain (
  `diive.pkgs.fluxprocessingchain.fluxprocessingchain.QuickFluxProcessingChain`)
- `FileSplitter`:
    - Added parameter to directly output splits as `parquet` files in `FileSplitter` and `FileSplitterMulti`. These two
      classes split longer time series files (e.g., 6 hours) into several smaller splits (e.g., 12 half-hourly files).
      Usage of parquet speeds up not only the splitting part, but also the process when later re-reading the files for
      other processing steps.
    - After splitting, missing values in the split files are numpy NAN (`diive.core.io.filesplitter.FileSplitter`)
- Added parameter to hide default plot when called. The method `defaultplot` is used e.g. by outlier detection methods
  to plot the data after outlier removal, to show flagged vs. unflagged values. (
  `diive.core.base.flagbase.FlagBase.defaultplot`)
- Added new filetype `ETH-SONICREAD-BICO-MOD-CSV-20HZ`
- Added `fig` property that contains the default plot for outlier removal methods. This is useful when the default plot
  is needed elsewhere, e.g. saved to a file. At the moment, the parameter `showplot` must be `True` for the property to
  be accessible. (`diive.core.base.flagbase.FlagBase`)
    - Example for class `zScoreRolling`:
      ```
      zsr = zScoreRolling(..., showplot=True, ...)
      zsr.calc(repeat=True)
      fig = zsr.fig  # Contains the figure instance
      fig.savefig(...)  # Figure can then be saved to a file etc.
      ```  

### Notebooks

- Added notebook example for creating lagged variants of variables (
  `notebooks/CalculateVariable/Create_lagged_variants.ipynb`)
- Updated flux processing chain notebook to `v9.0`: added option for MDS gap-filling, more descriptions
- Bugfix: import for loading from `Path` was missing in flux processing chain notebook
- Updated MDS gap-filling notebook to `v1.1`, added more descriptions and example for `min_n_vals_nt` parameter
- Updated quick flux processing chain notebook

### Unittests

- Added test case `tests.test_createvar.TestCreateVar.test_lagged_variants`
- Updated test case `tests.test_gapfilling.TestGapFilling.test_fluxmds`
- Updated test case `tests.test_fluxprocessingchain.TestFluxProcessingChain.test_fluxprocessingchain`
- 53/53 unittests ran successfully

### Bugfixes

- The setting for features that should not be lagged was not properly implemented (
  `diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain._get_ml_feature_settings`)
- Fixed bug when plotting (`diive.pkgs.preprocessing.outlier_detection.localsd.LocalSD`)

## v0.84.2 | 8 Nov 2024

### Changes

- Adjust version number to avoid publishing conflict

## v0.84.1 | 8 Nov 2024

### Bugfixes

- Removed invalid imports

### Tests

- Added test case for `diive` imports (`tests.test_imports.TestImports.test_imports`)
- 52/52 unittests ran successfully

## v0.84.0 | 7 Nov 2024

### New features

- New class `BinFitterCP` for fitting function to binned data, includes confidence interval and prediction interval (
  `diive.pkgs.fits.fitter.BinFitterCP`)

![DIIVE](images/BinFitterCP_diive_v0.84.0.png)

### Additions

- Added small function to detect duplicate entries in lists (`diive.core.funcs.funcs.find_duplicates_in_list`)
- Added new filetype (`diive/configs/filetypes/ETH-MERCURY-CSV-20HZ.yml`)
- Added new filetype (`diive/configs/filetypes/GENERIC-CSV-HEADER-1ROW-TS-END-FULL-NS-20HZ.yml`)

### Bugfixes

- Not directly a bug fix, but when reading EddyPro fluxnet files with `LoadEddyProOutputFiles` (e.g., in the flux
  processing chain) duplicate columns are now automatically renamed by adding a numbered suffix. For example, if two
  variables are named `CUSTOM_CH4_MEAN` in the output file, they are automatically renamed to `CUSTOM_CH4_MEAN_1` and
  `CUSTOM_CH4_MEAN_2` (`diive.core.dfun.frames.compare_len_header_vs_data`)

### Notebooks

- Added notebook example for `BinFitterCP` (`notebooks/Fits/BinFitterCP.ipynb`)
- Updated flux processing chain notebook to `v8.6`, import for loading EddyPro fluxnet output files was missing

### Tests

- Added test case for `BinFitterCP` (`tests.test_fits.TestFits.test_binfittercp`)
- 51/51 unittests ran successfully

## v0.83.2 | 25 Oct 2024

From now on Python version `3.11.10` is used for developing Python (up to now, version `3.9` was used). All unittests
were successfully executed with this new Python version. In addition, all notebooks were re-run, all looked good.

[JupyterLab](https://jupyterlab.readthedocs.io/en/4.2.x/index.html) is now included in the environment, which makes it
easier to quickly install `diive` (`pip install diive`) in an environment and directly use its notebooks, without the
need to install JupyterLab separately.

### Environment

- `diive` will now be developed using Python version `3.11.10`
- Added [JupyterLab](https://jupyterlab.readthedocs.io/en/4.2.x/index.html)
- Added [jupyter bokeh](https://github.com/bokeh/jupyter_bokeh)

### Notebooks

- All notebooks were re-run and updated using Python version `3.11.10`

### Tests

- 50/50 unittests ran successfully with Python version `3.11.10`

### Changes

- Adjusted flags check in QCF flag report, the progressive flag must be the same as the previously calculated overall
  flag (`diive.pkgs.preprocessing.qaqc.qcf.FlagQCF.report_qcf_evolution`)

## v0.83.1 | 23 Oct 2024

### Changes

- When detecting the frequency from the time delta of records, the inferred frequency is accepted if the most frequent
  timedelta was found for more than 50% of records (`diive.core.times.times.timestamp_infer_freq_from_timedelta`)
- Storage terms are now gap-filled using the rolling median in an expanding time window (
  `FluxStorageCorrectionSinglePointEddyPro._gapfill_storage_term`)

### Notebooks

- Added notebook example for using the flux processing chain for CH4 flux from a subcanopy eddy covariance station (
  `notebooks/Workbench/CH-DAS_2023_FluxProcessingChain/FluxProcessingChain_NEE_CH-DAS_2023.ipynb`)

### Bugfixes

- Fixed info for storage term correction report to account for cases when more storage terms than flux records are
  available (`FluxStorageCorrectionSinglePointEddyPro.report`)

### Tests

- 50/50 unittests ran successfully

## v0.83.0 | 4 Oct 2024

### MDS gap-filling

Finally it is possible to use the `MDS` (`marginal distribution sampling`) gap-filling method in `diive`. This method is
the current default and widely used gap-filling method for eddy covariance ecosystem fluxes. For a detailed description
of the method see Reichstein et al. (2005) and Pastorello et al. (2020; full references given below).

The implementation of `MDS` in `diive` (`FluxMDS`) follows the description in Reichstein et al. (2005) and should
therefore yield results similar to other implementations of this algorithm. `FluxMDS` can also easily output model
scores, such as r2 and error values.

At the moment it is not yet possible to use `FluxMDS` in the flux processing chain, but during the preparation of this
update the flux processing chain code was already refactored and prepared to include `FluxMDS` in one of the next
updates.

At the moment, `FluxMDS` is specifically tailored to gap-fill ecosystem fluxes, a more general implementation (e.g., to
gap-fill meteorological data) will follow.

### New features

- Added new gap-filling class `FluxMDS`:
    - `MDS` stands for `marginal distribution sampling`. The method uses a time window to first identify meteorological
      conditions (short-wave incoming radiation, air temperature and VPD) similar to those when the missing data
      occurred. Gaps are then filled with the mean flux in the time window.
    - `FluxMDS` cannot be used in the flux processing chain, but will be implemented soon.
    - (`diive.pkgs.gapfilling.mds.FluxMDS`)

### Changes

- **Storage correction**: By default, values missing in the storage term are now filled with a rolling mean in an
  expanding time window. Testing showed that the (single point) storage term is missing for between 2-3% of the data,
  which I think is reason enough to make filling these gaps the default option. Previously, it was optional to fill the
  gaps using random forest, however, results were not great since only the timestamp info was used as model features.
  Plots generated during Level-3.1 were also updated, now better showing the storage terms (gap-filled and
  non-gap-filled) and the flag indicating filled values (
  `diive.pkgs.fluxprocessingchain.level31_storagecorrection.FluxStorageCorrectionSinglePointEddyPro`)

### Notebooks

- Added notebook example for `FluxMDS` (`notebooks/GapFilling/FluxMDSGapFilling.ipynb`)

### Tests

- Added test case for `FluxMDS` (`tests.test_gapfilling.TestGapFilling.test_fluxmds`)
- 50/50 unittests ran successfully

### Bugfixes

- Fixed bug: overall quality flag `QCF` was not created correctly for the different USTAR scenarios (
  `diive.core.base.identify.identify_flagcols`) (`diive.pkgs.preprocessing.qaqc.qcf.FlagQCF`)
- Fixed bug: calculation of `QCF` flag sums is now strictly done on flag columns. Before, sums were calculated across
  all columns in the flags dataframe, which resulted in erroneous overall flags after USTAR filtering (
  `diive.pkgs.preprocessing.qaqc.qcf.FlagQCF._calculate_flagsums`)

### Environment

- Added [polars](https://pola.rs/)

### References

- Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux processing pipeline
  for eddy covariance data. 27. https://doi.org/10.1038/s41597-020-0534-3
- Reichstein, M., Falge, E., Baldocchi, D., Papale, D., Aubinet, M., Berbigier, P., Bernhofer, C., Buchmann, N.,
  Gilmanov, T., Granier, A., Grunwald, T., Havrankova, K., Ilvesniemi, H., Janous, D., Knohl, A., Laurila, T., Lohila,
  A., Loustau, D., Matteucci, G., … Valentini, R. (2005). On the separation of net ecosystem exchange into assimilation
  and ecosystem respiration: Review and improved algorithm. Global Change Biology, 11(9),
  1424–1439. https://doi.org/10.1111/j.1365-2486.2005.001002.x

## v0.82.1 | 22 Sep 2024

### Notebooks

- Added notebook showing an example for `LongTermGapFillingRandomForestTS` (
  `notebooks/GapFilling/LongTermRandomForestGapFilling.ipynb`)
- Added notebook example for `MeasurementOffset` (`notebooks/Corrections/MeasurementOffset.ipynb`)

### Tests

- Added unittest for `LongTermGapFillingRandomForestTS` (
  `tests.test_gapfilling.TestGapFilling.test_gapfilling_longterm_randomforest`)
- Added unittest for `WindDirOffset` (`tests.test_corrections.TestCorrections.test_winddiroffset`)
- Added unittest for `DaytimeNighttimeFlag` (`tests.test_createvar.TestCreateVar.test_daytime_nighttime_flag`)
- Added unittest for `calc_vpd_from_ta_rh` (`tests.test_createvar.TestCreateVar.test_calc_vpd`)
- Added unittest for `percentiles101` (`tests.test_analyses.TestAnalyses.test_percentiles`)
- Added unittest for `GapFinder` (`tests.test_analyses.TestAnalyses.test_gapfinder`)
- Added unittest for `SortingBinsMethod` (`tests.test_analyses.TestAnalyses.test_sorting_bins_method`)
- Added unittest for `daily_correlation` (`tests.test_analyses.TestAnalyses.test_daily_correlation`)
- Added unittest for `QuantileXYAggZ` (`tests.test_analyses.TestCreateVar.test_quantilexyaggz`)
- 49/49 unittests ran successfully

### Bugfixes

- Fixed bug that caused results from long-term gap-filling to be inconsistent *despite* using a fixed random state. I
  found the following: when reducing features across years, the removal of duplicate features from a list of found
  features created a list where the order of elements changed each run. This in turn produced slightly different
  gap-filling results each time the long-term gap-filling was executed. Used Python version where this issue occurred
  was `3.9.19`.
    - Here is a simplified example, where `input_list` is a list of elements with some duplicate elements:
    - Running `output_list = list(set(input_list))` generates `output_list` where the elements would have a different
      output order each run. The elements were otherwise the same, only their order changed.
    - To keep the order of elements consistent it was necessary to `output_list.sort()`.
    - (`diive.pkgs.gapfilling.longterm.LongTermGapFillingBase.reduce_features_across_years`)
- Corrected wind direction could be 360°, but will now be 0° (
  `diive.pkgs.preprocessing.corrections.winddiroffset.WindDirOffset._correct_degrees`)

## v0.82.0 | 19 Sep 2024

### Long-term gap-filling

It is now possible to gap-fill multi-year datasets using the class `LongTermGapFillingRandomForestTS`. In this approach,
data from neighboring years are pooled together before training the random forest model for gap-filling a specific year.
This is especially useful for long-term, multi-year datasets where environmental conditions and drivers might change
over years and decades.

Why random forest? Because it performed well and to me it looks like the first choice for gap-filling ecosystem fluxes,
at least at the moment.

Long-term gap-filling using random forest is now also built into the flux processing chain (Level-4.1). This allows to
quickly gap-fill the different USTAR scenarios and to create some useful plots (I
hope). [See the flux processing chain notebook for how this looks like](https://github.com/holukas/diive/blob/main/notebooks/FluxProcessingChain/FluxProcessingChain.ipynb).

In a future update it will be possible to either directly switch to `XGBoost` for gap-filling, or to use it (and other
machine-learning models) in combination with random forest in the flux processing chain.

### Example

Here is an example for a dataset containing CO2 flux (`NEE`) measurements from 2005 to 2023:

- for gap-filling the year 2005, the model is trained on data from 2005, 2006 and 2007 (*2005 has no previous year*)
- for gap-filling the year 2006, the model is trained on data from 2005, 2006 and 2007 (same model as for 2005)
- for gap-filling the year 2007, the model is trained on data from 2006, 2007 and 2008
- ...
- for gap-filling the year 2012, the model is trained on data from 2011, 2012 and 2013
- for gap-filling the year 2013, the model is trained on data from 2012, 2013 and 2014
- for gap-filling the year 2014, the model is trained on data from 2013, 2014 and 2015
- ...
- for gap-filling the year 2021, the model is trained on data from 2020, 2021 and 2022
- for gap-filling the year 2022, the model is trained on data from 2021, 2022 and 2023 (same model as for 2023)
- for gap-filling the year 2023, the model is trained on data from 2021, 2022 and 2023 (*2023 has no next year*)

### New features

- Added new method for long-term (multiple years) gap-filling using random forest to flux processing chain (
  `diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain.level41_gapfilling_longterm`)
- Added new class for long-term (multiple years) gap-filling using random forest (
  `diive.pkgs.gapfilling.longterm.LongTermGapFillingRandomForestTS`)
- Added class for plotting cumulative sums across all data, for multiple columns (
  `diive.core.plotting.cumulative.Cumulative`)
- Added class to detect a constant offset between two measurements (
  `diive.pkgs.preprocessing.corrections.measurementoffset.MeasurementOffset`)

### Changes

- Creating lagged variants creates gaps which then leads to incomplete features in machine learning models. Now, gaps
  are filled using simple forward and backward filling, limited to the number of values defined in *lag*. For example,
  if variable TA is lagged by -2 value this creates two missing values for this variant at the start of the time series,
  which then are then gap-filled using the simple backwards fill with `limit=2`. (
  `diive.core.dfun.frames.lagged_variants`)

### Notebooks

- Updated flux processing chain notebook to include long-term gap-filling using random forest (
  `notebooks/FluxProcessingChain/FluxProcessingChain.ipynb`)
- Added new notebook for plotting cumulative sums across all data, for multiple columns (
  `notebooks/Plotting/Cumulative.ipynb`)

### Tests

- Unittest for flux processing chain now includes many more methods (
  `tests.test_fluxprocessingchain.TestFluxProcessingChain.test_fluxprocessingchain`)
- 39/39 unittests ran successfully

### Bugfixes

- Fixed deprecation warning in (`diive.core.ml.common.prediction_scores_regr`)

## v0.81.0 | 11 Sep 2024

### Expanding Flux Processing Capabilities

This update brings advancements for post-processing eddy covariance data in the context of the `FluxProcessingChain`.
The goal is to offer a complete chain for post-processing ecosystem flux data, specifically designed to work seamlessly
with the standardized `_fluxnet` output file from the
widely-used [EddyPro](https://www.licor.com/env/products/eddy-covariance/eddypro) software.

Now, diive offers the option for USTAR filtering based on *known* constant thresholds across the entire dataset (similar
to the `CUT` scenarios in FLUXNET data). While seasonal (DJF, MAM, JJA, SON) thresholds are calculated internally,
applying them on a seasonal basis or using variable thresholds per year (like FLUXNET's `VUT` scenarios) isn't yet
implemented.

With this update, the `FluxProcessingChain` class can handle various data processing steps:

- Level-2: Quality flag expansion
- Level-3.1: Storage correction
- Level-3.2: Outlier removal
- Level-3.3: (new) USTAR filtering (with constant thresholds for now)
- (upcoming) Level-4.1: long-term gap-filling using random forest and XGBoost
- For info about the different flux levels
  see [Swiss FluxNet flux processing chain](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/)

### New features

- Added class to apply multiple known constant USTAR (friction velocity) thresholds, creating flags that indicate time
  periods characterized by low turbulence for multiple USTAR scenarios. The constant thresholds must be known
  beforehand, e.g., from an earlier USTAR detection run, or from results from FLUXNET (
  `diive.pkgs.flux.ustarthreshold.FlagMultipleConstantUstarThresholds`)
- Added class to apply one single known constant USTAR thresholds (
  `diive.pkgs.flux.ustarthreshold.FlagSingleConstantUstarThreshold`)
- Added `FlagMultipleConstantUstarThresholds` to the flux processing chain (
  `diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain.level33_constant_ustar`)
- Added USTAR detection algorithm based on Papale et al., 2006 (`diive.pkgs.flux.ustarthreshold.UstarDetectionMPT`)
- Added function to analyze high-quality ecosystem fluxes that helps in understanding the range of highest-quality data(
  `diive.pkgs.flux.hqflux.analyze_highest_quality_flux`)

### Additions

- `LocalSD` outlier detection can now use a constant SD:
    - Added parameter to use standard deviation across all data (constant) instead of the rolling SD to calculate the
      upper and lower limits that define outliers in the median rolling window (
      `diive.pkgs.preprocessing.outlier_detection.localsd.LocalSD`)
    - Added to step-wise outlier detection (
      `diive.pkgs.preprocessing.outlier_detection.stepwiseoutlierdetection.StepwiseOutlierDetection.flag_outliers_localsd_test`)
    - Added to meteoscreening from database (
      `diive.pkgs.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb.flag_outliers_localsd_test`)
    - Added to flux processing chain (
      `diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain.level32_flag_outliers_localsd_test`)

### Changes

- Replaced `.plot_date()` from the Matplotlib library with `.plot()` due to deprecation

### Notebooks

- Added notebook for plotting cumulative sums per year (`notebooks/Plotting/CumulativesPerYear.ipynb`)
- Added notebook for removing outliers based on the z-score in rolling time window (
  `notebooks/OutlierDetection/zScoreRolling.ipynb`)

### Bugfixes

- Fixed bug when saving a pandas Series to parquet (`diive.core.io.files.save_parquet`)
- Fixed bug when plotting `doy_mean_cumulative`: no longer crashes when years defined in parameter
  `excl_years_from_reference` are not in dataset (`diive.core.times.times.doy_mean_cumulative`)
- Fixed deprecation warning when plotting in `bokeh` (interactive plots)

### Tests

- Added unittest for `LocalSD` using constant SD (
  `tests.test_outlierdetection.TestOutlierDetection.test_localsd_with_constantsd`)
- Added unittest for rolling z-score outlier removal (
  `tests.test_outlierdetection.TestOutlierDetection.test_zscore_rolling`)
- Improved check if figure and axis were created in (`tests.test_plots.TestPlots.test_histogram`)
- 39/39 unittests ran successfully

### Environment

- Added new package `scikit-optimize`
- Added new package `category_encoders`

## v0.80.0 | 28 Aug 2024

### Additions

- Added outlier tests to step-wise meteoscreening from database: `Hampel`, `HampelDaytimeNighttime` and `TrimLow` (
  `diive.pkgs.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb`)
- Added parameter to control whether or not to output the middle timestamp when loading parquet files with
  `load_parquet()`. By default, `output_middle_timestamp=True`. (`diive.core.io.files.load_parquet`)

### Environment

- Re-created environment and created new `lock` file
- Currently using Python 3.9.19

### Notebooks

- Added new notebook for creating a flag that indicates missing values (
  `notebooks/OutlierDetection/MissingValues.ipynb`)
- Updated notebook for meteoscreening from database (
  `notebooks/MeteoScreening/StepwiseMeteoScreeningFromDatabase.ipynb`)
- Updated notebook for loading and saving parquet files (`notebooks/Formats/LoadSaveParquetFile.ipynb`)

### Tests

- Added unittest for flagging missing values (`tests.test_outlierdetection.TestOutlierDetection.test_missing_values`)
- 37/37 unittests ran successfully

### Bugfixes

- Fixed links in README, needed absolute links to notebooks
- Fixed issue with return list in (`diive.pkgs.analysis.histogram.Histogram.peakbins`)

## v0.79.1 | 26 Aug 2024

### Additions

- Added new function to apply quality flags to certain time periods only (
  `diive.pkgs.preprocessing.qaqc.flags.restrict_application`)
- Added to option to restrict the application of the angle-of-attack flag to certain time periods (
  `diive.pkgs.fluxprocessingchain.level2_qualityflags.FluxQualityFlagsEddyPro.angle_of_attack_test`)

### Changes

- Test options in `FluxProcessingChain` are now always passed as dict. This has the advantage that in addition to run
  the test by setting the dict key `apply` to `True`, various other test settings can be passed, for example the new
  parameter `application dates` for the angle-of-attack flag. (
  `diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain`)

### Tests

- Added unittest for Flux Processing Chain up to Level-2 (
  `tests.test_fluxprocessingchain.TestFluxProcessingChain.test_fluxprocessingchain_level2`)
- 36/36 unittests ran successfully

## v0.79.0 | 22 Aug 2024

This version introduces a histogram plot that has the option to display z-score as vertical lines superimposed on the
distribution, which helps in assessing z-score settings used by some outlier removal functions.

![DIIVE](images/plotHistogram_diive_v0.79.0.png)

*Histogram plot of half-hourly air temperature measurements at the ICOS Class 1 ecosystem
station [Davos](https://www.swissfluxnet.ethz.ch/index.php/sites/site-info-ch-dav/) between 2013 and 2022, displayed in
20 equally-spaced bins. The dashed vertical lines show the z-score and the corresponding value calculated based on the
time series. The bin with most counts is highlighted orange.*

### New features

- Added new class `HistogramPlot`for plotting histograms, based on the Matplotlib
  implementation (`diive.core.plotting.histogram.HistogramPlot`)
- Added function to calculate the value for a specific z-score, e.g., based on a time series it calculates the value
  where z-score = `3` etc. (`diive.core.funcs.funcs.val_from_zscore`)

### Additions

- Added histogram plots to `FlagBase`, histograms are now shown for all outlier methods (
  `diive.core.base.flagbase.FlagBase.defaultplot`)
- Added daytime/nighttime histogram plots to (
  `diive.pkgs.preprocessing.outlier_detection.hampel.HampelDaytimeNighttime`)
- Added daytime/nighttime histogram plots to (
  `diive.pkgs.preprocessing.outlier_detection.zscore.zScoreDaytimeNighttime`)
- Added daytime/nighttime histogram plots to (
  `diive.pkgs.preprocessing.outlier_detection.lof.LocalOutlierFactorDaytimeNighttime`)
- Added daytime/nighttime histogram plots to (
  `diive.pkgs.preprocessing.outlier_detection.absolutelimits.AbsoluteLimitsDaytimeNighttime`)
- Added option to calculate the z-score with sign instead of absolute (`diive.core.funcs.funcs.zscore`)

### Changes

- Improved daytime/nighttime outlier plot used by various outlier removal classes (
  `diive.core.base.flagbase.FlagBase.plot_outlier_daytime_nighttime`)

### Notebooks

- Added notebook for plotting histograms (`notebooks/Plotting/Histogram.ipynb`)
- Added notebook for manual removal of data points (`notebooks/OutlierDetection/ManualRemoval.ipynb`)
- Added notebook for outlier detection using local outlier factor, separately during daytime and nighttime (
  `notebooks/OutlierDetection/LocalOutlierFactorDaytimeNighttime.ipynb`)
- Updated notebook (`notebooks/OutlierDetection/HampelDaytimeNighttime.ipynb`)
- Updated notebook (`notebooks/OutlierDetection/AbsoluteLimitsDaytimeNighttime.ipynb`)
- Updated notebook (`notebooks/OutlierDetection/zScoreDaytimeNighttime.ipynb`)
- Updated notebook (`notebooks/OutlierDetection/LocalOutlierFactorAllData.ipynb`)

### Tests

- Added unittest for plotting histograms (`tests.test_plots.TestPlots.test_histogram`)
- Added unittest for calculating histograms (without plotting) (`tests.test_analyses.TestCreateVar.test_histogram`)

## v0.78.1.1 | 19 Aug 2024

### Additions

- Added CITATIONS file

## v0.78.1 | 19 Aug 2024

### Changes

- Added option to set different `n_sigma` for daytime and nightime data
  in `HampelDaytimeNighttime` (`diive.pkgs.preprocessing.outlier_detection.hampel.HampelDaytimeNighttime`)
- Updated `flag_outliers_hampel_dtnt_test` in step-wise outlier detection
- Updated `level32_flag_outliers_hampel_dtnt_test` in flux processing chain

### Notebooks

- Updated notebook `HampelDaytimeNighttime`
- Updated notebook `FluxProcessingChain`

### Tests

- Updated unittest `test_hampel_filter_daytime_nighttime`
- 35/35 unittests ran successfully

## v0.78.0 | 18 Aug 2024

### New features

- Added new class for outlier removal, based on the rolling z-score. It can also be used in step-wise outlier detection
  and during meteoscreening from the
  database. (`diive.pkgs.preprocessing.outlier_detection.zscore.zScoreRolling`,
  `diive.pkgs.preprocessing.outlier_detection.stepwiseoutlierdetection.StepwiseOutlierDetection`,
  `diive.pkgs.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb`).
- Added Hampel filter for outlier removal (`diive.pkgs.preprocessing.outlier_detection.hampel.Hampel`)
- Added Hampel filter (separate daytime, nighttime) for outlier
  removal (`diive.pkgs.preprocessing.outlier_detection.hampel.HampelDaytimeNighttime`)
- Added function to plot daytime and nighttime outliers during outlier
  tests (`diive.core.plotting.outlier_dtnt.outlier_daytime_nighttime`)

### Changes

- Flux processing chain:
    - Several changes to the flux processing chain to make sure it can also work with data files not directly output by
      EddyPro. The class `FluxProcessingChain` can now handle files that have a different format than the two EddyPro
      output files `EDDYPRO-FLUXNET-CSV-30MIN` and `EDDYPRO-FULL-OUTPUT-CSV-30MIN`. See following notes.
    - Removed option to process EddyPro `_full_output_` files, since it as an older format and its variables do not
      follow FLUXNET conventions.
    - Removed keyword `filetype` in class `FluxProcessingChain`. It is now assumed that the variable names follow the
      FLUXNET convention. Variables used in FLUXNET are
      listed [here](https://fluxnet.org/data/fluxnet2015-dataset/fullset-data-product/) (
      `diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain`)
    - When detecting the base variable from which a flux variable was calculated, the variables defined for
      filetype `EDDYPRO-FLUXNET-CSV-30MIN` are now assumed by default. (`diive.pkgs.flux.common.detect_basevar`)
    - Renamed function that detects the base variable that was used to calculate the respective
      flux  (`diive.pkgs.flux.common.detect_fluxbasevar`)
    - Renamed `gas` in functions related to completeness tests to `fluxbasevar` to better reflect that the completeness
      test does not necessarily require a gas (e.g. `T_SONIC` is used to calculate the completeness for sensible heat
      flux) (`flag_fluxbasevar_completeness_eddypro_test`)
- Removing the radiation offset now uses `0.001` (W m-2) instead of `50` as the threshold value to flag nighttime values
  for the correction (`diive.pkgs.preprocessing.corrections.offsetcorrection.remove_radiation_zero_offset`)
- The database tag for meteo data screened with `diive` is
  now `meteoscreening_diive` (`diive.pkgs.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb.resample`)
- During noise generation, function now uses the absolute values of the min/max of a series to calculate minimum noise
  and maximum noise (`diive.pkgs.createvar.noise.add_impulse_noise`)

### Notebooks

- Added new notebook for outlier detection using class `zScore` (`notebooks/OutlierDetection/zScore.ipynb`)
- Added new notebook for outlier detection using
  class `zScoreDaytimeNighttime` (`notebooks/OutlierDetection/zScoreDaytimeNighttime.ipynb`)
- Added new notebook for outlier removal using trimming (`notebooks/OutlierDetection/TrimLow.ipynb`)
- Updated notebook (`notebooks/MeteoScreening/StepwiseMeteoScreeningFromDatabase_v7.0.ipynb`)
- When uploading screened meteo data to the database using the notebook `StepwiseMeteoScreeningFromDatabase`, variables
  with the same name, measurement and data version as the screened variable(s) are now deleted from the database before
  the new data are uploaded. Implemented in the Python package `dbc-influxdb` to avoid duplicates in the database. Such
  duplicates can occur when one of the tags of an otherwise identical variable changed, e.g., when one of the tags of
  the originally uploaded data was wrong and needed correction. The database `InfluxDB` stores a new time series
  alongside the previous time series when one of the tags is different in an otherwise identical time series.

### Tests

- Added test case for `Hampel` filter (`tests.test_outlierdetection.TestOutlierDetection.test_hampel_filter`)
- Added test case for `HampelDaytimeNighttime`
  filter (`tests.test_outlierdetection.TestOutlierDetection.test_hampel_filter_daytime_nighttime`)
- Added test case for `zScore` (`tests.test_outlierdetection.TestOutlierDetection.test_zscore`)
- Added test case for `TrimLow` (`tests.test_outlierdetection.TestOutlierDetection.test_trim_low_nt`)
- Added test case
  for `zScoreDaytimeNighttime` (`tests.test_outlierdetection.TestOutlierDetection.test_zscore_daytime_nighttime`)
- 33/33 unittests ran successfully

### Environment

- Added package [sktime](https://www.sktime.net/en/stable/index.html), a unified framework for machine learning with
  time series.

## v0.77.0 | 11 Jun 2024

### Additions

- Plotting cumulatives with `CumulativeYear` now also shows the cumulative for the reference, i.e. for the mean over the
  reference years (`diive.core.plotting.cumulative.CumulativeYear`)
- Plotting `DielCycle` now accepts `ylim` parameter (`diive.core.plotting.dielcycle.DielCycle`)
- Added long-term dataset for local testing purposes (internal
  only) (`diive.configs.exampledata.load_exampledata_parquet_long`)
- Added several classes in preparation for long-term gap-filling for a future update

### Changes

- Several updates and changes to the base class for regressor decision
  trees (`diive.core.ml.common.MlRegressorGapFillingBase`):
    - The data are now split into training set and test set at the very start of regressor setup. This test set is used
      to evaluate models on unseen data. The default split is 80% training and 20% test data.
    - Plotting (scores, importances etc.) is now generally separated from the method where they are calculated.
    - the same `random_state` is now used for all processing steps
    - refactored code
    - beautified console output
- When correcting for relative humidity values above 100%, the maximum of the corrected time series is now set to 100,
  after the (daily) offset was removed (
  `diive.pkgs.preprocessing.corrections.offsetcorrection.remove_relativehumidity_offset`)
- During feature reduction in machine learning regressors, features with permutation importance < 0 are now always
  removed (`diive.core.ml.common.MlRegressorGapFillingBase._remove_rejected_features`)
- Changed default parameters for quick random forest gap-filling (`diive.pkgs.gapfilling.randomforest_ts.QuickFillRFTS`)
- I tried to improve the console output (clarity) for several functions and methods

### Environment

- Added package [dtreeviz](https://github.com/parrt/dtreeviz?tab=readme-ov-file) to visualize decision trees

### Notebooks

- Updated notebook (`notebooks/GapFilling/RandomForestGapFilling.ipynb`)
- Updated notebook (`notebooks/GapFilling/LinearInterpolation.ipynb`)
- Updated notebook (`notebooks/GapFilling/XGBoostGapFillingExtensive.ipynb`)
- Updated notebook (`notebooks/GapFilling/XGBoostGapFillingMinimal.ipynb`)
- Updated notebook (`notebooks/GapFilling/RandomForestParamOptimization.ipynb`)
- Updated notebook (`notebooks/GapFilling/QuickRandomForestGapFilling.ipynb`)

### Tests

- Updated and fixed test case (`tests.test_outlierdetection.TestOutlierDetection.test_zscore_increments`)
- Updated and fixed test case (`tests.test_gapfilling.TestGapFilling.test_gapfilling_randomforest`)

## v0.76.2 | 23 May 2024

### Additions

- Added function to calculate absolute double differences of a time series, which is the sum of absolute differences
  between a data record and its preceding and next record. Used in class `zScoreIncrements` for finding (isolated)
  outliers that are distant from neighboring records. (`diive.core.dfun.stats.double_diff_absolute`)
- Added small function to calculate z-score stats of a time series (`diive.core.dfun.stats.sstats_zscore`)
- Added small function to calculate stats for absolute double differences of a time
  series (`diive.core.dfun.stats.sstats_doublediff_abs`)

### Changes

- Changed the algorithm for outlier detection when using `zScoreIncrements`. Data points are now flagged as outliers if
  the z-scores of three absolute differences (previous record, next record and the sum of both) all exceed a specified
  threshold.  (`diive.pkgs.preprocessing.outlier_detection.incremental.zScoreIncrements`)

### Notebooks

- Added new notebook for outlier detection using
  class `LocalOutlierFactorAllData` (`notebooks/OutlierDetection/LocalOutlierFactorAllData.ipynb`)

### Tests

- Added new test case
  for `LocalOutlierFactorAllData` (`tests.test_outlierdetection.TestOutlierDetection.test_lof_alldata`)

## v0.76.1 | 17 May 2024

### Additions

- It is now possible to set a fixed random seed when creating impulse
  noise (`diive.pkgs.createvar.noise.add_impulse_noise`)

### Changes

- In class `zScoreIncrements`, outliers are now detected by calculating the sum of the absolute differences between a
  data point and its respective preceding and next data point. Before, only the non-absolute difference of the preceding
  data point was considered. The sum of absolute differences is then used to calculate the z-score and in further
  consequence to flag outliers. (`diive.pkgs.preprocessing.outlier_detection.incremental.zScoreIncrements`)

### Notebooks

- Added new notebook for outlier detection using
  class `zScoreIncrements` (`notebooks/OutlierDetection/zScoreIncremental.ipynb`)
- Added new notebook for outlier detection using
  class `LocalSD` (`notebooks/OutlierDetection/LocalSD.ipynb`)

### Tests

- Added new test case for `zScoreIncrements` (`tests.test_outlierdetection.TestOutlierDetection.test_zscore_increments`)
- Added new test case for `LocalSD` (`tests.test_outlierdetection.TestOutlierDetection.test_localsd`)

## v0.76.0 | 14 May 2024

### Diel cycle plot

The new class `DielCycle` allows to plot diel cycles per month or across all data for time series data. At the moment,
it plots the (monthly) diel cycles as means (+/- standard deviation). It makes use of the time info contained in the
datetime timestamp index of the data. All aggregates are calculated by grouping data by time and (optional) separately
for each month. The diel cycles have the same time resolution as the time component of the timestamp index, e.g. hourly.

![DIIVE](images/plotDielCycle_diive_v0.76.0.png)

### New features

- Added new class `DielCycle` for plotting diel cycles per month (`diive.core.plotting.dielcycle.DielCycle`)
- Added new function `diel_cycle` for calculating diel cycles per month. This function is also used by the plotting
  class `DielCycle` (`diive.core.times.resampling.diel_cycle`)

### Additions

- Added color scheme that contains 12 colors, one for each month. Not perfect, but better than
  before. (`diive.core.plotting.styles.LightTheme.colors_12_months`)

### Notebooks

- Added new notebook for plotting diel cycles (per month) (`notebooks/Plotting/DielCycle.ipynb`)
- Added new notebook for calculating diel cycles (per month) (`notebooks/Resampling/ResamplingDielCycle.ipynb`)

### Tests

- Added test case for new function `diel_cycle` (`tests.test_resampling.TestResampling.test_diel_cycle`)

## v0.75.0 | 26 Apr 2024

### XGBoost gap-filling

[XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) can now be used to fill gaps in time series data.
In `diive`, `XGBoost` is implemented in class `XGBoostTS`, which adds additional options for easily including e.g.
lagged variants of feature variables, timestamp info (DOY, month, ...) and a continuous record number. It also allows
direct feature reduction by including a purely random feature (consisting of completely random numbers) and calculating
the 'permutation importance'. All features where the permutation importance is lower than for the random feature can
then be removed from the dataset, i.e., the list of features, before building the final model.

`XGBoostTS` and `RandomForestTS` both use the same base class `MlRegressorGapFillingBase`. This base class will also
facilitate the implementation of other gap-filling algorithms in the future.

Another fun (for me) addition is the new class `TimeSince`. It allows to calculate the time since the last occurrence of
specific conditions. One example where this class can be useful is the calculation of 'time since last precipitation',
expressed as number of records, which can be helpful in identifying dry conditions. More examples: 'time since freezing
conditions' based on air temperature; 'time since management' based on management info, e.g. fertilization events.
Please see the notebook for some illustrative examples.

**Please note that `diive` is still under developement and bugs can be expected.**

### New features

- Added gap-filling class `XGBoostTS` for time series data,
  using [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) (`diive.pkgs.gapfilling.xgboost_ts.XGBoostTS`)
- Added new class `TimeSince`: counts number of records (inceremental number / counter) since the last time a time
  series was inside a specified range, useful for e.g. counting the time since last precipitation, since last freezing
  temperature, etc. (`diive.pkgs.createvar.timesince.TimeSince`)

### Additions

- Added base class for machine learning regressors, which is basically the code shared between the different
  methods. At the moment used by `RandomForestTS` and `XGBoostTS`. (`diive.core.ml.common.MlRegressorGapFillingBase`)
- Added option to change line color directly in `TimeSeries` plots (`diive.core.plotting.timeseries.TimeSeries.plot`)

### Notebooks

- Added new notebook for gap-filling using `XGBoostTS` with mininmal
  settings (`notebooks/GapFilling/XGBoostGapFillingMinimal.ipynb`)
- Added new notebook for gap-filling using `XGBoostTS` with more extensive
  settings (`notebooks/GapFilling/XGBoostGapFillingExtensive.ipynb`)
- Added new notebook for creating `TimeSince` variables (`notebooks/CalculateVariable/TimeSince.ipynb`)

### Tests

- Added test case for XGBoost gap-filling (`tests.test_gapfilling.TestGapFilling.test_gapfilling_xgboost`)
- Updated test case for random forest gap-filling (`tests.test_gapfilling.TestGapFilling.test_gapfilling_randomforest`)
- Harmonized test case for XGBoostTS with test case of RandomForestTS
- Added test case for `TimeSince` variable creation (`tests.test_createvar.TestCreateVar.test_timesince`)

## v0.74.1 | 23 Apr 2024

This update adds the first notebooks (and tests) for outlier detection methods. Only two tests are included so far and
both tests are relatively simple, but both notebooks already show in principle how outlier removal is handled. An
important aspect is that `diive` single outlier methods do not remove outliers by default, but instead a flag is created
that shows where the outliers are located. The flag can then be used to remove the data points.  
This update also includes the addition of a small function that creates artificial spikes in time series data and is
therefore very useful for testing outlier detection methods.  
More outlier removal notebooks will be added in the future, including a notebook that shows how to combine results from
multiple outlier tests into one single overall outlier flag.

### New features

- **Added**: new function to add impulse noise to time series (`diive.pkgs.createvar.noise.impulse`)

### Notebooks

- **Added**: new notebook for outlier detection: absolute limits, separately for daytime and nighttime
  data (`notebooks/OutlierDetection/AbsoluteLimitsDaytimeNighttime.ipynb`)
- **Added**: new notebook for outlier detection: absolute limits (`notebooks/OutlierDetection/AbsoluteLimits.ipynb`)

### Tests

- **Added**: test case for outlier detection: absolute limits, separately for daytime and
  nighttime data (`tests.test_outlierdetection.TestOutlierDetection.test_absolute_limits`)
- **Added**: test case for outlier detection: absolute
  limits (`tests.test_outlierdetection.TestOutlierDetection.test_absolute_limits`)

## v0.74.0 | 21 Apr 2024

### Additions

- **Added**: new function to remove rows that do not have timestamp
  info (`NaT`) (`diive.core.times.times.remove_rows_nat` and `diive.core.times.times.TimestampSanitizer`)
- **Added**: new settings `VARNAMES_ROW` and `VARUNITS_ROW` in filetypes YAML files, allows better and more specific
  configuration when reading data files (`diive/configs/filetypes`)
- **Added**: many (small) example data files for various filetypes, e.g. `ETH-RECORD-TOA5-CSVGZ-20HZ`
- **Added**: new optional check in `TimestampSanitizer` that compares the detected time resolution of a time series with
  the nominal (expected) time resolution. Runs automatically when reading files with `ReadFileType`, in which case
  the `FREQUENCY` from the filetype configs is used as the nominal time
  resolution. (`diive.core.times.times.TimestampSanitizer`, `diive.core.io.filereader.ReadFileType`)
- **Added**: application of `TimestampSanitizer` after inserting a timestamp and setting it as index with
  function `insert_timestamp`, this makes sure the freq/freqstr info is available for the new timestamp
  index (`diive.core.times.times.insert_timestamp`)

### Notebooks

- General: Ran all notebook examples to make sure they work with this version of `diive`
- **Added**: new notebook for reading EddyPro _fluxnet_ output file with `DataFileReader`
  parameters (`notebooks/ReadFiles/Read_single_EddyPro_fluxnet_output_file_with_DataFileReader.ipynb`)
- **Added**: new notebook for reading EddyPro _fluxnet_ output file with `ReadFileType` and pre-defined
  filetype `EDDYPRO-FLUXNET-CSV-30MIN` (
  `notebooks/ReadFiles/Read_single_EddyPro_fluxnet_output_file_with_ReadFileType.ipynb`)
- **Added**: new notebook for reading multiple EddyPro _fluxnet_ output files with `MultiDataFileReader` and pre-defined
  filetype `EDDYPRO-FLUXNET-CSV-30MIN` (
  `notebooks/ReadFiles/Read_multiple_EddyPro_fluxnet_output_files_with_MultiDataFileReader.ipynb`)

### Changes

- **Renamed**: function `get_len_header` to `parse_header`(`diive.core.dfun.frames.parse_header`)
- **Renamed**: exampledata files (`diive/configs/exampledata`)
- **Renamed**: filetypes YAML files to always include the file extension in the file name (`diive/configs/filetypes`)
- **Reduced**: file size for most example data files

### Tests

- **Added**: various test cases for loading filetypes (`tests/test_loaddata.py`)
- **Added**: test case for loading and merging multiple
  files (`tests.test_loaddata.TestLoadFiletypes.test_load_exampledata_multiple_EDDYPRO_FLUXNET_CSV_30MIN`)
- **Added**: test case for reading EddyPro _fluxnet_ output file with `DataFileReader`
  parameters (
  `tests.test_loaddata.TestLoadFiletypes.test_load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN_datafilereader_parameters`)
- **Added**: test case for resampling series to 30MIN time
  resolution (`tests.test_time.TestTime.test_resampling_to_30MIN`)
- **Added**: test case for inserting timestamp with a different convention (middle, start,
  end) (`tests.test_time.TestTime.test_insert_timestamp`)
- **Added**: test case for inserting timestamp as index (`tests.test_time.TestTime.test_insert_timestamp_as_index`)

### Bugfixes

- **Fixed**: bug in class `DetectFrequency` when inferred frequency is `None` (`diive.core.times.times.DetectFrequency`)
- **Fixed**: bug in class `DetectFrequency` where `pd.Timedelta()` would crash if the input frequency does not have a
  number. `Timedelta` does not accept e.g. the frequency string `min` for minutely time resolution, even though
  e.g. `pd.infer_freq()` outputs `min` for data in 1-minute time resolution. `TimeDelta` requires a number, in this
  case `1min`. Results from `infer_freq()` are now checked if they contain a number and if not, `1` is added at the
  beginning of the frequency string. (`diive.core.times.times.DetectFrequency`)
- **Fixed**: bug in notebook `WindDirectionOffset`, related to frequency detection during heatmap plotting
- **Fixed**: bug in `TimestampSanitizer` where the script would crash if the timestamp contained an element that could
  not be converted to datetime, e.g., when there is a string mixed in with the regular timestamps. Data rows with
  invalid timestamps are now parsed as `NaT` by using `errors='coerce'`
  in `pd.to_datetime(data.index, errors='coerce')`.  (`diive.core.times.times.convert_timestamp_to_datetime`
  and `diive.core.times.times.TimestampSanitizer`)
- **Fixed**: bug when plotting heatmap (`diive.core.plotting.heatmap_datetime.HeatmapDateTime`)

## v0.73.0 | 17 Apr 2024

### New features

- Added new function `trim_frame` that allows to trim the start and end of a dataframe based on available records of a
  variable (`diive.core.dfun.frames.trim_frame`)
- Added new option to export borderless
  heatmaps (`diive.core.plotting.heatmap_base.HeatmapBase.export_borderless_heatmap`)

### Additions

- Added more info in comments of class `WindRotation2D` (`diive.pkgs.flux.hires.windrotation.WindRotation2D`)
- Added example data for EddyPro full_output
  files (`diive.configs.exampledata.load_exampledata_eddypro_full_output_CSV_30MIN`)
- Added code in an attempt to harmonize frequency detection from data: in class `DetectFrequency` the detected
  frequency strings are now converted from `Timedelta` (pandas) to `offset` (pandas) to `.freqstr`. This will yield
  the frequency string as seen by (the current version of) pandas. The idea is to harmonize between different
  representations e.g. `T` or `min` for minutes (
  see [here](https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html)). (
  `diive.core.times.times.DetectFrequency`)

### Changes

- Updated class `DataFileReader` to comply with new `pandas` kwargs when
  using `.read_csv()` (`diive.core.io.filereader.DataFileReader._parse_file`)
- Environment: updated `pandas` to v2.2.2 and `pyarrow` to v15.0.2
- Updated date offsets in config filetypes to be compliant with `pandas` version 2.2+ (
  see [here](https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html)
  and [here](https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects)), e.g., `30T` was changed
  to `30min`. This seems to work without raising a warning, however, if frequency is inferred from available data,
  the resulting frequency string shows e.g. `30T`, i.e. still showing `T` for minutes instead
  of `min`. (`diive/configs/filetypes`)
- Changed variable names in `WindRotation2D` to be in line with the variable names given in the paper by Wilczak et
  al. (2001) https://doi.org/10.1023/A:1018966204465

### Removals

- Removed function `timedelta_to_string` because this can be done with pandas `to_offset().freqstr`
- Removed function `generate_freq_str` (unused)

### Tests

- Added test case for reading EddyPro full_output
  files (`tests.test_loaddata.TestLoadFiletypes.test_load_exampledata_eddypro_full_output_CSV_30MIN`)
- Updated test for frequency detection (`tests.test_timestamps.TestTime.test_detect_freq`)

## v0.72.1 | 26 Mar 2024

- `pyproject.toml` now uses the inequality syntax `>=` instead of caret syntax `^` because the version capping is
  restrictive and prevents compatibility in conda installations. See [#74](https://github.com/holukas/diive/pull/74)
- Added badges in `README.md`
- Smaller `diive` logo in `README.md`

## v0.72.0 | 25 Mar 2024

### New feature

- Added new heatmap plotting class `HeatmapYearMonth` that allows to plot a variable in year/month
  classes(`diive.core.plotting.heatmap_datetime.HeatmapYearMonth`)

![DIIVE](images/plotHeatmapYearMonth_diive_v0.72.0.png)

### Changes

- Refactored code for class `HeatmapDateTime` (`diive.core.plotting.heatmap_datetime.HeatmapDateTime`)
- Added new base class `HeatmapBase` for heatmap plots. Currently used by `HeatmapYearMonth`
  and `HeatmapDateTime` (`diive.core.plotting.heatmap_base.HeatmapBase`)

### Notebooks

- Added new notebook for `HeatmapDateTime` (`notebooks/Plotting/HeatmapDateTime.ipynb`)
- Added new notebook for `HeatmapYearMonth` (`notebooks/Plotting/HeatmapYearMonth.ipynb`)

### Bugfixes

- Fixed bug in `HeatmapDateTime` where the last record of each day was not shown

## v0.71.6 | 23 Mar 2024

![DIIVE](images/analysesZaggregatesInQuantileClassesOfXY_diive_v0.71.6.png)

### Notebooks

- Added new notebook for `Percentiles` (`notebooks/Analyses/Percentiles.ipynb`)
- Added new notebook for `LinearInterpolation` (`notebooks/GapFilling/LinearInterpolation.ipynb`)
- Added new notebook for calculating z-aggregates in quantiles (classes) of x and
  y  (`notebooks/Analyses/CalculateZaggregatesInQuantileClassesOfXY.ipynb`)
- Updated notebook for `DaytimeNighttimeFlag` (`notebooks/CalculateVariable/DaytimeNighttimeFlag.ipynb`)

## v0.71.5 | 22 Mar 2024

### Changes

- Updated notebook for `SortingBinsMethod` (`diive.pkgs.analysis.decoupling.SortingBinsMethod`)

![DIIVE](images/analysesDecoupling_sortingBinsMethod_diive_v0.71.5.png)

*Plot showing vapor pressure deficit (y) in 10 classes of short-wave incoming radiation (x), separate for 5 classes of
air temperature (z). All values shown are medians of the respective variable. The shaded errorbars refer to the
interquartile range for the respective class. Plot was generated using the class `SortingBinsMethod`.*

## v0.71.4 | 20 Mar 2024

### Changes

- Refactored class `LongtermAnomaliesYear` (`diive.core.plotting.bar.LongtermAnomaliesYear`)

![DIIVE](images/plotBarLongtermAnomaliesYear_diive_v0.71.4.png)

### Notebooks

- Added new notebook for `LongtermAnomaliesYear` (`notebooks/Plotting/LongTermAnomalies.ipynb`)

## v0.71.3 | 19 Mar 2024

### Changes

- Refactored class `SortingBinsMethod`: Allows to investigate binned aggregates of a variable z in binned classes of x
  and y. All bins now show medians and interquartile
  ranges. (`diive.pkgs.analysis.decoupling.SortingBinsMethod`)

### Notebooks

- Added new notebook for `SortingBinsMethod`

### Bugfixes

- Added absolute links to example notebooks in `README.md`

### Other

- From now on, `diive` is officially published on [pypi](https://pypi.org/project/diive/)

## v0.71.2 | 18 Mar 2024

### Notebooks

- Added new notebook for `daily_correlation` function (`notebooks/Analyses/DailyCorrelation.ipynb`)
- Added new notebook for `Histogram` class (`notebooks/Analyses/Histogram.ipynb`)

### Bugfixes & changes

- Daily correlations are now returned with daily (`1d`) timestamp
  index (`diive.pkgs.analysis.correlation.daily_correlation`)
- Updated README
- Environment: Added [ruff](https://github.com/astral-sh/ruff) to dev dependencies for linting

## v0.71.1 | 15 Mar 2024

### Bugfixes & changes

- Fixed: Replaced all references to old filetypes using the underscore to their respective new filetype names,
  e.g. all occurrences of `EDDYPRO_FLUXNET_30MIN` were replaced with the new name `EDDYPRO-FLUXNET-CSV-30MIN`.
- Environment: Python 3.11 is now allowed in `pyproject.toml`: `python = ">=3.9,<3.12"`
- Environment: Removed `fitter` library from dependencies, was not used.
- Docs: Testing documentation generation using [Sphinx](https://www.sphinx-doc.org/en/master/), although it looks very
  rough at the moment.

## v0.71.0 | 14 Mar 2024

### High-resolution update

This update focuses on the implementation of several classes that work with high-resolution (20 Hz) data.

The main motivation behind these implementations is the upcoming new version of another
script, [dyco](https://github.com/holukas/dyco), which will make direct use of these new classes. `dyco` allows
to detect and remove time lags from time series data and can also handle drifting lags, i.e., lags that
are not constant over time. This is especially useful for eddy covariance data, where the detection of
accurate time lags is of high importance for the calculation of ecosystem fluxes.

![DIIVE](images/lagMaxCovariance_diive_v0.71.0.png)
*Plot showing the covariance between the turbulent departures of vertical wind and CO2 measurements.
Maximum (absolute) covariance was found at record -26, which means that the CO2 signal has to be shifted
by 26 records in relation to the wind data to obtain the maximum covariance between the two variables.
Since the covariance was calculated on 20 Hz data, this corresponds to a time lag of 1.3 seconds
between CO2 and wind (20 Hz = measurement every 0.05 seconds, 26 * 0.05 = 1.3), or, to put it
another way, the CO2 signal arrived 1.3 seconds later at the sensor than the wind signal. Maximum
covariance was calculated using the `MaxCovariance` class.*

### New features

- Added new class `MaxCovariance` to find the maximum covariance between two
  variables (`diive.pkgs.flux.hires.lag.MaxCovariance`)
- Added new class `FileDetector` to detect expected and unexpected files from a list of
  files (`diive.core.io.filesdetector.FileDetector`)
- Added new class `FileSplitter` to split file into multiple smaller parts and export them as multiple CSV
  files. (`diive.core.io.filesplitter.FileSplitter`)
- Added new class `FileSplitterMulti` to split multiple files into multiple smaller parts
  and save them as CSV or compressed CSV files. (`diive.core.io.filesplitter.FileSplitterMulti`)
- Added new function `create_timestamp` that calculates the timestamp for each record in a dataframe,
  based on number of records in the file and the file duration. (`diive.core.times.times.create_timestamp`)

### Additions

- Added new filetype `ETH-SONICREAD-BICO-CSVGZ-20HZ`, these files contain data that were originally logged
  by the `sonicread` script which is in use in the [ETH Grassland Sciences group](https://gl.ethz.ch/) since the early
  2000s to record eddy covariance data within the [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/). Data were
  then converted to a regular format using the Python script [bico](https://github.com/holukas/bico), which
  also compressed the resulting CSV files to `gz` files (`gzipped`).
- Added new filetype `GENERIC-CSV-HEADER-1ROW-TS-MIDDLE-FULL-NS-20HZ`, which corresponds to a CSV file with
  one header row with variable names, a timestamp that describes the middle of the averaging period, whereby
  the timestamp also includes nanoseconds. Time resolution of the file is 20 Hz.

### Changes

- Renamed class `TurbFlux` to `WindRotation2D` and updated code a bit, e.g., now it is possible to get
  rotated values for all three wind components (`u'`, `v'`, `w'`) in addition to the rotated
  scalar `c'`. (`diive.pkgs.flux.hires.windrotation.WindRotation2D`)
- Renamed filetypes: all filetypes now use the dash instead of an underscore
- Renamed filetype to `ETH-RECORD-DAT-20HZ`: this filetype originates from the new eddy covariance real-time
  logging script `rECord` (currently not open source)
- Missing values are now defined for all files
  as: `NA_VALUES: [ -9999, -6999, -999, "nan", "NaN", "NAN", "NA", "inf", "-inf", "-" ]`

## v0.70.1 | 1 Mar 2024

- Updated (and cleaned) notebook `StepwiseMeteoScreeningFromDatabase.ipynb`

## v0.70.0 | 28 Feb 2024

### New features

- In `StepwiseOutlierDetection`, it is now possible to re-run an outlier detection method. The re-run(s)
  would produce flag(s) with the same name(s) as for the first (original) run. Therefore, an integer is added
  to the flag name. For example, if the test z-score daytime/nighttime is run the first time, it produces the
  flag with the name `FLAG_TA_T1_2_1_OUTLIER_ZSCOREDTNT_TEST`. When the test is run again (e.g. with different
  settings) then the name of the flag of this second run is `FLAG_TA_T1_2_1_OUTLIER_ZSCOREDTNT_2_TEST`,
  etc ... The script now checks whether a flag of the same name was already created, in which case an
  integer is added to the flag name. These re-runs are now available in addition to the `repeat=True` keyword.
  (`diive.pkgs.preprocessing.outlier_detection.stepwiseoutlierdetection.StepwiseOutlierDetection.addflag`)
  Example:
    - `METHOD` with `SETTINGS` is applied with `repeat=True` and therefore repeated until no more outliers
      were found with these settings. The name of the flag produced is `TEST_METHOD_FLAG`.
    - Next, `METHOD` is applied again with `repeat=True`, but this time with different `SETTINGS`. Like before,
      the test is repeated until no more outliers were found with the new settings. The name of the flag produced
      is `TEST_METHOD_2_FLAG`.
    - `METHOD` can be re-run any number of times, each time producing a new
      flag: `TEST_METHOD_3_FLAG`, `TEST_METHOD_4_FLAG`, ...
- Added new function to format timestamps to FLUXNET ISO
  format (`YYYYMMDDhhmm`) (`diive.core.times.times.format_timestamp_to_fluxnet_format`)

### Bugfixes

- Refactored and fixed class to reformat data for FLUXNET
  upload (`diive.pkgs.formats.fluxnet.FormatEddyProFluxnetFileForUpload`)
- Fixed `None` error when reading data files (`diive.core.io.filereader.DataFileReader._parse_file`)

### Notebooks

- Updated notebook `FormatEddyProFluxnetFileForUpload.ipynb`

## v0.69.0 | 23 Feb 2024

### New features

- Added new functions to extract info from a binary that was stored as
  integer. These functions convert a subrange of bits from an integer or an integer series to floats with an
  optional gain applied. See docstring of the respective functions for more
  info. (`diive.pkgs.binary.extract.get_encoded_value_from_int`) (`diive.pkgs.binary.extract.get_encoded_value_series`)
- Added new filetype `RECORD_DAT_20HZ` (`diive/configs/filetypes/RECORD_DAT_20HZ.yml`) for eddy covariance
  high-resolution (20Hz) raw data files recorded by the ETH `rECord` logging script.

## v0.68.1 | 5 Feb 2024

- Fixed bugs in `FluxProcessingChain`, flag creation for missing values did not work because of the missing `repeat`
  keyword (`diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain`)

## v0.68.0 | 30 Jan 2024

### Updates to stepwise outlier detection

Harmonized the way outlier flags are calculated. Outlier flags are all based on the same base
class `diive.core.base.flagbase.FlagBase` like before, but the base class now includes more code that
is shared by the different outlier detection methods. For example, `FlagBase` includes a method that
enables repeated execution of a single outlier detection method multiple times until all outliers
are removed. Results from all iterations are then combined into one single flag.

The class `StepwiseMeteoScreeningDb` that makes direct use of the stepwise outlier detection was
adjusted accordingly.

### Notebooks

- Updated notebook `StepwiseMeteoScreeningFromDatabase.ipynb`

### Removed features

- Removed outlier test based on seasonal-trend decomposition and z-score calculations (`OutlierSTLRZ`).
  The test worked in principle, but at the moment it is unclear how to set reliable parameters. In addition
  the test is slow when used with multiple years of high-resolution data. De-activated for the moment.

## v0.67.1 | 10 Jan 2024

- Updated: many docstrings.

## v0.67.0 | 9 Jan 2024

### Updates to flux processing chain

The flux processing chain was updated in an attempt to make processing more streamlined and easier to follow. One of the
biggest changes is the implementation of the `repeat` keyword for outlier tests. With this keyword set to `True`, the
respective test is repeated until no more outliers can be found. How the flux processing chain can be used is shown in
the updated `FluxProcessingChain`notebook (`notebooks/FluxProcessingChain/FluxProcessingChain.ipynb`).

### New features

- Added new class `QuickFluxProcessingChain`, which allows to quickly execute a simplified version of the flux
  processing chain. This quick version runs with a lot of default values and thus not a lot of user input is needed,
  only some basic settings. (`diive.pkgs.fluxprocessingchain.fluxprocessingchain.QuickFluxProcessingChain`)
- Added new repeater function for outlier detection: `repeater` is wrapper that allows to execute an outlier detection
  method multiple times, where each iteration gets its own outlier flag. As an example: the simple z-score test is run
  a first time and then repeated until no more outliers are found. Each iteration outputs a flag. This is now used in
  the `StepwiseOutlierDetection` and thus the flux processing chain Level-3.2 (outlier detection) and the meteoscreening
  in `StepwiseMeteoScreeningDb` (not yet checked in this update). To repeat an outlier method use the `repeat` keyword
  arg (see the `FluxProcessingChain` notebook for examples).(
  `diive.pkgs.preprocessing.outlier_detection.repeater.repeater`)
- Added new function `filter_strings_by_elements`: Returns a list of strings from list1 that contain all of the elements
  in list2.(`core.funcs.funcs.filter_strings_by_elements`)
- Added new function `flag_steadiness_horizontal_wind_eddypro_test`: Create flag for steadiness of horizontal wind u
  from the sonic anemometer. Makes direct use of the EddyPro output files and converts the flag to a standardized 0/1
  flag.(`pkgs.qaqc.eddyproflags.flag_steadiness_horizontal_wind_eddypro_test`)

### Changes

- Added automatic calculation of daytime and nighttime flags whenever the flux processing chain is started
  flags (`diive.pkgs.fluxprocessingchain.fluxprocessingchain.FluxProcessingChain._add_swinpot_dt_nt_flag`)

### Removed features

- Removed class `ThymeBoostOutlier` for outlier detection. At the moment it was not possible to get it to work properly.

### Changes

- It appears that the kwarg `fmt` is used slightly differently for `plot_date` and `plot` in `matplotlib`. It seems it
  is always defined for `plot_date`, while it is optional for `plot`. Now using `fmt` kwarg to avoid the warning:
  *UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "o" (-> marker='o').
  The keyword argument will take precedence.* Therefore using 'fmt="X"' instead of 'marker="X"'. See also
  answer [here](https://stackoverflow.com/questions/69188540/userwarning-marker-is-redundantly-defined-by-the-marker-keyword-argument-when)

### Environment

- Removed `thymeboost`

## v0.66.0 | 2 Nov 2023

### New features

- Added new class `ScatterXY`: a simple scatter plot that supports bins (`core.plotting.scatter.ScatterXY`)

![DIIVE](images/ScatterXY_diive_v0.66.0.png)

### Notebooks

- Added notebook `notebooks/Plotting/ScatterXY.ipynb`

## v0.64.0 | 31 Oct 2023

### New features

- Added new class `DaytimeNighttimeFlag` to calculate daytime flag (1=daytime, 0=nighttime),
  nighttime flag (1=nighttime, 0=daytime) and potential radiation from latitude and
  longitude (`diive.pkgs.createvar.daynightflag.DaytimeNighttimeFlag`)

### Additions

- Added support for N2O and CH4 fluxes during the calculation of the `QCF` quality flag in class `FlagQCF`
- Added first code for USTAR threshold detection for NEE

### Notebooks

- Added new notebook `notebooks/CalculateVariable/Daytime_and_nighttime_flag.ipynb`

## v0.63.1 | 25 Oct 2023

### Changes

- `diive` repository is now hosted on GitHub.

### Additions

- Added first code for XGBoost gap-filling, not production-ready yet
- Added check if enough columns for lagging features in class `RandomForestTS`
- Added more details in report for class `FluxStorageCorrectionSinglePointEddyPro`

### Bugfixes

- Fixed check in `RandomForestTS` for bug in `QuickFillRFTS`: number of available columns was checked too early
- Fixed `QuickFillRFTS` implementation in `OutlierSTLRZ`
- Fixed `QuickFillRFTS` implementation in `ThymeBoostOutlier`

### Environment

- Added new package [xgboost](https://xgboost.readthedocs.io/en/stable/#)
- Updated all packages

## v0.63.0 | 5 Oct 2023

### New features

- Implemented feature reduction (permutation importance) as separate method in `RandomForestTS`
- Added new function to set values within specified time ranges to a constant
  value(`pkgs.corrections.setto_value.setto_value`)
    - The function is now also implemented as method
      in `StepwiseMeteoScreeningDb` (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.correction_setto_value`)

### Notebooks

- Updated notebook `notebooks/GapFilling/RandomForestGapFilling.ipynb`
- Updated notebook `notebooks/GapFilling/QuickRandomForestGapFilling.ipynb`
- Updated notebook `notebooks/MeteoScreening/StepwiseMeteoScreeningFromDatabase.ipynb`

### Environment

- Added new package [SHAP](https://shap.readthedocs.io/en/latest/)
- Added new package [eli5](https://pypi.org/project/eli5/)

### Tests

- Updated testcase for gap-filling with random
  forest (`test_gapfilling.TestGapFilling.test_gapfilling_randomforest`)

## v0.62.0 | 1 Oct 2023

### New features

- Re-implemented gap-filling of long-term time series spanning multiple years, where the model
  to gap-fill a specific year is built from data from the respective year and its two closest
  neighboring years. (`pkgs.gapfilling.randomforest_ts.LongTermRandomForestTS`)

### Bugfixes

- Fixed bug in `StepwiseMeteoScreeningDb` where position of `return` during setup was incorrect

## v0.61.0 | 28 Sep 2023

### New features

- Added function to calculate the daily correlation between two time
  series (`pkgs.analyses.correlation.daily_correlation`)
- Added function to calculate potential radiation (`pkgs.createvar.potentialradiation.potrad`)

### Bugfixes

- Fixed bug in `StepwiseMeteoScreeningDb` where the subclass `StepwiseOutlierDetection`
  did not use the already sanitized timestamp from the parent class, but sanitized the timestamp
  a second time, leading to potentially erroneous and irregular timestamps.

### Changes

- `RandomForestTS` now has the following functions included as methods:
    - `steplagged_variants`: includes lagged variants of features
    - `include_timestamp_as_cols`: includes timestamp info as data columns
    - `add_continuous_record_number`: adds continuous record number as new column
    - `sanitize`: validates and prepares timestamps for further processing
- `RandomForestTS` now outputs an additional predictions column where predictions from
  the full model and predictions from the fallback model are collected
- Renamed function `steplagged_variants` to `lagged_variants` (`core.dfun.frames.lagged_variants`)
- Updated function `lagged_variants`: now accepts a list of lag times. This makes it possible
  to lag variables in both directions, i.e., the observed value can be paired with values before
  and after the actual time. For example, the variable `TA` is the observed value at the current
  timestamp, `TA-1` is the value from the preceding record, and `TA+1` is the value from the next
  record. Using values from the next record can be useful when modeling observations using data
  from a neighboring measurement location that has similar records but lagged in time due to
  distance.
- Updated README

### Tests

- Updated testcase for gap-filling with random
  forest (`test_gapfilling.TestGapFilling.test_gapfilling_randomforest`)

### Notebooks

- Updated `notebooks/MeteoScreening/StepwiseMeteoScreeningFromDatabase.ipynb`

### Additions

- Added more args for better control of `TimestampSanitizer` (`core.times.times.TimestampSanitizer`)
- Refined various docstrings

## v0.60.0 | 17 Sep 2023

### New features

- Added new class for optimizing random forest parameters (`pkgs.gapfilling.randomforest_ts.OptimizeParamsRFTS`)
- Added new plots for prediction error and residuals (`core.ml.common.plot_prediction_residuals_error_regr`)
- Added function that adds a continuous record number as new column in a dataframe. This
  could be useful to include as feature in gap-filling models for long-term datasets spanning multiple years.
  (`core.dfun.frames.add_continuous_record_number`)

### Changes

- When reading CSV files with pandas `.read_csv()`, the arg `mangle_dupe_cols=True`
  was removed because it is deprecated since pandas 2.0 ...
- ... therefore the check for duplicate column names in class `ColumnNamesSanitizer`
  has been refactored. In case of duplicate columns names, an integer suffix is added to
  the column name. For example: `VAR` is renamed to `VAR.1` if it already exists in the
  dataframe. In case `VAR.1` also already exists, it is renamed to `VAR.2`, and so on.
  The integer suffix is increased until the variable name is unique. (`core.io.filereader.ColumnNamesSanitizer`)
- Similarly, when reading CSV files with pandas `.read_csv()`, the arg `date_parser` was
  removed because it is deprecated since pandas 2.0. When reading a CSV, the arg `date_format`
  is now used instead. The input format remains unchanged, it is still a string giving the datetime
  format, such as `"%Y%m%d%H%M"`.
- The random feature variable is now generated using the same random state as the
  model. (`pkgs.gapfilling.randomforest_ts.RandomForestTS`)
- Similarly, `train_test_split` is now also using the same random state as the
  model. (`pkgs.gapfilling.randomforest_ts.RandomForestTS`)

### Notebooks

- Added new notebook `notebooks/GapFilling/RandomForestParamOptimization.ipynb`

### Tests

- Added testcase for loading dataframe from parquet file (`test_loaddata.TestLoadFiletypes.test_exampledata_parquet`)
- Added testcase for gap-filling with random forest (`test_gapfilling.TestGapFilling.test_gapfilling_randomforest`)

### Environment

- Updated `poetry` to latest version `1.6.1`
- Updated all packages to their latest versions
- Added new package [yellowbrick](https://www.scikit-yb.org/en/latest/)

## v0.59.0 | 14 Sep 2023

### MeteoScreening from database - update

The class `StepwiseMeteoScreeningDb`, which is used for quality-screening of meteo data
stored in the ETH Grassland Sciences database, has been refactored. It is now using the
previously introduced class `StepwiseOutlierDetection` for outlier
tests. (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb`)

### Removed

The following classes are no longer used and were removed from step-wise outlier detection:

- Removed z-score IQR test, too unreliable (`pkgs.outlierdetection.zscore.zScoreIQR`)
- Similarly, removed seasonal trend decomposition that used z-score IQR test, too
  unreliable (`pkgs.outlierdetection.seasonaltrend.OutlierSTLRIQRZ`)

### Notebooks

- Updated notebook `notebooks/MeteoScreening/StepwiseMeteoScreeningFromDatabase.ipynb`

## v0.58.1 | 13 Sep 2023

### Notebooks

- Added new notebook `notebooks/GapFilling/RandomForestGapFilling.ipynb`
- Added new notebook `notebooks/GapFilling/QuickRandomForestGapFilling.ipynb`
- Added new notebook `notebooks/Workbench/Remove_unneeded_cols.ipynb`

## v0.58.0 | 7 Sep 2023

### Random forest update

The class `RandomForestTS` has been refactored. In essence, it still uses the same
`RandomForestRegressor` as before, but now outputs feature importances additionally
as computed by permutation. More details about permutation importance can be found
in scikit's official documentation
here: [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

When the model is trained using `.trainmodel()`, a random variable is included as additional
feature. Permutation importances of all features - including the random variable - are then
analyzed. Variables that yield a lower importance score than the random variables are removed
from the dataset and are not used to build the model. Typically, the permutation importance for
the random variable is very close to zero or even negative.

The built-in importance calculation in the `RandomForestRegressor` uses the Gini importance,
an impurity-based feature importance that favors high cardinality features over low cardinality
features. This is not ideal in case of time series data that is combined with categorical data.
Permutation importance is therefore a better indicator whether a variable included in the model
is an important predictor or not.

The class now splits input data into training and testing datasets (holdout set). By
default, the training set comprises 75% of the input data, the testing set 25%. After
the model was trained, it is tested on the testing set. This should give a
better indication of how well the model works on unseen data.

Once `.trainmodel()` is finished, the model is stored internally and can be used to gap-fill
the target variable by calling `.fillgaps()`.

In addition, the class now offers improved output with additional text output and plots
that give more info about model training, testing and application during gap-filling.

`RandomForestTS` has also been streamlined. The option to include timestamp info as features
(e.g., a column describing the season of the respective record) during model building is now
its own function (`.include_timestamp_as_cols()`) and was removed from the class.

### New features

- New class `QuickFillRFTS` that uses `RandomForestTS` in the background to quickly fill time series
  data (`pkgs.gapfilling.randomforest_ts.QuickFillRFTS`)
- New function to include timestamp info as features, e.g. YEAR and DOY (`core.times.times.include_timestamp_as_cols`)
- New function to calculate various model scores, e.g. mean absolute error, R2 and
  more (`core.ml.common.prediction_scores_regr`)
- New function to insert the meteorological season (Northern hemisphere) as variable (`core.times.times.insert_season`).
  For each record in the time series, the seasonal info between spring (March, April, May) and winter (December,
  January, February) is added as integer number (0=spring, summer=1, autumn=2, winter=3).

### Additions

- Added new example dataset, comprising ecosystem fluxes between 1997 and 2022 from the
  [ICOS Class 1 Ecosystem station CH-Dav](https://www.swissfluxnet.ethz.ch/index.php/sites/ch-dav-davos/site-info-ch-dav/).
  This dataset will be used for testing code on long-term time series. The dataset is stored in the `parquet`
  file format, which allows fast loading and saving of datafiles in combination with good compression.
  The simplest way to load the dataset is to use:

```python
from diive.configs.exampledata import load_exampledata_parquet

df = load_exampledata_parquet()
```

### Changes

- Updated README with installation details

### Notebooks

- Updated notebook `notebooks/CalculateVariable/Calculate_VPD_from_TA_and_RH.ipynb`

## v0.57.1 | 23 Aug 2023

### Changes

Updates to class `FormatEddyProFluxnetFileForUpload`, for quickly formatting the EddyPro _fluxnet_
output file to comply with [FLUXNET](https://fluxnet.org/) requirements for uploading data.

### Additions

- **Formatting EddyPro _fluxnet_ files for upload to FLUXNET**: `FormatEddyProFluxnetFileForUpload`

    - Added new method to rename variables from the EddyPro _fluxnet_ file to comply
      with [FLUXNET variable codes](http://www.europe-fluxdata.eu/home/guidelines/how-to-submit-data/variables-codes).
      `._rename_to_variable_codes()`
    - Added new method to remove errneous time periods from dataset `.remove_erroneous_data()`
    - Added new method to remove fluxes from time periods of insufficient signal strength / AGC
      `.remove_low_signal_data()`

### Bugfixes

- Fixed bug: when data points are removed manually using class `ManualRemoval` and the data to be removed
  is a single datetime (e.g., `2005-07-05 23:15:00`) then the removal now also works if the
  provided datetime is not found in the time series. Previously, the class raised the error that
  the provided datetime is not part of the index. (`pkgs.outlierdetection.manualremoval.ManualRemoval`)

### Notebooks

- Updated notebook `notebooks/Formats\FormatEddyProFluxnetFileForUpload.ipynb` to version `3`

## v0.57.0 | 22 Aug 2023

### Changes

- Relaxed conditions a bit when inferring time resolution of time
  series (`core.times.times.timestamp_infer_freq_progressively`, `core.times.times.timestamp_infer_freq_from_timedelta`)

### Additions

- When reading parquet files, the TimestampSanitizer is applied by default to detect e.g. the time resolution
  of the time series. Parquet files do not store info on time resolution like it is stored in pandas dataframes
  (e.g. `30T` for 30MIN time resolution), even if the dataframe containing that info was saved to a parquet file.

### Bugfixes

- Fixed bug where interactive time series plot did not show in Jupyter notebooks (`core.plotting.timeseries.TimeSeries`)
- Fixed bug where certain parts of the flux processing chain could not be used for the sensible heat flux `H`.
  The issue was that `H` is calculated from sonic temperature (`T_SONIC` in EddyPro `_fluxnet_` output files),
  which was not considered in function `pkgs.flux.common.detect_flux_basevar`.
- Fixed bug: interactive plotting in notebooks using `bokeh` did not work. The reason was that the `bokeh` plot
  tools (controls) `ZoomInTool()` and `ZoomOutTool()` do not seem to work anymore. Both tools are now deactivated.

### Notebooks

- Added new notebook for simple (interactive) time series plotting `notebooks/Plotting/TimeSeries.ipynb`
- Updated notebook `notebooks/FluxProcessingChain/FluxProcessingChain.ipynb` to version 3

## v0.55.0 | 18 Aug 2023

This update focuses on the flux processing chain, in particular the creation of the extended
quality flags, the flux storage correction and the creation of the overall quality flag `QCF`.

### New Features

- Added new class `StepwiseOutlierDetection` that can be used for general outlier detection in
  time series data. It is based on the `StepwiseMeteoScreeningDb` class introduced in v0.50.0,
  but aims to be more generally applicable to all sorts of time series data stored in
  files (`pkgs.outlierdetection.stepwiseoutlierdetection.StepwiseOutlierDetection`)
- Added new outlier detection class that identifies outliers based on seasonal-trend decomposition
  and z-score calculations (`pkgs.outlierdetection.seasonaltrend.OutlierSTLRZ`)
- Added new outlier detection class that flags values based on absolute limits that can be defined
  separately for daytime and nighttime (`pkgs.outlierdetection.absolutelimits.AbsoluteLimitsDaytimeNighttime`)
- Added small functions to directly save (`core.io.files.save_as_parquet`) and
  load (`core.io.files.load_parquet`) parquet files. Parquet files offer fast loading and saving in
  combination with good compression. For more information about the Parquet format
  see [here](https://parquet.apache.org/)

### Additions

- **Angle-of-attack**: The angle-of-attack test can now be used during QC flag creation
  (`pkgs.fluxprocessingchain.level2_qualityflags.FluxQualityFlagsLevel2.angle_of_attack_test`)
- Various smaller additions

### Changes

- Renamed class `FluxQualityFlagsLevel2` to `FluxQualityFlagsLevel2EddyPro` because it is directly based
  on the EddyPro output (`pkgs.fluxprocessingchain.level2_qualityflags.FluxQualityFlagsLevel2EddyPro`)
- Renamed class `FluxStorageCorrectionSinglePoint`
  to `FluxStorageCorrectionSinglePointEddyPro` (
  `pkgs.fluxprocessingchain.level31_storagecorrection.FluxStorageCorrectionSinglePointEddyPro`)
- Refactored creation of flux quality
  flags (`pkgs.fluxprocessingchain.level2_qualityflags.FluxQualityFlagsLevel2EddyPro`)
- **Missing storage correction terms** are now gap-filled using random forest before the storage terms are
  added to the flux. For some records, the calculated flux was available but the storage term was missing, resulting
  in a missing storage-corrected flux (example: 97% of fluxes had storage term available, but for 3% it was missing).
  The gap-filling makes sure that each flux values has a corresponding storage term and thus more values are
  available for further processing. The gap-filling is done solely based on timestamp information, such as DOY
  and hour. (`pkgs.fluxprocessingchain.level31_storagecorrection.FluxStorageCorrectionSinglePoint`)
- The **outlier detection using z-scores for daytime and nighttime data** uses latitude/longitude settings to
  calculate daytime/nighttime via `pkgs.createvar.daynightflag.nighttime_flag_from_latlon`. Before z-score
  calculation, the time resolution of the time series is now checked and assigned automatically.
  (`pkgs.outlierdetection.zscore.zScoreDaytimeNighttime`)
- Removed `pkgs.fluxprocessingchain.level32_outlierremoval.FluxOutlierRemovalLevel32` since flux outlier
  removal is now done in the generally applicable class `StepwiseOutlierDetection` (see new features)
- Various smaller changes and refactorings

### Environment

- Updated `poetry` to newest version `v1.5.1`. The `lock` files have a new format since `v1.3.0`.
- Created new `lock` file for `poetry`.
- Added new package `pyarrow`.
- Added new package `pymannkendall` (see [GitHub](https://pypi.org/project/pymannkendall/)) to analyze
  time series data for trends. Functions of this package are not yet implemented in `diive`.

### Notebooks

- Added new notebook for loading and saving parquet files in `notebooks/Formats/LoadSaveParquetFile.ipynb`
- **Flux processing chain**: Added new notebook for flux post-processing
  in `notebooks/FluxProcessingChain/FluxProcessingChain.ipynb`.

## v0.54.0 | 16 Jul 2023

### New Features

- Identify critical heat days for ecosytem flux NEE (net ecosystem exchange, based on air temperature and VPD
  (`pkgs.flux.criticalheatdays.FluxCriticalHeatDaysP95`)
- Calculate z-aggregates in classes of x and y (`pkgs.analyses.quantilexyaggz.QuantileXYAggZ`)
- Plot heatmap from pivoted dataframe, using x,y,z values (`core.plotting.heatmap_xyz.HeatmapPivotXYZ`)
- Calculate stats for time series and store results in dataframe (`core.dfun.stats.sstats`)
- New helper function to load and merge files of a specific filetype (`core.io.files.loadfiles`)

### Additions

- Added more parameters when formatting EddyPro _fluxnet_ file for FLUXNET
  (`pkgs.formats.fluxnet.FormatEddyProFluxnetFileForUpload`)

### Changes

- Removed left-over code
- Multiple smaller refactorings

### Notebooks

- Added new notebook for calculating VPD in `notebooks/CalculateVariable/Calculate_VPD_from_TA_and_RH.ipynb`
- Added new notebook for calculating time series stats `notebooks/Stats/TimeSeriesStats.ipynb`
- Added new notebook for formatting EddyPro output for upload to
  FLUXNET `notebooks/Formats/FormatEddyProFluxnetFileForUpload.ipynb`

## v0.53.3 | 23 Apr 2023

### Notebooks

- Added new notebooks for reading data files (ICOS BM files)
- Added additional output to other notebooks
- Added new notebook section `Workbench` for practical use cases

### Additions

- New filetype `configs/filetypes/ICOS_H1R_CSVZIP_1MIN.yml`

## v0.53.2 | 23 Apr 2023

### Changes

- Added more output for detecting frequency from timeseries index (`core.times.times.DetectFrequency`)
    - The associated functions have been updated accordingly: `core.times.times.timestamp_infer_freq_from_fullset`,
      `core.times.times.timestamp_infer_freq_progressively`, `core.times.times.timestamp_infer_freq_from_timedelta`
    - Added new notebook (`notebooks/TimeStamps/Detect_time_resolution.ipynb` )
    - Added new unittest (`tests/test_timestamps.py`)

## v0.53.1 | 18 Apr 2023

### Changes

- **GapFinder** now gives by default sorted output, i.e. the output dataframe shows start and
  end date for the largest gaps first (`pkgs.analyses.gapfinder.GapFinder`)

### Notebooks

- Added new notebook for **finding gaps in time series** in `notebooks/Analyses/GapFinder.ipynb`
- Added new notebook for **time functions** in `notebooks/TimeFunctions/times.ipynb`

### Other

- New repository branch `indev` is used as developement branch from now on
- Branch `main` will contain code from the most recent release

## v0.53.0 | 17 Apr 2023

This update focuses on wind direction time series and adds the first example notebooks
to `diive`. From now on, new example notebooks will be added regularly.

### New features

- **Wind direction offset correction**: Compare yearly wind direction histograms to
  reference, detect offset in comparison to reference and correct wind directions
  for offset per year (`pkgs.corrections.winddiroffset.WindDirOffset`)
- **Wind direction aggregation**: Calculate mean etc. of wind direction in
  degrees (`core.funcs.funcs.winddirection_agg_kanda`)

### Notebooks

- Added new notebook for **wind direction offset correction** in `notebooks/Corrections/WindDirectionOffset.ipynb`
- Added new notebok for **reading ICOS BM files** in `notebooks/ReadFiles/Read_data_from_ICOS_BM_files.ipynb`

### Changes

- **Histogram analysis** now accepts pandas Series as input (`pkgs.analyses.histogram.Histogram`)

### Additions

- Added unittests for reading (some) filetypes

## v0.52.7 | 16 Mar 2023

### Additions

- The DataFileReader can now directly read zipped files (`core.io.filereader.DataFileReader`)
- **Interactive time series plot**: (`core.plotting.timeseries.TimeSeries.plot_interactive`)
    - added x- and y-axis to the plots
    - new parameters `width` and `height` allow to control the size of the plot
    - more controls such as undo/redo and zoom in/zoom out buttons were added
- The filetypes defined in `diive/configs/filetypes` now accept the setting `COMPRESSION: "zip"`.
  In essence, this allows to read zipped files directly.
- New filetype `ICOS_H2R_CSVZIP_10S`

### Changes

- Compression in filetypes is now given as `COMPRESSION: "None"` for no compression,
  and `COMPRESSION: "zip"` for zipped CSV files.

## v0.52.6 | 12 Mar 2023

### Additions

- `LocalSD` in `StepwiseMeteoScreeningDb` now accepts the parameter `winsize` to
  define the size of the rolling window (default `None`, in which case the window
  size is calculated automatically as 1/20 of the number of records).
  (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.flag_outliers_localsd_test`)

### Bugfix

- Fixed bug: outlier test `LocalSD` did not consider user input `n_sd`
  (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.flag_outliers_localsd_test`)

## v0.52.4 and v0.52.5 | 10 Mar 2023

### Bugfix

- Fixed bug: during resampling, the info for the tag `data_version` was incorrectly
  stored in tag `freq`. (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.resample`)

## v0.52.3 | 10 Mar 2023

### Additions

- Added plotting library `bokeh` to dependencies

### Changes

- When combining data of different time resolutions, the data are now combined using
  `.combine_first()` instead of `.concat()` to avoid duplicates during merging. This
  should work reliably because data of the highest resolution are available first, and then
  lower resolution upsampled (backfilled) data are added, filling gaps in the high
  resolution data. Because gaps are filled, overlaps between the two resolutions are avoided.
  With `.concat()`, gaps were not filled, but timestamps were simply added as new records,
  and thus duplicates in the timestamp occurred.
  (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb._harmonize_timeresolution`)
- Updated dependencies to newest possible versions

## v0.52.2 | 9 Mar 2023

### Changes

- Removed the packages `jupyterlab` and `jupyter-bokeh` from dependencies, because
  the latter caused issues when trying to install `diive` in a `conda` environment
  on a shared machine. Both dependencies are still listed in the `pyproject.toml`
  file as `dev` dependencies. It makes sense to keep both packages separate from
  `diive` because they are specifically for `jupyter` notebooks and not strictly
  related to `diive` functionality.

## v0.52.1 | 7 Mar 2023

### Additions

- In `StepwiseMeteoScreeningDb` the current cleaned timeseries can now be
  plotted with `showplot_current_cleaned`.
- Timeseries can now be plotted using the `bokeh` library. This plot are interactive
  and can be directly used in jupyter notebooks. (`core.plotting.timeseries.TimeSeries`)
- Added new plotting package `jupyter_bokeh` for interactive plotting in Jupyter lab.
- Added new plotting package `seaborn`.

### Bugfixes

- `StepwiseMeteoScreeningDb` now works on a copy of the input data to avoid
  unintended data overwrite of input.

## v0.52.0 | 6 Mar 2023

### New Features

- **Data formats**: Added new package `diive/pkgs/formats` that assists in converting
  data outputs to formats required e.g. for data sharing with FLUXNET.
    - Convert the EddyPro `_fluxnet_` output file to the FLUXNET data format for
      data upload (data sharing). (`pkgs.formats.fluxnet.ConvertEddyProFluxnetFileForUpload`)
- **Insert timestamp column**: Insert timestamp column that shows the START, END
  or MIDDLE time of the averaging interval (`core.times.times.insert_timestamp`)
- **Manual removal of data points**: Flag manually defined data points as outliers.
  (`pkgs.outlierdetection.manualremoval.ManualRemoval`)

### Additions

Added additional outlier detection algorithms
to `StepwiseMeteoScreeningDb` (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb`):

- Added local outlier factor test, across all data
  (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.flag_outliers_lof_test`)
- Added local outlier factor test, separately for daytime and nighttime
  (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.flag_outliers_lof_dtnt_test`)

## v0.51.0 | 3 Mar 2023

### Random uncertainty

- Implemented random flux uncertainty calculation, based on Holliner and Richardson (2005)
  and Pastorello et al. (2020). Calculations also include a first estimate of the error
  propagation when summing up flux values to annual sums. See end of CHANGELOG for links to references.
  (`pkgs.flux.uncertainty.RandomUncertaintyPAS20`)

### Additions

- Added example data in `diive/configs/exampledata`, including functions to load the data.

### Changes

- In `core.io.filereader`, the following classes now also accept `output_middle_timestamp`
  (boolean with default `True`) as parameter: `MultiDataFileReader`, `ReadFileType`,`DataFileReader`.
  This allows to keep the original timestamp of the data.
- Some minor plotting adjustments

## v0.50.0 | 12 Feb 2023

### StepwiseMeteoScreeningDb

**Stepwise quality-screening of meteorological data, directly from the database**

In this update, the stepwise meteoscreening directly from the database introduced in the
previous update was further refined and extended, with additional outlier tests and corrections
implemented. The stepwise meteoscreening allows to perform step-by-step quality tests on
meteorological. A preview plot after running a test is shown and the user can decide if
results are satisfactory or if the same test with different parameters should be re-run.
Once results are satisfactory, the respective test flag is added to the data. After running
the desired tests, an overall flag `QCF` is calculated from all individual tests.

In addition to the creation of quality flags, the stepwise screening allows to correct
data for common issues. For example, short-wave radiation sensors often measure negative
values during the night. These negative values are useful because they give info about
the accuracy and precision of the sensor. In this case, values during the night should
be zero. Instead of cutting off negative values, `diive` detects the nighttime offset
for each day and then calculates a correction slope between individual days. This way,
the daytime values are also corrected.

After quality-screening and corrections, data are resampled to 30MIN time resolution.

At the moment, the stepwise meteoscreening works for data downloaded from the `InfluxDB`
database. The screening respects the database format (including tags) and prepares
the screened, corrected and resampled data for direct database upload.

Due to its modular approach, the stepwise screening can be easily adjusted
to work with any type of data files. This adjustment will be done in one of the next
updates.

### Changes

- Renamed class `MetScrDbMeasurementVars`
  to `StepwiseMeteoScreeningDb` (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb`)

### Additions

- **Stepwise MeteoScreening**:
  Added access to multiple methods for easy stepwise execution:
    - Added local SD outlier test (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.flag_outliers_localsd_test`)
    - Added absolute limits outlier test (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.flag_outliers_abslim_test`)
    - Added correction to remove radiation zero
      offset (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.correction_remove_radiation_zero_offset`)
    - Added correction to remove relative humidity
      offset (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.correction_remove_relativehumidity_offset`)
    - Added correction to set values above a threshold to
      threshold (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.correction_setto_max_threshold`)
    - Added correction to set values below a threshold to
      threshold (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.correction_setto_min_threshold`)
    - Added comparison plot before/after QC and
      corrections (`pkgs.qaqc.meteoscreening.StepwiseMeteoScreeningDb.showplot_resampled`)

## v0.49.0 | 10 Feb 2023

### New Features

- **Stepwise MeteoScreening**: (`pkgs.qaqc.meteoscreening.MetScrDbMeasurementVars`)
    - **Helper class to screen time series of meteo variables directly from the
      database**. The class is optimized to work in Jupyter notebooks. Various outlier
      detection methods can be called on-demand. Outlier results are displayed and
      the user can accept the results and proceed, or repeat the step with adjusted
      method parameters. An unlimited amount of tests can be chained together. At
      the end of the screening, an overall flag is calculated from ALL single flags.
      The overall flag is then used to filter the time series.
    - **Variables**: The class allows the simultaneous quality-screening of multiple
      variables from one single measurement, e.g., multiple air temperature variables.
    - **Resampling**:Filtered time series are resampled to 30MIN time resolution.
    - **Database tags**: Is optimized to work with the InfluxDB format of the ETH
      Grassland Sciences Group. The class can handle database tags and updates tags
      after data screening and resampling.
    - **Handling different time resolutions**: One challenging aspect of the screening
      were the different time resolutions of the raw data. In some cases, the time
      resolution changed from e.g. 10MIN for older data to 1MIN for newer date. In
      cases of different time resolution, the lower resolution is upsampled to the
      higher resolution, the emerging gaps are back-filled with available data.
      Back-filling is used because the timestamp in the database always is TIMESTAMP_END,
      i.e., it gives the *end* of the averaging interval. The advantage of upsampling
      is that all outlier detection routines can be applied to the whole dataset.
      Since data are resampled to 30MIN after screening and since the TIMESTAMP_END
      is respected, the upsampling itself has no impact on resulting aggregates.

### Changes

- Generating the plot NEP penalty vs hours above threshold now requires a
  minimum of 2 bootstrap runs to calculate prediction intervals
  (`pkgs.flux.nep_penalty.NEPpenalty.plot_critical_hours`)

### Bugfixes

- Fixed bug in `BinFitter`, the parameter to set the number of predictions is now correctly
  named `n_predictions`. Similar `n_bins_x`.
- Fixed typos in functions `insert_aggregated_in_hires`, `SortingBinsMethod`, `FindOptimumRange`
  and `pkgs.analyses.optimumrange.FindOptimumRange._values_in_optimum_range` and others.
- Other typos

## v0.48.0 | 1 Feb 2023

### New Features

- **USTAR threshold**: (`pkgs.flux.ustarthreshold.UstarThresholdConstantScenarios`)
    - Calculates how many records of e.g. a flux variable are still available after the application
      of different USTAR thresholds. In essence, it gives an overview of the sensitivity of the
      variable to different thresholds.
- **Outlier detection, LOF across all data**: (`pkgs.outlierdetection.lof.LocalOutlierFactorAllData`)
    - Calculation of the local outlier factor across all data, i.e., no differentiation between
      daytime and nighttime data.
- **Outlier detection, increments**: (`pkgs.outlierdetection.incremental.zScoreIncremental`)
    - Based on the absolute change of on record in comparison to the previous record. These
      differences are stored as timeseries, the z-score is calculated and outliers are removed
      based on the observed differences. Works well with data that do not have a diel cycle,
      e.g. soil water content.

![DIIVE](images/fluxUstarthreshold_UstarThresholdConstantScenarios_diive_v0.48.0.png)

## v0.47.0 | 28 Jan 2023

### New Features

- **Outlier detection**: LOF, local outlier factor**: (`pkgs.outlierdetection.lof.LocalOutlierFactorDaytimeNighttime`)
    - Identify outliers based on the local outlier factor, done separately for
      daytime and nighttime data
- **Multiple z-score outlier detections**:
    - Simple outlier detection based on the z-score of observations, calculated from
      mean and std from the complete timeseries. (`pkgs.outlierdetection.zscore.zScore`)
    - z-score outlier detection separately for daytime and nighttime
      data (`pkgs.outlierdetection.zscore.zScoreDaytimeNighttime`)
    - Identify outliers based on the z-score of the interquartile range data (`pkgs.outlierdetection.zscore.zScoreIQR`)
- **Outlier detection**: (`pkgs.fluxprocessingchain.level32_outlierremoval.OutlierRemovalLevel32`):
    - Class that allows to apply multiple methods for outlier detection during as part of the flux processing chain

### Changes

- **Flux Processing Chain**:
    - Worked on making the chain more accessible to users. The purpose of the modules in
      `pkgs/fluxprocessingchain` is to expose functionality to the user, i.e., they make
      functionality needed in the chain accessible to the user. This should be as easy as possible
      and this update further simplified this access. At the moment there are three modules in
      `pkgs/fluxprocessingchain/`: `level2_qualityflags.py`, `level31_storagecorrection.py` and
      `level32_outlierremoval.py`. An example for the chain is given in `fluxprocessingchain.py`.
- **QCF flag**: (`pkgs.qaqc.qcf.FlagQCF`)
    - Refactored code: the creation of overall quality flags `QCF` is now done using the same
      code for flux and meteo data. The general logic of the `QCF` calculation is that results
      from multiple quality checks that are stored as flags in the data are combined into
      one single quality flag.
- **Outlier Removal using STL**:
    - Module was renamed to `pkgs.outlierdetection.seasonaltrend.OutlierSTLRIQRZ`. It is not the
      most convenient name, I know, but it stands for **S**easonal **T**rend decomposition using
      **L**OESS, based on **R**esidual analysis of the **I**nter**Q**uartile **R**ange using **Z**-scores
- **Search files** can now search in subfolders of multiple base folders (`core.io.filereader.search_files`)

## v0.46.0 | 23 Jan 2023

### New Features

- **Outlier Removal using STL**: (`pkgs.outlierdetection.seasonaltrend.OutlierSTLIQR`)
    - Implemented first code to remove outliers using seasonal-srend decomposition using LOESS.
      This method divides a time series into seasonal, trend and residual components. `diive`
      uses the residuals to detect outliers based on z-score calculations.
- **Overall quality flag for meteo data**: (`pkgs.qaqc.qcf.MeteoQCF`)
    - Combines the results from multiple flags into one single flag
    - Very similar to the calculation of the flux QCF flag

### Changes

- **MeteoScreening**: (`diive/pkgs/qaqc/meteoscreening.py`)
    - Refactored most of the code relating to the quality-screening of meteo data
    - Implemented the calculation of the overall quality flag QCF
    - Two overview figures are now created at the end on the screening
    - Flags for tests used during screening are now created using a base class (`core.base.flagbase.FlagBase`)
- **Flux Processing Chain**: All modules relating to the Swiss FluxNet flux processing
  chain are now collected in the dedicated package `fluxprocessingchain`. Relevant
  modules were moved to this package, some renamed:
    - `pkgs.fluxprocessingchain.level2_qualityflags.QualityFlagsLevel2`
    - `pkgs.fluxprocessingchain.level31_storagecorrection.StorageCorrectionSinglePoint`
    - `pkgs.fluxprocessingchain.qcf.QCF`
- **Reading YAML files**: (`core.io.filereader.ConfigFileReader`)
    - Only filetype configuration files are validated, i.e. checked if they follow the
      expected file structure. However, there can be other YAML files, such as the file
      `pipes_meteo.yaml` that defines the QA/QC steps for each meteo variable. For the
      moment, only the filetype files are validated and the validation is skipped for
      the pipes file.
- Refactored calculation of nighttime flag from sun altitude: code is now vectorized
  and runs - unsurprisingly - much faster (`pkgs.createvar.nighttime_latlon.nighttime_flag_from_latlon`)
- Some smaller changes relating to text output to the console

## v0.45.0 | 13 Jan 2023

### New Features

- **Flux storage correction**: (`pkgs.flux.storage.StorageCorrectionSinglePoint`)
    - Calculate storage-corrected fluxes
    - Creates Level-3.1 in the flux processing chain
- **Overall quality flag**: (`pkgs.qaqc.qcf.QCF`)
    - Calculate overall quality flag from multiple individual flags

### Changes

- **Flux quality-control**: (`pkgs.qaqc.fluxes.QualityFlagsLevel2`)
    - Flags now have the string `_L2_` in their name to identify them as
      flags created during Level-2 calculations in the Swiss FluxNet flux
      processing chain.
    - All flags can now be returned to the main data
- Renamed `pkgs.qaqc.fluxes.FluxQualityControlFlag` to `pkgs.qaqc.fluxes.QualityFlagsLevel2`

## v0.44.1 | 11 Jan 2023

### Changes

- **Flux quality-control**: (`pkgs.qaqc.fluxes.FluxQualityControlFlag`)
    - Added heatmap plots for before/after QC comparison
    - Improved code for calculation of overall flag `QCF`
    - Improved console output

## v0.44.0 | 9 Jan 2023

### New Features

- **Flux quality-control**: (`pkgs.qaqc.fluxes.FluxQualityControlFlag`)
    - First implementation of quality control of ecosystem fluxes. Generates one
      overall flag (`QCF`=quality control flag) from multiple quality test results
      in EddyPro's `fluxnet` output file. The resulting `QCF` is Level-2 in the
      Swiss FluxNet processing chain,
      described [here](https://www.swissfluxnet.ethz.ch/index.php/data/ecosystem-fluxes/flux-processing-chain/).
      `QCF` is mostly based on the ICOS methodology, described
      by [Sabbatini et al. (2018)](https://doi.org/10.1515/intag-2017-0043).
- **Histogram**: (`pkgs.analyses.histogram.Histogram`)
    - Calculates histogram from time series, identifies peak distribution
- **Percentiles**: (`pkgs.analyses.quantiles.percentiles`)
    - Calculates percentiles (0-100) for a time series
- **Scatter**: Implemented first version of `core.plotting.scatter.Scatter`, which will
  be used for scatter plots in the future

### Changes

- **Critical days**: (`pkgs.flux.criticaldays.CriticalDays`)
    - Renamed Variables, now using Dcrit (instead of CRD) and nDcrit (instead of nCRD)
- **NEP Penalty**: (`pkgs.flux.nep_penalty.NEPpenalty`)
    - Code was refactored to work with NEP (net ecosystem productivity) instead of NEE
      (net ecosystem exchange)
    - CO2 penalty was renamed to the more descriptive NEP penalty
- **Sanitize column names**: implemented in `core.io.filereader.ColumnNamesSanitizer`
  Column names are now checked for duplicates. Found duplicates are renamed by adding a
  suffix to the column name. Example: `co2_mean` and `co2_mean` are renamed to
  `co2_mean.1` and `co2_mean.2`. This check is now implemented during the reading of
  the data file in `core.io.filereader.DataFileReader`.
- **Configuration files**: When reading filetype configuration files in `core.io.filereader.ConfigFileReader`,
  the resulting dictionary that contains all configurations is now validated. The validation makes
  sure the parameters for `.read_csv()` are in the proper format.
- Updated all dependencies to their newest (possible) version

### Additions

- Added support for filetype `EDDYPRO_FLUXNET_30MIN` (`configs/filetypes/EDDYPRO_FLUXNET_30MIN.yml`)

## v0.43.0 | 8 Dec 2022

### New Features

- **Frequency groups detection**: Data in long-term datasets are often characterized by changing time
  resolutions at which data were recorded. `core.times.times.detect_freq_groups` detects changing
  time resolutions in datasets and adds a group identifier in a new column that gives info about the
  detected time resolution in seconds, e.g., `600` for 10MIN data records. This info allows to
  address and process the different time resolutions separately during later processing, which is
  needed e.g. during data quality-screening and resampling.
- **Outlier removal using z-score**: First version of `pkgs.outlierdetection.zscore.zscoreiqr`
  Removes outliers based on the z-score of interquartile range data. Data are divided
  into 8 groups based on quantiles. The z-score is calculated for each data point
  in the respective group and based on the mean and SD of the respective group.
  The z-score threshold to identify outlier data is calculated as the max of
  z-scores found in IQR data multiplied by *factor*. z-scores above the threshold
  are marked as outliers.
- **Outlier removal using local standard deviation**: First version of `pkgs.outlierdetection.local3sd.localsd`
  Calculates mean and SD in a rolling window and marks data points outside a specified range.

### Additions

- **MeteoScreening**: Added the new parameter `resampling_aggregation` in the meteoscreening setting
  `diive/pkgs/qaqc/pipes_meteo.yaml`. For example, `TA` needs `mean`, `PRECIP` needs `sum`.

### Changes

- **MeteoScreening**: `pkgs.qaqc.meteoscreening.MeteoScreeningFromDatabaseSingleVar`
  Refactored the merging of quality-controlled 30MIN data when more than one raw data time
  resolution is involved.
- **Resampling**: `core.times.resampling.resample_series_to_30MIN`
  The minimum required values for resampling is `1`. However, this is only relevant for
  lower resolution data e.g. 10MIN and 30MIN, because for higher resolutions the calculated value
  for minimum required values yields values > 1 anyway. In addition, if data are already in
  30MIN resolution, they are still going through the resampling processing although it would not
  be necessary, because the processing includes other steps relevant to all data resolutions, such
  as the change of the timestamp from TIMESTAMP_MIDDLE to TIMESTAMP_END.

### Bugs

- Removed display bug when showing data after high-res meteoscreening in heatmap. Plot showed
  original instead of meteoscreened data

## v0.42.0 | 27 Nov 2022

### New Features

- **Decoupling**: Added first version of decoupling code (`pkgs.analyses.decoupling.SortingBinsMethod`).
  This allows the investigation of binned aggregates of a variable `z` in binned classes of
  `x` and `y`. For example: show mean GPP (`y`) in 5 classes of VPD (`x`), separate for
  10 classes of air temperature (`z`).

![DIIVE](images/analysesDecoupling_sortingBinsMethod_diive_v0.42.0.png)

- **Time series plot**: `core.plotting.timeseries.TimeSeries` plots a simple time series. This will
  be the default method to plot time series.

### Changes

- **Critical days**: Several changes in `pkgs.flux.criticaldays.CriticalDays`:

    - By default, daily aggregates are now calculated from 00:00 to 00:00 (before it was
      7:00 to 07:00).
    - Added parameters for specifying the labels for the x- and y-axis in output figure
    - Added parameter for setting dpi of output figure
    - Some smaller adjustments
    - `pkgs.flux.co2penalty.CO2Penalty.plot_critical_hours`: 95% predicion bands are now
      smoothed (rolling mean)

- **CO2 penalty**: (since v0.44.0 renamed to NEP penalty)

    - Some code refactoring in `pkgs.flux.co2penalty.CO2Penalty`, e.g. relating to plot appearances

## v0.41.0 | 5 Oct 2022

### BinFitterBTS

- `pkgs.fits.binfitter.BinFitterBTS` fits a quadratic or linear equation to data.
- This is a refactored version of the previous `BinFitter` to allow more options.
- Implemented `pkgs.fits.binfitter.PlotBinFitterBTS` for plotting `BinFitterBTS` results
- `PlotBinFitterBTS` now allows plotting of confidence intervals for the upper and
  lower prediction bands
- The updated `BinFitterBTS` is now implemented in `pkgs.flux.criticaldays.CriticalDays`

#### Example of updated `BinFitterBTS` as used in `CriticalDays`

It is now possible to show confidence intervals for the upper and lower prediction bands.  
![DIIVE](images/fluxCriticalDaysWithUpdatedBinFitterBTS_diive_v0.41.0.png)

### Other

- `core.plotting.heatmap_datetime.HeatmapDateTime` now accepts `figsize`
- When reading a file using `core.io.filereader.ReadFileType`, the index column is now
  parsed to a temporarily named column. After reading the file data, the temporary column
  name is renamed to the correct name. This was implemented to avoid duplicate issues
  regarding the index column when parsing the file, because a data column with the same
  name as the index column might be in the dataset.

### Bugfixes

- Fixed bug in `pkgs.gapfilling.randomforest_ts.RandomForestTS`: fallback option for
  gap-filling was never used and some gaps would remain in the time series.

## v0.40.0 | 23 Sep 2022

### CO2 Penalty

- New analysis: `pkgs.flux.co2penalty.CO2Penalty` calculates the CO2 penalty as
  the difference between the observed co2 flux and the potential co2 flux modelled
  from less extreme environmental conditions.

![DIIVE](images/fluxCO2penalty_cumulative_diive_v0.40.0.png)

![DIIVE](images/fluxCO2penalty_penaltyPerYear_diive_v0.40.0.png)

![DIIVE](images/fluxCO2penalty_dielCycles_diive_v0.40.0.png)

### VPD Calculation

- New calculation: `pkgs.createvar.vpd.calc_vpd_from_ta_rh` calculates vapor pressure
  deficit (VPD) from air temperature and relative humidity

### Fixes

- Fixed: `core.plotting.cumulative.CumulativeYear` now shows zero line if needed
- Fixed: `core.plotting.cumulative.CumulativeYear` now shows proper axis labels

## v0.39.0 | 4 Sep 2022

### Critical Days

- New analysis: `pkgs.flux.criticaldays.CriticalDays` detects days in y that are
  above a detected x threshold. At the moment, this is implemented to work with
  half-hourly flux data as input and was tested with VPD (x) and NEE (y). In the
  example below critical days are defined as the VPD daily max value where the daily
  sum of NEE (in g CO2 m-2 d-1) becomes positive (i.e., emission of CO2 from the
  ecosystem to the atmosphere).
  ![DIIVE](images/fluxCriticalDays_diive_v0.39.0.png)

## v0.38.0 | 3 Sep 2022

### Optimum Range Detection

- New analysis: `pkgs.analyses.optimumrange.FindOptimumRange` finds the optimum for a
  variable in binned other variable. This is useful for e.g. detecting the VPD
  range where CO2 uptake was highest (=most negative).  
  ![DIIVE](images/analysesOptimumRange_diive_v0.38.0.png)

## v0.37.0 | 2 Sep 2022

### Cumulative and Anomaly Plots

- New plot: `core.plotting.cumulative.CumulativeYear` plots cumulative sums per year  
  ![DIIVE](images/plotCumulativeYear_diive_v0.37.0.png)
- New plot: `core.plotting.bar.LongtermAnomaliesYear` plots yearly anomalies in relation to a reference period  
  ![DIIVE](images/plotBarLongtermAnomaliesYear_diive_v0.37.0.png)
- Refactored various code bits for plotting

## v0.36.0 | 27 Aug 2022

### Random Forest Update

- Refactored code for `pkgs/gapfilling/randomforest_ts.py`
    - Implemented lagged variants of variables
    - Implemented long-term gap-filling, where the model to gap-fill a specific year is built from the
      respective year and its neighboring years
    - Implemented feature reduction using sklearn's RFECV
    - Implemented TimeSeriesSplit used as the cross-validation splitting strategy during feature reduction
- Implemented `TimestampSanitizer` also when reading from file with `core.io.filereader.DataFileReader`
- Removed old code in `.core.dfun.files` and moved files logistics to `.core.io.files` instead
- Implemented saving and loading Python `pickles` in `.core.io.files`

## v0.35.0 | 19 Aug 2022

### Meteoscreening PA, RH

- Added function `pkgs.corrections.offsetcorrection.remove_relativehumidity_offset` to correct
  humidity measurements for values > 100%

### Other

- Added first code for outlier detection via seasonal trends in `pkgs/outlierdetection/seasonaltrend.py`
- Prepared `pkgs/analyses/optimumrange.py` for future updates

## v0.34.0 | 29 Jul 2022

### MeteoScreening Radiation

#### MeteoScreening

- Implemented corrections and quality screening for radiation data in `pkgs.qaqc.meteoscreening`

#### Corrections

Additions to `pkgs.corrections`:

- Added function `.offsetcorrection.remove_radiation_zero_offset` to correct radiation
  data for nighttime offsets
- Added function `.setto_threshold.setto_threshold` to set values above or below a
  specfied threshold value to the threshold.

#### Plotting

- Added function `core.plotting.plotfuncs.quickplot` for quickly plotting pandas
  Series and DataFrame data

#### Resampling

- Implemented `TimeSanitizer` in `core.times.resampling.resample_series_to_30MIN`

#### Other

- Added decorator class `core.utils.prints.ConsoleOutputDecorator`, a wrapper to
  execute functions with additional info that is output to the console.

## v0.33.0 | 26 Jul 2022

### MeteoScreening Preparations

- Added new class `core.times.times.TimestampSanitizer`
    - Class that handles timestamp checks and fixes, such as the creation of a continuous
      timestamp without date gaps.
- Added `pkgs.createvar.nighttime_latlon.nighttime_flag_from_latlon`
    - Function for the calculation of a nighttime flag (1=nighttime) from latitude and
      longitude coordinates of a specific location.
- Added `core.plotting.heatmap_datetime.HeatmapDateTime`
    - Class to generate a heatmap plot from timeseries data.

## v0.32.0 | 22 Jul 2022

### MeteoScreening Air Temperature

MeteoScreening uses a general settings file `pipes_meteo.yaml` that contains info how
specific `measurements` should be screened. Such `measurements` group similar variables
together, e.g. different air temperatures are measurement `TA`.   
Additions to module `pkgs.qaqc.meteoscreening`:

- Added class `ScreenVar`
    - Performs quality screening of air temperature `TA`.
    - As first check, I implemented outlier detection via the newly added package `ThymeBoost`,
      along with checks for absolute limits.
    - Screening applies the checks defined in the file `pipes_meteo.yaml` for the respective
      `measurement`, e.g. `TA` for air temperature.
    - The screening outputs a separate dataframe that contains `QCF` flags for each check.
    - The checks do not change the original time series. Instead, only the flags are generated.
    - Screening routines for more variables will be added over the next updates.
- Added class `MeteoScreeningFromDatabaseSingleVar`
    - Performs quality screening *and* resampling to 30MIN of variables downloaded from the database.
    - It uses the `detailed` data when downloading data from the database using `dbc-influxdb`.
    - The `detailed` data contains the measurement of the variable, along with multiple tags that
      describe the data. The tags are needed for storage in the database.
    - After quality screening of the original high-resolution data, flagged values are removed and
      then data are resampled.
    - It also handles the issue that data downloaded for a specific variable can have different time
      resolution over the years, although I still need to test this.
    - After screening and resampling, data are in a format that can be directly uploaded to the
      database using `dbc-influxdb`.
- Added class `MeteoScreeningFromDatabaseMultipleVars`
    - Wrapper where multiple variables can be screened in one run.
    - This should also work in combination of different `measurements`. For example, screening
      radiation and temperature data in one run.

### Outlier Detection

Additions to `pkgs.outlierdetection`:

- Added module `thymeboost`
- Added module `absolute_limits`

[//]: # (- optimum range)

[//]: # (- `diive.core.times` `DetectFrequency` )

[//]: # (- `diive.core.times`: `resampling` module )

[//]: # (- New package in env: `ThymeBoost` [GitHub]&#40;https://github.com/tblume1992/ThymeBoost/tree/main/ThymeBoost&#41; )

## v0.31.0 | 4 Apr 2022

### Carbon cost

#### **GENERAL**

- This version introduces the code for calculating carbon cost and critical heat days.

#### **NEW PACKAGES**

- Added new package for flux-specific calculations: `diive.pkgs.flux`

#### **NEW MODULES**

- Added new module for calculating carbon cost: `diive.pkgs.flux.carboncost`
- Added new module for calculating critical heat days: `diive.pkgs.flux.criticalheatdays`

#### **CHANGES & ADDITIONS**

- None

#### **BUGFIXES**

- None

## v0.30.0 | 15 Feb 2022

### Starting diive library

#### **GENERAL**

The `diive` library contains packages and modules that aim to facilitate working
with time series data, in particular ecosystem data.

Previous versions of `diive` included a GUI. The GUI component will from now on
be developed separately as `diive-gui`, which makes use of the `diive` library.

Previous versions of `diive` (up to v0.22.0) can be found in the separate repo
[diive-legacy](https://gitlab.ethz.ch/diive/diive-legacy).

This initial version of the `diive` library contains several first versions of
packages that will be extended with the next versions.

Notable introduction in this version is the package `echires` for working with
high-resolution eddy covariance data. This package contains the module `fluxdetectionlimit`,
which allows the calculation of the flux detection limit following Langford et al. (2015).

#### **NEW PACKAGES**

- Added `common`: Common functionality, e.g. reading data files
- Added `pkgs > analyses`: General analyses
- Added `pkgs > corrections`: Calculate corrections for existing variables
- Added `pkgs > createflag`: Create flag variables, e.g. for quality checks
- Added `pkgs > createvar`: Calculate new variables, e.g. potential radiation
- Added `pkgs > echires`: Calculations for eddy covariance high-resolution data, e.g. 20Hz data
- Added `pkgs > gapfilling`: Gap-filling routines
- Added `pkgs > outlierdetection`: Outlier detection
- Added `pkgs > qaqc`: Quality screening for timeseries variables

#### **NEW MODULES**

- Added `optimumrange` in `pkgs > analyses`
- Added `gapfinder` in `pkgs > analyses`
- Added `offsetcorrection` in `pkgs > corrections`
- Added `setto_threshold` in `pkgs > corrections`
- Added `outsiderange` in `pkgs > createflag`
- Added `potentialradiation` in `pkgs > createvar`
- Added `fluxdetectionlimit` in `pkgs > echires`
- Added `interpolate` in `pkgs > gapfilling`
- Added `hampel` in `pkgs > outlierdetection`
- Added `meteoscreening` in `pkgs > qaqc`

#### **CHANGES & ADDITIONS**

- None

#### **BUGFIXES**

- None

## **REFERENCES**

- Hollinger, D. Y., & Richardson, A. D. (2005). Uncertainty in eddy covariance measurements
  and its application to physiological models. Tree Physiology, 25(7),
  873–885. https://doi.org/10.1093/treephys/25.7.873
- Langford, B., Acton, W., Ammann, C., Valach, A., & Nemitz, E. (2015). Eddy-covariance data with low signal-to-noise
  ratio: Time-lag determination, uncertainties and limit of detection. Atmospheric Measurement Techniques, 8(10),
  4197–4213. https://doi.org/10.5194/amt-8-4197-2015
- Papale, D., Reichstein, M., Aubinet, M., Canfora, E., Bernhofer, C., Kutsch, W., Longdoz, B., Rambal, S., Valentini,
  R., Vesala, T., & Yakir, D. (2006). Towards a standardized processing of Net Ecosystem Exchange measured with eddy
  covariance technique: Algorithms and uncertainty estimation. Biogeosciences, 3(4),
  571–583. https://doi.org/10.5194/bg-3-571-2006
- Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux processing pipeline
  for eddy covariance data. 27. https://doi.org/10.1038/s41597-020-0534-3
- Reichstein, M., Falge, E., Baldocchi, D., Papale, D., Aubinet, M., Berbigier, P., Bernhofer, C., Buchmann, N.,
  Gilmanov, T., Granier, A., Grunwald, T., Havrankova, K., Ilvesniemi, H., Janous, D., Knohl, A., Laurila, T., Lohila,
  A., Loustau, D., Matteucci, G., … Valentini, R. (2005). On the separation of net ecosystem exchange into assimilation
  and ecosystem respiration: Review and improved algorithm. Global Change Biology, 11(9),
  1424–1439. https://doi.org/10.1111/j.1365-2486.2005.001002.x
- Vekuri, H., Tuovinen, J.-P., Kulmala, L., Papale, D., Kolari, P., Aurela, M., Laurila, T., Liski, J., & Lohila, A. (
  2023). A widely-used eddy covariance gap-filling method creates systematic bias in carbon balance estimates.
  Scientific Reports, 13(1), 1720. https://doi.org/10.1038/s41598-023-28827-2
