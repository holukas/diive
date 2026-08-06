# Example & Test Coverage Gaps

Survey date: 2026-07-26 · diive v0.91.0 · branch `indev`

> **Since this survey:** `flux/hires` moved to [dyco](https://github.com/holukas/dyco), taking
> ~10 000 LOC and `tests/test_echires.py`'s 74 tests with it. Its entries are removed below; the
> measured percentages and the 460-test run still include it and were not recomputed.

Working document. Goal state: every important public function has at least one example and one test.

## How this was measured

Two independent passes:

1. **Line coverage.** `pytest-cov` added to the `dev` group; full suite run with per-test contexts:
   `uv run pytest tests/ -q --cov=diive --cov-context=test`.
   All optional extras were installed, so nothing was skipped: **460 passed, 12 subtests, 16m 06s.**
   Contexts let us ask not just *"is this line covered?"* but *"which test covered it?"*
2. **Symbol cross-reference.** Word-boundary grep of the public API against `examples/**/*.py`.
   Coverage cannot answer the example question — examples are not part of the suite — so this pass
   stays grep-based.

What coverage settles, and what it does not:

- It **settles** whether a line executes, so it resolves every "is this indirectly covered?"
  question the grep pass had to hedge on. Several of those hedges turned out to be wrong in both
  directions; corrections are marked **[revised]** below.
- It does **not** measure assertion quality. A line executed inside a test with no assertion about
  its result still counts as covered. Treat high percentages on the plotting modules with
  particular suspicion: a `.plot()` call that raises no exception covers most of a module.

---

## Headline numbers

Re-measured 2026-07-26 after this session's test work (**665 tests**, up from 460; 29 min).

| Scope | Baseline | Now |
|---|---|---|
| Library (`diive/`, excluding `diive/gui/`) | 57 % | **61 %** (14 789 / 24 414) |
| GUI (`diive/gui/`) | 68 % | 67 % |
| Combined | 62 % | **64 %** |
| Library files at 0 % | 13 | **12** |

### The structural finding: the GUI test suite is carrying the library

Baseline: **5 114 of the library's covered lines — 37 % — were reachable only through
`tests/test_gui.py`.** Now **3 995 (27 %)**.

> Delete `tests/test_gui.py` and library coverage falls from 61 % to **44 %** — it was 36 %.

That 36 % → 44 % figure is the honest measure of this session's work: the headline library number
moved 4 points, but coverage that does not depend on a GUI smoke path moved **8**. Roughly 1 100
lines were converted from incidentally-executed to actually-asserted.

Still fully GUI-dependent (≥25 statements, every covered line from `test_gui.py`):

| Covered / total | Module |
|---|---|
| 69 / 667 | `flux/lowres/selfheating.py` |
| 36 / 36 | `flux/__init__.py` |
| 33 / 37 | `preprocessing/outlier_detection/manualremoval.py` |
| 27 / 222 | `analysis/optimumrange.py` |
| 25 / 302 | `core/ml/optimization.py` |
| 25 / 79 | `core/io/db/influxdb.py` |
| 23 / 211 | `flux/lowres/timelag_analysis.py` |
| 19 / 193 | `flux/lowres/ustar_vekuri_detection.py` |
| 10 / 171 | `flux/lowres/hqflux.py` |
| 10 / 53 | `analysis/granger.py` |

The whole plotting cluster is gone from that list. Largest remaining GUI-only line counts:
`flux/fluxprocessingchain/container.py` (235 of 269 covered lines), `analysis/gapfinder.py` (124),
`flux/lowres/ustarthreshold.py` (92).

### Runtime — investigated, no regression

The re-run clocked 29 min against the baseline's 16 min. Investigated; **it does not reproduce.**

| Segment | Measured |
|---|---|
| Whole suite except `test_gui.py` (with coverage) | **402 s** |
| `test_gui.py` alone (with coverage) | **676 s** |
| Steady-state total | **~1 078 s (18 min)** |

`test_gui.py` measured four ways came out 580 s / 715 s / 1 337 s / 676 s — three clustered, one
outlier, and the outlier is the 29-minute run. The six test files added this session cost **16 s**
under coverage, so they account for essentially none of it. Baseline 966 s → steady state 1 078 s
is the +16 s plus noise. Treat single-sample wall-clock comparisons on this suite as unreliable;
`test_gui.py` alone varies by more than 2x.

**The real cost centre — found and fixed.** `tests/test_driveranalysis.py::TestDriverAnalysisTemporal`
had a `setUpClass` costing **220 s**, 55 % of the entire non-GUI suite. Its fixture is now
`months=2` instead of `months=4`:

| | Before | After |
|---|---|---|
| `TestDriverAnalysisTemporal` setup | 220 s | **27.6 s** |
| `tests/test_driveranalysis.py` (whole file) | ~230 s | **32.8 s** |
| Non-GUI suite | 402 s | **199 s** |

Row count, not lag span, is what costs: every temporal stage ends in a TreeSHAP pass over the full
matrix, and more rows also deepen the forest (`min_samples_leaf=3`). Measured across four
configurations — **2x the data costs 7.5x the time**, while widening the lags from `-2..0` to
`-6..0` costs 2 s. So the lag span stays wide (it exercises `lagged_variants` and the
`(driver, lag)` attribution) and only the fixture shrinks. All five assertion groups pass at the
smaller size, and the timing is stable across three repeat runs (32.74 / 32.95 / 32.73 s).

**The trade, stated honestly:** `months=2` covers 540/887 driveranalysis statements against
549/887 at `months=4` — **10 lines lost**. All ten are convergence/verdict branches
(`agreement='diverge'`, `'partial'`, `'spurious_correlate'`, `shap_high_ale_flat`,
`sign_flip_by_regime` and its `_regime_dir_conflict` helper). **No test asserts on any of them** —
they fired only because the noise at `months=4` happened to land that way. That is accidental
coverage, not verified behaviour, and paying 180 s per suite run for it is a bad trade. Those
branches deserve dedicated tests with inputs crafted to trigger them deterministically; see the
follow-up below.

Next slowest after the fix: the daytime ONEFlux / REddyProc partitioning tests at 12-21 s each,
which is expected for faithful ports.

### Follow-up: DriverAnalysis verdict branches

Ten branches in `analysis/driveranalysis/driveranalysis.py` (lines ~881, 975, 981, 986, 991-992,
1011-1015, 1041) decide the convergence verdict and are reached by no assertion. They need a test
that constructs driver/target relationships where SHAP and ALE deliberately disagree, so
`diverge` / `partial` / `spurious_correlate` / `sign_flip_by_regime` each fire on purpose.

This is worth treating as its own problem, separate from the raw gaps below. Those 5 114 lines are
covered *incidentally*: a GUI test drives a widget, the widget calls the library, the lines execute,
and nothing asserts anything about the library-level result. It also means library regressions
surface as GUI test failures, which is the least informative place for them to surface.

Library modules where **every** covered line is GUI-test-only (≥25 statements):

| Covered / total stmts | Module |
|---|---|
| 162 / 179 | `core/plotting/ridgeline.py` |
| 145 / 154 | `core/plotting/codegen.py` |
| 139 / 152 | `core/plotting/heatmap_datetime.py` |
| 130 / 142 | `core/plotting/cumulative.py` |
| 106 / 223 | `core/plotting/treering.py` |
| 89 / 96 | `core/plotting/shifted_distribution.py` |
| 69 / 667 | `flux/lowres/selfheating.py` |
| 51 / 57 | `core/plotting/bar.py` |
| 45 / 53 | `core/plotting/waterfall.py` |
| 39 / 40 | `preprocessing/qaqc/measurements.py` |
| 38 / 70 | `preprocessing/outlier_detection/codegen.py` |
| 33 / 37 | `preprocessing/outlier_detection/manualremoval.py` |
| 32 / 33 | `variables/classification.py` |
| 27 / 222 | `analysis/optimumrange.py` |
| 25 / 302 | `core/ml/optimization.py` |
| 25 / 79 | `core/io/db/influxdb.py` |
| 23 / 211 | `flux/lowres/timelag_analysis.py` |
| 19 / 193 | `flux/lowres/ustar_vekuri_detection.py` |
| 15 / 43 | `gapfilling/codegen.py` |
| 14 / 29 | `core/plotting/surface_grid.py` |

The plotting cluster was the clearest case, and is now **fixed** (2026-07-26). `tests/test_plots.py`
covered exactly five classes — `HistogramPlot`, `ScatterXY`, `TimeSeries`, `WindRosePlot`,
`CompoundExtremesPlot` — and never touched `HeatmapDateTime` (used by 16 of the 122 examples),
`Cumulative`, `RidgeLinePlot`, `TreeRingPlot`, `ShiftedDistributionPlot`, `WaterfallPlot`, or
`bar.py`. A `TestPlotClasses` class now covers all of them; see
[Tier 1b](#tier-1b--plot-classes--done-2026-07-26).

### Library files at 0 %

Baseline list below; **`diive/corrections/__init__.py` is now covered**, leaving 12 files / 1 293
statements never executed by any test. `core/plotting/seasonaltrend.py` (149) is the one this
session touched adjacent work on without closing — its four functions take prepared inputs
(a decomposition object, a harmonics list, frequency/power arrays) rather than a series.

| Stmts | File | Example? |
|---|---|---|
| ~~281~~ | ~~`preprocessing/qaqc/detect_timestamp_shifts.py`~~ — **now 92 %** | yes |
| 205 | `io/formats/fluxnet.py` | — |
| 167 | `io/formats/meteo.py` | — |
| 149 | `core/plotting/seasonaltrend.py` | yes (`LongtermAnomaliesYear`) |
| 49 | `core/utils/vargroups.py` | — |
| 40 | `core/plotting/_fitplot.py` | yes (via `Fitter`) |
| 16 | `core/dfun/regression.py` | — |
| 11 | **`diive/corrections/__init__.py`** | — |
| 5 | `core/plotting/rectangle.py` | — |
| 4 | `core/io/dirs.py` | — |
| 3 | `io/formats/__init__.py` | — |

`diive/corrections/__init__.py` at 0 % is a finding in itself: **no test ever imports the
`dv.corrections` namespace.** Tests reach corrections via `diive.preprocessing.corrections`
instead, so the public namespace's re-export list is unverified — a symbol could be dropped from
`__all__` and every test would still pass.

---

## Tier 1 — Codegen — **DONE** (2026-07-26)

Closed by `tests/test_codegen.py`: 67 tests, ~3 s, no GUI or data. All 55 generators are now
covered, and a completeness test fails if a new `*_to_code` lands without one.

Coverage of the codegen modules, from `test_codegen.py` + `test_flux_codegen.py` alone (i.e. with
no GUI-test contribution at all):

| Module | Before (GUI-only share) | After, without any GUI test |
|---|---|---|
| `preprocessing/outlier_detection/codegen.py` | 54 % (100 % GUI-only) | **100 %** |
| `core/plotting/codegen.py` | 94 % (100 % GUI-only) | **99 %** |
| `flux/fluxprocessingchain/codegen.py` | 80 % | **94 %** |
| `gapfilling/codegen.py` | 35 % (100 % GUI-only) | **93 %** |
| `flux/lowres/codegen.py` | 85 % (~100 % GUI-only) | **92 %** |
| `flux/partitioning/codegen.py` | 12 % (100 % GUI-only) | **92 %** |
| `preprocessing/corrections/codegen.py` | 62 % | 54 % |

Two notes for whoever extends this:

- The signature check (`assertKwargsAccepted`) is the part with teeth. It was mutation-tested: a
  typo'd `plot()` keyword, a typo'd constructor keyword, and a removed parameter name are all
  caught; valid snippets pass. The first version of it silently skipped every outlier-detector
  constructor, because `inspect.signature` sees `**legacy` and reports "accepts anything" — but
  `reject_legacy_params` raises on unknown names, so the named parameters are the real accepted
  set. That carve-out is `_REJECTING_VAR_KEYWORD` in the test module.
- All 55 generators passed the signature check on the first run, so no library drift existed at the
  time of writing. The value is forward-looking.

### Original finding, kept for context

## Tier 1j — `DetectTimestampShifts` — **DONE** (2026-07-26)

`tests/test_timestamp_shifts.py` (new): 21 tests + 1 expected failure, 3.3 s.
**`preprocessing/qaqc/detect_timestamp_shifts.py`: 0 % → 92 %** — the largest library file that no
test executed at all.

The module recovers a clock offset by comparing measured against potential radiation, so the tests
**plant a known shift** in noise-free synthetic radiation and check each method recovers it. That
validates the algorithms rather than their plumbing, and it immediately found a defect.

| Method | Planted 60 min late | Before | After |
|---|---|---|---|
| `fft_phase_shift` | -60.0 | correct | correct |
| `noon_shift` | -60.0 | correct (resolution = one record) | correct |
| `crosscorr` | -60.0 | **0 — broken** | **fixed** |

### Found and fixed: `crosscorr` could not recover a shift

Documented as the "high-precision 1-minute lag search" and promoted in the example, it returned 0
for a clean 60-minute offset and -54 for a 120-minute one. A brute-force Pearson scan over the same
day found the true lag at **r = 1.0000**, which is what proved the signal was there and the method
was losing it.

Two causes. The daytime mask (`sun_up = ts_pot_hr > 10`, derived from *potential*) clipped both
series before correlating, truncating the shifted measured curve; and `sp_correlate` output was not
normalised per lag by the overlap count, so lags with less overlap scored lower and the argmax was
pulled toward zero. Normalising alone only moved the answer from 0 to -5, which identified the mask
as the primary defect.

The fix pads the daytime window by `max_shift_min` on each side and replaces the FFT correlation
with a direct Pearson lag scan — what the docstring always described.

| | Before | After |
|---|---|---|
| Planted 60 min | 0 | **-60.0** |
| Perfect-match correlation | 0.913 | **1.000** |
| CH-DAV 2022 median | ~0 (disagreed with FFT by ~6 min) | **-5.0 min** (FFT: -5.9) |
| One year | 0.31 s | 0.78 s |

The 0.913 mattered: `plot_crosscorr_results` filters at `min_corr=0.97` by default, so even
perfectly aligned days were being hidden. `max_shift_min` and the result are now converted through
the upsampled step, so a non-default `upsample_freq` no longer reinterprets minutes as samples.

A first cut using `np.corrcoef` per lag cost 5.0 s/year; computing Pearson directly from dot
products brought that to 0.78 s with identical output.

**Fixture note:** the algorithm tests are noise-free so recovery is exact, but `TestPlots` adds 3 %
noise. With perfectly uniform data every day yields the identical shift, and `plot_fft_results`
then asks numpy for a 50-bin histogram of a zero-range series, which raises "Too many bins for data
range". Real records always have spread, so this is a fixture artifact — but it is a genuine edge
case for a perfectly constant offset.

## Tier 1i — `FlagQCF` — **DONE** (2026-07-26)

Four new classes in `tests/test_qaqc.py` (21 tests total in the file, 2.1 s).
**`preprocessing/qaqc/qcf.py`: 53 % → 95 %** from this file alone, 16 statements missing.

The three existing tests covered aggregation, the filtered series and the OVERALL screening report.
Everything else was untested — including the entire `swinpot_col` day/night path, all three console
reports and both plot methods.

| Added | Covers |
|---|---|
| `TestFlagQCFRules` | The `> 3 soft flags` boundary the old tests jumped over (2 soft, then 4). Exactly 3 must still be QCF 1. Also the flag sums, which are summed by *value* — two hard flags sum to 4, not 2 |
| `TestFlagQCFDayNight` | The `swinpot_col` path: `daytime_accept_qcf_below=1` must promote marginal **daytime** records to QCF 2 while nighttime QCF 1 survives; equal thresholds leave both alone; the screening report gains DAYTIME/NIGHTTIME periods that partition the OVERALL count |
| `TestFlagQCFReports` | `report_qcf_flags` / `report_qcf_evolution` / `report_qcf_series`, via a console sink (`add_console_sink`) — all three previously at 0 % |
| `TestFlagQCFPlots` | `showplot_qcf_heatmaps` / `showplot_qcf_timeseries` produce a figure |
| `TestFlagQCFValidation` | `KeyError` for a missing `target_col` or `swinpot_col` |

Mutation-tested by breaking three rules in the source in turn (soft threshold `>3` → `>4`, daytime
threshold ignored, hard-flag rule weakened) — each is caught.

### Two bugs found while writing these — spawned as separate tasks

1. ~~**`report_qcf_flags()` crashes on a cp1252 stdout.**~~ — **fixed** (2026-07-26). An AST scan
   for literals passed to console emitters found the real scope to be far smaller than the raw
   character count suggested: of 28 library files containing non-cp1252 characters, only **two**
   had them in printed strings (`qcf.py`, 18 literals; `gapfilling/interpolate.py`, 2). The rest
   are docstrings, comments, matplotlib axis labels, or Textual TUI widgets — none of which touch
   stdout. Three further stdout paths the AST scan could not see were found by reading:
   `detect_and_remove_tlag.py`'s argparse `description` (printed by `--help`), an `out()` progress
   line, and Greek letters assembled into a variable before `console.log`. All fixed;
   `tests/test_console.py::TestConsoleStringsAreCp1252Safe` guards against regression and
   documents its own blind spots.
2. ~~**`FlagQCF` without `idstr` names its columns `FLAGNone_FC_QCF`.**~~ — **fixed** (2026-07-26).
   `validate_id_string` now normalises a falsy idstr to `''` rather than passing `None` through.
   Surveying its six callers first showed the fix belongs there, not in `FlagQCF`: `eddyproflags`,
   `storage_correction` and `quality_flags` interpolate the value unguarded and carried the same
   latent bug, while `FlagBase` and the USTAR flaggers branch on `if idstr:` and are unaffected
   because `''` is falsy too. Every production call site passes an explicit idstr (`L2`, `L3.1`,
   `STEPWISE`, `METSCR`), so no shipped path changed. Pinned by `TestValidateIdString` and
   `TestFlagQCFColumnNames` in `tests/test_qaqc.py`, including an assertion that no output column
   contains a literal "None"; mutation-tested by restoring the old return.

## Tier 1h — `stl_decompose` regression test — **DONE** (2026-07-26)

`TestStlDecompose` in `tests/test_time.py`: 8 tests, 1.2 s. **`stl_decompose` itself: 84 %.**

Two real bugs were fixed in `core/times/decomposition_utils.py::stl_decompose` with no test left
behind. Both are now pinned, and **both were mutation-tested by restoring the original bug in the
source** (with a guaranteed restore) and confirming the test fails:

| Bug | How the test catches it |
|---|---|
| `seasonal` never reached statsmodels as `period` — the caller's cycle length was ignored | A known 24-step cycle must come back with lag-24 autocorrelation > 0.99 in the seasonal component. Correct period gives **0.9999**, a wrong period **0.005**, so the two are cleanly separable. A control test asserts the wrong period genuinely fails to recover the cycle, so the first assertion is not vacuous |
| `STL.fit(weights=...)` — statsmodels accepts no observation weights, so any `weights=` call raised | Passing weights must not raise and must echo them back. Restoring the bug reproduces the historical `TypeError: fit() got an unexpected keyword argument 'weights'` |

Also covered: additive reconstruction (the three components must sum back to the input, which
catches the internal integer-index swap failing to restore the original index), trend-window
normalisation (statsmodels needs an odd window strictly greater than the period; the wrapper fixes
up both instead of passing them through to a raise), argument validation, and the short-series
warning.

**The module is still low overall** (28 % from this file alone) because `stl_decompose` is one of
six functions in it and the other five have no test at all:

| Function | Covered |
|---|---|
| `stl_decompose` | **84 %** |
| `classical_decompose` | 8 % |
| `quality_weighted_decompose` | 7 % |
| `reconstruct_from_components` | 6 % |
| `harmonic_decompose` | 3 % |
| `detect_seasonality` | 2 % |

## Tier 1g — Flux-chain re-run cascade — **DONE** (2026-07-26)

`TestRerunCascade` in `tests/test_fluxprocessingchain.py`: 10 tests, 3.4 s. The chain is built once
in `setUpClass` (0.6 s) and every test re-runs from those snapshots, which the levels' purity makes
safe.

`levels/_rerun.py`: 87 % with coverage arriving only incidentally from the GUI driving levels
repeatedly → **98 % from real tests** (this class alone).

Covered:

- **The cascade itself** — re-running L2 on a fully-built chain returns `level_ids == ['L2']`, an
  `fpc_df` with the same column count as a fresh first L2 run (not doubled), zero duplicate labels,
  and every downstream `LevelResults` field cleared. Re-running L3.1 keeps L2 and clears below it.
- **Purity** — the input container is unchanged by a re-run.
- **`filteredseries` fallback** — it always belongs to the most recently completed level, so a
  cascade must move it back to the newest survivor: from L3.2 → `NEE_L3.1_QCF`, from L3.3 →
  `NEE_L3.1_L3.2_QCF`, from L2 → `None`.
- **Additive-level clearing** — L4.1 / L4.2 do not cascade among themselves, but a cascade from any
  earlier level must clear them, because their output was computed against now-stale upstream.
- **`drop_columns_for_key`** — the per-method cleanup that keeps L4.1 additive: dropping the MDS
  columns must leave the random-forest columns alone. No-op for an unknown key.
- **`record_added_columns`** — new columns attributed to the running level, existing entries
  carried through.

Mutation-tested: neutering `cascade_reset` to an identity function makes the L2 re-run test fail.

**Shape note for whoever extends this:** cleared fields reset to a *type-appropriate* default, not
uniformly `None` — `level33_qcf` is a `dict[str, FlagQCF]` (L3.3 keys its QCF by USTAR scenario) and
resets to `{}`. Asserting `None` across the board fails.

## Tier 1f — `GapFillingResult` / `prediction_scores` — **DONE** (2026-07-26)

Added to `tests/test_gapfilling.py`: `TestPredictionScores` (6 tests) and `TestGapFillingResult`
(7 tests), plus the MDS half of the contract folded into the existing `test_fluxmds` run. 2.3 s.

| Module | Before | After |
|---|---|---|
| `core/ml/results.py` | 100 %, **all 16 lines GUI-only** | **100 %**, real tests |
| `core/ml/scores.py` | untested | **100 %** |

`GapFillingResult` is the documented return type of `.results` on every gap-filler, and no non-GUI
test had ever constructed or inspected one. What is pinned now:

- **The ML contract** — `gapfilled` has no NaN and keeps the input index; `flag` ⊆ {0, 1, 2} with
  the expected counts; **observed records come back untouched** (filling must never overwrite
  measured data); `scores` and `scores_traintest` both carry all seven metrics; `model` and
  `feature_importances` are populated.
- **The reduction fields** — `feature_importances_reduction` / `accepted_features` /
  `rejected_features` are `None` unless `reduce_features` ran, and populated (including the
  `.RANDOM` benchmark row) when it did.
- **The MDS contract** — same dataclass, but `scores_traintest`, `feature_importances`, `model`
  and the reduction fields are all `None`, because MDS is not a regressor. Asserted on the
  existing `test_fluxmds` run rather than paying for a second one.
- **`prediction_scores` argument order.** `r2` and `mape` are asymmetric in (true, predicted), so
  a swapped internal call changes them while `mae`/`rmse`/`maxe`/`medae` stay identical. The test
  pins the asymmetric pair to exact values; mutation-tested by swapping the internal order, which
  the test catches.

**Gotcha worth knowing:** `trainmodel()` and `fillgaps()` both default `showplot_scores=True` and
`showplot_importance=True`, so a bare `model.run()` **blocks on a plot window** outside a headless
backend. It cost a couple of minutes of debugging here (the run looked like an infinite hang; with
the plots off it takes 0.02 s). Always pass `showplot_scores=False, showplot_importance=False`
from a test or script.

## Tier 1e — Namespace export surface — **DONE** (2026-07-26)

`tests/test_imports.py` rewritten: 10 tests, **515 subtests**, ~2.5 s. Covers every symbol of all
ten namespaces, replacing the three per-file `__all__` tests written earlier (those were strict
duplicates; the implementation-object check in `tests/test_corrections.py` is kept, since the
generic test cannot know each namespace's backing module).

Driven off `diive.__init__._LAZY_SUBMODULES` rather than a hard-coded list, so a new namespace is
covered the moment it is registered — and the registration itself is checked. CLAUDE.md documents
that a namespace must be added in **four** places at once; the test now enforces all four:

| Place | Failure mode if missed |
|---|---|
| `_LAZY_SUBMODULES` | the namespace is not reachable as `dv.<name>` at all |
| the `TYPE_CHECKING` block | silent — only a stale IDE / type-checker view |
| `diive.__all__` | absent from the documented public surface |
| `packaging/diive_gui.spec` `hiddenimports` | **fails only in the frozen GUI build** — PyInstaller cannot follow a PEP 562 `__getattr__` |

The last one is the reason this is worth having: an unlisted namespace passes every test and every
dev run, then is simply missing from the packaged app.

Mutation-tested — all four guards fire: a bogus name in an `__all__`, a namespace package on disk
that is not in `_LAZY_SUBMODULES`, and a registered namespace absent from either the
`TYPE_CHECKING` block or the PyInstaller spec.

Also checked per namespace: `__all__` exists and is non-empty, has no duplicates, exports no
underscore-private names, and every symbol is the same object through the module and through `dv`.

## Tier 1d — `dv.qaqc` registry and `dv.variables` classification — **DONE** (2026-07-26)

`tests/test_qaqc.py` 3 tests → 14 (+80 subtests); `tests/test_createvar.py` 21 → 41 (+49 subtests).
Both files run in under 3 s.

| Module | Before | After |
|---|---|---|
| `preprocessing/qaqc/measurements.py` | 98 %, **all GUI-only** | **100 %**, real tests |
| `variables/classification.py` | 97 %, **all GUI-only** | **100 %**, real tests |
| `variables/temporal.py` | partial | 93 % |
| `diive/qaqc/__init__.py` | — | **100 %** |
| `diive/variables/__init__.py` | — | **100 %** |

Both modules read as well-covered before this and were verified by nothing — the GUI drove them,
nothing asserted on the answers. They are pure lookup tables plus two string heuristics, which is
exactly the shape that regresses silently: a wrong answer is still a plausible-looking value.

What the tests pin:

- **`detect_measurement`** — every naming convention in the table, plus the ordering trap that
  `SWC` (soil water content) must beat `SW` (shortwave radiation). Getting that backwards would
  offer a soil probe the radiation zero-offset correction.
- **`classify_variable`** — the two documented boundary cases: `FC` must not swallow `FCH4`
  (CO2 vs methane flux), and bare `TA` is exact-matched so it does not catch `TARGET` / `TAU`.
- **Cross-checks between the tables.** Every code `detect_measurement` can return must exist in
  `MEASUREMENTS`, and every key `corrections_for_measurement` returns must exist in `CORRECTIONS` —
  otherwise the downstream label and correction lookups degrade silently.
- **`corrections_for_measurement` ordering** — measurement-specific corrections come before the
  generic ones, and an unknown code, a code with no specific physics, and `None` all behave alike.
- **`combine_variables`** — each method's arithmetic, and the `keep_overlap_only=False` identity
  fill (0 for add/subtract, 1 for multiply/divide) that lets a one-sided record survive.
  `fillgaps` is checked to ignore the flag, since filling gaps is a union by definition.
- **`auto_pick_column`** — `prefer` ranking, `avoid` exclusion (including on the preferred pass),
  and the empty-string miss.

Also added: an `__all__` resolution test for `dv.qaqc` and `dv.variables`, matching the one written
for `dv.corrections`. **Seven namespace `__init__` files still have no such test** — `dv.outliers`,
`dv.events`, `dv.gapfilling`, `dv.flux`, `dv.analysis`, `dv.plotting`, `dv.times`.

## Tier 1c — `dv.corrections` — **DONE** (2026-07-26)

`tests/test_corrections.py` went from 2 tests to 26 (plus 25 subtests), ~3.4 s. All ten public
symbols are now covered, along with the dispatch table and the namespace module itself.

| Module | Before | After (this file alone) |
|---|---|---|
| `diive/corrections/__init__.py` | **0 %** | **100 %** |
| `preprocessing/corrections/apply.py` | 67 %, GUI-only | **100 %** |
| `preprocessing/corrections/offsetcorrection.py` | 67 %, mostly GUI-only | **98 %** |
| `preprocessing/corrections/setto.py` | partial | **94 %** |
| `preprocessing/qaqc/measurements.py` | 98 %, GUI-only | 68 % from this file (bonus) |

What the tests pin:

- **The namespace `__all__`.** Every exported name resolves and *is* the implementation object, not
  a look-alike. This was the 0 % file — a symbol could have been dropped from the re-export list
  with the whole suite still green.
- **The dispatch table.** Each key's result is compared against calling the underlying function
  directly, so a mis-wired branch fails rather than merely running. `clamp_negatives` is checked to
  actually reach `remove_nighttime_zero_offset`.
- **Registry ↔ dispatch agreement.** Every `CorrectionSpec` key in `dv.qaqc.CORRECTIONS` must be
  dispatchable, and the test fails loudly if the registry gains a key. A correction the GUI offers
  but `apply_corrections` cannot route would otherwise only fail when a user clicks it.
- **Exact numeric contracts.** The nighttime offset is injected at a known constant, so the
  detected daily offset is an exact expected value (3.0); `nighttime_zero_offset_diagnostics(...)
  .corrected` is asserted identical to `remove_nighttime_zero_offset(...)`, which is the documented
  contract; `MeasurementOffsetFromReplicate` recovers a planted +5 offset exactly.
- **Ordering.** Cap-then-floor vs floor-then-cap give different results, proving each correction
  sees the previous one's output.

**Found along the way — since fixed.** `setto_threshold`, `set_exact_values_to_missing` and
`remove_relativehumidity_offset` renamed the caller's Series in place (`series.name =
"input_data"`). Each now binds a renamed copy instead, the pattern `_nighttime_zero_offset` already
used. A `TestCorrectionsDoNotMutateTheInput` class covers all six correction entry points for name,
values and index preservation, plus the rejected-call path (`setto_threshold` validated its `type`
argument after the old rename, so even a raising call renamed the input). Mutation-tested: the
regression test fails against the old behaviour.

## Tier 1b — Plot classes — **DONE** (2026-07-26)

Closed by `TestPlotClasses` in `tests/test_plots.py`: 14 tests, ~7 s for the whole file. Fixture is
a deterministic synthetic three-year hourly series (annual + diel cycle, no randomness), so the
aggregates are exact expected values rather than tolerances.

Coverage from `tests/test_plots.py` alone — i.e. with no GUI-test contribution at all — against
what each module had before, when every covered line came from `test_gui.py`:

| Module | Before (all GUI-only) | After, without any GUI test |
|---|---|---|
| `core/plotting/surface_grid.py` | 14 / 29 | **100 %** |
| `core/plotting/heatmap_datetime.py` | 139 / 152 | **91 %** |
| `core/plotting/ridgeline.py` | 162 / 179 | **91 %** |
| `core/plotting/shifted_distribution.py` | 89 / 96 | **90 %** |
| `core/plotting/bar.py` | 51 / 57 | **89 %** |
| `core/plotting/cumulative.py` | 130 / 142 | **87 %** |
| `core/plotting/waterfall.py` | 45 / 53 | **85 %** |
| `core/plotting/treering.py` | 106 / 223 | **80 %** |
| `core/plotting/heatmap_base.py` | 92 / 103 | **63 %** |

Assertions target each class's contract rather than "the call did not raise":

- `Cumulative` — the drawn curve's last value equals the series sum exactly (verified tight: a 1 %
  error fails).
- `WaterfallPlot` — one bar per resampled period, and the running budget closes on the total.
- `HeatmapYearMonth` — `agg='mean'` vs `'max'` vs `ranks=True` produce genuinely different meshes.
- `HeatmapDateTime` — the two orientations swap the axis labels.
- `RidgeLinePlot` — one panel per group, and `hspace` lands on the **gridspec**, pinning the
  documented gotcha that a later `gs.update(hspace=)` is a silent no-op.
- `datetime_surface_grid` — grid shape, hour axis, and NaN preservation across gaps.
- `LongtermAnomaliesYear` — anomalies are relative to the reference-period mean, and above/below
  are disjoint series.

Two notes for whoever extends this:

- `HeatmapYearMonth` lives in `core/plotting/heatmap_datetime.py`, not a module of its own.
- `LongtermAnomaliesYear` takes **one value per year with an integer year index**, not a
  datetime-indexed series — passing the latter raises a confusing
  `TypeError: Invalid comparison between dtype=datetime64[us] and int`. Its docstring is right;
  the error message is not obvious.

Still at 0 % after this pass: `core/plotting/seasonaltrend.py` (149 stmts — four module-level
functions `plot_decomposition`, `plot_seasonal_strength_by_period`, `plot_harmonics`,
`plot_spectral_density`, which take prepared inputs rather than a series, so they need their own
fixtures).

## Tier 1 (as found) — Codegen: 47 of 55 functions had no test

55 `*_to_code` functions. The grep pass found 47 with no test; coverage confirms the shape and
adds the detail that most of the "covered" ones are covered only by the GUI.

| Module | Line coverage | Covered lines that are GUI-test-only |
|---|---|---|
| `core/plotting/codegen.py` (16 fns) | 94 % | **100 %** |
| `flux/lowres/codegen.py` (2 fns) | 85 % | ~100 % |
| `flux/fluxprocessingchain/codegen.py` (7 fns) | 80 % | 40 % — the one with real tests (`test_flux_codegen.py`) |
| `preprocessing/corrections/codegen.py` (1 fn) | 62 % | high |
| `preprocessing/outlier_detection/codegen.py` (10 fns) | 54 % | **100 %** |
| `gapfilling/codegen.py` (7 fns) | 35 % | **100 %** |
| `flux/partitioning/codegen.py` (1 fn) | **12 %** | 100 % |

So `core/plotting/codegen.py` reads as well-covered at 94 % and is in fact verified by nothing but
a GUI smoke path. Only the six flux-chain functions in `test_flux_codegen.py` have real tests.

Functions with **no test at all** (47), grouped:

- `core/plotting/codegen.py` (16): `heatmap_datetime_to_code`, `heatmap_yearmonth_to_code`,
  `timeseries_to_code`, `dielcycle_to_code`, `cumulative_to_code`, `cumulative_year_to_code`,
  `waterfall_to_code`, `histogram_to_code`, `ridgeline_to_code`,
  `shifted_distribution_to_code`, `hexbin_to_code`, `heatmap_xyz_to_code`, `windrose_to_code`,
  `treering_to_code`, `datetime_surface_to_code`, `surface_xyz_to_code`
- `preprocessing/outlier_detection/codegen.py` (10): `stepwise_to_code`, `localsd_to_code`,
  `lof_to_code`, `absolutelimits_to_code`, `zscore_to_code`, `zscorerolling_to_code`,
  `zscoreincrements_to_code`, `trimlow_to_code`, `manualremoval_to_code`, `hampel_to_code`
- `gapfilling/codegen.py` (6): `ml_gapfill_to_code`, `xgboost_gapfill_to_code`,
  `randomforest_gapfill_to_code`, `longterm_ml_gapfill_to_code`,
  `longterm_xgboost_gapfill_to_code`, `longterm_randomforest_gapfill_to_code`
- `variables/utilities.py` (3): `combine_variables_to_code`, `potrad_to_code`,
  `calc_vpd_from_ta_rh_to_code`
- `analysis/` (5): `compound_extremes_to_code`, `rank_drivers_to_code`, `gapstats_to_code`,
  `spectrogram_to_code`, `seasonal_trend_to_code`
- one each: `level42_to_code`, `jointunc_to_code`, `partitioning_to_code`, `corrections_to_code`,
  `scatter_to_code`, `select_records_to_code`, `feature_engineer_to_code`

**0 of 55 appear in any example.**

**Suggested fix:** one table-driven test module. These are pure `dict -> str`, so the test is
`compile(generated, "<gen>", "exec")` plus a substring assertion per function — roughly 200 lines
covering ~1 500 LOC. Consider also `exec`ing a subset against example data to catch signature
drift between generator and library, which a `compile()` check alone will miss.

---

## Tier 2 — Public API with neither example nor test

Coverage confirms the grep findings here, except where marked.

### `dv.qaqc` measurement registry — 7 of 10 symbols

`MEASUREMENTS`, `Measurement`, `CORRECTIONS`, `CorrectionSpec`, `corrections_for_measurement`,
`correction_spec`, `detect_measurement`, `measurement_label`.

`preprocessing/qaqc/measurements.py` is 98 % covered — and **39 of 40 covered lines are
GUI-test-only.** So the registry is exercised, but only by the GUI, and asserted on nowhere.
`detect_measurement` maps a variable name to a measurement kind: exactly the string-heuristic
function that regresses silently. Pure lookup tables, trivial to test directly.

### `dv.variables` — classification and combination

| Symbol | Status |
|---|---|
| `classify_variable`, `VariableClass`, `CATEGORY_*` | `variables/classification.py` 97 % covered, **32 of 33 lines GUI-only**. Drives every GUI pill colour |
| `combine_variables`, `combine_variables_to_code` | `variables/utilities.py` 71 %, majority GUI-only. Backs the Combine-variables tab |
| `potrad_to_code`, `calc_vpd_from_ta_rh_to_code` | same module, uncovered lines 103-109 / 66-81 |
| `auto_pick_column` | no test, no example |
| `daytime_nighttime_flag_from_swinpot` | no test, no example (the `DaytimeNighttimeFlag` class is covered) |

### `dv.corrections` — weakest namespace relative to its size

`tests/test_corrections.py` is 37 lines / 2 tests for 10 public symbols, and the namespace
`__init__` is never imported (0 %). `preprocessing/corrections/offsetcorrection.py` is 67 %
covered with two thirds of that GUI-only; `preprocessing/corrections/apply.py` is 67 % and
GUI-only.

| Symbol | Example | Test |
|---|---|---|
| `apply_corrections` | — | GUI-only |
| `nighttime_zero_offset_diagnostics` | — | — |
| `NighttimeZeroOffsetResult` | — | — |
| `MeasurementOffsetFromReplicate` | yes | — |
| `remove_relativehumidity_offset` | yes | — |
| `setto_value` | yes | — |
| `remove_nighttime_zero_offset` | yes | GUI-only |
| `setto_threshold` | yes | GUI-only |
| `set_exact_values_to_missing`, `WindDirOffset` | yes | yes |

`apply_corrections` is the dispatch entry point every correction tab routes through — worth a
direct test of the dispatch table, not just the individual correction functions.

### Gap-filling result contract

| Symbol | Status |
|---|---|
| `GapFillingResult` | `core/ml/results.py` is 100 % covered but **entirely GUI-only**. The documented `.results` return type for every gap-filler, and no non-GUI test constructs or inspects one. Appears only in `examples/COOKBOOK.md` prose |
| `prediction_scores` | exported on `dv.gapfilling`; no test, no example |
| `OptimizeParamsRFTS` | no test, no example |
| `OptimizeParamsTS` | example, no test — `core/ml/optimization.py` 302 stmts, 25 covered, all GUI-only |
| `LongTermGapFillingXGBoostTS` | example, no test (the RF sibling is tested) |

### Other

| Symbol | Status |
|---|---|
| `dv.plotting.DateTimeSurface`, `datetime_surface_grid` | `core/plotting/surface_grid.py` 48 %, all GUI-only. Runs without the `gui3d` extra, so testable headless |
| `dv.events.CATEGORY_COLORS` | no test, no example |
| `dv.load_parquet_many` | no test, no example, despite a documented `progress_callback` |
| `UstarDetectionMPT`, `UstarThresholdConstantScenarios`, `FlagSingleConstantUstarThreshold` | `flux/lowres/ustarthreshold.py` **29 %**, 92 of 126 covered lines GUI-only |

---

## Tier 3 — Tested but no example

Coverage cannot speak to this tier; it stays grep-derived. Each is a documented public entry point.

| Symbol(s) | Area |
|---|---|
| `partition_nee_nighttime_oneflux`, `partition_nee_nighttime_reddyproc`, `partition_nee_daytime_reddyproc`, `partition_nee_daytime_oneflux` | The functional wrappers. All four *classes* have examples; the function form does not |
| `JointUncertaintyPAS20`, `joint_uncertainty_pas20` | `RandomUncertaintyPAS20` has an example; the joint step does not, so the two-step workflow is never shown end to end |
| `accumulated_local_effects`, `accumulated_local_effects_2d`, `AleCurve`, `Ale2DResult` | `analysis.experimental` — only reachable via the `DriverAnalysis` example |
| `count_gaps`, `dataframe_overview`, `profile_dataframe`, `rank_drivers`, `daily_correlation` | `dv.analysis` inspection helpers |
| `format_timestamp`, `insert_timestamp`, `validate_timestamp_column_name` | `dv.times` |
| `keep_vars`, `to_diive_format`, `transform_yearmonth_matrix_to_longform` | top-level |
| `HampelDaytimeNighttime`, `LocalOutlierFactorAllData`, `LocalOutlierFactorDaytimeNighttime` | The day/night alias-vs-wrapper names CLAUDE.md flags as easy to confuse — an example showing the difference would pay for itself |
| `make_event_flag_name` | `dv.events` |
| `InfluxIO`, `save_project`, `load_project`, `MetadataStore`, `add_console_sink` | infrastructure; example may not be warranted |

---

## Tier 4 — Has an example but no test

| Symbol | Real line coverage of backing module |
|---|---|
| `DetectTimestampShifts` | **0 %** (281 stmts) |
| `LongtermAnomaliesYear` | **0 %** (`core/plotting/seasonaltrend.py`, 149 stmts) |
| `Fitter` | `core/plotting/_fitplot.py` **0 %**; `core/dfun/fits.py` low |
| `MultiDataFileReader`, `ReadFileType`, `search_files` | `core/io/filereader.py` partial, 82 of 290 covered lines GUI-only |
| `UstarVekuriThresholdDetection` | 19 of 193 stmts, all GUI-only |
| `TimeLagAnalysis` | 23 of 211 stmts, all GUI-only |
| `FindOptimumRange` | 27 of 222 stmts, all GUI-only |
| `harmonic_analysis` + `reconstruct_harmonics`, `periodogram`, `fft_decompose`, `multi_scale_harmonics` | 6 of 7 symbols in `analysis/harmonic.py` untested |
| `GrangerCausality` | 10 of 53 stmts, all GUI-only |
| `ManualRemoval` | 33 of 37 stmts, all GUI-only |
| `DailyCorrelation`, `StratifiedAnalysis` | `analysis/correlation.py`, `analysis/decoupling.py` |
| `get_encoded_value_from_int`, `get_encoded_value_series` | top-level |
| `add_driver` | `flux/fluxprocessingchain/container.py` — 235 of 269 covered lines GUI-only |
| `detect_fluxbasevar`, `run_level33_variable_ustar` | chain; `levels/level33.py` 79 of 122 GUI-only |
| `run_level42_nighttime_reddyproc`, `run_level42_daytime_reddyproc`, `run_level42_daytime_oneflux` | 3 of 4 L4.2 entry points reached only through the ONEFlux-nighttime path |

---

## Tier 5 — Revised against real coverage

The grep pass guessed at indirect coverage. Coverage settles it. **Four of my earlier calls were
wrong** — corrected here.

### Overturned: these are genuinely covered by real tests

| Module | Real coverage | Earlier claim |
|---|---|---|
| `gapfilling/similarity.py` | **88 %**, only 32/128 GUI-only | **[revised]** I called it "indirect only, no direct test". It is well exercised by the `FluxMDS` and `RandomUncertaintyPAS20` tests. Deprioritise |
| `core/base/flagbase.py` | **69 %**, 44/134 GUI-only | **[revised]** Genuinely covered via the outlier detectors, as suspected but now confirmed |
| `flux/lowres/storage_correction.py` | **47 %**, 26/72 GUI-only | **[revised]** I listed it as "no test". Real tests reach it via `run_level31` |
| `flux/fluxprocessingchain/codegen.py` | 80 %, 40 % GUI-only | **[revised]** The healthiest codegen module, thanks to `test_flux_codegen.py` |

### Confirmed or worse than the grep pass suggested

| Module | Coverage | Note |
|---|---|---|
| `flux/lowres/selfheating.py` | **10 %** (69 of 667), all GUI-only | Largest under-tested module. 3 examples, no real test |
| `flux/lowres/ustarthreshold.py` | **29 %**, 73 % of that GUI-only | |
| `core/times/decomposition_utils.py` | **45 %**, mostly GUI-only | CLAUDE.md records a **fixed bug** here (missing `period`, unsupported `fit(weights=)`). The regression test still does not exist |
| `core/plotting/heatmap_base.py` | 73 %, **92 of 103 covered lines GUI-only** | **[revised]** Worse than the "indirect via HeatmapDateTime" I assumed — `HeatmapDateTime` itself is GUI-only |
| `preprocessing/qaqc/eddyproflags.py` | 80 %, 55/99 GUI-only | Confirmed indirect via `run_level2`; 7 flag functions, no direct unit test |
| `preprocessing/outlier_detection/stepwiseoutlierdetection.py` | 84 %, 81 of 119 GUI-only | The documented chained-detection API |
| `flux/fluxprocessingchain/levels/_rerun.py` | 87 %, mostly GUI-only | **The documented re-run cascade has no test in `test_fluxprocessingchain.py`.** Its coverage comes from the GUI driving levels repeatedly |
| `flux/fluxprocessingchain/levels/_qcf.py` | 86 %, GUI-only | |
| `preprocessing/qaqc/qcf.py` | **53 %** | `FlagQCF` — 339 stmts, 159 missing |
| `preprocessing/qaqc/meteoscreening.py` | **55 %**, half GUI-only | |
| `preprocessing/outlier_detection/localsd.py` | **63 %** | Lowest of the outlier detectors (siblings are 80-95 %) |

### Partially tested modules worth a second pass

Public symbols never named in a non-GUI test:

| Module | Untested / total | Examples |
|---|---|---|
| `core/times/times.py` | 30/37 | `format_timestamp_to_fluxnet_format`, `detect_freq_groups`, `sort_timestamp_ascending`, `remove_rows_nat`, `validate_timestamp_monotonic`, … |
| `core/dfun/stats.py` | 29/31 | quantile helpers `q01`…`q99`, `series_start`, … |
| `core/plotting/plotfuncs.py` | 24/29 | `format_ticks`, `format_spines`, `hide_xaxis_yaxis`, … (mostly cosmetic) |
| `core/dfun/frames.py` | 17/20 | `keep_vars`, `trim_frame`, `detect_new_columns`, `aggregated_as_hires`, `rename_cols`, … |
| `core/io/files.py` | 9/10 | `to_diive_format`, `save_parquet`, `load_parquet_many`, `unzip_file`, … |
| `core/ml/common.py` | 3/4 | the three diagnostic plot functions |
| `preprocessing/outlier_detection/lof.py` | 2/4 | `lof`, `suggest_lof_params` |

---

## Suggested order of work

Reordered after seeing the coverage data. The GUI-dependency problem was not visible to the grep
pass and is arguably the most important item.

1. ~~**Codegen test module** (Tier 1)~~ — **done**, `tests/test_codegen.py`.
2. ~~**Break the plotting classes out of GUI-only coverage**~~ — **done**, `TestPlotClasses` in
   `tests/test_plots.py`. Follow-up left: `core/plotting/seasonaltrend.py`, still 0 %.
3. ~~**`dv.corrections` namespace import + `apply_corrections` dispatch test**~~ — **done**,
   `tests/test_corrections.py`. The same `__all__` check is still worth adding for the **other
   nine namespace `__init__` files** (a one-line import test each); only `dv.corrections` has it.
4. ~~**`dv.qaqc` registry and `dv.variables` classification tests**~~ — **done**, in
   `tests/test_qaqc.py` and `tests/test_createvar.py`.
4b. ~~**`__all__` tests for the remaining namespaces**~~ — **done**, all ten in
   `tests/test_imports.py`, plus the four-place namespace-registration check.
5. ~~**`GapFillingResult` / `prediction_scores`**~~ — **done**, in `tests/test_gapfilling.py`.
6. ~~**Re-run cascade test**~~ — **done**, `TestRerunCascade` in `tests/test_fluxprocessingchain.py`.
7. ~~**`stl_decompose` regression test**~~ — **done**, `TestStlDecompose` in `tests/test_time.py`
   (the function is at 84 %). Follow-up: the other five functions in
   `core/times/decomposition_utils.py` remain at 2-8 %.
8. ~~**Raise `FlagQCF` (53 %)**~~ — **done**, now 95 %. `localsd.py` (63 %) is still open.
9. Examples for the Tier 3 list, starting with the joint-uncertainty workflow and the
   `partition_nee_*` function forms.
10. Drop `similarity.py`, `flagbase.py`, and `storage_correction.py` from the worry list — measured,
    fine.

## Notes on the tooling change

- `pytest-cov` 7.1.0 and `coverage` 7.15.2 were added to the `dev` group; `pyproject.toml` and
  `uv.lock` are modified accordingly.
- `.coverage` and `htmlcov/` are already in `.gitignore`, so no artifacts leak into the repo.
- Useful invocations:

```bash
uv run pytest tests/ --cov=diive --cov-report=term-missing
```

```bash
uv run pytest tests/ --cov=diive --cov-context=test --cov-report=html
```

The context-recording form is what produced the GUI-dependency analysis above; the HTML report
lets you filter a file's lines by covering test.

Two things to know if you want a library-only number:

- There is no `--cov-omit` flag. Omissions go in config — add to `pyproject.toml`:

  ```toml
  [tool.coverage.run]
  omit = ["diive/gui/*"]
  ```

- Omitting `diive/gui` from the *report* does not stop `test_gui.py` from *contributing* library
  coverage. To reproduce the 36 % figure above you have to deselect the test file itself:

```bash
uv run pytest tests/ --ignore=tests/test_gui.py --cov=diive --cov-report=term-missing
```
