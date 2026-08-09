# Code Review Findings

Review dates: 2026-08-06 (round 1: core numerics + GUI) · 2026-08-07 (round 2: the modules round 1
left out; round 3: everything both left out) · diive v0.91.0 · branch `indev` (at `af022000`)

Working document. The review itself changed no code; fixes land as separate commits and each one
is recorded in place, in the entry it closes. Findings are ordered by severity within each section;
each carries a file:line anchor and, where marked **[reproduced]**, a runnable repro that was
actually executed during the review.

Scope reviewed: the library's numerical/scientific code (`core/`, `preprocessing/`, `gapfilling/`,
`flux/`, `variables/`, `analysis/`, `core/io/`, `core/metadata/`) and the desktop GUI
(`diive/gui/`). **Left out by rounds 1–2, and since covered by round 3** (see *Round 3* below;
what is still unreviewed after it is listed at the end of *Reviewed and found sound*):
`flux/lowres/selfheating.py`,
`flux/lowres/ustar_mp_detection.py` / `ustarthreshold.py` / `ustar_bootstrap.py` /
`ustar_vekuri_detection.py`, `flux/lowres/timelag_analysis.py`, `analysis/driveranalysis/`,
`analysis/harmonic.py` / `granger.py` / `decoupling.py`, most of `core/plotting/`,
`core/io/db/influx/`, and the GUI's `surface3d.py` / `surfacexyz.py` (optional `gui3d` extra) and
`icons.py` / `theme.py`. The four NEE-partitioning ports were read (round 2) but not audited
line-by-line against their R/C references — they are faithful ports with dedicated test files, so
auditing them without the reference sources has low yield.

Status column: `[ ]` open · `[x]` fixed · `[-]` won't fix / by design (add a note saying why).

---

# Before this work is called done

Fixing the findings is not the whole job. These are deliberately **not** done per-fix — batching
them once the backlog settles avoids rewriting the same entries repeatedly — but they must happen
before a release, and several are already stale today.

### [ ] CHANGELOG.md

Nothing in this effort has been written up yet. **Four breaking changes** have landed on `indev` so
far and each needs an entry saying what silently changes for existing code:

| Commit | Breaks |
|---|---|
| `45614fb3` `feat!: remove the H2O self-heating path` | `flux_type` gone from `ScopPhysics` / `ScopOptimizer` / `ScopApplicator`; LE correction no longer offered at all |
| `a327a4ee` `fix!: flag missing records as NaN, not 0` | `overall_flag` is NaN at missing records; any `(flag == 0).count()` changes |
| `876bec12` `feat!: combine variables only where both are available` | `keep_overlap_only` gone from `combine_variables` and its codegen |
| `57b8b845` `refactor!: remove the dead harmonic functions` | `reconstruct_harmonics`, `periodogram`, `fft_decompose`, `multi_scale_harmonics` removed |
| L50's fix | `quality_weighted_decompose` removed; `weights` gone from `stl_decompose`; `quality` / `quality_weighted` gone from `SeasonalTrendDecomposition`. None of it ever weighted anything |

The non-breaking fixes are user-visible too and change numbers people may have already published —
in particular **L48/L49** (FFT amplitudes were ~54% of truth, and one bin off), **L61/L63** (cells
drawn over regions holding no data), **L42** (hqflux reported missing records as valid),
**L54/L55** (`DriverAnalysis` — every number it reports moves, though it is experimental) and
**L51/L52** (`StratifiedAnalysis` now analyses the records and keeps the z bins it used to drop,
so both the curves and the bin count in an existing figure change), **L47/L73** (STL returns
components instead of NaN for any gappy series, and `harmonic_decompose` picks different
components — which also shifts anything built on `features_stl`) and **L30** (a record whose u\*
is missing is now rejected rather than accepted, so u\*-filtered fluxes lose those records — how
many depends entirely on how gappy the site's u\* is). The
v0.91.0 entry already opens with *"Two of these change results silently, with no error and no
warning"*; this round needs the same treatment.

### [~] `diive/gui/MANUAL.md`

The *"Keep overlapping data points only"* checkbox description (line 969) was stale from
`876bec12` and has been **fixed 2026-08-07** — replaced with the overlap-only rule plus the new
record-loss reporting, and `MANUAL.html` regenerated via `diive/gui/build_manual.py`. Fixed
immediately rather than batched because it was left behind by a change in this same effort.

**Still to do:** sweep the rest of the manual for the other removals before release — the H2O
self-heating path and the outlier flag semantics both have user-facing descriptions that may
still describe the old behaviour.

### [ ] `CLAUDE.md`

Updated piecemeal so far (the combine-variables tab description, `1c392e66`). Re-read the sections
covering every changed area once the backlog settles: the outlier day/night conventions (flag
semantics), `dv.analysis` and `dv.plotting` namespace tables, and the self-heating notes.

### [ ] `docs/`

`docs/auto_examples/` is **generated** by sphinx-gallery from `examples/`, so it regenerates on the
next build — no hand-editing. It is currently stale (`selfheating.py`/`.ipynb`/`.rst` still pass
`flux_type=FLUX_TYPE`). Just remember to rebuild, and check the API pages no longer list the four
deleted harmonic functions.

### [ ] Examples

`examples/analysis/analysis_harmonic.py` is the only caller of `harmonic_analysis`, and it was
written against the buggy amplitudes — its narrative text about the window's effect may now say the
opposite of what the code produces. Re-read it against the corrected behaviour.

---

# Triage index — all 81 findings by severity

The detailed entries below stay grouped by review round and module. This index is the **fix order**.

**Ranking rubric.** For a scientific library the ordering is *not* crash-first. A traceback is
self-announcing and nobody publishes one; a plausible-looking wrong number gets into a paper. So:

| Tier | Meaning |
|---|---|
| **S1** | **Silently wrong scientific output.** Runs clean, returns a believable number that is wrong. |
| **S2** | **Silently does nothing, or silently loses data.** The user believes an operation ran; it didn't, or it dropped records without saying so. |
| **S3** | **Crash on a legitimate documented input.** Loud, blocking, but never corrupts a result. |
| **S4** | **Contract mismatch.** Behaviour is defensible; the docstring or the exposed control is wrong about it. |
| **S5** | **Cosmetic, dead code, or latent robustness.** No user-visible wrong outcome today. |
| **?** | **Unresolved — needs a decision or an external source**, and could land at S1 once settled. |

## ? — Decide these first (they gate other work)

| ID | Finding | Where |
|---|---|---|
| ~~L5~~ | ~~Does ONEFlux clip or trim window indices?~~ **Answered 2026-08-07: it trims** — diive clipped, so edge fills were biased. Fixed | `gapfilling/similarity.py:240` |
| ~~L46~~ | ~~Does Burba 2008 intend BUR08 to drop the WPL dilution factor?~~ **Answered 2026-08-07: no, it applies to every method.** Fixed | `flux/lowres/selfheating.py:614` |
| L72 | Is InfluxDB v2's *delete* range stop-exclusive? If so the pre-upload delete leaves the last record and duplicates survive | `core/io/db/influx/influxio.py:122` |

## S1 — Silently wrong scientific output (12 + 1 by design)

| ID | Finding | Where |
|---|---|---|
| ~~L37~~ | ~~**The H2O/LE self-heating path must be removed**~~ (done 2026-08-07) — no self-heating correction for LE exists in EC science | `flux/lowres/selfheating.py` |
| ~~L28~~ | ~~USTAR bootstrap bypasses the 3000-record minimum `detect()` enforces~~ (done 2026-08-07) | `flux/lowres/ustar_mp_detection.py:561` |
| ~~L14~~ | ~~`combine_variables(keep_overlap_only=False)`: subtract/divide return the **negation / reciprocal**~~ (done 2026-08-07) — option removed | `variables/utilities.py:73` |
| ~~L54~~ | ~~`DriverAnalysis(deseasonalize=True)` fabricates target values by interpolation~~ (done 2026-08-07) | `analysis/driveranalysis/driveranalysis.py:80` |
| ~~L55~~ | ~~Per-regime relevance judged against the *global* model's `.RANDOM` floor~~ (done 2026-08-07) | `analysis/driveranalysis/driveranalysis.py:760` |
| ~~L48~~ | ~~`harmonic_analysis` reads the wrong FFT bin~~ (done 2026-08-07) | `analysis/harmonic.py:89` |
| ~~L49~~ | ~~Windowed FFT amplitudes never corrected for coherent gain~~ (done 2026-08-07) | `analysis/harmonic.py:70` |
| ~~L74~~ | ~~u\* filtering keeps the first high-turbulence record after a low-turbulence period~~ (by design 2026-08-07) — diive favours data availability; documented as a deviation | `flux/lowres/ustarthreshold.py:139` |
| ~~L52~~ | ~~`StratifiedAnalysis` listwise-drops on **every** column — 20 000 rows → 100, silently~~ (done 2026-08-07) | `analysis/decoupling.py:68` |
| ~~L61~~ | ~~`HeatmapYearMonth` mis-places cells when months are non-contiguous~~ (done 2026-08-07) | `core/plotting/heatmap_datetime.py:395` |
| ~~L63~~ | ~~Empty X/Y/Z bins dropped, then rendered as measured cells~~ (done 2026-08-07) — fixed in `GridAggregator`, so `HeatmapXYZ` benefits too | `analysis/gridaggregator.py:429` |
| ~~L2~~ | ~~`WindDirOffset` ignores `hist_n_bins`~~ (done 2026-08-07) — also pinned the bins to the full circle | `preprocessing/corrections/offsetcorrection.py:476` |
| ~~G1~~ | ~~Partitioning tabs run at **(0, 0) UTC** when the project site is unconfigured~~ (done 2026-08-07) — 3 of the 4 ports | `gui/tabs/_partitioning_base.py:300` |

## S2 — Silently does nothing / silently loses data (25)

| ID | Finding | Where |
|---|---|---|
| ~~L75~~ | ~~25 `detail()` debug lines cannot print at any verbosity~~ (done 2026-08-07) — module default is now settable | `core/utils/console.py:191` |
| ~~L36~~ | ~~Self-heating gap-fill drops every gap before filling~~ (done 2026-08-07) | `flux/lowres/selfheating.py:390` |
| ~~L38~~ | ~~Corrected flux becomes NaN wherever the correction term is missing~~ (done 2026-08-07) — carried through + flagged | `flux/lowres/selfheating.py:1225` |
| ~~L40~~ | ~~Unrecognised `correction_method_base` returns an empty result instead of raising~~ (done 2026-08-07) | `flux/lowres/selfheating.py:270` |
| ~~L45~~ | ~~`ScopOptimizer` silently drops classes with <10 rows~~ (done 2026-08-07) | `flux/lowres/selfheating.py:904` |
| ~~L42~~ | ~~`analyze_highest_quality_flux` counts missing records as valid~~ (done 2026-08-07) — 1519 of 3000 reported as "99.5%" | `flux/lowres/hqflux.py:251` |
| ~~L7~~ | ~~**Outlier flags are never NaN**~~ (done 2026-08-07) — was the root cause of L30 and L42 | `core/base/flagbase.py:182` |
| ~~L30~~ | ~~NaN u\* or NaN per-record threshold → flagged 0 (accepted)~~ (done 2026-08-07) — completes the L7 half-fix | `flux/lowres/ustarthreshold.py:142` |
| ~~L29~~ | ~~A failing bootstrap window is completely silent~~ (done 2026-08-07, with L28) | `flux/lowres/ustar_bootstrap.py:39` |
| ~~L26~~ | ~~`_class_bounds` can emit `end < start` → NaN class mean → up to 11 candidate classes skipped~~ (done 2026-08-07) | `flux/lowres/ustar_mp_detection.py:285` |
| ~~L32~~ | ~~`UstarDetectionMPT` is exported but never stores its results~~ (done 2026-08-07) — class removed | `flux/lowres/ustarthreshold.py:561` |
| ~~L34~~ | ~~`annual_thresholds_` holds the sentinel `10.0` on failure — a plausible threshold that filters everything~~ (done 2026-08-07) | `flux/lowres/ustar_mp_detection.py:520` |
| ~~L47~~ | ~~STL returns an all-NaN decomposition from a **single** NaN; `seasonality_strength` reads 0.0~~ (done 2026-08-07) | `core/times/decomposition_utils.py:133` |
| ~~L50~~ | ~~`quality_weighted_decompose` ignores the weights entirely; `summary()` prints "Quality-weighted: True"~~ (done 2026-08-07) — fake path removed | `core/times/decomposition_utils.py:100` |
| ~~L58~~ | ~~`detect_seasonality` fabricates `primary_period=365` when the periodogram yields nothing~~ (done 2026-08-07) | `core/times/decomposition_utils.py:490` |
| ~~L51~~ | ~~`StratifiedAnalysis` drops z-bins whose rounded label collides — 19 of 120 lost, no warning~~ (done 2026-08-07) | `analysis/decoupling.py:213` |
| L76 | BUR06 uses a canopy `ra` (`u/u*^2`) where Burba 2006 specifies a per-element one (`7.4*sqrt(d/U)`, ~6x apart) and drops the retained fraction `fr`; the fitted SF absorbs both | `flux/lowres/selfheating.py:471` |
| ~~L53~~ | ~~`CompoundExtremes` returns zero classified periods for a single year, silently~~ (done 2026-08-07) | `analysis/compoundextremes.py:168` |
| ~~L59~~ | ~~`multi_scale_harmonics` swallows every exception~~ (done 2026-08-07) — function deleted as dead code | `analysis/harmonic.py:432` |
| ~~L19~~ | ~~`features_stl=True` can produce nothing — any single NaN skips a column, logged only at DEBUG~~ (done 2026-08-07) | `core/ml/feature_engineer.py:726` |
| L16 | SWIN short-gap interpolation never fires near dawn/dusk — the night's NaN run inflates the gap length | `gapfilling/swin.py:794` |
| L4 | `keep_records_where`: an "open" bound is not open when `inclusive != 'both'` — drops the extreme record | `core/dfun/frames.py:110` |
| L17 | `MetadataStore.rename` silently drops a record on a name collision | `core/metadata/__init__.py:325` |
| L64 | glTF `.glb` export bakes the texture **mirrored** along the date axis | `gui/tabs/surface3d.py:744` |
| L66 | `datetime_surface_grid` omits the `TIMESTAMP_START` conversion — 3-D surface offset half a period from the heatmap | `core/plotting/surface_grid.py:68` |
| L67 | 3-D export buttons write the **previous** variable's relief after a render that produced nothing | `gui/tabs/surface3d.py:945` |
| G4 | `restore_controls` silently keeps the current combo value when the saved entry is gone — can flip the joint-uncertainty divisor | `gui/widgets/state_utils.py:51` |
| ~~L73~~ | ~~`harmonic_decompose` picks the top-N bins by power, so a windowed strong component's leakage outranks a genuine weaker one — the same component is returned twice~~ (done 2026-08-07) | `core/times/decomposition_utils.py:275` |

## S3 — Crash on legitimate input (14)

| ID | Finding | Where |
|---|---|---|
| L1 | `Hampel` crashes on any non-fixed frequency (monthly/yearly/business-day) | `preprocessing/outlier_detection/hampel.py:228` |
| L3 | Frequency detection: off-by-one denominator → clean 2-row series "too irregular", 1-row bare `KeyError` | `core/times/times.py:1386` |
| L27 | `set_storage_to_zero=True` still requires the storage column — the exact case it documents | `flux/lowres/storage_correction.py:150` |
| L39 | `ScopApplicator`'s undocumented column-name contract — legal inputs raise `KeyError` | `flux/lowres/selfheating.py:1214` |
| L43 | Default fringe-bin trimming empties the time-lag histogram; the `IndexError` escapes the batch helpers | `flux/lowres/timelag_analysis.py:148` |
| L31 | `UstarThresholdConstantScenarios.calc(showplot=True)` crashes on pandas 3 | `flux/lowres/ustarthreshold.py:337` |
| L33 | `UstarVekuriThresholdDetection.summary()` crashes on its own guard before `detect()` | `flux/lowres/ustar_vekuri_detection.py:187` |
| ~~L56~~ | ~~`harmonic_decompose` returns `frequencies` one element longer~~ (done 2026-08-07) | `core/times/decomposition_utils.py:307` |
| L65 | `RidgeLinePlot` cannot plot any series containing a gap | `core/plotting/ridgeline.py:196` |
| L69 | `Cumulative.plot` raises on an all-NaN column | `core/plotting/cumulative.py:327` |
| L68 | `datetime_surface_grid` destroys a variable literally named `DATE` or `TIME` | `core/plotting/surface_grid.py:71` |
| L13 | `transform_yearmonth_matrix_to_longform` hardcodes the column names it drops | `core/dfun/frames.py:644` |
| L22 | `MultiDataFileReader` raises `AttributeError` when every file is empty | `core/io/filereader.py:320` |
| G2 | `_outlier_base._on_done` can `KeyError` when the dataset changes mid-run | `gui/tabs/_outlier_base.py:666` |

## S4 — Contract mismatch (14)

| ID | Finding | Where |
|---|---|---|
| L15 | `flag_ssitc_eddypro_test` performs no conversion despite documenting one | `preprocessing/qaqc/eddyproflags.py:490` |
| L41 | `ScopPhysics` documents an RF + MDV gap-fill that does not exist | `flux/lowres/selfheating.py:152` |
| L44 | `TimeLagAnalysis` docstring states three parameter facts the code contradicts | `flux/lowres/timelag_analysis.py:90` |
| L57 | `reconstruct_from_components` forces the trend's NaN onto reconstructions excluding the trend | `core/times/decomposition_utils.py:420` |
| L60 | `seasonality_strength` formula and `'iterations'` return value both mis-documented | `analysis/seasonaltrend.py:174` |
| L62 | `show_less_xticklabels` accepted and documented by `HeatmapDateTime` but never applied | `core/plotting/heatmap_datetime.py:245` |
| L21 | `crosscorr` omits dates for three early-outs while others write NaN | `preprocessing/qaqc/detect_timestamp_shifts.py` |
| L18 | `FeatureEngineer` rolling stages re-engineer already-engineered columns; other stages skip them | `core/ml/feature_engineer.py` |
| L9 | `GridAggregator` keys its frame by Series name — two roles sharing a name collide | `analysis/gridaggregator.py:119` |
| L11 | `JointUncertaintyPAS20` cumulative scenario term not masked to available flux | `flux/lowres/uncertainty.py:867` |
| L6 | Random-uncertainty method 4 uses an asymmetric neighbour window (5 below, 4 above) | `flux/lowres/uncertainty.py:708` |
| L8 | `FlagQCF`'s documented "QCF is NaN if no flag available" branch is unreachable | `preprocessing/qaqc/qcf.py:640` |
| G3 | Pinned tabs are not actually frozen against added columns | `gui/app.py:899` |
| G6 | `WorkerRunner` clears `is_running` before emitting — re-entry window | `gui/widgets/worker.py:73` |

## S5 — Cosmetic / dead / latent (13)

| ID | Finding | Where |
|---|---|---|
| L10 | `vectorize_timestamps` `.SEASON` as `Int64` forces object-dtype arrays into every ML fit | `core/times/times.py:1245` |
| L12 | `LocalSD`: values exactly on the limit are in neither `ok` nor `rejected` | `preprocessing/outlier_detection/localsd.py:279` |
| L20 | `lagged_variants` edge-fill is conditional but documented as unconditional | `variables/temporal.py:461` |
| L23 | `sort_multiindex_columns_names` mutates the list it iterates (reverses moved columns) | `core/dfun/frames.py:510` |
| L24 | Nested-quote f-string prints a literal `{limit}`; `_calculate_gap_sizes` is dead | `gapfilling/interpolate.py:143` |
| L25 | `_extract_and_convert_flag_from_multidigit` turns a scalar `0` code into NaN | `preprocessing/qaqc/eddyproflags.py:47` |
| L35 | USTAR docstring examples import from the wrong namespace | `flux/lowres/ustar_bootstrap.py:133` |
| L70 | Rolling cell aggregator uses an `n+1`-row window for even `n` | `gui/tabs/surface3d.py:98` |
| L71 | `convert_ts_to_timezone` cannot accept the `DatetimeIndex` its docstring promises | `core/io/db/influx/common.py:59` |
| G5 | `_screening_base._run` starts unbounded concurrent worker threads | `gui/tabs/_screening_base.py:729` |
| G7 | `_compute_payload` writes tab state from the worker thread | `gui/tabs/_outlier_base.py:463` |
| G8 | `save_config` catches only `OSError` | `gui/config.py:36` |
| G9 | Project load transiently materialises the previous session's event columns | `gui/app.py:1372` |

## Cross-cutting observations

- **`selfheating.py` has no test coverage at all** — six findings (L36–L41, L45) live in ~1700
  untested lines. A single smoke test would have caught the gap-fill no-op and the silent
  empty-result-on-typo.
- **L7 is a root cause, not an isolated finding.** The "missing record → flag 0" rule in
  `FlagBase.repeat` propagates into L30 (u\* filtering) and L42 (hqflux statistics). Fixing L7
  resolves the family; fixing L30/L42 individually does not.
- **Three findings share one shape: "the aggregator kept only the bins that occurred, and the
  consumer treated them as contiguous"** — L61 (`HeatmapYearMonth`), L63 (`GridAggregator` →
  X/Y/Z surface and `HeatmapXYZ`). Worth one shared fix in the reindex step.
- **`analysis/harmonic.py` + `decomposition_utils.py` carry four S1/S2 numerical defects
  (L47–L50, L56–L59)** between them. If the FFT/STL surface is not actively used, deprecating it
  is cheaper than repairing it.

---

## Library — real bugs

### [ ] L1. `Hampel` crashes on any non-fixed frequency (monthly / yearly / business-day)

`diive/preprocessing/outlier_detection/hampel.py:228`

```python
if index.freq is not None:
    step = pd.Timedelta(index.freq.nanos, unit='ns')   # ValueError on MS / YS / B
else:
    step = pd.Series(index).diff().median()            # this branch handles it fine
```

`DateOffset.nanos` raises for non-fixed offsets. The `else` branch already computes a usable step
the same way; the guard simply never reaches it. `_gap_flanking_records`'s own docstring promises an
all-False fallback "when the index is not a usable time axis" — this case slips past that promise
and takes the whole detector down.

**[reproduced]**

```python
idx = pd.date_range('2000-01-01', periods=60, freq='MS', name='TIMESTAMP_MIDDLE')
s = pd.Series(np.random.randn(60), index=idx, name='x')
dv.outliers.Hampel(series=s, separate_day_night=False, window_length=12).run(repeat=False)
# ValueError: <MonthBegin> is a non-fixed frequency
```

Suggested fix: wrap the `.nanos` access (or test `index.freq.nanos` availability) and fall through
to the median-diff branch.

---

### [x] L2. `WindDirOffset` silently ignores its `hist_n_bins` argument

> **Fixed 2026-08-07.** Both histograms now go through one `_wind_histogram()` helper that
> honours `hist_n_bins`. The follow-on question raised in this entry — that `np.histogram`
> derives its edges from each subset's own range, so bin *i* was not the same direction in
> two histograms — is fixed at the same time: the edges are pinned to
> `linspace(0, 360, hist_n_bins + 1)`. Correlating bin-by-bin is only meaningful once they
> mean the same thing, so the two are really one fix.
>
> `_correct_degrees` now wraps with `% 360` instead of a single add/subtract: it handles any
> offset magnitude (the old form only covered one wrap) and maps an exact 360 to 0, keeping
> the domain the half-open [0, 360) a compass actually has.
>
> Covered by `TestWindDirOffsetCircularBinning` (coarse-bin recovery at 36/72/360 bins, an
> offset planted across north, output stays on the circle, wrap magnitudes, full-circle bin
> span). Mutation-checked: reinstating the hardcoded 360 fails 2 of them. The pre-existing
> `test_winddiroffset` still passes with its exact values unchanged.

`diive/preprocessing/corrections/offsetcorrection.py:476` vs `:495`

The per-year histogram hardcodes `n_bins=360`; only the reference histogram uses `self.hist_n_bins`.
With any `hist_n_bins != 360` the two `COUNTS` series have different lengths, and `Series.corr()`
aligns them on the RangeIndex — so the correlation is computed over a truncated, mismatched bin set
and the "best" offset is picked from garbage. No error is raised.

Suggested fix: pass `n_bins=self.hist_n_bins` at line 476. Consider also fixing the bin **edges**
across years (`np.histogram` with an int `bins` derives edges from each subset's own min/max, so
bin *i* is not the same wind-direction interval in two different years — it happens to work only
because wind direction spans ~0–360 in every year).

---

### [ ] L3. Frequency detection: off-by-one denominator breaks short series and mis-reports confidence

`diive/core/times/times.py:1386-1393`

```python
n_rows = df['delta'].size          # n rows, but only n-1 deltas exist
most_frequent_delta_perc = most_frequent_delta_counts / n_rows
if most_frequent_delta_perc > 0.50:
```

Three consequences, all **[reproduced]**:

| Input | Actual | Expected |
|---|---|---|
| Perfectly regular 100-record index | `percent_matching = 99.0` | `100.0` |
| Clean 2-record 30-min series | `RuntimeError: timestamps are too irregular` (1/2 is not > 0.50) | `'30min'` |
| 1-record series (allowed by `TimestampSanitizer._validate_input`) | bare `KeyError: 0` from `df['delta'].mode()[0]` | the informative `RuntimeError` |

`percent_matching` and `confidence` are public — `DetectFrequency.percent_matching` and
`TimestampSanitizer.get_status()['frequency_percent_matching']` both surface the wrong number.

Suggested fix: divide by `n_rows - 1` (or `df['delta'].count()`), and guard `mode()` against an
empty result so a 1-record input reaches the intended `RuntimeError`.

---

### [ ] L4. `keep_records_where`: an "open" bound is not open when `inclusive != 'both'`

`diive/core/dfun/frames.py:110-112`

```python
eff_lower = cond.min() if lower is None else lower
eff_upper = cond.max() if upper is None else upper
mask = cond.between(eff_lower, eff_upper, inclusive=inclusive)
```

The docstring says "an unset bound means that side is open (no limit)", but substituting the
observed min/max and then passing `inclusive='neither'` (or `'left'`/`'right'`) **excludes the
extreme record**.

**[reproduced]** — condition values `[10, 20, 30, 40, 50]`, `upper=40, inclusive='neither'`:

```
[nan, 2.0, 3.0, nan, nan]     # the record at the minimum (C=10) was dropped
```

Reachable straight from the GUI: **Select records by condition** exposes both the "use lower/upper"
checkboxes and the `inclusive` dropdown (`diive/gui/tabs/select_records.py:381`).

Suggested fix: use `-np.inf` / `np.inf` for the unset side instead of the observed extremes.

---

## Library — worth confirming against the reference implementation

### [x] L5. MDS cascade clips window positions instead of trimming them — edge records counted many times

> **Answered and fixed 2026-08-07 against the ONEFlux source.** ONEFlux **trims**, never clips.
> For methods 1 and 2 it narrows the bounds before looping (`common.c:2525-2533`:
> `if (window_start < 0) window_start = 0;` / `if (window_end > end_window) window_end =
> end_window;`, then `for (window_current = window_start; window_current < window_end; ...)`),
> and the diurnal method skips out-of-range positions outright (`:2630`:
> `if (((window_current+y) < 0) || (window_current+y) >= end_window) continue;`). Either way a
> real record enters a fill **at most once**, so diive's `np.clip` was a bias, not fidelity.
> `window_idx` now returns only in-range positions. Measured on the 40-day synthetic record from
> this entry:
>
> ```
>                   before (clip)              after (trim)
> gap at idx 2      fill -3.3071  sd 0.5381  count 453    fill -4.0003  sd 0.6638  count 120
> gap in middle     fill -5.2940  sd 0.7179  count 271    unchanged
> gap near end      fill -6.5064  sd 0.5293  count 490    fill -6.5465  sd 0.9359  count 157
> ```
>
> Note the SD as much as the mean: duplicates carry no spread, so the reported uncertainty at the
> edges was understated by nearly half. Interior fills are bit-identical, which is why the
> "~5e-7 vs native ONEFlux" validation never caught it. Covered by three tests in
> `test_uncertainty.py`; mutation-checked.

`diive/gapfilling/similarity.py:240-248`

```python
w = index + off
np.clip(w, 0, n - 1, out=w)
```

Near the start/end of the record every out-of-range offset collapses onto position `0` (or `n-1`),
and those duplicates flow into `nongap`, the similarity mask, the mean, the SD and `count`.

**[reproduced]** on a synthetic 40-day half-hourly record: a gap at position 2 reports
`count = 509` from a ±7-day window that contains only **338 distinct** records — the first records
are weighted ~300× in the fill.

**Decision needed.** The module header states this is a faithful port of ONEFlux's
`uncert_via_gapFill`, "validated to ~5e-7 against a native ONEFlux run on CH-DAV".

- If ONEFlux clips the same way → this is correct and should stay; add a comment saying so, because
  the code reads like an oversight.
- If ONEFlux filters out-of-range indices instead → fills in the first/last half-window are biased,
  and the largest cascade windows (up to 427 days, loop 6) push that band deep into the record.

Affects `FluxMDS`, `RandomUncertaintyPAS20` (shared kernel) and the daytime-partitioning NEE
uncertainty.

---

### [ ] L6. Random-uncertainty method 4 uses an asymmetric neighbour window

`diive/flux/lowres/uncertainty.py:708-710`

```python
start_ix = max(0, cur_ix - 5)
end_ix = cur_ix + 5
seg = randunc_sorted[start_ix:end_ix]   # exclusive stop -> 5 below, self, only 4 above
```

The docstring says "the fluxes closest in magnitude"; the slice is skewed toward lower fluxes.
`cur_ix + 6` makes it symmetric. Method 4 is a last-resort fallback (diive extension, not ONEFlux),
so the numeric impact is small — but it isn't what's documented.

---

## Library — contract / documentation mismatches

### [x] L7. Outlier flags never contain NaN — missing records come out flagged 0 ("valid")

> **Fixed 2026-08-07.** `FlagBase` gained a `nan_flag_at_missing` class attribute (default True);
> `repeat()` now masks `overall_flag` to NaN wherever the input record is missing. `MissingValues`
> sets it False — missing records are that detector's subject, so the mask would have erased its
> entire output. The seven wrong docstring claims were rewritten, `trim.py`'s (correct for the *old*
> behaviour, so newly wrong) was updated, and the outlier codegen comment now reads
> `# 0 = ok, 2 = outlier, NaN = input missing`. `FlagQCF` output is bit-identical before and after
> (verified), so the flux chain is unaffected. Resolves **L42**; see **L30** for what it does *not*
> resolve.

`diive/core/base/flagbase.py:182`

```python
overall_flag = iteration_flags_df[iteration_flags_df == 2].sum(axis=1)
```

An all-NaN row sums to `0`, so records that were **missing in the input** end up with flag 0. Every
detector module docstring claims otherwise — `zscore.py:13`, `hampel.py:21`, `localsd.py:24`, … all
say "NaN: Original missing data preserved".

**[reproduced]** — a series with a 5-record gap: `flag.iloc[10:15] == [0.0]*5`, and
`flag.isna().any()` is `False`.

Fed into `FlagQCF`, a missing record therefore contributes 0 to the flag sums and lands as QCF=0
("good quality"). The flux chain compensates via the separate `flag_missingvals_test`, so this bites
standalone detector use. Either make the behaviour match the docstrings (mask the flag where the
input is NaN) or fix the docstrings — currently they contradict the code in nine modules.

---

### [ ] L8. `FlagQCF`: "QCF is NaN if no flag is available" is unreachable

`diive/preprocessing/qaqc/qcf.py:640-645`, docstring at `:222`

`_calculate_flagsums` produces `0` (not NaN) for rows with no flags, so the `sumflags == 0` branch
immediately sets QCF=0. The NaN initialisation and the documented "or NaN if no flags available"
describe a state that cannot occur.

---

## Library — lower severity

### [ ] L9. `GridAggregator` builds its frame from a dict keyed by Series names

`diive/analysis/gridaggregator.py:119-123`

```python
self._df_long = pd.DataFrame({self.x_col_name: self.x,
                              self.y_col_name: self.y,
                              self.z_col_name: self.z})
```

If two of x/y/z carry the same name, one silently overwrites the other and the `BIN_*` column names
collide too. Harmless when the two roles are literally the same column (the common GUI case in the
3-D X/Y/Z surface tab), wrong when two *different* Series happen to share a name. `ScatterXY`
already guards exactly this with internal `_x`/`_y`/`_z` keys — apply the same pattern here.

### [ ] L10. `vectorize_timestamps` makes `.SEASON` a nullable `Int64`, forcing object-dtype arrays

`diive/core/times/times.py:1245` (`insert_season` returns `.astype('Int64')`)

One nullable-extension column turns the whole frame's `.to_numpy()` into **object dtype**
(**[reproduced]**). That array flows through `convert_to_arrays` into `model.fit` / `predict` /
`shap` on every ML gap-fill and every fallback fill. sklearn and XGBoost both accept it (verified),
so nothing breaks — but every run pays a hidden object→float conversion. A plain `int` cast (or
`.astype('float64')`) avoids it.

### [ ] L11. `JointUncertaintyPAS20`: cumulative scenario term is not masked to available flux

`diive/flux/lowres/uncertainty.py:867-871` — the random term uses `.where(flux.notna())`, the
scenario term does not. If the scenario columns have NaN where the flux does not (or vice versa),
the two cumulative terms sum over different record sets.

### [ ] L12. `LocalSD`: values exactly on the limit are in neither `ok` nor `rejected`

`diive/preprocessing/outlier_detection/localsd.py:279-282` — `ok` uses `<`/`>`, `rejected` uses
`>`/`<`, so an exact-limit record gets a NaN flag for that iteration (resolving to 0 via L7).
Same effective outcome as "ok", but the asymmetry is easy to misread.

### [ ] L13. `transform_yearmonth_matrix_to_longform` hardcodes the column names it drops

`diive/core/dfun/frames.py:644` — derives `rows`/`cols` generically but then
`drop(['YEAR', 'MONTH'])`, so any matrix not produced by `resample_to_monthly_agg_matrix` raises
`KeyError`.

---

## GUI — real bugs

### [x] G1. Partitioning tabs run with (0, 0) at UTC when the project site is unconfigured

> **Fixed 2026-08-07.** `BasePartitioningTab` gained a `needs_coords` property and a
> `_coords_missing()` guard, called from both `_run()` and `_python_code()` (the snippet would
> otherwise carry `lat=0.0, lon=0.0`). Message and behaviour mirror the existing guards in
> `_outlier_base.py:426` and `_correction_base.py:342`.
>
> **Correction to the original finding: it affected 3 of the 4 tabs, not all four.**
> `DaytimePartitioningOneFluxTab` declares no `needs_lat/lon/utc` because ONEFlux's daytime
> split is measured-`Rg` ≤ 4 / > 4 with no solar geometry — it never reads a coordinate.
> Regression test `test_partitioning_tabs_refuse_to_run_without_site_coords` covers all four
> (three must refuse, one must not) and was mutation-checked: with the guard removed it fails
> with `NighttimePartitioningOneFluxTab started a run`.

`diive/gui/tabs/_partitioning_base.py:227-236` and `:300-336`

`_seed_site()` returns early when `site.manager.configured` is False, leaving `lat`/`lon`/`utc` at
their `QDoubleSpinBox`/`QSpinBox` defaults of **0.0 / 0.0 / 0**. `_run()` validates the input
*columns* but never checks `site.manager.configured`, so all four partitioning tabs (Nighttime
ONEFlux/REddyProc, Daytime REddyProc/ONEFlux) will happily run at the equator on the Greenwich
meridian in UTC. Every port uses those coordinates for its day/night split (`sunrise_sunset`,
potential radiation), so the output is systematically wrong with no warning.

This is inconsistent with the rest of the GUI, which guards this exact case:

- `_outlier_base.py:426` — refuses to run, with an explicit "running now would silently split at
  (0, 0) at UTC and corrupt the result" message.
- `_correction_base.py:342` — same guard behind `needs_coords`.
- `derived_potrad.py:125` — `_compute` raises.

Suggested fix: add the same `configured` guard to `BasePartitioningTab._run` (and ideally to
`_python_code`, which currently emits a snippet with `lat=0.0, lon=0.0`).

---

### [ ] G2. `_outlier_base._on_done` can raise `KeyError` when the dataset changes mid-run

`diive/gui/tabs/_outlier_base.py:666`

```python
self._draw(self._df[payload["var"]], ...)
```

Detection runs on a background thread, so the GUI stays live: the user can create a feature, narrow
the date range, or delete/rename a variable while it runs. `on_data_loaded` (`:392`) then rebinds
`self._df` and may clear `self._var`. When the worker finishes, `_on_done` indexes the *new* frame
with the *old* variable name — `KeyError` inside a Qt slot if the column is gone, and a plot of a
differently-indexed series against the old flag/cleaned arrays if the range merely changed.

`_rerender_last` (`:691-694`) already guards exactly this (`p["var"] not in self._df.columns`); the
completion slot does not. `_screening_base` solves the general problem properly with a `run_id`
staleness check (`:710-712`, `:785`) — that pattern is the model to copy.

---

### [ ] G3. Pinned tabs are not actually frozen against added columns

`diive/gui/app.py:899` (`_add_features`) and `:1035` (`_sync_event_columns`)

```python
for col in new_df.columns:
    self._full_data[col] = new_df[col]     # in-place mutation
```

When no date range is active, `_apply_range` sets `self._data = self._full_data` — the **same
object** that was pushed to every tab. `_push_data` skips pinned tabs so they don't receive a new
frame, but they still hold a reference to the frame that is now being mutated, so newly created
columns leak into them. Column **drops** (`:1025`, `:1066`) rebind instead of mutating, so those do
*not* leak — the freeze is inconsistent in both directions, and it only applies at all when a date
range happens to be active.

Suggested fix: build a new frame in `_add_features` (`self._full_data = self._full_data.assign(...)`)
so every mutation rebinds, matching the drop path.

---

## GUI — lower severity

### [ ] G4. `restore_controls` silently keeps the current value when a saved combo entry is gone

`diive/gui/widgets/state_utils.py:51-57`

```python
i = w.findText(str(value))
if i >= 0:
    w.setCurrentIndex(i)
# else: keep whatever is selected — no signal to the user
```

Reopening a project whose columns were renamed (or whose preset labels changed) leaves the affected
tab pointing at a *different* input than was saved, with no indication. Affects every tab using
`save_controls`/`restore_controls` — notably the partitioning input pickers and the joint-uncertainty
**divisor** preset (`uncertainty_jointunc.py:62-63`), where a silent fallback to index 0 changes
`JOINT_DIVISOR_IQR` (1.349) back to `JOINT_DIVISOR_1SIGMA` (2.0).

Suggested fix: return a list of unrestored keys and surface it in the tab's status line.

### [ ] G5. `_screening_base._run` starts an unbounded number of concurrent worker threads

`diive/gui/tabs/_screening_base.py:729-733` — no `is_running` guard. Stale *results* are correctly
discarded via `run_id`, but rapid chain edits stack up CPU-heavy threads that all run to completion.
`WorkerRunner` (used by the other tabs) already has the guard.

### [ ] G6. `WorkerRunner` clears `is_running` before emitting

`diive/gui/widgets/worker.py:73` and `:80` — `self._running = False` precedes the queued
cross-thread `emit`, so `is_running` reads False during the window before the GUI thread processes
`done`/`failed`. Callers that use it as a re-entry guard (`_ml_gapfilling_base.py:708`,
`_partitioning_base.py:301`) can therefore start a second run whose result arrives interleaved with
the first. Also, `str(err)` is empty for some exception types, leaving a bare "Failed: " in the
status line.

### [ ] G7. `_compute_payload` writes tab state from the worker thread

`diive/gui/tabs/_outlier_base.py:463` — `self._live_is_daytime = ...` is assigned off the GUI
thread, in a method whose own call site comments that "the worker must not read live Qt widgets".
Single attribute assignment so it's GIL-safe in practice, but it contradicts the stated contract;
carry it in the returned payload instead.

### [ ] G8. `save_config` only catches `OSError`

`diive/gui/config.py:36-41` — a non-JSON-serializable value anywhere in the persisted blob (theme,
site, events, `variable_metadata`) raises `TypeError` out of `MainWindow.closeEvent`. All current
producers emit plain types, so this is latent, not active. Catching `(OSError, TypeError, ValueError)`
would match the module's stated "all failures are swallowed" contract.

### [ ] G9. Project load transiently materialises the *previous* session's event columns

`diive/gui/app.py:1372` vs `:1380` — `_set_data` calls `_sync_event_columns()` while
`events.manager` still holds the outgoing session's events, so their `EVENT_*` columns are created
on the incoming data and then replaced when the project's own events load eight lines later. Net
result is correct; the intermediate state is wasted work and briefly wrong.

---

## Round 2 — library: real bugs

### [x] L14. `combine_variables(keep_overlap_only=False)`: subtract/divide return the negation / reciprocal, not the surviving record

> **Fixed 2026-08-07 by removing the option, not by repairing it** (project owner's call).
> Combining two variables is defined only where both were measured; arithmetic is now always
> overlap-only. The sign flip was a symptom — even for the commutative methods a one-sided
> `A + B` returns `B`, which is not a sum but `B` wearing a sum's label. Substituting a value
> for a gap is a *gap-filling* decision, and `method='fillgaps'` already states it plainly,
> so no capability was lost.
>
> Because the fix costs records, the GUI now reports what it costs: *"750 values. 250
> record(s) dropped where only one variable was available (100 only NEE, 150 only RECO)."*
> — the split matters, since a large one-sided count usually means the two variables cover
> different periods rather than that the data are bad. `fillgaps` reports instead how many
> gaps it filled.
>
> Removed from the library signature, the codegen, the GUI checkbox, its state key and its
> provenance params; `CLAUDE.md`'s tab description updated. The old test pinned the buggy
> values (`sub == [-9.0, 2.0, -30.0, -36.0]`) and was replaced by one asserting NaN at every
> one-sided record, plus a GUI test for the loss reporting.

`diive/variables/utilities.py:73`

```python
fill_value = None if keep_overlap_only else identity
result = getattr(series1, op)(series2, fill_value=fill_value)
```

The docstring promises that with `keep_overlap_only=False` "a missing value is treated as the
operation's identity (0 for add/subtract, 1 for multiply/divide), so **records present in only one
input survive**". That holds for `add` and `multiply` (commutative with their identity) but **not**
for `subtract` and `divide` when the *left* operand is the missing one:

**[reproduced]** — `A = [nan, 2, nan, 4]`, `B = [5, nan, 8, 2]`, `keep_overlap_only=False`:

| method | result | records 0 and 2 (A missing, B present) |
|---|---|---|
| `add` | `[5, 2, 8, 6]` | 5, 8 — B survives ✔ |
| `multiply` | `[5, 2, 8, 8]` | 5, 8 — B survives ✔ |
| `subtract` | `[-5, 2, -8, 2]` | **−5, −8** — B *negated* ✘ |
| `divide` | `[0.2, 2, 0.125, 2]` | **0.2, 0.125** — B *inverted* ✘ |

User-visible through the GUI's **Combine variables** tab, where "keep overlapping only" is a plain
checkbox. Computing e.g. `NEE - RECO` with it unticked silently yields `-RECO` wherever NEE is
missing.

Suggested fix: either restrict `keep_overlap_only=False` to the commutative methods (raise for
subtract/divide), or fill only the *right* operand for the non-commutative ones, or — cheapest —
correct the docstring and warn in the GUI.

---

### [ ] L15. `flag_ssitc_eddypro_test` performs no conversion despite documenting one

`diive/preprocessing/qaqc/eddyproflags.py:490`

```python
ssitc_flag = Series(index=df.index, data=df[flagname], name=flagname_out)
```

The raw EddyPro column is copied verbatim. Its docstring says twice that the flag "is extracted from
EddyPro FluxNet output and **converted to DIIVE standard format (0=good, 2=bad)**". Every *other*
flag function in this module really does convert (via `_extract_and_convert_flag_from_multidigit`
or explicit thresholding); SSITC is the only pass-through.

Consequence: EddyPro's Mauder & Foken value **1** (intermediate quality) lands in `FlagQCF` as a
**soft** flag, so those records pass QCF as marginal rather than being rejected as the docstring
implies.

The pass-through is probably the *intended* behaviour — the `setflag_timeperiod` parameter exists
precisely so a user can promote 1→2 for chosen periods, and FLUXNET treats SSITC 0/1 as usable. If
so, **the docstring is the bug** and should say "values 0/1/2 are passed through; 1 becomes a diive
soft flag". Decide which, but they cannot both stand.

---

### [ ] L16. SWIN short-gap interpolation silently never fires near dawn/dusk

`diive/gapfilling/swin.py:794-799`

```python
missing = kt.isna()
run_length = missing.groupby((~missing).cumsum()).transform('sum')
accepted = missing & (run_length <= limit) & interp.notna() & series.isna()
```

`kt` (the clearness index) is NaN wherever `SW_IN_POT` is below the floor — i.e. through every
night. The run-length is computed over the whole series, so a daytime gap adjacent to dawn or dusk
is merged into the neighbouring night's NaN run and its `run_length` becomes hundreds of records.
The `<= limit` test then rejects it.

**[reproduced]** — synthetic 5-day half-hourly CH-DAV-like series, `interpolate_short_gaps=2`, two
identical 2-record gaps:

```
dawn gap interpolated?   [False, False]
midday gap interpolated? [True, True]
```

So the documented "2-record limit" is really "2-record limit, and only away from the day edges".
The docstring's stated exclusions are only "gaps that were too long or could not be anchored".
Given `'auto'` sets `limit=2`, this quietly removes most of the benefit at exactly the times of day
where SW_IN changes fastest. Either compute the run length over daytime records only, or document
the edge behaviour.

---

## Round 2 — library: contract / documentation mismatches

### [ ] L17. `MetadataStore.rename` silently drops an entry on a name collision

`diive/core/metadata/__init__.py:325`

```python
self._items = {md.name: md for md in self._items.values()}
```

Renaming `A` → `B` when `B` already exists collapses the two records into one (last wins) — the
metadata, provenance and tags of the loser are lost with no error. The GUI's single-variable rename
guards against this (`app.py:1084`), and the prefix/suffix tab renames every column simultaneously
so it cannot collide — but the library API is unguarded, and `MainWindow._rename_variables`
(`app.py:1091`) does not validate its `mapping` either, so a colliding mapping would additionally
give the DataFrame duplicate column labels.

Suggested fix: raise on a collision in `MetadataStore.rename`.

### [ ] L18. `FeatureEngineer`: the rolling stages re-engineer already-engineered columns, the others don't

`diive/core/ml/feature_engineer.py` — `_rolling_features` / `_rolling_features_advanced` select
`feature_cols` with no `.`-prefix filter, while `_differencing_features`, `_ema_features`,
`_polynomial_features` and `_stl_features` all exclude `.`-prefixed (already engineered) columns.

Running the engineer on a frame that already contains engineered columns — exactly what the GUI's
`Data ▸ Feature engineering` tab enables, since its output is merged into the dataset and shows up
in the picker on the next run — therefore produces names like `..Tair_f_POL2_MEAN12` from the
rolling stages only. Inconsistent, and it grows the feature count quadratically across repeat runs.

(Related, harmless: `_create_features` passes the *expanded* frame to the polynomial stage
(`:455`) while every other stage gets `df[self.original_input_features]`. The `.`-prefix filter
inside `_polynomial_features` neutralises the difference, so this is a readability wart, not a bug.)

### [x] L19. `features_stl=True` can silently produce nothing

> **Fixed 2026-08-07, after L47.** The blanket skip is gone: it existed because `stl_decompose`
> could not fit around a gap, which L47 settled. A gappy driver now gets its STL features, and
> those components are NaN in exactly the records the column itself already loses — measured on
> a 300-gap driver, zero records lost beyond the source, so the feature costs nothing in model
> rows. `'classical'` genuinely cannot decompose gaps (statsmodels raises); that column alone is
> skipped, and the message is now a `warn()` rather than a DEBUG line, plus a summary warning
> when *no* column could be decomposed. Verified: 4 STL columns instead of 3 under `'stl'` and
> `'harmonic'`, 3 under `'classical'` with the reason printed.

`diive/core/ml/feature_engineer.py:726` — `_stl_features` skips any column containing a single NaN,
and only says so via `detail()` (DEBUG level, verbose ≥ 3). Real flux drivers essentially always
have gaps, so a user who enables STL features at the default verbosity gets no STL features and no
visible indication. Promote the skip message to `warn()`, or gap-fill the driver before decomposing.

### [ ] L20. `lagged_variants`: the edge-fill is conditional but documented as unconditional

`diive/variables/temporal.py:461` — the shift-induced NaNs at the series edges are backfilled only
when the source column is **completely gap-free** (`n_missing_vals_before == 0`). For any real
driver with gaps they stay NaN, which then drops those rows from `model_df.dropna()` and demotes
them to the flag-2 fallback. The closing verbose message states unconditionally that the shift
"created gaps which were then filled with the nearest value".

The behaviour is defensible (don't fill genuine gaps) and is already documented as a *consequence*
in the SWIN class docstring; only the message here is wrong.

### [ ] L21. `crosscorr` omits dates instead of writing NaN for three of its early-outs

`diive/preprocessing/qaqc/detect_timestamp_shifts.py` — the `pot_sum < 100` and clearness-index
branches write `{'shift_minutes': nan, 'max_corr': nan}`, but the `(pot > 0).sum() < 5`,
`sun_up.sum() < 5` and `len(pot_arr) == 0` branches `continue` without writing anything. The result
frame therefore has *missing rows* for some days and *NaN rows* for others, so callers aligning it
to a full date index get holes where they expect NaN.

---

## Round 2 — library: lower severity

### [ ] L22. `MultiDataFileReader` raises `AttributeError` when every file is empty

`diive/core/io/filereader.py:320` — `data_df` stays `None` if every file raises `EmptyDataError`
(or the file list is empty), and `sort_multiindex_columns_names(df=None, ...)` then fails with
`'NoneType' object has no attribute 'columns'` instead of a clear "no readable data" error.

### [ ] L23. `sort_multiindex_columns_names` mutates the list it is iterating

`diive/core/dfun/frames.py:510` and `:516`

```python
for ix, col in enumerate(cols_list):
    if col[0].startswith('.'):
        cols_list.insert(0, cols_list.pop(ix))
```

`pop(ix)` + `insert(0, …)` leaves positions after `ix` unchanged, so nothing is skipped — but the
moved columns end up in **reverse** order at the front, in both the `priority_vars` block and the
dot-prefix block. Cosmetic (column ordering only), but the pattern is a trap for the next edit.

### [ ] L24. `interpolate.py`: a nested-quote f-string prints a literal `{limit}`

`diive/gapfilling/interpolate.py:143` and `:183`

```python
_console.print(f"\n{'Gap Analysis (limit={limit})':-^80}")
```

The `{limit}` sits inside a single-quoted literal *within* the f-string, so it is never
interpolated: the verbose report header reads `Gap Analysis (limit={limit})`.

Also in this module: `_calculate_gap_sizes` (`:19`) is dead — nothing calls it; the module reads
`GAP_LENGTH` off `GapFinder` directly. Flagging, not removing.

### [ ] L25. `_extract_and_convert_flag_from_multidigit` turns a scalar `0` code into NaN

`diive/preprocessing/qaqc/eddyproflags.py:47-52` — the float→string round-trip makes `0` become
`'0.0'`, so `str[1]` reads `'.'` and `to_numeric(errors='coerce')` yields NaN rather than flag 0.
Harmless in effect (NaN contributes 0 to the QCF sums, same as a 0 flag) and EddyPro normally writes
9-digit codes or `-9999`, so this is latent — but it is accidental rather than intended.

---

## Round 2 — GUI

No new GUI defects beyond G1–G9 were found in this pass. Areas re-checked and found sound:

- **`tabs/plotting.py` view preservation** — the restore loop is correctly gated on
  `len(prev_limits) == len(axes)` (`:674`), so changing the panel count cannot index out of range.
- **`tabs/fluxchain.py` level gating** — `_run_level` blocks on `self._reached < idx - 1` and
  `_finalize` rolls `_reached` back when an earlier level is re-run; the worker receives the
  container and levels are pure functions, so no shared-state mutation.
- **`tabs/meteo_screening.py` END→MIDDLE conversion** — resamples on a renamed *copy*, converts via
  `convert_series_timestamp_to_middle`, and picks a collision-free output name.
- **`tabs/select_records.py`** — the GUI mask chain and `select_records_to_code`'s emitted snippet
  agree for both keep and remove, including NaN-condition handling. (It does, however, reach L4:
  the "use lower/upper" checkboxes combined with the `inclusive` dropdown hit the open-bound bug.)
- **`tabs/combine_variables.py`** — correctly delegates to the library; the defect is L14, in the
  library function, not the tab.

## Round 3 — the previously unreviewed modules

Covered by four parallel reviewers on 2026-08-07: the USTAR detection family, self-heating /
time-lag, the `analysis` package + decomposition utilities, and `core/plotting` + the InfluxDB layer
+ the 3-D surface tabs. Every finding below carries a repro that was executed. Items marked
**[verified independently]** were additionally re-run by the lead reviewer against the source.

### USTAR detection (`flux/lowres/ustar*.py`, `storage_correction.py`)

**[x] L26. `_class_bounds` can emit a class with `end < start`, giving a NaN class mean**

> **Fixed 2026-08-07 against the C source.** Two deviations, not one. (1) The tie-extension loop
> was missing C's `if ( ustar_class_start == ustar_class_end )` branch (`ustar.c:713`), which
> adopts the next value and keeps walking when a tie run has overshot this class's nominal end;
> without it the class comes out inverted. (2) Inverted classes still arise in C, and there the
> mean is **0.0** — its accumulation loop `for (y = start; y <= end; y++)` simply never runs —
> whereas diive's cumulative-sum shortcut returned a *plausible* value (0.215 where C has 0.0)
> because the negative count and negative numerator cancel; only the exactly-zero-width case
> gave the NaN reported here. `_class_means` now treats `end < start` as empty, like `start < 0`.
> Verified against a line-by-line transcription of the C loop over five tie patterns (heavy
> 2-decimal ties, one long run, all-equal, no ties, 1-decimal): bounds identical, means identical
> to the C accumulation loop. On CH-LAE the thresholds do not move (0.5198 / 0.3259 / 0.5499 /
> 0.4964), matching this entry's own finding that no final threshold flipped.
`ustar_mp_detection.py:285` — the tie-extension loop advances `class_end` past tied values, but the
next iteration recomputes `class_end` without checking it is still `>= class_start`. A run of equal
u* values longer than one class width yields `(class_start, class_start-1)`; `_class_means` then
computes `0/0` → NaN (plus a numpy RuntimeWarning) although its docstring promises "empty classes ->
0.0, as in C". `_forward_mode` skips every candidate class whose 10-wide look-ahead window contains
the NaN, silently dropping up to 11 candidates. Triggered by u* reported at 2-decimal precision:
`degenerate classes: 3` on 6000 synthetic nighttime records. No final threshold flipped across 30
seeds, so the wrong *intermediate* value is confirmed but a wrong *result* is not.

**[ ] L27. `set_storage_to_zero=True` still requires the storage column — the exact case it documents**
`storage_correction.py:150` — the flag is documented for "fluxes where no storage profile is
available (e.g. H, LE)", and `run_level31`'s docstring repeats it, but `_detect_storage_var()` runs
unconditionally in `__init__` and `storage_correction()` opens with `self.df[[fluxcol, strgcol]]`.
Following the documented advice crashes: `KeyError: "['SLE_SINGLE'] not in index"`.

**[x] L28. The 3000-record minimum is enforced in `detect()` but bypassed by every bootstrap path**

> **Fixed 2026-08-07.** The check moved into `_night_valid_arrays()`, the one point every
> public entry goes through, so `detect()`, `bootstrap()` and `bootstrap_annual_samples()`
> now share it. Verified: at 1000 records all three refuse with the same message; at 6000 all
> three run.
>
> **This dragged in most of L29.** With the minimum enforced, `UstarBootstrapThresholds`'s
> blanket `except` turned the refusal into a silent all-NaN result — trading a confident wrong
> number for an undiagnosable one, which is not a fix. The worker now returns
> `(year, samples, reason)` and both the sequential and parallel branches report an empty
> window at `warn` level (always visible) with the reason attached:
>
> ```
> !   2020: no valid thresholds - ValueError: Insufficient data: 1600 records, need at least 3000
> ```
>
> Covered by `TestRecordMinimumIsEnforcedEverywhere` and
> `TestBootstrapReportsWhyAWindowFailed` (which also checks a mistyped column name is not
> reported as missing data). Mutation-checked: 4 tests fail with the shared check removed.
`ustar_mp_detection.py:561`, `:596` — `bootstrap_annual_samples()` / `bootstrap()` call
`_night_valid_arrays()` and `_compute_seasonal()` directly, skipping the `MIN_SAMPLES_PERIOD` gate
the class docstring advertises. `UstarBootstrapThresholds` takes that fast path, so
`run_level33_ustar_detection` emits a threshold from a record `detect()` refuses (1000 records:
`detect()` raises, `bootstrap()` returns 0.419).

**[x] L29. A failing bootstrap window is completely silent**

> **Closed 2026-08-07.** The remaining half was the unreachable `detail()`, and the root cause
> is general: `detail()` defaults to `verbose=VERBOSE_PROGRESS` (2) while its own `min_level` is
> `VERBOSE_DEBUG` (3), so a bare `detail(...)` can never print at any setting — and wrapping it
> in `if self.verbose >= 2:` (the house convention) hides that rather than fixing it. The
> bootstrap call now passes `verbose=self.verbose`, as do five more in the u\* files
> (`ustar_mp_detection.py` x3, `ustar_vekuri_detection.py` x3) that had the same defect. Verified:
> the per-season lines now print at `verbose=3`, where they never printed before. **A sweep found
> 25 more across the library — filed as L75.**

> **Mostly fixed 2026-08-07 alongside L28** (see there): the blanket `except` now carries the
> reason out and both execution branches report it at `warn` level, so an empty window no
> longer looks like a quiet NaN. **Still open:** the `detail()` at `ustar_bootstrap.py:232`
> ("running sequentially") is still called without `verbose=`, so it cannot print at any
> setting — the sequential and parallel branches therefore still announce themselves
> differently.
`ustar_bootstrap.py:39`, `:264` — the fast path wraps detector construction *and* all iterations in
`except Exception: samples = []`, and the only signal is `detail(...)` called without `verbose=`, so
it never prints at any level. A typo'd column name gives `CUT (pooled): p16=nan p50=nan p84=nan`
with no error; `run_level33_ustar_detection` then misdiagnoses it as "insufficient nighttime
records". (The sequential branch uses `detail`, the parallel branch at `:288` uses `info` — so the
two report differently.)

**[x] L30. NaN USTAR or NaN per-record threshold is flagged 0 (accepted), not rejected**

> **Fixed 2026-08-07.** u\* filtering is a *positive* test: a record is kept only where the
> measured turbulence can be shown to reach the threshold, so a record with no USTAR fails it.
> `_flagtests` now derives the rejected set as the complement of the passing set
> (`rejected = ~(ustar >= threshold)`) instead of testing `ustar < threshold`, which matched
> neither way against NaN. Masking the flag to NaN instead would **not** have worked:
> `FlagQCF._calculate_flagsums` sums only 1s and 2s, so a NaN flag is accepted downstream just
> the same. The NaN *threshold* half is now a validation error rather than a silent
> mass-rejection: `FlagMultipleVariableUstarThresholds.calc` raises if the reindexed threshold
> Series does not cover every record, which is what its docstring already required. Verified on
> a 6-record frame: flux present + USTAR missing now flags 2 (was 0); flux missing still flags
> NaN (L7); a threshold Series with two holes raises naming the scenario.
>
> **Confirmed against ONEFlux** (`nee_proc/src/dataset.c:4555` VUT, `:4840` CUT): the reference
> filter is `if (USTAR_VALUE < threshold) NEE = INVALID_VALUE`, and a missing USTAR *is*
> `INVALID_VALUE` (= `-9999`, `common.h:159`), which is below every threshold — so ONEFlux
> rejects exactly these records. diive accepting them was a real deviation, not a judgement call.
>
> **Scope note on the threshold-Series guard:** it protects a hand-built Series passed through
> the composable API. The in-chain VUT path never hits it — `run_level33_ustar_detection`
> (`levels/level33.py:536`) expands the per-*year* VUT table to per-record values and already
> falls back to the pooled CUT threshold for a year with none, so its Series spans the record
> by construction.

> **Partially addressed 2026-08-07 by the L7 fix.** `FlagBase` now masks the flag to NaN where the
> *flux* is missing, so that half is covered. The half that matters here is **not**: a record with a
> present flux but a NaN `ustar` (or a NaN per-record threshold) is still flagged 0 — verified:
>
> ```
>                      NEE  USTAR  flag
> 2023-01-01 01:00:00  3.0    NaN   0.0     <- turbulence unknown, still "accepted"
> 2023-01-01 02:00:00  NaN   0.50   NaN     <- flux missing, now correctly NaN
> ```
>
> Closing this needs the u\* flaggers to mask on `ustar.notna()` (and on the threshold Series) too,
> or to validate the threshold Series for NaN as their docstring already claims they require.
`ustarthreshold.py:142` — same root cause as **L7**: neither comparison matches, the record lands in
neither index, and `FlagBase.repeat`'s all-NaN row sums to 0. `FlagMultipleVariableUstarThresholds`
documents that the threshold Series "must … contain no NaN" but never validates it, so a
partially-covered VUT series silently passes low-turbulence data. The in-library caller
(`level33.py:531`) fills NaN years from CUT, so the chain is currently safe; the exposure is via the
public class.

**[ ] L31. `UstarThresholdConstantScenarios.calc(showplot=True)` crashes on pandas 3**
`ustarthreshold.py:337`, `:345` — `counts.div(counts[0])` uses positional `__getitem__` on a
label-indexed Series, removed in pandas 3 (the project pins 3.0+). `KeyError: 0`. Plotting is the
only purpose of this class.

**[x] L32. `UstarDetectionMPT` is publicly exported but half of it references attributes that never exist**

> **Removed 2026-08-07** (653 lines), with its `dv.flux` export. Nothing in the library, GUI,
> tests or examples used it, and `UstarMovingPointDetection` is the faithful ONEFlux port of the
> same algorithm, wired into the chain, the bootstrap wrapper and the GUI. Keeping a second,
> half-broken implementation that only *prints* its threshold invited someone to use it. A static
> pass confirmed the diagnosis before deletion: `set_yearly_thresholds`, `collect_year_results`
> and `collect_yearly_thresholds` between them read 11 attributes that are never assigned
> anywhere in the class. The removal made `info`, `warn`, `insert_season` and `potrad` unused in
> the module; those imports went with it.
`ustarthreshold.py:561` — it is in `diive/flux/__init__.py.__all__`. `set_yearly_thresholds`,
`collect_year_results` and `collect_yearly_thresholds` reference ten attributes that are never
assigned anywhere in the file; `collect_yearly_thresholds` also reads `yearly_thresholds_df` before
assignment. `run()` never calls the collectors, so `results_seasons_df`/`results_years_df` stay
all-NaN and the detected threshold is only *printed*, never stored — contradicting the class
docstring. `bts_results_df` is not reset in `run()`, so a second run mixes both runs' quantiles.

**[ ] L33. `UstarVekuriThresholdDetection.summary()` crashes before `detect()`**
`ustar_vekuri_detection.py:187` — `__init__` sets `results_ = {}` (a dict) while `summary()` does
`if self.results_.empty:` to print its "run detect() first" message, so the guard itself raises
`AttributeError`. `UstarMovingPointDetection` gets this right with `pd.DataFrame()`. Also, the
documented `bootstrap_stats_` attribute is never assigned (`:432`) — always an `AttributeError`.

**[x] L34. `annual_thresholds_` holds the sentinel `10.0` when detection fails**

> **Fixed 2026-08-07.** `detect()` stores the aggregate as it comes out of `_aggregate_annual`
> — NaN when no season yielded a threshold — so the attribute and `get_annual_thresholds()`
> now agree, and the accessor's sentinel-to-NaN conversion is gone with them. The sentinel
> stays where it belongs, inside the ONEFlux port's own comparisons. Verified on a synthetic
> record with constant USTAR (nothing detectable): all four seasons NaN, `annual_thresholds_`
> NaN, was 10.0. All consumers (`ustar_bootstrap`, the GUI detection tab, the examples) read
> through the accessor, so none of them saw the sentinel.
`ustar_mp_detection.py:520` — only the `get_annual_thresholds()` accessor converts
`THRESHOLD_NOT_FOUND` back to NaN. The attribute is documented in the class Attributes section with
no mention of the sentinel, so reading it directly after a failed detection yields 10.0 m/s — a
plausible-looking threshold that would filter out every record.

**[x] L75. 25 `detail()` calls across the library can never print**

> **Fixed 2026-08-07.** Threading `verbose=` through was not available: **24 of the 25 sites have
> no verbosity source at all** — no `self.verbose`, no `verbose` parameter — so closing it that
> way would have meant adding a parameter to 20+ public functions. Instead the helpers now
> default to `verbose=None`, meaning "use the module default", and that default is settable:
> `dv.set_verbosity(dv.VERBOSE_DEBUG)` / `get_verbosity()` (exported at top level). Behaviour at
> the default (PROGRESS) is unchanged — `detail()` still stays quiet, `info()` still prints — but
> the 25 lines are now *reachable*, which they were not at any setting before. An explicit
> `verbose=` at the call site still wins. The CLAUDE.md convention that produced the defect
> ("call helpers WITHOUT `verbose=` inside a guard") is corrected in the same commit.
⚠ **[found 2026-08-07 while closing L29]**

`detail()` (`core/utils/console.py:191`) defaults to `verbose=VERBOSE_PROGRESS` (2) but its own
`min_level` is `VERBOSE_DEBUG` (3), so **a bare `detail(msg)` prints at no verbosity setting at
all**. The house convention in CLAUDE.md — "when using `if self.verbose >= N:` guards, call
helpers WITHOUT `verbose=` inside the block" — is what produces the defect: the guard reads as if
it controls visibility, and the call silently refuses. Every one of these is a debug line the
author believed they had written:

| File | Lines |
|---|---|
| `flux/lowres/storage_correction.py` | 222, 223, 224, 225, 251, 276, 336 |
| `preprocessing/qaqc/meteoscreening.py` | 678, 866, 870 |
| `flux/lowres/selfheating.py` | 895, 1252, 1263 |
| `preprocessing/corrections/offsetcorrection.py` | 103, 470 |
| `core/io/files.py` | 169, 267 |
| `flux/fluxprocessingchain/levels/` | `level41.py:190`, `_init.py:198`, `_qcf.py:69` |
| `preprocessing/corrections/setto.py` | 73 |
| `preprocessing/outlier_detection/manualremoval.py` | 149 |
| `preprocessing/qaqc/eddyproflags.py` | 503 |
| `preprocessing/qaqc/qcf.py` | 239 |
| `core/io/db/base.py` | 47 |

The seven in the u\* files were fixed with L29; these are the rest. Two ways to close it: pass
`verbose=` at every call site, or give `detail()` a default that can actually fire. The second is
one line but changes what every existing call does, so it needs a deliberate decision — and the
CLAUDE.md convention should be corrected either way, since following it is what causes this.

**[ ] L35. USTAR docstring examples import from the wrong namespace**
`ustar_bootstrap.py:133`, `ustar_vekuri_detection.py:80` — `dv.UstarBootstrapThresholds`,
`dv.UstarMovingPointDetection`, `dv.UstarVekuriThresholdDetection` all live on `dv.flux`; the
snippets raise `AttributeError` as written.

**[-] L74. u\* filtering keeps the first high-turbulence record after a low-turbulence period**
⚠ **[found 2026-08-07 while cross-checking L30 against ONEFlux and Pastorello 2020]**

> **Won't fix — deliberate deviation, decided 2026-08-07.** ONEFlux is the stricter of the two
> and its reasoning is sound, but diive favours data availability here: the record is kept, and
> a user who cares about the flush burst can drop those records themselves. Documented as an
> explicit deviation in `FlagMultipleConstantUstarThresholds` (with the reason and the
> reference), and pointed at from `FlagSingleConstantUstarThreshold` and
> `FlagMultipleVariableUstarThresholds`, so it cannot be mistaken for an oversight and
> "corrected" later. Everything else in the u\* comparison follows ONEFlux, including L30.

`ustarthreshold.py:139` (`FlagSingleConstantUstarThreshold._flagtests`) — the flag is a pure
element-wise `ustar >= threshold`. FLUXNET/ONEFlux discards **one more** record than that: the
first half-hour *above* the threshold that follows a period below it. ONEFlux does it in both
filter branches, `nee_proc/src/dataset.c:4571` (VUT) and `:4859` (CUT):

```c
/* filter out also the first value after a low turbulence period (even if just one hh) */
if ( row < rows_count-1 ) {
    if ( !IS_INVALID_VALUE(datasets[dataset].rows[index+row+1].value[NEE_VALUE]) ) {
        datasets[dataset].rows[index+row+1].value[NEE_VALUE] = INVALID_VALUE;
```

and Pastorello et al. 2020 (Sci Data 7:225, *The FLUXNET2015 dataset…*) states the reason:
"removing also the first half-hour with high turbulence after a period of low turbulence to
avoid false emission pulses due to CO2 accumulated under the canopy."

So diive keeps precisely the records that carry the flushed sub-canopy CO2 burst, which biases
nighttime NEE (and therefore RECO, and the partitioned GPP) — in the one direction the rule
exists to prevent. It affects every u\* scenario, CUT and VUT, and both flagger classes.

ONEFlux also grades the removal in its NEE flags (1 = first record removed for low u\*, 2 = a
subsequent one, 3 = the high-u\* record removed at the end of the period); diive's flag is 0/2
only, so a fix has to decide whether to reproduce that granularity or just reject.

Everything else in the port was checked against the same reference and matches: the constants
(`ustar_mp/src/types.h` — 3000 / 160 / 100 / 0.5 / 0.2 / 1.0 / window 10 / night SW_IN 10),
`median_ustar_threshold` (`ustar.c:214`, excludes both INVALID and NOT_FOUND, even/odd median)
against `_detect_season`, and `forward_mode` (`ustar.c`, percentile check disabled) against
`_forward_mode` — same loop bounds, same `meanws(i+1+y, window)`, same
`fx_mean[i+y] >= mean*THRESHOLD_CHECK` plateau test, same `ustar_mean[i]` return.

---

### Self-heating and time lag (`flux/lowres/selfheating.py`, `timelag_analysis.py`, `hqflux.py`)

**[x] L36. `_gapfill()` drops every gap before gap-filling — the gap-fill is a no-op** ⚠ **[verified independently]**

> **Fixed 2026-08-07.** The `dropna()` now names the *driver* columns
> (`dropna(subset=drivers)`), so a record missing only the target — which is what a gap *is* —
> stays in the frame for XGBoost to fill. Measured on CH-LAE 2016-06..08: gaps after gap-filling
> 497 -> 0 (before: 497 in, 497 out). This also fixes the secondary effect noted here, since the
> lag/rolling features are no longer built across removed rows.
`selfheating.py:390` — `pd.DataFrame.from_dict(frame).dropna()` uses `how='any'` over all columns
*including the target* `FCT_UNSC`, so exactly the rows to be filled are deleted from the training
frame. XGBoost reports `Filling 0 missing records`, `fct_unsc_gf` comes back identical to
`fct_unsc`, and the console blames "expected at edges with insufficient drivers". With the
`.dropna()` removed, the same setup fills all 740 gaps. Secondary effect: the lag/rolling features
built afterwards span the removed rows, so they reach across arbitrary time jumps.

**[x] L37. The whole H2O / LE self-heating path should not exist — remove it, do not fix it**
`selfheating.py` (see the removal scope below)

> **Removed 2026-08-07.** `flux_type` / `FluxType`, `_calc_latent_heat_vaporization_j_umol` and the
> `self.lv` attribute, both `latent_heat_vaporization` parameters, the dead `_fct_for_opt` block, the
> `Lv` column and its unit conversion, and the `'LE' if …` prefix are all gone; `col_flux_corr` is
> now always `NEE_OP_CORR`. The `flux_type="CO2"` argument was dropped from all 9 example call sites
> (it had exactly one legal value left). Verified by a three-stage smoke test: `ScopOptimizer`
> recovers a planted scaling factor of 1.500 exactly on the CO2 path — the same input returned
> 0.0666 through the H2O branch, confirming the removed path was the broken one.

**Scientific verdict (project owner, 2026-08-07): there is no self-heating correction for LE. Whether
a Burba-type correction applies to the latent heat flux is an unresolved question in eddy covariance,
so diive must not offer one.** The `flux_type="H2O"` branch is therefore out of scope by definition,
regardless of whether its arithmetic is right — and it is not: `_fct_for_opt = fct_unsc *
latent_heat_vaporization` (`:862`) is assigned on both branches and **never read again** (the name
appears only on lines 862 and 865), so `:872` stores the *unconverted* term, the optimizer fits
µmol m-2 s-1 against LE in W m-2, `SF_MEDIAN` silently absorbs Lv, and `ScopApplicator.run()`
(`:1219`) multiplies by Lv a second time — a true SF of 1.500 comes back as 0.0666. That is
evidence the path was never exercised, not a defect worth repairing.

**The removal is low-risk — the H2O path is entirely unused:**

- Both examples pass `flux_type="CO2"` at all 9 call sites (`examples/flux/lowres/
  flux_selfheating.py`, `flux_selfheating_production.py`); neither ever constructs an H2O run.
- **No test references the module at all** — `Scop*` and `selfheating` appear nowhere under `tests/`.
- No GUI reference.
- Only `ScopApplicator` is exported (`diive/flux/lowres/__init__.py:12`, `:24`); `ScopPhysics` and
  `ScopOptimizer` are not in any `__all__`.

**Removal scope** (every site that exists only to serve H2O/LE):

| Location | What it is |
|---|---|
| `:105` | `FluxType = Literal["CO2", "H2O"]` — and every `flux_type` parameter/attribute it types (`:172`, `:833`, `:1166`) |
| `:230-231` | `self.lv = self._calc_latent_heat_vaporization_j_umol(ta=self.ta)` |
| `:460-475` | `_calc_latent_heat_vaporization_j_umol` (its docstring: "Needed for the correction of the latent heat flux LE") |
| `:841`, `:857-865` | `ScopOptimizer`'s `latent_heat_vaporization` parameter and the dead `_fct_for_opt` branch |
| `:1172`, `:1186`, `:1198-1200`, `:1216-1221` | `ScopApplicator`'s `latent_heat_vaporization` parameter, the `Lv` column, and the `if flux_type == "H2O"` unit conversion |
| `:1191` | `prefix = 'LE' if self.flux_type == 'H2O' else 'NEE'` → always `'NEE'` |
| `:17`, `:77`, `:186`, `:297`, `:791`, `:808`, `:844-846`, `:1066`, `:1153`, `:1175-1177`, `:1439` | Docstrings and console lines advertising LE / H2O support |

Note the module-level docstring at `:17` currently states the correction "can be applied to CO2
fluxes (NEE, µmol m-2 s-1) **and optionally to H2O fluxes (LE, W m-2)**" — that sentence is the
claim to retract.

Also worth recording while this module is open: **`selfheating.py` has no test coverage whatsoever**,
which is the common context for L36, L38, L39, L40, L41 and L45 below.

**[x] L38. Corrected flux silently becomes NaN wherever the correction term is missing**

> **Fixed 2026-08-07.** A record with a measured flux but no correction term is now carried
> through *uncorrected* instead of being deleted (`flux + FCT.fillna(0)`), and the new
> informational flag `FLAG_NEE_OP_CORR_ISCORRECTED` says which is which: 1 = corrected,
> 0 = carried through, NaN = no measured flux. `run()` warns with the count. Consistent with how
> diive treats this trade-off elsewhere (see L74): keep the measurement, state what happened.
`selfheating.py:1225` — `flux_corr = flux_openpath + FCT` propagates NaN, so a record with a valid
open-path flux but no correction term drops out of the deliverable rather than being carried through
uncorrected or flagged. Combined with L36 this deletes real measurements: 200 of 1000 records lost
in the reviewer's run.

**[ ] L39. `ScopApplicator` has an undocumented column-name contract; legal inputs raise `KeyError`**
`selfheating.py:1214` — `run()` looks up the hardcoded `'FCT_UNSC_gfRF'`, but `__init__` stores the
series under `fct_unsc.name`. Passing what `ScopPhysics.run(gapfill=False)` produces (`FCT_UNSC`) or
`physics.fct_unsc_gf` (`FCT_UNSC_gfXG`) raises. Independently, `:1250` passes `by=self.daytime.name`
into `merge_asof` while the right-hand table always names that column `DAYTIME`, so a day/night flag
named anything else raises before any correction happens.

**[x] L40. An unrecognised `correction_method_base` returns an empty result instead of raising**

> **Fixed 2026-08-07.** The `if/elif` chain has an `else` that raises `ValueError` naming the
> three valid options.
`selfheating.py:270` — the `if/elif/elif` chain has no `else`, so a typo leaves `fct_unsc` as the
empty placeholder from `__init__`. `run()` completes without warning and every consumer sees an
empty correction.

**[ ] L76. BUR06 uses a canopy aerodynamic resistance where Burba 2006 specifies a per-element one, and drops the retained fraction `fr`**
⚠ **[found 2026-08-07 while reading Burba et al. 2006 alongside 2008]**

`selfheating.py:471` / `:584` — Burba et al. (2006), *Correcting apparent off-season CO2 uptake…*
(AMS 2006), Eq. (8)/(9), gives the correction for already-WPL-corrected data as

```
Fc_new = Fc + fr * (Ts - Ta) / (ra * (Ta + 273.15)) * qc * (1 + 1.6077 rho_v/rho_d)
```

with **two** things diive does not have:

1. **`ra` is a per-element instrument resistance**, `ra = 7.4 * sqrt(d/U)` (Eqs. 10-11), with
   `d = 0.133 m` for the can and `0.042 m` for the ball — the resistance of the boundary layer on
   the instrument body. diive passes `ra = u/u*^2`
   (`variables/thermodynamic.py:19`), the canopy-scale momentum-transfer resistance. On CH-LAE
   these differ by a factor of ~6 (median 11.46 vs 1.87 s m-1).
2. **`fr`, the fraction of instrument heat retained in the optical path** (= H_P/H_I, estimated as
   the summed thermal boundary-layer thicknesses of can and ball over the 0.128 m pathlength).
   2006 introduces it because "the wind will remove much of the warmed air from the optical path,
   carrying away most of H without affecting the measurements". diive has no `fr`.

Neither is fatal in diive's pipeline, because `ScopOptimizer` **fits** a scaling factor (bounded
0-50) against a closed-path reference, and that fitted factor absorbs both the resistance
mis-scaling and the missing `fr`. But it means BUR06/JAR09 as implemented are *semi-empirical*:
the physics term sets the shape, the fitted SF sets the magnitude. That should be stated, because
the class docstrings present them as the published formulations, and because a user who runs
`ScopPhysics` **without** the optimizer gets a correction term whose absolute scale is not the
paper's.

Not to be confused with L46 (the dilution factor), which is settled: 2006 Eq. (8), 2008 Eq. (1)
and the LI-7500 poster all carry `(1 + 1.6077 rho_v/rho_d)`, and 2006 even quantifies it as
"typically very small, on the order of 0.5-2%" — diive measured +1.16 % on CH-LAE.

**[ ] L41. `ScopPhysics` documents an RF + MDV gap-fill that does not exist**
`selfheating.py:152` — the class docstring promises "a hybrid approach using Random Forest and Mean
Diurnal Variation (MDV)", `stats()` prints "-> Imputed (RF + MDV)" (`:311`) and the output column is
`FCT_UNSC_gfRF` (`:127`). The implementation is XGBoost-only with no MDV stage, and (L36) imputes
nothing. The name mismatch also means `physics.fct_unsc_gf` and `results_df['FCT_UNSC_gfRF']` carry
different `.name`s.

**[x] L42. `analyze_highest_quality_flux` counts missing records as valid**

> **Fixed 2026-08-07** as a consequence of L7 (the flag is now NaN at missing records, so
> `(flag == 0).sum()` counts only records that were actually tested), plus an explicit
> `n_measured` in the summary and rates expressed *per measured record* rather than per
> potential record — an unmeasured record was never a candidate for being an outlier.
`hqflux.py:251` — `n_valid = (flag == 0).sum()`, and per **L7** Hampel flags never-measured records
as 0. A run with 1519 of 3000 records containing data reports `Valid records: 2986 (99.5%)`. The
same numbers are returned in the public `summary` dict.

**[ ] L43. Default fringe-bin trimming can empty the time-lag histogram; the `IndexError` escapes the batch helpers**
`timelag_analysis.py:148` — `ignore_fringe_bins or [5, 10]` drops 5 leading and 10 trailing bins;
`Histogram(method='uniques')` produces `n_unique - 1` bins, so a TLAG column with few distinct lag
values leaves nothing and `peakbins[0]` raises a bare `IndexError`. `analyze_all_gases` /
`plot_all_gases` catch only `ValueError`, so the whole batch aborts — contradicting their
docstrings' "Failed analyses … print warnings but do not raise exceptions".

**[ ] L44. `TimeLagAnalysis` class docstring states three parameter facts the code contradicts**
`timelag_analysis.py:90` — (a) `ignore_fringe_bins` "Default: None" but the code defaults to
`[5, 10]`; (b) `zoom_margin` "Default: [0.5, 0.8]" but code and `__init__` docstring both say
`[0.5, 1.5]`; (c) `histogram_startbin`/`histogram_endbin` are documented as bin *indices* but are
compared against `BIN_START_INCL` (`:410`), i.e. they are lag values in **seconds**.

**[x] L45. `ScopOptimizer` silently drops classes with fewer than 10 valid rows**

> **Fixed 2026-08-07.** The threshold is now the named constant `MIN_ROWS_PER_CLASS`, skipped
> classes are collected in `skipped_classes` (daytime, class, complete records, total records)
> and reported by `run()` at `warn` level, spelling out that their records will take a
> neighbouring class's factor. Verified on a fixture whose top quartile keeps 3 of 100 complete
> records: `1 of 4 classes had fewer than 10 complete records ...  daytime=1 class=3: 3 complete
> of 100 records`.
`selfheating.py:904` — `if len(valid_bin) < 10: continue` emits nothing. `_assign_scaling_factors`
then resolves those records via `merge_asof(direction='backward')` to a neighbouring regime's SF,
and `stats()` prints no target bin count, so a missing bin is invisible.

**[x] L46. BUR08 omits the water-vapour dilution factor that BUR06/JAR09 applies**

> **Answered and fixed 2026-08-07 from the source papers** (supplied by the user:
> Burba et al. 2008, *Glob Change Biol* 14:1854-1876, and the LI-7500 poster
> *Additional Heat Flux Term in the WPL Correction for the Open-path Gas Analyzer*).
> The factor belongs to **every** method. In the paper the instrument-surface fluxes are
> *added to the ambient sensible heat flux* — Method 4, `S = rho*Cp*w'T'a + Sbot + Stop +
> 0.15*Sspar` — and that total `S` enters WPL equation (1), whose sensible-heat term carries
> `(1 + 1.6077 rho_v/rho_d)`. The poster states the already-WPL-corrected form diive implements,
> verbatim and with the factor:
>
> ```
> Fc_new = Fct + (Sbot + Stop + Sspar)/(rho*Cp) * (qc/Ta) * (1 + 1.6077 rho_v/rho_d)
> ```
>
> So BUR08 was missing it, and `correction_method_base` silently changed two things at once.
> Measured on CH-LAE 2016-2017: BUR08 mean correction term 0.528266 -> 0.534392 (+1.16 %),
> median 0.585076 -> 0.592756 (+1.31 %) — the ~1 % this entry predicted. One record fewer
> (24947 -> 24946): a record with no humidity cannot carry the factor, exactly as for the other
> two methods. Covered by `tests/test_selfheating.py`, which pins that all three methods scale
> with humidity by the same factor; mutation-checked.
`selfheating.py:614` vs `:639` — `_flux_correction_term_unscaled_bur08` lacks the
`1 + 1.6077*(rho_v/rho_d)` factor its sibling applies, so switching `correction_method_base` changes
both the surface-temperature model and whether the WPL-style factor is applied (~1% systematic
offset at typical humidity). Settling this needs the Burba 2008 source equation.

---

### `analysis` package and decomposition utilities

**[x] L47. STL silently returns an all-NaN decomposition when the input contains a single NaN** ⚠ **[verified independently]**

> **Fixed 2026-08-07.** `stl_decompose` now fits on a linearly interpolated copy
> (`limit_direction='both'`, so leading and trailing gaps are covered too — one of them is
> enough to poison the fit) and masks the gaps back out of all three components. A gap in the
> input is a gap in the output; no interpolated value is ever returned. The count is reported:
> an `info` line and `n_interpolated` in the result dict. An all-NaN series now raises instead
> of returning NaN components. Verified: one gap in 1460 gave `trend non-NaN: 0` before, 1459
> after, and the components still sum back to the measured records. Same pattern as the L54 fix.
`core/times/decomposition_utils.py:133` — `statsmodels.STL` has no NaN handling and propagates
rather than raising. One missing day in 1460 gives `trend non-NaN: 0 of 1460`, `seasonality_strength
= 0.0`, and `summary()` prints `nan ± nan` happily. This contradicts four separate docstring claims
("Robust decomposition for … series with gaps", "Input time series (may contain NaN)", "Handles
edge cases: … all-NaN sections", and the module header). It reaches `SeasonalTrendDecomposition`
(`seasonaltrend.py:301`), whose docstring repeats "may contain NaN". Gaps are the normal state of EC
data, so the default path for a real series yields a silent no-result. (The `classical` method at
least raises.)

**[x] L48. `harmonic_analysis` reads the amplitude/phase from the wrong FFT bin (off by one)** ⚠ **[verified independently]**

> **Fixed 2026-08-07, together with L49.** Not originally in scope, but the two are entangled
> in this function: correcting the amplitude scale while still reading the neighbouring bin
> leaves it returning 0.0, so the L49 fix would have been unobservable here. The full-rfft bin
> number is now converted to the DC-stripped index (`bin_full - 1`), and `actual_frequency`
> reports the bin actually read.
`analysis/harmonic.py:89` — `idx = int(np.round(target_freq * n))` is an index into the *full* rfft
array, but it indexes `amplitudes`/`phases`/`power`, which were built as `fft_vals[1:]` (DC
removed). Every harmonic is read one bin high. Confirmed on a pure cosine of amplitude 3.0 sitting
exactly on bin 20:

```
target_f 0.02000  actual_f 0.02100  amplitude 0.0000      <- reported
full-rfft amplitude at bin 20 (the true one): 3.0
full-rfft amplitude at bin 21: 0.0
```

The top-level `'amplitudes'` array in the same result dict *is* correctly aligned, so the harmonics
list and the spectrum disagree with each other.

**[x] L49. Windowed FFT amplitudes are never corrected for the window's coherent gain**

> **Fixed 2026-08-07.** `harmonic_analysis` and `harmonic_decompose` now divide by the
> window's coherent gain (its mean), so a reported amplitude is the signal's rather than the
> windowed signal's. Verified: a 3.0 cosine returns 3.0 under boxcar, hamming, hann and
> blackman. `fft_decompose`, the third site, was deleted as dead code.
`analysis/harmonic.py:70`, `decomposition_utils.py:266` — `harmonic_analysis`, `fft_decompose` and
`harmonic_decompose` all apply a window (default `'hamming'`, mean ≈ 0.54) and then report
`2*|rfft(x)|/n` without dividing by the window mean. Every amplitude is ~0.54× the truth, the
reconstruction is 54% of the signal, and `residual = original − reconstructed` is dominated by the
missing 46% rather than by reconstruction error. `fft_decompose` reports 1.62 for a true 3.0;
with `window='boxcar'` it returns exactly 3.0 and a residual of 1.8e-14.

**[x] L50. `quality_weighted_decompose` and `stl_decompose(weights=...)` ignore the weights entirely**

> **Resolved 2026-08-07 by removing the fake path** (user's call, offered against keeping the
> API honest or inventing a threshold rule). statsmodels' STL takes no observation weights and
> there is no way to inject them, so the feature could not be delivered — and nothing in the
> library, GUI, tests or examples used it. Gone: `weights` from `stl_decompose`,
> `quality_weighted_decompose` entirely, and `quality` / `quality_weighted` from
> `SeasonalTrendDecomposition` along with the "Quality-weighted:" summary line. `robust=` is
> the real outlier knob and stays. **Breaking change** — needs a changelog entry.
`decomposition_utils.py:100` — `weights_norm` is computed and never used, the STL fit takes no
weights, and `:152` returns the *raw* weights rather than the normalised ones. The docstrings
promise "High-quality observations influence the fit more"; `SeasonalTrendDecomposition.summary()`
prints `Quality-weighted: True` for a run in which no weighting happened. With 100 records corrupted
by +40 and flagged quality 0, weighted and unweighted output are byte-identical.

**[x] L51. `StratifiedAnalysis` silently discards z-bins whose rounded label collides**

> **Fixed 2026-08-07.** The label is now built once for all bins by `_z_bin_labels`, which
> widens the rounding (2 decimals, then 3, …) until every label is distinct, and falls back to
> a positional suffix for bins that agree to nine decimals. Verified on 60 quantile bins over a
> z range of 0.1, where the medians sit ~0.0017 apart: 60 bins in, 60 stored (before: 11).
> The legend keeps reading in real units — the extra decimals appear only when they are needed
> to tell two bins apart.
`analysis/decoupling.py:213` — per-bin results are keyed by `f"{median:.2f}"`, so two adjacent
quantile bins that round to the same 2 decimals overwrite each other. 120 z-groups iterated → 101
stored, **19 silently lost**, with no warning (the existing `warn` covers only x-bin failures) and a
plot legend that blames "not generated" classes. Guaranteed with `conversion='percentile'` at large
`n_bins_z` (the permitted range reaches 120, where bin spacing 1/120 is below the rounding
resolution).

**[x] L52. `StratifiedAnalysis` listwise-drops rows on *every* column of the input frame**

> **Fixed 2026-08-07.** The three analysis variables are extracted *before* the `dropna`, so
> only gaps in `zvar`/`xvar`/`yvar` remove a record. What is kept is no longer silent: the
> constructor stores `n_records_input` / `n_records_used` and prints how many records were
> dropped and which three variables decided it. The on-plot count text is therefore about the
> data rather than about an unrelated column. The narrower frame also keeps `conversion` from
> z-scoring columns nobody asked about.
`analysis/decoupling.py:68` — `df.copy().dropna()` drops any row with a NaN anywhere in `df`, not
just in `zvar`/`xvar`/`yvar`. The documented input is "df: Dataframe with variables", so passing the
working dataframe — the obvious call — silently reduces the analysis: 20 000 rows → **100** with one
unrelated gappy column present. The on-plot count text reports the survivor count as if it were the
data.

**[x] L53. `CompoundExtremes` returns zero classified periods for a single year, silently**

> **Fixed 2026-08-07.** An empty classification now raises instead of being returned as a result
> that is indistinguishable from "no extremes occurred", and the message names the cause and the
> way out: seasonal standardization compares each calendar month against the same month in other
> years, so it needs at least two years — the error reports how many the record actually spans
> and points at `standardize_by='record'`. A *partial* loss no longer passes unmentioned either:
> the count of unclassifiable periods is reported at `warn` level and the rest are still
> returned. Verified on CH-DAV: one year raises, the same year with `standardize_by='record'`
> classifies all 12 months, five years classify all 60, and a 13-month slice keeps 2 periods
> while warning about the 11 it dropped. The GUI tab already wraps construction in try/except,
> so the message lands in its status line.
`analysis/compoundextremes.py:168` — with the documented defaults (`agg='monthly'`,
`standardize_by='season'`) the z-score groups by calendar month; one year gives each group a single
member, so `transform('std')` (ddof=1) is NaN and every row is dropped. `results` is empty and
`counts` all zeros, with no warning — indistinguishable from "no extremes occurred". `season` is
documented as "the default and the standard choice" and the GUI exposes it.

**[x] L54. `DriverAnalysis(deseasonalize=True)` silently linear-interpolates every gap**

> **Fixed 2026-08-07.** `_stl_components` still interpolates — statsmodels' STL has no NaN
> handling, so there is no way to fit without it — but the interpolated positions are masked
> back to NaN in all three components before they leave the function. A gap in the input is a
> gap in the output, so `_build_matrix.dropna()` removes those rows again and no fabricated
> record reaches a model or the chronological hold-out. That covers all three call sites at
> once (`_apply_deseasonalize`, `scale_resolved`'s STL components, `granger`), which is why
> the fix went into the helper rather than into `_apply_deseasonalize`.
>
> On the example CH-DAV month used by `analysis_driveranalysis.py` (1488 records, 579 measured
> after QCF filtering), `deseasonalize=True` and `deseasonalize=False` now both build a
> 579-row matrix. Covered by `TestDeseasonalizeKeepsGaps`; mutation-checked (both tests fail
> with the mask removed). `granger()` is unaffected in shape — `GrangerCausality` drops NaN
> itself — it just no longer tests interpolated values.

`analysis/driveranalysis/driveranalysis.py:80` — `_stl_components` opens with
`series.interpolate(limit_direction='both')`, and the interpolated values survive into
`trend + resid`; `_apply_deseasonalize`'s `reindex` (`:288`) is a no-op because the index is already
full. Target and drivers come back gap-free, so rows `_build_matrix` would have dropped enter the
model with fabricated target values — including the chronological hold-out that `fit_model` scores.
A 4-day gap (200 NaN) comes back with 0 NaN. `granger()` (`:774`) shares the path. The constructor
documents `deseasonalize` only as "STL-deseasonalize target and drivers up front".

**[x] L55. Per-regime and per-scale relevance is judged against the *global* model's `.RANDOM` floor**

> **Fixed 2026-08-07.** `_fit_importance` now returns `(importance, random_baseline, extras)`
> instead of discarding the baseline; `scale_resolved` and `stratified` keep it per column
> (`_scale_baselines` / `_stratified_baselines`), and `_temporal_fields` judges each column
> against the floor of the fit that produced it via the new `_relevances_vs_own_floor`. A
> column with no recorded floor still falls back to the headline model's.
>
> The per-submodel floors are not a detail: on the CH-DAV example month they span
> `stl_trend 0.013 … daily 0.392` against a global `0.246`, so individual labels do move —
> SW_IN at `stl_trend` reads `no` against the global floor and `yes` against its own. The two
> day/night floors happened to land within 1% of each other on that dataset, so no regime
> verdict changed there; the reviewer's four-season repro is the case where it does.
>
> Covered by `TestPerSubmodelNoiseFloor`; mutation-checked (the relevance test fails when the
> lookup is replaced by the global floor).

`analysis/driveranalysis/driveranalysis.py:760` — `_fit_importance` discards the `random_val`
`_shap_per_driver` returns (`imp, _ = ...`), and `_temporal_fields` (`:896`, `:902`) compares each
throwaway model's importances against `self._random_baseline` from the headline model. Each submodel
is fitted on a different subset with its own noise scale, so the comparison is apples-to-oranges —
and it decides the headline verdict, since `regime_dependence` is the sole trigger for
`verdict='context_dependent'` (`:1043`). Measured: against the global floor VPD is "yes" in all four
seasons (`regime_dependence=False`); against each model's own floor it is weak/weak/yes/yes
(`regime_dependence=True`).

**[x] L56. `harmonic_decompose` returns `frequencies` one element longer than the arrays it pairs with**

> **Fixed 2026-08-07.** `frequencies` now drops the DC bin, so it pairs element-wise with
> `amplitudes`/`phases`/`spectrum` as documented; the docstring states the pairing explicitly.
`decomposition_utils.py:307` — `frequencies` is the full `rfftfreq(n)` (`n//2+1`) while
`amplitudes`/`phases`/`spectrum` exclude DC (`n//2`). The docstring presents them as a matched set,
so `plot(frequencies, amplitudes)` raises. `harmonic_analysis` prepends a zero to fix exactly this —
the two functions disagree.

**[ ] L57. `reconstruct_from_components` forces the trend's NaN onto reconstructions that exclude the trend**
`decomposition_utils.py:420` — `result[trend.isna()] = np.nan` runs unconditionally, ignoring
`components_to_use`. A seasonal-only reconstruction from a classical decomposition is blanked at the
`(period-1)//2` edges where the trend is NaN by design, even though the seasonal component is fully
defined there (30 NaN in the output, 0 in the seasonal input).

**[x] L58. `detect_seasonality` fabricates a period of 365 when the periodogram yields nothing**

> **Fixed 2026-08-07.** The fallback is gone: with no candidate period in `[2, max_period]`
> the function raises `ValueError` naming the range, the number of valid values, and the way
> out (pass `seasonal_period` explicitly). A failed detection can no longer be mistaken for a
> successful one, and `_get_seasonal_period` no longer decomposes a 5-point series at period
> 365. The "no peaks -> use max power" branch is untouched; that one is a real answer.
`decomposition_utils.py:490` — with no frequency bin in `[2, max_period]` it returns
`primary_period=365`, `secondary_periods=[7, 30]` and `strength=0.0`, with no warning.
`SeasonalTrendDecomposition._get_seasonal_period` (`seasonaltrend.py:342`) calls it whenever
`seasonal_period` is not supplied, so a short series is decomposed at period 365 with no signal that
detection failed (`n=5 -> 365, [7, 30], 0.0`).

**[x] L59. `multi_scale_harmonics` swallows every exception and returns an empty result**

> **Resolved 2026-08-07 by deleting the function.** It had no caller anywhere — library, GUI,
> tests or examples — and was not in `dv.analysis.__all__`. Removed along with
> `reconstruct_harmonics`, `periodogram` and `fft_decompose` for the same reason. The dead
> `signal.hamming` fallback went with them; the two surviving functions now let
> `signal.get_window` raise on a bad window name instead of masking it.
>
> Checked before removing: the GUI spectrogram tab calls only `dv.analysis.spectrogram`, which
> goes straight to `scipy.signal.spectrogram`. The only internal edge was
> `multi_scale_harmonics` -> `harmonic_analysis`, i.e. dead calling live.
`analysis/harmonic.py:432` — a bare `except Exception: continue` drops any failed period silently.
Related: the `signal.hamming` fallback at `:68` / `:330` is dead on scipy ≥ 1.13 (the attribute no
longer exists), so a bad `window` name raises `AttributeError` instead of falling back, every period
fails, and the caller receives `{'scales': []}` as if the analysis had run and found nothing.

**[ ] L60. Two more docstring claims the code does not honour**
`seasonaltrend.py:174` — `seasonality_strength` documents "Ratio of seasonal variance to total
variance of **trend** + residual" but computes `var_seasonal / (var_seasonal + var_residual)`; the
trend is not involved. `decomposition_utils.py:59` — the returned `'iterations'` is `decomp.nobs`
(an observation count, a shape tuple on the installed statsmodels), printed as `iterations=` by the
verbose message.

---

### `core/plotting`, InfluxDB layer, 3-D surface tabs

**[x] L61. `HeatmapYearMonth` draws month cells at the wrong positions when months are not contiguous** ⚠ **[verified independently]**

> **Fixed 2026-08-07.** `_prepare_data` reindexes the matrix onto the complete year x month
> lattice (months 1-12, years min..max) before `_set_bounds` turns the labels into pcolormesh
> boundaries. The Nov-Feb repro now gives 12 cells each exactly one month wide against 12
> ticks, with the unobserved months NaN. Covered by `TestHeatmapYearMonthLattice`;
> mutation-checked (3 of its 4 tests fail with the reindex removed).
`core/plotting/heatmap_datetime.py:395` (`_set_bounds`) and `:527` (ticks) — `_set_bounds` uses the
surviving month labels directly as `pcolormesh` **boundaries** and appends only `last+1`, so a
record that skips months draws each cell all the way to the next surviving label. Reproduced on
Nov 2019 – Feb 2020:

```
plotdf columns : [1, 2, 10, 11, 12]      <- non-contiguous
x boundaries   : [1, 2, 10, 11, 12, 13]  <- cell "2" spans x=2..10, eight months wide
z shape        : (2, 5)
xticks         : 12 ticks labelled '1'..'12'
```

February's mean is painted across the region labelled March–September. `HeatmapDateTime` is safe
(`TimestampSanitizer(regularize=True)` guarantees a complete lattice); `HeatmapXYZ._prepare_data`
(`heatmap_xyz.py:269`) shares the construction — see L63.

**[ ] L62. `show_less_xticklabels` is accepted and documented by `HeatmapDateTime` but never applied** ⚠ **[verified independently]**
`core/plotting/heatmap_datetime.py:245` — the parameter is documented, forwarded to `super().plot()`
and stored (`heatmap_base.py:304`), but only `HeatmapYearMonth` reads it (`:537`). Label visibility
is identical with `True` and `False`. The GUI exposes the same checkbox for both heatmap types and
`heatmap_datetime_to_code` emits it into copied snippets, so the control silently does nothing.

**[x] L63. Missing equal-width bins are dropped, so empty regions of an X/Y/Z surface render as measured cells**

> **Fixed 2026-08-07 in `GridAggregator`**, so the x/y/z heatmap and the 3-D surface are both
> covered. Each binning method now records every label it defined (not just the occupied
> ones) and `_transform_and_pivot` reindexes onto that for **all** binning types — the
> reindex previously ran for `'custom'` only, yet `min_n_vals_per_bin` can empty a bin of any
> type. The bimodal repro now returns the full 30x30 grid with the void preserved as 25 empty
> columns and evenly spaced labels, and the 150 aggregated values unchanged.
>
> One trap worth recording: the expected labels must be read off the *categories* of the
> `pd.cut` result, not off its `retbins` edges. `pd.cut` rounds the edges it labels with
> (`precision=3`), so `retbins` gives near-misses (`0.046955` vs the label `0.047`) that match
> nothing on reindex — which silently blanks the entire grid. Caught by the repro, not by the
> tests, which is why the first attempt looked plausible.
>
> Covered by `TestEmptyBinsArePreserved`; mutation-checked (2 of its 4 tests fail when the
> reindex is restricted to `'custom'` again).
`gui/tabs/surfacexyz.py:129`, root cause `analysis/gridaggregator.py:429` — `_transform_and_pivot`
reindexes onto the full bin lattice only for `binning_type == 'custom'`; for `equal_width` (what the
tab uses) `pivot_table` emits only occurring labels and `dropna=True` removes all-NaN bins.
`surfacexyz` then treats the survivors as cell *centres*, and `_cell_edges` widens each to the
midpoint of its neighbour — filling the gap with real data. Bimodal X (500 points in 0–10, 500 in
95–100, `n_bins=30`) collapses to 5 columns; the cell centred on 6.71 spans 5%–52% of the plot
width. This also defeats `_drop_gap_risers = True` (`surfacexyz.py:44`, commented "keep them truly
empty") because those bins are never NaN cells. `HeatmapXYZ` has the same exposure.

**[ ] L64. glTF / `.glb` export bakes the texture mirrored along the date axis**
`gui/tabs/surface3d.py:744` and `:817` — UV rows are assigned `v = i/(d-1)` (top-left texture
origin) but trimesh/glTF use a **lower**-left origin, so vertex row `i` samples texel row `d-1-i`.
Geometry is right, colours are flipped: an exported annual NEE surface paints the winter ridge with
the summer colours. Verified against the installed trimesh 4.12.2 with a monotone 4×2 grid — the
sampled row colours came back exactly reversed. Both the smooth and extruded paths are affected;
`u` is correct.

**[ ] L65. `RidgeLinePlot` cannot plot any series that contains a gap**
`core/plotting/ridgeline.py:196`, `:234` — `np.array(series)` goes straight into
`KernelDensity.fit`, which rejects NaN. Every gappy time series raises
`ValueError: Input X contains NaN`. The GUI path works only because `ridgeline_to_code` and the
plotting tab `.dropna()` first (`codegen.py:248`), so the library's public API is strictly worse
than the GUI's.

**[ ] L66. `datetime_surface_grid` does not force `TIMESTAMP_START`, so the 3-D surface is offset from the 2-D heatmap**
`core/plotting/surface_grid.py:68` — the docstring claims "the same preparation the 2-D heatmap
uses", but `HeatmapBase._setup_timestamp` additionally converts to `TIMESTAMP_START` via
`insert_timestamp` (`heatmap_base.py:145`) while this only calls
`TimestampSanitizer(output_middle_timestamp=False)`, which leaves a `TIMESTAMP_MIDDLE` index alone.
Since MIDDLE is diive's working convention, the surface's time-of-day axis sits half a period later
than the heatmap's for the same data (`heatmap hours [0, 0.5, 1.0]` vs `surface [0.25, 0.75, 1.25]`).

**[ ] L67. 3-D export buttons write a stale surface after a render that produced nothing**
`gui/tabs/surface3d.py:945`, `:915` — `_render_surface` returns at `if not finite.any():` *before*
assigning `_grid_xn/_yn/_height/_z/_style` (`:963`), and `_compute` returns early when
`_grid_data()` is None; neither clears the previous grid. The export handlers guard only on
`self._grid_height is None`. Render a normal variable, then select an all-NaN one: the canvas
clears but "VR (.glb)" / "3-D print (.stl)" still write the *previous* variable's relief under a
filename built from the *current* target.

**[ ] L68. `datetime_surface_grid` silently destroys a variable named `DATE` or `TIME`**
`core/plotting/surface_grid.py:71` — `df["DATE"] = df.index.date` overwrites the data column when
`series.name` is one of those. The crash (`TypeError: float() argument must be … not
'datetime.time'`) is the lucky outcome; the defect is the silent overwrite.
`HeatmapDateTime._prepare_data` (`heatmap_datetime.py:96`) has the identical construction.

**[ ] L69. `Cumulative.plot` raises on an all-NaN column**
`core/plotting/cumulative.py:327` — `series.dropna().iloc[-1]` inside an f-string label indexes an
empty Series when a column has no valid values (normal for a scenario column that was never filled).
Same pattern at `cumulative.py:186` in `CumulativeYear`. `IndexError: single positional indexer is
out-of-bounds`.

**[ ] L70. Rolling cell aggregator uses an `n+1`-row window for even `n`**
`gui/tabs/surface3d.py:98` — `z[max(0, i-half) : i+half+1]` with `half = n//2` gives five rows for
`n = 4`. The docstring says "a centred rolling window of `n` rows" and the tooltip says "the window
width"; every even setting of the "Y cell (days)" spin box smooths one day wider than requested.

**[ ] L71. `convert_ts_to_timezone` cannot accept the `DatetimeIndex` its docstring promises**
`core/io/db/influx/common.py:59` — the body calls `.dt.tz_convert(...)`, which exists only on a
Series. The one in-tree caller passes a Series, so this is documentation-only — but the function is
exported from the influx package and reads as index-safe.

**[ ] L72. Open question: does InfluxDB v2's delete range exclude `stop`?**
`core/io/db/influx/influxio.py:122` — `upload_singlevar`'s pre-upload delete uses
`stop = str(var_df.index[-1])`. If the delete range is stop-exclusive (as the Flux *query* range is,
which the code documents), the final record of the range survives and the delete does not achieve
its stated purpose of avoiding duplicates when only a tag changed. No live database was available;
settling this needs one, or the InfluxDB v2 delete-API docs.

---

## Reviewed and found sound (no action)

Recorded so a future pass doesn't re-derive these:

- **No `pyplot` anywhere in `diive/gui/`** — the GUI↔library separation on the plotting side holds;
  all embedded figures go through `MplCanvas`'s explicit `Figure` + `FigureCanvasQTAgg`.
- **PySide6 holds only a weak reference to a bound-method receiver**, even when the receiver is a
  plain (non-`QObject`) `DiiveTab`. **[verified]** — so the ~15 `theme.manager` / `site.manager` /
  `metadata_store.manager` connections made by closable tabs do *not* leak the tab or fire zombie
  slots after close. No teardown needed.
- **`potrad`** (`diive/variables/radiation.py`) — window-mean accumulation, day indexing, solar-noon
  shift and New-Year straddling all check out.
- **`_rerun.py` cascade logic** — level ordering, additive-level handling and `filteredseries`
  restoration are consistent.
- **`_screening_base`'s `run_id` staleness protocol** — the correct pattern; see G2.
- **`load_parquet_many` / `MultiDataFileReader` merging** — `combine_first` preserves both column
  order and `index.freq` on current pandas (**[verified]**), so a multi-file load is not silently
  reordered or de-frequencied as older pandas would have done.
- **`neighboring_years`** (long-term gap-filling) — the year-pool selection matches its documented
  example for the first, middle and last years, and for 1- and 2-year records.
- **`potrad`-driven `_partitioning_base` inputs, `_ml_gapfilling_base` completion payload** — the ML
  tab carries `observed` in the payload rather than re-indexing `self._df`, so it does not have G2's
  problem.
- **`FeatureEngineer` rolling/EMA windows are past-only** (`center=False`), so no look-ahead leaks
  into the gap-filling features.
- **`swin.py` `_interpolate_short_gaps` protects observations** — `& series.isna()` in the accept
  mask means interpolation can never overwrite a measured value (the defect is L16, coverage, not
  corruption).

Round 3 additions (checked by the parallel reviewers, no defect found):

- **USTAR**: `_month_per_group` end-of-period shift (incl. the January→December wrap); `_pearson`'s
  zero-denominator guard and the correlation being taken on raw class records; bootstrap resampling
  semantics in `_iter_bootstrap_seasonal`; the one-big-season fallback and `_aggregate_annual`'s
  max-across-valid-seasons rule; `UstarBootstrapThresholds._get_window_years` for 1/2/3-year records
  and the first/last year; `FlagMultipleConstantUstarThresholds` /
  `FlagMultipleVariableUstarThresholds` flag naming and the element-wise `ustar >= threshold`
  broadcast. `FluxStorageCorrectionSinglePointEddyPro` never mutates the caller's frame and its
  rolling-median fill is correct — but note the window is uncapped in time, so a sparse storage
  column can be filled from values thousands of records away (documented behaviour; the ISFILLED
  flag is the only signal).
- **Self-heating**: the BUR06/JAR09 correction term matches Burba 2006 eq. 8 (Kelvin applied only to
  the denominator, ΔT left in °C) and the BUR08 unit chain resolves to µmol m-2 s-1;
  `merge_asof(..., by=)` row order verified empirically, so the positional write-back is aligned;
  `_block_bootstrap_indices` is a correct circular block bootstrap; `_remove_outliers_fast`'s
  MAD scaling is right and it only mutates freshly computed series; the MDV fallback in
  `_assign_scaling_factors` is correctly scoped by period.
- **Analysis**: the recorded `stl_decompose` fix is complete and numerically correct (recomposition
  error 3.6e-15; recovered amplitude 11.4 for a true 10, slope 2.93 for a true 2.9) — only the NaN
  and `weights` claims around it are wrong (L47, L50). `quantiles.percentiles101`, `profile`'s gap
  counting and overview guards, `granger`'s explicit index intersection and statsmodels column
  order, `compoundextremes`' alignment and zero-variance guards, `harmonic.spectrogram`'s
  segment-centre→timestamp mapping across gaps, ALE bin assignment and count-weighted centring, and
  `driveranalysis`' lag naming/parsing round-trip all check out.
- **Plotting**: `HeatmapDateTime`'s data path (complete lattice, START conversion, single-day case);
  `HeatmapBase.set_cmap`'s copy-before-`set_bad`; `TreeRingPlot`'s leap-year slot mapping and
  `shading='flat'` shapes; `DielCycle`'s aggregate plumbing; `ScatterXY`'s unique `_x`/`_y`/`_z`
  keys; `FormatStyle.apply`'s `None`-resolution and non-mutating `merged()`; every kwarg emitted by
  `codegen.py` exists on the corresponding `plot()` signature (checked across all eleven classes);
  no caller mutation anywhere in the plotting data path.
- **InfluxDB**: Flux `range(start:, stop:)` stop-exclusivity matches the documented behaviour;
  `_format_utc_offset` handles negative and fractional offsets; schema helpers push `start` back so
  the 30-day default cannot hide history; `delete()`'s falsy-but-not-`True` guards genuinely prevent
  a silent no-op and `measurements=True` is scoped to `data_version`; **no token or credential is
  written to disk or logged anywhere** in `influxio.py` / `config.py`.
- **3-D surface render pipeline**: `StructuredGrid` scalar ordering vs `ravel(order="F")`,
  `_staircase_cell_values` slicing and taller-neighbour rule, `_extruded_grid`/`_cell_edges`
  doubling incl. `N == 1`, `_bin_rows` reshape and NaN padding, `_roll_rows` gap preservation,
  `_MAX_ROWS` striding, and axis orientation vs labels.
- **`gui/theme.py`**: `load_dict` re-pins the structural tokens so a stale config cannot shadow
  them; `tag_color` is deterministic and its luminance branch correct; `reset()` deep-copies.

Still not reviewed even after round 3: `windrose.py`, `hexbin.py`, `histogram.py`, `waterfall.py`,
`shifted_distribution.py`, `timeseries.py`, `bar.py` in `core/plotting`, and `gui/icons.py`.

---

**[x] L73. `harmonic_decompose` returns the same component twice under a window**

> **Fixed 2026-08-07.** Selection is now over the spectral *peaks* (`find_peaks`, ranked by
> power) rather than the strongest bins, so a leakage shoulder — which sits on the flank of
> its peak, not on a local maximum — can no longer be returned as a component. The two-component
> repro now yields `[(50, 3.0), (25, 1.0)]` under boxcar, hamming, hann and blackman alike.
> Fewer than `n_harmonics` components come back when the spectrum holds fewer peaks; a
> monotonic spectrum falls back to the old behaviour.
>
> **Knock-on:** the reconstruction is built from N distinct components instead of N bins, so it
> carries slightly less of the signal's energy (a component's leakage skirt no longer adds bins).
> Where that reconstruction is used as a *feature* it is marginally weaker:
> `test_gapfilling_stl_features_xgboost` (3 trees, harmonic STL features) moved mae 2.79 -> 3.01,
> r2 0.53 -> 0.47, and its `mae < 3.0` bound was widened to 3.5 with that noted at the assertion.
> Distinct components are what the function documents; reconstruction fidelity is a separate
> knob (it would take reconstructing each component from its whole peak, not just the peak bin).
`core/times/decomposition_utils.py:275`

Found while fixing L49 — not a regression from it. `top_idx = np.argsort(-powers)[:n_harmonics]`
takes the strongest N bins, but a window spreads a tone across several bins, so the leakage
shoulder of a strong component can outrank a genuine weaker one. A two-component signal
(amplitude 3.0 at period 50, 1.0 at period 25) with the default hamming window:

```
boxcar   picked (period, amplitude): [(50, 3.0), (25, 1.0)]     <- both components
hamming  picked (period, amplitude): [(53, 1.278), (50, 3.0)]   <- period 50 twice
```

The amplitudes are now correct (L49); the *selection* is not. The reconstruction therefore
double-counts one component and misses the other, and `residual` absorbs the difference.
Reachable from the GUI Seasonal-trend tab's "Harmonic" option.

Fixing it means peak-picking with mainlobe exclusion (skip bins adjacent to one already
taken) rather than a plain top-N — a design change, so it is recorded rather than done.
