# Code Review Findings

Review dates: 2026-08-06 (round 1: core numerics + GUI) · 2026-08-07 (round 2: the modules round 1
left out; round 3: everything both left out) · 2026-08-16 (round 4: the eight files round 3 still
left out) · diive v0.91.0 · branch `indev` (at `af022000`)

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

### [x] CHANGELOG.md

Drafted 2026-08-15 into `CHANGELOG.md`'s unreleased v0.91.0 entry, as `### Fixed (code review)`.
**Six breaking changes** have landed on `indev` and each needs an entry saying what silently changes
for existing code. Count the `!` commits (`git log af022000..HEAD --oneline | grep '!:'`) rather than
the rows below — this table was recounted twice and was wrong both times, because a row was added
without the count being updated and `85eb97cc` was never listed at all:

| Commit | Breaks |
|---|---|
| `85eb97cc` `refactor!: remove UstarDetectionMPT` | `UstarDetectionMPT` gone from `dv.flux` (it was in `__all__`); superseded by `UstarMovingPointDetection`. Landed with L32's fix |
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

### [x] `diive/gui/MANUAL.md`

The *"Keep overlapping data points only"* checkbox description (line 969) was stale from
`876bec12` and has been **fixed 2026-08-07** — replaced with the overlap-only rule plus the new
record-loss reporting, and `MANUAL.html` regenerated via `diive/gui/build_manual.py`. Fixed
immediately rather than batched because it was left behind by a change in this same effort.

**Still to do:** sweep the rest of the manual for the other removals before release — the H2O
self-heating path and the outlier flag semantics both have user-facing descriptions that may
still describe the old behaviour.

### [x] `CLAUDE.md`

Updated piecemeal so far (the combine-variables tab description, `1c392e66`). Re-read the sections
covering every changed area once the backlog settles: the outlier day/night conventions (flag
semantics), `dv.analysis` and `dv.plotting` namespace tables, and the self-heating notes.

### [ ] `docs/`

**Its own separate project, not part of this campaign** (user's decision, 2026-08-15). Do not
pick it up as review follow-up work. The tree
predates this campaign: written against the dead `diive.pkgs.*` layout, with hand-written pages
calling API that no longer exists, and `auto_examples/` is Sphinx-Gallery output whose rebuild
executes every example. Stale generated content was tracked as **L90, now removed** rather than left
open — it is a symptom of this deferral, not a separate defect. Known stale at the time of removal:
`auto_examples/flux/uncertainty.*` (method 2 as "±5 days", method 4 as "5 nearest fluxes", the latter
doubly wrong after L6), `auto_examples/flux/selfheating.*` (passes the removed `flux_type`), and
`api/pkgs/*.rst` (lists the removed `UstarDetectionMPT`; orphaned, no toctree reaches it).
Regenerate once the code has settled.

`docs/auto_examples/` is **generated** by sphinx-gallery from `examples/`, so it regenerates on the
next build — no hand-editing. It is currently stale (`selfheating.py`/`.ipynb`/`.rst` still pass
`flux_type=FLUX_TYPE`). Just remember to rebuild, and check the API pages no longer list the four
deleted harmonic functions.

### [ ] Examples

`examples/analysis/analysis_harmonic.py` is the only caller of `harmonic_analysis`, and it was
written against the buggy amplitudes — its narrative text about the window's effect may now say the
opposite of what the code produces. Re-read it against the corrected behaviour.

---

# Triage index — all 155 findings by severity

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

## S1 — Silently wrong scientific output (16 + 1 by design)

| ID | Finding | Where |
|---|---|---|
| ~~L107~~ | ~~`HexbinPlot`'s `mincnt=0` default paints hexagons over cells holding no data — 85% of a two-cloud plot fabricated with `np.sum`, `ValueError` with `np.max`~~ (done 2026-08-16) | `core/plotting/hexbin.py:76` |
| ~~L108~~ | ~~Histogram KDE overlay scaled by the *first* bin width, so it is wrong for any non-uniform bin list — 5.1x above the tallest bar~~ (done 2026-08-16) | `core/plotting/histogram.py:142` |
| ~~L109~~ | ~~`LongtermAnomaliesYear` discards its own `sort_index`, so an unsorted record is plotted **and averaged** in file order~~ (done 2026-08-16) | `core/plotting/bar.py:64` |
| ~~L110~~ | ~~`LongtermAnomaliesYear` keys its working frame by the caller's Series name — a collision zeroes every anomaly (L9 family)~~ (done 2026-08-16) | `core/plotting/bar.py:94` |
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

## S2 — Silently does nothing / silently loses data (37)

| ID | Finding | Where |
|---|---|---|
| L111 | `WaterfallPlot` draws a fully missing period as a 0.0 bar under the default `agg='sum'` — 429 of 3652 bars on bundled `LW_IN`; `agg='mean'` drops them instead | `core/plotting/waterfall.py:66` |
| L112 | `TimeSeries` colour-by draws **measured** records fully transparent wherever the *colour* series has a gap — 80 of 200 | `core/plotting/timeseries.py:315` |
| L113 | Colour-by silently degrades to a plain line on index mismatch; `cmap`/`show_colorbar`/`color_label` become no-ops | `core/plotting/timeseries.py:405` |
| L114 | Missing years drawn as adjacent bars while the title asserts the full span (L61/L63/L79 family; GUI `dropna()` feeds it) | `core/plotting/bar.py:147` |
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
| ~~L97~~ | ~~`UstarBootstrapThresholds` resamples unseeded in **both** paths~~ (done 2026-08-15) — and a **third** site the entry missed: `UstarMovingPointDetection.bootstrap()`. Seed derived per year, so `n_jobs` cannot change the answer | `flux/lowres/ustar_bootstrap.py:50` |
| ~~L86~~ | ~~`UstarVekuriThresholdDetection.bootstrap()` has no `random_state`, so u* thresholds differ run to run~~ (done 2026-08-15) — **the same defect in `UstarBootstrapThresholds` is open as L97** | `flux/lowres/ustar_vekuri_detection.py` |
| ~~L87~~ | ~~`classical_decompose` passes `extrapolate=` where the parameter is `extrapolate_trend`, so it always raises and the trend edges are always NaN~~ (done 2026-08-15) — dead branch removed; NaN edges kept on purpose | `core/times/decomposition_utils.py:207` |
| ~~L92~~ | ~~`ScreeningTabBase._select` does not bump `_run_id` — G2's bug in the tab cited as the correct pattern~~ (done 2026-08-15) | `gui/tabs/_screening_base.py` |
| ~~L99~~ | ~~Running the test suite **overwrites the developer's real GUI preferences**: three tests call `win.close()`, and `closeEvent` -> `save_config()` writes theme/geometry/`last_project`/`variable_metadata` to the live `QStandardPaths` file, non-atomically~~ (done 2026-08-15) | `gui/config.py` + `tests/test_gui.py` |
| L76 | ~~BUR06 uses a canopy `ra` where Burba 2006 specifies a per-element one and drops `fr`~~ **answered from the paper 2026-08-15, partly wrong**: dropping `fr` is fine, but the SF does **not** absorb the `ra` shape — ~4.8x residual spread *within* each USTAR class. **deferred to a future session**, methodology still open | `flux/lowres/selfheating.py:471` |
| ~~L53~~ | ~~`CompoundExtremes` returns zero classified periods for a single year, silently~~ (done 2026-08-07) | `analysis/compoundextremes.py:168` |
| ~~L59~~ | ~~`multi_scale_harmonics` swallows every exception~~ (done 2026-08-07) — function deleted as dead code | `analysis/harmonic.py:432` |
| ~~L19~~ | ~~`features_stl=True` can produce nothing — any single NaN skips a column, logged only at DEBUG~~ (done 2026-08-07) | `core/ml/feature_engineer.py:726` |
| ~~L16~~ | ~~SWIN short-gap interpolation never fires near dawn/dusk — the night's NaN run inflates the gap length~~ **(not a bug 2026-08-15)** — the run length is redundant where the mask can accept and load-bearing where it isn't; documented + tested instead | `gapfilling/swin.py:794` |
| ~~L4~~ | ~~`keep_records_where`: an "open" bound is not open when `inclusive != 'both'` — drops the extreme record~~ (done 2026-08-15) | `core/dfun/frames.py:110` |
| ~~L17~~ | ~~`MetadataStore.rename` silently drops a record on a name collision~~ (done 2026-08-15) | `core/metadata/__init__.py:325` |
| ~~L64~~ | ~~glTF `.glb` export bakes the texture **mirrored** along the date axis~~ (done 2026-08-15) | `gui/tabs/surface3d.py:744` |
| ~~L66~~ | ~~`datetime_surface_grid` omits the `TIMESTAMP_START` conversion — 3-D surface offset half a period from the heatmap~~ (done 2026-08-15) | `core/plotting/surface_grid.py:68` |
| ~~L67~~ | ~~3-D export buttons write the **previous** variable's relief after a render that produced nothing~~ (done 2026-08-15) | `gui/tabs/surface3d.py:945` |
| ~~G4~~ | ~~`restore_controls` silently keeps the current combo value when the saved entry is gone — can flip the joint-uncertainty divisor~~ (done 2026-08-15) | `gui/widgets/state_utils.py:51` |
| ~~L73~~ | ~~`harmonic_decompose` picks the top-N bins by power, so a windowed strong component's leakage outranks a genuine weaker one — the same component is returned twice~~ (done 2026-08-07) | `core/times/decomposition_utils.py:275` |

## S3 — Crash on legitimate input (23)

| ID | Finding | Where |
|---|---|---|
| L115 | `HistogramPlot.plot` raises on a constant series — **inside the outlier detectors' own `showplot=True` diagnostic** | `core/plotting/histogram.py:177` |
| L116 | `HistogramPlot.plot` raises on an all-NaN column (L69 family) | `core/plotting/histogram.py:118` |
| L117 | `WaterfallPlot.plot` raises `IndexError` on an all-NaN column (L69 family) | `core/plotting/waterfall.py:164` |
| L118 | `ShiftedDistributionPlot` dies on an empty / all-NaN / single-record / constant period — unguarded Silverman bandwidth | `core/plotting/shifted_distribution.py:94` |
| L119 | `TimeSeries.plot_interactive()` raises on an unnamed Series; `plot()` and `plot_rangetool()` handle it | `core/plotting/timeseries.py:156` |
| L120 | `LongtermAnomaliesYear` raises `KeyError: None` on an unnamed Series (same root as L110) | `core/plotting/bar.py:101` |
| ~~L1~~ | ~~`Hampel` crashes on any non-fixed frequency (monthly/yearly/business-day)~~ (done 2026-08-15) | `preprocessing/outlier_detection/hampel.py:228` |
| ~~L3~~ | ~~Frequency detection: off-by-one denominator → clean 2-row series "too irregular", 1-row bare `KeyError`~~ (done 2026-08-15) — one bad denominator caused both the crash and the wrong confidence | `core/times/times.py:1386` |
| ~~L27~~ | ~~`set_storage_to_zero=True` still requires the storage column — the exact case it documents~~ (done 2026-08-15) | `flux/lowres/storage_correction.py:150` |
| ~~L39~~ | ~~`ScopApplicator`'s undocumented column-name contract — legal inputs raise `KeyError`~~ (done 2026-08-15) — names normalised at the boundary | `flux/lowres/selfheating.py:1214` |
| ~~L43~~ | ~~Default fringe-bin trimming empties the time-lag histogram; the `IndexError` escapes the batch helpers~~ (done 2026-08-15) | `flux/lowres/timelag_analysis.py:148` |
| ~~L31~~ | ~~`UstarThresholdConstantScenarios.calc(showplot=True)` crashes on pandas 3~~ (done 2026-08-15) — **two** sites, not the one this entry named | `flux/lowres/ustarthreshold.py:337` |
| ~~L33~~ | ~~`UstarVekuriThresholdDetection.summary()` crashes on its own guard before `detect()`~~ (done 2026-08-15) — guard fixed, not made to raise | `flux/lowres/ustar_vekuri_detection.py:187` |
| ~~L56~~ | ~~`harmonic_decompose` returns `frequencies` one element longer~~ (done 2026-08-07) | `core/times/decomposition_utils.py:307` |
| ~~L65~~ | ~~`RidgeLinePlot` cannot plot any series containing a gap~~ (done 2026-08-15) | `core/plotting/ridgeline.py:196` |
| ~~L69~~ | ~~`Cumulative.plot` raises on an all-NaN column~~ (done 2026-08-15) — also `CumulativeYear.plot` | `core/plotting/cumulative.py:327` |
| ~~L68~~ | ~~`datetime_surface_grid` destroys a variable literally named `DATE` or `TIME`~~ (done 2026-08-15) — **the `HeatmapDateTime` half is S1, not S3**: no crash there, it silently paints the timestamps | `core/plotting/surface_grid.py:71` |
| ~~L13~~ | ~~`transform_yearmonth_matrix_to_longform` hardcodes the column names it drops~~ (done 2026-08-15) | `core/dfun/frames.py:644` |
| ~~L22~~ | ~~`MultiDataFileReader` raises `AttributeError` when every file is empty~~ (done 2026-08-15) | `core/io/filereader.py:320` |
| ~~G2~~ | ~~`_outlier_base._on_done` can `KeyError` when the dataset changes mid-run~~ (done 2026-08-15) — **has a silent half too**: a narrowed frame is adopted with no exception at all | `gui/tabs/_outlier_base.py:666` |
| ~~L79~~ | ~~`transform_yearmonth_matrix_to_longform` rejects non-contiguous month columns — a seasonal record from its own producer raises; 4th instance of the contiguity family (L61/L63)~~ (done 2026-08-15) | `core/dfun/frames.py` |
| ~~L81~~ | ~~`TimeLagAnalysis`: a `histogram_startbin`/`endbin` range excluding every lag empties `results`, then `detect_peak_range` fails~~ (done 2026-08-15) | `flux/lowres/timelag_analysis.py` |
| ~~L101~~ | ~~**Two examples broken by this campaign's own breaking change** since `45614fb3`: `flux_selfheating.py` and `flux_selfheating_production.py` index `LATENT_HEAT_VAPORIZATION_J_UMOL`, removed with the H2O path (L37)~~ (done 2026-08-15) | `examples/flux/lowres/` |

## S4 — Contract mismatch (31)

| ID | Finding | Where |
|---|---|---|
| L121 | `ignore_fringe_bins` accepted, documented and stored by `HistogramPlot`; nothing applies it (L62/L91 family — a working impl exists in `analysis/histogram.py`) | `core/plotting/histogram.py:45` |
| L122 | `minticks`/`maxticks` accepted, documented and forwarded by hexbin; only `nice_date_ticks` consumes them and hexbin never reaches it | `core/plotting/hexbin.py:268` |
| L123 | `color_bad` accepted, documented and forwarded by hexbin; it takes effect only via `set_cmap`, which hexbin never calls | `core/plotting/hexbin.py:270` |
| L124 | Hexbin's auto `cb_extend` reads the **raw** `z` range while the colorbar maps the aggregate — arrows asserting data that is not clipped | `core/plotting/hexbin.py:330` |
| L125 | Hexbin pairs x/y/z positionally, never by index — same-labelled Series in a different order are mispaired silently *(latent)* | `core/plotting/hexbin.py:377` |
| L126 | Histogram/hexbin derive bin edges per subset with no way to pin them, and `flagbase` puts two such panels side by side (L2 family) | `core/plotting/histogram.py:118` |
| L127 | `WindRosePlot.plot(ax=...)` writes the **figure** suptitle and adjusts the caller's layout | `core/plotting/windrose.py:448` |
| L128 | The wind rose ignores every `FormatStyle` field but the title, while the GUI feeds it the full shared Format section | `core/plotting/windrose.py:347` |
| L129 | `ShiftedDistributionPlot` overrides a caller-set `FormatStyle.ylabel` via `merged()` where `apply(default_ylabel=)` is correct, and forces grid off | `core/plotting/shifted_distribution.py:186` |
| L130 | A zone breakpoint outside the evaluation grid leaves one zone unpainted and puts its label over a region it does not describe | `core/plotting/shifted_distribution.py:154` |
| ~~L15~~ | ~~`flag_ssitc_eddypro_test` performs no conversion despite documenting one~~ (done 2026-08-15) | `preprocessing/qaqc/eddyproflags.py:490` |
| ~~L41~~ | ~~`ScopPhysics` documents an RF + MDV gap-fill that does not exist~~ (done 2026-08-15) | `flux/lowres/selfheating.py:152` |
| ~~L44~~ | ~~`TimeLagAnalysis` docstring states three parameter facts the code contradicts~~ (done 2026-08-15) | `flux/lowres/timelag_analysis.py:90` |
| ~~L57~~ | ~~`reconstruct_from_components` forces the trend's NaN onto reconstructions excluding the trend~~ (done 2026-08-15) | `core/times/decomposition_utils.py:420` |
| ~~L60~~ | ~~`seasonality_strength` formula and `'iterations'` return value both mis-documented~~ (done 2026-08-15) | `analysis/seasonaltrend.py:174` |
| ~~L62~~ | ~~`show_less_xticklabels` accepted and documented by `HeatmapDateTime` but never applied~~ (done 2026-08-15) | `core/plotting/heatmap_datetime.py:245` |
| ~~L21~~ | ~~`crosscorr` omits dates for three early-outs while others write NaN~~ (done 2026-08-15) | `preprocessing/qaqc/detect_timestamp_shifts.py` |
| ~~L18~~ | ~~`FeatureEngineer` rolling stages re-engineer already-engineered columns; other stages skip them~~ (done 2026-08-15) | `core/ml/feature_engineer.py` |
| ~~L9~~ | ~~`GridAggregator` keys its frame by Series name — two roles sharing a name collide~~ (done 2026-08-15) | `analysis/gridaggregator.py:119` |
| ~~L11~~ | ~~`JointUncertaintyPAS20` cumulative scenario term not masked to available flux~~ (done 2026-08-15) | `flux/lowres/uncertainty.py:867` |
| ~~L6~~ | ~~Random-uncertainty method 4 uses an asymmetric neighbour window (5 below, 4 above)~~ (done 2026-08-15) | `flux/lowres/uncertainty.py:708` |
| ~~L8~~ | ~~`FlagQCF`'s documented "QCF is NaN if no flag available" branch is unreachable~~ (done 2026-08-15) | `preprocessing/qaqc/qcf.py:640` |
| ~~G3~~ | ~~Pinned tabs are not actually frozen against added columns~~ (done 2026-08-15) | `gui/app.py:899` |
| ~~G6~~ | ~~`WorkerRunner` clears `is_running` before emitting — re-entry window~~ (done 2026-08-15) | `gui/widgets/worker.py:73` |
| ~~L78~~ | ~~`percent_matching`/`confidence` parsed back out of a `'{:.0f}%'` string, so 99.9% reports as 100.0~~ (done 2026-08-15) | `core/times/times.py` |
| ~~L103~~ | ~~`ScopApplicator` labels an **un**-gap-filled input `FCT_UNSC_gfXG`: `__init__` normalises whatever term it is handed to `ColumnConfig.fct_unsc_gf`, so the frame claims a fill that did not happen. Predates L95 (it said `gfRF`) and follows from L39's boundary normalisation~~ (done 2026-08-15) | `flux/lowres/selfheating.py` |
| ~~L83~~ | ~~`histogram_startbin`/`endbin` are seconds but named as bin indices~~ (done 2026-08-15) — also fixed the integer GUI spinboxes that could not express 0.40 s | `flux/lowres/timelag_analysis.py` |
| ~~L84~~ | ~~`ignore_fringe_bins` still described as "bin indices" in `__init__` (they are counts)~~ (done 2026-08-15) | `flux/lowres/timelag_analysis.py` |
| ~~L88~~ | ~~`detect_seasonality`'s `'strength'` is a peak-power ratio, documented as a variance ratio~~ (done 2026-08-15) | `core/times/decomposition_utils.py` |
| ~~L91~~ | ~~`hexbin.py` accepts, documents and forwards `show_less_xticklabels`, and nothing applies it~~ (done 2026-08-15) | `core/plotting/hexbin.py:272` |
| ~~L95~~ | ~~`ScopPhysics.fct_unsc_gf` is `'FCT_UNSC_gfRF'` though the fill is XGBoost~~ (done 2026-08-15) — the suffix was hardcoded twice, which is why it drifted; now one string | `flux/lowres/selfheating.py` |

## S5 — Cosmetic / dead / latent (44)

| ID | Finding | Where |
|---|---|---|
| ~~L147~~ | ~~**Menu-action lambdas capture `self`** — the actual, sole reason no `MainWindow` is ever collected (L105's real cause)~~ (done 2026-08-16) — all 7 lambdas out of `app.py`; **4 windows live -> 0**. ~55 remain elsewhere in `gui/`, mostly rescued by L106; the unrescued ones are listed in the entry | `gui/app.py:423` |
| L131 | Histogram info box appends itself — the text is printed up to four times | `core/plotting/histogram.py:160` |
| L132 | The wind rose drops out-of-range directions (sentinels, radians) without reporting the count | `core/plotting/windrose.py:198` |
| ~~L133~~ | ~~`WindRosePlot`'s docstring example calls a function that does not exist — **and reST `Example::` blocks are invisible to both docstring tests** (L85 hole)~~ (done 2026-08-16) — the real count was **41** literal blocks, not 13; the extended check found 5 more dead names | `core/plotting/windrose.py:85` |
| L134 | A waterfall contribution of exactly 0.0 is coloured "release" (compounds L111) | `core/plotting/waterfall.py:142` |
| L135 | `zone_colors`/`zone_labels` lengths unvalidated — 3 colours raise, 3 labels silently under-label | `core/plotting/shifted_distribution.py:171` |
| L136 | Colour-by replaces the caller's axes limits instead of `update_datalim` + `autoscale_view` | `core/plotting/timeseries.py:330` |
| L137 | A second `plot()` on the **same** axes stacks artists and colorbars — all three of timeseries/bar/shifted-distribution | `core/plotting/timeseries.py:339` |
| L138 | `fig.tight_layout()` on a figure built `layout='constrained'` — warns and silently disables the engine | `core/plotting/bar.py:173` |
| L139 | `LongtermAnomaliesYear.get()` before `plot()` raises `AttributeError` | `core/plotting/bar.py:176` |
| L140 | `icons.py`'s `('calculate', _ln_gear)` rule is unreachable — both derived-variable calculators fall back to the generic glyph | `gui/icons.py:571` |
| L141 | Icons baked at 16x16 with `devicePixelRatio` 1 — blurry at Windows 150%/200% scaling | `gui/icons.py:26` |
| L142 | Sub-pixel coordinates discarded by PySide6's integer `drawLine` overload (~12 glyphs) | `gui/icons.py` |
| L143 | `menu_icon(None)` raises, though the docstring promises unknown labels fall back | `gui/icons.py:730` |
| L144 | Both bokeh methods call `show(p)` unconditionally — no `showplot` toggle (L104 family) | `core/plotting/timeseries.py:217` |
| L145 | `bar.py` uses Material 400-level colours where the convention specifies 300 | `core/plotting/bar.py:143` |
| L146 | `ShiftedDistributionPlot` uses the population sd (ddof=0) for its zone boundaries where diive uses ddof=1 | `core/plotting/shifted_distribution.py:75` |
| ~~L10~~ | ~~`vectorize_timestamps` `.SEASON` as `Int64` forces object-dtype arrays into every ML fit~~ (done 2026-08-15) | `core/times/times.py:1245` |
| ~~L12~~ | ~~`LocalSD`: values exactly on the limit are in neither `ok` nor `rejected`~~ (done 2026-08-15) | `preprocessing/outlier_detection/localsd.py:279` |
| ~~L20~~ | ~~`lagged_variants` edge-fill is conditional but documented as unconditional~~ (done 2026-08-15) | `variables/temporal.py:461` |
| ~~L23~~ | ~~`sort_multiindex_columns_names` mutates the list it iterates (reverses moved columns)~~ (done 2026-08-15) | `core/dfun/frames.py:510` |
| ~~L24~~ | ~~Nested-quote f-string prints a literal `{limit}`; `_calculate_gap_sizes` is dead~~ (done 2026-08-15) | `gapfilling/interpolate.py:143` |
| ~~L25~~ | ~~`_extract_and_convert_flag_from_multidigit` turns a scalar `0` code into NaN~~ (done 2026-08-15) | `preprocessing/qaqc/eddyproflags.py:47` |
| ~~L35~~ | ~~USTAR docstring examples import from the wrong namespace~~ (done 2026-08-15) | `flux/lowres/ustar_bootstrap.py:133` |
| ~~L70~~ | ~~Rolling cell aggregator uses an `n+1`-row window for even `n`~~ (done 2026-08-15) | `gui/tabs/surface3d.py:98` |
| ~~L71~~ | ~~`convert_ts_to_timezone` cannot accept the `DatetimeIndex` its docstring promises~~ (done 2026-08-15) | `core/io/db/influx/common.py:59` |
| ~~G5~~ | ~~`_screening_base._run` starts unbounded concurrent worker threads~~ (done 2026-08-15) | `gui/tabs/_screening_base.py:729` |
| ~~G7~~ | ~~`_compute_payload` writes tab state from the worker thread~~ (done 2026-08-15) | `gui/tabs/_outlier_base.py:463` |
| ~~G8~~ | ~~`save_config` catches only `OSError`~~ (done 2026-08-15) | `gui/config.py:36` |
| ~~G9~~ | ~~Project load transiently materialises the previous session's event columns~~ (done 2026-08-15) | `gui/app.py:1372` |
| ~~L77~~ | ~~`MultiDataFileReader.metadata_df`'s guard tests `self._data_df`, the wrong attribute~~ (done 2026-08-15) | `core/io/filereader.py` |
| ~~L80~~ | ~~`UstarVekuriThresholdDetection.bootstrap_results_` initialised, never read~~ (done 2026-08-15) | `flux/lowres/ustar_vekuri_detection.py` |
| ~~L82~~ | ~~Exceptions in Qt-invoked slots are swallowed — GUI tests driving via signals may be weaker than they look~~ (done 2026-08-15) — the guard uncovered a real leaked-slot bug 44 tests were passing over | `tests/test_gui.py` (methodology) |
| ~~L85~~ | ~~No doctest runner anywhere — docstring examples are never executed (L35 had been broken twice over)~~ (done 2026-08-16) — 34 samples → 22, 12 executed by `test_docstring_examples.py`, 10 excluded with a stated reason each; found two more stale samples on the way | `tests/` (methodology) |
| ~~L104~~ | ~~`ScopPhysics.plot_diel_cycles()`, `ScopOptimizer.plot()` and `ScopApplicator.plot_dashboard()` call `plt.show()` unconditionally with no `showplot` toggle, so an example cannot satisfy the disable-showplot standard and figures accumulate~~ (done 2026-08-15) | `flux/lowres/selfheating.py` |
| ~~L105~~ | ~~`FramelessResizeHelper` stores `self._window` while also being parented to that window's grip, so **no `MainWindow` can ever be garbage-collected**~~ (done 2026-08-16) — **diagnosis was wrong**: the collector *can* break that cycle; the weakref only makes collection refcount-deterministic. The real pin is **L147** | `gui/widgets/frameless.py:28` |
| ~~L106~~ | ~~Parentless widgets accumulate for the session~~ (done 2026-08-16) — two passes: `_on_tab_close` deletes the page (**511 leaked widgets per open/close -> 1**), and the `self`-capturing lambdas that made a `DiiveTab` uncollectable are gone (**28 of 41 tab classes leaked on drop -> 0**; suite-end live widgets **13 149 -> 133**, live tabs **0**). `CopyPythonButton._provider` was **not** part of it | `gui/app.py:792`, `gui/widgets/weak_slot.py` |
| ~~L98~~ | ~~`run_all_examples.py` forces no matplotlib backend, so the 16 examples ending in `plt.show()` block~~ (done 2026-08-15) — corrected: the 120 s timeout meant **spurious failures**, not a permanent hang | `examples/run_all_examples.py` |
| ~~L89~~ | ~~`-9999` at position 6 still reads as a passing flag~~ (done 2026-08-15) — **two** holes, not one: `-1` at position 1 read as a soft warning | `preprocessing/qaqc/eddyproflags.py` |
| ~~L93~~ | ~~`EventManager.load_dict({})` early-returns without clearing, so previous events survive~~ (done 2026-08-15) | `gui/events.py` |
| ~~L94~~ | ~~`crosscorr`'s `len(pot_arr) == 0` branch is unreachable dead code~~ (annotated, kept 2026-08-15) | `preprocessing/qaqc/detect_timestamp_shifts.py` |
| ~~L96~~ | ~~`ScatterXY` uses `plt.colorbar` instead of `ax.figure.colorbar`~~ (done 2026-08-15) — also `DetectTimestampShifts.plot_radiation_fingerprint` | `core/plotting/scatter.py:169` |
| ~~L100~~ | ~~4 test modules never set the matplotlib backend and inherit `Agg` from an alphabetically earlier module; the real default here is `qtagg`, so they pass serially by accident~~ (done 2026-08-15) | `tests/` (methodology) |
| ~~L102~~ | ~~`show_values=True` on a large date/time heatmap draws one text artist per cell (17 520 for a year of half-hourly data) with no guard: 15.8 s to render, and every later re-layout walks them - the next tab open went 1.2 s -> 43.1 s~~ (done 2026-08-15) | `core/plotting/heatmap_base.py` |

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

### [x] L1. `Hampel` crashes on any non-fixed frequency (monthly / yearly / business-day)

> **Fixed 2026-08-15.** `_gap_flanking_records` guards the `index.freq.nanos` read and falls through
> to the existing median-of-diffs branch, so a non-fixed offset takes its step from the timestamps.
> A `try/except ValueError` on that single attribute is deliberate: pandas 3 offers no non-raising
> fixedness test (`Day` is no longer a `Tick` yet has working `.nanos`, and `hasattr` propagates the
> error), so the attempt *is* the test. The fixed-frequency path is unchanged code, so today's
> numbers cannot move — confirmed, `test_outlierdetection.py`'s exact-value assertions still pass.
> Covered by `TestNonFixedFrequencyIndex` (monthly `MS`, yearly `YS`), which also asserts
> `_untestable` stays False in the interior, pinning the computed step rather than only the absence
> of a crash. Mutation-checked: `ValueError: <YearBegin: month=1> is a non-fixed frequency`.

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

### [x] L3. Frequency detection: off-by-one denominator breaks short series and mis-reports confidence

> **Fixed 2026-08-15.** One fix, two lines. Dividing the modal-delta count by the number of
> *intervals* (`len(df) - 1`) rather than rows corrects both the reported confidence and the 2-row
> failure — the same expression caused both. The 1-row `KeyError` needed a second line: an
> empty-mode guard returning `'-not-enough-datarows-'`, which routes to `DetectFrequency`'s existing
> informative `RuntimeError` instead of `KeyError: 0`.
>
> Because this runs at the front of nearly every diive workflow, the no-change claim was verified
> rather than asserted: all four bundled datasets detect the identical frequency before and after
> (only the reported match rises to a correct 100%), and a 3000-trial randomised sweep shows the
> detected frequency can change in exactly one arithmetically constrained case — even `n` where the
> top delta occurs `n/2` times, a genuine majority of `n-1` intervals that the old code rejected
> outright. So the change only *adds* detections.
>
> Mutation-checked both lines separately: `('30min', '99% occurrence')` vs `100%` and `98.0 != 100.0`
> for the denominator, `KeyError: 0` for the guard. See L78 for the string-parsing wart left behind.

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

### [x] L4. `keep_records_where`: an "open" bound is not open when `inclusive != 'both'`

> **Fixed 2026-08-15.** An unset bound now substitutes `-np.inf` / `np.inf` instead of the
> condition's observed `min()` / `max()`, so the open side stays open for every `inclusive` value.
>
> Neither sibling needed a change, both checked: `select_records_to_code` omits `lower`/`upper`
> when they are `None`, so generated code inherits the fix; the GUI's own `min()`/`max()`
> substitution (`select_records.py:541`) only sizes the shaded preview band — an infinite span
> would wreck the y-limits — while the mask itself comes from this function, so the preview
> markers and the result now agree.
>
> Covered by `TestKeepRecordsWhere::test_open_bound_stays_open_for_all_inclusive` (open-lower and
> open-upper across all four `inclusive` values, via `subTest`). Mutation-checked: reinstating the
> `cond.min()`/`cond.max()` lines fails exactly the four exclusive-on-the-open-side combinations
> (`open lower` + `neither`/`right`, `open upper` + `neither`/`left`).

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

### [x] L6. Random-uncertainty method 4 uses an asymmetric neighbour window

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

### [x] L8. `FlagQCF`: "QCF is NaN if no flag is available" is unreachable

`diive/preprocessing/qaqc/qcf.py:640-645`, docstring at `:222`

`_calculate_flagsums` produces `0` (not NaN) for rows with no flags, so the `sumflags == 0` branch
immediately sets QCF=0. The NaN initialisation and the documented "or NaN if no flags available"
describe a state that cannot occur.

---

## Library — lower severity

### [x] L9. `GridAggregator` builds its frame from a dict keyed by Series names

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

### [x] L10. `vectorize_timestamps` makes `.SEASON` a nullable `Int64`, forcing object-dtype arrays

`diive/core/times/times.py:1245` (`insert_season` returns `.astype('Int64')`)

One nullable-extension column turns the whole frame's `.to_numpy()` into **object dtype**
(**[reproduced]**). That array flows through `convert_to_arrays` into `model.fit` / `predict` /
`shap` on every ML gap-fill and every fallback fill. sklearn and XGBoost both accept it (verified),
so nothing breaks — but every run pays a hidden object→float conversion. A plain `int` cast (or
`.astype('float64')`) avoids it.

### [x] L11. `JointUncertaintyPAS20`: cumulative scenario term is not masked to available flux

`diive/flux/lowres/uncertainty.py:867-871` — the random term uses `.where(flux.notna())`, the
scenario term does not. If the scenario columns have NaN where the flux does not (or vice versa),
the two cumulative terms sum over different record sets.

### [x] L12. `LocalSD`: values exactly on the limit are in neither `ok` nor `rejected`

`diive/preprocessing/outlier_detection/localsd.py:279-282` — `ok` uses `<`/`>`, `rejected` uses
`>`/`<`, so an exact-limit record gets a NaN flag for that iteration (resolving to 0 via L7).
Same effective outcome as "ok", but the asymmetry is easy to misread.

### [x] L13. `transform_yearmonth_matrix_to_longform` hardcodes the column names it drops

> **Fixed 2026-08-15.** Pins the axis names (`rename_axis(index='YEAR', columns='MONTH')`) before
> melting, so the later lookups and `drop` always match; `rename_axis` copies, so the caller's frame
> keeps its own names. Two distinct failures existed on documented input — unnamed axes gave
> `KeyError: None`, any other naming gave `KeyError: "['YEAR', 'MONTH'] not found in axis"`.
>
> Covered by `TestYearMonthMatrixToLongform` (4 tests incl. a round-trip from
> `resample_to_monthly_agg_matrix`). Mutation-checked, and worth recording: the round-trip test
> **kept passing** with the bug reinstated, because the producer names its axes correctly — the
> other three carry the regression. A test that does not discriminate is not evidence.
>
> A second defect in the same function is **not** fixed and is filed separately as L79.

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

### [x] G2. `_outlier_base._on_done` can raise `KeyError` when the dataset changes mid-run

> **Fixed 2026-08-15.** `_on_done` discards a completed detection whose target column or frame index
> no longer matches the dataset the tab now holds, after resetting `run_btn`/`progress` so the tab
> stays usable. The run's index travels in the **payload**, not in a new `self._run_*` attribute, so
> `_compute_payload` still writes no tab state — G7 stays exactly as fixable as before, and G6 does
> not interact (a second run would carry its own payload index). No `run_id` counter: `WorkerRunner`
> allows one job in flight here, so out-of-order completion is impossible.
>
> **This entry understates the severity.** Mutating away only the index check, keeping the column
> check, produces *no exception at all* — the narrowed frame's result is silently adopted
> (`assert <17519 rows x 2 columns> is None`). So G2 has a silent-corruption half as well as the
> `KeyError` half, and by this document's own rubric that half ranks above S3.
>
> `_rerender_last` was tightened too, which goes slightly beyond this entry's text — the entry says
> it "already guards exactly this", but it guarded only the missing-column half. Mutation 3:
> `IndexingError: Unalignable boolean Series provided as indexer` after a record extension. Covered by
> `test_outlier_tab_discards_result_when_dataset_changed_midrun` (4 cases incl. the unchanged frame,
> so the guard is not a blanket refusal). See L82 for a testing caveat found here that affects other
> GUI tests.

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

### [x] G3. Pinned tabs are not actually frozen against added columns

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

### [x] G4. `restore_controls` silently keeps the current value when a saved combo entry is gone

> **Fixed 2026-08-15.** `restore_controls` now returns the list of keys it could not apply, and
> `set_widget_value` returns `bool` (`False` only when a non-editable `QComboBox` has no matching
> item). The change is purely additive — every existing caller ignores the return value and
> behaves exactly as before — so no tab was forced to change.
>
> Wiring was deliberately partial, targeting the tabs where a silent fallback changes a *number*:
> joint uncertainty (the `JOINT_DIVISOR_IQR` 1.349 → `JOINT_DIVISOR_1SIGMA` 2.0 case this entry
> names), random uncertainty, and all four partitioning tabs via `_partitioning_base`. Left
> unwired: fluxchain, features, select_records, compound_extremes, the ML/MDS gap-fillers, the
> outlier/correction/derived bases, ustar_detection, seasonaltrend, spectrogram, timelag and the
> two 3-D surface tabs. Message wording lives in `unrestored_message()` in the GUI, per the
> separation rule.
>
> Covered by `test_restore_controls_reports_missing_combo_entry`, which asserts the reported key
> set, the labelled status text, **and** that the divisor really did fall back — the reason the
> warning matters. Mutation-checked: restoring the silent branch gives
> `AssertionError: assert set() == {'divisor_combo', 'randunc'}`.

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

### [x] G5. `_screening_base._run` starts an unbounded number of concurrent worker threads

`diive/gui/tabs/_screening_base.py:729-733` — no `is_running` guard. Stale *results* are correctly
discarded via `run_id`, but rapid chain edits stack up CPU-heavy threads that all run to completion.
`WorkerRunner` (used by the other tabs) already has the guard.

### [x] G6. `WorkerRunner` clears `is_running` before emitting

`diive/gui/widgets/worker.py:73` and `:80` — `self._running = False` precedes the queued
cross-thread `emit`, so `is_running` reads False during the window before the GUI thread processes
`done`/`failed`. Callers that use it as a re-entry guard (`_ml_gapfilling_base.py:708`,
`_partitioning_base.py:301`) can therefore start a second run whose result arrives interleaved with
the first. Also, `str(err)` is empty for some exception types, leaving a bare "Failed: " in the
status line.

### [x] G7. `_compute_payload` writes tab state from the worker thread

`diive/gui/tabs/_outlier_base.py:463` — `self._live_is_daytime = ...` is assigned off the GUI
thread, in a method whose own call site comments that "the worker must not read live Qt widgets".
Single attribute assignment so it's GIL-safe in practice, but it contradicts the stated contract;
carry it in the returned payload instead.

### [x] G8. `save_config` only catches `OSError`

`diive/gui/config.py:36-41` — a non-JSON-serializable value anywhere in the persisted blob (theme,
site, events, `variable_metadata`) raises `TypeError` out of `MainWindow.closeEvent`. All current
producers emit plain types, so this is latent, not active. Catching `(OSError, TypeError, ValueError)`
would match the module's stated "all failures are swallowed" contract.

### [x] G9. Project load transiently materialises the *previous* session's event columns

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

### [x] L15. `flag_ssitc_eddypro_test` performs no conversion despite documenting one

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

### [-] L16. SWIN short-gap interpolation silently never fires near dawn/dusk

> **Not a bug — closed 2026-08-15 after re-examination; the behaviour is now documented and
> regression-tested.** The diagnosis in this entry is wrong, and acting on it would have
> introduced a bug.
>
> Acceptance also requires `interp.notna()`, and `interp` is computed **per calendar day with
> `limit_area='inside'`**. A record can therefore only be accepted if a kt-valid record exists
> before *and* after it within the same day, which bounds its whole-series kt-NaN run to those
> anchors; since the above-floor band is contiguous within a day, the whole-series run length
> **equals** the daytime-only run length on exactly the records the mask can accept. Measured, not
> just argued: 0 differences across all 239 possible 2-record gap positions and 800 randomised
> multi-gap scenarios (4 densities, limits 1/2/4/16).
>
> What the repro above actually placed was a gap **on** the day's first two above-floor records.
> Its run length is indeed 20 rather than 2, but `interp` is NaN there either way — there is no
> earlier observation that day to interpolate from. A dawn gap one record later *is* interpolated
> today (flag 4, verified). The docstring's existing "could not be anchored" was already the
> correct explanation.
>
> The proposed fix would also break a documented promise. When the above-floor band splits *inside*
> one calendar day (solar noon near calendar midnight — a site near the date line, or a wrong
> `utc_offset`), per-day interpolation happily anchors across that dark band and the whole-series
> run length is the only thing refusing it. So the count is load-bearing exactly where it is not
> redundant.
>
> Recorded where it can be seen: a WHY comment at the run-length lines, and a docstring paragraph
> stating that a gap touching a day's first/last above-floor record goes to the model however short
> it is. Covered by `test_swin_gapfiller_short_gap_interpolation_at_day_edges` — four identical
> 2-record gaps (dawn-adjacent, midday, on the day's first, on its last) asserting flags 4/4/1/1,
> plus the split-band night-bridging guard. Mutation-checked in the reverse direction: applying the
> proposed change leaves the dawn/midday/edge assertions passing (confirming the no-op) and fails
> the night-bridging guard.

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

### [x] L17. `MetadataStore.rename` silently drops an entry on a name collision

> **Fixed 2026-08-15.** `MetadataStore.rename` now raises `ValueError` on a collision, checked
> before any mutation so a rejected rename leaves the store untouched.
>
> Collision is judged on the **result**, not the input: it builds the post-rename name of every
> stored entry (renamed or not) and raises if any appears twice. So `{A: B}` with `B` already
> stored raises, and so does `{A: C, B: C}`, while a simultaneous swap `{A: B, B: A}` and a
> whole-set prefix rename produce distinct results and still work — both are real workflows and
> both are asserted.
>
> `MainWindow._rename_variables` (`gui/app.py`) validates its mapping the same way against
> `_full_data.columns` (the frame can hold columns the store does not) and aborts with a warning
> instead of handing the frame duplicate labels.
>
> Covered by `test_rename_collision_raises`, `test_rename_swap_is_allowed`,
> `test_rename_bulk_prefix_is_allowed`. Mutation-checked: deleting the guard gives
> `AssertionError: ValueError not raised`.

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

### [x] L18. `FeatureEngineer`: the rolling stages re-engineer already-engineered columns, the others don't

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

### [x] L20. `lagged_variants`: the edge-fill is conditional but documented as unconditional

`diive/variables/temporal.py:461` — the shift-induced NaNs at the series edges are backfilled only
when the source column is **completely gap-free** (`n_missing_vals_before == 0`). For any real
driver with gaps they stay NaN, which then drops those rows from `model_df.dropna()` and demotes
them to the flag-2 fallback. The closing verbose message states unconditionally that the shift
"created gaps which were then filled with the nearest value".

The behaviour is defensible (don't fill genuine gaps) and is already documented as a *consequence*
in the SWIN class docstring; only the message here is wrong.

### [x] L21. `crosscorr` omits dates instead of writing NaN for three of its early-outs

`diive/preprocessing/qaqc/detect_timestamp_shifts.py` — the `pot_sum < 100` and clearness-index
branches write `{'shift_minutes': nan, 'max_corr': nan}`, but the `(pot > 0).sum() < 5`,
`sun_up.sum() < 5` and `len(pot_arr) == 0` branches `continue` without writing anything. The result
frame therefore has *missing rows* for some days and *NaN rows* for others, so callers aligning it
to a full date index get holes where they expect NaN.

---

## Round 2 — library: lower severity

### [x] L22. `MultiDataFileReader` raises `AttributeError` when every file is empty

> **Fixed 2026-08-15.** Raises a clear `ValueError` rather than returning an empty result: an empty
> frame has no timestamp index to reindex and would travel on into `continuous_timestamp_freq` and
> into the GUI as an inexplicably blank dataset. Two messages so the cause is named — no files given
> vs all files empty, the latter including the first path so a bad glob is visible.
>
> The partially-empty case (some files empty, some not) was verified to already work — real example
> file + a 0-byte file merges to 1488×101, `progress_callback` still fires for the skipped file — and
> locked in with its own test rather than assumed. Covered by `TestMultiDataFileReaderEmptyFiles`
> (3 tests). Mutation-checked: `AttributeError: 'NoneType' object has no attribute 'columns'`.
>
> An adjacent copy-paste defect found here is filed separately as L77.

`diive/core/io/filereader.py:320` — `data_df` stays `None` if every file raises `EmptyDataError`
(or the file list is empty), and `sort_multiindex_columns_names(df=None, ...)` then fails with
`'NoneType' object has no attribute 'columns'` instead of a clear "no readable data" error.

### [x] L23. `sort_multiindex_columns_names` mutates the list it is iterating

`diive/core/dfun/frames.py:510` and `:516`

```python
for ix, col in enumerate(cols_list):
    if col[0].startswith('.'):
        cols_list.insert(0, cols_list.pop(ix))
```

`pop(ix)` + `insert(0, …)` leaves positions after `ix` unchanged, so nothing is skipped — but the
moved columns end up in **reverse** order at the front, in both the `priority_vars` block and the
dot-prefix block. Cosmetic (column ordering only), but the pattern is a trap for the next edit.

### [x] L24. `interpolate.py`: a nested-quote f-string prints a literal `{limit}`

`diive/gapfilling/interpolate.py:143` and `:183`

```python
_console.print(f"\n{'Gap Analysis (limit={limit})':-^80}")
```

The `{limit}` sits inside a single-quoted literal *within* the f-string, so it is never
interpolated: the verbose report header reads `Gap Analysis (limit={limit})`.

Also in this module: `_calculate_gap_sizes` (`:19`) is dead — nothing calls it; the module reads
`GAP_LENGTH` off `GapFinder` directly. Flagging, not removing.

### [x] L25. `_extract_and_convert_flag_from_multidigit` turns a scalar `0` code into NaN

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

**[x] L27. `set_storage_to_zero=True` still requires the storage column — the exact case it documents**

> **Fixed 2026-08-15.** With `set_storage_to_zero=True` and the storage column absent, `strgcol` is
> set to `None` and the three sites that read it cope: `storage_correction()` selects only the flux
> column, `report()` prints a "storage term set to zero" line instead of computing availability, and
> `showplot()` derives its scale from the flux. The explicit-`strgcol` existence check now raises only
> when the storage term is actually used; the real (non-zero) paths are untouched. Covered by
> `test_level31_set_storage_to_zero_without_storage_column`, which drives the documented H/LE path
> end to end (`init_flux_data(fluxcol='LE')` → L2 → L3.1 with `SLE_SINGLE` dropped) and asserts
> `LE_L3.1` equals measured `LE`. Mutation-checked: `KeyError: "['SLE_SINGLE'] not in index"`.


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

**[x] L31. `UstarThresholdConstantScenarios.calc(showplot=True)` crashes on pandas 3**

> **Fixed 2026-08-15.** `counts` comes from `describe().loc['count']` and is indexed by column name,
> so the lookups in `_bartxt` must be positional: `counts.div(counts.iloc[0])` and
> `counts_perc.iloc[ix]`. **This entry named only the first site**; the second has the identical
> defect one line later and would crash immediately after. Mutation-checked the second site *alone*
> (first restored) to prove the test covers both edits: `KeyError: 0` either way. Covered by
> `TestScenarioPlotCountsAreAddressedPositionally`, asserting all 6 bar annotations and that the
> unfiltered column reads 100%; `matplotlib.use("Agg")` added to `test_ustar_mp.py`, which had no
> backend pin. Display-only code — no threshold value moved.


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

**[x] L33. `UstarVekuriThresholdDetection.summary()` crashes before `detect()`**

> **Fixed 2026-08-15.** `summary()` before `detect()` is deliberately **not** made an error: the
> method already intends to return "Run detect() first", and the sibling ONEFlux port
> `UstarMovingPointDetection` behaves identically, so raising would put a gratuitous divergence
> between two classes users swap between. The actual bug was `results_ = {}` in `__init__` — a dict
> has no `.empty`, so the guard itself raised `AttributeError`. Now `pd.DataFrame()`, as is
> `bootstrap_stats_`, which was documented in Attributes but never assigned (the entry's second
> half); `bootstrap()` now assigns it. Docstrings updated to match. Covered by
> `TestVekuriSummaryBeforeDetect`. Mutation-checked: `AttributeError: 'dict' object has no attribute
> 'empty'`. No computed number touched. See L80 for dead code noticed alongside.


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

**[x] L35. USTAR docstring examples import from the wrong namespace**
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

**[x] L39. `ScopApplicator` has an undocumented column-name contract; legal inputs raise `KeyError`**

> **Fixed 2026-08-15.** Any legal input is now **accepted** rather than the contract being tightened:
> `__init__` renames `fct_unsc` → `ColumnConfig.fct_unsc_gf` and `daytime` → `ColumnConfig.daytime`
> at the boundary. That is the right direction because the rest of the class already reads both
> columns through its own `ColumnConfig` — the canonical-name convention was the intended design and
> simply was not applied at the entry point, so one rename replaces changes at ~10 downstream `.name`
> sites. It also fixes `name=None` inputs. Docstring now states which names are normalised and which
> are kept.
>
> All three legal inputs the entry names were reproduced first (`FCT_UNSC`, `FCT_UNSC_gfXG`,
> `DAYTIME_FLAG` — each `KeyError`). Covered by `TestApplicatorAcceptsAnyInputSeriesName`, which
> asserts the corrected series is *numerically identical* across namings, not merely that it runs.
> Mutation-checked: all three `KeyError`s return. No H2O/LE leftover in this path. L41 is untouched
> and stays a one-line change (`fct_unsc_gf` is still `'FCT_UNSC_gfRF'` though the fill is XGBoost).


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

**[~] L76. BUR06 uses a canopy aerodynamic resistance where Burba 2006 specifies a per-element one, and drops the retained fraction `fr`**

> **Answered from the paper 2026-08-15** (Burba et al. 2006, AMS progress report, supplied by the
> user). Documented in `_flux_correction_term_unscaled_jar09_bur06`. **The entry's own conclusion —
> "the fitted SF absorbs both" — is half wrong, and measurement says so.**
>
> What the paper specifies: Eq. 8 carries `fr`, the fraction of instrument sensible heat retained in
> the optical path; Eqs. 10/11 give per-element forced-convection resistances
> `ra ~ 7.4*sqrt(d/U)` for the can (d=0.133 m) and ball (d=0.042 m); Eqs. 12-16 build `fr` from
> boundary-layer thicknesses and a Reynolds number.
>
> **`fr` being dropped is fine.** It is near-constant (~0.06 on the bundled CH-LAE record), and a
> near-constant factor is exactly what a fitted scaling factor absorbs.
>
> **The `ra` form is not fine, and the scaling factor does not rescue it.** In the paper's own
> formulation the U-dependence of `fr` and of `ra` largely cancel, so `fr/ra` is nearly flat in wind
> speed; the bulk `ustar**2/u` is not. Measured on CH-LAE (n=27331, U 0.2-11.9 m/s), the ratio of the
> two forms varies by a **factor of ~30 across USTAR classes** — absorbed, because `ScopOptimizer`
> fits one constant per class — but still spreads by a **factor of ~4.8 (p10-p90) within each of the
> 20 USTAR classes**, worst 9.3. A per-class constant cannot absorb a within-class spread. So the
> correction's wind-speed dependence is wrong by up to several-fold, and worst at low wind, which is
> when self-heating matters most (the paper reports gradients "may exceed 2 C per 1 mm" in cold, calm
> air).
>
> **Why it is still not simply a bug.** Sect. 5.2 explicitly sanctions solving Eq. 9 for the `fr/ra`
> ratio empirically against a closed-path reference *"in place of the Eqs. 9-16"* — which is what
> diive does. But it asks for that ratio *"for different wind speeds and directions"*, i.e. as a
> function; USTAR classes are only a partial proxy. And Sect. 5.1 cautions that Eqs. 10-16 assume a
> near-vertical instrument, laminar boundary layers and no flow obstruction, so the paper's form is
> not automatically better for a given site.
>
> **DEFERRED TO A FUTURE SESSION (user's decision, 2026-08-15). Do not pick this up as review
> follow-up work.** Open methodology questions remain beyond what this entry settles — the paper is a
> progress report, its Sect. 5.1 caveats are unresolved, and Burba et al. (2008) supersedes parts of
> it. The measurement below stands and is worth keeping; the choice it feeds does not belong to this
> campaign.
>
> When it is taken up, the scoped question is: implement Eqs. 10-16 as a selectable
> `ra` method and compare both against the reference instrument (CH-LAE has IRGA72 and IRGA75, so this
> is testable on the bundled data), or keep the bulk form and justify it on its own terms. Either way
> the residual within-class spread should be stated wherever this correction's uncertainty is reported.
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

**[x] L41. `ScopPhysics` documents an RF + MDV gap-fill that does not exist**
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

**[x] L43. Default fringe-bin trimming can empty the time-lag histogram; the `IndexError` escapes the batch helpers**

> **Fixed 2026-08-15.** `Histogram(method='uniques')` yields `n_unique - 1` bins, so
> `ignore_fringe_bins` is now compared against that count first. If trimming would remove every bin
> it is **skipped for that gas** with a visible `warn` naming the bin/value counts — trimming anyway
> and returning a peak from a partly-trimmed histogram would convert this crash into a quietly wrong
> number, which this document ranks as worse. Below two distinct lag values a `ValueError` explains
> why, and `analyze_all_gases` / `plot_all_gases` already catch `ValueError`, so the batch warns and
> continues as their docstrings promise instead of aborting on a bare `IndexError`.
>
> New test file `tests/test_timelag.py` (nothing existing fits — `TimeLagAnalysis` is `dv.flux`
> lowres). Mutation-checked: `IndexError: list index out of range` at `peak = peakbins[0]`. A related
> emptying path is filed separately as L81.


`timelag_analysis.py:148` — `ignore_fringe_bins or [5, 10]` drops 5 leading and 10 trailing bins;
`Histogram(method='uniques')` produces `n_unique - 1` bins, so a TLAG column with few distinct lag
values leaves nothing and `peakbins[0]` raises a bare `IndexError`. `analyze_all_gases` /
`plot_all_gases` catch only `ValueError`, so the whole batch aborts — contradicting their
docstrings' "Failed analyses … print warnings but do not raise exceptions".

**[x] L44. `TimeLagAnalysis` class docstring states three parameter facts the code contradicts**
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

**[x] L57. `reconstruct_from_components` forces the trend's NaN onto reconstructions that exclude the trend**
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

**[x] L60. Two more docstring claims the code does not honour**
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

**[x] L62. `show_less_xticklabels` is accepted and documented by `HeatmapDateTime` but never applied** ⚠ **[verified independently]**
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

**[x] L64. glTF / `.glb` export bakes the texture mirrored along the date axis**

> **Fixed 2026-08-15.** The UV row coordinate is flipped (`1.0 - v`) in both export paths
> (`_smooth_export_arrays`, `_extruded_export_arrays`), matching glTF/trimesh's lower-left texture
> origin; `u` untouched.
>
> Covered by `test_surface3d_export_texture_rows_not_mirrored`, which samples through the real
> `trimesh.visual.uv_to_color` on a monotone 4×2 grid (one texel per cell) and asserts vertex row
> `i` samples texel row `i` for both styles, plus a guard that the reversed mapping is wrong.
> Mutation-checked per path: reverting either gives exactly reversed row colours
> (`[255, 255, 170, …]` where `[0, 0, 85, …]` is required). Confirmed end-to-end through
> `_build_export_surface`: early dates now bake blue, late dates red.


`gui/tabs/surface3d.py:744` and `:817` — UV rows are assigned `v = i/(d-1)` (top-left texture
origin) but trimesh/glTF use a **lower**-left origin, so vertex row `i` samples texel row `d-1-i`.
Geometry is right, colours are flipped: an exported annual NEE surface paints the winter ridge with
the summer colours. Verified against the installed trimesh 4.12.2 with a monotone 4×2 grid — the
sampled row colours came back exactly reversed. Both the smooth and extruded paths are affected;
`u` is correct.

**[x] L65. `RidgeLinePlot` cannot plot any series that contains a gap**

> **Fixed 2026-08-15.** `dropna()` once in `__init__`, before grouping — one place rather than at the
> two `kde.fit` sites — so a group left with no valid values simply gets no ridge, and an all-NaN
> series raises a `ValueError` that says so. Docstring documents both. Covered by
> `test_ridgeline_plots_a_series_that_contains_gaps` (scattered gaps plus one month blanked → 11
> ridges, July absent). Mutation-checked: `ValueError: Input X contains NaN. KernelDensity does not
> accept missing values…`.


`core/plotting/ridgeline.py:196`, `:234` — `np.array(series)` goes straight into
`KernelDensity.fit`, which rejects NaN. Every gappy time series raises
`ValueError: Input X contains NaN`. The GUI path works only because `ridgeline_to_code` and the
plotting tab `.dropna()` first (`codegen.py:248`), so the library's public API is strictly worse
than the GUI's.

**[x] L66. `datetime_surface_grid` does not force `TIMESTAMP_START`, so the 3-D surface is offset from the 2-D heatmap**

> **Fixed 2026-08-15.** `datetime_surface_grid` now converts to `TIMESTAMP_START`, mirroring
> `HeatmapBase._setup_timestamp` (same `index.name` guard, same `insert_timestamp(convention='start',
> set_as_index=True)`). Converting — rather than correcting the docstring — is right because
> `_setup_timestamp` runs `TimestampSanitizer` with exactly the arguments this function already
> used; the missing conversion was the *only* divergence, and the surface is documented as the
> heatmap's 3-D analogue.
>
> Covered by `test_datetime_surface_grid_axes_match_the_heatmap`, asserting the grid's hours, dates
> and z all equal `HeatmapDateTime`'s for the same series (its x/y are pcolormesh boundaries, hence
> the dropped trailing bound). Verified independently outside the suite: hours `[0.5 1.5 2.5 3.5]`
> for both, dates and z equal.
>
> **Two pre-existing assertions in `test_datetime_surface_grid_shape_and_axes` moved with the fix**
> because they encoded the offset: `x_hours[0]` 0.0 → 0.5 and `len(y_days)` `3*365` → `3*365 + 1`.
> Both follow from the shift being *earlier*: the fixture's MIDDLE stamps sit at `:00`, so START is
> `:30` of the previous hour — the sorted hour list starts at 0.5 (23.5 sorts last) and the first
> record moves onto the preceding date, adding one leading row. The heatmap has always had that row.
>
> `surface3d.py` needed no change (it consumes the grid generically) and was owned by another
> change at the time; L68, in the same function, is untouched and still open.


`core/plotting/surface_grid.py:68` — the docstring claims "the same preparation the 2-D heatmap
uses", but `HeatmapBase._setup_timestamp` additionally converts to `TIMESTAMP_START` via
`insert_timestamp` (`heatmap_base.py:145`) while this only calls
`TimestampSanitizer(output_middle_timestamp=False)`, which leaves a `TIMESTAMP_MIDDLE` index alone.
Since MIDDLE is diive's working convention, the surface's time-of-day axis sits half a period later
than the heatmap's for the same data (`heatmap hours [0, 0.5, 1.0]` vs `surface [0.25, 0.75, 1.25]`).

**[x] L67. 3-D export buttons write a stale surface after a render that produced nothing**

> **Fixed 2026-08-15.** A new `_clear_grid_state()` helper clears the stashed
> `_grid_xn/_yn/_height/_z/_style` and is called from both early returns — `_compute`'s
> `data is None` branch and `_render_surface`'s `not finite.any()` branch — so the export handlers'
> `_grid_height is None` guard actually holds.
>
> Covered by `test_surface3d_export_state_cleared_when_render_shows_nothing`, exercising both
> branches and both classes (`Surface3DTab` with an all-NaN variable after a good one, and the
> `SurfaceXYZTab` subclass with a Z role that is not a real column). Mutation-checked per branch:
> reverting either leaves the previous variable's grid stashed (`shape=(21, 48)`, and the 30×30
> XYZ grid). The test stubs `Pyvista3DCanvas` because VTK cannot create a GL context under
> `QT_QPA_PLATFORM=offscreen`; mesh construction is pure CPU, so the pipeline still runs.


`gui/tabs/surface3d.py:945`, `:915` — `_render_surface` returns at `if not finite.any():` *before*
assigning `_grid_xn/_yn/_height/_z/_style` (`:963`), and `_compute` returns early when
`_grid_data()` is None; neither clears the previous grid. The export handlers guard only on
`self._grid_height is None`. Render a normal variable, then select an all-NaN one: the canvas
clears but "VR (.glb)" / "3-D print (.stl)" still write the *previous* variable's relief under a
filename built from the *current* target.

**[x] L68. `datetime_surface_grid` silently destroys a variable named `DATE` or `TIME`**

> **Fixed 2026-08-15.** Both files pivot through an internal value key
> (`series.rename('_values')`, `values='_values'`) — the `ScatterXY` `_x`/`_y`/`_z` precedent — so the
> `DATE`/`TIME` helper columns can never collide with the data column. L66's `TIMESTAMP_START`
> conversion is untouched.
>
> **This was filed at the wrong severity for its second half.** In `HeatmapDateTime._prepare_data`
> there is no crash at all: `to_numpy()` without `dtype=float` yields an object array, so the heatmap
> **silently paints the timestamps** and produces a plausible-looking wrong figure. By this document's
> rubric that half is S1, not S3 — the entry's own "the crash is the lucky outcome" remark was more
> literally true than it read.
>
> Covered by `test_datetime_surface_grid_keeps_a_variable_named_date` (subtests for both names,
> asserting the grid's and the heatmap's `z` equal the un-renamed series'). Mutation-checked per file:
> `TypeError: float() argument must be … not 'datetime.time'` for the grid, and
> `TypeError: unsupported operand type(s) for -: 'datetime.time' and 'float'` with only
> `heatmap_datetime.py` reverted — proving the heatmap half is genuinely covered.


`core/plotting/surface_grid.py:71` — `df["DATE"] = df.index.date` overwrites the data column when
`series.name` is one of those. The crash (`TypeError: float() argument must be … not
'datetime.time'`) is the lucky outcome; the defect is the silent overwrite.
`HeatmapDateTime._prepare_data` (`heatmap_datetime.py:96`) has the identical construction.

**[x] L69. `Cumulative.plot` raises on an all-NaN column**

> **Fixed 2026-08-15.** `valid = series.dropna()` computed once; an empty column is labelled
> `"{col}: no data"` so the legend keeps one entry per column and states there is nothing there, and
> the end-point marker/annotation is skipped rather than annotating `nan`. The same one-liner was
> applied to `CumulativeYear.plot`, which this entry names as the same pattern. Covered by
> `test_cumulative_labels_an_all_nan_column_instead_of_raising`. Mutation-checked twice, so both
> halves bite: `IndexError: single positional indexer is out-of-bounds` for the label, and
> `'nan' unexpectedly found in '262800 nan'` for the marker guard.


`core/plotting/cumulative.py:327` — `series.dropna().iloc[-1]` inside an f-string label indexes an
empty Series when a column has no valid values (normal for a scenario column that was never filled).
Same pattern at `cumulative.py:186` in `CumulativeYear`. `IndexError: single positional indexer is
out-of-bounds`.

**[x] L70. Rolling cell aggregator uses an `n+1`-row window for even `n`**
`gui/tabs/surface3d.py:98` — `z[max(0, i-half) : i+half+1]` with `half = n//2` gives five rows for
`n = 4`. The docstring says "a centred rolling window of `n` rows" and the tooltip says "the window
width"; every even setting of the "Y cell (days)" spin box smooths one day wider than requested.

**[x] L71. `convert_ts_to_timezone` cannot accept the `DatetimeIndex` its docstring promises**
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
**All eight were reviewed in round 4 (2026-08-16) — see the *Round 4* section. Nothing in the
reviewed scope is now unreviewed.**

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

---

# Round 4 — found while fixing S3 (2026-08-15)

Six defects noticed *adjacent* to an S3 fix and deliberately left alone, so that each fix stayed
surgical. Each was reported by the agent that found it rather than folded into an unrelated commit.
None is reproduced beyond what its note says.

**[x] L77. `MultiDataFileReader.metadata_df` tests the wrong attribute**
`core/io/filereader.py` — the `metadata_df` property's emptiness guard checks `self._data_df`, not
`self._metadata_df`, so it reports on the wrong frame. Found while fixing L22, and now unreachable
via that path (L22's `ValueError` fires first), which is exactly why it should be recorded rather
than left to be rediscovered. Both properties also still carry bare
`raise Exception('data is empty')` fallbacks.

**[x] L78. `percent_matching` / `confidence` are recovered by parsing a formatted string**
`core/times/times.py` — `DetectFrequency` stores its match rate as `'{:.0f}% occurrence'` and later
parses the number back out of that string, so both values are rounded to whole percent: a genuine
99.9% is reported as `100.0`, indistinguishable from a perfectly regular record. Found while fixing
L3, which corrected the *denominator* feeding this; the rounding is a separate defect and survives
the fix. S4 — the number is defensible, the precision it implies is not.

**[x] L79. `transform_yearmonth_matrix_to_longform` rejects non-contiguous month columns**
`core/dfun/frames.py` — with non-contiguous months `pd.infer_freq` returns `None`, so the
`freqstr == 'MS'` guard trips and the function raises
`ValueError('Failed building monthly timestamp for long-form time series.')`. Reachable from its own
documented producer: a seasonal record (e.g. June–August across 2018–2020) makes
`resample_to_monthly_agg_matrix` emit a 3-column matrix that this function then refuses. The
docstring's own example (`MONTH 1 2 3`) hits the same path. Found while fixing L13.

**This is the fourth instance of the shape already called out under *Cross-cutting observations*** —
"the aggregator kept only the bins that occurred, and the consumer treated them as contiguous"
(L61 `HeatmapYearMonth`, L63 `GridAggregator`). Worth fixing as that family rather than one-off.

**[x] L80. `UstarVekuriThresholdDetection.bootstrap_results_` is never read**
`flux/lowres/ustar_vekuri_detection.py` — initialised in `__init__` and never assigned or read;
`bootstrap()` builds its own local `boot_results`. Dead attribute, not deleted per the
"mention, don't delete" rule. Found while fixing L33.

**[x] L81. `TimeLagAnalysis`: a bin range excluding every lag empties `results`**
`flux/lowres/timelag_analysis.py` — with `histogram_startbin`/`histogram_endbin` set so that no lag
value falls inside, `results` ends up empty and `detect_peak_range` fails on the zero-size array.
A second route to the emptiness L43 fixed for `ignore_fringe_bins`, through different parameters.
Overlaps L44's territory (the `TimeLagAnalysis` docstring/parameter contract), so it belongs with
that entry.

**[x] L82. An exception inside a Qt-invoked slot is swallowed, so GUI tests can pass over a crash**
`gui/` (methodology, not one file) — PySide6's signal machinery absorbs exceptions raised inside a
slot it invokes, so a test that exercises a widget *through a signal* (e.g. toggling a checkbox whose
`toggled` handler raises) sees no failure. G2's regression test calls `_rerender_last()` directly for
this reason. **This casts doubt on any existing GUI test that drives behaviour via signals and
asserts only that nothing raised** — those tests may be weaker than they look. Worth an audit of
`tests/test_gui.py` for that pattern. Found while fixing G2.

---

# Round 5 — found while fixing S4/S5 (2026-08-15)

Fourteen defects noticed *adjacent* to a fix and deliberately left alone. Three are S2 (silent), and
those three are the ones to take first.

**[x] L86. `UstarVekuriThresholdDetection.bootstrap()` is not reproducible**

> **Fixed 2026-08-15.** New `random_state: int | None = 42` on `__init__`, threaded into the
> resample as `self.random_state + boot_idx` so the draws still differ from each other while the run
> as a whole is reproducible; `None` restores the old non-deterministic behaviour. Covered by
> `TestVekuriBootstrapIsReproducible` — one test that two same-seed runs agree, and one that two
> *different* seeds disagree, so the seed is proven to reach `sample` rather than the first test
> passing because the bootstrap collapsed to a constant. Mutation-checked: unseeded gives
> `DataFrame.iloc[:, 0] (column name="mean") values are different (50.0 %)`.
>
> **The same defect is still open in `UstarBootstrapThresholds` — see L97, and it matters more:**
> that is the class the flux chain uses for CUT/VUT detection.
`flux/lowres/ustar_vekuri_detection.py` — calls `df.sample` with no `random_state`, so the bootstrap
percentiles differ run to run (observed: one season's p50 moved 0.1634 -> 0.1488 across two runs).
These thresholds feed u\* filtering and therefore the whole flux chain. CLAUDE.md is explicit that the
rf/xgb seeds are pinned *because* output drifts without them; this path was missed. Found while fixing L80.

**[x] L87. `classical_decompose` passes a parameter name that does not exist**

> **Fixed 2026-08-15 by removing the dead branch, NOT by enabling extrapolation.** The
> `try: extrapolate='freq' / except TypeError: <no extrapolation>` pair always took the fallback, so
> the behaviour users have always had is the no-extrapolation one; the branch merely read as if
> extrapolation happened. Measured before deciding: `extrapolate_trend='freq'` fills all 30 edge NaN
> in both trend and residual and leaves interior trend values identical.
>
> Enabling it was the first attempt and was reverted. The L2 analogy in this entry does not hold —
> `hist_n_bins` was a *caller's* argument being ignored, whereas `extrapolate` was an internal
> literal no user ever passed, so no expectation was being violated. Switching it on would fill
> (period-1)//2 records at each end with a least-squares extrapolation indistinguishable from a
> measured trend, and shift `seasonality_strength` because the residual variance would gain those
> records. The docstring now states the NaN edges are deliberate and why.
>
> Covered by `TestClassicalDecomposeRequestsNoExtrapolation`: one test pinning the NaN counts, one
> pinning the interior trend against a manual centred rolling mean so enabling extrapolation later
> cannot pass unnoticed. Mutation-checked in the meaningful direction — switching extrapolation on
> fails 3 tests, two of them **L57's**, which confirms the NaN edges are load-bearing for that
> contract rather than incidental. Reinstating the original misspelling is behaviour-identical by
> construction and no test can catch it; that is the nature of a dead-code fix.
`core/times/decomposition_utils.py:207` — passes `extrapolate='freq'` to `seasonal_decompose`, whose
parameter is `extrapolate_trend`. It therefore *always* raises `TypeError` and *always* falls into the
no-extrapolation fallback, so the trend edges are unconditionally NaN. A silently-ignored argument, the
same species as L2. This dead kwarg is what made **L57** visible on real data. Found while fixing L57.

**[x] L92. `ScreeningTabBase._select` does not bump `_run_id`**

> **Fixed 2026-08-15.** `_run_id += 1` in `_show_variable`, not `_select`: it is the common path for
> both a user pick and a data reload, and it is already where the "clear any prior run (it's now
> stale)" logic lives — its docstring claimed the staleness the code did not enforce. G5's `_running`
> flag does not cover this: switching variable does not stop the running worker, so the stale result
> still arrives and only the id check can reject it.
>
> Covered by `test_stepwise_screening_discards_a_run_for_the_previous_variable`, which blocks a
> stand-in worker mid-run (no sleeps), switches variable, then hands the stale payload to `_on_done`
> **directly** rather than through a signal — per L82, a signal-driven version could pass over a
> crash. Mutation-checked: `AssertionError: switching variable must invalidate the run in flight`
> / `assert 1 != 1`.
`gui/tabs/_screening_base.py` — switching the selected variable mid-run can still adopt the previous
variable's chain result, with no exception. This is **G2's bug, in the tab this document cites as the
correct `run_id` pattern to copy**. Found while fixing G5.

**[x] L83. `histogram_startbin` / `histogram_endbin` are seconds, not bin indices**
> **Fixed 2026-08-15** (`76674429`). Renamed to `histogram_start_seconds` / `histogram_end_seconds`;
> the `histogram_` prefix stays because these bound the analysed histogram, a different range from the
> neighbouring `lag_window_min`/`lag_window_max`. An old name raises a `TypeError` naming its
> replacement, the way the outlier detectors answer their pre-unification names, while anything else
> still raises the normal unexpected-keyword error so a typo cannot pass through `**legacy`.
>
> **A second defect fell out of it:** the GUI fields were `QSpinBox`, i.e. integer-only, so a 0.40 s
> bound could not be entered at all. Now `QDoubleSpinBox`, labelled "(s)". A misleading name had
> produced a matching wrong widget. Two CHANGELOG bullets this rename falsified were corrected too.
`flux/lowres/timelag_analysis.py` — L44 corrected the types and docstrings, but the parameter *names*
still say "bin". Renaming is a public-API change, deferred.

**[x] L84. `ignore_fringe_bins` described as "bin indices" in `__init__`**

> **Fixed 2026-08-15.** Doc-only. The `__init__` entry now says the values are
> [leading, trailing] *counts* of bins to drop, not indices, states the real default
> (`None`, meaning `[5, 10]`), and mentions that trimming is skipped with a warning when it
> would empty the histogram (L43).
`flux/lowres/timelag_analysis.py` — `Histogram` treats `[i, j]` as *counts* of leading/trailing bins to
drop. L44 fixed the class docstring; the same loose wording survives in `__init__`.

**[x] L85. Docstring examples are not executed by anything**

> **Fixed 2026-08-16.** `tests/test_docstring_examples.py` runs the samples. It parses the tree for
> sample-bearing modules, imports them, and executes every `>>>` sample through `doctest` unless its
> name is in a module-level `SKIP` dict carrying a one-line reason. A new sample is therefore
> executed without anyone remembering to register it, and a second test fails on a `SKIP` key that
> no longer matches a real sample, so the exclusion list cannot rot. 0.7 s for the doctest pass.
>
> **The `>>>` samples: 34 became 30.** Twelve were dropped as internal helpers, none reachable from
> `dv.*`: `ColumnNamesSanitizer` and `MultiDataFileReader` in `core/io/filereader.py`, eight
> `core/times/times.py` step functions, and both `FluxStorageCorrectionSinglePointEddyPro` samples.
> `Args:`/`Parameters:` text was left alone, so the GUI tooltips are untouched. Every keeper is a
> public entry point and a self-contained call-shape snippet of at most four statements — 17 of the
> original keepers referenced names they never defined (`df`, `data`, `ax1`, `radiation`,
> `corrupted_data`), fixed with a two-line setup rather than a `doctest_namespace` fixture.
> **17 execute; 13 are excluded**, each with its reason in `SKIP`: `GrangerCausality` (statsmodels'
> `grangercausalitytests` prints its own report to stdout, which doctest reads as unexpected
> output), `TimeSeries.plot_interactive` / `.plot_rangetool` (bokeh `show()` opens a browser tab),
> the three USTAR detectors, the four partitioning ports (~20 s/year), `GapFillingResult` (trains a
> random forest on the full record), `run_chain` and `add_driver` (need a built `FluxLevelData`).
>
> **The reST half was the bigger hole**, and it is why L133 exists. `test_docstring_refs` and the new
> runner both keyed off `>>>`, so a sample written as an `Example::` literal block was checked by
> nothing. A grep for `Example::` finds 13 such blocks — but reST introduces a literal block with
> *any* line ending in `::`, and there are **41**, including
> `Example (recommended with GridAggregator)::`. Two things close it: six blocks that name public API
> were converted to `>>>` (`GapFillingResult`, `HeatmapXYZ.from_gridaggregator`, `TreeRingPlot`,
> `WindRosePlot`, `run_chain`, `add_driver`), and `test_docstring_refs` now resolves `dv.<attr>`
> inside literal blocks **and inside attribute docstrings**, which are invisible at runtime and so
> unreachable by `doctest` at all. Seven blocks stay literal on purpose: five `container.py` samples
> that call methods on a `data` the reader already holds and name no importable symbol, the
> `FluxConfig.level2_test_settings` dict literal, and the `swin.py` `feature_kwargs=` fragment, which
> is not even a statement. reST directives (`.. note::`, `.. warning::`) are skipped — their bodies
> are prose.
>
> **Eight broken samples, every one documentation wrong against correct code.** Two among the `>>>`
> samples: the `FlagQCF` module sample documented `series=` / `outname=` / `swinpot=`, a signature the
> class has not taken in a long time (`TypeError` on the first line), and `classify_variable('TA_f')`
> was documented as returning `None` when the `TA_` rule classifies it as meteo. Six in the reST
> blocks: `WindRosePlot` called a `dv.load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN` that is not a
> top-level export (**L133**); `TreeRingPlot` called a non-existent `dv.plot_treering` *and* passed a
> `title=` removed in v0.91.0; `HeatmapXYZ.from_gridaggregator` called `dv.ga` and `dv.heatmap_xyz`,
> neither of which exists, and passed `show_values=` to Phase 1 where it raises `TypeError`; and the
> `heatmap_xyz` and `hexbin` **module** docstrings repeated the same dead names — found only after the
> literal-block check went in, which is the argument for that check.
>
> Two side findings fixed while there. Five prose "Top-level alias" claims named functions that do not
> exist (`dv.plot_heatmap_xyz`, `dv.hexbinplot` twice, `dv.heatmap_datetime`, `dv.plot_treering`), all
> corrected to the real `dv.plotting.*` names. And `HeatmapXYZ.from_gridaggregator`'s `**kwargs` entry
> advertised `figsize` / `cmap` / `vmin` / `vmax` / `show_values` — every one a Phase-2 `plot()`
> argument that raises `TypeError` if passed to Phase 1.

> **First step done 2026-08-15; the entry stayed open because nothing *executed* a sample yet.**
> `tests/test_docstring_refs.py` now checks the two things that can be checked without any doctest
> infrastructure: every `examples/...` pointer resolves on disk, and every `dv.<attr>` inside a `>>>`
> line resolves on the real public API. Milliseconds to run, and it would have caught L35.
>
> Landing it required fixing what it found — 43 pointer occurrences across 30 dead paths (a folder
> rename), and 7 bad `dv.` references (`dv.TimestampSanitizer`, `dv.TimeSince`,
> `dv.plot_longterm_anomalies_year`, the last of which never existed).
>
> **A survey of all 34 samples informs what is left** (2026-08-15): 11 of 34 were faulty, 7 badly
> enough to raise on the first line; 29 of 34 reference names they never define, so a doctest runner
> needs a `doctest_namespace` fixture and most samples need rewriting; 7 are too heavy to execute
> (bootstrap loops, the partitioning ports at ~20 s/year). Also established: `core/utils/docstrings`
> and the GUI tooltips read **only** `Args:`/`Parameters:` text, never the `>>>` block, so pruning
> samples cannot change the GUI — but Sphinx autodoc does publish them.
>
> Agreed direction: keep a ≤4-line call-shape snippet on public entry points plus the pointer, drop
> the ~15 samples on internal helpers, and make only the cheap subset executable — a pointer cannot
> serve `help()` or an IDE hover, which is where the question gets asked.
`tests/` — there is no doctest runner (`grep -rn doctest` over the config and tests is empty), and
`test_docstrings.py` only tests the tooltip-extraction helpers. **L35 was a docstring example that had
been broken twice over** (wrong namespace *and* a column that does not exist in the example data) and
nothing caught it. A doctest or example-execution pass would have.

**[x] L88. `detect_seasonality`'s `'strength'` is mis-documented**

> **Fixed 2026-08-15.** Doc-only; the code is a legitimate measure, the description was of a
> different one. It is the share of periodogram power in the detected peaks,
> `sum(power at peaks) / sum(power over all periods)` — a power ratio, not the variance ratio the
> docstring claimed. The docstring now also says it is **not** the same quantity as
> `seasonality_strength` in `analysis/seasonaltrend.py`, which works on the decomposed
> components; having two differently-defined "strength" numbers is the trap L60 already hit.
`core/times/decomposition_utils.py` — documented as "seasonal var / total var", actually a
peak-power / total-power ratio. Same species as L60(a), not named in that entry. Found while fixing L60.

**[x] L89. `-9999` at position 6 yields flag 0**

> **Fixed 2026-08-15.** Negative codes are now treated as missing, like NaN. Reproduced first,
> and it was **two** holes rather than the one this entry names: `-9999` at position 6 read as
> `0.0` ("tested and good"), and `-1` at position 1 read as `1.0` (a soft warning). The cause is
> that the digit is taken from the string form, where the minus sign shifts every position —
> `-9999` becomes `'-9999.0'`, whose character 6 is `'0'`.
>
> Deliberately still not touching the float->string round-trip itself, as L25 decided: one
> `where(flag >= 0)` fixes the reachable defect without changing how valid codes are parsed.
> Covered by `test_negative_code_is_not_testable`, which checks both positions via subTest and
> asserts a real flag at the same position is still read, so the guard is not a blanket veto.
> Mutation-checked: both subtests fail with the guard removed.
`preprocessing/qaqc/eddyproflags.py` — L25 fixed the scalar-`0` case narrowly and deliberately did not
touch the float->string round-trip; a junk `-9999` value still reads as a passing flag at some positions.

**[x] L91. `hexbin.py` also accepts `show_less_xticklabels` and never applies it**

> **Fixed 2026-08-15.** The same hide block `HeatmapDateTime` uses (L62), placed after
> `format()`, which is what sets the tick labels. `HeatmapBase` only *stores* the flag — every
> subclass has to apply it, which is why this went missing in one of the three.
> Covered by `TestHexbinShowLessXticklabels`: one test that the thinned labels equal every second
> original label, one that the flag off changes nothing. Mutation-checked: both fail with the
> block removed, the second showing the two label lists identical.
`core/plotting/hexbin.py:272` — accepts, documents and forwards it, and nothing applies it. L62's exact
defect in a second file. Found while fixing L62.

**[x] L93. `EventManager.load_dict({})` does not clear**

> **Fixed 2026-08-15.** An empty load now clears the events and emits `changed`, matching the
> non-empty path. A load replaces state, so opening a project with no events must not leave the
> previous project's standing. Both callers pass `{}` when nothing is saved, and the project-open
> path (`project.extras.get("events") or {}`) is the one that leaked; the startup path is a no-op
> because events are already empty there.
> Covered by `test_event_manager_load_dict_clears_on_empty`, which also asserts a non-empty load
> still restores, so the change is not a blanket wipe. Mutation-checked:
> `AssertionError: an empty load must clear the previous events`.
`gui/events.py` — early-returns on an empty dict, so opening a project with no events keeps the previous
session's events. Adjacent to G9. Found while fixing G9.

**[-] L94. `crosscorr`'s `len(pot_arr) == 0` branch is unreachable**

> **Closed 2026-08-15 as annotate-don't-delete.** The branch is unreachable as analysed, and L21
> already made it consistent with the other early-outs (a NaN row, not an omitted date). Deleting
> a defensive guard buys nothing and would turn a wrong-but-safe path into a crash if the
> reachability analysis is off, so it stays with a comment recording why. CLAUDE.md's rule is to
> mention dead code rather than remove it; this entry was that mention, and the comment carries it
> to where a reader will see it.
`preprocessing/qaqc/detect_timestamp_shifts.py` — `window` always spans at least the sun-up rows, which
the preceding check guarantees non-empty. L21 made it consistent anyway; it is dead code.

**[x] L95. `ScopPhysics.fct_unsc_gf` carries a legacy RF name**
> **Fixed 2026-08-15** (`20cb27f3`). Straight rename to `FCT_UNSC_gfXG`, no alias: two columns
> differing only by a regressor tag invite publishing numbers from the one that names the wrong method,
> and v0.91.0 is unreleased with breaking changes already listed.
>
> **The root cause was worth more than the rename.** The suffix was hardcoded in two places - once as
> `ColumnConfig`'s results key, once as `_gapfill`'s lookup into `XGBoostTS` output - which is why they
> drifted. `_gapfill` now looks the column up under `ColumnConfig` itself, so the emitted column, the
> series `.name` and the lookup key are one string by construction: a future regressor change raises
> `KeyError` rather than shipping a mislabelled column. Confirmed the regressor is genuinely fixed
> (`_gapfill` instantiates `XGBoostTS`; no argument selects it). See L103 for the applicator's
> remaining mislabel.
`flux/lowres/selfheating.py` — `ColumnConfig.fct_unsc_gf` is `'FCT_UNSC_gfRF'` while the fill is
XGBoost, so `physics.fct_unsc_gf.name` is `'FCT_UNSC_gfXG'` but `get_results()` carries `FCT_UNSC_gfRF`.
L41 documented this rather than renaming: the name is indexed by two examples, the generated docs and
`tests/test_selfheating.py`, so a rename is **breaking**. Deferred, not forgotten.

**[x] L96. `ScatterXY` uses `plt.colorbar`**

> **Fixed 2026-08-15.** `ax.figure.colorbar` in `ScatterXY`, and `fig.colorbar` +
> `fig.tight_layout()` in `DetectTimestampShifts.plot_radiation_fingerprint`, which had the same
> defect plus a `plt.tight_layout()` on the same wrong figure. Verified by passing an axes belonging
> to a bare `Figure()` (not created through pyplot) and asserting no warning; mutation-checked, the
> warning returns. A repo-wide sweep found the two remaining `plt.colorbar` callers create their own
> figure and accept no `ax`, so pyplot's current figure is correct there. The rule is now in
> CLAUDE.md's **Plotting** conventions rather than buried in one tab's bullet, which is why it was
> missed twice.
`core/plotting/scatter.py:169` — against CLAUDE.md's rule that library plots use `ax.figure.colorbar` so
they embed in a GUI figure; emits `UserWarning: Adding colorbar to a different Figure`. Found while
auditing for L82.

## Also for the CHANGELOG (behaviour changes from this round)

- **L79** — `transform_yearmonth_matrix_to_longform` now always returns a full-year span, so a
  partial-year matrix that was contiguous within one year is padded to 12 months with NaN.
- **L60** — the `'iterations'` key is gone from `stl_decompose`'s result dict (it held a shape tuple,
  not a count).
- **L6** — random-uncertainty method 4 now averages 10 neighbours rather than 9; affects only records
  that fall through to method 4 (~0.01% on CH-DAV).
- **L11** — a joint-uncertainty cumulative record with incomplete scenario inputs is now NaN rather
  than a wrong number.
- **L78** — `frequency_percent_matching` / `confidence` are no longer rounded to whole percent.
- **L82's fix** — `VariablePanel` no longer leaks a dead slot per closed tab; in the real GUI this was
  raising and swallowing one `RuntimeError` per closed tab on every metadata edit.

**[x] L97. `UstarBootstrapThresholds` resamples unseeded, in both of its paths**

> **Fixed 2026-08-15.** New `random_state: int | None = 42` on `UstarBootstrapThresholds`, and the
> same on `UstarMovingPointDetection` — which turned out to be a **third** unseeded site this entry
> did not name: its own `bootstrap()` built `np.random.default_rng()` with no seed
> (`ustar_mp_detection.py:660`). Both wrapper paths are now seeded: the fast path gets
> `rng=np.random.default_rng(seed)`, and the generic loop gets `random_state=seed + i` so its draws
> differ from each other the way the fast path's single Generator already did.
>
> **The seed is derived per window year** (`_seed_for`, `random_state + year`), which was the open
> design question. A single shared seed would be deterministic but wrong in a subtler way: every
> window would resample the same positions, correlating years that are meant to be independent
> draws. Deriving from the year also makes the result independent of `n_jobs` and of the order
> windows finish in — asserted, and measured identical for `n_jobs=1` vs `2`.
>
> Covered by `TestBootstrapThresholdsAreReproducible`: same seed agrees, *different* seeds disagree
> (so the seed is proven to reach both paths), serial equals parallel, and the per-year seeds are
> distinct with `None` propagating. Mutation-checked: reverting both paths gives
> `DataFrame.iloc[:, 0] (column name="p16") values are different (100.0 %)`.
>
> Since the default is 42, `run_chain`'s CUT detection (`level33.py`) becomes reproducible with no
> change at the call site. No threshold algorithm was touched, so ONEFlux parity is unaffected —
> seeding chooses which resamples are drawn, not how a threshold is computed.
>
> One self-inflicted bug found and fixed en route: the first edit put `_seed_for` inside `__init__`,
> which left the results-storage assignments after it unreachable. `test_vut_before_run_raises`
> caught it.
`flux/lowres/ustar_bootstrap.py:50` — the generic loop calls
`df_window.sample(n=..., replace=True)` with no `random_state`, and the fast path
(`:36`) calls `detector.bootstrap_annual_samples(n_iter)` without the `rng` that method
**already accepts** (`ustar_mp_detection.py:598`). So VUT and CUT thresholds differ between
runs. Found while fixing L86, and this is the more consequential half: `run_chain` does its
CUT detection through this class, whereas the Vekuri detector is a standalone tool.

Not fixed with L86 because it needs a design decision L86 did not: with `n_jobs > 1` the
years run in parallel, so a single seed must be **derived per year** (`seed + year`, or a
`SeedSequence` spawn) — one shared seed would make every year draw the same resample
indices, which is worse than unseeded. Inventing that scheme silently for the flux chain's
u\* thresholds is not a one-liner, so it is recorded instead.

**[x] L98. The example suite reports spurious timeouts: `plt.show()` blocks**

> **Fixed 2026-08-15.** `run_all_examples.py` now runs each example with `MPLBACKEND=Agg`, and a
> caller-set `MPLBACKEND` still wins so the plots can be watched deliberately. Also corrected the
> timeout message, which said 60 seconds against a 120 s timeout.
>
> **This entry's original wording was wrong and is corrected here:** the runner has a 120 s
> `subprocess.run` timeout, so the suite did *not* stall forever. Each of the 16 blocking examples
> burned the full 120 s and was then reported as a **timeout failure it had not earned** — worse in
> one way than a hang, because the run completes and the report looks like 16 broken examples.
>
> Measured on `analysis/analysis_gapstats.py` through the runner's own `run_example()`:
> `timeout, 120.0s` with a window backend versus `pass, 1.8s` with the new default. 16 of the 113
> listed examples end in a bare `plt.show()`.
>
> CLAUDE.md's "NEVER RUN EXAMPLE SUITE" rule is kept — 113 examples is genuinely expensive — but it
> now states that the reason is cost rather than breakage, and the documented single-example command
> carries `MPLBACKEND=Agg`, since running one of those 16 by hand hits the same block. Headless, that
> block is indistinguishable from a slow example; the giveaway is an output file that stops growing.
`examples/run_all_examples.py` — sets no matplotlib backend, and **16 of ~76 examples end with a
bare `plt.show()`**. With PySide6 installed the Qt backend is the default, so `plt.show()` blocks
until a human closes the window. Running the suite headless stalls on the first such example
indefinitely — not slowly, permanently. Found while verifying `outlier_stepwise.py` after L12's
fix: two log checks 20 minutes apart were byte-identical, and `MPLBACKEND=Agg` made the same
example finish in seconds (exit 0).

This is distinct from the `showplot=True` item on CLAUDE.md's example checklist — the detectors in
these examples all pass `showplot=False` correctly; it is the trailing hand-rolled figure that
blocks. **It may also be the unstated reason for CLAUDE.md's "[CRITICAL] NEVER RUN EXAMPLE SUITE"
rule**, which currently gives none; if so, the rule and this finding should cross-reference.

The fix is one line in the runner (force `Agg`, or set `MPLBACKEND` in its environment), but it
changes how the suite behaves for someone running it interactively to look at the plots, so it is
recorded rather than done. The alternative — dropping `plt.show()` from 16 examples — is worse:
it is what a human running one example by hand actually wants.

**[x] L99. Running the test suite overwrites the developer's real GUI preferences**
> **Fixed 2026-08-15** in two parts. The test side in `ebdf57f0`: a session fixture in
> `tests/conftest.py` redirects `config.config_file`, so the suite no longer overwrites the developer's
> real preferences. The library side in `2c84d943`: `save_config` serialises first (so an
> unserializable value fails before touching the filesystem), then writes a uniquely named temp file in
> the target's own directory, flushes, `fsync`s and `os.replace`s it - `os.rename` is not atomic over
> an existing file on Windows - cleaning up the temp file on every failure path.
>
> G8's `(OSError, TypeError, ValueError)` swallowing is unchanged, and its promise that the previous
> file survives a `TypeError` is now actually true rather than accidentally true. A further defect
> found on the way: `load_config` returned valid-JSON-that-is-not-an-object (`null`, a list) as-is, it
> reached the first `cfg.get()` and crashed startup; now falls back to `{}`.
`gui/config.py` + `tests/test_gui.py` — three GUI tests call `win.close()`. `MainWindow.closeEvent`
calls `config.save_config()`, which writes theme, geometry, `last_project` and `variable_metadata` to
the **live** `QStandardPaths` application-config file, with a non-atomic `write_text`. So running the
test suite silently replaces the developer's own saved preferences with whatever state a test
happened to leave behind, and a crash mid-write truncates the file.

Two separate defects in one: the tests should not touch the real config path (a session fixture
redirecting it to a tmp dir fixes that side), and `save_config` should write atomically — write to a
temporary file in the same directory and `os.replace` — so an interrupted save cannot destroy an
existing config. The second half is a library defect independent of the tests, and G8 already touched
this function's error handling without noticing it.

Found while profiling the test suite for runtime, not by looking for it.

**[x] L100. Four test modules inherit their matplotlib backend from an unrelated module**
> **Fixed 2026-08-15** (`ebdf57f0`). `matplotlib.use("Agg")` at `tests/conftest.py` import time, before
> any test module is imported, so the backend is stated once instead of inherited from whichever module
> sorted first.
`tests/` (methodology) — the default backend in this environment is `qtagg`, not `Agg`.
`test_analyses.py`, `test_heatmap_xyz.py`, `test_hexbin_plot.py` and `test_selfheating.py` never set
one, and pass only because an alphabetically earlier module (`test_corrections.py`,
`test_events.py`, …) calls `matplotlib.use("Agg")` first and the backend is process-global.

Same family as the console-rebinding order dependency fixed in `752e42a8`: correct serially by
accident, and it breaks the moment tests are distributed across processes or run as a subset. The fix
is one `matplotlib.use('Agg')` in a `conftest.py`, before any test module imports. Related to L98 —
the example suite had the same "no backend pinned" problem from the other direction.

**[x] L101. Two examples were broken by this campaign and nobody could see it**
> **Fixed 2026-08-15** (`21b4e2a5`, `d9435e3c`). `ScopOptimizer` no longer takes the
> `latent_heat_vaporization` kwarg at all - it was only the umol->W conversion for the removed LE path
> - so dropping it is the whole functional fix and nothing the examples taught for CO2 is lost.
>
> Both examples were corrected beyond the crash, recorded here because it goes past the finding: the
> quickstart claimed Random Forest gap-filling (it is XGBoost) and had no reference to judge the
> correction against; the production example printed a non-cp1252 character that would have raised
> `UnicodeEncodeError` the moment the `KeyError` was fixed, and its "Phase 2" fitted and re-applied to
> the *same data* while claiming to teach transfer - it now calibrates on one half-year and applies to
> a held-back one. Hardcoded scaling-factor ranges belonging to the old period were removed.
> 166 s-then-crash -> exit 0 in 4 s and 6 s. See L104 for the `plt.show()` issue underneath.
`examples/flux/lowres/flux_selfheating.py:98` and `flux_selfheating_production.py:136` both do
`results_physics_df["LATENT_HEAT_VAPORIZATION_J_UMOL"]` and raise `KeyError`. That column was removed
by **`45614fb3` `feat!: remove the H2O self-heating path`** — L37's fix, from this review. `grep`
confirms the name exists nowhere in `diive/` any more.

**The interesting part is why it went unnoticed for the whole campaign.** L98: the example runner
pinned no matplotlib backend, so 16 examples blocked on `plt.show()` and were reported as timeout
failures. Against that noise, two genuine failures were indistinguishable from the spurious ones —
and CLAUDE.md's "NEVER RUN EXAMPLE SUITE" rule meant nobody looked. Fixing L98 is what made these
visible: the after-run reports exactly 2 failures instead of 3-plus-noise.

So this is not really an S3 crash on its own; it is the cost of a breaking change landing without the
one check that would have caught it. The *fix* is small (drop or replace those two lines, since LE
self-heating no longer exists as a concept — see the `no-selfheating-correction-for-le` note), but it
belongs with a decision about what those two examples should now demonstrate, which is why it is
recorded rather than patched blind.

Found by running the example suite for timing, not by looking for it.

**[x] L102. `show_values` has no cell-count guard, so it can appear to hang the GUI**
> **Fixed 2026-08-15** (`7789c7f6`). `show_vals_in_plot` skips above `SHOW_VALUES_MAX_CELLS = 2000`,
> overridable per call as `plot(show_values_max_cells=...)`, `None` for no limit. The limit was measured
> rather than guessed - 144 / 1000 / 2000 / 4000 / 17520 cells - at the point where both the render and
> the following redraw stay under a second.
>
> Two deliberate choices: the guard lives in the library on the grid size, so it covers every caller
> including `HeatmapXYZ` and plain scripts rather than only the GUI; and it **warns** instead of
> skipping quietly, which would have converted a visible slowness into a silently ignored parameter,
> the L2/L62 defect class. The warning passes `verbose=self.verbose or None`, because
> `HeatmapBase.verbose` defaults `False` which resolves to silent and would have hidden the one
> explanation for the missing labels. Year/month heatmaps cannot reach the limit (12 cells a year) and
> the GUI test's 3-day range is ~144 cells, so both still label.
`gui/widgets/plot_settings.py` (the checkbox) + `core/plotting/heatmap_base.py`
(`show_vals_in_plot`) — the overlay writes one text artist per cell. That is fine for a
year/month heatmap (12 x N cells, which is what it was designed for) and unusable for a
date/time heatmap, where one year of half-hourly data is **17 520 cells**.

Measured on the GUI test fixture (one year, CH-DAV):

| | render | next tab open | text artists |
|---|---|---|---|
| `show_values=True` | 15.84 s | 43.12 s | 17 520 |
| `show_values=False` | 0.22 s | 1.15 s | 0 |

The second column is the real problem: the artists persist, so **every subsequent
re-layout in the whole application** walks them. A user ticking the box on a multi-year
heatmap gets no feedback and a frozen window, and would reasonably force-quit. 17 520
overlaid numbers are also illegible, so the render is expensive *and* useless.

Options, in the order I would consider them: skip the overlay with a status-line note
above some cell count; or gate the checkbox on the current cell count so it cannot be
ticked where it cannot work; or draw labels only for the visible axes range. The first is
smallest and honest. Not fixed here because which of the three is right is a UX decision.

Found while cutting `test_plot_settings_live_render` from 283 s to 11 s — that one
checkbox was ~58 s of it, and the test never actually checked the labels appeared.

**[x] L103. `ScopApplicator` labels an un-gap-filled input as gap-filled**
> **Fixed 2026-08-15** (`d60ff96d`). `ScopApplicator.__init__` normalises the input to the neutral
> `ColumnConfig.fct_unsc` ('FCT_UNSC') instead of `.fct_unsc_gf`. The class gap-fills nothing and
> explicitly accepts an ungapfilled term, so a `_gf` label on its own results column asserted a fill
> that provably had not happened. L39's accept-any-input-name behaviour is untouched, and L95's
> single-source property holds - the name still lives only in `ColumnConfig`, a different field of it.
>
> Breaking, but narrowly: only code reading `ScopApplicator.get_results()["FCT_UNSC_gfXG"]` is
> affected, and nothing in the repo did. `ScopPhysics`'s own column of that name is unchanged.
> Mutation-checked: pointing the rename back at `fct_unsc_gf` fails both new tests.
`flux/lowres/selfheating.py` — `ScopApplicator.__init__` renames whatever correction term it is
handed to `ColumnConfig.fct_unsc_gf`, which is L39's deliberate boundary normalisation (accept any
input name, work internally under one). The side effect is that its own results frame labels an input
that was never gap-filled `FCT_UNSC_gfXG`, i.e. claims a fill that did not happen.

The mislabel predates the L95 rename — it said `FCT_UNSC_gfRF` before, so L95 changed which wrong
name it uses, not the wrongness. Found while doing L95, and left alone because L39's normalisation is
intentional and the fix is a design call: the applicator's internal name should probably be the
neutral `FCT_UNSC`, but that touches the column its results frame exposes.

**[x] L104. Three self-heating plots call `plt.show()` with no `showplot` toggle**
> **Fixed 2026-08-15** (library half committed by mistake in `cc0857eb`, remainder in `d60ff96d`).
> All three plots take `showplot: bool = True` and **return their figure**, matching
> `DailyCorrelation.plot`. Returning the figure is what answers the accumulation half of the finding:
> with `showplot=False` and no handle, a loop caller could not close a 24x20-inch figure. The
> production example passes `showplot=False` at all five call sites - five calls against three library
> sites, which is why the finding saw three warnings - and its `FigureCanvasAgg` warnings are gone.
>
> Mutation-checked twice: restoring the unconditional `plt.show()` fails all three tests, and
> returning `None` instead of the figure fails all three, so the flag is pinned in both directions
> rather than only when off.
`flux/lowres/selfheating.py` — `ScopPhysics.plot_diel_cycles()`, `ScopOptimizer.plot()` and
`ScopApplicator.plot_dashboard()` each call `plt.show()` unconditionally. Consequences:

- an example using them **cannot** satisfy CLAUDE.md's "disable `showplot=True`" standard, because
  there is no parameter to disable; `examples/flux/lowres/flux_selfheating_production.py` emits three
  `FigureCanvasAgg is non-interactive` warnings for this reason;
- figures accumulate rather than being closed (the dashboard is 24x20 in), so a script calling these
  in a loop grows its figure count until matplotlib warns;
- it is the same shape as L98 from the other direction — library code deciding to open a window,
  where the caller should decide.

Every other diive plot class takes `showplot` or an `ax`. These three predate that convention. Found
while repairing the examples for L101.

**[x] L105. No `MainWindow` can ever be garbage-collected**

> **Fixed 2026-08-16 — with a correction to this entry's diagnosis.**
> `FramelessResizeHelper._window` is now a `weakref.ref`, and the single deref in `eventFilter`
> resolves it and falls through when the referent is gone (a raise there would be swallowed by Qt).
> That form matches `header_bar.py`, which already reaches the window through `self.window()` rather
> than storing it.
>
> **But this entry blamed the wrong code.** A direct-referrer census of a leaked `MainWindow` showed
> two referrers before the change and one after; the survivor is a closure cell from
> `MainWindow._build_menus`. Stub `_build_menus` out and 4 of 4 windows are collected **with the
> pre-fix strong reference too** — PySide6 traverses the Qt parent/child tree in `tp_traverse`, so the
> collector *can* break the window -> grip -> helper -> window cycle. What it cannot break is
> `act.triggered.connect(lambda _checked, lab=label: self._open_menu_tab(lab))` (`gui/app.py:423`,
> `:436`, ~30 sibling sites): the lambda captures `self`, the `QAction` holds it on the C++ side where
> the collector cannot see it, and the QAction is parented to the window. The headline stays true and
> the measured leak is **unchanged** — 4 built, 4 live, before and after. Recorded as **L147**.
>
> What the weakref does buy was measured with the cyclic collector disabled, so only refcounting can
> free the window: before, 4 of 4 survived the last `del` (both a minimal frameless shell and a real
> `MainWindow` with menus stubbed); after, 0 of 4. The window now dies deterministically at its last
> reference instead of waiting for a gc generation pass, and the helper is off its referrer list.
>
> Covered by `test_frameless_helper_does_not_pin_its_window` (gc off, three shells, every weakref dead
> — a concrete post-condition, not the absence of a traceback) and
> `test_frameless_resize_starts_native_resize` (an edge press hands `Edge.RightEdge` to
> `startSystemResize` and consumes the event; an interior press does neither). `startSystemResize`
> hands off to the window manager, so an offscreen test can assert the handoff but **not** an actual
> geometry change — real edge-drag resizing on a desktop session is unverified. Mutation-checked:
> restoring the strong reference fails the leak test with three live shells. Full GUI suite: 123
> passed.
>
> `tests/test_gui.py`'s `shiboken6.delete` workaround **stays** — windows still leak via L147, and it
> also destroys the C++ widget tree, which no Python-side change achieves. Its docstring at
> `tests/test_gui.py:95` still names `FramelessResizeHelper._window` as the cause and is now wrong.

`gui/widgets/frameless.py:28-31` — `FramelessResizeHelper.__init__` parents itself to the window's
size grip *and* stores `self._window = window`. That closes a window -> grip -> helper -> window
reference cycle through Qt's C++ parentage, which Python's collector cannot break, so the
`MainWindow` wrapper outlives every attempt to drop it: measured, dropping the last reference and
running `gc.collect()` freed nothing and the live `MainWindow` count grew 1, 2, 3, … (+14 top-level
widgets each time).

In the shipped app this is close to harmless — there is normally one window for the process
lifetime. It matters because **each leaked window stays subscribed to the process-wide singletons**
(`theme.manager`, `metadata_store.manager`, `site.manager`, `events.manager`, `db.manager`), so every
emit fans out into all of them. Measured `theme.manager.apply()`: **2.15 s behind one window, 21 s
behind thirty.** A `weakref` for `_window` would be the library-side fix.

Found while cutting `tests/test_gui.py` from 25 min to 5.5 min (`606554db`); the test suite now
destroys each window explicitly with `shiboken6.delete`, which works around it but does not fix it.
Related to the "retain tab instances" gotcha in CLAUDE.md, which is the same lifetime problem from
the other direction.

**[x] L106. Parentless widgets accumulate for the whole session**

> **Fixed 2026-08-16, in two passes.**
>
> **The orphan `QFrame`s are not authored frames at all.** Instrumenting `QFrame`/`QWidget`/`QMenu`
> `__init__` with a creation stack over the real suite shows no separate creation site: they are
> `QComboBoxPrivateContainer`s — one popup container per `QComboBox`, which PySide6 wraps as a plain
> `QFrame`. So "46 parentless `QFrame`s per test" is just the flux-chain tab's 46 combo boxes, and
> what actually survives is **the whole tab**.
>
> **Fixed at tab close.** `MainWindow._on_tab_close` (`gui/app.py:792`) called `removeTab` and
> stopped. `QTabWidget.removeTab` only *detaches* the page; it never deletes it, and nothing on the
> Python side collects it either, so every closed tab's entire widget tree stayed alive and
> parentless for the rest of the session. Measured on the flux-chain tab: **511 widgets leaked per
> open/close**, never released (155 -> 666 -> 1177 -> 1688 over three cycles). `widget.setParent(None)`
> + `deleteLater()` — the pattern `corrections_panel.py:272` and `stepwise_cards.py:110` already use —
> takes that to **+1 per cycle**, and `theme.manager.apply()` after three cycles from **0.987 s to
> 0.090 s** (`app.setStyleSheet` re-polishes every live widget, so the cost tracks the count). It also
> *releases* the `DiiveTab`: once the C++ children die, the connections and `__dict__`s holding `self`
> go with them, so the tab is collected and its `theme`/`site`/`events`/`db` manager subscriptions
> drop — no stale-slot window. `tests/test_gui.py`: 123 passed.
>
> **Pass 2 — why nothing was collected: `self`-capturing lambdas in Qt connections.** A `DiiveTab` was
> uncollectable the moment it was built. PySide6 holds a **bound method** connected to a signal only
> weakly, but a **lambda** is owned by the connection object, which lives on the C++ side — so
> tab -> root widget -> C++ children -> connection -> closure -> tab is a cycle with no Python leg and
> the collector cannot walk it. **Same family as L147**, one level down: there the lambda pins the
> window, here it pins the tab. Pass 1 hid it in the shipped app (deleting the C++ children destroys
> the connections), but not in the suite, which builds tabs and drops them.
>
> **The blocking set was much smaller than the grep suggests.** ~95 `self`-referencing lambdas exist
> under `gui/tabs`, but most capture a *sub-widget* (`_PanelPills`, `_EventCard`, `_ColorSwatch`), not
> the tab. Probing every registered tab — build, push data, drop, `gc.collect()`, read
> `gc.get_referrers` — narrowed it to **30 sites in 13 files**, five of them shared templates covering
> 25 of the 41 tab classes (`_outlier_base.py:152`, `_correction_base.py:129`,
> `_ml_gapfilling_base.py:385`, `_screening_base.py` x8, `plotting.py:209`). Two sites are built only
> *after* interaction and a build-time sweep misses them: the screening step cards
> (`_screening_base.py:499-504`, only after a run — the reason a first pass still ended with 3 live
> `StepwiseScreeningTab`) and the event cards (`events.py:615-619`).
>
> **Fix:** 43 lambda sites removed across 15 files — 29 to a new `weak_slot(method, *args)`
> (`gui/widgets/weak_slot.py`; binds the arguments a signal does not carry, holds the method's object
> weakly, truncates signal args to the target's arity the way Qt does), 12 to a plain bound method,
> and 2 to `widget.setDisabled` (`toggled(on)` is exactly the old `setEnabled(not on)`). Measured:
> **28 of 41 tab classes leaked on drop -> 0** (64 instances probed); flux-chain **+509 widgets per
> build/drop cycle -> +0**; `theme.manager.apply()` after three such cycles **0.65 s -> 0.000 s**;
> suite-end **13 149 live widgets -> 133, live `DiiveTab` -> 0**; `test_live_theme_edit` **~18 s ->
> 7.57 s** in-file (2.5 s alone). Regression test `test_dropped_tab_is_collectable`.
>
> **`CopyPythonButton._provider` was a red herring — correction to the earlier diagnosis.** A bound
> method of the tab stored in a child `QPushButton`'s `__dict__` does **not** pin the tab: PySide6
> traverses a wrapper's instance dict, so that cycle is collectable. It appeared in `get_referrers`
> only as a co-referrer of a tab the lambdas were already holding. A `weakref.WeakMethod` version was
> written, measured to change nothing, and reverted — the same mistake as the L105 weakref, made twice
> in one campaign.
>
> **Freeing the Events tab exposed a latent crash.** Once collectable, `EventsTab` took the process
> down with an access violation inside `gc.collect()` — not an exception, so `slot_exceptions` would
> never have caught it. `_refresh` runs twice (from `build()`, then on the first data push) and
> `QScrollArea.setWidget` destroys the old board on the C++ side while its Python wrappers live on;
> collecting those in the same pass that frees the tab puts a virtual call
> (`_AddCard.mousePressEvent`, a Python override on a `QFrame` subclass) on a half-finalized wrapper.
> Fixed by taking the old board back and deleting it deliberately (`takeWidget` + `setParent(None)` +
> `deleteLater()`). Verified with `DeferredDelete` drained, so the tab is genuinely freed rather than
> the crash merely deferred. **Generalizes:** any Python `QWidget` subclass with a virtual override,
> destroyed implicitly by a container swap, is the same hazard.
>
> **Not a regression risk for open tabs** (CLAUDE.md "retain tab instances"): `MainWindow._tabs`
> retention is what keeps a tab alive, and the change only makes a tab collectable when nothing holds
> it. Verified against a live window with seven open tabs after `gc.collect()`: all survive,
> `VariablePanel.selected` still reaches `_select`, `weak_slot` slots with bound arguments still fire,
> the Plot Update button still re-renders, `theme.manager` / `metadata_store.manager` emits still land,
> and a data push still repopulates a variable list — 12/12, zero swallowed slot exceptions.

`gui/` — after 31 GUI tests, with every `MainWindow` explicitly destroyed, **201 top-level and 4593
total widgets** were still alive with *no* Python referrer: 137 `QFrame`, 36 `QMenu`, 24 `QWidget`,
4 `MplCanvas`. Largest single contributor is the flux-chain tab, **46 parentless `QFrame`s per test**
(`test_flux_chain_tab_level32`); the wind rose adds 7 and the screening tab 2.

Why it is not merely untidy: `app.setStyleSheet` re-polishes **every** widget in the application, so
an accumulating pool of orphans slows every theme change. It is why `test_live_theme_edit` is still
9-10 s inside the file against 2.5 s alone, after the window leak (L105) was dealt with. In a
long-running GUI session the same growth applies to any app-wide restyle.

Two known-correct sites were ruled out — `corrections_panel.py:272` and `stepwise_cards.py:110` both
do `setParent(None)` + `deleteLater()` properly — so the `QFrame`s come from somewhere else in
`diive/gui` that was not located. Finding the source is the first step, not a fix.

---

# Round 4 — the eight files rounds 1-3 left unreviewed

Review date: 2026-08-16 · two parallel reviewers, four files each · diive v0.91.0 · branch `indev`.

Scope: `core/plotting/windrose.py`, `hexbin.py`, `histogram.py`, `waterfall.py`,
`shifted_distribution.py`, `timeseries.py`, `bar.py`, and `gui/icons.py` — the list recorded at the
end of *Reviewed and found sound* as still outstanding after round 3. **That list is now empty.**

Every finding below was reproduced with executed output; nothing here is a hypothesis. L125 is
reproduced but latent (no diive call site currently triggers it) and is marked as such.

## S1 — silently wrong scientific output

**[x] L107. `HexbinPlot`'s `mincnt=0` default paints hexagons over cells holding no data**
> **Fixed 2026-08-16** (`6f7d6bb9`). `mincnt` defaults to `1` and below 1 raises, naming `mincnt=1` as the
> replacement the way the outlier detectors answer their removed parameter names. `1` over `None` because
> matplotlib's `None` already resolves to 1, and `None` would print as `mincnt=None` in copied GUI code.
>
> **Measured before fixing, and it refines this entry.** Two clouds, n=1000, `gridsize=10`, 11 of 116 cells
> occupied; emptiness checked against a reference `hexbin(reduce_C_function=len)` rather than assumed:
> `np.sum` drew 116 hexagons of which **105 covered empty cells (91% fabricated)**; `np.max` raised
> `ValueError: zero-size array to reduction operation maximum` and produced no plot at all; **diive's shipped
> `np.median` default drew 0 fabricated cells** — matplotlib discards the NaN — but emitted **210
> `RuntimeWarning`s**. So the default path was already rendering correctly and only the summing reducers
> fabricated. Breaking twice over: an explicit `mincnt=0` now raises, and a plot that relied on the old
> default with a summing reducer loses its invented cells. The GUI spinbox floors at 1 so it cannot request
> what the library rejects.
`core/plotting/hexbin.py:76` — matplotlib's own default is `mincnt=None` -> 1, and its docs state
that `mincnt=0` "will pass empty input to the reduction function". Since matplotlib unified the
cutoff to `len(acc) >= mincnt` (`axes/_axes.py:5329`), diive's `0` means *include cells with zero
points*, and what happens next depends entirely on the reducer — which the `__init__` docstring
advertises as "`np.mean`, `np.sum`, etc.":

| reducer | result for an empty cell |
|---|---|
| `np.sum` | `0.0` — the hexagon **is drawn and coloured as a measured zero** |
| `np.max` | `ValueError: zero-size array to reduction operation maximum` |
| `np.median` (diive default) | NaN + one `RuntimeWarning` per empty cell, dropped from the render |

**[reproduced]** On a synthetic two-cloud record with a genuinely empty region between the clusters
(`gridsize=10`, `reduce_C_function=np.sum`): **116 hexagons drawn, 99 of them holding no data at
all** — 85% of the plot fabricated. With `mincnt=1`: 17 hexagons, none empty. On the bundled
`load_exampledata_parquet()` (`Tair_f` x `VPD_f` x `NEE_CUT_REF_f`, 175 296 records) the default call
also emits **202 `RuntimeWarning`s** (`Mean of empty slice`, `invalid value encountered in scalar
divide`).

This is the L61/L63/L79 family — "the aggregator kept bins that never occurred and the consumer
rendered them" — with the twist that the fabricated cells carry a *plausible* value rather than a
misplaced one.

Suggested fix: default `mincnt: int = 1`, and say in the docstring that `0` passes empty input to the
reducer. One line, and it also removes the 202 warnings from the default plot.

**[x] L108. `HistogramPlot`'s KDE overlay is scaled by the *first* bin width**
> **Fixed 2026-08-16** (`965f425d`). The bars are raw counts, so the comparable curve value at `x` is
> `N * density(x) * (width of the bin containing x)`; the width is now looked up per bin from
> `np.diff(edges)` instead of being taken from `edges[1] - edges[0]`.
>
> Measured on edges `[0,5,8,9,10,11,12,15,20]`: the curve peaked **4.47x** above the tallest bar, now
> **1.54x**, and that residual is legitimate — it sits inside the 3-wide bin where the density rises steeply
> and is pointwise correct. The bin-averaged curve now tracks the counts (80/78, 72/66, 96/104). **Uniform
> bins are unchanged** — the plotted curve still equals the old single-width expression to 2.8e-13, verified
> independently — and no call site in `diive/`, `tests/` or `examples/` passes an edge list, so nothing
> shipped moves. Mutation-checked: restoring `bin_widths[0]` fails both non-uniform tests and correctly
> leaves the uniform test passing, since the bug is a no-op there.
`core/plotting/histogram.py:142` -> `:146` — the density is rescaled onto the counts axis as
`density * N * bin_width`, with `bin_width = self.edges[1] - self.edges[0]`. That identity holds only
for uniform bins. `n_bins` is documented as "int or list", and an explicit edge list need not be
uniform, so the curve is off by `w_i / w_0` in every bin — drawn on the same axis as the bars, where
it reads as a fit.

**[reproduced]** edges `[0, 5, 8, 9, 10, 11, 12, 15, 20]` over `normal(10, 2)`:

```
counts         : [2, 73, 74, 103, 110, 85, 49, 4]
bin_width used : 5.0        (the first bin; the narrow bins are 1.0)
max KDE y      : 561.54     vs max count: 110.0
```

The KDE tops out 5.1x above the tallest bar. Suggested fix: scale per bin from `np.diff(edges)`
interpolated at `xvals`, or refuse the overlay when the edges are non-uniform.

**[x] L109. `LongtermAnomaliesYear` discards its own `sort_index`, so an unsorted record is plotted and averaged in file order**
> **Fixed 2026-08-16** (`f580914e`). `self.series = self.series.sort_index(ascending=True)` — the result was
> being computed and discarded.
>
> Scoped precisely: this moves bar order, x-axis labels, results-frame row order and the **"last 10 years
> mean ± sd"** annotation, but **only for a record that arrives out of chronological order**. Per-year
> `anomaly` / `reference_mean` / `reference_sd` are **unaffected**, because the reference subset is selected
> by year comparison, which is order-independent. Sorted input is byte-identical. The GUI path resamples
> yearly and already arrives sorted. Mutation-checked with a seeded 31-year shuffle.
`core/plotting/bar.py:64`

```python
self.series.sort_index(ascending=True)   # return value discarded; nothing else sorts
```

pandas returns a new object. Nothing else in the class sorts, so the bars are drawn in input order
**and** `_annotate_reference` (`:75`) computes its "last 10 years mean ± sd" from `tail(10)` of the
same unsorted frame — ten arbitrary years reported as a decade.

**[reproduced]** 31-year record, identical values, shuffled index:

```
sorted   'last 10 years' -> [2011 ... 2020]   mean=9.914 sd=0.807
shuffled 'last 10 years' -> [2006, 1990, 2005, 2020, 2019, 1999, 1998, 2002, 2001, 1995]
shuffled                                      mean=9.417 sd=0.827
```

The printed period label ("2006-1995") is visibly odd, but the mean and sd are believable numbers
over the wrong decade. The GUI path is safe (`gui/tabs/seasonaltrend.py:194` resamples yearly, so it
arrives sorted); this bites library and notebook callers who build the yearly series by hand.

Suggested fix: assign the result.

**[x] L110. `LongtermAnomaliesYear` builds its working frame keyed by the caller's Series name**
> **Fixed 2026-08-16** (`f580914e`). Built on an internal `_VALUECOL = '_values'` key with the caller's name
> restored last, in a new `anomalies_df` property — the **L9 / `ScatterXY` / L68 pattern**, fourth instance,
> no new mechanism. Deliberately **not** a collision guard: the same name legitimately appearing twice is the
> case that pattern exists to support.
>
> Only bites when the caller's Series is named `reference_mean`, `reference_sd`, `anomaly`, `anomaly_above`
> or `anomaly_below`; previously the data column was overwritten *before* the subtraction, so every anomaly
> came out 0.0 or a constant. No public column name changed, so no consumer needed editing. `anomalies_df` is
> now a property returning a copy, the contract `GridAggregator.df_long` already had. Mutation-checked over
> all three colliding names.
>
> **Incidentally closes part of L120:** an unnamed Series no longer raises `KeyError: None`.
`core/plotting/bar.py:94-104` — the L9 / `ScatterXY` family. `_calc_reference` does
`pd.DataFrame(self.series)` (column = the caller's name) and then adds `reference_mean`,
`reference_sd` and `anomaly` beside it. A collision overwrites the data column *before* the anomaly
is computed.

**[reproduced]** correct anomalies are `[-0.5, 0.5, 1.5, 2.5]`:

```
series named 'reference_mean' -> anomaly = [0.0, 0.0, 0.0, 0.0]
series named 'reference_sd'   -> anomaly = [-9.793, -9.793, -9.793, -9.793]
```

Low likelihood (a variable named `reference_mean` is unusual), but the established internal-key fix
(`_values`) is cheap and also closes L120.

## S2 — silently does nothing / silently loses data

**[ ] L111. `WaterfallPlot` turns a fully missing period into a 0.0 contribution bar under the default `agg='sum'`**
`core/plotting/waterfall.py:66` — `series.dropna()` then `.resample(...).agg(agg).dropna()`. Pandas'
`sum` over an empty group returns `0.0` (`min_count=0`), not NaN, so the trailing `dropna()` cannot
remove it. A period with **no measurements at all** is drawn as a zero-height bar with a flat
connector — visually identical to a period whose fluxes genuinely balanced. The running total is
unaffected (adding 0 == skipping), so nothing warns.

**[reproduced]** on bundled CH-DAV `LW_IN`, 2013-2022, with exactly the call the GUI Overview
waterfall panel makes (`resample="D", agg="sum"`):

```
bars drawn: 3652   bars for days with NO data: 429   of those drawn as exactly 0.0: 429
```

`agg='mean'` returns NaN for the same group and drops the period (30 bars vs 27 on a synthetic
3-day outage), so the two aggregations disagree about what the plot even contains. Compounded by
L134: those 429 fabricated bars are painted in the "release" colour.

Suggested fix: `min_count=1` for sum-like aggregations, or mask periods whose `count()` is 0 before
aggregating, so an empty period is NaN for every `agg`.

**[ ] L112. `TimeSeries` colour-by renders *measured* data fully transparent wherever the colour series has a gap**
`core/plotting/timeseries.py:315-317`

```python
seg_c = (color_vals[:-1] + color_vals[1:]) / 2.0
keep = ~(np.isnan(y[:-1]) | np.isnan(y[1:]))   # gaps in y only, never in the colour
```

A gap in the colour driver leaves `seg_c` NaN, which maps to the colormap's "bad" colour — whose
matplotlib default is `(0, 0, 0, 0)`, fully transparent. Measured records vanish, and the result is
indistinguishable from a data gap.

**[reproduced]** complete `FC`, gappy colour driver: **80 of 200 measured records drawn fully
transparent**; the plain path draws all 200. GUI-reachable — the Time-series plot tab passes any
picked variable straight through (`gui/tabs/plotting.py:1206-1215`), and meteo drivers routinely
have gaps.

Suggested fix: extend `keep` to cover `isnan(seg_c)` and draw those segments in a neutral grey, or
`cmap.copy().set_bad(<visible grey>)` — anything that stops "measured but uncoloured" from reading as
"missing".

**[ ] L113. Colour-by silently degrades to a plain line when the colour series does not align**
`core/plotting/timeseries.py:82-84` and `:405` — `color_series.reindex(self.series.index)` yields
all-NaN when the indices differ (TIMESTAMP_END vs MIDDLE, or a differently-resampled driver). The
guard at `:405` then takes the plain-line branch with no warning, so `cmap`, `show_colorbar` and
`color_label` all become no-ops.

**[reproduced]** zero index overlap: `0` `LineCollection`s drawn, `0` colorbar axes, plain fallback
taken. The fallback is described only in a code comment, and the `plot()` docstring's claim that the
scalar `color` "is ignored when a `color_series` was given" is false in exactly this branch.

Suggested fix: `warn()` on the fallback naming zero-overlap as the likely cause, and correct the
docstring.

**[ ] L114. `LongtermAnomaliesYear` draws missing years as adjacent bars while the title asserts the full span**
`core/plotting/bar.py:147-160`, title at `:163` — `plot.bar` is categorical, so a year absent from
the index consumes zero axis width. Fourth member of the L61/L63/L79 family, and GUI-reachable:
`gui/tabs/seasonaltrend.py:196` calls `yearly.dropna()`, which *removes* empty years rather than
keeping them as NaN rows.

**[reproduced]** 1950-2021 with 1980-1991 absent:

```
n years present : 60
title           : TA anomaly per year (1950-2021)
visible x labels: ['1950','1952', ... ,'1976','1978','1992','1994', ... ,'2020']
```

The 12-year outage is a single bar-width jump between two evenly spaced ticks, and
`locator_params(nbins=50)` thins the labels so 1979-1991 carry no tick at all.

Suggested fix: reindex onto `range(first, last + 1)` before plotting so NaN bars leave visible holes,
or use `ax.bar(x=years, ...)` on a numeric axis.

## S3 — crash on legitimate input

**[ ] L115. `HistogramPlot.plot` raises on a constant series, inside the outlier detectors' own diagnostic plot**
`core/plotting/histogram.py:177` — `zscore()` divides by `np.std`, which is 0 for a constant series,
so every z-score is NaN and `int(math.floor(zscores.min()))` fails. `show_zscores` defaults to
**True**, and `core/base/flagbase.py:243,248` draws `HistogramPlot(...)` on both the raw series and
the retained (`ok`) subset.

**[reproduced]** `dv.outliers.AbsoluteLimits(...).run(showplot=True)` on a constant 5.0 series with
three spikes:

```
File ".../histogram.py", line 177, in plot
    for z in range(int(math.floor(zscores.min())), int(math.ceil(zscores.max()))):
ValueError: cannot convert float NaN to integer
```

So any detector run with `showplot=True` on a variable whose retained subset is constant dies inside
its own diagnostic. Suggested fix: skip the z-score overlay when `zscores` holds no finite value.

**[ ] L116. `HistogramPlot.plot` raises on an all-NaN column**
`core/plotting/histogram.py:118` — the closed L69 family. `ax.hist` autodetects the range:
`ValueError: autodetected range of [nan, nan] is not finite`. Reached through the same `flagbase`
path and through the GUI Overview histogram panel (which catches it and prints "Cannot plot").
Suggested fix: early-out with an empty-axes message, as L69 did.

**[ ] L117. `WaterfallPlot.plot` raises `IndexError` on an all-NaN column**
`core/plotting/waterfall.py:164` — `dropna()` empties the series, `__init__` still succeeds, and
`plot` indexes `self.cumulative.index[-1]`: `IndexError: index -1 is out of bounds for axis 0 with
size 0`. Same L69 family. Suggested fix: guard on `self.cumulative.empty`.

**[ ] L118. `ShiftedDistributionPlot.__init__` dies with an opaque error on an empty, all-NaN, single-record or constant period**
`core/plotting/shifted_distribution.py:94` — Silverman's
`bw = 1.06 * data.std() * len(data) ** (-0.2)` has no guard.

**[reproduced]**, all four ordinary user paths:

```
empty comp period        : ZeroDivisionError: 0.0 cannot be raised to a negative power
all-NaN ref period       : ZeroDivisionError: 0.0 cannot be raised to a negative power
constant ref period      : InvalidParameterError: 'bandwidth' ... Got np.float64(0.0)
single-record comp period: InvalidParameterError: 'bandwidth' ... Got np.float64(0.0)
```

A mistyped period, a variable that starts mid-record, or a genuinely constant period (precipitation
in a dry reference decade, snow depth, a flag) all land here. The GUI paints the raw message on the
canvas (`gui/tabs/plotting.py:882`), so the user sees *"Cannot plot 'TA': 0.0 cannot be raised to a
negative power"*.

Suggested fix: validate both periods in `__init__` and raise a named error ("reference period
1990-2000 contains 0 non-missing records / has zero variance"). Related, same file: NaNs are dropped
at `:71-72` with no report, so a 95%-gappy reference period yields a confident-looking KDE from 5% of
the records.

**[ ] L119. `TimeSeries.plot_interactive()` raises on an unnamed Series, which `plot()` handles fine**
`core/plotting/timeseries.py:156` — `legend_label=self.series.name` is `None`, which bokeh rejects
(`ValueError: legend_label value must be a string`). `plot()` and `plot_rangetool()` both succeed on
the same input, so the inconsistency sits inside one class. Suggested fix:
`str(self.series.name or "value")`.

**[ ] L120. `LongtermAnomaliesYear` raises `KeyError: None` on an unnamed Series**
> **Partly closed 2026-08-16 by L110's fix** (`f580914e`): an unnamed Series no longer raises
> `KeyError: None`, because the working frame is keyed internally rather than by `series.name`. The
> other two cases in this entry are untouched — an empty series still raises `IndexError`, and a
> reference period outside the record still annotates `nan±nan`.

`core/plotting/bar.py:101` — same root cause as L110; the internal-key fix closes both. Two
neighbours in the same family: an *empty* series gives `IndexError: index 0 is out of bounds` at
`:83` (`last10.index[0]`), and a reference period outside the record annotates
`"reference period mean: nan±nansd"` rather than raising.

## S4 — contract mismatch

**[ ] L121. `ignore_fringe_bins` is accepted, documented and stored by `HistogramPlot`, and nothing applies it**
`core/plotting/histogram.py:36`, `:45`, `:61` — the L62/L91 defect exactly. The name is not
speculative: `analysis/histogram.py:107` carries a working `_ignore_fringe_bins`, so the plotting
class looks like it inherited the parameter and lost the behaviour. **[reproduced]** counts with
`False` and with `[1, 1]` are identical; the identifier occurs 4 times in the module, never in a
computation.

**[ ] L122. `minticks` / `maxticks` are accepted, documented and forwarded by `hexbin.py`, and nothing applies them**
`core/plotting/hexbin.py:268-269`, `:299-300`, `:360-361` — sibling of the already-fixed L91 in the
same file. `HeatmapBase.plot` only *stores* them; the sole consumer of `self.minticks` in the whole
plotting package is `heatmap_datetime.py:308,314` (`nice_date_ticks`), which hexbin never reaches —
its axes are not date axes. **[reproduced]** `maxticks=3` and `maxticks=30` give identical ticks.

**[ ] L123. `color_bad` is accepted, documented and forwarded by `hexbin.py`, and nothing applies it**
`core/plotting/hexbin.py:270`, `:301`, `:362` — third sibling in the same file. `color_bad` takes
effect only through `HeatmapBase.set_cmap`, called from `plot_pcolormesh`; hexbin renders via
`ax.hexbin` and calls neither. **[reproduced]** stored value `grey`, colormap bad colour actually
used `[0, 0, 0, 0]`.

**[ ] L124. Hexbin's auto `cb_extend` is derived from the raw `z` range while the colorbar maps the aggregate**
`core/plotting/hexbin.py:330-344` — `z_min`/`z_max` come from `self.z` (per-record), but the
mappable's data is the per-hexagon reduction, whose range is always narrower. **[reproduced]** raw
range `-3.7719 … 14.7499`, aggregate range `-3.2043 … 14.6104`; setting `vmin`/`vmax` to exactly the
aggregate range clips nothing and still draws both extension arrows, asserting data outside the
colour scale. Suggested fix: compare against `self.p.get_array()` after the hexbin is drawn, or
document that `cb_extend` refers to the raw range.

**[ ] L125. Hexbin pairs `x` / `y` / `z` positionally, never by index**  *(latent)*
`core/plotting/hexbin.py:108-109`, `:377-379` — the only cross-Series validation is equal length, and
each is taken through `.to_numpy()`. Three Series carrying the same labels in a different order are
mispaired silently. **[reproduced]** z passed with index `3,2,1,0` yields aggregate values
`[10, 40, 20, 30]` where index alignment would give `[40, 30, 20, 10]`. No diive call site currently
passes misaligned Series, so this is latent. Suggested fix:
`pd.concat([x, y, z], axis=1, keys=['_x','_y','_z'])` — the internal-key idiom already used by
`ScatterXY` and `GridAggregator` aligns and makes the frame collision-proof at once.

**[ ] L126. Histogram and hexbin derive bin edges per subset, with no way to pin them — and `flagbase` puts two such panels side by side**
`core/plotting/histogram.py:118-123` (no `range=` or explicit-edge argument on either `__init__` or
`plot`) — the L2 (`WindDirOffset`) family: bin *i* is not the same interval across two histograms of
related subsets. `core/base/flagbase.py:243,248` draws `HistogramPlot(series)` and
`HistogramPlot(series[ok])` in one figure with `n_bins=None` for both, inviting exactly the
before/after shape comparison the differing grids invalidate.

**[reproduced]**

```
full-series edges : [-9.0, -6.7, -4.4, -2.1, 0.2, 2.5, 4.8, 7.1, 9.4, 11.7, 14.0]   width 2.3
'ok'-subset edges : [-2.686, -2.111, ... , 3.056]                                   width 0.574
-> bar 5 left covers (0.2, 2.5); bar 5 right covers (-0.39, 0.19)
```

Hexbin has the identical gap (no `extent=`): daytime and nighttime subsets get different hex grids,
first hexagon centre `[0.0534, 6.7152]` vs `[0.0638, 0.0461]`. Suggested fix: expose the pinned-grid
control (`range=` / explicit edges; `extent=` for hexbin) so a caller comparing subsets can share one
grid.

**[ ] L127. `WindRosePlot.plot(ax=...)` writes the *figure* suptitle and adjusts the caller's figure layout**
`core/plotting/windrose.py:448-451` — when a title is set the class calls `self.fig.suptitle(...)`
and `self.fig.subplots_adjust(top=0.92)` even for a caller-supplied axes. **[reproduced]** plotting
into one panel of a multi-panel figure sets the figure suptitle to `'panel A'` and leaves
`axes[0].get_title()` empty. Impact today is limited (the GUI gives the rose its own figure), but it
is a live trap for anyone composing panels. Suggested fix: `ax.set_title(...)` when `ax` was passed
in; keep the suptitle path for the figure the class created itself.

**[ ] L128. The wind rose ignores every `FormatStyle` field except the title, and the GUI feeds it the full shared Format section**
`core/plotting/windrose.py:347-354`, hardcoded chrome at `:438-446` — the docstring is honest ("only
the `title` / title-font fields apply"), so the *library* contract holds. The *exposure* does not:
`gui/tabs/plotting.py:1122` passes `FormatStyle(**opts["_format"])` from the one shared Format
section, so its grid, chrome-colour and axis-label-font controls are inert for this plot type with no
indication. **[reproduced]** `show_grid=False` leaves 8 gridlines drawn and visible;
`chrome_color='red'` leaves tick labels black. Suggested fix: honour at least `show_grid` and
`chrome_color` on the polar axes (one line each), or hide the inapplicable controls for this type.

**[ ] L129. `ShiftedDistributionPlot.plot()` overrides a caller-set `FormatStyle.ylabel` and forces the grid off while documenting it as controllable**
`core/plotting/shifted_distribution.py:186-189` — `chrome = style.merged(ylabel="Density")`.
`merged()` applies every non-`None` override unconditionally, so it *replaces* a caller-set ylabel
instead of supplying a default; the correct call is `apply(default_ylabel="Density")`, which is what
the rest of the family does. Grid is then hard-forced off three lines later, while the `format_style`
docstring at `:126-128` advertises the style as covering "*(title/x-label/font sizes/colours/ticks/
grid)*". **[reproduced]** caller asks for `'Probability density (1/K)'` and `show_grid=True`; the axes
shows `'Density'` with no grid. The caller's style object is *not* mutated (`merged()` copies), and
the GUI is unaffected (`plot_settings.py:1184` omits both controls for this tab), so this is a
library-caller bug only.

Latent, same lines: `merged()` returns `self` when it receives no non-`None` overrides, and the next
three lines mutate the result in place. It is safe today only because `ylabel="Density"` is always
passed; dropping that argument would start mutating the caller's `FormatStyle`.

**[ ] L130. A zone breakpoint outside the evaluation grid leaves one zone unpainted and mis-places its label**
`core/plotting/shifted_distribution.py:154`, `:166-172`, `:208` — `zone_edges = [x[0]] + breakpoints
+ [x[-1]]` assumes the ±3σ breakpoints lie inside the grid, but the grid spans `[all_min - 1σ,
all_max + 1σ]`. For any skewed or bounded variable a breakpoint falls outside, `zone_edges` stops
being monotonic, the inverted interval yields an empty mask (zone silently not drawn), and the label
lands at the midpoint of the inverted interval.

**[reproduced]**, bounded-above (RH-like):

```
grid span  : 40.59 -> 104.53
zone_edges : [40.59, 82.89, 91.95, 101.02, 110.09, 104.53]   monotonic: False
  Hot            x=105.55   <- beyond the data end, outside its own fill
  Extremely hot  x=107.31   <- 1.8 units from "Hot" on a 76-unit axis: overlapping
zone fills drawn: 5   -> one zone missing
```

Bounded-below (gamma/precipitation) reproduces identically. The `axvline` at the out-of-range
breakpoint also stretches the x-axis into empty space. Because the zone labels are the interpretive
key of this plot, a label sitting over a region it does not describe is a contract error, not a
cosmetic one. Suggested fix: clip the breakpoints into `[x[0], x[-1]]` before building `zone_edges`,
and skip the label when the clipped interval is empty.

## S5 — cosmetic / dead / latent

**[ ] L131. Histogram info box appends itself, printing the text up to four times**
`core/plotting/histogram.py:160,162` — `info_txt += f"..." if self.method == 'n_bins' else info_txt`
appends `info_txt` to itself on the false branch. Two such lines run, so the string doubles twice.
**[reproduced]** `'method: uniformmethod: uniformmethod: uniformmethod: uniform'`. Suggested fix: a
plain `if`.

**[ ] L132. The wind rose drops out-of-range directions without reporting the count**
`core/plotting/windrose.py:198`, mirrored at `:235` — `df[(df['wd'] >= 0) & (df['wd'] <= 360)]`
silently removes sentinels (-9999, 999) and anything off the circle. Dropping them is right; being
silent is not — `report()` prints `n_used` but never how many were rejected or why, so a
wind-direction column in radians or carrying a bad sentinel yields a rose built on a fraction of the
record with no hint. **[reproduced]** 8 records in (1 NaN + 3 out of range) -> `n_used: 4`. On the
real EddyPro example the count is clean (`445 of 468`, matching `co2_flux` non-NaN exactly), so this
is a robustness gap rather than an active loss.

**[x] L133. `WindRosePlot`'s class-docstring example is not runnable, and no test can see it**

> **Fixed 2026-08-16, together with L85's second pass.** The sample was converted to a `>>>` block
> with the correct import, and `test_docstring_refs` was extended to resolve `dv.<attr>` inside reST
> literal blocks and attribute docstrings — so the whole family is now checked, not just this one.
> Turning that check on immediately found **five more** dead names nobody had listed (the
> `heatmap_xyz` and `hexbin` module docstrings, `TreeRingPlot`, and
> `HeatmapXYZ.from_gridaggregator`), plus five stale "Top-level alias" prose claims. See L85 for the
> full accounting; the count of `::`-introduced literal blocks is **41**, not the 13 this entry
> found by grepping for `Example::`.

`core/plotting/windrose.py:85` — calls `dv.load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN()`, which
does not exist (`diive/__init__.py` exports only `load_exampledata_parquet` / `_lae`):
`AttributeError` on the second line. The bundled example file gets it right
(`from diive.configs.exampledata import ...`); only the docstring is wrong. The rest of the sample
runs once the import is corrected.

**This is an L85 coverage hole, not just a typo.** The sample is a reST `Example::` literal block, not
a `>>>` block, so neither `test_docstring_refs.py` nor the new `test_docstring_examples.py` sees it.
There are **13 such blocks across 7 files** (`core/ml/results.py`, `core/plotting/heatmap_xyz.py`,
`treering.py`, `windrose.py`, `flux/fluxprocessingchain/container.py`, `run_chain.py`,
`gapfilling/swin.py`), none of them currently checked by anything.

**[ ] L134. A waterfall contribution of exactly 0.0 is coloured "release"**
`core/plotting/waterfall.py:142` — `uptake_mask = contributions < 0` puts 0.0 in the `False` bucket,
taking the red release colour the docstring reserves for positive values. **[reproduced]**
`[1.0, 0.0, -1.0]` -> `[red, red, blue]`. Compounds L111: a no-data period is painted as a red
release day.

**[ ] L135. `zone_colors` / `zone_labels` lengths are unvalidated**
`core/plotting/shifted_distribution.py:171`, `:209` — the docstring says 5 of each. Three colours
raise `IndexError: list index out of range`; three labels silently under-label (`zip(...,
strict=False)`), leaving 2 zones unlabelled. **[reproduced]** both.

**[ ] L136. Colour-by replaces the caller's axes limits with its own data range**
`core/plotting/timeseries.py:330-335` — `LineCollection` genuinely does not autoscale, but the fix is
`update_datalim` + `autoscale_view`, not `set_xlim`/`set_ylim`. **[reproduced]** ylim
`(47.25, 52.75)` -> `(-1.1, 1.1)`, putting a pre-existing series at y=50 off-screen.

**[ ] L137. A second `plot()` on the same axes stacks artists and colorbars**
`core/plotting/timeseries.py:339`, `bar.py:147`, `shifted_distribution.py:158` — all three classes.
**[reproduced]** TimeSeries figure axes after 1/2/3 calls: 2/3/4 (each colorbar steals more width);
bar `(patches, texts)` `(12,1)` -> `(24,2)`; shifted distribution `(collections, lines, texts)`
`(6,6,5)` -> `(12,12,10)`. The two-phase docstrings promise re-callability "with different styling",
which holds across *different* axes; the same-axes limitation is worth stating.

**[ ] L138. `fig.tight_layout()` on a figure built with `layout='constrained'`**
`core/plotting/bar.py:173`, `timeseries.py:431` — `ax=None` path only. **[reproduced]**
`UserWarning: The figure layout has changed to tight` — the constrained engine is silently disabled.

**[ ] L139. `LongtermAnomaliesYear.get()` before `plot()` raises `AttributeError`**
`core/plotting/bar.py:176-178` — `self.ax` is created only in `plot()`.

**[ ] L140. `icons.py`'s `('calculate', _ln_gear)` rule is unreachable**
`gui/icons.py:571` — the comment claims it covers "derived-variable calculators (VPD, …)", but no
menu label contains "calculate"; it is an `addSection` header (`app.py:481`) and never passed to
`menu_icon`. **[reproduced]** *Potential radiation*, *VPD (TA + RH)* and *diive on &PyPI* fall back to
the generic chart glyph. The other 90 of 93 real labels resolve correctly.

**[ ] L141. Icons are baked at 16x16 with `devicePixelRatio` 1**
`gui/icons.py:26-31` — at Windows 150%/200% display scaling Qt upscales the bitmap, so the glyphs are
blurry on exactly the hardware this runs on. **[reproduced]** `requested 16x16 @dpr=2.0 -> got 16x16
px, devicePixelRatio=1.0`; `availableSizes(): [QSize(16,16)]`.

**[ ] L142. Sub-pixel coordinates are discarded by PySide6's integer `drawLine` overload**
`gui/icons.py`, ~12 glyphs (`_ln_gear`, `_ln_waterfall`, `_ln_windrose`, `_ln_clock`, `_ln_calendar`,
`_ln_lag`, …) — `p.drawLine(4.2, 9, 5.6, 9)` binds `drawLine(int,int,int,int)`, while the `QRectF` /
`QPointF` / `_poly` calls in the same functions keep sub-pixel placement. **[reproduced]** the float
call renders identically to the truncated-int call.

**[ ] L143. `menu_icon(None)` raises `AttributeError`**
`gui/icons.py:730` — no current caller passes `None`; noted only because the docstring promises that
unknown labels fall back.

**[ ] L144. Both bokeh methods call `show(p)` unconditionally**
`core/plotting/timeseries.py:217`, `:295` — the L104 family: no `showplot` toggle, so they always try
to open a browser. Documented behaviour, hence low priority.

**[ ] L145. `bar.py` uses Material 400-level colours where the convention specifies 300**
`core/plotting/bar.py:143-144` — `#EF5350` / `#42A5F5` against CLAUDE.md's `#E57373` / `#64B5F6` for
bars and lines.

**[ ] L146. `ShiftedDistributionPlot` uses the population sd for its zone boundaries**
`core/plotting/shifted_distribution.py:75` — `.values.std()` is ddof=0 where the rest of diive uses
pandas' ddof=1. **[reproduced]** on an 11 000-record reference the +3σ breakpoint differs by 0.0022
(ddof=0 `5.971005` vs ddof=1 `5.971748`) — negligible for a long reference period, not for a short
one.

## Round 4 — reviewed and found sound (no action)

**`windrose.py`**

- **The 0/360 wrap is handled correctly.** Sector 0 is centred on North by the half-sector shift at
  `:183`, and 360 folds to 0 at `:201`. Directions `[0, 10, 350, 359, 360, 22.4, 337.6]` all land in
  `N`; `22.6` / `337.4` correctly land in `NE` / `NW`. **Sector edges are pinned to the full circle**
  (`i * 360/n`), *not* derived from the observed range — the L2 family does not apply here.
- **Binning is exhaustive and equal-width**: one reading per degree over 0-359 gives
  `[45, 45, 45, 45, 45, 45, 45, 45]`, all 360 binned. Verified for `n_sectors` 2, 3, 5, 8, 16.
- **Degrees -> radians happens once**, at `:399` and `:437`; the sector arithmetic stays in degrees.
  No mixed-unit path.
- **The results table agrees with the drawn wedges**, verified numerically on synthetic *and* real
  EddyPro data: per bar, the signed radial extent equals `results[col]` exactly. The GUI's side table
  (`gui/tabs/plotting.py:1050`) is filled from the same `rose.results` object it plots, so table and
  picture cannot drift.
- **Negative aggregates render correctly** — bars anchored at the zero ring
  (`bottoms = min(v, 0)`, `heights = |v|`), `rorigin` set below `rmin`; verified against
  `ax.patches` geometry.
- `fig.colorbar` at `:463`, not `plt.colorbar` — no warning against an axes on a bare `Figure()`.
- No caller-frame mutation; two-phase respected. An all-NaN variable raises a deliberate
  `ValueError("No finite aggregated values to plot (all sectors empty).")`.
- **Calm/zero wind is not special-cased** — `wd == 0` is a valid North bearing and is binned as such.
  Defensible (there is no wind-speed input to define "calm"), but worth knowing.

**`hexbin.py`**

- **NaN in `z` really is ignored, as documented** — matplotlib calls `cbook.delete_masked_points`
  before binning, so a NaN removes one record, not the hexagon. 10% NaN blanked 0 of 77 hexagons.
- `show_values` places labels correctly (`get_offsets()` is in data coordinates and `get_array()` is
  filtered by the same `good_idxs`, so the zip cannot skew).
- Colorbar goes through `HeatmapBase.format` -> `fig.colorbar`; no warning against a bare `Figure()`.
- `HeatmapBase.format`'s `self.series.name` auto-title — which `HexbinPlot` has no `self.series` for —
  is unreachable here, since hexbin calls `format()` without `shown_freq`.
- No caller-frame mutation, including with `normalize_axes=True`. Percentile output spans `(0, 100]`,
  not `[0, 100]` — harmless, but not what "0-100 scale" literally says.
- L91 (`show_less_xticklabels`) is applied at `:412-415`. Confirmed fixed.

**`histogram.py`**

- Counts exclude NaN and `counts.sum()` equals the non-NaN record count (40 of 50 on a 20%-gappy
  series) — no silent double-counting.
- A gappy series does **not** crash (matplotlib uses `np.nanmin`/`np.nanmax` for the auto range);
  only the all-NaN and constant cases do (L115/L116). A 0/2 flag column plots fine.
- No caller-frame mutation; no `plt.colorbar`/`plt.tight_layout`; chrome routed through
  `style.apply(...)` at `:203` with no removed flat kwargs surviving.

**`waterfall.py`**

- **The running total is exact**: `sum(contributions) == cumulative.iloc[-1] == sum(raw series)` to
  floating point on a 30-day half-hourly record.
- **Negative increments are placed correctly and the connectors line up.** For `[+2, -3, +1.5]` the
  bars span `[0,2] -> [2,-1] -> [-1,0.5]` and the connectors sit at exactly `cumulative[:-1]`.
- `uptake_is_negative=False` flips colours only; the cumulative is bit-identical. Single-record input
  works. No `plt.*`, no caller-frame mutation, `series_units` folded onto a *copied* `FormatStyle`.

**`timeseries.py`, `bar.py`, `shifted_distribution.py`**

- **No pyplot leakage in any of the three.** `timeseries.py:339` uses `ax.figure.colorbar(...)`;
  verified against an axes on a bare `Figure()` never created through pyplot — the colorbar lands on
  the caller's figure with **zero warnings**. No `plt.colorbar`, `plt.tight_layout`, `plt.show`,
  `plt.gca` or `plt.subplots` anywhere; `fig.show()` fires only when the class created the figure.
- **No caller-frame mutation** — verified with `.equals()` before/after a full `plot()`, for both the
  series and the colour series.
- **`FormatStyle` compliance**: no leftover flat chrome kwargs on any `plot()` signature.
  `TimeSeries.plot` round-trips a full style correctly (title, units-appended ylabel, `show_grid` and
  `show_legend` honoured); `LongtermAnomaliesYear.plot` routes everything through `style.apply(...)`
  with correct `default_*` and `zeroline_data`. `ShiftedDistributionPlot` is the only offender (L129),
  and even there the caller's style object is not mutated.
- **`ShiftedDistributionPlot`'s distribution alignment is sound** — the L2 concern does not apply.
  Both KDEs are evaluated on the **same** 1000-point grid spanning both periods, and both integrate
  to 1.000 by trapezoid. Each curve gets its own Silverman bandwidth (1.2038 / 1.2070), fitted to its
  own data — the standard choice. Overlapping ref/comp periods are permitted without warning, which
  is defensible since a "vs. full record" reference is a legitimate framing.
- **`TimeSeries` degenerate inputs all survive**: all-NaN (plain and coloured), single-record (the
  `>= 2 finite` guard catches it), duplicate timestamps, and `drop_gaps=True` combined with a colour
  series (indices stay aligned, 490/490). Default gap handling is correct and correctly documented —
  NaNs are kept so matplotlib breaks the line. `drop_gaps=True` does bridge gaps (max plotted step
  `1 day 16:30` on 30-min data), which is what the parameter means and what the docstring says.
- **`timeseries.py`'s internal frame keys are literals** (`df['date']` / `df['value']` built from the
  index and `.to_numpy()`), so a Series named `date`, `value`, `DATE` or `TIME` cannot collide — the
  L68 hazard does not reach here. No resampling happens in the module, so the START/MIDDLE
  half-period question (L66/L68) does not arise.
- **`bar.py`'s bar conventions are not applicable, correctly**: the class draws no value labels and
  sets no figure height, so the contrast formula, `va='center_baseline'` and the dynamic-height rule
  have nothing to apply to. Negative and zero values split into two non-overlapping columns drawn at
  identical positions and width; NaN falls into neither and leaves the slot empty.

**`gui/icons.py`**

- All 44 `_ln_*` factories and all 8 public builders run, call `p.end()`, and produce a non-blank
  16x16 icon; a fresh `QPainter.begin()` succeeds afterwards, so **no painter is left active**.
- **No domain knowledge or algorithm in the module** — it is a keyword table over drawing primitives,
  correctly GUI-only. The one piece of domain mapping that exists (variable name -> pill kind) is
  properly in the library.
- **Ink comes from `theme.manager.tokens["INK"]` and follows a live edit** (`#1E2226` -> `#FF0000`
  verified end-to-end on a rendered pixmap). The only two hex literals are the `_line_ink` fallback
  for isolated test contexts.
- `menu_icon` handles empty string, whitespace, digits, unicode and `&` mnemonics without raising.
  The `_LINE_RULES` ordering comment is accurate: every documented precedence trap ("reset" before
  "set to", "screening" before "database", "removal" before "manual", "partition"/"time lag" before
  "time", "gap-filling" before "gap", "open project" before "open") fires in the right order.

## Round 4 — coverage limits

- These classes were not exercised *through* the GUI; the four relevant call sites
  (`gui/tabs/plotting.py`, `gui/tabs/overview.py`, `core/base/flagbase.py`) were read to judge realism
  and are quoted where they matter.
- The bundled examples were not run end to end (the never-run-the-example-suite rule). The wind-rose
  docstring snippet was re-run on its own, which is how L133 surfaced.
- `HeatmapBase` is out of scope; it was read only far enough to trace which hexbin parameters die
  there (L122/L123). Its own `minticks` / `color_bad` handling for `HeatmapDateTime` / `HeatmapXYZ` is
  correct and untouched by these findings.
- `codegen.py`'s `histogram_to_code` / `waterfall_to_code` round-trips were not audited.
- Bokeh visual output was exercised only to model construction, with `show` / `output_file`
  monkeypatched. The rendered HTML, `window_axis='x'` auto-scaling and the `RangeTool` linkage were
  not verified, nor was `init_range` edge behaviour for 1- and 2-record series.
- The 44 rendered glyphs were verified non-blank and correctly inked, but not inspected to judge
  whether each reads as its intended object.

---

**[x] L147. Menu-action lambdas capture `self`, which is what actually pins every `MainWindow`**

> **Fixed 2026-08-16.** Every `self`-capturing lambda is out of `gui/app.py` — all seven, not just the
> menu ones. The five menu sites collapsed into one `MainWindow._menu_tab_action(label)`:
> `act.setData(label)` plus `connect(self._on_menu_tab_action)`, which reads the label back off
> `sender().data()`. Sweeping the whole file found two siblings with the identical shape (emitter is a
> long-lived child of the window): the tab close button (`:822`), whose slot now matches `sender()`
> against `bar.tabButton(i, RightSide)` instead of closing over the page widget, and the tab
> pin/unpin entry (`:845`), whose tab now travels on `act.setData(tab)` — that menu is parented to the
> window and never deleted, so its action outlives the click. `_make_close_button` lost its now-unused
> `widget` parameter. `&`-escaping, icons, mnemonics and label->tab routing are unchanged.
>
> This form rather than a `functools.partial` or a weakref dance: PySide6 holds a connected bound
> method's receiver only **weakly**, which is why the neighbouring `_act(text, self._open_file)` calls
> never leaked. The label/tab has to survive on the Qt side, and `QAction.data()` is where Qt already
> keeps per-action payload.
>
> Measured (offscreen, weakrefs after `gc.collect()`): 4 minimal windows built and dropped — **4 live
> before, 0 after**. Realistic path (data auto-loaded, `show()`n, a menu tab opened and closed through
> its close button), 3 windows — **3 live before, 0 after**. Before the fix the leaked window's only
> Python referrers were closure cells (1 minimal, 65 realistic); after, none, so there is no second
> pin behind this one — with L105's weakref in place the window is now fully released.
>
> Covered by `test_mainwindow_is_garbage_collectable`, `test_menu_actions_carry_their_label` (all 65
> registered labels: an action exists for each, text is `&`-escaped, triggering routes each to
> `_open_menu_tab` with its own label), `test_menu_action_opens_its_tab` (end-to-end) and
> `test_tab_context_menu_pins_tab`; the close-button rewiring is covered by the existing
> `test_tabs_movable_renamable_and_close_buttons`. Mutation-checked: reinstating the lambda fails the
> gc test with three live windows; hard-coding a label in `_on_menu_tab_action` fails both menu tests.
>
> **The rest of `gui/` still has ~55 `self`-capturing lambdas of this family.** Most are now rescued
> by L106's page deletion and `weak_slot` sweep; the ones that are **not** are
> `tabs/database_explorer.py:105-117` (emitters are parentless `WorkerRunner` QObjects held as tab
> attributes, so page deletion never reaches them) and the never-deleted context menus in
> `widgets/variable_panel.py:198,215`, `widgets/notes_wall.py:179`, `tabs/events.py:305-313`,
> `widgets/header_bar.py:209` — every right-click permanently adds a menu plus actions and pins the
> panel. Separately, `widgets/hover.py:59,60` is the same class of bug in another framework:
> matplotlib's `CallbackRegistry` holds plain lambdas **strongly** while weakref'ing bound methods.

`gui/app.py:423`, `:436`, and ~30 sibling call sites in `_build_menus`

```python
act = QAction(menu_icon(label), label.replace("&", "&&"), self)
act.triggered.connect(lambda _checked, lab=label: self._open_menu_tab(lab))
```

The lambda captures `self`; the `QAction` holds the lambda on the C++ side, where Python's cyclic
collector cannot see it; and the `QAction` is parented to the window. That closes a pin the collector
cannot break, and it is the **sole** reason a `MainWindow` survives — see the L105 correction, where
stubbing `_build_menus` frees 4 of 4 windows even with L105's pre-fix strong reference in place.

Same shape as CLAUDE.md's existing "never connect a lambda to a process-wide singleton signal" rule,
one level in: there the lambda outlives the receiver, here it outlives the *sender's owner*. The
neighbouring `_act(text, self._open_file)` calls are fine, because PySide6 holds only a **weak**
reference to a connected bound method — which is exactly the fix shape here:
`act.setData(label)` + `connect(self._on_menu_tab_action)`, reading the label back off `sender()`.

Consequence is L105's, undiminished: every leaked window stays subscribed to `theme.manager`,
`metadata_store.manager`, `site.manager`, `events.manager` and `db.manager`, so each emit fans out
into all of them (`theme.manager.apply()`: 2.15 s behind one window, 21 s behind thirty). Harmless in
the shipped single-window app; it is the test suite and any embed-diive-in-a-bigger-app scenario that
pay.
