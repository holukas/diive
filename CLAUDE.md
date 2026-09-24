# CLAUDE.md - diive Development Guide

Version history: `CHANGELOG.md`. Per-symbol detail lives in the docstrings; this file keeps only rules, decisions and traps the code does not make obvious.

## Behavioral Guidelines

**Bias toward caution over speed. For trivial tasks, use judgment.**

**Think before coding:** State assumptions explicitly. If multiple interpretations exist, present them. If something is unclear, ask. Push back when a simpler approach exists.

**Simplicity first:** Minimum code to solve the problem. No speculative features, abstractions for single-use code, or error handling for impossible scenarios.

**Surgical changes:** Touch only what you must. Don't "improve" adjacent code. Match existing style. Mention unrelated dead code — don't delete it. Remove only imports/variables that YOUR changes made unused.

**Goal-driven execution:** For multi-step tasks, state a plan with verifiable success criteria before coding.

## Environment

Python 3.12-3.13, `uv`. Minimum pins in `pyproject.toml`.

```bash
uv sync                              # core + dev
uv sync --all-extras --all-groups    # everything (extras gui/gui3d/db + groups dev/db/build)
uv run pytest tests/test_gapfilling.py -v
```

`--all-extras` alone is not everything: `db` (`influxdb-client`) is **both** an extra and a group, on purpose. Working on diive, use `uv sync --group db`. The extra exists for projects that depend on diive, because a dependency group never reaches published metadata. The group is `db = ["diive[db]"]`, so the pin lives only in the extra. `influxdb-client` is imported lazily.

## Repository rules

**`devnotes/`** holds internal working documents (excluded from the sdist, outside `docs/`): `CODE_REVIEW_FINDINGS.md` (entries `L1`…, `S1`–`S5`), `COVERAGE_GAPS.md`, `GUI_PERFORMANCE.md` (re-measure with `devnotes/gui_timing_pass.py`). Items carry a status box (`[ ]` open, `[x]` fixed with its commit, `[-]` won't fix, with reason), a file anchor and evidence. Update the matching entry when you fix something it lists; add new findings there, not in the repo root.

**[CRITICAL] Keep `diive/__init__.py` lazy.** Namespaces (and `diive.io`'s `binary`/`formats`) resolve via PEP 562 `__getattr__`. A plain `from diive import <namespace>` re-imports sklearn/xgboost/shap/statsmodels on `import diive` (~1.4 s). New namespaces go in `_LAZY_SUBMODULES`, the `TYPE_CHECKING` block, **and** `packaging/diive_gui.spec`'s `hiddenimports` (PyInstaller cannot follow `__getattr__`).

**[CRITICAL] `core/` must not import from `gapfilling/`, `flux/` or `preprocessing/`.** Their `__init__`s pull in the ML stack, and it caused a real import cycle (why `prediction_scores` lives in `core/ml/scores.py`). If one leaf function is genuinely needed, import it inside the function (see `aggregated_as_hires` in `core/dfun/frames.py`).

Raw 10/20 Hz tooling moved to [dyco](https://github.com/holukas/dyco); diive starts at averaged flux data.

## Flux partitioning and uncertainty (faithful ports — do not "fix")

Four NEE partitioning ports coexist via suffixes `*_NT_OF`, `*_NT_RP`, `*_DT_RP`, `*_DT_OF` (ONEFlux / REddyProc, nighttime / daytime), wired into the chain as Level 4.2. Parity numbers and implementation detail are in each module's docstring.

- **`*_DT_OF` reproduces a real ONEFlux 1.3.7 bug by default** (alpha-at-starting-guess guard never fires under float32). `reject_alpha_at_start=True` applies the intended check; the L4.2 chain and GUI use the default.
- **The remaining `*_DT_OF` vs ONEFlux residual is ONEFlux's own non-reproducibility**: where `k` ≈ 0 its sign is decided by rounding, and ONEFlux under two NumPy/SciPy versions differs by more than diive differs from either. Don't chase it further.
- The float32/float64 choices in the ONEFlux ports are deliberate parity decisions, documented at each site. Don't normalise dtypes.
- REddyProc ports and ONEFlux nighttime `sunrise_sunset` keep their own potential-radiation routines for parity; everything else uses `dv.variables.potrad` (ONEFlux `get_rpot` port, period mean).

**MDS cascade** (`gapfilling/similarity.py::mds_gapfill_cascade`) has exactly two callers: `FluxMDS` and the `*_DT_OF` NEE uncertainty. **Edge handling differs on purpose**: `FluxMDS` uses `edge='trim'` (C `gf_mds` reference), the daytime uncertainty `edge='clip'` (ONEFlux Python reference, which produced FLUXNET2015). Don't unify them. The cascade is dtype-preserving.

**`RandomUncertaintyPAS20` is not an MDS caller.** It shares only the similarity tolerances, with its own window loop, so cascade changes don't reach it. Its cumulative is NaN-safe quadrature, not an object-dtype `ufloat` cumsum (one NaN must not poison the tail).

**Units for these ports are stated in docstrings, not validated** (VPD in kPa by default via `vpd_in_kpa=True`). Don't add unit-guessing warnings or EddyPro-specific hints; the caller owns units.

## Gap-filling

**[VALIDATION — expect this question]** ML held-out scores (`scores_traintest_`) come from a **random** split of complete rows, not a temporal/block split. This is correct for gap-filling: gaps are interspersed with observed data and filled from same-timestamp drivers, so a random hold-out reproduces the task. A block split measures transferability to an unseen period instead. Do **not** frame this as "long gaps belong to MDS": MDS degrades on long gaps, driver-based ML often handles them better.

**[SWIN — expect these questions]** `SWINGapFillerXGBoost` is its own class because the nighttime zero-offset correction is partly a gap-fill (it sets every nighttime record to zero), so correction and fill must run in one place, in order, sharing `nighttime_threshold=0.001`. Without `context_df` it can only reproduce a **climatology** (all features are functions of the timestamp); no timestamp-derived feature raises that ceiling. A second radiation sensor via `context_df` breaks it. Measurements and defaults are in the class docstring; example `examples/gapfilling/gapfill_swin.py`.

**`FeatureEngineer`** names features `.{col}_TYPE{detail}` (e.g. `.Tair_f_POL2`); all per-column stages skip `.`-prefixed columns. Guard that filter with `str(c)` (unnamed Series give non-string labels). New stage: param in `__init__`, `_stagename_features()`, call from `_create_features()`.

**`DetectFrequency`** runs at the front of nearly every workflow. Verify any change leaves the *detected frequency* unmoved on the bundled datasets, not just the reported percentage.

## Flux Processing Chain

L2 → L3.1 → L3.2 → L3.3 → L4.1 → L4.2 (optional). `run_chain(data, FluxConfig)` for the standard pipeline; composable `run_level*` for custom ones (`FluxConfig` is only for `run_chain`). Per-level signatures intentionally differ. Example: `examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py`.

- Each level is a pure function; never mutate input. Treat `LevelResults` as immutable (rebuilt via `dataclasses.replace`).
- Re-running level N drops N and every later level (`levels/_rerun.py`); L4.1 is per-method and additive.
- L4.1 features and MDS drivers must be in `data.full_df`, not `fpc_df` — use `add_driver()`.
- `FLAG_*_ISFILLED` is informational only, not consumed by `FlagQCF`.
- USTAR filtering applies only to CO2/CH4/N2O; for H/LE use `thresholds=[0], threshold_labels=['CUT_NONE']` and `run_level31(data, set_storage_to_zero=True)`.
- **[DELIBERATE DEVIATION from ONEFlux — do not "fix"]** u\* filtering is `ustar >= threshold` only. ONEFlux also drops the first record above the threshold after a period below it; diive keeps it (L74). Missing u\* is rejected, as in ONEFlux.
- Auto-generated threshold labels `CUT_0`, `CUT_1`, … are positional, not percentiles. `run_chain` only does CUT detection; VUT is composable-only.
- Default `daytime_accept_qcf_below=1` is stricter than FLUXNET's `2`.

## Desktop GUI (`diive.gui`)

PySide6, optional `gui` extra, launch `diive-gui`. **File map and per-tab detail: `diive/gui/README.md`.** User manual: `diive/gui/MANUAL.md`. Windows build: `packaging/README.md`.

**[CRITICAL] Strict GUI ↔ library separation.** `diive/gui/` holds only GUI code (widgets, layout, rendering glue, events, presentation). All algorithms and domain logic live in the library; the GUI calls them. Nothing outside `gui/` imports `diive.gui`. If a GUI piece is reusable, domain knowledge or an algorithm, **tell the user and propose moving it** to the library.

### Patterns to follow

- **Reuse the templates, don't hand-roll:** `VariablePanel` for every left-hand list; `WorkerRunner`/`LatestRunner`/`ProcessLatestRunner` (`widgets/worker.py`) for background work with a **pure** `_compute_payload`; `Debouncer` for per-keystroke controls; and the tab bases `_explorer_base`, `_ml_gapfilling_base`, `_outlier_base`, `_correction_base`, `_partitioning_base`, `_derived_variable_base`, `_screening_base`.
- **Registry-driven tabs.** Menu tabs are `LazyTab` factories imported on first open, so `import diive.gui.app` pulls in no ML stack. A menu tab must live in `diive.gui.tabs`. When creating one lazily, call `tab.widget()` before connecting `featuresCreated`.
- **Hidden tabs are not pushed to** (marked `_data_stale`). Make a tab current before reaching into it.
- **Pinned tabs rely on every writer rebinding, not mutating.** A new `df[col] = ...` on the shared frame silently un-freezes every pinned tab; use `assign`.
- **Outlier tabs** need the library detector to meet the contract: `.run(repeat, progress_callback)`, `.filteredseries`, `.overall_flag`, `.last_lower_bound`/`.last_upper_bound`, `.is_daytime`. Extend the library class; don't reimplement detection.
- **Keep console strings cp1252-safe** (ASCII `->`, not `→`).

### PySide6 gotchas (already handled — don't reintroduce)

- **Retain tab instances** (`MainWindow._tabs`); a GC'd `DiiveTab` makes its signals go inert.
- **A stylesheet touching `QListWidget::item` disables per-item colours** — colour rows via `VariableDelegate`.
- **Use synchronous `canvas.draw()`** after user actions (Overview zoom's `draw_idle()` is the deliberate exception).
- **A widget with its own stylesheet loses app-wide tooltip styling** — append `theme.manager.tooltip_qss()`, and wrap bare properties in a selector first (`"QLabel { color:x; }" + tooltip_qss()`), else the `QToolTip` block is dropped.
- **A widget's stylesheet also styles dialogs parented to it** — scope rules to its `objectName` (`QPushButton#colorswatch`).
- **Don't `p.scale(dpr, dpr)` on a `QPainter`** whose device carries a `devicePixelRatio`; verify icon changes at dpr 1.0/1.5/2.0.
- **[CRITICAL] Qt swallows exceptions raised in slots.** Keep the autouse `slot_exceptions` fixture in `tests/test_gui.py`, and assert a concrete post-condition (a value, a column, `_axes_replaced()`), never just "no traceback" — a stale figure passes that.
- **[CRITICAL] Never connect a `self`-capturing lambda to a signal** — neither a singleton's (`theme`/`metadata_store`/`site`/`events`/`db` managers: dead slots fire after close) nor your own child widget's (uncollectable cycle; leaked every `MainWindow` and 28 of 41 tabs). Use a bound method, `widgets/weak_slot.py`'s `weak_slot(method, *args)`, or `act.setData()` + `sender().data()`. Parent/child cycles are *not* this bug. matplotlib's `mpl_connect` has the same asymmetry.
- **Replacing a container's child** (`QScrollArea.setWidget` etc.) can crash inside `gc.collect()`. Take the old widget back and delete it deliberately (`takeWidget()` + `setParent(None)` + `deleteLater()`).
- **Window must fit the work area:** `show_filling_workarea()`, not `showMaximized()` (a frameless maximize covers the taskbar).

## Outlier Detection & QC

`StepwiseOutlierDetection` is **not** on `dv.outliers`; import from `diive.preprocessing.outlier_detection`.

**[CONVENTION] Day/night thresholds.** New day/night-capable methods MUST match:

1. The switch is always `separate_day_night` (default `True` only for `Hampel`).
2. One global knob (`n_sigma`, `n_sd`, …) is the source of truth. Per-period overrides are `{knob}_daytime`/`{knob}_nighttime`, **default `None`** (never a literal, which shadows the global), falling back to the global. No lists or packed pairs.
3. Separation is not a no-op with equal thresholds, except for pointwise `AbsoluteLimits`: subset-derived methods (`zScore`, `LocalSD`, `LOF`, `Hampel`) compute their statistic per period.
4. A GUI exposing it should expose per-period thresholds (reference: Hampel tab).
5. A removed parameter must say what replaced it: take `**legacy` and call `reject_legacy_params`.

Exceptions: `TrimLow`'s `trim_daytime`/`trim_nighttime` choose which period to trim; `LocalOutlierFactor` has no per-period knobs. `*DaytimeNighttime` names are wrappers (not subclasses — `@ConsoleOutputDecorator` returns a function) or plain aliases.

## Coding Standards

- Validate input only at system boundaries. Let exceptions propagate unless you can recover. Comments explain WHY, not WHAT.
- **Console output:** use the Rich helpers from `diive/core/utils/console.py` (`rule`/`info`/`success` at PROGRESS, `warn`/`error` at ERROR, `detail` at DEBUG, `_console.print` for reports). **No `print()` in production code** (allowed in examples, docstrings, `__main__`, `_cli_main()`), no separate `Console`, no `logging` for general output.
- **[CRITICAL] Always pass `verbose=` when the caller has one**, even inside an `if self.verbose >= N:` guard. A bare `detail(msg)` resolves to the module default (PROGRESS), below `detail`'s own DEBUG level, so it never prints.
- **Module docstring:** `MODULE_NAME: DESCRIPTIVE_TITLE`, `===` underline, one-line scope, then `Part of the diive library: https://github.com/holukas/diive`.
- **Written text** (docs, comments, commit messages, examples): use the `/llm-detox` skill.
- **Examples** (Sphinx Gallery): `# %%` cells, no file I/O, one year of data, no `showplot=True`. New example: register in `examples/run_all_examples.py` + `examples/CATALOG.md`, category README, source docstring "Example" section, `examples/README.md` count, CHANGELOG, verify it runs.

## Plotting

- **Two-phase pattern:** `__init__()` takes data and computation params only; `plot(ax=None, ...)` does all styling and can be called repeatedly.
- **`FormatStyle`** is the only way to set chrome (title/labels/fontsizes/grid/legend…). The old flat chrome kwargs were removed in v0.91.0. Data-render (`color`/`cmap`/`vmin`…) and colorbar (`cb_*`) args stay direct `plot()` kwargs.
- **[CRITICAL] Functions taking `ax` use `ax.figure.colorbar(...)` / `fig.tight_layout()`, never `plt.colorbar` / `plt.tight_layout()`** — pyplot targets a different figure in the GUI. Verify with an axes from a bare `Figure()`.
- **Colours:** Material Design; 500-level lines/bars (`#2196F3`, `#F44336`, `#FFC107`), `#455A64` ink. The 300-bar/500-background split applies only to a bar panel sharing a figure with a shaded panel (`analysis/optimumrange.py`). Don't "correct" 400/500 bar fills to 300 (L145).
- Bar labels `va='center_baseline'`; label contrast `'white' if 0.299*r + 0.587*g + 0.114*b < 0.5 else 'black'`; multi-year panel height `max(1.5, n_years * 0.38)`.
- **Use internal column keys** (`_x`/`_y`/`_z`, `_values`) when building a frame from caller-supplied Series; a shared name silently swaps one role's data for another's.
- Don't replace `_ArrayOffsetsLegend` (fast `loc='best'` on large collections) with a subsample or fixed `loc`. Find scatter collections with `isinstance(c, PathCollection)`, never by class name.
- `RidgeLinePlot` must set gridspec `hspace` at creation; a later `gs.update()` is a no-op for embedded figures.

## Development Workflow

**[CRITICAL] NEVER COMMIT CHANGES.** The user stages and commits exclusively.

**[CRITICAL] NEVER RUN THE EXAMPLE SUITE** (113 examples, several minutes each). Run single examples with `MPLBACKEND=Agg` — 16 end in a bare `plt.show()` that blocks headless:

```bash
MPLBACKEND=Agg uv run python examples/gapfilling/gapfill_randomforest.py
```

**Do NOT:** run `uv` commands without explicit approval, skip pre-commit hooks (`--no-verify`), force-push to main/master, include Claude as co-author.

**Commit message style:** one-line title (< 50 chars) + bullet points.

**Release:** bump the version in `pyproject.toml` **and** the fallback `__version__` in `diive/__init__.py`, then push a `vX.Y.Z` tag. `.github/workflows/publish.yml` checks that all three versions match, builds the package and publishes it to PyPI (trusted publishing). Docs build on Read the Docs from `.readthedocs.yml`.

## Testing

```bash
pytest tests/test_gapfilling.py -v
pytest tests/test_fluxprocessingchain.py -v
pytest tests/test_gui.py -v        # offscreen, needs 'gui' extra
pytest tests/ -v
```

- SHAP importance fluctuates ±5-10%: use flexible ranges (`assertGreater/assertLess`).
- Don't mock databases in integration tests.
- GUI tests wait with `_wait_for_worker(tab)` / `_wait_for_io()`.

---

**Last Updated:** 2026-09-24 | **Version:** v0.91.2
