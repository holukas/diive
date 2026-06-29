# diive desktop GUI

> Using the app? See the **[user manual](MANUAL.md)**. This file is the developer
> map (architecture, gotchas).
>
> **User manual:** `MANUAL.md` is the source of truth. `build_manual.py` renders it
> to the styled `MANUAL.html` that **Help ▸ User manual** opens (and the packaged
> exe bundles). After editing `MANUAL.md`, regenerate the HTML — don't hand-edit it:
> `uv run python -m diive.gui.build_manual`. (`build_gui.ps1` runs this automatically.)

PySide6 desktop application for diive. Optional dependency — install and launch with:

```bash
uv sync --extra gui      # or: pip install 'diive[gui]'
diive-gui                # or: uv run diive-gui
```

To ship the GUI as a **standalone Windows app** (no Python/uv for end users), see
[`packaging/README.md`](../../packaging/README.md) — a PyInstaller one-folder build.

## Architecture

| File | Role |
|---|---|
| `theme.py` | **Central appearance config** — `ThemeManager` (`manager`): editable colour tokens, pill colours/labels, time-series palette, stylesheet; emits `changed`; `as_dict`/`load_dict` for persistence |
| `config.py` | Persist preferences (theme, site, window geometry, last filetype, **variable user-tags**, last project dir) as JSON in the user config dir |
| `widgets/save_project_dialog.py` | `SaveProjectDialog` — project name + location for **File ▸ Save project as…** (writes a `<name>.diive` folder via the library's `save_project`) |
| `metadata_store.py` | **App-wide variable metadata** — `MetadataManager` (`manager`): wraps the library `MetadataStore` (tags + provenance), emits `changed`; edited via `add_user_tag`/`toggle_user_tag`. Also relays variable-list right-click actions app-wide: `editRequested` / `renameRequested` / `deleteRequested` (+ `request_*`) |
| `site.py` | **Project settings store** — `SiteManager` (`manager`): author, description, site coords/UTC offset, and the **notes wall** cards (`notes`); `as_dict`/`load_dict` so it travels with the project + GUI prefs |
| `events.py` | **App-wide event store** — `EventManager` (`manager`): live `list[diive.events.Event]` + a `visible` toggle; `add`/`replace`/`remove`/`set_visible` emit `changed`; `as_dict`/`load_dict` so events travel with the project + GUI prefs |
| `tabs/events.py` | **Events** tab — reflowing board of category-accented event **cards** (`FlowLayout`) on a soft-grey board; per-card header has a **locate** button (`icons.locate_icon` → show on Overview), a **⋯ menu** (`icons.dots_icon` → edit / duplicate / shift), and a **trashcan** delete (`icons.trash_icon`); double-click also edits; **filter** field, **Group** (None/Category/Year) + **Density** (Comfortable/Compact) combos, **Add event…**, **Manage categories…**, master **Show events on plots** checkbox; edits `events.manager` (no domain logic) |
| `widgets/combo.py` | `install_combo_popup_fix(app)` — one app-wide event filter that strips the native frame/shadow from **every** `QComboBox` popup (the black bars on the frameless translucent window) |
| `widgets/menu.py` | `studio_menu(parent)` — a `QMenu` factory styled as a rounded white Studio card (frameless + no-shadow + translucent + `#studiomenu`); used for every context menu so none renders black |
| `widgets/flow_layout.py` | `FlowLayout` — left-to-right wrapping `QLayout` (height-for-width aware) for the event-card grid |
| `widgets/category_dialog.py` | `CategoryDialog` — add / rename / recolour / remove event categories (edits `events.manager.categories`; seeded with `category1/2/3`, the last one can't be deleted) |
| `widgets/add_event_dialog.py` | `AddEventDialog` — name/category, three timing modes (single date/time, from/to, start + duration) with calendar pickers + colour; `make_event()` returns a `diive.events.Event` |
| `tabs/settings.py` | Appearance settings tab — live colour editing with a pill/highlight preview |
| `app.py` | `QApplication` bootstrap, `MainWindow` (menu bar + `QTabWidget`); window starts maximized |
| `splash.py` | Startup splash + **Help ▸ About** dialog (`QPainter`-drawn waves + wordmark/version/tagline/credits); `AUTHOR` + `SUPPORTERS` |
| `build_manual.py` | Renders `MANUAL.md` → `MANUAL.html` (the styled manual **Help ▸ User manual** opens). Dependency-free; run `python -m diive.gui.build_manual` after editing the Markdown. `MANUAL.html` is generated — don't hand-edit |
| `registry.py` | `TAB_CLASSES` (always-on), `MENU_TABS` (menu-opened factories), `SINGLE_INSTANCE_TABS` |
| `tabs/base.py` | `DiiveTab` ABC: `title` + `build()` + `on_data_loaded(df, created)` — the extension point |
| `tabs/overview.py` | Overview tab (first/default): 3×6 panel figure (tall time series top, full-height date/time heatmap right, bottom strip of cumulative/diel/daily/histogram/waterfall; varname in the figure suptitle; datetime panels share an x-axis) + a compact borderless **metrics ribbon** (`_StatItem`, `dv.sstats` + `SSTATS_DESCRIPTIONS` tooltips); panels via `_PANELS`. Exposes `_StatCard` for the Gaps/Drivers/Seasonal tabs |
| `tabs/variable_selector.py` | **Select variables** tab — dual-list picker (available ↔ selected); `subsetSelected` → `MainWindow._apply_var_subset` (app-wide narrowing via `dv.keep_vars`). Opts into `_full_data` (`wants_full_data`) so it can always pick from every column |
| `tabs/rename_variables.py` | **Rename variables** tab — add a prefix/suffix to all variables with a live old→new preview; double-click a row to rename one; `variablesRenamed` → `MainWindow._rename_variables` |
| `tabs/combine_variables.py` | **Combine variables** tab — drag a variable onto **heatmap 1** and another onto **heatmap 2** (`_HeatmapSlot` drop targets), pick a method (multiply/add/subtract/divide, or fill a's gaps with b) + "keep overlapping only", and **heatmap 3** previews the result; **Add to dataset** emits `{name}` (DERIVED), **Copy Python** emits the `dv.variables.combine_variables_to_code` snippet. All maths is the library's `dv.variables.combine_variables`; the tab only collects operands/method, previews the three date/time heatmaps, and emits the column |
| `tabs/site.py` | **Project settings** tab — author/description + site details form (→ `site.manager`) plus the **notes wall** (`widgets/notes_wall.py`) filling the empty space |
| `tabs/plotting.py` | `PlottingTab(plot_type)` — one closable tab per plot method (opened from the Plot menu); var list + live settings panel + canvas |
| `icons.py` | `menu_icon(label)` — tiny `QPainter`-drawn glyphs for **all** menu entries (folder/disk/calendar/gear/palette/… + plot shapes), keyword-matched |
| `widgets/plot_settings.py` | `PlotSettingsPanel(plot_type)` — live plot-parameter controls (between list and canvas); `changed` re-renders; defines `HEATMAP`/`TIMESERIES` |
| `tabs/features.py` | Feature engineering tab (FeatureEngineer; created features get a "NEW" pill) |
| `tabs/fluxchain.py` | Flux processing chain tab — Input + L2 + L3.1 + L3.2 + L3.3 + L4.1 via the composable callables (`init_flux_data`/`run_level2`/`run_level31`/`make_level32_detector`+`run_level32`/`run_level33_constant_ustar`/`run_level33_ustar_detection`/`run_level41_*`); L3.3 supports constant thresholds **or** moving-point detection (Apply: CUT / VUT); shows the deepest level's QCF-filtered flux as a heatmap, **Copy Python** emits a reproducible script. Per-level **Add to dataset** buttons (`_add_level`, gated on the level having run) emit that level's columns + QCF-filtered flux via `featuresCreated` (DERIVED provenance) |
| `tabs/ustar_detection.py` | **USTAR detection** tab — standalone moving-point u\* threshold detection (`UstarMovingPointDetection`): single seasonal detection (per-season + annual) or multi-year bootstrap (`UstarBootstrapThresholds`) for VUT (per-year) + CUT (constant); table + diagnostic plot, worker thread |
| `tabs/timelag.py` | Time-lag analysis tab — pick a gas, analyse its `*_TLAG_ACTUAL` lag distribution (`dv.flux.TimeLagAnalysis`), embed the 4-panel peak/range/EddyPro figure; **Load example TLAG data** loads the bundled level-0 lag dataset locally |
| `tabs/_partitioning_base.py` | `BasePartitioningTab` — shared machinery for the NEE-partitioning tabs (declarative input-column combos + ✓/✗ markers, site coords from `site.manager`, optional VPD-in-kPa toggle, worker thread, GPP/RECO daily-mean + cumulative preview, Add results). Subclasses set `inputs`/`needs_*`/`reco_col`/`gpp_col` and implement `_build_partitioner` |
| `tabs/partitioning_nighttime_oneflux.py` / `tabs/partitioning_nighttime_reddyproc.py` | Nighttime NEE→GPP+RECO partitioning tabs (`dv.flux.NighttimePartitioningOneFlux` → `*_NT_OF` / `NighttimePartitioningReddyProc` → `*_NT_RP`) — `BasePartitioningTab` subclasses |
| `tabs/partitioning_daytime_reddyproc.py` / `tabs/partitioning_daytime_oneflux.py` | Daytime NEE→GPP+RECO partitioning tabs (`dv.flux.DaytimePartitioningReddyProc` → `*_DT_RP` / `DaytimePartitioningOneFlux` → `*_DT_OF`) — `BasePartitioningTab` subclasses; show the VPD-in-kPa toggle |
| `tabs/_ml_gapfilling_base.py` | `MlGapFillingTab` — **template** for the ML gap-filling tabs (Model/Results sub-tabs, three-list target/feature picker, performance hero, observed-vs-gap-filled heatmaps + SHAP table, Results dashboard, SHAP feature reduction, worker/emit flow). Subclasses set `title`/`method_name`/`method_chip_*` and implement `_model_class`/`_build_model_box`/`_method_kwargs`/`_method_controls`/`_codegen` |
| `tabs/gapfilling.py` | XGBoost gap-filling tab (`dv.gapfilling.XGBoostTS` → `*_gfXG`) — `MlGapFillingTab` subclass; `Gap-filling ▸ XGBoost gap-filling`, own top-level **Gap-filling** menu |
| `tabs/gapfilling_randomforest.py` | Random Forest gap-filling tab (`dv.gapfilling.RandomForestTS` → `*_gfRF`) — `MlGapFillingTab` subclass; `Gap-filling ▸ Random Forest gap-filling` |
| `widgets/gapfill_results.py` | `GapFillResultsPanel` — the ML gap-filler **Results** dashboard: card layout of the console-report tables (performance / configuration / feature-reduction with info-button equation + dashed-red threshold line + coloured verdicts / gap-fill quality) over a row of plots (predicted-vs-observed, SHAP, diel cycle, cumulative). Reads only library model attributes |
| `tabs/gaps.py` | Gap & coverage dashboard — stat cards + clickable gap map (`GapStats` availability heatmap + gap timeline) + long-gap table |
| `tabs/drivers.py` | Driver explorer — rank variables by correlation with a target (`rank_drivers`, optional lag scan); click a driver for its scatter |
| `tabs/seasonaltrend.py` | Seasonal-trend & anomaly explorer — STL/classical/harmonic decomposition + yearly anomalies vs a reference period |
| `tabs/spectrogram.py` | Spectrogram explorer — time-frequency map (`dv.analysis.spectrogram`) on calendar/cycles-per-day axes + an explanation |
| `tabs/surface3d.py` | 3-D surface explorer — date×time-of-day relief rendered with PyVista (`dv.plotting.datetime_surface_grid` for the grid); extruded-heatmap (stepped bars, default) or smooth-surface style, Y-stretch + day-binning, optional cast shadows; optional `gui3d` extra, shows install notice if absent |
| `widgets/pyvista_canvas.py` | `Pyvista3DCanvas` — embedded `pyvistaqt.QtInteractor` (GPU/VTK render window); lazy-imports VTK, `pyvista_available()` gate + `Missing3DDependency`. `frame_default` = orthographic lower-left 45° framing; `apply_shadows` = flat headlight or overhead spotlight + shadow mapping |
| `tabs/_outlier_base.py` | `BaseOutlierTab` — shared machinery for the Outliers tabs (var list, two-panel preview, worker thread, iteration progress, live preview, limit lines, day/night colouring, Add/Copy Python). `supports_daynight` toggles the day/night box |
| `tabs/outliers.py` / `tabs/outliers_localsd.py` | Hampel / Local SD outlier tabs (`dv.outliers.Hampel` / `LocalSD`) — `BaseOutlierTab` subclasses |
| `tabs/outliers_absolutelimits.py` | Absolute limits outlier tab (`dv.outliers.AbsoluteLimits`) — flag values outside a fixed min/max range (min/max drawn as the limit-line band); `BaseOutlierTab` subclass |
| `tabs/outliers_zscore.py` / `tabs/outliers_zscorerolling.py` / `tabs/outliers_zscoreincrements.py` | Z-score outlier tabs (`dv.outliers.zScore` / `zScoreRolling` / `zScoreIncrements`) — `BaseOutlierTab` subclasses; rolling & increments set `supports_daynight = False` |
| `tabs/outliers_lof.py` | Local Outlier Factor tab (`dv.outliers.LocalOutlierFactor`) — density-based detection; `BaseOutlierTab` subclass |
| `tabs/outliers_trim.py` | Trim-low tab (`dv.outliers.TrimLow`) — symmetric positional trim; `supports_daynight = False`, opt-in `trim_daytime`/`trim_nighttime` rows, no detection band |
| `tabs/outliers_manualremoval.py` | Manual removal tab (`dv.outliers.ManualRemoval`) — flag known timestamps/periods; `supports_daynight = False`, `supports_repeat = False`, no detection band |
| `tabs/stepwise.py` | Stepwise screening tab (`dv.outliers.StepwiseOutlierDetection`) — chain multiple outlier methods with QCF aggregation, plus a **corrections** phase (`dv.corrections.apply_corrections` on the QCF-filtered series). Layout: shared variable list + a segmented inspector (Outliers / Corrections / Report, one `QStackedWidget` page shown at a time) + a large always-visible plot grid; a **measurement** dropdown gates applicable corrections; per-section **Run outliers** / **Run corrections** buttons apply edits (nothing auto-runs); runs invalidated only when the variable's data actually changes |
| `tabs/_correction_base.py` | `BaseCorrectionTab` — **template** for the standalone Corrections tabs (XGBoost-style title bar + Copy Python, "Target (click to set target)" var list, "Settings" panel, method hero chip with a stats strip, original/corrected two-panel preview, worker thread, Add). Routes through the library `apply_corrections` / `corrections_to_code`; subclasses set `corr_key`/`method_suffix`/`method_chip_*`/`needs_coords` and implement `_add_method_rows`/`_current_kwargs`/`_validate`/`_method_controls`, and may override `_apply` (return `(corrected, extra)`), `_hero_metrics`, `_render_result`, `_status_text` for richer output. One tab per correction, so all corrections are independently available (measurement is a hint, not a lock) |
| `tabs/corrections_nighttime_offset.py` / `tabs/corrections_relativehumidity_offset.py` | Offset-removal correction tabs (`dv.corrections.remove_nighttime_zero_offset` / `remove_relativehumidity_offset`) — `BaseCorrectionTab` subclasses. **Remove nighttime zero offset** (`NighttimeZeroOffsetTab`) is for variables that read zero at night (SW/PPFD); needs site coords, has a **Clamp negative values to zero** checkbox (`clamp_negatives`, default on), and overrides the hooks to show a **4-panel diagnostic preview** (original → daily offset → series−offset → final corrected) + a **below-zero stats hero** (records < 0 before/after, overall + nighttime, confirming the night no longer dips below zero), driven by the library's `nighttime_zero_offset_diagnostics`. The same `clamp_negatives` option is mirrored on the stepwise panel (`widgets/corrections_panel.py`) |
| `tabs/corrections_setto_threshold.py` / `tabs/corrections_setto_value.py` / `tabs/corrections_set_missing.py` | Generic correction tabs (`dv.corrections.setto_threshold` max/min, `setto_value`, `set_exact_values_to_missing`) — `BaseCorrectionTab` subclasses; own top-level **Corrections** menu |
| `tabs/metadata_explorer.py` | Metadata explorer — per-variable origin badge, editable tag chips (favorite/add/remove, auto-coloured), a 50-word note, provenance timeline; reads `metadata_store.manager` |
| `tabs/log.py` | Log tab wrapping `ConsolePanel` (live coloured library output) |
| `widgets/mpl_canvas.py` | `MplCanvas` — embedded matplotlib figure + bottom-right toolbar (with a Save-DPI spinbox); attaches a `HoverAnnotator` |
| `widgets/hover.py` | `HoverAnnotator` — value-under-cursor tooltip (line snap + heatmap cell) via blitting |
| `widgets/variable_panel.py` | **`VariablePanel`** — the shared variable list (filter + pills) used by every tab; right-click menu (rename/delete/favorite/tags) routed via `metadata_store.manager`; `scroll_to(name)` / `clear_filter()` helpers |
| `widgets/notes_wall.py` | `NotesWall` — sticky-note pinboard (draggable/resizable/recolourable cards, bold header + body); `state()`/`set_state()` serialise to plain dicts (stored in `site.manager.notes`) |
| `widgets/variable_list.py` | `VariableList` — list emitting `selected(name, ctrl_held)` |
| `widgets/variable_delegate.py` | `VariableDelegate` — paints row highlight + NEE/GPP/Reco pills |
| `widgets/open_data_dialog.py` | `OpenDataDialog` — file + filetype picker with a parsed live preview |
| `widgets/daterange_dialog.py` | `DateRangeDialog` — from/to picker (clamped to the data span) for date-range subselection |
| `widgets/header_bar.py` | `StudioHeaderBar` — frameless Studio chrome header: wordmark + inline File/Data/… hover-dropdown menus + centred title |
| `widgets/frameless.py` | `FramelessResizeHelper` — edge/corner resize for the frameless Studio window |
| `widgets/console_panel.py` | `ConsolePanel` — mirrors diive's Rich output in colour (used by the Log tab) |
| `widgets/stepwise_method_params.py` | One param widget per `StepwiseOutlierDetection.flag_*` method; produces a `{"method", "kwargs"}` step for the L3.2 / stepwise chains (the shape `level32_to_code` consumes) |
| `widgets/stepwise_cards.py` | The stepwise chain's editable **method cards** (`StepCard` shows every setting + removed count + reorder ▲▼ / edit / delete; `AddStepCard`; `StepEditorDialog`) — display widgets around the `stepwise_method_params` registry |
| `widgets/corrections_panel.py` | `CorrectionsPanel` — checkable correction rows filtered to the selected measurement (`dv.qaqc.corrections_for_measurement`), with inline descriptions; parses date-range / value text into the `{"key","kwargs"}` corrections `apply_corrections` consumes |
| `widgets/copy_button.py` | Reusable **Copy Python** button — copies library-generated code to the clipboard (GUI never builds the script, only copies it) |
| `widgets/sub_tabs.py` | `SubTabs` — standardized in-tab sub-navigation (segmented pills over a `QStackedWidget`) for output-heavy tabs; `add_page`/`set_page`/`changed`, `add_corner_widget` (action buttons by the pills) + `add_corner_separator` (faded `_CornerSeparator` divider) |
| `widgets/state_utils.py` | `save_controls`/`restore_controls` — serialize a tab's standard Qt controls by stable key for `save_state`/`restore_state` |

**Adding a tab:** always-on tabs (Overview, Log) go in `TAB_CLASSES`. Menu-opened tabs go in `registry.MENU_TABS`
(grouped by menu; values are factories) — they open as **new numbered instances** each time (Heatmap 1, 2, 3 ...), all
closable, unless listed in `SINGLE_INSTANCE_TABS` — reserved for the app-wide singleton editors Appearance, Project
settings, and Metadata explorer, which re-selecting focuses instead of duplicating. The main window is agnostic to
concrete tabs.

**Menu icons:** every menu entry (File/Data/Plot/Outliers/Flux/Analyze/Settings/Help) gets a small `QPainter`-drawn glyph via
`gui/icons.py::menu_icon(label)`, matched to the label by keyword (`&` mnemonics stripped first). `_build_menus` wraps
each action with it. Add a menu entry → add a keyword rule in `icons._LINE_RULES` (the thin-line monochrome glyph table;
unknown labels fall back to a chart glyph).

**Studio look:** the GUI has one design — Studio (a clean, minimal VIBECAD-style look: near-white surfaces, soft
borderless panels, monochrome line icons, uppercase tracked labels, and a frameless rounded window with a custom
`StudioHeaderBar`). `gui/theme.py::manager` holds `STUDIO_TOKENS` (edited live in **Settings ▸ Appearance**) and
`STUDIO_TYPOGRAPHY`; `MainWindow` always builds the frameless shell (`_build_studio_chrome`).

**Plot menu:** each method is its own closable tab, with a small drawn icon. The **Plot** menu lists methods (Heatmap
date/time, Heatmap year/month, Heatmap x/y/z, Time series, Diel cycle, Cumulative year, Cumulative, Ridgeline, Scatter XY, Hexbin, Histogram, Shifted distribution, Wind rose); selecting one
opens a new `PlottingTab(plot_type, title)` instance. Add a method via a factory in `registry.MENU_TABS["Plot"]` + a
branch in `plotting._draw_one` (and matching controls in `plot_settings`). Ctrl+click adds comparison panels: heatmaps
(both kinds, in `_HEATMAP_TYPES`) go side by side (shared x/y); time series and diel cycle stack top-to-bottom (shared
x-axis — time, resp. hour-of-day).
The **ridgeline** is single-variable and whole-figure: `RidgeLinePlot` builds its own stacked-density gridspec, so the
tab passes `canvas.fig` to the class's `fig=` param and sets `canvas.auto_layout=False` (so the constrained-layout
freeze/resize machinery doesn't reflow its overlapping ridges) — see `_render_ridgeline`.
The **histogram** is also single-variable (it's information-dense: bar counts + a z-score twiny axis + peak/info box),
but a normal per-`ax` plot — `HistogramPlot(series, n_bins=...).plot(ax=...)` with toggles for the peak highlight,
z-score axis, counts, info box, title and grid.
The **shifted distribution** is single-variable too: it compares one variable's KDE between a reference and a comparison
period (`ShiftedDistributionPlot(series, ref_period, comp_period).plot(ax=...)`), colouring the comparison curve by the
reference period's ±1σ/±3σ zones. The two periods are partial date strings (a year, or `YYYY-MM-DD`), seeded by
`set_periods` from the data's year range (earlier half vs later half). Like the ridgeline it sets `canvas.auto_layout=False`
and places its single axes with an explicit rect (`add_axes`) so the zone labels above the top spine aren't clipped — see
`_render_shifted_distribution`.
The **heatmap x/y/z** is role-picked (X/Y/Z by click order, in `_XYZ_TYPES` like Hexbin) but `HeatmapXYZ` needs one Z per
(x,y) bin, so the tab **pre-aggregates** the raw data via `dv.analysis.GridAggregator` and feeds the binned grid through
`HeatmapXYZ.from_gridaggregator(...)` rather than passing raw columns — see `_render_heatmap_xyz`.
The **tree ring** is single-variable and whole-figure/polar: like the ridgeline/wind rose it sets `canvas.auto_layout=False`
(polar axes, no constrained-layout reflow). Its **line** style dispatches to `TreeRingPlot.plot_line` instead of `.plot`
(the codegen `_script` gained a `plot_method` arg to emit the matching call) — see `_render_treering`.
The **cumulative** is single-variable but flows through the standard panel machinery (a branch in `_draw_one`, so it's
Ctrl+click multi-panel capable) like Cumulative year; it calls `Cumulative(df=series.to_frame(), ...)`. **Distinct from
Cumulative year** (`CUMULATIVE` whole-record running total vs `CUMULATIVE_YEAR` per-year reset).

**Plot settings:** between the variable list and the canvas sits a `PlotSettingsPanel(plot_type)` — a scrollable
strip of controls, one per `plot()` parameter of the underlying diive plot class (heatmap: colormap, vmin/vmax,
orientation, colorbar, cell-value overlay, ticks, grid, …; time series: line width, opacity, markers, drop-gaps,
labels/units). **Editing a control does NOT re-render** — the tab's **Update plot** button (`PlottingTab.update_btn`,
a left-aligned row just below the tab header) reads `settings.values()` and calls `_render()` on click, so the apply
trigger is identical for every control type (no `editingFinished`-vs-`valueChanged` inconsistency). The button is
**dirty-gated**: disabled until `settings.changed` (or `xyz_changed`) fires, re-disabled at each render. For the
comparison types, variable selection in the list *does* still render live (`_on_selected` → `run_with_loading`); for the
role-dropdown types (`_ROLE_DROPDOWN_TYPES` = Scatter, Wind rose, Hexbin, Heatmap x/y/z) the roles are assigned via
X/Y/Z `_DropComboBox` dropdowns (pick from the complete list or drag a variable onto a field) and apply on Update like
any other setting (`_build_role_combos(labels, none_ok=)` / `set_xyz` / `xyz_values`); list-click is a no-op. Scatter and
Wind rose make the colour role optional; Hexbin and Heatmap x/y/z require all three. **Every** plot tab has a title-bar **Copy Python** button
(`_python_code` dispatches per plot type to the library codegen: `scatter_to_code` in `scatter.py`, the rest in
`core/plotting/codegen.py`); it is a no-op while role picks are incomplete. Multi-panel tabs emit the active panel's variable.
`_draw_one` reads `settings.values()` into the library plot call. The panel is GUI-only (it just collects parameters); the
`HEATMAP`/`TIMESERIES` constants live in `plot_settings.py` and `plotting.py` re-exports them (so no import cycle). Line
*colours* stay theme-driven (`theme.manager.ts_colors`, Appearance tab), not duplicated here. Add a parameter = add a
control in `plot_settings._build_*` + a key in `values()` + pass it through in `_draw_one`.

**GUI-only post-render passes:** a couple of settings have no library `plot()` parameter and are applied *after* the
diive plot renders, analogous to the Overview's uniform-font pass:
- **Axes group** (`_build_axes_group`) — X/Y limits (blank = auto), log X/Y, invert Y. Present on the
  line/scatter types only (`TIMESERIES`, `CUMULATIVE_YEAR`, `SCATTER`, and Y-only for `DIELCYCLE`); `values()` carries
  them under the `_axes` key and `plotting._apply_axes(axes)` applies them to the data axes. It is **plot-type-aware**:
  heatmaps and the ridgeline have no `_axes` (no-op), and the diel cycle's fixed 0–24 h x-axis is left alone (Y-only
  group). For shared multi-panel stacks the limits apply to every panel. The grid is **not** here — it has a single
  source in the Format group's `show_grid` (a duplicate Axes grid could only add, never remove, so it appeared broken).
  `_apply_axes` also re-applies the preserved pan/zoom only when the kept range still **overlaps** the new data
  (`_ranges_overlap`), so e.g. flipping a heatmap's orientation can't scroll the view off the data.
- **Reverse colormap** — a checkbox in the Colors group (heatmap, heatmap year/month, hexbin, scatter) that appends/strips
  the matplotlib `_r` suffix on the chosen cmap in `values()` (`_reverse_cmap`); no library change.

**Figure export DPI:** `MplCanvas`'s bottom bar has a *Save DPI* spinbox (default 150); its `_SaveDpiToolbar`
overrides the toolbar's Save to export `savefig` at that DPI, so saved images aren't capped at screen DPI.

**Data flow:** **File ▸ Open data file…** shows `OpenDataDialog` — pick one or more files, choose the filetype, and
preview the first parsed rows before loading (parquet via `dv.load_parquet`, other formats via `dv.ReadFileType`).
Selecting multiple files merges them (`MultiDataFileReader`, or `combine_first` for parquet). All reading is library
work, the dialog only orchestrates it. `MainWindow` holds the current DataFrame and pushes it to every tab via
`DiiveTab.on_data_loaded(df, created)`; tabs that present data override that hook to refresh. Example data auto-loads on
startup. **File ▸ Save data as parquet…** writes a diive-format parquet (`to_diive_parquet_frame`: single-level columns
+ valid `TIMESTAMP_*` index name) via `dv.save_parquet`.

**Date-range subselection (`Data` menu):** non-destructive. `MainWindow` keeps the whole loaded record in `_full_data`;
`_data` (pushed to every tab) is `_full_data` optionally narrowed to `_range=(start,end)` via `dv.times.keep_daterange`.
**Data ▸ Select date range…** opens `DateRangeDialog` (from/to pickers seeded and clamped to the data span); **Data ▸
Reset to full range** clears the window. `_apply_range()` re-derives `_data`, retitles the window with the active
window, enables/disables the reset action, and re-pushes. Engineered features merge into `_full_data`, so they survive a
reset (out-of-range rows align to NaN). All plots and processing then run on the narrowed `_data`; saving writes it too.

**Flux processing chain (`tabs/fluxchain.py`):** opened from **Flux ▸ Flux processing chain** (single-instance). A
guided tab for the Swiss-FluxNet chain, covering **Input + L2 + L3.1 + L3.2 + L3.3 + L4.1**. It
collects site/flux-column + which L2 quality tests to run, the L3.1 storage-correction options, an optional L3.2
outlier-detection chain, optional L3.3 USTAR filtering, and L4.1 gap-filling, then on a worker thread calls the composable library
callables (`init_flux_data` → `run_level2` → `run_level31` → `make_level32_detector`/`run_level32` →
`run_level33_constant_ustar` / `run_level33_ustar_detection` → `run_level41_*`), shows the deepest level's QCF-filtered
flux as a date/time heatmap (plus the L3.2 QCF distribution / L3.3 scenarios), and — the point of the feature — **Copy
Python** emits the exact runnable script via the library's per-level codegen
(`level2_to_code`/`level31_to_code`/`level32_to_code`/`level33_to_code`/`level41_to_code`). **L3.3 has two modes**
(a Mode selector swaps the inspector page): *constant thresholds* (enter value/label scenarios) or *moving-point
detection* (`run_level33_ustar_detection`, with TA/SW_IN pickers, bootstrap params, and an **Apply** selector choosing
**CUT** (constant → `CUT_16/50/84`) vs **VUT** (per-year → `VUT_16/50/84`) — mutually exclusive strategies). The
script-gen lives in the library (`flux/fluxprocessingchain/codegen.py`: `chain_to_code` for the full
`run_chain`/`FluxConfig` path, the `level*_to_code` functions for the composable path) because it encodes the API call
shape; the GUI only calls it. Needs real EddyPro-FLUXNET input (FC/USTAR/`*_TEST` columns) —
`load_exampledata_parquet_lae_level1_30MIN`, not the default CH-DAV.

**USTAR detection (`tabs/ustar_detection.py`):** opened from **Flux ▸ USTAR detection** (single-instance). Standalone
friction-velocity threshold detection independent of the chain. Pick NEE/TA/USTAR/SW_IN + stratification params (TA/USTAR
classes, forward-mode-n) and run either a **single seasonal detection** (`UstarMovingPointDetection` → per-season +
annual threshold table) or, with **Multi-year bootstrap** ticked, `UstarBootstrapThresholds` → **VUT** (variable,
per-year p16/p50/p84) + **CUT** (constant, pooled) thresholds, as a table + a diagnostic plot. Runs on a worker thread;
all detection is library work — the small result plot is the only presentation in the tab (a candidate to move to a
library plot helper if reused elsewhere).

**Gap & coverage dashboard (`tabs/gaps.py`):** opened from **Analyze ▸ Gaps & coverage** (single-instance). Pick a
variable; the right side shows KPI stat cards, a two-panel **gap map** (daily-availability heatmap + gap-spike timeline)
and a table of the longest gaps. The map is **clickable both ways**: clicking a table row highlights that gap on the
timeline (a blue span + ring overlay); clicking the timeline calls `GapStats.gap_at(timestamp)` to find the nearest gap,
highlights it, and selects its table row (a `_syncing` flag prevents the two selections echoing). *All* gap logic is the
library's `dv.analysis.GapStats` — the tab reads `.summary` / `.long_gaps`, calls the per-`ax` `plot_availability_heatmap` /
`plot_gap_spike_timeline`, and the new `gap_at()` lookup; it implements no gap maths itself (separation rule). It defaults
to the gappiest column (`df.isna().sum().idxmax()`) so it's useful on open, and a "long gap ≥ records" spinbox re-runs
`GapStats`. The library's panel `plot_*` methods were made embed-safe (`ax.figure.colorbar`, not `plt.colorbar`) so they
render into the shared canvas — a one-line fix that benefits any embedding caller.

**Driver explorer (`tabs/drivers.py`):** opened from **Analyze ▸ Driver explorer** (single-instance). Answers "what
relates to this variable, and at what lag?" Pick a target; a ranked table lists every other variable by correlation
strength (the `r` cell tinted green/red by sign and magnitude), with its best lag and overlap count; clicking a driver
renders the target-vs-driver scatter (shifted by that driver's best lag). The ranking + lead/lag scan is the library's
new `dv.analysis.rank_drivers(df, target, method=, max_lag=)` (returns `[DRIVER, CORR, ABS_CORR, BEST_LAG, N]`); the
scatter is `dv.plotting.ScatterXY`. The tab implements no statistics — it only collects target/method/max-lag, fills the
table/cards, and renders the selected scatter. Target selection is **live**; method and max-lag apply on a **Rank
drivers** button (the lag scan can be heavier). The table sorts numerically via a small `_NumItem` (compares the stored
value, not the display string). Defaults to a continuous flux target (`NEE_CUT_REF_f`) so the ranking is informative on
open. A natural next step: a "send top-N drivers to Feature engineering / gap-filling" handoff.

**Seasonal-trend & anomaly explorer (`tabs/seasonaltrend.py`):** opened from **Analyze ▸ Seasonal trend & anomalies**
(single-instance). Pick a variable → its daily-mean series is split into **trend / seasonal / residual** (four stacked
panels), and a second **view** shows each year's **anomaly** vs a reference period (red above / blue below). Everything is
library-backed: `dv.times.resample_to_daily_agg` builds the daily series, `dv.analysis.SeasonalTrendDecomposition`
(STL / classical / harmonic) decomposes it, `dv.plotting.LongtermAnomaliesYear` draws the anomaly bars. The tab only
collects method/robust/view/reference-years, lays out the panels, and renders. STL runs at the annual period (365) with
`robust=False` + Loess `*_jump≈12` so it's sub-second (a **Robust** checkbox opts into the slower outlier-resistant fit).
Variable / view / reference-year changes re-render live; **method/robust apply on Update** (STL is the heavy recompute).
On <2 years of data it shows a friendly message (annual decomposition needs two cycles) and keeps the anomaly view
working. *(Building this surfaced and fixed a real library bug: `stl_decompose` never passed `period` to statsmodels and
called `STL.fit(weights=…)`, which isn't supported — STL had been raising on all real data.)*

**3-D surface explorer (`tabs/surface3d.py`):** opened from **Plot ▸ 3D surface** (single-instance). Pick a variable →
its date×time-of-day grid is rendered as a GPU-accelerated, rotatable relief — the 3-D analogue of the date/time heatmap.
The numeric grid is the **library's** `dv.plotting.datetime_surface_grid(series)` → `DateTimeSurface` (sanitize + pivot to
a complete date×time matrix; pure domain logic, no rendering backend). Everything else is presentation in the tab. Two
**Style**s: an **extruded heatmap** (default) builds a doubled-coordinate "staircase" `StructuredGrid` so each cell is a
flat bar with vertical walls (`_extruded_grid`), with NaN cells removed via `threshold` so the opaque mesh needs no
translucent pass; or a **smooth surface** with optional `subdivide` + `smooth_taubin` rounding. The base is normalised
(hours-vs-days ranges differ wildly), the date axis widened by **Y stretch**, and rows optionally binned by day
(`_bin_rows`, NaN-aware mean/median/max/min) to widen the bars; relief height comes from the exaggeration control. Colour
is by the real values (extruded mode renders opaque; the smooth mode hides gaps via `nan_opacity=0`). Lighting is flat by
default (true colours); an optional **Shadows** toggle casts short shadows from an overhead spotlight via `enable_shadows`,
with an adjustable length (`Pyvista3DCanvas.apply_shadows`). `frame_default` sets an orthographic lower-left 45° view that
re-frames only when the variable changes, so a settings tweak keeps the user's current view. 3-D is the optional
**`gui3d`** extra (`pyvista` + `pyvistaqt`, VTK-based) — lazy-imported like `gui`/`causal`, so a plain `gui` install never
pulls in VTK; without it the tab shows an install notice (`widgets/pyvista_canvas.py::pyvista_available`) instead of
failing. Install: `uv sync --extra gui --extra gui3d`.

**Spectrogram explorer (`tabs/spectrogram.py`):** opened from **Analyze ▸ Spectrogram** (single-instance). Pick a
variable → a spectrogram (short-time FFT) shows how the strength of its cycles changes over time, with a plain-language
**explanation label** above the plot describing what to look for (the bright 1-cycle/day diel band, overtones, the
window trade-off). The transform is `dv.analysis.spectrogram`; the tab maps the record-based result onto **calendar
time × cycles-per-day** axes — records-per-day is inferred from the index spacing, and each segment centre is mapped to
a real timestamp through the non-NaN sample index so the time axis stays correct even across gaps. Window length /
overlap / window function recompute on **Update**; the frequency-axis limit and colormap are live re-renders. The GUI
does no signal processing — it only calls the library and arranges the output.

**Feature engineering:** opened from **Data ▸ Feature engineering** (a menu-activated tab — `registry.MENU_TAB_CLASSES`
— not shown until selected, and closable; always-on tabs have their close button removed). It runs `FeatureEngineer`
(library) on user-selected variables and emits the new columns via a `featuresCreated` signal; `MainWindow` merges them
into the dataset, records them in a `created` set, and re-pushes. The plotting list tags created columns with a pink
**✦ NEW** pill (delegate `CREATED_ROLE`). Heavy runs go on a worker thread; progress shows in the Log tab. The created
columns are also listed explicitly in a "Newly created features" panel in the tab. Three fixed-width columns
(available / selected / settings) packed left keep it compact. **Timestamp features and the continuous record number
need no selected variable** (they derive from the index) — the run only requires a selected variable when a per-variable
stage (lag/rolling/diff/EMA/poly/STL) is enabled.

**Events (`gui/events.py`, `tabs/events.py`, `widgets/add_event_dialog.py`):** mark *when something happened* —
fertilization, harvest, grazing, a management step. Open **Data ▸ Add event…** (or the **Data ▸ Events** tab) and pick
a single date/time, a from/to range, or a start + duration (calendar pickers). The model is the **library's**
`dv.events`: an `Event` (instant or period), `event_to_flag(event, index)` (the 0/1 yes/no column), and
`overlay_events(ax, events, axis=, colors=)` (the line/span overlay; `colors` is an optional `{category: hex}` override
map). The GUI holds the live list + a `visible` toggle **+ a user `categories` palette** (`{name: hex}`) in the
`events.manager` singleton (like `site.manager`; also a `focus_requested` signal and `duplicate`/`shift`/`request_focus`
helpers); the Events tab presents the events as a **reflowing board of cards** (`widgets/flow_layout.py::FlowLayout`) on a
soft-grey board (so the white cards stand out), one per event — soft shadow, coloured left accent + category pill (with a
stable per-category glyph), title + date settings, a relative-time hint, a position-in-record mini bar (`_SpanBar`) and a
description preview, **accented by its category colour and sorted by start date**. Each card: double-click-to-edit, a
**trashcan** delete (`icons.trash_icon`, reddening on hover), and a **⋯ menu** (show on the Overview, edit, duplicate,
shift ±1 day). The board has a **filter** field (name/category), a **Group** combo (None / Category / Year → light section
headers) and a **Density** combo (Comfortable / Compact), plus a dashed add-event ghost card, **Add event…**, **Manage
categories…** (`widgets/category_dialog.py::CategoryDialog` — add / rename / recolour / remove categories, palette seeded
with generic `category1/2/3` and the last one undeletable; `categories_changed` repaints cards + overlays without
rebuilding columns), and a master **Show events on plots** checkbox (mirrored by a checkable Data-menu action). "Show on
the Overview" fires `events.manager.request_focus(start, end)` → `MainWindow._focus_event_on_overview` → switches to the
Overview and `OverviewTab.focus_on(start, end)` zooms the linked datetime panels onto the event. A category's colour
overrides the library default everywhere via `Event.resolved_color(i, colors=...)` — consistent on the cards and the plot
overlays. `EventsTab.save_state`/`restore_state` persist the group + density choices with a project. Each event becomes a real `EVENT_<name>` 0/1
column (1 = the event took place) — `MainWindow._sync_event_columns` reconciles those columns to the event list on every
`events.manager.changed`, tracking the ones it created (`_event_columns`) so it never drops an `EVENT_`-named column that
came in as plain data. The Overview draws the overlays (`_overlay_events`): a dashed line for an instant and a shaded
band for a period on the time-series/cumulative/daily panels (labelled on the time series), and a horizontal line/band on
the date/time heatmap (date is on its y-axis). Events persist in GUI prefs (`config` `"events"`) and inside projects
(`extras["events"]`), and `_set_data` rebuilds their columns on whatever data is loaded so they survive loads.

**Project settings & notes wall (`tabs/site.py`):** the Project settings tab's form writes author/description/site to
`site.manager`; the otherwise-empty right side holds a **`NotesWall`** (`widgets/notes_wall.py`) — a free-positioning
pinboard of sticky-note cards (drag by the header bar, resize by the corner grip, recolour from a palette, delete). The
wall mirrors its `state()` into `site.manager.notes` on every edit (a plain attribute set — no `changed` signal — to
avoid a rebuild loop), and rebuilds from the store only when the notes genuinely differ (e.g. a project was opened). It
reuses the existing site persistence, so notes travel with the project (`extras["site"]`) and GUI prefs with no
`app.py` change. Pure presentation (cards/colours/positions); card text colour is the WCAG-contrast pick.

**Cumulative provenance + tolerant project metadata (library, `diive.core.metadata`):** `MetadataStore.record_derived`
makes a new column **inherit its parent's full history** on first creation (a *copied* snapshot), so a chain like
`FC → FC_LOCALSD → FC_LOCALSD_HAMPEL` shows all three steps in the explorer, not just the last. `MetadataStore.rename`
re-keys records and rewrites parent/provenance links. Deserialization is **tolerant of older project layouts**:
`VariableMetadata.from_dict` accepts an aliased/missing name and either the current `{tag: source}` dict or an older
bare tag list (`_coerce_tag_sources`), and `load_dict` skips malformed entries instead of failing the whole project
load — so a stale `.diive` folder still opens with whatever metadata is recoverable.

**Shared variable list (`VariablePanel`):** every tab's left list MUST be this one widget so styling, pills, filtering
(separator-insensitive subsequence), and width are identical everywhere. Its width is a shared appearance setting
(`theme.manager.list_width`, editable in Appearance). `run_with_loading(name, fn)` shows a busy indicator on the clicked
variable + wait cursor while `fn` (a synchronous matplotlib render) runs — a static cue, since the render blocks the
event loop (true animation would need off-thread Agg rendering).

**Variable list stays in sync:** every data change (file load, feature add) goes through `MainWindow._push_data()`,
which calls `on_data_loaded(df, created)` on all active tabs. A menu tab gets the current data on open and is then
subscribed; on close it's removed so it can't go stale.

**Rename / delete a variable (any tab):** the `VariablePanel` right-click menu offers **Rename…** and **Delete…**
everywhere — both route through `metadata_store.manager` (`request_rename`/`request_delete` → `renameRequested`/
`deleteRequested`, connected once in `MainWindow`), so no per-tab wiring is needed (same pattern as `editRequested`).
`MainWindow._rename_one_variable` prompts + checks for collisions, then `_rename_variables` renames the column in
`_full_data`, remaps the `created` set, and calls the library `MetadataStore.rename(mapping)` (which re-keys records and
rewrites parent/provenance links so lineage survives). `_delete_variable` drops the column. Both are non-destructive to
the source file and re-derive the active range so every tab refreshes. The **Rename variables** tab (Data menu) reuses
this: bulk prefix/suffix emits `variablesRenamed`, and a row double-click calls `request_rename`.

**Newly created variables surface in the Overview.** When columns are added (outlier/feature tabs → `featuresCreated` →
`_add_features` → push), `OverviewTab.on_data_loaded` diffs the incoming `created` set against the previous one to find
the **new** columns and: clears the list's fuzzy filter (so a non-matching new name isn't hidden), `scroll_to`s the new
row, and auto-selects/plots the new variable (skipping its `FLAG_…_TEST`). The **variable subset is app-wide** — it
narrows `MainWindow._data` (see `_apply_var_subset`/`_reset_var_subset`), so every non-pinned tab (not just the Overview)
sees only the chosen variables; the Overview holds no subset state of its own.

**Tab UX & pinning:** tabs are movable (drag), renamable (**left** double-click → `_rename_tab`), and menu tabs carry a
custom visible "×" (`icons.close_icon`); the always-on Overview/Log are not closable (a `tabCloseRequested` for them —
incl. middle-click — is ignored). `tabBarDoubleClicked` fires for any button, so an `eventFilter` on the tab bar records
the double-click button and `_rename_tab` ignores middle/right ones. **Right-click a menu tab → Pin** freezes it on its current dataset: pinned tabs
(`MainWindow._pinned`) are skipped by `_push_data` (cheap — references + pandas Copy-on-Write) and show a pin glyph
(`icons.pin_icon`); unpin re-syncs. Overview/Log are never pinnable. The app/taskbar icon is drawn from the splash
motif (`splash.app_icon`); a Windows AppUserModelID is set in `run()` so the taskbar uses it.

**Hover tooltip (`HoverAnnotator`):** `MplCanvas` attaches one in its constructor; it works on every figure rendered into the
canvas (Overview, plotting tabs) with no per-tab wiring. On mouse-move it shows a small box with the value under the cursor:
**line** artists snap to the nearest sample along x (`np.searchsorted` on the unit-converted floats — use `get_xdata(orig=False)`,
not the raw datetimes — so it stays O(log n) on large series) and show a marker; **`pcolormesh`** heatmaps read the cell from the
grid (`get_coordinates()` + reshaped `get_array()`, cached per draw). It renders by **blitting** (cache the background on
`draw_event`, redraw just the annotation on move), so it never forces a full repaint. Pure presentation — no data/domain logic —
so it lives in the GUI. A **"Hover values"** checkbox in the canvas's bottom row (next to the navigation toolbar) toggles it
(`hover.set_enabled`); the toolbar's own x/y coordinate readout is disabled (`coordinates=False`) since the tooltip replaces it.

**Editable-field styling:** every editable input (`QLineEdit` / `QSpinBox` / `QDoubleSpinBox` / `QComboBox`) gets a tinted
background (the `INPUT_BG` theme token) in `theme.build_qss`, so it's obvious what can be edited; read-only areas (lists,
the flux summary box) stay plain. `INPUT_BG` is editable in the Appearance tab like the other tokens.

**Parameter tooltips from docstrings (not hardcoded):** `diive.core.utils.docstrings.param_docs(obj)` (library) extracts
`{param: description}` from Google-style `Args:` sections and class attribute docstrings (e.g. `FluxConfig` fields). Tabs
map each control to its library param and `setToolTip` from that — the plot-settings panel (`_apply_tooltips`, over the
`plot()` method) and the flux-chain tab (over `init_flux_data` / `FluxConfig` / `run_level2`). Help text stays in sync
with the library automatically; add a control → it gets a tooltip for free if the param is documented.

**Output console:** the **Log** tab (`LogTab` → `ConsolePanel`) mirrors diive's Rich output in colour. It registers a
Rich mirror console via `add_console_sink` (in `diive.core.utils.console`) — the library tees its output to any
registered sink; the panel renders the ANSI stream into a `QTextEdit`. The redirect hook lives in the library; the
panel only renders.

## PySide6 gotchas baked into this code

- **Keep tab instances alive.** `MainWindow` stores tabs in `self._tabs`; Qt owns the QWidgets, but the Python tab
  objects (holding the signal slots) would otherwise be garbage-collected and their signals go inert.
- **A stylesheet touching `QListWidget::item` disables per-item `setBackground`/`setForeground`.** Row colouring is
  therefore done in `VariableDelegate.paint`, not via item roles.
- **The matplotlib Qt toolbar recolours icons from the widget palette.** `MplCanvas` sets a light palette *before*
  building the toolbar so icons stay dark on the white background (otherwise white-on-white on dark system themes).
- **Use synchronous `canvas.draw()`, not `draw_idle()`,** after a user action so the plot updates immediately.
- **Combo-box popups show a native frame/shadow as black bars** on the frameless translucent window. `widgets/combo.py::
  install_combo_popup_fix(app)` (called in `run()`) installs one app-wide event filter that re-flags every popup
  container frameless + no-shadow + translucent at creation — fixes all dropdowns from one place.
- **`QMenu` popups render black** the same way (the menu *is* the popup, so the combo trick can't catch them — Polish/Show
  don't route through the app filter reliably). Build every context menu with `widgets/menu.py::studio_menu(parent)`,
  which applies the frameless/no-shadow/translucent + `#studiomenu` treatment the QSS rounds into a white card.
- **Drawn glyphs, not font characters, for tiny buttons.** Unicode marks like `⋯` are missing from many fonts and render
  blank (the card's "more" button was invisible). Use a `QPainter`-drawn icon (`icons.dots_icon` / `locate_icon` /
  `trash_icon`) instead.
- **The global `QWidget { background: CANVAS }` QSS rule paints every widget white.** A tab wanting a different surface
  (e.g. the Events board's grey) must set its own `#objectName` background *and* give child labels
  `background: transparent`, or they stay white.
- **`DiiveTab` is a plain `ABC`, not a `QObject`** — class-level `Signal`s on it won't bind. Put tab signals on a
  small `QObject` helper (see `FeatureEngineerTab`).
- **matplotlib renders synchronously** on the GUI thread (blocks the event loop), so loading indicators are static
  cues, not smooth animations.
- **Hover snapping must use `line.get_xdata(orig=False)`** (the unit-converted floats), not `get_xdata()` (the raw
  datetimes). `event.xdata` and `transData` are in the converted coordinate system; mixing the two snaps to the wrong
  point and breaks `num2date`. The `HoverAnnotator` re-captures its blit background on every `draw_event`, so it stays
  correct across re-renders, pan/zoom, and resize.
- **Constrained layout is frozen after the initial render, but re-solved on resize.** `constrained_layout` re-solves on
  every draw, so during interactive pan/zoom the panels jump as tick-label widths change. `MplCanvas.draw()` turns the
  layout engine off (`set_layout_engine("none")`) once the panels are placed; each new render re-enables it via
  `MplCanvas.reset_layout()` (call that, not `fig.clear()`, when building panels directly — e.g. the Overview's gridspec).
  **But** the first render happens at the tiny pre-show canvas size, so the frozen layout must adapt when the widget gets
  its real size: `_on_resize` (on `resize_event`) briefly re-enables constrained, solves via `draw_without_rendering()`,
  and re-freezes. Pan/zoom never resizes, so it stays frozen there. Forgetting the resize half leaves panels collapsed.
