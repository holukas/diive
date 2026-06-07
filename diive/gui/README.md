# diive desktop GUI

> Using the app? See the **[user manual](MANUAL.md)**. This file is the developer
> map (architecture, gotchas).

PySide6 desktop application for diive. Optional dependency — install and launch with:

```bash
uv sync --extra gui      # or: pip install 'diive[gui]'
diive-gui                # or: uv run diive-gui
```

## Architecture

| File | Role |
|---|---|
| `theme.py` | **Central appearance config** — `ThemeManager` (`manager`): editable colour tokens, pill colours/labels, time-series palette, stylesheet; emits `changed`; `as_dict`/`load_dict` for persistence |
| `config.py` | Persist preferences (theme, window geometry, last filetype) as JSON in the user config dir |
| `tabs/settings.py` | Appearance settings tab — live colour editing with a pill/highlight preview |
| `app.py` | `QApplication` bootstrap, `MainWindow` (menu bar + `QTabWidget`); window sized to ~88% of screen |
| `registry.py` | `TAB_CLASSES` (always-on), `MENU_TABS` (menu-opened factories), `SINGLE_INSTANCE_TABS` |
| `tabs/base.py` | `DiiveTab` ABC: `title` + `build()` + `on_data_loaded(df, created)` — the extension point |
| `tabs/overview.py` | Overview tab (first/default): 2×4 panel figure (top) + full-width KPI stat-card strip (bottom); panels via `_PANELS` |
| `tabs/plotting.py` | `PlottingTab(plot_type)` — one closable tab per plot method (opened from the Plot menu); var list + live settings panel + canvas |
| `icons.py` | `menu_icon(label)` — tiny `QPainter`-drawn glyphs for **all** menu entries (folder/disk/calendar/gear/palette/… + plot shapes), keyword-matched |
| `widgets/plot_settings.py` | `PlotSettingsPanel(plot_type)` — live plot-parameter controls (between list and canvas); `changed` re-renders; defines `HEATMAP`/`TIMESERIES` |
| `tabs/features.py` | Feature engineering tab (FeatureEngineer; created features get a "NEW" pill) |
| `tabs/fluxchain.py` | Flux processing chain tab — Input + Level 2 (first slice); runs `init_flux_data`/`run_level2`, **Copy Python** emits a reproducible script |
| `tabs/log.py` | Log tab wrapping `ConsolePanel` (live coloured library output) |
| `widgets/mpl_canvas.py` | `MplCanvas` — embedded matplotlib figure + bottom-right toolbar; attaches a `HoverAnnotator` |
| `widgets/hover.py` | `HoverAnnotator` — value-under-cursor tooltip (line snap + heatmap cell) via blitting |
| `widgets/variable_panel.py` | **`VariablePanel`** — the shared variable list (filter + pills) used by every tab |
| `widgets/variable_list.py` | `VariableList` — list emitting `selected(name, ctrl_held)` |
| `widgets/variable_delegate.py` | `VariableDelegate` — paints row highlight + NEE/GPP/Reco pills |
| `widgets/open_data_dialog.py` | `OpenDataDialog` — file + filetype picker with a parsed live preview |
| `widgets/daterange_dialog.py` | `DateRangeDialog` — from/to picker (clamped to the data span) for date-range subselection |
| `widgets/console_panel.py` | `ConsolePanel` — mirrors diive's Rich output in colour (used by the Log tab) |

**Adding a tab:** always-on tabs (Overview, Log) go in `TAB_CLASSES`. Menu-opened tabs go in `registry.MENU_TABS`
(grouped by menu; values are factories) — they open as **new numbered instances** each time (Heatmap 1, 2, 3 ...), all
closable, unless listed in `SINGLE_INSTANCE_TABS` (e.g. Appearance). The main window is agnostic to concrete tabs.

**Menu icons:** every menu entry (File/Data/Plot/Tools/Settings/Help) gets a small `QPainter`-drawn glyph via
`gui/icons.py::menu_icon(label)`, matched to the label by keyword (`&` mnemonics stripped first). `_build_menu` wraps
each action with it. Add a menu entry → add a keyword rule in `icons._RULES` (unknown labels fall back to a chart glyph).

**Plot menu:** each method is its own closable tab, with a small drawn icon. The **Plot** menu lists methods (Heatmap
date/time, Heatmap year/month, Time series, Ridgeline); selecting one
opens a new `PlottingTab(plot_type, title)` instance. Add a method via a factory in `registry.MENU_TABS["Plot"]` + a
branch in `plotting._draw_one` (and matching controls in `plot_settings`). Ctrl+click adds comparison panels: heatmaps
(both kinds, in `_HEATMAP_TYPES`) go side by side (shared x/y), time series stack top-to-bottom (shared time x-axis).
The **ridgeline** is single-variable and whole-figure: `RidgeLinePlot` builds its own stacked-density gridspec, so the
tab passes `canvas.fig` to the class's `fig=` param and sets `canvas.auto_layout=False` (so the constrained-layout
freeze/resize machinery doesn't reflow its overlapping ridges) — see `_render_ridgeline`.

**Live plot settings:** between the variable list and the canvas sits a `PlotSettingsPanel(plot_type)` — a scrollable
strip of controls, one per `plot()` parameter of the underlying diive plot class (heatmap: colormap, vmin/vmax,
orientation, colorbar, cell-value overlay, ticks, grid, …; time series: line width, opacity, markers, drop-gaps,
labels/units). Editing any control emits `changed`; the tab re-renders the current panels (`_on_settings_changed` →
`_render`), and `_draw_one` reads `settings.values()` into the library plot call. The panel is GUI-only (it just
collects parameters); the `HEATMAP`/`TIMESERIES` constants live in `plot_settings.py` and `plotting.py` re-exports them
(so no import cycle). Line *colours* stay theme-driven (`theme.manager.ts_colors`, Appearance tab), not duplicated here.
Add a parameter = add a control in `plot_settings._build_*` + a key in `values()` + pass it through in `_draw_one`.

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

**Flux processing chain (`tabs/fluxchain.py`):** opened from **Tools ▸ Flux processing chain** (single-instance). A
guided tab for the Swiss-FluxNet chain. **First slice = Input + Level 2:** collects site/flux-column + which L2 quality
tests to run, then on a worker thread calls the composable library callables (`init_flux_data` → `run_level2`), shows the
L2 QCF-filtered flux as a date/time heatmap, and — the point of the feature — **Copy Python** emits the exact runnable
script via the library's `level2_to_code`. The script-gen lives in the library (`flux/fluxprocessingchain/codegen.py`:
`chain_to_code` for the full `run_chain`/`FluxConfig` path, `level2_to_code` for the composable path) because it encodes
the API call shape; the GUI only calls it. Needs real EddyPro-FLUXNET input (FC/USTAR/`*_TEST` columns) —
`load_exampledata_parquet_lae_level1_30MIN`, not the default CH-DAV. **Later slices** add L3.1/3.2/3.3/4.1 groups and
switch to `run_chain`/`chain_to_code`; per-level live preview can reuse the cascade-aware `run_level*` callables.

**Feature engineering:** opened from **Tools ▸ Feature engineering** (a menu-activated tab — `registry.MENU_TAB_CLASSES`
— not shown until selected, and closable; always-on tabs have their close button removed). It runs `FeatureEngineer`
(library) on user-selected variables and emits the new columns via a `featuresCreated` signal; `MainWindow` merges them
into the dataset, records them in a `created` set, and re-pushes. The plotting list tags created columns with a pink
**✦ NEW** pill (delegate `CREATED_ROLE`). Heavy runs go on a worker thread; progress shows in the Log tab. The created
columns are also listed explicitly in a "Newly created features" panel in the tab. Three fixed-width columns
(available / selected / settings) packed left keep it compact. **Timestamp features and the continuous record number
need no selected variable** (they derive from the index) — the run only requires a selected variable when a per-variable
stage (lag/rolling/diff/EMA/poly/STL) is enabled.

**Shared variable list (`VariablePanel`):** every tab's left list MUST be this one widget so styling, pills, filtering
(separator-insensitive subsequence), and width are identical everywhere. Its width is a shared appearance setting
(`theme.manager.list_width`, editable in Appearance). `run_with_loading(name, fn)` shows a busy indicator on the clicked
variable + wait cursor while `fn` (a synchronous matplotlib render) runs — a static cue, since the render blocks the
event loop (true animation would need off-thread Agg rendering).

**Variable list stays in sync:** every data change (file load, feature add) goes through `MainWindow._push_data()`,
which calls `on_data_loaded(df, created)` on all active tabs. A menu tab gets the current data on open and is then
subscribed; on close it's removed so it can't go stale.

**Hover tooltip (`HoverAnnotator`):** `MplCanvas` attaches one in its constructor; it works on every figure rendered into the
canvas (Overview, plotting tabs) with no per-tab wiring. On mouse-move it shows a small box with the value under the cursor:
**line** artists snap to the nearest sample along x (`np.searchsorted` on the unit-converted floats — use `get_xdata(orig=False)`,
not the raw datetimes — so it stays O(log n) on large series) and show a marker; **`pcolormesh`** heatmaps read the cell from the
grid (`get_coordinates()` + reshaped `get_array()`, cached per draw). It renders by **blitting** (cache the background on
`draw_event`, redraw just the annotation on move), so it never forces a full repaint. Pure presentation — no data/domain logic —
so it lives in the GUI. A **"Hover values"** checkbox in the canvas's bottom row (next to the navigation toolbar) toggles it
(`hover.set_enabled`); the toolbar's own x/y coordinate readout is disabled (`coordinates=False`) since the tooltip replaces it.

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
