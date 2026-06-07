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
| `tabs/overview.py` | Overview tab (first/default): figure (top) + full-width KPI stat-card strip (bottom) |
| `tabs/plotting.py` | `PlottingTab(plot_type)` — one closable tab per plot method (opened from the Plot menu); var list + live settings panel + canvas |
| `widgets/plot_settings.py` | `PlotSettingsPanel(plot_type)` — live plot-parameter controls (between list and canvas); `changed` re-renders; defines `HEATMAP`/`TIMESERIES` |
| `tabs/features.py` | Feature engineering tab (FeatureEngineer; created features get a "NEW" pill) |
| `tabs/log.py` | Log tab wrapping `ConsolePanel` (live coloured library output) |
| `widgets/mpl_canvas.py` | `MplCanvas` — embedded matplotlib figure + bottom-right toolbar |
| `widgets/variable_panel.py` | **`VariablePanel`** — the shared variable list (filter + pills) used by every tab |
| `widgets/variable_list.py` | `VariableList` — list emitting `selected(name, ctrl_held)` |
| `widgets/variable_delegate.py` | `VariableDelegate` — paints row highlight + NEE/GPP/Reco pills |
| `widgets/open_data_dialog.py` | `OpenDataDialog` — file + filetype picker with a parsed live preview |
| `widgets/daterange_dialog.py` | `DateRangeDialog` — from/to picker (clamped to the data span) for date-range subselection |
| `widgets/console_panel.py` | `ConsolePanel` — mirrors diive's Rich output in colour (used by the Log tab) |

**Adding a tab:** always-on tabs (Overview, Log) go in `TAB_CLASSES`. Menu-opened tabs go in `registry.MENU_TABS`
(grouped by menu; values are factories) — they open as **new numbered instances** each time (Heatmap 1, 2, 3 ...), all
closable, unless listed in `SINGLE_INSTANCE_TABS` (e.g. Appearance). The main window is agnostic to concrete tabs.

**Plot menu:** each method is its own closable tab. The **Plot** menu lists methods (Heatmap, Time series); selecting one
opens a new `PlottingTab(plot_type, title)` instance. Add a method via a factory in `registry.MENU_TABS["Plot"]` + a
branch in `plotting._draw_one`. Ctrl+click adds comparison panels: heatmaps go side by side (shared x/y), time series
stack top-to-bottom (shared time x-axis).

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

**Feature engineering:** opened from **Tools ▸ Feature engineering** (a menu-activated tab — `registry.MENU_TAB_CLASSES`
— not shown until selected, and closable; always-on tabs have their close button removed). It runs `FeatureEngineer`
(library) on user-selected variables and emits the new columns via a `featuresCreated` signal; `MainWindow` merges them
into the dataset, records them in a `created` set, and re-pushes. The plotting list tags created columns with a pink
**✦ NEW** pill (delegate `CREATED_ROLE`). Heavy runs go on a worker thread; progress shows in the Log tab.

**Shared variable list (`VariablePanel`):** every tab's left list MUST be this one widget so styling, pills, filtering
(separator-insensitive subsequence), and width are identical everywhere. Its width is a shared appearance setting
(`theme.manager.list_width`, editable in Appearance). `run_with_loading(name, fn)` shows a busy indicator on the clicked
variable + wait cursor while `fn` (a synchronous matplotlib render) runs — a static cue, since the render blocks the
event loop (true animation would need off-thread Agg rendering).

**Variable list stays in sync:** every data change (file load, feature add) goes through `MainWindow._push_data()`,
which calls `on_data_loaded(df, created)` on all active tabs. A menu tab gets the current data on open and is then
subscribed; on close it's removed so it can't go stale.

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
