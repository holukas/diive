# diive desktop GUI

PySide6 desktop application for diive. Optional dependency — install and launch with:

```bash
uv sync --extra gui      # or: pip install 'diive[gui]'
diive-gui                # or: uv run diive-gui
```

## Architecture

| File | Role |
|---|---|
| `theme.py` | **Central appearance config** — `ThemeManager` (`manager`): editable colour tokens, pill colours/labels, time-series palette, stylesheet; emits `changed` |
| `tabs/settings.py` | Appearance settings tab — live colour editing with a pill/highlight preview |
| `app.py` | `QApplication` bootstrap, `MainWindow` (menu bar + `QTabWidget`) |
| `registry.py` | `TAB_CLASSES` — the single list the main window builds tabs from |
| `tabs/base.py` | `DiiveTab` ABC: `title` + `build()` + `on_data_loaded(df)` — the extension point |
| `tabs/plotting.py` | Interactive plotting tab (variable list + heatmap panels) |
| `tabs/features.py` | Feature engineering tab (FeatureEngineer; created features get a "NEW" pill) |
| `tabs/log.py` | Log tab wrapping `ConsolePanel` (live coloured library output) |
| `widgets/mpl_canvas.py` | `MplCanvas` — embedded matplotlib figure + bottom-right toolbar |
| `widgets/variable_list.py` | `VariableList` — list emitting `selected(name, ctrl_held)` |
| `widgets/variable_delegate.py` | `VariableDelegate` — paints row highlight + NEE/GPP/Reco pills |
| `widgets/open_data_dialog.py` | `OpenDataDialog` — file + filetype picker with a parsed live preview |
| `widgets/console_panel.py` | `ConsolePanel` — dockable panel mirroring diive's Rich output in colour |

**Adding a tab:** write a `DiiveTab` subclass and append its class to `TAB_CLASSES`. The main window is agnostic to
concrete tabs — it just iterates the registry. This is how the flux processing chain will plug in later.

**Plot menu:** the menu bar's **Plot** menu is built generically from any tab exposing `plot_type_labels()` /
`set_plot_type()` (the plotting tab). It's an exclusive checkable selector; the first type (Heatmap) is the default,
checked and applied on startup. Add a plot type by extending `_PLOT_TYPES` + `_draw_one` in `tabs/plotting.py`. Ctrl+click
adds comparison panels: heatmaps go side by side (shared x/y), time series stack top-to-bottom (shared time x-axis).

**Data flow:** **File ▸ Open data file…** shows `OpenDataDialog` — pick one or more files, choose the filetype, and
preview the first parsed rows before loading (parquet via `dv.load_parquet`, other formats via `dv.ReadFileType`).
Selecting multiple files merges them (`MultiDataFileReader`, or `combine_first` for parquet). All reading is library
work, the dialog only orchestrates it. `MainWindow` holds the current DataFrame and pushes it to every tab via
`DiiveTab.on_data_loaded(df, created)`; tabs that present data override that hook to refresh. Example data auto-loads on
startup.

**Feature engineering:** opened from **Tools ▸ Feature engineering** (a menu-activated tab — `registry.MENU_TAB_CLASSES`
— not shown until selected, and closable; always-on tabs have their close button removed). It runs `FeatureEngineer`
(library) on user-selected variables and emits the new columns via a `featuresCreated` signal; `MainWindow` merges them
into the dataset, records them in a `created` set, and re-pushes. The plotting list tags created columns with a pink
**✦ NEW** pill (delegate `CREATED_ROLE`). Heavy runs go on a worker thread; progress shows in the Log tab.

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
