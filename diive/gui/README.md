# diive desktop GUI

PySide6 desktop application for diive. Optional dependency — install and launch with:

```bash
uv sync --extra gui      # or: pip install 'diive[gui]'
diive-gui                # or: uv run diive-gui
```

## Architecture

| File | Role |
|---|---|
| `app.py` | `QApplication` bootstrap, `MainWindow` (menu bar + `QTabWidget`), light-theme stylesheet |
| `registry.py` | `TAB_CLASSES` — the single list the main window builds tabs from |
| `tabs/base.py` | `DiiveTab` ABC: `title` + `build()` + `on_data_loaded(df)` — the extension point |
| `tabs/plotting.py` | Interactive plotting tab (variable list + heatmap panels) |
| `tabs/log.py` | Log tab wrapping `ConsolePanel` (live coloured library output) |
| `widgets/mpl_canvas.py` | `MplCanvas` — embedded matplotlib figure + bottom-right toolbar |
| `widgets/variable_list.py` | `VariableList` — list emitting `selected(name, ctrl_held)` |
| `widgets/variable_delegate.py` | `VariableDelegate` — paints row highlight + NEE/GPP/Reco pills |
| `widgets/open_data_dialog.py` | `OpenDataDialog` — file + filetype picker with a parsed live preview |
| `widgets/console_panel.py` | `ConsolePanel` — dockable panel mirroring diive's Rich output in colour |

**Adding a tab:** write a `DiiveTab` subclass and append its class to `TAB_CLASSES`. The main window is agnostic to
concrete tabs — it just iterates the registry. This is how the flux processing chain will plug in later.

**Data flow:** **File ▸ Open data file…** shows `OpenDataDialog` — pick one or more files, choose the filetype, and
preview the first parsed rows before loading (parquet via `dv.load_parquet`, other formats via `dv.ReadFileType`).
Selecting multiple files merges them (`MultiDataFileReader`, or `combine_first` for parquet). All reading is library
work, the dialog only orchestrates it. `MainWindow` holds the current DataFrame and pushes it to every tab via
`DiiveTab.on_data_loaded(df)`; tabs that present data override that hook to refresh. Example data auto-loads on startup.

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
