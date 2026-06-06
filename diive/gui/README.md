# diive desktop GUI

PySide6 desktop application for diive. Optional dependency — install and launch with:

```bash
uv sync --extra gui      # or: pip install 'diive[gui]'
diive-gui                # or: uv run diive-gui
```

## Architecture

| File | Role |
|---|---|
| `app.py` | `QApplication` bootstrap, `MainWindow` (a `QTabWidget`), light-theme stylesheet |
| `registry.py` | `TAB_CLASSES` — the single list the main window builds tabs from |
| `tabs/base.py` | `DiiveTab` ABC: `title` + `build()` — the extension point |
| `tabs/plotting.py` | Interactive plotting tab (variable list + heatmap panels) |
| `widgets/mpl_canvas.py` | `MplCanvas` — embedded matplotlib figure + bottom-right toolbar |
| `widgets/variable_list.py` | `VariableList` — list emitting `selected(name, ctrl_held)` |
| `widgets/variable_delegate.py` | `VariableDelegate` — paints row highlight + NEE/GPP/Reco pills |

**Adding a tab:** write a `DiiveTab` subclass and append its class to `TAB_CLASSES`. The main window is agnostic to
concrete tabs — it just iterates the registry. This is how the flux processing chain will plug in later.

## PySide6 gotchas baked into this code

- **Keep tab instances alive.** `MainWindow` stores tabs in `self._tabs`; Qt owns the QWidgets, but the Python tab
  objects (holding the signal slots) would otherwise be garbage-collected and their signals go inert.
- **A stylesheet touching `QListWidget::item` disables per-item `setBackground`/`setForeground`.** Row colouring is
  therefore done in `VariableDelegate.paint`, not via item roles.
- **The matplotlib Qt toolbar recolours icons from the widget palette.** `MplCanvas` sets a light palette *before*
  building the toolbar so icons stay dark on the white background (otherwise white-on-white on dark system themes).
- **Use synchronous `canvas.draw()`, not `draw_idle()`,** after a user action so the plot updates immediately.
