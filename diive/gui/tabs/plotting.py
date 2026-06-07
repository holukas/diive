"""
GUI.TABS.PLOTTING: INTERACTIVE PLOTTING TAB
===========================================

Two-column plotting tab: a list of variables on the left, the plot area on the
right. On startup the bundled example dataset is loaded, the variable list is
populated from its columns, and NEE is selected and rendered as a date x
time-of-day heatmap.

Selection model:
- Plain click  -> reset to a single panel showing the clicked variable.
- Ctrl + click -> append another panel to the right (up to `_MAX_PANELS`),
  for side-by-side comparison.

Selected variables are highlighted in the list and numbered with their panel
position (left to right). All panels share one date y-axis.

diive's plot classes use the two-phase `__init__(data)` / `plot(ax=...)`
pattern, so each panel is just `HeatmapDateTime(series).plot(ax=..., fig=...)`.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QWidget

import diive as dv
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_panel import VariablePanel

#: Column selected on startup -- gap-filled (continuous) NEE from the bundled
#: CH-DAV example dataset.
_DEFAULT_VAR = "NEE_CUT_REF_f"

#: Maximum number of side-by-side panels (further Ctrl+clicks are ignored).
_MAX_PANELS = 5

#: Plot-method identifiers; `_draw_one` dispatches on these. Each method is
#: opened as its own tab from the Plot menu (see registry).
HEATMAP = "Heatmap (date/time)"
TIMESERIES = "Time series"

# Time-series line colors are read live from theme.manager.ts_colors.


class PlottingTab(DiiveTab):
    """One plot method (heatmap, time series, ...) as its own closable tab."""

    def __init__(self, plot_type: str, title: str | None = None) -> None:
        super().__init__()
        self._plot_type = plot_type
        self.title = title or plot_type

    def build(self) -> QWidget:
        self._df = None
        self._panels: list[str] = []

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Shared variable list (filter + pills). Plain click resets to one
        # panel; Ctrl+click toggles additional panels.
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(self._on_selected)

        # Right: embedded matplotlib canvas.
        self.canvas = MplCanvas()

        splitter.addWidget(self.varpanel)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)   # list keeps its width
        splitter.setStretchFactor(1, 1)   # canvas takes extra space
        splitter.setSizes([260, 840])
        layout.addWidget(splitter)

        # Live theme preview: repaint pills, and re-render if colors affect the
        # current plot (time-series line colors).
        theme.manager.changed.connect(self._on_theme_changed)
        return root

    def _on_theme_changed(self) -> None:
        # The panel repaints its own pills; re-render only if colors affect the
        # current plot (time-series line colors).
        if self._plot_type == TIMESERIES and self._panels:
            self._render()

    def on_data_loaded(self, df, created: set | None = None) -> None:
        """Populate the variable list from the dataset and render.

        `created` marks user-engineered columns so they get the "NEW" pill.
        """
        self._df = df
        self._panels = []
        self.varpanel.set_variables(df.columns, created)
        self._select_default()

    def _select_default(self) -> None:
        """Highlight and render the startup variable in a single panel."""
        cols = [str(c) for c in self._df.columns]
        if _DEFAULT_VAR in cols:
            self._panels = [_DEFAULT_VAR]
        elif cols:
            self._panels = [cols[0]]
        else:
            self._panels = []
        self._render()

    def _on_selected(self, name: str, additive: bool) -> None:
        if not name:
            return
        if additive:
            # Ctrl+click toggles a panel: remove if already shown, else append.
            if name in self._panels:
                self._panels.remove(name)
            elif len(self._panels) < _MAX_PANELS:
                self._panels.append(name)
            else:
                return  # cap reached -- ignore further panels
        else:
            self._panels = [name]
        self.varpanel.run_with_loading(name, self._render)

    def _render(self) -> None:
        """Render one panel per entry in `self._panels`.

        Heatmaps go side by side (shared date/time-of-day axes; date labels on
        the leftmost only). Time series stack top to bottom (shared time x-axis;
        x labels on the bottom panel only, independent y per panel).
        """
        if not self._panels:
            # All panels toggled off -- show a blank canvas.
            self.canvas.new_axes(1)
            self.canvas.draw()
            self._mark_selected()
            return

        if self._plot_type == HEATMAP:
            axes = self.canvas.new_axes(
                len(self._panels), orientation="horizontal", sharex=True, sharey=True)
        else:
            axes = self.canvas.new_axes(
                len(self._panels), orientation="vertical", sharex=True, sharey=False)

        for i, (ax, name) in enumerate(zip(axes, self._panels)):
            self._draw_one(ax, name, i)

        if self._plot_type == HEATMAP:
            # Date axis only on the leftmost panel.
            for ax in axes[1:]:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
        else:
            # Shared time axis: x labels/ticks only on the bottom panel.
            for ax in axes[:-1]:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

        self.canvas.draw()
        self._mark_selected()

    def _draw_one(self, ax, name: str, index: int = 0) -> None:
        """Draw one variable into `ax`, or an explanatory message on failure.

        `index` is the panel position, used to pick a distinct time-series
        color. Columns that cannot be plotted (non-numeric, all-NaN) show a
        message instead of raising, so the variable list stays usable.
        """
        series = self._df[name]
        try:
            if self._plot_type == HEATMAP:
                dv.plotting.HeatmapDateTime(series).plot(
                    ax=ax, fig=self.canvas.fig, title=name,
                    cb_digits_after_comma='auto',
                )
            elif self._plot_type == TIMESERIES:
                ts_colors = theme.manager.ts_colors
                dv.plotting.TimeSeries(series).plot(
                    ax=ax, title=name, color=ts_colors[index % len(ts_colors)],
                )
            else:
                raise ValueError(f"Unknown plot type: {self._plot_type}")
        except Exception as err:
            ax.text(
                0.5, 0.5, f"Cannot plot '{name}':\n{err}",
                ha="center", va="center", wrap=True, transform=ax.transAxes,
            )

    def _mark_selected(self) -> None:
        """Highlight the selected panels in the shared variable list."""
        self.varpanel.set_panels(self._panels)
