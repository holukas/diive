"""
GUI.TABS.PLOTTING: INTERACTIVE PLOTTING TAB
===========================================

Three-column plotting tab: a list of variables on the left, a live plot-settings
panel in the middle, the plot area on the right. On startup the bundled example
dataset is loaded, the variable list is populated from its columns, and NEE is
selected and rendered as a date x time-of-day heatmap. Editing any setting
re-renders the current panels for a live preview (see `plot_settings.py`).

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
from diive.gui.widgets.plot_settings import (
    DIELCYCLE,
    HEATMAP,
    HEATMAP_YEARMONTH,
    HEXBIN,
    RIDGELINE,
    TIMESERIES,
    PlotSettingsPanel,
)
from diive.gui.widgets.variable_panel import VariablePanel

#: Plot types laid out like a heatmap (panels side by side, shared axes).
_HEATMAP_TYPES = (HEATMAP, HEATMAP_YEARMONTH)

#: Column selected on startup -- gap-filled (continuous) NEE from the bundled
#: CH-DAV example dataset.
_DEFAULT_VAR = "NEE_CUT_REF_f"

#: Maximum number of side-by-side panels (further Ctrl+clicks are ignored).
_MAX_PANELS = 5

# Plot-method identifiers (HEATMAP / TIMESERIES) are defined in plot_settings
# and re-exported here so the registry can keep importing them from this module.
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
        self._xyz: list[str] = []  # hexbin role order: [X, Y, Z]

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Shared variable list (filter + pills). Plain click resets to one
        # panel; Ctrl+click toggles additional panels.
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(self._on_selected)

        # Middle: live plot-parameter controls. Editing any control re-renders
        # the current panels (live preview).
        self.settings = PlotSettingsPanel(self._plot_type)
        self.settings.changed.connect(self._on_settings_changed)

        # Right: embedded matplotlib canvas.
        self.canvas = MplCanvas()
        # The ridgeline builds its own overlapping gridspec; keep the canvas
        # from re-flowing it (constrained layout / resize).
        if self._plot_type == RIDGELINE:
            self.canvas.auto_layout = False

        splitter.addWidget(self.varpanel)
        splitter.addWidget(self.settings)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)   # list keeps its width
        splitter.setStretchFactor(1, 0)   # settings keep their width
        splitter.setStretchFactor(2, 1)   # canvas takes extra space
        splitter.setSizes([220, 320, 780])
        layout.addWidget(splitter)

        # Live theme preview: repaint pills, and re-render if colors affect the
        # current plot (time-series line colors).
        theme.manager.changed.connect(self._on_theme_changed)
        return root

    def _on_theme_changed(self) -> None:
        # The panel repaints its own pills; re-render only if colors affect the
        # current plot (per-panel line colors come from the theme palette).
        if self._plot_type in (TIMESERIES, DIELCYCLE) and self._panels:
            self._render()

    def _on_settings_changed(self) -> None:
        # A plot parameter changed -- re-render the current panels for a live
        # preview (no-op while nothing is plotted yet).
        if self._panels:
            self._render()

    def on_data_loaded(self, df, created: set | None = None) -> None:
        """Populate the variable list from the dataset and render.

        `created` marks user-engineered columns so they get the "NEW" pill.
        """
        self._df = df
        self._panels = []
        self._xyz = []
        self.varpanel.set_variables(df.columns, created)
        self._select_default()

    def _select_default(self) -> None:
        """Highlight and render the startup variable(s)."""
        cols = [str(c) for c in self._df.columns]
        if self._plot_type == HEXBIN:
            # Hexbin needs three variables; seed a sensible driver/driver/flux
            # triple so the tab shows something on open.
            preferred = ["Tair_f", "VPD_f", "NEE_CUT_REF_f"]
            self._xyz = preferred if all(c in cols for c in preferred) else cols[:3]
            self._render()
            return
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
        if self._plot_type == HEXBIN:
            # Click cycles roles: fill X, then Y, then Z; clicking an assigned
            # variable removes it; once all three are set a new pick replaces the
            # oldest (X), sliding Y->X, Z->Y, new->Z.
            if name in self._xyz:
                self._xyz.remove(name)
            elif len(self._xyz) < 3:
                self._xyz.append(name)
            else:
                self._xyz = self._xyz[1:] + [name]
            self.varpanel.run_with_loading(name, self._render)
            return
        if self._plot_type == RIDGELINE:
            # The ridgeline uses the whole figure -> single variable only.
            self._panels = [name]
            self.varpanel.run_with_loading(name, self._render)
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
        x labels on the bottom panel only, independent y per panel). The
        ridgeline is single-variable and manages its own figure.
        """
        if self._plot_type == HEXBIN:
            self._render_hexbin()
            return

        if self._plot_type == RIDGELINE:
            self._render_ridgeline()
            return

        if not self._panels:
            # All panels toggled off -- show a blank canvas.
            self.canvas.new_axes(1)
            self.canvas.draw()
            self._mark_selected()
            return

        if self._plot_type in _HEATMAP_TYPES:
            axes = self.canvas.new_axes(
                len(self._panels), orientation="horizontal", sharex=True, sharey=True)
        else:
            axes = self.canvas.new_axes(
                len(self._panels), orientation="vertical", sharex=True, sharey=False)

        for i, (ax, name) in enumerate(zip(axes, self._panels)):
            self._draw_one(ax, name, i)

        if self._plot_type in _HEATMAP_TYPES:
            # y axis only on the leftmost panel.
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

    def _render_ridgeline(self) -> None:
        """Render the ridgeline, which builds its own stacked-density figure.

        Unlike the other plot types (one diive plot per `ax`), `RidgeLinePlot`
        lays out one density ridge per period on the whole figure, so it gets the
        canvas figure directly (via its `fig=` parameter) and we leave the layout
        to it (`canvas.auto_layout` is False for this tab).
        """
        fig = self.canvas.fig
        fig.clear()
        fig.set_layout_engine("none")
        if self._panels:
            name = self._panels[0]
            opts = self.settings.values()
            series = self._df[name].dropna()  # KDE can't fit NaNs
            try:
                dv.plotting.RidgeLinePlot(series).plot(
                    fig=fig, showplot=False, how=opts["how"],
                    hspace=opts["hspace"], shade_percentile=opts["shade_percentile"],
                    show_mean_line=opts["show_mean_line"], ascending=opts["ascending"],
                    xlabel=opts["xlabel"], kd_kwargs=opts["kd_kwargs"],
                )
            except Exception as err:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Cannot plot '{name}':\n{err}", ha="center",
                        va="center", wrap=True, transform=ax.transAxes)
        self.canvas.draw()
        self._mark_selected()

    def _render_hexbin(self) -> None:
        """Render the hexbin (single figure): z aggregated into 2D x/y bins.

        Needs all three roles set. `HexbinPlot` requires x and y to be NaN-free,
        so rows with a missing x or y are dropped jointly (keeping x/y/z aligned);
        z may keep NaNs (ignored during aggregation). z is aggregated with the
        class default (median).
        """
        self.settings.set_xyz(*(self._xyz + [None, None, None])[:3])
        ax = self.canvas.new_axes(1)[0]
        if len(self._xyz) < 3:
            ax.text(0.5, 0.5,
                    f"Click 3 variables to set X, Y, Z  ({len(self._xyz)}/3)",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            self._mark_selected()
            return
        xn, yn, zn = self._xyz
        opts = self.settings.values()
        try:
            sub = self._df[[xn, yn, zn]].dropna(subset=[xn, yn])
            dv.plotting.HexbinPlot(
                x=sub[xn], y=sub[yn], z=sub[zn],
                gridsize=opts["gridsize"], normalize_axes=opts["normalize_axes"],
                mincnt=opts["mincnt"],
            ).plot(
                ax=ax, fig=self.canvas.fig,
                cmap=opts["cmap"], vmin=opts["vmin"], vmax=opts["vmax"],
                color_bad=opts["color_bad"], zlabel=opts["zlabel"],
                xlabel=opts["xlabel"], ylabel=opts["ylabel"],
                cb_digits_after_comma=opts["cb_digits_after_comma"],
                cb_extend=opts["cb_extend"], show_colormap=opts["show_colormap"],
                show_values=opts["show_values"],
                show_values_n_dec_places=opts["show_values_n_dec_places"],
                show_values_fontsize=opts["show_values_fontsize"],
                axlabels_fontsize=opts["axlabels_fontsize"],
                ticks_labelsize=opts["ticks_labelsize"],
                cb_labelsize=opts["cb_labelsize"],
            )
        except Exception as err:
            ax.clear()
            ax.text(0.5, 0.5, f"Cannot plot hexbin:\n{err}", ha="center",
                    va="center", wrap=True, transform=ax.transAxes)
        self.canvas.draw()
        self._mark_selected()

    def _draw_one(self, ax, name: str, index: int = 0) -> None:
        """Draw one variable into `ax`, or an explanatory message on failure.

        `index` is the panel position, used to pick a distinct time-series
        color. Columns that cannot be plotted (non-numeric, all-NaN) show a
        message instead of raising, so the variable list stays usable.
        """
        series = self._df[name]
        opts = self.settings.values()
        try:
            if self._plot_type == HEATMAP:
                dv.plotting.HeatmapDateTime(
                    series, ax_orientation=opts["ax_orientation"]).plot(
                    ax=ax, fig=self.canvas.fig, title=name,
                    cmap=opts["cmap"], vmin=opts["vmin"], vmax=opts["vmax"],
                    color_bad=opts["color_bad"], zlabel=opts["zlabel"],
                    cb_digits_after_comma=opts["cb_digits_after_comma"],
                    cb_extend=opts["cb_extend"],
                    show_colormap=opts["show_colormap"],
                    show_grid=opts["show_grid"],
                    show_less_xticklabels=opts["show_less_xticklabels"],
                    show_values=opts["show_values"],
                    show_values_n_dec_places=opts["show_values_n_dec_places"],
                    show_values_fontsize=opts["show_values_fontsize"],
                    axlabels_fontsize=opts["axlabels_fontsize"],
                    ticks_labelsize=opts["ticks_labelsize"],
                    cb_labelsize=opts["cb_labelsize"],
                    minticks=opts["minticks"], maxticks=opts["maxticks"],
                )
            elif self._plot_type == HEATMAP_YEARMONTH:
                dv.plotting.HeatmapYearMonth(
                    series, agg=opts["agg"], ranks=opts["ranks"],
                    ax_orientation=opts["ax_orientation"]).plot(
                    ax=ax, fig=self.canvas.fig, title=name,
                    cmap=opts["cmap"], vmin=opts["vmin"], vmax=opts["vmax"],
                    color_bad=opts["color_bad"], zlabel=opts["zlabel"],
                    cb_digits_after_comma=opts["cb_digits_after_comma"],
                    cb_extend=opts["cb_extend"],
                    show_colormap=opts["show_colormap"],
                    show_grid=opts["show_grid"],
                    show_less_xticklabels=opts["show_less_xticklabels"],
                    show_values=opts["show_values"],
                    show_values_n_dec_places=opts["show_values_n_dec_places"],
                    show_values_fontsize=opts["show_values_fontsize"],
                    axlabels_fontsize=opts["axlabels_fontsize"],
                    ticks_labelsize=opts["ticks_labelsize"],
                    cb_labelsize=opts["cb_labelsize"],
                )
            elif self._plot_type == TIMESERIES:
                ts_colors = theme.manager.ts_colors
                dv.plotting.TimeSeries(series, drop_gaps=opts["drop_gaps"]).plot(
                    ax=ax, title=name, color=ts_colors[index % len(ts_colors)],
                    linewidth=opts["linewidth"], alpha=opts["alpha"],
                    marker=opts["marker"], xlabel=opts["xlabel"],
                    ylabel=opts["ylabel"], series_units=opts["series_units"],
                )
            elif self._plot_type == DIELCYCLE:
                ts_colors = theme.manager.ts_colors
                dv.plotting.DielCycle(series).plot(
                    ax=ax, title=name, color=ts_colors[index % len(ts_colors)],
                    mean=opts["mean"], std=opts["std"], each_month=opts["each_month"],
                    show_legend=opts["show_legend"], showgrid=opts["showgrid"],
                    legend_n_col=opts["legend_n_col"], ylabel=opts["ylabel"],
                    txt_ylabel_units=opts["txt_ylabel_units"],
                )
            else:
                raise ValueError(f"Unknown plot type: {self._plot_type}")
        except Exception as err:
            ax.text(
                0.5, 0.5, f"Cannot plot '{name}':\n{err}",
                ha="center", va="center", wrap=True, transform=ax.transAxes,
            )

    def _mark_selected(self) -> None:
        """Highlight the selected variables in the shared list.

        Hexbin numbers its three role picks (1=X, 2=Y, 3=Z); other plot types
        number their panels left-to-right / top-to-bottom.
        """
        marks = self._xyz if self._plot_type == HEXBIN else self._panels
        self.varpanel.set_panels(marks)
