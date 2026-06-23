"""
GUI.TABS.PLOTTING: INTERACTIVE PLOTTING TAB
===========================================

Three-column plotting tab: a list of variables on the left, a live plot-settings
panel in the middle, the plot area on the right. On startup the bundled example
dataset is loaded, the variable list is populated from its columns, and NEE is
selected and rendered as a date x time-of-day heatmap. Editing a setting does
not re-render on its own; the user clicks the **Update plot** button to apply all
pending parameter changes at once (see `plot_settings.py`). Selecting a variable
in the list still re-renders immediately.

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

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.flow_layout import FlowLayout
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.plot_settings import (
    CUMULATIVE_YEAR,
    DIELCYCLE,
    HEATMAP,
    HEATMAP_YEARMONTH,
    HEXBIN,
    HISTOGRAM,
    RIDGELINE,
    SCATTER,
    TIMESERIES,
    WINDROSE,
    PlotSettingsPanel,
)

#: Plot types that pick variables by role (not comparison panels): hexbin/scatter
#: X/Y/Z and the wind rose value/wind-direction/colour.
_XYZ_TYPES = (HEXBIN, SCATTER, WINDROSE)
from diive.gui.widgets.variable_panel import VariablePanel, lock_panel_handle

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


class _PanelPills(QWidget):
    """A wrapping row of segmented pill buttons — one per subplot panel — for
    choosing which panel's settings the controls below currently edit.

    Reuses the `SubTabs` pill look (accent fill on the active pill) but without a
    stacked widget: there is one settings panel; switching pills only swaps which
    panel's saved settings are loaded into it. Labels wrap (`FlowLayout`) so they
    fit the narrow settings column; long variable names are elided with a tooltip.
    """

    changed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._buttons: list[QPushButton] = []
        self._flow = FlowLayout(self, margin=0, hspacing=4, vspacing=4)
        theme.manager.changed.connect(self._apply_style)

    def set_panels(self, labels: list[str], current: int = 0) -> None:
        while self._flow.count():
            item = self._flow.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        self._buttons = []
        for i, label in enumerate(labels):
            b = QPushButton(self._elide(label))
            b.setToolTip(label)
            b.setCheckable(True)
            b.setChecked(i == current)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _checked=False, idx=i: self._on_click(idx))
            self._buttons.append(b)
            self._flow.addWidget(b)
        self._apply_style()

    def set_current(self, idx: int) -> None:
        for i, b in enumerate(self._buttons):
            b.setChecked(i == idx)

    def _on_click(self, idx: int) -> None:
        self.set_current(idx)
        self.changed.emit(idx)

    @staticmethod
    def _elide(text: str, n: int = 16) -> str:
        return text if len(text) <= n else text[: n - 1] + "…"

    def _apply_style(self) -> None:
        accent = theme.manager.tokens.get("ACCENT", "#3A4D5C")
        border = theme.manager.tokens.get("BORDER", "#E6E6E3")
        qss = (
            f"QPushButton {{ padding: 4px 10px; border: 0.5px solid {border}; "
            f"border-radius: 6px; background: transparent; }} "
            f"QPushButton:checked {{ background: {accent}; color: white; "
            f"border-color: {accent}; }}")
        for b in self._buttons:
            b.setStyleSheet(qss)


class PlottingTab(DiiveTab):
    """One plot method (heatmap, time series, ...) as its own closable tab."""

    def __init__(self, plot_type: str, title: str | None = None) -> None:
        super().__init__()
        self._plot_type = plot_type
        self.title = title or plot_type

    def build(self) -> QWidget:
        self._df = None
        self._panels: list[str] = []
        self._xyz: list[str] = []  # hexbin/scatter role order: [X, Y, Z]
        # Per-subplot settings: each panel keeps its own raw control snapshot
        # (PlotSettingsPanel.state()), keyed by variable name. `_active_panel`
        # is the one the live controls currently edit.
        self._panel_settings: dict[str, list] = {}
        self._active_panel: str | None = None
        self._last_axes: list = []  # main panel axes of the last render (for zoom-preserve)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Main header (tracked/bold, e.g. "TIME SERIES PLOT"), matching the
        # correction/gap-filling tabs' title bar.
        outer.addLayout(build_titlebar(f"{self.title} plot"))

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Shared variable list (filter + pills) under its own header. Plain click
        # resets to one panel; Ctrl+click toggles additional panels. Time series
        # lists are draggable so a variable can be dropped onto the colour-by field.
        self.varpanel = VariablePanel(draggable=self._plot_type == TIMESERIES)
        self.varpanel.selected.connect(self._on_selected)
        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.addWidget(list_header("Variable", "click to plot"))
        llay.addWidget(self.varpanel, 1)

        # Middle: plot-parameter controls (under a header) plus an explicit
        # "Update plot" button. Editing a control does NOT re-render — the user
        # clicks the button to apply all pending changes at once (a single,
        # consistent trigger across every control type). Variable selection in
        # the list still renders live.
        self.settings = PlotSettingsPanel(self._plot_type)
        # Segmented pill row to pick which subplot's settings to edit (shown only
        # with more than one panel). Switching a pill swaps the loaded settings.
        self.panel_pills = _PanelPills()
        self.panel_pills.changed.connect(self._on_pill_changed)
        self.panel_pills.setVisible(False)
        self.update_btn = QPushButton("Update plot")
        theme.set_button_role(self.update_btn, "confirm")
        # Keep the current pan/zoom when applying setting changes.
        self.update_btn.clicked.connect(lambda: self._render(preserve_view=True))
        middle = QWidget()
        mlay = QVBoxLayout(middle)
        mlay.setContentsMargins(0, 0, 0, 0)
        mlay.setSpacing(4)
        mlay.addWidget(list_header("Plot settings", "adjust & update"))
        mlay.addWidget(self.panel_pills)
        mlay.addWidget(self.settings, 1)
        mlay.addWidget(self.update_btn)

        # Right: embedded matplotlib canvas.
        self.canvas = MplCanvas()
        # The ridgeline builds its own overlapping gridspec; the wind rose builds
        # its own polar axes (+ colorbar) and sets its own margins. Keep the
        # canvas from re-flowing either (constrained layout / resize).
        if self._plot_type in (RIDGELINE, WINDROSE):
            self.canvas.auto_layout = False

        splitter.addWidget(left)
        splitter.addWidget(middle)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)   # list keeps its width
        splitter.setStretchFactor(1, 0)   # settings keep their width
        splitter.setStretchFactor(2, 1)   # canvas takes extra space
        splitter.setSizes([220, 320, 780])
        # Lock only the handle next to the fixed-width list (handle 1); the
        # settings/canvas handle stays draggable.
        lock_panel_handle(splitter)
        layout.addWidget(splitter)
        outer.addWidget(body, stretch=1)

        # Live theme preview: repaint pills, and re-render if colors affect the
        # current plot (time-series line colors).
        theme.manager.changed.connect(self._on_theme_changed)
        return root

    def _on_theme_changed(self) -> None:
        # The panel repaints its own pills; re-render only if colors affect the
        # current plot (per-panel line colors come from the theme palette).
        # This tab instance is retained after its tab is closed (MainWindow keeps
        # the Python object) but its canvas C++ widget is gone — guard against
        # rendering into a deleted widget when the theme changes later.
        import shiboken6
        if not shiboken6.isValid(self.canvas):
            theme.manager.changed.disconnect(self._on_theme_changed)
            return
        if self._plot_type in (TIMESERIES, DIELCYCLE) and self._panels:
            self._render()

    # --- per-subplot settings ------------------------------------------------
    def _capture_active(self) -> None:
        """Save the live control values into the active panel's settings slot."""
        if self._active_panel is not None:
            self._panel_settings[self._active_panel] = self.settings.state()

    def _sync_panel_settings(self, active: str | None = None) -> None:
        """Reconcile per-panel snapshots to the current panels, pick the active
        panel, load its settings into the controls, and rebuild the pills.

        New panels are seeded from the current control state (so a Ctrl+click
        panel inherits the active panel's look, then can be tweaked); snapshots
        for removed panels are dropped.
        """
        self._capture_active()
        baseline = self.settings.state()  # seed for newly added panels
        panels = self._panels
        self._panel_settings = {
            n: s for n, s in self._panel_settings.items() if n in panels}
        for n in panels:
            if n in self._panel_settings:
                continue
            # Seed a new panel from the active baseline, but give additional
            # panels a distinct line colour so a stacked panel doesn't repeat the
            # active panel's (possibly explicit) colour.
            self.settings.apply_state(baseline)
            self._assign_distinct_color(n)
            self._panel_settings[n] = self.settings.state()
        if active in panels:
            self._active_panel = active
        elif self._active_panel not in panels:
            self._active_panel = panels[0] if panels else None
        if self._active_panel is not None:
            self.settings.apply_state(self._panel_settings[self._active_panel])
        self._sync_pills()

    def _assign_distinct_color(self, name: str) -> None:
        """Give a newly added subplot (panel position > 0) a distinct line colour
        from the theme palette, so it doesn't repeat the active panel's colour.
        The primary panel (position 0) keeps its 'auto' default."""
        if not hasattr(self.settings, "line_color"):
            return
        idx = self._panels.index(name)
        if idx == 0:
            return
        ts_colors = theme.manager.ts_colors
        if ts_colors:
            self.settings.line_color.setText(ts_colors[idx % len(ts_colors)])

    def _sync_pills(self) -> None:
        panels = self._panels
        show = len(panels) > 1
        self.panel_pills.setVisible(show)
        if show:
            cur = panels.index(self._active_panel) if self._active_panel in panels else 0
            self.panel_pills.set_panels(list(panels), cur)

    def _on_pill_changed(self, idx: int) -> None:
        """Switch the controls to edit another panel's settings (no re-render —
        the user applies changes with Update plot)."""
        if idx < 0 or idx >= len(self._panels):
            return
        self._capture_active()
        self._active_panel = self._panels[idx]
        self.settings.apply_state(self._panel_settings.get(self._active_panel))

    def on_data_loaded(self, df, created: set | None = None) -> None:
        """Populate the variable list from the dataset and render.

        `created` marks user-engineered columns so they get the "NEW" pill.
        """
        self._df = df
        self._panels = []
        self._xyz = []
        self._panel_settings = {}
        self._active_panel = None
        self.varpanel.set_variables(df.columns, created)
        if self._plot_type == CUMULATIVE_YEAR:
            # Offer the data's years in the highlight-year dropdown.
            self.settings.set_years(sorted(set(df.index.year)))
        if self._plot_type == TIMESERIES:
            # Offer every column as a colour-by variable.
            self.settings.set_colorby_options(df.columns)
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
        if self._plot_type == SCATTER:
            # Scatter needs X and Y (Z optional for colour); seed a driver/flux pair.
            preferred = ["Tair_f", "NEE_CUT_REF_f"]
            self._xyz = preferred if all(c in cols for c in preferred) else cols[:2]
            self._render()
            return
        if self._plot_type == WINDROSE:
            # Needs a value + a wind-direction column. Seed the value with a flux
            # (or the first column) and the direction with the first wind-direction
            # -named column, if any (the bundled CH-DAV example has none, so the
            # tab then shows a prompt to pick variables).
            wd = next((c for c in cols if any(t in c.lower()
                       for t in ("wind_dir", "winddir", "wd", "_dir"))), None)
            val = next((c for c in ("NEE_CUT_REF_f", "Tair_f") if c in cols),
                       cols[0] if cols else None)
            self._xyz = [c for c in (val, wd) if c]
            self._render()
            return
        if _DEFAULT_VAR in cols:
            self._panels = [_DEFAULT_VAR]
        elif cols:
            self._panels = [cols[0]]
        else:
            self._panels = []
        self._sync_panel_settings(active=self._panels[0] if self._panels else None)
        self._render()

    def save_state(self) -> dict:
        self._capture_active()  # fold pending edits into the active panel's slot
        sel = self._xyz if self._plot_type in _XYZ_TYPES else self._panels
        return {"sel": list(sel), "settings": self.settings.state(),
                "panel_settings": dict(self._panel_settings),
                "active": self._active_panel}

    def restore_state(self, state: dict) -> None:
        if "settings" in state:
            self.settings.apply_state(state["settings"])  # before the render reads them
        names = set(self.varpanel.names())
        sel = [n for n in (state.get("sel") or []) if n in names]
        for i, name in enumerate(sel):
            self._on_selected(name, i > 0)  # replay clicks (first resets, rest add)
        # Overlay the saved per-panel settings (the replay above seeded panels
        # from the shared snapshot); then restore the active panel + controls.
        for n, snap in (state.get("panel_settings") or {}).items():
            if n in self._panel_settings:
                self._panel_settings[n] = snap
        active = state.get("active")
        if active in self._panel_settings:
            self._active_panel = active
        if self._active_panel is not None and self._panel_settings.get(self._active_panel):
            self.settings.apply_state(self._panel_settings[self._active_panel])
        self._sync_pills()
        self._render()

    def _on_selected(self, name: str, additive: bool) -> None:
        if not name:
            return
        if self._plot_type in _XYZ_TYPES:
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
        if self._plot_type in (RIDGELINE, HISTOGRAM):
            # The ridgeline uses the whole figure; the histogram is information-
            # dense (counts, z-score axis) -> both are single-variable.
            self._panels = [name]
            self._sync_panel_settings(active=name)
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
        self._sync_panel_settings(active=name)
        self.varpanel.run_with_loading(name, self._render)

    def _render(self, preserve_view: bool = False) -> None:
        """Render one panel per entry in `self._panels`.

        Heatmaps go side by side (shared date/time-of-day axes; date labels on
        the leftmost only). Time series stack top to bottom (shared time x-axis;
        x labels on the bottom panel only, independent y per panel). The
        ridgeline is single-variable and manages its own figure.

        `preserve_view` keeps the current pan/zoom (axis limits) across the
        rebuild — set by the "Update plot" button so tweaking a setting doesn't
        snap the view back to the full data range. Selection changes render with
        `preserve_view=False` so a newly picked variable shows in full.
        """
        if self._plot_type == HEXBIN:
            self._render_hexbin()
            return

        if self._plot_type == SCATTER:
            self._render_scatter()
            return

        if self._plot_type == RIDGELINE:
            self._render_ridgeline()
            return

        if self._plot_type == WINDROSE:
            self._render_windrose()
            return

        if not self._panels:
            # All panels toggled off -- show a blank canvas.
            self.canvas.new_axes(1)
            self.canvas.draw()
            self.canvas.reset_history()
            self._last_axes = []
            self._mark_selected()
            return

        # Capture the current view to optionally restore it after the rebuild
        # (matplotlib Axes stay readable until new_axes clears the figure).
        prev_limits = None
        if preserve_view and self._last_axes:
            prev_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in self._last_axes]

        # Persist any pending edits to the active panel before drawing, so the
        # per-panel snapshots are current.
        self._capture_active()

        if self._plot_type in _HEATMAP_TYPES:
            axes = self.canvas.new_axes(
                len(self._panels), orientation="horizontal", sharex=True, sharey=True)
        else:
            axes = self.canvas.new_axes(
                len(self._panels), orientation="vertical", sharex=True, sharey=False)

        # Restore the pre-update view only when the panel layout is unchanged.
        restore = prev_limits if (prev_limits and len(prev_limits) == len(axes)) else None

        # Each panel draws with its own settings: load that panel's snapshot into
        # the controls, draw, then apply its Axes options. This is the *full*
        # view (data extent, honouring any explicit limits) — what Home resets to.
        # `explicit` records which dims the user pinned, so the preserved zoom
        # below doesn't override an explicit limit.
        explicit: list[tuple[bool, bool]] = []
        for i, (ax, name) in enumerate(zip(axes, self._panels)):
            if self._panel_settings.get(name) is not None:
                self.settings.apply_state(self._panel_settings[name])
            self._draw_one(ax, name, i)
            self._apply_axes([ax])
            axo = self.settings.values().get("_axes") or {}
            explicit.append((
                axo.get("xmin") is not None or axo.get("xmax") is not None,
                axo.get("ymin") is not None or axo.get("ymax") is not None))
        if self._active_panel is not None and self._panel_settings.get(self._active_panel):
            self.settings.apply_state(self._panel_settings[self._active_panel])

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

        # Draw the full view and make it the toolbar's Home view.
        self.canvas.draw()
        self.canvas.reset_history()

        # Re-apply the preserved zoom on top (Update plot keeps the view), but
        # never override a dim the user pinned with an explicit limit. Recorded
        # as a second history entry so Home still returns to the full view.
        if restore is not None:
            changed = False
            for i, ax in enumerate(axes):
                if not explicit[i][0]:
                    ax.set_xlim(restore[i][0])
                if not explicit[i][1]:
                    ax.set_ylim(restore[i][1])
                changed = changed or not all(explicit[i])
            if changed:
                self.canvas.draw_idle()
                self.canvas.push_view()

        self._last_axes = list(axes)
        self._mark_selected()

    def _apply_axes(self, axes) -> None:
        """Apply the GUI-only Axes settings (limits, log scale, invert, grid).

        Pure presentation, run after the library plot has rendered. Only the
        line/scatter plot types expose an Axes group (`values()` carries an
        `_axes` dict); for everything else this is a no-op. Heatmaps, the
        ridgeline, and the diel cycle's fixed 0-24 hour x-axis are never touched
        (heatmaps/ridgeline have no `_axes`; the diel cycle's group is Y-only).
        For shared multi-panel stacks the settings apply to every panel.
        """
        ax_opts = self.settings.values().get("_axes")
        if not ax_opts:
            return
        for ax in axes:
            if ax_opts["xmin"] is not None or ax_opts["xmax"] is not None:
                ax.set_xlim(ax_opts["xmin"], ax_opts["xmax"])
            if ax_opts["ymin"] is not None or ax_opts["ymax"] is not None:
                ax.set_ylim(ax_opts["ymin"], ax_opts["ymax"])
            if ax_opts["logx"]:
                ax.set_xscale("log")
            if ax_opts["logy"]:
                ax.set_yscale("log")
            if ax_opts["invert_y"]:
                ax.invert_yaxis()
            if ax_opts["grid"]:
                # Additive: turn the grid on without stripping a plot type's
                # native grid (e.g. the time series' default y-grid) when unset.
                ax.grid(True)

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
                    fig=fig, showplot=False,
                    format_style=dv.plotting.FormatStyle(**opts["_format"]),
                    how=opts["how"],
                    hspace=opts["hspace"], shade_percentile=opts["shade_percentile"],
                    show_mean_line=opts["show_mean_line"], ascending=opts["ascending"],
                    kd_kwargs=opts["kd_kwargs"],
                )
            except Exception as err:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Cannot plot '{name}':\n{err}", ha="center",
                        va="center", wrap=True, transform=ax.transAxes)
        self.canvas.draw()
        self.canvas.reset_history()
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
                format_style=dv.plotting.FormatStyle(**opts["_format"]),
                cmap=opts["cmap"], vmin=opts["vmin"], vmax=opts["vmax"],
                color_bad=opts["color_bad"], zlabel=opts["zlabel"],
                cb_digits_after_comma=opts["cb_digits_after_comma"],
                cb_extend=opts["cb_extend"], show_colormap=opts["show_colormap"],
                show_values=opts["show_values"],
                show_values_n_dec_places=opts["show_values_n_dec_places"],
                show_values_fontsize=opts["show_values_fontsize"],
                cb_labelsize=opts["cb_labelsize"],
            )
        except Exception as err:
            ax.clear()
            ax.text(0.5, 0.5, f"Cannot plot hexbin:\n{err}", ha="center",
                    va="center", wrap=True, transform=ax.transAxes)
        self.canvas.draw()
        self.canvas.reset_history()
        self._mark_selected()

    def _render_scatter(self) -> None:
        """Scatter X vs Y, with an optional third variable (Z) colouring points.

        Single panel, picked by role (1=X, 2=Y, 3=Z). X and Y are required; Z is
        optional (omit it for a plain 2-variable scatter).
        """
        self.settings.set_xyz(*(self._xyz + [None, None, None])[:3])
        ax = self.canvas.new_axes(1)[0]
        if len(self._xyz) < 2:
            ax.text(0.5, 0.5,
                    f"Click 2 variables to set X, Y  (Z optional for colour)  "
                    f"({len(self._xyz)}/2)",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            self._mark_selected()
            return
        xn, yn = self._xyz[0], self._xyz[1]
        zn = self._xyz[2] if len(self._xyz) >= 3 else None
        opts = self.settings.values()
        try:
            cols = [xn, yn] + ([zn] if zn else [])
            sub = self._df[cols].dropna(subset=[xn, yn])
            dv.plotting.ScatterXY(
                x=sub[xn], y=sub[yn], z=(sub[zn] if zn else None),
                nbins=opts["nbins"], binagg=opts["binagg"],
            ).plot(
                ax=ax, format_style=dv.plotting.FormatStyle(**opts["_format"]),
                cmap=opts["cmap"], show_colorbar=opts["show_colorbar"],
                markersize=opts["markersize"], alpha=opts["alpha"],
                vmin=opts["vmin"], vmax=opts["vmax"],
            )
            self._apply_axes([ax])
        except Exception as err:
            ax.clear()
            ax.text(0.5, 0.5, f"Cannot plot scatter:\n{err}", ha="center",
                    va="center", wrap=True, transform=ax.transAxes)
        self.canvas.draw()
        self.canvas.reset_history()
        self._mark_selected()

    def _render_windrose(self) -> None:
        """Render the wind rose: a variable aggregated into wind-direction sectors.

        Role-picked (1 = value, 2 = wind direction, 3 = optional colour variable);
        value + direction are required, the colour variable is optional. The plot
        is polar with its own colorbar, so — like the ridgeline — it builds its own
        axes on the canvas figure (the tab leaves the layout to it).
        """
        self.settings.set_xyz(*(self._xyz + [None, None, None])[:3])
        fig = self.canvas.fig
        fig.clear()
        fig.set_layout_engine("none")
        if len(self._xyz) < 2:
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5,
                    f"Click 2 variables: value, then wind direction  "
                    f"(colour optional)  ({len(self._xyz)}/2)",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            self._mark_selected()
            return
        valn, wdn = self._xyz[0], self._xyz[1]
        zn = self._xyz[2] if len(self._xyz) >= 3 else None
        opts = self.settings.values()
        ax = fig.add_subplot(111, projection="polar")
        try:
            dv.plotting.WindRosePlot(
                series=self._df[valn], wind_dir=self._df[wdn],
                agg=opts["agg"], n_sectors=opts["n_sectors"],
                z=(self._df[zn] if zn else None), z_agg=opts["z_agg"],
            ).plot(
                ax=ax, format_style=dv.plotting.FormatStyle(**opts["_format"]),
                cmap=opts["cmap"], color=opts["color"],
                vmin=opts["vmin"], vmax=opts["vmax"],
                show_colorbar=opts["show_colorbar"], cb_label=opts["cb_label"],
                cb_digits_after_comma=opts["cb_digits_after_comma"],
                max_sector_labels=opts["max_sector_labels"],
            )
        except Exception as err:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, f"Cannot plot wind rose:\n{err}", ha="center",
                    va="center", wrap=True, transform=ax.transAxes)
        self.canvas.draw()
        self.canvas.reset_history()
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
                    ax=ax, fig=self.canvas.fig,
                    format_style=dv.plotting.FormatStyle(**opts["_format"]),
                    cmap=opts["cmap"], vmin=opts["vmin"], vmax=opts["vmax"],
                    color_bad=opts["color_bad"], zlabel=opts["zlabel"],
                    cb_digits_after_comma=opts["cb_digits_after_comma"],
                    cb_extend=opts["cb_extend"],
                    show_colormap=opts["show_colormap"],
                    show_less_xticklabels=opts["show_less_xticklabels"],
                    show_values=opts["show_values"],
                    show_values_n_dec_places=opts["show_values_n_dec_places"],
                    show_values_fontsize=opts["show_values_fontsize"],
                    cb_labelsize=opts["cb_labelsize"],
                    minticks=opts["minticks"], maxticks=opts["maxticks"],
                )
            elif self._plot_type == HEATMAP_YEARMONTH:
                dv.plotting.HeatmapYearMonth(
                    series, agg=opts["agg"], ranks=opts["ranks"],
                    ax_orientation=opts["ax_orientation"]).plot(
                    ax=ax, fig=self.canvas.fig,
                    format_style=dv.plotting.FormatStyle(**opts["_format"]),
                    cmap=opts["cmap"], vmin=opts["vmin"], vmax=opts["vmax"],
                    color_bad=opts["color_bad"], zlabel=opts["zlabel"],
                    cb_digits_after_comma=opts["cb_digits_after_comma"],
                    cb_extend=opts["cb_extend"],
                    show_colormap=opts["show_colormap"],
                    show_less_xticklabels=opts["show_less_xticklabels"],
                    show_values=opts["show_values"],
                    show_values_n_dec_places=opts["show_values_n_dec_places"],
                    show_values_fontsize=opts["show_values_fontsize"],
                    cb_labelsize=opts["cb_labelsize"],
                )
            elif self._plot_type == TIMESERIES:
                # Explicit per-panel colour wins; "auto" falls back to the theme
                # palette colour for this panel. The line colour also colours the
                # markers (single library plot() call).
                ts_colors = theme.manager.ts_colors
                color = opts.get("color") or ts_colors[index % len(ts_colors)]
                # Colour-by-variable: pass the second series; the library colours
                # the line by its values (the single colour is then ignored).
                cby = opts.get("color_by")
                color_series = (self._df[cby] if cby and cby in self._df.columns
                                else None)
                # Chrome (title/labels/units/fonts) goes through the shared
                # FormatStyle; the line-rendering args stay direct.
                fmt = dict(opts["_format"])
                if fmt.get("title") is None:
                    fmt["title"] = name  # default the title to the variable name
                dv.plotting.TimeSeries(
                    series, drop_gaps=opts["drop_gaps"], color_series=color_series).plot(
                    ax=ax, format_style=dv.plotting.FormatStyle(**fmt),
                    color=color,
                    linewidth=opts["linewidth"], alpha=opts["alpha"],
                    marker=opts["marker"], markersize=opts["markersize"],
                    cmap=opts["color_by_cmap"], color_label=cby,
                )
            elif self._plot_type == DIELCYCLE:
                ts_colors = theme.manager.ts_colors
                # DielCycle auto-colors per month only when color is None; pass
                # None for the per-month view so each month gets its own colour,
                # otherwise the theme line colour for this panel.
                color = None if opts["each_month"] else ts_colors[index % len(ts_colors)]
                fmt = dict(opts["_format"])
                if fmt.get("title") is None:
                    fmt["title"] = name  # default the title to the variable name
                dv.plotting.DielCycle(series).plot(
                    ax=ax, format_style=dv.plotting.FormatStyle(**fmt), color=color,
                    mean=opts["mean"], std=opts["std"], each_month=opts["each_month"],
                )
            elif self._plot_type == CUMULATIVE_YEAR:
                dv.plotting.CumulativeYear(
                    series, series_units=opts["series_units"],
                    yearly_end_date=opts["yearly_end_date"],
                    show_reference=opts["show_reference"],
                    highlight_year=opts["highlight_year"],
                ).plot(ax=ax, showplot=False,
                       format_style=dv.plotting.FormatStyle(**opts["_format"]),
                       digits_after_comma=opts["digits_after_comma"])
            elif self._plot_type == HISTOGRAM:
                fmt = dict(opts["_format"])
                if fmt.get("title") is None:
                    fmt["title"] = name  # default the title to the variable name
                dv.plotting.HistogramPlot(
                    series.dropna(), method="n_bins", n_bins=opts["n_bins"],
                ).plot(
                    ax=ax, format_style=dv.plotting.FormatStyle(**fmt),
                    highlight_peak=opts["highlight_peak"],
                    show_zscores=opts["show_zscores"],
                    show_zscore_values=opts["show_zscore_values"],
                    show_info=opts["show_info"], show_counts=opts["show_counts"],
                    show_title=opts["show_title"],
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

        Hexbin/scatter number their role picks (1=X, 2=Y, 3=Z); other plot types
        number their panels left-to-right / top-to-bottom.
        """
        marks = self._xyz if self._plot_type in _XYZ_TYPES else self._panels
        self.varpanel.set_panels(marks)
