"""
GUI.TABS.PLOTTING: INTERACTIVE PLOTTING TAB
===========================================

Three-column plotting tab, one closable tab per plot method (`PlottingTab(plot_type,
title)`, opened from the Plot menu): a list of variables on the left, a live
plot-settings panel in the middle, the plot area on the right. When data arrive,
the variable list is populated from its columns. The selected variables and
their settings are kept for every column still in the data. Only when none is
left (or on the first load) is a default variable (`_DEFAULT_VAR`, or seeded
roles for the role-picked types) rendered. Editing
a setting does not re-render on its own. The user clicks the **Update plot**
button to apply all pending parameter changes at once (see `plot_settings.py`).
Selecting a variable in the list still re-renders immediately.

Selection model (heatmaps, time series, diel cycle, cumulative types):
- Plain click  -> reset to a single panel showing the clicked variable.
- Ctrl + click -> add or remove a panel (up to `_MAX_PANELS`). Heatmaps go
  side by side with shared axes, the other types stack with a shared x-axis.
  With more than one panel, a row of pills picks which panel's settings the
  controls edit.

Single-variable types (ridgeline, histogram, tree ring, shifted distribution,
waterfall) show only the clicked variable. Role-picked types
(`_ROLE_DROPDOWN_TYPES`) assign X/Y/Z through dropdowns instead of the list.

diive's plot classes use the two-phase `__init__(data)` / `plot(ax=...)`
pattern, so each panel is one library plot call into its own axes.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import weakref

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.core.plotting.plotfuncs import decimate_line
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.flow_layout import FlowLayout
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.plot_settings import (
    CUMULATIVE,
    CUMULATIVE_YEAR,
    DIELCYCLE,
    HEATMAP,
    HEATMAP_XYZ,
    HEATMAP_YEARMONTH,
    HEXBIN,
    HISTOGRAM,
    RIDGELINE,
    SCATTER,
    SHIFTEDDIST,
    TIMESERIES,
    TREERING,
    WATERFALL,
    WINDROSE,
    PlotSettingsPanel,
)

#: Plot types that pick variables by role (not comparison panels): hexbin and
#: x/y/z heatmap X/Y/Z, scatter X/Y/Z, and the wind rose value/wind-dir/colour.
_XYZ_TYPES = (HEXBIN, HEATMAP_XYZ, SCATTER, WINDROSE)

#: Role-picked types whose roles are assigned via X/Y/Z dropdowns (drag a variable
#: onto a field or pick it from the complete list) rather than clicking the list in
#: order. This is the standard role-selection method for every X/Y/Z plot type.
_ROLE_DROPDOWN_TYPES = (SCATTER, WINDROSE, HEXBIN, HEATMAP_XYZ)
from diive.gui.widgets.variable_panel import VariablePanel, lock_panel_handle
from diive.gui.widgets.weak_slot import weak_slot

#: Plot types laid out like a heatmap (panels side by side, shared axes).
_HEATMAP_TYPES = (HEATMAP, HEATMAP_YEARMONTH)

#: Plot types that support several variables at once -- one subplot per variable
#: (with its own settings sub-tab), added via Ctrl+click. Heatmaps tile side by
#: side; time series and diel cycles stack with a shared x-axis.
_MULTI_PANEL_TYPES = _HEATMAP_TYPES + (TIMESERIES, DIELCYCLE)

#: Column selected on startup -- gap-filled (continuous) NEE from the bundled
#: CH-DAV example dataset.
_DEFAULT_VAR = "NEE_CUT_REF_f"

#: Maximum number of side-by-side panels (further Ctrl+clicks are ignored).
_MAX_PANELS = 5

#: A time-series line is drawn thinned (`decimate_line`) to this many columns
#: per pixel of figure width. The line here is thicker than in the Overview
#: (2.2 pt), and at two columns per pixel the dense stretches still showed thin
#: light streaks. At four the thinned line differs from the full one about as
#: much as matplotlib's own path simplification differs from an unsimplified
#: line, and draws about a quarter faster (ten years of half-hourly NEE).
_TS_COLUMNS_PER_PIXEL = 4


class _ShowWatcher(QObject):
    """Call a tab method soon after a widget is shown.

    The call waits for the event loop: layout changes that showing the widget
    sets off (a new tab grows the tab bar) are posted events, and only after
    them does the widget have its final size. Holds the method weakly (like
    `weak_slot`) and is parented to the widget, so it neither keeps the tab
    alive nor outlives the widget.
    """

    def __init__(self, widget: QWidget, method) -> None:
        super().__init__(widget)
        self._method = weakref.WeakMethod(method)
        widget.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Show:
            QTimer.singleShot(0, self._call)
        return False

    def _call(self) -> None:
        method = self._method()
        if method is not None:
            method()


def _ranges_overlap(a, b) -> bool:
    """True if the closed intervals `a` and `b` overlap (order-insensitive)."""
    a0, a1 = sorted(a)
    b0, b1 = sorted(b)
    return a0 <= b1 and b0 <= a1

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
        # Time-series lines drawn thinned: (line, full x, full y) per line.
        self._thinned: list = []
        # A render asked for while the tab is hidden: None, or its preserve_view.
        self._pending_render: bool | None = None

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Title-bar action: a "Copy Python" button, right-aligned like the
        # gap-filling tabs. Every plot type has a codegen (see `_python_code`);
        # the button is a no-op while the current role picks are incomplete
        # (the provider returns None).
        self.copy_btn = CopyPythonButton(self._python_code)
        title_trailing = [self.copy_btn]

        # Main header (tracked/bold, e.g. "TIME SERIES PLOT"), matching the
        # correction/gap-filling tabs' title bar.
        outer.addLayout(build_titlebar(f"{self.title} plot", *title_trailing))

        # "Update plot" button, left-aligned in its own row just below the header
        # (mirrors the gap-filling tabs' Run button). The plot updates ONLY on a
        # click: editing a control or changing a scatter X/Y/Colour dropdown marks
        # it dirty (enables the button) but does not re-render. Disabled until
        # something changes, re-disabled after each render.
        self.update_btn = QPushButton("Update plot")
        theme.set_button_role(self.update_btn, "confirm")
        self.update_btn.setEnabled(False)
        # Keep the current pan/zoom when applying setting changes.
        self.update_btn.clicked.connect(weak_slot(self._render, True))
        action_row = QHBoxLayout()
        action_row.setContentsMargins(10, 0, 10, 4)
        action_row.addWidget(self.update_btn)
        action_row.addStretch(1)
        outer.addLayout(action_row)

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Shared variable list (filter + pills) under its own header. Plain click
        # resets to one panel; Ctrl+click toggles additional panels. Time-series
        # and every role-dropdown list (scatter, wind rose, hexbin, x/y/z heatmap)
        # are draggable so a variable can be dropped onto a colour-by / X / Y /
        # Z / Colour / value / wind-direction field.
        self.varpanel = VariablePanel(
            draggable=self._plot_type in (TIMESERIES,) + _ROLE_DROPDOWN_TYPES)
        self.varpanel.selected.connect(self._on_selected)
        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        # Header hint matches the tab's selection model: role-dropdown types assign
        # variables via dropdowns (drag onto a field); the multi-panel tabs add a
        # per-variable subplot (with its own settings sub-tab) on Ctrl+click; the
        # rest plot the clicked variable.
        if self._plot_type in _ROLE_DROPDOWN_TYPES:
            list_hint = "drag onto a field"
        elif self._plot_type in _MULTI_PANEL_TYPES:
            list_hint = "click to plot · Ctrl+click to add"
        else:
            list_hint = "click to plot"
        llay.addWidget(list_header("Variable", list_hint))
        llay.addWidget(self.varpanel, 1)

        # Middle: plot-parameter controls under a header. Editing a control does
        # NOT re-render; the Update plot button below the header applies pending
        # changes. Variable selection in the list still renders live.
        self.settings = PlotSettingsPanel(self._plot_type)
        # Any control edit marks the plot dirty -> enables the Update plot button.
        self.settings.changed.connect(self._mark_dirty)
        # Scatter / wind rose assign variables through the role dropdowns (pick or
        # drag onto them), not by clicking the list; a change marks the plot dirty
        # and waits for the Update plot button (no live re-render).
        if self._plot_type in _ROLE_DROPDOWN_TYPES:
            self.settings.xyz_changed.connect(self._on_xyz_changed)
        # Segmented pill row to pick which subplot's settings to edit (shown only
        # with more than one panel). Switching a pill swaps the loaded settings.
        self.panel_pills = _PanelPills()
        self.panel_pills.changed.connect(self._on_pill_changed)
        self.panel_pills.setVisible(False)
        middle = QWidget()
        mlay = QVBoxLayout(middle)
        mlay.setContentsMargins(0, 0, 0, 0)
        mlay.setSpacing(4)
        mlay.addWidget(list_header("Plot settings", "adjust & update"))
        mlay.addWidget(self.panel_pills)
        mlay.addWidget(self.settings, 1)

        # Right: embedded matplotlib canvas.
        self.canvas = MplCanvas()
        # The ridgeline builds its own overlapping gridspec; the wind rose builds
        # its own polar axes (+ colorbar) and sets its own margins. Keep the
        # canvas from re-flowing either (constrained layout / resize).
        if self._plot_type in (RIDGELINE, WINDROSE, TREERING, SHIFTEDDIST):
            self.canvas.auto_layout = False
        # A wider canvas needs more columns for the thinned time series.
        self.canvas.mpl_connect("resize_event", self._on_canvas_resize)
        # A render asked for while the tab is hidden runs when it is shown.
        self._show_watcher = _ShowWatcher(self.canvas, self._render_pending)

        splitter.addWidget(left)
        splitter.addWidget(middle)
        # The wind rose also surfaces its per-sector aggregation as a table (with
        # a copy-to-clipboard button) to the right of the canvas; every other plot
        # type puts the bare canvas on the right.
        if self._plot_type == WINDROSE:
            right = QSplitter(Qt.Orientation.Horizontal)
            right.addWidget(self.canvas)
            right.addWidget(self._build_windrose_results())
            right.setStretchFactor(0, 1)
            right.setStretchFactor(1, 0)
            right.setSizes([620, 300])
            splitter.addWidget(right)
        else:
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
        self._load_panel_state(self._panel_settings.get(self._active_panel))

    def _load_panel_state(self, snapshot) -> None:
        """Load a saved panel snapshot into the controls without marking the plot dirty.

        ``apply_state`` sets the controls with their signals live, so a snapshot
        that differs from what they showed fires ``settings.changed`` and
        ``_mark_dirty`` would enable Update plot although nothing is pending (L152).
        """
        blocked = self.settings.blockSignals(True)
        try:
            self.settings.apply_state(snapshot)
        finally:
            self.settings.blockSignals(blocked)

    def on_data_loaded(self, df, created: set | None = None) -> None:
        """Populate the variable list from the dataset and render.

        `created` marks user-engineered columns so they get the "NEW" pill.
        The current panels, roles and per-panel settings are kept for every
        variable still in `df` (L338), so adding a column elsewhere does not
        discard the plot setup. Only when none of the selected variables is
        left (or on the first load) does the tab fall back to its defaults.
        """
        self._df = df
        cols = [str(c) for c in df.columns]
        self.varpanel.set_variables(df.columns, created)
        if self._plot_type == CUMULATIVE_YEAR:
            # Offer the data's years in the highlight-year dropdown.
            self.settings.set_years(sorted(set(df.index.year)))
        if self._plot_type == SHIFTEDDIST:
            # Seed the reference/comparison periods from the data's year range.
            self.settings.set_periods(sorted(set(df.index.year)))
        if self._plot_type == TIMESERIES:
            # Offer every column as a colour-by variable.
            self.settings.set_colorby_options(df.columns)
        if self._plot_type in _ROLE_DROPDOWN_TYPES:
            # Offer every column in the role dropdowns (scatter X/Y/Colour,
            # wind-rose value/wind-direction/colour).
            self.settings.set_xyz_options(df.columns)
        if self._keep_selection(cols):
            self._render()
        else:
            self._select_default()

    def _keep_selection(self, cols: list[str]) -> bool:
        """Keep the current selection for the variables still in `cols`.

        Panels whose variable is gone are dropped with their settings. A role
        (X/Y/Z) whose variable is gone takes the tab's default for that role,
        or is left empty when it is optional or has no default. Returns False
        when nothing of the selection is left, so the caller seeds the defaults.
        """
        names = set(cols)
        if self._plot_type in _XYZ_TYPES:
            if not any(n in names for n in self._xyz):
                self._xyz = []
                return False
            seed = self._default_xyz(cols)
            # The colour role of scatter and wind rose is optional.
            optional = {2} if self._plot_type in (SCATTER, WINDROSE) else set()
            roles = []
            for i, n in enumerate(self._xyz):
                if n not in names:
                    n = seed[i] if i < len(seed) and i not in optional else None
                if n is None:
                    break  # later roles cannot sit behind an empty one
                roles.append(n)
            self._xyz = roles
            self.settings.set_xyz(*(roles + [None, None, None])[:3])
            return True

        self._capture_active()  # pending edits of the active panel count too
        panels = [n for n in self._panels if n in names]
        if not panels:
            self._panels = []
            self._panel_settings = {}
            self._active_panel = None
            return False
        # Load each kept snapshot once so data-dependent dropdowns (colour-by
        # variable, highlight year) fall back to their first entry when the
        # saved pick is gone, instead of keeping the previous panel's pick.
        kept = {}
        for n in panels:
            snap = self._panel_settings.get(n)
            if snap is None:
                continue
            for combo in (getattr(self.settings, "color_by", None),
                          getattr(self.settings, "cy_highlight", None)):
                if combo is not None:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)
                    combo.blockSignals(False)
            self._load_panel_state(snap)
            kept[n] = self.settings.state()
        active = self._active_panel if self._active_panel in panels else panels[0]
        self._panels = panels
        self._panel_settings = kept
        self._active_panel = None  # the controls hold the last loaded snapshot
        self._sync_panel_settings(active=active)
        return True

    def _default_xyz(self, cols: list[str]) -> list[str]:
        """Default roles of the role-picked types for the columns `cols`."""
        if self._plot_type in (HEXBIN, HEATMAP_XYZ):
            # Both need three variables; seed a sensible driver/driver/flux
            # triple so the tab shows something on open.
            preferred = ["Tair_f", "VPD_f", "NEE_CUT_REF_f"]
            return preferred if all(c in cols for c in preferred) else cols[:3]
        if self._plot_type == SCATTER:
            # Scatter needs X and Y (Z optional for colour); seed a driver/flux pair.
            preferred = ["Tair_f", "NEE_CUT_REF_f"]
            return preferred if all(c in cols for c in preferred) else cols[:2]
        # Wind rose: needs a value + a wind-direction column. Seed the value with
        # a flux (or the first column) and the direction with the first
        # wind-direction-named column, if any (the bundled CH-DAV example has
        # none, so the tab then shows a prompt to pick variables).
        wd = next((c for c in cols if any(t in c.lower()
                   for t in ("wind_dir", "winddir", "wd", "_dir"))), None)
        val = next((c for c in ("NEE_CUT_REF_f", "Tair_f") if c in cols),
                   cols[0] if cols else None)
        return [c for c in (val, wd) if c]

    def _select_default(self) -> None:
        """Highlight and render the startup variable(s)."""
        cols = [str(c) for c in self._df.columns]
        if self._plot_type in _XYZ_TYPES:
            self._xyz = self._default_xyz(cols)
            self.settings.set_xyz(*(self._xyz + [None, None, None])[:3])
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
        if self._plot_type in _ROLE_DROPDOWN_TYPES:
            # Restore the role picks straight into the dropdowns (list clicks no
            # longer assign roles for scatter / wind rose).
            self.settings.set_xyz_options(self.varpanel.names())
            self._xyz = sel
            self.settings.set_xyz(*(sel + [None, None, None])[:3])
        else:
            for i, name in enumerate(sel):
                # Replay the clicks (first resets, rest add) without their
                # queued renders: the render below draws the restored setup.
                self._on_selected(name, i > 0, render=False)
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

    def _on_selected(self, name: str, additive: bool, render: bool = True) -> None:
        """Apply a click on `name` in the variable list.

        ``render=False`` updates the selection only, for a caller that renders
        once itself afterwards (`restore_state`).
        """
        if not name:
            return
        if self._plot_type in _ROLE_DROPDOWN_TYPES:
            # Scatter / wind-rose roles are assigned via the dropdowns (pick or
            # drag a variable onto them); clicking the list does not assign a role.
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
            if render:
                self.varpanel.run_with_loading(name, self._render)
            return
        if self._plot_type in (RIDGELINE, HISTOGRAM, TREERING, SHIFTEDDIST,
                               WATERFALL):
            # The ridgeline and tree ring use the whole (polar) figure; the
            # histogram is information-dense (counts, z-score axis); the shifted
            # distribution compares two periods of one variable; the waterfall
            # packs a dense bar budget with an end-total annotation -> all
            # single-variable.
            self._panels = [name]
            self._sync_panel_settings(active=name)
            if render:
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
        if render:
            self.varpanel.run_with_loading(name, self._render)

    def _on_xyz_changed(self) -> None:
        """A scatter X/Y/Colour dropdown changed (pick or drag-drop): rebuild the
        role list and mark the plot dirty. The plot updates only when the user
        clicks Update plot (the list highlight refreshes immediately as feedback)."""
        self._xyz = self.settings.xyz_values()
        self._mark_selected()
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        """A setting changed since the last render: enable the Update plot button
        so the user can apply it (no auto-render)."""
        self.update_btn.setEnabled(True)

    def _python_code(self) -> str | None:
        """Runnable snippet reproducing the current plot (Copy Python button).

        Every plot type has a library codegen. Returns None when the current
        selection is incomplete (e.g. a hexbin/scatter/wind-rose role not yet
        picked, or no variable selected) — the Copy Python button is then a
        no-op. Multi-panel tabs (heatmap/time series/diel cycle) reproduce the
        active panel's variable as a single-figure snippet.
        """
        from diive.core.plotting import codegen
        opts = self.settings.values()
        pt = self._plot_type

        if pt == SCATTER:
            if len(self._xyz) < 2:
                return None
            from diive.core.plotting.scatter import scatter_to_code
            xn, yn = self._xyz[0], self._xyz[1]
            zn = self._xyz[2] if len(self._xyz) >= 3 else None
            return scatter_to_code(
                xn, yn, zn,
                nbins=opts["nbins"], binagg=opts["binagg"],
                cmap=opts["cmap"], show_colorbar=opts["show_colorbar"],
                markersize=opts["markersize"], alpha=opts["alpha"],
                vmin=opts["vmin"], vmax=opts["vmax"],
                format_kwargs=opts.get("_format"),
            )
        if pt == HEXBIN:
            if len(self._xyz) < 3:
                return None
            return codegen.hexbin_to_code(*self._xyz[:3], opts)
        if pt == HEATMAP_XYZ:
            if len(self._xyz) < 3:
                return None
            return codegen.heatmap_xyz_to_code(*self._xyz[:3], opts)
        if pt == WINDROSE:
            if len(self._xyz) < 2:
                return None
            zn = self._xyz[2] if len(self._xyz) >= 3 else None
            return codegen.windrose_to_code(self._xyz[0], self._xyz[1], zn, opts)

        # Single-/multi-panel variable plots: reproduce the active (or first) panel.
        name = self._active_panel or (self._panels[0] if self._panels else None)
        if not name:
            return None
        builder = {
            HEATMAP: codegen.heatmap_datetime_to_code,
            HEATMAP_YEARMONTH: codegen.heatmap_yearmonth_to_code,
            TIMESERIES: codegen.timeseries_to_code,
            DIELCYCLE: codegen.dielcycle_to_code,
            CUMULATIVE_YEAR: codegen.cumulative_year_to_code,
            CUMULATIVE: codegen.cumulative_to_code,
            HISTOGRAM: codegen.histogram_to_code,
            RIDGELINE: codegen.ridgeline_to_code,
            SHIFTEDDIST: codegen.shifted_distribution_to_code,
            TREERING: codegen.treering_to_code,
            WATERFALL: codegen.waterfall_to_code,
        }.get(pt)
        if builder is None:
            return None
        return builder(name, opts)

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

        While the tab is hidden in a shown window (e.g. a theme change re-renders
        every open Time series tab), the render waits until the tab is shown. A
        hidden canvas has a stale size, so the resize on showing would solve
        the layout again and render a second time.
        """
        # A render applies the current settings -> nothing pending to update.
        self.update_btn.setEnabled(False)
        if not self.canvas.isVisible() and self.canvas.window().isVisible():
            # Keep the view only if every render asked for while hidden did.
            pending = self._pending_render
            self._pending_render = preserve_view if pending is None \
                else (pending and preserve_view)
            return
        self._pending_render = None
        self._thinned = []
        if self._plot_type == HEXBIN:
            self._render_hexbin()
            return

        if self._plot_type == HEATMAP_XYZ:
            self._render_heatmap_xyz()
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

        if self._plot_type == TREERING:
            self._render_treering()
            return

        if self._plot_type == SHIFTEDDIST:
            self._render_shifted_distribution()
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
        for i, (ax, name) in enumerate(zip(axes, self._panels, strict=False)):
            if self._panel_settings.get(name) is not None:
                self._load_panel_state(self._panel_settings[name])
            self._draw_one(ax, name, i)
            self._apply_axes([ax])
            axo = self.settings.values().get("_axes") or {}
            explicit.append((
                axo.get("xmin") is not None or axo.get("xmax") is not None,
                axo.get("ymin") is not None or axo.get("ymax") is not None))
        if self._active_panel is not None and self._panel_settings.get(self._active_panel):
            self._load_panel_state(self._panel_settings[self._active_panel])

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
        self._thin_lines()
        self.canvas.draw()
        self.canvas.reset_history()

        # Re-apply the preserved zoom on top (Update plot keeps the view), but
        # never override a dim the user pinned with an explicit limit. Recorded
        # as a second history entry so Home still returns to the full view.
        #
        # Skip a dim whose preserved range no longer overlaps the new data range:
        # e.g. flipping a heatmap's orientation swaps the x-axis from hours (0-24)
        # to dates, so restoring the old hour limits would scroll the view far off
        # the data and show an empty plot.
        if restore is not None:
            changed = False
            for i, ax in enumerate(axes):
                if not explicit[i][0] and _ranges_overlap(ax.get_xlim(), restore[i][0]):
                    ax.set_xlim(restore[i][0])
                    changed = True
                if not explicit[i][1] and _ranges_overlap(ax.get_ylim(), restore[i][1]):
                    ax.set_ylim(restore[i][1])
                    changed = True
            if changed:
                self.canvas.draw_idle()
                self.canvas.push_view()

        self._last_axes = list(axes)
        self._mark_selected()

    def _render_pending(self) -> None:
        """Run the render that waited for the tab to be shown (see `_render`)."""
        if self._pending_render is not None:
            self._render(self._pending_render)

    def _keep_thinned(self, ax, line, n_records: int) -> None:
        """Draw a long time-series line thinned to the screen columns.

        At canvas size most of a long record overlaps: 175k half-hourly records
        cost about 0.3 s per draw. `decimate_line` keeps each column's first,
        last, lowest and highest value, so the line looks the same and no spike
        is lost (see `_TS_COLUMNS_PER_PIXEL`). The full record stays on the
        line for the hover (``_diive_hover_xy``), and the line is thinned again
        for every new
        x-range (zoom, pan, Home) and canvas size. Lines with markers are drawn
        in full, because a thinned line would drop markers.
        """
        x = np.asarray(line.get_xdata(orig=False), dtype=float)
        y = np.asarray(line.get_ydata(orig=False), dtype=float)
        # Thinning needs ascending x, which a sanitized time index has.
        if x.size != n_records or x.size < 2 or not np.all(np.diff(x) > 0):
            return
        line._diive_hover_xy = (x, y)
        self._thinned.append((line, x, y))
        ax.callbacks.connect("xlim_changed", self._on_ts_xlim)

    def _thin_lines(self, ax=None) -> None:
        """Point the thinned lines (of `ax`, or all) at the samples their view needs."""
        if not self._thinned:
            return
        n_columns = int(self.canvas.fig.bbox.width * _TS_COLUMNS_PER_PIXEL)
        for line, x, y in self._thinned:
            if line.axes is None or (ax is not None and line.axes is not ax):
                continue
            lo, hi = sorted(line.axes.get_xlim())
            line.set_data(*decimate_line(x, y, lo, hi, n_columns))

    def _on_ts_xlim(self, ax) -> None:
        self._thin_lines(ax)

    def _on_canvas_resize(self, _event) -> None:
        self._thin_lines()

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

    def _render_treering(self) -> None:
        """Render the tree ring: one variable as concentric annual rings.

        Single-variable and polar: like the wind rose it builds its own polar
        axes + colorbar on the canvas figure (the tab leaves the layout to it,
        `canvas.auto_layout` is False). The render style picks the filled colour
        mesh (`plot`) or the radial line traces (`plot_line`).
        """
        fig = self.canvas.fig
        fig.clear()
        fig.set_layout_engine("none")
        if not self._panels:
            self.canvas.draw()
            self._mark_selected()
            return
        name = self._panels[0]
        opts = self.settings.values()
        ax = fig.add_subplot(111, projection="polar")
        try:
            tr = dv.plotting.TreeRingPlot(
                df=self._df[[name]], value_col=name,
                resample_freq=opts["resample_freq"],
            )
            fmt = dict(opts["_format"])
            if fmt.get("title") is None:
                fmt["title"] = name  # default the title to the variable name
            common = dict(
                ax=ax, format_style=dv.plotting.FormatStyle(**fmt),
                cmap=opts["cmap"], vmin=opts["vmin"], vmax=opts["vmax"],
                show_month_labels=opts["show_month_labels"],
                show_month_lines=opts["show_month_lines"],
                show_year_labels=opts["show_year_labels"],
                show_year_separators=opts["show_year_separators"],
                year_label_frequency=opts["year_label_frequency"],
                cb_label=opts["cb_label"],
                cb_digits_after_comma=opts["cb_digits_after_comma"],
                cb_labelsize=opts["cb_labelsize"],
            )
            if opts["style"] == "line":
                tr.plot_line(
                    linewidth=opts["linewidth"], alpha=opts["alpha"],
                    amplitude_scale=opts["amplitude_scale"],
                    ring_width=opts["ring_width"], **common,
                )
            else:
                tr.plot(**common)
        except Exception as err:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, f"Cannot plot '{name}':\n{err}", ha="center",
                    va="center", wrap=True, transform=ax.transAxes)
        self.canvas.draw()
        self.canvas.reset_history()
        self._mark_selected()

    def _render_shifted_distribution(self) -> None:
        """Render the shifted distribution: one variable's density compared
        between a reference and a comparison period.

        Single-variable, single-axis. The plot draws its zone labels just above
        the top spine and a padded left-aligned title, so the canvas manages its
        own margins (`canvas.auto_layout` is False for this tab) with extra
        headroom rather than letting constrained layout clip them.
        """
        fig = self.canvas.fig
        fig.clear()
        fig.set_layout_engine("none")
        # Place the single axes with an explicit rect (left, bottom, width,
        # height) rather than subplots_adjust: the "none" layout engine is a
        # placeholder that rejects subplots_adjust, but honours add_axes. The top
        # edge sits at 0.82 to leave headroom for the zone labels (drawn just
        # above the top spine) and the padded left-aligned title.
        ax = fig.add_axes((0.08, 0.12, 0.89, 0.70))
        if not self._panels:
            self.canvas.draw()
            self.canvas.reset_history()
            self._mark_selected()
            return
        name = self._panels[0]
        opts = self.settings.values()
        try:
            dv.plotting.ShiftedDistributionPlot(
                series=self._df[name],
                ref_period=opts["ref_period"], comp_period=opts["comp_period"],
            ).plot(
                ax=ax, format_style=dv.plotting.FormatStyle(**opts["_format"]),
                ref_label=opts["ref_label"], comp_label=opts["comp_label"],
                zone_labels=opts["zone_labels"],
                show_legend=opts["show_legend"], show_title=opts["show_title"],
                show_xaxis=opts["show_xaxis"], show_yaxis=opts["show_yaxis"],
            )
        except Exception as err:
            ax.clear()
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

    def _render_heatmap_xyz(self) -> None:
        """Render the x/y/z heatmap (single figure): z aggregated into a 2D grid
        of x/y bins.

        Needs all three roles set. ``HeatmapXYZ`` requires pre-aggregated input
        (one z value per unique (x, y) bin), so the raw variables are first binned
        and aggregated through ``GridAggregator`` (rows with a missing x or y are
        dropped jointly; z may keep NaNs — ignored during aggregation).
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
            agg = dv.analysis.GridAggregator(
                x=sub[xn], y=sub[yn], z=sub[zn],
                binning_type=opts["binning_type"], n_bins=opts["n_bins"],
                aggfunc=opts["aggfunc"],
                min_n_vals_per_bin=opts["min_n_vals_per_bin"],
            )
            dv.plotting.HeatmapXYZ.from_gridaggregator(agg, xn, yn, zn).plot(
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
            ax.text(0.5, 0.5, f"Cannot plot heatmap:\n{err}", ha="center",
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
            # Pass each role as its own Series (ScatterXY aligns/drops NaNs and
            # tolerates the same variable used in two roles, e.g. colour-by-X).
            dv.plotting.ScatterXY(
                x=self._df[xn], y=self._df[yn],
                z=(self._df[zn] if zn else None),
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

    def _build_windrose_results(self) -> QWidget:
        """The wind rose's per-sector results table + a copy-to-clipboard button."""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)

        head = QHBoxLayout()
        head.addWidget(list_header("Per-sector results", "aggregation table"))
        head.addStretch(1)
        self.wr_copy_btn = QPushButton("Copy table")
        self.wr_copy_btn.setToolTip("Copy the per-sector table to the clipboard "
                                    "(tab-separated, pastes into Excel).")
        self.wr_copy_btn.clicked.connect(self._copy_windrose_table)
        self.wr_copy_btn.setEnabled(False)
        head.addWidget(self.wr_copy_btn)
        lay.addLayout(head)

        self.wr_table = QTableWidget()
        self.wr_table.verticalHeader().setVisible(False)
        self.wr_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.wr_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        lay.addWidget(self.wr_table, 1)
        self._wr_results = None  # the last rendered .results DataFrame
        return panel

    def _fill_windrose_table(self, results) -> None:
        """Populate the results table from a WindRosePlot `.results` DataFrame."""
        self._wr_results = results
        cols = ["Sector"] + list(results.columns)
        self.wr_table.setColumnCount(len(cols))
        self.wr_table.setHorizontalHeaderLabels(cols)
        self.wr_table.setRowCount(len(results))
        for r, (sector, row) in enumerate(results.iterrows()):
            cells = [str(sector)] + [
                ("" if v != v else  # NaN
                 (f"{int(v)}" if c in ("N_VALS",) else f"{v:.3g}"))
                for c, v in row.items()
            ]
            for cidx, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if cidx > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                self.wr_table.setItem(r, cidx, item)
        hdr = self.wr_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for c in range(1, len(cols)):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        self.wr_copy_btn.setEnabled(True)

    def _copy_windrose_table(self) -> None:
        """Copy the per-sector results to the clipboard as tab-separated text."""
        if self._wr_results is None:
            return
        # index_label keeps the sector column header; \t pastes cleanly into Excel.
        text = self._wr_results.to_csv(sep="\t", index_label="Sector")
        QApplication.clipboard().setText(text)
        self.wr_copy_btn.setText("Copied ✓")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1200, lambda: self.wr_copy_btn.setText("Copy table"))

    def _render_windrose(self) -> None:
        """Render the wind rose: a variable aggregated into wind-direction sectors.

        Role-picked (1 = value, 2 = wind direction, 3 = optional colour variable);
        value + direction are required, the colour variable is optional. The plot
        is polar with its own colorbar, so — like the ridgeline — it builds its own
        axes on the canvas figure (the tab leaves the layout to it). The per-sector
        `.results` table is mirrored into the side table and reported to the Log
        (Rich console), since both come from the same library object.
        """
        self.settings.set_xyz(*(self._xyz + [None, None, None])[:3])
        fig = self.canvas.fig
        fig.clear()
        fig.set_layout_engine("none")
        if len(self._xyz) < 2:
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5,
                    f"Pick a value and a wind-direction variable  "
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
            rose = dv.plotting.WindRosePlot(
                series=self._df[valn], wind_dir=self._df[wdn],
                agg=opts["agg"], n_sectors=opts["n_sectors"],
                z=(self._df[zn] if zn else None), z_agg=opts["z_agg"],
            )
            rose.plot(
                ax=ax, format_style=dv.plotting.FormatStyle(**opts["_format"]),
                cmap=opts["cmap"], color=opts["color"],
                vmin=opts["vmin"], vmax=opts["vmax"],
                show_colorbar=opts["show_colorbar"], cb_label=opts["cb_label"],
                cb_digits_after_comma=opts["cb_digits_after_comma"],
                max_sector_labels=opts["max_sector_labels"],
            )
            self._fill_windrose_table(rose.results)
            # Rich per-sector table + summary lines -> streamed to the Log tab.
            # Bump verbosity so report()'s info() summary lines also print (the
            # table prints regardless); set after construction to avoid a second
            # report during __init__.
            rose.verbose = 2
            rose.report()
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
                    # One image instead of a quad per cell: the same picture.
                    # On narrower panels (several heatmaps side by side) it
                    # draws and pans about twice as fast as the mesh.
                    as_image=True,
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
                # Both axes are plain integers (month 1-12, calendar year). Tag
                # them so the hover tooltip labels them as Month/Year instead of
                # misreading the 1-12 month axis as a clock time (05:30).
                if opts["ax_orientation"] == "vertical":
                    ax.xaxis._diive_hover_intlabel = "Month"
                    ax.yaxis._diive_hover_intlabel = "Year"
                else:
                    ax.xaxis._diive_hover_intlabel = "Year"
                    ax.yaxis._diive_hover_intlabel = "Month"
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
                n_lines = len(ax.lines)
                ts = dv.plotting.TimeSeries(
                    series, drop_gaps=opts["drop_gaps"], color_series=color_series)
                ts.plot(
                    ax=ax, format_style=dv.plotting.FormatStyle(**fmt),
                    color=color,
                    linewidth=opts["linewidth"], alpha=opts["alpha"],
                    marker=opts["marker"], markersize=opts["markersize"],
                    cmap=opts["color_by_cmap"], color_label=cby,
                )
                # The plain line is the first line the plot adds (the colour-by
                # path draws a LineCollection, so the record-length check in
                # _keep_thinned skips a zero line added instead).
                if not opts["marker"] and len(ax.lines) > n_lines:
                    self._keep_thinned(ax, ax.lines[n_lines], len(ts.series))
            elif self._plot_type == DIELCYCLE:
                ts_colors = theme.manager.ts_colors
                # DielCycle auto-colors per month only when color is None; pass
                # None for the per-month view so each month gets its own colour,
                # otherwise the theme line colour for this panel.
                color = None if opts["each_month"] else ts_colors[index % len(ts_colors)]
                fmt = dict(opts["_format"])
                if fmt.get("title") is None:
                    fmt["title"] = name  # default the title to the variable name
                # All stacked panels share the same months, so draw the legend on
                # the first panel only. Auto column count keeps it compact (and
                # stops a tall 1-column legend from collapsing the layout): 3
                # columns for >8 months, 2 for >4, else 1.
                n_months = (series.dropna().index.month.nunique()
                            if opts["each_month"] else 1)
                fmt["legend_ncol"] = 3 if n_months > 8 else (2 if n_months > 4 else 1)
                fmt["show_legend"] = bool(fmt.get("show_legend", True)) and index == 0
                dv.plotting.DielCycle(series).plot(
                    ax=ax, format_style=dv.plotting.FormatStyle(**fmt), color=color,
                    agg=opts["agg"], band=opts["band"], each_month=opts["each_month"],
                    cmap=opts["cmap"], marker=opts["marker"],
                    markersize=opts["markersize"],
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
            elif self._plot_type == CUMULATIVE:
                # Plain running total across the whole record (one curve). The
                # class takes a DataFrame (one curve per column); a single-column
                # frame plots this variable's cumulative sum.
                dv.plotting.Cumulative(
                    df=series.to_frame(), units=opts["units"],
                ).plot(ax=ax, showplot=False,
                       format_style=dv.plotting.FormatStyle(**opts["_format"]),
                       digits_after_comma=opts["digits_after_comma"],
                       show_title=opts["show_title"], fill=opts["fill"])
            elif self._plot_type == WATERFALL:
                # Cumulative budget: contributions aggregated to one floating bar
                # per period (resample/agg), building up a running total. Keeps
                # its own auto title unless the user sets one.
                dv.plotting.WaterfallPlot(
                    series, series_units=opts["series_units"],
                    resample=opts["resample"], agg=opts["agg"],
                    uptake_is_negative=opts["uptake_is_negative"],
                ).plot(
                    ax=ax, showplot=False,
                    format_style=dv.plotting.FormatStyle(**opts["_format"]),
                    digits_after_comma=opts["digits_after_comma"],
                    color_uptake=opts["color_uptake"],
                    color_release=opts["color_release"],
                    bar_width=opts["bar_width"],
                    show_connectors=opts["show_connectors"],
                )
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
