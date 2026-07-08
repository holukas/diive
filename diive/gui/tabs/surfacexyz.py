"""
GUI.TABS.SURFACEXYZ: 3-D X / Y / Z COORDINATE SURFACE
=====================================================

The coordinate-surface sibling of the date x time-of-day :class:`Surface3DTab`.
Instead of a variable's calendar grid, this renders an arbitrary **Z over X-Y**:
pick three variables (drag one from the list onto the X / Y / Z field, or click
to fill X -> Y -> Z in turn), and the scattered points are gridded onto a regular
X-Y mesh and shown as the same rotatable relief.

Almost everything is inherited from :class:`Surface3DTab` -- the relief controls,
camera presets, orbit/flyover animations, glTF/STL export, and the whole mesh /
render pipeline. This variant overrides only the data source: the gridding is the
library's :class:`~diive.analysis.GridAggregator` (equal-width X/Y bins, an
aggregation of Z per cell) instead of ``datetime_surface_grid`` (strict
GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QComboBox, QSpinBox

import diive as dv
from diive.gui.tabs.surface3d import Surface3DTab
from diive.gui.widgets.column_picker import NONE_ITEM, ColumnPicker

#: Aggregators offered for combining the Z values that fall in each X/Y cell
#: (a subset of GridAggregator's; 'count' is dropped as it ignores Z).
_XYZ_AGGS = ("mean", "median", "max", "min", "sum")


class SurfaceXYZTab(Surface3DTab):
    """3-D relief of Z gridded over two chosen X and Y variables."""

    title = "3D surface (X/Y/Z)"
    #: The left list is a drag palette for the X/Y/Z fields (drag a name onto a
    #: field). Enable dragging on the shared explorer list.
    list_title = "Variables"
    list_hint = "drag onto X / Y / Z"
    list_draggable = True
    _controls_hint = "pick X / Y / Z"
    #: Sparse grids: don't drape walls into empty bins -- keep them truly empty.
    _drop_gap_risers = True
    default_var = None

    # --- data-source controls (override the date/time hooks) -----------
    def _build_top_controls(self, lay) -> None:
        # X/Y/Z picker above the relief form. DropComboBox fields accept a
        # variable dragged from the list; `changed` re-renders.
        self.picker = ColumnPicker(
            [{"key": "x", "label": "X", "needle": "x",
              "tip": "Variable on the X axis."},
             {"key": "y", "label": "Y", "needle": "y",
              "tip": "Variable on the Y axis."},
             {"key": "z", "label": "Z (height / colour)", "needle": "z",
              "tip": "Variable gridded over X-Y as the relief height and colour."}],
            title="Coordinates")
        self.picker.changed.connect(self._rerender_view)
        lay.addWidget(self.picker)

    def _add_data_rows(self, form) -> None:
        self.nbins = QSpinBox()
        self.nbins.setRange(3, 200)
        self.nbins.setValue(30)
        self.nbins.setToolTip("Number of equal-width X and Y bins the scattered "
                              "points are gridded into (finer = more, smaller "
                              "cells).")
        self.nbins.valueChanged.connect(self._rerender_view)
        form.addRow("Bins (X/Y)", self.nbins)

        self.agg = QComboBox()
        self.agg.addItems(list(_XYZ_AGGS))
        self.agg.setToolTip("How the Z values that fall in each X/Y cell are "
                            "combined into the cell's height/colour.")
        self.agg.currentTextChanged.connect(self._rerender_view)
        form.addRow("Z aggregator", self.agg)

    # --- selection / data flow -----------------------------------------
    def _rerender_view(self, *_a) -> None:
        # No single "_target" here; render whenever the canvas exists.
        if getattr(self, "canvas", None) is not None:
            self._compute()

    def on_data_loaded(self, df, created=None) -> None:
        self._df = df
        self.varpanel.set_variables(df.columns, created)
        self.picker.seed(df.columns)
        # Seed X/Y/Z to three distinct numeric columns for a sensible first view.
        numeric = [str(c) for c in df.select_dtypes(include="number").columns]
        combos = self.picker.combos()
        for key, col in zip(("x", "y", "z"), numeric[:3]):
            combo = combos[key]
            combo.blockSignals(True)
            combo.setCurrentText(col)
            combo.blockSignals(False)
        self.picker.refresh_availability()
        self.varpanel.run_with_loading("surface", self._compute)

    def _on_select(self, name: str, _additive: bool = False) -> None:
        # Drag-only: dragging a variable onto an X/Y/Z field is the way to assign
        # roles (clicking the list does nothing, so a stray click can't reshuffle
        # the picked coordinates).
        return

    # --- gridding (library GridAggregator) -----------------------------
    def _grid_data(self):
        if self._df is None:
            return None
        picks = self.picker.picks()
        x, y, z = picks.get("x"), picks.get("y"), picks.get("z")
        cols = set(self._df.columns)
        if not all(v and v != NONE_ITEM and v in cols for v in (x, y, z)):
            return None
        try:
            agg = dv.analysis.GridAggregator(
                self._df[x], self._df[y], self._df[z],
                binning_type="equal_width", n_bins=self.nbins.value(),
                min_n_vals_per_bin=1, aggfunc=self.agg.currentText())
            wide = agg.df_agg_wide
        except Exception:
            return None  # too few distinct values, all-NaN, etc.
        if wide.empty:
            return None
        # Columns are the X bin midpoints, index the Y bin midpoints; values the
        # aggregated Z as (n_y, n_x) -- the same (rows=Y, cols=X) layout the
        # shared renderer expects. Reframe when any variable changes (not on a
        # bins/aggregator tweak), so frame_key is the (x, y, z) trio.
        x_vals = wide.columns.to_numpy(dtype=float)
        y_vals = wide.index.to_numpy(dtype=float)
        z_grid = wide.to_numpy(dtype=float)
        return x_vals, y_vals, z_grid, (x, y, z), z

    # --- codegen -------------------------------------------------------
    def _python_code(self) -> str | None:
        if self._df is None:
            return None
        picks = self.picker.picks()
        x, y, z = picks.get("x"), picks.get("y"), picks.get("z")
        if not all(v and v != NONE_ITEM and v in self._df.columns
                   for v in (x, y, z)):
            return None
        from diive.core.plotting.codegen import surface_xyz_to_code
        return surface_xyz_to_code(x, y, z, n_bins=self.nbins.value(),
                                   aggfunc=self.agg.currentText(),
                                   cmap=self.cmap.currentText())

    # --- state (own controls; the base saves the date/time ones) -------
    def _state_widgets(self) -> dict:
        widgets = {"style": self.style, "cmap": self.cmap, "exag": self.exag,
                   "opacity": self.opacity, "ystretch": self.ystretch,
                   "nbins": self.nbins, "agg": self.agg,
                   "shadows": self.shadows, "shadow_len": self.shadow_len,
                   "smooth": self.smooth, "edges": self.edges,
                   "smoothing": self.smoothing, "orbit_speed": self.orbit_speed}
        widgets.update({f"role_{k}": c for k, c in self.picker.combos().items()})
        return widgets

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._state_widgets())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._state_widgets(), state.get("controls") or state)
        self._sync_style_enabled()
        self._sync_shadows_enabled()
        if self._df is not None:
            self.picker.refresh_availability()
            self._rerender_view()
