"""
GUI.TABS.SURFACE3D: 3-D DATE x TIME-OF-DAY RELIEF SURFACE
========================================================

"What does a year of half-hourly data look like as a landscape?" Pick a
variable and see its date x time-of-day grid rendered as a smooth, rotatable
3-D relief — the GPU-accelerated 3-D analogue of the date/time heatmap. The
bright diel band, seasonal swells, and gaps become hills and valleys you can
orbit around.

Strict GUI<->library separation: the numeric grid is the library's
`dv.plotting.datetime_surface_grid` (sanitize + pivot — domain logic); this tab
only builds a PyVista mesh from those arrays and styles the scene (camera,
colormap, lighting). PyVista/VTK is the optional ``gui3d`` extra; without it the
tab shows install instructions instead of failing.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import warnings

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui.tabs._explorer_base import SingleVariableExplorerTab
from diive.gui.widgets.pyvista_canvas import (
    INSTALL_HINT,
    Pyvista3DCanvas,
    pyvista_available,
)
from diive.gui.widgets.tab_chrome import list_header

#: A continuous flux with a strong diel cycle makes the relief instantly legible.
_DEFAULT_VAR = "NEE_CUT_REF_f"

#: Keep rendering snappy: stride the date rows when the grid gets very large.
_MAX_ROWS = 2000

#: Default broadening of the date (Y) base relative to the time-of-day (X) base
#: so the relief reads like a wide landscape instead of a thin ridge. Perspective
#: foreshortens the depth axis, so a square base looks pinched in Y; stretching it
#: counters that. User-adjustable via the "Y stretch" control.
_Y_ASPECT = 3.0


def _unit(a: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1] (flat array -> all zeros)."""
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


#: NaN-aware row aggregators for binning the date axis. ``mean``/``median``
#: broaden *and* lower spikes; ``max`` keeps peaks tall with a wider base
#: (``min`` is the symmetric choice for deep troughs).
_AGGS = {"mean": np.nanmean, "max": np.nanmax,
         "median": np.nanmedian, "min": np.nanmin}


def _bin_rows(z: np.ndarray, y_days: np.ndarray, n: int, agg: str = "mean"):
    """Aggregate every ``n`` consecutive date rows into one wider cell.

    Coarsens the date (Y) resolution so single-day spikes spread across the bin
    and read as broad relief instead of sharp 1-cell points. ``agg`` selects how
    the rows combine (see ``_AGGS``). NaN-aware (gap cells stay NaN); a trailing
    partial bin is padded with NaN. Row positions (``y_days``) always use the
    mean so cells stay evenly spaced regardless of the value aggregator.
    """
    if n <= 1:
        return z, y_days
    fn = _AGGS.get(agg, np.nanmean)
    d, t = z.shape
    n_bins = int(np.ceil(d / n))
    pad = n_bins * n - d
    if pad:
        z = np.vstack([z, np.full((pad, t), np.nan)])
        y_days = np.concatenate([y_days, np.full(pad, np.nan)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bins -> NaN
        zb = fn(z.reshape(n_bins, n, t), axis=1)
        yb = np.nanmean(y_days.reshape(n_bins, n), axis=1)
    return zb, yb


#: Render styles: an extruded heatmap (one flat bar per cell, like the 2-D
#: heatmap given height) vs. a smoothly interpolated relief surface.
_STYLE_EXTRUDED = "Extruded heatmap"
_STYLE_SURFACE = "Smooth surface"


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell-boundary coordinates (length N+1) from N evenly-spaced cell centres."""
    c = np.asarray(centers, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mids = (c[:-1] + c[1:]) / 2.0
    return np.concatenate(([2 * c[0] - mids[0]], mids, [2 * c[-1] - mids[-1]]))


def _extruded_grid(xn: np.ndarray, yn: np.ndarray,
                   height: np.ndarray, scalars: np.ndarray):
    """Doubled-coordinate ("staircase") arrays so each cell renders flat.

    A smooth ``StructuredGrid`` puts one height per *point*, so neighbouring
    cells share corners and the surface ramps between them (the diagonal
    "peaks"). Here each cell instead gets its own flat top: cell-edge
    coordinates are duplicated (``[edge0, edge1, edge1, edge2, ...]``) so every
    cell spans a full 2x2 patch at constant height, and the zero-width gap at
    each shared edge becomes a vertical wall — an extruded heatmap. Returns
    ``(xx, yy, hh, ss)`` each shaped ``(2D, 2T)``.
    """
    xe = _cell_edges(xn)
    ye = _cell_edges(yn)
    x2 = np.empty(2 * xn.size)
    x2[0::2], x2[1::2] = xe[:-1], xe[1:]
    y2 = np.empty(2 * yn.size)
    y2[0::2], y2[1::2] = ye[:-1], ye[1:]
    xx, yy = np.meshgrid(x2, y2)
    hh = np.repeat(np.repeat(height, 2, axis=0), 2, axis=1)
    ss = np.repeat(np.repeat(scalars, 2, axis=0), 2, axis=1)
    return xx, yy, hh, ss


class Surface3DTab(SingleVariableExplorerTab):
    """Rotatable 3-D relief surface of a variable's date x time-of-day grid."""

    title = "3D surface"
    #: A continuous flux with a strong diel cycle makes the relief instantly legible.
    default_var = _DEFAULT_VAR
    #: Header above the variable list (matching the outlier / correction tabs).
    list_title = "Variable"
    list_hint = "click to render"

    def _init_state(self) -> None:
        # Variable whose surface the camera was last framed for; lets a settings
        # change re-render in place (keep the user's view) while a fresh variable
        # selection snaps back to the default 45° framing.
        self._framed_target: str | None = None

    def _build_right(self) -> QWidget:
        # Three-column tab (like the outlier / correction tabs): the variable
        # list (base), a vertical controls column, then the canvas.
        right = QWidget()
        rl = QHBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_controls())

        if pyvista_available():
            self.canvas = Pyvista3DCanvas()
            rl.addWidget(self.canvas, stretch=1)
        else:
            self.canvas = None
            rl.addWidget(self._build_missing_notice(), stretch=1)
        return right

    # --- sub-widgets ---------------------------------------------------
    def _build_missing_notice(self) -> QWidget:
        msg = QLabel(
            "<b>3-D plotting needs the optional <code>gui3d</code> extra.</b><br><br>"
            "Install it, then reopen this tab:<br>"
            f"<code>{INSTALL_HINT}</code>")
        msg.setWordWrap(True)
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("QLabel { padding: 24px; color: #37474F; }")
        return msg

    def _build_controls(self) -> QWidget:
        col = QWidget()
        col.setFixedWidth(250)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        lay.addWidget(list_header("Controls", "adjust the relief"))

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        # Default to the extruded heatmap (flat bar per cell, no interpolated
        # peaks); signal connected at the end so toggling can safely reference
        # the smooth-shading / smooth-terrain widgets built below.
        self.style = QComboBox()
        self.style.addItems([_STYLE_EXTRUDED, _STYLE_SURFACE])
        self.style.setToolTip(
            "Extruded heatmap = one flat rectangle per cell with vertical walls "
            "(stepped, like the 2-D heatmap raised into 3-D). Smooth surface = "
            "interpolated relief (rolling hills, but with sharp peaks).")
        form.addRow("Style", self.style)

        self.cmap = QComboBox()
        self.cmap.addItems(["Spectral_r", "terrain", "viridis", "turbo", "magma",
                            "plasma", "cividis", "RdYlBu_r"])
        self.cmap.currentTextChanged.connect(self._rerender_view)
        form.addRow("Colormap", self.cmap)

        self.exag = QDoubleSpinBox()
        self.exag.setRange(0.0, 5.0)
        self.exag.setSingleStep(0.1)
        self.exag.setValue(0.6)
        self.exag.setToolTip("Height of the relief relative to the base "
                             "(0 = flat, larger = more dramatic).")
        self.exag.valueChanged.connect(self._rerender_view)
        form.addRow("Vertical exaggeration", self.exag)

        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(1.0)  # opaque by default — no transparency
        self.opacity.setToolTip("Surface opacity (1 = solid, lower = see-through).")
        self.opacity.valueChanged.connect(self._rerender_view)
        form.addRow("Opacity", self.opacity)

        self.ystretch = QDoubleSpinBox()
        self.ystretch.setRange(0.1, 10.0)
        self.ystretch.setSingleStep(0.5)
        self.ystretch.setValue(_Y_ASPECT)
        self.ystretch.setToolTip("Width of the date axis relative to the "
                                 "time-of-day axis (larger = wider, more "
                                 "landscape-like; 1 = square base).")
        self.ystretch.valueChanged.connect(self._rerender_view)
        form.addRow("Y stretch", self.ystretch)

        self.ybin = QSpinBox()
        self.ybin.setRange(1, 90)
        self.ybin.setValue(1)
        self.ybin.setToolTip("Average this many consecutive days into each "
                             "date cell (1 = full daily detail). Larger = "
                             "wider cells, so spikes become broad, rounded "
                             "relief instead of sharp points.")
        self.ybin.valueChanged.connect(self._rerender_view)
        form.addRow("Y cell (days)", self.ybin)

        self.ybin_agg = QComboBox()
        self.ybin_agg.addItems(list(_AGGS))
        self.ybin_agg.setToolTip("How to combine the days in each cell: "
                                 "mean/median broaden and lower spikes; max "
                                 "keeps peaks tall with a wider base; min for "
                                 "troughs.")
        self.ybin_agg.currentTextChanged.connect(self._rerender_view)
        form.addRow("Cell aggregator", self.ybin_agg)

        self.smoothing = QSpinBox()
        self.smoothing.setRange(0, 2)
        self.smoothing.setValue(0)
        self.smoothing.setToolTip("Round the surface into rolling hills by "
                                  "subdividing + relaxing the mesh (0 = off). "
                                  "Higher = smoother but heavier.")
        self.smoothing.valueChanged.connect(self._rerender_view)
        form.addRow("Smooth terrain", self.smoothing)

        # Optional cast shadows from an overhead spotlight. Off = flat, evenly
        # lit (true colours, no haze). On = short cast shadows for a little depth;
        # "Shadow length" lowers the light so they stretch out.
        self.shadows = QCheckBox("Shadows")
        self.shadows.setChecked(False)
        self.shadows.setToolTip("Cast short shadows from an overhead spotlight "
                                "for a little depth (off = flat, evenly lit). "
                                "May be unavailable on some graphics drivers.")
        self.shadows.toggled.connect(self._on_shadows_changed)
        form.addRow(self.shadows)

        self.shadow_len = QSpinBox()
        self.shadow_len.setRange(1, 10)
        self.shadow_len.setValue(3)
        self.shadow_len.setToolTip("Shadow length: 1 = short (high light), "
                                   "10 = long (low light).")
        self.shadow_len.valueChanged.connect(self._rerender_view)
        form.addRow("Shadow length", self.shadow_len)

        lay.addLayout(form)

        self.smooth = QCheckBox("Smooth shading")
        self.smooth.setChecked(True)
        self.smooth.toggled.connect(self._rerender_view)
        lay.addWidget(self.smooth)

        self.edges = QCheckBox("Show mesh")
        self.edges.setChecked(False)
        self.edges.toggled.connect(self._rerender_view)
        lay.addWidget(self.edges)

        self.reset_btn = QPushButton("Reset view")
        self.reset_btn.clicked.connect(self._reset_camera)
        lay.addWidget(self.reset_btn)

        hint = QLabel(
            "<b>Mouse</b><br>"
            "Drag - rotate<br>"
            "Shift + drag - pan<br>"
            "Ctrl + drag - roll<br>"
            "Scroll / right-drag - zoom")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #90A4AE;")
        lay.addWidget(hint)

        # Now that smooth/smoothing exist, wire the style toggle and set the
        # initial enabled state (those two controls only apply to the surface).
        self.style.currentTextChanged.connect(self._on_style_changed)
        self._sync_style_enabled()
        self._sync_shadows_enabled()  # grey out Shadow length when shadows off

        lay.addStretch(1)
        return col

    # --- state ---------------------------------------------------------
    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"target": self._target,
                "controls": save_controls(
                    {"style": self.style, "cmap": self.cmap, "exag": self.exag,
                     "opacity": self.opacity,
                     "ystretch": self.ystretch, "ybin": self.ybin,
                     "ybin_agg": self.ybin_agg,
                     "shadows": self.shadows, "shadow_len": self.shadow_len,
                     "smooth": self.smooth, "edges": self.edges,
                     "smoothing": self.smoothing})}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls({"style": self.style, "cmap": self.cmap,
                          "exag": self.exag, "opacity": self.opacity,
                          "ystretch": self.ystretch, "ybin": self.ybin,
                          "ybin_agg": self.ybin_agg,
                          "shadows": self.shadows, "shadow_len": self.shadow_len,
                          "smooth": self.smooth, "edges": self.edges,
                          "smoothing": self.smoothing},
                         state.get("controls") or state)
        self._sync_style_enabled()
        self._sync_shadows_enabled()
        t = state.get("target")
        if t and self._df is not None and t in self._df.columns:
            self._on_select(t)

    # --- data flow -----------------------------------------------------
    def _rerender_view(self, *_a) -> None:
        # Colormap / exaggeration / shading all need the mesh rebuilt (height is
        # baked into geometry), but it's cheap, so just re-render the current var.
        if self._target is not None and self.canvas is not None:
            self._compute()

    def _reset_camera(self) -> None:
        if self.canvas is not None:
            self.canvas.frame_default()
            self.canvas.render()

    def _on_style_changed(self, _text: str) -> None:
        self._sync_style_enabled()
        self._rerender_view()

    def _sync_style_enabled(self) -> None:
        # Terrain smoothing (mesh subdivision) only applies to the interpolated
        # surface; grey it out for the heatmap. Smooth shading stays available in
        # both styles (off = crisp facets, on = rounded lighting).
        surface = self.style.currentText() == _STYLE_SURFACE
        self.smoothing.setEnabled(surface)

    def _on_shadows_changed(self, _checked: bool) -> None:
        self._sync_shadows_enabled()
        self._rerender_view()

    def _sync_shadows_enabled(self) -> None:
        self.shadow_len.setEnabled(self.shadows.isChecked())

    def _shadow_elevation(self) -> float:
        # Shadow length 1..10 -> light elevation 80..30 deg (short -> long).
        return 80.0 - (self.shadow_len.value() - 1) * (50.0 / 9.0)

    # --- codegen -------------------------------------------------------
    def _python_code(self) -> str | None:
        if self._df is None or self._target is None or self._target not in self._df.columns:
            return None
        from diive.core.plotting.codegen import datetime_surface_to_code
        return datetime_surface_to_code(self._target, cmap=self.cmap.currentText())

    # --- rendering -----------------------------------------------------
    def _compute(self) -> None:
        if self.canvas is None or self._target is None:
            return
        import pyvista as pv

        grid_data = dv.plotting.datetime_surface_grid(self._df[self._target])
        x_hours, y_days, z = grid_data.x_hours, grid_data.y_days, grid_data.z

        # Coarsen the date axis (wider Y cells) to broaden sharp spikes.
        z, y_days = _bin_rows(z, y_days, self.ybin.value(),
                              self.ybin_agg.currentText())

        # Stride the date rows if the grid is huge, to keep orbiting smooth.
        if z.shape[0] > _MAX_ROWS:
            step = int(np.ceil(z.shape[0] / _MAX_ROWS))
            y_days = y_days[::step]
            z = z[::step, :]

        # Normalise the base (x and y ranges differ wildly: hours 0-24 vs.
        # thousands of days) so the surface is well-proportioned regardless of
        # record length; exaggeration then sets the relief height. The date axis
        # is stretched by the "Y stretch" control so the relief reads as a wide
        # landscape.
        xn = _unit(x_hours)
        yn = _unit(y_days) * self.ystretch.value()

        finite = np.isfinite(z)
        if not finite.any():
            self.canvas.clear()
            self.canvas.render()
            return
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        span = zmax - zmin or 1.0
        # Geometry height: NaN cells dropped to the floor so the mesh stays
        # closed; their colour is hidden via nan_opacity below.
        z_fill = np.where(finite, z, zmin)
        height = (z_fill - zmin) / span * self.exag.value()
        # Colour by the real values (NaN preserved); order="F" matches PyVista's
        # StructuredGrid point ordering (see its structured-surface example).
        scalars = np.where(finite, z, np.nan)

        if self.style.currentText() == _STYLE_EXTRUDED:
            # One flat rectangle per cell with vertical walls (extruded heatmap):
            # no interpolated peaks, and flat shading keeps the facets crisp.
            xx, yy, height, scalars = _extruded_grid(xn, yn, height, scalars)
            mesh = pv.StructuredGrid(xx, yy, height)
            mesh["values"] = scalars.ravel(order="F")
            # Drop gap (NaN) cells outright rather than hiding them via
            # nan_opacity: a scalar opacity array forces VTK's translucent pass,
            # so the stacked walls/bars would blend through each other even at
            # opacity 1. Thresholding to the finite value range removes those
            # cells, leaving a fully opaque mesh.
            mesh = mesh.threshold([zmin, zmax], scalars="values", all_scalars=True)
            smooth_shading = self.smooth.isChecked()
            hide_nan = False
        else:
            xx, yy = np.meshgrid(xn, yn)  # (D, T), matching z
            mesh = pv.StructuredGrid(xx, yy, height)
            mesh["values"] = scalars.ravel(order="F")
            n_sub = self.smoothing.value()
            if n_sub > 0:
                # Round the faceted grid into rolling hills: triangulate,
                # subdivide to add vertices, then Taubin-relax (smooths without
                # the shrinkage plain Laplacian smoothing causes).
                mesh = (mesh.extract_surface(algorithm=None).triangulate()
                        .subdivide(n_sub).smooth_taubin(n_iter=20))
            smooth_shading = self.smooth.isChecked()
            hide_nan = True  # single-layer height field: opacity hiding is fine

        add_kwargs = dict(
            scalars="values",
            cmap=self.cmap.currentText(),
            opacity=self.opacity.value(),
            reset_camera=False,         # keep the view; we frame explicitly below
            smooth_shading=smooth_shading,
            show_edges=self.edges.isChecked(),
            edge_color="#90A4AE",
            scalar_bar_args={"title": self._target, "n_labels": 5},
        )
        if hide_nan:
            add_kwargs["nan_opacity"] = 0.0  # hide gap cells
        shadows = self.shadows.isChecked()
        if shadows:
            # High ambient keeps colours essentially flat (no hazy face shading);
            # a little diffuse lets the cast shadows read as gentle darkening.
            add_kwargs.update(ambient=0.85, diffuse=0.3, specular=0.0)
        else:
            # Fully flat: every face its true colourmap colour, no shading.
            add_kwargs.update(ambient=1.0, diffuse=0.0, specular=0.0)

        p = self.canvas.plotter
        p.clear()
        p.add_mesh(mesh, **add_kwargs)
        self.canvas.apply_shadows(shadows, self._shadow_elevation())
        p.show_axes()  # orientation marker (X=time of day, Y=date, Z=value)
        # Snap to the default 45° framing only for a newly selected variable; a
        # settings tweak re-renders in place so the user keeps their current view.
        if self._target != self._framed_target:
            self.canvas.frame_default()
            self._framed_target = self._target
        self.canvas.render()
