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

import math
import warnings

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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
_Y_ASPECT = 5.5


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

#: Rolling (sliding-window) variants. Unlike the block aggregators above, these
#: keep full daily resolution: each row is smoothed over a centred window of
#: ``n`` days instead of collapsing every ``n`` rows into one. Gaps are
#: preserved (values are only smoothed where a cell already had data).
_ROLLING_AGGS = {"rolling mean": np.nanmean, "rolling median": np.nanmedian}


def _roll_rows(z: np.ndarray, n: int, fn) -> np.ndarray:
    """Smooth the date axis with a centred rolling window of ``n`` rows.

    Keeps the row count (unlike ``_bin_rows``); ``fn`` is a NaN-aware reducer
    (nanmean/nanmedian) applied across each window. The window shrinks at the
    ends, and original gap cells stay NaN so smoothing never invents data.
    """
    d = z.shape[0]
    half = n // 2
    finite = np.isfinite(z)
    out = np.full_like(z, np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN window -> NaN
        for i in range(d):
            out[i] = fn(z[max(0, i - half):i + half + 1], axis=0)
    out[~finite] = np.nan  # don't fabricate values where the cell was a gap
    return out


def _bin_rows(z: np.ndarray, y_days: np.ndarray, n: int, agg: str = "mean"):
    """Aggregate every ``n`` consecutive date rows into one wider cell.

    Coarsens the date (Y) resolution so single-day spikes spread across the bin
    and read as broad relief instead of sharp 1-cell points. ``agg`` selects how
    the rows combine (see ``_AGGS``). The ``rolling *`` aggregators instead slide
    a window and keep every row (see ``_roll_rows``). NaN-aware (gap cells stay
    NaN); a trailing partial bin is padded with NaN. Row positions (``y_days``)
    always use the mean so cells stay evenly spaced regardless of the aggregator.
    """
    if n <= 1:
        return z, y_days
    if agg in _ROLLING_AGGS:
        # Rolling smooth keeps all rows (and their positions), just smoothed.
        return _roll_rows(z, n, _ROLLING_AGGS[agg]), y_days
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

#: Camera presets: (label, view vector, view-up). Scene axes are X = time of
#: day, Y = date, Z = value. The vector points from the scene centre toward the
#: camera. Ordered to sit two-per-row: the two overviews (Isometric / Top), the
#: date-axis pair (Front / Back), the time-axis pair (Left / Right), then the two
#: gently-tilted presentation views (Front 20° from the front, Side 20° from the
#: side, each 20° above horizontal). Top looks straight down Z, so its view-up
#: must be an in-plane axis (+Y). The tilted vectors are (cos20, sin20) = ~0.940,
#: 0.342 split between the horizontal look-axis and Z.
_VIEWS = (
    ("Isometric", (1.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
    ("Top", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("Front", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ("Back", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ("Left", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("Right", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("Front 20°", (0.0, -0.9397, 0.3420), (0.0, 0.0, 1.0)),
    ("Side 20°", (0.9397, 0.0, 0.3420), (0.0, 0.0, 1.0)),
)

#: Cinematic orbit: frame rate, plus the gentle rise-and-fall sweep. Elevation
#: traces a sine over the orbit (``_ORBIT_SWEEPS`` full up-and-down cycles per
#: 360°), amplitude in degrees. Small enough to never approach the poles where
#: VTK's ``Elevation`` would flip the view-up.
_ORBIT_FPS = 30
_ORBIT_ELEV_AMPLITUDE = 10.0
_ORBIT_SWEEPS = 2.0

#: Flyover reuses the shared speed knob (deg/s, sized for the orbit). Dividing by
#: this maps it to path-fractions per second, so at 20 deg/s a full pass over the
#: record takes ~6 s — a slow, cinematic drift.
_FLY_SPEED_DIV = 120.0

#: Default base-plate thickness for the 3-D-print export, in the surface's
#: normalised height units (the relief spans 0..vertical-exaggeration, so ~0.15
#: gives a sturdy plate the relief sits on). Absolute size is set later in the
#: slicer; this only fixes the proportion of plate to relief. Gap cells fall to
#: the plate so the solid stays watertight.
_PRINT_BASE = 0.15

#: Saturation multiplier applied to the baked VR/PowerPoint texture. The emissive
#: material shows colours flat, and many colormaps read pale, so we push them to
#: stay vivid in a slide (1.0 = colormap as-is).
_EXPORT_SATURATION = 1.8

#: The 12 triangles (as indices into a box's 8 corners) for one extruded cell:
#: corners 0-3 are the base ring (z=0), 4-7 the top ring (z=height). Base, top,
#: then the four walls. Material is double-sided, so winding isn't critical.
_BOX_FACES = np.array([
    (0, 2, 1), (0, 3, 2),   # base
    (4, 5, 6), (4, 6, 7),   # top
    (0, 1, 5), (0, 5, 4),   # wall y0
    (1, 2, 6), (1, 6, 5),   # wall x1
    (2, 3, 7), (2, 7, 6),   # wall y1
    (3, 0, 4), (3, 4, 7),   # wall x0
], dtype=np.int64)


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
        # Cinematic camera animations (orbit turntable / flyover), sharing one
        # timer. `_anim_mode` selects which runs; the rest is per-animation phase
        # (orbit: accumulated azimuth -> elevation sweep; fly: 0..1 ping-pong
        # position along the date axis). Timer is built with the controls column.
        self._anim_timer: QTimer | None = None
        self._anim_mode: str | None = None
        self._orbit_azimuth = 0.0
        self._orbit_last_elev = 0.0
        self._fly_phase = 0.0
        self._fly_dir = 1
        # Last computed (pre-doubling) height field, kept so the 3-D-print
        # export can rebuild a watertight solid without recomputing the grid.
        self._grid_xn: np.ndarray | None = None
        self._grid_yn: np.ndarray | None = None
        self._grid_height: np.ndarray | None = None
        self._grid_z: np.ndarray | None = None  # real values (NaN kept) for texture
        self._grid_style: str | None = None

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
            self.canvas.on_interaction_start(self._stop_anim)  # grab -> stop anim
            # Re-bake once the canvas is really shown: a render during tab build
            # (pre-realisation) can produce a bad shadow map.
            self.canvas.on_first_show(self._on_canvas_shown)
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
        self.cmap.setCurrentText("RdYlBu_r")
        self.cmap.currentTextChanged.connect(self._rerender_view)
        form.addRow("Colormap", self.cmap)

        self.exag = QDoubleSpinBox()
        self.exag.setRange(0.0, 5.0)
        self.exag.setSingleStep(0.1)
        self.exag.setValue(0.2)
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
        self.ystretch.setRange(0.1, 100.0)
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
        self.ybin.setToolTip("Days combined per date cell — the block size, or "
                             "the window width for the rolling aggregators "
                             "(1 = full daily detail, no binning). Larger = "
                             "broader, smoother relief.")
        self.ybin.valueChanged.connect(self._rerender_view)
        form.addRow("Y cell (days)", self.ybin)

        self.ybin_agg = QComboBox()
        self.ybin_agg.addItems(list(_AGGS) + list(_ROLLING_AGGS))
        self.ybin_agg.setToolTip(
            "How to combine the days per date cell. Block: mean/median broaden "
            "and lower spikes, max keeps peaks tall with a wider base, min for "
            "troughs. Rolling mean/median keep full daily resolution instead, "
            "smoothing each day over a centred window (gaps preserved).")
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
        self.shadow_len.setValue(1)
        self.shadow_len.setToolTip("Shadow length: 1 = short (high light), "
                                   "10 = long (low light).")
        self.shadow_len.valueChanged.connect(self._rerender_view)
        form.addRow("Shadow length", self.shadow_len)

        lay.addLayout(form)

        self.smooth = QCheckBox("Smooth shading")
        self.smooth.setChecked(False)
        self.smooth.toggled.connect(self._rerender_view)
        lay.addWidget(self.smooth)

        self.edges = QCheckBox("Show mesh")
        self.edges.setChecked(False)
        self.edges.toggled.connect(self._rerender_view)
        lay.addWidget(self.edges)

        # Camera preset buttons, two per row (see _VIEWS for the ordering).
        views = QGridLayout()
        views.setContentsMargins(0, 0, 0, 0)
        views.setHorizontalSpacing(6)
        views.setVerticalSpacing(6)
        n = len(_VIEWS)
        for i, (label, vector, viewup) in enumerate(_VIEWS):
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _=False, v=vector, u=viewup: self._set_view(v, u))
            if i == n - 1 and n % 2 == 1:
                views.addWidget(btn, i // 2, 0, 1, 2)  # trailing odd -> full width
            else:
                views.addWidget(btn, i // 2, i % 2)
        lay.addLayout(views)

        # Cinematic camera animations. Orbit = slow turntable with a gentle
        # rise-and-fall sweep; Flyover = a drone shot travelling over the relief
        # along the date axis. Mutually exclusive, driven by one timer; either
        # stops on any camera interaction.
        anim_row = QHBoxLayout()
        anim_row.setContentsMargins(0, 0, 0, 0)
        anim_row.setSpacing(6)
        self.orbit_btn = QPushButton("Orbit")
        self.orbit_btn.setCheckable(True)
        self.orbit_btn.setToolTip(
            "Slowly orbit the scene (turntable) with a gentle rise-and-fall "
            "sweep. Click the scene (or the button again) to stop.")
        self.orbit_btn.toggled.connect(self._toggle_orbit)
        anim_row.addWidget(self.orbit_btn, stretch=1)
        self.fly_btn = QPushButton("Flyover")
        self.fly_btn.setCheckable(True)
        self.fly_btn.setToolTip(
            "Drone-style flyover: glide over the relief along the date axis, "
            "forward and looking slightly down, then back. Click the scene (or "
            "the button again) to stop.")
        self.fly_btn.toggled.connect(self._toggle_fly)
        anim_row.addWidget(self.fly_btn, stretch=1)
        lay.addLayout(anim_row)

        speed_row = QHBoxLayout()
        speed_row.setContentsMargins(0, 0, 0, 0)
        speed_row.setSpacing(6)
        speed_row.addWidget(QLabel("Speed"))
        self.orbit_speed = QDoubleSpinBox()
        self.orbit_speed.setRange(2.0, 90.0)
        self.orbit_speed.setSingleStep(2.0)
        self.orbit_speed.setValue(20.0)
        self.orbit_speed.setSuffix(" deg/s")
        self.orbit_speed.setToolTip("Animation speed for the orbit and flyover.")
        speed_row.addWidget(self.orbit_speed, stretch=1)
        lay.addLayout(speed_row)

        self._anim_timer = QTimer(col)
        self._anim_timer.setInterval(int(1000 / _ORBIT_FPS))
        self._anim_timer.timeout.connect(self._anim_tick)

        # Export the surface to open 3-D formats: glTF for VR headsets / web,
        # a watertight STL solid for 3-D printing.
        lay.addWidget(list_header("Export", "VR / 3D printing"))
        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setSpacing(6)
        self.export_vr_btn = QPushButton("VR (.glb)")
        self.export_vr_btn.setToolTip(
            "Export the styled scene as glTF (.glb/.gltf) - the open 3-D format "
            "Meta Quest, Blender and web (WebXR) viewers load directly. Colours "
            "are baked in.")
        self.export_vr_btn.clicked.connect(self._export_vr)
        export_row.addWidget(self.export_vr_btn, stretch=1)
        self.export_print_btn = QPushButton("3D print (.stl)")
        self.export_print_btn.setToolTip(
            "Export a watertight solid with a base plate (STL) - ready to slice "
            "and 3-D print. Gaps fall to the plate so the model stays closed.")
        self.export_print_btn.clicked.connect(self._export_print)
        export_row.addWidget(self.export_print_btn, stretch=1)
        lay.addLayout(export_row)

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
                     "smoothing": self.smoothing,
                     "orbit_speed": self.orbit_speed})}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls({"style": self.style, "cmap": self.cmap,
                          "exag": self.exag, "opacity": self.opacity,
                          "ystretch": self.ystretch, "ybin": self.ybin,
                          "ybin_agg": self.ybin_agg,
                          "shadows": self.shadows, "shadow_len": self.shadow_len,
                          "smooth": self.smooth, "edges": self.edges,
                          "smoothing": self.smoothing,
                          "orbit_speed": self.orbit_speed},
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

    def _set_view(self, vector: tuple[float, float, float],
                  viewup: tuple[float, float, float]) -> None:
        self._stop_anim()  # a fixed view and an animation can't both hold the camera
        if self.canvas is not None:
            self.canvas.set_view(vector, viewup=viewup, tight=True)  # fill viewport
            self.canvas.render()

    # --- cinematic animations (orbit / flyover) ------------------------
    def _toggle_orbit(self, on: bool) -> None:
        self._set_anim("orbit" if on else None, self.orbit_btn)

    def _toggle_fly(self, on: bool) -> None:
        self._set_anim("fly" if on else None, self.fly_btn)

    def _set_anim(self, mode: str | None, sender: QPushButton) -> None:
        if self._anim_timer is None:
            return
        if mode is not None and self.canvas is None:  # nothing to animate
            sender.setChecked(False)
            return
        # Only one animation at a time: unchecking the other button routes back
        # through here with mode=None (stopping the timer) before we (re)start.
        if mode == "orbit" and self.fly_btn.isChecked():
            self.fly_btn.setChecked(False)
        elif mode == "fly" and self.orbit_btn.isChecked():
            self.orbit_btn.setChecked(False)
        self._anim_mode = mode
        if mode == "orbit":
            self._orbit_azimuth = 0.0
            self._orbit_last_elev = 0.0
            self.canvas.frame_default()  # clean iso start (also restores parallel)
            self._anim_timer.start()
        elif mode == "fly":
            self._fly_phase = 0.0
            self._fly_dir = 1
            self._anim_timer.start()
        else:
            self._anim_timer.stop()

    def _stop_anim(self) -> None:
        # Called on user interaction or when a fixed view is picked. Unchecking a
        # button routes through _set_anim(None) to stop the timer.
        for btn in (self.orbit_btn, self.fly_btn):
            if btn.isChecked():
                btn.setChecked(False)
        if self._anim_timer is not None:
            self._anim_timer.stop()
        self._anim_mode = None

    def _anim_tick(self) -> None:
        if self.canvas is None or self._anim_timer is None:
            return
        dt = self._anim_timer.interval() / 1000.0
        if self._anim_mode == "orbit":
            # Per-frame azimuth advance from the speed (deg/s).
            az_delta = self.orbit_speed.value() * dt
            self._orbit_azimuth += az_delta
            # Gentle rise-and-fall: elevation offset traces a sine over the
            # orbit, applied as the delta since the previous frame (VTK's
            # Elevation is incremental).
            target = _ORBIT_ELEV_AMPLITUDE * math.sin(
                math.radians(self._orbit_azimuth) * _ORBIT_SWEEPS)
            elev_delta = target - self._orbit_last_elev
            self._orbit_last_elev = target
            self.canvas.orbit_step(az_delta, elev_delta)
        elif self._anim_mode == "fly":
            # Advance the 0..1 path position, ping-ponging at the ends so the
            # flight reverses smoothly instead of cutting back to the start.
            self._fly_phase += (self.orbit_speed.value() / _FLY_SPEED_DIV) \
                * dt * self._fly_dir
            if self._fly_phase >= 1.0:
                self._fly_phase, self._fly_dir = 1.0, -1
            elif self._fly_phase <= 0.0:
                self._fly_phase, self._fly_dir = 0.0, 1
            # Ease in/out (cosine) so it slows near each end of the record.
            eased = 0.5 - 0.5 * math.cos(math.pi * self._fly_phase)
            self.canvas.fly_to(eased)

    # --- export --------------------------------------------------------
    def _export_vr(self) -> None:
        if self.canvas is None or self._grid_height is None:
            return
        self._stop_anim()
        path, _ = QFileDialog.getSaveFileName(
            self.export_vr_btn, "Export for VR / PowerPoint",
            f"{self._target}_surface.glb", "glTF scene (*.glb *.gltf)")
        if not path:
            return
        try:
            mesh = self._build_export_surface()
            if mesh is None:
                raise RuntimeError("No renderable surface to export.")
            mesh.export(path)
        except Exception as exc:
            # Fall back to the live-scene glTF (per-vertex colours). Some viewers
            # (e.g. PowerPoint) then show it white, but it's better than nothing.
            try:
                path = self.canvas.export_vr(path)
            except Exception:
                QMessageBox.warning(self.export_vr_btn, "Export failed", str(exc))
                return
        self._export_notice(path)

    def _build_export_surface(self):
        """Textured mesh of the current relief for VR / PowerPoint.

        Matches the on-screen style: smooth surface or extruded (stepped bars).
        Colours are baked into an image *texture* (per-point UV mapping) rather
        than per-vertex colours, which viewers like PowerPoint ignore (they
        render the model white). The texture is applied as an **emissive**
        (self-lit) material so it shows the true colormap colours instead of
        being washed out / brightened by the viewer's lighting — mirroring the
        flat-shaded on-screen plot. Gap cells are dropped as real holes (no
        transparency, which those viewers also handle poorly). Returns a
        ``trimesh.Trimesh`` or None if there's nothing to show.
        """
        if self._grid_height is None or self._grid_z is None:
            return None
        import matplotlib
        import trimesh
        from matplotlib.colors import Normalize
        from PIL import Image

        xn, yn = self._grid_xn, self._grid_yn
        height, z = self._grid_height, self._grid_z
        d, t = height.shape
        finite = np.isfinite(z)

        if self._grid_style == _STYLE_EXTRUDED:
            verts, faces, uv = self._extruded_export_arrays(
                xn, yn, height, z, d, t)
        else:
            verts, faces, uv = self._smooth_export_arrays(
                xn, yn, height, finite, d, t)
        if faces is None or len(faces) == 0:
            return None

        # Colormap texture (one texel per grid cell); emissive so lighting can't
        # wash it out. Black base -> no lit contribution added on top.
        zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
        cmap = matplotlib.colormaps[self.cmap.currentText()]
        rgb = cmap(Normalize(zmin, zmax)(np.where(finite, z, zmin)))[:, :, :3]
        # Push saturation so the colours stay vivid (emissive shows them flat).
        from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
        hsv = rgb_to_hsv(rgb)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * _EXPORT_SATURATION, 0.0, 1.0)
        image = Image.fromarray((hsv_to_rgb(hsv) * 255).astype(np.uint8))
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0, 0, 0, 255], emissiveTexture=image,
            emissiveFactor=[1.0, 1.0, 1.0], metallicFactor=0.0,
            roughnessFactor=1.0, doubleSided=True)
        visual = trimesh.visual.TextureVisuals(uv=uv, image=image,
                                               material=material)
        return trimesh.Trimesh(vertices=verts, faces=faces, visual=visual,
                               process=False)

    @staticmethod
    def _grid_faces(idx: np.ndarray, keep: np.ndarray):
        """Two upward-facing triangles per grid quad, kept where ``keep`` is set.

        ``keep`` is the (rows-1, cols-1) mask of quads to emit (all four corners
        measured). Winding is chosen so the top faces point +Z.
        """
        tl, tr = idx[:-1, :-1], idx[:-1, 1:]
        bl, br = idx[1:, :-1], idx[1:, 1:]
        g = keep.ravel()
        tri1 = np.column_stack([tl.ravel(), br.ravel(), bl.ravel()])[g]
        tri2 = np.column_stack([tl.ravel(), tr.ravel(), br.ravel()])[g]
        return np.vstack([tri1, tri2])

    def _smooth_export_arrays(self, xn, yn, height, finite, d, t):
        xx, yy = np.meshgrid(xn, yn)
        verts = np.column_stack([xx.ravel(), yy.ravel(), height.ravel()])
        idx = np.arange(d * t).reshape(d, t)
        keep = (finite[:-1, :-1] & finite[:-1, 1:]
                & finite[1:, :-1] & finite[1:, 1:])
        faces = self._grid_faces(idx, keep)
        # Continuous UVs: vertex (i,j) samples texel (i,j).
        uu, vv = np.meshgrid(np.arange(t) / max(t - 1, 1),
                             np.arange(d) / max(d - 1, 1))
        uv = np.column_stack([uu.ravel(), vv.ravel()])
        return verts, faces, uv

    @staticmethod
    def _staircase_cell_values(height, z):
        """Per-quad colour values for the extruded staircase shell (on screen).

        The doubled grid's quads are: cell tops (even row, even col), x-risers
        (even row, odd col), y-risers (odd row, even col) and zero-area corners
        (odd, odd). Tops take their own cell value; a riser takes the value of
        the **taller** of its two cells -- the bar whose front face that riser
        is -- so a bar's walls match its top instead of showing a neighbour's
        colour. Corners stay NaN (dropped by the threshold; zero-area anyway).
        Returns a ``(2D-1, 2T-1)`` array matching the structured-grid cell order.
        """
        d, t = z.shape
        vals = np.full((2 * d - 1, 2 * t - 1), np.nan)
        vals[0::2, 0::2] = z  # cell tops
        vals[0::2, 1::2] = np.where(height[:, :-1] >= height[:, 1:],
                                    z[:, :-1], z[:, 1:])  # x-riser -> taller
        vals[1::2, 0::2] = np.where(height[:-1, :] >= height[1:, :],
                                    z[:-1, :], z[1:, :])  # y-riser -> taller
        return vals

    @staticmethod
    def _extruded_box_geometry(xn, yn, height, z):
        """One box per measured cell (base plane -> the cell's height).

        Returns ``(verts (N*8, 3), faces (N*12, 3) triangles, ii, jj)`` with
        ``ii``/``jj`` the finite-cell row/col indices in block order, or
        ``(None, None, None, None)`` if nothing is measured. Boxes don't share
        vertices, so colour never bleeds between neighbours -- the top and all
        four walls of a bar carry that cell's single value (as a texel in the
        export, as cell-data on screen). Shared by the live render and the VR
        export so both show identical bars.
        """
        xe, ye = _cell_edges(xn), _cell_edges(yn)  # cell boundaries (len +1)
        ii, jj = np.where(np.isfinite(z))
        n = ii.size
        if n == 0:
            return None, None, None, None
        x0, x1 = xe[jj], xe[jj + 1]
        y0, y1 = ye[ii], ye[ii + 1]
        zt = height[ii, jj]
        z0 = np.zeros(n)
        # 8 corners/box: 0-3 base ring (z0), 4-7 top ring (zt).
        cx = np.stack([x0, x1, x1, x0, x0, x1, x1, x0], axis=1)
        cy = np.stack([y0, y0, y1, y1, y0, y0, y1, y1], axis=1)
        cz = np.stack([z0, z0, z0, z0, zt, zt, zt, zt], axis=1)
        verts = np.stack([cx, cy, cz], axis=2).reshape(n * 8, 3)
        faces = (_BOX_FACES[None] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 3)
        return verts, faces, ii, jj

    def _extruded_export_arrays(self, xn, yn, height, z, d, t):
        # Per-cell boxes + a UV per corner at its cell's texel centre, so every
        # face of a bar samples one texel -> one uniform colour.
        verts, faces, ii, jj = self._extruded_box_geometry(xn, yn, height, z)
        if verts is None:
            return None, None, None
        uv = np.column_stack([np.repeat((jj + 0.5) / t, 8),
                              np.repeat((ii + 0.5) / d, 8)])
        return verts, faces, uv

    def _export_print(self) -> None:
        if self.canvas is None or self._grid_height is None:
            return
        self._stop_anim()
        path, _ = QFileDialog.getSaveFileName(
            self.export_print_btn, "Export for 3-D printing",
            f"{self._target}_surface.stl",
            "STL model (*.stl);;PLY model (*.ply)")
        if not path:
            return
        try:
            self._printable_solid().save(path)
        except Exception as exc:
            QMessageBox.warning(self.export_print_btn, "Export failed", str(exc))
            return
        self._export_notice(path)

    def _printable_solid(self):
        """Watertight solid of the current relief with a base plate (for STL).

        The rendered surface is a single-layer height *sheet* — not printable.
        Here it becomes a closed solid: the relief is the top, a flat plate the
        bottom, joined by side walls (a two-layer structured grid -> its outer
        surface is watertight). Gap cells already sit at height 0, so they rest
        on the plate rather than punching holes. The extruded style keeps its
        stepped bars; the smooth style prints as smooth relief.
        """
        import pyvista as pv

        xn, yn, height = self._grid_xn, self._grid_yn, self._grid_height
        if self._grid_style == _STYLE_EXTRUDED:
            xx, yy, hh, _ = _extruded_grid(xn, yn, height, height)
        else:
            xx, yy = np.meshgrid(xn, yn)
            hh = height
        ztop = _PRINT_BASE + hh
        zbot = np.zeros_like(hh)
        gx = np.stack([xx, xx])
        gy = np.stack([yy, yy])
        gz = np.stack([zbot, ztop])
        solid = pv.StructuredGrid(gx, gy, gz)
        # clean() before triangulate(): merge the doubled-edge coincident points
        # (the extruded style's zero-width walls) so STL gets clean triangles.
        return solid.extract_surface(algorithm=None).clean().triangulate()

    def _export_notice(self, path: str) -> None:
        QMessageBox.information(self.export_vr_btn, "Export complete",
                                f"Saved:\n{path}")

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

        # Stash the pre-doubling height field for the printable-solid export and
        # the real values (NaN kept) for the textured VR export.
        self._grid_xn, self._grid_yn = xn, yn
        self._grid_height = height
        self._grid_z = scalars
        self._grid_style = self.style.currentText()

        if self.style.currentText() == _STYLE_EXTRUDED:
            # Doubled "staircase" grid: each cell's flat top at its height with
            # short step-risers between neighbours, open underneath -- a draped
            # heatmap, not solid bars sitting on a base plane (no side walls to
            # the floor, no flat bottom). Colour via CELL data (flat per quad) so
            # each top and riser is one solid colour with no gradient blending; a
            # riser takes the TALLER neighbour's value so a bar's front face
            # matches its top.
            xx, yy, hh, _ = _extruded_grid(xn, yn, height, scalars)
            mesh = pv.StructuredGrid(xx, yy, hh)
            mesh.cell_data["values"] = \
                self._staircase_cell_values(height, scalars).ravel(order="F")
            # Drop gap cells (NaN value) so the shell stays opaque.
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
        p.show_axes()  # orientation marker (X=time of day, Y=date, Z=value)
        # Snap to the isometric framing (same tight fit as the Isometric button)
        # only for a newly selected variable; a settings tweak re-renders in
        # place so the user keeps their current view.
        if self._target != self._framed_target:
            self.canvas.set_view(_VIEWS[0][1], viewup=_VIEWS[0][2], tight=True)
            self._framed_target = self._target
        # Apply shadows AFTER framing so the shadow-map pass is set up against the
        # final camera/scene; applied earlier, the first render shows wrong
        # shadows until the user toggles them (which re-applies with the settled
        # view).
        self.canvas.apply_shadows(shadows, self._shadow_elevation())
        self.canvas.render()
        if shadows:
            # VTK's shadow-map baker can bake a bad map on the first render after
            # the tab is shown (window not yet realised/sized). Re-render once the
            # event loop has settled so it re-bakes against the real window --
            # exactly what a manual shadow toggle did.
            QTimer.singleShot(0, self._rebake_shadows)

    def _rebake_shadows(self) -> None:
        if self.canvas is not None:
            self.canvas.render()

    def _on_canvas_shown(self) -> None:
        # First time the canvas is realised: re-apply shadows (if on) and render
        # so the shadow map bakes against the real window, not the pre-show state
        # -- the automatic equivalent of a manual shadow off/on toggle.
        if self.canvas is None or self._target is None:
            return
        if self.shadows.isChecked():
            self.canvas.apply_shadows(True, self._shadow_elevation())
        self.canvas.render()
