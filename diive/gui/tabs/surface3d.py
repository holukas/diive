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

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.pyvista_canvas import (
    INSTALL_HINT,
    Pyvista3DCanvas,
    pyvista_available,
)
from diive.gui.widgets.variable_panel import VariablePanel

#: A continuous flux with a strong diel cycle makes the relief instantly legible.
_DEFAULT_VAR = "NEE_CUT_REF_f"

#: Keep rendering snappy: stride the date rows when the grid gets very large.
_MAX_ROWS = 2000


def _unit(a: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1] (flat array -> all zeros)."""
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


class Surface3DTab(DiiveTab):
    """Rotatable 3-D relief surface of a variable's date x time-of-day grid."""

    title = "3D surface"

    def build(self) -> QWidget:
        self._df = None
        self._target = None

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(self._on_select)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_controls())

        if pyvista_available():
            self.canvas = Pyvista3DCanvas()
            rl.addWidget(self.canvas, stretch=1)
        else:
            self.canvas = None
            rl.addWidget(self._build_missing_notice(), stretch=1)

        splitter.addWidget(self.varpanel)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)
        return root

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
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)

        lay.addWidget(QLabel("Colormap"))
        self.cmap = QComboBox()
        self.cmap.addItems(["terrain", "viridis", "turbo", "magma", "plasma",
                            "cividis", "RdYlBu_r", "Spectral_r"])
        self.cmap.currentTextChanged.connect(self._rerender_view)
        lay.addWidget(self.cmap)

        lay.addWidget(QLabel("Vertical exaggeration"))
        self.exag = QDoubleSpinBox()
        self.exag.setRange(0.0, 5.0)
        self.exag.setSingleStep(0.1)
        self.exag.setValue(0.6)
        self.exag.setToolTip("Height of the relief relative to the base "
                             "(0 = flat, larger = more dramatic).")
        self.exag.valueChanged.connect(self._rerender_view)
        lay.addWidget(self.exag)

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
        lay.addStretch(1)
        return bar

    # --- state ---------------------------------------------------------
    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"target": self._target,
                "controls": save_controls(
                    {"cmap": self.cmap, "exag": self.exag,
                     "smooth": self.smooth, "edges": self.edges})}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls({"cmap": self.cmap, "exag": self.exag,
                          "smooth": self.smooth, "edges": self.edges},
                         state.get("controls") or state)
        t = state.get("target")
        if t and self._df is not None and t in self._df.columns:
            self._on_select(t)

    # --- data flow -----------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self.varpanel.set_variables(df.columns, created)
        numeric = df.select_dtypes(include="number").columns.tolist()
        cols = [str(c) for c in numeric]
        if _DEFAULT_VAR in cols:
            default = _DEFAULT_VAR
        elif numeric:
            default = str(numeric[0])
        else:
            return
        self._on_select(default)

    def _on_select(self, name: str, _additive: bool = False) -> None:
        if not name or self._df is None or self.canvas is None:
            return
        self._target = name
        self.varpanel.set_panels([name])
        self.varpanel.run_with_loading(name, self._render)

    def _rerender_view(self, *_a) -> None:
        # Colormap / exaggeration / shading all need the mesh rebuilt (height is
        # baked into geometry), but it's cheap, so just re-render the current var.
        if self._target is not None and self.canvas is not None:
            self._render()

    def _reset_camera(self) -> None:
        if self.canvas is not None:
            self.canvas.reset_camera()
            self.canvas.render()

    # --- rendering -----------------------------------------------------
    def _render(self) -> None:
        if self.canvas is None or self._target is None:
            return
        import pyvista as pv

        grid_data = dv.plotting.datetime_surface_grid(self._df[self._target])
        x_hours, y_days, z = grid_data.x_hours, grid_data.y_days, grid_data.z

        # Stride the date rows if the grid is huge, to keep orbiting smooth.
        if z.shape[0] > _MAX_ROWS:
            step = int(np.ceil(z.shape[0] / _MAX_ROWS))
            y_days = y_days[::step]
            z = z[::step, :]

        # Normalise the base to a unit square (x and y ranges differ wildly:
        # hours 0-24 vs. thousands of days), so the surface is well-proportioned
        # regardless of record length; exaggeration then sets the relief height.
        xn = _unit(x_hours)
        yn = _unit(y_days)
        xx, yy = np.meshgrid(xn, yn)  # (D, T), matching z

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

        surf = pv.StructuredGrid(xx, yy, height)
        # Colour by the real values (NaN preserved). order="F" matches PyVista's
        # StructuredGrid point ordering (see its structured-surface example).
        scalars = np.where(finite, z, np.nan)
        surf["values"] = scalars.ravel(order="F")

        p = self.canvas.plotter
        p.clear()
        p.add_mesh(
            surf,
            scalars="values",
            cmap=self.cmap.currentText(),
            nan_opacity=0.0,            # hide gap cells
            smooth_shading=self.smooth.isChecked(),
            show_edges=self.edges.isChecked(),
            edge_color="#90A4AE",
            scalar_bar_args={"title": self._target, "n_labels": 5},
        )
        p.show_axes()  # orientation marker (X=time of day, Y=date, Z=value)
        p.reset_camera()
        self.canvas.render()
