"""
GUI.WIDGETS.PYVISTA_CANVAS: EMBEDDED GPU 3-D CANVAS
===================================================

A Qt widget wrapping ``pyvistaqt.QtInteractor`` — a VTK/OpenGL render window
that embeds in a PySide6 layout and gives smooth, GPU-accelerated rotation of
3-D scenes (trackball camera, no event-loop blocking).

PyVista/VTK is an **optional** dependency (the ``gui3d`` extra). It is imported
lazily here so a plain ``gui`` install never pulls in VTK; `pyvista_available()`
reports whether it's present, and `Pyvista3DCanvas` raises a clear
:class:`Missing3DDependency` if constructed without it (the tab catches this and
shows install instructions instead of a stack trace).

Pure presentation glue (camera, lighting, mesh styling) — no domain logic; the
numeric surface grid comes from the library (`dv.plotting.datetime_surface_grid`).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import importlib.util

from PySide6.QtWidgets import QVBoxLayout, QWidget

#: Console-script hint shown when the extra is missing.
INSTALL_HINT = "uv sync --extra gui --extra gui3d  (or: pip install 'diive[gui,gui3d]')"


class Missing3DDependency(RuntimeError):
    """Raised when the 3-D canvas is built without the ``gui3d`` extra."""


def pyvista_available() -> bool:
    """True if both ``pyvista`` and ``pyvistaqt`` can be imported."""
    return (importlib.util.find_spec("pyvista") is not None
            and importlib.util.find_spec("pyvistaqt") is not None)


class Pyvista3DCanvas(QWidget):
    """Embeddable PyVista render window (lazy-loads VTK).

    Use `plotter` (the `pyvistaqt.QtInteractor`) to add meshes, then call
    `render()`. `clear()` empties the scene between renders.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if not pyvista_available():
            raise Missing3DDependency(
                "3-D plotting needs the 'gui3d' extra. Install:\n" + INSTALL_HINT)

        # Imported here (not at module top) so the module loads in a plain
        # 'gui' install; the registry imports the tab unconditionally.
        import pyvista as pv
        from pyvistaqt import QtInteractor

        pv.set_plot_theme("document")  # white background, dark text

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

    def clear(self) -> None:
        """Remove all actors from the scene."""
        self.plotter.clear()

    def render(self) -> None:
        """Repaint the render window."""
        self.plotter.render()

    def reset_camera(self) -> None:
        """Frame the current scene."""
        self.plotter.reset_camera()

    def frame_default(self, zoom: float = 1.2) -> None:
        """Default framing: a 45°-ish view from the lower-left, zoomed to fill.

        Looks down the (+X, -Y, +Z) corner so the date (Y) axis runs from the
        lower-left to the upper-right of the screen (time reads diagonally
        upward). Parallel (orthographic) projection keeps the elongated relief
        centred in the viewport — perspective would enlarge the near end and push
        the surface off to a corner. ``reset_camera`` frames the scene and the
        extra zoom tightens the margins the bounding-sphere fit leaves.
        """
        self.plotter.enable_parallel_projection()
        self.plotter.view_vector((1.0, -1.0, 1.0), viewup=(0.0, 0.0, 1.0))
        self.plotter.reset_camera()
        if zoom != 1.0:
            self.plotter.camera.zoom(zoom)

    def apply_shadows(self, shadows: bool, elevation: float = 70.0,
                      azimuth: float = 315.0) -> None:
        """Flat even light, or cast shadows from an overhead spotlight.

        With ``shadows`` off, a single camera headlight gives even, flat
        illumination (paired with high-ambient colouring the mesh shows its true
        colours). With it on, a positional spotlight above the scene aimed from
        (``azimuth``, ``elevation``) degrees casts real shadows via VTK shadow
        mapping — a high ``elevation`` keeps them short. A spotlight (positional)
        is used because VTK shadow maps need one; the wide cone + no attenuation
        keep the surface evenly lit. Shadow mapping is wrapped defensively (it can
        be unavailable on software/offscreen OpenGL backends).
        """
        import math

        import pyvista as pv

        p = self.plotter
        try:
            p.disable_ssao()  # drop any previously-enabled occlusion pass
        except Exception:
            pass
        p.remove_all_lights()
        if not shadows:
            p.add_light(pv.Light(light_type="headlight"))  # even, flat fill
            try:
                p.disable_shadows()
            except Exception:
                pass
            return

        # Position a spotlight on the (azimuth, elevation) ray, out beyond the
        # scene, pointed at its centre.
        b = p.bounds  # xmin, xmax, ymin, ymax, zmin, zmax
        cx, cy, cz = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2
        diag = math.dist((b[0], b[2], b[4]), (b[1], b[3], b[5])) or 1.0
        el, az = math.radians(elevation), math.radians(azimuth)
        dx, dy, dz = (math.cos(el) * math.sin(az),
                      math.cos(el) * math.cos(az), math.sin(el))
        dist = diag * 1.5
        light = pv.Light(position=(cx + dx * dist, cy + dy * dist, cz + dz * dist),
                         focal_point=(cx, cy, cz),
                         light_type="scene light", intensity=0.9)
        light.positional = True
        light.cone_angle = 80.0               # wide enough to cover the scene
        light.attenuation_values = (1, 0, 0)  # constant — no distance falloff
        p.add_light(light)
        try:
            p.enable_shadows()
        except Exception:
            pass  # shadow mapping unsupported on this GL backend

    def closeEvent(self, event) -> None:
        # QtInteractor holds a VTK render window that must be closed explicitly,
        # else it can leak / warn on shutdown.
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
