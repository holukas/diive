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

from PySide6.QtCore import QTimer
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

        # Fired once, the first time the widget is actually shown (see below).
        self._on_first_show = None

    def on_first_show(self, callback) -> None:
        """Register a callback fired once, the first time the widget is shown.

        The GL render window isn't realised until the widget is displayed, so a
        render before that (e.g. during tab construction) can mis-bake -- most
        visibly the shadow map. This lets the tab re-render/re-apply once the
        window truly exists.
        """
        self._on_first_show = callback

    def showEvent(self, event) -> None:
        super().showEvent(event)
        cb, self._on_first_show = self._on_first_show, None  # fire once
        if cb is not None:
            # Defer to the next event-loop tick so the first paint/GL realisation
            # has happened before the callback re-renders.
            QTimer.singleShot(0, cb)

    def clear(self) -> None:
        """Remove all actors from the scene."""
        self.plotter.clear()

    def render(self) -> None:
        """Repaint the render window."""
        self.plotter.render()

    def reset_camera(self) -> None:
        """Frame the current scene."""
        self.plotter.reset_camera()

    def set_view(self, vector: tuple[float, float, float],
                 viewup: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 zoom: float = 1.2, tight: bool = False) -> None:
        """Frame the scene from a fixed direction (parallel projection).

        ``vector`` points from the focal point toward the camera (so
        ``(1, -1, 1)`` puts the camera in the +X/-Y/+Z corner). Parallel
        (orthographic) projection keeps the elongated relief centred in the
        viewport — perspective would enlarge the near end and push the surface
        off to a corner. ``reset_camera`` frames the scene; ``tight`` then
        maximises the mesh in the viewport for *this* orientation (used by the
        preset view buttons), while the looser ``zoom`` path (bounding-sphere
        fit) is kept for framing that must survive rotation (the orbit).
        """
        self.plotter.enable_parallel_projection()
        self.plotter.view_vector(vector, viewup=viewup)
        self.plotter.reset_camera()
        if tight:
            self._fit_tight()
        elif zoom != 1.0:
            self.plotter.camera.zoom(zoom)

    def _fit_tight(self, padding: float = 0.04) -> None:
        """Scale the parallel view so the mesh fills the viewport (this view).

        Projects the bounding-box corners onto the camera's right/up axes and
        sets the parallel scale to just contain them — a tight, orientation-
        specific fit (unlike ``reset_camera``'s orientation-independent sphere
        fit, which leaves an elongated plot small when seen from the side).
        """
        import numpy as np

        p = self.plotter
        cam = p.camera
        pos = np.asarray(cam.position, float)
        foc = np.asarray(cam.focal_point, float)
        fwd = foc - pos
        if not np.linalg.norm(fwd):
            return
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.asarray(cam.up, float))
        if not np.linalg.norm(right):
            return
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        b = p.bounds
        corners = np.array([[b[i], b[2 + j], b[4 + k]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
        rel = corners - foc
        half_w = np.ptp(rel @ right) / 2.0
        half_h = np.ptp(rel @ up) / 2.0
        w, h = p.window_size
        aspect = (w / h) if h else 1.0
        scale = max(half_h, half_w / aspect if aspect else half_h)
        if scale > 0:
            cam.parallel_scale = scale * (1.0 + padding)
        try:
            p.renderer.reset_camera_clipping_range()
        except Exception:
            pass

    def frame_default(self, zoom: float = 1.2) -> None:
        """Default framing: a 45°-ish isometric view from the lower-left.

        Looks down the (+X, -Y, +Z) corner so the date (Y) axis runs from the
        lower-left to the upper-right of the screen (time reads diagonally
        upward).
        """
        self.set_view((1.0, -1.0, 1.0), viewup=(0.0, 0.0, 1.0), zoom=zoom)

    def orbit_step(self, azimuth_delta: float,
                   elevation_delta: float = 0.0) -> None:
        """Nudge the camera one animation frame around the scene, then repaint.

        Azimuth spins the camera about the scene's up axis; the optional
        elevation tilt is re-orthogonalised afterwards so the horizon stays
        level (VTK's ``Elevation`` otherwise skews the view-up). Wrapped
        defensively — a queued frame can fire after the render window is torn
        down, and shadow/clipping updates can raise on some GL backends.
        """
        try:
            cam = self.plotter.camera
            cam.Azimuth(azimuth_delta)
            if elevation_delta:
                cam.Elevation(elevation_delta)
                cam.OrthogonalizeViewUp()
            self.plotter.renderer.reset_camera_clipping_range()
            self.plotter.render()
        except Exception:
            pass

    def fly_to(self, fraction: float, trail: float = 1.1,
               margin: float = 0.05, look_down: float = 0.32,
               view_angle: float = 50.0) -> None:
        """Flyover that glides over the relief along the date (Y) axis.

        Unlike the orbit (a turntable circling the centre), this travels down the
        record. The **look point** sweeps from just before the first record
        (``fraction`` 0) to just past the last (``fraction`` 1); the ``margin``
        (a fraction of the record length) gives a little lead-in / run-out so the
        first and last records are fully seen. Crucially, the camera trails and
        rises by ``trail`` / ``look_down`` measured in **cross-section widths**
        (X-span), NOT record length — so it starts close to the data no matter
        how long the record is (a fraction of a multi-year Y-span would start the
        camera absurdly far back, shrinking the plot to a dot). The result is a
        ~16 deg downward look from just ahead of the first records. Perspective
        projection sells the depth. Wrapped defensively: a queued frame can fire
        after teardown.
        """
        try:
            xmin, xmax, ymin, ymax, zmin, zmax = self.plotter.bounds
            yspan = (ymax - ymin) or 1.0
            xspan = (xmax - xmin) or 1.0
            zspan = (zmax - zmin) or (0.1 * xspan)
            cx = (xmin + xmax) / 2.0
            # Look point spans the whole record (+/- margin along Y); the camera
            # sits a fixed short distance behind and above it (in X-span units).
            foc_y = ymin + (fraction * (1.0 + 2.0 * margin) - margin) * yspan
            foc_z = zmin + 0.30 * zspan
            cam = self.plotter.camera
            cam.position = (cx, foc_y - trail * xspan, foc_z + look_down * xspan)
            cam.focal_point = (cx, foc_y, foc_z)
            cam.up = (0.0, 0.0, 1.0)
            cam.view_angle = view_angle
            self.plotter.disable_parallel_projection()  # perspective = depth
            self.plotter.renderer.reset_camera_clipping_range()
            self.plotter.render()
        except Exception:
            pass

    def on_interaction_start(self, callback) -> None:
        """Call ``callback`` when the user grabs the camera (mouse press/wheel).

        Lets the tab cancel an automatic orbit the moment the user starts
        interacting, so their input isn't fought by the animation.
        """
        iren = self.plotter.iren
        for event in ("LeftButtonPressEvent", "RightButtonPressEvent",
                      "MiddleButtonPressEvent", "MouseWheelForwardEvent",
                      "MouseWheelBackwardEvent"):
            iren.add_observer(event, lambda *_a: callback())

    def export_vr(self, path: str) -> str:
        """Export the current styled scene as glTF for VR/AR; return the path.

        glTF (``.gltf``/``.glb``) is the open standard that Meta Quest, Blender
        and WebXR viewers load directly, with the colormap colours baked in as
        vertex data. ``inline_data`` embeds the buffers so a ``.gltf`` is a
        single portable file. A ``.glb`` target is packed to one binary file
        when ``trimesh`` is installed; otherwise the ``.gltf`` written beside it
        is kept and its path returned instead.
        """
        import os

        # Drop the orientation-axis marker so its arrows/cones don't get baked
        # into the exported model; restore it afterwards.
        try:
            self.plotter.hide_axes()
        except Exception:
            pass
        try:
            if path.lower().endswith(".glb"):
                gltf_path = os.path.splitext(path)[0] + ".gltf"
                self.plotter.export_gltf(gltf_path, inline_data=True)
                try:
                    import trimesh
                    trimesh.load(gltf_path).export(path)
                    os.remove(gltf_path)
                    return path
                except Exception:
                    return gltf_path  # trimesh missing/failed -> keep .gltf
            self.plotter.export_gltf(path, inline_data=True)
            return path
        finally:
            try:
                self.plotter.show_axes()
            except Exception:
                pass

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
        # else it can leak / warn on shutdown. Drop the shadow pass first: its
        # shadow-map framebuffer wants releasing before the GL context tears down
        # (else VTK logs "FrameBufferObject should have been deleted ...").
        try:
            self.plotter.disable_shadows()
        except Exception:
            pass
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
