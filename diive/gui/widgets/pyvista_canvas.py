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

    def closeEvent(self, event) -> None:
        # QtInteractor holds a VTK render window that must be closed explicitly,
        # else it can leak / warn on shutdown.
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
