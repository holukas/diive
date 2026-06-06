"""
GUI.WIDGETS.MPL_CANVAS: EMBEDDED MATPLOTLIB CANVAS
==================================================

A Qt widget bundling a matplotlib `Figure` with its `FigureCanvasQTAgg` and
the standard navigation toolbar (pan/zoom/save). Callers obtain a row of axes
via `new_axes(ncols)`, render into them, and call `draw()`.

Uses the Agg-on-Qt backend explicitly so it works regardless of the global
matplotlib backend (diive plot classes default to interactive use).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget


class MplCanvas(QWidget):
    """Embeddable matplotlib canvas with a navigation toolbar.

    Render by calling `new_axes(ncols)` to get a fresh row of axes (the figure
    is cleared first), drawing into them, then `draw()`.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Give this widget a light palette BEFORE building the toolbar. The
        # matplotlib Qt toolbar recolors its icons to a light tint when it
        # detects a dark background palette; our white stylesheet alone does
        # not change the palette, so on a dark system theme the icons would
        # render white-on-white. A light palette (inherited by the toolbar)
        # keeps the dark icons.
        pal = self.palette()
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Button,
                     QPalette.ColorRole.Base):
            pal.setColor(role, QColor("#FFFFFF"))
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText,
                     QPalette.ColorRole.Text):
            pal.setColor(role, QColor("#212121"))
        self.setPalette(pal)

        self.fig = Figure(layout="constrained", facecolor="white")
        self._canvas = FigureCanvasQTAgg(self.fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas, stretch=1)

        # Toolbar in a bottom row, pushed to the right corner by a stretch.
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 4, 2)
        bottom.addStretch(1)
        bottom.addWidget(self._toolbar)
        layout.addLayout(bottom)

    def new_axes(self, n: int = 1, orientation: str = "horizontal",
                 sharex: bool = False, sharey: bool = False) -> list[Axes]:
        """Clear the figure and return a fresh strip of `n` axes.

        `orientation='horizontal'` lays panels side by side (one row);
        `'vertical'` stacks them top to bottom (one column). Clearing the whole
        figure (not just `ax.clear()`) discards any extra axes a previous render
        added -- e.g. the colorbar `HeatmapDateTime` appends -- so they do not
        stack up across renders. `sharex`/`sharey` link the panels' axes so
        pan/zoom on one applies to all.
        """
        self.fig.clear()
        if orientation == "vertical":
            grid = self.fig.subplots(n, 1, squeeze=False, sharex=sharex, sharey=sharey)
            return list(grid[:, 0])
        grid = self.fig.subplots(1, n, squeeze=False, sharex=sharex, sharey=sharey)
        return list(grid[0])

    def draw(self) -> None:
        """Force a synchronous repaint of the canvas.

        Use `draw()` (not `draw_idle()`): after re-rendering on a user action
        we want the new plot on screen immediately. `draw_idle()` only
        schedules a repaint for the next idle moment, which on some Qt/OS
        combinations leaves the canvas stale until the window is resized or
        refocused.
        """
        self._canvas.draw()
        self._canvas.flush_events()
