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
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget


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

        # When True (default), draw() freezes constrained layout and resize
        # re-solves it. Set False for plots that manage their own figure layout
        # (e.g. the ridgeline's manual overlapping gridspec), so neither touches
        # their positions.
        self.auto_layout = True

        self.fig = Figure(layout="constrained", facecolor="white")
        self._canvas = FigureCanvasQTAgg(self.fig)
        # coordinates=False drops the toolbar's x/y readout label (not needed
        # here -- the hover tooltip shows values instead).
        self._toolbar = NavigationToolbar2QT(self._canvas, self, coordinates=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas, stretch=1)

        # Value-under-cursor tooltip (line snap + heatmap cell). Attaches to the
        # canvas events and works across re-renders; see widgets/hover.py.
        from diive.gui.widgets.hover import HoverAnnotator
        self.hover = HoverAnnotator(self)

        # Bottom row: a "Hover values" toggle next to the navigation toolbar,
        # both pushed to the right corner by a stretch.
        self._hover_toggle = QCheckBox("Hover values")
        self._hover_toggle.setChecked(True)
        self._hover_toggle.toggled.connect(self.hover.set_enabled)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 4, 2)
        bottom.addStretch(1)
        bottom.addWidget(self._hover_toggle)
        bottom.addWidget(self._toolbar)
        layout.addLayout(bottom)

        # Re-solve the (frozen) constrained layout whenever the canvas resizes,
        # so panels adapt to the real widget size. Pan/zoom doesn't resize, so
        # it stays frozen -- see draw()/_on_resize.
        self._canvas.mpl_connect("resize_event", self._on_resize)

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
        self.reset_layout()
        if orientation == "vertical":
            grid = self.fig.subplots(n, 1, squeeze=False, sharex=sharex, sharey=sharey)
            return list(grid[:, 0])
        grid = self.fig.subplots(1, n, squeeze=False, sharex=sharex, sharey=sharey)
        return list(grid[0])

    def _on_resize(self, _event) -> None:
        """Re-solve the constrained layout for the new size, then re-freeze.

        `draw()` freezes the layout so interactive pan/zoom stays stable, but a
        layout frozen at the initial (pre-show) canvas size would not match the
        real widget size. A resize is exactly when re-solving is wanted (and
        pan/zoom never resizes), so here we briefly re-enable the constrained
        engine, solve at the new size with `draw_without_rendering()`, then turn
        it off again -- leaving correct, frozen positions for the resize repaint.
        """
        if not self.auto_layout:
            return  # the plot manages its own layout (e.g. ridgeline)
        self.fig.set_layout_engine("constrained")
        try:
            self.fig.draw_without_rendering()
        except Exception:
            pass  # no renderer yet (very early); the next resize/render fixes it
        self.fig.set_layout_engine("none")

    def reset_layout(self) -> None:
        """Clear the figure and re-enable constrained layout for a fresh render.

        `draw()` *freezes* the layout afterwards (see there), so before building
        a new set of panels the constrained engine must be turned back on to
        size them and place the colorbar. Callers that build panels directly
        (e.g. the Overview's gridspec) call this instead of `fig.clear()`.
        """
        self.fig.clear()
        self.fig.set_layout_engine("constrained")

    def draw(self) -> None:
        """Repaint synchronously, then freeze the computed layout.

        Use `draw()` (not `draw_idle()`): after re-rendering on a user action we
        want the new plot on screen immediately.

        After the constrained layout has positioned the panels, switch the
        layout engine off. Constrained layout otherwise re-solves on *every*
        draw, so interactive pan/zoom would reposition all panels as tick-label
        widths change -- the panels visibly jump. Freezing keeps the nice
        initial layout while making zoom/pan stable; the next render re-enables
        it via `reset_layout()`.
        """
        self._canvas.draw()
        if self.auto_layout:
            self.fig.set_layout_engine("none")
        self._canvas.flush_events()
