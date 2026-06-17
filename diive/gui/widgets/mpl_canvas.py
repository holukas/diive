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

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class _SaveDpiToolbar(NavigationToolbar2QT):
    """Navigation toolbar whose Save action exports at a user-chosen DPI.

    The embedded figure is sized for the screen, so a plain save would bake in
    the (low) screen DPI. `save_figure` temporarily raises `savefig.dpi` to the
    value the canvas's DPI spinbox reports, so exported images are crisp.
    """

    def __init__(self, canvas, parent, dpi_getter) -> None:
        super().__init__(canvas, parent, coordinates=False)
        self._dpi_getter = dpi_getter

    def save_figure(self, *args):
        old = mpl.rcParams["savefig.dpi"]
        mpl.rcParams["savefig.dpi"] = self._dpi_getter()
        try:
            return super().save_figure(*args)
        finally:
            mpl.rcParams["savefig.dpi"] = old


class MplCanvas(QWidget):
    """Embeddable matplotlib canvas with a navigation toolbar.

    Render by calling `new_axes(ncols)` to get a fresh row of axes (the figure
    is cleared first), drawing into them, then `draw()`.
    """

    def __init__(self, parent: QWidget | None = None, *,
                 show_toolbar: bool = True) -> None:
        """Embeddable matplotlib canvas.

        ``show_toolbar=False`` omits the bottom navigation/DPI/hover row. The
        toolbar's many buttons impose a wide minimum width, so dropping it lets
        the canvas shrink into a narrow side panel (e.g. the gap-filling SHAP
        panel)."""
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
        # DPI spinbox for figure export; the toolbar's Save reads it (see
        # _SaveDpiToolbar). 150 is a sensible default above typical screen DPI.
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(50, 600)
        self._dpi_spin.setSingleStep(50)
        self._dpi_spin.setValue(150)
        self._dpi_spin.setToolTip("DPI used when saving the figure")
        self._toolbar = _SaveDpiToolbar(self._canvas, self, self.save_dpi) if show_toolbar else None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas, stretch=1)

        # Value-under-cursor tooltip (line snap + heatmap cell). Attaches to the
        # canvas events and works across re-renders; see widgets/hover.py.
        from diive.gui.widgets.hover import HoverAnnotator
        self.hover = HoverAnnotator(self)

        # Bottom row: a "Hover values" toggle next to the navigation toolbar,
        # both pushed to the right corner by a stretch. Omitted when the toolbar
        # is hidden (its width minimum would otherwise stop the canvas going narrow).
        if show_toolbar:
            self._hover_toggle = QCheckBox("Hover values")
            self._hover_toggle.setChecked(True)
            self._hover_toggle.toggled.connect(self.hover.set_enabled)
            bottom = QHBoxLayout()
            bottom.setContentsMargins(0, 0, 4, 2)
            bottom.addStretch(1)
            bottom.addWidget(QLabel("Save DPI"))
            bottom.addWidget(self._dpi_spin)
            bottom.addWidget(self._hover_toggle)
            bottom.addWidget(self._toolbar)
            layout.addLayout(bottom)
        else:
            self._hover_toggle = None

        # Re-solve the (frozen) constrained layout whenever the canvas resizes,
        # so panels adapt to the real widget size. Pan/zoom doesn't resize, so
        # it stays frozen -- see draw()/_on_resize.
        self._canvas.mpl_connect("resize_event", self._on_resize)

    def save_dpi(self) -> int:
        """Current DPI selected for figure export (read by the Save action)."""
        return self._dpi_spin.value()

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
            # Two passes: constrained layout solves iteratively, and a single
            # pass can leave panels collapsed when several axes carry wide tick
            # labels (e.g. a zoomed datetime range linked across panels). A
            # second solve lets it converge.
            self.fig.draw_without_rendering()
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

    def draw_idle(self) -> None:
        """Schedule a repaint without touching the (frozen) layout engine.

        For incremental updates (e.g. the Overview's live zoom sync) that repaint
        a couple of panels but must NOT re-freeze or re-solve the constrained
        layout the way `draw()` does -- calling `draw()` here would flip the
        layout engine off and can abort an in-progress resize re-solve.
        """
        self._canvas.draw_idle()
