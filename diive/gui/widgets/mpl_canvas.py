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
from PySide6.QtCore import QEvent, QRectF, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


#: Quiet time after the last resize event before the layout is re-solved and
#: the figure rendered at the new size.
_RELAYOUT_DELAY_MS = 120


class _ResizeDeferringCanvas(FigureCanvasQTAgg):
    """`FigureCanvasQTAgg` that can hold back the render a resize queues.

    matplotlib's `resizeEvent` ends in `draw_idle()`, a full Agg render per
    resize event, so dragging a window edge or a splitter re-renders the whole
    figure on every step. The owning `MplCanvas` decides per event, from its
    resize_event callback, whether that render can wait until the size has
    settled. If so, `defer_resize_draw()` drops it, and until the settle render
    the canvas paints its last frame scaled to the new size.

    That frame is a copy taken at the first deferred resize, not the live Agg
    buffer: `get_renderer()` replaces the buffer with a blank one as soon as
    anything asks for a renderer at the new size.
    """

    # Class-level defaults, so the overrides below work even when matplotlib's
    # __init__ reaches them before this class could set instance attributes.
    _resizing = False       # inside resizeEvent
    _defer_draw = False     # skip the draw_idle() that ends this resizeEvent
    _last_frame = None      # QImage painted while the settle render is pending
    _settle_timer = None    # the owner's timer; active until the size settles

    def set_settle_timer(self, timer: QTimer) -> None:
        """The owner's settle timer; the scaled frame is painted only while it runs."""
        self._settle_timer = timer

    def settling(self) -> bool:
        """True while a deferred settle render is still to come."""
        return self._settle_timer is not None and self._settle_timer.isActive()

    def defer_resize_draw(self) -> None:
        """Skip the render of the resize event being handled.

        Call from a resize_event callback; the settle timer renders once
        instead. With nothing rendered yet there is no frame to show, so the
        render goes ahead."""
        if not self._resizing:
            return
        if self._last_frame is None:
            self._last_frame = self._snapshot()
        self._defer_draw = self._last_frame is not None

    def _snapshot(self) -> QImage | None:
        renderer = getattr(self, "renderer", None)
        if renderer is None:
            return None
        w, h = int(renderer.width), int(renderer.height)
        if w <= 0 or h <= 0:
            return None
        image = QImage(renderer.buffer_rgba(), w, h, 4 * w,
                       QImage.Format.Format_RGBA8888).copy()
        image.setDevicePixelRatio(self.device_pixel_ratio)
        return image

    def resizeEvent(self, event):
        self._resizing = True
        self._defer_draw = False
        try:
            super().resizeEvent(event)
        finally:
            self._resizing = False
            self._defer_draw = False

    def draw_idle(self):
        if self._resizing and self._defer_draw:
            return  # the settle timer renders at the final size
        super().draw_idle()

    def draw(self):
        super().draw()
        self._last_frame = None  # the buffer matches the widget size again

    def paintEvent(self, event):
        if self._last_frame is not None and not self._draw_pending:
            if self.settling():
                painter = QPainter(self)
                try:
                    painter.setRenderHint(
                        QPainter.RenderHint.SmoothPixmapTransform)
                    painter.drawImage(QRectF(self.rect()), self._last_frame)
                finally:
                    painter.end()
                return
            # The settle render will not come (its timer was stopped). Render
            # now: the Agg buffer does not match the widget size, and copying
            # it would paint a blank or misplaced image.
            self._draw_pending = True
        super().paintEvent(event)


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
        self._canvas = _ResizeDeferringCanvas(self.fig)
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
        self._fresh_layout = True  # no resize since the last render yet
        self._relayout_timer = QTimer(self)
        self._relayout_timer.setSingleShot(True)
        self._relayout_timer.setInterval(_RELAYOUT_DELAY_MS)
        self._relayout_timer.timeout.connect(self._relayout)
        self._canvas.set_settle_timer(self._relayout_timer)

        # The matplotlib canvas accepts wheel events, so a wheel over a plot
        # embedded in a scroll area (e.g. the results dashboards) would not
        # scroll the page. Filter the canvas's wheel events and forward them to
        # an enclosing scroll area instead. diive binds no wheel interaction on
        # the canvas, so nothing is lost.
        self._canvas.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._canvas and event.type() == QEvent.Type.Wheel:
            area = self._enclosing_scroll_area()
            if area is not None:
                QApplication.sendEvent(area.viewport(), event)
                return True  # consumed here; the scroll area handled it
        return super().eventFilter(obj, event)

    def _enclosing_scroll_area(self) -> QAbstractScrollArea | None:
        """Nearest ancestor scroll area, or None if this canvas isn't inside one
        (e.g. a standalone plotting tab — then the wheel is left to the canvas)."""
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QAbstractScrollArea):
                return w
            w = w.parentWidget()
        return None

    def save_dpi(self) -> int:
        """Current DPI selected for figure export (read by the Save action)."""
        return self._dpi_spin.value()

    def mpl_connect(self, event: str, callback):
        """Connect `callback` to a matplotlib canvas event (e.g. 'draw_event').

        matplotlib holds a bound method weakly, so connecting one does not keep
        its owner alive; a lambda would be held strongly."""
        return self._canvas.mpl_connect(event, callback)

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

    def show_message(self, text: str) -> None:
        """Clear the figure to a single blank axis and draw a centered message.

        The shared empty/error state for result canvases: a failed compute or a
        not-yet-run tab shows one centered line instead of stale panels. Replaces
        the hand-rolled ``ax.text(0.5, 0.5, ..., transform=ax.transAxes)`` idiom.
        Use only for the whole-canvas case; per-subpanel messages stay inline.
        """
        ax = self.new_axes(1)[0]
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        self.draw()

    def reset_history(self) -> None:
        """Reset the navigation toolbar's view history to the current view.

        Each render recreates the axes (``new_axes`` clears the figure), so the
        toolbar's per-axes view stack ends up referencing discarded axes and its
        Home/Back/Forward buttons do nothing. Clearing the stack and recording
        the current view makes Home reset to the freshly rendered (full) view.
        Call AFTER ``draw()`` so the captured axes positions are the solved ones.
        """
        if self._toolbar is not None:
            self._toolbar.update()        # clear the stale per-axes view stack
            self._toolbar.push_current()  # the current view becomes Home

    def push_view(self) -> None:
        """Record the current view as a new history entry (e.g. after restoring a
        zoom on top of the just-captured Home view, so Home still returns to it)."""
        if self._toolbar is not None:
            self._toolbar.push_current()

    def _on_resize(self, _event) -> None:
        """Re-solve the layout and re-render for the new size (see `_relayout`).

        The first resize after a render solves the layout and renders at once:
        a render often happens before the canvas has its real size (pre-show,
        or in a hidden tab), and that layout would show collapsed until the
        next solve. Later resizes (dragging the window edge or a splitter) are
        debounced: the solve and the render run once, when the size settles,
        instead of on every resize event. Until then a visible canvas paints
        its last frame scaled to the new size. A hidden canvas renders at once
        as before: Qt delivers its resize only when it is shown, as one event.
        """
        if not self.fig.axes:
            # Cleared for a render that is being built: nothing to lay out,
            # and the render's own draw() solves the layout at this size.
            return
        if self._fresh_layout:
            self._fresh_layout = False
            if self.auto_layout:  # else the plot manages its own layout
                self._solve_layout()
            return  # matplotlib's queued draw renders the new size at once
        self._relayout_timer.start()
        if self._canvas.isVisible():
            self._canvas.defer_resize_draw()

    def _relayout(self) -> None:
        """Debounced resize: solve the layout at the settled size and render."""
        self._relayout_timer.stop()  # also when a test emits the timeout
        if self.auto_layout and self.fig.axes:
            self._solve_layout()
        self._canvas.draw_idle()

    def _solve_layout(self) -> None:
        """Re-solve the constrained layout for the current size, then re-freeze.

        `draw()` freezes the layout so interactive pan/zoom stays stable, but a
        layout frozen at the initial (pre-show) canvas size would not match the
        real widget size. A resize is exactly when re-solving is wanted (and
        pan/zoom never resizes), so here we briefly re-enable the constrained
        engine, solve at the new size with `draw_without_rendering()`, then turn
        it off again -- leaving correct, frozen positions for the next repaint.
        """
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
        self._fresh_layout = True
        # A pending settle belongs to the old figure; the new render solves
        # and draws at the current size itself.
        self._relayout_timer.stop()

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
        # An idle draw queued before this one (e.g. by a resize while the
        # figure was being built) would only render the same figure again, a
        # full second draw. matplotlib skips a queued idle draw once its
        # pending flag is cleared.
        self._canvas._draw_pending = False
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
