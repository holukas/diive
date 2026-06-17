"""
GUI.WIDGETS.FRAMELESS: EDGE-RESIZE FOR THE FRAMELESS STUDIO WINDOW
=================================================================

A frameless window loses the native resize grips. `FramelessResizeHelper`
restores them: it watches a "grip" widget (the window's root container) for
mouse activity near its edges/corners and hands off to the native
``QWindow.startSystemResize`` so the OS does the actual resize.

Because the helper only sees events delivered to the grip widget itself (child
widgets consume their own), the live resize zone is exactly the exposed margin
frame around the content — which is also where the soft gradient backdrop shows.

GUI-only presentation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt


class FramelessResizeHelper(QObject):
    """Native edge/corner resize for a frameless window via an event filter."""

    MARGIN = 8  # px from an edge that counts as a resize zone

    def __init__(self, window, grip) -> None:
        super().__init__(grip)
        self._window = window
        self._grip = grip
        grip.setMouseTracking(True)
        grip.installEventFilter(self)

    def _edges(self, pos) -> Qt.Edge:
        r = self._grip.rect()
        m = self.MARGIN
        edges = Qt.Edge(0)
        if pos.x() <= m:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= r.width() - m:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= m:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= r.height() - m:
            edges |= Qt.Edge.BottomEdge
        return edges

    def _cursor_for(self, edges) -> Qt.CursorShape:
        # PySide6 flag enums aren't int()-able; use `.value` for the bit tests.
        e = edges.value
        left, right = e & Qt.Edge.LeftEdge.value, e & Qt.Edge.RightEdge.value
        top, bottom = e & Qt.Edge.TopEdge.value, e & Qt.Edge.BottomEdge.value
        if (left and top) or (right and bottom):
            return Qt.CursorShape.SizeFDiagCursor
        if (right and top) or (left and bottom):
            return Qt.CursorShape.SizeBDiagCursor
        if left or right:
            return Qt.CursorShape.SizeHorCursor
        if top or bottom:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def eventFilter(self, obj, event) -> bool:
        et = event.type()
        if et == QEvent.Type.MouseMove:
            edges = self._edges(event.position().toPoint())
            if edges.value:
                self._grip.setCursor(self._cursor_for(edges))
            else:
                # Don't leave a sticky resize cursor on the grip: child widgets
                # without their own cursor inherit the grip's, so a leftover
                # SizeHorCursor would show as a resize cursor over buttons.
                self._grip.unsetCursor()
        elif et == QEvent.Type.Leave:
            # Moving onto a child widget fires Leave on the grip — reset so the
            # resize cursor never persists outside the edge margin.
            self._grip.unsetCursor()
        elif (et == QEvent.Type.MouseButtonPress
              and event.button() == Qt.MouseButton.LeftButton):
            edges = self._edges(event.position().toPoint())
            if edges.value:
                handle = self._window.windowHandle()
                if handle is not None:
                    handle.startSystemResize(edges)
                    return True
        return super().eventFilter(obj, event)
