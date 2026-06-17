"""
GUI.WIDGETS.MENU: STUDIO CONTEXT MENU
=====================================

A factory for ``QMenu``s styled to match the Studio look. A plain ``QMenu`` on the
frameless, translucent Studio window renders its popup with a black background /
shadow (unreadable); the Studio header menus avoid this by being frameless +
no-shadow + translucent with the ``#studiomenu`` object name (the QSS then rounds
them into a white card). :func:`studio_menu` packages that treatment so every
context menu (card actions, tab pin, variable-list right-click, …) looks the same.

Pure presentation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QMenu


class _StudioMenu(QMenu):
    """Studio-styled menu with a pressed effect on the clicked entry.

    Qt stylesheets expose no ``QMenu::item:pressed`` state, so the press feedback
    is painted: a translucent wash over the item under the cursor while the mouse
    button is held, cleared on release. The inset + radius mirror the
    ``#studiomenu::item`` QSS so the wash lines up with the rounded item.
    """

    _PRESS = QColor(0, 0, 0, 34)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pressed_action = None

    def mousePressEvent(self, event) -> None:
        act = self.actionAt(event.pos())
        self._pressed_action = (
            act if (act is not None and act.isEnabled()
                    and not act.isSeparator()) else None)
        if self._pressed_action is not None:
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pressed_action is not None:
            self._pressed_action = None
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._pressed_action is None:
            return
        rect = self.actionGeometry(self._pressed_action)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._PRESS)
        p.drawRoundedRect(QRectF(rect).adjusted(4, 0, -4, 0), 6, 6)
        p.end()


def studio_menu(parent=None) -> QMenu:
    """A ``QMenu`` styled as a rounded white Studio card (no black popup frame)."""
    m = _StudioMenu(parent)
    m.setObjectName("studiomenu")
    m.setWindowFlags(m.windowFlags()
                     | Qt.WindowType.FramelessWindowHint
                     | Qt.WindowType.NoDropShadowWindowHint)
    m.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    return m
