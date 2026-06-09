"""
GUI.WIDGETS.PILL_TOOLBAR: FLOATING STUDIO TOOLBAR
=================================================

A small floating, rounded ("pill") toolbar for the frameless Studio chrome,
holding the most-used actions (open / save / date range / reset). It is an
overlay child of the window's root container, anchored bottom-centre and
repositioned by the root on resize.

GUI-only presentation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QToolButton,
    QWidget,
)


class PillToolbar(QWidget):
    """Floating rounded action bar (styled via the ``#pilltoolbar`` selector)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pilltoolbar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 5, 10, 5)
        lay.setSpacing(4)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(26)
        shadow.setColor(QColor(0, 0, 0, 55))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def add_action(self, icon: QIcon, tooltip: str, slot) -> QToolButton:
        btn = QToolButton()
        btn.setIcon(icon)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.clicked.connect(slot)
        self.layout().addWidget(btn)
        return btn
