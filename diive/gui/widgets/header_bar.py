"""
GUI.WIDGETS.HEADER_BAR: STUDIO CUSTOM HEADER
============================================

The custom title bar for the frameless "Studio" window chrome (see
`diive.gui.theme` presets): a "diive" wordmark on the left, then the full menu
tree as inline dropdown buttons (File ⌄, Data ⌄, Plot ⌄, …), and a centred
title. Replaces the native title bar + menu bar in Studio mode only.

The menus open **on hover** (no click needed) and switch as the cursor moves
across the bar — implemented like a classic menu bar: each button's ``Enter``
pops its menu up (`QMenu.popup`, non-blocking), and while a menu is open its
own mouse-move events drive switching to whichever button the cursor is over.
Dragging an empty area of the bar moves the window (native ``startSystemMove``).
Labels use the active preset's tracked/uppercase typography (`theme.manager`).

GUI-only presentation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QToolButton, QWidget

from diive.gui import theme


class StudioHeaderBar(QWidget):
    """Custom header bar for the frameless Studio chrome."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(46)

        self._buttons: list[QToolButton] = []
        self._btn_menu: dict[QToolButton, QMenu] = {}

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 6, 14, 6)
        lay.setSpacing(6)

        self._wordmark = QLabel("diive")
        wf = theme.manager.tracked_font(point_delta=4.0)
        wf.setBold(True)
        self._wordmark.setFont(wf)
        lay.addWidget(self._wordmark)

        lay.addSpacing(10)
        self._menus = QHBoxLayout()
        self._menus.setSpacing(2)
        lay.addLayout(self._menus)

        lay.addStretch(1)
        self._title = QLabel("")
        self._title.setFont(theme.manager.tracked_font())
        self._title.setStyleSheet("color: #6B7780;")
        lay.addWidget(self._title)
        lay.addStretch(1)

    def add_menu(self, label: str, menu: QMenu) -> QToolButton:
        """Add a top-level dropdown button (e.g. "File ⌄") wired to `menu`.

        The menu is kept on the button (so `btn.menu()` works) but opening is
        driven manually for hover support; the built-in delayed/instant popup
        is left alone (a plain click also opens it).
        """
        btn = QToolButton()
        btn.setObjectName("headermenu")
        btn.setText(theme.manager.label_text(label.replace("&", "")) + "  ⌄")
        btn.setFont(theme.manager.tracked_font())
        btn.setAutoRaise(True)
        btn.setMenu(menu)
        btn.clicked.connect(lambda _checked=False, b=btn: self._show_menu(b))
        btn.installEventFilter(self)   # Enter -> open on hover
        menu.installEventFilter(self)  # MouseMove -> switch while open
        self._buttons.append(btn)
        self._btn_menu[btn] = menu
        self._menus.addWidget(btn)
        return btn

    def _show_menu(self, btn: QToolButton) -> None:
        """Open `btn`'s menu, closing any other open header menu first."""
        menu = self._btn_menu.get(btn)
        if menu is None:
            return
        for other in self._btn_menu.values():
            if other is not menu and other.isVisible():
                other.close()
        if menu.isVisible():
            return
        menu.popup(btn.mapToGlobal(QPoint(0, btn.height() + 2)))

    def eventFilter(self, obj, event) -> bool:
        et = event.type()
        # Hover over a header button opens its menu (when none is grabbing).
        if et == QEvent.Type.Enter and obj in self._buttons:
            self._show_menu(obj)
            return False
        # While a menu is open it holds the mouse grab, so sibling buttons never
        # see Enter; use the open menu's mouse-move to switch to a hovered button.
        if et == QEvent.Type.MouseMove and isinstance(obj, QMenu):
            gpos = event.globalPosition().toPoint()
            for b in self._buttons:
                if self._btn_menu[b].isVisible():
                    continue
                if b.rect().contains(b.mapFromGlobal(gpos)):
                    self._show_menu(b)
                    break
        return super().eventFilter(obj, event)

    def set_title(self, text: str) -> None:
        self._title.setText(theme.manager.label_text(text))

    # --- frameless window drag ---
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemMove()
                event.accept()
                return
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        # Soft hairline separator along the bottom edge.
        p = QPainter(self)
        p.setPen(QPen(QColor(theme.manager.tokens.get("BORDER", "#E6E6E3"))))
        y = self.height() - 1
        p.drawLine(8, y, self.width() - 8, y)
        p.end()
