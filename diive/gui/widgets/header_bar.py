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

from PySide6.QtCore import QEvent, QPoint, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QWidget,
)

from diive.gui import icons, theme


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
        # Transparent background so the wordmark sits cleanly on the card's soft
        # gradient — the global `QWidget { background: CANVAS }` rule would
        # otherwise paint a white box behind it.
        self._wordmark.setStyleSheet("background: transparent;")
        lay.addWidget(self._wordmark)

        lay.addSpacing(10)
        self._menus = QHBoxLayout()
        self._menus.setSpacing(2)
        lay.addLayout(self._menus)

        lay.addStretch(1)
        self._title = QLabel("")
        self._title.setFont(theme.manager.tracked_font())
        self._title.setStyleSheet("color: #6B7780; background: transparent;")
        lay.addWidget(self._title)
        lay.addStretch(1)

        # Database-connection pill, far right. Hidden until connected, then a
        # clearly-coloured green "Connected: ..." badge.
        self._db_pill = QLabel("")
        self._db_pill.setFont(theme.manager.tracked_font())
        self._db_pill.setVisible(False)
        lay.addWidget(self._db_pill)

        # Window controls (minimize / maximize-restore / close) in the far-right
        # corner: a frameless window has no native title-bar buttons.
        lay.addSpacing(8)
        self._maxed = False           # filled-to-work-area vs. normal geometry
        self._normal_geo = None       # geometry to restore to
        self._build_window_controls(lay)

    # --- window controls (frameless title-bar buttons) ---
    def _control_button(self, icon, tooltip: str, slot, *, danger=False) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("winclose" if danger else "winctl")
        btn.setIcon(icon)
        btn.setIconSize(QSize(16, 16))
        btn.setFixedSize(QSize(38, 28))
        btn.setAutoRaise(True)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        hover = "#E81123" if danger else "rgba(0, 0, 0, 0.08)"
        btn.setStyleSheet(
            "QToolButton { border: none; background: transparent; border-radius: 6px; }"
            f"QToolButton:hover {{ background: {hover}; }}"
            + theme.manager.tooltip_qss())
        btn.clicked.connect(slot)
        return btn

    def _build_window_controls(self, lay: QHBoxLayout) -> None:
        self._btn_min = self._control_button(
            icons.minimize_icon(), "Minimize", self._minimize)
        self._btn_max = self._control_button(
            icons.maximize_icon(), "Maximize", self._toggle_max)
        self._btn_close = self._control_button(
            icons.close_icon(), "Close", lambda: self.window().close(), danger=True)
        # White × on the red hover; restore the ink × when the cursor leaves.
        self._btn_close.installEventFilter(self)
        for b in (self._btn_min, self._btn_max, self._btn_close):
            lay.addWidget(b)

    def _minimize(self) -> None:
        self.window().showMinimized()

    def _toggle_max(self) -> None:
        """Toggle between filling the screen work area and the normal geometry.

        Uses ``availableGeometry`` (not ``showMaximized``) because a *frameless*
        maximize on Windows covers the taskbar and clips the active tab's bottom
        (same rationale as MainWindow.show_filling_workarea)."""
        win = self.window()
        if self._maxed:
            if self._normal_geo is not None:
                win.setGeometry(self._normal_geo)
            self._maxed = False
        else:
            self._normal_geo = win.geometry()
            screen = win.screen() or QApplication.primaryScreen()
            if screen is not None:
                win.setGeometry(screen.availableGeometry())
            self._maxed = True
        self._btn_max.setIcon(
            icons.restore_icon() if self._maxed else icons.maximize_icon())
        self._btn_max.setToolTip("Restore" if self._maxed else "Maximize")

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
        # Close button: white × on the red hover background, ink × otherwise.
        if obj is getattr(self, "_btn_close", None):
            if et == QEvent.Type.Enter:
                obj.setIcon(icons.close_icon("#FFFFFF"))
            elif et == QEvent.Type.Leave:
                obj.setIcon(icons.close_icon())
            return False
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

    def set_db_status(self, connected: bool, text: str = "") -> None:
        """Show/hide the database-connection pill in the header.

        When *connected*, a green badge reads *text* (e.g. "Connected:
        InfluxDB @ ..."); otherwise the pill is hidden.
        """
        if not connected:
            self._db_pill.setVisible(False)
            return
        self._db_pill.setText(theme.manager.label_text(text))
        self._db_pill.setStyleSheet(
            "QLabel { background: #2E7D32; color: white; border-radius: 9px; "
            "padding: 3px 10px; }" + theme.manager.tooltip_qss())
        self._db_pill.setToolTip(text)
        self._db_pill.setVisible(True)

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
