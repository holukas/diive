"""
GUI.WIDGETS.NOTES_WALL: STICKY-NOTE PINBOARD
============================================

A small "wall" of draggable sticky-note cards for the Project settings tab. Each
card has a bold, larger header and a free-text body, can be recoloured from a
sticky-note palette, dragged to any position, resized, and deleted. The whole
arrangement serialises to a list of plain dicts so it travels with the project
(stored in ``site.manager.notes``).

Presentation only — no domain logic. The card text colour is chosen for contrast
against the card background (WCAG luminance rule, same as the plotting helpers).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

#: Sticky-note palette (Material 200-level), plus white. First entry is default.
_PALETTE = [
    "#FFF59D",  # yellow
    "#FFCC80",  # orange
    "#A5D6A7",  # green
    "#80DEEA",  # cyan
    "#90CAF9",  # blue
    "#F48FB1",  # pink
    "#CE93D8",  # purple
    "#E0E0E0",  # grey
    "#FFFFFF",  # white
]

_DEFAULT_W, _DEFAULT_H = 220, 160
_MIN_W, _MIN_H = 140, 90


def _ink_for(bg: str) -> str:
    """Black or white, whichever contrasts better with ``bg`` (WCAG luminance)."""
    c = QColor(bg)
    lum = (0.299 * c.redF() + 0.587 * c.greenF() + 0.114 * c.blueF())
    return "#FFFFFF" if lum < 0.5 else "#1A1A1A"


class _ResizeGrip(QWidget):
    """Bottom-right corner grip that resizes its parent card on drag."""

    def __init__(self, card: "_NoteCard") -> None:
        super().__init__(card)
        self._card = card
        self.setFixedSize(14, 14)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self._origin = None
        self._start = None

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        pen = QPen(QColor(self._card.ink))
        pen.setWidth(1)
        p.setPen(pen)
        for off in (4, 8, 12):  # three diagonal ticks
            p.drawLine(self.width() - off, self.height() - 2,
                       self.width() - 2, self.height() - off)
        p.end()

    def mousePressEvent(self, event) -> None:
        self._origin = event.globalPosition().toPoint()
        self._start = self._card.size()

    def mouseMoveEvent(self, event) -> None:
        if self._origin is None:
            return
        delta = event.globalPosition().toPoint() - self._origin
        w = max(_MIN_W, self._start.width() + delta.x())
        h = max(_MIN_H, self._start.height() + delta.y())
        self._card.resize(w, h)

    def mouseReleaseEvent(self, _event) -> None:
        self._origin = None
        self._card.wall.notify()


class _NoteCard(QFrame):
    """One draggable, editable, recolourable sticky note."""

    def __init__(self, data: dict, wall: "NotesWall") -> None:
        super().__init__(wall.canvas)
        self.wall = wall
        self.color = str(data.get("color") or _PALETTE[0])
        self.ink = _ink_for(self.color)
        self._drag_origin = None  # set while dragging the header
        self._drag_start = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 8)
        outer.setSpacing(4)

        # Header bar: drag handle + colour + delete. The whole bar is the drag
        # grip (the title sits below it, so editing the title never moves the card).
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        self._grip = QLabel("⠿")
        self._grip.setCursor(Qt.CursorShape.OpenHandCursor)
        bar.addWidget(self._grip)
        bar.addStretch(1)
        self._color_btn = QToolButton()
        self._color_btn.setText("●")
        self._color_btn.setAutoRaise(True)
        self._color_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._color_btn.clicked.connect(self._pick_color)
        bar.addWidget(self._color_btn)
        self._close_btn = QToolButton()
        self._close_btn.setText("✕")
        self._close_btn.setAutoRaise(True)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.clicked.connect(self._delete)
        bar.addWidget(self._close_btn)
        outer.addLayout(bar)

        self.title = QLineEdit(str(data.get("title") or ""))
        self.title.setPlaceholderText("Title")
        tf = self.title.font()
        tf.setBold(True)
        tf.setPointSizeF(tf.pointSizeF() + 3)
        self.title.setFont(tf)
        self.title.setFrame(False)
        self.title.textChanged.connect(self.wall.notify)
        outer.addWidget(self.title)

        self.body = QPlainTextEdit(str(data.get("body") or ""))
        self.body.setPlaceholderText("Note…")
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.textChanged.connect(self.wall.notify)
        outer.addWidget(self.body, stretch=1)

        self._resizer = _ResizeGrip(self)

        x = int(data.get("x", 24))
        y = int(data.get("y", 24))
        w = int(data.get("w", _DEFAULT_W))
        h = int(data.get("h", _DEFAULT_H))
        self.setGeometry(x, y, max(_MIN_W, w), max(_MIN_H, h))
        self._apply_color()

    # --- appearance ---
    def _apply_color(self) -> None:
        self.ink = _ink_for(self.color)
        # Subtle darker border from the fill so the card reads as a tile.
        border = QColor(self.color).darker(115).name()
        self.setStyleSheet(
            f"_NoteCard {{ background: {self.color}; border: 1px solid {border};"
            f" border-radius: 8px; }}"
            f" QLineEdit, QPlainTextEdit {{ background: transparent; color: {self.ink};"
            f" border: none; }}"
            f" QLabel, QToolButton {{ color: {self.ink}; background: transparent;"
            f" border: none; font-weight: bold; }}")
        self._resizer.update()

    def _pick_color(self) -> None:
        menu = QMenu(self)
        for hexc in _PALETTE:
            act = menu.addAction(hexc)
            sw = QPixmap(16, 16)
            sw.fill(QColor(hexc))
            act.setIcon(QIcon(sw))
            act.triggered.connect(lambda _checked=False, c=hexc: self._set_color(c))
        menu.addSeparator()
        custom = menu.addAction("Custom…")
        custom.triggered.connect(self._pick_custom)
        menu.exec(self._color_btn.mapToGlobal(self._color_btn.rect().bottomLeft()))

    def _pick_custom(self) -> None:
        c = QColorDialog.getColor(QColor(self.color), self, "Card colour")
        if c.isValid():
            self._set_color(c.name())

    def _set_color(self, hexc: str) -> None:
        self.color = hexc
        self._apply_color()
        self.wall.notify()

    def _delete(self) -> None:
        self.wall.remove_card(self)

    # --- dragging (from the header bar / grip) ---
    def _in_header(self, pos: QPoint) -> bool:
        return pos.y() <= 22 and self._color_btn.geometry().left() > pos.x()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._in_header(event.position().toPoint()):
            self._drag_origin = event.globalPosition().toPoint()
            self._drag_start = self.pos()
            self.raise_()
            self._grip.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin is None:
            return super().mouseMoveEvent(event)
        delta = event.globalPosition().toPoint() - self._drag_origin
        new = self._drag_start + delta
        # Clamp inside the wall canvas so cards can't be lost off-screen.
        cw, ch = self.wall.canvas.width(), self.wall.canvas.height()
        nx = max(0, min(new.x(), max(0, cw - self.width())))
        ny = max(0, min(new.y(), max(0, ch - self.height())))
        self.move(nx, ny)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_origin is not None:
            self._drag_origin = None
            self._grip.setCursor(Qt.CursorShape.OpenHandCursor)
            self.wall.notify()
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resizer.move(self.width() - self._resizer.width(),
                           self.height() - self._resizer.height())

    def data(self) -> dict:
        g = self.geometry()
        return {"title": self.title.text(), "body": self.body.toPlainText(),
                "color": self.color, "x": g.x(), "y": g.y(),
                "w": g.width(), "h": g.height()}


class _WallSignals(QObject):
    changed = Signal()


class NotesWall(QWidget):
    """A pinboard of draggable sticky notes; serialises to a list of dicts."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cards: list[_NoteCard] = []
        self._sig = _WallSignals()
        #: Emitted on any add/edit/move/resize/recolour/delete.
        self.changed = self._sig.changed

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        bar = QHBoxLayout()
        self._add_btn = QToolButton()
        self._add_btn.setText("+  Add note")
        self._add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_btn.clicked.connect(self.add_note)
        bar.addWidget(self._add_btn)
        bar.addStretch(1)
        outer.addLayout(bar)

        # Free-positioning canvas (no layout — cards are placed by geometry).
        self.canvas = QWidget()
        self.canvas.setObjectName("noteswall")
        self.canvas.setMinimumHeight(240)
        self.canvas.setStyleSheet(
            "#noteswall { background: #FAFAF7; border: 1px dashed #D8D8D2;"
            " border-radius: 10px; }")
        outer.addWidget(self.canvas, stretch=1)

        self._hint = QLabel("Pin small notes here — click “Add note”.", self.canvas)
        self._hint.setStyleSheet("color: #9AA0A6; background: transparent;")
        self._hint.move(16, 14)

    # --- card lifecycle ---
    def add_note(self) -> None:
        n = len(self._cards)
        offset = 18 * (n % 6)
        self._add_card({"x": 24 + offset, "y": 24 + offset})
        self.notify()

    def _add_card(self, data: dict) -> _NoteCard:
        card = _NoteCard(data, self)
        card.show()
        self._cards.append(card)
        self._hint.setVisible(False)
        return card

    def remove_card(self, card: _NoteCard) -> None:
        if card in self._cards:
            self._cards.remove(card)
            card.setParent(None)
            card.deleteLater()
        self._hint.setVisible(not self._cards)
        self.notify()

    def notify(self) -> None:
        self.changed.emit()

    # --- state (project persistence) ---
    def state(self) -> list:
        return [c.data() for c in self._cards]

    def set_state(self, notes) -> None:
        """Rebuild the wall from a list of card dicts (replaces current cards)."""
        for c in list(self._cards):
            c.setParent(None)
            c.deleteLater()
        self._cards.clear()
        for nd in (notes or []):
            if isinstance(nd, dict):
                self._add_card(nd)
        self._hint.setVisible(not self._cards)
