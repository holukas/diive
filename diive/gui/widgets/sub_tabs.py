"""
GUI.WIDGETS.SUB_TABS: STANDARDIZED IN-TAB SUB-NAVIGATION
========================================================

A horizontal segmented control over a ``QStackedWidget`` — the shared look for
tabs that produce a lot of output and split their content across sections
(e.g. a *Model* configuration page and a *Results* page of plots/tables). Only
the active page takes space, so a results page can use the full tab width.

Same visual language as the stepwise-screening inspector segments (pill buttons,
accent fill on the active page), extracted here so every output-heavy tab uses
one identical control instead of re-deriving it. The segment chips repaint when
the appearance theme changes.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from diive.gui import theme


class _CornerSeparator(QWidget):
    """A slim vertical hairline that fades out at both ends — a soft, modern
    divider between groups of items in the segment row."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(3)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        theme.manager.changed.connect(self.update)

    def paintEvent(self, _event) -> None:
        base = QColor(theme.manager.tokens.get("BORDER", "#D9D9D6"))
        h = self.height()
        inset = int(h * 0.18)  # leave the line a touch shorter than the pills
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, inset, 0, h - inset)
        edge = QColor(base); edge.setAlpha(0)
        mid = QColor(base); mid.setAlpha(170)
        grad.setColorAt(0.0, edge)
        grad.setColorAt(0.5, mid)
        grad.setColorAt(1.0, edge)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        x = (self.width() - 1.5) / 2.0
        p.drawRoundedRect(x, inset, 1.5, h - 2 * inset, 0.75, 0.75)


class SubTabs(QWidget):
    """Segmented sub-tab navigation: one pill button per page over a stack.

    Build it, then ``add_page(label, widget)`` for each section. ``changed``
    fires with the new index whenever the active page switches."""

    changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: list[QPushButton] = []

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        self._segrow = QHBoxLayout()
        self._segrow.setSpacing(4)
        self._segrow.addStretch(1)  # gap between the left pills and right corner
        v.addLayout(self._segrow)

        self._stack = QStackedWidget()
        v.addWidget(self._stack, stretch=1)

        self._apply_style()
        theme.manager.changed.connect(self._apply_style)

    def add_page(self, label: str, widget: QWidget) -> int:
        """Append a page; returns its index."""
        idx = self._stack.count()
        b = QPushButton(label)
        b.setCheckable(True)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(lambda _c=False, i=idx: self.set_page(i))
        self._buttons.append(b)
        # Insert before the trailing stretch so pills stay left-aligned.
        self._segrow.insertWidget(self._segrow.count() - 1, b)
        self._stack.addWidget(widget)
        self._apply_style()
        if idx == 0:
            self.set_page(0)
        return idx

    def add_corner_widget(self, widget: QWidget) -> None:
        """Add a widget (e.g. an action button) immediately to the right of the
        page pills, before the trailing stretch."""
        self._segrow.insertWidget(self._segrow.count() - 1, widget)

    def add_corner_separator(self, spacing: int = 8) -> None:
        """Add a slim vertical divider between corner items (e.g. to set the
        action buttons apart from the page pills)."""
        self._segrow.insertSpacing(self._segrow.count() - 1, spacing)
        sep = _CornerSeparator()
        self._segrow.insertWidget(self._segrow.count() - 1, sep)
        self._segrow.insertSpacing(self._segrow.count() - 1, spacing)

    def set_page(self, idx: int) -> None:
        if idx < 0 or idx >= self._stack.count():
            return
        if self._stack.currentIndex() == idx and self._buttons[idx].isChecked():
            return
        self._stack.setCurrentIndex(idx)
        for i, b in enumerate(self._buttons):
            b.setChecked(i == idx)
        self.changed.emit(idx)

    def current_index(self) -> int:
        return self._stack.currentIndex()

    def set_label(self, idx: int, text: str) -> None:
        """Update a page's pill text (e.g. to show a count badge)."""
        if 0 <= idx < len(self._buttons):
            self._buttons[idx].setText(text)

    def _apply_style(self) -> None:
        accent = theme.manager.tokens.get("ACCENT", "#3A4D5C")
        border = theme.manager.tokens.get("BORDER", "#E6E6E3")
        qss = (
            f"QPushButton {{ padding: 6px 16px; border: 0.5px solid {border}; "
            f"border-radius: 6px; background: transparent; }} "
            f"QPushButton:checked {{ background: {accent}; color: white; "
            f"border-color: {accent}; }}")
        for b in self._buttons:
            b.setStyleSheet(qss)
