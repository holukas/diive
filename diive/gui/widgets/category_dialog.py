"""
GUI.WIDGETS.CATEGORY_DIALOG: MANAGE EVENT CATEGORIES
====================================================

A small dialog to define the event **categories** and the colour each one is
drawn in (on the event cards and on the plot overlays). It edits the app-wide
``events.manager`` category palette (``{name: hex}``): add a category, rename it,
recolour it, or remove it — with one rule, the **last remaining** category can't
be removed, so events can always be categorised.

Pure presentation: categories/colours are GUI configuration, like the theme; the
event model itself lives in the library.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from diive.gui import events as events_store


def _swatch(color: str) -> QIcon:
    """A small rounded colour chip icon for the list row."""
    pix = QPixmap(20, 20)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(QColor("#CFD8DC"))
    p.drawRoundedRect(2, 2, 16, 16, 5, 5)
    p.end()
    return QIcon(pix)


class CategoryDialog(QDialog):
    """Add / rename / recolour / remove event categories (edits ``events.manager``)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage event categories")
        self.setMinimumSize(380, 340)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        hint = QLabel(
            "Categories colour your events. Rename or recolour any of them; the "
            "last one can't be deleted.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #607D8B;")
        outer.addWidget(hint)

        self.list = QListWidget()
        self.list.setIconSize(QSize(20, 20))
        self.list.setStyleSheet("QListWidget { border: 1px solid #E0E4E7;"
                                " border-radius: 8px; padding: 4px; }")
        self.list.itemDoubleClicked.connect(lambda *_: self._rename())
        self.list.currentRowChanged.connect(lambda *_: self._sync_buttons())
        outer.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self._add)
        self.rename_btn = QPushButton("Rename…")
        self.rename_btn.clicked.connect(self._rename)
        self.color_btn = QPushButton("Colour…")
        self.color_btn.clicked.connect(self._recolor)
        self.del_btn = QPushButton("Remove")
        self.del_btn.clicked.connect(self._remove)
        for b in (self.add_btn, self.rename_btn, self.color_btn, self.del_btn):
            row.addWidget(b)
        row.addStretch(1)
        close = QPushButton("Close")
        close.setDefault(True)
        close.clicked.connect(self.accept)
        row.addWidget(close)
        outer.addLayout(row)

        self._reload()

    # --- helpers -------------------------------------------------------
    def _reload(self, select: str | None = None) -> None:
        self.list.clear()
        # Insertion order (not sorted) so category1/2/3 and renames keep position.
        for name, color in events_store.manager.categories.items():
            item = QListWidgetItem(_swatch(color), name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(item)
            if name == select:
                self.list.setCurrentItem(item)
        if select is None and self.list.count():
            self.list.setCurrentRow(0)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        has = self.list.currentItem() is not None
        # Never allow deleting the last remaining category.
        self.del_btn.setEnabled(has and len(events_store.manager.categories) > 1)
        self.rename_btn.setEnabled(has)
        self.color_btn.setEnabled(has)

    def _selected_name(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    # --- actions -------------------------------------------------------
    def _add(self) -> None:
        name = events_store.manager.add_category()
        self._reload(select=name)

    def _rename(self) -> None:
        old = self._selected_name()
        if not old:
            return
        new, ok = QInputDialog.getText(self, "Rename category",
                                       "New name:", text=old)
        if ok and events_store.manager.rename_category(old, new.strip()):
            self._reload(select=new.strip())

    def _recolor(self) -> None:
        name = self._selected_name()
        if not name:
            return
        current = events_store.manager.categories.get(name, "#42A5F5")
        chosen = QColorDialog.getColor(QColor(current), self,
                                       f"Colour for '{name}'")
        if chosen.isValid():
            events_store.manager.set_category(name, chosen.name())
            self._reload(select=name)

    def _remove(self) -> None:
        name = self._selected_name()
        if name:
            events_store.manager.remove_category(name)
            self._reload()
