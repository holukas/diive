"""
GUI.WIDGETS.VARIABLE_LIST: VARIABLE PICKER
==========================================

`VariableList` is a `QListWidget` that reports whether Ctrl was held on click.
It emits `selected(name, additive)` on every item click:

- plain click  -> `additive=False` (caller resets to a single panel)
- Ctrl + click -> `additive=True`  (caller appends another panel)

Ctrl+click is consumed so it does not disturb the list's own selection
highlight, which tracks the most recent plain click.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QListWidget


class VariableList(QListWidget):
    """Variable picker reporting the Ctrl modifier on each click."""

    #: Emitted on item click as (variable_name, additive); additive is True
    #: when Ctrl was held.
    selected = Signal(str, bool)

    def _name(self, item) -> str:
        """Variable name for an item: the UserRole value, or the text."""
        return item.data(Qt.ItemDataRole.UserRole) or item.text()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        item = self.itemAt(event.position().toPoint())
        if item is None:
            super().mousePressEvent(event)
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Consume the event so the selection highlight is not changed.
            self.selected.emit(self._name(item), True)
            return
        super().mousePressEvent(event)
        self.selected.emit(self._name(item), False)
