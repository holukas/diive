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

from html import escape

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QListWidget, QToolTip

from diive.core.metadata import ORIGINAL
from diive.gui import metadata_store


class VariableList(QListWidget):
    """Variable picker reporting the Ctrl modifier on each click."""

    #: Emitted on item click as (variable_name, additive); additive is True
    #: when Ctrl was held.
    selected = Signal(str, bool)

    def _name(self, item) -> str:
        """Variable name for an item: the UserRole value, or the text."""
        return item.data(Qt.ItemDataRole.UserRole) or item.text()

    def event(self, e):
        """Show a rich metadata tooltip (origin, tags, provenance) on hover.

        Built on demand from the live store so it reflects edits made elsewhere;
        the first variable-list hover in the GUI.
        """
        if e.type() == QEvent.Type.ToolTip:
            item = self.itemAt(e.pos())
            if item is not None:
                QToolTip.showText(e.globalPos(), self._tooltip_html(self._name(item)), self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)

    @staticmethod
    def _tooltip_html(name: str) -> str:
        """Rich-text tooltip summarising a variable's metadata."""
        md = metadata_store.manager.store.peek(name)
        rows = [f"<b>{escape(str(name))}</b>"]
        if md is None:
            return rows[0]
        rows.append(f"<span style='color:#90A4AE'>origin:</span> {escape(md.origin)}")
        if md.parents:
            rows.append("<span style='color:#90A4AE'>from:</span> "
                        + escape(", ".join(md.parents)))
        shown_tags = sorted(t for t in md.tags if t != ORIGINAL)
        if shown_tags:
            rows.append("<span style='color:#90A4AE'>tags:</span> "
                        + escape(", ".join(shown_tags)))
        if md.description:
            rows.append(f"<i>{escape(md.description)}</i>")
        if md.provenance:
            steps = "".join(
                f"<li>{escape(p.describe())}"
                + (f" <span style='color:#90A4AE'>· {escape(p.timestamp)}</span>"
                   if p.timestamp else "")
                + "</li>"
                for p in md.provenance)
            rows.append("<span style='color:#90A4AE'>history:</span>"
                        f"<ol style='margin:2px 0 0 -22px'>{steps}</ol>")
        return "<br>".join(rows)

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
