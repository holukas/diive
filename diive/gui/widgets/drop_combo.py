"""
GUI.WIDGETS.DROP_COMBO: A COMBO BOX THAT ACCEPTS A DROPPED VARIABLE NAME
=======================================================================

A non-editable :class:`QComboBox` that also accepts a variable name dropped onto
it as plain text (drag a variable from a :class:`VariablePanel` list into the
field). If the dropped name is one of the combo's items, it is selected. Shared
by the plot role pickers and the derived-variable tabs.

Presentation only; no domain logic.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox


class DropComboBox(QComboBox):
    """A non-editable combo that also accepts a variable name dropped onto it as
    plain text. If the dropped name is one of its items, it is selected."""

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        text = event.mimeData().text().strip()
        i = self.findText(text)
        if i >= 0:
            self.setCurrentIndex(i)
            event.acceptProposedAction()
