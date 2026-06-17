"""
GUI.WIDGETS.DUAL_VARIABLE_PICKER: SHARED TWO-LIST VARIABLE SELECTOR
==================================================================

Two side-by-side variable lists. Click a name in the left ("available") list to
move it into the right ("selected") list; click one on the right to move it back.
Shared so every dual-list selection looks and behaves identically — the
``Data ▸ Select variables`` subset picker, the gap-filling feature picker, and
any future two-list selection.

Both lists are the shared :class:`VariablePanel` (pills, fuzzy filter), so the
styling and filtering come for free; this widget only owns the move-between-lists
logic and selection order. GUI-only presentation.

Usage:
    picker = DualVariablePicker(selected_title="Features")
    picker.changed.connect(on_changed)
    picker.set_variables(df.columns, created)   # populate the left list
    picker.selected_names()                      # -> chosen names, in pick order

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLayout,
    QVBoxLayout,
    QWidget,
)

from diive.gui.widgets.variable_panel import VariablePanel


class DualVariablePicker(QWidget):
    """Available list (left) ↔ selected list (right); click to move between them."""

    #: Emitted whenever the selection changes (a move in either direction).
    changed = Signal()

    def __init__(self, parent=None, *,
                 available_title: str = "Available",
                 selected_title: str = "Selected",
                 available_hint: str = "click to add",
                 selected_hint: str = "click to remove",
                 available_footer=None,
                 selected_footer=None) -> None:
        super().__init__(parent)
        self._all: list[str] = []        # all names, original (dataset) order
        self._selected: list[str] = []   # chosen names, in selection order
        self._created: set = set()

        self.available = VariablePanel()
        self.selected = VariablePanel()
        self.available.selected.connect(lambda name, _ctrl: self.select(name))
        self.selected.selected.connect(lambda name, _ctrl: self.deselect(name))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self._column(
            available_title, available_hint, self.available, available_footer))
        lay.addLayout(self._column(
            selected_title, selected_hint, self.selected, selected_footer))

    @staticmethod
    def _column(title: str, hint: str, panel: VariablePanel, footer) -> QVBoxLayout:
        """A labelled column wrapping a panel, with an optional footer below it.

        ``footer`` is a widget (added directly), a layout (added as a sub-layout),
        or ``None``.
        """
        col = QVBoxLayout()
        head = f"<b>{title}</b>"
        if hint:
            head += f" <span style='color:#90A4AE'>({hint})</span>"
        label = QLabel(head)
        label.setWordWrap(True)
        col.addWidget(label)
        col.addWidget(panel, stretch=1)
        if footer is not None:
            if isinstance(footer, QLayout):
                col.addLayout(footer)
            else:
                col.addWidget(footer)
        return col

    # --- population ----------------------------------------------------
    def set_variables(self, names, created: set | None = None) -> None:
        """Replace the source list; keeps any current selection still present."""
        self._all = [str(c) for c in names]
        self._created = set(created or set())
        self._selected = [n for n in self._selected if n in self._all]
        self._refresh()

    def set_selected(self, names) -> None:
        """Set the selection (filtered to names that exist), in the given order."""
        self._selected = [n for n in (names or []) if n in self._all]
        self._refresh()

    def all_names(self) -> list[str]:
        return list(self._all)

    def selected_names(self) -> list[str]:
        return list(self._selected)

    # --- moves ---------------------------------------------------------
    def select(self, name: str) -> None:
        if name in self._all and name not in self._selected:
            self._selected.append(name)
            self._refresh()
            self.changed.emit()

    def deselect(self, name: str) -> None:
        if name in self._selected:
            self._selected.remove(name)
            self._refresh()
            self.changed.emit()

    def select_all(self) -> None:
        if len(self._selected) != len(self._all):
            self._selected = list(self._all)
            self._refresh()
            self.changed.emit()

    def clear(self) -> None:
        if self._selected:
            self._selected = []
            self._refresh()
            self.changed.emit()

    def _refresh(self) -> None:
        available = [n for n in self._all if n not in self._selected]
        self.available.set_variables(available, self._created)
        self.selected.set_variables(self._selected, self._created)
