"""
GUI.TABS.VARIABLE_SELECTOR: PICK A SUBSET OF VARIABLES
======================================================

A dual-list picker: click a variable in the left ("Available") list to move it
to the right ("Selected") list; click one on the right to put it back. Confirm
to update the Overview tab's variable list to the chosen subset.

The subset operation itself is the library's ``dv.keep_vars`` (used by the
Overview when it applies the subset); this tab only collects the selection and
emits it. GUI-only presentation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.variable_panel import VariablePanel


class _SelectorSignals(QObject):
    """Qt signals for the tab (DiiveTab is a plain ABC, not a QObject)."""

    subset_selected = Signal(list)


class VariableSelectorTab(DiiveTab):
    """Dual-list picker for a subset of variables; updates the Overview list."""

    title = "Select variables"

    def build(self) -> QWidget:
        self._all: list[str] = []       # all variable names, original order
        self._selected: list[str] = []  # chosen names, in selection order
        self._created: set = set()

        self._sig = _SelectorSignals()
        #: Exposed bound signal the main window connects to.
        self.subsetSelected = self._sig.subset_selected

        root = QWidget()
        outer = QVBoxLayout(root)

        intro = QLabel(
            "Click a variable on the left to add it to your selection; click one "
            "on the right to remove it. Confirm to show only those in the Overview.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6B7780;")
        outer.addWidget(intro)

        # Available (left) | Selected (right).
        self.available = VariablePanel()
        self.selected = VariablePanel()
        self.available.selected.connect(lambda name, _ctrl: self._select(name))
        self.selected.selected.connect(lambda name, _ctrl: self._deselect(name))

        # Footer buttons. Available: "Add all". Selected: "Clear" (danger, red)
        # + "Confirm" (confirm, green) side by side so the primary action is
        # right next to the list it acts on.
        self.add_all_btn = QPushButton("Add all →")
        self.add_all_btn.clicked.connect(self._add_all)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)
        theme.set_button_role(self.clear_btn, "danger")
        self.confirm_btn = QPushButton("Confirm → update Overview")
        self.confirm_btn.clicked.connect(self._confirm)
        theme.set_button_role(self.confirm_btn, "confirm")

        selected_footer = QHBoxLayout()
        selected_footer.addWidget(self.clear_btn)
        selected_footer.addWidget(self.confirm_btn, stretch=1)

        lists = QHBoxLayout()
        lists.addLayout(self._column("Available", self.available, self.add_all_btn))
        lists.addLayout(self._column("Selected", self.selected, selected_footer))
        lists.addStretch(1)
        outer.addLayout(lists, stretch=1)

        # Status line below the lists.
        self.status = QLabel("")
        self.status.setStyleSheet("color: #6B7780;")
        outer.addWidget(self.status)
        return root

    @staticmethod
    def _column(title: str, panel: VariablePanel, footer) -> QVBoxLayout:
        """A labelled column wrapping a VariablePanel with a footer.

        ``footer`` is either a widget (added directly) or a layout (added as a
        sub-layout) placed below the list.
        """
        col = QVBoxLayout()
        header = QLabel(theme.manager.label_text(title))
        hf = theme.manager.tracked_font(header.font())
        hf.setBold(True)
        header.setFont(hf)
        col.addWidget(header)
        col.addWidget(panel, stretch=1)
        if isinstance(footer, QHBoxLayout):
            col.addLayout(footer)
        else:
            col.addWidget(footer)
        return col

    def save_state(self) -> dict:
        return {"selected": list(self._selected)}

    def restore_state(self, state: dict) -> None:
        sel = [n for n in (state.get("selected") or []) if n in self._all]
        if not sel:
            return
        self._clear()
        for name in sel:
            self._select(name)
        self._confirm()  # re-apply the subset to the Overview

    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._created = created or set()
        self._all = [str(c) for c in df.columns]
        # Drop selections that no longer exist (e.g. after loading new data).
        self._selected = [n for n in self._selected if n in self._all]
        self._refresh()

    def _refresh(self) -> None:
        available = [n for n in self._all if n not in self._selected]
        self.available.set_variables(available, self._created)
        self.selected.set_variables(self._selected, self._created)
        n = len(self._selected)
        self.status.setText(f"{n} variable{'' if n == 1 else 's'} selected")
        self.confirm_btn.setEnabled(n > 0)
        self.add_all_btn.setEnabled(len(available) > 0)
        self.clear_btn.setEnabled(n > 0)

    def _select(self, name: str) -> None:
        if name in self._all and name not in self._selected:
            self._selected.append(name)
            self._refresh()

    def _deselect(self, name: str) -> None:
        if name in self._selected:
            self._selected.remove(name)
            self._refresh()

    def _add_all(self) -> None:
        self._selected = [n for n in self._all]
        self._refresh()

    def _clear(self) -> None:
        self._selected = []
        self._refresh()

    def _confirm(self) -> None:
        if self._selected:
            self.subsetSelected.emit(list(self._selected))
            self.status.setText(
                f"Applied {len(self._selected)} variables to the Overview.")
