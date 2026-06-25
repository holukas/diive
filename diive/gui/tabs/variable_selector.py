"""
GUI.TABS.VARIABLE_SELECTOR: PICK A SUBSET OF VARIABLES
======================================================

A dual-list picker: click a variable in the left ("Available") list to move it
to the right ("Selected") list; click one on the right to put it back. Confirm
to update the Overview tab's variable list to the chosen subset.

The two-list move logic is the shared :class:`DualVariablePicker` (also used by
the gap-filling feature picker); the subset operation itself is the library's
``dv.keep_vars`` (applied by the Overview). This tab only collects the selection
and emits it. GUI-only presentation.

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
from diive.gui.widgets.dual_variable_picker import DualVariablePicker
from diive.gui.widgets.tab_chrome import build_titlebar


class _SelectorSignals(QObject):
    """Qt signals for the tab (DiiveTab is a plain ABC, not a QObject)."""

    subset_selected = Signal(list)


class VariableSelectorTab(DiiveTab):
    """Dual-list picker for a subset of variables; updates the Overview list."""

    title = "Select variables"

    #: Always pick from the complete variable list, even while an app-wide subset
    #: is active (the subset narrows `_data`; this tab is its source, so it needs
    #: `_full_data` to let the user re-add variables the subset dropped).
    wants_full_data = True

    def build(self) -> QWidget:
        self._all: list[str] = []  # all variable names (mirrors the picker)

        self._sig = _SelectorSignals()
        #: Exposed bound signal the main window connects to.
        self.subsetSelected = self._sig.subset_selected

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(build_titlebar(self.title))

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(10, 4, 10, 4)
        outer.addWidget(body, stretch=1)

        intro = QLabel(
            "Click a variable on the left to add it to your selection; click one "
            "on the right to remove it. Confirm to show only those in the Overview.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6B7780;")
        body_lay.addWidget(intro)

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

        self.picker = DualVariablePicker(
            available_footer=self.add_all_btn, selected_footer=selected_footer)
        self.picker.changed.connect(self._on_changed)
        # Back-compat aliases (used by tests and external callers).
        self.available = self.picker.available
        self.selected = self.picker.selected

        lists = QHBoxLayout()
        lists.addWidget(self.picker)
        lists.addStretch(1)
        body_lay.addLayout(lists, stretch=1)

        # Status line below the lists.
        self.status = QLabel("")
        self.status.setStyleSheet("color: #6B7780;")
        body_lay.addWidget(self.status)
        self._on_changed()
        return root

    def save_state(self) -> dict:
        return {"selected": self.picker.selected_names()}

    def restore_state(self, state: dict) -> None:
        sel = [n for n in (state.get("selected") or []) if n in self._all]
        if not sel:
            return
        self.picker.set_selected(sel)
        self._confirm()  # re-apply the subset to the Overview

    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._all = [str(c) for c in df.columns]
        self.picker.set_variables(df.columns, created)
        self._on_changed()

    def set_active_subset(self, names) -> None:
        """Seed the selection from the app-wide subset currently in effect, so a
        freshly opened picker reflects what's actually shown (set by MainWindow)."""
        self.picker.set_selected(names)
        self._on_changed()

    # --- back-compat thin wrappers (tests call _select) ---
    def _select(self, name: str) -> None:
        self.picker.select(name)

    def _add_all(self) -> None:
        self.picker.select_all()

    def _clear(self) -> None:
        self.picker.clear()

    def _on_changed(self) -> None:
        n = len(self.picker.selected_names())
        self.status.setText(f"{n} variable{'' if n == 1 else 's'} selected")
        self.confirm_btn.setEnabled(n > 0)
        self.clear_btn.setEnabled(n > 0)
        self.add_all_btn.setEnabled(len(self._all) - n > 0)

    def _confirm(self) -> None:
        sel = self.picker.selected_names()
        if sel:
            self.subsetSelected.emit(list(sel))
            self.status.setText(f"Applied {len(sel)} variables to the Overview.")
