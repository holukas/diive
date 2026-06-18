"""
GUI.TABS.CORRECTIONS_SET_MISSING: SET EXACT VALUES TO MISSING TAB
================================================================

Set records that exactly equal any of the listed values to missing (NaN) — e.g.
to drop a stuck sentinel like 0 or -9999
(`dv.corrections.set_exact_values_to_missing`).

A thin :class:`BaseCorrectionTab` subclass: a comma-separated value-list field.
The small text-parsing helper is shared with the corrections picker. Generic —
applicable to any variable.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLineEdit

from diive.gui.tabs._correction_base import BaseCorrectionTab
from diive.gui.widgets.corrections_panel import _parse_values
from diive.preprocessing.qaqc.measurements import CORR_SET_EXACT_TO_MISSING


class SetExactToMissingTab(BaseCorrectionTab):
    """Set records equal to any listed value to missing (NaN)."""

    title = "Set exact values to missing"
    intro = ("Sets records that exactly equal any of the listed values to missing "
             "(NaN) — e.g. to drop a stuck sentinel like 0 or -9999.")
    method_suffix = "SETMISSING"
    corr_key = CORR_SET_EXACT_TO_MISSING
    method_chip_label = "SET MISSING"
    method_chip_bg = "#FFEBEE"
    method_chip_fg = "#C62828"

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.values = QLineEdit()
        self.values.setPlaceholderText("0, -9999")
        self.values.setToolTip("Comma-separated values set to missing (NaN).")
        form.addRow("Values", self.values)

    def _current_kwargs(self) -> dict:
        return {"values": _parse_values(self.values.text())}

    def _validate(self, kwargs: dict) -> str | None:
        if not kwargs.get("values"):
            return "Enter at least one value to set to missing first."
        return None

    def _method_controls(self) -> dict:
        return {"values": self.values}
