"""
GUI.TABS.CORRECTIONS_SETTO_VALUE: SET TIME RANGES TO A VALUE TAB
===============================================================

Overwrite every record inside one or more date ranges with a fixed value — e.g.
to blank out a period of known instrument trouble
(`dv.corrections.setto_value`).

A thin :class:`BaseCorrectionTab` subclass: a date-range text field + a value
spinbox. The small text-parsing helpers are shared with the corrections picker.
Generic — applicable to any variable.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QLineEdit

from diive.gui.tabs._correction_base import BaseCorrectionTab
from diive.gui.widgets.corrections_panel import _parse_dates
from diive.preprocessing.qaqc.measurements import CORR_SETTO_VALUE


class SetToValueTab(BaseCorrectionTab):
    """Set every record inside one or more date ranges to a fixed value."""

    title = "Set to value"
    intro = ("Overwrites every record inside one or more date ranges with a fixed "
             "value — e.g. to blank out a period of known instrument trouble.")
    method_suffix = "SETVAL"
    corr_key = CORR_SETTO_VALUE
    method_chip_label = "SET VALUE"
    method_chip_bg = "#E8EAF6"
    method_chip_fg = "#283593"

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.dates = QLineEdit()
        self.dates.setPlaceholderText("2022-04-01..2022-04-05; ...")
        self.dates.setToolTip(
            "Date ranges separated by ';'. A range uses '..' (start..end, both "
            "inclusive); a single timestamp stands alone.")
        form.addRow("Dates", self.dates)
        self.value = QDoubleSpinBox()
        self.value.setRange(-1e6, 1e6)
        self.value.setDecimals(3)
        self.value.setToolTip("Value written to every record in the date ranges.")
        form.addRow("Value", self.value)

    def _current_kwargs(self) -> dict:
        return {"dates": _parse_dates(self.dates.text()), "value": self.value.value()}

    def _validate(self, kwargs: dict) -> str | None:
        if not kwargs.get("dates"):
            return "Enter at least one date or date range first."
        return None

    def _method_controls(self) -> dict:
        return {"dates": self.dates, "value": self.value}
