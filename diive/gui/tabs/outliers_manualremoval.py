"""
GUI.TABS.OUTLIERS_MANUALREMOVAL: MANUAL REMOVAL OUTLIER TAB
==========================================================

Flag known-bad records by listing them explicitly, instead of detecting them
statistically (`dv.outliers.ManualRemoval`). The user types the dates/ranges to
remove; matched records are flagged 2 and set to NaN in the cleaned copy. Keeps
the **original** (untouched), the **cleaned** series (`{var}_MANUAL`), and the
**flag**.

Selection is purely time-based, so there is no detection band, no day/night
mode, and no repeat (re-flagging the same fixed timestamps would never
converge). All date interpretation/validation stays in the library; this tab
only turns the text box (one entry per line) into the library's ``remove_dates``
list and supplies the detector and codegen.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLabel, QPlainTextEdit

import diive as dv
from diive.gui.tabs._outlier_base import BaseOutlierTab
from diive.preprocessing.outlier_detection.codegen import manualremoval_to_code


class ManualRemovalOutlierTab(BaseOutlierTab):
    """Flag explicitly listed records/ranges for removal; keep original +
    cleaned + flag."""

    title = "Manual removal"
    intro = ("Flag known-bad records by listing them — for instrument failures, "
             "maintenance, or disturbances whose timing is known. Keeps the "
             "original, a cleaned copy, and the flag.")
    settings_title = "Records to remove"
    method_suffix = "MANUAL"
    supports_daynight = False
    supports_repeat = False
    run_label = "Flag listed dates"

    def _add_method_rows(self, form: QFormLayout) -> None:
        form.addRow(QLabel("Dates / ranges to remove (one per line):"))
        self.dates_edit = QPlainTextEdit()
        self.dates_edit.setFixedHeight(150)
        self.dates_edit.setPlaceholderText(
            "2018-07-10\n"
            "2018-07-05 10:45:00\n"
            "2018-07-02 to 2018-07-04")
        self.dates_edit.setToolTip(
            "One entry per line. Each line is either:\n"
            "  • a single date or datetime — flags that record; a bare date (no "
            "time) flags the whole day, e.g. 2018-07-10\n"
            "  • a range 'start to end' (or 'start, end') — flags everything in "
            "the closed interval; bare dates span whole days, e.g. "
            "2018-07-02 to 2018-07-04")
        form.addRow(self.dates_edit)

    def _state_controls(self) -> dict:
        return {**super()._state_controls(), "dates": self.dates_edit}

    def _parse_remove_dates(self) -> list:
        """Map the text box (one entry per line) to the library's ``remove_dates``
        list. Trivial input glue — date interpretation/validation is the library's.
        A line with ' to ' or a comma is a [start, end] range; otherwise a single
        record."""
        entries: list = []
        for raw in self.dates_edit.toPlainText().splitlines():
            line = raw.strip()
            if not line:
                continue
            if " to " in line:
                start, end = line.split(" to ", 1)
                entries.append([start.strip(), end.strip()])
            elif "," in line:
                start, end = line.split(",", 1)
                entries.append([start.strip(), end.strip()])
            else:
                entries.append(line)
        return entries

    def _current_kwargs(self) -> dict:
        return dict(remove_dates=self._parse_remove_dates())

    def _make_detector(self, series, kwargs: dict):
        return dv.outliers.ManualRemoval(series=series, **kwargs)

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        return manualremoval_to_code(kwargs, repeat=repeat, var_name=var_name)
