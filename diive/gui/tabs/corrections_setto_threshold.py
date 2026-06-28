"""
GUI.TABS.CORRECTIONS_SETTO_THRESHOLD: SET-TO-MAX / SET-TO-MIN TABS
=================================================================

Cap (set-to-max) or floor (set-to-min) a variable at a known physical limit:
every value beyond the threshold is set to the threshold
(`dv.corrections.setto_threshold`).

Two thin :class:`BaseCorrectionTab` subclasses share one threshold spinbox row;
they differ only in the correction key, chip, and default threshold. Generic —
applicable to any variable.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout

from diive.gui.tabs._correction_base import BaseCorrectionTab
from diive.preprocessing.qaqc.measurements import CORR_SETTO_MAX, CORR_SETTO_MIN


class _SetToThresholdTab(BaseCorrectionTab):
    """Common threshold row; concrete max/min tabs set the key and default."""

    #: Default threshold seeded into the spinbox.
    default_threshold = 0.0

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-1e6, 1e6)
        self.threshold.setDecimals(3)
        self.threshold.setValue(self.default_threshold)
        self.threshold.setToolTip("Values beyond this threshold are set to it.")
        form.addRow("Threshold", self.threshold)

    def _current_kwargs(self) -> dict:
        return {"threshold": self.threshold.value()}

    def _method_controls(self) -> dict:
        return {"threshold": self.threshold}


class SetToMaxThresholdTab(_SetToThresholdTab):
    """Cap the series: every value above the threshold is set to the threshold."""

    title = "Set to max threshold"
    intro = ("Caps the series at a known physical maximum: every value above the "
             "threshold is set to the threshold.")
    method_suffix = "SETMAX"
    corr_key = CORR_SETTO_MAX
    method_chip_label = "SET MAX"
    method_chip_bg = "#FCE4EC"
    method_chip_fg = "#AD1457"
    default_threshold = 30.0


class SetToMinThresholdTab(_SetToThresholdTab):
    """Floor the series: every value below the threshold is set to the threshold."""

    title = "Set to min threshold"
    intro = ("Floors the series at a known physical minimum: every value below the "
             "threshold is set to the threshold.")
    method_suffix = "SETMIN"
    corr_key = CORR_SETTO_MIN
    method_chip_label = "SET MIN"
    method_chip_bg = "#F3E5F5"
    method_chip_fg = "#6A1B9A"
    default_threshold = -5.0
