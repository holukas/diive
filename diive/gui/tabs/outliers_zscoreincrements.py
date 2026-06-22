"""
GUI.TABS.OUTLIERS_ZSCOREINCREMENTS: Z-SCORE INCREMENTS OUTLIER DETECTION TAB
===========================================================================

Run the library's z-score increments filter (`dv.outliers.zScoreIncrements`) on a
selected variable: outliers are abrupt changes — a point is flagged only when the
z-scores of its forward, backward, *and* combined increments all exceed a
threshold. This isolates spikes while tolerating gradual change. Keeps the
**original** (untouched), the **cleaned** series (`{var}_ZSCOREINCREMENTS`,
outliers set to NaN), and the **flag**.

The method works on increment z-scores (not the values directly), so there is no
data-unit detection band and no day/night mode. All detection is library work;
this tab only supplies the parameter widget, kwargs, detector, and codegen.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout

import diive as dv
from diive.gui.tabs._outlier_base import BaseOutlierTab
from diive.preprocessing.outlier_detection.codegen import zscoreincrements_to_code


class ZScoreIncrementsOutlierTab(BaseOutlierTab):
    """Detect abrupt-change outliers via increment z-scores; keep original +
    cleaned + flag."""

    title = "Z-score (increments) filter"
    intro = ("Detect abrupt changes: a point is an outlier only when the z-scores "
             "of its forward, backward, and combined increments all exceed a "
             "threshold. Keeps the original, a cleaned copy, and the flag.")
    settings_title = "Z-score increments settings"
    method_suffix = "ZSCOREINCREMENTS"
    method_chip_label = "Z-SCORE INCR"
    method_chip_bg = "#E0F2F1"
    method_chip_fg = "#00695C"
    supports_daynight = False

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.thres = QDoubleSpinBox()
        self.thres.setRange(0.1, 50.0)
        self.thres.setSingleStep(0.5)
        self.thres.setValue(4.0)
        self.thres.setToolTip(
            "Z-score threshold applied to each increment series (forward, backward, "
            "combined). A point is flagged only when all three exceed it. Lower = "
            "stricter (flags more).")
        form.addRow("Threshold", self.thres)

    def _state_controls(self) -> dict:
        return {**super()._state_controls(), "thres": self.thres}

    def _current_kwargs(self) -> dict:
        return dict(thres_zscore=self.thres.value())

    def _make_detector(self, series, kwargs: dict):
        return dv.outliers.zScoreIncrements(series=series, **kwargs)

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        return zscoreincrements_to_code(kwargs, repeat=repeat, var_name=var_name)
