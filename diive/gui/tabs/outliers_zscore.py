"""
GUI.TABS.OUTLIERS_ZSCORE: Z-SCORE OUTLIER DETECTION TAB
=======================================================

Run the library's z-score filter (`dv.outliers.zScore`) on a selected variable:
outliers are points whose absolute z-score (deviation from the mean in units of
standard deviation) exceeds a threshold. Keeps the **original** (untouched), the
**cleaned** series (`{var}_ZSCORE`, outliers set to NaN), and the **flag**.

In day/night mode the library computes the z-score separately for daytime and
nighttime records (each with its own mean/SD) and applies a per-period threshold,
so this tab exposes a threshold per period (seeded from the global value).

All the preview/threading/plotting machinery lives in :class:`BaseOutlierTab`;
this tab only supplies the z-score-specific parameter widgets, kwargs, detector,
and codegen. All detection is library work.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout

import diive as dv
from diive.gui.tabs._outlier_base import BaseOutlierTab
from diive.preprocessing.outlier_detection.codegen import zscore_to_code


class ZScoreOutlierTab(BaseOutlierTab):
    """Detect outliers with the z-score filter; keep original + cleaned + flag."""

    title = "Z-score filter"
    intro = ("Detect outliers as points whose absolute z-score (deviation from the "
             "mean in standard deviations) exceeds a threshold. Keeps the original, "
             "a cleaned copy, and the flag.")
    settings_title = "Z-score settings"
    method_suffix = "ZSCORE"

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.thres = QDoubleSpinBox()
        self.thres.setRange(0.1, 50.0)
        self.thres.setSingleStep(0.5)
        self.thres.setValue(4.0)
        self.thres.setToolTip(
            "Z-score threshold: a point is an outlier if its absolute z-score "
            "exceeds this value. Lower = stricter (flags more). Used for all records "
            "unless day/night separation is on.")
        form.addRow("Threshold (global)", self.thres)

    def _add_daynight_threshold_rows(self, form: QFormLayout) -> tuple:
        # Per-period thresholds make day/night meaningful. Seeded from the global
        # threshold; the z-score itself is also computed per period (own mean/SD).
        self.thres_dt = QDoubleSpinBox()
        self.thres_dt.setRange(0.1, 50.0); self.thres_dt.setSingleStep(0.5)
        self.thres_dt.setValue(4.0)
        self.thres_dt.setToolTip("Z-score threshold applied to daytime records.")
        self.thres_nt = QDoubleSpinBox()
        self.thres_nt.setRange(0.1, 50.0); self.thres_nt.setSingleStep(0.5)
        self.thres_nt.setValue(4.0)
        self.thres_nt.setToolTip("Z-score threshold applied to nighttime records.")
        form.addRow("Daytime threshold", self.thres_dt)
        form.addRow("Nighttime threshold", self.thres_nt)
        return (self.thres_dt, self.thres_nt)

    def _seed_daynight_thresholds(self) -> None:
        self.thres_dt.setValue(self.thres.value())
        self.thres_nt.setValue(self.thres.value())

    def _state_controls(self) -> dict:
        return {**super()._state_controls(), "thres": self.thres,
                "thres_dt": self.thres_dt, "thres_nt": self.thres_nt}

    def _current_kwargs(self) -> dict:
        kwargs = dict(thres_zscore=self.thres.value())
        if self.daynight_cb.isChecked():
            kwargs.update(separate_daytime_nighttime=True,
                          lat=self.lat.value(), lon=self.lon.value(),
                          utc_offset=self.utc.value(),
                          thres_zscore_daytime=self.thres_dt.value(),
                          thres_zscore_nighttime=self.thres_nt.value())
        return kwargs

    def _make_detector(self, series, kwargs: dict):
        return dv.outliers.zScore(series=series, **kwargs)

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        return zscore_to_code(kwargs, repeat=repeat, var_name=var_name)
