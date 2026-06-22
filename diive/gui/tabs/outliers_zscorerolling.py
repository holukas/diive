"""
GUI.TABS.OUTLIERS_ZSCOREROLLING: ROLLING Z-SCORE OUTLIER DETECTION TAB
=====================================================================

Run the library's rolling z-score filter (`dv.outliers.zScoreRolling`) on a
selected variable: outliers are points whose z-score relative to a *rolling*
mean and standard deviation (centred on each point) exceeds a threshold. The
adaptive, local band suits non-stationary series. Keeps the **original**
(untouched), the **cleaned** series (`{var}_ZSCOREROLLING`, outliers set to NaN),
and the **flag**.

The rolling z-score has no day/night mode (the rolling window already adapts to
local conditions), so this tab omits the day/night box. All detection is library
work; this tab only supplies the parameter widgets, kwargs, detector, and codegen.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QSpinBox

import diive as dv
from diive.gui.tabs._outlier_base import BaseOutlierTab
from diive.preprocessing.outlier_detection.codegen import zscorerolling_to_code


class ZScoreRollingOutlierTab(BaseOutlierTab):
    """Detect outliers with a rolling z-score; keep original + cleaned + flag."""

    title = "Z-score (rolling) filter"
    intro = ("Detect outliers as points whose z-score relative to a rolling mean "
             "and standard deviation exceeds a threshold. Adapts to local "
             "variability. Keeps the original, a cleaned copy, and the flag.")
    settings_title = "Rolling z-score settings"
    method_suffix = "ZSCOREROLLING"
    method_chip_label = "Z-SCORE ROLL"
    method_chip_bg = "#FFF8E1"
    method_chip_fg = "#F57F17"
    supports_daynight = False
    # The band is rolling_mean ± threshold * rolling_std, so its centre is the
    # rolling mean — worth showing alongside the limits for this method.
    band_center_label = "rolling mean"

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.thres = QDoubleSpinBox()
        self.thres.setRange(0.1, 50.0)
        self.thres.setSingleStep(0.5)
        self.thres.setValue(4.0)
        self.thres.setToolTip(
            "Z-score threshold: a point is an outlier if its absolute rolling "
            "z-score exceeds this value. Lower = stricter (flags more).")
        form.addRow("Threshold", self.thres)
        self.winsize = QSpinBox()
        self.winsize.setRange(3, 1_000_000)
        self.winsize.setValue(480)  # refined from the data length on load
        self.winsize.setToolTip(
            "Rolling window size (record count, centred on each point) for the "
            "mean and standard deviation. Seeded to ~5% of the series length when "
            "data loads; adjust as needed.")
        self._winsize_touched = False
        self.winsize.valueChanged.connect(self._mark_winsize_touched)
        form.addRow("Window (records)", self.winsize)

    def _mark_winsize_touched(self) -> None:
        self._winsize_touched = True

    def on_data_loaded(self, df, created=None) -> None:
        super().on_data_loaded(df, created)
        # Seed the window to the library's default (~5% of length) until the user
        # edits it, so it's sensible for the loaded dataset.
        if not self._winsize_touched and len(df):
            self.winsize.blockSignals(True)
            self.winsize.setValue(max(3, int(len(df) / 20)))
            self.winsize.blockSignals(False)

    def _state_controls(self) -> dict:
        return {**super()._state_controls(), "thres": self.thres,
                "winsize": self.winsize}

    def _current_kwargs(self) -> dict:
        return dict(thres_zscore=self.thres.value(), winsize=self.winsize.value())

    def _make_detector(self, series, kwargs: dict):
        return dv.outliers.zScoreRolling(series=series, **kwargs)

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        return zscorerolling_to_code(kwargs, repeat=repeat, var_name=var_name)
