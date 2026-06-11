"""
GUI.TABS.OUTLIERS: HAMPEL OUTLIER DETECTION TAB
===============================================

Run the library's Hampel filter (`dv.outliers.Hampel`) on a selected variable.
Keeps all three series: the **original** (the existing variable, untouched), the
**cleaned** series (`{var}_HAMPEL`, outliers set to NaN), and the **flag** that
produced it (`FLAG_{var}_OUTLIER_HAMPEL_TEST`, 0 = ok, 2 = outlier).

All the preview/threading/plotting machinery lives in :class:`BaseOutlierTab`;
this tab only supplies the Hampel-specific parameter widgets, kwargs, detector,
and codegen. All detection is library work.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QSpinBox

import diive as dv
from diive.gui.tabs._outlier_base import BaseOutlierTab
from diive.preprocessing.outlier_detection.codegen import hampel_to_code


class HampelOutlierTab(BaseOutlierTab):
    """Detect outliers with the Hampel filter; keep original + cleaned + flag."""

    title = "Hampel filter"
    intro = ("Detect spikes with the Hampel filter (median absolute deviation). "
             "Keeps the original, a cleaned copy, and the flag.")
    settings_title = "Hampel settings"
    method_suffix = "HAMPEL"

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.window = QSpinBox()
        self.window.setRange(3, 1_000_000)
        self.window.setValue(48 * 13)  # 13 days at 30-min sampling (Papale 2006)
        self.window.setToolTip(
            "Size of the sliding window (record count, centred on each point) over "
            "which the local median and MAD are computed. 624 = 13 days at 30-min "
            "sampling (Papale 2006); use 24*13 for hourly data.")
        form.addRow("Window (records)", self.window)
        self.n_sigma = QDoubleSpinBox()
        self.n_sigma.setRange(0.1, 50.0)
        self.n_sigma.setSingleStep(0.5)
        self.n_sigma.setValue(5.5)
        self.n_sigma.setToolTip(
            "Threshold width: a point is an outlier if it deviates from the local "
            "median by more than n_sigma × k × MAD. Lower = stricter (flags more). "
            "Used for all records unless day/night separation is on.")
        form.addRow("n sigma (global)", self.n_sigma)
        self.diff_cb = QCheckBox("Use double-differencing (Papale 2006)")
        self.diff_cb.setChecked(True)
        self.diff_cb.setToolTip(
            "Run the test on the double-differenced series d = 2·xₜ − (xₜ₋₁ + xₜ₊₁) "
            "instead of raw values. Removes trends and isolates short spikes "
            "(recommended for flux/meteo data). Off = test the raw values.")
        form.addRow(self.diff_cb)

    def _add_daynight_threshold_rows(self, form: QFormLayout) -> tuple:
        # Separate thresholds make day/night meaningful (with one shared sigma the
        # result is identical to no separation). Seeded from the global n sigma.
        self.n_sigma_dt = QDoubleSpinBox()
        self.n_sigma_dt.setRange(0.1, 50.0); self.n_sigma_dt.setSingleStep(0.5)
        self.n_sigma_dt.setValue(5.5)
        self.n_sigma_dt.setToolTip("Threshold (n sigma) applied to daytime records.")
        self.n_sigma_nt = QDoubleSpinBox()
        self.n_sigma_nt.setRange(0.1, 50.0); self.n_sigma_nt.setSingleStep(0.5)
        self.n_sigma_nt.setValue(5.5)
        self.n_sigma_nt.setToolTip("Threshold (n sigma) applied to nighttime records.")
        form.addRow("Daytime n sigma", self.n_sigma_dt)
        form.addRow("Nighttime n sigma", self.n_sigma_nt)
        return (self.n_sigma_dt, self.n_sigma_nt)

    def _seed_daynight_thresholds(self) -> None:
        self.n_sigma_dt.setValue(self.n_sigma.value())
        self.n_sigma_nt.setValue(self.n_sigma.value())

    def _state_controls(self) -> dict:
        return {**super()._state_controls(), "window": self.window,
                "n_sigma": self.n_sigma, "diff": self.diff_cb,
                "n_sigma_dt": self.n_sigma_dt, "n_sigma_nt": self.n_sigma_nt}

    def _current_kwargs(self) -> dict:
        kwargs = dict(
            window_length=self.window.value(),
            n_sigma=self.n_sigma.value(),
            use_differencing=self.diff_cb.isChecked(),
            separate_day_night=self.daynight_cb.isChecked(),
        )
        if self.daynight_cb.isChecked():
            kwargs.update(lat=self.lat.value(), lon=self.lon.value(),
                          utc_offset=self.utc.value(),
                          n_sigma_daytime=self.n_sigma_dt.value(),
                          n_sigma_nighttime=self.n_sigma_nt.value())
        return kwargs

    def _make_detector(self, series, kwargs: dict):
        return dv.outliers.Hampel(series=series, **kwargs)

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        return hampel_to_code(kwargs, repeat=repeat, var_name=var_name)
