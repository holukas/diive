"""
GUI.TABS.OUTLIERS_LOCALSD: LOCAL SD OUTLIER DETECTION TAB
=========================================================

Run the library's Local SD filter (`dv.outliers.LocalSD`) on a selected
variable: outliers are points that deviate from a rolling-window median by more
than ``n_sd`` standard deviations. Keeps the **original** (untouched), the
**cleaned** series (`{var}_LOCALSD`, outliers set to NaN), and the **flag**.

All the preview/threading/plotting machinery lives in :class:`BaseOutlierTab`;
this tab only supplies the LocalSD-specific parameter widgets, kwargs, detector,
and codegen. In day/night mode the library takes ``n_sd`` and ``winsize`` as
``[daytime, nighttime]`` lists, so this tab exposes a field per period.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QSpinBox

import diive as dv
from diive.gui.tabs._outlier_base import BaseOutlierTab
from diive.preprocessing.outlier_detection.codegen import localsd_to_code


class LocalSDOutlierTab(BaseOutlierTab):
    """Detect outliers with the Local SD filter; keep original + cleaned + flag."""

    title = "Local SD filter"
    intro = ("Detect outliers as points deviating from a rolling-window median by "
             "more than n_sd standard deviations. Keeps the original, a cleaned "
             "copy, and the flag.")
    settings_title = "Local SD settings"
    method_suffix = "LOCALSD"

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.winsize = QSpinBox()
        self.winsize.setRange(3, 1_000_000)
        self.winsize.setValue(480)  # refined from the data length on load
        self.winsize.setToolTip(
            "Rolling window size (record count, centred on each point) for the "
            "median and standard deviation. Seeded to ~5% of the series length when "
            "data loads; adjust as needed.")
        self._winsize_touched = False
        self.winsize.valueChanged.connect(self._mark_winsize_touched)
        form.addRow("Window (records)", self.winsize)
        self.n_sd = QDoubleSpinBox()
        self.n_sd.setRange(0.1, 50.0)
        self.n_sd.setSingleStep(0.5)
        self.n_sd.setValue(7.0)
        self.n_sd.setToolTip(
            "Threshold width: a point is an outlier if it deviates from the rolling "
            "median by more than n_sd standard deviations. Lower = stricter (flags "
            "more). Used for all records unless day/night separation is on.")
        form.addRow("n SD (global)", self.n_sd)
        self.constant_cb = QCheckBox("Constant SD (whole series)")
        self.constant_cb.setChecked(False)
        self.constant_cb.setToolTip(
            "Use the standard deviation of the entire series instead of a rolling SD "
            "within the window. Off = rolling SD (adapts to local variability).")
        form.addRow(self.constant_cb)

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

    def _add_daynight_threshold_rows(self, form: QFormLayout) -> tuple:
        # Per-period n_sd + window (the library wants [daytime, nighttime] lists).
        self.n_sd_dt = QDoubleSpinBox()
        self.n_sd_dt.setRange(0.1, 50.0); self.n_sd_dt.setSingleStep(0.5)
        self.n_sd_dt.setValue(7.0)
        self.n_sd_dt.setToolTip("n SD applied to daytime records.")
        self.n_sd_nt = QDoubleSpinBox()
        self.n_sd_nt.setRange(0.1, 50.0); self.n_sd_nt.setSingleStep(0.5)
        self.n_sd_nt.setValue(7.0)
        self.n_sd_nt.setToolTip("n SD applied to nighttime records.")
        self.winsize_dt = QSpinBox(); self.winsize_dt.setRange(3, 1_000_000)
        self.winsize_dt.setValue(480)
        self.winsize_dt.setToolTip("Rolling window (records) for daytime records.")
        self.winsize_nt = QSpinBox(); self.winsize_nt.setRange(3, 1_000_000)
        self.winsize_nt.setValue(480)
        self.winsize_nt.setToolTip("Rolling window (records) for nighttime records.")
        form.addRow("Daytime n SD", self.n_sd_dt)
        form.addRow("Nighttime n SD", self.n_sd_nt)
        form.addRow("Daytime window", self.winsize_dt)
        form.addRow("Nighttime window", self.winsize_nt)
        return (self.n_sd_dt, self.n_sd_nt, self.winsize_dt, self.winsize_nt)

    def _seed_daynight_thresholds(self) -> None:
        self.n_sd_dt.setValue(self.n_sd.value())
        self.n_sd_nt.setValue(self.n_sd.value())
        self.winsize_dt.setValue(self.winsize.value())
        self.winsize_nt.setValue(self.winsize.value())

    def _current_kwargs(self) -> dict:
        constant_sd = self.constant_cb.isChecked()
        if self.daynight_cb.isChecked():
            return dict(
                n_sd=[self.n_sd_dt.value(), self.n_sd_nt.value()],
                winsize=[self.winsize_dt.value(), self.winsize_nt.value()],
                constant_sd=constant_sd,
                separate_daytime_nighttime=True,
                lat=self.lat.value(), lon=self.lon.value(), utc_offset=self.utc.value(),
            )
        return dict(
            n_sd=self.n_sd.value(),
            winsize=self.winsize.value(),
            constant_sd=constant_sd,
        )

    def _make_detector(self, series, kwargs: dict):
        return dv.outliers.LocalSD(series=series, **kwargs)

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        return localsd_to_code(kwargs, repeat=repeat, var_name=var_name)
