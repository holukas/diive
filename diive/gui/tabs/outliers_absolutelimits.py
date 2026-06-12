"""
GUI.TABS.OUTLIERS_ABSOLUTELIMITS: ABSOLUTE LIMITS DETECTION TAB
===============================================================

Run the library's absolute-limits filter (`dv.outliers.AbsoluteLimits`) on a
selected variable: a point is an outlier when it falls outside a fixed
``[minval, maxval]`` range. Keeps the **original** (untouched), the **cleaned**
series (`{var}_ABSLIM`, outliers set to NaN), and the **flag**.

Unlike the density-based LOF method, the min/max here *are* a data-unit detection
band, so the "Show limit lines" overlay draws the actual limits (flat lines in
global mode; per-period day/night limits when separation is on). In day/night
mode the library takes separate ``[min, max]`` ranges for daytime and nighttime.
All detection is library work; this tab only supplies the parameter widgets,
kwargs, detector, and codegen.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout

import diive as dv
from diive.gui.tabs._outlier_base import BaseOutlierTab
from diive.preprocessing.outlier_detection.codegen import absolutelimits_to_code

# Wide enough for real flux/meteo values (negative, large magnitudes).
_LIMIT_MIN, _LIMIT_MAX = -1e12, 1e12


def _limit_spinbox() -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(_LIMIT_MIN, _LIMIT_MAX)
    box.setDecimals(4)
    return box


class AbsoluteLimitsTab(BaseOutlierTab):
    """Detect outliers outside a fixed value range; keep original + cleaned + flag."""

    title = "Absolute limits filter"
    intro = ("Detect outliers as values outside a fixed minimum/maximum range. "
             "Keeps the original, a cleaned copy, and the flag.")
    settings_title = "Absolute limits settings"
    method_suffix = "ABSLIM"

    def _add_method_rows(self, form: QFormLayout) -> None:
        # No universal default exists for a value range, so the limits are seeded
        # per variable from its data range on selection (see _select).
        self._limits_touched = False
        self.minval = _limit_spinbox()
        self.minval.setToolTip(
            "Lower acceptable limit. Values below this are flagged as outliers. "
            "Seeded from the selected variable's minimum; tighten as needed.")
        self.minval.valueChanged.connect(self._mark_limits_touched)
        form.addRow("Minimum", self.minval)
        self.maxval = _limit_spinbox()
        self.maxval.setToolTip(
            "Upper acceptable limit. Values above this are flagged as outliers. "
            "Seeded from the selected variable's maximum; tighten as needed.")
        self.maxval.valueChanged.connect(self._mark_limits_touched)
        form.addRow("Maximum", self.maxval)

    def _mark_limits_touched(self) -> None:
        self._limits_touched = True

    def _select(self, name: str) -> None:
        super()._select(name)
        # Seed the limits from the selected series' actual range (so the initial
        # state flags nothing); skip once the user has edited them by hand.
        if self._limits_touched or self._df is None or name not in self._df.columns:
            return
        s = self._df[name].dropna()
        if not len(s):
            return
        self.minval.blockSignals(True)
        self.maxval.blockSignals(True)
        self.minval.setValue(float(s.min()))
        self.maxval.setValue(float(s.max()))
        self.minval.blockSignals(False)
        self.maxval.blockSignals(False)

    def _add_daynight_threshold_rows(self, form: QFormLayout) -> tuple:
        self.dt_min = _limit_spinbox()
        self.dt_max = _limit_spinbox()
        self.nt_min = _limit_spinbox()
        self.nt_max = _limit_spinbox()
        form.addRow("Daytime min", self.dt_min)
        form.addRow("Daytime max", self.dt_max)
        form.addRow("Nighttime min", self.nt_min)
        form.addRow("Nighttime max", self.nt_max)
        return (self.dt_min, self.dt_max, self.nt_min, self.nt_max)

    def _seed_daynight_thresholds(self) -> None:
        self.dt_min.setValue(self.minval.value())
        self.dt_max.setValue(self.maxval.value())
        self.nt_min.setValue(self.minval.value())
        self.nt_max.setValue(self.maxval.value())

    def _state_controls(self) -> dict:
        return {**super()._state_controls(), "minval": self.minval, "maxval": self.maxval,
                "dt_min": self.dt_min, "dt_max": self.dt_max,
                "nt_min": self.nt_min, "nt_max": self.nt_max}

    def _current_kwargs(self) -> dict:
        if self.daynight_cb.isChecked():
            return dict(
                separate_daytime_nighttime=True,
                daytime_minmax=[self.dt_min.value(), self.dt_max.value()],
                nighttime_minmax=[self.nt_min.value(), self.nt_max.value()],
                lat=self.lat.value(), lon=self.lon.value(), utc_offset=self.utc.value(),
            )
        return dict(minval=self.minval.value(), maxval=self.maxval.value())

    def _make_detector(self, series, kwargs: dict):
        return dv.outliers.AbsoluteLimits(series=series, **kwargs)

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        return absolutelimits_to_code(kwargs, repeat=repeat, var_name=var_name)
