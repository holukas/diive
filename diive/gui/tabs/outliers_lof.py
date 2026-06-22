"""
GUI.TABS.OUTLIERS_LOF: LOCAL OUTLIER FACTOR DETECTION TAB
=========================================================

Run the library's Local Outlier Factor filter (`dv.outliers.LocalOutlierFactor`)
on a selected variable: a point is an outlier when its local density is
substantially lower than that of its `n_neighbors` nearest neighbors. Keeps the
**original** (untouched), the **cleaned** series (`{var}_LOF`, outliers set to
NaN), and the **flag**.

LOF scores density, not value, so there is no data-unit detection band to overlay.
Its parameters (`n_neighbors`, `contamination`) are global — in day/night mode the
library applies them *separately within* each subset, so separation changes the
result even with identical settings (no per-period thresholds needed, unlike the
SD-based methods). All detection is library work; this tab only supplies the
parameter widgets, kwargs, detector, and codegen.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
)

import diive as dv
from diive.gui.tabs._outlier_base import BaseOutlierTab
from diive.preprocessing.outlier_detection.codegen import lof_to_code
from diive.preprocessing.outlier_detection.lof import suggest_lof_params


class LocalOutlierFactorTab(BaseOutlierTab):
    """Detect outliers with the Local Outlier Factor; keep original + cleaned + flag."""

    title = "Local Outlier Factor filter"
    intro = ("Detect outliers by local density: a point is an outlier when its "
             "density is much lower than that of its nearest neighbors. Keeps the "
             "original, a cleaned copy, and the flag.")
    settings_title = "Local Outlier Factor settings"
    method_suffix = "LOF"
    method_chip_label = "LOF"
    method_chip_bg = "#F3E5F5"
    method_chip_fg = "#6A1B9A"

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.n_neighbors = QSpinBox()
        self.n_neighbors.setRange(1, 1_000_000)
        self.n_neighbors.setValue(20)
        self.n_neighbors.setToolTip(
            "Number of nearest neighbors used to estimate each point's local "
            "density. Larger = smoother density estimate (less sensitive to "
            "isolated points). Capped at the sample count by scikit-learn.")
        form.addRow("Neighbors", self.n_neighbors)
        self.contamination = QDoubleSpinBox()
        self.contamination.setRange(0.001, 0.5)
        self.contamination.setSingleStep(0.01)
        self.contamination.setDecimals(3)
        self.contamination.setValue(0.01)
        self.contamination.setToolTip(
            "Expected proportion of outliers in the data (0–0.5). Sets the density "
            "threshold: higher flags more points. 0.01 = expect ~1% outliers.")
        form.addRow("Contamination", self.contamination)
        self.contamination_auto = QCheckBox("Auto (threshold as in the LOF paper)")
        self.contamination_auto.setChecked(False)
        self.contamination_auto.setToolTip(
            "Let scikit-learn set the outlier threshold automatically (the original "
            "LOF paper's rule) instead of a fixed proportion. Disables the "
            "contamination field above.")
        self.contamination_auto.toggled.connect(
            lambda on: self.contamination.setEnabled(not on))
        form.addRow(self.contamination_auto)

        # Fill the fields with library-recommended settings for the picked variable.
        self.suggest_btn = QPushButton("Suggest settings")
        self.suggest_btn.setToolTip(
            "Fill the fields with recommended settings for the selected variable: "
            "n_neighbors at scikit-learn's default (clamped to the sample count) and "
            "contamination on Auto. Thresholds for LOF aren't data-derivable, so this "
            "only sets what can be recommended.")
        self.suggest_btn.clicked.connect(self._suggest)
        form.addRow(self.suggest_btn)

    def _suggest(self) -> None:
        """Populate the controls from the library's recommended params."""
        if self._df is None or self._var is None:
            self.status.setText("Select a variable first to suggest settings.")
            return
        params = suggest_lof_params(self._df[self._var])
        self.n_neighbors.setValue(params["n_neighbors"])
        self.contamination_auto.setChecked(params["contamination"] == "auto")
        self.status.setText(
            f"Suggested settings for '{self._var}': n_neighbors="
            f"{params['n_neighbors']}, contamination=auto.")

    def _state_controls(self) -> dict:
        return {**super()._state_controls(), "n_neighbors": self.n_neighbors,
                "contamination": self.contamination,
                "contamination_auto": self.contamination_auto}

    def _current_kwargs(self) -> dict:
        contamination = ("auto" if self.contamination_auto.isChecked()
                         else self.contamination.value())
        kwargs = dict(
            n_neighbors=self.n_neighbors.value(),
            contamination=contamination,
        )
        # LOF takes a single separate-flag plus coordinates; the global params are
        # reused within each day/night subset (no per-period thresholds).
        if self.daynight_cb.isChecked():
            kwargs.update(
                separate_daytime_nighttime=True,
                lat=self.lat.value(), lon=self.lon.value(), utc_offset=self.utc.value(),
            )
        return kwargs

    def _make_detector(self, series, kwargs: dict):
        return dv.outliers.LocalOutlierFactor(series=series, **kwargs)

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        return lof_to_code(kwargs, repeat=repeat, var_name=var_name)
