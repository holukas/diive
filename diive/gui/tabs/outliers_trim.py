"""
GUI.TABS.OUTLIERS_TRIM: TRIM-LOW (SYMMETRIC) OUTLIER DETECTION TAB
=================================================================

Run the library's trim-low filter (`dv.outliers.TrimLow`) on a selected variable:
it rejects the values below ``lower_limit`` and then an equal number of the
highest values, keeping the distribution symmetric (the trimmed-mean rationale).
Keeps the **original** (untouched), the **cleaned** series (`{var}_TRIMLOW`,
outliers set to NaN), and the **flag**.

By default the whole series is trimmed and **no coordinates are needed**. The
optional **Trim daytime only** / **Trim nighttime only** checkboxes restrict (and
split) the trim to those periods, each screened against its own distribution —
only then are site coordinates used (and enabled). This is TrimLow's own
method-specific day/night model, not the standard "separate thresholds" toggle, so
the shared day/night box is disabled (`supports_daynight = False`) and the
trim-side checkboxes + coordinates live in the method rows. The symmetric
positional trim has no single data-unit band, so there is no limit-line overlay
(like the increments method). All detection is library work; this tab only
supplies the parameter widgets, kwargs, detector, and codegen.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QSpinBox

import diive as dv
from diive.gui import site
from diive.gui.tabs._outlier_base import BaseOutlierTab
from diive.preprocessing.outlier_detection.codegen import trimlow_to_code

# Wide enough for real flux/meteo values (negative, large magnitudes).
_LIMIT_MIN, _LIMIT_MAX = -1e12, 1e12


class TrimLowOutlierTab(BaseOutlierTab):
    """Detect low outliers via a symmetric trim; keep original + cleaned + flag."""

    title = "Trim-low filter"
    intro = ("Reject values below a lower limit, plus an equal number of the "
             "highest values, keeping the distribution symmetric (trimmed mean). "
             "Keeps the original, a cleaned copy, and the flag.")
    settings_title = "Trim-low settings"
    method_suffix = "TRIMLOW"
    # TrimLow's day/night handling is method-specific and opt-in (which period to
    # screen, default whole series), not the standard per-period-threshold
    # separation — so the shared box is off and the optional coordinates live in
    # the method rows below.
    supports_daynight = False

    def _add_method_rows(self, form: QFormLayout) -> None:
        # No universal default for a value floor — seeded per variable from its
        # data minimum on selection (so the initial state flags nothing).
        self._limit_touched = False
        self.lower_limit = QDoubleSpinBox()
        self.lower_limit.setRange(_LIMIT_MIN, _LIMIT_MAX)
        self.lower_limit.setDecimals(4)
        self.lower_limit.setToolTip(
            "Values below this limit are rejected; an equal number of the highest "
            "values is then also rejected (symmetric trim). Seeded from the "
            "selected variable's minimum; raise it to start trimming.")
        self.lower_limit.valueChanged.connect(self._mark_limit_touched)
        form.addRow("Lower limit", self.lower_limit)

        # Day/night is opt-in. With both off (the default) the whole series is
        # trimmed against one distribution and no coordinates are needed.
        self.trim_daytime = QCheckBox("Trim daytime only")
        self.trim_daytime.setToolTip(
            "Screen daytime records against their own distribution. Leave both "
            "day/night boxes off to trim the whole series (no coordinates needed).")
        self.trim_daytime.toggled.connect(self._sync_coord_enabled)
        form.addRow(self.trim_daytime)
        self.trim_nighttime = QCheckBox("Trim nighttime only")
        self.trim_nighttime.setToolTip(
            "Screen nighttime records against their own distribution. Leave both "
            "day/night boxes off to trim the whole series (no coordinates needed).")
        self.trim_nighttime.toggled.connect(self._sync_coord_enabled)
        form.addRow(self.trim_nighttime)

        # Coordinates are only needed when a day/night split is requested. Named
        # with a `trim_` prefix so they don't collide with the base's
        # self.lat/lon/utc (which it nulls when supports_daynight is False).
        self.trim_lat = QDoubleSpinBox()
        self.trim_lat.setRange(-90.0, 90.0)
        self.trim_lat.setDecimals(4)
        self.trim_lat.setToolTip("Site latitude in decimal degrees (north positive); "
                                 "used to split day from night.")
        form.addRow("Latitude", self.trim_lat)
        self.trim_lon = QDoubleSpinBox()
        self.trim_lon.setRange(-180.0, 180.0)
        self.trim_lon.setDecimals(4)
        self.trim_lon.setToolTip("Site longitude in decimal degrees (east positive); "
                                 "used to split day from night.")
        form.addRow("Longitude", self.trim_lon)
        self.trim_utc = QSpinBox()
        self.trim_utc.setRange(-12, 14)
        self.trim_utc.setToolTip("UTC offset (hours) of the timestamps; used to align "
                                 "solar position for the day/night split.")
        form.addRow("UTC offset (h)", self.trim_utc)
        self._coord_widgets = (self.trim_lat, self.trim_lon, self.trim_utc)
        self._sync_coord_enabled()  # disabled until a day/night box is ticked

    def _sync_coord_enabled(self, *_args) -> None:
        """Coordinates are only used for the day/night split, so enable them only
        when a side is selected; seed from the project site when first enabled."""
        on = self.trim_daytime.isChecked() or self.trim_nighttime.isChecked()
        for w in self._coord_widgets:
            w.setEnabled(on)
        if on:
            self._seed_trim_site()

    def _mark_limit_touched(self) -> None:
        self._limit_touched = True

    def _seed_trim_site(self) -> None:
        """Prefill lat/lon/UTC from the app-wide Project settings, if configured."""
        m = site.manager
        if not m.configured:
            return
        self.trim_lat.setValue(m.latitude)
        self.trim_lon.setValue(m.longitude)
        self.trim_utc.setValue(m.utc_offset)

    def _on_site_changed(self) -> None:
        # Keep coordinates in sync with edits to the project's site details, but
        # only while a day/night split (which uses them) is selected.
        if self.trim_daytime.isChecked() or self.trim_nighttime.isChecked():
            self._seed_trim_site()

    def _select(self, name: str) -> None:
        super()._select(name)
        # Seed the lower limit from the selected series' minimum (so nothing is
        # flagged initially); skip once the user has edited it by hand.
        if self._limit_touched or self._df is None or name not in self._df.columns:
            return
        s = self._df[name].dropna()
        if not len(s):
            return
        self.lower_limit.blockSignals(True)
        self.lower_limit.setValue(float(s.min()))
        self.lower_limit.blockSignals(False)

    def _state_controls(self) -> dict:
        return {**super()._state_controls(),
                "lower_limit": self.lower_limit,
                "trim_daytime": self.trim_daytime,
                "trim_nighttime": self.trim_nighttime,
                "trim_lat": self.trim_lat, "trim_lon": self.trim_lon,
                "trim_utc": self.trim_utc}

    def _current_kwargs(self) -> dict:
        kwargs = dict(
            lower_limit=self.lower_limit.value(),
            trim_daytime=self.trim_daytime.isChecked(),
            trim_nighttime=self.trim_nighttime.isChecked(),
        )
        # Coordinates are only passed (and required) when a day/night split is on;
        # omitting them in trim-all mode keeps the call — and the codegen — clean.
        if self.trim_daytime.isChecked() or self.trim_nighttime.isChecked():
            kwargs.update(lat=self.trim_lat.value(), lon=self.trim_lon.value(),
                          utc_offset=self.trim_utc.value())
        return kwargs

    def _make_detector(self, series, kwargs: dict):
        return dv.outliers.TrimLow(series=series, **kwargs)

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        return trimlow_to_code(kwargs, repeat=repeat, var_name=var_name)
