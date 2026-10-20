"""
GUI.TABS.STEPWISE: STEPWISE OUTLIER SCREENING TAB
=================================================

Chain several outlier tests on one variable of the working dataset as a list of
editable **method cards**, apply corrections, and inspect what each step removes
plus the overall **QCF**. This is the plain (no-resampling) variant of the shared
screening experience — all the machinery lives in
:class:`~diive.gui.tabs._screening_base.ScreeningTabBase`; the database variant
(:mod:`diive.gui.tabs.meteo_screening`) adds resampling.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import shiboken6
from PySide6.QtWidgets import QWidget

from diive.gui import site
from diive.gui.tabs._screening_base import ScreeningTabBase
from diive.gui.widgets.project_offset import NOT_SET_WARNING, project_utc_offset_is_set


class StepwiseScreeningTab(ScreeningTabBase):
    """Chain outlier tests + corrections + QCF on a working-dataset variable.

    The variable list is the working dataset's columns, and 'Add' emits the
    per-test flags, the QCF flag, the QCF-filtered series, and (if any) the
    corrected series. Coordinates and the UTC offset come from Project settings;
    results computed with other ones are cleared when those settings change.
    """

    title = "Stepwise screening"

    def build(self) -> QWidget:
        root = super().build()
        self._site_coords: dict = self._coords()
        # Bound method, not a lambda: site.manager is a singleton.
        site.manager.changed.connect(self._on_site_changed)
        return root

    def _on_site_changed(self) -> None:
        """Project settings saved: clear results computed with the old coordinates
        or UTC offset and mark the run buttons pending (as a new selection does)."""
        if not shiboken6.isValid(self.status):
            return  # site.manager outlives the tab: its widgets may be deleted
        self.corrections_panel.set_coords_available(site.manager.configured)
        coords = self._coords()
        if coords == self._site_coords:
            return  # e.g. only the site name or notes changed
        self._site_coords = coords
        if self._var is None or (self._payload is None and self._corrected is None):
            return
        self._show_variable(self._var, redetect_measurement=False)
        self.status.setText(
            "Project settings changed: click Run outliers / Run corrections to "
            f"screen {self._var} with the new coordinates and UTC offset.")

    def _on_done(self, payload: dict) -> None:
        super()._on_done(payload)
        if payload.get("run_id") == self._run_id and not project_utc_offset_is_set():
            self.status.setText(f"{self.status.text()} {NOT_SET_WARNING}")
