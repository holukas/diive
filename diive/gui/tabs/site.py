"""
GUI.TABS.SITE: SITE DETAILS

Enter the measurement site's metadata (name, latitude, longitude, elevation, UTC
offset) once and store it app-wide in ``diive.gui.site.manager`` so diive
functions that need site coordinates (daytime/nighttime separation, the flux
processing chain, ...) can reuse it. The form reads the current values on build
and writes them back through ``site.manager.update`` on **Save**; persistence is
handled with the other GUI preferences (``config.py``).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

#: Width cap for the form column so fields/buttons stay compact instead of
#: stretching across a wide window.
_COL_WIDTH = 360

from diive.gui import site
from diive.gui.tabs.base import DiiveTab


class SiteDetailsTab(DiiveTab):
    """Form for the site's coordinates / UTC offset, stored in ``site.manager``."""

    title = "Site details"

    def build(self) -> QWidget:
        root = QWidget()
        outer = QVBoxLayout(root)

        # A single, width-capped column so the inputs/buttons stay compact and
        # left-aligned rather than stretching across the whole window.
        column = QWidget()
        column.setMaximumWidth(_COL_WIDTH)
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 0, 0)

        intro = QLabel(
            "Describe the measurement site once here. The coordinates and UTC "
            "offset are stored app-wide and reused by functions that need them "
            "(e.g. daytime/nighttime separation, the flux processing chain).")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6B7780;")
        col.addWidget(intro)

        box = QGroupBox("Site details")
        form = QFormLayout(box)

        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. CH-DAV")
        form.addRow("Site name", self.name)

        self.lat = QDoubleSpinBox()
        self.lat.setRange(-90.0, 90.0)
        self.lat.setDecimals(6)
        self.lat.setSuffix(" °N")
        form.addRow("Latitude", self.lat)

        self.lon = QDoubleSpinBox()
        self.lon.setRange(-180.0, 180.0)
        self.lon.setDecimals(6)
        self.lon.setSuffix(" °E")
        form.addRow("Longitude", self.lon)

        self.elevation = QDoubleSpinBox()
        self.elevation.setRange(-500.0, 9000.0)
        self.elevation.setDecimals(1)
        self.elevation.setSuffix(" m")
        form.addRow("Elevation", self.elevation)

        self.utc = QSpinBox()
        self.utc.setRange(-12, 14)
        self.utc.setPrefix("UTC ")
        self.utc.setValue(0)
        form.addRow("UTC offset (h)", self.utc)

        col.addWidget(box)

        # Save button takes its natural width (a trailing stretch keeps it left).
        self.save_btn = QPushButton("Save site details")
        self.save_btn.clicked.connect(self._save)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch(1)
        col.addLayout(btn_row)

        self.status = QLabel()
        self.status.setWordWrap(True)
        col.addWidget(self.status)

        # Centre the compact column in the tab via spacers on all four sides.
        # (An alignment flag on addWidget would instead under-size the column and
        # squash the input rows, clipping their text.)
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(column)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        self._load_from_manager()
        # Keep the form in sync if another part of the app updates the site.
        site.manager.changed.connect(self._load_from_manager)
        return root

    def _load_from_manager(self) -> None:
        m = site.manager
        for w, val in (
            (self.lat, m.latitude), (self.lon, m.longitude),
            (self.elevation, m.elevation), (self.utc, m.utc_offset),
        ):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self.name.blockSignals(True)
        self.name.setText(m.name)
        self.name.blockSignals(False)
        self.status.setText(
            "Stored." if m.configured else "Not set yet — fill in and Save.")

    def _save(self) -> None:
        site.manager.update(
            name=self.name.text().strip(),
            latitude=self.lat.value(),
            longitude=self.lon.value(),
            elevation=self.elevation.value(),
            utc_offset=self.utc.value(),
        )
        label = self.name.text().strip() or "site"
        self.status.setText(
            f"Saved {label}: {self.lat.value():.5f} °N, {self.lon.value():.5f} °E, "
            f"UTC {self.utc.value():+d}. Reused by functions that need site details.")
