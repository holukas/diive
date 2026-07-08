"""
GUI.TABS.SITE: PROJECT SETTINGS

Enter the project's settings — the author name, a free-text project description,
and the measurement site's metadata (site name, latitude, longitude, elevation,
UTC offset) — and store them app-wide in ``diive.gui.site.manager`` so diive
functions that need site coordinates (daytime/nighttime separation, the flux
processing chain, ...) can reuse them, and so they travel with a saved project.
The form reads the current values on build and writes them back through
``site.manager.update`` on **Save**; persistence is handled with the other GUI
preferences (``config.py``) and inside ``.diive`` projects (``app.py`` extras).

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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

#: Width cap for the form column so fields/buttons stay readable instead of
#: stretching across a wide window.
_COL_WIDTH = 480

from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.notes_wall import NotesWall
from diive.gui.widgets.tab_chrome import build_titlebar


class ProjectSettingsTab(DiiveTab):
    """Form for the project's author/description + site details, stored in
    ``site.manager`` and saved with the project."""

    title = "Project settings"

    def build(self) -> QWidget:
        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)
        root_lay.addLayout(build_titlebar(self.title))  # shared tab header
        body = QWidget()
        outer = QHBoxLayout(body)
        outer.setContentsMargins(24, 24, 24, 24)

        # A single, width-capped column anchored to the top-left (a trailing
        # stretch keeps it from stretching across the window; no centring spacers,
        # which left it floating in the middle of an empty tab).
        column = QWidget()
        column.setMaximumWidth(_COL_WIDTH)
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(14)

        intro = QLabel(
            "Settings for this project — your name, a description, and the "
            "measurement site. The site coordinates and UTC offset are reused by "
            "functions that need them (e.g. daytime/nighttime separation, the flux "
            "processing chain). All of these are saved with the project.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6B7780;")
        col.addWidget(intro)

        # --- Project (author + free-text description) ---
        proj_box = QGroupBox("Project")
        proj = QFormLayout(proj_box)
        self.author = QLineEdit()
        self.author.setPlaceholderText("e.g. Jane Doe")
        proj.addRow("Your name", self.author)
        self.description = QPlainTextEdit()
        self.description.setPlaceholderText(
            "Free-text notes about this project (purpose, data source, processing "
            "decisions, ...).")
        self.description.setMinimumHeight(96)
        proj.addRow("Description", self.description)
        col.addWidget(proj_box)

        # --- Site details (coordinates / UTC offset) ---
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
        self.save_btn = QPushButton("Save project settings")
        self.save_btn.clicked.connect(self._save)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch(1)
        col.addLayout(btn_row)

        self.status = QLabel()
        self.status.setWordWrap(True)
        col.addWidget(self.status)

        col.addStretch(1)  # keep the rows packed at the top of the column

        # Anchor the form column to the left; the notes wall fills the rest of the
        # otherwise-empty space.
        outer.addWidget(column)

        notes_col = QVBoxLayout()
        notes_col.setContentsMargins(0, 0, 0, 0)
        notes_col.setSpacing(8)
        header = QLabel(theme.manager.label_text("Notes"))
        hf = theme.manager.tracked_font(header.font())
        hf.setBold(True)
        header.setFont(hf)
        notes_col.addWidget(header)
        self.notes = NotesWall()
        self.notes.set_state(site.manager.notes)
        self.notes.changed.connect(self._on_notes_changed)
        notes_col.addWidget(self.notes, stretch=1)
        outer.addLayout(notes_col, stretch=1)

        root_lay.addWidget(body, stretch=1)
        self._load_from_manager()
        # Keep the form in sync if another part of the app updates the site.
        site.manager.changed.connect(self._load_from_manager)
        return root

    def _on_notes_changed(self) -> None:
        """Mirror the wall into the store so it travels with the project / prefs.
        A plain attribute set (no ``changed`` signal) avoids a rebuild loop."""
        site.manager.notes = self.notes.state()

    def _load_from_manager(self) -> None:
        m = site.manager
        for w, val in (
            (self.lat, m.latitude), (self.lon, m.longitude),
            (self.elevation, m.elevation), (self.utc, m.utc_offset),
        ):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        for w, text in ((self.name, m.name), (self.author, m.author)):
            w.blockSignals(True)
            w.setText(text)
            w.blockSignals(False)
        self.description.blockSignals(True)
        self.description.setPlainText(m.description)
        self.description.blockSignals(False)
        # Rebuild the notes wall only when the store's notes genuinely differ from
        # what's on the wall (e.g. a project was opened), so unrelated `changed`
        # signals — like the Save below — don't wipe an in-progress edit.
        if hasattr(self, "notes") and m.notes != self.notes.state():
            self.notes.set_state(m.notes)
        self.status.setText(
            "Stored." if m.configured else "Not set yet — fill in and Save.")

    def _save(self) -> None:
        site.manager.update(
            name=self.name.text().strip(),
            author=self.author.text().strip(),
            description=self.description.toPlainText().strip(),
            latitude=self.lat.value(),
            longitude=self.lon.value(),
            elevation=self.elevation.value(),
            utc_offset=self.utc.value(),
        )
        label = self.name.text().strip() or "site"
        self.status.setText(
            f"Saved project settings ({label}: {self.lat.value():.5f} °N, "
            f"{self.lon.value():.5f} °E, UTC {self.utc.value():+d}). "
            f"Saved with the project and reused by functions that need site details.")


#: Backwards-compatible alias (the tab was previously named SiteDetailsTab).
SiteDetailsTab = ProjectSettingsTab
