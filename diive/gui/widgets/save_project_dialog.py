"""
GUI.WIDGETS.SAVE_PROJECT_DIALOG: NAME + LOCATION FOR A NEW PROJECT
=================================================================

Collects the two things needed to save a diive project: a **project name** and a
**location** (parent folder). The project is written to ``<location>/<name>.diive``.
Presentation only — the actual write is the library's ``save_project``.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SaveProjectDialog(QDialog):
    """Ask for a project name and a parent location."""

    def __init__(self, default_name: str = "", default_location: str = "",
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save as project")
        self.setMinimumWidth(460)

        outer = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(default_name)
        self.name_edit.setPlaceholderText("e.g. CH-DAV 2023 screening")
        form.addRow("Project name", self.name_edit)

        loc_row = QWidget()
        loc_lay = QHBoxLayout(loc_row)
        loc_lay.setContentsMargins(0, 0, 0, 0)
        self.loc_edit = QLineEdit(default_location)
        self.loc_edit.setPlaceholderText("parent folder for the project")
        loc_lay.addWidget(self.loc_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        loc_lay.addWidget(browse)
        form.addRow("Location", loc_row)
        outer.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _browse(self) -> None:
        start = self.loc_edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Choose location", start)
        if chosen:
            self.loc_edit.setText(chosen)

    def _on_accept(self) -> None:
        name, location = self.values()
        if not name or not location or not Path(location).is_dir():
            return  # keep the dialog open until both are valid
        self.accept()

    def values(self) -> tuple[str, str]:
        return self.name_edit.text().strip(), self.loc_edit.text().strip()
