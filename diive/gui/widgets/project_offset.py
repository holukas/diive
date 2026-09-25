"""
GUI.WIDGETS.PROJECT_OFFSET: THE PROJECT'S UTC OFFSET, READ-ONLY
===============================================================

One source for the UTC offset in the GUI: the value set under **Project
settings**. Tabs that need an offset show this widget instead of their own
editable field, so every computation (day/night split, potential radiation,
database download) uses the same offset. The widget is an inactive field showing
the project's offset; when Project settings have none, it reads "not set" and a
red exclamation mark tells the user where to set it, and
:meth:`ProjectUtcOffset.value` returns 0.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from diive.gui import site, theme

#: Shown wherever an offset is needed but Project settings have none.
NOT_SET_WARNING = ("UTC offset not set: using UTC+00:00. Set it under "
                   "Settings > Project settings.")

_SET_TOOLTIP = ("UTC offset from Project settings. The timestamps are taken to be "
                "in this offset. Change it under Settings > Project settings.")
_NOT_SET_TOOLTIP = "Set the UTC offset in Settings > Project settings first."


def project_utc_offset() -> int:
    """Return the project's UTC offset in hours (0 when it is not set)."""
    return int(site.manager.utc_offset) if site.manager.configured else 0


def project_utc_offset_is_set() -> bool:
    """True once the UTC offset has been saved in Project settings."""
    return bool(site.manager.configured)


def format_utc_offset(hours: int) -> str:
    """Format an offset in hours as ``UTC+01:00``."""
    sign = "+" if hours >= 0 else "-"
    return f"UTC{sign}{abs(int(hours)):02d}:00"


class ProjectUtcOffset(QWidget):
    """Inactive display of the project's UTC offset, with a warning mark when unset.

    Drop-in for the editable offset spin boxes the tabs used to have: read the
    offset with :meth:`value`. Follows ``site.manager`` and emits ``changed``
    when the project settings change.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.field = QLineEdit()
        self.field.setReadOnly(True)
        self.field.setEnabled(False)
        layout.addWidget(self.field, 1)
        self.warning_mark = QLabel("!")
        self.warning_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_mark.setFixedSize(18, 18)
        self.warning_mark.setToolTip(_NOT_SET_TOOLTIP)
        layout.addWidget(self.warning_mark)
        self._refresh()
        # Bound method, not a lambda: the singleton must be able to drop the
        # connection when this widget is deleted.
        site.manager.changed.connect(self._on_site_changed)

    def value(self) -> int:
        """The offset in hours to pass to library functions (0 when unset)."""
        return project_utc_offset()

    def is_set(self) -> bool:
        """True once Project settings define the offset."""
        return project_utc_offset_is_set()

    def text(self) -> str:
        """What the field shows, e.g. ``UTC+01:00`` or ``not set``."""
        return self.field.text()

    def _on_site_changed(self) -> None:
        self._refresh()
        self.changed.emit()

    def _refresh(self) -> None:
        is_set = self.is_set()
        self.field.setText(format_utc_offset(self.value()) if is_set else "not set")
        tip = _SET_TOOLTIP if is_set else _NOT_SET_TOOLTIP
        self.field.setToolTip(tip)
        self.setToolTip(tip)
        self.warning_mark.setVisible(not is_set)
        danger = theme.manager.tokens.get("DANGER_BG", "#E04646")
        self.warning_mark.setStyleSheet(
            f"QLabel {{ background: {danger}; color: white; font-weight: 700; "
            f"border-radius: 9px; }}" + theme.manager.tooltip_qss())
