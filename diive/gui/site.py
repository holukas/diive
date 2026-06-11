"""
GUI.SITE: PROJECT SETTINGS STORE
================================

A tiny app-wide store for the project's settings — the author name, a free-text
project description, and the measurement site's metadata (site name, latitude,
longitude, elevation, UTC offset). Entered in the **Settings ▸ Project settings**
tab and reused wherever a diive function needs site coordinates (e.g.
daytime/nighttime separation, the flux processing chain).

This holds *values only* (no domain logic): the GUI collects them here and passes
them to the library functions, which already accept ``lat`` / ``lon`` /
``utc_offset`` arguments. Persisted with the other GUI preferences via
``config.py`` (``as_dict`` / ``load_dict``). A ``changed`` signal lets dependent
tabs refresh when the site is updated.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class SiteManager(QObject):
    """Live, app-wide holder of the current site's metadata.

    Access the singleton as ``site.manager``. Read the fields directly
    (``manager.latitude`` etc.); write them through :meth:`update` so the
    ``changed`` signal fires. ``configured`` is False until the user saves the
    site details at least once, so callers can tell "unset" from a deliberate
    ``(0, 0)``.
    """

    #: Emitted after :meth:`update` so dependent widgets can refresh.
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.name: str = ""
        self.author: str = ""        # the user's name (project author)
        self.description: str = ""   # free-text notes about the project
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.elevation: float = 0.0
        self.utc_offset: int = 0
        self.configured: bool = False
        #: Free-form sticky notes shown on the Project settings "wall"; a list of
        #: card dicts (title, body, color, x, y, w, h). Presentation-only; managed
        #: by the GUI's notes wall. Travels with the project (and GUI prefs).
        self.notes: list = []

    def update(self, *, name: str, latitude: float, longitude: float,
               elevation: float, utc_offset: int, author: str = "",
               description: str = "") -> None:
        """Set all fields, mark the site configured, and emit ``changed``."""
        self.name = name
        self.author = author
        self.description = description
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.utc_offset = utc_offset
        self.configured = True
        self.changed.emit()

    def as_dict(self) -> dict:
        """Serialise for persistence (see ``config.py``)."""
        return {
            "name": self.name,
            "author": self.author,
            "description": self.description,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "utc_offset": self.utc_offset,
            "configured": self.configured,
            "notes": list(self.notes),
        }

    def load_dict(self, data: dict) -> None:
        """Restore from a persisted dict (missing keys keep their defaults)."""
        if not data:
            return
        self.name = str(data.get("name", self.name))
        self.author = str(data.get("author", self.author))
        self.description = str(data.get("description", self.description))
        self.latitude = float(data.get("latitude", self.latitude))
        self.longitude = float(data.get("longitude", self.longitude))
        self.elevation = float(data.get("elevation", self.elevation))
        self.utc_offset = int(data.get("utc_offset", self.utc_offset))
        self.configured = bool(data.get("configured", self.configured))
        notes = data.get("notes")
        if isinstance(notes, list):
            self.notes = [dict(n) for n in notes if isinstance(n, dict)]


#: Process-wide singleton; import as ``from diive.gui import site`` then
#: ``site.manager``.
manager = SiteManager()
