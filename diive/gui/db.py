"""
GUI.DB: DATABASE CONNECTION STORE
=================================

A tiny app-wide store for the current database connection. It holds the active
:class:`~diive.core.io.db.base.DatabaseBackend` (a generic port; InfluxDB is the
first adapter) and the InfluxDB config-directory path used to build it.

Secrets are NOT kept here and NOT persisted — only the directory path is
remembered (safe to write to the plaintext GUI prefs). The token stays in the
``<dir>_secret`` file on disk, read by the backend each time it is built.

This holds *state + a handle only* (no domain logic): connecting, browsing and
download/upload are the backend's job (library + ``dbc-influxdb``). The InfluxDB
backend needs the optional ``db`` dependency group; :func:`influxdb_available`
reports whether it is installed so the GUI can degrade gracefully.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from diive.core.io.db import (
    INSTALL_HINT,
    DatabaseBackend,
    InfluxDBBackend,
    influxdb_available,
)

__all__ = ["DbConnectionManager", "manager", "influxdb_available", "INSTALL_HINT"]


class DbConnectionManager(QObject):
    """Live, app-wide holder of the current database connection.

    Access the singleton as ``db.manager``. Read :attr:`backend` (``None`` until
    connected) and :attr:`connected`; build a connection with :meth:`connect`
    (run it off the GUI thread — it does network I/O). ``changed`` fires on every
    state change so the header pill, the connection tab and the explorer refresh.
    """

    #: Emitted when the path or connection state changes.
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.dirconf: str = ""              # InfluxDB config dir (persisted)
        self.backend: DatabaseBackend | None = None
        self.connected: bool = False

    def set_dirconf(self, path: str) -> None:
        """Point at a config directory; drops any existing connection."""
        self.dirconf = path
        self.backend = None
        self.connected = False
        self.changed.emit()

    def connect(self) -> DatabaseBackend:
        """Build and verify the InfluxDB backend from ``dirconf``.

        Returns the live backend (also stored on :attr:`backend`). Raises if the
        ``db`` extra is missing, the config can't be read, or the server is
        unreachable. Run off the GUI thread (see :class:`WorkerRunner`); the
        caller marks :attr:`connected` and emits :attr:`changed` on success.
        """
        if not influxdb_available():
            raise RuntimeError(INSTALL_HINT)
        backend = InfluxDBBackend(dirconf=self.dirconf)  # constructs + pings
        self.backend = backend
        return backend

    def mark_connected(self) -> None:
        """Flag the (already built) backend as connected and notify listeners."""
        self.connected = self.backend is not None
        self.changed.emit()

    def disconnect(self) -> None:
        """Drop the active connection (keeps the remembered path)."""
        self.backend = None
        self.connected = False
        self.changed.emit()

    def status_text(self) -> str:
        """Pill/label text for the header: what we are connected to, if anything."""
        if self.connected and self.backend is not None:
            return f"Connected: {self.backend.describe()}"
        return "Not connected"

    def as_dict(self) -> dict:
        """Serialise for persistence — path only, never the secret token."""
        return {"dirconf": self.dirconf}

    def load_dict(self, data: dict) -> None:
        """Restore from a persisted dict (missing keys keep their defaults)."""
        if data:
            self.dirconf = str(data.get("dirconf", self.dirconf))


#: Process-wide singleton; import as ``from diive.gui import db`` then ``db.manager``.
manager = DbConnectionManager()
