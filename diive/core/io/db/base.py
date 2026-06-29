"""
CORE.IO.DB.BASE: DATABASE BACKEND PORT
======================================

The backend-agnostic interface diive uses to talk to a database. It is a *port*
in the ports-and-adapters sense: diive depends on this small abstract contract,
and each concrete database (InfluxDB, and others in future) is an *adapter* that
implements it (see :mod:`diive.core.io.db.influxdb`).

This module has **no** database dependencies — it only defines the contract — so
importing it never pulls in a database client. The heavy client libraries live
behind the adapters and their optional install extras.

The base contract is deliberately minimal: a backend identifies itself and can
verify it is reachable. Schema browsing (e.g. InfluxDB buckets / measurements /
fields) is data-model-specific, so it is declared on the concrete adapter that
has that data model, not forced onto every backend.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class DatabaseBackend(ABC):
    """A connection to one database.

    Concrete adapters (e.g. :class:`~diive.core.io.db.influxdb.InfluxDBBackend`)
    construct themselves from their own connection config and implement
    :meth:`test_connection`. The GUI holds the active backend behind this type
    so the connection seam stays generic; download/upload and any schema
    browsing are added by the adapters as their data models allow.
    """

    #: Human-readable backend name shown in the GUI (e.g. ``"InfluxDB"``).
    name: str = "Database"

    @abstractmethod
    def test_connection(self) -> None:
        """Verify the database is reachable. Raise on failure (do not return a
        bool — the raised error carries the reason to show the user)."""
        raise NotImplementedError

    def describe(self) -> str:
        """Short one-line description of what this backend is connected to,
        for status display. Override to add detail (e.g. the server URL)."""
        return self.name
