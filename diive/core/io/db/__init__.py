"""
CORE.IO.DB: DATABASE BACKENDS
=============================

A small ports-and-adapters layer for connecting diive to a database. The
:class:`DatabaseBackend` port is the backend-agnostic contract; concrete
adapters implement it for a specific database. :class:`InfluxDBBackend` (over
the optional ``dbc-influxdb`` package) is the first adapter.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.core.io.db.base import DatabaseBackend
from diive.core.io.db.influxdb import INSTALL_HINT, InfluxDBBackend, influxdb_available

__all__ = [
    "DatabaseBackend",
    "InfluxDBBackend",
    "influxdb_available",
    "INSTALL_HINT",
]
