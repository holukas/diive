"""
CORE.IO.DB.INFLUX: INFLUXDB I/O ENGINE
======================================

diive's in-house InfluxDB v2 engine (download / upload / delete / schema
browsing), a clean port of the former external ``dbc-influxdb`` package. The
optional ``db`` dependency group provides the only third-party requirement
(``influxdb-client``); the client is imported lazily, so importing this package
never requires the ``db`` extra.

The GUI talks to this through the generic backend adapter in
:mod:`diive.core.io.db.influxdb`, not directly.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.core.io.db.influx.influxio import InfluxIO

__all__ = ["InfluxIO"]
