"""
CORE.IO.DB.INFLUXDB: INFLUXDB ADAPTER
=====================================

The InfluxDB adapter for the :class:`~diive.core.io.db.base.DatabaseBackend`
port. It is a thin wrapper over the ``dbc-influxdb`` package (``dbcInflux``),
which owns the actual InfluxDB I/O and the POET config-directory conventions.
diive only adapts that API to the generic backend contract plus the InfluxDB
schema-browsing helpers the GUI explorer needs.

``dbc-influxdb`` is the optional ``db`` dependency group (``uv sync --group db``)
and is imported lazily inside the constructor, so importing this module never
requires it; :func:`influxdb_available` reports whether it can be used.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.core.io.db.base import DatabaseBackend

#: Shown when the optional dependency is missing.
INSTALL_HINT = ("InfluxDB support needs the optional 'db' extra:\n"
                "    uv sync --group db\n"
                "(installs dbc-influxdb).")


def influxdb_available() -> bool:
    """True if the optional ``dbc-influxdb`` dependency is importable."""
    try:
        import dbc_influxdb  # noqa: F401
        return True
    except ImportError:
        return False


class InfluxDBBackend(DatabaseBackend):
    """InfluxDB connection backed by ``dbc-influxdb``.

    Constructed from a ``dbc-influxdb`` config-directory path; the secret
    url/org/token are read from that directory's ``<dir>_secret`` sibling.
    Construction also pings the server, so a successfully built backend is live.
    """

    name = "InfluxDB"

    def __init__(self, dirconf: str) -> None:
        from dbc_influxdb import dbcInflux  # lazy: only this adapter needs it
        self.dirconf = dirconf
        # Constructor reads <dirconf>_secret/dbconf.yaml and pings the server,
        # raising on bad config / unreachable DB.
        self._dbc = dbcInflux(dirconf=dirconf)

    @property
    def url(self) -> str:
        return str(self._dbc.conf_db.get("url", ""))

    @property
    def org(self) -> str:
        return str(self._dbc.conf_db.get("org", ""))

    def test_connection(self) -> None:
        self._dbc.test_connection()

    def describe(self) -> str:
        return f"InfluxDB @ {self.url}" if self.url else "InfluxDB"

    # --- InfluxDB schema browsing (data-model-specific) ---

    def list_buckets(self) -> list[str]:
        """All (non-system) buckets in the database."""
        return self._dbc.show_buckets()

    def list_data_versions(self, bucket: str) -> list[str]:
        """Distinct data versions (``data_version`` tag) in *bucket*."""
        return self._dbc.show_data_versions_in_bucket(bucket=bucket, verbose=False)

    def list_measurements(self, bucket: str, data_version: str | None = None) -> list[str]:
        """Measurements in *bucket*, optionally narrowed to one *data_version*."""
        return self._dbc.show_measurements_in_bucket(
            bucket=bucket, data_version=data_version, verbose=False)

    def list_fields(self, bucket: str, measurement: str,
                    data_version: str | None = None) -> list[str]:
        """Fields (variable names) in *measurement*, optionally narrowed to one
        *data_version*."""
        return self._dbc.show_fields_in_measurement(
            bucket=bucket, measurement=measurement, data_version=data_version,
            verbose=False)

    def list_units(self, bucket: str, measurement: str, field: str,
                   data_version: str | None = None) -> list[str]:
        """Distinct units (``units`` tag) for *field* of *measurement*,
        optionally narrowed to one *data_version*."""
        return self._dbc.show_units_in_field(
            bucket=bucket, measurement=measurement, field=field,
            data_version=data_version, verbose=False)
