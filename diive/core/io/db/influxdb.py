"""
CORE.IO.DB.INFLUXDB: INFLUXDB BACKEND ADAPTER
=============================================

The InfluxDB adapter for the :class:`~diive.core.io.db.base.DatabaseBackend`
port. It wraps diive's in-house InfluxDB engine
(:class:`~diive.core.io.db.influx.influxio.InfluxIO`), adapting it to the generic
backend contract plus the schema-browsing helpers the GUI explorer needs.

The engine needs the optional ``db`` dependency group (``uv sync --group db``,
which installs ``influxdb-client``) and imports the client lazily, so importing
this module never requires it; :func:`influxdb_available` reports whether it can
be used.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.core.io.db.base import DatabaseBackend

#: Shown when the optional dependency is missing.
INSTALL_HINT = ("InfluxDB support needs the optional 'db' dependency group:\n"
                "    uv sync --group db\n"
                "(installs influxdb-client).")


def influxdb_available() -> bool:
    """True if the optional ``influxdb-client`` dependency is importable."""
    try:
        import influxdb_client  # noqa: F401
        return True
    except ImportError:
        return False


class InfluxDBBackend(DatabaseBackend):
    """InfluxDB connection backed by :class:`~diive.core.io.db.influx.influxio.InfluxIO`.

    Constructed from a config-directory path; the secret url/org/token are read
    from that directory's ``<dir>_secret`` sibling. Construction also pings the
    server, so a successfully built backend is live.
    """

    name = "InfluxDB"

    def __init__(self, dirconf: str) -> None:
        from diive.core.io.db.influx import InfluxIO  # lazy: needs the 'db' extra
        self.dirconf = dirconf
        # Constructor reads <dirconf>_secret/dbconf.yaml and pings the server,
        # raising on bad config / unreachable DB.
        self._io = InfluxIO(dirconf=dirconf)

    @property
    def url(self) -> str:
        return str(self._io.conf_db.get("url", ""))

    @property
    def org(self) -> str:
        return str(self._io.conf_db.get("org", ""))

    def test_connection(self) -> None:
        self._io.test_connection()

    def describe(self) -> str:
        return f"InfluxDB @ {self.url}" if self.url else "InfluxDB"

    # --- InfluxDB schema browsing (data-model-specific) ---

    def list_buckets(self) -> list[str]:
        """All (non-system) buckets in the database."""
        return self._io.show_buckets()

    def list_data_versions(self, bucket: str) -> list[str]:
        """Distinct data versions (``data_version`` tag) in *bucket*."""
        return self._io.show_data_versions_in_bucket(bucket=bucket, verbose=False)

    def list_measurements(self, bucket: str, data_version: str | None = None) -> list[str]:
        """Measurements in *bucket*, optionally narrowed to one *data_version*."""
        return self._io.show_measurements_in_bucket(
            bucket=bucket, data_version=data_version, verbose=False)

    def list_fields(self, bucket: str, measurement: str,
                    data_version: str | None = None) -> list[str]:
        """Fields (variable names) in *measurement*, optionally narrowed to one
        *data_version*."""
        return self._io.show_fields_in_measurement(
            bucket=bucket, measurement=measurement, data_version=data_version,
            verbose=False)

    def list_units(self, bucket: str, measurement: str, field: str,
                   data_version: str | None = None) -> list[str]:
        """Distinct units (``units`` tag) for *field* of *measurement*,
        optionally narrowed to one *data_version*."""
        return self._io.show_units_in_field(
            bucket=bucket, measurement=measurement, field=field,
            data_version=data_version, verbose=False)
