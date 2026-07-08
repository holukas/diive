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

    def field_overview(self, bucket: str, measurement: str, field: str,
                       data_version: str | None = None) -> dict:
        """Full overview of *field*: all tags (incl. units) plus first / last
        record timestamps. See
        :meth:`~diive.core.io.db.influx.influxio.InfluxIO.show_field_overview`."""
        return self._io.show_field_overview(
            bucket=bucket, measurement=measurement, field=field,
            data_version=data_version, verbose=False)

    def download_field(self, bucket: str, measurement: str, field: str,
                       start: str, stop: str, data_version: str | None = None,
                       utc_offset: int | float = 0):
        """Download one *field* between *start* and *stop* as a pandas Series.

        Times are interpreted in / returned for the ``utc_offset`` timezone
        (default UTC, since the explorer browses the database directly). Returns
        the field's data Series (empty Series if no records in range)."""
        data_simple, _, _ = self._io.download(
            bucket=bucket, start=start, stop=stop,
            timezone_offset_to_utc_hours=utc_offset,
            data_version=[data_version] if data_version else None,
            measurements=[measurement], fields=[field])
        if field in data_simple.columns:
            return data_simple[field]
        import pandas as pd
        return pd.Series(dtype="float64", name=field)

    def download_detailed(self, bucket: str, measurement: str, field: str,
                          start: str, stop: str, data_version: str | None = None,
                          utc_offset: int | float = 1) -> dict:
        """Download one *field* with its tags as a ``data_detailed`` dict.

        Returns ``{field: DataFrame}`` where the DataFrame carries the field's
        data column plus every database tag (units, gain, offset, data_version,
        ...) on a ``TIMESTAMP_END`` index — the input shape
        :class:`~diive.qaqc.StepwiseMeteoScreeningDb` consumes. Times are
        interpreted in the ``utc_offset`` timezone (default CET, +1)."""
        _, data_detailed, _ = self._io.download(
            bucket=bucket, start=start, stop=stop,
            timezone_offset_to_utc_hours=utc_offset,
            data_version=[data_version] if data_version else None,
            measurements=[measurement], fields=[field])
        return data_detailed

    @staticmethod
    def _chunk_bounds(start: str, stop: str, n_chunks: int) -> list[str]:
        """Split ``[start, stop]`` into ``n_chunks`` adjacent sub-ranges.

        Returns ``n_chunks + 1`` boundary strings; chunk i is ``[b[i], b[i+1])``
        (download stop is exclusive, so chunks neither overlap nor gap)."""
        import pandas as pd
        n = max(1, int(n_chunks))
        edges = pd.date_range(pd.Timestamp(start), pd.Timestamp(stop), periods=n + 1)
        return [e.strftime("%Y-%m-%d %H:%M:%S") for e in edges]

    def download_field_chunked(self, bucket: str, measurement: str, field: str,
                               start: str, stop: str, data_version: str | None = None,
                               utc_offset: int | float = 1, n_chunks: int = 1,
                               progress_callback=None):
        """Download a field in *n_chunks* time slices, accumulating the Series.

        After each chunk, calls ``progress_callback(accumulated_series, done, total)``
        so the GUI can show a growing plot + determinate progress. Returns the full
        Series."""
        import pandas as pd
        bounds = self._chunk_bounds(start, stop, n_chunks)
        acc = pd.Series(dtype="float64", name=field)
        for i in range(len(bounds) - 1):
            s = self.download_field(bucket, measurement, field, bounds[i],
                                    bounds[i + 1], data_version, utc_offset)
            if s is not None and not s.empty:
                acc = pd.concat([acc, s])
                acc = acc[~acc.index.duplicated(keep="last")].sort_index()
            if progress_callback is not None:
                progress_callback(acc, i + 1, len(bounds) - 1)
        return acc

    def download_detailed_chunked(self, bucket: str, measurement: str, field: str,
                                  start: str, stop: str, data_version: str | None = None,
                                  utc_offset: int | float = 1, n_chunks: int = 1,
                                  progress_callback=None) -> dict:
        """Download a field's ``data_detailed`` in *n_chunks* slices, merging tags.

        After each chunk, calls ``progress_callback(accumulated_field_series, done,
        total)`` (the field's data column, for a growing plot). Returns the merged
        ``data_detailed`` dict."""
        bounds = self._chunk_bounds(start, stop, n_chunks)
        merged: dict = {}
        for i in range(len(bounds) - 1):
            dd = self.download_detailed(bucket, measurement, field, bounds[i],
                                        bounds[i + 1], data_version, utc_offset)
            for key, frame in dd.items():
                if key in merged:
                    merged[key] = merged[key].combine_first(frame)
                    merged[key] = merged[key][
                        ~merged[key].index.duplicated(keep="last")].sort_index()
                else:
                    merged[key] = frame
            if progress_callback is not None:
                acc = merged.get(field)
                series = acc[field] if (acc is not None and field in acc.columns) else None
                progress_callback(series, i + 1, len(bounds) - 1)
        return merged
