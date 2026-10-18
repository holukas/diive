import io
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from diive.core.io.db.influx import fluxql
from diive.core.io.db.influx.common import TAGS, convert_ts_to_timezone
from diive.core.io.db.influx.config import get_conf_filetypes, read_configfile
from diive.core.io.db.influx.influxio import InfluxIO


class TestInfluxFluxQL(unittest.TestCase):
    """Pure Flux-query-string builders (no database needed)."""

    def test_basic_builders(self):
        self.assertEqual(fluxql.bucketstring("ch-dav_raw"), 'from(bucket: "ch-dav_raw")')
        self.assertEqual(fluxql.rangestring("a", "b"), '|> range(start: a, stop: b)')
        self.assertIn('columnKey: ["_field"]', fluxql.pivotstring())

    def test_filterstring_or(self):
        out = fluxql.filterstring(queryfor="_field", querylist=["TA", "SW"], logic="or")
        self.assertEqual(
            out,
            '|> filter(fn: (r) => r["_field"] == "TA" or r["_field"] == "SW")')

    def test_predicatestring(self):
        self.assertEqual(fluxql.predicatestring({}), "(r) => true")
        self.assertEqual(
            fluxql.predicatestring({"data_version": "raw"}),
            '(r) => r["data_version"] == "raw"')
        self.assertEqual(
            fluxql.predicatestring({"_measurement": "TA", "data_version": "raw"}),
            '(r) => r["_measurement"] == "TA" and r["data_version"] == "raw"')

    def test_tag_values_query(self):
        q = " ".join(fluxql.tag_values(
            bucket="b", tag="units",
            conditions={"_measurement": "TA", "varname": "TA_T1_2_1"}).split())
        self.assertIn('schema.tagValues(', q)
        self.assertIn('bucket: "b"', q)
        self.assertIn('tag: "units"', q)
        self.assertIn('r["_measurement"] == "TA" and r["varname"] == "TA_T1_2_1"', q)
        self.assertIn("start: -9999d", q)

    def test_field_records_query(self):
        q = " ".join(fluxql.field_records(
            bucket="b", measurement="TA", field="TA_T1_2_1",
            data_version="raw", reducer="last").split())
        self.assertIn('from(bucket: "b")', q)
        self.assertIn("|> range(start: -9999d)", q)
        self.assertIn('r["_measurement"] == "TA" and r["_field"] == "TA_T1_2_1" '
                      'and r["data_version"] == "raw"', q)
        self.assertTrue(q.rstrip().endswith("|> last()"))

    def test_field_records_query_no_version(self):
        q = " ".join(fluxql.field_records(
            bucket="b", measurement="TA", field="TA_T1_2_1", reducer="first").split())
        self.assertNotIn("data_version", q)
        self.assertTrue(q.rstrip().endswith("|> first()"))

    def test_fields_in_measurement_query(self):
        q = " ".join(fluxql.fields_in_measurement("b", "TA", days=9999).split())
        self.assertIn("schema.measurementFieldKeys(", q)
        self.assertIn('measurement: "TA"', q)
        self.assertIn("start: -9999d", q)

    def test_measurements_in_bucket_query(self):
        # Without an explicit start the schema functions look back only 30d,
        # which hides buckets whose newest record is older than that.
        q = " ".join(fluxql.measurements_in_bucket("b").split())
        self.assertIn("schema.measurements(", q)
        self.assertIn('bucket: "b"', q)
        self.assertIn("start: -9999d", q)

    def test_fields_in_bucket_query(self):
        q = " ".join(fluxql.fields_in_bucket("b").split())
        self.assertIn("schema.fieldKeys(", q)
        self.assertIn('bucket: "b"', q)
        self.assertIn("start: -9999d", q)


class TestInfluxTimeHelpers(unittest.TestCase):
    """Timezone / ISO conversion helpers (no database needed)."""

    def test_format_utc_offset(self):
        self.assertEqual(InfluxIO._format_utc_offset(1), "+01:00")
        self.assertEqual(InfluxIO._format_utc_offset(-5), "-05:00")
        self.assertEqual(InfluxIO._format_utc_offset(10), "+10:00")
        self.assertEqual(InfluxIO._format_utc_offset(5.5), "+05:30")
        self.assertEqual(InfluxIO._format_utc_offset(0), "+00:00")

    def test_convert_datestr_to_iso8601(self):
        self.assertEqual(
            InfluxIO._convert_datestr_to_iso8601("2022-05-27 00:00:00", 1),
            "2022-05-27T00:00:00+01:00")
        self.assertEqual(
            InfluxIO._convert_datestr_to_iso8601("2022-05-27 12:30:00", -5),
            "2022-05-27T12:30:00-05:00")

    def test_convert_ts_to_timezone(self):
        # tz-aware UTC timestamps -> fixed +01:00 offset
        s = pd.Series(pd.to_datetime(["2022-01-01 00:00:00", "2022-01-01 01:00:00"], utc=True))
        out = convert_ts_to_timezone(timezone_offset_to_utc_hours=1, timestamp_index=s)
        # Same instant, shifted wall-clock representation by +1h.
        self.assertEqual(out.iloc[0].hour, 1)
        self.assertEqual(int(out.iloc[0].utcoffset().total_seconds()), 3600)

    def test_convert_ts_to_timezone_fractional(self):
        s = pd.Series(pd.to_datetime(["2022-01-01 00:00:00"], utc=True))
        out = convert_ts_to_timezone(timezone_offset_to_utc_hours=5.5, timestamp_index=s)
        self.assertEqual(int(out.iloc[0].utcoffset().total_seconds()), 5 * 3600 + 30 * 60)

    def test_convert_ts_to_timezone_accepts_datetimeindex(self):
        # A DatetimeIndex has no .dt accessor; it used to raise AttributeError
        # although the docstring and the parameter name promise index support.
        idx = pd.to_datetime(["2022-01-01 00:00:00", "2022-01-01 01:00:00"], utc=True)
        out = convert_ts_to_timezone(timezone_offset_to_utc_hours=1, timestamp_index=idx)
        self.assertEqual(out[0].hour, 1)
        self.assertEqual(int(out[0].utcoffset().total_seconds()), 3600)


class TestInfluxConfig(unittest.TestCase):
    """YAML config reading against a self-contained temp config directory."""

    def test_read_configfile_and_filetypes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "units.yaml").write_text("TA: degC\nSW: 'W m-2'\n", encoding="utf-8")
            fg = root / "filegroups"
            fg.mkdir()
            (fg / "example.yaml").write_text(
                "EXAMPLE_FILETYPE:\n  freq: 30min\n  delimiter: ','\n", encoding="utf-8")

            units = read_configfile(root / "units.yaml")
            self.assertEqual(units["TA"], "degC")

            filetypes = get_conf_filetypes(folder=fg)
            self.assertIn("EXAMPLE_FILETYPE", filetypes)
            self.assertEqual(filetypes["EXAMPLE_FILETYPE"]["freq"], "30min")

    @staticmethod
    def _make_dirconf(tmp, dbconf="url: http://127.0.0.1:1\ntoken: t\norg: o\n",
                      units="TA: degC\n", dirs="x: y\n", filegroups=None):
        """Write a config folder into *tmp*. None leaves a file out."""
        root = Path(tmp) / "conf"
        (root / "filegroups").mkdir(parents=True)
        for name, text in (filegroups or {"ft.yaml": "FT:\n  freq: 30min\n"}).items():
            (root / "filegroups" / name).write_text(text, encoding="utf-8")
        if units is not None:
            (root / "units.yaml").write_text(units, encoding="utf-8")
        if dirs is not None:
            (root / "dirs.yaml").write_text(dirs, encoding="utf-8")
        secret = Path(f"{root}_secret")
        secret.mkdir()
        if dbconf is not None:
            (secret / "dbconf.yaml").write_text(dbconf, encoding="utf-8")
        return root

    @staticmethod
    def _influxio(root):
        client = MagicMock()
        client.__enter__.return_value = client
        client.ping.return_value = True
        mod = "diive.core.io.db.influx.influxio"
        with patch(f"{mod}.get_client", return_value=client), patch(f"{mod}.success"), \
                patch(f"{mod}.info"), patch(f"{mod}.detail"):
            return InfluxIO(dirconf=str(root))

    def test_complete_config_folder(self):
        with TemporaryDirectory() as tmp:
            dbc = self._influxio(self._make_dirconf(tmp))
        self.assertEqual(dbc.conf_db["token"], "t")
        self.assertEqual(dbc.conf_unitmapper, {"TA": "degC"})
        self.assertEqual(dbc.conf_filetypes, {"FT": {"freq": "30min"}})

    def test_units_and_dirs_are_optional(self):
        with TemporaryDirectory() as tmp:
            dbc = self._influxio(self._make_dirconf(tmp, units=None, dirs=""))
        self.assertEqual(dbc.conf_unitmapper, {})
        self.assertEqual(dbc.conf_dirs, {})
        self.assertEqual(dbc.conf_db["url"], "http://127.0.0.1:1")

    def test_empty_filegroup_file_names_the_file(self):
        with TemporaryDirectory() as tmp:
            root = self._make_dirconf(tmp, filegroups={"broken.yaml": "# nothing\n"})
            with self.assertRaisesRegex(ValueError, r"broken\.yaml is empty"):
                self._influxio(root)

    def test_missing_dbconf_names_the_file(self):
        with TemporaryDirectory() as tmp:
            root = self._make_dirconf(tmp, dbconf=None)
            with self.assertRaisesRegex(FileNotFoundError, r"dbconf\.yaml"):
                self._influxio(root)

    def test_empty_dbconf_names_the_file(self):
        with TemporaryDirectory() as tmp:
            root = self._make_dirconf(tmp, dbconf="")
            with self.assertRaisesRegex(ValueError, r"dbconf\.yaml is empty"):
                self._influxio(root)

    def test_dbconf_missing_keys_are_listed(self):
        with TemporaryDirectory() as tmp:
            root = self._make_dirconf(tmp, dbconf="url: http://127.0.0.1:1\ntoken:\n")
            with self.assertRaisesRegex(ValueError, r"lacks a value for: token, org"):
                self._influxio(root)


class TestInfluxDelete(unittest.TestCase):
    """Delete predicates, recorded at the connection boundary (no database needed)."""

    def setUp(self):
        # Skip __init__: it reads config files and pings the server.
        self.dbc = InfluxIO.__new__(InfluxIO)
        self.dbc.conf_db = {}
        self.dbc.verbose = 0
        self.delete_api = MagicMock()
        patchers = [patch("diive.core.io.db.influx.influxio.get_client", return_value=MagicMock()),
                    patch("diive.core.io.db.influx.influxio.get_delete_api", return_value=self.delete_api)]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def _predicates(self):
        return [c.kwargs["predicate"] for c in self.delete_api.delete.call_args_list]

    def _delete(self, **kwargs):
        self.dbc.delete(bucket="b", measurements=["TA"], start="2022-01-01 00:00:00",
                        stop="2022-01-02 00:00:00", timezone_offset_to_utc_hours=1,
                        data_version="v1", **kwargs)

    def test_fields_list_without_site(self):
        self._delete(fields=["TA_1", "TA_2"])
        self.assertEqual(self._predicates(), [
            '_measurement="TA" AND varname="TA_1" AND data_version="v1"',
            '_measurement="TA" AND varname="TA_2" AND data_version="v1"'])

    def test_fields_list_with_site(self):
        self._delete(fields=["TA_1"], site="CH-DAV")
        self.assertEqual(self._predicates(), [
            '_measurement="TA" AND varname="TA_1" AND data_version="v1" AND site="CH-DAV"'])

    def test_all_fields_without_site(self):
        self._delete(fields=True)
        self.assertEqual(self._predicates(), ['_measurement="TA" AND data_version="v1"'])

    def test_all_fields_with_site(self):
        self._delete(fields=True, site="CH-DAV")
        self.assertEqual(self._predicates(),
                         ['_measurement="TA" AND data_version="v1" AND site="CH-DAV"'])

    def _var_df(self, sites):
        idx = pd.date_range("2022-01-01 00:30", periods=len(sites), freq="30min")
        df = pd.DataFrame({"TA_1": range(len(sites))}, index=idx)
        for tag in TAGS:
            df[tag] = "x"
        df["site"] = sites
        df["data_version"] = "v1"
        return df

    def test_upload_singlevar_deletes_only_own_site(self):
        with patch.object(InfluxIO, "delete") as delete:
            self.dbc.upload_singlevar(var_df=self._var_df(["CH-DAV", "CH-DAV"]), to_bucket="b",
                                      to_measurement="TA", timezone_offset_to_utc_hours=1)
        delete.assert_called_once()
        self.assertEqual(delete.call_args.kwargs["site"], "CH-DAV")
        self.assertEqual(delete.call_args.kwargs["fields"], ["TA_1"])
        self.assertEqual(delete.call_args.kwargs["data_version"], "v1")

    def test_upload_singlevar_rejects_multiple_sites(self):
        with patch.object(InfluxIO, "delete") as delete:
            with self.assertRaises(ValueError):
                self.dbc.upload_singlevar(var_df=self._var_df(["CH-DAV", "CH-LAE"]), to_bucket="b",
                                          to_measurement="TA", timezone_offset_to_utc_hours=1)
        delete.assert_not_called()


try:
    import influxdb_client  # noqa: F401
    HAS_INFLUXDB_CLIENT = True
except ImportError:
    HAS_INFLUXDB_CLIENT = False

FIELD = "TA_T1_4_2"


def _upload_frame(n=48, start="2026-03-01 00:30", site="ch-hon", all_nan=False):
    idx = pd.date_range(start, periods=n, freq="30min", name="TIMESTAMP_END")
    df = pd.DataFrame({FIELD: np.arange(n, dtype=float)}, index=idx)
    if all_nan:
        df[FIELD] = np.nan
    for tag in TAGS:
        df[tag] = "x"
    df["site"] = site
    df["varname"] = FIELD
    df["data_version"] = "meteoscreening_diive"
    return df


@unittest.skipUnless(HAS_INFLUXDB_CLIENT, "needs the db group (influxdb-client)")
class TestInfluxUploadMockedHttp(unittest.TestCase):
    """upload_singlevar with the real client, only the HTTP calls replaced."""

    def setUp(self):
        from influxdb_client.client.delete_api import DeleteApi
        from influxdb_client.service.write_service import WriteService

        self.dbc = InfluxIO.__new__(InfluxIO)
        self.dbc.conf_db = {"url": "http://127.0.0.1:1", "token": "x", "org": "o"}  # never contacted
        self.dbc.verbose = 2
        self.deletes, self.bodies = [], []
        self.write_error = None

        def fake_delete(api, start, stop, predicate, bucket, org=None):
            self.deletes.append(dict(start=start, stop=stop, predicate=predicate))

        def fake_post_write(service, org, bucket, body, **kw):
            if self.write_error is not None:
                raise self.write_error
            self.bodies.append(body.decode() if isinstance(body, bytes) else body)

        mod = "diive.core.io.db.influx.influxio"
        self.success = MagicMock()
        self.warn = MagicMock()
        patchers = [patch.object(DeleteApi, "delete", fake_delete),
                    patch.object(WriteService, "post_write", fake_post_write),
                    patch(f"{mod}.success", self.success),
                    patch(f"{mod}.warn", self.warn),
                    patch(f"{mod}.info"),
                    patch(f"{mod}.detail")]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def _upload(self, df, **kwargs):
        self.dbc.upload_singlevar(var_df=df, to_bucket="b", to_measurement="TA",
                                  timezone_offset_to_utc_hours=1, **kwargs)

    def _lines(self):
        return [line for body in self.bodies for line in body.split("\n") if line]

    def test_writes_all_points_then_reports_success(self):
        df = _upload_frame()
        df.iloc[[0, 5], 0] = np.nan
        self._upload(df)
        self.assertEqual(len(self.deletes), 1)
        self.assertEqual(len(self._lines()), 46)
        self.success.assert_called_once()
        self.assertIn("46 points", self.success.call_args.args[0])

    def test_large_frame_is_written_in_chunks(self):
        with patch("diive.core.io.db.influx.influxio._WRITE_CHUNK_ROWS", 100):
            self._upload(_upload_frame(n=250))
        self.assertEqual(len(self.bodies), 3)
        self.assertEqual(len(self._lines()), 250)

    def test_write_error_after_delete_raises(self):
        from influxdb_client.rest import ApiException
        self.write_error = ApiException(status=400, reason="field type conflict")
        with self.assertRaises(RuntimeError) as ctx:
            self._upload(_upload_frame())
        msg = str(ctx.exception)
        self.assertIn("field type conflict", msg)
        self.assertIn("2026-03-01 00:30:00 to 2026-03-02 00:00:00 (both included) was deleted", msg)
        self.assertIn("Re-run the upload", msg)
        self.assertIsInstance(ctx.exception.__cause__, ApiException)
        self.assertEqual(len(self.deletes), 1)
        self.success.assert_not_called()

    def test_delete_range_from_start_stop(self):
        self._upload(_upload_frame(), delete_start="2026-03-01 00:00:00",
                     delete_stop="2026-03-03 00:00:00")
        self.assertEqual(self.deletes[0]["start"], "2026-03-01T00:00:00+01:00")
        self.assertEqual(self.deletes[0]["stop"], "2026-03-03T00:00:00+01:00")

    def test_delete_range_defaults_to_frame(self):
        self._upload(_upload_frame())
        self.assertEqual(self.deletes[0]["start"], "2026-03-01T00:30:00+01:00")
        self.assertEqual(self.deletes[0]["stop"], "2026-03-02T00:00:00+01:00")

    def test_stamps_outside_delete_range_raise(self):
        # Last stamp 2026-03-02 00:00 lies after the stop: rejected before the delete.
        with self.assertRaisesRegex(ValueError, "1 timestamps outside the delete range"):
            self._upload(_upload_frame(), delete_start="2026-03-01 00:00:00",
                         delete_stop="2026-03-01 23:59:59")
        with self.assertRaisesRegex(ValueError, "outside the delete range"):
            self._upload(_upload_frame(), delete_start="2026-03-01 01:00:00")
        self.assertEqual(self.deletes, [])
        self.assertEqual(self.bodies, [])

    def test_delete_range_needs_delete_flag(self):
        with self.assertRaises(ValueError):
            self._upload(_upload_frame(), delete_from_db_before_upload=False,
                         delete_stop="2026-03-03 00:00:00")
        self.assertEqual(self.bodies, [])

    def test_joined_site_tag_rejected(self):
        with self.assertRaisesRegex(ValueError, "several sites"):
            self._upload(_upload_frame(site="ch-hon,ch-oth"))
        self.assertEqual(self.deletes, [])

    def test_all_nan_frame_deletes_and_warns(self):
        self._upload(_upload_frame(all_nan=True))
        self.assertEqual(len(self.deletes), 1)
        self.assertEqual(self.bodies, [])
        self.success.assert_not_called()
        self.warn.assert_called_once()
        msg = self.warn.call_args.args[0]
        self.assertIn("nothing was written", msg)
        self.assertIn("was deleted", msg)
        self.assertEqual(self.warn.call_args.kwargs["verbose"], 2)

    def test_empty_frame_raises_before_delete(self):
        with self.assertRaisesRegex(ValueError, "no records"):
            self._upload(_upload_frame(n=0))
        self.assertEqual(self.deletes, [])


@unittest.skipUnless(HAS_INFLUXDB_CLIENT, "needs the db group (influxdb-client)")
class TestInfluxDownloadMergeMockedHttp(unittest.TestCase):
    """Several tables of one variable (different tag sets) at the same timestamps."""

    def setUp(self):
        self.dbc = InfluxIO.__new__(InfluxIO)
        self.dbc.conf_db = {"url": "http://127.0.0.1:1", "token": "x", "org": "o"}  # never contacted
        self.dbc.verbose = 2
        self.warn = MagicMock()
        mod = "diive.core.io.db.influx.influxio"
        for p in [patch(f"{mod}.warn", self.warn), patch(f"{mod}.info"), patch(f"{mod}.detail")]:
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def _table(gain, values):
        t = pd.date_range("2026-02-28 23:10", periods=len(values), freq="10min", tz="UTC")
        df = pd.DataFrame({"_time": t, "_measurement": "TA", FIELD: values})
        for tag in TAGS:
            df[tag] = "x"
        df["site"] = "ch-hon"
        df["varname"] = FIELD
        df["data_version"] = "raw"
        df["gain"] = gain
        return df

    def _download(self, tables):
        from influxdb_client.client.query_api import QueryApi

        def fake_query(api, query, **kw):
            if "schema" in query:
                return pd.DataFrame({"_value": [FIELD]})
            return [t.copy() for t in tables]

        with patch.object(QueryApi, "query_data_frame", fake_query):
            return self.dbc.download(bucket="b", measurements=["TA"], fields=[FIELD],
                                     start="2026-03-01 00:00:01", stop="2026-03-02 00:00:01",
                                     timezone_offset_to_utc_hours=1, data_version="raw")

    def test_winner_does_not_depend_on_table_order(self):
        a = self._table("1", [5.0, 5.0, 5.0, np.nan])
        b = self._table("1.02", [6.0, 6.0, np.nan, 6.0])
        for tables in ([a, b], [b, a]):
            self.warn.reset_mock()
            simple, detailed, _ = self._download(tables)
            # Same start, so gain '1.02' sorts after '1' and wins where both hold a value.
            self.assertEqual(detailed[FIELD][FIELD].tolist(), [6.0, 6.0, 5.0, 6.0])
            self.assertEqual(detailed[FIELD]["gain"].tolist(), ["1.02", "1.02", "1", "1.02"])
            self.assertEqual(simple[FIELD].tolist(), [6.0, 6.0, 5.0, 6.0])
            self.warn.assert_called_once()
            msg = self.warn.call_args.args[0]
            self.assertIn("2 timestamps", msg)
            self.assertIn("gain ['1.02'] kept over ['1']", msg)
            self.assertEqual(self.warn.call_args.kwargs["verbose"], 2)

    def test_table_that_starts_later_wins(self):
        old = self._table("2", [5.0, 5.0, 5.0])
        new = self._table("1", [6.0, 6.0])
        new["_time"] = new["_time"] + pd.Timedelta("10min")  # newer setup, smaller tag value
        for tables in ([old, new], [new, old]):
            _, detailed, _ = self._download(tables)
            self.assertEqual(detailed[FIELD][FIELD].tolist(), [5.0, 6.0, 6.0])

    def test_no_warning_when_tables_agree(self):
        self._download([self._table("1", [5.0, 5.0]), self._table("1.02", [5.0, np.nan])])
        self.warn.assert_not_called()


FIELD2 = "TA_T1_2_1"
_RAW_TAGS = dict(site="ch-hon", units="degC", raw_varname="TA", raw_units="degC", hpos="T1",
                 vpos="4", repl="2", data_raw_freq="10min", freq="10min", filegroup="meteo",
                 config_filetype="TOA5", data_version="raw", gain="1", offset="0")


def _series(values, start_utc="2026-02-28 23:10", field=FIELD, **tags):
    """One stored series (Flux table) of the pivoted download query."""
    t = pd.date_range(start_utc, periods=len(values), freq="10min", tz="UTC")
    df = pd.DataFrame({"_time": t, "_measurement": "TA", field: values})
    for tag, val in {**_RAW_TAGS, "varname": field, **tags}.items():
        if val is not None:
            df[tag] = val
    return df


def _annotated_csv(tables, fields=None):
    """InfluxDB annotated CSV of *tables*, header rows only when the columns change.

    *fields* gives the field columns of every table (empty where a table has no
    value), which puts tables of several fields under one header.
    """
    def _time(ts):
        return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines, prev = [], None
    for n, t in enumerate(tables):
        tags = sorted(c for c in t.columns if c in TAGS)
        tfields = fields or [c for c in t.columns if c not in tags and c not in ("_time", "_measurement")]
        cols = ["result", "table", "_start", "_stop", "_time", "_measurement"] + tags + tfields
        if cols != prev:
            dtypes = (["string", "long"] + ["dateTime:RFC3339"] * 3 + ["string"] * (1 + len(tags))
                      + ["double"] * len(tfields))
            group = ["false", "false", "true", "true", "false"] + ["true"] * (1 + len(tags)) \
                + ["false"] * len(tfields)
            if prev is not None:
                lines.append("")
            lines += ["#group," + ",".join(group), "#datatype," + ",".join(dtypes),
                      "#default,_result" + "," * (len(cols) - 1), "," + ",".join(cols)]
            prev = cols
        for _, r in t.iterrows():
            vals = ["" if f not in t.columns or pd.isna(r[f]) else repr(float(r[f])) for f in tfields]
            lines.append(",".join(["", "", str(n), "2026-02-28T23:00:01Z", "2026-03-31T23:00:01Z",
                                   _time(r["_time"]), r["_measurement"]] + [str(r[c]) for c in tags] + vals))
    return "\n".join(lines) + "\n"


def _lines_to_tables(lines):
    """Parse written line protocol (no escaped characters) back into one table per series."""
    rows = []
    for line in lines:
        series, fieldset, ts = line.split(" ")
        measurement, *tagpairs = series.split(",")
        name, value = fieldset.split("=")
        rows.append(dict(_measurement=measurement, **dict(p.split("=") for p in tagpairs),
                         _time=pd.Timestamp(int(ts), unit="s", tz="UTC"), **{name: float(value)}))
    df = pd.DataFrame(rows)
    tagcols = [c for c in df.columns if c in TAGS] + ["_measurement"]
    return [g.dropna(axis=1, how="all") for _, g in df.groupby(tagcols, sort=True)]


@unittest.skipUnless(HAS_INFLUXDB_CLIENT, "needs the db group (influxdb-client)")
class TestInfluxRealParser(unittest.TestCase):
    """Download through the real client CSV parser and upload through the real serializer.

    Only the HTTP calls are replaced. The real client puts all tables with the same
    columns into one DataFrame (L299).
    """

    def setUp(self):
        import urllib3
        from influxdb_client.client.delete_api import DeleteApi
        from influxdb_client.service.query_service import QueryService
        from influxdb_client.service.write_service import WriteService

        self.dbc = InfluxIO.__new__(InfluxIO)
        self.dbc.conf_db = {"url": "http://127.0.0.1:1", "token": "x", "org": "o"}  # never contacted
        self.dbc.verbose = 2
        self.csv = ""
        self.deletes, self.bodies = [], []

        def fake_post_query(service, **kw):
            text = self.csv
            if "schema" in kw["query"].query:
                text = ("#group,false,false,false\n#datatype,string,long,string\n#default,_result,,\n"
                        f",result,table,_value\n,,0,{FIELD}\n,,0,{FIELD2}\n")
            return urllib3.HTTPResponse(body=io.BytesIO(text.encode()), preload_content=False)

        def fake_delete(api, start, stop, predicate, bucket, org=None):
            self.deletes.append(dict(start=start, stop=stop, predicate=predicate))

        def fake_post_write(service, org, bucket, body, **kw):
            self.bodies.append(body.decode() if isinstance(body, bytes) else body)

        mod = "diive.core.io.db.influx.influxio"
        self.warn = MagicMock()
        patchers = [patch.object(QueryService, "post_query", fake_post_query),
                    patch.object(DeleteApi, "delete", fake_delete),
                    patch.object(WriteService, "post_write", fake_post_write),
                    patch(f"{mod}.warn", self.warn), patch(f"{mod}.info"),
                    patch(f"{mod}.success"), patch(f"{mod}.detail")]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def _download(self, tables=None, fields=None, data_version="raw"):
        if tables is not None:
            self.csv = _annotated_csv(tables, fields=fields)
        return self.dbc.download(bucket="b", measurements=["TA"], fields=[FIELD, FIELD2],
                                 start="2026-03-01 00:00:01", stop="2026-04-01 00:00:01",
                                 timezone_offset_to_utc_hours=1, data_version=data_version)

    def _lines(self):
        return [line for body in self.bodies for line in body.split("\n") if line]

    def test_client_returns_tag_sets_in_one_frame(self):
        # Premise of L299: two tag sets of one field arrive as one DataFrame.
        self.csv = _annotated_csv([_series([5.0] * 4), _series([5.1] * 2, gain="1.02")])
        res = self.dbc._query_df(fluxql.pivotstring())
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(sorted(res["table"].unique()), [0, 1])

    def test_same_schema_tag_sets_merged_by_start(self):
        old = _series([5.0] * 6)
        new = _series([5.1] * 3, start_utc="2026-02-28 23:40", gain="1.02")
        for tables in ([old, new], [new, old]):
            self.warn.reset_mock()
            simple, detailed, _ = self._download(tables)
            d = detailed[FIELD]
            self.assertEqual(d[FIELD].tolist(), [5.0] * 3 + [5.1] * 3)
            self.assertEqual(d["gain"].tolist(), ["1"] * 3 + ["1.02"] * 3)
            self.assertEqual(simple[FIELD].tolist(), [5.0] * 3 + [5.1] * 3)
            self.assertTrue(d.index.is_monotonic_increasing and d.index.is_unique)
            self.warn.assert_called_once()
            msg = self.warn.call_args.args[0]
            self.assertIn("3 timestamps", msg)
            self.assertIn("gain ['1.02'] kept over ['1']", msg)

    def test_nan_in_later_table_keeps_older_row(self):
        old = _series([5.0] * 4)
        new = _series([5.1, np.nan], start_utc="2026-02-28 23:30", gain="1.02")
        _, detailed, _ = self._download([old, new])
        d = detailed[FIELD]
        self.assertEqual(d[FIELD].tolist(), [5.0, 5.0, 5.1, 5.0])
        self.assertEqual(d["gain"].tolist(), ["1", "1", "1.02", "1"])
        self.assertIn("1 timestamps", self.warn.call_args.args[0])

    def test_several_fields_in_one_frame(self):
        tables = [_series([5.0] * 4), _series([1.0, 2.0, 3.0], field=FIELD2),
                  _series([5.1] * 2, start_utc="2026-02-28 23:30", gain="1.02")]
        self.csv = _annotated_csv(tables, fields=[FIELD, FIELD2])
        self.assertIsInstance(self.dbc._query_df(fluxql.pivotstring()), pd.DataFrame)
        simple, detailed, assigned = self._download()
        self.assertEqual(detailed[FIELD][FIELD].tolist(), [5.0, 5.0, 5.1, 5.1])
        self.assertEqual(detailed[FIELD]["gain"].tolist(), ["1", "1", "1.02", "1.02"])
        self.assertEqual(detailed[FIELD2][FIELD2].tolist(), [1.0, 2.0, 3.0])
        self.assertNotIn(FIELD2, detailed[FIELD].columns)
        self.assertEqual(simple[FIELD2].dropna().tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(assigned, {FIELD: "TA", FIELD2: "TA"})

    def test_round_trip_without_gain_and_offset_tags(self):
        # L300: raw data stored without gain and offset can be uploaded. The
        # points carry the raw tags plus gain 1 and offset 0 (not scaled).
        raw = _series([5.0, 5.5, 6.0, 6.5], gain=None, offset=None)
        _, detailed, _ = self._download([raw])
        var_df = detailed[FIELD].assign(data_version="meteoscreening_diive")
        self.assertNotIn("gain", var_df.columns)
        self.dbc.upload_singlevar(var_df=var_df, to_bucket="b", to_measurement="TA",
                                  timezone_offset_to_utc_hours=1,
                                  delete_start="2026-03-01 00:00:01", delete_stop="2026-04-01 00:00:01")
        lines = self._lines()
        self.assertEqual(len(lines), 4)
        tagset = dict(p.split("=") for p in lines[0].split(" ")[0].split(",")[1:])
        expected = {**_RAW_TAGS, "varname": FIELD, "data_version": "meteoscreening_diive",
                    "gain": "1.0", "offset": "0.0"}
        self.assertEqual(tagset, expected)
        self.assertEqual(self.deletes[0]["predicate"],
                         f'_measurement="TA" AND varname="{FIELD}" '
                         f'AND data_version="meteoscreening_diive" AND site="ch-hon"')

        # Same delete as for a frame with every tag.
        self.dbc.upload_singlevar(var_df=var_df.assign(gain="1.0", offset="0.0"), to_bucket="b",
                                  to_measurement="TA", timezone_offset_to_utc_hours=1,
                                  delete_start="2026-03-01 00:00:01", delete_stop="2026-04-01 00:00:01")
        self.assertEqual(self.deletes[1], self.deletes[0])

        # Read the written points back through the parser.
        _, back, _ = self._download(_lines_to_tables(lines), data_version="meteoscreening_diive")
        pd.testing.assert_series_equal(back[FIELD][FIELD], detailed[FIELD][FIELD])
        self.assertEqual(back[FIELD].index.tolist(), detailed[FIELD].index.tolist())

    def test_download_reports_detected_resolution_and_freq_tag(self):
        self.csv = _annotated_csv([_series([5.0] * 6), _series([1.0, 2.0, 3.0], field=FIELD2, freq=None)],
                                  fields=[FIELD, FIELD2])
        with patch("diive.core.io.db.influx.influxio.info") as info:
            self._download()
        lines = [c.args[0] for c in info.call_args_list if c.args[0].startswith("<--")]
        self.assertEqual(len(lines), 2)
        line = next(s for s in lines if s.startswith(f"<-- {FIELD}:"))
        self.assertIn("6 records", line)
        self.assertIn("time resolution 10min (freq tag: 10min)", line)
        line2 = next(s for s in lines if s.startswith(f"<-- {FIELD2}:"))
        self.assertIn("time resolution 10min (no freq tag)", line2)

    def test_upload_still_needs_site_varname_and_data_version(self):
        var_df = _upload_frame().drop(columns=["gain", "offset", "varname"])
        with self.assertRaisesRegex(ValueError, r"required tag columns: \['varname'\]"):
            self.dbc.upload_singlevar(var_df=var_df, to_bucket="b", to_measurement="TA",
                                      timezone_offset_to_utc_hours=1)
        self.assertEqual(self.deletes, [])


class TestInfluxRetry(unittest.TestCase):
    """get_client retries each request, also long after the client was created (L245)."""

    def setUp(self):
        import influxdb_client.client.write.retry as wr

        import diive.core.io.db.influx.connection as conn
        self.client = conn.get_client({"url": "http://127.0.0.1:1", "token": "x", "org": "o"})
        self.addCleanup(self.client.close)
        self.retries = self.client.api_client.rest_client.pool_manager.connection_pool_kw["retries"]
        clock = self.clock = [datetime.now()]

        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock[0]

        patchers = [patch.object(wr, "datetime", FakeDatetime),
                    patch.object(conn, "datetime", FakeDatetime),
                    patch.object(wr, "logger")]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def _advance(self, seconds):
        self.clock[0] += timedelta(seconds=seconds)

    def test_late_errors_are_retried(self):
        import urllib3
        from urllib3.exceptions import ReadTimeoutError
        self._advance(86400)  # one day: far beyond any client-wide deadline
        new = self.retries.increment(method="POST", url="/api/v2/write",
                                     response=urllib3.HTTPResponse(status=503, body=b""))
        self.assertEqual(new.total, 4)
        new = self.retries.increment(method="POST", url="/api/v2/write",
                                     error=ReadTimeoutError(None, "/api/v2/write", "read timed out"))
        self.assertEqual(new.total, 4)

    def test_five_read_timeouts_in_a_row_are_retried(self):
        from urllib3.exceptions import MaxRetryError, ReadTimeoutError

        from diive.core.io.db.influx.connection import RETRY_MAX_DELAY_S, TIMEOUT_S
        err = ReadTimeoutError(None, "/api/v2/write", "read timed out")
        self._advance(86400)
        retry = self.retries
        for _ in range(5):
            retry = retry.increment(method="POST", url="/api/v2/write", error=err)
            self._advance(TIMEOUT_S + RETRY_MAX_DELAY_S)
        self.assertEqual(retry.total, 0)
        with self.assertRaises(MaxRetryError):
            retry.increment(method="POST", url="/api/v2/write", error=err)

    def test_shared_retry_object_is_not_changed(self):
        from urllib3.exceptions import ReadTimeoutError
        before = self.retries.retry_timeout
        self._advance(86400)
        self.retries.increment(method="POST", url="/api/v2/write",
                               error=ReadTimeoutError(None, "/x", "read timed out"))
        self.assertEqual(self.retries.retry_timeout, before)
        self.assertEqual(self.retries.history, ())

    def test_write_through_client_is_retried_after_503(self):
        import urllib3
        from influxdb_client.client.write_api import SYNCHRONOUS
        from urllib3.connectionpool import HTTPConnectionPool
        self._advance(86400)
        responses = [urllib3.HTTPResponse(status=503, body=io.BytesIO(b""), preload_content=False),
                     urllib3.HTTPResponse(status=204, body=io.BytesIO(b""), preload_content=False)]
        with patch.object(HTTPConnectionPool, "_make_request", side_effect=responses) as req, \
                patch("urllib3.util.retry.time.sleep") as sleep:
            self.client.write_api(write_options=SYNCHRONOUS).write(bucket="b", record="m f=1")
        self.assertEqual(req.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
