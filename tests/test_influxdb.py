import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from diive.core.io.db.influx import fluxql
from diive.core.io.db.influx.common import convert_ts_to_timezone
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


if __name__ == "__main__":
    unittest.main()
