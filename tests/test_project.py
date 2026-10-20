"""Unit tests for diive project folders (diive.core.io.project)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from diive.core.io.project import (
    DiiveProject,
    is_project,
    load_project,
    project_name_to_dirname,
    save_project,
)
from diive.core.metadata import MetadataStore


def _sample():
    idx = pd.date_range("2023-01-01", periods=48, freq="30min", name="TIMESTAMP_MIDDLE")
    df = pd.DataFrame({"NEE": np.arange(48.0), "TA": np.arange(48.0)}, index=idx)
    store = MetadataStore()
    store.record_original(df.columns, operation="Imported", timestamp="2026-06-11 14:00")
    store.add_user_tag("NEE", "favorite")
    store.set_description("NEE", "net ecosystem exchange")
    store.record_derived("NEE", parent="NEE", operation="Hampel",
                         params={"n_sigma": 5.5}, tags=["hampel"])
    return df, store


class TestProject(unittest.TestCase):

    def test_dirname_suffix(self):
        self.assertEqual(project_name_to_dirname("Proj"), "Proj.diive")
        self.assertEqual(project_name_to_dirname("Proj.diive"), "Proj.diive")

    def test_save_and_load_roundtrip(self):
        df, store = _sample()
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / project_name_to_dirname("MyProj")
            save_project(folder, DiiveProject(
                name="MyProj", data=df, metadata=store,
                extras={"site": {"latitude": 46.8}, "range": None}))

            self.assertTrue(is_project(folder))
            self.assertTrue((folder / "__diive__").is_file())

            proj = load_project(folder)
            self.assertEqual(proj.name, "MyProj")
            pd.testing.assert_frame_equal(proj.data, df)
            md = proj.metadata.get("NEE")
            self.assertEqual(md.origin, "modified")
            self.assertEqual(sorted(md.tags), ["favorite", "hampel"])
            self.assertEqual(md.description, "net ecosystem exchange")
            self.assertEqual(len(md.provenance), 2)
            self.assertEqual(proj.extras["site"], {"latitude": 46.8})

    def test_roundtrip_mixed_resolutions_is_identical(self):
        """L314: reopening a project must not regularize the saved data.

        30-min data with 10-min data added keep all records, the timestamp
        convention and no frequency.
        """
        i30 = pd.date_range("2023-01-01 00:30", periods=48 * 7, freq="30min")
        i10 = pd.date_range("2023-01-01 00:10", periods=144 * 7, freq="10min")
        a = pd.DataFrame({"TA": np.arange(len(i30), dtype=float)}, index=i30)
        b = pd.DataFrame({"TA_10MIN": np.arange(len(i10), dtype=float)}, index=i10)
        df = a.combine_first(b)
        df.index.name = "TIMESTAMP_END"
        self.assertIsNone(df.index.freq)
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / project_name_to_dirname("Mixed")
            save_project(folder, DiiveProject(name="Mixed", data=df))
            proj = load_project(folder)
        pd.testing.assert_frame_equal(proj.data, df, check_freq=True)
        self.assertIsNone(proj.data.index.freq)
        self.assertEqual(int(proj.data["TA_10MIN"].notna().sum()), 1008)

    def test_roundtrip_restores_freq_of_older_project(self):
        """Projects saved without data_freq get the frequency of a regular index."""
        import json
        df, store = _sample()
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / project_name_to_dirname("Old")
            save_project(folder, DiiveProject(name="Old", data=df, metadata=store))
            manifest_path = folder / "project.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["data_freq"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            proj = load_project(folder)
        pd.testing.assert_frame_equal(proj.data, df, check_freq=True)

    def test_is_project_false_for_plain_folder(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(is_project(d))

    def test_load_non_project_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_project(d)


if __name__ == "__main__":
    unittest.main()
