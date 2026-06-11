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

    def test_is_project_false_for_plain_folder(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(is_project(d))

    def test_load_non_project_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_project(d)


if __name__ == "__main__":
    unittest.main()
