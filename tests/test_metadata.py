"""Unit tests for the per-variable metadata model (diive.core.metadata)."""
from __future__ import annotations

import unittest

from diive.core.metadata import (
    ATTRS_KEY,
    DERIVED,
    FAVORITE,
    MODIFIED,
    ORIGINAL,
    MetadataStore,
    provenance_attr,
)


class TestMetadataStore(unittest.TestCase):

    def test_record_original_baseline(self):
        store = MetadataStore()
        store.record_original(["NEE", "TA"])
        md = store.get("NEE")
        self.assertEqual(md.origin, ORIGINAL)
        self.assertIn(ORIGINAL, md.tags)
        self.assertEqual(md.provenance, [])

    def test_record_original_with_import_history(self):
        store = MetadataStore()
        store.record_original(
            ["NEE"], operation="Imported from example", timestamp="2026-06-11 09:00")
        md = store.get("NEE")
        self.assertEqual(md.origin, ORIGINAL)
        self.assertEqual(len(md.provenance), 1)
        self.assertEqual(md.provenance[0].operation, "Imported from example")
        self.assertEqual(md.provenance[0].timestamp, "2026-06-11 09:00")

    def test_get_autocreates_original(self):
        store = MetadataStore()
        md = store.get("brand_new")
        self.assertEqual(md.origin, ORIGINAL)
        self.assertIn(ORIGINAL, md.tags)

    def test_peek_does_not_create(self):
        store = MetadataStore()
        self.assertIsNone(store.peek("absent"))
        self.assertNotIn("absent", store)

    def test_record_derived_sets_origin_parent_provenance(self):
        store = MetadataStore()
        store.record_original(["NEE"])
        store.record_derived(
            "NEE_HAMPEL", parent="NEE", operation="Hampel",
            params={"n_sigma": 5.5}, tags=["hampel"], origin=MODIFIED,
            timestamp="2026-06-11 14:02")
        md = store.get("NEE_HAMPEL")
        self.assertEqual(md.origin, MODIFIED)
        self.assertEqual(md.parents, ["NEE"])
        self.assertIn("hampel", md.tags)
        self.assertNotIn(ORIGINAL, md.tags)  # no longer original
        self.assertEqual(len(md.provenance), 1)
        self.assertEqual(md.provenance[0].operation, "Hampel")
        self.assertEqual(md.provenance[0].timestamp, "2026-06-11 14:02")
        self.assertIn("n_sigma=5.5", md.provenance[0].describe())

    def test_user_tags_roundtrip(self):
        store = MetadataStore()
        store.record_original(["NEE"])
        store.add_user_tag("NEE", FAVORITE)
        store.record_derived("NEE", operation="auto", tags=["autotag"])
        # Only the user tag round-trips, not the function-set one.
        saved = store.user_tags()
        self.assertEqual(saved, {"NEE": [FAVORITE]})

        restored = MetadataStore()
        restored.record_original(["NEE"])
        restored.load_user_tags(saved)
        md = restored.get("NEE")
        self.assertTrue(md.is_user_tag(FAVORITE))
        self.assertIn(FAVORITE, md.tags)

    def test_user_source_is_sticky(self):
        store = MetadataStore()
        store.add_user_tag("X", "keep")
        store.get("X").add_tag("keep", source="function")  # must not downgrade
        self.assertTrue(store.get("X").is_user_tag("keep"))

    def test_remove_user_tag(self):
        store = MetadataStore()
        store.add_user_tag("X", FAVORITE)
        store.remove_user_tag("X", FAVORITE)
        self.assertNotIn(FAVORITE, store.get("X").tags)

    def test_set_description_truncates_to_50_words(self):
        store = MetadataStore()
        long_text = " ".join(f"w{i}" for i in range(80))
        stored = store.set_description("NEE", long_text)
        assert len(stored.split()) == 50
        assert len(store.get("NEE").description.split()) == 50

    def test_user_data_roundtrip_tags_and_descriptions(self):
        store = MetadataStore()
        store.record_original(["NEE"])
        store.add_user_tag("NEE", FAVORITE)
        store.set_description("NEE", "the net ecosystem exchange flux")
        data = store.user_data()
        assert data["tags"] == {"NEE": [FAVORITE]}
        assert data["descriptions"] == {"NEE": "the net ecosystem exchange flux"}

        restored = MetadataStore()
        restored.record_original(["NEE"])
        restored.load_user_data(data)
        assert restored.get("NEE").description == "the net ecosystem exchange flux"
        assert FAVORITE in restored.get("NEE").tags

    def test_load_user_data_accepts_legacy_flat_tags(self):
        store = MetadataStore()
        store.load_user_data({"NEE": [FAVORITE]})  # old flat shape
        assert FAVORITE in store.get("NEE").tags

    def test_drop(self):
        store = MetadataStore()
        store.record_original(["X"])
        store.drop("X")
        self.assertNotIn("X", store)

    def test_from_attrs(self):
        store = MetadataStore()
        store.record_original(["NEE"])
        attrs = {
            "NEE_HAMPEL": provenance_attr(
                origin=MODIFIED, parent="NEE", operation="Hampel",
                params={"n_sigma": 5.5}, tags=["hampel"]),
            "FLAG_NEE_OUTLIER_HAMPEL_TEST": provenance_attr(
                origin=DERIVED, parent="NEE", operation="Hampel flag",
                tags=["flag", "hampel"]),
        }
        store.from_attrs(attrs, timestamp="2026-06-11 14:02")
        self.assertEqual(store.get("NEE_HAMPEL").origin, MODIFIED)
        flag = store.get("FLAG_NEE_OUTLIER_HAMPEL_TEST")
        self.assertEqual(flag.origin, DERIVED)
        self.assertIn("flag", flag.tags)
        self.assertEqual(flag.parents, ["NEE"])

    def test_provenance_attr_shape(self):
        entry = provenance_attr(origin=DERIVED, parent="NEE", operation="op")
        self.assertEqual(
            set(entry), {"origin", "parent", "operation", "params", "tags"})
        self.assertEqual(entry["params"], {})
        self.assertEqual(entry["tags"], [])

    def test_attrs_key_is_stable(self):
        self.assertEqual(ATTRS_KEY, "diive_metadata")


if __name__ == "__main__":
    unittest.main()
