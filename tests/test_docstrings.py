"""
TEST_DOCSTRINGS: per-parameter help extraction
==============================================

Pure tests for `diive.core.utils.docstrings` — Google-style Args parsing and
class attribute-docstring extraction (used for GUI tooltips).

Run: pytest tests/test_docstrings.py -v
"""
import unittest


class TestDocstrings(unittest.TestCase):

    def test_parse_google_args(self):
        from diive.core.utils.docstrings import parse_google_args
        doc = (
            "Summary line.\n\n"
            "Args:\n"
            "    alpha: First param.\n"
            "    beta (int): Second param, with a\n"
            "        continuation line.\n\n"
            "Returns:\n"
            "    nothing\n"
        )
        out = parse_google_args(doc)
        self.assertEqual(out["alpha"], "First param.")
        self.assertEqual(out["beta"], "Second param, with a continuation line.")
        self.assertNotIn("Returns", out)

    def test_attribute_docstrings(self):
        from diive.core.utils.docstrings import attribute_docstrings
        from diive.flux.fluxprocessingchain import FluxConfig
        docs = attribute_docstrings(FluxConfig)
        self.assertIn("ustar_thresholds", docs)
        self.assertIn("USTAR threshold", docs["ustar_thresholds"])

    def test_param_docs_callable(self):
        from diive.core.utils.docstrings import param_docs
        from diive.flux.fluxprocessingchain import init_flux_data
        docs = param_docs(init_flux_data)
        self.assertIn("site_lat", docs)
        self.assertIn("latitude", docs["site_lat"].lower())

    def test_param_docs_class(self):
        from diive.core.utils.docstrings import param_docs
        from diive.flux.fluxprocessingchain import FluxConfig
        docs = param_docs(FluxConfig)
        self.assertGreater(len(docs), 10)
        self.assertIn("gapfill_mds", docs)


if __name__ == "__main__":
    unittest.main()
