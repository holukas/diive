"""
TEST_FLUX_CODEGEN: reproducible-script generation for the flux chain
====================================================================

Pure (no GUI, no data) tests that the chain code-serializers emit runnable,
default-omitting scripts.

Run: pytest tests/test_flux_codegen.py -v
"""
import unittest


class TestFluxCodegen(unittest.TestCase):

    def test_chain_to_code(self):
        from diive.flux.fluxprocessingchain import chain_to_code
        code = chain_to_code(
            init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
            config_kwargs=dict(
                fluxcol="FC", ustar_thresholds=[0.18], ustar_labels=["CUT_50"],
                outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,
                gapfill_rf=True,       # equals default -> must be omitted
                gapfill_xgb=True,      # differs from default -> must appear
            ),
            load_hint="dv.load_parquet('data.parquet')")
        compile(code, "<gen>", "exec")                 # syntactically valid
        self.assertIn("run_chain(data, cfg)", code)
        self.assertIn("FluxConfig(", code)
        self.assertIn("ustar_thresholds=[0.18]", code)
        self.assertIn("df = dv.load_parquet('data.parquet')", code)
        self.assertIn("drop(columns=", code)           # reserved-cols drop
        self.assertNotIn("gapfill_rf", code)           # default omitted
        self.assertIn("gapfill_xgb=True", code)        # non-default kept

    def test_chain_to_code_omits_init_defaults(self):
        from diive.flux.fluxprocessingchain import chain_to_code
        code = chain_to_code(
            init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1,
                             nighttime_threshold=20,   # equals default -> omitted
                             daytime_accept_qcf_below=2),  # differs -> kept
            config_kwargs=dict(fluxcol="FC"))
        self.assertNotIn("nighttime_threshold", code)
        self.assertIn("daytime_accept_qcf_below=2", code)
        # Required init args are always present.
        self.assertIn("site_lat=46.6", code)

    def test_level2_to_code(self):
        from diive.flux.fluxprocessingchain import level2_to_code
        code = level2_to_code(
            init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
            level2_settings={
                "ssitc": {"apply": True, "setflag_timeperiod": None},
                "gas_completeness": {"apply": True},
            })
        compile(code, "<gen>", "exec")
        self.assertIn("init_flux_data, run_level2", code)
        self.assertIn("run_level2(", code)
        self.assertIn("ssitc={'apply': True", code)

    def test_level31_to_code(self):
        from diive.flux.fluxprocessingchain import level31_to_code
        code = level31_to_code(
            init_kwargs=dict(fluxcol="LE", site_lat=46.6, site_lon=9.8, utc_offset=1),
            level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
            level31_kwargs={"gapfill_storage_term": True, "set_storage_to_zero": True})
        compile(code, "<gen>", "exec")
        self.assertIn("init_flux_data, run_level2, run_level31", code)
        self.assertIn("run_level31(", code)
        # set_storage_to_zero is non-default -> shown; gapfill_storage_term is default -> omitted.
        self.assertIn("set_storage_to_zero=True", code)
        self.assertNotIn("gapfill_storage_term", code)


if __name__ == "__main__":
    unittest.main()
