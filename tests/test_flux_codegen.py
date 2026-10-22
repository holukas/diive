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
        from diive.flux.postprocessingchain import chain_to_code
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
        from diive.flux.postprocessingchain import chain_to_code
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
        from diive.flux.postprocessingchain import level2_to_code
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
        from diive.flux.postprocessingchain import level31_to_code
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

    def test_level32_to_code(self):
        from diive.flux.postprocessingchain import level32_to_code
        code = level32_to_code(
            init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
            level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
            level31_kwargs={"gapfill_storage_term": True},
            level32_steps=[
                {"method": "flag_outliers_hampel_test",
                 "kwargs": {"n_sigma": 5.5, "n_sigma_daytime": 4.0}},
                {"method": "flag_outliers_zscore_test", "kwargs": {"thres_zscore": 4}},
                {"method": "flag_missingvals_test", "kwargs": {}},
            ])
        compile(code, "<gen>", "exec")
        self.assertIn("make_level32_detector, run_level32", code)
        # Each committed test is followed by addflag(); the chain ends with run_level32.
        self.assertEqual(code.count("sod.addflag()"), 3)
        self.assertIn("data = run_level32(data, outlier_detector=sod)", code)
        # Non-default kwarg shown; default-valued ones (n_sigma=5.5, thres_zscore=4) omitted.
        self.assertIn("n_sigma_daytime=4.0", code)
        self.assertNotIn("n_sigma=5.5", code)
        # A method with no non-default kwargs renders as a bare call.
        self.assertIn("sod.flag_missingvals_test()", code)

    def test_level33_to_code(self):
        from diive.flux.postprocessingchain import level33_to_code
        code = level33_to_code(
            init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
            level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
            level31_kwargs={},
            level32_steps=[{"method": "flag_outliers_hampel_test", "kwargs": {}}],
            level33_kwargs={"thresholds": [0.18], "threshold_labels": ["CUT_50"]})
        compile(code, "<gen>", "exec")
        # L3.3 always renders the L3.2 chain before it (USTAR needs screened data).
        self.assertIn("make_level32_detector", code)
        # Percentile labels -> the one L3.3 entry point with given thresholds
        self.assertIn("run_level33_ustar_detection(", code)
        self.assertIn("thresholds={50: 0.18}", code)
        self.assertNotIn("threshold_labels", code)

    def test_level33_to_code_free_labels_stay_constant(self):
        from diive.flux.postprocessingchain import level33_to_code
        kw = dict(init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
                  level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
                  level31_kwargs={},
                  level32_steps=[{"method": "flag_outliers_hampel_test", "kwargs": {}}])
        for labels in (["CUT_NONE"], ["CUT_5"], ["low"]):
            with self.subTest(labels=labels):
                code = level33_to_code(**kw, level33_kwargs={"thresholds": [0.1],
                                                             "threshold_labels": labels})
                compile(code, "<gen>", "exec")
                self.assertIn("run_level33_constant_ustar(", code)
                self.assertIn(f"threshold_labels={labels!r}", code)

    def test_level33_to_code_detection_with_reference(self):
        from diive.flux.postprocessingchain import level41_to_code
        code = level41_to_code(
            init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
            level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
            level31_kwargs={},
            level32_steps=[{"method": "flag_outliers_hampel_test", "kwargs": {}}],
            level33_kwargs={"_detection": True, "ta_col": "TA", "swin_col": "SW_IN",
                            "mode": "vut", "reference": True},
            level41_cfg={"methods": ["mds"], "ustar_reference": True,
                         "mds": {"swin": "SW_IN", "ta": "TA", "vpd": "VPD", "mode": "oneflux"}})
        compile(code, "<gen>", "exec")
        self.assertIn("reference=True,", code)
        self.assertIn("mode='vut',", code)
        self.assertIn("run_level41_ustar_reference", code.split("init_flux_data(")[0])
        self.assertLess(code.index("run_level41_mds("), code.index("data = run_level41_ustar_reference(data)"))

    def test_level41_to_code(self):
        from diive.flux.postprocessingchain import level41_to_code
        code = level41_to_code(
            init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
            level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
            level31_kwargs={},
            level32_steps=[{"method": "flag_outliers_hampel_test", "kwargs": {}}],
            level33_kwargs={"thresholds": [0.18], "threshold_labels": ["CUT_50"]},
            level41_cfg={
                "methods": ["rf", "xgb", "mds"],
                "features": ["TA_1_1_1", "SW_IN_1_1_1", "VPD_1_1_1"],
                "mds": {"swin": "SW_IN_1_1_1", "ta": "TA_1_1_1", "vpd": "VPD_1_1_1"},
            })
        compile(code, "<gen>", "exec")
        # L4.1 always renders the full L2 -> L3.3 chain before it.
        self.assertIn("make_level32_detector", code)
        self.assertIn("run_level33_ustar_detection(", code)
        # One engineer shared by rf + xgb; mds takes driver columns.
        self.assertEqual(code.count("make_level41_engineer("), 1)
        self.assertIn("run_level41_rf(", code)
        self.assertIn("run_level41_xgb(", code)
        self.assertIn("run_level41_mds(", code)
        self.assertIn("swin='SW_IN_1_1_1'", code)
        # Only the requested method's functions are imported.
        self.assertIn("make_level41_engineer", code)

    def test_level41_to_code_mds_driver_qc(self):
        from diive.flux.postprocessingchain import level41_to_code
        kw = dict(init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
                  level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
                  level31_kwargs={},
                  level32_steps=[{"method": "flag_outliers_hampel_test", "kwargs": {}}],
                  level33_kwargs={"thresholds": [0.18], "threshold_labels": ["CUT_50"]})
        code = level41_to_code(**kw, level41_cfg={
            "methods": ["mds"],
            "mds": {"swin": "SW_IN_F", "ta": "TA_F", "vpd": "VPD_F",
                    "swin_qc": "SW_IN_F_QC", "ta_qc": "TA_F_QC", "vpd_qc": None}})
        compile(code, "<gen>", "exec")
        self.assertIn("swin_qc='SW_IN_F_QC'", code)
        self.assertIn("ta_qc='TA_F_QC'", code)
        self.assertNotIn("vpd_qc", code)            # None -> omitted

    def test_chain_to_code_mds_driver_qc(self):
        from diive.flux.postprocessingchain import chain_to_code
        code = chain_to_code(
            init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
            config_kwargs=dict(fluxcol="FC", ustar_thresholds=[0.18], ustar_labels=["CUT_50"],
                               mds_swin="SW_IN_F", mds_ta="TA_F", mds_vpd="VPD_F",
                               mds_swin_qc="SW_IN_F_QC", mds_ta_qc=None))
        compile(code, "<gen>", "exec")
        self.assertIn("mds_swin_qc='SW_IN_F_QC'", code)
        self.assertNotIn("mds_ta_qc", code)         # default None -> omitted

    def test_level41_to_code_mds_only_omits_engineer(self):
        from diive.flux.postprocessingchain import level41_to_code
        code = level41_to_code(
            init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
            level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
            level31_kwargs={},
            level32_steps=[{"method": "flag_outliers_hampel_test", "kwargs": {}}],
            level33_kwargs={"thresholds": [0.18], "threshold_labels": ["CUT_50"]},
            level41_cfg={"methods": ["mds"],
                         "mds": {"swin": "SW_IN", "ta": "TA", "vpd": "VPD"}})
        compile(code, "<gen>", "exec")
        # MDS needs no FeatureEngineer; rf/xgb absent.
        self.assertNotIn("make_level41_engineer", code)
        self.assertNotIn("run_level41_rf(", code)
        self.assertNotIn("run_level41_xgb(", code)
        self.assertIn("run_level41_mds(", code)

    def test_level42_to_code_writes_none_when_default_differs(self):
        """gapfill_method=None (daytime ONEFlux without nee_f) must reach the script (L350).

        Dropping it made the script run the default 'mds'. Values equal to the
        function default and unset drivers stay out of the script.
        """
        from diive.flux.postprocessingchain import level42_to_code
        kw = dict(init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
                  level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
                  level31_kwargs={},
                  level32_steps=[{"method": "flag_outliers_hampel_test", "kwargs": {}}],
                  level33_kwargs={"thresholds": [0.18], "threshold_labels": ["CUT_50"]},
                  level41_cfg={"methods": ["mds"],
                               "mds": {"swin": "SW_IN", "ta": "TA", "vpd": "VPD"}})
        drivers = {"ta": "TA", "sw_in": "SW_IN", "ta_f": "TA_F", "sw_in_f": "SW_IN_F",
                   "vpd_f": "VPD_F"}

        # Through run_level42_partitioning and through the per-variant function
        for cfg in ({"method": "daytime", "mode": "oneflux"}, {"variants": ["dt_of"]}):
            with self.subTest(cfg=cfg):
                code = level42_to_code(**kw, level42_cfg={
                    **cfg, **drivers, "gapfill_method": None, "nee_sd": None})
                compile(code, "<gen>", "exec")
                l42 = code[code.index("run_level42_"):]
                self.assertIn("gapfill_method=None,", l42)
                self.assertNotIn("nee_sd", l42)

        # Default values are still left out
        code = level42_to_code(**kw, level42_cfg={
            "method": "daytime", "mode": "oneflux", **drivers,
            "gapfill_method": "mds", "vpd_in_kpa": True})
        l42 = code[code.index("run_level42_partitioning("):]
        self.assertNotIn("gapfill_method", l42)
        self.assertNotIn("vpd_in_kpa", l42)
        self.assertIn("ta='TA',", l42)

        # A driver without a default is not written as None
        code = level42_to_code(**kw, level42_cfg={
            "variants": ["nt_of"], "ta": "TA", "sw_in": None, "ta_f": "TA_F",
            "gapfill_method": "rf"})
        l42 = code[code.index("data = run_level42_nighttime_oneflux("):]
        self.assertNotIn("sw_in", l42)
        self.assertIn("gapfill_method='rf',", l42)


if __name__ == "__main__":
    unittest.main()
