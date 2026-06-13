import unittest


class TestFluxProcessingChainComposable(unittest.TestCase):
    """Exercise the standalone level callables (composable API) directly."""

    def test_partial_pipeline_l2_l31_l32(self):
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.fluxprocessingchain import (
            FluxLevelData, LevelResults,
            init_flux_data, run_level2, run_level31,
            make_level32_detector, run_level32,
        )

        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        # Drop reserved columns init_flux_data computes itself; otherwise the
        # reserved-name guard rejects the input.
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                              if c in df.columns])
        df['TA_1_1_1'] = df['TA_1_1_1'].bfill()
        df['SW_IN_1_1_1'] = df['SW_IN_1_1_1'].bfill()
        df['VPD_EP'] = df['VPD_EP'].bfill()

        # --- init ---
        data = init_flux_data(
            df=df, fluxcol='FC',
            site_lat=46.583056, site_lon=9.790639, utc_offset=1,
            nighttime_threshold=20,
            daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2,
        )
        self.assertIsInstance(data, FluxLevelData)
        self.assertIsInstance(data.levels, LevelResults)
        self.assertIsNone(data.filteredseries)
        self.assertEqual(data.level_ids, [])
        self.assertEqual(data.meta.fluxcol, 'FC')
        self.assertEqual(data.meta.outname, 'NEE')

        # --- Level-2 ---
        data2 = run_level2(
            data,
            ssitc={'apply': True, 'setflag_timeperiod': None},
            gas_completeness={'apply': True},
            spectral_correction_factor={'apply': True},
            signal_strength={
                'apply': True,
                'signal_strength_col': 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
                'method': 'discard below', 'threshold': 60,
            },
            raw_data_screening_vm97={
                'apply': True,
                'spikes': True, 'amplitude': False, 'dropout': True,
                'abslim': False, 'skewkurt_hf': False, 'skewkurt_sf': False,
                'discont_hf': False, 'discont_sf': False,
            },
            angle_of_attack={
                'apply': True,
                'application_dates': [['2023-07-01', '2023-09-01']],
            },
        )
        # Pure function contract: input data unchanged
        self.assertIsNone(data.filteredseries)
        self.assertEqual(data.level_ids, [])
        # Output has L2 results
        self.assertEqual(data2.level_ids, ['L2'])
        self.assertEqual(data2.filteredseries.name, 'FC_L2_QCF')
        self.assertEqual(data2.filteredseries.dropna().count(), 778)
        self.assertIsNotNone(data2.levels.level2)
        self.assertIsNotNone(data2.levels.level2_qcf)
        self.assertIsNotNone(data2.levels.filteredseries_hq)

        # --- Level-3.1 ---
        data31 = run_level31(data2, gapfill_storage_term=True)
        self.assertEqual(data31.level_ids, ['L2', 'L3.1'])
        self.assertEqual(data31.filteredseries.name, 'NEE_L3.1_QCF')
        self.assertEqual(data31.filteredseries.dropna().count(), 778)
        self.assertEqual(data31.levels.flux_corrected_col, 'NEE_L3.1')
        # data2 untouched
        self.assertEqual(data2.level_ids, ['L2'])
        self.assertIsNone(data2.levels.flux_corrected_col)

        # --- Level-3.2 with factory ---
        data31, sod = make_level32_detector(data31)
        sod.flag_outliers_abslim_test(
            separate_daytime_nighttime=True,
            daytime_minmax=[-50, 50], nighttime_minmax=[-50, 50],
            showplot=False, verbose=False,
        )
        sod.addflag()
        data32 = run_level32(data31, outlier_detector=sod)
        self.assertEqual(data32.level_ids, ['L2', 'L3.1', 'L3.2'])
        self.assertIsNotNone(data32.levels.level32)
        self.assertIsNotNone(data32.levels.level32_qcf)
        # data31 untouched
        self.assertEqual(data31.level_ids, ['L2', 'L3.1'])
        self.assertIsNone(data31.levels.level32)

    def test_level2_test_inputs_and_vm97_subtests(self):
        from diive.flux.fluxprocessingchain import VM97_SUBTESTS, level2_test_inputs

        # Eight VM97 sub-tests, each (key, label, kind in {'hard','soft'}).
        self.assertEqual(len(VM97_SUBTESTS), 8)
        keys = [k for k, _, _ in VM97_SUBTESTS]
        self.assertIn("spikes", keys)
        self.assertIn("discont_sf", keys)
        self.assertTrue(all(kind in ("hard", "soft") for _, _, kind in VM97_SUBTESTS))

        # Input columns are templated on the flux column + its base variable.
        info = level2_test_inputs("FC", "CO2")
        self.assertEqual(info["ssitc"]["inputs"], ["FC_SSITC_TEST"])
        self.assertEqual(info["raw_data_screening_vm97"]["inputs"], ["CO2_VM97_TEST"])
        self.assertIn("CO2_NR", info["gas_completeness"]["inputs"])
        # Signal strength reads a user-chosen column (no fixed input).
        self.assertTrue(info["signal_strength"]["user_col"])
        self.assertEqual(info["signal_strength"]["inputs"], [])
        # A different flux re-templates the columns.
        self.assertEqual(level2_test_inputs("LE", "H2O")["spectral_correction_factor"]["inputs"],
                         ["LE_SCF"])

    def test_level31_storage_col(self):
        from diive.flux.fluxprocessingchain import level31_storage_col
        self.assertEqual(level31_storage_col("FC"), "SC_SINGLE")
        self.assertEqual(level31_storage_col("LE"), "SLE_SINGLE")
        self.assertEqual(level31_storage_col("H"), "SH_SINGLE")
        self.assertIsNone(level31_storage_col("NOT_A_FLUX"))

    def test_level2_custom_input_columns(self):
        # Each L2 test can read a differently-named column via a 'col' override
        # (two keys for the two-column completeness test).
        from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
        from diive.flux.fluxprocessingchain import init_flux_data, run_level2

        df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]
        df = df.drop(columns=[c for c in ("SW_IN_POT", "DAYTIME", "NIGHTTIME")
                              if c in df.columns])
        # Rename the standard inputs to non-standard names.
        df = df.rename(columns={"FC_SSITC_TEST": "MY_SSITC",
                                "CO2_VM97_TEST": "MY_VM97",
                                "CO2_NR": "MY_CO2_NR"})
        data = init_flux_data(df=df, fluxcol="FC", site_lat=47.4, site_lon=8.5, utc_offset=1)
        data = run_level2(
            data,
            ssitc={"apply": True, "setflag_timeperiod": None, "col": "MY_SSITC"},
            gas_completeness={"apply": True, "basevar_nr_col": "MY_CO2_NR"},
            raw_data_screening_vm97={
                "apply": True, "spikes": True, "dropout": True, "amplitude": False,
                "abslim": False, "skewkurt_hf": False, "skewkurt_sf": False,
                "discont_hf": False, "discont_sf": False, "col": "MY_VM97"},
        )
        # The chain ran on the renamed columns and produced the standard flags.
        assert data.filteredseries.dropna().count() > 0
        assert any("SSITC" in str(c) for c in data.fpc_df.columns)
        assert any("VM97_DROPOUT" in str(c) for c in data.fpc_df.columns)

    def test_ordering_errors(self):
        """Level callables should fail loudly when called out of order."""
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.fluxprocessingchain import (
            init_flux_data, run_level31, make_level32_detector, run_level33_constant_ustar,
            run_level41_mds,
        )

        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                              if c in df.columns])
        data = init_flux_data(
            df=df, fluxcol='FC',
            site_lat=46.583056, site_lon=9.790639, utc_offset=1,
        )

        # run_level31 without run_level2
        with self.assertRaises(RuntimeError):
            run_level31(data, gapfill_storage_term=True)

        # make_level32_detector without any filtered series
        with self.assertRaises(RuntimeError):
            make_level32_detector(data)

        # run_level33 without run_level31
        with self.assertRaises(RuntimeError):
            run_level33_constant_ustar(
                data, thresholds=[0.05], threshold_labels=['CUT_50'], showplot=False)

        # run_level41_mds without run_level33
        with self.assertRaises(RuntimeError):
            run_level41_mds(data, swin='SW_IN_1_1_1', ta='TA_1_1_1', vpd='VPD_EP')

    def test_run_chain_single_call_driver(self):
        """Smoke-test the headline single-call FLUXNET driver (run_chain + FluxConfig)."""
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.fluxprocessingchain import (
            FluxConfig, FluxLevelData, init_flux_data, run_chain,
        )
        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        df['TA_1_1_1'] = df['TA_1_1_1'].bfill()
        df['SW_IN_1_1_1'] = df['SW_IN_1_1_1'].bfill()
        df['VPD_kPa'] = df['VPD_EP'].bfill().multiply(0.1)  # hPa -> kPa for MDS

        data = init_flux_data(df=df, fluxcol='FC',
                              site_lat=46.583056, site_lon=9.790639, utc_offset=1)

        cfg = FluxConfig(
            fluxcol='FC',
            ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
            outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,
            level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
            gapfill_rf=False, gapfill_xgb=False, gapfill_mds=True,   # MDS only (no ML training)
            mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa',
        )
        out = run_chain(data, cfg)

        self.assertIsInstance(out, FluxLevelData)
        # All levels ran in order.
        for lvl in ('L2', 'L3.1', 'L3.2', 'L3.3'):
            self.assertIn(lvl, out.level_ids)
        # L3.3 QCF column carries the chained-idstr provenance.
        self.assertTrue(any('L3.3' in str(c) and str(c).endswith('_QCF')
                            for c in out.fpc_df.columns))
        # MDS gap-filled column is produced for the USTAR scenario.
        gf = out.gapfilled_cols()
        self.assertIn('mds', gf)
        self.assertIn('CUT_50', gf['mds'])
        self.assertIn(gf['mds']['CUT_50'], out.fpc_df.columns)


if __name__ == '__main__':
    unittest.main()
