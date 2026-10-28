import unittest


class TestFluxPostProcessingChainComposable(unittest.TestCase):
    """Exercise the standalone level callables (composable API) directly."""

    def test_partial_pipeline_l2_l31_l32(self):
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import (
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
            separate_day_night=True,
            minval_daytime=-50, maxval_daytime=50, minval_nighttime=-50, maxval_nighttime=50,
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
        from diive.flux.postprocessingchain import VM97_SUBTESTS, level2_test_inputs

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
        from diive.flux.postprocessingchain import level31_storage_col
        self.assertEqual(level31_storage_col("FC"), "SC_SINGLE")
        self.assertEqual(level31_storage_col("LE"), "SLE_SINGLE")
        self.assertEqual(level31_storage_col("H"), "SH_SINGLE")
        self.assertIsNone(level31_storage_col("NOT_A_FLUX"))

    def test_level31_set_storage_to_zero_without_storage_column(self):
        """set_storage_to_zero=True must not require a storage column.

        This is the documented H / LE path: those fluxes have no storage
        profile, so the storage column is missing from the input data.
        """
        import pandas as pd
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import init_flux_data, run_level2, run_level31

        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                              if c in df.columns])
        df = df.drop(columns=['SLE_SINGLE'])  # no storage profile for LE
        data = init_flux_data(df=df, fluxcol='LE',
                              site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})

        data31 = run_level31(data, set_storage_to_zero=True)

        self.assertEqual(data31.level_ids, ['L2', 'L3.1'])
        self.assertEqual(data31.levels.flux_corrected_col, 'LE_L3.1')
        self.assertIsNone(data31.levels.level31.strgcol)
        self.assertNotIn('SLE_SINGLE', data31.fpc_df.columns)
        # Storage term zero -> corrected flux is the measured flux
        pd.testing.assert_series_equal(data31.fpc_df['LE_L3.1'], df['LE'], check_names=False)
        # The report must not need the storage column either
        data31.levels.level31.report()

    def test_level2_custom_input_columns(self):
        # Each L2 test can read a differently-named column via a 'col' override
        # (two keys for the two-column completeness test).
        from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
        from diive.flux.postprocessingchain import init_flux_data, run_level2

        df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]
        df = df.drop(columns=[c for c in ("SW_IN_POT", "DAYTIME", "NIGHTTIME")
                              if c in df.columns])
        # Rename the standard inputs to non-standard names.
        df = df.rename(columns={"FC_SSITC_TEST": "MY_SSITC",
                                "CO2_VM97_TEST": "MY_VM97",
                                "CO2_NR": "MY_CO2_NR"})
        data = init_flux_data(df=df, fluxcol="FC", site_lat=47.478333, site_lon=8.364389, utc_offset=1)
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
        from diive.flux.postprocessingchain import (
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
        from diive.flux.postprocessingchain import (
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


    def test_run_chain_mds_driver_qc(self):
        """FluxConfig mds_*_qc reach FluxMDS; a missing QC column fails up front."""
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import FluxConfig, init_flux_data, run_chain
        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        for c in ('TA_1_1_1', 'SW_IN_1_1_1'):
            df[f'{c}_QC'] = df[c].isna().astype(float)   # 1 = gap-filled below
            df[c] = df[c].bfill()
        df['VPD_kPa'] = df['VPD_EP'].bfill().multiply(0.1)
        data = init_flux_data(df=df, fluxcol='FC',
                              site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        kw = dict(fluxcol='FC', ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
                  outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,
                  level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
                  gapfill_rf=False, gapfill_xgb=False, gapfill_mds=True,
                  mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa')

        with self.assertRaises(KeyError):
            run_chain(data, FluxConfig(**kw, mds_vpd_qc='NOT_THERE'))

        out = run_chain(data, FluxConfig(**kw, mds_swin_qc='SW_IN_1_1_1_QC',
                                         mds_ta_qc='TA_1_1_1_QC'))
        mds = out.levels.level41_mds['CUT_50']
        self.assertEqual((mds.swin_qc, mds.ta_qc, mds.vpd_qc),
                         ('SW_IN_1_1_1_QC', 'TA_1_1_1_QC', None))
        self.assertEqual(set(mds._driver_qc), {'swin', 'ta'})

    def test_level42_partitioning_run_chain(self):
        """Wire the four NEE partitioning variants through run_chain (L4.2)."""
        import warnings
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import (
            FluxConfig, FluxLevelData, init_flux_data, run_chain,
        )
        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        # Gap-filled meteo drivers the partitioning needs (the chain doesn't
        # produce them itself). bfill is a test stand-in for real gap-filling.
        df['TA_F'] = df['TA_1_1_1'].bfill()
        df['SW_IN_F'] = df['SW_IN_1_1_1'].bfill()
        df['VPD_F'] = df['VPD_EP'].bfill().multiply(0.1)  # hPa -> kPa

        data = init_flux_data(df=df, fluxcol='FC',
                              site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        cfg = FluxConfig(
            fluxcol='FC', ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
            level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
            gapfill_rf=False, gapfill_xgb=False, gapfill_mds=True,   # MDS feeds nighttime
            mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_F',
            partition_nighttime_oneflux=True, partition_nighttime_reddyproc=True,
            partition_daytime_reddyproc=True, partition_daytime_oneflux=True,
            partition_ta='TA_1_1_1', partition_sw_in='SW_IN_1_1_1',
            partition_ta_f='TA_F', partition_sw_in_f='SW_IN_F', partition_vpd_f='VPD_F',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # short test record -> unconstrained fits
            out = run_chain(data, cfg)

        self.assertIsInstance(out, FluxLevelData)
        self.assertIn('L4.2', out.level_ids)
        # All four variants produced their scenario-suffixed columns.
        pc = out.partitioned_cols()
        for variant in ('nt_of', 'nt_rp', 'dt_rp', 'dt_of'):
            self.assertIn(variant, pc)
            self.assertIn('CUT_50', pc[variant])
            for col in pc[variant]['CUT_50']:
                self.assertIn(col, out.fpc_df.columns)
        # The RECO/GPP columns carry the variant token and the USTAR suffix.
        self.assertIn('RECO_NT_OF_CUT_50', out.fpc_df.columns)
        self.assertIn('GPP_DT_RP_CUT_50', out.fpc_df.columns)
        # Instances are stored per variant, keyed by USTAR scenario.
        self.assertIn('CUT_50', out.levels.level42_nt_of)
        self.assertEqual(set(out.levels.level42_variants()),
                         {'nt_of', 'nt_rp', 'dt_rp', 'dt_of'})

        # Re-running an upstream cascade-aware level (L3.3) must clear L4.2.
        from diive.flux.postprocessingchain import run_level33_constant_ustar
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out2 = run_level33_constant_ustar(
                out, thresholds=[0.2], threshold_labels=['CUT_50'], showplot=False, verbose=False)
        self.assertNotIn('L4.2', out2.level_ids)
        self.assertFalse(out2.levels.has_level42())
        self.assertNotIn('RECO_NT_OF_CUT_50', out2.fpc_df.columns)

    def test_level42_nighttime_requires_gapfilled_nee(self):
        """A nighttime partitioning variant must have its gap-fill method run first."""
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import (
            init_flux_data, run_level2, run_level31, make_level32_detector,
            run_level32, run_level33_constant_ustar, run_level42_nighttime_oneflux,
        )
        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        df['TA_F'] = df['TA_1_1_1'].bfill()
        data = init_flux_data(df=df, fluxcol='FC',
                              site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
        data = run_level31(data, gapfill_storage_term=True)
        data, sod = make_level32_detector(data)
        sod.flag_outliers_hampel_test(showplot=False, verbose=False)
        sod.addflag()
        data = run_level32(data, outlier_detector=sod)
        data = run_level33_constant_ustar(
            data, thresholds=[0.1], threshold_labels=['CUT_50'], showplot=False, verbose=False)
        # No L4.1 gap-filling has run -> nighttime partitioning cannot find nee_f.
        with self.assertRaises(RuntimeError):
            run_level42_nighttime_oneflux(
                data, ta='TA_1_1_1', sw_in='SW_IN_1_1_1', ta_f='TA_F')


class TestFluxPostProcessingChainLevel2(unittest.TestCase):
    """Level 2 in isolation: quality-flag expansion + QCF on EddyPro output.

    Mirrors ``examples/flux/postprocessingchain/postprocessingchain_level2.py``,
    which runs L2 standalone on a real EddyPro FLUXNET output file.
    """

    def _init_data(self, **init_kwargs):
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import init_flux_data

        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                              if c in df.columns])
        kwargs = dict(site_lat=46.583056, site_lon=9.790639, utc_offset=1,
                      nighttime_threshold=20,
                      daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)
        kwargs.update(init_kwargs)
        return init_flux_data(df=df, fluxcol='FC', **kwargs)

    def test_level2_produces_qcf_and_flags(self):
        """L2 emits the QCF-filtered flux, the HQ series, and per-test flags."""
        from diive.flux.postprocessingchain import run_level2

        data = self._init_data()
        out = run_level2(
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
        )

        # Pure-function contract: the input container is untouched.
        self.assertEqual(data.level_ids, [])
        self.assertIsNone(data.filteredseries)

        # Output has L2 results.
        self.assertEqual(out.level_ids, ['L2'])
        self.assertEqual(out.filteredseries.name, 'FC_L2_QCF')
        self.assertIsNotNone(out.levels.level2)
        self.assertIsNotNone(out.levels.level2_qcf)

        # The HQ (QCF=0) series is a strict subset of the accepted (QCF<2) flux.
        n_accepted = out.filteredseries.dropna().count()
        n_hq = out.levels.filteredseries_hq.dropna().count()
        self.assertGreater(n_accepted, 0)
        self.assertLessEqual(n_hq, n_accepted)

        # The enabled tests each produced a flag column under the L2 idstr.
        cols = [str(c) for c in out.fpc_df.columns]
        self.assertTrue(any('SSITC' in c for c in cols))
        self.assertTrue(any('VM97_SPIKE' in c for c in cols))
        self.assertTrue(any('VM97_DROPOUT' in c for c in cols))
        self.assertTrue(any(c.endswith('_QCF') for c in cols))

    def test_level2_skips_unset_tests(self):
        """A test whose config is omitted produces no flag column for it."""
        from diive.flux.postprocessingchain import run_level2

        out = run_level2(self._init_data(),
                         ssitc={'apply': True, 'setflag_timeperiod': None})
        cols = [str(c) for c in out.fpc_df.columns]
        self.assertTrue(any('SSITC' in c for c in cols))
        # VM97 raw-data screening was not requested.
        self.assertFalse(any('VM97_SPIKE' in c for c in cols))

    def test_level2_signal_strength_requires_keys(self):
        """Enabling signal_strength without its keys raises a clear KeyError."""
        from diive.flux.postprocessingchain import run_level2

        with self.assertRaises(KeyError):
            run_level2(self._init_data(), signal_strength={'apply': True})

    def test_level2_vm97_requires_all_eight_subkeys(self):
        """Enabling VM97 with a missing sub-key raises a clear KeyError."""
        from diive.flux.postprocessingchain import run_level2

        with self.assertRaises(KeyError):
            run_level2(self._init_data(),
                       raw_data_screening_vm97={'apply': True, 'spikes': True})

    def test_level2_accept_threshold_changes_retained_count(self):
        """A stricter daytime accept threshold cannot retain more records."""
        from diive.flux.postprocessingchain import run_level2

        settings = dict(
            ssitc={'apply': True, 'setflag_timeperiod': None},
            gas_completeness={'apply': True},
            spectral_correction_factor={'apply': True},
        )
        lenient = run_level2(self._init_data(daytime_accept_qcf_below=2,
                                             nighttime_accept_qcf_below=2), **settings)
        strict = run_level2(self._init_data(daytime_accept_qcf_below=1,
                                            nighttime_accept_qcf_below=1), **settings)
        self.assertLessEqual(strict.filteredseries.dropna().count(),
                             lenient.filteredseries.dropna().count())


class TestRerunCascade(unittest.TestCase):
    """Re-running a level must invalidate that level and every later one.

    Documented behaviour with no test: without the cascade, a second
    `run_level2` would concat duplicate column labels into `fpc_df`, leaving
    ambiguous lookups and stale flags for `FlagQCF` to consume. Its coverage
    came only from the GUI driving levels repeatedly.
    """

    LEVEL2_SETTINGS = dict(ssitc={'apply': True, 'setflag_timeperiod': None},
                           gas_completeness={'apply': True})

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import (
            init_flux_data, run_level2, run_level31, make_level32_detector,
            run_level32, run_level33_constant_ustar)

        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                              if c in df.columns])
        for col in ('TA_1_1_1', 'SW_IN_1_1_1', 'VPD_EP'):
            df[col] = df[col].bfill()
        cls.data = init_flux_data(
            df=df, fluxcol='FC', site_lat=46.583056, site_lon=9.790639,
            utc_offset=1, nighttime_threshold=20,
            daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)

        # Build the full cascade-aware chain once; levels are pure, so every
        # test can re-run from these snapshots.
        cls.d2 = run_level2(cls.data, **cls.LEVEL2_SETTINGS)
        cls.d31 = run_level31(cls.d2, gapfill_storage_term=True)
        d31b, sod = make_level32_detector(cls.d31)
        sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False,
                                      verbose=False)
        sod.addflag()
        cls.d32 = run_level32(d31b, outlier_detector=sod)
        cls.d33 = run_level33_constant_ustar(cls.d32, thresholds=[0.1],
                                             threshold_labels=['CUT_50'])

    # --- integration: re-running a real level ---

    def test_rerun_level2_drops_its_own_columns_and_cascades(self):
        from diive.flux.postprocessingchain import run_level2
        again = run_level2(self.d33, **self.LEVEL2_SETTINGS)

        # Back to a container that has only been through L2.
        self.assertEqual(again.level_ids, ['L2'])
        self.assertEqual(list(again.added_columns), ['L2'])
        # The previous L2 columns were dropped before the new ones landed, so
        # the frame matches a fresh first L2 run instead of carrying doubles.
        self.assertEqual(len(again.fpc_df.columns), len(self.d2.fpc_df.columns))
        self.assertEqual(int(again.fpc_df.columns.duplicated().sum()), 0)
        # Downstream state is gone. Scalar fields reset to None; the per-scenario
        # dicts (L3.3 keys its QCF by USTAR scenario) reset to {}, not None.
        for field in ('level31', 'level31_qcf', 'level32', 'level32_qcf',
                      'level33'):
            with self.subTest(field=field):
                self.assertIsNone(getattr(again.levels, field))
        self.assertEqual(again.levels.level33_qcf, {})

    def test_rerun_level31_keeps_level2_but_clears_below(self):
        from diive.flux.postprocessingchain import run_level31
        again = run_level31(self.d33, gapfill_storage_term=True)
        self.assertEqual(again.level_ids, ['L2', 'L3.1'])
        self.assertIsNotNone(again.levels.level2)          # upstream survives
        self.assertIsNone(again.levels.level32)
        self.assertIsNone(again.levels.level33)
        self.assertEqual(set(again.added_columns), {'L2', 'L3.1'})

    def test_rerun_does_not_mutate_the_input_container(self):
        from diive.flux.postprocessingchain import run_level2
        before_ids = list(self.d33.level_ids)
        before_cols = len(self.d33.fpc_df.columns)
        run_level2(self.d33, **self.LEVEL2_SETTINGS)
        self.assertEqual(self.d33.level_ids, before_ids)
        self.assertEqual(len(self.d33.fpc_df.columns), before_cols)

    # --- unit: cascade_reset ---

    def test_cascade_reset_keeps_upstream_levels(self):
        from diive.flux.postprocessingchain.levels._rerun import cascade_reset
        reset = cascade_reset(self.d33, 'L3.2')
        self.assertEqual(reset.level_ids, ['L2', 'L3.1'])
        self.assertIsNotNone(reset.levels.level2)
        self.assertIsNotNone(reset.levels.level31)
        self.assertIsNone(reset.levels.level32)
        self.assertIsNone(reset.levels.level33)
        self.assertEqual(set(reset.added_columns), {'L2', 'L3.1'})

    def test_cascade_reset_restores_the_surviving_filteredseries(self):
        # filteredseries always belongs to the most recently completed level, so
        # after a cascade it must fall back to the newest survivor -- not linger
        # on the invalidated level's series.
        from diive.flux.postprocessingchain.levels._rerun import cascade_reset
        self.assertEqual(cascade_reset(self.d33, 'L3.2').filteredseries.name,
                         'NEE_L3.1_QCF')
        self.assertEqual(cascade_reset(self.d33, 'L3.3').filteredseries.name,
                         'NEE_L3.1_L3.2_QCF')
        # Cascading from the first level leaves nothing to fall back to.
        self.assertIsNone(cascade_reset(self.d33, 'L2').filteredseries)

    def test_cascade_reset_rejects_an_unknown_level(self):
        from diive.flux.postprocessingchain.levels._rerun import cascade_reset
        with self.assertRaises(ValueError) as ctx:
            cascade_reset(self.d33, 'L4.1')
        self.assertIn('L4.1', str(ctx.exception))

    def test_cascade_clears_the_additive_levels(self):
        # L4.1 / L4.2 do not cascade among themselves, but a cascade from any
        # earlier level must clear them: their output was computed against
        # upstream inputs that just became stale.
        import dataclasses
        from diive.flux.postprocessingchain.levels._rerun import cascade_reset
        seeded = dataclasses.replace(
            self.d33,
            levels=dataclasses.replace(self.d33.levels,
                                       level41_mds={'CUT_50': 'stale'},
                                       level42_nt_of={'CUT_50': 'stale'}),
            level_ids=list(self.d33.level_ids) + ['L4.1', 'L4.2'],
            added_columns={**self.d33.added_columns,
                           'L4.1_mds': [], 'L4.2_nt_of': []})
        reset = cascade_reset(seeded, 'L3.1')
        self.assertEqual(reset.levels.level41_mds, {})
        self.assertEqual(reset.levels.level42_nt_of, {})
        self.assertEqual(reset.level_ids, ['L2'])
        self.assertNotIn('L4.1_mds', reset.added_columns)
        self.assertNotIn('L4.2_nt_of', reset.added_columns)

    # --- unit: the additive (per-method) helpers ---

    def test_drop_columns_for_key_touches_only_that_key(self):
        # This is what keeps L4.1 additive: re-running MDS must not disturb the
        # random-forest or XGBoost output sitting in the same frame.
        import dataclasses
        from diive.flux.postprocessingchain.levels._rerun import drop_columns_for_key
        mds_col, rf_col = list(self.d33.fpc_df.columns)[:2]
        seeded = dataclasses.replace(
            self.d33,
            added_columns={**self.d33.added_columns,
                           'L4.1_mds': [mds_col], 'L4.1_rf': [rf_col]})
        dropped = drop_columns_for_key(seeded, 'L4.1_mds')
        self.assertNotIn(mds_col, dropped.fpc_df.columns)
        self.assertIn(rf_col, dropped.fpc_df.columns)      # the other method survives
        self.assertNotIn('L4.1_mds', dropped.added_columns)
        self.assertIn('L4.1_rf', dropped.added_columns)

    def test_drop_columns_for_key_is_a_noop_for_an_unknown_key(self):
        from diive.flux.postprocessingchain.levels._rerun import drop_columns_for_key
        self.assertIs(drop_columns_for_key(self.d33, 'L4.1_never_ran'), self.d33)

    def test_record_added_columns_attributes_new_columns(self):
        from diive.flux.postprocessingchain.levels._rerun import record_added_columns
        pre = list(self.d2.fpc_df.columns)
        recorded = record_added_columns(self.d31, 'L3.1', pre)
        expected = [c for c in self.d31.fpc_df.columns if c not in set(pre)]
        self.assertEqual(recorded['L3.1'], expected)
        self.assertTrue(expected, "L3.1 should add at least one column")
        # Existing entries are carried through untouched.
        self.assertEqual(recorded['L2'], list(self.d31.added_columns['L2']))


class TestLevel42DaytimeOneFluxSharedRanges(unittest.TestCase):
    """L4.2 daytime ONEFlux shares the days removed after a failed fit between
    USTAR scenarios, as ONEFlux shares its error file between u* percentiles."""

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import (
            init_flux_data, run_level2, run_level31, make_level32_detector,
            run_level32, run_level33_constant_ustar)

        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                              if c in df.columns])
        df['TA_F'] = df['TA_1_1_1'].bfill()
        df['SW_IN_F'] = df['SW_IN_1_1_1'].bfill()
        df['VPD_F'] = df['VPD_EP'].bfill().multiply(0.1)  # hPa -> kPa
        data = init_flux_data(
            df=df, fluxcol='FC', site_lat=46.583056, site_lon=9.790639,
            utc_offset=1, nighttime_threshold=20,
            daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)
        data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
        data = run_level31(data, gapfill_storage_term=True)
        data, sod = make_level32_detector(data)
        sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False,
                                      verbose=False)
        sod.addflag()
        data = run_level32(data, outlier_detector=sod)
        cls.d33 = run_level33_constant_ustar(
            data, thresholds=[0.1, 0.3], threshold_labels=['CUT_16', 'CUT_84'],
            showplot=False, verbose=False)

    def test_broken_fit_in_one_scenario_removes_the_range_in_all(self):
        # The fit breaks only in the second scenario. The first one must be
        # partitioned again with that range removed, and both must report it.
        from unittest import mock
        import numpy as np
        from diive.flux.partitioning import daytime_oneflux as mod
        from diive.flux.postprocessingchain import run_level42_daytime_oneflux

        nee = self.d33.levels.filteredseries_level33_qcf
        n_measured = {s: int(nee[s].notna().sum()) for s in ('CUT_16', 'CUT_84')}
        self.assertNotEqual(n_measured['CUT_16'], n_measured['CUT_84'])
        rng = (190, 202)
        cols = ['RECO_DT_OF', 'GPP_DT_OF', 'SE_GPP_DT_OF', 'ALPHA_DT_OF',
                'BETA_DT_OF', 'K_DT_OF', 'RREF_DT_OF', 'E0_DT_OF']
        calls = []

        def fake(**kw):
            calls.append((kw['nee'], kw['julday']))
            if len(calls) > 10:  # a range that removes nothing would loop forever
                raise AssertionError("partitioning did not settle")
            if int((kw['nee'] > mod.NAN_TEST).sum()) == n_measured['CUT_84']:
                e = mod._BrokenWindow('HLRC_LloydVPD')
                e.day_range = rng
                raise e
            return {c: np.full(kw['nee'].size, np.nan) for c in cols}

        with mock.patch.object(mod, '_partition_one_year', side_effect=fake):
            out = run_level42_daytime_oneflux(
                self.d33, ta='TA_1_1_1', sw_in='SW_IN_1_1_1', ta_f='TA_F',
                sw_in_f='SW_IN_F', vpd_f='VPD_F', verbose=0)

        parts = out.levels.level42_dt_of
        self.assertEqual(set(parts), {'CUT_16', 'CUT_84'})
        for scen in parts:
            self.assertEqual(parts[scen].removed_days, {2021: [rng]})
        # CUT_16, CUT_84 (breaks), CUT_84 again with the range removed, then
        # CUT_16 again in the final sweep.
        self.assertEqual(len(calls), 4)
        first, julday = calls[0]
        rerun = calls[3][0]
        self.assertFalse(np.array_equal(first, rerun))
        np.testing.assert_array_equal(rerun, mod._remove_days(first, julday, [rng]))
        for scen in parts:
            self.assertIn(f'GPP_DT_OF_{scen}', out.fpc_df.columns)


class TestLevel42Partitioning(unittest.TestCase):
    """run_level42_partitioning selects a variant with method/mode and equals
    the matching per-variant function. FluxConfig does the same with
    partition_method / partition_mode."""

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        import warnings
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import (
            init_flux_data, run_level2, run_level31, make_level32_detector,
            run_level32, run_level33_constant_ustar, run_level41_mds)

        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                              if c in df.columns])
        df['TA_F'] = df['TA_1_1_1'].bfill()
        df['SW_IN_F'] = df['SW_IN_1_1_1'].bfill()
        df['VPD_F'] = df['VPD_EP'].bfill().multiply(0.1)  # hPa -> kPa
        cls.data0 = init_flux_data(df=df, fluxcol='FC', site_lat=46.583056,
                                   site_lon=9.790639, utc_offset=1)
        data = run_level2(cls.data0, ssitc={'apply': True, 'setflag_timeperiod': None})
        data = run_level31(data, gapfill_storage_term=True)
        data, sod = make_level32_detector(data)
        sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False,
                                      verbose=False)
        sod.addflag()
        data = run_level32(data, outlier_detector=sod)
        data = run_level33_constant_ustar(
            data, thresholds=[0.1], threshold_labels=['CUT_50'],
            showplot=False, verbose=False)
        cls.d33 = data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.d41 = run_level41_mds(data, swin='SW_IN_1_1_1', ta='TA_1_1_1',
                                      vpd='VPD_F')
        cls.drivers = dict(ta='TA_1_1_1', sw_in='SW_IN_1_1_1', ta_f='TA_F',
                           sw_in_f='SW_IN_F', vpd_f='VPD_F')

    def test_equals_per_variant_function(self):
        import warnings
        import pandas as pd
        from diive.flux.postprocessingchain import (
            run_level42_partitioning, run_level42_nighttime_oneflux,
            run_level42_nighttime_reddyproc, run_level42_daytime_oneflux,
            run_level42_daytime_reddyproc)
        d = self.drivers
        cases = {
            ('nighttime', 'oneflux'): ('nt_of', lambda data: run_level42_nighttime_oneflux(
                data, ta=d['ta'], sw_in=d['sw_in'], ta_f=d['ta_f'], verbose=0)),
            ('nighttime', 'reddyproc'): ('nt_rp', lambda data: run_level42_nighttime_reddyproc(
                data, ta=d['ta'], sw_in=d['sw_in'], ta_f=d['ta_f'], verbose=0)),
            ('daytime', 'oneflux'): ('dt_of', lambda data: run_level42_daytime_oneflux(
                data, ta=d['ta'], sw_in=d['sw_in'], ta_f=d['ta_f'],
                sw_in_f=d['sw_in_f'], vpd_f=d['vpd_f'], verbose=0)),
            ('daytime', 'reddyproc'): ('dt_rp', lambda data: run_level42_daytime_reddyproc(
                data, ta_f=d['ta_f'], vpd_f=d['vpd_f'], sw_in_f=d['sw_in_f'], verbose=0)),
        }
        for (method, mode), (variant, direct_fn) in cases.items():
            with self.subTest(method=method, mode=mode), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # All drivers are passed: those a variant does not use are ignored.
                out = run_level42_partitioning(self.d41, method=method, mode=mode,
                                               verbose=0, **d)
                ref = direct_fn(self.d41)
                self.assertEqual(set(out.levels.level42_variants()), {variant})
                pd.testing.assert_frame_equal(out.fpc_df, ref.fpc_df, check_exact=True)
                self.assertEqual(out.partitioned_cols(), ref.partitioned_cols())

    def _daytime_oneflux_nee_f_first(self, data, **kw):
        """Run L4.2 daytime ONEFlux with the year's partitioning stubbed out and
        return the gap-filled NEE of the first record that reached it."""
        from unittest import mock
        import numpy as np
        from diive.flux.partitioning import daytime_oneflux as mod
        from diive.flux.postprocessingchain import run_level42_daytime_oneflux
        cols = ['RECO_DT_OF', 'GPP_DT_OF', 'SE_GPP_DT_OF', 'ALPHA_DT_OF',
                'BETA_DT_OF', 'K_DT_OF', 'RREF_DT_OF', 'E0_DT_OF']
        seen = []

        def fake(**k):
            seen.append(k['nee_f_first'])
            return {c: np.full(k['nee'].size, np.nan) for c in cols}

        with mock.patch.object(mod, '_partition_one_year', side_effect=fake):
            run_level42_daytime_oneflux(data, verbose=0, **self.drivers, **kw)
        return seen[0]

    def test_daytime_oneflux_passes_the_l41_gapfilled_nee(self):
        # ONEFlux reads the gap-filled NEE of the year's first record for
        # windows without measured nighttime NEE (L215). L4.2 passes the L4.1
        # gap-filled NEE of the scenario on, and nothing when L4.1 did not run
        # or gapfill_method=None.
        import numpy as np
        col = self.d41.gapfilled_cols()['mds']['CUT_50']
        gf = self.d41.fpc_df[col]
        expected = np.float32(gf[gf.index.year == gf.index.year[0]].iloc[0])
        self.assertEqual(self._daytime_oneflux_nee_f_first(self.d41), expected)
        self.assertIsNone(self._daytime_oneflux_nee_f_first(self.d41, gapfill_method=None))
        self.assertIsNone(self._daytime_oneflux_nee_f_first(self.d33))
        with self.assertRaisesRegex(ValueError, 'gapfill_method'):
            self._daytime_oneflux_nee_f_first(self.d41, gapfill_method='rfx')

    def test_default_mode_is_oneflux(self):
        import warnings
        from diive.flux.postprocessingchain import run_level42_partitioning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = run_level42_partitioning(self.d41, method='daytime', verbose=0,
                                           **self.drivers)
        self.assertEqual(set(out.levels.level42_variants()), {'dt_of'})

    def test_missing_driver_and_invalid_values(self):
        from diive.flux.postprocessingchain import run_level42_partitioning
        with self.assertRaisesRegex(ValueError, r'Missing: vpd_f, sw_in_f\.'):
            run_level42_partitioning(self.d41, method='daytime', mode='reddyproc',
                                     ta_f='TA_F')
        with self.assertRaisesRegex(ValueError, 'mode'):
            run_level42_partitioning(self.d41, method='daytime', mode='diive',
                                     **self.drivers)
        with self.assertRaisesRegex(ValueError, 'method'):
            run_level42_partitioning(self.d41, method='night', **self.drivers)

    def _config(self, **kw):
        from diive.flux.postprocessingchain import FluxConfig
        return FluxConfig(
            fluxcol='FC', ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
            level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
            gapfill_rf=False, gapfill_xgb=False, gapfill_mds=True,
            mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_F', **kw)

    def test_run_chain_daytime_oneflux_gets_the_l41_gapfilled_nee(self):
        # run_chain passes partition_gapfill_method on to daytime ONEFlux, so
        # its MDS gap-filled NEE reaches the partitioning (L215).
        import warnings
        from unittest import mock
        import numpy as np
        from diive.flux.partitioning import daytime_oneflux as mod
        from diive.flux.postprocessingchain import run_chain
        cols = ['RECO_DT_OF', 'GPP_DT_OF', 'SE_GPP_DT_OF', 'ALPHA_DT_OF',
                'BETA_DT_OF', 'K_DT_OF', 'RREF_DT_OF', 'E0_DT_OF']
        seen = []

        def fake(**k):
            seen.append(k['nee_f_first'])
            return {c: np.full(k['nee'].size, np.nan) for c in cols}

        cfg = self._config(partition_method='daytime', partition_ta='TA_1_1_1',
                           partition_sw_in='SW_IN_1_1_1', partition_ta_f='TA_F',
                           partition_sw_in_f='SW_IN_F', partition_vpd_f='VPD_F')
        with warnings.catch_warnings(), \
                mock.patch.object(mod, '_partition_one_year', side_effect=fake):
            warnings.simplefilter('ignore')
            out = run_chain(self.data0, cfg)
        col = out.gapfilled_cols()['mds']['CUT_50']
        gf = out.fpc_df[col]
        self.assertEqual(seen[0], np.float32(gf[gf.index.year == gf.index.year[0]].iloc[0]))

    def test_run_chain_partition_method_mode(self):
        import warnings
        from diive.flux.postprocessingchain import run_chain
        # The flag selects the same variant as method/mode, so it runs once.
        cfg = self._config(partition_method='daytime', partition_mode='reddyproc',
                           partition_daytime_reddyproc=True,
                           partition_ta_f='TA_F', partition_sw_in_f='SW_IN_F',
                           partition_vpd_f='VPD_F')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = run_chain(self.data0, cfg)
        self.assertEqual(set(out.levels.level42_variants()), {'dt_rp'})
        self.assertIn('GPP_DT_RP_CUT_50', out.fpc_df.columns)
        self.assertEqual(out.level_ids.count('L4.2'), 1)

    def test_run_chain_partition_validation(self):
        from diive.flux.postprocessingchain import run_chain
        with self.assertRaisesRegex(ValueError, 'partition_mode'):
            run_chain(self.data0, self._config(partition_method='daytime',
                                               partition_mode='diive'))
        with self.assertRaisesRegex(ValueError, 'partition_method'):
            run_chain(self.data0, self._config(partition_method='night'))
        # A missing driver is reported with the method/mode that needs it.
        with self.assertRaisesRegex(ValueError,
                                    "partition_sw_in_f .partition_method='daytime'"):
            run_chain(self.data0, self._config(partition_method='daytime',
                                               partition_ta='TA_1_1_1',
                                               partition_sw_in='SW_IN_1_1_1',
                                               partition_ta_f='TA_F',
                                               partition_vpd_f='VPD_F'))


class TestChainReports(unittest.TestCase):
    """Reporting over a finished chain (diive/flux/postprocessingchain/reports.py)."""

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import (
            FluxConfig, init_flux_data, run_chain,
        )
        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        df['TA_1_1_1'] = df['TA_1_1_1'].bfill()
        df['SW_IN_1_1_1'] = df['SW_IN_1_1_1'].bfill()
        df['VPD_kPa'] = df['VPD_EP'].bfill().multiply(0.1)
        data = init_flux_data(df=df, fluxcol='FC',
                              site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        cfg = FluxConfig(
            fluxcol='FC',
            ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
            outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,
            level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
            # MDS only: it needs no model training, so the whole class stays fast.
            gapfill_rf=False, gapfill_xgb=False, gapfill_mds=True,
            mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa',
        )
        cls.data = run_chain(data, cfg)

    @staticmethod
    def _capture(fn, *args, **kwargs) -> str:
        """Run fn and return what it printed to the shared console."""
        from diive.core.utils.console import console
        with console.capture() as cap:
            fn(*args, **kwargs)
        return cap.get()

    def test_nongapfilled_cols_mirrors_gapfilled_cols(self):
        gf = self.data.gapfilled_cols()
        ngf = self.data.nongapfilled_cols()
        self.assertEqual(set(gf.keys()), set(ngf.keys()))
        for method in gf:
            self.assertEqual(set(gf[method].keys()), set(ngf[method].keys()))
        target = ngf['mds']['CUT_50']
        self.assertIn(target, self.data.fpc_df.columns)
        # The target is the measured series, not the gap-filled one.
        self.assertNotEqual(target, gf['mds']['CUT_50'])

    def test_merged_df_adds_columns_without_overwriting_input(self):
        merged = self.data.merged_df(verbose=0)
        full = self.data.full_df
        self.assertEqual(len(merged), len(full))
        # Every input column survives unchanged.
        for c in full.columns:
            self.assertIn(c, merged.columns)
        pd_testing_equal = merged[full.columns.tolist()].equals(full)
        self.assertTrue(pd_testing_equal)
        # The gap-filled column came along.
        self.assertIn(self.data.gapfilled_cols()['mds']['CUT_50'], merged.columns)

    def test_merged_df_reports_the_added_columns(self):
        out = self._capture(self.data.merged_df)
        self.assertIn("New variables from the flux post-processing chain", out)
        self.assertIn("only new variables added", out)

    def test_gapfilled_variables_holds_both_sides(self):
        from diive.flux.postprocessingchain import gapfilled_variables
        df = gapfilled_variables(self.data)
        self.assertIn(self.data.gapfilled_cols()['mds']['CUT_50'], df.columns)
        self.assertIn(self.data.nongapfilled_cols()['mds']['CUT_50'], df.columns)
        # A copy, not a view into fpc_df.
        self.assertIsNot(df, self.data.fpc_df)

    def test_report_gapfilling_variables_names_target_and_output(self):
        from diive.flux.postprocessingchain import report_gapfilling_variables
        out = self._capture(report_gapfilling_variables, self.data)
        self.assertIn(self.data.nongapfilled_cols()['mds']['CUT_50'], out)
        self.assertIn(self.data.gapfilled_cols()['mds']['CUT_50'], out)
        self.assertIn('MDS', out)

    def test_ml_only_reports_say_so_for_mds_instead_of_raising(self):
        """MDS has no train/test split, no year pools and no SHAP importances.

        The long-term gap-fillers expose those as properties that *raise* when
        unset, so a plain hasattr() guard would let the exception through.
        """
        from diive.flux.postprocessingchain import (
            report_gapfilling_feature_importances,
            report_gapfilling_poolyears,
            report_traintest_details,
            report_traintest_model_scores,
        )
        for fn in (report_traintest_model_scores, report_traintest_details,
                   report_gapfilling_feature_importances, report_gapfilling_poolyears):
            out = self._capture(fn, self.data)
            self.assertIn('MDS', out, msg=f"{fn.__name__} said nothing about MDS")

    def test_report_annual_sums_matches_the_gap_filled_series(self):
        from diive.flux.postprocessingchain import report_annual_sums
        gain = 12.011 * 1e-6 * 1800
        out = self._capture(report_annual_sums, self.data, gain=gain, units='gC m-2')
        sums = report_annual_sums(self.data, gain=gain, verbose=0)
        col = self.data.gapfilled_cols()['mds']['CUT_50']
        series = self.data.fpc_df[col]
        expected = series.multiply(gain).groupby(series.index.year).sum(min_count=1)
        self.assertTrue(sums[col].equals(expected))
        self.assertIn('Annual sums (gC m-2)', out)
        self.assertIn(f"{expected.iloc[0]:,.0f}", out)

    def test_merged_df_summary_counts_columns_per_level(self):
        out = self._capture(self.data.merged_df)
        new_cols = [c for c in self.data.fpc_df.columns if c not in self.data.full_df.columns]
        self.assertIn(f"chain: {len(new_cols)} (", out)
        self.assertIn('L4.1: ', out)
        # The full column list is a debug detail, not part of the default output.
        self.assertNotIn(self.data.gapfilled_cols()['mds']['CUT_50'], out)

    def test_report_gapfilling_model_scores_prints_a_table(self):
        from diive.flux.postprocessingchain import report_gapfilling_model_scores
        out = self._capture(report_gapfilling_model_scores, self.data)
        self.assertIn('MDS', out)

    def test_report_writes_csv_when_outpath_given(self):
        import tempfile
        from pathlib import Path
        from diive.flux.postprocessingchain import report_gapfilling_model_scores
        with tempfile.TemporaryDirectory() as tmp:
            report_gapfilling_model_scores(self.data, outpath=tmp)
            written = list(Path(tmp).glob('*.csv'))
            self.assertTrue(written, "no CSV written")
            self.assertIn('CUT_50', written[0].name)

    def test_plot_gapfilled_cumulative_over_the_whole_record(self):
        import matplotlib.pyplot as plt
        from diive.flux.postprocessingchain import plot_gapfilled_cumulative
        before = len(plt.get_fignums())
        plot_gapfilled_cumulative(self.data, gain=12.011 * 1e-6 * 1800, units='gC m-2',
                                  per_year=False, showplot=False)
        self.assertGreater(len(plt.get_fignums()), before)
        plt.close('all')

    def test_plot_mds_gapfilling_qualities_runs(self):
        import matplotlib.pyplot as plt
        from diive.flux.postprocessingchain import plot_mds_gapfilling_qualities
        plot_mds_gapfilling_qualities(self.data)
        plt.close('all')

    def test_quality_classes_split_the_records(self):
        """QCF 0 lies inside the used records, and used and rejected never share a record."""
        from diive.flux.postprocessingchain.reports import _quality_classes
        q = _quality_classes(self.data, 'CUT_50')
        self.assertTrue((q['best'].notna() <= q['used'].notna()).all())
        self.assertFalse((q['used'].notna() & q['rejected'].notna()).any())
        self.assertEqual(int(q['qcf1'].count()), int(q['used'].count() - q['best'].count()))

    def test_quality_plots_draw_one_figure_each(self):
        import matplotlib.pyplot as plt
        from diive.flux.postprocessingchain import (plot_nighttime_negatives, plot_quality_heatmaps,
                                                    plot_quality_timeseries)
        for fn in (plot_quality_heatmaps, plot_quality_timeseries, plot_nighttime_negatives):
            plt.close('all')
            fn(self.data, showplot=False)
            self.assertEqual(len(plt.get_fignums()), 1, msg=fn.__name__)
        plt.close('all')

    def test_report_nighttime_negatives_counts_below_the_limit(self):
        from diive.flux.postprocessingchain import report_nighttime_negatives
        table = report_nighttime_negatives(self.data, z_values=(1.96,), verbose=0)
        used = self.data.levels.filteredseries_level33_qcf['CUT_50']
        night = self.data.fpc_df['NIGHTTIME'] == 1
        nee = used[night].dropna()
        sigma = self.data.full_df['FC_RANDUNC_HF'].reindex(nee.index)
        keep = sigma.notna() & (sigma > 0)
        expected = int((nee[keep] / sigma[keep] < -1.96).sum())
        self.assertEqual(int(table.loc[0, 'records']), expected)
        self.assertEqual(int(table.loc[0, 'night records']), int(keep.sum()))

    def test_gapfilling_input_counts_add_up(self):
        from diive.flux.postprocessingchain import report_gapfilling_input
        out = report_gapfilling_input(self.data, verbose=0)
        self.assertEqual(list(out.index), list(self.data.levels.filteredseries_level33_qcf))
        for scenario, row in out.iterrows():
            self.assertEqual(row['removed by tests and outliers'] + row['removed by USTAR']
                             + row['used for gap-filling'], row['measured'])
            self.assertEqual(row['QCF 0'] + row['QCF 1'], row['used for gap-filling'])
            self.assertEqual(row['used by day'] + row['used by night'], row['used for gap-filling'])
            used = self.data.levels.filteredseries_level33_qcf[scenario]
            self.assertEqual(row['used for gap-filling'], int(used.count()))
            self.assertEqual(row['QCF 0'], int(self.data.levels.filteredseries_level33_hq[scenario].count()))

    def test_report_gaps(self):
        from diive.flux.postprocessingchain import report_gaps
        out = report_gaps(self.data, verbose=0)
        used = self.data.levels.filteredseries_level33_qcf['CUT_50']
        self.assertAlmostEqual(out.loc['CUT_50', 'missing %'], round(100 * used.isna().mean(), 1))
        self.assertGreater(out.loc['CUT_50', 'gaps'], 0)

    def test_plot_gapfilling_input_draws_two_rows(self):
        import matplotlib.pyplot as plt
        from diive.flux.postprocessingchain import plot_gapfilling_input
        try:
            plot_gapfilling_input(self.data, units='umol m-2 s-1', showplot=False)
            fig = plt.gcf()
            panels = [ax for ax in fig.axes if ax.images]
            self.assertEqual(len(panels), 2 * len(self.data.levels.filteredseries_level33_qcf))
        finally:
            plt.close('all')

    def test_nighttime_uncertainty_gives_four_uncertainties(self):
        from diive.flux.postprocessingchain import make_level32_detector, nighttime_uncertainty
        data, sod = make_level32_detector(self.data)
        sigma = nighttime_uncertainty(data, sod, randunc_col='FC_RANDUNC_HF', ta='TA_1_1_1',
                                      vpd='VPD_kPa', swin='SW_IN_1_1_1', verbose=0)
        self.assertEqual(list(sigma.columns), ['FS', 'PAS20', 'MAX', 'POOLED'])
        both = sigma.dropna()
        self.assertGreater(len(both), 0)
        self.assertTrue((both['MAX'] >= both['FS']).all() and (both['MAX'] >= both['PAS20']).all())
        # Only records with a value get an uncertainty.
        self.assertFalse(sigma['FS'][sod.series_hires_cleaned.isna()].notna().any())

    def test_nighttime_uncertainty_falls_back_where_fs_is_missing(self):
        from dataclasses import replace
        import pandas as pd
        from diive.flux.postprocessingchain import make_level32_detector, nighttime_uncertainty
        full = self.data.full_df.copy()
        full.iloc[:500, full.columns.get_loc('FC_RANDUNC_HF')] = float('nan')
        data, sod = make_level32_detector(replace(self.data, full_df=full))
        sigma = nighttime_uncertainty(data, sod, randunc_col='FC_RANDUNC_HF', ta='TA_1_1_1',
                                      vpd='VPD_kPa', swin='SW_IN_1_1_1', verbose=0)
        no_fs = sigma['FS'].isna() & sigma['PAS20'].notna()
        self.assertGreater(int(no_fs.sum()), 0)
        pd.testing.assert_series_equal(sigma.loc[no_fs, 'MAX'], sigma.loc[no_fs, 'PAS20'],
                                       check_names=False)

    def test_pooled_night_spread_is_not_widened_by_a_few_bad_values(self):
        """POOLED: robust spread of all night values within 15 days, at night records only. A few
        bad night values barely change it, so they fall far below -4 times it."""
        from dataclasses import replace
        import numpy as np
        import pandas as pd
        from diive.flux.postprocessingchain import make_level32_detector, nighttime_uncertainty
        args = dict(randunc_col='FC_RANDUNC_HF', ta='TA_1_1_1', vpd='VPD_kPa', swin='SW_IN_1_1_1',
                    verbose=0)
        data, sod = make_level32_detector(self.data)
        clean = nighttime_uncertainty(data, sod, **args)['POOLED']
        night = self.data.fpc_df['NIGHTTIME'] == 1
        self.assertFalse(clean[~night.reindex(clean.index, fill_value=False)].notna().any())
        # The value against a manual computation, at one night record in the middle.
        series = sod.series_hires_cleaned
        values = series[night.reindex(series.index, fill_value=False) & series.notna()]
        t = values.index[len(values) // 2]
        window = values[t - pd.Timedelta(days=15):t + pd.Timedelta(days=15)]
        expected = 1.4826 * np.median(np.abs(window - window.median()))
        self.assertAlmostEqual(float(clean[t]), float(expected), places=9)

        bad = values.index[len(values) // 2: len(values) // 2 + 3]
        df = self.data.fpc_df.copy()
        df.loc[bad, sod.series_hires_cleaned.name] = -40.0
        data_bad, sod_bad = make_level32_detector(replace(self.data, fpc_df=df))
        pooled = nighttime_uncertainty(data_bad, sod_bad, **args)['POOLED']
        self.assertLess(float((pooled[bad] / clean[bad]).max()), 1.2)
        self.assertTrue((-40.0 < -4 * pooled[bad]).all())


class TestLevel41SharedFeatures(unittest.TestCase):
    """With reduce_features, all USTAR scenarios use the features any of them selected."""

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        from diive.flux.postprocessingchain import (
            init_flux_data, run_level2, run_level31, make_level32_detector,
            run_level32, run_level33_constant_ustar)
        df = _eddypro_input()
        data = init_flux_data(df=df, fluxcol='FC', site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
        data = run_level31(data, gapfill_storage_term=True)
        data, sod = make_level32_detector(data)
        sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False, verbose=False)
        sod.addflag()
        data = run_level32(data, outlier_detector=sod)
        cls.data = run_level33_constant_ustar(data, thresholds=[0.1, 0.2],
                                              threshold_labels=['CUT_16', 'CUT_84'],
                                              showplot=False, verbose=False)

    def _run(self, shared):
        import pandas as pd
        from diive.flux.postprocessingchain.levels.level41 import _run_level41_ml
        selections = iter([['a'], ['b']])
        made = []

        class Engineer:
            def fit_transform(self, df):
                return pd.DataFrame({'a': 1.0, 'b': 2.0}, index=df.index)

        class Model:
            def __init__(self, input_df, target_col, verbose=0, test_records=None, **kw):
                self.input_df, self.target_col, self.used = input_df, target_col, None
                # Long-term models expose the test records they hold out (L412).
                self.test_records_ = test_records if test_records is not None else frozenset()
                made.append(self)

            def create_yearpools(self):
                pass

            def initialize_yearly_models(self):
                pass

            def reduce_features_across_years(self):
                self.features_reduced_across_years = next(selections)

            def use_features(self, features):
                self.used = list(features)

            def fillgaps(self):
                name = f'{self.target_col}_gfFAKE'
                self.gapfilled_ = self.input_df[self.target_col].fillna(0).rename(name)
                self.gapfilling_df_ = pd.DataFrame({
                    name: self.gapfilled_,
                    f'FLAG_{name}_ISFILLED': self.input_df[self.target_col].isna().astype(int)})

        _run_level41_ml(self.data, model_factory=Model, features=['TA_1_1_1'], engineer=Engineer(),
                        reduce_features=True, verbose=0, model_kwargs={}, results_attr='level41_xgb',
                        shared_features=shared)
        return made

    def test_scenarios_share_the_union_of_their_features(self):
        made = self._run(shared=True)
        self.assertEqual(len(made), 2)
        self.assertEqual([m.used for m in made], [['a', 'b'], ['a', 'b']])

    def test_each_scenario_keeps_its_own_selection_when_not_shared(self):
        made = self._run(shared=False)
        self.assertEqual([m.used for m in made], [None, None])


def _eddypro_input():
    """EddyPro example data without the columns init_flux_data computes."""
    from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
    df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
    df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
    df['TA_1_1_1'] = df['TA_1_1_1'].bfill()
    df['SW_IN_1_1_1'] = df['SW_IN_1_1_1'].bfill()
    df['VPD_kPa'] = df['VPD_EP'].bfill().multiply(0.1)  # hPa -> kPa for MDS
    return df


class TestInitIrregularIndex(unittest.TestCase):
    """init_flux_data puts data with missing timestamps on a regular grid (L332)."""

    def test_chain_runs_on_data_with_missing_timestamps(self):
        import pandas as pd
        from diive.flux.postprocessingchain import FluxConfig, init_flux_data, run_chain

        df = _eddypro_input()
        dropped = df.index[500:510]
        df = df.drop(index=dropped)
        self.assertIsNone(df.index.freq)
        df_before = df.copy()

        data = init_flux_data(df=df, fluxcol='FC',
                              site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        # The caller's frame is not changed
        pd.testing.assert_frame_equal(df, df_before)
        self.assertIsNone(df.index.freq)
        # The missing timestamps are back, as empty records
        self.assertEqual(len(data.fpc_df), len(df) + len(dropped))
        self.assertTrue(data.fpc_df.loc[dropped, 'FC'].isna().all())
        self.assertTrue(data.full_df.loc[dropped, 'TA_1_1_1'].isna().all())
        self.assertIsNotNone(data.fpc_df.index.freq)

        cfg = FluxConfig(
            fluxcol='FC',
            ustar_thresholds=[0.1], ustar_labels=['CUT_50'],
            outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,
            level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
            gapfill_rf=False, gapfill_xgb=False, gapfill_mds=True,
            mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa',
        )
        out = run_chain(data, cfg)
        for lvl in ('L2', 'L3.1', 'L3.2', 'L3.3', 'L4.1'):
            self.assertIn(lvl, out.level_ids)
        self.assertTrue(out.levels.filteredseries_level32_qcf.loc[dropped].isna().all())
        self.assertIn(out.gapfilled_cols()['mds']['CUT_50'], out.fpc_df.columns)

    def test_undetectable_frequency_raises_a_clear_error(self):
        import numpy as np
        import pandas as pd
        from diive.flux.postprocessingchain import init_flux_data

        # Time steps between 1 and 300 minutes, no step dominates
        steps = np.random.default_rng(1).integers(1, 300, size=200)
        ix = pd.DatetimeIndex(pd.Timestamp('2024-01-01')
                              + pd.to_timedelta(np.cumsum(steps), unit='min'),
                              name='TIMESTAMP_MIDDLE')
        df = pd.DataFrame({'FC': 1.0, 'USTAR': 0.3}, index=ix)
        with self.assertRaises(ValueError) as ctx:
            init_flux_data(df=df, fluxcol='FC',
                           site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        self.assertIn('time resolution', str(ctx.exception))
        self.assertIn("asfreq", str(ctx.exception))


class TestUstarNoneKeepsMissingUstar(unittest.TestCase):
    """thresholds=[0] with CUT_NONE flags nothing, also without USTAR (L347)."""

    @staticmethod
    def _run_to_l33(fluxcol, thresholds, labels, no_ustar):
        from diive.flux.postprocessingchain import (
            init_flux_data, run_level2, run_level31, make_level32_detector,
            run_level32, run_level33_constant_ustar)
        df = _eddypro_input()
        df.loc[no_ustar, 'USTAR'] = float('nan')
        data = init_flux_data(df=df, fluxcol=fluxcol,
                              site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
        if fluxcol == 'FC':
            data = run_level31(data, gapfill_storage_term=True)
        else:
            data = run_level31(data, set_storage_to_zero=True)
        data, sod = make_level32_detector(data)
        sod.flag_outliers_abslim_test(minval=-1000, maxval=1000, showplot=False, verbose=False)
        sod.addflag()
        data = run_level32(data, outlier_detector=sod)
        data = run_level33_constant_ustar(data, thresholds=thresholds,
                                          threshold_labels=labels, showplot=False)
        results = data.levels.level33.get_results()
        flagcol = [c for c in results.columns if str(c).startswith('FLAG') and 'USTAR' in str(c)]
        return data, results[flagcol[0]]

    def test_cut_none_keeps_records_without_ustar(self):
        df = _eddypro_input()
        # Records with a measured flux, USTAR removed
        no_ustar = df.index[df['LE'].notna() & df['USTAR'].notna()][100:130]
        data, flag = self._run_to_l33('LE', [0], ['CUT_NONE'], no_ustar)
        self.assertTrue((flag.loc[no_ustar] == 0).all())
        self.assertEqual(int((flag == 2).sum()), 0)
        # Nothing is lost between L3.2 and L3.3
        l32 = data.levels.filteredseries_level32_qcf
        l33 = data.levels.filteredseries_level33_qcf['CUT_NONE']
        self.assertEqual(l33.dropna().count(), l32.dropna().count())

    def test_real_threshold_still_rejects_records_without_ustar(self):
        df = _eddypro_input()
        no_ustar = df.index[df['FC'].notna() & df['USTAR'].notna()][100:130]
        _, flag = self._run_to_l33('FC', [0.1], ['CUT_50'], no_ustar)
        self.assertTrue((flag.loc[no_ustar] == 2).all())



class TestFluxProcessingChainAlias(unittest.TestCase):
    """The old import path keeps working and points at the same modules."""

    def test_old_path_warns_and_aliases_new_modules(self):
        import importlib
        import sys
        import diive.flux.postprocessingchain as new
        for name in [k for k in sys.modules if k.startswith('diive.flux.fluxprocessingchain')]:
            del sys.modules[name]
        with self.assertWarns(FutureWarning):
            old = importlib.import_module('diive.flux.fluxprocessingchain')
        self.assertIs(old, new)
        from diive.flux.fluxprocessingchain.levels.level2 import run_level2
        self.assertIs(run_level2, new.run_level2)
        from diive.flux.fluxprocessingchain.container import FluxLevelData
        self.assertIs(FluxLevelData, new.FluxLevelData)


class TestStorageScreening(unittest.TestCase):
    """screen_storage_term removes spikes from the storage term and fills them."""

    @staticmethod
    def _df():
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(5)
        index = pd.date_range('2022-07-01 00:15', periods=48 * 30, freq='30min', name='TIMESTAMP_MIDDLE')
        df = pd.DataFrame({'FC': rng.normal(-5, 2, len(index)),
                           'SC_SINGLE': rng.normal(0, 0.5, len(index))}, index=index)
        spike = index[700]
        df.loc[spike, 'SC_SINGLE'] = 60.0
        df.loc[index[100], 'SC_SINGLE'] = np.nan  # an ordinary gap
        return df, spike

    def test_spike_is_removed_and_filled(self):
        from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
        df, spike = self._df()
        sc = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2',
                                                      screen_storage_term=True)
        sc.storage_correction()
        r = sc.results
        # The double differences also take the spike's two neighbours; they are filled too.
        self.assertIn(spike, sc.storage_outliers_)
        self.assertLessEqual(len(sc.storage_outliers_), 3)
        self.assertEqual(r.loc[spike, sc.flag_isgapfilled], 2)
        self.assertEqual(r.loc[df.index[100], sc.flag_isgapfilled], 1)
        # The block (spikes and one record on either side) gets its own mean: the spike is
        # spread out, the sum of the block is kept.
        block = sc.storage_replaced_
        self.assertTrue(set(sc.storage_outliers_) <= set(block))
        self.assertEqual(len(block), len(sc.storage_outliers_) + 2)
        self.assertAlmostEqual(r.loc[block, sc.gapfilled_strgcol].sum(), r.loc[block, 'SC_SINGLE'].sum(), places=9)
        self.assertLess(r.loc[spike, sc.gapfilled_strgcol], 60.0 / len(block) + 2)
        self.assertTrue((r.loc[block, sc.flag_isgapfilled] == 2).all())
        self.assertEqual(r.loc[spike, 'SC_SINGLE'], 60.0)  # the measured value is kept
        # Every measured flux keeps a storage term.
        self.assertEqual(int(r['NEE_L3.1'].notna().sum()), int(df['FC'].notna().sum()))

    def test_screening_summary_and_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
        df, spike = self._df()
        sc = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2',
                                                      screen_storage_term=True)
        sc.storage_correction()
        s = sc.screening_summary()
        self.assertEqual(s['spikes'], len(sc.storage_outliers_))
        self.assertEqual(s['replaced'], len(sc.storage_replaced_))
        r = sc.results
        # Runs keep their sums, so the screening does not change the storage sum.
        self.assertAlmostEqual(s['sum_unscreened'], s['sum_used'], places=9)
        try:
            fig = sc.plot_screening(showplot=False)
            self.assertEqual(len(fig.axes), 3)
        finally:
            plt.close('all')

    def test_single_spike_is_spread_over_three_records(self):
        """A spike flagged alone still changes: it shares its value with its neighbours."""
        import pandas as pd
        from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
        df, spike = self._df()
        sc = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2',
                                                      screen_storage_term=True)
        sc.storage_outliers_ = pd.DatetimeIndex([spike])  # as if the filter flagged it alone
        sc._results = df[['FC', 'SC_SINGLE']].copy()
        means = sc._screened_block_means()
        self.assertEqual(len(means), 3)
        self.assertAlmostEqual(float(means.iloc[0]), float(df['SC_SINGLE'].loc[means.index].mean()))
        self.assertLess(float(means.loc[spike]), 25)

    def test_long_runs_stay_as_measured(self):
        """A noisy block is not a spike: it is kept, not filled with one value."""
        import numpy as np
        from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
        df, spike = self._df()
        block = df.index[900:920]
        df.loc[block, 'SC_SINGLE'] = np.tile([40.0, -40.0], 10)
        sc = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2',
                                                      screen_storage_term=True)
        sc.storage_correction()
        self.assertIn(spike, sc.storage_outliers_)
        self.assertFalse(sc.storage_outliers_.isin(block).any())

    def test_without_screening_the_spike_stays(self):
        from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
        df, spike = self._df()
        sc = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2')
        sc.storage_correction()
        self.assertEqual(sc.results.loc[spike, sc.gapfilled_strgcol], 60.0)

    def test_screening_needs_gapfilling(self):
        from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
        df, _ = self._df()
        with self.assertRaises(ValueError):
            FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2',
                                                    gapfill_storage_term=False, screen_storage_term=True)

    def test_run_level31_passes_it_on(self):
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.postprocessingchain import init_flux_data, run_level2, run_level31
        df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
        spike = df.index[df['FC'].notna() & df['SC_SINGLE'].notna()][300]
        df.loc[spike, 'SC_SINGLE'] = 500.0
        data = init_flux_data(df=df, fluxcol='FC', site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
        data31 = run_level31(data, screen_storage_term=True)
        self.assertIn(spike, data31.levels.level31.storage_outliers_)


class TestStorageRestarts(unittest.TestCase):
    """concentration_col discards the storage term where the analyser is off next to a gap."""

    H = 40.0  # m, storage column height of the synthetic site

    def _df(self):
        """30 days of a concentration with a daily cycle and its single-point storage term.

        At record 700 the analyser restarts after a 6-record gap, reading 15 mmol m-3 high,
        and returns to the real concentration over 6 records. Before the gap at record 1000
        it drifts 12 mmol m-3 up over 4 records. As in EddyPro, the first storage value after
        a gap is missing.
        """
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(3)
        index = pd.date_range('2022-07-01 00:15', periods=48 * 30, freq='30min', name='TIMESTAMP_MIDDLE')
        hour = index.hour + index.minute / 60
        real = 16.5 + 0.8 * np.cos(2 * np.pi * (hour - 5) / 24) + rng.normal(0, 0.05, len(index))
        conc = pd.Series(real, index=index)
        conc.iloc[694:700] = np.nan
        conc.iloc[700:706] += 15 * np.exp(-np.arange(6) / 1.5)
        conc.iloc[996:1000] += [3, 6, 9, 12]
        conc.iloc[1000:1003] = np.nan
        strg = self.H * conc.diff() * 1000 / 1800
        df = pd.DataFrame({'FC': rng.normal(-5, 2, len(index)), 'SC_SINGLE': strg,
                           'CO2_MOLAR_DENSITY': conc}, index=index)
        return df

    def _run(self, **kwargs):
        from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
        df = self._df()
        sc = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2',
                                                      concentration_col='CO2_MOLAR_DENSITY', **kwargs)
        sc.storage_correction()
        return df, sc

    def test_transient_after_restart_is_discarded_and_filled(self):
        df, sc = self._run()
        r = sc.results
        after = df.index[701:706]  # the first record after the gap has no storage value
        self.assertIn(df.index[700], sc.storage_restarts_)
        self.assertTrue(set(after) <= set(sc.storage_discarded_))
        self.assertTrue((r.loc[after, sc.flag_isgapfilled] == 3).all())
        self.assertTrue((r.loc[after, sc.gapfilled_strgcol].abs() < 5).all())
        self.assertTrue((r.loc[after, 'SC_SINGLE'] < -20).any())  # the measured values are kept
        # Normal records stay as measured; every measured flux keeps a storage term.
        self.assertTrue((r.iloc[300:600][sc.flag_isgapfilled] == 0).all())
        self.assertEqual(int(r['NEE_L3.1'].notna().sum()), int(df['FC'].notna().sum()))

    def test_drift_before_a_gap_is_discarded(self):
        df, sc = self._run()
        self.assertTrue(set(df.index[996:1000]) <= set(sc.storage_discarded_))
        _, sc_off = self._run(restart_screening={'before_gaps': False})
        self.assertFalse(sc_off.storage_discarded_.isin(df.index[996:1000]).any())

    def test_storage_sum_is_balanced_again(self):
        df, sc = self._run()
        s = sc.screening_summary()
        self.assertEqual(s['discarded'], len(sc.storage_discarded_))
        self.assertGreater(s['restarts'], 0)
        # Measured: the one-sided return after the restart and the drift before the gap.
        self.assertLess(abs(s['sum_used']), 0.2 * abs(s['sum_unscreened']))

    def test_plot_and_report(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _, sc = self._run()
        sc.report()
        try:
            self.assertEqual(len(sc.plot_screening(showplot=False).axes), 3)
        finally:
            plt.close('all')

    def test_needs_the_column_and_gapfilling(self):
        from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
        df = self._df()
        with self.assertRaises(KeyError):
            FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2',
                                                    concentration_col='CO2_X')
        with self.assertRaises(ValueError):
            FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2',
                                                    gapfill_storage_term=False,
                                                    concentration_col='CO2_MOLAR_DENSITY')



class TestModelOverview(unittest.TestCase):
    """report_model_overview describes each year's XGBoost model."""

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        from diive.core.ml.feature_engineer import FeatureEngineer
        from diive.flux.postprocessingchain import (
            init_flux_data, run_level2, run_level31, make_level32_detector,
            run_level32, run_level33_constant_ustar, run_level41_xgb)
        df = _eddypro_input()
        data = init_flux_data(df=df, fluxcol='FC', site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
        data = run_level31(data, gapfill_storage_term=True)
        data, sod = make_level32_detector(data)
        sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False, verbose=False)
        sod.addflag()
        data = run_level32(data, outlier_detector=sod)
        data = run_level33_constant_ustar(data, thresholds=[0.1, 0.2], threshold_labels=['CUT_16', 'CUT_84'],
                                          showplot=False, verbose=False)
        engineer = FeatureEngineer(target_col='_target_', features_lag=[-1, 1], vectorize_timestamps=True,
                                   verbose=0)
        data = run_level41_xgb(data, features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_kPa'],
                               engineer=engineer, n_estimators=60, learning_rate=0.3,
                               early_stopping_rounds=5, random_state=42, verbose=0)
        import warnings
        from diive.flux.postprocessingchain import run_level41_mds
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.data = run_level41_mds(data, swin='SW_IN_1_1_1', ta='TA_1_1_1', vpd='VPD_kPa')

    def test_feature_ranks_skip_mds(self):
        import matplotlib.pyplot as plt
        from diive.flux.postprocessingchain import plot_feature_ranks_per_year
        plt.close('all')
        try:
            plot_feature_ranks_per_year(self.data)
            self.assertEqual(len(plt.gcf().axes), 2)  # one panel per XGBoost scenario
        finally:
            plt.close('all')

    def test_wide_report_table_is_not_cut(self):
        import io
        from unittest import mock
        from rich.console import Console
        from diive.flux.postprocessingchain import reports
        buf = io.StringIO()
        table = reports._table()
        for year in range(2016, 2027):
            table.add_column(str(year), justify='right')
        table.add_row(*['12.345'] * 11)
        with mock.patch.object(reports, '_console', Console(width=60, file=buf)), \
                mock.patch.object(reports, '_show', return_value=True):
            reports._print_report('wide', table)
            self.assertEqual(reports._console.width, 60)  # restored afterwards
        self.assertNotIn('…', buf.getvalue())
        self.assertEqual(buf.getvalue().count('12.345'), 11)

    def test_one_row_per_scenario_and_year(self):
        from diive.flux.postprocessingchain import report_model_overview
        out = report_model_overview(self.data, verbose=0)
        self.assertEqual(list(out['USTAR']), ['CUT_16', 'CUT_84'])
        for _, row in out.iterrows():
            inst = self.data.levels.level41_xgb[row['USTAR']].results_yearly_[row['year']]
            self.assertEqual(row['trees'], inst._n_trees)
            self.assertLessEqual(row['trees'], 60)
            self.assertEqual(row['train'] + row['test'], row['final fit'])
            self.assertEqual(row['measured'] + row['filled'] + row['fallback'],
                             int((inst.gapfilling_df_.index.year == int(row['year'])).sum()))

    def test_gapfilling_comparison_adds_up(self):
        from diive.flux.postprocessingchain import report_gapfilling_comparison
        out = report_gapfilling_comparison(self.data, methods=('mds', 'xgb'), verbose=0)
        self.assertEqual(list(out.index), ['CUT_16', 'CUT_84'])
        gf = self.data.gapfilled_cols()
        for sc, r in out.iterrows():
            a, b = self.data.fpc_df[gf['mds'][sc]], self.data.fpc_df[gf['xgb'][sc]]
            self.assertAlmostEqual(r['sum XGBoost'] - r['sum MDS'], r['difference'], places=6)
            self.assertAlmostEqual(float(b.sum() - a.sum()), r['difference'], places=6)
            self.assertAlmostEqual(r['from day gaps'] + r['from night gaps'], r['difference'], places=6)
            self.assertAlmostEqual(sum(r[f'gaps {c}'] for c in ('1', '2-12', '13-48', '>48')),
                                   r['difference'], places=6)
            self.assertGreater(r['test records'], 0)

    def test_gapfilling_comparison_plot(self):
        import matplotlib.pyplot as plt
        from diive.flux.postprocessingchain import plot_gapfilling_comparison
        try:
            fig = plot_gapfilling_comparison(self.data, showplot=False)
            self.assertEqual(len([ax for ax in fig.axes if ax.get_label() != '<colorbar>']), 4)
        finally:
            plt.close('all')

    def test_quality_plots_take_the_method(self):
        from diive.flux.postprocessingchain.reports import _quality_classes
        q = _quality_classes(self.data, 'CUT_16', method='xgb')
        self.assertEqual(q['method'], 'xgb')
        self.assertEqual(q['filled'].name, self.data.gapfilled_cols()['xgb']['CUT_16'])
        self.assertEqual(_quality_classes(self.data, 'CUT_16')['method'], 'mds')  # first that ran
        with self.assertRaises(ValueError):
            _quality_classes(self.data, 'CUT_16', method='rf')

    def test_feature_importances_for_one_scenario(self):
        from diive.core.utils.console import console
        from diive.flux.postprocessingchain import report_gapfilling_feature_importances
        with console.capture() as cap:
            report_gapfilling_feature_importances(self.data, ustar_scenario='CUT_84', verbose=True)
        text = cap.get()
        self.assertIn('CUT_84', text)
        self.assertNotIn('CUT_16', text)


class TestBlockedTreeSelection(unittest.TestCase):
    """n_estimators_cv chooses one number of trees by blocked cross-validation."""

    def test_blocked_folds_hold_out_whole_blocks(self):
        import numpy as np
        import pandas as pd
        from diive.gapfilling.tree_selection import blocked_folds
        index = pd.date_range('2022-01-01 00:15', periods=48 * 90, freq='30min')
        index = index.delete(np.arange(100, 400))  # a gap does not matter
        folds = blocked_folds(index, block_days=7, n_folds=4, random_state=1)
        self.assertEqual(len(folds), 4)
        held_out = np.concatenate([test for _, test in folds])
        self.assertEqual(sorted(held_out.tolist()), list(range(len(index))))
        block = ((index - index[0]) / pd.Timedelta(days=7)).astype(int)
        for train, test in folds:
            self.assertEqual(len(set(train) & set(test)), 0)
            # A block is either held out as a whole or not at all.
            self.assertEqual(set(block[test]) & set(block[train]), set())
        with self.assertRaises(ValueError):
            blocked_folds(index[:48], block_days=7, n_folds=4)

    def test_selection_on_synthetic_pools(self):
        import numpy as np
        import pandas as pd
        from diive.gapfilling.tree_selection import select_n_estimators_blocked_cv
        rng = np.random.default_rng(0)
        pools = {}
        for label in ('2020-2022', '2021-2023'):
            index = pd.date_range('2021-01-01 00:15', periods=48 * 120, freq='30min')
            x = rng.normal(size=(len(index), 3))
            y = 2 * x[:, 0] - x[:, 1] ** 2 + rng.normal(0, 0.5, len(index))
            pools[label] = pd.DataFrame({'y': y, 'a': x[:, 0], 'b': x[:, 1], 'c': x[:, 2]},
                                        index=index)
        sel = select_n_estimators_blocked_cv(pools, 'y', dict(n_estimators=150, learning_rate=0.1,
                                                                 max_depth=3, random_state=1,
                                                                 early_stopping_rounds=10),
                                             block_days=10, n_folds=4)
        self.assertLessEqual(sel.n_estimators, 150)
        self.assertEqual(sel.n_estimators % 10, 0)
        self.assertEqual(sel.curves.shape, (150, 2))
        self.assertEqual(list(sel.scores.index), list(pools))
        # The minimum of the median curve, rounded to ten.
        best = int(sel.median_curve.to_numpy().argmin()) + 1
        self.assertLessEqual(abs(sel.n_estimators - best), 5)
        # With a tolerance, the smallest number within it of the minimum: never more trees.
        sel_tol = select_n_estimators_blocked_cv(pools, 'y', dict(n_estimators=150, learning_rate=0.1,
                                                                     max_depth=3, random_state=1),
                                                 block_days=10, n_folds=4, tolerance=0.05)
        self.assertLessEqual(sel_tol.n_estimators, sel.n_estimators)
        self.assertTrue((sel.scores['r2'] > 0.5).all())

    def test_run_level41_xgb_uses_one_fixed_number(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from diive.core.ml.feature_engineer import FeatureEngineer
        from diive.flux.postprocessingchain import (
            init_flux_data, run_level2, run_level31, make_level32_detector, run_level32,
            run_level33_constant_ustar, run_level41_xgb, report_tree_selection,
            plot_tree_selection, report_model_overview)
        df = _eddypro_input()
        data = init_flux_data(df=df, fluxcol='FC', site_lat=46.583056, site_lon=9.790639, utc_offset=1)
        data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
        data = run_level31(data, gapfill_storage_term=True)
        data, sod = make_level32_detector(data)
        sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False, verbose=False)
        sod.addflag()
        data = run_level32(data, outlier_detector=sod)
        data = run_level33_constant_ustar(data, thresholds=[0.1, 0.2], threshold_labels=['CUT_16', 'CUT_84'],
                                          showplot=False, verbose=False)
        engineer = FeatureEngineer(target_col='_target_', features_lag=[-1, 1], vectorize_timestamps=True,
                                   verbose=0)
        data = run_level41_xgb(data, features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_kPa'], engineer=engineer,
                               n_estimators=80, learning_rate=0.3, early_stopping_rounds=5,
                               random_state=42, verbose=0,
                               n_estimators_cv={'block_days': 7, 'n_folds': 3})
        sel = data.levels.level41_xgb['CUT_84'].tree_selection_
        self.assertEqual(sel.scenario, 'CUT_84')  # the middle of two scenarios
        self.assertLessEqual(sel.n_estimators, 80)
        for inst in data.levels.level41_xgb.values():
            self.assertIs(inst.tree_selection_, sel)
            for model in inst.results_yearly_.values():
                self.assertEqual(model.kwargs['n_estimators'], sel.n_estimators)
                self.assertIsNone(model.kwargs['early_stopping_rounds'])
                self.assertIsNone(model._n_trees)
                self.assertEqual(model.model_.n_estimators, sel.n_estimators)
        out = report_tree_selection(data, verbose=0)
        self.assertEqual(len(out), len(sel.scores))
        self.assertTrue({'rmse', 'random RMSE', 'random R2'} <= set(out.columns))
        overview = report_model_overview(data, verbose=0)
        self.assertTrue((overview['trees'] == sel.n_estimators).all())
        try:
            fig = plot_tree_selection(data, showplot=False)
            self.assertEqual(len(fig.axes), 2)
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
