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

    def test_level31_set_storage_to_zero_without_storage_column(self):
        """set_storage_to_zero=True must not require a storage column.

        This is the documented H / LE path: those fluxes have no storage
        profile, so the storage column is missing from the input data.
        """
        import pandas as pd
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.fluxprocessingchain import init_flux_data, run_level2, run_level31

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


    def test_level42_partitioning_run_chain(self):
        """Wire the four NEE partitioning variants through run_chain (L4.2)."""
        import warnings
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.fluxprocessingchain import (
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
        from diive.flux.fluxprocessingchain import run_level33_constant_ustar
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
        from diive.flux.fluxprocessingchain import (
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


class TestFluxProcessingChainLevel2(unittest.TestCase):
    """Level 2 in isolation: quality-flag expansion + QCF on EddyPro output.

    Mirrors ``examples/flux/fluxprocessingchain/fluxprocessingchain_level2.py``,
    which runs L2 standalone on a real EddyPro FLUXNET output file.
    """

    def _init_data(self, **init_kwargs):
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.fluxprocessingchain import init_flux_data

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
        from diive.flux.fluxprocessingchain import run_level2

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
        from diive.flux.fluxprocessingchain import run_level2

        out = run_level2(self._init_data(),
                         ssitc={'apply': True, 'setflag_timeperiod': None})
        cols = [str(c) for c in out.fpc_df.columns]
        self.assertTrue(any('SSITC' in c for c in cols))
        # VM97 raw-data screening was not requested.
        self.assertFalse(any('VM97_SPIKE' in c for c in cols))

    def test_level2_signal_strength_requires_keys(self):
        """Enabling signal_strength without its keys raises a clear KeyError."""
        from diive.flux.fluxprocessingchain import run_level2

        with self.assertRaises(KeyError):
            run_level2(self._init_data(), signal_strength={'apply': True})

    def test_level2_vm97_requires_all_eight_subkeys(self):
        """Enabling VM97 with a missing sub-key raises a clear KeyError."""
        from diive.flux.fluxprocessingchain import run_level2

        with self.assertRaises(KeyError):
            run_level2(self._init_data(),
                       raw_data_screening_vm97={'apply': True, 'spikes': True})

    def test_level2_accept_threshold_changes_retained_count(self):
        """A stricter daytime accept threshold cannot retain more records."""
        from diive.flux.fluxprocessingchain import run_level2

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
        from diive.flux.fluxprocessingchain import (
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
        from diive.flux.fluxprocessingchain import run_level2
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
        from diive.flux.fluxprocessingchain import run_level31
        again = run_level31(self.d33, gapfill_storage_term=True)
        self.assertEqual(again.level_ids, ['L2', 'L3.1'])
        self.assertIsNotNone(again.levels.level2)          # upstream survives
        self.assertIsNone(again.levels.level32)
        self.assertIsNone(again.levels.level33)
        self.assertEqual(set(again.added_columns), {'L2', 'L3.1'})

    def test_rerun_does_not_mutate_the_input_container(self):
        from diive.flux.fluxprocessingchain import run_level2
        before_ids = list(self.d33.level_ids)
        before_cols = len(self.d33.fpc_df.columns)
        run_level2(self.d33, **self.LEVEL2_SETTINGS)
        self.assertEqual(self.d33.level_ids, before_ids)
        self.assertEqual(len(self.d33.fpc_df.columns), before_cols)

    # --- unit: cascade_reset ---

    def test_cascade_reset_keeps_upstream_levels(self):
        from diive.flux.fluxprocessingchain.levels._rerun import cascade_reset
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
        from diive.flux.fluxprocessingchain.levels._rerun import cascade_reset
        self.assertEqual(cascade_reset(self.d33, 'L3.2').filteredseries.name,
                         'NEE_L3.1_QCF')
        self.assertEqual(cascade_reset(self.d33, 'L3.3').filteredseries.name,
                         'NEE_L3.1_L3.2_QCF')
        # Cascading from the first level leaves nothing to fall back to.
        self.assertIsNone(cascade_reset(self.d33, 'L2').filteredseries)

    def test_cascade_reset_rejects_an_unknown_level(self):
        from diive.flux.fluxprocessingchain.levels._rerun import cascade_reset
        with self.assertRaises(ValueError) as ctx:
            cascade_reset(self.d33, 'L4.1')
        self.assertIn('L4.1', str(ctx.exception))

    def test_cascade_clears_the_additive_levels(self):
        # L4.1 / L4.2 do not cascade among themselves, but a cascade from any
        # earlier level must clear them: their output was computed against
        # upstream inputs that just became stale.
        import dataclasses
        from diive.flux.fluxprocessingchain.levels._rerun import cascade_reset
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
        from diive.flux.fluxprocessingchain.levels._rerun import drop_columns_for_key
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
        from diive.flux.fluxprocessingchain.levels._rerun import drop_columns_for_key
        self.assertIs(drop_columns_for_key(self.d33, 'L4.1_never_ran'), self.d33)

    def test_record_added_columns_attributes_new_columns(self):
        from diive.flux.fluxprocessingchain.levels._rerun import record_added_columns
        pre = list(self.d2.fpc_df.columns)
        recorded = record_added_columns(self.d31, 'L3.1', pre)
        expected = [c for c in self.d31.fpc_df.columns if c not in set(pre)]
        self.assertEqual(recorded['L3.1'], expected)
        self.assertTrue(expected, "L3.1 should add at least one column")
        # Existing entries are carried through untouched.
        self.assertEqual(recorded['L2'], list(self.d31.added_columns['L2']))


class TestChainReports(unittest.TestCase):
    """Reporting over a finished chain (diive/flux/fluxprocessingchain/reports.py)."""

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.flux.fluxprocessingchain import (
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
        self.assertIn("New variables from the flux processing chain", out)
        self.assertIn("only new variables added", out)

    def test_gapfilled_variables_holds_both_sides(self):
        from diive.flux.fluxprocessingchain import gapfilled_variables
        df = gapfilled_variables(self.data)
        self.assertIn(self.data.gapfilled_cols()['mds']['CUT_50'], df.columns)
        self.assertIn(self.data.nongapfilled_cols()['mds']['CUT_50'], df.columns)
        # A copy, not a view into fpc_df.
        self.assertIsNot(df, self.data.fpc_df)

    def test_report_gapfilling_variables_names_target_and_output(self):
        from diive.flux.fluxprocessingchain import report_gapfilling_variables
        out = self._capture(report_gapfilling_variables, self.data)
        self.assertIn(self.data.nongapfilled_cols()['mds']['CUT_50'], out)
        self.assertIn(self.data.gapfilled_cols()['mds']['CUT_50'], out)
        self.assertIn('MDS', out)

    def test_ml_only_reports_say_so_for_mds_instead_of_raising(self):
        """MDS has no train/test split, no year pools and no SHAP importances.

        The long-term gap-fillers expose those as properties that *raise* when
        unset, so a plain hasattr() guard would let the exception through.
        """
        from diive.flux.fluxprocessingchain import (
            report_gapfilling_feature_importances,
            report_gapfilling_poolyears,
            report_traintest_details,
            report_traintest_model_scores,
        )
        for fn in (report_traintest_model_scores, report_traintest_details,
                   report_gapfilling_feature_importances, report_gapfilling_poolyears):
            out = self._capture(fn, self.data)
            self.assertIn('MDS', out, msg=f"{fn.__name__} said nothing about MDS")

    def test_report_gapfilling_model_scores_prints_a_table(self):
        from diive.flux.fluxprocessingchain import report_gapfilling_model_scores
        out = self._capture(report_gapfilling_model_scores, self.data)
        self.assertIn('MDS', out)

    def test_report_writes_csv_when_outpath_given(self):
        import tempfile
        from pathlib import Path
        from diive.flux.fluxprocessingchain import report_gapfilling_model_scores
        with tempfile.TemporaryDirectory() as tmp:
            report_gapfilling_model_scores(self.data, outpath=tmp)
            written = list(Path(tmp).glob('*.csv'))
            self.assertTrue(written, "no CSV written")
            self.assertIn('CUT_50', written[0].name)

    def test_plot_gapfilled_cumulative_over_the_whole_record(self):
        import matplotlib.pyplot as plt
        from diive.flux.fluxprocessingchain import plot_gapfilled_cumulative
        before = len(plt.get_fignums())
        plot_gapfilled_cumulative(self.data, gain=12.011 * 1e-6 * 1800, units='gC m-2',
                                  per_year=False, showplot=False)
        self.assertGreater(len(plt.get_fignums()), before)
        plt.close('all')

    def test_plot_mds_gapfilling_qualities_runs(self):
        import matplotlib.pyplot as plt
        from diive.flux.fluxprocessingchain import plot_mds_gapfilling_qualities
        plot_mds_gapfilling_qualities(self.data)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
