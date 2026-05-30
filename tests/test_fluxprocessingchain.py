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


if __name__ == '__main__':
    unittest.main()
