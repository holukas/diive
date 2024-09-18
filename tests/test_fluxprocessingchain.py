import unittest


class TestFluxProcessingChain(unittest.TestCase):

    def test_fluxprocessingchain(self):
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.pkgs.fluxprocessingchain.fluxprocessingchain import FluxProcessingChain
        df, meta = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

        # Meteo variables have 1 missing value, fill
        # Used as features during gap-filling, should be gapless
        df['TA_1_1_1'] = df['TA_1_1_1'].bfill()
        df['TA_1_1_1'].isnull().sum()
        df['SW_IN_1_1_1'] = df['SW_IN_1_1_1'].bfill()
        df['SW_IN_1_1_1'].isnull().sum()
        df['VPD_EP'] = df['VPD_EP'].bfill()
        df['VPD_EP'].isnull().sum()

        # Flux processing chain settings
        FLUXVAR = "FC"
        SITE_LAT = 46.583056  # CH-AWS
        SITE_LON = 9.790639  # CH-AWS
        UTC_OFFSET = 1
        NIGHTTIME_THRESHOLD = 50  # Threshold for potential radiation in W m-2, conditions below threshold are nighttime
        DAYTIME_ACCEPT_QCF_BELOW = 2
        NIGHTTIMETIME_ACCEPT_QCF_BELOW = 2

        fpc = FluxProcessingChain(
            df=df,
            fluxcol=FLUXVAR,
            site_lat=SITE_LAT,
            site_lon=SITE_LON,
            utc_offset=UTC_OFFSET,
            nighttime_threshold=NIGHTTIME_THRESHOLD,
            daytime_accept_qcf_below=DAYTIME_ACCEPT_QCF_BELOW,
            nighttimetime_accept_qcf_below=NIGHTTIMETIME_ACCEPT_QCF_BELOW
        )
        self.assertEqual(len(fpc.fpc_df.columns), 5)

        # --------------------
        # Level-2
        # --------------------
        TEST_SSITC = True  # Default True
        TEST_GAS_COMPLETENESS = True  # Default True
        TEST_SPECTRAL_CORRECTION_FACTOR = True  # Default True
        TEST_SIGNAL_STRENGTH = True
        TEST_SIGNAL_STRENGTH_COL = 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN'
        TEST_SIGNAL_STRENGTH_METHOD = 'discard below'
        TEST_SIGNAL_STRENGTH_THRESHOLD = 60
        TEST_RAWDATA = True  # Default True
        TEST_RAWDATA_SPIKES = True  # Default True
        TEST_RAWDATA_AMPLITUDE = False  # Default True
        TEST_RAWDATA_DROPOUT = True  # Default True
        TEST_RAWDATA_ABSLIM = False  # Default False
        TEST_RAWDATA_SKEWKURT_HF = False  # Default False
        TEST_RAWDATA_SKEWKURT_SF = False  # Default False
        TEST_RAWDATA_DISCONT_HF = False  # Default False
        TEST_RAWDATA_DISCONT_SF = False  # Default False
        TEST_RAWDATA_ANGLE_OF_ATTACK = True  # Default False
        TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = [['2023-07-01', '2023-09-01']]  # Default False
        # TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = False  # Default False
        TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND = False  # Default False

        LEVEL2_SETTINGS = {
            'signal_strength': {
                'apply': TEST_SIGNAL_STRENGTH,
                'signal_strength_col': TEST_SIGNAL_STRENGTH_COL,
                'method': TEST_SIGNAL_STRENGTH_METHOD,
                'threshold': TEST_SIGNAL_STRENGTH_THRESHOLD},
            'raw_data_screening_vm97': {
                'apply': TEST_RAWDATA,
                'spikes': TEST_RAWDATA_SPIKES,
                'amplitude': TEST_RAWDATA_AMPLITUDE,
                'dropout': TEST_RAWDATA_DROPOUT,
                'abslim': TEST_RAWDATA_ABSLIM,
                'skewkurt_hf': TEST_RAWDATA_SKEWKURT_HF,
                'skewkurt_sf': TEST_RAWDATA_SKEWKURT_SF,
                'discont_hf': TEST_RAWDATA_DISCONT_HF,
                'discont_sf': TEST_RAWDATA_DISCONT_SF},
            'ssitc': {
                'apply': TEST_SSITC},
            'gas_completeness': {
                'apply': TEST_GAS_COMPLETENESS},
            'spectral_correction_factor': {
                'apply': TEST_SPECTRAL_CORRECTION_FACTOR},
            'angle_of_attack': {
                'apply': TEST_RAWDATA_ANGLE_OF_ATTACK,
                'application_dates': TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES},
            'steadiness_of_horizontal_wind': {
                'apply': TEST_RAWDATA_STEADINESS_OF_HORIZONTAL_WIND}
        }
        fpc.level2_quality_flag_expansion(**LEVEL2_SETTINGS)
        fpc.finalize_level2()
        from diive.pkgs.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsEddyPro
        self.assertEqual(type(fpc.level2), FluxQualityFlagsEddyPro)
        # fpc.level2_qcf.showplot_qcf_heatmaps()
        # fpc.level2_qcf.report_qcf_evolution()
        # fpc.level2_qcf.report_qcf_flags()
        # fpc.level2.results
        # fpc.fpc_df
        # fpc.filteredseries
        # [x for x in fpc.fpc_df.columns if 'L2' in x]
        # print(fpc.fpc_df)

        res = fpc.level2.results
        self.assertEqual(len(res.columns), 9)

        flagcols = []
        [flagcols.append(c) for c in res.columns if str(c).startswith("FLAG_") & str(c).endswith("_TEST")]
        self.assertEqual(len(flagcols), 8)
        self.assertEqual(res[flagcols].sum().sum(), 1797)

        self.assertEqual(df[FLUXVAR].dropna().count(), 811)
        self.assertEqual(fpc.filteredseries.dropna().count(), 778)
        self.assertEqual(fpc.filteredseries.name, f'{FLUXVAR}_L2_QCF')
        self.assertEqual(len(fpc.fpc_df.columns), 19)

        # --------------------
        # Level-3.1
        # --------------------
        fpc.level31_storage_correction(gapfill_storage_term=True)
        fpc.finalize_level31()
        from diive.pkgs.fluxprocessingchain.level31_storagecorrection import FluxStorageCorrectionSinglePointEddyPro
        self.assertEqual(type(fpc.level31), FluxStorageCorrectionSinglePointEddyPro)
        self.assertEqual(fpc.level31.gapfilled_strgcol, "SC_SINGLE_gfRF_L3.1")
        self.assertEqual(fpc.filteredseries.dropna().count(), 778)
        self.assertEqual(fpc.filteredseries.name, "NEE_L3.1_QCF")
        self.assertEqual(len(fpc.fpc_df.columns), 25)

        # fpc.level31.report()
        # fpc.level31.showplot(maxflux=50)
        # fpc.fpc_df
        # fpc.filteredseries
        # fpc.level31.results
        # [x for x in fpc.fpc_df.columns if 'L3.1' in x]

        # --------------------
        # Level-3.2
        # --------------------
        fpc.level32_stepwise_outlier_detection()
        kwargs = dict(showplot=False, verbose=False)
        fpc.level32_flag_outliers_abslim_dtnt_test(
            daytime_minmax=[-50, 50], nighttime_minmax=[-50, 50], **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_hampel_dtnt_test(
            window_length=48 * 3, n_sigma_dt=3.5, n_sigma_nt=3.5, repeat=False, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_manualremoval_test(
            remove_dates=[['2022-07-01 12:15:00', '2022-07-01 13:45:00']], **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_zscore_dtnt_test(
            thres_zscore=4, repeat=True, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_hampel_test(
            window_length=48 * 7, n_sigma=5, repeat=False, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_zscore_rolling_test(
            winsize=48 * 7, thres_zscore=5, repeat=True, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_localsd_test(
            n_sd=5, winsize=48 * 7, constant_sd=False, repeat=True, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_localsd_test(
            n_sd=3, winsize=48 * 7, constant_sd=True, repeat=True, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_increments_zcore_test(thres_zscore=5, repeat=True, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_lof_dtnt_test(n_neighbors=48 * 7, contamination=None, repeat=True, n_jobs=-1,
                                                **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_lof_test(n_neighbors=100, contamination=None, repeat=False, n_jobs=-1, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_zscore_test(thres_zscore=5, repeat=True, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_abslim_test(minval=-50, maxval=50, **kwargs)
        fpc.level32_addflag()
        fpc.level32_flag_outliers_trim_low_test(trim_nighttime=True, lower_limit=-3, **kwargs)
        fpc.level32_addflag()
        fpc.finalize_level32()
        from diive.pkgs.outlierdetection.stepwiseoutlierdetection import StepwiseOutlierDetection
        self.assertEqual(type(fpc.level32), StepwiseOutlierDetection)
        self.assertEqual(len(fpc.fpc_df.columns), 45)
        flagcols = [c for c in fpc.fpc_df.columns if str(c).startswith("FLAG_") and str(c).endswith("_TEST")]
        self.assertEqual(len(flagcols), 22)

        # --------------------
        # Level-3.3
        # --------------------
        ustar_scenarios = ['CUT_16', 'CUT_50', 'CUT_84']
        ustar_thresholds = [0.05, 0.07, 0.1]
        fpc.level33_constant_ustar(thresholds=ustar_thresholds,
                                   threshold_labels=ustar_scenarios,
                                   showplot=False)
        # Finalize: stores results for each USTAR scenario in a dict
        fpc.finalize_level33()
        from diive.pkgs.flux.ustarthreshold import FlagMultipleConstantUstarThresholds
        self.assertEqual(type(fpc.level33), FlagMultipleConstantUstarThresholds)
        self.assertEqual(len(fpc.fpc_df.columns), 66)
        flagcols = [c for c in fpc.fpc_df.columns if str(c).startswith("FLAG_") and str(c).endswith("_TEST")]
        self.assertEqual(len(flagcols), 25)

        # --------------------
        # Level-4.1
        # --------------------

        fpc.level41_gapfilling_longterm(
            random_forest=True,
            features=["TA_1_1_1", "SW_IN_1_1_1", "VPD_EP"],
            features_lag=[-1, -1],
            # reduce_features=False,
            include_timestamp_as_features=True,
            # add_continuous_record_number=True,
            verbose=True,
            perm_n_repeats=1,
            rf_kwargs={
                'n_estimators': 3,
                'random_state': 42,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'n_jobs': -1
            }
        )
        from diive.pkgs.gapfilling.longterm import LongTermGapFillingRandomForestTS
        self.assertEqual(type(fpc.level41['CUT_16']), LongTermGapFillingRandomForestTS)
        self.assertEqual(type(fpc.level41['CUT_50']), LongTermGapFillingRandomForestTS)
        self.assertEqual(type(fpc.level41['CUT_84']), LongTermGapFillingRandomForestTS)
        self.assertAlmostEqual(fpc.level41['CUT_50'].gapfilling_df_.sum().sum(), -630979.0954346551, places=5)
        self.assertEqual(len(fpc.fpc_df.columns), 87)
        flagcols = [c for c in fpc.fpc_df.columns if str(c).startswith("FLAG_") and str(c).endswith("_TEST")]
        self.assertEqual(len(flagcols), 25)



if __name__ == '__main__':
    unittest.main()
