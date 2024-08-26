import unittest


class TestFluxProcessingChain(unittest.TestCase):

    def test_fluxprocessingchain_level2(self):
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
        from diive.pkgs.fluxprocessingchain.fluxprocessingchain import FluxProcessingChain
        df, meta = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

        # Flux processing chain settings
        FLUXVAR = "FC"
        SITE_LAT = 46.583056  # CH-AWS
        SITE_LON = 9.790639  # CH-AWS
        UTC_OFFSET = 1
        NIGHTTIME_THRESHOLD = 50  # Threshold for potential radiation in W m-2, conditions below threshold are nighttime
        DAYTIME_ACCEPT_QCF_BELOW = 2
        NIGHTTIMETIME_ACCEPT_QCF_BELOW = 2

        fpc = FluxProcessingChain(
            maindf=df,
            fluxcol=FLUXVAR,
            site_lat=SITE_LAT,
            site_lon=SITE_LON,
            utc_offset=UTC_OFFSET
        )

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
        # TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = [['2023-07-01', '2023-09-01']]  # Default False
        TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = False  # Default False
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
        fpc.finalize_level2(nighttime_threshold=NIGHTTIME_THRESHOLD, daytime_accept_qcf_below=DAYTIME_ACCEPT_QCF_BELOW,
                            nighttimetime_accept_qcf_below=NIGHTTIMETIME_ACCEPT_QCF_BELOW)
        # fpc.level2_qcf.showplot_qcf_heatmaps()
        # fpc.level2_qcf.report_qcf_evolution()
        # fpc.level2_qcf.report_qcf_flags()
        # fpc.level2.results
        # fpc.fpc_df
        # fpc.filteredseries
        # [x for x in fpc.fpc_df.columns if 'L2' in x]

        print(fpc.fpc_df)

        res = fpc.level2.results
        self.assertEqual(len(res.columns), 9)

        flagcols = []
        [flagcols.append(c) for c in fpc.fpc_df.columns if str(c).startswith("FLAG_") & str(c).endswith("_TEST")]
        self.assertEqual(len(flagcols), 8)
        self.assertEqual(fpc.fpc_df[flagcols].sum().sum(), 1911)

        flagcols = []
        [flagcols.append(c) for c in res.columns if str(c).startswith("FLAG_") & str(c).endswith("_TEST")]
        self.assertEqual(len(flagcols), 8)
        self.assertEqual(res[flagcols].sum().sum(), 1911)

        self.assertEqual(df[FLUXVAR].dropna().count(), 811)
        self.assertEqual(fpc.filteredseries.dropna().count(), 748)
        self.assertEqual(fpc.filteredseries.name, f'{FLUXVAR}_L2_QCF')


if __name__ == '__main__':
    unittest.main()
