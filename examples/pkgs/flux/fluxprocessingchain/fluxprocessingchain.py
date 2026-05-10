import diive as dv

from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN


def _example():
    df = load_exampledata_parquet_lae_level1_30MIN()

    # Flux processing chain settings
    FLUXVAR = "FC"
    SITE_LAT = 47.41887  # CH-HON
    SITE_LON = 8.491318  # CH-HON
    UTC_OFFSET = 1
    NIGHTTIME_THRESHOLD = 20  # Threshold for potential radiation in W m-2, conditions below threshold are nighttime
    DAYTIME_ACCEPT_QCF_BELOW = 2
    NIGHTTIMETIME_ACCEPT_QCF_BELOW = 2

    # dv.plot_time_series(series=df[FLUXVAR]).plot()

    fpc = dv.flux_processing_chain(
        df=df,
        fluxcol=FLUXVAR,
        site_lat=SITE_LAT,
        site_lon=SITE_LON,
        utc_offset=UTC_OFFSET,
        nighttime_threshold=NIGHTTIME_THRESHOLD,
        daytime_accept_qcf_below=DAYTIME_ACCEPT_QCF_BELOW,
        nighttimetime_accept_qcf_below=NIGHTTIMETIME_ACCEPT_QCF_BELOW
    )

    # --------------------
    # Level-2
    # --------------------
    TEST_SSITC = True  # Default True
    TEST_SSITC_SETFLAG_TIMEPERIOD = None
    # TEST_SSITC_SETFLAG_TIMEPERIOD = {2: [[1, '2022-05-01', '2023-09-30']]}  # Set flag 1 to 2 (bad) during time period
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
    TEST_RAWDATA_ANGLE_OF_ATTACK = False  # Default False
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
            'apply': TEST_SSITC,
            'setflag_timeperiod': TEST_SSITC_SETFLAG_TIMEPERIOD},
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
    # fpc.level2_qcf.showplot_qcf_heatmaps()
    # fpc.level2_qcf.report_qcf_evolution()
    # fpc.level2_qcf.report_qcf_flags()
    # fpc.level2.results
    # fpc.fpc_df
    # fpc.filteredseries

    # --------------------
    # Level-3.1
    # --------------------
    fpc.level31_storage_correction(gapfill_storage_term=True, set_storage_to_zero=False)
    fpc.finalize_level31()
    # fpc.level31.report()
    # fpc.level31.showplot()
    # fpc.fpc_df
    # fpc.filteredseries
    # fpc.level31.results
    # -------------------------------------------------------------------------

    # --------------------
    # (OPTIONAL) ANALYZE
    # --------------------
    # fpc.analyze_highest_quality_flux(showplot=True)
    # -------------------------------------------------------------------------

    # --------------------
    # Level-3.2
    # --------------------
    fpc.level32_stepwise_outlier_detection()
    fpc.level32_flag_outliers_hampel_dtnt_test(
        window_length=48 * 13, n_sigma_dt=5.5, n_sigma_nt=5.5, showplot=False,
        verbose=True, use_differencing=True, separate_day_night=True, repeat=True)
    fpc.level32_addflag()
    fpc.finalize_level32()
    # fpc.filteredseries
    # fpc.level32.flags
    fpc.level32_qcf.showplot_qcf_heatmaps()
    # fpc.level32_qcf.showplot_qcf_timeseries()
    # fpc.level32_qcf.report_qcf_flags()
    # fpc.level32_qcf.report_qcf_evolution()
    # fpc.level32_qcf.report_qcf_series()
    # print("STOP")
    # -------------------------------------------------------------------------

    # --------------------
    # Level-3.3
    # --------------------
    ustar_scenarios = ['CUT_50']
    ustar_thresholds = [0.30]  # CH-LAE
    fpc.level33_constant_ustar(thresholds=ustar_thresholds,
                               threshold_labels=ustar_scenarios,
                               showplot=False)
    fpc.finalize_level33()
    # for ustar_scenario in ustar_scenarios:
    #     fpc.level33_qcf[ustar_scenario].showplot_qcf_heatmaps()
    #     fpc.level33_qcf[ustar_scenario].report_qcf_evolution()
    #     # fpc.filteredseries
    #     # fpc.level33
    #     # fpc.level33_qcf.showplot_qcf_timeseries()
    #     # fpc.level33_qcf.report_qcf_flags()
    #     # fpc.level33_qcf.report_qcf_series()
    #     # fpc.levelidstr
    #     # fpc.filteredseries_level2_qcf
    #     # fpc.filteredseries_level31_qcf
    #     # fpc.filteredseries_level32_qcf
    #     # fpc.filteredseries_level33_qcf

    # --------------------
    # Level-4.1
    # --------------------
    FEATURES = ["TA_T1_47_1_gfXG", "SW_IN_T1_47_1_gfXG", "VPD_T1_47_1_gfXG"]

    # Shared gap-filling parameters (RF and XGB)
    GAPFILLING_PARAMS = {
        'features': FEATURES,
        # Feature Engineering (8-stage pipeline)
        'features_lag': [-2, -1],
        'features_lag_stepsize': 1,
        'features_lag_exclude_cols': None,
        'features_rolling': [2, 4, 12, 24, 48],
        'features_rolling_exclude_cols': None,
        'features_rolling_stats': ['median', 'min', 'max', 'std', 'q25', 'q75'],
        'features_diff': [1, 2],
        'features_diff_exclude_cols': None,
        'features_ema': [6, 12, 24, 48],
        'features_ema_exclude_cols': None,
        'features_poly_degree': 2,
        'features_poly_exclude_cols': None,
        'features_stl': True,
        'features_stl_method': 'stl',
        'features_stl_seasonal_period': 48,
        'features_stl_exclude_cols': None,
        'features_stl_components': ['trend', 'seasonal', 'residual'],
        'vectorize_timestamps': True,
        'add_continuous_record_number': True,
        'sanitize_timestamp': True,
        'reduce_features': True,
        'verbose': True,
        'n_jobs': -1,
        'random_state': 42,
    }

    # --------------------
    # Level-4.1: Random Forest Gap-Filling
    # --------------------
    fpc.level41_longterm_random_forest(
        **GAPFILLING_PARAMS,
        # RF-specific hyperparameters
        n_estimators=2,
        max_depth=1,
        min_samples_split=5,
        min_samples_leaf=2,
    )
    # model = fpc.level41['long_term_random_forest']['CUT_50']
    # gapfilled = model.gapfilled_
    # scores = model.scores_  # R², MAE, RMSE
    # feature_importance = model.feature_importances_

    # --------------------
    # Level-4.1: XGBoost Gap-Filling
    # --------------------
    fpc.level41_longterm_xgboost(
        **GAPFILLING_PARAMS,
        # XGB-specific hyperparameters
        n_estimators=2,  # Demo setting
        max_depth=1,  # Demo setting
        learning_rate=0.05,  # Shrinkage parameter
        early_stopping_rounds=30,  # Stop if validation doesn't improve
        min_child_weight=5,
    )
    # model = fpc.level41['long_term_xgboost']['CUT_50']
    # gapfilled = model.gapfilled_
    # scores = model.scores_  # R², MAE, RMSE
    # feature_importance = model.feature_importances_
    #
    # # fpc.level41_mds(
    # #     swin="SW_IN_POT",
    # #     ta="TA_EP",
    # #     vpd="VPD_EP",
    # #     swin_tol=[20, 50],
    # #     ta_tol=2.5,
    # #     vpd_tol=0.5,
    # #     avg_min_n_vals=5
    # # )
    #
    # results = fpc.get_data()
    # # gapfilled_names = fpc.get_gapfilled_names()
    # # nongapfilled_names = fpc.get_nongapfilled_names()
    # # gapfilled_vars = fpc.get_gapfilled_variables()
    # fpc.report_gapfilling_variables()
    # fpc.report_gapfilling_model_scores()
    # fpc.report_traintest_model_scores()
    # fpc.report_traintest_details()
    # # fpc.report_gapfilling_feature_importances()
    #
    # # # Only ML models:
    # # fpc.report_gapfilling_poolyears()
    #
    # # todo get full data
    #
    fpc.showplot_gapfilled_heatmap(vmin=-30, vmax=30)
    fpc.showplot_gapfilled_cumulative(gain=0.02161926, units=r'($\mathrm{µmol\ CO_2\ m^{-2}}$)', per_year=True)
    # fpc.showplot_gapfilled_cumulative(gain=0.02161926, units=r'($\mathrm{g\ C\ m^{-2}}$)', per_year=False)
    #
    # from diive.core.plotting.dielcycle import DielCycle
    # series = results['NEE_L3.1_L3.3_CUT_50_QCF_gfXG'].copy()
    # # series = results['NEE_L3.1_L3.3_CUT_50_QCF_gfMDS'].copy()
    # dc = DielCycle(series=series)
    # title = r'$\mathrm{Mean\ CO_2\ flux\ (Feb 2024 - Mar 2026)}$'
    # units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
    # dc.plot(ax=None, title=title, txt_ylabel_units=units,
    #         each_month=True, legend_n_col=2)
    #
    # # # # Only ML models:
    # # fpc.showplot_feature_ranks_per_year()
    #
    # # # # Only MDS:
    # # fpc.showplot_mds_gapfilling_qualities()
    #
    # print("END")


def _example_orig():
    # Source data
    from pathlib import Path

    SOURCEDIR = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-hon_flux_product\dataset_ch-hon_flux_product\notebooks\0_data\OPENLAG-IRGA-Level-0_fluxnet_2024-2026.03"

    # # Search files and store filepaths in list
    # from diive.core.io.filereader import search_files, MultiDataFileReader
    # sourcefiles = search_files(searchdirs=SOURCEDIR, pattern='*_fluxnet_*.csv')
    # d = MultiDataFileReader(filepaths=sourcefiles,
    #                         filetype='EDDYPRO-FLUXNET-CSV-30MIN',
    #                         output_middle_timestamp=True)
    # df = d.data_df
    # from diive.core.io.files import save_parquet
    # filepath = save_parquet(filename="FLUXES_L0_ALL", data=df, outpath=SOURCEDIR)

    # Load from parquet
    from diive.core.io.files import load_parquet
    FILENAME = r"FLUXES_L0_ALL.parquet"
    FILEPATH = Path(SOURCEDIR) / FILENAME
    maindf = load_parquet(filepath=str(FILEPATH))

    # # Or load EddyPro _fluxnet_ output files
    # SOURCEDIRS = [r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-hon_flux_product\dataset_ch-hon_flux_product\notebooks\0_data\OPENLAG-IRGA-Level-0_fluxnet_2024-2026.03"]
    # ep = LoadEddyProOutputFiles(sourcedir=SOURCEDIRS, filetype='EDDYPRO-FLUXNET-CSV-30MIN')
    # ep.searchfiles()
    # ep.loadfiles()
    # maindf = ep.maindf
    # metadata = ep.metadata

    # # Restrict time range
    # locs = ((maindf.index.year >= 2023) & (maindf.index.year <= 2023)
    #         & (maindf.index.month >= 6) & (maindf.index.month <= 7))
    # maindf = maindf.loc[locs, :].copy()
    # # metadata = None
    # # print(maindf)

    # Restrict by wind direction (CH-HON)
    import numpy as np
    locs = (maindf['WD'] > 180) & (maindf['WD'] < 350)
    maindf.loc[locs, :] = np.nan

    # Flux processing chain settings
    FLUXVAR = "FC"
    SITE_LAT = 47.41887  # CH-HON
    SITE_LON = 8.491318  # CH-HON
    UTC_OFFSET = 1
    NIGHTTIME_THRESHOLD = 20  # Threshold for potential radiation in W m-2, conditions below threshold are nighttime
    DAYTIME_ACCEPT_QCF_BELOW = 1
    NIGHTTIMETIME_ACCEPT_QCF_BELOW = 1

    # from diive.core.dfun.stats import sstats  # Time series stats
    # sstats(maindf[FLUXVAR])
    # TimeSeries(series=level1_df[FLUXVAR]).plot()

    fpc = FluxProcessingChain(
        df=maindf,
        fluxcol=FLUXVAR,
        site_lat=SITE_LAT,
        site_lon=SITE_LON,
        utc_offset=UTC_OFFSET,
        nighttime_threshold=NIGHTTIME_THRESHOLD,
        daytime_accept_qcf_below=DAYTIME_ACCEPT_QCF_BELOW,
        nighttimetime_accept_qcf_below=NIGHTTIMETIME_ACCEPT_QCF_BELOW
    )

    # --------------------
    # Level-2
    # --------------------
    TEST_SSITC = True  # Default True
    TEST_SSITC_SETFLAG_TIMEPERIOD = None
    # TEST_SSITC_SETFLAG_TIMEPERIOD = {2: [[1, '2022-05-01', '2023-09-30']]}  # Set flag 1 to 2 (bad) during time period
    TEST_GAS_COMPLETENESS = False  # Default True
    TEST_SPECTRAL_CORRECTION_FACTOR = True  # Default True
    TEST_SIGNAL_STRENGTH = True
    TEST_SIGNAL_STRENGTH_COL = 'CUSTOM_AGC_MEAN'
    TEST_SIGNAL_STRENGTH_METHOD = 'discard above'
    TEST_SIGNAL_STRENGTH_THRESHOLD = 90
    # TimeSeries(series=maindf[TEST_SIGNAL_STRENGTH_COL]).plot()
    TEST_RAWDATA = True  # Default True
    TEST_RAWDATA_SPIKES = True  # Default True
    TEST_RAWDATA_AMPLITUDE = False  # Default True
    TEST_RAWDATA_DROPOUT = True  # Default True
    TEST_RAWDATA_ABSLIM = False  # Default False
    TEST_RAWDATA_SKEWKURT_HF = False  # Default False
    TEST_RAWDATA_SKEWKURT_SF = False  # Default False
    TEST_RAWDATA_DISCONT_HF = False  # Default False
    TEST_RAWDATA_DISCONT_SF = False  # Default False
    TEST_RAWDATA_ANGLE_OF_ATTACK = False  # Default False
    TEST_RAWDATA_ANGLE_OF_ATTACK_APPLICATION_DATES = [['2008-01-01', '2010-01-01'],
                                                      ['2016-03-01', '2016-05-01']]  # Default False
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
            'apply': TEST_SSITC,
            'setflag_timeperiod': TEST_SSITC_SETFLAG_TIMEPERIOD},
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
    # fpc.level2_qcf.showplot_qcf_heatmaps()
    fpc.level2_qcf.report_qcf_evolution()
    # fpc.level2_qcf.analyze_highest_quality_flux()
    # fpc.level2_qcf.report_qcf_flags()
    # fpc.level2.results
    # fpc.fpc_df
    # fpc.filteredseries
    # [x for x in fpc.fpc_df.columns if 'L2' in x]

    # --------------------
    # Level-3.1
    # --------------------
    fpc.level31_storage_correction(gapfill_storage_term=False, set_storage_to_zero=False)
    # fpc.level31_storage_correction(gapfill_storage_term=False)
    fpc.finalize_level31()
    # fpc.level31.report()
    # fpc.level31.showplot()
    # fpc.fpc_df
    # fpc.filteredseries
    # fpc.level31.results
    # [x for x in fpc.fpc_df.columns if 'L3.1' in x]
    # -------------------------------------------------------------------------

    # # --------------------
    # # (OPTIONAL) ANALYZE
    # # --------------------
    # fpc.analyze_highest_quality_flux(showplot=True)
    # # -------------------------------------------------------------------------

    # --------------------
    # Level-3.2
    # --------------------
    fpc.level32_stepwise_outlier_detection()

    # fpc.level32_flag_manualremoval_test(
    #     remove_dates=[
    #         ['2016-03-18 12:15:00', '2016-05-03 06:45:00'],
    #         # '2022-12-12 12:45:00',
    #     ],
    #     showplot=False, verbose=True)
    # fpc.level32_addflag()

    # DAYTIME_MINMAX = [-50, 50]
    # NIGHTTIME_MINMAX = [-50, 50]
    # fpc.level32_flag_outliers_abslim_dtnt_test(daytime_minmax=DAYTIME_MINMAX, nighttime_minmax=NIGHTTIME_MINMAX,
    #                                            showplot=False, verbose=False)
    # # fpc.level32_flag_outliers_abslim_dtnt_test(daytime_minmax=DAYTIME_MINMAX, nighttime_minmax=NIGHTTIME_MINMAX, showplot=True, verbose=True)
    # fpc.level32_addflag()
    # # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_zscore_dtnt_test(thres_zscore=4, showplot=True, verbose=False, repeat=True)
    # fpc.level32_addflag()

    # # fpc.level32_flag_outliers_hampel_dtnt_test(window_length=48 * 3, n_sigma_dt=3.5, n_sigma_nt=3.5, showplot=False, verbose=False, repeat=False)
    fpc.level32_flag_outliers_hampel_dtnt_test(window_length=48 * 13, n_sigma_dt=5.5, n_sigma_nt=5.5, showplot=False,
                                               verbose=True, use_differencing=True, separate_day_night=True,
                                               repeat=True)
    fpc.level32_addflag()

    # fpc.level32_flag_outliers_zscore_rolling_test(winsize=48 * 7, thres_zscore=4, showplot=False, verbose=True,
    #                                               repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_localsd_test(n_sd=5, winsize=48 * 13, constant_sd=False, showplot=False, verbose=True,
    #                                        repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_localsd_test(n_sd=4, winsize=48 * 13, constant_sd=True, showplot=False, verbose=True, repeat=True)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_localsd_test(n_sd=[1.2, 99], winsize=[48 * 13, 48 * 2], constant_sd=False,
    #                                        separate_daytime_nighttime=True, lat=SITE_LAT, lon=SITE_LON, utc_offset=1,
    #                                        showplot=True, verbose=True, repeat=False)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_increments_zcore_test(thres_zscore=4, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()
    # fpc.level32.showplot_cleaned()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_lof_dtnt_test(n_neighbors=48 * 5, contamination=None, showplot=True, verbose=True, repeat=True, n_jobs=-1)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_lof_test(n_neighbors=20, contamination=None, showplot=True, verbose=True,
    #                                    repeat=False, n_jobs=-1)
    # fpc.level32_addflag()

    # fpc.level32_flag_outliers_zscore_test(thres_zscore=3, showplot=True, verbose=True, repeat=True)
    # fpc.level32_addflag()
    # fpc.level32.results

    # fpc.level32_flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False, verbose=True)
    # fpc.level32_addflag()
    # fpc.level32.results  # Stores Level-3.2 flags up to this point

    # fpc.level32_flag_outliers_trim_low_test(trim_nighttime=True, lower_limit=-10, showplot=True, verbose=True)
    # fpc.level32_addflag()

    fpc.finalize_level32()

    # # fpc.filteredseries
    # # fpc.level32.flags
    # fpc.level32_qcf.showplot_qcf_heatmaps()
    # # fpc.level32_qcf.showplot_qcf_timeseries()
    # # fpc.level32_qcf.report_qcf_flags()
    # fpc.level32_qcf.report_qcf_evolution()
    # # fpc.level32_qcf.report_qcf_series()
    # print("STOP")
    # -------------------------------------------------------------------------

    # --------------------
    # Level-3.3
    # --------------------
    # 0.052945, 0.069898, 0.092841
    # ustar_scenarios = ['NO_USTAR']
    # ustar_thresholds = [-9999]
    # ustar_scenarios = ['CUT_50']
    # ustar_thresholds = [0.069898]
    ustar_scenarios = ['CUT_50']
    ustar_thresholds = [0.09]  # CH-LAE
    # ustar_scenarios = ['CUT_16', 'CUT_50', 'CUT_84']
    # ustar_thresholds = [0.271619922, 0.303628125, 0.339684084]  # CH-LAE
    # ustar_scenarios = ['CUT_16', 'CUT_50', 'CUT_84']
    # ustar_thresholds = [0.052945, 0.069898, 0.092841]
    fpc.level33_constant_ustar(thresholds=ustar_thresholds,
                               threshold_labels=ustar_scenarios,
                               showplot=False)
    # Finalize: stores results for each USTAR scenario in a dict
    fpc.finalize_level33()

    # # Save current instance to pickle for faster testing
    # from diive.core.io.files import save_as_pickle
    # save_as_pickle(outpath=r"F:\TMP", filename="test", data=fpc)
    # fpc = load_pickle(filepath=r"F:\TMP\test.pickle")

    for ustar_scenario in ustar_scenarios:
        # fpc.level33_qcf[ustar_scenario].showplot_qcf_heatmaps()
        fpc.level33_qcf[ustar_scenario].report_qcf_evolution()
        # fpc.filteredseries
        # fpc.level33
        # fpc.level33_qcf.showplot_qcf_timeseries()
        # fpc.level33_qcf.report_qcf_flags()
        # fpc.level33_qcf.report_qcf_series()
        # fpc.levelidstr
        # fpc.filteredseries_level2_qcf
        # fpc.filteredseries_level31_qcf
        # fpc.filteredseries_level32_qcf
        # fpc.filteredseries_level33_qcf

    # --------------------
    # Level-4.1
    # --------------------
    # FEATURES = ["TA_T1_47_1_gfXG", "SW_IN_T1_47_1_gfXG", "VPD_T1_47_1_gfXG"]
    FEATURES = ["TA_EP", "SW_IN_POT", "VPD_EP"]

    # # --------------------
    # # Level-4.1: Random Forest Gap-Filling
    # # --------------------
    #
    # fpc.level41_longterm_random_forest(
    #     features=FEATURES,
    #
    #     # ===== FEATURE ENGINEERING PARAMETERS (Identical to XGBoost) =====
    #
    #     # Stage 1: LAG FEATURES (Immediate past context)
    #     # For CO2: Short lags (1-2 steps = 30-60 min) as flux responds quickly to environment
    #     features_lag=[-2, -1],          # Past 30-60 min only (no future, no current)
    #     features_lag_stepsize=1,        # Include every lag (dense temporal context)
    #     features_lag_exclude_cols=None, # Lag all input features
    #
    #     # Stage 2: ROLLING STATISTICS (Diurnal pattern capture)
    #     # For CO2: 30-min, 1-hr, 2-hr, 6-hr, 12-hr, 24-hr windows to capture diurnal patterns
    #     # Window sizes for 30-min data: 2=1hr, 4=2hr, 12=6hr, 24=12hr, 48=24hr
    #     features_rolling=[2, 4, 12, 24, 48],  # 1hr, 2hr, 6hr, 12hr, 24hr windows
    #     features_rolling_exclude_cols=None,   # Apply to all input features
    #     # Advanced rolling stats: Median robust to outliers, min/max/quantiles capture asymmetry
    #     features_rolling_stats=['median', 'min', 'max', 'std', 'q25', 'q75'],
    #
    #     # Stage 3: TEMPORAL DIFFERENCING (Rate of change, flux transitions)
    #     # For CO2: Order-1 (rate) captures sunrise/sunset transitions and weather events
    #     # Order-2 (acceleration) helps detect rapid state changes
    #     features_diff=[1, 2],           # First and second-order differencing
    #     features_diff_exclude_cols=None,
    #
    #     # Stage 4: EXPONENTIAL MOVING AVERAGE (Multi-timescale memory)
    #     # For CO2: Captures stomatal/photosynthetic adjustment at multiple timescales
    #     # Spans for 30-min data: 6=3hr, 12=6hr, 24=12hr, 48=24hr
    #     features_ema=[6, 12, 24, 48],   # 3hr, 6hr, 12hr, 24hr exponential moving averages
    #     features_ema_exclude_cols=None,
    #
    #     # Stage 5: POLYNOMIAL EXPANSION (Non-linear relationships)
    #     # For CO2: Degree-2 essential for light saturation (Michaelis-Menten curve)
    #     # Captures photosynthetic saturation and respiratory asymmetry
    #     features_poly_degree=2,         # Quadratic terms (e.g., Tair², Rg² for saturation)
    #     features_poly_exclude_cols=None,
    #
    #     # Stage 6: STL DECOMPOSITION (Trend/Seasonal separation)
    #     # For CO2: CRITICAL - separates respiration trend from photosynthetic pattern
    #     # Daily cycle: photosynthesis (daytime negative NEE), respiration (nighttime positive)
    #     # Seasonal cycle: dormancy (winter respiration), growth (summer photosynthesis)
    #     features_stl=True,                      # Enable STL decomposition
    #     features_stl_method='stl',              # Robust LOESS method (handles gaps)
    #     features_stl_seasonal_period=48,        # 30-min × 48 = 24 hours (daily cycle)
    #     features_stl_exclude_cols=None,         # Apply to all input features
    #     features_stl_components=['trend', 'seasonal', 'residual'],  # Extract all
    #
    #     # Stage 7: TIMESTAMP FEATURES (Diurnal/Seasonal cycles)
    #     # For CO2: ESSENTIAL - photosynthesis depends on time-of-day (solar elevation)
    #     # and season (leaf phenology, dormancy)
    #     vectorize_timestamps=True,      # Creates ~19 features: year, season, DOY, hour, etc.
    #
    #     # Stage 8: SEQUENTIAL RECORD NUMBER (Long-term drift)
    #     # For CO2: Useful if site shows long-term drift (instrument aging, vegetation change)
    #     add_continuous_record_number=True,  # 1, 2, 3, ... for drift capture
    #
    #     # Data quality preprocessing
    #     sanitize_timestamp=True,        # Validate timestamps (catch gaps/duplicates)
    #
    #     # ===== GAP-FILLING PARAMETERS =====
    #     reduce_features=True,          # ENABLED: Apply SHAP-based feature selection
    #                                     # Selects only important features across all years
    #                                     # Reduces feature count from ~45-50 to ~10-20 features
    #                                     # Benefits: Faster training, better generalization, smaller models
    #                                     # Drawback: Removes potentially useful features
    #     verbose=True,                   # Print progress and model scores
    #
    #     # ===== RANDOM FOREST HYPERPARAMETERS =====
    #     # Tuned for flux data (non-linear, heteroscedastic, with clear diurnal cycle)
    #     # RF typically needs more estimators than XGBoost to reach same performance
    #
    #     n_estimators=350,               # 350 trees (more than XGBoost due to bagging)
    #                                     # Random Forest needs ~50% more trees than XGBoost
    #                                     # Increase (400-500) if underfitting (R² too low)
    #                                     # Decrease (200-300) if overfitting or memory-limited
    #                                     # Training time ~linear with n_estimators
    #
    #     max_depth=15,                   # Tree depth (deeper than XGBoost)
    #                                     # RF can use deeper trees without overfitting
    #                                     # Default 15 balances complexity and stability
    #                                     # Increase (20-25) for complex patterns
    #                                     # Decrease (8-10) if overfitting
    #
    #     min_samples_split=5,            # Minimum samples required to split a node
    #                                     # Higher values prevent overfitting to individual data points
    #                                     # 5 = good balance for 30-min flux data (~hours of training data)
    #                                     # Increase (10-15) if overfitting to noise
    #                                     # Decrease (2-3) if underfitting
    #
    #     min_samples_leaf=2,             # Minimum samples required at leaf node
    #                                     # Higher = smoother predictions, less overfitting
    #                                     # 2 = permissive, allows feature detection
    #                                     # Increase (5-10) if overfitting
    #                                     # Decrease (1) if underfitting
    #
    #     n_jobs=-1,                      # Use all CPU cores (parallel tree building)
    #     random_state=42,                # Reproducibility (same results every run)
    # )
    # # ===== ACCESS RESULTS =====
    # # model = fpc.level41['long_term_random_forest']['CUT_50']
    # # gapfilled_co2 = model.gapfilled_
    # # scores = model.scores_  # R², MAE, RMSE on test data
    # # feature_importance = model.feature_importances_  # SHAP importance per feature
    # # yearly_models = model.results_yearly_  # Per-year model results (dict keyed by year)
    # #
    # # ===== COMPARE RF VS XGBOOST =====
    # # rf_r2 = fpc.level41['long_term_random_forest']['CUT_50'].scores_['r2']
    # # xgb_r2 = fpc.level41['long_term_xgboost']['CUT_50'].scores_['r2']
    # # print(f"Random Forest R²: {rf_r2:.3f}")
    # # print(f"XGBoost R²: {xgb_r2:.3f}")
    # # print(f"Winner: {'XGBoost' if xgb_r2 > rf_r2 else 'Random Forest'}")

    # ---------------------------
    # XGBOOST WITH COMPREHENSIVE CO2 FLUX CONFIGURATION
    # ---------------------------
    # Level-4.1 XGBoost Gap-Filling optimized for CO2 half-hourly flux (NEE) data
    # This configuration balances capture of diurnal photosynthetic patterns with
    # computational efficiency. XGBoost often outperforms Random Forest on non-linear
    # flux responses (light saturation, stomatal conductance).
    #
    # KEY TUNING FOR CO2 FLUX (30-min resolution):
    # - Lag features: Short windows (1-2 steps = 30-60 min) for fast response
    # - Rolling windows: 3-24 hours (6-48 steps) for diurnal pattern context
    # - Differencing: Order-1 for rate-of-change, helps detect flux transitions
    # - EMA: Multi-timescale (3-24 hours) captures memory effects from stomata/photosystem
    # - STL: CRITICAL for CO2 - strong daily cycle + seasonal dormancy/growth
    # - Timestamps: ESSENTIAL - diurnal photosynthesis is time-of-day dependent
    # - Polynomial: Captures non-linear light saturation (Michaelis-Menten kinetics)
    #
    # Feature count: ~45-50 engineered features
    # Typical training time: 2-5 min per year per USTAR scenario
    # Expected model R²: 0.65-0.85 depending on site complexity

    fpc.level41_longterm_xgboost(
        features=FEATURES,

        # ===== FEATURE ENGINEERING PARAMETERS =====

        # Stage 1: LAG FEATURES (Immediate past context)
        # For CO2: Short lags (1-2 steps = 30-60 min) as flux responds quickly to environment
        features_lag=[-2, -1],  # Past 30-60 min only (no future, no current)
        features_lag_stepsize=1,  # Include every lag (dense temporal context)
        features_lag_exclude_cols=None,  # Lag all input features

        # Stage 2: ROLLING STATISTICS (Diurnal pattern capture)
        # For CO2: 30-min, 1-hr, 2-hr, 6-hr, 12-hr, 24-hr windows to capture diurnal patterns
        # Window sizes for 30-min data: 2=1hr, 4=2hr, 12=6hr, 24=12hr, 48=24hr
        features_rolling=[2, 4, 12, 24, 48],  # 1hr, 2hr, 6hr, 12hr, 24hr windows
        features_rolling_exclude_cols=None,  # Apply to all input features
        # Advanced rolling stats: Median robust to outliers, min/max/quantiles capture asymmetry
        features_rolling_stats=['median', 'min', 'max', 'std', 'q25', 'q75'],

        # Stage 3: TEMPORAL DIFFERENCING (Rate of change, flux transitions)
        # For CO2: Order-1 (rate) captures sunrise/sunset transitions and weather events
        # Order-2 (acceleration) helps detect rapid state changes
        features_diff=[1, 2],  # First and second-order differencing
        features_diff_exclude_cols=None,

        # Stage 4: EXPONENTIAL MOVING AVERAGE (Multi-timescale memory)
        # For CO2: Captures stomatal/photosynthetic adjustment at multiple timescales
        # Spans for 30-min data: 6=3hr, 12=6hr, 24=12hr, 48=24hr
        features_ema=[6, 12, 24, 48],  # 3hr, 6hr, 12hr, 24hr exponential moving averages
        features_ema_exclude_cols=None,

        # Stage 5: POLYNOMIAL EXPANSION (Non-linear relationships)
        # For CO2: Degree-2 essential for light saturation (Michaelis-Menten curve)
        # Captures photosynthetic saturation and respiratory asymmetry
        features_poly_degree=2,  # Quadratic terms (e.g., Tair², Rg² for saturation)
        features_poly_exclude_cols=None,

        # Stage 6: STL DECOMPOSITION (Trend/Seasonal separation)
        # For CO2: CRITICAL - separates respiration trend from photosynthetic pattern
        # Daily cycle: photosynthesis (daytime negative NEE), respiration (nighttime positive)
        # Seasonal cycle: dormancy (winter respiration), growth (summer photosynthesis)
        features_stl=True,  # Enable STL decomposition
        features_stl_method='stl',  # Robust LOESS method (handles gaps)
        features_stl_seasonal_period=48,  # 30-min × 48 = 24 hours (daily cycle)
        features_stl_exclude_cols=None,  # Apply to all input features
        features_stl_components=['trend', 'seasonal', 'residual'],  # Extract all

        # Stage 7: TIMESTAMP FEATURES (Diurnal/Seasonal cycles)
        # For CO2: ESSENTIAL - photosynthesis depends on time-of-day (solar elevation)
        # and season (leaf phenology, dormancy)
        vectorize_timestamps=True,  # Creates ~19 features: year, season, DOY, hour, etc.

        # Stage 8: SEQUENTIAL RECORD NUMBER (Long-term drift)
        # For CO2: Useful if site shows long-term drift (instrument aging, vegetation change)
        add_continuous_record_number=True,  # 1, 2, 3, ... for drift capture

        # Data quality preprocessing
        sanitize_timestamp=True,  # Validate timestamps (catch gaps/duplicates)

        # ===== GAP-FILLING PARAMETERS =====
        reduce_features=True,  # ENABLED: Apply SHAP-based feature selection
        # Selects only important features across all years
        # Reduces feature count from ~45-50 to ~10-20 features
        # Benefits: Faster training, better generalization, smaller models
        # Drawback: Removes potentially useful features
        verbose=True,  # Print progress and model scores

        # ===== XGBOOST HYPERPARAMETERS =====
        # Demo settings (similar to Random Forest above for comparison)

        n_estimators=2,  # Boosting rounds (demo setting)
        max_depth=1,  # Tree depth (demo setting)
        learning_rate=0.05,  # Shrinkage parameter
        early_stopping_rounds=30,  # Stop if validation doesn't improve
        n_jobs=-1,  # Use all CPU cores
        random_state=42,  # Reproducibility
        min_child_weight=5,
    )
    # ===== ACCESS RESULTS =====
    # model = fpc.level41['long_term_xgboost']['CUT_50']
    # gapfilled_co2 = model.gapfilled_
    # scores = model.scores_  # R², MAE, RMSE on test data
    # feature_importance = model.feature_importances_  # SHAP importance per feature
    # yearly_models = model.results_yearly_  # Per-year model results (dict keyed by year)

    # fpc.level41_mds(
    #     swin="SW_IN_POT",
    #     ta="TA_EP",
    #     vpd="VPD_EP",
    #     swin_tol=[20, 50],
    #     ta_tol=2.5,
    #     vpd_tol=0.5,
    #     avg_min_n_vals=5
    # )

    results = fpc.get_data()
    # gapfilled_names = fpc.get_gapfilled_names()
    # nongapfilled_names = fpc.get_nongapfilled_names()
    # gapfilled_vars = fpc.get_gapfilled_variables()
    fpc.report_gapfilling_variables()
    fpc.report_gapfilling_model_scores()
    fpc.report_traintest_model_scores()
    fpc.report_traintest_details()
    # fpc.report_gapfilling_feature_importances()

    # # Only ML models:
    # fpc.report_gapfilling_poolyears()

    # todo get full data

    fpc.showplot_gapfilled_heatmap(vmin=-30, vmax=30)
    fpc.showplot_gapfilled_cumulative(gain=0.02161926, units=r'($\mathrm{µmol\ CO_2\ m^{-2}}$)', per_year=True)
    fpc.showplot_gapfilled_cumulative(gain=0.02161926, units=r'($\mathrm{g\ C\ m^{-2}}$)', per_year=False)

    from diive.core.plotting.dielcycle import DielCycle
    series = results['NEE_L3.1_L3.3_CUT_50_QCF_gfXG'].copy()
    # series = results['NEE_L3.1_L3.3_CUT_50_QCF_gfMDS'].copy()
    dc = DielCycle(series=series)
    title = r'$\mathrm{Mean\ CO_2\ flux\ (Feb 2024 - Mar 2026)}$'
    units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
    dc.plot(ax=None, title=title, txt_ylabel_units=units,
            each_month=True, legend_n_col=2)

    # # # Only ML models:
    # fpc.showplot_feature_ranks_per_year()

    # # # Only MDS:
    # fpc.showplot_mds_gapfilling_qualities()

    print("END")


if __name__ == '__main__':
    _example()
