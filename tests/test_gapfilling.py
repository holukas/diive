import unittest

import numpy as np
from numpy import mean

import diive.configs.exampledata as ed
from diive.core.ml.common import MlRegressorGapFillingBase
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.gapfilling.randomforest_ts import RandomForestTS
from diive.gapfilling.xgboost_ts import XGBoostTS


class TestGapFilling(unittest.TestCase):

    def test_optimize_rf_params(self):
        pass

    def test_quickfill(self):
        """Test QuickFillRFTS for rapid gap-filling exploration"""
        from diive.gapfilling.randomforest_ts import QuickFillRFTS
        df = ed.load_exampledata_parquet()
        df = df.loc[(df.index.year == 2020) & (df.index.month == 7)].copy()

        TARGET_COL = 'NEE_CUT_REF_orig'
        subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

        # Subset with target and features
        lowquality = df["QCF_NEE"] > 0
        df.loc[lowquality, TARGET_COL] = np.nan
        df = df[subsetcols].copy()

        # QuickFillRFTS uses minimal internal feature engineering (pre-configured for speed)
        # Only takes df and target_col, all other parameters are hardcoded for fast exploration
        qfill = QuickFillRFTS(df=df, target_col=TARGET_COL)

        # Train and fill gaps with minimal parameters (for speed)
        qfill.fill()

        # Verify quick fill works
        gapfilled = qfill.get_gapfilled_target()
        self.assertGreater(len(gapfilled), 0)
        self.assertLess(gapfilled.isnull().sum(), len(gapfilled))  # Some gaps filled
        self.assertEqual(gapfilled.isnull().sum(), 0)  # Some gaps remain (only filled where possible)

    def test_fluxmds(self):
        from collections import Counter
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.gapfilling.mds import FluxMDS
        df = load_exampledata_parquet()
        locs = (df.index.year >= 2022) & (df.index.year <= 2022)
        df = df.loc[locs].copy()
        locs = (df.index.month >= 7) & (df.index.month <= 7)
        df = df.loc[locs].copy()

        mds = FluxMDS(
            df=df,
            flux='NEE_CUT_REF_orig',
            ta='Tair_f',
            swin='Rg_f',
            vpd='VPD_f',
            # swin_tol=[20, 50],
            # ta_tol=2.5,
            # vpd_tol=0.5,  # kPa; 5 hPa is default for reference
            # avg_min_n_vals=5
        )
        mds.run()
        # mds.report()
        # mds.showplot()

        results = mds.gapfilling_df_
        self.assertEqual(len(results), 1488)
        self.assertEqual(mds.scores_['r2'], 0.720601595663773)
        self.assertAlmostEqual(mds.scores_['medae'], 1.70347175141243, places=10)
        self.assertEqual(results[mds.target_gapfilled].isnull().sum(), 0)
        # Faithful ONEFlux quality (1/2/3) averaged over the gap predictions.
        self.assertAlmostEqual(mds.scores_['mean_quality_flag_gap_predictions'],
                               1.1285714285714286, places=10)
        flag = results[mds.target_gapfilled_flag]
        counts = Counter(flag.dropna().astype(int))
        # Missing in measured as indicated by flag > 0
        a = flag[flag > 0].count()
        # Missing in measured
        b = results[mds.flux].isnull().sum()
        self.assertEqual(a, 770)
        self.assertEqual(b, 770)
        # Flag is granular: 0 = measured, else method*1000 + time_window (days).
        self.assertEqual(counts[0], 718)       # measured
        self.assertEqual(counts[1014], 515)    # method 1 (SWIN+TA+VPD), 14 d
        self.assertEqual(counts[1028], 99)     # method 1, 28 d
        self.assertEqual(counts[2014], 156)    # method 2 (SWIN only), 14 d

    def test_gapfilling_longterm_randomforest(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.gapfilling.longterm import LongTermGapFillingRandomForestTS
        TARGET_COL = 'NEE_CUT_REF_orig'
        subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']
        source_df = load_exampledata_parquet()
        df = source_df.copy()
        locs = (df.index.year >= 2017) & (df.index.year <= 2018)
        df = df.loc[locs].copy()
        # This example uses NEE flux data, only records where the quality flag QCF is 0 (highest quality) are retained
        lowquality = df["QCF_NEE"] > 0
        df.loc[lowquality, TARGET_COL] = np.nan  # Set fluxes of lower quality to missing
        df = df[subsetcols].copy()  # Keep subset columns only

        # Step 1: Engineer features
        engineer = FeatureEngineer(
            target_col=TARGET_COL,
            features_lag=[-1, -1],
            features_lag_stepsize=1,
            features_lag_exclude_cols=None,
            features_rolling=None,
            features_rolling_exclude_cols=None,
            features_rolling_stats=None,
            features_diff=None,
            features_diff_exclude_cols=None,
            features_poly_degree=None,
            features_poly_exclude_cols=None,
            vectorize_timestamps=False,
            add_continuous_record_number=False,
            sanitize_timestamp=False
        )
        df_engineered = engineer.fit_transform(df)

        # Step 2: Create long-term gap-filling model with engineered features
        gf = LongTermGapFillingRandomForestTS(
            input_df=df_engineered,
            target_col=TARGET_COL,
            verbose=1,
            test_size=0.25,
            n_estimators=3,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        gf.create_yearpools()
        gf.initialize_yearly_models()
        gf.reduce_features_across_years()
        gf.fillgaps()

        results = gf.gapfilling_df_
        # Verify results structure
        self.assertGreater(len(results), 0)
        self.assertEqual(len(results.columns), 8)
        self.assertEqual(results[gf.gapfilled_.name].sum(), gf.gapfilled_.sum())
        self.assertEqual(results[gf.gapfilled_.name].isnull().sum(), 0)
        self.assertGreater(results[TARGET_COL].isnull().sum(), 0)
        self.assertGreater(results['FLAG_NEE_CUT_REF_orig_gfRF_ISFILLED'].sum(), 0)

        # Verify feature reduction and yearly models
        self.assertGreater(len(gf.features_reduced_across_years), 0)
        self.assertEqual(gf.feature_ranks_per_year.min().min(), 1)
        self.assertGreater(gf.feature_ranks_per_year.max().max(), 0)
        self.assertEqual(len(gf.feature_importances_yearly_.keys()), 2)

        # Verify SHAP importance exists for all years
        for year in gf.feature_importances_yearly_.keys():
            self.assertIn('SHAP_IMPORTANCE', gf.feature_importances_yearly_[year].columns)

        # Verify scores exist and are reasonable
        scores = []
        r2s = []
        for year, s in gf.scores_.items():
            scores.append(s['mae'])
            r2s.append(s['r2'])
        self.assertEqual(len(scores), 2)
        self.assertGreater(mean(scores), 0)
        self.assertLess(mean(r2s), 1.0)
        self.assertGreater(mean(r2s), 0)

        # Verify yearly results exist
        self.assertEqual(type(gf.results_yearly_['2017']), MlRegressorGapFillingBase)
        self.assertGreater(gf.results_yearly_['2017'].scores_['rmse'], 0)

    def test_linear_interpolation(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.gapfilling.interpolate import linear_interpolation
        df = load_exampledata_parquet()
        df = df.loc[df.index.year == 2022].copy()
        series = df['NEE_CUT_REF_orig'].copy()
        series_gapfilled = linear_interpolation(series=series, limit=10)
        self.assertEqual(series_gapfilled.isnull().sum(), 7856)
        self.assertEqual(series.isnull().sum(), 11412)

    def test_observed_preserved_when_feature_missing(self):
        """Observed targets must never be overwritten/mis-flagged when a feature
        is missing at that row (driver gap not aligned with the target gap)."""
        import pandas as pd
        idx = pd.date_range('2022-07-01', periods=400, freq='30min', name='TIMESTAMP_END')
        rng = np.random.RandomState(1)
        f1 = rng.normal(size=400).astype(float)
        target = (3 * f1 + rng.normal(scale=0.3, size=400)).astype(float)
        df = pd.DataFrame({'target': target, 'f1': f1}, index=idx)
        # Driver-only gaps at rows where the target IS observed, plus real target gaps.
        driver_gap_rows = [100, 150, 200, 250]
        df.loc[df.index[driver_gap_rows], 'f1'] = np.nan
        df.loc[df.index[[120, 170]], 'target'] = np.nan
        observed = df['target'].copy()

        rf = RandomForestTS(input_df=df, target_col='target', n_estimators=30, verbose=0)
        rf.run(showplot_scores=False, showplot_importance=False)
        gf, flag = rf.results.gapfilled, rf.results.flag

        obs_mask = observed.notna()
        # Every observed value is preserved exactly and flagged 0 (observed)...
        self.assertTrue(np.allclose(gf[obs_mask], observed[obs_mask]))
        self.assertTrue((flag[obs_mask] == 0).all())
        # ...including the rows where the driver was missing.
        self.assertTrue(np.allclose(gf.iloc[driver_gap_rows], observed.iloc[driver_gap_rows]))
        self.assertTrue((flag.iloc[driver_gap_rows] == 0).all())
        # The gap-filled series is complete (no remaining gaps).
        self.assertEqual(int(gf.isna().sum()), 0)

    def test_swin_gapfiller(self):
        """SW_IN gap-filling with the physical nighttime constraint."""
        import pandas as pd
        from diive.gapfilling.swin import SWINGapFillerXGBoost
        from diive.variables import potrad_oneflux
        lat, lon, utc = 47.0, 8.0, 1
        idx = pd.date_range('2022-06-01', '2022-06-30 23:30', freq='30min', name='TIMESTAMP_END')
        pot = potrad_oneflux(timestamp_index=idx, lat=lat, lon=lon, utc_offset=utc)
        rng = np.random.RandomState(0)
        swin = (pot * (0.7 + 0.3 * rng.rand(len(idx)))).clip(lower=0)  # cloudy modulation
        swin.name = 'SW_IN'
        swin_gappy = swin.copy()
        swin_gappy[rng.rand(len(idx)) < 0.3] = np.nan  # punch ~30% gaps

        # Seed both the train/test split and the regressor: without random_state
        # the split is redrawn every run and the scores below drift.
        g = SWINGapFillerXGBoost(series=swin_gappy, lat=lat, lon=lon, utc_offset=utc,
                                 random_state=42, verbose=0)
        g.run()
        gf = g.results.gapfilled
        self.assertEqual(int(gf.isna().sum()), 0)   # complete after gap-filling
        self.assertTrue((gf >= 0).all())            # radiation is non-negative
        # Nighttime (SW_IN_POT below threshold) is forced to ~0 by physics.
        night = pot < 0.001
        self.assertLess(float(gf[night].abs().max()), 1.0)

        # The checks above are satisfied by the physics wrapper alone and pass
        # even if the daytime model is broken. Score XGBoost against the known
        # synthetic truth at the daytime gaps it is actually responsible for.
        gaps = swin_gappy.isna()
        daytime_gaps = gaps & (pot >= 0.001)
        truth = swin[daytime_gaps]
        pred = gf[daytime_gaps]
        # Cloudiness here is IID uniform noise, so nothing can beat "mean
        # cloudiness x potential radiation" and r2 cannot exceed ~0.957. Seeded
        # r2 is ~0.926, so 0.90 catches a real regression without being brittle.
        ss_res = float(((truth - pred) ** 2).sum())
        ss_tot = float(((truth - truth.mean()) ** 2).sum())
        self.assertGreater(1 - ss_res / ss_tot, 0.90)
        # Mean cloudiness is unbiased, so the fill must not systematically over-
        # or under-estimate daytime radiation (seeded: ~1.7%).
        self.assertLess(abs(float(pred.mean() - truth.mean())) / float(truth.mean()), 0.03)

        # Flags 0/1/2 keep their library-wide meaning (observed / model /
        # fallback); 3 is the nighttime physics branch. Without context_df every
        # feature derives from the timestamp alone, so a complete feature row
        # always exists and no daytime gap can reach the fallback.
        flag = g.results.flag
        self.assertTrue((flag[swin_gappy.notna()] == 0).all())
        self.assertTrue((flag[daytime_gaps] == 1).all())
        self.assertTrue((flag[gaps & (pot < 0.001)] == 3).all())

    def test_swin_gapfiller_interpolate_short_gaps(self):
        """Short-gap interpolation must beat the model, and never bridge a night."""
        import pandas as pd
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.gapfilling.swin import SWINGapFillerXGBoost
        from diive.variables import potrad_oneflux
        lat, lon, utc = 46.8153, 9.8559, 1  # CH-DAV

        # Real data: the sky state must be autocorrelated for interpolation to have
        # anything to exploit. Synthetic IID cloud noise would rig this against it.
        df = load_exampledata_parquet().loc['2020-06-01':'2020-06-30'].copy()
        truth = df['Rg_f'].copy()
        truth.name = 'SW_IN'
        pot = potrad_oneflux(timestamp_index=truth.index, lat=lat, lon=lon, utc_offset=utc)

        rng = np.random.RandomState(0)
        gappy = truth.copy()
        gappy[rng.rand(len(truth)) < 0.15] = np.nan  # scattered short gaps

        common = dict(series=gappy, lat=lat, lon=lon, utc_offset=utc,
                      random_state=42, verbose=0)
        model_only = SWINGapFillerXGBoost(**common).run().results
        with_interp = SWINGapFillerXGBoost(interpolate_short_gaps=16, **common).run().results

        flag = with_interp.flag
        interp_locs = flag == 4
        self.assertGreater(int(interp_locs.sum()), 0)

        # Interpolation must only touch daytime gaps.
        self.assertTrue(gappy[interp_locs].isna().all())
        self.assertTrue((pot[interp_locs] >= SWINGapFillerXGBoost.KT_MIN_SWINPOT).all())

        # It must never bridge a night: every interpolated record needs an observed
        # daytime value on both sides within its own calendar day.
        day_obs = gappy.notna() & (pot >= SWINGapFillerXGBoost.KT_MIN_SWINPOT)
        for ts in flag[interp_locs].index:
            same_day = day_obs[day_obs.index.normalize() == ts.normalize()]
            self.assertTrue((same_day.index < ts).any() and (same_day.index > ts).any())

        # The point of the feature: better than the model on the gaps it takes over.
        def rmse(pred):
            d = truth[interp_locs] - pred[interp_locs]
            return float(np.sqrt((d ** 2).mean()))

        self.assertLess(rmse(with_interp.gapfilled), rmse(model_only.gapfilled))
        self.assertEqual(int(with_interp.gapfilled.isna().sum()), 0)
        # Observed records stay untouched.
        obs = gappy.notna()
        self.assertTrue(np.allclose(with_interp.gapfilled[obs], truth[obs]))

    def test_swin_gapfiller_fallback_flag(self):
        """A context-driver gap must surface as flag 2, not hide inside flag 1."""
        import pandas as pd
        from diive.gapfilling.swin import SWINGapFillerXGBoost
        from diive.variables import potrad_oneflux
        lat, lon, utc = 47.0, 8.0, 1
        idx = pd.date_range('2022-06-01', '2022-06-30 23:30', freq='30min', name='TIMESTAMP_END')
        pot = potrad_oneflux(timestamp_index=idx, lat=lat, lon=lon, utc_offset=utc)
        rng = np.random.RandomState(0)
        swin = (pot * (0.7 + 0.3 * rng.rand(len(idx)))).clip(lower=0)
        swin.name = 'SW_IN'
        swin_gappy = swin.copy()
        swin_gappy[rng.rand(len(idx)) < 0.3] = np.nan

        # A driver gap overlapping daytime target gaps leaves those records without
        # a complete feature row, so they can only be filled from timestamps.
        ta = pd.Series(15 + 10 * np.sin(np.arange(len(idx)) * 2 * np.pi / 48),
                       index=idx, name='TA')
        ta.iloc[500:560] = np.nan

        g = SWINGapFillerXGBoost(series=swin_gappy, lat=lat, lon=lon, utc_offset=utc,
                                 context_df=ta.to_frame(), random_state=42, verbose=0)
        g.run()

        flag = g.results.flag
        fallback = flag == 2
        self.assertGreater(int(fallback.sum()), 0)      # the branch is reachable ...
        self.assertTrue(swin_gappy[fallback].isna().all())    # ... only at gaps ...
        self.assertTrue((pot[fallback] >= 0.001).all())       # ... in daytime ...
        self.assertEqual(int(g.results.gapfilled.isna().sum()), 0)  # ... and still complete

    def test_gapfilling_randomforest(self):
        """Fill gaps using random forest"""
        df = ed.load_exampledata_parquet()
        df = df.loc[(df.index.year == 2020) & (df.index.month == 7)].copy()

        TARGET_COL = 'NEE_CUT_REF_orig'
        subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

        # Subset with target and features
        # Only High-quality (QCF=0) measured NEE used for model training in this example
        lowquality = df["QCF_NEE"] > 0
        df.loc[lowquality, TARGET_COL] = np.nan
        df = df[subsetcols].copy()

        # Step 1: Engineer features
        engineer = FeatureEngineer(
            target_col=TARGET_COL,
            features_lag=[-1, -1],
            features_lag_stepsize=1,
            features_rolling=None,
            features_rolling_exclude_cols=None,
            features_rolling_stats=None,
            features_diff=None,
            features_diff_exclude_cols=None,
            features_poly_degree=None,
            features_poly_exclude_cols=None,
            vectorize_timestamps=False,
            add_continuous_record_number=False,
            sanitize_timestamp=False
        )
        df_engineered = engineer.fit_transform(df)

        # Step 2: Create gap-filling model with engineered features
        rfts = RandomForestTS(
            input_df=df_engineered,
            target_col=TARGET_COL,
            verbose=True,
            n_estimators=3,
            random_state=42,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1
        )
        rfts.reduce_features(shap_threshold_factor=0.5)
        rfts.report_feature_reduction()
        rfts.trainmodel(showplot_scores=False, showplot_importance=False)
        rfts.report_traintest()
        rfts.fillgaps(showplot_scores=False, showplot_importance=False)
        rfts.report_gapfilling()

        fi = rfts.feature_importances_
        scores = rfts.scores_
        gfdf = rfts.gapfilling_df_
        gapfilled = rfts.get_gapfilled_target()

        # # Plot
        # import matplotlib.pyplot as plt
        # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        # observed = df[TARGET_COL].copy()
        # HeatmapDateTime(series=observed).show()
        # HeatmapDateTime(series=gapfilled).show()
        # gapfilled.cumsum().plot()
        # plt.show()

        # Note: Values use flexible ranges for minimal parameter RF model
        # Simple model (n_estimators=3, no timestamp features) with good generalization
        self.assertGreater(scores['mae'], 1.0)
        self.assertLess(scores['mae'], 2.5)
        self.assertGreater(scores['r2'], 0.5)
        self.assertLess(scores['r2'], 1.0)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfRF'].sum(), gapfilled.sum())
        self.assertGreater(len(fi['SHAP_IMPORTANCE']), 0)  # Has feature importances

    def test_gapfilling_xgboost(self):
        """Fill gaps using XGBoost"""
        df = ed.load_exampledata_parquet()
        df = df.loc[(df.index.year == 2020) & (df.index.month == 7)].copy()

        TARGET_COL = 'NEE_CUT_REF_orig'
        subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

        # Subset with target and features
        # Only High-quality (QCF=0) measured NEE used for model training in this example
        lowquality = df["QCF_NEE"] > 0
        df.loc[lowquality, TARGET_COL] = np.nan
        df = df[subsetcols].copy()

        # Step 1: Engineer features
        engineer = FeatureEngineer(
            target_col=TARGET_COL,
            features_lag=[-1, -1],
            features_lag_stepsize=1,
            features_rolling=None,
            features_rolling_exclude_cols=None,
            features_rolling_stats=None,
            features_diff=None,
            features_diff_exclude_cols=None,
            features_poly_degree=None,
            features_poly_exclude_cols=None,
            vectorize_timestamps=True,
            add_continuous_record_number=True,
            sanitize_timestamp=True
        )
        df_engineered = engineer.fit_transform(df)

        # Step 2: Create gap-filling model with engineered features
        xgbts = XGBoostTS(
            input_df=df_engineered,
            target_col=TARGET_COL,
            verbose=1,
            n_estimators=9,
            random_state=42,
            validate_parameters=True,
            early_stopping_rounds=10,
            max_depth=6,
            learning_rate=0.3,
            n_jobs=-1
        )
        xgbts.reduce_features(shap_threshold_factor=0.5)
        xgbts.report_feature_reduction()
        xgbts.trainmodel(showplot_scores=False, showplot_importance=False)
        xgbts.report_traintest()
        xgbts.fillgaps(showplot_scores=False, showplot_importance=False)
        xgbts.report_gapfilling()

        fi = xgbts.feature_importances_
        scores = xgbts.scores_
        gfdf = xgbts.gapfilling_df_
        gapfilled = xgbts.get_gapfilled_target()

        # # Plot
        # import matplotlib.pyplot as plt
        # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        # observed = df[TARGET_COL].copy()
        # HeatmapDateTime(series=observed).show()
        # HeatmapDateTime(series=gapfilled).show()
        # gapfilled.cumsum().plot()
        # plt.show()

        # Note: Values updated to reflect SHAP-based feature importance and shap_threshold_factor=0.5
        # Using flexible ranges due to slight variability in SHAP calculations.
        # Upper bound of the gap-filled sum widened after early stopping was fixed
        # to use a genuine hold-out in the feature-reduction path (early_stopping_rounds
        # is set here), which shifts the fitted model slightly.
        self.assertGreater(scores['mae'], 1.2)
        self.assertLess(scores['mae'], 1.6)
        self.assertGreater(scores['r2'], 0.82)
        self.assertLess(scores['r2'], 0.92)
        self.assertGreater(gfdf['NEE_CUT_REF_orig_gfXG'].sum(), -2000)
        self.assertLess(gfdf['NEE_CUT_REF_orig_gfXG'].sum(), -1200)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfXG'].sum(), gapfilled.sum())
        self.assertGreater(fi['SHAP_IMPORTANCE']['Rg_f'], 2.5)
        self.assertLess(fi['SHAP_IMPORTANCE']['Rg_f'], 3.5)

    def test_gapfilling_stl_features_randomforest(self):
        """Test STL decomposition features with RandomForest gap-filling"""
        df = ed.load_exampledata_parquet()
        # Use longer time period for STL decomposition (requires substantial data)
        df = df.loc[(df.index.year >= 2019) & (df.index.year <= 2019)].copy()

        TARGET_COL = 'NEE_CUT_REF_orig'
        subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

        # Subset with target and features
        lowquality = df["QCF_NEE"] > 0
        df.loc[lowquality, TARGET_COL] = np.nan
        df = df[subsetcols].copy()

        # Step 1: Engineer features with STL decomposition
        engineer = FeatureEngineer(
            target_col=TARGET_COL,
            features_lag=[-1, -1],
            features_lag_stepsize=1,
            features_rolling=[6],  # Short rolling window for testing
            features_rolling_exclude_cols=None,
            features_rolling_stats=None,
            features_diff=None,
            features_diff_exclude_cols=None,
            features_ema=None,
            features_ema_exclude_cols=None,
            features_poly_degree=None,
            features_poly_exclude_cols=None,
            features_stl=True,  # Enable STL features
            features_stl_method='harmonic',  # Use harmonic method for better compatibility
            features_stl_seasonal_period=48,  # Daily cycle for 30-min data
            features_stl_exclude_cols=None,
            features_stl_components=['trend', 'seasonal', 'residual'],  # Extract all
            vectorize_timestamps=False,
            add_continuous_record_number=False,
            sanitize_timestamp=False
        )
        df_engineered = engineer.fit_transform(df)

        # Step 2: Create gap-filling model with engineered features (STL included)
        rfts = RandomForestTS(
            input_df=df_engineered,
            target_col=TARGET_COL,
            verbose=True,
            n_estimators=3,
            random_state=42,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1
        )

        # Verify that STL features were created in the model dataframe
        model_df_cols = rfts.model_df.columns.tolist()
        stl_cols = [c for c in model_df_cols if 'STL' in c]
        self.assertGreater(len(stl_cols), 0, "No STL feature columns found in model dataframe")

        # Verify STL column naming convention
        for col in stl_cols:
            self.assertIn('_STL_', col)
            # Check that it ends with a component name
            valid_endings = ['_STL_TREND', '_STL_SEASONAL', '_STL_RESIDUAL']
            found_valid = any(col.endswith(ending) for ending in valid_endings)
            self.assertTrue(found_valid, f"STL column {col} doesn't end with valid component name")

        # Run gap-filling with STL features
        rfts.trainmodel(showplot_scores=False, showplot_importance=False)
        rfts.fillgaps(showplot_scores=False, showplot_importance=False)

        scores = rfts.scores_
        gapfilled = rfts.get_gapfilled_target()

        # Verify results are reasonable
        self.assertGreater(scores['mae'], 0.5)
        self.assertLess(scores['mae'], 3.0)
        self.assertGreater(scores['r2'], -0.5)
        self.assertLess(scores['r2'], 1.0)
        self.assertGreater(len(gapfilled), 0)

    def test_gapfilling_stl_features_xgboost(self):
        """Test STL decomposition features with XGBoost gap-filling"""
        df = ed.load_exampledata_parquet()
        # Use longer time period for STL decomposition
        df = df.loc[(df.index.year >= 2019) & (df.index.year <= 2019)].copy()

        TARGET_COL = 'NEE_CUT_REF_orig'
        subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

        # Subset with target and features
        lowquality = df["QCF_NEE"] > 0
        df.loc[lowquality, TARGET_COL] = np.nan
        df = df[subsetcols].copy()

        # Step 1: Engineer features with selective STL components
        engineer = FeatureEngineer(
            target_col=TARGET_COL,
            features_lag=[-1, -1],
            features_lag_stepsize=1,
            features_rolling=None,
            features_rolling_exclude_cols=None,
            features_rolling_stats=None,
            features_diff=None,
            features_diff_exclude_cols=None,
            features_ema=None,
            features_ema_exclude_cols=None,
            features_poly_degree=None,
            features_poly_exclude_cols=None,
            features_stl=True,  # Enable STL features
            features_stl_method='harmonic',  # Use harmonic method for better compatibility
            features_stl_seasonal_period=48,  # Daily cycle for 30-min data
            features_stl_exclude_cols=None,
            features_stl_components=['trend', 'seasonal'],  # Only trend and seasonal
            vectorize_timestamps=False,
            add_continuous_record_number=False,
            sanitize_timestamp=False
        )
        df_engineered = engineer.fit_transform(df)

        # Step 2: Create gap-filling model with engineered features (STL included)
        xgbts = XGBoostTS(
            input_df=df_engineered,
            target_col=TARGET_COL,
            verbose=1,
            n_estimators=3,
            random_state=42,
            validate_parameters=False,
            early_stopping_rounds=5,
            max_depth=3
        )

        # Verify that STL features were created
        model_df_cols = xgbts.model_df.columns.tolist()
        stl_cols = [c for c in model_df_cols if 'STL' in c]
        self.assertGreater(len(stl_cols), 0, "No STL feature columns found in model dataframe")

        # Should have trend and seasonal, but NOT residual
        residual_cols = [c for c in stl_cols if c.endswith('_STL_RESIDUAL')]
        self.assertEqual(len(residual_cols), 0, "Found residual STL columns when only trend/seasonal requested")

        # Run gap-filling
        xgbts.trainmodel(showplot_scores=False, showplot_importance=False)
        xgbts.fillgaps(showplot_scores=False, showplot_importance=False)

        scores = xgbts.scores_
        gapfilled = xgbts.get_gapfilled_target()

        # Verify results are reasonable
        self.assertGreater(scores['mae'], 0.5)
        self.assertLess(scores['mae'], 3.0)
        self.assertGreater(len(gapfilled), 0)

    def test_shap_treeexplainer_xgboost(self):
        """shap must parse XGBoost's base_score unaided.

        XGBoost serializes base_score as e.g. '[-3.18E0]'. shap <= 0.49 chokes on
        the brackets and diive carried a monkey-patch for it; shap >= 0.50 parses
        it natively, which is why the pin has that floor. This fails if the floor
        is ever lowered again.
        """
        import shap
        from xgboost import XGBRegressor

        rng = np.random.default_rng(42)
        X = rng.random((200, 3))
        y = X[:, 0] * 3 + rng.random(200)
        model = XGBRegressor(n_estimators=10, random_state=42).fit(X, y)

        shap_values = shap.TreeExplainer(model).shap_values(X)

        self.assertEqual(shap_values.shape, X.shape)
        self.assertTrue(np.isfinite(shap_values).all())

        # y depends only on X[:, 0], so that feature must dominate. Flexible bounds
        # because SHAP values fluctuate between runs.
        importances = np.abs(shap_values).mean(axis=0)
        self.assertGreater(importances[0], importances[1])
        self.assertGreater(importances[0], importances[2])


if __name__ == '__main__':
    unittest.main()
