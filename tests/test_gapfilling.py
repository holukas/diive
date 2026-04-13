import unittest

import numpy as np
from numpy import mean

import diive.configs.exampledata as ed
from diive.core.ml.common import MlRegressorGapFillingBase
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS
from diive.pkgs.gapfilling.xgboost_ts import XGBoostTS


class TestGapFilling(unittest.TestCase):

    def test_optimize_rf_params(self):
        pass

    def test_quickfill(self):
        pass

    def test_fluxmds(self):
        from collections import Counter
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.pkgs.gapfilling.mds import FluxMDS
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
        self.assertEqual(mds.scores_['r2'], 0.648227237609387)
        self.assertEqual(mds.scores_['medae'], 2.0329177096370477)
        self.assertEqual(results[mds.target_gapfilled].isnull().sum(), 0)
        self.assertEqual(results[mds.target_gapfilled_flag].sum(), 1919)
        counts = Counter(results[mds.target_gapfilled_flag])
        # Missing in measured as indicated by flag > 0
        a = results[mds.target_gapfilled_flag][results[mds.target_gapfilled_flag] > 0].count()
        # Missing in measured
        b = results[mds.flux].isnull().sum()
        self.assertEqual(a, 770)
        self.assertEqual(b, 770)
        self.assertEqual(counts[0], 718)
        self.assertEqual(counts[1], 166)
        self.assertEqual(counts[2], 59)
        self.assertEqual(counts[3], 545)

    def test_gapfilling_longterm_randomforest(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.pkgs.gapfilling.longterm import LongTermGapFillingRandomForestTS
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
        gf = LongTermGapFillingRandomForestTS(
            input_df=df,
            target_col=TARGET_COL,
            verbose=1,
            features_lag=[-1, -1],
            features_lag_exclude_cols=None,
            features_lag_stepsize=1,
            vectorize_timestamps=False,
            add_continuous_record_number=False,
            sanitize_timestamp=False,
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
        from diive.pkgs.gapfilling.interpolate import linear_interpolation
        df = load_exampledata_parquet()
        df = df.loc[df.index.year == 2022].copy()
        series = df['NEE_CUT_REF_orig'].copy()
        series_gapfilled = linear_interpolation(series=series, limit=10)
        self.assertEqual(series_gapfilled.isnull().sum(), 7856)
        self.assertEqual(series.isnull().sum(), 11412)

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

        rfts = RandomForestTS(
            input_df=df,
            target_col=TARGET_COL,
            verbose=True,
            features_lag=[-1, -1],
            features_lag_stepsize=1,
            vectorize_timestamps=False,
            add_continuous_record_number=False,
            sanitize_timestamp=False,
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

        xgbts = XGBoostTS(
            input_df=df,
            target_col=TARGET_COL,
            verbose=1,
            features_lag=[-1, -1],
            features_lag_stepsize=1,
            vectorize_timestamps=True,
            add_continuous_record_number=True,
            sanitize_timestamp=True,
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
        # Using flexible ranges due to slight variability in SHAP calculations
        self.assertGreater(scores['mae'], 1.2)
        self.assertLess(scores['mae'], 1.6)
        self.assertGreater(scores['r2'], 0.82)
        self.assertLess(scores['r2'], 0.92)
        self.assertGreater(gfdf['NEE_CUT_REF_orig_gfXG'].sum(), -2000)
        self.assertLess(gfdf['NEE_CUT_REF_orig_gfXG'].sum(), -1400)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfXG'].sum(), gapfilled.sum())
        self.assertGreater(fi['SHAP_IMPORTANCE']['Rg_f'], 2.5)
        self.assertLess(fi['SHAP_IMPORTANCE']['Rg_f'], 3.5)


if __name__ == '__main__':
    unittest.main()
