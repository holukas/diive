import unittest

import numpy as np

import diive.configs.exampledata as ed
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS, QuickFillRFTS
from diive.pkgs.gapfilling.xgboost_ts import XGBoostTS


class TestGapFilling(unittest.TestCase):

    def test_optimize_rf_params(self):
        pass

    def test_quickfill(self):
        pass

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
            include_timestamp_as_features=True,
            add_continuous_record_number=True,
            sanitize_timestamp=True,
            n_estimators=9,
            random_state=42,
            min_samples_split=20,
            min_samples_leaf=10,
            perm_n_repeats=3,
            n_jobs=-1
        )
        rfts.reduce_features()
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

        self.assertEqual(scores['mae'], 1.681033504987219)
        self.assertEqual(scores['r2'], 0.7932199707457357)
        self.assertEqual(scores['mse'], 5.628773469718195)
        self.assertAlmostEqual(gfdf['NEE_CUT_REF_orig_gfRF'].sum(), -612.7853333778166, places=5)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfRF'].sum(), gapfilled.sum())
        self.assertEqual(fi['PERM_IMPORTANCE']['Rg_f'], 1.2153029844335206)

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
            include_timestamp_as_features=True,
            add_continuous_record_number=True,
            sanitize_timestamp=True,
            n_estimators=9,
            random_state=42,
            perm_n_repeats=3,
            validate_parameters=True,
            early_stopping_rounds=10,
            max_depth=6,
            learning_rate=0.3,
            n_jobs=-1
        )
        xgbts.reduce_features()
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

        self.assertEqual(scores['mae'], 1.474872398011102)
        self.assertEqual(scores['r2'], 0.8472293439937181)
        self.assertEqual(scores['mse'], 4.158580587210508)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfXG'].sum(), -1364.7877804577352)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfXG'].sum(), gapfilled.sum())
        self.assertEqual(fi['PERM_IMPORTANCE']['Rg_f'], 1.1121007247659092)


if __name__ == '__main__':
    unittest.main()
