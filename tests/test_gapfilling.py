import unittest

import numpy as np

import diive.configs.exampledata as ed
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS


class TestGapFilling(unittest.TestCase):

    def test_optimize_rf_params(self):
        pass

    def test_quickfill(self):
        pass

    def test_gapfilling_randomforest(self):
        """Fill gaps using random forest"""
        df = ed.load_exampledata_parquet()

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
            verbose=1,
            features_lag=[-2, 2],
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
        # observed = df[TARGET_COL]
        # gapfilled = rfts.get_gapfilled_target()

        fi = rfts.feature_importances_
        scores = rfts.scores_
        gfdf = rfts.gapfilling_df_

        # # Plot
        # HeatmapDateTime(series=observed).show()
        # HeatmapDateTime(series=gapfilled).show()
        # gapfilled.cumsum().plot()
        # plt.show()

        self.assertAlmostEqual(scores['mae'], 1.9028138023273018, places=3)
        self.assertAlmostEqual(scores['r2'], 0.8027689585127069, places=3)
        self.assertAlmostEqual(scores['mse'], 7.303819917384151, places=2)
        self.assertAlmostEqual(gfdf['NEE_CUT_REF_orig_gfRF'].sum(), -63372.39446397088, places=0)
        self.assertAlmostEqual(fi['PERM_IMPORTANCE']['Rg_f'], 0.9711855286033096, places=2)


if __name__ == '__main__':
    unittest.main()
