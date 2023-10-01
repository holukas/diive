import importlib.metadata
import unittest
from datetime import datetime

import numpy as np

import diive.configs.exampledata as ed
from diive.core.dfun.stats import sstats  # Time series stats
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS


class TestGapFilling(unittest.TestCase):

    def test_optimize_rf_params(self):
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
        df.describe()
        statsdf = sstats(df[TARGET_COL])
        print(statsdf)

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
            perm_n_repeats=11,
            n_jobs=-1
        )
        rfts.trainmodel(showplot_predictions=False, showplot_importance=False, verbose=1)
        rfts.fillgaps(showplot_scores=False, showplot_importance=False, verbose=1)
        rfts.report()
        observed = df[TARGET_COL]
        gapfilled = rfts.get_gapfilled_target()

        fi = rfts.feature_importances_
        scores = rfts.scores_
        gfdf = rfts.gapfilling_df_

        # # Plot
        # HeatmapDateTime(series=observed).show()
        # HeatmapDateTime(series=gapfilled).show()
        # gapfilled.cumsum().plot()
        # plt.show()

        self.assertEqual(scores['mae'], 1.9060864968011548)
        self.assertEqual(scores['r2'], 0.8026866876500566)
        self.assertEqual(scores['mse'], 7.306871131968254)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfRF'].sum(), -64754.77103135061)
        self.assertEqual(fi['importances']['PERM_IMPORTANCE']['Rg_f'], 0.9903003111303493)


if __name__ == '__main__':
    unittest.main()
