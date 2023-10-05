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
        rfts.reduce_features()
        rfts.report_feature_reduction()
        rfts.trainmodel(showplot_scores=True, showplot_importance=True)
        rfts.report_traintest()
        rfts.fillgaps(showplot_scores=True, showplot_importance=True)
        rfts.report_gapfilling()
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

        self.assertEqual(scores['mae'], 1.902421699006966)
        self.assertEqual(scores['r2'], 0.8028454341680735)
        self.assertEqual(scores['mse'], 7.300992459434791)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfRF'].sum(), -63541.1261782166)
        self.assertEqual(fi['PERM_IMPORTANCE']['Rg_f'], 0.9831618002267694)


if __name__ == '__main__':
    unittest.main()
