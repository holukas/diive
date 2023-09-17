import importlib.metadata
import unittest
from datetime import datetime

import numpy as np

import diive.configs.exampledata as ed
from diive.core.dfun.frames import steplagged_variants, add_continuous_record_number
from diive.core.dfun.stats import sstats  # Time series stats
from diive.core.times.times import include_timestamp_as_cols
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS

dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"This page was last modified on: {dt_string}")
version_diive = importlib.metadata.version("diive")
print(f"diive version: v{version_diive}")


class TestGapFilling(unittest.TestCase):

    def test_optimize_rf_params(self):


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

        # Lagged variants
        df = steplagged_variants(df=df,
                                 stepsize=1,
                                 stepmax=1,
                                 exclude_cols=[TARGET_COL])

        # Include timestamp info as data columns
        df = include_timestamp_as_cols(df=df, txt="(...)")

        # Add continuous record number as new column
        df = add_continuous_record_number(df=df)

        # Random forest
        rfts = RandomForestTS(
            input_df=df,
            target_col=TARGET_COL,
            verbose=1,
            n_estimators=9,
            random_state=42,
            min_samples_split=10,
            min_samples_leaf=5,
            perm_n_repeats=22,
            n_jobs=-1
        )
        rfts.trainmodel(showplot_predictions=False, showplot_importance=False, verbose=1)
        rfts.fillgaps(showplot_scores=False, showplot_importance=False, verbose=1)
        rfts.report()
        observed = df[TARGET_COL]
        gapfilled = rfts.get_gapfilled_target()

        fi = rfts.feature_importances
        scores = rfts.scores
        gfdf = rfts.gapfilling_df

        # # Plot
        # HeatmapDateTime(series=observed).show()
        # HeatmapDateTime(series=gapfilled).show()
        # gapfilled.cumsum().plot()
        # plt.show()

        self.assertEqual(scores['mae'], 1.718126947003101)
        self.assertEqual(scores['r2'], 0.8319161959118784)
        self.assertEqual(scores['mse'], 6.2244492336365935)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfRF'].sum(), -64338.46474027201)
        self.assertEqual(fi['importances']['PERM_IMPORTANCE']['Rg_f'], 1.1448277365835955)


if __name__ == '__main__':
    unittest.main()
