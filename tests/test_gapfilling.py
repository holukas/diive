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
            swin_class=50,
            ta_class=2.5,
            vpd_class=0.5,  # kPa; 5 hPa is default for reference
            min_n_vals_nt=0
        )
        mds.run()
        # mds.report()
        # mds.showplot()

        results = mds.gapfilling_df_
        self.assertEqual(len(results), 1488)
        self.assertEqual(mds.scores_['r2'], 0.8489359708564446)
        self.assertEqual(mds.scores_['medae'], 0.7622857142857145)
        self.assertEqual(results[mds.target_gapfilled].isnull().sum(), 0)
        self.assertEqual(results[mds.target_gapfilled_flag].sum(), 914)
        counts = Counter(results[mds.target_gapfilled_flag])
        # Missing in measured as indicated by flag > 0
        a = results[mds.target_gapfilled_flag][results[mds.target_gapfilled_flag] > 0].count()
        # Missing in measured
        b = results[mds.flux].isnull().sum()
        self.assertEqual(a, 770)
        self.assertEqual(b, 770)
        self.assertEqual(counts[0], 718)
        self.assertEqual(counts[1], 677)

        mds = FluxMDS(
            df=df,
            flux='NEE_CUT_REF_orig',
            ta='Tair_f',
            swin='Rg_f',
            vpd='VPD_f',
            swin_class=50,
            ta_class=2.5,
            vpd_class=0.5,  # kPa; 5 hPa is default for reference
            min_n_vals_nt=5
        )
        mds.run()
        # mds.report()
        # mds.showplot()

        results = mds.gapfilling_df_
        self.assertEqual(len(results), 1488)
        self.assertEqual(mds.scores_['r2'], 0.8191458989901276)
        self.assertEqual(mds.scores_['medae'], 0.9554999999999998)
        self.assertEqual(results[mds.target_gapfilled].isnull().sum(), 0)
        self.assertEqual(results[mds.target_gapfilled_flag].sum(), 1529)
        counts = Counter(results[mds.target_gapfilled_flag])
        # Missing in measured as indicated by flag > 0
        a = results[mds.target_gapfilled_flag][results[mds.target_gapfilled_flag] > 0].count()
        # Missing in measured
        b = results[mds.flux].isnull().sum()
        self.assertEqual(a, 770)
        self.assertEqual(b, 770)
        self.assertEqual(counts[0], 718)
        self.assertEqual(counts[1], 322)

    def test_gapfilling_longterm_randomforest(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.pkgs.gapfilling.longterm import LongTermGapFillingRandomForestTS
        TARGET_COL = 'NEE_CUT_REF_orig'
        subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']
        source_df = load_exampledata_parquet()
        df = source_df.copy()
        locs = (df.index.year >= 2014) & (df.index.year <= 2018)
        df = df.loc[locs].copy()
        # This example uses NEE flux data, only records where the quality flag QCF is 0 (highest quality) are retained
        lowquality = df["QCF_NEE"] > 0
        df.loc[lowquality, TARGET_COL] = np.nan  # Set fluxes of lower quality to missing
        df = df[subsetcols].copy()  # Keep subset columns only
        gf = LongTermGapFillingRandomForestTS(
            input_df=df,
            target_col=TARGET_COL,
            verbose=2,
            features_lag=[-1, -1],
            features_lag_exclude_cols=None,
            include_timestamp_as_features=True,
            add_continuous_record_number=True,
            sanitize_timestamp=True,
            perm_n_repeats=3,
            test_size=0.25,
            n_estimators=9,
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1
        )
        gf.create_yearpools()
        gf.initialize_yearly_models()
        gf.reduce_features_across_years()
        gf.fillgaps()

        results = gf.gapfilling_df_
        self.assertEqual(len(results), 87648)
        self.assertEqual(len(results.columns), 8)
        self.assertEqual(results[gf.gapfilled_.name].sum(), gf.gapfilled_.sum())
        self.assertEqual(results[TARGET_COL].isnull().sum(), 67444)
        self.assertEqual(results[gf.gapfilled_.name].isnull().sum(), 0)
        self.assertAlmostEqual(results[gf.gapfilled_.name].sum(), -39072.126777777776, places=5)
        self.assertEqual(results['FLAG_NEE_CUT_REF_orig_gfRF_ISFILLED'].sum(), 67444)
        self.assertEqual(len(gf.features_reduced_across_years), 11)
        self.assertEqual(gf.feature_ranks_per_year.min().min(), 1)
        self.assertEqual(gf.feature_ranks_per_year.max().max(), 11)
        self.assertEqual(len(gf.feature_importances_yearly_.keys()), 5)
        self.assertAlmostEqual(gf.feature_importances_yearly_['2017']['PERM_IMPORTANCE']['Rg_f'], 1.1685750257993892,
                               places=5)
        scores = []
        r2s = []
        for year, s in gf.scores_.items():
            scores.append(s['mae'])
            r2s.append(s['r2'])
        self.assertEqual(len(scores), 5)
        self.assertAlmostEqual(mean(scores), 1.3502779449542526, places=5)
        self.assertAlmostEqual(mean(r2s), 0.8878948369317223, places=5)
        self.assertEqual(type(gf.results_yearly_['2017']), MlRegressorGapFillingBase)
        self.assertAlmostEqual(gf.results_yearly_['2014'].scores_['rmse'], 2.169915773456053, places=5)
        self.assertEqual(gf.yearpools['2014']['poolyears'], [2014, 2015, 2016])
        self.assertEqual(gf.yearpools['2018']['poolyears'], [2016, 2017, 2018])
        self.assertEqual(gf.yearpools['2016']['poolyears'], [2015, 2016, 2017])
        self.assertAlmostEqual(gf.yearpools['2017']['df'].sum().sum(), 83261338263.81297, places=5)
        self.assertAlmostEqual(gf.feature_importance_per_year.sum().sum(), 11.707053647511076, places=5)

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

        self.assertAlmostEqual(scores['mae'], 1.7173094522108192, places=5)
        self.assertEqual(scores['r2'], 0.7942644518782538)
        self.assertEqual(scores['mse'], 5.600341576611584)
        self.assertAlmostEqual(gfdf['NEE_CUT_REF_orig_gfRF'].sum(), -706.6381230007763, places=5)
        self.assertEqual(gfdf['NEE_CUT_REF_orig_gfRF'].sum(), gapfilled.sum())
        self.assertEqual(fi['PERM_IMPORTANCE']['Rg_f'], 1.1749124161423874)

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
