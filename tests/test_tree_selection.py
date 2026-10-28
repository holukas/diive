"""Tests for the train/test split of the long-term models and the blocked tree selection (L412).

No final test record may reach the feature selection or the blocked tree-count
cross-validation, and identical runs must give identical scores.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd


def _synthetic_years(n_years: int = 4, seed: int = 0) -> pd.DataFrame:
    """A few records per day over several years, with gaps in the target."""
    rng = np.random.default_rng(seed)
    index = pd.date_range('2020-01-01 01:30', periods=n_years * 365 * 4, freq='6h')
    df = pd.DataFrame(rng.normal(size=(len(index), 3)), index=index, columns=['x1', 'x2', 'noise'])
    df['y'] = 3 * df['x1'] - df['x2'] ** 2 + rng.normal(scale=0.3, size=len(index))
    df.loc[rng.random(len(index)) < 0.3, 'y'] = np.nan
    return df


def _eddypro_input():
    """One month of EddyPro example data without the columns init_flux_data computes."""
    from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
    df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
    df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME') if c in df.columns])
    df['TA_1_1_1'] = df['TA_1_1_1'].bfill()
    df['SW_IN_1_1_1'] = df['SW_IN_1_1_1'].bfill()
    df['VPD_kPa'] = df['VPD_EP'].bfill().multiply(0.1)
    return df


def _level33():
    from diive.flux.postprocessingchain import (
        init_flux_data, make_level32_detector, run_level2, run_level31, run_level32,
        run_level33_constant_ustar)
    data = init_flux_data(df=_eddypro_input(), fluxcol='FC', site_lat=46.583056, site_lon=9.790639,
                          utc_offset=1)
    data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
    data = run_level31(data)
    data, sod = make_level32_detector(data)
    sod.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False, verbose=False)
    sod.addflag()
    data = run_level32(data, outlier_detector=sod)
    return run_level33_constant_ustar(data, thresholds=[0.1, 0.15, 0.2],
                                      threshold_labels=['CUT_16', 'CUT_50', 'CUT_84'],
                                      showplot=False, verbose=False)


def _run_xgb(data, **kwargs):
    """run_level41_xgb with the blocked tree selection; returns the result and the CV pools."""
    import diive.gapfilling.tree_selection as ts
    from diive.core.ml.feature_engineer import FeatureEngineer
    from diive.flux.postprocessingchain import run_level41_xgb
    pools = {}
    original = ts.select_n_estimators_blocked_cv

    def spy(p, *args, **kw):
        pools.update(p)
        return original(p, *args, **kw)

    engineer = FeatureEngineer(target_col='_target_', features_lag=[-1, 1], vectorize_timestamps=True,
                               verbose=0)
    with mock.patch.object(ts, 'select_n_estimators_blocked_cv', spy):
        out = run_level41_xgb(data, features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_kPa'], engineer=engineer,
                              n_estimators=40, learning_rate=0.3, early_stopping_rounds=5, verbose=0,
                              n_estimators_cv={'block_days': 7, 'n_folds': 3}, **kwargs)
    return out, pools


def _test_records(instances) -> pd.Index:
    """Test records of all yearly models of all instances."""
    idx = pd.Index([])
    for inst in instances:
        for model in inst.results_yearly_.values():
            idx = idx.append(model.test_df.index)
    return idx.unique()


class TestLongTermSplit(unittest.TestCase):
    """All yearly models hold out the same timestamps, and keep them when set up again."""

    def _instance(self, **kwargs):
        from diive.gapfilling.longterm import LongTermGapFillingXGBoostTS
        gf = LongTermGapFillingXGBoostTS(input_df=_synthetic_years(), target_col='y', verbose=0,
                                         n_estimators=15, max_depth=3, **kwargs)
        gf.create_yearpools()
        gf.initialize_yearly_models()
        return gf

    def test_no_model_trains_on_a_test_record_of_another(self):
        gf = self._instance()
        tests = _test_records([gf])
        self.assertTrue(gf.test_records_.isin(gf.input_df.index).all())
        for model in gf.results_yearly_.values():
            self.assertFalse(model.train_df.index.isin(tests).any())
            complete = model.model_df.dropna().index
            self.assertTrue(model.test_df.index.equals(complete[complete.isin(gf.test_records_)]))
            share = len(model.test_df) / len(complete)
            self.assertAlmostEqual(share, 0.25, delta=0.03)
        # The pools overlap: the middle years are in the pools of several models.
        self.assertGreater(len(gf.results_yearly_), 2)

    def test_split_kept_after_reinitialisation_and_feature_selection(self):
        gf = self._instance()
        before = {y: m.test_df.index for y, m in gf.results_yearly_.items()}
        gf.kwargs = {**gf.kwargs, 'n_estimators': 10}
        gf.initialize_yearly_models()
        for year, model in gf.results_yearly_.items():
            self.assertTrue(model.test_df.index.equals(before[year]))
        gf.reduce_features_across_years()
        # Fewer features can only add complete records, never move one between the sets.
        for year, model in gf.results_yearly_.items():
            self.assertTrue(before[year].isin(model.test_df.index).all())
            self.assertFalse(model.train_df.index.isin(before[year]).any())
            self.assertTrue(model.test_df.index.isin(gf.test_records_).all())

    def test_default_seed_and_explicit_records(self):
        from diive.gapfilling.longterm import DEFAULT_RANDOM_STATE
        a, b = self._instance(), self._instance()
        self.assertEqual(a.kwargs['random_state'], DEFAULT_RANDOM_STATE)
        self.assertTrue(a.test_records_.equals(b.test_records_))
        self.assertFalse(a.test_records_.equals(self._instance(random_state=7).test_records_))
        # Unseeded instances draw their own records, unless they are handed over.
        c = self._instance(random_state=None)
        d = self._instance(random_state=None, test_records=c.test_records_)
        self.assertTrue(d.test_records_.equals(c.test_records_))
        for year, model in d.results_yearly_.items():
            self.assertTrue(model.test_df.index.equals(c.results_yearly_[year].test_df.index))


class TestTreeSelectionLeakage(unittest.TestCase):
    """n_estimators_cv never sees a final test record (L412)."""

    @classmethod
    def setUpClass(cls):
        cls.data = _level33()

    def _assert_no_leak(self, out, pools):
        instances = list(out.levels.level41_xgb.values())
        tests = _test_records(instances)
        self.assertGreater(len(tests), 0)
        self.assertGreater(len(pools), 0)
        for label, pool in pools.items():
            self.assertFalse(pool.index.isin(tests).any(), f"pool {label} holds test records")
        # Every scenario holds out the same timestamps.
        first = instances[0].test_records_
        for inst in instances[1:]:
            self.assertTrue(inst.test_records_.equals(first))

    def test_cv_pools_hold_no_test_record_of_any_scenario(self):
        out, pools = _run_xgb(self.data)  # no random_state: the default seed
        self._assert_no_leak(out, pools)

    def test_cv_pools_after_shared_feature_selection(self):
        out, pools = _run_xgb(self.data, reduce_features=True)
        self._assert_no_leak(out, pools)

    def test_unseeded_run_shares_the_test_records(self):
        out, pools = _run_xgb(self.data, random_state=None)
        self._assert_no_leak(out, pools)

    def test_identical_runs_give_identical_scores(self):
        results = []
        for global_seed in (1, 2):
            # Separate processes start from different global random states.
            np.random.seed(global_seed)
            out, _ = _run_xgb(self.data)
            results.append(out.levels.level41_xgb)
        a, b = results
        self.assertEqual(list(a), list(b))
        for scen in a:
            self.assertEqual(a[scen].tree_selection_.n_estimators, b[scen].tree_selection_.n_estimators)
            pd.testing.assert_frame_equal(a[scen].tree_selection_.curves, b[scen].tree_selection_.curves)
            for year, model in a[scen].results_yearly_.items():
                other = b[scen].results_yearly_[year]
                self.assertTrue(model.test_df.index.equals(other.test_df.index))
                self.assertEqual(model.scores_traintest_, other.scores_traintest_)
            pd.testing.assert_series_equal(a[scen].gapfilled_, b[scen].gapfilled_)


class TestNotBoosterSettings(unittest.TestCase):

    def test_model_settings_do_not_reach_xgb_cv(self):
        import xgboost as xgb
        from diive.gapfilling.tree_selection import select_n_estimators_blocked_cv
        rng = np.random.default_rng(0)
        index = pd.date_range('2021-01-01 00:15', periods=48 * 40, freq='30min')
        x = rng.normal(size=(len(index), 2))
        pool = pd.DataFrame({'y': x[:, 0] + rng.normal(0, 0.1, len(index)), 'a': x[:, 0], 'b': x[:, 1]},
                            index=index)
        with mock.patch.object(xgb, 'cv', wraps=xgb.cv) as cv:
            select_n_estimators_blocked_cv({'p': pool}, 'y',
                                           dict(n_estimators=20, max_depth=2, test_size=0.25,
                                                below_zero='zero', shap_max_rows=100,
                                                early_stopping_rounds=5),
                                           block_days=7, n_folds=3)
        params = cv.call_args.args[0]
        for key in ('test_size', 'below_zero', 'shap_max_rows', 'early_stopping_rounds', 'n_estimators'):
            self.assertNotIn(key, params)


if __name__ == '__main__':
    unittest.main()
