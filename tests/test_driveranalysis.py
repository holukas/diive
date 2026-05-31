"""
Unit tests for diive.analysis.driveranalysis (DriverAnalysis + zero-dep ALE).

The tests use small synthetic time series with known driver relationships so the
expected verdicts are deterministic enough for flexible assertions (SHAP
importances fluctuate a few percent between fits — see CLAUDE.md).
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import diive as dv
from diive.analysis.driveranalysis import (
    DriverAnalysis,
    DriverAnalysisResult,
    AleCurve,
    accumulated_local_effects,
    accumulated_local_effects_2d,
)


def _synthetic(months: int = 4, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """30-min series where NEE responds to SW_IN and TA, not to JUNK."""
    n = 48 * 30 * months
    idx = pd.date_range('2020-05-01', periods=n, freq='30min')
    rng = np.random.RandomState(seed)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    ta = 15 + 8 * np.sin(2 * np.pi * doy / 365) + 4 * np.sin(2 * np.pi * hour / 24) + rng.randn(n)
    sw = np.clip(600 * np.sin(2 * np.pi * (hour - 6) / 24), 0, None) + rng.randn(n) * 15
    vpd = np.clip(0.25 * ta + 0.002 * sw + rng.randn(n) * 0.2, 0, None)
    junk = rng.randn(n)
    nee = -0.02 * sw - 0.08 * ta + rng.randn(n) * 1.2
    target = pd.Series(nee, index=idx, name='NEE')
    drivers = pd.DataFrame({'TA': ta, 'SW_IN': sw, 'VPD': vpd, 'JUNK': junk}, index=idx)
    return target, drivers


def _fast_rf():
    """Small forest so the full multi-fit pipeline stays quick in tests."""
    return RandomForestRegressor(n_estimators=30, min_samples_leaf=3, random_state=42)


class TestAle(unittest.TestCase):

    def test_ale_increasing_decreasing(self):
        """ALE recovers the sign of a known linear response."""
        rng = np.random.RandomState(1)
        x1 = rng.uniform(0, 10, 2000)
        x2 = rng.uniform(0, 10, 2000)
        y = 2.0 * x1 - 1.0 * x2 + rng.randn(2000) * 0.01
        X = pd.DataFrame({'x1': x1, 'x2': x2})
        model = LinearRegression().fit(X, y)

        c1 = accumulated_local_effects(model, X, 'x1', grid_size=20)
        c2 = accumulated_local_effects(model, X, 'x2', grid_size=20)
        self.assertIsInstance(c1, AleCurve)
        self.assertEqual(c1.direction(flat_threshold=0.1), '+')
        self.assertEqual(c2.direction(flat_threshold=0.1), '-')
        # Range ~ slope * span (2 * 10 and 1 * 10), allow generous tolerance.
        self.assertGreater(c1.ale_range, 15)
        self.assertGreater(c2.ale_range, 7)
        # Centered: mean effect ~ 0.
        self.assertAlmostEqual(float(np.mean(c1.ale)), 0.0, delta=2.0)

    def test_ale_flat_for_irrelevant_feature(self):
        """A feature the model ignores yields a flat ALE."""
        rng = np.random.RandomState(2)
        x1 = rng.uniform(0, 10, 1500)
        x2 = rng.uniform(0, 10, 1500)
        y = 3.0 * x1 + rng.randn(1500) * 0.01
        X = pd.DataFrame({'x1': x1, 'x2': x2})
        model = LinearRegression().fit(X, y)
        c2 = accumulated_local_effects(model, X, 'x2', grid_size=20)
        self.assertEqual(c2.direction(flat_threshold=0.5), 'flat')

    def test_ale_2d_shape(self):
        """2D ALE returns a finite surface of the expected shape."""
        rng = np.random.RandomState(3)
        x1 = rng.uniform(0, 10, 1500)
        x2 = rng.uniform(0, 10, 1500)
        y = x1 * x2 + rng.randn(1500) * 0.01  # genuine interaction
        X = pd.DataFrame({'x1': x1, 'x2': x2})
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
        res = accumulated_local_effects_2d(model, X, 'x1', 'x2', grid_size=6)
        self.assertEqual(res.ale.shape, (res.y_edges.size, res.x_edges.size))
        self.assertTrue(np.isfinite(res.ale).all())
        self.assertGreater(res.interaction_strength, 0)


class TestDriverAnalysisStatic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        target, drivers = _synthetic(months=2)
        cls.da = DriverAnalysis(target=target, drivers=drivers, model=_fast_rf(),
                                verbose=0).run(levels=('static',))

    def test_results_type_and_scores_heldout(self):
        res = self.da.results
        self.assertIsInstance(res, DriverAnalysisResult)
        # Held-out (out-of-sample) score is present and sane.
        self.assertIn('r2', res.model_scores)
        self.assertGreater(res.model_scores['r2'], 0.3)

    def test_shap_relevance_vs_random(self):
        shap_df = self.da.results.shap_importance
        self.assertEqual(len(shap_df), 4)
        # The real drivers should outrank the irrelevant JUNK column.
        self.assertLess(shap_df.loc['SW_IN', 'shap_rank'], shap_df.loc['JUNK', 'shap_rank'])
        # A pure-noise feature must not clear the .RANDOM benchmark (SHAP noise
        # can still leave it 'weak'; it must never be 'yes').
        self.assertIn(shap_df.loc['JUNK', 'shap_relevant'], ('no', 'weak'))
        self.assertEqual(shap_df.loc['SW_IN', 'shap_relevant'], 'yes')

    def test_ale_curves_present(self):
        ale = self.da.results.ale
        for d in ['TA', 'SW_IN', 'VPD', 'JUNK']:
            self.assertIn(d, ale)
            self.assertIsInstance(ale[d], AleCurve)

    def test_convergence_schema(self):
        conv = self.da.results.convergence
        for col in ['shap_importance', 'shap_rank', 'shap_relevant', 'ale_direction',
                    'ale_range', 'ale_relevant', 'n_methods_run', 'level',
                    'relevance_votes', 'agreement', 'verdict', 'flags']:
            self.assertIn(col, conv.columns)
        self.assertEqual(set(conv.index), {'TA', 'SW_IN', 'VPD', 'JUNK'})
        # JUNK must never be promoted to a real driver verdict.
        self.assertNotIn(conv.loc['JUNK', 'verdict'],
                         ['robust_driver', 'associational_only', 'context_dependent'])


class TestDriverAnalysisTemporal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        target, drivers = _synthetic(months=4)
        cls.da = DriverAnalysis(target=target, drivers=drivers, model=_fast_rf(),
                                lags=list(range(-6, 1)), verbose=0
                                ).run(levels=('static', 'temporal'))

    def test_lagged_importance(self):
        li = self.da.results.lagged_importance
        self.assertIsNotNone(li)
        self.assertEqual(set(li.index), {'TA', 'SW_IN', 'VPD', 'JUNK'})
        self.assertIn(0, li.columns)

    def test_scale_resolved(self):
        sr = self.da.results.scale_resolved
        self.assertIsNotNone(sr)
        self.assertTrue(any('stl' in str(c) for c in sr.columns))

    def test_stratified(self):
        st = self.da.results.stratified
        self.assertIsNotNone(st)
        self.assertGreaterEqual(len(st.columns), 1)

    def test_convergence_has_temporal_fields(self):
        conv = self.da.results.convergence
        for col in ['dominant_lag', 'timescale', 'scale_dependence', 'regime_dependence']:
            self.assertIn(col, conv.columns)
        self.assertEqual(self.da.results.levels_run, ['static', 'temporal'])


class TestDriverAnalysisValidation(unittest.TestCase):

    def test_target_must_be_named(self):
        target, drivers = _synthetic(months=1)
        target.name = None
        with self.assertRaises(ValueError):
            DriverAnalysis(target=target, drivers=drivers)

    def test_target_driver_collision(self):
        target, drivers = _synthetic(months=1)
        drivers = drivers.rename(columns={'TA': 'NEE'})
        with self.assertRaises(ValueError):
            DriverAnalysis(target=target, drivers=drivers)

    def test_pcmci_requires_optional_extra(self):
        """Without tigramite installed, pcmci() raises a helpful ImportError."""
        try:
            import tigramite  # noqa: F401
            self.skipTest("tigramite is installed; cannot test the missing-extra path.")
        except ImportError:
            pass
        target, drivers = _synthetic(months=1)
        da = DriverAnalysis(target=target, drivers=drivers, verbose=0)
        da.fit_model()
        with self.assertRaises(ImportError):
            da.pcmci(tau_max=4)


class TestPublicApi(unittest.TestCase):

    def test_experimental_namespace_exports(self):
        # DriverAnalysis is exposed via the experimental subnamespace, NOT the
        # stable dv.analysis namespace.
        for name in ['DriverAnalysis', 'DriverAnalysisResult', 'AleCurve',
                     'Ale2DResult', 'accumulated_local_effects',
                     'accumulated_local_effects_2d']:
            self.assertTrue(hasattr(dv.analysis.experimental, name),
                            f"missing dv.analysis.experimental.{name}")

    def test_not_in_stable_namespace(self):
        # Guard the experimental boundary: it must stay out of dv.analysis until
        # promoted, so accidental re-exports get caught.
        self.assertFalse(hasattr(dv.analysis, 'DriverAnalysis'),
                         "DriverAnalysis leaked into the stable dv.analysis namespace")

    def test_instantiation_warns_experimental(self):
        # Instantiating must emit an ExperimentalWarning at least once. Reset the
        # module latch so this test is order-independent.
        import warnings
        import diive.analysis.driveranalysis.driveranalysis as da_mod
        from diive.analysis.driveranalysis import ExperimentalWarning
        da_mod._EXPERIMENTAL_WARNED = False
        target, drivers = _synthetic(months=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            DriverAnalysis(target=target, drivers=drivers, verbose=0)
        self.assertTrue(any(issubclass(w.category, ExperimentalWarning) for w in caught))


if __name__ == '__main__':
    unittest.main()
