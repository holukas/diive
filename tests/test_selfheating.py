"""
Tests for the self-heating (SCOP) physics.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest

import numpy as np
import pandas as pd


def _physics(rho_v_value, method):
    """One ScopPhysics run on a small synthetic record, everything but humidity fixed."""
    from diive.flux.lowres.selfheating import ScopPhysics
    n = 480  # 10 days at 30 min
    idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min', name='TIMESTAMP_MIDDLE')
    hr = idx.hour + idx.minute / 60
    ta = pd.Series(15 + 8 * np.sin(2 * np.pi * (hr - 9) / 24), index=idx)
    physics = ScopPhysics(
        ta=ta,
        gas_density=pd.Series(1.7e7, index=idx),      # umol m-3
        rho_a=pd.Series(1.2, index=idx),              # kg m-3
        rho_v=pd.Series(rho_v_value, index=idx),      # kg m-3
        u=pd.Series(2.5, index=idx),                  # m s-1
        c_p=pd.Series(1005.0, index=idx),             # J K-1 kg-1
        ustar=pd.Series(0.4, index=idx),              # m s-1
        lat=47.478333, lon=8.364389, utc_offset=1,
    )
    physics.run(correction_method_base=method, gapfill=False)
    return physics.fct_unsc


class TestWaterVapourDilutionIsAppliedByEveryMethod(unittest.TestCase):
    """BUR08 must carry the (1 + 1.6077 rho_v/rho_d) factor, like BUR06 and JAR09.

    In Burba et al. (2008) the instrument-surface heat fluxes are *added to* the
    ambient sensible heat flux (Method 4) and the total enters the WPL equation,
    whose sensible-heat term carries that factor; their poster states the
    already-corrected form used here verbatim. BUR08 used to omit it, so choosing
    a correction_method_base silently changed two things at once: the
    surface-temperature model and whether the factor applied at all.
    """

    RHO_V = 0.012  # kg m-3, ~12 g m-3, a normal summer humidity

    def _expected_factor(self):
        rho_d = 1.2 - self.RHO_V  # dry air density = rho_a - rho_v
        return 1 + 1.6077 * (self.RHO_V / rho_d)

    def test_every_method_scales_with_humidity_by_the_same_factor(self):
        for method in ('BUR08', 'BUR06', 'JAR09'):
            with self.subTest(method=method):
                dry = _physics(0.0, method)
                humid = _physics(self.RHO_V, method)
                ratio = (humid / dry).dropna()
                self.assertGreater(len(ratio), 0)
                np.testing.assert_allclose(ratio.to_numpy(),
                                           self._expected_factor(),
                                           rtol=1e-9)

    def test_the_factor_is_not_negligible(self):
        # ~1% at ordinary humidity - small, but systematic and one-signed.
        self.assertGreater(self._expected_factor(), 1.015)


def _physics_gapfilled():
    """One gap-filled ScopPhysics run: wind gaps make the correction term uncomputable."""
    from diive.flux.lowres.selfheating import ScopPhysics
    n = 480  # 10 days at 30 min
    idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min', name='TIMESTAMP_MIDDLE')
    hr = idx.hour + idx.minute / 60
    ta = pd.Series(15 + 8 * np.sin(2 * np.pi * (hr - 9) / 24), index=idx)
    u = pd.Series(2.5, index=idx)
    u.iloc[100:140] = np.nan  # wind gaps -> the correction term cannot be computed
    physics = ScopPhysics(
        ta=ta, gas_density=pd.Series(1.7e7, index=idx), rho_a=pd.Series(1.2, index=idx),
        rho_v=pd.Series(0.012, index=idx), u=u, c_p=pd.Series(1005.0, index=idx),
        ustar=pd.Series(0.4, index=idx), lat=47.478333, lon=8.364389, utc_offset=1)
    physics.run(correction_method_base="BUR08", gapfill=True)
    return physics


class TestGapFillFillsTheGaps(unittest.TestCase):
    """The gap-fill must fill; it used to delete the gaps and report success.

    `pd.DataFrame.from_dict(frame).dropna()` dropped every row with a NaN in *any*
    column, the target included - so exactly the records to be filled left the
    training frame. XGBoost then reported "Filling 0 missing records" and the
    console blamed "insufficient drivers".
    """

    def test_gaps_in_the_correction_term_are_filled(self):
        physics = _physics_gapfilled()
        before = int(physics.fct_unsc.isna().sum())
        after = int(physics.fct_unsc_gf.isna().sum())
        self.assertGreater(before, 0, 'the fixture must produce gaps')
        self.assertLess(after, before)


class TestGapFilledColumnNamesTheRegressorThatFilledIt(unittest.TestCase):
    """The attribute, the results column and the regressor must all say the same thing.

    `ColumnConfig.fct_unsc_gf` was 'FCT_UNSC_gfRF', left over from a Random Forest
    implementation, while `_gapfill()` read XGBoostTS's own 'FCT_UNSC_gfXG' column and
    returned it unrenamed. So `physics.fct_unsc_gf.name` and the `get_results()` column
    holding that very series disagreed, and the results frame credited a regressor that
    never ran. Renamed to 'FCT_UNSC_gfXG' in v0.91.0 (breaking for code indexing the
    old name).
    """

    def test_the_series_name_the_results_column_and_the_regressor_agree(self):
        from diive.flux.lowres.selfheating import ColumnConfig
        physics = _physics_gapfilled()
        results = physics.get_results()
        gfcol = ColumnConfig().fct_unsc_gf
        # The gap-fill is XGBoostTS, hardcoded in _gapfill(), so the suffix is _gfXG.
        self.assertEqual(gfcol, 'FCT_UNSC_gfXG')
        self.assertEqual(physics.fct_unsc_gf.name, gfcol)
        self.assertIn(gfcol, results.columns)
        self.assertNotIn('FCT_UNSC_gfRF', results.columns)
        # And the column really holds the gap-filled series, not the ungapfilled one.
        np.testing.assert_allclose(results[gfcol].to_numpy(),
                                   physics.fct_unsc_gf.to_numpy())
        self.assertLess(int(results[gfcol].isna().sum()),
                        int(results[ColumnConfig().fct_unsc].isna().sum()))


class TestUnknownMethodRaises(unittest.TestCase):
    """A typo used to leave fct_unsc as the empty placeholder, silently."""

    def test_an_unknown_correction_method_base_raises(self):
        from diive.flux.lowres.selfheating import ScopPhysics
        idx = pd.date_range('2023-06-01 00:15', periods=48, freq='30min')
        physics = ScopPhysics(
            ta=pd.Series(15.0, index=idx), gas_density=pd.Series(1.7e7, index=idx),
            rho_a=pd.Series(1.2, index=idx), rho_v=pd.Series(0.012, index=idx),
            u=pd.Series(2.5, index=idx), c_p=pd.Series(1005.0, index=idx),
            ustar=pd.Series(0.4, index=idx), lat=47.5, lon=8.4, utc_offset=1)
        with self.assertRaises(ValueError) as ctx:
            physics.run(correction_method_base="BUR99", gapfill=False)
        self.assertIn('BUR99', str(ctx.exception))


class TestSkippedClassesAreReported(unittest.TestCase):
    """A class too small to fit takes a neighbour's factor - so it must be named."""

    def test_a_small_class_is_recorded_and_reported(self):
        from diive.core.utils.console import console
        from diive.flux.lowres.selfheating import ScopOptimizer
        n = 400
        idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min')
        rng = np.random.RandomState(0)
        # Four equal quantile classes; the top one is left with 3 complete records,
        # below MIN_ROWS_PER_CLASS, so it cannot be fitted.
        fct = pd.Series(rng.normal(1, 0.1, n), index=idx)
        fct.iloc[300:397] = np.nan
        opt = ScopOptimizer(
            class_var=pd.Series(np.linspace(0, 1, n), index=idx), n_classes=4,
            fct_unsc=fct, daytime=pd.Series(1, index=idx), n_bootstrap_runs=0,
            flux_openpath=pd.Series(rng.normal(-5, 1, n), index=idx),
            flux_closedpath=pd.Series(rng.normal(-5, 1, n), index=idx))
        with console.capture() as cap:
            opt.run()
        self.assertEqual(len(opt.skipped_classes), 1)
        daytime, bin_id, n_valid, n_total = opt.skipped_classes[0]
        self.assertEqual((n_valid, n_total), (3, 100))
        # The class is named on the console, not merely dropped.
        self.assertIn('scaling factor', cap.get())
        self.assertIn('3 complete of 100', cap.get())
        # And it really is absent from the fitted table.
        self.assertEqual(len(opt.scaling_factors_df), 3)


class TestMeasuredFluxSurvivesAMissingCorrection(unittest.TestCase):
    """`flux + NaN` used to delete a real measurement from the deliverable."""

    def test_a_record_without_a_correction_is_carried_through_and_flagged(self):
        from diive.flux.lowres.selfheating import ScopApplicator
        n = 96
        idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min')
        rng = np.random.RandomState(0)
        fct_unsc = pd.Series(rng.normal(1.0, 0.1, n), index=idx, name='FCT_UNSC_gfXG')
        fct_unsc.iloc[10:20] = np.nan  # no correction term for these records
        flux = pd.Series(rng.normal(-5, 1, n), index=idx, name='NEE')
        classvar = pd.Series(0.4, index=idx, name='USTAR')
        daytime = pd.Series(np.tile([1] * 24 + [0] * 24, n // 48), index=idx, name='DAYTIME')
        sf = pd.DataFrame({'DAYTIME': [0, 1], 'GROUP_CLASSVAR': [0, 0],
                           'GROUP_CLASSVAR_MIN': [0.0, 0.0], 'GROUP_CLASSVAR_MAX': [1.0, 1.0],
                           'SF_MEDIAN': [2.0, 2.0]})
        app = ScopApplicator(fct_unsc=fct_unsc, scaling_factors_df=sf, flux_openpath=flux,
                             classvar=classvar, daytime=daytime)
        app.run()
        corrected = app.df[app.col_flux_corr]
        flag = app.df[app.col_flux_corr_flag]
        # Every measured record survives.
        self.assertEqual(int(corrected.isna().sum()), int(flux.isna().sum()))
        # The uncorrected ones are carried through unchanged and flagged 0.
        carried = flag == 0
        self.assertGreater(int(carried.sum()), 0)
        np.testing.assert_allclose(corrected[carried].to_numpy(), flux[carried].to_numpy())
        self.assertTrue((flag[~carried & flag.notna()] == 1).all())


class TestApplicatorAcceptsAnyInputSeriesName(unittest.TestCase):
    """The applicator used to demand two exact, undocumented series names.

    `run()` looked up one hardcoded canonical name although __init__ stored the
    correction term under `fct_unsc.name`, and the `merge_asof` keyed on
    `daytime.name` while the scaling-factors table always names that column
    'DAYTIME'. So `ScopPhysics.run(gapfill=False)` output ('FCT_UNSC'), the
    gap-filled series and any day/night flag not called 'DAYTIME' each raised
    KeyError before a single record was corrected.
    """

    def _inputs(self):
        n = 96
        idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min')
        rng = np.random.RandomState(0)
        return dict(
            fct_unsc=pd.Series(rng.normal(1.0, 0.1, n), index=idx),
            flux_openpath=pd.Series(rng.normal(-5, 1, n), index=idx, name='NEE'),
            classvar=pd.Series(0.4, index=idx, name='USTAR'),
            daytime=pd.Series(np.tile([1] * 24 + [0] * 24, n // 48), index=idx),
            scaling_factors_df=pd.DataFrame(
                {'DAYTIME': [0, 1], 'GROUP_CLASSVAR': [0, 0],
                 'GROUP_CLASSVAR_MIN': [0.0, 0.0], 'GROUP_CLASSVAR_MAX': [1.0, 1.0],
                 'SF_MEDIAN': [2.0, 2.0]}))

    def _corrected(self, fct_name, daytime_name):
        from diive.flux.lowres.selfheating import ScopApplicator
        args = self._inputs()
        args['fct_unsc'] = args['fct_unsc'].rename(fct_name)
        args['daytime'] = args['daytime'].rename(daytime_name)
        app = ScopApplicator(**args)
        app.run()
        return app.df[app.col_flux_corr]

    def test_the_correction_term_may_carry_any_name(self):
        # 'FCT_UNSC_gfXG' = .fct_unsc_gf and the results-dataframe column used by the
        # examples, 'FCT_UNSC' = ScopPhysics.run(gapfill=False), 'FCT_UNSC_gfRF' = the
        # pre-v0.91.0 column name an old script may still be carrying around.
        reference = self._corrected('FCT_UNSC_gfXG', 'DAYTIME')
        self.assertGreater(int(reference.notna().sum()), 0)
        for name in ('FCT_UNSC', 'FCT_UNSC_gfRF'):
            with self.subTest(fct_unsc=name):
                np.testing.assert_allclose(self._corrected(name, 'DAYTIME').to_numpy(),
                                           reference.to_numpy())

    def test_the_daytime_flag_may_carry_any_name(self):
        reference = self._corrected('FCT_UNSC_gfXG', 'DAYTIME')
        np.testing.assert_allclose(self._corrected('FCT_UNSC_gfXG', 'DAYTIME_FLAG').to_numpy(),
                                   reference.to_numpy())


class TestApplicatorDoesNotClaimAGapFill(unittest.TestCase):
    """The applicator's own column must not credit a gap-fill it never performed.

    `__init__` renames the input correction term to one canonical name (L39, so that
    any legal input name is accepted). That target used to be `ColumnConfig.fct_unsc_gf`
    ('FCT_UNSC_gfXG'), so the results frame labelled an ungapfilled input as gap-filled -
    the applicator gap-fills nothing. The target is now the neutral `ColumnConfig.fct_unsc`.
    """

    def _applicator(self, fct_name):
        from diive.flux.lowres.selfheating import ScopApplicator
        n = 96
        idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min')
        rng = np.random.RandomState(0)
        fct = pd.Series(rng.normal(1.0, 0.1, n), index=idx, name=fct_name)
        fct.iloc[10:20] = np.nan  # never gap-filled by anyone
        app = ScopApplicator(
            fct_unsc=fct,
            scaling_factors_df=pd.DataFrame(
                {'DAYTIME': [0, 1], 'GROUP_CLASSVAR': [0, 0],
                 'GROUP_CLASSVAR_MIN': [0.0, 0.0], 'GROUP_CLASSVAR_MAX': [1.0, 1.0],
                 'SF_MEDIAN': [2.0, 2.0]}),
            flux_openpath=pd.Series(rng.normal(-5, 1, n), index=idx, name='NEE'),
            classvar=pd.Series(0.4, index=idx, name='USTAR'),
            daytime=pd.Series(np.tile([1] * 24 + [0] * 24, n // 48), index=idx, name='DAYTIME'))
        app.run()
        return app, fct

    def test_the_ungapfilled_input_is_not_labelled_gap_filled(self):
        from diive.flux.lowres.selfheating import ColumnConfig
        app, fct = self._applicator('FCT_UNSC')
        results = app.get_results()
        self.assertIn(ColumnConfig().fct_unsc, results.columns)
        self.assertNotIn(ColumnConfig().fct_unsc_gf, results.columns)
        # And the column really is the input term, gaps included - nothing was filled.
        np.testing.assert_allclose(results[ColumnConfig().fct_unsc].to_numpy(),
                                   fct.to_numpy())
        self.assertGreater(int(results[ColumnConfig().fct_unsc].isna().sum()), 0)

    def test_a_gapfilled_input_lands_under_the_same_neutral_name(self):
        # L39: any input name is accepted, and there is exactly one internal name.
        from diive.flux.lowres.selfheating import ColumnConfig
        app, _ = self._applicator('FCT_UNSC_gfXG')
        self.assertIn(ColumnConfig().fct_unsc, app.get_results().columns)
        self.assertNotIn(ColumnConfig().fct_unsc_gf, app.get_results().columns)
        self.assertEqual(app.fct_unsc.name, ColumnConfig().fct_unsc)


class TestPlotsCanBeSilenced(unittest.TestCase):
    """The three SCOP plots used to call plt.show() unconditionally.

    An example could therefore not satisfy the 'disable showplot=True' standard, and
    every call leaked a figure the caller had no handle on (the dashboard is 24x20 in).
    Each plot now takes showplot and returns its figure.
    """

    def setUp(self):
        import matplotlib.pyplot as plt
        self._plt = plt
        self._shown = []
        self._real_show = plt.show
        plt.show = lambda *a, **kw: self._shown.append(1)

    def tearDown(self):
        self._plt.show = self._real_show
        self._plt.close('all')

    def _physics(self):
        from diive.flux.lowres.selfheating import ScopPhysics
        n = 480
        idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min', name='TIMESTAMP_MIDDLE')
        hr = idx.hour + idx.minute / 60
        physics = ScopPhysics(
            ta=pd.Series(15 + 8 * np.sin(2 * np.pi * (hr - 9) / 24), index=idx),
            gas_density=pd.Series(1.7e7, index=idx), rho_a=pd.Series(1.2, index=idx),
            rho_v=pd.Series(0.012, index=idx), u=pd.Series(2.5, index=idx),
            c_p=pd.Series(1005.0, index=idx), ustar=pd.Series(0.4, index=idx),
            lat=47.478333, lon=8.364389, utc_offset=1)
        physics.run(correction_method_base="JAR09", gapfill=False)
        return physics

    def _optimizer(self):
        from diive.flux.lowres.selfheating import ScopOptimizer
        n = 400
        idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min')
        rng = np.random.RandomState(0)
        opt = ScopOptimizer(
            class_var=pd.Series(np.linspace(0, 1, n), index=idx), n_classes=2,
            fct_unsc=pd.Series(rng.normal(1, 0.1, n), index=idx),
            daytime=pd.Series(((idx.hour >= 6) & (idx.hour < 18)).astype(int), index=idx),
            n_bootstrap_runs=0,
            flux_openpath=pd.Series(rng.normal(-5, 1, n), index=idx),
            flux_closedpath=pd.Series(rng.normal(-5, 1, n), index=idx))
        opt.run()
        return opt

    def _applicator(self):
        from diive.flux.lowres.selfheating import ScopApplicator
        n = 96
        idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min')
        rng = np.random.RandomState(0)
        app = ScopApplicator(
            fct_unsc=pd.Series(rng.normal(1.0, 0.1, n), index=idx),
            scaling_factors_df=pd.DataFrame(
                {'DAYTIME': [0, 1], 'GROUP_CLASSVAR': [0, 0],
                 'GROUP_CLASSVAR_MIN': [0.0, 0.0], 'GROUP_CLASSVAR_MAX': [1.0, 1.0],
                 'SF_MEDIAN': [2.0, 2.0]}),
            flux_openpath=pd.Series(rng.normal(-5, 1, n), index=idx, name='NEE'),
            classvar=pd.Series(0.4, index=idx, name='USTAR'),
            daytime=pd.Series(np.tile([1] * 24 + [0] * 24, n // 48), index=idx))
        app.run()
        return app

    def _check(self, plotfunc):
        from matplotlib.figure import Figure
        fig = plotfunc(showplot=False)
        self.assertIsInstance(fig, Figure)
        self.assertEqual(self._shown, [], 'showplot=False still called plt.show()')
        # The flag must be wired both ways, not merely off.
        plotfunc(showplot=True)
        self.assertEqual(len(self._shown), 1)

    def test_physics_diel_cycles(self):
        self._check(self._physics().plot_diel_cycles)

    def test_optimizer_scaling_factors(self):
        self._check(self._optimizer().plot)

    def test_applicator_dashboard(self):
        self._check(self._applicator().plot_dashboard)


if __name__ == '__main__':
    unittest.main()
