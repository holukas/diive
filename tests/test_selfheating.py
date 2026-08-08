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


class TestGapFillFillsTheGaps(unittest.TestCase):
    """The gap-fill must fill; it used to delete the gaps and report success.

    `pd.DataFrame.from_dict(frame).dropna()` dropped every row with a NaN in *any*
    column, the target included - so exactly the records to be filled left the
    training frame. XGBoost then reported "Filling 0 missing records" and the
    console blamed "insufficient drivers".
    """

    def test_gaps_in_the_correction_term_are_filled(self):
        from diive.flux.lowres.selfheating import ScopPhysics
        n = 480
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
        before = int(physics.fct_unsc.isna().sum())
        after = int(physics.fct_unsc_gf.isna().sum())
        self.assertGreater(before, 0, 'the fixture must produce gaps')
        self.assertLess(after, before)


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
        fct_unsc = pd.Series(rng.normal(1.0, 0.1, n), index=idx, name='FCT_UNSC_gfRF')
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


if __name__ == '__main__':
    unittest.main()
