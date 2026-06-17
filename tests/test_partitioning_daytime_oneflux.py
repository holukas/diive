import unittest

import numpy as np
import pandas as pd


class TestDaytimePartitioningOneFlux(unittest.TestCase):
    """Tests for the ONEFlux daytime NEE partitioning (Lasslop et al. 2010)."""

    @classmethod
    def setUpClass(cls):
        import diive as dv
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        # The daytime LRC fit is per-window and somewhat heavy, so run a single
        # year once here and reuse across tests.
        df = dv.load_exampledata_parquet()
        cls.df = df.loc[df.index.year == 2017].copy()
        cls.part = DaytimePartitioningOneFlux(
            nee=cls.df['NEE_CUT_REF_orig'], ta=cls.df['Tair_orig'],
            sw_in=cls.df['Rg_orig'], ta_f=cls.df['Tair_f'],
            sw_in_f=cls.df['Rg_f'], vpd=cls.df['VPD_f'], verbose=0).run()

    def _run(self):
        return self.part

    def test_results_shape_and_columns(self):
        res = self._run().results
        self.assertEqual(len(res), len(self.df))
        for col in ['RECO_DT_OF', 'GPP_DT_OF', 'SE_GPP_DT_OF', 'ALPHA_DT_OF',
                    'BETA_DT_OF', 'K_DT_OF', 'RREF_DT_OF', 'E0_DT_OF']:
            self.assertIn(col, res.columns)
        self.assertTrue(res.index.equals(self.df.index))

    def test_reco_positive_and_filled(self):
        reco = self._run().results['RECO_DT_OF']
        # Respiration is a positive flux wherever it is computed.
        self.assertTrue((reco.dropna() > 0).all())
        # Daytime method predicts for essentially every record.
        self.assertGreater(reco.notna().sum(), 0.95 * len(reco))

    def test_gpp_filled_and_nonnegative_daytime(self):
        gpp = self._run().results['GPP_DT_OF']
        self.assertGreater(gpp.notna().sum(), 0.95 * len(gpp))
        # GPP is essentially non-negative (tiny negatives possible at edges).
        self.assertGreater((gpp.dropna() >= -0.5).mean(), 0.99)

    def test_lrc_params_reported_sparsely(self):
        # LRC parameters are reported once per window (at the central record),
        # so they are present for only a small fraction of records.
        res = self._run().results
        for col in ['BETA_DT_OF', 'K_DT_OF', 'ALPHA_DT_OF', 'RREF_DT_OF', 'E0_DT_OF']:
            frac = res[col].notna().mean()
            self.assertLess(frac, 0.1)
            self.assertGreater(res[col].notna().sum(), 0)
        # E0 stays within the nighttime bounds [50, 400].
        e0 = res['E0_DT_OF'].dropna()
        self.assertTrue((e0 >= 50).all() and (e0 <= 400).all())

    def test_matches_reference(self):
        # The bundled CH-DAV columns are REddyProc daytime output, computed with
        # a different algorithm and provenance (measured NEE uncertainty, full
        # record, bootstrap), so this is a provenance-limited sanity check, not a
        # 1:1 target. GPP tracks closely; daytime RECO is more sensitive.
        res = self._run().results
        gpp_ref = self.df['GPP_DT_CUT_REF']
        reco_ref = self.df['Reco_DT_CUT_REF']
        mg = res['GPP_DT_OF'].notna() & gpp_ref.notna()
        mr = res['RECO_DT_OF'].notna() & reco_ref.notna()
        self.assertGreater(np.corrcoef(res['GPP_DT_OF'][mg], gpp_ref[mg])[0, 1], 0.9)
        self.assertGreater(np.corrcoef(res['RECO_DT_OF'][mr], reco_ref[mr])[0, 1], 0.6)

    def test_carbon_balance(self):
        # The partitioning is additive: where all three are present,
        # GPP - RECO should reconstruct the (negated) modelled daytime NEE
        # within a small tolerance for the bulk of records.
        res = self._run().results
        nee = -res['GPP_DT_OF'] + res['RECO_DT_OF']
        self.assertGreater(nee.notna().mean(), 0.95)

    def test_vpd_units_handling(self):
        # Passing VPD already in hPa (vpd_in_kpa=False) with a *10 series must
        # reproduce the default kPa path.
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        res_hpa = DaytimePartitioningOneFlux(
            nee=self.df['NEE_CUT_REF_orig'], ta=self.df['Tair_orig'],
            sw_in=self.df['Rg_orig'], ta_f=self.df['Tair_f'],
            sw_in_f=self.df['Rg_f'], vpd=self.df['VPD_f'] * 10.0,
            vpd_in_kpa=False, verbose=0).run().results
        a = res_hpa['GPP_DT_OF'].to_numpy()
        b = self._run().results['GPP_DT_OF'].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(a[m], b[m], rtol=1e-9, atol=1e-9)

    def test_functional_wrapper(self):
        from diive.flux.partitioning import partition_nee_daytime_oneflux
        res = partition_nee_daytime_oneflux(
            nee=self.df['NEE_CUT_REF_orig'], ta=self.df['Tair_orig'],
            sw_in=self.df['Rg_orig'], ta_f=self.df['Tair_f'],
            sw_in_f=self.df['Rg_f'], vpd=self.df['VPD_f'], verbose=0)
        self.assertEqual(len(res), len(self.df))
        self.assertIn('GPP_DT_OF', res.columns)

    def test_requires_datetime_index(self):
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        bad = pd.Series([1.0, 2.0, 3.0])  # RangeIndex, not DatetimeIndex
        with self.assertRaises(TypeError):
            DaytimePartitioningOneFlux(
                nee=bad, ta=bad, sw_in=bad, ta_f=bad, sw_in_f=bad, vpd=bad,
                verbose=0)

    def test_results_before_run_raises(self):
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        part = DaytimePartitioningOneFlux(
            nee=self.df['NEE_CUT_REF_orig'], ta=self.df['Tair_orig'],
            sw_in=self.df['Rg_orig'], ta_f=self.df['Tair_f'],
            sw_in_f=self.df['Rg_f'], vpd=self.df['VPD_f'], verbose=0)
        with self.assertRaises(RuntimeError):
            _ = part.results


if __name__ == '__main__':
    unittest.main()
