import unittest

import numpy as np
import pandas as pd


class TestNighttimePartitioning(unittest.TestCase):
    """Tests for the nighttime NEE partitioning (Reichstein et al. 2005)."""

    @classmethod
    def setUpClass(cls):
        import diive as dv
        df = dv.load_exampledata_parquet()
        # Two years keep the run fast while exercising the per-year loop.
        cls.df = df.loc[df.index.year.isin([2017, 2018])].copy()
        cls.lat = 46.815

    def _run(self):
        from diive.flux.partitioning import NighttimePartitioning
        df = self.df
        part = NighttimePartitioning(
            nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
            nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'], lat=self.lat, verbose=0)
        return part.run()

    def test_lloyd_taylor_reference_value(self):
        from diive.flux.partitioning import lloyd_taylor
        # At the reference temperature, respiration equals rref.
        self.assertAlmostEqual(float(lloyd_taylor(np.array([15.0]), rref=2.0, e0=200.0)[0]),
                               2.0, places=6)
        # Respiration increases with temperature.
        warm = lloyd_taylor(np.array([20.0]), rref=2.0, e0=200.0)[0]
        cold = lloyd_taylor(np.array([5.0]), rref=2.0, e0=200.0)[0]
        self.assertGreater(warm, cold)

    def test_sunrise_sunset_ordering(self):
        from diive.flux.partitioning import sunrise_sunset
        sunrise, sunset = sunrise_sunset(np.array([172]), lat=46.815)  # ~summer solstice
        self.assertLess(sunrise[0], sunset[0])
        self.assertLess(sunrise[0], 12.0)
        self.assertGreater(sunset[0], 12.0)

    def test_results_shape_and_columns(self):
        part = self._run()
        res = part.results
        self.assertEqual(len(res), len(self.df))
        for col in ['NEE_NIGHT', 'RECO_NT', 'RECO_NT_ROB',
                    'GPP_NT', 'GPP_NT_ROB', 'RREF_NT', 'E0_NT']:
            self.assertIn(col, res.columns)
        self.assertTrue(res.index.equals(self.df.index))

    def test_reco_is_positive_and_filled(self):
        part = self._run()
        reco = part.results['RECO_NT']
        # RECO should be filled for most records (tair_f is gap-filled).
        self.assertGreater(reco.notna().sum(), 0.9 * len(reco))
        # Respiration is a positive flux.
        self.assertTrue((reco.dropna() > 0).all())

    def test_e0_within_physical_range(self):
        part = self._run()
        e0 = part.results['E0_NT'].dropna().unique()
        self.assertTrue(len(e0) >= 1)
        for val in e0:
            self.assertGreater(val, 0.0)
            self.assertLess(val, 450.0)

    def test_matches_fluxnet_reference(self):
        part = self._run()
        res = part.results
        reco_ref = self.df['Reco_CUT_REF']
        m = res['RECO_NT'].notna() & reco_ref.notna()
        corr = np.corrcoef(res['RECO_NT'][m], reco_ref[m])[0, 1]
        # Faithful port: strong correlation with FLUXNET-produced RECO.
        self.assertGreater(corr, 0.9)
        # Means should be close.
        self.assertLess(abs(res['RECO_NT'][m].mean() - reco_ref[m].mean()), 0.5)

    def test_gpp_definition(self):
        part = self._run()
        res = part.results
        # GPP = RECO - NEE_f wherever both are present.
        m = res['GPP_NT'].notna() & res['RECO_NT'].notna()
        expected = res['RECO_NT'][m] - self.df['NEE_CUT_REF_f'][m]
        np.testing.assert_allclose(res['GPP_NT'][m].to_numpy(),
                                   expected.to_numpy(), rtol=1e-6, atol=1e-6)

    def test_functional_wrapper(self):
        from diive.flux.partitioning import partition_nee_nighttime
        df = self.df
        res = partition_nee_nighttime(
            nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
            nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'], lat=self.lat, verbose=0)
        self.assertEqual(len(res), len(df))
        self.assertIn('RECO_NT', res.columns)

    def test_requires_datetime_index(self):
        from diive.flux.partitioning import NighttimePartitioning
        bad = pd.Series([1.0, 2.0, 3.0])  # RangeIndex, not DatetimeIndex
        with self.assertRaises(TypeError):
            NighttimePartitioning(nee=bad, ta=bad, sw_in=bad, nee_f=bad,
                                  ta_f=bad, lat=self.lat, verbose=0)

    def test_results_before_run_raises(self):
        from diive.flux.partitioning import NighttimePartitioning
        df = self.df
        part = NighttimePartitioning(
            nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
            nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'], lat=self.lat, verbose=0)
        with self.assertRaises(RuntimeError):
            _ = part.results


if __name__ == '__main__':
    unittest.main()
