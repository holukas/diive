import unittest

import numpy as np
import pandas as pd


class TestDaytimePartitioningReddyProc(unittest.TestCase):
    """Tests for the REddyProc daytime NEE partitioning (Lasslop et al. 2010)."""

    @classmethod
    def setUpClass(cls):
        import diive as dv
        from diive.flux.partitioning import DaytimePartitioningReddyProc
        # The daytime LRC fit is per-window and somewhat heavy, so run a single
        # year once here and reuse across tests.
        df = dv.load_exampledata_parquet()
        cls.df = df.loc[df.index.year == 2017].copy()
        cls.lat, cls.lon, cls.utc_offset = 46.815, 9.855, 1
        cls.part = DaytimePartitioningReddyProc(
            nee=cls.df['NEE_CUT_REF_orig'], ta=cls.df['Tair_f'],
            vpd=cls.df['VPD_f'], sw_in=cls.df['Rg_f'],
            lat=cls.lat, lon=cls.lon, utc_offset=cls.utc_offset, verbose=0).run()

    def _run(self):
        return self.part

    def test_results_shape_and_columns(self):
        res = self._run().results
        self.assertEqual(len(res), len(self.df))
        for col in ['RECO_DT_RP', 'GPP_DT_RP', 'K_DT_RP', 'BETA_DT_RP',
                    'ALPHA_DT_RP', 'RREF_DT_RP', 'E0_DT_RP']:
            self.assertIn(col, res.columns)
        self.assertTrue(res.index.equals(self.df.index))

    def test_reco_positive_and_filled(self):
        reco = self._run().results['RECO_DT_RP']
        # Respiration is a positive flux wherever it is computed.
        self.assertTrue((reco.dropna() > 0).all())
        # Daytime method predicts for essentially every record.
        self.assertGreater(reco.notna().sum(), 0.95 * len(reco))

    def test_gpp_filled_and_nonnegative_daytime(self):
        gpp = self._run().results['GPP_DT_RP']
        self.assertGreater(gpp.notna().sum(), 0.95 * len(gpp))
        # GPP is essentially non-negative (tiny negatives possible at edges).
        self.assertGreater((gpp.dropna() >= -0.5).mean(), 0.99)

    def test_lrc_params_reported_sparsely(self):
        # LRC parameters are reported once per window (at the central record),
        # so they are present for only a small fraction of records.
        res = self._run().results
        for col in ['K_DT_RP', 'BETA_DT_RP', 'ALPHA_DT_RP', 'RREF_DT_RP', 'E0_DT_RP']:
            frac = res[col].notna().mean()
            self.assertLess(frac, 0.1)
            self.assertGreater(res[col].notna().sum(), 0)
        # E0 stays within the nighttime bounds [50, 400].
        e0 = res['E0_DT_RP'].dropna()
        self.assertTrue((e0 >= 50).all() and (e0 <= 400).all())

    def test_matches_reddyproc_reference(self):
        # The bundled CH-DAV columns are REddyProc daytime output, but computed
        # with the measured NEE uncertainty (not shipped) and the full record,
        # so this is a provenance-limited sanity check, not a 1:1 target. GPP
        # tracks closely; daytime RECO is more sensitive (a documented bias).
        res = self._run().results
        gpp_ref = self.df['GPP_DT_CUT_REF']
        reco_ref = self.df['Reco_DT_CUT_REF']
        mg = res['GPP_DT_RP'].notna() & gpp_ref.notna()
        mr = res['RECO_DT_RP'].notna() & reco_ref.notna()
        self.assertGreater(np.corrcoef(res['GPP_DT_RP'][mg], gpp_ref[mg])[0, 1], 0.9)
        self.assertGreater(np.corrcoef(res['RECO_DT_RP'][mr], reco_ref[mr])[0, 1], 0.6)

    def test_vpd_units_handling(self):
        # Passing VPD already in hPa (vpd_in_kpa=False) with a /10 series must
        # reproduce the default kPa path.
        from diive.flux.partitioning import DaytimePartitioningReddyProc
        res_hpa = DaytimePartitioningReddyProc(
            nee=self.df['NEE_CUT_REF_orig'], ta=self.df['Tair_f'],
            vpd=self.df['VPD_f'] * 10.0, sw_in=self.df['Rg_f'],
            lat=self.lat, lon=self.lon, utc_offset=self.utc_offset,
            vpd_in_kpa=False, verbose=0).run().results
        a = res_hpa['GPP_DT_RP'].to_numpy()
        b = self._run().results['GPP_DT_RP'].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(a[m], b[m], rtol=1e-9, atol=1e-9)

    def test_functional_wrapper(self):
        from diive.flux.partitioning import partition_nee_daytime_reddyproc
        res = partition_nee_daytime_reddyproc(
            nee=self.df['NEE_CUT_REF_orig'], ta=self.df['Tair_f'],
            vpd=self.df['VPD_f'], sw_in=self.df['Rg_f'],
            lat=self.lat, lon=self.lon, utc_offset=self.utc_offset, verbose=0)
        self.assertEqual(len(res), len(self.df))
        self.assertIn('GPP_DT_RP', res.columns)

    def test_requires_datetime_index(self):
        from diive.flux.partitioning import DaytimePartitioningReddyProc
        bad = pd.Series([1.0, 2.0, 3.0])  # RangeIndex, not DatetimeIndex
        with self.assertRaises(TypeError):
            DaytimePartitioningReddyProc(
                nee=bad, ta=bad, vpd=bad, sw_in=bad,
                lat=self.lat, lon=self.lon, utc_offset=self.utc_offset, verbose=0)

    def test_results_before_run_raises(self):
        from diive.flux.partitioning import DaytimePartitioningReddyProc
        part = DaytimePartitioningReddyProc(
            nee=self.df['NEE_CUT_REF_orig'], ta=self.df['Tair_f'],
            vpd=self.df['VPD_f'], sw_in=self.df['Rg_f'],
            lat=self.lat, lon=self.lon, utc_offset=self.utc_offset, verbose=0)
        with self.assertRaises(RuntimeError):
            _ = part.results


if __name__ == '__main__':
    unittest.main()
