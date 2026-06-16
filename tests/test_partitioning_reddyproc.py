import unittest

import numpy as np
import pandas as pd


class TestNighttimePartitioningReddyProc(unittest.TestCase):
    """Tests for the REddyProc nighttime NEE partitioning (Reichstein et al. 2005)."""

    @classmethod
    def setUpClass(cls):
        import diive as dv
        from diive.flux.partitioning import NighttimePartitioningReddyProc
        # REddyProc partitions the whole record at once with a single E0, so the
        # ReddyProc-derived reference columns only match a full-record run. The
        # full 10 years run in ~1 s, so partition once here and reuse.
        cls.df = dv.load_exampledata_parquet()
        cls.lat = 46.815
        cls.lon = 9.855
        cls.utc_offset = 1
        cls.part = NighttimePartitioningReddyProc(
            nee=cls.df['NEE_CUT_REF_orig'], ta=cls.df['Tair_orig'],
            sw_in=cls.df['Rg_orig'], nee_f=cls.df['NEE_CUT_REF_f'],
            ta_f=cls.df['Tair_f'], lat=cls.lat, lon=cls.lon,
            utc_offset=cls.utc_offset, verbose=0).run()

    def _run(self):
        return self.part

    def test_lloyd_taylor_kelvin_reference_value(self):
        from diive.flux.partitioning import lloyd_taylor_kelvin
        # At the reference temperature (288.15 K), respiration equals rref.
        self.assertAlmostEqual(
            float(lloyd_taylor_kelvin(np.array([288.15]), rref=2.0, e0=200.0)[0]),
            2.0, places=6)
        # Respiration increases with temperature.
        warm = lloyd_taylor_kelvin(np.array([293.15]), rref=2.0, e0=200.0)[0]
        cold = lloyd_taylor_kelvin(np.array([278.15]), rref=2.0, e0=200.0)[0]
        self.assertGreater(warm, cold)

    def test_potential_radiation_day_night(self):
        from diive.flux.partitioning import potential_radiation
        # Around local solar noon in summer the sun is up -> positive potrad.
        noon = potential_radiation(np.array([172]), np.array([12.0]),
                                   lat=self.lat, lon=self.lon, utc_offset=1)[0]
        # At midnight the sun is below the horizon -> zero potential radiation.
        midnight = potential_radiation(np.array([172]), np.array([0.0]),
                                       lat=self.lat, lon=self.lon, utc_offset=1)[0]
        self.assertGreater(noon, 0.0)
        self.assertEqual(midnight, 0.0)

    def test_results_shape_and_columns(self):
        part = self._run()
        res = part.results
        self.assertEqual(len(res), len(self.df))
        for col in ['NEE_NIGHT_RP', 'RECO_NT_RP', 'GPP_NT_RP', 'RREF_NT_RP', 'E0_NT_RP']:
            self.assertIn(col, res.columns)
        # REddyProc has no outlier-robust variant -> no *_ROB columns.
        self.assertNotIn('RECO_NT_RP_ROB', res.columns)
        self.assertTrue(res.index.equals(self.df.index))

    def test_reco_is_positive_and_filled(self):
        part = self._run()
        reco = part.results['RECO_NT_RP']
        # Respiration is a positive flux wherever it is computed.
        self.assertTrue((reco.dropna() > 0).all())
        # Whole-record processing with gap-filled temperature -> essentially all
        # records are partitioned.
        self.assertGreater(reco.notna().sum(), 0.95 * len(reco))

    def test_single_e0_for_whole_record(self):
        part = self._run()
        e0 = part.results['E0_NT_RP'].dropna().unique()
        # REddyProc estimates exactly one E0 for the entire series.
        self.assertEqual(len(e0), 1)
        self.assertGreater(e0[0], 30.0)
        self.assertLess(e0[0], 350.0)

    def test_matches_reddyproc_reference(self):
        part = self._run()
        res = part.results
        reco_ref = self.df['Reco_CUT_REF']
        gpp_ref = self.df['GPP_CUT_REF_f']
        # These reference columns are themselves REddyProc-derived, so this is a
        # genuine 1:1 parity target (not just a sanity check).
        m = res['RECO_NT_RP'].notna() & reco_ref.notna()
        self.assertGreater(np.corrcoef(res['RECO_NT_RP'][m], reco_ref[m])[0, 1], 0.99)
        self.assertLess(abs(res['RECO_NT_RP'][m].mean() - reco_ref[m].mean()), 0.25)
        mg = res['GPP_NT_RP'].notna() & gpp_ref.notna()
        self.assertGreater(np.corrcoef(res['GPP_NT_RP'][mg], gpp_ref[mg])[0, 1], 0.99)

    def test_gpp_definition(self):
        part = self._run()
        res = part.results
        m = res['GPP_NT_RP'].notna() & res['RECO_NT_RP'].notna()
        expected = res['RECO_NT_RP'][m] - self.df['NEE_CUT_REF_f'][m]
        np.testing.assert_allclose(res['GPP_NT_RP'][m].to_numpy(),
                                   expected.to_numpy(), rtol=1e-6, atol=1e-6)

    def test_abort_when_no_temperature_range(self):
        from diive.flux.partitioning import NighttimePartitioningReddyProc
        # Constant temperature -> no window reaches the temperature-range
        # threshold -> no valid E0 -> REddyProc aborts -> all NaN.
        idx = pd.date_range('2020-01-01 00:15', '2020-12-31 23:45', freq='30min')
        n = len(idx)
        rng = np.random.default_rng(0)
        nee = pd.Series(rng.normal(2.0, 0.5, n), index=idx)
        ta = pd.Series(np.full(n, 10.0), index=idx)
        sw_in = pd.Series(np.zeros(n), index=idx)
        part = NighttimePartitioningReddyProc(
            nee=nee, ta=ta, sw_in=sw_in, nee_f=nee, ta_f=ta,
            lat=self.lat, lon=self.lon, utc_offset=self.utc_offset, verbose=0).run()
        res = part.results
        self.assertEqual(res['RECO_NT_RP'].notna().sum(), 0)
        self.assertEqual(res['GPP_NT_RP'].notna().sum(), 0)
        self.assertTrue(res['E0_NT_RP'].isna().all())

    def test_functional_wrapper(self):
        from diive.flux.partitioning import partition_nee_nighttime_reddyproc
        df = self.df
        res = partition_nee_nighttime_reddyproc(
            nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
            nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'],
            lat=self.lat, lon=self.lon, utc_offset=self.utc_offset, verbose=0)
        self.assertEqual(len(res), len(df))
        self.assertIn('RECO_NT_RP', res.columns)

    def test_requires_datetime_index(self):
        from diive.flux.partitioning import NighttimePartitioningReddyProc
        bad = pd.Series([1.0, 2.0, 3.0])  # RangeIndex, not DatetimeIndex
        with self.assertRaises(TypeError):
            NighttimePartitioningReddyProc(
                nee=bad, ta=bad, sw_in=bad, nee_f=bad, ta_f=bad,
                lat=self.lat, lon=self.lon, utc_offset=self.utc_offset, verbose=0)

    def test_results_before_run_raises(self):
        from diive.flux.partitioning import NighttimePartitioningReddyProc
        df = self.df
        part = NighttimePartitioningReddyProc(
            nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
            nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'],
            lat=self.lat, lon=self.lon, utc_offset=self.utc_offset, verbose=0)
        with self.assertRaises(RuntimeError):
            _ = part.results


if __name__ == '__main__':
    unittest.main()
