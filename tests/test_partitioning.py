import unittest
from unittest import mock

import numpy as np
import pandas as pd


class TestNighttimePartitioningOneFlux(unittest.TestCase):
    """Tests for the nighttime NEE partitioning (Reichstein et al. 2005)."""

    @classmethod
    def setUpClass(cls):
        import diive as dv
        df = dv.load_exampledata_parquet()
        # Two years keep the run fast while exercising the per-year loop.
        cls.df = df.loc[df.index.year.isin([2017, 2018])].copy()
        cls.lat = 46.815

    def _run(self):
        from diive.flux.partitioning import NighttimePartitioningOneFlux
        df = self.df
        part = NighttimePartitioningOneFlux(
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
        for col in ['NEE_NIGHT_OF', 'RECO_NT_OF', 'RECO_NT_OF_ROB',
                    'GPP_NT_OF', 'GPP_NT_OF_ROB', 'RREF_NT_OF', 'E0_NT_OF']:
            self.assertIn(col, res.columns)
        self.assertTrue(res.index.equals(self.df.index))

    def test_reco_is_positive_and_filled(self):
        part = self._run()
        reco = part.results['RECO_NT_OF']
        # Respiration is a positive flux wherever it is computed.
        self.assertTrue((reco.dropna() > 0).all())
        # ONEFlux parity: each year is either fully partitioned (~all records,
        # since tair_f is gap-filled) or, when no short-term E0 is
        # well-constrained, left entirely unpartitioned by the E0 quality gate.
        partitioned_years = 0
        for _, g in reco.groupby(reco.index.year):
            filled = g.notna().sum()
            if filled == 0:
                continue  # year gated out (e.g. 2018 at CH-DAV)
            self.assertGreater(filled, 0.9 * len(g))
            partitioned_years += 1
        self.assertGreater(partitioned_years, 0)

    def test_gate_unconstrained_year_returns_all_nan(self):
        from diive.flux.partitioning import NighttimePartitioningOneFlux
        # A year with no temperature range yields no well-constrained short-term
        # E0, so the ONEFlux E0 quality gate leaves it entirely unpartitioned.
        idx = pd.date_range('2020-01-01 00:15', '2020-12-31 23:45', freq='30min')
        n = len(idx)
        rng = np.random.default_rng(0)
        nee = pd.Series(rng.normal(2.0, 0.5, n), index=idx)  # noise, no T signal
        ta = pd.Series(np.full(n, 10.0), index=idx)          # constant temperature
        sw_in = pd.Series(np.zeros(n), index=idx)            # day/night from sun angle
        part = NighttimePartitioningOneFlux(nee=nee, ta=ta, sw_in=sw_in, nee_f=nee,
                                     ta_f=ta, lat=self.lat, verbose=0).run()
        res = part.results
        self.assertEqual(res['RECO_NT_OF'].notna().sum(), 0)
        self.assertEqual(res['GPP_NT_OF'].notna().sum(), 0)

    def test_e0_within_physical_range(self):
        part = self._run()
        e0 = part.results['E0_NT_OF'].dropna().unique()
        self.assertTrue(len(e0) >= 1)
        for val in e0:
            self.assertGreater(val, 0.0)
            self.assertLess(val, 450.0)

    def test_matches_fluxnet_reference(self):
        part = self._run()
        res = part.results
        reco_ref = self.df['Reco_CUT_REF']
        m = res['RECO_NT_OF'].notna() & reco_ref.notna()
        corr = np.corrcoef(res['RECO_NT_OF'][m], reco_ref[m])[0, 1]
        # Faithful port: strong correlation with FLUXNET-produced RECO.
        self.assertGreater(corr, 0.9)
        # Means should be close.
        self.assertLess(abs(res['RECO_NT_OF'][m].mean() - reco_ref[m].mean()), 0.5)

    def test_gpp_definition(self):
        part = self._run()
        res = part.results
        # GPP = RECO - NEE_f wherever both are present.
        m = res['GPP_NT_OF'].notna() & res['RECO_NT_OF'].notna()
        expected = res['RECO_NT_OF'][m] - self.df['NEE_CUT_REF_f'][m]
        np.testing.assert_allclose(res['GPP_NT_OF'][m].to_numpy(),
                                   expected.to_numpy(), rtol=1e-6, atol=1e-6)

    def test_functional_wrapper(self):
        from diive.flux.partitioning import partition_nee_nighttime_oneflux
        df = self.df
        res = partition_nee_nighttime_oneflux(
            nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
            nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'], lat=self.lat, verbose=0)
        self.assertEqual(len(res), len(df))
        self.assertIn('RECO_NT_OF', res.columns)

    def test_requires_datetime_index(self):
        from diive.flux.partitioning import NighttimePartitioningOneFlux
        bad = pd.Series([1.0, 2.0, 3.0])  # RangeIndex, not DatetimeIndex
        with self.assertRaises(TypeError):
            NighttimePartitioningOneFlux(nee=bad, ta=bad, sw_in=bad, nee_f=bad,
                                  ta_f=bad, lat=self.lat, verbose=0)

    def _run_on(self, df):
        from diive.flux.partitioning import NighttimePartitioningOneFlux
        return NighttimePartitioningOneFlux(
            nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
            nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'], lat=self.lat,
            verbose=0).run().results

    def test_hour_and_day_come_from_the_period_end(self):
        # The index stays MIDDLE, but ONEFlux takes the hour and day of year of
        # the period END, and repairs the year's last record, whose END falls on
        # 1 January, to 366 (367 in a leap year).
        from diive.flux.partitioning import nighttime_oneflux as mod
        seen = []

        def stub(**kwargs):
            seen.append((np.asarray(kwargs['doy']), np.asarray(kwargs['hr'])))
            n = len(kwargs['hr'])
            return {c: np.full(n, np.nan) for c in (
                'NEE_NIGHT_OF', 'RECO_NT_OF', 'RECO_NT_OF_ROB', 'GPP_NT_OF',
                'GPP_NT_OF_ROB', 'RREF_NT_OF', 'E0_NT_OF')}

        with mock.patch.object(mod, '_partition_one_year', side_effect=stub):
            self._run_on(self.df)
        doy, hr = seen[0]  # 2017; the first record is 00:15 MIDDLE
        self.assertEqual(hr[0], 0.5)
        self.assertTrue(np.all(np.mod(hr, 0.5) == 0))
        self.assertEqual(doy[0], 1)
        self.assertEqual(doy[-2], 365)
        self.assertEqual(doy[-1], 366)

    def test_missing_sw_in_does_not_exclude_night_records(self):
        # ONEFlux stores a missing SW_IN as -9999, which passes its `< 10`
        # night test; only the sunrise/sunset flag decides then. Blank SW_IN on
        # measured records around midnight, which are night on any day.
        df = self.df.copy()
        midnight = (df.index.hour == 0) & df['NEE_CUT_REF_orig'].notna()
        df.loc[midnight, 'Rg_orig'] = np.nan
        res = self._run_on(df)
        self.assertGreater(int(midnight.sum()), 0)
        self.assertTrue(res.loc[midnight, 'NEE_NIGHT_OF'].notna().all())

    def test_equinox_sunrise_record_counts_as_night(self):
        # ONEFlux compares the hour against sunrise/sunset in float32. On day 80
        # sunrise is 5.99999990 in float64 but exactly 6.0 in float32, so the
        # record ending 06:00 is not after sunrise and counts as night.
        df = self.df.copy()
        t = pd.Timestamp('2017-03-21 05:45')  # END 06:00 on day 80
        df.loc[t, 'NEE_CUT_REF_orig'] = 1.5
        df.loc[t, 'Rg_orig'] = 0.0
        res = self._run_on(df)
        self.assertEqual(res.loc[t, 'NEE_NIGHT_OF'], 1.5)

    def test_rref_knot_uses_the_truncated_window_midpoint(self):
        # ONEFlux places each window's Rref at int(mean(indices)), a truncation.
        # Two windows of four records each: knot at index 1 and index 5, so
        # index 2 is interpolated between the two (2.2). Rounding would put the
        # knot on index 2 itself (2.0).
        from diive.flux.partitioning.nighttime_oneflux import _reanalyse_rref
        julday_dec = np.array([2, 3, 4, 5, 12, 13, 14, 15], dtype=float)
        nee_night = np.array([1.8, 2.2, 1.9, 2.1, 3.8, 4.2, 3.9, 4.1])
        tair = np.full(8, 15.0)  # Lloyd-Taylor factor is exactly 1 at TREF
        _, _, rref = _reanalyse_rref(nee_night, tair, tair, julday_dec,
                                     e0=200.0, step=10, window=10)
        self.assertAlmostEqual(rref[1], 2.0)
        self.assertAlmostEqual(rref[2], 2.2)

    def test_results_before_run_raises(self):
        from diive.flux.partitioning import NighttimePartitioningOneFlux
        df = self.df
        part = NighttimePartitioningOneFlux(
            nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
            nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'], lat=self.lat, verbose=0)
        with self.assertRaises(RuntimeError):
            _ = part.results


if __name__ == '__main__':
    unittest.main()
