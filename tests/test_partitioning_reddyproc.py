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
        self.assertLess(e0[0], 450.0)

    def test_matches_reddyproc_reference(self):
        part = self._run()
        res = part.results
        # Native REddyProc 1.3.4 sMRFluxPartition on exactly these ten years and
        # coordinates gives E0 = 282.89 (with the old E0 bound of 350: 153.96).
        self.assertEqual(res['E0_NT_RP'].iloc[0], 282.89)
        # The bundled Reco/GPP columns are REddyProc-derived but come from other
        # runs (per year, other settings), so they are a plausibility check only.
        reco_ref = self.df['Reco_CUT_REF']
        gpp_ref = self.df['GPP_CUT_REF_f']
        m = res['RECO_NT_RP'].notna() & reco_ref.notna()
        self.assertGreater(np.corrcoef(res['RECO_NT_RP'][m], reco_ref[m])[0, 1], 0.98)
        self.assertLess(abs(res['RECO_NT_RP'][m].mean() - reco_ref[m].mean()), 0.25)
        mg = res['GPP_NT_RP'].notna() & gpp_ref.notna()
        self.assertGreater(np.corrcoef(res['GPP_NT_RP'][mg], gpp_ref[mg])[0, 1], 0.98)

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


class TestNighttimeReddyProcParity(unittest.TestCase):
    """Parity with native REddyProc 1.3.4 (R 4.5.3) on single years of CH-DAV.

    Reference numbers come from sMRFluxPartition run on the same inputs (the
    REddyProc parity harness: nee/ta/sw_in measured, nee_f/ta_f gap-filled,
    lat 46.8153, lon 9.8559, UTC+1).
    """
    LAT, LON = 46.8153, 9.8559

    @classmethod
    def setUpClass(cls):
        import diive as dv
        cls.df = dv.load_exampledata_parquet()

    def _year(self, year):
        cols = {'nee': 'NEE_CUT_REF_orig', 'nee_f': 'NEE_CUT_REF_f', 'ta': 'Tair_orig',
                'ta_f': 'Tair_f', 'sw_in': 'Rg_orig'}
        sub = self.df.loc[str(year), list(cols.values())].copy()
        sub.columns = list(cols)
        return sub

    def _run(self, year):
        from diive.flux.partitioning import NighttimePartitioningReddyProc
        sub = self._year(year)
        return NighttimePartitioningReddyProc(
            nee=sub['nee'], ta=sub['ta'], sw_in=sub['sw_in'], nee_f=sub['nee_f'],
            ta_f=sub['ta_f'], lat=self.LAT, lon=self.LON, utc_offset=1, verbose=0).run().results

    def _window(self, year, start):
        """Nighttime NEE and temperature (K) of the E0 window starting at DayCounter `start`."""
        import diive.flux.partitioning.nighttime_reddyproc as nr
        sub = self._year(year)
        idx = sub.index
        potrad = nr.potential_radiation(idx.dayofyear.to_numpy(),
                                        (idx.hour + idx.minute / 60.0).to_numpy(),
                                        self.LAT, self.LON, 1)
        nee, ta = sub['nee'].to_numpy(), sub['ta'].to_numpy()
        night = (sub['sw_in'].to_numpy() <= 10) & (potrad <= 0) & ~np.isnan(nee)
        day_counter = np.arange(1, len(sub) + 1) // 48
        sel = (day_counter >= start) & (day_counter <= start + 14) & night & ~np.isnan(ta)
        return nee[sel], ta[sel] + 273.15

    def test_e0_upper_bound_is_450(self):
        # sMRFluxPartition passes the temperature as 'FP_Temp_NEW', so the
        # 'Tair' branch (350) of sRegrE0fromShortTerm never runs: 450 applies.
        import diive.flux.partitioning.nighttime_reddyproc as nr
        self.assertEqual(nr.E0_MAX, 450.0)
        # In 2016 the window starting on day 281 (E0 367.6 +- 42.0) is one of
        # the three averaged; with a bound of 350 E0 was 231.11.
        self.assertEqual(self._run(2016)['E0_NT_RP'].iloc[0], 280.68)
        self.assertEqual(self._run(2019)['E0_NT_RP'].iloc[0], 167.5)

    def test_reco_matches_reddyproc(self):
        res = self._run(2016)
        # (time stamp, REddyProc R_ref, REddyProc Reco)
        ref = [('2016-01-01 00:15', 0.6189940733139093, 0.12808948041285195),
               ('2016-04-14 04:15', 10.5843350869417, 2.3424526857547097),
               ('2016-07-06 12:15', 8.979994261178613, 9.344891963616682),
               ('2016-12-20 04:15', 4.80739608660523, 0.701560369170886)]
        for ts, rref, reco in ref:
            self.assertAlmostEqual(res.loc[ts, 'RREF_NT_RP'], rref, delta=1e-13 * rref)
            self.assertAlmostEqual(res.loc[ts, 'RECO_NT_RP'], reco, delta=1e-13 * reco)

    def test_window_dropped_where_r_nls_fails(self):
        import diive.flux.partitioning.nighttime_reddyproc as nr
        # 2016 day 11: nls does not converge within 50 iterations. 2019 day 131:
        # nls runs into a singular gradient. REddyProc drops both windows; a
        # generic least-squares solver returned E0 4724 and -1971 there.
        for year, start in ((2016, 11), (2019, 131)):
            nee, ta_k = self._window(year, start)
            self.assertIsNone(nr._fit_e0_single(nee, ta_k, nr.TREF_K))

    def test_window_fit_matches_r_nls(self):
        import diive.flux.partitioning.nighttime_reddyproc as nr
        # REddyProc's E_0_trim and E_0_trim_SD of the three averaged windows of
        # 2016. The leastsq-based port was off by up to 6e-6 relative; what is
        # left comes from exp() rounding differently in its last bit.
        ref = {271: (207.78782739839917, 47.99914068123538),
               276: (266.6322669497816, 51.81372936729844),
               281: (367.62269001416365, 42.033458506377904)}
        for start, (e0, sd) in ref.items():
            nee, ta_k = self._window(2016, start)
            fit = nr._fit_e0_single(nee, ta_k, nr.TREF_K)
            self.assertIsNotNone(fit)
            self.assertAlmostEqual(fit[4], e0, delta=1e-6 * e0)
            self.assertAlmostEqual(fit[5], sd, delta=1e-6 * sd)

    def test_r_arithmetic_helpers(self):
        import diive.flux.partitioning.nighttime_reddyproc as nr
        # R 4.5.3: round(c(276.455, 326.925), 2); Python's round gives the other neighbour.
        self.assertEqual(nr._r_round(276.455, 2), 276.46)
        self.assertEqual(nr._r_round(326.925, 2), 326.92)
        # R: coef(lm(y ~ 0 + x)) through the QR; sum(x*y)/sum(x^2) is 1.2531129185484733.
        x = np.array([1.75, 2.29, 2.05, 0.95, 1.1, 2.25])
        y = np.array([0.03, 4.11, 3.99, 2.34, 1.52, 1.39])
        self.assertEqual(nr._lm_through_origin(x, y), 1.2531129185484731)
        # R: qr(cbind(x, 2 * x))$rank is 1 -> nls reports a singular gradient.
        self.assertEqual(nr._dqrdc2(np.column_stack([x, 2 * x]))[2], 1)


if __name__ == '__main__':
    unittest.main()
