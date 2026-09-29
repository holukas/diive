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
        # A short slice for the two plumbing tests below (wrapper delegation and
        # the kPa/hPa equivalence). Neither needs a full year: measured on this
        # fixture, one month is 0.5 s against 5 s for the year. One month still fits 14
        # windows, so a dropped unit conversion or a wrapper that lost rows or
        # renamed a column cannot hide in it.
        cls.short = cls.df.loc['2017-06-01':'2017-06-30']

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
        # Passing VPD already in hPa (vpd_in_kpa=False) with a *10 series must
        # reproduce the default kPa path. Both paths run on the short slice, so
        # the comparison is like-for-like.
        from diive.flux.partitioning import DaytimePartitioningReddyProc
        short = self.short

        def _gpp(vpd, vpd_in_kpa):
            return DaytimePartitioningReddyProc(
                nee=short['NEE_CUT_REF_orig'], ta=short['Tair_f'],
                vpd=vpd, sw_in=short['Rg_f'],
                lat=self.lat, lon=self.lon, utc_offset=self.utc_offset,
                vpd_in_kpa=vpd_in_kpa, verbose=0).run().results['GPP_DT_RP'].to_numpy()

        a = _gpp(short['VPD_f'] * 10.0, False)
        b = _gpp(short['VPD_f'], True)
        m = np.isfinite(a) & np.isfinite(b)
        # Guard against a vacuous pass: allclose over an empty selection succeeds.
        self.assertGreater(int(m.sum()), 0)
        np.testing.assert_allclose(a[m], b[m], rtol=1e-9, atol=1e-9)

    def test_functional_wrapper(self):
        from diive.flux.partitioning import partition_nee_daytime_reddyproc
        short = self.short
        res = partition_nee_daytime_reddyproc(
            nee=short['NEE_CUT_REF_orig'], ta=short['Tair_f'],
            vpd=short['VPD_f'], sw_in=short['Rg_f'],
            lat=self.lat, lon=self.lon, utc_offset=self.utc_offset, verbose=0)
        self.assertEqual(len(res), len(short))
        self.assertIn('GPP_DT_RP', res.columns)
        self.assertTrue(res.index.equals(short.index))

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


class TestDaytimeReddyProcRNumerics(unittest.TestCase):
    """The port's building blocks against values computed in R.

    Reference values: REddyProc 1.3.4 and mlegp 3.1.9 in R 4.5.3 (Windows,
    reference BLAS/LAPACK), printed with ``%.17g``. The inputs are exact
    decimal literals, which R and Python parse to the same doubles.
    """

    @classmethod
    def setUpClass(cls):
        import diive.flux.partitioning.daytime_reddyproc as DR
        cls.DR = DR

    # First 60 finite nighttime E0 windows of the CH-DAV 2019 parity run
    # (E0Fit, sdE0Fit rounded to 4 decimals), as partGLSmoothTempSens passes
    # them: X = iCentralRec, Z = E0, nugget = sdE0^2.
    GP_X = 97.0 + 96.0 * np.r_[0:16, 20:64]
    GP_Z = np.array([
        117.4342, 109.5473, 113.0604, 281.3127, 311.0457, 220.021, 225.5298, 306.0262,
        294.2352, 185.1608, 278.1403, 268.878, 177.6864, 294.7593, 275.2557, 58.0169,
        82.6775, 84.8409, 55.3182, 55.3204, 179.1349, 219.9984, 101.4883, 110.5935,
        123.6746, 197.939, 217.7044, 167.4228, 355.6518, 365.799, 357.1247, 337.6191,
        344.9604, 361.6876, 212.5529, 259.2349, 186.5017, 384.6273, 125.1755, 70.0998,
        69.0281, 82.0797, 277.1429, 336.609, 302.5841, 211.9626, 382.8528, 329.7798,
        336.6117, 324.8737, 132.1647, 155.3308, 168.3569, 208.662, 245.4393, 384.4316,
        380.5566, 122.5628, 379.8451, 377.8093])
    GP_SD = np.array([
        66.6011, 74.2529, 75.8263, 82.2541, 112.2316, 104.4432, 95.259, 136.1699,
        134.0292, 168.4391, 145.9826, 140.896, 153.5267, 415.5018, 366.3819, 125.8386,
        154.5435, 149.1353, 144.6502, 144.6501, 156.6971, 168.4898, 144.7719, 284.398,
        242.081, 194.8732, 197.1832, 169.8942, 221.9017, 273.6708, 216.7606, 256.1336,
        253.1915, 265.6397, 462.1851, 588.0249, 245.3301, 320.825, 281.2478, 265.2837,
        295.1808, 285.4354, 200.4042, 205.4418, 295.9915, 161.3641, 238.1689, 103.7105,
        173.379, 256.2072, 174.7175, 203.4691, 190.9256, 162.5506, 158.2853, 216.9126,
        412.7342, 303.9788, 106.1925, 105.564])

    def test_mlegp_random_starts(self):
        # mlegp seeds its own SFMT-607 generator with 0 on every call; fitGP
        # draws one uniform per simplex start.
        u = self.DR._sfmt607_res53(0, 5)
        self.assertEqual(u, [0.4933620949207319, 0.8782963048540589, 0.9588460293484754,
                             0.7950198455327621, 0.17368634832817373])
        # mlegp(X, Z, nugget, simplex.maxiter = 0, BFGS.maxiter = 0)$beta is
        # exp(log(start)) of the first start, between -log(0.65) and -log(0.3)
        # over the smallest squared spacing (96^2).
        m1 = -self.DR._log_c(0.65) / 9216.0
        m2 = -self.DR._log_c(0.3) / 9216.0
        start = self.DR._exp_c(self.DR._log_c(m1 + (m2 - m1) * u[0]))
        self.assertEqual(start, 8.8134277248573241e-05)

    def test_gp_smoother_matches_mlegp(self):
        DR = self.DR
        beta, mu, sig2, _nscale = DR._mlegp_fit(self.GP_X, self.GP_Z, self.GP_SD ** 2)
        # mlegp stops on a flat likelihood; only its own optimizer path gives
        # its beta, and that to the bit.
        self.assertEqual(beta, 4.6916433782657807e-05)
        np.testing.assert_allclose([sig2, mu], [8647.0840318012506, 218.68173297935977],
                                   rtol=1e-12)
        predict, nugget = DR._gp_smooth(self.GP_X, self.GP_Z, self.GP_SD ** 2)
        # gpFit$nugget is the absolute nugget variance sdE0^2 * scale * sig2;
        # without sig2 the E0 uncertainty was 50-80 % too low.
        np.testing.assert_allclose(nugget[0], 230.06634662459925, rtol=1e-12)
        fit, se = predict(np.array([97.0, 1633.0, 6337.0]))  # observed, gap, beyond
        np.testing.assert_allclose(fit, [120.94526188776447, 99.25952585273873,
                                         229.4152786354832], rtol=1e-12)
        np.testing.assert_allclose(se, [14.766005630249561, 71.11300590848488,
                                        90.99309941007544], rtol=1e-12)

    def test_nighttime_e0_fit_matches_r_nls(self):
        # partGLEstimateTempSensInBoundsE0Only(REco, TK, prevE0, TRefFit)
        tc = np.array([-0.5, 0.3, 1.1, 2.4, 3.0, 3.9, 4.4, 5.8, 6.1, 7.3, 8.0, 8.8,
                       9.9, 10.4, 11.7, 12.2])
        reco = np.array([0.9, 1.3, 1.1, 1.6, 1.4, 1.9, 1.7, 2.3, 2.0, 2.6, 2.4, 3.1,
                         2.8, 3.3, 3.6, 3.4])
        tref = float(np.median(tc)) + 273.15
        for prev, ref in ((150.0, (258.33135449933008, 17.676610788557479, 2.1310423011781978)),
                          (np.nan, (258.33169297132929, 17.67661854624421, 2.1310417026676181))):
            e0, sde0, _tref, rref = self.DR._fit_e0_window(reco, tc + 273.15, prev, tref)
            # numpy's QR, sums and mean moved E0 by ~1e-7 here
            np.testing.assert_allclose([e0, sde0, rref], ref, rtol=1e-13)

    def test_exp_rounds_like_r(self):
        # exp() in R on Windows: x87 extended precision, rounded twice. numpy
        # differs in the last bit for the first, third and fourth argument.
        x = np.array([0.40090808598324656, 0.648769767023623, 0.14537000702694058,
                      0.560505997389555])
        r_exp = np.array([1.4931800180192432, 1.9131857164840285, 1.1564673921761173,
                          1.7515585601643018])
        self.assertTrue(np.array_equal(self.DR._exp_r(x), r_exp))

    def _lrc_window(self):
        # One synthetic 60-record LRC window, NEE rounded to 3 decimals.
        i = np.arange(60)
        up = np.where(i < 30, i, 59 - i)
        rg = 30.0 + 30.0 * up
        vpd = 3.25 + 0.75 * up
        temp = 8.125 + 0.5 * up
        nee = np.array([
            0.574, 0.376, -1.525, -2.887, -2.508, -2.961, -4.801, -5.327, -4.694, -5.572,
            -6.939, -6.519, -5.784, -6.815, -7.614, -6.625, -6.256, -7.449, -7.561, -6.336,
            -6.489, -7.58, -7.013, -5.901, -6.556, -7.273, -6.182, -5.471, -6.451, -6.596,
            -5.402, -5.565, -6.893, -6.701, -5.717, -6.437, -7.488, -6.723, -6.068, -7.169,
            -7.673, -6.536, -6.376, -7.557, -7.363, -6.123, -6.462, -7.366, -6.464, -5.371,
            -6.03, -6.144, -4.496, -3.565, -4.111, -3.448, -1.495, -0.945, -1.166, 0.391])
        sd = np.tile([0.8, 1.1, 0.9, 1.4], 15)
        return nee, sd, rg, vpd, temp

    def test_lrc_fit_matches_r(self):
        # RectangularLRCFitter()$fitLRC(dsDay, E0 = 180, sdE0 = 25,
        #   RRefNight = 2.2, partGLControl(nBootUncertainty = 0))
        res = self.DR._fit_lrc(self._lrc_window(), 180.0, 25.0, 2.2, np.full(5, np.nan))
        self.assertEqual(res['iopt'], [0, 1, 2, 3])
        np.testing.assert_allclose(
            res['theta'][:4], [0.02392561859679989, 19.233225120938663,
                               0.051145015446522814, 3.1820468865344664], rtol=1e-12)

    def test_lrc_fit_without_vpd_effect_matches_r(self):
        # The same window with isNeglectVPDEffect = TRUE, as REddyProc fits a
        # window with fewer than 10 usable records that have VPD: k fixed at 0.
        res = self.DR._fit_lrc(self._lrc_window(), 180.0, 25.0, 2.2, np.full(5, np.nan),
                               neglect_vpd=True)
        self.assertEqual(res['iopt'], [1, 2, 3])
        self.assertEqual(res['theta'][0], 0.0)
        np.testing.assert_allclose(
            res['theta'][1:4], [17.221555223837271, 0.093472750234682656,
                                5.6599000990408781], rtol=1e-12)

    def test_rref_series_starts_with_first_nighttime_fit_rref(self):
        # partGLFitNightTimeTRespSens fills windows without an RRef estimate with
        # fillNAForward(RRef, firstValue = E0Smooth$RRef[which(is.finite(E0Smooth$RRef))[1]]).
        # E0Smooth has no RRef column; R's `$` partially matches RRefFit, the RRef
        # of the nighttime nls fit. So leading windows get the first finite
        # RRefFit, not the first estimated RRef (CH-LAE 2017: 4.25 vs 13.86).
        DR = self.DR
        dts, n = 48, 48 * 30
        _start, ic = DR._window_grid(n, dts)
        hour = np.arange(n) % dts
        day = np.arange(n) // dts
        is_night = hour < 12
        temp = 5.0 + 0.5 * (hour % 12) + 0.1 * day
        # nighttime NEE only from the 13th day on: the first windows have no estimate
        nee = np.where(day >= 12, 2.0 * DR._exp_r(150.0 * (
            1.0 / (DR.TREF_K - DR.T0_K) - 1.0 / (temp + 273.15 - DR.T0_K))), np.nan)
        e0 = np.full(ic.size, 150.0)
        rref_fit = np.full(ic.size, np.nan)
        rref_fit[1], rref_fit[5] = 4.25, 7.5
        rref = DR._fit_rref_windows(nee, temp, is_night, e0, ic, dts, n, rref_fit)
        n_lead = 3  # windows 1-3 (centred on days 3, 5, 7) end before the 13th day
        self.assertTrue(np.all(rref[:n_lead] == 4.25))
        np.testing.assert_allclose(rref[n_lead:], 2.0, rtol=1e-12)
        # no estimate at all: every window takes the first finite RRefFit
        rref = DR._fit_rref_windows(np.full(n, np.nan), temp, is_night, e0, ic, dts, n,
                                    rref_fit)
        self.assertTrue(np.all(rref == 4.25))

    def test_missing_vpd_gives_na_gpp_unless_k_is_zero(self):
        # R's ifelse(VPD > VPD0, ...) is NA for missing VPD; only k = 0 (VPD
        # effect off) predicts GPP there. These NAs trigger REddyProc's refit.
        rg = np.array([500.0, 500.0, 500.0])
        vpd = np.array([15.0, np.nan, 5.0])
        ta = np.array([15.0, 15.0, 15.0])
        params = np.array([[0.05, 20.0, 0.05, 2.0, 150.0]])
        _reco, gpp = self.DR._interpolate_fluxes(np.array([2]), params, rg, vpd, ta, 3)
        self.assertTrue(np.isfinite(gpp[[0, 2]]).all())
        self.assertTrue(np.isnan(gpp[1]))
        params[0, 0] = 0.0
        _reco, gpp = self.DR._interpolate_fluxes(np.array([2]), params, rg, vpd, ta, 3)
        self.assertTrue(np.isfinite(gpp).all())


if __name__ == '__main__':
    unittest.main()
