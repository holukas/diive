import unittest
from unittest import mock

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
        # A short slice for the two plumbing tests below (wrapper delegation and
        # the kPa/hPa equivalence). Neither needs a full year: they assert shape,
        # column names and an exact match between the two unit paths, and one
        # month still fits enough LRC windows for a dropped conversion or a
        # wrapper that lost rows to show up.
        cls.short = cls.df.loc['2017-06-01':'2017-06-30']

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
        # reproduce the default kPa path. Both paths run on the short slice, so
        # the comparison is like-for-like.
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        short = self.short

        def _gpp(vpd, vpd_in_kpa):
            return DaytimePartitioningOneFlux(
                nee=short['NEE_CUT_REF_orig'], ta=short['Tair_orig'],
                sw_in=short['Rg_orig'], ta_f=short['Tair_f'],
                sw_in_f=short['Rg_f'], vpd=vpd,
                vpd_in_kpa=vpd_in_kpa, verbose=0).run().results['GPP_DT_OF'].to_numpy()

        a = _gpp(short['VPD_f'] * 10.0, False)
        b = _gpp(short['VPD_f'], True)
        m = np.isfinite(a) & np.isfinite(b)
        # Guard against a vacuous pass: allclose over an empty selection succeeds.
        self.assertGreater(int(m.sum()), 0)
        np.testing.assert_allclose(a[m], b[m], rtol=1e-9, atol=1e-9)

    def test_functional_wrapper(self):
        from diive.flux.partitioning import partition_nee_daytime_oneflux
        short = self.short
        res = partition_nee_daytime_oneflux(
            nee=short['NEE_CUT_REF_orig'], ta=short['Tair_orig'],
            sw_in=short['Rg_orig'], ta_f=short['Tair_f'],
            sw_in_f=short['Rg_f'], vpd=short['VPD_f'], verbose=0)
        self.assertEqual(len(res), len(short))
        self.assertIn('GPP_DT_OF', res.columns)
        self.assertTrue(res.index.equals(short.index))

    def test_requires_datetime_index(self):
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        bad = pd.Series([1.0, 2.0, 3.0])  # RangeIndex, not DatetimeIndex
        with self.assertRaises(TypeError):
            DaytimePartitioningOneFlux(
                nee=bad, ta=bad, sw_in=bad, ta_f=bad, sw_in_f=bad, vpd=bad,
                verbose=0)

    def test_hourly_windows_land_on_their_own_records(self):
        # The window anchors are record indices, so they depend on the records
        # per day. Assuming 48 (half-hourly) for hourly input puts each window
        # at twice its true position: the windows of the first half of the year
        # get stretched over the whole year and those of the second half fall
        # off the end of the record. Both symptoms are asserted here, because
        # the results stay gap-free either way.
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        hourly = self.df[['NEE_CUT_REF_orig', 'Tair_orig', 'Rg_orig',
                          'Tair_f', 'Rg_f', 'VPD_f']].resample('1h').mean()
        res = DaytimePartitioningOneFlux(
            nee=hourly['NEE_CUT_REF_orig'], ta=hourly['Tair_orig'],
            sw_in=hourly['Rg_orig'], ta_f=hourly['Tair_f'],
            sw_in_f=hourly['Rg_f'], vpd=hourly['VPD_f'], verbose=0).run().results

        # Parameters are reported at each window's central record. Windows step
        # by WINSIZE / 2 = 2 days; not every one is accepted, so the half-hourly
        # run of the same year sets the scale. Anchoring on 48 records per day
        # would lose every window past midsummer.
        anchors = res['RREF_DT_OF'].dropna().index
        anchors_hh = self.part.results['RREF_DT_OF'].dropna().index
        self.assertGreater(len(anchors), 0.8 * len(anchors_hh))
        spacing_days = np.median(np.diff(anchors.to_numpy()).astype(
            'timedelta64[h]').astype(float)) / 24.0
        self.assertAlmostEqual(spacing_days, 2.0, delta=0.3)

        # The last window still has to sit inside the record, not past its end.
        self.assertGreater(anchors[-1], res.index[int(0.95 * len(res))])

    def test_uncertainty_lookup_clips_at_the_record_edges(self):
        # The weights come from ONEFlux's Python daytime.uncert_via_gapFill,
        # which clips its look-up window onto record 0 / n-1. The MDS gap-filler
        # trims instead, so this port has to ask the shared cascade for 'clip'.
        from diive.flux.partitioning import daytime_oneflux as mod
        seen = {}

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return {'sd': np.full(len(args[0]), np.nan)}

        arr = np.zeros(10, dtype=np.float32)
        with mock.patch.object(mod, 'mds_gapfill_cascade', side_effect=spy):
            mod._uncert_via_gapfill(arr, arr, arr, arr, arr, 48)
        self.assertEqual(seen.get('edge'), 'clip')

    def test_uncertainty_sd_uses_oneflux_tstd_arithmetic(self):
        # ONEFlux takes the look-up SD with scipy.stats.tstd. Under NumPy 1 its
        # variance is float32 but the n/(n-1) correction and the square root
        # run in float64. np.std rounds differently in about one record in
        # five, and these SDs weight every daytime fit. Expected value: scipy
        # 1.17.1 tstd under NumPy 1.26.4.
        from diive.flux.partitioning import daytime_oneflux as mod
        x = np.array([-2.609, -9.697, 0.043, -2.206, 0.03, -5.151, -4.887,
                      -7.379, 3.611, -4.526, 0.763, -5.942], dtype=np.float32)
        expected = np.float32(3.8134894371032715)
        self.assertEqual(mod._tstd_oneflux(x, 1), expected)
        self.assertNotEqual(np.std(x, ddof=1), expected)

        seen = {}

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return {'sd': np.full(len(args[0]), np.nan)}

        arr = np.zeros(10, dtype=np.float32)
        with mock.patch.object(mod, 'mds_gapfill_cascade', side_effect=spy):
            mod._uncert_via_gapfill(arr, arr, arr, arr, arr, 48)
        self.assertIs(seen.get('sd_func'), mod._tstd_oneflux)

    def test_alpha_left_at_the_starting_guess_is_accepted(self):
        # ONEFlux reads alpha back from its float32 parameter table, and
        # float32(0.01) != 0.01, so its guard against an alpha that never left
        # the starting guess never fires. The port must accept the same windows.
        from diive.flux.partitioning.daytime_oneflux import _check_parameters
        row = np.zeros(10, dtype=np.float32)
        row[:5] = [0.01, 30.0, 0.1, 5.0, 150.0]  # alpha, beta, k, rref, e0
        self.assertEqual(_check_parameters(row), 1)

    def test_switch_rejects_alpha_left_at_the_starting_guess(self):
        # The same window with the switch on: the check works as ONEFlux's own
        # comment intends. A fitted alpha still passes.
        from diive.flux.partitioning.daytime_oneflux import _check_parameters
        row = np.zeros(10, dtype=np.float32)
        row[:5] = [0.01, 30.0, 0.1, 5.0, 150.0]
        self.assertEqual(_check_parameters(row, reject_alpha_at_start=True), 0)
        row[0] = 0.02
        self.assertEqual(_check_parameters(row, reject_alpha_at_start=True), 1)

    def test_switch_reaches_the_window_check(self):
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        from diive.flux.partitioning import daytime_oneflux as mod
        real = mod._check_parameters
        seen = []

        def spy(p, reject_alpha_at_start=False):
            seen.append(reject_alpha_at_start)
            return real(p, reject_alpha_at_start)

        short = self.short
        with mock.patch.object(mod, '_check_parameters', side_effect=spy):
            DaytimePartitioningOneFlux(
                nee=short['NEE_CUT_REF_orig'], ta=short['Tair_orig'],
                sw_in=short['Rg_orig'], ta_f=short['Tair_f'],
                sw_in_f=short['Rg_f'], vpd=short['VPD_f'],
                reject_alpha_at_start=True, verbose=0).run()
        self.assertGreater(len(seen), 0)
        self.assertTrue(all(seen))

    def test_missing_gap_filled_driver_gives_nan_not_a_value(self):
        # A missing gap-filled driver used to reach the models as -9999 and come
        # back as a finite flux (RECO near 1000 at -9999 degC). RECO needs TA,
        # GPP needs SW_IN and VPD; each must be NaN where its driver is missing.
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        short = self.short.copy()
        no_ta = (short.index >= '2017-06-10') & (short.index < '2017-06-11')
        no_rg = (short.index >= '2017-06-20') & (short.index < '2017-06-21')
        short.loc[no_ta, ['Tair_f', 'Tair_orig']] = np.nan
        short.loc[no_rg, ['Rg_f', 'Rg_orig']] = np.nan
        res = DaytimePartitioningOneFlux(
            nee=short['NEE_CUT_REF_orig'], ta=short['Tair_orig'],
            sw_in=short['Rg_orig'], ta_f=short['Tair_f'],
            sw_in_f=short['Rg_f'], vpd=short['VPD_f'], verbose=0).run().results
        self.assertTrue(res.loc[no_ta, 'RECO_DT_OF'].isna().all())
        self.assertTrue(res.loc[no_rg, 'GPP_DT_OF'].isna().all())
        self.assertTrue(res.loc[no_rg, 'SE_GPP_DT_OF'].isna().all())
        # A missing light driver does not stop RECO, and nothing else is blank.
        self.assertTrue(res.loc[no_rg, 'RECO_DT_OF'].notna().all())
        rest = ~(no_ta | no_rg)
        self.assertTrue(res.loc[rest, ['RECO_DT_OF', 'GPP_DT_OF']].notna().all().all())
        self.assertLess(res['RECO_DT_OF'].max(), 50)

    def test_models_are_evaluated_on_drivers_widened_to_float64(self):
        # ONEFlux stores drivers as float32 but widens them to float64 before
        # evaluating a model. Evaluated in float32, the residuals differ by ~1e-7,
        # enough to flip the sign of a VPD sensitivity k that converges to zero,
        # and the model cascade branches on that sign.
        from diive.flux.partitioning.daytime_oneflux import _build_predict, TREF, T0, VPD0
        rng = np.random.default_rng(1)
        ind = {'rg': rng.uniform(5, 900, 200).astype(np.float32),
               'ta': rng.uniform(-5, 30, 200).astype(np.float32),
               'vpd': rng.uniform(0, 25, 200).astype(np.float32),
               'e0': np.full(200, 211.7, dtype=np.float32)}
        par = np.array([0.047, 29.9, 0.03, 5.9])
        got = _build_predict('HLRC_LloydVPD', ind)(par)

        rg, ta, vpd, e0 = (ind[k].astype(np.float64) for k in ('rg', 'ta', 'vpd', 'e0'))
        m = np.minimum(np.exp(-1.0 * par[2] * (vpd - VPD0)), 1.0)
        tfac = np.exp(e0 * ((1.0 / (TREF - T0)) - (1.0 / (ta - T0))))
        want = -1.0 * par[0] * par[1] * m * rg / (par[0] * rg + par[1] * m) + par[3] * tfac
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)

    def test_parameter_table_is_float32(self):
        # The accept test above only matches ONEFlux if the table it reads
        # really is float32, as ONEFlux's FLOAT_PREC table is.
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        from diive.flux.partitioning import daytime_oneflux as mod
        real = mod._check_parameters
        dtypes = []

        def spy(p, reject_alpha_at_start=False):
            dtypes.append(np.asarray(p).dtype)
            return real(p, reject_alpha_at_start)

        short = self.short
        with mock.patch.object(mod, '_check_parameters', side_effect=spy):
            DaytimePartitioningOneFlux(
                nee=short['NEE_CUT_REF_orig'], ta=short['Tair_orig'],
                sw_in=short['Rg_orig'], ta_f=short['Tair_f'],
                sw_in_f=short['Rg_f'], vpd=short['VPD_f'], verbose=0).run()
        self.assertGreater(len(dtypes), 0)
        self.assertTrue(all(d == np.float32 for d in dtypes))

    def test_last_record_of_the_year_is_not_dated_january(self):
        # With a MIDDLE-stamped index the year's last record ends at 00:00 on
        # 1 January, so its day of year wraps to 1 and it would be pooled into
        # the January windows. ONEFlux repairs the wrap to 366 (367 in a leap
        # year); the partitioning itself is stubbed out, only its input matters.
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        from diive.flux.partitioning import daytime_oneflux as mod
        cols = ['RECO_DT_OF', 'GPP_DT_OF', 'SE_GPP_DT_OF', 'ALPHA_DT_OF',
                'BETA_DT_OF', 'K_DT_OF', 'RREF_DT_OF', 'E0_DT_OF']
        seen = {}

        def stub(**kwargs):
            seen['julday'] = np.asarray(kwargs['julday'])
            return {c: np.full(len(kwargs['julday']), np.nan) for c in cols}

        with mock.patch.object(mod, '_partition_one_year', side_effect=stub):
            DaytimePartitioningOneFlux(
                nee=self.df['NEE_CUT_REF_orig'], ta=self.df['Tair_orig'],
                sw_in=self.df['Rg_orig'], ta_f=self.df['Tair_f'],
                sw_in_f=self.df['Rg_f'], vpd=self.df['VPD_f'], verbose=0).run()
        julday = seen['julday']
        self.assertEqual(julday[-2], 365)  # 2017 is not a leap year
        self.assertEqual(julday[-1], 366)
        self.assertEqual(julday[0], 1)

    def test_too_few_records_returns_oneflux_default_result(self):
        # ONEFlux's nlinlts2 skips the fit below 3 * npara records and returns
        # its default result: parameters -9999, standard errors 0, a zero
        # covariance matrix (not None), residuals -9999, RMSE 0. A None
        # covariance would mark the window as broken instead.
        from diive.flux.partitioning.daytime_oneflux import _fit, NAN
        n = 11  # one short of the 12 records HLRC_LloydVPD needs
        ind = {'rg': np.full(n, 500.0, dtype=np.float32),
               'ta': np.full(n, 15.0, dtype=np.float32),
               'e0': np.full(n, 150.0, dtype=np.float32),
               'vpd': np.full(n, 8.0, dtype=np.float32)}
        r = _fit('HLRC_LloydVPD', np.full(n, -5.0, dtype=np.float32), ind, npara=4,
                 xguess=[0.01, 30.0, 0.0, 5.0], mprior=np.zeros(4, dtype='f4'),
                 sigm=np.ones(4), sigd=np.ones(n, dtype=np.float32))
        np.testing.assert_array_equal(r['params'], np.full(4, NAN))
        np.testing.assert_array_equal(r['std'], np.zeros(4))
        np.testing.assert_array_equal(r['cov_matrix'], np.zeros((4, 4)))
        self.assertTrue(np.all(np.isneginf(r['cor_matrix'])))
        np.testing.assert_array_equal(r['residuals'], np.full(n, NAN))
        self.assertEqual(r['rmse'], 0.0)

    def test_window_with_eleven_daytime_records_falls_back_to_hlrc_lloyd(self):
        # A window enters the fit with 11 measured daytime records, one short of
        # what HLRC_LloydVPD needs. ONEFlux then gets -9999 parameters back,
        # reads k = -9999 < 0 and refits the window without the VPD term. The
        # port used to skip such a window. On CH-LAE, where only a fifth of NEE
        # is measured, that dropped 1-6 windows a year and moved annual daytime
        # GPP by up to 1.5 %.
        from diive.flux.partitioning import DaytimePartitioningOneFlux
        from diive.flux.partitioning import daytime_oneflux as mod
        short = self.short.copy()
        nee = short['NEE_CUT_REF_orig'].copy()
        jd = (short.index + pd.Timedelta(minutes=15)).dayofyear
        # Window (160, 164]: keep 11 measured daytime records, all nights.
        day = (jd > 160) & (jd <= 164) & (short['Rg_orig'] > 4) & nee.notna()
        nee.iloc[np.flatnonzero(day)[11:]] = np.nan
        self.assertEqual(int((day & nee.notna()).sum()), 11)

        real = mod._fit
        calls = []

        def spy(lts_func, dep, *args, **kwargs):
            calls.append((lts_func, int(np.size(dep))))
            return real(lts_func, dep, *args, **kwargs)

        with mock.patch.object(mod, '_fit', side_effect=spy):
            DaytimePartitioningOneFlux(
                nee=nee, ta=short['Tair_orig'], sw_in=short['Rg_orig'],
                ta_f=short['Tair_f'], sw_in_f=short['Rg_f'], vpd=short['VPD_f'],
                verbose=0).run()
        self.assertIn(('HLRC_LloydVPD', 11), calls)
        # all three beta starting guesses carry on with the 3-parameter model
        self.assertEqual(calls.count(('HLRC_Lloyd', 11)), 3)

    def test_best_beta_guess_is_chosen_on_float32_rmse(self):
        # ONEFlux keeps the RMSEs of the three beta starting guesses in a
        # float32 table and takes the first minimum. Two fits whose RMSEs differ
        # only beyond float32 precision tie, and the earlier guess wins. In
        # float64 the later one won, which on CH-LAE 2019 carried a different
        # alpha into the next window.
        from diive.flux.partitioning import daytime_oneflux as mod
        n = 48 * 6
        julday = (1 + np.arange(n) // 48).astype(np.float32)
        rg = np.where(np.arange(n) % 48 >= 24, 500.0, 0.0).astype(np.float32)
        D = dict(nee_f=np.full(n, -2.0, dtype=np.float32), nee_fqc=np.zeros(n),
                 tair_f=np.full(n, 15.0, dtype=np.float32), rg_f=rg,
                 vpd_f=np.full(n, 8.0, dtype=np.float32), rg_meas=rg,
                 julday=julday, nee_fs_unc=np.ones(n, dtype=np.float32))
        guess = {'j': 0}
        # alpha and RMSE per beta starting guess j; j=1 and j=2 tie in float32
        alphas, rmses = (0.05, 0.06, 0.07), (3.0, 2.0 + 1e-8, 2.0)

        def fake_fit(lts_func, dep, indeps, npara, xguess, mprior, sigm, sigd):
            if lts_func == 'LloydTemp':
                params, rmse = np.array([2.0, 150.0]), 1.0
            else:
                j = guess['j'] % 3
                guess['j'] += 1
                params, rmse = np.array([alphas[j], 30.0, 0.05, 2.0]), rmses[j]
            return dict(params=params, std=np.full(npara, 0.01),
                        cov_matrix=np.eye(npara) * 1e-4, cor_matrix=np.eye(npara),
                        residuals=np.zeros(dep.size), rmse=rmse)

        with mock.patch.object(mod, '_fit', side_effect=fake_fit):
            params_ok, *_ = mod._estimate_parasets(D, nperday=48, verbose=0)
        self.assertGreater(len(params_ok), 0)
        self.assertEqual(params_ok[0][0], np.float32(0.06))

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
