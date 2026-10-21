"""
TEST_USTAR_REDDYPROC: u* threshold detection ported from REddyProc 1.3.4
=======================================================================

Unit tests of the binning helpers, the Fw2 test, the season factor and R's
long-double ``mean()``, and checks against a native REddyProc 1.3.4 run on the
bundled CH-LAE example data. The reference values in
``tests/data/ustar_reddyproc_r134.npz`` come from the parity harness
``diive-data/references/ustar_parity/reddyproc/`` (R 4.5.3, Windows).

Part of the diive library: https://github.com/holukas/diive
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE = Path(__file__).resolve().parent / 'data' / 'ustar_reddyproc_r134.npz'
LAE_COLS = dict(nee_col='NEE_L3.1_L3.2_QCF_IRGA72', ta_col='TA_T1_47_1_gfXG_IRGA72',
                ustar_col='USTAR_IRGA72', swin_col='SW_IN_T1_47_1_gfXG_IRGA72')


class TestRNumerics(unittest.TestCase):

    def test_mean_uses_long_double(self):
        from diive.flux.lowres._ustar_rp_numerics import r_mean, r_mean_ld
        # 2**60 + 1 fits a 64-bit significand, not a 53-bit one (R gives 1/3)
        x = np.array([2.0 ** 60, 1.0, -2.0 ** 60])
        self.assertEqual(r_mean(x), 1.0 / 3.0)
        self.assertEqual(r_mean_ld(x), 1.0 / 3.0)
        # 2**64 + 1 does not fit either: R gives 0
        self.assertEqual(r_mean(np.array([2.0 ** 64, 1.0, -2.0 ** 64])), 0.0)

    def test_mean_matches_r_where_exact_rounding_does_not(self):
        from diive.flux.lowres._ustar_rp_numerics import r_mean, r_mean_ld
        fx = np.load(FIXTURE)
        ends = np.cumsum(fx['mean_vec_len'])
        for i, n in enumerate(fx['mean_vec_len']):
            x = fx['mean_vec_values'][ends[i] - n:ends[i]]
            self.assertEqual(r_mean(x), fx['mean_vec_r'][i])
            self.assertEqual(r_mean_ld(x), fx['mean_vec_r'][i])

    def test_fast_path_equals_emulation(self):
        from diive.flux.lowres._ustar_rp_numerics import r_mean, r_mean_ld
        rng = np.random.default_rng(7)
        for i in range(3000):
            n = int(rng.integers(1, 80))
            x = rng.normal(rng.normal(0, 3), 3, n)
            if i % 3 == 0:
                x = np.round(x, 4)
            self.assertEqual(r_mean(x), r_mean_ld(x))

    def test_cor_matches_r(self):
        from diive.flux.lowres._ustar_rp_numerics import abs_cor_below, r_cor
        fx = np.load(FIXTURE)
        ends = np.cumsum(fx['cor_len'])
        for i, n in enumerate(fx['cor_len']):
            x = fx['cor_x'][ends[i] - n:ends[i]]
            y = fx['cor_y'][ends[i] - n:ends[i]]
            self.assertEqual(r_cor(x, y), fx['cor_r'][i])
            self.assertEqual(abs_cor_below(x, y, 0.5), abs(fx['cor_r'][i]) < 0.5)
        # R: NA for a constant vector, clamped to 1 for two records
        self.assertTrue(np.isnan(r_cor(np.ones(5), np.arange(5.0))))
        self.assertFalse(abs_cor_below(np.ones(5), np.arange(5.0), 0.5))
        self.assertEqual(r_cor(np.array([0.1, 0.7]), np.array([3.0, 5.0])), 1.0)

    def test_quantile_type7(self):
        from diive.flux.lowres._ustar_rp_numerics import r_quantile7
        q = r_quantile7([4.0, 1.0, np.nan, 3.0, 2.0], [0.05, 0.5, 1.0])
        # index 1 + 3 * 0.05 = 1.15: (1 - 0.15) * 1 + 0.15 * 2
        self.assertEqual(q[0], (1 - (1.0 + 3 * 0.05 - 1)) * 1.0 + (1.0 + 3 * 0.05 - 1) * 2.0)
        self.assertEqual(q[1], 2.5)
        self.assertEqual(q[2], 4.0)

    def test_median_even_count(self):
        from diive.flux.lowres._ustar_rp_numerics import r_median_finite
        self.assertEqual(r_median_finite(np.array([0.3, np.nan, 0.1])), 0.2)
        self.assertTrue(np.isnan(r_median_finite(np.array([np.nan, np.nan]))))


class TestBinning(unittest.TestCase):

    def test_which_ge_as_r(self):
        from diive.flux.lowres._ustar_rp_kernels import which_ge
        x = np.array([1, 3, 5])
        # REddyProc:::whichValueGreaterEqualC; 0 stands for NA
        self.assertEqual(which_ge(x, 3, 1), 2)
        self.assertEqual(which_ge(x, 6, 1), 0)
        self.assertEqual(which_ge(x, 1, 3), 3)
        self.assertEqual(which_ge(x, 1, 4), 0)
        self.assertEqual(which_ge(np.array([], dtype=np.int64), 1, 1), 0)

    def test_min_rec_equal_sizes(self):
        from diive.flux.lowres._ustar_rp_kernels import bin_equal_values_min_rec
        ids = bin_equal_values_min_rec(np.arange(40.0), 20, 1e-14)
        np.testing.assert_array_equal(ids, np.repeat(np.arange(1, 21), 2))

    def test_min_rec_ties_extend_class(self):
        from diive.flux.lowres._ustar_rp_kernels import bin_equal_values_min_rec
        x = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
        ids = bin_equal_values_min_rec(x, 5, 1e-14)
        np.testing.assert_array_equal(ids, [1, 1, 1, 2, 2, 3, 3, 4, 4, 4])

    def test_min_rec_short_rest_merged(self):
        from diive.flux.lowres._ustar_rp_kernels import bin_equal_values_min_rec
        # 3 records left over, less than 90 % of 5: they join class 3
        x = np.array([0.0] * 7 + list(range(1, 14)), dtype=float)
        ids = bin_equal_values_min_rec(x, 4, 1e-14)
        np.testing.assert_array_equal(ids, [1] * 7 + [2] * 5 + [3] * 8)

    def test_min_rec_ids_capped(self):
        from diive.flux.lowres._ustar_rp_kernels import bin_equal_values_min_rec
        # class size 1 gives 39 classes; ids above 20 become 20
        ids = bin_equal_values_min_rec(np.arange(39.0), 20, 1e-14)
        np.testing.assert_array_equal(ids, list(range(1, 20)) + [20] * 20)

    def test_balanced_equal_sizes_and_ties(self):
        from diive.flux.lowres._ustar_rp_kernels import OK, bin_equal_values_balanced
        ids, st = bin_equal_values_balanced(np.arange(21.0), 7, 1e-8)
        self.assertEqual(st, OK)
        np.testing.assert_array_equal(ids, np.repeat(np.arange(1, 8), 3))
        # five equal values: class 2 starts at the first change and gets shorter
        x = np.array([0.0] * 5 + list(range(1, 17)), dtype=float)
        ids, st = bin_equal_values_balanced(x, 7, 1e-8)
        np.testing.assert_array_equal(ids, [1] * 5 + [2] + [3] * 3 + [4] * 3 + [5] * 3 + [6] * 3 + [7] * 3)

    def test_balanced_no_change_left_is_r_error(self):
        from diive.flux.lowres._ustar_rp_kernels import ERR_TEMP_BREAK, bin_equal_values_balanced
        _, st = bin_equal_values_balanced(np.array([0.0] * 10 + [1.0] * 11), 7, 1e-8)
        self.assertEqual(st, ERR_TEMP_BREAK)


class TestFw2(unittest.TestCase):

    @staticmethod
    def fw2(nee):
        from diive.flux.lowres._ustar_rp_kernels import fw2
        nee = np.asarray(nee, dtype=float)
        ust = np.round(np.arange(1, nee.size + 1) * 0.01, 2)
        return fw2(nee, ust, nee.size, 10, 0.95, 3)

    def test_plateau(self):
        self.assertEqual(self.fw2([1, 2, 3, 4] + [10] * 16), 0.05)

    def test_next_class_must_also_reach_plateau(self):
        # class 5 passes, class 6 does not: the threshold moves to class 7
        self.assertEqual(self.fw2([1, 1, 1, 1, 10, 1] + [10] * 14), 0.07)

    def test_window_cut_at_last_class(self):
        self.assertEqual(self.fw2(list(range(1, 17)) + [20] * 4), 0.17)
        # the threshold must leave at least 3 classes after it
        self.assertTrue(np.isnan(self.fw2(list(range(1, 18)) + [20] * 3)))

    def test_flat_nee_gives_first_class(self):
        self.assertEqual(self.fw2([1] * 20), 0.01)

    def test_too_few_classes(self):
        self.assertTrue(np.isnan(self.fw2([1, 10, 10])))


class TestSeasons(unittest.TestCase):

    def test_seasons_cross_the_year(self):
        from diive.flux.lowres.ustar_reddyproc_detection import (_year_of_seasons,
                                                                  season_factor_month)
        t = pd.date_range('2016-01-01 00:15', '2017-12-31 23:45', freq='30min')
        s = season_factor_month(t)
        codes = np.unique(s)
        np.testing.assert_array_equal(codes, [2016001, 2016003, 2016006, 2016009, 2016012,
                                              2017003, 2017006, 2017009, 2017012])
        for ts, code in [('2016-12-15 12:15', 2016012), ('2017-02-15 12:15', 2016012),
                         ('2017-12-15 12:15', 2017012)]:
            self.assertEqual(s[t.get_loc(pd.Timestamp(ts))], code)
        idx = np.searchsorted(codes, s)
        t_sec = ((t - pd.Timestamp(0)) / pd.Timedelta(seconds=1)).to_numpy(float)
        years = _year_of_seasons(idx, t_sec, len(codes))
        # Dec 2016 - Feb 2017 has its midpoint in January 2017
        np.testing.assert_array_equal(years, [2016] * 4 + [2017] * 5)

    def test_start_months_with_january(self):
        from diive.flux.lowres.ustar_reddyproc_detection import season_factor_month
        t = pd.date_range('2016-01-01 00:15', '2016-12-31 23:45', freq='30min')
        np.testing.assert_array_equal(np.unique(season_factor_month(t, (1, 4, 7, 10))),
                                      [2016001, 2016004, 2016007, 2016010])


class TestAgainstReddyProc(unittest.TestCase):
    """Bitwise against native REddyProc 1.3.4 on the bundled CH-LAE data."""

    @classmethod
    def setUpClass(cls):
        import diive as dv
        from diive.flux.lowres.ustar_reddyproc_detection import UstarReddyProcDetection
        cls.fx = np.load(FIXTURE)
        df = dv.load_exampledata_parquet_lae()
        cls.df16 = df.loc['2016', list(LAE_COLS.values())]
        cls.df17 = df.loc['2017', list(LAE_COLS.values())]
        cls.det16 = UstarReddyProcDetection(cls.df16, **LAE_COLS, verbose=0).run()
        cls.det17 = UstarReddyProcDetection(cls.df17, **LAE_COLS, verbose=0).run()

    def _check_single(self, det, key):
        fx = self.fx
        np.testing.assert_array_equal(det.thresholds_['ustar'].to_numpy(), fx[f'{key}_ustar_th'])
        np.testing.assert_array_equal(det.temp_class_thresholds_.columns.to_numpy(),
                                      fx[f'{key}_temp_class_seasons'])
        np.testing.assert_array_equal(det.temp_class_thresholds_.to_numpy(), fx[f'{key}_temp_class'])
        self.assertEqual(int(det.records_['NIGHT_VALID'].sum()), int(fx[f'{key}_n_valid'][0]))

    def test_single_estimate_2017(self):
        self._check_single(self.det17, 'lae2017')

    def test_single_estimate_2016_pooled_year(self):
        # 2016 has fewer than 3000 night records: the year is estimated pooled
        self._check_single(self.det16, 'lae2016')
        self.assertTrue(np.isfinite(self.det16.years_['ustar_pooled'].iloc[0]))
        self.assertEqual(self.det16.years_['ustar'].iloc[0], self.det16.years_['ustar_pooled'].iloc[0])

    def test_bootstrap_with_r_resamples(self):
        idx = self.fx['lae2017_boot_idx1'].astype(np.int64) - 1
        self.det17.bootstrap(indices=idx)
        np.testing.assert_array_equal(self.det17.boot_samples_.to_numpy()[1:],
                                      self.fx['lae2017_boot_rows'])

    def test_end_timestamps_give_same_result(self):
        from diive.flux.lowres.ustar_reddyproc_detection import UstarReddyProcDetection
        df = self.df17.copy()
        df.index = pd.DatetimeIndex(df.index + pd.Timedelta('15min'), name='TIMESTAMP_END')
        det = UstarReddyProcDetection(df, **LAE_COLS, verbose=0).run()
        np.testing.assert_array_equal(det.thresholds_['ustar'].to_numpy(),
                                      self.det17.thresholds_['ustar'].to_numpy())

    def test_indices_hook_and_determinism(self):
        det = self.det17
        rng = np.random.default_rng(5)
        idx = [det.resample_indices(rng) for _ in range(9)]
        det.bootstrap(indices=idx)
        a = det.boot_samples_.to_numpy()
        det.bootstrap(n_sample=10, rng=np.random.default_rng(5))
        np.testing.assert_array_equal(det.boot_samples_.to_numpy(), a)
        det.bootstrap(n_sample=10, seed=5)
        np.testing.assert_array_equal(det.boot_samples_.to_numpy(), a)
        self.assertEqual(list(det.scenarios_annual_.columns), ['season', 'uStar', 'U05', 'U50', 'U95'])
        self.assertEqual(len(det.scenarios_annual_by_year_), 1)


if __name__ == '__main__':
    unittest.main()
