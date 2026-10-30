"""Tests for the u* threshold detection entry point (UstarThresholdDetection).

The methods behind it (moving point, change point, REddyProc) have their own tests,
including parity with ONEFlux and REddyProc. These tests cover what the entry point
adds: the ONEFlux percentile, the pooling of years and methods, and the input checks.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from diive.flux.lowres import ustar_detection
from diive.flux.lowres.ustar_detection import UstarThresholdDetection, oneflux_percentile


def _three_years():
    """Half-hourly dummy data for 2016 to 2018; the stubbed workers ignore the values."""
    idx = pd.date_range('2016-01-01 00:15', '2018-12-31 23:45', freq='30min',
                        name='TIMESTAMP_MIDDLE')
    return pd.DataFrame({'NEE': 1.0, 'TA': 10.0, 'USTAR': 0.3, 'SW_IN': 0.0}, index=idx)


def _fake_worker(year, method, df_year, kwargs, n_iter, seed):
    """Three draws per year and method, so every value says where it came from:
    year 2016 MP gives 16.1, 16.2, 16.3; CP gives 16.6, 16.7, 16.8."""
    base = year - 2000 + (0.0 if method == 'mp' else 0.5)
    return year, method, base + np.array([0.1, 0.2, 0.3]), None


def _run(mode='oneflux', worker=_fake_worker, **kwargs):
    det = UstarThresholdDetection(_three_years(), nee_col='NEE', ta_col='TA', ustar_col='USTAR',
                                  swin_col='SW_IN', mode=mode, **kwargs)
    with mock.patch.object(ustar_detection, '_oneflux_year_worker', worker):
        det.run()
    return det


class TestOnefluxPercentile(unittest.TestCase):
    """Nearest rank on the sorted valid values: index ROUND(p / 100 * n) - 1."""

    def test_nearest_rank_by_hand(self):
        v = np.arange(10, 0, -1, dtype=float)  # 10 values, unsorted on purpose
        self.assertEqual(oneflux_percentile(v, 50), 5.0)   # 5.0 -> rank 5
        self.assertEqual(oneflux_percentile(v, 16), 2.0)   # 1.6 -> rank 2
        self.assertEqual(oneflux_percentile(v, 84), 8.0)   # 8.4 -> rank 8
        self.assertEqual(oneflux_percentile(v, 95), 10.0)  # 9.5 rounds up -> rank 10

    def test_ends_and_missing_values(self):
        v = np.array([3.0, np.nan, 1.0, 2.0, np.inf])
        self.assertEqual(oneflux_percentile(v, 50), 2.0)   # NaN and inf left out, n = 3
        self.assertEqual(oneflux_percentile(v, 0), 1.0)    # clamped to the first value
        self.assertEqual(oneflux_percentile(v, 100), 3.0)  # and to the last
        self.assertEqual(oneflux_percentile([7.0], 16), 7.0)
        self.assertTrue(np.isnan(oneflux_percentile([np.nan], 50)))
        self.assertTrue(np.isnan(oneflux_percentile([], 50)))


class TestOnefluxPooling(unittest.TestCase):
    """VUT of year Y pools years Y-1 to Y+1, CUT pools all years, both over the methods
    of the mode (ONEFlux dataset.c get_uts_y / get_uts_c)."""

    def test_vut_pools_the_neighbouring_years_and_both_methods(self):
        det = _run()
        self.assertEqual(det.years_, [2016, 2017, 2018])
        expected = {2016: (16, 17), 2017: (16, 17, 18), 2018: (17, 18)}
        for year, years in expected.items():
            want = sorted(y + d for y in years for d in (0.1, 0.2, 0.3, 0.6, 0.7, 0.8))
            np.testing.assert_allclose(sorted(det.vut_samples_[year]), want)
        self.assertEqual(len(det.cut_samples_), 3 * 2 * 3)
        # The tables are the ONEFlux percentiles of these pools.
        for year in det.years_:
            for p in (16, 50, 84):
                self.assertEqual(det.get_vut_thresholds().loc[year, f'p{p}'],
                                 oneflux_percentile(det.vut_samples_[year], p))
        self.assertEqual(det.get_cut_threshold(),
                         {f'p{p}': oneflux_percentile(det.cut_samples_, p) for p in (16, 50, 84)})

    def test_single_method_modes_pool_only_that_method(self):
        for mode, method in (('oneflux_mp', 'mp'), ('oneflux_cp', 'cp')):
            with self.subTest(mode=mode):
                det = _run(mode=mode)
                self.assertEqual({m for y in det.years_ for m in det.samples_[y]}, {method})
                self.assertEqual(len(det.vut_samples_[2017]), 3 * 3)
                self.assertEqual(len(det.cut_samples_), 3 * 3)

    def test_a_failed_year_is_left_out_of_every_pool(self):
        def worker(year, method, df_year, kwargs, n_iter, seed):
            if year == 2017 and method == 'cp':
                return year, method, None, 'too few night records'
            return _fake_worker(year, method, df_year, kwargs, n_iter, seed)

        det = _run(worker=worker)
        self.assertEqual(det.failures_[2017], {'cp': 'too few night records'})
        self.assertNotIn('cp', det.samples_[2017])
        self.assertEqual(len(det.vut_samples_[2016]), 3 * 3)  # 2016 MP+CP, 2017 MP
        self.assertEqual(len(det.cut_samples_), 3 * 2 * 3 - 3)
        self.assertIn('CP none', det.summary())

    def test_each_year_and_method_gets_its_own_seed(self):
        seeds = {}

        def worker(year, method, df_year, kwargs, n_iter, seed):
            seeds[(year, method)] = seed
            return _fake_worker(year, method, df_year, kwargs, n_iter, seed)

        _run(worker=worker, random_state=7)
        self.assertEqual(seeds[(2016, 'mp')], [7, 2016, 0])
        self.assertEqual(seeds[(2018, 'cp')], [7, 2018, 1])
        self.assertEqual(len(set(map(tuple, seeds.values()))), 6)

    def test_each_year_is_padded_to_its_full_grid(self):
        sizes = {}

        def worker(year, method, df_year, kwargs, n_iter, seed):
            sizes[year] = len(df_year)
            return _fake_worker(year, method, df_year, kwargs, n_iter, seed)

        _run(worker=worker)
        self.assertEqual(sizes, {2016: 366 * 48, 2017: 365 * 48, 2018: 365 * 48})


class TestInputChecks(unittest.TestCase):

    def test_bad_input_raises(self):
        df = _three_years()
        kw = dict(nee_col='NEE', ta_col='TA', ustar_col='USTAR', swin_col='SW_IN')
        with self.assertRaisesRegex(ValueError, 'mode must be one of'):
            UstarThresholdDetection(df, mode='vekuri', **kw)
        with self.assertRaisesRegex(ValueError, 'Columns not found'):
            UstarThresholdDetection(df.drop(columns='TA'), **kw)
        with self.assertRaisesRegex(ValueError, 'DatetimeIndex'):
            UstarThresholdDetection(df.reset_index(), **kw)
        with self.assertRaisesRegex(RuntimeError, 'Call run'):
            UstarThresholdDetection(df, **kw).get_cut_threshold()
        with self.assertRaisesRegex(ValueError, "random_state='oneflux'"):
            UstarThresholdDetection(df, mode='reddyproc', random_state='oneflux', **kw).run()

    def test_defaults_follow_the_mode(self):
        df = _three_years()
        kw = dict(nee_col='NEE', ta_col='TA', ustar_col='USTAR', swin_col='SW_IN')
        one = UstarThresholdDetection(df, **kw)
        rp = UstarThresholdDetection(df, mode='reddyproc', **kw)
        self.assertEqual((one.n_iter, one.percentiles), (100, (16, 50, 84)))
        self.assertEqual((rp.n_iter, rp.percentiles), (200, (5, 50, 95)))


if __name__ == '__main__':
    unittest.main()
