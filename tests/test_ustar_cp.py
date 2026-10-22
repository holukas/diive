"""Tests for the ONEFlux change-point u* threshold port (UstarChangePointDetection)."""

import unittest

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import PchipInterpolator

from diive.flux.lowres._ustar_cp_core import (
    assign_annual,
    december_flags,
    find_change_point,
)
from diive.flux.lowres._ustar_cp_numerics import (
    fmax2p_cp2,
    matlab_colon,
    matlab_default_stream,
    matlab_pchip,
    matlab_prctile,
    matlab_randi_positions,
    matlab_round,
    sequential_sum,
)
from diive.flux.lowres.ustar_cp_detection import UstarChangePointDetection

COLS = dict(nee_col='NEE', ta_col='TA', ustar_col='USTAR', swin_col='SW_IN')


def _synthetic_year(threshold=0.3, year=2021, seed=1, slope=True):
    """Half-hourly year (TIMESTAMP_MIDDLE) with night NEE limited below ``threshold``."""
    idx = pd.date_range(f'{year}-01-01 00:15', f'{year}-12-31 23:45', freq='30min')
    n = len(idx)
    rng = np.random.default_rng(seed)
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    doy = idx.dayofyear.to_numpy()
    swin = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 600
    ta = 8 - 10 * np.cos(2 * np.pi * (doy - 15) / 365) + 3 * np.sin((hour - 9) / 24 * 2 * np.pi) \
        + rng.normal(0, 1.5, n)
    ustar = rng.gamma(2.5, 0.15, n)
    reco = 1.5 * np.exp(0.07 * ta)
    if slope:
        nee = reco * np.minimum(ustar / threshold, 1.0)
    else:
        nee = reco.copy()
    nee = nee + rng.normal(0, 0.4, n)
    return pd.DataFrame({'NEE': nee, 'TA': ta, 'USTAR': ustar, 'SW_IN': swin}, index=idx)


class TestNumerics(unittest.TestCase):

    def test_round_halves_away_from_zero(self):
        np.testing.assert_array_equal(matlab_round([0.5, 1.5, 2.5, -2.5, 2.4]), [1, 2, 3, -3, 2])

    def test_prctile_hand_values(self):
        # Positions 100*(i-0.5)/n: 12.5, 37.5, 62.5, 87.5 for n = 4
        x = [4.0, 1.0, 3.0, 2.0]
        y = matlab_prctile(x, [0, 10, 12.5, 30, 50, 100])
        np.testing.assert_array_equal(y, [1.0, 1.0, 1.0, 1.7, 2.5, 4.0])
        # Equal neighbours are copied, NaN ignored
        np.testing.assert_array_equal(matlab_prctile([1.0, 2.0, 2.0, np.nan, 3.0], [50]), [2.0])

    def test_colon_symmetric_and_exact_end(self):
        d = 100 / 7
        v = matlab_colon(0.0, d, 100.0)
        self.assertEqual(v.size, 8)
        self.assertEqual(v[0], 0.0)
        self.assertEqual(v[-1], 100.0)
        self.assertEqual(v[3], 3 * d)      # first half from the start
        self.assertEqual(v[5], 100.0 - 2 * d)  # second half from the end
        np.testing.assert_array_equal(matlab_colon(0.0, 25.0, 100.0), [0, 25, 50, 75, 100])

    def test_pchip_hand_value(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])
        # Slopes 0, 1.5, 3.75, 6 (pchipslopes / pchipend), Hermite cubic at 1.5
        self.assertAlmostEqual(matlab_pchip(x, y, 1.5), 2.21875, places=14)
        ref = PchipInterpolator(x, y)
        for xi in (0.3, 1.0, 2.2, 2.9):
            self.assertAlmostEqual(matlab_pchip(x, y, xi), float(ref(xi)), places=12)

    def test_fmax_p_values(self):
        # Fmax on a table node gives 1 - pTable in double precision
        self.assertEqual(fmax2p_cp2(6.2992, 10), 1 - 0.90)
        p95 = fmax2p_cp2(9.1471, 10)
        self.assertEqual(p95, 1 - 0.95)
        self.assertGreater(p95, 0.05)  # 0.050000000000000044: not significant
        # Below the table: F(3, n) scaled to the 80 % critical value
        exp_lo = 2 * (1 - stats.f.cdf(stats.f.ppf(0.90, 3, 10) * 2.0 / 3.9293, 3, 10))
        self.assertAlmostEqual(fmax2p_cp2(2.0, 10), min(exp_lo, 1.0), places=10)
        # Above the table: scaled by the 95 % value (cpdFmax2pCp2.m:46)
        exp_hi = 2 * (1 - stats.f.cdf(stats.f.ppf(0.995, 3, 10) * 20.0 / 9.1471, 3, 10))
        self.assertAlmostEqual(fmax2p_cp2(20.0, 10), exp_hi, places=10)
        self.assertTrue(np.isnan(fmax2p_cp2(5.0, 9)))

    def test_matlab_stream_and_randi(self):
        # rng default; rand(1,5) and randi(10,1,5) from the MATLAB documentation
        u = matlab_default_stream().random_sample(5)
        np.testing.assert_allclose(u, [0.8147, 0.9058, 0.1270, 0.9134, 0.6324], atol=5e-5)
        pos = matlab_randi_positions(matlab_default_stream(), 10, 'ceil')[:5] + 1
        np.testing.assert_array_equal(pos, [9, 10, 2, 10, 7])

    def test_sequential_sum(self):
        x = np.array([[0.1] * 20 + [1e16, -1e16]])
        s = 0.0
        for v in x[0]:
            s += v
        self.assertEqual(sequential_sum(x, axis=1)[0], s)

    def test_december_in_matlab_year_zero(self):
        # Non-leap year: real 1 December (DOY 335) is still November in year 0
        t = np.cumsum(np.r_[1 + 1 / 48, np.full(17519, 1 / 48)])
        dec = december_flags(t)
        first = int(np.flatnonzero(dec)[0])
        self.assertEqual(first, 335 * 48)  # first record after t = 336.0
        self.assertTrue(dec[-1])

    def test_december_boundary_hourly(self):
        # Hourly t accumulates 1.3e-11 days above 336 at row 8040. datevec rounds to
        # the millisecond, so that record stays in November (ONEFlux, L371).
        for year, nt in ((2019, 8760), (2020, 8784)):
            idx = pd.date_range(f'{year}-01-01 01:00', periods=nt, freq='60min')
            df = pd.DataFrame({'NEE': 1.0, 'TA': 5.0, 'USTAR': 0.3, 'SW_IN': 0.0}, index=idx)
            det = UstarChangePointDetection(df, **COLS)
            t, dec = det._t, det._dec
            self.assertGreater(t[8039], 336.0)
            self.assertFalse(dec[8039])
            self.assertEqual(int(np.flatnonzero(dec)[0]), 8040)
            # The last record (t = 366 or 367, also a bit above) is 24:00 of 31 December
            self.assertGreater(t[-1], 1.0 + nt / 24)
            self.assertTrue(dec[-1])
            self.assertEqual(int(dec.sum()), (nt // 24 - 335) * 24)


class TestChangePoint(unittest.TestCase):

    def test_recovers_known_change_point(self):
        x = np.linspace(0.05, 1.0, 50)
        y = np.minimum(x / 0.4, 1.0) * 5 + np.random.default_rng(3).normal(0, 0.1, 50)
        cp, s = find_change_point(x, y)
        self.assertAlmostEqual(cp, 0.4, delta=0.05)
        self.assertLess(s['p'], 0.05)
        self.assertGreater(s['b1'], 0)
        self.assertEqual(s['n'], 50)

    def test_too_few_points(self):
        cp, s = find_change_point(np.arange(9.0), np.arange(9.0))
        self.assertTrue(np.isnan(cp))
        self.assertTrue(np.isnan(s['n']))

    def test_change_point_at_end_not_stored(self):
        # A ramp without a plateau: the best change point lies in the end zone
        # (last max(3, floor(0.05 n)) points), so the statistics stay NaN
        x = np.linspace(0.05, 1.0, 40)
        y = 2 * x + np.random.default_rng(0).normal(0, 0.01, 40)
        cp, s = find_change_point(x, y)
        self.assertTrue(np.isnan(s['Cp']))
        self.assertTrue(np.isnan(s['b1']))
        self.assertEqual(s['n'], 40)


class TestAssign(unittest.TestCase):

    def _stats(self, n_boot=2):
        shape = (n_boot, 4, 8)
        k = np.prod(shape)
        cp = np.linspace(0.1, 0.5, k).reshape(shape)
        b1 = (0.3 + np.linspace(-0.01, 0.01, k)).reshape(shape)
        cib1 = np.linspace(0.0, 100.0, k).reshape(shape)
        p = np.full(shape, 0.01)
        mt = np.ones(shape)
        return cp, b1, cib1, p, mt

    def test_column_bug_reproduced(self):
        cp, b1, cib1, p, mt = self._stats()
        res = assign_annual(cp, b1, cib1, p, mt)
        self.assertEqual(res['failure'], '')
        # Cp is screened against the median and IQR of b1 (the bug): only Cp
        # within 5 IQR(b1) of median(b1) survive
        b1f = b1.ravel()
        q = matlab_prctile(b1f, [25, 75])
        expected = np.abs(cp - np.median(b1f)) / (q[1] - q[0]) <= 5
        np.testing.assert_array_equal(res['selected'], expected)
        self.assertLess(res['selected'].sum(), cp.size)
        # A correct screen of b1 by its own spread would keep every b1
        self.assertTrue(np.all(np.abs(b1f - np.median(b1f)) / (q[1] - q[0]) <= 5))
        for b in range(2):
            np.testing.assert_allclose(res['cp_annual'][b], cp[b][expected[b]].mean(), rtol=1e-14)

    def test_failure_messages(self):
        cp, b1, cib1, p, mt = self._stats()
        p_bad = np.full_like(p, 0.5)
        self.assertEqual(assign_annual(cp, b1, cib1, p_bad, mt)['failure'],
                         'Less than 10% successful detections. ')
        nan = np.full_like(cp, np.nan)
        res = assign_annual(nan, nan, nan, nan, nan)
        self.assertEqual(res['failure'], 'Too few selected change points: 0/8')
        self.assertTrue(np.isnan(res['cp_annual']).all())

    def test_mode_e_majority(self):
        cp, b1, cib1, p, mt = self._stats()
        res = assign_annual(cp, -b1, cib1, p, mt)
        self.assertEqual(res['mode'], 'E')


class TestDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = _synthetic_year()

    def test_rejects_partial_year(self):
        with self.assertRaises(ValueError):
            UstarChangePointDetection(self.df.iloc[:-5], **COLS)

    def test_bootstrap_recovers_threshold(self):
        det = UstarChangePointDetection(self.df, **COLS)
        v = det.bootstrap_annual_samples(n_iter=10, rng=7)
        self.assertEqual(det.failure_, '')
        self.assertEqual(det.mode_, 'D')
        self.assertEqual(v.shape, (10,))
        self.assertAlmostEqual(np.nanmedian(v), 0.3, delta=0.06)
        self.assertEqual(det.bootstrap_cp2_.shape, (10, 4, 8))

    def test_no_change_point_mostly_rejected(self):
        det = UstarChangePointDetection(_synthetic_year(slope=False, seed=2), **COLS)
        det.bootstrap_annual_samples(n_iter=5, rng=3)
        sig = np.isfinite(det.bootstrap_cp2_).sum() / np.isfinite(det.bootstrap_stats_['mt']).sum()
        self.assertLess(sig, 0.2)

    def test_detect_unresampled(self):
        det = UstarChangePointDetection(self.df, **COLS)
        res = det.detect()
        self.assertEqual(res.index.names, ['season', 'stratum'])
        self.assertGreaterEqual(len(res), 16)
        self.assertAlmostEqual(det.annual_threshold_, 0.3, delta=0.08)
        self.assertEqual(det.detect_failure_, '')

    def test_record_minimum(self):
        df = self.df.copy()
        night = df['SW_IN'] < 5
        drop = night & (np.arange(len(df)) % 3 != 0)  # keep a third of the nights
        df.loc[drop, 'NEE'] = np.nan
        det = UstarChangePointDetection(df, **COLS)
        stream = matlab_default_stream()
        state = stream.get_state()[1].copy()
        v = det.bootstrap_annual_samples(n_iter=100, rng=stream)
        self.assertFalse(det.bootstrap_ran_)
        self.assertEqual(det.failure_, 'Too few selected change points: 0/400')
        self.assertTrue(np.isnan(v).all())
        # A skipped year draws no random numbers, so the MATLAB stream is untouched
        np.testing.assert_array_equal(stream.get_state()[1], state)

    def test_empty_nee(self):
        df = self.df.copy()
        df['NEE'] = -9999.0
        det = UstarChangePointDetection(df, **COLS)
        v = det.bootstrap_annual_samples(n_iter=3)
        self.assertEqual(det.failure_, 'NEE is empty!')
        self.assertTrue(np.isnan(v).all())

    def test_indices_hook(self):
        det = UstarChangePointDetection(self.df, **COLS)
        pos = det.draw_positions(4, rng=11)
        a = det.bootstrap_annual_samples(rng=11, n_iter=4)
        cp_a = det.bootstrap_cp2_.copy()
        rng = np.random.default_rng(0)
        b = det.bootstrap_annual_samples(indices=rng.permuted(pos, axis=1))  # rows get sorted
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(cp_a, det.bootstrap_cp2_)
        # The identity draw is the unresampled evaluation of detect()
        det.bootstrap_annual_samples(indices=np.arange(len(self.df))[None, :])
        res = det.detect()
        got = det.bootstrap_cp2_[0][np.isfinite(det.bootstrap_stats_['mt'][0])]
        np.testing.assert_array_equal(got, res['Cp2'].to_numpy())

    def test_determinism(self):
        det = UstarChangePointDetection(self.df, **COLS)
        a = det.bootstrap_annual_samples(n_iter=3, rng=5)
        b = det.bootstrap_annual_samples(n_iter=3, rng=5)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(det.draw_positions(2), det.draw_positions(2, rng='matlab'))
        # The MATLAB stream continues across calls when a RandomState is passed
        stream = matlab_default_stream()
        p1 = det.draw_positions(1, rng=stream)
        p2 = det.draw_positions(1, rng=stream)
        np.testing.assert_array_equal(np.r_[p1, p2], det.draw_positions(2))


if __name__ == '__main__':
    unittest.main()
