"""Hampel with double differences and repeat=True must not eat into good data (L226).

Records removed by an earlier iteration used to be bridged with the
equal-spacing formula. On a diurnal ramp or across a level shift that looks
like a spike, so each iteration removed the next record. Removed records now end
a run like an input gap does.
"""
import unittest

import numpy as np
import pandas as pd

from diive.preprocessing.outlier_detection.hampel import Hampel


def _diurnal(periods, freq, sd=0.1, seed=1):
    idx = pd.date_range('2024-03-01 00:05', periods=periods, freq=freq, name='TIMESTAMP_MIDDLE')
    hour = idx.hour + idx.minute / 60
    values = 15 + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + np.random.default_rng(seed).normal(0, sd, len(idx))
    return pd.Series(values, index=idx, name='TA')


def _flags(series, repeat, **kwargs):
    opts = dict(window_length='7D', n_sigma=5.5, use_differencing=True, separate_day_night=False)
    opts.update(kwargs)
    ham = Hampel(series=series, **opts)
    ham.calc(repeat=repeat)
    return ham.get_flag()


class TestNoCascade(unittest.TestCase):

    def test_level_shift(self):
        # A sensor exchange: +5 degC from the middle on. Before the fix 1-min
        # data lost about 1700 records.
        for shift in (2.0, 5.0):
            with self.subTest(shift=shift):
                s = _diurnal(1440 * 4, '1min')
                start = len(s) // 2
                s.iloc[start:] += shift
                first = _flags(s, repeat=False)
                flag = _flags(s, repeat=True)
                rejected = np.flatnonzero((flag == 2).to_numpy())
                self.assertEqual(list(rejected), list(np.flatnonzero((first == 2).to_numpy())))
                self.assertEqual(list(rejected), [start - 1, start])

    def test_block_of_bad_values(self):
        # Seven bad 10-min records (a logger glitch). Before the fix 134 records
        # were rejected. Now only the block edges and their good neighbours.
        s = _diurnal(144 * 14, '10min')
        start = len(s) // 2
        s.iloc[start:start + 7] = 0.0
        flag = _flags(s, repeat=True)
        rejected = np.flatnonzero((flag == 2).to_numpy())
        self.assertEqual(list(rejected - start), [-1, 0, 6, 7])

    def test_smooth_radiation_ramp(self):
        # Clear-sky shortwave radiation: a kink at sunrise and sunset and a
        # strong curvature in between. Before the fix almost every record went.
        idx = pd.date_range('2026-03-01 00:05', periods=144 * 31, freq='10min', name='TIMESTAMP_MIDDLE')
        hour = idx.hour + idx.minute / 60
        clear_sky = np.clip(800 * np.sin(2 * np.pi * (hour - 6) / 24), 0, None)
        s = pd.Series(clear_sky - 4 + np.random.default_rng(0).normal(0, 0.4, len(idx)), index=idx, name='SW_IN')
        n_first = int((_flags(s, repeat=False) == 2).sum())
        n_repeat = int((_flags(s, repeat=True) == 2).sum())
        self.assertLessEqual(n_repeat, 1.1 * n_first + 5)


class TestBlocks(unittest.TestCase):

    def test_blocks_of_three_and_four_are_removed(self):
        # The record or two left inside the block after the first pass are
        # compared with the line across the holes.
        for length in (3, 4):
            with self.subTest(length=length):
                s = _diurnal(144 * 14, '10min')
                start = len(s) // 2
                s.iloc[start:start + length] = 25.0
                flag = _flags(s, repeat=True)
                rejected = np.flatnonzero((flag == 2).to_numpy())
                self.assertEqual(list(rejected - start), list(range(-1, length + 1)))

    def test_longer_block_keeps_its_inner_values(self):
        # Documented limit: the inner values of a block of five or more are
        # consistent with each other. Such runs belong to absolute limits or
        # manual removal.
        s = _diurnal(144 * 14, '10min')
        start = len(s) // 2
        s.iloc[start:start + 5] = 25.0
        flag = _flags(s, repeat=True)
        rejected = np.flatnonzero((flag == 2).to_numpy())
        self.assertEqual(list(rejected - start), [-1, 0, 4, 5])

    def test_input_gap_is_not_bridged(self):
        # Two records between an input gap and a removed spike form a short run.
        # It must not be compared with the line across the input gap, which here
        # also spans a genuine level shift.
        rng = np.random.default_rng(7)
        idx = pd.date_range('2020-05-01', periods=2000, freq='1min', name='TIMESTAMP_MIDDLE')
        s = pd.Series(20 + rng.normal(0, 0.05, len(idx)), index=idx, name='SWC')
        s.iloc[1000:1003] = np.nan
        s.iloc[1003:] += 5.0
        s.iloc[1006] += 30.0
        flag = _flags(s, repeat=True, window_length=60, n_sigma=8)
        self.assertEqual(list(np.flatnonzero((flag == 2).to_numpy())), [1005, 1006, 1007])


if __name__ == '__main__':
    unittest.main()
