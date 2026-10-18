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


class TestBlocksAtEndsAndGaps(unittest.TestCase):
    """L240: one to three records between a removal hole and a series end or an
    input gap are compared with the line from the other side of the hole. Before
    the fix the middle of the block stayed."""

    N = 144 * 14
    GAP = (1000, 1005)  # input gap, records 1000-1004

    def _rejected(self, bad, gap=None, seed=1):
        s = _diurnal(self.N, '10min', seed=seed)
        if gap:
            s.iloc[gap[0]:gap[1]] = np.nan
        s.iloc[bad] += 10
        return np.flatnonzero((_flags(s, repeat=True) == 2).to_numpy())

    def test_blocks_are_removed_with_their_good_neighbours(self):
        N, (g0, g1) = self.N, self.GAP
        cases = {
            'start, bad 1-3': (range(1, 4), None),
            'start, bad 0-2': (range(0, 3), None),
            'start, bad 1-4': (range(1, 5), None),
            'start, bad 0-3': (range(0, 4), None),
            'after gap, bad +1..+3': (range(g1 + 1, g1 + 4), self.GAP),
            'after gap, bad 0..+2': (range(g1, g1 + 3), self.GAP),
            'after gap, bad 0..+3': (range(g1, g1 + 4), self.GAP),
            'before gap, bad -4..-2': (range(g0 - 4, g0 - 1), self.GAP),
            'before gap, bad -3..-1': (range(g0 - 3, g0), self.GAP),
            'before gap, bad -4..-1': (range(g0 - 4, g0), self.GAP),
            'end, bad -4..-2': (range(N - 4, N - 1), None),
            'end, bad -3..-1': (range(N - 3, N), None),
            'end, bad -4..-1': (range(N - 4, N), None),
        }
        for label, (bad, gap) in cases.items():
            with self.subTest(label):
                bad = list(bad)
                # The block and the good record on either side, never a record
                # of the input gap or outside the series.
                expected = [i for i in range(bad[0] - 1, bad[-1] + 2)
                            if 0 <= i < N and not (gap and gap[0] <= i < gap[1])]
                self.assertEqual(list(self._rejected(bad, gap)), expected)

    def test_line_comes_only_from_a_run_of_three(self):
        # Pass 1 leaves records 0-2 (good) at the start and 5-6 (bad) as the rest
        # of the block. The line for 0-2 must not run through 5-6.
        self.assertEqual(list(self._rejected(list(range(4, 8)))), list(range(3, 9)))

    def test_level_shift_before_a_gap(self):
        # Documented limit: the records between the step and an input gap that
        # follows within four records look like a bad block and go too.
        s = _diurnal(1440 * 4, '1min')
        t = len(s) // 2
        s.iloc[t:] += 5.0
        s.iloc[t + 4:t + 7] = np.nan
        rejected = np.flatnonzero((_flags(s, repeat=True) == 2).to_numpy())
        self.assertEqual(list(rejected - t), [-1, 0, 1, 2, 3])


class TestStepwiseChain(unittest.TestCase):
    """L240 in the notebook path: a record removed by absolute limits is an input
    gap for Hampel. Before the fix the middle of the block next to it stayed."""

    def test_block_next_to_an_abslim_removal(self):
        from diive.preprocessing.outlier_detection import StepwiseOutlierDetection
        idx = pd.date_range('2026-03-01 00:05', '2026-03-30 23:55', freq='10min', name='TIMESTAMP_MIDDLE')
        hour = idx.hour + idx.minute / 60
        values = 5 + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + np.random.default_rng(4).normal(0, 0.1, len(idx))
        df = pd.DataFrame({'TA': values}, index=idx)
        t0 = idx.get_loc(pd.Timestamp('2026-03-15 12:55'))
        df.iloc[t0, 0] = -999.0
        block = [t0 + 2, t0 + 3, t0 + 4]  # one good record, then three bad
        df.iloc[block, 0] += 15
        sod = StepwiseOutlierDetection(dfin=df, col='TA', site_lat=47.4, site_lon=8.5, utc_offset=1)
        sod.flag_outliers_abslim_test(minval=-50, maxval=50)
        sod.addflag()
        sod.flag_outliers_hampel_test(window_length='7D', n_sigma=5.5)
        sod.addflag()
        cleaned = sod.series_hires_cleaned
        self.assertTrue(cleaned.iloc[block].isna().all())
        removed = np.flatnonzero((df['TA'].notna() & cleaned.isna()).to_numpy())
        self.assertEqual(list(removed - t0), [0, 1, 2, 3, 4, 5])


class TestDayNightStatistics(unittest.TestCase):
    """L239: with separate_day_night=True the median and MAD come from the
    records of the same period. Shared statistics let the quiet night set the
    MAD, and clean daytime radiation lost more than half its records."""

    @staticmethod
    def _swin(spikes=()):
        from diive.variables.radiation import potrad
        rng = np.random.default_rng(0)
        idx = pd.date_range('2026-03-01 00:05', periods=144 * 31, freq='10min', name='TIMESTAMP_MIDDLE')
        pot = potrad(timestamp_index=idx, lat=47.42, lon=8.49, utc_offset=1).to_numpy()
        cloud = np.clip(np.convolve(rng.normal(0.7, 0.25, len(idx)), np.ones(3) / 3, 'same'), 0.1, 1)
        values = pot * cloud + rng.normal(0, 1.2, len(idx)) * (pot > 0) + rng.normal(-3, 0.3, len(idx))
        values[list(spikes)] += 600
        return pd.Series(values, index=idx, name='SW_IN')

    @staticmethod
    def _hampel(series, repeat):
        ham = Hampel(series=series, window_length='7D', n_sigma=5.5, lat=47.42, lon=8.49, utc_offset=1)
        ham.calc(repeat=repeat)
        return ham

    def test_clean_daytime_radiation_is_kept(self):
        for repeat in (False, True):
            with self.subTest(repeat=repeat):
                ham = self._hampel(self._swin(), repeat)
                rejected = ham.get_flag() == 2
                day = ham.is_daytime.reindex(rejected.index, fill_value=False)
                self.assertEqual(int((rejected & day).sum()), 0)  # was 1625 / 2130
                self.assertLess(int(rejected.sum()), 100)  # sunrise/sunset records counted as night

    def test_daytime_spikes_are_caught(self):
        swin = self._swin()
        day_idx = np.flatnonzero(potrad_positive(swin))
        spikes = np.random.default_rng(1).choice(day_idx, 20, replace=False)
        flag = self._hampel(self._swin(spikes), repeat=True).get_flag()
        self.assertEqual(int((flag.iloc[spikes] == 2).sum()), 20)

    def test_global_mode_is_unchanged(self):
        # One median and MAD for all records, as before
        s = _diurnal(144 * 14, '10min', sd=0.1)
        s.iloc[[300, 900, 1500]] += 5
        flag = _flags(s, repeat=False)
        self.assertEqual(sorted(np.flatnonzero((flag == 2).to_numpy())), [299, 300, 301, 899, 900, 901, 1499, 1500, 1501])


class TestClearSkyRunEdges(unittest.TestCase):
    """L255: every test has the curvature sign of the central double difference,
    and differences stay inside one period. Before, the edge tests had the
    opposite sign: against the daytime median of its own (L239) each run edge
    looked like a spike, and repeat=True removed clean clear-sky days from both
    ends, record by record. The step from night to day at sunrise and sunset
    also cost the first and last daytime record of each day."""

    LAT, LON = 46.8, 9.86

    def _swin(self):
        from diive.variables.radiation import potrad
        idx = pd.date_range('2023-01-01 00:15', periods=48 * 59, freq='30min', name='TIMESTAMP_MIDDLE')
        pot = potrad(timestamp_index=idx, lat=self.LAT, lon=self.LON, utc_offset=1).to_numpy()
        values = 0.8 * pot + np.random.default_rng(2).normal(0, 0.5, len(idx))
        return pd.Series(values, index=idx, name='SW_IN')

    def _hampel(self, series, repeat):
        ham = Hampel(series=series, window_length=624, n_sigma=5.5, lat=self.LAT, lon=self.LON, utc_offset=1)
        ham.calc(repeat=repeat)
        return ham

    def test_clean_clear_sky_days_are_kept(self):
        for repeat in (False, True):
            with self.subTest(repeat=repeat):
                ham = self._hampel(self._swin(), repeat)
                rejected = ham.get_flag() == 2
                day = ham.is_daytime.reindex(rejected.index, fill_value=False)
                # Was 105 (repeat=False) and 920 (repeat=True) daytime records.
                self.assertEqual(int((rejected & day).sum()), 0)

    def test_repeat_does_not_cascade(self):
        s = self._swin()
        n_first = int((self._hampel(s, repeat=False).get_flag() == 2).sum())
        n_repeat = int((self._hampel(s, repeat=True).get_flag() == 2).sum())
        self.assertLessEqual(n_repeat, 1.1 * n_first + 5)

    def test_spikes_next_to_sunrise_and_sunset_are_caught(self):
        # The first and last daytime record are tested against the line from
        # records of their own period.
        s = self._swin()
        ham = self._hampel(s, repeat=False)
        day = ham.is_daytime.reindex(s.index, fill_value=False).to_numpy().astype(int)
        firsts = np.flatnonzero(np.diff(day) == 1) + 1
        lasts = np.flatnonzero(np.diff(day) == -1)
        spikes = np.concatenate([firsts[::4], lasts[2::4]])
        s.iloc[spikes] += np.where(np.arange(len(spikes)) % 2, -60, 60)
        flag = self._hampel(s, repeat=True).get_flag()
        self.assertTrue((flag.iloc[spikes] == 2).all())
        rejected_day = (flag == 2) & ham.is_daytime.reindex(s.index, fill_value=False)
        self.assertLessEqual(int(rejected_day.sum()), 3 * len(spikes))

    def test_band_in_data_units_matches_the_decision(self):
        # The extrapolation tests have a negative factor, so upper and lower
        # swap when the band is mapped back to data units.
        s = self._swin()
        s.iloc[[100, 101, 600, 1500, 1501, 1502]] = np.nan
        s.iloc[[102, 603, 1503, 2000]] += 80
        ham = self._hampel(s, repeat=False)
        lower, upper = ham.last_lower_bound, ham.last_upper_bound
        has_band = lower.notna() & upper.notna()
        self.assertTrue((upper[has_band] >= lower[has_band]).all())
        x = s.reindex(lower.index)
        outside = ((x > upper) | (x < lower)) & has_band
        flagged = (ham.get_flag() == 2).reindex(lower.index)
        self.assertTrue((outside == flagged).all())


def potrad_positive(series):
    from diive.variables.radiation import potrad
    return potrad(timestamp_index=series.index, lat=47.42, lon=8.49, utc_offset=1).to_numpy() > 50


if __name__ == '__main__':
    unittest.main()
