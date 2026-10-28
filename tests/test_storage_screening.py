"""Spike screening of the storage term (screen_storage_term=True), L413.

The double difference of a spike also flags its two neighbours, so the run-length rule
judges the run without its two ends. TestStorageScreening and TestStorageRestarts in
test_postprocessingchain.py cover the rest of the screening.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro

FILLED = 'SC_SINGLE_gfRMED_L3.1'


def _df(freq: str = '30min', days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    per_day = int(pd.Timedelta('1D') / pd.Timedelta(freq))
    index = pd.date_range('2022-07-01 00:15', periods=per_day * days, freq=freq,
                          name='TIMESTAMP_MIDDLE')
    return pd.DataFrame({'FC': rng.normal(-5, 2, len(index)),
                         'SC_SINGLE': rng.normal(0, 0.5, len(index))}, index=index)


def _screen(df: pd.DataFrame, **kwargs) -> FluxStorageCorrectionSinglePointEddyPro:
    sc = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC', basevar='CO2',
                                                  screen_storage_term=True, **kwargs)
    sc.storage_correction()
    return sc


class TestSpikeCore(unittest.TestCase):

    def _check_block(self, sc, df, spikes, n_block):
        r = sc.results
        self.assertEqual(list(sc.storage_outliers_), list(spikes))
        self.assertEqual(len(sc.storage_replaced_), n_block)
        i0 = df.index.get_loc(spikes[0])
        self.assertEqual(list(sc.storage_replaced_), list(df.index[i0 - 1:i0 - 1 + n_block]))
        block = sc.storage_replaced_
        self.assertTrue((r.loc[block, sc.flag_isgapfilled] == 2).all())
        # The block keeps its sum, and every measured flux keeps a storage term.
        self.assertAlmostEqual(r.loc[block, FILLED].sum(), df.loc[block, 'SC_SINGLE'].sum(), places=9)
        self.assertEqual(int(r['NEE_L3.1'].notna().sum()), int(df['FC'].notna().sum()))
        return r

    def test_single_spike_is_spread_over_three_records(self):
        df = _df()
        spike = df.index[700]
        df.loc[spike, 'SC_SINGLE'] = 60.0
        sc = _screen(df)
        r = self._check_block(sc, df, [spike], 3)
        self.assertAlmostEqual(r.loc[spike, FILLED], 20.0, delta=1.0)

    def test_concentration_spike_is_replaced(self):
        """+X then -X (backward difference) flags 4 records, of which 2 are spikes."""
        df = _df()
        pair = df.index[700:702]
        df.loc[pair, 'SC_SINGLE'] = [60.0, -60.0]
        sc = _screen(df)
        r = self._check_block(sc, df, pair, 4)
        self.assertLess(r.loc[sc.storage_replaced_, FILLED].abs().max(), 1.0)

    def test_two_records_of_the_same_sign_are_replaced(self):
        df = _df()
        pair = df.index[700:702]
        df.loc[pair, 'SC_SINGLE'] = [30.0, 30.0]
        sc = _screen(df)
        r = self._check_block(sc, df, pair, 4)
        self.assertAlmostEqual(r.loc[pair[0], FILLED], 15.0, delta=1.0)

    def test_max_run_counts_spikes_not_flagged_records(self):
        df = _df()
        three = df.index[700:703]
        df.loc[three, 'SC_SINGLE'] = [40.0, -40.0, 40.0]
        four = df.index[900:904]
        df.loc[four, 'SC_SINGLE'] = [40.0, -40.0, 40.0, -40.0]
        sc = _screen(df)
        # Three spikes in a row (5 flagged records) are replaced, four stay as measured.
        self.assertEqual(list(sc.storage_outliers_), list(three))
        self.assertTrue(sc.results.loc[four, sc.flag_isgapfilled].eq(0).all())
        self.assertEqual(list(sc.results.loc[four, FILLED]), [40.0, -40.0, 40.0, -40.0])
        # With max_run=4 the four are replaced too.
        sc = _screen(df, storage_screening={'max_run': 4})
        self.assertTrue(set(four) <= set(sc.storage_outliers_))

    def test_hourly_data(self):
        df = _df(freq='60min')
        pair = df.index[350:352]
        df.loc[pair, 'SC_SINGLE'] = [60.0, -60.0]
        sc = _screen(df)
        self._check_block(sc, df, pair, 4)

    def test_spikes_at_the_series_ends(self):
        df = _df()
        df.iloc[0, 1] = 80.0
        df.iloc[-1, 1] = -80.0
        sc = _screen(df)
        r = sc.results
        for rec in (df.index[0], df.index[-1]):
            self.assertIn(rec, sc.storage_outliers_)
            self.assertEqual(r.loc[rec, sc.flag_isgapfilled], 2)
            self.assertLess(abs(r.loc[rec, FILLED]), 45)
        self.assertAlmostEqual(r[FILLED].sum(), df['SC_SINGLE'].sum(), places=9)

    def test_all_nan_storage(self):
        df = _df()
        df['SC_SINGLE'] = np.nan
        sc = _screen(df)
        self.assertEqual(len(sc.storage_outliers_), 0)
        self.assertTrue((sc.results[sc.flag_isgapfilled] == 1).all())
        self.assertEqual(int(sc.results['NEE_L3.1'].notna().sum()), len(df))


class TestRealStorageEventsStay(unittest.TestCase):

    def test_sunrise_flush_is_not_replaced(self):
        """Several hours of large positive storage are a real event, not a spike."""
        df = _df()
        flush = df.index[48 * 15 + 12:48 * 15 + 26]  # 06:15 to 12:45
        df.loc[flush, 'SC_SINGLE'] += [5, 10, 15, 20, 20, 20, 20, 20, 20, 20, 20, 15, 10, 5]
        sc = _screen(df)
        self.assertFalse(flush.isin(sc.storage_replaced_).any())
        self.assertTrue((sc.results.loc[flush, sc.flag_isgapfilled] == 0).all())

    def test_daily_morning_flush_is_not_replaced(self):
        df = _df()
        hour = df.index.hour + df.index.minute / 60
        morning = (hour > 6) & (hour < 12)
        df['SC_SINGLE'] += np.where(morning, 15 * np.sin(np.pi * (hour - 6) / 6), 0)
        sc = _screen(df)
        self.assertFalse(df.index[morning].isin(sc.storage_replaced_).any())


class TestReportCountsSpikes(unittest.TestCase):

    def test_report_separates_spikes_from_replaced_values(self):
        df = _df()
        for i in (300, 700, 1100):
            df.iloc[i, 1] = 60.0
        sc = _screen(df)
        self.assertEqual(len(sc.storage_outliers_), 3)
        self.assertEqual(len(sc.storage_replaced_), 9)
        with mock.patch('diive.flux.lowres.storage_correction.info') as info:
            sc.report()
        line = next(c.args[0] for c in info.call_args_list if c.args[0].startswith('Screening'))
        self.assertIn('3 storage spike(s)', line)
        self.assertIn('9 value(s) replaced', line)


if __name__ == '__main__':
    unittest.main()
