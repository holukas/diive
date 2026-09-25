import unittest

import numpy as np
import pandas as pd

import diive as dv
import diive.configs.exampledata as ed
from diive.configs.exampledata import load_exampledata_parquet_long
from diive.core.dfun.frames import transform_yearmonth_matrix_to_longform
from diive.core.times.resampling import (
    diel_cycle,
    resample_series_to_30MIN,
    resample_series_to_freq,
)


class TestResampleToFreq(unittest.TestCase):
    """General-resolution downsampling (resample_series_to_freq)."""

    def _hires(self, freq='10min', days=3):
        idx = pd.date_range('2022-06-01 00:10:00', periods=days * 24 * 6, freq=freq)
        idx.name = 'TIMESTAMP_END'
        return pd.Series(np.arange(len(idx), dtype=float), index=idx, name='TA')

    def test_10min_to_30min_mean_end(self):
        s = self._hires()
        out = resample_series_to_freq(s, '30min', agg='mean', mincounts_perc=0.9)
        self.assertEqual(out.index.name, 'TIMESTAMP_END')
        # First 30MIN bin labelled at its END holds the first three 10-min values.
        self.assertEqual(out.index[0], pd.Timestamp('2022-06-01 00:30:00'))
        self.assertAlmostEqual(out.iloc[0], 1.0)  # mean of 0,1,2

    def test_10min_to_1h_sum(self):
        s = self._hires()
        out = resample_series_to_freq(s, '1h', agg='sum', mincounts_perc=0.9)
        self.assertEqual(out.index[0], pd.Timestamp('2022-06-01 01:00:00'))
        self.assertAlmostEqual(out.iloc[0], float(0 + 1 + 2 + 3 + 4 + 5))

    def test_30MIN_wrapper_matches_general(self):
        s = self._hires()
        self.assertTrue(
            resample_series_to_30MIN(s, agg='mean').equals(
                resample_series_to_freq(s, '30min', agg='mean')))

    def test_upsampling_rejected(self):
        s = self._hires(freq='30min')
        with self.assertRaises(NotImplementedError):
            resample_series_to_freq(s, '10min')

    def test_same_resolution_is_noop(self):
        # Processed 30MIN data "resampled" to 30min: no aggregation, values kept.
        s = self._hires(freq='30min')
        out = resample_series_to_freq(s, '30min', agg='mean', mincounts_perc=0.9)
        self.assertEqual(len(out), len(s))
        self.assertTrue(out.equals(s))
        self.assertEqual(out.index.name, 'TIMESTAMP_END')

    @staticmethod
    def _to_middle(s):
        s = s.copy()
        s.index = s.index - pd.Timedelta(s.index.freq) / 2
        s.index.name = 'TIMESTAMP_MIDDLE'
        return s

    def test_same_resolution_middle_input_returns_end(self):
        # Meteo screening hands over TIMESTAMP_MIDDLE data; uploading it under
        # its middle timestamps would shift every value half a period early.
        s_end = self._hires(freq='30min')
        out = resample_series_to_freq(self._to_middle(s_end), '30min')
        self.assertEqual(out.index.name, 'TIMESTAMP_END')
        self.assertEqual(out.index[0], pd.Timestamp('2022-06-01 00:10:00'))
        self.assertAlmostEqual(out.iloc[0], 0.0)
        self.assertTrue(out.index.equals(s_end.index))
        self.assertTrue(np.array_equal(out.to_numpy(), s_end.to_numpy()))

    def test_same_resolution_middle_output(self):
        s_end = self._hires(freq='30min')
        for s in (s_end, self._to_middle(s_end)):
            out = resample_series_to_freq(s, '30min', output_timestamp_shows='middle')
            self.assertEqual(out.index.name, 'TIMESTAMP_MIDDLE')
            self.assertEqual(out.index[0], pd.Timestamp('2022-06-01 00:10:00') - pd.Timedelta('15min'))

    def test_middle_input_coarser_matches_end_input(self):
        s_end = self._hires()
        ref = resample_series_to_freq(s_end, '30min', agg='mean')
        out = resample_series_to_freq(self._to_middle(s_end), '30min', agg='mean')
        self.assertEqual(out.index.name, 'TIMESTAMP_END')
        self.assertEqual(out.index[0], pd.Timestamp('2022-06-01 00:30:00'))
        self.assertAlmostEqual(out.iloc[0], 1.0)
        self.assertTrue(out.equals(ref))

    def test_rejected_edge_intervals_kept_as_nan(self):
        # Rejected intervals at the start/end stay in the index as NaN, like the
        # ones in between, so a re-upload deletes old values across the full range.
        s = self._hires()
        s.iloc[:6] = np.nan
        s.iloc[-6:] = np.nan
        out = resample_series_to_freq(s, '30min', agg='mean', mincounts_perc=0.9)
        full = resample_series_to_freq(self._hires(), '30min', agg='mean', mincounts_perc=0.9)
        self.assertTrue(out.index.equals(full.index))
        self.assertTrue(out.iloc[:2].isna().all())
        self.assertTrue(out.iloc[-2:].isna().all())
        self.assertAlmostEqual(out.iloc[2], full.iloc[2])


class TestResampling(unittest.TestCase):

    def test_resample_to_monthly_agg_matrix(self):
        df = load_exampledata_parquet_long()
        series = df['Tair_f'].copy()
        monthly_means = dv.times.resample_to_monthly_agg_matrix(series=series, agg='mean', ranks=False)
        monthly_means_ranks = dv.times.resample_to_monthly_agg_matrix(series=series, agg='mean', ranks=True)
        self.assertEqual(monthly_means.shape, (10, 12))
        self.assertEqual(monthly_means.loc[2013, 3], -1.8029825268817206)
        self.assertEqual(monthly_means.loc[2019, 6], 14.38481597222222)
        self.assertEqual(monthly_means_ranks.loc[2013, 3], 10)
        self.assertEqual(monthly_means_ranks.loc[2019, 6], 1)
        self.assertEqual(monthly_means.sum().sum(), 560.8347446140001)
        self.assertEqual(monthly_means_ranks.sum().sum(), 660)

        # Test transformation to long-form time series
        longform_means = transform_yearmonth_matrix_to_longform(matrixdf=monthly_means, z_var_name='TA')
        longform_means_ranks = transform_yearmonth_matrix_to_longform(matrixdf=monthly_means_ranks,
                                                                      z_var_name='TA_RANK')
        self.assertEqual(longform_means.loc['2013-03-01'], monthly_means.loc[2013, 3])
        self.assertEqual(longform_means.loc['2019-06-01'], monthly_means.loc[2019, 6])
        self.assertEqual(longform_means_ranks.loc['2013-03-01'], monthly_means_ranks.loc[2013, 3])
        self.assertEqual(longform_means_ranks.loc['2019-06-01'], monthly_means_ranks.loc[2019, 6])
        self.assertAlmostEqual(longform_means.sum(), monthly_means.sum().sum(), places=12)
        self.assertEqual(longform_means_ranks.sum(), monthly_means_ranks.sum().sum())

    def test_diel_cycle(self):
        df = ed.load_exampledata_parquet()
        s = df['Tair_f'].copy()
        s = s.loc[s.index.year == 2018].copy()
        aggs = diel_cycle(series=s, mincounts=1, mean=True, std=True, median=True, each_month=True)
        months = set(aggs.index.get_level_values(0).tolist())
        self.assertEqual(months, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
        self.assertEqual(aggs.loc[1].sum().sum(), 1235.2002345850228)
        self.assertEqual(aggs.loc[6].sum().sum(), 4928.0285111555195)
        self.assertEqual(aggs.loc[12].sum().sum(), 1043.884056104728)
