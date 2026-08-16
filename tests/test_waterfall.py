"""
TESTS: WATERFALL PLOT
=====================

Tests for the period aggregation of `WaterfallPlot`: a period holding no
measurements at all must be absent for every aggregation, and a gap-free
series must be aggregated exactly as before.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd

from diive.core.plotting.waterfall import WaterfallPlot

# Days 11-13 of the synthetic month hold no measurements at all.
OUTAGE_DAYS = [pd.Timestamp('2020-01-11'), pd.Timestamp('2020-01-12'), pd.Timestamp('2020-01-13')]


def _series(with_outage: bool) -> pd.Series:
    """Return a reproducible 30-min series over 30 days, optionally with a 3-day outage."""
    rng = np.random.default_rng(42)
    index = pd.date_range('2020-01-01', periods=30 * 48, freq='30min')
    series = pd.Series(rng.normal(1.0, 0.2, len(index)), index=index, name='FLUX')
    if with_outage:
        series.loc['2020-01-11':'2020-01-13'] = np.nan
    return series


class TestWaterfallEmptyPeriods(unittest.TestCase):

    def test_empty_period_absent_for_sum(self):
        """A day with no measurements must not become a zero-height bar under agg='sum'."""
        wf = WaterfallPlot(_series(with_outage=True), resample='D', agg='sum')
        for day in OUTAGE_DAYS:
            self.assertNotIn(day, wf.contributions.index)
        self.assertEqual(len(wf.contributions), 27)

    def test_empty_period_absent_for_mean(self):
        """The same day must be absent under agg='mean' (the reference behaviour)."""
        wf = WaterfallPlot(_series(with_outage=True), resample='D', agg='mean')
        for day in OUTAGE_DAYS:
            self.assertNotIn(day, wf.contributions.index)
        self.assertEqual(len(wf.contributions), 27)

    def test_sum_and_mean_agree_on_which_periods_exist(self):
        """sum and mean must draw bars for exactly the same periods."""
        series = _series(with_outage=True)
        by_sum = WaterfallPlot(series, resample='D', agg='sum').contributions
        by_mean = WaterfallPlot(series, resample='D', agg='mean').contributions
        pd.testing.assert_index_equal(by_sum.index, by_mean.index)

    def test_empty_period_absent_for_count_and_prod(self):
        """'count' (-> 0) and 'prod' (-> 1.0) also return a value for an empty group."""
        series = _series(with_outage=True)
        for agg in ('count', 'prod'):
            with self.subTest(agg=agg):
                wf = WaterfallPlot(series, resample='D', agg=agg)
                for day in OUTAGE_DAYS:
                    self.assertNotIn(day, wf.contributions.index)

    def test_partly_covered_period_is_kept(self):
        """A period with at least one measurement stays, even if mostly missing."""
        series = _series(with_outage=False)
        series.loc['2020-01-20 00:00':'2020-01-20 22:30'] = np.nan
        wf = WaterfallPlot(series, resample='D', agg='sum')
        self.assertIn(pd.Timestamp('2020-01-20'), wf.contributions.index)
        self.assertEqual(len(wf.contributions), 30)

    def test_running_total_unchanged_by_dropping_empty_periods(self):
        """Dropping empty periods must not move the cumulative sum (adding 0 == skipping)."""
        wf = WaterfallPlot(_series(with_outage=True), resample='D', agg='sum')
        expected = _series(with_outage=True).dropna().sum()
        self.assertAlmostEqual(wf.cumulative.iloc[-1], expected, places=9)


class TestWaterfallGapFree(unittest.TestCase):

    def test_gapfree_series_unchanged(self):
        """A gap-free series is aggregated exactly as a plain resample().agg()."""
        series = _series(with_outage=False)
        wf = WaterfallPlot(series, resample='D', agg='sum')
        expected = series.resample('D').agg('sum')
        self.assertEqual(len(wf.contributions), 30)
        np.testing.assert_array_equal(wf.contributions.values, expected.values)
        np.testing.assert_array_equal(wf.cumulative.values, expected.cumsum().values)
        np.testing.assert_array_equal(wf.bar_bottoms.values,
                                      expected.cumsum().shift(1).fillna(0.0).values)

    def test_resample_none_passes_series_through(self):
        """With resample=None the series is used as-is."""
        series = _series(with_outage=False)
        wf = WaterfallPlot(series, resample=None)
        self.assertEqual(len(wf.contributions), len(series))

    def test_plot_renders_with_gaps(self):
        """The gappy series still renders end to end."""
        wf = WaterfallPlot(_series(with_outage=True), resample='D', agg='sum')
        ax = wf.plot(showplot=False)
        self.assertIsNotNone(ax)
        self.assertEqual(len(ax.patches), 27)


if __name__ == '__main__':
    unittest.main()
