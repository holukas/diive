"""
TESTS: WATERFALL PLOT
=====================

Tests for the period aggregation of `WaterfallPlot`: a period holding no
measurements at all must be absent for every aggregation, and a gap-free
series must be aggregated exactly as before. Also for its rendering of an
input without a single valid value, which must say so instead of raising.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
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


class TestWaterfallNoData(unittest.TestCase):
    """An input without a single valid value must say so instead of raising."""

    @staticmethod
    def _plot(series, **kwargs):
        fig, ax = plt.subplots()
        WaterfallPlot(series, **kwargs).plot(ax=ax, showplot=False)
        return ax

    def test_all_nan_series_says_no_data(self):
        """An all-NaN column draws no bars and states there is no data."""
        series = pd.Series(np.nan, index=pd.date_range('2020-01-01', periods=48, freq='30min'),
                           name='FLUX')
        ax = self._plot(series, resample='D', agg='sum')
        self.assertEqual(len(ax.patches), 0)
        self.assertEqual([t.get_text() for t in ax.texts], ['FLUX: no data'])

    def test_all_nan_series_says_no_data_without_resampling(self):
        """The same holds with resample=None, where the series is used as-is."""
        series = pd.Series(np.nan, index=pd.date_range('2020-01-01', periods=48, freq='30min'),
                           name='FLUX')
        ax = self._plot(series, resample=None)
        self.assertEqual(len(ax.patches), 0)
        self.assertEqual([t.get_text() for t in ax.texts], ['FLUX: no data'])

    def test_empty_series_says_no_data(self):
        """An empty series has no periods either."""
        series = pd.Series(dtype=float, index=pd.DatetimeIndex([]), name='FLUX')
        ax = self._plot(series, resample='D', agg='sum')
        self.assertEqual(len(ax.patches), 0)
        self.assertEqual([t.get_text() for t in ax.texts], ['FLUX: no data'])

    def test_unnamed_all_nan_series_says_no_data(self):
        """Without a series name the message must not read 'None: no data'."""
        series = pd.Series(np.nan, index=pd.date_range('2020-01-01', periods=48, freq='30min'))
        ax = self._plot(series, resample='D', agg='sum')
        self.assertEqual([t.get_text() for t in ax.texts], ['No data'])

    def test_no_data_axes_draws_nothing_else(self):
        """The message replaces the plot: no bars, no connectors, no end marker, no ticks."""
        series = pd.Series(np.nan, index=pd.date_range('2020-01-01', periods=48, freq='30min'),
                           name='FLUX')
        ax = self._plot(series, resample='D', agg='sum')
        self.assertEqual(len(ax.lines), 0)
        self.assertFalse(ax.xaxis.get_tick_params()['labelbottom'])

    def test_single_valid_record_still_plots(self):
        """One valid record is a real (if minimal) waterfall, not a no-data case."""
        series = pd.Series(np.nan, index=pd.date_range('2020-01-01', periods=48, freq='30min'),
                           name='FLUX')
        series.iloc[10] = 2.5
        ax = self._plot(series, resample='D', agg='sum')
        self.assertEqual(len(ax.patches), 1)
        self.assertAlmostEqual(ax.patches[0].get_height(), 2.5, places=9)
        self.assertIn('2', [t.get_text() for t in ax.texts])


class TestWaterfallOrdinarySeriesUnaffected(unittest.TestCase):
    """The no-data guard must not touch a series that has data."""

    def test_ordinary_series_renders_unchanged(self):
        """Bar heights, bottoms, connectors, end marker and annotation are as computed."""
        series = _series(with_outage=True)
        expected = series.dropna().resample('D').agg('sum')
        expected = expected[series.dropna().resample('D').count() > 0]

        fig, ax = plt.subplots()
        WaterfallPlot(series, resample='D', agg='sum').plot(ax=ax, showplot=False)

        self.assertEqual(len(ax.patches), 27)
        np.testing.assert_allclose([p.get_height() for p in ax.patches], expected.values)
        np.testing.assert_allclose([p.get_y() for p in ax.patches],
                                   expected.cumsum().shift(1).fillna(0.0).values)
        # 26 connectors between the 27 bars, plus the end-of-series marker.
        self.assertEqual(len(ax.lines), 27)
        np.testing.assert_allclose(ax.lines[-1].get_ydata(), [expected.sum()])
        self.assertIn(f"{expected.sum():.0f}", [t.get_text() for t in ax.texts])


if __name__ == '__main__':
    unittest.main()
