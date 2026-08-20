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
from matplotlib.colors import to_hex

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


class TestWaterfallZeroContribution(unittest.TestCase):
    """A contribution of exactly 0.0 is neither uptake nor release.

    It takes the release colour under both sign conventions, which is documented
    rather than given a third colour because a zero-height bar paints no pixels.
    """

    UPTAKE = '#2196f3'
    RELEASE = '#f44336'

    @staticmethod
    def _facecolors(series, **kwargs):
        fig, ax = plt.subplots()
        WaterfallPlot(series, resample=None, **kwargs).plot(ax=ax, showplot=False)
        colors = [to_hex(p.get_facecolor()) for p in ax.patches]
        plt.close(fig)
        return colors

    def test_zero_contribution_takes_release_color(self):
        """0.0 falls in the release bucket with the default NEE sign convention."""
        series = pd.Series([1.0, 0.0, -1.0],
                           index=pd.date_range('2020-01-01', periods=3, freq='D'), name='FLUX')
        self.assertEqual(self._facecolors(series),
                         [self.RELEASE, self.RELEASE, self.UPTAKE])

    def test_zero_contribution_takes_release_color_when_uptake_is_positive(self):
        """The same holds sign-flipped: 0.0 is release, not uptake."""
        series = pd.Series([1.0, 0.0, -1.0],
                           index=pd.date_range('2020-01-01', periods=3, freq='D'), name='FLUX')
        self.assertEqual(self._facecolors(series, uptake_is_negative=False),
                         [self.UPTAKE, self.RELEASE, self.RELEASE])

    def test_signed_bars_keep_their_own_colors(self):
        """Small-but-real contributions must stay signed, not be lumped in with 0.0."""
        series = pd.Series([1.0, 0.25, 0.0, -0.25, -1.0],
                           index=pd.date_range('2020-01-01', periods=5, freq='D'), name='FLUX')
        self.assertEqual(self._facecolors(series),
                         [self.RELEASE, self.RELEASE, self.RELEASE, self.UPTAKE, self.UPTAKE])

    def test_zero_bar_paints_no_pixels(self):
        """The zero bar's colour is unobservable, which is why it gets no colour of its own.

        The same repaint on a non-zero bar is measured first, so a count of 0 means
        the colour is invisible rather than that the measurement is inert.
        """
        series = pd.Series([1.0, 0.0, -1.0],
                           index=pd.date_range('2020-01-01', periods=3, freq='D'), name='FLUX')

        def pixels_moved_by_repainting(bar_index):
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
            WaterfallPlot(series, resample=None).plot(ax=ax, showplot=False)
            fig.canvas.draw()
            before = np.asarray(fig.canvas.buffer_rgba()).copy()
            ax.patches[bar_index].set_facecolor('#00FF00')
            fig.canvas.draw()
            after = np.asarray(fig.canvas.buffer_rgba()).copy()
            plt.close(fig)
            return int((before != after).any(axis=2).sum())

        self.assertGreater(pixels_moved_by_repainting(0), 0)  # control: a real bar does move pixels
        self.assertEqual(pixels_moved_by_repainting(1), 0)

    def test_genuinely_zero_period_is_drawn_not_dropped(self):
        """A dry day is measured, so it keeps its (invisible) bar — unlike an empty day."""
        index = pd.date_range('2020-01-01', periods=5 * 48, freq='30min')
        series = pd.Series(0.0, index=index, name='PREC')
        series.loc['2020-01-03'] = 0.5  # one wet day among four dry ones
        wf = WaterfallPlot(series, resample='D', agg='sum')
        self.assertEqual(len(wf.contributions), 5)
        self.assertEqual(int((wf.contributions == 0.0).sum()), 4)

        fig, ax = plt.subplots()
        wf.plot(ax=ax, showplot=False)
        colors = [to_hex(p.get_facecolor()) for p in ax.patches]
        plt.close(fig)
        self.assertEqual(colors, [self.RELEASE] * 5)


if __name__ == '__main__':
    unittest.main()
