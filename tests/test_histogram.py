"""
TESTS: HISTOGRAM PLOT
=====================

Tests for the KDE overlay scaling of `HistogramPlot`, for uniform and for
explicit non-uniform bin edges.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diive.core.plotting.histogram import HistogramPlot

# Deliberately non-uniform: the first bin is 5 wide, the middle bins are 1 wide.
NONUNIFORM_EDGES = [0, 5, 8, 9, 10, 11, 12, 15, 20]


def _series(n: int = 500) -> pd.Series:
    """Return a reproducible normal series with a proper timestamp index."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(10, 2, n),
                     index=pd.date_range('2020-01-01', periods=n, freq='30min'),
                     name='X')


def _plot_kde(series: pd.Series, bins):
    """Plot with the KDE overlay and return (hist, kde_x, kde_y)."""
    fig, ax = plt.subplots()
    hist = HistogramPlot(series=series, method='n_bins', n_bins=bins)
    hist.plot(ax=ax, show_kde=True, show_zscores=False, show_info=False,
              show_counts=False, highlight_peak=False)
    lines = [line for line in ax.get_lines() if line.get_label() == 'KDE']
    assert len(lines) == 1
    x, y = lines[0].get_xdata(), lines[0].get_ydata()
    plt.close(fig)
    return hist, np.asarray(x, dtype=float), np.asarray(y, dtype=float)


class TestHistogramKde(unittest.TestCase):

    def test_kde_matches_counts_per_bin_with_nonuniform_bins(self):
        """The curve, averaged over each bin, must reproduce that bin's count.

        The bars are counts, so the expected height at x is
        N * density(x) * width(bin containing x); averaged over a bin that is
        N * integral(density) over the bin, i.e. the bin's expected count.
        Scaling by a single bin width instead is wrong by w_i / w_0 per bin.
        """
        series = _series()
        hist, x, y = _plot_kde(series, NONUNIFORM_EDGES)
        edges = np.asarray(hist.edges, dtype=float)

        self.assertFalse(np.allclose(np.diff(edges), np.diff(edges)[0]),
                         msg="test fixture must have non-uniform bins")

        for i in range(len(edges) - 1):
            inbin = (x >= edges[i]) & (x <= edges[i + 1])
            curve_mean = float(y[inbin].mean())
            count = float(hist.counts[i])
            if count < 10:  # KDE smoothing dominates in near-empty bins
                continue
            self.assertAlmostEqual(curve_mean / count, 1.0, delta=0.2,
                                   msg=f"bin {i} [{edges[i]}, {edges[i + 1]}]: "
                                       f"curve mean {curve_mean:.2f} vs count {count:.0f}")

    def test_kde_peak_stays_near_the_bars_with_nonuniform_bins(self):
        """The overlay must not tower over the histogram it is drawn on.

        Before the per-bin fix the curve peaked ~4.5x above the tallest bar.
        """
        series = _series()
        hist, _, y = _plot_kde(series, NONUNIFORM_EDGES)
        ratio = float(y.max()) / float(np.max(hist.counts))
        self.assertLess(ratio, 2.0, msg=f"KDE peak is {ratio:.2f}x the tallest bar")

    def test_kde_scaling_unchanged_for_uniform_bins(self):
        """Uniform bins keep the old `N * bin_width` result to fp tolerance."""
        from scipy.stats import gaussian_kde

        series = _series()
        hist, x, y = _plot_kde(series, 8)
        edges = np.asarray(hist.edges, dtype=float)
        vals = series.dropna().to_numpy()

        expected = gaussian_kde(vals)(x) * hist.counts.sum() * (edges[1] - edges[0])
        np.testing.assert_allclose(y, expected, rtol=1e-12, atol=0.0)

    def test_kde_defined_over_the_whole_edge_range(self):
        """Sample points on the closing edge stay inside the last bin."""
        series = _series()
        hist, x, y = _plot_kde(series, NONUNIFORM_EDGES)
        self.assertAlmostEqual(x[0], float(hist.edges[0]))
        self.assertAlmostEqual(x[-1], float(hist.edges[-1]))
        self.assertTrue(np.all(np.isfinite(y)))


class TestHistogramDegenerateInput(unittest.TestCase):
    """Degenerate series must produce an honest axes, not a traceback.

    Both are reached from the outlier detectors' own diagnostic plot
    (`core/base/flagbase.py` histograms the raw series *and* the retained
    subset), so a raise there kills the detector run over its own diagnostic.
    """

    @staticmethod
    def _plot(values, **kwargs):
        """Plot `values` on a fresh axes and return (hist, fig, ax)."""
        series = pd.Series(np.asarray(values, dtype=float),
                           index=pd.date_range('2020-01-01', periods=len(values), freq='30min'),
                           name='X')
        fig, ax = plt.subplots()
        hist = HistogramPlot(series=series, method='n_bins', n_bins=None)
        hist.plot(ax=ax, **kwargs)
        return hist, fig, ax

    def test_constant_series_draws_bars_but_no_zscore_overlay(self):
        """Zero standard deviation makes every z-score NaN: drop only the overlay.

        The histogram of a constant series still says something (all records in
        one bin), so it must be drawn.
        """
        hist, fig, ax = self._plot(np.full(500, 5.0))
        try:
            self.assertEqual(float(np.sum(hist.counts)), 500.0,
                             msg="all records must still be binned")
            self.assertGreater(len(ax.patches), 0, msg="no bars drawn")
            # The overlay lives on a twiny axes; without it the figure has one axes.
            self.assertEqual(len(fig.axes), 1,
                             msg=f"z-score twiny axes created for a constant series: {fig.axes}")
            self.assertFalse(hasattr(hist, 'axx'))
        finally:
            plt.close(fig)

    def test_constant_series_with_gaps_draws_no_zscore_overlay(self):
        """The detectors' retained subset is gappy, so cover NaN + constant too."""
        hist, fig, ax = self._plot([5.0, np.nan, 5.0, np.nan, 5.0])
        try:
            self.assertEqual(float(np.sum(hist.counts)), 3.0)
            self.assertEqual(len(fig.axes), 1)
        finally:
            plt.close(fig)

    def test_all_nan_series_says_no_data_instead_of_raising(self):
        """`ax.hist` autodetects `[nan, nan]` and raises; an honest label is better."""
        hist, fig, ax = self._plot(np.full(200, np.nan))
        try:
            texts = [t.get_text() for t in ax.texts]
            self.assertEqual(len(texts), 1, msg=f"expected one message, got {texts}")
            self.assertIn('no data', texts[0])
            self.assertIn('X', texts[0], msg="the message must name the variable")
            self.assertEqual(len(ax.patches), 0, msg="nothing to draw, but bars were drawn")
            self.assertIsNone(hist.counts)
        finally:
            plt.close(fig)

    def test_ordinary_series_keeps_bars_and_zscore_overlay(self):
        """Control: neither guard may fire on a series with actual spread."""
        rng = np.random.default_rng(7)
        values = rng.normal(10, 2, 400)
        hist, fig, ax = self._plot(values)
        try:
            self.assertEqual(float(np.sum(hist.counts)), 400.0)
            self.assertGreater(len(ax.patches), 0)
            texts = [t.get_text() for t in ax.texts]
            self.assertTrue(any('method:' in t for t in texts),
                            msg=f"the info box must still be drawn, got {texts}")
            self.assertFalse(any('no data' in t for t in texts),
                             msg=f"the empty-data guard fired on real data: {texts}")
            twins = [a for a in fig.axes if a is not ax]
            self.assertEqual(len(twins), 1, msg="z-score overlay is missing")
            self.assertGreater(len(twins[0].lines), 0, msg="no z-score marker lines")
            self.assertEqual(twins[0].get_xlabel(), 'z-score')
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
