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


if __name__ == '__main__':
    unittest.main()
