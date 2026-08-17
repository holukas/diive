"""
TESTS: HISTOGRAM PLOT
=====================

Tests for the KDE overlay scaling of `HistogramPlot`, for uniform and for
explicit non-uniform bin edges.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest
import warnings

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

    def test_empty_series_says_no_data_instead_of_raising(self):
        """A series with no records at all must not raise in `__init__`.

        `first_date = series.index[0]` used to raise `IndexError` before `plot`
        (and its `dropna().empty` guard) was ever reached.
        """
        hist, fig, ax = self._plot([])
        try:
            texts = [t.get_text() for t in ax.texts]
            self.assertEqual(len(texts), 1, msg=f"expected one message, got {texts}")
            self.assertIn('no data', texts[0])
            self.assertEqual(len(ax.patches), 0, msg="nothing to draw, but bars were drawn")
            self.assertIsNone(hist.counts)
        finally:
            plt.close(fig)

    def test_empty_series_title_states_no_records_instead_of_a_date_range(self):
        """`first_date`/`last_date` only feed the title, so report the absence."""
        hist, fig, ax = self._plot([])
        try:
            self.assertIsNone(hist.first_date)
            self.assertIsNone(hist.last_date)
            self.assertEqual(ax.get_title(), 'X (no records)')
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
            self.assertEqual(ax.get_title(),
                             f"X (between {hist.first_date} and {hist.last_date})",
                             msg="the default title must still state the covered period")
            twins = [a for a in fig.axes if a is not ax]
            self.assertEqual(len(twins), 1, msg="z-score overlay is missing")
            self.assertGreater(len(twins[0].lines), 0, msg="no z-score marker lines")
            self.assertEqual(twins[0].get_xlabel(), 'z-score')
        finally:
            plt.close(fig)


class TestHistogramFromOutlierDetector(unittest.TestCase):
    """The detectors' own diagnostic plot is how an empty series is reached.

    `core/base/flagbase.py::defaultplot` histograms the raw series *and* the
    retained (`flag == 0`) subset. A detector that rejects every record hands
    the second `HistogramPlot` an empty series, so a raise there kills the
    detector run over its own diagnostic.
    """

    def test_detector_rejecting_every_record_still_draws_its_diagnostic(self):
        import diive as dv

        series = pd.Series(np.full(200, 5.0),
                           index=pd.date_range('2020-01-01', periods=200, freq='30min'),
                           name='X')
        with warnings.catch_warnings():
            # Agg cannot show a figure; the diagnostic still gets drawn.
            warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')
            detector = dv.outliers.AbsoluteLimits(series, minval=100, maxval=200,
                                                  showplot=True).run()
        fig = detector.fig
        try:
            self.assertEqual(int((detector.overall_flag == 2).sum()), 200,
                             msg="fixture is broken: not every record was rejected")
            self.assertTrue(detector.filteredseries.dropna().empty)
            # The retained-subset panel must say so rather than being missing.
            messages = [t.get_text() for ax in fig.axes for t in ax.texts if 'no data' in t.get_text()]
            self.assertEqual(len(messages), 1,
                             msg=f"expected exactly one empty-panel message, got {messages}")
        finally:
            plt.close(fig)


def _spiked_series(n: int = 500) -> pd.Series:
    """Normal series with one extreme value at each end, so the fringe bins are sparse."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, n)
    values[10] = 14.0
    values[20] = -9.0
    return pd.Series(values,
                     index=pd.date_range('2020-01-01', periods=n, freq='30min'),
                     name='X')


def _plot(series: pd.Series, **ctor_kwargs):
    """Plot on a fresh axes and return (hist, bar geometry, info texts)."""
    fig, ax = plt.subplots()
    hist = HistogramPlot(series=series, method='n_bins', **ctor_kwargs)
    hist.plot(ax=ax, show_zscores=False)
    bars = [(p.get_x(), p.get_width(), p.get_height()) for p in ax.patches]
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    return hist, bars, texts


class TestHistogramIgnoreFringeBins(unittest.TestCase):
    """`ignore_fringe_bins` must drop the first i and last j bins from the plot.

    Semantics are `diive.analysis.Histogram`'s: the edges come from the full
    series first, so the bins that survive are the ones the untrimmed histogram
    would have shown -- not a re-binning of the trimmed data.
    """

    def test_fringe_bins_are_dropped_from_counts_edges_and_bars(self):
        series = _spiked_series()
        full, full_bars, _ = _plot(series, n_bins=10)
        trimmed, trimmed_bars, _ = _plot(series, n_bins=10, ignore_fringe_bins=[1, 1])

        np.testing.assert_array_equal(trimmed.counts, np.asarray(full.counts)[1:-1])
        np.testing.assert_allclose(trimmed.edges, np.asarray(full.edges)[1:-1])
        self.assertEqual(len(trimmed_bars), len(full_bars) - 2,
                         msg=f"{len(trimmed_bars)} bars drawn, expected {len(full_bars) - 2}")
        # The surviving bars must sit exactly where they sat before the trim.
        for kept, orig in zip(trimmed_bars, full_bars[1:-1], strict=True):
            self.assertAlmostEqual(kept[0], orig[0])
            self.assertAlmostEqual(kept[1], orig[1])
            self.assertAlmostEqual(kept[2], orig[2])

    def test_fringe_trim_matches_the_analysis_class(self):
        """The plotting class must not invent its own trim semantics."""
        from diive.analysis.histogram import Histogram

        series = _spiked_series()
        for trim in ([1, 1], [2, 0], [0, 3], [3, 4]):
            with self.subTest(trim=trim):
                hist, _, _ = _plot(series, n_bins=10, ignore_fringe_bins=trim)
                ref = Histogram(series=series, method='n_bins', n_bins=10,
                                ignore_fringe_bins=trim)
                np.testing.assert_array_equal(np.asarray(hist.counts, dtype=int),
                                              ref.results['COUNTS'].to_numpy())
                # `Histogram` reports left edges only, so compare against edges[:-1].
                np.testing.assert_allclose(np.asarray(hist.edges)[:-1],
                                           ref.results['BIN_START_INCL'].to_numpy())

    def test_zero_at_one_end_trims_nothing_at_that_end(self):
        """`[2, 0]` must keep the last bin: a naive `edges[2:-0]` empties the array."""
        series = _spiked_series()
        full, _, _ = _plot(series, n_bins=10)
        trimmed, _, _ = _plot(series, n_bins=10, ignore_fringe_bins=[2, 0])

        self.assertEqual(len(trimmed.counts), len(full.counts) - 2)
        self.assertAlmostEqual(float(trimmed.edges[-1]), float(full.edges[-1]),
                               msg="the closing edge was dropped although n_last is 0")
        np.testing.assert_array_equal(trimmed.counts, np.asarray(full.counts)[2:])

    def test_info_box_reports_the_trim(self):
        """The box prints `n_bins`, which is no longer the number of bars drawn."""
        series = _spiked_series()
        _, _, texts = _plot(series, n_bins=10, ignore_fringe_bins=[1, 1])
        self.assertTrue(any('ignore_fringe_bins: [1, 1]' in t for t in texts),
                        msg=f"the trim is not stated in the info box: {texts}")

    def test_no_trim_requested_leaves_the_histogram_untouched(self):
        """Nothing to trim keeps all bins, bar for bar.

        `[0, 0]` is included because it is truthy and so takes the trimming path;
        only its info box differs, which is checked separately.
        """
        series = _spiked_series()
        full, full_bars, full_texts = _plot(series, n_bins=10)
        for value in (False, None, [], [0, 0]):
            with self.subTest(ignore_fringe_bins=value):
                hist, bars, texts = _plot(series, n_bins=10, ignore_fringe_bins=value)
                np.testing.assert_array_equal(hist.counts, full.counts)
                np.testing.assert_array_equal(hist.edges, full.edges)
                self.assertEqual(bars, full_bars)
                if value:
                    continue
                self.assertEqual(texts, full_texts)

    def test_trimming_every_bin_raises_instead_of_drawing_nothing(self):
        series = _spiked_series()
        with self.assertRaises(ValueError) as cm:
            _plot(series, n_bins=10, ignore_fringe_bins=[5, 5])
        self.assertIn('ignore_fringe_bins=[5, 5]', str(cm.exception))


class TestHistogramPinnedBinGrid(unittest.TestCase):
    """A sequence passed as `n_bins` is the explicit edge list.

    That is how two histograms of related subsets are put on one grid, so that
    bin *i* is the same interval in both. Without it each subset bins over its
    own range (`core/base/flagbase.py` draws such a pair on purpose).
    """

    def test_same_edge_list_gives_two_subsets_the_same_grid(self):
        series = _spiked_series()
        subset = series[series.abs() < 3]
        edges = np.linspace(-10.0, 15.0, 11)

        full, full_bars, _ = _plot(series, n_bins=edges)
        part, part_bars, _ = _plot(subset, n_bins=edges)

        np.testing.assert_allclose(full.edges, edges)
        np.testing.assert_allclose(part.edges, edges)
        for i, (a, b) in enumerate(zip(full_bars, part_bars, strict=True)):
            self.assertAlmostEqual(a[0], b[0], msg=f"bar {i} starts at a different value")
            self.assertAlmostEqual(a[1], b[1], msg=f"bar {i} has a different width")

    def test_without_pinned_edges_the_two_grids_differ(self):
        """Control: the grids really are subset-derived when no edges are given."""
        series = _spiked_series()
        subset = series[series.abs() < 3]
        full, _, _ = _plot(series, n_bins=10)
        part, _, _ = _plot(subset, n_bins=10)
        self.assertFalse(np.allclose(full.edges, part.edges),
                         msg="fixture is broken: the two subsets already share a grid")


def _info_text(series: pd.Series, method, **kwargs) -> str:
    """Plot with an arbitrary `method` and return the single info-box string."""
    fig, ax = plt.subplots()
    hist = HistogramPlot(series=series, method=method, n_bins=10, **kwargs)
    hist.plot(ax=ax, show_zscores=False, show_counts=False)
    boxes = [t.get_text() for t in ax.texts if t.get_text().startswith('method:')]
    plt.close(fig)
    assert len(boxes) == 1, boxes
    return boxes[0], hist


class TestHistogramInfoBox(unittest.TestCase):
    """Each fact must appear in the box exactly once.

    `info_txt += f"..." if cond else info_txt` appended the box to itself on the
    false branch, so any `method` other than 'n_bins' doubled the string twice.
    """

    def test_n_bins_box_is_the_expected_string(self):
        series = _series()
        text, hist = _info_text(series, 'n_bins')
        ix_max = int(np.asarray(hist.counts).argmax())
        expected = (f"method: n_bins\nn_bins: 10\n"
                    f"PEAK between {hist.edges[ix_max]:.02f} and {hist.edges[ix_max + 1]:.02f}")
        self.assertEqual(text, expected)

    def test_other_method_states_the_method_once(self):
        series = _series()
        for method in ('uniform', 'uniques', None):
            with self.subTest(method=method):
                text, _ = _info_text(series, method)
                self.assertEqual(text, f"method: {method}")

    def test_other_method_with_trim_states_both_facts_once(self):
        series = _spiked_series()
        text, _ = _info_text(series, 'uniform', ignore_fringe_bins=[1, 1])
        self.assertEqual(text, "method: uniform\nignore_fringe_bins: [1, 1]")

    def test_other_method_without_peak_highlight(self):
        """Only one of the two conditional lines runs, so this doubles once, not twice."""
        series = _series()
        fig, ax = plt.subplots()
        HistogramPlot(series=series, method='uniform', n_bins=10).plot(
            ax=ax, show_zscores=False, show_counts=False, highlight_peak=False)
        boxes = [t.get_text() for t in ax.texts if t.get_text().startswith('method:')]
        plt.close(fig)
        self.assertEqual(boxes, ["method: uniform"])


class TestNonNumericError(unittest.TestCase):
    """`non_numeric_error` must draw on the axes it is handed.

    It positions with `transform=ax.transAxes`, so drawing via `plt.text` put the
    message in that axes' coordinates on pyplot's current figure instead. Currently
    the helper has no callers in the repo, so this guards a latent defect.
    """

    def test_text_lands_on_the_passed_axes_not_on_pyplots_figure(self):
        from matplotlib.figure import Figure

        import diive.core.plotting.plotfuncs as pf

        # A bare Figure() is not registered with pyplot, so it can never be gcf().
        bare = Figure()
        bare_ax = bare.add_subplot(1, 1, 1)
        pyplot_fig, pyplot_ax = plt.subplots()  # this is what plt.text would target
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pf.non_numeric_error(bare_ax)
            self.assertEqual(caught, [], msg=f"unexpected warnings: {[str(w.message) for w in caught]}")
            self.assertEqual([t.get_text() for t in bare_ax.texts],
                             ['Sorry, no plot. Data are non-numeric.'])
            self.assertEqual([t.get_text() for t in pyplot_ax.texts], [],
                             msg="the message landed on pyplot's figure, not the given axes")
            self.assertIs(bare_ax.texts[0].axes, bare_ax)
            self.assertIs(bare_ax.texts[0].get_figure(root=True), bare)
            self.assertIsNot(plt.gcf(), bare, msg="fixture is broken: the bare figure became gcf()")
        finally:
            plt.close(pyplot_fig)


if __name__ == '__main__':
    unittest.main()
