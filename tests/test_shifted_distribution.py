"""
TESTS: SHIFTED DISTRIBUTION PLOT
================================

Tests for degenerate reference/comparison periods in `ShiftedDistributionPlot`.
A period with no records at all must be labelled rather than raise, a period
with no spread (constant, or a single record) must still be drawn as the spike
it is, and an ordinary two-period comparison must be untouched.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from diive.core.plotting.shifted_distribution import ShiftedDistributionPlot

REF_PERIOD = ('1990-01-01', '2000-12-31')
COMP_PERIOD = ('2010-01-01', '2020-12-31')
OUTSIDE_PERIOD = ('2050-01-01', '2050-12-31')
CONSTANT_VALUE = 5.0

INDEX = pd.date_range('1990-01-01', '2020-12-31', freq='D')


def _series(period: tuple = None, fill=None) -> pd.Series:
    """Return a reproducible daily series, optionally overwriting one period with `fill`."""
    values = np.random.default_rng(42).normal(10.0, 3.0, len(INDEX))
    series = pd.Series(values, index=INDEX, name='TA')
    if period is not None:
        series.loc[period[0]:period[1]] = fill
    return series


def _plotted(sdp: ShiftedDistributionPlot):
    """Render on a throwaway axes and return it."""
    fig, ax = plt.subplots()
    sdp.plot(ax=ax)
    plt.close(fig)
    return ax


def _spike_stats(x: np.ndarray, density: np.ndarray) -> tuple:
    """Return (peak position, integral, full width at half maximum) of a density curve."""
    above_half = density >= density.max() / 2
    return x[np.argmax(density)], np.trapezoid(density, x), x[above_half][-1] - x[above_half][0]


class TestEmptyPeriods(unittest.TestCase):
    """A period holding no records has nothing to draw and must say so."""

    def test_empty_comparison_period_is_labelled_not_raised(self):
        """A comparison period outside the record keeps the plot, labelled 'no data'."""
        sdp = ShiftedDistributionPlot(_series(), REF_PERIOD, OUTSIDE_PERIOD)
        self.assertIsNone(sdp._comp_kde)
        self.assertIsNotNone(sdp._ref_kde)
        ax = _plotted(sdp)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels[1], f"Comparison ({OUTSIDE_PERIOD[0]} - {OUTSIDE_PERIOD[1]}): no data")
        self.assertNotIn('no data', labels[0])
        # The reference is intact, so its outline, its hatched fill and all four
        # breakpoint lines and five zone labels are still drawn.
        self.assertEqual(len(ax.lines), 5)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.texts), 5)

    def test_all_nan_reference_period_drops_the_zones(self):
        """An all-NaN reference gives no breakpoints, so no zone lines and no zone labels."""
        sdp = ShiftedDistributionPlot(_series(REF_PERIOD, np.nan), REF_PERIOD, COMP_PERIOD)
        self.assertIsNone(sdp._ref_kde)
        self.assertTrue(np.isnan(sdp.breakpoints).all())
        ax = _plotted(sdp)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels[0], f"Reference ({REF_PERIOD[0]} - {REF_PERIOD[1]}): no data")
        self.assertEqual(len(ax.texts), 0)  # No zones, hence no zone labels
        self.assertEqual(len(ax.lines), 1)  # Comparison outline only, no breakpoint lines
        self.assertEqual(len(ax.collections), 1)  # Comparison still filled, unzoned

    def test_both_periods_empty_states_so_on_the_axes(self):
        """With nothing in either period the axes carry a message and no artists."""
        sdp = ShiftedDistributionPlot(_series(), OUTSIDE_PERIOD, ('2060-01-01', '2060-12-31'))
        self.assertIsNone(sdp._x)
        ax = _plotted(sdp)
        self.assertEqual([t.get_text() for t in ax.texts],
                         ["No data in reference or comparison period"])
        self.assertEqual(len(ax.lines), 0)
        self.assertEqual(len(ax.collections), 0)


class TestZeroSpreadPeriods(unittest.TestCase):
    """A constant period, or one holding a single record, is a spike — draw it as one."""

    def test_constant_reference_period_draws_a_spike_at_its_value(self):
        sdp = ShiftedDistributionPlot(_series(REF_PERIOD, CONSTANT_VALUE), REF_PERIOD, COMP_PERIOD)
        self.assertIsNotNone(sdp._ref_kde)
        peak, integral, fwhm = _spike_stats(sdp._x, sdp._ref_kde)
        grid_step = sdp._x[1] - sdp._x[0]
        self.assertLess(abs(peak - CONSTANT_VALUE), grid_step)
        self.assertAlmostEqual(integral, 1.0, places=6)  # A proper density, not a stub
        self.assertLess(fwhm, 0.02 * (sdp._x[-1] - sdp._x[0]))  # Narrow: reads as a spike
        # Zero spread collapses all four breakpoints onto the constant value.
        np.testing.assert_allclose(sdp.breakpoints, [CONSTANT_VALUE] * 4)
        self.assertEqual(len(_plotted(sdp).texts), 5)

    def test_constant_comparison_period_draws_a_spike_at_its_value(self):
        sdp = ShiftedDistributionPlot(_series(COMP_PERIOD, CONSTANT_VALUE), REF_PERIOD, COMP_PERIOD)
        peak, integral, fwhm = _spike_stats(sdp._x, sdp._comp_kde)
        self.assertLess(abs(peak - CONSTANT_VALUE), sdp._x[1] - sdp._x[0])
        self.assertAlmostEqual(integral, 1.0, places=6)
        self.assertLess(fwhm, 0.02 * (sdp._x[-1] - sdp._x[0]))

    def test_single_record_comparison_period_draws_a_spike_at_that_record(self):
        series = _series()
        one_day = ('2010-06-01', '2010-06-01')
        sdp = ShiftedDistributionPlot(series, REF_PERIOD, one_day)
        self.assertEqual(len(sdp._comp_data), 1)
        peak, integral, _ = _spike_stats(sdp._x, sdp._comp_kde)
        self.assertLess(abs(peak - series.loc[one_day[0]]), sdp._x[1] - sdp._x[0])
        self.assertAlmostEqual(integral, 1.0, places=6)

    def test_single_record_reference_period_keeps_the_plot(self):
        sdp = ShiftedDistributionPlot(_series(), ('1990-06-01', '1990-06-01'), COMP_PERIOD)
        self.assertIsNotNone(sdp._ref_kde)
        ax = _plotted(sdp)
        self.assertEqual(len(ax.texts), 5)
        self.assertNotIn('no data', ' '.join(t.get_text() for t in ax.get_legend().get_texts()))


class TestOrdinaryComparisonUnchanged(unittest.TestCase):
    """Two well-populated periods must be computed and drawn exactly as before."""

    def setUp(self):
        self.series = _series()
        self.sdp = ShiftedDistributionPlot(self.series, REF_PERIOD, COMP_PERIOD)

    def test_grid_and_kdes_match_the_plain_silverman_computation(self):
        """No guard may perturb the grid or either KDE when neither period is degenerate."""
        ref = self.series.loc[REF_PERIOD[0]:REF_PERIOD[1]].dropna().values
        comp = self.series.loc[COMP_PERIOD[0]:COMP_PERIOD[1]].dropna().values
        all_vals = np.concatenate([ref, comp])
        margin = ref.std()
        x = np.linspace(all_vals.min() - margin, all_vals.max() + margin, 1000)
        np.testing.assert_array_equal(self.sdp._x, x)
        for name, data, got in (('ref', ref, self.sdp._ref_kde), ('comp', comp, self.sdp._comp_kde)):
            with self.subTest(period=name):
                bw = 1.06 * data.std() * len(data) ** (-0.2)
                kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data.reshape(-1, 1))
                np.testing.assert_array_equal(got, np.exp(kde.score_samples(x.reshape(-1, 1))))

    def test_breakpoints_are_the_reference_mean_plus_minus_one_and_three_sd(self):
        ref = self.series.loc[REF_PERIOD[0]:REF_PERIOD[1]].dropna().values
        expected = [ref.mean() + k * ref.std() for k in (-3, -1, 1, 3)]
        np.testing.assert_array_equal(self.sdp.breakpoints, expected)

    def test_all_artists_and_labels_are_drawn(self):
        ax = _plotted(self.sdp)
        self.assertEqual(len(ax.lines), 6)  # 2 KDE outlines + 4 breakpoint lines
        self.assertEqual(len(ax.collections), 6)  # 1 hatched reference + 5 zone fills
        self.assertEqual([t.get_text() for t in ax.texts],
                         ['Extremely cold', 'Cold', 'Normal', 'Hot', 'Extremely hot'])
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                         [f"Reference ({REF_PERIOD[0]} - {REF_PERIOD[1]})",
                          f"Comparison ({COMP_PERIOD[0]} - {COMP_PERIOD[1]})"])
        self.assertEqual(ax.get_title(loc='left'), 'Shifted distribution: TA')


if __name__ == '__main__':
    unittest.main()
