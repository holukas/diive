"""
TESTS: SHIFTED DISTRIBUTION PLOT
================================

Tests for degenerate reference/comparison periods in `ShiftedDistributionPlot`.
A period with no records at all must be labelled rather than raise, a period
with no spread (constant, or a single record) must still be drawn as the spike
it is, and an ordinary two-period comparison must be untouched.

Also covers the plot's `FormatStyle` contract — the caller's chrome fields must
survive, the class overriding only the title and legend it re-draws itself — the
zone geometry when a +-3 SD breakpoint falls outside the evaluation grid, the
five-entry contract on the zone label/colour lists, what a repeated `plot()` on
one axes does, and the sample (ddof=1) standard deviation behind the zones.

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
from diive.core.plotting.styles.format import FormatStyle

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


def _styled(sdp: ShiftedDistributionPlot, **plot_kwargs):
    """Render on a throwaway axes with the given plot() arguments and return it."""
    fig, ax = plt.subplots()
    sdp.plot(ax=ax, **plot_kwargs)
    plt.close(fig)
    return ax


def _visible_gridlines(ax) -> list:
    return [gl for gl in ax.get_xgridlines() + ax.get_ygridlines() if gl.get_visible()]


def _zone_labels(ax) -> list:
    return [t.get_text() for t in ax.texts]


def _label_x(ax, label: str) -> float:
    return next(t.get_position()[0] for t in ax.texts if t.get_text() == label)


def _bounded_above() -> ShiftedDistributionPlot:
    """RH-like: a reference pressed against the 100 ceiling, so mean + 3 SD overshoots the grid."""
    rng = np.random.default_rng(7)
    values = np.clip(100 - np.abs(rng.normal(0, 4.5, len(INDEX))), 0, 100)
    in_comp = (INDEX >= COMP_PERIOD[0]) & (INDEX <= COMP_PERIOD[1])
    values[in_comp] = np.clip(100 - np.abs(rng.normal(0, 18.0, in_comp.sum())), 0, 100)
    return ShiftedDistributionPlot(pd.Series(values, index=INDEX, name='RH'), REF_PERIOD, COMP_PERIOD)


def _bounded_below() -> ShiftedDistributionPlot:
    """Precipitation-like gamma: mean - 3 SD undershoots the grid's lower end."""
    rng = np.random.default_rng(7)
    values = rng.gamma(1.2, 3.0, len(INDEX))
    return ShiftedDistributionPlot(pd.Series(values, index=INDEX, name='PREC'), REF_PERIOD, COMP_PERIOD)


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
        margin = ref.std(ddof=1)  # The grid margin is the reference *sample* sd
        x = np.linspace(all_vals.min() - margin, all_vals.max() + margin, 1000)
        np.testing.assert_array_equal(self.sdp._x, x)
        for name, data, got in (('ref', ref, self.sdp._ref_kde), ('comp', comp, self.sdp._comp_kde)):
            with self.subTest(period=name):
                bw = 1.06 * data.std() * len(data) ** (-0.2)
                kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data.reshape(-1, 1))
                np.testing.assert_array_equal(got, np.exp(kde.score_samples(x.reshape(-1, 1))))

    def test_breakpoints_are_the_reference_mean_plus_minus_one_and_three_sd(self):
        ref = self.series.loc[REF_PERIOD[0]:REF_PERIOD[1]].dropna().values
        expected = [ref.mean() + k * ref.std(ddof=1) for k in (-3, -1, 1, 3)]
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


class TestFormatStyleContract(unittest.TestCase):
    """The caller's FormatStyle owns the chrome; the class overrides only what it re-draws."""

    def setUp(self):
        self.sdp = ShiftedDistributionPlot(_series(), REF_PERIOD, COMP_PERIOD)

    def test_a_caller_set_ylabel_wins_over_the_density_default(self):
        """'Density' is the default y-label, not an override of the caller's."""
        self.assertEqual(_styled(self.sdp).get_ylabel(), 'Density')
        self.assertEqual(_styled(self.sdp, format_style=FormatStyle()).get_ylabel(), 'Density')
        ax = _styled(self.sdp, format_style=FormatStyle(ylabel='Probability density (1/K)'))
        self.assertEqual(ax.get_ylabel(), 'Probability density (1/K)')
        # Units still append to whichever label won.
        ax = _styled(self.sdp, format_style=FormatStyle(ylabel='p', yunits='(1/K)'))
        self.assertEqual(ax.get_ylabel(), 'p (1/K)')

    def test_the_grid_is_off_by_default_but_the_caller_can_turn_it_on(self):
        """Grid-off belongs to this plot's default style, not on top of the caller's."""
        self.assertEqual(_visible_gridlines(_styled(self.sdp)), [])
        self.assertEqual(_visible_gridlines(_styled(self.sdp, format_style=FormatStyle(show_grid=False))), [])
        ax = _styled(self.sdp, format_style=FormatStyle(show_grid=True, grid_color='#FF0000'))
        gridlines = _visible_gridlines(ax)
        self.assertGreater(len(gridlines), 0)
        self.assertEqual(gridlines[0].get_color(), '#FF0000')

    def test_a_caller_set_title_is_drawn_once_and_left_aligned(self):
        """apply() must not also write a centred copy of the title this plot places itself."""
        ax = _styled(self.sdp, format_style=FormatStyle(title='MY TITLE'))
        self.assertEqual(ax.get_title(loc='left'), 'MY TITLE')
        self.assertEqual(ax.get_title(loc='center'), '')
        # show_title=False must leave no title at all, in either position.
        ax = _styled(self.sdp, format_style=FormatStyle(title='MY TITLE'), show_title=False)
        self.assertEqual(ax.get_title(loc='left'), '')
        self.assertEqual(ax.get_title(loc='center'), '')

    def test_the_remaining_chrome_fields_reach_the_axes(self):
        """Colours, fonts and spine geometry are the caller's, unchanged by this plot."""
        style = FormatStyle(xlabel='MY X', axlabel_fontsize=34.0, text_color='#FF00FF',
                            chrome_color='#00FF00', facecolor='#FFFF00', spine_linewidth=4.5)
        ax = _styled(self.sdp, format_style=style)
        self.assertEqual(ax.get_xlabel(), 'MY X')
        self.assertEqual(ax.xaxis.label.get_fontsize(), 34.0)
        self.assertEqual(ax.xaxis.label.get_color(), '#FF00FF')
        self.assertEqual(ax.spines['bottom'].get_linewidth(), 4.5)
        self.assertEqual(ax.get_facecolor(), (1.0, 1.0, 0.0, 1.0))

    def test_plot_does_not_mutate_the_caller_style(self):
        """The forced title/legend suppression must land on a copy."""
        style = FormatStyle(ylabel='CALLER', title='CALLER TITLE', show_grid=True)
        _styled(self.sdp, format_style=style)
        self.assertEqual((style.ylabel, style.title), ('CALLER', 'CALLER TITLE'))
        self.assertEqual((style.show_grid, style.show_legend, style.show_zeroline), (True, True, True))


class TestZonesOutsideTheEvaluationGrid(unittest.TestCase):
    """A +-3 SD breakpoint beyond the grid must not leave a zone unpainted under its label."""

    def test_bounded_above_drops_the_highest_zone_and_recentres_its_neighbour(self):
        sdp = _bounded_above()
        x, bp = sdp._x, sdp.breakpoints
        self.assertGreater(bp[3], x[-1])  # Precondition: +3 SD overshoots the grid
        ax = _plotted(sdp)
        # 'Extremely hot' spans [+3 SD, grid end], which is empty: not painted, not labelled.
        self.assertEqual(_zone_labels(ax), ['Extremely cold', 'Cold', 'Normal', 'Hot'])
        self.assertEqual(len(ax.collections), 5)  # 1 hatched reference + 4 zone fills
        # 'Hot' is clipped to [+1 SD, grid end] and its label sits at that midpoint.
        self.assertAlmostEqual(_label_x(ax, 'Hot'), (bp[2] + x[-1]) / 2)

    def test_bounded_below_drops_the_lowest_zone_and_recentres_its_neighbour(self):
        sdp = _bounded_below()
        x, bp = sdp._x, sdp.breakpoints
        self.assertLess(bp[0], x[0])  # Precondition: -3 SD undershoots the grid
        ax = _plotted(sdp)
        self.assertEqual(_zone_labels(ax), ['Cold', 'Normal', 'Hot', 'Extremely hot'])
        self.assertEqual(len(ax.collections), 5)
        self.assertAlmostEqual(_label_x(ax, 'Cold'), (x[0] + bp[1]) / 2)

    def test_a_breakpoint_off_the_grid_draws_no_line_and_leaves_the_axis_alone(self):
        sdp = _bounded_below()
        x = sdp._x
        ax = _plotted(sdp)
        self.assertEqual(len(ax.lines), 5)  # 2 KDE outlines + 3 in-grid breakpoint lines
        for line in ax.lines[2:]:
            self.assertTrue(x[0] <= line.get_xdata()[0] <= x[-1])
        # Nothing is drawn beyond the grid, so the axis keeps its plain 5% autoscale margin.
        span = x[-1] - x[0]
        np.testing.assert_allclose(ax.get_xlim(), (x[0] - 0.05 * span, x[-1] + 0.05 * span))

    def test_every_zone_label_sits_over_the_zone_it_names(self):
        """The invariant behind both findings, checked on skewed and symmetric data alike."""
        cases = {'bounded above': _bounded_above(), 'bounded below': _bounded_below(),
                 'symmetric': ShiftedDistributionPlot(_series(), REF_PERIOD, COMP_PERIOD)}
        for name, sdp in cases.items():
            with self.subTest(case=name):
                x = sdp._x
                ax = _plotted(sdp)
                fills = [c for c in ax.collections if c.get_hatch() is None]
                self.assertEqual(len(_zone_labels(ax)), len(fills))
                for text, fill in zip(_zone_labels(ax), fills, strict=True):
                    lo, hi = fill.get_paths()[0].vertices[:, 0].min(), fill.get_paths()[0].vertices[:, 0].max()
                    pos = _label_x(ax, text)
                    self.assertGreater(hi, lo, f"{text} is painted with no width")
                    self.assertTrue(x[0] <= pos <= x[-1], f"{text} at {pos} is off the grid")
                    # Within one grid step of the painted span, which is discretised on x.
                    step = x[1] - x[0]
                    self.assertTrue(lo - step <= pos <= hi + step,
                                    f"{text} at {pos} is not over its fill [{lo}, {hi}]")


class TestZoneListLengths(unittest.TestCase):
    """The plot has exactly five zones, so a list of any other length is caller error."""

    def setUp(self):
        self.sdp = ShiftedDistributionPlot(_series(), REF_PERIOD, COMP_PERIOD)

    def test_a_short_or_long_zone_list_raises_naming_the_argument_and_the_count(self):
        cases = {'zone_colors 3': (dict(zone_colors=['#111111'] * 3), 'zone_colors', 3),
                 'zone_labels 3': (dict(zone_labels=list('abc')), 'zone_labels', 3),
                 'zone_labels 6': (dict(zone_labels=list('abcdef')), 'zone_labels', 6),
                 'zone_colors 6': (dict(zone_colors=['#111111'] * 6), 'zone_colors', 6)}
        for name, (kwargs, argname, count) in cases.items():
            with self.subTest(case=name):
                fig, ax = plt.subplots()
                with self.assertRaises(ValueError) as raised:
                    self.sdp.plot(ax=ax, **kwargs)
                plt.close(fig)
                self.assertEqual(
                    str(raised.exception),
                    f"ShiftedDistributionPlot: `{argname}` needs exactly 5 entries, "
                    f"one per zone from lowest to highest, but got {count}.")

    def test_the_deprecated_constructor_lists_are_validated_too(self):
        """The constructor values are resolved in plot(), so they meet the same check."""
        with self.assertWarns(DeprecationWarning):
            sdp = ShiftedDistributionPlot(_series(), REF_PERIOD, COMP_PERIOD, zone_labels=list('abc'))
        fig, ax = plt.subplots()
        with self.assertRaisesRegex(ValueError, r'`zone_labels` needs exactly 5 entries'):
            sdp.plot(ax=ax)
        plt.close(fig)

    def test_five_entries_are_accepted_and_reach_the_axes(self):
        """The check must pass valid lists through untouched, colours included."""
        ax = _styled(self.sdp, zone_labels=list('abcde'), zone_colors=['#123456'] * 5)
        self.assertEqual(_zone_labels(ax), list('abcde'))
        self.assertEqual([t.get_color() for t in ax.texts], ['#123456'] * 5)
        fills = [c for c in ax.collections if c.get_hatch() is None]
        self.assertEqual(len(fills), 5)


class TestRepeatedPlotOnTheSameAxes(unittest.TestCase):
    """A second plot() on one axes replaces this plot; on another axes both survive."""

    @staticmethod
    def _counts(ax) -> tuple:
        return len(ax.collections), len(ax.lines), len(ax.texts)

    @staticmethod
    def _geometry(ax) -> list:
        return ([(t.get_text(), t.get_position()) for t in ax.texts]
                + [(np.asarray(l.get_xdata()).tolist(), np.asarray(l.get_ydata()).tolist())
                   for l in ax.lines])

    def test_a_second_plot_replaces_the_first_instead_of_stacking(self):
        sdp = ShiftedDistributionPlot(_series(), REF_PERIOD, COMP_PERIOD)
        fig, ax = plt.subplots()
        sdp.plot(ax=ax)
        first_counts, first_geometry = self._counts(ax), self._geometry(ax)
        first_limits = (ax.get_xlim(), ax.get_ylim())
        for repeat in (2, 3):
            sdp.plot(ax=ax)
            with self.subTest(call=repeat):
                self.assertEqual(self._counts(ax), first_counts)
                self.assertEqual(self._geometry(ax), first_geometry)
                self.assertEqual((ax.get_xlim(), ax.get_ylim()), first_limits)
        plt.close(fig)
        self.assertEqual(first_counts, (6, 6, 5))  # The single-plot artist counts

    def test_a_repeat_plot_removes_only_this_plots_own_artists(self):
        """Whatever the caller drew on the axes first must still be there afterwards."""
        fig, ax = plt.subplots()
        caller_line, = ax.plot([0.0, 1.0], [0.0, 1.0])
        caller_text = ax.text(0.5, 0.5, 'CALLER')
        sdp = ShiftedDistributionPlot(_series(), REF_PERIOD, COMP_PERIOD)
        sdp.plot(ax=ax)
        sdp.plot(ax=ax)
        self.assertIn(caller_line, list(ax.lines))
        self.assertIn(caller_text, list(ax.texts))
        self.assertEqual((len(ax.collections), len(ax.lines), len(ax.texts)), (6, 7, 6))
        plt.close(fig)

    def test_plotting_on_a_different_axes_leaves_the_earlier_one_drawn(self):
        """Re-callability across axes is the two-phase contract; only the same axes is replaced."""
        sdp = ShiftedDistributionPlot(_series(), REF_PERIOD, COMP_PERIOD)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sdp.plot(ax=ax1)
        sdp.plot(ax=ax2)
        self.assertEqual(self._counts(ax1), (6, 6, 5))
        self.assertEqual(self._counts(ax2), (6, 6, 5))
        plt.close(fig)

    def test_the_no_data_message_is_not_stacked_either(self):
        """The early-return path draws a text and must take it back as well."""
        sdp = ShiftedDistributionPlot(_series(), OUTSIDE_PERIOD, ('2060-01-01', '2060-12-31'))
        fig, ax = plt.subplots()
        sdp.plot(ax=ax)
        sdp.plot(ax=ax)
        self.assertEqual(_zone_labels(ax), ["No data in reference or comparison period"])
        plt.close(fig)

    def test_a_repeat_plot_survives_the_caller_clearing_the_axes(self):
        """Cleared artists are detached, so there is nothing left for this plot to take back."""
        sdp = ShiftedDistributionPlot(_series(), REF_PERIOD, COMP_PERIOD)
        fig, ax = plt.subplots()
        sdp.plot(ax=ax)
        ax.clear()
        sdp.plot(ax=ax)
        self.assertEqual(self._counts(ax), (6, 6, 5))
        plt.close(fig)


class TestSampleStandardDeviation(unittest.TestCase):
    """Zone boundaries come from the reference *sample* sd (ddof=1), as elsewhere in diive."""

    def test_breakpoints_match_pandas_ddof_1_and_not_the_population_sd(self):
        series = _series()
        sdp = ShiftedDistributionPlot(series, REF_PERIOD, COMP_PERIOD)
        ref = series.loc[REF_PERIOD[0]:REF_PERIOD[1]].dropna()
        np.testing.assert_array_equal(
            sdp.breakpoints, [ref.mean() + k * ref.std() for k in (-3, -1, 1, 3)])
        population = [ref.mean() + k * ref.values.std() for k in (-3, -1, 1, 3)]
        self.assertNotEqual(list(sdp.breakpoints), population)

    def test_a_short_reference_widens_the_zones_by_the_bessel_factor(self):
        """The shorter the reference, the more the population sd understated its spread."""
        series = _series()
        for n, ref_period in ((10, ('1990-01-01', '1990-01-10')), (31, ('1990-01-01', '1990-01-31'))):
            with self.subTest(n=n):
                sdp = ShiftedDistributionPlot(series, ref_period, COMP_PERIOD)
                ref = series.loc[ref_period[0]:ref_period[1]].dropna().values
                self.assertEqual(len(ref), n)
                population = np.array([ref.mean() + k * ref.std() for k in (-3, -1, 1, 3)])
                widening = np.sqrt(n / (n - 1))
                # Every boundary moves outward by (sqrt(n/(n-1)) - 1) times its sigma distance.
                np.testing.assert_allclose(
                    np.asarray(sdp.breakpoints) - ref.mean(),
                    (population - ref.mean()) * widening)
                self.assertGreater(sdp.breakpoints[3] - sdp.breakpoints[0],
                                   population[3] - population[0])

    def test_a_single_record_reference_keeps_its_collapsed_breakpoints(self):
        """One record has no sample sd; NaN there would drop the zones L118 established."""
        sdp = ShiftedDistributionPlot(_series(), ('1990-06-01', '1990-06-01'), COMP_PERIOD)
        value = _series().loc['1990-06-01']
        np.testing.assert_array_equal(sdp.breakpoints, [value] * 4)
        self.assertEqual(len(_plotted(sdp).texts), 5)  # Zones still drawn


if __name__ == '__main__':
    unittest.main()
