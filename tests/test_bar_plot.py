"""
TESTS: LONG-TERM ANNUAL ANOMALIES BAR PLOT
==========================================

Tests that `LongtermAnomaliesYear` draws one bar slot per calendar year of the
span its title asserts, so a record with a multi-year outage leaves a visible
hole instead of closing the gap.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest
import warnings

import matplotlib

matplotlib.use('Agg')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diive.core.plotting.bar import LongtermAnomaliesYear
from diive.core.plotting.styles.format import FormatStyle

# The reported case: a 72-year record with a 12-year outage in the middle.
FIRST_YEAR, LAST_YEAR = 1950, 2021
OUTAGE = range(1980, 1992)


def _yearly(years, seed: int = 42) -> pd.Series:
    """Return one warming-trend value per year, indexed by integer year."""
    years = list(years)
    rng = np.random.default_rng(seed)
    values = 8.0 + 0.02 * (np.array(years) - FIRST_YEAR) + rng.normal(0, 0.3, len(years))
    return pd.Series(values, index=years, name='TA')


def _bar_slots(ax, n_years: int):
    """Return (x centres, above heights, below heights), one entry per year.

    The two series are drawn one after the other, so the patch list holds all
    above-bars first and then all below-bars, each with one rectangle per year.
    """
    patches = ax.patches
    centres = [p.get_x() + p.get_width() / 2 for p in patches[:n_years]]
    above = [p.get_height() for p in patches[:n_years]]
    below = [p.get_height() for p in patches[n_years:]]
    return centres, above, below


class TestLongtermAnomaliesYearLattice(unittest.TestCase):
    """L114: `plot.bar` is categorical, so an absent year used to take no axis width."""

    def test_outage_years_occupy_their_own_bar_slots(self):
        series = _yearly(y for y in range(FIRST_YEAR, LAST_YEAR + 1) if y not in OUTAGE)
        self.assertEqual(len(series), 60)  # guard: the outage really is missing

        plot = LongtermAnomaliesYear(series=series, reference_start_year=1950,
                                     reference_end_year=1979, series_label='TA')
        anomalies = plot.anomalies_df
        n_years = LAST_YEAR - FIRST_YEAR + 1
        self.assertEqual(list(anomalies.index), list(range(FIRST_YEAR, LAST_YEAR + 1)))

        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        centres, above, below = _bar_slots(ax, n_years)
        self.assertEqual(len(ax.patches), 2 * n_years)
        # One evenly spaced slot per calendar year, the outage included.
        self.assertEqual(centres, [float(i) for i in range(n_years)])
        # The outage spans twelve slots, not one bar-width jump.
        pos = {yr: centres[i] for i, yr in enumerate(anomalies.index)}
        self.assertEqual(pos[1992] - pos[1979], 13.0)
        # Nothing is painted in them: no anomaly, and both bars are flat.
        for yr in OUTAGE:
            i = list(anomalies.index).index(yr)
            self.assertTrue(np.isnan(anomalies.loc[yr, 'anomaly']))
            self.assertEqual(above[i], 0.0)
            self.assertEqual(below[i], 0.0)
        # The measured years on either side of the outage still carry their bars.
        for yr in (1979, 1992):
            i = list(anomalies.index).index(yr)
            self.assertNotEqual(above[i] + below[i], 0.0)
        plt.close(fig)

    def test_title_span_matches_the_number_of_bar_slots(self):
        series = _yearly(y for y in range(FIRST_YEAR, LAST_YEAR + 1) if y not in OUTAGE)
        plot = LongtermAnomaliesYear(series=series, reference_start_year=1950,
                                     reference_end_year=1979, series_label='TA')
        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        self.assertEqual(ax.get_title(), f"TA anomaly per year ({FIRST_YEAR}-{LAST_YEAR})")
        n_years = LAST_YEAR - FIRST_YEAR + 1
        self.assertEqual(len(ax.patches), 2 * n_years)
        # The x limits are set from the same count, so the axis covers the span too.
        self.assertEqual(ax.get_xlim(), (-1.0, float(n_years)))
        plt.close(fig)

    def test_reference_statistics_ignore_the_injected_years(self):
        # The reference period itself has an outage, so the injected NaN years land
        # inside the subset the mean and sd are computed over.
        years = [y for y in range(2000, 2021) if y not in (2005, 2006, 2007)]
        series = pd.Series(np.arange(len(years), dtype=float), index=years, name='TA')
        plot = LongtermAnomaliesYear(series=series, reference_start_year=2000,
                                     reference_end_year=2010)
        anomalies = plot.anomalies_df

        measured = series.loc[(series.index >= 2000) & (series.index <= 2010)]
        self.assertEqual(len(measured), 8)  # guard: three reference years are missing
        self.assertAlmostEqual(float(anomalies['reference_mean'].iloc[-1]), float(measured.mean()))
        self.assertAlmostEqual(float(anomalies['reference_sd'].iloc[-1]), float(measured.std()))
        # The injected years are present as empty rows, not as zeros.
        for yr in (2005, 2006, 2007):
            self.assertIn(yr, anomalies.index)
            self.assertTrue(np.isnan(anomalies.loc[yr, 'TA']))
            self.assertTrue(np.isnan(anomalies.loc[yr, 'anomaly']))
        # Every measured year keeps its anomaly against that unchanged mean.
        for yr, value in measured.items():
            self.assertAlmostEqual(float(anomalies.loc[yr, 'anomaly']),
                                   value - float(measured.mean()))

    def test_last_ten_years_annotation_covers_ten_calendar_years(self):
        # With a gap inside the tail the annotation used to average the last ten
        # *measured* years while labelling itself with their (wider) span.
        years = [y for y in range(2000, 2021) if y not in (2015, 2016, 2017)]
        series = pd.Series(np.arange(len(years), dtype=float), index=years, name='TA')
        plot = LongtermAnomaliesYear(series=series, reference_start_year=2000,
                                     reference_end_year=2010)
        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        annotation = [t.get_text() for t in ax.texts if 'last 10 years mean' in t.get_text()]
        self.assertEqual(len(annotation), 1)
        self.assertIn("(2011-2020)", annotation[0])
        tail = series.loc[series.index >= 2011]
        self.assertIn(f"last 10 years mean: {float(tail.mean()):.2f}", annotation[0])
        plt.close(fig)

    def test_contiguous_record_is_untouched(self):
        # The reindex must be a no-op on a gapless record: same rows, same values,
        # same bar geometry.
        years = list(range(FIRST_YEAR, LAST_YEAR + 1))
        series = _yearly(years, seed=7)
        plot = LongtermAnomaliesYear(series=series, reference_start_year=1950,
                                     reference_end_year=1979, series_label='TA')
        anomalies = plot.anomalies_df
        self.assertEqual(list(anomalies.index), years)
        pd.testing.assert_series_equal(anomalies['TA'], series, check_names=False)

        measured = series.loc[(series.index >= 1950) & (series.index <= 1979)]
        ref_mean = float(measured.mean())
        self.assertAlmostEqual(float(anomalies['reference_mean'].iloc[-1]), ref_mean)
        self.assertAlmostEqual(float(anomalies['reference_sd'].iloc[-1]), float(measured.std()))

        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        centres, above, below = _bar_slots(ax, len(years))
        self.assertEqual(len(ax.patches), 2 * len(years))
        self.assertEqual(centres, [float(i) for i in range(len(years))])
        drawn = [a + b for a, b in zip(above, below)]
        for got, want in zip(drawn, [v - ref_mean for v in series]):
            self.assertAlmostEqual(got, want)
        plt.close(fig)


class TestLongtermAnomaliesYearEmptyInput(unittest.TestCase):
    """L120: input holding nothing to anomalise used to fail internally or draw nothing."""

    def test_empty_series_is_rejected_by_name(self):
        # Used to fail inside the year lattice with "cannot convert float NaN to
        # integer", which names an internal detail rather than the empty input.
        for label, series in (('typed index', pd.Series([], index=pd.Index([], dtype='int64'),
                                                        dtype=float, name='TA')),
                              ('default index', pd.Series([], dtype=float, name='TA'))):
            with self.subTest(series=label):
                with self.assertRaises(ValueError) as caught:
                    LongtermAnomaliesYear(series=series, reference_start_year=2000,
                                          reference_end_year=2010)
                message = str(caught.exception)
                self.assertIn('LongtermAnomaliesYear', message)
                self.assertIn('at least one year of data', message)
                self.assertNotIn('convert float NaN', message)

    def test_all_missing_series_is_rejected_by_name(self):
        # Not the same input as an empty series: it has years, and the GUI's own
        # `.empty` guard lets it through.
        series = pd.Series([np.nan] * 5, index=[2016, 2017, 2018, 2019, 2020], name='TA')
        with self.assertRaises(ValueError) as caught:
            LongtermAnomaliesYear(series=series, reference_start_year=2016,
                                  reference_end_year=2018)
        self.assertIn('at least one year of data', str(caught.exception))
        self.assertIn('length 5', str(caught.exception))


class TestLongtermAnomaliesYearReferencePeriod(unittest.TestCase):
    """L120: a reference period covering no measurement annotated "nan±nansd"."""

    @staticmethod
    def _record():
        years = list(range(2000, 2021))
        return pd.Series(np.arange(len(years), dtype=float), index=years, name='TA')

    def test_reference_period_without_data_is_rejected(self):
        series = self._record()
        for label, (start, end) in (('entirely before the record', (1950, 1960)),
                                    ('entirely after the record', (2030, 2040)),
                                    ('single year outside', (1990, 1990)),
                                    ('reversed', (2010, 2000))):
            with self.subTest(period=label):
                with self.assertRaises(ValueError) as caught:
                    LongtermAnomaliesYear(series=series, reference_start_year=start,
                                          reference_end_year=end)
                message = str(caught.exception)
                # Both the rejected period and the span it should have been inside.
                self.assertIn(f"{start}-{end}", message)
                self.assertIn('2000-2020', message)

    def test_reference_period_inside_an_outage_is_rejected(self):
        # The years exist in the frame after the L114 reindex, but they are all NaN,
        # so the reference statistics come out NaN just the same. This is the case a
        # GUI user can reach, the reference spin boxes being clamped to the record.
        years = [y for y in range(2000, 2021) if y not in range(2005, 2011)]
        series = pd.Series(np.arange(len(years), dtype=float), index=years, name='TA')
        with self.assertRaises(ValueError) as caught:
            LongtermAnomaliesYear(series=series, reference_start_year=2005,
                                  reference_end_year=2010)
        self.assertIn('2005-2010', str(caught.exception))

    def test_reference_period_that_overlaps_at_all_still_plots(self):
        # No-regression guard: only a period with *no* data is rejected. A partial
        # overlap and a single covered year keep their (unchanged) statistics.
        series = self._record()
        for label, (start, end, expected_mean) in (
                ('overlapping below', (1995, 2005, float(series.loc[2000:2005].mean()))),
                ('overlapping above', (2015, 2030, float(series.loc[2015:2020].mean()))),
                ('single covered year', (2005, 2005, float(series.loc[2005])))):
            with self.subTest(period=label):
                plot = LongtermAnomaliesYear(series=series, reference_start_year=start,
                                             reference_end_year=end, series_label='TA')
                self.assertAlmostEqual(float(plot.anomalies_df['reference_mean'].iloc[-1]),
                                       expected_mean)
                fig, ax = plt.subplots()
                plot.plot(ax=ax)
                annotation = [t.get_text() for t in ax.texts if 'reference period mean' in t.get_text()]
                self.assertEqual(len(annotation), 1)
                self.assertIn(f"reference period mean: {expected_mean:.2f}", annotation[0])
                self.assertEqual(len(ax.patches), 2 * len(series))
                plt.close(fig)

    def test_ordinary_record_annotation_is_unchanged(self):
        # No-regression guard: the exact text both guards must leave alone.
        series = pd.Series([1.0, 3.0, 2.0, 5.0, 0.0],
                           index=[2016, 2017, 2018, 2019, 2020], name='TA')
        plot = LongtermAnomaliesYear(series=series, reference_start_year=2016,
                                     reference_end_year=2018, series_label='TA')
        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        # The "last N" is the tail's real length since L149; this record is 5 years.
        self.assertEqual([t.get_text() for t in ax.texts],
                         ["reference period mean: 2.00±1.00sd (2016-2018, 3 years)\n"
                          "last 5 years mean: 2.20±1.92sd (2016-2020)"])
        heights = [p.get_height() for p in ax.patches]
        self.assertEqual(heights, [0.0, 1.0, 0.0, 3.0, 0.0, -1.0, 0.0, 0.0, 0.0, -2.0])
        plt.close(fig)


class TestLongtermAnomaliesYearAnnotationCounts(unittest.TestCase):
    """L149: the annotation's "N years" was `end - start + 1`, a label, never a count."""

    @staticmethod
    def _record(years):
        years = list(years)
        return pd.Series(np.arange(len(years), dtype=float), index=years, name='TA')

    @staticmethod
    def _annotation(plot, needle):
        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        found = [t.get_text() for t in ax.texts if needle in t.get_text()]
        plt.close(fig)
        return found

    def test_partly_overlapping_reference_counts_the_overlap(self):
        # The reported case: 1995-2005 is 11 nominal years but only 2000-2005 is
        # covered, so the mean and sd come from 6 years and used to say "11 years".
        series = self._record(range(2000, 2021))
        plot = LongtermAnomaliesYear(series=series, reference_start_year=1995,
                                     reference_end_year=2005, series_label='TA')
        covered = series.loc[2000:2005]
        self.assertEqual(len(covered), 6)  # guard: the overlap really is 6 years
        annotation = self._annotation(plot, 'reference period mean')
        self.assertEqual(len(annotation), 1)
        # The requested window stays, next to the count actually behind the number.
        self.assertIn(f"reference period mean: {float(covered.mean()):.2f}"
                      f"±{float(covered.std()):.2f}sd (1995-2005, 6 years)", annotation[0])

    def test_outage_inside_the_reference_window_is_not_counted(self):
        # After L114's reindex the missing years are rows, all NaN. mean() and std()
        # skip them, so the count has to as well.
        years = [y for y in range(2000, 2021) if y not in (2005, 2006, 2007)]
        series = self._record(years)
        plot = LongtermAnomaliesYear(series=series, reference_start_year=2000,
                                     reference_end_year=2010, series_label='TA')
        measured = series.loc[(series.index >= 2000) & (series.index <= 2010)]
        self.assertEqual(len(measured), 8)  # guard: three reference years are missing
        annotation = self._annotation(plot, 'reference period mean')
        self.assertIn("(2000-2010, 8 years)", annotation[0])
        self.assertNotIn("11 years", annotation[0])

    def test_fully_measured_reference_prints_its_nominal_span(self):
        # No-regression guard: where every requested year is measured the count and
        # the nominal span coincide, so these strings must not move at all.
        series = self._record(range(2000, 2021))
        for label, (start, end, expected) in (
                ('reference is the whole record', (2000, 2020, '(2000-2020, 21 years)')),
                ('sub-window', (2000, 2010, '(2000-2010, 11 years)')),
                ('single year', (2005, 2005, '(2005-2005, 1 years)'))):
            with self.subTest(period=label):
                plot = LongtermAnomaliesYear(series=series, reference_start_year=start,
                                             reference_end_year=end, series_label='TA')
                annotation = self._annotation(plot, 'reference period mean')
                self.assertEqual(len(annotation), 1)
                self.assertIn(expected, annotation[0])

    def test_last_years_label_matches_the_span_it_prints(self):
        # A record shorter than ten years read "last 10 years mean" beside its own
        # five-year span - the figure contradicting itself.
        series = self._record(range(2016, 2021))
        plot = LongtermAnomaliesYear(series=series, reference_start_year=2016,
                                     reference_end_year=2018, series_label='TA')
        annotation = self._annotation(plot, 'years mean:')
        self.assertEqual(len(annotation), 1)
        self.assertIn("last 5 years mean: ", annotation[0])
        self.assertIn("(2016-2020)", annotation[0])
        self.assertNotIn("last 10 years", annotation[0])

    def test_last_ten_years_label_is_unchanged_on_a_long_record(self):
        # No-regression guard: with at least ten calendar years the tail is ten rows
        # long, gaps in it included, so the label stays exactly "last 10 years".
        years = [y for y in range(2000, 2021) if y not in (2015, 2016, 2017)]
        plot = LongtermAnomaliesYear(series=self._record(years), reference_start_year=2000,
                                     reference_end_year=2010, series_label='TA')
        annotation = self._annotation(plot, 'years mean:')
        self.assertIn("last 10 years mean: ", annotation[0])
        self.assertIn("(2011-2020)", annotation[0])


class TestLongtermAnomaliesYearGetBeforePlot(unittest.TestCase):
    """L139: `get()` before `plot()` raised a bare `AttributeError` on `self.ax`."""

    @staticmethod
    def _record():
        years = list(range(2016, 2022))
        return pd.Series(np.arange(len(years), dtype=float), index=years, name='TA')

    def test_get_before_plot_names_the_missing_step(self):
        plot = LongtermAnomaliesYear(series=self._record(), reference_start_year=2016,
                                     reference_end_year=2018, series_label='TA')
        # The attribute now exists, so the failure cannot be about a missing name.
        self.assertIsNone(plot.ax)
        with self.assertRaises(RuntimeError) as caught:
            plot.get()
        message = str(caught.exception)
        self.assertIn('LongtermAnomaliesYear', message)
        self.assertIn('plot() before get()', message)
        # The old message named an internal attribute instead of the skipped step.
        self.assertNotIn("has no attribute 'ax'", message)

    def test_get_after_plot_returns_that_axes(self):
        # No-regression guard: the documented order keeps working, on a caller's
        # axes and on the one the class makes for itself.
        plot = LongtermAnomaliesYear(series=self._record(), reference_start_year=2016,
                                     reference_end_year=2018, series_label='TA')
        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        self.assertIs(plot.get(), ax)
        self.assertIsNone(plot.fig)  # a caller-supplied axes brings no figure
        plt.close(fig)

        fig2, ax2 = plt.subplots()
        plot.plot(ax=ax2)
        self.assertIs(plot.get(), ax2)
        plt.close(fig2)


class TestLongtermAnomaliesYearReplot(unittest.TestCase):
    """L137: a second `plot()` on the same axes stacked artists instead of replacing them."""

    @staticmethod
    def _record():
        years = list(range(2016, 2022))
        return pd.Series([1.0, 3.0, 2.0, 5.0, 0.0, 4.0], index=years, name='TA')

    def _plot(self):
        return LongtermAnomaliesYear(series=self._record(), reference_start_year=2016,
                                     reference_end_year=2018, series_label='TA')

    def test_replotting_the_same_axes_replaces_the_rendering(self):
        # Went (patches, texts, lines) 12/1/1 -> 24/2/2 -> 36/3/3, the overlaid
        # alpha darkening the figure a little more on every call.
        plot = self._plot()
        fig, ax = plt.subplots()
        n_years = len(self._record())
        for call in (1, 2, 3):
            plot.plot(ax=ax)
            with self.subTest(call=call):
                self.assertEqual(len(ax.patches), 2 * n_years)
                self.assertEqual(len(ax.texts), 1)
                self.assertEqual(len(ax.lines), 1)
                # BarContainers are not Artists, so they outlive their own
                # rectangles unless they are dropped as well.
                self.assertEqual(len(ax.containers), 2)
        plt.close(fig)

    def test_replotting_leaves_the_figure_it_drew_the_first_time(self):
        # Replacing must not mean re-drawing something different: every rendered
        # field is identical after the second call.
        plot = self._plot()
        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        first = ([p.get_height() for p in ax.patches],
                 [p.get_x() for p in ax.patches],
                 [t.get_text() for t in ax.texts],
                 ax.get_title(), ax.get_ylabel(), ax.get_xlim(), len(ax.get_xticks()))
        plot.plot(ax=ax)
        second = ([p.get_height() for p in ax.patches],
                  [p.get_x() for p in ax.patches],
                  [t.get_text() for t in ax.texts],
                  ax.get_title(), ax.get_ylabel(), ax.get_xlim(), len(ax.get_xticks()))
        self.assertEqual(first, second)
        plt.close(fig)

    def test_only_this_instances_own_artists_are_removed(self):
        # The axes is not cleared, so anything the caller drew on it beforehand,
        # and anything a second plotter drew on it, has to survive.
        plot = self._plot()
        fig, ax = plt.subplots()
        foreign_line, = ax.plot([0, 1], [0, 1], color='black')
        foreign_text = ax.text(0.1, 0.1, 'foreign')
        plot.plot(ax=ax)
        plot.plot(ax=ax)
        self.assertIn(foreign_line, ax.lines)
        self.assertIn(foreign_text, ax.texts)
        self.assertEqual(len(ax.texts), 2)  # the foreign one plus one annotation
        self.assertEqual(len(ax.patches), 2 * len(self._record()))
        plt.close(fig)

        # A second plotter deliberately overlaying the same axes keeps both sets.
        other = self._plot()
        fig, ax = plt.subplots()
        plot2 = self._plot()
        plot2.plot(ax=ax)
        other.plot(ax=ax)
        self.assertEqual(len(ax.patches), 4 * len(self._record()))
        self.assertEqual(len(ax.texts), 2)
        plt.close(fig)

    def test_drawing_on_a_second_axes_leaves_the_first_alone(self):
        # No-regression guard: the documented use is one plotter across several
        # axes, and neither may be emptied by the other.
        plot = self._plot()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plot.plot(ax=ax1)
        plot.plot(ax=ax2)
        n_years = len(self._record())
        self.assertEqual(len(ax1.patches), 2 * n_years)
        self.assertEqual(len(ax2.patches), 2 * n_years)
        self.assertEqual(len(ax1.texts), 1)
        self.assertEqual(len(ax2.texts), 1)
        plt.close(fig)

    def test_replotting_after_the_caller_cleared_the_axes(self):
        # ax.clear() detaches the artists itself, so the removal must skip them
        # rather than fail on "cannot remove artist".
        plot = self._plot()
        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        ax.clear()
        plot.plot(ax=ax)
        self.assertEqual(len(ax.patches), 2 * len(self._record()))
        self.assertEqual(len(ax.texts), 1)
        self.assertEqual(len(ax.containers), 2)
        plt.close(fig)

    def test_replotting_applies_the_new_styling(self):
        # The docstring's reason for allowing repeat calls: different styling.
        plot = self._plot()
        fig, ax = plt.subplots()
        plot.plot(ax=ax, format_style=FormatStyle(title='FIRST'))
        plot.plot(ax=ax, format_style=FormatStyle(title='SECOND'))
        self.assertEqual(ax.get_title(), 'SECOND')
        self.assertEqual(len(ax.patches), 2 * len(self._record()))
        plt.close(fig)


class TestLongtermAnomaliesYearColors(unittest.TestCase):
    """L145 concluded against: the drawn colours are pinned so a change stays deliberate.

    The finding reads bar.py's red/blue as out of step with CLAUDE.md's "300-level
    (bars/lines)". Measured against the rest of the package, the code is not the
    outlier: no diive plot fills a bar with a 300-level colour, `histogram.py` fills
    at 400 like this class, and `waterfall.py` (the other bar chart) fills at 500 --
    which is what CLAUDE.md's own listed hexes (#F44336 / #2196F3) actually are.
    """

    def test_above_and_below_bars_keep_their_colours(self):
        series = pd.Series([1.0, 3.0, 2.0, 5.0, 0.0, 4.0],
                           index=list(range(2016, 2022)), name='TA')
        plot = LongtermAnomaliesYear(series=series, reference_start_year=2016,
                                     reference_end_year=2018, series_label='TA')
        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        drawn = {mcolors.to_hex(p.get_facecolor()) for p in ax.patches}
        self.assertEqual(drawn, {'#ef5350', '#42a5f5'})  # red 400, blue 400
        self.assertEqual({p.get_alpha() for p in ax.patches}, {0.9})

        # Above-reference years are red, below-reference years blue. The two series
        # are drawn one after the other, so the first half of the patch list is
        # 'anomaly_above' and the second half 'anomaly_below'.
        n = len(series)
        self.assertEqual({mcolors.to_hex(p.get_facecolor()) for p in ax.patches[:n]}, {'#ef5350'})
        self.assertEqual({mcolors.to_hex(p.get_facecolor()) for p in ax.patches[n:]}, {'#42a5f5'})
        plt.close(fig)


class TestLongtermAnomaliesYearStandaloneLayout(unittest.TestCase):
    """L138: `fig.tight_layout()` disabled the constrained engine `create_ax` asked for."""

    @staticmethod
    def _record():
        years = list(range(1950, 2022))
        values = 8.0 + 0.02 * (np.array(years) - 1950)
        return pd.Series(values, index=years, name='TA')

    def _plot(self):
        return LongtermAnomaliesYear(series=self._record(), reference_start_year=1950,
                                     reference_end_year=1979, series_label='TA',
                                     series_units='(degC)')

    @staticmethod
    def _layout_warnings(caught):
        # plot() also calls fig.show(), which warns on the Agg backend used here.
        return [str(w.message) for w in caught if 'layout' in str(w.message)]

    def test_standalone_figure_keeps_its_constrained_layout(self):
        plot = self._plot()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plot.plot()
        self.assertEqual(self._layout_warnings(caught), [])
        # tight_layout() replaced the engine with a PlaceHolderLayoutEngine.
        self.assertEqual(type(plot.fig.get_layout_engine()).__name__, 'ConstrainedLayoutEngine')
        plt.close(plot.fig)

    def test_standalone_figure_clips_nothing(self):
        # No-regression guard for the layout change: the chrome and the bottom-right
        # reference annotation must still sit inside the figure without the tight
        # padding. Only artists that are actually drawn are checked - matplotlib
        # keeps tick labels for locations outside the view limits too.
        plot = self._plot()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot.plot()
        fig, ax = plot.fig, plot.ax
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        figure_box = fig.get_window_extent()

        checked = [(ax.title, 'title'), (ax.xaxis.label, 'xlabel'), (ax.yaxis.label, 'ylabel')]
        checked += [(t, 'annotation') for t in ax.texts]
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            for location, label in zip(axis.get_ticklocs(), axis.get_ticklabels()):
                if label.get_text() and low <= location <= high:
                    checked.append((label, f'tick {label.get_text()!r}'))

        self.assertGreater(len(checked), 5)  # guard: the loop really found artists
        for artist, name in checked:
            with self.subTest(artist=name):
                box = artist.get_window_extent(renderer)
                self.assertGreaterEqual(round(box.x0, 1), -0.5, f'{name} runs off the left')
                self.assertGreaterEqual(round(box.y0, 1), -0.5, f'{name} runs off the bottom')
                self.assertLessEqual(round(box.x1, 1), figure_box.x1 + 0.5, f'{name} runs off the right')
                self.assertLessEqual(round(box.y1, 1), figure_box.y1 + 0.5, f'{name} runs off the top')
        plt.close(fig)

    def test_caller_supplied_axes_is_unaffected(self):
        # No-regression guard: the layout call only ever ran on the ax=None path,
        # so an embedded plot must be exactly what it was.
        plot = self._plot()
        fig, ax = plt.subplots()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plot.plot(ax=ax)
        self.assertEqual(self._layout_warnings(caught), [])
        self.assertIsNone(plot.fig)  # no figure is created or touched for a given axes
        self.assertEqual(len(ax.patches), 2 * len(self._record()))
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
