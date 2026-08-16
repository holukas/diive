"""
TESTS: LONG-TERM ANNUAL ANOMALIES BAR PLOT
==========================================

Tests that `LongtermAnomaliesYear` draws one bar slot per calendar year of the
span its title asserts, so a record with a multi-year outage leaves a visible
hole instead of closing the gap.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diive.core.plotting.bar import LongtermAnomaliesYear

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
        self.assertEqual([t.get_text() for t in ax.texts],
                         ["reference period mean: 2.00±1.00sd (2016-2018, 3 years)\n"
                          "last 10 years mean: 2.20±1.92sd (2016-2020)"])
        heights = [p.get_height() for p in ax.patches]
        self.assertEqual(heights, [0.0, 1.0, 0.0, 3.0, 0.0, -1.0, 0.0, 0.0, 0.0, -2.0])
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
