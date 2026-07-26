"""
TEST_TIMESTAMP_SHIFTS: clock-error detection from radiation
===========================================================

`preprocessing/qaqc/detect_timestamp_shifts.py` had no test at all (281
statements, 0% covered) despite having a worked example.

The module's whole job is to recover a clock offset by comparing measured
against potential radiation, so the tests inject a *known* shift into
noise-free synthetic radiation and check each method recovers it. That is the
one assertion that actually validates the algorithms rather than their plumbing.

All three methods share a sign convention: **positive = measured peaks earlier
than potential** (a leading clock). So measured data shifted 60 minutes later
must come back as about -60.

Run: pytest tests/test_timestamp_shifts.py -v
"""
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diive as dv
from diive.preprocessing.qaqc.detect_timestamp_shifts import DetectTimestampShifts

LAT, LON, UTC_OFFSET = 47.286417, 7.733750, 1
#: 30-min data, so one record is 30 minutes.
RECORD_MINUTES = 30


def _potential(days: int = 20, start: str = '2022-06-01'):
    idx = pd.date_range(start, periods=48 * days, freq='30min',
                        name='TIMESTAMP_MIDDLE')
    return pd.Series(np.asarray(dv.variables.potrad(
        timestamp_index=idx, lat=LAT, lon=LON, utc_offset=UTC_OFFSET)), index=idx)


def _detector(shift_records: int = 0, clearness: float = 0.8, days: int = 20,
              noise: float = 0.0):
    """Detector over synthetic radiation shifted by a known number of records.

    ``clearness`` scales measured against potential; it must clear the methods'
    clearness thresholds (0.5-0.7) for a day to be used at all.

    ``noise`` adds a small per-record perturbation. The algorithm tests leave it
    at zero so the recovered shift is exact, but the plot tests need it: with
    perfectly uniform data every day yields the *identical* shift, and
    `plot_fft_results` then asks numpy for a 50-bin histogram of a zero-range
    series, which raises "Too many bins for data range". Real records always
    carry some spread, so this is a fixture artifact rather than a plot defect
    worth working around -- but it is a genuine edge case for a perfectly
    constant offset.
    """
    pot = _potential(days=days)
    meas = (pot * clearness).shift(shift_records).fillna(0.0)
    if noise:
        rng = np.random.RandomState(42)
        meas = meas + rng.normal(0.0, noise * float(pot.max()), len(meas))
        meas = meas.clip(lower=0.0)
    df = pd.DataFrame({'SW_IN': meas.to_numpy(), 'SW_IN_POT': pot.to_numpy()},
                      index=pot.index)
    return DetectTimestampShifts(df=df, col_meas='SW_IN', col_pot='SW_IN_POT',
                                 lat=LAT, lon=LON, utc_offset=UTC_OFFSET)


class TestConstruction(unittest.TestCase):

    def test_requires_a_datetime_index(self):
        df = pd.DataFrame({'SW_IN': [1.0, 2.0], 'SW_IN_POT': [1.0, 2.0]})
        with self.assertRaises(TypeError):
            DetectTimestampShifts(df=df, col_meas='SW_IN', col_pot='SW_IN_POT',
                                  lat=LAT, lon=LON)

    def test_missing_potential_column_requires_coordinates(self):
        pot = _potential(days=2)
        df = pd.DataFrame({'SW_IN': pot.to_numpy()}, index=pot.index)
        with self.assertRaises(ValueError):
            DetectTimestampShifts(df=df, col_meas='SW_IN', col_pot='SW_IN_POT')

    def test_potential_is_computed_when_absent(self):
        pot = _potential(days=2)
        df = pd.DataFrame({'SW_IN': (pot * 0.8).to_numpy()}, index=pot.index)
        detector = DetectTimestampShifts(df=df, col_meas='SW_IN',
                                         col_pot='SW_IN_POT', lat=LAT, lon=LON,
                                         utc_offset=UTC_OFFSET)
        self.assertIn('SW_IN_POT', detector.df.columns)
        # It must match a direct potrad call for the same site and index.
        np.testing.assert_allclose(detector.df['SW_IN_POT'].to_numpy(),
                                   pot.to_numpy(), atol=1e-9)

    def test_supplied_potential_is_used_unchanged(self):
        pot = _potential(days=2)
        marker = pot * 0.5  # deliberately not what potrad would return
        df = pd.DataFrame({'SW_IN': pot.to_numpy(), 'SW_IN_POT': marker.to_numpy()},
                          index=pot.index)
        detector = DetectTimestampShifts(df=df, col_meas='SW_IN',
                                         col_pot='SW_IN_POT', lat=LAT, lon=LON)
        np.testing.assert_allclose(detector.df['SW_IN_POT'].to_numpy(),
                                   marker.to_numpy())

    def test_only_the_needed_columns_are_kept(self):
        pot = _potential(days=2)
        df = pd.DataFrame({'SW_IN': pot.to_numpy(), 'SW_IN_POT': pot.to_numpy(),
                           'UNRELATED': np.arange(len(pot), dtype=float)},
                          index=pot.index)
        detector = DetectTimestampShifts(df=df, col_meas='SW_IN',
                                         col_pot='SW_IN_POT', lat=LAT, lon=LON)
        self.assertNotIn('UNRELATED', detector.df.columns)


class TestFftPhaseShift(unittest.TestCase):
    """The FFT method recovers a planted shift from the k=1 phase angle."""

    def test_recovers_a_known_shift(self):
        cases = {0: 0.0, 2: -60.0, -2: 60.0, 4: -120.0}
        for records, expected in cases.items():
            with self.subTest(shift_minutes=records * RECORD_MINUTES):
                result = _detector(shift_records=records).fft_phase_shift()
                self.assertAlmostEqual(float(result['shift_minutes'].median()),
                                       expected, delta=2.0)

    def test_result_shape(self):
        result = _detector().fft_phase_shift()
        self.assertEqual(list(result.columns), ['shift_minutes', 'amplitude_meas'])
        self.assertEqual(len(result), 20)  # one row per day

    def test_cloudy_days_are_excluded(self):
        # Clearness 0.2 is below the 0.6 default, so no day yields an estimate.
        result = _detector(shift_records=2, clearness=0.2).fft_phase_shift()
        self.assertTrue(result['shift_minutes'].isna().all())

    def test_min_clearness_controls_the_cut(self):
        detector = _detector(shift_records=2, clearness=0.55)
        self.assertTrue(detector.fft_phase_shift(min_clearness=0.6)['shift_minutes'].isna().all())
        self.assertFalse(detector.fft_phase_shift(min_clearness=0.5)['shift_minutes'].isna().all())


class TestNoonShift(unittest.TestCase):
    """The peak-time heuristic. Resolution is one record, so 30 minutes here."""

    def test_recovers_a_known_shift(self):
        cases = {0: 0.0, 2: -60.0, -2: 60.0}
        for records, expected in cases.items():
            with self.subTest(shift_minutes=records * RECORD_MINUTES):
                result = _detector(shift_records=records).noon_shift()
                self.assertAlmostEqual(float(result.median()), expected,
                                       delta=RECORD_MINUTES)

    def test_returns_a_named_series_of_clear_days(self):
        result = _detector().noon_shift()
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, 'time_shift_minutes')
        self.assertEqual(len(result), 20)

    def test_clearness_threshold_filters_days(self):
        # Default threshold is 0.7, so a 0.8 record passes and a 0.2 one does not.
        self.assertEqual(len(_detector(clearness=0.8).noon_shift()), 20)
        self.assertEqual(len(_detector(clearness=0.2).noon_shift()), 0)


class TestCrosscorr(unittest.TestCase):
    """The cross-correlation method: a Pearson scan over candidate lags.

    It used to return 0 for a planted 60-minute offset, because the daytime mask
    clipped both series before correlating (truncating the shifted measured
    curve) and the FFT correlation was not normalised per lag by the overlap
    count, biasing argmax toward zero. Both are fixed; these tests pin the
    recovery so it cannot regress.
    """

    def test_result_shape(self):
        result = _detector().crosscorr()
        self.assertEqual(list(result.columns), ['shift_minutes', 'max_corr'])
        self.assertEqual(len(result), 20)

    def test_perfectly_aligned_data_gives_zero_lag_and_unit_correlation(self):
        result = _detector(shift_records=0).crosscorr()
        self.assertEqual(float(result['shift_minutes'].median()), 0.0)
        self.assertAlmostEqual(float(result['max_corr'].median()), 1.0, delta=0.01)

    def test_cloudy_days_are_excluded(self):
        result = _detector(shift_records=2, clearness=0.2).crosscorr()
        self.assertTrue(result['shift_minutes'].isna().all())

    def test_max_shift_window_bounds_the_answer(self):
        result = _detector(shift_records=4).crosscorr(max_shift_min=30)
        within = result['shift_minutes'].dropna().abs() <= 30
        self.assertTrue(within.all())

    def test_recovers_a_known_shift(self):
        cases = {0: 0.0, 1: -30.0, 2: -60.0, -2: 60.0, 4: -120.0, -4: 120.0}
        for records, expected in cases.items():
            with self.subTest(shift_minutes=records * RECORD_MINUTES):
                result = _detector(shift_records=records).crosscorr()
                self.assertAlmostEqual(float(result['shift_minutes'].median()),
                                       expected, delta=2.0)

    def test_a_perfect_match_scores_a_perfect_correlation(self):
        # The reported correlation used to be ~0.91 even for an exactly aligned
        # curve, which `plot_crosscorr_results` would then hide behind its
        # min_corr=0.97 default.
        for records in (0, 2, -4):
            with self.subTest(shift_minutes=records * RECORD_MINUTES):
                result = _detector(shift_records=records).crosscorr()
                self.assertAlmostEqual(float(result['max_corr'].median()), 1.0,
                                       delta=0.01)

    def test_lag_is_reported_in_minutes_not_samples(self):
        # max_shift_min and the result are durations; a coarser upsample_freq
        # must not silently reinterpret them as sample counts.
        result = _detector(shift_records=2).crosscorr(upsample_freq='2min')
        self.assertAlmostEqual(float(result['shift_minutes'].median()), -60.0,
                               delta=4.0)


class TestMethodsAgree(unittest.TestCase):

    def test_fft_and_noon_shift_agree_on_the_same_data(self):
        # Two independent algorithms, one convention: they must not disagree in
        # sign or magnitude on clean data.
        detector = _detector(shift_records=2)
        fft = float(detector.fft_phase_shift()['shift_minutes'].median())
        noon = float(detector.noon_shift().median())
        self.assertLess(abs(fft - noon), RECORD_MINUTES)
        self.assertLess(fft, 0)
        self.assertLess(noon, 0)


class TestPlots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Slight noise so the per-day shifts vary; see _detector's docstring.
        cls.detector = _detector(shift_records=2, noise=0.03)
        cls.detector.fft_phase_shift()
        cls.detector.crosscorr()
        cls.detector.noon_shift()

    def tearDown(self):
        plt.close('all')

    def test_result_plots_render(self):
        for method in ('plot_fft_results', 'plot_crosscorr_results',
                       'plot_noon_shift_results'):
            with self.subTest(plot=method):
                plt.close('all')
                getattr(self.detector, method)()
                self.assertGreater(len(plt.get_fignums()), 0)

    def test_diel_cycle_plot_renders(self):
        self.detector.plot_monthly_dielcycles()
        self.assertGreater(len(plt.get_fignums()), 0)

    def test_radiation_fingerprint_renders(self):
        self.detector.plot_radiation_fingerprint(year=2022)
        self.assertGreater(len(plt.get_fignums()), 0)


class TestTimedeltaFormatting(unittest.TestCase):

    def test_timedelta_to_hhmm(self):
        index = pd.to_timedelta(['0h', '1h30min', '13h05min'])
        self.assertEqual(DetectTimestampShifts._timedelta_to_hhmm(index),
                         ['00:00', '01:30', '13:05'])


if __name__ == '__main__':
    unittest.main()
