import unittest

import numpy as np
import pandas as pd

from diive.preprocessing.outlier_detection.hampel import Hampel

LAT, LON, UTC = 47.478333, 8.364389, 1


def _hampel(series, **kwargs):
    opts = dict(window_length=60, n_sigma=8, use_differencing=True,
                separate_day_night=False, showplot=False, verbose=False)
    opts.update(kwargs)
    return Hampel(series=series, lat=LAT, lon=LON, utc_offset=UTC, **opts)


class TestDegenerateScale(unittest.TestCase):
    """A window where more than half the records are identical has MAD == 0.
    The detection band then has zero width and every change in the signal looks
    like an outlier, which used to reject the signal wholesale."""

    def _upsampled(self):
        """Coarse data upsampled onto a finer grid: runs of ten identical values,
        exactly what StepwiseMeteoScreeningDb produces for a record whose logger
        changed sampling rate."""
        idx = pd.date_range('2020-05-01', periods=3000, freq='1min', name='TIMESTAMP_MIDDLE')
        coarse = np.repeat(np.linspace(30.0, 33.0, 300), 10)
        return pd.Series(coarse, index=idx, name='SWC')

    def test_upsampled_data_is_not_mass_rejected(self):
        s = self._upsampled()
        ham = _hampel(s)
        ham.calc(repeat=False)
        flag = ham.get_flag()
        rejected = (flag == 2).sum()
        # Nothing in this series is an outlier: it is a clean ramp, just coarse.
        self.assertEqual(rejected, 0)
        # ... and the undecided records are reported rather than silently dropped.
        self.assertGreater(ham._n_degenerate_scale, 0)

    def test_a_real_spike_in_normal_data_is_still_caught(self):
        """The guard must not disarm the filter where the scale is well defined."""
        rng = np.random.default_rng(42)
        idx = pd.date_range('2020-05-01', periods=3000, freq='1min', name='TIMESTAMP_MIDDLE')
        s = pd.Series(20 + rng.normal(0, 0.05, len(idx)), index=idx, name='SWC')
        s.iloc[1500] = 45.0  # unmistakable spike
        ham = _hampel(s)
        ham.calc(repeat=False)
        flag = ham.get_flag()
        self.assertEqual(flag.iloc[1500], 2)
        self.assertLess((flag == 2).sum(), 20)


class TestGapSpanningDifferences(unittest.TestCase):
    """Missing records are dropped before differencing, so without care the two
    records flanking a gap are compared across it and look like spikes."""

    def _with_gap(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range('2020-05-01', periods=2000, freq='1min', name='TIMESTAMP_MIDDLE')
        s = pd.Series(20 + rng.normal(0, 0.05, len(idx)), index=idx, name='SWC')
        # A day-long gap, and a genuine level shift across it (the soil wetted up
        # while the logger was down) - not a spike, and not judgeable from the data.
        s.iloc[1000:1400] = np.nan
        s.iloc[1400:] += 5.0
        return s

    def test_clustered_spikes_are_still_all_removed(self):
        """Iterating must keep working: the gap mask comes from the INPUT data, so
        a value removed in an earlier iteration must not make its neighbours
        untestable - otherwise a cluster of spikes shelters itself after pass one."""
        rng = np.random.default_rng(11)
        idx = pd.date_range('2020-05-01', periods=3000, freq='1min', name='TIMESTAMP_MIDDLE')
        s = pd.Series(20 + rng.normal(0, 0.05, len(idx)), index=idx, name='SWC')
        for offset, value in [(0, 44.0), (1, 46.0), (2, 45.0)]:
            s.iloc[1500 + offset] = value
        ham = _hampel(s)
        ham.calc(repeat=True)
        flag = ham.get_flag()
        self.assertEqual(list(flag.iloc[1500:1503]), [2, 2, 2])

    def test_records_flanking_a_gap_are_not_flagged(self):
        s = self._with_gap()
        ham = _hampel(s)
        ham.calc(repeat=False)
        flag = ham.get_flag()
        edge_before = flag.iloc[999]
        edge_after = flag.iloc[1400]
        self.assertNotEqual(edge_before, 2)
        self.assertNotEqual(edge_after, 2)


class TestNonFixedFrequencyIndex(unittest.TestCase):
    """A monthly or yearly index carries a non-fixed frequency offset, which has no
    constant duration: reading its length in nanoseconds raises. The step must come
    from the timestamps instead, otherwise the whole detector goes down on data that
    the class documents no restriction against."""

    @staticmethod
    def _monthly():
        rng = np.random.default_rng(3)
        idx = pd.date_range('2000-01-01', periods=120, freq='MS', name='TIMESTAMP_MIDDLE')
        s = pd.Series(10 + rng.normal(0, 0.5, len(idx)), index=idx, name='x')
        s.iloc[60] = 40.0  # unmistakable spike
        return s

    def test_monthly_index_runs_and_finds_the_spike(self):
        s = self._monthly()
        ham = _hampel(s, window_length=12)
        ham.calc(repeat=False)
        flag = ham.get_flag()
        self.assertEqual(flag.iloc[60], 2)
        # The regular monthly spacing must not be mistaken for gaps, which would
        # leave every record untestable.
        self.assertFalse(ham._untestable.iloc[1:-1].any())

    def test_yearly_index_runs(self):
        rng = np.random.default_rng(5)
        idx = pd.date_range('2000-01-01', periods=40, freq='YS', name='TIMESTAMP_MIDDLE')
        s = pd.Series(10 + rng.normal(0, 0.5, len(idx)), index=idx, name='x')
        s.iloc[20] = 40.0
        ham = _hampel(s, window_length=8)
        ham.calc(repeat=False)
        self.assertEqual(ham.get_flag().iloc[20], 2)


if __name__ == '__main__':
    unittest.main()
