import unittest

import numpy as np
import pandas as pd


class TestTimeLagAnalysis(unittest.TestCase):
    """Time-lag histogram analysis of EddyPro *_TLAG_ACTUAL columns."""

    @staticmethod
    def _tlag_df(lags, n=200):
        """Dataframe with one CO2_TLAG_ACTUAL column drawn from *lags*."""
        ix = pd.date_range('2024-06-01 00:15', periods=n, freq='30min')
        rng = np.random.default_rng(42)
        return pd.DataFrame({'CO2_TLAG_ACTUAL': rng.choice(lags, size=n)}, index=ix)

    def test_few_distinct_lags_survive_default_fringe_trimming(self):
        """Default fringe trimming must not empty the histogram.

        Six distinct lag values give five bins, far fewer than the 15 bins the
        default ignore_fringe_bins=[5, 10] removes.
        """
        from diive.flux.lowres.timelag_analysis import TimeLagAnalysis

        df = self._tlag_df([0.30, 0.35, 0.40, 0.45, 0.50, 0.55])

        analysis = TimeLagAnalysis(df=df)
        res = analysis.analyze_gas('CO2')
        self.assertEqual(len(res['histogram_results']), 5)
        self.assertIn(res['peak'], [0.30, 0.35, 0.40, 0.45, 0.50])
        self.assertLessEqual(res['peak_min'], res['peak_max'])

        # The batch helper returns the gas instead of aborting
        batch = TimeLagAnalysis(df=df).analyze_all_gases()
        self.assertEqual(list(batch.keys()), ['CO2'])

    def test_single_distinct_lag_is_reported_not_a_bare_indexerror(self):
        """Fewer than two distinct lags cannot form a bin.

        That must be a ValueError with an explanation, so the batch helpers can
        warn and continue as their docstrings promise.
        """
        from diive.flux.lowres.timelag_analysis import TimeLagAnalysis

        df = self._tlag_df([0.30])
        with self.assertRaises(ValueError):
            TimeLagAnalysis(df=df).analyze_gas('CO2')
        self.assertEqual(TimeLagAnalysis(df=df).analyze_all_gases(), {})

    def test_bin_range_excluding_every_lag_keeps_all_bins(self):
        """A lag range that no bin falls into must not empty the histogram.

        histogram_startbin/histogram_endbin are lag seconds, so a range above
        every measured lag used to leave zero bins and crash detect_peak_range
        on the empty array.
        """
        from diive.flux.lowres.timelag_analysis import TimeLagAnalysis

        lags = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
        df = self._tlag_df(lags)

        kwargs = dict(histogram_startbin=5, histogram_endbin=10)
        res = TimeLagAnalysis(df=df, **kwargs).analyze_gas('CO2')
        self.assertEqual(len(res['histogram_results']), 5)
        self.assertIn(res['peak'], lags)

        # The batch helper returns the gas instead of dropping it
        batch = TimeLagAnalysis(df=df, **kwargs).analyze_all_gases()
        self.assertEqual(list(batch.keys()), ['CO2'])

    def test_bin_range_keeping_some_bins_still_trims(self):
        """The range must keep doing its job when bins do fall inside."""
        from diive.flux.lowres.timelag_analysis import TimeLagAnalysis

        df = self._tlag_df([0.30, 0.35, 0.40, 0.45, 0.50, 0.55])

        analysis = TimeLagAnalysis(df=df, histogram_startbin=0.40, histogram_endbin=10)
        res = analysis.analyze_gas('CO2')
        self.assertEqual(list(res['histogram_results']['BIN_START_INCL']), [0.40, 0.45, 0.50])


if __name__ == '__main__':
    unittest.main()
