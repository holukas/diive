"""Speedups in plotting paths that must not change the output (finding L296)."""
import subprocess
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TestAggregatedAsHiresImport(unittest.TestCase):

    def test_no_gapfilling_import_without_interpolation(self):
        # The library callers never interpolate, so the ML stack behind
        # diive.gapfilling must not be imported (0.7 s on the first call).
        code = (
            "import sys\n"
            "import numpy as np, pandas as pd\n"
            "from diive.core.dfun.frames import aggregated_as_hires\n"
            "idx = pd.date_range('2021-01-01', periods=480, freq='30min')\n"
            "s = pd.Series(np.arange(480.0), index=idx, name='x')\n"
            "out = aggregated_as_hires(aggregate_series=s, hires_timestamp=idx)\n"
            "assert len(out) == 480\n"
            "print('diive.gapfilling' in sys.modules)\n")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")

    def test_interpolation_still_fills_missing_days(self):
        from diive.core.dfun.frames import aggregated_as_hires
        idx = pd.date_range('2021-01-01', periods=48 * 5, freq='30min')
        s = pd.Series(np.repeat([1.0, 2.0, np.nan, 4.0, 5.0], 48), index=idx, name='x')
        out = aggregated_as_hires(aggregate_series=s, hires_timestamp=idx,
                                  interpolate_missing_vals=True, interpolation_lim=1)
        self.assertTrue((out[idx.date == pd.Timestamp('2021-01-03').date()] == 3.0).all())


class TestDaytimeNighttimePreview(unittest.TestCase):

    def test_marker_lines_hold_only_their_records(self):
        # Each panel draws its column without NaN rows: the same markers.
        from diive.core.base.flagbase import FlagBase
        idx = pd.date_range('2021-07-01', periods=48 * 4, freq='30min')
        rng = np.random.default_rng(1)
        series = pd.Series(rng.normal(10, 1, len(idx)), index=idx, name='TA')
        daytime = pd.Series(((idx.hour >= 6) & (idx.hour < 18)).astype(int), index=idx)
        quality = pd.Series(np.where(rng.random(len(idx)) < 0.05, 2, 0), index=idx)
        try:
            from types import SimpleNamespace
            holder = SimpleNamespace()  # the method keeps the figure on its instance
            FlagBase.plot_outlier_daytime_nighttime(holder, series=series, flag_daytime=daytime,
                                                    flag_quality=quality)
            fig = holder._fig
            lines = [line for ax in fig.axes for line in ax.lines
                     if line.get_marker() not in (None, 'None', '') and len(line.get_ydata())]
            self.assertGreater(len(lines), 0)
            for line in lines:
                y = np.asarray(line.get_ydata(), dtype=float)
                self.assertFalse(np.isnan(y).any())
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
