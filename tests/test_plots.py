import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
from diive.core.plotting.histogram import HistogramPlot
from diive.core.plotting.styles.format import FormatStyle


class TestPlots(unittest.TestCase):

    def test_histogram(self):
        data_df, metadata_df = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        series = data_df['FC'].copy()

        hist = HistogramPlot(series=series, method='n_bins', n_bins=20)
        hist.plot(format_style=FormatStyle(xlabel='flux'), highlight_peak=True, show_zscores=True, show_info=True)

        edges = hist.edges
        counts = hist.counts
        self.assertEqual(edges[0], -46.2179)
        self.assertEqual(edges.mean(), -2.679900000000003)
        self.assertEqual(counts[5], 58)

        from matplotlib.axes._axes import Axes
        self.assertEqual(type(hist.get_ax()), Axes)
        from matplotlib.figure import Figure
        self.assertEqual(type(hist.get_fig()), Figure)

    def test_scatter_new_params(self):
        # markersize / alpha / vmin / vmax are honored on a caller-supplied ax.
        import pandas as pd
        from diive.core.plotting.scatter import ScatterXY
        idx = pd.date_range("2021-01-01", periods=200, freq="30min")
        x = pd.Series(range(200), index=idx, name="x", dtype=float)
        y = pd.Series([v * 2.0 for v in range(200)], index=idx, name="y")
        z = pd.Series([v % 10 for v in range(200)], index=idx, name="z", dtype=float)
        fig, ax = plt.subplots()
        ScatterXY(x=x, y=y, z=z).plot(ax=ax, markersize=12, alpha=0.4, vmin=2, vmax=8)
        coll = ax.collections[0]
        self.assertAlmostEqual(coll.get_sizes()[0], 12)
        self.assertAlmostEqual(coll.get_alpha(), 0.4)
        self.assertEqual(coll.norm.vmin, 2)
        self.assertEqual(coll.norm.vmax, 8)
        plt.close(fig)

    def test_timeseries_title_and_markersize(self):
        # On a caller ax, an explicit title is honored and marker size applied.
        import pandas as pd
        from diive.core.plotting.timeseries import TimeSeries
        idx = pd.date_range("2021-01-01", periods=50, freq="30min")
        s = pd.Series(range(50), index=idx, name="ser", dtype=float)
        fig, ax = plt.subplots()
        TimeSeries(s).plot(ax=ax, format_style=FormatStyle(title="My Title"), marker=True, markersize=7)
        self.assertEqual(ax.get_title(), "My Title")
        line = next(l for l in ax.get_lines() if l.get_markersize() > 0)
        self.assertAlmostEqual(line.get_markersize(), 7)
        plt.close(fig)

    def test_dielcycle_legend_loc(self):
        import pandas as pd
        from diive.core.plotting.dielcycle import DielCycle
        idx = pd.date_range("2021-01-01", periods=48 * 60, freq="30min")
        s = pd.Series([i % 48 for i in range(len(idx))], index=idx, name="ser", dtype=float)
        fig, ax = plt.subplots()
        DielCycle(s).plot(ax=ax, format_style=FormatStyle(legend_loc="upper right"))
        self.assertIsNotNone(ax.get_legend())
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
