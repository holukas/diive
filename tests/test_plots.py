import unittest

from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
from diive.core.plotting.histogram import HistogramPlot


class TestPlots(unittest.TestCase):

    def test_histogram(self):
        data_df, metadata_df = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        series = data_df['FC'].copy()

        hist = HistogramPlot(
            s=series,
            method='n_bins',
            n_bins=20,
            xlabel='flux',
            highlight_peak=True,
            show_zscores=True,
            show_info=True
        )
        hist.plot()

        edges = hist.edges
        counts = hist.counts
        self.assertEqual(edges[0], -46.2179)
        self.assertEqual(edges.mean(), -2.679900000000003)
        self.assertEqual(counts[5], 58)

        from matplotlib.axes._axes import Axes
        self.assertEqual(type(hist.get_ax()), Axes)
        from matplotlib.figure import Figure
        self.assertEqual(type(hist.get_fig()), Figure)


if __name__ == '__main__':
    unittest.main()
