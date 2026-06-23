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

    def test_windrose(self):
        import numpy as np
        import pandas as pd
        from diive.core.plotting.windrose import WindRosePlot, _COMPASS_16

        # Build a deterministic dataset: each sector's value equals its index, so
        # the per-sector aggregation is exactly predictable.
        idx = pd.date_range("2021-01-01", periods=8 * 50, freq="30min")
        n_sectors = 8
        sector_width = 360.0 / n_sectors
        sec = np.arange(len(idx)) % n_sectors
        # Place each direction at its sector centre (0, 45, 90, ... degrees).
        wd = pd.Series(sec * sector_width, index=idx, name="wind_dir", dtype=float)
        val = pd.Series(sec.astype(float), index=idx, name="myvar")

        rose = WindRosePlot(series=val, wind_dir=wd, agg='mean', n_sectors=n_sectors)

        # Compass labels and per-sector means.
        self.assertEqual(list(rose.results.index),
                         ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        self.assertAlmostEqual(rose.results.loc['N', 'MEAN'], 0.0)
        self.assertAlmostEqual(rose.results.loc['SE', 'MEAN'], 3.0)
        self.assertEqual(int(rose.results.loc['N', 'N_VALS']), 50)
        # Sum aggregate: sector index 4 ('S') has value 4 over 50 records.
        self.assertAlmostEqual(rose.results.loc['S', 'SUM'], 200.0)

        # North-sector folding: 360 deg must fall in the same sector as 0 deg.
        wd2 = wd.copy()
        wd2.iloc[0] = 360.0
        rose2 = WindRosePlot(series=val, wind_dir=wd2, agg='mean', n_sectors=n_sectors)
        self.assertEqual(int(rose2.results.loc['N', 'N_VALS']),
                         int(rose.results.loc['N', 'N_VALS']))

        # Plot returns a polar axes and draws one bar per sector.
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        out = rose.plot(ax=ax, cmap='viridis')
        from matplotlib.projections.polar import PolarAxes
        self.assertIsInstance(out, PolarAxes)
        self.assertEqual(len(ax.patches), n_sectors)
        plt.close(fig)

        # Bars are anchored at the zero line: with values spanning negative and
        # positive, each bar spans [min(v, 0), max(v, 0)] — not from a global hub.
        val_signed = pd.Series((sec - 3).astype(float), index=idx, name="myvar")  # -3..4
        rose3 = WindRosePlot(series=val_signed, wind_dir=wd, agg='mean', n_sectors=n_sectors)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        rose3.plot(ax=ax, cmap='RdBu_r')
        means = rose3.results['MEAN'].to_numpy()
        for patch, v in zip(ax.patches, means):
            bottom = patch.get_y()
            top = bottom + patch.get_height()
            self.assertAlmostEqual(bottom, min(v, 0.0))
            self.assertAlmostEqual(top, max(v, 0.0))
        plt.close(fig)

        # Optional z colour variable: bar length from `series`, colour from `z`.
        # Each sector's z value equals 10 + its index, aggregated by mean.
        zvar = pd.Series((sec + 10).astype(float), index=idx, name="ztemp")
        rose_z = WindRosePlot(series=val, wind_dir=wd, agg='mean', n_sectors=n_sectors,
                              z=zvar, z_agg='mean')
        self.assertIn('Z', rose_z.results.columns)
        self.assertAlmostEqual(rose_z.results.loc['N', 'Z'], 10.0)
        self.assertAlmostEqual(rose_z.results.loc['NW', 'Z'], 17.0)
        # Bar lengths still track the main variable, unchanged by z.
        self.assertAlmostEqual(rose_z.results.loc['SE', 'MEAN'], 3.0)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        rose_z.plot(ax=ax, cmap='plasma')
        # Colorbar maps the z range (10..17), not the bar-value range (0..7).
        cb_ax = ax.figure.axes[-1]
        self.assertEqual(cb_ax.get_ylabel(), 'mean ztemp')
        plt.close(fig)

        # Many sectors: per-sector labels would collide, so a fixed ring of 16
        # compass bearings is shown instead of one degree label per sector.
        wd_many = pd.Series(np.linspace(0, 359, len(idx)), index=idx, name="wd")
        rose_many = WindRosePlot(series=val, wind_dir=wd_many, agg='mean', n_sectors=64)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        rose_many.plot(ax=ax)
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], _COMPASS_16)
        ax.clear()
        rose_many.plot(ax=ax, max_sector_labels=8)
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], _COMPASS_16[::2])
        plt.close(fig)

        # Colorbar decimals: integer ticks -> 0 decimals, fractional -> as needed.
        self.assertEqual(WindRosePlot._auto_decimals([280, 282, 284]), 0)
        self.assertEqual(WindRosePlot._auto_decimals([10.0, 12.5, 15.0]), 1)
        self.assertEqual(WindRosePlot._auto_decimals([0.0, 0.005, 0.01]), 3)
        self.assertEqual(WindRosePlot._auto_decimals([float('nan')]), 0)

        # Integer-valued colorbar must render without ".0" and stay that way after
        # a draw (a colorbar resets its axis formatter on every draw).
        val_big = pd.Series((sec * 5).astype(float), index=idx, name="myvar")  # 0,5,..35
        rose_cb = WindRosePlot(series=val_big, wind_dir=wd, agg='mean', n_sectors=n_sectors)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        rose_cb.plot(ax=ax, cmap='RdBu_r')
        fig.canvas.draw()
        cb_labels = [t.get_text() for t in ax.figure.axes[-1].get_yticklabels() if t.get_text()]
        self.assertTrue(cb_labels)
        self.assertTrue(all('.' not in lbl for lbl in cb_labels), cb_labels)
        plt.close(fig)

        # show_colorbar=False draws no colorbar axes (the radial scale remains).
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        rose.plot(ax=ax, show_colorbar=False)
        self.assertEqual(len(fig.axes), 1)
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
