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

    def test_scatter_same_variable_in_two_roles(self):
        # A variable may fill more than one role (e.g. colour points by x, or
        # x == y): duplicate names must not collapse xy_df columns into a frame.
        import pandas as pd
        from diive.core.plotting.scatter import ScatterXY
        idx = pd.date_range("2021-01-01", periods=200, freq="30min")
        x = pd.Series(range(200), index=idx, name="Tair", dtype=float)
        y = pd.Series([v * 2.0 for v in range(200)], index=idx, name="NEE")
        # z shares x's name; raw and binned paths must both render.
        fig, ax = plt.subplots()
        ScatterXY(x=x, y=y, z=x.copy()).plot(ax=ax, show_colorbar=True)
        self.assertTrue(ax.collections)
        self.assertEqual(ax.get_xlabel(), "Tair")  # display name preserved
        plt.close(fig)
        fig, ax = plt.subplots()
        ScatterXY(x=x, y=y, z=x.copy(), nbins=10, binagg="median").plot(ax=ax)
        self.assertTrue(ax.collections)
        plt.close(fig)

    @staticmethod
    def _scatter_xy(n=500):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(3)
        idx = pd.date_range("2021-01-01", periods=n, freq="30min")
        x = pd.Series(rng.normal(10, 5, n), index=idx, name="TA")
        y = pd.Series(0.5 * x.to_numpy() + rng.normal(0, 2, n), index=idx, name="NEE")
        return x, y

    @staticmethod
    def _record_draw_calls(fig):
        """Draw `fig` and return the number of points each renderer call drew."""
        import types
        renderer = fig.canvas.get_renderer()
        calls = {"markers": [], "collection": []}
        draw_markers, draw_collection = renderer.draw_markers, renderer.draw_path_collection

        def markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
            calls["markers"].append(len(path.vertices))
            return draw_markers(gc, marker_path, marker_trans, path, trans, rgbFace)

        def collection(self, gc, master, paths, transforms, offsets, *args):
            calls["collection"].append(len(offsets))
            return draw_collection(gc, master, paths, transforms, offsets, *args)

        # RendererAgg binds these per instance; the canvas reuses this one.
        renderer.draw_markers = types.MethodType(markers, renderer)
        renderer.draw_path_collection = types.MethodType(collection, renderer)
        fig.canvas.draw()
        return calls

    def test_scatter_uncoloured_points_draw_as_one_stamped_marker(self):
        # Hollow markers go through draw_markers (one marker rasterized once,
        # stamped at every point) rather than stroking each marker on its own,
        # and the artist is still the PathCollection that ax.scatter returned.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import PathCollection
        from matplotlib.colors import to_rgba
        from matplotlib.figure import Figure
        from diive.core.plotting.scatter import ScatterXY
        x, y = self._scatter_xy()
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ScatterXY(x=x, y=y).plot(ax=ax, markersize=30, alpha=0.5)
        coll = ax.collections[0]
        self.assertIsInstance(coll, PathCollection)
        calls = self._record_draw_calls(fig)
        self.assertIn(len(x), calls["markers"])
        self.assertNotIn(len(x), calls["collection"])
        # The face stays 'none' outside the draw, as ax.scatter set it.
        self.assertEqual(len(coll.get_facecolor()), 0)
        self.assertEqual(tuple(coll.get_edgecolor()[0]), to_rgba("#607D8B", 0.5))
        self.assertAlmostEqual(coll.get_sizes()[0], 30)
        self.assertEqual(coll.get_label(), "NEE")
        # The legend entry is a scatter handle of the same marker size.
        handle = ax.get_legend().legend_handles[0]
        self.assertIsInstance(handle, PathCollection)
        self.assertAlmostEqual(handle.get_sizes()[0], 30)

        # A colour-coded scatter has one colour per point, so it keeps the
        # per-point path.
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ScatterXY(x=x, y=y, z=x.copy()).plot(ax=ax)
        calls = self._record_draw_calls(fig)
        self.assertIn(len(x), calls["collection"])
        self.assertNotIn(len(x), calls["markers"])

    def test_scatter_fast_path_differs_only_by_the_pixel_snap(self):
        # draw_markers centres each marker on the nearest pixel. Re-rendering
        # the slow way with every marker moved to that pixel centre must give
        # the same image, pixel for pixel: the snap is the only difference.
        import types
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.transforms import IdentityTransform
        from diive.core.plotting import scatter as scmod
        x, y = self._scatter_xy()

        def render(snap_reference):
            fig = Figure(figsize=(6, 4), dpi=100)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            scmod.ScatterXY(x=x, y=y, nbins=8).plot(ax=ax)
            if snap_reference:
                renderer = fig.canvas.get_renderer()
                fast = renderer.draw_markers
                height = renderer.height

                def reference(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
                    # Only the hollow scatter (points and legend handle) passes
                    # a fully transparent face; ticks and the binned line don't.
                    if rgbFace is None or rgbFace[3] != 0:
                        return fast(gc, marker_path, marker_trans, path, trans, rgbFace)
                    pts = trans.transform(path.vertices)
                    # Agg: pixel column floor(x + 0.5), row floor(h - y + 0.5),
                    # marker drawn at that pixel's centre.
                    px = np.floor(pts[:, 0] + 0.5) + 0.5
                    py = height - (np.floor(height - pts[:, 1] + 0.5) + 0.5)
                    self.draw_path_collection(
                        gc, marker_trans, [marker_path], np.zeros((0, 3, 3)),
                        np.column_stack([px, py]), IdentityTransform(),
                        np.zeros((0, 4)), [gc.get_rgb()], [gc.get_linewidth()],
                        [gc.get_dashes()], [gc.get_antialiased()], [None], "screen")

                renderer.draw_markers = types.MethodType(reference, renderer)
            fig.canvas.draw()
            return np.asarray(fig.canvas.buffer_rgba()).copy()

        np.testing.assert_array_equal(render(False), render(True))

    @staticmethod
    def _legend_bounds(build, plain, zoom=None):
        """Draw a figure made by `build(ax)` and return its legend's bounds.

        With `plain`, the legend is turned back into a matplotlib `Legend`
        first, so its 'best' search runs matplotlib's own code.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.legend import Legend
        fig = Figure(figsize=(6, 4), dpi=100)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        build(ax)
        legend = ax.get_legend()
        if plain:
            legend.__class__ = Legend
        if zoom:
            ax.set_xlim(*zoom)
        fig.canvas.draw()
        return legend.get_window_extent().bounds

    def test_scatter_legend_best_lands_where_matplotlib_puts_it(self):
        # The 'best' search of a large scatter tests the points as one array.
        # It must pick the same place matplotlib picks, with and without the
        # binned line, colour-coded, and after a zoom moves the data.
        from diive.core.plotting import plotfuncs as pf
        from diive.core.plotting.scatter import ScatterXY
        n = pf._LEGEND_BEST_MAX_POINTS + 5000
        x, y = self._scatter_xy(n)
        cases = [
            (lambda ax: ScatterXY(x=x, y=y).plot(ax=ax), None),
            (lambda ax: ScatterXY(x=x, y=y, nbins=10).plot(ax=ax), None),
            (lambda ax: ScatterXY(x=x, y=y, nbins=10, binagg='mean').plot(ax=ax), None),
            (lambda ax: ScatterXY(x=x, y=y, z=x.copy()).plot(ax=ax, show_colorbar=False), None),
            (lambda ax: ScatterXY(x=x, y=y).plot(ax=ax), (10, 30)),
            (lambda ax: ScatterXY(x=x.iloc[:300], y=y.iloc[:300]).plot(ax=ax), None),
        ]
        seen = set()
        for build, zoom in cases:
            ours = self._legend_bounds(build, plain=False, zoom=zoom)
            theirs = self._legend_bounds(build, plain=True, zoom=zoom)
            self.assertEqual(ours, theirs)
            seen.add(ours)
        # The cases exercise more than one location.
        self.assertGreater(len(seen), 1)

    def test_array_legend_data_mirrors_matplotlib(self):
        # _ArrayOffsetsLegend copies matplotlib's private _auto_legend_data
        # for large collections. Compare the two on an axes holding every kind
        # of artist the search looks at, so a change in matplotlib shows here.
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.legend import Legend
        from matplotlib.patches import Circle
        from diive.core.plotting import plotfuncs as pf
        rng = np.random.default_rng(1)
        n = pf._LEGEND_BEST_MAX_POINTS + 1
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.scatter(rng.random(n), rng.random(n), label='points')
        ax.scatter([0.2, np.nan], [0.3, 0.4])
        ax.plot([0, 1], [0, 1], label='line')
        ax.bar([0.5], [0.5], width=0.1)
        ax.add_patch(Circle((0.3, 0.7), 0.1))
        ax.fill_between([0, 0.5, 1], [0, 0.1, 0], [0.2, 0.3, 0.2])
        ax.text(0.8, 0.2, 'note')
        pf.default_legend(ax=ax)
        legend = ax.get_legend()
        self.assertIsInstance(legend, pf._ArrayOffsetsLegend)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        bboxes, lines, offsets = legend._auto_legend_data(renderer)
        ref_bboxes, ref_lines, ref_offsets = Legend._auto_legend_data(legend, renderer)
        self.assertEqual([b.bounds for b in bboxes], [b.bounds for b in ref_bboxes])
        self.assertEqual(len(lines), len(ref_lines))
        for path, ref in zip(lines, ref_lines):
            np.testing.assert_array_equal(path.vertices, ref.vertices)
        self.assertIsInstance(offsets, np.ndarray)
        np.testing.assert_array_equal(offsets, np.asarray(ref_offsets))

    def test_array_legend_data_is_used_only_for_large_collections(self):
        import types
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from diive.core.plotting import plotfuncs as pf

        def calls_for(n):
            fig = Figure()
            FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            ax.scatter(np.linspace(0, 1, n), np.linspace(0, 1, n), label='points')
            pf.default_legend(ax=ax)
            legend = ax.get_legend()
            calls = []
            arrays = legend._auto_legend_data_arrays

            def spy(self, renderer):
                calls.append(1)
                return arrays(renderer)

            legend._auto_legend_data_arrays = types.MethodType(spy, legend)
            fig.canvas.draw()
            return len(calls)

        # Matplotlib still asks the legend for its search data (if it stops,
        # the override is dead code and large scatters are slow again).
        self.assertGreater(calls_for(pf._LEGEND_BEST_MAX_POINTS + 1), 0)
        self.assertEqual(calls_for(pf._LEGEND_BEST_MAX_POINTS), 0)

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

    def test_compound_extremes(self):
        import numpy as np
        import pandas as pd
        from diive.analysis.compoundextremes import CompoundExtremes, CAT_NONE, CAT_COMPOUND
        from diive.core.plotting.compoundextremes import CompoundExtremesPlot

        # Deterministic synthetic series with one guaranteed compound month.
        idx = pd.date_range('2010-01-01', periods=120, freq='MS', name='TIMESTAMP_MIDDLE')
        rng = np.random.default_rng(0)
        v1 = pd.Series(rng.normal(0, 1, 120), index=idx, name='VPD')
        v2 = pd.Series(rng.normal(0, 1, 120), index=idx, name='SWC')
        v1.iloc[60] += 10.0
        v2.iloc[60] -= 10.0  # same month -> compound
        ce = CompoundExtremes(var1=v1, var2=v2, agg='monthly', threshold=2.0,
                              var1_extreme='high', var2_extreme='low',
                              standardize_by='record', var1_label='Air', var2_label='Soil')

        # Build the plot straight from the analysis instance.
        cep = CompoundExtremesPlot.from_compound_extremes(ce)
        # Threshold lines are signed by each variable's extreme direction.
        self.assertEqual(cep.threshold_x, 2.0)
        self.assertEqual(cep.threshold_y, -2.0)

        fig, ax = plt.subplots()
        cep.plot(ax=ax)
        from matplotlib.axes._axes import Axes
        self.assertIsInstance(ax, Axes)
        # Quadrant lines drawn (one vertical + one horizontal).
        self.assertIn(2.0, [l.get_xdata()[0] for l in ax.get_lines()])
        self.assertIn(-2.0, [l.get_ydata()[0] for l in ax.get_lines()])
        # One scatter collection per present category; compound is present here.
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn('Compound', labels)
        self.assertIn('None', labels)
        plt.close(fig)

        # Custom styling + pre-classified data path (arbitrary category labels).
        x = pd.Series([0.0, 3.0, 0.0, 3.0], name='x z')
        y = pd.Series([0.0, 0.0, -3.0, -3.0], name='y z')
        cat = pd.Series(['normal', 'A', 'B', 'both'])
        styles = {'both': {'color': '#D32F2F', 'marker': 'D', 'label': 'Compound'}}
        plot = CompoundExtremesPlot(x=x, y=y, category=cat, category_styles=styles,
                                    category_order=['normal', 'A', 'B', 'both'],
                                    threshold_x=2.0, threshold_y=-2.0)
        self.assertEqual(plot.category_styles['both']['label'], 'Compound')
        fig, ax = plt.subplots()
        plot.plot(ax=ax, annotate=False, legend=True)
        self.assertEqual(ax.get_xlabel(), 'x z')
        self.assertEqual(ax.get_ylabel(), 'y z')
        plt.close(fig)

        # Threshold lines can be disabled.
        fig, ax = plt.subplots()
        CompoundExtremesPlot(x=x, y=y, category=cat,
                             threshold_x=None, threshold_y=None).plot(ax=ax)
        self.assertEqual(len(ax.get_lines()), 0)
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

    def test_dielcycle_leaves_linked_neighbours_labelled(self):
        """Drawing a diel cycle must not hide another panel's shared-x tick labels.

        It used to plot through pandas, which hides the x tick labels of every
        non-bottom axes in the figure that shares its x-axis (and swaps in a
        minor locator), blanking the dates of the GUI Overview's time series.
        """
        import matplotlib.ticker as mticker
        import pandas as pd
        from diive.core.plotting.dielcycle import DielCycle
        idx = pd.date_range("2021-01-01", periods=48 * 60, freq="30min")
        s = pd.Series([i % 48 for i in range(len(idx))], index=idx, name="ser", dtype=float)
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        top = fig.add_subplot(gs[0, :])
        top.plot(idx, s.to_numpy())
        fig.add_subplot(gs[1, 0], sharex=top)
        DielCycle(s).plot(ax=fig.add_subplot(gs[1, 1]))
        fig.canvas.draw()
        self.assertTrue(all(t.label1.get_visible() for t in top.xaxis.get_major_ticks()))
        self.assertIsInstance(top.xaxis.get_minor_locator(), mticker.NullLocator)
        plt.close(fig)

    def test_quickplot_keeps_same_named_series(self):
        """A list of same-named series must give one panel each, not one panel total.

        Correction routines pass several stages of one variable (raw, corrected),
        which share the variable name. Keying them by name dropped all but the last
        and left the survivor labelled with the dropped series' name.
        """
        import numpy as np
        import pandas as pd
        from diive.core.plotting.plotfuncs import quickplot
        idx = pd.date_range("2020-01-01", periods=3, freq="30min")
        a = pd.Series([1.0, 2.0, 3.0], index=idx, name="X")
        b = pd.Series([4.0, 5.0, 6.0], index=idx, name="X")
        c = pd.Series([7.0, 8.0, 9.0], index=idx, name="Y")

        quickplot([a, b, c], subplots=True, showplot=False, title="dup")
        fig = plt.gcf()
        self.assertEqual(len(fig.axes), 3)
        # Every series keeps its own data, in the order it was passed.
        drawn = [ax.lines[0].get_ydata() for ax in fig.axes]
        for expected, actual in zip([a, b, c], drawn):
            np.testing.assert_allclose(actual, expected.values)
        plt.close(fig)


class TestDefaultFormatLabels(unittest.TestCase):
    """default_format used to write the string 'False' into the axis labels,
    because False is its 'no label' default and was passed straight to matplotlib."""

    def test_no_label_means_empty_label(self):
        from diive.core.plotting.plotfuncs import default_format
        fig, ax = plt.subplots()
        default_format(ax=ax)
        self.assertEqual(ax.get_xlabel(), '')
        self.assertEqual(ax.get_ylabel(), '')
        plt.close(fig)

    def test_labels_still_work_when_given(self):
        from diive.core.plotting.plotfuncs import default_format
        fig, ax = plt.subplots()
        default_format(ax=ax, ax_xlabel_txt='time', ax_ylabel_txt='SWC', txt_ylabel_units='[%]')
        self.assertEqual(ax.get_xlabel(), 'time')
        self.assertEqual(ax.get_ylabel(), 'SWC  [%]')
        plt.close(fig)


class TestPlotfuncsHelpers(unittest.TestCase):
    """Helpers with no coverage, which is how a live crash went unnoticed.

    make_patch_spines_invisible called ax.spines.to_numpy()() and raised
    AttributeError on every call. Both of its call sites are real
    (heatmap_base's black-and-white render, make_secondary_yaxis), but
    neither is exercised by another test.
    """

    def test_make_patch_spines_invisible(self):
        from diive.core.plotting.plotfuncs import make_patch_spines_invisible
        fig, ax = plt.subplots()
        make_patch_spines_invisible(ax)
        self.assertFalse(ax.patch.get_visible())
        self.assertTrue(ax.get_frame_on())
        self.assertEqual([sp.get_visible() for sp in ax.spines.values()], [False] * 4)
        plt.close(fig)

    def test_make_secondary_yaxis_uses_the_same_helper(self):
        from diive.core.plotting.plotfuncs import make_secondary_yaxis
        fig, ax = plt.subplots()
        twin = make_secondary_yaxis(ax)  # raised before the spines fix
        self.assertIsNotNone(twin)
        plt.close(fig)

    def test_adjust_color_lightness_accepts_every_colour_form(self):
        import numpy as np
        from diive.core.plotting.styles.LightTheme import adjust_color_lightness
        # The name lookup raises for anything that is not a named colour, and
        # callers pass three different forms. A hex string raises KeyError; an
        # RGBA tuple or numpy array is not hashable at all and raises TypeError.
        # RidgeLinePlot passes colormap output, i.e. the array form, so all
        # three have to fall through to "use the value as given".
        forms = {
            'named': 'red',
            'hex': '#ff0000',
            'rgba tuple': (1.0, 0.0, 0.0, 1.0),
            'numpy array': np.array([1.0, 0.0, 0.0, 1.0]),
        }
        results = {}
        for label, value in forms.items():
            with self.subTest(form=label):
                out = adjust_color_lightness(value, 0.5)
                self.assertEqual(len(out), 3)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in out))
                results[label] = out
        # All four describe the same red, so they must lighten identically.
        self.assertEqual(len(set(results.values())), 1)


class TestDecimateLine(unittest.TestCase):
    """`decimate_line` must thin a line without changing what it shows."""

    @staticmethod
    def _gappy_walk(n: int = 60_000, seed: int = 1):
        import numpy as np
        rng = np.random.default_rng(seed)
        x = np.arange(n, dtype=float) / 48.0  # half-hourly, in days
        y = np.cumsum(rng.normal(size=n)) + rng.normal(scale=5.0, size=n)
        y[rng.integers(0, n, 40)] += 80.0  # isolated spikes
        y[10_000:10_500] = np.nan          # a long gap
        y[rng.integers(0, n, 3_000)] = np.nan  # scattered single gaps
        return x, y

    def test_keeps_only_real_samples_in_order(self):
        import numpy as np
        from diive.core.plotting.plotfuncs import decimate_line
        x, y = self._gappy_walk()
        xd, yd = decimate_line(x, y, x[0], x[-1], 400)
        self.assertLess(xd.size, x.size / 3)  # gappy: every short run keeps its ends
        valid = ~np.isnan(yd)
        pos = np.searchsorted(x, xd[valid])
        np.testing.assert_array_equal(x[pos], xd[valid])
        np.testing.assert_array_equal(y[pos], yd[valid])
        self.assertTrue(np.all(np.diff(pos) > 0), msg="samples out of order or repeated")

    def test_every_column_keeps_its_extremes(self):
        import numpy as np
        from diive.core.plotting.plotfuncs import decimate_line
        x, y = self._gappy_walk()
        xmin, xmax, n_bins = x[5_000], x[40_000], 300
        xd, yd = decimate_line(x, y, xmin, xmax, n_bins)
        col = np.floor((x - xmin) / (xmax - xmin) * n_bins)
        cold = np.floor((xd - xmin) / (xmax - xmin) * n_bins)
        for c in range(n_bins):
            full = y[(col == c) & ~np.isnan(y)]
            if full.size == 0:
                continue
            kept = yd[(cold == c) & ~np.isnan(yd)]
            self.assertEqual(kept.min(), full.min(), msg=f"column {c} lost its minimum")
            self.assertEqual(kept.max(), full.max(), msg=f"column {c} lost its maximum")

    def test_breaks_exactly_where_the_record_has_a_gap(self):
        import numpy as np
        from diive.core.plotting.plotfuncs import decimate_line
        x, y = self._gappy_walk()
        xd, yd = decimate_line(x, y, x[0], x[-1], 400)
        # Between two consecutive kept samples, the output breaks (NaN) if and
        # only if the full record has a missing value between them.
        kept = np.flatnonzero(~np.isnan(yd))
        pos = np.searchsorted(x, xd[kept])
        missing = np.cumsum(np.isnan(y))
        for a, b, pa, pb in zip(kept[:-1], kept[1:], pos[:-1], pos[1:], strict=True):
            broken_out = b - a > 1
            broken_full = missing[pb - 1] - missing[pa] > 0
            self.assertEqual(broken_out, broken_full, msg=f"records {pa}..{pb}")

    def test_short_slice_is_returned_whole_with_one_sample_beyond_each_edge(self):
        import numpy as np
        from diive.core.plotting.plotfuncs import decimate_line
        x, y = self._gappy_walk()
        xd, yd = decimate_line(x, y, x[100] + 0.001, x[200] - 0.001, 1000)
        np.testing.assert_array_equal(xd, x[100:201])
        np.testing.assert_array_equal(yd, y[100:201])

    def test_drawn_line_looks_the_same(self):
        """At two columns per pixel the thinned line covers the same pixels."""
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from diive.core.plotting.plotfuncs import decimate_line
        x, y = self._gappy_walk()

        def render(xs, ys):
            fig = Figure(figsize=(8, 3), dpi=100)
            FigureCanvasAgg(fig)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.plot(xs, ys, color="black", linewidth=0.7)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(np.nanmin(y), np.nanmax(y))
            ax.axis("off")
            fig.canvas.draw()
            return np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(int)

        full = render(x, y)
        thin = render(*decimate_line(x, y, x[0], x[-1], 2 * 800))
        ink_ratio = (255 - thin).sum() / (255 - full).sum()
        self.assertGreater(ink_ratio, 0.97)
        self.assertLess(np.mean(np.abs(full - thin) > 96), 0.01)


def _synthetic_series(years: int = 3, name: str = "TA", start: str = "2019-01-01"):
    """Deterministic hourly series with an annual and a diel cycle.

    No randomness, so every aggregate below is an exact expected value rather
    than a tolerance. Hourly (not 30-min) keeps three years at ~26k points, which
    is enough for the year/month/diel groupings while staying fast.
    """
    import numpy as np
    import pandas as pd
    n = years * 365 * 24
    idx = pd.date_range(start, periods=n, freq="1h", name="TIMESTAMP_MIDDLE")
    t = np.arange(n)
    values = (10.0
              + 10.0 * np.sin(2 * np.pi * t / (24 * 365))   # annual cycle
              + 5.0 * np.sin(2 * np.pi * t / 24))           # diel cycle
    return pd.Series(values, index=idx, name=name)


class TestPlotClasses(unittest.TestCase):
    """The plot classes that no non-GUI test reached.

    `tests/test_plots.py` covered five classes (Histogram, ScatterXY, TimeSeries,
    WindRose, CompoundExtremes). Everything below was executed only incidentally
    by `tests/test_gui.py` -- `HeatmapDateTime` most conspicuously, since 16 of
    the 122 examples use it. Assertions target each class's actual contract
    (cumulative totals, panel counts, aggregation differences), not just "the
    call did not raise".
    """

    @classmethod
    def setUpClass(cls):
        cls.series = _synthetic_series()

    # --- heatmaps ---

    def test_heatmap_datetime_orientation_swaps_the_axes(self):
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        expected = {"vertical": ("Time (hours)", "Date"),
                    "horizontal": ("Date", "Time (hours)")}
        for orientation, (xlabel, ylabel) in expected.items():
            with self.subTest(orientation=orientation):
                fig, ax = plt.subplots()
                HeatmapDateTime(self.series, ax_orientation=orientation).plot(ax=ax, fig=fig)
                self.assertEqual(ax.get_xlabel(), xlabel)
                self.assertEqual(ax.get_ylabel(), ylabel)
                self.assertEqual(len(ax.collections), 1)  # one QuadMesh
                plt.close(fig)

    def test_heatmap_datetime_as_image_matches_the_mesh(self):
        """`as_image=True` draws the same cells and chrome as the default mesh.

        Same values in the same cell layout, same axis limits (the date axis in
        date numbers), same tick labels and colorbar, in both orientations.
        """
        import numpy as np
        from matplotlib.collections import QuadMesh
        from matplotlib.image import AxesImage
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        for orientation in ("vertical", "horizontal"):
            with self.subTest(orientation=orientation):
                drawn = {}
                for as_image in (False, True):
                    fig, ax = plt.subplots()
                    HeatmapDateTime(self.series, ax_orientation=orientation).plot(
                        ax=ax, fig=fig, as_image=as_image)
                    fig.canvas.draw()
                    artist = (ax.get_images() or ax.collections)[0]
                    drawn[as_image] = dict(
                        type=type(artist),
                        values=np.ma.filled(artist.get_array(), np.nan).reshape(-1),
                        xlim=ax.get_xlim(), ylim=ax.get_ylim(),
                        xticks=[t.get_text() for t in ax.get_xticklabels()],
                        yticks=[t.get_text() for t in ax.get_yticklabels()],
                        cbar=[t.get_text() for t in fig.axes[1].get_yticklabels()])
                    plt.close(fig)
                mesh, image = drawn[False], drawn[True]
                self.assertIs(mesh["type"], QuadMesh)
                self.assertIs(image["type"], AxesImage)
                np.testing.assert_array_equal(image["values"], mesh["values"])
                np.testing.assert_allclose(image["xlim"], mesh["xlim"])
                np.testing.assert_allclose(image["ylim"], mesh["ylim"])
                for key in ("xticks", "yticks", "cbar"):
                    self.assertEqual(image[key], mesh[key], msg=key)

    def test_heatmap_datetime_grid_equals_the_date_time_pivot(self):
        """The shortcut grid must be exactly the frame `pivot` builds.

        Covers a partial first and last day, gaps, a 10-min step and a
        microsecond-resolution index, in both orientations.
        """
        import numpy as np
        import pandas as pd
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        rng = np.random.default_rng(5)
        values = rng.normal(size=5000)
        values[rng.integers(0, 5000, 400)] = np.nan
        series = pd.Series(values, index=pd.date_range(
            "2021-03-27 07:10", periods=5000, freq="10min", unit="us", name="TIMESTAMP_END"))
        for orientation in ("vertical", "horizontal"):
            with self.subTest(orientation=orientation):
                hm = HeatmapDateTime(series, ax_orientation=orientation)
                ref = hm.series.rename('_values').to_frame()
                ref['DATE'] = ref.index.date
                ref['TIME'] = ref.index.time
                keys = ('DATE', 'TIME') if orientation == "vertical" else ('TIME', 'DATE')
                ref = ref.reset_index(drop=True).pivot(index=keys[0], columns=keys[1], values='_values')
                pd.testing.assert_frame_equal(hm.get_plot_data(), ref, check_exact=True)

    def test_heatmap_datetime_as_image_falls_back_to_the_mesh_on_an_uneven_grid(self):
        from matplotlib.collections import QuadMesh
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        hm = HeatmapDateTime(self.series.head(24 * 5))
        hm.x = hm.x.copy()
        hm.x[1] += 0.25  # one uneven time step: no image can hold that grid
        fig, ax = plt.subplots()
        hm.plot(ax=ax, fig=fig, as_image=True)
        self.assertEqual(ax.get_images(), [])
        self.assertIsInstance(ax.collections[0], QuadMesh)
        plt.close(fig)

    def test_heatmap_datetime_show_values_annotates_cells(self):
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        short = self.series.head(24 * 5)
        fig, ax = plt.subplots()
        HeatmapDateTime(short).plot(ax=ax, fig=fig)
        without = len(ax.texts)
        plt.close(fig)
        fig, ax = plt.subplots()
        HeatmapDateTime(short).plot(ax=ax, fig=fig, show_values=True)
        self.assertGreater(len(ax.texts), without)
        plt.close(fig)

    def test_heatmap_yearmonth_aggregation_changes_the_values(self):
        # HeatmapYearMonth shares heatmap_datetime's module with HeatmapDateTime.
        from diive.core.plotting.heatmap_datetime import HeatmapYearMonth

        def mesh_values(**kwargs):
            fig, ax = plt.subplots()
            HeatmapYearMonth(self.series, **kwargs).plot(ax=ax, fig=fig)
            arr = ax.collections[0].get_array().copy()
            plt.close(fig)
            return arr

        import numpy as np
        means, maxima = mesh_values(agg="mean"), mesh_values(agg="max")
        # Whole 12-month rows covering at least the three years present (the
        # mesh carries a trailing row of edges, so this is not exactly 3 x 12).
        self.assertEqual(means.size % 12, 0)
        self.assertGreaterEqual(means.size, 3 * 12)
        # The same cells aggregated differently: max must top mean somewhere.
        self.assertTrue(np.nanmax(maxima) > np.nanmax(means))
        # ranks= replaces the values with their rank, so the scale changes.
        ranks = mesh_values(agg="mean", ranks=True)
        self.assertFalse(np.allclose(np.asarray(means, dtype=float),
                                     np.asarray(ranks, dtype=float),
                                     equal_nan=True))

    # --- show_values cell-count guard ---

    @staticmethod
    def _datetime_heatmap_labels(series, **plot_kwargs):
        """Render a HeatmapDateTime and report the overlay's cost and console output.

        Returns ``(n_text_artists, n_cells, console_text)``. Text-artist count is
        the measurable signal for the ``show_values`` guard: the overlay writes one
        artist per cell and those artists stay on the axes.
        """
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        from diive.core.utils.console import add_console_sink, remove_console_sink

        class _Sink:
            """Mirror console collecting everything the library prints."""

            def __init__(self):
                self.lines = []

            def print(self, *args, **kwargs):
                self.lines.append(" ".join(str(a) for a in args))

            def log(self, *args, **kwargs):
                self.print(*args, **kwargs)

        sink = _Sink()
        hm = HeatmapDateTime(series)
        fig, ax = plt.subplots()
        add_console_sink(sink)
        try:
            hm.plot(ax=ax, fig=fig, **plot_kwargs)
        finally:
            remove_console_sink(sink)
        n_texts, n_cells = len(ax.texts), int(hm.z.size)
        plt.close(fig)
        return n_texts, n_cells, "\n".join(sink.lines)

    def test_show_values_labels_every_cell_below_the_cell_limit(self):
        from diive.core.plotting.heatmap_base import SHOW_VALUES_MAX_CELLS
        n_texts, n_cells, out = self._datetime_heatmap_labels(
            self.series.head(24 * 5), show_values=True)
        self.assertLess(n_cells, SHOW_VALUES_MAX_CELLS)
        self.assertEqual(n_texts, n_cells)  # one text artist per cell
        self.assertNotIn("show_values skipped", out)

    def test_show_values_skipped_above_the_cell_limit_and_says_so(self):
        # ~100 days of hourly data, just past the limit. One year of half-hourly
        # data (17 520 cells) took 6.3 s to draw and 4.9 s per later redraw.
        from diive.core.plotting.heatmap_base import SHOW_VALUES_MAX_CELLS
        n_texts, n_cells, out = self._datetime_heatmap_labels(
            self.series.head(24 * 100), show_values=True)
        self.assertGreater(n_cells, SHOW_VALUES_MAX_CELLS)
        self.assertEqual(n_texts, 0)  # no artists left behind
        self.assertIn("show_values skipped", out)  # not a silent skip
        self.assertIn(str(n_cells), out)
        self.assertIn(str(SHOW_VALUES_MAX_CELLS), out)

    def test_show_values_max_cells_override_forces_the_labels(self):
        from diive.core.plotting.heatmap_base import SHOW_VALUES_MAX_CELLS
        n_texts, n_cells, out = self._datetime_heatmap_labels(
            self.series.head(24 * 100), show_values=True, show_values_max_cells=None)
        self.assertGreater(n_cells, SHOW_VALUES_MAX_CELLS)
        self.assertEqual(n_texts, n_cells)
        self.assertNotIn("show_values skipped", out)

    def test_yearmonth_show_values_unaffected_by_the_cell_limit(self):
        # A year x month grid is 12 cells per year, far below the limit, so the
        # guard must never fire on the plot type the overlay was designed for.
        from diive.core.plotting.heatmap_base import SHOW_VALUES_MAX_CELLS
        from diive.core.plotting.heatmap_datetime import HeatmapYearMonth
        hm = HeatmapYearMonth(self.series)
        fig, ax = plt.subplots()
        hm.plot(ax=ax, fig=fig, show_values=True)
        n_texts, n_cells = len(ax.texts), int(hm.z.size)
        plt.close(fig)
        # Whole 12-month rows covering at least the three years present (the
        # TIMESTAMP_START convention pulls in one more calendar year).
        self.assertEqual(n_cells % 12, 0)
        self.assertGreaterEqual(n_cells, 3 * 12)
        self.assertLess(n_cells, SHOW_VALUES_MAX_CELLS)
        self.assertEqual(n_texts, n_cells)

    # --- cumulative / waterfall ---

    def test_cumulative_ends_at_the_series_sum(self):
        # The defining contract of a running total.
        from diive.core.plotting.cumulative import Cumulative
        fig, ax = plt.subplots()
        Cumulative(self.series.to_frame(), units="units").plot(ax=ax, showplot=False)
        self.assertAlmostEqual(float(ax.lines[0].get_ydata()[-1]),
                               float(self.series.sum()), places=3)
        plt.close(fig)

    def test_cumulative_fill_draws_what_fill_between_draws(self):
        """Gap-free: pixel for pixel the `fill_between` shading it replaces."""
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from diive.core.plotting.cumulative import Cumulative

        def render(old: bool):
            fig = Figure(figsize=(6, 3), dpi=100)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            cum = Cumulative(self.series.to_frame())
            if old:
                cum._fill_to_zero = lambda s, color: cum.ax.fill_between(
                    s.index, s.to_numpy(), 0, color=color, alpha=0.12,
                    edgecolor='none', zorder=1)
            cum.plot(ax=ax, showplot=False, fill=True)
            fig.canvas.draw()
            return np.asarray(fig.canvas.buffer_rgba()).copy()

        np.testing.assert_array_equal(render(old=False), render(old=True))

    def test_cumulative_fill_breaks_where_the_curve_breaks(self):
        """One path, one closed outline per unbroken stretch, down to zero."""
        import numpy as np
        from matplotlib.path import Path
        from diive.core.plotting.cumulative import Cumulative
        series = self.series.copy()
        series.iloc[100:130] = np.nan   # a gap
        series.iloc[500] = np.nan       # a single missing record
        series.iloc[502] = np.nan       # ... leaving one isolated value
        fig, ax = plt.subplots()
        cum = Cumulative(series.to_frame())
        cum.plot(ax=ax, showplot=False, fill=True)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.collections[0].get_paths()), 1)
        path = ax.collections[0].get_paths()[0]
        starts = np.flatnonzero(path.codes == Path.MOVETO)
        self.assertEqual(len(starts), 4)  # [0:100), [130:500), [501], [503:]
        self.assertEqual(int((path.codes == Path.CLOSEPOLY).sum()), 4)
        curve = cum.cumulative.iloc[:, 0]
        on_curve = path.vertices[path.codes == Path.LINETO]
        # Every valid point of the curve is on the outline, plus the point
        # where each stretch returns to zero.
        self.assertEqual(len(on_curve), int(curve.notna().sum()) + 4)
        self.assertTrue(np.isin(curve.dropna().to_numpy(), on_curve[:, 1]).all())
        plt.close(fig)

    def test_cumulative_year_draws_one_line_per_year(self):
        from diive.core.plotting.cumulative import CumulativeYear
        fig, ax = plt.subplots()
        CumulativeYear(self.series, series_units="units").plot(ax=ax, showplot=False)
        labels = [line.get_label() for line in ax.lines]
        self.assertEqual(len(labels), 3)
        for year in (2019, 2020, 2021):
            self.assertTrue(any(str(year) in lbl for lbl in labels), labels)
        plt.close(fig)

    def test_waterfall_bar_per_period_and_total_matches(self):
        from diive.core.plotting.waterfall import WaterfallPlot
        monthly = self.series.resample("ME").sum()
        fig, ax = plt.subplots()
        WaterfallPlot(self.series, resample="ME", agg="sum").plot(ax=ax, showplot=False)
        bars = next(c for c in ax.collections if c.get_gid() == "waterfall_bars")
        corners = [p.vertices[:4] for p in bars.get_paths()]
        self.assertEqual(len(corners), len(monthly))
        # The running budget closes on the series total.
        tops = [c[2, 1] for c in corners]
        bottoms = [c[0, 1] for c in corners]
        final = tops[-1] if abs(tops[-1]) > abs(bottoms[-1]) else bottoms[-1]
        self.assertAlmostEqual(final, float(monthly.sum()), places=3)
        plt.close(fig)

    def test_waterfall_colours_split_by_sign(self):
        import numpy as np
        import pandas as pd
        from diive.core.plotting.waterfall import WaterfallPlot
        # Alternating monthly totals so both directions are present.
        idx = pd.date_range("2021-01-01", periods=24 * 300, freq="1h")
        values = np.where((idx.month % 2) == 0, 1.0, -1.0)
        series = pd.Series(values, index=idx, name="NEE")
        fig, ax = plt.subplots()
        WaterfallPlot(series, resample="ME", agg="sum").plot(
            ax=ax, showplot=False, color_uptake="#111111", color_release="#EEEEEE")
        bars = next(c for c in ax.collections if c.get_gid() == "waterfall_bars")
        colors = {tuple(fc[:3]) for fc in bars.get_facecolor()}
        self.assertEqual(len(colors), 2, "both uptake and release colours expected")
        plt.close(fig)

    # --- distributions ---

    def test_ridgeline_one_panel_per_group(self):
        from diive.core.plotting.ridgeline import RidgeLinePlot
        fig = plt.figure()
        RidgeLinePlot(self.series).plot(fig=fig, how="monthly", showplot=False)
        self.assertEqual(len(fig.axes), 12)
        plt.close(fig)

    def test_ridgeline_hspace_is_set_on_the_gridspec(self):
        # Documented gotcha: the overlap must be set at gridspec creation -- a
        # later gs.update(hspace=) is a silent no-op for an embedded figure.
        from diive.core.plotting.ridgeline import RidgeLinePlot
        fig = plt.figure()
        RidgeLinePlot(self.series).plot(fig=fig, how="monthly", hspace=-0.7,
                                        showplot=False)
        gridspec = fig.axes[0].get_subplotspec().get_gridspec()
        self.assertAlmostEqual(gridspec.hspace, -0.7)
        plt.close(fig)

    def test_shifted_distribution_labels_both_periods(self):
        from diive.core.plotting.shifted_distribution import ShiftedDistributionPlot
        fig, ax = plt.subplots()
        ShiftedDistributionPlot(self.series,
                                ref_period=("2019-01-01", "2019-12-31"),
                                comp_period=("2021-01-01", "2021-12-31")).plot(ax=ax)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertTrue(any("2019-01-01" in lbl for lbl in labels), labels)
        self.assertTrue(any("2021-01-01" in lbl for lbl in labels), labels)
        plt.close(fig)

    def test_shifted_distribution_periods_select_different_data(self):
        from diive.core.plotting.shifted_distribution import ShiftedDistributionPlot
        import numpy as np
        plot = ShiftedDistributionPlot(self.series,
                                       ref_period=("2019-01-01", "2019-06-30"),
                                       comp_period=("2019-07-01", "2019-12-31"))
        # First half vs second half of the annual cycle -> different means.
        self.assertFalse(np.isclose(plot._ref_data.mean(), plot._comp_data.mean()))

    # --- polar ---

    def test_treering_filled_and_line_use_different_renderers(self):
        from diive.core.plotting.treering import TreeRingPlot
        from matplotlib.projections.polar import PolarAxes
        frame = self.series.to_frame()

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        TreeRingPlot(df=frame, value_col="TA").plot(ax=ax)
        self.assertIsInstance(ax, PolarAxes)
        filled_collections = len(ax.collections)
        plt.close(fig)

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        TreeRingPlot(df=frame, value_col="TA").plot_line(ax=ax)
        # plot_line draws one trace per year instead of a single colour mesh.
        self.assertNotEqual(len(ax.collections), filled_collections)
        plt.close(fig)

    # --- yearly anomalies ---

    def test_longterm_anomalies_are_relative_to_the_reference_mean(self):
        import pandas as pd
        from diive.core.plotting.bar import LongtermAnomaliesYear
        # This class takes one value per year, indexed by integer year.
        yearly = pd.Series([1.0, 3.0, 2.0, 5.0, 0.0],
                           index=[2016, 2017, 2018, 2019, 2020], name="TA")
        plot = LongtermAnomaliesYear(series=yearly, reference_start_year=2016,
                                     reference_end_year=2018)
        # Reference mean over 2016-2018 is 2.0, so 2018 sits exactly on it.
        anomalies = plot.anomalies_df
        self.assertAlmostEqual(float(anomalies["reference_mean"].iloc[-1]), 2.0)
        self.assertAlmostEqual(float(anomalies.loc[2018, "anomaly"]), 0.0)
        self.assertAlmostEqual(float(anomalies.loc[2019, "anomaly"]), 3.0)
        self.assertAlmostEqual(float(anomalies.loc[2020, "anomaly"]), -2.0)
        # Above and below are split into two series so they can be coloured
        # differently, and each year appears in exactly one of them.
        above, below = anomalies["anomaly_above"], anomalies["anomaly_below"]
        self.assertEqual(int((above.notna() & below.notna()).sum()), 0)
        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        # One bar per year in each of the two series.
        self.assertEqual(len(ax.patches), 2 * len(yearly))
        plt.close(fig)

    def test_longterm_anomalies_sorts_an_unsorted_record(self):
        # L109: the class used to discard its own sort_index(), so an unsorted
        # record was plotted AND averaged in input order. An already-sorted input
        # cannot show this, so the index is deliberately shuffled here.
        import random
        import pandas as pd
        from diive.core.plotting.bar import LongtermAnomaliesYear
        years = list(range(1990, 2021))
        values = {yr: 8.0 + (yr - 1990) * 0.1 for yr in years}  # distinct, monotonic in year
        shuffled = years.copy()
        random.Random(42).shuffle(shuffled)
        self.assertNotEqual(shuffled, years)  # guard: the input really is unsorted
        yearly = pd.Series([values[yr] for yr in shuffled], index=shuffled, name="TA")

        plot = LongtermAnomaliesYear(series=yearly, reference_start_year=1990,
                                     reference_end_year=2000)
        anomalies = plot.anomalies_df
        self.assertEqual(list(anomalies.index), years)

        # The reference statistics are selected by year (order-independent), but
        # the "last 10 years" annotation is a tail() of the frame and is not.
        ref_mean = float(anomalies["reference_mean"].iloc[-1])
        self.assertAlmostEqual(ref_mean, sum(values[yr] for yr in range(1990, 2001)) / 11)
        last10 = [values[yr] for yr in range(2011, 2021)]
        expected_last10_mean = sum(last10) / 10

        fig, ax = plt.subplots()
        plot.plot(ax=ax)
        annotation = [t.get_text() for t in ax.texts if "last 10 years mean" in t.get_text()]
        self.assertEqual(len(annotation), 1)
        self.assertIn("(2011-2020)", annotation[0])
        self.assertIn(f"last 10 years mean: {expected_last10_mean:.2f}", annotation[0])

        # The bars themselves are drawn in frame order: above-bars first, then
        # below-bars, each with one rectangle per year (0-height where the year
        # belongs to the other series).
        n = len(years)
        heights = [p.get_height() for p in ax.patches]
        self.assertEqual(len(heights), 2 * n)
        drawn = [heights[i] + heights[n + i] for i in range(n)]
        expected = [values[yr] - ref_mean for yr in years]
        for got, want in zip(drawn, expected):
            self.assertAlmostEqual(got, want)
        plt.close(fig)

    def test_longterm_anomalies_survive_a_reference_name_collision(self):
        # L110: the working frame used to be keyed by the caller's Series name, so a
        # variable called 'reference_mean' overwrote the data column before the
        # anomaly was computed, zeroing every anomaly.
        import pandas as pd
        from diive.core.plotting.bar import LongtermAnomaliesYear
        for colliding_name in ("reference_mean", "reference_sd", "anomaly"):
            with self.subTest(name=colliding_name):
                yearly = pd.Series([1.0, 3.0, 2.0, 5.0, 0.0],
                                   index=[2016, 2017, 2018, 2019, 2020],
                                   name=colliding_name)
                plot = LongtermAnomaliesYear(series=yearly, reference_start_year=2016,
                                            reference_end_year=2018)
                # Reference mean over 2016-2018 is 2.0, unchanged by the name.
                anomaly = plot.anomalies_df["anomaly"]
                if colliding_name == "anomaly":
                    # The caller-facing rename puts two columns under this name.
                    anomaly = anomaly.iloc[:, -1]
                self.assertAlmostEqual(float(anomaly.loc[2018]), 0.0)
                self.assertAlmostEqual(float(anomaly.loc[2019]), 3.0)
                self.assertAlmostEqual(float(anomaly.loc[2020]), -2.0)
                fig, ax = plt.subplots()
                plot.plot(ax=ax)
                self.assertEqual(len(ax.patches), 2 * len(yearly))
                plt.close(fig)

    # --- 3D surface grid (library path, no gui3d extra needed) ---

    def test_datetime_surface_grid_shape_and_axes(self):
        from diive.core.plotting.surface_grid import datetime_surface_grid
        import numpy as np
        grid = datetime_surface_grid(self.series)
        self.assertEqual(grid.z.shape, (len(grid.y_days), len(grid.x_hours)))
        self.assertEqual(len(grid.x_hours), 24)          # hourly data
        # The MIDDLE index is converted to TIMESTAMP_START (as the heatmap does),
        # which shifts every record half a period back and so adds one leading
        # date row holding a single record.
        self.assertEqual(len(grid.y_days), 3 * 365 + 1)
        self.assertEqual(grid.name, "TA")
        np.testing.assert_allclose(grid.x_hours[0], 0.5)
        self.assertLess(grid.x_hours[-1], 24.0)

    def test_datetime_surface_grid_axes_match_the_heatmap(self):
        """Surface and heatmap must place the same data on the same axes.

        The 3-D surface is the 3-D analogue of `HeatmapDateTime`, so both must
        run the same timestamp preparation. Sanitizing without converting to
        TIMESTAMP_START put the surface's time-of-day axis half a period later
        than the heatmap's for a (MIDDLE-convention) diive series.
        """
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        from diive.core.plotting.surface_grid import datetime_surface_grid
        import numpy as np
        grid = datetime_surface_grid(self.series)
        hm = HeatmapDateTime(self.series, ax_orientation="vertical")
        # The heatmap's x/y are pcolormesh *boundaries*: one entry longer than
        # the data, so drop the trailing bound before comparing.
        np.testing.assert_allclose(grid.x_hours, hm.x[:-1])
        np.testing.assert_array_equal(grid.dates, hm.y[:-1])
        np.testing.assert_allclose(grid.z, hm.z, equal_nan=True)

    def test_datetime_surface_grid_keeps_gaps_as_nan(self):
        from diive.core.plotting.surface_grid import datetime_surface_grid
        import numpy as np
        gappy = self.series.copy()
        gappy.iloc[:24] = np.nan  # blank the first whole day
        grid = datetime_surface_grid(gappy)
        self.assertTrue(np.all(np.isnan(grid.z[0])))
        self.assertFalse(np.all(np.isnan(grid.z[1])))

    # --- crash-on-legitimate-input regressions ---

    def test_ridgeline_plots_a_series_that_contains_gaps(self):
        """L65: the KDE rejects NaN, so a gappy series must be cleaned by the class.

        Every real time series has gaps. The GUI path only worked because it
        called `.dropna()` first, which made the library API strictly worse.
        """
        import numpy as np
        from diive.core.plotting.ridgeline import RidgeLinePlot
        gappy = self.series.copy()
        gappy.iloc[::7] = np.nan  # gaps scattered across every group
        gappy.loc[gappy.index.month == 7] = np.nan  # one group with nothing left
        fig = plt.figure()
        RidgeLinePlot(gappy).plot(fig=fig, how="monthly", showplot=False)
        # Eleven ridges: July dropped out entirely instead of raising.
        self.assertEqual(len(fig.axes), 11)
        labels = [t.get_text() for ax in fig.axes for t in ax.texts]
        self.assertNotIn("7", labels)
        plt.close(fig)
        # A series with nothing left at all says so rather than failing inside sklearn.
        with self.assertRaises(ValueError) as ctx:
            RidgeLinePlot(self.series * np.nan)
        self.assertIn("no valid", str(ctx.exception))

    def test_cumulative_labels_an_all_nan_column_instead_of_raising(self):
        """L69: an unfilled scenario column is all-NaN, and had no legend total to index."""
        import numpy as np
        from diive.core.plotting.cumulative import Cumulative
        frame = self.series.to_frame()
        frame["SCENARIO_UNFILLED"] = np.nan
        fig, ax = plt.subplots()
        Cumulative(frame, units="units").plot(ax=ax, showplot=False)
        labels = [line.get_label() for line in ax.lines]
        # One line per column, and the empty one is labelled as such.
        self.assertTrue(any(lbl.startswith("TA: ") for lbl in labels), labels)
        self.assertIn("SCENARIO_UNFILLED: no data", labels)
        # No end-point marker/annotation was invented for the empty column.
        self.assertNotIn("nan", " ".join(t.get_text() for t in ax.texts))
        plt.close(fig)

    def test_datetime_surface_grid_keeps_a_variable_named_date(self):
        """L68: the DATE/TIME helper columns used to overwrite a same-named variable."""
        import numpy as np
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        from diive.core.plotting.surface_grid import datetime_surface_grid
        expected = datetime_surface_grid(self.series).z
        for name in ("DATE", "TIME"):
            with self.subTest(name=name):
                renamed = self.series.rename(name)
                grid = datetime_surface_grid(renamed)
                self.assertEqual(grid.name, name)
                np.testing.assert_allclose(grid.z, expected, equal_nan=True)
                # The heatmap builds the same DATE/TIME columns the same way.
                hm = HeatmapDateTime(renamed, ax_orientation="vertical")
                np.testing.assert_allclose(hm.z, expected, equal_nan=True)


if __name__ == '__main__':
    unittest.main()


class TestHeatmapDateTimeShowLessXticklabels(unittest.TestCase):
    """`show_less_xticklabels=True` must actually thin out the x-tick labels.

    The parameter was documented, exposed by the GUI's plot settings and emitted
    into copied code snippets, but `HeatmapDateTime.plot` only stored it — label
    visibility was identical for True and False. (`get_xticklabels` returns only
    the visible labels, so hiding shows up as a shorter list.)
    """

    @staticmethod
    def _labels(orientation, show_less):
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        fig, ax = plt.subplots()
        HeatmapDateTime(_synthetic_series(years=1), ax_orientation=orientation).plot(
            ax=ax, fig=fig, show_less_xticklabels=show_less)
        texts = [label.get_text() for label in ax.get_xticklabels()]
        plt.close(fig)
        return texts

    def test_every_second_hour_label_is_hidden(self):
        # Hourly data puts ticks on every 3rd hour.
        self.assertEqual(self._labels("vertical", False),
                         ['3', '6', '9', '12', '15', '18', '21'])
        self.assertEqual(self._labels("vertical", True), ['3', '9', '15', '21'])

    def test_it_also_thins_the_date_axis(self):
        # Horizontal puts the dates on x, where the labels come from the
        # auto date locator instead of a fixed tick list.
        full = self._labels("horizontal", False)
        thinned = self._labels("horizontal", True)
        self.assertEqual(thinned, full[::2])
        self.assertLess(len(thinned), len(full))


class TestHeatmapYearMonthLattice(unittest.TestCase):
    """Cells must sit on a complete year x month lattice.

    `_set_bounds` hands the surviving labels to pcolormesh as cell *boundaries*,
    so a month nothing fell into is not drawn empty — its neighbour stretches
    across the gap while the axis keeps its regular 1..12 ticks, putting one
    month's colour under another month's label.
    """

    @staticmethod
    def _winter_campaign():
        import numpy as np
        import pandas as pd
        # Nov 2019 - Feb 2020: months 3..10 never occur.
        ix = pd.date_range('2019-11-01', '2020-02-28 23:30', freq='30min',
                           name='TIMESTAMP_MIDDLE')
        return pd.Series(np.arange(len(ix), dtype=float), index=ix, name='X')

    def _plot(self):
        import matplotlib.pyplot as plt
        from diive.core.plotting.heatmap_datetime import HeatmapYearMonth
        h = HeatmapYearMonth(series=self._winter_campaign())
        fig, ax = plt.subplots()
        h.plot(ax=ax)
        plt.close(fig)
        return h, ax

    def test_every_month_gets_a_cell(self):
        h, _ax = self._plot()
        self.assertEqual(h.z.shape[1], 12, "all 12 months must be present")
        self.assertEqual(len(h.x), 13, "12 cells need 13 boundaries")

    def test_no_cell_spans_more_than_one_month(self):
        import numpy as np
        h, _ax = self._plot()
        widths = np.diff(h.x)
        self.assertTrue((widths == 1).all(),
                        f"every cell must be one month wide, got {sorted(set(widths))}")

    def test_unobserved_months_are_empty_not_borrowed(self):
        import numpy as np
        h, _ax = self._plot()
        # Months 3..9 (0-based columns 2..8) have no data in either year. October
        # is deliberately excluded: the heatmap converts to TIMESTAMP_START, which
        # moves the first record of 2019-11-01 00:00 back to 2019-10-31 23:45.
        self.assertTrue(np.isnan(h.z[:, 2:9]).all())
        # ...and each observed month kept a value in at least one year (the
        # campaign spans Nov-Dec of 2019 and Jan-Feb of 2020, so no single month
        # is filled in both).
        observed = h.z[:, [0, 1, 10, 11]]
        self.assertTrue(np.isfinite(observed).any(axis=0).all())

    def test_cells_and_ticks_agree(self):
        h, ax = self._plot()
        # 12 labelled ticks over 12 cells; before the fix there were 12 ticks
        # over 5 cells, so a tick pointed at the wrong month's colour.
        self.assertEqual(len(ax.get_xticks()), h.z.shape[1])
