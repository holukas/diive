"""Tests for FootprintPlot, the map of a flux footprint or footprint climatology."""
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import diive as dv
from diive.core.plotting.footprint import FootprintPlot
from diive.core.plotting.styles._material_design_colors import amber
from diive.flux.lowres.footprint import ffp_climatology, ffp_footprint

SINGLE = dict(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4, z0=0.15, verbose=0)
SERIES = dict(zm=20., h=[1500., 900., 1200., 2000., 600., 800., 1100.],
              ol=[-50., 200., -500., 30., -20., 1000., -2000.],
              sigmav=[0.6, 0.4, 0.8, 0.3, 1.1, 0.5, 0.7],
              ustar=[0.4, 0.3, 0.5, 0.35, 0.25, 0.35, 0.6],
              wind_dir=[210., 225., 190., 45., 260., 300., 15.])


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def _contour_lines(ax, plot):
    """Line2D artists drawn from the contour vertices of the result."""
    lines = []
    for _, xr, _ in plot.contours:
        lines += [ln for ln in ax.lines if len(ln.get_xdata()) == len(xr)
                  and np.array_equal(ln.get_xdata(), xr)]
    return lines


class TestFootprintPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.single = ffp_footprint(**SINGLE, wind_dir=220.)
        cls.unrotated = ffp_footprint(**SINGLE)
        cls.no_contours = ffp_footprint(**SINGLE, wind_dir=100., rs=None)
        cls.clim = ffp_climatology(**SERIES, z0=0.2, nx=200, verbose=0)

    def tearDown(self):
        plt.close('all')

    def test_climatology_on_given_axes(self):
        # A bare Figure: the plot must not draw through pyplot
        fig = Figure(figsize=(8, 7))
        ax = fig.add_subplot()
        p = FootprintPlot(self.clim)
        self.assertIs(p.plot(ax=ax), ax)
        self.assertTrue(p.is_climatology)
        self.assertEqual(p.n, self.clim.n)

        # Field: one mesh with finite values
        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        self.assertEqual(len(meshes), 1)
        self.assertGreater(np.isfinite(meshes[0].get_array()).sum(), 0)

        # Every contour inside the domain is drawn, the 80 % line in the highlight colour
        n_inside = sum(fr is not None for fr in self.clim.fr)
        self.assertGreater(n_inside, 0)
        self.assertEqual(len(p.contours), n_inside)
        lines = _contour_lines(ax, p)
        self.assertEqual(len(lines), n_inside)
        self.assertEqual(p.source_area_r, 0.8)
        hi = [ln for ln in lines if to_hex(ln.get_color()) == to_hex(amber(700))]
        self.assertEqual(len(hi), 1)
        self.assertGreater(hi[0].get_linewidth(), lines[0].get_linewidth())

        texts = _texts(ax)
        self.assertIn('80 %', texts)
        self.assertIn('N', texts)  # north arrow
        info = [t for t in texts if t.startswith('n = ')]
        self.assertEqual(info, [p.info_text()])
        self.assertIn(f"n = {self.clim.n:,} footprints", info[0])
        self.assertIn(f"80 % contour reaches {p.source_area_distance:.0f} m", info[0])
        self.assertEqual(ax.get_title(loc='left'), 'Footprint climatology')
        self.assertEqual(ax.get_xlabel(), 'Distance east of tower (m)')

        # Tower at the origin, peak at the maximum of the field
        tower = [ln for ln in ax.lines if ln.get_label() == 'Tower']
        self.assertEqual((tower[0].get_xdata()[0], tower[0].get_ydata()[0]), (0, 0))
        i, j = np.unravel_index(np.argmax(self.clim.fclim_2d), self.clim.fclim_2d.shape)
        self.assertAlmostEqual(p.peak_distance, np.hypot(self.clim.x_2d[i, j], self.clim.y_2d[i, j]))
        self.assertIsNotNone(ax.get_legend())

        # Colour bar added to the same figure, with the contour levels marked
        self.assertEqual(len(fig.axes), 2)
        self.assertIs(p.colorbar.ax, fig.axes[1])
        self.assertEqual(len(p.colorbar.ax.lines), n_inside)

        # The map covers the 80 % line and the tower, with equal scales
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        _, xr, yr = [c for c in p.contours if c[0] == 0.8][0]
        self.assertTrue(xmin < min(xr.min(), 0) and max(xr.max(), 0) < xmax)
        self.assertTrue(ymin < min(yr.min(), 0) and max(yr.max(), 0) < ymax)
        self.assertEqual(ax.get_aspect(), 1.0)

    def test_single_footprint_with_profile(self):
        p = dv.plotting.FootprintPlot(self.single)
        ax = p.plot(figdpi=50)
        self.assertFalse(p.is_climatology)
        self.assertTrue(p.is_geographic)
        # Map, the two profile panels and the colour bar (an inset of the map)
        self.assertEqual(len(p.fig.axes), 3)
        self.assertIs(p.ax, ax)
        self.assertIs(p.colorbar.ax.figure, p.fig)
        self.assertIsNotNone(p.ax_profile)
        self.assertEqual(len(_contour_lines(ax, p)), len(self.single.rs))

        # Peak marker at x_ci_max, in the wind direction (wind from SW: peak to the SW)
        self.assertAlmostEqual(p.peak_distance, self.single.x_ci_max)
        self.assertAlmostEqual(np.hypot(*p.peak_xy), self.single.x_ci_max)
        self.assertLess(p.peak_xy[0], 0)
        self.assertLess(p.peak_xy[1], 0)
        peak = [ln for ln in ax.lines if ln.get_label() == 'Peak'][0]
        self.assertAlmostEqual(peak.get_xdata()[0], p.peak_xy[0])

        # Profile: the curve is f_ci (km-1), the cumulative reaches 80 % at the marked distance
        prof, cum = p.ax_profile, p.ax_cumulative
        self.assertAlmostEqual(prof.lines[0].get_ydata().max(), self.single.f_ci.max() * 1000)
        x_ci, f_ci = self.single.x_ci, self.single.f_ci
        cum_ref = np.concatenate(([0.], np.cumsum(np.diff(x_ci) * (f_ci[1:] + f_ci[:-1]) / 2)))
        self.assertAlmostEqual(np.interp(p.cumulative_distance, x_ci, cum_ref), 0.8, places=6)
        self.assertTrue(p.peak_distance < p.cumulative_distance < p.source_area_distance)
        self.assertIn(f"80 % within {p.cumulative_distance:.0f} m", _texts(cum))
        self.assertIn(f"Peak {p.peak_distance:.0f} m", _texts(prof))
        self.assertEqual(cum.get_ylim(), (0, 100))

    def test_unrotated_footprint_stays_in_wind_frame(self):
        p = FootprintPlot(self.unrotated)
        ax = p.plot(show_profile=False, figdpi=50)
        self.assertFalse(p.is_geographic)
        self.assertEqual(len(p.fig.axes), 1)
        self.assertIsNone(p.ax_profile)
        self.assertEqual(ax.get_xlabel(), 'Distance upwind of tower (m)')
        self.assertNotIn('N', _texts(ax))
        self.assertAlmostEqual(p.peak_xy[1], 0., delta=1.)
        self.assertAlmostEqual(p.peak_xy[0], self.unrotated.x_ci_max)

    def test_without_contour_lines(self):
        p = FootprintPlot(self.no_contours)
        fig = Figure()
        ax = fig.add_subplot()
        p.plot(ax=ax)
        self.assertEqual(p.contours, [])
        self.assertIsNone(p.source_area_distance)
        self.assertFalse(any('%' in t for t in _texts(ax)))
        self.assertEqual(len([c for c in ax.collections if isinstance(c, QuadMesh)]), 1)

    def test_source_area_and_style(self):
        p = FootprintPlot(self.single, source_area=50)
        fig = Figure()
        ax = fig.add_subplot()
        p.plot(ax=ax, format_style=dv.plotting.FormatStyle(title='Site X', show_legend=False),
               show_colorbar=False, show_info=False, show_rings=False)
        self.assertEqual(p.source_area_r, 0.5)
        self.assertEqual(ax.get_title(loc='left'), 'Site X')
        self.assertEqual(ax.get_title(), '')
        self.assertIsNone(ax.get_legend())
        self.assertEqual(len(fig.axes), 1)
        self.assertIsNone(p.colorbar)
        self.assertFalse(any(t.startswith('Peak at') for t in _texts(ax)))
        # A second call draws again on new axes
        ax2 = Figure().add_subplot()
        p.plot(ax=ax2)
        self.assertIs(p.ax, ax2)
        self.assertEqual(len(_contour_lines(ax2, p)), len(self.single.rs))

    def test_legend_leaves_labels_tower_and_peak_free(self):
        # Stable records of the EddyPro example: matplotlib's 'best' puts the legend
        # over the 80 % label here, because it counts the range-ring vertices.
        from diive.configs.exampledata import load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN
        df, _ = load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN()
        df = df[df['L'] > 0]
        clim = ffp_climatology(zm=2.23, h=np.full(len(df), 1000.), ol=df['L'],
                               sigmav=np.sqrt(df['v_var']), ustar=df['u*'],
                               wind_dir=df['wind_dir'], umean=df['wind_speed'],
                               domain=[-250, 250, -250, 250], dx=1., verbose=0)
        for result in (clim, self.clim, self.single, self.unrotated):
            # Right panel of two, as in the gallery example
            fig = Figure(figsize=(16, 7.5), layout='constrained')
            ax = fig.subplots(1, 2)[1]
            p = FootprintPlot(result)
            p.plot(ax=ax)
            fig.canvas.draw()
            legend = ax.get_legend().get_window_extent()
            labels = [t.get_window_extent() for t in ax.texts if t.get_text().endswith(' %')]
            self.assertGreater(len(labels), 0)
            self.assertFalse(any(legend.overlaps(b) for b in labels))
            for xy in ((0., 0.), p.peak_xy):
                self.assertFalse(legend.contains(*ax.transData.transform(xy)))

    def test_title_and_subtitle_do_not_overlap(self):
        # The subtitle sat 2 % of the map height above the map, so on a tall map
        # (e.g. the GUI canvas) it reached into the title.
        for size in ((6, 5), (14, 14)):
            with self.subTest(size=size):
                fig = Figure(figsize=size)
                ax = fig.add_axes((0.1, 0.05, 0.8, 0.8))
                p = FootprintPlot(self.single)
                p.plot(ax=ax, show_colorbar=False)
                FigureCanvasAgg(fig).draw()  # the equal aspect moves the map at draw time
                renderer = fig.canvas.get_renderer()
                info = [t for t in ax.texts if t.get_text() == p.info_text()][0]
                title = ax._left_title.get_window_extent(renderer)
                sub = info.get_window_extent(renderer)
                self.assertGreater(title.y0, sub.y1)
                self.assertGreater(sub.y0, ax.get_window_extent(renderer).y1)

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            FootprintPlot(object())
        with self.assertRaises(ValueError):
            FootprintPlot(self.clim).plot_profile()


if __name__ == '__main__':
    unittest.main()
