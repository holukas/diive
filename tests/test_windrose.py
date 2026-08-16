"""
TESTS: WIND ROSE PLOT
=====================

Tests for how `WindRosePlot.plot` treats a caller-supplied axes and a shared
`FormatStyle`. A rose drawn into one panel of someone else's figure must title
its own axes and leave that figure's suptitle and other panels alone, while a
rose that created its own figure keeps the suptitle. The `FormatStyle` fields
that describe something a polar rose actually draws must reach the axes, and
the cartesian-only fields must stay ignored.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diive.core.plotting.styles.format import FormatStyle
from diive.core.plotting.windrose import WindRosePlot

N_SECTORS = 8
COMPASS_8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
CALLER_SUPTITLE = 'CALLER SUPTITLE'


def _rose() -> WindRosePlot:
    """A rose whose sector means are exactly 0, 1, ..., 7 (one reading per sector centre)."""
    centers = np.arange(N_SECTORS) * (360.0 / N_SECTORS)
    index = pd.date_range('2021-01-01', periods=N_SECTORS, freq='30min')
    wind_dir = pd.Series(centers, index=index, name='WD')
    values = pd.Series(np.arange(N_SECTORS, dtype=float), index=index, name='FC')
    return WindRosePlot(series=values, wind_dir=wind_dir, agg='mean', n_sectors=N_SECTORS)


def _own_figure(**style_kwargs):
    """Draw a rose on its own figure and return its axes."""
    ax = _rose().plot(format_style=FormatStyle(**style_kwargs), show_colorbar=False)
    ax.figure.canvas.draw()
    return ax


def _panel_figure():
    """Return (figure, cartesian neighbour axes, empty polar axes) with a caller suptitle."""
    fig = plt.figure(figsize=(10, 5))
    neighbour = fig.add_subplot(1, 2, 1)
    neighbour.plot([0, 1], [0, 1])
    polar = fig.add_subplot(1, 2, 2, projection='polar')
    fig.suptitle(CALLER_SUPTITLE)
    fig.canvas.draw()
    return fig, neighbour, polar


class TestCallerSuppliedAxes(unittest.TestCase):
    """A caller who passes `ax` owns the figure; the rose must stay inside that axes."""

    def test_title_goes_on_the_axes_not_the_figure(self):
        """The caller's suptitle survives and the title lands on the rose's own axes."""
        fig, _, polar = _panel_figure()
        _rose().plot(ax=polar, format_style=FormatStyle(title='panel B'), show_colorbar=False)
        fig.canvas.draw()
        self.assertEqual(fig._suptitle.get_text(), CALLER_SUPTITLE)
        self.assertEqual(polar.get_title(), 'panel B')
        plt.close(fig)

    def test_neighbour_panel_is_not_moved(self):
        """Drawing the rose must not reflow the caller's figure layout."""
        fig, neighbour, polar = _panel_figure()
        before = np.array(neighbour.get_position().bounds)
        _rose().plot(ax=polar, format_style=FormatStyle(title='panel B'), show_colorbar=False)
        fig.canvas.draw()
        after = np.array(neighbour.get_position().bounds)
        self.assertEqual(np.abs(after - before).max(), 0.0)
        plt.close(fig)

    def test_axes_title_clears_the_compass_ring(self):
        """The title sits above the 'N' compass label instead of on top of it."""
        fig, _, polar = _panel_figure()
        _rose().plot(ax=polar, format_style=FormatStyle(title='panel B'), show_colorbar=False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        north = [t for t in polar.get_xticklabels() if t.get_text() == 'N'][0]
        self.assertGreater(polar.title.get_window_extent(renderer).y0,
                           north.get_window_extent(renderer).y1)
        plt.close(fig)

    def test_own_figure_keeps_the_suptitle(self):
        """Without a caller axes the rose owns the figure, so the title stays a suptitle."""
        ax = _own_figure(title='My rose')
        self.assertEqual(ax.figure._suptitle.get_text(), 'My rose')
        self.assertEqual(ax.get_title(), '')
        plt.close(ax.figure)


class TestHonouredFormatStyleFields(unittest.TestCase):
    """The `FormatStyle` fields that describe something a polar rose draws."""

    def test_ticks_fontsize_sizes_compass_and_radial_labels(self):
        ax = _own_figure(ticks_fontsize=29.0)
        self.assertEqual(ax.get_xticklabels()[0].get_fontsize(), 29.0)
        self.assertEqual(ax.get_yticklabels()[0].get_fontsize(), 29.0)
        plt.close(ax.figure)

    def test_explicit_sector_label_fontsize_wins_over_style(self):
        """The dedicated argument is the direct knob and overrides the shared style."""
        ax = _rose().plot(format_style=FormatStyle(ticks_fontsize=29.0),
                          sector_label_fontsize=8.0, show_colorbar=False)
        ax.figure.canvas.draw()
        self.assertEqual(ax.get_xticklabels()[0].get_fontsize(), 8.0)
        self.assertEqual(ax.get_yticklabels()[0].get_fontsize(), 29.0)
        plt.close(ax.figure)

    def test_chrome_color_colours_tick_labels(self):
        ax = _own_figure(chrome_color='#00ff00')
        self.assertEqual(ax.get_xticklabels()[0].get_color(), '#00ff00')
        self.assertEqual(ax.get_yticklabels()[0].get_color(), '#00ff00')
        # Recolouring must not undo the compass label font size.
        self.assertEqual(ax.get_xticklabels()[0].get_fontsize(), 16.0)
        plt.close(ax.figure)

    def test_show_grid_false_hides_every_gridline(self):
        ax = _own_figure(show_grid=False)
        self.assertFalse(any(gl.get_visible() for gl in ax.xaxis.get_gridlines()))
        self.assertFalse(any(gl.get_visible() for gl in ax.yaxis.get_gridlines()))
        plt.close(ax.figure)

    def test_grid_color_and_facecolor(self):
        ax = _own_figure(grid_color='#ff00ff', facecolor='#eeddcc')
        self.assertEqual(ax.xaxis.get_gridlines()[0].get_color(), '#ff00ff')
        self.assertTrue(ax.xaxis.get_gridlines()[0].get_visible())
        self.assertEqual(ax.get_facecolor(), (0xee / 255, 0xdd / 255, 0xcc / 255, 1.0))
        plt.close(ax.figure)

    def test_title_font_fields(self):
        ax = _own_figure(title='T', title_fontsize=33.0, title_fontweight='light',
                         text_color='#ff0000')
        suptitle = ax.figure._suptitle
        self.assertEqual(suptitle.get_fontsize(), 33.0)
        self.assertEqual(suptitle.get_fontweight(), 'light')
        self.assertEqual(suptitle.get_color(), '#ff0000')
        plt.close(ax.figure)


class TestIgnoredFormatStyleFields(unittest.TestCase):
    """Cartesian-only fields stay ignored on the polar rose, as documented."""

    def test_axis_label_fields_draw_nothing(self):
        ax = _own_figure(xlabel='X', ylabel='Y', xunits='(xu)', yunits='(yu)',
                         zlabel='Z', axlabel_fontsize=27.0)
        self.assertEqual(ax.get_xlabel(), '')
        self.assertEqual(ax.get_ylabel(), '')
        plt.close(ax.figure)

    def test_legend_fields_draw_nothing(self):
        """The rose has no labelled artists, so there is nothing to put in a legend."""
        ax = _own_figure(show_legend=True, legend_loc='upper left', legend_ncol=3)
        self.assertIsNone(ax.get_legend())
        plt.close(ax.figure)

    def test_tick_geometry_fields_leave_the_ticks_alone(self):
        """`ticks_length`/`ticks_width`/`ticks_direction` describe cartesian spines."""
        default = _own_figure()
        expected = default.xaxis.get_major_ticks()[0].tick1line.get_markersize()
        plt.close(default.figure)
        ax = _own_figure(ticks_length=12.0, ticks_width=3.5, ticks_direction='out',
                         spine_linewidth=4.5)
        self.assertEqual(ax.xaxis.get_major_ticks()[0].tick1line.get_markersize(), expected)
        plt.close(ax.figure)


class TestDefaultRenderUnchanged(unittest.TestCase):
    """The default render (no `format_style`, no caller axes) must be untouched."""

    def test_default_render(self):
        ax = _rose().plot(show_colorbar=False)
        ax.figure.canvas.draw()

        # Bars: one per sector, anchored at zero, height = the sector mean (0..7),
        # angular width 0.9 of the sector, drawn at the sector centres.
        bars = ax.patches
        self.assertEqual(len(bars), N_SECTORS)
        sector_width = 2.0 * np.pi / N_SECTORS
        for i, bar in enumerate(bars):
            # get_x() is the bar's left edge, i.e. half a bar width before the centre.
            self.assertAlmostEqual(bar.get_x(), i * sector_width - sector_width * 0.9 / 2,
                                   places=12)
            self.assertAlmostEqual(bar.get_y(), 0.0, places=12)
            self.assertAlmostEqual(bar.get_height(), float(i), places=12)
            self.assertAlmostEqual(bar.get_width(), sector_width * 0.9, places=12)

        # Meteorological layout, compass ticks, and the radial extent around them.
        self.assertEqual(ax.get_theta_direction(), -1)
        self.assertAlmostEqual(ax.get_theta_offset(), np.pi / 2, places=12)
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], COMPASS_8)
        np.testing.assert_allclose(ax.get_xticks(),
                                   np.arange(N_SECTORS) * sector_width, atol=1e-12)
        self.assertAlmostEqual(ax.get_ylim()[0], 0.0, places=12)
        self.assertAlmostEqual(ax.get_ylim()[1], 7.0 + 7.0 * 0.08, places=12)
        self.assertAlmostEqual(ax.get_rorigin(), -7.0 * 0.05, places=12)

        # Default chrome: no title anywhere, house font sizes, grid on in the
        # matplotlib default colour at alpha 0.3, white face.
        self.assertEqual(ax.get_title(), '')
        self.assertIsNone(ax.figure._suptitle)
        self.assertEqual(ax.get_xticklabels()[0].get_fontsize(), 16.0)
        self.assertEqual(ax.get_yticklabels()[0].get_fontsize(), 12.0)
        self.assertEqual(ax.get_xticklabels()[0].get_color(), 'black')
        self.assertTrue(all(gl.get_visible() for gl in ax.xaxis.get_gridlines()))
        self.assertEqual(ax.xaxis.get_gridlines()[0].get_color(), '#b0b0b0')
        self.assertEqual(ax.xaxis.get_gridlines()[0].get_alpha(), 0.3)
        self.assertEqual(ax.get_facecolor(), (1.0, 1.0, 1.0, 1.0))
        plt.close(ax.figure)


if __name__ == '__main__':
    unittest.main()
