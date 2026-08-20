# tests/test_timeseries_plot.py
"""Tests for `TimeSeries` colour-by rendering (findings L112 and L113)."""
import unittest
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.io.state import curstate
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.layout_engine import ConstrainedLayoutEngine

import diive.core.plotting.timeseries as timeseries
from diive.core.plotting.timeseries import TimeSeries, _COLOR_NOCOLOR
from diive.core.utils.console import add_console_sink, remove_console_sink


class _ConsoleSink:
    """Collects everything the shared diive console prints (`add_console_sink`)."""

    def __init__(self):
        self.lines = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    def log(self, *args, **kwargs):
        self.print(*args, **kwargs)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class _TimeSeriesRenderMixin:
    N = 200

    def setUp(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range('2024-06-01', periods=self.N, freq='30min')
        self.idx = idx
        self.fc = pd.Series(rng.normal(0, 3, self.N), index=idx, name='FC')
        self.ta = pd.Series(rng.normal(15, 5, self.N), index=idx, name='TA')

    def tearDown(self):
        plt.close('all')

    def _render(self, color_series, **kwargs):
        """Renders and returns the axes, drawn so scalar-mapped colours resolve.

        `Collection.update_scalarmappable` runs at draw time, so the RGBA that is
        actually painted only exists after `draw()`.
        """
        fig, ax = plt.subplots()
        TimeSeries(series=self.fc, color_series=color_series).plot(ax=ax, **kwargs)
        fig.canvas.draw()
        return fig, ax

    @staticmethod
    def _linecollection(ax):
        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        return lcs[0] if lcs else None

    @staticmethod
    def _scatter(ax):
        scs = [c for c in ax.collections if not isinstance(c, LineCollection)]
        return scs[0] if scs else None


class TestColourGapsStayVisible(_TimeSeriesRenderMixin, unittest.TestCase):
    """L112: a gap in the *colour* series must not erase *measured* data.

    A NaN colour value maps to the colormap's "bad" colour, and matplotlib's
    default is `(0, 0, 0, 0)`. Worse, an all-zero bad colour also makes
    `Colormap.__call__` discard the collection alpha, so those segments were
    painted with alpha exactly 0 — measured records indistinguishable from a
    data gap. Measured before the fix: 81 of 199 segments and 80 of 200 markers
    at alpha 0, leaving 80 of 200 measured records invisible.
    """

    GAP = slice(60, 140)  # 80 records where the colour driver is missing

    def setUp(self):
        super().setUp()
        self.ta_gappy = self.ta.copy()
        self.ta_gappy.iloc[self.GAP] = np.nan
        self.assertEqual(self.ta_gappy.isna().sum(), 80)
        self.assertEqual(self.fc.notna().sum(), self.N)  # data itself is complete

    def test_no_segment_is_fully_transparent(self):
        _, ax = self._render(self.ta_gappy)
        lc = self._linecollection(ax)
        rgba = lc.get_colors()
        self.assertEqual(len(rgba), self.N - 1)
        self.assertEqual(int((rgba[:, 3] == 0).sum()), 0)

    def test_every_measured_record_is_touched_by_a_visible_segment(self):
        _, ax = self._render(self.ta_gappy)
        rgba = self._linecollection(ax).get_colors()
        visible = np.zeros(self.N, dtype=bool)
        for i, alpha in enumerate(rgba[:, 3]):
            if alpha > 0:
                visible[i] = visible[i + 1] = True
        self.assertEqual(int(visible.sum()), self.N)

    def test_missing_colour_is_painted_the_neutral_grey(self):
        """Not just "visible": the colour must read as "no colour value here"."""
        _, ax = self._render(self.ta_gappy)
        lc = self._linecollection(ax)
        is_nan = np.isnan(np.asarray(lc.get_array(), dtype=float))
        self.assertGreater(int(is_nan.sum()), 0, "test needs NaN-coloured segments")
        painted = lc.get_colors()[is_nan][:, :3]
        np.testing.assert_allclose(painted, np.broadcast_to(to_rgb(_COLOR_NOCOLOR),
                                                            painted.shape), atol=1e-6)

    def test_markers_over_a_colour_gap_are_visible_too(self):
        """The scatter markers share the colormap, so they had the same defect."""
        _, ax = self._render(self.ta_gappy, marker=True)
        facecolors = self._scatter(ax).get_facecolor()
        self.assertEqual(len(facecolors), self.N)
        self.assertEqual(int((facecolors[:, 3] == 0).sum()), 0)

    def test_gap_in_the_data_still_breaks_the_line(self):
        """The other direction: a gap in `series` must still drop its segments."""
        fc_gappy = self.fc.copy()
        fc_gappy.iloc[100:110] = np.nan
        fig, ax = plt.subplots()
        TimeSeries(series=fc_gappy, color_series=self.ta).plot(ax=ax)
        fig.canvas.draw()
        # 10 missing records remove the 11 segments that touch them.
        self.assertEqual(len(self._linecollection(ax).get_segments()), self.N - 1 - 11)

    def test_a_caller_supplied_colormap_is_not_mutated(self):
        """`set_bad` must land on a copy, not on the object the caller handed in.

        `matplotlib.colormaps[name]` already hands back a copy, so the string
        path is safe either way; a `Colormap` *instance* is what the explicit
        `.copy()` protects.
        """
        caller_cmap = matplotlib.colormaps['viridis']
        before = caller_cmap.get_bad().copy()
        self._render(self.ta_gappy, cmap=caller_cmap)
        np.testing.assert_array_equal(caller_cmap.get_bad(), before)

    def test_a_colormap_instance_still_greys_the_gaps(self):
        """The non-string `cmap` path must get the same treatment."""
        _, ax = self._render(self.ta_gappy, cmap=matplotlib.colormaps['plasma'])
        self.assertEqual(int((self._linecollection(ax).get_colors()[:, 3] == 0).sum()), 0)


class TestCompleteColourSeriesUnchanged(_TimeSeriesRenderMixin, unittest.TestCase):
    """The L112 fix must move nothing when the colour series has no gaps."""

    def test_no_bad_colour_appears(self):
        _, ax = self._render(self.ta, marker=True)
        lc = self._linecollection(ax)
        self.assertFalse(np.isnan(np.asarray(lc.get_array(), dtype=float)).any())
        rgba = lc.get_colors()
        self.assertEqual(int((rgba[:, 3] == 0).sum()), 0)
        # Nothing is painted the "no colour" grey when every record has a value.
        grey = np.asarray(to_rgb(_COLOR_NOCOLOR))
        self.assertEqual(int(np.all(np.isclose(rgba[:, :3], grey, atol=1e-6), axis=1).sum()), 0)

    def test_colours_match_the_colormap_lookup_exactly(self):
        """Segment colours are still the plain cmap(norm(seg_c)) values."""
        _, ax = self._render(self.ta, cmap='plasma', alpha=0.95)
        lc = self._linecollection(ax)
        seg_c = np.asarray(lc.get_array(), dtype=float)
        expected = matplotlib.colormaps['plasma'](lc.norm(seg_c))
        expected[:, 3] = 0.95
        np.testing.assert_allclose(lc.get_colors(), expected, rtol=0, atol=0)

    def test_colorbar_is_drawn(self):
        fig, ax = self._render(self.ta, show_colorbar=True, color_label='TA (degC)')
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[1].get_ylabel(), 'TA (degC)')

    def test_plain_path_draws_a_single_coloured_line(self):
        """No `color_series` at all: unaffected by either fix."""
        fig, ax = plt.subplots()
        TimeSeries(series=self.fc).plot(ax=ax, color='#FF0000')
        fig.canvas.draw()
        self.assertEqual([c for c in ax.collections if isinstance(c, LineCollection)], [])
        self.assertEqual(ax.lines[0].get_color(), '#FF0000')
        self.assertEqual(len(ax.lines[0].get_ydata()), self.N)


class TestUnalignedColourSeriesWarns(_TimeSeriesRenderMixin, unittest.TestCase):
    """L113: colour-by silently degraded to a plain line on a non-overlapping index.

    `color_series.reindex(self.series.index)` returns all-NaN, the `>= 2 finite`
    guard takes the plain branch, and `cmap` / `show_colorbar` / `color_label`
    become no-ops with nothing said. Measured before the fix: 0 LineCollections,
    0 colorbar axes, no output.
    """

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(7)
        # TIMESTAMP_END against the data's TIMESTAMP_MIDDLE: zero shared stamps.
        shifted = self.idx + pd.Timedelta(minutes=15)
        self.ta_unaligned = pd.Series(rng.normal(15, 5, self.N), index=shifted, name='TA')
        self.assertEqual(len(self.idx.intersection(shifted)), 0)
        self.sink = _ConsoleSink()
        add_console_sink(self.sink)
        self.addCleanup(remove_console_sink, self.sink)

    def test_warns_and_names_the_likely_cause(self):
        self._render(self.ta_unaligned)
        text = self.sink.text
        self.assertIn("Cannot colour 'FC' by 'TA'", text)
        self.assertIn("0 of 200 records", text)
        self.assertIn("TIMESTAMP_END", text)
        self.assertIn("plain line", text)

    def test_still_renders_the_full_series(self):
        """Degrading is fine; losing the data is not."""
        fig, ax = self._render(self.ta_unaligned, color='#FF0000')
        self.assertIsNone(self._linecollection(ax))
        self.assertEqual(len(fig.axes), 1)  # no colorbar
        self.assertEqual(len(ax.lines[0].get_ydata()), self.N)
        # The docstring now says so: `color` is *not* ignored in this branch.
        self.assertEqual(ax.lines[0].get_color(), '#FF0000')

    def test_partial_overlap_still_colours_and_stays_quiet(self):
        """A normal gappy driver must not trip the warning (L112/L113 interaction)."""
        ta_half = self.ta.copy()
        ta_half.iloc[100:] = np.nan
        _, ax = self._render(ta_half)
        self.assertIsNotNone(self._linecollection(ax))
        self.assertEqual(self.sink.lines, [])

    def test_no_colour_series_stays_quiet(self):
        self._render(None)
        self.assertEqual(self.sink.lines, [])

    def test_single_aligned_value_warns(self):
        """The guard needs two finite values; one is still a degenerate case."""
        ta_one = pd.Series(np.nan, index=self.idx, name='TA')
        ta_one.iloc[0] = 12.0
        self._render(ta_one)
        self.assertIn("1 of 200 records", self.sink.text)


class TestUnnamedSeriesLabels(unittest.TestCase):
    """L119: `plot_interactive()` raised on an unnamed Series; `plot()` did not.

    Bokeh rejects `legend_label=None` outright (`ValueError: legend_label value
    must be a string`), so the whole method died at the `p.line(...)` call. Its
    two siblings on the same input got through: `plot()` draws a blank-labelled
    line, and `plot_rangetool()` rendered the *literal string* "None" as its
    title, from `title=f"{self.series.name}"` — the same missing-name defect,
    non-fatal. Measured before the fix: interactive raised; rangetool titled
    'None'; `plot()` titled ''.
    """

    N = 20

    def setUp(self):
        idx = pd.date_range('2024-06-01', periods=self.N, freq='30min')
        self.unnamed = pd.Series(np.linspace(0, 5, self.N), index=idx)
        self.named = pd.Series(np.linspace(0, 5, self.N), index=idx, name='TA')
        self.assertIsNone(self.unnamed.name)
        # Bokeh's `show()` opens a browser tab; capture the figure instead. Only
        # `show()` writes the HTML file, so `output_file()` stays harmless.
        self._shown = []
        self._real_show = timeseries.show
        timeseries.show = lambda obj, *a, **kw: self._shown.append(obj)
        self.addCleanup(setattr, timeseries, 'show', self._real_show)

    def tearDown(self):
        plt.close('all')

    def _interactive(self, series, **kwargs):
        TimeSeries(series=series).plot_interactive(**kwargs)
        return self._shown[-1]

    def _rangetool_detail(self, series, **kwargs):
        TimeSeries(series=series).plot_rangetool(**kwargs)
        return self._shown[-1].children[0]  # column(detail, overview)

    @staticmethod
    def _legend_labels(fig):
        return [item.label.value for legend in fig.legend for item in legend.items]

    def test_unnamed_series_gets_a_string_legend_label(self):
        fig = self._interactive(self.unnamed)
        self.assertEqual(self._legend_labels(fig), ['value'])

    def test_named_series_keeps_its_own_legend_label(self):
        fig = self._interactive(self.named)
        self.assertEqual(self._legend_labels(fig), ['TA'])

    def test_unnamed_series_title_and_axis_are_not_the_string_none(self):
        fig = self._interactive(self.unnamed)
        self.assertEqual(fig.title.text, 'value')
        self.assertEqual(fig.yaxis[0].axis_label, 'value')

    def test_named_series_title_and_axis_unchanged(self):
        fig = self._interactive(self.named)
        self.assertEqual(fig.title.text, 'TA')
        self.assertEqual(fig.yaxis[0].axis_label, 'TA')

    def test_rangetool_unnamed_title_is_not_the_string_none(self):
        """The sibling defect: this one never raised, it just titled itself 'None'."""
        detail = self._rangetool_detail(self.unnamed)
        self.assertEqual(detail.title.text, 'value')
        self.assertEqual(detail.yaxis[0].axis_label, 'value')

    def test_rangetool_named_title_unchanged(self):
        detail = self._rangetool_detail(self.named)
        self.assertEqual(detail.title.text, 'TA')
        self.assertEqual(detail.yaxis[0].axis_label, 'TA')

    def test_saved_filename_uses_the_fallback_not_none(self):
        """`save_to_file=True` on an unnamed Series wrote 'None_interactive.html'."""
        self._interactive(self.unnamed, save_to_file=True)
        file_config = curstate().file
        self.assertEqual(file_config.filename, 'value_interactive.html')
        self.assertEqual(file_config.title, 'value')

    def test_saved_filename_of_a_named_series_unchanged(self):
        self._interactive(self.named, save_to_file=True)
        self.assertEqual(curstate().file.filename, 'TA_interactive.html')

    def test_unnamed_series_still_draws_a_blank_labelled_matplotlib_line(self):
        """`plot()` is the sibling the finding compares against: leave it alone."""
        fig, ax = plt.subplots()
        TimeSeries(series=self.unnamed).plot(ax=ax)
        self.assertEqual(ax.get_title(), '')
        self.assertEqual(ax.get_ylabel(), '')
        self.assertEqual(len(ax.lines[0].get_ydata()), self.N)


class TestColourByRespectsTheCallerAxes(_TimeSeriesRenderMixin, unittest.TestCase):
    """L136: colour-by replaced the caller's axes limits with its own data range.

    A `LineCollection` really does not rescale the view, but `set_xlim`/`set_ylim`
    is the wrong repair: it discards everything else on the axes. Measured before
    the fix, with a series already drawn at y=50: ylim (49.68, 50.29) -> (-7.15,
    9.50), i.e. the caller's line off-screen. `update_datalim` + `autoscale_view`
    unions instead. It also makes the lone colour-by plot agree with the plain
    path on x, which the explicit `set_xlim(nanmin, nanmax)` did not (no margin).
    """

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(3)
        self.other = pd.Series(50 + rng.normal(0, 0.1, self.N), index=self.idx, name='OTHER')

    def test_a_series_already_on_the_axes_stays_in_view(self):
        fig, ax = plt.subplots()
        ax.plot(self.other.index, self.other)
        TimeSeries(series=self.fc, color_series=self.ta).plot(ax=ax)
        fig.canvas.draw()
        lo, hi = ax.get_ylim()
        self.assertGreater(hi, float(self.other.max()), "the caller's line is above the view")
        self.assertLess(lo, float(self.fc.min()), "the colour-by line is below the view")

    def test_limits_the_caller_set_deliberately_are_not_touched(self):
        """`set_ylim` turns that axis' autoscaling off, so the view must survive."""
        fig, ax = plt.subplots()
        ax.set_ylim(-100.0, 100.0)
        TimeSeries(series=self.fc, color_series=self.ta).plot(ax=ax)
        fig.canvas.draw()
        self.assertEqual(ax.get_ylim(), (-100.0, 100.0))

    def test_lone_colour_plot_now_agrees_with_the_plain_path(self):
        """The deliberate change: x gained the standard 5% margin the plain path has.

        y was already identical (the old explicit pad was 5% too).
        """
        _, ax_colour = self._render(self.ta)
        fig, ax_plain = plt.subplots()
        TimeSeries(series=self.fc).plot(ax=ax_plain)
        fig.canvas.draw()
        self.assertEqual(ax_colour.get_xlim(), ax_plain.get_xlim())
        self.assertEqual(ax_colour.get_ylim(), ax_plain.get_ylim())

    def test_an_isolated_measured_record_is_inside_the_view(self):
        """A record no *segment* touches must still count towards the limits.

        `add_collection` contributes only the drawn segments, so a lone finite
        value between two gaps would fall outside the view — the plain path keeps
        it. This is what the explicit `update_datalim` is for.
        """
        vals = np.full(self.N, np.nan)
        vals[:2] = [0.0, 1.0]
        vals[100] = 99.0  # isolated: both neighbours are NaN, so no segment covers it
        fc_iso = pd.Series(vals, index=self.idx, name='FC')
        fig, ax = plt.subplots()
        TimeSeries(series=fc_iso, color_series=self.ta).plot(ax=ax)
        fig.canvas.draw()
        self.assertGreater(ax.get_ylim()[1], 99.0)


class TestSecondPlotOnTheSameAxes(_TimeSeriesRenderMixin, unittest.TestCase):
    """L137: a second `plot()` on one axes appends — now documented, not changed.

    Clearing would defeat the point of accepting an `ax` (compositing, which is
    exactly what L136 restores), and no sibling plot class clears a caller's
    axes. Measured: figure axes after 1/2/3 colour-by calls 2/3/4, main-axes
    width 0.651 -> 0.547 -> 0.459 as each colorbar takes its share.
    """

    def test_two_colour_by_calls_append_a_line_and_a_colorbar_each(self):
        fig, ax = plt.subplots()
        ts = TimeSeries(series=self.fc, color_series=self.ta)
        ts.plot(ax=ax)
        fig.canvas.draw()
        first_width = ax.get_position().width
        self.assertEqual((len(fig.axes), len(ax.collections)), (2, 1))
        ts.plot(ax=ax)
        fig.canvas.draw()
        self.assertEqual((len(fig.axes), len(ax.collections)), (3, 2))
        self.assertLess(ax.get_position().width, first_width)

    def test_show_colorbar_false_keeps_a_single_axes(self):
        """The documented escape hatch for the shrinking main axes."""
        fig, ax = plt.subplots()
        ts = TimeSeries(series=self.fc, color_series=self.ta)
        ts.plot(ax=ax, show_colorbar=False)
        ts.plot(ax=ax, show_colorbar=False)
        fig.canvas.draw()
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(ax.collections), 2)

    def test_overlaying_two_series_keeps_both_in_view(self):
        """What appending is *for*, and what L136 makes work.

        The second call used to set the limits to its own range, so the first
        series went off-screen.
        """
        far = self.fc + 50.0
        far.name = 'FC_FAR'
        fig, ax = plt.subplots()
        TimeSeries(series=far, color_series=self.ta).plot(ax=ax, show_colorbar=False)
        TimeSeries(series=self.fc, color_series=self.ta).plot(ax=ax, show_colorbar=False)
        fig.canvas.draw()
        lo, hi = ax.get_ylim()
        self.assertLess(lo, float(self.fc.min()))
        self.assertGreater(hi, float(far.max()))

    def test_clearing_the_axes_first_gives_a_single_line(self):
        """The documented way to re-draw instead of overlay."""
        fig, ax = plt.subplots()
        ts = TimeSeries(series=self.fc)
        ts.plot(ax=ax)
        ax.clear()
        ts.plot(ax=ax)
        fig.canvas.draw()
        # One data line plus the FormatStyle zero line, not two of each.
        self.assertEqual(len(ax.lines), 2)


class TestBokehShowplot(unittest.TestCase):
    """L144: both bokeh methods called `show()` unconditionally.

    There was no way to build the figure without opening a browser tab — the L119
    tests had to rebind the module-level `show` to test anything at all, which is
    what made the missing toggle worth a real control. `showplot=True` stays the
    default, matching the matplotlib classes' spelling.
    """

    N = 20

    def setUp(self):
        idx = pd.date_range('2024-06-01', periods=self.N, freq='30min')
        self.series = pd.Series(np.linspace(0, 5, self.N), index=idx, name='TA')
        self.ts = TimeSeries(series=self.series)
        self._shown = []
        self._real_show = timeseries.show
        timeseries.show = lambda obj, *a, **kw: self._shown.append(obj)
        self.addCleanup(setattr, timeseries, 'show', self._real_show)

    def test_interactive_still_shows_by_default(self):
        returned = self.ts.plot_interactive()
        self.assertEqual(len(self._shown), 1)
        self.assertIs(returned, self._shown[0])

    def test_rangetool_still_shows_by_default(self):
        returned = self.ts.plot_rangetool()
        self.assertEqual(len(self._shown), 1)
        self.assertIs(returned, self._shown[0])

    def test_interactive_showplot_false_builds_but_does_not_show(self):
        fig = self.ts.plot_interactive(showplot=False)
        self.assertEqual(self._shown, [])
        self.assertEqual(fig.title.text, 'TA')
        # A usable figure, not a stub: the line and the markers are both on it.
        self.assertEqual(len(fig.renderers), 2)

    def test_rangetool_showplot_false_builds_but_does_not_show(self):
        layout = self.ts.plot_rangetool(showplot=False)
        self.assertEqual(self._shown, [])
        detail, overview = layout.children
        self.assertEqual(detail.title.text, 'TA')
        self.assertEqual(len(overview.renderers), 1)


class TestOwnFigureLayout(_TimeSeriesRenderMixin, unittest.TestCase):
    """L138: `fig.tight_layout()` on a figure built with `layout='constrained'`.

    `pf.create_ax` asks for the constrained engine, so the `ax=None` path was
    replacing it on every call. The entry files this as the silent `UserWarning:
    The figure layout has changed to tight`, which is what the plain path does —
    but with a colorbar matplotlib refuses the swap outright, so the *default*
    colour-by call `TimeSeries(series, color_series=c).plot()` raised
    `RuntimeError: Colorbar layout of new layout engine not compatible with old
    engine` and drew nothing. One call site, one fix for both.
    """

    @staticmethod
    def _layout_warnings(caught):
        return [str(w.message) for w in caught if 'layout' in str(w.message).lower()]

    def test_colour_by_with_a_colorbar_renders_without_an_ax(self):
        """The crash: the default colour-by call on its own figure."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ax = TimeSeries(series=self.fc, color_series=self.ta).plot()
            ax.figure.canvas.draw()
        self.assertEqual(len(ax.figure.axes), 2)  # main axes + colorbar
        self.assertEqual(ax.figure.axes[1].get_ylabel(), 'TA')
        self.assertEqual(self._layout_warnings(caught), [])

    def test_plain_own_figure_no_longer_swaps_the_layout_engine(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ax = TimeSeries(series=self.fc).plot()
            ax.figure.canvas.draw()
        self.assertEqual(self._layout_warnings(caught), [])
        self.assertIsInstance(ax.figure.get_layout_engine(), ConstrainedLayoutEngine)

    def test_the_colorbar_gets_its_own_strip_beside_the_axes(self):
        """Not just "no crash": constrained must actually lay the colorbar out."""
        ax = TimeSeries(series=self.fc, color_series=self.ta).plot()
        ax.figure.canvas.draw()
        cb_x0 = ax.figure.axes[1].get_position().x0
        self.assertGreater(cb_x0, ax.get_position().x1)  # to the right, not overlapping
        self.assertLess(ax.get_position().x1, 1.0)  # main axes made room for it

    def test_a_caller_supplied_figure_is_still_left_alone(self):
        """The `ax=` path must not start touching the caller's figure either."""
        fig, ax = plt.subplots(layout='constrained', facecolor='#EEEEEE')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            TimeSeries(series=self.fc, color_series=self.ta).plot(ax=ax)
            fig.canvas.draw()
        self.assertEqual(self._layout_warnings(caught), [])
        self.assertIsInstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
        self.assertEqual(to_rgb(fig.get_facecolor()), to_rgb('#EEEEEE'))


if __name__ == "__main__":
    unittest.main()
