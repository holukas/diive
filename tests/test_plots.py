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

    # --- cumulative / waterfall ---

    def test_cumulative_ends_at_the_series_sum(self):
        # The defining contract of a running total.
        from diive.core.plotting.cumulative import Cumulative
        fig, ax = plt.subplots()
        Cumulative(self.series.to_frame(), units="units").plot(ax=ax, showplot=False)
        self.assertAlmostEqual(float(ax.lines[0].get_ydata()[-1]),
                               float(self.series.sum()), places=3)
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
        self.assertEqual(len(ax.patches), len(monthly))
        # The running budget closes on the series total.
        tops = [p.get_y() + p.get_height() for p in ax.patches]
        bottoms = [p.get_y() for p in ax.patches]
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
        colors = {p.get_facecolor()[:3] for p in ax.patches}
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
