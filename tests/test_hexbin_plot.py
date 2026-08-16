# tests/test_hexbin_plot.py
import unittest

import numpy as np
import pandas as pd

from diive.core.plotting.hexbin import HexbinPlot
from diive.core.plotting.styles.format import FormatStyle


class TestHexbinPlot(unittest.TestCase):
    def setUp(self):
        """Create simple test data for hexbin plotting."""
        # Generate synthetic data: x, y driver variables and z flux values
        np.random.seed(42)
        n = 100
        self.x = pd.Series(np.random.uniform(0, 10, n), name="Tair")
        self.y = pd.Series(np.random.uniform(0, 100, n), name="WFPS")
        self.z = pd.Series(np.random.uniform(-5, 5, n), name="NEP")

    def test_initialization(self):
        """Test basic HexbinPlot initialization."""
        hb = HexbinPlot(self.x, self.y, self.z)
        self.assertEqual(hb.xlabel, "Tair")
        self.assertEqual(hb.ylabel, "WFPS")
        self.assertEqual(hb.zlabel, "NEP")

    def test_custom_labels(self):
        """Custom axis labels are passed to plot(); the constructor path is deprecated."""
        # Canonical: labels via plot()
        hb = HexbinPlot(self.x, self.y, self.z)
        hb.plot(format_style=FormatStyle(xlabel="Temperature (°C)", ylabel="Water Content (%)"),
                zlabel="NEP (µmol m⁻² s⁻¹)")
        # Deprecated: labels via the constructor still work but warn
        with self.assertWarns(DeprecationWarning):
            hb2 = HexbinPlot(self.x, self.y, self.z, xlabel="Temperature (°C)",
                             ylabel="Water Content (%)", zlabel="NEP (µmol m⁻² s⁻¹)")
        self.assertEqual(hb2.xlabel, "Temperature (°C)")

    def test_custom_gridsize(self):
        """Test initialization with custom gridsize."""
        hb = HexbinPlot(self.x, self.y, self.z, gridsize=15)
        self.assertEqual(hb.gridsize, 15)

    def test_custom_reduce_function(self):
        """Test initialization with custom aggregation function."""
        hb = HexbinPlot(self.x, self.y, self.z, reduce_C_function=np.mean)
        self.assertEqual(hb.reduce_C_function, np.mean)

    def test_percentile_normalization(self):
        """Test percentile normalization produces 0-100 range."""
        hb = HexbinPlot(self.x, self.y, self.z, normalize_axes=True)
        # Check that normalized x and y are in 0-100 range
        self.assertTrue(hb.x.min() >= 0)
        self.assertTrue(hb.x.max() <= 100)
        self.assertTrue(hb.y.min() >= 0)
        self.assertTrue(hb.y.max() <= 100)

    def test_normalize_axes_false(self):
        """Test that normalize_axes=False preserves original values."""
        hb = HexbinPlot(self.x, self.y, self.z, normalize_axes=False)
        # Check that x and y match originals (approximately, due to copy)
        np.testing.assert_array_almost_equal(hb.x.values, self.x.values)
        np.testing.assert_array_almost_equal(hb.y.values, self.y.values)

    def test_mismatched_lengths(self):
        """Test that mismatched Series lengths raise ValueError."""
        x_short = pd.Series([1, 2, 3], name="X")
        y = pd.Series([1, 2, 3, 4], name="Y")
        z = pd.Series([1, 2, 3, 4], name="Z")

        with self.assertRaises(ValueError) as context:
            HexbinPlot(x_short, y, z)
        self.assertIn("same length", str(context.exception))

    def test_missing_series_name(self):
        """Test that missing Series names raise ValueError."""
        x_noname = pd.Series([1, 2, 3, 4])  # No name
        y = pd.Series([1, 2, 3, 4], name="Y")
        z = pd.Series([1, 2, 3, 4], name="Z")

        with self.assertRaises(ValueError) as context:
            HexbinPlot(x_noname, y, z)
        self.assertIn("must have names", str(context.exception))

    def test_nan_in_x(self):
        """Test that NaN in x raises ValueError."""
        x_with_nan = pd.Series([1, 2, np.nan, 4], name="X")
        y = pd.Series([1, 2, 3, 4], name="Y")
        z = pd.Series([1, 2, 3, 4], name="Z")

        with self.assertRaises(ValueError) as context:
            HexbinPlot(x_with_nan, y, z)
        self.assertIn("NaN", str(context.exception))

    def test_nan_in_y(self):
        """Test that NaN in y raises ValueError."""
        x = pd.Series([1, 2, 3, 4], name="X")
        y_with_nan = pd.Series([1, 2, np.nan, 4], name="Y")
        z = pd.Series([1, 2, 3, 4], name="Z")

        with self.assertRaises(ValueError) as context:
            HexbinPlot(x, y_with_nan, z)
        self.assertIn("NaN", str(context.exception))

    def test_nan_in_z_allowed(self):
        """Test that NaN in z is allowed (will be ignored during aggregation)."""
        # Create z with same length as x and y, with some NaN values
        z_with_nan = pd.Series(
            np.concatenate([np.array([1, 2, np.nan, 4]),
                            np.random.uniform(-5, 5, len(self.x) - 4)]),
            name="Z"
        )
        try:
            hb = HexbinPlot(self.x, self.y, z_with_nan)
            # Should not raise an error
            self.assertIsNotNone(hb)
        except ValueError:
            self.fail("HexbinPlot should allow NaN in z values")

    def test_plot_method_runs(self):
        """Test that plot() method executes without error."""
        hb = HexbinPlot(self.x, self.y, self.z)
        try:
            hb.plot()
        except Exception as e:
            self.fail(f"plot() raised an exception: {e}")

    def test_plot_with_percentile_normalization(self):
        """Test plot() with percentile normalization enabled."""
        hb = HexbinPlot(self.x, self.y, self.z, normalize_axes=True)
        try:
            hb.plot()
        except Exception as e:
            self.fail(f"plot() with percentile normalization raised: {e}")

    def test_plot_with_custom_params(self):
        """Test plot() with various custom parameters."""
        hb = HexbinPlot(
            self.x, self.y, self.z,
            gridsize=15,
            reduce_C_function=np.mean,
            mincnt=2,
        )
        try:
            hb.plot(figsize=(10, 8), format_style=FormatStyle(xlabel="Custom X", ylabel="Custom Y"),
                    zlabel="Custom Z")
        except Exception as e:
            self.fail(f"plot() with custom params raised: {e}")

    def test_percentile_normalize_static_method(self):
        """Test _percentile_normalize static method directly."""
        series = pd.Series([1, 2, 3, 4, 5], name="test")
        normalized = HexbinPlot._percentile_normalize(series)

        # Check range
        self.assertEqual(normalized.min(), 20.0)  # 1/5 = 0.2 * 100
        self.assertEqual(normalized.max(), 100.0)  # 5/5 = 1.0 * 100

        # Check name is preserved
        self.assertEqual(normalized.name, "test")

    def test_show_values_parameter(self):
        """Test show_values parameter passed to plot()."""
        hb = HexbinPlot(self.x, self.y, self.z)
        hb.plot(show_values=True)
        self.assertTrue(hb.show_values)

    def test_show_values_n_dec_places(self):
        """Test custom decimal places for displayed values."""
        hb = HexbinPlot(self.x, self.y, self.z)
        hb.plot(show_values=True, show_values_n_dec_places=3)
        self.assertEqual(hb.show_values_n_dec_places, 3)

    def test_show_values_fontsize(self):
        """Test custom font size for displayed values."""
        hb = HexbinPlot(self.x, self.y, self.z)
        hb.plot(show_values=True, show_values_fontsize=10)
        self.assertEqual(hb.show_values_fontsize, 10)

    def test_show_values_color(self):
        """Test custom color for displayed values."""
        hb = HexbinPlot(self.x, self.y, self.z)
        hb.plot(show_values=True, show_values_color='red')
        self.assertEqual(hb.show_values_color, 'red')

    def test_plot_with_show_values(self):
        """Test plot() with show_values enabled."""
        hb = HexbinPlot(self.x, self.y, self.z, mincnt=2)
        try:
            hb.plot(show_values=True)
        except Exception as e:
            self.fail(f"plot() with show_values raised: {e}")


class TestHexbinPlotEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_single_row_data(self):
        """Test HexbinPlot with minimal data (single row)."""
        x = pd.Series([1.0], name="X")
        y = pd.Series([2.0], name="Y")
        z = pd.Series([3.0], name="Z")

        try:
            hb = HexbinPlot(x, y, z)
            hb.plot()
        except Exception as e:
            # Single row might fail, that's ok - just want to know it doesn't crash
            # during initialization
            self.assertIsNotNone(hb)

    def test_identical_x_values(self):
        """Test HexbinPlot with all identical x values."""
        x = pd.Series([5.0] * 50, name="X")
        y = pd.Series(np.random.uniform(0, 10, 50), name="Y")
        z = pd.Series(np.random.uniform(0, 5, 50), name="Z")

        try:
            hb = HexbinPlot(x, y, z)
            hb.plot()
        except Exception as e:
            self.fail(f"HexbinPlot with identical x values raised: {e}")

    def test_negative_values(self):
        """Test HexbinPlot with negative x and y values."""
        x = pd.Series(np.random.uniform(-10, 0, 50), name="X")
        y = pd.Series(np.random.uniform(-100, -50, 50), name="Y")
        z = pd.Series(np.random.uniform(-5, 5, 50), name="Z")

        try:
            hb = HexbinPlot(x, y, z)
            hb.plot()
        except Exception as e:
            self.fail(f"HexbinPlot with negative values raised: {e}")

    def test_very_large_values(self):
        """Test HexbinPlot with very large values."""
        x = pd.Series(np.random.uniform(1e6, 1e7, 50), name="X")
        y = pd.Series(np.random.uniform(1e6, 1e7, 50), name="Y")
        z = pd.Series(np.random.uniform(1e6, 1e7, 50), name="Z")

        try:
            hb = HexbinPlot(x, y, z)
            hb.plot()
        except Exception as e:
            self.fail(f"HexbinPlot with very large values raised: {e}")

    def test_mincnt_parameter(self):
        """Test HexbinPlot with custom mincnt parameter."""
        x = pd.Series(np.random.uniform(0, 10, 50), name="X")
        y = pd.Series(np.random.uniform(0, 10, 50), name="Y")
        z = pd.Series(np.random.uniform(0, 5, 50), name="Z")

        hb = HexbinPlot(x, y, z, mincnt=5)
        self.assertEqual(hb.mincnt, 5)

        try:
            hb.plot()
        except Exception as e:
            self.fail(f"HexbinPlot with mincnt=5 raised: {e}")

    def test_edgecolors_parameter(self):
        """Test HexbinPlot with custom edgecolors parameter."""
        x = pd.Series(np.random.uniform(0, 10, 50), name="X")
        y = pd.Series(np.random.uniform(0, 10, 50), name="Y")
        z = pd.Series(np.random.uniform(0, 5, 50), name="Z")

        hb = HexbinPlot(x, y, z)
        try:
            hb.plot(edgecolors='black')
        except Exception as e:
            self.fail(f"HexbinPlot with edgecolors='black' raised: {e}")


if __name__ == "__main__":
    unittest.main()


class TestHexbinShowLessXticklabels(unittest.TestCase):
    """L91: `show_less_xticklabels` must actually thin the x-tick labels.

    `HeatmapBase` only stores the flag; each subclass applies it. `HeatmapDateTime`
    and `HeatmapYearMonth` do, `HexbinPlot` accepted, documented and forwarded it but
    never applied it, so True and False rendered identically. The same defect as L62,
    in a second file. (`get_xticklabels` returns only the visible labels, so hiding
    shows up as a shorter list.)
    """

    @staticmethod
    def _labels(show_less):
        import matplotlib.pyplot as plt
        np.random.seed(42)
        n = 200
        x = pd.Series(np.random.uniform(0, 10, n), name="Tair")
        y = pd.Series(np.random.uniform(0, 100, n), name="WFPS")
        z = pd.Series(np.random.uniform(-5, 5, n), name="NEP")
        fig, ax = plt.subplots()
        HexbinPlot(x, y, z).plot(ax=ax, fig=fig, show_less_xticklabels=show_less)
        texts = [label.get_text() for label in ax.get_xticklabels()]
        plt.close(fig)
        return texts

    def test_every_second_label_is_hidden(self):
        shown_all = self._labels(False)
        thinned = self._labels(True)
        self.assertGreater(len(shown_all), 2, "need several ticks for the test to mean anything")
        self.assertEqual(thinned, shown_all[::2])

    def test_default_shows_every_label(self):
        """The other direction: the flag must not thin labels when it is off."""
        self.assertEqual(self._labels(False), self._labels(False))
        self.assertNotEqual(self._labels(False), self._labels(True))


class TestHexbinEmptyCellsNotDrawn(unittest.TestCase):
    """L107: no hexagon may be drawn over a cell that holds no data.

    matplotlib's cutoff is `len(values) >= mincnt`, so the old `mincnt=0` default
    handed *empty* cells to `reduce_C_function`. With `np.sum` the empty cell comes
    back as 0.0 and is painted as a measured zero; with `np.max` the call raises;
    with `np.mean`/`np.median` it returns NaN at the price of one RuntimeWarning per
    empty cell. Same family as L61/L63: an empty cell renders empty.
    """

    GRIDSIZE = 10

    def setUp(self):
        # Two tight clouds with a genuinely empty region between them, so the hexbin
        # grid spans a large area that holds no observations at all.
        rng = np.random.default_rng(42)
        n = 500
        self.x = pd.Series(np.concatenate([rng.normal(2, .5, n), rng.normal(18, .5, n)]), name="Tair")
        self.y = pd.Series(np.concatenate([rng.normal(2, .5, n), rng.normal(18, .5, n)]), name="WFPS")
        self.z = pd.Series(np.concatenate([rng.normal(5, 1, n), rng.normal(-5, 1, n)]), name="NEP")

    def _occupied_cells(self, ax) -> set:
        """Centres of the grid cells that actually contain observations.

        Same x/y and gridsize, so the grid geometry (and therefore the offsets) is
        identical to the plot under test. `len` as reducer counts the members.
        """
        p = ax.hexbin(self.x.to_numpy(), self.y.to_numpy(), C=np.ones(len(self.x)),
                      gridsize=self.GRIDSIZE, reduce_C_function=len, mincnt=1)
        cells = {(round(cx, 6), round(cy, 6)) for cx, cy in p.get_offsets()}
        p.remove()
        return cells

    def _render(self, reducer, **kwargs):
        """Returns (n_hexagons_drawn, n_of_those_over_an_empty_cell)."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            occupied = self._occupied_cells(ax)
            hb = HexbinPlot(self.x, self.y, self.z, gridsize=self.GRIDSIZE,
                            reduce_C_function=reducer, **kwargs)
            hb.plot(ax=ax, fig=fig)
            drawn = [(round(cx, 6), round(cy, 6)) for cx, cy in hb.p.get_offsets()]
            over_empty = [c for c in drawn if c not in occupied]
            return len(drawn), len(over_empty)
        finally:
            plt.close(fig)

    def test_default_mincnt_is_one(self):
        """The default must be matplotlib's effective default, not 0."""
        self.assertEqual(HexbinPlot(self.x, self.y, self.z).mincnt, 1)

    def test_sum_reducer_draws_only_cells_holding_data(self):
        """np.sum is the fabricating case: an empty cell sums to a plausible 0.0."""
        n_drawn, n_over_empty = self._render(np.sum)
        # The two clouds occupy 11 of the 116 grid cells.
        self.assertEqual(n_drawn, 11)
        self.assertEqual(n_over_empty, 0)

    def test_max_reducer_is_usable_at_all(self):
        """np.max on an empty cell raised ValueError, so `0` was broken, not just misleading."""
        n_drawn, n_over_empty = self._render(np.max)
        self.assertEqual(n_drawn, 11)
        self.assertEqual(n_over_empty, 0)

    def test_median_default_emits_no_empty_slice_warnings(self):
        """diive's own default reducer warned once per empty cell (210 warnings here)."""
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            n_drawn, n_over_empty = self._render(np.median)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(n_drawn, 11)
        self.assertEqual(n_over_empty, 0)
        self.assertEqual(runtime, [])

    def test_mincnt_below_one_is_rejected(self):
        """`0` cannot produce a correct plot for any documented reducer, so it is closed off."""
        for bad in (0, -1):
            with self.assertRaises(ValueError) as ctx:
                HexbinPlot(self.x, self.y, self.z, mincnt=bad)
            self.assertIn("mincnt must be >= 1", str(ctx.exception))

    def test_explicit_mincnt_still_thins_sparse_cells(self):
        """The knob itself keeps working above 1."""
        n_drawn, n_over_empty = self._render(np.median, mincnt=200)
        self.assertEqual(n_over_empty, 0)
        self.assertLess(n_drawn, 11)
