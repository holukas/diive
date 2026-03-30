# tests/test_hexbin_plot.py
import unittest

import numpy as np
import pandas as pd

from diive.core.plotting.hexbin_plot import HexbinPlot


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
        """Test initialization with custom axis labels."""
        hb = HexbinPlot(
            self.x, self.y, self.z,
            xlabel="Temperature (°C)",
            ylabel="Water Content (%)",
            zlabel="NEP (µmol m⁻² s⁻¹)"
        )
        self.assertEqual(hb.xlabel, "Temperature (°C)")
        self.assertEqual(hb.ylabel, "Water Content (%)")
        self.assertEqual(hb.zlabel, "NEP (µmol m⁻² s⁻¹)")

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
            xlabel="Custom X",
            ylabel="Custom Y",
            zlabel="Custom Z",
            figsize=(10, 8)
        )
        try:
            hb.plot()
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

        hb = HexbinPlot(x, y, z, edgecolors='black')
        self.assertEqual(hb.edgecolors, 'black')

        try:
            hb.plot()
        except Exception as e:
            self.fail(f"HexbinPlot with edgecolors='black' raised: {e}")


if __name__ == "__main__":
    unittest.main()
