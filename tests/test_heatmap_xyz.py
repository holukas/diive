# tests/test_heatmap_xyz.py
import unittest

import numpy as np
import pandas as pd

from diive.core.plotting.heatmap_xyz import HeatmapXYZ


class TestHeatmapXYZ(unittest.TestCase):
    def setUp(self):
        self.x = pd.Series(np.array([1, 2, 1, 2]), name="X")
        self.y = pd.Series(np.array([1, 1, 2, 2]), name="Y")
        self.z = pd.Series(np.array([10, 20, 30, 40]), name="Z")

    def test_initialization(self):
        heatmap = HeatmapXYZ(self.x, self.y, self.z, xlabel="Custom X", ylabel="Custom Y", zlabel="Custom Z")
        self.assertEqual(heatmap.xlabel, "Custom X")
        self.assertEqual(heatmap.ylabel, "Custom Y")
        self.assertEqual(heatmap.zlabel, "Custom Z")

    def test_default_labels(self):
        heatmap = HeatmapXYZ(self.x, self.y, self.z)
        self.assertEqual(heatmap.xlabel, "X")
        self.assertEqual(heatmap.ylabel, "Y")
        self.assertEqual(heatmap.zlabel, "Z")

    def test_prepare_data(self):
        heatmap = HeatmapXYZ(self.x, self.y, self.z)
        expected_x_edges = np.array([1, 2, 3])  # Calculated edges for x
        expected_y_edges = np.array([1, 2, 3])  # Calculated edges for y
        expected_z_values = np.array([[10, 20], [30, 40]])  # Pivoted z values

        np.testing.assert_array_equal(heatmap.x, expected_x_edges)
        np.testing.assert_array_equal(heatmap.y, expected_y_edges)
        np.testing.assert_array_equal(heatmap.z, expected_z_values)

    def test_xticks_yticks(self):
        heatmap = HeatmapXYZ(self.x, self.y, self.z, xtickpos=[1.5, 2.5], xticklabels=["A", "B"], ytickpos=[1.5],
                             yticklabels=["C"])
        self.assertEqual(heatmap.xtickpos, [1.5, 2.5])
        self.assertEqual(heatmap.xticklabels, ["A", "B"])
        self.assertEqual(heatmap.ytickpos, [1.5])
        self.assertEqual(heatmap.yticklabels, ["C"])

    def test_plot(self):
        heatmap = HeatmapXYZ(self.x, self.y, self.z)
        try:
            heatmap.plot()
        except Exception as e:
            self.fail(f"plot() raised an exception: {e}")

    def test_integration_with_gridaggregator(self):
        """Integration test: verify HeatmapXYZ with pre-aggregated GridAggregator output.

        This test ensures that different aggregation functions (mean vs std) produce
        different heatmap values. Previously, HeatmapXYZ would silently re-aggregate
        using mean regardless of the original aggfunc parameter.
        """
        try:
            import diive as dv
        except ImportError:
            self.skipTest("diive not available for integration test")

        # Create simple test data with duplicates (multiple values per bin)
        # This data will be binned into a 2x2 grid
        x_vals = [1.0, 1.5, 2.5, 3.0, 1.0, 1.5, 2.5, 3.0]
        y_vals = [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]
        z_vals = [100.0, 110.0, 200.0, 210.0, 300.0, 310.0, 400.0, 410.0]

        x = pd.Series(x_vals, name="X")
        y = pd.Series(y_vals, name="Y")
        z = pd.Series(z_vals, name="Z")

        # Aggregate with mean
        q_mean = dv.ga(x=x, y=y, z=z, binning_type='quantiles', n_bins=2,
                       min_n_vals_per_bin=1, aggfunc='mean')
        df_mean = q_mean.df_agg_long

        # Aggregate with std
        q_std = dv.ga(x=x, y=y, z=z, binning_type='quantiles', n_bins=2,
                      min_n_vals_per_bin=1, aggfunc='std')
        df_std = q_std.df_agg_long

        # Create heatmaps
        hm_mean = HeatmapXYZ(x=df_mean['BIN_X'], y=df_mean['BIN_Y'], z=df_mean['Z'])
        hm_std = HeatmapXYZ(x=df_std['BIN_X'], y=df_std['BIN_Y'], z=df_std['Z'])

        # Extract z values
        z_mean = hm_mean.z
        z_std = hm_std.z

        # Verify that mean and std produce different results
        # If HeatmapXYZ were re-aggregating with mean, both would be identical
        self.assertFalse(np.allclose(z_mean, z_std),
                         msg="Mean and std aggregations should produce different z values")


class TestHeatmapXYZFromGridAggregator(unittest.TestCase):
    """Tests for HeatmapXYZ.from_gridaggregator() class method."""

    def setUp(self):
        try:
            import diive as dv
            self.dv = dv
        except ImportError:
            self.skipTest("diive not available")

    def test_from_gridaggregator_basic(self):
        """Test from_gridaggregator creates valid heatmap from GridAggregator output."""
        # Create simple test data
        x = pd.Series([1.0, 1.5, 2.5, 3.0, 1.0, 1.5, 2.5, 3.0], name="X")
        y = pd.Series([10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0], name="Y")
        z = pd.Series([100.0, 110.0, 200.0, 210.0, 300.0, 310.0, 400.0, 410.0], name="Z")

        # Create GridAggregator
        q = self.dv.ga(x=x, y=y, z=z, binning_type='quantiles', n_bins=2,
                       min_n_vals_per_bin=1, aggfunc='mean')

        # Create heatmap using class method
        hm = HeatmapXYZ.from_gridaggregator(q, 'X', 'Y', 'Z')

        # Verify heatmap was created and has correct labels
        self.assertEqual(hm.xlabel, 'X')
        self.assertEqual(hm.ylabel, 'Y')
        self.assertEqual(hm.zlabel, 'Z')
        self.assertIsNotNone(hm.z)

    def test_from_gridaggregator_with_custom_labels(self):
        """Test from_gridaggregator respects custom axis labels."""
        x = pd.Series([1.0, 1.5, 2.5, 3.0], name="X")
        y = pd.Series([10.0, 10.0, 20.0, 20.0], name="Y")
        z = pd.Series([100.0, 110.0, 300.0, 310.0], name="Z")

        q = self.dv.ga(x=x, y=y, z=z, binning_type='quantiles', n_bins=2,
                       min_n_vals_per_bin=1, aggfunc='mean')

        hm = HeatmapXYZ.from_gridaggregator(
            q, 'X', 'Y', 'Z',
            xlabel='Custom X Label',
            ylabel='Custom Y Label',
            zlabel='Custom Z Label'
        )

        self.assertEqual(hm.xlabel, 'Custom X Label')
        self.assertEqual(hm.ylabel, 'Custom Y Label')
        self.assertEqual(hm.zlabel, 'Custom Z Label')

    def test_from_gridaggregator_missing_column(self):
        """Test from_gridaggregator raises KeyError for invalid column name."""
        x = pd.Series([1.0, 1.5, 2.5, 3.0], name="X")
        y = pd.Series([10.0, 10.0, 20.0, 20.0], name="Y")
        z = pd.Series([100.0, 110.0, 300.0, 310.0], name="Z")

        q = self.dv.ga(x=x, y=y, z=z, binning_type='quantiles', n_bins=2,
                       min_n_vals_per_bin=1, aggfunc='mean')

        # Try to create heatmap with non-existent column name
        with self.assertRaises(KeyError):
            HeatmapXYZ.from_gridaggregator(q, 'NonExistent', 'Y', 'Z')

    def test_from_gridaggregator_equivalence(self):
        """Test from_gridaggregator produces same result as manual extraction."""
        x = pd.Series([1.0, 1.5, 2.5, 3.0, 1.0, 1.5, 2.5, 3.0], name="X")
        y = pd.Series([10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0], name="Y")
        z = pd.Series([100.0, 110.0, 200.0, 210.0, 300.0, 310.0, 400.0, 410.0], name="Z")

        q = self.dv.ga(x=x, y=y, z=z, binning_type='quantiles', n_bins=2,
                       min_n_vals_per_bin=1, aggfunc='mean')

        # Create heatmap using class method
        hm_from_method = HeatmapXYZ.from_gridaggregator(q, 'X', 'Y', 'Z')

        # Create heatmap using manual extraction
        df_agg = q.df_agg_long
        hm_manual = HeatmapXYZ(
            x=df_agg['BIN_X'],
            y=df_agg['BIN_Y'],
            z=df_agg['Z']
        )

        # Verify z-values are identical
        np.testing.assert_array_equal(hm_from_method.z, hm_manual.z)


if __name__ == "__main__":
    unittest.main()
