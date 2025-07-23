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


if __name__ == "__main__":
    unittest.main()
