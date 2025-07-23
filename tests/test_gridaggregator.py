# File: tests/test_gridaggregator.py

import unittest

import numpy as np
import pandas as pd

from diive.pkgs.analyses.gridaggregator import GridAggregator


class TestGridAggregator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.x = pd.Series(np.random.rand(100), name="x_data")
        self.y = pd.Series(np.random.rand(100), name="y_data")
        self.z = pd.Series(np.random.rand(100), name="z_data")

    def test_quantile_binning(self):
        aggregator = GridAggregator(
            x=self.x,
            y=self.y,
            z=self.z,
            binning_type="quantiles",
            n_bins=5,
            min_n_vals_per_bin=2,
            aggfunc="mean"
        )
        agg_wide = aggregator.df_agg_wide
        self.assertIsInstance(agg_wide, pd.DataFrame)
        self.assertEqual(agg_wide.index.name, "BIN_y_data")
        self.assertEqual(agg_wide.columns.name, "BIN_x_data")
        self.assertEqual(list(agg_wide.columns), [0.0, 20.0, 40.0, 60.0, 80.0])
        self.assertEqual(list(agg_wide.index), [0.0, 20.0, 40.0, 60.0, 80.0])
        self.assertEqual(agg_wide.sum().sum(), 12.778815109306617)

    def test_equal_width_binning(self):
        aggregator = GridAggregator(
            x=self.x,
            y=self.y,
            z=self.z,
            binning_type="equal_width",
            n_bins=10,
            min_n_vals_per_bin=1,
            aggfunc="sum"
        )
        agg_long = aggregator.df_agg_long
        self.assertIsInstance(agg_long, pd.DataFrame)
        self.assertIn("BIN_x_data", agg_long.columns)
        self.assertIn("BIN_y_data", agg_long.columns)
        self.assertIn("z_data", agg_long.columns)
        self.assertEqual(agg_long.sum().sum(), 113.21846307472441)
        agg_wide = aggregator.df_agg_wide
        self.assertEqual(list(agg_wide.columns), [0.00552, 0.104, 0.202, 0.3, 0.398, 0.496, 0.594, 0.692, 0.791, 0.889])
        self.assertEqual(list(agg_wide.index), [0.00695, 0.105, 0.203, 0.301, 0.398, 0.496, 0.594, 0.692, 0.79, 0.888])

    def test_custom_binning(self):
        custom_x_bins = [0, 0.25, 0.5, 0.75, 1]
        custom_y_bins = [0, 0.3, 0.6, 1]
        aggregator = GridAggregator(
            x=self.x,
            y=self.y,
            z=self.z,
            binning_type="custom",
            custom_x_bins=custom_x_bins,
            custom_y_bins=custom_y_bins,
            min_n_vals_per_bin=1,
            aggfunc="max"
        )
        agg_wide = aggregator.df_agg_wide
        self.assertIsInstance(agg_wide, pd.DataFrame)
        self.assertEqual(len(agg_wide.index), len(custom_y_bins) - 1)
        self.assertEqual(len(agg_wide.columns), len(custom_x_bins) - 1)
        self.assertEqual(agg_wide.sum().sum(), 10.860205042306513)
        self.assertEqual(list(agg_wide.columns), [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(list(agg_wide.index), [0.0, 0.3, 0.6])
        # Check if the min/max x values were correctly assigned to lowest/highest bin (0.0 and 0.75, respectively)
        long = aggregator.df_long
        max_x = self.x.max()
        min_x = self.x.min()
        self.assertEqual(long.loc[long[self.x.name] == min_x, 'BIN_x_data'].iloc[0], 0.0)
        self.assertEqual(long.loc[long[self.x.name] == min_x, 'x_data'].iloc[0], min_x)
        self.assertEqual(long.loc[long[self.x.name] == max_x, 'BIN_x_data'].iloc[0], 0.75)
        self.assertEqual(long.loc[long[self.x.name] == max_x, 'x_data'].iloc[0], max_x)
        max_y = self.y.max()
        min_y = self.y.min()
        self.assertEqual(long.loc[long[self.y.name] == min_y, 'BIN_y_data'].iloc[0], 0.0)
        self.assertEqual(long.loc[long[self.y.name] == min_y, 'y_data'].iloc[0], min_y)
        self.assertEqual(long.loc[long[self.y.name] == max_y, 'BIN_y_data'].iloc[0], 0.6)
        self.assertEqual(long.loc[long[self.y.name] == max_y, 'y_data'].iloc[0], max_y)

    def test_empty_data(self):
        empty_x = pd.Series([], name="x_data", dtype=float)
        empty_y = pd.Series([], name="y_data", dtype=float)
        empty_z = pd.Series([], name="z_data", dtype=float)
        aggregator = GridAggregator(
            x=empty_x,
            y=empty_y,
            z=empty_z,
            binning_type="quantiles",
            n_bins=3,
            min_n_vals_per_bin=1,
            aggfunc="mean"
        )
        with self.assertRaises(AttributeError):
            _ = aggregator.df_agg_wide

    def test_aggregation_function_count(self):
        aggregator = GridAggregator(
            x=self.x,
            y=self.y,
            z=self.z,
            binning_type="equal_width",
            n_bins=5,
            min_n_vals_per_bin=1,
            aggfunc="count"
        )
        agg_wide = aggregator.df_agg_wide
        self.assertIsInstance(agg_wide, pd.DataFrame)
        self.assertGreaterEqual(agg_wide.sum().sum(), 0)

    def test_invalid_binning_type(self):
        with self.assertRaises(ValueError):
            _ = GridAggregator(
                x=self.x,
                y=self.y,
                z=self.z,
                binning_type="invalid_type",
                n_bins=3
            )
