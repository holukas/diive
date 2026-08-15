# File: tests/test_gridaggregator.py

import unittest

import numpy as np
import pandas as pd

from diive.analysis.gridaggregator import GridAggregator


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


class TestSharedSeriesNames(unittest.TestCase):
    """Two of x/y/z may carry the same Series name — the aggregation must not care.

    The working frame used to be keyed by the Series names, so a shared name kept
    only one of the two roles: x was then binned on z's values (silently wrong
    axis), and x and y sharing a name made `pivot_table` raise
    `Grouper for 'BIN_...' not 1-dimensional`.
    """

    @staticmethod
    def _series():
        rng = np.random.RandomState(7)
        # Three disjoint value ranges, so which role a bin label belongs to is visible.
        return (pd.Series(rng.uniform(0, 10, 400)),
                pd.Series(rng.uniform(100, 200, 400)),
                pd.Series(rng.randn(400)))

    @staticmethod
    def _agg(x, y, z):
        return GridAggregator(x=x, y=y, z=z, binning_type='equal_width', n_bins=4,
                              aggfunc='mean')

    def test_same_variable_as_x_and_z(self):
        # The GUI's X/Y/Z surface lets one variable fill two roles. Harmless even
        # before the fix (overwriting a column with itself changes nothing), so
        # this documents the case rather than carrying the regression.
        x, y, _ = self._series()
        shared = self._agg(x.rename('A'), y.rename('B'), x.rename('A'))
        # Bin labels are the (rounded) lower bin edges: x on the columns, y on the index.
        self.assertLess(shared.df_agg_wide.columns.max(), 10)
        self.assertGreater(shared.df_agg_wide.index.min(), 99)
        # Aggregating x over x's own bins: cell means rise with the x bin.
        means = shared.df_agg_wide.mean(axis=0).to_numpy()
        self.assertTrue(np.all(np.diff(means) > 0))

    def test_z_sharing_the_x_name_is_still_aggregated_as_z(self):
        x, y, z = self._series()
        shared = self._agg(x.rename('A'), y.rename('B'), z.rename('A'))
        reference = self._agg(x.rename('A'), y.rename('B'), z.rename('Z'))
        # Renaming a role must not move a single value or bin label.
        np.testing.assert_allclose(shared.df_agg_wide.to_numpy(),
                                   reference.df_agg_wide.to_numpy())
        self.assertEqual(list(shared.df_agg_wide.columns), list(reference.df_agg_wide.columns))
        self.assertEqual(list(shared.df_agg_wide.index), list(reference.df_agg_wide.index))
        # The x axis carries x's range (0-10), not z's (standard normal).
        self.assertGreater(shared.df_agg_wide.columns.max(), 5)

    def test_x_and_y_sharing_a_name(self):
        x, y, z = self._series()
        shared = self._agg(x.rename('A'), y.rename('A'), z.rename('Z'))
        reference = self._agg(x.rename('A'), y.rename('B'), z.rename('Z'))
        np.testing.assert_allclose(shared.df_agg_wide.to_numpy(),
                                   reference.df_agg_wide.to_numpy())
        # x on the columns (0-10), y on the index (100-200) — not y twice.
        self.assertLess(shared.df_agg_wide.columns.max(), 10)
        self.assertGreater(shared.df_agg_wide.index.min(), 100)

    def test_public_column_names_are_unchanged(self):
        x, y, z = self._series()
        ga = self._agg(x.rename('A'), y.rename('B'), z.rename('Z'))
        self.assertEqual(ga.df_agg_wide.columns.name, 'BIN_A')
        self.assertEqual(ga.df_agg_wide.index.name, 'BIN_B')
        self.assertEqual(list(ga.df_agg_long.columns), ['BIN_B', 'BIN_A', 'Z'])
        self.assertEqual(list(ga.df_long.columns),
                         ['INDEX', 'A', 'B', 'Z', 'BIN_A', 'BIN_B', 'BIN_COMBINED_STR'])


class TestEmptyBinsArePreserved(unittest.TestCase):
    """A bin nothing fell into must survive as an empty (NaN) cell.

    The pivot only emits occupied bins. Consumers that draw the grid (the x/y/z
    heatmap, the 3-D surface) treat consecutive labels as adjacent cells, so a
    dropped bin lets the cell beside it widen silently across the gap — painting
    a region that holds no measurements.
    """

    def _bimodal(self, n_bins):
        rng = np.random.RandomState(0)
        # X is bimodal: 0-10 and 95-100, with nothing in between.
        x = pd.Series(np.concatenate([rng.uniform(0, 10, 500),
                                      rng.uniform(95, 100, 500)]), name='X')
        y = pd.Series(rng.uniform(0, 10, 1000), name='Y')
        z = pd.Series(rng.randn(1000), name='Z')
        return GridAggregator(x=x, y=y, z=z, binning_type='equal_width', n_bins=n_bins)

    def test_equal_width_keeps_the_empty_middle(self):
        ga = self._bimodal(n_bins=30)
        wide = ga.df_agg_wide
        self.assertEqual(wide.shape, (30, 30), "every requested bin must be present")
        occupied = ~wide.isna().all(axis=0)
        self.assertEqual(int(occupied.sum()), 5, "only the two clusters hold data")
        self.assertGreater(int((~occupied).sum()), 0, "the gap must remain as empty cells")

    def test_equal_width_bins_are_evenly_spaced(self):
        # The whole point: consumers space cells by their labels, so the labels
        # must march uniformly rather than jump across a dropped bin.
        widths = np.diff(self._bimodal(n_bins=30).df_agg_wide.columns.to_numpy())
        self.assertTrue(np.allclose(widths, widths[0], atol=1e-3))

    def test_the_data_itself_is_unchanged(self):
        ga = self._bimodal(n_bins=30)
        # Restoring empty bins must not add, drop or move any aggregated value.
        self.assertEqual(int(ga.df_agg_wide.notna().to_numpy().sum()), 150)

    def test_quantile_bins_emptied_by_min_n_vals_are_kept(self):
        # min_n_vals_per_bin can empty a bin of any type, not just equal-width.
        rng = np.random.RandomState(0)
        x = pd.Series(rng.uniform(0, 10, 400), name='X')
        y = pd.Series(rng.uniform(0, 10, 400), name='Y')
        z = pd.Series(rng.randn(400), name='Z')
        ga = GridAggregator(x=x, y=y, z=z, binning_type='quantiles', n_bins=10,
                            min_n_vals_per_bin=5)
        self.assertEqual(ga.df_agg_wide.shape, (10, 10))
