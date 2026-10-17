import unittest

from diive.fits.fitter import BinFitterCP


class TestFits(unittest.TestCase):

    def test_binfittercp(self):
        from diive.configs.exampledata import load_exampledata_parquet
        df_orig = load_exampledata_parquet()
        vpd_col = 'VPD_f'
        ta_col = 'Tair_f'
        xcol = ta_col
        ycol = vpd_col
        df = df_orig.loc[(df_orig.index.month >= 6) & (df_orig.index.month <= 9)].copy()
        df = df.loc[df['Rg_f'] > 20]
        # Convert units
        df[vpd_col] = df[vpd_col].multiply(0.1)  # hPa --> kPa
        bf = BinFitterCP(
            df=df,
            xcol=xcol,
            ycol=ycol,
            # predict_max_x=None,
            # predict_min_x=None,
            n_predictions=1000,
            n_bins_x=10,
            bins_y_agg='mean',
            fit_type='quadratic_offset'  # 'linear', 'quadratic_offset', 'quadratic', 'cubic'
        )
        bf.run()
        fit_results = bf.fit_results

        self.assertEqual(len(fit_results['input_df']['group'].unique()), 10)
        self.assertAlmostEqual(fit_results['bin_df'].sum().sum(), 63443.00512672013, places=5)
        self.assertAlmostEqual(fit_results['fit_df']['nom'].sum(), 678.9973275669132, places=5)
        self.assertEqual(fit_results['fit_equation_str'], 'y = 0.0050x^2-0.0424x+0.1842')
        self.assertEqual(fit_results['fit_type'], 'quadratic_offset')
        self.assertAlmostEqual(fit_results['fit_params_opt'][0], 0.005026230803512368, places=5)
        self.assertAlmostEqual(fit_results['fit_params_opt'][1], -0.042381023692167855, places=5)
        self.assertAlmostEqual(fit_results['fit_params_opt'][2], 0.18422391667068622, places=5)
        # r2 computed over the binned points (N=n_bins): the previous value
        # (0.999999685) used the full unbinned record count in the SS_tot
        # denominator, which inflated r2 ~3000x toward 1.
        self.assertAlmostEqual(fit_results['fit_r2'], 0.9989086228432604, places=6)
        self.assertEqual(fit_results['n_vals_per_bin'], {'min': 3088.0, 'max': 3150.0})
        self.assertEqual(fit_results['bins_x'].sum(), fit_results['bin_df'][xcol]['mean'].sum())
        self.assertEqual(fit_results['bins_y'].sum(), fit_results['bin_df'][ycol]['mean'].sum())


class TestBinFitterCPShowplot(unittest.TestCase):

    @staticmethod
    def _fitted(n_bins_x, bins_y_agg='mean'):
        import numpy as np
        import pandas as pd
        rng = np.random.RandomState(0)
        n = 2000
        df = pd.DataFrame({'x': rng.uniform(0, 30, n)},
                          index=pd.date_range('2020-01-01', periods=n, freq='30min'))
        # Skewed noise, so mean and median bins differ
        df['y'] = 0.01 * df['x'] ** 2 + rng.exponential(1, n)
        bf = BinFitterCP(df=df, xcol='x', ycol='y', n_bins_x=n_bins_x,
                         bins_y_agg=bins_y_agg, fit_type='quadratic_offset')
        bf.run()
        return bf

    def test_no_bins_and_no_unbinned_data_plots_the_fit(self):
        # Neither scatter is drawn here, and the returned scatter was unbound.
        from matplotlib.figure import Figure
        ax = Figure().add_subplot()
        line_xy, line_fit, *_ = self._fitted(n_bins_x=0).showplot(ax=ax, show_unbinned_data=False)
        self.assertIsNone(line_xy)
        self.assertIsNotNone(line_fit)

    def test_bin_scatter_follows_bins_y_agg(self):
        # The scatter always drew the bin means, also for median bins.
        import numpy as np
        from matplotlib.figure import Figure
        bf = self._fitted(n_bins_x=10, bins_y_agg='median')
        line_xy = bf.showplot(ax=Figure().add_subplot(), show_unbinned_data=False)[0]
        bin_df = bf.fit_results['bin_df']
        np.testing.assert_array_equal(line_xy.get_offsets()[:, 0], bin_df['x']['median'])
        np.testing.assert_array_equal(line_xy.get_offsets()[:, 1], bin_df['y']['median'])
        self.assertFalse(np.allclose(bin_df['y']['median'], bin_df['y']['mean']))


if __name__ == '__main__':
    unittest.main()
