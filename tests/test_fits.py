import unittest

from diive.pkgs.fits.fitter import BinFitterCP


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
        self.assertAlmostEqual(fit_results['bin_df'].sum().sum(), 63443.00512672013, places=5   )
        self.assertEqual(fit_results['fit_df']['nom'].sum(), 678.9973275669132)
        self.assertEqual(fit_results['fit_equation_str'], 'y = 0.0050x^2-0.0424x+0.1842')
        self.assertEqual(fit_results['fit_type'], 'quadratic_offset')
        self.assertEqual(fit_results['fit_params_opt'][0], 0.005026230803512368)
        self.assertEqual(fit_results['fit_params_opt'][1], -0.042381023692167855)
        self.assertEqual(fit_results['fit_params_opt'][2], 0.18422391667068622)
        self.assertEqual(fit_results['fit_r2'], 0.999999685472016)
        self.assertEqual(fit_results['n_vals_per_bin'], {'min': 3088.0, 'max': 3150.0})
        self.assertEqual(fit_results['bins_x'].sum(), fit_results['bin_df'][xcol]['mean'].sum())
        self.assertEqual(fit_results['bins_y'].sum(), fit_results['bin_df'][ycol]['mean'].sum())


if __name__ == '__main__':
    unittest.main()
