import unittest

import pandas as pd

import diive as dv
from diive.configs.exampledata import load_exampledata_GENERIC_TXT_EDDY_COVARIANCE_10Hz


class TestEcHires(unittest.TestCase):

    def test_flux_detection_limit(self):
        df = load_exampledata_GENERIC_TXT_EDDY_COVARIANCE_10Hz()
        df = df[['x', 'y', 'z', 'N2Od', 'Ts', 'H2O']].copy()
        df['pressure'] = 100000
        df['N2Od'] = df['N2Od'].multiply(10 ** 3)  # Convert from umol mol-1 to nmol mol-1
        df['H2O'] = df['H2O'].div(10 ** 6)  # Convert from umol mol-1 to mol mol-1
        df['Ts'] = df['Ts'].add(273.15)  # From degC to K
        fdl = dv.FluxDetectionLimit(
            df=df,
            u_col='x',  # m s-1
            v_col='y',  # m s-1
            w_col='z',  # m s-1
            c_col='N2Od',  # nmol mol-1 (ppb)
            ts_col='Ts',  # degC
            h2o_col='H2O',  # mol mol-1
            press_col='pressure',  # Pa
            noise_range=20,  # seconds
            default_lag=2.8,  # seconds
            lag_range=[-180, 180],  # seconds, calculate covariance for all steps between -180s and +180s
            lag_stepsize=1,  # number of records, step size for lag search
            sampling_rate=10,  # Hz
            show_covariance_plot=True,
            title_covariance_plot="Covariance vs time lag for example file")
        fdl.run()
        results = fdl.get_detection_limit()
        self.assertEqual(len(fdl.hires_df.columns), 13)
        self.assertIn("e", fdl.hires_df.columns)  # e_col
        self.assertIn("pd", fdl.hires_df.columns)  # pd_col
        self.assertIsInstance(fdl.hires_df, pd.DataFrame)
        self.assertEqual(fdl.lag_from, -1800)
        self.assertEqual(fdl.lag_to, 1800)
        self.assertIn('flux_detection_limit', results)
        self.assertIn('flux_noise_rmse', results)
        self.assertEqual(results['flux_detection_limit'], 1.9300179626373497)
        self.assertEqual(results['flux_noise_rmse'], 0.6433393208791166)

        # Verify that the flux conversion factor is applied correctly
        flux_conversion_factor = fdl.cov_df['cov_flux'] / fdl.cov_df['cov']
        calculated_flux_conversion_factor = (
                1 / ((8.31446261815324 * fdl.hires_df['Ts'].mean()) / fdl.hires_df['pd'].mean()))
        # Ensure the calculated factor matches the expected value
        self.assertEqual(flux_conversion_factor.iloc[0], calculated_flux_conversion_factor)

        # Ensure that the flux detection limit and noise RMSE are calculated correctly
        detection_limit = fdl.results['flux_detection_limit']
        noise_rmse = fdl.results['flux_noise_rmse']
        # Check the detection limit follows 3 * RMSE rule
        self.assertEqual(detection_limit, 3 * noise_rmse)


if __name__ == '__main__':
    unittest.main()
