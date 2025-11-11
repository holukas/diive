"""
FLUX DETECTION LIMIT
====================

    Based on Langford et al. (2015)
    Parts of this code are based on the source code from Striednig et al. (2020)

    Tilt correction in EddyPro:
    https://www.licor.com/env/support/EddyPro/topics/anemometer-tilt-correction.html


    References:

    (CAM98) Campbell, G. S., & Norman, J. M. (1998). An Introduction to Environmental Biophysics.
        Springer New York. https://doi.org/10.1007/978-1-4612-1626-1

    (LAN15) Langford, B., Acton, W., Ammann, C., Valach, A., & Nemitz, E. (2015). Eddy-covariance
        data with low signal-to-noise ratio: Time-lag determination, uncertainties and limit of
        detection. Atmospheric Measurement Techniques, 8(10), 4197–4213.
        https://doi.org/10.5194/amt-8-4197-2015

    (MAM16) Mammarella, I., Peltola, O., Nordbo, A., Järvi, L., & Rannik, Ü. (2016). Quantifying
        the uncertainty of eddy covariance fluxes due to the use ofdifferent software packages
        and combinations of processing steps in two contrasting ecosystems. Atmospheric Measurement
        Techniques, 9(10), 4915–4933. https://doi.org/10.5194/amt-9-4915-2016

    (NEM18) Nemitz, E., Mammarella, I., Ibrom, A., Aurela, M., Burba, G. G., Dengel, S., Gielen, B., Grelle,
        A., Heinesch, B., Herbst, M., Hörtnagl, L., Klemedtsson, L., Lindroth, A., Lohila, A., McDermitt,
        D. K., Meier, P., Merbold, L., Nelson, D., Nicolini, G., … Zahniser, M. (2018). Standardisation of
        eddy-covariance flux measurements of methane and nitrous oxide. International Agrophysics, 32(4),
        517–549. https://doi.org/10.1515/intag-2017-0042

    (SAB18) Sabbatini, S., Mammarella, I., Arriga, N., Fratini, G., Graf, A., Hörtnagl, L., Ibrom, A.,
        Longdoz, B., Mauder, M., Merbold, L., Metzger, S., Montagnani, L., Pitacco, A., Rebmann, C.,
        Sedlák, P., Šigut, L., Vitale, D., & Papale, D. (2018). Eddy covariance raw data processing
        for CO2 and energy fluxes calculation at ICOS ecosystem stations. International Agrophysics,
        32(4), 495–515. https://doi.org/10.1515/intag-2017-0043

    (STR20) Striednig, M., Graus, M., Märk, T. D., & Karl, T. G. (2020). InnFLUX – an open-source
        code for conventional and disjunct eddy covariance analysis of trace gas measurements:
        An urban test case. Atmospheric Measurement Techniques, 13(3), 1447–1465.
        https://doi.org/10.5194/amt-13-1447-2020
        Source code: https://www.atm-phys-chem.at/innflux/
        Source code: https://git.uibk.ac.at/acinn/apc/innflux


"""
import math
import numbers
import time
from pathlib import Path

import matplotlib
import pandas as pd
from pandas import DataFrame

from diive.pkgs.createvar.conversions import air_temp_from_sonic_temp
from diive.pkgs.echires.lag import MaxCovariance
from diive.pkgs.echires.windrotation import WindRotation2D


class FluxDetectionLimit:
    """Calculates the flux detection limit from high-resolution eddy covariance data.

    This class implements the method described by Langford et al. (2015) to
    determine the flux detection limit (FDL) and signal-to-noise ratio for
    a given scalar flux (e.g., N2O, CH4).

    The FDL is calculated based on the noise in the cross-covariance function
    at large time lags, which are assumed to be uncorrelated with the turbulent
    flux. The flux noise is quantified as the Root Mean Square Error (RMSE)
    of the covariances in these "noise" windows (e.g., +/- 160-180s). The FDL
    is then defined as 3 * RMSE.

    The implementation follows the methodology used in Striednig et al. (2020)
    for the noise RMSE calculation (based on LAN15, eq. 9) and Sabbatini et al.
    (2018) for the flux conversion factor.

    The class processes a pandas DataFrame of high-resolution (e.g., 10Hz or
    20Hz) eddy covariance data.

    The main workflow executed by the `run()` method is:
    1.  Convert sonic temperature to air temperature.
    2.  Calculate turbulent fluctuations (primes) via 2D wind rotation.
    3.  Compute the full cross-covariance function between vertical wind (w')
        and the scalar (c') over the specified `lag_range`.
    4.  Convert covariance to physical flux units (e.g., nmol m-2 s-1)
        using the mean air temperature and dry air pressure.
    5.  Calculate the noise RMSE from the covariance function at the large
        lag windows defined by `lag_range` and `noise_range`.
    6.  Determine the FDL (3 * RMSE).
    7.  Find the maximum covariance (the flux signal) and calculate
        signal-to-noise ratios.

    Args:
        df (pd.DataFrame): High-resolution time series data,
            e.g., a half-hourly eddy covariance raw data file.
        u_col (str): Column name for the u component of wind speed (m s-1).
        v_col (str): Column name for the v component of wind speed (m s-1).
        w_col (str): Column name for the w component of wind speed (m s-1).
        c_col (str): Column name for the scalar concentration
            e.g., N2O in nmol mol-1, but it can also be any other scalar
            such as CO2 in umol mol-1. The units affect the units of the
            FDL, for N2O FDL would be nmol m-2 s-1, for CO2 it would be
            umol m-2 s-1, etc.
        ts_col (str): Column name for the sonic temperature (K).
            Note: This is converted to air temperature internally.
        h2o_col (str): Column name for the H2O mole fraction (mol mol-1).
        press_col (str): Column name for the air pressure (Pa).
        default_lag (int): The default time lag (in seconds) to use for the
            calculation of the "signal" in the signal-to-noise ratio. A positive
            number means turbulent departures of c lag behind turbulent w, i.e,
            the c signal needs that much longer to reach the sensor than w, e.g.,
            because it has to travel through an inlet tube.
        noise_range (int): The width of the time window (in seconds) at the
            edges of the lag range used to calculate noise (e.g., 20s).
            This means that if the lag range is [-180, 180] (in seconds),
            then the noise will be calculated between -180 and -160s and
            then again between +160 and +180s.
        lag_range (list): A two-element list specifying the total time lag
            range [min, max] (in seconds) for the covariance calculation
            (e.g., [-180, 180] will calculate all covariances between -180s
            and +180s).
        lag_stepsize (int): The step size (in records) for iterating
            through the covariance lag search. Note that this is given as
            number of records, not seconds.
        sampling_rate (int): The data sampling rate (in Hz), e.g., 20 for 20Hz.

    Attributes:
        hires_df (pd.DataFrame): A copy of the input DataFrame with added
            calculated columns (e.g., 'e', 'pd', primes for w and c).
        cov_df (pd.DataFrame): A DataFrame holding the results of the
            cross-covariance calculation, including shifts (lags in records),
            covariance values, and flux-converted values.
        results (dict): A dictionary containing the final calculated results
            after `run()` is called. Keys include:
            - 'flux_detection_limit'
            - 'flux_noise_rmse'
            - 'cov_max_ix' (index of max covariance)
            - 'cov_max_shift' (lag in records of max covariance)
            - 'flux_signal_at_cov_max_shift'
            - 'signal_to_noise'
            - 'signal_to_detection_limit'

    References:
        (LAN15) Langford, B., et al. (2015). Eddy-covariance
            data with low signal-to-noise ratio: Time-lag determination,
            uncertainties and limit of detection. Atmospheric Measurement
            Techniques, 8(10), 4197–4213.

        (SAB18) Sabbatini, S., et al. (2018). Eddy covariance raw data
            processing for CO2 and energy fluxes calculation at ICOS
            ecosystem stations. International Agrophysics, 32(4), 495–515.

        (STR20) Striednig, M., et al. (2020). InnFLUX – an open-source
            code for conventional and disjunct eddy covariance analysis of
            trace gas measurements: An urban test case. Atmospheric
            Measurement Techniques, 13(3), 1447–1465.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            u_col: str,
            v_col: str,
            w_col: str,
            c_col: str,
            ts_col: str,
            h2o_col: str,
            press_col: str,
            default_lag: float,
            noise_range: int,
            lag_range: list,
            lag_stepsize: int,
            sampling_rate: int,
            show_covariance_plot: bool = True,
            title_covariance_plot: str = None
    ):
        self.hires_df = df.copy()
        self.u_col = u_col
        self.v_col = v_col
        self.w_col = w_col
        self.c_col = c_col
        self.ts_col = ts_col
        self.h2o_col = h2o_col
        self.press_col = press_col
        self.noise_range = noise_range
        self.lag_range = lag_range
        self.default_lag = default_lag
        self.lag_stepsize = lag_stepsize
        self.sampling_rate = sampling_rate
        self.show_covariance_plot = show_covariance_plot
        self.title_covariance_plot = title_covariance_plot

        # Convert s to number or records,
        #   e.g. for 20Hz data (20 records = 1s): 180s * 20 = 3600 records
        self.lag_from = lag_range[0] * sampling_rate
        self.lag_to = lag_range[1] * sampling_rate

        # Calculate e
        # e = partial pressure of water vapor (Pa) = H2O mole fraction (mol mol-1) * air pressure (Pa)
        # CAM98, p39 eq.(3.5), solve for e
        self.e_col = 'e'
        self.hires_df[self.e_col] = self.hires_df[self.h2o_col] * self.hires_df[self.press_col]

        # Calculate pd
        # pd = dry air partial pressure (Pa)
        # pd (in Pa) = pa (in Pa) - e (in Pa)
        # SAB18, p513
        self.pd_col = 'pd'
        self.hires_df[self.pd_col] = self.hires_df[self.press_col] - self.hires_df['e']

        # Convert sonic temperature (K) to air temperature (K)
        self.hires_df[ts_col] = (
            air_temp_from_sonic_temp(sonic_temp=self.hires_df[ts_col], h2o=self.hires_df[self.h2o_col]))

        # New variables
        self.cov_df = pd.DataFrame()
        self.results = {}
        self.fig_cov = None

    def run(self):

        # Calculate turbulent fluctuations
        self.hires_df = self._turbulent_fluctuations(df=self.hires_df)

        # Calculate covariances
        self.cov_df, cov_max_ix, cov_max_shift, self.fig_cov = self._crosscovariance()

        # Flux conversion factor
        # Convert covariance to flux units
        self.cov_df['cov_flux'] = self.cov_df['cov'].copy()
        self.cov_df = self._flux_conversion_factor(cov_df=self.cov_df)

        # Flux detection limit
        flux_detection_limit, flux_noise_rmse = self._flux_detection_limit(cov_df=self.cov_df.copy())

        # Get flux at default lag
        if isinstance(cov_max_ix, numbers.Integral):
            # Calculate flux at default lag, this is the "signal" in the signal-to-noise ratio
            # Default lag is given as seconds, here it is converted to shift of "number of records", i.e.,
            # by how many records the time series needs to be shifted in one direction.
            flux = self.cov_df.loc[self.cov_df['shift'] == -self.default_lag * self.sampling_rate]['cov_flux'].values[0]
            # flux = self.cov_df.iloc[cov_max_ix]['cov_flux']
        else:
            raise Exception("No default lag covariance found")
        signal_to_noise = abs(flux) / flux_noise_rmse if flux else None
        signal_to_detection_limit = abs(flux) / flux_detection_limit if flux else None

        self.results = {
            'flux_detection_limit': flux_detection_limit,
            'flux_noise_rmse': flux_noise_rmse,
            'cov_max_ix': cov_max_ix,
            'cov_max_shift': cov_max_shift,
            'flux_signal_at_default_lag': flux,
            'signal_to_noise': signal_to_noise,
            'signal_to_detection_limit': signal_to_detection_limit
        }

    def _max_abs_covariance(self, cov_df: DataFrame):
        """Get index of max covariance and respective lag time (shift in records)"""

        # The realistic time window for lag search is 0-5s
        # This corresponds to records with shifts -99 (5s) and 0 (0s)
        # Here we search for the max covariance in the realistic time window
        subset_df = cov_df.loc[(cov_df['shift'] >= -99) & (cov_df['shift'] <= 0), :].copy()
        subset_cov_max_ix = subset_df['cov_abs'].idxmax()
        subset_cov_max_shift = int(subset_df.loc[subset_df.index == subset_cov_max_ix, 'shift'])

        # Check if cov max within allowed range
        # Allowed range here:
        #   5s (record -99, fringe record not allowed i.e. -99 not allowed)
        #   0s (record 0, fringe records not allowed i.e. 0 not allowed)
        # Negative shift means scalar arrived after wind
        if (subset_cov_max_shift > -99) and (subset_cov_max_shift < 0):
            pass
        else:
            subset_cov_max_shift = -28  # Set to nominal time lag, in 2020 = 1.40s for both N2O and CH4
            subset_cov_max_ix = subset_df.loc[subset_df['shift'] == subset_cov_max_shift].index[0]
            # subset_cov_max_shift = subset_df.loc[subset_df.index == subset_cov_max_ix, 'shift'].index[0]
        return subset_cov_max_ix, subset_cov_max_shift

    def get_detection_limit(self) -> dict:
        return self.results

    def get_fig_cov(self) -> matplotlib.figure.Figure:
        return self.fig_cov

    def _turbulent_fluctuations(self, df: pd.DataFrame) -> pd.DataFrame:
        r = WindRotation2D(u=df[self.u_col],
                           v=df[self.v_col],
                           w=df[self.w_col],
                           c=df[self.c_col])
        primes_df = r.get_primes()
        df = pd.concat([df, primes_df], axis=1)
        return df

    def _crosscovariance(self) -> tuple[pd.DataFrame, int, int, matplotlib.figure.Figure]:
        # # 20 Hz data: 1 record = 0.05s, 20 records = 1s
        # # we want to shift from -180s to -160s and from +160s to +180s
        # sampling_rate = 20  # Hz
        # lag_from = -180 * sampling_rate  # 160s * 20 = 3200 records
        # lag_to = -160 * sampling_rate
        # shift_stepsize = 1

        start_time = time.perf_counter()
        mc = MaxCovariance(
            df=self.hires_df,
            var_reference=f"{self.w_col}_TURB",
            var_lagged=f"{self.c_col}_TURB",
            lgs_winsize_from=self.lag_from,
            lgs_winsize_to=self.lag_to,
            shift_stepsize=self.lag_stepsize,
            segment_name="cross-covariance",

        )

        mc.run()
        end_time = time.perf_counter()
        elapsed_time_seconds = end_time - start_time
        print(f"Time needed for covariance calculation: {elapsed_time_seconds:0.3f}s")
        cov_df, props_peak_auto = mc.get()

        if self.show_covariance_plot:
            fig_cov = mc.plot_scatter_cov(title=f"Covariance vs time lag {self.title_covariance_plot}")
        else:
            fig_cov = None

        # Max covariance
        foundlag = cov_df.loc[cov_df['flag_peak_max_cov_abs'] == True]

        # Location of max abs covariance
        cov_max_ix = foundlag.index[0]
        cov_max_shift = foundlag.iloc[0]['shift']

        return cov_df, cov_max_ix, cov_max_shift, fig_cov

    def _flux_conversion_factor(self, cov_df: DataFrame) -> pd.DataFrame:
        """Calculate flux conversion factor

        Conversion from covariances to flux units:
            see SAB18, eq.(16) (or similar: MAM16, eq.(2))
            flux_conversion_factor = 1 / ( (R Ta) / pd)   [same as: 1 / ( (Ta / pd) * R )]
                Ta  ... air temperature, K
                R   ... Universal gas constant, m3 Pa K-1 mol-1
                pd  ... dry air partial pressure, Pa

        From STR20 (innFLUX script):
            Calculate conversion factor for flux in nmol m-2 s-1
            flux_conversion_factor = 1/( (T_mean/273.15) * (1013.25/p_mean) * 22.414e-3 );
            They achieve the identical result (the dry air molar density),
            but the innFLUX script expresses the conversion as a ratio against STP reference
            conditions, while SAB18 uses the direct P/RT Ideal Gas Law formulation.

        """
        # Flux conversion factor, SAB18, eq.(16)
        R = 8.31446261815324  # Universal gas constant, m3 Pa K-1 mol-1
        ta_mean = self.hires_df[self.ts_col].mean()  # K
        pd_mean = self.hires_df[self.pd_col].mean()  # Pa
        flux_conversion_factor = 1 / ((R * ta_mean) / pd_mean)
        cov_df['cov_flux'] = cov_df['cov'] * flux_conversion_factor
        return cov_df

    def _flux_detection_limit(self, cov_df: DataFrame):
        """Calculate flux detection limit

        Flux noise criterium using RMSE noise of covariance between +/- 160-180s
        see LAN15

        """

        # from -3600 to -3200 (-180 to -160s)
        # from +3200 to +3600 (+160 to +180s)

        # 20s @ 20Hz = 20 * 20 = 400 records
        # 20s @ 10Hz = 20 * 10 = 200 records
        winsize = self.noise_range * self.sampling_rate

        # LAN15, eq.(9)
        # Left lag window
        cov_df_left = cov_df.loc[(cov_df['shift'] >= self.lag_from) & (cov_df['shift'] <= self.lag_from + winsize)]
        # Right lag window
        cov_df_right = cov_df.loc[
            (cov_df['shift'] >= abs(self.lag_to) - winsize) & (cov_df['shift'] <= abs(self.lag_to))]

        # Implementation by STR20
        # their source code: https://git.uibk.ac.at/acinn/apc/innflux/-/blob/master/innFLUX_step1.m#L435
        #   flux_noise_rmse = sqrt(0.5 *
        #       (  nanstd(cov_wc(idx_left))^2 + nanmean(cov_wc(idx_left))^2 +
        #       nanstd(cov_wc(idx_right))^2 + nanmean(cov_wc(idx_right))^2)  )
        #       * flux_conversion_factor;
        flux_noise_rmse = math.sqrt(0.5 * (
                (cov_df_left['cov_flux'].std()) ** 2 + (cov_df_left['cov_flux'].mean()) ** 2 +
                (cov_df_right['cov_flux'].std()) ** 2 + (cov_df_right['cov_flux'].mean()) ** 2
        ))
        flux_detection_limit = flux_noise_rmse * 3
        print(f"Flux noise RMSE: {flux_noise_rmse}")
        print(f"Flux detection limit: {flux_detection_limit}")
        return flux_detection_limit, flux_noise_rmse


if __name__ == '__main__':

    from diive.core.io.filereader import search_files

    # Dirs
    INDIR = [r'F:\Sync\luhk_work\CURRENT\flux_detection_limit\raw']
    OUTDIR = r'F:\Sync\luhk_work\CURRENT\flux_detection_limit\OUT'

    # Search high-res data files (30MIN files in this example)
    filepaths = search_files(INDIR, "*.txt")

    # Column names
    u_col = 'x'  # m s-1
    v_col = 'y'  # m s-1
    w_col = 'z'  # m s-1
    n2o_col = 'N2Od'  # dry umol mol-1 (ppm), but must be dry mole fraction in nmol mol-1 (ppb)
    h2o_col = 'H2O'  # umol mol-1 (ppm), but needs to be in mol mol-1 (ppt)
    press_col = 'Pressure'  # Pa, example value
    ta_col = 'Ts'  # °C, sonic temperature

    file_results_df = pd.DataFrame(columns=['file',
                                            'flux_detection_limit', 'flux_noise_rmse',
                                            'lag', 'flux_signal_at_cov_max_shift', 'signal_to_noise',
                                            'signal_to_detection_limit'])

    fig_cov = None
    for ix, fp in enumerate(filepaths):
        # todo testing
        if ix > 0:
            break

        # Load data
        print(f"Reading file #{ix}: {fp} ...")
        df = pd.read_csv(fp)

        # Conversions
        df[n2o_col] = df[n2o_col].multiply(10 ** 3)  # Convert from umol mol-1 to nmol mol-1
        df[h2o_col] = df[h2o_col].div(10 ** 6)  # Convert from umol mol-1 to mol mol-1
        df[press_col] = 100_000  # Pa, example value
        # df[press_col] = df[press_col].multiply(133.322)  # From Torr to Pa
        df[ta_col] = df[ta_col].add(273.15)  # From degC to K

        fdl = FluxDetectionLimit(
            df=df,
            u_col=u_col,  # m s-1
            v_col=v_col,  # m s-1
            w_col=w_col,  # m s-1
            c_col=n2o_col,  # nmol mol-1 (ppb)
            ts_col=ta_col,  # degC
            h2o_col=h2o_col,  # mol mol-1
            press_col=press_col,  # Pa
            noise_range=20,  # seconds
            default_lag=3,  # seconds, positive number means c lags behind turbulent w
            lag_range=[-180, 180],  # seconds
            lag_stepsize=10,
            sampling_rate=10,
            show_covariance_plot=True,
            title_covariance_plot=fp.name)  # Hz

        fdl.run()

        results = fdl.get_detection_limit()
        fig_cov = fdl.get_fig_cov()

        new_results = [
            fp.name,
            results['flux_detection_limit'],
            results['flux_noise_rmse'],
            results['cov_max_shift'],
            results['flux_signal_at_default_lag'],
            results['signal_to_noise'],
            results['signal_to_detection_limit']
        ]
        file_results_df.loc[len(file_results_df)] = new_results

        if fig_cov:
            fig_cov.savefig(Path(OUTDIR) / f'cov_{fp.name}.png')

    print(file_results_df)

    # Save
    outfile = Path(OUTDIR) / 'results_N2O.csv'
    # outfile = Path(outdir) / 'results_CH4.csv'
    file_results_df.to_csv(outfile)

    # print(file_results_df)
