"""
FLUX DETECTION LIMIT
====================

    Based on Langford et al. (2015)
    Parts of this code are based on the source code from Striednig et al. (2020)

    Tilt correction in EddyPro:
    https://www.licor.com/env/support/EddyPro/topics/anemometer-tilt-correction.html


    References:

    (LAN15) Langford, B., Acton, W., Ammann, C., Valach, A., & Nemitz, E. (2015). Eddy-covariance
        data with low signal-to-noise ratio: Time-lag determination, uncertainties and limit of
        detection. Atmospheric Measurement Techniques, 8(10), 4197–4213.
        https://doi.org/10.5194/amt-8-4197-2015

    (MAM16) Mammarella, I., Peltola, O., Nordbo, A., Järvi, L., & Rannik, Ü. (2016). Quantifying
        the uncertainty of eddy covariance fluxes due to the use ofdifferent software packages
        and combinations of processing steps in twocontrasting ecosystems. Atmospheric Measurement
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


"""
# from diive.common.io.filereader import MultiDataFileReader, search_files
import math
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

import diive.core.io.filereader as filereader
from diive.pkgs.echires.windrotation import WindRotation2D


class FluxDetectionLimit:
    R = 8.31446261815324  # Universal gas constant, m3 Pa K-1 mol-1

    def __init__(self, u, v, w, c, ta, h2o, press,
                 lag_from: int, lag_to: int, shift_stepsize: int, sampling_rate: int):
        self.u = u
        self.v = v
        self.w = w
        self.c = c
        self.ta = ta
        self.h2o = h2o
        self.press = press

        # Convert s to number or records,
        #   e.g. for 20Hz data (20 records = 1s): 180s * 20 = 3600 records
        self.lag_from = lag_from * sampling_rate
        self.lag_to = lag_to * sampling_rate
        self.shift_stepsize = shift_stepsize

        self.w_prime, self.c_prime = self._turbulent_fluctuations()
        self.cov_df = self._crosscovariance()
        self.cov_max_ix, self.cov_max_shift = self._max_abs_covariance(cov_df=self.cov_df)
        self.cov_df['cov_flux'] = self._flux_conversion_factor(cov_df=self.cov_df)

        self.flux_detection_limit, self.flux_noise_rmse = self._flux_detection_limit(cov_df=self.cov_df.copy())

        self.flux = self.cov_df['cov_flux'].iloc[self.cov_max_ix] if self.cov_max_ix else False
        self.signal_to_noise = abs(self.flux) / self.flux_noise_rmse if self.flux else None
        self.signal_to_detection_limit = abs(self.flux) / self.flux_detection_limit if self.flux else None

        self.plot_(cov_df=self.cov_df)

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

    def get_detection_limit(self):
        return self.flux_detection_limit, self.flux_noise_rmse, self.cov_max_ix, self.cov_max_shift, \
               self.flux, self.signal_to_noise, self.signal_to_detection_limit

    def _turbulent_fluctuations(self):
        r = WindRotation2D(u=self.u, v=self.v, w=self.w, c=self.c)
        w_prime, c_prime = r.get_primes()
        return w_prime, c_prime

    def _crosscovariance(self) -> DataFrame:
        # # 20 Hz data: 1 record = 0.05s, 20 records = 1s
        # # we want to shift from -180s to -160s and from +160s to +180s
        # sampling_rate = 20  # Hz
        # lag_from = -180 * sampling_rate  # 160s * 20 = 3200 records
        # lag_to = -160 * sampling_rate
        # shift_stepsize = 1
        cov_df = pd.DataFrame(columns=['shift', 'cov', 'cov_flux', 'cov_abs'])
        # Complete lag range
        lagrange = range(self.lag_from, abs(self.lag_from), self.shift_stepsize)
        cov_df['shift'] = Series(lagrange)

        for ix, row in cov_df.iterrows():
            shift = row['shift']
            cov = self.w_prime.cov(self.c_prime.shift(shift))
            cov_df.loc[cov_df['shift'] == row['shift'], 'cov'] = cov
        cov_df['cov'] = cov_df['cov'].astype(float)
        cov_df['cov_abs'] = cov_df['cov'].abs()

        # cov_df.loc[cov_max_ix, 'flag_peak_max_cov_abs'] = True
        return cov_df

    def _flux_conversion_factor(self, cov_df: DataFrame) -> Series:
        """Calculate flux conversion factor

        Conversion from covariances to flux units:
            see SAB18, eq.(16) (or similar: MAM16, eq.(2))
            flux_conversion_factor = 1 / ( (R Ta) / pd)   [same as: 1 / ( (Ta / pd) * R )]
                Ta  ... air temperature, K (in SAB18 degC were used)
                R   ... Universal gas constant, m3 Pa K-1 mol-1
                pd  ... dry air partial pressure, Pa
        """
        # Partial pressure of water vapor (Pa) (CAM98, p39, eq.(3.5), see their example)
        e = self.h2o * self.press

        # Dry air partial pressure (Pa) (SAB18, p513)
        pd = self.press - e

        # Flux conversion factor, SAB18, eq.(16)
        flux_conversion_factor = 1 / ((self.R * self.ta) / pd)
        flux_conversion_factor = flux_conversion_factor.mean()  # Use overall mean
        return cov_df['cov'] * flux_conversion_factor

    def _flux_detection_limit(self, cov_df: DataFrame):
        """Calculate flux detection limit

        Flux noise criterium using RMSE noise of covariance between +/- 160-180s
        see LAN15

        """
        # LAN15, eq.(9)
        # Left lag window
        cov_df_left = cov_df.loc[(cov_df['shift'] >= self.lag_from) & (cov_df['shift'] <= self.lag_to)]
        # Right lag window
        cov_df_right = cov_df.loc[(cov_df['shift'] >= abs(self.lag_to)) & (cov_df['shift'] <= abs(self.lag_from))]

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

    def plot_(self, cov_df: DataFrame) -> None:
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        gs.update(wspace=.2, hspace=0, left=.05, right=.95, top=.95, bottom=.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(cov_df['shift'], cov_df['cov_flux'])
        fig.show()
        # plt.close(fig=fig)


if __name__ == '__main__':

    # Dirs
    indir = r'F:\CH-AES\flux_detection_limit\1-in'
    outdir = r'F:\CH-AES\flux_detection_limit\2-out'

    # Search & merge high-res data files
    searchdir = indir
    pattern = '*.csv'
    filepaths = filereader.search_files(searchdirs=searchdir, pattern=pattern)

    # Column names
    u_col = 'U_[R350-A]'  # m s-1
    v_col = 'V_[R350-A]'  # m s-1
    w_col = 'W_[R350-A]'  # m s-1
    n2o_col = 'N2O_DRY_[LGR-A]'  # dry mole fraction umol mol-1 (ppm), but will be converted to ppb
    ch4_col = 'CH4_[LGR-A]'  # dry mole fraction umol mol-1 (ppm), but will be converted to ppb
    pd_col = 'PRESS_BOX_[IRGA75-A]'  # hPa, but will be converted to Pa
    ta_col = 'T_SONIC_[R350-A]'  # K, we use this as air temperature for now
    h2o_col = 'H2O_[LGR-A]'  # umol mol-1 (ppm)

    filepart_results_df = pd.DataFrame(columns=['file', 'part',
                                                'flux_detection_limit', 'flux_noise_rmse',
                                                'lag', 'flux', 'signal_to_noise',
                                                'signal_to_detection_limit'])

    for fp in filepaths:
        print(f"File: {fp}")
        df = pd.read_csv(fp, skiprows=[1, 2])
        fileparts = range(0, 432000, 36000)  # One filepart = 1 half-hour = 36000 records (at 20Hz)
        for filepart in fileparts:
            filepart_df = df.iloc[filepart:filepart + 36000]
            print(f"File: {fp} / Filepart: {filepart} / Number of records: {len(filepart_df.index)}")
            if len(filepart_df.index) < 35640:
                print("(!)Skipping filepart, not enough records.")
                continue
            fdl = FluxDetectionLimit(u=filepart_df[u_col].copy(),
                                     v=filepart_df[v_col].copy(),
                                     w=filepart_df[w_col].copy(),
                                     c=filepart_df[ch4_col].copy().multiply(1000),
                                     # c=filepart_df[ch4_col].copy().multiply(1000),
                                     # Convert from umol mol-1 to nmol mol-1
                                     ta=filepart_df[ta_col].copy(),  # K; R also has K
                                     h2o=filepart_df[h2o_col].div(10 ** 6),  # Convert from umol mol-1 to mol mol-1
                                     press=filepart_df[pd_col].copy().multiply(100),
                                     # Convert from hPa to Pa; R also has Pa)
                                     lag_from=-180,
                                     lag_to=-160,
                                     shift_stepsize=1,
                                     sampling_rate=20)

            flux_detection_limit, \
            flux_noise_rmse, \
            cov_max_ix, \
            cov_max_shift, \
            flux, \
            signal_to_noise, \
            signal_to_detection_limit = fdl.get_detection_limit()

            new_results = [fp.name, filepart, flux_detection_limit, flux_noise_rmse, cov_max_shift,
                           flux, signal_to_noise, signal_to_detection_limit]
            filepart_results_df.loc[len(filepart_results_df)] = new_results

        print(filepart_results_df)
        # Save after each file
        outfile = Path(outdir) / 'results_CH4.csv'
        # outfile = Path(outdir) / 'results_CH4.csv'
        filepart_results_df.to_csv(outfile)
