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

import math
from pathlib import Path

import pandas as pd
from pandas import Series, DataFrame

from diive.pkgs.echires.lag import MaxCovariance
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

        # Max covariance
        foundlag = self.cov_df.loc[self.cov_df['flag_peak_max_cov_abs'] == True]

        # Location of max abs covariance
        self.cov_max_ix = foundlag.index[0]
        self.cov_max_shift = foundlag.iloc[0]['shift']

        # self.cov_max_ix, self.cov_max_shift = self._max_abs_covariance(cov_df=self.cov_df)

        # Convert covariance to flux units
        self.cov_df['cov_flux'] = self.cov_df['cov'].copy()
        self.cov_df['cov_flux'] = self._flux_conversion_factor(cov_df=self.cov_df)  # todo check

        self.flux_detection_limit, self.flux_noise_rmse = self._flux_detection_limit(cov_df=self.cov_df.copy())

        if self.cov_max_ix:
            self.flux = self.cov_df.iloc[self.cov_max_ix]['cov_flux']
        self.signal_to_noise = abs(self.flux) / self.flux_noise_rmse if self.flux else None
        self.signal_to_detection_limit = abs(self.flux) / self.flux_detection_limit if self.flux else None

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
        primes_df = r.get_primes()
        return primes_df[f"{self.w.name}_TURB"], primes_df[f"{self.c.name}_TURB"]

    def _crosscovariance(self) -> DataFrame:
        # # 20 Hz data: 1 record = 0.05s, 20 records = 1s
        # # we want to shift from -180s to -160s and from +160s to +180s
        # sampling_rate = 20  # Hz
        # lag_from = -180 * sampling_rate  # 160s * 20 = 3200 records
        # lag_to = -160 * sampling_rate
        # shift_stepsize = 1

        mc = MaxCovariance(
            df=df,
            var_reference=f"{self.w.name}_TURB",
            var_lagged=f"{self.c.name}_TURB",
            lgs_winsize_from=self.lag_from,
            lgs_winsize_to=self.lag_to,
            shift_stepsize=self.shift_stepsize,
            segment_name="cross-covariance"
        )

        mc.run()
        cov_df, props_peak_auto = mc.get()

        mc.plot_scatter_cov(title="test")

        # # Old: todo delete
        # cov_df = pd.DataFrame(columns=['shift', 'cov', 'cov_flux', 'cov_abs'])
        # # Complete lag range
        # lagrange = range(self.lag_from, abs(self.lag_from), self.shift_stepsize)
        # cov_df['shift'] = Series(lagrange)
        #
        # for ix, row in cov_df.iterrows():
        #     shift = row['shift']
        #     cov = self.w_prime.cov(self.c_prime.shift(shift))
        #     cov_df.loc[cov_df['shift'] == row['shift'], 'cov'] = cov
        # cov_df['cov'] = cov_df['cov'].astype(float)
        # cov_df['cov_abs'] = cov_df['cov'].abs()
        # # cov_df.loc[cov_max_ix, 'flag_peak_max_cov_abs'] = True
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
        #   "The mole fraction of a gas can be calculated as the ratio of its partial pressure
        #   and the total atmospheric pressure."
        
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

        # from -3600 to -3200 (-180 to -160s)
        # from +3200 to +3600 (+160 to +180s)

        # 20s @ 20Hz = 20 * 20 = 400 records
        winsize = 20 * 20

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

    from diive.core.io.filereader import search_files, ReadFileType

    # Dirs
    INDIR = [r'F:\Sync\luhk_work\CURRENT\DAS_detectionlimit_test\IN']
    OUTDIR = r'F:\Sync\luhk_work\CURRENT\DAS_detectionlimit_test\OUT'

    # Search & merge high-res data files (30MIN files in this example)
    filepaths = search_files(INDIR, "*.csv")
    print(filepaths)

    # Column names
    u_col = 'U_[R350-B]'  # m s-1
    v_col = 'V_[R350-B]'  # m s-1
    w_col = 'W_[R350-B]'  # m s-1
    # n2o_col = 'N2O_DRY_[LGR-A]'  # dry mole fraction in nmol mol-1 (ppb)
    ch4_col = 'CH4_DRY_[QCL-C2]'  # dry mole fraction in nmol mol-1 (ppb)
    h2o_col = 'H2O_DRY_[QCL-C2]'  # nmol mol-1 (ppb)
    pd_col = 'PRESS_CELL_[QCL-C2]'  # hPa, but will be converted to Pa
    ta_col = 'T_CELL_[QCL-C2]'  # K, we use this as air temperature for now

    filepart_results_df = pd.DataFrame(columns=['file', 'part',
                                                'flux_detection_limit', 'flux_noise_rmse',
                                                'lag', 'flux', 'signal_to_noise',
                                                'signal_to_detection_limit'])

    for fp in filepaths:

        print(f"File: {fp}")
        loaddatafile = ReadFileType(filetype='ETH-SONICREAD-BICO-MOD-CSV-20HZ',
                                    filepath=fp,
                                    data_nrows=None)
        data_df, metadata_df = loaddatafile.get_filedata()
        df = loaddatafile.data_df

        file_n_records = len(df)

        fileparts = range(0, file_n_records, 36000)  # One filepart = 1 half-hour = 36000 records (at 20Hz)
        for filepart in fileparts:
            filepart_df = df.iloc[filepart:filepart + 36000]
            print(f"File: {fp} / Filepart: {filepart} / Number of records: {len(filepart_df.index)}")
            if file_n_records < 29000:
                print("(!)Skipping filepart, not enough records.")
                continue
            fdl = FluxDetectionLimit(
                u=filepart_df[u_col].copy(),
                v=filepart_df[v_col].copy(),
                w=filepart_df[w_col].copy(),
                c=filepart_df[ch4_col].copy(),
                ta=filepart_df[ta_col].copy(),  # K; R also has K
                h2o=filepart_df[h2o_col].copy().div(10 ** 9),  # Convert from nmol mol-1 to mol mol-1
                press=filepart_df[pd_col].copy().multiply(100),  # From hPa to Pa; R also has Pa
                lag_from=-180,  # in seconds
                lag_to=180,  # in seconds
                shift_stepsize=10,
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
        outfile = Path(OUTDIR) / 'results_CH4.csv'
        # outfile = Path(outdir) / 'results_CH4.csv'
        filepart_results_df.to_csv(outfile)
