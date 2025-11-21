from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import selfheating_calc
import selfheating_constants
import selfheating_files
import selfheating_frames
import selfheating_newcols
import selfheating_plots
from diive.pkgs.createvar.air import dry_air_density, aerodynamic_resistance

pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 14)
pd.set_option('display.max_rows', 30)


class Scop(selfheating_newcols.NewCols, selfheating_constants.Constants):
    """
    Calculate scaling factors by comparing parallel measurements
    """

    def __init__(
            self,
            df: pd.DataFrame,
            site: str,
            title: str,
            outdir: str,
            pm_op_flux_wpl_col: str,
            pm_qcf_op_flux_wpl_col: str,
            pm_cp_trueflux_col: str,
            pm_qcf_cp_trueflux_col: str,
            pm_airheatcap_jkkg_col: str,
            pm_qc_mmol_col: str,
            pm_u_col: str,
            pm_ustar_col: str,
            pm_rho_v_col: str,
            pm_rho_a_col: str,
            pm_ta_col: str,
            pm_swin_col: str,
            pm_num_classes: int,
            pm_num_bootstrap_runs: int,
            pm_class_var_col: str,
            cr_ta_col: str,
            cr_swin_col: str,
            cr_ustar_col: str,
            cr_u_col: str,
            cr_rho_v_col: str,
            cr_rho_a_col: str,
            cr_qc_mmol_col: str,
            cr_op_flux_wpl_col: str,
            cr_class_var_col: str,
            pm_file: str = None,
            cr_file: str = None,
            cr_sf_file: str = None
    ):
        """
        Initializes an object with configuration parameters related to processing and
        analyzing flux and meteorological data. This method sets up all the necessary
        attributes for further data handling and computations.

        Abbreviations:
            pm ... Parallel measurements
            OP ... open-path
            CP ... enclosed-path
            wpl ... WPL-correction
            cr ... correction
            Settings for calculation of scaling factors from parallel measurements
            Correction (cr) needed for application of scaling factors to (uncorrected) fluxes

        Args:
            df:
            site (str): Site identifier where the data is collected.
            title (str): Title or name for the dataset or processing task.
            outdir (str): Output directory to store results and generated files.
            pm_file (str): Path to file with parallel measurements.
            pm_op_flux_wpl_col (str): Column name for OP flux with WPL correction in PM file.
            pm_qcf_op_flux_wpl_col (str): Column name for QC value of OP flux in PM file.
            pm_cp_trueflux_col (str): Column name for CP true flux in PM file.
            pm_qcf_cp_trueflux_col (str): Column name for QC value of CP true flux in PM file.
            pm_airheatcap_jkkg_col (str): Column name for specific heat at constant pressure of
                ambient air (J K-1 kg-1) in PM file.
            pm_qc_mmol_col (str): Column name for CO2 molar density column (mmol m-3) in PM file.
            pm_u_col (str): Column name for horizontal wind speed (m s-1) in PM file.
            pm_ustar_col (str): Column name for USTAR friction velocity (m s-1) in PM file.
            pm_rho_v_col (str): Column name for water vapor density (kg m-3) in PM file.
            pm_rho_a_col (str): Column name for air density (kg m-3) in PM file.
            pm_ta_col (str): Column name for ambient air temperature (°C) in PM file.
            pm_swin_col (str): Column name for shortwave-incoming radiation (W m-2) in PM file.
            pm_num_classes (int): Number of classes for categorization in PM processing, ignored
                if class_var_col = 'custom'.
            pm_num_bootstrap_runs (int): Number of bootstraps in each class, *0* uses measured
                data only w/o bootstrapping.
            pm_class_var_col (str): Column name for the class variable in PM file. Each class
                has its own scaling factor.
            cr_file (str): Path to file with fluxes that need correction.
            cr_sf_file (str): Path to file with scaling factors per class.
            cr_ta_col (str): Column name for ambient air temperature (°C) in CR file.
            cr_swin_col (str): Column name for shortwave-incoming radiation (W m-2) in CR file.
            cr_ustar_col (str): Column name for USTAR friction velocity (m s-1) in CR file.
            cr_u_col (str): Column name for horizontal wind speed (m s-1) in CR file.
            cr_rho_v_col (str): Column name for water vapor density (kg m-3) in CR file.
            cr_rho_a_col (str): Column name for air density (kg m-3) in CR file.
            cr_qc_mmol_col (str): Column name for CO2 molar density column (mmol m-3) in CR file.
            cr_op_flux_wpl_col (str): Column name for uncorrected flux (uncorrected for self-heating,
                but with WPL correction) in CR file.
            cr_class_var_col (str): Column name for classifying variable in CR file. Scaling factors
                are calculated for each class of the class_var
        """
        super().__init__()
        # self.userconfig = selfheating_settings.ReadSettingsFile()
        self.df = df
        self.site = site
        self.title = title
        self.outdir = Path(outdir)
        self.pm_file = pm_file
        self.pm_op_flux_wpl_col = pm_op_flux_wpl_col
        self.pm_qcf_op_flux_wpl_col = pm_qcf_op_flux_wpl_col
        self.pm_cp_trueflux_col = pm_cp_trueflux_col
        self.pm_qcf_cp_trueflux_col = pm_qcf_cp_trueflux_col
        self.pm_airheatcap_jkkg_col = pm_airheatcap_jkkg_col
        self.pm_qc_mmol_col = pm_qc_mmol_col
        self.pm_u_col = pm_u_col
        self.pm_ustar_col = pm_ustar_col
        self.pm_rho_v_col = pm_rho_v_col
        self.pm_rho_a_col = pm_rho_a_col
        self.pm_ta_col = pm_ta_col
        self.pm_swin_col = pm_swin_col
        self.pm_num_classes = pm_num_classes
        self.pm_num_bootstrap_runs = pm_num_bootstrap_runs
        self.pm_class_var_col = pm_class_var_col
        self.cr_file = cr_file
        self.cr_sf_file = cr_sf_file
        self.cr_ta_col = cr_ta_col
        self.cr_swin_col = cr_swin_col
        self.cr_ustar_col = cr_ustar_col
        self.cr_u_col = cr_u_col
        self.cr_rho_v_col = cr_rho_v_col
        self.cr_rho_a_col = cr_rho_a_col
        self.cr_qc_mmol_col = cr_qc_mmol_col
        self.cr_op_flux_wpl_col = cr_op_flux_wpl_col
        self.cr_class_var_col = cr_class_var_col

        # os.path.dirname(__file__)
        self.df = pd.DataFrame()
        self.scaling_factors_df = pd.DataFrame()
        self.outfile = None

    def apply_sf(self):
        """Apply scaling factors from file, correct fluxes"""
        self.outdir = self.outdir / "2-Application"
        self.outfile = self.outdir / "corrected_fluxes.csv"
        # self.readfile(file=self.cr_file)
        self.readfile(file=self.cr_file, nrows=10)
        self.init_newcols()
        self.detect_daytime(swin=self.df[self.cr_swin_col])
        self.aerodynamic_resistance(u=self.df[self.cr_u_col],
                                    ustar=self.df[self.cr_ustar_col],
                                    rem_outliers=True)
        self.dry_air_density(rho_v=self.df[self.cr_rho_v_col],
                             rho_a=self.df[self.cr_rho_a_col])
        self.instrument_surface_temperature(ta=self.df[self.cr_ta_col])
        self.unscaled_flux_correction_term(ts=self.df[self.ts_col],
                                           ta=self.df[self.cr_ta_col],
                                           qc_umol=self.df[self.cr_qc_mmol_col].multiply(1000),
                                           ra=self.df[self.ra_col],
                                           rho_v=self.df[self.cr_rho_v_col],
                                           rho_d=self.df[self.rho_d_col],
                                           lut_gapfill=True,
                                           rem_outliers=True)
        self.assign_scaling_factors(file=self.cr_sf_file,
                                    classvar_col=self.cr_class_var_col,
                                    lut_gapfill=True)
        self.calc_corrected_fluxes(op_co2_flux_nocorr_col=self.cr_op_flux_wpl_col)
        self.plot_cr()
        self.savefile()

    def calc_sf(self):
        """Calculate scaling factors from parallel measurements"""
        self.outdir = Path(self.outdir) / "1-Calculation"
        self.outfile = self.outdir / "data.csv"
        self.readfile(file=self.pm_file)
        self.restrict_flux_quality(flag_col=self.pm_qcf_op_flux_wpl_col, flux_col=self.pm_op_flux_wpl_col)
        # self.custom_class_var()
        self.init_newcols()
        self.detect_daytime(swin=self.df[self.pm_swin_col])
        self.aerodynamic_resistance(u=self.df[self.pm_u_col], ustar=self.df[self.pm_ustar_col], rem_outliers=True)
        self.dry_air_density(rho_v=self.df[self.pm_rho_v_col], rho_a=self.df[self.pm_rho_a_col])
        self.instrument_surface_temperature(ta=self.df[self.pm_ta_col])
        self.unscaled_flux_correction_term(ts=self.df[self.ts_col],
                                           ta=self.df[self.pm_ta_col],
                                           qc_umol=self.df[self.pm_qc_mmol_col].multiply(1000),
                                           ra=self.df[self.ra_col],
                                           rho_v=self.df[self.pm_rho_v_col],
                                           rho_d=self.df[self.rho_d_col],
                                           rem_outliers=True,
                                           lut_gapfill=True)
        self.scaling_factors_df = self.optimize_scaling_factors(class_var_col=self.pm_class_var_col)
        self.assign_scaling_factors(classvar_col=self.pm_class_var_col, lut_gapfill=True)
        self.calc_corrected_fluxes(op_co2_flux_nocorr_col=self.pm_op_flux_wpl_col)
        self.stats()
        self.plot_pm()
        self.savefile()

    def restrict_flux_quality(self, flag_col, flux_col):
        """Keep only fluxes of a certain quality"""
        _filter_qcf = self.df[flag_col] == 0
        self.df[flux_col] = self.df.loc[_filter_qcf, flux_col]

    def readfile(self, file, nrows=None):
        """Read file to dataframe

        :param file: str
        :return:
        """
        self.df = selfheating_files.read(src=file, nrows=nrows)

    # def custom_class_var(self):
    #     """Select custom variable that is then used to divide data into x classes
    #
    #     This method enables the calculation of scaling factors in dependence of the
    #     time stamp, e.g. scaling factors for each month.
    #     """
    #     if self.class_var_col == 'custom':
    #         self.df[('_custom', '--')] = self.df.index.month  #
    #         self.num_classes = None
    #         self.class_var_col = ('_custom', '--')

    def detect_daytime(self, swin):
        # Daytime, nighttime
        self.df[self.daytime_col], self.daytime_filter, self.nighttime_filter = \
            selfheating_frames.detect_daytime(swin=swin)

    def stats(self):
        cols = [self.pm_op_flux_wpl_col,
                self.pm_cp_trueflux_col]

        cols.append(self.op_co2_flux_corr_col)

        _stats_df = self.df[cols].copy()
        _stats_df = _stats_df.dropna()
        _numvals = len(_stats_df)

        print("\nCUMULATIVE FLUXES:")
        print(f"Values: {_numvals}")
        _cumsum_opnocorr = _stats_df[self.pm_op_flux_wpl_col].sum()
        _cumsum_cptrueflux = _stats_df[self.pm_cp_trueflux_col].sum()
        _perc = (_cumsum_opnocorr / _cumsum_cptrueflux) * 100
        print(f"OPEN-PATH (uncorrected): {_cumsum_opnocorr:.0f}  ({_perc:.1f}% of true flux)")
        print(f"ENCLOSED-PATH (true flux): {_cumsum_cptrueflux:.0f}")
        _cumsum = _stats_df[self.op_co2_flux_corr_col].sum()
        _perc = (_cumsum / _cumsum_cptrueflux) * 100
        print(f"OPEN-PATH (corrected): {_cumsum:.0f}  ({_perc:.1f}% of true flux)")
        print("\n\n")

    def savefile(self):
        self.df.to_csv(self.outfile)

    def init_newcols(self):
        self.df[self.daytime_col] = np.nan
        self.df[self.ra_col] = np.nan
        self.df[self.rho_d_col] = np.nan
        self.df[self.ts_col] = np.nan
        self.df[self.fct_unsc_col] = np.nan
        self.df[self.fct_unsc_gf_col] = np.nan
        self.df[self.fct_unsc_lutvals_col] = np.nan
        self.df[self.class_var_group_col] = np.nan
        self.df[self.sf_col] = np.nan
        self.df[self.sf_lutvals_col] = np.nan
        self.df[self.class_var_group_col] = np.nan
        self.df[self.fct_col] = np.nan
        self.df[self.op_co2_flux_corr_col] = np.nan

    def plot_cr(self):
        """Plot results from applying scaling factors from file to data that needs correction"""
        selfheating_plots.SeriesVars(plot_df=self.df.copy(), outdir=self.outdir)
        selfheating_plots.SeriesFlux(outdir=self.outdir,
                                     daytime=self.df[self.daytime_col],
                                     uncorrected_flux=self.df[self.cr_op_flux_wpl_col],
                                     corrected_flux=self.df[self.op_co2_flux_corr_col])
        selfheating_plots.DielCyclesVars(plot_df=self.df.copy(), outdir=self.outdir)
        selfheating_plots.DielCyclesFlux(df=self.df.copy(), outdir=self.outdir,
                                         uncorrected_flux_col=self.cr_op_flux_wpl_col,
                                         corrected_flux_col=self.op_co2_flux_corr_col)
        selfheating_plots.CumulativeFlux(df=self.df.copy(), outdir=self.outdir,
                                         daytime_col=self.daytime_col,
                                         uncorrected_flux_col=self.cr_op_flux_wpl_col,
                                         corrected_flux_col=self.op_co2_flux_corr_col)

    def plot_pm(self):
        """Plot results from scaling factor calculations from parallel measurements"""
        # pass
        selfheating_plots.SeriesVars(plot_df=self.df.copy(), outdir=self.outdir)
        selfheating_plots.SeriesFlux(outdir=self.outdir,
                                     daytime=self.df[self.daytime_col],
                                     uncorrected_flux=self.df[self.pm_op_flux_wpl_col],
                                     true_flux=self.df[self.pm_cp_trueflux_col],
                                     corrected_flux=self.df[self.op_co2_flux_corr_col])
        selfheating_plots.DielCyclesVars(plot_df=self.df.copy(), outdir=self.outdir)
        selfheating_plots.DielCyclesFlux(df=self.df.copy(), outdir=self.outdir,
                                         uncorrected_flux_col=self.pm_op_flux_wpl_col,
                                         corrected_flux_col=self.op_co2_flux_corr_col,
                                         true_flux_col=self.pm_cp_trueflux_col)
        selfheating_plots.CumulativeFlux(df=self.df.copy(), outdir=self.outdir,
                                         daytime_col=self.daytime_col,
                                         uncorrected_flux_col=self.pm_op_flux_wpl_col,
                                         corrected_flux_col=self.op_co2_flux_corr_col,
                                         true_flux_col=self.pm_cp_trueflux_col)

    def calc_corrected_fluxes(self, op_co2_flux_nocorr_col):
        # Use only a fraction of the unscaled flux correction term
        # Add scaled correction flux to original WPL-only open-path flux
        self.df[self.op_co2_flux_corr_col], self.df[self.fct_col] = \
            selfheating_calc.corrected_flux(uncorrected_flux=self.df[op_co2_flux_nocorr_col],
                                            fct_unsc_gf=self.df[self.fct_unsc_gf_col],
                                            sf_gf=self.df[self.sf_gf_col])

    def assign_scaling_factors(self, classvar_col: tuple, file: str = None, lut_gapfill: bool = False):
        # TODO Since the scaling factors are assigned using the LUT, there are probably some flux values where the class variable (ustar) is outside the lookup range the LUT can provide.
        # TODO For these fluxes, the last known scaling factor is used.
        if file:
            file = Path(file)
            self.scaling_factors_df = pd.read_csv(file)

        for ix, row in self.scaling_factors_df.iterrows():
            _filter_group = row
            _filter_group = (self.df[self.daytime_col] == row['DAYTIME']) \
                            & (self.df[classvar_col] >= row['GROUP_CLASSVAR_MIN']) \
                            & (self.df[classvar_col] <= row['GROUP_CLASSVAR_MAX'])
            self.df.loc[_filter_group, self.class_var_group_col] = row['GROUP_CLASSVAR']
            self.df.loc[_filter_group, self.sf_col] = row['SF_MEDIAN']

        if lut_gapfill:
            self.df[self.sf_gf_col], self.df[self.sf_lutvals_col] = \
                selfheating_calc.gapfilling_lut(self.df[self.sf_col])

    def optimize_scaling_factors(self, class_var_col):
        optimize = OptimizeScalingFactors(df=self.df,
                                          daytime_col=self.daytime_col,
                                          class_var_col=class_var_col,
                                          class_var_group_col=self.class_var_group_col,
                                          num_classes=self.pm_num_classes,
                                          outdir=self.outdir)
        return optimize.get()

    def aerodynamic_resistance(self, u, ustar, rem_outliers: bool = False):
        # Aerodynamic resistance
        self.df[self.ra_col] = \
            aerodynamic_resistance(u_ms=u,
                                   ustar_ms=ustar)
        if rem_outliers:
            self.df[self.ra_col] = \
                selfheating_calc.remove_outliers(series=self.df[self.ra_col], plot_title="X", n_sigmas=20)

    def dry_air_density(self, rho_v, rho_a):
        # Dry air density
        self.df[self.rho_d_col] = \
            dry_air_density(rho_v=rho_v,
                            rho_a=rho_a,
                            M_AIR=self.M_AIR,
                            M_H2O=self.M_H2O)

    def instrument_surface_temperature(self, ta):
        self.df[self.ts_col] = \
            selfheating_calc.surface_temp_jar09(ta=ta,
                                                daytime_filter=self.daytime_filter,
                                                nighttime_filter=self.nighttime_filter)

    def unscaled_flux_correction_term(self, ts, ta, qc_umol, ra, rho_v, rho_d,
                                      lut_gapfill: bool = False, rem_outliers: bool = False):
        print("Calculating unscaled flux correction term ...")

        self.df[self.fct_unsc_col] = \
            selfheating_calc.flux_correction_term_unscaled(ts=ts, ta=ta, qc_umol=qc_umol, ra=ra, rho_v=rho_v,
                                                           rho_d=rho_d)

        # Remove outliers
        if rem_outliers:
            self.df[self.fct_unsc_col] = \
                selfheating_calc.remove_outliers(series=self.df[self.fct_unsc_col],
                                                 plot_title="XXX")

        # Gap-filling (LUT)
        if lut_gapfill:
            self.df[self.fct_unsc_gf_col], self.df[self.fct_unsc_lutvals_col] = \
                selfheating_calc.gapfilling_lut(series=self.df[self.fct_unsc_col])


class OptimizeScalingFactors(selfheating_newcols.NewCols):

    def __init__(self, df, daytime_col, class_var_col, class_var_group_col,
                 num_classes, outdir, userconfig):
        self.userconfig = userconfig
        self.df = df
        self.daytime_col = daytime_col
        self.class_var_col = class_var_col
        self.class_var_group_col = class_var_group_col
        self.num_classes = num_classes

        # If data are not bootstrapped, set flag to False
        if self.pm_num_bootstrap_runs == 0:
            self.bootstrapped = False
            self.pm_num_bootstrap_runs = 9999  # Set to arbitrary number
        else:
            self.bootstrapped = True

        if self.class_var_col[0] == '_custom':
            num_classes = len(self.df[self.class_var_col].unique())
        self.scaling_factors_df = selfheating_frames.init_scaling_factors_df(num_classes=num_classes)

        self.outfile_csv = outdir / "self-heating_scaling_factors.csv"
        self.outfile_plot = outdir / "self-heating_scaling_factors_per_class.png"

        self.run()

    def run(self):
        self.bootstrapping()
        self.plot()
        self.print_stats()
        self.savefile()

    def plot(self):
        selfheating_plots.ScalingFactors(plot_df=self.scaling_factors_df,
                                         outplot=self.outfile_plot,
                                         classvar_col=self.class_var_col,
                                         userconfig=self.userconfig)

    def savefile(self):
        print(f"--> Saving scaling factors to file: {self.outfile_csv}")
        self.scaling_factors_df.to_csv(self.outfile_csv)

    def print_stats(self):
        print("BOOTSTRAPPING RESULTS")
        print(f"class variable: {self.class_var_col}")
        print(f"number of classes: {self.num_classes}")

        print("\nSF results:")
        print("====================")
        print(f"scaling factor (daytime median): {self.scaling_factors_df.loc[1, 'SF_MEDIAN'].median():.3f}")
        print(
            f"scaling factor (nighttime median): {self.scaling_factors_df.loc[0, 'SF_MEDIAN'].median():.3f}")
        print(f"scaling factor (overall median): {self.scaling_factors_df.loc[:, 'SF_MEDIAN'].median():.3f}")
        print(
            f"sum of squares (daytime median): {self.scaling_factors_df.loc[1, 'SOS_MEDIAN'].median():.3f}")
        print(
            f"sum of squares (nighttime median): {self.scaling_factors_df.loc[0, 'SOS_MEDIAN'].median():.3f}")
        print(
            f"sum of squares (overall median): {self.scaling_factors_df.loc[:, 'SOS_MEDIAN'].median():.3f}")

    def get(self):
        return self.scaling_factors_df

    def bootstrapping(self):

        # Group data records by daytime / nighttime membership
        _grouped_by_daynighttime = self.df.groupby(self.daytime_col)

        # Loop through daytime / nighttime data
        for _group_daynighttime, _group_daynighttime_df in _grouped_by_daynighttime:

            if self.class_var_col[0] == '_custom':
                _group_daynighttime_df[self.class_var_group_col] = \
                    _group_daynighttime_df[self.class_var_col]
            else:
                # Divide data into x class variable groups w/ same number of values
                _group_daynighttime_df[self.class_var_group_col] = \
                    pd.qcut(_group_daynighttime_df[self.class_var_col],
                            q=self.num_classes, labels=False)

            # Group data records by class variable membership
            _grouped_by_class_var = _group_daynighttime_df.groupby(self.class_var_group_col)

            # Loop through class variable groups
            for _group_class_var, _group_class_var_df in _grouped_by_class_var:
                # Bootstrap group data

                bts_factors = []
                bts_sum_of_squares = []
                bts_num_vals = []

                for bts_run in range(0, self.pm_num_bootstrap_runs):
                    num_rows = int(len(_group_class_var_df.index))

                    if self.bootstrapped:
                        # Use bootstrapped data
                        bts_sample_df = _group_class_var_df.sample(n=num_rows,
                                                                   replace=True)  # Take sample w/ replacement
                    else:
                        # Use measured data in cas
                        bts_sample_df = _group_class_var_df

                    bts_sample_df.sort_index(inplace=True)

                    result = self.optimize_factor(target=bts_sample_df[self.pm_op_co2_flux_nocorr_col],
                                                  reference=bts_sample_df[self.pm_cp_co2_flux_col],
                                                  fct_unsc_gf=bts_sample_df[self.fct_unsc_gf_col])
                    bts_factors.append(result.x)  # x = scaling factor
                    bts_sum_of_squares.append(result.fun)
                    bts_num_vals.append(bts_sample_df[self.class_var_col].count())

                    # Break if only working with measured data (no bootstrapping)
                    if not self.bootstrapped:
                        break

                print(f"Finished {self.pm_num_bootstrap_runs} bootstrap runs for group {_group_class_var} "
                      f"in daytime = {_group_daynighttime}")

                # Stats, aggregates for current class group
                location = tuple([_group_daynighttime, _group_class_var])
                self.scaling_factors_df.loc[location, f'DAYTIME'] = _group_daynighttime
                self.scaling_factors_df.loc[location, f'GROUP_CLASSVAR'] = _group_class_var
                self.scaling_factors_df.loc[location, f'GROUP_CLASSVAR_MIN'] = _group_class_var_df[
                    self.class_var_col].min()
                self.scaling_factors_df.loc[location, f'GROUP_CLASSVAR_MAX'] = _group_class_var_df[
                    self.class_var_col].max()
                self.scaling_factors_df.loc[location, f'BOOTSTRAP_RUNS'] = self.pm_num_bootstrap_runs

                self.scaling_factors_df.loc[location, f'SF_MEDIAN'] = np.median(bts_factors)
                self.scaling_factors_df.loc[location, f'SOS_MEDIAN'] = np.median(
                    bts_sum_of_squares)
                self.scaling_factors_df.loc[location, f'NUMVALS_AVG'] = np.mean(bts_num_vals)
                self.scaling_factors_df.loc[location, f'SF_Q25'] = np.quantile(bts_factors, 0.25)
                self.scaling_factors_df.loc[location, f'SF_Q75'] = np.quantile(bts_factors, 0.75)
                self.scaling_factors_df.loc[location, f'SF_Q01'] = np.quantile(bts_factors, 0.01)
                self.scaling_factors_df.loc[location, f'SF_Q99'] = np.quantile(bts_factors, 0.99)

    def optimize_factor(self, target, reference, fct_unsc_gf):
        """Optimize factor by minimizing sum of squares b/w corrected target and reference"""
        optimization_df = pd.DataFrame()
        optimization_df['target'] = target
        optimization_df['reference'] = reference
        optimization_df['fct_unscaled_col'] = fct_unsc_gf
        optimization_df = optimization_df.dropna()

        target = optimization_df['target'].copy()
        reference = optimization_df['reference'].copy()
        fct_unsc_gf = optimization_df['fct_unscaled_col'].copy()  # Unscaled flux correction term

        result = minimize_scalar(self.calc_sumofsquares, args=(fct_unsc_gf, target, reference),
                                 method='Bounded', bounds=[-1, 4])
        # result = minimize_scalar(self.calc_sumofsquares, args=(fct_unscaled, target, reference),
        #                          method='Golden', tol=0)
        # print("=" * 40)
        return result

    def calc_sumofsquares(self, factor, unsc_flux_corr_term, target, reference):
        corrected = target + unsc_flux_corr_term.multiply(factor)

        diff2 = np.sqrt((reference - corrected) ** 2)
        sum_of_squares = diff2.sum()

        # diff2 = (reference - corrected)
        # # diff2 = diff2.abs()
        # sum_of_squares = diff2.sum()

        # diff2=(corrected.corr(reference))**2
        # sum_of_squares = 1 - diff2

        # diff2 = corrected.sum() - reference.sum()
        # sum_of_squares = np.abs(diff2)

        # print(f"factor: {factor}  sos: {sum_of_squares}")
        return sum_of_squares


def main():
    AIR_CP = "AIR_CP"
    AIR_DENSITY = "AIR_DENSITY"
    VAPOR_DENSITY = "VAPOR_DENSITY"
    U = "U"
    USTAR = "USTAR"
    TA = "TA_T1_47_1_gfXG"
    SWIN = "SW_IN_T1_47_1_gfXG"
    CO2_MOLAR_DENSITY = "CO2_MOLAR_DENSITY"
    FLUX_72 = "NEE_L3.1_L3.3_CUT_50_QCF_IRGA72"
    FLUX_75 = "NEE_L3.1_L3.3_CUT_50_QCF_IRGA75"

    scop = Scop(
        site="CH-LAE",
        title="CH-LAE self-heating correction",
        # outdir="XXX",
        # pm_file="XXX",
        pm_op_flux_wpl_col=FLUX_75,
        # pm_qcf_op_flux_wpl_col="XXX",
        pm_cp_trueflux_col=FLUX_72,
        # pm_qcf_cp_trueflux_col="XXX",
        pm_airheatcap_jkkg_col=AIR_CP,
        pm_qc_mmol_col=CO2_MOLAR_DENSITY,
        pm_u_col=U,
        pm_ustar_col=USTAR,
        pm_rho_v_col=VAPOR_DENSITY,
        pm_rho_a_col=AIR_DENSITY,
        pm_ta_col=TA,
        pm_swin_col=SWIN,
        pm_num_classes=20,
        pm_num_bootstrap_runs=0,
        pm_class_var_col=USTAR,
        # cr_file="XXX",
        # cr_sf_file="XXX",
        cr_ta_col=TA,
        cr_swin_col=SWIN,
        cr_ustar_col=USTAR,
        cr_u_col=U,
        cr_rho_v_col=VAPOR_DENSITY,
        cr_rho_a_col=AIR_DENSITY,
        cr_qc_mmol_col=CO2_MOLAR_DENSITY,
        cr_op_flux_wpl_col=FLUX_75,
        cr_class_var_col=USTAR
    )
    # scop.calc_sf()
    # apply_scaling_factors = Scop().apply_sf()
    print("End.")


if __name__ == '__main__':
    main()
