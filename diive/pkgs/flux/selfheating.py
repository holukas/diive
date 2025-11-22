"""
References:
    (BUR06) Burba et al. (2006). Correcting apparent off-season CO2 uptake due
        to surface heating of an open path gas analyzer: Progress report
        of an ongoing study. 13.
    (JAR09) Järvi, L., Mammarella, I., Eugster, W., Ibrom, A., Siivola, E., Dellwik,
        E., Keronen, P., Burba, G., & Vesala, T. (2009). Comparison of net CO2
        fluxes measured with open- and closed-path infrared gas analyzers in
        an urban complex environment. 14, 16.
    (KIT17) Kittler et al. (2017). High-quality eddy-covariance CO2 budgets
        under cold climate conditions: Arctic Eddy-Covariance CO2 Budgets.
        Journal of Geophysical Research: Biogeosciences, 122(8), 2064–2084.
        https://doi.org/10.1002/2017JG003830

"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import selfheating_calc
import selfheating_constants
import selfheating_files
import selfheating_frames
import selfheating_newcols
import selfheating_plots
from diive.core.plotting.plotfuncs import default_format
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
            inputdf: pd.DataFrame,
            site: str,
            title: str,
            flux_openpath: str,
            flux_closedpath: str,
            air_heat_capacity: str,
            co2_molar_density: str,
            u: str,
            ustar: str,
            water_vapor_density: str,
            air_density: str,
            air_temperature: str,
            swin: str,
            n_classes: int,
            n_bootstrap_runs: int,
            classvar: str,
            cr_ta_col: str,
            cr_swin_col: str,
            cr_ustar_col: str,
            cr_u_col: str,
            cr_rho_v_col: str,
            cr_rho_a_col: str,
            cr_qc_mmol_col: str,
            cr_op_flux_wpl_col: str,
            cr_class_var_col: str,
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
            flux_openpath (str): Column name for OP flux with WPL correction in PM file.
            flux_closedpath (str): Column name for CP true flux in PM file.
            air_heat_capacity (str): Column name for specific heat at constant pressure of
                ambient air (J K-1 kg-1) in PM file.
            co2_molar_density (str): Column name for CO2 molar density column (mmol m-3) in PM file. [qc]
            u (str): Column name for horizontal wind speed (m s-1) in PM file.
            ustar (str): Column name for USTAR friction velocity (m s-1) in PM file.
            water_vapor_density (str): Column name for water vapor density (kg m-3) in PM file. [rho_v]
            air_density (str): Column name for air density (kg m-3) in PM file. [rho_a]
            air_temperature (str): Column name for ambient air temperature (°C) in PM file.
            swin (str): Column name for shortwave-incoming radiation (W m-2) in PM file.
            n_classes (int): Number of classes for categorization in PM processing, ignored
                if class_var_col = 'custom'.
            n_bootstrap_runs (int): Number of bootstraps in each class, *0* uses measured
                data only w/o bootstrapping.
            classvar (str): Column name for the class variable in PM file. Each class
                has its own scaling factor.
        """
        super().__init__()
        self.inputdf = inputdf.copy()
        self.site = site
        self.title = title
        self.flux_openpath = flux_openpath
        self.flux_closedpath = flux_closedpath
        self.pm_airheatcap_jkkg_col = air_heat_capacity
        self.co2_molar_density = co2_molar_density
        self.u = u
        self.ustar = ustar
        self.water_vapor_density = water_vapor_density
        self.air_density = air_density
        self.air_temperature = air_temperature
        self.swin = swin
        self.n_classes = n_classes
        self.n_bootstrap_runs = n_bootstrap_runs
        self.classvar = classvar

        self.df = pd.DataFrame()
        self.scaling_factors_df = pd.DataFrame()

    def calc_sf(self):
        """Calculate scaling factors from parallel measurements"""
        df = self.inputdf.copy()
        df = self.init_newcols(df=df)
        df[self.daytime] = self.detect_daytime(swin=df[self.swin])
        df[self.aerodynamic_resistance] = self.calc_aerodynamic_resistance(
            u=df[self.u], ustar=df[self.ustar], rem_outliers=True)
        df[self.dry_air_density] = self.calc_dry_air_density(
            rho_v=df[self.water_vapor_density], rho_a=df[self.air_density])
        df[self.t_instrument_surface] = self.calc_instrument_surface_temperature(
            ta=df[self.air_temperature], daytime=df[self.daytime])
        df[self.fct_unsc], df[self.fct_unsc_gf], df[self.fct_unsc_lutvals] = (
            self.calc_unscaled_flux_correction_term(
                t_instrument_surface=df[self.t_instrument_surface],
                air_temperature=df[self.air_temperature],
                co2_molar_density=df[self.co2_molar_density].multiply(1000),
                aerodynamic_resistance=df[self.aerodynamic_resistance],
                water_vapor_density=df[self.water_vapor_density],
                dry_air_density=df[self.dry_air_density],
                rem_outliers=True,
                lut_gapfill=True))
        scaling_factors_df = self.optimize_scaling_factors(df=df, class_var_col=self.classvar)
        df = self.assign_scaling_factors(df=df, scaling_factors_df=scaling_factors_df,
                                         classvar_col=self.classvar, lut_gapfill=True)
        df[self.nee_op_corr], df[self.fct] = selfheating_calc.corrected_flux(
            uncorrected_flux=df[self.flux_openpath],
            fct_unsc_gf=df[self.fct_unsc_gf],
            sf_gf=df[self.sf_gf])
        self.stats(df=df)
        self.plot_pm(df=df)

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

    def detect_daytime(self, swin) -> pd.Series:
        # Daytime, nighttime
        nighttime_filter = swin <= 20
        daytime_filter = swin > 20
        daytime_series = pd.Series(index=swin.index)
        daytime_series.loc[nighttime_filter] = 0
        daytime_series.loc[daytime_filter] = 1
        return daytime_series

    def stats(self, df: pd.DataFrame):
        cols = [self.flux_openpath, self.flux_closedpath, self.nee_op_corr]

        _stats_df = df[cols].copy()
        _stats_df = _stats_df.dropna()
        _numvals = len(_stats_df)

        print("\nCUMULATIVE FLUXES:")
        print(f"Values: {_numvals}")
        _cumsum_opnocorr = _stats_df[self.flux_openpath].sum()
        _cumsum_cptrueflux = _stats_df[self.flux_closedpath].sum()
        _perc = (_cumsum_opnocorr / _cumsum_cptrueflux) * 100
        print(f"OPEN-PATH (uncorrected): {_cumsum_opnocorr:.0f}  ({_perc:.1f}% of true flux)")
        print(f"ENCLOSED-PATH (true flux): {_cumsum_cptrueflux:.0f}")
        _cumsum = _stats_df[self.nee_op_corr].sum()
        _perc = (_cumsum / _cumsum_cptrueflux) * 100
        print(f"OPEN-PATH (corrected): {_cumsum:.0f}  ({_perc:.1f}% of true flux)")
        print("\n\n")

    # def savefile(self):
    #     self.df.to_csv(self.outfile)

    def init_newcols(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.daytime] = np.nan
        df[self.aerodynamic_resistance] = np.nan
        df[self.dry_air_density] = np.nan
        df[self.t_instrument_surface] = np.nan
        df[self.fct_unsc] = np.nan
        df[self.fct_unsc_gf] = np.nan
        df[self.fct_unsc_lutvals] = np.nan
        df[self.class_var_group] = np.nan
        df[self.sf] = np.nan
        df[self.sf_lutvals_col] = np.nan
        df[self.class_var_group] = np.nan
        df[self.fct] = np.nan
        df[self.nee_op_corr] = np.nan
        return df

    def plot_pm(self, df: pd.DataFrame):
        """Plot results from scaling factor calculations from parallel measurements"""
        # selfheating_plots.SeriesVars(plot_df=df.copy())
        # selfheating_plots.SeriesFlux(daytime=df[self.daytime_col],
        #                              uncorrected_flux=df[self.flux_openpath],
        #                              true_flux=df[self.flux_closedpath],
        #                              corrected_flux=df[self.op_co2_flux_corr_col])
        # selfheating_plots.DielCyclesVars(plot_df=df.copy())
        # selfheating_plots.DielCyclesFlux(df=df.copy(),
        #                                  uncorrected_flux_col=self.flux_openpath,
        #                                  corrected_flux_col=self.op_co2_flux_corr_col,
        #                                  true_flux_col=self.flux_closedpath)
        selfheating_plots.CumulativeFlux(df=df.copy(),
                                         daytime_col=self.daytime,
                                         uncorrected_flux_col=self.flux_openpath,
                                         corrected_flux_col=self.nee_op_corr,
                                         true_flux_col=self.flux_closedpath)

    def calc_corrected_fluxes(self, op_co2_flux_nocorr_col):
        # Use only a fraction of the unscaled flux correction term
        # Add scaled correction flux to original WPL-only open-path flux
        return selfheating_calc.corrected_flux(uncorrected_flux=self.df[op_co2_flux_nocorr_col],
                                               fct_unsc_gf=self.df[self.fct_unsc_gf],
                                               sf_gf=self.df[self.sf_gf])

    def assign_scaling_factors(self, df: pd.DataFrame, scaling_factors_df: pd.DataFrame,
                               classvar_col: str, lut_gapfill: bool = False) -> pd.DataFrame:
        # TODO Since the scaling factors are assigned using the LUT, there are probably some flux values where the
        # TODO class variable (ustar) is outside the lookup range the LUT can provide.
        # TODO For these fluxes, the last known scaling factor is used.

        for ix, row in scaling_factors_df.iterrows():
            _filter_group = row
            _filter_group = (df[self.daytime] == row['DAYTIME']) \
                            & (df[classvar_col] >= row['GROUP_CLASSVAR_MIN']) \
                            & (df[classvar_col] <= row['GROUP_CLASSVAR_MAX'])
            df.loc[_filter_group, self.class_var_group] = row['GROUP_CLASSVAR']
            df.loc[_filter_group, self.sf] = row['SF_MEDIAN']

        if lut_gapfill:
            df[self.sf_gf], df[self.sf_lutvals_col] = \
                selfheating_calc.gapfilling_lut(df[self.sf])
        return df

    def optimize_scaling_factors(self, df: pd.DataFrame, class_var_col: str):
        optimize = OptimizeScalingFactors(df=df,
                                          daytime_col=self.daytime,
                                          class_var_col=class_var_col,
                                          class_var_group_col=self.class_var_group,
                                          num_classes=self.n_classes,
                                          # outdir=self.outdir,
                                          pm_num_bootstrap_runs=self.n_bootstrap_runs,
                                          pm_op_co2_flux_nocorr_col=self.flux_openpath,
                                          pm_cp_co2_flux_col=self.flux_closedpath)
        return optimize.get()

    def calc_aerodynamic_resistance(self, u, ustar, rem_outliers: bool = False) -> pd.Series:
        # Aerodynamic resistance
        ra = aerodynamic_resistance(u_ms=u, ustar_ms=ustar)
        if rem_outliers:
            ra = selfheating_calc.remove_outliers(series=ra, plot_title="X", n_sigmas=20)
        return ra

    def calc_dry_air_density(self, rho_v, rho_a) -> pd.Series:
        # Dry air density
        return dry_air_density(rho_v=rho_v, rho_a=rho_a)

    def calc_instrument_surface_temperature(self, ta, daytime) -> pd.Series:
        return selfheating_calc.surface_temp_jar09(ta=ta, daytime=daytime)

    def calc_unscaled_flux_correction_term(self, t_instrument_surface, air_temperature, co2_molar_density, aerodynamic_resistance, water_vapor_density, dry_air_density,
                                           lut_gapfill: bool = False, rem_outliers: bool = False):
        print("\nCalculating unscaled flux correction term ...")
        fct_unsc_gf = pd.Series()
        fct_unsc_lutvals = pd.Series()

        fct_unsc = selfheating_calc.flux_correction_term_unscaled(
            ts=t_instrument_surface, ta=air_temperature, qc_umol=co2_molar_density, ra=aerodynamic_resistance, rho_v=water_vapor_density, rho_d=dry_air_density)

        # Remove outliers
        if rem_outliers:
            fct_unsc = selfheating_calc.remove_outliers(series=fct_unsc, plot_title="XXX")

        # Gap-filling (LUT)
        if lut_gapfill:
            fct_unsc_gf, fct_unsc_lutvals = selfheating_calc.gapfilling_lut(series=fct_unsc)
        return fct_unsc, fct_unsc_gf, fct_unsc_lutvals


class OptimizeScalingFactors(selfheating_newcols.NewCols):

    def __init__(self, df, daytime_col, class_var_col, class_var_group_col,
                 num_classes, pm_num_bootstrap_runs, pm_op_co2_flux_nocorr_col, pm_cp_co2_flux_col):
        # self.userconfig = userconfig
        self.df = df
        self.daytime_col = daytime_col
        self.class_var_col = class_var_col
        self.class_var_group_col = class_var_group_col
        self.num_classes = num_classes
        self.pm_num_bootstrap_runs = pm_num_bootstrap_runs
        self.pm_op_co2_flux_nocorr_col = pm_op_co2_flux_nocorr_col
        self.pm_cp_co2_flux_col = pm_cp_co2_flux_col

        # If data are not bootstrapped, set flag to False
        if self.pm_num_bootstrap_runs == 0:
            self.bootstrapped = False
            self.pm_num_bootstrap_runs = 9999  # Set to arbitrary number
        else:
            self.bootstrapped = True

        if self.class_var_col[0] == '_custom':
            num_classes = len(self.df[self.class_var_col].unique())
        self.scaling_factors_df = selfheating_frames.init_scaling_factors_df(num_classes=num_classes)

        self.run()

    def run(self):
        self.bootstrapping()
        self.plot()
        self.print_stats()

    def plot(self):
        # sf = selfheating_plots.ScalingFactors(plot_df=self.scaling_factors_df,
        #                                       # outplot=self.outfile_plot,
        #                                       classvar_col=self.class_var_col,
        #                                       # userconfig=self.userconfig,
        #                                       pm_num_bootstrap_runs=self.pm_num_bootstrap_runs)

        plot_df = self.scaling_factors_df.copy()
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.figure(facecolor='white', figsize=(12, 9))
        ax1 = fig.add_subplot(gs[0, 0])

        marker_cycle = {'day': 'o', 'night': 's'}
        color_cycle = {'day': '#FF9800',
                       'night': '#3F51B5'}  # Orange 500 / Indigo 500; https://www.materialui.co/colors
        xmax = ymax = -9999
        xmin = ymin = 9999
        # Separate groups for daytime and nighttime data
        _daynight_grouped = plot_df.groupby('DAYTIME')
        for _daynight_group_ix, _daynight_group_df in _daynight_grouped:
            _time = 'day' if _daynight_group_ix == 1 else 'night'
            ax1.plot(_daynight_group_df['GROUP_CLASSVAR_MIN'], _daynight_group_df['SF_MEDIAN'],
                     color=color_cycle[_time],
                     alpha=1, ls='-',
                     marker=marker_cycle[_time], markeredgecolor='none', ms=4, zorder=98,
                     label=f'{_time}: median from {self.pm_num_bootstrap_runs}x bootstrap, '
                           f'incl. 1-99% and interquartile range ({_daynight_group_df["NUMVALS_AVG"].mean():.0f} values per data point)')
            ax1.fill_between(_daynight_group_df['GROUP_CLASSVAR_MIN'], _daynight_group_df['SF_Q99'],
                             _daynight_group_df['SF_Q01'],
                             color=color_cycle[_time], alpha=0.1, lw=0)
            ax1.fill_between(_daynight_group_df['GROUP_CLASSVAR_MIN'], _daynight_group_df['SF_Q75'],
                             _daynight_group_df['SF_Q25'],
                             color=color_cycle[_time], alpha=0.2, lw=0)

            #     fit_line = ax.plot(px, nom, c='black', lw=2, zorder=99, alpha=1, label="prediction for higher classes with 95% region")
            #     fit_confidence_intervals = ax.fill_between(px, nom - 1.96 * std, nom + 1.96 * std, alpha=.2, color='#90A4AE', zorder=1)  # uncertainty lines (95% confidence)

            xmax = _daynight_group_df['GROUP_CLASSVAR_MIN'].max() if _daynight_group_df[
                                                                         'GROUP_CLASSVAR_MIN'].max() > xmax else xmax
            xmin = _daynight_group_df['GROUP_CLASSVAR_MIN'].min() if _daynight_group_df[
                                                                         'GROUP_CLASSVAR_MIN'].min() < xmin else xmin
            ymax = _daynight_group_df['SF_Q75'].max() if _daynight_group_df['SF_Q75'].max() > ymax else ymax
            ymin = _daynight_group_df['SF_Q25'].min() if _daynight_group_df['SF_Q25'].min() < ymin else ymin
            ymax = _daynight_group_df['SF_Q99'].max() if _daynight_group_df['SF_Q99'].max() > ymax else ymax
            ymin = _daynight_group_df['SF_Q01'].min() if _daynight_group_df['SF_Q01'].min() < ymin else ymin

        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_title(f"Scaling factors per {self.class_var_col} class, for daytime and nighttime")
        ax1.set_xlabel('class variable (units)', color='black', fontsize=12, fontweight='bold')
        ax1.set_ylabel('scaling factor ξ', color='black', fontsize=12, fontweight='bold')
        ax1.axhline(0, lw=1, color='black')
        ax1.legend()
        default_format(ax=ax1)

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
                                                  fct_unsc_gf=bts_sample_df[self.fct_unsc_gf])
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
    from diive.core.io.files import load_parquet
    df = load_parquet(
        filepath=r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\30_FLUX_PROCESSING_CHAIN\32_SELF-HEATING_CORRECTION\21_MERGED_IRGA75-noSHC+IRGA72_FluxProcessingChain_after-L3.2_NEE_QCF10_2016-2017.parquet")

    AIR_CP = "AIR_CP_IRGA72"
    AIR_DENSITY = "AIR_DENSITY_IRGA72"
    VAPOR_DENSITY = "VAPOR_DENSITY_IRGA72"
    U = "U_IRGA72"
    USTAR = "USTAR_IRGA72"
    TA = "TA_T1_47_1_gfXG_IRGA72"
    SWIN = "SW_IN_T1_47_1_gfXG_IRGA72"
    CO2_MOLAR_DENSITY = "CO2_MOLAR_DENSITY_IRGA72"
    FLUX_72 = "NEE_L3.1_L3.2_QCF_IRGA72"
    FLUX_75 = "NEE_L3.1_L3.2_QCF_IRGA75"

    scop = Scop(
        inputdf=df,
        site="CH-LAE",
        title="CH-LAE self-heating correction",
        # outdir="XXX",
        # pm_file="XXX",
        flux_openpath=FLUX_75,
        # pm_qcf_op_flux_wpl_col="XXX",
        flux_closedpath=FLUX_72,
        # pm_qcf_cp_trueflux_col="XXX",
        air_heat_capacity=AIR_CP,
        co2_molar_density=CO2_MOLAR_DENSITY,
        u=U,
        ustar=USTAR,
        water_vapor_density=VAPOR_DENSITY,
        air_density=AIR_DENSITY,
        air_temperature=TA,
        swin=SWIN,
        n_classes=5,
        n_bootstrap_runs=0,
        classvar=USTAR,
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
    scop.calc_sf()
    # apply_scaling_factors = Scop().apply_sf()
    print("End.")


if __name__ == '__main__':
    main()
