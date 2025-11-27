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
import time
from dataclasses import dataclass
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from diive.pkgs.createvar.air import dry_air_density, aerodynamic_resistance
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS
from diive.pkgs.outlierdetection.hampel import HampelDaytimeNighttime

pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 14)
pd.set_option('display.max_rows', 30)


@dataclass(frozen=True)
class ColumnConfig:
    """Immutable configuration for newly created column names"""
    class_var_group: str = 'GROUP_CLASSVAR'
    daytime: str = 'DAYTIME'
    air_thermal_conductivity: str = 'AIR_THERMAL_CONDUCTIVITY'
    aerodynamic_resistance: str = 'AERODYNAMIC_RESISTANCE'
    dry_air_density: str = 'DRY_AIR_DENSITY'
    t_instr_surface: str = 'T_INSTRUMENT_SURFACE'
    flux_op_corr: str = 'NEE_OP_CORR'
    fct_unsc: str = 'FCT_UNSC'
    fct_unsc_gf: str = 'FCT_UNSC_gfRF'
    fct: str = 'FCT'
    sf: str = 'SF'
    sf_gf: str = '.SF_GF'
    s: str = 'S'  # Sensible heat from all key instrument surfaces (W m-2) (BUR08)


class Scop:
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
            lat: float,
            lon: float,
            utc_offset: int,
            correction_method_base: Literal["JAR09", "BUR06", "BUR08"] = "JAR09",
            remove_outliers_method: Literal["fast", "separate"] = "fast"
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
                ambient air (J K-1 kg-1) in PM file. [c_p]
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

        # Initialize configuration
        self.cols = ColumnConfig()

        self.inputdf = inputdf.copy()
        self.site = site
        self.title = title
        self.flux_openpath = flux_openpath
        self.flux_closedpath = flux_closedpath
        self.air_heat_capacity = air_heat_capacity
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
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset
        self.remove_outliers_method = remove_outliers_method
        self.correction_method_base = correction_method_base

        self.df = pd.DataFrame()
        self.scaling_factors_df = pd.DataFrame()

    def calc_sf(self):
        """Calculate flux correction term and scaling factors from parallel measurements"""
        df = self.inputdf.copy()
        df = self.init_newcols(df=df)

        # Calculate scaling factors
        optimize = ScopOptimizer(df=df,
                                 class_var=self.classvar,
                                 n_classes=self.n_classes,
                                 n_bootstrap_runs=self.n_bootstrap_runs,
                                 flux_openpath=self.flux_openpath,
                                 flux_closedpath=self.flux_closedpath,
                                 showplot=True,
                                 cols=self.cols)
        scaling_factors_df = optimize.get()


class ScopApplicator:

    def __init__(self, fct_unsc: pd.Series, scaling_factors_df: pd.DataFrame, flux_openpath: pd.Series,
                 flux_closedpath: pd.Series, classvar: pd.Series, daytime: pd.Series, swin: pd.Series):

        self.fct_unsc = fct_unsc.copy()
        self.scaling_factors_df = scaling_factors_df.copy()
        self.flux_openpath = flux_openpath.copy()
        self.flux_closedpath = flux_closedpath.copy()
        self.classvar = classvar.copy()
        self.daytime = daytime.copy()
        self.swin = swin.copy()

        self.cols = ColumnConfig()

        frame = {self.fct_unsc.name: self.fct_unsc, self.flux_openpath.name: self.flux_openpath,
                 self.flux_closedpath.name: self.flux_closedpath,
                 self.classvar.name: self.classvar, self.daytime.name: self.daytime, self.swin.name: self.swin}
        self.df = pd.DataFrame(frame)

    def run(self):

        # Assign scaling factors depending on the class var (e.g. USTAR)
        self.df = self._assign_scaling_factors()

        # Final flux correction
        # Corrected OP = uncorrected OP + (FCT_unscaled * ScalingFactor)
        self.df[self.cols.fct] = self.df[self.cols.fct_unsc_gf] * self.df[self.cols.sf]
        self.df[self.cols.flux_op_corr] = self.df[self.flux_openpath.name] + self.df[self.cols.fct]

        # Visualization
        self.stats()
        self.plot_diel_cycles()
        self.plot_flux_analysis_dashboard()

        # self.df = df.copy()

    # def init_newcols(self, df: pd.DataFrame) -> pd.DataFrame:
    #     df[self.cols.daytime] = np.nan
    #     df[self.cols.aerodynamic_resistance] = np.nan
    #     df[self.cols.dry_air_density] = np.nan
    #     df[self.cols.t_instr_surface] = np.nan
    #     df[self.cols.fct_unsc] = np.nan
    #     df[self.cols.fct_unsc_gf] = np.nan
    #     df[self.cols.class_var_group] = np.nan
    #     df[self.cols.sf] = np.nan
    #     df[self.cols.class_var_group] = np.nan
    #     df[self.cols.fct] = np.nan
    #     df[self.cols.flux_op_corr] = np.nan
    #     return df

    def _assign_scaling_factors(self) -> pd.DataFrame:

        # # Initialize destination columns
        # if self.cols.class_var_group not in df.columns:
        #     df[self.cols.class_var_group] = np.nan
        # if self.cols.sf not in df.columns:
        #     df[self.cols.sf] = np.nan

        # Filter for valid keys
        valid_mask = self.classvar.notna() & self.daytime.notna()
        if valid_mask.sum() == 0:
            raise ValueError("No valid keys found in DataFrame. Check class variable and daytime columns.")

        # Prepare left subset
        df_valid = self.df.loc[valid_mask].sort_values(by=str(self.classvar.name))
        cols_needed = [self.classvar.name, self.cols.daytime]
        df_valid = df_valid.loc[valid_mask, cols_needed].sort_values(by=self.classvar.name)

        # Prepare right subset (lookup table)
        sf_sorted = self.scaling_factors_df.sort_values(by='GROUP_CLASSVAR_MIN')

        # Perform backward merge
        # direction='backward': Finds the bin where [classvar] >= [GROUP_CLASSVAR_MIN]
        # - If value is inside range: matches correct bin.
        # - If value is > max: matches the highest bin (conceptually "last known" bin).
        # - If value is < min: returns NaN.
        merged_subset = pd.merge_asof(
            df_valid,
            sf_sorted[['DAYTIME', 'GROUP_CLASSVAR_MIN', 'GROUP_CLASSVAR', 'SF_MEDIAN']],
            left_on=self.classvar.name,
            right_on='GROUP_CLASSVAR_MIN',
            by=self.daytime.name,
            direction='backward'
        )

        # Handle low outliers (< min)
        # Because 'backward' merge produces NaNs for values smaller than the lowest bin
        # These NaNs are filled with the SF of the FIRST bin (first known SF)
        # This is done by grouping by daytime and filling NaNs with the first valid observation
        if merged_subset['SF_MEDIAN'].isna().any():
            merged_subset['SF_MEDIAN'] = merged_subset.groupby(self.cols.daytime)['SF_MEDIAN'] \
                .transform(lambda x: x.bfill())
            merged_subset['GROUP_CLASSVAR'] = merged_subset.groupby(self.cols.daytime)['GROUP_CLASSVAR'] \
                .transform(lambda x: x.bfill())

        # Assign 'merged_subset' data back to original DataFrame
        self.df.loc[df_valid.index, self.cols.class_var_group] = merged_subset['GROUP_CLASSVAR'].values
        self.df.loc[df_valid.index, self.cols.sf] = merged_subset['SF_MEDIAN'].values

        # Check if all open-path fluxes have a scaling factor
        # logic: Where Flux exists AND Scaling Factor is NaN
        missing_sf_mask = self.df[self.flux_openpath.name].notna() & self.df[self.cols.sf].isna()
        if missing_sf_mask.any():
            n_missing = missing_sf_mask.sum()
            # # Optional: Print the first few problematic rows for debugging
            # print("Rows with Flux but no Scaling Factor:")
            # print(df.loc[missing_sf_mask, [self.flux_openpath, self.cols.sf, self.ustar, self.cols.daytime]].head())
            raise ValueError(f"Not all open-path fluxes have a scaling factor. Found {n_missing} missing values.")

        # # 8. Gapfilling (Time-based interpolation for missing USTAR data)
        # if lut_gapfill:
        #     df[self.cols.sf_gf] = df[self.cols.sf].interpolate(method='time').bfill().ffill()
        # else:
        #     df[self.cols.sf_gf] = df[self.cols.sf]

        return self.df.sort_index()

    def stats(self):
        cols = [self.flux_openpath.name, self.flux_closedpath.name, self.cols.flux_op_corr]

        _stats_df = self.df[cols].copy()
        _stats_df = _stats_df.dropna()
        _numvals = len(_stats_df)

        print("\nCUMULATIVE FLUXES:")
        print(f"Values: {_numvals}")
        _cumsum_opnocorr = _stats_df[self.flux_openpath.name].sum()
        _cumsum_cptrueflux = _stats_df[self.flux_closedpath.name].sum()
        _perc = (_cumsum_opnocorr / _cumsum_cptrueflux) * 100
        print(f"OPEN-PATH (uncorrected): {_cumsum_opnocorr:.0f}  ({_perc:.1f}% of true flux)")
        print(f"ENCLOSED-PATH (true flux): {_cumsum_cptrueflux:.0f}")
        _cumsum = _stats_df[self.cols.flux_op_corr].sum()
        _perc = (_cumsum / _cumsum_cptrueflux) * 100
        print(f"OPEN-PATH (corrected): {_cumsum:.0f}  ({_perc:.1f}% of true flux)")
        print("\n\n")

    # def gapfilling_lut(self, series):
    #     """Gap-fill time series using look-up table (LUT)
    #
    #     Optimized version using groupby().transform()
    #     """
    #     # 1. Calculate the mean for every (Month, Hour) group
    #     # 'transform' calculates the mean and broadcasts it back to the original index size
    #     lutvals = series.groupby([series.index.month, series.index.hour]).transform('mean')
    #
    #     # 2. Fill the gaps in the original series using these means
    #     series_gf = series.fillna(lutvals)
    #
    #     return series_gf, lutvals

    def plot_series(self, ax, series, title):
        ax.plot_date(x=series.index, y=series,
                     ms=2, alpha=.3, ls='-', marker='o', markeredgecolor='none')
        ax.set_title(title, fontsize=9, fontweight='bold', y=1)

    def format_spines(self, ax, color, lw):
        spines = ['top', 'bottom', 'left', 'right']
        for spine in spines:
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(lw)
        return None

    def default_format(self, ax, fontsize=9, label_color='black',
                       txt_xlabel=False, txt_ylabel=False, txt_ylabel_units=False,
                       width=0.5, length=3, direction='in', colors='black', facecolor='white'):
        """ Apply default format to plot. """
        ax.set_facecolor(facecolor)
        ax.tick_params(axis='x', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize)
        ax.tick_params(axis='y', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize)
        self.format_spines(ax=ax, color=colors, lw=1)
        if txt_xlabel:
            ax.set_xlabel(txt_xlabel, color=label_color, fontsize=fontsize, fontweight='bold')
        if txt_ylabel and txt_ylabel_units:
            ax.set_ylabel(f'{txt_ylabel}  {txt_ylabel_units}', color=label_color, fontsize=fontsize, fontweight='bold')
        if txt_ylabel and not txt_ylabel_units:
            ax.set_ylabel(f'{txt_ylabel}', color=label_color, fontsize=fontsize, fontweight='bold')
        return None

    def plot_diel_cycles(self):
        """
        Combines flux results and auxiliary variables into a single
        publication-quality dashboard plot.

        Layout:
        Row 1: Physics/Drivers (SWIN, Ts, Ra, Rho)
        Row 2: Correction Mechanics (FCT_unsc, SF, FCT_final)
        Row 3: Flux Results (Uncorrected, Reference, Corrected, Residual)
        """
        print("Plotting Comprehensive Diel Cycles (Modern Dashboard)...")

        # --- PRE-CALCULATION ---
        # Create a temporary column for the Residual (Corrected - Reference)
        # We handle cases where reference might be missing
        diff_col_name = 'diff_corr_ref'
        plot_df = self.df.copy()

        if self.flux_closedpath.name and self.flux_closedpath.name in plot_df.columns:
            plot_df[diff_col_name] = plot_df[self.cols.flux_op_corr] - plot_df[self.flux_closedpath.name]
        else:
            plot_df[diff_col_name] = np.nan

        # --- CONFIGURATION ---
        # Define the grid layout (variable list)
        plot_vars = [
            # ROW 1: DRIVERS
            {'col': self.swin.name, 'title': '1. Shortwave Incoming (Driver)', 'unit': '$W\\ m^{-2}$'},
            {'col': self.cols.t_instr_surface, 'title': '2. Instrument Surface Temp', 'unit': '°C'},
            {'col': self.cols.aerodynamic_resistance, 'title': '3. Aerodynamic Resistance', 'unit': '$s\\ m^{-1}$'},
            {'col': self.cols.dry_air_density, 'title': '4. Dry Air Density', 'unit': '$kg\\ m^{-3}$'},

            # ROW 2: CORRECTION MECHANISM
            {'col': self.cols.fct_unsc_gf, 'title': '5. Unscaled Correction ($FCT_{unsc}$)',
             'unit': '$\\mu mol\\ m^{-2}\\ s^{-1}$'},
            {'col': "SF", 'title': '6. Scaling Factor ($\\xi$)', 'unit': '-'},
            {'col': self.cols.fct, 'title': '7. Final Correction Term', 'unit': '$\\mu mol\\ m^{-2}\\ s^{-1}$'},
            {'col': None},  # Spacer to align grid if needed, or we can simply skip

            # ROW 3: FLUX RESULTS
            {'col': self.flux_openpath.name, 'title': '8. OP Flux (Uncorrected)', 'unit': '$\\mu mol\\ m^{-2}\\ s^{-1}$'},
            {'col': self.flux_closedpath.name, 'title': '9. CP Flux (Reference)', 'unit': '$\\mu mol\\ m^{-2}\\ s^{-1}$'},
            {'col': self.cols.flux_op_corr, 'title': '10. OP Flux (Corrected)', 'unit': '$\\mu mol\\ m^{-2}\\ s^{-1}$'},
            {'col': diff_col_name, 'title': '11. Residual (Corrected - Ref)', 'unit': '$\\mu mol\\ m^{-2}\\ s^{-1}$'},
        ]

        # --- FIGURE SETUP ---
        # 3 Rows, 4 Columns
        rows, cols = 3, 4
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12), constrained_layout=True, sharex=True)

        # Adjust padding for readability
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, wspace=0.05, hspace=0.05)

        axes_flat = axes.flatten()

        # Colormap
        # Viridis with 12 discrete steps for months
        cmap = plt.get_cmap('viridis', 12)
        norm = plt.Normalize(vmin=0.5, vmax=12.5)

        for i, var in enumerate(plot_vars):
            ax = axes_flat[i]

            # Handle spacer or missing columns
            if var['col'] is None or var['col'] not in plot_df.columns:
                ax.axis('off')
                continue

            series = plot_df[var['col']].copy()
            series.index = pd.to_datetime(series.index)

            # Helper call
            self._plot_diel_cycle_modern(ax, series, cmap=cmap, norm=norm)

            # Styling
            ax.set_title(var['title'], fontsize=11, fontweight='bold', loc='left', color='#333333')
            ax.set_ylabel(var['unit'], fontsize=9, color='#555555')
            ax.grid(True, linestyle=':', alpha=0.6, color='gray')

            # Zero line for flux-related variables
            if (series.min() < 0) and (series.max() > 0):
                ax.axhline(0, lw=1, color='#333333', linestyle='-', zorder=1)

            # Despine
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

        # Shared colorbar on the right
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', aspect=40, shrink=0.6, label='Month')
        cbar.set_ticks(np.arange(1, 13))
        cbar.set_ticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        cbar.outline.set_visible(False)

        # Global X Label
        fig.supxlabel("Hour of Day (Local Winter Time)", fontsize=13, fontweight='normal')

        fig.show()

    def _plot_diel_cycle_modern(self, ax, series, cmap, norm):
        """Helper to plot hourly means by month."""
        diel_df = series.groupby([series.index.month, series.index.hour]).mean().unstack()
        hours = diel_df.columns
        for month in diel_df.index:
            vals = diel_df.loc[month]
            color = cmap(norm(month))
            ax.plot(hours, vals, color=color, alpha=0.8, linewidth=1.5)

        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 24])

    def plot_flux_analysis_dashboard(self):
        print("Plotting Flux Analysis Dashboard (Time Series + Cumulative)...")

        # --- CONFIGURATION ---
        # Consistent styling across all panels
        # zorder: Corrected on top (3), Reference middle (2), Uncorrected bottom (1)
        style_config = {
            self.flux_openpath.name: {
                'label': 'Uncorrected (OP)', 'color': '#95a5a6', 'zorder': 1, 'alpha_pts': 0.1, 'alpha_line': 0.7
            },
            self.flux_closedpath.name: {
                'label': 'Reference (CP)', 'color': '#2c3e50', 'zorder': 2, 'alpha_pts': 0.1, 'alpha_line': 0.8
            },
            self.cols.flux_op_corr: {
                'label': 'Corrected (OP)', 'color': '#e74c3c', 'zorder': 3, 'alpha_pts': 0.15, 'alpha_line': 1.0
            }
        }

        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 0.8])  # Top row slightly taller

        # Define axes
        ax_ts = fig.add_subplot(gs[0, :])  # Top row: Time Series (Spans all columns)
        ax_day = fig.add_subplot(gs[1, 0])  # Bottom Left: Daytime Cumulative
        ax_night = fig.add_subplot(gs[1, 1])  # Bottom Mid: Nighttime Cumulative
        ax_all = fig.add_subplot(gs[1, 2])  # Bottom Right: All Cumulative

        # Adjust padding
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, wspace=0.1, hspace=0.1)

        # Top row: time series
        cols_to_plot = [self.flux_openpath.name, self.flux_closedpath.name, self.cols.flux_op_corr]

        for col in cols_to_plot:
            if col not in self.df.columns or col is None:
                continue

            series = self.df[col].dropna()
            if series.empty:
                continue

            style = style_config[col]

            # Raw data scatter
            ax_ts.scatter(series.index, series, s=2, color=style['color'],
                          alpha=style['alpha_pts'], edgecolors='none',
                          zorder=style['zorder'])

            # Trend line (3-day rolling mean)
            rolling = series.rolling(window=144, center=True, min_periods=1).mean()
            ax_ts.plot(rolling.index, rolling, color=style['color'], lw=2,
                       alpha=style['alpha_line'], zorder=style['zorder'] + 10,
                       label=f"{style['label']}")

        # Styling Top Row
        ax_ts.set_title("A. Flux Time Series & Trends", fontsize=12, fontweight='bold', loc='left', color='#333333')
        ax_ts.set_ylabel(r"CO$_2$ Flux ($\mu mol\ m^{-2}\ s^{-1}$)", fontsize=10, color='#555555')
        ax_ts.axhline(0, lw=1.5, color='#333333', linestyle='-', zorder=5)
        ax_ts.legend(loc='upper left', frameon=False, fontsize=10, ncol=3)
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        # Bottom row: cumulative fluxes
        # Prepare scenarios
        scenarios = [
            {'ax': ax_day, 'title': 'B. Daytime Budget', 'filter': self.df[self.cols.daytime] == 1},
            {'ax': ax_night, 'title': 'C. Nighttime Budget', 'filter': self.df[self.cols.daytime] == 0},
            {'ax': ax_all, 'title': 'D. Total Budget', 'filter': slice(None)}  # No filter
        ]

        for scen in scenarios:
            ax = scen['ax']

            # Filter data first
            if isinstance(scen['filter'], slice):
                subset = self.df.copy()
            else:
                subset = self.df.loc[scen['filter']].copy()

            # Drop rows where any of the key fluxes are missing to ensure fair cumulative comparison
            # (If one sensor is down, we shouldn't sum the other one during that time)
            subset = subset.dropna(subset=cols_to_plot, how='any')

            # Calculate Cumulative Sum
            for col in cols_to_plot:
                if col not in subset.columns or col is None: continue
                style = style_config[col]

                cumsum_series = subset[col].cumsum()

                # Plot Line
                ax.plot(subset.index, cumsum_series, color=style['color'],
                        lw=2, zorder=style['zorder'], label=style['label'])

                # Add final value annotation at the end of the line
                if not cumsum_series.empty:
                    final_val = cumsum_series.iloc[-1]
                    ax.text(subset.index[-1], final_val, f" {final_val:.0f}",
                            verticalalignment='center', fontsize=8, color=style['color'], fontweight='bold')

            # Styling Bottom Row
            ax.set_title(scen['title'], fontsize=11, fontweight='bold', loc='left', color='#333333')
            if scen['ax'] == ax_day:
                ax.set_ylabel(r"Cumulative Sum ($\Sigma$)", fontsize=10, color='#555555')

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Short month name

            # Simplified legend only on the last plot to reduce clutter
            if scen['ax'] == ax_all:
                ax.legend(loc='best', frameon=False, fontsize=8)

        # Global styling
        for ax in [ax_ts, ax_day, ax_night, ax_all]:
            ax.grid(True, which='major', linestyle=':', alpha=0.6, color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

        fig.show()

    def plot_cumulative(self, ax: plt.axis, title: str, which: str, df: pd.DataFrame,
                        daytime_col: str):
        df = df.dropna()

        if which == 'daytime':
            df = df.loc[df[daytime_col] == 1]
        elif which == 'nighttime':
            df = df.loc[df[daytime_col] == 0]
        elif which == 'all':
            pass

        # # Convert to gC m-2
        # _subset[self.op_co2_flux_QC01_nocorr_col] = \
        #     _subset[self.op_co2_flux_QC01_nocorr_col].multiply(0.02161926)
        # _subset[self.cp_co2_flux_QC01_col] = \
        #     _subset[self.cp_co2_flux_QC01_col].multiply(0.02161926)
        # _subset[self.op_co2_flux_corr_jar09_col] = \
        #     _subset[self.op_co2_flux_corr_jar09_col].multiply(0.02161926)

        df = df.cumsum()
        for col in df.columns:
            if col == daytime_col:
                continue
            ax.plot_date(x=df.index, y=df[col], label=col, ms=3, alpha=.5)
        ax.set_title(title, fontsize=9, fontweight='bold', y=1)
        ax.legend()


class ScopPhysics:

    def __init__(self, ta: pd.Series, qc: pd.Series, rho_a: pd.Series, rho_v: pd.Series,
                 u: pd.Series, c_p: pd.Series, ustar: pd.Series, swin: pd.Series,
                 lat: float, lon: float, utc_offset: int,
                 remove_outliers_method: Literal["fast", "separate"] = "fast"):
        """
        Args:
            ta: series, air temperature (°C)
            qc: series, CO2 molar density (µmol m-3)
            rho_a: series, air density (kg m-3)
            rho_v: series, water vapor density (kg m-3)          
            u: series, wind speed (m s-1)            
            c_p: series, air heat capacity, specific heat at constant pressure of ambient air (J K-1 kg-1)
            ustar: series, friction velocity (m s-1)
            swin: series, shortwave incoming radiation (W m-2)
            lat: float, latitude of the site
            lon: float, longitude of the site
            utc_offset: int, UTC offset of the timestamp (hours)
            remove_outliers_method: str, method to remove outliers from the data
                Used for removing outliers in 'ra' and 'fct_unsc'.
        """

        self.ta = ta
        self.qc = qc
        self.rho_a = rho_a
        self.rho_v = rho_v
        self.u = u
        self.c_p = c_p
        self.ustar = ustar
        self.swin = swin
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset
        self.remove_outliers_method = remove_outliers_method

        self.cols = ColumnConfig()

        # Detect daytime/nighttime (daytime=1, nighttime=0)
        self.daytime = self._detect_daytime(swin=self.swin)

        # Calculate aerodynamic resistance (ra) (s m-1)
        self.ra = self._calc_aerodynamic_resistance(u=self.u, ustar=self.ustar, rem_outliers=True)

        # Calculate dry air density (rho_d) (kg m-3)
        self.rho_d = dry_air_density(rho_v=rho_v, rho_a=rho_a)

        # Calculate thermal conductivity of air (required for BUR08) (k_air) (W m-1 K-1)
        self.k_air = self._calc_air_thermal_conductivity(ta=self.ta)

        # Calculated in .run()
        self.ts = pd.Series(name=self.cols.t_instr_surface)  # Bulk instrument surface temperature (BUR06, JAR09)
        self.S = pd.Series(name=self.cols.s)  # Sensible heat from all key instrument surfaces  (BUR08)
        self.fct_unsc = pd.Series(name=self.cols.fct_unsc)  # Unscaled flux correction term, produced by all methods
        self.fct_unsc_gf = pd.Series(name=self.cols.fct_unsc_gf)  # Unscaled flux correction term, gap-filled

    def get_results(self) -> pd.DataFrame:
        frame = {
            self.cols.fct_unsc_gf: self.fct_unsc_gf,
            self.cols.fct_unsc: self.fct_unsc,
            self.cols.s: self.S,
            self.cols.t_instr_surface: self.ts,
            self.ta.name: self.ta,
            self.c_p.name: self.c_p,
            self.ra.name: self.ra,
            self.rho_a.name: self.rho_a,
            self.rho_d.name: self.rho_d,
            self.rho_v.name: self.rho_v,
            self.u.name: self.u,
            self.ustar.name: self.ustar,
            self.qc.name: self.qc,
            self.swin.name: self.swin,
            self.k_air.name: self.k_air,
            self.daytime.name: self.daytime
        }
        return pd.DataFrame.from_dict(frame)

    def run(self, correction_method_base: Literal["JAR09", "BUR06", "BUR08"] = "JAR09",
            gapfill: bool = True):
        if correction_method_base == "BUR06":
            self.ts = self._estimate_surface_temp_bur06()
            self.fct_unsc = self._flux_correction_term_unscaled_jar09_bur06(ts=self.ts)
        elif correction_method_base == "JAR09":
            self.ts = self._estimate_surface_temp_jar09()
            self.fct_unsc = self._flux_correction_term_unscaled_jar09_bur06(ts=self.ts)
        elif correction_method_base == "BUR08":
            self.fct_unsc, self.S = self._flux_correction_term_unscaled_bur08()

        self.fct_unsc.name = self.cols.fct_unsc

        # Remove outliers from unscaled flux correction term
        if self.remove_outliers_method == "fast":
            self.fct_unsc = self._remove_outliers_fast(series=self.fct_unsc)
        elif self.remove_outliers_method == "separate":
            self.fct_unsc = self._remove_outliers(series=self.fct_unsc)
        else:
            raise ValueError(f"Unknown remove_outliers_method: {self.remove_outliers_method}")

        if gapfill:
            self.fct_unsc_gf = self._gapfill()

    def _gapfill(self):
        # Gap-fill unscaled flux correction term
        frame = {
            self.cols.fct_unsc: self.fct_unsc,
            self.ta.name: self.ta,
            self.c_p.name: self.c_p,
            self.rho_a.name: self.rho_a,
            self.u.name: self.u,
            self.ustar.name: self.ustar
        }
        gfdf = pd.DataFrame.from_dict(frame)

        gfdf = gfdf.dropna()
        rfts = RandomForestTS(
            input_df=gfdf,
            target_col=self.cols.fct_unsc,
            verbose=True,
            # features_lag=None,
            features_lag=[-1, -1],
            # features_lag_exclude_cols=['test', 'test2'],
            vectorize_timestamps=True,
            add_continuous_record_number=True,
            sanitize_timestamp=True,
            perm_n_repeats=1,
            n_estimators=3,
            random_state=42,
            # random_state=None,
            max_depth=None,
            min_samples_split=20,
            min_samples_leaf=10,
            criterion='squared_error',
            # test_size=0.2,
            n_jobs=-1
        )
        rfts.trainmodel(showplot_scores=False, showplot_importance=False)
        rfts.report_traintest()
        rfts.fillgaps(showplot_scores=False, showplot_importance=False)
        rfts.report_gapfilling()
        print(rfts.feature_importances_)
        print(rfts.scores_)
        print(rfts.gapfilling_df_)
        fct_unsc_gf = rfts.gapfilling_df_[self.cols.fct_unsc_gf].copy()
        return fct_unsc_gf

    def _detect_daytime(self, swin) -> pd.Series:
        # Daytime, nighttime
        # daytime_series: series (daytime=1, nighttime=0)
        nighttime_filter = swin <= 20
        daytime_filter = swin > 20
        daytime_series = pd.Series(index=swin.index)
        daytime_series.loc[nighttime_filter] = 0
        daytime_series.loc[daytime_filter] = 1
        daytime_series.name = self.cols.daytime
        return daytime_series

    @staticmethod
    def _calc_air_thermal_conductivity(ta: pd.Series) -> pd.Series:
        """
        Calculates the thermal conductivity of air using a linear approximation.

        This method uses a linear approximation to estimate the thermal conductivity of air 
        for temperatures ranging from -50°C to 100°C. The calculation is based on the 
        temperature-dependent relationship where the thermal conductivity at 0°C is approximately 
        0.02425 W/m·K and increases by approximately 0.00007 W/m·K per degree Celsius.

        Args:
            ta (pd.Series): A pandas Series representing the air temperature in degrees Celsius.

        Returns:
            pd.Series: A pandas Series containing the calculated thermal conductivity of air (W m-1 K-1)
                corresponding to the input temperatures. 
        """
        # Linear approximation suitable for atmospheric range (-50 to 100 C)
        # k ~ 0.02425 at 0C, increasing by ~0.00007 per degree C
        k_air = 0.02425 + (0.00007 * ta)
        k_air.name = "AIR_THERMAL_CONDUCTIVITY"
        return k_air

    def _remove_outliers(self, series: pd.Series):
        ham = HampelDaytimeNighttime(
            series=series,
            n_sigma_dt=4,
            n_sigma_nt=4,
            window_length=48 * 5,
            showplot=True,
            verbose=True,
            lat=self.lat,
            lon=self.lon,
            utc_offset=self.utc_offset
        )
        ham.calc(repeat=False)
        return ham.filteredseries

    def _calc_aerodynamic_resistance(self, u, ustar, rem_outliers: bool = False) -> pd.Series:
        # Aerodynamic resistance
        # ra: series, aerodynamic resistance (s m-1)
        ra = aerodynamic_resistance(u_ms=u, ustar_ms=ustar)
        if rem_outliers:
            if self.remove_outliers_method == "fast":
                ra = self._remove_outliers_fast(series=ra)
            elif self.remove_outliers_method == "separate":
                ra = self._remove_outliers(series=ra)
            else:
                raise ValueError(f"Unknown remove_outliers_method: {self.remove_outliers_method}")
        return ra

    @staticmethod
    def _remove_outliers_fast(series, plot_title: str = "X", n_sigmas: int = 5):
        """Remove outliers with Hampel filter, using running MAD (median absolute deviation)

        :param series: pandas.Series, data from which outliers are removed
        :param plot_title: Plot title
        :param n_sigmas:
        :return:
        pandas.Series with outliers removed
        """
        # n_sigmas = 4  # Number of sigmas for limits
        k = 1.4826  # Scale factor for Gaussian distribution
        window = 1440  # Rolling time window in number of records
        min_periods = 1  # Min number of records in window

        _series_rolling = series.rolling(window=window, min_periods=min_periods, center=True)
        _series_running_median = _series_rolling.median()
        _series_sub_winmed = series.sub(_series_running_median).abs()  # Data series minus median in time window
        _series_running_mad = _series_sub_winmed.rolling(window=window, min_periods=min_periods,
                                                         center=True).median().multiply(k)
        _diff_series_runmed = np.abs(series - _series_running_median)
        _diff_limit = _series_running_mad.multiply(n_sigmas)  # Limit

        _outliers_ix = _diff_series_runmed > _diff_limit
        series.loc[_outliers_ix] = np.nan  # Remove outliers from series (set to missing)
        return series

        # # Plot
        # figsize = (14, 5)
        # plt.figure()
        # series_orig.plot(figsize=figsize, title=f"{plot_title} BEFORE outlier removal");
        # plt.figure()
        # series.plot(title=f"{plot_title} AFTER outlier removal", figsize=figsize, label="series after outlier removal");
        # _diff_series_runmed.loc[~_outliers_ix].plot(
        #     label="absolute difference of: series - running median of series; for outlier detection")
        # _diff_limit.plot(label="limit from: series running MAD * number of sigmas")
        # _series_running_mad.plot(label="series running MAD (median absolute deviation)")
        # _series_running_median.plot(label="series running median")
        # plt.legend()
        # print(series.describe())

        # return series, series_orig

    def _flux_correction_term_unscaled_bur08(self) -> tuple[pd.Series, pd.Series]:
        """Calculate bulk instrument surface temperature (BUR08)"""

        # TOP OF WINDOW ---
        # Surface temperatures day/night
        # Calculate and then combine day and night temperatures
        _ts_bur08_day_top = 1.005 * self.ta + 0.24
        _ts_bur08_night_top = 1.008 * self.ta - 0.41
        ts_top = pd.Series(index=_ts_bur08_day_top.index)
        ts_top.loc[self.daytime == 1] = _ts_bur08_day_top  # Use daytime Ts in daytime data rows
        ts_top.loc[self.daytime == 0] = _ts_bur08_night_top  # Use nighttime Ts in nighttime data rows
        # Calculate sigma below top window
        l_top = 0.045  # Diameter of the detector housing (m)
        sigma_top = 0.0028 * np.sqrt(l_top / self.u) + (0.00025 / self.u) + 0.0045
        # Calculate top window sensible heat
        r_top = 0.0225  # Radius of the detector sphere (m)
        a = (r_top + sigma_top) * (ts_top - self.ta)
        b = r_top * sigma_top
        S_top = self.k_air * (a / b)

        # BOTTOM WINDOW ---
        # Surface temperatures day/night
        # Calculate and then combine day and night temperatures
        _ts_bur08_day_bottom = 0.944 * self.ta + 2.57
        _ts_bur08_night_bottom = 0.883 * self.ta + 2.17
        ts_bottom = pd.Series(index=_ts_bur08_day_bottom.index)
        ts_bottom.loc[self.daytime == 1] = _ts_bur08_day_bottom  # Use daytime Ts in daytime data rows
        ts_bottom.loc[self.daytime == 0] = _ts_bur08_night_bottom  # Use nighttime Ts in nighttime data rows
        # Calculate sigma above bottom window
        l_bottom = 0.065  # Diameter of the source housing (m)
        sigma_bottom = 0.004 * np.sqrt(l_bottom / self.u) + 0.004
        # Calculate bottom window sensible heat
        S_bottom = self.k_air * ((ts_bottom - self.ta) / sigma_bottom)

        # SPAR ---
        # Surface temperatures day/night
        # Calculate and then combine day and night temperatures
        _ts_bur08_day_spar = 1.01 * self.ta + 0.36
        _ts_bur08_night_spar = 1.01 * self.ta - 0.17
        ts_spar = pd.Series(index=_ts_bur08_day_spar.index)
        ts_spar.loc[self.daytime == 1] = _ts_bur08_day_spar  # Use daytime Ts in daytime data rows
        ts_spar.loc[self.daytime == 0] = _ts_bur08_night_spar  # Use nighttime Ts in nighttime data rows
        # Calculate sigma around spar
        l_spar = 0.005  # Diameter of the spar (m)
        sigma_spar = 0.0058 * np.sqrt(l_spar / self.u)
        # Calculate spar sensible heat
        r_spar = 0.0025  # Radius of the spar cylinder (m)
        a = ts_spar - self.ta
        b = r_spar * np.log((r_spar + sigma_spar) / r_spar)
        S_spar = self.k_air * (a / b)

        # Calculate sensible heat from all key instrument surfaces
        S = S_bottom + S_top + 0.15 * S_spar  # W m-2

        # Calculate unscaled flux correction term
        fct_unsc = (S / (self.rho_a * self.c_p)) * (self.qc / (self.ta + 273.15))

        # print(f"Ts (BUR08), mean = {fct_unsc.mean():.2f}")
        return fct_unsc, S

    def _flux_correction_term_unscaled_jar09_bur06(self, ts: pd.Series) -> pd.Series:
        """
        Calculate unscaled flux correction term.
        Equation (8) in Burba et al. (2006).
        Used by JAR09 and BUR06.

        Args:
            ts: series, instrument surface temperature (°C)
        """
        # Convert temperatures to Kelvin for the denominator
        ta_k = self.ta + 273.15

        # Term A: Surface - Air temp delta * CO2 density
        term_a = (ts - self.ta) * self.qc

        # Term B: Aerodynamic resistance * Temp
        term_b = self.ra * ta_k

        # Term C: Water vapor correction factor
        term_c = 1 + 1.6077 * (self.rho_v / self.rho_d)

        return (term_a / term_b) * term_c

    def _estimate_surface_temp_bur06(self) -> pd.Series:
        """
        Estimates the surface temperature based on the provided air temperature
        values using the BUR06 model. The BUR06 model applies a quadratic
        relationship to approximate the surface temperature from air temperature.

        Returns:
            pd.Series: Estimated surface temperature values calculated using the
            BUR06 model.
        """
        ts = 0.0025 * self.ta ** 2 + 0.9 * self.ta + 2.07
        return ts

    def _estimate_surface_temp_jar09(self) -> pd.Series:
        """
        Estimates surface temperature using the JAR09 method.

        This method calculates surface temperature based on ambient temperature and
        a binary series representing daytime or nighttime conditions. It applies
        different linear relationships for daytime and nighttime data.

        Returns:
            pd.Series: A Pandas Series representing the estimated surface temperatures.
        """
        ts = pd.Series(index=self.ta.index, dtype=float)

        # Daytime relation
        ts.loc[self.daytime == 1] = 0.93 * self.ta.loc[self.daytime == 1] + 3.17
        # Nighttime relation
        ts.loc[self.daytime == 0] = 1.05 * self.ta.loc[self.daytime == 0] + 1.52

        return ts


class ScopOptimizer:

    def __init__(self, class_var: pd.Series, n_classes: int, fct_unsc: pd.Series, daytime: pd.Series,
                 n_bootstrap_runs: int, flux_openpath: pd.Series, flux_closedpath: pd.Series, showplot: bool = True):
        self.class_var = class_var
        self.n_classes = n_classes
        self.n_bootstrap_runs = n_bootstrap_runs
        self.flux_openpath = flux_openpath
        self.flux_closedpath = flux_closedpath
        self.fct_unsc = fct_unsc
        self.daytime = daytime
        self.showplot = showplot

        self.cols = ColumnConfig()

        # If data are not bootstrapped, set flag to False
        if self.n_bootstrap_runs == 0:
            self.bootstrapped = False
            self.n_bootstrap_runs = 9999  # Set to arbitrary number
        else:
            self.bootstrapped = True

        self.scaling_factors_df = self.init_scaling_factors_df(num_classes=n_classes)

        frame = {
            self.class_var.name: self.class_var,
            self.flux_openpath.name: self.flux_openpath,
            self.flux_closedpath.name: self.flux_closedpath,
            self.fct_unsc.name: self.fct_unsc,
            self.daytime.name: self.daytime
        }
        self.df = pd.DataFrame.from_dict(frame)

    def run(self):
        self._bootstrapping()
        self._plot()
        self._print_stats()

    def _plot(self):
        """
        Plots the optimized scaling factors with bootstrapped confidence intervals.
        """
        if not self.showplot:
            return

        print("Plotting Scaling Factors (Modern Style)...")

        # Create a copy to ensure sorting works for line plots
        plot_df = self.scaling_factors_df.copy()

        # --- CONFIGURATION ---
        # Scientific color palette: High contrast, warm vs cool
        # 1 = Day (Warm Orange), 0 = Night (Deep Blue)
        styles = {
            1: {'color': '#d35400', 'label': 'Daytime', 'marker': 'o'},
            0: {'color': '#2980b9', 'label': 'Nighttime', 'marker': 's'}
        }

        # --- FIGURE SETUP ---
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

        # --- PLOTTING LOOP ---
        # Group by DAYTIME (0 or 1)
        for day_flag, group in plot_df.groupby('DAYTIME'):
            # Sort by the class variable (x-axis) to ensure lines connect correctly
            group = group.sort_values('GROUP_CLASSVAR_MIN')

            x = group['GROUP_CLASSVAR_MIN']
            y = group['SF_MEDIAN']
            style = styles.get(day_flag, {'color': 'black', 'label': 'Unknown', 'marker': 'x'})

            # Calculate average sample size for the legend
            avg_n = group['NUMVALS_AVG'].mean()
            label_text = f"{style['label']} (N $\\approx$ {avg_n:.0f})"

            # 1. Outer Confidence Interval (1% - 99%)
            # Very transparent, shows extreme range
            ax.fill_between(x, group['SF_Q01'], group['SF_Q99'],
                            color=style['color'], alpha=0.1, edgecolor='none', zorder=1)

            # 2. Inner Confidence Interval (25% - 75% / IQR)
            # More visible, shows typical variation
            ax.fill_between(x, group['SF_Q25'], group['SF_Q75'],
                            color=style['color'], alpha=0.25, edgecolor='none', zorder=2)

            # 3. Median Line
            # Solid, distinct line
            ax.plot(x, y, color=style['color'], marker=style['marker'],
                    markersize=5, linewidth=2, zorder=3, label=label_text)

        # --- STYLING ---
        # Clean formatting for the Class Variable name
        xlabel = str(self.class_var.name).replace('_', ' ').title()

        ax.set_title(f"Scaling Factors vs. {xlabel}", fontsize=14, fontweight='bold', loc='left', color='#333333')
        ax.set_xlabel(f"Class Variable: {xlabel} (units)", fontsize=11, color='#333333')
        ax.set_ylabel(r"Scaling Factor ($\xi$)", fontsize=11, color='#333333')

        # Reference line at 0
        ax.axhline(0, lw=1, color='#333333', linestyle='-', zorder=0)

        # Modern Grid
        ax.grid(True, which='major', linestyle=':', alpha=0.6, color='gray')

        # Despine (Remove top and right borders)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Legend
        ax.legend(loc='best', frameon=False, fontsize=10)

        fig.show()

    def _print_stats(self):
        print("BOOTSTRAPPING RESULTS")
        print(f"class variable: {self.class_var}")
        print(f"number of classes: {self.n_classes}")

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

    def get(self) -> pd.DataFrame:
        return self.scaling_factors_df

    def _bootstrapping(self):

        # Group data records by daytime / nighttime membership
        _grouped_by_daynighttime = self.df.groupby(self.cols.daytime)

        # Loop through daytime / nighttime data
        for _group_daynighttime, _group_daynighttime_df in _grouped_by_daynighttime:

            # Divide data into x class variable groups w/ same number of values
            _group_daynighttime_df[self.cols.class_var_group] = \
                pd.qcut(_group_daynighttime_df[self.class_var.name],
                        q=self.n_classes, labels=False)

            # Group data records by class variable membership
            _grouped_by_class_var = _group_daynighttime_df.groupby(self.cols.class_var_group)

            # Loop through class variable groups
            for _group_class_var, _group_class_var_df in _grouped_by_class_var:
                # Bootstrap group data

                bts_factors = []
                bts_sum_of_squares = []
                bts_num_vals = []

                for bts_run in range(0, self.n_bootstrap_runs):
                    num_rows = int(len(_group_class_var_df.index))

                    if self.bootstrapped:
                        # Use bootstrapped data
                        bts_sample_df = _group_class_var_df.sample(n=num_rows,
                                                                   replace=True)  # Take sample w/ replacement
                    else:
                        # Use measured data in cas
                        bts_sample_df = _group_class_var_df

                    bts_sample_df.sort_index(inplace=True)

                    result = self.optimize_factor(target=bts_sample_df[self.flux_openpath.name],
                                                  reference=bts_sample_df[self.flux_closedpath.name],
                                                  fct_unsc_gf=bts_sample_df[self.cols.fct_unsc_gf])

                    bts_factors.append(result.x)  # x = scaling factor
                    bts_sum_of_squares.append(result.fun)
                    bts_num_vals.append(bts_sample_df[self.class_var.name].count())

                    # Break if only working with measured data (no bootstrapping)
                    if not self.bootstrapped:
                        break

                print(f"Finished {self.n_bootstrap_runs} bootstrap runs for group {_group_class_var} "
                      f"in daytime = {_group_daynighttime}")

                # Stats, aggregates for current class group
                location = tuple([_group_daynighttime, _group_class_var])
                self.scaling_factors_df.loc[location, f'DAYTIME'] = _group_daynighttime
                self.scaling_factors_df.loc[location, f'GROUP_CLASSVAR'] = _group_class_var
                self.scaling_factors_df.loc[location, f'GROUP_CLASSVAR_MIN'] = _group_class_var_df[
                    self.class_var.name].min()
                self.scaling_factors_df.loc[location, f'GROUP_CLASSVAR_MAX'] = _group_class_var_df[
                    self.class_var.name].max()
                self.scaling_factors_df.loc[location, f'BOOTSTRAP_RUNS'] = self.n_bootstrap_runs

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
                                 method='Bounded', bounds=[-1, 200])
        return result

    def calc_sumofsquares(self, factor, unsc_flux_corr_term, target, reference):
        corrected = target + unsc_flux_corr_term.multiply(factor)

        # diff2 = np.sqrt((corrected - reference) ** 2)
        # sum_of_squares = diff2.sum()

        corrected_cumsum = corrected.cumsum()
        reference_cumsum = reference.cumsum()
        diff2 = np.sqrt((corrected_cumsum - reference_cumsum) ** 2)
        sum_of_squares = diff2.sum()

        return sum_of_squares

    def init_scaling_factors_df(self, num_classes):
        """Initialize df that collects results for scaling factors
        - Needs to be initialized with a Multiindex
        - Multiindex consists of two indices: (1) daytime and (2) sonic temperature class
        """
        _list_class_var_classes = [*range(0, num_classes)]
        _iterables = [[1, 0], _list_class_var_classes]
        _multi_ix = pd.MultiIndex.from_product(_iterables, names=["daytime_ix", "sonic_temperature_class_ix"])
        scaling_factors_df = pd.DataFrame(index=_multi_ix)

        cols = ['SF_MEDIAN', 'SOS_MEDIAN', 'NUMVALS_AVG', 'SF_Q25',
                'SF_Q75', 'SF_Q01', 'SF_Q99']

        for col in cols:
            scaling_factors_df[col] = np.nan

        return scaling_factors_df


def main():
    from diive.core.io.files import load_parquet
    df = load_parquet(
        filepath=r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\30_FLUX_PROCESSING_CHAIN\32_SELF-HEATING_CORRECTION\21_MERGED_IRGA75-noSHC+IRGA72_FluxProcessingChain_after-L3.2_NEE_QCF10_2016-2017.parquet")
    # [print(c) for c in df.columns if "AIR" in c];

    # Parallel measurements starting 27 May 2016
    df = df.loc["2016-05-27 00:15:00":"2017-12-11 23:45:00"].copy()

    # Variables from EddyPro _fluxnet_ output file
    AIR_CP = "AIR_CP_IRGA72"  # air_heat_capacity (J kg-1 K-1)
    AIR_DENSITY = "AIR_DENSITY_IRGA72"  # air_density (kg m-3)
    VAPOR_DENSITY = "VAPOR_DENSITY_IRGA72"  # water_vapor_density (kg m-3)
    U = "U_IRGA72"  # (m s-1)
    USTAR = "USTAR_IRGA72"  # (m s-1)
    TA = "TA_T1_47_1_gfXG_IRGA72"  # (degC)
    SWIN = "SW_IN_T1_47_1_gfXG_IRGA72"  # (W m-2)
    CO2_MOLAR_DENSITY = "CO2_MOLAR_DENSITY_IRGA72"  # (mol mol-1)
    FLUX_72 = "NEE_L3.1_L3.2_QCF_IRGA72"  # (umol m-2 s-1)
    FLUX_75 = "NEE_L3.1_L3.2_QCF_IRGA75"  # (umol m-2 s-1)

    # Conversions
    df[CO2_MOLAR_DENSITY] = df[CO2_MOLAR_DENSITY] * 1000  # Convert to umol mol-1

    tic = time.time()

    # Calculate
    physics = ScopPhysics(
        ta=df[TA].copy(),
        qc=df[CO2_MOLAR_DENSITY].copy(),
        rho_a=df[AIR_DENSITY].copy(),
        rho_v=df[VAPOR_DENSITY].copy(),
        u=df[U].copy(),
        c_p=df[AIR_CP].copy(),
        ustar=df[USTAR].copy(),
        swin=df[SWIN].copy(),
        lat=47.478333,  # CH–LAE
        lon=8.364389,  # CH–LAE
        utc_offset=1,
    )
    physics.run(correction_method_base="JAR09", gapfill=True)
    # physics.run(correction_method_base="JAR09", gapfill=True)
    results_physics_df = physics.get_results()

    optimizer = ScopOptimizer(
        fct_unsc=results_physics_df["FCT_UNSC_gfRF"],
        class_var=df[USTAR].copy(),
        n_classes=3,
        n_bootstrap_runs=0,
        flux_openpath=df[FLUX_75].copy(),
        flux_closedpath=df[FLUX_72].copy(),
        daytime=results_physics_df["DAYTIME"],
        showplot=True
    )
    optimizer.run()
    scaling_factors_df = optimizer.get()

    applicator = ScopApplicator(
        fct_unsc=results_physics_df["FCT_UNSC_gfRF"],
        scaling_factors_df=scaling_factors_df,
        flux_openpath=df[FLUX_75].copy(),
        flux_closedpath=df[FLUX_72].copy(),
        classvar=df[USTAR].copy(),
        daytime=results_physics_df["DAYTIME"].copy(),
        swin=df[SWIN].copy()
    )
    applicator.run()

    # scop = ScopOptimizer(
    #     inputdf=df,
    #     site="CH-LAE",
    #     lat=47.478333,  # CH–LAE
    #     lon=8.364389,  # CH–LAE
    #     utc_offset=1,
    #     title="CH-LAE self-heating correction",
    #     flux_openpath=FLUX_75,
    #     flux_closedpath=FLUX_72,
    #     air_heat_capacity=AIR_CP,
    #     co2_molar_density=CO2_MOLAR_DENSITY,  # in umol mol-1
    #     u=U,
    #     ustar=USTAR,
    #     water_vapor_density=VAPOR_DENSITY,
    #     air_density=AIR_DENSITY,
    #     air_temperature=TA,
    #     swin=SWIN,
    #     n_classes=5,
    #     n_bootstrap_runs=0,
    #     classvar=USTAR,
    #     remove_outliers_method="fast",
    #     # remove_outliers_method="separate"
    #     correction_method_base="BUR08"
    #     # correction_method_base="BUR06"
    #     # correction_method_base="JAR09"
    # )
    # scop.calc_sf()
    # apply_scaling_factors = Scop().apply_sf()

    toc = time.time()
    print(f"Time elapsed: {toc - tic:.2f} seconds")

    print("End.")


if __name__ == '__main__':
    main()
