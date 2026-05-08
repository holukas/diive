import datetime as dt
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from uncertainties import ufloat

import diive.core.plotting.styles.LightTheme as theme
from diive.core.dfun.frames import df_between_two_dates
from diive.core.plotting.plotfuncs import default_format, default_legend, nice_date_ticks
from diive.core.plotting.scatter import ScatterXY


# todo
# class JointUncertaintyPAS20:
#
#     def __init__(self, df: DataFrame, randunccol: str):
#         self.df = df
#         self.randunccol = randunccol
#
#         self.df['JOINTUNC'] = np.nan


class RandomUncertaintyPAS20:
    """

    References:
        Hollinger, D. Y., & Richardson, A. D. (2005). Uncertainty in eddy covariance measurements
            and its application to physiological models. Tree Physiology, 25(7), 873–885.
            https://doi.org/10.1093/treephys/25.7.873
        Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux processing
            pipeline for eddy covariance data. 27. https://doi.org/10.1038/s41597-020-0534-3

        https://cran.r-project.org/web/packages/REddyProc/vignettes/aggUncertainty.html
        https://pythonhosted.org/uncertainties/user_guide.html#correlated-variables

    """

    def __init__(self,
                 df: DataFrame,
                 fluxcol: str,
                 fluxgapfilledcol: str,
                 tacol: str,
                 vpdcol: str,
                 swincol: str):
        self.df = df
        self.fluxcol = fluxcol
        self.fluxgapfilledcol = fluxgapfilledcol
        self.tacol = tacol
        self.vpdcol = vpdcol
        self.swincol = swincol

        self.subset = self._make_subset()
        self._randunc_results = self.subset.copy()
        self.randunccol = f"{self.fluxcol}_RANDUNC"
        self._randunc_results[self.randunccol] = np.nan
        self._randunc_results_cumulatives = None

    @property
    def randunc_results(self):
        """Results subset containing uncertainty and auxiliary variables"""
        if not isinstance(self._randunc_results, DataFrame):
            raise Exception(f'No results available. Calculate results by calling .calc_random_uncertainty()')
        return self._randunc_results

    @property
    def randunc_results_cumulatives(self):
        """Cumulatives including random uncertainty propagation"""
        if not isinstance(self._randunc_results_cumulatives, DataFrame):
            raise Exception(f'No results available. Calculate results by calling .calc_cumulative_error_propagation()')
        return self._randunc_results_cumulatives

    @property
    def randunc_series(self) -> Series:
        """Return the calculated random uncertainty as series"""
        return self.randunc_results[self.randunccol]

    def run(self):
        self._calc_random_uncertainty()
        self._calc_cumulative_uncertainty_propagation()

    def _calc_random_uncertainty(self, ta_similarity: float = 2.5, vpd_similarity: float = 5,
                                 swin_similarity: float = 50):

        self._method1(ta_similarity=ta_similarity, vpd_similarity=vpd_similarity, swin_similarity=swin_similarity,
                      winsize_days=7, winsize_hours=1)

        self._method2(winsize_days=5, winsize_hours=1)

        # Method 2 is repeated with expanding time windows to get uncertainty for all records
        missing = True
        winsize_days = 5
        n_missing_randunc_prev = 0
        while missing:
            winsize_days += 1
            self._method2(winsize_days=winsize_days, winsize_hours=1)
            n_missing_randunc = self.randunc_results[self.randunccol].isnull().sum()
            missing = True if n_missing_randunc > 0 else False
            if n_missing_randunc == n_missing_randunc_prev: missing = False
            n_missing_randunc_prev = n_missing_randunc

        self._method3()
        self._method4()

    def _calc_cumulative_uncertainty_propagation(self):
        """Calculate the cumulative random uncertainty propagation

        Uses the uncertainties package for proper error propagation.
        Assumes independent measurement uncertainties (random errors).
        """

        fluxunc = 'FLUX+/-UNC'
        flux_upper = 'FLUX+UNC'
        flux_lower = 'FLUX-UNC'
        unc_cum = 'UNC_CUMULATIVE'

        # Combine gapfilled flux and random uncertainties to ufloats (from uncertainty package)
        subset = self.randunc_results[[self.fluxgapfilledcol, self.randunccol]].copy()

        # Combine gap-filled flux measurements with random uncertainties as ufloat objects
        # ufloat(value, uncertainty) creates objects that propagate errors correctly during arithmetic
        subset[fluxunc] = subset.apply(lambda row: ufloat(row[self.fluxgapfilledcol], row[self.randunccol]), axis=1)

        # Calculate cumulatives using only the necessary columns
        # This avoids unnecessary operations and makes the intent clearer
        subset_cumu = pd.DataFrame(index=subset.index)
        subset_cumu[self.fluxgapfilledcol] = subset[self.fluxgapfilledcol].cumsum()

        # Cumsum on ufloat Series properly propagates uncertainties via __add__ operator
        # Result: cumulative flux with sqrt-of-sum-of-squares error propagation
        subset_cumu[fluxunc] = subset[fluxunc].cumsum()

        # Extract the cumulative uncertainty (standard deviation) from each ufloat
        # The uncertainties package stores std_dev in the .s attribute
        subset_cumu[unc_cum] = subset_cumu[fluxunc].apply(lambda row: row.s if row is not None else np.nan)

        # Calculate upper and lower cumulative flux bounds
        subset_cumu[flux_upper] = subset_cumu[self.fluxgapfilledcol].add(subset_cumu[unc_cum])
        subset_cumu[flux_lower] = subset_cumu[self.fluxgapfilledcol].sub(subset_cumu[unc_cum])

        self._randunc_results_cumulatives = subset_cumu.copy()

    def report_cumulative_uncertainty_propagation(self):
        """Report cumulative uncertainty propagation with formatted table output."""
        fluxcum = self.randunc_results_cumulatives[self.fluxgapfilledcol].iloc[-1]
        unc = self.randunc_results_cumulatives['UNC_CUMULATIVE'].iloc[-1]
        ufloat = self.randunc_results_cumulatives['FLUX+/-UNC'].iloc[-1]
        lower = self.randunc_results_cumulatives['FLUX-UNC'].iloc[-1]
        upper = self.randunc_results_cumulatives['FLUX+UNC'].iloc[-1]

        # Summary table (using ASCII-compatible units)
        summary_data = {
            'Metric': [
                f'Cumulative {self.fluxgapfilledcol}',
                'Cumulative Uncertainty (+/- sigma)',
                'Lower Bound (flux - unc)',
                'Upper Bound (flux + unc)',
                'Range (upper - lower)'
            ],
            'Value': [
                f'{fluxcum:.4f}',
                f'{unc:.4f}',
                f'{lower:.4f}',
                f'{upper:.4f}',
                f'{(upper - lower):.4f}'
            ],
            'Unit': [
                'umol CO2 m-2 s-1',
                'umol CO2 m-2 s-1',
                'umol CO2 m-2 s-1',
                'umol CO2 m-2 s-1',
                'umol CO2 m-2 s-1'
            ]
        }
        df_summary = pd.DataFrame(summary_data)

        print(f"\n{'=' * 80}")
        print(f"CUMULATIVE UNCERTAINTY PROPAGATION")
        print(f"{'=' * 80}")
        print(df_summary.to_string(index=False))
        print(f"{'=' * 80}")
        print(f"Uncertainties package notation: {ufloat:.3f}\n")

    def report_method_summary(self):
        """Report summary of 4-method hierarchical uncertainty quantification."""
        n_records = len(self.randunc_results)
        n_measured = self.randunc_results[self.fluxcol].notna().sum()
        n_gapfilled = self.randunc_results[self.fluxgapfilledcol].notna().sum() - n_measured

        # Count records where each method provided the uncertainty
        method1_count = self.randunc_results['WINDOW_N_VALS_METHOD1'].notna().sum()
        method2_count = self.randunc_results['WINDOW_N_VALS_METHOD2'].notna().sum()
        method3_count = self.randunc_results['WINDOW_N_VALS_METHOD3'].notna().sum()
        method4_count = self.randunc_results['WINDOW_N_VALS_METHOD4'].notna().sum()

        # Statistics
        mean_unc = self.randunc_series.mean()
        std_unc = self.randunc_series.std()
        min_unc = self.randunc_series.min()
        max_unc = self.randunc_series.max()

        method_data = {
            'Method': [
                '1: Sliding Window (±7d, ±1h)',
                '2: Expanding Window (±5d+, ±1h)',
                '3: Similar Flux Range (no window)',
                '4: Nearest 5 Fluxes'
            ],
            'Records': [method1_count, method2_count, method3_count, method4_count],
            'Percentage': [
                f'{100*method1_count/n_records:.1f}%',
                f'{100*method2_count/n_records:.1f}%',
                f'{100*method3_count/n_records:.1f}%',
                f'{100*method4_count/n_records:.1f}%'
            ]
        }
        df_methods = pd.DataFrame(method_data)

        stats_data = {
            'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Range'],
            'Value (umol CO2 m-2 s-1)': [
                f'{mean_unc:.4f}',
                f'{std_unc:.4f}',
                f'{min_unc:.4f}',
                f'{max_unc:.4f}',
                f'{max_unc - min_unc:.4f}'
            ]
        }
        df_stats = pd.DataFrame(stats_data)

        print(f"\n{'=' * 80}")
        print(f"RANDOM UNCERTAINTY QUANTIFICATION - 4-METHOD SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total Records: {n_records:,}  |  Measured: {n_measured:,}  |  Gap-filled: {n_gapfilled:,}\n")

        print("METHOD DISTRIBUTION:")
        print(df_methods.to_string(index=False))

        print(f"\nUNCERTAINTY STATISTICS (umol CO2 m-2 s-1):")
        print(df_stats.to_string(index=False))
        print(f"{'=' * 80}\n")

    def showplot_cumulative_uncertainty_propagation(self):
        """Plot cumulative flux with uncertainty bounds."""
        fig, ax = plt.subplots(figsize=(14, 5.5))
        fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.1)

        df_plot = self.randunc_results_cumulatives[[self.fluxgapfilledcol, 'FLUX+UNC', 'FLUX-UNC']].copy()
        df_plot.columns = ['Cumulative Flux', 'Upper Bound (+σ)', 'Lower Bound (-σ)']

        ax.plot(df_plot.index, df_plot['Cumulative Flux'], linewidth=2.5, label='Cumulative Flux',
                color='black', zorder=3)
        ax.fill_between(df_plot.index, df_plot['Lower Bound (-σ)'], df_plot['Upper Bound (+σ)'],
                        alpha=0.3, color='red', label='Uncertainty Range (±σ)', zorder=1)
        ax.plot(df_plot.index, df_plot['Upper Bound (+σ)'], linewidth=1.5, linestyle='--',
                color='red', alpha=0.6, label='Bounds', zorder=2)
        ax.plot(df_plot.index, df_plot['Lower Bound (-σ)'], linewidth=1.5, linestyle='--',
                color='red', alpha=0.6, zorder=2)

        ax.set_xlabel('Datetime', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Cumulative {self.fluxgapfilledcol} (umol CO2 m-2 s-1)', fontsize=11, fontweight='bold')
        ax.set_title('Cumulative Uncertainty Propagation with Error Bounds', fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle=':')

        plt.show()

    def showplot_random_uncertainty(self):
        fig = plt.figure(facecolor='white', figsize=(18, 9))
        fig.suptitle("Random Uncertainties - 4-Method Hierarchical Analysis",
                     fontsize=theme.FIGHEADER_FONTSIZE, y=0.98)

        # Tight gridspec with minimal spacing
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.25,
                              left=0.05, right=0.98, top=0.94, bottom=0.08)

        # Axes
        ax_orig = fig.add_subplot(gs[0, 0:2])
        ax_gapfilled = fig.add_subplot(gs[1, 0:2], sharex=ax_orig)
        ax_scatter = fig.add_subplot(gs[0:2, 2:4])

        # Time series (measured, method 1)
        _df = self.randunc_results[[self.fluxcol, self.randunccol, 'WINDOW_N_VALS_METHOD1']].copy()
        _df = _df.loc[_df['WINDOW_N_VALS_METHOD1'] >= 5]
        _df = _df.dropna()
        ax_orig.plot(_df.index, _df[self.fluxcol], label=f"measured {self.fluxcol}", color="black",
                     alpha=1, markersize=5, markeredgecolor="black", mfc='none')
        ax_orig.errorbar(_df.index.values, _df[self.fluxcol], _df[self.randunccol],
                         elinewidth=5, ecolor='red', alpha=.2)
        ax_orig.set_title(f"Measured {self.fluxcol} with Method 1 Uncertainties\n(±7 days, ±1 hour, meteorological similarity)",
                         fontsize=10, fontweight='bold')
        ax_orig.set_ylabel(f'{self.fluxcol}\n(umol CO2 m-2 s-1)', fontsize=9)
        ax_orig.grid(True, alpha=0.3, linestyle=':')

        # Scatter (measured, method 1)
        ScatterXY(x=_df[self.fluxcol], y=_df[self.randunccol], ax=ax_scatter).plot()
        ax_scatter.set_title(f"Measured {self.fluxcol} vs. Method 1 Uncertainty\n(Flux-Uncertainty Relationship)",
                            fontsize=10, fontweight='bold')
        ax_scatter.set_xlabel(f'{self.fluxcol} (μmol CO₂ m⁻² s⁻¹)', fontsize=9)
        ax_scatter.set_ylabel(f'Uncertainty (+/- sigma)\n(umol CO2 m-2 s-1)', fontsize=9)
        ax_scatter.grid(True, alpha=0.3, linestyle=':')

        # Time series (gapfilled, method 1-4)
        _df = self.randunc_results[[self.fluxgapfilledcol, self.randunccol]].copy()
        _df = _df.dropna()
        ax_gapfilled.plot(_df.index, _df[self.fluxgapfilledcol], label=f"gap-filled {self.fluxgapfilledcol}",
                          color="black",
                          alpha=1, markersize=5, markeredgecolor="black", mfc='none')
        ax_gapfilled.errorbar(_df.index.values, _df[self.fluxgapfilledcol], _df[self.randunccol],
                              elinewidth=5, ecolor='red', alpha=.2)
        ax_gapfilled.set_title(f"Gap-filled {self.fluxgapfilledcol} with Methods 1-4 Uncertainties\n(Hierarchical: Method 1→2→3→4)",
                              fontsize=10, fontweight='bold')
        ax_gapfilled.set_ylabel(f'{self.fluxgapfilledcol}\n(umol CO2 m-2 s-1)', fontsize=9)
        ax_gapfilled.set_xlabel('Datetime', fontsize=9)
        ax_gapfilled.grid(True, alpha=0.3, linestyle=':')

        default_legend(ax=ax_orig)
        default_legend(ax=ax_gapfilled)
        default_format(ax=ax_orig)
        default_format(ax=ax_gapfilled)
        plt.setp(ax_orig.get_xticklabels(), visible=False)
        nice_date_ticks(ax=ax_orig)

        fig.show()

    #     d = res[[fluxcol, randunc.randunccol]].copy()
    #     d = d.dropna()
    #     from uncertainties import ufloat
    #     d['ufloats'] = d.apply(lambda row: ufloat(row['NEE_CUT_REF_orig'], row['NEE_CUT_REF_orig_RANDUNC']), axis=1)
    #
    #     import matplotlib.pyplot as plt
    #     d.plot(alpha=.5)
    #     # res[[fluxcol, randunc.randunccol]].plot(alpha=.5)
    #     plt.show()
    #     from diive.core.plotting.scatter import Scatter
    #

    def _make_subset(self):
        return self.df[[self.fluxcol, self.fluxgapfilledcol, self.tacol, self.vpdcol, self.swincol]].copy()

    def _method1(self, ta_similarity: float = 2.5, vpd_similarity: float = 5, swin_similarity: float = 50,
                 winsize_days: int = 7, winsize_hours: int = 1):
        """

        From Pastorello et al. (2020):
            NEE-RANDUNC Method 1 (direct standard deviation method): For a sliding
            window of ±7 days and ±1 hour of the time-of-day of the current timestamp,
            the random uncertainty is calculated as the standard deviation of the
            measured fluxes. The similarity in the meteorological conditions evaluated
            as in the MDS gap-filling method and a minimum of five measured values
            must be present; otherwise, method 2 is used.

        From ONEflux source code:
            The random uncertainty in the measurements has been estimated starting from
            the filtered NEE using two hierarchical methods: for half hours where original
            fluxes are available the random uncertainty is calculated (method 1) as the
            standard deviation of the fluxes measured in a moving window of +/- 7 days
            and +/- one hour with similar meteorological conditions (as for the MDS
            gapfilling: TA +/- 2.5 °C, VPD +/- 5 hPa, SW_IN +/- 50 W m-2 if radiation
            is higher than 50 W m-2, +/-20 if lower).
            source: https://github.com/fluxnet/ONEFlux/blob/55b6610499e8104450d84f134c3e53284d05e137/oneflux_steps/nee_proc/info/info_hh.txt

        """
        print(f"Calculating random uncertainty with window size +/-{winsize_days} days "
              f"and +/-{winsize_hours} hours (method 1) ...")
        tic = time.time()
        for ix, row in self.subset.iterrows():
            # Current data
            cur_dt = pd.to_datetime(ix)
            cur_flux = row[self.fluxcol]
            if np.isnan(cur_flux): continue
            cur_ta = row[self.tacol]
            cur_vpd = row[self.vpdcol]
            cur_swin = row[self.swincol]

            # Window limits
            cur_ta_upper = cur_ta + ta_similarity
            cur_ta_lower = cur_ta - ta_similarity
            cur_swin_upper = cur_swin + swin_similarity
            cur_swin_lower = cur_swin - swin_similarity
            cur_vpd_upper = cur_vpd + vpd_similarity
            cur_vpd_lower = cur_vpd - vpd_similarity

            # Time range of window
            start_dt = cur_dt - dt.timedelta(days=winsize_days)
            end_dt = cur_dt + dt.timedelta(days=winsize_days)
            starttime = (cur_dt - dt.timedelta(hours=winsize_hours)).time()
            endtime = (cur_dt + dt.timedelta(hours=winsize_hours)).time()

            # Window data, datetime range
            subset_win = df_between_two_dates(df=self.subset, start_date=start_dt, end_date=end_dt).copy()

            # Window data, limit datetime range to +/- 1 hour
            subset_win = subset_win.between_time(start_time=starttime, end_time=endtime)  # Thank you pandas!!

            # Remove data outside limits
            _filter = (subset_win[self.tacol] >= cur_ta_lower) & (subset_win[self.tacol] <= cur_ta_upper)
            subset_win = subset_win[_filter]
            _filter = (subset_win[self.swincol] >= cur_swin_lower) & (subset_win[self.swincol] <= cur_swin_upper)
            subset_win = subset_win[_filter]
            _filter = (subset_win[self.vpdcol] >= cur_vpd_lower) & (subset_win[self.vpdcol] <= cur_vpd_upper)
            subset_win = subset_win[_filter]

            # Calculate if min. 5 values, otherwise NaN
            n_vals = subset_win[self.fluxcol].count()
            randunc = subset_win[self.fluxcol].std() if n_vals >= 5 else np.nan

            self._randunc_results.loc[cur_dt, self.randunccol] = randunc
            self._randunc_results.loc[cur_dt, 'WINDOW_N_VALS_METHOD1'] = n_vals

        toc = time.time() - tic
        print(f"Time needed: {toc:.2f}s")

    def _method2(self, winsize_days: int = 5, winsize_hours: int = 1):
        """

        From Pastorello et al. (2020):
            NEE-RANDUNC Method 2 (median standard deviation method): For a sliding
            window of ±5 days and ±1 hour of the time-of-day of the current timestamp,
            random uncertainty is calculated as the median of the random uncertainty
            (calculated with NEE-RANDUNC Method 1) of similar fluxes, i.e., within the
            range of ±20% and not less than 2 μmolCO2 m–2 s–1.

        From ONEflux source code:
            Random uncertainty for gapfilled halfhours or original halfhours where it
            has been impossible to find at least 5 measurements in the moving window with
            similar meteorological conditions, is estimated (method 2) as the median of
            the random uncertainty (calculated with method 1) of similar fluxes, where
            similar is defined as in the range of +/- 20% (but not less than 2 umolCO2 m-2 s-1).
            NEE is always expressed as umolCO2 m-2 s-1

        """
        print(f"Calculating random uncertainty with window size +/-{winsize_days} days "
              f"and +/-{winsize_hours} hours (method 2) ...")
        tic = time.time()
        subset = self.randunc_results.copy()
        ix_missing_randunc = subset[self.randunccol].isnull()

        for ix, row in subset.iterrows():
            # Current data
            cur_randunc = row[self.randunccol]
            if not np.isnan(cur_randunc): continue  # Continue with next if random uncertainty already available

            cur_dt = pd.to_datetime(ix)
            cur_gapfilledflux = row[self.fluxgapfilledcol]

            # Window limits
            # Similar is defined as in the range of +/- 20% (but not less than 2 umolCO2 m-2 s-1)
            cur_gapfilledflux_perc20 = cur_gapfilledflux * 0.2
            add = 2 if cur_gapfilledflux_perc20 < 2 else cur_gapfilledflux_perc20
            cur_flux_upper = cur_gapfilledflux + add
            cur_flux_lower = cur_gapfilledflux - add

            # Time range of window
            start_dt = cur_dt - dt.timedelta(days=winsize_days)
            end_dt = cur_dt + dt.timedelta(days=winsize_days)
            starttime = (cur_dt - dt.timedelta(hours=winsize_hours)).time()
            endtime = (cur_dt + dt.timedelta(hours=winsize_hours)).time()

            # Window data, datetime range
            subset_win = df_between_two_dates(df=subset, start_date=start_dt, end_date=end_dt).copy()

            # Window data, limit datetime range to +/- 1 hour
            subset_win = subset_win.between_time(start_time=starttime, end_time=endtime)  # Thank you pandas!!

            # Remove data outside limits
            _filter = (subset_win[self.fluxgapfilledcol] >= cur_flux_lower) \
                      & (subset_win[self.fluxgapfilledcol] <= cur_flux_upper)
            subset_win = subset_win[_filter]

            n_vals = subset_win[self.randunccol].count()
            randunc = subset_win[self.randunccol].median()

            self._randunc_results.loc[cur_dt, self.randunccol] = randunc
            self._randunc_results.loc[cur_dt, 'WINDOW_N_VALS_METHOD2'] = n_vals

        toc = time.time() - tic
        print(f"Time needed: {toc:.2f}s")

    def _method3(self):
        """
        Fill left-over gaps with uncertainty from similar fluxes
        """
        print(f"Calculating random uncertainty from similar fluxes (method 3) ...")
        tic = time.time()
        subset = self.randunc_results.copy()
        ix_missing_randunc = subset[self.randunccol].isnull()

        for ix, row in subset.iterrows():
            # Current data
            cur_randunc = row[self.randunccol]
            if not np.isnan(cur_randunc): continue  # Continue with next if random uncertainty already available

            cur_dt = pd.to_datetime(ix)
            cur_gapfilledflux = row[self.fluxgapfilledcol]

            # Window limits
            # Similar is defined as in the range of +/- 20% (but not less than 2 umolCO2 m-2 s-1)
            cur_gapfilledflux_perc20 = cur_gapfilledflux * 0.2
            add = 2 if cur_gapfilledflux_perc20 < 2 else cur_gapfilledflux_perc20
            cur_flux_upper = cur_gapfilledflux + add
            cur_flux_lower = cur_gapfilledflux - add

            subset_win = subset.copy()

            # Remove data outside limits
            _filter = (subset_win[self.fluxgapfilledcol] >= cur_flux_lower) \
                      & (subset_win[self.fluxgapfilledcol] <= cur_flux_upper)
            subset_win = subset_win[_filter]

            n_vals = subset_win[self.randunccol].count()
            randunc = subset_win[self.randunccol].median()

            self._randunc_results.loc[cur_dt, self.randunccol] = randunc
            self._randunc_results.loc[cur_dt, 'WINDOW_N_VALS_METHOD3'] = n_vals

        toc = time.time() - tic
        print(f"Time needed: {toc:.2f}s")

    def _method4(self):
        """
        Fill left-over gaps with uncertainty from fluxes closest to current flux,
        without similarity restrictions

        Useful if there are fluxes higher outside the +/- 20% flux similarity used
        in method 2 and method 3.
        """
        print(f"Calculating random uncertainty from similar fluxes (method 3) ...")
        tic = time.time()
        subset = self.randunc_results.copy()

        for ix, row in subset.iterrows():
            # Current data
            cur_randunc = row[self.randunccol]
            if not np.isnan(cur_randunc): continue  # Continue with next if random uncertainty already available

            cur_dt = pd.to_datetime(ix)
            cur_gapfilledflux = row[self.fluxgapfilledcol]

            subset_win = subset.sort_values(by=self.fluxgapfilledcol, ascending=True)
            cur_ix = subset_win.index.get_loc(cur_dt)
            start_ix = cur_ix - 5
            start_ix = 0 if start_ix < 0 else start_ix
            end_ix = cur_ix + 5
            subset_win = subset_win.iloc[start_ix:end_ix]

            n_vals = subset_win[self.randunccol].count()
            randunc = subset_win[self.randunccol].median()

            self._randunc_results.loc[cur_dt, self.randunccol] = randunc
            self._randunc_results.loc[cur_dt, 'WINDOW_N_VALS_METHOD4'] = n_vals

        toc = time.time() - tic
        print(f"Time needed: {toc:.2f}s")
