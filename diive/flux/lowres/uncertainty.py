"""
UNCERTAINTY: FLUX MEASUREMENT UNCERTAINTY ESTIMATION
=====================================================

Calculate random and systematic uncertainties for flux measurements.

Part of the diive library: https://github.com/holukas/diive
"""

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
from diive.core.utils.console import console as _console, info
from diive.gapfilling.similarity import (
    TA_TOLERANCE,
    VPD_TOLERANCE,
    swin_tolerance,
    window_mean_sd_count,
)


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
    Hierarchical random uncertainty quantification for eddy covariance flux data.

    Implements the PAS20 (Pastorello et al. 2020) uncertainty methodology
    which applies hierarchical methods to estimate random measurement uncertainty
    in flux measurements. The approach fills gaps and propagates uncertainties through
    gap-filling to produce cumulative uncertainty estimates.

    Methods 1 and 2 are faithful ports of the ONEFlux C reference
    (``oneflux_steps/nee_proc/src/randunc.c``, functions ``random_method_1`` /
    ``random_method_2``). Methods 3 and 4 are diive extensions (not in ONEFlux):
    ONEFlux leaves a record's random uncertainty undefined when method 2 finds no
    similar fluxes, whereas methods 3 and 4 relax the matching further to guarantee
    every record gets an estimate.

    Core Methods:
    * **Method 1** (ONEFlux) — standard deviation of measured fluxes in a sliding
      ±7-day / ±1-hour window under similar meteorological conditions (TA, VPD,
      SW_IN), requiring more than 5 matching values.
    * **Method 2** (ONEFlux) — median of the method-1 uncertainties of similar
      fluxes (within ±20%, floor 2 µmol) in a ±14-day window.
    * **Method 3** (diive extension) — median of method-1 uncertainties of similar
      fluxes over the whole record (no time window).
    * **Method 4** (diive extension) — median of the uncertainties of the fluxes
      closest in magnitude, with no similarity restriction.

    Key Features:
    * Hierarchical approach ensures all records have uncertainty estimates
    * Proper error propagation through cumulative calculations using uncertainties package
    * Separate day/night or combined uncertainty depending on data quality

    References:
        Hollinger, D. Y., & Richardson, A. D. (2005). Uncertainty in eddy covariance measurements
            and its application to physiological models. Tree Physiology, 25(7), 873–885.
            https://doi.org/10.1093/treephys/25.7.873
        Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux processing
            pipeline for eddy covariance data. 27. https://doi.org/10.1038/s41597-020-0534-3

    Example:
        See `examples/flux/lowres/flux_uncertainty.py` for complete examples of
        random uncertainty calculation and cumulative uncertainty propagation.
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

    def _calc_random_uncertainty(self, ta_similarity: float = TA_TOLERANCE,
                                 vpd_similarity: float = VPD_TOLERANCE):

        # Initialize window count columns for all methods
        self._randunc_results['WINDOW_N_VALS_METHOD1'] = np.nan
        self._randunc_results['WINDOW_N_VALS_METHOD2'] = np.nan
        self._randunc_results['WINDOW_N_VALS_METHOD3'] = np.nan
        self._randunc_results['WINDOW_N_VALS_METHOD4'] = np.nan

        # ONEFlux methods (randunc.c): ±7-day/±1-hour std (method 1),
        # then ±14-day median of method-1 results (method 2).
        self._method1(ta_similarity=ta_similarity, vpd_similarity=vpd_similarity,
                      winsize_days=7, winsize_hours=1)
        self._method2(winsize_days=14)

        # diive extensions to fill records ONEFlux leaves undefined (see class docstring).
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

        _console.print(f"\n{'=' * 80}")
        _console.print(f"CUMULATIVE UNCERTAINTY PROPAGATION")
        _console.print(f"{'=' * 80}")
        _console.print(df_summary.to_string(index=False))
        _console.print(f"{'=' * 80}")
        _console.print(f"Uncertainties package notation: {ufloat:.3f}\n")

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
                '1: Sliding Window (+/-7d, +/-1h) [ONEFlux]',
                '2: Similar Flux Median (+/-14d) [ONEFlux]',
                '3: Similar Flux Median (no window) [diive]',
                '4: Nearest Fluxes [diive]'
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

        _console.print(f"\n{'=' * 80}")
        _console.print(f"RANDOM UNCERTAINTY QUANTIFICATION - 4-METHOD SUMMARY")
        _console.print(f"{'=' * 80}")
        _console.print(f"Total Records: {n_records:,}  |  Measured: {n_measured:,}  |  Gap-filled: {n_gapfilled:,}\n")

        _console.print("METHOD DISTRIBUTION:")
        _console.print(df_methods.to_string(index=False))

        _console.print(f"\nUNCERTAINTY STATISTICS (umol CO2 m-2 s-1):")
        _console.print(df_stats.to_string(index=False))
        _console.print(f"{'=' * 80}\n")

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
        ax_orig.errorbar(_df.index.to_numpy(), _df[self.fluxcol], _df[self.randunccol],
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
        ax_gapfilled.errorbar(_df.index.to_numpy(), _df[self.fluxgapfilledcol], _df[self.randunccol],
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

    def _method1(self, ta_similarity: float = TA_TOLERANCE, vpd_similarity: float = VPD_TOLERANCE,
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
        info(f"Calculating random uncertainty with window size +/-{winsize_days} days "
             f"and +/-{winsize_hours} hours (method 1) ...")
        tic = time.time()
        for ix in self.subset.index:
            # Current data
            cur_dt = pd.to_datetime(ix)
            cur_flux = self.subset.loc[ix, self.fluxcol]
            if np.isnan(cur_flux): continue
            cur_ta = self.subset.loc[ix, self.tacol]
            cur_vpd = self.subset.loc[ix, self.vpdcol]
            cur_swin = self.subset.loc[ix, self.swincol]

            # SWIN tolerance is clamped to the radiation level of the current record
            # (ONEFlux: tolerance grows with SWIN between 20 and 50 W m-2).
            cur_swin_tol = swin_tolerance(cur_swin)

            # Time range of window
            start_dt = cur_dt - dt.timedelta(days=winsize_days)
            end_dt = cur_dt + dt.timedelta(days=winsize_days)
            starttime = (cur_dt - dt.timedelta(hours=winsize_hours)).time()
            endtime = (cur_dt + dt.timedelta(hours=winsize_hours)).time()

            # Window data, datetime range
            subset_win = df_between_two_dates(df=self.subset, start_date=start_dt, end_date=end_dt).copy()

            # Window data, limit datetime range to +/- 1 hour
            subset_win = subset_win.between_time(start_time=starttime, end_time=endtime)  # Thank you pandas!!

            # Keep records within tolerance of the current meteo conditions.
            # ONEFlux uses a strict less-than on the absolute difference.
            _filter = (
                    ((subset_win[self.tacol] - cur_ta).abs() < ta_similarity)
                    & ((subset_win[self.swincol] - cur_swin).abs() < cur_swin_tol)
                    & ((subset_win[self.vpdcol] - cur_vpd).abs() < vpd_similarity)
            )
            subset_win = subset_win[_filter]

            # ONEFlux requires more than 5 matching values; the uncertainty is the
            # sample standard deviation (N-1) of the similar fluxes.
            _, randunc, n_vals = window_mean_sd_count(
                subset_win[self.fluxcol].to_numpy(), min_count=6, ddof=1)

            self._randunc_results.loc[cur_dt, self.randunccol] = randunc
            self._randunc_results.loc[cur_dt, 'WINDOW_N_VALS_METHOD1'] = n_vals

        toc = time.time() - tic
        info(f"Time needed: {toc:.2f}s")

    def _method2(self, winsize_days: int = 14):
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

        Note:
            The Pastorello et al. (2020) text describes a ±5-day / ±1-hour window, but the
            ONEFlux C reference (``random_method_2``) uses a ±14-day window with no
            time-of-day restriction. This implementation follows the C code.

        """
        info(f"Calculating random uncertainty with window size +/-{winsize_days} days (method 2) ...")
        tic = time.time()
        # Snapshot: method 2 draws only from method-1 results (rows whose
        # uncertainty is already set at this point), never from its own output.
        subset = self.randunc_results.copy()

        for ix in subset.index:
            # Current data
            cur_randunc = subset.loc[ix, self.randunccol]
            if not np.isnan(cur_randunc): continue  # Continue with next if random uncertainty already available

            cur_dt = pd.to_datetime(ix)
            cur_gapfilledflux = subset.loc[ix, self.fluxgapfilledcol]

            # Flux-similarity limits: +/- 20% of the flux magnitude, but not less
            # than 2 umolCO2 m-2 s-1 (ONEFlux uses the absolute value, so the
            # window is symmetric for negative fluxes too).
            add = abs(cur_gapfilledflux) * 0.2
            if add < 2:
                add = 2
            cur_flux_upper = cur_gapfilledflux + add
            cur_flux_lower = cur_gapfilledflux - add

            # Time range of window (no time-of-day restriction in ONEFlux method 2)
            start_dt = cur_dt - dt.timedelta(days=winsize_days)
            end_dt = cur_dt + dt.timedelta(days=winsize_days)
            subset_win = df_between_two_dates(df=subset, start_date=start_dt, end_date=end_dt)

            # Keep method-1 results (non-null uncertainty) for similar fluxes.
            _filter = (
                    (subset_win[self.fluxgapfilledcol] >= cur_flux_lower)
                    & (subset_win[self.fluxgapfilledcol] <= cur_flux_upper)
                    & subset_win[self.randunccol].notna()
            )
            similar_randunc = subset_win.loc[_filter, self.randunccol]

            n_vals = similar_randunc.count()
            randunc = similar_randunc.median()  # NaN when no similar fluxes found

            self._randunc_results.loc[cur_dt, self.randunccol] = randunc
            self._randunc_results.loc[cur_dt, 'WINDOW_N_VALS_METHOD2'] = n_vals

        toc = time.time() - tic
        info(f"Time needed: {toc:.2f}s")

    def _method3(self):
        """
        diive extension (not in ONEFlux): fill left-over gaps with the median
        uncertainty of similar fluxes over the whole record (no time window).
        """
        info(f"Calculating random uncertainty from similar fluxes (method 3) ...")
        tic = time.time()
        subset = self.randunc_results.copy()

        for ix in subset.index:
            # Current data
            cur_randunc = subset.loc[ix, self.randunccol]
            if not np.isnan(cur_randunc): continue  # Continue with next if random uncertainty already available

            cur_dt = pd.to_datetime(ix)
            cur_gapfilledflux = subset.loc[ix, self.fluxgapfilledcol]

            # Flux-similarity limits: +/- 20% of the flux magnitude, floor 2 umolCO2 m-2 s-1
            add = abs(cur_gapfilledflux) * 0.2
            if add < 2:
                add = 2
            cur_flux_upper = cur_gapfilledflux + add
            cur_flux_lower = cur_gapfilledflux - add

            # Remove data outside limits
            _filter = (subset[self.fluxgapfilledcol] >= cur_flux_lower) \
                      & (subset[self.fluxgapfilledcol] <= cur_flux_upper)
            similar_randunc = subset.loc[_filter, self.randunccol]

            n_vals = similar_randunc.count()
            randunc = similar_randunc.median()

            self._randunc_results.loc[cur_dt, self.randunccol] = randunc
            self._randunc_results.loc[cur_dt, 'WINDOW_N_VALS_METHOD3'] = n_vals

        toc = time.time() - tic
        info(f"Time needed: {toc:.2f}s")

    def _method4(self):
        """
        diive extension (not in ONEFlux): fill left-over gaps with the median
        uncertainty of the fluxes closest in magnitude to the current flux,
        without similarity restrictions.

        Useful if there are fluxes outside the +/- 20% flux similarity used
        in method 2 and method 3.
        """
        info(f"Calculating random uncertainty from similar fluxes (method 4) ...")
        tic = time.time()
        subset = self.randunc_results.copy()

        for ix in subset.index:
            # Current data
            cur_randunc = subset.loc[ix, self.randunccol]
            if not np.isnan(cur_randunc): continue  # Continue with next if random uncertainty already available

            cur_dt = pd.to_datetime(ix)
            cur_gapfilledflux = subset.loc[ix, self.fluxgapfilledcol]

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
        info(f"Time needed: {toc:.2f}s")
