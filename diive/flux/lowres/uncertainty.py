"""
UNCERTAINTY: FLUX MEASUREMENT UNCERTAINTY ESTIMATION
=====================================================

Calculate random and systematic uncertainties for flux measurements.

Part of the diive library: https://github.com/holukas/diive
"""

import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from uncertainties import ufloat

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import default_format, default_legend, nice_date_ticks
from diive.core.plotting.scatter import ScatterXY
from diive.core.utils.console import console as _console, info
from diive.gapfilling.similarity import (
    TA_TOLERANCE,
    VPD_TOLERANCE,
    swin_tolerance,
)


#: Divisor that turns a percentile range into a 1-sigma-equivalent for the joint
#: uncertainty. ONEFlux uses the 16th/84th percentiles for NEE (which bracket
#: +/-1 sigma, so the range is divided by 2) and the 25th/75th percentiles for
#: the energy fluxes LE/H (the interquartile range, IQR = 1.349 sigma).
JOINT_DIVISOR_1SIGMA = 2.0    # 16th/84th percentiles (NEE USTAR scenarios)
JOINT_DIVISOR_IQR = 1.349     # 25th/75th percentile IQR (LE/H energy-balance corr.)


def joint_uncertainty_pas20(randunc: Series,
                            scenario_lower: Series,
                            scenario_upper: Series,
                            divisor: float = JOINT_DIVISOR_1SIGMA) -> Series:
    """Per-record joint uncertainty (Pastorello et al. 2020 / ONEFlux ``compute_join``).

    Combines, in quadrature, the random measurement uncertainty with the
    uncertainty from the flux-partitioning/filtering scenario ensemble (for NEE:
    the USTAR-threshold percentile scenarios; for LE/H: the energy-balance
    correction percentiles). The scenario spread is first converted to a
    1-sigma-equivalent by ``divisor``:

        JOINTUNC = sqrt( RANDUNC^2 + ((scenario_upper - scenario_lower) / divisor)^2 )

    Faithful port of ONEFlux ``compute_join`` (``oneflux_steps/nee_proc/src/dataset.c``)
    and the energy-flux variant (``energy_proc/src/dataset.c``). ONEFlux returns an
    invalid value when any of the three inputs is invalid; here that maps to NaN,
    which numpy propagates through ``sqrt`` automatically.

    Args:
        randunc: per-record random uncertainty (e.g. ``{flux}_RANDUNC`` from
            :class:`RandomUncertaintyPAS20`).
        scenario_lower: lower-percentile scenario flux (NEE: 16th, e.g.
            ``NEE_CUT_16``; LE/H: 25th, e.g. ``LEcorr25``).
        scenario_upper: upper-percentile scenario flux (NEE: 84th, e.g.
            ``NEE_CUT_84``; LE/H: 75th, e.g. ``LEcorr75``).
        divisor: percentile-range -> 1-sigma factor. Use
            :data:`JOINT_DIVISOR_1SIGMA` (2.0) for the 16th/84th NEE percentiles
            and :data:`JOINT_DIVISOR_IQR` (1.349) for the 25th/75th LE/H IQR.

    Returns:
        The joint uncertainty as a Series aligned to ``randunc``'s index, NaN
        where any input is missing.
    """
    randunc = randunc.astype(float)
    scenario_lower = scenario_lower.astype(float).reindex(randunc.index)
    scenario_upper = scenario_upper.astype(float).reindex(randunc.index)
    scenario_sigma = (scenario_upper - scenario_lower) / divisor
    joint = np.sqrt(randunc ** 2 + scenario_sigma ** 2)
    joint.name = 'JOINTUNC'
    return joint


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
        Pastorello, G., Trotta, C., Canfora, E., Chu, H., Christianson, D., Cheah, Y.-W.,
            Poindexter, C., Chen, J., Elbashandy, A., Humphrey, M., Isaac, P., Polidori, D.,
            Reichstein, M., Ribeca, A., Van Ingen, C., Vuichard, N., Zhang, L., Amiro, B.,
            Ammann, C., … Papale, D. (2020). The FLUXNET2015 dataset and the ONEFlux processing
            pipeline for eddy covariance data. Scientific Data, 7(1), 225.
            https://doi.org/10.1038/s41597-020-0534-3

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
                 swincol: str,
                 vpd_in_kpa: bool = True):
        """Random uncertainty estimation.

        Args:
            df: DataFrame with the columns below and a regular datetime index.
            fluxcol: measured flux to estimate uncertainty for (umol CO2 m-2 s-1).
            fluxgapfilledcol: gap-filled flux (umol CO2 m-2 s-1).
            tacol: air temperature (deg C). Similarity tolerance: 2.5 deg C.
            vpdcol: vapor pressure deficit. Unit is set by ``vpd_in_kpa``.
            swincol: short-wave incoming radiation (W m-2). Similarity tolerance:
                the record's own SW_IN clamped into [20, 50] W m-2.
            vpd_in_kpa: if True (default, diive convention, matching the MDS
                gap-filler) ``vpdcol`` is in kPa and is converted to hPa
                internally for the faithful ONEFlux 5-hPa similarity tolerance.
                Pass False if ``vpdcol`` is already in hPa.
        """
        self.df = df
        self.fluxcol = fluxcol
        self.fluxgapfilledcol = fluxgapfilledcol
        self.tacol = tacol
        self.vpdcol = vpdcol
        self.swincol = swincol
        self.vpd_in_kpa = vpd_in_kpa
        # The ONEFlux VPD similarity tolerance is 5 hPa; convert a kPa column to
        # hPa for the comparison while leaving the user-facing results in kPa.
        self._vpd_factor = 10.0 if vpd_in_kpa else 1.0

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

    def run(self, progress_callback=None):
        """Run the 4-method hierarchical uncertainty quantification.

        Args:
            progress_callback: optional ``callable(phase, n_phases, done, total)``
                for a GUI progress bar. ``phase`` is 1..4 (the method being run),
                ``n_phases`` is 4, and ``done``/``total`` track the per-record loop
                within that method. The per-record loops (method 1 especially)
                dominate runtime, hence the within-method reporting.
        """
        self._calc_random_uncertainty(progress_callback=progress_callback)
        self._calc_cumulative_uncertainty_propagation()

    def _calc_random_uncertainty(self, ta_similarity: float = TA_TOLERANCE,
                                 vpd_similarity: float = VPD_TOLERANCE,
                                 progress_callback=None):

        # Initialize window count columns for all methods
        self._randunc_results['WINDOW_N_VALS_METHOD1'] = np.nan
        self._randunc_results['WINDOW_N_VALS_METHOD2'] = np.nan
        self._randunc_results['WINDOW_N_VALS_METHOD3'] = np.nan
        self._randunc_results['WINDOW_N_VALS_METHOD4'] = np.nan

        # ONEFlux methods (randunc.c): ±7-day/±1-hour std (method 1),
        # then ±14-day median of method-1 results (method 2).
        self._method1(ta_similarity=ta_similarity, vpd_similarity=vpd_similarity,
                      winsize_days=7, winsize_hours=1, progress_callback=progress_callback)
        self._method2(winsize_days=14, progress_callback=progress_callback)

        # diive extensions to fill records ONEFlux leaves undefined (see class docstring).
        self._method3(progress_callback=progress_callback)
        self._method4(progress_callback=progress_callback)

    @staticmethod
    def _report_progress(progress_callback, phase: int, i: int, total: int):
        """Emit (phase, n_phases=4, done, total) at ~1% steps and on the last record."""
        if progress_callback is None:
            return
        step = max(1, total // 100)
        if i % step == 0 or i == total - 1:
            progress_callback(phase, 4, i + 1, total)

    def _calc_cumulative_uncertainty_propagation(self):
        """Calculate the cumulative random uncertainty propagation

        Propagates the per-record random uncertainties through the running flux
        sum. Random errors are assumed independent, so the cumulative uncertainty
        is their quadrature (root-sum-of-squares) combination:

            UNC_CUMULATIVE[k] = sqrt( sum_{i<=k} randunc_i^2 )

        Both running sums use pandas' skipna semantics, so a missing flux or
        uncertainty for a single record leaves the downstream cumulatives defined
        (that record simply contributes nothing). This avoids the failure mode of
        an object-dtype ``ufloat`` cumsum, where one NaN poisons every later value.
        """

        fluxunc = 'FLUX+/-UNC'
        flux_upper = 'FLUX+UNC'
        flux_lower = 'FLUX-UNC'
        unc_cum = 'UNC_CUMULATIVE'

        flux = self.randunc_results[self.fluxgapfilledcol]
        randunc = self.randunc_results[self.randunccol]

        subset_cumu = pd.DataFrame(index=flux.index)
        # Cumulative flux (skips NaN flux, as before).
        subset_cumu[self.fluxgapfilledcol] = flux.cumsum()

        # Quadrature sum of independent random errors. Count a record's variance
        # only where it contributes to the flux sum; cumsum skips NaN, so a record
        # with a missing uncertainty contributes nothing rather than nullifying the
        # rest of the series.
        variance = (randunc ** 2).where(flux.notna())
        subset_cumu[unc_cum] = np.sqrt(variance.cumsum())

        # Cumulative flux with its uncertainty as ufloat objects (value = cumulative
        # flux, std = cumulative uncertainty); kept for the report's +/- notation.
        # Built directly from the two Series above (O(n)) instead of an O(n^2)
        # running ufloat cumsum.
        subset_cumu[fluxunc] = [
            ufloat(n if pd.notna(n) else np.nan, s if pd.notna(s) else np.nan)
            for n, s in zip(subset_cumu[self.fluxgapfilledcol], subset_cumu[unc_cum])]

        # Calculate upper and lower cumulative flux bounds (+/- 1 sigma)
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
        """Plot the random-uncertainty results."""
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
                 winsize_days: int = 7, winsize_hours: int = 1, progress_callback=None):
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

        # Vectorised inner loop: pull the columns into numpy once and locate each
        # record's +/-winsize_days window by position via searchsorted on the
        # (sorted) timestamps, instead of building a fresh DataFrame slice per
        # record. Bit-identical to the pandas version: df_between_two_dates is
        # inclusive on both ends, and between_time's +/-1 h time-of-day band
        # (which wraps around midnight) is reproduced on the window's time-of-day.
        idx = self.subset.index
        index_values = idx.values  # datetime64, for searchsorted
        flux = self.subset[self.fluxcol].to_numpy(dtype=float)
        ta = self.subset[self.tacol].to_numpy(dtype=float)
        vpd = self.subset[self.vpdcol].to_numpy(dtype=float) * self._vpd_factor
        swin = self.subset[self.swincol].to_numpy(dtype=float)
        hr = (idx.hour + idx.minute / 60.0 + idx.second / 3600.0).to_numpy(dtype=float)

        randunc_out = self._randunc_results[self.randunccol].to_numpy(dtype=float).copy()
        nvals_out = self._randunc_results['WINDOW_N_VALS_METHOD1'].to_numpy(dtype=float).copy()

        win = np.timedelta64(int(winsize_days), 'D')
        measured = np.where(np.isfinite(flux))[0]
        for k, i in enumerate(measured):
            self._report_progress(progress_callback, 1, k, measured.size)
            cur_ta = ta[i]
            cur_vpd = vpd[i]
            cur_swin = swin[i]
            # SWIN tolerance is clamped to the radiation level of the current record
            # (ONEFlux: tolerance grows with SWIN between 20 and 50 W m-2).
            cur_swin_tol = swin_tolerance(cur_swin)

            # +/-winsize_days window by position (df_between_two_dates is inclusive).
            t = index_values[i]
            lo = int(np.searchsorted(index_values, t - win, side='left'))
            hi = int(np.searchsorted(index_values, t + win, side='right'))

            # +/-winsize_hours time-of-day band: the time-of-day of (t - 1 h) ..
            # (t + 1 h), inclusive and wrap-aware (matches pandas between_time).
            start_h = (hr[i] - winsize_hours) % 24.0
            end_h = (hr[i] + winsize_hours) % 24.0
            hw = hr[lo:hi]
            if start_h <= end_h:
                tmask = (hw >= start_h) & (hw <= end_h)
            else:  # band crosses midnight
                tmask = (hw >= start_h) | (hw <= end_h)

            # Meteorological similarity (strict < on the absolute difference);
            # only finite fluxes contribute to the standard deviation.
            fw = flux[lo:hi]
            sel = (tmask
                   & (np.abs(ta[lo:hi] - cur_ta) < ta_similarity)
                   & (np.abs(swin[lo:hi] - cur_swin) < cur_swin_tol)
                   & (np.abs(vpd[lo:hi] - cur_vpd) < vpd_similarity)
                   & np.isfinite(fw))
            vals = fw[sel]
            n_vals = vals.size
            # ONEFlux requires more than 5 matching values; sample SD (N-1).
            randunc_out[i] = np.std(vals, ddof=1) if n_vals >= 6 else np.nan
            nvals_out[i] = n_vals

        self._randunc_results[self.randunccol] = randunc_out
        self._randunc_results['WINDOW_N_VALS_METHOD1'] = nvals_out

        toc = time.time() - tic
        info(f"Time needed: {toc:.2f}s")

    def _method2(self, winsize_days: int = 14, progress_callback=None):
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
        # Pulled into numpy once; each record's +/-winsize_days window is a
        # contiguous slice located by searchsorted (bit-identical to the pandas
        # df_between_two_dates slice, which is inclusive on both ends).
        idx = self.randunc_results.index
        index_values = idx.values
        gf = self.randunc_results[self.fluxgapfilledcol].to_numpy(dtype=float)
        randunc = self.randunc_results[self.randunccol].to_numpy(dtype=float)
        out = randunc.copy()
        nvals_out = self._randunc_results['WINDOW_N_VALS_METHOD2'].to_numpy(dtype=float).copy()

        win = np.timedelta64(int(winsize_days), 'D')
        todo = np.where(~np.isfinite(randunc))[0]
        for k, i in enumerate(todo):
            self._report_progress(progress_callback, 2, k, todo.size)
            cur_gf = gf[i]
            # Flux-similarity limits: +/- 20% of the flux magnitude, but not less
            # than 2 umolCO2 m-2 s-1 (ONEFlux uses the absolute value, so the
            # window is symmetric for negative fluxes too).
            add = abs(cur_gf) * 0.2
            if add < 2:
                add = 2
            upper = cur_gf + add
            lower = cur_gf - add

            t = index_values[i]
            lo = int(np.searchsorted(index_values, t - win, side='left'))
            hi = int(np.searchsorted(index_values, t + win, side='right'))
            gw = gf[lo:hi]
            rw = randunc[lo:hi]
            # Keep method-1 results (non-null uncertainty) for similar fluxes.
            similar = rw[(gw >= lower) & (gw <= upper) & np.isfinite(rw)]
            n_vals = similar.size
            out[i] = np.median(similar) if n_vals > 0 else np.nan  # NaN when none found
            nvals_out[i] = n_vals

        self._randunc_results[self.randunccol] = out
        self._randunc_results['WINDOW_N_VALS_METHOD2'] = nvals_out

        toc = time.time() - tic
        info(f"Time needed: {toc:.2f}s")

    def _method3(self, progress_callback=None):
        """
        diive extension (not in ONEFlux): fill left-over gaps with the median
        uncertainty of similar fluxes over the whole record (no time window).
        """
        info(f"Calculating random uncertainty from similar fluxes (method 3) ...")
        tic = time.time()
        gf = self.randunc_results[self.fluxgapfilledcol].to_numpy(dtype=float)
        randunc = self.randunc_results[self.randunccol].to_numpy(dtype=float)
        out = randunc.copy()
        nvals_out = self._randunc_results['WINDOW_N_VALS_METHOD3'].to_numpy(dtype=float).copy()

        # Sort by gap-filled flux once so each record's +/-20% flux band is a
        # contiguous slice (searchsorted), instead of masking the whole record per
        # gap. NaN fluxes sort to the end (np.argsort) and never enter a band.
        order = np.argsort(gf, kind='stable')
        gf_sorted = gf[order]
        randunc_sorted = randunc[order]

        todo = np.where(~np.isfinite(randunc))[0]
        for k, i in enumerate(todo):
            self._report_progress(progress_callback, 3, k, todo.size)
            cur_gf = gf[i]
            if not np.isfinite(cur_gf):  # no finite +/-20% band -> no similar fluxes
                nvals_out[i] = 0
                continue
            # Flux-similarity limits: +/- 20% of the flux magnitude, floor 2 umolCO2 m-2 s-1
            add = abs(cur_gf) * 0.2
            if add < 2:
                add = 2
            lo = int(np.searchsorted(gf_sorted, cur_gf - add, side='left'))
            hi = int(np.searchsorted(gf_sorted, cur_gf + add, side='right'))
            seg = randunc_sorted[lo:hi]
            valid = seg[np.isfinite(seg)]
            n_vals = valid.size
            out[i] = np.median(valid) if n_vals > 0 else np.nan
            nvals_out[i] = n_vals

        self._randunc_results[self.randunccol] = out
        self._randunc_results['WINDOW_N_VALS_METHOD3'] = nvals_out

        toc = time.time() - tic
        info(f"Time needed: {toc:.2f}s")

    def _method4(self, progress_callback=None):
        """
        diive extension (not in ONEFlux): fill left-over gaps with the median
        uncertainty of the fluxes closest in magnitude to the current flux,
        without similarity restrictions.

        Useful if there are fluxes outside the +/- 20% flux similarity used
        in method 2 and method 3.
        """
        info(f"Calculating random uncertainty from similar fluxes (method 4) ...")
        tic = time.time()
        subset = self.randunc_results

        # Sort by gap-filled flux ONCE (the old code re-sorted the whole frame on
        # every iteration); a deterministic sort gives identical neighbour windows.
        subset_sorted = subset.sort_values(by=self.fluxgapfilledcol, ascending=True)
        sorted_index = subset_sorted.index
        randunc_sorted = subset_sorted[self.randunccol].to_numpy(dtype=float)

        record_index = subset.index
        randunc = subset[self.randunccol].to_numpy(dtype=float)
        out = randunc.copy()
        nvals_out = self._randunc_results['WINDOW_N_VALS_METHOD4'].to_numpy(dtype=float).copy()

        todo = np.where(~np.isfinite(randunc))[0]
        for k, i in enumerate(todo):
            self._report_progress(progress_callback, 4, k, todo.size)
            cur_ix = sorted_index.get_loc(record_index[i])
            start_ix = max(0, cur_ix - 5)
            end_ix = cur_ix + 5
            seg = randunc_sorted[start_ix:end_ix]
            valid = seg[np.isfinite(seg)]
            n_vals = valid.size
            out[i] = np.median(valid) if n_vals > 0 else np.nan
            nvals_out[i] = n_vals

        self._randunc_results[self.randunccol] = out
        self._randunc_results['WINDOW_N_VALS_METHOD4'] = nvals_out

        toc = time.time() - tic
        info(f"Time needed: {toc:.2f}s")


class JointUncertaintyPAS20:
    """Joint uncertainty estimation (Pastorello et al. 2020 / ONEFlux ``compute_join``).

    Combines the per-record **random measurement uncertainty** (e.g.
    ``{flux}_RANDUNC`` from :class:`RandomUncertaintyPAS20`) with the
    **scenario-ensemble uncertainty** from the flux-partitioning/filtering
    percentile scenarios, added in quadrature:

        JOINTUNC = sqrt( RANDUNC^2 + ((scenario_upper - scenario_lower) / divisor)^2 )

    For NEE the scenario ensemble is the USTAR-threshold percentile scenarios and
    the 16th/84th percentiles bracket +/-1 sigma (``divisor=2``, the default,
    :data:`JOINT_DIVISOR_1SIGMA`); ONEFlux emits this as e.g.
    ``NEE_CUT_REF_JOINTUNC``. For the energy fluxes LE/H the ensemble is the
    energy-balance-correction percentiles and the 25th/75th interquartile range
    is used (``divisor=`` :data:`JOINT_DIVISOR_IQR` ``=1.349``), emitted as
    ``LE_CORR_JOINTUNC`` / ``H_CORR_JOINTUNC``.

    Faithful port of the ONEFlux ``compute_join`` helper
    (``oneflux_steps/nee_proc/src/dataset.c``) and its energy-flux variant
    (``energy_proc/src/dataset.c``).

    Cumulative propagation treats the two error sources differently, matching
    their nature: the random error is independent between records and propagates
    in quadrature, whereas the scenario (e.g. USTAR-threshold) choice is fully
    correlated across the record — the same threshold applies to every half-hour
    — so its cumulative uncertainty is the running spread of the cumulative
    scenario sums. The two cumulative terms are then combined in quadrature.

    References:
        Pastorello et al. (2020), Scientific Data 7, 225.
        https://doi.org/10.1038/s41597-020-0534-3

    Example:
        See ``examples/flux/lowres/flux_uncertainty.py``.
    """

    def __init__(self,
                 df: DataFrame,
                 randunccol: str,
                 scenario_lower_col: str,
                 scenario_upper_col: str,
                 fluxgapfilledcol: str | None = None,
                 divisor: float = JOINT_DIVISOR_1SIGMA,
                 name: str | None = None):
        """Joint uncertainty estimation.

        Args:
            df: DataFrame with the columns below and a regular datetime index.
            randunccol: per-record random uncertainty (e.g. ``NEE_CUT_REF_RANDUNC``).
            scenario_lower_col: lower-percentile scenario flux (NEE: 16th, e.g.
                ``NEE_CUT_16``; LE/H: 25th).
            scenario_upper_col: upper-percentile scenario flux (NEE: 84th, e.g.
                ``NEE_CUT_84``; LE/H: 75th).
            fluxgapfilledcol: optional gap-filled flux for the cumulative
                propagation (the central line the cumulative band brackets). When
                omitted, only the per-record joint uncertainty is computed.
            divisor: percentile-range -> 1-sigma factor. :data:`JOINT_DIVISOR_1SIGMA`
                (2.0, default) for the 16th/84th NEE percentiles,
                :data:`JOINT_DIVISOR_IQR` (1.349) for the 25th/75th LE/H IQR.
            name: output column name. Defaults to the random-uncertainty column
                with a trailing ``_RANDUNC`` replaced by ``_JOINTUNC`` (so
                ``NEE_CUT_REF_RANDUNC`` -> ``NEE_CUT_REF_JOINTUNC``), else
                ``{randunccol}_JOINTUNC``.
        """
        self.df = df
        self.randunccol = randunccol
        self.scenario_lower_col = scenario_lower_col
        self.scenario_upper_col = scenario_upper_col
        self.fluxgapfilledcol = fluxgapfilledcol
        self.divisor = divisor

        if name is not None:
            self.jointunccol = name
        elif randunccol.endswith('_RANDUNC'):
            self.jointunccol = randunccol[:-len('_RANDUNC')] + '_JOINTUNC'
        else:
            self.jointunccol = f"{randunccol}_JOINTUNC"

        #: per-record scenario (e.g. USTAR-filtering) 1-sigma-equivalent term.
        self.scenarionunccol = f"{self.jointunccol}_SCENARIO"

        self._jointunc_results: DataFrame | None = None
        self._jointunc_results_cumulatives: DataFrame | None = None

    @property
    def jointunc_results(self) -> DataFrame:
        """Results subset: random term, scenario term and the joint uncertainty."""
        if not isinstance(self._jointunc_results, DataFrame):
            raise Exception("No results available. Call .run() first.")
        return self._jointunc_results

    @property
    def jointunc_results_cumulatives(self) -> DataFrame:
        """Cumulative flux with the propagated joint-uncertainty bounds."""
        if not isinstance(self._jointunc_results_cumulatives, DataFrame):
            raise Exception(
                "No cumulatives available. Pass fluxgapfilledcol and call .run().")
        return self._jointunc_results_cumulatives

    @property
    def jointunc_series(self) -> Series:
        """Return the joint uncertainty as a series."""
        return self.jointunc_results[self.jointunccol]

    def run(self) -> None:
        """Compute the per-record joint uncertainty (and cumulatives if a
        gap-filled flux was provided)."""
        randunc = self.df[self.randunccol]
        lower = self.df[self.scenario_lower_col]
        upper = self.df[self.scenario_upper_col]

        joint = joint_uncertainty_pas20(randunc, lower, upper, divisor=self.divisor)
        scenario_sigma = (upper.astype(float).reindex(randunc.index)
                          - lower.astype(float).reindex(randunc.index)) / self.divisor

        out = DataFrame(index=randunc.index)
        out[self.randunccol] = randunc.astype(float)
        out[self.scenarionunccol] = scenario_sigma
        out[self.jointunccol] = joint
        self._jointunc_results = out

        if self.fluxgapfilledcol is not None:
            self._calc_cumulative(lower, upper, randunc)

    def _calc_cumulative(self, lower: Series, upper: Series, randunc: Series) -> None:
        """Cumulative flux with its propagated joint-uncertainty bounds.

        The random term is independent -> quadrature accumulation
        ``sqrt(cumsum(randunc^2))``; the scenario (USTAR) term is fully correlated
        -> the running spread of the cumulative scenario sums
        ``(cumsum(upper) - cumsum(lower)) / divisor``. Both use pandas' skipna
        cumsum so a single missing record never poisons the tail.
        """
        flux = self.df[self.fluxgapfilledcol].astype(float)
        lower = lower.astype(float).reindex(flux.index)
        upper = upper.astype(float).reindex(flux.index)
        randunc = randunc.astype(float).reindex(flux.index)

        cum = DataFrame(index=flux.index)
        cum[self.fluxgapfilledcol] = flux.cumsum()

        # Random part: count a record's variance only where it contributes to the
        # flux sum (cumsum skips NaN, mirroring RandomUncertaintyPAS20).
        variance = (randunc ** 2).where(flux.notna())
        cum_random = np.sqrt(variance.cumsum())
        # Scenario part: fully correlated across records -> running spread of the
        # cumulative scenario sums.
        cum_scenario = (upper.cumsum() - lower.cumsum()) / self.divisor

        cum['UNC_RANDOM_CUMULATIVE'] = cum_random
        cum['UNC_SCENARIO_CUMULATIVE'] = cum_scenario
        cum['UNC_CUMULATIVE'] = np.sqrt(cum_random ** 2 + cum_scenario ** 2)
        cum['FLUX+UNC'] = cum[self.fluxgapfilledcol] + cum['UNC_CUMULATIVE']
        cum['FLUX-UNC'] = cum[self.fluxgapfilledcol] - cum['UNC_CUMULATIVE']
        self._jointunc_results_cumulatives = cum

    def report_summary(self) -> None:
        """Report a compact joint-uncertainty summary."""
        joint = self.jointunc_series
        rand = self.jointunc_results[self.randunccol]
        scen = self.jointunc_results[self.scenarionunccol]
        n = int(joint.notna().sum())
        _console.print(f"\n{'=' * 80}")
        _console.print("JOINT UNCERTAINTY (PAS20) SUMMARY")
        _console.print(f"{'=' * 80}")
        _console.print(f"Output column: {self.jointunccol}  |  records with estimate: {n:,}")
        _console.print(f"  random term      mean: {rand.mean():.4f}  median: {rand.median():.4f}")
        _console.print(f"  scenario term    mean: {scen.mean():.4f}  median: {scen.median():.4f}")
        _console.print(f"  joint            mean: {joint.mean():.4f}  median: {joint.median():.4f}")
        _console.print(f"{'=' * 80}\n")
