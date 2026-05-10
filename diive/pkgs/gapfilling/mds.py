"""

MARGINAL DISTRIBUTION SAMPLING (MDS)
Gap-filling after Reichstein et al (2005)

Reference: https://doi.org/10.1111/j.1365-2486.2005.001002.x

"""
from collections import Counter

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import default_format, default_legend
from diive.core.plotting.plotfuncs import nice_date_ticks
from diive.core.plotting.styles.LightTheme import colorwheel_36_blackfirst, generate_plot_marker_list
from diive.pkgs.gapfilling.scores import prediction_scores


class _FluxMDS:
    gfsuffix = '_gfMDS'

    def __init__(self,
                 df: DataFrame,
                 flux: str,
                 swin: str,
                 ta: str,
                 vpd: str,
                 swin_tol: list = None,  # Default defined below: [20, 50]
                 ta_tol: float = 2.5,
                 vpd_tol: float = 0.5,
                 avg_min_n_vals: int = 5,
                 verbose: int = 1):
        """Gap-filling for ecosystem fluxes, based on marginal distribution sampling (MDS
        described in Reichstein et al. (2005).

        Missing values are replaced by the average *flux* value during
        similar meteorological conditions.

        The MDS method in diive was implemented following the descriptions in
        Reichstein et al. (2005) and Vekuri et al. (2023).

        References:
            Reichstein et al. (2005). On the separation of net ecosystem exchange
                into assimilation and ecosystem respiration: Review and improved
                algorithm. Global Change Biology, 11(9), 1424–1439.
                https://doi.org/10.1111/j.1365-2486.2005.001002.x
            Vekuri et al. (2023). A widely-used eddy covariance gap-filling method
                creates systematic bias in carbon balance estimates.
                Scientific Reports, 13(1), 1720.
                https://doi.org/10.1038/s41598-023-28827-2

        Args:
            df: Dataframe that contains data for *flux*, *swin*, *ta* and *vpd*.
            flux: Name of flux variable in *df* that will be gap-filled.
            swin: Name of short-wave incoming radiation variable in *df*. (W m-2)
            ta: Name of air temperature variable in *df*. (°C)
            vpd: Name of vapor pressure deficit variable in *df*. (kPa)
            todo swin_class: Used for grouping *flux* data into groups of similar
                meteorological conditions. Data in the respective group must
                not deviate by more than +/- 50 W m-2 (default). (W m-2)
            ta_tol: Used for grouping *flux* data into groups of similar
                meteorological conditions. Data in the respective group must
                not deviate by more than +/- 2.5 °C (default). (°C)
            vpd_tol: Used for grouping *flux* data into groups of similar
                meteorological conditions. Data in the respective group must
                not deviate by more than +/- 0.5 kPa (default). (kPa)
            todo avg_min_n_vals: Minimum number of measured *flux* values required to
                calculate the average *flux* value for gaps during nighttime.
            verbose: Value 1 creates more text output.
        """
        self._gapfilling_df = df[[flux, swin, ta, vpd]].copy()
        self.flux = flux
        self.swin = swin
        self.ta = ta
        self.vpd = vpd
        if not swin_tol:
            self.swin_tol = [20, 50]
        else:
            if isinstance(swin_tol, list):
                self.swin_tol = swin_tol
            else:
                raise TypeError('swin_class must be a list with two elements. (default: [20, 50])')
        self.ta_tol = ta_tol
        self.vpd_tol = vpd_tol
        self.avg_min_n_vals = avg_min_n_vals if avg_min_n_vals else 0
        self.verbose = verbose

        self._scores = dict()

        self.target_gapfilled = f"{self.flux}{self.gfsuffix}"
        self.target_gapfilled_flag = f"FLAG_{self.flux}{self.gfsuffix}_ISFILLED"

        self._gapfilling_df = self._add_newcols()

        self.workdf = DataFrame()

    def get_gapfilled_target(self):
        """Gap-filled target time series"""
        return self.gapfilling_df_[self.target_gapfilled].copy()

    def get_flag(self):
        """Gap-filling flag, where 0=observed, 1+=gap-filled"""
        return self.gapfilling_df_[self.target_gapfilled_flag]

    @property
    def gapfilled_(self) -> pd.Series:
        """Gap-filled data."""
        series = self.get_gapfilled_target()
        if not isinstance(series, pd.Series):
            raise Exception('No gap-filled data available.')
        return series

    @property
    def target_col(self) -> str:
        """Gap-filled data."""
        if not isinstance(self.flux, str):
            raise Exception('No name for gap-filled variable available.')
        return self.flux

    @property
    def gapfilling_df_(self) -> DataFrame:
        """Dataframe containing all data."""
        if not isinstance(self._gapfilling_df, DataFrame):
            raise Exception('No dataframe containing all data available.')
        return self._gapfilling_df

    @property
    def scores_(self) -> dict:
        """Return scores for model used in gap-filling"""
        if not self._scores:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores

    def run(self):
        # https://www.geeksforgeeks.org/apply-function-to-every-row-in-a-pandas-dataframe/
        # https://labs.quansight.org/blog/unlocking-c-level-performance-in-df-apply

        print(f"\n{'=' * 30}\nStarting MDS gap-filling of flux {self.flux}.")

        self._gapfilling_df = self.gapfilling_df_.copy()
        locs_missing = self.gapfilling_df_['.PREDICTIONS'].isnull()
        self.workdf = self._gapfilling_df[locs_missing].copy()

        # A1: SWIN, TA, VPD, NEE available within 7 days (highest quality gap-filling).
        self.workdf, self._gapfilling_df = self._run_all_available(days=7, quality=1)

        # A2: SWIN, TA, VPD, NEE available within 14 days
        self.workdf, self._gapfilling_df = self._run_all_available(days=14, quality=2)

        # A3: SWIN, NEE available within 7 days
        self.workdf, self._gapfilling_df = self._run_two_available(days=7, quality=3)

        # A4: NEE available within |dt| <= 1h on same day
        self.workdf, self._gapfilling_df = self._run_mdc(days=0, hours=1, quality=4)

        # B1: same hour NEE available within |dt| <= 1 day
        self.workdf, self._gapfilling_df = self._run_mdc(days=1, hours=1, quality=5)

        # B2: SWIN, TA, VPD, NEE available within 21 days
        self.workdf, self._gapfilling_df = self._run_all_available(days=21, quality=6)

        # B3: SWIN, TA, VPD, NEE available within 28 days
        self.workdf, self._gapfilling_df = self._run_all_available(days=28, quality=7)

        # B4: SWIN, NEE available within 14 days
        self.workdf, self._gapfilling_df = self._run_two_available(days=14, quality=8)

        # C+: SWIN, TA, VPD, NEE available within 35-140 days
        quality = 8  # Quality from previous step B4
        for d in range(35, 147, 7):
            quality += 1
            self.workdf, self._gapfilling_df = self._run_all_available(days=d, quality=quality)

        # C+: SWIN, NEE available within 21-140 days
        quality = 24  # Maximum possible quality from previous step C+
        for d in range(21, 147, 7):
            quality += 1
            self.workdf, self._gapfilling_df = self._run_two_available(days=d, quality=quality)

        # C+: same hour NEE available within |dt| <= 7-X days
        quality = 42  # Maximum possible quality from previous step C+
        for d in range(21, 147, 7):
            quality += 1
            self.workdf, self._gapfilling_df = self._run_mdc(days=d, hours=1, quality=quality)

        # Gap-filled measurement time series
        self.gapfilling_df_[self.target_gapfilled] = self.gapfilling_df_[self.flux].fillna(
            self.gapfilling_df_['.PREDICTIONS'])

        # Gap-filling flag is 0 where measurement available
        locs_measured_missing = self.gapfilling_df_[self.flux].isnull()
        locs_measured_available = ~locs_measured_missing
        self.gapfilling_df_.loc[locs_measured_available, self.target_gapfilled_flag] = 0

        # Gap-filling flag is equal to prediction quality where measurement was missing
        self.gapfilling_df_.loc[locs_measured_missing, self.target_gapfilled_flag] = \
            self.gapfilling_df_.loc[locs_measured_missing, '.PREDICTIONS_QUALITY']
        # self.gapfilling_df_[self.target_gapfilled_flag] = \
        #     self.gapfilling_df_[self.target_gapfilled_flag].fillna(self.gapfilling_df_['.PREDICTIONS_QUALITY'])

        # # Flag
        # # Make flag column that indicates where predictions for
        # # missing targets are available, where 0=observed, 1=gapfilled
        # # todo Note that missing predicted gaps = 0. change?
        # _gapfilled_locs = self._gapfilling_df[self.pred_gaps_col].isnull()  # Non-gapfilled locations
        # _gapfilled_locs = ~_gapfilled_locs  # Inverse for gapfilled locations
        # self._gapfilling_df[self.target_gapfilled_flag_col] = _gapfilled_locs
        # self._gapfilling_df[self.target_gapfilled_flag_col] = self._gapfilling_df[
        #     self.target_gapfilled_flag_col].astype(
        #     int)

        # import matplotlib.pyplot as plt
        # # self.df[self.target_gapfilled_flag].plot(label="gapfilled", ls='none', markersize=4, marker="o")
        # # self.df[self.target_gapfilled].plot(label="gapfilled", ls='none', markersize=4, marker="o")
        # self.gapfilling_df_['.PREDICTIONS'].plot(label="predictions", ls='none', markersize=4, marker="o")
        # self.gapfilling_df_[self.flux].plot(ls='none', markersize=4, marker="o")
        # plt.legend()
        # plt.show()

        # Calculate scores
        scoredf = self.gapfilling_df_[['.PREDICTIONS', self.flux]].copy()
        scoredf = scoredf.dropna()
        self._scores = prediction_scores(predictions=scoredf['.PREDICTIONS'], targets=scoredf[self.flux])

        self._scores['mean_quality_flag_gap_predictions'] = \
            self.gapfilling_df_.loc[locs_measured_missing, self.target_gapfilled_flag].mean()

        print("MDS gap-filling done.")

    def showplot(self):
        fig = plt.figure(facecolor='white', figsize=(16, 9), dpi=100, layout='constrained')
        gs = gridspec.GridSpec(3, 1, figure=fig)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        ax_flag = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_counts = fig.add_subplot(gs[2, 0], sharex=ax)
        flag = self.gapfilling_df_[self.target_gapfilled_flag]
        uniqueflags = list(flag.unique())
        uniqueflags.sort()
        colors = colorwheel_36_blackfirst()
        maxcolors = len(colors)
        markers = generate_plot_marker_list()
        maxmarker = len(markers)
        for ix, uf in enumerate(uniqueflags):
            locs = flag == uf
            data = self.gapfilling_df_.loc[locs, :]
            label = f"measured ({self.flux})" if uf == 0 else f"gap-filled quality {uf}, mean ± SD"
            n_vals = data[self.target_gapfilled].count()
            color = colors[35] if ix > (maxcolors - 1) else colors[ix]
            marker = markers[9] if ix > (maxmarker - 1) else markers[ix]
            ax.plot(data.index, data[self.target_gapfilled],
                    label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                    marker=marker, alpha=1, markersize=5, markeredgecolor=color, fillstyle='full')

            if uf > 0:
                # Add errorbars for gap-filled values, use fmt to not draw lines between the bars
                ax.errorbar(data.index, data[self.target_gapfilled], data['.PREDICTIONS_SD'],
                            elinewidth=5, ecolor=color, alpha=.2, fmt='none')

                # Add counts for gap-filled values
                ax_counts.plot(data.index, data['.PREDICTIONS_COUNTS'],
                               label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                               marker=marker, alpha=1, markersize=5, markeredgecolor=color, fillstyle='full')

            ax_flag.plot(data.index, data[self.target_gapfilled_flag],
                         label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                         marker=marker, alpha=1, markersize=5, markeredgecolor=color, fillstyle='full')
        fig.suptitle(f"Variable {self.flux} gap-filled using MDS: {self.target_gapfilled}",
                     fontsize=theme.FIGHEADER_FONTSIZE)
        default_format(ax=ax, ax_ylabel_txt=f"{self.flux}", ax_labels_fontsize=theme.AX_LABELS_FONTSIZE_12)
        ax.tick_params(labelbottom=False)
        ax_flag.tick_params(labelbottom=False)
        default_format(ax=ax_flag, ax_ylabel_txt="flag value", ax_labels_fontsize=theme.AX_LABELS_FONTSIZE_12)
        default_format(ax=ax_counts, ax_ylabel_txt="number of values for mean",
                       ax_labels_fontsize=theme.AX_LABELS_FONTSIZE_12)
        default_legend(ax=ax_flag, textsize=theme.FONTSIZE_TXT_LEGEND_SMALL_9, ncol=2)
        # default_legend(ax=ax_counts, textsize=theme.FONTSIZE_TXT_LEGEND_SMALL)

        nice_date_ticks(ax=ax)
        nice_date_ticks(ax=ax_flag)
        nice_date_ticks(ax=ax_counts)
        fig.show()

    def report(self):
        """Comprehensive MDS gap-filling report with detailed statistics and metrics."""
        potential_vals = len(self.gapfilling_df_.index)
        n_vals_before = self.gapfilling_df_[self.flux].count()
        n_vals_missing_before = self.gapfilling_df_[self.flux].isnull().sum()
        pct_missing_before = 100.0 * n_vals_missing_before / potential_vals

        n_vals_after = self.gapfilling_df_[self.target_gapfilled].count()
        n_vals_missing_after = self.gapfilling_df_[self.target_gapfilled].isnull().sum()
        pct_missing_after = 100.0 * n_vals_missing_after / potential_vals

        n_vals_filled = n_vals_missing_before - n_vals_missing_after
        pct_recovery = 100.0 * n_vals_filled / max(n_vals_missing_before, 1)

        mean_quality = self.gapfilling_df_['.PREDICTIONS_QUALITY'].mean()
        flagcounts = Counter(self.gapfilling_df_[self.target_gapfilled_flag])

        print(f"\n{'Marginal Distribution Sampling (MDS) Gap-Filling Report':=^80}")
        print(f"  Reference: Reichstein et al. (2005) - https://doi.org/10.1111/j.1365-2486.2005.001002.x")

        print(f"\n{'Algorithm Overview':-^80}")
        print(f"  MDS fills gaps using average flux from similar meteorological conditions.")
        print(f"  Hierarchical quality-based approach with progressively relaxed windows:")
        print(f"    • Quality 1-3 (A1-A3):  7-14 days,  all 3 variables (SWIN, TA, VPD)")
        print(f"    • Quality 4-8 (B1-B4):  21-28 days, 2-3 variables")
        print(f"    • Quality 9+  (C+):     35-140+ days, progressively fewer constraints")

        print(f"\n{'Parameters':-^80}")
        print(f"  {'Flux variable':<35} {self.flux}")
        print(f"  {'SWIN tolerance':<35} [{self.swin_tol[0]}, {self.swin_tol[1]}] W m-2")
        print(f"  {'TA tolerance':<35} {self.ta_tol} °C")
        print(f"  {'VPD tolerance':<35} {self.vpd_tol} kPa")
        print(f"  {'Min records for average':<35} {self.avg_min_n_vals}")

        print(f"\n{'Input Data Summary':-^80}")
        print(f"  {'Total records':<35} {potential_vals:>15,d}")
        print(
            f"  {'Available measurements':<35} {n_vals_before:>15,d}  ({100.0 * n_vals_before / potential_vals:>6.2f}%)")
        print(f"  {'Missing values':<35} {n_vals_missing_before:>15,d}  ({pct_missing_before:>6.2f}%)")

        print(f"\n{'Gap-Filling Performance':-^80}")
        print(f"  {'Values filled':<35} {n_vals_filled:>15,d}  ({pct_recovery:>6.2f}% of gaps)")
        print(f"  {'Remaining missing':<35} {n_vals_missing_after:>15,d}  ({pct_missing_after:>6.2f}% of total)")
        print(f"  {'Final data coverage':<35} {n_vals_after:>15,d}  ({100.0 * n_vals_after / potential_vals:>6.2f}%)")
        gap_recovery_efficiency = pct_recovery if pct_recovery > 0 else 0
        print(f"  {'Gap recovery efficiency':<35} {gap_recovery_efficiency:>15.2f}%")
        print(f"  {'Mean quality score':<35} {mean_quality:>15.3f}  (1=best, 9+=low)")

        print(f"\n{'Quality Distribution':-^80}")
        measured_count = flagcounts.get(0, 0)
        print(
            f"  {'Measured (original data)':<35} {measured_count:>15,d}  ({100.0 * measured_count / potential_vals:>6.2f}%)")

        quality_counts = {k: v for k, v in sorted(flagcounts.items()) if k > 0}
        if quality_counts:
            quality_groups = {
                'High quality (A1-A3, 1-3)': sum(v for k, v in quality_counts.items() if 1 <= k <= 3),
                'Medium quality (B1-B4, 4-8)': sum(v for k, v in quality_counts.items() if 4 <= k <= 8),
                'Low quality (C+, 9+)': sum(v for k, v in quality_counts.items() if k >= 9),
            }
            for group_name, count in quality_groups.items():
                if count > 0:
                    pct = 100.0 * count / potential_vals
                    print(f"  {group_name:<35} {count:>15,d}  ({pct:>6.2f}%)")

            print(f"\n  Quality level breakdown (detailed):")
            print(f"    {'Level':<10} {'Count':>12} {'Percentage':>12} {'Description':<45}")
            print(f"    {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 45}")
            quality_descriptions = {
                1: 'SWIN, TA, VPD: 7 days',
                2: 'SWIN, TA, VPD: 14 days',
                3: 'SWIN only: 7 days',
                4: 'NEE only: 1 hour same day',
                5: 'NEE only: 1 hour within 1 day',
                6: 'SWIN, TA, VPD: 21 days',
                7: 'SWIN, TA, VPD: 28 days',
                8: 'SWIN only: 14 days',
            }
            for quality, count in sorted(quality_counts.items()):
                pct = 100.0 * count / potential_vals
                # Get description or generate one for C+ levels (9+)
                if quality in quality_descriptions:
                    desc = quality_descriptions[quality]
                else:
                    # C+ levels: 9-24 use SWIN/TA/VPD, 25-40 use SWIN, 41+ use NEE
                    if 9 <= quality <= 24:
                        days = 35 + (quality - 9) * 7
                        desc = f'SWIN, TA, VPD: {days} days'
                    elif 25 <= quality <= 40:
                        days = 21 + (quality - 25) * 7
                        desc = f'SWIN only: {days} days'
                    elif quality >= 41:
                        days = 21 + (quality - 41) * 7
                        desc = f'NEE only: {days} days'
                    else:
                        desc = 'Unknown'
                print(f"    {quality:<10} {count:>12,d} {pct:>11.2f}% {desc:<45}")

        self.report_scores()
        print(f"{'':=^80}\n")

    def report_scores(self):
        """Print model performance scores with interpretation."""
        print(f"\n{'Model Performance Scores':-^80}")
        if self.scores_:
            for score, val in self.scores_.items():
                score_display = score.replace('_', ' ').title()
                print(f"  {score_display:<35} {val:>15.4f}")
        else:
            print("  No scores available")
        print(f"\n{'Key Insights':-^80}")
        print(f"  • Higher quality levels (e.g., 1-3) indicate more similar meteorological conditions")
        print(f"  • Mean quality score closer to 1 = more gap-filled values from strict criteria")
        print(f"  • Check 'Final data coverage' to assess overall gap-filling success")
        print(f"  • Review 'Quality Distribution' to understand data source reliability")

    def _run_all_available(self, days: int, quality: int):

        _df, workdf = self._prepare_dataframes()
        if workdf.empty:
            return workdf, _df

        print(f"MDS gap-filling quality {quality}    using SW_IN, TA, VPD in {days} days window ...")

        offset = pd.DateOffset(days=days)
        workdf['.START'] = pd.to_datetime(workdf.index) - offset
        workdf['.END'] = pd.to_datetime(workdf.index) + offset
        # .apply() needs .to_list() to return multiple series
        workdf[['.PREDICTIONS', '.PREDICTIONS_SD', '.PREDICTIONS_COUNTS']] = workdf.apply(self._a_1_2, axis=1).to_list()
        workdf['.PREDICTIONS_QUALITY'] = quality

        if quality == 1:
            _df['.PREDICTIONS'] = workdf['.PREDICTIONS'].copy()
            _df['.PREDICTIONS_SD'] = workdf['.PREDICTIONS_SD'].copy()
            _df['.PREDICTIONS_COUNTS'] = workdf['.PREDICTIONS_COUNTS'].copy()
            _df['.START'] = workdf['.START'].copy()
            _df['.END'] = workdf['.END'].copy()
            _df['.PREDICTIONS_QUALITY'] = workdf['.PREDICTIONS_QUALITY'].copy()
        else:
            _df = self._fill_predictions(_df, workdf)

        locs_missing = workdf['.PREDICTIONS'].isnull()  # Still missing values after gap-filling
        workdf = workdf[locs_missing].copy()  # Prepare dataframe for next gap-filling
        return workdf, _df

    def _prepare_dataframes(self) -> tuple[DataFrame, DataFrame]:
        _df = self.gapfilling_df_.copy()
        workdf = self.workdf.copy()
        return _df, workdf

    def _run_two_available(self, days: int, quality: int):

        _df, workdf = self._prepare_dataframes()
        if workdf.empty:
            return workdf, _df

        print(f"MDS gap-filling quality {quality}    using SW_IN in {days} days window ...")

        offset = pd.DateOffset(days=days)
        workdf['.START'] = pd.to_datetime(workdf.index) - offset
        workdf['.END'] = pd.to_datetime(workdf.index) + offset
        workdf[['.PREDICTIONS', '.PREDICTIONS_SD', '.PREDICTIONS_COUNTS']] = workdf.apply(self._a3, axis=1).to_list()
        workdf['.PREDICTIONS_QUALITY'] = quality

        _df = self._fill_predictions(_df, workdf)

        locs_missing = workdf['.PREDICTIONS'].isnull()  # Still missing values after gap-filling
        workdf = workdf[locs_missing].copy()  # Prepare dataframe for next gap-filling
        return workdf, _df

    def _run_mdc(self, days: int, hours: int, quality: int):

        _df, workdf = self._prepare_dataframes()
        if workdf.empty:
            return workdf, _df

        print(f"MDS gap-filling quality {quality}    using mean diurnal cycle of flux in "
              f"{days} days, {hours} hours window ...")

        offset = pd.DateOffset(days=days, hours=hours)
        workdf['.START'] = pd.to_datetime(workdf.index) - offset
        workdf['.END'] = pd.to_datetime(workdf.index) + offset
        if days == 0:
            workdf[['.PREDICTIONS', '.PREDICTIONS_SD', '.PREDICTIONS_COUNTS']] = workdf.apply(self._a4,
                                                                                              axis=1).to_list()
        else:
            workdf[['.PREDICTIONS', '.PREDICTIONS_SD', '.PREDICTIONS_COUNTS']] = workdf.apply(self._b1,
                                                                                              axis=1).to_list()
        workdf['.PREDICTIONS_QUALITY'] = quality

        _df = self._fill_predictions(_df, workdf)

        locs_missing = workdf['.PREDICTIONS'].isnull()  # Still missing values after gap-filling
        workdf = workdf[locs_missing].copy()  # Prepare dataframe for next gap-filling
        return workdf, _df

    def _fill_predictions(self, _df, workdf) -> DataFrame:

        # Check where no new predictions are available
        locs_not_available_fills = workdf['.PREDICTIONS'].isnull()
        # The inverse are locations where predictions are available
        locs_available_fills = ~locs_not_available_fills
        # Check where predictions are still needed
        locs_need_fill = _df['.PREDICTIONS'].isnull()
        # Locations where new predictions are available and still needed (both locs_ must be True)
        locs = locs_available_fills & locs_need_fill

        _df.loc[locs, '.PREDICTIONS'] = workdf.loc[locs, '.PREDICTIONS']
        _df.loc[locs, '.PREDICTIONS_SD'] = workdf.loc[locs, '.PREDICTIONS_SD']
        _df.loc[locs, '.PREDICTIONS_COUNTS'] = workdf.loc[locs, '.PREDICTIONS_COUNTS']
        _df.loc[locs, '.PREDICTIONS_QUALITY'] = workdf.loc[locs, '.PREDICTIONS_QUALITY']
        _df.loc[locs, '.START'] = workdf.loc[locs, '.START']
        _df.loc[locs, '.END'] = workdf.loc[locs, '.END']

        return _df

    def _a_1_2(self, row):
        locs = (
                (self.gapfilling_df_.index >= row['.START'])
                & (self.gapfilling_df_.index <= row['.END'])
                & (self.gapfilling_df_[f'.{self.ta}_UPPERLIM'] > row[self.ta])
                & (self.gapfilling_df_[f'.{self.ta}_LOWERLIM'] < row[self.ta])
                & (self.gapfilling_df_[f'.{self.swin}_UPPERLIM'] > row[self.swin])
                & (self.gapfilling_df_[f'.{self.swin}_LOWERLIM'] < row[self.swin])
                & (self.gapfilling_df_[f'.{self.vpd}_UPPERLIM'] > row[self.vpd])
                & (self.gapfilling_df_[f'.{self.vpd}_LOWERLIM'] < row[self.vpd])
        )
        avg, sd, counts = self._calc_avg(locs=locs)
        return avg, sd, counts

    def _a3(self, row):
        locs = (
                (self.gapfilling_df_.index >= row['.START'])
                & (self.gapfilling_df_.index <= row['.END'])
                & (self.gapfilling_df_[f'.{self.swin}_UPPERLIM'] > row[self.swin])
                & (self.gapfilling_df_[f'.{self.swin}_LOWERLIM'] < row[self.swin])
        )
        avg, sd, counts = self._calc_avg(locs=locs)
        return avg, sd, counts

    def _a4(self, row):
        locs = (
                (self.gapfilling_df_.index >= row['.START'])
                & (self.gapfilling_df_.index <= row['.END'])
        )
        avg, sd, counts = self._calc_avg(locs=locs)
        return avg, sd, counts

    def _b1(self, row):
        locs = (
                (self.gapfilling_df_.index >= row['.START'])
                & (self.gapfilling_df_.index <= row['.END'])
                & (self.gapfilling_df_.index.hour == row.name.hour)
        )
        avg, sd, counts = self._calc_avg(locs=locs)
        return avg, sd, counts

    def _calc_avg(self, locs: bool) -> [float, float]:
        _df = self.gapfilling_df_.loc[locs, [self.flux]].copy()
        _array = _df[self.flux].to_numpy()
        counts = len(_array[~np.isnan(_array)])

        # Return NaN if no flux records available
        if counts == 0:
            avg = np.nan
            sd = np.nan
            return avg, sd, counts

        if counts >= self.avg_min_n_vals:
            avg = np.nanmean(_array)
            sd = np.nanstd(_array)
        else:
            avg = np.nan
            sd = np.nan
        return avg, sd, counts

    def _add_newcols(self) -> pd.DataFrame:
        df = self.gapfilling_df_.copy()
        # Init new cols
        df['.TIMESTAMP'] = df.index
        df[self.target_gapfilled] = np.nan  # Gap-filling measurement
        df[self.target_gapfilled_flag] = np.nan  # Gap-filling flag
        df['.PREDICTIONS'] = np.nan
        df['.PREDICTIONS_SD'] = np.nan
        df['.PREDICTIONS_COUNTS'] = np.nan
        df['.PREDICTIONS_QUALITY'] = np.nan
        df['.START'] = np.nan
        df['.END'] = np.nan
        df[f'.{self.swin}_LOWERLIM'] = np.nan
        df[f'.{self.swin}_UPPERLIM'] = np.nan
        df[f'.{self.ta}_LOWERLIM'] = np.nan
        df[f'.{self.ta}_UPPERLIM'] = np.nan
        df[f'.{self.vpd}_LOWERLIM'] = np.nan
        df[f'.{self.vpd}_UPPERLIM'] = np.nan

        # Similarity limits for low radiation measurements
        lowrad = df[self.swin] <= 50
        df.loc[lowrad, f'.{self.swin}_LOWERLIM'] = df.loc[lowrad, self.swin].sub(self.swin_tol[0])
        df.loc[lowrad, f'.{self.swin}_UPPERLIM'] = df.loc[lowrad, self.swin].add(self.swin_tol[0])

        # Similarity limits for high radiation measurements
        highrad = df[self.swin] > 50
        df.loc[highrad, f'.{self.swin}_LOWERLIM'] = df.loc[highrad, self.swin].sub(self.swin_tol[1])
        df.loc[highrad, f'.{self.swin}_UPPERLIM'] = df.loc[highrad, self.swin].add(self.swin_tol[1])

        df[f'.{self.ta}_LOWERLIM'] = df[self.ta].sub(self.ta_tol)
        df[f'.{self.ta}_UPPERLIM'] = df[self.ta].add(self.ta_tol)
        df[f'.{self.vpd}_LOWERLIM'] = df[self.vpd].sub(self.vpd_tol)
        df[f'.{self.vpd}_UPPERLIM'] = df[self.vpd].add(self.vpd_tol)
        return df


class FluxMDS:
    """Gap-filling for ecosystem fluxes using Marginal Distribution Sampling (MDS).

    Fills missing flux data by matching similar meteorological conditions based on
    solar radiation (SWIN), air temperature (TA), and vapor pressure deficit (VPD).
    Uses the Reichstein et al. (2005) methodology with hierarchical quality levels
    that progressively relax meteorological constraints to fill gaps while preserving
    data quality.

    Gap-filling process:
    - Matches missing observations to complete observations with similar meteorological
      conditions within specified tolerance windows
    - Uses multiple quality levels with increasingly relaxed time-window and variable
      constraints to ensure coverage of all gaps
    - Flags gap-filled values with quality information

    Reference: https://doi.org/10.1111/j.1365-2486.2005.001002.x

    Examples:
        See examples/pkgs/gapfilling/gapfill_mds.py for basic usage.
        See examples/pkgs/gapfilling/gapfill_comparison.py for side-by-side comparison with
        Random Forest and XGBoost gap-filling methods.
    """

    gfsuffix = '_gfMDS'

    def __init__(self,
                 df: DataFrame,
                 flux: str,
                 swin: str,
                 ta: str,
                 vpd: str,
                 swin_tol: list = None,
                 ta_tol: float = 2.5,
                 vpd_tol: float = 0.5,
                 avg_min_n_vals: int = 5,
                 verbose: int = 1):
        """Initialize optimized MDS gap-filling.

        Args:
            df: DataFrame containing flux, swin, ta, vpd columns
            flux: Name of flux variable to gap-fill
            swin: Name of short-wave incoming radiation variable (W m-2)
            ta: Name of air temperature variable (°C)
            vpd: Name of vapor pressure deficit variable (kPa)
            swin_tol: SWIN tolerance [low_rad, high_rad] (default: [20, 50])
            ta_tol: TA tolerance in °C (default: 2.5)
            vpd_tol: VPD tolerance in kPa (default: 0.5)
            avg_min_n_vals: Minimum records for average (default: 5)
            verbose: Verbosity level (default: 1)
        """
        self._gapfilling_df = df[[flux, swin, ta, vpd]].copy()
        self.flux = flux
        self.swin = swin
        self.ta = ta
        self.vpd = vpd
        if not swin_tol:
            self.swin_tol = [20, 50]
        else:
            if isinstance(swin_tol, list):
                self.swin_tol = swin_tol
            else:
                raise TypeError('swin_class must be a list with two elements. (default: [20, 50])')
        self.ta_tol = ta_tol
        self.vpd_tol = vpd_tol
        self.avg_min_n_vals = avg_min_n_vals if avg_min_n_vals else 0
        self.verbose = verbose

        self._scores = dict()

        self.target_gapfilled = f"{self.flux}{self.gfsuffix}"
        self.target_gapfilled_flag = f"FLAG_{self.flux}{self.gfsuffix}_ISFILLED"

        self._gapfilling_df = self._add_newcols()

        self.workdf = DataFrame()

    def get_gapfilled_target(self):
        """Gap-filled target time series"""
        return self.gapfilling_df_[self.target_gapfilled].copy()

    def get_flag(self):
        """Gap-filling flag, where 0=observed, 1+=gap-filled"""
        return self.gapfilling_df_[self.target_gapfilled_flag]

    @property
    def gapfilled_(self) -> pd.Series:
        """Gap-filled data."""
        series = self.get_gapfilled_target()
        if not isinstance(series, pd.Series):
            raise Exception('No gap-filled data available.')
        return series

    @property
    def target_col(self) -> str:
        """Gap-filled data."""
        if not isinstance(self.flux, str):
            raise Exception('No name for gap-filled variable available.')
        return self.flux

    @property
    def gapfilling_df_(self) -> DataFrame:
        """Dataframe containing all data."""
        if not isinstance(self._gapfilling_df, DataFrame):
            raise Exception('No dataframe containing all data available.')
        return self._gapfilling_df

    @property
    def scores_(self) -> dict:
        """Return scores for model used in gap-filling"""
        if not self._scores:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores

    def run(self):
        """Execute optimized MDS gap-filling algorithm."""
        print(f"\n{'=' * 30}\nStarting MDS gap-filling of flux {self.flux}.")

        locs_missing = self.gapfilling_df_['.PREDICTIONS'].isnull()
        self._missing_mask = locs_missing.copy()

        # A1: SWIN, TA, VPD, NEE available within 7 days (highest quality gap-filling).
        self._run_all_available(days=7, quality=1)

        # A2: SWIN, TA, VPD, NEE available within 14 days
        self._run_all_available(days=14, quality=2)

        # A3: SWIN, NEE available within 7 days
        self._run_two_available(days=7, quality=3)

        # A4: NEE available within |dt| <= 1h on same day
        self._run_mdc(days=0, hours=1, quality=4)

        # B1: same hour NEE available within |dt| <= 1 day
        self._run_mdc(days=1, hours=1, quality=5)

        # B2: SWIN, TA, VPD, NEE available within 21 days
        self._run_all_available(days=21, quality=6)

        # B3: SWIN, TA, VPD, NEE available within 28 days
        self._run_all_available(days=28, quality=7)

        # B4: SWIN, NEE available within 14 days
        self._run_two_available(days=14, quality=8)

        # C+: SWIN, TA, VPD, NEE available within 35-140 days
        quality = 8
        for d in range(35, 147, 7):
            quality += 1
            self._run_all_available(days=d, quality=quality)

        # C+: SWIN, NEE available within 21-140 days
        quality = 24
        for d in range(21, 147, 7):
            quality += 1
            self._run_two_available(days=d, quality=quality)

        # C+: same hour NEE available within |dt| <= 7-X days
        quality = 42
        for d in range(21, 147, 7):
            quality += 1
            self._run_mdc(days=d, hours=1, quality=quality)

        # Gap-filled measurement time series
        self.gapfilling_df_[self.target_gapfilled] = self.gapfilling_df_[self.flux].fillna(
            self.gapfilling_df_['.PREDICTIONS'])

        # Gap-filling flag is 0 where measurement available
        locs_measured_missing = self.gapfilling_df_[self.flux].isnull()
        locs_measured_available = ~locs_measured_missing
        self.gapfilling_df_.loc[locs_measured_available, self.target_gapfilled_flag] = 0

        # Gap-filling flag is equal to prediction quality where measurement was missing
        self.gapfilling_df_.loc[locs_measured_missing, self.target_gapfilled_flag] = \
            self.gapfilling_df_.loc[locs_measured_missing, '.PREDICTIONS_QUALITY']

        # Calculate scores
        scoredf = self.gapfilling_df_[['.PREDICTIONS', self.flux]].copy()
        scoredf = scoredf.dropna()
        self._scores = prediction_scores(predictions=scoredf['.PREDICTIONS'], targets=scoredf[self.flux])

        self._scores['mean_quality_flag_gap_predictions'] = \
            self.gapfilling_df_.loc[locs_measured_missing, self.target_gapfilled_flag].mean()

        print("MDS gap-filling done.")

    def showplot(self):
        """Display MDS gap-filling results with plots."""
        fig = plt.figure(facecolor='white', figsize=(16, 9), dpi=100, layout='constrained')
        gs = gridspec.GridSpec(3, 1, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax_flag = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_counts = fig.add_subplot(gs[2, 0], sharex=ax)
        flag = self.gapfilling_df_[self.target_gapfilled_flag]
        uniqueflags = list(flag.unique())
        uniqueflags.sort()
        colors = colorwheel_36_blackfirst()
        maxcolors = len(colors)
        markers = generate_plot_marker_list()
        maxmarker = len(markers)
        for ix, uf in enumerate(uniqueflags):
            locs = flag == uf
            data = self.gapfilling_df_.loc[locs, :]
            label = f"measured ({self.flux})" if uf == 0 else f"gap-filled quality {uf}, mean ± SD"
            n_vals = data[self.target_gapfilled].count()
            color = colors[35] if ix > (maxcolors - 1) else colors[ix]
            marker = markers[9] if ix > (maxmarker - 1) else markers[ix]
            ax.plot(data.index, data[self.target_gapfilled],
                    label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                    marker=marker, alpha=1, markersize=5, markeredgecolor=color, fillstyle='full')

            if uf > 0:
                ax.errorbar(data.index, data[self.target_gapfilled], data['.PREDICTIONS_SD'],
                            elinewidth=5, ecolor=color, alpha=.2, fmt='none')

                ax_counts.plot(data.index, data['.PREDICTIONS_COUNTS'],
                               label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                               marker=marker, alpha=1, markersize=5, markeredgecolor=color, fillstyle='full')

            ax_flag.plot(data.index, data[self.target_gapfilled_flag],
                         label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                         marker=marker, alpha=1, markersize=5, markeredgecolor=color, fillstyle='full')
        fig.suptitle(f"Variable {self.flux} gap-filled using MDS: {self.target_gapfilled}",
                     fontsize=theme.FIGHEADER_FONTSIZE)
        default_format(ax=ax, ax_ylabel_txt=f"{self.flux}", ax_labels_fontsize=theme.AX_LABELS_FONTSIZE_12)
        ax.tick_params(labelbottom=False)
        ax_flag.tick_params(labelbottom=False)
        default_format(ax=ax_flag, ax_ylabel_txt="flag value", ax_labels_fontsize=theme.AX_LABELS_FONTSIZE_12)
        default_format(ax=ax_counts, ax_ylabel_txt="number of values for mean",
                       ax_labels_fontsize=theme.AX_LABELS_FONTSIZE_12)
        default_legend(ax=ax_flag, textsize=theme.FONTSIZE_TXT_LEGEND_SMALL_9, ncol=2)

        nice_date_ticks(ax=ax)
        nice_date_ticks(ax=ax_flag)
        nice_date_ticks(ax=ax_counts)
        fig.show()

    def report(self):
        """Print comprehensive MDS gap-filling report."""
        potential_vals = len(self.gapfilling_df_.index)
        n_vals_before = self.gapfilling_df_[self.flux].count()
        n_vals_missing_before = self.gapfilling_df_[self.flux].isnull().sum()
        pct_missing_before = 100.0 * n_vals_missing_before / potential_vals

        n_vals_after = self.gapfilling_df_[self.target_gapfilled].count()
        n_vals_missing_after = self.gapfilling_df_[self.target_gapfilled].isnull().sum()
        pct_missing_after = 100.0 * n_vals_missing_after / potential_vals

        n_vals_filled = n_vals_missing_before - n_vals_missing_after
        pct_recovery = 100.0 * n_vals_filled / max(n_vals_missing_before, 1)

        mean_quality = self.gapfilling_df_['.PREDICTIONS_QUALITY'].mean()
        flagcounts = Counter(self.gapfilling_df_[self.target_gapfilled_flag])

        print(f"\n{'Marginal Distribution Sampling (MDS) Gap-Filling Report':=^80}")
        print(f"  Reference: Reichstein et al. (2005) - https://doi.org/10.1111/j.1365-2486.2005.001002.x")

        print(f"\n{'Algorithm Overview':-^80}")
        print(f"  MDS fills gaps using average flux from similar meteorological conditions.")
        print(f"  Hierarchical quality-based approach with progressively relaxed windows:")
        print(f"    • Quality 1-3 (A1-A3):  7-14 days,  all 3 variables (SWIN, TA, VPD)")
        print(f"    • Quality 4-8 (B1-B4):  21-28 days, 2-3 variables")
        print(f"    • Quality 9+  (C+):     35-140+ days, progressively fewer constraints")

        print(f"\n{'Parameters':-^80}")
        print(f"  {'Flux variable':<35} {self.flux}")
        print(f"  {'SWIN tolerance':<35} [{self.swin_tol[0]}, {self.swin_tol[1]}] W m-2")
        print(f"  {'TA tolerance':<35} {self.ta_tol} °C")
        print(f"  {'VPD tolerance':<35} {self.vpd_tol} kPa")
        print(f"  {'Min records for average':<35} {self.avg_min_n_vals}")

        print(f"\n{'Input Data Summary':-^80}")
        print(f"  {'Total records':<35} {potential_vals:>15,d}")
        print(f"  {'Available measurements':<35} {n_vals_before:>15,d}  ({100.0*n_vals_before/potential_vals:>6.2f}%)")
        print(f"  {'Missing values':<35} {n_vals_missing_before:>15,d}  ({pct_missing_before:>6.2f}%)")

        print(f"\n{'Gap-Filling Performance':-^80}")
        print(f"  {'Values filled':<35} {n_vals_filled:>15,d}  ({pct_recovery:>6.2f}% of gaps)")
        print(f"  {'Remaining missing':<35} {n_vals_missing_after:>15,d}  ({pct_missing_after:>6.2f}% of total)")
        print(f"  {'Final data coverage':<35} {n_vals_after:>15,d}  ({100.0*n_vals_after/potential_vals:>6.2f}%)")
        gap_recovery_efficiency = pct_recovery if pct_recovery > 0 else 0
        print(f"  {'Gap recovery efficiency':<35} {gap_recovery_efficiency:>15.2f}%")
        print(f"  {'Mean quality score':<35} {mean_quality:>15.3f}  (1=best, 9+=low)")

        print(f"\n{'Quality Distribution':-^80}")
        measured_count = flagcounts.get(0, 0)
        print(f"  {'Measured (original data)':<35} {measured_count:>15,d}  ({100.0*measured_count/potential_vals:>6.2f}%)")

        quality_counts = {k: v for k, v in sorted(flagcounts.items()) if k > 0}
        if quality_counts:
            quality_groups = {
                'High quality (A1-A3, 1-3)': sum(v for k, v in quality_counts.items() if 1 <= k <= 3),
                'Medium quality (B1-B4, 4-8)': sum(v for k, v in quality_counts.items() if 4 <= k <= 8),
                'Low quality (C+, 9+)': sum(v for k, v in quality_counts.items() if k >= 9),
            }
            for group_name, count in quality_groups.items():
                if count > 0:
                    pct = 100.0 * count / potential_vals
                    print(f"  {group_name:<35} {count:>15,d}  ({pct:>6.2f}%)")

            print(f"\n  Quality level breakdown (detailed):")
            print(f"    {'Level':<10} {'Count':>12} {'Percentage':>12} {'Description':<45}")
            print(f"    {'-'*10} {'-'*12} {'-'*12} {'-'*45}")
            quality_descriptions = {
                1: 'SWIN, TA, VPD: 7 days',
                2: 'SWIN, TA, VPD: 14 days',
                3: 'SWIN only: 7 days',
                4: 'NEE only: 1 hour same day',
                5: 'NEE only: 1 hour within 1 day',
                6: 'SWIN, TA, VPD: 21 days',
                7: 'SWIN, TA, VPD: 28 days',
                8: 'SWIN only: 14 days',
            }
            for quality, count in sorted(quality_counts.items()):
                pct = 100.0 * count / potential_vals
                if quality in quality_descriptions:
                    desc = quality_descriptions[quality]
                else:
                    if 9 <= quality <= 24:
                        days = 35 + (quality - 9) * 7
                        desc = f'SWIN, TA, VPD: {days} days'
                    elif 25 <= quality <= 40:
                        days = 21 + (quality - 25) * 7
                        desc = f'SWIN only: {days} days'
                    elif quality >= 41:
                        days = 21 + (quality - 41) * 7
                        desc = f'NEE only: {days} days'
                    else:
                        desc = 'Unknown'
                print(f"    {quality:<10} {count:>12,d} {pct:>11.2f}% {desc:<45}")

        self.report_scores()
        print(f"{'':=^80}\n")

    def report_scores(self):
        """Print model performance scores with interpretation."""
        print(f"\n{'Model Performance Scores':-^80}")
        if self.scores_:
            for score, val in self.scores_.items():
                score_display = score.replace('_', ' ').title()
                print(f"  {score_display:<35} {val:>15.4f}")
        else:
            print("  No scores available")
        print(f"\n{'Key Insights':-^80}")
        print(f"  • Higher quality levels (e.g., 1-3) indicate more similar meteorological conditions")
        print(f"  • Mean quality score closer to 1 = more gap-filled values from strict criteria")
        print(f"  • Check 'Final data coverage' to assess overall gap-filling success")
        print(f"  • Review 'Quality Distribution' to understand data source reliability")

    def _fill_gap_predictions(self, gap_indices, predictions, sds, counts, quality):
        """Fill gap predictions into main dataframe using index locations.

        Directly updates self._gapfilling_df and self._missing_mask without copying.
        """
        valid_mask = ~np.isnan(predictions)

        if not np.any(valid_mask):
            return

        # Get the integer positions from gap_indices where predictions are valid
        filled_positions = gap_indices[valid_mask]

        # Convert integer positions to index labels for direct assignment
        filled_index_labels = self._gapfilling_df.index[filled_positions]

        self._gapfilling_df.loc[filled_index_labels, '.PREDICTIONS'] = predictions[valid_mask]
        self._gapfilling_df.loc[filled_index_labels, '.PREDICTIONS_SD'] = sds[valid_mask]
        self._gapfilling_df.loc[filled_index_labels, '.PREDICTIONS_COUNTS'] = counts[valid_mask].astype('float64')
        self._gapfilling_df.loc[filled_index_labels, '.PREDICTIONS_QUALITY'] = float(quality) if isinstance(quality, (int, np.integer)) else quality

        # Update missing mask: remove filled positions from remaining gaps
        self._missing_mask.iloc[filled_positions] = False

    def _add_newcols(self) -> pd.DataFrame:
        """Initialize new columns for gap-filling tracking."""
        df = self.gapfilling_df_.copy()
        df['.TIMESTAMP'] = df.index
        df[self.target_gapfilled] = np.nan
        df[self.target_gapfilled_flag] = np.nan
        df['.PREDICTIONS'] = np.nan
        df['.PREDICTIONS_SD'] = np.nan
        df['.PREDICTIONS_COUNTS'] = np.nan
        df['.PREDICTIONS_QUALITY'] = np.nan
        df['.START'] = np.nan
        df['.END'] = np.nan
        df[f'.{self.swin}_LOWERLIM'] = np.nan
        df[f'.{self.swin}_UPPERLIM'] = np.nan
        df[f'.{self.ta}_LOWERLIM'] = np.nan
        df[f'.{self.ta}_UPPERLIM'] = np.nan
        df[f'.{self.vpd}_LOWERLIM'] = np.nan
        df[f'.{self.vpd}_UPPERLIM'] = np.nan

        lowrad = df[self.swin] <= 50
        df.loc[lowrad, f'.{self.swin}_LOWERLIM'] = df.loc[lowrad, self.swin].sub(self.swin_tol[0])
        df.loc[lowrad, f'.{self.swin}_UPPERLIM'] = df.loc[lowrad, self.swin].add(self.swin_tol[0])

        highrad = df[self.swin] > 50
        df.loc[highrad, f'.{self.swin}_LOWERLIM'] = df.loc[highrad, self.swin].sub(self.swin_tol[1])
        df.loc[highrad, f'.{self.swin}_UPPERLIM'] = df.loc[highrad, self.swin].add(self.swin_tol[1])

        df[f'.{self.ta}_LOWERLIM'] = df[self.ta].sub(self.ta_tol)
        df[f'.{self.ta}_UPPERLIM'] = df[self.ta].add(self.ta_tol)
        df[f'.{self.vpd}_LOWERLIM'] = df[self.vpd].sub(self.vpd_tol)
        df[f'.{self.vpd}_UPPERLIM'] = df[self.vpd].add(self.vpd_tol)
        return df

    def _run_all_available(self, days: int, quality: int):
        """Optimized version using vectorized predictions (all 3 variables)."""
        # Get indices of gaps that still need filling
        gap_indices = np.where(self._missing_mask)[0]
        if len(gap_indices) == 0:
            return

        print(f"MDS gap-filling quality {quality}    using SW_IN, TA, VPD in {days} days window ...")

        # Build start/end times for each gap
        gap_timestamps = self._gapfilling_df.iloc[gap_indices].index
        offset = pd.DateOffset(days=days)
        start_times = gap_timestamps - offset
        end_times = gap_timestamps + offset

        # Vectorized prediction calculation
        predictions, sds, counts = self._vectorized_predictions_all_available(
            gap_indices, start_times, end_times
        )

        # Fill predictions directly into main dataframe
        self._fill_gap_predictions(gap_indices, predictions, sds, counts, quality)

    def _run_two_available(self, days: int, quality: int):
        """Optimized version using vectorized predictions (SWIN only)."""
        # Get indices of gaps that still need filling
        gap_indices = np.where(self._missing_mask)[0]
        if len(gap_indices) == 0:
            return

        print(f"MDS gap-filling quality {quality}    using SW_IN in {days} days window ...")

        # Build start/end times for each gap
        gap_timestamps = self._gapfilling_df.iloc[gap_indices].index
        offset = pd.DateOffset(days=days)
        start_times = gap_timestamps - offset
        end_times = gap_timestamps + offset

        # Vectorized prediction calculation
        predictions, sds, counts = self._vectorized_predictions_two_available(
            gap_indices, start_times, end_times
        )

        # Fill predictions directly into main dataframe
        self._fill_gap_predictions(gap_indices, predictions, sds, counts, quality)

    def _run_mdc(self, days: int, hours: int, quality: int):
        """Optimized version using vectorized mean diurnal cycle predictions."""
        # Get indices of gaps that still need filling
        gap_indices = np.where(self._missing_mask)[0]
        if len(gap_indices) == 0:
            return

        print(f"MDS gap-filling quality {quality}    using mean diurnal cycle of flux in "
              f"{days} days, {hours} hours window ...")

        # Build start/end times for each gap
        gap_timestamps = self._gapfilling_df.iloc[gap_indices].index
        offset = pd.DateOffset(days=days, hours=hours)
        start_times = gap_timestamps - offset
        end_times = gap_timestamps + offset

        # Vectorized prediction calculation
        if days == 0:
            predictions, sds, counts = self._vectorized_predictions_mdc_sameday(
                gap_indices, start_times, end_times
            )
        else:
            predictions, sds, counts = self._vectorized_predictions_mdc_multiday(
                gap_indices, start_times, end_times
            )

        # Fill predictions directly into main dataframe
        self._fill_gap_predictions(gap_indices, predictions, sds, counts, quality)

    def _vectorized_predictions_all_available(self, gap_indices, start_times, end_times):
        """Calculate predictions for gaps using vectorized operations (3 variables).

        Uses searchsorted() for O(log n) time-window lookups instead of O(n) comparisons.
        Returns arrays of predictions, standard deviations, and counts.
        """
        n_gaps = len(gap_indices)
        predictions = np.full(n_gaps, np.nan)
        sds = np.full(n_gaps, np.nan)
        counts = np.zeros(n_gaps, dtype=int)

        # Pre-extract arrays for faster access
        gf_index_values = self.gapfilling_df_.index.values
        gf_flux = self.gapfilling_df_[self.flux].values
        gf_ta = self.gapfilling_df_[self.ta].values
        gf_swin = self.gapfilling_df_[self.swin].values
        gf_vpd = self.gapfilling_df_[self.vpd].values

        # Process each gap row
        for i in range(n_gaps):
            gap_idx = gap_indices[i]
            start = np.datetime64(start_times[i])
            end = np.datetime64(end_times[i])
            gap_ta = self._gapfilling_df.iloc[gap_idx][self.ta]
            gap_swin = self._gapfilling_df.iloc[gap_idx][self.swin]
            gap_vpd = self._gapfilling_df.iloc[gap_idx][self.vpd]

            # Find time window using searchsorted (O(log n))
            start_idx = gf_index_values.searchsorted(start, side='left')
            end_idx = gf_index_values.searchsorted(end, side='right')

            # Extract data within time window
            window_flux = gf_flux[start_idx:end_idx]
            window_ta = gf_ta[start_idx:end_idx]
            window_swin = gf_swin[start_idx:end_idx]
            window_vpd = gf_vpd[start_idx:end_idx]

            # Apply meteorological similarity conditions
            # Check if window data is within tolerance of gap row values
            # SWIN tolerance depends on whether the RECORD's SWIN is low or high radiation
            lowrad_mask = window_swin <= 50
            swin_tol_window = np.where(lowrad_mask, self.swin_tol[0], self.swin_tol[1])

            locs = (
                    (window_ta < gap_ta + self.ta_tol)
                    & (window_ta > gap_ta - self.ta_tol)
                    & (window_swin < gap_swin + swin_tol_window)
                    & (window_swin > gap_swin - swin_tol_window)
                    & (window_vpd < gap_vpd + self.vpd_tol)
                    & (window_vpd > gap_vpd - self.vpd_tol)
            )

            matching_flux = window_flux[locs]
            valid_flux = matching_flux[~np.isnan(matching_flux)]

            if len(valid_flux) >= self.avg_min_n_vals:
                predictions[i] = np.mean(valid_flux)
                sds[i] = np.std(valid_flux)
                counts[i] = len(valid_flux)

        return predictions, sds, counts

    def _vectorized_predictions_two_available(self, gap_indices, start_times, end_times):
        """Calculate predictions for gaps using vectorized operations (SWIN only).

        Uses searchsorted() for O(log n) time-window lookups.
        """
        n_gaps = len(gap_indices)
        predictions = np.full(n_gaps, np.nan)
        sds = np.full(n_gaps, np.nan)
        counts = np.zeros(n_gaps, dtype=int)

        # Pre-extract arrays
        gf_index_values = self.gapfilling_df_.index.values
        gf_flux = self.gapfilling_df_[self.flux].values
        gf_swin = self.gapfilling_df_[self.swin].values

        # Process each gap row
        for i in range(n_gaps):
            gap_idx = gap_indices[i]
            start = np.datetime64(start_times[i])
            end = np.datetime64(end_times[i])
            gap_swin = self._gapfilling_df.iloc[gap_idx][self.swin]

            # Find time window using searchsorted (O(log n))
            start_idx = gf_index_values.searchsorted(start, side='left')
            end_idx = gf_index_values.searchsorted(end, side='right')

            # Extract data within time window
            window_flux = gf_flux[start_idx:end_idx]
            window_swin = gf_swin[start_idx:end_idx]

            # Apply SWIN similarity condition only
            # SWIN tolerance depends on whether the RECORD's SWIN is low or high radiation
            lowrad_mask = window_swin <= 50
            swin_tol_window = np.where(lowrad_mask, self.swin_tol[0], self.swin_tol[1])
            locs = (
                    (window_swin < gap_swin + swin_tol_window)
                    & (window_swin > gap_swin - swin_tol_window)
            )

            matching_flux = window_flux[locs]
            valid_flux = matching_flux[~np.isnan(matching_flux)]

            if len(valid_flux) >= self.avg_min_n_vals:
                predictions[i] = np.mean(valid_flux)
                sds[i] = np.std(valid_flux)
                counts[i] = len(valid_flux)

        return predictions, sds, counts

    def _vectorized_predictions_mdc_sameday(self, gap_indices, start_times, end_times):
        """Calculate predictions for MDC same-day using vectorized operations.

        Uses searchsorted() for O(log n) time-window lookups.
        """
        n_gaps = len(gap_indices)
        predictions = np.full(n_gaps, np.nan)
        sds = np.full(n_gaps, np.nan)
        counts = np.zeros(n_gaps, dtype=int)

        # Pre-extract arrays
        gf_index_values = self.gapfilling_df_.index.values
        gf_flux = self.gapfilling_df_[self.flux].values

        # Process each gap row (no meteorological conditions, just time window)
        for i in range(n_gaps):
            start = np.datetime64(start_times[i])
            end = np.datetime64(end_times[i])

            # Find time window using searchsorted (O(log n))
            start_idx = gf_index_values.searchsorted(start, side='left')
            end_idx = gf_index_values.searchsorted(end, side='right')

            # Extract flux data within time window
            window_flux = gf_flux[start_idx:end_idx]
            valid_flux = window_flux[~np.isnan(window_flux)]

            if len(valid_flux) >= self.avg_min_n_vals:
                predictions[i] = np.mean(valid_flux)
                sds[i] = np.std(valid_flux)
                counts[i] = len(valid_flux)

        return predictions, sds, counts

    def _vectorized_predictions_mdc_multiday(self, gap_indices, start_times, end_times):
        """Calculate predictions for MDC multi-day using vectorized operations.

        Same hour + time window.
        Uses searchsorted() for O(log n) time-window lookups.
        """
        n_gaps = len(gap_indices)
        predictions = np.full(n_gaps, np.nan)
        sds = np.full(n_gaps, np.nan)
        counts = np.zeros(n_gaps, dtype=int)

        # Pre-extract arrays
        gf_index = self.gapfilling_df_.index
        gf_index_values = gf_index.values
        gf_flux = self.gapfilling_df_[self.flux].values
        gf_hours = gf_index.hour.values

        # Process each gap row
        for i in range(n_gaps):
            gap_idx = gap_indices[i]
            start = np.datetime64(start_times[i])
            end = np.datetime64(end_times[i])
            row_hour = self._gapfilling_df.index[gap_idx].hour

            # Find time window using searchsorted (O(log n))
            start_idx = gf_index_values.searchsorted(start, side='left')
            end_idx = gf_index_values.searchsorted(end, side='right')

            # Extract data and apply hour filter
            window_flux = gf_flux[start_idx:end_idx]
            window_hours = gf_hours[start_idx:end_idx]

            # Filter by hour and valid flux values
            hour_mask = window_hours == row_hour
            matching_flux = window_flux[hour_mask]
            valid_flux = matching_flux[~np.isnan(matching_flux)]

            if len(valid_flux) >= self.avg_min_n_vals:
                predictions[i] = np.mean(valid_flux)
                sds[i] = np.std(valid_flux)
                counts[i] = len(valid_flux)

        return predictions, sds, counts

# See examples/pkgs/gapfilling/gapfill_mds.py for usage examples.
# For the original unoptimized implementation, use _FluxMDS.
