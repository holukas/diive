"""
GAP-FILLING: MARGINAL DISTRIBUTION SAMPLING
============================================

Statistical gap-filling using meteorological similarity.
No training required; based on Reichstein et al (2005).

Part of the diive library: https://github.com/holukas/diive
"""
from collections import Counter

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.ml.results import GapFillingResult
from diive.core.plotting.plotfuncs import default_format, default_legend
from diive.core.plotting.plotfuncs import nice_date_ticks
from diive.core.plotting.styles.LightTheme import colorwheel_36_blackfirst, generate_plot_marker_list
from diive.core.utils.console import console as _console, detail, info, rule
from diive.gapfilling.scores import prediction_scores


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
        _required = {
            flux: 'flux (no unit restriction)',
            swin: 'SWIN - short-wave incoming radiation (W m-2)',
            ta: 'TA - air temperature (deg C)',
            vpd: 'VPD - vapor pressure deficit (kPa)',
        }
        _missing = [col for col in _required if col not in df.columns]
        if _missing:
            _msgs = [f"  '{col}': {_required[col]}" for col in _missing]
            raise KeyError(
                "Column(s) not found in df - MDS requires flux, SWIN (W m-2), TA (deg C), VPD (kPa):\n"
                + "\n".join(_msgs)
            )

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
    def result(self) -> DataFrame:
        """Primary result: full gap-filling DataFrame (target + flag columns)."""
        return self.gapfilling_df_

    @property
    def results(self) -> GapFillingResult:
        """Structured result after .run() — all outputs in one object.

        Returns a :class:`~diive.core.ml.results.GapFillingResult` with:
        ``gapfilled``, ``flag``, ``scores``, ``gapfilling_df``.
        ML-only fields (``scores_traintest``, ``feature_importances``, ``model``) are None.

        Raises:
            Exception: if called before :meth:`run`.
        """
        if not self._scores:
            raise Exception("Results not available: call .run() first.")
        return GapFillingResult(
            gapfilled=self.get_gapfilled_target(),
            flag=self.get_flag(),
            scores=self._scores,
            gapfilling_df=self._gapfilling_df,
        )

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

        rule(f"MDS Gap-Filling: {self.flux}")

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

        info(f"MDS gap-filling done: {self.flux}")

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

        from rich.table import Table

        rule(f"MDS Gap-Filling Report: {self.flux}")
        _console.print(
            "  Reference: Reichstein et al. (2005) — "
            "https://doi.org/10.1111/j.1365-2486.2005.001002.x\n"
            "\n"
            "  [bold]Algorithm:[/bold] hierarchical similarity matching on SWIN / TA / VPD\n"
            "    Quality 1-3 (A1-A3):  7-14 days  —  all 3 variables\n"
            "    Quality 4-8 (B1-B4):  21-28 days —  2-3 variables\n"
            "    Quality 9+  (C+):     35-140+ days — progressively fewer constraints"
        )

        rule("Parameters", min_level=2)
        _console.print(
            f"  Flux variable          {self.flux}\n"
            f"  SWIN tolerance         [{self.swin_tol[0]}, {self.swin_tol[1]}] W m-2\n"
            f"  TA tolerance           {self.ta_tol} deg C\n"
            f"  VPD tolerance          {self.vpd_tol} kPa\n"
            f"  Min records for mean   {self.avg_min_n_vals}"
        )

        rule("Data & Performance", min_level=2)
        _console.print(
            f"  Total records          {potential_vals:>12,d}\n"
            f"  Available before       {n_vals_before:>12,d}  ({100.0 * n_vals_before / potential_vals:.1f}%)\n"
            f"  Missing before         {n_vals_missing_before:>12,d}  ({pct_missing_before:.1f}%)\n"
            f"  Filled                 {n_vals_filled:>12,d}  ({pct_recovery:.1f}% of gaps)\n"
            f"  Remaining missing      {n_vals_missing_after:>12,d}  ({pct_missing_after:.1f}% of total)\n"
            f"  Final coverage         {n_vals_after:>12,d}  ({100.0 * n_vals_after / potential_vals:.1f}%)\n"
            f"  Mean quality score     {mean_quality:>12.3f}  (1=best, 9+=low)"
        )

        quality_counts = {k: v for k, v in sorted(flagcounts.items()) if k > 0}
        measured_count = flagcounts.get(0, 0)
        if quality_counts:
            rule("Quality Distribution", min_level=2)
            table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
            table.add_column("Level", style="dim", no_wrap=True)
            table.add_column("Count", justify="right")
            table.add_column("  %", justify="right")
            table.add_column("Description")
            table.add_row("0 (observed)", f"{measured_count:,d}",
                          f"{100.0 * measured_count / potential_vals:.1f}%", "measured")
            for q, cnt in sorted(quality_counts.items()):
                pct = 100.0 * cnt / potential_vals
                table.add_row(str(q), f"{cnt:,d}", f"{pct:.1f}%", mds_quality_description(q))
            _console.print(table)

        self.report_scores()

    def report_scores(self):
        """Print model performance scores with interpretation."""
        rule("Model Performance Scores", min_level=2)
        if self.scores_:
            for score, val in self.scores_.items():
                score_display = score.replace('_', ' ').title()
                _console.print(f"  {score_display:<35} {val:.4f}")
        else:
            _console.print("  No scores available")
        _console.print(
            "\n  [dim]Higher quality levels (1-3) = stricter meteorological match.[/dim]\n"
            "  [dim]Mean quality near 1 = most gaps filled under tight constraints.[/dim]"
        )

    def _run_all_available(self, days: int, quality: int):

        _df, workdf = self._prepare_dataframes()
        if workdf.empty:
            return workdf, _df

        detail(f"MDS Q{quality}: SWIN+TA+VPD, {days}d window")

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

        detail(f"MDS Q{quality}: SWIN only, {days}d window")

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

        detail(f"MDS Q{quality}: MDC, {days}d {hours}h window")

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


#: MDS quality-level descriptions for the fixed A1-B4 tiers (Reichstein 2005).
_MDS_QUALITY_FIXED = {
    1: 'SWIN, TA, VPD: 7 days',
    2: 'SWIN, TA, VPD: 14 days',
    3: 'SWIN only: 7 days',
    4: 'Diurnal cycle: 1 h same day',
    5: 'Diurnal cycle: 1 h within 1 day',
    6: 'SWIN, TA, VPD: 21 days',
    7: 'SWIN, TA, VPD: 28 days',
    8: 'SWIN only: 14 days',
}


def mds_quality_description(quality: int) -> str:
    """Human description of an MDS gap-filling quality level (flag value).

    0 = measured; 1-8 = the fixed A1-B4 hierarchy; 9+ = the progressively relaxed
    C+ tiers (Reichstein et al. 2005). Higher levels relax the meteorological match.
    """
    q = int(quality)
    if q == 0:
        return 'measured'
    if q in _MDS_QUALITY_FIXED:
        return _MDS_QUALITY_FIXED[q]
    if 9 <= q <= 24:
        return f'SWIN, TA, VPD: {35 + (q - 9) * 7} days'
    if 25 <= q <= 40:
        return f'SWIN only: {21 + (q - 25) * 7} days'
    if q >= 41:
        return f'Diurnal cycle: {21 + (q - 41) * 7} days'
    return 'unknown'


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
        See examples/gapfilling/gapfill_mds.py for basic usage.
        See examples/gapfilling/gapfill_comparison.py for side-by-side comparison with
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
        _required = {
            flux: 'flux (no unit restriction)',
            swin: 'SWIN - short-wave incoming radiation (W m-2)',
            ta: 'TA - air temperature (deg C)',
            vpd: 'VPD - vapor pressure deficit (kPa)',
        }
        _missing = [col for col in _required if col not in df.columns]
        if _missing:
            _msgs = [f"  '{col}': {_required[col]}" for col in _missing]
            raise KeyError(
                "Column(s) not found in df - MDS requires flux, SWIN (W m-2), TA (deg C), VPD (kPa):\n"
                + "\n".join(_msgs)
            )

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
    def result(self) -> DataFrame:
        """Primary result: full gap-filling DataFrame (target + flag columns)."""
        return self.gapfilling_df_

    @property
    def results(self) -> GapFillingResult:
        """Structured result after .run() — all outputs in one object.

        Returns a :class:`~diive.core.ml.results.GapFillingResult` with:
        ``gapfilled``, ``flag``, ``scores``, ``gapfilling_df``.
        ML-only fields (``scores_traintest``, ``feature_importances``, ``model``) are None.

        Raises:
            Exception: if called before :meth:`run`.
        """
        if not self._scores:
            raise Exception("Results not available: call .run() first.")
        return GapFillingResult(
            gapfilled=self.get_gapfilled_target(),
            flag=self.get_flag(),
            scores=self._scores,
            gapfilling_df=self._gapfilling_df,
        )

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

    def quality_breakdown(self) -> DataFrame:
        """Per-quality-level breakdown of the gap-filling flag.

        One row per flag value present, sorted ascending: level 0 = measured,
        1+ = gap-filled at that quality tier (higher = looser meteorological
        match). Columns: ``level``, ``count``, ``pct`` (% of all records),
        ``description``. This is the data behind the console report's Quality
        Distribution table and the GUI quality-level plot.
        """
        flag = self.gapfilling_df_[self.target_gapfilled_flag]
        total = len(flag)
        counts = flag.value_counts().sort_index()
        rows = [
            {'level': int(level), 'count': int(count),
             'pct': 100.0 * count / total if total else 0.0,
             'description': mds_quality_description(int(level))}
            for level, count in counts.items()
        ]
        return DataFrame(rows, columns=['level', 'count', 'pct', 'description'])

    @staticmethod
    def _level_plan() -> list:
        """The ordered MDS quality hierarchy as ``(kind, days, hours, quality)``
        steps. ``kind`` is ``'all'`` (SWIN+TA+VPD), ``'two'`` (SWIN only) or
        ``'mdc'`` (mean diurnal cycle). Driving ``run`` from this single list lets
        it report per-level progress without duplicating the sequence."""
        plan = [
            ('all', 7, 0, 1),    # A1: SWIN, TA, VPD within 7 days (highest quality)
            ('all', 14, 0, 2),   # A2: SWIN, TA, VPD within 14 days
            ('two', 7, 0, 3),    # A3: SWIN only within 7 days
            ('mdc', 0, 1, 4),    # A4: same hour within |dt| <= 1h on same day
            ('mdc', 1, 1, 5),    # B1: same hour within |dt| <= 1 day
            ('all', 21, 0, 6),   # B2: SWIN, TA, VPD within 21 days
            ('all', 28, 0, 7),   # B3: SWIN, TA, VPD within 28 days
            ('two', 14, 0, 8),   # B4: SWIN only within 14 days
        ]
        quality = 8                                  # C+: SWIN, TA, VPD, 35-140 days
        for d in range(35, 147, 7):
            quality += 1
            plan.append(('all', d, 0, quality))
        quality = 24                                 # C+: SWIN only, 21-140 days
        for d in range(21, 147, 7):
            quality += 1
            plan.append(('two', d, 0, quality))
        quality = 42                                 # C+: same hour, 21-140 days
        for d in range(21, 147, 7):
            quality += 1
            plan.append(('mdc', d, 1, quality))
        return plan

    def _emit_subprogress(self, i: int, n_gaps: int, predictions) -> None:
        """Mid-level progress tick from the per-gap loops, so the bar advances
        *within* a long quality level instead of only at its end.

        Progress is measured in **gaps**: already-filled gaps from prior levels
        plus the valid predictions computed so far in this level (those gaps will
        be filled when the level finishes). So the percentage reflects how many
        gaps are (about to be) filled, not how many quality levels have run."""
        cb = getattr(self, '_progress_callback', None)
        if cb is None or self._initial_gaps <= 0:
            return
        will_fill = int(np.count_nonzero(~np.isnan(predictions[:i])))
        filled = self._filled_before_level + will_fill
        remaining = self._initial_gaps - filled
        cb(filled, self._initial_gaps, self._cur_quality, 0, remaining)

    def run(self, progress_callback=None):
        """Execute optimized MDS gap-filling algorithm.

        Args:
            progress_callback: optional ``callable(filled, total, quality, n_filled,
                n_remaining)`` for a GUI progress bar. ``filled``/``total`` count
                **gaps** (filled so far / total gaps to fill), so the percentage is
                gap-based; ``quality`` is the current quality level.
        """
        rule(f"MDS Gap-Filling: {self.flux}")

        locs_missing = self.gapfilling_df_['.PREDICTIONS'].isnull()
        self._missing_mask = locs_missing.copy()

        runners = {'all': self._run_all_available, 'two': self._run_two_available}
        plan = self._level_plan()
        # Progress state shared with the per-gap loops (see _emit_subprogress).
        # Everything is measured in gaps: total = the gaps present at the start.
        self._progress_callback = progress_callback
        self._initial_gaps = int(self._missing_mask.sum())
        for kind, days, hours, quality in plan:
            before = int(self._missing_mask.sum())
            self._cur_quality = quality
            self._filled_before_level = self._initial_gaps - before
            if kind == 'mdc':
                self._run_mdc(days=days, hours=hours, quality=quality)
            else:
                runners[kind](days=days, quality=quality)
            remaining = int(self._missing_mask.sum())
            n_filled = before - remaining
            # Log only the levels that actually filled something, at the caller's
            # verbosity (hidden at the default verbose=1; shown at >=2, e.g. GUI).
            if n_filled > 0:
                info(f"Quality {quality} ({mds_quality_description(quality)}): "
                     f"filled {n_filled:,} gaps, {remaining:,} remaining",
                     verbose=self.verbose)
            if progress_callback is not None:
                progress_callback(self._initial_gaps - remaining, self._initial_gaps,
                                  quality, n_filled, remaining)
            if remaining == 0:
                break
        self._progress_callback = None

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

        info(f"MDS gap-filling done: {self.flux}")

    def _flag_style(self, ix: int):
        """(colour, marker) for the ix-th sorted quality level — wraps when the
        record has more levels than the palette/marker list."""
        colors = colorwheel_36_blackfirst()
        markers = generate_plot_marker_list()
        color = colors[35] if ix > (len(colors) - 1) else colors[ix]
        marker = markers[9] if ix > (len(markers) - 1) else markers[ix]
        return color, marker

    def plot_quality_timeseries(self, ax=None, legend: bool = True,
                                ax_labels_fontsize: int = None,
                                legend_textsize: int = None,
                                legend_ncol: int = None):
        """Plot the gap-filled series over time, each point coloured + markered by
        its MDS quality level (0 = measured, 1+ = gap-filled; mean ± SD whiskers on
        the filled points).

        Two-phase: ``ax=None`` builds a standalone figure; pass an ``ax`` (e.g. a
        GUI canvas axes) to embed it. Returns the axes used.

        The quality hierarchy has many levels, so the legend is placed *below* the
        axes across several columns (``legend_ncol``, auto-sized for ~6 rows when
        ``None``) — otherwise it overlaps and is clipped.
        """
        if ax_labels_fontsize is None:
            ax_labels_fontsize = theme.AX_LABELS_FONTSIZE_12
        if legend_textsize is None:
            legend_textsize = theme.FONTSIZE_TXT_LEGEND_SMALL_9
        if ax is None:
            fig = plt.figure(facecolor='white', figsize=(16, 5), dpi=100, layout='constrained')
            ax = fig.add_subplot(1, 1, 1)
        flag = self.gapfilling_df_[self.target_gapfilled_flag]
        levels = sorted(flag.unique())
        for ix, uf in enumerate(levels):
            data = self.gapfilling_df_.loc[flag == uf, :]
            n_vals = data[self.target_gapfilled].count()
            label = (f"measured ({self.flux})" if uf == 0
                     else f"gap-filled quality {int(uf)}, mean ± SD")
            color, marker = self._flag_style(ix)
            ax.plot(data.index, data[self.target_gapfilled],
                    label=f"{label} ({n_vals} values)", color=color, linestyle='none',
                    markeredgewidth=1, marker=marker, alpha=1, markersize=5,
                    markeredgecolor=color, fillstyle='full')
            if uf > 0:
                ax.errorbar(data.index, data[self.target_gapfilled], data['.PREDICTIONS_SD'],
                            elinewidth=5, ecolor=color, alpha=.2, fmt='none')
        # ax_xlabel_txt='' avoids default_format's `False` default rendering a
        # literal "False" x-axis label (the axis is dates, no label needed).
        default_format(ax=ax, ax_xlabel_txt='', ax_ylabel_txt=f"{self.flux}",
                       ax_labels_fontsize=ax_labels_fontsize)
        if legend:
            # Auto-size columns for ~6 rows so the many quality levels stay
            # readable; place the legend below the axes so it never covers data.
            ncol = legend_ncol or max(2, min(10, -(-len(levels) // 6)))
            default_legend(ax=ax, textsize=legend_textsize, ncol=ncol,
                           loc='upper center', bbox_to_anchor=(0.5, -0.12))
        nice_date_ticks(ax=ax)
        return ax

    def showplot(self):
        """Display MDS gap-filling results with plots."""
        fig = plt.figure(facecolor='white', figsize=(16, 9), dpi=100, layout='constrained')
        gs = gridspec.GridSpec(3, 1, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax_flag = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_counts = fig.add_subplot(gs[2, 0], sharex=ax)
        # Top panel: the colour/marker-by-quality time series (shared with the GUI).
        self.plot_quality_timeseries(ax=ax, legend=False)
        flag = self.gapfilling_df_[self.target_gapfilled_flag]
        for ix, uf in enumerate(sorted(flag.unique())):
            data = self.gapfilling_df_.loc[flag == uf, :]
            label = f"measured ({self.flux})" if uf == 0 else f"gap-filled quality {uf}, mean ± SD"
            n_vals = data[self.target_gapfilled].count()
            color, marker = self._flag_style(ix)
            if uf > 0:
                ax_counts.plot(data.index, data['.PREDICTIONS_COUNTS'],
                               label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                               marker=marker, alpha=1, markersize=5, markeredgecolor=color, fillstyle='full')

            ax_flag.plot(data.index, data[self.target_gapfilled_flag],
                         label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                         marker=marker, alpha=1, markersize=5, markeredgecolor=color, fillstyle='full')
        fig.suptitle(f"Variable {self.flux} gap-filled using MDS: {self.target_gapfilled}",
                     fontsize=theme.FIGHEADER_FONTSIZE)
        ax.tick_params(labelbottom=False)
        ax_flag.tick_params(labelbottom=False)
        default_format(ax=ax_flag, ax_ylabel_txt="flag value", ax_labels_fontsize=theme.AX_LABELS_FONTSIZE_12)
        default_format(ax=ax_counts, ax_ylabel_txt="number of values for mean",
                       ax_labels_fontsize=theme.AX_LABELS_FONTSIZE_12)
        default_legend(ax=ax_flag, textsize=theme.FONTSIZE_TXT_LEGEND_SMALL_9, ncol=2)

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

        from rich.table import Table

        rule(f"MDS Gap-Filling Report: {self.flux}")
        _console.print(
            "  Reference: Reichstein et al. (2005) — "
            "https://doi.org/10.1111/j.1365-2486.2005.001002.x\n"
            "\n"
            "  [bold]Algorithm:[/bold] hierarchical similarity matching on SWIN / TA / VPD\n"
            "    Quality 1-3 (A1-A3):  7-14 days  —  all 3 variables\n"
            "    Quality 4-8 (B1-B4):  21-28 days —  2-3 variables\n"
            "    Quality 9+  (C+):     35-140+ days — progressively fewer constraints"
        )

        rule("Parameters", min_level=2)
        _console.print(
            f"  Flux variable          {self.flux}\n"
            f"  SWIN tolerance         [{self.swin_tol[0]}, {self.swin_tol[1]}] W m-2\n"
            f"  TA tolerance           {self.ta_tol} deg C\n"
            f"  VPD tolerance          {self.vpd_tol} kPa\n"
            f"  Min records for mean   {self.avg_min_n_vals}"
        )

        rule("Data & Performance", min_level=2)
        _console.print(
            f"  Total records          {potential_vals:>12,d}\n"
            f"  Available before       {n_vals_before:>12,d}  ({100.0 * n_vals_before / potential_vals:.1f}%)\n"
            f"  Missing before         {n_vals_missing_before:>12,d}  ({pct_missing_before:.1f}%)\n"
            f"  Filled                 {n_vals_filled:>12,d}  ({pct_recovery:.1f}% of gaps)\n"
            f"  Remaining missing      {n_vals_missing_after:>12,d}  ({pct_missing_after:.1f}% of total)\n"
            f"  Final coverage         {n_vals_after:>12,d}  ({100.0 * n_vals_after / potential_vals:.1f}%)\n"
            f"  Mean quality score     {mean_quality:>12.3f}  (1=best, 9+=low)"
        )

        quality_counts = {k: v for k, v in sorted(flagcounts.items()) if k > 0}
        measured_count = flagcounts.get(0, 0)
        if quality_counts:
            rule("Quality Distribution", min_level=2)
            table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
            table.add_column("Level", style="dim", no_wrap=True)
            table.add_column("Count", justify="right")
            table.add_column("  %", justify="right")
            table.add_column("Description")
            table.add_row("0 (observed)", f"{measured_count:,d}",
                          f"{100.0 * measured_count / potential_vals:.1f}%", "measured")
            for q, cnt in sorted(quality_counts.items()):
                pct = 100.0 * cnt / potential_vals
                table.add_row(str(q), f"{cnt:,d}", f"{pct:.1f}%", mds_quality_description(q))
            _console.print(table)

        self.report_scores()

    def report_scores(self):
        """Print model performance scores with interpretation."""
        rule("Model Performance Scores", min_level=2)
        if self.scores_:
            for score, val in self.scores_.items():
                score_display = score.replace('_', ' ').title()
                _console.print(f"  {score_display:<35} {val:.4f}")
        else:
            _console.print("  No scores available")
        _console.print(
            "\n  [dim]Higher quality levels (1-3) = stricter meteorological match.[/dim]\n"
            "  [dim]Mean quality near 1 = most gaps filled under tight constraints.[/dim]"
        )

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

        detail(f"MDS Q{quality}: SWIN+TA+VPD, {days}d window")

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

        detail(f"MDS Q{quality}: SWIN only, {days}d window")

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

        detail(f"MDS Q{quality}: MDC, {days}d {hours}h window")

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
        gf_index_values = self.gapfilling_df_.index.to_numpy()
        gf_flux = self.gapfilling_df_[self.flux].values
        gf_ta = self.gapfilling_df_[self.ta].values
        gf_swin = self.gapfilling_df_[self.swin].values
        gf_vpd = self.gapfilling_df_[self.vpd].values

        # Process each gap row
        for i in range(n_gaps):
            if (i & 1023) == 0:  # report progress every ~1k gaps (cheap, frequent)
                self._emit_subprogress(i, n_gaps, predictions)
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
        gf_index_values = self.gapfilling_df_.index.to_numpy()
        gf_flux = self.gapfilling_df_[self.flux].values
        gf_swin = self.gapfilling_df_[self.swin].values

        # Process each gap row
        for i in range(n_gaps):
            if (i & 1023) == 0:  # report progress every ~1k gaps (cheap, frequent)
                self._emit_subprogress(i, n_gaps, predictions)
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
        gf_index_values = self.gapfilling_df_.index.to_numpy()
        gf_flux = self.gapfilling_df_[self.flux].values

        # Process each gap row (no meteorological conditions, just time window)
        for i in range(n_gaps):
            if (i & 1023) == 0:  # report progress every ~1k gaps (cheap, frequent)
                self._emit_subprogress(i, n_gaps, predictions)
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
        gf_index_values = gf_index.to_numpy()
        gf_flux = self.gapfilling_df_[self.flux].values
        gf_hours = gf_index.hour.to_numpy()

        # Process each gap row
        for i in range(n_gaps):
            if (i & 1023) == 0:  # report progress every ~1k gaps (cheap, frequent)
                self._emit_subprogress(i, n_gaps, predictions)
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

# See examples/gapfilling/gapfill_mds.py for usage examples.
# For the original unoptimized implementation, use _FluxMDS.
