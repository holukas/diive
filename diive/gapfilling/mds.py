"""
GAP-FILLING: MARGINAL DISTRIBUTION SAMPLING
============================================

Statistical gap-filling using meteorological similarity. No training required.

Faithful port of the ONEFlux marginal-distribution-sampling gap-filler: the
6-stage expanding-window cascade, the >=2-sample acceptance rule and the 1/2/3
quality collapse all follow ONEFlux (``oneflux/partition/daytime.py``
``uncert_via_gapFill`` / C ``common.c`` ``gf_mds``). The cascade itself lives in
:mod:`diive.gapfilling.similarity` (shared with the random-uncertainty step and
the daytime-partitioning NEE uncertainty); this module wraps it as a gap-filler.

The public gap-fill **flag** (``FLAG_..._gfMDS_ISFILLED``) is *granular*: it
encodes ``method * 1000 + time_window`` (0 = measured), so both the driver
method (1 = SWIN+TA+VPD, 2 = SWIN only, 3 = diurnal cycle) and the window are
recoverable. The faithful ONEFlux 1/2/3 quality is kept alongside in
``.PREDICTIONS_QUALITY``.

References:
    Reichstein et al. (2005). On the separation of net ecosystem exchange into
        assimilation and ecosystem respiration. Global Change Biology, 11(9),
        1424-1439. https://doi.org/10.1111/j.1365-2486.2005.001002.x
    Vekuri et al. (2023). A widely-used eddy covariance gap-filling method
        creates systematic bias in carbon balance estimates. Scientific Reports,
        13(1), 1720. https://doi.org/10.1038/s41598-023-28827-2

Part of the diive library: https://github.com/holukas/diive
"""
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
from diive.core.utils.console import console as _console, info, rule
from diive.gapfilling.scores import prediction_scores
from diive.gapfilling.similarity import (
    METHOD_ALL, METHOD_SWIN, METHOD_MDC,
    mds_gapfill_cascade, mds_quality_from,
)

#: Human names for the ONEFlux MDS driver methods.
_METHOD_NAMES = {
    METHOD_ALL: 'SWIN, TA, VPD',
    METHOD_SWIN: 'SWIN only',
    METHOD_MDC: 'diurnal cycle',
}


def mds_quality_description(flag: int) -> str:
    """Human description of a granular MDS gap-fill flag value.

    ``0`` = measured. A filled value encodes ``method * 1000 + time_window``
    (see :func:`diive.gapfilling.similarity.mds_granular_flag`): the method
    (1 = SWIN+TA+VPD, 2 = SWIN only, 3 = diurnal cycle) and the ONEFlux
    ``time_window`` in days, plus the collapsed 1/2/3 quality.
    """
    f = int(flag)
    if f == 0:
        return 'measured'
    method, tw = f // 1000, f % 1000
    name = _METHOD_NAMES.get(method, f'method {method}')
    return f'{name}: {tw} d window (quality {mds_quality_from(method, tw)})'


def _infer_nperday(index: pd.DatetimeIndex) -> int:
    """Records per day from a regular datetime index (48 for half-hourly)."""
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        raise ValueError("MDS needs at least two timestamps to infer the time resolution.")
    secs = deltas.median().total_seconds()
    if secs <= 0:
        raise ValueError("MDS needs a strictly increasing, regular datetime index.")
    return int(round(86400.0 / secs))


class _MdsGapFillingBase:
    """Shared implementation behind :class:`FluxMDS`.

    Delegates the algorithm to the single shared cascade in
    :mod:`diive.gapfilling.similarity`.
    """

    gfsuffix = '_gfMDS'

    def __init__(self,
                 df: DataFrame,
                 flux: str,
                 swin: str,
                 ta: str,
                 vpd: str,
                 swin_tol: list = None,  # default [20, 50]
                 ta_tol: float = 2.5,
                 vpd_tol: float = 0.5,
                 avg_min_n_vals: int = 2,
                 sym_mean: bool = False,
                 fill_marginal_gaps: bool = True,
                 vpd_in_kpa: bool = True,
                 verbose: int = 1):
        """Gap-fill an ecosystem flux by marginal distribution sampling (MDS).

        Missing values are replaced by the average measured *flux* during
        meteorologically similar conditions, using the faithful ONEFlux 6-stage
        expanding-window cascade.

        Args:
            df: DataFrame with *flux*, *swin*, *ta* and *vpd*. The index must be
                a regular (gap-free) datetime index.
            flux: flux variable to gap-fill.
            swin: short-wave incoming radiation (W m-2).
            ta: air temperature (deg C).
            vpd: vapor pressure deficit. Unit set by *vpd_in_kpa*.
            swin_tol: SWIN tolerance clamp ``[min, max]`` in W m-2 (default
                ``[20, 50]``). Following ONEFlux, a record's SWIN tolerance is its
                own SWIN clamped into this range (grows with radiation).
            ta_tol: TA similarity tolerance (deg C, default 2.5).
            vpd_tol: VPD similarity tolerance (kPa, default 0.5 = ONEFlux 5 hPa).
            avg_min_n_vals: minimum similar samples to accept a fill. ONEFlux
                gap-filling uses 2 (default); larger values are stricter.
            sym_mean: use the Vekuri (2023) symmetric mean (SWIN-driven methods)
                instead of the plain mean. Off by default (standard ONEFlux).
            fill_marginal_gaps: if True (default), fill everything (ONEFlux
                ``gf_mds`` gap-filling behaviour). If False, leave leading/
                trailing gaps longer than 60 days unfilled (the ONEFlux
                ``uncert_via_gapFill`` ``longestMarginalgap`` guard).
            vpd_in_kpa: if True (default), *vpd* is in kPa and *vpd_tol* applies
                directly. If False, *vpd* is in hPa and is converted to kPa
                internally so the kPa *vpd_tol* still applies.
            verbose: verbosity level.
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
        elif isinstance(swin_tol, list):
            self.swin_tol = swin_tol
        else:
            raise TypeError('swin_tol must be a list with two elements. (default: [20, 50])')
        self.ta_tol = ta_tol
        self.vpd_tol = vpd_tol
        self.avg_min_n_vals = avg_min_n_vals if avg_min_n_vals else 2
        self.sym_mean = sym_mean
        self.fill_marginal_gaps = fill_marginal_gaps
        self.vpd_in_kpa = vpd_in_kpa
        self.verbose = verbose

        self._scores = dict()
        self.target_gapfilled = f"{self.flux}{self.gfsuffix}"
        self.target_gapfilled_flag = f"FLAG_{self.flux}{self.gfsuffix}_ISFILLED"

    # --- results access ------------------------------------------------------
    def get_gapfilled_target(self):
        """Gap-filled target time series."""
        return self.gapfilling_df_[self.target_gapfilled].copy()

    def get_flag(self):
        """Gap-filling flag: 0 = observed, else granular (method*1000+window)."""
        return self.gapfilling_df_[self.target_gapfilled_flag]

    @property
    def result(self) -> DataFrame:
        """Primary result: full gap-filling DataFrame (target + flag columns)."""
        return self.gapfilling_df_

    @property
    def results(self) -> GapFillingResult:
        """Structured result after :meth:`run`."""
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
        series = self.get_gapfilled_target()
        if not isinstance(series, pd.Series):
            raise Exception('No gap-filled data available.')
        return series

    @property
    def target_col(self) -> str:
        if not isinstance(self.flux, str):
            raise Exception('No name for gap-filled variable available.')
        return self.flux

    @property
    def gapfilling_df_(self) -> DataFrame:
        if not isinstance(self._gapfilling_df, DataFrame):
            raise Exception('No dataframe containing all data available.')
        return self._gapfilling_df

    @property
    def scores_(self) -> dict:
        if not self._scores:
            raise Exception('Not available: model scores for gap-filling.')
        return self._scores

    # --- run -----------------------------------------------------------------
    def run(self, progress_callback=None):
        """Execute the faithful ONEFlux MDS cascade.

        Args:
            progress_callback: optional ``callable(filled, total, quality,
                n_filled, remaining)`` for a GUI progress bar (gap counts).
        """
        rule(f"MDS Gap-Filling: {self.flux}")

        df = self.gapfilling_df_
        index = df.index
        nperday = _infer_nperday(index)
        hr = (index.hour + index.minute / 60.0).to_numpy(dtype=float)

        def arr(col):
            a = df[col].to_numpy(dtype=float).copy()
            a[~np.isfinite(a)] = np.nan
            return a

        flux_arr = arr(self.flux)
        # vpd_tol is in kPa; convert an hPa VPD column so the tolerance applies.
        vpd_arr = arr(self.vpd)
        if not self.vpd_in_kpa:
            vpd_arr = vpd_arr * 0.1

        # Predict at every record (fill_all): gap predictions fill the gaps, and
        # predictions at measured records give the in-sample score (matches the
        # standard MDS validation). Marginal gaps are filled unless disabled.
        _cb = None
        if progress_callback is not None:
            def _cb(gaps_filled, gaps_total, quality):  # -> GUI 5-arg signature
                progress_callback(gaps_filled, gaps_total or 1, quality, 0,
                                  (gaps_total or 0) - gaps_filled)

        res = mds_gapfill_cascade(
            flux_arr, arr(self.swin), arr(self.ta), vpd_arr, hr, nperday,
            min_samples=self.avg_min_n_vals,
            swin_tol=(self.swin_tol[0], self.swin_tol[1]),
            ta_tol=self.ta_tol, vpd_tol=self.vpd_tol,
            ddof=1, sym_mean=self.sym_mean, fill_all=True,
            longest_marginal_gap=10 ** 9 if self.fill_marginal_gaps else 60,
            progress_callback=_cb,
        )

        predictions = res['filled']
        measured = np.isfinite(flux_arr)
        gap = ~measured

        df['.PREDICTIONS'] = predictions
        df['.PREDICTIONS_SD'] = res['sd']
        df['.PREDICTIONS_COUNTS'] = np.where(res['count'] > 0, res['count'], np.nan)
        df['.PREDICTIONS_METHOD'] = np.where(res['method'] > 0, res['method'], np.nan)
        df['.PREDICTIONS_TIMEWINDOW'] = np.where(res['method'] > 0, res['time_window'], np.nan)
        df['.PREDICTIONS_QUALITY'] = np.where(res['method'] > 0, res['quality'], np.nan)

        # Gap-filled series: measured kept, gaps filled where a prediction exists.
        df[self.target_gapfilled] = np.where(measured, flux_arr, predictions)

        # Flag: 0 = measured; granular at filled gaps; NaN at unfilled gaps.
        flag = np.where(measured, 0.0, np.nan)
        filled_gap = gap & np.isfinite(predictions)
        flag = np.where(filled_gap, res['flag'].astype(float), flag)
        df[self.target_gapfilled_flag] = flag

        # Scores: in-sample, predictions vs measured where both exist.
        score_mask = measured & np.isfinite(predictions)
        self._scores = prediction_scores(
            predictions=pd.Series(predictions[score_mask]),
            targets=pd.Series(flux_arr[score_mask]))
        # Mean faithful (1/2/3) quality across the gap predictions.
        q_gap = res['quality'][filled_gap]
        self._scores['mean_quality_flag_gap_predictions'] = (
            float(np.mean(q_gap)) if q_gap.size else float('nan'))

        info(f"MDS gap-filling done: {self.flux}", verbose=self.verbose)

    # --- diagnostics ---------------------------------------------------------
    def quality_breakdown(self) -> DataFrame:
        """Per-flag-level breakdown of the gap-filling flag.

        One row per flag value present (sorted): level 0 = measured, else the
        granular ``method*1000+window`` flag. Columns: ``level``, ``count``,
        ``pct`` (% of all records), ``description``.
        """
        flag = self.gapfilling_df_[self.target_gapfilled_flag]
        total = len(flag)
        counts = flag.dropna().astype(int).value_counts().sort_index()
        rows = [
            {'level': int(level), 'count': int(count),
             'pct': 100.0 * count / total if total else 0.0,
             'description': mds_quality_description(int(level))}
            for level, count in counts.items()
        ]
        return DataFrame(rows, columns=['level', 'count', 'pct', 'description'])

    def _flag_style(self, ix: int):
        """(colour, marker) for the ix-th sorted flag level (wraps the palette)."""
        colors = colorwheel_36_blackfirst()
        markers = generate_plot_marker_list()
        color = colors[35] if ix > (len(colors) - 1) else colors[ix]
        marker = markers[9] if ix > (len(markers) - 1) else markers[ix]
        return color, marker

    def plot_quality_timeseries(self, ax=None, legend: bool = True,
                                ax_labels_fontsize: int = None,
                                legend_textsize: int = None,
                                legend_ncol: int = None):
        """Gap-filled series over time, each point coloured + markered by its MDS
        flag level (0 = measured, else granular). Two-phase: ``ax=None`` builds a
        standalone figure; pass an ``ax`` to embed it. Returns the axes."""
        if ax_labels_fontsize is None:
            ax_labels_fontsize = theme.AX_LABELS_FONTSIZE_12
        if legend_textsize is None:
            legend_textsize = theme.FONTSIZE_TXT_LEGEND_SMALL_9
        if ax is None:
            fig = plt.figure(facecolor='white', figsize=(16, 5), dpi=100, layout='constrained')
            ax = fig.add_subplot(1, 1, 1)
        flag = self.gapfilling_df_[self.target_gapfilled_flag]
        levels = sorted(flag.dropna().unique())
        for ix, uf in enumerate(levels):
            data = self.gapfilling_df_.loc[flag == uf, :]
            n_vals = data[self.target_gapfilled].count()
            label = (f"measured ({self.flux})" if uf == 0
                     else f"{mds_quality_description(int(uf))}, mean +/- SD")
            color, marker = self._flag_style(ix)
            # Layer by quality tier so the best data is on top: measured (0) at the
            # very top, then quality 1 -> 2 -> 3; whiskers sit just under their
            # markers. (Draw order alone would bury measured under the looser fills.)
            tier = 0 if uf == 0 else mds_quality_from(int(uf) // 1000, int(uf) % 1000)
            zmark = 10 - tier
            # Smaller markers for measured so the dense observed series reads as
            # fine points on top rather than a heavy black layer.
            ms = 1.5 if uf == 0 else 5
            ax.plot(data.index, data[self.target_gapfilled],
                    label=f"{label} ({n_vals} values)", color=color, linestyle='none',
                    markeredgewidth=1, marker=marker, alpha=1, markersize=ms,
                    markeredgecolor=color, fillstyle='full', zorder=zmark)
            if uf > 0:
                ax.errorbar(data.index, data[self.target_gapfilled], data['.PREDICTIONS_SD'],
                            elinewidth=5, ecolor=color, alpha=.2, fmt='none', zorder=zmark - 0.5)
        default_format(ax=ax, ax_xlabel_txt='', ax_ylabel_txt=f"{self.flux}",
                       ax_labels_fontsize=ax_labels_fontsize)
        if legend:
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
        self.plot_quality_timeseries(ax=ax, legend=False)
        flag = self.gapfilling_df_[self.target_gapfilled_flag]
        for ix, uf in enumerate(sorted(flag.dropna().unique())):
            data = self.gapfilling_df_.loc[flag == uf, :]
            label = (f"measured ({self.flux})" if uf == 0
                     else f"{mds_quality_description(int(uf))}, mean +/- SD")
            n_vals = data[self.target_gapfilled].count()
            color, marker = self._flag_style(ix)
            if uf > 0:
                ax_counts.plot(data.index, data['.PREDICTIONS_COUNTS'],
                               label=f"{label} ({n_vals} values)", color=color, linestyle='none',
                               markeredgewidth=1, marker=marker, alpha=1, markersize=5,
                               markeredgecolor=color, fillstyle='full')
            ax_flag.plot(data.index, data[self.target_gapfilled_flag],
                         label=f"{label} ({n_vals} values)", color=color, linestyle='none',
                         markeredgewidth=1, marker=marker, alpha=1, markersize=5,
                         markeredgecolor=color, fillstyle='full')
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

    # --- reports -------------------------------------------------------------
    def report(self):
        """Print a comprehensive MDS gap-filling report."""
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

        from rich.table import Table

        rule(f"MDS Gap-Filling Report: {self.flux}")
        _console.print(
            "  Reference: ONEFlux marginal distribution sampling "
            "(Reichstein 2005 / Vekuri 2023)\n"
            "\n"
            "  [bold]Algorithm:[/bold] 6-stage expanding-window similarity cascade\n"
            "    method 1 (SWIN+TA+VPD), 2 (SWIN only), 3 (diurnal cycle)\n"
            "    flag = method*1000 + window (days); quality 1-3 collapses the window"
        )

        rule("Parameters", min_level=2)
        _console.print(
            f"  Flux variable          {self.flux}\n"
            f"  SWIN tolerance         [{self.swin_tol[0]}, {self.swin_tol[1]}] W m-2\n"
            f"  TA tolerance           {self.ta_tol} deg C\n"
            f"  VPD tolerance          {self.vpd_tol} kPa\n"
            f"  Min samples for mean   {self.avg_min_n_vals}\n"
            f"  Symmetric mean         {self.sym_mean}"
        )

        rule("Data & Performance", min_level=2)
        _console.print(
            f"  Total records          {potential_vals:>12,d}\n"
            f"  Available before       {n_vals_before:>12,d}  ({100.0 * n_vals_before / potential_vals:.1f}%)\n"
            f"  Missing before         {n_vals_missing_before:>12,d}  ({pct_missing_before:.1f}%)\n"
            f"  Filled                 {n_vals_filled:>12,d}  ({pct_recovery:.1f}% of gaps)\n"
            f"  Remaining missing      {n_vals_missing_after:>12,d}  ({pct_missing_after:.1f}% of total)\n"
            f"  Final coverage         {n_vals_after:>12,d}  ({100.0 * n_vals_after / potential_vals:.1f}%)\n"
            f"  Mean quality (1-3)     {mean_quality:>12.3f}  (1 = tightest match)"
        )

        breakdown = self.quality_breakdown()
        filled = breakdown[breakdown['level'] > 0]
        if not filled.empty:
            rule("Quality Distribution", min_level=2)
            table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
            table.add_column("Flag", style="dim", no_wrap=True)
            table.add_column("Count", justify="right")
            table.add_column("  %", justify="right")
            table.add_column("Description")
            measured_row = breakdown[breakdown['level'] == 0]
            if not measured_row.empty:
                m = measured_row.iloc[0]
                table.add_row("0", f"{int(m['count']):,d}", f"{m['pct']:.1f}%", "measured")
            for _, row in filled.iterrows():
                table.add_row(str(int(row['level'])), f"{int(row['count']):,d}",
                              f"{row['pct']:.1f}%", row['description'])
            _console.print(table)

        self.report_scores()

    def report_scores(self):
        """Print model performance scores with interpretation."""
        rule("Model Performance Scores", min_level=2)
        if self._scores:
            for score, val in self._scores.items():
                score_display = score.replace('_', ' ').title()
                _console.print(f"  {score_display:<35} {val:.4f}")
        else:
            _console.print("  No scores available")
        _console.print(
            "\n  [dim]Flag = method*1000 + window; quality 1-3 collapses the window "
            "(1 = tightest match).[/dim]"
        )


class FluxMDS(_MdsGapFillingBase):
    """Gap-filling for ecosystem fluxes using Marginal Distribution Sampling (MDS).

    Fills missing flux data by matching meteorologically similar conditions
    (SWIN, TA, VPD) following the faithful ONEFlux 6-stage expanding-window
    cascade (Reichstein 2005 / Vekuri 2023). The cascade is shared with the
    random-uncertainty step (:mod:`diive.gapfilling.similarity`).

    Examples:
        See examples/gapfilling/gapfill_mds.py for basic usage.
    """


# See examples/gapfilling/gapfill_mds.py for usage examples.
