"""
USTAR THRESHOLD: FRICTION VELOCITY FILTERING
==============================================

Flag low-turbulence data using constant friction velocity thresholds with uncertainty scenarios.

Part of the diive library: https://github.com/holukas/diive
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from pandas import Series, DataFrame

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme
from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import detail
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.variables import daytime_nighttime_flag_from_swinpot


class FlagMultipleConstantUstarThresholds:
    """
    Apply multiple constant USTAR thresholds to filter low-turbulence flux data.

    USTAR (friction velocity) threshold filtering removes unreliable nighttime flux
    measurements during low-turbulence conditions when turbulent mixing is insufficient
    to properly represent true surface exchange. This class applies multiple threshold
    scenarios (e.g., 16th, 50th, 84th percentiles) to quantify uncertainty in
    the filtering decision.

    Critical Note (FLUXNET Standard):
    USTAR filtering should ONLY be applied to scalar fluxes (CO2, CH4, N2O),
    NOT to energy fluxes (H sensible heat, LE latent heat) because advective
    fluxes don't proportionally affect energy balance at night.

    Deliberate deviation from ONEFlux:
    ONEFlux also discards the first record *above* the threshold that follows a
    period below it, to drop the CO2 that accumulated under the canopy and is
    flushed past the sensor in one burst (Pastorello et al. 2020, Sci Data 7:225;
    ``nee_proc/src/dataset.c``). diive keeps that record. The ONEFlux rule is the
    stricter one and defensible; diive favours data availability here and leaves
    the trade-off to the user, who can drop those records afterwards if the burst
    matters for their analysis. Everything else about the comparison follows
    ONEFlux, including rejecting a record whose USTAR is missing.

    Args:
        series: pandas Series containing flux data
        ustar: pandas Series containing USTAR (friction velocity) data
        thresholds: list of float threshold values (e.g., [0.05, 0.07, 0.09])
        threshold_labels: list of str labels for each threshold scenario
        showplot: If True, displays filtering results. Default True.
        verbose: If True, prints filtering statistics. Default True.
        idstr: Optional identifier string for output columns

    Example:
        See `examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py` for complete examples
        of USTAR threshold detection and data filtering.
    """

    def __init__(self, series, ustar, thresholds, threshold_labels,
                 showplot: bool = True, verbose: bool = True, idstr: str = None):
        """Set up flagging for several constant USTAR thresholds. See the class docstring for parameters."""
        self.series = series
        self.ustar = ustar
        self.thresholds = thresholds
        self.threshold_labels = threshold_labels
        self.showplot = showplot
        self.verbose = verbose
        self.idstr = idstr

        self._results = pd.concat([self.series, self.ustar], axis=1).copy()

    @property
    def results(self) -> DataFrame:
        """Return high-resolution detailed data with tags as dict of DataFrames."""
        if not isinstance(self._results, DataFrame):
            raise Exception("No USTAR flags available.")
        return self._results

    def get_results(self) -> DataFrame:
        """Return the results DataFrame (flux, USTAR and one flag column per threshold)."""
        return self.results

    def calc(self):
        """Compute one USTAR threshold flag per scenario and append them to the results."""
        for ix, threshold in enumerate(self.thresholds):
            idstr = f"{self.idstr}_{self.threshold_labels[ix]}" if self.idstr else f"{self.threshold_labels[ix]}"
            ust = FlagSingleConstantUstarThreshold(
                series=self.results[self.series.name],
                ustar=self.results[self.ustar.name],
                threshold=threshold,
                idstr=idstr,
                showplot=self.showplot,
                verbose=self.verbose
            )
            ust.calc()
            flag = ust.get_flag()
            self._results[flag.name] = flag.copy()


@ConsoleOutputDecorator()
class FlagSingleConstantUstarThreshold(FlagBase):
    """Flag records below a single constant USTAR threshold. See :meth:`__init__`."""

    flagid = 'USTAR'

    def __init__(self,
                 series: Series,
                 ustar: Series,
                 threshold: float,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Flag records below a single constant USTAR threshold.

        A record is kept only where ``ustar >= threshold`` can be shown, so a record
        with no USTAR measurement is rejected (as in ONEFlux, where a missing USTAR
        is ``INVALID_VALUE`` and therefore below every threshold). diive deliberately
        does *not* apply ONEFlux's follow-up rule that also discards the first record
        above the threshold after a period below it — see
        :class:`FlagMultipleConstantUstarThresholds` for why.

        Args:
            series: Flux series to flag.
            ustar: Friction velocity (USTAR) series aligned to *series*.
            threshold: Records with ``ustar < threshold``, or with no USTAR at all,
                are flagged as rejected.
            idstr: Optional identifier string appended to the flag column name.
            showplot: If True, show the default rejected-values plot.
            verbose: If True, print detection statistics.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.ustar = ustar
        self.threshold = threshold
        self.showplot = False
        self.verbose = False
        self.showplot = showplot
        self.verbose = verbose

        # if self.showplot:
        #     self.fig, self.ax, self.ax2 = self._plot_init()

    def calc(self):
        """Calculate the overall flag (single pass; USTAR thresholding does not iterate)."""

        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=False)
        if self.showplot:
            # Default plot for outlier tests, showing rejected values
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # USTAR filtering is a positive test: a record is kept only where the
        # measured turbulence can be shown to reach the threshold. A record whose
        # USTAR is missing therefore fails it. Testing `ustar < threshold` for the
        # rejected set instead left those records in neither list, so their flag
        # summed to 0 - accepted, silently, with turbulence unknown. Flagging them
        # NaN ("not testable") would not help: FlagQCF sums only 1s and 2s, so a
        # NaN flag is accepted downstream just the same.
        passes = (self.ustar >= self.threshold)
        passes = passes.fillna(False).astype(bool)
        ok = passes[passes].index
        rejected = passes[~passes].index
        n_outliers = len(rejected)

        if self.verbose:
            thr_repr = "variable (per-record)" if isinstance(self.threshold, Series) else self.threshold
            n_no_ustar = int(self.ustar.isna().sum())
            detail(f"Total found outliers for USTAR threshold {self._idstr} {thr_repr}: {len(rejected)} values"
                   + (f" (of which {n_no_ustar} have no USTAR measurement)" if n_no_ustar else ""),
                   verbose=self.verbose)

        return ok, rejected, n_outliers


class FlagMultipleVariableUstarThresholds:
    """
    Apply multiple *time-varying* USTAR thresholds (e.g. per-year, VUT) to filter flux data.

    Variable-threshold counterpart of :class:`FlagMultipleConstantUstarThresholds`: instead
    of one scalar per scenario, each scenario carries a full per-record threshold Series
    (aligned to ``series``/``ustar``). This is what the FLUXNET/ONEFlux **VUT** (Variable
    U\\* Threshold) approach needs — each year (or any sub-period) filtered by its own
    threshold. A constant threshold is just a constant Series, so this class can also
    express CUT scenarios when CUT and VUT are applied together.

    The element-wise comparison ``ustar >= threshold`` is identical to the constant case;
    only the threshold is broadcast per record rather than as a single value. That includes
    the deliberate deviation from ONEFlux described in
    :class:`FlagMultipleConstantUstarThresholds`: the first record above the threshold after
    a period below it is kept, not discarded.

    Args:
        series: flux data (pandas Series).
        ustar: USTAR / friction velocity (pandas Series).
        threshold_series: mapping ``{scenario_label: per_record_threshold_Series}``. Each
            threshold Series must align to ``series.index`` and contain no NaN (the caller
            resolves any missing periods first).
        showplot, verbose, idstr: as in :class:`FlagMultipleConstantUstarThresholds`.
    """

    def __init__(self, series, ustar, threshold_series: dict,
                 showplot: bool = True, verbose: bool = True, idstr: str = None):
        """Set up flagging for per-record (variable) USTAR thresholds. See the class docstring."""
        self.series = series
        self.ustar = ustar
        self.threshold_series = threshold_series
        self.showplot = showplot
        self.verbose = verbose
        self.idstr = idstr
        self._results = pd.concat([self.series, self.ustar], axis=1).copy()

    @property
    def results(self) -> DataFrame:
        """Return the results DataFrame (flux, USTAR and one flag column per scenario)."""
        if not isinstance(self._results, DataFrame):
            raise Exception("No USTAR flags available.")
        return self._results

    def get_results(self) -> DataFrame:
        """Return the results DataFrame (flux, USTAR and one flag column per scenario)."""
        return self.results

    def calc(self):
        """Compute one USTAR flag per scenario from its per-record threshold and append them."""
        for label, thr in self.threshold_series.items():
            idstr = f"{self.idstr}_{label}" if self.idstr else f"{label}"
            thr = thr.reindex(self.results.index)  # per-record threshold
            # Reindexing a threshold Series that does not cover the whole record
            # fills the rest with NaN, and a record with no threshold cannot pass
            # the test - it would be rejected wholesale without saying why. The
            # docstring already requires a gap-free Series; enforce it.
            n_missing = int(thr.isna().sum())
            if n_missing:
                raise ValueError(
                    f"Threshold series '{label}' has no threshold for {n_missing} of "
                    f"{len(thr)} records. Provide a threshold for every record "
                    f"(e.g. fall back to the pooled CUT value for years without one).")
            ust = FlagSingleConstantUstarThreshold(
                series=self.results[self.series.name],
                ustar=self.results[self.ustar.name],
                threshold=thr,
                idstr=idstr,
                showplot=self.showplot,
                verbose=self.verbose,
            )
            ust.calc()
            flag = ust.get_flag()
            self._results[flag.name] = flag.copy()


@ConsoleOutputDecorator()
class UstarThresholdConstantScenarios:
    """
    Check impact of different constant USTAR thresholds on available data

    Constant means that the threshold is the same for all data, e.g. the same for
    all years.
    ...

    Methods:
        calc(ustarthresholds:list=None, showplot: bool = False, verbose: bool = False):
            Creates timeseries of *series* in after application of different USTAR
            thresholds given in *ustarthresholds*.

    Properties:
        scenariosdf: DataFrame of the timeseries of *series* in different USTAR
            scenarios. Records of *series* where USTAR was below the respective
            threshold are set to missing.

    """

    def __init__(self, series: Series, ustar: Series, swinpot: Series):
        """Set up constant-threshold USTAR scenario testing. See the class docstring."""
        self.series = series
        self.ustar = ustar
        self.swinpot = swinpot
        self.showplot = False
        self.verbose = False

        self._scenariosdf = None

        # Detect daytime and nighttime from potential radiation
        self.daytime, self.nighttime = \
            daytime_nighttime_flag_from_swinpot(swinpot=swinpot, nighttime_threshold=20)

        # Convert 0/1 flags to False/True flag
        self.daytime = self.daytime == 1
        self.nighttime = self.nighttime == 1

    @property
    def scenariosdf(self):
        """Return timeseries of *series* of each USTAR threshold, values
        below the respective threshold were removed"""
        if not isinstance(self._scenariosdf, DataFrame):
            raise Exception(f'USTAR scenarios are empty. '
                            f'Solution: run .calc() to create USTAR scenarios for {self.series.name}.')
        return self._scenariosdf

    def calc(self, ustarthresholds: list = None, showplot: bool = False, verbose: bool = False):
        """Calculate flag"""
        if ustarthresholds is None:
            ustarthresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.showplot = showplot
        self.verbose = verbose
        # self.reset()

        self._scenariosdf = pd.DataFrame(self.series).copy()

        # Create timeseries for each USTAR threshold
        for u in ustarthresholds:
            suffix = f"CUT{u}"
            colname = f"{self.series.name}_{suffix}"
            series_cut = self.series.copy()
            series_cut.loc[self.ustar < u] = np.nan
            self._scenariosdf[colname] = series_cut

        # Get daytime and nighttime data separately
        _scenariosdf_daytime = self._scenariosdf.loc[self.daytime].copy()
        _scenariosdf_nighttime = self._scenariosdf.loc[self.nighttime].copy()

        # total_potential = len(self._scenariosdf)

        if self.showplot: self._plot(daytimedf=_scenariosdf_daytime, nighttimedf=_scenariosdf_nighttime)

    def _plot(self, daytimedf, nighttimedf):
        # Count available records for each USTAR threshold
        counts = self._scenariosdf.describe().loc['count']
        counts_daytime = daytimedf.describe().loc['count']
        counts_nighttime = nighttimedf.describe().loc['count']

        # Create new figure
        fig = plt.figure(facecolor='white', figsize=(12, 16))
        gs = gridspec.GridSpec(4, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.1, left=0.1, right=0.98, top=0.95, bottom=0.05)
        ax_dtnt = fig.add_subplot(gs[0, 0])
        ax_dt = fig.add_subplot(gs[1, 0], sharex=ax_dtnt)
        ax_nt = fig.add_subplot(gs[2, 0], sharex=ax_dtnt)
        ax_stacked = fig.add_subplot(gs[3, 0], sharex=ax_dtnt)

        # Generate bar plots
        bar_dtnt = ax_dtnt.bar(counts.index, counts, label='daytime + nighttime', width=.9, fc='#9CCC65')
        bar_dt = ax_dt.bar(counts_daytime.index, counts_daytime, label='daytime', width=.9, fc='#FFA726')
        bar_nt = ax_nt.bar(counts_nighttime.index, counts_nighttime, label='nighttime', width=.9, fc='#42A5F5')

        # Show text in bar plots
        axes_lst = [ax_dtnt, ax_dt, ax_nt]
        counts_lst = [counts, counts_daytime, counts_nighttime]
        bar_lst = [bar_dtnt, bar_dt, bar_nt]
        for ix, a in enumerate(axes_lst):
            self._bartxt(ax=axes_lst[ix], counts=counts_lst[ix], bar=bar_lst[ix])
            pf.default_legend(ax=a)
            plt.setp(a.get_xticklabels(), visible=False)

        # Stacked bar
        bar_stacked = ax_stacked.bar(counts_nighttime.index, counts_nighttime,
                                     width=.9, label='nighttime', fc='#42A5F5')
        bar_stacked = ax_stacked.bar(counts_daytime.index, counts_daytime,
                                     width=.9, bottom=counts_nighttime, label='daytime', fc='#FFA726')
        pf.default_legend(ax=ax_stacked)

        # Figure title
        title = "Available values after applying different constant USTAR thresholds"
        fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
        fig.text(0.5, 0.01, 'USTAR thresholds', ha='center', size=20)
        fig.text(0.02, 0.5, 'Available values', va='center', rotation='vertical', size=20)

        fig.show()

    def _bartxt(self, ax, counts, bar):
        # counts is label-indexed (column names), so both lookups must be positional:
        # the first column is the unfiltered series, i.e. 100% of available values.
        counts_perc = counts.div(counts.iloc[0]).multiply(100)
        for ix, rect in enumerate(bar):
            height = rect.get_height()
            # Number of values
            ax.text(rect.get_x() + rect.get_width() / 2.0, height,
                    f'{height:.0f}', size=16, ha='center', va='bottom')
            # Percentage
            ax.text(rect.get_x() + rect.get_width() / 2.0, height / 2,
                    f'{counts_perc.iloc[ix]:.0f}%', size=20, ha='center', va='bottom')

        # _bottom = np.nan  # Needed for stacking multiple bars on top of each other
        #
        #         for flag, row in _plot_df.iterrows():
        #             _flag_counts = _plot_df.loc[flag].replace(np.nan, 0)  # Needs 0 for correct counts
        #             if flag == 0:
        #                 ax.bar(labels, _flag_counts, width=0.8, label=flag)
        #                 for bar_ix, bar in enumerate(ax.patches):
        #                     # kudos: https://www.pythoncharts.com/matplotlib/stacked-bar-charts-labels/
        #
        #                     # Show flag 0 (best) counts in plot
        #                     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
        #                             f"{round(bar.get_height())}", ha='center', color='w', weight='bold', size=6)
        #
        #                     # Show labels *inside* plot
        #                     ax.text(bar.get_x() + bar.get_width() / 10, bar.get_y() / 2,
        #                             f"{labels[bar_ix]}", ha='left', color='white', weight='bold', size=7, rotation=90)
        #
        #                 _bottom = _flag_counts
        #             else:
        #                 ax.bar(labels, _flag_counts, width=0.8, bottom=_bottom, label=flag)
        #                 _bottom = _flag_counts + _bottom

        # ok, rejected = self._flagtests(threshold=threshold)
        # self.setflag(ok=ok, rejected=rejected)
        # self.setfiltered(rejected=rejected)

