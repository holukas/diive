"""
BASE: FLAGBASE
==============

Base class for quality flags and outlier detectors.

Part of the diive library: https://github.com/holukas/diive
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex
from pandas import Series

import diive.core.plotting.styles.chainplots as cp
from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.plotfuncs import nice_date_ticks
from diive.core.times.times import DetectFrequency, continuous_timestamp_freq


def sum_flag_value(df: DataFrame, value: int) -> Series:
    """Row sums of the cells equal to *value*, same result as ``df[df == value].sum(axis=1)``.

    Builds no masked frame. The sums are small integers, so they are exact in any
    summation order. Frames with a column other than float64 use the pandas
    formula, which keeps the result dtype of that formula.
    """
    if len(df.columns) and all(dt == np.float64 for dt in df.dtypes):
        hits = np.where(df.to_numpy() == value, float(value), 0.0)
        return Series(hits.sum(axis=1), index=df.index)
    return df[df == value].sum(axis=1)


def _draw_points(ax, series: Series, color: str, label: str, **kwargs) -> None:
    """Time series as small rasterised points (a year of half-hours is 17,520 of them)."""
    s = series.dropna()
    ax.plot(s.index, s.to_numpy(), color=color, label=label, **{**cp.POINTS, **kwargs})


def _marginal_histogram(ax, layers: list) -> None:
    """Horizontal stacked histogram beside a time series panel, sharing its y-axis.

    `layers` is a list of ``(series, colour, label)``. All layers share one set of bins over
    their joint range, so a rejected record sits in the same bin as it would among the kept
    ones. The count axis is logarithmic: the few records in the tails are what an outlier
    test is about, and on a linear axis they disappear next to the central peak.
    """
    data = [(s.dropna().to_numpy(dtype=float), c, lbl) for s, c, lbl in layers]
    data = [d for d in data if d[0].size]
    if not data:
        return
    edges = np.histogram_bin_edges(np.concatenate([d[0] for d in data]), bins=40)
    ax.hist([d[0] for d in data], bins=edges, orientation='horizontal', stacked=True,
            color=[d[1] for d in data], label=[d[2] for d in data], histtype='barstacked',
            rwidth=1.0, edgecolor='white', linewidth=0.3)
    ax.set_xscale('log')
    ax.set_xlabel('records')
    cp.style_axes(ax, grid_axis='x')
    ax.tick_params(axis='y', labelleft=False)


#: Plain names for the tests behind the flag names, for the figure titles.
_TEST_NAMES = {
    'ABSLIM': 'Absolute limits', 'TRIMLOW': 'Lower limit', 'MANUALREMOVAL': 'Manual removal',
    'HAMPEL': 'Hampel filter', 'LOCALSD': 'Local standard deviation', 'ZSCORE': 'z-score',
    'LOF': 'Local outlier factor', 'INCRZ': 'Increments z-score', 'USTAR': 'USTAR filter',
    'NEGNIGHT': 'Negative nighttime values',
}


def _test_name(flagname: str) -> str:
    """'Absolute limits' for 'FLAG_..._OUTLIER_ABSLIM_TEST', else the tag the flag name ends in."""
    # A rerun of a test adds a number (..._HAMPEL_2_TEST), which is not part of its name.
    parts = [p for p in str(flagname).removesuffix('_TEST').split('_') if not p.isdigit()]
    tag = parts[-1] if parts else str(flagname)
    return _TEST_NAMES.get(tag, tag.replace('_', ' ').capitalize())


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f} %" if whole else "-"


class FlagBase:
    """Base class for flag/outlier detectors: holds the series and exposes the flag and filtered result."""

    #: Give records that are missing in the input a NaN flag instead of 0.
    #: A flag records a test *result*, and no test can be performed where there
    #: is no value, so 0 ("tested, passed") would be a claim the detector has not
    #: earned — and it silently inflates any `(flag == 0).count()` a consumer does.
    #: Subclasses whose test subject *is* the missing records (MissingValues) set
    #: this False, otherwise their own output would be masked away.
    nan_flag_at_missing: bool = True

    def __init__(self, series: Series, flagid: str, idstr: str = None, verbose: bool = True):
        """Store the series and flag identifiers; ensure the index has a frequency.

        A series without a frequency is put on a regular time grid. The grid
        follows the position within the time step that most records share.
        Records that are not on the grid are removed and their number is shown
        as a warning.
        """
        self.series = series
        self._flagid = flagid
        self._idstr = validate_id_string(idstr=idstr)
        self.verbose = verbose

        # Make sure time series has frequency
        # Freq is needed for the detection of daytime/nighttime from lat/lon.
        # The grid follows the records (asfreq would start it at the first
        # record). Records that are not on the grid are removed with a warning.
        if not self.series.index.freq:
            freq = DetectFrequency(index=self.series.index, verbose=True).get()
            self.series = continuous_timestamp_freq(data=self.series, freq=freq, verbose=self.verbose)

        self._overall_flag = None
        self._filteredseries = None
        self._flag = None

        self._fig = None

    @property
    def overall_flag(self) -> Series:
        """Overall flag, calculated from individual flags from multiple iterations.

        0 = tested and accepted, 2 = rejected as an outlier, NaN = not testable
        because the record is missing in the input (see `nan_flag_at_missing`).
        """
        if not isinstance(self._overall_flag, Series):
            raise Exception('No overall flag available.')
        return self._overall_flag

    def run(self, **kwargs):
        """Unified trigger. Delegates to .calc()."""
        self.calc(**kwargs)
        return self

    @property
    def result(self) -> Series:
        """Primary result: filtered series with outliers removed."""
        return self.filteredseries

    def get_flag(self):
        """Return the overall flag Series."""
        return self.overall_flag

    @property
    def flag(self) -> Series:
        """Return flag as Series"""
        if not isinstance(self._flag, Series):
            raise Exception(f'Flag is empty. '
                            f'Solution: run .calc() to create flag for {self.series.name}.')
        return self._flag

    @property
    def fig(self) -> Series:
        """Return figure instance of default plot"""
        if not self._fig:
            raise Exception(f'No figure available.')
        return self._fig

    @property
    def filteredseries(self) -> Series:
        """Return data without rejected records"""
        if not isinstance(self._filteredseries, Series):
            raise Exception(f'Filtered data not available. '
                            f'Solution: run .calc() to create flag for {self.series.name}.')
        return self._filteredseries

    def collect_results(self) -> DataFrame:
        """Store flag of this iteration in dataframe."""
        frame = {
            # self.filteredseries.name: self.filteredseries.copy(),
            self.flag.name: self.flag.copy()
        }
        iteration_df = pd.DataFrame.from_dict(frame)
        return iteration_df

    def get_filteredseries(self, iteration) -> Series:
        """For the first iteration the original input series is used,
        for all other iterations the filtered series from the previous
        iteration is used."""
        filteredseries = self.series.copy() if iteration == 1 else self.filteredseries
        # Rename filtered series to include iteration number
        filteredname = self.generate_iteration_filtered_variable_name(iteration=iteration)
        filteredseries.name = filteredname
        return filteredseries

    def init_flag(self, iteration):
        """Initiate (empty) flag for this iteration."""
        flagname = self.generate_flagname(iteration=iteration)
        flag = pd.Series(index=self.filteredseries.index, data=np.nan, name=flagname)
        return flag

    def setflag(self, ok: DatetimeIndex, rejected: DatetimeIndex):
        """Set flag with values 0=ok, 2=rejected"""
        self._flag.loc[ok] = 0
        if isinstance(rejected, DatetimeIndex):
            self._flag.loc[rejected] = 2
        else:
            # If no rejected values were found, pass because otherwise
            # NaT is inserted I think at the end of the flag series.
            pass

    def setfiltered(self, rejected: DatetimeIndex):
        """Set rejected values to missing"""
        # Only when rejected exists. If there are no rejected values
        # and the lines here would be executed, it would attach an
        # additional line to the series with NaT as index and Nan
        # as value.
        if isinstance(rejected, DatetimeIndex):
            self._filteredseries.loc[rejected] = np.nan

    def generate_flagname(self, iteration: int = None) -> str:
        """Generate standardized name for flag variable"""
        flagname = "FLAG"
        if self._idstr:
            flagname += f"{self._idstr}"
        flagname += f"_{self.series.name}"
        if self._flagid:
            flagname += f"_{self._flagid}"
        if iteration:
            flagname += f'_ITER{iteration}_TEST'
        else:
            flagname += '_TEST'
        return flagname

    def generate_iteration_filtered_variable_name(self, iteration: int):
        """Build the filtered-series column name for a given detection *iteration*."""
        filteredname = f"{self.series.name}_FILTERED-AFTER-ITER{iteration}"
        return filteredname

    def repeat(self, func, repeat, progress_callback=None):
        """Repeat function until no more outliers found.

        Args:
            func: Per-iteration flag test, returns ``(iteration_df, n_outliers)``.
            repeat: If *True*, repeat until an iteration finds no more outliers.
            progress_callback: Optional ``callable(iteration, n_outliers,
                filteredseries)`` invoked after each iteration (e.g. to drive a
                progress indicator / live plot). ``filteredseries`` is a copy of the
                series cleaned so far (outliers removed up to this iteration). The
                final call carries ``n_outliers == 0`` when the loop converges.
        """
        n_outliers = 9999
        iteration = 0
        iteration_flags_df = pd.DataFrame()
        while n_outliers > 0:
            iteration += 1
            cur_iteration_flag_df, n_outliers = func(iteration=iteration)
            iteration_flags_df = pd.concat([iteration_flags_df, cur_iteration_flag_df], axis=1)
            if progress_callback is not None:
                # Copy so the consumer (possibly another thread) can't race the
                # next iteration's in-place mutation of the filtered series.
                progress_callback(iteration, n_outliers, self.filteredseries.copy())
            if not repeat:
                break

        # Calculate the sum of all flags that show 2, for each data row
        overall_flag = sum_flag_value(iteration_flags_df, 2)
        # An all-NaN row sums to 0, so records that were never testable (missing
        # in the input) would otherwise be indistinguishable from records that
        # were tested and passed. See `nan_flag_at_missing`.
        if self.nan_flag_at_missing:
            overall_flag = overall_flag.where(self.series.notna())
        overall_flag.name = self.generate_flagname()

        n_iterations = len(iteration_flags_df.columns)

        return overall_flag, n_iterations

    def run_flagtests(self, iteration):
        """Calculate flag for given iteration."""
        self._filteredseries = self.get_filteredseries(iteration=iteration)
        self._flag = self.init_flag(iteration=iteration)
        ok, rejected, n_outliers = self._flagtests(iteration=iteration)
        self.setflag(ok=ok, rejected=rejected)
        self.setfiltered(rejected=rejected)
        iteration_df = self.collect_results()
        return iteration_df, n_outliers

    def defaultplot(self, n_iterations: int = 1, showplot: bool = True, title: str = None):
        """Time series before and after the test, each with its distribution beside it.

        The upper row shows every record with the rejected ones in red, the lower row the
        records the test kept, on their own y-range. The histogram beside each panel shares
        its y-axis and has a z-score axis on the right. If ``title`` is given it replaces the
        default title.
        """
        ok = self.overall_flag == 0
        rejected = self.overall_flag == 2
        kept, rej = self.series[ok], self.series[rejected]
        n_tested = int(self.series.notna().sum())

        fig = plt.figure(facecolor='white', figsize=(14, 7.5), layout='constrained')
        gs = fig.add_gridspec(2, 2, width_ratios=[5, 1])
        ax_all = fig.add_subplot(gs[0, 0])
        ax_all_hist = fig.add_subplot(gs[0, 1], sharey=ax_all)
        ax_kept = fig.add_subplot(gs[1, 0], sharex=ax_all)
        ax_kept_hist = fig.add_subplot(gs[1, 1], sharey=ax_kept)

        _draw_points(ax_all, kept, cp.KEPT, f"kept ({kept.count():,})")
        _draw_points(ax_all, rej, cp.REJECTED, f"rejected ({rej.count():,})", markersize=4)
        _marginal_histogram(ax_all_hist, [(kept, cp.KEPT, 'kept'), (rej, cp.REJECTED, 'rejected')])
        cp.zscore_axis(ax_all_hist, self.series)
        if kept.dropna().empty:
            cp.note(ax_kept, f"{self.series.name}: no data")
        else:
            _draw_points(ax_kept, kept, cp.CLEANED, f"kept ({kept.count():,})")
            _marginal_histogram(ax_kept_hist, [(kept, cp.CLEANED, 'kept')])
            cp.zscore_axis(ax_kept_hist, kept)

        for ax in (ax_all, ax_kept):
            cp.style_axes(ax)
            ax.set_ylabel(self.series.name)
        cp.panel_label(ax_all, "All records")
        cp.panel_label(ax_kept, "After the test")
        if ax_all.get_legend_handles_labels()[0]:
            ax_all.legend(loc='lower right', bbox_to_anchor=(1, 1), frameon=False, markerscale=3,
                          fontsize=9, ncol=2, borderaxespad=0.2)
        ax_all.tick_params(axis='x', labelbottom=False)
        nice_date_ticks(ax=ax_kept)

        cp.figure_title(
            fig, title if title is not None else f"{_test_name(self.overall_flag.name)}: {self.series.name}",
            f"{rej.count():,} of {n_tested:,} records rejected ({_percent(rej.count(), n_tested)}), "
            f"{n_iterations} iteration{'s' if n_iterations != 1 else ''}")
        if showplot:
            fig.show()

        self._fig = fig

    def plot_outlier_daytime_nighttime(self, series: Series, flag_daytime: Series,
                                       flag_quality: Series, title: str = None,
                                       n_iterations: int = None):
        """Every record with the rejected ones in red, then the kept daytime and nighttime records.

        Three rows, each a time series with its distribution beside it on a shared y-axis and
        a z-score axis on the right. The top row colours the kept records by day (amber) and
        night (indigo); the two rows below show what the test kept in each period, on their
        own y-range.
        """
        day, night = flag_daytime == 1, flag_daytime == 0
        kept, rejected = flag_quality == 0, flag_quality == 2
        valid = series.notna()
        n_rej = int((rejected & valid).sum())
        n_day, n_night = int((day & valid).sum()), int((night & valid).sum())
        n_rej_day, n_rej_night = int((rejected & day & valid).sum()), int((rejected & night & valid).sum())

        fig = plt.figure(facecolor='white', figsize=(14, 10), layout='constrained')
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax_all = fig.add_subplot(gs[0, 0])
        ax_all_hist = fig.add_subplot(gs[0, 1], sharey=ax_all)
        ax_day = fig.add_subplot(gs[1, 0], sharex=ax_all)
        ax_day_hist = fig.add_subplot(gs[1, 1], sharey=ax_day)
        ax_night = fig.add_subplot(gs[2, 0], sharex=ax_all)
        ax_night_hist = fig.add_subplot(gs[2, 1], sharey=ax_night)

        kept_day, kept_night = series[kept & day], series[kept & night]
        rej_all = series[rejected]
        _draw_points(ax_all, kept_day, cp.DAY, f"kept, daytime ({kept_day.count():,})", alpha=.45)
        _draw_points(ax_all, kept_night, cp.NIGHT, f"kept, nighttime ({kept_night.count():,})", alpha=.45)
        _draw_points(ax_all, rej_all, cp.REJECTED, f"rejected ({rej_all.count():,})", markersize=4)
        _marginal_histogram(ax_all_hist, [(series[kept], cp.KEPT, 'kept'),
                                          (rej_all, cp.REJECTED, 'rejected')])
        cp.zscore_axis(ax_all_hist, series)

        for ax, ax_hist, s, color, label in (
                (ax_day, ax_day_hist, kept_day, cp.DAY, "Daytime, after the test"),
                (ax_night, ax_night_hist, kept_night, cp.NIGHT, "Nighttime, after the test")):
            if s.dropna().empty:
                cp.note(ax, "no data")
            else:
                _draw_points(ax, s, color, None)
                _marginal_histogram(ax_hist, [(s, color, 'kept')])
                cp.zscore_axis(ax_hist, s)
            cp.panel_label(ax, label)

        for ax in (ax_all, ax_day, ax_night):
            cp.style_axes(ax)
            ax.set_ylabel(series.name)
        cp.panel_label(ax_all, "All records")
        ax_all.legend(loc='lower right', bbox_to_anchor=(1, 1), frameon=False, markerscale=3,
                      fontsize=9, ncol=3, borderaxespad=0.2)
        for ax in (ax_all, ax_day):
            ax.tick_params(axis='x', labelbottom=False)
        nice_date_ticks(ax=ax_night)

        subtitle = (f"{n_rej:,} records rejected: {n_rej_day:,} of {n_day:,} daytime "
                    f"({_percent(n_rej_day, n_day)}) and {n_rej_night:,} of {n_night:,} nighttime "
                    f"({_percent(n_rej_night, n_night)})")
        if n_iterations is not None:
            subtitle += f", {n_iterations} iteration{'s' if n_iterations != 1 else ''}"
        cp.figure_title(fig, title or f"{series.name}: outlier test, day and night separately", subtitle)
        fig.show()
        # A detector that draws only this figure still answers `.fig` with what it showed.
        self._fig = fig
