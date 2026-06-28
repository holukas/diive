"""
ANALYSIS: GAP DETECTION
=======================

Identify and analyze missing data patterns in time series.
Report gap locations, duration, and statistics for data quality assessment.

Classes:
    GapFinder  — lightweight gap detection; returns per-gap table and headline stats.
    GapStats   — extended analysis built on GapFinder; adds monthly / annual breakdown,
                 long-gap listing, and a three-panel visualization.

Part of the diive library: https://github.com/holukas/diive
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series

from diive.core.plotting.plotfuncs import save_fig
from diive.core.utils.console import console as _console, rule


def _nearest_gap(results: DataFrame, timestamp):
    """Return the gap row containing ``timestamp``, else the nearest one.

    ``results`` must have ``GAP_START`` / ``GAP_END`` columns. ``timestamp`` is
    coerced to a tz-naive :class:`pandas.Timestamp` (gap timestamps taken from a
    series index are tz-naive; an interactive caller may pass a tz-aware value,
    e.g. from a matplotlib date axis). Returns the single matching row as a
    :class:`pandas.Series`, or ``None`` when there are no gaps.
    """
    if results is None or results.empty:
        return None
    ts = pd.Timestamp(timestamp)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    # Distance from ts to each gap interval: 0 when inside, else the gap to the
    # nearer end. before/after are mutually exclusive (one is clipped to 0), so
    # their sum is the signed-free distance; idxmin picks the containing/nearest.
    before = (ts - results['GAP_END']).dt.total_seconds().clip(lower=0)
    after = (results['GAP_START'] - ts).dt.total_seconds().clip(lower=0)
    return results.loc[(before + after).idxmin()]


class GapFinder:
    """Find gaps in time series.

    Detects consecutive missing values (NaN) and reports their locations,
    duration, and statistics. Useful for understanding data quality and
    planning gap-filling strategies.

    Uses a cumulative-sum grouping trick to identify consecutive NaN runs in a
    single vectorized pass — no explicit loops.

    Args:
        series: Time series with a DatetimeIndex
        max_length: Only return gaps with GAP_LENGTH <= max_length (upper size filter)
        min_length: Only return gaps with GAP_LENGTH >= min_length (lower size filter)
        sort_results: Sort results by GAP_LENGTH descending (default True)

    Properties:
        .results: DataFrame with columns GAP_START, GAP_END, GAP_LENGTH, GAP_DURATION.
            GAP_DURATION is NaT when time resolution cannot be inferred.
        .summary: Dict of headline statistics (n_gaps, missing_pct, longest gap, etc.).

    Methods:
        .showfig(): Two-panel figure — availability heatmap and gap-length histogram.
        .gap_at(timestamp): The gap containing a timestamp, or the nearest one.

    Example:
        See `examples/analysis/analysis_gapfinder.py` for complete usage including
        size filtering, summary statistics, and visualization.

    See Also:
        TimestampSanitizer : Handles missing time steps and data gaps at source.
    """

    def __init__(self,
                 series: Series,
                 max_length: int = None,
                 min_length: int = None,
                 sort_results: bool = True):
        """
        Args:
            series: Time series with a DatetimeIndex
            max_length: Only return gaps with GAP_LENGTH <= max_length (upper size filter)
            min_length: Only return gaps with GAP_LENGTH >= min_length (lower size filter)
            sort_results: Sort results by GAP_LENGTH descending (default True)
        """
        self.max_length = max_length
        self.min_length = min_length
        self._series_col = series.name
        self.sort_results = sort_results
        self._series = series.copy()

        if len(series) > 1:
            median_delta = series.index.to_series().diff().median()
            self._records_per_hour = (
                pd.Timedelta('1h') / median_delta
                if pd.notna(median_delta) and median_delta > pd.Timedelta(0)
                else None
            )
        else:
            self._records_per_hour = None

        self._gapfinder_df = self._detect_gaps()

    def _detect_gaps(self) -> DataFrame:
        s = self._series

        if not s.hasnans:
            return pd.DataFrame(columns=['GAP_START', 'GAP_END', 'GAP_LENGTH', 'GAP_DURATION'])

        # notna().cumsum() stays constant across each consecutive NaN run,
        # giving every gap period a unique group id.
        cumsum = s.notna().cumsum()
        is_na = s.isna()
        gap_df = pd.DataFrame({'ts': s.index[is_na], 'cumsum': cumsum[is_na]})

        result = gap_df.groupby('cumsum').agg(
            GAP_START=('ts', 'min'),
            GAP_END=('ts', 'max'),
            GAP_LENGTH=('ts', 'count'),
        ).reset_index(drop=True)

        if self._records_per_hour is not None:
            record_duration = pd.Timedelta('1h') / self._records_per_hour
            result['GAP_DURATION'] = result['GAP_LENGTH'] * record_duration
        else:
            result['GAP_DURATION'] = pd.NaT

        if self.max_length is not None:
            result = result[result['GAP_LENGTH'] <= self.max_length]

        if self.min_length is not None:
            result = result[result['GAP_LENGTH'] >= self.min_length]

        if self.sort_results:
            result = result.sort_values('GAP_LENGTH', ascending=False).reset_index(drop=True)

        return result

    @property
    def results(self) -> DataFrame:
        """Gap detection results as DataFrame.

        Columns: GAP_START, GAP_END, GAP_LENGTH, GAP_DURATION.
        GAP_DURATION is NaT when time resolution cannot be inferred.
        Sorted by gap length descending if sort_results=True.
        """
        return self._gapfinder_df

    @property
    def summary(self) -> dict:
        """Headline gap statistics as a dict.

        Keys: n_gaps, total_missing_records, missing_pct, longest_gap_records,
        median_gap_records, mean_gap_records, longest_gap_duration.
        Counts reflect any max_length / min_length filters applied.
        """
        gaps = self.results
        total = len(self._series)
        missing = int(self._series.isna().sum())
        return {
            'n_gaps': len(gaps),
            'total_missing_records': missing,
            'missing_pct': round(100 * missing / total, 2) if total > 0 else 0.0,
            'longest_gap_records': int(gaps['GAP_LENGTH'].max()) if not gaps.empty else 0,
            'median_gap_records': float(gaps['GAP_LENGTH'].median()) if not gaps.empty else 0.0,
            'mean_gap_records': round(float(gaps['GAP_LENGTH'].mean()), 1) if not gaps.empty else 0.0,
            'longest_gap_duration': gaps['GAP_DURATION'].max() if not gaps.empty else pd.NaT,
        }

    def gap_at(self, timestamp):
        """Return the detected gap containing ``timestamp``, or the nearest one.

        The returned row (a :class:`pandas.Series`) has GAP_START, GAP_END,
        GAP_LENGTH and GAP_DURATION. Returns ``None`` if no gaps were found.
        Handy for interactive tools that map a clicked time to a gap.

        Args:
            timestamp: A datetime / pandas Timestamp (tz-aware values are
                accepted and reduced to tz-naive wall time).
        """
        return _nearest_gap(self._gapfinder_df, timestamp)

    def __str__(self) -> str:
        s = self.summary
        gaps = self.results
        n_total = len(self._series)
        start = self._series.index[0].strftime('%Y-%m-%d')
        end = self._series.index[-1].strftime('%Y-%m-%d')

        w = 70
        sep = '=' * w
        lines = [
            sep,
            f"  GAP ANALYSIS  --  {self._series_col}",
            sep,
            f"  {'Series:':<14}{n_total:>10,} records   ({start} to {end})",
            f"  {'Missing:':<14}{s['total_missing_records']:>10,} records   ({s['missing_pct']:.1f}%)",
            f"  {'Gap periods:':<14}{s['n_gaps']:>10,}",
        ]

        if not gaps.empty:
            longest = gaps.iloc[0]
            dur = s['longest_gap_duration']
            dur_str = str(dur) if pd.notna(dur) else 'unknown'
            gs = longest['GAP_START'].strftime('%Y-%m-%d %H:%M')
            ge = longest['GAP_END'].strftime('%Y-%m-%d %H:%M')
            lines += [
                '',
                f"  {'Longest:':<14}{s['longest_gap_records']:>10,} records   ({dur_str})",
                f"  {'':14}  {gs}  ->  {ge}",
                f"  {'Median:':<14}{s['median_gap_records']:>10.0f} records",
                f"  {'Mean:':<14}{s['mean_gap_records']:>10.1f} records",
            ]

        filters = []
        if self.max_length is not None:
            filters.append(f"max_length={self.max_length}")
        if self.min_length is not None:
            filters.append(f"min_length={self.min_length}")
        lines += [
            '',
            f"  {'Filters:':<14}{'  '.join(filters) if filters else 'none'}",
            sep,
        ]
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def showfig(self,
                saveplot: bool = False,
                title: str = None,
                path: Path | str = None):
        """Display gap analysis visualizations.

        Two-panel figure:
        - Top: daily data availability heatmap (day-of-year × year)
        - Bottom: gap length histogram with duration reference lines

        Args:
            saveplot: Save figure to file
            title: Figure title
            path: Directory for saved figure
        """
        n_years = self._series.index.year.nunique()
        heatmap_h = max(2.0, n_years * 0.42)
        hist_h = 3.2
        fig = plt.figure(figsize=(16, heatmap_h + hist_h + 1.2))
        gs = gridspec.GridSpec(2, 1, height_ratios=[heatmap_h, hist_h])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        self.plot_availability_heatmap(ax=ax1)
        self.plot_gap_length_histogram(ax=ax2)
        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold')
        fig.tight_layout(pad=1.5, h_pad=2.5)
        plt.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)

    def plot_availability_heatmap(self, ax):
        """Heatmap of daily data availability: day-of-year (x) vs year (y).

        Green = fully available, red = all missing. Reveals seasonal and
        annual patterns in data loss at a glance.
        """
        daily_count = self._series.resample('D').count()
        daily_total = self._series.resample('D').size()
        daily_frac = daily_count / daily_total

        df = pd.DataFrame({
            'year': daily_frac.index.year,
            'doy': daily_frac.index.dayofyear,
            'frac': daily_frac.values,
        })
        matrix = df.pivot_table(index='year', columns='doy', values='frac', fill_value=np.nan)
        years = matrix.index.values
        doys = matrix.columns.values

        im = ax.imshow(matrix.values, aspect='auto', cmap='RdYlBu',
                       vmin=0, vmax=1, interpolation='nearest', origin='upper')

        month_doys = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        tick_cols = [int(np.searchsorted(doys, d)) for d in month_doys if d <= doys[-1]]
        ax.set_xticks(tick_cols)
        ax.set_xticklabels(month_names[:len(tick_cols)])

        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        ax.margins(y=0.02)

        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label('Availability', fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0% (all missing)', '50%', '100%'])

        total_days = len(daily_frac)
        missing_days = (daily_frac == 0).sum()
        partial_days = ((daily_frac > 0) & (daily_frac < 1)).sum()
        ax.set_title(
            f"Daily data availability — {self._series_col}  "
            f"({missing_days} fully missing days, {partial_days} partial days out of {total_days})",
            fontsize=10,
        )

    def plot_gap_length_histogram(self, ax):
        """Histogram of gap lengths on a log scale with duration reference lines.

        Duration references (1h, 1 day, 1 week) are inferred from the time
        resolution of the input series.
        """
        gap_lengths = self.results['GAP_LENGTH']
        if gap_lengths.empty:
            ax.text(0.5, 0.5, 'No gaps found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            return

        max_len = int(gap_lengths.max())
        bins = np.logspace(0, np.log10(max_len + 1), 50) if max_len > 1 else np.arange(1, 10)
        ax.hist(gap_lengths, bins=bins, color='#64B5F6', edgecolor='white',
                linewidth=0.5, zorder=2)
        ax.set_xscale('log')

        if self._records_per_hour is not None:
            refs = [
                (self._records_per_hour, '1 h'),
                (self._records_per_hour * 24, '1 day'),
                (self._records_per_hour * 24 * 7, '1 week'),
            ]
            for n_records, label in refs:
                if 1 <= n_records <= max_len * 1.5:
                    ax.axvline(n_records, color='#455A64', linestyle='--',
                               linewidth=1.0, alpha=0.7, zorder=3)
                    ax.text(n_records * 1.06, 0.96, label,
                            transform=ax.get_xaxis_transform(),
                            color='#455A64', fontsize=8, va='top')

        ax.set_xlabel(f'Gap length (records) — {self._series_col}')
        ax.set_ylabel('Number of gaps')
        ax.set_title(
            f"Gap length distribution  |  {len(gap_lengths)} gaps  "
            f"|  {gap_lengths.sum()} missing records total  "
            f"|  longest: {max_len} records",
            fontsize=10,
        )
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4, zorder=0)


class GapStats:
    """Extended gap analysis with seasonal breakdown and long-gap flagging.

    Wraps :class:`GapFinder` and adds monthly statistics, annual coverage,
    identification of long gaps, and a three-panel visualization.

    Use :class:`GapFinder` when you only need per-gap locations and headline
    counts. Use ``GapStats`` when you need seasonal context, annual trends,
    or a report of which specific long gaps occurred.

    Args:
        series: Time series with a DatetimeIndex.
        long_gap_records: Gaps with GAP_LENGTH >= this value are reported
            separately as *long gaps*. Default is 48, which equals one day for
            30-min data. Pass ``None`` to disable long-gap identification.

    Properties:
        .results: Gap table (chronological) with YEAR and MONTH columns added.
        .monthly_stats: Per-calendar-month missing statistics, all years combined.
        .annual_coverage: Per-year data coverage percentages.
        .long_gaps: Subset of .results where GAP_LENGTH >= long_gap_records.
        .summary: Extended headline statistics dict.

    Methods:
        .gap_at(timestamp): The gap containing a timestamp, or the nearest one
            (row includes YEAR / MONTH).
        .report(): Rich-formatted console report — rules, coloured tables,
            long-gap list, monthly and annual breakdowns.
        .showfig(): Four-panel figure — RdYlGn availability heatmap, gap spike
            timeline, monthly polar chart, gap-length histogram.
        .plot_availability_heatmap(ax): Heatmap panel for custom figures.
        .plot_gap_spike_timeline(ax): Gap spike panel for custom figures.
        .plot_monthly_polar(ax): Polar monthly chart (requires polar axes).
        .plot_monthly_gap_distribution(ax): Flat bar chart alternative to polar.

    Example:
        See ``examples/analysis/analysis_gapstats.py`` for complete usage.

        Typical flux-chain usage after L3.3::

            from diive.analysis import GapStats

            for scen, series in data.levels.filteredseries_level33_qcf.items():
                gs = GapStats(series, long_gap_records=48)
                print(gs)
                gs.showfig(title=f"Gap analysis — {scen}")

    See Also:
        GapFinder : Lightweight gap detection (used internally by GapStats).
    """

    def __init__(self,
                 series: Series,
                 long_gap_records: int = 48):
        """Compute gap statistics for a series. See the class docstring for parameters."""
        self._series = series.copy()
        self._series_col = series.name
        self.long_gap_records = long_gap_records

        # GapFinder in chronological order so YEAR/MONTH enrichment is stable
        self._gf = GapFinder(series, sort_results=False)
        self._results = self._enrich_results()
        self._monthly_stats = self._compute_monthly_stats()
        self._annual_coverage = self._compute_annual_coverage()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enrich_results(self) -> DataFrame:
        df = self._gf.results.copy()
        if df.empty:
            df['YEAR'] = pd.Series(dtype='int64')
            df['MONTH'] = pd.Series(dtype='int64')
            return df
        df['YEAR'] = df['GAP_START'].dt.year
        df['MONTH'] = df['GAP_START'].dt.month
        return df

    def _compute_monthly_stats(self) -> DataFrame:
        s = self._series
        monthly = pd.DataFrame({
            'total': s.resample('ME').size(),
            'missing': s.isna().resample('ME').sum(),
        })
        monthly['month'] = monthly.index.month

        agg = monthly.groupby('month').agg(
            total_records=('total', 'sum'),
            missing_records=('missing', 'sum'),
        )
        agg['missing_pct'] = (
            100 * agg['missing_records'] / agg['total_records'].clip(lower=1)
        ).round(2)

        if not self._results.empty:
            gap_counts = (
                self._results.groupby('MONTH').size()
                .reindex(range(1, 13), fill_value=0)
            )
        else:
            gap_counts = pd.Series(0, index=range(1, 13))

        agg['n_gaps'] = gap_counts.reindex(agg.index, fill_value=0)
        agg.index.name = 'MONTH'
        return agg

    def _compute_annual_coverage(self) -> DataFrame:
        s = self._series
        df = pd.DataFrame({
            'total_records': s.resample('YE').size(),
            'valid_records': s.resample('YE').count(),
        })
        df['missing_records'] = df['total_records'] - df['valid_records']
        df['coverage_pct'] = (
            100 * df['valid_records'] / df['total_records'].clip(lower=1)
        ).round(2)
        df.index = df.index.year
        df.index.name = 'YEAR'
        return df

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def results(self) -> DataFrame:
        """Gap table sorted chronologically, with YEAR and MONTH columns.

        Columns: GAP_START, GAP_END, GAP_LENGTH, GAP_DURATION, YEAR, MONTH.
        """
        return self._results

    @property
    def monthly_stats(self) -> DataFrame:
        """Per-calendar-month gap statistics, all years combined.

        Columns: total_records, missing_records, missing_pct, n_gaps.
        Index: MONTH (1-12).
        """
        return self._monthly_stats

    @property
    def annual_coverage(self) -> DataFrame:
        """Per-year data coverage.

        Columns: total_records, valid_records, missing_records, coverage_pct.
        Index: YEAR.
        """
        return self._annual_coverage

    @property
    def long_gaps(self) -> DataFrame:
        """Gaps with GAP_LENGTH >= long_gap_records, sorted by length descending.

        Empty DataFrame if long_gap_records is None or no qualifying gaps exist.
        """
        if self.long_gap_records is None or self._results.empty:
            return self._results.iloc[0:0].copy()
        df = self._results[self._results['GAP_LENGTH'] >= self.long_gap_records]
        return df.sort_values('GAP_LENGTH', ascending=False).reset_index(drop=True)

    @property
    def summary(self) -> dict:
        """Extended headline statistics dict.

        Includes all keys from :attr:`GapFinder.summary` plus:
        ``n_long_gaps``, ``long_gap_records_threshold``, ``worst_month`` (1-12
        or None), ``worst_month_missing_pct``, ``annual_coverage`` (dict mapping
        year to coverage_pct).
        """
        base = self._gf.summary
        mon = self._monthly_stats
        ann = self._annual_coverage
        if not mon.empty:
            worst_month = int(mon['missing_pct'].idxmax())
            worst_pct = float(mon.loc[worst_month, 'missing_pct'])
        else:
            worst_month = None
            worst_pct = 0.0
        return {
            **base,
            'n_long_gaps': len(self.long_gaps),
            'long_gap_records_threshold': self.long_gap_records,
            'worst_month': worst_month,
            'worst_month_missing_pct': worst_pct,
            'annual_coverage': ann['coverage_pct'].to_dict() if not ann.empty else {},
        }

    def gap_at(self, timestamp):
        """Return the gap containing ``timestamp``, or the nearest one.

        The returned row (a :class:`pandas.Series`) includes the YEAR / MONTH
        columns alongside GAP_START, GAP_END, GAP_LENGTH and GAP_DURATION.
        Returns ``None`` if no gaps were found. Handy for interactive tools that
        map a clicked time to a gap.

        Args:
            timestamp: A datetime / pandas Timestamp (tz-aware values are
                accepted and reduced to tz-naive wall time).
        """
        return _nearest_gap(self._results, timestamp)

    # ------------------------------------------------------------------
    # Text representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        s = self.summary
        n_total = len(self._series)
        start = self._series.index[0].strftime('%Y-%m-%d')
        end = self._series.index[-1].strftime('%Y-%m-%d')

        _MONTH_NAMES = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        w = 72
        sep = '=' * w
        lines = [
            sep,
            f"  GAP ANALYSIS  --  {self._series_col}",
            sep,
            f"  {'Series:':<20}{n_total:>10,} records   ({start} to {end})",
            f"  {'Missing:':<20}{s['total_missing_records']:>10,} records   ({s['missing_pct']:.1f}%)",
            f"  {'Gap periods:':<20}{s['n_gaps']:>10,}",
            f"  {'Long gaps:':<20}{s['n_long_gaps']:>10,}"
            + (f"   (>= {s['long_gap_records_threshold']} records)"
               if s['long_gap_records_threshold'] else ""),
        ]

        # Long gaps table (top 10)
        lg = self.long_gaps
        if not lg.empty:
            thr = self.long_gap_records
            lines += ['', f"  LONG GAPS  (>= {thr} records):"]
            lines.append(
                f"  {'#':<5}{'Start':<22}{'End':<22}{'Records':>9}  {'Duration'}"
            )
            lines.append('  ' + '-' * (w - 4))
            for rank, (_, row) in enumerate(lg.head(10).iterrows(), start=1):
                gs_str = row['GAP_START'].strftime('%Y-%m-%d %H:%M')
                ge_str = row['GAP_END'].strftime('%Y-%m-%d %H:%M')
                dur = str(row['GAP_DURATION']) if pd.notna(row['GAP_DURATION']) else 'n/a'
                lines.append(
                    f"  {rank:<5}{gs_str:<22}{ge_str:<22}"
                    f"{int(row['GAP_LENGTH']):>9,}  {dur}"
                )
            if len(lg) > 10:
                lines.append(f"  ... ({len(lg) - 10} more long gaps not shown)")

        # Monthly stats table
        mon = self._monthly_stats
        if not mon.empty:
            lines += ['', '  MONTHLY MISSING  (all years combined):']
            lines.append(
                f"  {'Month':<8}{'Missing%':>10}  {'Records missing':>16}  {'Gap periods':>12}"
            )
            lines.append('  ' + '-' * 52)
            for m in range(1, 13):
                if m in mon.index:
                    r = mon.loc[m]
                    lines.append(
                        f"  {_MONTH_NAMES[m]:<8}{r['missing_pct']:>9.1f}%"
                        f"  {int(r['missing_records']):>16,}"
                        f"  {int(r['n_gaps']):>12}"
                    )

        # Annual coverage (only when multi-year)
        ann = self._annual_coverage
        if not ann.empty and len(ann) > 1:
            lines += ['', '  ANNUAL COVERAGE:']
            lines.append(
                f"  {'Year':<7}{'Coverage%':>10}  {'Valid':>8}  {'Missing':>8}"
            )
            lines.append('  ' + '-' * 38)
            for yr, row in ann.iterrows():
                lines.append(
                    f"  {yr:<7}{row['coverage_pct']:>9.1f}%"
                    f"  {int(row['valid_records']):>8,}"
                    f"  {int(row['missing_records']):>8,}"
                )

        lines += ['', sep]
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    # ------------------------------------------------------------------
    # Rich report
    # ------------------------------------------------------------------

    def report(self):
        """Rich-formatted console report of gap statistics.

        Prints colour-coded sections using the diive Rich console:
        - Overview (missing%, gap count, long-gap count)
        - Long gaps table (sorted longest-first, top 20)
        - Monthly missing table (all years combined)
        - Annual coverage table (only when series spans > 1 year)
        """
        from rich.table import Table

        s = self.summary
        n_total = len(self._series)
        start = self._series.index[0].strftime('%Y-%m-%d')
        end = self._series.index[-1].strftime('%Y-%m-%d')

        _MONTH_NAMES = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # ---- Overview ----
        rule(f"Gap Analysis  --  {self._series_col}")
        _console.print(
            f"  Series        {n_total:>10,} records   "
            f"({start}  ->  {end})\n"
            f"  Missing       {s['total_missing_records']:>10,} records   "
            f"([bold]{s['missing_pct']:.1f}%[/bold])\n"
            f"  Gap periods   {s['n_gaps']:>10,}\n"
            f"  Long gaps     {s['n_long_gaps']:>10,}"
            + (f"   (>= {s['long_gap_records_threshold']} records)"
               if s['long_gap_records_threshold'] else "")
        )

        # ---- Long gaps ----
        lg = self.long_gaps
        if not lg.empty:
            rule(f"Long Gaps  (>= {self.long_gap_records} records)",
                 min_level=2)
            tbl = Table(show_header=True, header_style="bold cyan",
                        box=None, padding=(0, 2))
            tbl.add_column("#", justify="right", style="dim")
            tbl.add_column("Start", no_wrap=True)
            tbl.add_column("End", no_wrap=True)
            tbl.add_column("Records", justify="right")
            tbl.add_column("Duration")
            for rank, (_, row) in enumerate(lg.head(20).iterrows(), start=1):
                gs_str = row['GAP_START'].strftime('%Y-%m-%d %H:%M')
                ge_str = row['GAP_END'].strftime('%Y-%m-%d %H:%M')
                dur = str(row['GAP_DURATION']) if pd.notna(row['GAP_DURATION']) else 'n/a'
                tbl.add_row(str(rank), gs_str, ge_str,
                            f"{int(row['GAP_LENGTH']):,}", dur)
            _console.print(tbl)
            if len(lg) > 20:
                _console.print(f"  [dim]... {len(lg) - 20} more long gaps not shown[/dim]")

        # ---- Monthly ----
        mon = self._monthly_stats
        if not mon.empty:
            rule("Monthly Missing  (all years combined)", min_level=2)
            tbl = Table(show_header=True, header_style="bold cyan",
                        box=None, padding=(0, 2))
            tbl.add_column("Month")
            tbl.add_column("Missing%", justify="right")
            tbl.add_column("Records missing", justify="right")
            tbl.add_column("Gap periods", justify="right")
            worst_m = int(mon['missing_pct'].idxmax())
            for m in range(1, 13):
                if m not in mon.index:
                    continue
                r = mon.loc[m]
                pct_str = f"{r['missing_pct']:.1f}%"
                name = _MONTH_NAMES[m]
                if m == worst_m:
                    name = f"[bold yellow]{name}[/bold yellow]"
                    pct_str = f"[bold yellow]{pct_str}[/bold yellow]"
                tbl.add_row(name, pct_str,
                            f"{int(r['missing_records']):,}",
                            str(int(r['n_gaps'])))
            _console.print(tbl)

        # ---- Annual coverage ----
        ann = self._annual_coverage
        if not ann.empty and len(ann) > 1:
            rule("Annual Coverage", min_level=2)
            tbl = Table(show_header=True, header_style="bold cyan",
                        box=None, padding=(0, 2))
            tbl.add_column("Year")
            tbl.add_column("Coverage%", justify="right")
            tbl.add_column("Valid", justify="right")
            tbl.add_column("Missing", justify="right")
            worst_yr = int(ann['coverage_pct'].idxmin())
            for yr, row in ann.iterrows():
                cov_str = f"{row['coverage_pct']:.1f}%"
                yr_str = str(yr)
                if yr == worst_yr:
                    yr_str = f"[bold yellow]{yr_str}[/bold yellow]"
                    cov_str = f"[bold yellow]{cov_str}[/bold yellow]"
                tbl.add_row(yr_str, cov_str,
                            f"{int(row['valid_records']):,}",
                            f"{int(row['missing_records']):,}")
            _console.print(tbl)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def showfig(self,
                saveplot: bool = False,
                title: str = None,
                path: Path | str = None):
        """Four-panel gap analysis figure.

        Layout (top to bottom):
        - Row 1 (full width): daily availability heatmap — green = data, red = gap
        - Row 2 (full width): gap spike timeline — each gap as a coloured
          vertical line, tall spikes mark long gaps
        - Row 3 left: monthly missing-data polar chart (Jan at top, clockwise)
        - Row 3 right: gap-length histogram (log scale)

        Args:
            saveplot: Save figure to file.
            title: Figure title.
            path: Directory for saved figure.
        """
        n_years = self._series.index.year.nunique()
        heatmap_h = max(2.5, n_years * 0.52)
        timeline_h = 3.2
        bottom_h = 3.8

        fig = plt.figure(
            figsize=(16, heatmap_h + timeline_h + bottom_h + 1.8),
            constrained_layout=True,
        )
        gs = gridspec.GridSpec(
            3, 2,
            height_ratios=[heatmap_h, timeline_h, bottom_h],
            figure=fig,
        )
        ax_hm = fig.add_subplot(gs[0, :])
        ax_tl = fig.add_subplot(gs[1, :])
        ax_po = fig.add_subplot(gs[2, 0], projection='polar')
        ax_hi = fig.add_subplot(gs[2, 1])

        self.plot_availability_heatmap(ax=ax_hm)
        self.plot_gap_spike_timeline(ax=ax_tl)
        self.plot_monthly_polar(ax=ax_po)
        self._gf.plot_gap_length_histogram(ax=ax_hi)

        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.show()
        if saveplot:
            save_fig(fig=fig, title=title, path=path)

    def plot_availability_heatmap(self, ax):
        """Daily availability heatmap: day-of-year (x) vs year (y).

        Green = fully available, red = all missing. Uses RdYlGn colormap with
        white year-separator lines and subtle month grid lines.

        Args:
            ax: Matplotlib Axes to draw on.
        """
        daily_count = self._series.resample('D').count()
        daily_total = self._series.resample('D').size()
        daily_frac = daily_count / daily_total

        df = pd.DataFrame({
            'year': daily_frac.index.year,
            'doy': daily_frac.index.dayofyear,
            'frac': daily_frac.values,
        })
        matrix = df.pivot_table(
            index='year', columns='doy', values='frac', fill_value=np.nan
        )
        years = matrix.index.values
        doys = matrix.columns.values

        im = ax.imshow(matrix.values, aspect='auto', cmap='RdYlBu',
                       vmin=0, vmax=1, interpolation='nearest', origin='upper')

        # White separators between years
        for i in range(len(years) - 1):
            ax.axhline(i + 0.5, color='white', linewidth=2.0, alpha=0.9)

        month_doys = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        tick_cols = [int(np.searchsorted(doys, d)) for d in month_doys if d <= doys[-1]]
        for col_idx in tick_cols[1:]:
            ax.axvline(col_idx - 0.5, color='white', linewidth=0.5, alpha=0.35)

        ax.set_xticks(tick_cols)
        ax.set_xticklabels(month_names[:len(tick_cols)], fontsize=9)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years, fontsize=9)
        ax.margins(y=0.02)

        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label('Daily availability', fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0% (all missing)', '50%', '100% (full)'])

        total_days = len(daily_frac)
        missing_days = int((daily_frac == 0).sum())
        partial_days = int(((daily_frac > 0) & (daily_frac < 1)).sum())
        ax.set_title(
            f"Daily availability  --  {self._series_col}  "
            f"|  {missing_days} fully missing days  "
            f"|  {partial_days} partial  "
            f"|  {total_days} total days",
            fontsize=10, pad=6,
        )

    def plot_gap_spike_timeline(self, ax):
        """Gap spike chart: each gap as a coloured vertical line on a time axis.

        Short gaps form a low-amplitude base layer; long gaps spike upward in
        deep red. A dashed threshold line marks long_gap_records. The top-3
        longest gaps are annotated.

        Args:
            ax: Matplotlib Axes to draw on.
        """
        import matplotlib.dates as mdates
        from matplotlib.collections import LineCollection

        gaps = self._results
        if gaps.empty:
            ax.text(0.5, 0.5, 'No gaps found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            return

        lengths = gaps['GAP_LENGTH'].values.astype(float)
        dates_num = mdates.date2num(gaps['GAP_START'].dt.to_pydatetime())

        # Build segments as numpy array — fast even for tens of thousands of gaps
        n = len(dates_num)
        segs = np.zeros((n, 2, 2))
        segs[:, 0, 0] = dates_num   # x: gap start (both ends same -> vertical)
        segs[:, 1, 0] = dates_num
        segs[:, 1, 1] = lengths     # y: 0 -> gap length

        max_len = float(lengths.max())
        norm = plt.Normalize(vmin=1, vmax=max_len)
        lc = LineCollection(segs, cmap='YlOrRd', norm=norm,
                            linewidths=1.0, alpha=0.85)
        lc.set_array(lengths)
        ax.add_collection(lc)

        ax.set_xlim(dates_num.min() - 15, dates_num.max() + 15)
        ax.set_ylim(0, max_len * 1.15)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

        # Long-gap threshold line
        if self.long_gap_records is not None:
            ax.axhline(self.long_gap_records, color='#E53935', linewidth=1.2,
                       linestyle='--', alpha=0.9, zorder=5)
            ax.text(
                dates_num.max() + 18, self.long_gap_records,
                f'{self.long_gap_records} rec.',
                color='#E53935', fontsize=8, va='center',
            )

        # Annotate top-3 longest gaps
        top3 = self.long_gaps.head(3)
        for _, row in top3.iterrows():
            d = mdates.date2num(row['GAP_START'].to_pydatetime())
            lv = float(row['GAP_LENGTH'])
            label = f"{int(lv):,}\n{row['GAP_START'].strftime('%b %Y')}"
            ax.annotate(
                label,
                xy=(d, lv),
                xytext=(d, lv + max_len * 0.07),
                fontsize=7, ha='center', color='#B71C1C',
                arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=0.8),
            )

        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label('Gap length (records)', fontsize=8)

        n_long = len(self.long_gaps)
        ax.set_ylabel('Gap length (records)')
        ax.set_title(
            f"Gap timeline  --  {self._series_col}  "
            f"|  {len(gaps):,} gaps total  "
            f"|  {n_long} long gaps"
            + (f"  (>= {self.long_gap_records} records)" if self.long_gap_records else ""),
            fontsize=10,
        )
        ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)

    def plot_monthly_polar(self, ax):
        """Polar bar chart of monthly missing-data percentage.

        12 bars radiate clockwise from January at the top. Bar length encodes
        missing% for that calendar month (all years combined). Color intensity
        follows the same scale: light yellow = low missing, deep red = high.

        Args:
            ax: Matplotlib polar Axes (created with ``projection='polar'``).
        """
        _MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        mon = self._monthly_stats

        if mon.empty:
            return

        N = 12
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pcts = np.array([
            float(mon.loc[m, 'missing_pct']) if m in mon.index else 0.0
            for m in range(1, 13)
        ])
        width = 2 * np.pi / N * 0.82

        max_pct = max(float(pcts.max()), 1.0)
        norm = plt.Normalize(0, max_pct)
        colors = [plt.cm.YlOrRd(norm(p)) for p in pcts]

        ax.bar(theta, pcts, width=width, bottom=0.0,
               color=colors, alpha=0.88, edgecolor='white', linewidth=0.8, zorder=3)

        ax.set_theta_zero_location('N')   # Jan at 12 o'clock
        ax.set_theta_direction(-1)        # clockwise
        ax.set_xticks(theta)
        ax.set_xticklabels(_MONTH_NAMES, fontsize=8)
        ax.set_yticklabels([])
        ax.set_ylim(0, max_pct * 1.35)
        ax.grid(True, alpha=0.25)

        # Percentage labels above each bar
        for angle, pct in zip(theta, pcts):
            if pct >= max_pct * 0.05:
                ax.text(angle, pct + max_pct * 0.08,
                        f'{pct:.0f}%',
                        ha='center', va='bottom', fontsize=7, color='#333333')

        _MN = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        worst_m = int(mon['missing_pct'].idxmax())
        ax.set_title(
            f"Monthly missing %\n"
            f"worst: {_MN[worst_m]} ({mon.loc[worst_m, 'missing_pct']:.1f}%)",
            fontsize=9, pad=15,
        )

    def plot_monthly_gap_distribution(self, ax):
        """Monthly missing-data bar chart with gap-count annotations.

        Suitable for embedding in custom multi-panel figures. For the full
        built-in figure use :meth:`showfig`, which uses the polar variant.

        Bar height = percentage of records missing for that calendar month
        (all years combined). Bar annotation = number of gap periods.

        Args:
            ax: Matplotlib Axes to draw on.
        """
        _MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        mon = self._monthly_stats

        if mon.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            return

        x = np.arange(1, 13)
        pcts = [float(mon.loc[m, 'missing_pct']) if m in mon.index else 0.0 for m in x]
        n_gaps = [int(mon.loc[m, 'n_gaps']) if m in mon.index else 0 for m in x]

        bars = ax.bar(x, pcts, color='#64B5F6', edgecolor='white', linewidth=0.5, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(_MONTH_NAMES)
        ax.set_ylabel('Missing data (%)')
        ax.set_ylim(0, max(max(pcts) * 1.2, 5.0))
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4, zorder=0)

        for bar, ng in zip(bars, n_gaps):
            if ng > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(ng),
                    ha='center', va='bottom', fontsize=8, color='#455A64',
                )

        worst_m = int(mon['missing_pct'].idxmax())
        worst_str = (f"worst: {_MONTH_NAMES[worst_m - 1]}"
                     f" ({mon.loc[worst_m, 'missing_pct']:.1f}%)")
        ax.set_title(
            f"Monthly missing data  --  {self._series_col}  |  {worst_str}  "
            f"(bar annotations = number of gap periods)",
            fontsize=10,
        )


def gapstats_to_code(
        varname: str,
        *,
        long_gap_records: int = 48,
        df_name: str = 'df',
        with_plot: bool = True,
) -> str:
    """Render a runnable :class:`GapStats` snippet (summary + availability/timeline).

    Mirrors the GUI's Gaps & coverage tab: build :class:`GapStats`, print the
    summary and the long-gap table, and (optionally) draw the availability
    heatmap over the gap/spike timeline. Belongs in the library (not the GUI):
    it encodes the exact call shape and must stay correct as that API evolves;
    the GUI only calls it (the GUI <-> library separation rule).

    Args:
        varname: column to analyse.
        long_gap_records: gaps with at least this many missing records are
            reported as long gaps (passed to :class:`GapStats`).
        df_name: variable name used for the input DataFrame.
        with_plot: also emit the availability-heatmap + timeline plot lines.

    Returns:
        A runnable Python snippet as a string.
    """
    lines = ["import diive as dv", ""]
    if with_plot:
        lines = ["import matplotlib.pyplot as plt", *lines]
    lines += [
        "gs = dv.analysis.GapStats(",
        f"    {df_name}[{varname!r}],",
        f"    long_gap_records={long_gap_records!r},",
        ")",
        "print(gs.summary)",
        "print(gs.long_gaps)",
    ]
    if with_plot:
        lines += [
            "",
            "fig, (ax_hm, ax_tl) = plt.subplots(2, 1, figsize=(12, 7))",
            "gs.plot_availability_heatmap(ax=ax_hm)",
            "gs.plot_gap_spike_timeline(ax=ax_tl)",
            "plt.show()",
        ]
    return "\n".join(lines) + "\n"
