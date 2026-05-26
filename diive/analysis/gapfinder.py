"""
ANALYSIS: GAP DETECTION
=======================

Identify and analyze missing data patterns in time series.
Report gap locations, duration, and statistics for data quality assessment.

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
                pd.Timedelta('1H') / median_delta
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
            record_duration = pd.Timedelta('1H') / self._records_per_hour
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

        cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
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
