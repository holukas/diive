"""
RELATIVEHUMIDITY_OFFSET_PLOT: DIAGNOSTIC FIGURE OF THE RELATIVE HUMIDITY OFFSET CORRECTION
=========================================================================================

One figure that shows what ``remove_relativehumidity_offset`` did and how well it fits.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series

from diive.preprocessing.corrections._figure import (
    C_CORRECTED, C_FILLED, C_FIRST, C_INK, C_MEASURED, C_OFFSET, C_SUBTRACTED, C_TITLE, C_ZERO, date_axis,
    day_steps, draw_steps, draw_summary, fmt, legend, panel_title, style_axes)
from diive.preprocessing.corrections.offsetcorrection import RelativeHumidityOffsetResult

_LIMIT = 100  # relative humidity cannot exceed 100 %


class RelativeHumidityOffsetPlot:
    """Diagnostic figure of the relative humidity offset correction.

    Two-phase like the other diive plots: the constructor takes the result of
    ``relativehumidity_offset_diagnostics``, ``plot()`` draws. A strip at the
    top shows the five steps of the correction in order, each with its count:

    1. Find the values above 100 %.
    2. Daily offset: for each day with such values, their mean minus 100.
       Every record of the day gets it.
    3. Other days get the offset interpolated linearly, record by record,
       between the nearest days with values above 100 %, and carried before
       the first and after the last such day. Without any value above 100 %
       the offset is 0.
    4. The offset is subtracted from every record.
    5. Values still above 100 % are set to 100 %.

    Panels below the strip, titled with the steps they show:

    - Series: measured and corrected high-res data, with the 100 % line.
    - Offset: the offset of each record. Days with values above 100 % are
      drawn as one value per day with the standard deviation of their values
      above 100 % as a band, the interpolated and carried offset as a dashed
      line.
    - Near 100 %: distribution of the upper range, measured, after
      subtracting the offset and final. A pile just below 100 % without
      values above it means the sensor or logger caps the readings, so the
      correction cannot find an offset.
    - Daily maximum: before and after the correction, with the 100 % line.
    - Summary: records, values above, at and just below 100 %, offset
      statistics, capped records, and mean, min and max before and after.

    Example:
        >>> import diive as dv, numpy as np, pandas as pd
        >>> idx = pd.date_range('2024-06-01', periods=48 * 10, freq='30min', name='TIMESTAMP_MIDDLE')
        >>> rh = pd.Series(80 + 22 * np.sin(np.arange(len(idx)) / 48 * 2 * np.pi), index=idx, name='RH')
        >>> res = dv.corrections.relativehumidity_offset_diagnostics(rh)
        >>> fig = dv.corrections.RelativeHumidityOffsetPlot(res).plot(showplot=False)
    """

    def __init__(self, result: RelativeHumidityOffsetResult, records: Series | None = None):
        """Prepare the figure data.

        Args:
            result: Output of ``relativehumidity_offset_diagnostics``.
            records: Optional boolean Series on the index of *result*, True for
                rows that are records. Rows that are no record (empty grid slots
                between coarse records in meteo screening) are left out of the
                line plots, so lines join neighbouring records. Missing records
                stay NaN and break the lines.
        """
        self.result = result
        self.name = str(result.input.name) if result.input.name is not None else 'value'
        keep = records.reindex(result.input.index).fillna(False).astype(bool) if records is not None else None
        self._lines = (result.input, result.corrected, result.offset) if keep is None else \
            (result.input[keep], result.corrected[keep], result.offset[keep])
        self.daily = self._daily_stats()

    def _daily_stats(self) -> pd.DataFrame:
        """Per calendar day: offset, SD and count of the values above 100, daily max before and after."""
        r = self.result
        days = r.input.index.normalize()
        above = r.input.where(r.above_100)
        grouped = above.groupby(days)
        return pd.DataFrame({'offset': r.daily_offset.groupby(days).first(),
                             'sd_above': grouped.std(), 'n_above': grouped.count(),
                             'max_before': r.input.groupby(days).max(),
                             'max_after': r.corrected.groupby(days).max()})

    def plot(self, title: str = None, figsize: tuple = (14, 11.5), showplot: bool = True):
        """Draw the figure. Returns the matplotlib Figure.

        Args:
            title: Figure title. Default: 'Relative humidity offset: <variable>'.
            figsize: Figure size in inches.
            showplot: Show the figure.
        """
        fig = plt.figure(figsize=figsize, layout='constrained', facecolor='white')
        gs = fig.add_gridspec(4, 3, height_ratios=[0.42, 1.0, 0.9, 1.15], width_ratios=[1.0, 1.25, 1.0])
        ax_steps = fig.add_subplot(gs[0, :])
        ax_series = fig.add_subplot(gs[1, :])
        ax_offset = fig.add_subplot(gs[2, :], sharex=ax_series)
        ax_hist = fig.add_subplot(gs[3, 0])
        ax_max = fig.add_subplot(gs[3, 1])
        ax_text = fig.add_subplot(gs[3, 2])

        draw_steps(ax_steps, self._steps())
        self._plot_series(ax_series)
        self._plot_offset(ax_offset)
        self._plot_hist(ax_hist)
        self._plot_daily_max(ax_max)
        draw_summary(ax_text, self._summary_rows())
        for ax in (ax_series, ax_offset, ax_hist, ax_max):
            style_axes(ax)
        date_axis(ax_offset)
        date_axis(ax_max)
        ax_series.tick_params(labelbottom=False)
        fig.suptitle(title or f"Relative humidity offset: {self.name}", x=0.01, ha='left',
                     fontsize=15, fontweight='bold', color=C_TITLE)
        self.fig = fig
        if showplot:
            fig.show()
        return fig

    def _counts(self) -> dict:
        """Numbers shown in the step strip and the summary."""
        r = self.result
        d = self.daily
        measured_days = d['n_above'] > 0
        return dict(n_records=int(r.input.notna().sum()), n_above=r.n_above_100_before,
                    days_above=int(measured_days.sum()), days_other=int((~measured_days).sum()),
                    n_capped=r.n_capped, offsets=d['offset'][measured_days],
                    n_at_limit=int((r.input == _LIMIT).sum()),
                    n_near_limit=int(((r.input >= _LIMIT - 0.1) & (r.input < _LIMIT)).sum()))

    def _steps(self) -> list[tuple[str, str, str, str]]:
        """Colour, title, description and count of each correction step, in order."""
        r = self.result
        c = self._counts()
        before, subtracted = r.input.dropna(), r.corrected_by_offset.dropna()
        if c['days_above']:
            offset_count = f"{c['days_above']:,} days, median {fmt(c['offsets'].median())}"
            fill_count = f"{c['days_other']:,} days"
        else:
            offset_count = 'none: offset 0'
            fill_count = 'offset 0 on all days'
        return [
            (C_FIRST, 'Find > 100 %', 'values above 100 %',
             f"{c['n_above']:,} records on {c['days_above']:,} days"),
            (C_OFFSET, 'Daily offset', "mean of the day's values\nabove 100 % minus 100", offset_count),
            (C_FILLED, 'Other days', 'linear between those days,\ncarried before and after', fill_count),
            (C_SUBTRACTED, 'Subtract', 'offset from every record',
             f"mean {fmt(before.mean())} → {fmt(subtracted.mean())}"),
            (C_CORRECTED, 'Cap at 100 %', 'values still above 100 %\nset to 100', f"{c['n_capped']:,} records"),
        ]

    def _plot_series(self, ax):
        measured, corrected, _ = self._lines
        ax.plot(measured.index, measured.to_numpy(), color=C_MEASURED, lw=0.6, label='measured', zorder=2)
        ax.plot(corrected.index, corrected.to_numpy(), color=C_CORRECTED, lw=0.6, alpha=0.9,
                label='corrected', zorder=3)
        ax.axhline(_LIMIT, color=C_ZERO, lw=0.8, ls='--', zorder=1)
        ax.set_ylabel(self.name, color=C_INK, fontsize=10)
        ax.set_xmargin(0)
        legend(ax, loc='lower left', ncol=2)
        panel_title(ax, "Series", "measured and corrected high-resolution data, 100 % dashed")

    def _plot_offset(self, ax):
        d = self.daily
        _, _, offset = self._lines
        measured = d['n_above'] > 0
        # Every record's offset: interpolated and carried between the measured days
        ax.plot(offset.index, offset.to_numpy(), color=C_FILLED, lw=1.0, ls='--',
                label='3: interpolated or carried')
        if measured.any():
            values = d['offset'].where(measured)
            sx, sy = day_steps(values)
            sd = day_steps(d['sd_above'].where(measured))[1]
            ax.fill_between(sx, sy - sd, sy + sd, color=C_OFFSET, alpha=0.18, lw=0,
                            label='±1 SD of the values above 100 %')
            ax.plot(sx, sy, color=C_OFFSET, lw=1.4, label='2: daily offset, constant per day')
            median = float(d['offset'][measured].median())
            ax.axhline(median, color=C_INK, lw=0.8, ls=':', label=f'median {fmt(median)}')
        else:
            ax.text(0.5, 0.6, 'no value above 100 %, so the offset is 0', transform=ax.transAxes,
                    ha='center', va='center', color=C_INK, fontsize=10)
        ax.axhline(0, color=C_ZERO, lw=0.8)
        ax.set_ylabel('offset', color=C_INK, fontsize=10)
        legend(ax, loc='best', ncol=4)
        panel_title(ax, "Offset  ·  steps 1-3",
                    "mean of each day's values above 100 % minus 100, linear between those days")

    def _plot_hist(self, ax):
        r = self.result
        values = r.input.dropna()
        if values.empty:
            ax.text(0.5, 0.5, 'no values', ha='center', va='center', transform=ax.transAxes, color=C_INK)
            panel_title(ax, "Near 100 %  ·  steps 4-5")
            return
        lo = min(float(np.nanpercentile(values, 75)), _LIMIT - 5)
        hi = max(float(values.max()), _LIMIT + 1)
        bins = np.linspace(lo, hi, 70)
        ax.hist(values[values >= lo], bins=bins, color=C_MEASURED, alpha=0.8, label='measured')
        sub = r.corrected_by_offset.dropna()
        ax.hist(sub[sub >= lo], bins=bins, histtype='step', color=C_SUBTRACTED, lw=1.3,
                label='4: offset subtracted')
        fin = r.corrected.dropna()
        ax.hist(fin[fin >= lo], bins=bins, histtype='step', color=C_CORRECTED, lw=1.3,
                label='5: capped at 100 %')
        ax.axvline(_LIMIT, color=C_ZERO, lw=0.8, ls='--')
        ax.set_xlabel(self.name, color=C_INK, fontsize=9)
        ax.set_ylabel('records', color=C_INK, fontsize=9)
        legend(ax, loc='upper left')
        panel_title(ax, "Near 100 %  ·  steps 4-5", "upper range")

    def _plot_daily_max(self, ax):
        d = self.daily
        x = d.index + pd.Timedelta('12h')
        ax.plot(x, d['max_before'], color=C_MEASURED, lw=1.2, label='measured')
        ax.plot(x, d['max_after'], color=C_CORRECTED, lw=1.2, label='corrected')
        ax.axhline(_LIMIT, color=C_ZERO, lw=0.8, ls='--')
        ax.set_ylabel(self.name, color=C_INK, fontsize=9)
        legend(ax, loc='lower left', ncol=2)
        panel_title(ax, "Daily maximum")

    def _summary_rows(self) -> list[tuple[str, str]]:
        """Label and value of each summary line."""
        r = self.result
        c = self._counts()
        before, after = r.input.dropna(), r.corrected.dropna()
        off = c['offsets']
        if off.empty:
            offset_rows = [('Offset', '0 (no value above 100 %)')]
        else:
            offset_rows = [('Offset median', fmt(off.median())),
                           ('Offset mean ± SD', f"{fmt(off.mean())} ± {fmt(off.std())}"),
                           ('Offset min / max', f"{fmt(off.min())} / {fmt(off.max())}")]
        return [
            ('Records', f"{c['n_records']:,}"),
            ('Above 100 %', f"{c['n_above']:,} on {c['days_above']:,} days"),
            ('At exactly 100 %', f"{c['n_at_limit']:,}"),
            ('From 99.9 to 100 %', f"{c['n_near_limit']:,}"),
            ('Days with interpolated offset', f"{c['days_other'] if c['days_above'] else 0:,}"),
            *offset_rows,
            ('Capped at 100 %', f"{c['n_capped']:,}"),
            # Three decimals: a sensor that caps just below 100 % must not read as 100.00
            ('Mean before → after', f"{fmt(before.mean(), 3)} → {fmt(after.mean(), 3)}"),
            ('Min before → after', f"{fmt(before.min(), 3)} → {fmt(after.min(), 3)}"),
            ('Max before → after', f"{fmt(before.max(), 3)} → {fmt(after.max(), 3)}"),
        ]
