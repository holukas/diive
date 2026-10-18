"""
NIGHTTIME_OFFSET_PLOT: DIAGNOSTIC FIGURE OF THE NIGHTTIME ZERO-OFFSET CORRECTION
================================================================================

One figure that shows what ``remove_nighttime_zero_offset`` did and how well it fits.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pandas import Series

from diive.preprocessing.corrections._figure import (
    C_BOX_EDGE as _C_BOX_EDGE, C_CORRECTED as _C_CORRECTED, C_FILLED as _C_FILLED, C_FIRST as _C_NIGHTSTEP,
    C_INK as _C_INK, C_MEASURED as _C_MEASURED, C_OFFSET as _C_OFFSET, C_SHADE as _C_NIGHT,
    C_SUBTRACTED as _C_SUBTRACTED, C_TITLE as _C_TITLE, C_ZERO as _C_ZERO, date_axis, day_steps,
    draw_steps, draw_summary, fmt as _fmt, legend, panel_title, style_axes)
from diive.preprocessing.corrections.offsetcorrection import NighttimeZeroOffsetResult


class NighttimeZeroOffsetPlot:
    """Diagnostic figure of the nighttime zero-offset correction.

    Two-phase like the other diive plots: the constructor takes the result of
    ``nighttime_zero_offset_diagnostics``, ``plot()`` draws. A strip at the top
    shows the five steps of the correction in order, each with its count:

    1. Night: records with potential radiation below 0.001 W m-2.
    2. Daily offset: the mean of the night values of each calendar day (the
       night before sunrise and the night after sunset of that date). Every
       record of the day gets this value, there is no interpolation between
       days. Days without night values get the median of all daily offsets.
    3. The offset is subtracted from every record of the day.
    4. Night records are set to 0.
    5. Values still below 0 (daytime, e.g. at dawn and dusk) are set to 0,
       unless ``clamp_negatives=False`` kept them.

    Panels below the strip, titled with the steps they show:

    - Series: measured and corrected high-res data.
    - Daily offset: one value per calendar day, drawn as a step line, with the
      standard deviation of the day's night values as a band, the median as a
      line, and days without nighttime data (median offset) marked.
    - Diel cycle near zero: median of each time of day, measured, after
      subtracting the offset (step 3) and final (steps 4 and 5), limited to the
      low range where the offset matters, with the times of day that are
      mostly night shaded.
    - Nighttime values: distribution of the measured nighttime values and of
      their residuals after subtracting the day's offset. A narrow residual
      distribution means one offset per day describes the night well.
    - Summary: records, nighttime share, offset statistics, records below zero
      before and after, daytime records clamped to zero, and mean, min and max
      before and after.

    Example:
        >>> import diive as dv, numpy as np, pandas as pd
        >>> idx = pd.date_range('2024-06-01', periods=48 * 5, freq='30min', name='TIMESTAMP_MIDDLE')
        >>> pot = dv.variables.potrad(timestamp_index=idx, lat=47.1, lon=8.5, utc_offset=1)
        >>> sw = pd.Series(np.where(pot > 0.001, 0.7 * pot, 0) - 8.0, index=idx, name='SW_IN')
        >>> res = dv.corrections.nighttime_zero_offset_diagnostics(sw, lat=47.1, lon=8.5, utc_offset=1)
        >>> fig = dv.corrections.NighttimeZeroOffsetPlot(res).plot(showplot=False)
    """

    def __init__(self, result: NighttimeZeroOffsetResult, records: Series | None = None):
        """Prepare the figure data.

        Args:
            result: Output of ``nighttime_zero_offset_diagnostics``.
            records: Optional boolean Series on the index of *result*, True for
                rows that are records. Rows that are no record (empty grid slots
                between coarse records in meteo screening) are left out of the
                line plots, so lines join neighbouring records. Missing records
                stay NaN and break the lines.
        """
        self.result = result
        self.name = str(result.input.name) if result.input.name is not None else 'value'
        keep = records.reindex(result.input.index).fillna(False).astype(bool) if records is not None else None
        self._lines = (result.input, result.corrected) if keep is None else \
            (result.input[keep], result.corrected[keep])
        self.night = result.nighttime_flag.reindex(result.input.index) == 1
        self.daily = self._daily_stats()

    def _daily_stats(self) -> pd.DataFrame:
        """Per calendar day: offset applied, nighttime mean, SD and count of measured values."""
        r = self.result
        days = r.input.index.normalize()
        night_vals = r.input.where(self.night)
        grouped = night_vals.groupby(days)
        daily = pd.DataFrame({'night_mean': grouped.mean(), 'night_sd': grouped.std(),
                              'n_night': grouped.count()})
        daily['offset'] = r.offset.groupby(days).first()
        return daily

    def plot(self, title: str = None, figsize: tuple = (14, 11.5), showplot: bool = True):
        """Draw the figure. Returns the matplotlib Figure.

        Args:
            title: Figure title. Default: 'Nighttime zero offset: <variable>'.
            figsize: Figure size in inches.
            showplot: Show the figure.
        """
        fig = plt.figure(figsize=figsize, layout='constrained', facecolor='white')
        gs = fig.add_gridspec(4, 3, height_ratios=[0.42, 1.0, 0.9, 1.15], width_ratios=[1.25, 1.0, 1.0])
        ax_steps = fig.add_subplot(gs[0, :])
        ax_series = fig.add_subplot(gs[1, :])
        ax_offset = fig.add_subplot(gs[2, :], sharex=ax_series)
        ax_diel = fig.add_subplot(gs[3, 0])
        ax_hist = fig.add_subplot(gs[3, 1])
        ax_text = fig.add_subplot(gs[3, 2])

        self._plot_steps(ax_steps)
        self._plot_series(ax_series)
        self._plot_offset(ax_offset)
        self._plot_diel(ax_diel)
        self._plot_hist(ax_hist)
        self._plot_summary(ax_text)
        for ax in (ax_series, ax_offset, ax_diel, ax_hist):
            self._style(ax)
        date_axis(ax_offset)
        ax_series.tick_params(labelbottom=False)
        fig.suptitle(title or f"Nighttime zero offset: {self.name}", x=0.01, ha='left',
                     fontsize=15, fontweight='bold', color=_C_TITLE)
        self.fig = fig
        if showplot:
            fig.show()
        return fig

    @staticmethod
    def _style(ax):
        """Open frame, light horizontal gridlines, as the TimeSeries default look."""
        style_axes(ax)

    @staticmethod
    def _legend(ax, **kwargs):
        """Frameless legend with lines thick enough to tell the colours apart."""
        legend(ax, **kwargs)

    @staticmethod
    def _title(ax, text: str, note: str = None):
        """Left-aligned panel title with an optional grey note after it."""
        panel_title(ax, text, note)

    def _steps(self) -> list[tuple[str, str, str, str]]:
        """Colour, title, description and count of each correction step, in order."""
        r = self.result
        d = self.daily
        n_records = int(r.input.notna().sum())
        n_night = int((self.night & r.input.notna()).sum())
        n_filled = int((d['n_night'] == 0).sum())
        share = 100 * n_night / n_records if n_records else 0
        offset_count = f"{len(d):,} days, median {_fmt(d['offset'].median())}"
        if n_filled:
            offset_count += f", {n_filled:,} with median"
        before, subtracted = r.input.dropna(), r.corrected_by_offset.dropna()
        day = ~self.night & r.input.notna()
        negative_day = int(((r.corrected_by_offset < 0) & day).sum())
        if r.n_below_zero_after > 0:  # clamp_negatives=False
            last = ('Negatives kept', 'clamp_negatives=False', f"{r.n_below_zero_after:,} records below 0")
        else:
            last = ('Clamp', 'day values still below 0\nset to 0', f"{negative_day:,} records")
        return [
            (_C_NIGHTSTEP, 'Find night', 'potential radiation\nbelow 0.001 W m-2',
             f"{n_night:,} records ({share:.1f} %)"),
            (_C_OFFSET, 'Daily offset', "mean of each calendar day's\nnight values, same all day",
             offset_count),
            (_C_SUBTRACTED, 'Subtract', 'offset from every record\nof the day, day and night',
             f"mean {_fmt(before.mean())} \u2192 {_fmt(subtracted.mean())}"),
            (_C_CORRECTED, 'Night to 0', 'all night records\nset to 0', f"{n_night:,} records"),
            (_C_CORRECTED, *last),
        ]

    def _plot_steps(self, ax):
        """Numbered boxes with arrows: the order in which the correction works."""
        draw_steps(ax, self._steps())

    def _plot_series(self, ax):
        measured, corrected = self._lines
        ax.plot(measured.index, measured.to_numpy(), color=_C_MEASURED, lw=0.6, label='measured', zorder=2)
        ax.plot(corrected.index, corrected.to_numpy(), color=_C_CORRECTED, lw=0.6, alpha=0.9,
                label='corrected', zorder=3)
        ax.axhline(0, color=_C_ZERO, lw=0.8, zorder=1)
        ax.set_ylabel(self.name, color=_C_INK, fontsize=10)
        ax.set_xmargin(0)
        self._legend(ax, loc='upper left', ncol=2)
        self._title(ax, "Series", "measured and corrected high-resolution data")

    def _plot_offset(self, ax):
        d = self.daily
        x = d.index + pd.Timedelta('12h')  # the middle of each day
        measured = d['n_night'] > 0
        # Each day from midnight to midnight at its own value: the offset is
        # constant within a day and jumps at midnight, no interpolation.
        sx, sy = self._day_steps(d['offset'])
        sd = self._day_steps((d['night_sd'].where(measured)))[1]
        ax.fill_between(sx, sy - sd, sy + sd, color=_C_OFFSET, alpha=0.18, lw=0,
                        label='\u00b11 SD of the night')
        ax.plot(sx, sy, color=_C_OFFSET, lw=1.1, label='daily offset, constant per day')
        if (~measured).any():
            ax.plot(x[~measured], d['offset'][~measured], ls='none', marker='o', markersize=5,
                    mfc='white', mec=_C_FILLED, mew=1.2, label='no night data, median used')
        median = float(d['offset'].median())
        ax.axhline(median, color=_C_INK, lw=0.8, ls='--', label=f'median {_fmt(median)}')
        ax.axhline(0, color=_C_ZERO, lw=0.8)
        ax.set_ylabel('offset', color=_C_INK, fontsize=10)
        ax.legend(loc='best', frameon=False, fontsize=8.5, ncol=4)
        self._title(ax, "Daily offset  \u00b7  steps 1-2",
                    "mean of the night values of each calendar day, same for all its records")

    @staticmethod
    def _day_steps(values: Series) -> tuple[np.ndarray, np.ndarray]:
        """x and y of a step line holding each day's value from midnight to midnight."""
        return day_steps(values)

    def _plot_diel(self, ax):
        r = self.result
        idx = r.input.index
        tod = idx.hour + idx.minute / 60 + idx.second / 3600
        measured = r.input.groupby(tod).median()
        subtracted = r.corrected_by_offset.groupby(tod).median()
        corrected = r.corrected.groupby(tod).median()
        night_share = self.night.groupby(tod).mean()
        # Shade the times of day that are night on most days
        mostly_night = night_share > 0.5
        edges = np.flatnonzero(np.diff(np.r_[0, mostly_night.to_numpy().astype(int), 0]))
        tvals = night_share.index.to_numpy()
        step = float(np.median(np.diff(tvals))) if len(tvals) > 1 else 1.0
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            ax.axvspan(tvals[start] - step / 2, tvals[end - 1] + step / 2, color=_C_NIGHT, lw=0, zorder=0)
        ax.plot(measured.index, measured.to_numpy(), color=_C_MEASURED, lw=1.4, label='measured')
        ax.plot(subtracted.index, subtracted.to_numpy(), color=_C_SUBTRACTED, lw=1.2, ls='--',
                label='3: offset subtracted')
        ax.plot(corrected.index, corrected.to_numpy(), color=_C_CORRECTED, lw=1.4,
                label='4-5: night and negatives to 0')
        ax.axhline(0, color=_C_ZERO, lw=0.8)
        # Low range: from below the lowest night median to a few times the offset above zero
        low = float(np.nanmin([measured.min(), 0]))
        span = max(abs(low), abs(float(self.daily['offset'].median())), 1.0)
        ax.set_ylim(low - 0.6 * span, 4 * span)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel('time of day (h)', color=_C_INK, fontsize=9)
        handles = ax.get_legend_handles_labels()[0]
        handles.append(Patch(facecolor=_C_NIGHT, edgecolor=_C_BOX_EDGE, label='1: mostly night'))
        self._legend(ax, handles=handles, loc='upper left')
        self._title(ax, "Diel cycle near zero  \u00b7  steps 3-5", "median")

    def _plot_hist(self, ax):
        r = self.result
        night_vals = r.input[self.night].dropna()
        residual = (r.input - r.offset)[self.night].dropna()
        if night_vals.empty:
            ax.text(0.5, 0.5, 'no nighttime values', ha='center', va='center', transform=ax.transAxes,
                    color=_C_INK)
            self._title(ax, "Night values  \u00b7  step 3")
            return
        lo, hi = np.nanpercentile(np.r_[night_vals, residual], [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(night_vals, bins=bins, color=_C_MEASURED, alpha=0.8, label='measured')
        ax.hist(residual, bins=bins, histtype='step', color=_C_SUBTRACTED, lw=1.4,
                label='3: offset subtracted')
        ax.axvline(0, color=_C_ZERO, lw=0.8)
        ax.set_xlabel(self.name, color=_C_INK, fontsize=9)
        ax.set_ylabel('records', color=_C_INK, fontsize=9)
        ax.legend(loc='upper left', frameon=False, fontsize=8.5)
        self._title(ax, "Night values  \u00b7  step 3", f"SD {_fmt(night_vals.std())} → {_fmt(residual.std())}")

    def _summary_rows(self) -> list[tuple[str, str]]:
        """Label and value of each summary line."""
        r = self.result
        d = self.daily
        n_records = int(r.input.notna().sum())
        n_night = int((self.night & r.input.notna()).sum())
        n_days, n_filled = len(d), int((d['n_night'] == 0).sum())
        off = d['offset']
        day = ~self.night
        clamped = int(((r.corrected_by_offset < 0) & day & r.input.notna()).sum())
        before, after = r.input.dropna(), r.corrected.dropna()
        share = 100 * n_night / n_records if n_records else 0
        return [
            ('Records', f"{n_records:,}"),
            ('Nighttime records', f"{n_night:,} ({share:.1f} %)"),
            ('Days', f"{n_days:,}, {n_filled:,} without night data"),
            ('Offset median', _fmt(off.median())),
            ('Offset mean ± SD', f"{_fmt(off.mean())} ± {_fmt(off.std())}"),
            ('Offset min / max', f"{_fmt(off.min())} / {_fmt(off.max())}"),
            ('Below zero before', f"{r.n_below_zero_before:,} (night {r.n_below_zero_before_night:,})"),
            ('Day records clamped to 0', f"{clamped:,}"),
            ('Below zero after', f"{r.n_below_zero_after:,} (night {r.n_below_zero_after_night:,})"),
            ('Mean before → after', f"{_fmt(before.mean())} → {_fmt(after.mean())}"),
            ('Min before → after', f"{_fmt(before.min())} → {_fmt(after.min())}"),
            ('Max before → after', f"{_fmt(before.max())} → {_fmt(after.max())}"),
        ]

    def _plot_summary(self, ax):
        draw_summary(ax, self._summary_rows())
