"""
CORRECTIONS._FIGURE: SHARED LOOK OF THE CORRECTION DIAGNOSTIC FIGURES
=====================================================================

Colours, axes style, panel titles and the numbered step strip used by the
diagnostic figures of the offset corrections.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from pandas import Series

# Material Design, as the TimeSeries default look
C_MEASURED = '#90A4AE'  # blue-grey 300
C_CORRECTED = '#2196F3'  # blue 500
C_OFFSET = '#FB8C00'  # orange 600
C_FILLED = '#E53935'  # red 600, days that got a filled offset
C_INK = '#455A64'  # blue-grey 700
C_TITLE = '#263238'  # blue-grey 900
C_GRID = '#ECEFF1'  # blue-grey 50
C_SHADE = '#ECEFF1'  # night shading
C_ZERO = '#90A4AE'  # reference lines, spines
C_SUBTRACTED = '#AB47BC'  # purple 400, after subtracting the offset
C_FIRST = '#78909C'  # blue-grey 400, the first step
C_BOX = '#F5F7F8'
C_BOX_EDGE = '#CFD8DC'


def fmt(value: float, decimals: int = 2) -> str:
    """Number with thousands separators and a true minus sign."""
    return f"{value:,.{decimals}f}".replace('-', '−')


def style_axes(ax):
    """Open frame, light horizontal gridlines, as the TimeSeries default look."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(C_ZERO)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=C_ZERO, labelcolor=C_INK, labelsize=9, width=0.8, length=4, direction='out')
    ax.grid(False)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor('white')


def legend(ax, **kwargs):
    """Frameless legend with lines thick enough to tell the colours apart."""
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    return leg


def panel_title(ax, text: str, note: str = None):
    """Left-aligned panel title with an optional grey note on the right."""
    ax.set_title(text, loc='left', fontsize=11, fontweight='bold', color=C_TITLE)
    if note:
        ax.set_title(note, loc='right', fontsize=9, color=C_INK)


def date_axis(ax):
    """Concise date labels with the year at the end of the axis."""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.offset_formats = ['', '%Y', '%Y', '%Y', '%Y-%m-%d', '%Y-%m-%d %H:%M']
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def day_steps(values: Series) -> tuple[np.ndarray, np.ndarray]:
    """x and y of a step line holding each day's value from midnight to midnight.

    *values* has one value per day on the day's midnight. Days that do not
    follow each other are separated by NaN, so the line breaks.
    """
    starts = values.index
    x, y = [], []
    for k, (start, value) in enumerate(values.items()):
        if k and start - starts[k - 1] > pd.Timedelta('1D'):
            x.append(start)
            y.append(np.nan)
        x += [start, start + pd.Timedelta('1D')]
        y += [value, value]
    return np.array(x, dtype='datetime64[ns]'), np.array(y, dtype=float)


def draw_steps(ax, steps: list[tuple[str, str, str, str]]):
    """Numbered boxes joined by arrows: the order in which a correction works.

    Each step is (colour, title, description, count). The description may hold
    a line break.
    """
    ax.axis('off')
    n = len(steps)
    width = 0.172
    gap = (1 - n * width) / (n - 1)
    for k, (color, title, text, count) in enumerate(steps):
        x0 = k * (width + gap)
        ax.add_patch(FancyBboxPatch((x0, 0.04), width, 0.92, transform=ax.transAxes,
                                    boxstyle='round,pad=0,rounding_size=0.015',
                                    fc=C_BOX, ec=C_BOX_EDGE, lw=0.8))
        ax.text(x0 + 0.014, 0.76, str(k + 1), transform=ax.transAxes, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.3', fc=color, ec='none'))
        ax.text(x0 + 0.032, 0.76, title, transform=ax.transAxes, va='center', fontsize=10.5,
                fontweight='bold', color=C_TITLE)
        ax.text(x0 + 0.01, 0.47, text, transform=ax.transAxes, va='center', fontsize=8.5,
                color=C_INK, linespacing=1.3)
        ax.text(x0 + 0.01, 0.15, count, transform=ax.transAxes, va='center', fontsize=9,
                fontweight='bold', color=color)
        if k < n - 1:
            ax.annotate('', xy=(x0 + width + gap - 0.002, 0.5), xytext=(x0 + width + 0.002, 0.5),
                        xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='-|>', color=C_ZERO, lw=1.2, mutation_scale=12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_summary(ax, rows: list[tuple[str, str]]):
    """Label and value pairs, one per line, labels left and values right."""
    ax.axis('off')
    panel_title(ax, "Summary")
    y, dy = 0.97, 0.97 / len(rows)
    for label, value in rows:
        ax.text(0.0, y, label, transform=ax.transAxes, va='top', fontsize=9.5, color=C_INK)
        ax.text(1.0, y, value, transform=ax.transAxes, va='top', ha='right', fontsize=9.5,
                color=C_TITLE, fontweight='bold')
        y -= dy
