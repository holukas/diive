"""
CHAINPLOTS: SHARED LOOK OF THE DIAGNOSTIC PLOTS
===============================================

Colours and small helpers shared by the quality-control, outlier and gap-filling plots, so
that kept and rejected records, day and night, and the USTAR scenarios look the same in
every figure.

Part of the diive library: https://github.com/holukas/diive
"""
import numpy as np
from pandas import Series

from diive.core.plotting.styles._material_design_colors import (
    amber, blue, bluegray, indigo, red,
)

INK = bluegray(900)          # titles, axis labels
MUTED = bluegray(400)        # subtitles, tick labels, notes
GRID = bluegray(100)
KEPT = bluegray(300)         # records that passed a test
REJECTED = red(500)          # records a test rejected
CLEANED = blue(500)          # the series after a test
DAY = amber(700)
NIGHT = indigo(500)

#: Ordered colours for a few ordered scenarios (USTAR CUT_16 < CUT_50 < CUT_84, ...):
#: one hue from light to dark, so the order reads off the colour.
SCENARIO_COLORS = [blue(300), blue(600), indigo(900), bluegray(500), amber(700)]

#: One colour per outlier test in overview plots, in the order the tests ran.
TEST_COLORS = [red(600), '#FB8C00', '#8E24AA', blue(600), '#00897B', '#6D4C41', '#D81B60']

#: One colour per meteo driver, for features derived from it (time terms use KEPT).
DRIVER_COLORS = [amber(700), '#26A69A', indigo(400), red(400), blue(400), '#8D6E63']

#: Markers of large point clouds are rasterised: a year of half-hours is 17,520 points, and
#: as vectors they make a notebook megabytes larger without looking any different.
POINTS = dict(linestyle='none', marker='o', markersize=2.5, markeredgewidth=0, rasterized=True)


def style_axes(ax, grid_axis: str = 'y') -> None:
    """Open frame, light grid and muted ticks."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(bluegray(300))
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9, length=3)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)


def panel_label(ax, text: str) -> None:
    """Small bold label in the upper left corner of a panel."""
    ax.text(0.005, 1.02, text, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, fontweight='bold', color=INK)


def figure_title(fig, title: str, subtitle: str = None) -> None:
    """Left-aligned title, with an optional muted subtitle on the line below."""
    # The axes are laid out below a band of fixed height (in points, so it does not depend on
    # the figure size), and the title and subtitle are placed in that band.
    height_pt = fig.get_figheight() * 72
    pad, title_pt, sub_pt = 8, 14 * 1.35, 10 * 1.4
    band = (pad + title_pt + (sub_pt if subtitle else 0) + pad) / height_pt
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, 'set') and 'rect' in engine.get():
        engine.set(rect=(0, 0, 1, 1 - band))
    else:
        fig.subplots_adjust(top=1 - band - 0.02)
    y_title = 1 - pad / height_pt
    fig.text(0.01, y_title, title, ha='left', va='top', fontsize=14, fontweight='bold', color=INK)
    if subtitle:
        fig.text(0.01, y_title - title_pt / height_pt, subtitle, ha='left', va='top',
                 fontsize=10, color=MUTED)


def zscore_axis(ax, series: Series, which: str = 'y') -> None:
    """Secondary axis in z-scores of `series`, on the right (or top) of `ax`.

    Matplotlib places its ticks, so the labels stay readable whatever the range. Nothing is
    drawn when the series has no spread.
    """
    values = series.dropna().to_numpy(dtype=float)
    if values.size < 2:
        return
    mean, sd = float(np.mean(values)), float(np.std(values))
    if not np.isfinite(sd) or sd == 0:
        return
    functions = (lambda v: (v - mean) / sd, lambda z: z * sd + mean)
    if which == 'y':
        sec = ax.secondary_yaxis('right', functions=functions)
        sec.set_ylabel('z-score', color=MUTED, fontsize=9)
    else:
        sec = ax.secondary_xaxis('top', functions=functions)
        sec.set_xlabel('z-score', color=MUTED, fontsize=9)
    sec.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=8, length=3)
    for spine in sec.spines.values():
        spine.set_color(bluegray(200))


#: Colours of the overall quality flag: 0 passed, 1 passed with a warning, 2 rejected.
QCF_COLORS = ['#26A69A', amber(600), red(500)]
NAN_COLOR = bluegray(50)


def symmetric_limit(*series: Series, q: float = 0.99) -> float | None:
    """A colour limit for a variable that crosses zero: the q-quantile of |values|.

    One extreme half-hour would otherwise set the scale and wash out the rest of the record.
    """
    values = np.concatenate([s.dropna().to_numpy(dtype=float) for s in series]) if series else []
    values = np.abs(values[np.isfinite(values)]) if len(values) else values
    return float(np.quantile(values, q)) if len(values) else None


def heatmap_style(title: str):
    """FormatStyle for one heatmap panel of a multi-panel figure."""
    from diive.core.plotting.styles.format import FormatStyle
    return FormatStyle(title=title, title_fontsize=10, axlabel_fontsize=9, ticks_fontsize=8,
                       text_color=INK, chrome_color=MUTED, show_grid=False)


def integer_colorbar(cax, labels: dict | None = None) -> None:
    """Ticks of a colourbar for whole-number codes: one per code, optionally named."""
    from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator
    box = cax.get_position()
    axis = cax.yaxis if box.height >= box.width else cax.xaxis  # vertical or horizontal bar
    if labels:
        axis.set_major_locator(FixedLocator(list(labels)))
        axis.set_major_formatter(FixedFormatter(list(labels.values())))
    else:
        axis.set_major_locator(MaxNLocator(integer=True))
    axis.set_minor_locator(FixedLocator([]))


def note(ax, text: str) -> None:
    """Centred muted note on an empty panel."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=10, color=MUTED)
