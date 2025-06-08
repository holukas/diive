import time
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib import pyplot as plt, _pylab_helpers, dates as mdates
from pandas import DataFrame, Series

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.styles.LightTheme import *
from diive.core.times.times import current_datetime


# TODO generalize for other classes
def set_fig(ax: plt.Axes):
    if ax:
        # If ax is given, plot directly to ax, no fig needed
        fig = None
        # self.ax = self.ax
        showplot = False
    else:
        # If no ax is given, create fig and ax and then show the plot
        fig, ax = create_ax()
        showplot = True
    return fig, ax, showplot


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def show_ticks_on_all_spines(ax):
    ax.tick_params(left=True, right=True, top=True, bottom=True)


def hide_ticks_and_ticklabels(ax):
    ax.tick_params(left=False, right=False, top=False, bottom=False,
                   labelleft=False, labelright=False, labeltop=False, labelbottom=False)


def hide_xaxis_yaxis(ax):
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
    y_axis = ax.axes.get_yaxis()
    y_axis.set_visible(False)


def remove_prev_lines(ax):
    # Every time the slider multiplier is changed to a new value,
    # the marker that shows the outlier values are drawn.
    # In case there is already a marker in the plot, it needs to be
    # removed first, then the new markers are drawn.
    # Since the main plot of the time series is line 0, the marker
    # and the limit lines are lines > 0. Therefore, here we try
    # to remove all lines > 0. If there is a marker and aux lines,
    # then all are removed. If there are none, in the current plot,
    # nothing is removed. Line 0 is the main plot and is never removed.
    # Since the index of lines changes after a removal, 3 times line 1
    # is removed.

    num_lines = len(ax.lines)
    for l in range(num_lines):
        try:
            ax.lines[1].remove()
        except:
            pass

    # Remove all collections in axis, e.g. .broken_barh
    ax.collections = []

    # Remove all texts in axis
    ax.texts = []  ## this is so much simpler I cannot believe it

    # num_lines = len(ax.lines)
    # for l in range(num_lines):
    #     try:
    #         ax.lines[1].remove()
    #     except:
    #         pass

    # for ix, t in enumerate(ax.texts):
    #     ax.texts[ix].remove()

    return ax


# def remove_all_twin_ax_lines(twin_ax):
#     for ix, line in enumerate(twin_ax.lines):
#         try:
#             twin_ax.lines[ix].remove()
#         except:
#             pass
#     return twin_ax

def default_format(ax,
                   ax_labels_fontsize: float = theme.AX_LABELS_FONTSIZE,
                   ax_labels_fontcolor: str = theme.AX_LABELS_FONTCOLOR,
                   ax_labels_fontweight=theme.AX_LABELS_FONTWEIGHT,
                   ax_xlabel_txt=False,
                   ax_ylabel_txt=False,
                   spines_lw: float = None,
                   txt_ylabel_units=False,
                   ticks_width=theme.TICKS_WIDTH,
                   ticks_length=theme.TICKS_LENGTH,
                   ticks_direction=theme.TICKS_DIRECTION,
                   ticks_labels_fontsize=theme.TICKS_LABELS_FONTSIZE,
                   color='black',
                   facecolor='white',
                   showgrid: bool = True) -> None:
    """Apply default format to ax"""
    # Facecolor
    ax.set_facecolor(facecolor)

    # Ticks
    format_ticks(ax=ax, width=ticks_width, length=ticks_length,
                 direction=ticks_direction, color=color,
                 labelsize=ticks_labels_fontsize)

    # Spines
    format_spines(ax=ax, color=color, lw=spines_lw)

    # Labels
    if ax_xlabel_txt:
        ax.set_xlabel(ax_xlabel_txt, color=ax_labels_fontcolor, fontsize=ax_labels_fontsize,
                      fontweight=ax_labels_fontweight)
    if ax_ylabel_txt and txt_ylabel_units:
        ax.set_ylabel(f'{ax_ylabel_txt}  {txt_ylabel_units}', color=ax_labels_fontcolor, fontsize=ax_labels_fontsize,
                      fontweight=ax_labels_fontweight)
    if ax_ylabel_txt and not txt_ylabel_units:
        ax.set_ylabel(f'{ax_ylabel_txt}', color=ax_labels_fontcolor, fontsize=ax_labels_fontsize,
                      fontweight=ax_labels_fontweight)

    # Grid
    if showgrid:
        default_grid(ax=ax)
    else:
        ax.grid(False)


def format_ticks(ax, width, length, direction, color, labelsize):
    ax.tick_params(axis='x', width=width, length=length, direction=direction,
                   colors=color, labelsize=labelsize)
    ax.tick_params(axis='y', width=width, length=length, direction=direction,
                   colors=color, labelsize=labelsize)
    show_ticks_on_all_spines(ax)
    # from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    # ax.xaxis.set_minor_locator(AutoMinorLocator())


def format_spines(ax, color, lw):
    lw = theme.LINEWIDTH_SPINES if not lw else lw
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)
    return None


def non_numeric_error(ax):
    plt.text(0.5, 0.5, 'Sorry, no plot. Data are non-numeric.', horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.5))


def clear_ax(ax):
    ax.clear()


def pause(interval):
    """
    Adjusted from matplotlib source code

    Pause for *interval* seconds.
    If there is an active figure, it will be updated and displayed before the
    pause, and the gui event loop (if any) will run during the pause.
    This can be used for crude animation.  For more complex animation, see
    :mod:`matplotlib.animation`.

    Notes
    -----
    This function is experimental; its behavior may be changed or extended in a
    future release.
    """
    manager = _pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.fig.stale:
            canvas.draw_idle()
        # plt.show(block=False)
        canvas.start_event_loop(interval)
        canvas.stop_event_loop()  # added by me
    else:
        time.sleep(interval)


def make_secondary_yaxis(ax):
    # Secondary y-axis
    twin_ax = ax.twinx()
    ax_offset = 1
    twin_ax.spines["right"].set_position(("axes", ax_offset))
    make_patch_spines_invisible(ax=twin_ax)
    twin_ax.spines["right"].set_visible(True)

    return twin_ax


def has_twin(ax):
    # Check if axis already has a secondary axis
    # https://stackoverflow.com/questions/36209575/how-to-detect-if-a-twin-axis-has-been-generated-for-a-matplotlib-axis
    for other_ax in ax.fig.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return True
    return False


def set_marker_color(current_color_ix, plot_color_list, current_marker_ix, plot_marker_list):
    """ Change to next color and next marker type. """
    current_color_ix += 1  # ix 0 and 1 for quality A1 and A2, respectively
    current_color_ix = \
        current_color_ix + 1 if current_color_ix < len(plot_color_list) - 1 else 0
    current_marker_ix = \
        current_marker_ix + 1 if current_marker_ix < len(plot_marker_list) - 1 else 0
    return current_color_ix, current_marker_ix


## Dark colors:
# ax.set_facecolor('#29282d')
# ax.tick_params(axis='x', width=1, length=6, direction='in', colors='#999c9f')
# ax.tick_params(axis='y', colors='#999c9f', width=1, length=6, direction='in')
# format_spines(ax=ax, color='#444444', lw=1)


def nice_date_ticks(ax, minticks: int = 3, maxticks: int = 9, which: Literal['x', 'y'] = 'x', locator: str = 'auto'):
    """ Nice format for date ticks. """
    # years = mdates.YearLocator(base=3, month=12, day=31)
    # yearss_fmt = mdates.DateFormatter('%dn%bn%Y')
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(years_fmt)

    if locator == 'year':
        locator = mdates.YearLocator(base=3, month=12, day=31)
        formatter = mdates.DateFormatter('%Y')
    elif locator == 'month':
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter('%b')
    elif locator == 'hour':
        # locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
        # locator = mdates.HourLocator()
        locator = mdates.HourLocator(byhour=[6, 12, 18])
        formatter = mdates.DateFormatter('%H')
    elif locator == 'auto':
        locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
        formatter = mdates.ConciseDateFormatter(locator, show_offset=False)

    if which == 'y':
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
    elif which == 'x':
        # formatter.formats = ['%H:%M']
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    return None


def default_legend(ax,
                   loc: int or str = 0,
                   facecolor='None',
                   edgecolor='None',
                   shadow=False,
                   ncol=1,
                   labelspacing=0.5,
                   textcolor=theme.COLOR_TXT_LEGEND,
                   bbox_to_anchor=None,
                   from_line_collection=False,
                   line_collection=None,
                   textsize: int = theme.FONTSIZE_TXT_LEGEND,
                   markerscale: float = None,
                   title: str = None):
    # fontP = FontProperties()
    # fontP.set_size('x-large')
    if from_line_collection:
        labs = [l.get_label() for l in line_collection]
        legend = ax.legend(line_collection, labs,
                           loc=loc, bbox_to_anchor=bbox_to_anchor, shadow=shadow,
                           ncol=ncol, facecolor=facecolor, edgecolor=edgecolor,
                           labelspacing=labelspacing, prop={'size': textsize},
                           markerscale=markerscale)

    else:
        legend = ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, shadow=shadow,
                           ncol=ncol, facecolor=facecolor, edgecolor=edgecolor,
                           labelspacing=labelspacing, prop={'size': textsize},
                           markerscale=markerscale)

    if title:
        legend.set_title(title=title, prop={'size': textsize})

    for text in legend.get_texts():
        text.set_color(textcolor)


def default_grid(ax):
    ax.grid(True, ls='--', color=theme.COLOR_LINE_GRID, lw=theme.LINEWIDTH_SPINES, zorder=0)


def wheel_markers_7():
    markers = ['o', '^', 'v', 's', 'D', '*', 'd']
    return markers


def add_ax_title_inside(txt, ax):
    text = ax.text(0.01, 0.97, f"{txt}",
                   size=FONTSIZE_HEADER_AXIS, color=FONTCOLOR_HEADER_AXIS,
                   backgroundcolor='none', transform=ax.transAxes, alpha=1,
                   horizontalalignment='left', verticalalignment='top', zorder=99)
    return text


def add_zeroline_y(data: Series or DataFrame, ax):
    if isinstance(data, DataFrame):
        # Min/max across all columns in DataFrame
        _min = data.min().min()
        _max = data.max().max()
    else:
        # Min/max for Series
        _min = data.min()
        _max = data.max()
    if (_min < 0) & (_max > 0):
        ax.axhline(0, lw=LINEWIDTH_ZERO, color=COLOR_LINE_ZERO, zorder=98)


def remove_line(line):
    if line is None:
        pass
    else:
        line.remove()


def set_xylim(ax, series):
    try:
        ax.set_xlim(series.index.min(), series.index.max())
        ax.set_ylim(series.min(), series.max())
    except ValueError:
        pass


def quickplot(data: DataFrame or Series, hline: None or float = None, subplots: bool = True,
              title: str = None, saveplot: str or Path = None, showplot: bool = False,
              showstats: bool = True) -> None:
    if isinstance(data, Series):
        data = pd.DataFrame(data)
    elif isinstance(data, list):
        data_cols = {}
        for series in data:
            data_cols[series.name] = series
        data = pd.concat(data_cols, axis=1)

    fig = plt.figure(figsize=(20, 9))

    # Number of plots in figure
    n_plotrows = len(data.columns) if subplots else 1

    gs = gridspec.GridSpec(n_plotrows, 1)  # rows, cols
    # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)

    # Create axis for each column
    axes = {}
    for a in range(0, n_plotrows):
        axes[a] = fig.add_subplot(gs[a, 0])

    colors = colors_12(400)
    for ix, col in enumerate(data.columns):
        ax = axes[ix] if subplots else axes[0]
        mean = data[col].mean()
        std = data[col].std()
        max = data[col].max()
        min = data[col].min()
        ax.plot(data.index, data[col],
                color=colors[ix], marker='o', markersize=6, markeredgecolor=colors[ix],
                label=f"{col}\n"
                      f"mean: {mean:.2f}±{std:.2f}\n"
                      f"min: {min:.2f}  |  max: {max:.2f}")
        ax.text(0.02, 0.98, title,
                size=theme.AX_LABELS_FONTSIZE, color='black', backgroundcolor='none', transform=ax.transAxes,
                alpha=1, horizontalalignment='left', verticalalignment='top')
        default_legend(ax=ax, facecolor='white')
        if hline:
            ax.axhline(hline, label=f"value: {hline}", ls='--')

    if saveplot:
        save_fig(fig=fig, path=saveplot, title=title)

    if showplot:
        fig.tight_layout()
        fig.show()


def save_fig(fig,
             title: str = "",
             path: Path or str = None,
             sanitize_filename: bool = False):
    """Save figure to file

    Filename is sanitized, i.e. not-allowed characters are removed,
    removes also whitespace. Filename contains timestamp.
    """
    # Use alphanumeric for savename

    title = "plot" if not title else title
    filename_out = title.replace(" ", "_")

    if sanitize_filename:
        filename_out = [character for character in title if character.isalnum()]
        filename_out = "".join(filename_out)

    _, cur_time = current_datetime(str_format='%Y%m%d-%H%M%S-%f')
    filename_out = f"{filename_out}_{cur_time}.png"
    if path:
        outfilepath = Path(path) / filename_out
    else:
        outfilepath = filename_out
    fig.savefig(outfilepath)
    print(f"Saved plot {outfilepath}")


def create_ax(facecolor: str = 'white',
              figsize: tuple[float, float] = (14, 8),
              dpi: int = 100):
    """Create figure and axis"""
    # Figure setup
    fig = plt.figure(facecolor=facecolor, figsize=figsize, dpi=dpi, layout='constrained')
    gs = gridspec.GridSpec(1, 1, figure=fig)  # rows, cols
    # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    ax = fig.add_subplot(gs[0, 0])
    return fig, ax


def setup_figax(ax, figsize):
    # Create axis
    if ax:
        # If ax is given, plot directly to ax, no fig needed
        fig = None
        # self.ax = self.ax
        showplot = False
    else:
        # If no ax is given, create fig and ax and then show the plot
        fig, ax = create_ax(figsize=figsize)
        showplot = True
    return fig, ax, showplot


def n_legend_cols(n_legend_entries: int) -> int:
    """Set number of legend columns"""
    if 1 <= n_legend_entries <= 5:
        n_legend_cols = 1
    elif 6 <= n_legend_entries <= 15:
        n_legend_cols = 2
    else:
        n_legend_cols = 3
    return n_legend_cols


def example_quickplot():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    df = df.loc[df.index.year == 2021].copy()
    series = df['VPD_f'].copy()
    quickplot(data=series, showplot=True)


if __name__ == '__main__':
    example_quickplot()
