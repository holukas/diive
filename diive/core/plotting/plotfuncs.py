import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib import pyplot as plt, _pylab_helpers, dates as mdates
from matplotlib.text import Text
from pandas import DataFrame, Series

# from modboxes.plots.styles.LightTheme import *
# from modboxes.plots.styles.LightTheme import FONTSIZE_LABELS_AXIS, COLOR_TXT_LEGEND
import diive.core.plotting.styles.LightTheme as theme
from diive.core.times.times import current_datetime
# from modboxes.plots.styles.LightTheme import *
from diive.core.plotting.styles.LightTheme import *


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
                   axlabels_fontsize: int = theme.AXLABELS_FONTSIZE,
                   axlabels_fontcolor: str = theme.AXLABELS_FONTCOLOR,
                   axlabels_fontweight=theme.AXLABELS_FONTWEIGHT,
                   txt_xlabel=False,
                   txt_ylabel=False,
                   txt_ylabel_units=False,
                   ticks_width=theme.TICKS_WIDTH,
                   ticks_length=theme.TICKS_LENGTH,
                   ticks_direction=theme.TICKS_DIRECTION,
                   ticks_labelsize=theme.TICKS_LABELSIZE,
                   color='black',
                   facecolor='white'):
    """Apply default format to ax"""
    ax.set_facecolor(facecolor)
    format_ticks(ax=ax, width=ticks_width, length=ticks_length,
                 direction=ticks_direction, color=color, labelsize=ticks_labelsize)
    format_spines(ax=ax, color=color, lw=theme.LINEWIDTH_SPINES)

    if txt_xlabel:
        ax.set_xlabel(txt_xlabel, color=axlabels_fontcolor, fontsize=axlabels_fontsize, fontweight=axlabels_fontweight)
    if txt_ylabel and txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}  {txt_ylabel_units}', color=axlabels_fontcolor, fontsize=axlabels_fontsize,
                      fontweight=axlabels_fontweight)
    if txt_ylabel and not txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}', color=axlabels_fontcolor, fontsize=axlabels_fontsize,
                      fontweight=axlabels_fontweight)
    return None


def format_ticks(ax, width, length, direction, color, labelsize):
    ax.tick_params(axis='x', width=width, length=length, direction=direction,
                   colors=color, labelsize=labelsize)
    ax.tick_params(axis='y', width=width, length=length, direction=direction,
                   colors=color, labelsize=labelsize)


def format_spines(ax, color, lw):
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)
    return None


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def hide_ticks_and_ticklabels(ax):
    ax.tick_params(left=False, right=False, top=False, bottom=False,
                   labelleft=False, labelright=False, labeltop=False, labelbottom=False)


def hide_xaxis_yaxis(ax):
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
    y_axis = ax.axes.get_yaxis()
    y_axis.set_visible(False)


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


def nice_date_ticks(ax, minticks, maxticks, which):
    """ Nice format for date ticks. """
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    if which == 'y':
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
    elif which == 'x':
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    return None


def show_txt_in_ax_bad_values(fig, ax, df, sum_for_col):
    """ Show number of bad values / marked values as text """
    ax.text(0.02, 0.98, '{} bad values'.format(df[sum_for_col].sum()),
            size=theme.AXLABELS_FONTSIZE, color='#eab839', backgroundcolor='none', transform=ax.transAxes,
            alpha=0.8, horizontalalignment='left', verticalalignment='top')
    fig.canvas.draw()
    fig.canvas.flush_events()
    return None


def add_to_existing_ax_with_values(fig, add_to_ax, x, y, counts, ms, color, marker, linestyle, linewidth):
    # Adds plot to existing axis with focus time focus_series plot

    line = add_to_ax.plot_date(x=x, y=y, color=color, alpha=1, ls=linestyle, lw=linewidth,
                               marker=marker, markeredgecolor='none', ms=ms, zorder=99)

    # for x, y, counts in zip(x, y, counts):
    #     # text_in_ax = add_to_ax.text(x, y, '{}'.format(int(counts)), color='white', fontsize='large', weight='bold',
    #     #                horizontalalignment='center', verticalalignment='center', zorder=100)
    #

    # https://stackoverflow.com/questions/11067368/annotate-time-series-plot-in-matplotlib
    #     text_in_ax = \
    #         add_to_ax.annotate('{}'.format(int(counts)), (mdates.date2num(x), y), xytext=(0, 0), color='white',
    #                            weight='bold', fontsize='large', textcoords='offset points',
    #                            arrowprops=dict(arrowstyle='-|>'), zorder=100)

    # Good solution, x needs conversion w/ date2num to work
    # https://stackoverflow.com/questions/48387480/in-matplotlib-ax-texts-container-is-empty-why
    for x, y, counts in zip(x, y, counts):
        txt1 = Text(text='{}'.format(int(counts)), x=mdates.date2num(x), y=y, color='white', fontsize='large',
                    weight='bold',
                    horizontalalignment='center', verticalalignment='center', zorder=100)

        # with this method texts in the axis can be accessed w/ axis.texts, needed for removal
        add_to_ax._add_text(txt1)

        # add_to_ax._set_artist_props(txt1)
        # add_to_ax.texts.append(txt1)
        # txt1._remove_method = lambda h: add_to_ax.texts.remove(h)
        # add_to_ax.stale = True

    fig.canvas.draw()  # update plot
    pause(0.001)

    return line


def default_legend(ax, loc='upper right', facecolor='None', edgecolor='None',
                   shadow=False, ncol=1, labelspacing=0.5, textcolor=theme.COLOR_TXT_LEGEND,
                   bbox_to_anchor=None, from_line_collection=False, line_collection=None,
                   textsize: int = theme.FONTSIZE_TXT_LEGEND):
    # fontP = FontProperties()
    # fontP.set_size('x-large')
    if from_line_collection:
        labs = [l.get_label() for l in line_collection]
        legend = ax.legend(line_collection, labs,
                           loc=loc, bbox_to_anchor=bbox_to_anchor, shadow=shadow,
                           ncol=ncol, facecolor=facecolor, edgecolor=edgecolor,
                           labelspacing=labelspacing, prop={'size': textsize})
    else:
        legend = ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, shadow=shadow,
                           ncol=ncol, facecolor=facecolor, edgecolor=edgecolor,
                           labelspacing=labelspacing, prop={'size': textsize})
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


def _add_zeroline_y(series, ax):
    if (series.min() < 0) & (series.max() > 0):
        ax.axhline(0, lw=LINEWIDTH_ZERO, color=COLOR_LINE_ZERO)


def _remove_line(line):
    if line is None:
        pass
    else:
        line.remove()


def plotdate_limit(series, ax, prevline, label_txt):
    _remove_line(prevline)
    line, = ax.plot_date(x=series.index,
                         y=series,
                         color=COLOR_LINE_LIMIT, alpha=1, ls='-',
                         marker='None', lw=WIDTH_LINE_DEFAULT,
                         markeredgecolor='none', ms=0, zorder=98, label=label_txt)
    return line


def plotdate_markers(series, ax, prevline, label_txt):
    _remove_line(prevline)
    label_txt = series.name[0] if not label_txt else label_txt
    _numvals = series.dropna().count()
    line, = ax.plot_date(x=series.index,
                         y=series,
                         color=COLOR_LINE_LIMIT, alpha=.8, ls='None',
                         marker='o', lw=0,
                         markeredgecolor='red', ms=5, zorder=98,
                         label=f"{label_txt} ({_numvals} values)")
    return line


def plotdate_main(series, ax, label_txt: str = None):
    label_txt = series.name[0] if not label_txt else label_txt
    _numvals = series.dropna().count()
    line, = ax.plot_date(x=series.index,
                         y=series,
                         color=COLOR_LINE_DEFAULT, alpha=1, ls='-',
                         marker='o', lw=WIDTH_LINE_DEFAULT,
                         markeredgecolor='none', ms=4, zorder=98,
                         label=f"{label_txt} ({_numvals} values)")
    plot_title_ref = add_ax_title_inside(txt=f"{series.name[0]}", ax=ax)
    _add_zeroline_y(series=series, ax=ax)
    _set_xylim(ax=ax, series=series)
    return line, plot_title_ref


def _set_xylim(ax, series):
    try:
        ax.set_xlim(series.index.min(), series.index.max())
        ax.set_ylim(series.min(), series.max())
    except ValueError:
        pass


def update_plotdate_main(series, ax, prevline, plot_title_ref):
    # kudos: https://www.pythonguis.com/tutorials/plotting-matplotlib/
    prevline.set_xdata(series.index)
    prevline.set_ydata(series)
    plot_title_ref.remove()
    plot_title_ref = add_ax_title_inside(txt=f"{series.name[0]}", ax=ax)
    _add_zeroline_y(series=series, ax=ax)
    _set_xylim(ax=ax, series=series)
    return plot_title_ref


def quickplot_df(data: DataFrame or Series, hline: None or float = None, subplots: bool = True,
                 title: str = None, saveplot: str or Path = None, showplot: bool = False) -> None:
    if isinstance(data, Series):
        data = pd.DataFrame(data)
    elif isinstance(data, list):
        data_cols = {}
        for series in data:
            data_cols[series.name] = series
        data = pd.concat(data_cols, axis=1)

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 1)  # rows, cols
    # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
    ax = fig.add_subplot(gs[0, 0])
    data.plot(subplots=subplots, ax=ax, title=title, lw=2)

    if hline:
        ax.axhline(hline, label=f"value: {hline}", ls='--')

    if saveplot:
        save_fig(fig=fig, path=saveplot, title=title)

    if showplot:
        # pass
        fig.show()
    plt.close(fig)


def save_fig(fig, title: str, path=Path or str):
    """Save figure to file

    Filename is sanitized, i.e. not-allowed characters are removed,
    removes also whitespace. Filename contains timestamp.
    """
    # Use alphanumeric for savename
    filename_out = [character for character in title if character.isalnum()]
    filename_out = "".join(filename_out)
    _, cur_time = current_datetime(str_format='%Y%m%d-%H%M%S-%f')
    filename_out = f"{cur_time}_{filename_out}.png"
    outfilepath = Path(path) / filename_out
    fig.savefig(outfilepath)
    print(f"Saved plot {outfilepath}")
