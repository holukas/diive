import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pandas import DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import default_format, default_legend


def plot_outlier_daytime_nighttime(df: DataFrame, title: str):
    """Plot outlier and non-outlier time series for daytime and nighttime data."""
    fig = plt.figure(facecolor='white', figsize=(12, 16))
    gs = gridspec.GridSpec(6, 1)  # rows, cols
    gs.update(wspace=0.3, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax_series = fig.add_subplot(gs[0, 0])
    ax_cleaned = fig.add_subplot(gs[1, 0], sharex=ax_series)
    ax_cleaned_daytime = fig.add_subplot(gs[2, 0], sharex=ax_series)
    ax_cleaned_nighttime = fig.add_subplot(gs[3, 0], sharex=ax_series)
    ax_daytime = fig.add_subplot(gs[4, 0], sharex=ax_series)
    ax_nighttime = fig.add_subplot(gs[5, 0], sharex=ax_series)

    ax_series.plot_date(x=df.index, y=df['UNFILTERED'], fmt='o', mec='none',
                        alpha=.5, color='black', label=f"series ({df['UNFILTERED'].count()} values)")

    ax_cleaned.plot_date(x=df.index, y=df['CLEANED'], fmt='o', mec='none',
                         alpha=.5, label=f"cleaned series ({df['CLEANED'].count()} values)")

    ax_cleaned_daytime.plot_date(x=df.index, y=df['NOT_OUTLIER_DAYTIME'], fmt='o', mec='none',
                                 alpha=.5, label=f"cleaned daytime ({df['NOT_OUTLIER_DAYTIME'].count()} values)")

    ax_cleaned_nighttime.plot_date(x=df.index, y=df['NOT_OUTLIER_NIGHTTIME'], fmt='o', mec='none',
                                   alpha=.5, label=f"cleaned nighttime ({df['NOT_OUTLIER_NIGHTTIME'].count()} values)")

    ax_daytime.plot_date(x=df.index, y=df['NOT_OUTLIER_DAYTIME'], fmt='o', mec='none',
                         alpha=.5, label=f"OK daytime ({df['NOT_OUTLIER_DAYTIME'].count()} values)")
    ax_daytime.plot_date(x=df.index, y=df['OUTLIER_DAYTIME'], fmt='X', ms=10, mec='none',
                         alpha=.9, color='red', label=f"outlier daytime ({df['OUTLIER_DAYTIME'].count()} values)")

    ax_nighttime.plot_date(x=df.index, y=df['NOT_OUTLIER_NIGHTTIME'], fmt='o', mec='none',
                           alpha=.5, label=f"OK nighttime ({df['NOT_OUTLIER_NIGHTTIME'].count()} values)")
    ax_nighttime.plot_date(x=df.index, y=df['OUTLIER_NIGHTTIME'], fmt='X', ms=10, mec='none',
                           alpha=.9, color='red', label=f"outlier nighttime ({df['OUTLIER_NIGHTTIME'].count()} values)")

    default_format(ax=ax_series)
    default_format(ax=ax_cleaned)
    default_format(ax=ax_cleaned_daytime)
    default_format(ax=ax_cleaned_nighttime)
    default_format(ax=ax_daytime)
    default_format(ax=ax_nighttime)

    default_legend(ax=ax_series, ncol=1, loc=2)
    default_legend(ax=ax_cleaned, ncol=1, loc=2)
    default_legend(ax=ax_cleaned_daytime, ncol=1, loc=2)
    default_legend(ax=ax_cleaned_nighttime, ncol=1, loc=2)
    default_legend(ax=ax_daytime, ncol=1, loc=2)
    default_legend(ax=ax_nighttime, ncol=1, loc=2)

    plt.setp(ax_series.get_xticklabels(), visible=False)
    plt.setp(ax_cleaned.get_xticklabels(), visible=False)
    plt.setp(ax_cleaned_daytime.get_xticklabels(), visible=False)
    plt.setp(ax_cleaned_nighttime.get_xticklabels(), visible=False)
    plt.setp(ax_daytime.get_xticklabels(), visible=False)

    fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
    fig.show()
