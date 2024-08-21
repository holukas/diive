import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

from diive.core.plotting.histogram import HistogramPlot
from diive.core.plotting.plotfuncs import default_format, default_legend, nice_date_ticks


def plot_outlier_daytime_nighttime(series: Series, flag_daytime: Series, flag_quality: Series, title: str = None):
    """Plot outlier and non-outlier time series for daytime and nighttime data."""
    # Collect in dataframe for outlier daytime/nighttime plot
    frame = {
        'UNFILTERED': series,
        'UNFILTERED_DT': series[flag_daytime == 1],
        'UNFILTERED_NT': series[flag_daytime == 0],
        'CLEANED': series[flag_quality == 0],
        'CLEANED_DT': series[(flag_quality == 0) & (flag_daytime == 1)],
        'CLEANED_NT': series[(flag_quality == 0) & (flag_daytime == 0)],
        'OUTLIER': series[flag_quality == 2],
        'OUTLIER_DT': series[(flag_quality == 2) & (flag_daytime == 1)],
        'OUTLIER_NT': series[(flag_quality == 2) & (flag_daytime == 0)],
    }
    df = pd.DataFrame(frame)

    fig = plt.figure(facecolor='white', figsize=(24, 12))
    gs = gridspec.GridSpec(3, 4)  # rows, cols
    # gs.update(wspace=0.15, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)
    # gs.update(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.5)

    if title:
        fig.suptitle(title, fontsize=24, fontweight='bold')

    ax_series = fig.add_subplot(gs[0, 0])
    ax_series_hist = fig.add_subplot(gs[0, 1])
    ax_cleaned = fig.add_subplot(gs[0, 2], sharex=ax_series)
    ax_cleaned_hist = fig.add_subplot(gs[0, 3])

    ax_series_dt = fig.add_subplot(gs[1, 0])
    ax_series_dt_hist = fig.add_subplot(gs[1, 1])
    ax_cleaned_dt = fig.add_subplot(gs[1, 2], sharex=ax_series)
    ax_cleaned_dt_hist = fig.add_subplot(gs[1, 3])

    ax_series_nt = fig.add_subplot(gs[2, 0], sharex=ax_series)
    ax_series_nt_hist = fig.add_subplot(gs[2, 1])
    ax_cleaned_nt = fig.add_subplot(gs[2, 2], sharex=ax_series)
    ax_cleaned_nt_hist = fig.add_subplot(gs[2, 3])

    axes_series = [ax_series, ax_cleaned, ax_series_dt, ax_cleaned_dt, ax_series_nt, ax_cleaned_nt]
    axes_hist = [ax_series_hist, ax_cleaned_hist, ax_series_dt_hist,
                 ax_cleaned_dt_hist, ax_series_nt_hist, ax_cleaned_nt_hist]
    hist_kwargs = dict(method='n_bins', n_bins=None, highlight_peak=True, show_zscores=True, show_info=False,
                       show_title=False, show_zscore_values=False, show_grid=False)
    series_kwargs = dict(x=df.index, fmt='o', mec='none', alpha=.2, color='black')

    # Column 0
    ax_series.plot_date(
        y=df['CLEANED'], label=f"OK ({df['CLEANED'].count()} values)", **series_kwargs)
    ax_series.plot_date(
        x=df.index, y=df['OUTLIER'], fmt='X', ms=10, mec='none',
        alpha=.9, color='red', label=f"outlier ({df['OUTLIER'].count()} values)")
    ax_series_dt.plot_date(
        y=df['UNFILTERED_DT'], label=f"series ({df['UNFILTERED_DT'].count()} values)", **series_kwargs)
    ax_series_dt.plot_date(
        x=df.index, y=df['OUTLIER_DT'], fmt='X', ms=10, mec='none',
        alpha=.9, color='red', label=f"outlier ({df['OUTLIER_DT'].count()} values)")
    ax_series_nt.plot_date(
        y=df['UNFILTERED_NT'], label=f"series ({df['UNFILTERED_NT'].count()} values)", **series_kwargs)
    ax_series_nt.plot_date(
        x=df.index, y=df['OUTLIER_NT'], fmt='X', ms=10, mec='none',
        alpha=.9, color='red', label=f"outlier ({df['OUTLIER_NT'].count()} values)")

    # Column 1
    HistogramPlot(s=df['UNFILTERED'], **hist_kwargs).plot(ax=ax_series_hist)
    HistogramPlot(s=df['UNFILTERED_DT'], **hist_kwargs).plot(ax=ax_series_dt_hist)
    HistogramPlot(s=df['UNFILTERED_NT'], **hist_kwargs).plot(ax=ax_series_nt_hist)

    # Column 2
    ax_cleaned.plot_date(
        y=df['CLEANED'], label=f"cleaned ({df['CLEANED'].count()} values)", **series_kwargs)
    ax_cleaned_dt.plot_date(
        y=df['CLEANED_DT'], label=f"cleaned daytime ({df['CLEANED_DT'].count()} values)", **series_kwargs)
    ax_cleaned_nt.plot_date(
        y=df['CLEANED_NT'], label=f"cleaned nighttime ({df['CLEANED_NT'].count()} values)", **series_kwargs)

    # Column 3
    HistogramPlot(s=df['CLEANED'], **hist_kwargs).plot(ax=ax_cleaned_hist)
    HistogramPlot(s=df['CLEANED_DT'], **hist_kwargs).plot(ax=ax_cleaned_dt_hist)
    HistogramPlot(s=df['CLEANED_NT'], **hist_kwargs).plot(ax=ax_cleaned_nt_hist)

    for a in axes_series:
        default_format(ax=a, ax_ylabel_txt="value")
        default_legend(ax=a, ncol=1, loc=2)
        nice_date_ticks(ax=a)

    # plt.setp(ax_series.get_xticklabels(), visible=False)
    # plt.setp(ax_cleaned.get_xticklabels(), visible=False)
    # plt.setp(ax_cleaned_dt.get_xticklabels(), visible=False)
    # plt.setp(ax_cleaned_nt.get_xticklabels(), visible=False)
    # plt.setp(ax_daytime.get_xticklabels(), visible=False)

    fig.tight_layout()
    fig.show()
