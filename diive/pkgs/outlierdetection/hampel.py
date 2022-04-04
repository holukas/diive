from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from diive.core.plotting.plotfuncs import default_format, default_legend
from diive.core.plotting.plotfuncs import save_fig
from diive.core.plotting.styles import LightTheme as theme


def hampel_filter(input_series: Series, winsize: int = 50, winsize_min_periods: int = 1,
                  n_sigmas: int = 5, show: bool = False, saveplot: bool = False) -> Series:
    """
    kudos: https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
    """

    print(f"Removing outliers (Hampel) from {input_series.name} ...")

    k = 1.4826  # scale factor for Gaussian distribution

    # Absolute difference to rolling median
    rolling_median = input_series.rolling(window=2 * winsize, min_periods=winsize_min_periods,
                                          center=True).median()
    diff = np.abs(input_series - rolling_median)

    # MAD (median absolute deviation)
    MAD = lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))  # helper lambda function, ignores nan
    rolling_mad = k * input_series.rolling(window=2 * winsize, min_periods=winsize_min_periods, center=True).apply(MAD)

    # print(len(rolling_mad.dropna()))

    # Limit and outlier detection
    # Original implementation from kudos:
    #   indices = list(np.argwhere(diff > (n_sigmas * rolling_mad)).flatten())
    #   new_series[indices] = rolling_median[indices]
    limit = n_sigmas * rolling_mad
    outlier_ix = diff > limit  # Outlier indices
    flag = outlier_ix.astype(int)  # Convert to zeros and ones

    # Get series for non-outliers
    output_series = input_series.copy()
    output_series.loc[flag == 1] = np.nan

    # Outliers
    outliers = input_series.copy()
    outliers.loc[flag == 0] = np.nan

    # Collect in df
    df = pd.DataFrame()
    df['input_series'] = input_series
    df['output_series'] = output_series
    df['outliers'] = outliers
    df['flag'] = flag
    df['limit'] = limit
    df['rolling_mad'] = rolling_mad
    df['rolling_median'] = rolling_median
    df['diff'] = diff

    if saveplot:
        title = f"Remove High-res Outliers (Hampel): {input_series.name}"
        _plot_hampel(df, title=title, saveplot=saveplot)

    return output_series


def _plot_hampel(df: DataFrame, title: str = None, saveplot: str or Path = None) -> None:
    fig = plt.figure(figsize=(10, 15))
    gs = gridspec.GridSpec(3, 1)  # rows, cols
    gs.update(wspace=.2, hspace=0, left=.05, right=.95, top=.95, bottom=.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    ax1.plot_date(df.index, df['input_series'], label="input series")
    ax1.plot_date(df.index, df['rolling_median'], label="rolling median", lw=1, ms=1)

    ax2.plot_date(df.index, df['diff'], label="difference: absolute( input series - rolling median )")
    ax2.plot_date(df.index, df['limit'], lw=1, ms=0, ls='-', label="limit: rolling MAD * n_sigmas")

    ax3.plot_date(df.index, df['output_series'], label="non-outliers")
    ax3.plot_date(df.index, df['outliers'], color='red', ms=5, label="outliers")

    # Add info text
    num_vals_before = df['input_series'].dropna().count()
    num_outliers = df['outliers'].dropna().count()
    num_vals_after = df['output_series'].dropna().count()
    numrel_outliers = num_outliers / num_vals_before

    infotxt = f"Outlier removal: Hampel filter\n" \
              f"{num_vals_before} values before outlier removal\n" \
              f"{num_outliers} outliers detected ({numrel_outliers:.1f}%)\n" \
              f"{num_vals_after} after outlier removal"

    ax3.text(0.02, 0.98, infotxt,
             size=theme.INFOTXT_FONTSIZE, color='black', backgroundcolor='none',
             transform=ax3.transAxes, alpha=1,
             horizontalalignment='left', verticalalignment='top')

    default_format(ax=ax1)
    default_format(ax=ax2)
    default_format(ax=ax3)
    default_legend(ax=ax1)
    default_legend(ax=ax2)
    default_legend(ax=ax3)

    # ax3.text(results_chd['thres_chd'], _ratio_at_thres,
    #         f"    Ratio at threshold: {_ratio_at_thres}",
    #         size=theme.FONTSIZE_ANNOTATIONS_SMALL,
    #         color='#2196F3', backgroundcolor='none',
    #         alpha=1, horizontalalignment='left', verticalalignment='center',
    #         bbox=dict(boxstyle='square,pad=0', fc='none', ec='none'))

    fig.show()

    if saveplot:
        save_fig(fig=fig, title=title, path=saveplot)
