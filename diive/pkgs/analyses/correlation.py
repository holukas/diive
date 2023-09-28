import pandas as pd
from matplotlib import pyplot as plt, gridspec as gridspec
from pandas import Series

from diive.pkgs.createvar.potentialradiation import potrad


def daily_correlation(s1: Series,
                      s2: Series,
                      mincorr: float = 0.8,
                      showplot: bool = False) -> Series:
    """ Calculate daily correlation between two time series

    Args:
        s1: any time series, timestamp must overlap with *s2*
        s2: any time series, timestamp must overlap with *s1*
        mincorr: minimum absolute correlation, only relevant when *showplot=True*,
            must be between -1 and 1 (inclusive)
            Example: with *0.8* all correlations between -0.8 and +0.8 are considered low,
            and all correlations smaller than -0.8 and higher than +0.8 are considered high.
        showplot: if *True*, show plot of results

    Returns:
        series with correlations for each day
    """
    if -1 <= mincorr <= 1:
        # Use absolute value for mincorr
        mincorr = abs(mincorr)
    else:
        raise Exception("arg mincorr must be between -1 and 1.")

    # Combine potential radiation with measured radiation
    df = pd.concat([s1, s2], axis=1)

    # Add date column
    df['DATE'] = df.index.date
    df['DATE'] = df['DATE'].astype(str)

    # Group data by date
    groups = df.groupby('DATE')

    # Init series with date index for storing correlations per day
    daycorrs_index = groups.count().index
    daycorrs = pd.Series(index=daycorrs_index, name='daycorrs')

    # Calculate correlation between measured and potential for each day
    for day, day_df in groups:
        corr = day_df[s1.name].corr(day_df[s2.name])
        daycorrs.loc[day] = corr

    if showplot:
        _plot_daily_correlation(daycorrs=daycorrs, mincorr=mincorr,
                                df=df, s1=s1, s2=s2)

    return daycorrs


def _plot_daily_correlation(daycorrs, mincorr, df, s1, s2):
    # Identify dates with low correlation
    _lowcorrs = daycorrs.between(-mincorr, mincorr, inclusive='neither')
    lowcorrs = daycorrs[_lowcorrs]
    lowcorrs = lowcorrs.sort_values(key=abs, ascending=True)  # Sort by absolute correlation
    lowestcorrs = lowcorrs.head(3)
    lowcorrs = lowcorrs.index.astype(str).to_list()
    lowdays = df['DATE'].isin(lowcorrs)

    # Identify dates with high correlation
    highcorrs = daycorrs[~_lowcorrs]
    highcorrs = highcorrs.sort_values(key=abs, ascending=False)  # Sort by absolute correlation
    highestcorrs = highcorrs.head(3)
    highestcorrs = highestcorrs.index.astype(str).to_list()
    highestdays = df['DATE'].isin(highestcorrs)

    # Identify dates with lowest correlation
    lowestcorrs = lowestcorrs.index.astype(str).to_list()
    lowestdays = df['DATE'].isin(lowestcorrs)

    fig = plt.figure(facecolor='white', figsize=(9, 12), dpi=150)
    gs = gridspec.GridSpec(4, 3)  # rows, cols
    gs.update(wspace=0.3, hspace=0.4, left=0.05, right=0.97, top=0.9, bottom=0.1)
    ax1 = fig.add_subplot(gs[0, 0:])
    ax2 = fig.add_subplot(gs[1, 0:])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[2, 2])
    ax6 = fig.add_subplot(gs[3, 0])
    ax7 = fig.add_subplot(gs[3, 1])
    ax8 = fig.add_subplot(gs[3, 2])

    daycorrs.plot(
        ax=ax1, title=f"Correlation between {s2.name} and {s1.name} per day "
                      f"(n = {len(daycorrs)})\ncorrelation "
                      f"median = {daycorrs.median():.3f}, "
                      f"99th percentile = {daycorrs.quantile(.99):.3f} "
                      f"1st percentile = {daycorrs.quantile(.01):.3f}, "
                      f"min / max = {daycorrs.min():.3f} / {daycorrs.max():.3f} "
    )
    ax1.axhline(-mincorr, c='#ff0051')
    ax1.axhline(mincorr, c='#ff0051')
    ax1.set_ylim(-1, 1)

    # Get full resolution data for low-correlation days
    lowdays_fullres = df[lowdays].copy()
    groups2 = lowdays_fullres.groupby(lowdays_fullres['DATE'])
    for day, day_df in groups2:
        day_df.index = day_df.index.time
        day_df[[s2.name, s1.name]].plot(ax=ax2, legend=False, alpha=.3, color='grey')
    ax2.set_title(f"Found {len(lowcorrs)} low correlation days")

    # Get full resolution data for lowest-correlation days
    lowestdays_fullres = df[lowestdays].copy()
    groups3 = lowestdays_fullres.groupby(lowestdays_fullres['DATE'])
    axes = [ax3, ax4, ax5]
    counter = 0
    for day, day_df in groups3:
        day_df.index = day_df.index.time
        day_df[[s2.name, s1.name]].plot(ax=axes[counter])
        axes[counter].set_title(f"{day}, r = {daycorrs[day]:.3f}")
        counter += 1

    # Get full resolution data for highest-correlation days
    highestdays_fullres = df[highestdays].copy()
    groups4 = highestdays_fullres.groupby(highestdays_fullres['DATE'])
    axes = [ax6, ax7, ax8]
    counter = 0
    for day, day_df in groups4:
        day_df.index = day_df.index.time
        day_df[[s2.name, s1.name]].plot(ax=axes[counter])
        axes[counter].set_title(f"{day}, r = {daycorrs[day]:.3f}")
        counter += 1

    # fig.tight_layout()
    fig.suptitle(f"Comparison between {s1.name} and {s2.name}")
    fig.show()


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()

    series = df['Rg_f'].copy()

    # Calculate potential radiation SW_IN_POT
    reference = potrad(timestamp_index=series.index,
                       lat=47.286417,
                       lon=7.733750,
                       utc_offset=1)

    # series = df['NEE_CUT_REF_f'].copy()
    # reference = df['Tair_f']

    daily_correlation(
        s1=series,
        s2=reference,
        mincorr=0.8,
        showplot=True
    )
    # x = sw_in_pot.groupby([sw_in_pot.index.month, sw_in_pot.index.time]).mean().unstack().T
    # x.plot()
    # HeatmapDateTime(series=sw_in_pot).show()
    # import matplotlib.pyplot as plt
    # sw_in_pot.plot()
    # plt.show()


if __name__ == '__main__':
    example()
