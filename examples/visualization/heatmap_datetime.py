"""
Examples for HeatmapDateTime and HeatmapYearMonth visualization.

HeatmapDateTime: Visualize time series as a date × time-of-day grid.
HeatmapYearMonth: Visualize time series as a year × month grid with aggregation.

Run this script to display all example plots:
    python examples/visualization/heatmap_datetime.py
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import diive as dv


def example_heatmap_datetime_vertical():
    """HeatmapDateTime in vertical orientation (date × time-of-day grid).

    Visualizes a time series with dates on y-axis and time-of-day (0-24h) on x-axis.
    """
    df = dv.load_exampledata_parquet()

    var = 'NEE_CUT_REF_f'
    series = df[var].copy()
    locs = series.index.year >= 2020
    series = series.loc[locs]
    series.iloc[100:120] = np.nan  # For testing
    series = series.dropna()  # For testing

    hm = dv.heatmap_datetime(series=series, title=None, vmin=-10, vmax=10, ax_orientation="vertical")
    hm.show()


def example_heatmap_datetime_horizontal():
    """HeatmapDateTime in horizontal orientation (time-of-day × date grid).

    Visualizes a time series with time-of-day (0-24h) on y-axis and dates on x-axis.
    """
    df = dv.load_exampledata_parquet()

    var = 'NEE_CUT_REF_f'
    series = df[var].copy()
    locs = series.index.year >= 2020
    series = series.loc[locs]
    series.iloc[100:120] = np.nan
    series = series.dropna()

    hm = dv.heatmap_datetime(series=series, title=None, vmin=-10, vmax=10, ax_orientation="horizontal")
    hm.show()


def example_heatmap_yearmonth_basic():
    """HeatmapYearMonth with basic aggregation and ranks.

    Visualizes aggregated monthly time series with optional rank transformation.
    """
    df = dv.load_exampledata_parquet()
    series = df['Tair_f'].copy()
    series.name = None  # For testing
    series.index.freq = None  # For testing
    series.iloc[100:120] = np.nan

    dv.heatmap_year_month(series=series, ax_orientation="horizontal", ranks=True, show_values=True).show()
    dv.heatmap_year_month(series=series, ax_orientation="vertical", ranks=False, show_values=True).show()


def example_heatmap_yearmonth_multi_panel():
    """HeatmapYearMonth with multiple aggregation methods in a figure.

    Creates a figure with two panels showing mean and peak-to-peak (range)
    aggregations side-by-side.
    """
    df = dv.load_exampledata_parquet()
    fig = plt.figure(facecolor='white', figsize=(16, 9), layout="constrained", dpi=300)
    gs = gridspec.GridSpec(2, 1, figure=fig)  # rows, cols
    gs.update(wspace=0.5, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    zlabel = r'$\mathrm{\mu mol\ CO_2\ m^{-2}\ s^{-1}}$'
    settings = dict(ax_orientation='horizontal', zlabel=zlabel, cb_digits_after_comma=0,
                    show_values=True, show_values_n_dec_places=0)
    series = df['NEE_CUT_REF_f']

    dv.heatmap_year_month(ax=ax1, series=series, agg='mean', **settings).plot()
    dv.heatmap_year_month(ax=ax2, series=series, agg=np.ptp, **settings).plot()

    ax1.set_xlabel("")
    ax1.set_xticklabels("")
    ax1.set_title("")
    ax2.set_title("")
    fig.suptitle("NEE")
    fig.show()


def example_heatmap_yearmonth_colormaps():
    """Demonstrates all available colormaps with HeatmapYearMonth.

    Creates a heatmap for each colormap to preview color schemes.
    Note: This generates many plots; comment out if overwhelming.
    """
    from diive.core.plotting.heatmap_base import list_of_colormaps

    cmaps = list_of_colormaps()
    df = dv.load_exampledata_parquet()
    series = df['Tair_f'].copy()

    # Limit to first 5 colormaps for demo (uncomment to show all)
    for cmap in cmaps[:5]:
        hm = dv.heatmap_year_month(series=series, cb_digits_after_comma=0, zlabel="degC",
                                 ax_orientation="vertical", figsize=(14, 10), cmap=cmap, title=cmap)
        hm.show()
        # Uncomment below to save to file:
        # outfile = rf"F:\output\{cmap}.png"
        # hm.fig.savefig(outfile)
        # print(f"Saved {outfile}")


if __name__ == '__main__':
    print("Running HeatmapDateTime and HeatmapYearMonth examples...")
    print("\n1. HeatmapDateTime (vertical orientation)...")
    example_heatmap_datetime_vertical()

    print("\n2. HeatmapDateTime (horizontal orientation)...")
    example_heatmap_datetime_horizontal()

    print("\n3. HeatmapYearMonth (basic with ranks)...")
    example_heatmap_yearmonth_basic()

    print("\n4. HeatmapYearMonth (multi-panel layout)...")
    example_heatmap_yearmonth_multi_panel()

    print("\n5. HeatmapYearMonth (colormap preview - limited to 5)...")
    example_heatmap_yearmonth_colormaps()

    print("\nAll examples completed!")
