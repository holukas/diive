"""
Examples for RidgeLinePlot visualization.

RidgeLinePlot: Kernel density estimate plots stacked by time period (weekly, monthly, yearly).

Run this script to display example plots:
    python examples/visualization/ridgeline.py
"""

import diive as dv


def example_ridgeline_weekly():
    """RidgeLinePlot with weekly separation.

    Shows distribution of air temperature for each week of the year (2019 data).
    """
    df = dv.load_exampledata_parquet()

    # Filter to single year for clarity
    locs = (df.index.year >= 2019) & (df.index.year <= 2019)
    df = df[locs].copy()
    series = df['Tair_f'].copy()

    rp = dv.ridgeline(series=series)
    rp.plot(
        how='weekly',
        kd_kwargs=None,
        xlim=None,
        ylim=None,
        hspace=-0.5,
        xlabel=r'Air temperature (°C)',
        fig_width=8,
        fig_height=10,
        shade_percentile=0.5,
        show_mean_line=False
    )


def example_ridgeline_monthly():
    """RidgeLinePlot with monthly separation.

    Shows distribution of air temperature for each month across multiple years.
    """
    df = dv.load_exampledata_parquet()
    series = df['Tair_f'].copy()

    rp = dv.ridgeline(series=series)
    rp.plot(
        how='monthly',
        xlabel=r'Air temperature (°C)',
        fig_width=10,
        fig_height=8,
        shade_percentile=0.5
    )


if __name__ == '__main__':
    print("Running RidgeLinePlot examples...")

    print("\n1. RidgeLinePlot weekly (single year)...")
    example_ridgeline_weekly()

    print("\n2. RidgeLinePlot monthly...")
    example_ridgeline_monthly()

    print("\nAll examples completed!")
