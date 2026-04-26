"""
Examples for HistogramPlot visualization.

HistogramPlot: Distribution analysis with z-score overlay and peak highlighting.

Run this script to display example plots:
    python examples/visualization/histogram.py
"""

import diive as dv


def example_histogram_basic():
    """Basic histogram with z-scores and peak highlighting.

    Shows distribution of NEE flux values with z-score overlay and peak bin highlighted.
    """
    df = dv.load_exampledata_parquet()
    series = df['NEE_CUT_REF_f'].copy()

    hist = dv.plot_histogram(
        s=series,
        method='n_bins',
        n_bins=20,
        xlabel='NEE flux',
        highlight_peak=True,
        show_zscores=True,
        show_zscore_values=True,
        show_info=True
    )
    hist.plot()


def example_histogram_yearly():
    """Histogram for each year separately.

    Shows how NEE flux distribution changes across years.
    """
    df = dv.load_exampledata_parquet()
    years = df.index.year.unique()

    for year in years[:3]:  # Show first 3 years
        series = df[df.index.year == year]['NEE_CUT_REF_f'].copy()
        if len(series) < 10:
            continue

        hist = dv.plot_histogram(
            s=series,
            method='n_bins',
            n_bins=15,
            xlabel='NEE flux',
            highlight_peak=True,
            show_zscores=True
        )
        hist.plot()


if __name__ == '__main__':
    print("Running HistogramPlot examples...")

    print("\n1. Basic histogram with z-scores...")
    example_histogram_basic()

    print("\n2. Histogram per year...")
    example_histogram_yearly()

    print("\nAll examples completed!")
