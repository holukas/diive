"""
Examples for histogram analysis using Histogram.

Run this script to see histogram results:
    python examples/analyses/histogram.py
"""
import diive as dv


def example_histogram_binning():
    """Create histogram from time series data.

    Demonstrates histogram calculation with different binning strategies
    and removal of fringe bins for cleaner distribution analysis.
    """
    # Load example data
    data_df = dv.load_exampledata_parquet()

    # Select a variable
    series = data_df['NEE_CUT_REF_f'].copy()

    # Create histogram with n_bins method
    hist = dv.Histogram(
        s=series,
        method='n_bins',
        n_bins=20,
        ignore_fringe_bins=[1, 1]  # Remove first and last bin
    )

    # Access results
    print("Histogram results:")
    print(hist.results)

    # Get peak bins (top 5 bins with most counts)
    print("\nPeak bins (top 5 by count):")
    print(hist.peakbins)


if __name__ == '__main__':
    example_histogram_binning()
