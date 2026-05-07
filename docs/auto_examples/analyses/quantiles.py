"""
Examples for percentile analysis using percentiles101.

Run this script to see percentile results and visualizations:
    python examples/analyses/quantiles.py
"""
import diive as dv


def example_percentiles101_distribution():
    """Calculate and visualize percentiles 0-100 for air temperature.

    Demonstrates calculating percentiles across the full 0-100 range
    for air temperature data, showing the distribution from minimum
    to maximum values with detailed statistics.
    """
    # Load example data
    df = dv.load_exampledata_parquet()

    # Calculate percentiles 0-100 for air temperature
    percentiles_df = dv.percentiles101(series=df['Tair_f'], showplot=True, verbose=True)

    # Display full results
    print("\nAll percentiles (0-100):")
    print(percentiles_df)


if __name__ == '__main__':
    example_percentiles101_distribution()
