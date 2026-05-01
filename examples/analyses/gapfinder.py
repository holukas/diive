"""
Examples for gap detection and analysis using GapFinder.

Run this script to see gap detection results:
    python examples/analyses/gapfinder.py
"""
import diive as dv


def example_gapfinder_basic():
    """Detect gaps in a time series.

    Demonstrates basic gap detection: finds consecutive missing values (NaN)
    and reports their locations, duration, and statistics.
    """
    # Load example data
    data_df = dv.load_exampledata_parquet()

    # Get a single variable
    series = data_df['NEE_CUT_REF_f'].copy()

    # Intentionally create some gaps for demonstration
    series.iloc[100:105] = None  # 5-value gap
    series.iloc[500:503] = None  # 3-value gap
    series.iloc[1000:1001] = None  # 1-value gap

    # Find gaps
    gf = dv.GapFinder(series=series, limit=5, sort_results=True)
    results = gf.results

    print("Gap detection results:")
    print(f"Found {len(results)} gaps (limit=5 consecutive missing values)")
    print(results)


if __name__ == '__main__':
    example_gapfinder_basic()
