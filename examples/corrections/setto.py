"""
Examples for data correction functions using setto module.

Run this script to see correction results:
    python examples/corrections/setto.py
"""
import pandas as pd
import diive as dv


def example_set_exact_values_to_missing():
    """Set specific values in a series to missing (NaN).

    Demonstrates setting exact values to NaN, useful for removing
    error codes or sentinel values from data. Counts corrections
    and optionally shows visualization.
    """
    # Create test series with some sentinel values
    series = pd.Series([1, 2, 0, 4, 5, 6, 7, 0, 9, 10])
    series.name = "measurements"

    print("Example 1: Set exact values to missing")
    print(f"Original series: {series.values}")

    # Set values 0, 1, and 10 to missing
    series_corr = dv.set_exact_values_to_missing(
        series=series,
        values=[0, 1, 10],
        showplot=False,
        verbose=1
    )

    print(f"Corrected series: {series_corr.values}\n")


def example_setto_value():
    """Set data in time ranges to specific values.

    Demonstrates replacing data in specific time periods with
    a constant value. Useful for removing instrumental errors
    or periods of known malfunction.
    """
    # Create a time series with hourly data
    dates = pd.date_range('2022-06-01', periods=48, freq='h')
    values = range(1, 49)
    series = pd.Series(values, index=dates, name='flux')

    print("Example 2: Set time range to value")
    print(f"Original series (first 10 values):\n{series.head(10)}")

    # Set a single date and a date range to value 0
    series_corr = dv.setto_value(
        series=series,
        dates=[
            '2022-06-01 01:00:00',
            ['2022-06-01 03:00:00', '2022-06-01 15:00:00']
        ],
        value=0,
        verbose=1
    )

    print(f"Corrected series (first 30 values):\n{series_corr.head(30)}\n")


def example_setto_threshold():
    """Set values above or below threshold to threshold value.

    Demonstrates clipping data to min/max thresholds, useful for
    constraining values to physically realistic ranges.
    """
    # Create test series with values outside acceptable range
    series = pd.Series([0.5, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    series.name = "normalized_value"

    print("Example 3: Set values to threshold")
    print(f"Original series: {series.values}")

    # Clip maximum to 3.0
    series_corr_max = dv.setto_threshold(
        series=series.copy(),
        threshold=3.0,
        type='max',
        showplot=False
    )

    print(f"After max threshold 3.0: {series_corr_max.values}")

    # Clip minimum to 1.0
    series_corr_min = dv.setto_threshold(
        series=series.copy(),
        threshold=1.0,
        type='min',
        showplot=False
    )

    print(f"After min threshold 1.0: {series_corr_min.values}\n")


if __name__ == '__main__':
    example_set_exact_values_to_missing()
    example_setto_value()
    example_setto_threshold()
