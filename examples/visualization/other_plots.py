"""
Examples for miscellaneous visualization plots.

LongtermAnomaliesYear: Plot long-term anomalies per year compared to a reference period.

Run this script to display example plots:
    python examples/visualization/other_plots.py
"""

import diive as dv


def example_longterm_anomalies_temperature():
    """Long-term air temperature anomalies per year.

    Visualizes yearly air temperature anomalies as red/blue bars (above/below
    reference period mean), with reference mean ± standard deviation band.
    """
    df = dv.load_exampledata_parquet()

    # Resample to yearly mean
    series = df['Tair_f'].copy()
    series = series.resample('YE').mean()
    series.index = series.index.year

    series_label = "CH-DAV: Air temperature"

    dv.LongtermAnomaliesYear(
        series=series,
        series_label=series_label,
        series_units='(°C)',
        reference_start_year=2015,
        reference_end_year=2017
    ).plot()


if __name__ == '__main__':
    print("Running miscellaneous plot examples...")

    print("\n1. Long-term temperature anomalies per year...")
    example_longterm_anomalies_temperature()

    print("\nAll examples completed!")
