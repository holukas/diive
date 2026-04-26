"""
Examples for TimeSeries visualization.

TimeSeries: Interactive time series plot using Bokeh.

Run this script to display example plots:
    python examples/visualization/timeseries.py
"""

import diive as dv


def example_timeseries_basic():
    """Basic interactive time series plot.

    Displays a Bokeh interactive plot of NEE flux data with zoom, pan, and other tools.
    """
    df = dv.load_exampledata_parquet()
    series = df['NEE_CUT_REF_f'].copy()

    ts = dv.time_series(series=series, series_units=r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)')
    ts.plot()


if __name__ == '__main__':
    print("Running TimeSeries examples...")

    print("\n1. Basic interactive time series plot...")
    example_timeseries_basic()

    print("\nAll examples completed!")
