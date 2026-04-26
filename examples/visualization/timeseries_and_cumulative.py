"""
Examples for cumulative time series visualization.

Cumulative: Plot cumulative sums across all data.
CumulativeYear: Plot yearly cumulative sums with reference mean and standard deviation.

Run this script to display example plots:
    python examples/visualization/timeseries_and_cumulative.py
"""

import diive as dv


def example_cumulative_overall():
    """Cumulative sums across all data years.

    Visualizes total cumulative CO2 flux for multiple USTAR scenarios,
    converting from µmol CO2 m⁻² s⁻¹ to g C m⁻² 30min⁻¹.
    """
    df_orig = dv.load_exampledata_parquet()
    df = df_orig[['NEE_CUT_16_f', 'NEE_CUT_REF_f', 'NEE_CUT_84_f']].copy()
    df = df.multiply(0.02161926)  # umol CO2 m-2 s-1 --> g C m-2 30min-1
    series_units = r'($\mathrm{gC\ m^{-2}}$)'

    dv.Cumulative(
        df=df,
        units=series_units,
        start_year=2015,
        end_year=2019
    ).plot()


def example_cumulative_year_simple():
    """Yearly cumulative sums with reference mean and standard deviation.

    Shows annual CO2 cumulative totals with mean and ±1 standard deviation
    band based on reference period.
    """
    df_orig = dv.load_exampledata_parquet()

    series = df_orig['NEE_CUT_REF_f'].copy()
    series = series.multiply(0.02161926)  # umol CO2 m-2 s-1 --> g C m-2 30min-1
    series_units = r'($\mathrm{gC\ m^{-2}}$)'

    dv.CumulativeYear(
        series=series,
        series_units=series_units,
        yearly_end_date=None,
        start_year=2015,
        end_year=2019,
        show_reference=True,
        excl_years_from_reference=None,
        highlight_year_color='#F44336'
    ).plot()


def example_cumulative_year_with_highlight():
    """Yearly cumulative sums with highlighted year.

    Emphasizes a specific year (e.g., 2022) by showing it in a distinct color
    and with thicker line width for comparison against other years.
    """
    df_orig = dv.load_exampledata_parquet()

    series = df_orig['NEE_CUT_REF_f'].copy()
    series = series.multiply(0.02161926)  # umol CO2 m-2 s-1 --> g C m-2 30min-1
    series_units = r'($\mathrm{gC\ m^{-2}}$)'

    dv.CumulativeYear(
        series=series,
        series_units=series_units,
        yearly_end_date=None,
        start_year=2015,
        end_year=2019,
        show_reference=True,
        excl_years_from_reference=None,
        highlight_year=2017,
        highlight_year_color='#F44336'
    ).plot()


if __name__ == '__main__':
    print("Running cumulative examples...")

    print("\n1. Cumulative overall (across all years)...")
    example_cumulative_overall()

    print("\n2. Cumulative per year (simple with reference)...")
    example_cumulative_year_simple()

    print("\n3. Cumulative per year (with highlighted year)...")
    example_cumulative_year_with_highlight()

    print("\nAll examples completed!")
