"""
Examples for optimum range analysis using FindOptimumRange.

Run this script to see optimum range results and visualizations:
    python examples/analyses/optimumrange.py
"""
import diive as dv


def example_optimumrange_daytime():
    """Find optimum range of driver variable for optimized response.

    Demonstrates finding the air temperature range where net ecosystem
    productivity (NEE) is optimized (minimized for CO2 uptake) during
    daytime. Results include percentage of data in, above, and below
    the optimum range.
    """
    # Load example data
    df_orig = dv.load_exampledata_parquet()

    # Filter to daytime data (solar radiation > 20)
    df = df_orig.copy()
    df = df.loc[df['Rg_f'] > 20]

    # Find optimum temperature range for maximum CO2 uptake (minimum NEE)
    optrange = dv.find_optimum_range(
        df=df,
        xcol='Tair_f',  # Driver variable (air temperature)
        ycol='NEE_CUT_REF_f',  # Response variable (net ecosystem productivity)
        n_bins=100,  # Number of bins for x-axis
        define_optimum='min'  # Minimum NEE = maximum CO2 uptake
    )

    # Calculate optimum range
    optrange.find_optimum()

    # Access results
    results = optrange.results_optrange
    print(f"Optimum range: {results['optimum_xstart']:.2f} to {results['optimum_xend']:.2f} C")
    print(f"Mean NEE in optimum range: {results['optimum_ymean']:.3f} umol/m2/s")

    # Show percentage of data in optimum range
    vals_in_range = results['vals_in_optimum_range_df']
    print("\nPercentage of data in optimum range (by year):")
    print(vals_in_range[['vals_inoptimum_perc']])

    # Display visualizations
    optrange.showfig()


if __name__ == '__main__':
    example_optimumrange_daytime()
