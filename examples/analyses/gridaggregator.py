"""
Examples for grid aggregation analysis using GridAggregator.

Run this script to see grid aggregation results:
    python examples/analyses/gridaggregator.py
"""
import diive as dv
from scipy.stats import zscore


def example_gridaggregator_custom_binning():
    """Grid aggregation with custom binning.

    Demonstrates binning ecosystem flux data across two driver variables
    (VPD and soil water content) using custom bin edges, then aggregating
    flux values in each bin. Results shown in wide and long formats.
    """
    # Load example data
    df = dv.load_exampledata_parquet()

    # Select variables
    vpd_col = 'VPD_f'  # Vapor pressure deficit
    swc_col = 'SWC_FF0_0.15_1'  # Soil water content
    flux_col = 'NEE_CUT_REF_f'  # Net ecosystem productivity

    subset = df[[flux_col, vpd_col, swc_col]].copy()

    # Filter to growing season (May-October) and remove NaN
    subset = subset.loc[(subset.index.month >= 5) & (subset.index.month <= 10)].copy()
    subset = subset.dropna()

    # Convert to z-scores for standardized comparison
    subset = subset.apply(lambda x: zscore(x, nan_policy='omit'))

    # Create grid aggregation with custom bins
    ga = dv.gridaggregator(
        x=subset[vpd_col],
        y=subset[swc_col],
        z=subset[flux_col],
        binning_type='custom',
        custom_x_bins=list(range(-3, 5, 1)),  # VPD bins from -3 to 5
        custom_y_bins=list(range(-3, 5, 1)),  # SWC bins from -3 to 5
        min_n_vals_per_bin=5,
        aggfunc='mean'
    )

    # Access results in wide format (rows: SWC bins, columns: VPD bins)
    print("Grid aggregation results (wide format):")
    print(ga.df_agg_wide)

    # Access results in long format for plotting
    print("\nGrid aggregation results (long format):")
    print(ga.df_agg_long)


if __name__ == '__main__':
    example_gridaggregator_custom_binning()
