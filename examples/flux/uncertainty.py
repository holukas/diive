"""
Random uncertainty quantification for eddy covariance flux measurements.

Demonstrates the RandomUncertaintyPAS20 class for computing flux measurement
uncertainty across multiple error sources (instrumental, statistical, gap-filling).
"""

import matplotlib.pyplot as plt
import diive as dv


def example_random_uncertainty_pas20():
    """Compute and visualize random flux uncertainty using PAS20 method.

    The RandomUncertaintyPAS20 class uses 4-method hierarchical approach:
    - Method 1: Sliding window (±7 days, ±1 hr) of measured fluxes with similar meteorology
    - Method 2: Median of similar-flux uncertainties (±5 days, ±1 hr, ±20% flux similarity)
    - Method 3: Similar flux range without time window restrictions
    - Method 4: 5 nearest fluxes by magnitude (fallback for method gaps)

    The result is cumulative uncertainty propagation showing flux bounds.
    """
    # Load example data
    from diive.configs.exampledata import load_exampledata_parquet
    from diive.core.dfun.frames import df_between_two_dates

    data_df = load_exampledata_parquet()

    # Restrict to June-July 2022 for faster processing
    data_df = df_between_two_dates(
        df=data_df,
        start_date='2022-06-01',
        end_date='2022-07-01'
    ).copy()

    # Prepare subset with required columns
    subset = data_df[[
        'NEE_CUT_REF_orig', 'NEE_CUT_REF_f',
        'Tair_f', 'VPD_f', 'Rg_f'
    ]].copy()

    # Initialize uncertainty calculator
    randunc = dv.RandomUncertaintyPAS20(
        df=subset,
        fluxcol='NEE_CUT_REF_orig',
        fluxgapfilledcol='NEE_CUT_REF_f',
        tacol='Tair_f',
        vpdcol='VPD_f',
        swincol='Rg_f'
    )

    # Run all 4 methods
    randunc.run()

    # Get results
    randunc_results = randunc.randunc_results
    randunc_series = randunc.randunc_series
    randunc_results_cumulatives = randunc.randunc_results_cumulatives

    # Plot 1: Histogram of random uncertainties
    x = randunc_results['NEE_CUT_REF_orig_RANDUNC']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x, bins=20, rwidth=0.9, color='#607c8e')
    ax.grid(True, alpha=0.3)
    ax.set_title('Distribution of Random Uncertainties (NEE)')
    ax.set_xlabel('Random Uncertainty (μmol CO₂ m⁻² s⁻¹)')
    ax.set_ylabel('Count')
    fig.show()

    # Report cumulative uncertainty propagation
    randunc.report_cumulative_uncertainty_propagation()

    # Plot 2: Random uncertainties with measured flux
    randunc.showplot_random_uncertainty()

    # Plot 3: Cumulative uncertainty bounds
    randunc.showplot_cumulative_uncertainty_propagation()

    return {
        'randunc_results': randunc_results,
        'randunc_series': randunc_series,
        'randunc_results_cumulatives': randunc_results_cumulatives
    }


if __name__ == '__main__':
    example_random_uncertainty_pas20()
