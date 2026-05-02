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
    print("Running 4-method hierarchical uncertainty quantification...")
    randunc.run()
    print("✓ Uncertainty calculation complete\n")

    # Get results
    randunc_results = randunc.randunc_results
    randunc_series = randunc.randunc_series
    randunc_results_cumulatives = randunc.randunc_results_cumulatives

    # Report: Method summary with distribution
    randunc.report_method_summary()

    # Plot 1: Histogram of random uncertainties
    x = randunc_results['NEE_CUT_REF_orig_RANDUNC'].dropna()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.1)
    ax.hist(x, bins=25, rwidth=0.85, color='#607c8e', edgecolor='black', alpha=0.8)
    ax.axvline(x.mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean = {x.mean():.4f}')
    ax.axvline(x.median(), color='green', linestyle='--', linewidth=2.5, label=f'Median = {x.median():.4f}')
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax.set_title('Distribution of Random Uncertainties (NEE)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Random Uncertainty (+/- sigma) (umol CO2 m-2 s-1)', fontsize=11)
    ax.set_ylabel('Frequency (count)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
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
