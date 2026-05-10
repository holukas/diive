"""
=================================================
Random Uncertainty Estimation (PAS20 Method)
=================================================

Calculate random uncertainty in flux measurements using the PAS20 method.

Demonstrates hierarchical uncertainty quantification for eddy covariance flux data
using the 4-method PAS20 approach (gap-filling uncertainty, daytime, nighttime, quality).

Best for: Assessing measurement reliability and random error propagation in flux data.
"""

# %%
# Load data and calculate random uncertainty
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Initialize RandomUncertaintyPAS20 and run the 4-method hierarchical uncertainty analysis.

import diive as dv
import matplotlib.pyplot as plt

# Load example data
data_df = dv.load_exampledata_parquet()

# Prepare subset with required columns
subset = data_df[[
    'NEE_CUT_REF_orig', 'NEE_CUT_REF_f',
    'Tair_f', 'VPD_f', 'Rg_f'
]].copy()

print("=" * 80)
print("Random Uncertainty Estimation (PAS20 Method)")
print("=" * 80)
print(f"\nData shape: {subset.shape}")
print(f"Valid records: {subset['NEE_CUT_REF_orig'].count()}")
print(f"Missing values: {subset['NEE_CUT_REF_orig'].isnull().sum()}")

# %%
# Initialize and run uncertainty calculator
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Run the 4-method hierarchical uncertainty quantification.

randunc = dv.RandomUncertaintyPAS20(
    df=subset,
    fluxcol='NEE_CUT_REF_orig',
    fluxgapfilledcol='NEE_CUT_REF_f',
    tacol='Tair_f',
    vpdcol='VPD_f',
    swincol='Rg_f'
)

print("\nRunning 4-method hierarchical uncertainty quantification...")
randunc.run()
print("[OK] Uncertainty calculation complete\n")

# %%
# Examine results
# ^^^^^^^^^^^^^^^
#
# Get uncertainty results and view method summary.

randunc_series = randunc.randunc_series
randunc_results = randunc.randunc_results

# Report: Method summary with distribution
randunc.report_method_summary()

# %%
# Visualize uncertainty distribution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Plot the distribution of random uncertainties.

x = randunc_results['NEE_CUT_REF_orig_RANDUNC'].dropna()

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.1)
ax.hist(x, bins=25, rwidth=0.85, color='#607c8e', edgecolor='black', alpha=0.8)
ax.axvline(x.mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean = {x.mean():.4f}')
ax.axvline(x.median(), color='green', linestyle='--', linewidth=2.5, label=f'Median = {x.median():.4f}')
ax.grid(True, alpha=0.3, linestyle=':', axis='y')
ax.set_title('Distribution of Random Uncertainties (NEE)', fontsize=12, fontweight='bold')
ax.set_xlabel('Random Uncertainty (±sigma) (µmol CO2 m-2 s-1)', fontsize=11)
ax.set_ylabel('Frequency (count)', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
fig.show()

print("\n[OK] Random uncertainty estimation complete.")
