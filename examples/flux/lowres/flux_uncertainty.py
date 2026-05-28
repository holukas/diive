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

# Use subset only - June 2013 (smaller dataset for faster demo)
data_df = data_df.loc[(data_df.index.year == 2013) & (data_df.index.month == 3)].copy()

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

randunc = dv.flux.RandomUncertaintyPAS20(
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

# %%
# Cumulative uncertainty propagation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualize cumulative flux with uncertainty bounds over time.

df_cumulative = randunc.randunc_results_cumulatives[
    ['NEE_CUT_REF_f', 'FLUX+UNC', 'FLUX-UNC']
].copy()
df_cumulative.columns = ['Cumulative Flux', 'Upper Bound (+σ)', 'Lower Bound (-σ)']

fig, ax = plt.subplots(figsize=(14, 5.5))
fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.1)

ax.plot(df_cumulative.index, df_cumulative['Cumulative Flux'], linewidth=2.5,
        label='Cumulative Flux', color='black', zorder=3)
ax.fill_between(df_cumulative.index, df_cumulative['Lower Bound (-σ)'],
                df_cumulative['Upper Bound (+σ)'], alpha=0.3, color='red',
                label='Uncertainty Range (±σ)', zorder=1)
ax.plot(df_cumulative.index, df_cumulative['Upper Bound (+σ)'], linewidth=1.5,
        linestyle='--', color='red', alpha=0.6, label='Bounds', zorder=2)
ax.plot(df_cumulative.index, df_cumulative['Lower Bound (-σ)'], linewidth=1.5,
        linestyle='--', color='red', alpha=0.6, zorder=2)

ax.set_xlabel('Datetime', fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative NEE (µmol CO2 m-2)', fontsize=11, fontweight='bold')
ax.set_title('Cumulative Uncertainty Propagation with Error Bounds', fontsize=12, fontweight='bold')
ax.legend(loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle=':')
fig.show()

print("\n[OK] Random uncertainty estimation complete.")
