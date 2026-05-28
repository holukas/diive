"""
===========================
Curve Fitting with BinFitterCP
===========================

Demonstrates binned curve fitting with polynomial functions and confidence intervals.
BinFitterCP is useful for analyzing driver-response relationships in ecosystem data
with uncertainty quantification.

Best for: Understanding nonlinear relationships between flux drivers and response variables.
"""

# %%
# NEE Response to Vapor Pressure Deficit
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Fit a quadratic relationship between net ecosystem exchange (NEE) and vapor
# pressure deficit (VPD) using summer daytime data. This example shows:
# 1. Data filtering for specific conditions
# 2. Binning and aggregation
# 3. Curve fitting with confidence intervals
# 4. Visualization with prediction bands

import pandas as pd

import diive as dv

# Load example data
df_orig = dv.load_exampledata_parquet()

# Variables
vpd_col = 'VPD_f'
ta_col = 'Tair_f'
nee_col = 'NEE_CUT_REF_f'
xcol = vpd_col
ycol = nee_col

print("BinFitterCP - Driver-Response Analysis")
print("=" * 50)

# Filter for summer daytime with adequate radiation
subset = df_orig.loc[(df_orig.index.month >= 6) & (df_orig.index.month <= 9)].copy()
subset = subset.loc[subset['Rg_f'] > 200]

print(f"\nData Filtering:")
print(f"  Original records: {len(df_orig)}")
print(f"  Summer daytime (Jun-Sep, Rg>200): {len(subset)} records")

# Convert units
subset[vpd_col] = subset[vpd_col].multiply(0.1)  # hPa --> kPa
subset[nee_col] = subset[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1

print(f"\nUnit Conversions:")
print(f"  VPD: hPa -> kPa")
print(f"  NEE: umol CO2 m-2 s-1 -> g CO2 m-2 30min-1")

print(f"\nVariable Statistics:")
print(f"  {vpd_col}: {subset[vpd_col].min():.2f} to {subset[vpd_col].max():.2f} kPa")
print(f"  {nee_col}: {subset[nee_col].min():.2f} to {subset[nee_col].max():.2f} g CO2 m-2 30min-1")

# Format labels
y_units = r"gCO_{2}\ m^{-2}\ 30min^{-1}"
ylabel = f"NEE (${y_units}$)"
x_units = "kPa"
xlabel = f"Vapor Pressure Deficit (${x_units}$)"

print(f"\nCurve Fitting")
print("=" * 50)

# Create fitter and run
bf = dv.analysis.BinFitterCP(
    df=subset,
    xcol=xcol,
    ycol=ycol,
    n_predictions=1000,
    n_bins_x=10,  # Bin into 10 equal-width bins
    bins_y_agg='mean',
    fit_type='quadratic_offset'  # y = ax² + bx + c
)

bf.run()
fit_results = bf.fit_results

print(f"\nFitting Configuration:")
print(f"  Fit type: {fit_results['fit_type']}")
print(f"  Number of bins: 10")
print(f"  Aggregation: {fit_results['yvar']} mean per bin")
print(f"  Fit equation: {fit_results['fit_equation_str']}")
print(f"  R²: {fit_results['fit_r2']:.4f}")

n_vals_min = int(fit_results['n_vals_per_bin']['min'])
n_vals_max = int(fit_results['n_vals_per_bin']['max'])
print(f"\nBin Statistics:")
print(f"  Values per bin: {n_vals_min} to {n_vals_max}")
print(f"  Total bins: {len(fit_results['bin_df'])}")

# Display plot
print(f"\nVisualization")
print("=" * 50)

try:
    bf.showplot(
        show_unbinned_data=True,
        show_bin_variation=True,
        showfit=True,
        show_prediction_interval=True,
        title='NEE Response to Vapor Pressure Deficit (Summer Daytime)',
        xlabel=xlabel,
        ylabel=ylabel,
        unbinned_alpha=0.5
    )
    print("[OK] Fit plot generated successfully")
except Exception as e:
    print(f"Plot display info: {type(e).__name__}")
