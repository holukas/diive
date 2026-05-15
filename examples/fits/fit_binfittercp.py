"""
=============================
Binned Curve Fitting (BinFitterCP)
=============================

Demonstrates fitting polynomial functions to binned time series data with
confidence and prediction intervals. BinFitterCP is useful for analyzing
physical relationships and driver-response functions in environmental data.

This example shows how to:
- Bin data into equal-width intervals
- Fit polynomial equations (linear, quadratic, cubic)
- Extract and interpret results (fit parameters, confidence intervals, binned data)
- Visualize fitted curves with uncertainty bands

Best for: Understanding nonlinear relationships between variables with
quantified uncertainty.
"""

# %%
# Overview: Why Binned Fitting?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Binning data before fitting provides several benefits:
#
# 1. **Noise reduction** — Averaging within bins reduces scatter
# 2. **Relationship clarity** — Nonlinear patterns become visible
# 3. **Weighted fitting** — Each bin contributes equally regardless of density
# 4. **Uncertainty quantification** — Confidence and prediction intervals
# 5. **Visual clarity** — Binned points are easier to interpret than raw scatter

# %%
# Load and Prepare Data
# ^^^^^^^^^^^^^^^^^^^^^
# Filter for specific conditions (summer daytime with adequate radiation).

import diive as dv

# Load example data
df = dv.load_exampledata_parquet()

# Filter for summer months (June-September) with adequate radiation
df = df.loc[(df.index.month >= 6) & (df.index.month <= 9)].copy()
df = df.loc[df['Rg_f'] > 20]  # Daytime data only

print(f"Data Summary:")
print(f"  Records: {len(df)}")
print(f"  Time range: {df.index.min().date()} to {df.index.max().date()}")

# Select variables for analysis
xcol = 'Tair_f'  # Air temperature (°C)
ycol = 'VPD_f'   # Vapor pressure deficit (hPa)

print(f"\nVariables:")
print(f"  X (independent): {xcol} - {df[xcol].min():.1f} to {df[xcol].max():.1f} °C")
print(f"  Y (dependent): {ycol} - {df[ycol].min():.2f} to {df[ycol].max():.2f} hPa")

# %%
# Create and Run Fitter
# ^^^^^^^^^^^^^^^^^^^^^
# Configure BinFitterCP with binning and fitting parameters.

bf = dv.BinFitterCP(
    df=df,
    xcol=xcol,           # X variable (predictor)
    ycol=ycol,           # Y variable (response)
    predict_max_x=None,  # Use data max for predictions
    predict_min_x=None,  # Use data min for predictions
    n_predictions=1000,  # Number of fit points for smooth curve
    n_bins_x=10,         # Divide X data into 10 equal-width bins
    bins_y_agg='mean',   # Aggregate Y values by mean per bin ('mean' or 'median')
    fit_type='quadratic_offset'  # Polynomial: y = ax² + bx + c
)

bf.run()

print(f"\nFitting Configuration:")
print(f"  Number of bins: 10")
print(f"  Aggregation method: mean")
print(f"  Fit type: quadratic_offset (y = ax² + bx + c)")

# %%
# Access Fit Results
# ^^^^^^^^^^^^^^^^^^
# Results are stored in a dictionary accessible via get_results().

results = bf.get_results()

print(f"\nAvailable results:")
print(f"  Keys: {list(results.keys())}")

# %%
# Examine Input Data and Binning
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# See how the original data was binned into groups.

input_df = results['input_df']
print(f"\nInput Data (with bin assignments):")
print(f"  Shape: {input_df.shape}")
print(f"  Columns: {list(input_df.columns)}")
print(f"  First 5 rows:")
print(input_df.head())

print(f"\nBinned X values (means per bin):")
bins_x = results['bins_x']
print(f"  {bins_x.values}")

print(f"\nBinned Y values (means per bin):")
bins_y = results['bins_y']
print(f"  {bins_y.values}")

print(f"\nValues per bin:")
n_vals = results['n_vals_per_bin']
print(f"  Min: {int(n_vals['min'])}, Max: {int(n_vals['max'])}")

# %%
# Examine Fit Equation and Parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Extract the fitted polynomial equation and its parameters.

fit_eq_str = results['fit_equation_str']
fit_type = results['fit_type']
params_opt = results['fit_params_opt']
params_cov = results['fit_params_cov']
r2 = results['fit_r2']

print(f"\nFit Equation: {fit_eq_str}")
print(f"Fit Type: {fit_type}")
print(f"\nOptimal Parameters:")
for i, p in enumerate(params_opt):
    print(f"  Parameter {i}: {p:.6f}")

print(f"\nParameter Covariance Matrix:")
print(params_cov)

print(f"\nGoodness of Fit:")
print(f"  R²: {r2:.6f}")

# %%
# Examine Fitted Curve with Intervals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The fit_df contains the smooth fitted curve and uncertainty bands.

fit_df = results['fit_df']
print(f"\nFitted Curve Data:")
print(f"  Shape: {fit_df.shape}")
print(f"  Columns: {list(fit_df.columns)}")

print(f"\nFirst 5 fitted values:")
print(fit_df.head())

print(f"\nLast 5 fitted values (high X end):")
print(fit_df.tail())

print(f"\nInterval Interpretation:")
print(f"  fit_y: Fitted mean value ± standard error")
print(f"  nom: Nominal (central) fitted value")
print(f"  nom_lower_ci95 / nom_upper_ci95: 95% confidence interval (mean prediction)")
print(f"  lower_predband / upper_predband: 95% prediction interval (individual values)")

# %%
# Comparison: Central Estimate vs Intervals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Understand the difference between confidence and prediction intervals.

idx_mid = len(fit_df) // 2  # Middle of the curve
row = fit_df.iloc[idx_mid]

print(f"\nAt X ~= {row['fit_x']:.2f}:")
print(f"  Fitted value (nom): {row['nom']:.3f}")
print(f"  Std error: {row['std']:.3f}")
print(f"  95% CI (confidence interval): [{row['nom_lower_ci95']:.3f}, {row['nom_upper_ci95']:.3f}]")
print(f"    (Uncertainty in the mean fit)")
print(f"  95% PI (prediction interval): [{row['lower_predband']:.3f}, {row['upper_predband']:.3f}]")
print(f"    (Uncertainty for individual new values)")
print(f"\nNote: Prediction interval is always wider than confidence interval")

# %%
# Create Visualization
# ^^^^^^^^^^^^^^^^^^^^
# Generate a plot showing binned data, fit curve, and uncertainty bands.

print(f"\nGenerating visualization...")

bf.showplot(
    show_unbinned_data=True,      # Show original scatter
    show_bin_variation=True,       # Show variation within bins
    highlight_year=None,           # Don't highlight specific year
    bin_size=90,                   # Bin marker size
    bin_color='black',             # Bin point color
    bin_edgecolor='None',          # No edge around points
    bin_alpha=0.9,                 # Transparency of bins
    fitline_color='red',           # Fitted curve color
    showfit=True,                  # Show the fit curve
    show_prediction_interval=True, # Show uncertainty bands
    xlim=None,                     # Auto-scale X axis
    ylim=None,                     # Auto-scale Y axis
    title='Air Temperature vs Vapor Pressure Deficit\n(Summer Daytime, 2013-2022)',
    xlabel='Air Temperature (°C)',
    ylabel='VPD (hPa)'
)

print("[Plot displayed]")

# %%
# Summary
# ^^^^^^^
# Key takeaways from binned curve fitting:

print(f"\n" + "=" * 60)
print(f"SUMMARY")
print(f"=" * 60)
print(f"\n1. DATA BINNING:")
print(f"   - Divided {len(df)} records into 10 equal-width bins")
print(f"   - Each bin contains {int(n_vals['min'])}-{int(n_vals['max'])} values")

print(f"\n2. RELATIONSHIP:")
print(f"   - Equation: {fit_eq_str}")
print(f"   - Explains {100 * r2:.2f}% of variance (R²)")

print(f"\n3. UNCERTAINTY:")
print(f"   - Confidence interval: Uncertainty in the fitted mean")
print(f"   - Prediction interval: Uncertainty for future observations")
print(f"   - Both reflect inherent variability in the data")

print(f"\n4. APPLICATIONS:")
print(f"   - Parameterize ecosystem models")
print(f"   - Understand driver-response relationships")
print(f"   - Compare relationships across sites/periods")
print(f"   - Generate synthetic data within uncertainty bounds")
