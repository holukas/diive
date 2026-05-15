"""
================================================
Evapotranspiration from Latent Heat Flux
================================================

Convert latent heat flux to evapotranspiration rate by accounting for
temperature-dependent latent heat.

Latent heat flux (energy) is converted to evapotranspiration (mass) using
the temperature-dependent latent heat relationship. This example uses real
flux tower data and compares calculated ET with reference values.

Best for: Converting latent heat flux measurements to water flux (ET).
"""

# %%
# Evapotranspiration from Latent Heat Flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Latent heat flux (W m⁻²) can be converted to evapotranspiration rate
# (mm h⁻¹) using the temperature-dependent latent heat of vaporization.
# This example loads real eddy covariance flux tower data and compares
# calculated ET with EddyPro-derived reference values.

import pandas as pd
import diive as dv
from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN

# %%
# Load real flux tower data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Using example data from a Swiss flux tower (CH-AWS)

df, meta = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

# Extract variables for ET conversion
le = df['LE'].copy()           # Latent heat flux (W/m²)
et_eddypro = df['ET'].copy()   # Reference ET from EddyPro (mm/h)
ta = df['TA_1_1_1'].copy()     # Air temperature (°C)

print("Evapotranspiration from Latent Heat Flux")
print("=" * 50)
print(f"Latent heat flux range: {le.min():.1f} to {le.max():.1f} W/m²")
print(f"Air temperature range: {ta.min():.1f} to {ta.max():.1f}°C")
print(f"Number of records: {len(le)}")

# %%
# Convert latent heat to evapotranspiration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Apply temperature-dependent conversion

et_calculated = dv.et_from_le(le=le, ta=ta)

# %%
# Compare with EddyPro reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Validate calculated ET against EddyPro-derived values

et_valid = et_calculated.dropna()
et_eddypro_valid = et_eddypro[et_valid.index].dropna()

print(f"\nCalculated ET range: {et_valid.min():.4f} to {et_valid.max():.4f} mm/h")
print(f"EddyPro ET range: {et_eddypro_valid.min():.4f} to {et_eddypro_valid.max():.4f} mm/h")

if len(et_eddypro_valid) > 0:
    # Calculate agreement metrics
    diff = (et_valid - et_eddypro_valid).abs()
    mae = diff.mean()
    rmse = (((et_valid - et_eddypro_valid) ** 2).mean()) ** 0.5

    print(f"\nAgreement metrics:")
    print(f"  Mean Absolute Error: {mae:.4f} mm/h")
    print(f"  Root Mean Square Error: {rmse:.4f} mm/h")
    print(f"  Mean calculated ET: {et_valid.mean():.4f} mm/h")
    print(f"  Mean EddyPro ET: {et_eddypro_valid.mean():.4f} mm/h")

    # Correlation
    corr = et_valid.corr(et_eddypro_valid)
    print(f"  Correlation: {corr:.4f}")

# %%
# Visualize ET Comparison using Heatmaps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create heatmaps comparing calculated ET, EddyPro reference, and the difference.
# Reveals spatial and temporal patterns in ET agreement.

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

fig = plt.figure(facecolor='white', figsize=(18, 6), constrained_layout=True)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

dv.plot_heatmap_datetime(series=et_calculated).plot(ax=ax1)
dv.plot_heatmap_datetime(series=et_eddypro_valid).plot(ax=ax2)
dv.plot_heatmap_datetime(series=(et_valid - et_eddypro_valid)).plot(ax=ax3)

ax1.set_title("ET from Latent Heat Flux (mm/h)", fontsize=12, fontweight='bold')
ax2.set_title("ET from EddyPro Reference (mm/h)", fontsize=12, fontweight='bold')
ax3.set_title("Difference (calculated - reference)", fontsize=12, fontweight='bold')

ax2.tick_params(left=True, labelleft=False)
ax3.tick_params(left=True, labelleft=False)

fig.show()

# %%
# Output: Calculated Evapotranspiration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The main output is the calculated ET time series

print(f"\nCalculated ET time series (mm/h):")
print(f"  Shape: {et_calculated.shape}")
print(f"  Data type: {et_calculated.dtype}")
print(f"  First 10 values:")
print(et_calculated.head(10).to_string())

# %%
# Interpretation
# ^^^^^^^^^^^^^^
# The calculated ET closely matches EddyPro values because both use the same
# formula and temperature-dependent latent heat relationship. Small differences
# may arise from temperature measurement timing or sensor calibration.
# This conversion is essential for water cycle analysis and crop water demand
# estimation in agricultural and ecosystem studies.
#
# The et_calculated series can now be used directly for:
# - Daily/monthly cumulative water loss
# - Comparison with precipitation
# - Crop water requirement analysis
# - Water balance studies
