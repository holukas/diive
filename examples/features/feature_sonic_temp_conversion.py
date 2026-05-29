"""
======================================
Air Temperature from Sonic Temperature
======================================

Calculate air temperature from sonic temperature and water vapor concentration.

Sonic anemometers measure temperature indirectly through the speed of sound.
This example shows how to correct sonic temperature to true air temperature
using water vapor concentration for realistic eddy covariance measurements.

Best for: High-resolution (10 Hz, 20 Hz) eddy covariance time series correction.
"""

# %%
# Air temperature from sonic temperature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sonic anemometers measure the speed of sound between transducer pairs.
# This speed depends on both air temperature and humidity. The measured
# "sonic temperature" must be corrected using water vapor concentration
# to get true air temperature.

import numpy as np
import pandas as pd
import diive as dv

# %%
# Create realistic example data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Generate 3-day time series with diurnal patterns (typical for tower data)

# Create 30-minute time series for 3 days
dates = pd.date_range('2024-01-01', periods=144, freq='30min')

# Realistic sonic temperature (280-310 K, ~7-37°C) with diurnal pattern
sonic_temp = pd.Series(
    295 + 10 * np.sin(np.arange(144) * 2 * np.pi / 48),
    index=dates,
    name='sonic_temp'
)

# Realistic water vapor concentration (0.005-0.025 mol/mol) with diurnal pattern
h2o = pd.Series(
    0.015 + 0.01 * np.sin(np.arange(144) * 2 * np.pi / 48),
    index=dates,
    name='h2o'
)

print("Air Temperature from Sonic Temperature")
print("=" * 50)
print(f"Sonic temperature range: {sonic_temp.min():.2f} to {sonic_temp.max():.2f} K "
      f"({sonic_temp.min() - 273.15:.1f} to {sonic_temp.max() - 273.15:.1f}°C)")
print(f"Water vapor concentration: {h2o.min():.4f} to {h2o.max():.4f} mol/mol")

# %%
# Calculate air temperature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Apply the correction using the Striednig et al. (2020) formula

air_temp = dv.variables.air_temp_from_sonic_temp(sonic_temp=sonic_temp, h2o=h2o)

print(f"\nAir temperature range: {air_temp.min():.2f} to {air_temp.max():.2f} K "
      f"({air_temp.min() - 273.15:.1f} to {air_temp.max() - 273.15:.1f}°C)")
print(f"Temperature correction: {(sonic_temp - air_temp).mean():.3f} K on average")

# %%
# Comparison and interpretation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Visualize the relationship between sonic temperature, water vapor, and air temperature

df = pd.DataFrame({
    'Sonic Temp (K)': sonic_temp,
    'Air Temp (K)': air_temp,
    'H2O (mol/mol)': h2o
})

print("\nFirst 10 rows:")
print(df.head(10).to_string())

# %%
# Key insights
# ^^^^^^^^^^^^
# The correction typically reduces sonic temperature by 1-2 K depending on humidity.
# Higher water vapor concentrations require larger corrections because water vapor
# affects the speed of sound more than dry air. This correction is essential for
# accurate flux calculations in eddy covariance systems.

diff = sonic_temp - air_temp
corr = diff.corr(h2o)
print(f"\nRelationship between correction and water vapor:")
print(f"  Correlation between (Sonic T - Air T) and H2O: {corr:.3f}")
print(f"  Meaning: When H2O concentration increases, the temperature correction")
print(f"  also increases. Perfect 1.0 correlation means they move together exactly")
print(f"  due to the diurnal cycle (warm days = more moisture + larger correction).")
print(f"\nCorrection statistics:")
print(f"  Mean: {diff.mean():.3f} K")
print(f"  Range: {diff.min():.3f} to {diff.max():.3f} K")
