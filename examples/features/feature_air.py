"""
=========================
Air Variable Calculations
=========================

Demonstrates variable calculations for atmospheric air properties including
aerodynamic resistance, dry air density, and related thermodynamic calculations
commonly used in eddy covariance flux processing.

Best for: Understanding air property calculations and their role in flux processing.
"""

# %%
# Aerodynamic Resistance Calculation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Calculate aerodynamic resistance from wind speed and friction velocity.
# Aerodynamic resistance (ra) represents the resistance to momentum and heat
# transfer between the canopy and measurement height, using the relationship:
# ra = u / ustar^2

import numpy as np
import pandas as pd

import diive as dv

# Create synthetic data: 30-minute time series for 3 days
dates = pd.date_range('2024-01-01', periods=144, freq='30min')

# Realistic wind speed (0.5-4 m/s) with diurnal pattern
u_ms = pd.Series(
    1.5 + 1.5 * np.sin(np.arange(144) * 2 * np.pi / 48),
    index=dates,
    name='u_ms'
)

# Realistic ustar (0.1-0.6 m/s) with diurnal pattern
ustar_ms = pd.Series(
    0.25 + 0.2 * np.sin(np.arange(144) * 2 * np.pi / 48),
    index=dates,
    name='ustar_ms'
)

print("Aerodynamic Resistance Calculation")
print("=" * 50)
print(f"Wind speed range: {u_ms.min():.2f} to {u_ms.max():.2f} m/s")
print(f"Friction velocity range: {ustar_ms.min():.2f} to {ustar_ms.max():.2f} m/s")

# Calculate aerodynamic resistance
ra = dv.variables.aerodynamic_resistance(u_ms=u_ms, ustar_ms=ustar_ms)

print(f"\nAerodynamic resistance range: {ra.min():.2f} to {ra.max():.2f} s/m")
print(f"Median aerodynamic resistance: {ra.median():.2f} s/m")

# %%
# Dry Air Density Calculation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Calculate dry air density from total air density and water vapor density.
# Dry air density is calculated by subtracting water vapor density from
# total moist air density, essential for eddy covariance flux calculations.

dates = pd.date_range('2024-01-01', periods=144, freq='30min')

# Realistic total moist air density (1.15-1.25 kg/m³) with temperature variations
rho_a = pd.Series(
    1.20 - 0.05 * np.sin(np.arange(144) * 2 * np.pi / 48),
    index=dates,
    name='rho_a'
)

# Realistic water vapor density (absolute humidity: 5-15 g/m³ ≈ 0.005-0.015 kg/m³)
rho_v = pd.Series(
    0.010 + 0.005 * np.sin(np.arange(144) * 2 * np.pi / 48),
    index=dates,
    name='rho_v'
)

print("\nDry Air Density Calculation")
print("=" * 50)
print(f"Total air density range: {rho_a.min():.4f} to {rho_a.max():.4f} kg/m³")
print(f"Water vapor density range: {rho_v.min():.4f} to {rho_v.max():.4f} kg/m³")

# Calculate dry air density
rho_d = dv.variables.dry_air_density(rho_a=rho_a, rho_v=rho_v)

print(f"\nDry air density range: {rho_d.min():.4f} to {rho_d.max():.4f} kg/m³")
print(f"Dry air density is {(rho_a - rho_v).mean():.4f} kg/m³ on average")
