"""
========================================
Wind Rotation and Tilt Correction
========================================

Calculate turbulent fluctuations using wind rotation (coordinate transformation).

Wind rotation (tilt correction) aligns the coordinate system with mean wind
direction, enabling proper calculation of turbulent fluctuations for eddy
covariance flux calculations.

Best for: Coordinate system alignment and turbulent fluctuation extraction.
"""

# %%
# Create synthetic wind and scalar data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create synthetic high-resolution wind (10 Hz, 30 minutes) and scalar data
# with known mean components and turbulent deviations.

import pandas as pd
import numpy as np
import diive as dv

# Create synthetic wind data (10 Hz, 30 minutes = 18000 records)
n_records = 18000
np.random.seed(42)

# Mean wind components
u_mean = 3.0  # m/s, east-west
v_mean = 1.5  # m/s, north-south (tilted from u)
w_mean = 0.1  # m/s, vertical (small mean vertical wind)

# Turbulent deviations
u_turb = np.random.normal(0, 0.3, n_records)
v_turb = np.random.normal(0, 0.2, n_records)
w_turb = np.random.normal(0, 0.15, n_records)

# Full wind components
u = u_mean + u_turb
v = v_mean + v_turb
w = w_mean + w_turb

# Scalar data (e.g., CO2 concentration in ppm)
c_mean = 400.0  # ppm
c_turb = np.random.normal(0, 2.0, n_records)
c = c_mean + c_turb

# Create Series with proper names
u_series = pd.Series(u, name='u')
v_series = pd.Series(v, name='v')
w_series = pd.Series(w, name='w')
c_series = pd.Series(c, name='CO2')

print("=" * 80)
print("Synthetic Wind and Scalar Data")
print("=" * 80)
print(f"\nData shape: {n_records} records at 10 Hz")
print(f"\nMean wind components (before rotation):")
print(f"  u (east-west):   {u_mean:.2f} m/s")
print(f"  v (north-south):  {v_mean:.2f} m/s")
print(f"  w (vertical):    {w_mean:.3f} m/s")

# Calculate mean wind speed and direction
wind_speed_horizontal = np.sqrt(u_mean**2 + v_mean**2)
wind_speed_total = np.sqrt(wind_speed_horizontal**2 + w_mean**2)
print(f"\n  Horizontal wind speed: {wind_speed_horizontal:.2f} m/s")
print(f"  Total wind speed: {wind_speed_total:.2f} m/s")
print(f"  Mean wind direction: {np.degrees(np.arctan2(v_mean, u_mean)):.1f}°")

print(f"\nScalar data (CO2):")
print(f"  Mean: {c_mean:.1f} ppm")
print(f"  Std:  {c_turb.std():.2f} ppm")

# %%
# Perform coordinate rotation (tilt correction)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Apply WindRotation2D to align the coordinate system with mean wind direction
# and extract turbulent fluctuations.

wr = dv.WindRotation2D(u=u_series, v=v_series, w=w_series, c=c_series)
primes_df = wr.get_primes()

# %%
# Examine rotated components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# After rotation, the wind components should align with mean wind direction,
# and mean values should be near zero.

print(f"\n" + "=" * 80)
print("Rotation Results")
print("=" * 80)

print(f"\nRotated wind components (turbulent fluctuations after tilt correction):")
print(f"  u_TURB mean: {primes_df['u_TURB'].mean():.6f} m/s (deviation from mean wind)")
print(f"  v_TURB mean: {primes_df['v_TURB'].mean():.6f} m/s (should be ~0 after rotation)")
print(f"  w_TURB mean: {primes_df['w_TURB'].mean():.6f} m/s (should be ~0 after rotation)")

print(f"\nTurbulent fluctuation statistics:")
print(f"  u_TURB std: {primes_df['u_TURB'].std():.3f} m/s")
print(f"  v_TURB std: {primes_df['v_TURB'].std():.3f} m/s (minor turbulence, ~0 after rotation)")
print(f"  w_TURB std: {primes_df['w_TURB'].std():.3f} m/s (minor turbulence, ~0 after rotation)")
print(f"  CO2_TURB std: {primes_df['CO2_TURB'].std():.2f} ppm")

# %%
# Calculate eddy covariance flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Compute the vertical turbulent flux as covariance between vertical wind
# and scalar concentration.

w_prime = primes_df['w_TURB']
c_prime = primes_df['CO2_TURB']
flux = (w_prime * c_prime).mean()

print(f"\n" + "=" * 80)
print("Eddy Covariance Flux")
print("=" * 80)
print(f"\nVertical turbulent flux (w'c'):")
print(f"  Value: {flux:.6f} (m/s)(ppm)")
print(f"  Interpretation: Turbulent transport of scalar (CO2) by vertical wind")

print("\n[OK] Wind rotation and flux calculation complete.")
