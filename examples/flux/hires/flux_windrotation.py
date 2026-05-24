"""
========================================
Wind Rotation and Tilt Correction
========================================

Eddy covariance processing: coordinate rotation followed by Reynolds decomposition.

Wind rotation (tilt correction) aligns the coordinate system with mean wind
direction. Reynolds decomposition then extracts turbulent fluctuations from the
rotated wind components and scalars, which are combined to compute fluxes.

"""

# %%
# Create synthetic wind and scalar data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create synthetic high-resolution wind (10 Hz, 30 minutes) and scalar data
# with known mean components and turbulent deviations.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
wind_speed_horizontal = np.sqrt(u_mean ** 2 + v_mean ** 2)
wind_speed_total = np.sqrt(wind_speed_horizontal ** 2 + w_mean ** 2)
print(f"\n  Horizontal wind speed: {wind_speed_horizontal:.2f} m/s")
print(f"  Total wind speed: {wind_speed_total:.2f} m/s")
print(f"  Mean wind direction: {np.degrees(np.arctan2(v_mean, u_mean)):.1f}°")

print(f"\nScalar data (CO2):")
print(f"  Mean: {c_mean:.1f} ppm")
print(f"  Std:  {c_turb.std():.2f} ppm")

# %%
# Step 1: Coordinate rotation (tilt correction)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# WindDoubleRotation aligns the coordinate system with mean wind direction.
# After rotation, mean(v2) ~ 0 and mean(w2) ~ 0.

wr = dv.flux.WindDoubleRotation(u=u_series, v=v_series, w=w_series)

print(f"\n" + "=" * 80)
print("Rotation Results")
print("=" * 80)
print(f"\nRotation angles:")
print(f"  theta (yaw):   {wr.theta:.4f} rad")
print(f"  phi   (pitch): {wr.phi:.4f} rad")
print(f"\nMean of rotated components (should be ~0 for v2 and w2):")
print(f"  mean(v2): {wr.v2.mean():.6f} m/s")
print(f"  mean(w2): {wr.w2.mean():.6f} m/s")

# %%
# Step 2: Reynolds decomposition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Reynolds decomposition extracts turbulent fluctuations: x' = x - mean(x).
# Applied to the rotated wind components and the scalar.

u_prime = dv.flux.reynolds_decomposition(wr.u2)
v_prime = dv.flux.reynolds_decomposition(wr.v2)
w_prime = dv.flux.reynolds_decomposition(wr.w2)
c_prime = dv.flux.reynolds_decomposition(c_series)

print(f"\n" + "=" * 80)
print("Turbulent Fluctuations (Reynolds Decomposition)")
print("=" * 80)
print(f"\nMean of fluctuations (should be ~0 by construction):")
print(f"  mean(u'): {u_prime.mean():.6f} m/s")
print(f"  mean(v'): {v_prime.mean():.6f} m/s")
print(f"  mean(w'): {w_prime.mean():.6f} m/s")
print(f"\nStd of fluctuations:")
print(f"  std(u'): {u_prime.std():.3f} m/s")
print(f"  std(v'): {v_prime.std():.3f} m/s")
print(f"  std(w'): {w_prime.std():.3f} m/s")
print(f"  std(c'): {c_prime.std():.2f} ppm")

# %%
# Calculate eddy covariance flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Compute the vertical turbulent flux as covariance between w' and c'.

flux = (w_prime * c_prime).mean()

print(f"\n" + "=" * 80)
print("Eddy Covariance Flux")
print("=" * 80)
print(f"\nVertical turbulent flux (w'c'):")
print(f"  Value: {flux:.6f} (m/s)(ppm)")
print(f"  Interpretation: Turbulent transport of scalar (CO2) by vertical wind")

print("\n[OK] Wind rotation and flux calculation complete.")

# %%
# Visualize rotation effect and turbulent flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Three panels: mean wind components before/after rotation, short vertical wind
# time series showing the mean shift removed, and the w'c' scatter that yields the flux.

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel 1: mean wind components before vs after rotation
ax = axes[0]
labels = ['u', 'v', 'w']
before = [u_series.mean(), v_series.mean(), w_series.mean()]
after = [wr.u2.mean(), wr.v2.mean(), wr.w2.mean()]
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width / 2, before, width, label='Before', color='#5B9BD5')
ax.bar(x + width / 2, after, width, label='After', color='#ED7D31')
ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Mean (m s$^{-1}$)')
ax.set_title('Mean components: before vs after rotation')
ax.legend()

# Panel 2: vertical wind time series, first 200 records (20 s at 10 Hz)
ax = axes[1]
n = 200
t = np.arange(n) / 10
ax.plot(t, w_series.values[:n], color='#5B9BD5', alpha=0.8, label='w (raw)')
ax.plot(t, wr.w2.values[:n], color='#ED7D31', alpha=0.8, label='w2 (rotated)')
ax.axhline(w_series.mean(), color='#5B9BD5', linestyle='--', linewidth=0.8)
ax.axhline(wr.w2.mean(), color='#ED7D31', linestyle='--', linewidth=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('m s$^{-1}$')
ax.set_title('Vertical wind: raw vs rotated')
ax.legend()

# Panel 3: w' vs c' scatter — the covariance that gives the flux
ax = axes[2]
ax.scatter(w_prime.values, c_prime.values, alpha=0.05, s=2, color='#5B9BD5')
coeffs = np.polyfit(w_prime, c_prime, 1)
x_line = np.linspace(w_prime.min(), w_prime.max(), 100)
ax.plot(x_line, np.polyval(coeffs, x_line), color='#ED7D31', linewidth=1.5)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlabel("w' (m s$^{-1}$)")
ax.set_ylabel("c' (ppm)")
ax.set_title(f"Turbulent flux w'c' = {flux:.5f} (m/s)(ppm)")

fig.suptitle('Synthetic data — 10 Hz, 30 min, u=3 m/s, v=1.5 m/s, w=0.1 m/s', fontsize=9, color='grey')
plt.tight_layout()
plt.show()
