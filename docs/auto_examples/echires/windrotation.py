"""
Examples for wind rotation and coordinate system transformation.

Wind rotation (tilt correction) aligns the coordinate system with mean wind
direction, enabling proper calculation of turbulent fluctuations for eddy
covariance flux calculations.

Run this script to see WindRotation2D examples:
    python examples/echires/windrotation.py

See Also
--------
diive.WindRotation2D : Wind rotation and tilt correction class documentation.
"""
import pandas as pd

import diive as dv


def example_windrotation_synthetic():
    """Calculate turbulent fluctuations using wind rotation (tilt correction).

    Demonstrates WindRotation2D class by:
    1. Creating synthetic wind components (u, v, w) and scalar data
    2. Rotating coordinate system to align with mean wind
    3. Calculating turbulent fluctuations (deviations from rotated means)
    4. Returning rotated wind components and scalar fluctuations

    This example shows how coordinate rotation properly aligns the reference
    frame with mean wind direction, essential for eddy covariance analysis.
    """
    import numpy as np

    print("=" * 80)
    print("Example: Wind Rotation and Turbulent Fluctuation Calculation")
    print("=" * 80)

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

    print(f"\nData shape: {n_records} records at 10 Hz")
    print(f"\nMean wind components (before rotation):")
    print(f"  u (east-west):  {u_mean:.2f} m/s")
    print(f"  v (north-south): {v_mean:.2f} m/s")
    print(f"  w (vertical):   {w_mean:.3f} m/s")

    # Calculate mean wind speed
    wind_speed_horizontal = np.sqrt(u_mean**2 + v_mean**2)
    wind_speed_total = np.sqrt(wind_speed_horizontal**2 + w_mean**2)
    print(f"  Horizontal wind speed: {wind_speed_horizontal:.2f} m/s")
    print(f"  Total wind speed: {wind_speed_total:.2f} m/s")
    print(f"  Mean wind direction: {np.degrees(np.arctan2(v_mean, u_mean)):.1f}°")

    print(f"\nScalar data:")
    print(f"  CO2 mean: {c_mean:.1f} ppm")
    print(f"  CO2 std:  {c_turb.std():.2f} ppm")

    # Perform coordinate rotation
    wr = dv.WindRotation2D(u=u_series, v=v_series, w=w_series, c=c_series)
    primes_df = wr.get_primes()

    print(f"\n" + "=" * 80)
    print("Rotation Results")
    print("=" * 80)

    print(f"\nRotated wind components (turbulent fluctuations after tilt correction):")
    print(f"  u_TURB mean: {primes_df['u_TURB'].mean():.6f} m/s (should be ~0, is deviation from mean)")
    print(f"  v_TURB mean: {primes_df['v_TURB'].mean():.6f} m/s (should be ~0, v aligns with mean wind)")
    print(f"  w_TURB mean: {primes_df['w_TURB'].mean():.6f} m/s (should be ~0, w aligns with mean wind)")

    print(f"\nTurbulent fluctuation statistics:")
    print(f"  u_TURB std: {primes_df['u_TURB'].std():.3f} m/s (deviation from {wind_speed_total:.2f} m/s mean)")
    print(f"  v_TURB std: {primes_df['v_TURB'].std():.3f} m/s (minor turbulence, ~0 after rotation)")
    print(f"  w_TURB std: {primes_df['w_TURB'].std():.3f} m/s (minor turbulence, ~0 after rotation)")
    print(f"  CO2_TURB std: {primes_df['CO2_TURB'].std():.2f} ppm")

    # Calculate vertical flux (covariance between w and scalar)
    w_prime = primes_df['w_TURB']
    c_prime = primes_df['CO2_TURB']
    flux = (w_prime * c_prime).mean()

    print(f"\nEddy covariance flux:")
    print(f"  w'c' = {flux:.6f} (m/s)(ppm)")
    print(f"  This represents the turbulent transport of scalar")


if __name__ == '__main__':
    print("=" * 80)
    print("WindRotation2D Examples: Coordinate Rotation and Tilt Correction")
    print("=" * 80)
    print()

    example_windrotation_synthetic()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
