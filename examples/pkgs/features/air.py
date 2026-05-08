"""
Examples for air variable calculations using air module.

Run this script to see air calculation results:
    python examples/createvar/air.py
"""
import numpy as np
import pandas as pd

import diive as dv


def example_aerodynamic_resistance():
    """Calculate aerodynamic resistance from wind speed and friction velocity.

    Demonstrates calculating aerodynamic resistance (ra) using the momentum
    transfer approximation: ra = u / ustar^2. This is commonly used in
    flux tower processing to assess transfer efficiency between the canopy
    and measurement height.
    """
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

    print("Example 1: Calculate aerodynamic resistance")
    print(f"Wind speed range: {u_ms.min():.2f} to {u_ms.max():.2f} m/s")
    print(f"Friction velocity range: {ustar_ms.min():.2f} to {ustar_ms.max():.2f} m/s")

    # Calculate aerodynamic resistance
    ra = dv.aerodynamic_resistance(u_ms=u_ms, ustar_ms=ustar_ms)

    print(f"Aerodynamic resistance range: {ra.min():.2f} to {ra.max():.2f} s/m")
    print(f"Median aerodynamic resistance: {ra.median():.2f} s/m\n")


def example_dry_air_density():
    """Calculate dry air density from total air density and water vapor density.

    Demonstrates calculating the partial density of dry air by subtracting
    water vapor density from total moist air density. This is used in
    eddy covariance flux calculations and atmospheric chemistry.
    """
    # Create synthetic data: 30-minute time series for 3 days
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

    print("Example 2: Calculate dry air density")
    print(f"Total air density range: {rho_a.min():.4f} to {rho_a.max():.4f} kg/m³")
    print(f"Water vapor density range: {rho_v.min():.4f} to {rho_v.max():.4f} kg/m³")

    # Calculate dry air density
    rho_d = dv.dry_air_density(rho_a=rho_a, rho_v=rho_v)

    print(f"Dry air density range: {rho_d.min():.4f} to {rho_d.max():.4f} kg/m³")
    print(f"Dry air density is {(rho_a - rho_v).mean():.4f} kg/m³ on average\n")


if __name__ == '__main__':
    example_aerodynamic_resistance()
    example_dry_air_density()
