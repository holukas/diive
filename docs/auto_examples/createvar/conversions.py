"""
Examples for unit conversions and variable transformations using conversions module.

Run this script to see conversion results:
    python examples/createvar/conversions.py
"""
import numpy as np
import pandas as pd

import diive as dv


def example_air_temp_from_sonic_temp():
    """Calculate air temperature from sonic temperature and water vapor concentration.

    Demonstrates calculating true air temperature from sonic temperature,
    which accounts for the effects of water vapor on sonic wave speed.
    This is important for accurate flux tower temperature measurements.
    """
    # Create synthetic data: 30-minute time series for 3 days
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

    print("Example 1: Calculate air temperature from sonic temperature")
    print(f"Sonic temperature range: {sonic_temp.min():.2f} to {sonic_temp.max():.2f} K "
          f"({sonic_temp.min()-273.15:.1f} to {sonic_temp.max()-273.15:.1f}°C)")
    print(f"Water vapor concentration: {h2o.min():.4f} to {h2o.max():.4f} mol/mol")

    # Calculate air temperature
    ta = dv.air_temp_from_sonic_temp(sonic_temp=sonic_temp, h2o=h2o)

    print(f"Air temperature range: {ta.min():.2f} to {ta.max():.2f} K "
          f"({ta.min()-273.15:.1f} to {ta.max()-273.15:.1f}°C)")
    print(f"Temperature correction: {(sonic_temp - ta).mean():.3f} K on average\n")


def example_latent_heat_of_vaporization():
    """Calculate latent heat of vaporization as a function of air temperature.

    Demonstrates calculating the energy required for water evaporation,
    which varies with temperature. This is essential for converting between
    latent heat flux and evapotranspiration.
    """
    # Create synthetic data: temperature range 0-40°C (273-313 K)
    temperatures_c = np.linspace(0, 40, 50)
    ta = pd.Series(temperatures_c, name='ta')

    print("Example 2: Calculate latent heat of vaporization")
    print(f"Temperature range: {ta.min():.1f} to {ta.max():.1f}°C")

    # Calculate latent heat
    lv = dv.latent_heat_of_vaporization(ta=ta)

    print(f"Latent heat range: {lv.min():.0f} to {lv.max():.0f} J/kg")
    print(f"At 20°C: {lv[20]:.0f} J/kg (typical reference)")
    print(f"Temperature effect: {(lv.iloc[0] - lv.iloc[-1]):.0f} J/kg difference over 40°C range\n")


def example_et_from_le():
    """Convert latent heat flux to evapotranspiration rate.

    Demonstrates converting latent energy flux (W/m²) to evapotranspiration
    rate (mm H₂O/h), accounting for temperature-dependent latent heat.
    Compares calculated ET to EddyPro-derived ET values.
    """
    from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN

    # Load actual flux tower data
    df, meta = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
    le = df['LE'].copy()
    et_eddypro = df['ET'].copy()  # Reference ET from EddyPro (mm/h)
    ta = df['TA_1_1_1'].copy()

    print("Example 3: Convert latent heat flux to evapotranspiration")
    print(f"Latent heat flux range: {le.min():.1f} to {le.max():.1f} W/m²")
    print(f"Air temperature range: {ta.min():.1f} to {ta.max():.1f}°C")

    # Convert to evapotranspiration
    et = dv.et_from_le(le=le, ta=ta)

    # Compare with EddyPro-derived ET
    et_valid = et.dropna()
    et_eddypro_valid = et_eddypro[et_valid.index].dropna()

    if len(et_eddypro_valid) > 0:
        # Calculate mean absolute difference
        diff = (et_valid - et_eddypro_valid).abs().mean()
        print(f"\nCalculated ET range: {et_valid.min():.4f} to {et_valid.max():.4f} mm/h")
        print(f"EddyPro ET range: {et_eddypro_valid.min():.4f} to {et_eddypro_valid.max():.4f} mm/h")
        print(f"Mean absolute difference: {diff:.4f} mm/h")

    print(f"Mean calculated ET: {et_valid.mean():.4f} mm/h\n")


if __name__ == '__main__':
    example_air_temp_from_sonic_temp()
    example_latent_heat_of_vaporization()
    example_et_from_le()
