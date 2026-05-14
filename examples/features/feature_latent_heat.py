"""
========================================
Latent Heat of Vaporization Calculation
========================================

Calculate latent heat of vaporization as a function of air temperature.

Latent heat is the energy required for water evaporation and varies with
temperature. Essential for converting latent heat flux measurements to
evapotranspiration rates.

Best for: Energy balance calculations, evapotranspiration conversions.
"""

# %%
# Latent Heat of Vaporization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Latent heat of vaporization is the energy required to convert liquid water
# to water vapor. It varies with temperature: warmer air requires less energy
# to evaporate water. This is critical for converting measured latent heat
# flux (energy) to evapotranspiration (mass).

import numpy as np
import pandas as pd
import diive as dv

# %%
# Calculate latent heat across temperature range
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Temperature range typical for ecosystems (0-40°C)

temperatures_c = np.linspace(0, 40, 50)
ta = pd.Series(temperatures_c, name='ta')

print("Latent Heat of Vaporization")
print("=" * 50)
print(f"Temperature range: {ta.min():.1f} to {ta.max():.1f}°C")

# Calculate latent heat
lv = dv.latent_heat_of_vaporization(ta=ta)

print(f"\nLatent heat range: {lv.min():.0f} to {lv.max():.0f} J/kg")
print(f"At 20°C: {lv.iloc[20]:.0f} J/kg (typical reference)")
print(f"Temperature effect: {(lv.iloc[0] - lv.iloc[-1]):.0f} J/kg difference over 40°C range")

# %%
# Interpretation
# ^^^^^^^^^^^^^^
# Latent heat decreases with increasing temperature. This relationship is important
# because it means the same latent heat flux will convert to different evapotranspiration
# rates depending on air temperature. Warm days produce more ET per unit LE than cool days.

df_result = pd.DataFrame({
    'Temperature (°C)': ta,
    'Latent Heat (J/kg)': lv
})

print("\nSample values:")
print(df_result.iloc[::5].to_string(index=False))

# Temperature sensitivity
lv_range = lv.max() - lv.min()
temp_range = ta.max() - ta.min()
sensitivity = -lv_range / temp_range
print(f"\nTemperature sensitivity: {sensitivity:.0f} J/kg per °C")
print(f"  Meaning: For every 1°C increase in air temperature, latent heat")
print(f"  decreases by ~2370 J/kg. Negative because warmer air requires")
print(f"  LESS energy to evaporate water (molecules already more energetic).")
