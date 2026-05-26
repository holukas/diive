"""
VARIABLES: METEOROLOGICAL AND PHYSICAL PROPERTIES
==================================================

Derive physical variables: air properties, unit conversions, day/night flags, lagged variants,
synthetic noise, potential radiation, and time-since-condition tracking.

Part of the diive library: https://github.com/holukas/diive
"""

# Air properties
from diive.features.variables.air import aerodynamic_resistance, dry_air_density

# Conversions
from diive.features.variables.conversions import (
    air_temp_from_sonic_temp,
    latent_heat_of_vaporization,
    et_from_le,
)

# Day/night flags
from diive.features.variables.daynightflag import (
    DaytimeNighttimeFlag,
    daytime_nighttime_flag_from_swinpot,
)

# Lagged variants
from diive.features.variables.laggedvariants import lagged_variants

# Noise
from diive.features.variables.noise import (
    generate_noisy_timeseries,
    add_impulse_noise,
)

# Potential radiation
from diive.features.variables.potentialradiation import potrad, potrad_eot

# VPD
from diive.features.variables.vpd import calc_vpd_from_ta_rh

# Time since condition
from diive.features.variables.timesince import TimeSince

__all__ = [
    # air
    'aerodynamic_resistance',
    'dry_air_density',
    # conversions
    'air_temp_from_sonic_temp',
    'latent_heat_of_vaporization',
    'et_from_le',
    # daynightflag
    'DaytimeNighttimeFlag',
    'daytime_nighttime_flag_from_swinpot',
    # laggedvariants
    'lagged_variants',
    # noise
    'generate_noisy_timeseries',
    'add_impulse_noise',
    # potentialradiation
    'potrad',
    'potrad_eot',
    # timesince
    'TimeSince',
    # vpd
    'calc_vpd_from_ta_rh',
]
