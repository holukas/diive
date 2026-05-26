"""
FEATURES: FEATURE ENGINEERING
=============================

Calculate derived variables: VPD, unit conversions, day/night flags, lag features, potential radiation.
Composable 8-stage pipeline for time series feature creation.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.features import variables
from diive.features.variables import TimeSince
from diive.features.variables.air import aerodynamic_resistance
from diive.features.variables.air import dry_air_density
from diive.features.variables.conversions import air_temp_from_sonic_temp
from diive.features.variables.conversions import et_from_le
from diive.features.variables.conversions import latent_heat_of_vaporization
from diive.features.variables.daynightflag import DaytimeNighttimeFlag
from diive.features.variables.daynightflag import daytime_nighttime_flag_from_swinpot
from diive.features.variables.laggedvariants import lagged_variants
from diive.features.variables.noise import add_impulse_noise
from diive.features.variables.noise import generate_noisy_timeseries
from diive.features.variables.potentialradiation import potrad
from diive.features.variables.potentialradiation import potrad_eot
from diive.features.variables.vpd import calc_vpd_from_ta_rh

__all__ = [
    'variables',
    'TimeSince',
    'aerodynamic_resistance',
    'dry_air_density',
    'air_temp_from_sonic_temp',
    'et_from_le',
    'latent_heat_of_vaporization',
    'DaytimeNighttimeFlag',
    'daytime_nighttime_flag_from_swinpot',
    'lagged_variants',
    'add_impulse_noise',
    'generate_noisy_timeseries',
    'potrad',
    'potrad_eot',
    'calc_vpd_from_ta_rh',
]
