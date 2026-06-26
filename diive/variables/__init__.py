"""
VARIABLES: DERIVED METEOROLOGICAL VARIABLES
============================================

Calculate derived variables: radiation, thermodynamic properties, temporal features,
and synthetic data for testing. Composable functions and classes for feature creation.

Part of the diive library: https://github.com/holukas/diive
"""

# Radiation
from diive.variables.radiation import potrad, potrad_eot

# Thermodynamic
from diive.variables.thermodynamic import (
    aerodynamic_resistance,
    dry_air_density,
    calc_vpd_from_ta_rh,
    air_temp_from_sonic_temp,
    latent_heat_of_vaporization,
    et_from_le,
)

# Temporal
from diive.variables.temporal import (
    DaytimeNighttimeFlag,
    daytime_nighttime_flag_from_swinpot,
    TimeSince,
    lagged_variants,
)

# Utilities
from diive.variables.utilities import (
    generate_noisy_timeseries,
    add_impulse_noise,
    combine_variables,
    combine_variables_to_code,
)

# Classification
from diive.variables.classification import (
    classify_variable,
    VariableClass,
    CATEGORY_CARBON,
    CATEGORY_WATER,
    CATEGORY_RADIATION,
    CATEGORY_NITROGEN,
)

__all__ = [
    # radiation
    'potrad',
    'potrad_eot',
    # thermodynamic
    'aerodynamic_resistance',
    'dry_air_density',
    'calc_vpd_from_ta_rh',
    'air_temp_from_sonic_temp',
    'latent_heat_of_vaporization',
    'et_from_le',
    # temporal
    'DaytimeNighttimeFlag',
    'daytime_nighttime_flag_from_swinpot',
    'TimeSince',
    'lagged_variants',
    # utilities
    'generate_noisy_timeseries',
    'add_impulse_noise',
    'combine_variables',
    'combine_variables_to_code',
    # classification
    'classify_variable',
    'VariableClass',
    'CATEGORY_CARBON',
    'CATEGORY_WATER',
    'CATEGORY_RADIATION',
    'CATEGORY_NITROGEN',
]
