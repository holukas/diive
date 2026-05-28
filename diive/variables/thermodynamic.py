"""
THERMODYNAMIC: AIR PROPERTIES AND ENERGY CONVERSIONS
=====================================================

Calculate thermodynamic variables: air density, aerodynamic resistance,
vapor pressure deficit, temperature conversions, and energy conversions.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd
from numpy import exp
from pandas import DataFrame, Series

from diive.core.utils.console import info, warn


def aerodynamic_resistance(u_ms: pd.Series, ustar_ms: pd.Series) -> pd.Series:
    """Calculates aerodynamic resistance (ra) using wind speed and friction velocity.

    "The aerodynamic resistance is determined directly from the friction velocity USTAR (m s-1)
    and the horizontal wind speed u (m s-1) as ra = u / (USTAR^2) according to the simplified bulk
    approach described in Stull [1988]." (Kittler et al., 2017)

    This uses the momentum transfer approximation: ra = u / ustar^2.
    Reference: Kittler et al. (2017), eq.(5).

    Args:
        u_ms (pd.Series): Horizontal wind speed in [m s-1].
        ustar_ms (pd.Series): Friction velocity (ustar) in [m s-1].

    Returns:
        pd.Series: Aerodynamic resistance (ra) in [s m-1].
        Returns NaN where ustar is 0 or NaN.
    """
    ustar_clean = ustar_ms.copy()
    ustar_clean[ustar_clean <= 0] = np.nan

    ra = u_ms / (ustar_clean ** 2)

    ra.name = "AERODYNAMIC_RESISTANCE"

    valid_count = ra.count()
    if valid_count < len(ra):
        dropped = len(ra) - valid_count
        warn(f"{dropped} data points resulted in NaN due to ustar <= 0.")
    info(f"Aerodynamic resistance: Mean = {ra.mean():.2f} s m-1")
    return ra


def dry_air_density(rho_a: pd.Series, rho_v: pd.Series) -> pd.Series:
    """Calculates the partial density of dry air from moist air density.

    This function isolates the dry air component by subtracting the water vapor
    density (absolute humidity) from the total density of the moist air parcel.
    This relies on the definition of total density in a gas mixture:
    rho_total = rho_dry + rho_vapor.

    Note:
        Input series must have matching indices/lengths.
        Molar masses are not required for this calculation as the inputs
        are already in density units (kg m-3).

    Args:
        rho_a (pd.Series): Total density of the moist air (rho_total)
            in [kg m-3].
        rho_v (pd.Series): Water vapor density (Absolute Humidity)
            in [kg m-3].

    Returns:
        pd.Series: The calculated dry air density (rho_d) in [kg m-3].


    Raises:
        TypeError: If inputs are not pandas Series or numeric arrays.
        ValueError: If rho_v is larger than rho_a (resulting in negative dry density),
            though this is handled mathematically, it indicates bad input data.
    """
    rho_d = rho_a - rho_v

    if (rho_d < 0).any():
        warn("Input data contains water vapor density higher than "
             "total density. Negative dry air density detected.")

    info(f"Dry air density: Mean = {rho_d.mean():.4f} kg m-3")

    rho_d.name = "DRY_AIR_DENSITY"

    return rho_d


def calc_vpd_from_ta_rh(df: DataFrame, rh_col: str, ta_col: str) -> Series:
    """
    Calculate VPD from air temperature and relative humidity

    Args:
        df: Data
        rh_col: Name of column in *df* containing relative humidity data in %
        ta_col: Name of column in *df* containing air temperature data in °C

    Returns:
        VPD in kPa as Series

    Original code in ReddyProc:
        VPD.V.n <- 6.1078 * (1 - rH / 100) * exp(17.08085 * Tair / (234.175 + Tair))
        # See Kolle Logger Tools Software 2012 (Magnus coefficients for water between 0 and 100 degC)
        # Data vector of vapour pressure deficit (VPD, hPa (mbar))

    Reference:
        https://github.com/bgctw/REddyProc/blob/3c2b414c24900c17ab624c20b0e1726e6a813267/R/GeoFunctions.R#L96

    Checked against ReddyProc output, virtually the same, differences only 8 digits after comma.

    See Also
    --------
    examples/variables/feature_vpd.py : Complete examples with gap-filling and visualizations.

    """

    rh = df[[rh_col, ta_col]].copy()

    rh['a'] = 6.1078
    rh['b'] = 1 - df[rh_col] / 100
    rh['c'] = rh[ta_col].multiply(17.08085) / rh[ta_col].add(234.175)
    rh['cc'] = exp(rh['c'])

    rh['VPD'] = rh['a'] * rh['b'] * rh['cc']

    rh['VPD'] = rh['VPD'].multiply(0.1)
    return rh['VPD']


def air_temp_from_sonic_temp(sonic_temp: pd.Series, h2o: pd.Series) -> pd.Series:
    """
    Calculate air temperature from sonic temperature and water vapor concentration.

    This function computes the air temperature from the provided sonic temperature
    and water vapor concentration using a specific formula. The calculation takes
    into account the relationship between sonic temperature, air temperature, and
    the effect of water vapor on sonic wave speed.

    Based on the code in:
    Striednig, M., Graus, M., Märk, T. D., & Karl, T. G. (2020). InnFLUX – an open-source
        code for conventional and disjunct eddy covariance analysis of trace gas measurements:
        An urban test case. Atmospheric Measurement Techniques, 13(3), 1447–1465.
        https://doi.org/10.5194/amt-13-1447-2020
        Source code: https://www.atm-phys-chem.at/innflux/
        Source code: https://git.uibk.ac.at/acinn/apc/innflux
        Source code: https://git.uibk.ac.at/acinn/apc/innflux/-/blob/master/innFLUX_step1.m?ref_type=heads#L329


    Args:
        sonic_temp (pd.Series): Sonic temperature data in Kelvin.
        h2o (pd.Series): Water vapor concentration in the air in mol mol-1.

    Returns:
        pd.Series: Air temperature data in Kelvin.

    Example:
        See `examples/variables/feature_sonic_temp_conversion.py` for complete example.

    See Also:
        latent_heat_of_vaporization : Calculate latent heat from air temperature
        et_from_le : Convert latent heat flux to evapotranspiration
    """
    ta = sonic_temp / (1 + 0.32 * h2o)
    ta.name = "TA_SONIC"
    return ta


def latent_heat_of_vaporization(ta: pd.Series) -> pd.Series:
    """Calculate latent heat of vaporization as a function of air temperature.

    Kudos:
        Based on code of the R package 'bigleaf':
        https://rdrr.io/cran/bigleaf/src/R/meteorological_variables.r (latent.heat.vaporization)

    Reference:
        Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p.641)
        Kluwer Academic Publishers, Dordrecht, Netherlands

    Args:
        ta: air temperature (°C)

    Returns:
        Latent heat of vaporization (J kg-1)

    Example:
        See `examples/variables/feature_latent_heat.py` for complete example.

    See Also:
        et_from_le : Convert latent heat flux to evapotranspiration
    """
    k1 = 2.501
    k2 = 0.00237
    Lv = (k1 - k2 * ta) * (10 ** 6)
    return Lv


def et_from_le(le: pd.Series, ta: pd.Series) -> pd.Series:
    """Convert latent heat flux (energy) to evapotranspiration (mass).

    Kudos:
        Based on code of the R package 'bigleaf':
        https://rdrr.io/cran/bigleaf/src/R/unit_conversions.r (LE.to.ET)

    Args:
        le: latent heat flux (W m-2)
        ta: air temperature (°C)

    Returns:
        Evapotranspiration (mm H2O h-1)

    Example:
        See `examples/variables/feature_evapotranspiration.py` for complete example.

    See Also:
        latent_heat_of_vaporization : Calculate latent heat from air temperature
    """
    _lambda = latent_heat_of_vaporization(ta)
    et = le / _lambda
    et = et.multiply(3600)
    return et
