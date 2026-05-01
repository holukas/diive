import pandas as pd


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
        See `examples/createvar/conversions.py` for complete examples.
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
        See `examples/createvar/conversions.py` for complete examples.
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
        See `examples/createvar/conversions.py` for complete examples.
    """
    _lambda = latent_heat_of_vaporization(ta)
    et = le / _lambda  # kg m-2 s-1 = mm s-1
    et = et.multiply(3600)  # mm h-1
    return et
