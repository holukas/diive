import numpy as np
import pandas as pd


def aerodynamic_resistance(u_ms: pd.Series, ustar_ms: pd.Series) -> pd.Series:
    """Calculates aerodynamic resistance (ra) using wind speed and friction velocity.

    This uses the momentum transfer approximation: ra = u / ustar^2.
    Reference: Kittler et al. (2017), eq.(5).

    Args:
        u_ms (pd.Series): Horizontal wind speed in [m s-1].
        ustar_ms (pd.Series): Friction velocity (ustar) in [m s-1].

    Returns:
        pd.Series: Aerodynamic resistance (ra) in [s m-1].
        Returns NaN where ustar is 0 or NaN.
    """
    # Filter invalid ustar values to prevent DivisionByZero
    # Replace 0 or negative values with NaN
    # (Physically, ustar cannot be 0 for valid turbulence calculations)
    ustar_clean = ustar_ms.copy()
    ustar_clean[ustar_clean <= 0] = np.nan

    # Calculate resistance
    # ra = u / (u_*)^2
    ra = u_ms / (ustar_clean ** 2)

    ra.name = "AERODYNAMIC_RESISTANCE"

    # Report
    valid_count = ra.count()
    if valid_count < len(ra):
        dropped = len(ra) - valid_count
        print(f"Warning: {dropped} data points resulted in NaN due to ustar <= 0.")
    print(f"Aerodynamic resistance: Mean = {ra.mean():.2f} s m-1")
    return ra


# # old version from SCOP scripts
# def aerodynamic_resistance(u_ms, ustar_ms):
#     """Calculate aerodynamic resistance
#
#     :param u_ms: series, horizontal wind speed (m s-1)
#     :param ustar_ms: series, ustar (m s-1)
#     :return:
#     series, aerodynamic resistance (s m-1)
#     """
#     ra = u_ms / ustar_ms / ustar_ms  # This is the same as: ra = u_ms / (ustar_ms)^2 that is mentioned in Kittler et al. (2017), eq.(5)
#     print(f"Aerodynamic resistance, mean = {ra.mean():.2f} s m-1")
#     return ra


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
    # Calculate dry air density using simple subtraction
    rho_d = rho_a - rho_v

    # Check for physical impossibility (Dry density cannot be negative)
    if (rho_d < 0).any():
        print("Warning: Input data contains water vapor density higher than "
              "total density. Negative dry air density detected.")

    # Output statistics
    print(f"Dry air density: Mean = {rho_d.mean():.4f} kg m-3")

    rho_d.name = "DRY_AIR_DENSITY"

    return rho_d

# old version from SCOP scripts
# def dry_air_density(rho_a, rho_v, M_AIR, M_H2O):
#     """Calculate dry air density
#
#     :param rho_a: series, air density (kg m-3)
#     :param rho_v: series, water vapor density (kg m-3)
#     :param M_AIR: series, molar mass of dry air (kg mol-1)
#     :param M_H2O: series, molar mass of water vapor (kg mol-1)
#     :return:
#     series, dry air density (kg m-3)
#     """
#     rho_d = rho_a + rho_v.multiply(M_AIR / M_H2O - 1)
#     print(f"Dry air density, mean = {rho_d.mean():.2f} kg m-3")
#     return rho_d
