import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series


def potrad_eot(timestamp_index: pd.DatetimeIndex, lat: float, lon: float, utc_offset: int,
               use_atmospheric_transmission=False) -> pd.Series:
    """
    Calculate Potential Shortwave Radiation, uses equation of time. Alternative approach to `potrad`.
    Default is Top-of-Atmosphere (TOA). Set use_atmospheric_transmission=True for clear-sky surface approximation.

    Example:
        See `examples/createvar/potentialradiation.py` for complete examples.
    """

    # Input validation
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} is out of range (-90 to 90).")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} is out of range (-180 to 180).")

    # Constants
    S_SC = 1361  # Solar constant (W/m²)
    d_y = 365.25  # Average days per year
    d_r = 173  # Summer solstice DOY
    phi_r = np.deg2rad(23.45)  # Tropic of Cancer
    phi = np.deg2rad(lat)  # Site latitude in radians

    # Time calculations
    # Ensures we are working with a Series/Index that supports .dt accessors immediately
    timestamp_index = pd.to_datetime(timestamp_index)
    res = pd.DataFrame(index=timestamp_index)
    res['utc_time'] = timestamp_index - pd.Timedelta(hours=utc_offset)
    res['doy'] = res['utc_time'].dt.dayofyear

    # Decimal hour
    res['utc_h'] = (res['utc_time'].dt.hour +
                    res['utc_time'].dt.minute / 60.0 +
                    res['utc_time'].dt.second / 3600.0)

    # Solar Geometry

    # Solar declination (delta)
    # Uses cosine because we anchor to solstice (d_r)
    res['delta'] = phi_r * np.cos(2 * np.pi * (res['doy'] - d_r) / d_y)

    # Equation of Time (EoT)
    # Woolfs (1968) approximation
    B = 2 * np.pi * (res['doy'] - 81) / 365.0
    res['eot_min'] = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    res['eot_h'] = res['eot_min'] / 60.0

    # Solar hour angle (H)
    # (SolarTime - 12) * 15 degrees converted to radians
    lon_in_hours = lon / 15.0
    solar_time_h = res['utc_h'] + lon_in_hours + res['eot_h']
    res['H_rad'] = (solar_time_h - 12) * (np.pi / 12)

    # Radiation Calculation

    # Sine of solar elevation (sin_psi)
    res['sin_psi'] = (np.sin(phi) * np.sin(res['delta']) +
                      np.cos(phi) * np.cos(res['delta']) * np.cos(res['H_rad']))

    # Earth-Sun distance correction (eccentricity factor)
    # Earth is closer in Winter (Northern Hemisphere), further in Summer.
    # Factor ranges roughly from 0.96 to 1.03
    res['eccentricity'] = 1 + 0.033 * np.cos(2 * np.pi * res['doy'] / 365.0)

    # Calculate radiation
    # Standard TOA formula: S * eccentricity * sin(elevation)
    rad = S_SC * res['eccentricity'] * res['sin_psi']

    # Optional: Simple atmospheric transmission (Clear Sky Approximation)
    # A common simplified approximation is ~0.75 transmission or a function of elevation
    if use_atmospheric_transmission:
        # Calculate Air Mass (M = 1 / sin(elevation))
        # Clip to 0.01 to avoid division by zero at night/horizon
        sin_psi_clamped = res['sin_psi'].clip(lower=0.01)
        M = 1 / sin_psi_clamped
        tau = 0.75
        # Apply Beer-Lambert Law: Transmission decreases exponentially with path length
        rad = rad * (tau ** M)

    # Clamp night values to 0
    rad[rad < 0] = 0

    return rad


def potrad(timestamp_index: DatetimeIndex, lat: float, lon: float, utc_offset: int) -> Series:
    """
    Calculate potential shortwave-incoming radiation

    - Calculations by Stull (1988), p.257
    - Based on code from the old MeteoScreening Tool

    Example:
        See `examples/createvar/potentialradiation.py` for complete examples.

    Args:
        timestamp_index: time series index
        lat: latitude
        lon: longitude
        utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00

    Returns:
        potential radiation

    """
    if lat < -90 or lat > 90:
        raise Exception(f"Latitude {lat} (deg N) is out of range.")
    if lon < -180 or lat > 180:
        raise Exception(f"Longitude {lon} (deg E) is out of range.")
    if utc_offset < -12 or utc_offset > 12:
        raise Exception(f"UTC-offset {utc_offset} hours is out of range.")

    # Dataframe for collecting results
    res = pd.DataFrame(index=timestamp_index)

    # Solar irradiance, radiation 'constant'
    res['S'] = 1361  # W m-2   (According to Iris)
    # S = 1370  # W m-2   (Kyle, et al., 1985)

    # Average number of days per year
    res['d_y'] = 365.25

    # Day of the summer solstice
    res['d_r'] = 173

    # Latitude of the Tropic of Cancer (1. Wendekreis)
    # Convert 23.45° to radians
    res['phi_r'] = 23.45 * np.pi / 180

    res['utc_time'] = timestamp_index - pd.Timedelta(utc_offset, unit='h')
    res['utc_h'] = (
            res.utc_time.dt.hour
            + res.utc_time.dt.minute / 60
            + res.utc_time.dt.second / 3600
    )  # hour fraction
    res['utc_doy'] = res.utc_time.dt.dayofyear

    res['lambda_e'] = lon * np.pi / 180
    res['phi'] = lat * np.pi / 180

    res['delta'] = res.phi_r * np.cos(2 * np.pi * (res.utc_doy - res.d_r) / res.d_y)

    res['sin_psi'] = (np.sin(res.phi) * np.sin(res.delta) -
                      np.cos(res.phi) * np.cos(res.delta) *
                      np.cos((np.pi * res.utc_h) / 12 + res.lambda_e))

    # Calculating radiation
    # in W/m^2
    rad = res.S * res.sin_psi
    rad.values[rad < 0] = 0
    res['SW_IN_POT'] = rad

    # Calculating azimut
    # in degrees 0-360, S is 0
    res['azimut'] = (360 * res.utc_h / 24 + lon + 180) % 360

    # Calculating elevation
    # in deg (-90) to 90
    res['elevation'] = np.arcsin(res.sin_psi) * 180 / np.pi

    return res['SW_IN_POT']

# def solar(data, var, idx, param):
#     """
#     SOLAR
#     -----
#
#     Calculates theoretical azimut, elevation and radiation
#
#     Arguments:
#         var:   None
#         param: _RAD_ref_, _AZI_ref_, _ELE_ref_, lat, lon, UTC_offset
#
#     Note:
#         _RAD_ref_, _AZI_ref_, _ELE_ref_: new names for the theoretical
#                                          radiation, azimut and sun elevation.
#         lat, lon (float): Latitude/Longitude of measurement.
#         UTC_offset (float): hours offset to UTC.
#
#         Usually we use (47.286417, 7.733750, 1)
#     """
#     if len(param) != 6:
#         logging.error('SOLAR: invalid parameters. '
#                       'Expected (str, str, str, float, float, float).')
#         return
#     lat, lon, offset = map(parse_number, param[3:])
#     if lat < -90 or lat > 90:
#         logging.warning('Latitude %.1f (deg N) is out of range.', lat)
#     if lon < -180 or lat > 180:
#         logging.warning('Longitude %.1f (deg E) is out of range.', lon)
#     if offset < -12 or offset > 12:
#         logging.warning('UTC-offset %.1f hours is out of range.', offset)
#
#     # Calculations by Stull (1988), p.257
#
#     # radiation 'constant'
#     #     S = 1370 # W/m^2   (Kyle, et al., 1985)
#     S = 1361  # W/m^2   (According to Iris)
#
#     d_y = 365.25
#     d_r = 173  # summer solstice
#     phi_r = 23.45 * np.pi / 180  # tropic of cancer (1. Wendekreis)
#
#     utc_time = data.index[idx] - pd.Timedelta(offset, unit='h')
#     utc_h = utc_time.hour + utc_time.minute / 60 + utc_time.second / 3600  # hour fraction
#     utc_doy = utc_time.dayofyear
#
#     lambda_e = lon * np.pi / 180
#     phi = lat * np.pi / 180
#
#     delta = phi_r * np.cos(2 * np.pi * (utc_doy - d_r) / d_y)
#
#     sin_psi = (np.sin(phi) * np.sin(delta) -
#                np.cos(phi) * np.cos(delta) *
#                np.cos((np.pi * utc_h) / 12 + lambda_e))
#
#     # Calculating radiation
#     if param[0].upper() != 'NONE':
#         # in W/m^2
#         rad = S * sin_psi
#         rad.values[rad < 0] = 0
#         data.loc[idx, param[0]] = rad
#
#     # Calculating azimut
#     if param[1].upper() != 'NONE':
#         # in degrees 0-360, S is 0
#         data.loc[idx, param[1]] = (360 * utc_h / 24 + lon + 180) % 360
#
#     # Calculating elevation
#     if param[2].upper() != 'NONE':
#         # in deg (-90) to 90
#         data.loc[idx, param[2]] = np.arcsin(sin_psi) * 180 / np.pi
#
#     return data
