import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series


def potrad(timestamp_index: DatetimeIndex, lat: float, lon: float, utc_offset: int) -> Series:
    """
    Calculate potential shortwave-incoming radiation

    - Calculations by Stull (1988), p.257
    - Based on code from the old MeteoScreening Tool

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


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    df = load_exampledata_parquet()
    f = df.index.year == 2018
    df = df[f].copy()
    sw_in_pot = potrad(timestamp_index=df.index,
                       lat=47.286417,
                       lon=7.733750,
                       utc_offset=1)
    # x = sw_in_pot.groupby([sw_in_pot.index.month, sw_in_pot.index.time]).mean().unstack().T
    # x.plot()
    HeatmapDateTime(series=sw_in_pot).show()
    # import matplotlib.pyplot as plt
    # sw_in_pot.plot()
    # plt.show()


if __name__ == '__main__':
    example()

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
