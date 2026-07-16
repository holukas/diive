"""
RADIATION: SOLAR RADIATION CALCULATIONS
========================================

Calculate potential shortwave-incoming radiation (SW_IN_POT).

Part of the diive library: https://github.com/holukas/diive
"""

import math

import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series

# Solar constant used by ONEFlux `get_daily_rpot` (W m-2). ONEFlux's own value;
# parity requires it.
_ONEFLUX_SOLAR_CONSTANT = 1376.0


def _oneflux_daily_rpot(lat: float, doy, hrs):
    """Port of ONEFlux ``get_daily_rpot`` (Spencer 1971 declination + eccentricity).

    The C signature accepts a longitude but never reads it: the curve is built
    symmetric about 12:00 local standard time, and the longitude / equation-of-time
    offset is applied afterwards by shifting the whole day onto true solar noon.
    """
    tthet = 2.0 * np.pi * (doy - 1.0) / 365.0

    # Hour angle from |12 - h|; cos() is even, so the curve is symmetric about 12:00.
    omega = -15.0 * np.abs(12.0 - hrs)

    decl_rad = (0.006918
                - 0.399912 * np.cos(tthet) + 0.070257 * np.sin(tthet)
                - 0.006758 * np.cos(2 * tthet) + 0.000907 * np.sin(2 * tthet)
                - 0.002697 * np.cos(3 * tthet) + 0.00148 * np.sin(3 * tthet))
    lat_rad = np.deg2rad(lat)

    theta_rad = np.arccos(np.sin(decl_rad) * np.sin(lat_rad)
                          + np.cos(decl_rad) * np.cos(lat_rad) * np.cos(np.deg2rad(omega)))

    rpot = _ONEFLUX_SOLAR_CONSTANT * (1.00011
                                      + 0.034221 * np.cos(tthet) + 0.00128 * np.sin(tthet)
                                      + 0.000719 * np.cos(2 * tthet) + 0.000077 * np.sin(2 * tthet))
    return np.maximum(rpot * np.cos(theta_rad), 0.0)


def _noaa_julian_day(year: int, month: int, day: int) -> float:
    """Port of ONEFlux ``calcJD``."""
    if month <= 2:
        year -= 1
        month += 12
    a = math.floor(year / 100)
    b = 2 - a + math.floor(a / 4)
    return math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + b - 1524.5


def _noaa_julian_century(jd: float) -> float:
    return (jd - 2451545.0) / 36525.0


def _noaa_jd_from_century(t: float) -> float:
    return t * 36525.0 + 2451545.0


def _noaa_equation_of_time(t: float) -> float:
    """Port of ONEFlux ``calcEquationOfTime``. Returns minutes of time."""
    seconds = 21.448 - t * (46.8150 + t * (0.00059 - t * 0.001813))
    e0 = 23.0 + (26.0 + seconds / 60.0) / 60.0
    omega = 125.04 - 1934.136 * t
    epsilon = e0 + 0.00256 * math.cos(math.radians(omega))
    l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    y = math.tan(math.radians(epsilon) / 2.0) ** 2

    etime = (y * math.sin(2 * math.radians(l0))
             - 2.0 * e * math.sin(math.radians(m))
             + 4.0 * e * y * math.sin(math.radians(m)) * math.cos(2 * math.radians(l0))
             - 0.5 * y * y * math.sin(4 * math.radians(l0))
             - 1.25 * e * e * math.sin(2 * math.radians(m)))
    return math.degrees(etime) * 4.0


def _noaa_solar_noon_minutes(year: int, month: int, day: int, longitude: float, zone: float) -> int:
    """Port of ONEFlux ``get_solar_noon``: true solar noon as minute-of-day.

    *longitude* and *zone* follow ONEFlux's positive-west convention (both are
    sign-flipped by the caller). ONEFlux always passes ``day_saving = 0``, so the
    daylight-saving term is omitted here.
    """
    t = _noaa_julian_century(_noaa_julian_day(year, month, day))

    # Two passes: the first uses an approximate noon to evaluate the equation of time.
    tnoon = _noaa_julian_century(_noaa_jd_from_century(t) + longitude / 360.0)
    sol_noon_utc = 720 + (longitude * 4) - _noaa_equation_of_time(tnoon)
    newt = _noaa_julian_century(_noaa_jd_from_century(t) - 0.5 + sol_noon_utc / 1440.0)
    sol_noon_utc = 720 + (longitude * 4) - _noaa_equation_of_time(newt)

    noon = sol_noon_utc - (60 * zone)
    float_hour = noon / 60.0
    hour = math.floor(float_hour)
    float_minute = 60.0 * (float_hour - math.floor(float_hour))
    minute = math.floor(float_minute)
    second = math.floor(60.0 * (float_minute - math.floor(float_minute)) + 0.5)
    if second > 59:
        minute += 1
    return int(60 * hour + minute)


def _oneflux_window_mean(timestamp_index: DatetimeIndex, lat: float, lon: float,
                         utc_offset: int, period_min: int) -> np.ndarray:
    """Mean ONEFlux potential radiation over ``[t - period/2, t + period/2)`` per timestamp.

    ONEFlux builds a whole day of 1-minute values, shifts it onto solar noon and
    then averages fixed blocks. Evaluating each averaging window directly is
    equivalent for the periods ONEFlux defines, and additionally works for periods
    that do not tile the day and for windows crossing midnight or New Year.
    """
    # ONEFlux flips the sign of both before computing solar noon (positive-west).
    longitude = -lon
    zone = -utc_offset

    starts = timestamp_index - pd.Timedelta(minutes=period_min / 2)
    epoch = starts.min().normalize()
    # An odd period puts window starts on half-minutes, which the 1-minute grid
    # cannot represent. Round half up rather than np.rint: rint rounds halves to
    # even, which would make consecutive windows step unevenly (6, 8, 6, 8 ... for
    # a 7-minute period) instead of marching uniformly. Even periods -- including
    # the 30-minute and hourly ones ONEFlux defines -- land exactly on the grid
    # and are unaffected either way.
    start_off = np.floor((starts - epoch) / pd.Timedelta(minutes=1) + 0.5).astype(np.int64)

    # One entry per calendar day any window touches.
    n_days_span = int((start_off.max() + period_min - 1) // 1440) + 1
    dates = pd.date_range(epoch, periods=n_days_span, freq='D')
    shifts = np.array([_noaa_solar_noon_minutes(d.year, d.month, d.day, longitude, zone) - 720
                       for d in dates], dtype=np.int64)
    # ONEFlux's `row`: 0-based day index within the day's OWN calendar year, so a
    # window straddling New Year stays faithful on both sides of the boundary.
    day_rows = np.array([d.dayofyear - 1 for d in dates], dtype=np.int64)

    # Accumulate one minute-offset at a time: O(len(timestamp_index)) memory rather
    # than materialising every minute of the record.
    total = np.zeros(len(timestamp_index), dtype=float)
    for k in range(period_min):
        minute = start_off + k
        day_idx = minute // 1440
        # Undo that day's shift to find the position on the unshifted curve; the
        # positions the shift vacated are 0, as in the C `shift`.
        i = (minute % 1440) - shifts[day_idx]
        inside = (i >= 0) & (i < 1440)
        i_safe = np.where(inside, i, 0)
        # Fractional DOY exactly as ONEFlux indexes it. The /365 in
        # `_oneflux_daily_rpot` is not a leap-year typo: ONEFlux keeps 365 even in
        # leap years, and parity requires keeping it.
        doy = day_rows[day_idx] + i_safe / 1440.0 + 1.0
        total += np.where(inside, _oneflux_daily_rpot(lat, doy, i_safe / 60.0), 0.0)
    return total / period_min


def potrad(timestamp_index: DatetimeIndex, lat: float, lon: float, utc_offset: int) -> Series:
    """
    Calculate potential shortwave-incoming radiation, ONEFlux/FLUXNET parity

    Faithful port of ``get_rpot`` in ONEFlux ``oneflux_steps/common/common.c``
    (https://github.com/fluxnet/ONEFlux), the routine that produces the
    ``SW_IN_POT`` column of FLUXNET/AmeriFlux/ICOS products. Its defining
    properties:

    - solar constant 1376 W m-2,
    - Spencer (1971) declination and earth-sun distance (eccentricity) series,
      which carry the +/-3.4% annual amplitude cycle,
    - true solar noon from the NOAA solar position algorithm, i.e. the equation
      of time is included, so solar noon is not pinned to a fixed clock time,
    - the returned value is the **mean over each averaging period**, computed
      from 1-minute steps, not an instantaneous value at *timestamp_index*.

    Timestamps are assumed to be TIMESTAMP_MIDDLE (the diive convention): the
    record at ``t`` covers ``[t - period/2, t + period/2)``. The averaging period
    is inferred from the median spacing of *timestamp_index* (median, so gaps do
    not skew it). ONEFlux itself only defines 30-minute and hourly output; any
    other period is handled as the natural generalisation of the same algorithm,
    including periods that do not tile the day. ONEFlux resolves potential
    radiation on a 1-minute grid, so a sub-minute record cannot average over
    anything finer and gets the 1-minute value.

    Daylight saving is not applied: *utc_offset* is the local standard time
    offset, matching FLUXNET convention.

    Args:
        timestamp_index: time series index (TIMESTAMP_MIDDLE)
        lat: latitude
        lon: longitude
        utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00

    Returns:
        potential radiation, named ``SW_IN_POT``

    Example:
        See `examples/features/feature_potentialradiation.py` for complete examples.
    """
    if lat < -90 or lat > 90:
        raise Exception(f"Latitude {lat} (deg N) is out of range.")
    if lon < -180 or lon > 180:
        raise Exception(f"Longitude {lon} (deg E) is out of range.")
    if utc_offset < -12 or utc_offset > 12:
        raise Exception(f"UTC-offset {utc_offset} hours is out of range.")

    timestamp_index = pd.to_datetime(timestamp_index)
    if len(timestamp_index) < 2:
        raise ValueError("potrad needs at least two timestamps to infer the averaging period.")

    # Median is robust against gaps in the record. Clamped to the 1-minute grid
    # ONEFlux computes on (see docstring).
    period = pd.Series(timestamp_index).diff().median()
    period_min = max(1, int(round(period.total_seconds() / 60)))

    swinpot = _oneflux_window_mean(timestamp_index, lat, lon, utc_offset, period_min)
    return Series(swinpot, index=timestamp_index, name='SW_IN_POT')
