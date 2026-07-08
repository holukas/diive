"""
CORE.IO.DB.INFLUX.COMMON: SHARED INFLUXDB SCHEMA CONSTANTS & HELPERS
===================================================================

The tag-column model diive uses when reading from / writing to InfluxDB, plus
the timezone helper shared by download and upload.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from datetime import timedelta, timezone

#: Columns treated as InfluxDB *tags* (everything else in an upload frame is the
#: field, i.e. the variable data column). Used both to validate upload frames and
#: to select which columns to keep on download.
TAGS = [
    'site',
    'varname',
    'units',
    'raw_varname',
    'raw_units',
    'hpos',
    'vpos',
    'repl',
    'data_raw_freq',
    'freq',
    'filegroup',
    'config_filetype',
    'data_version',
    'gain',
    'offset',
]


def convert_ts_to_timezone(timezone_offset_to_utc_hours: int | float,
                           timestamp_index):
    """Convert a UTC-aware timestamp to a fixed offset from UTC.

    All data in the database are stored in UTC. This converts a UTC-aware
    timestamp (Series/DatetimeIndex) to the timezone given as a fixed offset to
    UTC in hours, e.g. ``1`` for CET (winter time), ``-5`` for US Eastern
    (winter time), or ``5.5`` for India.

    A *fixed* offset is applied exactly as given (no daylight-saving
    transitions). Negative and fractional offsets are supported; fractional
    hours are rounded to the nearest minute.

    Args:
        timezone_offset_to_utc_hours: offset to UTC in hours (may be negative or
            fractional), e.g. ``1`` for UTC+01:00.
        timestamp_index: a pandas Series/DatetimeIndex with tz-aware (UTC)
            timestamps.

    Returns:
        The timestamps converted to the requested fixed offset.
    """
    offset_minutes = round(timezone_offset_to_utc_hours * 60)
    return timestamp_index.dt.tz_convert(timezone(timedelta(minutes=offset_minutes)))
