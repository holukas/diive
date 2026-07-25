"""
COMMON: SHARED OUTLIER DETECTION UTILITIES
===========================================

Shared functions for daytime/nighttime flag generation and other detection helpers.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.variables import DaytimeNighttimeFlag

# Parameter names that changed when the day/night settings were unified across
# the outlier detectors. Old name -> new name.
_RENAMED_DAYNIGHT_PARAMS = {
    'separate_daytime_nighttime': 'separate_day_night',
}


def reject_legacy_params(unexpected: dict, detector: str, renamed: dict = None) -> None:
    """Raise for any leftover keyword argument, naming its replacement if it has one.

    Detectors accept ``**kwargs`` purely so a pre-unification call can be
    answered with a message that says what to change, instead of Python's bare
    "unexpected keyword argument". Names outside the rename table are still
    rejected, so a typo cannot pass silently through ``**kwargs``.
    """
    if not unexpected:
        return
    table = dict(_RENAMED_DAYNIGHT_PARAMS)
    if renamed:
        table.update(renamed)
    name = next(iter(unexpected))
    if name in table:
        raise TypeError(
            f"{detector}: '{name}' was renamed to '{table[name]}' when the day/night "
            f"settings were unified across the outlier detectors. Pass '{table[name]}' instead."
        )
    raise TypeError(f"{detector}.__init__() got an unexpected keyword argument '{name}'")


def create_daytime_nighttime_flags(timestamp_index, lat, lon, utc_offset):
    # Detect daytime and nighttime
    """Return daytime/nighttime flags (0/1 and boolean) from potential radiation for an index."""
    dnf = DaytimeNighttimeFlag(
        timestamp_index=timestamp_index,
        nighttime_threshold=20,
        lat=lat, lon=lon,
        utc_offset=utc_offset)
    flag_daytime = dnf.get_daytime_flag()
    flag_nighttime = dnf.get_nighttime_flag()  # 0/1 flag needed outside init

    is_daytime = flag_daytime == 1  # Convert 0/1 flag to False/True flag
    is_nighttime = flag_nighttime == 1  # Convert 0/1 flag to False/True flag
    return flag_daytime, flag_nighttime, is_daytime, is_nighttime
