"""
COMMON: SHARED OUTLIER DETECTION UTILITIES
===========================================

Shared functions for daytime/nighttime flag generation and other detection helpers.

Part of the diive library: https://github.com/holukas/diive
"""

import pandas as pd
from pandas import Series

from diive.core.utils.console import VERBOSE_PROGRESS, detail
from diive.variables import DaytimeNighttimeFlag

# Parameter names that changed when the day/night settings were unified across
# the outlier detectors. Old name -> new name.
_RENAMED_DAYNIGHT_PARAMS = {
    'separate_daytime_nighttime': 'separate_day_night',
    # AbsoluteLimits: one [min, max] pair per period became two overrides each.
    'daytime_minmax': 'minval_daytime / maxval_daytime',
    'nighttime_minmax': 'minval_nighttime / maxval_nighttime',
    # Hampel carried two names for each of its two thresholds.
    'n_sigma_dt': 'n_sigma_daytime',
    'n_sigma_nt': 'n_sigma_nighttime',
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


def window_to_records(window: int | str | None, series: Series, name: str = 'window',
                      verbose: bool = False) -> int | None:
    """Return a rolling window as a record count.

    An int (or None) is returned unchanged. A pandas time span such as ``'7D'``,
    ``'12h'`` or ``'30min'`` becomes the number of records it covers at the
    regular frequency of ``series.index``, so the same setting means the same
    duration at any data resolution.

    Raises:
        ValueError: If the index has no fixed frequency (irregular timestamps, or
            a calendar frequency such as month start), if ``window`` is not a
            fixed duration, or if the span is not a whole multiple of the
            frequency.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series(0.0, index=pd.date_range('2024-01-01', periods=2000, freq='10min'))
        >>> window_to_records('7D', s)
        1008
        >>> window_to_records(48, s)
        48
    """
    if not isinstance(window, str):
        return window

    freq = getattr(series.index, 'freq', None)
    step = None
    if freq is not None:
        # Calendar offsets (month, year, business day, week) have no fixed
        # duration and raise on `.nanos`.
        try:
            step = pd.Timedelta(freq.nanos, unit='ns')
        except ValueError:
            step = None
    if step is None:
        raise ValueError(
            f"{name}={window!r} is a time span, which needs a time index with a fixed "
            f"frequency to be converted into records, but the index has "
            f"{'the calendar frequency ' + repr(freq.freqstr) if freq is not None else 'no regular frequency'}. "
            f"Make the index regular (e.g. resample or asfreq) or pass {name} as a record count.")

    try:
        span = pd.Timedelta(window)
    except ValueError:
        raise ValueError(
            f"{name}={window!r} is not a fixed duration. Pass a time span such as "
            f"'7D', '12h' or '30min', or a record count.") from None

    n_records, remainder = divmod(span, step)
    if n_records < 1:
        raise ValueError(f"{name}={window!r} must span at least one record of {freq.freqstr!r} data.")
    if remainder != pd.Timedelta(0):
        raise ValueError(
            f"{name}={window!r} is not a whole multiple of the data frequency "
            f"{freq.freqstr!r}. Pass a span that is, or a record count.")

    detail(f"{name}={window!r} = {n_records} records at {freq.freqstr!r}",
           verbose=verbose, min_level=VERBOSE_PROGRESS)
    return int(n_records)


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
