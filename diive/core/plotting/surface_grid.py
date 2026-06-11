"""
SURFACE_GRID: DATE x TIME-OF-DAY VALUE GRID FOR 3-D SURFACES
===========================================================

Headless gridding of a time series into a rectangular *date x time-of-day*
matrix of values — the numeric data behind a 3-D relief surface (the natural
3-D analogue of :class:`~diive.core.plotting.heatmap_datetime.HeatmapDateTime`).

This is pure domain logic: sanitize the timestamp, pivot to a complete grid,
and return numeric arrays. It contains no plotting backend (matplotlib, VTK,
...) so any caller — the desktop GUI's PyVista surface, a notebook, a script —
can render it however it likes.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import Series

from diive.core.times.times import TimestampSanitizer


@dataclass
class DateTimeSurface:
    """Numeric grid of a time series as date x time-of-day.

    Attributes:
        x_hours: 1-D float array of time-of-day in hours since midnight
            (e.g. 0.5 for 00:30), one entry per time-of-day column. Length T.
        y_days: 1-D float array of days since the first date, one entry per
            date row. Length D. (Numeric so a 3-D backend can use it as a real
            axis; map back to calendar dates with :attr:`dates`.)
        z: 2-D value array of shape (D, T); NaN where a timestamp is missing.
        dates: 1-D array of ``datetime.date`` for the D rows (for axis ticks).
        name: Name of the source series.
    """

    x_hours: np.ndarray
    y_days: np.ndarray
    z: np.ndarray
    dates: np.ndarray
    name: str


def datetime_surface_grid(series: Series) -> DateTimeSurface:
    """Pivot a time series into a date x time-of-day value grid.

    The index is sanitized (sorted, de-duplicated, regularised to a complete
    frequency grid) so the pivot forms a full rectangle — every date row has
    the same set of time-of-day columns, with NaN for missing records. This is
    the same preparation the 2-D heatmap uses, exposed as plain arrays for 3-D
    rendering.

    Args:
        series: Pandas Series with a diive-convention datetime index
            (``TIMESTAMP_START`` / ``_MIDDLE`` / ``_END``). Any sub-daily
            resolution is supported.

    Returns:
        A :class:`DateTimeSurface` with the numeric grid and date labels.
    """
    series = series.copy()
    series.name = series.name if series.name else "data"
    series = TimestampSanitizer(data=series, output_middle_timestamp=False).get()

    df = pd.DataFrame(series)
    df["DATE"] = df.index.date
    df["TIME"] = df.index.time
    df = df.reset_index(drop=True)
    grid = df.pivot(index="DATE", columns="TIME", values=series.name)

    times = grid.columns.to_numpy()
    x_hours = np.array(
        [t.hour + t.minute / 60 + t.second / 3600 for t in times], dtype=float)

    dates = grid.index.to_numpy()
    first = grid.index[0]
    y_days = np.array([(d - first).days for d in grid.index], dtype=float)

    z = grid.to_numpy(dtype=float)
    return DateTimeSurface(x_hours=x_hours, y_days=y_days, z=z,
                           dates=dates, name=str(series.name))
