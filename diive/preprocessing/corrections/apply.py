"""
CORRECTIONS.APPLY: APPLY A CHAIN OF HIGH-RESOLUTION CORRECTIONS
===============================================================

Apply an ordered list of corrections to a series, in one call. This is the
single place that maps a correction *key* (the stable identifier used by the
GUI and the meteo-screening workflow, see
:mod:`diive.preprocessing.qaqc.measurements`) to the actual correction function
and its arguments — so callers (e.g. the GUI's stepwise-screening tab) never
re-encode that mapping, and :func:`corrections_to_code` can mirror it exactly.

A *correction* is a ``{"key": str, "kwargs": dict}`` dict. Corrections are
applied in order, each operating on the output of the previous one.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from pandas import Series

from diive.preprocessing.corrections.offsetcorrection import (
    remove_nighttime_zero_offset,
    remove_relativehumidity_offset,
)
from diive.preprocessing.corrections.setto import (
    set_exact_values_to_missing,
    setto_threshold,
    setto_value,
)


def apply_corrections(series: Series,
                      corrections: list[dict],
                      *,
                      lat: float | None = None,
                      lon: float | None = None,
                      utc_offset: int | None = None,
                      showplot: bool = False) -> Series:
    """Apply an ordered list of corrections to a series.

    Args:
        series: Series to correct.
        corrections: Ordered list of ``{"key": str, "kwargs": dict}`` dicts.
            ``key`` is one of the ``CORR_*`` constants in
            :mod:`diive.preprocessing.qaqc.measurements`; ``kwargs`` are the
            correction-specific parameters (e.g. ``{"threshold": 30}`` for the
            set-to-max correction).
        lat: Site latitude (required for the radiation zero-offset correction).
        lon: Site longitude (required for the radiation zero-offset correction).
        utc_offset: UTC offset of the timestamp index (required for the radiation
            zero-offset correction).
        showplot: Show a plot for each correction.

    Returns:
        The corrected series (named like the input).
    """
    # Lazy import avoids an import cycle: qaqc depends on corrections, so this
    # lower-level module must not import qaqc at load time.
    from diive.preprocessing.qaqc.measurements import (
        CORR_RADIATION_ZERO_OFFSET,
        CORR_RELATIVEHUMIDITY_OFFSET,
        CORR_SETTO_MAX,
        CORR_SETTO_MIN,
        CORR_SETTO_VALUE,
        CORR_SET_EXACT_TO_MISSING,
    )

    out = series.copy()
    for corr in corrections:
        key = corr["key"]
        kwargs = corr.get("kwargs", {})
        if key == CORR_RADIATION_ZERO_OFFSET:
            out = remove_nighttime_zero_offset(
                series=out, lat=lat, lon=lon, utc_offset=utc_offset,
                clamp_negatives=kwargs.get("clamp_negatives", True),
                showplot=showplot)
        elif key == CORR_RELATIVEHUMIDITY_OFFSET:
            out = remove_relativehumidity_offset(series=out, showplot=showplot)
        elif key == CORR_SETTO_MAX:
            out = setto_threshold(series=out, threshold=kwargs["threshold"],
                                  type='max', showplot=showplot)
        elif key == CORR_SETTO_MIN:
            out = setto_threshold(series=out, threshold=kwargs["threshold"],
                                  type='min', showplot=showplot)
        elif key == CORR_SETTO_VALUE:
            out = setto_value(series=out, dates=kwargs["dates"],
                              value=kwargs.get("value", 0))
        elif key == CORR_SET_EXACT_TO_MISSING:
            out = set_exact_values_to_missing(series=out, values=kwargs["values"],
                                              showplot=showplot)
        else:
            raise ValueError(f"Unknown correction key: {key!r}")
    return out
