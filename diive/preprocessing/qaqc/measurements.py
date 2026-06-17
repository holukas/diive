"""
QAQC.MEASUREMENTS: METEO MEASUREMENT GROUPS AND APPLICABLE CORRECTIONS
=====================================================================

A *measurement* is the database grouping used in the meteo-screening workflow
(e.g. ``TA`` holds all air-temperature variables, ``SW`` all shortwave-radiation
measurements). Knowing the measurement is what lets the workflow decide which
high-resolution corrections are physically meaningful — e.g. the radiation
zero-offset correction only makes sense for shortwave radiation / PPFD, and the
relative-humidity offset only for RH.

This is the authoritative place for "which measurement is this" and "which
corrections apply to it" — the GUI (and other callers) use it instead of
re-encoding the meteorological knowledge themselves.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from typing import NamedTuple


class Measurement(NamedTuple):
    """A meteo measurement group.

    Attributes:
        code: Short database measurement code, e.g. ``'TA'``.
        description: Human-readable description, e.g. ``'air temperature'``.
    """
    code: str
    description: str


#: Known meteo measurement groups, in a sensible display order.
MEASUREMENTS: tuple[Measurement, ...] = (
    Measurement('TA', 'air temperature'),
    Measurement('RH', 'relative humidity'),
    Measurement('VPD', 'vapor pressure deficit'),
    Measurement('SW', 'shortwave radiation'),
    Measurement('LW', 'longwave radiation'),
    Measurement('PPFD', 'photosynthetic photon flux density'),
    Measurement('PA', 'air pressure'),
    Measurement('PREC', 'precipitation'),
    Measurement('WS', 'wind speed'),
    Measurement('WD', 'wind direction'),
    Measurement('SWC', 'soil water content'),
    Measurement('TS', 'soil temperature'),
    Measurement('G', 'soil heat flux'),
)

#: Correction keys. These are stable identifiers used by the GUI and by
#: :func:`diive.preprocessing.corrections.codegen.corrections_to_code`; they are
#: NOT the library function names (those live in the codegen mapping).
CORR_RADIATION_ZERO_OFFSET = 'radiation_zero_offset'
CORR_RELATIVEHUMIDITY_OFFSET = 'relativehumidity_offset'
CORR_SETTO_MAX = 'setto_max'
CORR_SETTO_MIN = 'setto_min'
CORR_SETTO_VALUE = 'setto_value'
CORR_SET_EXACT_TO_MISSING = 'set_exact_to_missing'


class CorrectionSpec(NamedTuple):
    """Metadata for one high-resolution correction.

    Attributes:
        key: Stable correction identifier (one of the ``CORR_*`` constants).
        label: Human-readable name, e.g. ``'Remove radiation zero offset'``.
        description: One-line description of what the correction does.
        needs_coords: ``True`` if the correction needs site coordinates
            (latitude/longitude/UTC offset), e.g. for a day/night split.
    """
    key: str
    label: str
    description: str
    needs_coords: bool


#: All corrections, in the order they are offered.
CORRECTIONS: tuple[CorrectionSpec, ...] = (
    CorrectionSpec(
        CORR_RADIATION_ZERO_OFFSET, 'Remove radiation zero offset',
        'Radiation should read zero at night. The daily nighttime mean is '
        'subtracted from all records as the offset, then nighttime values are '
        'forced to zero. Use for SW_IN, SW_OUT, PPFD_IN, PPFD_OUT.',
        needs_coords=True),
    CorrectionSpec(
        CORR_RELATIVEHUMIDITY_OFFSET, 'Remove relative humidity offset',
        'Fixes relative humidity that drifts above 100%. The daily mean of the '
        'values exceeding 100% is removed as an offset and any remainder is '
        'capped at 100%. Use for RH.',
        needs_coords=False),
    CorrectionSpec(
        CORR_SETTO_MAX, 'Set to max threshold',
        'Caps the series at a known physical maximum: every value above the '
        'threshold is set to the threshold.',
        needs_coords=False),
    CorrectionSpec(
        CORR_SETTO_MIN, 'Set to min threshold',
        'Floors the series at a known physical minimum: every value below the '
        'threshold is set to the threshold.',
        needs_coords=False),
    CorrectionSpec(
        CORR_SETTO_VALUE, 'Set to value',
        'Overwrites every record inside one or more date ranges with a fixed '
        'value — e.g. to blank out a period of known instrument trouble.',
        needs_coords=False),
    CorrectionSpec(
        CORR_SET_EXACT_TO_MISSING, 'Set exact values to missing',
        'Sets records that exactly equal any of the listed values to missing '
        '(NaN) — e.g. to drop a stuck sentinel like 0 or -9999.',
        needs_coords=False),
)

#: Corrections that apply to any measurement.
GENERIC_CORRECTION_KEYS: tuple[str, ...] = (
    CORR_SETTO_MAX, CORR_SETTO_MIN, CORR_SETTO_VALUE, CORR_SET_EXACT_TO_MISSING,
)

#: Corrections tied to specific measurement codes (measurement-specific physics).
_MEASUREMENT_SPECIFIC: dict[str, frozenset[str]] = {
    CORR_RADIATION_ZERO_OFFSET: frozenset({'SW', 'PPFD'}),
    CORR_RELATIVEHUMIDITY_OFFSET: frozenset({'RH'}),
}

#: Variable-name prefixes mapped to a measurement code. First match wins; more
#: specific prefixes are listed first. Matching is case-sensitive, following the
#: column naming conventions.
_NAME_PREFIXES: tuple[tuple[str, str], ...] = (
    ('SWC', 'SWC'),     # soil water content (before SW)
    ('SW_', 'SW'),      # shortwave radiation
    ('SW', 'SW'),
    ('PPFD', 'PPFD'),
    ('LW', 'LW'),       # longwave radiation
    ('RH', 'RH'),
    ('VPD', 'VPD'),
    ('TA', 'TA'),       # air temperature ("TA"/"TA_*")
    ('Tair', 'TA'),
    ('TS', 'TS'),       # soil temperature
    ('PREC', 'PREC'),
    ('PA', 'PA'),       # air pressure
    ('WS', 'WS'),       # wind speed (before WD only by listing; distinct prefix)
    ('WD', 'WD'),       # wind direction
    ('G_', 'G'),        # soil heat flux
)

_BY_CODE: dict[str, Measurement] = {m.code: m for m in MEASUREMENTS}
_CORR_BY_KEY: dict[str, CorrectionSpec] = {c.key: c for c in CORRECTIONS}


def measurement_label(code: str) -> str:
    """Return a display label like ``'TA - air temperature'`` for a measurement
    code, or just the code if it is not a known measurement."""
    m = _BY_CODE.get(code)
    return f"{m.code} - {m.description}" if m is not None else code


def correction_spec(key: str) -> CorrectionSpec | None:
    """Return the :class:`CorrectionSpec` for a correction key, or ``None``."""
    return _CORR_BY_KEY.get(key)


def corrections_for_measurement(code: str | None) -> list[str]:
    """Return the correction keys that apply to a measurement, in display order.

    Measurement-specific corrections (e.g. radiation zero offset for ``SW`` /
    ``PPFD``) come first, followed by the generic corrections that apply to any
    measurement. An unknown or missing ``code`` yields the generic corrections
    only.

    Args:
        code: Measurement code, e.g. ``'SW'``, or ``None``.

    Returns:
        Ordered list of correction keys (``CORR_*`` values).
    """
    keys = [c.key for c in CORRECTIONS
            if code is not None and code in _MEASUREMENT_SPECIFIC.get(c.key, frozenset())]
    keys += [k for k in GENERIC_CORRECTION_KEYS]
    # Preserve the canonical CORRECTIONS order.
    order = {c.key: i for i, c in enumerate(CORRECTIONS)}
    return sorted(dict.fromkeys(keys), key=lambda k: order[k])


def detect_measurement(varname: str) -> str | None:
    """Guess the measurement code from a variable name by prefix.

    Args:
        varname: Variable / column name, e.g. ``'SW_IN_T1_2_1'``.

    Returns:
        A measurement code (e.g. ``'SW'``) or ``None`` if no prefix matches.

    Examples:
        >>> detect_measurement('SW_IN_T1_2_1')
        'SW'
        >>> detect_measurement('RH_T1_2_1')
        'RH'
        >>> detect_measurement('SWC_GF1_0.05_1')
        'SWC'
        >>> detect_measurement('FC') is None
        True
    """
    if not isinstance(varname, str):
        return None
    for prefix, code in _NAME_PREFIXES:
        if varname.startswith(prefix):
            return code
    return None
