"""
VARIABLES.CLASSIFICATION: VARIABLE CATEGORY FROM NAME
=====================================================

Classify a variable into its kind and physical category from its name, using
the column naming conventions of the Swiss FluxNet / FLUXNET workflow. This is
the authoritative place for "what kind of variable is this column" — callers
(e.g. the GUI) use it instead of re-encoding name prefixes themselves.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from typing import NamedTuple

#: Physical categories.
CATEGORY_CARBON = "carbon"
CATEGORY_WATER = "water"
CATEGORY_RADIATION = "radiation"
CATEGORY_METEO = "meteo"
CATEGORY_SOIL = "soil"
CATEGORY_NITROGEN = "nitrogen"


class VariableClass(NamedTuple):
    """Result of :func:`classify_variable`.

    Attributes:
        kind: Canonical short variable label, e.g. ``'NEE'``, ``'GPP'``,
            ``'Reco'``, ``'FCH4'``, ``'FN2O'``, ``'FH2O'``, ``'LE'``, ``'ET'``,
            ``'Rg'``, ``'SW_IN'``, ``'PPFD'``, ``'PAR'``, ``'LW'``.
        category: One of ``'carbon'``, ``'water'``, ``'radiation'``,
            ``'meteo'``, ``'soil'``, ``'nitrogen'``.
    """
    kind: str
    category: str


# Name prefix -> (kind, category). First match wins; list more specific
# prefixes first. Matching is case-sensitive, following column conventions.
_RULES: tuple[tuple[str, str, str], ...] = (
    ("NEE", "NEE", CATEGORY_CARBON),
    ("GPP", "GPP", CATEGORY_CARBON),
    ("Reco", "Reco", CATEGORY_CARBON),
    ("FCH4", "FCH4", CATEGORY_CARBON),       # methane flux ("FCH4" and "FCH4_*")
    ("FN2O", "FN2O", CATEGORY_NITROGEN),     # nitrous oxide flux ("FN2O"/"FN2O_*")
    ("FH2O", "FH2O", CATEGORY_WATER),        # water vapour flux ("FH2O"/"FH2O_*")
    ("LE_", "LE", CATEGORY_WATER),
    ("ET_", "ET", CATEGORY_WATER),
    ("Rg_", "Rg", CATEGORY_RADIATION),
    ("SW_IN_", "SW_IN", CATEGORY_RADIATION),
    ("PPFD", "PPFD", CATEGORY_RADIATION),  # bare "PPFD" and "PPFD_*"
    ("PAR_", "PAR", CATEGORY_RADIATION),
    ("LW_", "LW", CATEGORY_RADIATION),
    ("Tair", "TA", CATEGORY_METEO),        # "Tair" and "Tair_*"
    ("TA_", "TA", CATEGORY_METEO),         # "TA_*" (bare "TA" handled below)
    ("VPD", "VPD", CATEGORY_METEO),        # "VPD" and "VPD_*"
    ("SWC", "SWC", CATEGORY_SOIL),         # "SWC" and "SWC_*"
)


def classify_variable(name: str) -> VariableClass | None:
    """Classify a variable from its column name.

    Args:
        name: Variable / column name, e.g. ``'GPP_CUT_REF_f'``.

    Returns:
        A :class:`VariableClass` (``kind``, ``category``) for recognised
        variables, or ``None`` if the name matches no known prefix.

    Examples:
        >>> classify_variable('NEE_CUT_REF_f')
        VariableClass(kind='NEE', category='carbon')
        >>> classify_variable('LE_f')
        VariableClass(kind='LE', category='water')
        >>> classify_variable('TA_f') is None
        True
    """
    if not isinstance(name, str):
        return None
    # FC is the CO2 flux (the pre-NEE stage). Match it on a word boundary so it
    # does not also catch FCH4 (methane flux).
    if name == "FC" or name.startswith("FC_"):
        return VariableClass(kind="FC", category=CATEGORY_CARBON)
    # Bare "TA" is exact-matched (a "TA" prefix would also catch e.g. TARGET/TAU);
    # "TA_*" and "Tair*" are handled by the prefix rules below.
    if name == "TA":
        return VariableClass(kind="TA", category=CATEGORY_METEO)
    for prefix, kind, category in _RULES:
        if name.startswith(prefix):
            return VariableClass(kind=kind, category=category)
    return None
