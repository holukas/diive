"""
GUI.WIDGETS.PARTITIONING_MODE: NEE PARTITIONING METHOD AND MODE CHOICE
======================================================================

The two combos of the NEE partitioning tab. Method picks the algorithm
(nighttime or daytime), mode picks the reference implementation (ONEFlux or
REddyProc). Each item shows a label and carries the library value as item data
(``PartitioningNEE(method=, mode=)``), so projects store the value, not the
label.

Presentation only; the partitioning lives in ``diive.flux.partitioning``.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

#: (label, PartitioningNEE method), default first.
METHOD_ITEMS = (("Nighttime", "nighttime"),
                ("Daytime", "daytime"))

#: (label, PartitioningNEE mode), default first.
MODE_ITEMS = (("ONEFlux (ICOS, default)", "oneflux"),
              ("REddyProc", "reddyproc"))

METHOD_TIP = (
    "Which algorithm splits NEE into GPP and RECO.\n\n"
    "Nighttime (Reichstein et al. 2005): fits the temperature response of "
    "nighttime NEE, when there is no photosynthesis, extends it to the daytime "
    "as RECO and takes GPP as the rest.\n\n"
    "Daytime (Lasslop et al. 2010): fits a light-response curve to daytime NEE "
    "and gets GPP and RECO from the same fit. Also uses VPD.")

MODE_TIP = (
    "Which reference implementation the result reproduces.\n\n"
    "ONEFlux (ICOS, default): the processing used by ICOS and FLUXNET2015. "
    "Columns end in _NT_OF or _DT_OF.\n\n"
    "REddyProc: the R package REddyProc. Columns end in _NT_RP or _DT_RP.\n\n"
    "Each mode writes its own columns, so results of both modes can sit side "
    "by side in the dataset.")


def _make_combo(items, tip: str) -> QComboBox:
    combo = QComboBox()
    for label, value in items:
        combo.addItem(label, value)
    combo.setToolTip(tip)
    return combo


def make_method_combo() -> QComboBox:
    """A combo with the partitioning methods, 'nighttime' selected."""
    return _make_combo(METHOD_ITEMS, METHOD_TIP)


def make_mode_combo() -> QComboBox:
    """A combo with the partitioning modes, 'oneflux' selected."""
    return _make_combo(MODE_ITEMS, MODE_TIP)


def combo_value(combo: QComboBox) -> str:
    """The selected library value ('nighttime', 'daytime', 'oneflux', ...)."""
    return combo.currentData()


def set_combo_value(combo: QComboBox, value: str | None) -> None:
    """Select ``value``; an unknown value or None leaves the combo unchanged."""
    i = combo.findData(value) if value else -1
    if i >= 0:
        combo.setCurrentIndex(i)
