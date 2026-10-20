"""
GUI.WIDGETS.MDS_MODE: MDS MODE CHOICE (DIIVE / ONEFLUX / REDDYPROC)
===================================================================

The MDS mode combo shared by the MDS gap-filling tab and the flux chain's
Level 4.1. Each item shows a label and carries the library value as item data
(``FluxMDS(mode=)`` / ``run_level41_mds(mode=)``), so projects store the value,
not the label.

Presentation only; the fill rules live in ``diive.gapfilling.mds``.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

#: (label, FluxMDS mode), default first.
MODE_ITEMS = (("diive (Vekuri 2023 mean)", "diive"),
              ("ONEFlux (bit for bit)", "oneflux"),
              ("REddyProc", "reddyproc"))

#: Tooltip shared by both tabs.
MODE_TIP = (
    "How a gap is filled from the similar records.\n\n"
    "diive (Vekuri 2023 mean), default: in the look-ups that match on SW_IN, a "
    "daytime gap (SW_IN >= 20 W m-2) gets the mean of two means, one of the "
    "similar records with less SW_IN and one of those with more. This removes a "
    "bias when the similar records are spread unevenly around the gap's SW_IN "
    "(Vekuri et al. 2023). Otherwise the same rules as ONEFlux.\n\n"
    "ONEFlux (bit for bit): the plain mean of the similar records, the same "
    "values as ONEFlux's gf_mds tool.\n\n"
    "REddyProc: the plain mean with REddyProc's windows, as its sMDSGapFill "
    "does. Gaps longer than 60 days stay unfilled.")


def make_mode_combo() -> QComboBox:
    """A combo with the MDS modes, 'diive' selected."""
    combo = QComboBox()
    for label, mode in MODE_ITEMS:
        combo.addItem(label, mode)
    combo.setToolTip(MODE_TIP)
    return combo


def mode_value(combo: QComboBox) -> str:
    """The selected mode ('diive', 'oneflux' or 'reddyproc')."""
    return combo.currentData()


def set_mode_value(combo: QComboBox, mode: str | None) -> None:
    """Select ``mode``; an unknown value or None leaves the combo unchanged."""
    i = combo.findData(mode) if mode else -1
    if i >= 0:
        combo.setCurrentIndex(i)
