"""
GUI.WIDGETS.DRIVER_QC: OPTIONAL QC-FLAG PICK FOR AN MDS DRIVER
==============================================================

The optional quality-flag column next to an MDS driver (SW_IN / TA / VPD), used
by the MDS gap-filling tab and the flux chain's Level 4.1. The combo lists
"(none)" first, then the flag columns whose name matches the chosen driver
(``dv.variables.driver_flag_columns``), then every other column.
Nothing is preselected: a user who never touches it runs MDS without driver QC,
exactly as before the option existed.

Presentation only; the QC rule itself is ``FluxMDS(swin_qc=, ta_qc=, vpd_qc=)``.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from diive.gui.widgets.column_picker import NONE_ITEM
from diive.variables.classification import driver_flag_columns

#: Tooltip shared by every driver QC combo (both tabs).
QC_TIP = (
    "Optional quality flag of this driver: 0 = measured, above 0 = gap-filled, "
    "e.g. FLAG_..._ISFILLED or FLUXNET *_F_QC. Use it with the gap-filled driver "
    "column.\n\n"
    "When set, the record being filled still uses its own driver value, even if "
    "that value was gap-filled, but only records with a measured driver value "
    "(flag 0, or no flag) count as similar conditions. This is how ONEFlux "
    "gap-fills NEE. It affects the SW_IN/TA/VPD look-up, not the mean diurnal "
    "cycle.\n\n"
    "(none): every record with a driver value counts (default).")


def fill_qc_combo(combo: QComboBox, cols: list[str], driver: str) -> None:
    """Refill ``combo``: "(none)", the driver's flag columns, then the rest.

    Keeps the current pick when it is still a column, else falls back to
    "(none)". Signals are blocked while refilling, so callers refresh any
    dependent display themselves.
    """
    cur = combo.currentText()
    suggested = driver_flag_columns(cols, driver)
    rest = [c for c in cols if c not in suggested]
    combo.blockSignals(True)
    combo.clear()
    combo.addItems([NONE_ITEM, *suggested, *rest])
    combo.setCurrentText(cur if cur in cols else NONE_ITEM)
    combo.blockSignals(False)


def qc_value(combo: QComboBox) -> str | None:
    """The picked flag column, or None for "(none)"."""
    text = combo.currentText()
    return None if text in ("", NONE_ITEM) else text


def set_qc_value(combo: QComboBox, value: str | None) -> None:
    """Select ``value``; None, a missing key or a column no longer present -> "(none)"."""
    i = combo.findText(value) if value else -1
    combo.setCurrentIndex(i if i >= 0 else max(combo.findText(NONE_ITEM), 0))
