"""
GUI.TABS.CORRECTIONS_RELATIVEHUMIDITY_OFFSET: REMOVE RH OFFSET TAB
=================================================================

Fix relative humidity that drifts above 100%: the daily mean of the values
exceeding 100% is removed as an offset and any remainder is capped at 100%
(`dv.corrections.remove_relativehumidity_offset`).

All the preview/threading/plotting machinery lives in :class:`BaseCorrectionTab`;
this tab only declares the correction key and the hero chip. Intended for RH, but
available for any variable (suggestion, not a lock).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.gui.tabs._correction_base import BaseCorrectionTab
from diive.preprocessing.qaqc.measurements import CORR_RELATIVEHUMIDITY_OFFSET


class RelativeHumidityOffsetTab(BaseCorrectionTab):
    """Remove the >100% offset from relative humidity data."""

    title = "Remove relative humidity offset"
    intro = ("Fixes relative humidity that drifts above 100%. The daily mean of "
             "the values exceeding 100% is removed as an offset; any remainder is "
             "capped at 100%.")
    method_suffix = "RHOFFSET"
    corr_key = CORR_RELATIVEHUMIDITY_OFFSET
    method_chip_label = "RH OFFSET"
    method_chip_bg = "#E1F5FE"
    method_chip_fg = "#0277BD"
    needs_coords = False
    suited_for = "Suited for RH (relative humidity)."
