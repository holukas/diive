"""
GUI.WIDGETS.COLORMAPS: SHARED COLORMAP CHOICES
==============================================

One curated colormap list behind every colormap dropdown in the GUI, so the
plotting settings panel, the Appearance tab and the tabs that preview a heatmap
all offer the same choices.

The app-wide default colormap for the preview heatmaps lives in
``theme.manager.heatmap_cmap`` (edited in the Appearance tab).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

#: Curated colormaps offered in a colormap dropdown. The combos stay editable,
#: so any valid matplotlib name can also be typed. Diverging first (the diive
#: default), then perceptually-uniform sequential, then a few classics.
COLORMAPS = [
    "RdYlBu_r", "RdYlBu", "RdBu_r", "coolwarm", "Spectral", "Spectral_r",
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "YlOrRd", "YlGnBu", "Greys", "jet",
]


def colormap_combo(current: str | None = None,
                   first: list[str] | None = None) -> QComboBox:
    """An editable colormap dropdown over :data:`COLORMAPS`.

    Args:
        current: Colormap shown initially; any matplotlib name, also one that is
            not in the curated list.
        first: Entries prepended to the list (e.g. ``["auto"]``).
    """
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItems(list(first or []) + COLORMAPS)
    if current:
        combo.setCurrentText(current)
    return combo
