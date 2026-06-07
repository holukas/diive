"""
GUI.REGISTRY: TAB REGISTRY
==========================

Single source of truth for which tabs the main window shows, in order. To add
a feature area (e.g. the flux processing chain), implement a `DiiveTab`
subclass and append its class here -- nothing else changes.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.log import LogTab
from diive.gui.tabs.plotting import PlottingTab

#: Tab classes always shown in the main window, in display order.
#: Future: append FluxChainTab, OutlierTab, GapFillingTab, ...
TAB_CLASSES: list[type[DiiveTab]] = [
    PlottingTab,
    LogTab,
]

#: Tabs opened on demand from a menu (not shown until selected, closable).
#: Maps menu label -> tab class.
from diive.gui.tabs.features import FeatureEngineerTab  # noqa: E402
from diive.gui.tabs.settings import SettingsTab  # noqa: E402

MENU_TAB_CLASSES: dict[str, type[DiiveTab]] = {
    "Feature engineering": FeatureEngineerTab,
    "Appearance settings": SettingsTab,
}
