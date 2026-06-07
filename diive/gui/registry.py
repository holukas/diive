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
from diive.gui.tabs.features import FeatureEngineerTab
from diive.gui.tabs.fluxchain import FluxChainTab
from diive.gui.tabs.log import LogTab
from diive.gui.tabs.overview import OverviewTab
from diive.gui.tabs.plotting import (
    HEATMAP,
    HEATMAP_YEARMONTH,
    RIDGELINE,
    TIMESERIES,
    PlottingTab,
)
from diive.gui.tabs.settings import SettingsTab

#: Tab classes always shown in the main window, in display order.
#: Future: append FluxChainTab, OutlierTab, GapFillingTab, ...
TAB_CLASSES: list[type[DiiveTab]] = [
    OverviewTab,
    LogTab,
]

#: Tabs opened on demand from a menu (not shown until selected, closable),
#: grouped by the top-level menu they appear under: {menu: {label: factory}}.
#: Each plot method is its own tab; add a new method by adding a PlottingTab
#: factory here (and a branch in plotting._draw_one).
MENU_TABS: dict[str, dict[str, callable]] = {
    "Plot": {
        "Heatmap date/time": lambda: PlottingTab(HEATMAP, "Heatmap date/time"),
        "Heatmap year/month": lambda: PlottingTab(HEATMAP_YEARMONTH, "Heatmap year/month"),
        "Time series": lambda: PlottingTab(TIMESERIES, "Time series"),
        "Ridgeline": lambda: PlottingTab(RIDGELINE, "Ridgeline"),
    },
    "Tools": {
        "Feature engineering": FeatureEngineerTab,
        "Flux processing chain": FluxChainTab,
    },
    "Settings": {
        "Appearance": SettingsTab,
    },
}

#: Flat label -> factory lookup (used to open a tab by its menu label).
MENU_TAB_CLASSES: dict[str, callable] = {
    label: factory for group in MENU_TABS.values() for label, factory in group.items()
}

#: Menu tabs that may exist only once (re-selecting focuses the existing one).
#: Everything else opens a new, numbered instance each time.
SINGLE_INSTANCE_TABS: set[str] = {"Appearance", "Flux processing chain"}
