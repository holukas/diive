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
from diive.gui.tabs.drivers import DriverExplorerTab
from diive.gui.tabs.events import EventsTab
from diive.gui.tabs.features import FeatureEngineerTab
from diive.gui.tabs.fluxchain import FluxChainTab
from diive.gui.tabs.gaps import GapDashboardTab
from diive.gui.tabs.log import LogTab
from diive.gui.tabs.metadata_explorer import MetadataExplorerTab
from diive.gui.tabs.outliers import HampelOutlierTab
from diive.gui.tabs.outliers_absolutelimits import AbsoluteLimitsTab
from diive.gui.tabs.outliers_localsd import LocalSDOutlierTab
from diive.gui.tabs.outliers_lof import LocalOutlierFactorTab
from diive.gui.tabs.outliers_manualremoval import ManualRemovalOutlierTab
from diive.gui.tabs.outliers_zscore import ZScoreOutlierTab
from diive.gui.tabs.outliers_zscoreincrements import ZScoreIncrementsOutlierTab
from diive.gui.tabs.outliers_trim import TrimLowOutlierTab
from diive.gui.tabs.outliers_zscorerolling import ZScoreRollingOutlierTab
from diive.gui.tabs.profile import ProfileTab
from diive.gui.tabs.seasonaltrend import SeasonalTrendTab
from diive.gui.tabs.spectrogram import SpectrogramTab
from diive.gui.tabs.stepwise import StepwiseScreeningTab
from diive.gui.tabs.timelag import TimeLagAnalysisTab
from diive.gui.tabs.overview import OverviewTab
from diive.gui.tabs.rename_variables import RenameVariablesTab
from diive.gui.tabs.plotting import (
    CUMULATIVE_YEAR,
    DIELCYCLE,
    HEATMAP,
    HEATMAP_YEARMONTH,
    HEXBIN,
    HISTOGRAM,
    RIDGELINE,
    SCATTER,
    TIMESERIES,
    PlottingTab,
)
from diive.gui.tabs.settings import SettingsTab
from diive.gui.tabs.site import ProjectSettingsTab
from diive.gui.tabs.surface3d import Surface3DTab
from diive.gui.tabs.variable_selector import VariableSelectorTab

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
    "Data": {
        # Merged into the manually-built Data menu (date-range actions); see
        # MainWindow._build_menus. Not given its own top-level menu. These are
        # data/variable preparation: subset, per-variable metadata, and the
        # feature engineer (which derives new columns).
        "Select variables": VariableSelectorTab,
        "Rename variables": RenameVariablesTab,
        "Metadata explorer": MetadataExplorerTab,
        "Feature engineering": FeatureEngineerTab,
        "Events": EventsTab,
    },
    "Plot": {
        "Heatmap date/time": lambda: PlottingTab(HEATMAP, "Heatmap date/time"),
        "Heatmap year/month": lambda: PlottingTab(HEATMAP_YEARMONTH, "Heatmap year/month"),
        "Time series": lambda: PlottingTab(TIMESERIES, "Time series"),
        "Diel cycle": lambda: PlottingTab(DIELCYCLE, "Diel cycle"),
        "Cumulative year": lambda: PlottingTab(CUMULATIVE_YEAR, "Cumulative year"),
        "Ridgeline": lambda: PlottingTab(RIDGELINE, "Ridgeline"),
        "Scatter XY": lambda: PlottingTab(SCATTER, "Scatter XY"),
        "Hexbin": lambda: PlottingTab(HEXBIN, "Hexbin"),
        "Histogram": lambda: PlottingTab(HISTOGRAM, "Histogram"),
        "3D surface": Surface3DTab,
    },
    "Outliers": {
        "Absolute limits filter": AbsoluteLimitsTab,
        "Hampel filter": HampelOutlierTab,
        "Local SD filter": LocalSDOutlierTab,
        "Z-score filter": ZScoreOutlierTab,
        "Z-score (rolling) filter": ZScoreRollingOutlierTab,
        "Z-score (increments) filter": ZScoreIncrementsOutlierTab,
        "Local Outlier Factor filter": LocalOutlierFactorTab,
        "Trim-low filter": TrimLowOutlierTab,
        "Manual removal": ManualRemovalOutlierTab,
        "Stepwise screening": StepwiseScreeningTab,
    },
    # Eddy-covariance flux processing (dv.flux). Its own menu — a first-class
    # diive domain that will grow (gap-filling, USTAR, storage, ...).
    "Flux": {
        "Flux processing chain": FluxChainTab,
        "Time lag analysis": TimeLagAnalysisTab,
    },
    # Exploratory analysis & diagnostics (dv.analysis).
    "Analyze": {
        "Data profile": ProfileTab,
        "Gaps & coverage": GapDashboardTab,
        "Driver explorer": DriverExplorerTab,
        "Seasonal-trend & anomalies": SeasonalTrendTab,
        "Spectrogram": SpectrogramTab,
    },
    "Settings": {
        "Appearance": SettingsTab,
        "Project settings": ProjectSettingsTab,
    },
}

#: Flat label -> factory lookup (used to open a tab by its menu label).
MENU_TAB_CLASSES: dict[str, callable] = {
    label: factory for group in MENU_TABS.values() for label, factory in group.items()
}

#: Menu tabs that may exist only once (re-selecting focuses the existing one).
#: Everything else opens a new, numbered instance each time.
SINGLE_INSTANCE_TABS: set[str] = {
    "Appearance", "Project settings", "Flux processing chain", "Time lag analysis",
    "Data profile", "Gaps & coverage",
    "Driver explorer", "Seasonal-trend & anomalies", "Spectrogram",
    "Metadata explorer", "Select variables", "Rename variables", "3D surface",
    "Stepwise screening", "Events"}
