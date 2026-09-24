"""
GUI.REGISTRY: TAB REGISTRY
==========================

Single source of truth for which tabs the main window shows, in order. To add
a feature area (e.g. the flux processing chain), implement a `DiiveTab`
subclass and register it here -- nothing else changes.

Menu tabs are registered by module path, not imported: each entry is a
`LazyTab` that imports its module on first open. Importing every tab module up
front pulled in xgboost, scikit-learn, statsmodels and the flux chain before the
window could appear. The always-on tabs (`TAB_CLASSES`) stay eager because the
window builds them straight away.

PyInstaller cannot follow these string imports. `packaging/diive_gui.spec`
bundles every module under `diive.gui.tabs` for that reason, so a menu tab must
live in that package.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from importlib import import_module

from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.log import LogTab
from diive.gui.tabs.overview import OverviewTab
from diive.gui.widgets.plot_settings import (
    CUMULATIVE,
    CUMULATIVE_YEAR,
    DIELCYCLE,
    HEATMAP,
    HEATMAP_XYZ,
    HEATMAP_YEARMONTH,
    HEXBIN,
    HISTOGRAM,
    RIDGELINE,
    SCATTER,
    SHIFTEDDIST,
    TIMESERIES,
    TREERING,
    WATERFALL,
    WINDROSE,
)

#: Package every lazily registered tab module must live in (see the module
#: docstring: the PyInstaller spec bundles this package's modules).
TABS_PACKAGE = "diive.gui.tabs"


class LazyTab:
    """Menu-tab factory that imports its tab module on first call.

    ``LazyTab("diive.gui.tabs.gaps:GapDashboardTab")()`` imports the module and
    returns a new tab instance, exactly like calling the class did. Extra
    positional ``args`` are passed to the constructor (the plot tabs take their
    plot type and title).
    """

    __slots__ = ("path", "args")

    def __init__(self, path: str, *args) -> None:
        self.path = path
        self.args = args

    @property
    def module(self) -> str:
        """Dotted name of the module that defines the tab."""
        return self.path.partition(":")[0]

    def load(self) -> type[DiiveTab]:
        """Import the module (first time only) and return the tab class."""
        module, _, name = self.path.partition(":")
        return getattr(import_module(module), name)

    def __call__(self) -> DiiveTab:
        return self.load()(*self.args)

    def __repr__(self) -> str:
        args = "".join(f", {a!r}" for a in self.args)
        return f"LazyTab({self.path!r}{args})"


def _tab(module: str, name: str) -> LazyTab:
    return LazyTab(f"{TABS_PACKAGE}.{module}:{name}")


def _plot_tab(plot_type: str, title: str) -> LazyTab:
    return LazyTab(f"{TABS_PACKAGE}.plotting:PlottingTab", plot_type, title)


#: Tab classes always shown in the main window, in display order.
#: Future: append FluxChainTab, OutlierTab, GapFillingTab, ...
TAB_CLASSES: list[type[DiiveTab]] = [
    OverviewTab,
    LogTab,
]

#: Tabs opened on demand from a menu (not shown until selected, closable),
#: grouped by the top-level menu they appear under: {menu: {label: factory}}.
#: Each factory is a `LazyTab`; calling it imports the module and returns a new
#: tab. Each plot method is its own tab; add a new method by adding a
#: `_plot_tab` entry here (and a branch in plotting._draw_one).
MENU_TABS: dict[str, dict[str, LazyTab]] = {
    # Scope the working dataset (non-destructive subselection). Merged into the
    # manually-built Data menu next to the date-range actions; see
    # MainWindow._build_menus. Not given its own top-level menu.
    "Data": {
        "Select variables": _tab("variable_selector", "VariableSelectorTab"),
        "Select records by condition": _tab("select_records", "SelectRecordsTab"),
    },
    # Create & manage individual variables (columns). Built manually in
    # MainWindow so the "Add timestamp column..." action can sit between the
    # create tabs and a separator splits create from manage.
    "Variables": {
        "Feature engineering": _tab("features", "FeatureEngineerTab"),
        "Combine variables": _tab("combine_variables", "CombineVariablesTab"),
        "Rename variables": _tab("rename_variables", "RenameVariablesTab"),
        "Metadata explorer": _tab("metadata_explorer", "MetadataExplorerTab"),
        # Derived-variable calculators, shown under a "Calculate" section at the
        # end of the Variables menu (see MainWindow._build_menus).
        "VPD (TA + RH)": _tab("derived_vpd", "VpdFromTaRhTab"),
        "Potential radiation": _tab("derived_potrad", "PotradTab"),
    },
    # Time-stamped event markers (annotations layered over the data, not column
    # operations). Folded into the Data menu (under an "Events" section) in
    # MainWindow, alongside the "Add event..." / "Show events on plots" actions.
    "Events": {
        "Events": _tab("events", "EventsTab"),
    },
    "Plot": {
        "Heatmap date/time": _plot_tab(HEATMAP, "Heatmap date/time"),
        "Heatmap year/month": _plot_tab(HEATMAP_YEARMONTH, "Heatmap year/month"),
        "Heatmap x/y/z": _plot_tab(HEATMAP_XYZ, "Heatmap x/y/z"),
        "Time series": _plot_tab(TIMESERIES, "Time series"),
        "Diel cycle": _plot_tab(DIELCYCLE, "Diel cycle"),
        "Cumulative year": _plot_tab(CUMULATIVE_YEAR, "Cumulative year"),
        "Cumulative": _plot_tab(CUMULATIVE, "Cumulative"),
        "Ridgeline": _plot_tab(RIDGELINE, "Ridgeline"),
        "Scatter XY": _plot_tab(SCATTER, "Scatter XY"),
        "Hexbin": _plot_tab(HEXBIN, "Hexbin"),
        "Histogram": _plot_tab(HISTOGRAM, "Histogram"),
        "Shifted distribution": _plot_tab(SHIFTEDDIST, "Shifted distribution"),
        "Wind rose": _plot_tab(WINDROSE, "Wind rose"),
        "Tree ring": _plot_tab(TREERING, "Tree ring"),
        "Waterfall": _plot_tab(WATERFALL, "Waterfall"),
        "3D surface": _tab("surface3d", "Surface3DTab"),
        "3D surface (X/Y/Z)": _tab("surfacexyz", "SurfaceXYZTab"),
    },
    # Outlier detection. Combined with Corrections into one top-level "Cleaning"
    # menu in MainWindow._build_menus (the two per-variable cleaning families).
    "Outliers": {
        "Stepwise screening": _tab("stepwise", "StepwiseScreeningTab"),
        "Absolute limits filter": _tab("outliers_absolutelimits", "AbsoluteLimitsTab"),
        "Hampel filter": _tab("outliers", "HampelOutlierTab"),
        "Local SD filter": _tab("outliers_localsd", "LocalSDOutlierTab"),
        "Z-score filter": _tab("outliers_zscore", "ZScoreOutlierTab"),
        "Z-score (rolling) filter": _tab("outliers_zscorerolling", "ZScoreRollingOutlierTab"),
        "Z-score (increments) filter": _tab("outliers_zscoreincrements", "ZScoreIncrementsOutlierTab"),
        "Local Outlier Factor filter": _tab("outliers_lof", "LocalOutlierFactorTab"),
        "Trim-low filter": _tab("outliers_trim", "TrimLowOutlierTab"),
        "Manual removal": _tab("outliers_manualremoval", "ManualRemovalOutlierTab"),
    },
    # Eddy-covariance flux processing (dv.flux). Its own menu — a first-class
    # diive domain that will grow (gap-filling, USTAR, storage, ...).
    "Flux": {
        "Flux processing chain": _tab("fluxchain", "FluxChainTab"),
        "USTAR detection": _tab("ustar_detection", "UstarDetectionTab"),
        "Time lag analysis": _tab("timelag", "TimeLagAnalysisTab"),
        "Nighttime partitioning (ONEFlux)": _tab("partitioning_nighttime_oneflux", "NighttimePartitioningOneFluxTab"),
        "Nighttime partitioning (REddyProc)": _tab("partitioning_nighttime_reddyproc", "NighttimePartitioningReddyProcTab"),
        "Daytime partitioning (REddyProc)": _tab("partitioning_daytime_reddyproc", "DaytimePartitioningReddyProcTab"),
        "Daytime partitioning (ONEFlux)": _tab("partitioning_daytime_oneflux", "DaytimePartitioningOneFluxTab"),
        "Random uncertainty (PAS20)": _tab("uncertainty_randunc", "RandomUncertaintyTab"),
        "Joint uncertainty (PAS20)": _tab("uncertainty_jointunc", "JointUncertaintyTab"),
    },
    # Data corrections (dv.corrections). Folded into the combined "Cleaning" menu
    # alongside Outliers (see MainWindow._build_menus). One tab per correction,
    # all sharing BaseCorrectionTab (the RF/XGB shared-template approach).
    "Corrections": {
        "Remove nighttime zero offset": _tab("corrections_nighttime_offset", "NighttimeZeroOffsetTab"),
        "Remove relative humidity offset": _tab("corrections_relativehumidity_offset", "RelativeHumidityOffsetTab"),
        "Set to max threshold": _tab("corrections_setto_threshold", "SetToMaxThresholdTab"),
        "Set to min threshold": _tab("corrections_setto_threshold", "SetToMinThresholdTab"),
        "Set to value": _tab("corrections_setto_value", "SetToValueTab"),
        "Set exact values to missing": _tab("corrections_set_missing", "SetExactToMissingTab"),
    },
    # Gap-filling (dv.gapfilling). Its own menu.
    "Gap-filling": {
        "XGBoost gap-filling": _tab("gapfilling", "XGBoostGapFillingTab"),
        "Random Forest gap-filling": _tab("gapfilling_randomforest", "RandomForestGapFillingTab"),
        "MDS gap-filling": _tab("gapfilling_mds", "MdsGapFillingTab"),
    },
    # Exploratory analysis & diagnostics (dv.analysis).
    "Analyze": {
        "Data profile": _tab("profile", "ProfileTab"),
        "Gaps & coverage": _tab("gaps", "GapDashboardTab"),
        "Driver explorer": _tab("drivers", "DriverExplorerTab"),
        "Compound extremes": _tab("compound_extremes", "CompoundExtremesTab"),
        "Seasonal trend & anomalies": _tab("seasonaltrend", "SeasonalTrendTab"),
        "Spectrogram": _tab("spectrogram", "SpectrogramTab"),
    },
    "Settings": {
        "Appearance": _tab("settings", "SettingsTab"),
        "Project settings": _tab("site", "ProjectSettingsTab"),
    },
    # Database I/O (InfluxDB, optional 'db' group). Folded into the File menu as
    # a "Database ▸" submenu in MainWindow (it's another data source/sink).
    "Database": {
        "Database connection": _tab("database", "DatabaseConnectionTab"),
        "Database explorer": _tab("database_explorer", "DatabaseExplorerTab"),
        "Meteo screening (database)": _tab("meteo_screening", "MeteoScreeningTab"),
    },
}

#: Flat label -> factory lookup (used to open a tab by its menu label).
MENU_TAB_CLASSES: dict[str, LazyTab] = {
    label: factory for group in MENU_TABS.values() for label, factory in group.items()
}

#: Menu tabs that may exist only once (re-selecting focuses the existing one).
#: Everything else opens a new, numbered instance each time ("Hampel filter 1",
#: "Hampel filter 2", ...). Two kinds belong here: tabs that edit a single
#: app-wide singleton, where a second copy would show conflicting state —
#: Appearance (theme.manager) and Project settings (site.manager), plus Metadata
#: explorer (the target of the "Edit metadata..." relay from every variable
#: list), plus the Database tabs (the single app-wide InfluxDB handle); and the
#: single-variable explorers (Driver explorer, Gaps & coverage, Spectrogram),
#: where a duplicate would just re-do the same heavy compute on the same data.
SINGLE_INSTANCE_TABS: set[str] = {
    "Appearance", "Project settings", "Metadata explorer",
    "Database connection", "Database explorer", "Meteo screening (database)",
    "Driver explorer", "Gaps & coverage", "Spectrogram"}
