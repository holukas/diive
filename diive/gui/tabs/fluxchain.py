"""
GUI.TABS.FLUXCHAIN: FLUX POST-PROCESSING CHAIN (Input + L2 + L3.1 + L3.2 + L3.3 + L4.1 + L4.2)
=============================================================================================

A guided tab for the Swiss-FluxNet flux post-processing chain. It wires the **Input**
(site + flux column), **Level 2** (quality-flag tests), **Level 3.1** (single-point
storage correction), **Level 3.2** (an optional outlier-detection chain), and
**Level 3.3** (optional USTAR filtering — thresholds given by percentile or detected
with ONEFlux / REddyProc, CUT or VUT, optionally nighttime only and with the 40
ONEFlux REF scenarios) using the composable library callables (`init_flux_data` +
`run_level2` + `run_level31` + `make_level32_detector` / `run_level32` +
`run_level33_ustar_detection`), shows the
deepest level's QCF-filtered flux as a heatmap plus the L3.2 QCF distribution /
L3.3 scenarios, and — the point of the feature — emits the exact reproducible diive
script via **Copy Python**. L3.3 requires at least one L3.2 outlier test, and applies
only to CO2/CH4/N2O (for H/LE use threshold 0); the library validates and the tab
surfaces the error.

L3.2 is built as a chain of outlier tests: pick a method, set its parameters, and
**Add step**; each step is one `StepwiseOutlierDetection.flag_*` call (committed
with `addflag()`), and `run_level32` aggregates the flags into the overall QCF. The
per-method parameter widgets are the shared `widgets/stepwise_method_params.py`
registry; a step is a `{"method", "kwargs"}` dict (the shape `level32_to_code` wants).

All computation is library work (`init_flux_data`, `run_level2`, `run_level31`,
`make_level32_detector`/`run_level32`); this tab only collects parameters and calls
them (heavy runs on a worker thread). Script-generation is the library's
`level31_to_code` / `level32_to_code` (they own the API shape).

The site coordinates and the UTC offset are read-only mirrors of Project settings
(`site.manager`). Without a configured site the tab refuses to initialize the chain
or copy its script. After a settings change, levels initialized with the old values
can no longer be added to the dataset.

**Level 4.1** gap-fills the per-scenario L3.3 flux: tick any of rf / xgb / mds
(additive across methods — each replaces only its own previous result), pick
rf/xgb predictor features and the MDS SW_IN / TA / VPD driver columns (each
with an optional QC flag column, default none), mode (diive / ONEFlux / REddyProc)
and **Fill long gaps** (follows the mode, as ``fill_long_gaps=None``, until the
user clicks it, and only a choice other than the mode default is passed), and the
run fans out one gap-fill per USTAR scenario on the worker thread. The canvas
then shows the method comparison (cumulative overlay or side-by-side heatmaps)
via the library's `plot_cumulative_comparison(ax=...)` / `plot_gapfilled_heatmaps(fig=...)`,
and **Copy Python** switches to `level41_to_code`. With REF on at L3.3, L4.1 runs
MDS on all scenarios, then `run_level41_ustar_reference` (adds the CUT_REF /
VUT_REF series), then rf / xgb. These leave out the 40 REF versions but include
the scenario the REF maps to (the library's `select_scenarios`).

**Level 4.2** partitions the NEE of each u* scenario into GPP and RECO with
`run_level42_partitioning`: a method (nighttime / daytime) and a mode (ONEFlux /
REddyProc), the driver columns that variant needs, and for the nighttime method
(and daytime ONEFlux) the L4.1 method that gives the gap-filled NEE. Runs with
other variants add up, as at L4.1. With the NEE REF on (L3.3 REF and L4.1 MDS),
**Add GPP / RECO REF (ONEFlux)** also runs `run_level42_ustar_reference` for the
variant, which adds ``GPP_<tag>_CUT_REF`` / ``RECO_<tag>_CUT_REF`` (or VUT) and
reports the REF percentile per product and time resolution. A variant that reads
a gap-filled NEE must take it from mds for the REF (only MDS fills the 40 REF
versions). **Copy Python** switches to `level42_to_code`. Re-running L4.1 starts
from the container L4.2 started from, so the partitioning of the old gap-filling
goes.

Thresholds with free labels from projects saved before the percentile form
(e.g. ``CUT_NONE``) still run through `run_level33_constant_ustar`.

The chain stays on the **composable per-level** path (not `run_chain`/`FluxConfig`)
so L3.2 can be a real inspected `StepwiseOutlierDetection` chain with a separate QCF
surface. All script-generation is the library's `level*_to_code` renderers; the
tab only collects parameters and calls them.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import re
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import pandas as pd

import diive as dv
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.flux.postprocessingchain import (
    VM97_SUBTESTS, init_flux_data, level2_test_inputs, level31_storage_col,
    level31_to_code, level32_to_code, level33_to_code, level41_to_code, level42_to_code,
    make_level32_detector, make_level41_engineer, run_level2, run_level31,
    run_level32, run_level33_constant_ustar, run_level33_ustar_detection,
    run_level41_mds, run_level41_rf, run_level41_ustar_reference, run_level41_xgb,
    run_level42_partitioning, run_level42_ustar_reference, ustar_scenario_labels,
)
from diive.flux.postprocessingchain.levels.level42 import partition_variant
from diive.flux.partitioning import partitioning_inputs, partitioning_optional_inputs
from diive.flux.lowres.common import detect_fluxbasevar
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.ustar_detection import (
    USTAR_METHOD_SHORT, USTAR_MP_MODES, VUT_CUT_TIP, make_ustar_method_combo,
    set_ustar_method,
)
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.driver_qc import QC_TIP, fill_qc_combo, qc_value, set_qc_value
from diive.gui.widgets.feature_picker import FeaturePicker
from diive.gui.widgets.mds_mode import make_mode_combo, mode_value, set_mode_value
from diive.gui.widgets.partitioning_mode import (
    combo_value, make_method_combo, set_combo_value,
)
from diive.gui.widgets.partitioning_mode import make_mode_combo as make_partition_mode_combo
from diive.gui.widgets.flux_pipeline_rail import PipelineRail
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.project_offset import ProjectUtcOffset, project_utc_offset
from diive.gui.widgets.stepwise_method_params import STEP_METHOD_BY_KEY, method_labels
from diive.gui.widgets.tab_chrome import build_titlebar
from diive.gui.widgets.weak_slot import weak_slot

#: Pipeline stages shown on the rail -> (badge, title). Index = stacked page +
#: the level reached after a run (see _level_to_stage).
_STAGES = [
    ("INPUT", "Site & flux"),
    ("L2", "Quality flags"),
    ("L3.1", "Storage"),
    ("L3.2", "Outliers"),
    ("L3.3", "USTAR"),
    ("L4.1", "Gap-filling"),
    ("L4.2", "Partitioning"),
]

#: Columns init_flux_data computes itself and rejects if already present.
_RESERVED = ("SW_IN_POT", "DAYTIME", "NIGHTTIME")

_C_MUTED = "#6B7780"

#: Status when the chain would need the site but Project settings have none.
_SITE_NOT_SET = (
    "The flux chain needs the site location for potential radiation and the "
    "day/night split, but no coordinates are configured. Set latitude, longitude "
    "and UTC offset in Settings -> Project settings first.")

#: Status when the chain ran with other coordinates or UTC offset than the
#: project's current ones.
_SITE_CHANGED = (
    "Project settings changed after the chain was initialized. Run it again from "
    "Input to use the new site coordinates and UTC offset. Until then, its results "
    "cannot be added to the dataset.")

#: L2 tests offered as checkboxes -> (label, default-on). Each maps to a
#: run_level2 keyword; the settings dict is built in `_level2_settings`.
_L2_TESTS = [
    ("ssitc", "SSITC (steady-state)", True),
    ("gas_completeness", "Gas completeness", True),
    ("spectral_correction_factor", "Spectral correction", True),
    ("raw_data_screening_vm97", "Raw-data screening (VM97)", True),
    ("signal_strength", "Signal strength (AGC)", False),
    ("angle_of_attack", "Angle of attack", False),
    ("steadiness_of_horizontal_wind", "Steadiness of horiz. wind", False),
]


#: Tooltip of the L3.3 REF option.
_REF_TIP = (
    "Prepare the REF u* threshold as ONEFlux selects it. Adds the 40 ONEFlux "
    "percentiles (1.25 to 98.75) as extra u* scenarios. Level 4.1 gap-fills all "
    "scenarios with MDS and then selects the REF (CUT_REF or VUT_REF) by model "
    "efficiency.\n\n"
    "With given thresholds, the list must also hold the 40 percentiles "
    "1.25, 3.75, ..., 98.75.\n"
    "For the REF ONEFlux would select, run MDS with mode ONEFlux.\n"
    "The u* filter of diive keeps the first record after a low-u* period and "
    "ONEFlux removes it. The REF can therefore differ from ONEFlux's at some "
    "time resolutions.\n\n"
    "Random Forest and XGBoost run after the REF selection, on the main scenarios "
    "and the scenario the REF maps to, not on the other REF versions.")

#: L4.2 driver columns -> row label. The keys are run_level42_partitioning
#: arguments. Which rows show depends on the method and mode.
_L42_DRIVERS = (
    ("ta", "TA measured (°C)"),
    ("sw_in", "SW_IN measured (W m⁻²)"),
    ("ta_f", "TA gap-filled (°C)"),
    ("sw_in_f", "SW_IN gap-filled (W m⁻²)"),
    ("vpd_f", "VPD gap-filled (kPa)"),
)

#: PartitioningNEE input name -> run_level42_partitioning driver argument.
_L42_DRIVER_ARGS = {"ta": "ta", "sw_in": "sw_in", "ta_f": "ta_f",
                    "sw_in_f": "sw_in_f", "vpd": "vpd_f"}

#: Tooltip of the L4.2 GPP / RECO REF option.
_REF42_TIP = (
    "Select the REF of GPP and RECO as ONEFlux does (step ure). Partitions the 40 "
    "REF versions of the NEE REF and the u* scenario at percentile 50 with this "
    "method and mode, then selects one REF for GPP and one for RECO by model "
    "efficiency, on half-hourly, daily, weekly, monthly and yearly values.\n\n"
    "Needs the NEE REF: 'Add REF (ONEFlux)' in Level 3.3 and MDS in Level 4.1. "
    "Where the method reads a gap-filled NEE, it must come from mds, because only "
    "MDS fills the 40 REF versions.\n"
    "Slow: about 41 partitionings. Daytime ONEFlux takes minutes per year.\n\n"
    "For the REddyProc modes, which ONEFlux does not run, the same selection is "
    "applied to their results (a diive extension).")

#: Line colours of the L4.2 preview (as in the NEE partitioning tab).
_C_GPP = "#43A047"
_C_RECO = "#E53935"


def _label_percentile(label: str) -> float | None:
    """Percentile of a scenario label written as ``ustar_scenario_labels`` writes
    it (``CUT_16`` -> 16, ``CUT_0125`` -> 1.25), else None (free labels such
    as ``CUT_NONE`` or the auto labels ``CUT_0``, ``CUT_1``)."""
    m = re.fullmatch(r"CUT_(\d{2,4})", str(label))
    if m is None:
        return None
    digits = m.group(1)
    p = int(digits) / 100 if len(digits) == 4 else float(int(digits))
    return p if ustar_scenario_labels("CUT", [p]) == [label] else None


class _ChainSignals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject)."""
    done = Signal(object, object)  # (FluxLevelData, its L4.1 container or None), run-all
    level_done = Signal(object)  # (FluxLevelData, stage_idx) — single-level run
    failed = Signal(str)
    features_created = Signal(object)  # DataFrame — a level's results pushed to dataset


class FluxChainTab(DiiveTab):
    """Guided flux post-processing chain — Input + Level 2 (first slice)."""

    title = "Flux chain"

    def build(self) -> QWidget:
        self._df = None
        self._data = None        # the evolving container; each level appends to it
        self._reached = -1        # deepest stage index run on _data (-1 = none)
        # The container L4.2 started from. Re-running L4.1 starts from it, so
        # the partitioning of the old gap-filling goes with the confirm prompt.
        self._l41_data = None
        self._running = False
        self._level_run_btns: dict[int, QPushButton] = {}
        self._level_add_btns: dict[int, QPushButton] = {}
        self._sig = _ChainSignals()
        self._sig.done.connect(self._on_done)
        self._sig.level_done.connect(self._on_level_done)
        self._sig.failed.connect(self._on_failed)
        # MainWindow auto-wires this to merge the emitted columns into the dataset.
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        outer = QVBoxLayout(root)

        # Action bar: title + the chain-wide actions (run the whole chain, copy).
        self.run_btn = QPushButton("Run through Level 3.1")
        self.run_btn.setToolTip("Run the whole chain from scratch through the deepest "
                                "configured level.")
        self.run_btn.clicked.connect(self._run)
        theme.set_button_role(self.run_btn, "confirm")
        # Standardized copy button: copies the script to the clipboard with a
        # "Copied ✓" flash — no code dump in the summary box.
        self.code_btn = CopyPythonButton(self._code)
        outer.addLayout(build_titlebar("Flux post-processing chain", self.run_btn, self.code_btn))

        # Pipeline rail: the chain as selectable stage cards (the navigation).
        self.rail = PipelineRail(_STAGES)
        self.rail.selected.connect(self._select_stage)
        rail_scroll = QScrollArea()
        rail_scroll.setWidgetResizable(True)
        rail_scroll.setFixedHeight(108)
        rail_scroll.setFrameShape(QFrame.NoFrame)
        rail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        rail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        rail_scroll.setWidget(self.rail)
        outer.addWidget(rail_scroll)

        # Lower: inspector (the selected stage's controls) | diagnostics canvas |
        # a text column (run status + QCF report) to the right of the plot. Only
        # one stage's form shows at a time, so the config stays compact.
        lower = QHBoxLayout()
        self._pages = QStackedWidget()
        for page in (self._input_group(), self._level2_group(), self._level31_group(),
                     self._level32_group(), self._level33_group(), self._level41_group(),
                     self._level42_group()):
            self._pages.addWidget(page)
        # Combos with long item text (column names) would otherwise force the page
        # wider than the inspector, overflowing it (the left-clip). Cap their width
        # demand so the page always fits the fixed inspector width.
        for combo in self._pages.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(6)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        inspector = QScrollArea()
        inspector.setWidgetResizable(True)
        # Wide enough that the widest stage page (descriptive checkbox labels +
        # the shared L3.2 stepwise widgets) fits without horizontal overflow —
        # the cause of the left-clipped labels. The plot gets the remaining room.
        inspector.setFixedWidth(545)
        inspector.setFrameShape(QFrame.NoFrame)
        inspector.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inspector.setWidget(self._pages)
        lower.addWidget(inspector)

        self.canvas = MplCanvas()
        lower.addWidget(self.canvas, stretch=3)

        # Right text column: run status (small) over the QCF report (fills).
        textcol = QVBoxLayout()
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(84)
        textcol.addWidget(self.summary)
        rep_head = QHBoxLayout()
        rep_lbl = QLabel(theme.manager.label_text("QCF report"))
        rep_lbl.setFont(theme.manager.tracked_font())
        rep_lbl.setStyleSheet("font-weight: bold;")
        rep_head.addWidget(rep_lbl)
        rep_head.addStretch(1)
        self.report_copy = CopyPythonButton(self._report_text, text="Copy report")
        self.report_copy.setEnabled(False)
        rep_head.addWidget(self.report_copy)
        textcol.addLayout(rep_head)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.report.setFont(QFont("Consolas", 9))
        self.report.setPlaceholderText(
            "Run a level to see its QCF flag breakdown (retained / rejected per "
            "test, with daytime / nighttime split when the site is configured).")
        textcol.addWidget(self.report, stretch=1)
        lower.addLayout(textcol, stretch=2)
        outer.addLayout(lower, stretch=1)

        self._apply_tooltips()
        self._wire_status_refresh()
        self._select_stage(0)
        self._update_run_label()
        self._update_level_buttons()
        return root

    def _level_run_button(self, idx: int, label: str) -> QPushButton:
        """A per-level run button stored in `_level_run_btns` and enabled by reach."""
        btn = QPushButton(label)
        btn.setToolTip("Run only this level on the current chain output; the result "
                       "feeds the next level.")
        theme.set_button_role(btn, "confirm")
        btn.clicked.connect(weak_slot(self._run_level, idx))
        self._level_run_btns[idx] = btn
        return btn

    def _level_add_button(self, idx: int) -> QPushButton:
        """An "Add to dataset" button for a level, enabled once that level ran."""
        btn = QPushButton("Add to dataset")
        btn.setToolTip("Add this level's flag/QCF columns and QCF-filtered flux to "
                       "the dataset (available everywhere in the app).")
        btn.clicked.connect(weak_slot(self._add_level, idx))
        self._level_add_btns[idx] = btn
        return btn

    def _level_button_row(self, idx: int, run_label: str) -> QHBoxLayout:
        """Run button + its "Add to dataset" button on one row."""
        row = QHBoxLayout()
        row.addWidget(self._level_run_button(idx, run_label))
        row.addWidget(self._level_add_button(idx))
        row.addStretch(1)
        return row

    # --- pipeline rail navigation + status ---
    def _select_stage(self, idx: int) -> None:
        self._pages.setCurrentIndex(idx)
        self.rail.set_selected(idx)

    def _wire_status_refresh(self) -> None:
        """Refresh the rail's status pills whenever a stage's inputs change.

        The L3.2/L3.3/L4.1 controls already funnel through `_update_run_label`
        (which refreshes the rail); wire the rest here."""
        for cb in self.l2_checks.values():
            cb.toggled.connect(self._refresh_rail)
        self.l31_gapfill.toggled.connect(self._refresh_rail)
        self.l31_zero.toggled.connect(self._refresh_rail)
        self.strgcol.currentTextChanged.connect(self._refresh_rail)
        self.fluxcol.currentTextChanged.connect(self._refresh_rail)
        # The flux column determines the standard L2 input columns -> re-seed the
        # per-test column pickers (which then refreshes availability).
        self.fluxcol.currentTextChanged.connect(self._populate_l2_cols)
        # It also drives the L3.1 storage-column auto-detection marker.
        self.fluxcol.currentTextChanged.connect(self._refresh_levels_info)

    @staticmethod
    def _level_to_stage(level: str) -> int:
        """Map a reached level label to its rail card index (run reaches >= L3.1)."""
        return {"3.1": 2, "3.2": 3, "3.3": 4, "4.1": 5, "4.2": 6}.get(level, 2)

    def _stage_statuses(self) -> list[tuple[str, str]]:
        """One ``(pill_text, kind)`` per stage, derived from the live controls."""
        fc = self.fluxcol.currentText()
        n2 = sum(cb.isChecked() for cb in self.l2_checks.values())
        if self.l31_zero.isChecked():
            l31 = ("storage → 0", "set")
        elif self.l31_gapfill.isChecked():
            l31 = ("storage + gap-fill", "set")
        else:
            l31 = ("storage", "set")
        n32 = len(self._steps)
        if self.l33_enable.isChecked():
            l33kw = self._level33_kwargs()
            ref = " + REF" if l33kw and l33kw.get("reference") else ""
            if l33kw is None:
                l33 = ("configure", "warn")
            elif l33kw.get("_detection") and "thresholds" not in l33kw:
                n = len(l33kw["percentiles"])
                tag = {"cut": "CUT", "vut": "VUT"}[l33kw.get("mode", "cut")]
                l33 = (f"detect {tag} · {n}{ref}", "set")
            else:
                n = len(self._ustar)
                l33 = (f"{n} scenario" + ("s" if n != 1 else "") + ref, "set")
        else:
            l33 = ("off", "off")
        methods = self._level41_methods()
        l42cfg = self._level42_cfg()
        if l42cfg:
            l42 = ("NT" if l42cfg["method"] == "nighttime" else "DT") + " · " \
                + ("ONEFlux" if l42cfg["mode"] == "oneflux" else "REddyProc") \
                + (" + REF" if l42cfg.get("ustar_reference") else "")
            l42 = (l42, "set")
        else:
            l42 = ("off", "off")
        return [
            (fc or "—", "set" if fc else "todo"),
            (f"{n2} test" + ("s" if n2 != 1 else ""), "warn" if n2 == 0 else "set"),
            l31,
            (f"{n32} step" + ("s" if n32 != 1 else ""), "set") if n32 else ("none", "off"),
            l33,
            (" · ".join(methods), "set") if methods else ("off", "off"),
            l42,
        ]

    def _refresh_rail(self, *_) -> None:
        if getattr(self, "rail", None) is None:
            return  # called by a control signal before the rail exists (build order)
        for i, (text, kind) in enumerate(self._stage_statuses()):
            self.rail.card(i).set_status(text, kind)

    # --- config groups ---
    def _input_group(self) -> QGroupBox:
        box = QGroupBox("Input & site")
        form = QFormLayout(box)
        self.fluxcol = QComboBox()
        form.addRow("Flux column", self.fluxcol)
        # USTAR (friction velocity) column — required by init_flux_data and read
        # by L3.3; pickable so non-'USTAR'-named data can still initialize.
        self.ustarcol = QComboBox()
        self.ustarcol.currentTextChanged.connect(self._refresh_levels_info)
        form.addRow("USTAR column", self.ustarcol)
        self.ustar_mark = self._info_marker()
        form.addRow("", self.ustar_mark)
        # Read-only mirrors of Project settings: the coordinates and the offset
        # have one home, so the chain always runs for the project's site.
        self.site_lat = self._dspin(0.0, -90, 90, 4)
        form.addRow("Latitude", self.site_lat)
        self.site_lon = self._dspin(0.0, -180, 180, 4)
        form.addRow("Longitude", self.site_lon)
        for w in (self.site_lat, self.site_lon):
            # Disabled, not read-only: read-only still reads as an input field.
            w.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            w.setEnabled(False)
        self.utc_offset = ProjectUtcOffset()
        # Follows site.manager and dies with the tab's widgets (no direct
        # connection from the singleton to this plain-Python tab).
        self.utc_offset.changed.connect(self._on_project_settings_changed)
        form.addRow("UTC offset", self.utc_offset)
        self.site_note = QLabel()
        self.site_note.setWordWrap(True)
        form.addRow(self.site_note)
        self._seed_site()
        self.nighttime_threshold = self._dspin(20.0, 0, 200, 1)
        form.addRow("Night threshold (W m⁻²)", self.nighttime_threshold)
        self.day_qcf = QComboBox()
        self.day_qcf.addItems(["1", "2"])
        form.addRow("Daytime accept QCF below", self.day_qcf)
        self.night_qcf = QComboBox()
        self.night_qcf.addItems(["1", "2"])
        form.addRow("Nighttime accept QCF below", self.night_qcf)
        form.addRow(self._level_run_button(0, "Initialize chain"))
        return box

    #: Per-test column-picker row labels. One combo per input column; the
    #: completeness test reads two columns.
    _L2_COL_LABELS = {
        "ssitc": ["SSITC column"],
        "gas_completeness": ["expected-records col", "base-var records col"],
        "spectral_correction_factor": ["SCF column"],
        "raw_data_screening_vm97": ["VM97 column"],
        "angle_of_attack": ["AoA column"],
        "steadiness_of_horizontal_wind": ["NSHW column"],
    }

    def _level2_group(self) -> QGroupBox:
        box = QGroupBox("Level 2 — quality flags")
        v = QVBoxLayout(box)
        # Which variables L2 works on (flux column + its base var); filled in
        # _refresh_l2_availability once data + a flux column are known.
        self.l2_header = QLabel("Flux: —   ·   base var: —")
        self.l2_header.setStyleSheet("color: #2A3942; font-weight: 600;")
        v.addWidget(self.l2_header)
        intro = QLabel("The always-on missing-values test reads the flux column. "
                       "Each test below reads the column(s) you pick — seeded to the "
                       "standard EddyPro-FLUXNET name; a test with no column is disabled.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        v.addWidget(intro)

        self.l2_checks: dict[str, QCheckBox] = {}
        self.l2_cols: dict[str, list[QComboBox]] = {}   # per-test input-column pickers
        self.l2_inputs: dict[str, QLabel] = {}          # per-test availability marker
        self.l2_vm97_checks: dict[str, QCheckBox] = {}
        self._l2_tips: dict[str, str] = {}
        for key, label, on in _L2_TESTS:
            cb = QCheckBox(label)
            cb.setChecked(on)
            v.addWidget(cb)
            self.l2_checks[key] = cb
            if key == "signal_strength":
                v.addLayout(self._signal_strength_options())
            else:
                v.addLayout(self._col_picker_row(key))
            v.addWidget(self._col_marker(key))
            if key == "raw_data_screening_vm97":
                cb.toggled.connect(self._refresh_vm97_enabled)
                v.addLayout(self._vm97_subtests())
        v.addLayout(self._level_button_row(1, "Run Level 2"))
        return box

    def _col_picker_row(self, key: str) -> QFormLayout:
        """Column combo(s) for a test, stored in ``l2_cols[key]`` (seeded later)."""
        form = QFormLayout()
        form.setContentsMargins(22, 0, 0, 2)
        combos: list[QComboBox] = []
        for lab in self._L2_COL_LABELS.get(key, ["column"]):
            combo = QComboBox()
            combo.currentTextChanged.connect(self._refresh_l2_availability)
            form.addRow(lab, combo)
            combos.append(combo)
        self.l2_cols[key] = combos
        return form

    def _col_marker(self, key: str) -> QLabel:
        mark = QLabel("")
        mark.setWordWrap(True)  # long column names wrap instead of forcing width
        mark.setContentsMargins(22, 0, 0, 0)
        mark.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        self.l2_inputs[key] = mark
        return mark

    @staticmethod
    def _info_marker() -> QLabel:
        """A small wrapping availability/info label (shared by the level pages)."""
        lbl = QLabel("")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        return lbl

    @staticmethod
    def _set_marker(label: QLabel, ok: bool, text: str) -> None:
        label.setText(text)
        label.setStyleSheet(
            f"color: {'#2E9E5B' if ok else '#C0392B'}; font-size: 11px;")

    def _refresh_levels_info(self, *_) -> None:
        """Update the column-availability markers on the non-L2 level pages
        (mirrors L2's per-test availability so every level shows its inputs)."""
        if self._df is None:
            return
        cols = {str(c) for c in self._df.columns}
        fluxcol = self.fluxcol.currentText()
        # Input — USTAR column.
        if hasattr(self, "ustar_mark"):
            u = self.ustarcol.currentText().strip()
            ok = bool(u) and u in cols
            self._set_marker(self.ustar_mark, ok,
                             "✓ USTAR column present" if ok
                             else "✗ pick a USTAR column present in the data")
        # L3.1 — storage-term column (explicit pick, else the auto-detected name).
        if hasattr(self, "strg_mark"):
            default = level31_storage_col(fluxcol)
            chosen = self.strgcol.currentText().strip()
            eff = chosen or (default or "")
            ok = bool(eff) and eff in cols
            if chosen:
                txt = f"uses {chosen}" + ("" if ok else " — not in data")
            elif default:
                txt = f"auto-detect → {default}" + ("" if ok else " — not in data")
            else:
                txt = "no standard storage column for this flux — pick one"
            self._set_marker(self.strg_mark, ok, ("✓ " if ok else "✗ ") + txt)
        # L3.3 — the USTAR column it filters on (+ applicability reminder).
        if hasattr(self, "l33_info"):
            u = self.ustarcol.currentText().strip() or "USTAR"
            self.l33_info.setText(
                f"Filters on USTAR column '{u}'. Applies to CO2 / CH4 / N2O only — "
                f"for H / LE use threshold 0.")
        # L3.3 detect — TA / SW_IN driver availability.
        if hasattr(self, "l33_detect_mark"):
            parts, allok = [], True
            for role, combo in (("TA", self.l33_ta), ("SW_IN", self.l33_swin)):
                c = combo.currentText().strip()
                ok = bool(c) and c in cols
                allok = allok and ok
                parts.append(f"{role} {'✓' if ok else '✗'}")
            self._set_marker(self.l33_detect_mark, allok,
                             "Detection drivers — " + "   ".join(parts))
        # L4.1 — the three MDS driver columns.
        if hasattr(self, "mds_mark"):
            parts, allok = [], True
            for role, combo in (("SW_IN", self.mds_swin), ("TA", self.mds_ta),
                                ("VPD", self.mds_vpd)):
                c = combo.currentText().strip()
                ok = bool(c) and c in cols
                allok = allok and ok
                parts.append(f"{role} {'✓' if ok else '✗'}")
            self._set_marker(self.mds_mark, allok, "MDS drivers — " + "   ".join(parts))
        # L4.2: the driver columns the selected method and mode need.
        if hasattr(self, "l42_mark"):
            parts, allok = [], True
            for arg in self._l42_drivers():
                c = self._l42_cols[arg].currentText().strip()
                ok = bool(c) and c in cols
                allok = allok and ok
                parts.append(f"{arg} {'✓' if ok else '✗'}")
            self._set_marker(self.l42_mark, allok,
                             "Partitioning drivers: " + "   ".join(parts))

    def _signal_strength_options(self) -> QFormLayout:
        """Column + direction + threshold for the signal-strength (AGC) test."""
        form = QFormLayout()
        form.setContentsMargins(22, 0, 0, 4)
        self.signal_strength_col = QComboBox()
        self.signal_strength_col.currentTextChanged.connect(self._refresh_l2_availability)
        form.addRow("column", self.signal_strength_col)
        self.ss_method = QComboBox()
        self.ss_method.addItems(["discard below", "discard above"])
        self.ss_method.setToolTip("Flag records where signal strength falls below "
                                  "(low signal = problem on most IRGAs) or rises above "
                                  "the threshold (high AGC = dirty optics on some).")
        form.addRow("direction", self.ss_method)
        self.ss_threshold = QSpinBox()
        self.ss_threshold.setRange(0, 100000)
        self.ss_threshold.setValue(60)
        form.addRow("threshold", self.ss_threshold)
        return form

    def _vm97_subtests(self) -> QVBoxLayout:
        """The eight VM97 raw-data screening sub-tests as individual toggles."""
        wrap = QVBoxLayout()
        wrap.setContentsMargins(22, 0, 0, 4)
        #: defaults match the FLUXNET convention (spikes + dropout on).
        _on = {"spikes", "dropout"}
        for key, label, kind in VM97_SUBTESTS:
            cb = QCheckBox(label)  # the library label already encodes hard/soft
            cb.setChecked(key in _on)
            kindtxt = "soft (flag 1)" if kind == "soft" else "hard (flag 2)"
            cb.setToolTip(f"VM97 sub-test '{key}' — {kindtxt} flag, extracted from "
                          "the chosen VM97 column.")
            wrap.addWidget(cb)
            self.l2_vm97_checks[key] = cb
        return wrap

    def _refresh_vm97_enabled(self) -> None:
        master = self.l2_checks.get("raw_data_screening_vm97")
        on = bool(master) and master.isEnabled() and master.isChecked()
        for cb in self.l2_vm97_checks.values():
            cb.setEnabled(on)

    def _l2_basevar(self) -> str | None:
        fluxcol = self.fluxcol.currentText()
        try:
            return detect_fluxbasevar(fluxcol) if fluxcol else None
        except Exception:
            return None  # unknown flux column -> can't template the inputs

    def _populate_l2_cols(self, *_) -> None:
        """Fill each test's column combo(s) with dataset columns, seeded to the
        standard templated name when present (re-seeded on a flux-column change)."""
        if self._df is None or not hasattr(self, "l2_cols"):
            return
        cols = [str(c) for c in self._df.columns]
        basevar = self._l2_basevar()
        info = level2_test_inputs(self.fluxcol.currentText(), basevar) if basevar else {}
        for key, combos in self.l2_cols.items():
            defaults = info.get(key, {}).get("inputs", [])
            for i, combo in enumerate(combos):
                prev = combo.currentText()
                default = defaults[i] if i < len(defaults) else ""
                combo.blockSignals(True)
                combo.clear()
                combo.addItems([""] + cols)
                # Prefer the standard column; else keep a still-valid prior pick.
                combo.setCurrentText(default if default in cols
                                     else (prev if prev in cols else ""))
                combo.blockSignals(False)
        self._refresh_l2_availability()

    def _refresh_l2_availability(self, *_) -> None:
        """Update each L2 test's availability marker + gate it on a chosen column."""
        if self._df is None or not hasattr(self, "l2_cols"):
            return
        basevar = self._l2_basevar()
        self.l2_header.setText(
            f"Flux: {self.fluxcol.currentText() or '—'}   ·   base var: {basevar or '—'}")
        cols = {str(c) for c in self._df.columns}
        for key, cb in self.l2_checks.items():
            if key == "signal_strength":
                chosen = [self.signal_strength_col.currentText().strip()]
                cb.setEnabled(True)  # the user supplies the column
            else:
                chosen = [c.currentText().strip() for c in self.l2_cols.get(key, [])]
                present = bool(chosen) and all(ch and ch in cols for ch in chosen)
                # Disable (not uncheck) when no valid column: _level2_settings skips
                # disabled tests, and the checked intent survives transient states
                # (e.g. mid flux-column swap) instead of being silently cleared.
                cb.setEnabled(present)
                tip = self._l2_tips.get(key, "")
                cb.setToolTip(tip if present else
                              (tip + "\n" if tip else "") +
                              "Disabled — pick an input column that exists in the data.")
            ok = bool(chosen) and all(ch and ch in cols for ch in chosen)
            # The combo(s) already show the column; the marker is just the state
            # (no long column name here, or an unbreakable token would force the
            # inspector wider than its fixed width and clip the left of the page).
            self.l2_inputs[key].setText(
                "✓ column present" if ok else "✗ choose a column present in the data")
            self.l2_inputs[key].setStyleSheet(
                f"color: {'#2E9E5B' if ok else '#C0392B'}; font-size: 11px;")
        self._refresh_vm97_enabled()
        self._refresh_rail()

    def _level31_group(self) -> QGroupBox:
        box = QGroupBox("Level 3.1 — storage correction")
        v = QVBoxLayout(box)
        note = QLabel("Adds the single-point storage term to the L2 flux "
                      "(e.g. NEE = FC + SC_SINGLE).")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        v.addWidget(note)
        self.l31_gapfill = QCheckBox("Gap-fill storage term (rolling median)")
        self.l31_gapfill.setChecked(True)
        v.addWidget(self.l31_gapfill)
        self.l31_zero = QCheckBox("Set storage to zero (H / LE)")
        self.l31_zero.setChecked(False)
        # Setting storage to zero makes the gap-fill choice irrelevant.
        self.l31_zero.toggled.connect(self.l31_gapfill.setDisabled)
        v.addWidget(self.l31_zero)
        form = QFormLayout()
        self.strgcol = QComboBox()  # "" = auto-detect (FLUXNET/EddyPro naming)
        self.strgcol.currentTextChanged.connect(self._refresh_levels_info)
        form.addRow("Storage column (auto if blank)", self.strgcol)
        v.addLayout(form)
        self.strg_mark = self._info_marker()
        v.addWidget(self.strg_mark)
        v.addLayout(self._level_button_row(2, "Run Level 3.1"))
        return box

    def _level32_group(self) -> QGroupBox:
        box = QGroupBox("Level 3.2 — outlier detection (optional)")
        v = QVBoxLayout(box)
        note = QLabel("Chain outlier tests on the L3.1 flux. Each committed test adds a "
                      "flag column; the overall QCF aggregates them. Order matters.")
        note.setWordWrap(True)
        v.addWidget(note)
        # Method picker → swaps in that method's parameter widget.
        self.l32_method = QComboBox()
        for key, label in method_labels():
            self.l32_method.addItem(label, key)
        self.l32_method.currentIndexChanged.connect(self._on_l32_method_changed)
        v.addWidget(self.l32_method)
        self._l32_param_box = QVBoxLayout()
        self._l32_param_widget = None
        v.addLayout(self._l32_param_box)
        add_btn = QPushButton("Add step")
        add_btn.clicked.connect(self._add_l32_step)
        v.addWidget(add_btn)
        # The committed chain (insertion order = run order).
        self._steps: list[dict] = []
        self.l32_steps_list = QListWidget()
        self.l32_steps_list.setMaximumHeight(120)
        v.addWidget(self.l32_steps_list)
        row = QHBoxLayout()
        for label, slot in (("Remove", self._remove_l32_step),
                            ("Up", weak_slot(self._move_l32_step, -1)),
                            ("Down", weak_slot(self._move_l32_step, 1))):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        v.addLayout(self._level_button_row(3, "Run Level 3.2"))
        self._on_l32_method_changed()  # seed the first param widget
        return box

    # --- L3.2 step chain management ---
    def _on_l32_method_changed(self, *_) -> None:
        key = self.l32_method.currentData()
        if self._l32_param_widget is not None:
            self._l32_param_box.removeWidget(self._l32_param_widget)
            self._l32_param_widget.deleteLater()
        self._l32_param_widget = STEP_METHOD_BY_KEY[key]()
        self._l32_param_box.addWidget(self._l32_param_widget)

    def _add_l32_step(self) -> None:
        if self._l32_param_widget is None:
            return
        step = self._l32_param_widget.step()
        self._steps.append(step)
        self.l32_steps_list.addItem(self._step_summary(step))
        self._update_run_label()

    def _remove_l32_step(self) -> None:
        row = self.l32_steps_list.currentRow()
        if row < 0:
            return
        self.l32_steps_list.takeItem(row)
        del self._steps[row]
        self._update_run_label()

    def _move_l32_step(self, delta: int) -> None:
        row = self.l32_steps_list.currentRow()
        new = row + delta
        if row < 0 or new < 0 or new >= len(self._steps):
            return
        self._steps[row], self._steps[new] = self._steps[new], self._steps[row]
        item = self.l32_steps_list.takeItem(row)
        self.l32_steps_list.insertItem(new, item)
        self.l32_steps_list.setCurrentRow(new)

    @staticmethod
    def _step_summary(step: dict) -> str:
        label = STEP_METHOD_BY_KEY[step["method"]].label
        parts = ", ".join(f"{k}={v}" for k, v in step.get("kwargs", {}).items())
        return f"{label} — {parts}" if parts else label

    def _level33_group(self) -> QGroupBox:
        box = QGroupBox("Level 3.3 — USTAR filtering (optional)")
        v = QVBoxLayout(box)
        self.l33_enable = QCheckBox("Apply USTAR filtering")
        self.l33_enable.setToolTip(
            "Flag low-turbulence (low USTAR) periods. Requires at least one Level 3.2 "
            "outlier test. Only for CO2/CH4/N2O — for H/LE use a constant threshold 0.")
        self.l33_enable.toggled.connect(self._update_run_label)
        self.l33_enable.toggled.connect(self._sync_l33_mode)
        self.l33_enable.toggled.connect(self._sync_l41_ref)
        v.addWidget(self.l33_enable)
        # Which USTAR column it filters on (set on the Input page) + applicability.
        self.l33_info = self._info_marker()
        v.addWidget(self.l33_info)

        # Mode: known thresholds by percentile, or detection from the data.
        modeform = QFormLayout()
        self.l33_mode = QComboBox()
        self.l33_mode.addItems(["Given thresholds", "Detect from the data"])
        self.l33_mode.setToolTip(
            "Given: apply known thresholds, one per percentile. The scenarios get "
            "the labels of detected thresholds (CUT_16, CUT_50, CUT_84).\n"
            "Detect: estimate the threshold from the data with a bootstrap over the "
            "years (ONEFlux or REddyProc), then apply the percentiles as scenarios.")
        self.l33_mode.currentIndexChanged.connect(self._sync_l33_mode)
        self.l33_mode.currentIndexChanged.connect(self._update_run_label)
        modeform.addRow("Mode", self.l33_mode)
        # Night-only filtering applies to both modes. Its SW_IN column is also
        # the night definition of the detection, so one picker serves both.
        self.l33_night = QCheckBox("Filter nighttime only")
        self.l33_night.setToolTip(
            "Flag low-USTAR records at night only (SW_IN below 10 W m-2, or SW_IN "
            "missing), as REddyProc does.\n"
            "Off (default): day and night records are flagged, as in ONEFlux.")
        self.l33_night.toggled.connect(self._sync_l33_mode)
        self.l33_night.toggled.connect(self._update_run_label)
        modeform.addRow(self.l33_night)
        self.l33_swin = QComboBox()
        self.l33_swin.setToolTip("Incoming shortwave radiation (W m-2). Defines the "
                                 "night for the detection and for night-only filtering.")
        modeform.addRow("Shortwave in (SW_IN)", self.l33_swin)
        # REF: the 40 ONEFlux percentiles as extra scenarios (both modes).
        self.l33_ref = QCheckBox("Add REF (ONEFlux)")
        self.l33_ref.setToolTip(_REF_TIP)
        self.l33_ref.toggled.connect(self._sync_l33_mode)
        self.l33_ref.toggled.connect(self._sync_l41_ref)
        self.l33_ref.toggled.connect(self._update_run_label)
        modeform.addRow(self.l33_ref)
        self.l33_ref_note = self._info_marker()
        self.l33_ref_note.setText(
            "Adds 40 u* scenarios (percentiles 1.25 to 98.75). Level 4.1 gap-fills "
            "them with MDS and selects CUT_REF / VUT_REF. Random Forest and XGBoost "
            "run on the main scenarios and the REF scenario only.")
        modeform.addRow(self.l33_ref_note)
        self._l33_modeform = modeform
        v.addLayout(modeform)

        v.addWidget(self._level33_constant_box())
        v.addWidget(self._level33_detect_box())
        v.addLayout(self._level_button_row(4, "Run Level 3.3"))
        self._sync_l33_mode()
        return box

    def _level33_constant_box(self) -> QGroupBox:
        gb = QGroupBox("Given thresholds")
        v = QVBoxLayout(gb)
        note = QLabel("One scenario per percentile, threshold in m s⁻¹. The labels "
                      "are those of detected thresholds (CUT_16, CUT_50, CUT_84). "
                      "Adding a percentile again replaces its value.")
        note.setWordWrap(True)
        v.addWidget(note)
        row = QHBoxLayout()
        row.addWidget(QLabel("Percentile"))
        self.l33_pct = self._dspin(50.0, 0.0, 100.0, 2)
        self.l33_pct.setToolTip("Percentile of the threshold (0 to 100, two decimals).")
        row.addWidget(self.l33_pct)
        row.addWidget(QLabel("Threshold"))
        self.l33_value = self._dspin(0.10, 0.0, 5.0, 3)
        self.l33_value.setToolTip("u* threshold (m s-1).")
        row.addWidget(self.l33_value)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_ustar)
        row.addWidget(add_btn)
        v.addLayout(row)
        # Thresholds as (value, label). The label is the percentile label
        # (CUT_50). Projects saved before the percentile form can hold free
        # labels (CUT_NONE, or "" for auto): those run as constant thresholds.
        self._ustar: list[tuple[float, str]] = []
        self.l33_list = QListWidget()
        self.l33_list.setMaximumHeight(90)
        v.addWidget(self.l33_list)
        self.l33_legacy_note = self._info_marker()
        self.l33_legacy_note.setText(
            "Thresholds with their own labels (from an older project) run as "
            "constant thresholds with these labels, without REF. Remove them to "
            "use thresholds by percentile.")
        self.l33_legacy_note.setVisible(False)
        v.addWidget(self.l33_legacy_note)
        rm = QPushButton("Remove threshold")
        rm.clicked.connect(self._remove_ustar)
        v.addWidget(rm)
        self._l33_constant_box = gb
        return gb

    def _level33_detect_box(self) -> QGroupBox:
        gb = QGroupBox("Detect: bootstrap over the years")
        form = QFormLayout(gb)
        note = QLabel("Estimate the threshold from the nighttime NEE. "
                      "One scenario per percentile.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        form.addRow(note)
        self.l33_method = make_ustar_method_combo()
        self.l33_method.currentIndexChanged.connect(self._sync_l33_mode)
        self.l33_method.currentIndexChanged.connect(self._update_run_label)
        form.addRow("Method", self.l33_method)
        # Which detected threshold(s) to apply.
        self.l33_apply = QComboBox()
        self.l33_apply.addItems(["CUT (one for all years)", "VUT (one per year)"])
        self.l33_apply.setToolTip(
            "CUT: one constant threshold per percentile, applied to the whole record "
            "(CUT_16/50/84).\n"
            "VUT: a threshold per year and percentile, each year filtered by its own "
            "value (VUT_16/50/84).\n\n" + VUT_CUT_TIP + "\n\n"
            "CUT and VUT are alternatives. Pick one filtering strategy.")
        self.l33_apply.currentIndexChanged.connect(self._update_run_label)
        form.addRow("Thresholds", self.l33_apply)
        # Met driver the detector stratifies on (SW_IN is picked above).
        self.l33_ta = QComboBox()
        form.addRow("Air temperature (TA)", self.l33_ta)
        self.l33_detect_mark = self._info_marker()
        form.addRow("", self.l33_detect_mark)
        self.l33_niter = QSpinBox()
        self.l33_niter.setRange(0, 10000)
        self.l33_niter.setValue(0)
        self.l33_niter.setSpecialValueText("Default")
        self.l33_niter.setToolTip("Bootstrap draws per year and method. Default: 100 "
                                  "for ONEFlux, 200 for REddyProc.")
        form.addRow("Bootstrap iterations", self.l33_niter)
        self.l33_njobs = QSpinBox()
        self.l33_njobs.setRange(-1, 256)
        self.l33_njobs.setValue(1)
        self.l33_njobs.setToolTip("Parallel workers (1 = sequential, -1 = all CPUs).")
        form.addRow("Parallel workers", self.l33_njobs)
        self.l33_percentiles = QLineEdit("16, 50, 84")
        self.l33_percentiles.setToolTip("Comma-separated percentiles → one scenario "
                                        "each (CUT_<p> or VUT_<p>).")
        form.addRow("Percentiles", self.l33_percentiles)
        self.l33_ta_classes = QSpinBox()
        self.l33_ta_classes.setRange(2, 30)
        self.l33_ta_classes.setValue(7)
        form.addRow("TA classes", self.l33_ta_classes)
        self.l33_ustar_classes = QSpinBox()
        self.l33_ustar_classes.setRange(4, 60)
        self.l33_ustar_classes.setValue(20)
        form.addRow("USTAR classes", self.l33_ustar_classes)
        self.l33_fw = QSpinBox()
        self.l33_fw.setRange(1, 5)
        self.l33_fw.setValue(2)
        self.l33_fw.setToolTip("Forward-mode order: consecutive plateau bins required "
                               "(2 = Fw2, the ONEFlux/REddyProc default).")
        form.addRow("Forward-mode n", self.l33_fw)
        for w in (self.l33_ta_classes, self.l33_ustar_classes, self.l33_fw):
            w.setToolTip((w.toolTip() + "\n" if w.toolTip() else "")
                         + "Moving point method only.")
        for w in (self.l33_ta, self.l33_swin):
            w.currentTextChanged.connect(self._refresh_levels_info)
        self._l33_detect_box = gb
        return gb

    def _sync_l33_mode(self, *_) -> None:
        """Show only the active L3.3 mode's controls (declutter)."""
        on = self.l33_enable.isChecked()
        detect = self.l33_mode.currentIndex() == 1
        self.l33_mode.setEnabled(on)
        self._l33_constant_box.setVisible(on and not detect)
        self._l33_detect_box.setVisible(on and detect)
        self.l33_night.setVisible(on)
        # SW_IN is needed by the detection and by night-only filtering.
        self._l33_modeform.setRowVisible(
            self.l33_swin, on and (detect or self.l33_night.isChecked()))
        self._l33_modeform.setRowVisible(self.l33_ref, on)
        self._l33_modeform.setRowVisible(self.l33_ref_note, on and self.l33_ref.isChecked())
        # The stratification feeds the moving point method only.
        mp = self.l33_method.currentData() in USTAR_MP_MODES
        for w in (self.l33_ta_classes, self.l33_ustar_classes, self.l33_fw):
            w.setEnabled(mp)

    def _add_ustar(self) -> None:
        value = self.l33_value.value()
        label = ustar_scenario_labels("CUT", [self.l33_pct.value()])[0]
        for i, (_, lab) in enumerate(self._ustar):
            if lab == label:  # one threshold per percentile: replace
                self._ustar[i] = (value, label)
                break
        else:
            self._ustar.append((value, label))
        self._refill_ustar_list()
        self._update_run_label()

    def _remove_ustar(self) -> None:
        row = self.l33_list.currentRow()
        if row < 0:
            return
        del self._ustar[row]
        self._refill_ustar_list()
        self._update_run_label()

    def _refill_ustar_list(self) -> None:
        """Show the given thresholds; free labels from older projects as such."""
        self.l33_list.clear()
        legacy = False
        for value, label in self._ustar:
            if _label_percentile(label) is not None:
                self.l33_list.addItem(f"{label}:  {value:g} m s⁻¹")
            else:
                legacy = True
                self.l33_list.addItem(f"{value:g}  ({label or 'auto label'})")
        self.l33_legacy_note.setVisible(legacy)

    def _ustar_by_percentile(self) -> dict | None:
        """``{percentile: threshold}`` of the given thresholds, or None when one
        of them carries a free label (projects saved before the percentile form)."""
        out: dict = {}
        for value, label in self._ustar:
            p = _label_percentile(label)
            if p is None:
                return None
            out[int(p) if p.is_integer() else p] = value
        return out

    def _ref_on(self) -> bool:
        """True when L3.3 is on with 'Add REF (ONEFlux)'."""
        return self.l33_enable.isChecked() and self.l33_ref.isChecked()

    def _l33_problem(self) -> str:
        """Why `_level33_kwargs` returns None (for the status line)."""
        if not self.l33_enable.isChecked():
            return "Enable Level 3.3 and add at least one USTAR threshold."
        if self.l33_mode.currentIndex() == 1:
            return ("Pick TA and SW_IN columns and at least one percentile "
                    "for the threshold detection.")
        if not self._ustar:
            return "Add at least one USTAR threshold."
        if self.l33_ref.isChecked() and self._ustar_by_percentile() is None:
            return ("REF needs thresholds by percentile. Remove the thresholds "
                    "with their own labels (from an older project).")
        return "Pick an SW_IN column for nighttime-only filtering."

    def _level33_kwargs(self) -> dict | None:
        """L3.3 kwargs, or None when L3.3 is off / incompletely configured.

        Detect mode and given thresholds by percentile ->
        ``run_level33_ustar_detection`` kwargs (``thresholds={p: value}`` for
        given ones), tagged with a ``_detection`` marker the worker + codegen
        branch on. Thresholds with free labels from older projects ->
        ``run_level33_constant_ustar`` kwargs, as before.
        """
        if not self.l33_enable.isChecked():
            return None
        night = self.l33_night.isChecked()
        swin = self.l33_swin.currentText().strip()
        ref = self.l33_ref.isChecked()
        if self.l33_mode.currentIndex() == 1:  # detect
            ta = self.l33_ta.currentText().strip()
            pcts = self._l33_percentiles()
            if not ta or not swin or not pcts:
                return None
            mode = ("cut", "vut")[self.l33_apply.currentIndex()]
            method = self.l33_method.currentData()
            kwargs: dict = {
                "_detection": True,
                "ta_col": ta,
                "swin_col": swin,
                "detection_mode": method,
                "n_iter": self.l33_niter.value() or None,  # 0 = method default
                "n_jobs": self.l33_njobs.value(),
                "percentiles": pcts,
                "mode": mode,
            }
            if ref:
                kwargs["reference"] = True
            if night:
                kwargs["nighttime_only"] = True
            # Moving point knobs only when that method runs and they differ from
            # the library defaults (keeps the run + generated script minimal).
            mp: dict = {}
            if method in USTAR_MP_MODES:
                if self.l33_ta_classes.value() != 7:
                    mp["ta_classes_count"] = self.l33_ta_classes.value()
                if self.l33_ustar_classes.value() != 20:
                    mp["ustar_classes_count"] = self.l33_ustar_classes.value()
                if self.l33_fw.value() != 2:
                    mp["forward_mode_n"] = self.l33_fw.value()
            if mp:
                kwargs["detection_kwargs"] = {"mp_kwargs": mp}
            return kwargs
        # given
        if not self._ustar or (night and not swin):
            return None
        by_pct = self._ustar_by_percentile()
        if by_pct is not None:
            kwargs = {"_detection": True, "thresholds": by_pct}
            if ref:
                kwargs["reference"] = True
            if night:
                kwargs["nighttime_only"] = True
                kwargs["swin_col"] = swin
            return kwargs
        if ref:
            return None  # REF needs thresholds by percentile
        # Free labels from an older project: constant thresholds, as before.
        thresholds = [v for v, _ in self._ustar]
        labels = [lab for _, lab in self._ustar]
        kwargs = {"thresholds": thresholds}
        # All-or-nothing labels: pass them only if every scenario is labelled,
        # else let the library auto-generate CUT_0, CUT_1, …
        if all(labels):
            kwargs["threshold_labels"] = labels
        if night:
            kwargs["nighttime_only"] = True
            kwargs["swin_col"] = swin
        return kwargs

    def _l33_percentiles(self) -> tuple[int, ...]:
        """Parse the percentiles field into a tuple of ints (empty on bad input)."""
        out = []
        for tok in self.l33_percentiles.text().replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                p = int(round(float(tok)))
            except ValueError:
                return ()
            if 1 <= p <= 99:
                out.append(p)
        return tuple(out)

    def _level41_group(self) -> QGroupBox:
        box = QGroupBox("Level 4.1 — gap-filling (optional)")
        v = QVBoxLayout(box)
        note = QLabel("Tick the methods to run (additive — each replaces only its own "
                      "previous result). Only the enabled methods' settings are shown "
                      "below, each in its own section. Requires Level 3.3.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        v.addWidget(note)
        # Method toggles drive which setting sections are visible (declutter).
        self.l41_rf = QCheckBox("Random Forest (rf)")
        self.l41_xgb = QCheckBox("XGBoost (xgb)")
        self.l41_mds = QCheckBox("Marginal Distribution Sampling (mds)")
        for cb in (self.l41_rf, self.l41_xgb, self.l41_mds):
            cb.toggled.connect(self._update_run_label)
            cb.toggled.connect(self._sync_l41_visibility)
            v.addWidget(cb)
        self.l41_mds.toggled.connect(self._sync_l41_ref)
        # REF (set at L3.3): MDS only, then the REF selection.
        self.l41_ref_note = self._info_marker()
        v.addWidget(self.l41_ref_note)

        v.addWidget(self._l41_shared_box())
        v.addWidget(self._l41_rf_box_w())
        v.addWidget(self._l41_xgb_box_w())
        v.addWidget(self._l41_mds_box_w())

        # Which comparison to render after L4.1 (always visible).
        form2 = QFormLayout()
        self.l41_view = QComboBox()
        self.l41_view.addItems(["Cumulative comparison", "Gap-filled heatmaps"])
        form2.addRow("Comparison view", self.l41_view)
        v.addLayout(form2)
        v.addLayout(self._level_button_row(5, "Run Level 4.1"))
        self._sync_l41_visibility()
        self._sync_l41_ref()
        return box

    def _sync_l41_ref(self, *_) -> None:
        """With REF on, say what L4.1 runs and that the REF needs MDS."""
        if not hasattr(self, "l41_ref_note"):
            return  # the L3.3 controls fire before the L4.1 page exists
        ref = self._ref_on()
        if ref and self.l41_mds.isChecked():
            self._set_marker(
                self.l41_ref_note, True,
                "REF (ONEFlux) is on in Level 3.3: MDS gap-fills every u* scenario, "
                "then the REF is selected and added as CUT_REF / VUT_REF. Use MDS "
                "mode ONEFlux for the REF ONEFlux would select. Random Forest and "
                "XGBoost run after it, on the main scenarios and the REF scenario "
                "only (not on the other 39 REF versions).")
        elif ref:
            self._set_marker(self.l41_ref_note, False,
                             "REF (ONEFlux) is on in Level 3.3 and needs MDS: tick "
                             "Marginal Distribution Sampling.")
        self.l41_ref_note.setVisible(ref)

    def _l41_shared_box(self) -> QGroupBox:
        """RF/XGB predictors + shared settings (one feature set drives both)."""
        gb = QGroupBox("RF / XGB — predictors & settings")
        gv = QVBoxLayout(gb)
        feat_lbl = QLabel("Predictor features (filter, then tick to include):")
        feat_lbl.setWordWrap(True)
        gv.addWidget(feat_lbl)
        self.l41_features = FeaturePicker()
        gv.addWidget(self.l41_features)
        form = QFormLayout()
        # A fixed seed makes rf/xgb reproducible (without it sklearn/xgboost
        # reseed every run -> the output drifts run to run).
        self.l41_seed = QSpinBox()
        self.l41_seed.setRange(0, 2_000_000_000)
        self.l41_seed.setValue(42)
        self.l41_seed.setToolTip("Same seed -> identical gap-fill every run; "
                                 "change it to vary deliberately.")
        form.addRow("Random seed", self.l41_seed)
        gv.addLayout(form)
        self.l41_reduce = QCheckBox("Reduce features (SHAP selection)")
        self.l41_reduce.setToolTip("Drop low-importance engineered features via "
                                   "SHAP before the final fit (slower, often cleaner).")
        gv.addWidget(self.l41_reduce)
        self._l41_shared = gb
        return gb

    def _l41_rf_box_w(self) -> QGroupBox:
        gb = QGroupBox("Random Forest — hyperparameters")
        # Defaults match sklearn -> no behaviour change vs not passing them; only
        # what you edit is sent + rendered in the script.
        f = QFormLayout(gb)
        self.rf_n_est = self._ispin(100, 1, 5000)
        f.addRow("Trees (n_estimators)", self.rf_n_est)
        self.rf_max_depth = self._ispin(0, 0, 200)
        self.rf_max_depth.setSpecialValueText("None")  # 0 -> unlimited depth
        f.addRow("Max depth", self.rf_max_depth)
        self._l41_rf = gb
        return gb

    def _l41_xgb_box_w(self) -> QGroupBox:
        gb = QGroupBox("XGBoost — hyperparameters")
        f = QFormLayout(gb)
        self.xgb_n_est = self._ispin(100, 1, 5000)
        f.addRow("Trees (n_estimators)", self.xgb_n_est)
        self.xgb_max_depth = self._ispin(6, 1, 200)
        f.addRow("Max depth", self.xgb_max_depth)
        self.xgb_lr = self._dspin(0.30, 0.001, 1.0, 3)
        f.addRow("Learning rate", self.xgb_lr)
        self._l41_xgb = gb
        return gb

    def _l41_mds_box_w(self) -> QGroupBox:
        gb = QGroupBox("MDS — drivers & tolerances")
        gv = QVBoxLayout(gb)
        form = QFormLayout()
        self.mds_swin = QComboBox()
        self.mds_swin.currentTextChanged.connect(self._refresh_levels_info)
        form.addRow("SW_IN (W m⁻²)", self.mds_swin)
        self.mds_ta = QComboBox()
        self.mds_ta.currentTextChanged.connect(self._refresh_levels_info)
        form.addRow("TA (°C)", self.mds_ta)
        self.mds_vpd = QComboBox()
        self.mds_vpd.currentTextChanged.connect(self._refresh_levels_info)
        form.addRow("VPD (kPa)", self.mds_vpd)
        # Optional per-driver QC flag ("(none)" = no driver QC, the default).
        self._mds_qc: dict[str, QComboBox] = {}
        for key, label, driver in (("swin", "SW_IN QC flag", self.mds_swin),
                                   ("ta", "TA QC flag", self.mds_ta),
                                   ("vpd", "VPD QC flag", self.mds_vpd)):
            combo = QComboBox()
            combo.setToolTip(QC_TIP)
            driver.currentTextChanged.connect(self._refresh_mds_qc_items)
            form.addRow(label, combo)
            self._mds_qc[key] = combo
        # MDS similarity tolerances (Reichstein et al. 2005 defaults).
        self.mds_ta_tol = self._dspin(2.5, 0.1, 20.0, 2)
        form.addRow("TA tolerance (°C)", self.mds_ta_tol)
        self.mds_vpd_tol = self._dspin(0.5, 0.05, 10.0, 2)
        form.addRow("VPD tolerance (kPa)", self.mds_vpd_tol)
        self.mds_mode = make_mode_combo()
        self.mds_mode.setToolTip(
            (self.mds_mode.toolTip() + "\n\n" if self.mds_mode.toolTip() else "")
            + "With REF (ONEFlux) on in Level 3.3, mode ONEFlux gives the REF "
              "ONEFlux would select.")
        form.addRow("Mode", self.mds_mode)
        # Follows the mode (the library default) until the user clicks it.
        self._mds_fill_user: bool | None = None
        self.mds_fill = QCheckBox("Fill long gaps")
        self.mds_fill.setToolTip(
            "Off: runs of poorest-quality fills longer than 15 days are set missing "
            "again, except for 15 days next to measured data, as in the ONEFlux "
            "FLUXNET product.\nOn: every gap MDS can fill stays filled.\n\n"
            "Default: off in mode ONEFlux, on in the other modes. The box follows "
            "the mode until you change it.")
        self.mds_fill.clicked.connect(self._on_mds_fill_clicked)
        self.mds_mode.currentIndexChanged.connect(self._sync_mds_fill)
        form.addRow(self.mds_fill)
        self._sync_mds_fill()
        gv.addLayout(form)
        self.mds_mark = self._info_marker()
        gv.addWidget(self.mds_mark)
        vpd_note = QLabel("Units: VPD in kPa, TA in °C.")
        vpd_note.setWordWrap(True)
        vpd_note.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        gv.addWidget(vpd_note)
        self._l41_mds = gb
        return gb

    def _mds_fill_default(self) -> bool:
        """``fill_long_gaps`` of the library for the selected MDS mode (None)."""
        return mode_value(self.mds_mode) != "oneflux"

    def _on_mds_fill_clicked(self, checked: bool) -> None:
        self._mds_fill_user = bool(checked)

    def _sync_mds_fill(self, *_) -> None:
        """Show the user's choice, or the mode default when there is none."""
        fill = self._mds_fill_default() if self._mds_fill_user is None \
            else self._mds_fill_user
        self.mds_fill.setChecked(fill)

    def _refresh_mds_qc_items(self, *_) -> None:
        """Refill the MDS QC combos, listing each driver's flag columns first."""
        cols = [str(c) for c in self._df.columns] if self._df is not None else []
        for key, driver in (("swin", self.mds_swin), ("ta", self.mds_ta),
                            ("vpd", self.mds_vpd)):
            fill_qc_combo(self._mds_qc[key], cols, driver.currentText())

    def _sync_l41_visibility(self, *_) -> None:
        """Show each method's settings section only when that method is enabled."""
        self._l41_shared.setVisible(self.l41_rf.isChecked() or self.l41_xgb.isChecked())
        self._l41_rf.setVisible(self.l41_rf.isChecked())
        self._l41_xgb.setVisible(self.l41_xgb.isChecked())
        self._l41_mds.setVisible(self.l41_mds.isChecked())

    @staticmethod
    def _ispin(value, lo, hi) -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(value)
        return sp

    def _level41_methods(self) -> list[str]:
        """Selected L4.1 methods in canonical order (matches gapfilled_cols())."""
        out = []
        for key, cb in (("mds", self.l41_mds), ("rf", self.l41_rf), ("xgb", self.l41_xgb)):
            if cb.isChecked():
                out.append(key)
        return out

    def _level41_cfg(self) -> dict | None:
        """run/codegen config for L4.1, or None when no method is selected.

        Hyperparameters whose value still equals the library default are omitted
        (keeps the run identical and the generated script minimal); the random
        seed is always sent because pinning it is the whole point — it makes the
        rf/xgb output reproducible across runs.
        """
        methods = self._level41_methods()
        if not methods:
            return None
        cfg: dict = {"methods": methods}
        if "rf" in methods or "xgb" in methods:
            cfg["features"] = self.l41_features.selected()
            if self.l41_reduce.isChecked():
                cfg["reduce_features"] = True
            seed = self.l41_seed.value()
            if "rf" in methods:
                rf = {"random_state": seed}
                if self.rf_n_est.value() != 100:        # sklearn default
                    rf["n_estimators"] = self.rf_n_est.value()
                if self.rf_max_depth.value() != 0:       # 0 == None (unlimited)
                    rf["max_depth"] = self.rf_max_depth.value()
                cfg["rf_kwargs"] = rf
            if "xgb" in methods:
                xgb = {"random_state": seed}
                if self.xgb_n_est.value() != 100:        # xgboost default
                    xgb["n_estimators"] = self.xgb_n_est.value()
                if self.xgb_max_depth.value() != 6:      # xgboost default
                    xgb["max_depth"] = self.xgb_max_depth.value()
                if abs(self.xgb_lr.value() - 0.30) > 1e-9:  # xgboost default
                    xgb["learning_rate"] = self.xgb_lr.value()
                cfg["xgb_kwargs"] = xgb
        if "mds" in methods:
            mds = {"swin": self.mds_swin.currentText(),
                   "ta": self.mds_ta.currentText(),
                   "vpd": self.mds_vpd.currentText()}
            if abs(self.mds_ta_tol.value() - 2.5) > 1e-9:   # run_level41_mds default
                mds["ta_tol"] = self.mds_ta_tol.value()
            if abs(self.mds_vpd_tol.value() - 0.5) > 1e-9:  # run_level41_mds default
                mds["vpd_tol"] = self.mds_vpd_tol.value()
            # Always written: the default changed from the plain mean to 'diive',
            # so the script states which one it used.
            mds["mode"] = mode_value(self.mds_mode)
            # Only when it differs from the mode default (the library's None).
            if self.mds_fill.isChecked() != self._mds_fill_default():
                mds["fill_long_gaps"] = self.mds_fill.isChecked()
            for key, combo in self._mds_qc.items():  # only the flags that are set
                col = qc_value(combo)
                if col is not None:
                    mds[f"{key}_qc"] = col
            cfg["mds"] = mds
            if self._ref_on():
                cfg["ustar_reference"] = True  # run_level41_ustar_reference after MDS
        return cfg

    def _level42_group(self) -> QGroupBox:
        box = QGroupBox("Level 4.2: NEE partitioning (optional)")
        v = QVBoxLayout(box)
        self.l42_enable = QCheckBox("Partition NEE into GPP and RECO")
        self.l42_enable.setToolTip(
            "Split the NEE of each u* scenario into gross primary production (GPP) "
            "and ecosystem respiration (RECO). Requires Level 4.1.")
        v.addWidget(self.l42_enable)
        note = QLabel("One partitioning per u* scenario of Level 3.3 (with REF on: the "
                      "main scenarios and the REF scenario). Each method and mode writes "
                      "its own columns, so runs with other settings add up.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        v.addWidget(note)

        self._l42_box = QWidget()
        form = QFormLayout(self._l42_box)
        form.setContentsMargins(0, 0, 0, 0)
        self.l42_method = make_method_combo()
        form.addRow("Method", self.l42_method)
        self.l42_mode = make_partition_mode_combo()
        form.addRow("Mode", self.l42_mode)
        self._l42_cols: dict[str, QComboBox] = {}
        for arg, label in _L42_DRIVERS:
            combo = QComboBox()
            combo.currentTextChanged.connect(self._refresh_levels_info)
            form.addRow(label, combo)
            self._l42_cols[arg] = combo
        self.l42_gapfill = QComboBox()
        self.l42_gapfill.addItems(["mds", "rf", "xgb"])
        self.l42_gapfill.setToolTip(
            "Which Level 4.1 gap-filled NEE the partitioning uses. The nighttime "
            "method needs it for GPP. Daytime ONEFlux uses it for windows without "
            "measured nighttime NEE. The method must have run in Level 4.1.")
        form.addRow("Gap-filled NEE from", self.l42_gapfill)
        self.l42_mark = self._info_marker()
        form.addRow(self.l42_mark)
        units = QLabel("Units: TA in °C, SW_IN in W m⁻², VPD in kPa.")
        units.setWordWrap(True)
        units.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        form.addRow(units)
        self.l42_ref = QCheckBox("Add GPP / RECO REF (ONEFlux)")
        self.l42_ref.setToolTip(_REF42_TIP)
        form.addRow(self.l42_ref)
        self.l42_ref_note = self._info_marker()
        form.addRow(self.l42_ref_note)
        self._l42_form = form
        v.addWidget(self._l42_box)
        v.addLayout(self._level_button_row(6, "Run Level 4.2"))

        self.l42_enable.toggled.connect(self._sync_l42)
        self.l42_enable.toggled.connect(self._update_run_label)
        for combo in (self.l42_method, self.l42_mode):
            combo.currentIndexChanged.connect(self._sync_l42)
            combo.currentIndexChanged.connect(self._update_run_label)
        self.l42_ref.toggled.connect(self._update_run_label)
        # The REF needs the gap-filled NEE from MDS (only MDS fills the 40 versions).
        self.l42_gapfill.currentIndexChanged.connect(self._sync_l42)
        self.l42_gapfill.currentIndexChanged.connect(self._update_run_label)
        # The REF option follows the NEE REF set in Level 3.3 and Level 4.1.
        for cb in (self.l33_enable, self.l33_ref, self.l41_mds):
            cb.toggled.connect(self._sync_l42)
        self._sync_l42()
        return box

    def _l42_variant(self) -> str:
        """The L4.2 variant token of the selected method and mode ('nt_of', ...)."""
        return partition_variant(combo_value(self.l42_method), combo_value(self.l42_mode))

    def _l42_drivers(self) -> tuple[str, ...]:
        """The driver arguments the selected method and mode need, in row order."""
        inputs = partitioning_inputs(combo_value(self.l42_method), combo_value(self.l42_mode))
        return tuple(_L42_DRIVER_ARGS[k] for k in inputs if k in _L42_DRIVER_ARGS)

    def _l42_uses_nee_f(self) -> bool:
        """True when the selected variant reads an L4.1 gap-filled NEE."""
        m, mo = combo_value(self.l42_method), combo_value(self.l42_mode)
        return "nee_f" in partitioning_inputs(m, mo) + partitioning_optional_inputs(m, mo)

    def _l42_ref_problem(self) -> str | None:
        """Why the GPP / RECO REF cannot run with the current settings, or None.

        It needs the NEE REF (L3.3 REF on and L4.1 MDS ticked). A variant that
        reads an L4.1 gap-filled NEE must take it from MDS, because only MDS
        fills the 40 REF versions."""
        if not (self._ref_on() and self.l41_mds.isChecked()):
            return ("Needs the NEE REF: tick 'Add REF (ONEFlux)' in Level 3.3 and "
                    "Marginal Distribution Sampling in Level 4.1.")
        if self._l42_uses_nee_f() and self.l42_gapfill.currentText() != "mds":
            return (f"Needs the gap-filled NEE from mds. Only MDS fills the 40 REF "
                    f"versions, not {self.l42_gapfill.currentText()}. Pick mds in "
                    f"'Gap-filled NEE from'.")
        return None

    def _l42_ref_allowed(self) -> bool:
        """True when the GPP / RECO REF can run (see ``_l42_ref_problem``)."""
        return self._l42_ref_problem() is None

    def _sync_l42(self, *_) -> None:
        """Show the controls of the selected variant and gate the REF option."""
        if not hasattr(self, "l42_ref_note"):
            return  # fired while the page is built
        self._l42_box.setVisible(self.l42_enable.isChecked())
        needed = set(self._l42_drivers())
        for arg, combo in self._l42_cols.items():
            self._l42_form.setRowVisible(combo, arg in needed)
        self._l42_form.setRowVisible(self.l42_gapfill, self._l42_uses_nee_f())
        problem = self._l42_ref_problem()
        allowed = problem is None
        self.l42_ref.setEnabled(allowed)
        if allowed:
            tag = self._l42_variant().upper()
            text = (f"Adds GPP_{tag}_CUT_REF and RECO_{tag}_CUT_REF (VUT_REF with VUT "
                    "thresholds). Partitions the 40 REF versions and the scenario at "
                    "percentile 50 again, so the run takes about 41 times longer.")
        else:
            text = problem
        self._set_marker(self.l42_ref_note, allowed, text)
        self._refresh_levels_info()

    def _level42_cfg(self) -> dict | None:
        """run/codegen config for L4.2 (``level42_cfg``), or None when L4.2 is off.

        Holds the method, mode, the drivers that variant needs, the L4.1
        gap-filled NEE where the variant reads one, and ``ustar_reference``
        when the GPP / RECO REF is ticked and allowed."""
        if not hasattr(self, "l42_enable") or not self.l42_enable.isChecked():
            return None
        cfg: dict = {"method": combo_value(self.l42_method),
                     "mode": combo_value(self.l42_mode)}
        for arg in self._l42_drivers():
            cfg[arg] = self._l42_cols[arg].currentText().strip()
        if self._l42_uses_nee_f():
            cfg["gapfill_method"] = self.l42_gapfill.currentText()
        if self.l42_ref.isChecked() and self._l42_ref_allowed():
            cfg["ustar_reference"] = True
        return cfg

    def _l42_problem(self, l42: dict, l41_methods) -> str | None:
        """Why an L4.2 run cannot start (missing drivers, an L4.1 gap-filled NEE
        from a method that did not run, or the GPP / RECO REF with a gap-filled
        NEE that is not from MDS), or None."""
        missing = [arg for arg in self._l42_drivers() if not l42.get(arg)]
        if missing:
            return ("Pick the Level 4.2 driver columns: " + ", ".join(missing) + ".")
        method = l42.get("gapfill_method")  # set only for variants that read one
        if "gapfill_method" in l42 and method not in l41_methods:
            if l42["method"] == "nighttime":
                return (f"The nighttime method needs the gap-filled NEE of Level 4.1 "
                        f"method '{method}'. Run it in Level 4.1, or pick a method "
                        f"that ran.")
            # Daytime ONEFlux: the library would drop it with a console warning.
            return (f"Daytime ONEFlux uses the gap-filled NEE of Level 4.1 method "
                    f"'{method}', which did not run. Run it in Level 4.1, or pick a "
                    f"method that ran.")
        if l42.get("ustar_reference") and method not in (None, "mds"):
            return (f"The GPP / RECO REF needs the gap-filled NEE from mds. Only MDS "
                    f"fills the 40 REF versions, not {method}.")
        return None

    def _update_run_label(self, *_) -> None:
        # Reflects intent: L3.3 configured -> "3.3" even before an L3.2 step exists
        # (the run then surfaces the "needs an L3.2 test" requirement).
        if self._level42_cfg():
            deepest = "4.2"
        elif self._level41_methods():
            deepest = "4.1"
        elif self.l33_enable.isChecked() and self._level33_kwargs():
            deepest = "3.3"
        elif self._steps:
            deepest = "3.2"
        else:
            deepest = "3.1"
        self.run_btn.setText(f"Run through Level {deepest}")
        self._refresh_rail()

    def _apply_tooltips(self) -> None:
        """Tooltip each control with its library parameter docstring."""
        from diive.core.utils.docstrings import param_docs
        from diive.flux.postprocessingchain import FluxConfig
        docs: dict = {}
        for src in (init_flux_data, FluxConfig, run_level2, run_level31):
            docs.update(param_docs(src))
        for param, widget in (
            ("fluxcol", self.fluxcol), ("site_lat", self.site_lat),
            ("site_lon", self.site_lon),
            ("nighttime_threshold", self.nighttime_threshold),
            ("daytime_accept_qcf_below", self.day_qcf),
            ("nighttime_accept_qcf_below", self.night_qcf),
            ("signal_strength_col", self.signal_strength_col),
            ("gapfill_storage_term", self.l31_gapfill),
            ("set_storage_to_zero", self.l31_zero),
            ("strgcol", self.strgcol),
        ):
            tip = docs.get(param)
            if tip:
                widget.setToolTip(tip)
        for key, cb in self.l2_checks.items():
            tip = docs.get(key)
            if tip:
                cb.setToolTip(tip)
                # Stash the doc tip so the availability refresh can re-attach it
                # (it overrides the tooltip with a "missing inputs" note when off).
                self._l2_tips[key] = tip

    @staticmethod
    def _dspin(value, lo, hi, decimals) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setValue(value)
        return sp

    # --- state ---
    def _fx_controls(self) -> dict:
        # No "site_lat"/"site_lon"/"utc_offset": they come from Project settings,
        # so values saved by an older project are ignored on restore.
        return {"fluxcol": self.fluxcol, "ustarcol": self.ustarcol,
                "nighttime_threshold": self.nighttime_threshold,
                "day_qcf": self.day_qcf, "night_qcf": self.night_qcf,
                "signal_strength_col": self.signal_strength_col,
                "ss_method": self.ss_method, "ss_threshold": self.ss_threshold,
                "l31_gapfill": self.l31_gapfill, "l31_zero": self.l31_zero,
                "strgcol": self.strgcol,
                "l33_mode": self.l33_mode, "l33_apply": self.l33_apply,
                "l33_night": self.l33_night, "l33_ref": self.l33_ref,
                "l33_ta": self.l33_ta,
                "l33_swin": self.l33_swin, "l33_niter": self.l33_niter,
                "l33_njobs": self.l33_njobs, "l33_percentiles": self.l33_percentiles,
                "l33_ta_classes": self.l33_ta_classes,
                "l33_ustar_classes": self.l33_ustar_classes, "l33_fw": self.l33_fw,
                "l41_rf": self.l41_rf, "l41_xgb": self.l41_xgb, "l41_mds": self.l41_mds,
                "mds_swin": self.mds_swin, "mds_ta": self.mds_ta, "mds_vpd": self.mds_vpd,
                "l41_view": self.l41_view,
                "l41_seed": self.l41_seed, "l41_reduce": self.l41_reduce,
                "rf_n_est": self.rf_n_est, "rf_max_depth": self.rf_max_depth,
                "xgb_n_est": self.xgb_n_est, "xgb_max_depth": self.xgb_max_depth,
                "xgb_lr": self.xgb_lr,
                "mds_ta_tol": self.mds_ta_tol, "mds_vpd_tol": self.mds_vpd_tol,
                # After the drivers, which reorder these combos' items.
                **{f"mds_{k}_qc": c for k, c in self._mds_qc.items()},
                "l42_enable": self.l42_enable, "l42_gapfill": self.l42_gapfill,
                "l42_ref": self.l42_ref,
                **{f"l42_{arg}": c for arg, c in self._l42_cols.items()}}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        controls = save_controls(self._fx_controls())
        controls["mds_mode"] = mode_value(self.mds_mode)  # the value, not the label
        controls["l33_method"] = self.l33_method.currentData()  # likewise
        controls["mds_fill_long_gaps"] = self._mds_fill_user  # None = follow the mode
        controls["l42_method"] = combo_value(self.l42_method)
        controls["l42_mode"] = combo_value(self.l42_mode)
        return {"controls": controls,
                "l2": {k: cb.isChecked() for k, cb in self.l2_checks.items()},
                "l2_vm97": {k: cb.isChecked() for k, cb in self.l2_vm97_checks.items()},
                "l2_cols": {k: [c.currentText() for c in combos]
                            for k, combos in self.l2_cols.items()},
                "l32_steps": self._steps,
                "l33_enabled": self.l33_enable.isChecked(),
                "l33_ustar": [list(u) for u in self._ustar],
                "l41_features": self.l41_features.selected()}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        saved = state.get("controls") or {}
        restore_controls(self._fx_controls(), saved)
        # Projects saved before the QC option (or naming a vanished column) -> none.
        for k, combo in self._mds_qc.items():
            set_qc_value(combo, saved.get(f"mds_{k}_qc"))
        # The chain offered no averaging choice before the mode option, so older
        # projects were filled with the plain mean: restore them to 'oneflux'.
        set_mode_value(self.mds_mode,
                       saved.get("mds_mode", "oneflux" if saved else None))
        # Projects saved before the option follow the mode (the library default).
        fill = saved.get("mds_fill_long_gaps")
        self._mds_fill_user = None if fill is None else bool(fill)
        self._sync_mds_fill()
        # Before the u* method choice the detection was the moving point bootstrap.
        set_ustar_method(self.l33_method,
                         saved.get("l33_method", "oneflux_mp" if saved else None))
        set_combo_value(self.l42_method, saved.get("l42_method"))
        set_combo_value(self.l42_mode, saved.get("l42_mode"))
        if str(saved.get("l33_mode", "")).startswith("Detect"):
            self.l33_mode.setCurrentIndex(1)  # older label "Detect (moving point, ...)"
        elif str(saved.get("l33_mode", "")).startswith("Constant"):
            self.l33_mode.setCurrentIndex(0)  # older label of "Given thresholds"
        # Older labels "CUT (constant)" / "VUT (per-year)".
        if str(saved.get("l33_apply", "")).startswith("VUT"):
            self.l33_apply.setCurrentIndex(1)
        elif str(saved.get("l33_apply", "")).startswith("CUT"):
            self.l33_apply.setCurrentIndex(0)
        for k, val in (state.get("l2") or {}).items():
            if k in self.l2_checks:
                self.l2_checks[k].setChecked(bool(val))
        for k, val in (state.get("l2_vm97") or {}).items():
            if k in self.l2_vm97_checks:
                self.l2_vm97_checks[k].setChecked(bool(val))
        # Restore per-test column picks (combos seeded by restore_controls' fluxcol).
        for k, picks in (state.get("l2_cols") or {}).items():
            for combo, pick in zip(self.l2_cols.get(k, []), picks, strict=False):
                if pick and combo.findText(pick) >= 0:
                    combo.setCurrentText(pick)
        # Rebuild the L3.2 chain (a list of {method, kwargs} steps).
        self._steps = list(state.get("l32_steps") or [])
        self.l32_steps_list.clear()
        for step in self._steps:
            self.l32_steps_list.addItem(self._step_summary(step))
        # Rebuild the L3.3 USTAR thresholds. Same [value, label] pairs as before
        # the percentile form, so older projects restore unchanged.
        self.l33_enable.setChecked(bool(state.get("l33_enabled")))
        self._ustar = [(float(v), str(lab)) for v, lab in (state.get("l33_ustar") or [])]
        self._refill_ustar_list()
        # Re-check the saved L4.1 features (the picker was filled by on_data_loaded).
        self.l41_features.set_selected(state.get("l41_features") or [])
        self._refresh_l2_availability()
        self._refresh_levels_info()
        self._sync_l33_mode()
        self._sync_l41_ref()
        self._sync_l42()
        self._update_run_label()

    # --- data ---
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        # New dataset: the chain output and the run progress no longer apply.
        self._data = None
        self._l41_data = None
        self._reached = -1
        cols = [str(c) for c in df.columns]
        self.fluxcol.clear()
        self.fluxcol.addItems(cols)
        # Default to a common flux column if present.
        for cand in ("FC", "LE", "H", "NEE", "FC_orig"):
            if cand in cols:
                self.fluxcol.setCurrentText(cand)
                break
        # USTAR column (init_flux_data requires it) — default to 'USTAR' if present.
        self.ustarcol.clear()
        self.ustarcol.addItems(cols)
        for cand in ("USTAR", "ustar", "u*"):
            if cand in cols:
                self.ustarcol.setCurrentText(cand)
                break
        self.signal_strength_col.clear()
        self.signal_strength_col.addItems([""] + cols)
        for c in cols:
            if "SIGNAL_STRENGTH" in c.upper():
                self.signal_strength_col.setCurrentText(c)
                break
        # Storage-term column ("" = auto-detect from FLUXNET/EddyPro naming).
        self.strgcol.clear()
        self.strgcol.addItems([""] + cols)
        # L4.1: feature picker + MDS driver combos. Auto-pick obvious drivers.
        # L3.3 detection shares the TA / SW_IN auto-pick rules.
        self.l41_features.set_columns(cols, keep_selection=False)
        for combo, needle, avoid in (
            (self.mds_swin, "SW_IN", "POT"),
            (self.mds_ta, "TA", None),
            (self.mds_vpd, "VPD", None),
            (self.l33_swin, "SW_IN", "POT"),
            (self.l33_ta, "TA", None),
            (self._l42_cols["ta"], "TA", None),
            (self._l42_cols["sw_in"], "SW_IN", "POT"),
            (self._l42_cols["ta_f"], "TA", None),
            (self._l42_cols["sw_in_f"], "SW_IN", "POT"),
            (self._l42_cols["vpd_f"], "VPD", None),
        ):
            combo.clear()
            combo.addItems(cols)
            for c in cols:
                up = c.upper()
                if up.startswith("FLAG"):
                    continue  # FLAG_VPD_..._ISFILLED etc. are not driver columns
                # The needle must start a name part (TA_1_1_1, Tair_f), so TA does
                # not match inside USTAR or FC_STAGE1 (L394).
                if f"_{needle}" in f"_{up}" and (avoid is None or avoid not in up):
                    combo.setCurrentText(c)
                    break
        self._refresh_mds_qc_items()  # kept only if the column is still there
        # New dataset: the previous run's reach no longer applies.
        self.rail.set_reached_through(-1)
        self._populate_l2_cols()  # seed the per-test column pickers + availability
        self._refresh_levels_info()  # storage / USTAR / MDS availability markers
        self._refresh_rail()
        self._update_level_buttons()

    # --- site (mirrored from Project settings) ---
    @staticmethod
    def _site_coords() -> tuple[float, float, int]:
        """Latitude, longitude and UTC offset from Project settings.

        Read from ``site.manager``, not from the spin boxes, which round to four
        decimals for display."""
        m = site.manager
        return float(m.latitude), float(m.longitude), project_utc_offset()

    def _seed_site(self) -> None:
        """Show the project's coordinates, or say that the site is not set."""
        m = site.manager
        if m.configured:
            self.site_lat.setValue(m.latitude)
            self.site_lon.setValue(m.longitude)
            text, color = "From Settings ▸ Project settings. Change them there.", _C_MUTED
        else:
            text = ("<b>No site configured.</b> Set latitude, longitude and UTC "
                    "offset in Settings ▸ Project settings.")
            color = theme.manager.tokens.get("DANGER_BG", "#E04646")
        self.site_note.setText(text)
        self.site_note.setStyleSheet(f"QLabel {{ color: {color}; }}"
                                     + theme.manager.tooltip_qss())

    def _site_missing(self) -> bool:
        """Refuse to run without a configured site, and say why.

        Every level depends on SW_IN_POT and the day/night split computed at
        initialization, so an unguarded run would silently process the flux at
        latitude 0, longitude 0 on UTC."""
        if site.manager.configured:
            return False
        self.summary.setPlainText(_SITE_NOT_SET)
        return True

    def _results_stale(self) -> bool:
        """True when the chain output was initialized with other coordinates or
        another UTC offset than the project's current ones (``data.meta`` keeps
        the values ``init_flux_data`` used)."""
        if self._data is None:
            return False
        meta = self._data.meta
        used = (meta.site_lat, meta.site_lon, meta.utc_offset)
        return not site.manager.configured or used != self._site_coords()

    def _on_project_settings_changed(self) -> None:
        """Mirror the new settings. Results initialized with the old coordinates
        or offset stay visible but can no longer be added to the dataset."""
        self._seed_site()
        if self._running:
            return  # _finalize checks the finished run against the new settings
        self._update_level_buttons()
        if self._results_stale():
            self.summary.setPlainText(_SITE_CHANGED)

    # --- collect kwargs the library wants ---
    def _init_kwargs(self) -> dict:
        lat, lon, utc = self._site_coords()
        return dict(
            fluxcol=self.fluxcol.currentText(),
            ustarcol=self.ustarcol.currentText() or "USTAR",
            site_lat=lat,
            site_lon=lon,
            utc_offset=utc,
            nighttime_threshold=self.nighttime_threshold.value(),
            daytime_accept_qcf_below=int(self.day_qcf.currentText()),
            nighttime_accept_qcf_below=int(self.night_qcf.currentText()),
        )

    def _level2_settings(self) -> dict:
        basevar = self._l2_basevar()
        info = level2_test_inputs(self.fluxcol.currentText(), basevar) if basevar else {}
        settings: dict = {}
        for key, cb in self.l2_checks.items():
            # Skip unchecked OR disabled-because-no-valid-column tests.
            if not cb.isChecked() or not cb.isEnabled():
                continue
            if key == "ssitc":
                cfg = {"apply": True, "setflag_timeperiod": None}
            elif key == "raw_data_screening_vm97":
                # The library requires all eight sub-keys present (True/False).
                cfg = {"apply": True,
                       **{k: c.isChecked() for k, c in self.l2_vm97_checks.items()}}
            elif key == "signal_strength":
                col = self.signal_strength_col.currentText().strip()
                if not col:
                    continue  # can't run without a column
                settings[key] = {"apply": True, "signal_strength_col": col,
                                 "method": self.ss_method.currentText(),
                                 "threshold": self.ss_threshold.value()}
                continue
            else:
                cfg = {"apply": True}
            # Attach a column override only when the pick differs from the
            # standard templated name (keeps the run/codegen minimal).
            self._attach_l2_col_override(cfg, key, info.get(key, {}).get("inputs", []))
            settings[key] = cfg
        return settings

    def _attach_l2_col_override(self, cfg: dict, key: str, defaults: list[str]) -> None:
        chosen = [c.currentText().strip() for c in self.l2_cols.get(key, [])]
        if key == "gas_completeness":
            keys = ["expect_nr_col", "basevar_nr_col"]
            for i, slot in enumerate(keys):
                pick = chosen[i] if i < len(chosen) else ""
                default = defaults[i] if i < len(defaults) else ""
                if pick and pick != default:
                    cfg[slot] = pick
        else:
            pick = chosen[0] if chosen else ""
            default = defaults[0] if defaults else ""
            if pick and pick != default:
                cfg["col"] = pick

    def _level31_kwargs(self) -> dict:
        kwargs = dict(
            gapfill_storage_term=self.l31_gapfill.isChecked(),
            set_storage_to_zero=self.l31_zero.isChecked(),
        )
        strgcol = self.strgcol.currentText().strip()
        if strgcol:
            kwargs["strgcol"] = strgcol
        return kwargs

    # --- run (library work on a worker thread) ---
    def _compute(self, df, init_kwargs: dict, level2_settings: dict, level31_kwargs: dict,
                 level32_steps: list[dict] | None = None,
                 level33_kwargs: dict | None = None,
                 level41_cfg: dict | None = None):
        """Init container, run L2 → 3.1 → (opt) 3.2 → (opt) 3.3 → (opt) 4.1.

        L4.2 runs in ``_worker``, which keeps the L4.1 container for the GUI thread."""
        work = df.drop(columns=[c for c in _RESERVED if c in df.columns])
        data = init_flux_data(df=work, **init_kwargs)
        data = run_level2(data, **level2_settings)
        data = run_level31(data, **level31_kwargs)
        if level32_steps:
            # L3.2 is a stateful chain: build the detector wired to the current
            # data, commit each outlier test, then aggregate into the level QCF.
            data, sod = make_level32_detector(data)
            for step in level32_steps:
                getattr(sod, step["method"])(**step.get("kwargs", {}))
                sod.addflag()
            data = run_level32(data, outlier_detector=sod)
        if level33_kwargs:
            # showplot off — a worker thread must not pop a matplotlib window.
            data = self._run_level33(data, level33_kwargs)
        if level41_cfg:
            data = self._run_level41(data, level41_cfg)
        return data

    @staticmethod
    def _run_level33(data, kwargs: dict):
        """Run L3.3 — constant thresholds or moving-point detection (showplot off)."""
        if kwargs.get("_detection"):
            kw = {k: v for k, v in kwargs.items() if k != "_detection"}
            return run_level33_ustar_detection(data, showplot=False, verbose=False, **kw)
        return run_level33_constant_ustar(data, showplot=False, verbose=False, **kwargs)

    @staticmethod
    def _run_level41(data, cfg: dict):
        """Run the selected L4.1 gap-filling methods (additive across methods).

        With REF: MDS -> REF -> rf -> xgb. rf / xgb leave out the 40 REF
        versions but take the scenario the REF maps to, once it is selected."""
        methods = cfg["methods"]

        def ml(data):
            if "rf" in methods or "xgb" in methods:
                # One engineer, reused across rf/xgb (feature engineering runs once).
                features = cfg["features"]
                engineer = make_level41_engineer(data, features=features)
                reduce = cfg.get("reduce_features", False)
                if "rf" in methods:
                    data = run_level41_rf(data, features=features, engineer=engineer,
                                          reduce_features=reduce, **cfg.get("rf_kwargs", {}))
                if "xgb" in methods:
                    data = run_level41_xgb(data, features=features, engineer=engineer,
                                           reduce_features=reduce, **cfg.get("xgb_kwargs", {}))
            return data

        with_ref = "mds" in methods and cfg.get("ustar_reference")
        if not with_ref:
            data = ml(data)
        if "mds" in methods:
            mds = cfg["mds"]
            extra = {k: mds[k] for k in ("ta_tol", "vpd_tol", "mode", "fill_long_gaps",
                                         "swin_qc", "ta_qc", "vpd_qc")
                     if k in mds}
            data = run_level41_mds(data, swin=mds["swin"], ta=mds["ta"], vpd=mds["vpd"],
                                   **extra)
        if with_ref:
            data = run_level41_ustar_reference(data, verbose=False)
            data = ml(data)
        return data

    @staticmethod
    def _run_level42(data, cfg: dict):
        """Run the L4.2 partitioning, then the GPP / RECO REF of the same variant
        with the same drivers when ``cfg['ustar_reference']`` is set."""
        kwargs = {k: v for k, v in cfg.items() if k != "ustar_reference"}
        data = run_level42_partitioning(data, **kwargs)
        if cfg.get("ustar_reference"):
            # verbose=1: warnings reach the console, e.g. versions partitioned
            # without a gap-filled NEE or failed fits.
            data = run_level42_ustar_reference(data, verbose=1, **kwargs)
        return data

    def _run(self) -> None:
        if self._df is None:
            return
        ik, l2, l31 = self._init_kwargs(), self._level2_settings(), self._level31_kwargs()
        if not l2:
            self.summary.setPlainText("Enable at least one Level 2 test.")
            return
        steps = list(self._steps)
        l33 = self._level33_kwargs()
        l41 = self._level41_cfg()
        if l41 and not l33:
            # L4.1 gap-fills the L3.3 (per-scenario USTAR) flux (library raises
            # without it). Surface the requirement up front.
            msg = ("Level 4.1 (gap-filling) requires Level 3.3 (USTAR). Enable Level "
                   "3.3 with a threshold, or disable the Level 4.1 methods.")
            if self.l33_enable.isChecked():
                msg += "\n" + self._l33_problem()
            self.summary.setPlainText(msg)
            return
        if l33 and not steps:
            # USTAR filtering must operate on outlier-screened data (library raises).
            self.summary.setPlainText(
                "Level 3.3 (USTAR) requires at least one Level 3.2 outlier test. "
                "Add a step or disable Level 3.3.")
            return
        if l41 and self._ref_needs_mds(l41):
            return
        if l41 and ("rf" in l41["methods"] or "xgb" in l41["methods"]) \
                and not l41.get("features"):
            self.summary.setPlainText(
                "Random Forest / XGBoost need at least one feature. Select "
                "predictor columns in the Level 4.1 feature list.")
            return
        l42 = self._level42_cfg()
        if l42 and not l41:
            self.summary.setPlainText(
                "Level 4.2 (partitioning) requires Level 4.1 (gap-filling). Tick a "
                "Level 4.1 method, or turn Level 4.2 off.")
            return
        problem = self._l42_problem(l42, l41["methods"]) if l42 else None
        if problem:
            self.summary.setPlainText(problem)
            return
        if self._site_missing():
            return
        deepest = "4.2" if l42 else (
            "4.1" if l41 else ("3.3" if l33 else ("3.2" if steps else "3.1")))
        # Run-all rebuilds from scratch to `deepest`; any already-computed level
        # beyond it is cascaded away. Confirm before discarding that real work.
        if not self._confirm_discard(self._level_to_stage(deepest)):
            return
        self._set_running(True)
        self.summary.setPlainText(f"Running Level 2 → Level {deepest}…")
        threading.Thread(
            target=self._worker, args=(self._df, ik, l2, l31, steps, l33, l41, l42),
            daemon=True).start()

    def _worker(self, df, init_kwargs, level2_settings, level31_kwargs, level32_steps,
                level33_kwargs, level41_cfg=None, level42_cfg=None) -> None:
        try:
            data = self._compute(df, init_kwargs, level2_settings, level31_kwargs,
                                  level32_steps, level33_kwargs, level41_cfg)
            l41_data = None
            if level42_cfg:
                # Handed to _on_done with the result: a later L4.1 re-run starts
                # from it. Tab state changes on the GUI thread only, so a failed
                # run leaves it as it was.
                l41_data = data
                data = self._run_level42(data, level42_cfg)
        except Exception as err:
            self._sig.failed.emit(str(err))
            return
        self._sig.done.emit(data, l41_data)

    # --- per-level run (each level's output feeds the next) ---
    def _run_level(self, idx: int) -> None:
        """Run a single level on the current chain output, then feed forward."""
        if self._df is None or self._running:
            return
        if idx > 0 and self._reached < idx - 1:
            self.summary.setPlainText(
                f"Run {_STAGES[idx - 1][0]} first — each level feeds the next.")
            return
        plan = self._level_plan(idx)
        if plan is None:
            return  # _level_plan set the explanatory message
        if not self._confirm_discard(idx):
            return  # user declined to discard downstream results
        self._set_running(True)
        self.summary.setPlainText(f"Running {_STAGES[idx][0]} — {_STAGES[idx][1]}…")
        threading.Thread(target=self._level_worker,
                         args=(plan, self._level_base(idx), self._df),
                         daemon=True).start()

    def _level_base(self, idx: int):
        """The container a single-level run starts from.

        The first L4.2 run on an L4.1 result keeps that result. Re-running
        L4.1 after L4.2 starts from it, so the partitioning of the old
        gap-filling is discarded as the confirm prompt says (the library keeps
        L4.2 through an L4.1 re-run, L4.2 being additive). ``_l41_data`` is
        set when an L4.2 run completes (``_on_done``, ``_on_level_done``)."""
        if idx == 5 and self._reached >= 6 and self._l41_data is not None:
            return self._l41_data
        return self._data

    def _level_plan(self, idx: int) -> dict | None:
        """Collect + validate the kwargs for a single level; None blocks the run."""
        if idx == 0:
            # Only initialization reads the site; later levels inherit its result.
            if self._site_missing():
                return None
            return {"idx": 0, "kind": "init", "init_kwargs": self._init_kwargs()}
        if idx == 1:
            l2 = self._level2_settings()
            if not l2:
                self.summary.setPlainText("Enable at least one Level 2 test.")
                return None
            return {"idx": 1, "kind": "level2", "settings": l2}
        if idx == 2:
            return {"idx": 2, "kind": "level31", "kwargs": self._level31_kwargs()}
        if idx == 3:
            if not self._steps:
                self.summary.setPlainText("Add at least one Level 3.2 outlier step.")
                return None
            return {"idx": 3, "kind": "level32", "steps": list(self._steps)}
        if idx == 4:
            l33 = self._level33_kwargs()
            if not l33:
                self.summary.setPlainText(self._l33_problem())
                return None
            return {"idx": 4, "kind": "level33", "kwargs": l33}
        if idx == 6:
            return self._level42_plan()
        # idx == 5
        l41 = self._level41_cfg()
        if not l41 or self._ref_needs_mds(l41):
            return None
        scen = getattr(self._data.levels, "ustar_scenarios", None) \
            if self._data is not None else None
        if l41.get("ustar_reference") and (scen is None or not scen.reference_labels):
            self.summary.setPlainText(
                "The REF needs the 40 REF scenarios of Level 3.3. Run Level 3.3 "
                "again with 'Add REF (ONEFlux)' on.")
            return None
        if ("rf" in l41["methods"] or "xgb" in l41["methods"]) and not l41.get("features"):
            self.summary.setPlainText(
                "Random Forest / XGBoost need at least one feature. Select "
                "predictor columns in the Level 4.1 feature list.")
            return None
        return {"idx": 5, "kind": "level41", "cfg": l41}

    def _level42_plan(self) -> dict | None:
        """Validate a single L4.2 run on the current L4.1 output."""
        l42 = self._level42_cfg()
        if not l42:
            self.summary.setPlainText("Tick 'Partition NEE into GPP and RECO' first.")
            return None
        ran = sorted(self._data.gapfilled_cols()) if self._data is not None else []
        problem = self._l42_problem(l42, ran)
        if problem:
            self.summary.setPlainText(problem)
            return None
        if l42.get("ustar_reference") and (
                self._data is None or self._data.levels.ustar_reference is None):
            self.summary.setPlainText(
                "The GPP / RECO REF needs the NEE REF. Run Level 4.1 again with "
                "'Add REF (ONEFlux)' on in Level 3.3 and MDS ticked.")
            return None
        return {"idx": 6, "kind": "level42", "cfg": l42}

    def _ref_needs_mds(self, l41: dict | None) -> bool:
        """Refuse an L4.1 run without a method, or with REF on but no MDS (the
        REF is selected among the MDS gap-filled versions), and say why."""
        if not l41:
            self.summary.setPlainText("Select a Level 4.1 method (rf / xgb / mds).")
            return True
        if self._ref_on() and "mds" not in l41["methods"]:
            self.summary.setPlainText(
                "REF (ONEFlux) is on in Level 3.3 and needs MDS: tick Marginal "
                "Distribution Sampling (mds).")
            return True
        return False

    def _level_worker(self, plan: dict, base, df) -> None:
        try:
            kind = plan["kind"]
            if kind == "init":
                work = df.drop(columns=[c for c in _RESERVED if c in df.columns])
                data = init_flux_data(df=work, **plan["init_kwargs"])
            elif kind == "level2":
                data = run_level2(base, **plan["settings"])
            elif kind == "level31":
                data = run_level31(base, **plan["kwargs"])
            elif kind == "level32":
                data, sod = make_level32_detector(base)
                for step in plan["steps"]:
                    getattr(sod, step["method"])(**step.get("kwargs", {}))
                    sod.addflag()
                data = run_level32(data, outlier_detector=sod)
            elif kind == "level33":
                data = self._run_level33(base, plan["kwargs"])
            elif kind == "level41":
                data = self._run_level41(base, plan["cfg"])
            else:  # level42
                data = self._run_level42(base, plan["cfg"])
        except Exception as err:
            self._sig.failed.emit(str(err))
            return
        self._sig.level_done.emit((data, plan["idx"]))

    # --- completion ---
    @staticmethod
    def _reached_from_data(data) -> int:
        """Deepest stage index materialised in a container (for run-all)."""
        ids = list(getattr(data, "level_ids", []) or [])
        if data.levels.has_level42():
            return 6
        if data.gapfilled_cols():
            return 5
        for lid, stage in (("L3.3", 4), ("L3.2", 3), ("L3.1", 2), ("L2", 1)):
            if lid in ids:
                return stage
        return 0

    def _on_done(self, data, l41_data=None) -> None:
        self._l41_data = l41_data  # the L4.1 container L4.2 started from, or None
        self._finalize(data, self._reached_from_data(data))

    def _on_level_done(self, payload) -> None:
        data, idx = payload
        if idx == 6 and self._reached == 5:
            self._l41_data = self._data  # the L4.1 result this first L4.2 run started from
        self._finalize(data, idx)

    def _finalize(self, data, idx: int) -> None:
        """Adopt a completed container and render the level it reached."""
        self._data = data
        self._reached = idx
        self._set_running(False)
        self.rail.set_reached_through(idx)
        self._render_stage(idx, data)
        self._update_report(idx, data)
        if self._results_stale():  # settings changed while the level ran
            self.summary.appendPlainText(_SITE_CHANGED)

    @staticmethod
    def _level_qcf(idx: int, data):
        """The FlagQCF for the level a stage index reached (None for init/L4.1)."""
        levels = data.levels
        if idx == 1:
            return getattr(levels, "level2_qcf", None)
        if idx == 2:
            return getattr(levels, "level31_qcf", None)
        if idx == 3:
            return getattr(levels, "level32_qcf", None)
        if idx == 4:
            scen = getattr(levels, "level33_qcf", None) or {}
            return next(iter(scen.values()), None)
        return None

    def _update_report(self, idx: int, data) -> None:
        """Fill the QCF report panel with the screening report for this level."""
        qcf = self._level_qcf(idx, data)
        if idx >= 6:
            # L4.2: the GPP / RECO REF of the shown variant, if it was selected.
            variant = self._l42_shown_variant(data)
            prec = data.levels.partitioning_reference.get(variant) if variant else None
            if prec is None:
                self.report.setPlainText("")
                self.report_copy.setEnabled(False)
                return
            text = (self._ref42_text(prec) + "\n"
                    + "\n\n".join(f"{prod}:\n{prec.results[prod].summary()}"
                                  for prod in ("GPP", "RECO"))
                    + "\n\nThe u* filter of diive keeps the first record after a low-u* "
                    "period and ONEFlux removes it. The REF can therefore differ "
                    "from ONEFlux's at some time resolutions.")
            self.report.setPlainText(text)
            self.report_copy.setEnabled(True)
            return
        ref = getattr(data.levels, "ustar_reference", None) if idx == 5 else None
        if ref is not None:
            # L4.1 with REF: the selection per time resolution instead of a QCF.
            text = (self._ref_text(ref) + "\n" + ref.result.summary() + "\n\n"
                    "The u* filter of diive keeps the first record after a low-u* "
                    "period and ONEFlux removes it. The REF can therefore differ "
                    "from ONEFlux's at some time resolutions.")
            self.report.setPlainText(text)
            self.report_copy.setEnabled(True)
            return
        if qcf is None:
            self.report.setPlainText("")
            self.report_copy.setEnabled(False)
            return
        try:
            _, text = qcf.screening_report()
        except Exception as err:
            text = f"(QCF report unavailable: {err})"
        self.report.setPlainText(text)
        self.report_copy.setEnabled(bool(text))

    def _report_text(self) -> str | None:
        return self.report.toPlainText() or None

    def _render_stage(self, idx: int, data) -> None:
        """Render the diagnostic for the stage that just completed."""
        if idx >= 6 and data.levels.has_level42():
            self._show_level42(data)
            return
        if idx >= 5 and data.gapfilled_cols():
            self._show_level41(data)
            return
        self.canvas.auto_layout = True  # may have been disabled by an L4.1 heatmap
        note = ""
        if idx == 0:
            series = data.full_df.get(data.meta.fluxcol)
            label = "Input"
        elif idx == 4:
            scen = data.levels.filteredseries_level33_qcf
            main, n_ref = self._main_scenarios(data, scen)
            first = main[0]
            series = scen[first]
            label = "L3.3"
            note = (f"\nUSTAR scenarios: {', '.join(main)}"
                    + (f" + {n_ref} REF versions" if n_ref else "")
                    + f" (preview: {first}).")
            note += self._l33_threshold_note(data, main)
        else:
            series = data.filteredseries
            label = {1: "L2", 2: "L3.1", 3: "L3.2"}.get(idx, f"L{idx}")
            if idx == 3:
                l32_qcf = getattr(data.levels, "level32_qcf", None)
                if l32_qcf is not None:
                    try:
                        vc = l32_qcf.flagqcf.value_counts()
                        note = (f"\nL3.2 QCF — 0:{int(vc.get(0, 0))}  "
                                f"1:{int(vc.get(1, 0))}  2:{int(vc.get(2, 0))}.")
                    except Exception:
                        note = ""
        valid = int(series.dropna().count()) if series is not None else 0
        name = series.name if series is not None else "—"
        self.summary.setPlainText(
            f"{label} done — {valid} valid records (series: {name}).{note}\n"
            f"Run the next level, or 'Copy Python' for the reproducible script.")
        ax = self.canvas.new_axes(1)[0]
        try:
            dv.plotting.HeatmapDateTime(series).plot(
                ax=ax, fig=self.canvas.fig, cmap=theme.manager.heatmap_cmap,
                format_style=dv.plotting.FormatStyle(title=f"{label}: {name}"),
                cb_digits_after_comma="auto",
                as_image=True)  # one image: faster than a mesh at panel size, same picture
        except Exception as err:
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
        self.canvas.draw()

    @staticmethod
    def _main_scenarios(data, scenarios) -> tuple[list[str], int]:
        """The scenarios without the 40 REF versions, and the number of REF
        versions (all scenarios when every one is a REF version)."""
        rec = getattr(data.levels, "ustar_scenarios", None)
        ref = set(rec.reference_labels) if rec is not None else set()
        main = [s for s in scenarios if s not in ref]
        return (main or list(scenarios)), len(ref)

    @staticmethod
    def _l33_threshold_note(data, labels: list[str]) -> str:
        """Thresholds of the shown scenarios (detected or given), and the years
        and methods of a detection that gave no thresholds."""
        rec = getattr(data.levels, "ustar_scenarios", None)
        boot = getattr(data.levels, "ustar_detection", None)
        note = ""
        try:
            if rec is not None:
                parts = []
                for lab in labels:
                    v = rec.thresholds[lab]
                    if isinstance(v, dict):  # VUT: {year: threshold}
                        vals = [x for x in v.values() if x == x]
                        if vals:
                            parts.append(f"{lab}={min(vals):.3f}-{max(vals):.3f}")
                    else:
                        parts.append(f"{lab}={v:.3f}")
                src = "Detected" if rec.source == "detected" else "Given"
                method = (f" ({USTAR_METHOD_SHORT.get(getattr(boot, 'mode', ''), '')})"
                          if boot is not None else "")
                note += (f"\n{src} {rec.threshold_type.upper()}{method}: "
                         + "  ".join(parts))
            if boot is not None:
                fails = [f"{y} {m.upper()}: {why}"
                         for y, d in sorted((boot.failures_ or {}).items())
                         for m, why in d.items()]
                if fails:
                    note += "\nNo thresholds for: " + ". ".join(fails) + "."
        except Exception:
            pass
        return note

    @staticmethod
    def _ref_text(ref) -> str:
        """REF result as text: scenario, percentile, threshold, REF per time
        resolution."""
        thr = ref.threshold
        if isinstance(thr, dict):  # VUT: {year: threshold}
            thr_txt = ", ".join(f"{y}: {v:.3f}" for y, v in thr.items()) + " m s-1"
        else:
            thr_txt = f"{thr:.3f} m s-1"
        per = "  ".join(f"{agg} {lab or 'none'}" for agg, lab in ref.scenarios.items())
        return (f"{ref.label} = {ref.scenario} (percentile {ref.percentile:g}, "
                f"threshold {thr_txt}).\nREF per time resolution: {per}.\n")

    def _show_level41(self, data) -> None:
        """Render the L4.1 method comparison (cumulative or heatmaps) into the canvas."""
        cols = data.gapfilled_cols()
        scenarios = sorted({s for mc in cols.values() for s in mc})
        main, n_ref = self._main_scenarios(data, scenarios)
        methods = ", ".join(sorted(cols))
        ref = getattr(data.levels, "ustar_reference", None)
        first = ref.scenario if ref is not None else main[0]
        self.summary.setPlainText(
            f"Level 4.1 done — gap-filling methods: {methods}.\n"
            f"USTAR scenarios: {', '.join(main)}"
            + (f" + {n_ref} REF versions" if n_ref else "")
            + f" (preview: {first}).\n"
            + (self._ref_text(ref) if ref is not None else "")
            + "Use 'Copy Python' for the reproducible script.")
        heatmaps = self.l41_view.currentText().startswith("Gap-filled")
        try:
            if heatmaps:
                # Whole-figure gridspec (like the ridgeline) — let the plot own
                # the layout, render a single scenario into the canvas figure.
                self.canvas.auto_layout = False
                self.canvas.reset_layout()
                data.plot_gapfilled_heatmaps(
                    ustar_scenario=first, fig=self.canvas.fig, showplot=False)
            else:
                self.canvas.auto_layout = True
                ax = self.canvas.new_axes(1)[0]
                data.plot_cumulative_comparison(
                    ustar_scenario=first, ax=ax, fig=self.canvas.fig, showplot=False)
        except Exception as err:
            self.canvas.auto_layout = True
            ax = self.canvas.new_axes(1)[0]
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
        self.canvas.draw()

    def _l42_shown_variant(self, data) -> str | None:
        """The L4.2 variant to show: the selected one if it ran, else the first."""
        variants = data.levels.level42_variants()
        if not variants:
            return None
        selected = self._l42_variant()
        return selected if selected in variants else next(iter(variants))

    @staticmethod
    def _ref42_text(prec) -> str:
        """GPP / RECO REF as text: REF u* percentile per product and time resolution."""
        lines = []
        for prod in ("GPP", "RECO"):
            res = prec.results[prod]
            per = "  ".join(f"{agg} {p:g}" if p == p else f"{agg} none"
                            for agg, p in res.percentile.items())
            lines.append(f"{prec.columns[prod]}: REF u* percentile per time "
                         f"resolution: {per}.")
        return "\n".join(lines) + "\n"

    def _show_level42(self, data) -> None:
        """Render GPP and RECO of the shown variant: daily means and cumulative sums
        of one u* scenario, plus the GPP / RECO REF when it was selected."""
        variants = data.levels.level42_variants()
        variant = self._l42_shown_variant(data)
        scenarios = list(variants[variant])
        nee_ref = getattr(data.levels, "ustar_reference", None)
        first = nee_ref.scenario if nee_ref is not None and nee_ref.scenario in scenarios \
            else scenarios[0]
        tag = variant.upper()
        prec = data.levels.partitioning_reference.get(variant)
        others = [v.upper() for v in variants if v != variant]
        self.summary.setPlainText(
            f"Level 4.2 done: partitioning {tag}"
            + (f" (also in the chain: {', '.join(others)})" if others else "") + ".\n"
            + f"USTAR scenarios: {', '.join(scenarios)} (preview: {first}).\n"
            + (self._ref42_text(prec) if prec is not None else "")
            + "Use 'Copy Python' for the reproducible script.")
        lines = [(f"GPP_{tag}_{first}", _C_GPP, "-"), (f"RECO_{tag}_{first}", _C_RECO, "-")]
        if prec is not None:
            lines += [(prec.columns["GPP"], _C_GPP, "--"), (prec.columns["RECO"], _C_RECO, "--")]
        self.canvas.auto_layout = True  # may have been disabled by an L4.1 heatmap
        ax_ts, ax_cum = self.canvas.new_axes(2, orientation="vertical", sharex=True)
        try:
            for col, color, ls in lines:
                s = data.fpc_df.get(col)
                if s is None:
                    continue
                d = s.resample("1D").mean()
                ax_ts.plot(d.index, d.to_numpy(), color=color, ls=ls, lw=1.0,
                           label=f"{col} (daily mean)")
                c = s.fillna(0).cumsum()
                ax_cum.plot(c.index, c.to_numpy(), color=color, ls=ls, lw=1.0,
                            label=f"Σ {col}")
            for ax in (ax_ts, ax_cum):
                ax.axhline(0, color="#B0BEC5", lw=0.6, zorder=0)
                ax.legend(frameon=False, fontsize=8)
            ax_ts.set_ylabel("umol m-2 s-1")
            ax_ts.set_title(f"L4.2 {tag}, {first}: daily means", fontsize=9)
            ax_cum.set_ylabel("cumulative")
            ax_cum.set_title("Cumulative sum", fontsize=9)
        except Exception as err:
            ax_ts.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                       wrap=True, transform=ax_ts.transAxes)
        self.canvas.draw()

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.summary.setPlainText(f"Failed:\n{msg}")

    @staticmethod
    def _level_name(idx: int) -> str:
        """Human level name for a stage index ('Input' / 'Level 3.2')."""
        badge = _STAGES[idx][0]
        return "Input" if idx == 0 else badge.replace("L", "Level ")

    def _confirm_discard(self, target_idx: int) -> bool:
        """Confirm a run that would discard already-computed downstream levels.

        Returns True to proceed. A run reaching ``target_idx`` rebuilds that level
        and cascades away every deeper level the library already computed; only
        then (``self._reached > target_idx``) do we prompt, so the normal forward
        path and first runs never ask. No-op (returns True) when nothing is lost.
        """
        if self._reached <= target_idx:
            return True
        discarded = [self._level_name(i) for i in range(target_idx + 1, self._reached + 1)]
        names = " and ".join(discarded) if len(discarded) <= 2 else \
            ", ".join(discarded[:-1]) + ", and " + discarded[-1]
        answer = QMessageBox.question(
            self.run_btn, "Discard downstream results",
            f"Re-running {self._level_name(target_idx)} will discard the results of "
            f"{names}. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    # --- run-state + per-level button enablement ---
    def _set_running(self, on: bool) -> None:
        self._running = on
        if on:
            self.run_btn.setEnabled(False)
            for b in (*self._level_run_btns.values(), *self._level_add_btns.values()):
                b.setEnabled(False)
        else:
            self._update_level_buttons()

    def _update_level_buttons(self) -> None:
        """A level's run button is enabled only once its predecessor has run; its
        "Add to dataset" button only once that level itself has run."""
        if getattr(self, "_running", False):
            return
        has = self._df is not None
        self.run_btn.setEnabled(has)
        for idx, b in self._level_run_btns.items():
            b.setEnabled(has and (idx == 0 or self._reached >= idx - 1))
        # Not for results initialized with other Project settings than the current.
        stale = self._results_stale()
        for idx, b in self._level_add_btns.items():
            b.setEnabled(has and self._reached >= idx and not stale)

    # --- pushing a level's results back into the dataset ---
    #: Stage index -> the idstr(s) keying that level's columns in added_columns.
    _LEVEL_IDSTRS = {1: ("L2",), 2: ("L3.1",), 3: ("L3.2",), 4: ("L3.3",),
                     5: ("L4.1_mds", "L4.1_rf", "L4.1_xgb", "L4.1_ref"),
                     6: ("L4.2_nt_of", "L4.2_nt_rp", "L4.2_dt_rp", "L4.2_dt_of",
                         "L4.2_ref_nt_of", "L4.2_ref_nt_rp", "L4.2_ref_dt_rp",
                         "L4.2_ref_dt_of")}

    def _level_output_columns(self, idx: int, data) -> list[str]:
        """The fpc_df columns a level appended (flags / QCF / gap-filled)."""
        added = getattr(data, "added_columns", {}) or {}
        cols: list[str] = []
        for idstr in self._LEVEL_IDSTRS.get(idx, ()):
            cols += [c for c in added.get(idstr, []) if c in data.fpc_df.columns]
        return cols

    def _level_filtered_series(self, idx: int, data) -> dict[str, pd.Series]:
        """The QCF-filtered flux series a level produced, keyed by column name.

        Series names are already level-qualified (e.g. ``FC_L3.1_QCF``), so they
        don't collide with the original flux column. L3.3 has one per USTAR
        scenario; L4.1 has none (its gap-filled columns are in fpc_df)."""
        out: dict[str, pd.Series] = {}
        if idx in (1, 2, 3):
            s = data.filteredseries
            if s is not None and getattr(s, "name", None):
                out[str(s.name)] = s
        elif idx == 4:
            scen = getattr(data.levels, "filteredseries_level33_qcf", None) or {}
            for label, s in scen.items():
                if s is None:
                    continue
                name = str(getattr(s, "name", "") or f"{data.meta.fluxcol}_L3.3_{label}_QCF")
                while name in out:  # defensive: keep every scenario distinct
                    name = f"{name}_{label}"
                out[name] = s
        return out

    def _add_level(self, idx: int) -> None:
        """Emit this level's columns + filtered flux to the app-wide dataset."""
        data = self._data
        if data is None or self._reached < idx or self._results_stale():
            return
        label = _STAGES[idx][0]
        cols = self._level_output_columns(idx, data)
        filtered = self._level_filtered_series(idx, data)
        if not cols and not filtered:
            self.summary.setPlainText(f"{label}: no columns to add.")
            return
        out = data.fpc_df.loc[:, cols].copy() if cols \
            else pd.DataFrame(index=data.fpc_df.index)
        for name, s in filtered.items():
            out[name] = s.reindex(out.index)
        out.attrs[ATTRS_KEY] = {
            str(c): provenance_attr(
                origin=DERIVED, parent=str(data.meta.fluxcol),
                operation=f"Flux post-processing chain {label}",
                tags=["fluxchain", label.lower()])
            for c in out.columns
        }
        if not self._emit_features(out):
            self.summary.setPlainText("Nothing added: the existing variables were kept.")
            return
        self.summary.setPlainText(
            f"Added {len(out.columns)} {label} column(s) to the dataset.")

    # --- the killer button: the exact reproducible script ---
    def _code(self) -> str | None:
        if self._site_missing():
            return None  # the script would carry latitude 0, longitude 0 and UTC
        hint = "dv.load_parquet('your_data.parquet')"
        ik, l2, l31 = self._init_kwargs(), self._level2_settings(), self._level31_kwargs()
        l33 = self._level33_kwargs()
        l41 = self._level41_cfg()
        l42 = self._level42_cfg()
        if l42 and l41 and l33 and self._steps:
            return level42_to_code(ik, l2, l31, self._steps, l33, l41, l42, load_hint=hint)
        if l41 and l33 and self._steps:
            return level41_to_code(ik, l2, l31, self._steps, l33, l41, load_hint=hint)
        if l33 and self._steps:
            return level33_to_code(ik, l2, l31, self._steps, l33, load_hint=hint)
        if self._steps:
            return level32_to_code(ik, l2, l31, self._steps, load_hint=hint)
        return level31_to_code(ik, l2, l31, load_hint=hint)
