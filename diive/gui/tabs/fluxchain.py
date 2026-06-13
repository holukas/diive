"""
GUI.TABS.FLUXCHAIN: FLUX PROCESSING CHAIN (Input + L2 + L3.1 + L3.2 + L3.3 + L4.1)
=================================================================================

A guided tab for the Swiss-FluxNet flux processing chain. It wires the **Input**
(site + flux column), **Level 2** (quality-flag tests), **Level 3.1** (single-point
storage correction), **Level 3.2** (an optional outlier-detection chain), and
**Level 3.3** (optional constant-USTAR filtering) using the composable library
callables (`init_flux_data` + `run_level2` + `run_level31` +
`make_level32_detector` / `run_level32` + `run_level33_constant_ustar`), shows the
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

**Level 4.1** gap-fills the per-scenario L3.3 flux: tick any of rf / xgb / mds
(additive across methods — each replaces only its own previous result), pick
rf/xgb predictor features and the MDS SW_IN / TA / VPD driver columns, and the
run fans out one gap-fill per USTAR scenario on the worker thread. The canvas
then shows the method comparison (cumulative overlay or side-by-side heatmaps)
via the library's `plot_cumulative_comparison(ax=...)` / `plot_gapfilled_heatmaps(fig=...)`,
and **Copy Python** switches to `level41_to_code`.

The chain stays on the **composable per-level** path (not `run_chain`/`FluxConfig`)
so L3.2 can be a real inspected `StepwiseOutlierDetection` chain with a separate QCF
surface. All script-generation is the library's `level*_to_code` renderers; the
tab only collects parameters and calls them.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
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
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.flux.fluxprocessingchain import (
    VM97_SUBTESTS, init_flux_data, level2_test_inputs, level31_storage_col,
    level31_to_code, level32_to_code, level33_to_code, level41_to_code,
    make_level32_detector, make_level41_engineer, run_level2, run_level31,
    run_level32, run_level33_constant_ustar, run_level41_mds, run_level41_rf,
    run_level41_xgb,
)
from diive.flux.lowres.common import detect_fluxbasevar
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.flux_pipeline_rail import PipelineRail
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.stepwise_method_params import STEP_METHOD_BY_KEY, method_labels

#: Pipeline stages shown on the rail -> (badge, title). Index = stacked page +
#: the level reached after a run (see _level_to_stage).
_STAGES = [
    ("INPUT", "Site & flux"),
    ("L2", "Quality flags"),
    ("L3.1", "Storage"),
    ("L3.2", "Outliers"),
    ("L3.3", "USTAR"),
    ("L4.1", "Gap-filling"),
]

#: Columns init_flux_data computes itself and rejects if already present.
_RESERVED = ("SW_IN_POT", "DAYTIME", "NIGHTTIME")

_C_MUTED = "#6B7780"

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


class _ChainSignals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject)."""
    done = Signal(object)        # FluxLevelData (run-all completion)
    level_done = Signal(object)  # (FluxLevelData, stage_idx) — single-level run
    failed = Signal(str)


class FluxChainTab(DiiveTab):
    """Guided flux processing chain — Input + Level 2 (first slice)."""

    title = "Flux chain"

    def build(self) -> QWidget:
        self._df = None
        self._data = None        # the evolving container; each level appends to it
        self._reached = -1        # deepest stage index run on _data (-1 = none)
        self._running = False
        self._level_run_btns: dict[int, QPushButton] = {}
        self._sig = _ChainSignals()
        self._sig.done.connect(self._on_done)
        self._sig.level_done.connect(self._on_level_done)
        self._sig.failed.connect(self._on_failed)

        root = QWidget()
        outer = QVBoxLayout(root)

        # Action bar: title + the chain-wide actions (run the whole chain, copy).
        bar = QHBoxLayout()
        title = QLabel(theme.manager.label_text("Flux processing chain"))
        title.setFont(theme.manager.tracked_font(point_delta=1.0))
        title.setStyleSheet("font-weight: bold;")
        bar.addWidget(title)
        bar.addStretch(1)
        self.run_btn = QPushButton("Run through Level 3.1")
        self.run_btn.setToolTip("Run the whole chain from scratch through the deepest "
                                "configured level.")
        self.run_btn.clicked.connect(self._run)
        theme.set_button_role(self.run_btn, "confirm")
        bar.addWidget(self.run_btn)
        # Standardized copy button: copies the script to the clipboard with a
        # "Copied ✓" flash — no code dump in the summary box.
        self.code_btn = CopyPythonButton(self._code)
        bar.addWidget(self.code_btn)
        outer.addLayout(bar)

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
                     self._level32_group(), self._level33_group(), self._level41_group()):
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
        btn.clicked.connect(lambda: self._run_level(idx))
        self._level_run_btns[idx] = btn
        return btn

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
        return {"3.1": 2, "3.2": 3, "3.3": 4, "4.1": 5}.get(level, 2)

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
        if self.l33_enable.isChecked() and self._ustar:
            n = len(self._ustar)
            l33 = (f"{n} scenario" + ("s" if n != 1 else ""), "set")
        elif self.l33_enable.isChecked():
            l33 = ("add threshold", "warn")
        else:
            l33 = ("off", "off")
        methods = self._level41_methods()
        return [
            (fc or "—", "set" if fc else "todo"),
            (f"{n2} test" + ("s" if n2 != 1 else ""), "warn" if n2 == 0 else "set"),
            l31,
            (f"{n32} step" + ("s" if n32 != 1 else ""), "set") if n32 else ("none", "off"),
            l33,
            (" · ".join(methods), "set") if methods else ("off", "off"),
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
        self.site_lat = self._dspin(47.42, -90, 90, 4)
        form.addRow("Latitude", self.site_lat)
        self.site_lon = self._dspin(8.49, -180, 180, 4)
        form.addRow("Longitude", self.site_lon)
        self.utc_offset = QSpinBox()
        self.utc_offset.setRange(-12, 14)
        self.utc_offset.setValue(1)
        form.addRow("UTC offset (h)", self.utc_offset)
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
                cb.toggled.connect(lambda *_: self._refresh_vm97_enabled())
                v.addLayout(self._vm97_subtests())
        v.addWidget(self._level_run_button(1, "Run Level 2"))
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
        self.l31_zero.toggled.connect(
            lambda on: self.l31_gapfill.setEnabled(not on))
        v.addWidget(self.l31_zero)
        form = QFormLayout()
        self.strgcol = QComboBox()  # "" = auto-detect (FLUXNET/EddyPro naming)
        self.strgcol.currentTextChanged.connect(self._refresh_levels_info)
        form.addRow("Storage column (auto if blank)", self.strgcol)
        v.addLayout(form)
        self.strg_mark = self._info_marker()
        v.addWidget(self.strg_mark)
        v.addWidget(self._level_run_button(2, "Run Level 3.1"))
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
                            ("Up", lambda: self._move_l32_step(-1)),
                            ("Down", lambda: self._move_l32_step(1))):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        v.addWidget(self._level_run_button(3, "Run Level 3.2"))
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
        self.l33_enable = QCheckBox("Apply constant USTAR thresholds")
        self.l33_enable.setToolTip(
            "Flag low-turbulence (low USTAR) periods. Requires at least one Level 3.2 "
            "outlier test. Only for CO2/CH4/N2O — for H/LE use threshold 0.")
        self.l33_enable.toggled.connect(self._update_run_label)
        v.addWidget(self.l33_enable)
        # Which USTAR column it filters on (set on the Input page) + applicability.
        self.l33_info = self._info_marker()
        v.addWidget(self.l33_info)
        note = QLabel("One scenario per threshold (m s⁻¹). Label is optional "
                      "(auto CUT_0, CUT_1, …); use e.g. CUT_50 for a percentile.")
        note.setWordWrap(True)
        v.addWidget(note)
        row = QHBoxLayout()
        self.l33_value = self._dspin(0.10, 0.0, 5.0, 3)
        row.addWidget(self.l33_value)
        self.l33_label = QLineEdit()
        self.l33_label.setPlaceholderText("label (optional)")
        row.addWidget(self.l33_label)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_ustar)
        row.addWidget(add_btn)
        v.addLayout(row)
        # Thresholds as (value, label) — label "" means auto.
        self._ustar: list[tuple[float, str]] = []
        self.l33_list = QListWidget()
        self.l33_list.setMaximumHeight(90)
        v.addWidget(self.l33_list)
        rm = QPushButton("Remove threshold")
        rm.clicked.connect(self._remove_ustar)
        v.addWidget(rm)
        v.addWidget(self._level_run_button(4, "Run Level 3.3"))
        return box

    def _add_ustar(self) -> None:
        value = self.l33_value.value()
        label = self.l33_label.text().strip()
        self._ustar.append((value, label))
        self.l33_list.addItem(f"{value:g}" + (f"  ({label})" if label else ""))
        self.l33_label.clear()
        self._update_run_label()

    def _remove_ustar(self) -> None:
        row = self.l33_list.currentRow()
        if row < 0:
            return
        self.l33_list.takeItem(row)
        del self._ustar[row]
        self._update_run_label()

    def _level33_kwargs(self) -> dict | None:
        """run_level33_constant_ustar kwargs, or None when L3.3 is off/empty."""
        if not self.l33_enable.isChecked() or not self._ustar:
            return None
        thresholds = [v for v, _ in self._ustar]
        labels = [lab for _, lab in self._ustar]
        kwargs: dict = {"thresholds": thresholds}
        # All-or-nothing labels: pass them only if every scenario is labelled,
        # else let the library auto-generate CUT_0, CUT_1, …
        if all(labels):
            kwargs["threshold_labels"] = labels
        return kwargs

    def _level41_group(self) -> QGroupBox:
        box = QGroupBox("Level 4.1 — gap-filling (optional)")
        v = QVBoxLayout(box)
        note = QLabel("Fill remaining gaps in the L3.3 flux. One gap-fill per USTAR "
                      "scenario; methods are additive (each replaces only its own "
                      "previous result). Requires Level 3.3.")
        note.setWordWrap(True)
        v.addWidget(note)
        # Method toggles. ML methods (rf/xgb) are slow; off by default.
        self.l41_rf = QCheckBox("Random Forest (rf)")
        self.l41_xgb = QCheckBox("XGBoost (xgb)")
        self.l41_mds = QCheckBox("Marginal Data Substitution (mds)")
        for cb in (self.l41_rf, self.l41_xgb, self.l41_mds):
            cb.toggled.connect(self._update_run_label)
            v.addWidget(cb)
        # Shared ML settings. A fixed random seed makes rf/xgb runs reproducible
        # (without it sklearn/xgboost reseed every run -> the output drifts).
        shared = QFormLayout()
        self.l41_seed = QSpinBox()
        self.l41_seed.setRange(0, 2_000_000_000)
        self.l41_seed.setValue(42)
        self.l41_seed.setToolTip("Random seed for rf / xgb. Same seed -> identical "
                                 "gap-fill every run; change it to vary deliberately.")
        shared.addRow("Random seed (rf / xgb)", self.l41_seed)
        v.addLayout(shared)
        self.l41_reduce = QCheckBox("Reduce features (SHAP selection)")
        self.l41_reduce.setToolTip("Drop low-importance engineered features via "
                                   "SHAP before the final fit (slower, often cleaner).")
        v.addWidget(self.l41_reduce)
        # rf/xgb predictor features — multi-select from the dataset columns.
        v.addWidget(QLabel("Features (rf / xgb predictors):"))
        self.l41_features = QListWidget()
        self.l41_features.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.l41_features.setMaximumHeight(110)
        v.addWidget(self.l41_features)
        # Random-Forest hyperparameters (defaults match sklearn -> no behaviour
        # change vs not passing them; only what you edit is sent + rendered).
        rf_form = QFormLayout()
        self.rf_n_est = self._ispin(100, 1, 5000)
        rf_form.addRow("RF — trees (n_estimators)", self.rf_n_est)
        self.rf_max_depth = self._ispin(0, 0, 200)
        self.rf_max_depth.setSpecialValueText("None")  # 0 -> unlimited depth
        rf_form.addRow("RF — max depth", self.rf_max_depth)
        v.addLayout(rf_form)
        # XGBoost hyperparameters (defaults match xgboost).
        xgb_form = QFormLayout()
        self.xgb_n_est = self._ispin(100, 1, 5000)
        xgb_form.addRow("XGB — trees (n_estimators)", self.xgb_n_est)
        self.xgb_max_depth = self._ispin(6, 1, 200)
        xgb_form.addRow("XGB — max depth", self.xgb_max_depth)
        self.xgb_lr = self._dspin(0.30, 0.001, 1.0, 3)
        xgb_form.addRow("XGB — learning rate", self.xgb_lr)
        v.addLayout(xgb_form)
        # MDS driver columns — must exist in the input (full_df).
        form = QFormLayout()
        self.mds_swin = QComboBox()
        self.mds_swin.currentTextChanged.connect(self._refresh_levels_info)
        form.addRow("MDS SW_IN (W m⁻²)", self.mds_swin)
        self.mds_ta = QComboBox()
        self.mds_ta.currentTextChanged.connect(self._refresh_levels_info)
        form.addRow("MDS TA (°C)", self.mds_ta)
        self.mds_vpd = QComboBox()
        self.mds_vpd.currentTextChanged.connect(self._refresh_levels_info)
        form.addRow("MDS VPD (kPa)", self.mds_vpd)
        # MDS similarity tolerances (Reichstein et al. 2005 defaults).
        self.mds_ta_tol = self._dspin(2.5, 0.1, 20.0, 2)
        form.addRow("MDS — TA tolerance (°C)", self.mds_ta_tol)
        self.mds_vpd_tol = self._dspin(0.5, 0.05, 10.0, 2)
        form.addRow("MDS — VPD tolerance (kPa)", self.mds_vpd_tol)
        v.addLayout(form)
        self.mds_mark = self._info_marker()
        v.addWidget(self.mds_mark)
        vpd_note = QLabel("MDS units: VPD must be kPa (EddyPro outputs hPa — divide "
                          "by 10), TA in °C. The library warns on suspicious medians.")
        vpd_note.setWordWrap(True)
        v.addWidget(vpd_note)
        # Which comparison to render after L4.1.
        form2 = QFormLayout()
        self.l41_view = QComboBox()
        self.l41_view.addItems(["Cumulative comparison", "Gap-filled heatmaps"])
        form2.addRow("Comparison view", self.l41_view)
        v.addLayout(form2)
        v.addWidget(self._level_run_button(5, "Run Level 4.1"))
        return box

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
            cfg["features"] = [i.text() for i in self.l41_features.selectedItems()]
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
            cfg["mds"] = mds
        return cfg

    def _update_run_label(self, *_) -> None:
        # Reflects intent: L3.3 configured -> "3.3" even before an L3.2 step exists
        # (the run then surfaces the "needs an L3.2 test" requirement).
        if self._level41_methods():
            deepest = "4.1"
        elif self.l33_enable.isChecked() and self._ustar:
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
        from diive.flux.fluxprocessingchain import FluxConfig
        docs: dict = {}
        for src in (init_flux_data, FluxConfig, run_level2, run_level31):
            docs.update(param_docs(src))
        for param, widget in (
            ("fluxcol", self.fluxcol), ("site_lat", self.site_lat),
            ("site_lon", self.site_lon), ("utc_offset", self.utc_offset),
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
        return {"fluxcol": self.fluxcol, "ustarcol": self.ustarcol,
                "site_lat": self.site_lat,
                "site_lon": self.site_lon, "utc_offset": self.utc_offset,
                "nighttime_threshold": self.nighttime_threshold,
                "day_qcf": self.day_qcf, "night_qcf": self.night_qcf,
                "signal_strength_col": self.signal_strength_col,
                "ss_method": self.ss_method, "ss_threshold": self.ss_threshold,
                "l31_gapfill": self.l31_gapfill, "l31_zero": self.l31_zero,
                "strgcol": self.strgcol,
                "l41_rf": self.l41_rf, "l41_xgb": self.l41_xgb, "l41_mds": self.l41_mds,
                "mds_swin": self.mds_swin, "mds_ta": self.mds_ta, "mds_vpd": self.mds_vpd,
                "l41_view": self.l41_view,
                "l41_seed": self.l41_seed, "l41_reduce": self.l41_reduce,
                "rf_n_est": self.rf_n_est, "rf_max_depth": self.rf_max_depth,
                "xgb_n_est": self.xgb_n_est, "xgb_max_depth": self.xgb_max_depth,
                "xgb_lr": self.xgb_lr,
                "mds_ta_tol": self.mds_ta_tol, "mds_vpd_tol": self.mds_vpd_tol}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._fx_controls()),
                "l2": {k: cb.isChecked() for k, cb in self.l2_checks.items()},
                "l2_vm97": {k: cb.isChecked() for k, cb in self.l2_vm97_checks.items()},
                "l2_cols": {k: [c.currentText() for c in combos]
                            for k, combos in self.l2_cols.items()},
                "l32_steps": self._steps,
                "l33_enabled": self.l33_enable.isChecked(),
                "l33_ustar": [list(u) for u in self._ustar],
                "l41_features": [i.text() for i in self.l41_features.selectedItems()]}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._fx_controls(), state.get("controls"))
        for k, val in (state.get("l2") or {}).items():
            if k in self.l2_checks:
                self.l2_checks[k].setChecked(bool(val))
        for k, val in (state.get("l2_vm97") or {}).items():
            if k in self.l2_vm97_checks:
                self.l2_vm97_checks[k].setChecked(bool(val))
        # Restore per-test column picks (combos seeded by restore_controls' fluxcol).
        for k, picks in (state.get("l2_cols") or {}).items():
            for combo, pick in zip(self.l2_cols.get(k, []), picks):
                if pick and combo.findText(pick) >= 0:
                    combo.setCurrentText(pick)
        # Rebuild the L3.2 chain (a list of {method, kwargs} steps).
        self._steps = list(state.get("l32_steps") or [])
        self.l32_steps_list.clear()
        for step in self._steps:
            self.l32_steps_list.addItem(self._step_summary(step))
        # Rebuild the L3.3 USTAR thresholds.
        self.l33_enable.setChecked(bool(state.get("l33_enabled")))
        self._ustar = [tuple(u) for u in (state.get("l33_ustar") or [])]
        self.l33_list.clear()
        for value, label in self._ustar:
            self.l33_list.addItem(f"{value:g}" + (f"  ({label})" if label else ""))
        # Re-select the saved L4.1 feature rows (set after on_data_loaded filled them).
        wanted = set(state.get("l41_features") or [])
        for i in range(self.l41_features.count()):
            item = self.l41_features.item(i)
            item.setSelected(item.text() in wanted)
        self._refresh_l2_availability()
        self._refresh_levels_info()
        self._update_run_label()

    # --- data ---
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        # New dataset: the chain output and the run progress no longer apply.
        self._data = None
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
        # L4.1: feature list + MDS driver combos. Auto-pick obvious drivers.
        self.l41_features.clear()
        self.l41_features.addItems(cols)
        for combo, needle, avoid in (
            (self.mds_swin, "SW_IN", "POT"),
            (self.mds_ta, "TA", None),
            (self.mds_vpd, "VPD", None),
        ):
            combo.clear()
            combo.addItems(cols)
            for c in cols:
                up = c.upper()
                if up.startswith("FLAG"):
                    continue  # FLAG_VPD_..._ISFILLED etc. are not driver columns
                if needle in up and (avoid is None or avoid not in up):
                    combo.setCurrentText(c)
                    break
        # New dataset: the previous run's reach no longer applies.
        self.rail.set_reached_through(-1)
        self._populate_l2_cols()  # seed the per-test column pickers + availability
        self._refresh_levels_info()  # storage / USTAR / MDS availability markers
        self._refresh_rail()
        self._update_level_buttons()

    # --- collect kwargs the library wants ---
    def _init_kwargs(self) -> dict:
        return dict(
            fluxcol=self.fluxcol.currentText(),
            ustarcol=self.ustarcol.currentText() or "USTAR",
            site_lat=self.site_lat.value(),
            site_lon=self.site_lon.value(),
            utc_offset=self.utc_offset.value(),
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
        """Init container, run L2 → 3.1 → (opt) 3.2 → (opt) 3.3 → (opt) 4.1."""
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
            data = run_level33_constant_ustar(data, showplot=False, verbose=False,
                                              **level33_kwargs)
        if level41_cfg:
            data = self._run_level41(data, level41_cfg)
        return data

    @staticmethod
    def _run_level41(data, cfg: dict):
        """Run the selected L4.1 gap-filling methods (additive across methods)."""
        methods = cfg["methods"]
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
        if "mds" in methods:
            mds = cfg["mds"]
            extra = {k: mds[k] for k in ("ta_tol", "vpd_tol") if k in mds}
            data = run_level41_mds(data, swin=mds["swin"], ta=mds["ta"], vpd=mds["vpd"],
                                   **extra)
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
            self.summary.setPlainText(
                "Level 4.1 (gap-filling) requires Level 3.3 (USTAR). Enable Level "
                "3.3 with a threshold, or disable the Level 4.1 methods.")
            return
        if l33 and not steps:
            # USTAR filtering must operate on outlier-screened data (library raises).
            self.summary.setPlainText(
                "Level 3.3 (USTAR) requires at least one Level 3.2 outlier test. "
                "Add a step or disable Level 3.3.")
            return
        if l41 and ("rf" in l41["methods"] or "xgb" in l41["methods"]) \
                and not l41.get("features"):
            self.summary.setPlainText(
                "Random Forest / XGBoost need at least one feature. Select "
                "predictor columns in the Level 4.1 feature list.")
            return
        deepest = "4.1" if l41 else ("3.3" if l33 else ("3.2" if steps else "3.1"))
        self._set_running(True)
        self.summary.setPlainText(f"Running Level 2 → Level {deepest}…")
        threading.Thread(
            target=self._worker, args=(self._df, ik, l2, l31, steps, l33, l41),
            daemon=True).start()

    def _worker(self, df, init_kwargs, level2_settings, level31_kwargs, level32_steps,
                level33_kwargs, level41_cfg=None) -> None:
        try:
            data = self._compute(df, init_kwargs, level2_settings, level31_kwargs,
                                  level32_steps, level33_kwargs, level41_cfg)
        except Exception as err:
            self._sig.failed.emit(str(err))
            return
        self._sig.done.emit(data)

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
        self._set_running(True)
        self.summary.setPlainText(f"Running {_STAGES[idx][0]} — {_STAGES[idx][1]}…")
        threading.Thread(target=self._level_worker, args=(plan, self._data, self._df),
                         daemon=True).start()

    def _level_plan(self, idx: int) -> dict | None:
        """Collect + validate the kwargs for a single level; None blocks the run."""
        if idx == 0:
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
                self.summary.setPlainText(
                    "Enable Level 3.3 and add at least one USTAR threshold.")
                return None
            return {"idx": 4, "kind": "level33", "kwargs": l33}
        # idx == 5
        l41 = self._level41_cfg()
        if not l41:
            self.summary.setPlainText("Select a Level 4.1 method (rf / xgb / mds).")
            return None
        if ("rf" in l41["methods"] or "xgb" in l41["methods"]) and not l41.get("features"):
            self.summary.setPlainText(
                "Random Forest / XGBoost need at least one feature. Select "
                "predictor columns in the Level 4.1 feature list.")
            return None
        return {"idx": 5, "kind": "level41", "cfg": l41}

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
                data = run_level33_constant_ustar(base, showplot=False, verbose=False,
                                                  **plan["kwargs"])
            else:  # level41
                data = self._run_level41(base, plan["cfg"])
        except Exception as err:
            self._sig.failed.emit(str(err))
            return
        self._sig.level_done.emit((data, plan["idx"]))

    # --- completion ---
    @staticmethod
    def _reached_from_data(data) -> int:
        """Deepest stage index materialised in a container (for run-all)."""
        ids = list(getattr(data, "level_ids", []) or [])
        if data.gapfilled_cols():
            return 5
        for lid, stage in (("L3.3", 4), ("L3.2", 3), ("L3.1", 2), ("L2", 1)):
            if lid in ids:
                return stage
        return 0

    def _on_done(self, data) -> None:
        self._finalize(data, self._reached_from_data(data))

    def _on_level_done(self, payload) -> None:
        data, idx = payload
        self._finalize(data, idx)

    def _finalize(self, data, idx: int) -> None:
        """Adopt a completed container and render the level it reached."""
        self._data = data
        self._reached = idx
        self._set_running(False)
        self.rail.set_reached_through(idx)
        self._render_stage(idx, data)
        self._update_report(idx, data)

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
            first = next(iter(scen))
            series = scen[first]
            label = "L3.3"
            note = f"\nUSTAR scenarios: {', '.join(scen)} (preview: {first})."
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
                ax=ax, fig=self.canvas.fig, title=f"{label}: {name}",
                cb_digits_after_comma="auto")
        except Exception as err:
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
        self.canvas.draw()

    def _show_level41(self, data) -> None:
        """Render the L4.1 method comparison (cumulative or heatmaps) into the canvas."""
        cols = data.gapfilled_cols()
        scenarios = sorted({s for mc in cols.values() for s in mc})
        methods = ", ".join(sorted(cols))
        first = scenarios[0]
        self.summary.setPlainText(
            f"Level 4.1 done — gap-filling methods: {methods}.\n"
            f"USTAR scenarios: {', '.join(scenarios)} (preview: {first}).\n"
            f"Use 'Copy Python' for the reproducible script.")
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

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.summary.setPlainText(f"Failed:\n{msg}")

    # --- run-state + per-level button enablement ---
    def _set_running(self, on: bool) -> None:
        self._running = on
        if on:
            self.run_btn.setEnabled(False)
            for b in self._level_run_btns.values():
                b.setEnabled(False)
        else:
            self._update_level_buttons()

    def _update_level_buttons(self) -> None:
        """A level's run button is enabled only once its predecessor has run."""
        if getattr(self, "_running", False):
            return
        has = self._df is not None
        self.run_btn.setEnabled(has)
        for idx, b in self._level_run_btns.items():
            b.setEnabled(has and (idx == 0 or self._reached >= idx - 1))

    # --- the killer button: the exact reproducible script ---
    def _code(self) -> str:
        hint = "dv.load_parquet('your_data.parquet')"
        ik, l2, l31 = self._init_kwargs(), self._level2_settings(), self._level31_kwargs()
        l33 = self._level33_kwargs()
        l41 = self._level41_cfg()
        if l41 and l33 and self._steps:
            return level41_to_code(ik, l2, l31, self._steps, l33, l41, load_hint=hint)
        if l33 and self._steps:
            return level33_to_code(ik, l2, l31, self._steps, l33, load_hint=hint)
        if self._steps:
            return level32_to_code(ik, l2, l31, self._steps, load_hint=hint)
        return level31_to_code(ik, l2, l31, load_hint=hint)
