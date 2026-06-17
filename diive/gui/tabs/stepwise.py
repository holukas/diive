"""
GUI.TABS.STEPWISE: STEPWISE OUTLIER SCREENING TAB
=================================================

Chain several outlier tests on one variable as a list of editable **method
cards** and inspect what each step removes. Unlike the single-method Outliers
tabs (one detector, one pass), this drives the library's
`StepwiseOutlierDetection`: each committed step runs on the data the previous
steps already cleaned, so spikes are peeled off progressively. The overall
**QCF** (`dv.qaqc.FlagQCF`) is computed separately from the accumulated per-test
flags.

Layout: the shared variable list on the left, a segmented inspector in the centre
(Outliers / Corrections / Report — only the active page takes space; each of the
first two carries its own Run button below its list), and on the right an action
bar (limit-line toggle + Copy/Add) over a large, always-visible plot grid:
original + outliers (optionally with the selected step's detection band), the
cleaned (+ corrected) series, a date/time heatmap of the cleaned series, and the
QCF flag over time plus its 0/1/2 distribution. Edits apply only when the
relevant Run button is clicked.

The per-method parameter widgets and the cards are the shared
`widgets/stepwise_method_params.py` / `widgets/stepwise_cards.py` registries (a
step is a `{"method", "kwargs"}` dict). All detection, QCF, and the reproducible
script (`stepwise_to_code`) are library work; this tab only collects parameters,
runs them on a worker thread, previews, and emits columns.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import copy
import threading

import diive as dv
import pandas as pd
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, MODIFIED, provenance_attr
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.corrections_panel import CorrectionsPanel
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.stepwise_cards import (
    AddStepCard,
    StepCard,
    StepEditorDialog,
)
from diive.gui.widgets.variable_panel import VariablePanel
from diive.preprocessing.corrections import apply_corrections
from diive.preprocessing.outlier_detection import StepwiseOutlierDetection
from diive.preprocessing.outlier_detection.codegen import stepwise_to_code
from diive.qaqc import FlagQCF, MEASUREMENTS, detect_measurement, measurement_label

_C_RAW = "#B0BEC5"      # blue-grey 200 — original line
_C_CLEANED = "#43A047"  # green 600     — cleaned series
_C_REMOVED = "#E53935"  # red 600       — removed points
_C_LIMIT = "#8E24AA"    # purple 600    — detection band
_C_CORRECTED = "#FB8C00"  # orange 600   — corrected series (contrasts cleaned green)
_C_MUTED = "#6B7780"
# QCF levels: colorblind-safe blue→orange→red (RdYlBu), avoiding red/green.
_QCF_COLORS = {0: "#4575B4", 1: "#FDAE61", 2: "#D73027"}  # good / marginal / rejected
_QCF_CMAP = "RdYlBu_r"  # colorblind-safe cousin of RdYlGn (0=blue good, 2=red bad)


class _StepwiseSignals(QObject):
    run_done = Signal(object)
    run_failed = Signal(object)  # (run_id, message)
    features_created = Signal(object)


#: Private column name for the potential-radiation series fed to FlagQCF — kept
#: out of the way so screening a variable literally named "SW_IN_POT" can't
#: collide with (and overwrite) the target column.
_SWINPOT_COL = "_SWINPOT_QCF_"


class StepwiseScreeningTab(DiiveTab):
    """Chain outlier tests on one variable as cards; inspect removals + QCF."""

    title = "Stepwise screening"

    #: Inspector width per page — the Report's monospace table needs more room
    #: than the Outliers/Corrections forms, so the pane widens only for it.
    _INSPECTOR_W = 300
    _INSPECTOR_W_REPORT = 560

    def build(self) -> QWidget:
        self._df = None
        self._var: str | None = None
        self._steps: list[dict] = []          # the chain ({method, kwargs, enabled})
        self._committed_steps: list[dict] = []  # last chain that ran successfully
        self._chain_dirty = False             # chain edited since the last run
        self._corr_dirty = False              # corrections edited since last apply
        self._selected_step = -1              # highlighted card (-1 = all removals)
        self._payload = None                  # last completed run
        self._corrected = None                # corrected series (or None)
        self._result_df = None                # columns pending "Add"
        self._run_id = 0                       # guards against stale worker results
        self._step_cards: list[StepCard] = []
        self._sig = _StepwiseSignals()
        self._sig.run_done.connect(self._on_done)
        self._sig.run_failed.connect(self._on_failed)
        #: Bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        layout = QHBoxLayout(root)

        # Left: the shared variable list (identical to every other tab).
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(lambda name, _ctrl: self._select(name))
        layout.addWidget(self.varpanel)

        # Centre: the segmented inspector (Outliers / Corrections / Report) —
        # only the active page takes space, so the plots get the rest.
        layout.addWidget(self._build_inspector())

        # Right: action bar + status + the large, always-visible plot stage.
        right = QWidget()
        rlayout = QVBoxLayout(right)
        rlayout.setContentsMargins(0, 0, 0, 0)
        rlayout.addLayout(self._build_top_bar())

        info_row = QHBoxLayout()
        self.status = QLabel("Select a variable to screen.")
        self.status.setWordWrap(True)
        info_row.addWidget(self.status, stretch=1)
        self.qcf_label = QLabel("QCF: add a step to compute.")
        self.qcf_label.setStyleSheet(f"color: {_C_MUTED};")
        info_row.addWidget(self.qcf_label)
        rlayout.addLayout(info_row)

        self.canvas = MplCanvas()
        rlayout.addWidget(self.canvas, stretch=1)

        layout.addWidget(right, stretch=1)
        return root

    # --- inspector (segmented: Outliers / Corrections / Report) ---
    def _build_inspector(self) -> QWidget:
        panel = QWidget()
        self._inspector = panel
        panel.setFixedWidth(self._INSPECTOR_W)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        seg = QHBoxLayout()
        seg.setSpacing(4)
        self._seg_btns: list[QPushButton] = []
        for i, label in enumerate(("Outliers", "Corrections", "Report")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _c=False, idx=i: self._set_inspector_page(idx))
            seg.addWidget(b)
            self._seg_btns.append(b)
        self._apply_segment_style()
        # Repaint the segment chips when the appearance theme changes live.
        theme.manager.changed.connect(self._apply_segment_style)
        v.addLayout(seg)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_outliers_page())
        self._stack.addWidget(self._build_corrections_page())
        self._stack.addWidget(self._build_report_page())
        v.addWidget(self._stack, stretch=1)
        self._set_inspector_page(0)
        self._refresh_inspector_badges()
        return panel

    def _apply_segment_style(self) -> None:
        accent = theme.manager.tokens.get("ACCENT", "#3A4D5C")
        border = theme.manager.tokens.get("BORDER", "#E6E6E3")
        qss = (
            f"QPushButton {{ padding: 6px 6px; border: 0.5px solid {border}; "
            f"border-radius: 6px; background: transparent; }} "
            f"QPushButton:checked {{ background: {accent}; color: white; "
            f"border-color: {accent}; }}")
        for b in self._seg_btns:
            b.setStyleSheet(qss)

    def _set_inspector_page(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        for i, b in enumerate(self._seg_btns):
            b.setChecked(i == idx)
        # Widen for the Report page (wide monospace table), narrow otherwise so
        # the plots keep the space.
        self._inspector.setFixedWidth(
            self._INSPECTOR_W_REPORT if idx == 2 else self._INSPECTOR_W)

    def _refresh_inspector_badges(self) -> None:
        """Show the step / correction counts on the segment buttons. Defensive:
        runs during the staggered build before all pages exist."""
        if not getattr(self, "_seg_btns", None):
            return
        n_steps = len(self._steps)
        self._seg_btns[0].setText(f"Outliers  {n_steps}" if n_steps else "Outliers")
        panel = getattr(self, "corrections_panel", None)
        n_corr = len(panel.corrections()) if panel is not None else 0
        self._seg_btns[1].setText(f"Corrections  {n_corr}" if n_corr else "Corrections")

    def _build_outliers_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        host = QWidget()
        # Cards now stack vertically (full settings visible, no horizontal scroll).
        self._strip = QVBoxLayout(host)
        self._strip.setContentsMargins(4, 4, 4, 4)
        self._strip.setSpacing(8)
        self._strip.setAlignment(Qt.AlignTop)
        scroll.setWidget(host)
        self._strip_host = host
        self._rebuild_cards()
        outer.addWidget(scroll, stretch=1)

        # Run the outlier chain — sits right below the method cards.
        self.run_outliers_btn = QPushButton("Run outliers")
        self.run_outliers_btn.setToolTip(
            "Run the outlier chain on the selected variable and compute the QCF. "
            "Edits to the steps apply only when you run.")
        self.run_outliers_btn.clicked.connect(lambda: self._run())
        theme.set_button_role(self.run_outliers_btn, "confirm")
        outer.addWidget(self.run_outliers_btn)
        return page

    def _build_corrections_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # No horizontal scroll: force the content to the pane width so the input
        # fields shrink to fit instead of overflowing (and clipping) the pane.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(4, 4, 4, 4)

        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Measurement:"))
        self.meas_combo = QComboBox()
        # Don't let the longest item (e.g. the PPFD description) dictate a wide
        # minimum — that would push the whole pane past its width and clip the
        # input fields. Keep a small minimum; it still stretches to fill.
        self.meas_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.meas_combo.setMinimumContentsLength(6)
        self.meas_combo.setToolTip(
            "The measurement group of this variable. It decides which "
            "corrections are physically meaningful (e.g. radiation zero offset "
            "for SW / PPFD).")
        self.meas_combo.addItem("(none)", None)
        for m in MEASUREMENTS:
            self.meas_combo.addItem(measurement_label(m.code), m.code)
        self.meas_combo.currentIndexChanged.connect(self._on_measurement_changed)
        mrow.addWidget(self.meas_combo, stretch=1)
        v.addLayout(mrow)

        hint = QLabel("Applied to the QCF-filtered series, in order.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(hint)

        self.corrections_panel = CorrectionsPanel()
        self.corrections_panel.set_coords_available(site.manager.configured)
        self.corrections_panel.changed.connect(self._on_corrections_changed)
        v.addWidget(self.corrections_panel)
        v.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll, stretch=1)

        # Apply the corrections — sits right below the correction list.
        self.run_corrections_btn = QPushButton("Run corrections")
        self.run_corrections_btn.setToolTip(
            "Apply the corrections to the QCF-filtered series (or the raw series "
            "when no outlier chain has been run).")
        self.run_corrections_btn.clicked.connect(lambda: self._recompute_corrections())
        theme.set_button_role(self.run_corrections_btn, "confirm")
        outer.addWidget(self.run_corrections_btn)
        return page

    def _build_report_page(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)
        head = QHBoxLayout()
        title = QLabel("Screening report")
        title.setStyleSheet("font-weight: bold;")
        head.addWidget(title)
        head.addStretch(1)
        self.report_copy_btn = CopyPythonButton(self._report_provider, text="Copy report")
        self.report_copy_btn.setToolTip("Copy the screening statistics to the clipboard.")
        self.report_copy_btn.setEnabled(False)
        head.addWidget(self.report_copy_btn)
        v.addLayout(head)
        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.report_text.setPlaceholderText(
            "Per-step records lost (overall + day/night) appear here once a "
            "chain runs. Configure the site (Project settings) for the "
            "day/night split.")
        self.report_text.setFont(QFont("Consolas, monospace", 9))
        v.addWidget(self.report_text, stretch=1)
        return panel

    def _build_top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.limits_cb = QCheckBox("Show limit lines")
        self.limits_cb.setToolTip(
            "Overlay the selected step's upper/lower detection band on the "
            "original series (methods with a single envelope only).")
        self.limits_cb.toggled.connect(lambda *_: self._draw_payload())
        bar.addWidget(self.limits_cb)

        bar.addStretch(1)

        self.copy_btn = CopyPythonButton(self._code_provider)
        self.copy_btn.setToolTip("Copy a reproducible script for this chain to the clipboard.")
        self.copy_btn.setEnabled(False)
        bar.addWidget(self.copy_btn)

        self.add_btn = QPushButton("Add cleaned + flags + QCF to dataset")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add_to_dataset)
        theme.set_button_role(self.add_btn, "confirm")
        bar.addWidget(self.add_btn)
        return bar

    def _on_measurement_changed(self, *_) -> None:
        code = self.meas_combo.currentData()
        self.corrections_panel.set_measurement(code)

    def _on_corrections_changed(self) -> None:
        # Like step edits, corrections are applied only on Run — just flag the
        # pending change here.
        self._corr_dirty = True
        self._refresh_run_buttons()
        self._refresh_inspector_badges()

    def _refresh_run_buttons(self) -> None:
        """Mark each section's Run button pending when it has unapplied edits.
        Defensive: may run during the staggered build before both exist."""
        out_btn = getattr(self, "run_outliers_btn", None)
        if out_btn is not None:
            out_btn.setText("Run outliers •" if self._chain_dirty else "Run outliers")
        corr_btn = getattr(self, "run_corrections_btn", None)
        if corr_btn is not None:
            corr_btn.setText(
                "Run corrections •" if self._corr_dirty else "Run corrections")

    # --- card strip ---
    def _rebuild_cards(self) -> None:
        """Rebuild the card row from ``self._steps`` (+ a trailing add card)."""
        while self._strip.count():
            item = self._strip.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._step_cards = []
        # Show per-step removal counts only when the payload still matches the
        # current chain (not edited since the run, same length).
        valid = (self._payload is not None and not self._chain_dirty
                 and len(self._payload["removed"]) == len(self._steps))
        removed = self._payload["removed"] if valid else None
        for i, step in enumerate(self._steps):
            n = len(removed[i]) if removed is not None else None
            card = StepCard(i, step, removed=n, selected=(i == self._selected_step),
                            enabled=step.get("enabled", True))
            card.clicked.connect(lambda idx=i: self._select_step(idx))
            card.edit.connect(lambda idx=i: self._edit_step(idx))
            card.delete.connect(lambda idx=i: self._delete_step(idx))
            card.move_left.connect(lambda idx=i: self._move_step(idx, -1))
            card.move_right.connect(lambda idx=i: self._move_step(idx, 1))
            card.toggle.connect(lambda on, idx=i: self._toggle_step(idx, on))
            self._strip.addWidget(card)
            self._step_cards.append(card)
        add = AddStepCard()
        add.clicked.connect(self._add_step_dialog)
        self._strip.addWidget(add)
        self._refresh_inspector_badges()

    def _apply_chain_change(self) -> None:
        """Reflect a chain edit immediately (so cards appear/disappear at once),
        but do NOT re-run — the chain is applied only when the user clicks Run.
        The cards rebuild with no stale counts (dirty); Run fills them back in."""
        self._chain_dirty = True
        self._rebuild_cards()
        self._refresh_run_buttons()

    def _add_step_dialog(self) -> None:
        if self._var is None:
            self.status.setText("Select a variable first, then add a step.")
            return
        step = StepEditorDialog.get_step(self._strip_host)
        if step is None:
            return
        step.setdefault("enabled", True)
        self._steps.append(step)
        self._selected_step = len(self._steps) - 1
        self._apply_chain_change()

    def _edit_step(self, i: int) -> None:
        if not (0 <= i < len(self._steps)):
            return
        step = StepEditorDialog.get_step(self._strip_host, self._steps[i])
        if step is None:
            return
        step.setdefault("enabled", self._steps[i].get("enabled", True))
        self._steps[i] = step
        self._selected_step = i
        self._apply_chain_change()

    def _delete_step(self, i: int) -> None:
        if not (0 <= i < len(self._steps)):
            return
        del self._steps[i]
        self._selected_step = -1
        self._apply_chain_change()

    def _move_step(self, i: int, delta: int) -> None:
        j = i + delta
        if not (0 <= i < len(self._steps)) or not (0 <= j < len(self._steps)):
            return
        self._steps[i], self._steps[j] = self._steps[j], self._steps[i]
        self._selected_step = j
        self._apply_chain_change()

    def _toggle_step(self, i: int, on: bool) -> None:
        if not (0 <= i < len(self._steps)):
            return
        # Replace (don't mutate in place): the committed-steps snapshot may share
        # this dict, and an in-place edit would leak into the rollback copy.
        self._steps[i] = {**self._steps[i], "enabled": on}
        self._apply_chain_change()

    def _select_step(self, i: int) -> None:
        self._selected_step = -1 if i == self._selected_step else i
        for k, card in enumerate(self._step_cards):
            card.setSelected(k == self._selected_step)
        self._draw_payload()

    # --- data ---
    def on_data_loaded(self, df, created: set | None = None) -> None:
        prev_df = self._df
        self._df = df
        self.corrections_panel.set_coords_available(site.manager.configured)
        self.varpanel.set_variables(df.columns, created)
        if self._var is not None and self._var in df.columns:
            # A push leaves the variable's series untouched when only columns were
            # added (e.g. our own "Add"); the index changes when the data really
            # changed (range subselection, a new dataset). Keep the run in the
            # former case; treat it as stale in the latter. (Index equality is a
            # cheap proxy — in-place value edits to a column don't happen here.)
            unchanged = (prev_df is not None and self._var in prev_df.columns
                         and prev_df[self._var].index.equals(df[self._var].index))
            if unchanged:
                self.varpanel.set_panels([self._var])
            else:
                # Stale run: reset + redraw raw (the user re-runs to recompute).
                # Keep the measurement — don't re-detect on every push and discard
                # a manual choice.
                self._show_variable(self._var, redetect_measurement=False)
        elif len(df.columns):
            # No variable carried over: select the first so the list and the
            # preview agree.
            self._select(str(df.columns[0]))
        else:
            self._var = None
            self._result_df = None
            self.add_btn.setEnabled(False)

    def _enabled_steps(self) -> list[dict]:
        return [s for s in self._steps if s.get("enabled", True)]

    def _select(self, name: str) -> None:
        if not name or self._df is None or name not in self._df.columns:
            return
        self._show_variable(name, redetect_measurement=True)

    def _show_variable(self, name: str, *, redetect_measurement: bool) -> None:
        """Make `name` the active variable: clear any prior run (it's now stale),
        show the raw series, and leave the chain + corrections to be (re)applied
        on Run. `redetect_measurement` re-guesses the measurement from the name
        (on a user pick) or keeps the current one (on a data reload)."""
        self._var = name
        self.varpanel.set_panels([name])
        # Nothing is computed until the user clicks Run — only the raw series.
        self._payload = None
        self._corrected = None
        self._result_df = None
        self._selected_step = -1
        if redetect_measurement:
            self._apply_detected_measurement(name)
        self._rebuild_cards()
        self.qcf_label.setText("QCF: run to compute.")
        self.report_text.clear()
        self.report_copy_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        # Pending if there is anything to apply on this variable.
        self._chain_dirty = bool(self._enabled_steps())
        self._corr_dirty = bool(self.corrections_panel.corrections())
        self.copy_btn.setEnabled(self._code_provider() is not None)
        self._draw_raw(self._df[name])
        self._refresh_run_buttons()
        if self._chain_dirty or self._corr_dirty:
            self.status.setText(f"Selected '{name}'. Click Run chain to apply.")
        else:
            self.status.setText(f"Selected '{name}'. Add outlier tests to build a chain.")

    def _apply_detected_measurement(self, name: str) -> None:
        """Auto-pick the measurement for this variable from its name (the user
        can still override via the dropdown)."""
        code = detect_measurement(name)
        idx = self.meas_combo.findData(code)
        self.meas_combo.setCurrentIndex(idx if idx >= 0 else 0)

    # --- run (rebuild + replay on a worker thread) ---
    @staticmethod
    def _coords() -> dict:
        m = site.manager
        if m.configured:
            return dict(site_lat=m.latitude, site_lon=m.longitude,
                        utc_offset=m.utc_offset)
        return dict(site_lat=0.0, site_lon=0.0, utc_offset=0)

    def _base_series(self):
        """The series the corrections start from: the QCF-filtered series after
        a chain run, else the raw variable (so corrections work standalone)."""
        if self._payload is not None:
            return self._payload["qcf"].filteredseries
        if self._var is not None and self._df is not None:
            return self._df[self._var]
        return None

    def _recompute_corrections(self) -> None:
        """Apply the enabled corrections to the current base series, then refresh
        the result columns, the 'Add' button, the preview, and the copy button.
        Synchronous — corrections are cheap and don't touch the outlier chain."""
        corrs = self.corrections_panel.corrections()
        base = self._base_series()
        applied = True
        if not corrs or base is None:
            self._corrected = None
        else:
            coords = self._coords()
            try:
                self._corrected = apply_corrections(
                    base, corrs, lat=coords["site_lat"], lon=coords["site_lon"],
                    utc_offset=coords["utc_offset"])
            except Exception as err:
                self._corrected = None
                self.status.setText(f"Correction failed: {err}")
                applied = False  # keep the pending dot so the user can retry
        if applied:
            self._corr_dirty = False
        self._build_result()
        self.add_btn.setEnabled(self._result_df is not None
                                and not self._result_df.empty)
        self.copy_btn.setEnabled(self._code_provider() is not None)
        self._refresh_run_buttons()
        self._draw_payload()

    def _run(self) -> None:
        if self._df is None or self._var is None:
            return
        # Bump the run id so any in-flight worker's result is ignored on arrival
        # (results can land out of order under rapid edits).
        self._run_id += 1
        run_id = self._run_id
        # Nothing to screen (no steps, or all toggled off): show the raw series.
        # Corrections may still apply (on the raw series), so route through
        # _recompute_corrections rather than drawing raw directly.
        if not self._enabled_steps():
            self._payload = None
            self._chain_dirty = False
            self.report_text.clear()
            self.report_copy_btn.setEnabled(False)
            self.qcf_label.setText("QCF: add a step to compute.")
            self._recompute_corrections()
            return
        self.status.setText("Running chain…")
        self.add_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        threading.Thread(
            target=self._worker,
            args=(self._df, self._var, list(self._steps), self._coords(),
                  site.manager.configured, run_id),
            daemon=True).start()

    def _worker(self, df, var, steps, coords, configured, run_id) -> None:
        try:
            # output_middle_timestamp=False keeps the input index so the emitted
            # columns align to the app dataframe on merge.
            det = StepwiseOutlierDetection(dfin=df[[var]], col=var,
                                           output_middle_timestamp=False, **coords)
            removed = []  # per-step Index of newly-removed timestamps
            bounds = []   # per-step (lower, upper) detection band (or (None, None))
            prev = det.series_hires_orig
            for step in steps:
                # Skip toggled-off steps; keep aligned placeholders so the
                # removed/bounds lists index 1:1 with the cards.
                if not step.get("enabled", True):
                    removed.append(prev.index[:0])
                    bounds.append((None, None))
                    continue
                getattr(det, step["method"])(**step.get("kwargs", {}))
                bounds.append(det.last_bounds)
                det.addflag()
                # Snapshot: addflag mutates the detector's cleaned series in
                # place, so prev must be a copy or later steps would alias (and
                # mutate) it, making their per-step diff come out empty.
                clean = det.series_hires_cleaned.copy()
                newly = prev[prev.notna() & clean.reindex(prev.index).isna()].index
                removed.append(newly)
                prev = clean
            # Overall QCF from the accumulated test flags (aligned on the orig index).
            qcf_input = pd.concat([det.series_hires_orig.to_frame(var), det.flags], axis=1)
            # SW_IN_POT (from site coords) enables the QCF day/night split and the
            # day/night breakdown in the screening report; skip if site unset.
            swinpot_col = None
            if configured:
                qcf_input[_SWINPOT_COL] = dv.variables.potrad(
                    qcf_input.index, coords["site_lat"], coords["site_lon"],
                    coords["utc_offset"])
                swinpot_col = _SWINPOT_COL
            qcf = FlagQCF(df=qcf_input, target_col=var, idstr="STEPWISE",
                          swinpot_col=swinpot_col)
            qcf.calculate()
            _, report = qcf.screening_report()
            payload = {"var": var, "detector": det, "removed": removed,
                       "bounds": bounds, "qcf": qcf, "report": report,
                       "run_id": run_id}
        except Exception as err:
            self._sig.run_failed.emit((run_id, str(err)))
            return
        self._sig.run_done.emit(payload)

    def _on_done(self, payload: dict) -> None:
        # Ignore a stale result from a superseded run (out-of-order completion).
        if payload.get("run_id") != self._run_id:
            return
        self._payload = payload
        self._committed_steps = copy.deepcopy(self._steps)
        self._chain_dirty = False
        det = payload["detector"]
        if self._selected_step >= len(self._steps):
            self._selected_step = -1
        self._rebuild_cards()
        # QCF distribution (the separate overall-flag surface).
        qcf = payload["qcf"]
        vc = qcf.flagqcf.value_counts()
        kept = int(qcf.filteredseries.dropna().count())
        self.qcf_label.setText(
            f"QCF — 0:{int(vc.get(0, 0))}  1:{int(vc.get(1, 0))}  "
            f"2:{int(vc.get(2, 0))}  ·  kept {kept}")
        total_removed = int(det.series_hires_orig.notna().sum()
                            - det.series_hires_cleaned.notna().sum())
        self.status.setText(
            f"{len(self._steps)} step(s), {total_removed} points removed overall. "
            f"Click a card to highlight just its removals.")
        self.report_text.setPlainText(payload.get("report", ""))
        self.report_copy_btn.setEnabled(bool(payload.get("report")))
        # Applies corrections on the fresh QCF series, builds result columns,
        # toggles the Add/Copy buttons, and draws the preview.
        self._recompute_corrections()

    def _on_failed(self, data) -> None:
        run_id, msg = data
        if run_id != self._run_id:
            return  # a superseded run failed; the latest run governs the UI
        self.status.setText(f"Failed: {msg}")
        # Restore the last chain that ran cleanly so the UI stays consistent.
        self._steps = copy.deepcopy(self._committed_steps)
        self._selected_step = -1
        self._chain_dirty = False
        self._rebuild_cards()
        self._refresh_run_buttons()

    # --- result columns for "Add" ---
    def _build_result(self) -> None:
        """Build the columns the 'Add' button emits from the current run +
        corrections: all FLAG_*_TEST columns, the overall QCF flag, the
        QCF-filtered series, and (if any corrections are active) the corrected
        series. Works with only corrections (no outlier chain), and with only an
        outlier chain (no corrections)."""
        payload = self._payload
        pieces = []
        attrs = {}
        params = {"steps": copy.deepcopy(self._steps),
                  "corrections": copy.deepcopy(self.corrections_panel.corrections())}
        corrected_parent = self._var

        if payload is not None:
            det = payload["detector"]
            qcf = payload["qcf"]
            var = payload["var"]
            pieces += [det.flags,
                       qcf.flagqcf.rename(qcf.flagqcfcol),
                       qcf.filteredseries.rename(qcf.filteredseriescol)]
            attrs[qcf.filteredseriescol] = provenance_attr(
                origin=MODIFIED, parent=var, operation="Stepwise screening",
                params=params, tags=["outliers-removed", "qcf"])
            for c in det.flags.columns:
                attrs[str(c)] = provenance_attr(
                    origin=DERIVED, parent=var, operation="Stepwise screening flag",
                    params=params, tags=["flag"])
            attrs[qcf.flagqcfcol] = provenance_attr(
                origin=DERIVED, parent=var, operation="Stepwise screening QCF",
                params=params, tags=["flag", "qcf"])
            corrected_parent = qcf.filteredseriescol

        if self._corrected is not None and self._var is not None:
            corrected_col = f"{self._var}_CORRECTED"
            pieces.append(self._corrected.rename(corrected_col))
            attrs[corrected_col] = provenance_attr(
                origin=MODIFIED, parent=corrected_parent, operation="Meteo corrections",
                params=params, tags=["corrected"])

        if not pieces:
            self._result_df = None
            return
        cols = pd.concat(pieces, axis=1)
        cols.attrs[ATTRS_KEY] = attrs
        self._result_df = cols

    def _add_to_dataset(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        cols = list(self._result_df.columns)
        parts = []
        if self._payload is not None:
            parts.append("flags + QCF + filtered series")
        if any(str(c).endswith("_CORRECTED") for c in cols):
            parts.append("corrected series")
        self.featuresCreated.emit(self._result_df)
        self.status.setText(
            f"Added {len(cols)} column(s) ({', '.join(parts)}) to the variable list.")
        self.add_btn.setEnabled(False)

    def _code_provider(self) -> str | None:
        """Render the chain (enabled steps only) as a runnable script, or None if
        there is nothing to copy. The CopyPythonButton handles clipboard + the
        'Copied ✓' flash."""
        enabled = self._enabled_steps()
        if not enabled or self._var is None:
            return None
        return stepwise_to_code(enabled, var_name=self._var, **self._coords(),
                                load_hint="dv.load_parquet('your_data.parquet')",
                                corrections=self.corrections_panel.corrections())

    def _report_provider(self) -> str | None:
        """The current screening report text (or None), for the Copy button."""
        if self._payload is None:
            return None
        return self._payload.get("report") or None

    # --- preview ---
    def _draw_raw(self, series) -> None:
        """Single panel with the original series (before any step runs), plus the
        corrected series overlaid when corrections are active."""
        ax = self.canvas.new_axes(1)[0]
        ax.plot(series.index, series.to_numpy(), color=_C_RAW, lw=0.5, alpha=0.85,
                label="original")
        if self._corrected is not None:
            ax.plot(self._corrected.index, self._corrected.to_numpy(),
                    color=_C_CORRECTED, lw=0.7, alpha=0.9, label="corrected")
            ax.legend(loc="best", fontsize=7, framealpha=0.9)
            ax.set_title(f"{series.name} — original + corrected", fontsize=9)
        else:
            ax.set_title(f"{series.name} — original (add a step)", fontsize=9)
        self.canvas.draw()

    def _draw_payload(self) -> None:
        """The preview grid for the current run, honouring the selected step and
        the limit-line toggle. Top: a short time series with removals + the
        cleaned series. Bottom: cleaned heatmap, QCF heatmap, QCF distribution."""
        if self._payload is None:
            if self._var is not None and self._df is not None:
                self._draw_raw(self._df[self._var])
            return
        det = self._payload["detector"]
        removed = self._payload["removed"]
        bounds = self._payload["bounds"]
        qcf = self._payload["qcf"]
        orig = det.series_hires_orig
        cleaned = det.series_hires_cleaned
        sel = self._selected_step

        self.canvas.reset_layout()
        fig = self.canvas.fig
        # Small time series up top (short rows), the three map/dist panels below.
        gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 0.8, 1.6])
        ax_ts = fig.add_subplot(gs[0, :])
        ax_clean = fig.add_subplot(gs[1, :], sharex=ax_ts)
        ax_heat = fig.add_subplot(gs[2, 0])
        ax_qcf_heat = fig.add_subplot(gs[2, 1])
        ax_hist = fig.add_subplot(gs[2, 2])

        # 1) Original + removals (selected step's, else cumulative).
        ax_ts.plot(orig.index, orig.to_numpy(), color=_C_RAW, lw=0.5, alpha=0.8,
                   label="original", zorder=1)
        if 0 <= sel < len(removed):
            idx = removed[sel]
            extra = f"step {sel + 1} removals ({len(idx)})"
        else:
            idx = orig.index[orig.notna() & cleaned.reindex(orig.index).isna()]
            extra = f"all removals ({len(idx)})"
        pts = orig.reindex(idx)
        ax_ts.plot(pts.index, pts.to_numpy(), linestyle="none", marker="o",
                   color=_C_REMOVED, ms=3.5, markeredgecolor="none", alpha=0.85,
                   zorder=5, label=extra)
        # Optional detection band for the selected step (or the last step).
        if self.limits_cb.isChecked():
            band_step = sel if sel >= 0 else len(bounds) - 1
            if 0 <= band_step < len(bounds):
                lo, hi = bounds[band_step]
                if lo is not None and hi is not None:
                    ax_ts.plot(lo.index, lo.to_numpy(), color=_C_LIMIT, lw=0.7,
                               linestyle="--", alpha=0.8, label="limits", zorder=4)
                    ax_ts.plot(hi.index, hi.to_numpy(), color=_C_LIMIT, lw=0.7,
                               linestyle="--", alpha=0.8, zorder=4)
        ax_ts.set_title(f"{orig.name} — original + {extra}", fontsize=9)
        ax_ts.legend(loc="best", fontsize=7, framealpha=0.9)

        # 2) Cleaned series (+ corrected overlay when corrections are active).
        ax_clean.plot(cleaned.index, cleaned.to_numpy(), color=_C_CLEANED, lw=0.7,
                      label="cleaned")
        if self._corrected is not None:
            ax_clean.plot(self._corrected.index, self._corrected.to_numpy(),
                          color=_C_CORRECTED, lw=0.7, alpha=0.9, label="corrected")
            ax_clean.legend(loc="best", fontsize=7, framealpha=0.9)
            ax_clean.set_title("cleaned + corrected", fontsize=9)
        else:
            ax_clean.set_title("cleaned (outliers removed)", fontsize=9)

        # 3) Heatmap of the cleaned series.
        try:
            dv.plotting.HeatmapDateTime(cleaned).plot(
                ax=ax_heat, fig=fig, title="cleaned", cb_digits_after_comma="auto",
                axlabels_fontsize=8, ticks_labelsize=7, cb_labelsize=7)
            ax_heat.title.set_fontsize(9)
        except Exception as err:
            ax_heat.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                         wrap=True, transform=ax_heat.transAxes)

        # 4) Heatmap of the QCF flag (when/where records were rejected).
        flag = qcf.flagqcf
        try:
            dv.plotting.HeatmapDateTime(flag.rename("QCF")).plot(
                ax=ax_qcf_heat, fig=fig, title="QCF (0/1/2)", vmin=0, vmax=2,
                cmap=_QCF_CMAP, cb_digits_after_comma=0,
                axlabels_fontsize=8, ticks_labelsize=7, cb_labelsize=7)
            ax_qcf_heat.title.set_fontsize(9)
        except Exception as err:
            ax_qcf_heat.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center",
                             va="center", wrap=True, transform=ax_qcf_heat.transAxes)

        # 5) QCF distribution.
        vc = flag.value_counts()
        levels = [0, 1, 2]
        ax_hist.bar(levels, [int(vc.get(lv, 0)) for lv in levels],
                    color=[_QCF_COLORS[lv] for lv in levels])
        ax_hist.set_xticks(levels)
        ax_hist.set_title("QCF distribution", fontsize=9)

        self.canvas.draw()

    # --- state ---
    def save_state(self) -> dict:
        return {"var": self._var, "steps": self._steps,
                "limits": self.limits_cb.isChecked(),
                "corrections": self.corrections_panel.get_state(),
                "inspector": self._stack.currentIndex()}

    def restore_state(self, state: dict) -> None:
        self._steps = list(state.get("steps") or [])
        self.limits_cb.setChecked(bool(state.get("limits")))
        page = state.get("inspector")
        if isinstance(page, int) and 0 <= page < self._stack.count():
            self._set_inspector_page(page)
        # Restore the corrections + measurement, syncing the dropdown to match
        # (signals blocked so it doesn't trigger a premature recompute).
        corr_state = state.get("corrections")
        if corr_state:
            self.corrections_panel.blockSignals(True)
            self.corrections_panel.set_state(corr_state)
            self.corrections_panel.blockSignals(False)
            self.meas_combo.blockSignals(True)
            idx = self.meas_combo.findData(self.corrections_panel.measurement())
            self.meas_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.meas_combo.blockSignals(False)
        var = state.get("var")
        if var and self._df is not None and var in self._df.columns:
            self._var = var
            self.varpanel.set_panels([str(var)])
            self._rebuild_cards()
            if self._steps:
                self._run()  # replays the saved chain (then applies corrections)
            else:
                self._recompute_corrections()
