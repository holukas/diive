"""
GUI.TABS.STEPWISE: STEPWISE OUTLIER SCREENING TAB
=================================================

Chain several outlier tests on one variable and inspect what each step removes.
Unlike the single-method Outliers tabs (one detector, one pass), this drives the
library's `StepwiseOutlierDetection`: each committed step runs on the data the
previous steps already cleaned, so spikes are peeled off progressively. The
overall **QCF** (`dv.qaqc.FlagQCF`) is computed separately from the accumulated
per-test flags and shown on its own — then the cleaned series, the flags, and the
QCF-filtered series can be added to the dataset.

The per-method parameter widgets are the shared
`widgets/stepwise_method_params.py` registry (a step is a `{"method", "kwargs"}`
dict). The chain is rebuilt from scratch and replayed on every edit so steps can
be removed in any order; per-step removals are the diff of the cleaned series
before/after each step. All detection + QCF is library work; this tab only
collects parameters, runs them on a worker thread, previews, and emits columns.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, MODIFIED, provenance_attr
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.stepwise_method_params import STEP_METHOD_BY_KEY, method_labels
from diive.gui.widgets.variable_panel import VariablePanel
from diive.preprocessing.outlier_detection import StepwiseOutlierDetection
from diive.qaqc import FlagQCF

_C_RAW = "#B0BEC5"      # blue-grey 200 — original line
_C_CLEANED = "#43A047"  # green 600     — cleaned series
_C_REMOVED = "#E53935"  # red 600       — removed points
_C_MUTED = "#6B7780"


class _StepwiseSignals(QObject):
    run_done = Signal(object)
    run_failed = Signal(str)
    features_created = Signal(object)


class StepwiseScreeningTab(DiiveTab):
    """Chain outlier tests on one variable; inspect per-step removals + QCF."""

    title = "Stepwise screening"

    def build(self) -> QWidget:
        self._df = None
        self._var: str | None = None
        self._steps: list[dict] = []          # committed chain ({method, kwargs})
        self._payload = None                  # last completed run
        self._result_df = None                # columns pending "Add"
        self._sig = _StepwiseSignals()
        self._sig.run_done.connect(self._on_done)
        self._sig.run_failed.connect(self._on_failed)
        #: Bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        layout = QHBoxLayout(root)

        # Left: pick the variable to screen.
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(lambda name, _ctrl: self._select(name))
        layout.addWidget(self.varpanel)

        # Middle: chain builder + QCF.
        mid = self._build_chain_panel()
        mid.setFixedWidth(300)
        layout.addWidget(mid)

        # Right: two-panel preview (original + removals / cleaned).
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas, stretch=1)
        return root

    def _build_chain_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)

        intro = QLabel("Add outlier tests one by one; each runs on the data the "
                       "previous steps cleaned. Select a step to see what it removed.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        outer.addWidget(intro)

        # Method picker → swaps in that method's parameter widget.
        build_box = QGroupBox("Add a step")
        bb = QVBoxLayout(build_box)
        self.method = QComboBox()
        for key, label in method_labels():
            self.method.addItem(label, key)
        self.method.currentIndexChanged.connect(self._on_method_changed)
        bb.addWidget(self.method)
        self._param_box = QVBoxLayout()
        self._param_widget = None
        bb.addLayout(self._param_box)
        add_btn = QPushButton("Add step")
        add_btn.clicked.connect(self._add_step)
        bb.addWidget(add_btn)
        outer.addWidget(build_box)
        self._on_method_changed()  # seed the first param widget

        # The committed chain.
        chain_box = QGroupBox("Chain")
        cb = QVBoxLayout(chain_box)
        self.steps_list = QListWidget()
        self.steps_list.currentRowChanged.connect(self._on_step_selected)
        cb.addWidget(self.steps_list)
        row = QHBoxLayout()
        rm = QPushButton("Remove selected")
        rm.clicked.connect(self._remove_step)
        row.addWidget(rm)
        clr = QPushButton("Clear")
        clr.clicked.connect(self._clear_steps)
        row.addWidget(clr)
        cb.addLayout(row)
        outer.addWidget(chain_box)

        # The overall quality flag (computed separately from the test flags).
        self.qcf_label = QLabel("QCF: run a step to compute.")
        self.qcf_label.setWordWrap(True)
        qcf_box = QGroupBox("Overall quality flag (QCF)")
        qb = QVBoxLayout(qcf_box)
        qb.addWidget(self.qcf_label)
        outer.addWidget(qcf_box)

        self.status = QLabel("Select a variable on the left.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.add_btn = QPushButton("Add cleaned + flags + QCF to dataset")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add_to_dataset)
        theme.set_button_role(self.add_btn, "confirm")
        outer.addWidget(self.add_btn)

        outer.addStretch(1)
        return panel

    # --- method picker ---
    def _on_method_changed(self, *_) -> None:
        key = self.method.currentData()
        if self._param_widget is not None:
            self._param_box.removeWidget(self._param_widget)
            self._param_widget.deleteLater()
        self._param_widget = STEP_METHOD_BY_KEY[key]()
        self._param_box.addWidget(self._param_widget)

    # --- chain editing (each edit replays the whole chain) ---
    def _add_step(self) -> None:
        if self._param_widget is None or self._var is None:
            self.status.setText("Select a variable first, then add a step.")
            return
        self._steps.append(self._param_widget.step())
        self._run()

    def _remove_step(self) -> None:
        row = self.steps_list.currentRow()
        if row < 0:
            return
        del self._steps[row]
        self._run() if self._steps else self._reset_preview()

    def _clear_steps(self) -> None:
        self._steps = []
        self._reset_preview()

    def _reset_preview(self) -> None:
        self.steps_list.clear()
        self.qcf_label.setText("QCF: run a step to compute.")
        self.add_btn.setEnabled(False)
        self._result_df = None
        self._payload = None
        if self._var is not None and self._df is not None:
            self._draw(self._df[self._var])

    # --- data ---
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._result_df = None
        self.varpanel.set_variables(df.columns, created)
        self.add_btn.setEnabled(False)
        if self._var is not None and self._var not in df.columns:
            self._var = None

    def _select(self, name: str) -> None:
        if not name or self._df is None:
            return
        self._var = name
        self.varpanel.set_panels([name])
        self._steps = []
        self._reset_preview()
        self.status.setText(f"Selected '{name}'. Add outlier tests to build a chain.")
        self.varpanel.run_with_loading(name, lambda: self._draw(self._df[name]))

    # --- run (rebuild + replay on a worker thread) ---
    @staticmethod
    def _coords() -> dict:
        m = site.manager
        if m.configured:
            return dict(site_lat=m.latitude, site_lon=m.longitude,
                        utc_offset=m.utc_offset)
        return dict(site_lat=0.0, site_lon=0.0, utc_offset=0)

    def _run(self) -> None:
        if self._df is None or self._var is None or not self._steps:
            return
        self.status.setText("Running chain…")
        self.add_btn.setEnabled(False)
        threading.Thread(
            target=self._worker,
            args=(self._df, self._var, list(self._steps), self._coords()),
            daemon=True).start()

    def _worker(self, df, var, steps, coords) -> None:
        try:
            # output_middle_timestamp=False keeps the input index so the emitted
            # columns align to the app dataframe on merge.
            det = StepwiseOutlierDetection(dfin=df[[var]], col=var,
                                           output_middle_timestamp=False, **coords)
            removed = []  # per-step Index of newly-removed timestamps
            prev = det.series_hires_orig
            for step in steps:
                getattr(det, step["method"])(**step.get("kwargs", {}))
                det.addflag()
                clean = det.series_hires_cleaned
                newly = prev[prev.notna() & clean.reindex(prev.index).isna()].index
                removed.append(newly)
                prev = clean
            # Overall QCF from the accumulated test flags (aligned on the orig index).
            # idstr only names the QCF output columns (FLAG_STEPWISE_<var>_QCF,
            # <var>_STEPWISE_QCF) — it does not affect which test flags aggregate.
            qcf_input = pd.concat([det.series_hires_orig.to_frame(var), det.flags], axis=1)
            qcf = FlagQCF(df=qcf_input, target_col=var, idstr="STEPWISE")
            qcf.calculate()
            payload = {"var": var, "detector": det, "removed": removed, "qcf": qcf}
        except Exception as err:
            self._sig.run_failed.emit(str(err))
            return
        self._sig.run_done.emit(payload)

    def _on_done(self, payload: dict) -> None:
        self._payload = payload
        det = payload["detector"]
        removed = payload["removed"]
        var = payload["var"]
        # Refresh the step list with per-step removal counts.
        self.steps_list.blockSignals(True)
        self.steps_list.clear()
        for step, idx in zip(self._steps, removed):
            label = STEP_METHOD_BY_KEY[step["method"]].label
            self.steps_list.addItem(f"{label} — {len(idx)} removed")
        self.steps_list.blockSignals(False)
        # QCF distribution (the separate overall-flag surface).
        qcf = payload["qcf"]
        vc = qcf.flagqcf.value_counts()
        kept = int(qcf.filteredseries.dropna().count())
        self.qcf_label.setText(
            f"QCF — good (0): {int(vc.get(0, 0))}, marginal (1): {int(vc.get(1, 0))}, "
            f"rejected (2): {int(vc.get(2, 0))}. Kept (QCF<2): {kept}.")
        total_removed = int(det.series_hires_orig.notna().sum()
                            - det.series_hires_cleaned.notna().sum())
        self.status.setText(
            f"{len(self._steps)} step(s), {total_removed} points removed overall. "
            f"Select a step to highlight its removals.")
        self._build_result(payload)
        self.add_btn.setEnabled(True)
        self._draw_payload(selected=-1)

    def _on_failed(self, msg: str) -> None:
        self.status.setText(f"Failed: {msg}")
        # Roll back the step that caused the failure so the chain stays runnable.
        if self._steps:
            self._steps.pop()

    def _on_step_selected(self, row: int) -> None:
        if self._payload is not None:
            self._draw_payload(selected=row)

    # --- result columns for "Add" ---
    def _build_result(self, payload: dict) -> None:
        det = payload["detector"]
        qcf = payload["qcf"]
        var = payload["var"]
        # Emit: all FLAG_*_TEST columns + the overall QCF flag + the QCF-filtered series.
        cols = pd.concat([det.flags,
                          qcf.flagqcf.rename(qcf.flagqcfcol),
                          qcf.filteredseries.rename(qcf.filteredseriescol)], axis=1)
        params = {"steps": self._steps}
        attrs = {qcf.filteredseriescol: provenance_attr(
            origin=MODIFIED, parent=var, operation="Stepwise screening",
            params=params, tags=["outliers-removed", "qcf"])}
        for c in det.flags.columns:
            attrs[str(c)] = provenance_attr(
                origin=DERIVED, parent=var, operation="Stepwise screening flag",
                params=params, tags=["flag"])
        attrs[qcf.flagqcfcol] = provenance_attr(
            origin=DERIVED, parent=var, operation="Stepwise screening QCF",
            params=params, tags=["flag", "qcf"])
        cols.attrs[ATTRS_KEY] = attrs
        self._result_df = cols

    def _add_to_dataset(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        self.featuresCreated.emit(self._result_df)
        self.status.setText(
            f"Added {self._result_df.shape[1]} columns "
            f"(flags + QCF + filtered series) to the variable list.")
        self.add_btn.setEnabled(False)

    # --- preview ---
    def _draw(self, series, cleaned=None) -> None:
        """Two stacked panels: top = original (+ removals once run), bottom = cleaned."""
        self.canvas.reset_layout()
        fig = self.canvas.fig
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)
        ax_top.plot(series.index, series.to_numpy(), color=_C_RAW, lw=0.5, alpha=0.8,
                    label="original", zorder=1)
        ax_top.set_title(f"{series.name} — original", fontsize=9)
        if cleaned is not None:
            ax_bot.plot(cleaned.index, cleaned.to_numpy(), color=_C_CLEANED, lw=0.8)
            ax_bot.set_title("cleaned (outliers removed)", fontsize=9)
        else:
            ax_bot.set_title("cleaned (add a step)", fontsize=9)
        self._ax_top, self._ax_bot = ax_top, ax_bot
        self.canvas.draw()

    def _draw_payload(self, selected: int) -> None:
        """Redraw with removals highlighted: a selected step shows only its own
        removals; otherwise all removed points are shown (cumulative)."""
        if self._payload is None:
            return
        det = self._payload["detector"]
        removed = self._payload["removed"]
        orig = det.series_hires_orig
        cleaned = det.series_hires_cleaned
        self._draw(orig, cleaned=cleaned)
        if 0 <= selected < len(removed):
            idx = removed[selected]
            title_extra = f"step {selected + 1} removals ({len(idx)})"
        else:
            idx = orig.index[orig.notna() & cleaned.reindex(orig.index).isna()]
            title_extra = f"all removals ({len(idx)})"
        pts = orig.reindex(idx)
        self._ax_top.plot(pts.index, pts.to_numpy(), linestyle="none", marker="o",
                          color=_C_REMOVED, ms=4, markeredgecolor="none", alpha=0.85,
                          zorder=5, label=title_extra)
        self._ax_top.legend(loc="best", fontsize=8, framealpha=0.9)
        self._ax_top.set_title(f"{orig.name} — original + {title_extra}", fontsize=9)
        self.canvas.draw()

    # --- state ---
    def save_state(self) -> dict:
        return {"var": self._var, "steps": self._steps}

    def restore_state(self, state: dict) -> None:
        self._steps = list(state.get("steps") or [])
        var = state.get("var")
        if var and self._df is not None and var in self._df.columns:
            self._var = var
            self.varpanel.set_panels([var])
            if self._steps:
                self._run()  # replays the saved chain
            else:
                self._draw(self._df[var])
