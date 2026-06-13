"""
GUI.TABS.STEPWISE: STEPWISE OUTLIER SCREENING TAB
=================================================

Chain several outlier tests on one variable as a row of editable **method
cards** and inspect what each step removes. Unlike the single-method Outliers
tabs (one detector, one pass), this drives the library's
`StepwiseOutlierDetection`: each committed step runs on the data the previous
steps already cleaned, so spikes are peeled off progressively. The overall
**QCF** (`dv.qaqc.FlagQCF`) is computed separately from the accumulated per-test
flags.

Layout: a compact top bar (variable picker + limit-line toggle + actions), a
horizontal strip of method cards (add / edit / reorder / delete — any change
rebuilds and replays the chain on a worker thread), and a four-panel plot grid
below: original + outliers (optionally with the selected step's detection band),
the cleaned series, a date/time heatmap of the cleaned series, and the QCF flag
over time plus its 0/1/2 distribution.

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
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, MODIFIED, provenance_attr
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.stepwise_cards import (
    AddStepCard,
    StepCard,
    StepEditorDialog,
)
from diive.preprocessing.outlier_detection import StepwiseOutlierDetection
from diive.preprocessing.outlier_detection.codegen import stepwise_to_code
from diive.qaqc import FlagQCF

_C_RAW = "#B0BEC5"      # blue-grey 200 — original line
_C_CLEANED = "#43A047"  # green 600     — cleaned series
_C_REMOVED = "#E53935"  # red 600       — removed points
_C_LIMIT = "#8E24AA"    # purple 600    — detection band
_C_MUTED = "#6B7780"
_QCF_COLORS = {0: "#43A047", 1: "#FFB300", 2: "#E53935"}  # good / marginal / rejected


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

    def build(self) -> QWidget:
        self._df = None
        self._var: str | None = None
        self._steps: list[dict] = []          # the chain ({method, kwargs, enabled})
        self._committed_steps: list[dict] = []  # last chain that ran successfully
        self._chain_dirty = False             # chain edited since the last run
        self._selected_step = -1              # highlighted card (-1 = all removals)
        self._payload = None                  # last completed run
        self._result_df = None                # columns pending "Add"
        self._run_id = 0                       # guards against stale worker results
        self._step_cards: list[StepCard] = []
        self._sig = _StepwiseSignals()
        self._sig.run_done.connect(self._on_done)
        self._sig.run_failed.connect(self._on_failed)
        #: Bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        layout = QVBoxLayout(root)

        layout.addLayout(self._build_top_bar())
        layout.addWidget(self._build_card_strip())

        # Status + QCF readout line.
        info_row = QHBoxLayout()
        self.status = QLabel("Select a variable to screen.")
        self.status.setWordWrap(True)
        info_row.addWidget(self.status, stretch=1)
        self.qcf_label = QLabel("QCF: add a step to compute.")
        self.qcf_label.setStyleSheet(f"color: {_C_MUTED};")
        info_row.addWidget(self.qcf_label)
        layout.addLayout(info_row)

        # Lower area: plot grid (left) + publication report (right).
        lower = QHBoxLayout()
        self.canvas = MplCanvas()
        lower.addWidget(self.canvas, stretch=1)
        lower.addWidget(self._build_report_panel())
        layout.addLayout(lower, stretch=1)
        return root

    def _build_report_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(390)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
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
        bar.addWidget(QLabel("Variable:"))
        self.var_combo = QComboBox()
        self.var_combo.setMinimumWidth(220)
        self.var_combo.currentIndexChanged.connect(self._on_var_changed)
        bar.addWidget(self.var_combo)

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

    def _build_card_strip(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(160)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        self._strip = QHBoxLayout(host)
        self._strip.setContentsMargins(6, 6, 6, 6)
        self._strip.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        scroll.setWidget(host)
        self._strip_host = host
        self._rebuild_cards()
        return scroll

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

    def _apply_chain_change(self) -> None:
        """Reflect a chain edit immediately (so cards appear/disappear at once),
        then re-run. The cards rebuild with no stale counts (dirty), and the run
        fills them back in on completion."""
        self._chain_dirty = True
        self._rebuild_cards()
        self._run()

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
        self._df = df
        self._result_df = None
        self.add_btn.setEnabled(False)
        self.var_combo.blockSignals(True)
        self.var_combo.clear()
        self.var_combo.addItems([str(c) for c in df.columns])
        if self._var is not None and self._var in df.columns:
            self.var_combo.setCurrentText(str(self._var))
        else:
            self._var = None
        self.var_combo.blockSignals(False)
        # No variable carried over: select the first so the combo and the
        # preview agree (otherwise the combo shows row 0 but nothing is plotted).
        if self._var is None and len(df.columns):
            self._select(str(df.columns[0]))

    def _on_var_changed(self, *_) -> None:
        name = self.var_combo.currentText()
        if name:
            self._select(name)

    def _select(self, name: str) -> None:
        if not name or self._df is None or name not in self._df.columns:
            return
        self._var = name
        # The chain carries over across variables; re-run it on the new series.
        if self._steps:
            # Mark dirty + rebuild so cards don't show the previous variable's
            # removal counts during the re-run.
            self._chain_dirty = True
            self._rebuild_cards()
            self._run()
        else:
            self._reset_preview()
        self.status.setText(f"Selected '{name}'. Add outlier tests to build a chain.")

    def _reset_preview(self) -> None:
        self._payload = None
        self._result_df = None
        self._selected_step = -1
        self.qcf_label.setText("QCF: add a step to compute.")
        self.add_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.report_text.clear()
        self.report_copy_btn.setEnabled(False)
        self._rebuild_cards()
        if self._var is not None and self._df is not None:
            self._draw_raw(self._df[self._var])

    # --- run (rebuild + replay on a worker thread) ---
    @staticmethod
    def _coords() -> dict:
        m = site.manager
        if m.configured:
            return dict(site_lat=m.latitude, site_lon=m.longitude,
                        utc_offset=m.utc_offset)
        return dict(site_lat=0.0, site_lon=0.0, utc_offset=0)

    def _run(self) -> None:
        if self._df is None or self._var is None:
            return
        # Bump the run id so any in-flight worker's result is ignored on arrival
        # (results can land out of order under rapid edits).
        self._run_id += 1
        run_id = self._run_id
        # Nothing to run (no steps, or all toggled off): show the raw series.
        if not any(s.get("enabled", True) for s in self._steps):
            self._payload = None
            self._chain_dirty = False
            self.add_btn.setEnabled(False)
            self.copy_btn.setEnabled(False)
            self.report_text.clear()
            self.report_copy_btn.setEnabled(False)
            self.qcf_label.setText("QCF: add a step to compute.")
            self._draw_raw(self._df[self._var])
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
        self._build_result(payload)
        self.add_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self._draw_payload()

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

    # --- result columns for "Add" ---
    def _build_result(self, payload: dict) -> None:
        det = payload["detector"]
        qcf = payload["qcf"]
        var = payload["var"]
        # Emit: all FLAG_*_TEST columns + the overall QCF flag + the QCF-filtered series.
        cols = pd.concat([det.flags,
                          qcf.flagqcf.rename(qcf.flagqcfcol),
                          qcf.filteredseries.rename(qcf.filteredseriescol)], axis=1)
        # Snapshot the chain: provenance must capture what was emitted, not a
        # live reference that later edits would mutate.
        params = {"steps": copy.deepcopy(self._steps)}
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

    def _code_provider(self) -> str | None:
        """Render the chain (enabled steps only) as a runnable script, or None if
        there is nothing to copy. The CopyPythonButton handles clipboard + the
        'Copied ✓' flash."""
        enabled = [s for s in self._steps if s.get("enabled", True)]
        if not enabled or self._var is None:
            return None
        return stepwise_to_code(enabled, var_name=self._var, **self._coords(),
                                load_hint="dv.load_parquet('your_data.parquet')")

    def _report_provider(self) -> str | None:
        """The current screening report text (or None), for the Copy button."""
        if self._payload is None:
            return None
        return self._payload.get("report") or None

    # --- preview ---
    def _draw_raw(self, series) -> None:
        """Single panel with the original series (before any step runs)."""
        ax = self.canvas.new_axes(1)[0]
        ax.plot(series.index, series.to_numpy(), color=_C_RAW, lw=0.5, alpha=0.85)
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

        # 2) Cleaned series.
        ax_clean.plot(cleaned.index, cleaned.to_numpy(), color=_C_CLEANED, lw=0.7)
        ax_clean.set_title("cleaned (outliers removed)", fontsize=9)

        # 3) Heatmap of the cleaned series.
        try:
            dv.plotting.HeatmapDateTime(cleaned).plot(
                ax=ax_heat, fig=fig, title="cleaned", cb_digits_after_comma="auto")
        except Exception as err:
            ax_heat.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                         wrap=True, transform=ax_heat.transAxes)

        # 4) Heatmap of the QCF flag (when/where records were rejected).
        flag = qcf.flagqcf
        try:
            dv.plotting.HeatmapDateTime(flag.rename("QCF")).plot(
                ax=ax_qcf_heat, fig=fig, title="QCF (0/1/2)", vmin=0, vmax=2,
                cmap="RdYlGn_r", cb_digits_after_comma=0)
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
                "limits": self.limits_cb.isChecked()}

    def restore_state(self, state: dict) -> None:
        self._steps = list(state.get("steps") or [])
        self.limits_cb.setChecked(bool(state.get("limits")))
        var = state.get("var")
        if var and self._df is not None and var in self._df.columns:
            self._var = var
            self.var_combo.blockSignals(True)
            self.var_combo.setCurrentText(str(var))
            self.var_combo.blockSignals(False)
            self._rebuild_cards()
            if self._steps:
                self._run()  # replays the saved chain
            else:
                self._draw_raw(self._df[var])
