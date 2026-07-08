"""
GUI.TABS.SELECT_RECORDS: SELECT RECORDS BY A CONDITION VARIABLE
==============================================================

Build a filtered copy of a *target* variable by applying one or more
keep/remove operations driven by a *condition* variable: each operation keeps
(or removes) the target records where the condition falls within a chosen
``[lower, upper]`` range. Operations accumulate on a working series — so you can
chain several conditions (different variables, ranges, keep vs remove) and watch
the preview update after each — then add the result as a new column.

Wraps the library's :func:`diive.core.dfun.frames.keep_records_where` (range +
``invert`` logic) and :func:`select_records_to_code` (reproducible script). The
GUI only orchestrates the steps, previews, and emits the column — a row-on-a-
condition sibling of *Select variables* / *Select date range*, so it lives under
the **Data** menu. The output is a derived column ``{target}_SEL`` (the target
masked to NaN outside the kept set, preserving the time index).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diive.core.dfun.frames import keep_records_where, select_records_to_code
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import HeroBand
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.variable_panel import VariablePanel

_C_TARGET = "#B0BEC5"   # blue-grey 200 — the full target (faint reference)
_C_KEPT = "#1565C0"     # blue 800      — the surviving (kept) working series
_C_COND = "#90A4AE"     # blue-grey 300 — the condition line
_C_SEL = "#E53935"      # red 600       — the current operation's in-range records
_C_BAND = "#FFC107"     # amber 500     — the [lower, upper] band fill
_C_MUTED = "#6B7780"

#: Suffix for the emitted column name: {target}_SEL.
_SUFFIX = "SEL"

_INCLUSIVE = [
    ("both", "Both bounds inclusive"),
    ("neither", "Both bounds exclusive"),
    ("left", "Lower inclusive, upper exclusive"),
    ("right", "Lower exclusive, upper inclusive"),
]

_MODES = [("keep", "Keep selected records"), ("remove", "Remove selected records")]


class _Signals(QObject):
    """Qt signal (DiiveTab is a plain ABC, not a QObject)."""
    features_created = Signal(object)


class SelectRecordsTab(DiiveTab):
    """Iteratively keep/remove target records by a condition variable's range."""

    title = "Select records by condition"
    intro = ("Filter a target variable by another variable's value: keep (or "
             "remove) records where the condition is within [lower, upper]. "
             "Operations stack — chain several, then add the {target}_SEL column.")
    method_chip_label = "SELECT BY CONDITION"
    method_chip_bg = "#E3F2FD"
    method_chip_fg = "#1565C0"

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._all_cols: list[str] = []
        self._target: str | None = None
        #: Boolean Series (index-aligned), True = record currently kept.
        self._keep_mask: pd.Series | None = None
        #: Applied operations, for the status list, codegen and provenance.
        self._steps: list[dict] = []
        #: Most recent operation's in-range mask, for the preview markers.
        self._last_selected: pd.Series | None = None

        self._sig = _Signals()
        #: Exposed bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip("Copy a runnable diive script reproducing these selections.")
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)

        # Left: pick the target variable (records are filtered from it).
        tcol = QVBoxLayout()
        tcol.addWidget(list_header("Target", "click to set target"))
        self.varpanel = VariablePanel()
        self.varpanel.list.setToolTip("Click a variable to filter its records by a condition.")
        self.varpanel.selected.connect(lambda name, _ctrl: self._select(name))
        tcol.addWidget(self.varpanel, stretch=1)
        layout.addLayout(tcol)

        # Middle: settings.
        mid = self._build_settings()
        mid.setFixedWidth(320)
        layout.addWidget(mid)

        # Right: hero + preview.
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(self._build_hero())
        self.canvas = MplCanvas()
        rlay.addWidget(self.canvas, stretch=1)
        layout.addWidget(right, stretch=1)

        outer.addWidget(body, stretch=1)
        return root

    def _build_settings(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)

        outer.addWidget(list_header("Settings", "build an operation & apply"))
        intro = QLabel(self.intro)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        outer.addWidget(intro)

        box = QGroupBox("Operation")
        form = QFormLayout(box)

        # Condition variable + availability marker.
        self.cond_combo = QComboBox()
        self.cond_combo.setToolTip(
            "The variable whose value decides which target records are selected.")
        self.cond_combo.currentTextChanged.connect(self._on_condition_changed)
        self.cond_mark = QLabel("")
        cell = QWidget()
        ch = QHBoxLayout(cell)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.setSpacing(6)
        ch.addWidget(self.cond_combo, stretch=1)
        ch.addWidget(self.cond_mark)
        form.addRow("Condition variable", cell)

        # Lower / upper limits. A disabled limit means an open (one-sided) bound.
        self.use_lower = self._check("Lower limit", True,
                                     "Tick to apply a lower bound; untick for an open lower side.")
        self.lower = QDoubleSpinBox()
        self.lower.setRange(-1e12, 1e12)
        self.lower.setDecimals(4)
        self.lower.setToolTip("Records where the condition is >= this value (>, if exclusive).")
        form.addRow(self.use_lower, self.lower)

        self.use_upper = self._check("Upper limit", True,
                                     "Tick to apply an upper bound; untick for an open upper side.")
        self.upper = QDoubleSpinBox()
        self.upper.setRange(-1e12, 1e12)
        self.upper.setDecimals(4)
        self.upper.setToolTip("Records where the condition is <= this value (<, if exclusive).")
        form.addRow(self.use_upper, self.upper)

        self.use_lower.toggled.connect(self.lower.setEnabled)
        self.use_upper.toggled.connect(self.upper.setEnabled)

        # Boundary inclusivity.
        self.inclusive = QComboBox()
        for value, label in _INCLUSIVE:
            self.inclusive.addItem(label, value)
        self.inclusive.setToolTip("Whether the lower/upper bounds themselves are selected.")
        form.addRow("Bounds", self.inclusive)

        # Keep vs remove the selected (in-range) records.
        self.mode = QComboBox()
        for value, label in _MODES:
            self.mode.addItem(label, value)
        self.mode.setToolTip(
            "Keep: drop everything except the in-range records. "
            "Remove: drop the in-range records, keep the rest.")
        form.addRow("Action", self.mode)

        # Live updates of the band/markers as the operation is edited.
        for w in (self.cond_combo, self.inclusive, self.mode):
            w.currentIndexChanged.connect(self._refresh_preview)
        for w in (self.lower, self.upper):
            w.valueChanged.connect(self._refresh_preview)
        for w in (self.use_lower, self.use_upper):
            w.toggled.connect(self._refresh_preview)

        outer.addWidget(box)

        # Apply / undo / reset row.
        self.apply_btn = QPushButton("Select records")
        theme.set_button_role(self.apply_btn, "confirm")
        self.apply_btn.setToolTip("Apply this operation to the working selection.")
        self.apply_btn.clicked.connect(self._apply_op)
        outer.addWidget(self.apply_btn)

        sub = QHBoxLayout()
        self.undo_btn = QPushButton("Undo last")
        self.undo_btn.setToolTip("Remove the most recently applied operation.")
        self.undo_btn.clicked.connect(self._undo)
        sub.addWidget(self.undo_btn)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Discard all operations and start from the full target.")
        self.reset_btn.clicked.connect(self._reset_steps)
        sub.addWidget(self.reset_btn)
        outer.addLayout(sub)

        self.status = QLabel("Select a target on the left, then build an operation.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        # Applied-operations list.
        self.steps_label = QLabel("")
        self.steps_label.setWordWrap(True)
        self.steps_label.setStyleSheet(f"color: {_C_MUTED}; font-family: monospace;")
        outer.addWidget(self.steps_label)

        self.add_btn = QPushButton("Add selection to dataset")
        self.add_btn.setToolTip("Append the {target}_SEL column to the variable list.")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")
        outer.addWidget(self.add_btn)

        outer.addStretch(1)
        self._update_step_controls()
        return panel

    @staticmethod
    def _check(text: str, checked: bool, tip: str):
        from PySide6.QtWidgets import QCheckBox
        cb = QCheckBox(text)
        cb.setChecked(checked)
        cb.setToolTip(tip)
        return cb

    def _build_hero(self) -> QWidget:
        self._hero = HeroBand(self.method_chip_label, self.method_chip_bg,
                              self.method_chip_fg)
        return self._hero

    def _clear_hero(self) -> None:
        self._hero.clear()

    def _update_hero(self) -> None:
        if self._target is None or self._keep_mask is None:
            self._hero.clear()
            return
        target_s = self._df[self._target]
        n_orig = int(target_s.notna().sum())
        n_kept = int((self._keep_mask & target_s.notna()).sum())
        pct = (100.0 * n_kept / n_orig) if n_orig else 0.0
        self._hero.set_metrics([
            ("KEPT", f"{n_kept:,}", "Valid target records remaining after the operations"),
            ("OF VALID", f"{n_orig:,}", "Valid (non-missing) target records to begin with"),
            ("SHARE", f"{pct:.1f}%", "Share of valid records kept"),
            ("STEPS", f"{len(self._steps)}", "Number of operations applied"),
        ])

    # --- data / inputs -------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._all_cols = [str(c) for c in df.columns]
        self.varpanel.set_variables(df.columns, created)
        if self._target is not None and self._target not in df.columns:
            self._target = None
        # A new/edited dataset invalidates the accumulated selection.
        self._reset_state()

        cur = self.cond_combo.currentText()
        self.cond_combo.blockSignals(True)
        self.cond_combo.clear()
        self.cond_combo.addItems(self._all_cols)
        if cur in self._all_cols:
            self.cond_combo.setCurrentText(cur)
        self.cond_combo.blockSignals(False)
        self._refresh_condition_marker()
        self._seed_limits()
        if self._target is not None:
            self._keep_mask = pd.Series(True, index=df.index)
        self._update_step_controls()
        self._refresh_preview()

    def _reset_state(self) -> None:
        self._keep_mask = None
        self._steps = []
        self._last_selected = None
        self.add_btn.setEnabled(False)
        self._clear_hero()
        self._refresh_steps_label()

    def _select(self, name: str) -> None:
        if not name or self._df is None:
            return
        self._target = name
        self.varpanel.set_panels([name])
        self._keep_mask = pd.Series(True, index=self._df.index)
        self._steps = []
        self._last_selected = None
        self.add_btn.setEnabled(False)
        self.status.setText(f"Target '{name}'. Build an operation and apply.")
        self._refresh_steps_label()
        self._update_hero()
        self._update_step_controls()
        self.varpanel.run_with_loading(name, self._refresh_preview)

    def _on_condition_changed(self, *_) -> None:
        self._refresh_condition_marker()
        self._seed_limits()

    def _refresh_condition_marker(self) -> None:
        present = self.cond_combo.currentText() in self._all_cols
        self.cond_mark.setText("✓" if present else "✗")
        self.cond_mark.setStyleSheet(
            f"color: {'#2E7D32' if present else '#C62828'}; font-weight: bold;")

    def _seed_limits(self) -> None:
        """Seed the limit spinboxes to the condition variable's range, so a fresh
        operation starts spanning everything and the user narrows from there."""
        if self._df is None:
            return
        cond = self.cond_combo.currentText()
        if cond not in self._all_cols:
            return
        series = self._df[cond]
        lo, hi = series.min(skipna=True), series.max(skipna=True)
        if pd.notna(lo) and pd.notna(hi):
            for w in (self.lower, self.upper):
                w.blockSignals(True)
            self.lower.setValue(float(lo))
            self.upper.setValue(float(hi))
            for w in (self.lower, self.upper):
                w.blockSignals(False)
            self._refresh_preview()

    # --- state ---------------------------------------------------------
    def _controls(self) -> dict:
        return {"cond": self.cond_combo, "use_lower": self.use_lower,
                "lower": self.lower, "use_upper": self.use_upper, "upper": self.upper,
                "inclusive": self.inclusive, "mode": self.mode}

    def save_state(self) -> dict:
        # Persist the target + the current operation editor only; the accumulated
        # working selection is interactive and rebuilt from scratch on restore.
        from diive.gui.widgets.state_utils import save_controls
        return {"target": self._target, "controls": save_controls(self._controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._controls(), state.get("controls"))
        self._refresh_condition_marker()
        target = state.get("target")
        if target and self._df is not None and target in self._df.columns:
            self._select(target)

    # --- operation building --------------------------------------------
    def _params(self) -> dict:
        return {
            "lower": self.lower.value() if self.use_lower.isChecked() else None,
            "upper": self.upper.value() if self.use_upper.isChecked() else None,
            "inclusive": self.inclusive.currentData(),
            "mode": self.mode.currentData(),
        }

    def _validate(self) -> str | None:
        if self._df is None or self._target is None:
            return "Select a target on the left first."
        cond = self.cond_combo.currentText()
        if cond not in self._all_cols:
            return "Pick a valid condition variable."
        # The condition may be the target itself (filter a variable by its own value).
        if not self.use_lower.isChecked() and not self.use_upper.isChecked():
            return "Enable at least one of the lower / upper limits."
        return None

    def _selected_mask(self, cond: str, params: dict) -> pd.Series:
        """In-range mask for the condition (NaN condition -> False), via the
        library so the range/inclusive logic has a single home."""
        sel = keep_records_where(
            self._df, target=cond, condition_var=cond,
            lower=params["lower"], upper=params["upper"],
            inclusive=params["inclusive"], set_to_nan=True)
        return sel.notna()

    def _apply_op(self) -> None:
        err = self._validate()
        if err:
            self.status.setText(err)
            return
        cond = self.cond_combo.currentText()
        params = self._params()
        selected = self._selected_mask(cond, params)
        # keep -> intersect with the in-range set; remove -> subtract it.
        step_keep = ~selected if params["mode"] == "remove" else selected
        self._keep_mask = self._keep_mask & step_keep
        self._last_selected = selected
        self._steps.append({"cond": cond, **params})

        n_kept = int((self._keep_mask & self._df[self._target].notna()).sum())
        verb = "Removed" if params["mode"] == "remove" else "Kept"
        self.status.setText(
            f"Step {len(self._steps)}: {verb.lower()} records where '{cond}' "
            f"in range. {n_kept:,} valid records remain.")
        self.add_btn.setEnabled(True)
        self._refresh_steps_label()
        self._update_hero()
        self._update_step_controls()
        self._refresh_preview()

    def _undo(self) -> None:
        if not self._steps:
            return
        self._steps.pop()
        self._rebuild_mask()
        self.status.setText(
            f"Undid the last operation. {len(self._steps)} remaining.")
        self.add_btn.setEnabled(bool(self._steps))
        self._refresh_steps_label()
        self._update_hero()
        self._update_step_controls()
        self._refresh_preview()

    def _reset_steps(self) -> None:
        if self._target is None:
            return
        self._steps = []
        self._last_selected = None
        self._keep_mask = pd.Series(True, index=self._df.index)
        self.add_btn.setEnabled(False)
        self.status.setText("Reset to the full target.")
        self._refresh_steps_label()
        self._update_hero()
        self._update_step_controls()
        self._refresh_preview()

    def _rebuild_mask(self) -> None:
        """Recompute the working keep-mask by replaying all steps (after an undo)."""
        mask = pd.Series(True, index=self._df.index)
        last = None
        for step in self._steps:
            selected = self._selected_mask(step["cond"], step)
            mask &= ~selected if step["mode"] == "remove" else selected
            last = selected
        self._keep_mask = mask
        self._last_selected = last

    def _refresh_steps_label(self) -> None:
        if not self._steps:
            self.steps_label.setText("No operations applied yet.")
            return
        lines = []
        for i, s in enumerate(self._steps, 1):
            lo = "-inf" if s.get("lower") is None else f"{s['lower']:g}"
            hi = "+inf" if s.get("upper") is None else f"{s['upper']:g}"
            verb = "remove" if s["mode"] == "remove" else "keep"
            lines.append(f"{i}. {verb}: {s['cond']} in [{lo}, {hi}]")
        self.steps_label.setText("\n".join(lines))

    def _update_step_controls(self) -> None:
        has_steps = bool(self._steps)
        self.undo_btn.setEnabled(has_steps)
        self.reset_btn.setEnabled(has_steps)

    # --- code / emit ---------------------------------------------------
    def _out_name(self) -> str:
        return f"{self._target}_{_SUFFIX}".replace(".", "_")

    def _python_code(self) -> str | None:
        if self._target is None:
            self.status.setText("Select a target first to copy its code.")
            return None
        if not self._steps:
            self.status.setText("Apply at least one operation first to copy the code.")
            return None
        return select_records_to_code(str(self._target), self._steps, out_var=self._out_name())

    def _working_series(self) -> pd.Series:
        out = self._df[self._target].where(self._keep_mask).copy()
        out.name = self._out_name()
        return out

    def _add(self) -> None:
        if self._target is None or self._keep_mask is None or not self._steps:
            return
        out = self._working_series()
        result = pd.DataFrame({out.name: out})
        result.attrs[ATTRS_KEY] = {
            out.name: provenance_attr(
                origin=DERIVED, parent=str(self._target), operation=self.title,
                params={"steps": self._steps}, tags=["selected"]),
        }
        self.featuresCreated.emit(result)
        self.status.setText(f"Added {out.name} to the variable list.")
        self.add_btn.setEnabled(False)

    # --- preview -------------------------------------------------------
    def _refresh_preview(self) -> None:
        """Two stacked panels sharing the time axis: the condition variable with
        the editable [lower, upper] band on top (current operation's in-range
        records marked), the target below (the surviving working series in blue
        over a faint full-record reference)."""
        if self._df is None or self._target is None:
            return
        cond = self.cond_combo.currentText()
        have_cond = cond in self._all_cols
        target_s = self._df[self._target]

        self.canvas.reset_layout()
        fig = self.canvas.fig
        ax_cond = fig.add_subplot(2, 1, 1)
        ax_tgt = fig.add_subplot(2, 1, 2, sharex=ax_cond)

        # Top: condition variable + the (editable) selection band + last op markers.
        if have_cond:
            cond_s = self._df[cond]
            ax_cond.plot(cond_s.index, cond_s.to_numpy(), color=_C_COND, lw=0.6,
                         label=cond, zorder=1)
            p = self._params()
            lo = p["lower"] if p["lower"] is not None else cond_s.min(skipna=True)
            hi = p["upper"] if p["upper"] is not None else cond_s.max(skipna=True)
            if pd.notna(lo) and pd.notna(hi):
                ax_cond.axhspan(lo, hi, color=_C_BAND, alpha=0.20,
                                label=f"band [{lo:g}, {hi:g}]", zorder=0)
            sel = self._last_selected
            if sel is not None and sel.any():
                ax_cond.plot(cond_s.index[sel], cond_s[sel].to_numpy(), linestyle="none",
                             marker=".", color=_C_SEL, ms=3, zorder=5,
                             label=f"last selected ({int(sel.sum())})")
            ax_cond.legend(loc="best", fontsize=8, framealpha=0.9)
            ax_cond.set_title(f"Condition: {cond}", fontsize=9)
        else:
            ax_cond.set_title("Pick a condition variable", fontsize=9)

        # Bottom: faint full target + the kept working series highlighted.
        ax_tgt.plot(target_s.index, target_s.to_numpy(), color=_C_TARGET, lw=0.5,
                    alpha=0.7, label=f"{self._target} (full)", zorder=1)
        if self._keep_mask is not None and self._steps:
            keep = self._keep_mask & target_s.notna()
            if keep.any():
                ax_tgt.plot(target_s.index[keep], target_s[keep].to_numpy(),
                            linestyle="none", marker=".", color=_C_KEPT, ms=3,
                            zorder=5, label=f"kept ({int(keep.sum())})")
        ax_tgt.legend(loc="best", fontsize=8, framealpha=0.9)
        ax_tgt.set_title(f"Target: {self._target}"
                         + (" — kept records" if self._steps else " (no operations yet)"),
                         fontsize=9)
        self.canvas.draw()
