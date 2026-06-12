"""
GUI.TABS.FLUXCHAIN: FLUX PROCESSING CHAIN (Input + L2 + L3.1 + L3.2 + L3.3)
==========================================================================

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

The chain stays on the **composable per-level** path (not `run_chain`/`FluxConfig`)
so L3.2 can be a real inspected `StepwiseOutlierDetection` chain with a separate QCF
surface. Later slices add L3.3 / L4.1 groups and the matching `level*_to_code`
renderers, growing incrementally.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.flux.fluxprocessingchain import (
    init_flux_data, level31_to_code, level32_to_code, level33_to_code,
    make_level32_detector, run_level2, run_level31, run_level32,
    run_level33_constant_ustar,
)
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.stepwise_method_params import STEP_METHOD_BY_KEY, method_labels

#: Columns init_flux_data computes itself and rejects if already present.
_RESERVED = ("SW_IN_POT", "DAYTIME", "NIGHTTIME")

#: L2 tests offered as checkboxes -> (label, default-on). Each maps to a
#: run_level2 keyword; the settings dict is built in `_level2_settings`.
_L2_TESTS = [
    ("ssitc", "SSITC (steady-state / turbulence)", True),
    ("gas_completeness", "Gas completeness", True),
    ("spectral_correction_factor", "Spectral correction factor", True),
    ("raw_data_screening_vm97", "Raw-data screening (VM97)", True),
    ("signal_strength", "Signal strength (IRGA AGC)", False),
    ("angle_of_attack", "Angle of attack", False),
    ("steadiness_of_horizontal_wind", "Steadiness of horizontal wind", False),
]


class _ChainSignals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject)."""
    done = Signal(object)   # FluxLevelData
    failed = Signal(str)


class FluxChainTab(DiiveTab):
    """Guided flux processing chain — Input + Level 2 (first slice)."""

    title = "Flux chain"

    def build(self) -> QWidget:
        self._df = None
        self._sig = _ChainSignals()
        self._sig.done.connect(self._on_done)
        self._sig.failed.connect(self._on_failed)

        root = QWidget()
        row = QHBoxLayout(root)

        # Left: scrollable, fixed-width stepwise config.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.addWidget(self._input_group())
        col.addWidget(self._level2_group())
        col.addWidget(self._level31_group())
        col.addWidget(self._level32_group())
        col.addWidget(self._level33_group())
        self.run_btn = QPushButton("Run through Level 3.1")
        self.run_btn.clicked.connect(self._run)
        col.addWidget(self.run_btn)
        self.code_btn = QPushButton("Copy Python")
        self.code_btn.clicked.connect(self._copy_code)
        col.addWidget(self.code_btn)
        col.addStretch(1)
        scroll.setWidget(inner)
        row.addWidget(scroll)
        self._apply_tooltips()

        # Right: diagnostics canvas + summary/code text.
        right = QVBoxLayout()
        self.canvas = MplCanvas()
        right.addWidget(self.canvas, stretch=1)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(190)
        right.addWidget(self.summary)
        row.addLayout(right, stretch=1)
        return root

    # --- config groups ---
    def _input_group(self) -> QGroupBox:
        box = QGroupBox("Input & site")
        form = QFormLayout(box)
        self.fluxcol = QComboBox()
        form.addRow("Flux column", self.fluxcol)
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
        return box

    def _level2_group(self) -> QGroupBox:
        box = QGroupBox("Level 2 — quality flags")
        v = QVBoxLayout(box)
        self.l2_checks: dict[str, QCheckBox] = {}
        for key, label, on in _L2_TESTS:
            cb = QCheckBox(label)
            cb.setChecked(on)
            v.addWidget(cb)
            self.l2_checks[key] = cb
        # Signal-strength needs a column; show a combo for it.
        form = QFormLayout()
        self.signal_strength_col = QComboBox()
        form.addRow("Signal-strength column", self.signal_strength_col)
        v.addLayout(form)
        return box

    def _level31_group(self) -> QGroupBox:
        box = QGroupBox("Level 3.1 — storage correction")
        v = QVBoxLayout(box)
        self.l31_gapfill = QCheckBox("Gap-fill storage term (rolling median)")
        self.l31_gapfill.setChecked(True)
        v.addWidget(self.l31_gapfill)
        self.l31_zero = QCheckBox("Set storage to zero (H / LE — no storage profile)")
        self.l31_zero.setChecked(False)
        # Setting storage to zero makes the gap-fill choice irrelevant.
        self.l31_zero.toggled.connect(
            lambda on: self.l31_gapfill.setEnabled(not on))
        v.addWidget(self.l31_zero)
        form = QFormLayout()
        self.strgcol = QComboBox()  # "" = auto-detect (FLUXNET/EddyPro naming)
        form.addRow("Storage column (auto if blank)", self.strgcol)
        v.addLayout(form)
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

    def _update_run_label(self, *_) -> None:
        # Reflects intent: L3.3 configured -> "3.3" even before an L3.2 step exists
        # (the run then surfaces the "needs an L3.2 test" requirement).
        if self.l33_enable.isChecked() and self._ustar:
            deepest = "3.3"
        elif self._steps:
            deepest = "3.2"
        else:
            deepest = "3.1"
        self.run_btn.setText(f"Run through Level {deepest}")

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

    @staticmethod
    def _dspin(value, lo, hi, decimals) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setValue(value)
        return sp

    # --- state ---
    def _fx_controls(self) -> dict:
        return {"fluxcol": self.fluxcol, "site_lat": self.site_lat,
                "site_lon": self.site_lon, "utc_offset": self.utc_offset,
                "nighttime_threshold": self.nighttime_threshold,
                "day_qcf": self.day_qcf, "night_qcf": self.night_qcf,
                "signal_strength_col": self.signal_strength_col,
                "l31_gapfill": self.l31_gapfill, "l31_zero": self.l31_zero,
                "strgcol": self.strgcol}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._fx_controls()),
                "l2": {k: cb.isChecked() for k, cb in self.l2_checks.items()},
                "l32_steps": self._steps,
                "l33_enabled": self.l33_enable.isChecked(),
                "l33_ustar": [list(u) for u in self._ustar]}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._fx_controls(), state.get("controls"))
        for k, val in (state.get("l2") or {}).items():
            if k in self.l2_checks:
                self.l2_checks[k].setChecked(bool(val))
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
        self._update_run_label()

    # --- data ---
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        cols = [str(c) for c in df.columns]
        self.fluxcol.clear()
        self.fluxcol.addItems(cols)
        # Default to a common flux column if present.
        for cand in ("FC", "LE", "H", "NEE", "FC_orig"):
            if cand in cols:
                self.fluxcol.setCurrentText(cand)
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

    # --- collect kwargs the library wants ---
    def _init_kwargs(self) -> dict:
        return dict(
            fluxcol=self.fluxcol.currentText(),
            site_lat=self.site_lat.value(),
            site_lon=self.site_lon.value(),
            utc_offset=self.utc_offset.value(),
            nighttime_threshold=self.nighttime_threshold.value(),
            daytime_accept_qcf_below=int(self.day_qcf.currentText()),
            nighttime_accept_qcf_below=int(self.night_qcf.currentText()),
        )

    def _level2_settings(self) -> dict:
        settings: dict = {}
        for key, cb in self.l2_checks.items():
            if not cb.isChecked():
                continue
            if key == "ssitc":
                settings[key] = {"apply": True, "setflag_timeperiod": None}
            elif key == "raw_data_screening_vm97":
                settings[key] = {"apply": True, "spikes": True, "dropout": True,
                                 "amplitude": False, "abslim": False,
                                 "skewkurt_hf": False, "skewkurt_sf": False,
                                 "discont_hf": False, "discont_sf": False}
            elif key == "signal_strength":
                col = self.signal_strength_col.currentText().strip()
                if not col:
                    continue  # can't run without a column
                settings[key] = {"apply": True, "signal_strength_col": col,
                                 "method": "discard below", "threshold": 60}
            else:
                settings[key] = {"apply": True}
        return settings

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
                 level33_kwargs: dict | None = None):
        """Init the container and run Level 2 → 3.1 → (optional) 3.2 → (optional) 3.3."""
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
        if l33 and not steps:
            # USTAR filtering must operate on outlier-screened data (library raises).
            self.summary.setPlainText(
                "Level 3.3 (USTAR) requires at least one Level 3.2 outlier test. "
                "Add a step or disable Level 3.3.")
            return
        deepest = "3.3" if l33 else ("3.2" if steps else "3.1")
        self.run_btn.setEnabled(False)
        self.summary.setPlainText(f"Running Level 2 → Level {deepest}…")
        threading.Thread(
            target=self._worker, args=(self._df, ik, l2, l31, steps, l33),
            daemon=True).start()

    def _worker(self, df, init_kwargs, level2_settings, level31_kwargs, level32_steps,
                level33_kwargs) -> None:
        try:
            data = self._compute(df, init_kwargs, level2_settings, level31_kwargs,
                                  level32_steps, level33_kwargs)
        except Exception as err:
            self._sig.failed.emit(str(err))
            return
        self._sig.done.emit(data)

    def _on_done(self, data) -> None:
        self.run_btn.setEnabled(True)
        # Deepest level reached: L3.3 (USTAR scenarios) > L3.2 (outliers) > L3.1.
        l32_qcf = getattr(data.levels, "level32_qcf", None)
        l33_qcf = getattr(data.levels, "level33_qcf", None)
        level = "3.3" if l33_qcf else ("3.2" if l32_qcf is not None else "3.1")
        # After L3.3 there is one filtered series per USTAR scenario (data.filteredseries
        # is then ambiguous/None); preview the first scenario.
        series = data.filteredseries
        qcf_note = ""
        if l33_qcf:
            try:
                scen_series = data.levels.filteredseries_level33_qcf
                first = next(iter(scen_series))
                series = scen_series[first]
                qcf_note = (f"\nL3.3 USTAR scenarios: {', '.join(scen_series)} "
                            f"(preview: {first}).")
            except Exception:
                qcf_note = ""
        valid = int(series.dropna().count()) if series is not None else 0
        if l32_qcf is not None and not l33_qcf:
            try:
                vc = l32_qcf.flagqcf.value_counts()
                qcf_note = (f"\nL3.2 QCF — good (0): {int(vc.get(0, 0))}, "
                            f"marginal (1): {int(vc.get(1, 0))}, "
                            f"rejected (2): {int(vc.get(2, 0))}.")
            except Exception:
                qcf_note = ""
        self.summary.setPlainText(
            f"Level {level} done — {valid} accepted records "
            f"(filtered series: {series.name if series is not None else '—'})."
            f"{qcf_note}\nClick 'Copy Python' for the reproducible script.")
        # Show the deepest level's QCF-filtered flux as a date/time heatmap.
        ax = self.canvas.new_axes(1)[0]
        try:
            dv.plotting.HeatmapDateTime(series).plot(
                ax=ax, fig=self.canvas.fig, title=f"L{level} filtered: {series.name}",
                cb_digits_after_comma="auto")
        except Exception as err:
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
        self.canvas.draw()

    def _on_failed(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.summary.setPlainText(f"Failed:\n{msg}")

    # --- the killer button: the exact reproducible script ---
    def _code(self) -> str:
        hint = "dv.load_parquet('your_data.parquet')"
        ik, l2, l31 = self._init_kwargs(), self._level2_settings(), self._level31_kwargs()
        l33 = self._level33_kwargs()
        if l33 and self._steps:
            return level33_to_code(ik, l2, l31, self._steps, l33, load_hint=hint)
        if self._steps:
            return level32_to_code(ik, l2, l31, self._steps, load_hint=hint)
        return level31_to_code(ik, l2, l31, load_hint=hint)

    def _copy_code(self) -> None:
        code = self._code()
        QApplication.clipboard().setText(code)
        self.summary.setPlainText(code)
