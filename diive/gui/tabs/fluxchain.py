"""
GUI.TABS.FLUXCHAIN: FLUX PROCESSING CHAIN (first slice: Input + Level 2)
=======================================================================

A guided tab for the Swiss-FluxNet flux processing chain. This first slice
wires the **Input** (site + flux column) and **Level 2** (quality-flag tests)
steps using the composable library callables (`init_flux_data` + `run_level2`),
shows the L2 QCF-filtered flux as a heatmap, and — the point of the feature —
emits the exact reproducible diive script via **Copy Python**.

All computation is library work (`init_flux_data`, `run_level2`); this tab only
collects parameters and calls them (heavy runs on a worker thread). The
script-generation is the library's `level2_to_code` (it owns the API shape).

Later slices add L3.1 / L3.2 / L3.3 / L4.1 groups and switch the run/serialize
to the full `run_chain` + `chain_to_code` path. See `_render_ridgeline`-style
incremental growth.

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
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.flux.fluxprocessingchain import init_flux_data, level2_to_code, run_level2
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas

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
        self.run_btn = QPushButton("Run Level 2")
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

    def _apply_tooltips(self) -> None:
        """Tooltip each control with its library parameter docstring."""
        from diive.core.utils.docstrings import param_docs
        from diive.flux.fluxprocessingchain import FluxConfig
        docs: dict = {}
        for src in (init_flux_data, FluxConfig, run_level2):
            docs.update(param_docs(src))
        for param, widget in (
            ("fluxcol", self.fluxcol), ("site_lat", self.site_lat),
            ("site_lon", self.site_lon), ("utc_offset", self.utc_offset),
            ("nighttime_threshold", self.nighttime_threshold),
            ("daytime_accept_qcf_below", self.day_qcf),
            ("nighttime_accept_qcf_below", self.night_qcf),
            ("signal_strength_col", self.signal_strength_col),
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
                "signal_strength_col": self.signal_strength_col}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._fx_controls()),
                "l2": {k: cb.isChecked() for k, cb in self.l2_checks.items()}}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._fx_controls(), state.get("controls"))
        for k, val in (state.get("l2") or {}).items():
            if k in self.l2_checks:
                self.l2_checks[k].setChecked(bool(val))

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

    # --- run (library work on a worker thread) ---
    def _compute(self, df, init_kwargs: dict, level2_settings: dict):
        """Init the container and run Level 2 (synchronous; library calls)."""
        work = df.drop(columns=[c for c in _RESERVED if c in df.columns])
        data = init_flux_data(df=work, **init_kwargs)
        return run_level2(data, **level2_settings)

    def _run(self) -> None:
        if self._df is None:
            return
        ik, l2 = self._init_kwargs(), self._level2_settings()
        if not l2:
            self.summary.setPlainText("Enable at least one Level 2 test.")
            return
        self.run_btn.setEnabled(False)
        self.summary.setPlainText("Running Level 2…")
        threading.Thread(
            target=self._worker, args=(self._df, ik, l2), daemon=True).start()

    def _worker(self, df, init_kwargs, level2_settings) -> None:
        try:
            data = self._compute(df, init_kwargs, level2_settings)
        except Exception as err:
            self._sig.failed.emit(str(err))
            return
        self._sig.done.emit(data)

    def _on_done(self, data) -> None:
        self.run_btn.setEnabled(True)
        series = data.filteredseries
        valid = int(series.dropna().count()) if series is not None else 0
        self.summary.setPlainText(
            f"Level 2 done — {valid} accepted records "
            f"(filtered series: {series.name if series is not None else '—'}).\n"
            f"Click 'Copy Python' for the reproducible script.")
        # Show the L2 QCF-filtered flux as a date/time heatmap (gaps = flagged).
        ax = self.canvas.new_axes(1)[0]
        try:
            dv.plotting.HeatmapDateTime(series).plot(
                ax=ax, fig=self.canvas.fig, title=f"L2 filtered: {series.name}",
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
        return level2_to_code(
            self._init_kwargs(), self._level2_settings(),
            load_hint="dv.load_parquet('your_data.parquet')")

    def _copy_code(self) -> None:
        code = self._code()
        QApplication.clipboard().setText(code)
        self.summary.setPlainText(code)
