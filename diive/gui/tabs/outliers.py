"""
GUI.TABS.OUTLIERS: HAMPEL OUTLIER DETECTION TAB
===============================================

Run the library's Hampel filter (`dv.outliers.Hampel`) on a selected variable.
Keeps all three series: the **original** (the existing variable, untouched), the
**cleaned** series (`{var}_HAMPEL`, outliers set to NaN), and the **flag** that
produced it (`FLAG_{var}_OUTLIER_HAMPEL_TEST`, 0 = ok, 2 = outlier). "Add to
dataset" merges the cleaned + flag columns into the variable list (the same
mechanism the feature-engineering tab uses).

All detection is library work; this tab only collects parameters, runs Hampel on
a worker thread, previews the result, and emits the new columns.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_panel import VariablePanel


class _OutlierSignals(QObject):
    """Qt signals for the tab (DiiveTab is a plain ABC, not a QObject)."""
    run_done = Signal(object)
    run_failed = Signal(str)
    features_created = Signal(object)


class HampelOutlierTab(DiiveTab):
    """Detect outliers with the Hampel filter; keep original + cleaned + flag."""

    title = "Hampel filter"

    def build(self) -> QWidget:
        self._df = None
        self._var: str | None = None
        self._result_df = None  # cleaned + flag columns, pending "Add"
        self._sig = _OutlierSignals()
        #: Exposed bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        layout = QHBoxLayout(root)

        # Left: pick the variable to clean.
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(lambda name, _ctrl: self._select(name))
        layout.addWidget(self.varpanel)

        # Middle: Hampel settings + run/add.
        mid = self._build_settings()
        mid.setFixedWidth(290)
        layout.addWidget(mid)

        # Right: preview (original + outliers + cleaned).
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas, stretch=1)

        self._sig.run_done.connect(self._on_done)
        self._sig.run_failed.connect(self._on_failed)
        return root

    def _build_settings(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)

        intro = QLabel("Detect spikes with the Hampel filter (median absolute "
                       "deviation). Keeps the original, a cleaned copy, and the flag.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6B7780;")
        outer.addWidget(intro)

        form_box = QGroupBox("Hampel settings")
        form = QFormLayout(form_box)
        self.window = QSpinBox()
        self.window.setRange(3, 1_000_000)
        self.window.setValue(48 * 13)  # 13 days at 30-min sampling (Papale 2006)
        form.addRow("Window (records)", self.window)
        self.n_sigma = QDoubleSpinBox()
        self.n_sigma.setRange(0.1, 50.0)
        self.n_sigma.setSingleStep(0.5)
        self.n_sigma.setValue(5.5)
        form.addRow("n sigma", self.n_sigma)
        self.diff_cb = QCheckBox("Use double-differencing (Papale 2006)")
        self.diff_cb.setChecked(True)
        form.addRow(self.diff_cb)
        self.repeat_cb = QCheckBox("Repeat until no more outliers")
        self.repeat_cb.setChecked(True)
        form.addRow(self.repeat_cb)
        outer.addWidget(form_box)

        # Optional day/night separation (needs site coordinates).
        self.daynight_cb = QCheckBox("Separate daytime / nighttime")
        self.daynight_cb.toggled.connect(self._toggle_daynight)
        dn_box = QGroupBox("Day / night (optional)")
        dn = QFormLayout(dn_box)
        dn.addRow(self.daynight_cb)
        self.lat = QDoubleSpinBox(); self.lat.setRange(-90.0, 90.0); self.lat.setDecimals(4)
        self.lon = QDoubleSpinBox(); self.lon.setRange(-180.0, 180.0); self.lon.setDecimals(4)
        self.utc = QSpinBox(); self.utc.setRange(-12, 14)
        for w in (self.lat, self.lon, self.utc):
            w.setEnabled(False)
        dn.addRow("Latitude", self.lat)
        dn.addRow("Longitude", self.lon)
        dn.addRow("UTC offset (h)", self.utc)
        outer.addWidget(dn_box)

        self.run_btn = QPushButton("Detect outliers")
        self.run_btn.clicked.connect(self._run)
        outer.addWidget(self.run_btn)

        self.status = QLabel("Select a variable on the left.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.add_btn = QPushButton("Add cleaned + flag to dataset")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        outer.addWidget(self.add_btn)

        outer.addStretch(1)
        return panel

    def _toggle_daynight(self, on: bool) -> None:
        for w in (self.lat, self.lon, self.utc):
            w.setEnabled(on)

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
        self._result_df = None
        self.varpanel.set_panels([name])
        self.add_btn.setEnabled(False)
        self.status.setText(f"Selected '{name}'. Set parameters and detect outliers.")
        # Plot the raw series immediately (top panel); the cleaned panel fills in
        # after detection runs.
        self.varpanel.run_with_loading(name, lambda: self._draw(self._df[name]))

    # --- run ---
    def _run(self) -> None:
        if self._df is None or self._var is None:
            self.status.setText("Select a variable on the left first.")
            return
        kwargs = dict(
            window_length=self.window.value(),
            n_sigma=self.n_sigma.value(),
            use_differencing=self.diff_cb.isChecked(),
            separate_day_night=self.daynight_cb.isChecked(),
        )
        if self.daynight_cb.isChecked():
            kwargs.update(lat=self.lat.value(), lon=self.lon.value(),
                          utc_offset=self.utc.value())
        series = self._df[self._var]
        self.run_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.status.setText("Detecting outliers...")
        threading.Thread(
            target=self._worker,
            args=(series, kwargs, self.repeat_cb.isChecked()),
            daemon=True).start()

    def _worker(self, series, kwargs: dict, repeat: bool) -> None:
        try:
            h = dv.outliers.Hampel(series=series, **kwargs).run(repeat=repeat)
            cleaned = h.filteredseries.copy()
            cleaned.name = f"{series.name}_HAMPEL"
            flag = h.overall_flag.copy()  # name: FLAG_{var}_OUTLIER_HAMPEL_TEST
            result = pd.DataFrame({cleaned.name: cleaned, flag.name: flag})
            payload = {
                "var": series.name, "cleaned": cleaned, "flag": flag,
                "result": result, "n_outliers": int((flag == 2).sum()),
            }
        except Exception as err:  # surface the library error to the user
            self._sig.run_failed.emit(str(err))
            return
        self._sig.run_done.emit(payload)

    def _on_done(self, payload: dict) -> None:
        self.run_btn.setEnabled(True)
        self._result_df = payload["result"]
        self._draw(self._df[payload["var"]], flag=payload["flag"],
                   cleaned=payload["cleaned"], n_outliers=payload["n_outliers"])
        n = payload["n_outliers"]
        self.status.setText(
            f"{n} outliers flagged. 'Add' keeps {payload['cleaned'].name} and the flag "
            f"(original '{payload['var']}' is unchanged).")
        self.add_btn.setEnabled(True)

    def _on_failed(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(f"Failed: {msg}")

    def _draw(self, series, flag=None, cleaned=None, n_outliers: int = 0) -> None:
        """Two stacked panels: top = original (+ outlier markers once detected),
        bottom = the cleaned series. Called on variable select (series only) and
        after detection (with flag + cleaned)."""
        self.canvas.reset_layout()
        fig = self.canvas.fig
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top, sharey=ax_top)

        # Light, thin original line so the outlier markers stand out on top of it
        # (the series is dense, so a dark line would bury them).
        ax_top.plot(series.index, series.to_numpy(), color="#B0BEC5", lw=0.5,
                    alpha=0.8, label="original", zorder=1)
        if flag is not None:
            outliers = series[flag == 2]
            ax_top.plot(outliers.index, outliers.to_numpy(), linestyle="none",
                        marker="o", color="#E53935", ms=4, markeredgecolor="none",
                        alpha=0.85, zorder=5, label=f"outliers ({n_outliers})")
            ax_top.legend(loc="best", fontsize=8, framealpha=0.9)
        ax_top.set_title(f"{series.name} — original"
                         + (" + outliers" if flag is not None else ""), fontsize=9)

        if cleaned is not None:
            ax_bot.plot(cleaned.index, cleaned.to_numpy(), color="#43A047", lw=0.8,
                        zorder=1)
            ax_bot.set_title("cleaned (outliers removed)", fontsize=9)
        else:
            ax_bot.set_title("cleaned (run detection)", fontsize=9)
        self.canvas.draw()

    def _add(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        result = self._result_df
        self.featuresCreated.emit(result)  # MainWindow merges into the dataset
        self.status.setText(
            f"Added {', '.join(str(c) for c in result.columns)} to the variable list.")
        self.add_btn.setEnabled(False)
