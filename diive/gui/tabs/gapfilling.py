"""
GUI.TABS.GAPFILLING: XGBOOST GAP-FILLING
========================================

Gap-fill a target variable with gradient-boosted trees
(``dv.gapfilling.XGBoostTS``). Pick the target, then build the feature set with
the shared two-list picker (:class:`DualVariablePicker`): click a variable in
the left list to use it as a model feature, click it in the right list to drop
it. The selected variables are expanded by ``FeatureEngineer`` (lag, rolling,
timestamps, ...) before training.

All computation is library work (``FeatureEngineer`` + ``XGBoostTS``); this tab
only collects the inputs, runs them on a worker thread, previews the result
(observed vs. gap-filled heatmaps + model scores), and emits the gap-filled +
flag columns. Strict GUI<->library separation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading

import pandas as pd
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.gapfilling.xgboost_ts import XGBoostTS
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.dual_variable_picker import DualVariablePicker
from diive.gui.widgets.mpl_canvas import MplCanvas

_C_MUTED = "#6B7780"

#: below_zero options: label -> XGBoostTS value.
_BELOW_ZERO = {"Keep (default)": None, "Clip to zero": "zero", "Set to NaN": "nan"}


def _parse_ints(text: str) -> list[int]:
    return [int(p) for p in text.split(",") if p.strip()]


class _Signals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject)."""
    done = Signal(object)
    failed = Signal(str)
    features_created = Signal(object)


class XGBoostGapFillingTab(DiiveTab):
    """Select a target + features, configure XGBoost, gap-fill, emit the result."""

    title = "XGBoost gap-filling"

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._running = False
        self._result_df: pd.DataFrame | None = None  # columns to emit on "Add"

        self._sig = _Signals()
        self._sig.done.connect(self._on_done)
        self._sig.failed.connect(self._on_failed)
        #: Exposed bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Action bar.
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 8, 10, 8)
        title = QLabel(theme.manager.label_text(self.title))
        title.setFont(theme.manager.tracked_font(point_delta=1.0))
        title.setStyleSheet("font-weight: bold;")
        bar.addWidget(title)
        bar.addStretch(1)
        self.run_btn = QPushButton("Run gap-filling")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        bar.addWidget(self.run_btn)
        self.add_btn = QPushButton("Add results to dataset")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")
        bar.addWidget(self.add_btn)
        outer.addLayout(bar)

        # Three panes: inputs (target + feature picker) | settings | results.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_inputs())
        splitter.addWidget(self._build_settings())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        outer.addWidget(splitter)
        return root

    def _build_inputs(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(10, 6, 10, 6)

        intro = QLabel(
            "Choose the variable to gap-fill, then click variables on the left to "
            "use them as model features (click on the right to remove).")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(intro)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox()
        self.target_combo.setToolTip("The variable whose gaps will be filled.")
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        target_row.addWidget(self.target_combo, stretch=1)
        v.addLayout(target_row)

        self.picker = DualVariablePicker(
            available_title="Available", selected_title="Features",
            available_hint="click to use", selected_hint="click to remove")
        self.picker.changed.connect(self._update_status)
        v.addWidget(self.picker, stretch=1)
        return host

    def _build_settings(self) -> QWidget:
        host = QWidget()
        host.setFixedWidth(340)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(6, 6, 6, 6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        v = QVBoxLayout(inner)

        # --- Feature engineering ---
        # Selected variables are expanded into engineered features before
        # training. Defaults give a usable model out of the box.
        self.lag_cb = QCheckBox("Lag features")
        self.lag_cb.setChecked(True)
        self.lag_min = QSpinBox(); self.lag_min.setRange(-100, 0); self.lag_min.setValue(-2)
        self.lag_max = QSpinBox(); self.lag_max.setRange(-100, 100); self.lag_max.setValue(-1)
        self.lag_step = QSpinBox(); self.lag_step.setRange(1, 100); self.lag_step.setValue(1)
        v.addWidget(self._group("Lag", self.lag_cb, [
            ("Min lag", self.lag_min), ("Max lag", self.lag_max), ("Step", self.lag_step)]))

        self.roll_cb = QCheckBox("Rolling statistics")
        self.roll_cb.setChecked(True)
        self.roll_windows = QLineEdit("12,24")
        v.addWidget(self._group("Rolling", self.roll_cb, [("Windows", self.roll_windows)]))

        self.diff_cb = QCheckBox("Differencing")
        self.diff_orders = QLineEdit("1")
        v.addWidget(self._group("Differencing", self.diff_cb, [("Orders", self.diff_orders)]))

        self.ema_cb = QCheckBox("Exponential moving average")
        self.ema_spans = QLineEdit("6,24")
        v.addWidget(self._group("EMA", self.ema_cb, [("Spans", self.ema_spans)]))

        self.poly_cb = QCheckBox("Polynomial expansion")
        self.poly_deg = QSpinBox(); self.poly_deg.setRange(2, 5); self.poly_deg.setValue(2)
        v.addWidget(self._group("Polynomial", self.poly_cb, [("Degree", self.poly_deg)]))

        self.stl_cb = QCheckBox("STL decomposition")
        v.addWidget(self._group("STL", self.stl_cb, []))

        self.ts_cb = QCheckBox("Timestamp features (year, season, hour, ...)")
        self.ts_cb.setChecked(True)
        self.rec_cb = QCheckBox("Continuous record number")
        self.rec_cb.setChecked(True)
        misc = QGroupBox("Other features")
        ml = QVBoxLayout(misc)
        ml.addWidget(self.ts_cb)
        ml.addWidget(self.rec_cb)
        v.addWidget(misc)

        # --- XGBoost model ---
        model_box = QGroupBox("XGBoost model")
        mf = QFormLayout(model_box)
        self.n_estimators = QSpinBox(); self.n_estimators.setRange(1, 5000); self.n_estimators.setValue(100)
        mf.addRow("n_estimators", self.n_estimators)
        self.max_depth = QSpinBox(); self.max_depth.setRange(1, 50); self.max_depth.setValue(6)
        mf.addRow("max_depth", self.max_depth)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.001, 1.0); self.learning_rate.setDecimals(3)
        self.learning_rate.setSingleStep(0.01); self.learning_rate.setValue(0.1)
        mf.addRow("learning_rate", self.learning_rate)
        self.early_stop = QSpinBox(); self.early_stop.setRange(0, 1000); self.early_stop.setValue(10)
        self.early_stop.setToolTip("Early stopping rounds (0 = disabled).")
        mf.addRow("early_stopping", self.early_stop)
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.05, 0.9); self.test_size.setDecimals(2)
        self.test_size.setSingleStep(0.05); self.test_size.setValue(0.25)
        self.test_size.setToolTip("Fraction of complete data held out for the test score.")
        mf.addRow("test_size", self.test_size)
        self.random_state = QSpinBox(); self.random_state.setRange(0, 2_000_000); self.random_state.setValue(42)
        self.random_state.setToolTip(
            "Reproducibility seed. Without a fixed seed XGBoost reseeds every run "
            "and the output drifts between runs.")
        mf.addRow("random_state", self.random_state)
        self.below_zero = QComboBox(); self.below_zero.addItems(list(_BELOW_ZERO))
        self.below_zero.setToolTip(
            "How to treat predicted negative values. Keep for fluxes that can be "
            "negative (e.g. NEE); clip/NaN for non-negative variables (VPD, SW_IN).")
        mf.addRow("Negatives", self.below_zero)
        v.addWidget(model_box)

        # --- SHAP feature reduction ---
        self.reduce_cb = QCheckBox("Reduce features (SHAP importance)")
        self.shap_factor = QDoubleSpinBox()
        self.shap_factor.setRange(0.0, 5.0); self.shap_factor.setDecimals(2)
        self.shap_factor.setSingleStep(0.1); self.shap_factor.setValue(0.5)
        v.addWidget(self._group("Feature reduction", self.reduce_cb, [
            ("SHAP threshold factor", self.shap_factor)]))

        v.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)
        return host

    def _build_results(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.status = QLabel(
            "Pick a target and at least one feature, then run gap-filling.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding: 6px 10px; color: #444;")
        v.addWidget(self.status)
        self.canvas = MplCanvas()
        v.addWidget(self.canvas, stretch=1)
        return host

    @staticmethod
    def _group(title: str, checkbox: QCheckBox, rows: list) -> QGroupBox:
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        v.addWidget(checkbox)
        if rows:
            form = QFormLayout()
            for label, widget in rows:
                form.addRow(label, widget)
            v.addLayout(form)
        return box

    # --- data ----------------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._result_df = None
        self.add_btn.setEnabled(False)
        cols = [str(c) for c in df.columns]

        cur_target = self.target_combo.currentText()
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems(cols)
        if cur_target in cols:
            self.target_combo.setCurrentText(cur_target)
        self.target_combo.blockSignals(False)

        self.picker.set_variables(cols, created)
        self._exclude_target_from_features()
        self._update_status()

    def _on_target_changed(self, *_) -> None:
        # The target must not also be a feature of itself.
        self._exclude_target_from_features()
        self._update_status()

    def _exclude_target_from_features(self) -> None:
        target = self.target_combo.currentText()
        feats = [f for f in self.picker.selected_names() if f != target]
        if feats != self.picker.selected_names():
            self.picker.set_selected(feats)

    def _feature_cols(self) -> list[str]:
        target = self.target_combo.currentText()
        return [f for f in self.picker.selected_names() if f != target]

    def _update_status(self, *_) -> None:
        if self._running:
            return
        target = self.target_combo.currentText()
        feats = self._feature_cols()
        if not target:
            self.status.setText("Pick a target variable to gap-fill.")
        elif not feats:
            self.status.setText(
                f"Target: {target}. Now click variables on the left to use as features.")
        else:
            self.status.setText(
                f"Target: {target} — {len(feats)} feature(s) selected. Run gap-filling.")

    # --- state ---------------------------------------------------------
    def _controls(self) -> dict:
        return {"target": self.target_combo,
                "lag": self.lag_cb, "lag_min": self.lag_min, "lag_max": self.lag_max,
                "lag_step": self.lag_step, "roll": self.roll_cb,
                "roll_windows": self.roll_windows, "diff": self.diff_cb,
                "diff_orders": self.diff_orders, "ema": self.ema_cb,
                "ema_spans": self.ema_spans, "poly": self.poly_cb,
                "poly_deg": self.poly_deg, "stl": self.stl_cb, "ts": self.ts_cb,
                "rec": self.rec_cb, "n_estimators": self.n_estimators,
                "max_depth": self.max_depth, "learning_rate": self.learning_rate,
                "early_stop": self.early_stop, "test_size": self.test_size,
                "random_state": self.random_state, "below_zero": self.below_zero,
                "reduce": self.reduce_cb, "shap_factor": self.shap_factor}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"features": self.picker.selected_names(),
                "controls": save_controls(self._controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._controls(), state.get("controls"))
        feats = [f for f in (state.get("features") or []) if f in self.picker.all_names()]
        self.picker.set_selected(feats)
        self._exclude_target_from_features()
        self._update_status()

    # --- run -----------------------------------------------------------
    def _fe_kwargs(self) -> dict:
        kwargs: dict = {}
        if self.lag_cb.isChecked():
            kwargs["features_lag"] = [self.lag_min.value(), self.lag_max.value()]
            kwargs["features_lag_stepsize"] = self.lag_step.value()
        if self.roll_cb.isChecked():
            kwargs["features_rolling"] = _parse_ints(self.roll_windows.text())
        if self.diff_cb.isChecked():
            kwargs["features_diff"] = _parse_ints(self.diff_orders.text())
        if self.ema_cb.isChecked():
            kwargs["features_ema"] = _parse_ints(self.ema_spans.text())
        if self.poly_cb.isChecked():
            kwargs["features_poly_degree"] = self.poly_deg.value()
        if self.stl_cb.isChecked():
            kwargs["features_stl"] = True
        kwargs["vectorize_timestamps"] = self.ts_cb.isChecked()
        kwargs["add_continuous_record_number"] = self.rec_cb.isChecked()
        return kwargs

    def _xgb_kwargs(self) -> dict:
        kwargs: dict = {
            "test_size": self.test_size.value(),
            "below_zero": _BELOW_ZERO[self.below_zero.currentText()],
            "n_estimators": self.n_estimators.value(),
            "max_depth": self.max_depth.value(),
            "learning_rate": self.learning_rate.value(),
            "random_state": self.random_state.value(),
            "n_jobs": -1,
        }
        if self.early_stop.value() > 0:
            kwargs["early_stopping_rounds"] = self.early_stop.value()
        return kwargs

    def _run(self) -> None:
        if self._df is None or self._running:
            return
        target = self.target_combo.currentText()
        feats = self._feature_cols()
        if not target:
            self.status.setText("Pick a target variable first.")
            return
        if not feats:
            self.status.setText("Select at least one feature variable.")
            return
        try:
            fe_kwargs = self._fe_kwargs()
        except ValueError as err:
            self.status.setText(f"Invalid feature-engineering settings: {err}")
            return
        xgb_kwargs = self._xgb_kwargs()
        reduce = self.reduce_cb.isChecked()
        shap_factor = self.shap_factor.value()

        work = self._df[[target] + feats].copy()
        self._set_running(True)
        self.status.setText("Gap-filling… (training XGBoost — this can take a while)")
        threading.Thread(
            target=self._worker,
            args=(work, target, fe_kwargs, xgb_kwargs, reduce, shap_factor),
            daemon=True).start()

    def _worker(self, work, target, fe_kwargs, xgb_kwargs, reduce, shap_factor) -> None:
        try:
            engineered = FeatureEngineer(
                target_col=target, sanitize_timestamp=False, **fe_kwargs
            ).fit_transform(work)
            model = XGBoostTS(input_df=engineered, target_col=target,
                              verbose=1, **xgb_kwargs)
            if reduce:
                model.reduce_features(shap_threshold_factor=shap_factor)
            model.run(showplot_scores=False, showplot_importance=False)

            gapfilled = model.get_gapfilled_target()
            flag = model.get_flag()
            scores = model.scores_traintest_ or model.scores_

            out = pd.DataFrame({str(gapfilled.name): gapfilled,
                                str(flag.name): flag})
            attrs = {
                str(gapfilled.name): provenance_attr(
                    origin=DERIVED, parent=str(target), operation=self.title,
                    tags=["gapfilling", "xgboost"]),
                str(flag.name): provenance_attr(
                    origin=DERIVED, parent=str(target), operation=self.title,
                    tags=["gapfilling", "flag"]),
            }
            out.attrs[ATTRS_KEY] = attrs
            self._sig.done.emit((out, work[target], gapfilled, scores))
        except Exception as err:
            self._sig.failed.emit(str(err))

    def _set_running(self, on: bool) -> None:
        self._running = on
        self.run_btn.setEnabled(not on)
        if on:
            self.add_btn.setEnabled(False)

    # --- results -------------------------------------------------------
    def _on_done(self, payload) -> None:
        out, observed, gapfilled, scores = payload
        self._set_running(False)
        self._result_df = out
        n_filled = int((out[str(gapfilled.name)].notna()
                        & observed.isna()).sum())
        r2 = scores.get("r2")
        r2_txt = f"test R²={r2:.3f}, " if isinstance(r2, (int, float)) else ""
        self.status.setText(
            f"Done. {r2_txt}filled {n_filled} gaps. 'Add' appends "
            f"{', '.join(out.columns)}.")
        self.add_btn.setEnabled(True)
        self._plot(observed, gapfilled)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.status.setText(f"Failed: {msg}")
        ax = self.canvas.new_axes(1)[0]
        ax.text(0.5, 0.5, "Gap-filling failed", ha="center", va="center",
                transform=ax.transAxes)
        self.canvas.draw()

    def _plot(self, observed, gapfilled) -> None:
        """Side-by-side date/time heatmaps: observed (with gaps) vs gap-filled."""
        ax_obs, ax_gf = self.canvas.new_axes(2)
        try:
            dv.plotting.HeatmapDateTime(series=observed).plot(ax=ax_obs)
            ax_obs.set_title("Observed\n(with gaps)", fontsize=10, fontweight="bold")
            dv.plotting.HeatmapDateTime(series=gapfilled).plot(ax=ax_gf)
            ax_gf.set_title("XGBoost\ngap-filled", fontsize=10, fontweight="bold")
        except Exception as err:  # plotting must never crash the tab
            ax_obs.text(0.5, 0.5, f"Plot failed: {err}", ha="center", va="center",
                        transform=ax_obs.transAxes, fontsize=8)
        self.canvas.draw()

    def _add(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        result = self._result_df
        self.featuresCreated.emit(result)
        self.status.setText(f"Added {', '.join(result.columns)} to the variable list.")
        self.add_btn.setEnabled(False)
