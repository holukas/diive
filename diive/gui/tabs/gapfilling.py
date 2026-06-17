"""
GUI.TABS.GAPFILLING: XGBOOST GAP-FILLING
========================================

A focused tab to gap-fill one target variable with gradient-boosted trees
(``dv.gapfilling.XGBoostTS``). Three variable lists side by side:

  1. **Variables** (far left) — click a variable to set it as the gap-fill target.
  2. **Available features** (middle) — click a variable to use it as a model feature.
  3. **Selected features** (right) — the chosen features; click one to drop it.

The model is trained on the selected feature columns directly. Build engineered
features (lag / rolling / ...) in the *Feature engineering* tab first if needed,
then select them here — this tab deliberately carries no feature-engineering
settings, only the gap-filling model.

All computation is library work (``XGBoostTS``); this tab only collects the
inputs, runs them on a worker thread, previews the result (observed vs.
gap-filled heatmaps + model score), and emits the gap-filled + flag columns.
Strict GUI<->library separation.

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
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.gapfilling.xgboost_ts import XGBoostTS
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
# Reuse the Overview's hero building blocks (presentation-only helpers).
from diive.gui.tabs.overview import _MetricSlot, _chip_qss, _stat_separator
from diive.gui.widgets.dual_variable_picker import DualVariablePicker
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_panel import VariablePanel

_C_MUTED = "#6B7780"

#: below_zero options: label -> XGBoostTS value.
_BELOW_ZERO = {"Keep (default)": None, "Clip to zero": "zero", "Set to NaN": "nan"}


class _Signals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject)."""
    done = Signal(object)
    failed = Signal(str)
    features_created = Signal(object)


class _PerformanceHero(QFrame):
    """Hero band above the plots showing the gap-filling model's performance.

    Same visual pattern as the Overview's `_HeroBand` (identity row + a row of
    persistent `_MetricSlot`s with hairline separators), but the metrics are the
    held-out (test-set) model scores from gap-filling, not variable statistics.
    Built once; `set_metrics` updates the slot text in place (no flicker)."""

    #: (label, tooltip) per metric slot, in display order.
    _METRICS = [
        ("R²", "Coefficient of determination on the held-out test set (1 = perfect)."),
        ("RMSE", "Root mean squared error on the test set, in target units."),
        ("MAE", "Mean absolute error on the test set, in target units."),
        ("MAPE", "Mean absolute percentage error on the test set."),
        ("MAXE", "Maximum absolute error on the test set, in target units."),
        ("FILLED", "Number of gap records filled by the model."),
        ("FEATURES", "Number of feature variables used to train the model."),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("perfhero")
        border = theme.manager.tokens["BORDER"]
        self.setStyleSheet(
            f"QFrame#perfhero {{ background: #FFFFFF; border: 1px solid {border};"
            f" border-radius: 10px; }}")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 11, 16, 11)
        lay.setSpacing(8)

        # --- identity row ---
        idrow = QHBoxLayout()
        idrow.setSpacing(8)
        title = QLabel("Gap-filling performance")
        tf = title.font()
        tf.setPointSizeF(tf.pointSizeF() + 3.0)
        tf.setBold(True)
        title.setFont(tf)
        idrow.addWidget(title)
        self._target = QLabel()
        idrow.addWidget(self._target)
        self._method = QLabel("XGBOOST")
        self._method.setStyleSheet(_chip_qss("#F3E5F5", "#6A1B9A"))
        idrow.addWidget(self._method)
        idrow.addStretch(1)
        lay.addLayout(idrow)

        # --- metric row (persistent slots) ---
        self._slots: dict[str, _MetricSlot] = {}
        rowlay = QHBoxLayout()
        rowlay.setSpacing(14)
        for i, (label, _tip) in enumerate(self._METRICS):
            if i > 0:
                rowlay.addWidget(_stat_separator())
            slot = _MetricSlot()
            self._slots[label] = slot
            rowlay.addWidget(slot)
        rowlay.addStretch(1)
        lay.addLayout(rowlay)

        self.reset()

    def reset(self, target: str = "") -> None:
        """Blank the metrics (placeholder dashes); show the target if known."""
        if target:
            self._target.setText(target)
            self._target.setStyleSheet(_chip_qss("#E3F2FD", "#1565C0"))
            self._target.show()
        else:
            self._target.hide()
        for label, tip in self._METRICS:
            self._slots[label].update_metric(label, "—", tip)

    def set_metrics(self, target: str, n_features: int, n_filled: int,
                    scores: dict) -> None:
        self._target.setText(target)
        self._target.setStyleSheet(_chip_qss("#E3F2FD", "#1565C0"))
        self._target.show()

        def g(key: str) -> str:
            v = scores.get(key)
            return f"{v:.3g}" if isinstance(v, (int, float)) else "—"

        values = {
            "R²": g("r2"), "RMSE": g("rmse"), "MAE": g("mae"),
            "MAPE": g("mape"), "MAXE": g("maxe"),
            "FILLED": f"{n_filled:,}", "FEATURES": f"{n_features}",
        }
        for label, tip in self._METRICS:
            self._slots[label].update_metric(label, values.get(label, "—"), tip)


class XGBoostGapFillingTab(DiiveTab):
    """Pick a target + features in three lists, configure XGBoost, gap-fill, emit."""

    title = "XGBoost gap-filling"

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._all_cols: list[str] = []
        self._created: set = set()
        self._target: str = ""
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
        self.run_btn.setToolTip("Train XGBoost on the selected features and fill the target's gaps.")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        bar.addWidget(self.run_btn)
        self.add_btn = QPushButton("Add results to dataset")
        self.add_btn.setToolTip("Append the gap-filled series and its flag to the variable list.")
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
            "Click a variable in 'Variables' to set the gap-fill target. Click a "
            "variable in 'Available features' to use it as a model feature; click "
            "one in 'Selected features' to remove it.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(intro)

        row = QHBoxLayout()

        # --- target list (far left) ---
        tcol = QVBoxLayout()
        tcol.addWidget(self._list_header("Variables", "click to set target"))
        self.target_list = VariablePanel()
        self.target_list.list.setToolTip("Click a variable to set it as the gap-fill target.")
        self.target_list.selected.connect(lambda name, _c: self._set_target(name))
        tcol.addWidget(self.target_list, stretch=1)
        row.addLayout(tcol)

        # --- available features (middle) | selected features (right) ---
        self.picker = DualVariablePicker(
            available_title="Available features", selected_title="Selected features",
            available_hint="click to use", selected_hint="click to remove")
        self.picker.changed.connect(self._update_status)
        row.addWidget(self.picker, stretch=1)

        v.addLayout(row, stretch=1)

        self.target_label = QLabel("Target: (none)")
        self.target_label.setStyleSheet("font-weight: bold;")
        v.addWidget(self.target_label)
        return host

    def _build_settings(self) -> QWidget:
        host = QWidget()
        host.setFixedWidth(320)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(6, 6, 6, 6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        v = QVBoxLayout(inner)

        # --- XGBoost model ---
        model_box = QGroupBox("XGBoost model")
        mf = QFormLayout(model_box)
        self.n_estimators = QSpinBox(); self.n_estimators.setRange(1, 5000); self.n_estimators.setValue(100)
        self.n_estimators.setToolTip(
            "Number of boosting rounds (trees). More can improve the fit but slows "
            "training and may overfit.")
        mf.addRow("n_estimators", self.n_estimators)
        self.max_depth = QSpinBox(); self.max_depth.setRange(1, 50); self.max_depth.setValue(6)
        self.max_depth.setToolTip(
            "Maximum depth of each tree. Higher captures more feature interactions "
            "but risks overfitting.")
        mf.addRow("max_depth", self.max_depth)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.001, 1.0); self.learning_rate.setDecimals(3)
        self.learning_rate.setSingleStep(0.01); self.learning_rate.setValue(0.1)
        self.learning_rate.setToolTip(
            "Step-size shrinkage per boosting round. Lower values usually generalize "
            "better but need more trees.")
        mf.addRow("learning_rate", self.learning_rate)
        self.early_stop = QSpinBox(); self.early_stop.setRange(0, 1000); self.early_stop.setValue(10)
        self.early_stop.setToolTip(
            "Stop adding trees when the validation score has not improved for this "
            "many rounds (0 = disabled).")
        mf.addRow("early_stopping", self.early_stop)
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.05, 0.9); self.test_size.setDecimals(2)
        self.test_size.setSingleStep(0.05); self.test_size.setValue(0.25)
        self.test_size.setToolTip(
            "Fraction of complete records held out to compute the (honest) test "
            "score; the rest is used for training.")
        mf.addRow("test_size", self.test_size)
        self.random_state = QSpinBox(); self.random_state.setRange(0, 2_000_000); self.random_state.setValue(42)
        self.random_state.setToolTip(
            "Reproducibility seed. Without a fixed seed XGBoost reseeds every run and "
            "the output drifts between runs.")
        mf.addRow("random_state", self.random_state)
        self.below_zero = QComboBox(); self.below_zero.addItems(list(_BELOW_ZERO))
        self.below_zero.setToolTip(
            "How to treat predicted negative values. Keep for fluxes that can be "
            "negative (e.g. NEE); clip/NaN for non-negative variables (VPD, SW_IN).")
        mf.addRow("Negatives", self.below_zero)
        v.addWidget(model_box)

        # --- SHAP feature reduction ---
        red_box = QGroupBox("Feature reduction")
        rv = QVBoxLayout(red_box)
        self.reduce_cb = QCheckBox("Reduce features (SHAP importance)")
        self.reduce_cb.setToolTip(
            "Before the final model, drop features whose SHAP importance is below a "
            "random-baseline threshold.")
        rv.addWidget(self.reduce_cb)
        rf = QFormLayout()
        self.shap_factor = QDoubleSpinBox()
        self.shap_factor.setRange(0.0, 5.0); self.shap_factor.setDecimals(2)
        self.shap_factor.setSingleStep(0.1); self.shap_factor.setValue(0.5)
        self.shap_factor.setToolTip(
            "Higher = stricter reduction (keeps fewer features). Threshold = random "
            "importance + factor × random SD.")
        rf.addRow("SHAP threshold factor", self.shap_factor)
        rv.addLayout(rf)
        v.addWidget(red_box)

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
            "Set a target and at least one feature, then run gap-filling.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding: 6px 10px; color: #444;")
        v.addWidget(self.status)
        # Hero band with the model's performance metrics, above the plots.
        self.hero = _PerformanceHero()
        hwrap = QWidget()
        hl = QVBoxLayout(hwrap)
        hl.setContentsMargins(10, 0, 10, 6)
        hl.addWidget(self.hero)
        v.addWidget(hwrap)
        self.canvas = MplCanvas()
        v.addWidget(self.canvas, stretch=1)
        return host

    @staticmethod
    def _list_header(title: str, hint: str) -> QLabel:
        label = QLabel(f"<b>{title}</b> <span style='color:#90A4AE'>({hint})</span>")
        label.setWordWrap(True)
        return label

    # --- data ----------------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._created = set(created or set())
        self._all_cols = [str(c) for c in df.columns]
        self._result_df = None
        self.add_btn.setEnabled(False)

        self.target_list.set_variables(self._all_cols, self._created)
        if self._target not in self._all_cols:
            self._target = ""
        self._refresh_features()
        self.target_list.set_panels([self._target] if self._target else [])
        self._update_target_label()
        self._update_status()

    def _set_target(self, name: str) -> None:
        if not name or name not in self._all_cols:
            return
        self._target = name
        self.target_list.set_panels([name])
        self._refresh_features()  # drop the target from the feature pool
        self._update_target_label()
        self._update_status()

    def _refresh_features(self) -> None:
        """Feature pool = all columns except the target (keeps current selection)."""
        pool = [c for c in self._all_cols if c != self._target]
        self.picker.set_variables(pool, self._created)

    def _update_target_label(self) -> None:
        self.target_label.setText(f"Target: {self._target or '(none)'}")

    def _feature_cols(self) -> list[str]:
        return [f for f in self.picker.selected_names() if f != self._target]

    def _update_status(self, *_) -> None:
        if self._running:
            return
        feats = self._feature_cols()
        if not self._target:
            self.status.setText("Click a variable in 'Variables' to set the gap-fill target.")
        elif not feats:
            self.status.setText(
                f"Target: {self._target}. Now click variables in 'Available features'.")
        else:
            self.status.setText(
                f"Target: {self._target} — {len(feats)} feature(s) selected. Run gap-filling.")

    # --- state ---------------------------------------------------------
    def _controls(self) -> dict:
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "learning_rate": self.learning_rate, "early_stop": self.early_stop,
                "test_size": self.test_size, "random_state": self.random_state,
                "below_zero": self.below_zero, "reduce": self.reduce_cb,
                "shap_factor": self.shap_factor}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"target": self._target,
                "features": self.picker.selected_names(),
                "controls": save_controls(self._controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._controls(), state.get("controls"))
        tgt = state.get("target")
        if tgt in self._all_cols:
            self._set_target(tgt)
        feats = [f for f in (state.get("features") or []) if f in self.picker.all_names()]
        self.picker.set_selected(feats)
        self._update_status()

    # --- run -----------------------------------------------------------
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
        target = self._target
        feats = self._feature_cols()
        if not target:
            self.status.setText("Set a target variable first.")
            return
        if not feats:
            self.status.setText("Select at least one feature variable.")
            return
        xgb_kwargs = self._xgb_kwargs()
        reduce = self.reduce_cb.isChecked()
        shap_factor = self.shap_factor.value()

        work = self._df[[target] + feats].copy()
        self._set_running(True)
        self.status.setText("Gap-filling… (training XGBoost — this can take a while)")
        threading.Thread(
            target=self._worker,
            args=(work, target, xgb_kwargs, reduce, shap_factor),
            daemon=True).start()

    def _worker(self, work, target, xgb_kwargs, reduce, shap_factor) -> None:
        try:
            model = XGBoostTS(input_df=work, target_col=target,
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
            n_features = work.shape[1] - 1  # everything except the target
            self._sig.done.emit(
                (out, work[target], gapfilled, scores, target, n_features))
        except Exception as err:
            self._sig.failed.emit(str(err))

    def _set_running(self, on: bool) -> None:
        self._running = on
        self.run_btn.setEnabled(not on)
        if on:
            self.add_btn.setEnabled(False)

    # --- results -------------------------------------------------------
    def _on_done(self, payload) -> None:
        out, observed, gapfilled, scores, target, n_features = payload
        self._set_running(False)
        self._result_df = out
        n_filled = int((out[str(gapfilled.name)].notna() & observed.isna()).sum())
        self.hero.set_metrics(target=target, n_features=n_features,
                              n_filled=n_filled, scores=scores)
        self.status.setText(f"Done. 'Add' appends {', '.join(out.columns)}.")
        self.add_btn.setEnabled(True)
        self._plot(observed, gapfilled)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.hero.reset(self._target)
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
