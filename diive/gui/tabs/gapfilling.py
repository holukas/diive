"""
GUI.TABS.GAPFILLING: XGBOOST GAP-FILLING
========================================

Gap-fill one target variable with gradient-boosted trees
(``dv.gapfilling.XGBoostTS``). This is a thin, method-specific subclass of
``MlGapFillingTab`` (``tabs/_ml_gapfilling_base.py``) — the template that owns
the whole Model/Results layout, the three-list target/feature picker, the
performance hero, the heatmaps + SHAP table, the Results dashboard, and the
worker/emit flow shared by every ML gap-filler. Here we supply only the XGBoost
specifics: the model class, its hyperparameter widgets + kwargs, the save/restore
control map, and the reproducible-script renderer.

Build engineered features (lag / rolling / ...) in the *Feature engineering* tab
first if needed, then select them here — this tab carries no feature-engineering
settings, only the gap-filling model. All computation is library work; strict
GUI<->library separation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
)

from diive.gapfilling.codegen import (
    longterm_xgboost_gapfill_to_code,
    xgboost_gapfill_to_code,
)
from diive.gapfilling.longterm import LongTermGapFillingXGBoostTS
from diive.gapfilling.xgboost_ts import XGBoostTS
from diive.gui.tabs._ml_gapfilling_base import MlGapFillingTab

#: below_zero options: label -> XGBoostTS value.
_BELOW_ZERO = {"Keep (default)": None, "Clip to zero": "zero", "Set to NaN": "nan"}


class XGBoostGapFillingTab(MlGapFillingTab):
    """ML gap-filling tab specialised for XGBoost."""

    title = "XGBoost gap-filling"
    method_name = "XGBoost"
    method_chip_label = "XGBOOST"
    method_chip_bg = "#F3E5F5"   # lilac
    method_chip_fg = "#6A1B9A"

    # --- method hooks --------------------------------------------------
    def _model_class(self):
        return XGBoostTS

    def _longterm_model_class(self):
        return LongTermGapFillingXGBoostTS

    def _build_model_box(self) -> QGroupBox:
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
            "score; the rest is used for training. The hold-out is sampled RANDOMLY "
            "across the record (not a temporal block) — the appropriate test for "
            "gap-filling, which interpolates short scattered gaps from nearby "
            "observed data rather than extrapolating long unseen periods.")
        mf.addRow("test_size", self.test_size)
        self.random_state = QSpinBox()
        # -1 is the special "empty" value (shows "none") → no seed passed, so
        # XGBoost reseeds every run; any value >= 0 is a fixed reproducible seed.
        self.random_state.setRange(-1, 2_000_000)
        self.random_state.setSpecialValueText("none")
        self.random_state.setValue(42)
        self.random_state.setToolTip(
            "Reproducibility seed. Set to 'none' (spin below 0) to leave it unset — "
            "then XGBoost reseeds every run and the output drifts. Any fixed value "
            "makes runs reproducible.")
        mf.addRow("random_state", self.random_state)
        self.n_jobs = QSpinBox(); self.n_jobs.setRange(-1, 256); self.n_jobs.setValue(1)
        self.n_jobs.setToolTip(
            "Number of CPU cores used for training. 1 = single core (default); "
            "-1 = all available cores (faster, but uses the whole machine).")
        mf.addRow("n_jobs", self.n_jobs)
        self.below_zero = QComboBox(); self.below_zero.addItems(list(_BELOW_ZERO))
        self.below_zero.setToolTip(
            "How to treat predicted negative values. Keep for fluxes that can be "
            "negative (e.g. NEE); clip/NaN for non-negative variables (VPD, SW_IN).")
        mf.addRow("Negatives", self.below_zero)
        return model_box

    def _method_kwargs(self) -> dict:
        kwargs: dict = {
            "test_size": self.test_size.value(),
            "below_zero": _BELOW_ZERO[self.below_zero.currentText()],
            "n_estimators": self.n_estimators.value(),
            "max_depth": self.max_depth.value(),
            "learning_rate": self.learning_rate.value(),
            "n_jobs": self.n_jobs.value(),
        }
        # -1 = the "none" special value: leave random_state unset (XGBoost reseeds).
        if self.random_state.value() >= 0:
            kwargs["random_state"] = self.random_state.value()
        if self.early_stop.value() > 0:
            kwargs["early_stopping_rounds"] = self.early_stop.value()
        return kwargs

    def _method_controls(self) -> dict:
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "learning_rate": self.learning_rate, "early_stop": self.early_stop,
                "test_size": self.test_size, "random_state": self.random_state,
                "n_jobs": self.n_jobs, "below_zero": self.below_zero}

    def _codegen(self, target, features, kwargs, reduce, shap_factor) -> str:
        return xgboost_gapfill_to_code(
            target, features, kwargs, reduce=reduce,
            shap_threshold_factor=shap_factor)

    def _longterm_codegen(self, target, features, kwargs, reduce, shap_factor) -> str:
        return longterm_xgboost_gapfill_to_code(
            target, features, kwargs, reduce=reduce,
            shap_threshold_factor=shap_factor)
