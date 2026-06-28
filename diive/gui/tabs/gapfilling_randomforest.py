"""
GUI.TABS.GAPFILLING_RANDOMFOREST: RANDOM FOREST GAP-FILLING
==========================================================

Gap-fill one target variable with a random forest (``dv.gapfilling.RandomForestTS``).
A thin, method-specific subclass of ``MlGapFillingTab`` (``tabs/_ml_gapfilling_base.py``):
the template owns the whole Model/Results layout, the three-list target/feature
picker, the performance hero, the heatmaps + SHAP table, the Results dashboard,
and the worker/emit flow shared by every ML gap-filler. Here we supply only the
Random Forest specifics — the model class, its hyperparameter widgets + kwargs,
the save/restore control map, and the reproducible-script renderer.

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
    longterm_randomforest_gapfill_to_code,
    randomforest_gapfill_to_code,
)
from diive.gapfilling.longterm import LongTermGapFillingRandomForestTS
from diive.gapfilling.randomforest_ts import RandomForestTS
from diive.gui.tabs._ml_gapfilling_base import MlGapFillingTab

#: below_zero options: label -> RandomForestTS value.
_BELOW_ZERO = {"Keep (default)": None, "Clip to zero": "zero", "Set to NaN": "nan"}

#: max_features options: label -> RandomForestRegressor value.
_MAX_FEATURES = {"all (1.0)": 1.0, "sqrt": "sqrt", "log2": "log2"}


class RandomForestGapFillingTab(MlGapFillingTab):
    """ML gap-filling tab specialised for Random Forest."""

    title = "Random Forest gap-filling"
    method_name = "Random Forest"
    method_chip_label = "RANDOM FOREST"
    method_chip_bg = "#E8F5E9"   # green
    method_chip_fg = "#2E7D32"

    # --- method hooks --------------------------------------------------
    def _model_class(self):
        return RandomForestTS

    def _longterm_model_class(self):
        return LongTermGapFillingRandomForestTS

    def _build_model_box(self) -> QGroupBox:
        model_box = QGroupBox("Random Forest model")
        mf = QFormLayout(model_box)
        self.n_estimators = QSpinBox(); self.n_estimators.setRange(1, 5000); self.n_estimators.setValue(100)
        self.n_estimators.setToolTip(
            "Number of trees in the forest. More trees usually improve the fit and "
            "stability but slow training.")
        mf.addRow("n_estimators", self.n_estimators)
        self.max_depth = QSpinBox(); self.max_depth.setRange(0, 100)
        self.max_depth.setSpecialValueText("none")  # 0 -> None (unlimited)
        self.max_depth.setValue(0)
        self.max_depth.setToolTip(
            "Maximum depth of each tree. 'none' (spin to 0) grows trees until leaves "
            "are pure; a limit reduces overfitting.")
        mf.addRow("max_depth", self.max_depth)
        self.min_samples_split = QSpinBox(); self.min_samples_split.setRange(2, 1000)
        self.min_samples_split.setValue(2)
        self.min_samples_split.setToolTip(
            "Minimum number of samples required to split an internal node. Higher "
            "values make the trees more conservative.")
        mf.addRow("min_samples_split", self.min_samples_split)
        self.min_samples_leaf = QSpinBox(); self.min_samples_leaf.setRange(1, 1000)
        self.min_samples_leaf.setValue(1)
        self.min_samples_leaf.setToolTip(
            "Minimum number of samples required at a leaf node. Higher values smooth "
            "the model and reduce overfitting.")
        mf.addRow("min_samples_leaf", self.min_samples_leaf)
        self.max_features = QComboBox(); self.max_features.addItems(list(_MAX_FEATURES))
        self.max_features.setToolTip(
            "Number of features considered when looking for the best split. 'sqrt'/"
            "'log2' add randomness (more decorrelated trees); 'all' uses every feature.")
        mf.addRow("max_features", self.max_features)
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
        # -1 is the special "empty" value (shows "none") → no seed passed, so the
        # forest reseeds every run; any value >= 0 is a fixed reproducible seed.
        self.random_state.setRange(-1, 2_000_000)
        self.random_state.setSpecialValueText("none")
        self.random_state.setValue(42)
        self.random_state.setToolTip(
            "Reproducibility seed. Set to 'none' (spin below 0) to leave it unset — "
            "then the forest reseeds every run and the output drifts. Any fixed value "
            "makes runs reproducible.")
        mf.addRow("random_state", self.random_state)
        self.n_jobs = QSpinBox(); self.n_jobs.setRange(-1, 256); self.n_jobs.setValue(-1)
        self.n_jobs.setToolTip(
            "Number of CPU cores used for training. -1 = all available cores "
            "(default, faster); 1 = single core.")
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
            "min_samples_split": self.min_samples_split.value(),
            "min_samples_leaf": self.min_samples_leaf.value(),
            "max_features": _MAX_FEATURES[self.max_features.currentText()],
            "n_jobs": self.n_jobs.value(),
        }
        # 0 = the "none" special value: leave max_depth unset (trees grow fully).
        if self.max_depth.value() > 0:
            kwargs["max_depth"] = self.max_depth.value()
        # -1 = the "none" special value: leave random_state unset (forest reseeds).
        if self.random_state.value() >= 0:
            kwargs["random_state"] = self.random_state.value()
        return kwargs

    def _method_controls(self) -> dict:
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features, "test_size": self.test_size,
                "random_state": self.random_state, "n_jobs": self.n_jobs,
                "below_zero": self.below_zero}

    def _codegen(self, target, features, kwargs, reduce, shap_factor) -> str:
        return randomforest_gapfill_to_code(
            target, features, kwargs, reduce=reduce,
            shap_threshold_factor=shap_factor)

    def _longterm_codegen(self, target, features, kwargs, reduce, shap_factor) -> str:
        return longterm_randomforest_gapfill_to_code(
            target, features, kwargs, reduce=reduce,
            shap_threshold_factor=shap_factor)
