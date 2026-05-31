"""
DRIVERANALYSIS: EVIDENCE-TRIANGULATION DRIVER ATTRIBUTION
========================================================

Attribute a flux time series to its candidate drivers and organize every output
by *epistemic level* — association, temporal prediction, causation — rather than
as a co-equal pile of importance plots.

The scientific value is threefold: (1) shared, time-aware, eddy-covariance-correct
preprocessing feeds every method; (2) atemporal methods (SHAP, ALE) are given a
temporal treatment (lags, timescales, regimes) instead of throwing time away; and
(3) a convergence/divergence synthesis surfaces where methods *disagree* — which
is the actual scientific signal.

Hard rules honored here:
  - SHAP and ALE are association-level diagnostics and are NEVER labeled causal.
  - All train/test splits are time-aware (no shuffling) to avoid the leakage that
    diive recently fixed in its gap-fillers and OptimizeParamsTS.
  - Model scores are held-out (out-of-sample), never in-sample.
  - Heavy causal libraries (tigramite, econml/causalml) are optional extras
    (``diive[causal]``), lazy-imported only when their method is called.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.analysis.driveranalysis.ale import (
    AleCurve, Ale2DResult, accumulated_local_effects, accumulated_local_effects_2d,
)
from diive.core.utils.console import info, warn, success, rule, detail, error

# Material Design palette (CLAUDE.md plotting conventions).
_MD = {
    'green': '#4CAF50', 'green_bg': '#C8E6C9',
    'red': '#F44336', 'red_bg': '#FFCDD2',
    'amber': '#FFC107', 'amber_bg': '#FFECB3',
    'blue': '#2196F3', 'blue_bg': '#BBDEFB',
    'grey': '#455A64', 'grey_bg': '#CFD8DC',
}

# Name of the injected pure-noise benchmark column. Every model is trained with
# this column present; a real driver must score *above* it on SHAP importance to
# count as relevant. It is the noise floor that turns raw importances into a
# yes/weak/no decision. (Borrowed from diive's gap-filler feature-reduction path.)
_RANDOM_COL = '.RANDOM'

# ---------------------------------------------------------------------------
# How the pieces fit together (read this first):
#
#   1. fit_model()        trains ONE headline model on all drivers (+ .RANDOM),
#                         time-aware split, and scores it out-of-sample.
#   2. shap() / ale()     interrogate that one model -> Layer 1 (association).
#   3. lagged/scale/strat fit ADDITIONAL throwaway models on transformed data
#                         (lagged drivers, STL components, per-regime subsets)
#                         -> Layer 2 (temporal). Each returns per-driver SHAP.
#   4. granger/pcmci/cate optional Layer 3 (causation), opt-in + heavy deps.
#   5. _synthesize()      collapses every method's per-driver output into ONE
#                         ternary relevance ({yes,weak,no}) + direction, then
#                         compares them -> agreement + verdict (the headline).
#
# The recurring primitive is "(drivers, target) -> per-driver SHAP importance",
# implemented by _fit_importance(); the synthesis is just bookkeeping on top.
# ---------------------------------------------------------------------------


def _stl_components(series: Series, period: int, robust: bool = True):
    """STL trend/seasonal/residual via statsmodels with an explicit period.

    ``SeasonalTrendDecomposition`` infers the period from the index frequency,
    which statsmodels cannot do for sub-daily (e.g. 30-min) eddy-covariance data.
    Passing ``period`` explicitly (records per seasonal cycle — a daily cycle by
    default) makes STL reliable here.
    """
    from statsmodels.tsa.seasonal import STL
    s = series.interpolate(limit_direction='both').dropna()
    if len(s) < 2 * period:
        raise ValueError(f"series too short ({len(s)}) for STL period {period}")
    res = STL(s.to_numpy(), period=period, robust=robust).fit()
    idx = s.index
    return (pd.Series(res.trend, index=idx, name=series.name),
            pd.Series(res.seasonal, index=idx, name=series.name),
            pd.Series(res.resid, index=idx, name=series.name))


def _mean_abs_shap(model, X: DataFrame) -> Series:
    """Mean |SHAP| per feature for a fitted tree model.

    Reuses the gap-filler's ``_build_tree_explainer`` so the XGBoost ``base_score``
    parsing workaround applies here too.
    """
    import numpy as _np
    from diive.core.ml.common import MlRegressorGapFillingBase
    explainer = MlRegressorGapFillingBase._build_tree_explainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    importance = _np.abs(shap_values).mean(axis=0)
    return pd.Series(importance, index=list(X.columns))

# Verdict colors for the synthesis table / convergence plot.
_RELEVANCE_COLOR = {'yes': _MD['green'], 'weak': _MD['grey'], 'no': _MD['red']}


@dataclass
class DriverAnalysisResult:
    """Container for all outputs of a :class:`DriverAnalysis` run.

    Causal-layer fields are ``None`` unless ``'causal'`` was requested. Temporal
    fields are ``None`` unless ``'temporal'`` was requested.
    """
    # --- shared substrate ---
    model: object
    model_scores: dict                                  # HELD-OUT metrics
    drivers: list = field(default_factory=list)
    levels_run: list = field(default_factory=list)
    # --- association layer ---
    shap_importance: Optional[DataFrame] = None
    shap_interactions: Optional[DataFrame] = None
    ale: dict = field(default_factory=dict)             # driver -> AleCurve
    ale_2d: Optional[dict] = None                       # (f1, f2) -> Ale2DResult
    # --- temporal layer ---
    lagged_importance: Optional[DataFrame] = None       # index=driver, cols=lags
    scale_resolved: Optional[DataFrame] = None          # index=driver, cols=scales
    stratified: Optional[DataFrame] = None              # index=driver, cols=regimes
    rolling_importance: Optional[DataFrame] = None
    # --- causal layer ---
    granger: Optional[DataFrame] = None
    pcmci: object = None
    cate: object = None
    # --- synthesis ---
    convergence: Optional[DataFrame] = None
    stability: Optional[DataFrame] = None


# Module-level latch so the ExperimentalWarning fires only once per process,
# no matter how many DriverAnalysis instances are created.
_EXPERIMENTAL_WARNED = False


class DriverAnalysis:
    """Evidence-triangulation driver attribution for flux time series.

    .. warning::

        EXPERIMENTAL / provisional. Lives in ``dv.analysis.experimental`` (not the
        stable ``dv.analysis`` namespace); the API and the convergence-table
        schema may change without a deprecation cycle. Instantiating this class
        emits a one-time :class:`ExperimentalWarning`.

    Two-phase: ``__init__`` takes data + computation config; :meth:`run` does the
    work; results live on :attr:`results`; :meth:`summary` and the ``plot_*``
    methods render. Outputs are organized by epistemic level
    (association | temporal-prediction | causation) — SHAP/ALE are never
    presented as causal.

    Args:
        target: Flux series to attribute (e.g. ``NEE_CUT_REF_orig``). Consider
            partitioning NEE into GPP + Reco *before* driver analysis — their
            drivers differ and often oppose.
        drivers: Candidate drivers (e.g. TA, VPD, SW_IN, SWC, USTAR).
        model: ``'rf'`` | ``'xgb'`` | any unfitted sklearn-compatible regressor.
        feature_engineer: Optional pre-built ``FeatureEngineer``; ``None`` uses
            raw drivers. When given, per-driver SHAP is aggregated over that
            driver's engineered columns.
        lags: Lags in records to test per driver for :meth:`lagged_importance`,
            e.g. ``list(range(-48, 1))``. ``None`` disables the lagged analysis.
        deseasonalize: STL-deseasonalize target and drivers up front (recommended
            before causal methods).
        test_size: Held-out fraction for out-of-sample scoring.
        time_aware_split: Use a chronological holdout (no shuffling). Leave True.
        n_bootstrap: ``>0`` enables bootstrap stability of SHAP rankings.
        random_state: Seed.
        verbose: Verbosity (0=silent .. 3=debug).
        ale_grid_size: Quantile bins for 1D ALE.
        ale_range_threshold: ALE range below which a curve is "flat" (no response).
            ``None`` defaults to ``0.1 * target.std()``.
        granger_alpha: Significance level for Granger / relevance gating.
        top_k: Rank threshold for bootstrap stability ("frac of resamples in top-k").
        stl_period: Seasonal period (records) for STL; ``None`` infers a daily cycle.
    """

    def __init__(
        self,
        target: Series,
        drivers: DataFrame,
        model: Union[str, object] = 'rf',
        *,
        feature_engineer: object = None,
        lags: Optional[list] = None,
        deseasonalize: bool = False,
        test_size: float = 0.25,
        time_aware_split: bool = True,
        n_bootstrap: int = 0,
        random_state: int = 42,
        verbose: int = 2,
        ale_grid_size: int = 20,
        ale_range_threshold: Optional[float] = None,
        granger_alpha: float = 0.05,
        top_k: int = 3,
        stl_period: Optional[int] = None,
    ):
        # Announce experimental status once per process (see _EXPERIMENTAL_WARNED).
        global _EXPERIMENTAL_WARNED
        if not _EXPERIMENTAL_WARNED:
            import warnings
            from diive.analysis.driveranalysis import ExperimentalWarning
            warnings.warn(
                "DriverAnalysis is experimental: its API and the convergence-table "
                "schema may change without a deprecation cycle. It lives in "
                "dv.analysis.experimental until it stabilizes.",
                ExperimentalWarning, stacklevel=2)
            _EXPERIMENTAL_WARNED = True

        if not isinstance(target, Series):
            raise TypeError("target must be a pandas Series.")
        if not isinstance(drivers, DataFrame):
            raise TypeError("drivers must be a pandas DataFrame.")
        if target.name is None:
            raise ValueError("target Series must have a name.")
        if target.name in drivers.columns:
            raise ValueError(f"target name '{target.name}' collides with a driver column.")

        self.verbose = verbose
        self.model_spec = model
        self.feature_engineer = feature_engineer
        self.lags = list(lags) if lags is not None else None
        self.deseasonalize = deseasonalize
        self.test_size = test_size
        self.time_aware_split = time_aware_split
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.ale_grid_size = ale_grid_size
        self.granger_alpha = granger_alpha
        self.top_k = top_k

        # Align target and drivers on a common index. Everything downstream
        # assumes target and drivers share exactly this index.
        common = target.index.intersection(drivers.index)
        self.target = target.loc[common].copy()
        self.drivers_df = drivers.loc[common].copy()
        self.driver_names = list(self.drivers_df.columns)
        # Records per seasonal (daily) cycle — needed by every STL call, since
        # statsmodels can't infer the period for sub-daily data.
        self._stl_period = stl_period if stl_period else self._infer_daily_period(self.target.index)

        # Optional shared preprocessing: strip the seasonal cycle up front so
        # every layer sees deseasonalized data (recommended before causal tests).
        if self.deseasonalize:
            self._apply_deseasonalize()

        # An ALE curve whose vertical range is below this is treated as "flat"
        # (no response). Scaled to the target's spread so it is unit-agnostic.
        self.ale_range_threshold = (ale_range_threshold if ale_range_threshold is not None
                                    else 0.1 * float(self.target.std()))

        # Populated lazily by run()/fit_model(); kept on the instance so SHAP and
        # ALE can reuse the single fitted model without retraining.
        self.model_ = None
        self._X = None          # full feature matrix (incl. .RANDOM), modeling rows
        self._y = None
        self._X_attrib = None   # matrix ALE perturbs over (also incl. .RANDOM)
        self._random_baseline = None     # mean|SHAP| of .RANDOM = the noise floor
        self._stratified_directions = None  # {regime: {driver: ALE direction}}
        self._result = DriverAnalysisResult(model=None, model_scores={},
                                            drivers=self.driver_names)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _infer_daily_period(index) -> int:
        """Records per day from the index frequency (fallback 48 = 30-min)."""
        if len(index) < 3:
            return 48
        deltas = np.diff(index.values).astype('timedelta64[s]').astype(float)
        step_s = float(np.median(deltas))
        if step_s <= 0:
            return 48
        return max(2, int(round(86400.0 / step_s)))

    def _apply_deseasonalize(self):
        """Replace target and drivers with their STL-deseasonalized series."""
        info("Deseasonalizing target and drivers via STL (trend + residual).",
             verbose=self.verbose)

        def _ds(s: Series) -> Series:
            trend, _seasonal, resid = _stl_components(s, self._stl_period)
            out = trend + resid
            out.name = s.name
            return out.reindex(s.index)

        self.target = _ds(self.target)
        for c in self.driver_names:
            self.drivers_df[c] = _ds(self.drivers_df[c])

    def _new_model(self):
        """Return a fresh, unfitted regressor per ``model`` spec."""
        spec = self.model_spec
        if isinstance(spec, str):
            if spec == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                             random_state=self.random_state, n_jobs=-1)
            if spec == 'xgb':
                from xgboost import XGBRegressor
                return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=self.random_state,
                                    n_jobs=-1, verbosity=0)
            raise ValueError(f"Unknown model '{spec}'. Use 'rf', 'xgb', or a regressor instance.")
        from sklearn.base import clone
        return clone(spec)

    def _build_matrix(self, drivers_df: DataFrame, target: Series,
                      add_random: bool = True) -> tuple[DataFrame, Series]:
        """Build an aligned, NaN-free (features, target) pair.

        Applies the optional FeatureEngineer, appends a ``.RANDOM`` benchmark
        column, and drops rows with any missing value (time-aware: no shuffling).
        """
        # Optionally expand raw drivers into the 8-stage engineered feature set.
        # The FeatureEngineer needs the target column present during transform,
        # so we attach it, transform, then drop it back out.
        if self.feature_engineer is not None:
            tname = target.name
            eng_in = drivers_df.copy()
            eng_in[tname] = target
            self.feature_engineer.target_col = tname
            engineered = self.feature_engineer.fit_transform(eng_in)
            feats = engineered.drop(columns=[tname], errors='ignore')
        else:
            feats = drivers_df.copy()

        # Inject the pure-noise benchmark column (the relevance noise floor).
        if add_random:
            rng = np.random.RandomState(self.random_state)
            feats[_RANDOM_COL] = rng.randn(len(feats))

        # Listwise-drop rows with any NaN. We do NOT shuffle: row order stays
        # chronological so the caller's time-aware holdout remains leak-free.
        df = feats.copy()
        df[target.name] = target
        df = df.dropna()
        y = df[target.name]
        X = df.drop(columns=[target.name])
        return X, y

    def _feature_to_driver(self, feature: str) -> Optional[str]:
        """Map a (possibly engineered/lagged) feature column to its parent driver.

        Engineered columns look like ``.TA_MEAN12`` or ``.VPD-2``; we strip the
        leading dot and find the driver whose name prefixes it. Sorting by length
        descending ensures the most specific match wins, so a driver named 'TA'
        never steals a column that actually belongs to 'Tair_f'.
        """
        if feature == _RANDOM_COL:
            return None  # the noise benchmark belongs to no driver
        if feature in self.driver_names:
            return feature  # raw, untransformed driver column
        stripped = feature[1:] if feature.startswith('.') else feature
        for d in sorted(self.driver_names, key=len, reverse=True):
            if stripped == d or stripped.startswith(d) or d in feature:
                return d
        return None

    def _shap_per_driver(self, model, X: DataFrame) -> tuple[Series, float]:
        """Per-driver SHAP importance (summed over engineered cols) + RANDOM baseline.

        A driver may appear as several engineered columns (lags, rolling means,
        ...). Its overall importance is the SUM of |SHAP| over all of them, so a
        driver isn't penalised for having its signal spread across variants.
        Returns the per-driver Series plus the .RANDOM value (the noise floor).
        """
        imp = _mean_abs_shap(model, X)
        random_val = float(imp.get(_RANDOM_COL, 0.0))
        agg = {d: 0.0 for d in self.driver_names}
        for feat, val in imp.items():
            d = self._feature_to_driver(feat)
            if d is not None:
                agg[d] += float(val)
        return pd.Series(agg), random_val

    def _relevance(self, value: float, baseline: float) -> str:
        """Ternary relevance of an importance ``value`` vs a noise ``baseline``.

        This is the normalization that makes heterogeneous methods comparable:
        every method ultimately reduces to yes / weak / no. Here, "above the
        noise floor" = yes, "at least half the floor" = weak, else no.
        """
        if baseline <= 0:
            baseline = 1e-12  # guard against a degenerate (zero) noise floor
        if value >= baseline:
            return 'yes'
        if value >= 0.5 * baseline:
            return 'weak'
        return 'no'

    # ------------------------------------------------------------- orchestrator
    def run(self, levels: tuple = ('static', 'temporal')) -> 'DriverAnalysis':
        """Run the requested layers and populate :attr:`results`.

        Args:
            levels: Any of ``'static'``, ``'temporal'``, ``'causal'``. The causal
                layer is opt-in (heavier deps, identification assumptions); only
                Granger runs automatically within it, while PCMCI/CATE remain
                explicit method calls.
        """
        levels = tuple(levels)
        rule("DriverAnalysis", verbose=self.verbose)
        # Always train the headline model first — Layer 1 reads directly from it.
        self.fit_model()

        if 'static' in levels:
            info("Layer 1 (association): SHAP + ALE", verbose=self.verbose)
            self.shap()
            # One ALE response curve per driver (interrogates the headline model).
            for d in self.driver_names:
                self._result.ale[d] = self.ale(d)
            # SHAP interaction values are expensive (O(n_features^2)); call
            # shap_interactions() / ale_2d() explicitly when you need them.

        if 'temporal' in levels:
            info("Layer 2 (temporal-prediction): lagged / scale-resolved / stratified",
                 verbose=self.verbose)
            if self.lags:  # lagged importance is only meaningful if lags were asked for
                self.lagged_importance()
            self.scale_resolved()
            self.stratified(by='season')

        if 'causal' in levels:
            info("Layer 3 (causation): Granger sanity check (deseasonalized)",
                 verbose=self.verbose)
            # Only the cheap Granger check runs automatically; pcmci()/cate()
            # stay explicit because of their heavy deps and assumptions.
            self.granger()

        # Collapse every method's output into the per-driver verdict table.
        self._synthesize()
        self._result.levels_run = list(levels)
        success("DriverAnalysis complete.", verbose=self.verbose)
        return self

    # --------------------------------------------------------- shared substrate
    def fit_model(self) -> 'DriverAnalysis':
        """Train the headline regressor once (time-aware holdout) and score it
        out-of-sample. The fitted model is reused by SHAP and ALE."""
        X, y = self._build_matrix(self.drivers_df, self.target, add_random=True)
        if len(X) < 20:
            raise ValueError(f"Too few complete rows ({len(X)}) to fit a model.")

        # Hold out the most recent slice for scoring. Because _build_matrix keeps
        # rows in chronological order, slicing the tail = "train on the past,
        # test on the future" — the leak-free split this whole module insists on.
        n_test = max(1, int(round(len(X) * self.test_size)))
        if self.time_aware_split:
            X_train, y_train = X.iloc[:-n_test], y.iloc[:-n_test]
            X_test, y_test = X.iloc[-n_test:], y.iloc[-n_test:]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state)

        model = self._new_model()
        model.fit(X_train, y_train)

        from diive.gapfilling.scores import prediction_scores
        scores = prediction_scores(predictions=np.asarray(model.predict(X_test)),
                                   targets=y_test.to_numpy())

        self.model_ = model
        self._X, self._y = X, y
        # The model was fitted with .RANDOM present, so ALE evaluates over the
        # full matrix; ALE only ever perturbs its target feature, .RANDOM rides
        # along untouched.
        self._X_attrib = self._X
        self._result.model = model
        self._result.model_scores = scores
        info(f"Held-out R2 = {scores['r2']:.3f} (RMSE = {scores['rmse']:.3g}), "
             f"n_train={len(X_train)}, n_test={len(X_test)}", verbose=self.verbose)
        return self

    # ------------------------------------ Layer 1: static attribution (assoc.)
    def shap(self) -> DataFrame:
        """SHAP importance per driver, with relevance vs the ``.RANDOM`` baseline."""
        if self.model_ is None:
            self.fit_model()
        imp, random_val = self._shap_per_driver(self.model_, self._X)
        # Cache the noise floor; the synthesis layer reuses it for every method.
        self._random_baseline = random_val
        # Rank highest-importance first, then label each driver against the floor.
        out = imp.sort_values(ascending=False).to_frame('shap_importance')
        out['shap_rank'] = range(1, len(out) + 1)
        out['shap_relevant'] = [self._relevance(v, random_val) for v in out['shap_importance']]
        out.index.name = 'driver'
        self._result.shap_importance = out
        detail(f"SHAP baseline (.RANDOM) = {random_val:.3g}", verbose=self.verbose)
        return out

    def shap_interactions(self) -> Optional[DataFrame]:
        """Mean |SHAP interaction| per driver pair (raw-driver models only)."""
        if self.feature_engineer is not None:
            warn("SHAP interactions skipped: not defined per-driver with a FeatureEngineer.",
                 verbose=self.verbose)
            return None
        import shap
        X = self._X
        # Interaction values are O(n_features^2 * trees) per row, so subsample.
        sample = X.sample(min(len(X), 150), random_state=self.random_state).sort_index()
        explainer = shap.TreeExplainer(self.model_)
        inter = explainer.shap_interaction_values(sample)
        mean_abs = np.abs(inter).mean(axis=0)  # (n_feat, n_feat)
        cols = list(X.columns)
        rows = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                di, dj = self._feature_to_driver(cols[i]), self._feature_to_driver(cols[j])
                if di is None or dj is None:
                    continue
                rows.append({'driver_1': di, 'driver_2': dj,
                             'interaction': float(mean_abs[i, j] * 2)})
        out = (DataFrame(rows).sort_values('interaction', ascending=False)
               .reset_index(drop=True)) if rows else None
        self._result.shap_interactions = out
        return out

    def ale(self, feature: str, grid_size: Optional[int] = None) -> AleCurve:
        """1D ALE response curve for a driver (correlation-robust)."""
        if self.model_ is None:
            self.fit_model()
        curve = accumulated_local_effects(
            self.model_, self._X_attrib, feature,
            grid_size=grid_size or self.ale_grid_size)
        self._result.ale[feature] = curve
        return curve

    def ale_2d(self, f1: str, f2: str, grid_size: int = 10) -> Ale2DResult:
        """2D (interaction) ALE surface for a driver pair (e.g. VPD x TA)."""
        if self.model_ is None:
            self.fit_model()
        res = accumulated_local_effects_2d(self.model_, self._X_attrib, f1, f2,
                                           grid_size=grid_size)
        if self._result.ale_2d is None:
            self._result.ale_2d = {}
        self._result.ale_2d[(f1, f2)] = res
        return res

    # ------------------------------ Layer 2: temporal attribution (time-aware)
    def lagged_importance(self) -> DataFrame:
        """Per-driver SHAP importance across engineered lags → response timescale.

        Builds lagged variants of every driver over the requested lag range, fits
        one model, and maps each lagged feature's importance back to (driver, lag).
        """
        if not self.lags:
            raise ValueError("No lags configured; pass lags=... to the constructor.")
        from diive.variables.temporal import lagged_variants

        # Build every lagged variant of every driver over [min..max] lags, fit one
        # model on the whole lot, then attribute each lagged column's importance
        # back to its (driver, lag) cell. The lag where a driver peaks reveals its
        # response timescale (instant vs hours vs days).
        lo, hi = min(self.lags), max(self.lags)
        lagged = lagged_variants(self.drivers_df.copy(), lag=[lo, hi], stepsize=1,
                                 verbose=0)
        X, y = self._build_matrix(lagged, self.target, add_random=True)
        model = self._new_model()
        model.fit(X, y)

        imp = _mean_abs_shap(model, X)

        # Rows = drivers, columns = lags (in records). Accumulate importance into
        # the matching cell; unmatched columns (e.g. .RANDOM) are ignored.
        lag_values = sorted(set(range(lo, hi + 1)))
        out = DataFrame(0.0, index=self.driver_names, columns=lag_values)
        for feat, val in imp.items():
            d, lag = self._parse_lagged_feature(feat)
            if d is not None and lag in out.columns:
                out.loc[d, lag] += float(val)
        out.index.name = 'driver'
        self._result.lagged_importance = out
        return out

    def _parse_lagged_feature(self, feature: str) -> tuple[Optional[str], Optional[int]]:
        """Split a lagged-variant column name into (driver, lag_in_records).

        ``lagged_variants`` names columns like ``.TA-2`` (TA two records back) and
        ``.SW_IN+1`` (one record ahead). We peel off the driver prefix and parse
        the signed integer suffix; ``int('-2')`` and ``int('+1')`` both work.
        """
        if feature == _RANDOM_COL:
            return None, None
        if feature in self.driver_names:
            return feature, 0  # original column = lag 0
        if not feature.startswith('.'):
            return self._feature_to_driver(feature), 0
        body = feature[1:]
        for d in sorted(self.driver_names, key=len, reverse=True):
            if body.startswith(d):
                suffix = body[len(d):]
                if suffix == '':
                    return d, 0
                try:
                    return d, int(suffix)  # handles '-2' and '+1'
                except ValueError:
                    return d, None
        return None, None

    def scale_resolved(self, scales: tuple = ('stl', 'daily', 'monthly')) -> DataFrame:
        """SHAP importance per STL component and/or temporal aggregation.

        A driver important at the daily/monthly scale but not half-hourly (or
        confined to one STL component) responds on a different timescale than its
        raw importance suggests.
        """
        cols = {}  # column name (scale) -> per-driver importance Series
        # (a) STL components: attribute drivers to the slow trend, the recurring
        # seasonal cycle, and the fast residual separately. A driver that only
        # matters for one component acts on that timescale.
        if 'stl' in scales:
            try:
                trend, seasonal, resid = _stl_components(self.target, self._stl_period)
                components = (('stl_trend', trend), ('stl_seasonal', seasonal),
                             ('stl_residual', resid))
            except Exception as e:
                warn(f"STL scale skipped: {e}", verbose=self.verbose)
                components = ()
            for comp_name, comp in components:
                comp = comp.reindex(self.target.index)
                comp.name = self.target.name
                imp, _ = self._fit_importance(self.drivers_df, comp)
                cols[comp_name] = imp

        # (b) Temporal aggregations: re-attribute at coarser resolutions. A driver
        # that's weak half-hourly but strong monthly responds slowly. Whenever any
        # aggregation is requested we also include the native ('halfhourly')
        # resolution as the baseline to compare the aggregations against.
        agg_map = {'halfhourly': None, 'daily': 'D', 'monthly': 'MS'}
        want_agg = ('daily' in scales) or ('monthly' in scales)
        for scale in ('halfhourly', 'daily', 'monthly'):
            requested = (scale in scales) or (scale == 'halfhourly' and want_agg)
            if not requested:
                continue
            freq = agg_map[scale]
            if freq is None:
                d_df, t = self.drivers_df, self.target  # native resolution
            else:
                d_df = self.drivers_df.resample(freq).mean()
                t = self.target.resample(freq).mean()
                t.name = self.target.name
            if len(t.dropna()) < 20:  # too few rows to fit a meaningful model
                detail(f"Scale '{scale}' has too few rows; skipped.", verbose=self.verbose)
                continue
            imp, _ = self._fit_importance(d_df, t)
            cols[scale] = imp

        out = DataFrame(cols)  # rows = drivers, columns = scales
        out.index.name = 'driver'
        self._result.scale_resolved = out
        return out

    def stratified(self, by: Union[str, Series] = 'season') -> DataFrame:
        """SHAP importance (and ALE direction) within regimes.

        Args:
            by: ``'season'`` | ``'daynight'`` | ``'wetdry'`` or a Series of regime
                labels aligned to the data. Relevance or direction that changes
                across regimes flags context-dependence / nonstationarity.
        """
        regimes = self._regime_labels(by)
        cols, directions = {}, {}
        # Fit a separate model within each regime (season, day/night, ...). If a
        # driver's relevance or ALE *direction* flips between regimes, its effect
        # is context-dependent — caught later as regime_dependence in the verdict.
        for label in pd.unique(regimes.dropna()):
            mask = (regimes == label)
            d_df = self.drivers_df.loc[mask]
            t = self.target.loc[mask]
            if len(t.dropna()) < 30:  # too small to trust a per-regime model
                detail(f"Regime '{label}' too small ({int(mask.sum())} rows); skipped.",
                       verbose=self.verbose)
                continue
            imp, model_X = self._fit_importance(d_df, t, return_model=True)
            cols[str(label)] = imp
            # Also record each driver's ALE shape within this regime, so we can
            # later detect a sign flip across regimes.
            model, X_attrib = model_X
            dir_map = {}
            for d in self.driver_names:
                try:
                    curve = accumulated_local_effects(model, X_attrib, d,
                                                      grid_size=min(10, self.ale_grid_size))
                    dir_map[d] = curve.direction(self.ale_range_threshold)
                except Exception:
                    dir_map[d] = 'flat'  # ALE failed (e.g. constant feature)
            directions[str(label)] = dir_map

        out = DataFrame(cols)
        out.index.name = 'driver'
        self._result.stratified = out
        self._stratified_directions = directions
        return out

    def _regime_labels(self, by: Union[str, Series]) -> Series:
        """Resolve a regime specification into a per-row label Series."""
        idx = self.target.index
        if isinstance(by, Series):
            return by.reindex(idx)
        if by == 'season':
            month_to_season = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM',
                               5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON',
                               10: 'SON', 11: 'SON'}
            return Series([month_to_season[m] for m in idx.month], index=idx)
        if by == 'daynight':
            rad = self._find_driver(['SW_IN', 'Rg', 'PPFD', 'SW', 'RAD'])
            if rad is not None:
                return Series(np.where(self.drivers_df[rad] > 20, 'day', 'night'), index=idx)
            return Series(np.where((idx.hour >= 6) & (idx.hour < 18), 'day', 'night'), index=idx)
        if by == 'wetdry':
            wet = self._find_driver(['SWC', 'PREC', 'SM', 'moist'])
            if wet is None:
                raise ValueError("'wetdry' needs a soil-moisture/precip driver; none found.")
            med = self.drivers_df[wet].median()
            return Series(np.where(self.drivers_df[wet] >= med, 'wet', 'dry'), index=idx)
        raise ValueError(f"Unknown stratification '{by}'.")

    def _find_driver(self, keys: list) -> Optional[str]:
        """First driver whose name contains any of ``keys`` (case-insensitive)."""
        for d in self.driver_names:
            dl = d.lower()
            if any(k.lower() in dl for k in keys):
                return d
        return None

    def rolling_importance(self, window: str = '30D') -> DataFrame:
        """Time-varying SHAP importance over non-overlapping ``window`` blocks.

        Not part of the default ``run`` (expensive); call explicitly. Each window
        gets its own model, so windows with too few rows are dropped.
        """
        rows = {}
        for start, block in self.drivers_df.groupby(pd.Grouper(freq=window)):
            t = self.target.loc[block.index]
            if len(t.dropna()) < 50:
                continue
            imp, _ = self._fit_importance(block, t)
            rows[start] = imp
        out = DataFrame(rows).T
        out.index.name = 'window_start'
        self._result.rolling_importance = out
        return out

    def _fit_importance(self, drivers_df: DataFrame, target: Series,
                        return_model: bool = False):
        """Fit a fresh model on (drivers, target) and return per-driver SHAP.

        The workhorse of Layer 2: the lagged, scale-resolved, stratified, and
        rolling analyses all reduce to "transform the data, then call this". Each
        gets its OWN model (the headline model is for Layer 1 only). Returns a
        per-driver importance Series, and optionally (model, X) so the caller can
        compute ALE on the same fitted model.

        Note: this fit is for attribution, not scoring — it uses all rows (no
        holdout), which is fine because we never report its accuracy."""
        X, y = self._build_matrix(drivers_df, target, add_random=True)
        model = self._new_model()
        model.fit(X, y)
        imp, _ = self._shap_per_driver(model, X)
        if return_model:
            return imp, (model, X)  # X keeps .RANDOM (model was fitted with it)
        return imp, None

    # ----------------------------------- Layer 3: causal (opt-in, gated deps)
    def granger(self) -> DataFrame:
        """Bivariate Granger test per driver — a cheap, caveated sanity check.

        Inputs are STL-deseasonalized internally first (shared seasonality is the
        classic spurious-Granger trap). This is NOT causal evidence on its own.
        """
        from diive.analysis.granger import GrangerCausality

        def _ds(s: Series) -> Series:
            trend, _seasonal, resid = _stl_components(s, self._stl_period)
            out = trend + resid
            out.name = s.name
            return out

        max_lag = min(10, abs(min(self.lags)) if self.lags else 5)
        t_ds = self.target if self.deseasonalize else _ds(self.target)
        rows = []
        for d in self.driver_names:
            x_ds = self.drivers_df[d] if self.deseasonalize else _ds(self.drivers_df[d])
            try:
                gc = GrangerCausality(x=x_ds, y=t_ds, max_lag=max_lag, verbose=False)
                pvals = gc.p_values()
                p_min = float(pvals['p_value'].min())
                lag_min = int(pvals.loc[pvals['p_value'].idxmin(), 'Lag'])
            except Exception as e:
                detail(f"Granger failed for {d}: {e}", verbose=self.verbose)
                p_min, lag_min = np.nan, np.nan
            rows.append({'driver': d, 'granger_p': p_min, 'granger_lag': lag_min})
        out = DataFrame(rows).set_index('driver')
        self._result.granger = out
        return out

    def pcmci(self, tau_max: int = 48, alpha: float = 0.05, **kw):
        """PCMCI(+) causal discovery via tigramite (optional extra ``diive[causal]``).

        Lagged, confounded, autocorrelation-aware discovery — the real upgrade
        over bivariate Granger. Lazy-imported; raises a helpful error if tigramite
        is not installed.
        """
        try:
            from tigramite import data_processing as pp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
        except ImportError as e:
            raise ImportError(
                "pcmci() requires tigramite. Install the optional extra:\n"
                "    pip install 'diive[causal]'   (or, with uv)   uv sync --extra causal"
            ) from e

        df = self.drivers_df.copy()
        df[self.target.name] = self.target
        df = df.dropna()
        var_names = list(df.columns)
        dataframe = pp.DataFrame(df.to_numpy(), var_names=var_names)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
        results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=alpha, **kw)
        self._result.pcmci = {'results': results, 'var_names': var_names,
                              'target': self.target.name, 'tau_max': tau_max}
        self._merge_pcmci_into_convergence()
        return self._result.pcmci

    def cate(self, treatment: Series, adjust: list, **kw):
        """Conditional Average Treatment Effect — only with an explicit treatment
        definition and identification assumptions (manipulation experiments:
        warming / drought / management). Never auto-run on continuous drivers.

        Optional extra ``diive[causal]`` (econml). Lazy-imported.
        """
        try:
            from econml.dml import CausalForestDML
        except ImportError as e:
            raise ImportError(
                "cate() requires econml. Install the optional extra:\n"
                "    pip install 'diive[causal]'   (or, with uv)   uv sync --extra causal"
            ) from e
        df = self.drivers_df[adjust].copy()
        df['_t_'] = treatment.reindex(df.index)
        df['_y_'] = self.target.reindex(df.index)
        df = df.dropna()
        est = CausalForestDML(random_state=self.random_state, **kw)
        est.fit(Y=df['_y_'].to_numpy(), T=df['_t_'].to_numpy(),
                X=df[adjust].to_numpy())
        effect = est.effect(df[adjust].to_numpy())
        self._result.cate = {'estimator': est, 'effect': effect,
                             'mean_effect': float(np.mean(effect)),
                             'adjust': adjust}
        return self._result.cate

    # ------------------------------------------------------------- synthesis
    def _bootstrap_stability(self) -> Optional[DataFrame]:
        """Fraction of bootstrap resamples in which each driver lands in top-k.

        SHAP rankings wobble a few percent between fits (noted in CLAUDE.md), so a
        rank means little if it isn't stable. We resample rows with replacement,
        refit, and count how often each driver stays in the top-k. A low fraction
        => "unstable_rank" flag and an 'inconclusive' verdict downstream.
        """
        if self.n_bootstrap <= 0:
            return None
        counts = {d: 0 for d in self.driver_names}
        rng = np.random.RandomState(self.random_state)
        n = len(self._X)
        for _ in range(self.n_bootstrap):
            sel = rng.randint(0, n, n)  # bootstrap sample of row positions
            Xb = self._X.iloc[sel]
            yb = self._y.iloc[sel]
            model = self._new_model()
            model.fit(Xb, yb)
            imp, _ = self._shap_per_driver(model, Xb)
            topk = set(imp.sort_values(ascending=False).head(self.top_k).index)
            for d in topk:
                counts[d] += 1
        out = DataFrame({'stability': {d: counts[d] / self.n_bootstrap
                                       for d in self.driver_names}})
        out.index.name = 'driver'
        self._result.stability = out
        return out

    def _synthesize(self):
        """Build the per-driver convergence/divergence table across all run methods.

        This is the headline. One row per driver, assembling whatever each layer
        produced: SHAP + ALE (association), lag/scale/regime fields (temporal),
        Granger/PCMCI/CATE (causal), plus bootstrap stability. The per-method
        relevances are then compared in _finalize_verdicts() to decide agreement
        and a final verdict. Methods that weren't run simply leave NaN — they
        don't count against a driver.
        """
        if self._result.shap_importance is None:
            self.shap()
        shap_df = self._result.shap_importance
        stability = self._bootstrap_stability()
        # Convert a dominant lag (in records) into wall-clock for timescale labels.
        freq_min = 86400.0 / self._stl_period / 60.0  # minutes per record (approx)

        rows = []
        for d in self.driver_names:
            row = {'driver': d}
            # --- association: SHAP ---
            row['shap_importance'] = float(shap_df.loc[d, 'shap_importance'])
            row['shap_rank'] = int(shap_df.loc[d, 'shap_rank'])
            row['shap_relevant'] = shap_df.loc[d, 'shap_relevant']
            # --- association: ALE ---
            curve = self._result.ale.get(d)
            if curve is not None:
                row['ale_range'] = curve.ale_range
                row['ale_direction'] = curve.direction(self.ale_range_threshold)
                row['ale_relevant'] = self._relevance_from_range(curve.ale_range)
            else:
                row['ale_range'] = np.nan
                row['ale_direction'] = None
                row['ale_relevant'] = None
            # --- temporal ---
            row.update(self._temporal_fields(d, freq_min))
            # --- causal (NaN if not run) ---
            row.update(self._causal_fields(d))
            # --- stability ---
            row['stability'] = (float(stability.loc[d, 'stability'])
                                if stability is not None else np.nan)
            rows.append(row)

        conv = DataFrame(rows).set_index('driver')
        self._finalize_verdicts(conv)
        self._result.convergence = conv

    def _relevance_from_range(self, ale_range: float) -> str:
        thr = self.ale_range_threshold
        if ale_range >= thr:
            return 'yes'
        if ale_range >= 0.5 * thr:
            return 'weak'
        return 'no'

    def _temporal_fields(self, d: str, freq_min: float) -> dict:
        """Summarise the Layer-2 results for one driver into convergence columns."""
        out = {'dominant_lag': np.nan, 'timescale': None,
               'scale_dependence': False, 'regime_dependence': False}
        # Dominant lag = the lag at which this driver's importance peaks.
        li = self._result.lagged_importance
        if li is not None and d in li.index and li.loc[d].abs().sum() > 0:
            dom = int(li.loc[d].astype(float).idxmax())
            out['dominant_lag'] = dom
            out['timescale'] = self._timescale(dom, freq_min)
        # Scale dependence = relevance differs across STL components / aggregations.
        sr = self._result.scale_resolved
        if sr is not None and d in sr.index:
            rels = [self._relevance(v, self._random_baseline or 0.0)
                    for v in sr.loc[d].dropna()]
            out['scale_dependence'] = len(set(rels)) > 1
        # Regime dependence = relevance OR ALE direction differs across regimes.
        st = self._result.stratified
        if st is not None and d in st.index:
            rels = [self._relevance(v, self._random_baseline or 0.0)
                    for v in st.loc[d].dropna()]
            dirs = ([v for v in self._stratified_directions.values()]
                    if self._stratified_directions else [])
            driver_dirs = {dd.get(d) for dd in dirs if dd.get(d) in ('+', '-')}
            out['regime_dependence'] = (len(set(rels)) > 1) or (len(driver_dirs) > 1)
        return out

    @staticmethod
    def _timescale(lag: int, freq_min: float) -> str:
        if lag == 0:
            return 'instant'
        hours = abs(lag) * freq_min / 60.0
        if hours <= 24:
            return 'hours'
        if hours <= 24 * 30:
            return 'days'
        return 'seasonal'

    def _causal_fields(self, d: str) -> dict:
        out = {'granger_p': np.nan, 'pcmci_link': np.nan, 'pcmci_lag': np.nan,
               'pcmci_sign': None, 'cate': np.nan}
        g = self._result.granger
        if g is not None and d in g.index:
            out['granger_p'] = float(g.loc[d, 'granger_p'])
        return out

    def _merge_pcmci_into_convergence(self):
        """Fold a post-hoc PCMCI run into an existing convergence table."""
        if self._result.convergence is None or self._result.pcmci is None:
            return
        conv = self._result.convergence
        info = self._result.pcmci
        var_names = info['var_names']
        graph = info['results'].get('graph')
        t_idx = var_names.index(info['target'])
        if graph is None:
            return
        # tigramite's graph is an array indexed [cause, effect, lag], where each
        # cell is a link-type string. We look for a directed link from driver d
        # into the target at ANY lag tau; the first one found gives pcmci_lag.
        for d in self.driver_names:
            if d not in var_names:
                continue
            j = var_names.index(d)
            link = False
            lag = np.nan
            for tau in range(graph.shape[2]):
                if graph[j, t_idx, tau] in ('-->', 'o->'):  # directed / partially-directed
                    link = True
                    lag = tau
                    break
            conv.loc[d, 'pcmci_link'] = link
            conv.loc[d, 'pcmci_lag'] = lag
        # Re-run synthesis so the new causal column changes the verdicts.
        self._finalize_verdicts(conv)

    def _finalize_verdicts(self, conv: DataFrame):
        """Compute agreement, verdict, flags, votes per driver (in place).

        For each driver this (1) collects each run method's relevance vote into a
        common {yes,weak,no} list plus any direction (+/-), (2) decides whether
        those votes converge / partially agree / diverge, (3) raises diagnostic
        flags for known conflict patterns, and (4) maps everything to a verdict.
        """
        # Pre-create object columns so per-cell list assignment (flags) doesn't
        # get broadcast across the row by pandas.
        for col in ['relevance_votes', 'agreement', 'verdict', 'flags']:
            conv[col] = pd.Series([None] * len(conv), index=conv.index, dtype=object)
        conv['n_methods_run'] = 0
        levels = []
        for d in conv.index:
            r = conv.loc[d]
            votes, flags = [], []
            relevances, directions = [], []  # normalized votes + signed directions

            # --- step 1: normalize each method that ran into a relevance vote ---
            # SHAP (unsigned: importance only, no direction)
            if pd.notna(r.get('shap_relevant')):
                votes.append(f"shap:{r['shap_relevant']}")
                relevances.append(r['shap_relevant'])
            # ALE (signed: contributes both relevance and a +/- direction)
            if r.get('ale_relevant') is not None:
                votes.append(f"ale:{r['ale_relevant']}")
                relevances.append(r['ale_relevant'])
                if r.get('ale_direction') in ('+', '-'):
                    directions.append(r['ale_direction'])
            # Granger: significant p-value => relevant
            if pd.notna(r.get('granger_p')):
                gr = 'yes' if r['granger_p'] < self.granger_alpha else 'no'
                votes.append(f"granger:{gr}")
                relevances.append(gr)
            # PCMCI: a causal link at any lag => relevant (and may carry a sign)
            if pd.notna(r.get('pcmci_link')):
                pr = 'yes' if bool(r['pcmci_link']) else 'no'
                votes.append(f"pcmci:{pr}")
                relevances.append(pr)
                if r.get('pcmci_sign') in ('+', '-'):
                    directions.append(r['pcmci_sign'])
            # CATE: only ever recorded when a treatment effect was estimated
            if pd.notna(r.get('cate')):
                votes.append("cate:yes")
                relevances.append('yes')

            # --- step 2: agreement over the methods that actually ran ---
            n_methods = len(relevances)
            yes = sum(v == 'yes' for v in relevances)
            no = sum(v == 'no' for v in relevances)
            weak = sum(v == 'weak' for v in relevances)
            dir_conflict = ('+' in directions) and ('-' in directions)  # opposite signs
            rel_conflict = yes > 0 and no > 0  # some say relevant, some say not

            if n_methods == 0:
                agreement = 'partial'
            elif rel_conflict or dir_conflict:
                agreement = 'diverge'        # the scientifically interesting case
            elif weak > 0 and yes == 0:
                agreement = 'partial'        # only lukewarm support
            elif (yes == n_methods) or (no == n_methods):
                agreement = 'converge'       # unanimous (all yes or all no)
            else:
                agreement = 'partial'

            # --- step 3: diagnostic flags for specific known conflict patterns ---
            # Important but flat response => correlation/interaction artifact.
            if r.get('shap_relevant') == 'yes' and r.get('ale_relevant') == 'no':
                flags.append('shap_high_ale_flat')
            # Bivariate temporal link that disappears under confounder control.
            if (pd.notna(r.get('granger_p')) and r['granger_p'] < self.granger_alpha
                    and pd.notna(r.get('pcmci_link')) and not bool(r['pcmci_link'])):
                flags.append('granger_sig_pcmci_null')
            # Direction reverses (across ALE/PCMCI signs, or across regimes).
            if dir_conflict:
                flags.append('sign_flip_by_regime')
            if bool(r.get('regime_dependence')):
                if 'sign_flip_by_regime' not in flags:
                    flags.append('sign_flip_by_regime') if self._regime_dir_conflict(d) else None
            # Ranking didn't survive bootstrapping.
            stab = r.get('stability')
            if pd.notna(stab) and stab < 0.5:
                flags.append('unstable_rank')

            # --- step 4: collapse all of the above into one verdict ---
            verdict = self._verdict(r, relevances, agreement, flags, stab)
            level = self._highest_level(r)  # deepest epistemic level reached
            levels.append(level)

            conv.loc[d, 'n_methods_run'] = n_methods
            conv.loc[d, 'relevance_votes'] = ' '.join(votes)
            conv.loc[d, 'agreement'] = agreement
            conv.loc[d, 'verdict'] = verdict
            conv.at[d, 'flags'] = flags
        conv['level'] = levels

    def _regime_dir_conflict(self, d: str) -> bool:
        if not self._stratified_directions:
            return False
        dirs = {dd.get(d) for dd in self._stratified_directions.values()
                if dd.get(d) in ('+', '-')}
        return len(dirs) > 1

    def _verdict(self, r, relevances, agreement, flags, stab) -> str:
        """Map a driver's evidence to a single verdict via a PRIORITY CASCADE.

        Order matters: the first matching rule wins, strongest disqualifiers
        first. The cascade reads top-to-bottom as "is it unreliable? -> not a
        driver at all? -> important-but-no-response (artifact)? -> only in some
        regimes? -> confirmed causal? -> association only?".
        """
        yes = sum(v == 'yes' for v in relevances)
        no = sum(v == 'no' for v in relevances)
        n = len(relevances)
        # Did a causal method run, and did it confirm a link?
        has_causal = pd.notna(r.get('pcmci_link')) or pd.notna(r.get('cate'))
        causal_yes = (bool(r.get('pcmci_link')) if pd.notna(r.get('pcmci_link')) else False) \
            or (pd.notna(r.get('cate')))
        shap_yes = r.get('shap_relevant') == 'yes'
        ale_shape = r.get('ale_direction') in ('+', '-', '∩', '∪')  # any real shape
        ale_flat = r.get('ale_relevant') == 'no'                    # no response

        # 1. Unstable ranking -> we can't trust anything else about it.
        if pd.notna(stab) and stab < 0.5:
            return 'inconclusive'
        # 2. Every method that ran said "no".
        if n > 0 and no == n:
            return 'not_a_driver'
        # 3. High importance but a flat ALE -> importance without response shape,
        #    the classic correlation/interaction artifact.
        if shap_yes and ale_flat:
            return 'spurious_correlate'
        # 4. Relevance/direction changes across regimes -> only a driver in context.
        if bool(r.get('regime_dependence')):
            return 'context_dependent'
        # 5. Important + real response + a causal method CONFIRMS it (and no
        #    contradiction) -> the strongest verdict we can give.
        if shap_yes and ale_shape and has_causal and causal_yes and agreement != 'diverge':
            return 'robust_driver'
        # 6. Important but a causal method ran and found NO link -> likely
        #    confounded; association is real but do not call it causal.
        if shap_yes and has_causal and not causal_yes:
            return 'associational_only'
        # 7. Important + real response, but no causal test was run.
        if shap_yes and ale_shape:
            return 'associational_only'
        # 8. Anything left (contradictory or weak with no clear pattern).
        if agreement == 'diverge':
            return 'inconclusive'
        return 'inconclusive'

    @staticmethod
    def _highest_level(r) -> str:
        """Deepest epistemic level for which this driver has a verdict.

        Reported in the summary so the reader knows how far the evidence reaches:
        causal (a Layer-3 method ran) > temporal (Layer-2 produced something) >
        association (only Layer-1 SHAP/ALE).
        """
        if pd.notna(r.get('pcmci_link')) or pd.notna(r.get('cate')) or pd.notna(r.get('granger_p')):
            return 'causal'
        if pd.notna(r.get('dominant_lag')) or bool(r.get('scale_dependence')) \
                or bool(r.get('regime_dependence')):
            return 'temporal'
        return 'association'

    # ----------------------------------------------------------------- outputs
    @property
    def results(self) -> DriverAnalysisResult:
        """The :class:`DriverAnalysisResult` populated by :meth:`run`."""
        return self._result

    def summary(self) -> None:
        """Rich table: per-driver verdict by epistemic level, divergence highlighted."""
        from rich.table import Table
        from diive.core.utils.console import console
        conv = self._result.convergence
        if conv is None:
            warn("Nothing to summarize; call run() first.", verbose=self.verbose)
            return

        rule("DriverAnalysis - convergence by epistemic level", verbose=self.verbose)
        console.print(f"  Levels run: {self._result.levels_run}   "
                      f"Held-out R2: {self._result.model_scores.get('r2', float('nan')):.3f}")

        table = Table(show_header=True, header_style="bold")
        for col in ['driver', 'level', 'SHAP', 'ALE dir', 'timescale',
                    'votes', 'agreement', 'verdict', 'flags']:
            table.add_column(col)

        order = {'robust_driver': 0, 'context_dependent': 1, 'associational_only': 2,
                 'spurious_correlate': 3, 'inconclusive': 4, 'not_a_driver': 5}
        conv_sorted = conv.sort_values(
            by=['verdict', 'shap_rank'],
            key=lambda s: s.map(order) if s.name == 'verdict' else s)

        for d, r in conv_sorted.iterrows():
            diverge = r['agreement'] == 'diverge'
            style = "bold yellow" if diverge else None
            ale_dir = r.get('ale_direction') or '-'
            flags = ', '.join(r['flags']) if isinstance(r['flags'], list) else ''
            row = [str(d), str(r.get('level', '')), str(r.get('shap_relevant', '')),
                   str(ale_dir), str(r.get('timescale') or '-'),
                   str(r.get('relevance_votes', '')), str(r.get('agreement', '')),
                   str(r.get('verdict', '')), flags]
            table.add_row(*row, style=style)
        console.print(table)
        console.print("  [dim]Association (SHAP/ALE) is never causal. "
                      "Diverging rows (yellow) are where the science is.[/dim]")

    def plot_importance(self, ax=None, title: str = None, showplot: bool = False):
        """Horizontal SHAP importance bars, colored by relevance vs ``.RANDOM``."""
        import matplotlib.pyplot as plt
        shap_df = self._result.shap_importance
        if shap_df is None:
            self.shap()
            shap_df = self._result.shap_importance
        created = False
        if ax is None:
            from diive.core.plotting import plotfuncs as pf
            fig, ax = pf.create_ax(figsize=(9, max(2.0, 0.5 * len(shap_df))))
            created = True
        df = shap_df.sort_values('shap_importance')
        colors = [_RELEVANCE_COLOR[r] for r in df['shap_relevant']]
        ax.barh(df.index, df['shap_importance'], color=colors, zorder=3)
        if self._random_baseline is not None:
            ax.axvline(self._random_baseline, color=_MD['grey'], ls='--',
                       label='.RANDOM baseline', zorder=4)
            ax.legend(loc='lower right', fontsize=10)
        ax.set_xlabel('mean |SHAP| (association level — not causal)')
        ax.set_title(title or 'SHAP importance')
        if created:
            ax.grid(True, axis='x', ls='--', alpha=0.4, zorder=0)
            if showplot:
                fig.show()
        return ax

    def plot_ale(self, feature: str, ax=None, **kw):
        """Plot one driver's 1D ALE curve (delegates to :class:`AleCurve`)."""
        curve = self._result.ale.get(feature) or self.ale(feature)
        return curve.plot(ax=ax, **kw)

    def plot_lagged(self, ax=None, cmap: str = 'viridis', title: str = None,
                    showplot: bool = False):
        """Driver x lag importance heatmap → per-driver response-timescale fingerprint."""
        li = self._result.lagged_importance
        if li is None:
            raise ValueError("No lagged importance; pass lags=... and run the temporal layer.")
        created = False
        if ax is None:
            from diive.core.plotting import plotfuncs as pf
            fig, ax = pf.create_ax(figsize=(11, max(2.5, 0.5 * len(li))))
            created = True
        # 'nearest' shading: X/Y are cell centers matching C's columns/rows.
        mesh = ax.pcolormesh(li.columns.astype(float), np.arange(len(li.index)),
                             li.values, cmap=cmap, shading='nearest')
        ax.figure.colorbar(mesh, ax=ax, label='SHAP importance')
        ax.set_yticks(range(len(li.index)))
        ax.set_yticklabels(li.index)
        ax.set_xlabel('lag (records; negative = past)')
        ax.set_title(title or 'Lagged importance (response timescale)')
        if created and showplot:
            fig.show()
        return ax

    def plot_scale_resolved(self, ax=None, cmap: str = 'magma', title: str = None,
                            showplot: bool = False):
        """Driver x scale importance heatmap (STL components / aggregations)."""
        sr = self._result.scale_resolved
        if sr is None:
            raise ValueError("No scale-resolved result; run the temporal layer.")
        created = False
        if ax is None:
            from diive.core.plotting import plotfuncs as pf
            fig, ax = pf.create_ax(figsize=(10, max(2.5, 0.5 * len(sr))))
            created = True
        mesh = ax.pcolormesh(np.arange(len(sr.columns) + 1), np.arange(len(sr.index) + 1),
                             sr.values, cmap=cmap, shading='flat')
        ax.figure.colorbar(mesh, ax=ax, label='SHAP importance')
        ax.set_xticks(np.arange(len(sr.columns)) + 0.5)
        ax.set_xticklabels(sr.columns, rotation=30, ha='right')
        ax.set_yticks(np.arange(len(sr.index)) + 0.5)
        ax.set_yticklabels(sr.index)
        ax.set_title(title or 'Scale-resolved importance')
        if created and showplot:
            fig.show()
        return ax

    def plot_convergence(self, ax=None, title: str = None, showplot: bool = False):
        """Headline plot: per-driver x per-method relevance grid.

        Green = relevant, grey = weak, red = not relevant; direction glyphs (+/-)
        annotate signed methods. Divergence across a row is the scientific signal.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        conv = self._result.convergence
        if conv is None:
            raise ValueError("Nothing to plot; call run() first.")

        method_cols = self._convergence_grid(conv)
        drivers = list(conv.index)
        methods = list(method_cols.columns)
        # Map relevance to integers: no=0, weak=1, yes=2, absent=NaN.
        code = {'no': 0, 'weak': 1, 'yes': 2}
        grid = np.full((len(drivers), len(methods)), np.nan)
        for i, d in enumerate(drivers):
            for j, m in enumerate(methods):
                v = method_cols.loc[d, m]
                if isinstance(v, str) and v in code:
                    grid[i, j] = code[v]

        created = False
        if ax is None:
            from diive.core.plotting import plotfuncs as pf
            fig, ax = pf.create_ax(figsize=(1.6 * len(methods) + 3, 0.6 * len(drivers) + 2))
            created = True
        cmap = ListedColormap([_MD['red'], _MD['grey'], _MD['green']])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_yticks(range(len(drivers)))
        ax.set_yticklabels(drivers)
        # Direction glyphs from ALE.
        for i, d in enumerate(drivers):
            adir = conv.loc[d, 'ale_direction'] if 'ale_direction' in conv.columns else None
            if 'ale' in methods and adir in ('+', '-', '∩', '∪'):
                j = methods.index('ale')
                ax.text(j, i, adir, ha='center', va='center', color='white', fontsize=12)
        ax.set_title(title or 'Convergence grid (green=yes, grey=weak, red=no)')
        if created and showplot:
            fig.show()
        return ax

    def _convergence_grid(self, conv: DataFrame) -> DataFrame:
        """Per-driver relevance for each method present (for the convergence plot)."""
        cols = {}
        cols['shap'] = conv['shap_relevant']
        if 'ale_relevant' in conv.columns and conv['ale_relevant'].notna().any():
            cols['ale'] = conv['ale_relevant']
        if 'granger_p' in conv.columns and conv['granger_p'].notna().any():
            cols['granger'] = conv['granger_p'].apply(
                lambda p: 'yes' if pd.notna(p) and p < self.granger_alpha else 'no')
        if 'pcmci_link' in conv.columns and conv['pcmci_link'].notna().any():
            cols['pcmci'] = conv['pcmci_link'].apply(
                lambda x: 'yes' if x is True else ('no' if x is False else None))
        return DataFrame(cols)
