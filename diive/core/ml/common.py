"""
kudos: https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from yellowbrick.regressor import PredictionError, ResidualsPlot

from rich.table import Table

import diive.core.dfun.frames as fr
from diive.core.ml.results import GapFillingResult
from diive.core.times.times import vectorize_timestamps
from diive.core.utils.console import (
    VERBOSE_PROGRESS,
    console as _console,
    detail,
    info,
    rule,
    success,
    warn,
)
from diive.gapfilling.scores import prediction_scores

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


class MlRegressorGapFillingBase:
    """Base class for ML-regressor gap-fillers (Random Forest, XGBoost): training, scoring and gap-filling."""

    def __init__(self,
                 regressor,
                 input_df: DataFrame,
                 target_col: str or tuple,
                 verbose: int = 0,
                 test_size: float = 0.25,
                 below_zero: str = None,
                 **kwargs):
        """
        Base class for machine-learning gap-filling using Random Forest or XGBoost.

        Trains a predictive model on complete (non-gap) data to fill missing values in a target
        time series. Accepts PRE-ENGINEERED features to enable reuse across multiple models.

        **Important:** Feature engineering is now performed EXTERNALLY using FeatureEngineer class.
        See diive.core.ml.feature_engineer.FeatureEngineer for details.

        Workflow:
            1. Use FeatureEngineer to create features from raw time series
            2. Pass engineered features to this class for model training and gap-filling
            3. Same engineered features can be used with multiple models (RF, XGB, etc.)

        Args:
            regressor:
                Sklearn-compatible regressor class (RandomForestRegressor or XGBRegressor).

            input_df:
                Input DataFrame with time series data. Must contain:
                - 1 target column (to be gap-filled)
                - 1+ feature columns (ALREADY ENGINEERED by FeatureEngineer)
                Timestamps should be in DataFrame index (DatetimeIndex).

            target_col:
                Column name of the variable to gap-fill (string or tuple for multi-level columns).

            verbose:
                Verbosity level: 0=silent, 1=progress updates, 2+=detailed output.

            test_size:
                Proportion of complete (non-gap) data to reserve for testing, between 0.0-1.0.
                Default: 0.25 (75% train, 25% test on non-gap data only).
                - Smaller test_size: tighter train set but more noise in test metrics
                - Larger test_size: looser train set but better estimate stability
                Standard value 0.25 balances both considerations.
                The test set is sampled RANDOMLY across the record (not a temporal
                block) — the appropriate validation for gap-filling (interpolating
                short scattered gaps), not extrapolation. See the train/test split
                in this constructor for the full rationale.

            below_zero:
                How to treat predicted values below zero for variables that cannot be negative
                (e.g. VPD, SW_IN, PPFD). Applied to gap-filled predictions only, not to
                observed data.
                - None (default): no treatment, keep predictions as-is.
                - 'zero': clip negative predictions to 0.

            **kwargs:
                Regressor-specific hyperparameters passed to the sklearn regressor.
                For RandomForestRegressor: n_estimators, max_depth, min_samples_split, etc.
                For XGBRegressor: n_estimators, max_depth, learning_rate, early_stopping_rounds, etc.

        Attributes:
            model_: Trained regressor instance.
            gapfilling_df_: DataFrame with gap-filled target and auxiliary variables.
            feature_importances_: SHAP feature importance from gap-filling model.
            feature_importances_traintest_: SHAP feature importance from train/test model.
            scores_: Model performance metrics (MAE, RMSE, R²) for gap-filling.
            scores_traintest_: Held-out performance metrics from the train/test
                split (the honest generalization estimate). The split is random
                (scattered in time), not temporal — the right validation for
                gap-filling; see the split in __init__.

        For Feature Engineering Parameters:
            See diive.core.ml.feature_engineer.FeatureEngineer for comprehensive documentation
            of the 8-stage feature engineering pipeline (lag, rolling, diff, EMA, poly, STL,
            timestamps, record number).

        See Also:
            diive.gapfilling.randomforest_ts.RandomForestTS — Random Forest subclass
            diive.gapfilling.xgboost_ts.XGBoostTS — XGBoost subclass
            examples/gapfilling/gapfill_randomforest.py — Random Forest gap-filling with feature engineering
            examples/gapfilling/gapfill_xgboost.py — XGBoost gap-filling with feature engineering
            examples/gapfilling/gapfill_comparison.py — Multi-method comparison
        """

        # Store arguments
        self.regressor = regressor
        input_df = input_df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.verbose = verbose
        self.kwargs = kwargs

        _valid_below_zero = (None, 'zero')
        if below_zero not in _valid_below_zero:
            raise ValueError(f"below_zero must be one of {_valid_below_zero}, got '{below_zero}'")
        self.below_zero = below_zero

        self._random_state = self.kwargs['random_state'] if 'random_state' in self.kwargs else None

        if self.regressor == RandomForestRegressor:
            self.gfsuffix = '_gfRF'
        elif self.regressor == XGBRegressor:
            self.gfsuffix = '_gfXG'
        else:
            self.gfsuffix = '_gf'

        # Short method tag (RF / XGB) shown on every phase banner in the report.
        self._method_tag = {'_gfRF': 'RF', '_gfXG': 'XGB'}.get(self.gfsuffix, 'ML')

        if self.verbose >= VERBOSE_PROGRESS:
            _console.print("")
            rule(f"[bold]GAP-FILLING START — {self.target_col} · {self._method_tag} "
                 f"({self.regressor.__name__})[/bold]", verbose=self.verbose)

        # Model dataframe is the pre-engineered input (features are already computed)
        self.model_df = input_df.copy()

        if target_col not in self.model_df.columns:
            available = self.model_df.columns.tolist()
            raise KeyError(
                f"target_col '{target_col}' not found in input_df. "
                f"Available columns: {available}"
            )

        # Original input features (all features except target)
        self.original_input_features = self.model_df.drop(columns=self.target_col).columns.tolist()

        self._check_n_cols()

        # Check if features complete
        n_vals_index = len(self.model_df.index)
        fstats = self.model_df[self.original_input_features].describe()
        not_complete = fstats.loc['count'] < n_vals_index
        not_complete = not_complete[not_complete].index.tolist()
        if len(not_complete) > 0:
            warn(f"Some features are incomplete (<{n_vals_index} values): "
                 + ", ".join(f"{nc} ({fstats[nc]['count'].astype(int)})" for nc in not_complete))
            warn("Not all target values can be predicted based on the full model.")

        # Train/test split for the held-out generalization estimate.
        #
        # The split is RANDOM (shuffle=True), drawn from the COMPLETE rows only
        # (target observed AND all features present — `.dropna()`); gaps and rows
        # with any missing predictor are in neither set. The test set is therefore
        # observed target values at timestamps scattered across the whole record.
        #
        # A random (not temporal/block) split is the correct validation for
        # GAP-FILLING: real gaps are typically short and scattered among observed
        # data, so a random hold-out reproduces the actual task — interpolating an
        # isolated timestamp from mostly-present neighbours. A temporal/block split
        # would instead measure EXTRAPOLATION skill (the right question for
        # forecasting or driver attribution), which understates gap-filling
        # performance. Keep this random split unless the goal changes.
        _temp_df = self.model_df.copy().dropna()

        self.train_df, self.test_df = train_test_split(_temp_df, test_size=self.test_size,
                                                       random_state=self._random_state, shuffle=True)

        self.train_df = self.train_df.sort_index()
        self.test_df = self.test_df.sort_index()

        self.random_col = None

        # Instantiate model with params
        self._model = self.regressor(**self.kwargs)

        # Attributes
        self._gapfilling_df = None  # Will contain gapfilled target and auxiliary variables
        # self._model = None
        self._traintest_details = dict()
        self._feature_importances = dict()
        self._feature_importances_traintest = pd.DataFrame()
        self._feature_importances_reduction = pd.DataFrame()
        self._scores = dict()
        self._scores_traintest = dict()
        self._accepted_features = []
        self._rejected_features = []

        self.n_timeseriessplits = None

    def get_gapfilled_target(self):
        """Gap-filled target time series"""
        return self.gapfilling_df_[self.target_gapfilled_col]

    def get_flag(self):
        """Gap-filling flag, where 0=observed, 1=gap-filled, 2=gap-filled with fallback"""
        return self.gapfilling_df_[self.target_gapfilled_flag_col]

    @property
    def model_(self):
        """Return the model (trained on the training set)."""
        if not self._model:
            raise Exception(f'Not available: model.')
        return self._model

    @property
    def feature_importances_(self) -> DataFrame:
        """Return feature importance for model used in gap-filling"""
        if not isinstance(self._feature_importances, DataFrame):
            raise Exception(f'Not available: feature importances for gap-filling.')
        return self._feature_importances

    @property
    def feature_importances_traintest_(self) -> DataFrame:
        """Return feature importance from model training on training data,
        with importances calculated using test data (holdout set)"""
        if not isinstance(self._feature_importances_traintest, DataFrame):
            raise Exception(f'Not available: feature importances from training & testing.')
        return self._feature_importances_traintest

    @property
    def feature_importances_reduction_(self) -> pd.DataFrame:
        """Return feature importance from feature reduction, model training on training data,
        with importances calculated using test data (holdout set)"""
        if not isinstance(self._feature_importances_reduction, pd.DataFrame):
            raise Exception(f'Not available: feature importances from feature reduction.')
        return self._feature_importances_reduction

    @property
    def scores_(self) -> dict:
        """In-sample scores of the gap-filling model: the final model predicting on
        ALL complete rows, including the rows it was trained on, so these are
        optimistically biased. For an honest generalization estimate use
        scores_traintest_ (computed on the held-out test set)."""
        if not self._scores:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores

    @property
    def scores_traintest_(self) -> dict:
        """Held-out generalization scores: model trained on the training split,
        scored on the unseen test split.

        The test split is a RANDOM subset (``test_size``, default 25%) of the
        complete rows, scattered across the record — not a temporal block. This
        is the correct validation for gap-filling, which interpolates short
        scattered gaps from nearby observed data; a temporal/block split would
        measure extrapolation instead and understate gap-filling skill. See the
        train/test split in ``__init__`` for the full rationale."""
        if not self._scores_traintest:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores_traintest

    @property
    def gapfilling_df_(self) -> DataFrame:
        """Return gapfilled data and auxiliary variables"""
        if not isinstance(self._gapfilling_df, DataFrame):
            raise Exception(f'Gapfilled data not available.')
        return self._gapfilling_df

    @property
    def traintest_details_(self) -> dict:
        """Return details from train/test splits"""
        if not self._traintest_details:
            raise Exception(f'Not available: details about training & testing.')
        return self._traintest_details

    @property
    def accepted_features_(self) -> list:
        """Return list of accepted features from feature reduction"""
        if not self._accepted_features:
            raise Exception(f'Not available: accepted features from feature reduction.')
        return self._accepted_features

    @property
    def rejected_features_(self) -> list:
        """Return list of rejected features from feature reduction"""
        if self._rejected_features is None:
            raise Exception(f'Not available: rejected features from feature reduction.')
        return self._rejected_features

    @staticmethod
    def _fi_across_splits(feature_importances_splits) -> DataFrame:
        """Calculate overall feature importance as mean across splits."""
        fi_columns = [c for c in feature_importances_splits.columns if str(c).endswith("_IMPORTANCE")]
        fi_df = feature_importances_splits[fi_columns].copy()
        fi_df = fi_df.mean(axis=1)
        return fi_df

    def _remove_rejected_features(self, shap_threshold_factor: float = 1.0, infotxt="[ FEATURE REDUCTION ]") -> list:
        """Remove features that are below importance threshold or that are below
        zero from model dataframe. The updated model dataframe will then be used
        for the next (final) model.
        """

        fi_df = self.feature_importances_reduction_.copy()
        series = fi_df['SHAP_IMPORTANCE'].copy()

        # Threshold for feature reduction: random_importance + k * random_sd
        random_importance = series.loc[self.random_col]
        random_sd = fi_df.loc[self.random_col, 'SHAP_SD'] if 'SHAP_SD' in fi_df.columns else 0
        threshold = random_importance + shap_threshold_factor * random_sd
        info(f"Random benchmark: importance={random_importance:.6f}, SD={random_sd:.6f}  "
             f"-> keep threshold = random + {shap_threshold_factor}*SD = [bold]{threshold:.6f}[/bold]",
             verbose=self.verbose)

        # Get accepted features
        accepted_locs = ((series > threshold) & (series > 0))
        accepted_features = series[accepted_locs].index.tolist()

        # Get rejected features (everything not accepted, incl. boundary and random col)
        rejected_locs = ~accepted_locs
        rejected_features = series[rejected_locs].index.tolist()

        # One clean table: every feature, its SHAP importance, and the verdict
        # (benchmark / accepted / rejected) against the threshold.
        self._log_reduction_table(series, threshold, accepted_features)
        info(f"[green]Accepted {len(accepted_features)}[/green] / "
             f"[yellow]rejected {len(rejected_features) - 1}[/yellow] feature(s) "
             f"(excluding the random benchmark).", verbose=self.verbose)

        # Update dataframe, keep accepted columns
        accepted_cols = [self.target_col]
        accepted_cols = accepted_cols + accepted_features

        return accepted_cols

    def _log_reduction_table(self, series, threshold: float,
                             accepted: list) -> None:
        """Print the feature-reduction decision as one coloured table:
        every feature with its SHAP importance and verdict, strongest first."""
        if self.verbose < VERBOSE_PROGRESS:
            return
        ranked = series.sort_values(ascending=False)
        table = Table(title="Feature reduction (SHAP importance vs random benchmark)",
                      title_style="bold blue", header_style="bold",
                      expand=False, border_style="grey50")
        table.add_column("feature", style="cyan", no_wrap=True)
        table.add_column("mean |SHAP|", justify="right")
        table.add_column("verdict")
        for name, val in ranked.items():
            if name == self.random_col:
                verdict = "[blue]random benchmark[/blue]"
            elif name in accepted:
                verdict = "[green]accepted[/green]"
            else:
                verdict = "[yellow]rejected[/yellow]"
            table.add_row(str(name), f"{val:.6f}", verdict)
        _console.print(table)

    def _fitmodel(self, model, X_train, y_train, X_test, y_test):
        """Fit model.

        For XGBoost, ``early_stopping_rounds`` monitors the LAST ``eval_set``
        entry, so that entry must be a genuine hold-out. The feature-reduction
        and fallback paths call this with the training data itself as the eval
        set (``X_test is X_train``); left as-is, early stopping would only ever
        see training loss — which keeps falling as trees are added — so it would
        never trigger and the model would overfit. In that case, carve out a
        small validation split so early stopping watches unseen data. When early
        stopping is not configured, behaviour is unchanged.
        """
        if isinstance(model, RandomForestRegressor):
            model.fit(X=X_train, y=y_train)
        elif isinstance(model, XGBRegressor):
            if getattr(model, 'early_stopping_rounds', None) and X_test is X_train:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.1,
                    random_state=self._random_state, shuffle=True)
                model.fit(X=X_tr, y=y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)])
            else:
                model.fit(X=X_train, y=y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
        return model

    def run(self, **kwargs):
        """Unified trigger: trains model then fills gaps."""
        self.trainmodel(**kwargs)
        self.fillgaps(**kwargs)
        self._section(f"GAP-FILLING COMPLETE — {self.target_col}")
        if self.verbose >= VERBOSE_PROGRESS:
            _console.print("")
        return self

    @property
    def result(self) -> DataFrame:
        """Primary result: full gap-filling DataFrame (target + flag columns)."""
        return self.gapfilling_df_

    @property
    def results(self) -> GapFillingResult:
        """Structured result after .run() — all outputs in one object.

        Returns a :class:`~diive.core.ml.results.GapFillingResult` with:
        ``gapfilled``, ``flag``, ``scores``, ``scores_traintest``,
        ``feature_importances``, ``feature_importances_traintest``,
        ``gapfilling_df``, ``model``, ``accepted_features``, ``rejected_features``.

        Raises:
            Exception: if called before :meth:`run`.
        """
        if not isinstance(self._gapfilling_df, DataFrame):
            raise Exception("Results not available: call .run() first.")
        fi = self._feature_importances if isinstance(self._feature_importances, DataFrame) else None
        fi_tt = self._feature_importances_traintest if isinstance(self._feature_importances_traintest, DataFrame) else None
        return GapFillingResult(
            gapfilled=self.get_gapfilled_target(),
            flag=self.get_flag(),
            scores=self._scores,
            scores_traintest=self._scores_traintest or None,
            feature_importances=fi,
            feature_importances_traintest=fi_tt,
            gapfilling_df=self._gapfilling_df,
            model=self._model,
            accepted_features=self._accepted_features or None,
            rejected_features=self._rejected_features or None,
        )

    def plot_feature_importances(self,
                                 ax=None,
                                 traintest: bool = False,
                                 max_features: int = None,
                                 color: str = '#003A70',
                                 error_color: str = '#C41E3A',
                                 show_values: bool = True,
                                 title: str = "SHAP feature importance"):
        """Plot SHAP feature importances as a horizontal bar chart (two-phase).

        Importances are mean |SHAP value| per feature with SD error bars, sorted
        ascending so the strongest feature sits at the top.

        Two-phase rendering: with ``ax=None`` a new standalone figure is created
        and shown; passing an existing ``ax`` draws into it and returns it without
        showing — for embedding in a GUI canvas or a multi-panel figure.

        Args:
            ax: Target matplotlib Axes. None creates (and shows) a new figure.
            traintest: Plot the train/test (held-out) importances instead of the
                final gap-filling model's importances.
            max_features: Keep only the top-N features by importance (None = all).
            color: Bar fill color.
            error_color: SD error-bar color.
            show_values: Annotate each bar with its importance value.
            title: Axes title.

        Returns:
            The matplotlib Axes drawn into.
        """
        fi = self.feature_importances_traintest_ if traintest else self.feature_importances_
        fidf = fi.copy().sort_values(by='SHAP_IMPORTANCE', ascending=True)
        if max_features is not None and max_features > 0:
            fidf = fidf.tail(max_features)  # tail() keeps the largest (sorted ascending)
        importances = fidf['SHAP_IMPORTANCE']
        errors = fidf['SHAP_SD'] if 'SHAP_SD' in fidf.columns else None
        labels = fidf.index

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(10, max(4, len(fidf) * 0.35)), dpi=100)

        y = range(len(importances))
        ax.barh(y, importances, color=color, alpha=0.85,
                edgecolor='#2C3E50', linewidth=0.6)
        if errors is not None:
            ax.errorbar(importances, y, xerr=errors, fmt='none', ecolor=error_color,
                        elinewidth=1.5, capsize=3, capthick=1.5, alpha=0.8, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('mean |SHAP value|', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.4, color='#BDC3C7', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if show_values:
            for i, imp in enumerate(importances):
                ax.text(imp, i, f' {imp:.3f}', va='center', fontsize=7, color='#2C3E50')

        if standalone:
            ax.figure.tight_layout()
            ax.figure.show()
        return ax

    # --- rich console logging (shared by all ML gap-fillers) ----------------
    # These print nicely-formatted, coloured progress to the diive console at
    # VERBOSE_PROGRESS+, so the GUI Log tab (which mirrors the console) shows a
    # legible training/gap-filling report for every ML method. Default verbose=0
    # keeps scripts silent unless they opt in.

    def _section(self, title: str) -> None:
        """A blank line + a titled rule, so phases are clearly separated. The
        method tag (RF / XGB) is appended so each banner names the gap-filler."""
        if self.verbose < VERBOSE_PROGRESS:
            return
        _console.print("")
        rule(f"{title} · {self._method_tag}", verbose=self.verbose)

    def _log_config(self) -> None:
        """Print the run configuration (regressor + hyperparameters) for
        reproducibility — every value needed to repeat the run."""
        if self.verbose < VERBOSE_PROGRESS:
            return
        table = Table(title="Configuration", title_style="bold blue",
                      header_style="bold", expand=False, border_style="grey50")
        table.add_column("setting", style="cyan", no_wrap=True)
        table.add_column("value", justify="right")
        table.add_row("regressor", self.regressor.__name__)
        table.add_row("target", str(self.target_col))
        table.add_row("test_size", f"{self.test_size}")
        table.add_row("below_zero", str(self.below_zero))
        for key, val in self.kwargs.items():  # explicit hyperparameters only
            table.add_row(key, str(val))
        _console.print(table)

    def _log_scores_table(self, scores: dict, title: str) -> None:
        """Print a coloured table of model scores."""
        if self.verbose < VERBOSE_PROGRESS or not scores:
            return
        table = Table(title=title, title_style="bold blue", header_style="bold",
                      expand=False, border_style="grey50")
        table.add_column("metric", style="cyan", no_wrap=True)
        table.add_column("value", justify="right")
        table.add_column("description", style="dim")
        rows = [
            ("R2", scores.get("r2"), "coeff. of determination (1 = perfect)"),
            ("RMSE", scores.get("rmse"), "root mean squared error"),
            ("MAE", scores.get("mae"), "mean absolute error"),
            ("MedAE", scores.get("medae"), "median absolute error"),
            ("MAPE", scores.get("mape"), "mean absolute % error"),
            ("MAXE", scores.get("maxe"), "max error"),
            ("MSE", scores.get("mse"), "mean squared error"),
        ]
        for name, val, desc in rows:
            valstr = f"{val:.4g}" if isinstance(val, (int, float)) else "—"
            table.add_row(name, valstr, desc)
        _console.print(table)

    def _log_importances_table(self, fi, title: str, top_n: int = 15) -> None:
        """Print a coloured table of the strongest SHAP feature importances."""
        if self.verbose < VERBOSE_PROGRESS or not isinstance(fi, pd.DataFrame) \
                or "SHAP_IMPORTANCE" not in fi.columns:
            return
        ser = fi["SHAP_IMPORTANCE"].sort_values(ascending=False)
        ser = ser[[i for i in ser.index if i != self.random_col]]  # drop benchmark
        has_sd = "SHAP_SD" in fi.columns
        table = Table(title=title, title_style="bold blue", header_style="bold",
                      expand=False, border_style="grey50")
        table.add_column("#", justify="right", style="dim")
        table.add_column("feature", style="cyan", no_wrap=True)
        table.add_column("mean |SHAP|", justify="right")
        if has_sd:
            table.add_column("SD", justify="right", style="dim")
        for i, (name, val) in enumerate(ser.head(top_n).items(), start=1):
            cells = [str(i), str(name), f"{val:.4f}"]
            if has_sd:
                cells.append(f"{fi.loc[name, 'SHAP_SD']:.4f}")
            table.add_row(*cells)
        _console.print(table)
        if len(ser) > top_n:
            info(f"… and {len(ser) - top_n} more feature(s)", verbose=self.verbose)

    def _log_gapfill_summary(self) -> None:
        """Print a coloured summary of the gap-filling outcome (flag breakdown,
        coverage, in-sample scores, final feature importances)."""
        if self.verbose < VERBOSE_PROGRESS:
            return
        df = self._gapfilling_df
        flag = df[self.target_gapfilled_flag_col]
        n_obs = int((flag == 0).sum())
        n_hq = int((flag == 1).sum())
        n_fb = int((flag == 2).sum())
        n_total = len(df)
        n_avail = int(df[self.target_gapfilled_col].notna().sum())
        n_filled = n_hq + n_fb
        info(f"Observed: {n_obs:,}  |  gaps filled: {n_filled:,} "
             f"([green]{n_hq:,}[/green] full model, [yellow]{n_fb:,}[/yellow] fallback)",
             verbose=self.verbose)
        cov = 100.0 * n_avail / n_total if n_total else 0.0
        info(f"Coverage now {cov:.1f}%  ({n_avail:,} of {n_total:,} records)",
             verbose=self.verbose)
        still_missing = n_total - n_avail
        if still_missing > 0:
            info(f"{still_missing:,} record(s) still missing (no model could predict them)",
                 verbose=self.verbose)
        if n_fb > 0:
            info(f"[yellow]Note:[/yellow] {n_fb:,} fallback fill(s) use timestamp "
                 f"features only (a predictor was missing) — lower quality.",
                 verbose=self.verbose)
        # In-sample scores (final model predicting ALL data incl. training rows —
        # optimistically biased; the held-out test scores above are the honest
        # estimate). Surfaced so both are on the record.
        self._log_scores_table(
            self._scores, "In-sample scores (all data, incl. training — optimistic)")
        self._log_importances_table(self._feature_importances,
                                    "Feature importances (final gap-filling model)")

    def trainmodel(self,
                   showplot_scores: bool = True,
                   showplot_importance: bool = True):
        """
        Train the model for gap-filling and score it on the held-out test set.

        No gap-filling is done here, only the model is trained (on ``train_df``)
        and evaluated on the unseen ``test_df``. That test set is a **random**
        subset of the complete rows (scattered in time, not a temporal block) —
        the correct validation for gap-filling, which interpolates short scattered
        gaps from nearby observed data rather than extrapolating long unseen
        periods. See the train/test split in :meth:`__init__` for the rationale.

        Args:
            showplot_scores: shows plot of predicted vs observed
            showplot_importance: shows plot of SHAP feature importances

        """

        self._section(f"MODEL TRAINING & TESTING — {self.target_col}")
        n_complete = len(self.model_df.dropna())
        info(f"{len(self.train_df.columns) - 1} feature(s), {n_complete:,} complete "
             f"record(s) of {len(self.model_df):,} "
             f"({self.model_df.index[0]} to {self.model_df.index[-1]})", verbose=self.verbose)
        info(f"Train/test split: {len(self.train_df):,} train / {len(self.test_df):,} test "
             f"({self.test_size * 100:.0f}% held out)", verbose=self.verbose)
        info("[dim]Test set is a RANDOM subset (scattered in time), not a temporal block "
             "— the right validation for gap-filling (short scattered gaps), not "
             "extrapolation.[/dim]", verbose=self.verbose)
        self._log_config()
        idtxt = f"TRAIN & TEST "

        # Set training and testing data
        train_df = self.train_df.copy()
        y_train = np.array(train_df[self.target_col])
        X_train = np.array(self.train_df.drop(self.target_col, axis=1))
        X_test = np.array(self.test_df.drop(self.target_col, axis=1))
        y_test = np.array(self.test_df[self.target_col])
        X_names = self.train_df.drop(self.target_col, axis=1).columns.tolist()

        # Train the model on training data (step chatter at DEBUG; numbers below).
        detail(f"Fitting {self.regressor.__name__} on training data "
               f"({self.train_df.index[0]} to {self.train_df.index[-1]}) ...",
               verbose=self.verbose)
        self._model = self._fitmodel(model=self._model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # Predict targets in test data
        detail(f"Predicting {self.target_col} on unseen test data ...", verbose=self.verbose)
        pred_y_test = self.model_.predict(X=X_test)

        # Calculate SHAP-based feature importance on test data and store in dataframe
        detail("Calculating SHAP feature importances from test data ...", verbose=self.verbose)
        self._feature_importances_traintest = self._shap_importance(
            model=self.model_, X=X_test, X_names=X_names)

        if showplot_importance:
            detail("Plotting feature importances (SHAP) ...", verbose=self.verbose)
            plot_feature_importance(feature_importances=self.feature_importances_traintest_)

        # Scores
        detail(f"Calculating prediction scores for {self.target_col} ...", verbose=self.verbose)
        self._scores_traintest = prediction_scores(predictions=pred_y_test, targets=y_test)

        if showplot_scores:
            info("Plotting observed vs predicted values ...", verbose=self.verbose)
            plot_observed_predicted(predictions=pred_y_test,
                                    targets=y_test,
                                    scores=self.scores_traintest_,
                                    infotxt=f"{idtxt} trained on training set, tested on test set",
                                    random_state=self._random_state)

            info("Plotting residuals and prediction error ...", verbose=self.verbose)
            plot_prediction_residuals_error_regr(
                model=self.model_, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                infotxt=f"{idtxt} trained on training set, tested on test set")

        # Collect results
        detail("Collecting results (call .report_traintest() for details).", verbose=self.verbose)
        self._traintest_details = dict(
            train_df=self.train_df,
            test_df=self.test_df,
            test_size=self.test_size,
            X_names=X_names,
            model=self.model_,
        )

        # Formatted, coloured summary for the console / GUI Log tab.
        self._log_scores_table(self._scores_traintest, "Held-out test scores")
        self._log_importances_table(self._feature_importances_traintest,
                                    "Top feature importances (held-out test)")
        success("Training complete.", verbose=self.verbose)

    def report_traintest(self):
        """Results from model training on test data"""

        idtxt = "MODEL TRAINING & TESTING RESULTS"
        results = self.traintest_details_
        fi = self.feature_importances_traintest_
        test_size_perc = self.test_size * 100
        training_size_perc = 100 - test_size_perc

        rule(idtxt)
        _console.print(
            f"\n"
            f"## DATA\n"
            f"  > target: {self.target_col}\n"
            f"  > features: {len(results['X_names'])} {results['X_names']}\n"
            f"  > {len(self.model_df)} records (with missing)\n"
            f"  > {len(self.model_df.dropna())} available records for target and all features (no missing values)\n"
            f"  > training on {len(self.train_df)} records ({training_size_perc:.1f}%) "
            f"between {self.train_df.index[0]} and {self.train_df.index[-1]}\n"
            f"  > testing on {len(self.test_df)} unseen records ({test_size_perc:.1f}%) "
            f"of {self.target_col} between {self.test_df.index[0]} and {self.test_df.index[-1]}\n"
            f"\n"
            f"## MODEL\n"
            f"  > estimator:  {self.model_}\n"
            f"  > parameters:  {self.model_.get_params()}\n"
            f"  > number of features:  {len(results['X_names'])}\n"
            f"  > feature names:  {results['X_names']}\n"
            f"\n"
            f"## FEATURE IMPORTANCES\n"
            f"  > calculated from unseen test data ({len(self.test_df[self.target_col])} records)\n"
            f"  > shows mean absolute SHAP values\n"
            f"\n{fi}\n"
            f"\n"
            f"## MODEL SCORES (test set, {len(self.test_df[self.target_col])} records)\n"
            f"  > MAE:   {self.scores_traintest_['mae']} (mean absolute error)\n"
            f"  > MedAE: {self.scores_traintest_['medae']} (median absolute error)\n"
            f"  > MSE:   {self.scores_traintest_['mse']} (mean squared error)\n"
            f"  > RMSE:  {self.scores_traintest_['rmse']} (root mean squared error)\n"
            f"  > MAXE:  {self.scores_traintest_['maxe']} (max error)\n"
            f"  > MAPE:  {self.scores_traintest_['mape']:.3f} (mean absolute percentage error)\n"
            f"  > R2:    {self.scores_traintest_['r2']}\n"
        )

    def fillgaps(self,
                 showplot_scores: bool = True,
                 showplot_importance: bool = True):
        """
        Gap-fill data with previously built model

        No new model is built here, instead the last model built in
        the preceding step .trainmodel() is used.

        y = target
        X = features

        """
        self._section(f"GAP-FILLING — {self.target_col}")
        self._fillgaps_fullmodel(showplot_scores, showplot_importance)
        self._fillgaps_fallback()
        self._fillgaps_combinepredictions()
        self._log_gapfill_summary()

    def reduce_features(self, shap_threshold_factor: float = 0.5):
        """Reduce number of features using SHAP importance

        A random variable is added to features and SHAP importances
        are calculated. The SHAP importance of the random variable is the
        benchmark to determine whether a feature is relevant. Features where
        SHAP importance is smaller or equal to (random_importance + k * random_sd)
        are rejected, where k is shap_threshold_factor.

        Args:
            shap_threshold_factor:
                Factor k for SHAP-based feature reduction threshold.
                Threshold is calculated as: random_importance + k * random_sd
                Default 0.5 uses 0.5-sigma confidence (lenient). Higher values are more conservative
                (reject more features). Lower values are more lenient.
        """

        infotxt = "[ FEATURE REDUCTION ]"

        self._section(f"FEATURE REDUCTION — {self.target_col}")
        info(f"Selecting features by SHAP importance vs a random benchmark "
             f"(k = {shap_threshold_factor}) ...", verbose=self.verbose)

        df = self.train_df.copy()
        df = df.dropna()

        # Add random variable as feature
        df, self.random_col = self._add_random_variable(df=df)

        X = np.array(df.drop(self.target_col, axis=1))
        y = np.array(df[self.target_col])

        # Instantiate model with params
        model = self.regressor(**self.kwargs)

        model.get_params()

        # Fit model to training data
        model = self._fitmodel(model=model, X_train=X, y_train=y, X_test=X, y_test=y)

        # https://mljar.com/blog/visualize-tree-from-random-forest/
        # todo from dtreeviz.trees import dtreeviz  # will be used for tree visualization
        # _ = tree.plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)

        # Calculate SHAP importance for all data
        detail("Calculating SHAP feature importances ...", verbose=self.verbose)
        X_names = df.drop(self.target_col, axis=1).columns.tolist()
        feature_importances = self._shap_importance(model=model, X=X, X_names=X_names)
        self._feature_importances_reduction = feature_importances.sort_values(by='SHAP_IMPORTANCE', ascending=False)

        # Remove variables where mean feature importance across all splits is smaller
        # than or equal to random variable
        # Update dataframe for model building
        accepted_cols = self._remove_rejected_features(shap_threshold_factor=shap_threshold_factor)

        # Update model data, keep accepted features
        detail("Removing rejected features from model data ...", verbose=self.verbose)
        self.train_df = self.train_df[accepted_cols].copy()
        self.test_df = self.test_df[accepted_cols].copy()
        self.model_df = self.model_df[accepted_cols].copy()

        self._accepted_features = self.train_df.drop(columns=self.target_col).columns.tolist()
        self._rejected_features = [x for x in X_names if x not in self.accepted_features_ and x != self.random_col]

    def report_feature_reduction(self):
        """Results from feature reduction"""

        rule("Feature Reduction")
        _console.print(
            f"\n"
            f"- Target variable: {self.target_col}\n"
            f"- Random variable {self.random_col} used as benchmark for importance threshold.\n"
            f"\n"
            f"SHAP IMPORTANCE (mean absolute SHAP values):\n"
            f"\n"
            f"{self.feature_importances_reduction_}\n"
            f"\n"
            f"- These results are from feature reduction. Final model importances are\n"
            f"  recalculated during gap-filling.\n"
            f"\n"
            f"  {len(self.original_input_features)} original input features: {self.original_input_features}\n"
            f"  {len(self.rejected_features_)} rejected features: "
            f"{self.rejected_features_ if self.rejected_features_ else 'none'}\n"
            f"  {len(self.accepted_features_)} accepted features: {self.accepted_features_}\n"
        )

    def report_gapfilling(self):
        """Results from gap-filling"""
        # Setup
        idtxt = "GAP-FILLING RESULTS"

        df = self.gapfilling_df_
        model = self.model_
        scores = self.scores_
        fi = self.feature_importances_

        feature_names = fi.index.to_list()
        n_features = len(feature_names)

        locs_observed = df[self.target_gapfilled_flag_col] == 0
        locs_hq = df[self.target_gapfilled_flag_col] == 1
        locs_observed_missing_fromflag = df[self.target_gapfilled_flag_col] > 0
        locs_fallback = df[self.target_gapfilled_flag_col] == 2

        n_observed = locs_observed.sum()
        n_hq = locs_hq.sum()
        n_observed_missing_fromflag = locs_observed_missing_fromflag.sum()
        n_available = len(df[self.target_gapfilled_col].dropna())
        n_potential = len(df.index)
        n_fallback = locs_fallback.sum()
        test_size_perc = self.test_size * 100

        rule(idtxt)
        _console.print(
            f"\n"
            f"Scores and importances from predicted ({n_hq}) vs observed ({n_observed}) targets.\n"
            f"\n"
            f"## TARGET\n"
            f"- period:  {df.index[0]} to {df.index[-1]}\n"
            f"- potential values: {n_potential}\n"
            f"- observed ({self.target_col}):  missing {df[self.target_col].isnull().sum()}\n"
            f"- gap-filled ({self.target_gapfilled_col}):  {n_available} values, "
            f"missing {df[self.target_gapfilled_col].isnull().sum()}\n"
            f"- flag {self.target_gapfilled_flag_col}:  "
            f"0=observed ({n_observed})  1=gap-filled ({n_hq})  2=fallback ({n_fallback})\n"
            f"\n"
            f"## FEATURE IMPORTANCES  ({n_features} features, SHAP TreeExplainer)\n"
            f"\n{fi}\n"
            f"\n"
            f"## MODEL  (test size {test_size_perc:.1f}%)\n"
            f"- estimator:  {model}\n"
            f"- parameters:  {model.get_params()}\n"
            f"\n"
            f"## MODEL SCORES (in-sample: predicted on ALL data incl. training rows; "
            f"optimistically biased)\n"
            f"- MAE:   {scores['mae']} (mean absolute error)\n"
            f"- MedAE: {scores['medae']} (median absolute error)\n"
            f"- MSE:   {scores['mse']} (mean squared error)\n"
            f"- RMSE:  {scores['rmse']} (root mean squared error)\n"
            f"- MAXE:  {scores['maxe']} (max error)\n"
            f"- MAPE:  {scores['mape']:.3f} (mean absolute percentage error)\n"
            f"- R2:    {scores['r2']}\n"
        )

        # Held-out scores from the train/test split are the honest generalization
        # estimate; surface them alongside the (biased) in-sample scores above.
        if self._scores_traintest:
            scores_tt = self._scores_traintest
            _console.print(
                f"## MODEL SCORES (out-of-sample: held-out test set, {test_size_perc:.1f}%; "
                f"generalization estimate)\n"
                f"- MAE:   {scores_tt['mae']} (mean absolute error)\n"
                f"- MedAE: {scores_tt['medae']} (median absolute error)\n"
                f"- MSE:   {scores_tt['mse']} (mean squared error)\n"
                f"- RMSE:  {scores_tt['rmse']} (root mean squared error)\n"
                f"- MAXE:  {scores_tt['maxe']} (max error)\n"
                f"- MAPE:  {scores_tt['mape']:.3f} (mean absolute percentage error)\n"
                f"- R2:    {scores_tt['r2']}\n"
            )

    @staticmethod
    def _build_tree_explainer(model):
        """Build a shap TreeExplainer, scoping the XGBoost base_score parse
        workaround to shap's own module namespace (see _shap_importance)."""
        try:
            from shap.explainers import _tree as _shap_tree
        except ImportError:
            # Unexpected shap layout: fall back to the native path.
            return shap.TreeExplainer(model)

        _real_float = _shap_tree.__dict__.get('float', float)
        _real_ast = getattr(_shap_tree, 'ast', None)

        def _patched_float(x):
            """Strip XGBoost's enclosing brackets, e.g. '[-4.12E0]' -> -4.12."""
            if isinstance(x, str):
                stripped = x.strip('[]')
                if stripped != x:
                    return _real_float(stripped)
            return _real_float(x)

        class _AstShim:
            """Delegates to the real ast, but strips XGBoost's brackets first."""

            def literal_eval(self, s):
                if isinstance(s, str):
                    stripped = s.strip()
                    if stripped.startswith('[') and stripped.endswith(']'):
                        try:
                            return [float(stripped[1:-1].strip())]
                        except ValueError:
                            pass
                return _real_ast.literal_eval(s)

            def __getattr__(self, name):
                return getattr(_real_ast, name)

        had_float = 'float' in _shap_tree.__dict__
        _shap_tree.float = _patched_float
        if _real_ast is not None:
            _shap_tree.ast = _AstShim()
        try:
            return shap.TreeExplainer(model)
        finally:
            if had_float:
                _shap_tree.float = _real_float
            else:
                del _shap_tree.float
            if _real_ast is not None:
                _shap_tree.ast = _real_ast

    def _shap_importance(self, model, X, X_names) -> DataFrame:
        """
        Calculate SHAP-based feature importance.

        Uses TreeExplainer for tree-based models (XGBoost, RandomForest).
        Returns mean absolute SHAP values as feature importance.
        """

        # Create explainer and calculate SHAP values.
        # XGBoost serializes base_score as bracket-enclosed scientific notation
        # (e.g. '[-3.18E0]'). Depending on the shap/Python combination, shap's tree
        # loader parses this via builtins float() (older shap, which a bare '[..]'
        # string breaks) or via ast.literal_eval() (newer shap, which Python 3.13
        # tightened). We override these ONLY inside shap's tree-explainer module
        # namespace — never process-global builtins.float / ast — so concurrent
        # threads (e.g. joblib parallel gap-filling) are unaffected. This works
        # because a bare float() call inside shap resolves against that module's
        # globals before falling back to builtins.
        explainer = self._build_tree_explainer(model)
        shap_values = explainer.shap_values(X)

        # Handle case where shap_values is a list (for some model types)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Calculate mean absolute SHAP values as importance
        importance_values = np.abs(shap_values).mean(axis=0)

        # Calculate standard deviation for reference
        importance_std = np.abs(shap_values).std(axis=0)

        # Create DataFrame with feature importances
        fidf = pd.DataFrame({
            'SHAP_IMPORTANCE': importance_values,
            'SHAP_SD': importance_std
        }, index=X_names)

        fidf = fidf.sort_values(by='SHAP_IMPORTANCE', ascending=False)

        return fidf

    def _add_random_variable(self, df: DataFrame) -> tuple[DataFrame, str]:
        # Add random variable as benchmark for relevant feature importances
        random_col = '.RANDOM'  # Random variable as benchmark for relevant importances
        df[random_col] = np.random.RandomState(self._random_state).randn(df.shape[0])
        return df, random_col

    # def _lag_features(self, features_lag_exclude_cols):
    #     """Add lagged variants of variables as new features"""
    #     exclude_cols = [self.target_col]
    #     if features_lag_exclude_cols:
    #         exclude_cols += features_lag_exclude_cols
    #     return diive.pkgs.createvar.laggedvariants.lagged_variants(df=self.model_df,
    #                                                                stepsize=self.features_lag_stepsize,
    #                                                                lag=self.features_lag,
    #                                                                exclude_cols=exclude_cols,
    #                                                                verbose=self.verbose)

    def _check_n_cols(self):
        """Check number of columns"""
        if len(self.model_df.columns) == 1:
            raise Exception(f"(!) Stopping execution because dataset comprises "
                            f"only one single column : {self.model_df.columns}")

    def _apply_below_zero_treatment(self, predictions: np.ndarray) -> np.ndarray:
        """Clip negative predictions to 0 for physically non-negative variables.

        Applied only when below_zero is set. Has no effect on observed data.
        """
        if self.below_zero is None:
            return predictions
        predictions = predictions.copy().astype(float)
        neg_locs = predictions < 0
        if neg_locs.any():
            n_neg = int(neg_locs.sum())
            # below_zero == 'zero'
            predictions[neg_locs] = 0.0
            detail(f"below_zero='zero': clipped {n_neg} negative prediction(s) to 0.",
                   verbose=self.verbose)
        return predictions

    def _fillgaps_fullmodel(self, showplot_scores, showplot_importance):
        """Apply model to fill missing targets for records where all features are available
        (high-quality gap-filling)"""

        detail("Gap-filling using final model ...", verbose=self.verbose)

        # Original input data, contains target and features
        # This dataframe has the full timestamp
        df = self.model_df.copy()

        # Test how the model performs with all y data
        # Since the model was previously trained on test data,
        # here it is checked how well the model performs when
        # predicting all available y data.
        # This is needed to calculate feature importance and scores.
        y, X, X_names, timestamp = fr.convert_to_arrays(
            df=df, target_col=self.target_col, complete_rows=True)

        # Predict all targets (no test split)
        detail(f"Predicting {self.target_col} with full model on all data ...", verbose=self.verbose)
        pred_y = self.model_.predict(X=X)

        # Calculate SHAP-based feature importance and store in dataframe
        detail("Calculating SHAP feature importances ...", verbose=self.verbose)
        self._feature_importances = self._shap_importance(
            model=self._model, X=X, X_names=X_names)

        if showplot_importance:
            info("Plotting feature importances (SHAP) ...", verbose=self.verbose)
            plot_feature_importance(feature_importances=self.feature_importances_)

        # Scores, using all targets
        detail(f"Calculating prediction scores for {self.target_col} ...", verbose=self.verbose)
        self._scores = prediction_scores(predictions=pred_y, targets=y)

        if showplot_scores:
            info("Plotting observed vs predicted values ...", verbose=self.verbose)
            plot_observed_predicted(predictions=pred_y,
                                    targets=y,
                                    scores=self.scores_,
                                    infotxt=f"trained on training set, tested on FULL set",
                                    random_state=self._random_state)

            # print(f">>> Plotting residuals and prediction error based on all data ...")
            # plot_prediction_residuals_error_regr(
            #     model=self.model_, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            #     infotxt=f"trained on training set, tested on full set")

        # In the next step, all available features are used to
        # predict the target for records where all features are available.
        # Feature data for records where all features are available:
        features_df = df.drop(self.target_col, axis=1)  # Remove target data
        features_df = features_df.dropna()  # Keep rows where all features available
        X = features_df.to_numpy()  # Features are needed as numpy array
        feature_names = features_df.columns.tolist()

        # Predict targets for all records where all features are available
        pred_y = self.model_.predict(X=X)
        pred_y = self._apply_below_zero_treatment(pred_y)
        info(f"Predicted {len(pred_y)} records where all features available.", verbose=self.verbose)

        # Collect gapfilling results in df
        # Define column names for gapfilled_df
        detail("Collecting results for final model ...", verbose=self.verbose)
        self._define_cols()

        # Collect results on the FULL timestamp (df.index), NOT only the rows
        # where all features are available. This is critical for data integrity:
        # a target value that is OBSERVED at a row where some feature is missing
        # (e.g. a driver gap that does not coincide with the target gap) must be
        # preserved as observed — never dropped and re-filled by the fallback.
        # Full-model predictions exist only where all features are available;
        # they are aligned to the full index (NaN elsewhere).
        self._gapfilling_df = pd.DataFrame(index=df.index)
        self._gapfilling_df[self.target_col] = df[self.target_col].copy()
        self._gapfilling_df[self.pred_fullmodel_col] = pd.Series(data=pred_y, index=features_df.index)

        # Gap locations: where the observed target is missing
        _gap_locs = self._gapfilling_df[self.target_col].isnull()
        # Full-model predictions at the gap locations (NaN where the model could
        # not predict because a feature was missing there).
        self._gapfilling_df[self.pred_gaps_col] = self._gapfilling_df.loc[
            _gap_locs, self.pred_fullmodel_col]

        # Gap-filled series: keep every observed value, fill gaps with the full
        # model where it could predict. fillna never overwrites observed values.
        n_missing = int(_gap_locs.sum())
        info(f"Filling {n_missing} missing records in {self.target_gapfilled_col} ...",
             verbose=self.verbose)
        self._gapfilling_df[self.target_gapfilled_col] = \
            self._gapfilling_df[self.target_col].fillna(self._gapfilling_df[self.pred_fullmodel_col])

        # Flag: 0 = observed, 1 = gap-filled by the full model. Records still
        # missing here (target gap AND no full-model prediction) keep flag 0 for
        # now and are set to 2 by the fallback in _fillgaps_fallback.
        flag = pd.Series(0, index=df.index, dtype=int)
        _filled_by_model = _gap_locs & self._gapfilling_df[self.pred_fullmodel_col].notna()
        flag[_filled_by_model] = 1
        self._gapfilling_df[self.target_gapfilled_flag_col] = flag

        # SHAP values
        # https://pypi.org/project/shap/
        # https://mljar.com/blog/feature-importance-in-random-forest/

    def _fillgaps_fallback(self):

        # Fallback gapfilling
        # Fill still existing gaps in full timestamp data
        # Build fallback model exclusively from timestamp features.
        # Here, the model is trained on the already gapfilled time series,
        # using info from the timestamp, e.g. DOY
        _still_missing_locs = self._gapfilling_df[self.target_gapfilled_col].isnull()
        _num_still_missing = _still_missing_locs.sum()  # Count number of still-missing values

        info(f"Fallback: filling {_num_still_missing} remaining gaps in "
             f"{self.target_gapfilled_col} ...", verbose=self.verbose)

        if _num_still_missing > 0:

            info(f"Fallback model trained on {self.target_gapfilled_col} and timestamp features ...",
                 verbose=self.verbose)

            fallback_predictions, \
                fallback_timestamp = \
                self._predict_fallback(series=self._gapfilling_df[self.target_gapfilled_col])

            fallback_series = pd.Series(data=fallback_predictions, index=fallback_timestamp)
            self._gapfilling_df[self.pred_fallback_col] = fallback_series
            self._gapfilling_df[self.target_gapfilled_col] = \
                self._gapfilling_df[self.target_gapfilled_col].fillna(fallback_series)

            # The fallback is the last-resort fill and leaves no gaps (its
            # predictions are always finite; below_zero only clips to 0, never
            # to NaN), so every still-missing record is now filled.
            self._gapfilling_df.loc[_still_missing_locs, self.target_gapfilled_flag_col] = 2  # 2=fallback
        else:
            detail("Fallback model not needed — all gaps already filled.", verbose=self.verbose)
            self._gapfilling_df[self.pred_fallback_col] = None

        # Cumulative
        self._gapfilling_df[self.target_gapfilled_cumu_col] = \
            self._gapfilling_df[self.target_gapfilled_col].cumsum()

    def _fillgaps_combinepredictions(self):
        """Combine predictions of full model with fallback predictions"""
        detail("Combining full-model and fallback predictions ...", verbose=self.verbose)
        # First add predictions from full model
        self._gapfilling_df[self.pred_col] = self._gapfilling_df[self.pred_fullmodel_col].copy()
        # Then fill remaining gaps with predictions from fallback model
        self._gapfilling_df[self.pred_col] = (
            self._gapfilling_df[self.pred_col].fillna(self._gapfilling_df[self.pred_fallback_col]))

    def _predict_fallback(self, series: pd.Series):
        """Fill data gaps using timestamp features only, fallback for still existing gaps"""
        gf_fallback_df = pd.DataFrame(series)
        gf_fallback_df = vectorize_timestamps(df=gf_fallback_df, txt="(ONLY FALLBACK)")

        # Build model for target predictions *from timestamp*
        y_fallback, X_fallback, _, _ = \
            fr.convert_to_arrays(df=gf_fallback_df,
                                 target_col=self.target_gapfilled_col,
                                 complete_rows=True)

        # Instantiate new model with same params as before
        model_fallback = self.regressor(**self.kwargs)

        # Train the model on all available records ...
        model_fallback = self._fitmodel(model=model_fallback, X_train=X_fallback, y_train=y_fallback, X_test=X_fallback,
                                        y_test=y_fallback)
        # model_fallback.fit(X=X_fallback, y=y_fallback)

        # ... and use it to predict all records for full timestamp
        full_timestamp_df = gf_fallback_df.drop(self.target_gapfilled_col, axis=1)  # Remove target data
        X_fallback_full = full_timestamp_df.to_numpy()  # Features are needed as numpy array

        info(f"Predicting {self.target_gapfilled_col} using fallback model ...", verbose=self.verbose)
        pred_y_fallback = model_fallback.predict(X=X_fallback_full)  # Predict targets in test data
        pred_y_fallback = self._apply_below_zero_treatment(pred_y_fallback)
        full_timestamp = full_timestamp_df.index

        return pred_y_fallback, full_timestamp

    def _results(self, gapfilled_df, most_important_df, model_r2, still_missing_locs):
        """Summarize gap-filling results"""

        _vals_max = len(gapfilled_df.index)
        _vals_before = len(gapfilled_df[self.target_col].dropna())
        _vals_after = len(gapfilled_df[self.target_gapfilled_col].dropna())
        _vals_fallback_filled = still_missing_locs.sum()
        _perc_fallback_filled = (_vals_fallback_filled / _vals_max) * 100

        _console.print(
            f"Gap-filling results for {self.target_col}\n"
            f"max possible: {_vals_max} values\n"
            f"before gap-filling: {_vals_before} values\n"
            f"after gap-filling: {_vals_after} values\n"
            f"gap-filled with fallback: {_vals_fallback_filled} values / {_perc_fallback_filled:.1f}%\n"
            f"used features:\n{most_important_df}\n"
            f"predictions vs targets, R2 = {model_r2:.3f}"
        )

    def _define_cols(self):
        self.pred_col = ".PREDICTIONS"
        self.pred_fullmodel_col = ".PREDICTIONS_FULLMODEL"
        self.pred_fallback_col = ".PREDICTIONS_FALLBACK"
        self.pred_gaps_col = ".GAP_PREDICTIONS"
        self.target_gapfilled_col = f"{self.target_col}{self.gfsuffix}"
        self.target_gapfilled_flag_col = f"FLAG_{self.target_gapfilled_col}_ISFILLED"  # "[0=measured]"
        self.target_gapfilled_cumu_col = ".GAPFILLED_CUMULATIVE"


def plot_feature_importance(feature_importances: pd.DataFrame):
    """
    Plot SHAP feature importance as horizontal bar chart with error bars.

    Visualizes relative importance of features in the model with standard deviation
    as error bars. Features are sorted by importance for easy interpretation.
    """
    # Scientific color palette
    COLOR_BAR = '#003A70'  # Deep Blue
    COLOR_ERROR = '#C41E3A'  # Crimson Red (error bars)
    COLOR_GRID = '#BDC3C7'  # Cool Gray
    COLOR_TEXT = '#2C3E50'  # Dark Slate Gray

    fig, ax = plt.subplots(figsize=(10, max(8, len(feature_importances) * 0.35)), dpi=100)

    # Prepare data
    _fidf = feature_importances.copy().sort_values(by='SHAP_IMPORTANCE', ascending=True)
    importances = _fidf['SHAP_IMPORTANCE']
    errors = _fidf['SHAP_SD']
    labels = _fidf.index

    # Create horizontal bar chart
    bars = ax.barh(range(len(importances)), importances,
                   color=COLOR_BAR, alpha=0.85, edgecolor=COLOR_TEXT, linewidth=0.8)

    # Add error bars
    ax.errorbar(importances, range(len(importances)), xerr=errors,
                fmt='none', ecolor=COLOR_ERROR, elinewidth=2, capsize=4,
                capthick=2, alpha=0.8, zorder=3)

    # Styling
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, color=COLOR_TEXT)
    ax.set_xlabel('Feature Importance (mean |SHAP value|)', fontsize=14,
                  color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Feature', fontsize=14, color=COLOR_TEXT, fontweight='600')
    ax.set_title('SHAP Feature Importance', fontsize=16, fontweight='bold',
                 color='black', pad=15)

    # Grid
    ax.grid(True, axis='x', alpha=0.4, color=COLOR_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11,
                   length=4, width=1, color=COLOR_TEXT)
    ax.tick_params(axis='x', labelsize=10)

    # Add value labels on bars
    for i, (imp, err) in enumerate(zip(importances, errors)):
        ax.text(imp + err * 0.5, i, f'{imp:.3f}', va='center', fontsize=9.5,
                color=COLOR_TEXT, fontweight='500')

    fig.tight_layout()
    fig.show()


def plot_observed_predicted(targets: np.ndarray,
                            predictions: np.ndarray,
                            scores: dict,
                            infotxt: str = "",
                            random_state: int = None):
    """
    Plot observed vs. predicted values with enhanced visual styling.

    Creates a 2-panel figure showing:
    - Left: Actual vs. Predicted scatter with accuracy bands and perfect prediction line
    - Right: Residuals vs. Predicted with zero line and error regions

    Visual styling follows diive's Material Design theme with color-coded accuracy zones.
    """
    # Scientific color palette - high contrast, publication-ready
    COLOR_SCATTER = '#003A70'  # Deep Blue
    COLOR_RESIDUAL = '#C41E3A'  # Crimson Red
    COLOR_PERFECT = '#2C3E50'  # Dark Slate Blue-Gray
    COLOR_GOOD = '#F4A300'  # Golden Yellow (±10% error)
    COLOR_WARN = '#E67F0D'  # Deep Orange (±20% error)
    COLOR_ERROR = '#C41E3A'  # Crimson Red (>20% error)
    COLOR_GRID = '#BDC3C7'  # Cool Gray
    COLOR_ZERO = '#000000'  # Black
    COLOR_TEXT = '#2C3E50'  # Dark Slate Gray

    fig, axs = plt.subplots(ncols=2, figsize=(14, 5.5), dpi=100)

    # ==================== PANEL 1: Actual vs. Predicted ====================
    ax = axs[0]

    # Calculate data ranges for reference lines and zones
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    margin = (max_val - min_val) * 0.05
    plot_min, plot_max = min_val - margin, max_val + margin

    # Add accuracy zones (filled regions)
    # Perfect prediction line (y=x) with ±10% and ±20% bands
    x_ref = np.array([plot_min, plot_max])

    # ±20% error zone (strong orange, widest)
    ax.fill_between(x_ref, x_ref * 0.80, x_ref * 1.20,
                    color=COLOR_WARN, alpha=0.18, zorder=0, label='±20% error band')

    # ±10% error zone (strong green, narrower)
    ax.fill_between(x_ref, x_ref * 0.90, x_ref * 1.10,
                    color=COLOR_GOOD, alpha=0.22, zorder=1, label='±10% error band')

    # Perfect prediction line (diagonal)
    ax.plot(x_ref, x_ref, '--', color=COLOR_PERFECT, lw=2, alpha=0.9,
            label='Perfect prediction', zorder=2)

    # Scatter plot with custom styling
    ax.scatter(targets, predictions,
               c=COLOR_SCATTER, edgecolors=COLOR_PERFECT, s=50, alpha=0.75,
               linewidth=0.8, zorder=3, label='Predictions')

    # Formatting
    ax.set_xlabel('Observed values', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Predicted values', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_title('Actual vs. Predicted', fontsize=15, fontweight='bold', color='black', pad=10)
    ax.grid(True, alpha=0.3, color=COLOR_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim([plot_min, plot_max])
    ax.set_ylim([plot_min, plot_max])

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=13,
                   length=5, width=1, color=COLOR_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor=COLOR_GRID)

    # ==================== PANEL 2: Residuals vs. Predicted ====================
    ax = axs[1]
    residuals = targets - predictions

    # Calculate residual statistics
    mean_residual = residuals.mean()
    std_residual = residuals.std()

    # Add reference bands (±1σ, ±2σ)
    zero_line_y = [plot_min, plot_max]
    ax.fill_between([plot_min, plot_max], -2 * std_residual, 2 * std_residual,
                    color=COLOR_WARN, alpha=0.18, zorder=0, label='±2σ region')
    ax.fill_between([plot_min, plot_max], -1 * std_residual, 1 * std_residual,
                    color=COLOR_GOOD, alpha=0.22, zorder=1, label='±1σ region')

    # Zero line (perfect predictions have zero residuals)
    ax.axhline(y=0, color=COLOR_ZERO, linestyle='-', linewidth=1.5, alpha=0.85, zorder=2)

    # Scatter plot for residuals
    ax.scatter(predictions, residuals,
               c=COLOR_RESIDUAL, edgecolors=COLOR_PERFECT, s=50, alpha=0.75,
               linewidth=0.8, zorder=3, label='Residuals')

    # Formatting
    ax.set_xlabel('Predicted values', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Residuals (Observed − Predicted)', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_title('Residuals vs. Predicted', fontsize=15, fontweight='bold', color='black', pad=10)
    ax.grid(True, alpha=0.3, color=COLOR_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim([plot_min, plot_max])

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=13,
                   length=5, width=1, color=COLOR_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor=COLOR_GRID)

    # ==================== Figure Title and Info Box ====================
    n_vals = len(predictions)
    mae = scores['mae']
    rmse = scores['rmse']
    r2 = scores['r2']

    # Main title
    title_text = f"Model Predictions: {infotxt}" if infotxt else "Model Predictions"
    fig.suptitle(title_text, fontsize=18, fontweight='bold', color='black', y=0.99)

    # Info box with metrics (positioned in figure space)
    info_lines = [
        f"n = {n_vals:,} samples",
        f"MAE = {mae:.4f}",
        f"RMSE = {rmse:.4f}",
        f"R² = {r2:.4f}"
    ]
    info_text = '\n'.join(info_lines)

    fig.text(0.99, 0.01, info_text,
             fontsize=11, color=COLOR_TEXT, verticalalignment='bottom',
             horizontalalignment='right', family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                       edgecolor=COLOR_GRID, linewidth=1.2, alpha=0.95))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()


def plot_prediction_residuals_error_regr(model,
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_test: np.ndarray,
                                         y_test: np.ndarray,
                                         infotxt: str):
    """
    Plot residuals and prediction error diagnostics.

    Creates two diagnostic plots using yellowbrick visualizers:
    - Left: Q-Q plot of residuals to assess normality assumption
    - Right: Actual vs. Predicted with prediction error visualization

    Args:
        model: Fitted regression model with predict() method
        X_train: predictors in training data (n_samples, n_features)
        y_train: targets in training data (n_samples,)
        X_test: predictors in test data (n_samples, n_features)
        y_test: targets in test data (n_samples,)
        infotxt: text displayed in figure header for context

    Notes:
        - Q-Q plot: If residuals are normally distributed, points should fall on
          the diagonal line. Deviations suggest non-normality.
        - Prediction Error: Shows scatter of predictions vs. actual values with
          perfect prediction line (y=x) and prediction confidence region.

    References:
        - https://www.scikit-yb.org/en/latest/api/regressor/residuals.html
        - https://www.scikit-yb.org/en/latest/api/regressor/peplot.html
    """

    # Scientific color palette for consistency
    COLOR_POINTS = '#003A70'  # Deep Blue
    COLOR_LINE = '#F4A300'  # Golden Yellow
    COLOR_ERROR = '#C41E3A'  # Crimson Red
    COLOR_TEXT = '#2C3E50'  # Dark Slate Gray

    # ==================== PLOT 1: Q-Q Plot (Residuals Normality) ====================
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    fig.suptitle(f"Residuals Analysis: {infotxt}", fontsize=15, fontweight='bold', y=0.98)

    # Q-Q Plot detects normality violations
    vis = ResidualsPlot(model, hist=False, qqplot=True, ax=ax)
    vis.fit(X_train, y_train)
    vis.score(X_test, y_test)

    # Enhance styling
    ax.set_title('Q-Q Plot (Normality Assessment)', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Theoretical Quantiles', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Sample Quantiles', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Update line colors
    for line in ax.get_lines():
        if line.get_marker() in ['.', 'o']:
            line.set_color(COLOR_POINTS)
            line.set_alpha(0.7)
        else:
            line.set_color(COLOR_LINE)
            line.set_linewidth(2)

    plt.tight_layout()
    vis.show()

    # ==================== PLOT 2: Prediction Error Plot ====================
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    fig.suptitle(f"Prediction Error: {infotxt}", fontsize=15, fontweight='bold', y=0.98)

    vis = PredictionError(model)
    vis.fit(X_train, y_train)
    vis.score(X_test, y_test)

    # Enhance styling
    ax.set_title('Actual vs. Predicted (with Confidence Region)', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Actual Values', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Predicted Values', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Update scatter point colors
    for collection in ax.collections:
        collection.set_edgecolor(COLOR_TEXT)
        collection.set_facecolor(COLOR_POINTS)
        collection.set_alpha(0.7)
        collection.set_linewidth(0.5)

    # Update line colors
    for line in ax.get_lines():
        if line.get_marker() in ['.', 'o', 's']:
            line.set_color(COLOR_POINTS)
            line.set_alpha(0.7)
        elif line.get_linestyle() == '--':
            line.set_color(COLOR_LINE)
            line.set_linewidth(2)
            line.set_alpha(0.85)
        else:
            line.set_color(COLOR_ERROR)
            line.set_linewidth(1.5)
            line.set_alpha(0.7)

    plt.tight_layout()
    vis.show()
