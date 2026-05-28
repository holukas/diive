"""
============================================
HYPERPARAMETER OPTIMIZATION FOR TIME SERIES
optimization
============================================

Generic hyperparameter optimization for any sklearn-compatible regressor
using GridSearchCV with TimeSeriesSplit to avoid data leakage.

Supports Random Forest, XGBoost, and any model implementing the sklearn
regressor interface.
"""
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from itertools import product
from joblib import Parallel, delayed

import diive.core.dfun.frames as fr
from diive.core.utils.console import console as _console, info, rule, success, error
from diive.gapfilling.scores import prediction_scores

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


class OptimizeParamsTS:
    """
    Optimize hyperparameters for any sklearn-compatible regressor using time series cross-validation.

    Supports Random Forest, XGBoost, and any model implementing the sklearn regressor interface.
    Uses GridSearchCV with TimeSeriesSplit to avoid data leakage on time series data.

    Visualization Features:
        - Parameter slice plots showing sensitivity to each hyperparameter
        - Parallel coordinates plot colored by performance (red=poor, blue=excellent)
        - Convergence analysis and parameter importance
        - Filters out non-optimized parameters (single value only)

    See Also:
        `examples/gapfilling/gapfill_optimize_randomforest.py` — Random Forest hyperparameter optimization
        `examples/gapfilling/gapfill_optimize_xgboost.py` — XGBoost hyperparameter optimization
    """

    def __init__(self,
                 df: DataFrame,
                 target_col: str,
                 regressor_class,
                 **model_params: dict):
        """Optimize regressor hyperparameters using GridSearchCV with time series CV.

        Args:
            df: DataFrame with target and predictor time series (must have complete rows)
            target_col: Column name of target variable
            regressor_class: Regressor class (not instance), e.g. RandomForestRegressor,
                           XGBRegressor, or any sklearn-compatible regressor
            **model_params: Parameter ranges to test as lists, e.g.:
                {
                    'n_estimators': [10, 50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }

        Methods:
            optimize(): Run GridSearchCV to find best parameters
            report_optimization(top_n=5): Print comprehensive report with recommendations

        Examples:
            # Random Forest optimization
            from sklearn.ensemble import RandomForestRegressor
            opt = OptimizeParamsTS(df=df, target_col='NEE',
                                   regressor_class=RandomForestRegressor,
                                   n_estimators=[10, 50, 100],
                                   max_depth=[5, 10, 15])
            opt.optimize()
            opt.report_optimization()

            # XGBoost optimization
            import xgboost as xgb
            opt = OptimizeParamsTS(df=df, target_col='NEE',
                                   regressor_class=xgb.XGBRegressor,
                                   n_estimators=[50, 100, 200],
                                   max_depth=[3, 6, 9])
            opt.optimize()
            opt.report_optimization()

        See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
             https://xgboost.readthedocs.io/en/stable/python/python_intro.html
        """
        self.model_df = df.copy()
        self.target_col = target_col
        self.regressor_class = regressor_class

        self.params = model_params

        # Attributes
        self._best_params = None
        self._scores = None
        self._cv_results = None
        self._best_score = None
        self._cv_n_splits = None

    @property
    def best_params(self) -> dict:
        """Estimator which gave highest score (or smallest loss if specified) on the left out data"""
        if not self._best_params:
            raise Exception(f'Not available: model scores.')
        return self._best_params

    @property
    def scores(self) -> dict:
        """Return model scores for best model"""
        if not self._scores:
            raise Exception(f'Not available: model scores.')
        return self._scores

    @property
    def cv_results(self) -> DataFrame:
        """Cross-validation results"""
        if not isinstance(self._cv_results, DataFrame):
            raise Exception(f'Not available: cv scores.')
        return self._cv_results

    @property
    def best_score(self) -> float:
        """Mean cross-validated score of the best_estimator"""
        if not self._best_score:
            raise Exception(f'Not available: cv scores.')
        return self._best_score

    @property
    def cv_n_splits(self) -> int:
        """The number of cross-validation splits (folds/iterations)"""
        if not self._cv_n_splits:
            raise Exception(f'Not available: cv scores.')
        return self._cv_n_splits

    def optimize(self):

        y, X, X_names, timestamp = \
            fr.convert_to_arrays(df=self.model_df,
                                 target_col=self.target_col,
                                 complete_rows=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Calculate total combinations for progress tracking
        total_combinations = 1
        for param_values in self.params.values():
            total_combinations *= len(param_values)
        n_splits = 10
        total_fits = total_combinations * n_splits
        rule(f"Hyperparameter Optimization: {self.regressor_class.__name__}")
        info(f"{total_combinations} combinations x {n_splits} CV folds = {total_fits} model fits")

        grid = GridSearchCV(estimator=self.regressor_class(),
                            param_grid=self.params,
                            scoring='neg_mean_squared_error',
                            cv=TimeSeriesSplit(n_splits=n_splits),
                            n_jobs=1,
                            verbose=2)

        grid.fit(X_train, y_train)

        success(f"Optimization complete: {total_combinations} combinations tested")

        self._cv_results = pd.DataFrame.from_dict(grid.cv_results_)

        # Best parameters after tuning
        # Estimator which gave highest score (or smallest loss if specified) on the left out data
        self._best_params = grid.best_params_

        # Mean cross-validated score of the best_estimator
        self._best_score = grid.best_score_

        # Scorer function used on the held out data to choose the best parameters for the model
        self._scorer = grid.scorer_

        # The number of cross-validation splits (folds/iterations)
        self._cv_n_splits = grid.n_splits_

        grid_predictions = grid.predict(X_test)

        # Stats
        self._scores = prediction_scores(predictions=grid_predictions,
                                         targets=y_test)

    def report_optimization(self, top_n: int = 5) -> None:
        """Print comprehensive optimization report with recommendations.

        Args:
            top_n: Number of top parameter combinations to show (default 5)
        """
        from rich.table import Table

        if not self._best_params:
            error("Run optimize() first")
            return

        model_name = self.regressor_class.__name__
        rule(f"{model_name} Hyperparameter Optimization Report")

        # Tested parameter ranges
        info("Parameter ranges tested:")
        for param, values in sorted(self.params.items()):
            if isinstance(values, list) and len(values) > 5:
                _console.print(f"  [cyan]{param:<25}[/cyan]  {values[0]} to {values[-1]} ({len(values)} values)")
            else:
                _console.print(f"  [cyan]{param:<25}[/cyan]  {values}")

        # Best parameters
        rule("Best Parameters", min_level=2)
        for param, value in sorted(self._best_params.items()):
            _console.print(f"  [cyan]{param:<25}[/cyan]  [green]{value}[/green]")

        # Performance
        rule("Best Model Performance (test set)", min_level=2)
        _console.print(
            f"  R2 Score  {self._scores['r2']:>10.4f}  (0-1, higher is better)\n"
            f"  MAE       {self._scores['mae']:>10.4f}  (mean absolute error)\n"
            f"  RMSE      {self._scores['rmse']:>10.4f}  (root mean squared error)"
        )

        # Top N combinations
        rule(f"Top {top_n} Parameter Combinations (CV score)", min_level=2)
        top_results = self._cv_results.nsmallest(top_n, 'rank_test_score')
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        table.add_column("Rank", style="dim", no_wrap=True)
        table.add_column("CV MSE", justify="right")
        for param in sorted(self.params.keys()):
            table.add_column(param, justify="right")
        for idx, row_idx in enumerate(top_results.index, 1):
            mean_mse = -top_results.loc[row_idx, 'mean_test_score']
            row_vals = [str(idx), f"{mean_mse:.6f}"]
            for param in sorted(self.params.keys()):
                param_key = f'param_{param}'
                row_vals.append(str(top_results.loc[row_idx, param_key]) if param_key in top_results.columns else "-")
            table.add_row(*row_vals)
        _console.print(table)

        # Recommendation
        rule("Recommendation for Production", min_level=2)
        r2 = self._scores['r2']
        if 'RandomForest' in model_name:
            wrapper_class = 'RandomForestTS'
        elif 'XGB' in model_name:
            wrapper_class = 'XGBoostTS'
        else:
            wrapper_class = f'# custom wrapper for {model_name}'
        params_str = "".join(f"        {p}={v},\n" for p, v in sorted(self._best_params.items()))
        _console.print(
            f"\n  model = {wrapper_class}(\n"
            f"      input_df=df_engineered,\n"
            f"      target_col='<your_target>',\n"
            f"{params_str}"
            f"      verbose=1,\n"
            f"      random_state=42\n"
            f"  )\n\n"
            f"  Expected performance: R2 ~ {r2:.4f}\n"
        )

    def plot_optimization_analysis(self):
        """Comprehensive visualization of hyperparameter optimization results.

        Creates a dynamic grid showing:
        - (top-left) Optimization convergence history
        - (top-right) Parameter importance analysis
        - (remaining) Parameter slices for each numeric parameter
        - (separate figure) Parallel coordinates plot

        Requires: matplotlib, numpy
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not self._cv_results.size:
            error("Run optimize() first")
            return

        cv_results = self._cv_results.copy()
        param_cols = [f'param_{p}' for p in self.params.keys()]
        param_names = list(self.params.keys())

        # Get test scores and convert negative MSE to R² if needed
        test_scores = cv_results['mean_test_score'].values

        # Convert negative MSE to R² for better interpretability
        if (test_scores < 0).all():
            target_variance = self.model_df[self.target_col].var()
            test_scores = 1 - (np.abs(test_scores) / target_variance)
            score_metric = 'R² Score'
        else:
            score_metric = 'R² Score' if (test_scores <= 1).all() and (test_scores >= -1).all() else 'Score'

        # First pass: identify numeric parameters with multiple values (for layout calculation)
        # Include parameters that are purely numeric OR have mixed types (strings + numbers)
        # AND have more than 1 unique value (nothing to compare if only 1 value)
        numeric_param_cols_temp = []
        numeric_param_names_temp = []

        for param_col, param_name in zip(param_cols, param_names):
            param_vals = cv_results[param_col].values.copy()
            unique_vals = cv_results[param_col].unique()

            # Skip parameters with only 1 unique value
            if len(unique_vals) <= 1:
                continue

            # Try pure numeric conversion first
            try:
                if any(v is None for v in param_vals):
                    param_vals = np.array([1000 if v is None else v for v in param_vals], dtype=float)
                else:
                    param_vals = param_vals.astype(float)
                numeric_param_cols_temp.append(param_col)
                numeric_param_names_temp.append(param_name)
            except (ValueError, TypeError):
                # Try mixed-type conversion: convert all values to indices
                # This handles parameters like max_features=['sqrt', 'log2', 0.5, 1]
                try:
                    unique_vals_list = list(unique_vals)
                    unique_vals_sorted = sorted(unique_vals_list, key=lambda x: (x is None, str(x)))
                    val_to_idx = {v: i for i, v in enumerate(unique_vals_sorted)}
                    # Successfully created mapping, so this is a valid parameter for visualization
                    numeric_param_cols_temp.append(param_col)
                    numeric_param_names_temp.append(param_name)
                except Exception:
                    # Skip if we can't handle it
                    continue

        # Calculate grid dimensions: 1 row for convergence/importance, then 2 cols per additional row for slices
        n_params_to_plot = len(numeric_param_cols_temp)
        n_slice_rows = (n_params_to_plot + 1) // 2 if n_params_to_plot > 0 else 0  # Ceiling division for 2 columns
        n_rows = 1 + n_slice_rows  # 1 row for convergence/importance + slice rows

        # Create dynamic grid for main analysis plots (only create subplots as needed)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 + 4 * n_slice_rows))
        fig.suptitle('Random Forest Hyperparameter Optimization Analysis', fontsize=14, fontweight='bold')

        # 1. Optimization Convergence History
        best_scores = []
        current_best = -np.inf
        for score in test_scores:
            current_best = max(current_best, score)
            best_scores.append(current_best)

        ax = axes[0, 0]
        ax.plot(range(len(best_scores)), best_scores, marker='o', linewidth=2, markersize=6, color='steelblue')
        ax.fill_between(range(len(best_scores)), best_scores, alpha=0.3, color='steelblue')
        ax.set_xlabel('Iteration (GridSearchCV combination #)', fontweight='bold')
        ax.set_ylabel(f'Best {score_metric}', fontweight='bold')
        ax.set_title('Optimization Convergence History', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Parameter Importance Analysis & Identify Numeric Parameters
        ax = axes[0, 1]

        importances = []
        importance_param_names = []
        numeric_param_cols = []
        numeric_param_names = []

        for param_col, param_name in zip(param_cols, param_names):
            param_vals = cv_results[param_col].values.copy()

            # Try pure numeric conversion first
            try:
                if any(v is None for v in param_vals):
                    param_vals = np.array([1000 if v is None else v for v in param_vals], dtype=float)
                else:
                    param_vals = param_vals.astype(float)
            except (ValueError, TypeError):
                # Try mixed-type conversion: convert all values to indices
                # This handles parameters like max_features=['sqrt', 'log2', 0.5, 1]
                try:
                    unique_vals = list(cv_results[param_col].unique())
                    unique_vals_sorted = sorted(unique_vals, key=lambda x: (x is None, str(x)))
                    val_to_idx = {v: i for i, v in enumerate(unique_vals_sorted)}
                    param_vals = np.array([val_to_idx.get(v, np.nan) for v in cv_results[param_col].values])
                except Exception:
                    # Skip if we can't handle it (purely categorical with no numeric mapping)
                    continue

            # Track numeric parameters for later parameter slices
            numeric_param_cols.append(param_col)
            numeric_param_names.append(param_name)

            param_normalized = (param_vals - np.nanmin(param_vals)) / (np.nanmax(param_vals) - np.nanmin(param_vals) + 1e-10)

            correlation = np.corrcoef(param_normalized, test_scores)[0, 1]
            importance = abs(correlation)
            importances.append(importance)
            importance_param_names.append(param_name)

        sorted_idx = np.argsort(importances)[::-1]
        sorted_names = [importance_param_names[i] for i in sorted_idx]
        sorted_importances = [importances[i] for i in sorted_idx]

        colors_imp = plt.cm.RdYlGn(np.array(sorted_importances) / max(sorted_importances))
        bars = ax.barh(sorted_names, sorted_importances, color=colors_imp, edgecolor='black', linewidth=1)
        ax.set_xlabel('Importance (|correlation with R²|)', fontweight='bold')
        ax.set_title('Parameter Importance Analysis', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                   va='center', ha='left', fontsize=9, fontweight='bold')

        # 3+. Parameter Slices for numeric parameters (dynamic grid layout)
        # Use pre-filtered parameters from first pass (already excludes single-value params)
        for idx, (param_col, param_name) in enumerate(zip(numeric_param_cols_temp, numeric_param_names_temp)):
            # Map to subplot position: first slice starts at row 1, 2 columns
            ax_row = 1 + idx // 2
            ax_col = idx % 2

            # Handle axes indexing for single or multiple rows
            if n_rows == 1:
                ax = axes[ax_col]
            else:
                ax = axes[ax_row, ax_col]

            # Handle special case for None values (e.g., max_depth)
            param_values = cv_results[param_col].unique()

            # Smart sorting: try numeric, fallback to string
            def sort_key(x):
                if x is None:
                    return (True, 0)  # None comes first
                try:
                    return (False, float(x))  # Numeric sort
                except (ValueError, TypeError):
                    return (False, str(x))  # String sort

            sorted_vals = sorted(param_values, key=sort_key)

            for i, param_val in enumerate(sorted_vals):
                if param_val is None:
                    mask = cv_results[param_col].isna()
                    label = 'None (unlimited)'
                    x_pos = i
                else:
                    mask = cv_results[param_col] == param_val
                    # Format label: try int for numeric, otherwise use string representation
                    try:
                        label = f'{int(param_val)}'
                    except (ValueError, TypeError):
                        label = str(param_val)
                    x_pos = i

                scores = test_scores[mask]
                ax.scatter([x_pos] * len(scores), scores, s=80, alpha=0.6, label=label)

            ax.set_xlabel(param_name, fontweight='bold')
            ax.set_ylabel('Mean Test R² Score', fontweight='bold')
            ax.set_title(f'Parameter Slice: {param_name}', fontweight='bold')
            ax.set_xticks(range(len(sorted_vals)))

            # Format x-tick labels: handle both numeric and categorical parameters
            tick_labels = []
            for v in sorted_vals:
                if v is None:
                    tick_labels.append('None')
                else:
                    try:
                        num_val = float(v)
                        # Use more decimal places for values < 1 (fractions/rates)
                        if num_val < 1:
                            tick_labels.append(f'{num_val:.2f}')
                        else:
                            tick_labels.append(f'{int(num_val)}')
                    except (ValueError, TypeError):
                        tick_labels.append(str(v))
            ax.set_xticklabels(tick_labels, rotation=45 if any(len(str(v)) > 3 for v in sorted_vals) else 0)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot(s) if odd number of parameters to plot
        if n_params_to_plot > 0 and n_params_to_plot % 2 == 1:
            # Hide the rightmost subplot in the last row if we have an odd number
            ax_row = 1 + (n_params_to_plot - 1) // 2
            ax_col = 1
            if n_rows > 1:
                axes[ax_row, ax_col].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Parallel Coordinates plot (separate figure)
        self.plot_parallel_coordinates()

    def plot_parallel_coordinates(self):
        """Parallel coordinates visualization of all parameter combinations and performance.

        Shows each parameter combination as a line connecting normalized parameter values,
        colored by test score to identify high-performing regions.

        Requires: matplotlib, numpy, pandas
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not self._cv_results.size:
            error("Run optimize() first")
            return

        cv_results = self._cv_results.copy()
        param_cols = [f'param_{p}' for p in self.params.keys()]
        param_names = list(self.params.keys())

        # Get test scores and convert negative MSE to R² if needed
        test_scores = cv_results['mean_test_score'].values

        if (test_scores < 0).all():
            target_variance = self.model_df[self.target_col].var()
            test_scores = 1 - (np.abs(test_scores) / target_variance)

        # Prepare normalized data for parallel coordinates (skip categorical parameters)
        dimensions = []
        data_normalized = []

        for param_col, param_name in zip(param_cols, param_names):
            vals = cv_results[param_col].values.copy()

            # Skip categorical parameters for parallel coordinates
            try:
                if any(v is None for v in vals):
                    vals = np.array([1000 if v is None else v for v in vals], dtype=float)
                else:
                    vals = vals.astype(float)
            except (ValueError, TypeError):
                # Skip categorical parameters like 'criterion'
                continue

            # Skip parameters with only one unique value (not optimized)
            if len(np.unique(vals)) <= 1:
                continue

            normalized = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
            data_normalized.append(normalized)
            dimensions.append(param_name)

        dimensions.append('R² Score')
        r2_normalized = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-10)
        data_normalized.append(r2_normalized)
        num_dims = len(dimensions)

        # Create parallel coordinates plot
        fig, ax = plt.subplots(figsize=(14, 6))

        colors_norm = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-10)
        colormap = plt.cm.RdYlBu  # Red (low) -> Yellow (medium) -> Blue (high)

        for i, (row_data, color_val) in enumerate(zip(zip(*data_normalized), colors_norm)):
            color = colormap(color_val)
            ax.plot(range(num_dims), row_data, color=color, alpha=0.3, linewidth=1)

        ax.set_xticks(range(num_dims))
        ax.set_xticklabels(dimensions, fontsize=11, fontweight='bold')
        ax.set_ylabel('Normalized Value (0-1)', fontsize=11, fontweight='bold')
        ax.set_title('Parallel Coordinates: All Parameter Combinations & Performance', fontsize=13, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.3)

        for x in range(num_dims):
            ax.axvline(x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=test_scores.min(), vmax=test_scores.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('R² Score', fontweight='bold')

        plt.tight_layout()
        plt.show()

        info(f"Parallel coordinates: {len(cv_results)} parameter combinations. "
             f"Each line is one combination; blue = high R², red = poor performance.")
