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

import diive.core.dfun.frames as fr
from diive.pkgs.gapfilling.scores import prediction_scores

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


class OptimizeParamsTS:
    """
    Optimize hyperparameters for any sklearn-compatible regressor using time series cross-validation.

    Supports Random Forest, XGBoost, and any model implementing the sklearn regressor interface.
    Uses GridSearchCV with TimeSeriesSplit to avoid data leakage on time series data.

    See Also:
        examples/gap_filling/randomforest_ts.py — Random Forest hyperparameter optimization
        examples/gap_filling/xgboost_ts.py — XGBoost hyperparameter optimization
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

        grid = GridSearchCV(estimator=self.regressor_class(),
                            param_grid=self.params,
                            scoring='neg_mean_squared_error',
                            cv=TimeSeriesSplit(n_splits=10),
                            n_jobs=-1)
        grid.fit(X_train, y_train)

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
        if not self._best_params:
            print("ERROR: Run optimize() first")
            return

        model_name = self.regressor_class.__name__
        print("\n" + "=" * 80)
        print(f"{model_name.upper()} HYPERPARAMETER OPTIMIZATION REPORT")
        print("=" * 80)

        # Tested parameter ranges
        print("\nOK PARAMETER RANGES TESTED")
        print("-" * 80)
        for param, values in sorted(self.params.items()):
            if isinstance(values, list):
                if len(values) > 5:
                    print(f"  {param:<25} : {values[0]} to {values[-1]} ({len(values)} values)")
                else:
                    print(f"  {param:<25} : {values}")
            else:
                print(f"  {param:<25} : {values}")

        # Best parameters section
        print("\nOK BEST PARAMETERS (GridSearchCV winner)")
        print("-" * 80)
        for param, value in sorted(self._best_params.items()):
            print(f"  {param:<25} = {value}")

        # Best performance section
        print("\nOK BEST MODEL PERFORMANCE (test set)")
        print("-" * 80)
        print(f"  R2 Score              = {self._scores['r2']:>10.4f}  (0-1 scale, higher is better)")
        print(f"  MAE                   = {self._scores['mae']:>10.4f}  (mean absolute error)")
        print(f"  RMSE                  = {self._scores['rmse']:>10.4f}  (root mean squared error)")

        # Top N combinations
        print(f"\nOK TOP {top_n} PARAMETER COMBINATIONS (by CV score)")
        print("-" * 80)
        top_results = self._cv_results.nsmallest(top_n, 'rank_test_score')
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"\n  Rank {idx}:")
            mean_score = -row['mean_test_score']  # Negate because neg_mean_squared_error
            print(f"    CV Score: {mean_score:.6f} (lower MSE is better)")
            for param in sorted(self.params.keys()):
                param_key = f'param_{param}'
                if param_key in row:
                    print(f"    {param:<22} = {row[param_key]}")

        # Parameter sensitivity analysis
        print("\nOK PARAMETER SENSITIVITY (which parameters matter most)")
        print("-" * 80)
        for param in sorted(self.params.keys()):
            param_key = f'param_{param}'
            if param_key in self._cv_results.columns:
                unique_values = self._cv_results[param_key].unique()
                if len(unique_values) > 1:
                    print(f"  {param:<25} : {len(unique_values)} values tested")

        # Recommendation section
        print("\n" + "=" * 80)
        print("RECOMMENDATION FOR PRODUCTION")
        print("=" * 80)
        r2 = self._scores['r2']

        # Determine wrapper class based on regressor type
        model_name = self.regressor_class.__name__
        if 'RandomForest' in model_name:
            wrapper_class = 'RandomForestTS'
        elif 'XGB' in model_name:
            wrapper_class = 'XGBoostTS'
        else:
            wrapper_class = f'# Your custom gap-filling wrapper for {model_name}'

        # Generate parameters string from best_params
        params_str = ""
        for param, value in sorted(self._best_params.items()):
            params_str += f"        {param}={value},\n"

        print(f"""
Use these parameters for gap-filling with {model_name}:

    model = {wrapper_class}(
        input_df=df_engineered,
        target_col='<your_target>',
{params_str}        verbose=1,
        random_state=42
    )

Expected performance: R2 ~ {r2:.4f}
"""
              )
        print("=" * 80 + "\n")
