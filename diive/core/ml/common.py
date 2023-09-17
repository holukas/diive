# todo check for other estimators

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor  # Import the model we are using
from sklearn.inspection import permutation_importance
from sklearn.metrics import PredictionErrorDisplay, max_error, median_absolute_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score, mean_squared_error
from yellowbrick.regressor import PredictionError, ResidualsPlot


def feature_importances(estimator: RandomForestRegressor,
                        X: ndarray,
                        y: ndarray,
                        model_feature_names: list,
                        perm_n_repeats: int = 10,
                        random_col: str = None,
                        showplot: bool = True,
                        verbose: int = 1) -> dict:
    """
    Calculate feature importance, based on built-in method and permutation
    
    The built-in method for RandomForestRegressor() is Gini importance.


    See:
    - https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
    
    Args:
        estimator: fitted estimator
        X: features to predict y, required for permutation importance
        y: targets, required for permutation importance
        model_feature_names: list
        perm_n_repeats: number of repeats for computing permutation importance
        random_col: name of the random variable used as benchmark for relevant importance results
        showplot: shows plot of permutation importance results
        verbose: print details

    Returns:
        list of recommended features where permutation importance was higher than random, and
        two dataframes with overview of filtered and unfiltered importance results, respectively
    """
    # Store built-in feature importance (Gini)
    importances_gini_df = pd.DataFrame({'GINI_IMPORTANCE': estimator.feature_importances_},
                                       index=model_feature_names)

    # Calculate permutation importance
    # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-feature-importance
    perm_results = permutation_importance(estimator, X, y, n_repeats=perm_n_repeats, random_state=42,
                                          scoring='r2', n_jobs=-1)

    # Store permutation importance
    importances_perm_df = pd.DataFrame({'PERM_IMPORTANCE': perm_results.importances_mean,
                                        'PERM_SD': perm_results.importances_std},
                                       index=model_feature_names)

    # Store importance in one df
    importances_df = pd.concat([importances_perm_df, importances_gini_df], axis=1)
    importances_df = importances_df.sort_values(by='PERM_IMPORTANCE', ascending=True)

    # Keep features with higher permutation importance than random variable
    if random_col:
        perm_importance_threshold = importances_df['PERM_IMPORTANCE'][random_col]
        filtered_importances_df = \
            importances_df.loc[importances_df['PERM_IMPORTANCE'] > perm_importance_threshold].copy()

        # Get list of recommended features where permutation importance is larger than random
        recommended_features = filtered_importances_df.index.tolist()

        # Find rejected features below the importance threshold
        before_cols = importances_df.index.tolist()
        after_cols = filtered_importances_df.index.tolist()
        rejected_features = []
        for item in before_cols:
            if item not in after_cols:
                rejected_features.append(item)

        if verbose > 0:
            print(f"Accepted variables: {after_cols}  -->  "
                  f"above permutation importance threshold of {perm_importance_threshold}")
            print(f"Rejected variables: {rejected_features}  -->  "
                  f"below permutation importance threshold of {perm_importance_threshold}")
    else:
        # No random variable considered
        perm_importance_threshold = None
        recommended_features = importances_df.index.tolist()
        filtered_importances_df = importances_df.copy()

    if showplot:
        fig, axs = plt.subplots(ncols=2, figsize=(16, 9))

        importances_df['PERM_IMPORTANCE'].plot.barh(color='#008bfb', yerr=importances_df['PERM_SD'], ax=axs[0])
        axs[0].set_xlabel("Permutation importance")
        axs[0].set_ylabel("Feature")
        axs[0].set_title("Permutation importance")
        axs[0].legend(loc='lower right')

        importances_df['GINI_IMPORTANCE'].plot.barh(color='#008bfb', ax=axs[1])
        axs[1].set_xlabel("Gini importance")
        axs[1].set_title("Built-in importance (Gini)")
        axs[1].legend(loc='lower right')

        if random_col:
            # Check Gini importance of random variable, used for display purposes only (plot)
            gini_importance_threshold = importances_df['GINI_IMPORTANCE'][random_col]
            axs[0].axvline(perm_importance_threshold, color='#ff0051', ls='--', label="importance threshold")
            axs[1].axvline(gini_importance_threshold, color='#ff0051', ls='--', label="importance threshold")

        fig.tight_layout()
        fig.show()

    importances = {
        'recommended_features': recommended_features,
        'filtered_importances': filtered_importances_df,
        'importances': importances_df
    }

    return importances


def prediction_scores_regr(predictions: np.array,
                           targets: np.array,
                           infotxt: str = None,
                           showplot: bool = True) -> dict:
    """
    Calculate prediction scores for regression estimator

    See:
    - https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """

    # Calculate stats
    scores = {
        'mae': mean_absolute_error(targets, predictions),
        'medae': median_absolute_error(targets, predictions),
        'mse': mean_squared_error(targets, predictions),
        'rmse': mean_squared_error(targets, predictions, squared=False),  # root mean squared error
        'mape': mean_absolute_percentage_error(targets, predictions),
        'maxe': max_error(targets, predictions),
        'r2': r2_score(targets, predictions)
    }

    # Plot observed and predicted
    if showplot:
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))

        PredictionErrorDisplay.from_predictions(
            targets,
            y_pred=predictions,
            kind="actual_vs_predicted",
            subsample=None,
            ax=axs[0],
            random_state=42,
        )
        axs[0].set_title("Actual vs. Predicted values")

        PredictionErrorDisplay.from_predictions(
            targets,
            y_pred=predictions,
            kind="residual_vs_predicted",
            subsample=None,
            ax=axs[1],
            random_state=42,
        )
        axs[1].set_title("Residuals vs. Predicted Values")

        n_vals = len(predictions)
        fig.suptitle(f"Plotting cross-validated predictions ({infotxt})\n"
                     f"n_vals={n_vals}, MAE={scores['mae']:.3f}, RMSE={scores['rmse']:.3f}, r2={scores['r2']:.3f}")
        plt.tight_layout()
        plt.show()
    return scores


def plot_prediction_residuals_error_regr(model,
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_test: np.ndarray,
                                         y_test: np.ndarray,
                                         infotxt: str):
    """
    Plot residuals and prediction error

    Args:
        model:
        X_train: predictors in training data
        y_train: targets in training data
        X_test: predictors in test data
        y_test: targets in test data
        infotxt: text displayed in figure header

    Kudos:
    - https://www.scikit-yb.org/en/latest/api/regressor/residuals.html
    - https://www.scikit-yb.org/en/latest/api/regressor/peplot.html

    """

    # fig, axs = plt.subplots(ncols=2, figsize=(14, 4))
    # fig, ax = plt.subplots()

    # Histogram can be replaced with a Q-Q plot, which is a common way
    # to check that residuals are normally distributed. If the residuals
    # are normally distributed, then their quantiles when plotted against
    # quantiles of normal distribution should form a straight line.
    fig, ax = plt.subplots()
    fig.suptitle(f"{infotxt}")
    vis = ResidualsPlot(model, hist=False, qqplot=True, ax=ax)
    vis.fit(X_train, y_train)  # Fit the training data to the visualizer
    vis.score(X_test, y_test)  # Evaluate the model on the test data
    vis.show()  # Finalize and render the figure

    # difference between the observed value of the target variable (y)
    # and the predicted value (ŷ), i.e. the error of the prediction
    fig, ax = plt.subplots()
    fig.suptitle(f"{infotxt}")
    vis = PredictionError(model)
    vis.fit(X_train, y_train)  # Fit the training data to the visualizer
    vis.score(X_test, y_test)  # Evaluate the model on the test data
    vis.show()

    # fig.suptitle(f"{infotxt}")
    # plt.tight_layout()
    # fig.show()
