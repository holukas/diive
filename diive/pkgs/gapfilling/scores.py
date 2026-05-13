"""
GAP-FILLING: MODEL SCORING
===========================

Performance metrics for gap-filling models: R², MAE, RMSE, prediction intervals.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error, max_error, r2_score


def prediction_scores(predictions: np.array,
                      targets: np.array) -> dict:
    """Calculate regression performance metrics for model predictions.

    Computes seven standard regression quality metrics comparing predicted values
    to observed targets. Useful for evaluating gap-filling models and other
    regression tasks.

    Args:
        predictions: Array of predicted values.
        targets: Array of observed target values (same length as predictions).

    Returns:
        dict: Performance metrics with keys:
            - mae: Mean Absolute Error (average absolute difference)
            - medae: Median Absolute Error (robust to outliers)
            - mse: Mean Squared Error (penalizes larger errors more)
            - rmse: Root Mean Squared Error (same units as targets)
            - mape: Mean Absolute Percentage Error (relative error %)
            - maxe: Maximum Error (largest single prediction error)
            - r2: Coefficient of Determination (0-1, higher is better)

    See: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """

    # Calculate stats
    scores = {
        'mae': mean_absolute_error(targets, predictions),
        'medae': median_absolute_error(targets, predictions),
        'mse': mean_squared_error(targets, predictions),
        'rmse': root_mean_squared_error(targets, predictions),
        'mape': mean_absolute_percentage_error(targets, predictions),
        'maxe': max_error(targets, predictions),
        'r2': r2_score(targets, predictions)
    }
    return scores
