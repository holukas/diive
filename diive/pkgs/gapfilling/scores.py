import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error, max_error, r2_score


def prediction_scores(predictions: np.array,
                      targets: np.array) -> dict:
    """
    Calculate prediction scores

    See:
    - https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
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
