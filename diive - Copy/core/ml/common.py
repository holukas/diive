import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Import the model we are using


def train_random_forest_regressor(targets: np.array, features: np.array, **rf_model_params):
    """
    Create model and train on features and targets

    See:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    Args:
        targets:
        features:
        **rf_model_params:

    Returns:

    """

    # Instantiate model with x decision trees
    model = RandomForestRegressor(**rf_model_params)

    # Fit
    model.fit(X=features, y=targets)  # Train the model on data
    model_r2 = model.score(X=features, y=targets)

    return model, model_r2


def model_importances(model, feature_names, threshold_important_features: float or None = None):
    """Store all feature importances in sorted list"""
    # Get numerical feature importances from current model
    importances = list(model.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(
        feature, round(importance, 2)) for feature, importance in zip(feature_names, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    most_important_df = pd.DataFrame.from_records(feature_importances, columns=['Var', 'Importance'])
    if threshold_important_features:
        most_important_df = most_important_df.loc[most_important_df['Importance'] >= threshold_important_features, :]
    most_important_vars = most_important_df['Var'].to_list()

    return feature_importances, most_important_df, most_important_vars


def mape_acc(predictions: np.array, targets: np.array):
    """Calculate mean absolute percentage error (MAPE) and accuracy"""
    _abs_errors = abs(predictions - targets)  # Calculate the absolute errors
    mae = np.mean(_abs_errors)  # Mean absolute error
    _temp = 100 * (_abs_errors / targets)
    mape = np.mean(_temp)  # Mean absolute percentage error
    accuracy = 100 - np.mean(mape)
    return mape, accuracy, mae
