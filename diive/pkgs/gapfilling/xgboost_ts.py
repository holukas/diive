# TODO generalization bias
# TODO SHAP values
# https://pypi.org/project/shap/
# https://mljar.com/blog/feature-importance-in-random-forest/

"""
=========================================
XGBOOST GAP-FILLING FOR TIME SERIES
=========================================

This module is part of the diive library:
https://gitlab.ethz.ch/diive/diive

    - Example notebook available in:
        XXX

Kudos:
    - XXX

"""
import numpy as np
import xgboost as xgb
from pandas import DataFrame

from diive.core.ml.common import MlRegressorGapFillingBase


class XGBoostTS(MlRegressorGapFillingBase):

    def __init__(self, input_df: DataFrame, target_col: str or tuple, verbose: int = 0, perm_n_repeats: int = 10,
                 test_size: float = 0.25, features_lag: list = None, include_timestamp_as_features: bool = False,
                 add_continuous_record_number: bool = False, sanitize_timestamp: bool = False, **kwargs):
        """
        Gap-fill timeseries with predictions from random forest model

        Args:
            input_df:
                Contains timeseries of 1 target column and 1+ feature columns.

            target_col:
                Column name of variable in *input_df* that will be gap-filled.

            perm_n_repeats:
                Number of repeats for calculating permutation feature importance.

            test_size:
                Proportion of the dataset to include in the test split,
                between 0.0 and 1.0.

            features_lag:
                List of integers (number of records), includes lagged variants of predictors.
                If features_lag=None, no lagged variants are added.
                Example:
                    - features_lag=[-2, +2] includes variants that are lagged by -2, -1, +1 and
                    +2 records in the dataset, for each feature already present in the data.
                     For a variable named *TA*, this created the following output:
                    TA    = [  5,   6,   7, 8  ]
                    TA-2  = [NaN, NaN,   5, 6  ]
                    TA-1  = [NaN,   5,   6, 7  ]  --> each TA record is paired with the preceding record TA-1
                    TA+1  = [  6,   7,   8, NaN]  --> each TA record is paired with the next record TA+1
                    TA+2  = [  7,   8, NaN, NaN]

            include_timestamp_as_features:
                Include timestamp info as integer data: year, season, month, week, doy, hour

            add_continuous_record_number:
                Add continuous record number as new column

            sanitize_timestamp:
                Validate and prepare timestamps for further processing

            **kwargs:
                Keyword arguments for xgboost.XGBRegressor

        Attributes:
            gapfilled_df
            - .PREDICTIONS_FULLMODEL uses the output from the full RF model where
              all features where available.
            - .PREDICTIONS_FALLBACK uses the output from the fallback RF model, which
              was trained on the combined observed + .PREDICTIONS_FULLMODEL data, using
              only the timestamp info as features.
        """

        # Args
        super().__init__(
            regressor=xgb.XGBRegressor,
            input_df=input_df,
            target_col=target_col,
            verbose=verbose,
            perm_n_repeats=perm_n_repeats,
            test_size=test_size,
            features_lag=features_lag,
            include_timestamp_as_features=include_timestamp_as_features,
            add_continuous_record_number=add_continuous_record_number,
            sanitize_timestamp=sanitize_timestamp,
            **kwargs
        )


def example_xgbts():
    # Setup, user settings
    # TARGET_COL = 'LE_orig'
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    # Example data
    from diive.configs.exampledata import load_exampledata_parquet
    df_orig = load_exampledata_parquet()

    # # Create a large gap
    # remove = df.index.year != 2014
    # # df = df.drop(df.index[100:2200])
    # df = df[remove].copy()

    # Subset
    # keep = df_orig.index.year >= 2021
    # df = df_orig[keep].copy()
    df = df_orig.copy()

    # Subset with target and features
    # Only High-quality (QCF=0) measured NEE used for model training in this example
    lowquality = df["QCF_NEE"] > 0
    df.loc[lowquality, TARGET_COL] = np.nan
    df = df[subsetcols].copy()

    # from diive.core.plotting.timeseries import TimeSeries  # For simple (interactive) time series plotting
    # TimeSeries(series=df[TARGET_COL]).plot()

    # https://xgboost.readthedocs.io/en/stable/parameter.html
    # https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
    # https://medium.com/data-design/let-me-learn-the-learning-rate-eta-in-xgboost-d9ad6ec78363

    # XGBoost
    xgbts = XGBoostTS(
        input_df=df,
        target_col=TARGET_COL,
        verbose=1,
        # features_lag=None,
        features_lag=[-1, -1],
        # include_timestamp_as_features=False,
        include_timestamp_as_features=True,
        # add_continuous_record_number=False,
        add_continuous_record_number=True,
        sanitize_timestamp=True,
        perm_n_repeats=9,
        n_estimators=99,
        random_state=42,
        # booster='gbtree',  # gbtree (default), gblinear, dart
        # device='cpu',
        validate_parameters=True,
        # disable_default_eval_metric=False,
        early_stopping_rounds=10,
        max_depth=0,
        # max_delta_step=0,
        # subsample=1,
        learning_rate=0.3,
        # min_split_loss=0,
        # min_child_weight=1,
        # colsample_bytree=1,
        # colsample_bylevel=1,
        # colsample_bynode=1,
        # reg_lambda=1,
        # reg_alpha=0,
        tree_method='hist',  # auto, hist, approx, exact
        # scale_pos_weight=1,
        # grow_policy='depthwise',  # depthwise, lossguide
        # max_leaves=0,
        # max_bin=256,
        # num_parallel_tree=1,
        n_jobs=-1
    )
    xgbts.reduce_features()
    xgbts.report_feature_reduction()

    xgbts.trainmodel(showplot_scores=False, showplot_importance=False)
    xgbts.report_traintest()

    xgbts.fillgaps(showplot_scores=False, showplot_importance=False)
    xgbts.report_gapfilling()

    observed = df[TARGET_COL]
    gapfilled = xgbts.get_gapfilled_target()
    # rfts.feature_importances
    # rfts.scores
    # rfts.gapfilling_df

    # # https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
    # import shap
    # explainer = shap.TreeExplainer(rfts.model_)
    # xtest = rfts.traintest_details_['X_test']
    # shap_values = explainer.shap_values(xtest)
    # shap.summary_plot(shap_values, xtest)
    # # shap.summary_plot(shap_values[0], xtest)
    # shap.dependence_plot("Feature 12", shap_values, xtest, interaction_index="Feature 11")

    # Plot
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    HeatmapDateTime(series=observed).show()
    HeatmapDateTime(series=gapfilled).show()

    print(xgbts.get_gapfilled_target().cumsum())
    print(df_orig['NEE_CUT_REF_f'].cumsum())

    # mds = df_orig['NEE_CUT_REF_f'].copy()
    # mds = mds[mds.index.year >= 2016]
    import matplotlib.pyplot as plt
    # # rfts.gapfilling_df_['.PREDICTIONS_FALLBACK'].cumsum().plot()
    # # rfts.gapfilling_df_['.PREDICTIONS_FULLMODEL'].cumsum().plot()
    # # rfts.gapfilling_df_['.PREDICTIONS'].cumsum().plot()
    xgbts.get_gapfilled_target().cumsum().plot()
    df_orig['NEE_CUT_REF_f'].cumsum().plot()
    # mds.cumsum().plot()
    # plt.legend()
    plt.show()

    # from diive.core.plotting.timeseries import TimeSeries  # For simple (interactive) time series plotting
    # TimeSeries(series=df[TARGET_COL]).plot()

    # d = rfts.gapfilling_df['NEE_CUT_REF_orig'] - rfts.gapfilling_df['.PREDICTIONS']
    # d.plot()
    # plt.show()
    # d = abs(d)
    # d.mean()  # MAE

    print("Finished.")


if __name__ == '__main__':
    example_xgbts()
