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
import xgboost as xgb
from pandas import DataFrame

from diive.core.ml.common import MlRegressorGapFillingBase


class XGBoostTS(MlRegressorGapFillingBase):

    def __init__(self, input_df: DataFrame, target_col: str or tuple, verbose: int = 0,
                 test_size: float = 0.25, **kwargs):
        """
        XGBoost-based gap-filling for time series data.

        Trains an XGBoost gradient boosting model on complete (non-gap) observations to predict
        missing values in a target time series. Suitable for modeling non-linear relationships
        and complex temporal patterns.

        **IMPORTANT:** This class expects pre-engineered features. Use FeatureEngineer to
        create features before passing data to XGBoostTS. See example below.

        XGBoost is particularly effective for:
        - Non-linear flux dynamics (radiation, temperature effects)
        - Complex temporal interactions between predictors
        - Data with outliers (more robust than Random Forest)
        - Settings where model interpretability is secondary to accuracy

        Workflow:
            1. Use FeatureEngineer to create engineered features
            2. Create XGBoostTS instance with pre-engineered data
            3. Call trainmodel() to fit on training data and evaluate on test data
            4. Call fillgaps() to predict missing values and generate output
            5. Optional: Call reduce_features() before fillgaps() for feature selection

        Args:
            input_df:
                DataFrame with time series data. Must contain 1 target column and 1+ feature
                columns. **Features should be pre-engineered using FeatureEngineer.**
                Timestamps should be in DataFrame index (DatetimeIndex).

            target_col:
                Column name of variable to gap-fill (string or tuple for multi-level columns).

            verbose:
                Verbosity level: 0=silent, 1=progress updates, 2+=detailed output.

            test_size:
                Proportion of complete data for testing (0.0-1.0). Default: 0.25.
                Only complete (non-gap) rows are used for train/test split.

            **kwargs:
                XGBoost hyperparameters. Common settings:
                - n_estimators: Number of boosting rounds (default ~100)
                  Increase if underfitting, decrease if overfitting or memory-limited.
                - max_depth: Tree depth (default 6, range 3-10)
                  Increase for complex patterns, decrease if overfitting.
                - learning_rate: Step shrinkage (default 0.3, range 0.01-1)
                  Lower = slower learning but better generalization, higher = faster but less stable.
                - min_child_weight: Minimum sum of instance weights needed in a child node.
                  Default: 1 (permissive, allows many splits)
                  Range: 1-10 typical. For time series flux data:
                    - 1: Default, fine-grained features (good for exploration)
                    - 3-5: Moderate regularization (prevents overfitting to noise)
                    - 10+: Heavy regularization (smooth predictions, fewer splits)
                  Effect: Higher values create shallower trees, reduce overfitting but may
                    underfit if too restrictive. Essential for noisy flux data.
                - subsample: Fraction of samples used per tree (default 1.0)
                  Range 0.5-1.0. Lower values add randomness, reduce overfitting.
                - colsample_bytree: Fraction of features used per tree (default 1.0)
                  Range 0.5-1.0. Lower values add diversity, reduce overfitting.
                - early_stopping_rounds: Stop if no improvement after N rounds (default 10)
                  Higher values allow more exploration, may overfit.
                - random_state: Random seed for reproducibility
                See: https://xgboost.readthedocs.io/en/stable/parameter.html

        Methods:
            trainmodel(showplot_scores=False, showplot_importance=False)
                Train on training data, evaluate on test data, compute SHAP importances.

            fillgaps(showplot_scores=False, showplot_importance=False)
                Train on all complete data, predict all missing values, generate output.

            reduce_features(shap_threshold_factor=0.5)
                Feature selection based on SHAP importance. Call before trainmodel().

            report_traintest()
                Print model evaluation metrics and details.

            report_gapfilling()
                Print gap-filling results and statistics.

            get_gapfilled_target() -> Series
                Return gap-filled target time series.

            get_flag() -> Series
                Return gap-filling flags (0=observed, 1=gap-filled, 2=fallback).

        Attributes:
            model_: Trained XGBRegressor instance.
            gapfilling_df_: DataFrame with gap-filled target and auxiliary variables.
            feature_importances_: SHAP feature importance from gap-filling model.
            scores_: Model performance metrics (MAE, RMSE, R²) for gap-filling.

        Example:
            >>> from diive.core.ml.feature_engineer import FeatureEngineer
            >>> # Step 1: Create and apply feature engineer
            >>> engineer = FeatureEngineer(
            ...     target_col='NEE',
            ...     features_lag=[-1, -1],
            ...     features_rolling=[12, 24],
            ...     features_rolling_stats=['median', 'min', 'max'],
            ...     features_diff=[1],
            ...     features_poly_degree=2,
            ...     vectorize_timestamps=True
            ... )
            >>> df_engineered = engineer.fit_transform(df)
            >>> # Step 2: Create gap-filling model with engineered features
            >>> xgbts = XGBoostTS(
            ...     input_df=df_engineered,
            ...     target_col='NEE',
            ...     verbose=1,
            ...     n_estimators=100,
            ...     max_depth=6,
            ...     learning_rate=0.1,
            ...     early_stopping_rounds=10,
            ...     random_state=42,
            ... )
            >>> xgbts.trainmodel(showplot_scores=True, showplot_importance=True)
            >>> xgbts.fillgaps()
            >>> gapfilled = xgbts.get_gapfilled_target()
        """

        # Pass to parent class
        super().__init__(
            regressor=xgb.XGBRegressor,
            input_df=input_df,
            target_col=target_col,
            verbose=verbose,
            test_size=test_size,
            **kwargs
        )


def _example_xgbts():
    """
    Kudos:
        https://xgboost.readthedocs.io/en/stable/parameter.html
        https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
        https://medium.com/data-design/let-me-learn-the-learning-rate-eta-in-xgboost-d9ad6ec78363
    """
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']
    from diive.configs.exampledata import load_exampledata_parquet_long
    from diive.core.ml.feature_engineer import FeatureEngineer

    df_orig = load_exampledata_parquet_long()
    df = df_orig.copy()
    keep = (df.index.year >= 2013) & (df.index.year <= 2015)
    df = df[keep].copy()
    df = df[subsetcols].copy()

    # # TimeSince
    # from diive.pkgs.createvar.timesince import TimeSince
    # ts = TimeSince(df['PREC_TOT_T1_25+20_1'], upper_lim=None, lower_lim=0, include_lim=False)
    # ts.calc()
    # ts_full_results = ts.get_full_results()
    # df['TIMESINCE_PREC_TOT_T1_25+20_1'] = ts_full_results['TIMESINCE_PREC_TOT_T1_25+20_1'].copy()
    # df = df.drop('PREC_TOT_T1_25+20_1', axis=1)

    # Step 1: Engineer features with advanced configuration
    engineer = FeatureEngineer(
        target_col=TARGET_COL,
        features_lag=[-1, -1],
        features_lag_stepsize=1,
        features_lag_exclude_cols=None,
        features_rolling=[6, 12],
        features_rolling_exclude_cols=None,
        features_rolling_stats=['median', 'min', 'max', 'std', 'q25', 'q75'],
        features_diff=[1, 2],
        features_diff_exclude_cols=None,
        features_ema=[6, 12],
        features_ema_exclude_cols=None,
        features_poly_degree=2,
        features_poly_exclude_cols=None,
        features_stl=True,
        features_stl_method='harmonic',
        features_stl_seasonal_period=48,
        features_stl_exclude_cols=None,
        features_stl_components=['trend', 'seasonal'],
        vectorize_timestamps=True,
        add_continuous_record_number=True,
        sanitize_timestamp=True
    )
    df_engineered = engineer.fit_transform(df)

    # Step 2: Create XGBoost gap-filling model with engineered features
    xgbts = XGBoostTS(
        input_df=df_engineered,
        target_col=TARGET_COL,
        verbose=1,
        n_estimators=33,
        random_state=42,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    xgbts.reduce_features(shap_threshold_factor=0.5)
    xgbts.report_feature_reduction()

    xgbts.trainmodel(showplot_scores=False, showplot_importance=False)
    xgbts.report_traintest()

    xgbts.fillgaps(showplot_scores=True, showplot_importance=True)
    xgbts.report_gapfilling()

    observed = df[TARGET_COL]
    gapfilled = xgbts.get_gapfilled_target()

    # print(xgbts.feature_importances_)
    # print(xgbts.scores_)
    # print(xgbts.gapfilling_df_)

    # # Plot
    # from diive.core.plotting.timeseries import TimeSeries
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # TimeSeries(series=gapfilled.multiply(0.02161926).cumsum(), ax=ax).plot(color='blue')
    # fig.suptitle('XGB', fontsize=16)
    # # ax.set_ylim(-2000, 200)
    # fig.show()

    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=observed).show()
    # HeatmapDateTime(series=gapfilled).show()

    # from diive.core.plotting.cumulative import CumulativeYear
    # CumulativeYear(
    #     series=gapfilled.multiply(0.02161926),
    #     series_units="units",
    #     yearly_end_date=None,
    #     # yearly_end_date='08-11',
    #     start_year=1997,
    #     end_year=2022,
    #     show_reference=True,
    #     excl_years_from_reference=None,
    #     # excl_years_from_reference=[2022],
    #     # highlight_year=2022,
    #     highlight_year_color='#F44336').plot(digits_after_comma=0)

    # from diive.core.plotting.dielcycle import DielCycle
    # series = gapfilled.multiply(0.02161926).copy()
    # # for yr in [2004, 2006, 2015, 2022]:
    # for yr in range(2013, 2015):
    #     series1 = series.loc[series.index.year == yr].copy()
    #     dc = DielCycle(series=series1)
    #     dc.plot(ax=None, title=str(yr), txt_ylabel_units="units",
    #             each_month=True, legend_n_col=2, ylim=[-0.4, 0.2])
    #     # d = dc.get_data()

    print("Finished.")


if __name__ == '__main__':
    _example_xgbts()
