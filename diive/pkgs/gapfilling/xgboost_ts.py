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
                 test_size: float = 0.25, features_lag: list = None, features_lag_stepsize: int = 1,
                 features_lag_exclude_cols: list = None,
                 features_rolling: list = None, features_rolling_exclude_cols: list = None,
                 features_rolling_stats: list = None,
                 features_diff: list = None, features_diff_exclude_cols: list = None,
                 features_ema: list = None, features_ema_exclude_cols: list = None,
                 features_poly_degree: int = None, features_poly_exclude_cols: list = None,
                 vectorize_timestamps: bool = False, add_continuous_record_number: bool = False,
                 sanitize_timestamp: bool = False, **kwargs):
        """
        XGBoost-based gap-filling for time series data.

        Trains an XGBoost gradient boosting model on complete (non-gap) observations to predict
        missing values in a target time series. Suitable for modeling non-linear relationships
        and complex temporal patterns. Supports comprehensive feature engineering with lagged
        variants, rolling statistics, temporal differencing, and polynomial expansion.

        XGBoost is particularly effective for:
        - Non-linear flux dynamics (radiation, temperature effects)
        - Complex temporal interactions between predictors
        - Data with outliers (more robust than Random Forest)
        - Settings where model interpretability is secondary to accuracy

        Workflow:
            1. Create instance with input data and feature engineering parameters
            2. Call trainmodel() to fit on training data and evaluate on test data
            3. Call fillgaps() to predict missing values and generate output
            4. Optional: Call reduce_features() before fillgaps() for feature selection

        Args:
            input_df:
                DataFrame with time series data. Must contain 1 target column and 1+ feature
                columns. Timestamps should be in DataFrame index (DatetimeIndex).

            target_col:
                Column name of variable to gap-fill (string or tuple for multi-level columns).

            verbose:
                Verbosity level: 0=silent, 1=progress updates, 2+=detailed output.

            test_size:
                Proportion of complete data for testing (0.0-1.0). Default: 0.25.
                Only complete (non-gap) rows are used for train/test split.

            features_lag:
                List [min_lag, max_lag] specifying lag range. Creates lags at all integers
                between min_lag and max_lag (excluding 0). Default: None.
                Example: features_lag=[-2, 2] creates lags [-2, -1, +1, +2].

            features_lag_stepsize:
                Step size for lag generation (e.g., 2 creates every 2nd lag). Default: 1.

            features_lag_exclude_cols:
                Column names to exclude from lagging. Default: None.

            features_rolling:
                List of window sizes (records) for rolling statistics. Each window computes
                rolling mean and std. Default: None.
                Example: features_rolling=[6, 48] with 30-min data adds 3-hour and 24-hour
                rolling statistics.

            features_rolling_exclude_cols:
                Columns excluded from rolling statistics. Default: None.

            features_rolling_stats:
                Advanced rolling statistics: 'median', 'min', 'max', 'std', 'q25', 'q75'.
                Default: None (only mean and std computed if features_rolling specified).

            features_diff:
                List of difference orders for temporal momentum. Default: None.
                Example: features_diff=[1, 2] creates 1st and 2nd order differences.

            features_diff_exclude_cols:
                Columns excluded from differencing. Default: None.

            features_ema:
                List of span values for exponential moving average. Default: None.
                Example: features_ema=[6, 24, 48] with 30-min data adds 3h, 12h, 24h EMAs.

            features_ema_exclude_cols:
                Columns excluded from EMA computation. Default: None.

            features_poly_degree:
                Polynomial degree for non-linear expansion (2=squared, 3=cubed). Default: None.
                Example: features_poly_degree=2 creates squared terms.

            features_poly_exclude_cols:
                Columns excluded from polynomial expansion. Default: None.

            vectorize_timestamps:
                Add timestamp features (year, season, month, week, doy, hour). Default: False.

            add_continuous_record_number:
                Add sequential record numbering (1, 2, 3, ...). Default: False.

            sanitize_timestamp:
                Validate and prepare timestamps. Default: False.

            **kwargs:
                XGBoost hyperparameters. Common settings:
                - n_estimators: Number of boosting rounds (default ~100)
                - max_depth: Tree depth (default 6, range 3-10)
                - learning_rate: Step shrinkage (default 0.3, range 0.01-1)
                - early_stopping_rounds: Stop if no improvement after N rounds (default 10)
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
            >>> xgbts = XGBoostTS(
            ...     input_df=df,
            ...     target_col='NEE',
            ...     verbose=1,
            ...     features_lag=[-1, -1],
            ...     features_rolling=[12, 24],
            ...     features_rolling_stats=['median', 'min', 'max'],
            ...     features_diff=[1],
            ...     features_poly_degree=2,
            ...     vectorize_timestamps=True,
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

        # Args
        super().__init__(
            regressor=xgb.XGBRegressor,
            input_df=input_df,
            target_col=target_col,
            verbose=verbose,
            test_size=test_size,
            features_lag=features_lag,
            features_lag_stepsize=features_lag_stepsize,
            features_lag_exclude_cols=features_lag_exclude_cols,
            features_rolling=features_rolling,
            features_rolling_exclude_cols=features_rolling_exclude_cols,
            features_rolling_stats=features_rolling_stats,
            features_diff=features_diff,
            features_diff_exclude_cols=features_diff_exclude_cols,
            features_ema=features_ema,
            features_ema_exclude_cols=features_ema_exclude_cols,
            features_poly_degree=features_poly_degree,
            features_poly_exclude_cols=features_poly_exclude_cols,
            vectorize_timestamps=vectorize_timestamps,
            add_continuous_record_number=add_continuous_record_number,
            sanitize_timestamp=sanitize_timestamp,
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

    # XGBoost with advanced feature engineering
    xgbts = XGBoostTS(
        input_df=df,
        target_col=TARGET_COL,
        verbose=1,
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
        vectorize_timestamps=True,
        add_continuous_record_number=True,
        sanitize_timestamp=True,
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
