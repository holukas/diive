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
                 features_poly_degree: int = None, features_poly_exclude_cols: list = None,
                 vectorize_timestamps: bool = False, add_continuous_record_number: bool = False,
                 sanitize_timestamp: bool = False, **kwargs):
        """
        Gap-fill timeseries with predictions from XGBoost model

        Args:
            input_df:
                Contains timeseries of 1 target column and 1+ feature columns.

            target_col:
                Column name of variable in *input_df* that will be gap-filled.

            test_size:
                Proportion of the dataset to include in the test split,
                between 0.0 and 1.0.

            features_lag:
                List of integers (number of records), includes lagged variants of predictors.
                If features_lag=None, no lagged variants are added.
                Example:
                    - features_lag=[-2, +2] includes variants that are lagged by -2, -1, +1, and
                    +2 records in the dataset, for each feature already present in the data.
                     For a variable named *TA*, this created the following output:
                    TA    = [  5,   6,   7, 8  ]
                    TA-2  = [NaN, NaN,   5, 6  ]
                    TA-1  = [NaN,   5,   6, 7  ]  --> each TA record is paired with the preceding record TA-1
                    TA+1  = [  6,   7,   8, NaN]  --> each TA record is paired with the next record TA+1
                    TA+2  = [  7,   8, NaN, NaN]

            features_lag_exclude_cols:
                List of predictors for which no lagged variants are added.
                Example: with ['A', 'B'] no lagged variants for variables 'A' and 'B' are added.

            features_rolling:
                List of window sizes (in records) for rolling statistics.
                For each window size, rolling mean and rolling std are added for every
                feature column. If None, no rolling statistics are added.
                Example: features_rolling=[6, 48] with 30-min data adds 3-hour and 24-hour
                rolling mean and std for each driver variable.

            features_rolling_exclude_cols:
                List of column names excluded from rolling statistics.

            features_diff:
                List of difference orders for temporal momentum features.
                For each order, creates `.{col}_DIFF{order}` columns.
                Example: features_diff=[1, 2] creates 1st and 2nd order differences.
                If None, no differencing is applied.

            features_diff_exclude_cols:
                List of column names excluded from differencing.

            features_poly_degree:
                Polynomial degree for feature expansion (e.g., 2 for squared terms).
                If None, no polynomial features are added.

            features_poly_exclude_cols:
                List of column names excluded from polynomial expansion.

            vectorize_timestamps:
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
            test_size=test_size,
            features_lag=features_lag,
            features_lag_stepsize=features_lag_stepsize,
            features_lag_exclude_cols=features_lag_exclude_cols,
            features_rolling=features_rolling,
            features_rolling_exclude_cols=features_rolling_exclude_cols,
            features_rolling_stats=features_rolling_stats,
            features_diff=features_diff,
            features_diff_exclude_cols=features_diff_exclude_cols,
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
        features_rolling=[12, 24],
        features_rolling_exclude_cols=None,
        features_rolling_stats=['median', 'min', 'max', 'std', 'q25', 'q75'],
        features_diff=[1, 2],
        features_diff_exclude_cols=None,
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
