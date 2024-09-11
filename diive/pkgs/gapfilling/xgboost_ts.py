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

    def __init__(self, input_df: DataFrame, target_col: str or tuple, verbose: int = 0, perm_n_repeats: int = 3,
                 test_size: float = 0.25, features_lag: list = None, features_lag_exclude_cols: list = None,
                 include_timestamp_as_features: bool = False, add_continuous_record_number: bool = False,
                 sanitize_timestamp: bool = False, **kwargs):
        """
        Gap-fill timeseries with predictions from random forest model

        Args:
            input_df:
                Contains timeseries of 1 target column and 1+ feature columns.

            target_col:
                Column name of variable in *input_df* that will be gap-filled.

            perm_n_repeats:
                Number of repeats for calculating permutation feature importance.
                Must be greater than 0.

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

            features_lag_exclude_cols:
                List of predictors for which no lagged variants are added.
                Example: with ['A', 'B'] no lagged variants for variables 'A' and 'B' are added.

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
            features_lag_exclude_cols=features_lag_exclude_cols,
            include_timestamp_as_features=include_timestamp_as_features,
            add_continuous_record_number=add_continuous_record_number,
            sanitize_timestamp=sanitize_timestamp,
            **kwargs
        )


def example_xgbts():
    # Setup, user settings
    # TARGET_COL = 'LE_orig'
    # TARGET_COL = 'NEE_CUT_84_orig'
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']
    # subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f', 'PREC_TOT_T1_25+20_1']
    # subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f', 'SWC_FF0_0.15_1', 'PPFD']

    # Example data
    from diive.configs.exampledata import load_exampledata_parquet_long
    df_orig = load_exampledata_parquet_long()

    # # Create a large gap
    # remove = df.index.year != 2014
    # # df = df.drop(df.index[100:2200])
    # df = df[remove].copy()

    # Subset
    keep = (df_orig.index.year >= 1997) & (df_orig.index.year <= 2001)
    df = df_orig[keep].copy()
    # df = df_orig.copy()

    # Checking nighttime
    # nt_locs = df['Rg_f'] < 50
    # nt = df[nt_locs].groupby(df[nt_locs].index.year).agg(['mean'])
    # means_nt = nt[TARGET]['mean']
    # # import matplotlib.pyplot as plt
    # # means_nt.plot()
    # # plt.show()
    # mean_nt_9704 = means_nt.loc[1997:2004].mean()
    # mean_nt_0613 = means_nt.loc[2006:2013].mean()
    # corr_nt = mean_nt_0613 / mean_nt_9704

    # # corr_nt = 100
    # # corr_nt = 1.19
    # corr_nt = 0.7759670068746911
    # corr_df = df[['Rg_f', 'NEE_CUT_REF_orig']].copy()
    # corr_df['gain'] = 1
    # # nt_locs_9704 = (df.index.year >= 1997) & (df.index.year <= 2004)
    # nt_locs_9704 = (df.index.year >= 1997) & (df.index.year <= 2004) & (df['Rg_f'] < 50)
    # corr_df.loc[nt_locs_9704, 'gain'] = corr_nt
    # corr_df['NEE_CUT_REF_orig'] = corr_df['NEE_CUT_REF_orig'].multiply(corr_df['gain'])
    # df[TARGET_COL] = corr_df[TARGET_COL].copy()

    # df[nt_locs].groupby(df[nt_locs].index.year).agg(['mean'])['NEE_CUT_REF_orig']
    # df[~nt_locs].groupby(df[~nt_locs].index.year).agg(['mean'])['NEE_CUT_REF_orig']
    # df.loc[nt_locs_9704, TARGET_COL] = df.loc[nt_locs_9704, TARGET_COL].multiply(corr_nt)
    # df.loc[nt_locs_9704, TARGET_COL].describe()
    # df[TARGET_COL].describe()

    # # Checking daytime
    # dt = df['Rg_f'] > 50
    # dt = df[dt].groupby(df[dt].index.year).agg(['mean'])
    # means_dt = dt[TARGET]['mean']
    # import matplotlib.pyplot as plt
    # means_dt.plot()
    # plt.show()
    # mean_dt_9704 = means_dt.loc[1997:2004].mean()
    # mean_dt_0613 = means_dt.loc[2006:2013].mean()
    # corr_dt = mean_dt_0613 / mean_dt_9704

    nee_mds = df['NEE_CUT_REF_f'].copy()

    # Subset with target and features
    # Only High-quality (QCF=0) measured NEE used for model training in this example
    # lowquality = df["QCF_NEE"] > 0
    # df.loc[lowquality, TARGET_COL] = np.nan
    df = df[subsetcols].copy()

    # Calculate additional features

    # # TimeSince
    # from diive.pkgs.createvar.timesince import TimeSince
    # ts = TimeSince(df['PREC_TOT_T1_25+20_1'], upper_lim=None, lower_lim=0, include_lim=False)
    # ts.calc()
    # ts_full_results = ts.get_full_results()
    # df['TIMESINCE_PREC_TOT_T1_25+20_1'] = ts_full_results['TIMESINCE_PREC_TOT_T1_25+20_1'].copy()
    # df = df.drop('PREC_TOT_T1_25+20_1', axis=1)

    # from diive.pkgs.createvar.timesince import TimeSince
    # ts = TimeSince(df['Tair_f'], upper_lim=0, include_lim=True)
    # ts.calc()
    # ts_series = ts.get_timesince()
    # # xxx = ts.get_full_results()
    # df['TA>0'] = ts_series
    #
    # ts = TimeSince(df['Tair_f'], lower_lim=20, include_lim=True)
    # ts.calc()
    # ts_series = ts.get_timesince()
    # # xxx = ts.get_full_results()
    # df['TA>20'] = ts_series

    # from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag
    # dnf = DaytimeNighttimeFlag(
    #     timestamp_index=df.index,
    #     nighttime_threshold=50,
    #     lat=46.815333,
    #     lon=9.855972,
    #     utc_offset=1)
    # results = dnf.get_results()
    # df['DAYTIME'] = results['DAYTIME'].copy()
    # df['NIGHTTIME'] = results['NIGHTTIME'].copy()

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
        features_lag_exclude_cols=['TIMESINCE_PREC_TOT_T1_25+20_1'],
        # features_lag_exclude_cols=['Rg_f', 'TA>0', 'TA>20', 'DAYTIME', 'NIGHTTIME'],
        # include_timestamp_as_features=False,
        include_timestamp_as_features=True,
        add_continuous_record_number=False,
        # add_continuous_record_number=True,
        sanitize_timestamp=True,
        perm_n_repeats=3,
        n_estimators=200,
        # n_estimators=99,
        random_state=42,
        # booster='gbtree',  # gbtree (default), gblinear, dart
        # device='cpu',
        # validate_parameters=True,
        # disable_default_eval_metric=False,
        early_stopping_rounds=50,
        # learning_rate=0.1,
        # max_depth=9,
        # max_delta_step=0,
        # subsample=1,
        # min_split_loss=0,
        # min_child_weight=1,
        # colsample_bytree=1,
        # colsample_bylevel=1,
        # colsample_bynode=1,
        # reg_lambda=1,
        # reg_alpha=0,
        # tree_method='auto',  # auto, hist, approx, exact
        # scale_pos_weight=1,
        # grow_policy='depthwise',  # depthwise, lossguide
        # max_leaves=0,
        # max_bin=256,
        # num_parallel_tree=1,
        n_jobs=-1
    )
    xgbts.reduce_features()
    # xgbts.report_feature_reduction()

    # xgbts.trainmodel(showplot_scores=False, showplot_importance=False)
    # xgbts.report_traintest()
    #
    # xgbts.fillgaps(showplot_scores=False, showplot_importance=False)
    # xgbts.report_gapfilling()

    # observed = df[TARGET_COL]
    # gapfilled = xgbts.get_gapfilled_target()

    # frame = {
    #     nee_mds.name: nee_mds,
    #     gapfilled.name: gapfilled,
    # }
    # import pandas as pd
    # checkdf = pd.DataFrame.from_dict(frame, orient='columns')
    # checkdf = checkdf.groupby(checkdf.index.year).agg('sum')
    # checkdf['diff'] = checkdf[gapfilled.name].subtract(checkdf[nee_mds.name])
    # checkdf = checkdf.multiply(0.02161926)
    # print(checkdf)
    # print(checkdf.sum())

    # rfts.feature_importances
    # rfts.scores
    # rfts.gapfilling_df

    # # # https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
    # # import shap
    # # explainer = shap.TreeExplainer(rfts.model_)
    # # xtest = rfts.traintest_details_['X_test']
    # # shap_values = explainer.shap_values(xtest)
    # # shap.summary_plot(shap_values, xtest)
    # # # shap.summary_plot(shap_values[0], xtest)
    # # shap.dependence_plot("Feature 12", shap_values, xtest, interaction_index="Feature 11")

    # Plot
    # title = (
    #     f"N_ESTIMATORS: {N_ESTIMATORS} "
    #     f"/ MAX_DEPTH: {MAX_DEPTH} "
    #     f"/ CRITERION: {CRITERION} "
    #     f"\nMIN_SAMPLES_SPLIT: {MIN_SAMPLES_SPLIT} "
    #     f"/ MIN_SAMPLES_LEAF: {MIN_SAMPLES_LEAF}  "
    # )
    # title = "title"
    # from diive.core.plotting.timeseries import TimeSeries
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # TimeSeries(series=gapfilled.multiply(0.02161926).cumsum(), ax=ax).plot(color='blue')
    # TimeSeries(series=nee_mds.multiply(0.02161926).cumsum(), ax=ax).plot(color='orange')
    # fig.suptitle(f'{title}', fontsize=16)
    # # ax.set_ylim(-2000, 200)
    # fig.show()

    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=observed).show()
    # HeatmapDateTime(series=nee_mds).show()
    # HeatmapDateTime(series=gapfilled).show()
    # diff = gapfilled.subtract(nee_mds)
    # diff.name = "diff"
    # HeatmapDateTime(series=diff).show()

    # # mds = df_orig[TARGET].copy()
    # # mds = mds[mds.index.year >= 2016]
    # import matplotlib.pyplot as plt
    # # # rfts.gapfilling_df_['.PREDICTIONS_FALLBACK'].cumsum().plot()
    # # # rfts.gapfilling_df_['.PREDICTIONS_FULLMODEL'].cumsum().plot()
    # # # rfts.gapfilling_df_['.PREDICTIONS'].cumsum().plot()
    # rfts.get_gapfilled_target().cumsum().plot()
    # # mds.cumsum().plot()
    # # plt.legend()
    # plt.show()

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
    # CumulativeYear(
    #     series=nee_mds.multiply(0.02161926),
    #     series_units="units",
    #     yearly_end_date=None,
    #     # yearly_end_date='08-11',
    #     start_year=1997,
    #     end_year=2022,
    #     show_reference=True,
    #     excl_years_from_reference=None,
    #     # excl_years_from_reference=[2022],
    #     highlight_year=2022,
    #     highlight_year_color='#F44336').plot()

    # from diive.core.plotting.dielcycle import DielCycle
    # series = gapfilled.multiply(0.02161926).copy()
    # # for yr in [2004, 2006, 2015, 2022]:
    # for yr in range(1997, 2023):
    #     series1 = series.loc[series.index.year == yr].copy()
    #     dc = DielCycle(series=series1)
    #     dc.plot(ax=None, title=str(yr), txt_ylabel_units="units",
    #             each_month=True, legend_n_col=2, ylim=[-0.4, 0.2])
    #     # d = dc.get_data()

    from yellowbrick.model_selection import ValidationCurve, validation_curve
    import numpy as np
    # viz = ValidationCurve(
    #     xgbts.model_, param_name="max_depth",
    #     param_range=np.arange(3, 10), cv=10, scoring="r2"
    # )
    y = df[TARGET_COL]
    X = df[['Tair_f', 'VPD_f', 'Rg_f']]

    viz = validation_curve(
        xgbts.model_, X, y, param_name="n_estimators",
        param_range=np.arange(10, 20), cv=10, scoring="r2",
    )

    # Fit and show the visualizer
    viz.fit(X, y)
    viz.show()

    # from yellowbrick.datasets import load_energy
    # x,y = load_energy()


    print("Finished.")


if __name__ == '__main__':
    example_xgbts()
