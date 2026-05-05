"""XGBoost gap-filling examples.

XGBoost is a gradient boosting approach for gap-filling time series data.
Effective for non-linear relationships, complex temporal interactions, and data with outliers.
"""
import matplotlib.pyplot as plt
import pandas as pd

import diive as dv


def example_xgboost_nee_gapfilling():
    """XGBoost gap-filling for CO2 flux (NEE) with feature engineering.

    Demonstrates XGBoost gap-filling workflow:
    1. Load example ecosystem flux data (one year)
    2. Create engineered features (lag, rolling stats, STL, timestamps)
    3. Train XGBoostTS model on complete observations
    4. Predict missing values with feature reduction
    5. Evaluate feature importance using SHAP

    Features used: Tair_f (temperature), VPD_f (vapor pressure deficit), Rg_f (radiation)

    Returns:
        None (prints reports and displays plots)
    """
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    df_orig = dv.load_exampledata_parquet()
    df = df_orig.copy()
    keep = (df.index.year >= 2020) & (df.index.year <= 2020)
    df = df[keep].copy()
    df = df[subsetcols].copy()

    # Step 1: Engineer features (harmonized with Random Forest example)
    engineer = dv.FeatureEngineer(
        target_col=TARGET_COL,
        features_lag=[-2, -1],
        features_lag_stepsize=1,
        features_lag_exclude_cols=None,
        features_rolling=[2, 4, 12, 24, 48],
        features_rolling_exclude_cols=None,
        features_rolling_stats=['mean', 'median', 'min', 'max'],
        features_diff=[1, 2],
        features_diff_exclude_cols=None,
        features_ema=[6, 12, 24, 48],
        features_ema_exclude_cols=None,
        features_poly_degree=2,
        features_poly_exclude_cols=None,
        features_stl=True,
        features_stl_method='stl',
        features_stl_seasonal_period=None,
        features_stl_exclude_cols=None,
        features_stl_components=None,
        vectorize_timestamps=True,
        add_continuous_record_number=True,
        sanitize_timestamp=False
    )
    df_engineered = engineer.fit_transform(df)

    # Step 2: Create gap-filling model with engineered features
    xgbts = dv.XGBoostTS(
        input_df=df_engineered,
        target_col=TARGET_COL,
        verbose=1,
        n_estimators=50,
        random_state=42,
        max_depth=6,
        learning_rate=0.1,
        early_stopping_rounds=10,
        n_jobs=-1
    )

    # Feature reduction using SHAP importance
    xgbts.reduce_features(shap_threshold_factor=0.5)
    xgbts.report_feature_reduction()

    # Train model
    xgbts.trainmodel(showplot_scores=False, showplot_importance=False)
    xgbts.report_traintest()

    # Gap-fill data
    xgbts.fillgaps(showplot_scores=False, showplot_importance=False)
    xgbts.report_gapfilling()

    observed = df[TARGET_COL]
    gapfilled = xgbts.get_gapfilled_target()

    # Heatmap visualization: observed vs gap-filled
    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             gridspec_kw={'wspace': 0.15},
                             constrained_layout=True)

    dv.plot_heatmap_datetime(ax=axes[0], series=observed).plot()
    axes[0].set_title('Observed\n(with gaps)', fontsize=11, fontweight='bold')

    dv.plot_heatmap_datetime(ax=axes[1], series=gapfilled).plot()
    axes[1].set_title('XGBoost\nGap-Filled', fontsize=11, fontweight='bold')

    fig.suptitle('XGBoost Gap-Filling Comparison', fontsize=13, fontweight='bold', y=1.00)
    plt.show()

    # Plot cumulative carbon flux: observed vs gap-filled
    df_cumulative = pd.DataFrame({
        'Observed': observed,
        'Gap-filled': gapfilled
    })
    # Convert from umol CO2 m-2 s-1 to g C m-2 30min-1
    df_cumulative = df_cumulative.multiply(0.02161926)
    series_units = r'($\mathrm{gC\ m^{-2}}$)'

    dv.plot_cumulative(
        df=df_cumulative,
        units=series_units,
        start_year=2020,
        end_year=2020
    ).plot()

    print("Finished.")


def example_xgboost_hyperparameter_optimization():
    """Hyperparameter optimization for XGBoost gap-filling.

    Demonstrates OptimizeParamsTS for tuning XGBoost hyperparameters
    using GridSearchCV with time series cross-validation. Tests multiple
    parameter combinations to find optimal settings for gap-filling accuracy.

    Uses 2020 data only for faster optimization testing.

    Returns:
        None (prints best parameters, R² score, and cross-validation results)
    """
    import xgboost as xgb

    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    # Example data
    df = dv.load_exampledata_parquet()
    subset = df[subsetcols].copy()
    _subset = df.index.year >= 2020
    subset = subset[_subset].copy()

    # XGBoost parameters (minimal for speed: ~20 combinations × 10 CV = 200 fits)
    xgb_params = {
        'n_estimators': [30, 50],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    }

    # Optimization
    opt = dv.OptimizeParamsTS(
        df=subset,
        target_col=TARGET_COL,
        regressor_class=xgb.XGBRegressor,
        **xgb_params
    )

    opt.optimize()

    # Print comprehensive optimization report with recommendations
    opt.report_optimization(top_n=3)


if __name__ == '__main__':
    example_xgboost_nee_gapfilling()
    example_xgboost_hyperparameter_optimization()
