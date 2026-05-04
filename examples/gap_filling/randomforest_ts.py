"""Random Forest gap-filling examples.

Random Forest is a robust, interpretable machine learning approach for gap-filling
time series data. Effective for non-linear relationships and feature interactions.
"""
import pandas as pd

import diive as dv


def example_randomforest_nee_gapfilling():
    """Random Forest gap-filling for CO₂ flux (NEE) with feature engineering.

    Demonstrates Random Forest gap-filling workflow:
    1. Load example ecosystem flux data
    2. Create engineered features (lag, rolling stats, etc.)
    3. Train RandomForestTS model on complete observations
    4. Predict missing values
    5. Evaluate feature importance using SHAP

    Features used: Tair_f (temperature), VPD_f (vapor pressure deficit), Rg_f (radiation)

    Returns:
        None (prints reports and completion message)
    """
    TARGET_COL = 'NEE_CUT_REF_orig'
    subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

    df_orig = dv.load_exampledata_parquet()
    df = df_orig.copy()
    keep = (df.index.year >= 2020) & (df.index.year <= 2020)
    df = df[keep].copy()
    df = df[subsetcols].copy()

    # Step 1: Engineer features (production-quality settings for CO2 flux)
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
    rfts = dv.RandomForestTS(
        input_df=df_engineered,
        target_col=TARGET_COL,
        verbose=1,
        n_estimators=3,
        random_state=42,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )

    # Feature reduction using SHAP importance
    rfts.reduce_features(shap_threshold_factor=0.5)
    rfts.report_feature_reduction()

    # Train model
    rfts.trainmodel(showplot_scores=False, showplot_importance=False)
    rfts.report_traintest()

    # Gap-fill data
    rfts.fillgaps(showplot_scores=False, showplot_importance=False)
    rfts.report_gapfilling()

    observed = df[TARGET_COL]
    gapfilled = rfts.get_gapfilled_target()

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


if __name__ == '__main__':
    example_randomforest_nee_gapfilling()
