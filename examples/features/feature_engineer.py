"""
FEATURE ENGINEERING: 8-STAGE PIPELINE
=====================================

Use FeatureEngineer to create ML-ready features: temporal encoding, lag, rolling stats,
differencing, EMA, polynomial terms, STL decomposition, and record numbering.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

import diive as dv

df_orig = dv.load_exampledata_parquet()

# Use 2020 data only for faster processing
keep = (df_orig.index.year == 2020)
df_data = df_orig[keep][['Tair_f', 'VPD_f', 'Rg_f']].copy()
df_data.columns = ['temperature', 'vapor_pressure_deficit', 'radiation']

print("Example data:")
print(f"  Records: {len(df_data)}")
print(f"  Period: {df_data.index.min().date()} to {df_data.index.max().date()}")
print(f"  Columns: {list(df_data.columns)}")

# %%
# Full 8-stage pipeline
# ^^^^^^^^^^^^^^^^^^^^^
#
# Create ML-ready features by combining all 8 stages in one pass.

from diive.core.ml.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(
    target_col='temperature',
    # Stage 1: Temporal features
    vectorize_timestamps=True,
    # Stage 2: Lag features
    features_lag=[-2, -1],
    # Stage 3: Rolling stats
    features_rolling=[12, 24],  # 6-hour, 12-hour windows
    features_rolling_stats=['mean', 'std', 'min', 'max'],
    # Stage 4: Differencing
    features_diff=[1],
    # Stage 5: EMA
    features_ema=[6, 12],
    # Stage 6: Polynomial
    features_poly_degree=2,
    # Stage 7: STL decomposition (optional)
    features_stl=False,
)

df_full = engineer.fit_transform(df_data)

print(f"\nFeature engineering complete:")
print(f"  Input: 3 columns (temperature, vapor_pressure_deficit, radiation)")
print(f"  Output: {df_full.shape[1]} columns")
print(f"  Features created: {df_full.shape[1] - 3}")

# Show first few rows
print(f"\nFirst 3 rows (sample columns):")
cols_to_show = ['temperature', 'vapor_pressure_deficit', 'radiation',
                '.vapor_pressure_deficit-1', '.radiation_MEAN12', '.temperature_EMA6']
cols_available = [c for c in cols_to_show if c in df_full.columns]
print(df_full.iloc[:3][cols_available])

# %%
# All generated features
# ^^^^^^^^^^^^^^^^^^^^^^
#
# List all engineered features (columns in df_full not in df_data).

generated_features = [c for c in df_full.columns if c not in df_data.columns]

print("\n" + "=" * 70)
print(f"Generated Features ({len(generated_features)})")
print("=" * 70)

for i, feat in enumerate(sorted(generated_features), 1):
    print(f"{i:2d}. {feat}")

# %%
# Feature importance
# ^^^^^^^^^^^^^^^^^^
#
# Correlation with target shows which features matter most.

print("\n" + "=" * 60)
print("Feature importance (correlation with temperature)")
print("=" * 60)

correlations = df_full.corr()['temperature'].abs().sort_values(ascending=False)
for i, (feat, corr) in enumerate(correlations.head(12).items(), 1):
    if feat != 'temperature':
        print(f"{i:2}. {feat:<30} r = {corr:.3f}")
