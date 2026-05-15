"""
TEMPORAL FEATURE ENGINEERING: ML-READY TIME FEATURES
====================================================

Extract temporal features (year, month, hour) with sin/cos encoding for machine learning models.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^
#
# Use meteorological data with DatetimeIndex.

import pandas as pd
import diive as dv

df = dv.load_exampledata_parquet()
series = df['Tair_f'].copy()

print("Example data loaded:")
print(f"  Records: {len(series)}")
print(f"  Period: {series.index[0]} to {series.index[-1]}")

# %%
# Create all temporal features
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Extract year, season, month, week, day-of-year, and hour.
# Uses sin/cos encoding to preserve cyclical nature.

df_features = pd.DataFrame({'temperature': series})

# Vectorize all temporal components
from diive.core.ml.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(
    target_col='temperature',
    features_stl=False,
    features_poly_degree=0,
    features_rolling=[],
    features_diff=[],
    features_ema=[],
    features_lag=[],
    vectorize_timestamps=True,  # Add temporal features
)

df_engineered = engineer.fit_transform(df_features)

print(f"\nOriginal columns: {df_features.shape[1]}")
print(f"After temporal features: {df_engineered.shape[1]}")
print(f"Temporal features added: {df_engineered.shape[1] - df_features.shape[1]}")

print("\nFirst 5 rows of temporal features:")
print(df_engineered.iloc[:5, 1:15])  # Show first 14 temporal feature columns

# %%
# Understanding temporal feature encoding
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each temporal component is encoded as **sin/cos pair**:
# - cos(2π × value/max) — captures proximity within cycle
# - sin(2π × value/max) — orthogonal component
#
# This preserves cyclical nature (Dec→Jan proximity) for ML models.
#
# Example: Month encoding creates 2 features (cos/sin):
# - Dec (month 12) and Jan (month 1) are close in the cycle
# - sin/cos pair maintains this geometric proximity
#
# Features created (19 total):
# - year, year_sin, year_cos (3)
# - season, season_sin, season_cos (3)
# - month, month_sin, month_cos (3)
# - week, week_sin, week_cos (3)
# - doy (day-of-year), doy_sin, doy_cos (3)
# - hour, hour_sin, hour_cos (3)
# - records (continuous record number, 1)

temporal_cols = [col for col in df_engineered.columns if col != 'temperature']
print("\nTemporal features created:")
for col in temporal_cols[:10]:
    sample_vals = df_engineered[col].iloc[:3].values
    print(f"  {col:<20} sample: {sample_vals}")

# %%
# Use case: Prepare data for machine learning
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Temporal features are essential for ML models to learn diurnal/seasonal patterns.

print("\n" + "="*60)
print("Feature engineering for gap-filling models")
print("="*60)

# Create features with multiple engineering stages
engineer_ml = FeatureEngineer(
    target_col='temperature',
    features_lag=[-1, 1],              # Previous/next value
    features_rolling=[12, 24],         # 6-hour, 12-hour rolling stats
    features_rolling_stats=['mean', 'std', 'min', 'max'],
    features_diff=[1],                 # Rate of change
    features_ema=[6, 12],              # Exponential moving averages
    features_poly_degree=2,            # Squared terms
    vectorize_timestamps=True,         # Temporal features
)

df_ml = engineer_ml.fit_transform(df_features)

print(f"Final feature count: {df_ml.shape[1]}")
print(f"  Original: 1 (temperature)")
print(f"  Lag features: 2")
print(f"  Rolling features: 8 (2 windows × 4 stats)")
print(f"  Differencing: 1")
print(f"  EMA features: 2")
print(f"  Polynomial: 1")
print(f"  Temporal features: 19")
print(f"  Record number: 1")

# Show correlation with target
print("\nTop 10 features (by correlation with temperature):")
correlations = df_ml.corr()['temperature'].abs().sort_values(ascending=False)
for feat, corr in correlations.head(11).items():
    if feat != 'temperature':
        print(f"  {feat:<25} r = {corr:.3f}")

# %%
# Use case: Selective temporal features
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sometimes only certain temporal components are needed.

df_simple = pd.DataFrame({'temperature': series})

# Manually add just hour and day-of-week for specific use case
df_simple['hour'] = df_simple.index.hour
df_simple['hour_sin'] = dv.np.sin(2 * dv.np.pi * df_simple['hour'] / 24)
df_simple['hour_cos'] = dv.np.cos(2 * dv.np.pi * df_simple['hour'] / 24)
df_simple['doy'] = df_simple.index.dayofyear
df_simple['doy_sin'] = dv.np.sin(2 * dv.np.pi * df_simple['doy'] / 365.25)
df_simple['doy_cos'] = dv.np.cos(2 * dv.np.pi * df_simple['doy'] / 365.25)

print("\n" + "="*60)
print("Selective temporal features (diurnal + seasonal)")
print("="*60)
print(f"Created {df_simple.shape[1]-1} selective features")
print(f"Sample:\n{df_simple.head()}")

# %%
# Visualize cyclical encoding benefit
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sin/cos encoding preserves proximity at cycle boundaries.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Linear month encoding (bad - Dec and Jan are far apart)
months = range(1, 13)
ax = axes[0]
ax.scatter(months, months, s=100, alpha=0.6, color='#d62728')
ax.plot([1, 12], [1, 12], 'r--', alpha=0.3)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Month (linear)', fontsize=11)
ax.set_title('Bad: Linear encoding\n(Dec-Jan distance = 11)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 13)
ax.set_ylim(0, 13)

# Sin/cos month encoding (good - Dec and Jan are close)
month_sin = [dv.np.sin(2 * dv.np.pi * m / 12) for m in months]
month_cos = [dv.np.cos(2 * dv.np.pi * m / 12) for m in months]
ax = axes[1]
ax.scatter(month_cos, month_sin, s=100, alpha=0.6, color='#2ca02c')
# Highlight Dec (month 12) and Jan (month 1)
ax.scatter([month_cos[11]], [month_sin[11]], s=200, color='#d62728', marker='s', label='Dec', edgecolor='black', linewidths=2)
ax.scatter([month_cos[0]], [month_sin[0]], s=200, color='#1f77b4', marker='s', label='Jan', edgecolor='black', linewidths=2)
ax.plot([month_cos[11], month_cos[0]], [month_sin[11], month_sin[0]], 'k--', linewidth=2, alpha=0.5)
ax.set_xlabel('cos(month)', fontsize=11)
ax.set_ylabel('sin(month)', fontsize=11)
ax.set_title('Good: Sin/cos encoding\n(Dec-Jan distance preserved)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.show()

print("Sin/cos encoding places cyclically-adjacent months close together geometrically.")
print("This helps ML models learn seasonal patterns without artificial discontinuity at year boundary.")
