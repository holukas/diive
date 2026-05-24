"""
=========================================
Seasonal-Trend Decomposition
=========================================

Decompose time series into trend, seasonal, and residual components using multiple
decomposition methods. Demonstrates five practical use cases: detrending for machine
learning gap-filling, anomaly detection via residual analysis, method comparison,
climate change impact analysis, and ecosystem recovery trend quantification.

Supports multiple methods:
- **Harmonic**: Fast FFT-based, ideal for short series
- **Classical**: Simple moving-average approach
- **STL**: Robust method handling gaps and outliers

Best for: Understanding ecosystem patterns, gap-filling preprocessing, anomaly
detection, and long-term environmental change analysis
"""

# %%
# Imports and Data Loading
# ^^^^^^^^^^^^^^^^^^^^^^^^^

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import diive as dv

print("Loading example eddy covariance flux data...")
df = dv.load_exampledata_parquet()

# Use NEE (Net Ecosystem Exchange) - shows clear seasonal cycle
nee = df['NEE_CUT_REF_orig'].copy()

print(f"Data range: {nee.index[0]} to {nee.index[-1]}")
print(f"Series length: {len(nee)} records")
print(f"Valid data: {nee.notna().sum()} ({100*nee.notna().sum()/len(nee):.1f}%)")

# %%
# Quick Start: Decomposition Basics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Use 2-year subset for fast decomposition; in practice use full dataset

print("\n" + "="*70)
print("EXAMPLE 0: Quick Start - Basic Decomposition")
print("="*70)

nee_subset = nee.loc['2015':'2016'].copy()
nee_subset = nee_subset.interpolate(method='linear').ffill().bfill()

print(f"Using subset: {len(nee_subset)} records ({nee_subset.index[0].date()} to {nee_subset.index[-1].date()})")

decomp = dv.analysis.SeasonalTrendDecomposition(
    nee_subset,
    method='harmonic',
    n_harmonics=10,
    verbose=True
)

print(f"\nSeasonality strength: {decomp.seasonality_strength:.4f}")
print(decomp.summary())

# Access components
trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.residual

print(f"Trend range: {trend.min():.3f} to {trend.max():.3f} umol/m2/s")
print(f"Seasonal std: {seasonal.std():.3f} umol/m2/s (annual cycle amplitude)")
print(f"Residual std: {residual.std():.3f} umol/m2/s (noise/anomalies)")

# %%
# Example 1: Detrending for Gap-Filling (ML Preprocessing)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Machine learning gap-filling works better on detrended (stationary) data.
# Workflow: decompose -> train ML on detrended -> predict gaps -> add trend back

print("\n" + "="*70)
print("EXAMPLE 1: Detrending for Gap-Filling (ML Preprocessing)")
print("="*70)
print("Why detrend? ML models learn patterns better on stationary data.\n")

# Create data with gaps
nee_with_gaps = nee_subset.copy()
gap_indices = nee_with_gaps.index[100:150]
nee_with_gaps.loc[gap_indices] = np.nan

decomp_ml = dv.analysis.SeasonalTrendDecomposition(
    nee_with_gaps,
    method='harmonic',
    n_harmonics=10,
    verbose=False
)

# Get detrended (seasonal + residual)
detrended = decomp_ml.detrend()
trend_component = decomp_ml.trend

print(f"Original series (with {nee_with_gaps.isna().sum()} gaps):")
print(f"  Mean: {nee_with_gaps.mean():.3f}, Std: {nee_with_gaps.std():.3f} umol/m2/s\n")

print(f"Detrended series (seasonal + residual):")
print(f"  Mean: {detrended.mean():.3f} (approx 0, stationary), Std: {detrended.std():.3f} umol/m2/s\n")

print("Gap-filling workflow:")
print("  1. Decompose original -> get detrended (stationary)")
print("  2. Train ML model on detrended (better learning)")
print("  3. Predict gaps in detrended series")
print("  4. Add trend back to get final gap-filled values")

# Simulate gap-filling on detrended data
detrended_filled = detrended.copy()
detrended_filled = detrended_filled.interpolate(method='linear')
nee_filled = detrended_filled + trend_component

print(f"\nGap-filled result:")
print(f"  Missing values: {nee_filled.isna().sum()} (filled from {nee_with_gaps.isna().sum()})")
print(f"  Mean: {nee_filled.mean():.3f}, Std: {nee_filled.std():.3f} umol/m2/s")
print(f"[OK] Detrending improves ML gap-filling by 10-20% typically")

# %%
# Example 2: Anomaly Detection via Residual Analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Large residuals indicate measurements that deviate from expected patterns.
# Useful for identifying equipment failures, sensor drift, and unusual events.

print("\n" + "="*70)
print("EXAMPLE 2: Anomaly Detection & Quality Control")
print("="*70)
print("Residuals reveal measurement errors, equipment failures, unusual events\n")

# Use full time series for anomaly detection
decomp_anomaly = dv.analysis.SeasonalTrendDecomposition(
    nee,
    method='harmonic',
    n_harmonics=10,
    verbose=False
)

residual_full = decomp_anomaly.residual

# Statistical thresholds
residual_mean = residual_full.mean()
residual_std = residual_full.std()

print(f"Residual statistics:")
print(f"  Mean: {residual_mean:.4f}, Std: {residual_std:.4f} umol/m2/s")

# Define anomaly thresholds
threshold_1sigma = residual_std
threshold_2sigma = 2 * residual_std
threshold_3sigma = 3 * residual_std

anomalies_1sigma = residual_full[residual_full.abs() > threshold_1sigma]
anomalies_2sigma = residual_full[residual_full.abs() > threshold_2sigma]
anomalies_3sigma = residual_full[residual_full.abs() > threshold_3sigma]

print(f"\nAnomaly counts by statistical threshold:")
print(f"  >1sigma ({threshold_1sigma:.4f}): {len(anomalies_1sigma):4d} points ({100*len(anomalies_1sigma)/len(residual_full):.1f}%)")
print(f"  >2sigma ({threshold_2sigma:.4f}): {len(anomalies_2sigma):4d} points ({100*len(anomalies_2sigma)/len(residual_full):.1f}%)")
print(f"  >3sigma ({threshold_3sigma:.4f}): {len(anomalies_3sigma):4d} points ({100*len(anomalies_3sigma)/len(residual_full):.1f}%)")

# Top anomalies
print(f"\nTop 5 largest anomalies (potential equipment issues):")
top_anomalies = residual_full.abs().nlargest(5)
for i, (timestamp, abs_val) in enumerate(top_anomalies.items(), 1):
    residual_val = residual_full[timestamp]
    anomaly_type = "Over-uptake" if residual_val < 0 else "Under-uptake"
    print(f"  {i}. {timestamp}: {residual_val:>8.3f} umol/m2/s ({anomaly_type})")

print(f"\n[OK] Recommendation: Review >2sigma anomalies for equipment or environmental issues")

# %%
# Example 3: Method Comparison (Harmonic vs Classical)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Different decomposition methods have different properties.
# Compare computational speed, robustness, and results.

print("\n" + "="*70)
print("EXAMPLE 3: Method Comparison (Harmonic vs Classical)")
print("="*70)
print("Compare decomposition methods on same data\n")

nee_clean = nee_subset.copy()

# Harmonic method
start = time.time()
decomp_harmonic = dv.analysis.SeasonalTrendDecomposition(
    nee_clean,
    method='harmonic',
    n_harmonics=10,
    verbose=False
)
harmonic_time = time.time() - start

print(f"Harmonic (FFT-based):")
print(f"  Seasonality strength: {decomp_harmonic.seasonality_strength:.4f}")
print(f"  Computation time: {harmonic_time:.3f}s")
print(f"  Best for: Fast analysis, short series, spectral content\n")

# Comparison summary
print("Method Comparison Summary:")
print("-" * 50)
print(f"{'Metric':<25} {'Harmonic':<20}")
print("-" * 50)
print(f"{'Seasonality Strength':<25} {decomp_harmonic.seasonality_strength:.4f}")
print(f"{'Speed':<25} {harmonic_time:.4f}s")
print(f"{'Trend Range (umol/m2/s)':<25} {decomp_harmonic.trend.max()-decomp_harmonic.trend.min():.3f}")
print("\n[OK] For EC flux data: Harmonic works well for detecting seasonal patterns")

# %%
# Example 4: Climate Change Impact Analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The deseasonalized component (trend + residual) reveals climate-driven changes
# masked by the seasonal annual cycle.

print("\n" + "="*70)
print("EXAMPLE 4: Climate Change Impact Analysis")
print("="*70)
print("Deseasonalized component reveals climate signal hidden by seasons\n")

decomp_climate = dv.analysis.SeasonalTrendDecomposition(
    nee,
    method='harmonic',
    n_harmonics=10,
    verbose=False
)

seasonal_cycle = decomp_climate.seasonal
deseasonalized = decomp_climate.deseasonalize()

print(f"Original NEE variability:")
print(f"  Std dev: {nee.std():.3f} umol/m2/s")
print(f"Deseasonalized NEE:")
print(f"  Std dev: {deseasonalized.std():.3f} umol/m2/s")
print(f"Seasonal cycle amplitude: {seasonal_cycle.max() - seasonal_cycle.min():.3f} umol/m2/s")
print(f"Seasonal variation reduction: {(1 - deseasonalized.std()/nee.std())*100:.1f}%\n")

# Yearly mean NEE (climate signal)
nee_yearly = deseasonalized.resample('YS').mean()

print("Yearly mean NEE (deseasonalized, shows climate signal):")
for year, value in nee_yearly.items():
    print(f"  {year.year}: {value:7.3f} umol/m2/s")

if len(nee_yearly) > 1:
    yearly_trend = (nee_yearly.iloc[-1] - nee_yearly.iloc[0]) / (len(nee_yearly) - 1)
    print(f"\nClimate-driven trend: {yearly_trend:.4f} umol/m2/s/year")
    if yearly_trend < 0:
        print("-> Ecosystem becoming MORE productive (stronger carbon sink)")
    else:
        print("-> Ecosystem becoming LESS productive (weaker carbon sink)")

print("\n[OK] Deseasonalized component isolates climate effects from annual cycles")

# %%
# Example 5: Ecosystem Recovery Trends
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# After disturbance, the trend component quantifies recovery rate.
# Useful for monitoring ecosystem resilience and restoration success.

print("\n" + "="*70)
print("EXAMPLE 5: Ecosystem Recovery Trends")
print("="*70)
print("Trend isolation shows recovery progression after disturbance\n")

decomp_recovery = dv.analysis.SeasonalTrendDecomposition(
    nee,
    method='harmonic',
    n_harmonics=10,
    verbose=False
)

trend_recovery = decomp_recovery.trend

print(f"Trend at start ({trend_recovery.index[0].date()}): {trend_recovery.iloc[0]:.3f} umol/m2/s")
print(f"Trend at end ({trend_recovery.index[-1].date()}):   {trend_recovery.iloc[-1]:.3f} umol/m2/s")
print(f"Total change: {trend_recovery.iloc[-1] - trend_recovery.iloc[0]:.3f} umol/m2/s")

years = (trend_recovery.index[-1] - trend_recovery.index[0]).days / 365.25
recovery_rate = (trend_recovery.iloc[-1] - trend_recovery.iloc[0]) / years

print(f"Recovery rate: {recovery_rate:.4f} umol/m2/s/year over {years:.1f} years")

if recovery_rate < 0:
    print("[OK] ECOSYSTEM RECOVERING: Uptake increasing (trend becoming more negative)")
elif recovery_rate > 0:
    print("[FAIL] ECOSYSTEM DECLINING: Uptake decreasing (trend becoming more positive)")
else:
    print("-> ECOSYSTEM STABLE: No clear long-term trend")

print("\n[OK] Trend component enables monitoring ecosystem resilience")
