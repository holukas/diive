"""
=========================================
Seasonal-Trend Decomposition
=========================================

Decompose time series into trend, seasonal, and residual components.

Demonstrates seasonal-trend decomposition using fast harmonic (FFT-based)
method on multi-year flux data. Shows how to access decomposition components,
analyze seasonality strength, and interpret ecosystem patterns (annual cycles,
long-term trends, anomalies).

Best for: Understanding trend and seasonal components in flux data
"""

# %%
# Imports
# ^^^^^^^

import numpy as np
import pandas as pd

import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

print("Loading example NEE data...")
df = dv.load_exampledata_parquet()

# Use NEE data (flux data with annual cycles)
# For faster example, use a 2-year subset and interpolate missing values
nee_series = df['NEE_CUT_REF_orig'].loc['2015':'2016'].copy()
nee_series = nee_series.interpolate(method='linear', limit_direction='both').bfill().ffill()

print(f"Series length: {len(nee_series)} records")
print(f"Data range: {nee_series.index[0]} to {nee_series.index[-1]}")
print(f"Valid data after interpolation: {nee_series.notna().sum()}")

# %%
# Create quality flags
# ^^^^^^^^^^^^^^^^^^^

# Quality flags (assuming all data is good quality for this example)
quality_flags = pd.Series(
    0.8 * np.ones(len(nee_series)), index=nee_series.index
)

# %%
# Decompose with harmonic method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print("\nDecomposing with harmonic method (fast FFT-based)...")
decomp = dv.SeasonalTrendDecomposition(
    nee_series,
    quality=quality_flags,
    method='harmonic',
    n_harmonics=10,
    verbose=True
)

# %%
# Results and interpretation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

print("\nDecomposition results:")
print(decomp.summary())

print(f"\nSeasonality strength: {decomp.seasonality_strength:.4f}")
print(f"Trend range: {decomp.trend.min():.3f} to {decomp.trend.max():.3f}")
print(f"Seasonal amplitude (std): {decomp.seasonal.std():.3f}")
print(f"Residual std: {decomp.residual.std():.3f}")

# %%
# Component visualization
# ^^^^^^^^^^^^^^^^^^^^^^

print("\n" + "=" * 60)
print("DECOMPOSITION COMPONENTS")
print("=" * 60)

# Show sample of each component
print("\nTrend component (long-term pattern):")
print(decomp.trend.iloc[:10])

print("\nSeasonal component (annual cycle):")
print(decomp.seasonal.iloc[:10])

print("\nResidual component (unexplained variation):")
print(decomp.residual.iloc[:10])
