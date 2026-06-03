"""
===========================
Increment-Based Detection
===========================

Identify spikes and abrupt changes by detecting anomalous increments
between consecutive measurements.
"""

# %%
# Create test data with impulse noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate summer temperature data and add synthetic spikes.

import warnings
import diive as dv

warnings.filterwarnings('ignore')

df = dv.load_exampledata_parquet()
s = df['Tair_f'].copy()
s = s.loc[s.index.year == 2018].copy()
s = s.loc[s.index.month == 7].copy()

# Add synthetic impulse noise
s_noise = dv.variables.add_impulse_noise(
    series=s,
    factor_low=-10,
    factor_high=3,
    contamination=0.04,
    seed=42
)
s_noise.name = f"{s.name}+noise"

print("Test data created:")
print(f"  Records: {len(s_noise)}")
print(f"  Valid: {s_noise.notna().sum()}")
print(f"  Range: {s_noise.min():.2f} to {s_noise.max():.2f}°C")

# %%
# Detect incremental outliers
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Calculates z-scores of three types of increments:
# - Forward: difference from previous value
# - Backward: difference to next value
# - Combined: sum of forward and backward
#
# Values flagged in all three increment types are removed.

zsi = dv.outliers.zScoreIncrements(
    series=s_noise,
    thres_zscore=3,
    showplot=False,
    verbose=1
)
zsi.calc(repeat=True)

flag = zsi.get_flag()
filtered = s_noise.copy()
filtered.loc[flag == 2] = None

print("\nIncrement detection results:")
print(f"  Outliers detected: {(flag == 2).sum()}")
print(f"  Valid records: {filtered.notna().sum()}")
print(f"  Data retained: {100*filtered.notna().sum()/s_noise.notna().sum():.1f}%")

print(f"\nStatistics:")
print(f"Original: min={s_noise.min():.2f}, max={s_noise.max():.2f}, mean={s_noise.mean():.2f}")
print(f"Filtered: min={filtered.min():.2f}, max={filtered.max():.2f}, mean={filtered.mean():.2f}")

# %%
# Why increments work for spike detection
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Spikes cause anomalously large changes between consecutive records.
# This approach is particularly effective for instrumental errors and
# sudden measurement failures.
