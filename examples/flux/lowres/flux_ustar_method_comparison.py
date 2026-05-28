"""
==========================================================
Comparing USTAR Threshold Detection Methods
==========================================================

Side-by-side comparison of the ONEFlux moving point method (Papale et al., 2006)
and the Vekuri quantile-based approach.

Both use the same 3-year window bootstrap via UstarBootstrapThresholds.
Results differ due to stratification strategy (equal-sized vs. quantile bins),
correlation threshold (0.5 vs. 0.4), and first-USTAR class validation.
"""

# %%
# Import and Load Data
# ^^^^^^^^^^^^^^^^^^^^
# Set up data for both methods.

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diive as dv

warnings.filterwarnings('ignore')

data = dv.load_exampledata_parquet_lae()

print(f"Data: {len(data)} records, {data.index.min().date()} to {data.index.max().date()}")

# %%
# Run Both Detection Methods (seasonal)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Execute both methods to get seasonal thresholds.

print("=" * 70)
print("Running ONEFlux Moving Point Detection (Papale et al. 2006)")
print("=" * 70)

detector_mp = dv.flux.UstarMovingPointDetection(data, verbose=1)
results_mp = detector_mp.detect()

print("\n" + "=" * 70)
print("Running Vekuri Quantile-Based Detection")
print("=" * 70)

detector_vekuri = dv.flux.UstarVekuriThresholdDetection(data, verbose=1)
results_vekuri = detector_vekuri.detect()

# %%
# Compare Seasonal Thresholds
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Display threshold estimates for each season from both methods.

comparison_seasonal = pd.DataFrame({
    'ONEFlux': results_mp['threshold'],
    'Vekuri': results_vekuri['threshold'],
})

print("\nSeasonal Threshold Comparison (m/s):")
print(comparison_seasonal.round(4))

# %%
# Multi-Year Bootstrap: Both Methods
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Run multi-year bootstrap for both methods using UstarBootstrapThresholds.
# This yields per-year p16/p50/p84 thresholds and a pooled CUT threshold.

print("\n" + "=" * 70)
print("Multi-Year Bootstrap: ONEFlux")
print("=" * 70)

boot_mp = dv.flux.UstarBootstrapThresholds(
    df=data,
    detector_class=dv.flux.UstarMovingPointDetection,
    detector_kwargs=dict(ta_classes_count=7, ustar_classes_count=20),
    n_iter=100,
    percentiles=(16, 50, 84),
    n_jobs=-1,
    verbose=1,
)
annual_boot_mp = boot_mp.run()
cut_mp = boot_mp.get_cut_threshold()

print("\n" + "=" * 70)
print("Multi-Year Bootstrap: Vekuri")
print("=" * 70)

boot_vekuri = dv.flux.UstarBootstrapThresholds(
    df=data,
    detector_class=dv.flux.UstarVekuriThresholdDetection,
    n_iter=100,
    percentiles=(16, 50, 84),
    n_jobs=-1,
    verbose=1,
)
annual_boot_vekuri = boot_vekuri.run()
cut_vekuri = boot_vekuri.get_cut_threshold()

# %%
# Compare Annual Bootstrap Results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Side-by-side per-year p50 thresholds from both methods.

print("\n" + "=" * 70)
print("Per-Year Bootstrap Thresholds (p50, m/s)")
print("=" * 70)

comparison_annual = pd.DataFrame({
    'ONEFlux_p50': annual_boot_mp['p50'],
    'Vekuri_p50': annual_boot_vekuri['p50'],
})
comparison_annual['diff_pct'] = (
        (comparison_annual['Vekuri_p50'] - comparison_annual['ONEFlux_p50'])
        / comparison_annual['ONEFlux_p50'] * 100
)

print(comparison_annual.round(4))

# %%
# Compare CUT Thresholds
# ^^^^^^^^^^^^^^^^^^^^^^^
# The CUT (constant) threshold pools all years' bootstrap samples.

print("\n" + "=" * 70)
print("CUT Threshold Comparison (pooled across all years)")
print("=" * 70)

print(f"\n{'Percentile':<12} {'ONEFlux':>12} {'Vekuri':>12}")
print("-" * 36)
for pct in ('p16', 'p50', 'p84'):
    mp_val = cut_mp[pct]
    vek_val = cut_vekuri[pct]
    marker = "  <-- recommended" if pct == 'p50' else ""
    print(f"{pct:<12} {mp_val:>12.4f} {vek_val:>12.4f}{marker}")

# %%
# Visualize: Per-Year Bootstrap Comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Bar chart comparing per-year p50 thresholds, with p16/p84 as error bars.

years = annual_boot_mp.index.tolist()
x = np.arange(len(years))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Per-year p50 with p16/p84 error bars
ax = axes[0]
mp_p50 = annual_boot_mp['p50'].values
mp_err_lo = mp_p50 - annual_boot_mp['p16'].values
mp_err_hi = annual_boot_mp['p84'].values - mp_p50

vek_p50 = annual_boot_vekuri['p50'].values
vek_err_lo = vek_p50 - annual_boot_vekuri['p16'].values
vek_err_hi = annual_boot_vekuri['p84'].values - vek_p50

ax.bar(x - width / 2, mp_p50, width, label='ONEFlux p50', alpha=0.8,
       yerr=[mp_err_lo, mp_err_hi], capsize=4, edgecolor='black')
ax.bar(x + width / 2, vek_p50, width, label='Vekuri p50', alpha=0.8,
       yerr=[vek_err_lo, vek_err_hi], capsize=4, edgecolor='black')

ax.set_xlabel('Year')
ax.set_ylabel('USTAR Threshold (m/s)')
ax.set_title('Per-Year Bootstrap p50 (error bars: p16/p84)')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: CUT threshold comparison
ax = axes[1]
methods = ['ONEFlux', 'Vekuri']
p50_vals = [cut_mp['p50'], cut_vekuri['p50']]
p16_vals = [cut_mp['p16'], cut_vekuri['p16']]
p84_vals = [cut_mp['p84'], cut_vekuri['p84']]
colors = ['#1f77b4', '#ff7f0e']

for i, (method, p50, p16, p84, color) in enumerate(zip(methods, p50_vals, p16_vals, p84_vals, colors)):
    ax.bar(i, p50, color=color, alpha=0.8, edgecolor='black',
           yerr=[[p50 - p16], [p84 - p50]], capsize=6)
    if not np.isnan(p50):
        ax.text(i, p50 + 0.005, f'{p50:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks([0, 1])
ax.set_xticklabels(methods)
ax.set_ylabel('USTAR Threshold (m/s)')
ax.set_title('CUT Threshold Comparison\n(error bars: p16/p84, center: p50)')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()  # Uncomment to display

# %%
# Method Comparison Summary
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Detailed comparison of algorithm characteristics.

print("\n" + "=" * 70)
print("Algorithm Characteristics Comparison")
print("=" * 70)

comparison_table = pd.DataFrame({
    'Characteristic': [
        'Temperature Classes',
        'USTAR Classes',
        'Detection Mode',
        'Stratification',
        'TA-USTAR Correlation',
        'First USTAR Validation',
        'Window Size',
        'Aggregation',
        'Computational Speed',
        'Parameterization',
    ],
    'ONEFlux (Papale 2006)': [
        '7 (equal-sized)',
        '20 (equal-sized)',
        'Forward (ascending)',
        'Equal-sized bins',
        '<= 0.5',
        'Yes (mean < 0.2)',
        '10',
        'Median across TA',
        'Standard',
        'More parameters',
    ],
    'Vekuri (Quantile)': [
        '6 (quantiles)',
        '20 (quantiles)',
        'Forward (ascending)',
        'Quantile-based',
        '< 0.4',
        'No',
        '10',
        'Median across TA',
        'Faster',
        'Simpler',
    ],
})

print(comparison_table.to_string(index=False))

# %%
# Key Algorithmic Differences
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print("\n" + "=" * 70)
print("Key Algorithmic Differences")
print("=" * 70)

differences = """
1. Stratification Strategy
   - ONEFlux: Equal-sized bins (fixed n per bin)
   - Vekuri: Quantile-based bins (equal sample size via pd.qcut)
   -> Vekuri handles skewed distributions better

2. Temperature Class Validation
   - ONEFlux: Includes first USTAR class validity check (mean < 0.2 m/s)
   - Vekuri: No explicit first USTAR validation
   -> ONEFlux more conservative, may skip temperature classes with low USTAR range

3. Correlation Threshold
   - ONEFlux: |TA-USTAR| <= 0.5
   - Vekuri: |TA-USTAR| < 0.4
   -> Vekuri rejects more temperature classes at high correlation

4. Data Requirements
   - ONEFlux: More stringent (3000 total, 160/season, 100/TA)
   - Vekuri: More flexible (works with fewer records)
   -> Vekuri better for sparse or smaller datasets

5. Bootstrap Design
   - Both use UstarBootstrapThresholds for multi-year per-year bootstrap
   - CUT threshold pools all years' samples for a single constant value
   -> Architecture is identical; only the detection algorithm differs
"""

print(differences)
