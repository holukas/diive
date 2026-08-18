"""
================================
Granger Causality Testing
================================

Test whether one time series helps predict another using Granger causality analysis.
Granger causality measures predictive causality, not true causation.

Best for: Investigating directional relationships in time series data
"""

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^
#
# We'll test whether vapor pressure deficit (VPD) helps predict NEE flux.

import diive as dv

df = dv.load_exampledata_parquet()
print(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")

# Resample to daily averages
df_daily = df.resample('D').mean()
df_daily = df_daily.dropna(how='all')

# Filter to first year of data for focused analysis
df_daily = df_daily.iloc[:365]
print(f"Resampled to {len(df_daily)} daily records")

# Use VPD as the potential cause
vpd = df_daily['VPD_f'].copy()
vpd.name = 'VPD'

# Use NEE flux as the effect variable
nee = df_daily['NEE_CUT_REF_f'].copy()
nee.name = 'NEE'

print(f"\nVPD: {len(vpd)} records, {vpd.isna().sum()} missing")
print(f"NEE: {len(nee)} records, {nee.isna().sum()} missing")

# %%
# Test Granger causality
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Does VPD Granger-cause NEE? We test up to 5-step time lags.

gc = dv.analysis.GrangerCausality(x=vpd, y=nee, max_lag=5, verbose=True)

print(f"\nData after alignment: {len(gc.data)} records")
print(gc.data.head())

# %%
# Extract and interpret results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Get p-values for each lag and identify significant relationships.

p_df = gc.p_values()
print("\nP-values by lag:")
print(p_df)

sig_lag = gc.significant_lag(alpha=0.05)
print(f"\nFirst significant lag (alpha=0.05): {sig_lag}")
if sig_lag:
    print(f"  VPD at lag {sig_lag} helps predict NEE")
else:
    print(f"  No significant lags detected")

# %%
# Generate formatted report
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Print a summary of Granger causality results.
#
# Interpretation: A p-value < 0.05 at a given lag means past values of VPD
# significantly improve predictions of current NEE beyond what NEE's own history provides.

gc.report(alpha=0.05)

# %%
# Test the reverse direction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Does NEE Granger-cause VPD? This is expected to be non-significant.

gc_reverse = dv.analysis.GrangerCausality(x=nee, y=vpd, max_lag=5, verbose=False)
gc_reverse.report(alpha=0.05)

# %%
# Compare with radiation
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Test if radiation better predicts NEE than VPD does.

radiation = df_daily['Rg_f'].copy()
radiation.name = 'Radiation'

gc_rad = dv.analysis.GrangerCausality(x=radiation, y=nee, max_lag=5, verbose=False)

print("\nComparison: VPD vs Radiation as NEE predictors")
print("-" * 60)
vpd_pvals = gc.p_values()
rad_pvals = gc_rad.p_values()
comparison = vpd_pvals.copy()
comparison['p_value_rad'] = rad_pvals['p_value']
comparison.columns = ['Lag', 'p_vpd', 'p_radiation']
print(comparison)

vpd_sig = gc.significant_lag(alpha=0.05)
rad_sig = gc_rad.significant_lag(alpha=0.05)
print(f"\nFirst significant lag:")
print(f"  VPD: {vpd_sig if vpd_sig else 'None'}")
print(f"  Radiation: {rad_sig if rad_sig else 'None'}")

# %%
# Visualize Granger causality results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create a 4-panel figure showing: time series, scatter, p-values, and correlation by lag.

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Time series comparison (full daily dataset)
ax1 = axes[0, 0]
ax1_vpd = ax1.twinx()

line1 = ax1.plot(vpd.index, vpd, color='#FF9800', linewidth=2, alpha=0.8, label='VPD')
line2 = ax1_vpd.plot(nee.index, nee, color='#2196F3', linewidth=2, alpha=0.8, label='NEE')

ax1.set_xlabel('Date', fontsize=11, fontweight=600)
ax1.set_ylabel('VPD (hPa)', color='#FF9800', fontsize=11, fontweight=600)
ax1_vpd.set_ylabel('NEE (µmol/m²/s)', color='#2196F3', fontsize=11, fontweight=600)
ax1.tick_params(axis='y', labelcolor='#FF9800')
ax1_vpd.tick_params(axis='y', labelcolor='#2196F3')
ax1.set_title('Panel 1: Time Series (One Year)', fontsize=12, fontweight=600, loc='left')
ax1.grid(True, alpha=0.2)

# Panel 2: Scatter plot (VPD lagged by 1 step vs NEE)
ax2 = axes[0, 1]
vpd_lagged = vpd.shift(1).dropna()
nee_aligned = nee.loc[vpd_lagged.index]
scatter = ax2.scatter(vpd_lagged, nee_aligned, alpha=0.3, s=10, color='#4CAF50')
z = np.polyfit(vpd_lagged, nee_aligned, 1)
p = np.poly1d(z)
x_line = np.linspace(vpd_lagged.min(), vpd_lagged.max(), 100)
ax2.plot(x_line, p(x_line), 'r--', linewidth=2,
         label=f'Linear fit (R²={np.corrcoef(vpd_lagged, nee_aligned)[0, 1] ** 2:.3f})')
ax2.set_xlabel('VPD (lagged 1 day, hPa)', fontsize=11, fontweight=600)
ax2.set_ylabel('NEE (µmol/m²/s)', fontsize=11, fontweight=600)
ax2.set_title('Panel 2: Scatter (VPD Lag-1 day vs NEE)', fontsize=12, fontweight=600, loc='left')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2)

# Panel 3: P-values across lags (bar chart with significance threshold)
ax3 = axes[1, 0]
lags = vpd_pvals['Lag'].values
p_vals = vpd_pvals['p_value'].values
# Cap at -log10(1e-100) for visualization, add text labels for actual values
p_vals_capped = np.maximum(p_vals, 1e-100)
log_p_vals = -np.log10(p_vals_capped)
log_p_vals_clipped = np.minimum(log_p_vals, 50)  # Cap display at 50 for readability
colors = ['#4CAF50' if p < 0.05 else '#CCCCCC' for p in p_vals]
bars = ax3.bar(lags, log_p_vals_clipped, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
# Add text labels showing p-value significance
for i, (lag, log_p) in enumerate(zip(lags, log_p_vals)):
    label = 'p<1e-100' if log_p > 50 else f'p<1e-{int(log_p)}'
    ax3.text(lag, min(log_p, 50) + 2, label, ha='center', fontsize=9, fontweight=600)
ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='Significance threshold (α=0.05)')
ax3.set_xlabel('Lag (days)', fontsize=11, fontweight=600)
ax3.set_ylabel('-log10(p-value)', fontsize=11, fontweight=600)
ax3.set_title('Panel 3: Granger Causality P-values (capped at 50)', fontsize=12, fontweight=600, loc='left')
ax3.set_xticks(lags)
ax3.set_ylim(0, 55)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.2, axis='y')

# Panel 4: Correlation by lag (shows predictive power decay)
ax4 = axes[1, 1]
correlations = []
for lag in range(0, 25):
    vpd_at_lag = vpd.shift(lag).dropna()
    nee_at_lag = nee.loc[vpd_at_lag.index]
    if len(vpd_at_lag) > 1:
        corr = np.corrcoef(vpd_at_lag, nee_at_lag)[0, 1]
        correlations.append(corr)
    else:
        correlations.append(np.nan)

lags_extended = np.arange(len(correlations))
ax4.plot(lags_extended, correlations, marker='o', linewidth=2.5, markersize=6, color='#9C27B0', alpha=0.8)
ax4.axvline(x=1, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Lag 1 (optimal)')
ax4.fill_between(lags_extended, 0, correlations, alpha=0.2, color='#9C27B0')
ax4.set_xlabel('Lag (days)', fontsize=11, fontweight=600)
ax4.set_ylabel('Correlation Coefficient', fontsize=11, fontweight=600)
ax4.set_title('Panel 4: Correlation Decay by Lag', fontsize=12, fontweight=600, loc='left')
ax4.set_xlim(0, 24)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
