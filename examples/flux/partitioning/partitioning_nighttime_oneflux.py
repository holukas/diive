"""
=============================================
Nighttime NEE Partitioning ONEFlux (RECO/GPP)
=============================================

Partition net ecosystem exchange (NEE) into gross primary production (GPP) and
ecosystem respiration (RECO) with the nighttime method of Reichstein et al.
(2005), a faithful Python port of the ONEFlux reference implementation.

The nighttime method exploits the fact that there is no photosynthesis at
night, so nighttime NEE is pure respiration. A temperature-response model
(Lloyd & Taylor 1994) is fitted to nighttime NEE and extrapolated to daytime
temperatures to recover daytime respiration; GPP is then the difference
``GPP = RECO - NEE``.

Best for: deriving GPP and RECO from gap-filled, USTAR-filtered NEE.
"""

# %%
# Why partition NEE?
# ^^^^^^^^^^^^^^^^^^
# Eddy covariance measures the *net* CO2 flux (NEE), which bundles together two
# opposing gross fluxes: uptake by photosynthesis (GPP) and release by
# respiration (RECO). Ecologists usually want the two components separately.
# The nighttime method estimates RECO from the night and reconstructs GPP.

# %%
# Load data
# ^^^^^^^^^
# The bundled CH-DAV dataset already contains an independent reference
# partitioning (``Reco_CUT_REF`` / ``GPP_CUT_REF_f``, ReddyProc-derived), which
# lets us sanity-check the result.

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2017].copy()  # single year keeps the example quick

print(f"Period: {df.index.min().date()} to {df.index.max().date()} ({len(df)} records)")

# %%
# Run the partitioning
# ^^^^^^^^^^^^^^^^^^^^^
# Inputs:
#   - measured NEE and the gap-filled NEE (for the GPP residual)
#   - measured air temperature and gap-filled air temperature (for RECO)
#   - incoming shortwave radiation (for the day/night split)
#   - site latitude

part = dv.flux.NighttimePartitioningOneFlux(
    nee=df['NEE_CUT_REF_orig'],  # measured NEE (NaN where not measured)
    ta=df['Tair_orig'],  # measured air temperature
    sw_in=df['Rg_orig'],  # incoming shortwave radiation
    nee_f=df['NEE_CUT_REF_f'],  # gap-filled NEE
    ta_f=df['Tair_f'],  # gap-filled air temperature
    lat=46.815,  # CH-DAV latitude
)
part.run()
results = part.results

print("\nResult columns:", list(results.columns))
print(results[['RECO_NT_OF', 'GPP_NT_OF', 'E0_NT_OF']].describe())

# %%
# Compare to an independent reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RECO and GPP should track the reference columns (ReddyProc-derived) closely.
# Exact agreement is not expected: ReddyProc and ONEFlux differ in details such
# as the E0 quality gating, so this is a sanity check, not a parity target.

reco_ref = df['Reco_CUT_REF']
gpp_ref = df['GPP_CUT_REF_f']

m_reco = results['RECO_NT_OF'].notna() & reco_ref.notna()
m_gpp = results['GPP_NT_OF'].notna() & gpp_ref.notna()

print(f"RECO correlation vs ReddyProc: "
      f"{np.corrcoef(results['RECO_NT_OF'][m_reco], reco_ref[m_reco])[0, 1]:.4f}")
print(f"GPP  correlation vs ReddyProc: "
      f"{np.corrcoef(results['GPP_NT_OF'][m_gpp], gpp_ref[m_gpp])[0, 1]:.4f}")

print(f"\nAnnual sums (umol m-2 s-1, mean):")
print(f"  RECO_NT_OF mean: {results['RECO_NT_OF'].mean():.3f}   "
      f"ReddyProc Reco mean: {reco_ref.mean():.3f}")
print(f"  GPP_NT_OF  mean: {results['GPP_NT_OF'].mean():.3f}   "
      f"ReddyProc GPP  mean: {gpp_ref.mean():.3f}")

# %%
# Outlier-robust variants
# ^^^^^^^^^^^^^^^^^^^^^^^^
# ``RECO_NT_OF_ROB`` / ``GPP_NT_OF_ROB`` use a trimmed Rref estimate that downweights
# the largest nighttime deviations, useful when nighttime NEE is noisy.

print(results[['RECO_NT_OF', 'RECO_NT_OF_ROB', 'GPP_NT_OF', 'GPP_NT_OF_ROB']].mean())

# %%
# Visual comparison
# ^^^^^^^^^^^^^^^^^
# Scatter the partitioned RECO and GPP against the reference columns, with the
# 1:1 line for orientation. Tight clustering along the 1:1 line confirms the
# close agreement already seen in the correlations above.

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.27)

for ax, dcol, ref, name, color in [
    (axes[0], 'RECO_NT_OF', reco_ref, 'RECO', '#F44336'),  # red
    (axes[1], 'GPP_NT_OF', gpp_ref, 'GPP', '#2196F3'),  # blue
]:
    m = results[dcol].notna() & ref.notna()
    x, y = ref[m], results[dcol][m]
    ax.scatter(x, y, s=4, alpha=0.15, color=color, edgecolors='none')
    lim = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    ax.plot(lim, lim, color='#455A64', lw=1.0, ls='--', label='1:1')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'ReddyProc {name} (umol m-2 s-1)')
    ax.set_ylabel(f'diive {dcol} (umol m-2 s-1)')
    ax.set_title(f'{name}   r = {np.corrcoef(x, y)[0, 1]:.3f}')
    ax.legend(loc='upper left', frameon=False)

fig.suptitle('Nighttime partitioning ONEFlux vs ReddyProc (CH-DAV 2017)')
plt.show()


# %%
# Time-series comparison (shared axis)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Daily means of RECO and GPP over the year, diive (solid) vs ReddyProc
# (dashed), all on one shared axis so the seasonal cycles line up directly.

def _daily(s):
    return s.resample('D').mean()


fig, ax = plt.subplots(figsize=(13, 5))
fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12)

ax.plot(_daily(results['RECO_NT_OF']), color='#F44336', lw=1.4,
        label='RECO diive (ONEFlux)')
ax.plot(_daily(reco_ref), color='#F44336', lw=1.1, ls='--', label='RECO ReddyProc')
ax.plot(_daily(results['GPP_NT_OF']), color='#2196F3', lw=1.4,
        label='GPP diive (ONEFlux)')
ax.plot(_daily(gpp_ref), color='#2196F3', lw=1.1, ls='--', label='GPP ReddyProc')

ax.axhline(0, color='#455A64', lw=0.8)
ax.set_ylabel('flux (umol m-2 s-1)')
ax.set_xlabel('2017')
ax.set_title('Nighttime partitioning ONEFlux vs ReddyProc - daily means (CH-DAV 2017)')
ax.legend(ncol=2, frameon=False)
plt.show()
