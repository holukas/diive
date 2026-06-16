"""
=========================================
Daytime NEE Partitioning ONEFlux (RECO/GPP)
=========================================

Partition net ecosystem exchange (NEE) into gross primary production (GPP) and
ecosystem respiration (RECO) with the daytime method of Lasslop et al. (2010),
here as a faithful Python port of ONEFlux's ``flux_part_gl2010`` (the FLUXNET2015
daytime partitioning).

The daytime method fits a rectangular-hyperbola light-response curve (LRC) to
daytime NEE in short windows - GPP saturating with light, modulated by VPD - and
holds the temperature sensitivity ``E0`` fixed from a prior nighttime estimate.
RECO and GPP are then predicted for every record. Output columns carry the
``_DT_OF`` token (daytime, ONEFlux), so this port lives side by side with the
daytime REddyProc port (``_DT_RP``) and the nighttime ports (``_NT_OF`` /
``_NT_RP``) in one dataframe.

Best for: matching an ONEFlux / FLUXNET2015 daytime partitioning, or comparing
the ONEFlux and REddyProc daytime implementations.
"""

# %%
# Inputs and reference columns
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ONEFlux's daytime method uses BOTH the measured (gappy) and the gap-filled
# drivers: measured Rg/TA classify records (day if ``Rg > 4``) and feed the
# internal NEE-uncertainty look-up that weights the fits, while the gap-filled
# drivers feed the fits and the flux prediction. Unlike the nighttime method, it
# does NOT use solar geometry or latitude - the day/night split is the measured
# radiation threshold.
#
# The bundled CH-DAV dataset ships REddyProc daytime reference columns
# (``Reco_DT_CUT_REF`` / ``GPP_DT_CUT_REF``). They come from a different
# algorithm and provenance (measured NEE uncertainty, bootstrap, full record), so
# the ONEFlux port tracks GPP closely and RECO with a stable bias - a provenance
# artifact, not an algorithmic difference (the port matches a native ONEFlux run
# on identical inputs to RECO r ~ 0.999, GPP r ~ 0.9999).

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2017].copy()  # single year for the plots below

print("Partitioning method: DAYTIME (Lasslop et al. 2010, ONEFlux port) -> *_DT_OF")
print(f"Period: {df.index.min().date()} to {df.index.max().date()} ({len(df)} records)")

# %%
# Run the partitioning
# ^^^^^^^^^^^^^^^^^^^^^

part = dv.flux.DaytimePartitioningOneFlux(
    nee=df['NEE_CUT_REF_orig'],  # measured NEE (NaN where not measured)
    ta=df['Tair_orig'],  # measured air temperature
    sw_in=df['Rg_orig'],  # measured incoming shortwave radiation
    ta_f=df['Tair_f'],  # gap-filled air temperature
    sw_in_f=df['Rg_f'],  # gap-filled incoming shortwave radiation
    vpd=df['VPD_f'],  # gap-filled VPD (kPa, converted to hPa internally)
)
part.run()
results = part.results

print("\nResult columns:", list(results.columns))
print(results[['RECO_DT_OF', 'GPP_DT_OF']].describe())

# %%
# Compare to the bundled reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# GPP should track the reference column closely; daytime RECO is more sensitive
# and carries a bias (the reference is REddyProc-derived, see above).

reco_ref = df['Reco_DT_CUT_REF']
gpp_ref = df['GPP_DT_CUT_REF']

m_reco = results['RECO_DT_OF'].notna() & reco_ref.notna()
m_gpp = results['GPP_DT_OF'].notna() & gpp_ref.notna()

print(f"RECO correlation, diive (ONEFlux method) vs reference: "
      f"{np.corrcoef(results['RECO_DT_OF'][m_reco], reco_ref[m_reco])[0, 1]:.4f}")
print(f"GPP  correlation, diive (ONEFlux method) vs reference: "
      f"{np.corrcoef(results['GPP_DT_OF'][m_gpp], gpp_ref[m_gpp])[0, 1]:.4f}")

print(f"\nMeans (umol m-2 s-1):")
print(f"  RECO  diive (ONEFlux method): {results['RECO_DT_OF'].mean():.3f}   "
      f"reference: {reco_ref.mean():.3f}")
print(f"  GPP   diive (ONEFlux method): {results['GPP_DT_OF'].mean():.3f}   "
      f"reference: {gpp_ref.mean():.3f}")

# %%
# Visual comparison
# ^^^^^^^^^^^^^^^^^
# Scatter the partitioned RECO and GPP against the reference columns, with the
# 1:1 line for orientation.

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.27)

for ax, dcol, ref, name, color in [
    (axes[0], 'RECO_DT_OF', reco_ref, 'RECO', '#F44336'),  # red
    (axes[1], 'GPP_DT_OF', gpp_ref, 'GPP', '#2196F3'),  # blue
]:
    m = results[dcol].notna() & ref.notna()
    x, y = ref[m], results[dcol][m]
    ax.scatter(x, y, s=4, alpha=0.15, color=color, edgecolors='none')
    lim = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    ax.plot(lim, lim, color='#455A64', lw=1.0, ls='--', label='1:1')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'reference {name} (umol m-2 s-1)')
    ax.set_ylabel(f'diive (ONEFlux method) {name} (umol m-2 s-1)')
    ax.set_title(f'{name}   r = {np.corrcoef(x, y)[0, 1]:.3f}')
    ax.legend(loc='upper left', frameon=False)

fig.suptitle('diive (ONEFlux method) vs reference - daytime (CH-DAV 2017)')
plt.show()

# %%
# Time-series comparison (shared axis)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Daily means of RECO and GPP over the year, diive (ONEFlux method, solid) vs
# reference (dashed), all on one shared axis.


def _daily(s):
    return s.resample('D').mean()


fig, ax = plt.subplots(figsize=(13, 5))
fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12)

ax.plot(_daily(results['RECO_DT_OF']), color='#F44336', lw=1.4,
        label='RECO diive (ONEFlux method)')
ax.plot(_daily(reco_ref), color='#F44336', lw=1.1, ls='--', label='RECO reference')
ax.plot(_daily(results['GPP_DT_OF']), color='#2196F3', lw=1.4,
        label='GPP diive (ONEFlux method)')
ax.plot(_daily(gpp_ref), color='#2196F3', lw=1.1, ls='--', label='GPP reference')

ax.axhline(0, color='#455A64', lw=0.8)
ax.set_ylabel('flux (umol m-2 s-1)')
ax.set_xlabel('2017')
ax.set_title('diive (ONEFlux method) vs reference - daytime daily means (CH-DAV 2017)')
ax.legend(ncol=2, frameon=False)
plt.show()
