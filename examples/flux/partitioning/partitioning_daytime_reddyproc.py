"""
===========================================
Daytime NEE Partitioning REddyProc (RECO/GPP)
===========================================

Partition net ecosystem exchange (NEE) into gross primary production (GPP) and
ecosystem respiration (RECO) with the daytime method of Lasslop et al. (2010),
here as a faithful Python port of REddyProc's ``partitionNEEGL``.

The daytime method fits a rectangular-hyperbola light-response curve (LRC) to
daytime NEE in short windows - GPP saturating with light, modulated by VPD - and
holds the temperature sensitivity ``E0`` fixed from a prior nighttime estimate.
RECO and GPP are then predicted for every record. Output columns carry the
``_DT_RP`` token (daytime, REddyProc), so the daytime and nighttime
(``_NT_RP``) ports can live side by side in one dataframe.

Best for: matching a REddyProc daytime partitioning, or comparing the daytime
and nighttime approaches.
"""

# %%
# About the reference columns
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The bundled CH-DAV dataset ships REddyProc daytime reference columns
# ``Reco_DT_CUT_REF`` / ``GPP_DT_CUT_REF``. They were produced with the measured
# NEE uncertainty (which is not shipped) and on the full multi-year record, so
# the port is expected to track GPP closely and RECO with a small stable bias - a
# provenance artifact, not an algorithmic difference (the port matches a fresh
# REddyProc run on identical inputs to r > 0.999).

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2017].copy()  # single year for the plots below

print(f"Period: {df.index.min().date()} to {df.index.max().date()} ({len(df)} records)")

# %%
# Run the partitioning
# ^^^^^^^^^^^^^^^^^^^^^
# Inputs (REddyProc's daytime method uses the gap-filled meteo drivers
# throughout and quality-filters only NEE):
#   - measured NEE (gaps are records that were not measured / did not pass QC)
#   - gap-filled air temperature (for the fit and for RECO)
#   - gap-filled VPD (diive's kPa is converted to REddyProc's hPa internally)
#   - gap-filled incoming shortwave radiation (day/night split + LRC light driver)
#   - site latitude, longitude and UTC offset (solar-time day/night split)

part = dv.flux.DaytimePartitioningReddyProc(
    nee=df['NEE_CUT_REF_orig'],  # measured NEE (NaN where not measured)
    ta=df['Tair_f'],  # gap-filled air temperature
    vpd=df['VPD_f'],  # gap-filled VPD (kPa)
    sw_in=df['Rg_f'],  # gap-filled incoming shortwave radiation
    lat=46.815,  # CH-DAV latitude
    lon=9.855,  # CH-DAV longitude
    utc_offset=1,  # CET
)
part.run()
results = part.results

print("\nResult columns:", list(results.columns))
print(results[['RECO_DT_RP', 'GPP_DT_RP']].describe())

# %%
# Compare to the REddyProc reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# GPP should track the reference column closely; daytime RECO is more sensitive
# and carries a small bias (reference provenance, see above).

reco_ref = df['Reco_DT_CUT_REF']
gpp_ref = df['GPP_DT_CUT_REF']

m_reco = results['RECO_DT_RP'].notna() & reco_ref.notna()
m_gpp = results['GPP_DT_RP'].notna() & gpp_ref.notna()

print(f"RECO correlation, diive (REddyProc method) vs REddyProc reference: "
      f"{np.corrcoef(results['RECO_DT_RP'][m_reco], reco_ref[m_reco])[0, 1]:.4f}")
print(f"GPP  correlation, diive (REddyProc method) vs REddyProc reference: "
      f"{np.corrcoef(results['GPP_DT_RP'][m_gpp], gpp_ref[m_gpp])[0, 1]:.4f}")

print(f"\nMeans (umol m-2 s-1):")
print(f"  RECO  diive (REddyProc method): {results['RECO_DT_RP'].mean():.3f}   "
      f"REddyProc reference: {reco_ref.mean():.3f}")
print(f"  GPP   diive (REddyProc method): {results['GPP_DT_RP'].mean():.3f}   "
      f"REddyProc reference: {gpp_ref.mean():.3f}")

# %%
# Visual comparison
# ^^^^^^^^^^^^^^^^^
# Scatter the partitioned RECO and GPP against the reference columns, with the
# 1:1 line for orientation.

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.27)

for ax, dcol, ref, name, color in [
    (axes[0], 'RECO_DT_RP', reco_ref, 'RECO', '#F44336'),  # red
    (axes[1], 'GPP_DT_RP', gpp_ref, 'GPP', '#2196F3'),  # blue
]:
    m = results[dcol].notna() & ref.notna()
    x, y = ref[m], results[dcol][m]
    ax.scatter(x, y, s=4, alpha=0.15, color=color, edgecolors='none')
    lim = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    ax.plot(lim, lim, color='#455A64', lw=1.0, ls='--', label='1:1')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'REddyProc reference {name} (umol m-2 s-1)')
    ax.set_ylabel(f'diive (REddyProc method) {name} (umol m-2 s-1)')
    ax.set_title(f'{name}   r = {np.corrcoef(x, y)[0, 1]:.3f}')
    ax.legend(loc='upper left', frameon=False)

fig.suptitle('diive (REddyProc method) vs REddyProc reference - daytime (CH-DAV 2017)')
plt.show()

# %%
# Time-series comparison (shared axis)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Daily means of RECO and GPP over the year, diive (REddyProc method, solid) vs
# REddyProc reference (dashed), all on one shared axis.


def _daily(s):
    return s.resample('D').mean()


fig, ax = plt.subplots(figsize=(13, 5))
fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12)

ax.plot(_daily(results['RECO_DT_RP']), color='#F44336', lw=1.4,
        label='RECO diive (REddyProc method)')
ax.plot(_daily(reco_ref), color='#F44336', lw=1.1, ls='--', label='RECO REddyProc reference')
ax.plot(_daily(results['GPP_DT_RP']), color='#2196F3', lw=1.4,
        label='GPP diive (REddyProc method)')
ax.plot(_daily(gpp_ref), color='#2196F3', lw=1.1, ls='--', label='GPP REddyProc reference')

ax.axhline(0, color='#455A64', lw=0.8)
ax.set_ylabel('flux (umol m-2 s-1)')
ax.set_xlabel('2017')
ax.set_title('diive (REddyProc method) vs REddyProc reference - daytime daily means (CH-DAV 2017)')
ax.legend(ncol=2, frameon=False)
plt.show()
