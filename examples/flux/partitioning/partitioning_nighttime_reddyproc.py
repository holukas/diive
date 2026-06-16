"""
==============================================
Nighttime NEE Partitioning REddyProc (RECO/GPP)
==============================================

Partition net ecosystem exchange (NEE) into gross primary production (GPP) and
ecosystem respiration (RECO) with the nighttime method of Reichstein et al.
(2005), here as a faithful Python port of REddyProc's ``sMRFluxPartition``.

This is a *second*, independent port of the same paper alongside the ONEFlux
variant (see ``partitioning_nighttime.py``). REddyProc differs in the day/night
split (potential radiation, needs longitude + UTC offset), the E0 fitting
routine, the window geometry, and - crucially - it partitions the **whole
record at once** with a single temperature sensitivity E0. Output columns carry
the ``_RP`` token, so both variants can live in one dataframe.

Best for: matching a REddyProc partitioning, or cross-checking the ONEFlux one.
"""

# %%
# Why a second port?
# ^^^^^^^^^^^^^^^^^^
# The bundled CH-DAV dataset ships the REddyProc-produced reference columns
# ``Reco_CUT_REF`` / ``GPP_CUT_REF_f``. Because they are REddyProc-derived, the
# REddyProc port is expected to reproduce them closely (a genuine 1:1 target),
# unlike the ONEFlux port where they are only a loose sanity check.

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2017].copy()  # single year for the plots below

print(f"Period: {df.index.min().date()} to {df.index.max().date()} ({len(df)} records)")

# %%
# Run the partitioning
# ^^^^^^^^^^^^^^^^^^^^^
# Inputs:
#   - measured NEE and the gap-filled NEE (for the GPP residual)
#   - measured air temperature and gap-filled air temperature (for RECO)
#   - incoming shortwave radiation (for the day/night split)
#   - site latitude, longitude and UTC offset (for solar-time potential
#     radiation, which REddyProc uses to separate night from day)
#
# Note: REddyProc estimates one E0 for the entire record. A single year is shown
# here for clarity, but for a true match to the reference columns partition the
# full multi-year record at once.

part = dv.flux.NighttimePartitioningReddyProc(
    nee=df['NEE_CUT_REF_orig'],  # measured NEE (NaN where not measured)
    ta=df['Tair_orig'],  # measured air temperature
    sw_in=df['Rg_orig'],  # incoming shortwave radiation
    nee_f=df['NEE_CUT_REF_f'],  # gap-filled NEE
    ta_f=df['Tair_f'],  # gap-filled air temperature
    lat=46.815,  # CH-DAV latitude
    lon=9.855,  # CH-DAV longitude
    utc_offset=1,  # CET
)
part.run()
results = part.results

print("\nResult columns:", list(results.columns))
print(results[['RECO_NT_RP', 'GPP_NT_RP', 'E0_NT_RP']].describe())

# %%
# Compare to the REddyProc reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RECO and GPP should track the reference columns closely.

reco_ref = df['Reco_CUT_REF']
gpp_ref = df['GPP_CUT_REF_f']

m_reco = results['RECO_NT_RP'].notna() & reco_ref.notna()
m_gpp = results['GPP_NT_RP'].notna() & gpp_ref.notna()

print(f"RECO correlation, diive (REddyProc method) vs REddyProc reference: "
      f"{np.corrcoef(results['RECO_NT_RP'][m_reco], reco_ref[m_reco])[0, 1]:.4f}")
print(f"GPP  correlation, diive (REddyProc method) vs REddyProc reference: "
      f"{np.corrcoef(results['GPP_NT_RP'][m_gpp], gpp_ref[m_gpp])[0, 1]:.4f}")

print(f"\nMeans (umol m-2 s-1):")
print(f"  RECO  diive (REddyProc method): {results['RECO_NT_RP'].mean():.3f}   "
      f"REddyProc reference: {reco_ref.mean():.3f}")
print(f"  GPP   diive (REddyProc method): {results['GPP_NT_RP'].mean():.3f}   "
      f"REddyProc reference: {gpp_ref.mean():.3f}")

# %%
# Visual comparison
# ^^^^^^^^^^^^^^^^^
# Scatter the partitioned RECO and GPP against the reference columns, with the
# 1:1 line for orientation. Tight clustering along the 1:1 line confirms the
# close agreement seen in the correlations above.

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.27)

for ax, dcol, ref, name, color in [
    (axes[0], 'RECO_NT_RP', reco_ref, 'RECO', '#F44336'),  # red
    (axes[1], 'GPP_NT_RP', gpp_ref, 'GPP', '#2196F3'),  # blue
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

fig.suptitle('diive (REddyProc method) vs REddyProc reference (CH-DAV 2017)')
plt.show()


# %%
# Time-series comparison (shared axis)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Daily means of RECO and GPP over the year, diive (REddyProc method, solid) vs
# REddyProc reference (dashed), all on one shared axis so the seasonal cycles
# line up.

def _daily(s):
    return s.resample('D').mean()


fig, ax = plt.subplots(figsize=(13, 5))
fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12)

ax.plot(_daily(results['RECO_NT_RP']), color='#F44336', lw=1.4,
        label='RECO diive (REddyProc method)')
ax.plot(_daily(reco_ref), color='#F44336', lw=1.1, ls='--', label='RECO REddyProc reference')
ax.plot(_daily(results['GPP_NT_RP']), color='#2196F3', lw=1.4,
        label='GPP diive (REddyProc method)')
ax.plot(_daily(gpp_ref), color='#2196F3', lw=1.1, ls='--', label='GPP REddyProc reference')

ax.axhline(0, color='#455A64', lw=0.8)
ax.set_ylabel('flux (umol m-2 s-1)')
ax.set_xlabel('2017')
ax.set_title('diive (REddyProc method) vs REddyProc reference - daily means (CH-DAV 2017)')
ax.legend(ncol=2, frameon=False)
plt.show()
