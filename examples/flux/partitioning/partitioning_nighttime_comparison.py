"""
==========================================================
Nighttime NEE Partitioning: ONEFLUX vs REddyProc methods
==========================================================

diive ships two independent Python ports of the Reichstein et al. (2005)
nighttime partitioning method, referred to throughout as the **diive (ONEFLUX
method)** and **diive (REddyProc method)** implementations:
:class:`~diive.flux.NighttimePartitioningOneFlux` (``_NT_OF`` columns) ports the
ONEFlux reference, :class:`~diive.flux.NighttimePartitioningReddyProc`
(``_NT_RP`` columns) ports REddyProc's ``sMRFluxPartition``. They implement the
*same* paper but differ in the details, so they do not produce identical
numbers. This example runs both on the same input and compares them against each
other and against the (REddyProc-derived) reference columns.

The two differ in:

- **Window geometry** - ONEFlux and REddyProc use different short-term window
  widths and steps for the E0 and Rref regressions.
- **Day/night split** - REddyProc combines incoming shortwave with potential
  radiation (so it needs longitude + UTC offset); ONEFlux uses a radiation
  threshold with latitude only.
- **E0 estimation** - REddyProc fits one temperature sensitivity E0 for the
  **whole record**; ONEFlux estimates E0 **per calendar year**.
- **Robust variant** - ONEFlux emits an outlier-robust ``*_ROB`` RECO/GPP;
  REddyProc has no robust variant.

Best for: choosing between the two ports, or understanding how much the choice
of reference implementation matters.
"""

# %%
# Load data
# ^^^^^^^^^
# The bundled CH-DAV dataset carries a ReddyProc-derived reference partitioning
# (``Reco_CUT_REF`` / ``GPP_CUT_REF_f``), so we can place both diive ports next
# to that common reference.

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2017].copy()  # single year keeps the example quick

print(f"Period: {df.index.min().date()} to {df.index.max().date()} ({len(df)} records)")

# %%
# Run both ports on identical inputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Both take the same measured/gap-filled NEE and air temperature plus incoming
# shortwave radiation. The REddyProc port additionally needs longitude and the
# UTC offset for its solar-time day/night split.

shared = dict(
    nee=df['NEE_CUT_REF_orig'],  # measured NEE (NaN where not measured)
    ta=df['Tair_orig'],  # measured air temperature
    sw_in=df['Rg_orig'],  # incoming shortwave radiation
    nee_f=df['NEE_CUT_REF_f'],  # gap-filled NEE
    ta_f=df['Tair_f'],  # gap-filled air temperature
)

of = dv.flux.NighttimePartitioningOneFlux(
    lat=46.815, **shared).run().results

rp = dv.flux.NighttimePartitioningReddyProc(
    lat=46.815, lon=9.855, utc_offset=1, **shared).run().results

reco_ref = df['Reco_CUT_REF']
gpp_ref = df['GPP_CUT_REF_f']

# %%
# Numerical comparison
# ^^^^^^^^^^^^^^^^^^^^^
# Correlate the two ports against each other and against the common reference,
# and compare their means. REddyProc reproduces the reference closely (it is a
# port of the implementation that produced it); ONEFlux is a looser match.


def _r(a, b):
    m = a.notna() & b.notna()
    return np.corrcoef(a[m], b[m])[0, 1]


print("\nRECO correlations:")
print(f"  diive (ONEFLUX method) vs diive (REddyProc method): {_r(of['RECO_NT_OF'], rp['RECO_NT_RP']):.4f}")
print(f"  diive (ONEFLUX method) vs REddyProc reference     : {_r(of['RECO_NT_OF'], reco_ref):.4f}")
print(f"  diive (REddyProc method) vs REddyProc reference   : {_r(rp['RECO_NT_RP'], reco_ref):.4f}")

print("\nGPP correlations:")
print(f"  diive (ONEFLUX method) vs diive (REddyProc method): {_r(of['GPP_NT_OF'], rp['GPP_NT_RP']):.4f}")
print(f"  diive (ONEFLUX method) vs REddyProc reference     : {_r(of['GPP_NT_OF'], gpp_ref):.4f}")
print(f"  diive (REddyProc method) vs REddyProc reference   : {_r(rp['GPP_NT_RP'], gpp_ref):.4f}")

print("\nMeans (umol m-2 s-1):")
print(f"  RECO  diive (ONEFLUX method) {of['RECO_NT_OF'].mean():.3f}   "
      f"diive (REddyProc method) {rp['RECO_NT_RP'].mean():.3f}   REddyProc reference {reco_ref.mean():.3f}")
print(f"  GPP   diive (ONEFLUX method) {of['GPP_NT_OF'].mean():.3f}   "
      f"diive (REddyProc method) {rp['GPP_NT_RP'].mean():.3f}   REddyProc reference {gpp_ref.mean():.3f}")

# %%
# Scatter: the two ports against each other
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RECO and GPP from the two implementations, plotted against each other with the
# 1:1 line. Spread off the 1:1 line is the disagreement between the two ports.

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.27)

for ax, ofcol, rpcol, name, color in [
    (axes[0], 'RECO_NT_OF', 'RECO_NT_RP', 'RECO', '#F44336'),  # red
    (axes[1], 'GPP_NT_OF', 'GPP_NT_RP', 'GPP', '#2196F3'),  # blue
]:
    m = of[ofcol].notna() & rp[rpcol].notna()
    x, y = rp[rpcol][m], of[ofcol][m]
    ax.scatter(x, y, s=4, alpha=0.15, color=color, edgecolors='none')
    lim = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    ax.plot(lim, lim, color='#455A64', lw=1.0, ls='--', label='1:1')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'diive (REddyProc method) {name} (umol m-2 s-1)')
    ax.set_ylabel(f'diive (ONEFLUX method) {name} (umol m-2 s-1)')
    ax.set_title(f'{name}   r = {np.corrcoef(x, y)[0, 1]:.3f}')
    ax.legend(loc='upper left', frameon=False)

fig.suptitle('diive (ONEFLUX method) vs diive (REddyProc method) (CH-DAV 2017)')
plt.show()


# %%
# Time-series comparison (daily means + cumulative)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Three panels on a shared x-axis: RECO daily means (top), GPP daily means
# (middle), and the running cumulative of both RECO and GPP (bottom). Within
# each panel the diive ONEFLUX method (solid), the diive REddyProc method
# (dashed) and the REddyProc reference (dotted) line up directly; the cumulative
# panel shows how the per-method differences accumulate into the annual budget.

def _daily(s):
    return s.resample('D').mean()


def _cumulative(s):
    return s.fillna(0.0).cumsum()


fig, (ax_reco, ax_gpp, ax_cum) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.06, hspace=0.13)

ax_reco.plot(_daily(of['RECO_NT_OF']), color='#F44336', lw=1.4, label='diive (ONEFLUX method)')
ax_reco.plot(_daily(rp['RECO_NT_RP']), color='#F44336', lw=1.1, ls='--', label='diive (REddyProc method)')
ax_reco.plot(_daily(reco_ref), color='#F44336', lw=0.9, ls=':', label='REddyProc reference')
ax_reco.set_ylabel('RECO (umol m-2 s-1)')
ax_reco.legend(ncol=3, frameon=False)

ax_gpp.plot(_daily(of['GPP_NT_OF']), color='#2196F3', lw=1.4, label='diive (ONEFLUX method)')
ax_gpp.plot(_daily(rp['GPP_NT_RP']), color='#2196F3', lw=1.1, ls='--', label='diive (REddyProc method)')
ax_gpp.plot(_daily(gpp_ref), color='#2196F3', lw=0.9, ls=':', label='REddyProc reference')
ax_gpp.set_ylabel('GPP (umol m-2 s-1)')
ax_gpp.legend(ncol=3, frameon=False)

# Cumulative: RECO (red) and GPP (blue), one line style per method.
for s, color, ls, label in [
    (of['RECO_NT_OF'], '#F44336', '-', 'RECO diive (ONEFLUX method)'),
    (rp['RECO_NT_RP'], '#F44336', '--', 'RECO diive (REddyProc method)'),
    (reco_ref, '#F44336', ':', 'RECO REddyProc reference'),
    (of['GPP_NT_OF'], '#2196F3', '-', 'GPP diive (ONEFLUX method)'),
    (rp['GPP_NT_RP'], '#2196F3', '--', 'GPP diive (REddyProc method)'),
    (gpp_ref, '#2196F3', ':', 'GPP REddyProc reference'),
]:
    ax_cum.plot(_cumulative(s), color=color, ls=ls, lw=1.3, label=label)
ax_cum.set_ylabel('cumulative flux (umol m-2 s-1)')
ax_cum.set_xlabel('2017')
ax_cum.legend(ncol=2, frameon=False, fontsize=8)

for ax in (ax_reco, ax_gpp):
    ax.axhline(0, color='#455A64', lw=0.8)

fig.suptitle('diive nighttime partitioning: ONEFLUX vs REddyProc methods - '
             'daily means and cumulative (CH-DAV 2017)')
plt.show()
